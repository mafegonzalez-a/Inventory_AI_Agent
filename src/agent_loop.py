
import json
import re
from typing import Callable, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import PreTrainedTokenizerBase

def build_reasoning_prompt(question: str) -> str:
    """Prepara las instrucciones que recibe el modelo para consultar el inventario."""

    prompt = f"""
You are an assistant for an electronics store.

You have one tool available: get_information(query).

The query must be a JSON object. It can contain the following keys:
- filters
- select
- group_by
- operation
- sort_by
- limit

Available columns:
product_id, name, category, price, stock_quantity, rating.

Available operators:
==, !=, >, <, >=, <=.

Available operations:
sum, mean, count, max, min.

You must answer with only the JSON query, on a single line.

Example 1:
Question: What is the price of the 'Laptop Pro X'?
Query: {{"filters": [{{"column": "name", "operator": "==", "value": "Laptop Pro X"}}], "select": ["name", "price"]}}

Example 2:
Question: Which products belong to the category 'Accesorios'?
Query: {{"filters": [{{"column": "category", "operator": "==", "value": "Accesorios"}}], "select": ["name"]}}

Example 3:
Question: What is the average price of all products?
Query: {{"select": ["price"], "operation": "mean"}}

Now answer this question:
Question: {question}
Query:
"""

    return prompt

def generate_text(
    model: ORTModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 80,
) -> str:
    """Genera texto utilizando el modelo ONNX.

    Primero, el tokenizer convierte el texto en números que el modelo
    puede procesar. Luego, el modelo genera una respuesta y el tokenizer
    convierte el resultado nuevamente en texto.

    Args:
        model (ORTModelForCausalLM): Modelo ONNX previamente cargado.
        tokenizer (PreTrainedTokenizerBase): Tokenizer asociado al modelo.
        prompt (str): Instrucciones o pregunta que recibe el modelo.
        max_new_tokens (int): Cantidad máxima de tokens nuevos que puede
            generar el modelo. Por defecto es 80.

    Returns:
        str: Texto generado por el modelo, sin incluir el prompt original.
    """
    # Se convierte el texto en números para que el modelo pueda procesarlo
    inputs = tokenizer(prompt, return_tensors="pt")

    # Se desactiva la generación aleatoria para obtener resultados más consistentes
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Se separa el texto generado del prompt original
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_decision(decision_text: str) -> Optional[dict]:
    """Convierte la respuesta del modelo en una consulta que Python pueda utilizar.

    Busca el contenido JSON generado por el modelo y lo convierte
    en un diccionario para enviarlo a la función que consulta el inventario.

    Args:
        decision_text (str): Texto generado por el modelo.

    Returns:
        Optional[dict]: Consulta en formato diccionario. Devuelve None
            si no encuentra un JSON válido.
    """
    # Se toma la primera línea para evitar texto adicional del modelo
    first_line = decision_text.strip().split("\n")[0]

    # Se busca el contenido que se encuentra entre las llaves
    match = re.search(r"\{.*\}", first_line)
    if match is None:
        return None

    # Se intenta convertir el texto encontrado en un diccionario
    try:
        query = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # Se verifica que el resultado sea un diccionario
    if not isinstance(query, dict):
        return None
    return query


def run_agent_loop(
    question: str,
    model: ORTModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    tool_function: Callable[[dict], list[dict]],
) -> str:
    """Ejecuta el proceso del agente desde la pregunta hasta la respuesta final.

    La función conecta el modelo ONNX con la herramienta que consulta
    el inventario. Primero obtiene los parámetros de búsqueda, después
    consulta los productos y finalmente genera una respuesta para el usuario.

    Args:
        question (str): Pregunta realizada por el usuario.
        model (ORTModelForCausalLM): Modelo ONNX previamente cargado.
        tokenizer (PreTrainedTokenizerBase): Tokenizer asociado al modelo.
        tool_function (Callable[[dict], list[dict]]): Función que consulta
            el inventario y devuelve los resultados en una lista de diccionarios.

    Returns:
        str: Respuesta generada para el usuario o un mensaje de error.
    """

    # Paso 1: preparar las instrucciones para el modelo
    reasoning_prompt = build_reasoning_prompt(question)

    # Paso 2: generar los parámetros de búsqueda
    decision = generate_text(model, tokenizer, reasoning_prompt)
    print(f"[Decisión del modelo]: {decision.strip()}")

    # Paso 3: convertir la respuesta del modelo en un diccionario
    query = parse_decision(decision)

    if query is None:
        return "Lo siento, no pude entender cómo consultar esa información."

    # Paso 4: consultar el inventario utilizando los parámetros generados
    try:
        rows = tool_function(query)

    except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
        print(f"[Error de la herramienta]: {error}")
        return "Lo siento, no pude consultar el inventario con esa solicitud."

    # Convertir los resultados a JSON para enviarlos al modelo
    tool_result = json.dumps(rows, ensure_ascii=False)
    print(f"[Resultado de la herramienta]: {tool_result}")

    # Si no hay resultados, informar al usuario
    if not rows:
        return "No encontré productos que coincidan con tu pregunta."

    # Paso 5: preparar las instrucciones para generar la respuesta final
    final_prompt = f"""
Answer the question using only the tool result.

Question: {question}

Tool result: {tool_result}

Answer:
"""

    final_answer = generate_text(
        model,
        tokenizer,
        final_prompt,
        max_new_tokens=60
    )

    # Conservar únicamente la primera línea de la respuesta
    final_answer = final_answer.strip().split("\n")[0]

    if final_answer:
        return final_answer

    return tool_result

