import json
import re
from typing import Callable, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import PreTrainedTokenizerBase

def get_example_bank() -> list[tuple[str, dict, set[str]]]:
    """Devuelve el banco de ejemplos que se usan para guiar al modelo.

    Cada ejemplo contiene una pregunta, la consulta correcta para esa pregunta
    y un conjunto de palabras clave que permiten reconocer qué tipo de
    pregunta representa. Los productos usados existen en products.csv.

    Returns:
        list[tuple[str, dict, set[str]]]: Lista de ejemplos, cada uno con la
            pregunta, la consulta en formato diccionario y sus palabras clave.
    """
    return [
        (
            "What is the average price of all products?",
            {"select": ["price"], "operation": "mean"},
            {"average", "promedio", "mean", "all", "todos"},
        ),
        (
            "What is the price of the 'Laptop Pro X15'?",
            {"filters": [{"column": "name", "operator": "==", "value": "Laptop Pro X15"}],
             "select": ["name", "price"]},
            {"price", "precio", "cuesta", "cuánto", "cuanto", "cost", "much"},
        ),
        (
            "¿Cuántos 'Tablet S11' hay en stock?",
            {"filters": [{"column": "name", "operator": "==", "value": "Tablet S11"}],
             "select": ["name", "stock_quantity"]},
            {"stock", "hay", "cuántos", "cuantos", "many", "quantity", "unidades"},
        ),
        (
            "Which products belong to the category 'Accesorios'?",
            {"filters": [{"column": "category", "operator": "==", "value": "Accesorios"}],
             "select": ["name"]},
            {"category", "categoría", "categoria", "belong", "pertenecen"},
        ),
        (
            "¿Cuál es la calificación del 'Monitor Curvo 4K'?",
            {"filters": [{"column": "name", "operator": "==", "value": "Monitor Curvo 4K"}],
             "select": ["name", "rating"]},
            {"rating", "calificación", "calificacion", "score", "puntuación"},
        ),
        (
            "¿Qué productos cuestan más de 1000?",
            {"filters": [{"column": "price", "operator": ">", "value": 1000}],
             "select": ["name", "price"]},
            {"más", "mas", "cuestan", "over", "above", "than", "caros", "expensive"},
        ),
    ]


def select_examples(question: str, max_examples: int = 4) -> list[tuple[str, dict]]:
    """Elige los ejemplos más parecidos a la pregunta del usuario.

    Compara las palabras de la pregunta con las palabras clave de cada ejemplo
    y conserva los que más coinciden. Los ejemplos se ordenan de menos a más
    parecido, de modo que el más relevante quede al final del prompt, que es
    el que el modelo tiende a imitar.

    Args:
        question (str): Pregunta formulada por el usuario.
        max_examples (int): Cantidad máxima de ejemplos a devolver. Por
            defecto es 4.

    Returns:
        list[tuple[str, dict]]: Lista de ejemplos, cada uno con su pregunta y
            su consulta, ordenados de menor a mayor parecido.
    """
    question_words = set(re.findall(r"\w+", question.lower()))

    # Se puntúa cada ejemplo según cuántas palabras clave comparte con la pregunta
    scored_examples = [
        (len(question_words & keywords), example_question, query)
        for example_question, query, keywords in get_example_bank()
    ]

    # El orden es ascendente: los últimos elementos son los más parecidos
    scored_examples.sort(key=lambda item: item[0])
    best_examples = scored_examples[-max_examples:]
    return [(example_question, query) for _, example_question, query in best_examples]


def build_reasoning_prompt(question: str) -> str:
    """Prepara las instrucciones que recibe el modelo para consultar el inventario.

    Incluye las columnas disponibles y los ejemplos más parecidos a la
    pregunta del usuario, para que el modelo imite el formato de consulta
    esperado.

    Args:
        question (str): Pregunta realizada por el usuario.

    Returns:
        str: Instrucciones completas, con los ejemplos y la pregunta.
    """
    header = (
        "You are an assistant for an electronics store.\n"
        "Columns: product_id, name, category, price, stock_quantity, rating.\n"
        "Answer with only the JSON query, on one line.\n\n"
    )

    examples_text = ""
    for example_question, query in select_examples(question):
        query_text = json.dumps(query, ensure_ascii=False)
        examples_text += f"Question: {example_question}\nQuery: {query_text}\n\n"

    return (header + examples_text + f"Question: {question}\nQuery:").strip()


def build_answer_prompt(question: str, tool_result: str) -> str:
    """Prepara las instrucciones para que el modelo redacte la respuesta final.

    Incluye dos ejemplos de cómo convertir los datos de la herramienta en una
    frase, para que el modelo conteste en lugar de repetir la pregunta.

    Args:
        question (str): Pregunta realizada por el usuario.
        tool_result (str): Resultados de la herramienta, en formato JSON.

    Returns:
        str: Instrucciones completas, con los ejemplos y los datos obtenidos.
    """
    return (
        "Answer the question using only the tool result.\n\n"
        "Question: What is the price of the 'Laptop Pro X15'?\n"
        "Tool result: [{\"name\": \"Laptop Pro X15\", \"price\": 1250.5}]\n"
        "Answer: The Laptop Pro X15 costs 1250.5.\n\n"
        "Question: ¿Cuántos 'Tablet S11' hay en stock?\n"
        "Tool result: [{\"name\": \"Tablet S11\", \"stock_quantity\": 30}]\n"
        "Answer: Hay 30 unidades de Tablet S11 en stock.\n\n"
        f"Question: {question}\n"
        f"Tool result: {tool_result}\n"
        "Answer:"
    ).strip()


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

    # Se desactiva la generación aleatoria para obtener resultados consistentes
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

    # Línea de revisión: muestra el prompt enviado (borrar al terminar las pruebas)
    print(f"[Prompt enviado]:\n{reasoning_prompt}\n")

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

    # Paso 5: preparar las instrucciones y generar la respuesta final
    final_prompt = build_answer_prompt(question, tool_result)
    final_answer = generate_text(model, tokenizer, final_prompt, max_new_tokens=40)

    # Conservar únicamente la primera línea de la respuesta
    final_answer = final_answer.strip().split("\n")[0]

    # Si el modelo solo repite la pregunta, se devuelven los datos tal cual
    if final_answer and question not in final_answer:
        return final_answer

    return tool_result