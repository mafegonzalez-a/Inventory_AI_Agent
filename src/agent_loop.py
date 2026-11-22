import json
import re
from typing import Callable, Optional

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import PreTrainedTokenizerBase


def get_examples() -> list[tuple[str, dict, set[str]]]:
    """Proporciona ejemplos de preguntas y consultas para orientar al modelo.

    Cada ejemplo incluye una pregunta, la consulta que le corresponde y
    palabras clave para identificar el tipo de información solicitada.
    Se incluyen consultas sobre categorías, calificaciones, stock, promedios,
    precios y productos con la calificación o el precio más alto.

    Returns:
        list[tuple[str, dict, set[str]]]: Lista de ejemplos con la pregunta,
            su consulta y las palabras clave relacionadas.
    """
    return [
        (
            "Which products belong to the category 'Accesorios'?",
            {"filters": [{"column": "category", "operator": "==",
                          "value": "Accesorios"}],
             "select": ["name"]},
            {"category", "categoría", "categoria", "belong", "pertenecen"},
        ),
        (
            "What is the rating of the 'Monitor Curvo 4K'?",
            {"filters": [{"column": "name", "operator": "==",
                          "value": "Monitor Curvo 4K"}],
             "select": ["name", "rating"]},
            {"rating", "calificación", "calificacion", "score", "puntuación"},
        ),
        (
            "How many 'Tablet S11' are in stock?",
            {"filters": [{"column": "name", "operator": "==",
                          "value": "Tablet S11"}],
             "select": ["name", "stock_quantity"]},
            {"stock", "many", "cuántos", "cuantos", "unidades", "inventario"},
        ),
        (
            "What is the average price of all products?",
            {"select": ["price"], "operation": "mean"},
            {"average", "promedio", "mean", "media", "all", "todos", "todas"},
        ),
        (
            "What is the price of the 'Laptop Pro X15'?",
            {"filters": [{"column": "name", "operator": "==",
                          "value": "Laptop Pro X15"}],
             "select": ["name", "price"]},
            {"price", "precio", "cost", "cuesta", "cuestan", "cuánto",
             "cuanto", "much"},
        ),
        (
            "Which product is the most expensive?",
            {"select": ["name", "price"], "sort_by": "price",
             "ascending": False, "limit": 1},
            {"highest", "best", "top", "alta", "alto", "mayor", "mejor",
             "expensive", "caro", "costoso"},
        ),
        (
            "Which product has the highest rating?",
            {"select": ["name", "rating"], "sort_by": "rating",
             "ascending": False, "limit": 1},
            {"highest", "best", "top", "alta", "alto", "mayor", "mejor",
             "rating", "calificación", "calificacion"},
        ),
                (
            "¿Qué producto tiene la mejor calificación?",
            {"select": ["name", "rating"], "sort_by": "rating",
             "ascending": False, "limit": 1},
            {"alta", "alto", "mayor", "mejor", "calificación", "calificacion",
             "más"},
        ),
    ]


def order_examples(question: str) -> list[tuple[str, dict]]:
    """Organiza los ejemplos según su relación con la pregunta del usuario.

    Compara las palabras de la pregunta con las palabras clave de cada
    ejemplo. Los ejemplos con menos coincidencias aparecen primero y el
    más relacionado queda al final, para que el modelo lo tenga como
    referencia más cercana (se mantienen todos los ejemplos)

    Args:
        question (str): Pregunta realizada por el usuario.

    Returns:
        list[tuple[str, dict]]: Ejemplos ordenados de menor a mayor
            coincidencia, cada uno con su pregunta y consulta.
    """
    question_words = set(re.findall(r"\w+", question.lower()))

    # Se cuenta cuántas palabras clave coinciden con la pregunta
    scored_examples = [
        (len(question_words & keywords), example_question, query)
        for example_question, query, keywords in get_examples()
    ]

    # Si dos ejemplos tienen la misma puntuación, se conserva su orden original
    scored_examples.sort(key=lambda item: item[0])
    return [(example_question, query) for _, example_question, query in scored_examples]


def build_reasoning_prompt(question: str) -> str:
    """Prepara las instrucciones para que el modelo genere una consulta.

    Incluye las columnas disponibles en el inventario, la relación entre
    algunos términos en español y sus nombres en inglés, y ejemplos de
    preguntas con sus consultas correspondientes. Los ejemplos se ordenan
    según su relación con la pregunta del usuario.

    Args:
        question (str): Pregunta realizada por el usuario.

    Returns:
        str: Instrucciones completas que recibirá el modelo.
    """
    # Se eliminan los espacios al inicio y al final de la pregunta
    question = question.strip()

    # Se relacionan los términos en español con los nombres de las columnas
    header = (
        "You are an assistant for an electronics store.\n"
        "Columns: product_id, name, category, price, stock_quantity, rating.\n"
        "Spanish terms: precio = price, calificación = rating, "
        "categoría = category, stock = stock_quantity, nombre = name, "
        "promedio = average, más alta = highest.\n"
        "Answer with only the JSON query, on one line.\n\n"
    )

    examples_text = ""
    for example_question, query in order_examples(question):
        query_text = json.dumps(query, ensure_ascii=False)
        examples_text += f"Question: {example_question}\nQuery: {query_text}\n\n"

    return header + examples_text + f"Question: {question}\nQuery:"


def build_answer_prompt(question: str, tool_result: str) -> str:
    """Prepara las instrucciones para redactar la respuesta al usuario.

    Incluye tres ejemplos que muestran cómo utilizar los resultados de la
    herramienta para responder preguntas sobre listas de productos,
    cantidades disponibles y precios. También incorpora la pregunta original
    y los datos obtenidos de la consulta.

    Args:
        question (str): Pregunta realizada por el usuario.
        tool_result (str): Resultados de la consulta en formato JSON.

    Returns:
        str: Instrucciones completas con los ejemplos y los resultados.
    """
    return (
        "Answer the question using only the tool result.\n\n"
        "Question: Which products belong to the category 'Accesorios'?\n"
        'Tool result: [{"name": "Hub USB-C 8-en-1"}, {"name": "Webcam Pro 4K"}]\n'
        "Answer: The products are Hub USB-C 8-en-1 and Webcam Pro 4K.\n\n"
        "Question: How many 'Tablet S11' are in stock?\n"
        'Tool result: [{"name": "Tablet S11", "stock_quantity": 30}]\n'
        "Answer: There are 30 units of Tablet S11 in stock.\n\n"
        "Question: What is the price of the 'Laptop Pro X15'?\n"
        'Tool result: [{"name": "Laptop Pro X15", "price": 1250.5}]\n'
        "Answer: The Laptop Pro X15 costs 1250.5.\n\n"
        f"Question: {question}\n"
        f"Tool result: {tool_result}\n"
        "Answer:"
    )


def generate_text(
    model: ORTModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 80,
) -> str:
    """Genera texto a partir de las instrucciones recibidas.

    El tokenizer convierte el texto en números que el modelo puede procesar.
    Después, el modelo genera nuevos tokens y el tokenizer los convierte
    nuevamente en texto para obtener la respuesta.

    Args:
        model (ORTModelForCausalLM): Modelo ONNX previamente cargado.
        tokenizer (PreTrainedTokenizerBase): Tokenizer asociado al modelo.
        prompt (str): Instrucciones o pregunta que recibe el modelo.
        max_new_tokens (int): Número máximo de tokens nuevos que puede
            generar el modelo. Por defecto es 80.

    Returns:
        str: Texto generado por el modelo, sin incluir las instrucciones
            originales.
    """
    # Se convierte el texto en números para que el modelo pueda procesarlo
    inputs = tokenizer(prompt, return_tensors="pt")

    # Se desactiva la generación aleatoria para mantener respuestas consistentes
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Se toman únicamente los tokens generados después del prompt
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_decision(decision_text: str) -> Optional[dict]:
    """Convierte el texto generado por el modelo en un diccionario.

    Busca el contenido JSON en la respuesta del modelo y lo convierte
    en un diccionario que Python pueda utilizar para consultar el inventario.
    Si no encuentra un JSON válido, devuelve None.

    Args:
        decision_text (str): Texto generado por el modelo.

    Returns:
        Optional[dict]: Consulta en formato diccionario o None si el
            contenido no tiene un formato JSON válido.
    """
    # Se toma la primera línea para evitar texto adicional del modelo
    first_line = decision_text.strip().split("\n")[0]

    # Se busca el contenido que está entre las llaves del JSON
    match = re.search(r"\{.*\}", first_line)
    if match is None:
        return None

    # Se intenta convertir el contenido encontrado en un diccionario
    try:
        query = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    # Se comprueba que el resultado sea un diccionario
    if not isinstance(query, dict):
        return None
    return query


def answer_uses_result(answer: str, rows: list[dict], question: str) -> bool:
    """Comprueba que la respuesta incluya los datos obtenidos de la consulta.

    Revisa que el modelo utilice todos los valores nuevos devueltos por la
    herramienta. Esto permite identificar respuestas incompletas o que no
    corresponden a los resultados. Los textos que ya estaban en la pregunta,
    como el nombre del producto, no se consideran datos nuevos.

    Args:
        answer (str): Respuesta redactada por el modelo.
        rows (list[dict]): Resultados devueltos por la herramienta.
        question (str): Pregunta realizada por el usuario.

    Returns:
        bool: True si la respuesta contiene todos los valores nuevos o si
            no hay valores nuevos que deban comprobarse.
    """
    question_text = question.lower()

    # Se excluyen los textos que ya aparecen en la pregunta
    new_values = []
    for row in rows:
        for value in row.values():
            if isinstance(value, str) and value.lower() in question_text:
                continue
            new_values.append(value)

    # Si no hay datos nuevos, no hay valores que comprobar
    if not new_values:
        return True

    for value in new_values:
        candidates = {str(value)}

        # Se consideran distintos formatos para representar números decimales
        if isinstance(value, float):
            candidates.add(f"{value:.2f}")
            candidates.add(str(round(value, 1)))
            if value.is_integer():
                candidates.add(str(int(value)))

        # Si falta al menos un valor, la respuesta se considera incompleta
        if not any(candidate in answer for candidate in candidates):
            return False
    return True


def format_rows(rows: list[dict]) -> str:
    """Organiza los resultados de la consulta en un texto fácil de leer.

    Se utiliza como alternativa cuando la respuesta del modelo está
    incompleta o no coincide con los datos obtenidos. Si cada fila contiene
    un solo valor, los resultados se separan por comas. Si contiene varias
    columnas, se muestra cada dato junto con el nombre de su columna.

    Args:
        rows (list[dict]): Resultados devueltos por la herramienta.

    Returns:
        str: Texto con los resultados listo para mostrar al usuario.
    """
    # Cuando cada fila tiene un solo valor, se separan los resultados por comas
    if all(len(row) == 1 for row in rows):
        values = [str(next(iter(row.values()))) for row in rows]
        return ", ".join(values)

    # Cuando hay varias columnas, se muestra cada fila en una línea
    lines = []
    for row in rows:
        lines.append(", ".join(f"{column}: {value}" for column, value in row.items()))
    return "\n".join(lines)


def run_agent_loop(
    question: str,
    model: ORTModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    tool_function: Callable[[dict], list[dict]],
) -> str:
    """Coordina el proceso del agente desde la pregunta hasta la respuesta.

    Conecta el modelo ONNX con la herramienta que consulta el inventario.
    Primero, el modelo genera los parámetros de búsqueda. Después, la
    herramienta utiliza esos parámetros para obtener los resultados y,
    finalmente, el modelo prepara una respuesta para el usuario.

    Args:
        question (str): Pregunta realizada por el usuario.
        model (ORTModelForCausalLM): Modelo ONNX previamente cargado.
        tokenizer (PreTrainedTokenizerBase): Tokenizer asociado al modelo.
        tool_function (Callable[[dict], list[dict]]): Función que recibe
            los parámetros de búsqueda y devuelve los resultados del inventario.

    Returns:
        str: Respuesta para el usuario o un mensaje cuando no es posible
            completar la consulta.
    """
    # Paso 1: preparar las instrucciones que recibirá el modelo
    reasoning_prompt = build_reasoning_prompt(question)

    # Mostrar el prompt para revisar las instrucciones durante las pruebas
    print(f"[Prompt enviado]:\n{reasoning_prompt}\n")

    # Paso 2: generar los parámetros para consultar el inventario
    decision = generate_text(model, tokenizer, reasoning_prompt)
    print(f"[Decisión del modelo]: {decision.strip()}")

    # Paso 3: convertir la respuesta del modelo en un diccionario
    query = parse_decision(decision)
    if query is None:
        return "Lo siento, no pude entender cómo consultar esa información."

    # Paso 4: utilizar los parámetros generados para consultar el inventario
    try:
        rows = tool_function(query)
    except (KeyError, ValueError, TypeError, FileNotFoundError) as error:
        print(f"[Error de la herramienta]: {error}")
        return "Lo siento, no pude consultar el inventario con esa solicitud."

    # Convertir los resultados a JSON para enviarlos al modelo
    tool_result = json.dumps(rows, ensure_ascii=False)
    print(f"[Resultado de la herramienta]: {tool_result}")

    # Si la consulta no devuelve resultados, se informa al usuario
    if not rows:
        return "No encontré productos que coincidan con tu pregunta."

    # Paso 5: preparar las instrucciones y generar la respuesta final
    final_prompt = build_answer_prompt(question, tool_result)
    final_answer = generate_text(model, tokenizer, final_prompt, max_new_tokens=60)

    # Conservar únicamente la primera línea de la respuesta
    final_answer = final_answer.strip().split("\n")[0]

    # Aceptar la respuesta solo si no repite la pregunta y contiene
    # todos los datos nuevos obtenidos de la consulta
    if (
        final_answer
        and question not in final_answer
        and answer_uses_result(final_answer, rows, question)
    ):
        return final_answer

    # Si la respuesta no cumple las condiciones, mostrar los datos obtenidos
    return format_rows(rows)