# Inventory AI Agent

**Integrantes:** Samir Mateo Sierra Blanco, María Fernanda González Acevedo y Karla Paola Medina Escorcia.

## Descripción

Inventory AI Agent es un agente de inteligencia artificial que responde preguntas sobre el inventario de una tienda de electrónica ficticia. Puede consultar información sobre precios, existencias y categorías de productos.

El modelo de lenguaje no tiene acceso directo a los datos del inventario. En su lugar, utiliza ejemplos incluidos en el prompt para generar una consulta en formato JSON. Esta consulta es ejecutada por una herramienta de Python sobre el archivo `data/products.csv`, y sus resultados se utilizan para generar la respuesta al usuario.

Como parte del desafío del proyecto se implementó una versión que funciona con un modelo local exportado al formato ONNX mediante Optimum.

Algunas preguntas que puede responder son:

- What is the price of the 'Laptop Pro X15'? / ¿Cuál es el precio del 'Laptop Pro X15'?
- How many 'Smartphone G-Max' are in stock? / ¿Cuántos 'Smartphone G-Max' hay en stock?
- Which products belong to the category 'Accesorios'? / ¿Qué productos pertenecen a la categoría 'Accesorios'?
- Which product has the highest rating?

## El modelo

| Dato | Valor |
|---|---|
| Modelo | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Parámetros | 135 millones |
| Tamaño de la descarga original | Aproximadamente 269 MB |
| Formato utilizado | ONNX |
| Clase de carga | `ORTModelForCausalLM` (Optimum) |
| Hardware | CPU, no requiere GPU |
| Carpeta local | `onnx_model_smollm/` |

SmolLM2 es una familia de modelos de lenguaje pequeños, entrenados para seguir instrucciones. La versión de 135 millones de parámetros es una de las más ligeras de la familia y permite trabajar con recursos de hardware limitados.

## ¿Por qué se cambió el modelo?

El primer modelo que se probó fue `distilgpt2`, de aproximadamente 82 millones de parámetros, tal como sugiere el enunciado. Sin embargo, aunque es un modelo pequeño, no fue entrenado específicamente para seguir instrucciones, sino para continuar texto.

Durante las pruebas, se observó que frecuentemente copiaba el último ejemplo del prompt sin tener en cuenta la pregunta del usuario. Por ejemplo, al preguntar por el precio de un producto, podía generar una consulta para calcular el precio promedio y luego presentar ese resultado como si fuera el precio solicitado.

Otras alternativas consideradas:

- `Qwen/Qwen2.5-0.5B-Instruct`: tiene 494 millones de parámetros y ofrece mejores capacidades para seguir instrucciones, pero requiere más recursos. No fue posible exportarlo en el equipo de desarrollo. Por lo que se opto por:
- `SmolLM2-135M-Instruct`: es un modelo entrenado para seguir instrucciones, tiene un tamaño reducido y pudo exportarse y ejecutarse en el equipo disponible. Por estas razones, fue la opción seleccionada.

## ¿Cómo funciona el agente?

El agente sigue un flujo de varios pasos para convertir la pregunta del usuario en una consulta sobre el inventario:

1. **Pregunta del usuario:** el usuario escribe una pregunta sobre los productos.
2. **Construcción del prompt:** `agent_loop.py` prepara las instrucciones para el modelo. Incluye las columnas del inventario, un glosario con algunos términos en español y ejemplos de consultas. Los ejemplos se ordenan según su similitud con la pregunta, de manera que el más parecido quede al final, ya que el modelo tiende a imitar los ejemplos más cercanos.
3. **Generación de la consulta:** el modelo ONNX genera una consulta en formato JSON.
4. **Interpretación de la consulta:** `parse_decision` convierte el texto generado por el modelo en un diccionario de Python.
5. **Ejecución de la herramienta:** `get_information`, definida en `tools.py`, ejecuta la consulta sobre `data/products.csv`. Puede filtrar, agrupar, calcular valores, ordenar y limitar los resultados.
6. **Generación de la respuesta:** los resultados obtenidos se envían nuevamente al modelo para que redacte una respuesta en lenguaje natural.
7. **Validación de la respuesta:** el programa comprueba que la respuesta no esté vacía, que no repita la pregunta y que incluya los datos obtenidos. Si la respuesta está incompleta o no cumple estas condiciones, muestra directamente los resultados mediante `format_rows`.

## Limitaciones en preguntas en español

Durante las pruebas se encontró una dificultad con algunas preguntas en español. Por ejemplo, la pregunta "¿Cuál es el producto con la calificación más alta?" no siempre generaba la consulta correcta, aunque el modelo podía responderla en inglés utilizando un ejemplo similar.

## Archivos del proyecto

| Archivo o carpeta | Descripción |
|---|---|
| `main.py` | Punto de entrada del programa. Carga el modelo y permite realizar preguntas desde la consola hasta que el usuario escriba `salir`. |
| `src/export_model.py` | Descarga el modelo y lo exporta al formato ONNX. Se ejecuta una sola vez. |
| `src/load_model.py` | Carga el modelo ONNX y su tokenizer desde una carpeta local. |
| `src/agent_loop.py` | Contiene los prompts, la interpretación de las consultas, la llamada a la herramienta y la generación de la respuesta final. |
| `src/tools.py` | Contiene la herramienta `get_information`, encargada de consultar el inventario. Incluye filtros, agregaciones, agrupaciones, orden ascendente o descendente y límite de resultados. |
| `src/__init__.py` | Permite utilizar `src` como un paquete de Python. |
| `data/products.csv` | Archivo con los datos del inventario. |
| `onnx_model_smollm/` | Carpeta donde se guarda el modelo exportado. No está incluida en el repositorio, se genera mediante `export_model.py`. |
| `requirements.txt` | Lista de librerías necesarias para ejecutar el proyecto. |

## Funciones principales de `agent_loop.py`

| Función | Descripción |
|---|---|
| `get_examples` | Define los ejemplos de consultas y sus respectivas palabras clave. |
| `order_examples` | Ordena los ejemplos según su similitud con la pregunta del usuario. |
| `build_reasoning_prompt` | Prepara el prompt que recibe el modelo para generar la consulta. |
| `build_answer_prompt` | Prepara el prompt que recibe el modelo para generar la respuesta final. |
| `generate_text` | Envía el prompt al modelo y devuelve el texto generado. |
| `parse_decision` | Convierte el texto generado por el modelo en un diccionario de Python. |
| `answer_uses_result` | Comprueba que la respuesta incluya los datos nuevos obtenidos de la consulta. |
| `format_rows` | Organiza los resultados en un formato de texto legible cuando se necesita una respuesta alternativa. |
| `run_agent_loop` | Ejecuta el flujo completo del agente para responder una pregunta. |

## Instrucciones de instalación y ejecución

### Requisitos

- Python 3.12.
- Git.
- Conexión a internet para descargar el modelo por primera vez.

Se recomienda utilizar Python 3.12, ya que algunas de las librerías utilizadas pueden presentar problemas de compatibilidad con versiones más recientes.

### 1. Clonar el repositorio

Clona el repositorio y accede a la carpeta del proyecto.

### 2. Crear el entorno virtual e instalar las dependencias

Todos los comandos deben ejecutarse desde la carpeta raíz del proyecto, ya que el código busca los archivos `onnx_model_smollm/` y `data/products.csv` desde esa ubicación.

En Windows, ejecuta:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Exportar el modelo

Antes de ejecutar el agente, es necesario descargar y exportar el modelo a ONNX. Este proceso se realiza una sola vez y crea la carpeta `onnx_model_smollm/`.

```powershell
python src/export_model.py
```

Al finalizar, se muestra el mensaje:

```
Modelo exportado en: onnx_model_smollm
```

### 4. Ejecutar el agente

Una vez exportado el modelo, ejecuta:

```powershell
python main.py
```

Escribe una pregunta sobre el inventario y presiona Enter. Para terminar, escribe `salir`.

Recomendaciones al escribir las preguntas:

- Escribe el nombre del producto entre comillas simples.
- Utiliza las mayúsculas y minúsculas tal como aparecen en el archivo CSV.
- El modelo suele obtener mejores resultados con preguntas en inglés.

Durante la ejecución, se muestran en la consola los diferentes pasos del agente:

- `[Prompt enviado]`: instrucciones que recibe el modelo.
- `[Decisión del modelo]`: consulta generada en formato JSON.
- `[Resultado de la herramienta]`: datos obtenidos del archivo CSV.
- `Respuesta`: resultado final que recibe el usuario.

## Preguntas de prueba

Las siguientes preguntas permiten comprobar algunas de las funciones principales del agente.

| Pregunta en inglés | Pregunta en español | Resultado esperado |
|---|---|---|
| What is the price of the 'Laptop Pro X15'? | ¿Cuál es el precio del 'Laptop Pro X15'? | 1250.5 |
| How many 'Smartphone G-Max' are in stock? | ¿Cuántos 'Smartphone G-Max' hay en stock? | 35 |
| Which products belong to the category 'Accesorios'? | ¿Qué productos pertenecen a la categoría 'Accesorios'? | 7 productos |
| Which product has the highest rating? | ¿Cuál es el producto con la calificación más alta? | Un producto con calificación de 4.9 |

## Limitaciones conocidas

- **Tamaño del modelo:** SmolLM2-135M es un modelo pequeño y funciona mejor con preguntas similares a los ejemplos incluidos en el prompt. Cuando la pregunta tiene una estructura muy diferente, puede generar una consulta con una columna o un producto equivocado.
- **Preguntas en español:** el modelo obtiene mejores resultados con preguntas en inglés. Puede tener dificultades para interpretar algunas preguntas en español.
- **Respuestas en formato de datos:** cuando el modelo genera una frase incompleta o no incluye todos los resultados, el programa muestra directamente los datos obtenidos. Por ejemplo: `name: Phone Ultra Pro 16, rating: 4.9`.
- **Nombres exactos:** la herramienta utiliza una comparación exacta (`==`) para buscar los nombres de los productos. Por eso, es importante escribirlos tal como aparecen en el CSV, incluyendo mayúsculas y minúsculas.
- **Nombres largos o con símbolos:** al modelo le puede costar copiar correctamente algunos nombres de productos, especialmente cuando son largos o contienen caracteres especiales. Por ejemplo: `Auriculares con Cancelación Z-Pro` o `Funda para Laptop 16"`.