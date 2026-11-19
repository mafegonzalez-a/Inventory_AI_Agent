# Inventory AI Agent

*Proyecto final de Transformers, Parte 2. Desarrollado por Karla, Samir y Mafer.*

## Descripción

Agente de inteligencia artificial que responde preguntas sobre el inventario de una tienda de electrónica ficticia (precios, existencias, categorías y calificaciones). El modelo de lenguaje no conoce los datos de antemano: aprende, mediante ejemplos en el prompt, a usar una herramienta de Python que consulta el archivo `data/products.csv`.

Se implementó el **desafío**: el agente funciona con un modelo local (`distilgpt2`) exportado al formato **ONNX** con Optimum, sin depender de la API de Hugging Face. El bucle del agente se escribió manualmente, sin librerías de agentes.

Ejemplos de preguntas que puede responder:

- ¿Cuál es el precio del 'Laptop Pro X'?
- ¿Cuántos 'Smartphone G5' hay en stock?
- ¿Qué productos pertenecen a la categoría 'Accesorios'?
- ¿Cuál es el precio promedio de todos los productos?

### Cómo funciona

1. El usuario escribe una pregunta.
2. `build_reasoning_prompt` arma un prompt con la descripción de la herramienta `get_information` y tres ejemplos de consultas.
3. El modelo ONNX genera la consulta en formato JSON.
4. `parse_decision` convierte ese texto en un diccionario de Python.
5. `get_information` ejecuta la consulta sobre el CSV y devuelve una lista de resultados.
6. Los resultados vuelven al modelo, que redacta la respuesta final.

El modelo solo genera texto: no accede a los datos ni ejecuta código. El bucle del agente es el que conecta cada paso.

### Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `main.py` | Punto de entrada. Carga el modelo una vez y atiende preguntas por consola hasta que se escribe `salir`. |
| `src/export_model.py` | Descarga `distilgpt2` y lo exporta a ONNX en la carpeta `onnx_model/`. Se ejecuta una sola vez. |
| `src/load_model.py` | Carga el modelo ONNX y su tokenizer desde una carpeta local. |
| `src/agent_loop.py` | Construye los prompts, interpreta la consulta del modelo, llama a la herramienta y genera la respuesta final. |
| `src/tools.py` | Herramienta `get_information`: filtra, agrega, agrupa y ordena los datos del CSV. |
| `src/__init__.py` | Marca `src` como paquete de Python. |
| `data/products.csv` | Datos del inventario. |
| `onnx_model/` | Modelo exportado a ONNX (se genera con `export_model.py`). |
| `requirements.txt` | Librerías necesarias. |
| `.gitignore` | Archivos que Git no debe subir (entorno virtual, cachés). |
| `README.md` | Este archivo. |

### Organización del código

Cada archivo contiene funciones con tipos declarados y documentación estilo Google. Los módulos solo definen funciones, por lo que no ocurre nada al importarlos. Solo `main.py` y `export_model.py` se ejecutan directamente.

| Función | Archivo | Qué hace |
|---|---|---|
| `export_model` | `export_model.py` | Convierte un modelo de Hugging Face a ONNX y lo guarda. |
| `load_onnx_model` | `load_model.py` | Abre el modelo y el tokenizer. |
| `build_reasoning_prompt` | `agent_loop.py` | Crea el prompt con la herramienta, los ejemplos y la pregunta. |
| `generate_text` | `agent_loop.py` | Pasa un prompt al modelo y devuelve el texto generado. |
| `parse_decision` | `agent_loop.py` | Convierte el texto del modelo en un diccionario de consulta. |
| `run_agent_loop` | `agent_loop.py` | Ejecuta el flujo completo para una pregunta. |
| `get_information` | `tools.py` | Consulta el CSV según el diccionario recibido. |

## Instrucciones

### Requisitos

- Python 3.12. Las librerías de IA aún no son compatibles con versiones muy recientes (por ejemplo, 3.15).
- Git y, si el modelo se descarga desde el repositorio, Git LFS.
- Conexión a internet la primera vez, para descargar el modelo.

### Instalación

Todos los comandos se ejecutan desde la carpeta raíz del proyecto, porque el código busca `onnx_model/` y `data/products.csv` desde ahí.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Sabrás que el entorno está activo porque aparece `(.venv)` al inicio de la línea de la terminal.

### Exportar el modelo

Este paso se hace una sola vez y crea la carpeta `onnx_model/`. Si la carpeta ya existe, se puede omitir.

```powershell
python src/export_model.py
```

### Ejecutar el agente

```powershell
python main.py
```

Escribe una pregunta y presiona Enter. Para terminar, escribe `salir`.

Durante la ejecución se muestran tres mensajes por pregunta:

- `[Decisión del modelo]`: la consulta que generó el modelo.
- `[Resultado de la herramienta]`: los datos que devolvió el inventario.
- `Respuesta`: la respuesta final.

### Si algo falla

| Mensaje | Causa probable | Qué hacer |
|---|---|---|
| `No module named 'src'` | Se ejecutó desde otra carpeta | Ejecutar desde la raíz del proyecto |
| `Repository Not Found` para `onnx_model` | No existe la carpeta del modelo en la raíz | Ejecutar `python src/export_model.py` desde la raíz |
| `No module named 'optimum'` | El entorno virtual no está activo | Activar `.venv` e instalar `requirements.txt` |
| `Lo siento, no pude entender cómo consultar esa información.` | El modelo generó una consulta mal formada | Reformular la pregunta |

### Reiniciar desde cero

Para volver a generar el modelo, borra la carpeta `onnx_model/` y ejecuta de nuevo `python src/export_model.py`. Para cambiar de modelo, edita el nombre en la función `main` de `export_model.py` (por ejemplo, `Qwen/Qwen2.5-0.5B-Instruct`) y vuelve a exportar.


## Recursos

- Documentación de Optimum (exportación y ejecución de modelos ONNX): https://huggingface.co/docs/optimum/index
- Documentación de Transformers: https://huggingface.co/docs/transformers
- ONNX: https://onnx.ai
- Modelo utilizado: https://huggingface.co/distilbert/distilgpt2
- Datos del inventario: https://raw.githubusercontent.com/adiacla/transformers/refs/heads/main/productos.csv

## Equipo

Integrantes

Samir XXXXXX
Maria Fernanda XXXXXX
Karla Paola Medina Escorcia