"""Interfaz Gradio del agente de inventario (tienda de electronica ficticia).

Reproduce el flujo de main.py: carga el modelo ONNX una sola vez y, para cada
pregunta, llama a run_agent_loop(question, model, tokenizer, get_information).

Si los modulos del agente (src/agent_loop.py y src/load_model.py) todavia no
estan en esta rama, la interfaz corre en MODO DEMO, de modo que se pueda
desarrollar y probar antes de que se una la rama onnx_agent_loop.
"""
import contextlib
import io
import os
import threading

import gradio as gr

# Las rutas del proyecto (data/products.csv, onnx_model_smollm/) son relativas
# a la raiz del repositorio: nos aseguramos de trabajar desde ahi.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = "onnx_model_smollm"

try:
    from src.agent_loop import run_agent_loop
    from src.load_model import load_onnx_model
    from src.tools import get_information

    MODO_DEMO = False
except ModuleNotFoundError as exc:
    # Solo caemos al demo si faltan los modulos del agente; si falta otra
    # dependencia (p. ej. optimum) queremos ver el error real.
    if exc.name not in {"src", "src.agent_loop", "src.load_model"}:
        raise
    MODO_DEMO = True

_cache: dict = {}
_lock = threading.Lock()


def _cargar_modelo():
    """Carga el modelo ONNX y su tokenizer la primera vez que se necesitan."""
    if "modelo" not in _cache:
        if not os.path.isdir(MODEL_DIR):
            raise FileNotFoundError(
                f"No existe la carpeta '{MODEL_DIR}'. "
                "Generala con: python src/export_model.py"
            )
        _cache["modelo"] = load_onnx_model(MODEL_DIR)
    return _cache["modelo"]


def run_agent(question: str) -> tuple[str, str]:
    """Devuelve (respuesta, traza). La traza es lo que el agente imprima."""
    if MODO_DEMO:
        return (
            "Modo demo: el agente aun no esta conectado. "
            f"Pregunta recibida: {question!r}",
            "1. Pregunta recibida\n"
            "2. (pendiente) El modelo ONNX genera la consulta en JSON\n"
            "3. (pendiente) get_information consulta data/products.csv\n"
            "4. (pendiente) El modelo redacta la respuesta final",
        )
    with _lock:  # un solo usuario a la vez: el modelo y stdout son compartidos
        model, tokenizer = _cargar_modelo()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            answer = run_agent_loop(question, model, tokenizer, get_information)
    return str(answer), buffer.getvalue().strip()


EJEMPLOS = [
    "¿Cuál es el precio del 'Ultrabook Air M3'?",
    "¿Cuántos 'Smartphone G-Max' hay en stock?",
    "¿Qué productos pertenecen a la categoría 'Accesorios'?",
    "¿Cuál es el producto más caro?",
]


def responder(question: str) -> tuple[str, str]:
    question = (question or "").strip()
    if not question:
        return "Escribe una pregunta sobre el inventario.", ""
    try:
        answer, trace = run_agent(question)
    except Exception as exc:  # la interfaz no debe caerse si el agente falla
        return f"El agente produjo un error: {exc}", ""
    return answer, trace or "(el agente no imprimio pasos intermedios)"


with gr.Blocks(title="Agente de inventario") as demo:
    gr.Markdown(
        "# Agente de inventario\n"
        "Pregunta por precios, existencias, categorías o calificaciones de la "
        "tienda de electrónica. El modelo no conoce los datos: consulta "
        "`data/products.csv` mediante la herramienta `get_information`."
    )
    if MODO_DEMO:
        gr.Markdown("**Modo demo:** el agente (`src/agent_loop.py`) aún no está en esta rama.")

    pregunta = gr.Textbox(label="Pregunta", placeholder="¿Cuál es el precio del 'Laptop Pro X'?")
    enviar = gr.Button("Preguntar", variant="primary")
    respuesta = gr.Textbox(label="Respuesta", lines=4)
    with gr.Accordion("Razonamiento del agente", open=False):
        traza = gr.Textbox(label="Traza", lines=8)

    gr.Examples(examples=EJEMPLOS, inputs=pregunta)

    enviar.click(responder, inputs=pregunta, outputs=[respuesta, traza])
    pregunta.submit(responder, inputs=pregunta, outputs=[respuesta, traza])

if __name__ == "__main__":
    demo.launch()