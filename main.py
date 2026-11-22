from src.agent_loop import run_agent_loop
from src.load_model import load_onnx_model
from src.tools import get_information  


def main() -> None:
    """Punto de entrada del agente de consulta de inventario.

    Carga una sola vez el modelo ONNX (SmolLM2-135M-Instruct, guardado en la
    carpeta onnx_model_smollm) y su tokenizer, y luego atiende las preguntas
    del usuario por consola de forma repetida. Cada pregunta se procesa con
    el bucle del agente, que consulta el inventario mediante la herramienta
    get_information y devuelve una respuesta en lenguaje natural.
    El programa termina cuando el usuario escribe "salir".

    Returns:
        None
    """
    # El modelo se carga fuera del ciclo porque es la operación más lenta,
    # así se evita repetirla en cada pregunta. La carpeta se genera con
    # src/export_model.py y debe existir en la raíz del proyecto
    model, tokenizer = load_onnx_model("onnx_model_smollm")

    while True:
        question = input("\nPregunta (o 'salir'): ")

        # Se ignoran los espacios y las mayúsculas para detectar la salida
        if question.strip().lower() == "salir":
            break

        # El bucle del agente procesa la pregunta con la herramienta de consulta
        answer = run_agent_loop(question, model, tokenizer, get_information)
        print(f"Respuesta: {answer}")


if __name__ == "__main__":
    main()