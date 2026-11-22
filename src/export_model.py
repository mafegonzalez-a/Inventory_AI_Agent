from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer


def export_model(model_name: str, output_dir: str) -> None:
    """Exporta un modelo de Hugging Face al formato ONNX.

    Args:
        model_name (str): Nombre del modelo en Hugging Face (ej. "distilgpt2").
        output_dir (str): Carpeta donde se guardará el modelo ONNX.

    Returns:
        None
    """
    # export=True convierte el modelo original a ONNX automáticamente
    model = ORTModelForCausalLM.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Guardamos modelo y tokenizer juntos en la misma carpeta
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo exportado en: {output_dir}")


def main() -> None:
    """Ejecuta la exportación con el modelo elegido.

    Returns:
        None
    """
    export_model("HuggingFaceTB/SmolLM2-135M-Instruct", "onnx_model_smollm")


if __name__ == "__main__":
    main()