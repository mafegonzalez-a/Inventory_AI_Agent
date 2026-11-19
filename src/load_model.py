from typing import Tuple

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_onnx_model(model_dir: str) -> Tuple[ORTModelForCausalLM, PreTrainedTokenizerBase]:
    """Abre el modelo ONNX y su tokenizer para poder usarlos.

    El modelo ya fue convertido a ONNX y guardado en una carpeta (con el
    archivo export_model.py). Se trabaja con el modelo y el tokenizer.

    Args:
        model_dir (str): Nombre o ruta de la carpeta donde está guardado el
            modelo ONNX. 

    Returns:
        Tuple[ORTModelForCausalLM, PreTrainedTokenizerBase]: El modelo y el
            tokenizer, en ese orden
    """

    model = ORTModelForCausalLM.from_pretrained(model_dir)

    # Se abre el tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Este modelo no trae un pad_token, el cual es necesario por el tokenizer que a veces necesita para ordenar textos de distinto largo.
    # Si llega a faltar, se usa la ficha de "fin de texto" para evitar errores
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Modelo cargado correctamente")
    return model, tokenizer