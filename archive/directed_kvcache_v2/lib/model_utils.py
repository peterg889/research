"""
Model loading utilities with per-model-type dtype and quantization handling.

Centralizes model loading so experiments don't need model-specific boilerplate.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def _resolve_dtype(config: ExperimentConfig) -> torch.dtype:
    """Resolve compute dtype from config.

    Rules:
    - "auto": bfloat16 for gemma3 (fp16 produces empty output), float16 for others
    - "float16" / "bfloat16": explicit override
    """
    if config.compute_dtype == "auto":
        return torch.bfloat16 if config.model_type == "gemma3" else torch.float16
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[config.compute_dtype]


def load_model(config: ExperimentConfig):
    """Load model + tokenizer with correct dtype/quantization for model type.

    Args:
        config: ExperimentConfig with model_name, model_type, use_4bit, compute_dtype.

    Returns:
        Tuple of (model, tokenizer)
    """
    dtype = _resolve_dtype(config)

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }

    if config.use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model.eval()

    return model, tokenizer
