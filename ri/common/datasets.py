from __future__ import annotations

from typing import Any

import torch
from datasets import load_dataset


def get_dataset(dataset_name: str) -> Any:
    """
    Load a dataset by name or from a JSON file.
    """
    if dataset_name.endswith(".json"):
        return load_dataset("json", data_files=dataset_name)
    if dataset_name == "gsm8k":
        return load_dataset("openai/gsm8k", "main")
    raise ValueError(f"Dataset '{dataset_name}' not supported.")


def choose_torch_dtype(model_name: str, use_fp32: bool = False):
    """
    Determine an appropriate torch dtype for the model.
    """
    if use_fp32:
        return torch.float32
    # Prefer bf16 when the hardware supports it; fall back to letting
    # transformers decide rather than forcing full fp32 (which can
    # silently produce degenerate outputs on some GPUs).
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None
