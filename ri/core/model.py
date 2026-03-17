import re
from typing import Any, Optional

import torch
import transformers

from ri.config import settings


class ModelAndTokenizer:
    """An object to hold a GPT-style language model and tokenizer."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        low_cpu_mem_usage: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        require_grad: bool = False,
        use_fast: bool = True,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        if cache_dir is None:
            cache_dir = settings.MODEL_CACHE_DIR

        if tokenizer is None:
            assert model_name is not None
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, use_fast=use_fast, cache_dir=cache_dir
            )
        if model is None:
            assert model_name is not None
            kwargs = dict(
                cache_dir=cache_dir,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map="auto",
                dtype=torch_dtype,
                **kwargs,
            )
            set_requires_grad(require_grad, model)
            model.eval()

        self.tokenizer = tokenizer
        _strip_llama_default_metadata(self.tokenizer)
        self.model = model
        self.device_map = getattr(model, "hf_device_map", None)
        self.device = device
        self.model_name = self._derive_model_name(model_name, model)
        self.is_instruct_model = self._detect_instruct(self.model_name)
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if (re.match(r"^(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self) -> str:
        """String representation of this class."""
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )

    @staticmethod
    def _derive_model_name(model_name: Optional[str], model: Any) -> Optional[str]:
        if model_name:
            return model_name
        if model is not None:
            direct = getattr(model, "name_or_path", None)
            if direct:
                return direct
            config = getattr(model, "config", None)
            if config is not None:
                cfg_name = getattr(config, "_name_or_path", None)
                if cfg_name:
                    return cfg_name
        return None

    @staticmethod
    def _detect_instruct(name: Optional[str]) -> bool:
        if not name:
            return False
        return "instruct" in name.lower()


def set_requires_grad(requires_grad: bool, *models: Any) -> None:
    """Set requires_grad on all parameters of the given models."""
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad


def _strip_llama_default_metadata(tokenizer: Any) -> None:
    """
    Drop the banner that injects the Llama knowledge/date metadata.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return

    markers = ("Cutting Knowledge Date:", "Today Date:")
    if not any(marker in template for marker in markers):
        return

    lines = template.splitlines(keepends=True)
    cleaned = [line for line in lines if not any(
        marker in line for marker in markers)]
    tokenizer.chat_template = "".join(cleaned)
