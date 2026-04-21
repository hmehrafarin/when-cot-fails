from collections.abc import Iterable, Sequence
from typing import Any

import torch


def strip_llama_default_metadata(tokenizer: Any) -> None:
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
    cleaned = [line for line in lines if not any(marker in line for marker in markers)]
    tokenizer.chat_template = "".join(cleaned)


def decode_tokens(tokenizer: Any, token_array: Any) -> Any:
    """Decode tokens, handling both 1D and 2D arrays."""
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return tokenizer.decode(token_array)


def get_pad_id(tokenizer: Any) -> int:
    """
    Return the tokenizer pad token id, falling back to 0 when undefined.
    """
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_id, int) and pad_id >= 0:
        return pad_id
    return 0


def _flatten_special_values(values: Any, output: list[str]) -> None:
    """Recursively flatten special token values."""
    if isinstance(values, str):
        output.append(values)
        return
    if isinstance(values, dict):
        for val in values.values():
            _flatten_special_values(val, output)
        return
    if isinstance(values, Iterable):
        for val in values:
            _flatten_special_values(val, output)


def _gather_special_tokens(tokenizer: Any) -> list[str]:
    """Gather all special tokens from the tokenizer."""
    tokens: list[str] = []
    for attr in (
        "special_tokens_map",
        "special_tokens_map_extended",
        "all_special_tokens",
        "additional_special_tokens",
    ):
        values = getattr(tokenizer, attr, None)
        if values:
            _flatten_special_values(values, tokens)
    filtered = [tok for tok in tokens if isinstance(tok, str) and tok]
    seen = set()
    ordered: list[str] = []
    for tok in filtered:
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)
    return ordered


def find_special_token(
    tokenizer: Any,
    keywords: Sequence[str] | str,
    *,
    default: str | None = None,
) -> str | None:
    """
    Best-effort lookup for a special token whose string contains the provided
    keywords (case-insensitive).
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    lowered = [kw.lower() for kw in keywords if kw]
    if not lowered:
        return default
    for tok in _gather_special_tokens(tokenizer):
        low_tok = tok.lower()
        if all(kw in low_tok for kw in lowered):
            return tok
    return default


def get_eot_token(tokenizer: Any) -> str | None:
    """Return the model's end-of-turn token string when available."""
    return find_special_token(tokenizer, ["eot"], default=None)


def get_start_header_token(tokenizer: Any) -> str | None:
    """Return the model's start header token string when available."""
    return find_special_token(tokenizer, ["start", "header"], default=None)


def get_end_header_token(tokenizer: Any) -> str | None:
    """Return the model's end header token string when available."""
    return find_special_token(tokenizer, ["end", "header"], default=None)


def build_role_header(tokenizer: Any, role: str) -> str | None:
    """
    Construct the chat header marker for ``role`` using the tokenizer's
    special tokens.
    """
    start = get_start_header_token(tokenizer)
    end = get_end_header_token(tokenizer)
    if not start or not end:
        return None
    prefix = get_eot_token(tokenizer) or ""
    return f"{prefix}{start}{role}{end}"


def get_eos_token_ids(tokenizer: Any) -> int | list[int]:
    """
    Return the EOS token id(s) to use for generation.
    """
    ids: list[int] = []

    def _append(val: Any) -> None:
        if isinstance(val, int) and val >= 0:
            ids.append(val)
        elif isinstance(val, (list, tuple)):
            for inner in val:
                _append(inner)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if eos_token and callable(convert):
            eos_id = convert(eos_token)
    _append(eos_id)

    eot_token = get_eot_token(tokenizer)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if eot_token and callable(convert):
        eot_id = convert(eot_token)
        _append(eot_id)

    unique = []
    seen = set()
    for val in ids:
        if val not in seen:
            seen.add(val)
            unique.append(val)

    if not unique:
        return get_pad_id(tokenizer)
    if len(unique) == 1:
        return unique[0]
    return unique


def render_prompts(
    tokenizer: Any,
    prompts: Any,
    *,
    system_prompt: bool = False,
    add_generation_prompt: bool = False,
) -> list[str]:
    """
    Render a batch of prompts into the exact text passed to the tokenizer.
    """
    if system_prompt:
        return [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            for convo in prompts
        ]

    rendered: list[str] = []
    for convo in prompts:
        if isinstance(convo, (list, tuple)) and len(convo) > 1 and isinstance(convo[1], dict):
            rendered.append(convo[1].get("content", ""))
        else:
            rendered.append(str(convo))
    return rendered


def make_inputs(
    tokenizer: Any,
    prompts: Any,
    device: str | None = None,
    *,
    system_prompt: bool = False,
    add_generation_prompt: bool = False,
    rendered_prompts: list[str] | None = None,
    max_length: int | None = None,
) -> dict:
    """
    Prepare model inputs (``input_ids`` and ``attention_mask``) for a batch of prompts.

    Returns
    -------
    dict
        ``{"input_ids": tensor, "attention_mask": tensor}`` ready for
        ``model(**batch)``.
    """
    if rendered_prompts is None:
        rendered_prompts = render_prompts(
            tokenizer,
            prompts,
            system_prompt=system_prompt,
            add_generation_prompt=add_generation_prompt,
        )

    add_special_tokens = not system_prompt
    token_lists = [
        tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in rendered_prompts
    ]
    maxlen = max(len(toks) for toks in token_lists)
    if max_length is not None:
        maxlen = max(maxlen, max_length)

    pad_id = get_pad_id(tokenizer)

    input_ids = [[pad_id] * (maxlen - len(toks)) + toks for toks in token_lists]
    attention_mask = [[0] * (maxlen - len(toks)) + [1] * len(toks) for toks in token_lists]

    tensor_kwargs = {}
    if device:
        tensor_kwargs["device"] = device

    return dict(
        input_ids=torch.tensor(input_ids, **tensor_kwargs),
        attention_mask=torch.tensor(attention_mask, **tensor_kwargs),
    )
