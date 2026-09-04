from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def left_pad_offsets(tokenized_batch) -> list[int]:
    """
    Compute per-sample left-padding offsets given a tokenized batch
    (expects keys: 'input_ids', 'attention_mask').
    Offset_i = total_length_i - num_real_tokens_i
    """
    ids_list = tokenized_batch["input_ids"]
    mask_list = tokenized_batch["attention_mask"]
    offsets: list[int] = []
    for ids, mask in zip(ids_list, mask_list, strict=False):
        total = len(ids)
        valid = int(mask.sum().item()) if hasattr(mask, "sum") else int(sum(mask))
        offsets.append(total - valid)
    return offsets


def mask_to_positions(mask_row) -> list[int]:
    """Return absolute indices where the attention mask denotes real tokens."""
    mask_list = mask_row.tolist() if hasattr(mask_row, "tolist") else list(mask_row)
    positions: list[int] = []
    for idx, value in enumerate(mask_list):
        try:
            if int(value):
                positions.append(idx)
        except Exception:
            continue
    return positions


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Return the start index of *needle* inside *haystack*, or -1 when absent."""
    if not needle:
        return -1
    limit = len(haystack) - len(needle) + 1
    if limit <= 0:
        return -1
    for start in range(limit):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return -1


def compute_core_token_positions(
    tokenized_batch,
    core_texts: Sequence[str],
    tokenizer,
) -> tuple[list[list[int]], list[int]]:
    """
    Derive absolute token indices for the question+answer portion of each prompt.

    Parameters
    ----------
    tokenized_batch : dict
        Output of ``make_inputs`` containing ``input_ids`` and ``attention_mask``.
    core_texts : Sequence[str]
        Text for the prompt body (question + answer) per batch element.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer used to render inputs; only ``encode`` is required.

    Returns
    -------
    tuple(List[List[int]], List[int])
        A pair ``(positions, offsets)`` where ``positions[i]`` is the list of
        absolute token indices covering the core prompt for sample *i*, and
        ``offsets[i]`` is the absolute index of the first core token. Empty
        lists are returned when the core span cannot be located.
    """
    input_rows = tokenized_batch["input_ids"]
    mask_rows = tokenized_batch["attention_mask"]

    all_positions: list[list[int]] = []
    start_offsets: list[int] = []

    for idx, (ids_row, mask_row) in enumerate(zip(input_rows, mask_rows, strict=False)):
        ids_list = ids_row.tolist() if hasattr(ids_row, "tolist") else list(ids_row)
        valid_len = int(mask_row.sum().item()) if hasattr(mask_row, "sum") else int(sum(mask_row))
        total_len = len(ids_list)
        left_pad = total_len - valid_len
        valid_ids = ids_list[left_pad : left_pad + valid_len]

        core_text = ""
        if idx < len(core_texts) and isinstance(core_texts[idx], str):
            core_text = core_texts[idx]
        core_ids = tokenizer.encode(core_text, add_special_tokens=False) if core_text else []

        if not core_ids:
            all_positions.append([])
            start_offsets.append(left_pad)
            continue

        start_idx = _find_subsequence(valid_ids, core_ids)
        if start_idx < 0:
            all_positions.append([])
            start_offsets.append(left_pad)
            continue

        absolute_start = left_pad + start_idx
        span_positions = [absolute_start + pos for pos in range(len(core_ids))]

        all_positions.append(span_positions)
        start_offsets.append(absolute_start)

    return all_positions, start_offsets


def rotate_toward_random_direction(
    hidden: torch.Tensor,
    cosine: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a norm-matched vector at cosine similarity ``cosine`` to ``hidden`` (Appendix C).

    Computes ``h~ = ||h|| (a h^ + sqrt(1 - a^2) u^)`` along the last dimension, where ``u^`` is a
    random unit direction orthogonal to ``h`` drawn from ``generator`` (a CPU generator, so the
    draw is reproducible across devices). ``cosine=1`` returns ``h`` and ``cosine=0`` a fully
    random direction with the original magnitude. Leading dimensions are treated as batch
    dimensions; the result has the dtype of ``hidden``.
    """
    if not 0.0 <= cosine <= 1.0:
        raise ValueError(f"cosine must be in [0, 1], got {cosine!r}")

    h = hidden.detach().to(torch.float32)
    norm = h.norm(dim=-1, keepdim=True)
    unit = h / norm.clamp_min(1e-12)

    noise = torch.randn(h.shape, generator=generator, dtype=torch.float32).to(h.device)
    noise = noise - (noise * unit).sum(dim=-1, keepdim=True) * unit
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    rotated = norm * (cosine * unit + math.sqrt(1.0 - cosine * cosine) * noise)
    return rotated.to(hidden.dtype)
