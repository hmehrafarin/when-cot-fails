from __future__ import annotations

from collections.abc import Sequence
from typing import Any


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


def add_offsets_to_positions(ti_batch: list[list[dict[str, Any]]], offsets: list[int]) -> None:
    for i, imp_tokens in enumerate(ti_batch):
        off = offsets[i] if i < len(offsets) else 0
        for item in imp_tokens:
            item["pos"] = int(item.get("pos", 0)) + off


def build_word_span_map(
    imp_tokens: list[dict[str, Any]],
    input_ids_row,
    attention_mask_row,
    tokenizer,
) -> dict[int, dict[str, Any]]:
    """Map the starting token index of each importance entry to its token span."""
    if not imp_tokens:
        return {}

    ids_list = input_ids_row.tolist() if hasattr(input_ids_row, "tolist") else list(input_ids_row)
    mask_list = (
        attention_mask_row.tolist()
        if hasattr(attention_mask_row, "tolist")
        else list(attention_mask_row)
    )

    seq_len = len(ids_list)
    valid_len = sum(int(v) for v in mask_list)
    valid_start = seq_len - valid_len
    valid_end = seq_len

    ordered = sorted(imp_tokens, key=lambda item: int(item.get("pos", 0)))
    spans: dict[int, dict[str, Any]] = {}

    for idx, item in enumerate(ordered):
        start = int(item.get("pos", 0))
        if start < valid_start or start >= valid_end:
            continue

        next_start = valid_end
        for follow in ordered[idx + 1 :]:
            candidate = int(follow.get("pos", next_start))
            if candidate > start:
                next_start = candidate
                break

        end = min(next_start, valid_end)
        if end <= start:
            end = min(valid_end, start + 1)

        segment_ids = ids_list[start:end]
        if not segment_ids:
            continue

        decoded = tokenizer.decode(segment_ids, clean_up_tokenization_spaces=False)
        spans[start] = {
            "start": start,
            "end": end,
            "ids": segment_ids,
            "text": decoded,
        }
    return spans


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
