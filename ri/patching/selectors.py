from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from .config import HSSelectionMode, StepsType


def _answer_line_spans(prompt_text: str) -> list[tuple[int, int]]:
    marker = "Answer:"
    idx = prompt_text.find(marker)
    if idx < 0:
        return []

    spans: list[tuple[int, int]] = []
    cursor = idx + len(marker)
    tail = prompt_text[cursor:]
    absolute = cursor

    for segment in tail.splitlines(True):
        line = segment.rstrip("\r\n")
        stripped = line.strip()
        if stripped:
            start = absolute + segment.find(stripped)
            end = start + len(stripped)
            spans.append((start, end))
        absolute += len(segment)
    return spans


def select_step_positions(
    core_positions: Sequence[int],
    prompt_text: str,
    tokenizer,
    steps: StepsType,
) -> list[int]:
    if not core_positions or not prompt_text:
        return []

    try:
        encoding = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except TypeError:
        return list(core_positions)

    offsets = encoding.get("offset_mapping")
    input_ids = encoding.get("input_ids")
    if not offsets or not input_ids:
        return list(core_positions)

    positions = list(core_positions)
    usable = min(len(offsets), len(positions))
    offsets = offsets[:usable]
    positions = positions[:usable]

    spans = _answer_line_spans(prompt_text)
    if not spans:
        return positions

    if steps == "all":
        target_spans = spans
    elif isinstance(steps, int):
        idx = steps - 1
        if idx < 0 or idx >= len(spans):
            return []
        target_spans = [spans[idx]]
    else:
        return []

    def overlaps(t_start: int, t_end: int, span_start: int, span_end: int) -> bool:
        return t_start < span_end and t_end > span_start

    selected: list[int] = []
    for pos, offset in zip(positions, offsets, strict=False):
        tok_start, tok_end = offset
        if tok_start is None or tok_end is None:
            continue
        if not isinstance(tok_start, int):
            tok_start = int(tok_start)
        if not isinstance(tok_end, int):
            tok_end = int(tok_end)
        for span_start, span_end in target_spans:
            if overlaps(tok_start, tok_end, span_start, span_end):
                selected.append(int(pos))
                break
    return selected


def select_positions_with_mode(
    available_positions: Iterable[int],
    k: int,
    mode: HSSelectionMode,
) -> list[int]:
    if k <= 0:
        return []

    unique: list[int] = []
    seen = set()
    for pos in available_positions:
        try:
            ipos = int(pos)
        except (TypeError, ValueError):
            continue
        if ipos in seen:
            continue
        unique.append(ipos)
        seen.add(ipos)

    if not unique:
        return []

    if isinstance(mode, int):
        idx = mode
        length = len(unique)
        if idx < 0:
            idx += length
        idx = max(idx, 0)
        if idx >= length:
            idx = length - 1
        chosen_idx_val = unique[idx]
        return [chosen_idx_val] * k

    if mode == "random_k":
        if len(unique) >= k:
            chosen = sorted(random.sample(unique, k))
        else:
            chosen = unique + [unique[-1]] * (k - len(unique))
    elif mode == "mid":
        start = max((len(unique) - k) // 2, 0)
        chosen = unique[start : start + k]
    elif mode == "late":
        chosen = unique[-k:]
    elif mode == "early":
        chosen = unique[:k]
    else:
        raise ValueError(f"Unsupported HSSelectionMode: {mode!r}")

    if len(chosen) < k:
        chosen = chosen + [chosen[-1]] * (k - len(chosen))
    return chosen
