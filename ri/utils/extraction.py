import re
from typing import Any

from .tokenizer import find_special_token, get_end_header_token, get_eot_token

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_number(text: object) -> float | None:
    """
    Parse the last numeric token from ``text``.
    """
    if text is None:
        return None
    matches = NUM_RE.findall(str(text))
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    """
    Convert a float to a stable string form without trailing ``.0``.
    """
    if value.is_integer():
        return str(int(value))
    text = f"{value:.15f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _parse_number_text(text: object) -> str | None:
    value = parse_number(text)
    if value is None:
        return None
    return _format_number(value)


def extract_answer(batch_text: list[str]) -> list[str | None]:
    """
    Extracts the answer from text using the #### pattern (GSM8K format).
    """
    answers = []
    for text in batch_text:
        pattern = r"^\s*####\s+(-?[\d,]+)\s*$"
        matches = re.findall(pattern, text, re.MULTILINE)
        answers.append(matches[-1] if matches else None)
    return answers


def extract_final_answer(text: str) -> str | None:
    """
    Extract the numeric value that follows a 'Final Answer :' label.

    Designed for model outputs that conclude with a line such as:
        Final Answer :  $ 70 , 000

    Returns
    -------
    str | None
        The cleaned number as a string, or ``None`` if no match is found.
    """
    pattern = r"Final\s+Answer\s*:\s*(.+)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    if not matches:
        return None
    return _parse_number_text(matches[-1])


def _extract_last_numeric_token(text: str) -> str | None:
    """
    Return the last numeric token found in *text*.
    """
    return _parse_number_text(text)


def _split_after_marker(text: str, marker: str | None) -> str:
    """Split text after a marker."""
    if not marker:
        return text
    for variant in (f"{marker}\n\n", f"{marker}\n", marker):
        if variant in text:
            parts = text.split(variant)
            return parts[-1]
    return text


def _split_before_marker(text: str, marker: str | None) -> str:
    """Split text before a marker."""
    if not marker:
        return text
    for variant in (marker, f"{marker}\n", f"{marker}\n\n"):
        if variant in text:
            parts = text.split(variant)
            return parts[0]
    return text


_ANSWER_PROMPT_RE = re.compile(r"^\s*answer\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    unique: list[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _candidate_assistant_start_markers(tokenizer: Any | None) -> list[str]:
    """
    Candidate markers that precede assistant content in chat templates.
    """
    markers: list[str] = []

    im_start = None
    if tokenizer is not None:
        im_start = find_special_token(tokenizer, ["im", "start"], default=None)

    if im_start:
        markers.extend(
            [
                f"{im_start}assistant\n\n",
                f"{im_start}assistant\n",
                f"{im_start}assistant",
            ]
        )

    markers.extend(
        [
            "<|im_start|>assistant\n\n",
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
        ]
    )

    return _dedupe_preserve_order(markers)


def _candidate_turn_end_markers(tokenizer: Any | None) -> list[str]:
    """
    Candidate end-of-turn markers across supported chat formats.
    """
    markers: list[str] = []

    if tokenizer is not None:
        eot_token = get_eot_token(tokenizer)
        if eot_token:
            markers.append(eot_token)
        im_end = find_special_token(tokenizer, ["im", "end"], default=None)
        if im_end:
            markers.append(im_end)

    markers.extend(["<|eot_id|>", "<|im_end|>"])
    return _dedupe_preserve_order(markers)


def _split_after_any_marker(text: str, markers: list[str]) -> str:
    """
    Split after the latest matching marker variant in text.
    """
    best_idx = -1
    best_variant = ""
    for marker in markers:
        for variant in (f"{marker}\n\n", f"{marker}\n", marker):
            idx = text.rfind(variant)
            if idx < 0:
                continue
            if idx > best_idx or (idx == best_idx and len(variant) > len(best_variant)):
                best_idx = idx
                best_variant = variant
    if best_idx < 0:
        return text
    return text[best_idx + len(best_variant) :]


def _split_before_any_marker(text: str, markers: list[str]) -> str:
    """
    Split before the earliest matching marker variant in text.
    """
    earliest_idx: int | None = None
    for marker in markers:
        for variant in (marker, f"{marker}\n", f"{marker}\n\n"):
            idx = text.find(variant)
            if idx < 0:
                continue
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx
    if earliest_idx is None:
        return text
    return text[:earliest_idx]


def _extract_number_after_final_answer_label(text: str) -> str | None:
    """
    Extract last numeric token from the clause after the final 'Final Answer:' label.
    """
    matches = list(re.finditer(r"Final\s+Answer\s*:\s*", text, flags=re.IGNORECASE))
    if not matches:
        return None
    tail = text[matches[-1].end() :].strip()
    if not tail:
        return None
    return _parse_number_text(tail)


def _split_after_answer_prompt(text: str) -> str:
    """
    Remove everything up to (and including) the first standalone ``Answer:`` line.
    """
    match = _ANSWER_PROMPT_RE.search(text)
    if not match:
        return text
    return text[match.end() :]


def extract_answer_from_generation(
    batch_text: list[str],
    tokenizer: Any | None = None,
    template_name: str | None = None,
) -> dict[str, list[str | None]]:
    """
    Extracts the answer from model generation output.

    Returns
    -------
    dict
        ``{"answer_text": [...], "answer_num": [...]}``
    """
    answers: dict[str, list[str | None]] = {"answer_text": [], "answer_num": []}
    end_header = get_end_header_token(tokenizer) if tokenizer else None
    if not end_header:
        end_header = "<|end_header_id|>"

    assistant_start_markers = _candidate_assistant_start_markers(tokenizer)
    turn_end_markers = _candidate_turn_end_markers(tokenizer)

    for raw_text in batch_text:
        text_after_header = _split_after_marker(raw_text, end_header)
        prefix_trimmed = text_after_header != raw_text
        text = text_after_header

        if not prefix_trimmed:
            text_after_assistant = _split_after_any_marker(text, assistant_start_markers)
            if text_after_assistant != text:
                prefix_trimmed = True
                text = text_after_assistant

        text = _split_before_any_marker(text, turn_end_markers)

        if not prefix_trimmed:
            stripped = _split_after_answer_prompt(text)
            if stripped != text:
                prefix_trimmed = True
                text = stripped

        text = text.strip()

        matched_number = extract_final_answer(text)
        if not matched_number:
            matched_number = _extract_number_after_final_answer_label(text)
        if not matched_number:
            use_plain_numeric = False
            if template_name and "non_cot" in template_name.lower():
                use_plain_numeric = True
            else:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                if lines:
                    tail = lines[-1]
                    if re.search(r"\d", tail) and not re.search(r"[A-Za-z]", tail):
                        use_plain_numeric = True
            if use_plain_numeric:
                matched_number = _extract_last_numeric_token(text)

        answers["answer_text"].append(text)
        answers["answer_num"].append(matched_number if matched_number else None)

    return answers
