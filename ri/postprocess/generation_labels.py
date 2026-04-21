from __future__ import annotations

import re

BASE_GENERATION_ORDER = [
    "full_cot",
    "semi_cot",
    "partial_cot",
    "final_only",
    "text_only",
    "none",
]

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+")
STEP_MARKER_RE = re.compile(r"(?im)^\s*(?:step\s*\d+|\d+\.)")
ANSWER_PREFIX_RE = re.compile(r"(?is)^\s*(?:final\s+answer|answer)\s*[:\-]?\s*")
ANSWER_PHRASE_RE = re.compile(r"(?i)\b(final\s+answer|answer\s+is|correct\s+answer|final\s+result)\b")
REPEATED_SYMBOL_RE = re.compile(r"([\"'`.\-_])\1{9,}")
LEADING_PUNCT = set(":;,.()[]{}$")
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
WORD_CHAR_RE = re.compile(r"[A-Za-z]")


def generation_order(other_label: str = "noise") -> list[str]:
    return [*BASE_GENERATION_ORDER, other_label]


def generation_type_codes(other_label: str = "noise") -> dict[str, str]:
    return {
        "full_cot": "A multi-step chain-of-thought reasoning trace.",
        "semi_cot": "A concise equation-focused reasoning trace.",
        "partial_cot": "A partial or incomplete reasoning trace with mixed text and numbers.",
        "final_only": "A short final-answer style response without a clear reasoning trace.",
        "text_only": "Natural-language text with no numeric answer content.",
        "none": "No usable generation text was produced.",
        other_label: "Residual malformed output." if other_label == "other" else "Residual malformed output relabeled as noise.",
    }


def _is_repetitive_or_empty(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if REPEATED_SYMBOL_RE.search(stripped):
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True

    alpha_tokens = [word.lower() for word in WORD_RE.findall(stripped)]
    if alpha_tokens:
        unique = set(alpha_tokens)
        if len(alpha_tokens) >= 4 and len(unique) == 1:
            return True
        top_freq = max(alpha_tokens.count(token) for token in unique)
        if len(alpha_tokens) >= 8 and (top_freq / len(alpha_tokens)) >= 0.8:
            return True

    alnum = sum(ch.isalnum() for ch in stripped)
    return len(stripped) >= 40 and (alnum / len(stripped)) < 0.15


def _is_strict_final_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped or _is_repetitive_or_empty(stripped):
        return False

    stripped = ANSWER_PREFIX_RE.sub("", stripped, count=1).strip()
    if not stripped:
        return False

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    stripped = lines[0].strip().strip('"').strip("'").strip()
    if not stripped or STEP_MARKER_RE.search(stripped):
        return False
    if "=" in stripped or re.search(r"[+\*/\u00d7xX]", stripped):
        return False

    nums = NUM_RE.findall(stripped)
    if len(nums) != 1:
        return False

    remainder = re.sub(r"\$?\s*[-+]?\d[\d,]*(?:\.\d+)?\s*%?", " ", stripped)
    words = WORD_RE.findall(remainder)
    if len(words) > 3:
        return False

    banned = {
        "let",
        "lets",
        "step",
        "steps",
        "problem",
        "solve",
        "calculate",
        "assistant",
        "instruction",
    }
    if any(word.lower() in banned for word in words):
        return False

    residual = re.sub(r"[A-Za-z\s]", "", remainder)
    return len(residual) <= 8


def _normalize_generated_text(text: str) -> str:
    stripped = re.sub(r"[\t\r\n]+", " ", text.strip())
    stripped = re.sub(r" +", " ", stripped)
    stripped = stripped.lstrip()
    while stripped and stripped[0] in LEADING_PUNCT:
        stripped = stripped[1:].lstrip()
    if stripped.startswith("="):
        stripped = stripped[1:].lstrip()
    return stripped.strip()


def _relabel_other(text: str, other_label: str) -> str:
    normalized = _normalize_generated_text(text)
    if normalized and not WORD_CHAR_RE.search(normalized) and NUMERIC_TOKEN_RE.fullmatch(normalized):
        return "final_only"
    return other_label


def classify_generation_type(text: object, other_label: str = "noise") -> str:
    stripped = "" if text is None else str(text).strip()
    if not stripped:
        return "none"

    label = other_label

    if _is_strict_final_answer(stripped):
        return "final_only"

    has_number = bool(NUM_RE.search(stripped))
    has_alpha = bool(re.search(r"[A-Za-z]", stripped))
    if has_alpha and not has_number:
        return "text_only"
    if _is_repetitive_or_empty(stripped):
        return _relabel_other(stripped, other_label)

    word_count = len(WORD_RE.findall(stripped))
    line_count = stripped.count("\n") + 1
    eq_like = stripped.count("=")
    has_step_marker = bool(STEP_MARKER_RE.search(stripped))
    has_ops = bool(re.search(r"[+\-*/xX\u00d7]", stripped))

    if has_number:
        if (
            has_step_marker
            or (word_count >= 18 and line_count >= 2)
            or word_count >= 24
            or (eq_like >= 2 and word_count >= 12 and line_count >= 2)
        ):
            label = "full_cot"
        elif (
            (eq_like >= 1 and has_ops and word_count <= 14 and line_count >= 2)
            or (eq_like >= 1 and has_ops and line_count == 1 and word_count <= 6)
            or (eq_like >= 2 and has_ops and word_count <= 20)
        ):
            label = "semi_cot"

    num_count = len(NUM_RE.findall(stripped))
    if label == other_label and has_alpha and num_count == 1 and word_count < 10 and eq_like == 0 and not has_ops:
        label = "final_only"

    if label == other_label and has_number and has_alpha:
        label = "partial_cot"
        if ANSWER_PHRASE_RE.search(stripped):
            return label
        if eq_like >= 1 and has_ops:
            return label
        if eq_like >= 1 or has_ops:
            return label
        return label

    if label == other_label:
        return _relabel_other(stripped, other_label)
    return label
