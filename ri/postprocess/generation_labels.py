from __future__ import annotations

import re

# Post-patch generation types, named after the output-type taxonomy in the paper (Table 6).
FULL_COT = "full_cot"
EQUATION_ONLY = "equation_only"
PARTIAL_COT = "partial_cot"
FINAL_ONLY = "final_only"
TEXT_ONLY = "text_only"
NOISE = "noise"
NONE = "none"

GENERATION_TYPES = [FULL_COT, EQUATION_ONLY, PARTIAL_COT, FINAL_ONLY, TEXT_ONLY, NOISE, NONE]

GENERATION_TYPE_CODES: dict[str, str] = {
    FULL_COT: "Full CoT: complete multi-step natural-language reasoning.",
    EQUATION_ONLY: "Equation-Only: predominantly symbolic or arithmetic expressions with minimal prose.",
    PARTIAL_COT: "Partial CoT: abbreviated one-step reasoning.",
    FINAL_ONLY: "Final Only: direct answer, typically a single number and a few words.",
    TEXT_ONLY: "Text Only: non-answer prose without a valid final numeric answer.",
    NOISE: "Noise: symbol-like fragments or token repetition with no usable answer.",
    NONE: "None: empty generation.",
}

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+")
STEP_MARKER_RE = re.compile(r"(?im)^\s*(?:step\s*\d+|\d+\.)")
ANSWER_PREFIX_RE = re.compile(r"(?is)^\s*(?:final\s+answer|answer)\s*[:\-]?\s*")
REPEATED_SYMBOL_RE = re.compile(r"([\"'`.\-_])\1{9,}")
LEADING_PUNCT = set(":;,.()[]{}$")
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
WORD_CHAR_RE = re.compile(r"[A-Za-z]")


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


def _relabel_noise(text: str) -> str:
    """A bare number hidden behind punctuation is a final answer, not noise."""
    normalized = _normalize_generated_text(text)
    if (
        normalized
        and not WORD_CHAR_RE.search(normalized)
        and NUMERIC_TOKEN_RE.fullmatch(normalized)
    ):
        return FINAL_ONLY
    return NOISE


def classify_generation_type(text: object) -> str:
    """Assign one of ``GENERATION_TYPES`` to a post-patch generation."""
    stripped = "" if text is None else str(text).strip()
    if not stripped:
        return NONE

    if _is_strict_final_answer(stripped):
        return FINAL_ONLY

    has_number = bool(NUM_RE.search(stripped))
    has_alpha = bool(re.search(r"[A-Za-z]", stripped))
    if has_alpha and not has_number:
        return TEXT_ONLY
    if _is_repetitive_or_empty(stripped):
        return _relabel_noise(stripped)

    word_count = len(WORD_RE.findall(stripped))
    line_count = stripped.count("\n") + 1
    eq_like = stripped.count("=")
    has_step_marker = bool(STEP_MARKER_RE.search(stripped))
    has_ops = bool(re.search(r"[+\-*/xX\u00d7]", stripped))

    label = NOISE
    if has_number:
        if (
            has_step_marker
            or (word_count >= 18 and line_count >= 2)
            or word_count >= 24
            or (eq_like >= 2 and word_count >= 12 and line_count >= 2)
        ):
            label = FULL_COT
        elif (
            (eq_like >= 1 and has_ops and word_count <= 14 and line_count >= 2)
            or (eq_like >= 1 and has_ops and line_count == 1 and word_count <= 6)
            or (eq_like >= 2 and has_ops and word_count <= 20)
        ):
            label = EQUATION_ONLY

    num_count = len(NUM_RE.findall(stripped))
    if (
        label == NOISE
        and has_alpha
        and num_count == 1
        and word_count < 10
        and eq_like == 0
        and not has_ops
    ):
        label = FINAL_ONLY

    if label == NOISE and has_number and has_alpha:
        return PARTIAL_COT

    if label == NOISE:
        return _relabel_noise(stripped)
    return label
