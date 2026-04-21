from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Language = object
    Doc = object

PRIORITY = {
    "OTHER": 0,
    "PUNCT": 1,
    "VERB": 2,
    "ENTITY": 2,
    "NUMBER": 3,
    "QUANTITY": 4,
    "UNIT": 5,
    "STEP": 7,
    "OPERAND": 8,
    "RESULT": 9,
    "OPERATOR": 10,
    "EQUALS": 10,
}

VALID_LABELS = frozenset(PRIORITY)
MATH_OPERATORS = {"-", "+", "\u00d7", "x", "*", "/"}
CURRENCY_LEMMAS = {"dollar", "usd", "pound", "gbp", "euro", "eur", "cent"}
WORD_OPERATORS = frozenset(
    {
        "plus",
        "minus",
        "times",
        "add",
        "added",
        "subtract",
        "subtracted",
        "multiply",
        "multiplied",
        "divide",
        "divided",
    }
)
UNIT_ABBREV_TO_LEMMA = {
    "km": "kilometer",
    "cm": "centimeter",
    "mm": "millimeter",
    "mi": "mile",
    "ft": "foot",
    "h": "hour",
    "hr": "hour",
    "min": "minute",
    "kg": "kilogram",
    "g": "gram",
    "lb": "pound",
    "oz": "ounce",
    "ml": "milliliter",
    "mph": "mile",
    "kph": "kilometer",
    "kmh": "kilometer",
}
MEASURE_UNIT_LEMMAS = {
    "inch",
    "foot",
    "yard",
    "mile",
    "centimeter",
    "meter",
    "kilometer",
    "ounce",
    "pound",
    "gram",
    "kilogram",
    "second",
    "minute",
    "hour",
    "cup",
    "pint",
    "quart",
    "gallon",
    "liter",
    "litre",
    "milliliter",
} | set(UNIT_ABBREV_TO_LEMMA)
COMPOUND_UNIT_RE = re.compile(r"^[A-Za-z]+/[A-Za-z]+$")
NUMMOD_UNIT_BLOCKLIST = frozenset(
    {
        "answer",
        "step",
        "value",
        "increase",
        "decrease",
        "cost",
        "price",
        "total",
        "rate",
        "amount",
        "number",
        "result",
        "sum",
        "difference",
        "percentage",
    }
)
THIRD_PERSON_PRONOUNS = frozenset(
    {
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
    }
)
THIRD_PERSON_PRONOUNS_SUBJ_ONLY = frozenset({"it", "its", "itself"})


@dataclass(frozen=True)
class TagResult:
    doc: Doc
    tags: dict[int, str]
    reverse: dict[str, list[str]]
    orig_text: str
    char_map: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QuestionContext:
    entity_lemmas: frozenset[str]
    entity_texts: frozenset[str]


@dataclass(frozen=True)
class QuestionTagResult:
    result: TagResult
    context: QuestionContext


@dataclass(frozen=True)
class LabeledSpan:
    start: int
    end: int
    label: str


def _require_spacy() -> None:
    if spacy is None:
        raise RuntimeError(
            "spaCy is required for postprocessing. Install it with `uv sync --extra analysis`."
        )


@lru_cache(maxsize=4)
def get_nlp(model_name: str = "en_core_web_sm") -> Language:
    _require_spacy()
    nlp = spacy.load(model_name)
    infixes = list(nlp.Defaults.infixes)
    digit_fraction = r"(?<=[0-9])/(?=[0-9])"
    if digit_fraction not in infixes:
        infixes.append(digit_fraction)
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer
    return nlp


def _make_setter(tags: dict[int, str]):
    def set_tag(index: int, label: str) -> None:
        if PRIORITY[label] > PRIORITY[tags[index]]:
            tags[index] = label

    return set_tag


def _identity_char_map(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((index, 1) for index in range(len(text)))


def _replace_latex_times(
    text: str,
    char_map: Sequence[tuple[int, int]],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    out_chars: list[str] = []
    out_map: list[tuple[int, int]] = []
    index = 0

    while index < len(text):
        if text.startswith("\\times", index):
            orig_start = char_map[index][0]
            last_start, last_span = char_map[index + len("\\times") - 1]
            out_chars.append("\u00d7")
            out_map.append((orig_start, last_start + last_span - orig_start))
            index += len("\\times")
            continue

        out_chars.append(text[index])
        out_map.append(char_map[index])
        index += 1

    return "".join(out_chars), tuple(out_map)


def _normalize_slashes(
    text: str,
    char_map: Sequence[tuple[int, int]],
) -> tuple[str, tuple[tuple[int, int], ...]]:
    normalized: list[str] = []
    norm_map: list[tuple[int, int]] = []
    index = 0

    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text) and text[index + 1] == "/":
            orig_start = char_map[index][0]
            last_start, last_span = char_map[index + 1]
            normalized.append("/")
            norm_map.append((orig_start, last_start + last_span - orig_start))
            index += 2
            continue

        normalized.append(text[index])
        norm_map.append(char_map[index])
        index += 1

    return "".join(normalized), tuple(norm_map)


def _prepare_text(text: str) -> tuple[str, str, tuple[tuple[int, int], ...]]:
    orig_text = text
    char_map = _identity_char_map(orig_text)
    preprocessed, char_map = _replace_latex_times(orig_text, char_map)
    normalized, char_map = _normalize_slashes(preprocessed, char_map)
    return normalized, orig_text, char_map


def _raw_span(
    start: int,
    end: int,
    char_map: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    if start >= end:
        orig_start = char_map[start][0] if start < len(char_map) else 0
        return orig_start, orig_start

    orig_start, _ = char_map[start]
    last_start, last_width = char_map[end - 1]
    return orig_start, last_start + last_width


def _get_orig_token_text(token, orig_text: str | None, char_map: Sequence[tuple[int, int]] | None) -> str:
    if orig_text is None or char_map is None:
        return token.text
    start, end = _raw_span(token.idx, token.idx + len(token.text), char_map)
    return orig_text[start:end]


def _reverse_lookup(
    doc: Doc,
    tags: dict[int, str],
    orig_text: str,
    char_map: Sequence[tuple[int, int]],
) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for token_idx, label in tags.items():
        reverse[label].append(_get_orig_token_text(doc[token_idx], orig_text, char_map))
    return dict(reverse)


def _nearest_num_idx(doc: Doc, pos: int, direction: int) -> int | None:
    index = pos + direction
    while 0 <= index < len(doc):
        token = doc[index]
        if token.pos_ == "NUM" or token.like_num:
            return index
        if token.tag_ == "$" or token.is_space or token.text in "()":
            index += direction
        else:
            break
    return None


def _is_line_start(doc: Doc, token) -> bool:
    if token.i == 0:
        return True
    prev = doc[token.i - 1]
    return "\n" in doc.text[prev.idx : token.idx]


def _is_compound_unit_slash(doc: Doc, index: int) -> bool:
    if index <= 0 or index >= len(doc) - 1:
        return False

    left = doc[index - 1]
    right = doc[index + 1]
    right_txt = right.text.lower()
    if not right.like_num and (
        right.pos_ in ("NOUN", "PROPN")
        or right_txt in UNIT_ABBREV_TO_LEMMA
        or right.lemma_.lower() in MEASURE_UNIT_LEMMAS
    ):
        return True

    left_is_unit = (
        left.lemma_.lower() in MEASURE_UNIT_LEMMAS or left.text.lower() in UNIT_ABBREV_TO_LEMMA
    )
    return bool(left_is_unit and right.is_alpha and not right.like_num)


def _base_num_unit_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if token.tag_ == "$" or token.lemma_.lower() in CURRENCY_LEMMAS | MEASURE_UNIT_LEMMAS:
            set_tag(token.i, "UNIT")
        elif token.pos_ == "NUM" or token.like_num:
            set_tag(token.i, "NUMBER")

    for token in doc:
        if token.lower_ in ("per", "every"):
            end = token.i + 1
            while end < len(doc) and doc[end].pos_ in ("ADJ", "NOUN", "PROPN", "NUM"):
                end += 1
            for related in doc[token.i:end]:
                set_tag(related.i, "UNIT")


def _compound_unit_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if COMPOUND_UNIT_RE.match(token.text):
            set_tag(token.i, "UNIT")
            continue
        if token.text == "/" and _is_compound_unit_slash(doc, token.i):
            set_tag(doc[token.i - 1].i, "UNIT")
            set_tag(token.i, "UNIT")
            set_tag(doc[token.i + 1].i, "UNIT")


def _quantity_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if (
            token.dep_ == "nummod"
            and token.head.pos_ in ("NOUN", "PROPN")
            and token.head.lemma_.lower() not in NUMMOD_UNIT_BLOCKLIST
        ):
            set_tag(token.i, "QUANTITY")
            set_tag(token.head.i, "UNIT")


def _word_operator_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if token.lower_ in WORD_OPERATORS:
            left_idx = _nearest_num_idx(doc, token.i, -1)
            right_idx = _nearest_num_idx(doc, token.i, 1)
            if left_idx is not None or right_idx is not None:
                set_tag(token.i, "OPERATOR")
                if left_idx is not None:
                    set_tag(left_idx, "OPERAND")
                if right_idx is not None:
                    set_tag(right_idx, "OPERAND")


def _math_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if token.text == "=":
            set_tag(token.i, "EQUALS")
            left_idx = _nearest_num_idx(doc, token.i, -1)
            right_idx = _nearest_num_idx(doc, token.i, 1)
            if left_idx is not None:
                set_tag(left_idx, "OPERAND")
            if right_idx is not None:
                set_tag(right_idx, "RESULT")
        elif token.text in MATH_OPERATORS:
            if token.text == "/" and _is_compound_unit_slash(doc, token.i):
                continue
            set_tag(token.i, "OPERATOR")
            left_idx = _nearest_num_idx(doc, token.i, -1)
            right_idx = _nearest_num_idx(doc, token.i, 1)
            if left_idx is not None:
                set_tag(left_idx, "OPERAND")
            if right_idx is not None:
                set_tag(right_idx, "OPERAND")


def _step_marker_pass(doc: Doc, set_tag) -> None:
    index = 0
    while index < len(doc):
        token = doc[index]
        if token.lower_ == "step" and index + 1 < len(doc):
            nxt = doc[index + 1]
            if nxt.like_num or nxt.pos_ == "NUM":
                set_tag(token.i, "STEP")
                set_tag(nxt.i, "STEP")
                if index + 2 < len(doc) and doc[index + 2].text in (":", "."):
                    set_tag(doc[index + 2].i, "STEP")
                index += 3
                continue
        if (
            (token.like_num or token.pos_ == "NUM")
            and _is_line_start(doc, token)
            and index + 1 < len(doc)
            and doc[index + 1].text in (".", ":")
        ):
            set_tag(token.i, "STEP")
            set_tag(doc[index + 1].i, "STEP")
            index += 2
            continue
        index += 1


def _pronoun_pass(doc: Doc, set_tag) -> None:
    for token in doc:
        if token.pos_ != "PRON":
            continue
        low = token.lower_
        if low in THIRD_PERSON_PRONOUNS or (
            low in THIRD_PERSON_PRONOUNS_SUBJ_ONLY and token.dep_ in ("nsubj", "nsubjpass")
        ):
            set_tag(token.i, "ENTITY")


def _extract_entities(doc: Doc) -> tuple[frozenset[str], frozenset[str]]:
    lemmas: set[str] = set()
    texts: set[str] = set()
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
            lemmas.add(token.lemma_.lower())
            texts.add(token.text.lower())
    return frozenset(lemmas), frozenset(texts)


def _seed_context(
    doc: Doc,
    set_tag,
    entity_lemmas: Sequence[str],
    entity_texts: Sequence[str],
) -> None:
    entity_lemma_set = set(entity_lemmas)
    entity_text_set = set(entity_texts)
    for token in doc:
        lemma = token.lemma_.lower()
        text = token.text.lower()
        if lemma in entity_lemma_set or text in entity_text_set:
            set_tag(token.i, "ENTITY")


def tag_doc(question: str, nlp: Language | None = None) -> QuestionTagResult:
    norm_text, orig_text, char_map = _prepare_text(question)
    doc = (nlp or get_nlp())(norm_text)
    tags = {token.i: "OTHER" for token in doc}
    set_tag = _make_setter(tags)

    entity_lemmas, entity_texts = _extract_entities(doc)
    _seed_context(doc, set_tag, entity_lemmas, entity_texts)
    _pronoun_pass(doc, set_tag)

    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            set_tag(token.i, "VERB")
        if token.is_punct:
            set_tag(token.i, "PUNCT")

    _base_num_unit_pass(doc, set_tag)
    _compound_unit_pass(doc, set_tag)
    _quantity_pass(doc, set_tag)

    result = TagResult(
        doc=doc,
        tags=tags,
        reverse=_reverse_lookup(doc, tags, orig_text, char_map),
        orig_text=orig_text,
        char_map=char_map,
    )
    context = QuestionContext(entity_lemmas=entity_lemmas, entity_texts=entity_texts)
    return QuestionTagResult(result=result, context=context)


def build_question_context(question: str, nlp: Language | None = None) -> QuestionContext:
    return tag_doc(question=question, nlp=nlp).context


def tag_reasoning(reasoning: str, context: QuestionContext, nlp: Language | None = None) -> TagResult:
    norm_text, orig_text, char_map = _prepare_text(reasoning)
    doc = (nlp or get_nlp())(norm_text)
    tags = {token.i: "OTHER" for token in doc}
    set_tag = _make_setter(tags)

    _seed_context(doc, set_tag, context.entity_lemmas, context.entity_texts)
    _pronoun_pass(doc, set_tag)

    for token in doc:
        if token.pos_ in ("VERB", "AUX"):
            set_tag(token.i, "VERB")

    _base_num_unit_pass(doc, set_tag)
    _compound_unit_pass(doc, set_tag)
    _step_marker_pass(doc, set_tag)
    _quantity_pass(doc, set_tag)
    _word_operator_pass(doc, set_tag)
    _math_pass(doc, set_tag)

    for token in doc:
        if token.is_punct and token.text not in MATH_OPERATORS and token.text != "=":
            set_tag(token.i, "PUNCT")

    return TagResult(
        doc=doc,
        tags=tags,
        reverse=_reverse_lookup(doc, tags, orig_text, char_map),
        orig_text=orig_text,
        char_map=char_map,
    )


def labeled_token_char_spans(
    doc: Doc,
    tags: dict[int, str],
    char_map: Sequence[tuple[int, int]] | None = None,
) -> list[LabeledSpan]:
    spans: list[LabeledSpan] = []
    for token in doc:
        label = tags[token.i]
        if label not in VALID_LABELS:
            raise ValueError(f"Unknown label for token {token.i}: {label}")
        if char_map is None:
            start = token.idx
            end = token.idx + len(token.text)
        else:
            start, end = _raw_span(token.idx, token.idx + len(token.text), char_map)
        spans.append(LabeledSpan(start=start, end=end, label=label))
    return spans


def label_reasoning_with_question(
    question: str,
    reasoning: str,
    nlp: Language | None = None,
) -> tuple[QuestionContext, TagResult, list[LabeledSpan]]:
    language = nlp or get_nlp()
    context = build_question_context(question=question, nlp=language)
    reasoning_result = tag_reasoning(reasoning=reasoning, context=context, nlp=language)
    spans = labeled_token_char_spans(
        reasoning_result.doc,
        reasoning_result.tags,
        char_map=reasoning_result.char_map,
    )
    return context, reasoning_result, spans
