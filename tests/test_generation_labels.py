import pytest

from ri.postprocess.generation_labels import (
    EQUATION_ONLY,
    FINAL_ONLY,
    FULL_COT,
    GENERATION_TYPE_CODES,
    GENERATION_TYPES,
    NOISE,
    NONE,
    PARTIAL_COT,
    TEXT_ONLY,
    classify_generation_type,
)

CASES = [
    (None, NONE),
    ("  \n", NONE),
    ("18", FINAL_ONLY),
    ("Final Answer: $1,018", FINAL_ONLY),
    ("She has 18 eggs left", FINAL_ONLY),
    ("= 18", FINAL_ONLY),
    ("I cannot answer that.", TEXT_ONLY),
    ("5 * 3 = 15", EQUATION_ONLY),
    ("5 * 3 = 15\n15 + 3 = 18", EQUATION_ONLY),
    ("15 plus 3 gives 18.", PARTIAL_COT),
    ("Step 1: 5 * 3 = 15\nStep 2: 15 + 3 = 18\nFinal Answer: 18", FULL_COT),
    ('""""""""""""""""', NOISE),
    ("18 18 18 18 18 18 18 18", NOISE),
]


@pytest.mark.parametrize(("text", "label"), CASES)
def test_classify_generation_type(text: str | None, label: str) -> None:
    assert classify_generation_type(text) == label


def test_codebook_matches_types() -> None:
    assert list(GENERATION_TYPE_CODES) == GENERATION_TYPES


PUBLISHED_MISLABELS = [
    ("2.5", FINAL_ONLY),
    ("18.", FINAL_ONLY),
    ("3 extra boxes", FINAL_ONLY),
]


@pytest.mark.xfail(
    strict=True,
    reason="as published: a leading decimal or 'N.' reads as a step marker, an x inside a word as an operator",
)
@pytest.mark.parametrize(("text", "label"), PUBLISHED_MISLABELS)
def test_published_mislabels(text: str, label: str) -> None:
    assert classify_generation_type(text) == label
