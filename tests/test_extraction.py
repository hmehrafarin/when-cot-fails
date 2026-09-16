import pytest

from ri.utils.extraction import (
    extract_answer,
    extract_answer_from_generation,
    extract_final_answer,
    parse_number,
)

LLAMA = (
    "<|start_header_id|>user<|end_header_id|>\n\nHow many?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "She has 5 * 3 = 15 eggs.\nFinal Answer: 15<|eot_id|>"
)
QWEN = "<|im_start|>user\nHow many?<|im_end|>\n<|im_start|>assistant\n15 + 3 = 18\n18<|im_end|>"


def answer_nums(texts, **kwargs):
    return extract_answer_from_generation(texts, **kwargs)["answer_num"]


def test_parse_number() -> None:
    assert parse_number("She makes $1,018 a day.") == 1018.0
    assert parse_number("costs 2.5 then 3") == 3.0
    assert parse_number("-5") == -5.0
    assert parse_number("no digits") is None
    assert parse_number(None) is None


def test_extract_answer_keeps_gold_string() -> None:
    assert extract_answer(["work\n#### 1,018", "no marker"]) == ["1,018", None]


def test_extract_final_answer() -> None:
    assert extract_final_answer("Final Answer: $1,018.") == "1018"
    assert extract_final_answer("final answer: 2.50") == "2.5"
    assert extract_final_answer("Final Answer: 3\nFinal Answer: 4") == "4"
    assert extract_final_answer("Final Answer: none") is None
    assert extract_final_answer("18") is None


def test_generation_is_trimmed_to_the_assistant_turn() -> None:
    out = extract_answer_from_generation([LLAMA, QWEN, "Question: 5+13?\nAnswer: 18"])
    assert out["answer_text"] == [
        "She has 5 * 3 = 15 eggs.\nFinal Answer: 15",
        "15 + 3 = 18\n18",
        "18",
    ]
    assert out["answer_num"] == ["15", "18", "18"]


def test_flexible_fallback_needs_a_numeric_last_line() -> None:
    prose = "So she has 18 eggs left."
    assert answer_nums([prose]) == [None]
    assert answer_nums([prose], template_name="gsm8k_non_cot") == ["18"]
    assert answer_nums(["5 + 13\n18"]) == ["18"]


def test_strict_mode_needs_a_label() -> None:
    assert answer_nums(["18", "Final Answer: 18"], extraction_mode="strict") == [None, "18"]
    with pytest.raises(ValueError, match="extraction_mode"):
        answer_nums(["18"], extraction_mode="loose")
