from types import SimpleNamespace

from ri.utils.tokenizer import (
    find_special_token,
    get_end_header_token,
    get_eos_token_ids,
    get_eot_token,
    make_inputs,
    render_prompts,
)


def test_find_special_token(tokenizer) -> None:
    assert get_eot_token(tokenizer) == "<|eot_id|>"
    assert get_end_header_token(tokenizer) == "<|end_header_id|>"
    assert find_special_token(tokenizer, ["im", "start"]) is None
    qwen = SimpleNamespace(all_special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    assert find_special_token(qwen, ["IM", "start"]) == "<|im_start|>"
    assert get_eot_token(qwen) is None


def test_get_eos_token_ids(tokenizer) -> None:
    assert get_eos_token_ids(tokenizer) == [3, 4]
    assert get_eos_token_ids(SimpleNamespace(eos_token_id=[3, [3, 7]])) == [3, 7]
    assert get_eos_token_ids(SimpleNamespace(pad_token_id=9)) == 9


def test_render_prompts(tokenizer) -> None:
    convo = [{"role": "system", "content": "sys"}, {"role": "user", "content": "How many?"}]
    assert render_prompts(tokenizer, [convo, "plain"]) == ["How many?", "plain"]
    assert render_prompts(tokenizer, [convo], system_prompt=True, add_generation_prompt=True) == [
        "<system>sys<user>How many?<assistant>"
    ]


def test_make_inputs_left_pads(tokenizer) -> None:
    batch = make_inputs(tokenizer, ["abc", "a"])
    assert batch["input_ids"].tolist() == [[1, 97, 98, 99], [0, 0, 1, 97]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    chat = make_inputs(tokenizer, None, system_prompt=True, rendered_prompts=["abc"])
    assert chat["input_ids"].tolist() == [[97, 98, 99]]
