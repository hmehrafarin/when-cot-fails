import pytest


class CharTokenizer:
    """One token per character (its code point), with Llama-style special tokens."""

    pad_token_id = 0
    eos_token_id = 3
    all_special_tokens = (
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    )

    def encode(self, text, add_special_tokens=True):
        ids = [ord(ch) for ch in text]
        return [1, *ids] if add_special_tokens else ids

    def convert_tokens_to_ids(self, token):
        return {"<|begin_of_text|>": 1, "<|end_of_text|>": 3, "<|eot_id|>": 4}.get(token, -1)

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        return text + "<assistant>" if add_generation_prompt else text


@pytest.fixture
def tokenizer():
    return CharTokenizer()
