import pytest
import torch

from ri.patching.tensor_ops import (
    compute_core_token_positions,
    left_pad_offsets,
    mask_to_positions,
    rotate_toward_random_direction,
)
from ri.utils.tokenizer import make_inputs


def test_core_token_positions_follow_left_padding(tokenizer) -> None:
    batch = make_inputs(tokenizer, ["Q: 2+3", "Q: 12+3"])
    assert left_pad_offsets(batch) == [1, 0]
    assert mask_to_positions(batch["attention_mask"][0]) == list(range(1, 8))
    positions, starts = compute_core_token_positions(batch, ["2+3", "12+3"], tokenizer)
    assert positions == [[5, 6, 7], [4, 5, 6, 7]]
    assert starts == [5, 4]
    assert compute_core_token_positions(batch, ["", "zzz"], tokenizer) == ([[], []], [1, 0])


def test_rotation_rejects_bad_cosine_and_keeps_zero_vectors() -> None:
    gen = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="cosine"):
        rotate_toward_random_direction(torch.ones(4), 1.5, gen)
    zero = torch.zeros(2, 4)
    assert torch.equal(rotate_toward_random_direction(zero, 0.5, gen), zero)
