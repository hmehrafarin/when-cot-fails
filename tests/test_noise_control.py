"""Checks for the Appendix C direction-noise control.

Run with ``pytest`` or directly with ``python tests/test_noise_control.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from ri.patching import noise_control
from ri.patching.tensor_ops import rotate_toward_random_direction
from ri.utils.extraction import score_prediction


def test_rotation_preserves_norm_and_hits_cosine() -> None:
    hidden = torch.randn(3, 1, 64, generator=torch.Generator().manual_seed(0)) * 5
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = rotate_toward_random_direction(hidden, alpha, torch.Generator().manual_seed(1))
        assert out.shape == hidden.shape
        assert out.dtype == hidden.dtype
        assert torch.allclose(out.norm(dim=-1), hidden.norm(dim=-1), rtol=1e-5, atol=1e-5)
        cosine = torch.nn.functional.cosine_similarity(out, hidden, dim=-1)
        assert torch.allclose(cosine, torch.full_like(cosine, alpha), atol=1e-5)


def test_rotation_is_deterministic_and_keeps_dtype() -> None:
    hidden = (torch.randn(2, 16, generator=torch.Generator().manual_seed(0)) * 3).to(torch.bfloat16)
    first = rotate_toward_random_direction(hidden, 0.5, torch.Generator().manual_seed(7))
    second = rotate_toward_random_direction(hidden, 0.5, torch.Generator().manual_seed(7))
    other = rotate_toward_random_direction(hidden, 0.5, torch.Generator().manual_seed(8))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert first.dtype == torch.bfloat16


def test_stable_patch_seed() -> None:
    seed = noise_control.stable_patch_seed(42, 3, 17, -1, 10)
    assert seed == noise_control.stable_patch_seed(42, 3, 17, -1, 10)
    assert seed != noise_control.stable_patch_seed(42, 3, 17, -1, 11)
    assert seed != noise_control.stable_patch_seed(43, 3, 17, -1, 10)
    assert 0 <= seed < 2**63


def test_select_examples() -> None:
    pool = range(20)
    sampled = noise_control.select_examples(pool, 5, seed=1)
    assert sampled == noise_control.select_examples(pool, 5, seed=1)
    assert len(sampled) == 5
    assert noise_control.select_examples(pool, 50, seed=1) == list(range(20))
    assert noise_control.select_examples(pool, 5, seed=1, explicit=[7, 3]) == [3, 7]
    try:
        noise_control.select_examples(pool, 5, seed=1, explicit=[99])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown explicit example must raise")


def test_score_prediction() -> None:
    assert score_prediction("so she makes $1,018 a day", 1018.0).is_correct
    assert score_prediction("17", 18.0).signed_error == -1
    assert not score_prediction("no digits", 18.0).is_numeric


class _FakeRunner:
    """Stands in for PatchRunner: answers 18 unless the patch is rotated below cosine 0.5."""

    def __init__(self, patch_config) -> None:
        self.patch_config = patch_config
        self.calls = 0

    def _run_single_batch(self, sample_idx: int) -> dict[str, list]:
        self.calls += 1
        cosine = self.patch_config.perturb_cosine
        alpha = 1.0 if cosine is None else cosine
        return {
            "answer": ["... #### 18"],
            "Generated Answer_cot": ["18" if alpha >= 0.5 else "12"],
            "patch_from": ["Ġ16"],
        }


def test_run_noise_control_end_to_end(tmp_path: Path | None = None) -> None:
    root = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    csv_path = root / "full_results.csv"
    csv_path.write_text(
        "sample_idx,layer,target_pos,source_pos,patched_token_str,is_correct,generated_text\n"
        "0,1,-1,0,Step,True,18\n"
        "0,1,-1,1,Ġ16,False,12\n"
        "0,3,-1,1,Ġ16,True,18\n"
        "5,1,-1,2,Ġeggs,True,18\n"
        "5,1,0,2,Ġeggs,True,18\n",
        encoding="utf-8",
    )
    out_dir = root / "out"
    runners: list[_FakeRunner] = []

    def fake_make_runner(**kwargs):
        runners.append(_FakeRunner(kwargs["patch_config"]))
        return runners[-1]

    original = noise_control._make_runner
    noise_control._make_runner = fake_make_runner
    try:
        common: dict[str, Any] = {
            "results_csv": str(csv_path),
            "source_model_name": "model",
            "source_dataset": "source.json",
            "target_dataset": "target.json",
            "alphas": [1.0, 0.5, 0.0],
            "seed": 42,
        }
        noise_control.run_noise_control(output_dir=str(out_dir), **common)
        assert runners[-1].calls == 4 * 3  # four successful patches, three alpha levels

        rows = [json.loads(line) for line in (out_dir / "sample_0.jsonl").read_text().splitlines()]
        assert len(rows) == 6
        assert {row["alpha"] for row in rows} == {1.0, 0.5, 0.0}
        assert all(row["is_correct"] for row in rows if row["alpha"] >= 0.5)
        assert not any(row["is_correct"] for row in rows if row["alpha"] == 0.0)
        assert rows[0]["expected_token"] == "Step"

        summary = noise_control.summarize(str(out_dir)).set_index("alpha")
        assert list(summary.index) == [1.0, 0.5, 0.0]
        assert summary.loc[1.0, "n_patches"] == 4
        assert summary.loc[1.0, "success_rate"] == 100.0
        assert summary.loc[0.0, "success_rate"] == 0.0
        assert (out_dir / "summary.csv").exists()

        # Resume: drop the last finished pair and rerun; exactly one generation is redone.
        partial = out_dir / "sample_5.jsonl"
        lines = partial.read_text().splitlines()
        partial.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        noise_control.run_noise_control(output_dir=str(out_dir), resume=True, **common)
        assert runners[-1].calls == 1
        assert len(partial.read_text().splitlines()) == len(lines)

        # Explicit shard plus target-position filter.
        noise_control.run_noise_control(
            output_dir=str(root / "shard"), sample_indices=[5], target_pos=-1, **common
        )
        assert runners[-1].calls == 3
        assert not (root / "shard" / "sample_0.jsonl").exists()
    finally:
        noise_control._make_runner = original


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"{name}: ok")
    print("all checks passed")
