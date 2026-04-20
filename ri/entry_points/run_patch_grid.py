#!/usr/bin/env python3
import subprocess
from pathlib import Path

import fire

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def run(
    source_model_name: str | None = None,
    target_model_name: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    source_dataset: str = "outputs/token_importance.json",
    target_dataset: str = "outputs/token_importance.json",
    src_prompt_template: str = "gsm8k_cot",
    tgt_prompt_template: str = "gsm8k_cot",
    gold_step: bool = False,
    batch_size: int = 20,
    max_gen_len: int = 400,
    patch_from_generation: bool = False,
    steps: int | str = 1,
    layers: int = 31,
    target_layer: int | None = None,
    hs_selection: str = "-1",
    patching_k_values: tuple[int, ...] = (1,),
    seed: int = 42,
    output_dir: str = "patch",
    same_layer_patch: bool = False,
    patch_position: int | None = None,
    start_layer: int | None = None,
    gen_cache_dir: str | None = None,
) -> None:
    """
    Run patch experiments over layer and top-k grids.

    Parameters
    ----------
    layers : int
        Maximum layer index to sweep (inclusive).
    target_layer : int | None
        If provided, only patch into this target layer.
    patching_k_values : tuple[int, ...]
        Values of patching_k to sweep.
    output_dir : str
        Directory to write results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "python",
        "-m",
        "ri.entry_points.patch",
        f"--model_name={model_name}",
        f"--source_dataset={source_dataset}",
        f"--target_dataset={target_dataset}",
        f"--src_prompt_template={src_prompt_template}",
        f"--tgt_prompt_template={tgt_prompt_template}",
        f"--gold_step={gold_step}",
        f"--batch_size={batch_size}",
        f"--hs_selection={hs_selection}",
        f"--max_gen_len={max_gen_len}",
        f"--steps={steps}",
        f"--seed={seed}",
        f"--patch_from_generation={patch_from_generation}",
    ]

    if source_model_name is not None:
        base_cmd.append(f"--source_model_name={source_model_name}")
    if target_model_name is not None:
        base_cmd.append(f"--target_model_name={target_model_name}")
    if gen_cache_dir is not None:
        base_cmd.append(f"--gen_cache_dir={gen_cache_dir}")
    if patch_position is not None:
        base_cmd.append(f"--patch_position={patch_position}")

    source_start = start_layer if start_layer is not None else layers
    source_layers = range(source_start, -1, -1)
    target_layers = range(layers, -1, -1) if target_layer is None else [target_layer]

    for src_layer in source_layers:
        for tgt_layer in target_layers:
            if same_layer_patch and src_layer != tgt_layer:
                continue
            for patching_k in patching_k_values:
                out_file = (
                    output_path / f"src{src_layer}_tgt{tgt_layer}_k{patching_k}_step{steps}.json"
                )
                cmd = [
                    *base_cmd,
                    f"--source_layer={src_layer}",
                    f"--target_layer={tgt_layer}",
                    f"--patching_k={patching_k}",
                    f"--output_file={out_file}",
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    fire.Fire(run)
