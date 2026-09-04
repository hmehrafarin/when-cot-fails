import json
import pathlib

from tqdm import tqdm

from ri.common import build_prompt_batch, prepare_batch_data
from ri.patching.config import ExtractionMode, PatchConfig
from ri.patching.pipeline import get_source_hidden_states
from ri.patching.runner import PatchRunner
from ri.utils import make_inputs, render_prompts
from ri.utils.text import prompt_text_from_rendered


def _is_complete_result_file(
    file_path: pathlib.Path,
    expected_layer: int,
    expected_target_pos: int,
    expected_patch_count: int,
) -> bool:
    if not file_path.exists():
        return False

    try:
        with open(file_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False
    if payload.get("layer") != expected_layer:
        return False
    if payload.get("target_pos") != expected_target_pos:
        return False

    patch_result = payload.get("patch_result")
    if not isinstance(patch_result, list):
        return False
    return len(patch_result) == expected_patch_count


def _write_json_atomic(file_path: pathlib.Path, payload: dict) -> None:
    tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(file_path)


def run(
    *,
    source_model_name: str,
    source_dataset: str,
    target_dataset: str,
    target_model_name: str | None = None,
    src_prompt_template: str = "gsm8k_cot",
    tgt_prompt_template: str = "gsm8k_non_cot",
    sample_idx: int = 0,
    max_gen_len: int = 400,
    source_max_gen_len: int | None = None,
    layer: int | None = None,
    start_layer: int = 0,
    layer_stride: int = 1,
    include_final_layer: bool = False,
    target_pos: int | None = None,
    target_positions: list[int] | None = None,
    output_dir: str = "patch_pos_sweep_results",
    patch_from_generation: bool = True,
    gen_cache_dir: str | None = None,
    extraction_mode: ExtractionMode = "flexible",
    resume: bool = False,
    seed: int = 42,
) -> None:
    """Patch one source hidden state per ``(layer, target_pos, source_pos)`` and save the generations.

    One ``layer_<L>_pos_<T>.json`` file is written per (layer, target position), with ``<L>``
    1-indexed. ``max_gen_len`` bounds both the source CoT generation and the post-patch
    generation; ``source_max_gen_len`` overrides it for the source CoT only. Set
    ``gen_cache_dir`` to avoid regenerating the source CoT for every patch.
    """
    if target_positions is not None and target_pos is not None:
        raise ValueError("Provide either task.target_pos or task.target_positions, not both.")
    if layer is None and layer_stride <= 0:
        raise ValueError("task.layer_stride must be > 0 when sweeping layers.")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initial config; layer, target position and source position are updated per iteration.
    initial_layer = layer if layer is not None else 0
    patch_config = PatchConfig(
        max_gen_len=max_gen_len,
        source_max_gen_len=source_max_gen_len,
        source_layer=initial_layer,
        target_layer=initial_layer,
        patch_position=target_pos,
        hs_selection=0,
        include_all_tokens=True,
        gen_cache_dir=gen_cache_dir,
        extraction_mode=extraction_mode,
    )

    runner = PatchRunner(
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        src_prompt_template=src_prompt_template,
        tgt_prompt_template=tgt_prompt_template,
        patch_from_generation=patch_from_generation,
        patch_config=patch_config,
        seed=seed,
        batch_size=1,
    )

    # 1. Determine source length
    print("Determining source hidden states length...")
    _q, _a, batched_input_source = prepare_batch_data(runner.source_data, sample_idx, 1)

    source_convos = build_prompt_batch(runner.src_prompter, batched_input_source)
    rendered_source_prompts = render_prompts(
        runner.source_mt.tokenizer,
        source_convos,
        system_prompt=True,
        add_generation_prompt=True,
    )
    source_prompt_texts = [prompt_text_from_rendered(prompt) for prompt in source_convos]
    tokenized_source = make_inputs(
        runner.source_mt.tokenizer,
        source_convos,
        device=runner.source_mt.device,
        system_prompt=True,
        add_generation_prompt=True,
        rendered_prompts=rendered_source_prompts,
    )

    # Get source HS to find length
    source_hs, _source_ids, source_extracted_answers = get_source_hidden_states(
        runner.source_mt,
        tokenized_source,
        source_prompt_texts,
        patch_config,
        runner.tgt_prompter,
        patch_from_generation=patch_from_generation,
        batch_size=1,
    )
    num_source_tokens = source_hs.size(1)
    print(f"Source has {num_source_tokens} tokens.")

    # 2. Determine target length (rendered exactly as patch_and_generate renders it)
    print("Determining target prompt length...")
    target_supports_system_prompt = bool(getattr(runner.target_mt, "is_instruct_model", False))
    _q_tgt, _a_tgt, batched_input_tgt = prepare_batch_data(runner.target_data, sample_idx, 1)

    target_convos = build_prompt_batch(runner.tgt_prompter, batched_input_tgt)
    rendered_tgt_prompts = render_prompts(
        runner.target_mt.tokenizer,
        target_convos,
        system_prompt=target_supports_system_prompt,
        add_generation_prompt=target_supports_system_prompt,
    )
    tokenized_tgt = make_inputs(
        runner.target_mt.tokenizer,
        target_convos,
        device=runner.target_mt.device,
        system_prompt=target_supports_system_prompt,
        add_generation_prompt=target_supports_system_prompt,
        rendered_prompts=rendered_tgt_prompts,
    )

    num_target_tokens = tokenized_tgt["input_ids"].shape[1]
    print(f"Target has {num_target_tokens} tokens.")

    # Gold answers and the source model's own CoT, recorded in every output file
    source_answer = runner.source_data[sample_idx].get("answer", "")
    target_answer = runner.target_data[sample_idx].get("answer", "")

    source_generated_answer = ""
    answer_texts = source_extracted_answers.get("answer_text") if source_extracted_answers else None
    if answer_texts:
        source_generated_answer = answer_texts[0]

    source_prompt = rendered_source_prompts[0]
    target_prompt = rendered_tgt_prompts[0]

    model_num_layers = runner.source_mt.model.config.num_hidden_layers

    # Determine layer and target position ranges based on mode
    # Mode logic:
    #   - layer provided, no pos: sweep all positions for that layer
    #   - layer and pos provided: single mode (one layer, one position)
    #   - no layer, pos provided: sweep layers from start_layer, fixed position
    #   - no layer, no pos: full grid from start_layer
    if layer is not None:
        layer_range = [layer]
        if target_positions is not None:
            target_pos_range = target_positions
            print(f"Layer mode: layer={layer}, target_positions={target_positions}")
        elif target_pos is not None:
            target_pos_range = [target_pos]
            print(f"Single mode: layer={layer}, target_pos={target_pos}")
        else:
            target_pos_range = None  # Will sweep all target positions
            print(f"Layer mode: layer={layer}, sweeping all target positions")
    else:
        layer_range = list(range(start_layer, model_num_layers, layer_stride))
        if include_final_layer and model_num_layers > 0:
            final_layer_idx = model_num_layers - 1
            if final_layer_idx >= start_layer and final_layer_idx not in layer_range:
                layer_range.append(final_layer_idx)
                layer_range.sort()
        if target_positions is not None:
            target_pos_range = target_positions
            print(f"Layer sweep mode: layers={layer_range}, target_positions={target_positions}")
        elif target_pos is not None:
            target_pos_range = [target_pos]
            print(f"Layer sweep mode: layers={layer_range}, target_pos={target_pos}")
        else:
            target_pos_range = None  # Will sweep all target positions
            print(f"Grid mode: layers={layer_range}, all target positions")

    # Iterate Layers
    for layer_idx in tqdm(layer_range, desc="Layers"):
        runner.patch_config.source_layer = layer_idx
        runner.patch_config.target_layer = layer_idx

        # Determine target positions for this layer
        current_target_positions = (
            target_pos_range if target_pos_range is not None else range(num_target_tokens)
        )

        # Iterate Target Positions
        for tgt_pos in tqdm(
            current_target_positions, desc=f"Layer {layer_idx} Targets", leave=False
        ):
            file_name = f"layer_{layer_idx + 1}_pos_{tgt_pos}.json"
            file_path = output_path / file_name

            if resume:
                if _is_complete_result_file(
                    file_path=file_path,
                    expected_layer=layer_idx + 1,
                    expected_target_pos=tgt_pos,
                    expected_patch_count=num_source_tokens,
                ):
                    print(f"Skipping completed file: {file_path}")
                    continue
                if file_path.exists():
                    print(f"Recomputing incomplete file: {file_path}")

            runner.patch_config.patch_position = tgt_pos

            patch_results = []

            # Iterate Source Positions
            for src_pos in range(num_source_tokens):
                runner.patch_config.hs_selection = src_pos

                # Run single batch
                batch_res = runner._run_single_batch(sample_idx)

                single_res = {
                    k: v[0] if isinstance(v, list) and len(v) > 0 else None
                    for k, v in batch_res.items()
                }

                patch_entry = {
                    "pos": src_pos,
                    "patching_token": single_res.get("patch_from", ""),
                    "generated_text": single_res.get("Generated Answer_cot", ""),
                }
                patch_results.append(patch_entry)

            # Save results for this (layer, target_pos)
            final_output = {
                "layer": layer_idx + 1,
                "target_pos": tgt_pos,
                "source_prompt": source_prompt,
                "target_prompt": target_prompt,
                "source_gold_answer": source_answer,
                "source_generated_answer": source_generated_answer,
                "target_gold_answer": target_answer,
                "patch_result": patch_results,
            }

            _write_json_atomic(file_path, final_output)

    print("Done.")
