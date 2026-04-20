from __future__ import annotations

import json
import os
import traceback
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ri.core.hooks import remove_hooks, set_patch
from ri.patching.cma import CausalMediationRunner
from ri.patching.cma.pipeline import (
    build_prompt_inputs,
    make_batched_input,
    resolve_target_steps,
)
from ri.patching.config import PatchConfig
from ri.patching.logit_cache import LogitCache, LogitCacheConfig
from ri.patching.pipeline import _safe_token_from_id
from ri.patching.tensor_ops import left_pad_offsets
from ri.settings.settings import PATCH_LOGITS_CACHE_DIR
from ri.utils import (
    decode_tokens,
    extract_answer_from_generation,
    get_eos_token_ids,
    get_pad_id,
)


def compute_probs_batched(
    logits: torch.Tensor,
    token_ids: list[int],
    position: int = -1,
) -> torch.Tensor:
    """
    Compute sum of probabilities for token_ids for each item in batch.

    Returns tensor of shape (B,).
    """
    if not token_ids:
        return torch.zeros(logits.size(0), device=logits.device)

    pos_logits = logits[:, position, :]  # (B, V)
    probs = F.softmax(pos_logits, dim=-1)  # (B, V)
    selected_probs = probs[:, token_ids]  # (B, num_tokens)
    return selected_probs.sum(dim=1)  # (B,)


def _parse_target_positions_arg(
    target_positions: Any | None,
) -> list[int] | None:
    if target_positions is None:
        return None

    def _to_int_list(values: list[Any]) -> list[int]:
        parsed_values: list[int] = []
        for value in values:
            try:
                parsed_values.append(int(value))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid target position token '{value}' in '{target_positions}'. "
                    "Expected a comma-separated list of integers."
                ) from e
        return parsed_values

    if isinstance(target_positions, (list, tuple)):
        parsed = _to_int_list(list(target_positions))
        return parsed or None

    raw = str(target_positions).strip()
    if not raw or raw.lower() == "none":
        return None

    # Fire may parse `--target_positions=0,-1` as `(0, -1)`; accept either.
    if (
        (raw.startswith("(") and raw.endswith(")")) or (raw.startswith("[") and raw.endswith("]"))
    ) and len(raw) >= 2:
        raw = raw[1:-1].strip()
    if not raw:
        return None

    parsed_tokens: list[Any] = []
    for raw_token in raw.split(","):
        token = raw_token.strip()
        if not token:
            continue
        parsed_tokens.append(token)

    parsed = _to_int_list(parsed_tokens)
    return parsed or None


def _resolve_target_positions(
    valid_target_positions: list[int],
    requested_positions: list[int] | None,
) -> list[int]:
    if requested_positions is None:
        return list(valid_target_positions)

    num_valid = len(valid_target_positions)
    resolved: list[int] = []
    seen: set[int] = set()

    for requested in requested_positions:
        resolved_idx = requested if requested >= 0 else num_valid + requested
        if resolved_idx < 0 or resolved_idx >= num_valid:
            raise ValueError(
                f"Requested target position {requested} resolves to index {resolved_idx}, "
                f"but valid target positions have length {num_valid}."
            )
        abs_pos = valid_target_positions[resolved_idx]
        if abs_pos in seen:
            continue
        seen.add(abs_pos)
        resolved.append(abs_pos)

    return resolved


def _is_complete_source_result_file(
    file_path: str,
    sample_idx: int,
    source_position: int,
    expected_num_layers: int,
    expected_target_positions: list[int],
) -> bool:
    if not os.path.exists(file_path):
        return False

    try:
        with open(file_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(payload, dict):
        return False
    if payload.get("sample_idx") != sample_idx:
        return False
    if payload.get("source_position") != source_position:
        return False
    layer_results = payload.get("layer_results")
    if not isinstance(layer_results, dict):
        return False
    if len(layer_results) != expected_num_layers:
        return False

    for layer_idx in range(expected_num_layers):
        layer_key = str(layer_idx)
        layer_payload = layer_results.get(layer_key)
        if not isinstance(layer_payload, dict):
            return False
        positions = layer_payload.get("positions")
        if not isinstance(positions, list):
            return False
        if len(positions) != len(expected_target_positions):
            return False
        actual_target_positions = [entry.get("target_position") for entry in positions]
        if actual_target_positions != expected_target_positions:
            return False

    payload_target_positions = payload.get("target_positions_resolved")
    return payload_target_positions is None or payload_target_positions == expected_target_positions


def _write_json_atomic(file_path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{file_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, file_path)


class PatchEffectAnalyzer(CausalMediationRunner):
    """
    Analyzer for computing patch effects across all layers and target positions.

    For each layer, iterates over target positions and computes the
    patch effect when patching source hidden states.
    """

    def __init__(
        self,
        source_model_name: str,
        source_dataset: str,
        target_dataset: str,
        src_prompt_template: str = "gsm8k_cot",
        tgt_prompt_template: str = "gsm8k_non_cot",
        patch_from_generation: bool = True,
        seed: int = 42,
        gold_step: bool = True,
        target_model_name: str | None = None,
        max_gen_len: int = 400,
        start_src_pos: int | None = 0,
        cache_logits: bool = True,
        logit_cache_dir: str | None = None,
        target_positions: str | None = None,
        resume: bool = False,
    ):
        patch_config = PatchConfig(
            max_gen_len=max_gen_len,
            source_layer=0,
            target_layer=0,
        )
        super().__init__(
            source_model_name=source_model_name,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            src_prompt_template=src_prompt_template,
            tgt_prompt_template=tgt_prompt_template,
            patch_from_generation=patch_from_generation,
            patch_config=patch_config,
            seed=seed,
            gold_step=gold_step,
            target_model_name=target_model_name,
        )
        self.start_src_pos = start_src_pos
        self.target_positions = _parse_target_positions_arg(target_positions)
        self.resume = resume

        # Cache configuration
        self.cache_logits = cache_logits
        self.logit_cache_dir = logit_cache_dir or PATCH_LOGITS_CACHE_DIR
        # Store params needed for cache key
        self.source_dataset_path = source_dataset
        self.target_dataset_path = target_dataset
        self.seed = seed
        self.src_prompt_template = src_prompt_template
        self.tgt_prompt_template = tgt_prompt_template

    def _get_all_layers_hidden_states(
        self,
        tokenized_source: dict[str, torch.Tensor],
        source_prompt_texts: list[str],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Get hidden states from all layers during generation.

        Returns:
            - list of tensors, one per layer, each of shape (B, gen_len, H)
            - generated token IDs of shape (B, gen_len)
        """
        source_pad_id = get_pad_id(self.source_mt.tokenizer)
        source_eos_ids = get_eos_token_ids(self.source_mt.tokenizer)

        prompt_len = tokenized_source["input_ids"].size(1)

        generation = self.source_mt.model.generate(
            **tokenized_source,
            max_new_tokens=self.config.max_gen_len,
            eos_token_id=source_eos_ids,
            pad_token_id=source_pad_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        hidden_steps = generation.hidden_states
        if not hidden_steps:
            raise RuntimeError("No hidden states returned from generation")

        # Extract generated token IDs (excluding prompt)
        generated_ids = generation.sequences[:, prompt_len:]

        # hidden_steps: tuple of (steps) of tuple (layers) of tensor (B, 1, H)
        num_layers = len(hidden_steps[0])
        layers_hs_list: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        for step_states in hidden_steps:
            for layer_idx, layer_tensor in enumerate(step_states):
                # Extract last token hidden state
                hs = layer_tensor[:, -1, :] if layer_tensor.dim() == 3 else layer_tensor
                layers_hs_list[layer_idx].append(hs)

        # Stack to (B, T, H) for each layer
        return [torch.stack(layer_list, dim=1) for layer_list in layers_hs_list], generated_ids

    def _run_single_sample(self, sample_idx: int, output_folder: str) -> None:  # type: ignore[override]
        """Analyze patch effects for a single sample, saving per source position."""
        source_sample = self.source_data[sample_idx]
        target_sample = self.target_data[sample_idx]

        source_tokenizer = self.source_mt.tokenizer
        target_tokenizer = self.target_mt.tokenizer
        target_supports_system = bool(getattr(self.target_mt, "is_instruct_model", False))

        # Build source inputs
        batched_input_source = make_batched_input(source_sample, include_generated=True)
        _, _, source_prompt_texts, tokenized_source = build_prompt_inputs(
            source_tokenizer,
            self.src_prompter,
            batched_input_source,
            steps=self.config.steps,
            device=self.source_mt.device,
            system_prompt=True,
            add_generation_prompt=True,
        )

        # Get hidden states from all layers and generated token IDs
        all_layers_hs, generated_ids = self._get_all_layers_hidden_states(
            tokenized_source, source_prompt_texts
        )

        # Build target inputs
        batched_input_tgt = make_batched_input(target_sample, include_generated=False)
        tgt_template_name = getattr(self.tgt_prompter, "template_name", "unknown")
        tgt_steps = resolve_target_steps(self.config.steps, tgt_template_name, self.gold_step)

        _, _, _, tokenized_tgt = build_prompt_inputs(
            target_tokenizer,
            self.tgt_prompter,
            batched_input_tgt,
            steps=tgt_steps,
            device=self.target_mt.device,
            system_prompt=target_supports_system,
            add_generation_prompt=target_supports_system,
        )

        # Compute valid target positions from non-pad tokens.
        tgt_offsets = left_pad_offsets(tokenized_tgt)
        attention_mask = tokenized_tgt["attention_mask"][0]
        valid_len = int(attention_mask.sum().item())
        offset = tgt_offsets[0]
        valid_target_positions = list(range(offset, offset + valid_len))
        target_positions = _resolve_target_positions(
            valid_target_positions=valid_target_positions,
            requested_positions=self.target_positions,
        )
        if not target_positions:
            raise ValueError(
                f"No target positions resolved for sample {sample_idx}. "
                f"requested={self.target_positions}, valid_count={len(valid_target_positions)}"
            )

        # Get source CoT
        source_cot = source_sample.get(
            "Generated Answer_cot", source_sample.get("Generated Answer_CoT", "")
        )
        if isinstance(source_cot, list):
            source_cot = source_cot[0] if source_cot else ""
        source_cot = str(source_cot or "").strip()

        # Get target baseline generation and answer
        target_pad_id = get_pad_id(target_tokenizer)
        target_eos_ids = get_eos_token_ids(target_tokenizer)

        with torch.no_grad():
            target_baseline_gen = self.target_mt.model.generate(
                **tokenized_tgt,
                max_new_tokens=self.config.max_gen_len,
                eos_token_id=target_eos_ids,
                pad_token_id=target_pad_id,
                do_sample=False,
            )

        target_baseline_text = decode_tokens(target_tokenizer, target_baseline_gen)
        target_baseline_extracted = extract_answer_from_generation(
            target_baseline_text,
            tokenizer=target_tokenizer,
            template_name=tgt_template_name,
        )
        target_answer_num = (
            target_baseline_extracted["answer_num"][0]
            if target_baseline_extracted.get("answer_num")
            else None
        )
        target_answer_token_ids = (
            target_tokenizer.encode(str(target_answer_num).strip(), add_special_tokens=False)
            if target_answer_num is not None
            else []
        )

        # Compute baseline logits and probabilities
        with torch.no_grad():
            baseline_output = self.target_mt.model(**tokenized_tgt, output_hidden_states=False)
        baseline_logits = baseline_output.logits

        before_patch_target_prob = compute_probs_batched(
            baseline_logits, target_answer_token_ids
        ).item()

        # Get number of layers and source positions
        # all_layers_hs has N+1 elements (embeddings + N layers)
        num_layers = len(all_layers_hs) - 1
        # Use first layer to get num_source (all layers have same seq_len)
        num_source = all_layers_hs[1][0].size(0)

        # Compute effective start position (use local variable to avoid mutating self)
        start_pos = self.start_src_pos if self.start_src_pos is not None else 0
        if start_pos < 0:
            start_pos = num_source + start_pos
        if start_pos < 0 or start_pos >= num_source:
            raise ValueError(
                f"start_src_pos {self.start_src_pos} resolves to {start_pos}, "
                f"which is out of bounds for source length {num_source}"
            )

        # Common metadata for all source position files
        common_metadata = {
            "sample_idx": sample_idx,
            "target_positions_requested": self.target_positions,
            "target_positions_resolved": target_positions,
            "before_patch_target_prob": before_patch_target_prob,
            "source_generated_answer_cot": source_cot,
        }

        # Initialize logit cache
        logit_cache = None
        if self.cache_logits:
            cache_config = LogitCacheConfig(
                cache_dir=self.logit_cache_dir,
                sample_idx=sample_idx,
                source_model_name=self.source_mt.model_name or "",
                target_model_name=self.target_mt.model_name or "",
                source_dataset=self.source_dataset_path,
                target_dataset=self.target_dataset_path,
                max_gen_len=self.config.max_gen_len,
                seed=self.seed,
                src_prompt_template=self.src_prompt_template,
                tgt_prompt_template=self.tgt_prompt_template,
            )
            logit_cache = LogitCache(cache_config)
            cache_info = logit_cache.get_cache_info()
            if cache_info["exists"]:
                print(
                    f"Found existing cache at {cache_info['cache_path']} with {cache_info['num_files']} files"
                )

        # Get vocab size for cache tensor allocation
        vocab_size = self.target_mt.model.config.vocab_size

        # Process each source position separately
        skipped_completed = 0
        for src_pos in tqdm(
            range(start_pos, num_source), desc=f"Sample {sample_idx} Source positions", leave=False
        ):
            output_file = os.path.join(output_folder, f"source_{src_pos}.json")
            if self.resume:
                if _is_complete_source_result_file(
                    file_path=output_file,
                    sample_idx=sample_idx,
                    source_position=src_pos,
                    expected_num_layers=num_layers,
                    expected_target_positions=target_positions,
                ):
                    skipped_completed += 1
                    continue
                if os.path.exists(output_file):
                    print(f"  Recomputing incomplete result: {output_file}")

            # Get the token being patched at this source position
            patched_token_id = generated_ids[0, src_pos].item()
            patched_token_str = _safe_token_from_id(source_tokenizer, patched_token_id)

            # Try to load cached logits for this source position
            cached_logits = None
            if logit_cache is not None:
                cached_logits = logit_cache.load_logits(src_pos, self.target_mt.device)
                if cached_logits is not None:
                    print(f"  Loaded cached logits for src_pos {src_pos}")

            # Prepare tensor to accumulate logits for caching (if not cached)
            logits_to_cache = None
            if logit_cache is not None and cached_logits is None:
                logits_to_cache = torch.zeros(
                    (num_layers, len(target_positions), vocab_size),
                    dtype=torch.float32,
                    device="cpu",
                )

            layer_results: dict[int, dict[str, Any]] = {}

            for layer_idx in range(num_layers):
                # Output of layer_idx is at index layer_idx + 1
                hs_tensor = all_layers_hs[layer_idx + 1][0]  # (seq_len, H)
                src_hs = hs_tensor[src_pos : src_pos + 1]  # (1, H)

                layer_pos_results: list[dict[str, Any]] = []

                for tgt_idx, tgt_pos in enumerate(target_positions):
                    # Check if we can use cached logits
                    if cached_logits is not None:
                        # Load from cache: shape (num_layers, num_tgt_pos, vocab_size)
                        # Need shape (1, 1, vocab_size) for compute_probs_batched
                        logits = cached_logits[layer_idx, tgt_idx].unsqueeze(0).unsqueeze(0).float()
                    else:
                        # Compute fresh logits
                        patch_config_dict = {
                            "patch_method": "replace",
                            "layer_to_patch": layer_idx,
                            "hs_position": [tgt_pos],
                            "hs": src_hs,
                        }

                        hooks = set_patch(self.target_mt.model, [patch_config_dict])
                        try:
                            with torch.no_grad():
                                outputs = self.target_mt.model(
                                    **tokenized_tgt, output_hidden_states=False
                                )
                        finally:
                            remove_hooks(hooks)

                        logits = outputs.logits

                        # Store last-position logits for caching
                        if logits_to_cache is not None:
                            logits_to_cache[layer_idx, tgt_idx] = logits[0, -1].cpu()

                    after_patch_target_prob = compute_probs_batched(
                        logits, target_answer_token_ids
                    ).item()

                    pe = (before_patch_target_prob - after_patch_target_prob) / max(
                        after_patch_target_prob, 1e-10
                    )

                    layer_pos_results.append(
                        {
                            "target_position": tgt_pos,
                            "after_patch_target_prob": after_patch_target_prob,
                            "patch_effect": pe,
                        }
                    )

                layer_results[layer_idx] = {
                    "positions": layer_pos_results,
                }

            # Save logits to cache after completing this src_pos
            if logit_cache is not None and logits_to_cache is not None:
                logit_cache.save_logits(
                    src_pos=src_pos,
                    logits=logits_to_cache,
                    num_layers=num_layers,
                    target_positions=target_positions,
                )
                print(f"  Saved logits to cache for src_pos {src_pos}")

            # Save results for this source position
            result = {
                **common_metadata,
                "source_position": src_pos,
                "patched_token_str": patched_token_str,
                "layer_results": layer_results,
            }
            _write_json_atomic(output_file, result)

        if self.resume and skipped_completed:
            print(f"Skipped {skipped_completed} completed source files for sample {sample_idx}")

    def run(  # type: ignore[override]
        self,
        sample_idx: int = 0,
        output_dir: str = "pe_output",
    ) -> None:
        """
        Run patch effect analysis on a single sample.

        Parameters
        ----------
        sample_idx : int
            Index of the sample to analyze.
        output_dir : str
            Base directory for output files.
        """
        total_samples = min(len(self.source_data), len(self.target_data))
        if sample_idx >= total_samples:
            raise ValueError(f"sample_idx {sample_idx} exceeds available samples ({total_samples})")

        # Create output folder
        output_folder = os.path.join(output_dir, f"sample_{sample_idx}")
        os.makedirs(output_folder, exist_ok=True)

        try:
            self._run_single_sample(sample_idx, output_folder)
        except Exception as e:
            print(f"Error analyzing sample {sample_idx}: {e}")
            traceback.print_exc()
            error_file = os.path.join(output_folder, "error.json")
            with open(error_file, "w") as f:
                json.dump({"sample_idx": sample_idx, "error": str(e)}, f, indent=2)

        print(f"Results saved to {output_folder}")


def run_pe_analysis(
    source_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    target_model_name: str | None = None,
    source_dataset: str = "outputs/single_batch_output_cot.json",
    target_dataset: str = "outputs/single_batch_output_cot.json",
    output_dir: str = "pe_output",
    sample_idx: int = 0,
    start_src_pos: int | None = 0,
    seed: int = 42,
    max_gen_len: int = 400,
    cache_logits: bool = True,
    logit_cache_dir: str | None = None,
    target_positions: str | None = None,
    resume: bool = False,
) -> None:
    """
    Run patch effect analysis across layers and target positions.

    Parameters
    ----------
    source_model_name : str
        Source model to extract hidden states from.
    target_model_name : str | None
        Target model to patch (defaults to source model).
    source_dataset : str
        Path to source dataset JSON.
    target_dataset : str
        Path to target dataset JSON.
    output_dir : str
        Base directory for output files.
    sample_idx : int
        Index of the sample to analyze.
    start_src_pos : int | None
        Starting source position for analysis.
    seed : int
        Random seed.
    max_gen_len : int
        Maximum generation length.
    cache_logits : bool
        Whether to cache logits to disk for reuse across runs.
    logit_cache_dir : str | None
        Directory for logit cache. Defaults to $SCRATCHDIR/patch_logits.
    target_positions : str | None
        Comma-separated target token position indices resolved over valid
        (non-pad) target tokens. Example: "0,-1". None means full PE over all
        valid target positions.
    resume : bool
        If True, skip already completed source position files and only compute
        missing/incomplete files.
    """
    analyzer = PatchEffectAnalyzer(
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        seed=seed,
        max_gen_len=max_gen_len,
        start_src_pos=start_src_pos,
        cache_logits=cache_logits,
        logit_cache_dir=logit_cache_dir,
        target_positions=target_positions,
        resume=resume,
    )

    analyzer.run(
        sample_idx=sample_idx,
        output_dir=output_dir,
    )
