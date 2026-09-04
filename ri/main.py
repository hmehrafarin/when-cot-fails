"""Hydra entrypoint for all experiment tasks."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, ListConfig

from ri.tracking import ExperimentTracker


def _build_tracker(cfg: DictConfig) -> ExperimentTracker:
    tracking_cfg = cfg.get("tracking", {})
    return ExperimentTracker(
        project=tracking_cfg.get("project", "when-cot-fails"),
        name=tracking_cfg.get("name"),
        tags=list(tracking_cfg.get("tags") or []),
        enabled=tracking_cfg.get("enabled", False),
    )


def _resolve_int_list(raw: object) -> list[int] | None:
    """Accept None, a comma-separated string, or a list — return list[int] or None."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, ListConfig)):
        return [int(v) for v in raw] or None
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return None
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()] or None


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    task = cfg.task.name
    tracker = _build_tracker(cfg)

    try:
        _dispatch(cfg, task, tracker)
    finally:
        tracker.finish()


def _dispatch(cfg: DictConfig, task: str, tracker: ExperimentTracker) -> None:
    if task == "evaluate":
        from ri.evaluation.runner import run_evaluation

        run_evaluation(
            model_name=cfg.model.model_name,
            dataset=cfg.dataset.source_dataset,
            prompt_template=cfg.dataset.src_prompt_template,
            batch_size=cfg.task.batch_size,
            max_gen_len=cfg.task.max_gen_len,
            seed=cfg.seed,
            output_file=cfg.task.output_file,
            extraction_mode=cfg.task.get("extraction_mode", "flexible"),
            tracker=tracker,
        )

    elif task == "patch":
        from ri.patching.runner import run_patch

        run_patch(
            source_model_name=cfg.model.source_model_name,
            target_model_name=cfg.model.target_model_name,
            source_dataset=cfg.dataset.source_dataset,
            target_dataset=cfg.dataset.target_dataset,
            src_prompt_template=cfg.dataset.src_prompt_template,
            tgt_prompt_template=cfg.dataset.tgt_prompt_template,
            batch_size=cfg.task.batch_size,
            max_gen_len=cfg.task.max_gen_len,
            source_max_gen_len=cfg.task.source_max_gen_len,
            source_layer=cfg.task.source_layer,
            target_layer=cfg.task.target_layer,
            patch_position=cfg.task.patch_position,
            seed=cfg.seed,
            hs_selection=cfg.task.hs_selection,
            include_all_tokens=cfg.task.include_all_tokens,
            patch_from_generation=cfg.task.patch_from_generation,
            gen_cache_dir=cfg.task.gen_cache_dir,
            extraction_mode=cfg.task.get("extraction_mode", "flexible"),
            output_file=cfg.task.output_file,
            tracker=tracker,
        )

    elif task == "cma":
        from ri.patching.cma.runner import run_cma

        run_cma(
            source_model_name=cfg.model.source_model_name,
            target_model_name=cfg.model.target_model_name,
            source_dataset=cfg.dataset.source_dataset,
            target_dataset=cfg.dataset.target_dataset,
            src_prompt_template=cfg.dataset.src_prompt_template,
            tgt_prompt_template=cfg.dataset.tgt_prompt_template,
            source_layer=cfg.task.source_layer,
            target_layer=cfg.task.target_layer,
            seed=cfg.seed,
            hs_selection=cfg.task.hs_selection,
            include_all_tokens=cfg.task.include_all_tokens,
            patch_from_generation=cfg.task.patch_from_generation,
            max_gen_len=cfg.task.max_gen_len,
            source_max_gen_len=cfg.task.source_max_gen_len,
            patch_position=cfg.task.patch_position,
            extraction_mode=cfg.task.get("extraction_mode", "flexible"),
            output_file=cfg.task.output_file,
            tracker=tracker,
        )

    elif task == "pe_analysis":
        from ri.patching.pe_analysis import run_pe_analysis

        target_positions = _resolve_int_list(cfg.task.target_positions)
        target_positions_arg = (
            ",".join(str(v) for v in target_positions) if target_positions else None
        )

        run_pe_analysis(
            source_model_name=cfg.model.source_model_name,
            target_model_name=cfg.model.target_model_name,
            source_dataset=cfg.dataset.source_dataset,
            target_dataset=cfg.dataset.target_dataset,
            output_dir=cfg.task.output_dir,
            sample_idx=cfg.task.sample_idx,
            start_src_pos=cfg.task.start_src_pos,
            seed=cfg.seed,
            max_gen_len=cfg.task.max_gen_len,
            cache_logits=cfg.task.cache_logits,
            logit_cache_dir=cfg.task.logit_cache_dir,
            target_positions=target_positions_arg,
            resume=cfg.task.resume,
        )

    elif task == "patch_position_sweep":
        from ri.patching.patch_position_sweep import run as run_patch_position_sweep

        run_patch_position_sweep(
            sample_idx=cfg.task.sample_idx,
            max_gen_len=cfg.task.max_gen_len,
            source_max_gen_len=cfg.task.source_max_gen_len,
            layer=cfg.task.layer,
            start_layer=cfg.task.start_layer,
            layer_stride=cfg.task.layer_stride,
            include_final_layer=cfg.task.include_final_layer,
            target_pos=cfg.task.target_pos,
            target_positions=_resolve_int_list(cfg.task.target_positions),
            output_dir=cfg.task.output_dir,
            patch_from_generation=cfg.task.patch_from_generation,
            gen_cache_dir=cfg.task.gen_cache_dir,
            extraction_mode=cfg.task.get("extraction_mode", "flexible"),
            source_dataset=cfg.dataset.source_dataset,
            target_dataset=cfg.dataset.target_dataset,
            source_model_name=cfg.model.source_model_name,
            target_model_name=cfg.model.target_model_name,
            src_prompt_template=cfg.dataset.src_prompt_template,
            tgt_prompt_template=cfg.dataset.tgt_prompt_template,
            resume=cfg.task.resume,
            seed=cfg.seed,
        )

    elif task == "full_results":
        from ri.postprocess.runner import run_postprocess

        run_postprocess(
            sweep_root=cfg.task.sweep_root,
            output_file=cfg.task.output_file,
            model_name=cfg.model.target_model_name,
            output_schema=cfg.task.output_schema,
            pe_root=cfg.task.pe_root,
            eval_json=cfg.task.eval_json,
            sample_idx=cfg.task.sample_idx,
            spacy_model=cfg.task.spacy_model,
            progress_every=cfg.task.progress_every,
            source_tokens_file=cfg.task.source_tokens_file,
            entity_codes_file=cfg.task.entity_codes_file,
            behavior_codes_file=cfg.task.behavior_codes_file,
        )

    else:
        raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    main()
