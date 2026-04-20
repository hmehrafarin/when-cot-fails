# when-cot-fails

Toolkit for hidden-state patching experiments on reasoning models. Patches activations from a source model (with CoT) into a target model (without CoT) to measure how reasoning representations transfer across layers and token positions.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs the project in editable mode.

Run commands via `uv run` (e.g. `uv run python -m ri.main ...`), or activate the environment with `source .venv/bin/activate`.

Requires Python 3.10+, PyTorch, and Transformers.

## Configuration

All experiments are driven through [Hydra](https://hydra.cc/) configs composed from `ri/conf/`:

```
ri/conf/
├── config.yaml          # root — picks one task / model / dataset / tracking
├── task/                # evaluate, patch, cma, pe_analysis, patch_position_sweep
├── model/               # llama_8b, qwen_7b
├── dataset/             # gsm8k
└── tracking/            # disabled, wandb
```

The entrypoint is `ri/main.py`. Override any value from the CLI with dotted paths:

```bash
uv run python -m ri.main \
    task=patch \
    model=qwen_7b \
    task.source_layer=15 task.target_layer=15 \
    task.patch_from_generation=true
```

Pydantic v2 validates each config when it is constructed, so invalid values fail loudly at the start of a run.

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RI_CACHE_DIR` | HuggingFace model cache directory | `~/.cache/huggingface` |
| `RI_OUTPUT_DIR` | Generation hidden state cache directory | `outputs` |
| `PROJECTDIR` | Base for logit cache (`$PROJECTDIR/patch_logits`) | `/tmp` |
| `RI_DEFAULT_MODEL` | Default model name | `meta-llama/Llama-3.1-8B-Instruct` |

Standard HuggingFace variables (`TRANSFORMERS_CACHE`, `HF_HOME`, `HUGGINGFACE_HUB_CACHE`) are also respected as fallbacks for model caching.

### Caching

Generation hidden states and logits are cached to disk by default to speed up reruns.

- **Generation cache** — hidden states from source model forward passes are saved as `.pt` files in `RI_OUTPUT_DIR`. Controlled via `task.gen_cache_dir` on the `patch` task.
- **Logit cache** — per-source-position logits for PE analysis are saved under `$PROJECTDIR/patch_logits/`. Controlled via `task.cache_logits` (default: true) and `task.logit_cache_dir` on the `pe_analysis` task. Set `task.cache_logits=false` to disable.

To clear cached data, delete the relevant directories.

## Running experiments

### Step 1: Generate model outputs

```bash
# CoT outputs
uv run python -m ri.main task=evaluate \
    dataset.src_prompt_template=gsm8k_cot \
    task.batch_size=1 task.max_gen_len=400 \
    task.output_file=outputs/single_batch_output_cot.json

# non-CoT outputs
uv run python -m ri.main task=evaluate \
    dataset.src_prompt_template=gsm8k_non_cot \
    task.batch_size=1 task.max_gen_len=400 \
    task.output_file=outputs/single_batch_output_non_cot.json
```

**Note:** Batch size > 1 is not yet supported for patching experiments. Use `task.batch_size=1`.

### Step 2: Run patching experiments

Both `patch_position_sweep` and `pe_analysis` operate on a single sample (`task.sample_idx`) and sweep across layers and target positions.

#### Patch position sweep

```bash
uv run python -m ri.main task=patch_position_sweep \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.sample_idx=0 \
    task.patch_from_generation=true \
    task.output_dir=patch_pos_sweep_results
```

#### Patch effect (PE) analysis

```bash
uv run python -m ri.main task=pe_analysis \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.sample_idx=0 \
    task.output_dir=pe_output
```

The PE metric is:

```
PE = (before_patch_target_prob - after_patch_target_prob) / max(after_patch_target_prob, 1e-10)
```

#### Single-run patching

```bash
uv run python -m ri.main task=patch \
    task.source_layer=15 task.target_layer=15 \
    task.patch_from_generation=true \
    task.output_file=output_patched.json
```

#### Causal mediation analysis

```bash
uv run python -m ri.main task=cma \
    task.source_layer=25 task.target_layer=25 \
    task.output_file=patch_position_analysis.json
```

### Grid sweeps (`--multirun`)

Hydra's native `--multirun` (`-m`) replaces the old `ri-patch-grid` script. Sweep over any combination of fields by passing comma-separated values:

```bash
uv run python -m ri.main -m task=patch \
    task.source_layer=0,4,8,12,16,20,24,28 \
    task.target_layer=0,4,8,12,16,20,24,28 \
    task.patching_k=1,3,5
```

Each run writes to a unique subdirectory under `multirun/<date>/<time>/<job>`. Range syntax is also supported:

```bash
uv run python -m ri.main -m task=patch \
    'task.source_layer=range(0,32)' \
    'task.target_layer=range(0,32)'
```

## Project structure

- `ri/main.py` — Hydra entrypoint with task dispatch
- `ri/conf/` — YAML config tree (task/model/dataset/tracking)
- `ri/settings/` — module-level environment/path constants
- `ri/core/` — model loading and forward hook infrastructure
- `ri/patching/` — patching pipeline, Pydantic config, and tensor operations
- `ri/patching/cma/` — causal mediation analysis
- `ri/prompts/` — prompt templates and construction
- `ri/evaluation/` — generation and evaluation runner
- `ri/utils/` — tokenizer helpers, answer extraction
