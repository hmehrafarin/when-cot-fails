# when-cot-fails

Toolkit for hidden-state patching experiments on reasoning models. Patches activations from a source model (with CoT) into a target model (without CoT) to measure how reasoning representations transfer across layers and token positions.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs the project in editable mode.

Run commands via `uv run` (e.g. `uv run ri ...`), or activate the environment with `source .venv/bin/activate`.

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

The entrypoint is `ri/main.py`, exposed as the `ri` console script. Override any value from the CLI with dotted paths:

```bash
uv run ri task=patch model=qwen_7b task.source_layer=15 task.target_layer=15
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

Before running any patching experiment, you need to evaluate your model on GSM-8K to produce the output files that patching and PE analysis consume. Run the model with CoT and without CoT prompts separately:

```bash
# Generate CoT outputs
uv run ri task=evaluate \
    dataset.src_prompt_template=gsm8k_cot \
    task.batch_size=1 task.max_gen_len=400 seed=42 \
    task.output_file=outputs/single_batch_output_cot.json

# Generate non-CoT outputs
uv run ri task=evaluate \
    dataset.src_prompt_template=gsm8k_non_cot \
    task.batch_size=1 task.max_gen_len=400 seed=42 \
    task.output_file=outputs/single_batch_output_non_cot.json
```

**Note:** Batch size > 1 is not yet supported for patching experiments. Use `task.batch_size=1`.

These output JSON files are then passed as `dataset.source_dataset` and `dataset.target_dataset` to the patching tasks below.

### Step 2: Run patching experiments

There are two experiment tasks for full reproducibility, plus single-run tasks (`patch`, `cma`) for standalone invocations.

Both `patch_position_sweep` and `pe_analysis` operate on a single sample (`task.sample_idx`) and sweep across all layers and all target positions. For running patching across many configurations at once, use Hydra's `--multirun` (see *Grid sweeps* below).

### Patch position sweep

Sweeps over source positions, target positions, and layers for one sample. For each combination, patches a single source CoT hidden state (with `task.patch_from_generation=true`) into the target model and saves the generated output.

```bash
uv run ri task=patch_position_sweep \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.sample_idx=0 \
    task.patch_from_generation=true \
    task.output_dir=patch_pos_sweep_results
```

Key overrides:
- `task.layer` — patch a specific layer (otherwise sweeps all layers)
- `task.start_layer`, `task.layer_stride` — control layer sweep range
- `task.target_pos` — patch at a specific target position (otherwise sweeps all)
- `task.target_positions` — comma-separated list of target positions (e.g. `"0,-1"`)
- `task.patch_from_generation=true` — extract source hidden states from generation rather than the prompt
- `task.resume=true` — skip completed output files

### Patch effect (PE) analysis

Computes the patch effect metric across all layers and target positions for one sample. For each source token position, patches its hidden state from every layer into every target position and measures how the target model's output probability distribution changes.

The patch effect is defined as:

```
PE = (before_patch_target_prob - after_patch_target_prob) / max(after_patch_target_prob, 1e-10)
```

```bash
uv run ri task=pe_analysis \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.sample_idx=0 \
    task.output_dir=pe_output
```

Key overrides:
- `task.start_src_pos` — starting source position (supports negative indexing)
- `task.target_positions` — comma-separated target positions (e.g. `"0,-1"`)
- `task.cache_logits` — cache logits to disk for reuse across runs (default: true)
- `task.resume=true` — skip completed source position files

### Standalone experiments

For generation, patching, and causal mediation analysis without the full sweep machinery:

```bash
# Generate model outputs
uv run ri task=evaluate \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.src_prompt_template=gsm8k_cot \
    task.batch_size=1 task.max_gen_len=400 seed=42 \
    task.output_file=output.json

# Single patching run
uv run ri task=patch \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.source_layer=15 task.target_layer=15 \
    task.patch_from_generation=true \
    task.output_file=output_patched.json

# Causal mediation analysis
uv run ri task=cma \
    dataset.source_dataset=outputs/single_batch_output_cot.json \
    dataset.target_dataset=outputs/single_batch_output_cot.json \
    task.source_layer=25 task.target_layer=25 \
    task.output_file=patch_position_analysis.json
```

### Grid sweeps (`--multirun`)

Hydra's native `--multirun` (`-m`) replaces the old `ri-patch-grid` subprocess launcher. Sweep over any combination of fields with comma-separated values:

```bash
uv run ri -m task=patch \
    task.source_layer=0,4,8,12,16,20,24,28 \
    task.target_layer=0,4,8,12,16,20,24,28 \
    task.patching_k=1,3,5
```

Each job writes to a unique subdirectory under `multirun/<date>/<time>/<job>`. Range syntax is also supported:

```bash
uv run ri -m task=patch \
    'task.source_layer=range(0,32)' \
    'task.target_layer=range(0,32)'
```

## Project structure

- `ri/main.py` — Hydra entrypoint with task dispatch
- `ri/conf/` — YAML config tree (task/model/dataset/tracking)
- `ri/settings/` — module-level environment/path constants
- `ri/core/` — model loading and forward hook infrastructure
- `ri/patching/` — patching pipeline, Pydantic config, tensor operations, and the `pe_analysis` / `patch_position_sweep` experiment modules
- `ri/patching/cma/` — causal mediation analysis
- `ri/prompts/` — prompt templates and construction
- `ri/evaluation/` — generation and evaluation runner
- `ri/utils/` — tokenizer helpers, answer extraction
