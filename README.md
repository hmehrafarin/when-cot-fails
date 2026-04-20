# reasoning-interpretability

Toolkit for hidden-state patching experiments on reasoning models. Patches activations from a source model (with CoT) into a target model (without CoT) to measure how reasoning representations transfer across layers and token positions.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and package management.

```bash
uv sync
```

This creates a virtual environment in `.venv/` and installs the project in editable mode.

Run commands via `uv run` (e.g. `uv run ri ...`), or activate the environment with `source .venv/bin/activate`.

Requires Python 3.9+, PyTorch, and Transformers.

## Configuration

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

- **Generation cache** — hidden states from source model forward passes are saved as `.pt` files in `RI_OUTPUT_DIR`. Controlled via `--gen_cache_dir` on `run_patch_grid`.
- **Logit cache** — per-source-position logits for PE analysis are saved under `$PROJECTDIR/patch_logits/`. Controlled via `--cache_logits` (default: true) and `--logit_cache_dir` on `pe_analysis`. Set `--cache_logits=false` to disable.

To clear cached data, delete the relevant directories.

## Running experiments

### Step 1: Generate model outputs

Before running any patching experiment, you need to evaluate your model on GSM-8K to produce the output files that patching and PE analysis consume. Run the model with CoT and without CoT prompts separately:

```bash
# Generate CoT outputs
ri evaluate \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset gsm8k \
    --prompt_template gsm8k_cot \
    --batch_size 1 --max_gen_len 400 --seed 42 \
    --output_file outputs/single_batch_output_cot.json

# Generate non-CoT outputs
ri evaluate \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset gsm8k \
    --prompt_template gsm8k_non_cot \
    --batch_size 1 --max_gen_len 400 --seed 42 \
    --output_file outputs/single_batch_output_non_cot.json
```

**Note:** Batch size > 1 is not yet supported for patching experiments. Use `--batch_size 1`.

These output JSON files are then passed as `--source_dataset` and `--target_dataset` to the patching scripts below.

### Step 2: Run patching experiments

There are two experiment scripts for full reproducibility, plus a CLI for standalone runs.

Both `patch_position_sweep` and `pe_analysis` operate on a single sample (`--sample_idx`) and sweep across all layers and all target positions. For running patching across all samples at a fixed layer and position, use `run_patch_grid` (see below).

### Patch position sweep

Sweeps over source positions, target positions, and layers for one sample. For each combination, patches a single source CoT hidden state (with ```---patch_from_generation```) into the target model and saves the generated output.

```bash
python -m ri.entry_points.patch_position_sweep \
    --source_dataset ri/single_batch_output_cot.json \
    --target_dataset ri/single_batch_output_cot.json \
    --source_model_name meta-llama/Llama-3.1-8B-Instruct \
    --src_prompt_template gsm8k_cot \
    --tgt_prompt_template gsm8k_non_cot \
    --sample_idx 0 \
    --patch_from_generation \
    --output_dir patch_pos_sweep_results
```

Key arguments:
- `--layer` — patch a specific layer (otherwise sweeps all layers)
- `--start_layer`, `--layer_stride` — control layer sweep range
- `--target_pos` — patch at a specific target position (otherwise sweeps all)
- `--target_positions` — comma-separated list of target positions (e.g. `"0,-1"`)
- `--patch_from_generation` — extract source hidden states from generation rather than the prompt
- `--resume` — skip completed output files

### Patch effect (PE) analysis

Computes the patch effect metric across all layers and target positions for one sample. For each source token position, patches its hidden state from every layer into every target position and measures how the target model's output probability distribution changes.

The patch effect is defined as:

```
PE = (before_patch_target_prob - after_patch_target_prob) / max(after_patch_target_prob, 1e-10)
```

```bash
python -m ri.entry_points.pe_analysis \
    --source_dataset ri/single_batch_output_cot.json \
    --target_dataset ri/single_batch_output_cot.json \
    --source_model_name meta-llama/Llama-3.1-8B-Instruct \
    --sample_idx 0 \
    --output_dir pe_output
```

Key arguments:
- `--start_src_pos` — starting source position (supports negative indexing)
- `--target_positions` — comma-separated target positions (e.g. `"0,-1"`)
- `--cache_logits` — cache logits to disk for reuse across runs (default: true)
- `--resume` — skip completed source position files

### CLI for standalone experiments

The `ri` CLI provides subcommands for generation, patching, and causal mediation analysis without needing the full sweep scripts.

```bash
# Generate model outputs
ri evaluate \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset ri/single_batch_output_cot.json \
    --prompt_template gsm8k_cot \
    --batch_size 1 --max_gen_len 400 --seed 42 \
    --output_file output.json

# Single patching run
ri patch \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --source_dataset ri/single_batch_output_cot.json \
    --target_dataset ri/single_batch_output_cot.json \
    --src_prompt_template gsm8k_cot \
    --tgt_prompt_template gsm8k_non_cot \
    --source_layer 15 --target_layer 15 \
    --patch_from_generation \
    --output_file output_patched.json

# Causal mediation analysis
ri cma \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --source_dataset ri/single_batch_output_cot.json \
    --target_dataset ri/single_batch_output_cot.json \
    --source_layer 25 --target_layer 25 \
    --output_file patch_position_analysis.json
```

## Project structure

- `ri/core/` — model loading and forward hook infrastructure
- `ri/patching/` — patching pipeline, configuration, and tensor operations
- `ri/patching/cma/` — causal mediation analysis
- `ri/prompts/` — prompt templates and construction
- `ri/evaluation/` — generation and evaluation
- `ri/entry_points/` — experiment scripts (`patch_position_sweep`, `pe_analysis`, `run_patch_grid`)
- `ri/utils/` — tokenizer helpers, answer extraction
