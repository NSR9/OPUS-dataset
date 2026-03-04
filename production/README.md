# Production AdamW-OPUS Trainer (DeepSpeed ZeRO-2)

This package provides a production training path separate from local sandbox scripts.

## Launch

```bash
deepspeed --num_gpus 8 production/train_prod.py --config production/config.prod.json
```

## Model Target Switching

Use `model.target` in config (or CLI override) to switch backbone:

- `1b`: uses `/Users/rohanshravan/Documents/New project/recurrence_model_1b.py`
- `70b`: uses `/Users/rohanshravan/Documents/New project/recurrence_model_70b.py`

CLI overrides:

```bash
deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json \
  --model-target 70b
```

Growth staging examples:

1. 1B dense baseline (8 layers):
```bash
deepspeed --num_gpus 8 production/train_prod.py --model-target 1b
```
2. 3B linear growth (20-layer dense path on 1B backbone):
```bash
deepspeed --num_gpus 8 production/train_prod.py --model-target 1b --num-layers 20
```
3. 8B MoW growth (MoE path on 70B backbone with reduced experts; tune overrides):
```bash
deepspeed --num_gpus 8 production/train_prod.py \
  --model-target 70b \
  --num-real-experts 64 \
  --top-k 8 \
  --data-sparsity 0.5
```
4. 70B target config:
```bash
deepspeed --num_gpus 8 production/train_prod.py --model-target 70b
```

## Stage-A vs Stage-B Proxy

- Stage-A (default): random in-distribution proxy stream.
- Stage-B: set `BENCH_PROXY_TOKENS` to a prebuilt token tensor:

```bash
export BENCH_PROXY_TOKENS=/path/to/bench_proxy_tokens.pt
deepspeed --num_gpus 8 production/train_prod.py --config production/config.prod.json
```

Build Stage-B shard from pre-scored JSONL:

```bash
python scripts/build_bench_proxy.py \
  --input-jsonl /path/scored_docs.jsonl \
  --output /path/bench_proxy_tokens.pt \
  --token-budget 30000000 \
  --seq-len 512
```

## Notes

- Existing `main.py` and `training.py` remain sandbox/test paths.
- OPUS is AdamW-only in this production path.
- `opus.selection_mode=opus|random` switches between OPUS scoring and random baseline selection (same candidate buffer contract).
- Routed MoE expert capture for `W_gate` / `W_up` / `W_down` is enabled in OPUS scoring.
- CountSketch operators are parameter-unique (layer/expert keyed), not shape-only, to match per-block projection semantics.
- Scoring pass skips MTP compute (`next_token_ids=None`) to reduce selector overhead.
- Global sampling uses distributed Gumbel-Max (Boltzmann-equivalent) with scalar collectives instead of full-score all-gather.
- `opus.track_nonfinite_stats=false` disables expensive nonfinite counting in production (still sanitizes values).
- `opus.strict_shard_preconditioner=true` uses zero contribution when optimizer shard state is absent (no scalar hidden fallback).
- `opus.zero2_exact_global_scoring=true` uses shared scoring candidates across ranks and all-reduced utilities for ZeRO-2-faithful AdamW geometry.
- Default selector settings are in `production/config.prod.json`.
- Step-level end-to-end timing logs are written to `train.step_metrics_log_path` (default every step).

## Paper-Fidelity Presets

Paper-comparable OPUS knobs are preset in:

- `production/config.paper_fidelity.json` (OPUS mode)
- `production/config.paper_baseline_random.json` (random baseline mode)

These use:

- `N=32` (`candidate_multiplier=32` with micro-batch 1)
- `rho=0.5`
- `L_score=512`
- `K_proxy=8`
- `tau=0.9`
- `m=8192`

Run pair for overhead measurement:

```bash
deepspeed --num_gpus 8 production/train_prod.py --config production/config.paper_baseline_random.json
deepspeed --num_gpus 8 production/train_prod.py --config production/config.paper_fidelity.json
```

Then compute paper-style overhead:

```bash
python scripts/report_opus_overhead.py \
  --baseline-log step_timing_paper_random.jsonl \
  --opus-log step_timing_paper_opus.jsonl \
  --warmup-steps 20 \
  --output-json paper_overhead_report.json
```

## Dual-Mode Selector Benchmark (exact vs fast-local)

Enable runtime benchmark logging:

```bash
deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json \
  --benchmark-dual-mode 1 \
  --benchmark-every 50 \
  --benchmark-warmup-steps 20 \
  --benchmark-log-path /tmp/selector_benchmark.jsonl
```

What it does:

- Primary selector runs normally (`opus.zero2_exact_global_scoring` config).
- Shadow selector runs the opposite mode on the same sketch features.
- Rank0 logs one JSON line per benchmark step with selector times and step timing.

Important interpretation:

- On multi-rank runs, exact-global vs fast-local are not semantically equivalent.
- `semantics_valid=false` in the log means timing comparison is still useful, but selected ids/utility are not paper-equivalent.

## 70B Overhead Clarification (paper vs local estimate)

Paper’s `4.7%` is an empirical end-to-end compute delta (PFLOPs) against random sampling. It is not a simple token-ratio formula.

For our 70B planning, we should report two numbers:

1. Analytical approximation (before long run):
   `overhead_est ~= scoring_cost_per_step / baseline_train_cost_per_step`
2. Measured training overhead (after run):
   `overhead_measured = (total_step_time_opus - total_step_time_random) / total_step_time_random`

Practical approximation from runtime logs:

- Baseline run (random selector): median `step_total_s` = `T_base`
- OPUS run: median `step_total_s` = `T_opus`
- Estimated overhead: `(T_opus - T_base) / T_base`

Why keep both:

- Approximation helps capacity planning.
- Measured value is the source-of-truth for production and paper-style reporting.

## Local Mac Selection Debug

Run one OPUS selection round locally and export selected vs dropped samples:

```bash
python scripts/local_opus_selection_report.py \
  --dataset-path /Users/rohanshravan/TSAI/synth_local_en \
  --checkpoint-path checkpoints/baseline_checkpoint_step_0012000.pt \
  --golden-path examples/golden_samples_template.jsonl \
  --output-json selection_report.json \
  --output-csv selection_report.csv
```

Golden sample templates:

- `examples/golden_samples_template.jsonl`
- `examples/golden_samples_template.csv`
