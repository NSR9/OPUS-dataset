# OPUS Training Stack (AdamW, ZeRO-2, 1B -> 70B)

This repo has two paths:

- `production/`: production trainer path (DeepSpeed ZeRO-2, AdamW-only, OPUS selector).
- `main.py` / `training.py`: local sandbox/test path (not production runtime contract).

## What Is Implemented

- AdamW-only OPUS scoring (no Muon path).
- Ghost-factor capture + CountSketch projection.
- ZeRO-2 compatible preconditioner usage.
- Exact-global selector mode for distributed paper-faithful scoring.
- Fast-local selector mode for speed benchmarking.
- MoE routed-expert scoring path (`W_gate`, `W_up`, `W_down`) for 70B model.

## Repo Map

- `production/train_prod.py`: production entrypoint.
- `production/trainer.py`: training loop orchestration.
- `production/opus/`: OPUS selector internals.
- `production/config.prod.json`: default production config.
- `production/config.paper_fidelity.json`: paper-comparable OPUS preset.
- `production/config.paper_baseline_random.json`: paper-comparable random baseline preset.
- `scripts/local_opus_selection_report.py`: local sample-level selection report.
- `scripts/selection_report_quality.py`: report quality summary.
- `scripts/report_opus_overhead.py`: OPUS vs random overhead comparison.
- `scripts/build_bench_proxy.py`: Stage-B proxy shard builder.

## Environment

Run from repo root:

```bash
cd "/Users/rohanshravan/Documents/New project"
```

Recommended env (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets deepspeed
```

If module imports fail, prefix commands with `PYTHONPATH=.`.

## Local Mac: Inspect Which Samples OPUS Picks

Use this for quick debugging and visibility into selected vs removed candidates.

```bash
PYTHONPATH=. python scripts/local_opus_selection_report.py \
  --dataset-path "/Users/rohanshravan/TSAI/synth_local_en" \
  --checkpoint-path "checkpoints/baseline_checkpoint_step_0012000.pt" \
  --golden-path "examples/golden_samples_template.jsonl" \
  --output-json "selection_report.json" \
  --output-csv "selection_report.csv"
```

Random-init smoke mode (ignore checkpoint):

```bash
PYTHONPATH=. python scripts/local_opus_selection_report.py \
  --dataset-path "/Users/rohanshravan/TSAI/synth_local_en" \
  --random-init \
  --embedding-type kronecker \
  --seq-len 96 \
  --score-seq-len 64 \
  --candidates 4 \
  --proxy-size 2 \
  --sketch-dim 512 \
  --device cpu \
  --output-json "/tmp/selection_report_smoke.json" \
  --output-csv "/tmp/selection_report_smoke.csv"
```

Summarize report quality:

```bash
python scripts/selection_report_quality.py --report-json selection_report.json --top-k 8
```

## Production ZeRO-2 Training

Default production run:

```bash
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json
```

Switch model target:

```bash
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json \
  --model-target 70b
```

### Stage-A vs Stage-B Proxy

- Stage-A (default): random in-distribution proxy stream.
- Stage-B: set `BENCH_PROXY_TOKENS` to a token tensor shard.

Build Stage-B token shard:

```bash
python scripts/build_bench_proxy.py \
  --input-jsonl /path/scored_docs.jsonl \
  --output /path/bench_proxy_tokens.pt \
  --token-budget 30000000 \
  --seq-len 512
```

Run with Stage-B:

```bash
export BENCH_PROXY_TOKENS=/path/bench_proxy_tokens.pt
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json
```

## Paper-Fidelity Runs (Recommended For Comparability)

These presets align with:

- `N=32`
- `rho=0.5`
- `L_score=512`
- `K_proxy=8`
- `tau=0.9`
- `m=8192`

Random baseline run:

```bash
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.paper_baseline_random.json
```

OPUS run:

```bash
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.paper_fidelity.json
```

## Paper-Style Overhead Report (End-to-End)

Compare total step time between random baseline and OPUS:

```bash
python scripts/report_opus_overhead.py \
  --baseline-log step_timing_paper_random.jsonl \
  --opus-log step_timing_paper_opus.jsonl \
  --warmup-steps 20 \
  --output-json paper_overhead_report.json
```

This script reports:

- `paper_style_overhead_fraction`
- `paper_style_overhead_percent`
- timing summaries for scoring/feature/selector/train components

## Dual-Mode Selector Benchmark (Exact vs Fast-Local)

Use this only to compare selector runtime modes:

```bash
PYTHONPATH=. deepspeed --num_gpus 8 production/train_prod.py \
  --config production/config.prod.json \
  --benchmark-dual-mode 1 \
  --benchmark-every 50 \
  --benchmark-warmup-steps 20 \
  --benchmark-log-path /tmp/selector_benchmark.jsonl
```

Note: exact-global and fast-local are not semantically equivalent on multi-rank runs; use this for timing, not result parity.

## Golden Samples

Template files:

- `examples/golden_samples_template.jsonl`
- `examples/golden_samples_template.csv`

These are local diagnostics. They are not Bench-Proxy retrieval shards used by paper-style training.

## Notes

- OPUS in this repo is AdamW-only.
- For production, use `production/` path only.
- Existing checkpoints/sandbox scripts remain useful for local debugging, not as production runtime guarantees.
