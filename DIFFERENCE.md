# Differences: Our Production OPUS vs `AnotherImplementation`

This document compares:

- **Our implementation**: `production/` + `scripts/` in this repo.
- **Reference implementation**: `AnotherImplementation/`.

## Executive Summary

For your target (AWS cluster, DeepSpeed ZeRO-2, 1B -> 70B, MoE-aware path), **our implementation is the production-ready base**.

`AnotherImplementation` has a clean research core (especially FFT TensorSketch), but it is currently a standalone scoring library and is not integrated for your distributed training/runtime requirements.

## Side-by-Side Comparison

| Area | Our implementation (`production/`) | `AnotherImplementation/` |
|---|---|---|
| Primary goal | End-to-end production trainer path | Standalone OPUS scoring components |
| Training runtime | DeepSpeed ZeRO-2 integrated (`production/trainer.py`) | No DeepSpeed/ZeRO integration |
| Distributed selection | Yes: exact-global mode + fast-local mode | No distributed/global rank-aware selection |
| Optimizer scope | AdamW-only in production path | AdamW-inspired scoring utilities |
| Preconditioner used in scoring | Reads AdamW optimizer state directly (`AdamWPreconditionerView`) | Factored approximation (`compute_factored_preconditioner`) |
| ZeRO shard fidelity | Supports strict shard-only behavior (`strict_shard_preconditioner`) | No shard-aware optimizer-state handling |
| Ghost capture source | Real model hooks via `GhostCollector` (forward/backward pass) | Caller must provide `a`, `b`, `v` tensors manually |
| MoE routed expert support | Yes (routed `W_gate/W_up/W_down` scoring path) | No MoE routed expert path |
| Sketch operator | CountSketch projector with chunked outer accumulation | FFT TensorSketch (`IFFT(FFT(cs_b)*FFT(cs_a))`) |
| Sketch keying | Parameter/expert-unique sketch keys (layer/expert aware) | Per-layer operator (seeded), no expert-aware keyed variant |
| Selection sampler | Boltzmann-equivalent distributed Gumbel-Max | Softmax + `torch.multinomial` |
| Redundancy/history term | Implemented in iterative global selection | Implemented in iterative local selection |
| Failure handling | Timeout + numerical fallback to random selection | No equivalent production fallback policy |
| Nonfinite handling | Explicit sanitize and nonfinite metrics tracking | Not production-hardened to same level |
| Metrics/observability | Step timing logs, selector benchmark logs, overhead report tooling | No production overhead runbook/logging stack |
| Checkpoint/resume integration | Selector/proxy state checkpointed in trainer | No integrated trainer checkpoint flow |
| Paper-fidelity presets | Yes (`config.paper_fidelity.json`, random baseline pair) | No end-to-end preset/run pair in this repo |
| Local debugging tools | Selection report + quality scripts | Unit-style tests and scorer API tests |

## Paper Compliance Differences

- Our code follows the production AdamW path with optimizer-state-derived preconditioner and ZeRO-aware options, so it is closer to your required paper-faithful distributed setup.
- `AnotherImplementation` uses a **factored** preconditioner approximation for scoring. This can be useful for speed/research but is not the same as direct AdamW diagonal-state usage under ZeRO-2 sharding.

## Performance Differences

- `AnotherImplementation` FFT TensorSketch path may be fast in isolation for some layer shapes.
- Our code is optimized for end-to-end training reality: distributed comm patterns, shard-local preconditioner access, routed-MoE support, and operational fallbacks.
- For your workload, end-to-end throughput and stability matter more than isolated sketch microbench speed.

## Current Validation Status in This Repo

- Our production/local test suite currently passes (`tests/`).
- `AnotherImplementation` tests currently have failures in this repo state (API/test mismatch and behavior checks), so it should be treated as reference code, not release path.

## Recommendation

- Use **our `production/` path** for shared releases and cluster training.
- Treat `AnotherImplementation` as a research reference only, unless you explicitly decide to port specific ideas behind a feature flag and re-validate end-to-end.
