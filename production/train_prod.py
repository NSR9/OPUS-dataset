#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from production.config import ProductionConfig
from production.trainer import ProductionTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Production trainer with AdamW OPUS + DeepSpeed ZeRO-2")
    p.add_argument("--config", type=str, default="production/config.prod.json", help="Path to production JSON config")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--step-metrics-log-path", type=str, default=None)
    p.add_argument("--step-metrics-log-every", type=int, default=None)
    p.add_argument("--model-target", type=str, default=None, choices=["1b", "70b"])
    p.add_argument("--embedding-type", type=str, default=None, choices=["standard", "kronecker"])
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-real-experts", type=int, default=None)
    p.add_argument("--num-null-experts", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--data-sparsity", type=float, default=None)
    p.add_argument("--selection-mode", type=str, default=None, choices=["opus", "random"])
    p.add_argument("--zero2-exact-global-scoring", type=int, default=None, choices=[0, 1])
    p.add_argument("--strict-shard-preconditioner", type=int, default=None, choices=[0, 1])
    p.add_argument("--benchmark-dual-mode", type=int, default=None, choices=[0, 1])
    p.add_argument("--benchmark-every", type=int, default=None)
    p.add_argument("--benchmark-warmup-steps", type=int, default=None)
    p.add_argument("--benchmark-log-path", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = ProductionConfig.load(cfg_path)
    else:
        cfg = ProductionConfig()
        cfg.save(cfg_path)

    if args.total_steps is not None:
        cfg.train.total_steps = int(args.total_steps)
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = str(args.checkpoint_dir)
    if args.step_metrics_log_path is not None:
        cfg.train.step_metrics_log_path = str(args.step_metrics_log_path)
    if args.step_metrics_log_every is not None:
        cfg.train.step_metrics_log_every = int(args.step_metrics_log_every)
    if args.model_target is not None:
        cfg.model.target = str(args.model_target)
    if args.embedding_type is not None:
        cfg.model.embedding_type = str(args.embedding_type)
    if args.num_layers is not None:
        cfg.model.num_layers = int(args.num_layers)
    if args.num_real_experts is not None:
        cfg.model.num_real_experts = int(args.num_real_experts)
    if args.num_null_experts is not None:
        cfg.model.num_null_experts = int(args.num_null_experts)
    if args.top_k is not None:
        cfg.model.top_k = int(args.top_k)
    if args.data_sparsity is not None:
        cfg.model.data_sparsity = float(args.data_sparsity)
    if args.selection_mode is not None:
        cfg.opus.selection_mode = str(args.selection_mode)
    if args.zero2_exact_global_scoring is not None:
        cfg.opus.zero2_exact_global_scoring = bool(int(args.zero2_exact_global_scoring))
    if args.strict_shard_preconditioner is not None:
        cfg.opus.strict_shard_preconditioner = bool(int(args.strict_shard_preconditioner))
    if args.benchmark_dual_mode is not None:
        cfg.opus.benchmark_dual_mode = bool(int(args.benchmark_dual_mode))
    if args.benchmark_every is not None:
        cfg.opus.benchmark_every = int(args.benchmark_every)
    if args.benchmark_warmup_steps is not None:
        cfg.opus.benchmark_warmup_steps = int(args.benchmark_warmup_steps)
    if args.benchmark_log_path is not None:
        cfg.opus.benchmark_log_path = str(args.benchmark_log_path)

    trainer = ProductionTrainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
