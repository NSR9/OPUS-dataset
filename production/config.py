from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class OpusConfig:
    enabled: bool = True
    selection_mode: str = "opus"  # {"opus", "random"}
    candidate_multiplier: int = 16
    selection_ratio: float = 0.5
    score_seq_len: int = 512
    proxy_batch_size: int = 8
    sketch_dim: int = 8192
    temperature: float = 0.9
    sketch_seed: int = 42
    fallback_random_on_error: bool = True
    max_selector_time_s: float = 30.0
    include_embeddings: bool = False
    include_lm_head: bool = False
    score_dtype: str = "bf16"
    track_nonfinite_stats: bool = True
    zero2_exact_global_scoring: bool = True
    strict_shard_preconditioner: bool = False
    benchmark_dual_mode: bool = False
    benchmark_every: int = 100
    benchmark_warmup_steps: int = 20
    benchmark_log_path: str = "selector_benchmark.jsonl"


@dataclass
class DataConfig:
    dataset_name: str = "PleIAs/SYNTH"
    local_path: str = "../synth_local_en"
    filter_language: str = "en"
    seed: int = 42
    num_workers: int = 2


@dataclass
class ModelConfig:
    target: str = "1b"
    embedding_type: str = "standard"
    vocab_size: int = 131072
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_real_experts: Optional[int] = None
    num_null_experts: Optional[int] = None
    top_k: Optional[int] = None
    data_sparsity: Optional[float] = None


@dataclass
class TrainConfig:
    total_steps: int = 10000
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip: float = 1.0
    seq_len_train: int = 4096
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    checkpoint_every: int = 250
    log_every: int = 10
    checkpoint_dir: str = "checkpoints_prod"
    step_metrics_log_path: str = "step_timing.jsonl"
    step_metrics_log_every: int = 1
    deepspeed_config: str = "production/deepspeed_zero2_config.json"
    use_bf16: bool = True


@dataclass
class ProductionConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    opus: OpusConfig = field(default_factory=OpusConfig)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["train"]["betas"] = list(self.train.betas)
        return out

    @staticmethod
    def load(path: str | Path) -> "ProductionConfig":
        import json

        obj = json.loads(Path(path).read_text())
        cfg = ProductionConfig()
        for section in ("model", "data", "train", "opus"):
            if section in obj:
                section_obj = getattr(cfg, section)
                for k, v in obj[section].items():
                    if hasattr(section_obj, k):
                        setattr(section_obj, k, v)
        if isinstance(cfg.train.betas, list):
            cfg.train.betas = (float(cfg.train.betas[0]), float(cfg.train.betas[1]))
        return cfg

    def save(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
