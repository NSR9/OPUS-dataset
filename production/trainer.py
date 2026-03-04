from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from data import SYNTHStream, create_bpe_token_strings
from production.config import ProductionConfig
from production.distributed import (
    all_reduce_max,
    barrier,
    broadcast_tensor,
    get_rank,
    get_world_size,
    init_distributed,
    is_rank0,
    set_seed,
)
from production.opus import (
    AdamWPreconditionerView,
    BenchProxyProvider,
    GhostCollector,
    OpusSelector,
    RandomInDistributionProxyProvider,
    SelectionResult,
)
from recurrence_model_1b import (
    Model1B,
)
from recurrence_model_1b import (
    ModelConfig as Model1BConfig,
)
from recurrence_model_1b import (
    PFCodec as PFCodec1B,
)
from recurrence_model_1b import (
    PFConfig as PFConfig1B,
)
from recurrence_model_70b import (
    Model70B,
)
from recurrence_model_70b import (
    ModelConfig as Model70BConfig,
)
from recurrence_model_70b import (
    PFCodec as PFCodec70B,
)
from recurrence_model_70b import (
    PFConfig as PFConfig70B,
)
from training import setup_tokenizer


class _RankShardedIterator:
    """Simple batch-level sharding for iterable loaders."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.rank = get_rank()
        self.world = get_world_size()

    def __iter__(self):
        for i, batch in enumerate(iter(self.loader)):
            if i % self.world == self.rank:
                yield batch


class ProductionTrainer:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.rank = get_rank()
        self.world = get_world_size()
        self.logger = logging.getLogger("production_trainer")

        self.model_engine = None
        self.optimizer = None
        self.lr_scheduler = None

        self.train_loader = None
        self.train_loader_unsharded = None
        self.proxy_provider = None
        self.selector = OpusSelector(config.opus)
        self.selector_benchmark_other = None
        self.preconditioner_view = None
        self.benchmark_log_path = Path(self.config.opus.benchmark_log_path)
        self.step_metrics_log_path = Path(self.config.train.step_metrics_log_path)

        self.global_step = 0
        self.ignore_index = -100

    @property
    def device(self) -> torch.device:
        if self.model_engine is not None:
            return self.model_engine.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _build_logger(self) -> None:
        level = logging.INFO
        fmt = "%(asctime)s | %(levelname)s | %(message)s"
        logging.basicConfig(level=level, format=fmt)

    @staticmethod
    def _selector_mode_name(exact: bool) -> str:
        return "exact_global" if exact else "fast_local"

    def _selection_mode(self) -> str:
        mode = str(self.config.opus.selection_mode).strip().lower()
        if mode not in {"opus", "random"}:
            raise ValueError(
                f"Unsupported opus.selection_mode={self.config.opus.selection_mode!r}. Use 'opus' or 'random'."
            )
        return mode

    def _dual_benchmark_enabled(self, step: int) -> bool:
        if not bool(self.config.opus.benchmark_dual_mode):
            return False
        if step < int(self.config.opus.benchmark_warmup_steps):
            return False
        every = max(1, int(self.config.opus.benchmark_every))
        return (step % every) == 0

    def _write_benchmark_row(self, row: Dict[str, Any]) -> None:
        if not is_rank0():
            return
        self.benchmark_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.benchmark_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _write_step_metrics_row(self, row: Dict[str, Any]) -> None:
        if not is_rank0():
            return
        every = max(1, int(self.config.train.step_metrics_log_every))
        step = int(row.get("step", 0))
        if (step % every) != 0:
            return
        self.step_metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.step_metrics_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _load_deepspeed_config(self) -> Dict:
        cfg_path = Path(self.config.train.deepspeed_config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {cfg_path}")
        ds = json.loads(cfg_path.read_text())

        ds["train_micro_batch_size_per_gpu"] = int(self.config.train.micro_batch_size)
        ds["gradient_accumulation_steps"] = int(self.config.train.grad_accum_steps)
        ds.setdefault("gradient_clipping", float(self.config.train.grad_clip))

        ds.setdefault("optimizer", {})
        ds["optimizer"]["type"] = "AdamW"
        ds["optimizer"]["params"] = {
            "lr": float(self.config.train.learning_rate),
            "betas": list(self.config.train.betas),
            "eps": float(self.config.train.eps),
            "weight_decay": float(self.config.train.weight_decay),
        }

        if self.config.train.use_bf16:
            ds.setdefault("bf16", {})
            ds["bf16"]["enabled"] = True

        return ds

    def _model_target(self) -> str:
        target = str(self.config.model.target).strip().lower()
        if target not in {"1b", "70b"}:
            raise ValueError(
                f"Unsupported model.target={self.config.model.target!r}. Supported: '1b', '70b'"
            )
        return target

    @staticmethod
    def _recompute_layer_mix(cfg: Any) -> None:
        if not hasattr(cfg, "num_layers"):
            return
        n_layers = int(getattr(cfg, "num_layers"))
        if n_layers <= 0:
            raise ValueError(f"num_layers must be >0, got {n_layers}")
        if hasattr(cfg, "num_gsa_layers") and hasattr(cfg, "num_deltanet_layers"):
            gsa = max(1, int(round(0.25 * n_layers)))
            gsa = min(gsa, max(1, n_layers - 1))
            delta = max(1, n_layers - gsa)
            setattr(cfg, "num_gsa_layers", gsa)
            setattr(cfg, "num_deltanet_layers", delta)

    def _apply_model_overrides(self, cfg: Any) -> Any:
        mc = self.config.model
        cfg.vocab_size = int(mc.vocab_size)

        if mc.hidden_size is not None:
            cfg.hidden_size = int(mc.hidden_size)

        if mc.num_layers is not None:
            cfg.num_layers = int(mc.num_layers)
            self._recompute_layer_mix(cfg)

        if mc.num_real_experts is not None and hasattr(cfg, "num_real_experts"):
            cfg.num_real_experts = int(mc.num_real_experts)

        if mc.top_k is not None and hasattr(cfg, "top_k"):
            cfg.top_k = int(mc.top_k)

        if mc.data_sparsity is not None and hasattr(cfg, "data_sparsity"):
            cfg.data_sparsity = float(mc.data_sparsity)

        if mc.num_null_experts is not None and hasattr(cfg, "num_null_experts"):
            cfg.num_null_experts = int(mc.num_null_experts)
        elif (
            hasattr(cfg, "num_real_experts")
            and hasattr(cfg, "num_null_experts")
            and hasattr(cfg, "data_sparsity")
            and float(getattr(cfg, "data_sparsity")) > 0.0
            and float(getattr(cfg, "data_sparsity")) < 1.0
        ):
            num_real = int(getattr(cfg, "num_real_experts"))
            rho = float(getattr(cfg, "data_sparsity"))
            cfg.num_null_experts = int(round(num_real * (1.0 - rho) / rho))

        if (
            hasattr(cfg, "num_real_experts")
            and hasattr(cfg, "num_null_experts")
            and hasattr(cfg, "total_expert_slots")
        ):
            cfg.total_expert_slots = int(getattr(cfg, "num_real_experts")) + int(
                getattr(cfg, "num_null_experts")
            )

        return cfg

    def _build_kronecker_resources(
        self,
        tokenizer: GPT2Tokenizer,
        pf_config_cls: type,
        pf_codec_cls: type,
    ) -> Tuple[Any, list[str]]:
        # Keep PF config explicit for deterministic embedding construction across stages.
        pf_cfg = pf_config_cls(
            CHAR_DIM=256,
            POS_DIM=32,
            D=8192,
            length_normalize=True,
            truncate_long_words=True,
        )
        pf_codec = pf_codec_cls(pf_cfg)
        bpe_vocab = create_bpe_token_strings(
            tokenizer, vocab_size=self.config.model.vocab_size
        )
        return pf_codec, bpe_vocab

    def _build_model(self, tokenizer: GPT2Tokenizer) -> nn.Module:
        target = self._model_target()
        embedding = str(self.config.model.embedding_type).strip().lower()

        if target == "1b":
            cfg = self._apply_model_overrides(Model1BConfig())
            bpe_vocab = None
            pf_codec = None
            if embedding == "kronecker":
                pf_codec, bpe_vocab = self._build_kronecker_resources(
                    tokenizer, PFConfig1B, PFCodec1B
                )
            model = Model1B(
                cfg, embedding_type=embedding, bpe_vocab=bpe_vocab, pf_codec=pf_codec
            )
        else:
            cfg = self._apply_model_overrides(Model70BConfig())
            bpe_vocab = None
            pf_codec = None
            if embedding == "kronecker":
                pf_codec, bpe_vocab = self._build_kronecker_resources(
                    tokenizer, PFConfig70B, PFCodec70B
                )
            model = Model70B(
                cfg, embedding_type=embedding, bpe_vocab=bpe_vocab, pf_codec=pf_codec
            )

        if is_rank0():
            self.logger.info(
                "Model target=%s embedding=%s vocab=%d hidden=%s layers=%s experts=%s top_k=%s rho=%s",
                target,
                embedding,
                int(self.config.model.vocab_size),
                str(getattr(model.config, "hidden_size", "n/a")),
                str(getattr(model.config, "num_layers", "n/a")),
                str(getattr(model.config, "num_real_experts", "n/a")),
                str(getattr(model.config, "top_k", "n/a")),
                str(getattr(model.config, "data_sparsity", "n/a")),
            )
        return model

    def _build_tokenizer(self) -> GPT2Tokenizer:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer = setup_tokenizer(
            tokenizer, target_vocab_size=self.config.model.vocab_size
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _build_loaders(self, tokenizer: GPT2Tokenizer) -> Tuple[DataLoader, DataLoader]:
        ds_train = SYNTHStream(
            tokenizer=tokenizer,
            dataset_name=self.config.data.dataset_name,
            local_path=self.config.data.local_path,
            seq_len=self.config.train.seq_len_train,
            batch_size=self.config.train.micro_batch_size,
            seed=self.config.data.seed,
            filter_language=self.config.data.filter_language,
            start_step=0,
        )
        train_loader = DataLoader(
            ds_train,
            batch_size=self.config.train.micro_batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        ds_proxy = SYNTHStream(
            tokenizer=tokenizer,
            dataset_name=self.config.data.dataset_name,
            local_path=self.config.data.local_path,
            seq_len=max(self.config.opus.score_seq_len, 16),
            batch_size=max(self.config.opus.proxy_batch_size, 1),
            seed=self.config.data.seed + 103,
            filter_language=self.config.data.filter_language,
            start_step=0,
        )
        proxy_loader = DataLoader(
            ds_proxy,
            batch_size=max(self.config.opus.proxy_batch_size, 1),
            num_workers=self.config.data.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        return train_loader, proxy_loader

    def _broadcast_token_batch(
        self, batch: Optional[torch.Tensor], src: int = 0
    ) -> torch.Tensor:
        if get_world_size() == 1:
            assert batch is not None
            return batch
        device = self.device
        if get_rank() == src:
            assert batch is not None
            shape = torch.tensor(
                [batch.size(0), batch.size(1)], device=device, dtype=torch.long
            )
            payload = batch.to(device=device, dtype=torch.long, non_blocking=True)
        else:
            shape = torch.zeros(2, device=device, dtype=torch.long)
            payload = torch.empty(1, 1, device=device, dtype=torch.long)
        broadcast_tensor(shape, src=src)
        rows, cols = int(shape[0].item()), int(shape[1].item())
        if get_rank() != src:
            payload = torch.empty(rows, cols, device=device, dtype=torch.long)
        broadcast_tensor(payload, src=src)
        return payload

    def setup(self) -> None:
        self._build_logger()
        init_distributed()
        set_seed(self.config.data.seed)

        tokenizer = self._build_tokenizer()
        model = self._build_model(tokenizer)
        train_loader, proxy_loader = self._build_loaders(tokenizer)

        if self._model_target() == "70b" and is_rank0():
            self.logger.warning(
                "70B MoE path enabled: routed-expert OPUS capture for W_gate/W_up/W_down is active. "
                "Monitor selector_time_s and nonfinite metrics during bring-up."
            )

        # Batch-level sharding keeps rank streams disjoint for iterable dataset usage.
        self.train_loader = _RankShardedIterator(train_loader)
        self.train_loader_unsharded = train_loader

        stage_b_path = os.environ.get("BENCH_PROXY_TOKENS", "")
        if stage_b_path:
            self.proxy_provider = BenchProxyProvider(stage_b_path)
            if is_rank0():
                self.logger.info("Using Bench-Proxy provider from %s", stage_b_path)
        else:
            self.proxy_provider = RandomInDistributionProxyProvider(proxy_loader)
            if is_rank0():
                self.logger.info("Using Stage-A random in-distribution proxy provider")

        ds_cfg = self._load_deepspeed_config()
        import deepspeed

        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_cfg,
        )
        self.model_engine = engine
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.preconditioner_view = AdamWPreconditionerView(
            self.optimizer,
            strict_shard_only=bool(self.config.opus.strict_shard_preconditioner),
        )
        if (
            bool(self.config.opus.benchmark_dual_mode)
            and self._selection_mode() == "opus"
        ):
            alt_cfg = replace(
                self.config.opus,
                zero2_exact_global_scoring=not bool(
                    self.config.opus.zero2_exact_global_scoring
                ),
            )
            self.selector_benchmark_other = OpusSelector(alt_cfg)
            if is_rank0():
                self.logger.info(
                    "Dual selector benchmark enabled: primary=%s shadow=%s every=%d warmup=%d log=%s",
                    self._selector_mode_name(
                        bool(self.config.opus.zero2_exact_global_scoring)
                    ),
                    self._selector_mode_name(bool(alt_cfg.zero2_exact_global_scoring)),
                    int(self.config.opus.benchmark_every),
                    int(self.config.opus.benchmark_warmup_steps),
                    str(self.benchmark_log_path),
                )
        elif bool(self.config.opus.benchmark_dual_mode) and is_rank0():
            self.logger.info(
                "Dual selector benchmark is disabled for selection_mode=random"
            )

        self._try_resume()

    def _checkpoint_tag(self, step: int) -> str:
        return f"step_{step:08d}"

    def _try_resume(self) -> None:
        ckpt_dir = Path(self.config.train.checkpoint_dir)
        if not ckpt_dir.exists():
            return
        load_path, client_state = self.model_engine.load_checkpoint(str(ckpt_dir))
        if load_path is None:
            return
        self.global_step = int(client_state.get("step", 0))
        if "selector_state" in client_state:
            self.selector.load_state_dict(client_state["selector_state"])
        if "proxy_state" in client_state:
            self.proxy_provider.load_state_dict(client_state["proxy_state"])
        if is_rank0():
            self.logger.info("Resumed from %s at step %d", load_path, self.global_step)

    def _save_checkpoint(self, step: int, extra_metrics: Dict[str, float]) -> None:
        ckpt_dir = Path(self.config.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        client_state = {
            "step": int(step),
            "selector_state": self.selector.state_dict(),
            "proxy_state": self.proxy_provider.state_dict(),
            "config": self.config.to_dict(),
            "metrics": extra_metrics,
        }
        self.model_engine.save_checkpoint(
            str(ckpt_dir), tag=self._checkpoint_tag(step), client_state=client_state
        )

    @staticmethod
    def _shift_for_mtp(
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.size(1) < 3:
            raise ValueError("Need sequence length >= 3 for MTP/NTP loss")
        x = input_ids[:, :-2].contiguous()
        y_ntp = input_ids[:, 1:-1].contiguous()
        y_mtp = input_ids[:, 2:].contiguous()
        return x, y_ntp, y_mtp

    def _compute_loss(
        self, input_ids: torch.Tensor, scoring: bool = False
    ) -> torch.Tensor:
        x, y_ntp, y_mtp = self._shift_for_mtp(input_ids)
        if scoring:
            # OPUS scoring does not need MTP logits; skip this branch to cut selector overhead.
            # Don't pass targets - we need logits for scoring, not fused loss
            logits_ntp, logits_mtp, aux_loss = self.model_engine(
                x,
                next_token_ids=None,
                return_memory=False,
                return_loss=True,
            )
            ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
            loss_ntp = ce(logits_ntp.view(-1, logits_ntp.size(-1)), y_ntp.view(-1))
            return loss_ntp + aux_loss
        else:
            # Use fused linear+CE path - returns scalar losses directly
            loss_ntp, loss_mtp, aux_loss = self.model_engine(
                x,
                next_token_ids=y_ntp,
                ntp_targets=y_ntp,
                mtp_targets=y_mtp,
                return_memory=False,
                return_loss=True,
            )
            if loss_mtp is None:
                return loss_ntp + aux_loss
            else:
                return loss_ntp + 0.3 * loss_mtp + aux_loss

    def _next_candidate_buffer(
        self, train_iter: Iterator
    ) -> Tuple[torch.Tensor, Iterator]:
        chunks = []
        for _ in range(self.config.opus.candidate_multiplier):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            chunks.append(batch["input_ids"])
        buf = torch.cat(chunks, dim=0)
        if buf.size(1) > self.config.train.seq_len_train:
            buf = buf[:, : self.config.train.seq_len_train]
        return buf.to(self.device, non_blocking=True), train_iter

    def _pad_rows_same_across_ranks(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        local_n = torch.tensor([x.size(0)], device=x.device, dtype=torch.long)
        max_n = all_reduce_max(local_n.clone())
        target = int(max_n.item())
        if target == x.size(0):
            return x, x.size(0)
        pad_rows = target - x.size(0)
        pad = torch.zeros(pad_rows, x.size(1), dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=0), x.size(0)

    def _mask_padded_rows(self, y: torch.Tensor, valid_rows: int) -> torch.Tensor:
        if valid_rows >= y.size(0):
            return y
        y = y.clone()
        y[valid_rows:] = self.ignore_index
        return y

    def train(self) -> None:
        if self.model_engine is None:
            self.setup()

        train_iter = iter(self.train_loader)
        train_iter_unsharded = (
            iter(self.train_loader_unsharded)
            if self.train_loader_unsharded is not None
            else None
        )

        for step in range(self.global_step, self.config.train.total_steps):
            t0 = time.perf_counter()
            self.preconditioner_view.refresh()
            selection_mode = self._selection_mode()
            shadow_metrics = None
            t_scoring_start = t0
            t_scoring_end = t0
            t_features_start = t0
            t_features_end = t0
            t_select_start = t0
            t_select_end = t0
            score_seq_len = 0

            if selection_mode == "opus":
                if self.config.opus.zero2_exact_global_scoring and get_world_size() > 1:
                    if is_rank0():
                        assert train_iter_unsharded is not None
                        candidate_ids_rank0, train_iter_unsharded = (
                            self._next_candidate_buffer(train_iter_unsharded)
                        )
                    else:
                        candidate_ids_rank0 = None
                    candidate_ids = self._broadcast_token_batch(
                        candidate_ids_rank0, src=0
                    )

                    if is_rank0():
                        proxy_rank0 = self.proxy_provider.sample(
                            device=self.device,
                            k=self.config.opus.proxy_batch_size,
                            seq_len=min(
                                self.config.opus.score_seq_len, candidate_ids.size(1)
                            ),
                        )
                    else:
                        proxy_rank0 = None
                    proxy_ids = self._broadcast_token_batch(proxy_rank0, src=0)
                else:
                    candidate_ids, train_iter = self._next_candidate_buffer(train_iter)
                    proxy_ids = self.proxy_provider.sample(
                        device=self.device,
                        k=self.config.opus.proxy_batch_size,
                        seq_len=min(
                            self.config.opus.score_seq_len, candidate_ids.size(1)
                        ),
                    )

                n_local = candidate_ids.size(0)

                score_seq_len = min(
                    self.config.opus.score_seq_len,
                    candidate_ids.size(1),
                    proxy_ids.size(1),
                )
                score_batch = torch.cat(
                    [candidate_ids[:, :score_seq_len], proxy_ids[:, :score_seq_len]],
                    dim=0,
                )
                t_scoring_start = time.perf_counter()

                with GhostCollector(
                    self.model_engine.module,
                    include_embeddings=self.config.opus.include_embeddings,
                    include_lm_head=self.config.opus.include_lm_head,
                ) as collector:
                    self.model_engine.zero_grad(set_to_none=True)
                    score_loss = self._compute_loss(score_batch, scoring=True)
                    self.model_engine.backward(score_loss)
                    captures = collector.captures()
                    moe_captures = collector.moe_captures()
                    self.model_engine.zero_grad(set_to_none=True)
                t_scoring_end = time.perf_counter()

                t_features_start = time.perf_counter()
                c_feats, p_feats = self.selector.build_sketch_features(
                    captures=captures,
                    candidate_count=n_local,
                    proxy_count=self.config.opus.proxy_batch_size,
                    preconditioner=self.preconditioner_view,
                    moe_captures=moe_captures,
                    out_dtype=torch.float32,
                )
                captures.clear()
                moe_captures.clear()
                t_features_end = time.perf_counter()

                lr = float(self.optimizer.param_groups[0]["lr"])
                t_select_start = time.perf_counter()
                sel = self.selector.select(
                    candidate_features=c_feats,
                    proxy_features=p_feats,
                    learning_rate=lr,
                )
                t_select_end = time.perf_counter()

                if (
                    self.selector_benchmark_other is not None
                    and self._dual_benchmark_enabled(step)
                ):
                    # Keep RNG streams aligned for fair comparison.
                    self.selector_benchmark_other.load_state_dict(
                        self.selector.state_dict()
                    )
                    t_shadow_start = time.perf_counter()
                    shadow_sel = self.selector_benchmark_other.select(
                        candidate_features=c_feats,
                        proxy_features=p_feats,
                        learning_rate=lr,
                    )
                    t_shadow_end = time.perf_counter()
                    primary_exact = bool(self.config.opus.zero2_exact_global_scoring)
                    shadow_exact = not primary_exact
                    semantics_valid = bool(
                        get_world_size() == 1 or primary_exact == shadow_exact
                    )
                    shadow_metrics = {
                        "mode_primary": self._selector_mode_name(primary_exact),
                        "mode_shadow": self._selector_mode_name(shadow_exact),
                        "selector_primary_s": float(
                            sel.metrics.get(
                                "selector_time_s", t_select_end - t_select_start
                            )
                        ),
                        "selector_shadow_s": float(
                            shadow_sel.metrics.get(
                                "selector_time_s", t_shadow_end - t_shadow_start
                            )
                        ),
                        "semantics_valid": bool(semantics_valid),
                    }
            else:
                candidate_ids, train_iter = self._next_candidate_buffer(train_iter)
                n_local = candidate_ids.size(0)
                k_local = max(1, int(round(self.config.opus.selection_ratio * n_local)))
                t_select_start = time.perf_counter()
                local_idxs = torch.randperm(n_local, device=self.device)[:k_local]
                t_select_end = time.perf_counter()
                sel = SelectionResult(
                    selected_local_indices=local_idxs.to(torch.long),
                    selected_global_indices=(local_idxs + (get_rank() * n_local)).to(
                        torch.long
                    ),
                    used_fallback=False,
                    metrics={
                        "alignment": 0.0,
                        "redundancy": 0.0,
                        "entropy": 0.0,
                        "nonfinite_feature_values": 0.0,
                        "nonfinite_local_score_values": 0.0,
                        "fallback_no_finite_scores": 0.0,
                        "selector_time_s": float(t_select_end - t_select_start),
                    },
                )

            t_train_start = time.perf_counter()
            selected = candidate_ids[sel.selected_local_indices]
            selected, valid_rows = self._pad_rows_same_across_ranks(selected)

            x, y_ntp_raw, y_mtp_raw = self._shift_for_mtp(selected)
            y_ntp = self._mask_padded_rows(y_ntp_raw, valid_rows)
            y_mtp = self._mask_padded_rows(y_mtp_raw, valid_rows)

            # Use fused linear+CE path - returns scalar losses directly
            loss_ntp, loss_mtp, aux_loss = self.model_engine(
                x,
                next_token_ids=y_ntp_raw,
                ntp_targets=y_ntp,
                mtp_targets=y_mtp,
                return_memory=False,
                return_loss=True,
            )
            if loss_mtp is None:
                loss = loss_ntp + aux_loss
                loss_mtp = torch.zeros_like(loss_ntp)
            else:
                loss = loss_ntp + 0.3 * loss_mtp + aux_loss

            self.model_engine.backward(loss)
            self.model_engine.step()
            t_train_end = time.perf_counter()

            dt = time.perf_counter() - t0
            if (step % self.config.train.log_every == 0) and is_rank0():
                self.logger.info(
                    "step=%d mode=%s loss=%.4f ntp=%.4f mtp=%.4f sel_fallback=%s sel_t=%.3fs score_t=%.3fs feat_t=%.3fs train_t=%.3fs dt=%.3fs n_local=%d",
                    step,
                    selection_mode,
                    float(loss.detach().cpu().item()),
                    float(loss_ntp.detach().cpu().item()),
                    float(loss_mtp.detach().cpu().item()),
                    str(sel.used_fallback),
                    sel.metrics.get("selector_time_s", 0.0),
                    float(t_scoring_end - t_scoring_start),
                    float(t_features_end - t_features_start),
                    float(t_train_end - t_train_start),
                    dt,
                    n_local,
                )
            if shadow_metrics is not None and is_rank0():
                self.logger.info(
                    "selector-bench step=%d primary=%s shadow=%s primary_t=%.3fs shadow_t=%.3fs semantics_valid=%s",
                    step,
                    shadow_metrics["mode_primary"],
                    shadow_metrics["mode_shadow"],
                    shadow_metrics["selector_primary_s"],
                    shadow_metrics["selector_shadow_s"],
                    str(shadow_metrics["semantics_valid"]),
                )
                row = {
                    "step": int(step),
                    "world_size": int(get_world_size()),
                    "rank": int(get_rank()),
                    "mode_primary": shadow_metrics["mode_primary"],
                    "mode_shadow": shadow_metrics["mode_shadow"],
                    "selector_primary_s": shadow_metrics["selector_primary_s"],
                    "selector_shadow_s": shadow_metrics["selector_shadow_s"],
                    "selector_speed_ratio": (
                        shadow_metrics["selector_shadow_s"]
                        / max(shadow_metrics["selector_primary_s"], 1e-9)
                    ),
                    "semantics_valid": bool(shadow_metrics["semantics_valid"]),
                    "scoring_pass_s": float(t_scoring_end - t_scoring_start),
                    "feature_build_s": float(t_features_end - t_features_start),
                    "selector_total_s": float(t_select_end - t_select_start),
                    "train_pass_s": float(t_train_end - t_train_start),
                    "step_total_s": float(dt),
                    "candidate_count_local": int(n_local),
                    "proxy_size": int(self.config.opus.proxy_batch_size),
                    "score_seq_len": int(score_seq_len),
                    "opus_config": asdict(self.config.opus),
                }
                self._write_benchmark_row(row)

            self._write_step_metrics_row(
                {
                    "step": int(step),
                    "selection_mode": selection_mode,
                    "world_size": int(get_world_size()),
                    "rank": int(get_rank()),
                    "candidate_count_local": int(n_local),
                    "selected_count_local": int(sel.selected_local_indices.numel()),
                    "score_seq_len": int(score_seq_len),
                    "train_seq_len_effective": int(x.size(1)),
                    "train_tokens_local": int(valid_rows * x.size(1)),
                    "selector_used_fallback": bool(sel.used_fallback),
                    "scoring_pass_s": float(t_scoring_end - t_scoring_start),
                    "feature_build_s": float(t_features_end - t_features_start),
                    "selector_total_s": float(t_select_end - t_select_start),
                    "train_pass_s": float(t_train_end - t_train_start),
                    "step_total_s": float(dt),
                }
            )

            if step > 0 and (step % self.config.train.checkpoint_every == 0):
                metrics = {
                    "loss": float(loss.detach().cpu().item()),
                    "loss_ntp": float(loss_ntp.detach().cpu().item()),
                    "loss_mtp": float(loss_mtp.detach().cpu().item()),
                    **sel.metrics,
                }
                self._save_checkpoint(step, metrics)

        # Final checkpoint
        final_metrics = {"final_step": float(self.config.train.total_steps)}
        self._save_checkpoint(self.config.train.total_steps, final_metrics)
        barrier()
