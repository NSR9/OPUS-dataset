#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import SYNTHStream, create_bpe_token_strings
from recurrence_model_1b import Model1B, ModelConfig, PFConfig, PFCodec
from training import setup_tokenizer
from production.config import OpusConfig
from production.opus import GhostCollector, AdamWPreconditionerView, OpusSelector


WORD_RE = re.compile(r"[a-z0-9_]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local OPUS selection debugger (single-process, Mac-friendly)")
    p.add_argument("--dataset-path", type=str, default="/Users/rohanshravan/TSAI/synth_local_en")
    p.add_argument("--checkpoint-path", type=str, default="checkpoints/baseline_checkpoint_step_0012000.pt")
    p.add_argument("--random-init", action="store_true", help="Skip checkpoint loading and use random model initialization")
    p.add_argument("--golden-path", type=str, default="examples/golden_samples_template.jsonl")
    p.add_argument("--output-json", type=str, default="selection_report.json")
    p.add_argument("--output-csv", type=str, default="selection_report.csv")
    p.add_argument("--embedding-type", type=str, default="auto", choices=["auto", "standard", "baseline", "kronecker"])
    p.add_argument("--vocab-size", type=int, default=50272)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--score-seq-len", type=int, default=256)
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--proxy-size", type=int, default=8)
    p.add_argument("--selection-ratio", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--sketch-dim", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-snippet-chars", type=int, default=220)
    p.add_argument("--rounds", type=int, default=1, help="How many independent candidate/proxy selection rounds to run")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("[WARN] MPS requested but unavailable in this environment. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if choice == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] CUDA requested but unavailable in this environment. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_ckpt_model_stats(ckpt_state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    vocab = 50272
    hidden = 768
    if "token_embed.weight" in ckpt_state:
        vocab, hidden = ckpt_state["token_embed.weight"].shape

    layer_ids = []
    for k in ckpt_state.keys():
        if not k.startswith("layers."):
            continue
        parts = k.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            layer_ids.append(int(parts[1]))
    num_layers = (max(layer_ids) + 1) if layer_ids else 8

    return {"vocab_size": int(vocab), "hidden_size": int(hidden), "num_layers": int(num_layers)}


def build_local_model(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    seq_len: int,
    embedding_type: str,
    bpe_vocab: List[str] | None = None,
    pf_codec: PFCodec | None = None,
) -> Model1B:
    cfg = ModelConfig()
    cfg.vocab_size = int(vocab_size)
    cfg.hidden_size = int(hidden_size)
    cfg.num_layers = int(num_layers)

    cfg.num_deltanet_layers = max(1, int(round(cfg.num_layers * 0.75)))
    cfg.num_gsa_layers = max(1, cfg.num_layers - cfg.num_deltanet_layers)

    delta_head = 64 if cfg.hidden_size >= 64 else max(8, cfg.hidden_size // 2)
    cfg.delta_head_dim = delta_head
    cfg.delta_v_heads = max(1, cfg.hidden_size // max(1, cfg.delta_head_dim))
    cfg.delta_gate_dim = max(64, cfg.hidden_size // 2)

    gsa_head = 64 if cfg.hidden_size >= 64 else max(8, cfg.hidden_size // 2)
    cfg.gsa_head_dim = gsa_head
    cfg.gsa_num_heads = max(1, cfg.hidden_size // max(1, cfg.gsa_head_dim))
    cfg.gsa_indexer_heads = min(4, cfg.gsa_num_heads)

    cfg.shared_expert_intermediate_size = max(512, cfg.hidden_size * 2)
    cfg.max_seq_len = max(int(seq_len), 512)
    cfg.enable_mtp = True

    if embedding_type == "kronecker":
        model = Model1B(cfg, embedding_type=embedding_type, bpe_vocab=bpe_vocab, pf_codec=pf_codec)
    else:
        model = Model1B(cfg, embedding_type=embedding_type)
    return model


def load_compatible_state(model: nn.Module, ckpt_state: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model_state = model.state_dict()
    compatible = {}
    shape_mismatch = 0
    for k, v in ckpt_state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            compatible[k] = v
        elif k in model_state:
            shape_mismatch += 1

    missing, unexpected = model.load_state_dict(compatible, strict=False)

    loaded_params = sum(int(t.numel()) for t in compatible.values())
    total_params = sum(int(t.numel()) for t in model_state.values())
    ckpt_params = sum(int(t.numel()) for t in ckpt_state.values())

    return {
        "loaded_tensors": float(len(compatible)),
        "model_tensors": float(len(model_state)),
        "ckpt_tensors": float(len(ckpt_state)),
        "shape_mismatch_tensors": float(shape_mismatch),
        "loaded_params": float(loaded_params),
        "total_model_params": float(total_params),
        "total_ckpt_params": float(ckpt_params),
        "loaded_pct_of_model": 100.0 * loaded_params / max(total_params, 1),
        "loaded_pct_of_ckpt": 100.0 * loaded_params / max(ckpt_params, 1),
        "missing_tensors_after_load": float(len(missing)),
        "unexpected_tensors_after_load": float(len(unexpected)),
    }


def tokenize_words(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    sa = tokenize_words(a)
    sb = tokenize_words(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter) / float(max(union, 1))


def load_golden(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []

    out: List[Dict[str, str]] = []
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                out.append({
                    "id": (row.get("id") or "").strip() or f"g{len(out)+1}",
                    "tag": (row.get("tag") or "").strip(),
                    "text": text,
                })
        return out

    # json or jsonl
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    if p.suffix.lower() == ".json" and len(lines) == 1:
        blob = json.loads(lines[0])
        if isinstance(blob, list):
            for i, row in enumerate(blob):
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                out.append({
                    "id": str(row.get("id", f"g{i+1}")),
                    "tag": str(row.get("tag", "")),
                    "text": text,
                })
            return out

    for i, line in enumerate(lines):
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        out.append({
            "id": str(row.get("id", f"g{i+1}")),
            "tag": str(row.get("tag", "")),
            "text": text,
        })
    return out


def best_golden_match(text: str, golden: List[Dict[str, str]]) -> Tuple[str, str, float]:
    if not golden:
        return "", "", 0.0
    best_id, best_tag, best_score = "", "", -1.0
    for g in golden:
        s = jaccard(text, g["text"])
        if s > best_score:
            best_id = g["id"]
            best_tag = g["tag"]
            best_score = s
    return best_id, best_tag, float(best_score)


def next_samples(loader_iter, loader, n: int) -> Tuple[torch.Tensor, object]:
    rows = []
    for _ in range(n):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        rows.append(batch["input_ids"][0])
    return torch.stack(rows, dim=0), loader_iter


def nonfinite_count(x: torch.Tensor) -> int:
    return int((~torch.isfinite(x)).sum().item())


def random_init_stats(model: nn.Module) -> Dict[str, float]:
    model_state = model.state_dict()
    total_params = sum(int(t.numel()) for t in model_state.values())
    return {
        "loaded_tensors": 0.0,
        "model_tensors": float(len(model_state)),
        "ckpt_tensors": 0.0,
        "shape_mismatch_tensors": 0.0,
        "loaded_params": 0.0,
        "total_model_params": float(total_params),
        "total_ckpt_params": 0.0,
        "loaded_pct_of_model": 0.0,
        "loaded_pct_of_ckpt": 0.0,
        "missing_tensors_after_load": 0.0,
        "unexpected_tensors_after_load": 0.0,
    }


def build_kronecker_resources(tokenizer: GPT2Tokenizer, vocab_size: int) -> Tuple[PFCodec, List[str]]:
    print("[INFO] building Kronecker resources (PFCodec + BPE vocab strings)")
    pf_cfg = PFConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )
    pf_codec = PFCodec(pf_cfg)
    bpe_vocab = create_bpe_token_strings(tokenizer, vocab_size=vocab_size)
    return pf_codec, bpe_vocab


def compute_score_loss(model: nn.Module, batch_ids: torch.Tensor) -> torch.Tensor:
    if batch_ids.size(1) < 3:
        raise ValueError("Need seq len >=3")
    x = batch_ids[:, :-2].contiguous()
    y_ntp = batch_ids[:, 1:-1].contiguous()

    logits_ntp, _logits_mtp, aux_loss = model(
        x,
        next_token_ids=None,
        return_memory=False,
        return_loss=True,
    )

    ce = nn.CrossEntropyLoss()
    loss_ntp = ce(logits_ntp.view(-1, logits_ntp.size(-1)), y_ntp.view(-1))
    return loss_ntp + aux_loss


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[INFO] device={device}")

    use_checkpoint = False
    ckpt = {}
    ckpt_state: Dict[str, torch.Tensor] = {}
    ckpt_stats = {
        "vocab_size": int(args.vocab_size),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
    }
    if not args.random_init:
        ckpt_path = Path(args.checkpoint_path)
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            ckpt_state = ckpt.get("model_state_dict", {})
            ckpt_stats = infer_ckpt_model_stats(ckpt_state)
            use_checkpoint = True
            print(
                f"[INFO] checkpoint step={ckpt.get('step', 'n/a')} "
                f"loss={ckpt.get('loss', 'n/a')} embedding_type={ckpt.get('embedding_type', 'n/a')}"
            )
            print(f"[INFO] inferred checkpoint stats: {ckpt_stats}")
        else:
            print(f"[WARN] checkpoint not found at {args.checkpoint_path}; falling back to random init")
            print(f"[INFO] random-init model stats from CLI: {ckpt_stats}")
    else:
        print(f"[INFO] random-init mode enabled; model stats from CLI: {ckpt_stats}")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = setup_tokenizer(tokenizer, target_vocab_size=ckpt_stats["vocab_size"])

    dataset = SYNTHStream(
        tokenizer=tokenizer,
        local_path=args.dataset_path,
        seq_len=args.seq_len,
        batch_size=1,
        seed=args.seed,
        filter_language="en",
        start_step=0,
    )
    loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=True)
    loader_iter = iter(loader)

    ckpt_embedding = str(ckpt.get("embedding_type", "standard")).lower()
    if args.embedding_type == "auto":
        if use_checkpoint:
            if ckpt_embedding in ("standard", "baseline"):
                embedding_type = "standard"
            elif ckpt_embedding == "kronecker":
                embedding_type = "kronecker"
            else:
                embedding_type = "standard"
        else:
            embedding_type = "kronecker"
    elif args.embedding_type == "baseline":
        embedding_type = "standard"
    else:
        embedding_type = args.embedding_type
    print(f"[INFO] model embedding_type={embedding_type} (requested={args.embedding_type}, ckpt={ckpt_embedding})")

    bpe_vocab = None
    pf_codec = None
    if embedding_type == "kronecker":
        pf_codec, bpe_vocab = build_kronecker_resources(tokenizer, ckpt_stats["vocab_size"])

    model = build_local_model(
        vocab_size=ckpt_stats["vocab_size"],
        hidden_size=ckpt_stats["hidden_size"],
        num_layers=ckpt_stats["num_layers"],
        seq_len=args.seq_len,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    )
    if use_checkpoint:
        load_stats = load_compatible_state(model, ckpt_state)
    else:
        load_stats = random_init_stats(model)
    print("[INFO] load compatibility stats:")
    for k, v in load_stats.items():
        print(f"  - {k}: {v}")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    preconditioner = AdamWPreconditionerView(optimizer)

    opus_cfg = OpusConfig(
        enabled=True,
        candidate_multiplier=max(1, args.candidates),
        selection_ratio=float(args.selection_ratio),
        score_seq_len=int(args.score_seq_len),
        proxy_batch_size=max(1, int(args.proxy_size)),
        sketch_dim=max(64, int(args.sketch_dim)),
        temperature=float(args.temperature),
        sketch_seed=int(args.seed),
        include_embeddings=False,
        include_lm_head=False,
    )
    selector = OpusSelector(opus_cfg)

    golden = load_golden(args.golden_path)
    all_records = []
    round_summaries = []

    rounds = max(1, int(args.rounds))
    for round_idx in range(rounds):
        candidate_ids, loader_iter = next_samples(loader_iter, loader, args.candidates)
        proxy_ids, loader_iter = next_samples(loader_iter, loader, args.proxy_size)

        score_seq_len = min(args.score_seq_len, candidate_ids.size(1), proxy_ids.size(1))
        score_batch = torch.cat([
            candidate_ids[:, :score_seq_len],
            proxy_ids[:, :score_seq_len],
        ], dim=0).to(device)

        with GhostCollector(model, include_embeddings=False, include_lm_head=False) as collector:
            optimizer.zero_grad(set_to_none=True)
            score_loss = compute_score_loss(model, score_batch)
            if not bool(torch.isfinite(score_loss)):
                print(f"[WARN] round={round_idx}: non-finite score_loss={float(score_loss.detach().cpu().item())}")
            score_loss.backward()
            captures = collector.captures()
            moe_captures = collector.moe_captures()
            optimizer.zero_grad(set_to_none=True)

        capture_bad_values = 0
        for cap in captures.values():
            capture_bad_values += nonfinite_count(cap.activations)
            capture_bad_values += nonfinite_count(cap.grad_outputs)

        preconditioner.refresh()
        c_feats, p_feats = selector.build_sketch_features(
            captures=captures,
            candidate_count=args.candidates,
            proxy_count=args.proxy_size,
            preconditioner=preconditioner,
            moe_captures=moe_captures,
            out_dtype=torch.float32,
        )
        captures.clear()
        moe_captures.clear()

        feature_bad_values = 0
        for ln in c_feats.keys():
            feature_bad_values += nonfinite_count(c_feats[ln])
            feature_bad_values += nonfinite_count(p_feats[ln])

        sel = selector.select(c_feats, p_feats, learning_rate=args.lr)

        # Round-0 utility proxy for report readability.
        layer_names = sorted(c_feats.keys())
        round0_scores = torch.zeros(args.candidates, dtype=torch.float32)
        for ln in layer_names:
            contrib = args.lr * torch.einsum("nm,m->n", c_feats[ln].cpu(), p_feats[ln].cpu())
            contrib = torch.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0)
            round0_scores += contrib
        round0_bad_values = nonfinite_count(round0_scores)
        round0_scores = torch.nan_to_num(round0_scores, nan=0.0, posinf=0.0, neginf=0.0)

        selected = set(int(x) for x in sel.selected_local_indices.detach().cpu().tolist())
        order = {int(g): i + 1 for i, g in enumerate(sel.selected_global_indices.detach().cpu().tolist())}
        if not order:
            # Single-rank fallback path can have empty global order; preserve deterministic local order.
            order = {int(i): j + 1 for j, i in enumerate(sel.selected_local_indices.detach().cpu().tolist())}

        records = []
        for i in range(args.candidates):
            ids = candidate_ids[i]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            text = text.replace("\n", " ").strip()
            if len(text) > args.max_snippet_chars:
                text = text[: args.max_snippet_chars] + "..."

            g_id, g_tag, g_score = best_golden_match(text, golden)
            rec = {
                "round": round_idx,
                "candidate_index": i,
                "selected": i in selected,
                "selection_order": order.get(i, None),
                "round0_score": float(round0_scores[i].item()),
                "round0_score_is_finite": bool(torch.isfinite(round0_scores[i])),
                "golden_match_id": g_id,
                "golden_match_tag": g_tag,
                "golden_match_jaccard": g_score,
                "text_snippet": text,
            }
            records.append(rec)

        records.sort(key=lambda x: (not x["selected"], -(x["round0_score"])))
        all_records.extend(records)

        selected_scores = [r["golden_match_jaccard"] for r in records if r["selected"]]
        removed_scores = [r["golden_match_jaccard"] for r in records if not r["selected"]]
        summary = {
            "round": round_idx,
            "selector_metrics": sel.metrics,
            "selected_count": int(sum(1 for r in records if r["selected"])),
            "candidate_count": int(args.candidates),
            "capture_nonfinite_values": int(capture_bad_values),
            "feature_nonfinite_values": int(feature_bad_values),
            "round0_nonfinite_values": int(round0_bad_values),
            "selected_avg_golden_jaccard": float(sum(selected_scores) / max(1, len(selected_scores))),
            "removed_avg_golden_jaccard": float(sum(removed_scores) / max(1, len(removed_scores))),
        }
        round_summaries.append(summary)

        print(f"\n[ROUND {round_idx}] selector metrics: {sel.metrics}")
        print(
            f"[ROUND {round_idx}] selected {summary['selected_count']}/{args.candidates} | "
            f"capture_nonfinite={capture_bad_values} feature_nonfinite={feature_bad_values} round0_nonfinite={round0_bad_values}"
        )
        print(f"\n[ROUND {round_idx}] [SELECTED SAMPLES]")
        for r in records:
            if r["selected"]:
                print(
                    f"  idx={r['candidate_index']:>2} order={r['selection_order']} "
                    f"score={r['round0_score']:.4f} golden={r['golden_match_id']}({r['golden_match_jaccard']:.3f})"
                )
                print(f"    {r['text_snippet']}")

        print(f"\n[ROUND {round_idx}] [REMOVED SAMPLES]")
        for r in records:
            if not r["selected"]:
                print(
                    f"  idx={r['candidate_index']:>2} score={r['round0_score']:.4f} "
                    f"golden={r['golden_match_id']}({r['golden_match_jaccard']:.3f})"
                )
                print(f"    {r['text_snippet']}")

    overall_selected = int(sum(1 for r in all_records if r["selected"]))
    overall_candidates = int(len(all_records))
    out_json = {
        "checkpoint_path": args.checkpoint_path if use_checkpoint else None,
        "init_mode": "checkpoint" if use_checkpoint else "random",
        "dataset_path": args.dataset_path,
        "device": str(device),
        "load_stats": load_stats,
        "rounds": rounds,
        "overall_selected_count": overall_selected,
        "overall_candidate_count": overall_candidates,
        "round_summaries": round_summaries,
        "records": all_records,
    }

    Path(args.output_json).write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "candidate_index",
                "selected",
                "selection_order",
                "round0_score",
                "round0_score_is_finite",
                "golden_match_id",
                "golden_match_tag",
                "golden_match_jaccard",
                "text_snippet",
            ],
        )
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    print(f"\nSaved JSON report: {args.output_json}")
    print(f"Saved CSV report:  {args.output_csv}")


if __name__ == "__main__":
    main()
