#!/usr/bin/env python3
"""
Build Stage-B Bench-Proxy shard from pre-scored documents.

Expected input: JSONL with fields:
  - score: float (higher is better)
  - input_ids: list[int] tokenized sequence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Bench-Proxy token shard")
    p.add_argument("--input-jsonl", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--token-budget", type=int, default=30_000_000)
    p.add_argument("--seq-len", type=int, default=512)
    return p.parse_args()



def main() -> None:
    args = parse_args()
    src = Path(args.input_jsonl)
    rows: List[Tuple[float, List[int]]] = []

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            score = float(obj.get("score", 0.0))
            ids = obj.get("input_ids")
            if not isinstance(ids, list) or len(ids) == 0:
                continue
            rows.append((score, [int(x) for x in ids]))

    rows.sort(key=lambda x: x[0], reverse=True)

    # Pack sequences into fixed-length blocks (no padding).
    # Short sequences are concatenated until seq_len is filled.
    kept: List[torch.Tensor] = []
    tokens = 0
    buf: List[int] = []
    for _, ids in rows:
        buf.extend(ids)
        while len(buf) >= args.seq_len:
            x = torch.tensor(buf[: args.seq_len], dtype=torch.long)
            buf = buf[args.seq_len :]
            kept.append(x)
            tokens += args.seq_len
            if tokens >= args.token_budget:
                break
        if tokens >= args.token_budget:
            break

    if not kept:
        raise RuntimeError("No rows retained for bench proxy shard")

    out = torch.stack(kept, dim=0)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"Saved bench proxy tensor: {dst} shape={tuple(out.shape)}")


if __name__ == "__main__":
    main()
