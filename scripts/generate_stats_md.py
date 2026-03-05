#!/usr/bin/env python3
"""
Generate a comprehensive stats markdown report from a packed golden JSONL.

Usage:
  python scripts/generate_stats_md.py \
      --input examples/golden_packed_512.jsonl \
      --tokenizer tsai_131k_tokenizer \
      --output examples/golden_packed_stats.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate stats MD from packed golden JSONL"
    )
    parser.add_argument("--input", type=str, required=True, help="Packed JSONL file")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MD path (default: <input_stem>_stats.md)")
    args = parser.parse_args()

    # Load tokenizer
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    # Load records
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} packed records from {args.input}")

    if not records:
        print("No records found. Exiting.")
        sys.exit(1)

    # Tokenize and collect stats
    per_sample = []
    domain_stats = defaultdict(lambda: {"count": 0, "pad_total": 0, "eot_total": 0,
                                         "content_total": 0, "samples_per_bin": []})
    for r in records:
        ids = tokenizer.encode(r["text"])
        pad = ids.count(pad_id)
        eot = ids.count(eot_id)
        content = len(ids) - pad - eot
        tag = r.get("tag", "unknown")
        per_sample.append({
            "id": r.get("id", "?"),
            "tag": tag,
            "total": len(ids),
            "pad": pad,
            "eot": eot,
            "content": content,
            "source_count": len(r.get("source_ids", [])),
        })
        ds = domain_stats[tag]
        ds["count"] += 1
        ds["pad_total"] += pad
        ds["eot_total"] += eot
        ds["content_total"] += content
        ds["samples_per_bin"].append(len(r.get("source_ids", [])))

    total_pad = sum(s["pad"] for s in per_sample)
    total_eot = sum(s["eot"] for s in per_sample)
    total_content = sum(s["content"] for s in per_sample)
    total_tokens = sum(s["total"] for s in per_sample)
    n = len(per_sample)

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        inp = Path(args.input)
        out_path = inp.with_name(inp.stem + "_stats.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        # --- Title ---
        f.write("# OPUS Golden Dataset — Packed Statistics Report\n\n")

        # --- Overview ---
        f.write("## Overview\n\n")
        f.write(f"Source: `{args.input}`\n\n")
        f.write(f"Tokenizer: `{args.tokenizer}` (vocab={tokenizer.vocab_size}, "
                f"pad_id={pad_id}, eot_id={eot_id})\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total packed samples | **{n}** |\n")
        f.write(f"| Tokens per sample | **{per_sample[0]['total']}** (exact) |\n")
        f.write(f"| Total tokens | **{total_tokens:,}** |\n")
        f.write(f"| Total content tokens | **{total_content:,}** ({total_content*100//total_tokens}%) |\n")
        f.write(f"| Total pad tokens | **{total_pad:,}** ({total_pad*100//total_tokens}%) |\n")
        f.write(f"| Total end-of-text tokens | **{total_eot:,}** |\n")
        f.write(f"| Domains covered | **{len(domain_stats)}** |\n")
        f.write(f"| Avg padding per sample | {total_pad // n} tokens |\n")
        f.write(f"| Min padding | {min(s['pad'] for s in per_sample)} tokens |\n")
        f.write(f"| Max padding | {max(s['pad'] for s in per_sample)} tokens |\n")
        f.write(f"| Avg samples per bin | {sum(s['source_count'] for s in per_sample) / n:.1f} |\n")
        f.write("\n")

        # --- Domain Distribution ---
        f.write("## Domain Distribution\n\n")
        f.write("| Domain | Packed Samples | Avg Pad | Avg Content | Avg Samples/Bin |\n")
        f.write("|--------|---------------|---------|-------------|----------------|\n")
        for tag in sorted(domain_stats.keys()):
            ds = domain_stats[tag]
            c = ds["count"]
            avg_pad = ds["pad_total"] // c
            avg_content = ds["content_total"] // c
            avg_spb = sum(ds["samples_per_bin"]) / c
            f.write(f"| {tag} | {c} | {avg_pad} | {avg_content} | {avg_spb:.1f} |\n")
        f.write(f"| **Total** | **{n}** | **{total_pad // n}** | "
                f"**{total_content // n}** | "
                f"**{sum(s['source_count'] for s in per_sample) / n:.1f}** |\n")
        f.write("\n")

        # --- Per-Sample Token Statistics by Domain ---
        f.write("## Per-Sample Token Statistics\n\n")

        # Group by domain
        by_domain = defaultdict(list)
        for s in per_sample:
            by_domain[s["tag"]].append(s)

        for tag in sorted(by_domain.keys()):
            samples = by_domain[tag]
            f.write(f"### {tag} ({len(samples)} samples)\n\n")
            f.write("| ID | Total Tokens | Pad Tokens | End-of-Text Tokens | Content Tokens | Sources |\n")
            f.write("|----|-------------|------------|-------------------|----------------|--------|\n")
            for s in samples:
                f.write(f"| {s['id']} | {s['total']} | {s['pad']} | {s['eot']} | "
                        f"{s['content']} | {s['source_count']} |\n")
            # Summary row
            avg_p = sum(s["pad"] for s in samples) // len(samples)
            avg_e = round(sum(s["eot"] for s in samples) / len(samples), 1)
            avg_c = sum(s["content"] for s in samples) // len(samples)
            avg_src = round(sum(s["source_count"] for s in samples) / len(samples), 1)
            min_p = min(s["pad"] for s in samples)
            max_p = max(s["pad"] for s in samples)
            f.write(f"| **Avg** | **4096** | **{avg_p}** | **{avg_e}** | **{avg_c}** | **{avg_src}** |\n")
            f.write(f"| **Min** | 4096 | {min_p} | {min(s['eot'] for s in samples)} | "
                    f"{min(s['content'] for s in samples)} | {min(s['source_count'] for s in samples)} |\n")
            f.write(f"| **Max** | 4096 | {max_p} | {max(s['eot'] for s in samples)} | "
                    f"{max(s['content'] for s in samples)} | {max(s['source_count'] for s in samples)} |\n")
            f.write("\n")

    print(f"Stats report written to {out_path}")


if __name__ == "__main__":
    main()
