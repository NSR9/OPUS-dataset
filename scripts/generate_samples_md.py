#!/usr/bin/env python3
"""
Generate a shareable markdown file with 10 diverse sample previews from a
packed golden JSONL. Designed for architect/stakeholder review.

Picks one sample from each domain for maximum diversity, then fills remaining
slots from underrepresented domains.

Usage:
  python scripts/generate_samples_md.py \
      --input examples/golden_packed_512.jsonl \
      --tokenizer tsai_131k_tokenizer \
      --output examples/golden_samples_preview.md \
      --num-samples 10
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate shareable sample preview MD from packed golden JSONL"
    )
    parser.add_argument("--input", type=str, required=True, help="Packed JSONL file")
    parser.add_argument("--tokenizer", type=str, required=True, help="HF tokenizer path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MD path (default: <input_stem>_samples.md)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to show (default: 10)")
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

    n = args.num_samples

    # --- Pick diverse samples: one per domain first, then fill ---
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r.get("tag", "unknown")].append(r)

    selected = []
    seen_domains = set()

    # Round 1: one sample per domain (pick the one with lowest padding)
    for tag in sorted(by_domain.keys()):
        if len(selected) >= n:
            break
        # Pick sample with lowest pad for best showcase
        candidates = by_domain[tag]
        best = min(candidates, key=lambda r: r.get("pad_tokens", 9999))
        selected.append(best)
        seen_domains.add(tag)

    # Round 2: fill remaining from domains with most records
    if len(selected) < n:
        for tag in sorted(by_domain.keys(), key=lambda t: -len(by_domain[t])):
            for r in by_domain[tag]:
                if len(selected) >= n:
                    break
                if r not in selected:
                    selected.append(r)
            if len(selected) >= n:
                break

    # --- Domain distribution stats ---
    domain_counts = defaultdict(int)
    for r in records:
        domain_counts[r.get("tag", "unknown")] += 1

    total_tokens = sum(r.get("token_count", 4096) for r in records)
    pad_counts = [r.get("pad_tokens", 0) for r in records]

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        inp = Path(args.input)
        out_path = inp.with_name(inp.stem + "_samples.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# OPUS Golden Dataset — Sample Preview\n\n")
        f.write("> This document contains representative samples from the OPUS golden "
                "dataset for review.\n")
        f.write("> Each sample is a packed 4096-token sequence containing multiple QA pairs.\n\n")

        # --- Quick overview ---
        f.write("## Dataset Overview\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total packed samples | **{len(records)}** |\n")
        f.write(f"| Tokens per sample | **4096** (exact) |\n")
        f.write(f"| Total tokens | **{total_tokens:,}** |\n")
        f.write(f"| Domains | **{len(domain_counts)}** |\n")
        f.write(f"| Avg padding | {sum(pad_counts) // len(pad_counts)} tokens "
                f"({sum(pad_counts) * 100 // total_tokens}% waste) |\n")
        f.write(f"| Format | `<\\|user\\|> ... <\\|assistant\\|> ...` |\n")
        f.write(f"| Separator | `<\\|end_of_text\\|>` |\n")
        f.write(f"| Tokenizer | `{args.tokenizer}` |\n\n")

        # --- Domain distribution ---
        f.write("## Domain Distribution\n\n")
        f.write("| Domain | Samples | % |\n|--------|---------|---|\n")
        for tag, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            pct = count * 100 // len(records)
            f.write(f"| {tag} | {count} | {pct}% |\n")
        f.write(f"| **Total** | **{len(records)}** | **100%** |\n")
        f.write("\n---\n\n")

        # --- Samples ---
        f.write(f"## {len(selected)} Sample Previews\n\n")
        f.write("> Samples are picked for domain diversity. Each is exactly 4096 tokens.\n")
        f.write("> `<|end_of_text|>` separates packed QA pairs. `<|pad|>` fills remaining space.\n\n")

        for i, r in enumerate(selected):
            # Tokenize for accurate stats
            ids = tokenizer.encode(r["text"])
            pad = ids.count(pad_id)
            eot = ids.count(eot_id)
            content = len(ids) - pad - eot

            f.write(f"### Sample {i + 1} — {r.get('tag', '?')}\n\n")
            f.write("| Field | Value |\n|-------|-------|\n")
            f.write(f"| ID | `{r.get('id', '?')}` |\n")
            f.write(f"| Domain | {r.get('tag', '?')} |\n")
            f.write(f"| Token count | {len(ids)} |\n")
            f.write(f"| Content tokens | {content} |\n")
            f.write(f"| Pad tokens | {pad} |\n")
            f.write(f"| End-of-text tokens | {eot} |\n")
            f.write(f"| QA pairs packed | {eot + 1} (approx) |\n")
            sources = r.get("source_ids", [])
            if sources:
                f.write(f"| Source IDs | {', '.join(sources)} |\n")
            f.write("\n")

            # Full text in collapsible section
            escaped = _html.escape(r["text"])
            f.write(f"<details><summary>Click to expand full sample ({len(ids)} tokens)</summary>\n\n")
            f.write(f"<pre>\n{escaped}\n</pre>\n\n")
            f.write("</details>\n\n---\n\n")

    print(f"Sample preview written to {out_path}")
    print(f"Domains represented: {len(set(r.get('tag') for r in selected))} / {len(domain_counts)}")


if __name__ == "__main__":
    main()
