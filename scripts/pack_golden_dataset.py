#!/usr/bin/env python3
"""
Pack raw golden samples into fixed-length sequences grouped by domain.

Reads the raw JSONL from build_golden_dataset.py, tokenizes each sample,
greedily bins them by domain to ~seq_len tokens, and outputs packed JSONL
plus a 10-sample markdown preview.

Usage:
  python scripts/pack_golden_dataset.py \
    --input examples/golden_samples.jsonl \
    --tokenizer gpt2 \
    --seq-len 4096 \
    --packed-samples 512 \
    --output examples/golden_packed_512.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("golden_packer")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.setLevel(level)
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_samples(
    raw_samples: List[Dict],
    tokenizer,
    seq_len: int,
    packed_n: int,
) -> List[Dict]:
    """Pack raw samples into exactly seq_len-token sequences grouped by domain.

    Every packed sample is exactly seq_len tokens. Samples are never broken
    midway — if the next sample doesn't fit, the bin is padded with <|pad|>
    to exactly seq_len and emitted.

    Algorithm:
      1. Tokenize all raw samples upfront into (tag, token_ids, source_id)
      2. Group by domain, greedily fill bins:
         - If sample fits (buf_len + sep + sample_len <= seq_len): append
         - If not: pad buf to seq_len with <|pad|>, emit, start new bin
      3. Domain leftovers go to cross-domain buffer, same logic
      4. Returns list of {"id", "tag", "text", "token_count", "source_ids"} dicts
    """
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    sep_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    # Tokenize all samples upfront
    tokenized = []
    for s in raw_samples:
        ids = tokenizer.encode(s["text"])
        tokenized.append({
            "tag": s["tag"],
            "ids": ids,
            "source_id": s.get("id", "?"),
        })

    # Group by domain
    by_domain: Dict[str, list] = defaultdict(list)
    for t in tokenized:
        by_domain[t["tag"]].append(t)

    log.info("Packing %d raw samples across %d domains into %d × %d-token sequences",
             len(raw_samples), len(by_domain), packed_n, seq_len)

    def _emit_bin(buf_ids: List[int], tag: str, source_ids: List[str]) -> Dict:
        """Truncate or pad buf to exactly seq_len and return packed sample dict."""
        if len(buf_ids) > seq_len:
            buf_ids = buf_ids[:seq_len]
            pad_needed = 0
        else:
            pad_needed = seq_len - len(buf_ids)
            buf_ids = buf_ids + [pad_id] * pad_needed
        assert len(buf_ids) == seq_len
        text = tokenizer.decode(buf_ids)
        return {
            "tag": tag,
            "text": text,
            "input_ids": buf_ids,
            "token_count": seq_len,
            "source_ids": source_ids[:],
            "pad_tokens": pad_needed,
        }

    packed: List[Dict] = []
    cross_leftover: list = []  # list of tokenized dicts
    _used_sample_hashes: set = set()  # track used raw samples by text hash
    dup_skipped = 0

    # Deduplicate raw samples: each unique text used at most once
    deduped_by_domain: Dict[str, list] = {}
    for domain, samples in sorted(by_domain.items()):
        deduped = []
        for s in samples:
            h = hashlib.sha256(
                tokenizer.decode(s["ids"]).encode()
            ).hexdigest()
            if h not in _used_sample_hashes:
                _used_sample_hashes.add(h)
                deduped.append(s)
            else:
                dup_skipped += 1
        deduped_by_domain[domain] = deduped

    if dup_skipped:
        log.info("Dedup: skipped %d duplicate raw samples across domains", dup_skipped)

    for domain, samples in sorted(deduped_by_domain.items()):
        buf: List[int] = []
        buf_source_ids: List[str] = []

        for s in samples:
            sample_ids = s["ids"]
            # Cost to add: separator (if buf non-empty) + sample tokens
            cost = len(sample_ids) + (1 if buf else 0)

            if len(buf) + cost <= seq_len:
                # Fits — append
                if buf:
                    buf.append(sep_id)
                buf.extend(sample_ids)
                buf_source_ids.append(s["source_id"])
            else:
                # Doesn't fit — pad & emit current bin, start new one
                if buf:
                    packed.append(_emit_bin(buf, domain, buf_source_ids))
                buf = list(sample_ids)
                buf_source_ids = [s["source_id"]]

        # Domain leftover -> cross-domain
        if buf:
            # Store leftover as a single chunk to preserve domain grouping
            cross_leftover.append({
                "tag": domain,
                "ids": buf,
                "source_ids": buf_source_ids,
            })

    # Pack cross-domain leftovers
    buf: List[int] = []
    buf_source_ids: List[str] = []
    for chunk in cross_leftover:
        cost = len(chunk["ids"]) + (1 if buf else 0)
        if len(buf) + cost <= seq_len:
            if buf:
                buf.append(sep_id)
            buf.extend(chunk["ids"])
            buf_source_ids.extend(chunk["source_ids"])
        else:
            if buf:
                packed.append(_emit_bin(buf, "cross_domain", buf_source_ids))
            buf = list(chunk["ids"])
            buf_source_ids = list(chunk["source_ids"])

    # Final cross-domain bin (pad even if underfilled)
    if buf:
        packed.append(_emit_bin(buf, "cross_domain", buf_source_ids))

    # Deduplicate packed records: remove identical packed bins
    seen_packed_hashes: set = set()
    unique_packed: List[Dict] = []
    packed_dup_removed = 0
    for p in packed:
        h = hashlib.sha256(p["text"].encode()).hexdigest()
        if h not in seen_packed_hashes:
            seen_packed_hashes.add(h)
            unique_packed.append(p)
        else:
            packed_dup_removed += 1
    if packed_dup_removed:
        log.info("Dedup: removed %d identical packed records", packed_dup_removed)
    packed = unique_packed

    # Assign IDs
    for i, p in enumerate(packed):
        p["id"] = f"gp{i + 1}"

    # Trim to target
    if len(packed) > packed_n:
        log.info("Produced %d packed samples, trimming to %d", len(packed), packed_n)
        packed = packed[:packed_n]
    elif len(packed) < packed_n:
        log.warning("Only %d packed samples produced, need %d. "
                     "Increase raw sample count (re-run build_golden_dataset.py with higher --total-samples).",
                     len(packed), packed_n)

    # Stats
    pad_counts = [p["pad_tokens"] for p in packed]
    log.info("Pad stats: avg=%d, max=%d, min=%d",
             sum(pad_counts) // len(pad_counts), max(pad_counts), min(pad_counts))

    return packed


# ---------------------------------------------------------------------------
# Preview MD
# ---------------------------------------------------------------------------

def write_preview_md(packed: List[Dict], path: Path, n: int = 10) -> None:
    """Write overview tables + n diverse full packed samples as readable markdown."""
    import html as _html

    # Domain distribution
    domain_counts: Dict[str, int] = defaultdict(int)
    for p in packed:
        domain_counts[p["tag"]] += 1

    # Pad stats
    pad_counts = [p["pad_tokens"] for p in packed]
    avg_pad = sum(pad_counts) // len(pad_counts)
    total_tokens = sum(p["token_count"] for p in packed)

    # Pick n diverse samples
    seen = set()
    selected = []
    for p in packed:
        if p["tag"] not in seen and len(selected) < n:
            seen.add(p["tag"])
            selected.append(p)
    for p in packed:
        if len(selected) >= n:
            break
        if p not in selected:
            selected.append(p)

    with open(path, "w", encoding="utf-8") as f:
        # --- Overview table ---
        f.write("# OPUS Golden Dataset — Packed Samples Preview\n\n")
        f.write("## Overview\n\n")
        f.write("| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total packed samples | **{len(packed)}** |\n")
        f.write(f"| Tokens per sample | **{packed[0]['token_count']}** (exact) |\n")
        f.write(f"| Total tokens | **{total_tokens:,}** |\n")
        f.write(f"| Domains covered | **{len(domain_counts)}** |\n")
        f.write(f"| Avg padding per sample | {avg_pad} tokens |\n")
        f.write(f"| Min padding | {min(pad_counts)} tokens |\n")
        f.write(f"| Max padding | {max(pad_counts)} tokens |\n")
        f.write("| Format | `<\\|user\\|> ... <\\|assistant\\|> ...` (SFT loss masking ready) |\n")
        f.write("| Sample separator | `<\\|end_of_text\\|>` |\n")
        f.write("| Padding token | `<\\|pad\\|>` |\n\n")

        # --- Domain distribution table ---
        f.write("## Domain Distribution\n\n")
        f.write("| Domain | Packed Samples |\n|--------|---------------|\n")
        for tag, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {tag} | {count} |\n")
        f.write("\n---\n\n")

        # --- Sample previews ---
        f.write(f"## {n} Sample Previews (Full {packed[0]['token_count']} Tokens Each)\n\n")
        f.write("> Each sample below is **exactly 4096 tokens**. The full text is shown.\n")
        f.write("> Samples contain multiple raw QA pairs packed together, separated by `<|end_of_text|>`.\n")
        f.write("> Padding at the end is shown as `<|pad|>` tokens.\n\n")

        for i, p in enumerate(selected):
            f.write(f"### Sample {i + 1} — `{p['tag']}`\n\n")
            f.write(f"| Field | Value |\n|-------|-------|\n")
            f.write(f"| Token count | {p['token_count']} |\n")
            f.write(f"| Pad tokens | {p['pad_tokens']} |\n")
            f.write(f"| Sources | {', '.join(p['source_ids'])} |\n")
            f.write(f"| ID | {p.get('id', '?')} |\n\n")
            escaped = _html.escape(p["text"])
            f.write(f"<details><summary>Click to expand full 4096-token sample</summary>\n\n"
                    f"<pre>\n{escaped}\n</pre>\n\n</details>\n\n---\n\n")

    log.info("Wrote %d-sample preview to %s", len(selected), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pack raw golden samples into fixed-length sequences by domain"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input raw JSONL from build_golden_dataset.py"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True,
        help="HF tokenizer path for token counting (e.g. gpt2, path/to/custom)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=4096,
        help="Target token count per packed sample (default: 4096)"
    )
    parser.add_argument(
        "--packed-samples", type=int, default=512,
        help="Number of packed sequences to produce (default: 512)"
    )
    parser.add_argument(
        "--output", type=str, default="examples/golden_packed_512.jsonl",
        help="Output packed JSONL path"
    )
    parser.add_argument(
        "--preview", type=str, default="examples/golden_packed_preview.md",
        help="Output preview markdown path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    # Load tokenizer
    from transformers import AutoTokenizer
    log.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load raw samples
    raw_samples = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    log.info("Loaded %d raw samples from %s", len(raw_samples), args.input)

    # Pack
    packed = pack_samples(raw_samples, tokenizer, args.seq_len, args.packed_samples)

    # Write packed JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in packed:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Wrote %d packed samples to %s", len(packed), out_path)

    # Write preview MD
    if packed:
        write_preview_md(packed, Path(args.preview))

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("PACKING COMPLETE")
    log.info("=" * 60)
    log.info("Packed samples: %d / %d target", len(packed), args.packed_samples)
    total_tokens = sum(p["token_count"] for p in packed)
    log.info("Total tokens: %d (avg %d per sample)", total_tokens,
             total_tokens // len(packed) if packed else 0)

    # Domain distribution
    domain_counts: Dict[str, int] = defaultdict(int)
    for p in packed:
        domain_counts[p["tag"]] += 1
    log.info("")
    log.info("Domain distribution:")
    for tag, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        log.info("  %-35s %d samples", tag, count)


if __name__ == "__main__":
    main()
