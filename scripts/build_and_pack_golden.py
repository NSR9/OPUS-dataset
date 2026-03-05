#!/usr/bin/env python3
"""
Unified golden dataset builder + packer for OPUS.

Collects raw samples from HuggingFace, cleans, deduplicates, tokenizes, and
packs them into fixed-length sequences — all in one pass, one domain at a time.

Ensures uniform distribution across all 12 high-level domains via round-robin
interleaving of datasets within each domain.

Outputs:
  1. Full JSONL  (id, tag, text, input_ids, token_count, source_ids, pad_tokens)
  2. Light JSONL (id, tag, text)
  3. Preview markdown

Usage:
  # All domains:
  python scripts/build_and_pack_golden.py --total-packed 512 --seq-len 4096 \
      --tokenizer tsai_131k_tokenizer --output examples/golden_packed_512.jsonl

  # Single domain (appends to the same output file — run domain by domain):
  python scripts/build_and_pack_golden.py --domain "Math" --total-packed 512 \
      --tokenizer tsai_131k_tokenizer --output examples/golden_packed_512.jsonl
  python scripts/build_and_pack_golden.py --domain "Code" --total-packed 512 \
      --tokenizer tsai_131k_tokenizer --output examples/golden_packed_512.jsonl

  # Custom weights:
  python scripts/build_and_pack_golden.py --weights-json configs/golden_weights.json \
      --total-packed 512 --tokenizer tsai_131k_tokenizer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from clean_golden_samples import clean_text
import build_golden_dataset as _bgd
from build_golden_dataset import (
    DATASET_REGISTRY,
    DOMAIN_NAMES,
    DATASETS_PER_DOMAIN,
    DEFAULT_DOMAIN_WEIGHTS,
    ALL_DATASET_IDS,
    _has_cjk,
    _extract_content_key,
    _detect_hf_token,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("golden_unified")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    log.setLevel(level)
    log.addHandler(handler)


# ---------------------------------------------------------------------------
# Domain allocation
# ---------------------------------------------------------------------------

def compute_domain_quotas(
    total_packed: int,
    domain_names: List[str],
    domain_weights: Dict[str, float],
) -> Dict[str, int]:
    """Compute how many packed records each domain gets.

    Uses largest-remainder method for fair rounding.
    Every domain gets >= 1 record.
    """
    weight_sum = sum(domain_weights[d] for d in domain_names)
    raw = {d: (domain_weights[d] / weight_sum) * total_packed for d in domain_names}

    # Floor each, ensure >= 1
    quotas = {d: max(1, math.floor(raw[d])) for d in domain_names}

    # Distribute remainder by fractional part
    allocated = sum(quotas.values())
    remainder = total_packed - allocated
    if remainder > 0:
        frac_parts = [(raw[d] - math.floor(raw[d]), d) for d in domain_names]
        frac_parts.sort(reverse=True)
        for i in range(remainder):
            quotas[frac_parts[i % len(frac_parts)][1]] += 1
    elif remainder < 0:
        # Over-allocated from clamping to 1 — steal from largest
        overshoot = -remainder
        headroom = [(quotas[d] - 1, d) for d in domain_names]
        headroom.sort(reverse=True)
        for i in range(overshoot):
            d = headroom[i % len(headroom)][1]
            if quotas[d] > 1:
                quotas[d] -= 1

    return quotas


# ---------------------------------------------------------------------------
# Collect & clean per-dataset pool
# ---------------------------------------------------------------------------

def collect_dataset_pool(
    ds_id: str,
    proc_fn,
    target_n: int,
    tokenizer,
    seen_texts: set,
    seen_content: set,
    domain_idx: int,
    ds_idx: int,
    ds_total: int,
    domain_name: str,
    skip_errors: bool = False,
) -> List[Dict]:
    """Fetch, clean, filter CJK, dedup, and tokenize samples from one dataset.

    Returns list of dicts: {token_ids, source_id, tag, text}
    """
    label = f"[{domain_name} {ds_idx}/{ds_total}]"
    short_name = ds_id.split("/")[-1]
    log.info("%s Fetching from %s (requesting %d raw)...", label, ds_id, target_n)

    t0 = time.time()
    try:
        raw_samples = proc_fn(target_n)
    except Exception as e:
        dt = time.time() - t0
        if skip_errors:
            log.error("%s FAILED in %.1fs: %s", label, dt, e)
            return []
        else:
            log.error("%s FATAL in %.1fs:", label, dt)
            traceback.print_exc()
            sys.exit(1)

    dt = time.time() - t0
    if not raw_samples:
        log.warning("%s Fetched 0 raw samples in %.1fs (gated/broken)", label, dt)
        return []

    log.info("%s Fetched %d raw samples in %.1fs", label, len(raw_samples), dt)

    # --- Clean ---
    for s in raw_samples:
        if "text" in s:
            s["text"] = clean_text(s["text"])

    # --- CJK filter ---
    before_cjk = len(raw_samples)
    raw_samples = [s for s in raw_samples if not _has_cjk(s.get("text", ""))]
    cjk_removed = before_cjk - len(raw_samples)
    if cjk_removed:
        log.info("%s After CJK filter: %d (removed %d CJK)", label, len(raw_samples), cjk_removed)

    # --- Dedup (global) ---
    deduped = []
    text_dups = 0
    content_dups = 0
    for s in raw_samples:
        text = s.get("text", "")
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in seen_texts:
            text_dups += 1
            continue
        content_key = _extract_content_key(text)
        if content_key and content_key in seen_content:
            content_dups += 1
            continue
        seen_texts.add(text_hash)
        if content_key:
            seen_content.add(content_key)
        deduped.append(s)

    if text_dups or content_dups:
        log.info("%s After dedup: %d (removed %d text dups, %d content dups)",
                 label, len(deduped), text_dups, content_dups)

    # --- Tokenize ---
    pool = []
    for s in deduped:
        token_ids = tokenizer.encode(s["text"])
        pool.append({
            "token_ids": token_ids,
            "source_id": s.get("id", "?"),
            "tag": s.get("tag", "unknown"),
            "text": s["text"],
        })

    log.info("%s Tokenized %d clean samples → pool ready", label, len(pool))
    return pool


# ---------------------------------------------------------------------------
# Round-robin packing
# ---------------------------------------------------------------------------

def _try_fill_gap(
    dataset_pools: Dict[str, deque],
    pool_names: List[str],
    bin_ids: List[int],
    bin_sources: List[str],
    bin_tags: List[str],
    sep_id: int,
    seq_len: int,
) -> bool:
    """Scan all pools for any sample that fits the remaining bin space.

    Tries smaller samples from any dataset to minimize padding.
    Returns True if at least one sample was added.
    """
    remaining = seq_len - len(bin_ids)
    filled = False
    # Keep trying until nothing more fits
    while True:
        best_sample = None
        best_pool_name = None
        best_cost = 0
        # Find the largest sample that fits
        for pool_name in pool_names:
            pool = dataset_pools[pool_name]
            if not pool:
                continue
            for i, sample in enumerate(pool):
                cost = len(sample["token_ids"]) + (1 if bin_ids else 0)
                if cost <= remaining and cost > best_cost:
                    best_sample = sample
                    best_pool_name = pool_name
                    best_cost = cost
                    best_idx = i
                    break  # take first from this pool that fits (they're ordered)
        if best_sample is None:
            break
        # Remove from pool
        pool = dataset_pools[best_pool_name]
        del pool[best_idx]
        # Add to bin
        if bin_ids:
            bin_ids.append(sep_id)
        bin_ids.extend(best_sample["token_ids"])
        bin_sources.append(best_sample["source_id"])
        bin_tags.append(best_sample["tag"])
        remaining = seq_len - len(bin_ids)
        filled = True
    return filled


def pack_domain_round_robin(
    dataset_pools: Dict[str, deque],
    domain_name: str,
    quota: int,
    tokenizer,
    seq_len: int,
) -> List[Dict]:
    """Pack samples from multiple dataset pools into fixed-length sequences.

    Uses round-robin across dataset pools so every dataset contributes equally.
    """
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    sep_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    pool_names = [name for name, pool in dataset_pools.items() if len(pool) > 0]
    if not pool_names:
        log.warning("  No samples available for domain '%s'", domain_name)
        return []

    log.info("  Packing %s: starting round-robin across %d dataset pools",
             domain_name, len(pool_names))

    packed: List[Dict] = []
    bin_ids: List[int] = []
    bin_sources: List[str] = []
    bin_tags: List[str] = []

    def _emit_bin():
        """Pad current bin to seq_len and emit."""
        nonlocal bin_ids, bin_sources, bin_tags
        if not bin_ids:
            return
        pad_needed = seq_len - len(bin_ids)
        if pad_needed < 0:
            # Truncate (shouldn't happen in normal flow)
            bin_ids = bin_ids[:seq_len]
            pad_needed = 0
        else:
            bin_ids = bin_ids + [pad_id] * pad_needed

        text = tokenizer.decode(bin_ids)
        record = {
            "tag": domain_name,
            "text": text,
            "input_ids": bin_ids,
            "token_count": seq_len,
            "source_ids": bin_sources[:],
            "pad_tokens": pad_needed,
        }
        packed.append(record)
        n = len(packed)
        log.info("  [%s pack %d/%d] Bin filled: %d tokens, %d samples, %d pad tokens",
                 domain_name, n, quota, seq_len, len(bin_sources), pad_needed)
        bin_ids = []
        bin_sources = []
        bin_tags = []

    # Round-robin loop with best-fit gap filling
    while len(packed) < quota:
        active_pools = [name for name in pool_names if len(dataset_pools[name]) > 0]
        if not active_pools:
            if bin_ids:
                _emit_bin()
            break

        added_any = False
        for pool_name in list(pool_names):
            if len(packed) >= quota:
                break

            pool = dataset_pools[pool_name]
            if not pool:
                continue

            sample = pool[0]  # peek, don't pop yet
            sample_ids = sample["token_ids"]
            cost = len(sample_ids) + (1 if bin_ids else 0)

            if len(bin_ids) + cost <= seq_len:
                # Fits — pop and append
                pool.popleft()
                if bin_ids:
                    bin_ids.append(sep_id)
                bin_ids.extend(sample_ids)
                bin_sources.append(sample["source_id"])
                bin_tags.append(sample["tag"])
                added_any = True
            # else: skip this pool for now, try others that might fit

        if not added_any:
            # No sample from any pool fits the remaining space.
            # Before padding, scan ALL pools for any sample that fits.
            filled_gap = _try_fill_gap(
                dataset_pools, pool_names, bin_ids, bin_sources, bin_tags,
                sep_id, seq_len
            )
            if not filled_gap:
                # Truly nothing fits — pad and emit
                if bin_ids:
                    _emit_bin()
                    if len(packed) >= quota:
                        break
                # Start fresh: pop the first available sample for the new bin
                started = False
                for pool_name in pool_names:
                    pool = dataset_pools[pool_name]
                    if pool:
                        sample = pool.popleft()
                        bin_ids = list(sample["token_ids"])
                        bin_sources = [sample["source_id"]]
                        bin_tags = [sample["tag"]]
                        started = True
                        break
                if not started:
                    break  # all pools empty

        # Remove exhausted pools from rotation
        pool_names = [name for name in pool_names if len(dataset_pools[name]) > 0]

    # Emit final partial bin
    if bin_ids and len(packed) < quota:
        _emit_bin()

    return packed


# ---------------------------------------------------------------------------
# Packed record dedup
# ---------------------------------------------------------------------------

def dedup_packed(packed: List[Dict], domain_name: str) -> List[Dict]:
    """Remove exact duplicate packed records."""
    seen: set = set()
    unique: List[Dict] = []
    dups = 0
    for p in packed:
        h = hashlib.sha256(p["text"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(p)
        else:
            dups += 1
    if dups:
        log.info("  %s packed dedup: %d duplicates removed (%d unique records)",
                 domain_name, dups, len(unique))
    else:
        log.info("  %s packed dedup: 0 duplicates removed (%d unique records)",
                 domain_name, len(unique))
    return unique


# ---------------------------------------------------------------------------
# Preview MD (reused from pack_golden_dataset.py with minor tweaks)
# ---------------------------------------------------------------------------

def write_preview_md(packed: List[Dict], path: Path, n: int = 10) -> None:
    """Write overview tables + n diverse full packed samples as readable markdown."""
    import html as _html

    domain_counts: Dict[str, int] = defaultdict(int)
    for p in packed:
        domain_counts[p["tag"]] += 1

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

        f.write("## Domain Distribution\n\n")
        f.write("| Domain | Packed Samples |\n|--------|---------------|\n")
        for tag, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {tag} | {count} |\n")
        f.write("\n---\n\n")

        f.write(f"## {n} Sample Previews (Full {packed[0]['token_count']} Tokens Each)\n\n")
        f.write("> Each sample below is **exactly 4096 tokens**. The full text is shown.\n")
        f.write("> Samples contain multiple raw QA pairs packed together, separated by `<|end_of_text|>`.\n")
        f.write("> Padding at the end is shown as `<|pad|>` tokens.\n\n")

        for i, p in enumerate(selected):
            f.write(f"### Sample {i + 1} — `{p['tag']}`\n\n")
            f.write("| Field | Value |\n|-------|-------|\n")
            f.write(f"| Token count | {p['token_count']} |\n")
            f.write(f"| Pad tokens | {p['pad_tokens']} |\n")
            f.write(f"| Sources | {', '.join(p['source_ids'])} |\n")
            f.write(f"| ID | {p.get('id', '?')} |\n\n")
            escaped = _html.escape(p["text"])
            f.write(f"<details><summary>Click to expand full {p['token_count']}-token sample</summary>\n\n"
                    f"<pre>\n{escaped}\n</pre>\n\n</details>\n\n---\n\n")

    log.info("Wrote %d-sample preview to %s", len(selected), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified golden dataset builder + packer for OPUS"
    )
    parser.add_argument(
        "--total-packed", type=int, default=512,
        help="Total number of packed sequences to produce (default: 512)"
    )
    parser.add_argument(
        "--seq-len", type=int, default=4096,
        help="Target token count per packed sample (default: 4096)"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True,
        help="HF tokenizer path (e.g. tsai_131k_tokenizer)"
    )
    parser.add_argument(
        "--output", type=str, default="examples/golden_packed_512.jsonl",
        help="Output full JSONL path"
    )
    parser.add_argument(
        "--output-light", type=str, default=None,
        help="Output light JSONL path (auto-derived from --output if not set)"
    )
    parser.add_argument(
        "--preview", type=str, default="examples/golden_packed_preview.md",
        help="Output preview markdown path"
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Run only this domain (e.g. 'Math'). Produces only that domain's quota."
    )
    parser.add_argument(
        "--weights-json", type=str, default=None,
        help="Optional JSON file with domain_weights override"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-errors", action="store_true",
        help="Skip datasets that fail to load (instead of crashing)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)
    random.seed(args.seed)

    # --- HF Token ---
    import build_golden_dataset as _bgd
    _bgd.HF_TOKEN = _detect_hf_token()
    log.info("HF token: %s", "FOUND" if _bgd.HF_TOKEN else "NOT FOUND")
    if not _bgd.HF_TOKEN:
        gated = [ds_id for ds_id, _, _, _, g in DATASET_REGISTRY if g]
        log.warning("Gated datasets that will be skipped (%d):", len(gated))
        for g in gated:
            log.warning("  - %s", g)

    # --- Domain weights ---
    domain_weights = DEFAULT_DOMAIN_WEIGHTS.copy()
    if args.weights_json:
        with open(args.weights_json) as f:
            custom = json.load(f)
        if "domain_weights" in custom:
            # Normalize custom weights
            raw_w = custom["domain_weights"]
            w_sum = sum(raw_w.values())
            domain_weights = {d: raw_w.get(d, 1.0) / w_sum for d in DOMAIN_NAMES}
            log.info("Loaded custom weights from %s", args.weights_json)

    # --- Compute domain quotas ---
    quotas = compute_domain_quotas(args.total_packed, DOMAIN_NAMES, domain_weights)

    # --- Determine which domains to process ---
    if args.domain:
        if args.domain not in quotas:
            log.error("Unknown domain '%s'. Available: %s", args.domain, ", ".join(DOMAIN_NAMES))
            sys.exit(1)
        selected_domains = [args.domain]
        log.info("Single domain mode: '%s' (quota: %d packed records)",
                 args.domain, quotas[args.domain])
    else:
        selected_domains = list(DOMAIN_NAMES)

    # --- Print allocation table ---
    log.info("")
    log.info("=" * 60)
    log.info("OPUS Golden Dataset — Unified Builder + Packer")
    log.info("=" * 60)
    log.info("Total packed target: %d", args.total_packed)
    log.info("Seq length: %d tokens", args.seq_len)
    log.info("Domains to process: %d / %d", len(selected_domains), len(DOMAIN_NAMES))
    log.info("")
    log.info("ALLOCATION TABLE:")
    log.info("  %-35s %6s  %s", "Domain", "Quota", "Datasets")
    log.info("  " + "-" * 70)
    for domain in DOMAIN_NAMES:
        ds_list = DATASETS_PER_DOMAIN[domain]
        marker = " <--" if args.domain and domain == args.domain else ""
        log.info("  %-35s %6d  %d datasets%s", domain, quotas[domain], len(ds_list), marker)
    log.info("  " + "-" * 70)
    log.info("  %-35s %6d", "TOTAL", sum(quotas.values()))
    log.info("")

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    log.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")
    sep_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
    log.info("Tokenizer loaded. Vocab=%d, pad_id=%d, sep_id=%d",
             tokenizer.vocab_size, pad_id, sep_id)

    # --- Build processor lookup ---
    processor_lookup = {ds_id: (proc, gated)
                        for ds_id, _, _, proc, gated in DATASET_REGISTRY}

    # --- Global dedup state ---
    seen_texts: set = set()
    seen_content: set = set()
    global_text_dups = 0
    global_content_dups = 0

    # --- Process domains ---
    t_start = time.time()
    all_packed: List[Dict] = []
    failed_datasets: List[str] = []
    skipped_datasets: List[str] = []
    domain_stats: Dict[str, Dict] = {}

    for di, domain in enumerate(selected_domains, 1):
        quota = quotas[domain]
        ds_list = DATASETS_PER_DOMAIN[domain]

        log.info("")
        log.info("=" * 60)
        log.info("DOMAIN %d/%d: %s (quota: %d packed records)", di, len(selected_domains), domain, quota)
        log.info("=" * 60)

        # --- COLLECT: batch-fetch & clean per dataset ---
        dataset_pools: Dict[str, deque] = {}
        domain_cjk = 0
        domain_text_dups = 0
        domain_content_dups = 0

        for dsi, ds_id in enumerate(ds_list, 1):
            proc_fn, is_gated = processor_lookup[ds_id]
            # Each processor already fetches ~500 internally; we pass a generous target
            target_per_ds = max(200, quota * 10 // len(ds_list))

            pool = collect_dataset_pool(
                ds_id=ds_id,
                proc_fn=proc_fn,
                target_n=target_per_ds,
                tokenizer=tokenizer,
                seen_texts=seen_texts,
                seen_content=seen_content,
                domain_idx=di,
                ds_idx=dsi,
                ds_total=len(ds_list),
                domain_name=domain,
                skip_errors=args.skip_errors,
            )
            if pool:
                dataset_pools[ds_id] = deque(pool)
            else:
                skipped_datasets.append(ds_id)

        # --- Pool summary ---
        total_clean = sum(len(p) for p in dataset_pools.values())
        log.info("")
        log.info("  %s dataset pools:", domain)
        for ds_id, pool in dataset_pools.items():
            short = ds_id.split("/")[-1]
            log.info("    %-30s %d samples", short, len(pool))
        log.info("  Total clean samples for %s: %d", domain, total_clean)
        log.info("")

        if total_clean == 0:
            log.warning("  No clean samples for domain '%s' — skipping", domain)
            domain_stats[domain] = {"quota": quota, "produced": 0, "samples": 0}
            continue

        # --- PACK: round-robin across dataset pools ---
        packed = pack_domain_round_robin(
            dataset_pools=dataset_pools,
            domain_name=domain,
            quota=quota,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
        )

        # --- Dedup packed records ---
        packed = dedup_packed(packed, domain)

        # --- Trim to quota ---
        if len(packed) > quota:
            log.info("  Produced %d, trimming to %d", len(packed), quota)
            packed = packed[:quota]
        elif len(packed) < quota:
            log.warning("  Only %d packed samples produced for %s, need %d. "
                        "Increase raw data or reduce total-packed.",
                        len(packed), domain, quota)

        # --- Stats ---
        if packed:
            pad_counts = [p["pad_tokens"] for p in packed]
            log.info("  %s packing complete: %d/%d records, avg pad=%d tokens",
                     domain, len(packed), quota,
                     sum(pad_counts) // len(pad_counts))

            # Check for exhausted pools
            exhausted = [ds_id.split("/")[-1] for ds_id, pool in dataset_pools.items() if len(pool) == 0]
            if exhausted:
                log.info("  Datasets exhausted during packing: %s", ", ".join(exhausted))
            else:
                log.info("  Datasets exhausted during packing: none")

        domain_stats[domain] = {
            "quota": quota,
            "produced": len(packed),
            "samples": total_clean,
        }

        all_packed.extend(packed)
        log.info("")
        log.info("  Domain '%s' done: %d packed records written", domain, len(packed))
        log.info("=" * 60)

    total_time = time.time() - t_start

    # --- Output paths ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_light is None:
        light_path = out_path.with_name(out_path.stem + "_light" + out_path.suffix)
    else:
        light_path = Path(args.output_light)
    light_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Append mode: when --domain is set, merge with existing records ---
    existing_records: List[Dict] = []
    if args.domain and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))
        # Remove any existing records from the same domain (re-run overwrites that domain)
        existing_domains = set(r["tag"] for r in existing_records)
        if args.domain in existing_domains:
            before = len(existing_records)
            existing_records = [r for r in existing_records if r["tag"] != args.domain]
            log.info("Append mode: removed %d old '%s' records, keeping %d from other domains",
                     before - len(existing_records), args.domain, len(existing_records))
        else:
            log.info("Append mode: found %d existing records from %d domains",
                     len(existing_records), len(existing_domains))

    # --- Merge existing + new ---
    combined = existing_records + all_packed

    # --- Assign sequential IDs ---
    for i, p in enumerate(combined):
        p["id"] = f"gp{i + 1}"

    # --- Write full JSONL ---
    with out_path.open("w", encoding="utf-8") as f:
        for p in combined:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if existing_records:
        log.info("Wrote %d packed samples to %s (%d existing + %d new)",
                 len(combined), out_path, len(existing_records), len(all_packed))
    else:
        log.info("Wrote %d packed samples to %s", len(combined), out_path)

    # --- Write light JSONL ---
    with light_path.open("w", encoding="utf-8") as f:
        for p in combined:
            light_record = {"id": p["id"], "tag": p["tag"], "text": p["text"]}
            f.write(json.dumps(light_record, ensure_ascii=False) + "\n")
    log.info("Wrote %d light samples to %s", len(combined), light_path)

    # --- Write preview MD (uses all combined records) ---
    if combined:
        write_preview_md(combined, Path(args.preview))

    # --- Final summary ---
    log.info("")
    log.info("=" * 60)
    log.info("ALL DOMAINS COMPLETE")
    log.info("=" * 60)
    log.info("Total packed records: %d / %d target", len(all_packed),
             sum(quotas[d] for d in selected_domains))
    if existing_records:
        log.info("Combined with existing: %d total records in output file", len(combined))
    total_tokens = sum(p["token_count"] for p in combined)
    log.info("Total tokens: %d", total_tokens)
    log.info("Time: %.1fs", total_time)
    log.info("")
    log.info("Domain distribution (in output file):")
    combined_domain_counts: Dict[str, int] = defaultdict(int)
    for p in combined:
        combined_domain_counts[p["tag"]] += 1
    for tag, count in sorted(combined_domain_counts.items(), key=lambda x: -x[1]):
        log.info("  %-35s %d records", tag, count)

    log.info("")
    log.info("Global dedup stats: %d texts seen, %d content keys seen",
             len(seen_texts), len(seen_content))
    log.info("")
    log.info("Output (full):    %s", out_path)
    log.info("Output (light):   %s", light_path)
    log.info("Preview:          %s", args.preview)

    if failed_datasets:
        log.warning("Failed datasets (%d): %s", len(failed_datasets), ", ".join(failed_datasets))
    if skipped_datasets:
        log.info("Skipped/empty datasets (%d): %s", len(skipped_datasets), ", ".join(skipped_datasets))

    expected = sum(quotas[d] for d in selected_domains)
    if len(all_packed) == expected:
        log.info("Target of %d packed records reached!", expected)
    else:
        log.warning("Target of %d not reached (%d produced).", expected, len(all_packed))


if __name__ == "__main__":
    main()
