#!/usr/bin/env python3
"""Analyze golden dataset files for exact and near duplicates."""

import json
import hashlib
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import re
import sys

GOLDEN_RAW = "examples/golden_samples.jsonl"
GOLDEN_PACKED = "examples/golden_packed_512.jsonl"

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

# ── 1. Load raw golden samples ──────────────────────────────────────────────
print("=" * 80)
print("PART 1: RAW GOLDEN SAMPLES (golden_samples.jsonl)")
print("=" * 80)

raw = load_jsonl(GOLDEN_RAW)
print(f"\nTotal records: {len(raw)}")

# Tag distribution
tag_counts = Counter(r.get("tag", "UNKNOWN") for r in raw)
print("\nTag distribution:")
for tag, count in tag_counts.most_common():
    print(f"  {tag}: {count}")

# ── 1a. Exact duplicates (identical text) ────────────────────────────────────
print("\n" + "-" * 60)
print("1a. EXACT DUPLICATES (identical 'text' field)")
print("-" * 60)

text_to_records = defaultdict(list)
for r in raw:
    text_to_records[r["text"]].append(r)

exact_dup_groups = {k: v for k, v in text_to_records.items() if len(v) > 1}
if exact_dup_groups:
    print(f"\nFound {len(exact_dup_groups)} groups of exact duplicates:")
    for i, (text, records) in enumerate(exact_dup_groups.items(), 1):
        ids = [r["id"] for r in records]
        tags = [r.get("tag", "?") for r in records]
        print(f"\n  Group {i}: {len(records)} copies")
        print(f"    IDs: {ids}")
        print(f"    Tags: {tags}")
        print(f"    Text preview: {text[:150]}...")
else:
    print("\nNo exact duplicates found.")

# ── 1b. Near duplicates - same content, different prompt framing ─────────────
print("\n" + "-" * 60)
print("1b. NEAR DUPLICATES (same content, different prompt framing)")
print("-" * 60)

def extract_translation_content(text):
    """Extract the core translation content (source + target) ignoring prompt framing."""
    # Remove [USER] and [ASSISTANT] markers
    cleaned = text.replace("[USER]", "").replace("[ASSISTANT]", "").strip()

    # Common prompt patterns to strip
    prompt_patterns = [
        r"Translate the following [\w\s]+ text to [\w]+[:\.]?\s*",
        r"Translate to [\w]+[:\.]?\s*",
        r"Please translate the following [\w\s]+ text into [\w]+[:\.]?\s*",
        r"Convert the following [\w\s]+ text to [\w]+[:\.]?\s*",
        r"Translate this [\w\s]+ passage (?:in)?to [\w]+[:\.]?\s*",
        r"Provide (?:an? )?[\w]+ translation (?:of|for) the following[:\.]?\s*",
    ]

    content = cleaned
    for pat in prompt_patterns:
        content = re.sub(pat, "", content, flags=re.IGNORECASE)

    return content.strip()

def extract_translation_pair(text):
    """Try to extract (source_text, target_text) from a translation sample."""
    # Split on [ASSISTANT]
    parts = text.split("[ASSISTANT]")
    if len(parts) != 2:
        return None

    user_part = parts[0].replace("[USER]", "").strip()
    assistant_part = parts[1].strip()

    # Remove prompt instructions from user part
    prompt_patterns = [
        r"^Translate the following [\w\s]+ text to [\w]+[:\.\s]*",
        r"^Translate to [\w]+[:\.\s]*",
        r"^Please translate the following [\w\s]+ text into [\w]+[:\.\s]*",
        r"^Convert the following [\w\s]+ text to [\w]+[:\.\s]*",
        r"^Translate this [\w\s]+ passage (?:in)?to [\w]+[:\.\s]*",
        r"^Provide (?:an? )?[\w]+ translation (?:of|for) the following[:\.\s]*",
    ]

    source = user_part
    for pat in prompt_patterns:
        source = re.sub(pat, "", source, flags=re.IGNORECASE).strip()

    return (source, assistant_part)

# Focus on translation samples
translation_tags = [t for t in tag_counts if "translation" in t.lower() or "indic" in t.lower()]
print(f"\nTranslation-related tags: {translation_tags}")

translation_samples = [r for r in raw if r.get("tag", "") in translation_tags]
print(f"Total translation samples: {len(translation_samples)}")

# Check for near-duplicates by comparing extracted content
# Group by content hash (ignoring prompt)
content_to_records = defaultdict(list)
for r in translation_samples:
    content = extract_translation_content(r["text"])
    content_hash = hashlib.md5(content.encode()).hexdigest()
    content_to_records[content_hash].append(r)

content_dup_groups = {k: v for k, v in content_to_records.items() if len(v) > 1}
if content_dup_groups:
    print(f"\nFound {len(content_dup_groups)} groups of content-level duplicates (same content, possibly different prompt):")
    for i, (h, records) in enumerate(content_dup_groups.items(), 1):
        ids = [r["id"] for r in records]
        tags = [r.get("tag", "?") for r in records]
        print(f"\n  Group {i}: {len(records)} copies")
        print(f"    IDs: {ids}")
        print(f"    Tags: {tags}")
        for r in records:
            print(f"    [{r['id']}] Text start: {r['text'][:200]}...")
else:
    print("\nNo content-level duplicates found among translation samples (after stripping prompt framing).")

# ── 1c. Cross-check: same translation pair in different directions ───────────
print("\n" + "-" * 60)
print("1c. SAME TRANSLATION PAIR, DIFFERENT DIRECTION")
print("-" * 60)

pair_to_records = defaultdict(list)
for r in translation_samples:
    pair = extract_translation_pair(r["text"])
    if pair:
        # Normalize: sort (source, target) so direction doesn't matter
        normalized = tuple(sorted(pair))
        pair_hash = hashlib.md5((normalized[0] + "|||" + normalized[1]).encode()).hexdigest()
        pair_to_records[pair_hash].append(r)

direction_dup_groups = {k: v for k, v in pair_to_records.items() if len(v) > 1}
if direction_dup_groups:
    print(f"\nFound {len(direction_dup_groups)} groups of same translation pairs (possibly different directions):")
    for i, (h, records) in enumerate(direction_dup_groups.items(), 1):
        ids = [r["id"] for r in records]
        tags = [r.get("tag", "?") for r in records]
        print(f"\n  Group {i}: {len(records)} copies")
        print(f"    IDs: {ids}")
        print(f"    Tags: {tags}")
        for r in records:
            pair = extract_translation_pair(r["text"])
            if pair:
                print(f"    [{r['id']}] Source (first 100): {pair[0][:100]}...")
                print(f"    [{r['id']}] Target (first 100): {pair[1][:100]}...")
else:
    print("\nNo same-pair-different-direction duplicates found.")

# ── 1d. Fuzzy similarity check among all samples ────────────────────────────
print("\n" + "-" * 60)
print("1d. HIGH SIMILARITY PAIRS (>= 85% similar text, different IDs)")
print("-" * 60)

high_sim_pairs = []
for i in range(len(raw)):
    for j in range(i + 1, len(raw)):
        # Quick length check to skip obviously different pairs
        len_ratio = min(len(raw[i]["text"]), len(raw[j]["text"])) / max(len(raw[i]["text"]), len(raw[j]["text"]))
        if len_ratio < 0.7:
            continue
        ratio = SequenceMatcher(None, raw[i]["text"], raw[j]["text"]).ratio()
        if ratio >= 0.85:
            high_sim_pairs.append((raw[i], raw[j], ratio))

if high_sim_pairs:
    print(f"\nFound {len(high_sim_pairs)} pairs with >= 85% text similarity:")
    for r1, r2, ratio in sorted(high_sim_pairs, key=lambda x: -x[2]):
        print(f"\n  Similarity: {ratio:.1%}")
        print(f"    {r1['id']} ({r1.get('tag', '?')}) vs {r2['id']} ({r2.get('tag', '?')})")
        # Show the differences
        t1, t2 = r1["text"], r2["text"]
        if t1 != t2:
            # Find first difference
            for k in range(min(len(t1), len(t2))):
                if t1[k] != t2[k]:
                    start = max(0, k - 50)
                    print(f"    First diff at char {k}:")
                    print(f"      [{r1['id']}]: ...{t1[start:k+80]}...")
                    print(f"      [{r2['id']}]: ...{t2[start:k+80]}...")
                    break
else:
    print("\nNo pairs with >= 85% similarity found.")


# ── 2. Packed golden samples ─────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("PART 2: PACKED GOLDEN SAMPLES (golden_packed_512.jsonl)")
print("=" * 80)

packed = load_jsonl(GOLDEN_PACKED)
print(f"\nTotal packed records: {len(packed)}")

# ── 2a. Exact duplicates among packed records ────────────────────────────────
print("\n" + "-" * 60)
print("2a. EXACT DUPLICATES among packed records")
print("-" * 60)

packed_text_to_indices = defaultdict(list)
for i, r in enumerate(packed):
    packed_text_to_indices[r["text"]].append(i)

packed_exact_dups = {k: v for k, v in packed_text_to_indices.items() if len(v) > 1}
if packed_exact_dups:
    print(f"\nFound {len(packed_exact_dups)} groups of exact duplicate packed records:")
    for text, indices in list(packed_exact_dups.items())[:5]:
        print(f"  Indices: {indices}, text preview: {text[:150]}...")
    if len(packed_exact_dups) > 5:
        print(f"  ... and {len(packed_exact_dups) - 5} more groups")
else:
    print("\nNo exact duplicate packed records found.")

# ── 2b. Repeated segments WITHIN individual packed samples ───────────────────
print("\n" + "-" * 60)
print("2b. REPEATED SEGMENTS within individual packed samples")
print("-" * 60)

# Split packed samples on common delimiters and check for repeated segments
SEGMENT_DELIMITERS = ["<|end_of_text|>", "<|endoftext|>", "\n\n\n", "---"]

samples_with_internal_repeats = []
for i, r in enumerate(packed):
    text = r["text"]

    # Try splitting on end-of-text tokens
    segments = []
    for delim in SEGMENT_DELIMITERS:
        if delim in text:
            segments = [s.strip() for s in text.split(delim) if s.strip()]
            break

    if not segments or len(segments) < 2:
        continue

    # Check for exact repeated segments
    seg_counts = Counter()
    for seg in segments:
        seg_hash = hashlib.md5(seg.encode()).hexdigest()
        seg_counts[seg_hash] += 1

    repeated = {h: c for h, c in seg_counts.items() if c > 1}
    if repeated:
        # Find the actual repeated text
        hash_to_seg = {}
        for seg in segments:
            h = hashlib.md5(seg.encode()).hexdigest()
            hash_to_seg[h] = seg

        samples_with_internal_repeats.append({
            "index": i,
            "num_segments": len(segments),
            "repeated": [(hash_to_seg[h], c) for h, c in repeated.items()]
        })

    # Also check for high-similarity segments within the same sample
    for si in range(len(segments)):
        for sj in range(si + 1, len(segments)):
            if len(segments[si]) < 50 or len(segments[sj]) < 50:
                continue
            len_ratio = min(len(segments[si]), len(segments[sj])) / max(len(segments[si]), len(segments[sj]))
            if len_ratio < 0.7:
                continue
            ratio = SequenceMatcher(None, segments[si], segments[sj]).ratio()
            if ratio >= 0.80:
                samples_with_internal_repeats.append({
                    "index": i,
                    "type": "near_duplicate_segments",
                    "similarity": ratio,
                    "seg1_preview": segments[si][:200],
                    "seg2_preview": segments[sj][:200],
                })

if samples_with_internal_repeats:
    print(f"\nFound {len(samples_with_internal_repeats)} instances of repeated/similar segments within packed samples:")
    for info in samples_with_internal_repeats[:20]:
        if "repeated" in info:
            print(f"\n  Packed record index {info['index']}: {info['num_segments']} segments")
            for seg_text, count in info["repeated"]:
                print(f"    Exact repeat x{count}: {seg_text[:150]}...")
        elif info.get("type") == "near_duplicate_segments":
            print(f"\n  Packed record index {info['index']}: near-dup segments ({info['similarity']:.1%})")
            print(f"    Seg1: {info['seg1_preview'][:150]}...")
            print(f"    Seg2: {info['seg2_preview'][:150]}...")
    if len(samples_with_internal_repeats) > 20:
        print(f"\n  ... and {len(samples_with_internal_repeats) - 20} more")
else:
    print("\nNo repeated segments within packed samples found.")

# ── 2c. Check how raw samples map to packed samples ─────────────────────────
print("\n" + "-" * 60)
print("2c. RAW-TO-PACKED MAPPING: how many times each raw sample appears in packed")
print("-" * 60)

raw_in_packed_count = Counter()
raw_in_packed_indices = defaultdict(list)
for pi, p in enumerate(packed):
    for r in raw:
        if r["text"] in p["text"]:
            raw_in_packed_count[r["id"]] += 1
            raw_in_packed_indices[r["id"]].append(pi)

overrepresented = {rid: c for rid, c in raw_in_packed_count.items() if c > 1}
not_found = [r["id"] for r in raw if r["id"] not in raw_in_packed_count]

print(f"\nRaw samples found in packed: {len(raw_in_packed_count)}/{len(raw)}")
print(f"Raw samples NOT found in any packed record: {len(not_found)}")
if not_found:
    not_found_tags = [r.get("tag", "?") for r in raw if r["id"] in not_found]
    print(f"  Missing IDs: {not_found[:10]}{'...' if len(not_found) > 10 else ''}")
    print(f"  Missing tags: {Counter(not_found_tags).most_common()}")

if overrepresented:
    print(f"\nRaw samples appearing in MULTIPLE packed records ({len(overrepresented)} samples):")
    for rid, count in sorted(overrepresented.items(), key=lambda x: -x[1])[:20]:
        tag = next((r.get("tag", "?") for r in raw if r["id"] == rid), "?")
        indices = raw_in_packed_indices[rid]
        print(f"  {rid} ({tag}): appears in {count} packed records (indices: {indices[:10]}{'...' if len(indices) > 10 else ''})")
    if len(overrepresented) > 20:
        print(f"  ... and {len(overrepresented) - 20} more")
else:
    print("\nNo raw sample appears in more than one packed record.")

# ── 3. Indic translation deep dive ──────────────────────────────────────────
print("\n\n" + "=" * 80)
print("PART 3: INDIC TRANSLATION DEEP DIVE")
print("=" * 80)

indic_tags = [t for t in tag_counts if "indic" in t.lower()]
print(f"\nIndic tags: {indic_tags}")

for tag in indic_tags:
    samples = [r for r in raw if r.get("tag") == tag]
    print(f"\n--- {tag} ({len(samples)} samples) ---")

    # Extract and compare translation pairs
    pairs = []
    for r in samples:
        pair = extract_translation_pair(r["text"])
        if pair:
            pairs.append((r["id"], pair[0], pair[1]))

    # Check for duplicate source texts
    source_to_ids = defaultdict(list)
    for rid, src, tgt in pairs:
        # Normalize whitespace
        src_norm = " ".join(src.split())
        source_to_ids[src_norm].append(rid)

    dup_sources = {s: ids for s, ids in source_to_ids.items() if len(ids) > 1}
    if dup_sources:
        print(f"  Duplicate source texts: {len(dup_sources)} groups")
        for src, ids in dup_sources.items():
            print(f"    IDs {ids}: source = {src[:120]}...")
    else:
        print(f"  No duplicate source texts.")

    # Check for duplicate target texts
    target_to_ids = defaultdict(list)
    for rid, src, tgt in pairs:
        tgt_norm = " ".join(tgt.split())
        target_to_ids[tgt_norm].append(rid)

    dup_targets = {t: ids for t, ids in target_to_ids.items() if len(ids) > 1}
    if dup_targets:
        print(f"  Duplicate target texts: {len(dup_targets)} groups")
        for tgt, ids in dup_targets.items():
            print(f"    IDs {ids}: target = {tgt[:120]}...")
    else:
        print(f"  No duplicate target texts.")

    # Check for very similar source texts (fuzzy)
    print(f"  Checking pairwise similarity among {len(pairs)} samples...")
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            src_sim = SequenceMatcher(None, pairs[i][1], pairs[j][1]).ratio()
            if src_sim >= 0.80:
                print(f"    HIGH source similarity ({src_sim:.1%}): {pairs[i][0]} vs {pairs[j][0]}")
                print(f"      [{pairs[i][0]}] {pairs[i][1][:100]}...")
                print(f"      [{pairs[j][0]}] {pairs[j][1][:100]}...")

# ── 4. Cross-tag content overlap ─────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("PART 4: CROSS-TAG CONTENT OVERLAP")
print("=" * 80)
print("\nChecking if any Tamil translation content appears in Telugu samples or vice versa...")

tamil_samples = [r for r in raw if "tamil" in r.get("tag", "").lower()]
telugu_samples = [r for r in raw if "telugu" in r.get("tag", "").lower()]

print(f"Tamil samples: {len(tamil_samples)}")
print(f"Telugu samples: {len(telugu_samples)}")

# Extract English parts from both
def get_english_part(text):
    """Extract the English text from a translation pair."""
    pair = extract_translation_pair(text)
    if not pair:
        return None
    # The English part could be source or target
    # Heuristic: if it's mostly ASCII, it's English
    for part in pair:
        ascii_ratio = sum(1 for c in part if ord(c) < 128) / max(len(part), 1)
        if ascii_ratio > 0.8:
            return " ".join(part.split())  # normalize whitespace
    return None

tamil_english = [(r["id"], get_english_part(r["text"])) for r in tamil_samples]
telugu_english = [(r["id"], get_english_part(r["text"])) for r in telugu_samples]

tamil_english = [(rid, e) for rid, e in tamil_english if e]
telugu_english = [(rid, e) for rid, e in telugu_english if e]

cross_matches = []
for t_id, t_eng in tamil_english:
    for te_id, te_eng in telugu_english:
        if t_eng == te_eng:
            cross_matches.append((t_id, te_id, t_eng))
        elif t_eng and te_eng:
            sim = SequenceMatcher(None, t_eng, te_eng).ratio()
            if sim >= 0.85:
                cross_matches.append((t_id, te_id, f"~{sim:.1%} similar"))

if cross_matches:
    print(f"\nFound {len(cross_matches)} cross-language content matches:")
    for t_id, te_id, info in cross_matches:
        if info.startswith("~"):
            print(f"  {t_id} (Tamil) <-> {te_id} (Telugu): {info}")
        else:
            print(f"  {t_id} (Tamil) <-> {te_id} (Telugu): EXACT same English text")
            print(f"    English: {info[:150]}...")
else:
    print("\nNo cross-language content overlap found between Tamil and Telugu samples.")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Raw dataset:
  Total samples: {len(raw)}
  Exact text duplicates: {sum(len(v) - 1 for v in exact_dup_groups.values())} redundant copies in {len(exact_dup_groups)} groups
  Content-level duplicates (stripped prompts): {sum(len(v) - 1 for v in content_dup_groups.values())} in {len(content_dup_groups)} groups
  High-similarity pairs (>= 85%): {len(high_sim_pairs)}

Packed dataset:
  Total packed records: {len(packed)}
  Exact duplicate packed records: {sum(len(v) - 1 for v in packed_exact_dups.values())} in {len(packed_exact_dups)} groups
  Packed records with internal repeated segments: {len(samples_with_internal_repeats)}
  Raw samples in multiple packed records: {len(overrepresented)}
  Raw samples missing from packed: {len(not_found)}

Cross-language overlap (Tamil/Telugu):
  Shared English content: {len(cross_matches)} matches
""")
