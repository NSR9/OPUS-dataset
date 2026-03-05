#!/usr/bin/env python3
"""Deeper analysis of packed file duplicates and structure."""

import json
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import hashlib

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

packed = load_jsonl(GOLDEN_PACKED)
raw = load_jsonl(GOLDEN_RAW)

# ── 1. Exact duplicate packed records - full detail ─────────────────────────
print("=" * 80)
print("EXACT DUPLICATE PACKED RECORDS - FULL DETAIL")
print("=" * 80)

packed_text_to_indices = defaultdict(list)
for i, r in enumerate(packed):
    packed_text_to_indices[r["text"]].append(i)

packed_exact_dups = {k: v for k, v in packed_text_to_indices.items() if len(v) > 1}
total_wasted = 0
for text, indices in packed_exact_dups.items():
    wasted = len(indices) - 1
    total_wasted += wasted
    print(f"\nGroup: indices {indices} ({len(indices)} copies, {wasted} redundant)")
    print(f"  Text length: {len(text)} chars")
    print(f"  Preview: {text[:300]}...")

print(f"\nTotal exact duplicate packed records: {len(packed_exact_dups)} groups, {total_wasted} redundant records out of {len(packed)}")

# ── 2. Understanding packed format vs raw format ────────────────────────────
print("\n" + "=" * 80)
print("FORMAT COMPARISON: raw vs packed")
print("=" * 80)

print(f"\nRaw sample format (first record):")
print(f"  Keys: {list(raw[0].keys())}")
print(f"  Text starts with: {raw[0]['text'][:100]}...")

print(f"\nPacked sample format (first record):")
print(f"  Keys: {list(packed[0].keys())}")
print(f"  Text starts with: {packed[0]['text'][:100]}...")

# Check if raw uses [USER]/[ASSISTANT] and packed uses <|user|>/<|assistant|>
raw_markers = set()
packed_markers = set()
for r in raw[:5]:
    if "[USER]" in r["text"]: raw_markers.add("[USER]/[ASSISTANT]")
    if "<|user|>" in r["text"]: raw_markers.add("<|user|>/<|assistant|>")
for r in packed[:5]:
    if "[USER]" in r["text"]: packed_markers.add("[USER]/[ASSISTANT]")
    if "<|user|>" in r["text"]: packed_markers.add("<|user|>/<|assistant|>")

print(f"\nRaw format markers: {raw_markers}")
print(f"Packed format markers: {packed_markers}")

# ── 3. Near-duplicate segments in packed (meaningful, not table fragments) ──
print("\n" + "=" * 80)
print("MEANINGFUL NEAR-DUPLICATE SEGMENTS IN PACKED SAMPLES")
print("(Filtering out short/trivial segments)")
print("=" * 80)

DELIMITERS = ["<|end_of_text|>"]  # Only use the real delimiter
MIN_SEGMENT_LEN = 200  # Only consider substantial segments

meaningful_near_dups = []
for i, r in enumerate(packed):
    text = r["text"]
    if "<|end_of_text|>" not in text:
        continue

    segments = [s.strip() for s in text.split("<|end_of_text|>") if s.strip() and len(s.strip()) >= MIN_SEGMENT_LEN]

    if len(segments) < 2:
        continue

    # Check for exact repeated segments
    seg_hashes = defaultdict(list)
    for si, seg in enumerate(segments):
        h = hashlib.md5(seg.encode()).hexdigest()
        seg_hashes[h].append(si)

    for h, seg_indices in seg_hashes.items():
        if len(seg_indices) > 1:
            seg_text = segments[seg_indices[0]]
            meaningful_near_dups.append({
                "packed_index": i,
                "type": "exact_repeat",
                "count": len(seg_indices),
                "segment_indices": seg_indices,
                "preview": seg_text[:300],
                "seg_len": len(seg_text),
            })

    # Check near-duplicate segments (>= 85% similarity, substantial length)
    for si in range(len(segments)):
        for sj in range(si + 1, len(segments)):
            len_ratio = min(len(segments[si]), len(segments[sj])) / max(len(segments[si]), len(segments[sj]))
            if len_ratio < 0.7:
                continue
            ratio = SequenceMatcher(None, segments[si], segments[sj]).ratio()
            if ratio >= 0.85 and ratio < 1.0:
                meaningful_near_dups.append({
                    "packed_index": i,
                    "type": "near_duplicate",
                    "similarity": ratio,
                    "seg1_idx": si,
                    "seg2_idx": sj,
                    "seg1_preview": segments[si][:200],
                    "seg2_preview": segments[sj][:200],
                    "seg1_len": len(segments[si]),
                    "seg2_len": len(segments[sj]),
                })

print(f"\nTotal meaningful findings: {len(meaningful_near_dups)}")

exact_repeats = [d for d in meaningful_near_dups if d["type"] == "exact_repeat"]
near_dups = [d for d in meaningful_near_dups if d["type"] == "near_duplicate"]

print(f"  Exact segment repeats (>= 200 chars): {len(exact_repeats)}")
print(f"  Near-duplicate segments (>= 85% sim, >= 200 chars): {len(near_dups)}")

if exact_repeats:
    print(f"\n--- EXACT REPEATED SEGMENTS (substantial, >= 200 chars) ---")
    for d in exact_repeats[:15]:
        print(f"\n  Packed index {d['packed_index']}: segment repeated {d['count']}x ({d['seg_len']} chars)")
        print(f"    Preview: {d['preview'][:250]}...")

if near_dups:
    print(f"\n--- NEAR-DUPLICATE SEGMENTS (>= 85% sim, >= 200 chars) ---")
    for d in sorted(near_dups, key=lambda x: -x["similarity"])[:15]:
        print(f"\n  Packed index {d['packed_index']}: segments {d['seg1_idx']} & {d['seg2_idx']} ({d['similarity']:.1%} similar)")
        print(f"    Seg {d['seg1_idx']} ({d['seg1_len']} chars): {d['seg1_preview'][:200]}...")
        print(f"    Seg {d['seg2_idx']} ({d['seg2_len']} chars): {d['seg2_preview'][:200]}...")

# ── 4. How many unique packed records are there really? ─────────────────────
print("\n\n" + "=" * 80)
print("UNIQUE CONTENT ANALYSIS")
print("=" * 80)

unique_packed = len(packed_text_to_indices)
print(f"Total packed records: {len(packed)}")
print(f"Unique packed records (by text): {unique_packed}")
print(f"Redundant exact copies: {len(packed) - unique_packed}")

# Check packed records that contain indic content
print("\n--- Packed records containing Indic translation markers ---")
indic_packed = []
for i, r in enumerate(packed):
    text = r["text"]
    if any(marker in text for marker in ["Tamil", "Telugu", "Hindi", "Bengali", "Kannada", "తెలుగు", "தமிழ்", "हिन्दी", "বাংলা", "ಕನ್ನಡ"]):
        indic_packed.append(i)

print(f"Packed records with Indic content: {len(indic_packed)}")
print(f"Indices: {indic_packed[:50]}{'...' if len(indic_packed) > 50 else ''}")

# ── 5. Check for same-content packed records with different translations ────
print("\n\n" + "=" * 80)
print("INDIC TRANSLATION PACKED DUPLICATES - DEEP ANALYSIS")
print("=" * 80)

# For packed records in the Indic range, split into segments and compare across records
indic_segments_all = []
for i in indic_packed:
    text = packed[i]["text"]
    segs = [s.strip() for s in text.split("<|end_of_text|>") if s.strip() and len(s.strip()) >= 100]
    for si, seg in enumerate(segs):
        indic_segments_all.append({"packed_idx": i, "seg_idx": si, "text": seg})

print(f"Total substantial Indic segments across packed records: {len(indic_segments_all)}")

# Hash and group
seg_hash_groups = defaultdict(list)
for item in indic_segments_all:
    h = hashlib.md5(item["text"].encode()).hexdigest()
    seg_hash_groups[h].append(item)

cross_record_dups = {h: items for h, items in seg_hash_groups.items()
                     if len(items) > 1 and len(set(it["packed_idx"] for it in items)) > 1}

print(f"Segments appearing in multiple packed records: {len(cross_record_dups)}")
for h, items in list(cross_record_dups.items())[:10]:
    packed_indices = [it["packed_idx"] for it in items]
    print(f"\n  Segment in packed records {packed_indices}:")
    print(f"    Preview: {items[0]['text'][:200]}...")

# ── 6. Specifically check the duplicate groups for Tamil/Telugu ──────────────
print("\n\n" + "=" * 80)
print("SPECIFIC DUPLICATE GROUPS (from exact dup analysis)")
print("=" * 80)

for text, indices in packed_exact_dups.items():
    has_tamil = "Tamil" in text or "தமிழ்" in text
    has_telugu = "Telugu" in text or "తెలుగు" in text
    has_hindi = "Hindi" in text or "हिन्दी" in text or "हिंदी" in text
    has_bengali = "Bengali" in text or "বাংলা" in text
    lang = []
    if has_tamil: lang.append("Tamil")
    if has_telugu: lang.append("Telugu")
    if has_hindi: lang.append("Hindi")
    if has_bengali: lang.append("Bengali")
    if not lang: lang.append("Other")

    print(f"\n  Language: {', '.join(lang)}")
    print(f"  Packed indices: {indices}")
    print(f"  Text length: {len(text)}")
    # Count segments in this packed record
    segs = [s.strip() for s in text.split("<|end_of_text|>") if s.strip()]
    print(f"  Number of segments: {len(segs)}")
    if segs:
        print(f"  First segment preview: {segs[0][:200]}...")
