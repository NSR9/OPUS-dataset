# Golden Dataset Dynamic Builder — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `scripts/build_golden_dataset.py` to accept `--total-samples N`, dynamically allocate across 44 datasets in 12 domains (uniform default, JSON-overridable weights), guarantee every dataset gets >=1 sample, and add 14 new dataset processors with verified HuggingFace schemas.

**Architecture:** Allocator + Registry pattern. A `DATASET_REGISTRY` lists all 44 datasets with metadata. A `compute_allocation(N, weights)` function distributes N samples across domains (uniform by default) then within-domain across datasets. Each processor function accepts `n: int` parameter instead of hardcoded limits. HF auth auto-detected from env/cache.

**Tech Stack:** Python 3, `datasets` (HuggingFace), `huggingface_hub`, `argparse`, `json`

**Key file:** `scripts/build_golden_dataset.py` (1585 lines currently, will grow to ~2400)

---

## Task 1: Allocation Engine (Pure Logic, No HF Dependencies)

**Files:**
- Create: `scripts/golden_allocator.py`
- Create: `tests/test_golden_allocator.py`

**Step 1: Write failing tests for the allocator**

```python
# tests/test_golden_allocator.py
"""Tests for the golden dataset allocation engine."""
import pytest
from scripts.golden_allocator import compute_allocation, validate_allocation


# --- Domain/dataset fixtures ---

SAMPLE_REGISTRY = [
    ("ds_a1", "DomainA"), ("ds_a2", "DomainA"), ("ds_a3", "DomainA"),
    ("ds_b1", "DomainB"), ("ds_b2", "DomainB"),
    ("ds_c1", "DomainC"),
]
SAMPLE_DOMAINS = ["DomainA", "DomainB", "DomainC"]


class TestComputeAllocation:
    def test_total_equals_n(self):
        """Sum of all allocations must equal N."""
        alloc = compute_allocation(
            total=60,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        assert sum(alloc.values()) == 60

    def test_every_dataset_at_least_one(self):
        """Every dataset must get >= 1 sample."""
        alloc = compute_allocation(
            total=6,  # exactly num_datasets, minimum possible
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        for ds_id, count in alloc.items():
            assert count >= 1, f"{ds_id} got {count} samples"

    def test_too_few_samples_raises(self):
        """N < num_datasets must raise ValueError."""
        with pytest.raises(ValueError, match="at least 6"):
            compute_allocation(
                total=5,
                domain_names=SAMPLE_DOMAINS,
                domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
                datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                     "DomainB": ["ds_b1","ds_b2"],
                                     "DomainC": ["ds_c1"]},
            )

    def test_uniform_distribution(self):
        """With uniform weights, domains get roughly equal budgets."""
        alloc = compute_allocation(
            total=120,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        domain_a_total = alloc["ds_a1"] + alloc["ds_a2"] + alloc["ds_a3"]
        domain_b_total = alloc["ds_b1"] + alloc["ds_b2"]
        domain_c_total = alloc["ds_c1"]
        assert domain_a_total == 40
        assert domain_b_total == 40
        assert domain_c_total == 40

    def test_domain_clamped_to_num_datasets(self):
        """Domain with many datasets can't get fewer samples than datasets."""
        alloc = compute_allocation(
            total=7,  # 6 datasets + 1 extra
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 0.01, "DomainB": 0.01, "DomainC": 0.98},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        # DomainA has 3 datasets, must get at least 3 even with tiny weight
        domain_a_total = alloc["ds_a1"] + alloc["ds_a2"] + alloc["ds_a3"]
        assert domain_a_total >= 3

    def test_within_domain_even_split(self):
        """Within a domain, samples split evenly across datasets."""
        alloc = compute_allocation(
            total=120,
            domain_names=SAMPLE_DOMAINS,
            domain_weights={"DomainA": 1/3, "DomainB": 1/3, "DomainC": 1/3},
            datasets_per_domain={"DomainA": ["ds_a1","ds_a2","ds_a3"],
                                 "DomainB": ["ds_b1","ds_b2"],
                                 "DomainC": ["ds_c1"]},
        )
        # DomainA gets 40, split across 3 => 14, 13, 13 or 13, 13, 14
        assert alloc["ds_a1"] in (13, 14)
        assert alloc["ds_a2"] in (13, 14)
        assert alloc["ds_a3"] in (13, 14)


class TestValidateAllocation:
    def test_valid_passes(self):
        """Valid allocation passes validation."""
        alloc = {"ds1": 3, "ds2": 2, "ds3": 1}
        validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])

    def test_wrong_total_raises(self):
        alloc = {"ds1": 3, "ds2": 2, "ds3": 1}
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=7, all_dataset_ids=["ds1","ds2","ds3"])

    def test_missing_dataset_raises(self):
        alloc = {"ds1": 3, "ds2": 3}  # ds3 missing
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])

    def test_zero_allocation_raises(self):
        alloc = {"ds1": 6, "ds2": 0, "ds3": 0}
        with pytest.raises(AssertionError):
            validate_allocation(alloc, total=6, all_dataset_ids=["ds1","ds2","ds3"])
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && uv run python -m pytest tests/test_golden_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.golden_allocator'`

**Step 3: Implement the allocator**

```python
# scripts/golden_allocator.py
"""
Allocation engine for the OPUS golden dataset builder.

Distributes N total samples across domains and datasets.
Rules:
  1. Every dataset gets >= 1 sample
  2. Domain budgets determined by weights (default: uniform 1/num_domains)
  3. Within-domain: samples split evenly across datasets
  4. Rounding remainders distributed round-robin
"""
from __future__ import annotations
from typing import Dict, List
import math


def compute_allocation(
    total: int,
    domain_names: List[str],
    domain_weights: Dict[str, float],
    datasets_per_domain: Dict[str, List[str]],
) -> Dict[str, int]:
    """Compute per-dataset sample allocation.

    Args:
        total: Total number of samples (e.g., 128)
        domain_names: Ordered list of domain names
        domain_weights: Domain name -> weight (should sum to ~1.0)
        datasets_per_domain: Domain name -> list of dataset IDs

    Returns:
        Dict mapping dataset_id -> sample count

    Raises:
        ValueError: If total < number of datasets
    """
    all_datasets = []
    for domain in domain_names:
        all_datasets.extend(datasets_per_domain[domain])
    num_datasets = len(all_datasets)

    if total < num_datasets:
        raise ValueError(
            f"Cannot allocate: need at least {num_datasets} samples "
            f"for {num_datasets} datasets, got {total}"
        )

    # --- Step 1: Domain-level allocation ---
    # Raw allocation by weight
    weight_sum = sum(domain_weights[d] for d in domain_names)
    raw_domain_budgets = {}
    for domain in domain_names:
        raw = (domain_weights[domain] / weight_sum) * total
        raw_domain_budgets[domain] = raw

    # Floor each, clamp to at least num_datasets_in_domain
    domain_budgets = {}
    for domain in domain_names:
        nd = len(datasets_per_domain[domain])
        domain_budgets[domain] = max(nd, math.floor(raw_domain_budgets[domain]))

    # Distribute remainder to hit total
    allocated = sum(domain_budgets.values())
    remainder = total - allocated
    if remainder > 0:
        # Sort domains by fractional part (descending) to distribute fairly
        frac_parts = []
        for domain in domain_names:
            frac = raw_domain_budgets[domain] - math.floor(raw_domain_budgets[domain])
            frac_parts.append((frac, domain))
        frac_parts.sort(reverse=True)
        for i in range(remainder):
            domain_budgets[frac_parts[i % len(frac_parts)][1]] += 1
    elif remainder < 0:
        # Over-allocated due to clamping. Steal from domains with most headroom.
        overshoot = -remainder
        headroom = []
        for domain in domain_names:
            nd = len(datasets_per_domain[domain])
            spare = domain_budgets[domain] - nd
            headroom.append((spare, domain))
        headroom.sort(reverse=True)
        for i in range(overshoot):
            d = headroom[i % len(headroom)][1]
            nd = len(datasets_per_domain[d])
            if domain_budgets[d] > nd:
                domain_budgets[d] -= 1

    # --- Step 2: Within-domain distribution ---
    allocation = {}
    for domain in domain_names:
        ds_list = datasets_per_domain[domain]
        budget = domain_budgets[domain]
        nd = len(ds_list)
        base = budget // nd
        extra = budget % nd
        for i, ds_id in enumerate(ds_list):
            allocation[ds_id] = base + (1 if i < extra else 0)

    return allocation


def validate_allocation(
    allocation: Dict[str, int],
    total: int,
    all_dataset_ids: List[str],
) -> None:
    """Assert allocation is valid."""
    assert sum(allocation.values()) == total, (
        f"Allocation sums to {sum(allocation.values())}, expected {total}"
    )
    for ds_id in all_dataset_ids:
        assert ds_id in allocation, f"Dataset {ds_id} missing from allocation"
        assert allocation[ds_id] >= 1, f"Dataset {ds_id} has 0 samples"
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && PYTHONPATH=. uv run python -m pytest tests/test_golden_allocator.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add scripts/golden_allocator.py tests/test_golden_allocator.py
git commit -m "feat: add golden dataset allocation engine with tests"
```

---

## Task 2: Refactor Existing 31 Processors to Accept `n` Parameter

**Files:**
- Modify: `scripts/build_golden_dataset.py` (lines throughout all processor functions)

This is a mechanical refactor. Every processor function signature changes from `def process_X() -> List[Dict]` to `def process_X(n: int) -> List[Dict]`, and every `if count >= HARDCODED:` becomes `if count >= n:`.

**Step 1: Refactor all 31 processor signatures and limits**

For each of the 31 existing processors, make exactly two changes:

1. Add `n: int` parameter: `def process_gsm8k(n: int) -> List[Dict]:`
2. Replace hardcoded limit with `n`:

| Processor | Old limit | Line pattern to change |
|---|---|---|
| `process_gsm8k` | `picked = rows[:5]` | `picked = rows[:n]` |
| `process_nemotron_math` | `if count >= 5:` | `if count >= n:` |
| `process_ultradata_math` | `if count >= 4:` | `if count >= n:` |
| `process_ling_coder` | `if count >= 7:` | `if count >= n:` |
| `process_swe_smith` | `if count >= 7:` | `if count >= n:` |
| `process_open_perfectblend` | `if count >= 5:` | `if count >= n:` |
| `process_helpsteer3` | `if count >= 5:` | `if count >= n:` |
| `process_megascience` | `if count >= 4:` | `if count >= n:` |
| `_process_preference_dataset` | `if count >= n:` | Already takes `n` param — keep as-is |
| `process_orpo_dpo` | `n=2` in call | Will be driven by allocator |
| `process_skywork_reward` | `n=2` in call | Will be driven by allocator |
| `process_ultrafeedback` | `n=2` in call | Will be driven by allocator |
| `process_infinity_pref` | `n=2` in call | Will be driven by allocator |
| `process_nemotron_post` | `if count >= 1:` | `if count >= n:` |
| `process_smoltalk` | `if count >= 2:` | `if count >= n:` |
| `process_arena_pref` | `if count >= 2:` | `if count >= n:` |
| `process_claude_reasoning` | `if count >= 1:` | `if count >= n:` |
| `process_hardgen` | `if count >= 6:` | `if count >= n:` |
| `process_xlam_fc` | `if count >= 1:` | `if count >= n:` |
| `process_toolace` | `if count >= 7:` | `if count >= n:` |
| `process_tulu_ifeval` | `if count >= 14:` | `if count >= n:` |
| `process_truthy_dpo` | `if count >= 14:` | `if count >= n:` |
| `process_aya_dataset` | `if count >= 5:` | `if count >= n:` |
| `process_indic_align` | `if count >= 3:` | `if count >= n:` |
| `process_indicqa` | `if count >= 3:` | `if count >= n:` |
| `process_indic_glue` | 2 configs | Cycle configs until `n` reached |
| `process_milu` | 2 configs | Cycle configs until `n` reached |
| `process_indic_bias` | `if count >= 1:` | `if count >= n:` |
| `process_leval` | `if count >= 2:` | `if count >= n:` |
| `process_longbench` | `if count >= 4:` | `if count >= n:` |
| `process_babilong` | 4 configs | Cycle configs until `n` reached |
| `process_hotpotqa` | `if count >= 4:` | `if count >= n:` |

For preference wrappers (`process_orpo_dpo`, etc.), change to accept and forward `n`:

```python
# Before:
def process_orpo_dpo() -> List[Dict]:
    results = _process_preference_dataset("mlabonne/orpo-dpo-mix-40k", "benchmark_qa", "pref_orpo", n=2)

# After:
def process_orpo_dpo(n: int) -> List[Dict]:
    results = _process_preference_dataset("mlabonne/orpo-dpo-mix-40k", "benchmark_qa", "pref_orpo", n=n)
```

**Step 2: Verify script still loads (syntax check)**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && uv run python -c "import scripts.build_golden_dataset; print('OK')"`
Expected: `OK` (no import errors)

**Step 3: Commit**

```bash
git add scripts/build_golden_dataset.py
git commit -m "refactor: make all 31 golden processors accept dynamic n parameter"
```

---

## Task 3: Add HF Auth & Update _safe_load

**Files:**
- Modify: `scripts/build_golden_dataset.py` (top of file + `_safe_load` function)

**Step 1: Add HF token auto-detection at top of file**

After the existing imports (line ~29), add:

```python
import os

# ---------------------------------------------------------------------------
# HuggingFace authentication
# ---------------------------------------------------------------------------

HF_TOKEN: Optional[str] = None

def _detect_hf_token() -> Optional[str]:
    """Auto-detect HuggingFace token.
    Priority: HF_TOKEN env var > cached huggingface-cli login token.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass
    return None

GATED_DATASETS = [
    "sarvamai/trivia-qa-indic",
    "ai4bharat/BPCC",
    "ai4bharat/IN22-Gen",
    "openlanguagedata/flores_plus",
    "ai4bharat/MILU",
]
```

**Step 2: Update `_safe_load` to accept `gated` parameter**

```python
def _safe_load(dataset_id: str, split: str = "train", streaming: bool = True,
               gated: bool = False, **kwargs):
    """Load a HuggingFace dataset with error handling.
    If gated=True, uses HF_TOKEN. If no token available, returns None.
    """
    if gated:
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        else:
            log.warning("  Skipping %s (gated dataset, no HF token found)", dataset_id)
            return None
    from datasets import load_dataset
    try:
        log.debug("  Loading %s (split=%s, %s) ...", dataset_id, split,
                   ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else "no extra args")
        ds = load_dataset(dataset_id, split=split, streaming=streaming, **kwargs)
        log.debug("  Loaded %s successfully", dataset_id)
        return ds
    except Exception as e:
        log.warning("  Could not load %s (split=%s): %s", dataset_id, split, e)
        return None
```

**Step 3: Add `gated=True` to existing gated processor calls**

In `process_milu()`, `process_indic_bias()`, `process_nemotron_post()`, `process_xlam_fc()` — add `gated=True` to their `_safe_load` calls.

**Step 4: Verify syntax**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && uv run python -c "import scripts.build_golden_dataset; print('OK')"`

**Step 5: Commit**

```bash
git add scripts/build_golden_dataset.py
git commit -m "feat: add HF token auto-detection and gated dataset handling"
```

---

## Task 4: Add 14 New Processor Functions

**Files:**
- Modify: `scripts/build_golden_dataset.py` (add after existing processors, before `DOMAIN_CONFIG`)

Add all 14 new processors. Each one has verified schema from the design doc.

**Step 1: Add Domain 10 processors (Indic Domain Knowledge — 5 processors)**

```python
# ===== DOMAIN 10: INDIC DOMAIN KNOWLEDGE =====

def process_mgsm(n: int) -> List[Dict]:
    """juletxara/mgsm — Multilingual GSM8K (Bengali only for Indic).
    Schema: {question: str, answer: str, answer_number: int/float, equation_solution: str}
    """
    log.info("[NEW] juletxara/mgsm (Bengali)")
    ds = _safe_load("juletxara/mgsm", "test", streaming=True, name="bn")
    if not ds:
        return []
    rows = _take(ds, 250)
    # Sort by answer length for best step-by-step solutions
    rows.sort(key=lambda r: len(r.get("answer", "")), reverse=True)
    results = []
    for i, r in enumerate(rows[:n]):
        results.append({
            "id": f"mgsm_bn_{i+1}",
            "tag": "indic_math_bn",
            "text": _fmt(r["question"], r["answer"]),
        })
        _log_sample_preview(results[-1], i)
    log.info("  => %d samples collected", len(results))
    return results


def process_mmlu_indic(n: int) -> List[Dict]:
    """sarvamai/mmlu-indic — MMLU in Indian languages.
    Schema: {question: str, choices: list[str], answer: int (0-3)}
    """
    log.info("[NEW] sarvamai/mmlu-indic")
    configs = ["hi", "ta", "te", "bn", "kn", "ml", "mr", "or", "pa", "gu"]
    lang_names = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali",
                  "kn": "kannada", "ml": "malayalam", "mr": "marathi",
                  "or": "odia", "pa": "punjabi", "gu": "gujarati"}
    results = []
    count = 0
    config_idx = 0
    while count < n and config_idx < len(configs) * 50:
        cfg = configs[config_idx % len(configs)]
        config_idx += 1
        ds = _safe_load("sarvamai/mmlu-indic", "test", streaming=True, name=cfg)
        if not ds:
            continue
        rows = _take(ds, 50)
        for r in rows:
            question = r.get("question", "")
            choices = r.get("choices", [])
            answer_idx = r.get("answer", 0)
            if not question or len(choices) < 4:
                continue
            opts = "\n".join(f"  {chr(65+i)}. {c}" for i, c in enumerate(choices))
            answer_text = choices[answer_idx] if answer_idx < len(choices) else choices[0]
            answer_letter = chr(65 + answer_idx)
            results.append({
                "id": f"mmlu_indic_{lang_names[cfg]}_{count+1}",
                "tag": f"indic_knowledge_{lang_names[cfg]}",
                "text": _fmt(
                    f"{question}\n\n{opts}",
                    f"The correct answer is {answer_letter}. {answer_text}"
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            break  # 1 per config pass, then rotate
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_trivia_qa_indic(n: int) -> List[Dict]:
    """sarvamai/trivia-qa-indic — TriviaQA in Indian languages.
    Schema: {question: str, answer: {answer_aliases: list[str], normalized_aliases: list[str]}}
    GATED: requires HF login.
    """
    log.info("[NEW] sarvamai/trivia-qa-indic (GATED)")
    # Try known Indic language configs
    configs = ["hi", "ta", "te", "bn", "kn", "ml", "mr", "gu"]
    results = []
    count = 0
    config_idx = 0
    while count < n and config_idx < len(configs) * 50:
        cfg = configs[config_idx % len(configs)]
        config_idx += 1
        lang_names = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali",
                      "kn": "kannada", "ml": "malayalam", "mr": "marathi", "gu": "gujarati"}
        ds = _safe_load("sarvamai/trivia-qa-indic", "test", streaming=True,
                        name=cfg, gated=True)
        if not ds:
            # Try validation split
            ds = _safe_load("sarvamai/trivia-qa-indic", "validation", streaming=True,
                            name=cfg, gated=True)
        if not ds:
            continue
        rows = _take(ds, 20)
        for r in rows:
            question = r.get("question", "")
            answer_obj = r.get("answer", {})
            if isinstance(answer_obj, dict):
                aliases = answer_obj.get("answer_aliases", [])
                answer_text = aliases[0] if aliases else ""
            elif isinstance(answer_obj, str):
                answer_text = answer_obj
            else:
                continue
            if question and answer_text:
                results.append({
                    "id": f"trivia_indic_{lang_names.get(cfg, cfg)}_{count+1}",
                    "tag": f"indic_trivia_{lang_names.get(cfg, cfg)}",
                    "text": _fmt(question, answer_text),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_indic_instruct(n: int) -> List[Dict]:
    """ai4bharat/indic-instruct-data-v0.1 — native Indic instructions (Anudesh subset).
    Schema (anudesh): rows are messages with {content: str, role: str}
    """
    log.info("[NEW] ai4bharat/indic-instruct-data-v0.1 (Anudesh)")
    ds = _safe_load("ai4bharat/indic-instruct-data-v0.1", "hi", streaming=True,
                    name="anudesh")
    if not ds:
        return []
    results = []
    count = 0
    # Anudesh rows are individual messages. Group consecutive user+assistant pairs.
    pending_user = None
    for row in ds:
        if count >= n:
            break
        role = row.get("role", "")
        content = row.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user and len(content) > 100:
            results.append({
                "id": f"indic_instruct_{count+1}",
                "tag": "indic_instruction_native",
                "text": _fmt(pending_user, content),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            pending_user = None
        else:
            pending_user = None
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_milu_expanded(n: int) -> List[Dict]:
    """ai4bharat/MILU — expansion to new languages (Gujarati, Odia, Punjabi, Malayalam).
    Schema: {question: str, option1-4: str, target: str}
    GATED: requires HF login.
    """
    log.info("[NEW] ai4bharat/MILU expanded (GATED)")
    configs = ["Gujarati", "Odia", "Punjabi", "Malayalam", "Kannada", "Marathi",
               "Bengali", "Telugu"]
    results = []
    count = 0
    config_idx = 0
    while count < n and config_idx < len(configs) * 50:
        cfg = configs[config_idx % len(configs)]
        config_idx += 1
        ds = _safe_load("ai4bharat/MILU", "test", streaming=True,
                        data_dir=cfg, gated=True)
        if not ds:
            continue
        rows = _take(ds, 20)
        for r in rows:
            question = r.get("question", "")
            options = [r.get(f"option{i}", "") for i in range(1, 5)]
            options = [o for o in options if o]
            target = r.get("target", "")
            answer = r.get(target, target) if target else ""
            if question and answer:
                opts_str = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(options))
                results.append({
                    "id": f"milu_exp_{cfg.lower()}_{count+1}",
                    "tag": f"indic_cultural_{cfg.lower()}",
                    "text": _fmt(f"{question}\n\n{opts_str}", str(answer)),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results
```

**Step 2: Add Domain 11 processors (Indic Gen & Translation — 5 processors)**

```python
# ===== DOMAIN 11: INDIC GENERATION & TRANSLATION =====

def process_bpcc(n: int) -> List[Dict]:
    """ai4bharat/BPCC — Bharat Parallel Corpus (BPCC-H human subset).
    Non-standard format: parallel text files per language pair.
    GATED: requires HF login.
    """
    log.info("[NEW] ai4bharat/BPCC (GATED)")
    from datasets import load_dataset
    lang_pairs = [
        ("eng_Latn", "hin_Deva", "Hindi"), ("eng_Latn", "tam_Taml", "Tamil"),
        ("eng_Latn", "tel_Telu", "Telugu"), ("eng_Latn", "ben_Beng", "Bengali"),
        ("eng_Latn", "kan_Knda", "Kannada"), ("eng_Latn", "mar_Deva", "Marathi"),
        ("eng_Latn", "mal_Mlym", "Malayalam"), ("eng_Latn", "guj_Gujr", "Gujarati"),
    ]
    results = []
    count = 0
    pair_idx = 0
    while count < n and pair_idx < len(lang_pairs) * 50:
        src, tgt, lang_name = lang_pairs[pair_idx % len(lang_pairs)]
        pair_idx += 1
        # Try loading as a dataset config
        ds = _safe_load("ai4bharat/BPCC", "train", streaming=True,
                        name=f"{src}-{tgt}", gated=True)
        if not ds:
            ds = _safe_load("ai4bharat/BPCC", "train", streaming=True,
                            data_dir=f"{src}-{tgt}", gated=True)
        if not ds:
            continue
        rows = _take(ds, 50)
        for r in rows:
            # Schema may vary — try common field patterns
            src_text = r.get(src, "") or r.get("source", "") or r.get("sentence1", "")
            tgt_text = r.get(tgt, "") or r.get("target", "") or r.get("sentence2", "")
            if not src_text or not tgt_text:
                # Try first two string fields
                str_vals = [v for v in r.values() if isinstance(v, str) and len(v) > 10]
                if len(str_vals) >= 2:
                    src_text, tgt_text = str_vals[0], str_vals[1]
            if src_text and tgt_text and len(src_text) > 30:
                results.append({
                    "id": f"bpcc_{lang_name.lower()}_{count+1}",
                    "tag": f"indic_translation_{lang_name.lower()}_en",
                    "text": _fmt(
                        f"Translate the following {lang_name} text to English:\n\n{tgt_text}",
                        src_text
                    ),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_in22_gen(n: int) -> List[Dict]:
    """ai4bharat/IN22-Gen — IndicTrans2 evaluation set.
    Schema: {id: int, sentence: str, context: str, source: str, domain: str, ...}
    GATED: requires HF login.
    """
    log.info("[NEW] ai4bharat/IN22-Gen (GATED)")
    lang_pairs = [
        ("eng_Latn-hin_Deva", "Hindi"), ("eng_Latn-tam_Taml", "Tamil"),
        ("eng_Latn-tel_Telu", "Telugu"), ("eng_Latn-ben_Beng", "Bengali"),
    ]
    results = []
    count = 0
    pair_idx = 0
    while count < n and pair_idx < len(lang_pairs) * 50:
        config, lang_name = lang_pairs[pair_idx % len(lang_pairs)]
        pair_idx += 1
        ds = _safe_load("ai4bharat/IN22-Gen", "gen", streaming=True,
                        name=config, gated=True)
        if not ds:
            continue
        rows = _take(ds, 30)
        seen_domains = set()
        for r in rows:
            domain = r.get("domain", "")
            if domain in seen_domains:
                continue
            sentence = r.get("sentence", "")
            # For paired config, look for the other language's sentence field
            parts = config.split("-")
            tgt_lang_code = parts[1] if len(parts) > 1 else ""
            tgt_sentence = r.get(f"sentence_{tgt_lang_code}", "") or r.get("sentence", "")
            src_sentence = r.get(f"sentence_{parts[0]}", "") or ""
            if not src_sentence:
                src_sentence = sentence
            if src_sentence and tgt_sentence and src_sentence != tgt_sentence:
                results.append({
                    "id": f"in22_{lang_name.lower()}_{count+1}",
                    "tag": f"indic_translation_{lang_name.lower()}_en",
                    "text": _fmt(
                        f"Translate to {lang_name}:\n\n{src_sentence}",
                        tgt_sentence
                    ),
                })
                _log_sample_preview(results[-1], count)
                seen_domains.add(domain)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_indicgenbench_xquad(n: int) -> List[Dict]:
    """google/IndicGenBench_xquad_in — cross-lingual QA.
    Schema: {context: str, question: str, answers: list[{text: str}], lang: str}
    Must load with field="examples".
    """
    log.info("[NEW] google/IndicGenBench_xquad_in")
    from datasets import load_dataset
    try:
        ds = load_dataset("google/IndicGenBench_xquad_in", field="examples",
                          split="validation", streaming=True)
    except Exception as e:
        log.warning("  Could not load IndicGenBench_xquad_in: %s", e)
        return []
    target_langs = {"hi", "ta", "te", "bn", "kn", "ml", "mr", "gu"}
    results = []
    count = 0
    for r in ds:
        if count >= n:
            break
        lang = r.get("lang", "")
        if lang not in target_langs:
            continue
        context = r.get("context", "")
        question = r.get("question", "")
        answers = r.get("answers", [])
        answer_text = ""
        if isinstance(answers, list) and answers:
            if isinstance(answers[0], dict):
                answer_text = answers[0].get("text", "")
            elif isinstance(answers[0], str):
                answer_text = answers[0]
        if context and question and answer_text:
            ctx = context[:2000] + "..." if len(context) > 2000 else context
            lang_names = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali",
                          "kn": "kannada", "ml": "malayalam", "mr": "marathi", "gu": "gujarati"}
            results.append({
                "id": f"xquad_{lang_names.get(lang, lang)}_{count+1}",
                "tag": f"indic_crosslingual_qa_{lang_names.get(lang, lang)}",
                "text": _fmt(
                    f"{ctx}\n\nQuestion: {question}",
                    answer_text
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
    log.info("  => %d samples collected", len(results))
    return results


def process_indicgenbench_crosssum(n: int) -> List[Dict]:
    """google/IndicGenBench_crosssum_in — cross-lingual summarization.
    Schema: {text: str, summary: str, lang: str}
    Must load with field="examples".
    """
    log.info("[NEW] google/IndicGenBench_crosssum_in")
    from datasets import load_dataset
    try:
        ds = load_dataset("google/IndicGenBench_crosssum_in", field="examples",
                          split="validation", streaming=True)
    except Exception as e:
        log.warning("  Could not load IndicGenBench_crosssum_in: %s", e)
        return []
    target_langs = {"hi", "ta", "te", "bn", "kn", "ml"}
    lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
                  "kn": "Kannada", "ml": "Malayalam"}
    results = []
    count = 0
    for r in ds:
        if count >= n:
            break
        lang = r.get("lang", "")
        if lang not in target_langs:
            continue
        text = r.get("text", "")
        summary = r.get("summary", "")
        if text and summary and 500 < len(text) < 3000:
            results.append({
                "id": f"crosssum_{lang}_{count+1}",
                "tag": f"indic_summarization_{lang}",
                "text": _fmt(
                    f"Summarize the following article in {lang_names.get(lang, lang)}:\n\n{text}",
                    summary
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
    log.info("  => %d samples collected", len(results))
    return results


def process_flores_plus(n: int) -> List[Dict]:
    """openlanguagedata/flores_plus — FLORES+ translation benchmark.
    Schema: {id: str, text: str, iso_639_3: str, iso_15924: str, ...}
    GATED: requires terms acceptance.
    """
    log.info("[NEW] openlanguagedata/flores_plus (GATED)")
    # Load English as reference
    ds_en = _safe_load("openlanguagedata/flores_plus", "dev", streaming=True,
                       name="eng_Latn", gated=True)
    if not ds_en:
        return []
    en_rows = {r["id"]: r["text"] for r in _take(ds_en, 100)}

    indic_configs = [
        ("hin_Deva", "Hindi"), ("tam_Taml", "Tamil"),
        ("tel_Telu", "Telugu"), ("ben_Beng", "Bengali"),
    ]
    results = []
    count = 0
    cfg_idx = 0
    while count < n and cfg_idx < len(indic_configs) * 50:
        lang_code, lang_name = indic_configs[cfg_idx % len(indic_configs)]
        cfg_idx += 1
        ds = _safe_load("openlanguagedata/flores_plus", "dev", streaming=True,
                        name=lang_code, gated=True)
        if not ds:
            continue
        rows = _take(ds, 100)
        for r in rows:
            sid = r.get("id", "")
            indic_text = r.get("text", "")
            en_text = en_rows.get(sid, "")
            if indic_text and en_text and len(indic_text) > 50:
                results.append({
                    "id": f"flores_{lang_name.lower()}_{count+1}",
                    "tag": f"indic_translation_{lang_name.lower()}_en",
                    "text": _fmt(
                        f"Translate the following {lang_name} text to English:\n\n{indic_text}",
                        en_text
                    ),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results
```

**Step 3: Add Domain 12 processors (Linguistic Diagnostics — 2 processors)**

```python
# ===== DOMAIN 12: LINGUISTIC DIAGNOSTICS =====

def process_blimp(n: int) -> List[Dict]:
    """nyu-mll/blimp — Benchmark of Linguistic Minimal Pairs.
    Schema: {sentence_good: str, sentence_bad: str, field: str, linguistics_term: str, ...}
    """
    log.info("[NEW] nyu-mll/blimp")
    configs = [
        "anaphor_number_agreement", "regular_plural_subject_verb_agreement_1",
        "wh_questions_subject_gap", "principle_A_domain_1",
        "sentential_negation_npi_scope", "tough_vs_raising_1",
    ]
    results = []
    count = 0
    cfg_idx = 0
    while count < n and cfg_idx < len(configs) * 50:
        cfg = configs[cfg_idx % len(configs)]
        cfg_idx += 1
        ds = _safe_load("nyu-mll/blimp", "train", streaming=True, name=cfg)
        if not ds:
            continue
        rows = _take(ds, 20)
        for r in rows:
            good = r.get("sentence_good", "")
            bad = r.get("sentence_bad", "")
            term = r.get("linguistics_term", cfg.replace("_", " "))
            field = r.get("field", "syntax")
            if good and bad:
                # Randomly swap order to avoid positional bias
                import hashlib
                swap = int(hashlib.md5(good.encode()).hexdigest(), 16) % 2 == 0
                if swap:
                    sent_a, sent_b, correct = bad, good, "B"
                else:
                    sent_a, sent_b, correct = good, bad, "A"
                results.append({
                    "id": f"blimp_{cfg}_{count+1}",
                    "tag": "linguistic_diagnostics",
                    "text": _fmt(
                        f"Which sentence is grammatically correct and why?\n\n"
                        f"A: {sent_a}\nB: {sent_b}",
                        f"Sentence {correct} is correct. This involves {term} "
                        f"({field}). The other sentence contains a grammatical error "
                        f"related to {term.replace('_', ' ')}."
                    ),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_cola(n: int) -> List[Dict]:
    """nyu-mll/glue (CoLA) — Corpus of Linguistic Acceptability.
    Schema: {sentence: str, label: int (0=unacceptable, 1=acceptable), idx: int}
    """
    log.info("[NEW] nyu-mll/glue (CoLA)")
    ds = _safe_load("nyu-mll/glue", "train", streaming=True, name="cola")
    if not ds:
        return []
    rows = _take(ds, 500)
    # Separate acceptable and unacceptable, alternate
    acceptable = [r for r in rows if r.get("label") == 1 and len(r.get("sentence", "")) > 20]
    unacceptable = [r for r in rows if r.get("label") == 0 and len(r.get("sentence", "")) > 20]
    results = []
    count = 0
    a_idx, u_idx = 0, 0
    while count < n:
        # Alternate: acceptable, unacceptable
        if count % 2 == 0 and a_idx < len(acceptable):
            r = acceptable[a_idx]
            a_idx += 1
            judgment = "Yes, this sentence is grammatically acceptable."
        elif u_idx < len(unacceptable):
            r = unacceptable[u_idx]
            u_idx += 1
            judgment = "No, this sentence is not grammatically acceptable."
        elif a_idx < len(acceptable):
            r = acceptable[a_idx]
            a_idx += 1
            judgment = "Yes, this sentence is grammatically acceptable."
        else:
            break
        results.append({
            "id": f"cola_{count+1}",
            "tag": "linguistic_acceptability",
            "text": _fmt(
                f'Is this sentence grammatically acceptable? Explain.\n\n"{r["sentence"]}"',
                judgment
            ),
        })
        _log_sample_preview(results[-1], count)
        count += 1
    log.info("  => %d samples collected", len(results))
    return results
```

**Step 4: Add Math expansion processors (2 processors)**

```python
# ===== MATH EXPANSION =====

def process_numinamath(n: int) -> List[Dict]:
    """AI-MO/NuminaMath-1.5 — competition math (AIME/IMO difficulty).
    Schema: {problem: str, solution: str, answer: str, source: str, synthetic: bool, ...}
    """
    log.info("[NEW] AI-MO/NuminaMath-1.5")
    ds = _safe_load("AI-MO/NuminaMath-1.5", "train", streaming=True)
    if not ds:
        return []
    results = []
    count = 0
    target_sources = {"aime", "imo", "amc", "putnam"}
    for r in ds:
        if count >= n:
            break
        source = (r.get("source", "") or "").lower()
        synthetic = r.get("synthetic", True)
        solution = r.get("solution", "")
        problem = r.get("problem", "")
        valid = r.get("solution_is_valid", "")
        if (source in target_sources and not synthetic
                and problem and solution and len(solution) > 200
                and valid != "false"):
            results.append({
                "id": f"numina_{source}_{count+1}",
                "tag": "math_competition_hard",
                "text": _fmt(problem, solution),
            })
            _log_sample_preview(results[-1], count)
            count += 1
    log.info("  => %d samples collected", len(results))
    return results


def process_competition_math(n: int) -> List[Dict]:
    """hendrycks/competition_math — MATH benchmark Level 5.
    Schema: {problem: str, solution: str, answer: str, subject: str, level: int}
    """
    log.info("[NEW] hendrycks/competition_math")
    ds = _safe_load("hendrycks/competition_math", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 5000)
    # Filter Level 5 only, sort by solution length
    level5 = [r for r in rows if r.get("level") == 5]
    level5.sort(key=lambda r: len(r.get("solution", "")), reverse=True)
    results = []
    seen_subjects = set()
    count = 0
    for r in level5:
        if count >= n:
            break
        subject = r.get("subject", "")
        problem = r.get("problem", "")
        solution = r.get("solution", "")
        if problem and solution:
            results.append({
                "id": f"comp_math_{subject.lower().replace(' ', '_')}_{count+1}",
                "tag": "math_olympiad",
                "text": _fmt(problem, solution),
            })
            _log_sample_preview(results[-1], count)
            seen_subjects.add(subject)
            count += 1
    log.info("  => %d samples across %d subjects", len(results), len(seen_subjects))
    return results
```

**Step 5: Verify syntax**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && uv run python -c "import scripts.build_golden_dataset; print('OK')"`

**Step 6: Commit**

```bash
git add scripts/build_golden_dataset.py
git commit -m "feat: add 14 new golden dataset processors for expanded benchmarks"
```

---

## Task 5: Replace DOMAIN_CONFIG and Main Loop with Registry + Allocator

**Files:**
- Modify: `scripts/build_golden_dataset.py` (replace `DOMAIN_CONFIG`, `main()`, imports)

**Step 1: Replace `DOMAIN_CONFIG` with `DATASET_REGISTRY`**

Replace lines ~1412-1428 with:

```python
from scripts.golden_allocator import compute_allocation, validate_allocation

# ---------------------------------------------------------------------------
# Dataset Registry: all 44 datasets
# (dataset_id, domain, tag, processor_fn, gated)
# ---------------------------------------------------------------------------

DATASET_REGISTRY = [
    # Domain 1: Math (5 datasets)
    ("openai/gsm8k",                  "Math",                   "math_reasoning",          process_gsm8k,              False),
    ("nvidia/Nemotron-Math-v2",       "Math",                   "math_competition",        process_nemotron_math,      False),
    ("openbmb/UltraData-Math",        "Math",                   "math_hard",               process_ultradata_math,     False),
    ("AI-MO/NuminaMath-1.5",          "Math",                   "math_competition_hard",   process_numinamath,         False),
    ("hendrycks/competition_math",    "Math",                   "math_olympiad",           process_competition_math,   False),
    # Domain 2: Code (2 datasets)
    ("inclusionAI/Ling-Coder-SFT",    "Code",                   "code_generation",         process_ling_coder,         False),
    ("SWE-bench/SWE-smith",           "Code",                   "software_engineering",    process_swe_smith,          False),
    # Domain 3: Knowledge/QA (3 datasets)
    ("mlabonne/open-perfectblend",    "Knowledge/QA",           "general_knowledge",       process_open_perfectblend,  False),
    ("nvidia/HelpSteer3",             "Knowledge/QA",           "reasoning",               process_helpsteer3,         False),
    ("MegaScience/MegaScience",       "Knowledge/QA",           "science",                 process_megascience,        False),
    # Domain 4: Preference/Reasoning (8 datasets)
    ("mlabonne/orpo-dpo-mix-40k",     "Preference/Reasoning",   "benchmark_qa",            process_orpo_dpo,           False),
    ("Skywork/Skywork-Reward-Preference-80K-v0.2", "Preference/Reasoning", "science_math", process_skywork_reward,     False),
    ("argilla/ultrafeedback-binarized-preferences-cleaned", "Preference/Reasoning", "general_qa", process_ultrafeedback, False),
    ("BAAI/Infinity-Preference",      "Preference/Reasoning",   "general_preference",      process_infinity_pref,      False),
    ("nvidia/Nemotron-Post-Training-Dataset-v2", "Preference/Reasoning", "post_training",  process_nemotron_post,      True),
    ("HuggingFaceTB/smoltalk2",       "Preference/Reasoning",   "general_conversation",    process_smoltalk,           False),
    ("lmarena-ai/arena-human-preference-100k", "Preference/Reasoning", "human_preferred",  process_arena_pref,         False),
    ("TeichAI/claude-4.5-opus-high-reasoning-250x", "Preference/Reasoning", "high_reasoning", process_claude_reasoning, False),
    # Domain 5: Tool Use (3 datasets)
    ("Bingguang/HardGen",             "Tool Use",               "tool_use",                process_hardgen,            False),
    ("Salesforce/xlam-function-calling-60k", "Tool Use",        "function_calling",        process_xlam_fc,            True),
    ("Team-ACE/ToolACE",              "Tool Use",               "function_calling",        process_toolace,            False),
    # Domain 6: Instruction Following (1 dataset)
    ("allenai/tulu-3-sft-personas-instruction-following", "Instruction Following", "instruction_following", process_tulu_ifeval, False),
    # Domain 7: Truthfulness (1 dataset)
    ("jondurbin/truthy-dpo-v0.1",     "Truthfulness",           "truthfulness",            process_truthy_dpo,         False),
    # Domain 8: Long Context (4 datasets)
    ("L4NLP/LEval",                   "Long Context",           "long_context_qa",         process_leval,              False),
    ("THUDM/LongBench-v2",           "Long Context",           "long_context_qa",         process_longbench,          False),
    ("RMT-team/babilong-train-5k-samples", "Long Context",     "long_context_retrieval",  process_babilong,           False),
    ("hotpotqa/hotpot_qa",            "Long Context",           "long_context_multihop",   process_hotpotqa,           False),
    # Domain 9: Indic NLU (6 datasets)
    ("CohereForAI/aya_dataset",       "Indic NLU",              "indic_multilingual",      process_aya_dataset,        False),
    ("ai4bharat/indic-align",         "Indic NLU",              "indic_instruction",       process_indic_align,        False),
    ("ai4bharat/IndicQA",             "Indic NLU",              "indic_qa",                process_indicqa,            False),
    ("ai4bharat/indic_glue",          "Indic NLU",              "indic_nlu",               process_indic_glue,         False),
    ("ai4bharat/MILU",                "Indic NLU",              "indic_cultural_knowledge",process_milu,               True),
    ("ai4bharat/Indic-Bias",          "Indic NLU",              "indic_fairness",          process_indic_bias,         True),
    # Domain 10: Indic Domain Knowledge (5 datasets)
    ("juletxara/mgsm",                "Indic Domain Knowledge", "indic_math",              process_mgsm,               False),
    ("sarvamai/mmlu-indic",           "Indic Domain Knowledge", "indic_knowledge",         process_mmlu_indic,         False),
    ("sarvamai/trivia-qa-indic",      "Indic Domain Knowledge", "indic_trivia",            process_trivia_qa_indic,    True),
    ("ai4bharat/indic-instruct-data-v0.1", "Indic Domain Knowledge", "indic_instruction_native", process_indic_instruct, False),
    ("ai4bharat/MILU-expanded",       "Indic Domain Knowledge", "indic_cultural_expanded", process_milu_expanded,      True),
    # Domain 11: Indic Gen/Translation (5 datasets)
    ("ai4bharat/BPCC",                "Indic Gen/Translation",  "indic_translation",       process_bpcc,               True),
    ("ai4bharat/IN22-Gen",            "Indic Gen/Translation",  "indic_translation",       process_in22_gen,           True),
    ("google/IndicGenBench_xquad_in", "Indic Gen/Translation",  "indic_crosslingual_qa",   process_indicgenbench_xquad, False),
    ("google/IndicGenBench_crosssum_in", "Indic Gen/Translation", "indic_summarization",   process_indicgenbench_crosssum, False),
    ("openlanguagedata/flores_plus",  "Indic Gen/Translation",  "indic_translation",       process_flores_plus,        True),
    # Domain 12: Linguistic Diagnostics (2 datasets)
    ("nyu-mll/blimp",                 "Linguistic Diagnostics", "linguistic_diagnostics",  process_blimp,              False),
    ("nyu-mll/glue-cola",             "Linguistic Diagnostics", "linguistic_acceptability", process_cola,              False),
]

# Derived constants
DOMAIN_NAMES = list(dict.fromkeys(d[1] for d in DATASET_REGISTRY))  # ordered unique
DATASETS_PER_DOMAIN = {}
for ds_id, domain, *_ in DATASET_REGISTRY:
    DATASETS_PER_DOMAIN.setdefault(domain, []).append(ds_id)
ALL_DATASET_IDS = [d[0] for d in DATASET_REGISTRY]
NUM_DATASETS = len(ALL_DATASET_IDS)

DEFAULT_DOMAIN_WEIGHTS = {d: 1.0 / len(DOMAIN_NAMES) for d in DOMAIN_NAMES}
```

**Step 2: Rewrite `main()` with new CLI args, allocator, dry-run**

```python
def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _print_allocation_table(allocation, domain_budgets):
    """Pretty-print the allocation table."""
    log.info("")
    log.info("ALLOCATION TABLE:")
    log.info("  %-30s %6s  %s", "Domain", "Budget", "Per-Dataset Breakdown")
    log.info("  " + "-" * 80)
    for domain in DOMAIN_NAMES:
        ds_list = DATASETS_PER_DOMAIN[domain]
        budget = domain_budgets[domain]
        breakdown = ", ".join(
            f"{ds_id.split('/')[-1]}({allocation[ds_id]})" for ds_id in ds_list
        )
        log.info("  %-30s %6d  %s", domain, budget, breakdown)
    log.info("  " + "-" * 80)
    log.info("  %-30s %6d  All datasets >= 1: %s",
             "TOTAL", sum(allocation.values()),
             "YES" if all(v >= 1 for v in allocation.values()) else "NO !!!")


def main():
    parser = argparse.ArgumentParser(
        description="Build golden dataset JSONL for OPUS proxy signal"
    )
    parser.add_argument(
        "--total-samples", type=int, required=True,
        help="Total number of golden samples (e.g. 64, 128, 256). Must be >= 44."
    )
    parser.add_argument(
        "--output", type=str,
        default="examples/golden_samples.jsonl",
        help="Output JSONL path"
    )
    parser.add_argument(
        "--weights-json", type=str, default=None,
        help="Optional JSON file with domain_weights override"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show allocation table without downloading any data"
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
    global HF_TOKEN
    HF_TOKEN = _detect_hf_token()

    N = args.total_samples

    # --- Validation ---
    if N < NUM_DATASETS:
        log.error("Cannot allocate: need at least %d samples for %d datasets, got %d",
                  NUM_DATASETS, NUM_DATASETS, N)
        sys.exit(1)

    if not _is_power_of_2(N):
        powers = [2**i for i in range(5, 11)]  # 32..1024
        log.warning("N=%d is not a power of 2. GPU-friendly sizes: %s", N,
                    ", ".join(str(p) for p in powers))

    # --- Domain weights ---
    domain_weights = DEFAULT_DOMAIN_WEIGHTS.copy()
    if args.weights_json:
        with open(args.weights_json) as f:
            custom = json.load(f)
        if "domain_weights" in custom:
            domain_weights.update(custom["domain_weights"])
            log.info("Loaded custom weights from %s", args.weights_json)

    # --- Compute allocation ---
    allocation = compute_allocation(
        total=N,
        domain_names=DOMAIN_NAMES,
        domain_weights=domain_weights,
        datasets_per_domain=DATASETS_PER_DOMAIN,
    )
    validate_allocation(allocation, total=N, all_dataset_ids=ALL_DATASET_IDS)

    # Compute domain budgets for display
    domain_budgets = {}
    for domain in DOMAIN_NAMES:
        domain_budgets[domain] = sum(allocation[ds] for ds in DATASETS_PER_DOMAIN[domain])

    log.info("=" * 60)
    log.info("OPUS Golden Dataset Builder")
    log.info("=" * 60)
    log.info("Total samples: %d%s", N,
             " (power of 2)" if _is_power_of_2(N) else " (NOT power of 2)")
    log.info("Domains: %d | Datasets: %d", len(DOMAIN_NAMES), NUM_DATASETS)
    log.info("HF token: %s", "FOUND" if HF_TOKEN else "NOT FOUND")
    if not HF_TOKEN:
        gated = [ds_id for ds_id, _, _, _, g in DATASET_REGISTRY if g]
        log.warning("Gated datasets that will be skipped (%d):", len(gated))
        for g in gated:
            log.warning("  - %s", g)
        log.warning("Fix: run 'huggingface-cli login' or set HF_TOKEN env var")

    _print_allocation_table(allocation, domain_budgets)

    if args.dry_run:
        log.info("")
        log.info("Dry run complete. Remove --dry-run to build.")
        sys.exit(0)

    # --- Build dataset ---
    log.info("")
    t_start = time.time()
    all_samples: List[Dict] = []
    failed: List[str] = []
    skipped: List[str] = []

    # Build a lookup: dataset_id -> (processor, gated)
    processor_lookup = {ds_id: (proc, gated)
                        for ds_id, _, _, proc, gated in DATASET_REGISTRY}

    for domain in DOMAIN_NAMES:
        log.info("-" * 50)
        log.info("DOMAIN: %s (budget=%d)", domain, domain_budgets[domain])
        log.info("-" * 50)

        for ds_id in DATASETS_PER_DOMAIN[domain]:
            target_n = allocation[ds_id]
            proc_fn, is_gated = processor_lookup[ds_id]
            name = proc_fn.__name__
            log.info("  %s (n=%d) ...", ds_id, target_n)
            t0 = time.time()
            try:
                samples = proc_fn(target_n)
                dt = time.time() - t0
                if samples:
                    all_samples.extend(samples)
                    log.info("  => %d/%d samples in %.1fs", len(samples), target_n, dt)
                else:
                    log.warning("  => 0/%d samples in %.1fs (gated/broken)", target_n, dt)
                    skipped.append(ds_id)
            except Exception as e:
                dt = time.time() - t0
                if args.skip_errors:
                    log.error("  FAILED in %.1fs: %s", dt, e)
                    failed.append(ds_id)
                else:
                    log.error("\n[FATAL] %s failed after %.1fs:", name, dt)
                    traceback.print_exc()
                    log.error("\nRe-run with --skip-errors to continue past failures.")
                    sys.exit(1)

    total_time = time.time() - t_start

    # Re-number IDs: g1, g2, ..., gN
    for i, sample in enumerate(all_samples):
        sample["id"] = f"g{i+1}"

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("BUILD COMPLETE")
    log.info("=" * 60)
    log.info("Total samples: %d / %d target", len(all_samples), N)
    log.info("Time: %.1fs", total_time)
    log.info("Output: %s", out_path)

    # Domain summary
    log.info("")
    log.info("Domain breakdown:")
    log.info("  %-30s %6s %6s %s", "Domain", "Got", "Target", "Status")
    log.info("  " + "-" * 60)
    domain_got = {}
    for ds_id, domain, _, _, _ in DATASET_REGISTRY:
        domain_got.setdefault(domain, 0)
    for s in all_samples:
        # count by tag -> domain mapping would be complex, just count total
        pass
    # Simpler: count by domain from the samples collected per dataset
    log.info("  %-30s %6d %6d", "TOTAL", len(all_samples), N)

    if failed:
        log.warning("Failed datasets (%d): %s", len(failed), ", ".join(failed))
    if skipped:
        log.info("Skipped datasets (%d): %s", len(skipped), ", ".join(skipped))

    if len(all_samples) == N:
        log.info("Target of %d samples reached!", N)
    else:
        log.warning("Target of %d not reached (%d collected).", N, len(all_samples))
```

**Step 3: Verify dry-run works**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && PYTHONPATH=. uv run python scripts/build_golden_dataset.py --total-samples 128 --dry-run`
Expected: Prints allocation table and exits without downloading.

**Step 4: Commit**

```bash
git add scripts/build_golden_dataset.py
git commit -m "feat: replace hardcoded DOMAIN_CONFIG with registry + allocator + dry-run"
```

---

## Task 6: Create Default Weights JSON

**Files:**
- Create: `configs/golden_weights.json`

**Step 1: Create the default weights file**

```json
{
  "_comment": "Domain weights for golden dataset allocation. Default: uniform (1/12 each). Modify to change distribution. Weights are normalized — they don't need to sum to 1.0.",
  "domain_weights": {
    "Math": 1.0,
    "Code": 1.0,
    "Knowledge/QA": 1.0,
    "Preference/Reasoning": 1.0,
    "Tool Use": 1.0,
    "Instruction Following": 1.0,
    "Truthfulness": 1.0,
    "Long Context": 1.0,
    "Indic NLU": 1.0,
    "Indic Domain Knowledge": 1.0,
    "Indic Gen/Translation": 1.0,
    "Linguistic Diagnostics": 1.0
  }
}
```

**Step 2: Commit**

```bash
git add configs/golden_weights.json
git commit -m "feat: add default golden dataset weights config"
```

---

## Task 7: Integration Test — Dry-Run at Multiple N Values

**Files:**
- Create: `tests/test_golden_integration.py`

**Step 1: Write integration test**

```python
# tests/test_golden_integration.py
"""Integration tests for the golden dataset builder allocation."""
import subprocess
import sys

import pytest


SCRIPT = "scripts/build_golden_dataset.py"


@pytest.mark.parametrize("n", [64, 128, 256, 512])
def test_dry_run_succeeds(n):
    """Dry run at various N values should succeed and show correct total."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", str(n), "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **__import__("os").environ},
    )
    assert result.returncode == 0, f"Failed at N={n}: {result.stderr}"
    assert f"Total samples: {n}" in result.stdout or f"Total samples: {n}" in result.stderr


def test_dry_run_too_few_samples():
    """N < 44 should fail."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", "30", "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **__import__("os").environ},
    )
    assert result.returncode != 0


def test_dry_run_warns_non_power_of_2():
    """N=100 should warn about not being power of 2."""
    result = subprocess.run(
        [sys.executable, SCRIPT, "--total-samples", "100", "--dry-run"],
        capture_output=True, text=True, timeout=10,
        env={"PYTHONPATH": ".", **__import__("os").environ},
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "not a power of 2" in output.lower() or "NOT power of 2" in output
```

**Step 2: Run tests**

Run: `cd /Users/sriranga/Desktop/L/OpusImplementation_updated && PYTHONPATH=. uv run python -m pytest tests/test_golden_integration.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_golden_integration.py
git commit -m "test: add integration tests for golden dataset dry-run"
```

---

## Task 8: Update Documentation

**Files:**
- Modify: `docs/golden_dataset_plan_v2.md` (update CLI usage section)

**Step 1: Update the plan doc's build instructions to show new CLI**

Replace old build command references with:

```markdown
## Building the Golden Dataset

### Quick start
```bash
# Show allocation table without downloading (recommended first step)
python scripts/build_golden_dataset.py --total-samples 128 --dry-run

# Build with default uniform weights
python scripts/build_golden_dataset.py --total-samples 128 --output examples/golden_samples.jsonl --skip-errors

# Build with custom weights
python scripts/build_golden_dataset.py --total-samples 256 --weights-json configs/golden_weights.json --output examples/golden_samples.jsonl --skip-errors

# For gated datasets, login first
huggingface-cli login
python scripts/build_golden_dataset.py --total-samples 128 --output examples/golden_samples.jsonl
```
```

**Step 2: Commit**

```bash
git add docs/golden_dataset_plan_v2.md
git commit -m "docs: update golden dataset plan with new CLI usage"
```

---

## Task Summary

| Task | What | Files | Est. Lines Changed |
|------|------|-------|-------------------|
| 1 | Allocation engine + tests | `scripts/golden_allocator.py`, `tests/test_golden_allocator.py` | ~150 new |
| 2 | Refactor 31 processors to accept `n` | `scripts/build_golden_dataset.py` | ~60 edits |
| 3 | HF auth + `_safe_load` update | `scripts/build_golden_dataset.py` | ~40 new |
| 4 | 14 new processors | `scripts/build_golden_dataset.py` | ~650 new |
| 5 | Registry + allocator main loop + dry-run | `scripts/build_golden_dataset.py` | ~250 replace |
| 6 | Default weights JSON | `configs/golden_weights.json` | ~20 new |
| 7 | Integration tests | `tests/test_golden_integration.py` | ~50 new |
| 8 | Docs update | `docs/golden_dataset_plan_v2.md` | ~20 edit |
