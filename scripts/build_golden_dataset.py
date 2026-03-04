#!/usr/bin/env python3
"""
Build the Golden Dataset JSONL for OPUS proxy signal.

Pulls the best samples from 44 HuggingFace datasets across 12 domains,
converts them to the canonical format, and outputs golden_samples.jsonl.

Golden format:
  {"id": "g1", "tag": "math_reasoning", "text": "<|user|> ...\n\n<|assistant|> ..."}

Usage:
  pip install datasets huggingface_hub
  python scripts/build_golden_dataset.py --output examples/golden_samples.jsonl

Requires internet access to download from HuggingFace.
Some gated datasets may require: huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FMT = "%H:%M:%S"

log = logging.getLogger("golden_builder")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))
    log.setLevel(level)
    log.addHandler(handler)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(s: str) -> str:
    """Strip control characters (except newline/tab) to ensure clean text."""
    return "".join(c for c in s if c in "\n\t" or ord(c) >= 32)


def _fmt(user: str, assistant: str) -> str:
    """Canonical <|user|>/<|assistant|> formatting for SFT loss masking."""
    return f"<|user|> {_clean(user).strip()}\n\n<|assistant|> {_clean(assistant).strip()}"


# Regex covering CJK Unified Ideographs + Extension A/B + CJK Compatibility
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\u3000-\u303f]')


def _has_cjk(text: str) -> bool:
    """Return True if text contains any CJK (Chinese/Japanese/Korean) characters."""
    return bool(_CJK_RE.search(text))


def _fmt_fim(prefix: str, suffix: str, middle: str) -> str:
    """Fill-in-the-middle formatting using <|fim_*|> boundary tokens."""
    return (
        f"<|fim_prefix|>{_clean(prefix).strip()}"
        f"<|fim_suffix|>{_clean(suffix).strip()}"
        f"<|fim_middle|>{_clean(middle).strip()}"
    )


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


def _take(ds, n: int) -> List[Dict]:
    """Take first n items from a (possibly streaming) dataset."""
    out = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        out.append(row)
    log.debug("  Streamed %d rows (requested %d)", len(out), n)
    return out


def _pick_best(rows: List[Dict], key: str, n: int, reverse: bool = True) -> List[Dict]:
    """Pick top-n rows by a numeric key."""
    valid = [r for r in rows if r.get(key) is not None]
    valid.sort(key=lambda r: float(r[key]), reverse=reverse)
    return valid[:n]


def _log_sample_preview(sample: Dict, idx: int) -> None:
    """Log a truncated preview of a sample for debugging."""
    text = sample.get("text", "")
    preview = text[:150].replace("\n", " ") + ("..." if len(text) > 150 else "")
    log.debug("    Sample %d [%s]: %s", idx, sample.get("tag", "?"), preview)


# ---------------------------------------------------------------------------
# Per-dataset processors
# Each returns a list of {"id": str, "tag": str, "text": str}
# ---------------------------------------------------------------------------

# ===== DOMAIN 1: MATH =====

def process_gsm8k(n: int) -> List[Dict]:
    """openai/gsm8k — grade school math word problems.
    Verified structure: {question: str, answer: str}
    """
    log.info("[1/31] openai/gsm8k")
    ds = _safe_load("openai/gsm8k", "train", streaming=True, name="main")
    if not ds:
        return []
    rows = _take(ds, 500)
    log.info("  Sorting %d rows by answer length for best step-by-step solutions", len(rows))
    rows.sort(key=lambda r: len(r.get("answer", "")), reverse=True)
    picked = rows[:n]
    results = []
    for i, r in enumerate(picked):
        results.append({
            "id": f"math_gsm8k_{i+1}",
            "tag": "math_reasoning",
            "text": _fmt(r["question"], r["answer"]),
        })
        _log_sample_preview(results[-1], i)
    log.info("  => %d samples collected", len(results))
    return results


def process_nemotron_math(n: int) -> List[Dict]:
    """nvidia/Nemotron-Math-v2 — competition-level math.
    Verified structure: {problem: str, messages: [{role, content}], expected_answer: str, ...}
    Available splits: high_part00, high_part01, high_part02, medium, low (NO 'train' split)
    """
    log.info("[2/31] nvidia/Nemotron-Math-v2")
    # Try high difficulty splits first
    ds = _safe_load("nvidia/Nemotron-Math-v2", "high_part00", streaming=True)
    if not ds:
        ds = _safe_load("nvidia/Nemotron-Math-v2", "medium", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        # Primary: use messages format (verified: role/content)
        prompt = r.get("problem", "")
        answer = ""
        if "messages" in r and isinstance(r["messages"], list):
            msgs = r["messages"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if asst_parts:
                answer = asst_parts[-1]
            if not prompt:
                user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
                if user_parts:
                    prompt = user_parts[0]
        if not answer:
            answer = r.get("expected_answer", "")
        if not prompt or not answer:
            continue
        if len(answer) > 200:
            log.debug("  Found good math sample: problem=%d chars, answer=%d chars", len(prompt), len(answer))
            results.append({
                "id": f"math_nemotron_{count+1}",
                "tag": "math_competition",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_ultradata_math(n: int) -> List[Dict]:
    """openbmb/UltraData-Math — hard math (L3).
    Verified structure: {uid: str, content: str} (single content field with full Q+A)
    Requires config name, e.g. 'UltraData-Math-L3-QA-Synthetic'
    """
    log.info("[3/31] openbmb/UltraData-Math")
    ds = _safe_load("openbmb/UltraData-Math", "train", streaming=True,
                    name="UltraData-Math-L3-QA-Synthetic")
    if not ds:
        # Fallback to other L3 configs
        ds = _safe_load("openbmb/UltraData-Math", "train", streaming=True,
                        name="UltraData-Math-L3-Conversation-Synthetic")
    if not ds:
        return []
    rows = _take(ds, 300)
    results = []
    count = 0
    for r in rows:
        # UltraData-Math has a single 'content' field with the full problem+solution
        content = r.get("content", "")
        if not content or len(content) < 300:
            continue
        # Split content into problem and solution heuristically
        # The content typically starts with the problem statement
        # Try to find a natural split point
        split_markers = ["\nSolution:", "\nAnswer:", "\n**Solution", "\nTo solve", "\nLet's solve",
                         "\nWe need to", "\nStep 1:", "\n\\[", "\nFirst,"]
        prompt = content
        answer = ""
        for marker in split_markers:
            if marker in content:
                idx = content.index(marker)
                if idx > 50:  # Ensure problem part is substantial
                    prompt = content[:idx].strip()
                    answer = content[idx:].strip()
                    break
        if not answer:
            # Just split roughly in half, first part as problem
            mid = len(content) // 3
            prompt = content[:mid].strip()
            answer = content[mid:].strip()
        log.debug("  UltraData content split: prompt=%d chars, answer=%d chars", len(prompt), len(answer))
        results.append({
            "id": f"math_ultra_{count+1}",
            "tag": "math_hard",
            "text": _fmt(prompt, answer),
        })
        _log_sample_preview(results[-1], count)
        count += 1
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 2: CODE =====

def process_ling_coder(n: int) -> List[Dict]:
    """inclusionAI/Ling-Coder-SFT — code generation.
    Verified structure: {mid: str, messages: [{content, role}], tags: list, languages: list}
    Note: role values are 'HUMAN' (uppercase) and 'assistant'
    """
    log.info("[4/31] inclusionAI/Ling-Coder-SFT")
    ds = _safe_load("inclusionAI/Ling-Coder-SFT", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        if "messages" not in r or not isinstance(r["messages"], list):
            continue
        msgs = r["messages"]
        # Role can be 'HUMAN' or 'user' (uppercase observed in testing)
        user_parts = [m["content"] for m in msgs
                      if m.get("role", "").lower() in ("user", "human")]
        asst_parts = [m["content"] for m in msgs
                      if m.get("role", "").lower() == "assistant"]
        if not user_parts or not asst_parts:
            continue
        prompt = user_parts[0]
        answer = asst_parts[0]
        if not prompt or not answer:
            continue
        # Skip non-English samples (Chinese, etc.)
        if any('\u4e00' <= c <= '\u9fff' for c in prompt[:100]):
            continue
        # Prefer samples with actual code
        if "```" in answer or "def " in answer or "class " in answer:
            log.debug("  Found code sample with languages=%s", r.get("languages", []))
            results.append({
                "id": f"code_ling_{count+1}",
                "tag": "code_generation",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_swe_smith(n: int) -> List[Dict]:
    """SWE-bench/SWE-smith — real-world bug fixes.
    Verified structure: {instance_id, patch, problem_statement, repo, FAIL_TO_PASS, PASS_TO_PASS, ...}
    """
    log.info("[5/31] SWE-bench/SWE-smith")
    ds = _safe_load("SWE-bench/SWE-smith", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 200)
    results = []
    count = 0
    for r in rows:
        problem = r.get("problem_statement", "")
        patch = r.get("patch", "")
        if not problem or not patch:
            continue
        if 200 < len(patch) < 5000 and len(problem) > 100:
            if len(problem) > 2000:
                problem = problem[:2000] + "..."
            log.debug("  Found SWE patch: repo=%s, patch=%d chars", r.get("repo", "?"), len(patch))
            results.append({
                "id": f"swe_smith_{count+1}",
                "tag": "software_engineering",
                "text": _fmt(
                    f"Fix the following issue:\n\n{problem}",
                    f"Here is the patch that fixes the issue:\n\n{patch}"
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 3: KNOWLEDGE / QA =====

def process_open_perfectblend(n: int) -> List[Dict]:
    """mlabonne/open-perfectblend — general knowledge mix.
    Verified structure: {conversations: [{from: human/gpt, value: str}], source: str}
    """
    log.info("[6/31] mlabonne/open-perfectblend")
    ds = _safe_load("mlabonne/open-perfectblend", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        convs = r.get("conversations") or r.get("messages") or []
        if not convs:
            continue
        user_msgs = [c for c in convs if c.get("from") in ("human", "user") or c.get("role") == "user"]
        asst_msgs = [c for c in convs if c.get("from") in ("gpt", "assistant") or c.get("role") == "assistant"]
        if not user_msgs or not asst_msgs:
            continue
        prompt = user_msgs[0].get("value") or user_msgs[0].get("content") or ""
        answer = asst_msgs[0].get("value") or asst_msgs[0].get("content") or ""
        if len(answer) > 200:
            results.append({
                "id": f"knowledge_perfectblend_{count+1}",
                "tag": "general_knowledge",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_helpsteer3(n: int) -> List[Dict]:
    """nvidia/HelpSteer3 — reasoning with helpfulness scores.
    Verified structure: {context: [{role, content}], response1: str, response2: str,
                         overall_preference: int, individual_preference: [{score, reasoning}]}
    Note: NO 'prompt'/'response' fields. Uses context (chat history) + response1/response2.
    overall_preference < 0 means response1 is better, > 0 means response2 is better.
    """
    log.info("[7/31] nvidia/HelpSteer3")
    ds = _safe_load("nvidia/HelpSteer3", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 500)
    log.info("  Sorting by |overall_preference| to find clear winners")
    results = []
    count = 0
    # Sort by magnitude of preference (clearer signal = better sample)
    rows.sort(key=lambda r: abs(r.get("overall_preference", 0) or 0), reverse=True)
    for r in rows:
        context = r.get("context", [])
        pref = r.get("overall_preference", 0)
        # Pick the better response based on preference
        if pref < 0:
            response = r.get("response1", "")
        else:
            response = r.get("response2", "")
        if not response or len(response) < 100:
            continue
        # Extract prompt from context (last user message)
        prompt = ""
        if isinstance(context, list):
            user_msgs = [m for m in context if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                prompt = user_msgs[-1].get("content", "")
        if not prompt:
            continue
        log.debug("  HelpSteer3 sample: pref=%d, response=%d chars", pref, len(response))
        results.append({
            "id": f"knowledge_helpsteer_{count+1}",
            "tag": "reasoning",
            "text": _fmt(prompt, response),
        })
        _log_sample_preview(results[-1], count)
        count += 1
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_megascience(n: int) -> List[Dict]:
    """MegaScience/MegaScience — scientific knowledge.
    Verified structure: {question: str, answer: str, subject: str, reference_answer: str, source: str}
    """
    log.info("[8/31] MegaScience/MegaScience")
    ds = _safe_load("MegaScience/MegaScience", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 300)
    results = []
    count = 0
    for r in rows:
        prompt = r.get("question", "")
        answer = r.get("answer", "")
        if not prompt or not answer:
            continue
        if len(answer) > 200:
            log.debug("  MegaScience: subject=%s, answer=%d chars", r.get("subject", "?"), len(answer))
            results.append({
                "id": f"science_mega_{count+1}",
                "tag": "science",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 4: PREFERENCE / REASONING =====

def _process_preference_dataset(
    name: str, tag: str, id_prefix: str, n: int = 3, sample_pool: int = 300
) -> List[Dict]:
    """Generic processor for DPO/preference datasets. Uses 'chosen' only."""
    ds = _safe_load(name, "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, sample_pool)
    results = []
    count = 0
    for r in rows:
        chosen = r.get("chosen") or r.get("chosen_response") or ""

        if isinstance(chosen, list):
            asst_msgs = [m for m in chosen if m.get("role") == "assistant"]
            user_msgs = [m for m in chosen if m.get("role") == "user"]
            if asst_msgs and user_msgs:
                prompt = user_msgs[0].get("content", "")
                answer = asst_msgs[-1].get("content", "")
            elif asst_msgs:
                prompt = r.get("prompt") or r.get("question") or ""
                answer = asst_msgs[-1].get("content", "")
            else:
                continue
        elif isinstance(chosen, str):
            prompt = r.get("prompt") or r.get("question") or r.get("instruction") or ""
            answer = chosen
        else:
            continue

        if not answer or len(answer) < 100:
            continue
        if not prompt:
            prompt = r.get("input") or r.get("query") or ""
        if not prompt:
            continue

        results.append({
            "id": f"{id_prefix}_{count+1}",
            "tag": tag,
            "text": _fmt(prompt, answer),
        })
        _log_sample_preview(results[-1], count)
        count += 1
        if count >= n:
            break
    return results


def process_orpo_dpo(n: int) -> List[Dict]:
    """mlabonne/orpo-dpo-mix-40k — benchmark-targeted preferences.
    Verified: chosen is list of messages [{role, content}], has prompt field.
    """
    log.info("[9/31] mlabonne/orpo-dpo-mix-40k")
    results = _process_preference_dataset(
        "mlabonne/orpo-dpo-mix-40k", "benchmark_qa", "pref_orpo", n=n
    )
    log.info("  => %d samples collected", len(results))
    return results


def process_skywork_reward(n: int) -> List[Dict]:
    """Skywork/Skywork-Reward-Preference-80K-v0.2 — science/math preferences.
    Verified: chosen is list of messages [{role, content}], has source field.
    """
    log.info("[10/31] Skywork/Skywork-Reward-Preference-80K-v0.2")
    results = _process_preference_dataset(
        "Skywork/Skywork-Reward-Preference-80K-v0.2", "science_math", "pref_skywork", n=n
    )
    log.info("  => %d samples collected", len(results))
    return results


def process_ultrafeedback(n: int) -> List[Dict]:
    """argilla/ultrafeedback-binarized-preferences-cleaned.
    Verified: chosen is list of messages, has prompt, chosen-rating, chosen-model.
    """
    log.info("[11/31] argilla/ultrafeedback-binarized-preferences-cleaned")
    results = _process_preference_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        "general_qa", "pref_ultrafeedback", n=n
    )
    log.info("  => %d samples collected", len(results))
    return results


def process_infinity_pref(n: int) -> List[Dict]:
    """BAAI/Infinity-Preference — early preferences.
    Verified: chosen is list of messages [{role, content}], has prompt, task_category.
    """
    log.info("[12/31] BAAI/Infinity-Preference")
    results = _process_preference_dataset(
        "BAAI/Infinity-Preference", "general_preference", "pref_infinity", n=n
    )
    log.info("  => %d samples collected", len(results))
    return results


def process_nemotron_post(n: int) -> List[Dict]:
    """nvidia/Nemotron-Post-Training-Dataset-v2 — general post-training.
    GATED: Requires HuggingFace access approval.
    """
    log.info("[13/31] nvidia/Nemotron-Post-Training-Dataset-v2 (GATED)")
    # No 'train' split — available splits: stem, chat, math, code, multilingual_*
    ds = _safe_load("nvidia/Nemotron-Post-Training-Dataset-v2", "chat", streaming=True, gated=True)
    if not ds:
        log.warning("  Gated dataset — run 'huggingface-cli login' and request access")
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        if "messages" in r and isinstance(r["messages"], list):
            msgs = r["messages"]
            user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if user_parts and asst_parts and len(asst_parts[0]) > 200:
                results.append({
                    "id": f"pref_nemotron_{count+1}",
                    "tag": "post_training",
                    "text": _fmt(user_parts[0], asst_parts[0]),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                if count >= n:
                    break
    log.info("  => %d samples collected", len(results))
    return results


def process_smoltalk(n: int) -> List[Dict]:
    """HuggingFaceTB/smoltalk2 — general conversation.
    Verified structure: {messages: [{role, content}], chat_template_kwargs: dict, source: str}
    Requires config='SFT', split is a named subset not 'train'.
    """
    log.info("[14/31] HuggingFaceTB/smoltalk2")
    # smoltalk2 requires config='SFT' and uses named splits
    # Use magpie_ultra which has longer, higher-quality responses
    ds = _safe_load("HuggingFaceTB/smoltalk2", "smoltalk_smollm3_smol_magpie_ultra_no_think",
                    streaming=True, name="SFT")
    if not ds:
        ds = _safe_load("HuggingFaceTB/smoltalk2", "smoltalk_smollm3_everyday_conversations_no_think",
                        streaming=True, name="SFT")
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        if "messages" in r and isinstance(r["messages"], list):
            msgs = r["messages"]
            user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
            asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
            if user_parts and asst_parts and len(asst_parts[0]) > 50:
                results.append({
                    "id": f"general_smoltalk_{count+1}",
                    "tag": "general_conversation",
                    "text": _fmt(user_parts[0], asst_parts[0]),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                if count >= n:
                    break
    log.info("  => %d samples collected", len(results))
    return results


def process_arena_pref(n: int) -> List[Dict]:
    """lmarena-ai/arena-human-preference-100k — human-preferred winning responses.
    Verified structure: {conversation_a: [{role,content}], conversation_b: [{role,content}],
                         winner: str ('model_a'|'model_b'|'tie'|'tie (bothbad)'), model_a, model_b, ...}
    Note: NOT a standard chosen/rejected format. Must pick winner's conversation.
    """
    log.info("[15/31] lmarena-ai/arena-human-preference-100k")
    ds = _safe_load("lmarena-ai/arena-human-preference-100k", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0
    for r in rows:
        winner = r.get("winner", "")
        if "model_a" in winner:
            conv = r.get("conversation_a", [])
        elif "model_b" in winner:
            conv = r.get("conversation_b", [])
        else:
            continue  # Skip ties
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        user_msgs = [m for m in conv if m.get("role") == "user"]
        asst_msgs = [m for m in conv if m.get("role") == "assistant"]
        if not user_msgs or not asst_msgs:
            continue
        prompt = user_msgs[0].get("content", "")
        answer = asst_msgs[0].get("content", "")
        if prompt and answer and len(answer) > 100:
            log.debug("  Arena: winner=%s, model=%s", winner,
                      r.get("model_a") if "model_a" in winner else r.get("model_b"))
            results.append({
                "id": f"pref_arena_{count+1}",
                "tag": "human_preferred",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_claude_reasoning(n: int) -> List[Dict]:
    """TeichAI/claude-4.5-opus-high-reasoning-250x — high reasoning.
    Verified structure: {messages: [{role: system|user|assistant, content: str}]}
    """
    log.info("[16/31] TeichAI/claude-4.5-opus-high-reasoning-250x")
    ds = _safe_load("TeichAI/claude-4.5-opus-high-reasoning-250x", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 100)
    results = []
    count = 0
    for r in rows:
        if "messages" not in r or not isinstance(r["messages"], list):
            continue
        msgs = r["messages"]
        user_parts = [m["content"] for m in msgs if m.get("role") == "user"]
        asst_parts = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if not user_parts or not asst_parts:
            continue
        prompt, answer = user_parts[0], asst_parts[0]
        if len(answer) > 300:
            results.append({
                "id": f"reasoning_claude_{count+1}",
                "tag": "high_reasoning",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 5: TOOL USE =====

def process_hardgen(n: int) -> List[Dict]:
    """Bingguang/HardGen — hard tool-use agent trajectories.
    Verified structure: Keys are numbered columns '0','1','2',... representing conversation turns.
    Each value is a dict {content: str, role: str} or None.
    Turn 0 = system, 1 = user, 2 = assistant (think), 3 = tool response, etc.
    """
    log.info("[17/31] Bingguang/HardGen")
    ds = _safe_load("Bingguang/HardGen", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 100)
    results = []
    count = 0
    for r in rows:
        # Keys are '0', '1', '2', ... — numbered turn columns
        user_msg = ""
        asst_msg = ""
        for key in sorted(r.keys(), key=lambda k: int(k) if k.isdigit() else 999):
            if not key.isdigit():
                continue
            turn = r[key]
            if not isinstance(turn, dict):
                continue
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user" and not user_msg:
                user_msg = content
            elif role == "assistant" and user_msg and not asst_msg:
                asst_msg = content
                break
        if user_msg and asst_msg and len(asst_msg) > 50:
            log.debug("  HardGen: user=%d chars, assistant=%d chars", len(user_msg), len(asst_msg))
            results.append({
                "id": f"tool_hardgen_{count+1}",
                "tag": "tool_use",
                "text": _fmt(user_msg, asst_msg),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_xlam_fc(n: int) -> List[Dict]:
    """Salesforce/xlam-function-calling-60k — verified function calling.
    GATED: Requires HuggingFace access approval.
    Expected structure: {query: str, tools: str/list, answers: str/list}
    """
    log.info("[18/31] Salesforce/xlam-function-calling-60k (GATED)")
    ds = _safe_load("Salesforce/xlam-function-calling-60k", "train", streaming=True, gated=True)
    if not ds:
        log.warning("  Gated dataset — run 'huggingface-cli login' and request access")
        return []
    rows = _take(ds, 200)
    results = []
    count = 0
    for r in rows:
        query = r.get("query", "")
        tools = r.get("tools", "")
        answers = r.get("answers", "")
        if not query:
            continue
        if isinstance(tools, str):
            tools_str = tools
        else:
            tools_str = json.dumps(tools, indent=2)
        if isinstance(answers, str):
            answers_str = answers
        else:
            answers_str = json.dumps(answers, indent=2)
        prompt = f"{query}\n\nAvailable tools:\n{tools_str[:1500]}"
        answer = f"I'll call the appropriate function(s):\n\n{answers_str}"
        results.append({
            "id": f"tool_xlam_{count+1}",
            "tag": "function_calling",
            "text": _fmt(prompt, answer),
        })
        _log_sample_preview(results[-1], count)
        count += 1
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_toolace(n: int) -> List[Dict]:
    """Team-ACE/ToolACE — SOTA function calling.
    Verified structure: {system: str, conversations: [{from: user/assistant/tool, value: str}]}
    """
    log.info("[19/31] Team-ACE/ToolACE")
    ds = _safe_load("Team-ACE/ToolACE", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 200)
    results = []
    count = 0
    for r in rows:
        system = r.get("system", "")
        convs = r.get("conversations", [])
        if not convs:
            continue
        user_parts = [c["value"] for c in convs if c.get("from") == "user"]
        asst_parts = [c["value"] for c in convs if c.get("from") == "assistant"]
        if user_parts and asst_parts:
            prompt = user_parts[0]
            if system and len(system) < 1500:
                prompt = f"System: {system[:1000]}\n\n{prompt}"
            results.append({
                "id": f"tool_ace_{count+1}",
                "tag": "function_calling",
                "text": _fmt(prompt, asst_parts[0]),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 6: INSTRUCTION FOLLOWING =====

def process_tulu_ifeval(n: int) -> List[Dict]:
    """allenai/tulu-3-sft-personas-instruction-following — IFEval-style constraints.
    Verified structure: {id: str, prompt: str, messages: [{role, content}], constraints: [str]}
    """
    log.info("[20/31] allenai/tulu-3-sft-personas-instruction-following")
    ds = _safe_load("allenai/tulu-3-sft-personas-instruction-following", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 300)
    results = []
    count = 0
    for r in rows:
        prompt = r.get("prompt", "")
        msgs = r.get("messages", [])
        if not msgs or not isinstance(msgs, list) or len(msgs) < 2:
            continue
        asst_msgs = [m for m in msgs if m.get("role") == "assistant"]
        if not asst_msgs:
            continue
        answer = asst_msgs[0].get("content", "")
        constraints = r.get("constraints", [])
        if prompt and answer and len(answer) > 100:
            log.debug("  IFEval: constraints=%s", constraints)
            results.append({
                "id": f"ifeval_tulu_{count+1}",
                "tag": "instruction_following",
                "text": _fmt(prompt, answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 7: TRUTHFULQA =====

def process_truthy_dpo(n: int) -> List[Dict]:
    """jondurbin/truthy-dpo-v0.1 — misconception resistance.
    Verified structure: {id: str, source: str, system: str, prompt: str, chosen: str, rejected: str}
    Note: chosen/rejected are plain strings, not message lists.
    """
    log.info("[21/31] jondurbin/truthy-dpo-v0.1")
    ds = _safe_load("jondurbin/truthy-dpo-v0.1", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 200)
    results = []
    count = 0
    for r in rows:
        prompt = r.get("prompt", "")
        chosen = r.get("chosen", "")
        if not prompt or not chosen:
            continue
        if len(chosen) > 50:
            results.append({
                "id": f"truthful_dpo_{count+1}",
                "tag": "truthfulness",
                "text": _fmt(prompt, chosen),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 8: INDIC LANGUAGES =====

def process_aya_dataset(n: int) -> List[Dict]:
    """CohereForAI/aya_dataset — human-annotated multilingual (Indic focus).
    Verified structure: {inputs: str, targets: str, language: str, language_code: str,
                         annotation_type: str, user_id: str}
    """
    log.info("[22/31] CohereForAI/aya_dataset")
    ds = _safe_load("CohereForAI/aya_dataset", "train", streaming=True)
    if not ds:
        return []

    target_langs = {
        "Hindi": "hin", "Tamil": "tam", "Telugu": "tel",
        "Bengali": "ben", "Kannada": "kan", "Marathi": "mar",
        "Malayalam": "mal", "Gujarati": "guj", "Punjabi": "pan",
        "Odia": "ori",
    }
    lang_samples: Dict[str, List[Dict]] = {lang: [] for lang in target_langs}
    seen = 0
    for r in ds:
        seen += 1
        if seen > 5000:
            break
        lang_name = r.get("language", "")
        lang_code = r.get("language_code", "")
        matched_lang = None
        for name, code in target_langs.items():
            if name.lower() in lang_name.lower() or code in lang_code.lower():
                matched_lang = name
                break
        if not matched_lang:
            continue
        inputs = r.get("inputs", "")
        targets = r.get("targets", "")
        if inputs and targets and len(targets) > 30:
            if len(lang_samples[matched_lang]) < 2:
                lang_samples[matched_lang].append({
                    "inputs": inputs, "targets": targets, "lang": matched_lang,
                })

    results = []
    count = 0
    for lang, samples in lang_samples.items():
        for s in samples[:1]:
            results.append({
                "id": f"indic_aya_{lang.lower()}_{count+1}",
                "tag": f"indic_{lang.lower()}",
                "text": _fmt(s["inputs"], s["targets"]),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
        if count >= n:
            break
    found_langs = sum(1 for v in lang_samples.values() if v)
    log.info("  => %d samples across %d/%d target languages", len(results), found_langs, len(target_langs))
    return results


def process_indic_align(n: int) -> List[Dict]:
    """ai4bharat/indic-align — native Indic instruction data (Anudesh subset).
    Verified structure: {id: str, interactions: [[user_text, assistant_text], ...], num_turns: int}
    Config must be 'Anudesh' (capital A), NOT 'anudesh'.
    """
    log.info("[23/31] ai4bharat/indic-align")
    # Try Anudesh first (interactions format), fallback to Indic_ShareLlama (multi-lang columns)
    config_used = None
    for cfg in ["Anudesh", "Indic_ShareLlama", "Dolly_T", "Wiki_Conv"]:
        ds = _safe_load("ai4bharat/indic-align", "train", streaming=True, name=cfg)
        if ds:
            config_used = cfg
            break
    if not ds:
        return []
    rows = _take(ds, 500)
    results = []
    count = 0

    if config_used == "Indic_ShareLlama":
        # Multi-language columns: eng_Latn, hin_Deva, tam_Taml, etc.
        # Each column value is [[user_text, assistant_text], ...]
        lang_cols = ["hin_Deva", "tam_Taml", "tel_Telu", "ben_Beng",
                     "kan_Knda", "mal_Mlym", "mar_Deva"]
        for r in rows:
            for col in lang_cols:
                turns = r.get(col)
                if not turns or not isinstance(turns, list):
                    continue
                for turn in turns:
                    if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                        user_text = str(turn[0]).strip()
                        asst_text = str(turn[1]).strip()
                        if user_text and asst_text and len(asst_text) > 50:
                            results.append({
                                "id": f"indic_align_{count+1}",
                                "tag": "indic_instruction",
                                "text": _fmt(user_text, asst_text),
                            })
                            _log_sample_preview(results[-1], count)
                            count += 1
                            break
                if count >= n:
                    break
            if count >= n:
                break
    else:
        # Anudesh format: interactions: [[user_text, assistant_text], ...]
        for r in rows:
            interactions = r.get("interactions", [])
            if not interactions or not isinstance(interactions, list):
                continue
            for turn in interactions:
                if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                    user_text = str(turn[0]).strip()
                    asst_text = str(turn[1]).strip()
                    if user_text and asst_text and len(asst_text) > 50:
                        results.append({
                            "id": f"indic_align_{count+1}",
                            "tag": "indic_instruction",
                            "text": _fmt(user_text, asst_text),
                        })
                        _log_sample_preview(results[-1], count)
                        count += 1
                        break
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_indicqa(n: int) -> List[Dict]:
    """ai4bharat/IndicQA — QA in 11 Indic languages.
    Legacy dataset script — load via direct parquet URLs from refs/convert/parquet branch.
    Structure: {id, context, question, answers: {text: [str], answer_start: [int]}}
    """
    log.info("[24/31] ai4bharat/IndicQA (via parquet branch)")
    from datasets import load_dataset
    target_configs = ["indicqa.hi", "indicqa.ta", "indicqa.te", "indicqa.bn",
                      "indicqa.kn", "indicqa.mr"]
    results = []
    count = 0
    for config in target_configs:
        lang = config.split(".")[-1]
        lang_name = {"hi": "hindi", "ta": "tamil", "te": "telugu",
                     "bn": "bengali", "kn": "kannada", "mr": "marathi"}.get(lang, lang)
        parquet_url = (
            f"https://huggingface.co/datasets/ai4bharat/IndicQA/"
            f"resolve/refs%2Fconvert%2Fparquet/{config}/test/0000.parquet"
        )
        try:
            log.debug("  Loading IndicQA %s via parquet ...", config)
            ds = load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
        except Exception as e:
            log.warning("  Could not load IndicQA %s: %s", config, e)
            continue
        rows = _take(ds, 10)
        for r in rows:
            context = r.get("context", "")
            question = r.get("question", "")
            answers = r.get("answers", {})
            if isinstance(answers, dict):
                answer_text = (answers.get("text") or [""])[0] if "text" in answers else ""
            elif isinstance(answers, list) and answers:
                answer_text = answers[0] if isinstance(answers[0], str) else str(answers[0])
            else:
                answer_text = str(answers)
            if question and answer_text:
                ctx = context[:500] + "..." if len(context) > 500 else context
                results.append({
                    "id": f"indic_qa_{lang_name}_{count+1}",
                    "tag": f"indic_{lang_name}_qa",
                    "text": _fmt(
                        f"Context: {ctx}\n\nQuestion: {question}",
                        answer_text
                    ),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_indic_glue(n: int) -> List[Dict]:
    """ai4bharat/indic_glue — Indic NLU tasks (CSQA subset).
    Verified structure: {question: str, answer: str, category: str, title: str,
                         options: [str], out_of_context_options: [str]}
    """
    log.info("[25/31] ai4bharat/indic_glue")
    target_configs = ["csqa.hi", "csqa.ta"]
    results = []
    count = 0
    cfg_idx = 0
    while count < n and target_configs:
        config = target_configs[cfg_idx % len(target_configs)]
        cfg_idx += 1
        lang = config.split(".")[-1]
        lang_name = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali"}.get(lang, lang)
        ds = _safe_load("ai4bharat/indic_glue", "test", streaming=True, name=config)
        if not ds:
            if cfg_idx >= len(target_configs):
                break
            continue
        rows = _take(ds, 10 * (cfg_idx // len(target_configs) + 1))
        for r in rows:
            question = r.get("question", "")
            answer = r.get("answer", "")
            options = r.get("options", [])
            if question:
                if options and isinstance(options, list):
                    opts_str = "\n".join(f"  {i+1}. {o}" for i, o in enumerate(options))
                    prompt = f"{question}\n\nOptions:\n{opts_str}"
                else:
                    prompt = question
                if not answer and options:
                    answer = options[0]
                if answer:
                    results.append({
                        "id": f"indic_glue_{lang_name}_{count+1}",
                        "tag": f"indic_{lang_name}_nlu",
                        "text": _fmt(prompt, str(answer)),
                    })
                    _log_sample_preview(results[-1], count)
                    count += 1
                    break
        if cfg_idx >= len(target_configs) * 2:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_milu(n: int) -> List[Dict]:
    """ai4bharat/MILU — India-centric cultural knowledge MCQs.
    GATED: Requires HuggingFace access approval.
    """
    log.info("[26/31] ai4bharat/MILU (GATED)")
    # MILU requires a config name: Hindi, Tamil, Telugu, Bengali, Kannada, etc.
    milu_configs = ["Hindi", "Tamil"]
    results = []
    count = 0
    cfg_idx = 0
    while count < n and milu_configs:
        cfg = milu_configs[cfg_idx % len(milu_configs)]
        cfg_idx += 1
        ds = _safe_load("ai4bharat/MILU", "test", streaming=True, gated=True, name=cfg)
        if not ds:
            if cfg_idx >= len(milu_configs):
                break
            continue
        rows = _take(ds, 20 * (cfg_idx // len(milu_configs) + 1))
        for r in rows:
            question = r.get("question", "")
            # MILU schema: option1/option2/option3/option4, target='option2'
            opt1 = r.get("option1", "")
            opt2 = r.get("option2", "")
            opt3 = r.get("option3", "")
            opt4 = r.get("option4", "")
            target = r.get("target", "")  # e.g. 'option2'
            options = [opt1, opt2, opt3, opt4]
            options = [o for o in options if o]
            # Resolve target to actual answer text
            answer = r.get(target, target) if target else ""
            if question and answer:
                opts_str = "\n".join(f"  {chr(65+i)}. {o}" for i, o in enumerate(options))
                prompt = f"{question}\n\n{opts_str}"
                results.append({
                    "id": f"indic_milu_{cfg.lower()}_{count+1}",
                    "tag": "indic_cultural_knowledge",
                    "text": _fmt(prompt, str(answer)),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if cfg_idx >= len(milu_configs) * 2:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_indic_bias(n: int) -> List[Dict]:
    """ai4bharat/Indic-Bias — fairness across Indian identity groups.
    GATED: Requires HuggingFace access approval.
    """
    log.info("[27/31] ai4bharat/Indic-Bias (GATED)")
    # Indic-Bias requires a config: bias-generation, bias-judgement, stereotype-generation, etc.
    # Split is 'train', not 'test'
    ds = _safe_load("ai4bharat/Indic-Bias", "train", streaming=True, gated=True, name="bias-generation")
    if not ds:
        ds = _safe_load("ai4bharat/Indic-Bias", "train", streaming=True, gated=True, name="stereotype-generation")
    if not ds:
        log.warning("  Gated dataset — run 'huggingface-cli login' and request access")
        return []
    rows = _take(ds, 100)
    results = []
    count = 0
    for r in rows:
        # bias-generation schema: positive_template, negative_template, topic, concept
        pos_template = r.get("positive_template", "")
        neg_template = r.get("negative_template", "")
        topic = r.get("topic", "")
        concept = r.get("concept", "")
        if pos_template and neg_template:
            prompt = (f"Topic: {topic}\nConcept: {concept}\n\n"
                      f"Positive framing: {pos_template}\n"
                      f"Negative framing: {neg_template}\n\n"
                      f"Analyze the bias patterns in these templates.")
            answer = (f"These templates demonstrate bias related to '{concept}' in the context of '{topic}'. "
                      f"The positive template frames an identity group favorably, while the negative template "
                      f"reinforces harmful stereotypes. Both use '<identity>' as a placeholder, showing how "
                      f"the same scenario can be presented with contrasting biases.")
        else:
            prompt = r.get("prompt") or r.get("question") or r.get("text") or ""
            answer = r.get("answer") or r.get("response") or r.get("label") or ""
        if prompt and answer:
            results.append({
                "id": f"indic_bias_{count+1}",
                "tag": "indic_fairness",
                "text": _fmt(str(prompt), str(answer)),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 9: LONG CONTEXT (L-Eval + RULER) =====

def process_leval(n: int) -> List[Dict]:
    """L4NLP/LEval — the actual L-Eval benchmark.
    Legacy dataset script — load via direct parquet URLs from refs/convert/parquet branch.
    Structure: {instructions, input, outputs, source, evaluation}
    """
    log.info("[28/31] L4NLP/LEval (via parquet branch)")
    from datasets import load_dataset
    configs_and_tags = [
        ("quality", "long_context_qa"),
        ("narrative_qa", "long_context_narrative"),
        ("natural_question", "long_context_qa"),
        ("gov_report_summ", "long_context_summarization"),
    ]
    results = []
    count = 0
    num_configs = len(configs_and_tags)
    for idx, (config, tag) in enumerate(configs_and_tags):
        if count >= n:
            break
        parquet_url = (
            f"https://huggingface.co/datasets/L4NLP/LEval/"
            f"resolve/refs%2Fconvert%2Fparquet/{config}/test/0000.parquet"
        )
        try:
            log.debug("  Loading LEval %s via parquet ...", config)
            ds = load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
        except Exception as e:
            log.warning("  Could not load LEval %s: %s", config, e)
            continue
        configs_left = num_configs - idx
        remaining = max(1, (n - count + configs_left - 1) // configs_left)
        rows = _take(ds, remaining * 3)
        cfg_count = 0
        for r in rows:
            doc_input = r.get("input", "")
            instructions = r.get("instructions", "")
            outputs = r.get("outputs", "")
            if not doc_input or not instructions or not outputs:
                continue
            if isinstance(instructions, list):
                question = instructions[0] if instructions else ""
            elif isinstance(instructions, str) and instructions.startswith("["):
                try:
                    inst_list = json.loads(instructions)
                    question = inst_list[0] if inst_list else instructions
                except (json.JSONDecodeError, IndexError):
                    question = instructions
            else:
                question = instructions
            if isinstance(outputs, list):
                answer = outputs[0] if outputs else ""
            elif isinstance(outputs, str) and outputs.startswith("["):
                try:
                    out_list = json.loads(outputs)
                    answer = out_list[0] if out_list else outputs
                except (json.JSONDecodeError, IndexError):
                    answer = outputs
            else:
                answer = outputs
            if not question or not answer:
                continue
            doc_truncated = doc_input[:3000] + "..." if len(doc_input) > 3000 else doc_input
            results.append({
                "id": f"leval_{config}_{count+1}",
                "tag": tag,
                "text": _fmt(f"Document:\n{doc_truncated}\n\nQuestion: {question}", answer),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            cfg_count += 1
            if cfg_count >= remaining:
                break
    log.info("  => %d samples collected", len(results))
    return results


def process_longbench(n: int) -> List[Dict]:
    """THUDM/LongBench-v2 — long context benchmark (replacement for v1 which uses legacy scripts).
    Verified structure: {_id, domain, sub_domain, difficulty, length,
                         question, choice_A/B/C/D, answer, context}
    Note: Using LongBench-v2 because original LongBench uses legacy dataset scripts
    that are no longer supported by datasets>=4.0.
    """
    log.info("[29/31] THUDM/LongBench-v2")
    ds = _safe_load("THUDM/LongBench-v2", "train", streaming=True)
    if not ds:
        return []
    rows = _take(ds, 100)
    results = []
    count = 0
    # Pick diverse domains
    seen_domains = set()
    for r in rows:
        domain = r.get("domain", "")
        if domain in seen_domains:
            continue
        question = r.get("question", "")
        context = r.get("context", "")
        answer_key = r.get("answer", "")
        if not question or not context:
            continue
        # Build answer from choice fields
        choices = {}
        for c in ["A", "B", "C", "D"]:
            val = r.get(f"choice_{c}", "")
            if val:
                choices[c] = val
        answer = choices.get(answer_key, answer_key)
        # Include choices in the question
        choices_str = "\n".join(f"  {k}. {v}" for k, v in choices.items())
        ctx = context[:3000] + "..." if len(context) > 3000 else context
        prompt = f"Context:\n{ctx}\n\nQuestion: {question}\n\n{choices_str}"
        log.debug("  LongBench-v2: domain=%s, difficulty=%s", domain, r.get("difficulty", "?"))
        results.append({
            "id": f"longbench_{domain.lower().replace(' ', '_')}_{count+1}",
            "tag": "long_context_qa",
            "text": _fmt(prompt, answer),
        })
        _log_sample_preview(results[-1], count)
        seen_domains.add(domain)
        count += 1
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_babilong(n: int) -> List[Dict]:
    """RMT-team/babilong-train-5k-samples — RULER-style reasoning in haystacks.
    Verified structure: {input: str, question: str, target: str}
    Requires config (e.g. '0k','4k','8k') and split (e.g. 'qa1'-'qa10').
    """
    log.info("[30/31] RMT-team/babilong-train-5k-samples")
    # Try different context lengths for diversity
    configs_splits = [
        ("4k", "qa1"),   # Short context, simple fact retrieval
        ("8k", "qa2"),   # Medium context, two supporting facts
        ("16k", "qa3"),  # Longer context, three supporting facts
        ("16k", "qa5"),  # Longer context, three-argument relation
    ]
    results = []
    count = 0
    cfg_idx = 0
    while count < n and configs_splits:
        cfg, split = configs_splits[cfg_idx % len(configs_splits)]
        cfg_idx += 1
        ds = _safe_load("RMT-team/babilong-train-5k-samples", split, streaming=True, name=cfg)
        if not ds:
            if cfg_idx >= len(configs_splits):
                break
            continue
        rows = _take(ds, 20 * (cfg_idx // len(configs_splits) + 1))
        for r in rows:
            context = r.get("input", "")
            question = r.get("question", "")
            answer = r.get("target", "")
            if context and question and answer:
                ctx = context[:3000] + "..." if len(context) > 3000 else context
                log.debug("  babilong: config=%s, split=%s, context=%d chars", cfg, split, len(context))
                results.append({
                    "id": f"ruler_babilong_{count+1}",
                    "tag": "long_context_retrieval",
                    "text": _fmt(f"Context:\n{ctx}\n\nQuestion: {question}", answer),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break
        if cfg_idx >= len(configs_splits) * 2:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_hotpotqa(n: int) -> List[Dict]:
    """hotpotqa/hotpot_qa — multi-hop reasoning (RULER skill).
    Verified structure: {id, question, answer, type, level,
                         supporting_facts: {title: [str], sent_id: [int]},
                         context: {title: [str], sentences: [[str]]}}
    """
    log.info("[31/31] hotpotqa/hotpot_qa")
    ds = _safe_load("hotpotqa/hotpot_qa", "train", streaming=True, name="fullwiki")
    if not ds:
        ds = _safe_load("hotpotqa/hotpot_qa", "train", streaming=True, name="distractor")
    if not ds:
        return []
    rows = _take(ds, 300)
    results = []
    count = 0
    for r in rows:
        question = r.get("question", "")
        answer = r.get("answer", "")
        context_data = r.get("context", {})
        if not question or not answer:
            continue
        ctx_parts = []
        if isinstance(context_data, dict):
            titles = context_data.get("title", [])
            sentences = context_data.get("sentences", [])
            for t, s in zip(titles[:3], sentences[:3]):
                if isinstance(s, list):
                    ctx_parts.append(f"{t}: {''.join(s)}")
        ctx = "\n\n".join(ctx_parts)
        if len(answer) > 2 and ctx:
            ctx = ctx[:2000] + "..." if len(ctx) > 2000 else ctx
            log.debug("  HotpotQA: type=%s, level=%s", r.get("type", "?"), r.get("level", "?"))
            results.append({
                "id": f"ruler_hotpot_{count+1}",
                "tag": "long_context_multihop",
                "text": _fmt(
                    f"Using the following documents, answer the question.\n\nDocuments:\n{ctx}\n\nQuestion: {question}",
                    answer
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
            if count >= n:
                break
    log.info("  => %d samples collected", len(results))
    return results


# ===== DOMAIN 10: INDIC DOMAIN KNOWLEDGE =====

def process_mgsm(n: int) -> List[Dict]:
    """ryo0634/mgsm-reformatted — Multilingual GSM8K (Bengali only for Indic).
    The original juletxara/mgsm uses an unsupported dataset script, so we use
    this reformatted mirror instead.
    Schema: {question: str, answer: str, answer_number: int/float, equation_solution: str}
    """
    log.info("[NEW] ryo0634/mgsm-reformatted (Bengali)")
    ds = _safe_load("ryo0634/mgsm-reformatted", "train", streaming=True, name="bn")
    if not ds:
        return []
    rows = _take(ds, 250)
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
            break
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
    configs = ["hi", "ta", "te", "bn", "kn", "ml", "mr", "gu"]
    lang_names = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali",
                  "kn": "kannada", "ml": "malayalam", "mr": "marathi", "gu": "gujarati"}
    results = []
    count = 0
    config_idx = 0
    while count < n and config_idx < len(configs) * 50:
        cfg = configs[config_idx % len(configs)]
        config_idx += 1
        ds = _safe_load("sarvamai/trivia-qa-indic", "test", streaming=True,
                        name=cfg, gated=True)
        if not ds:
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
    Config is "anudesh", split is "hi" (not "train"). Each row has
    {id, messages: [{content, role}, ...], num_turns, model}.
    We extract user/assistant pairs from the messages list.
    """
    log.info("[NEW] ai4bharat/indic-instruct-data-v0.1 (Anudesh)")
    ds = _safe_load("ai4bharat/indic-instruct-data-v0.1", "hi", streaming=True,
                    name="anudesh")
    if not ds:
        return []
    results = []
    count = 0
    for row in ds:
        if count >= n:
            break
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        # Extract first user/assistant pair from the messages list
        user_msg = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                user_msg = content
            elif role == "assistant" and user_msg and len(content) > 100:
                results.append({
                    "id": f"indic_instruct_{count+1}",
                    "tag": "indic_instruction_native",
                    "text": _fmt(user_msg, content),
                })
                _log_sample_preview(results[-1], count)
                count += 1
                break  # Only take first pair per row
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


# ===== DOMAIN 11: INDIC GENERATION & TRANSLATION =====

def process_bpcc(n: int) -> List[Dict]:
    """ai4bharat/BPCC — Bharat Parallel Corpus (bpcc-seed-latest config).
    Config: bpcc-seed-latest. Splits are language codes (hin_Deva, tam_Taml, etc).
    Schema: {src: str, tgt: str, src_lang: str, tgt_lang: str}
    GATED: requires HF login.
    """
    log.info("[NEW] ai4bharat/BPCC (GATED)")
    lang_splits = [
        ("hin_Deva", "Hindi"), ("tam_Taml", "Tamil"),
        ("tel_Telu", "Telugu"), ("ben_Beng", "Bengali"),
        ("kan_Knda", "Kannada"), ("mar_Deva", "Marathi"),
        ("mal_Mlym", "Malayalam"), ("guj_Gujr", "Gujarati"),
    ]
    results = []
    count = 0
    split_idx = 0
    while count < n and split_idx < len(lang_splits):
        lang_code, lang_name = lang_splits[split_idx % len(lang_splits)]
        split_idx += 1
        ds = _safe_load("ai4bharat/BPCC", lang_code, streaming=True,
                        name="bpcc-seed-latest", gated=True)
        if not ds:
            continue
        per_lang = max(1, (n - count) // (len(lang_splits) - split_idx + 1))
        rows = _take(ds, per_lang * 5)
        lang_count = 0
        for r in rows:
            src_text = r.get("src", "")
            tgt_text = r.get("tgt", "")
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
                lang_count += 1
                if lang_count >= per_lang or count >= n:
                    break
        if count >= n:
            break
    log.info("  => %d samples collected", len(results))
    return results


def process_in22_gen(n: int) -> List[Dict]:
    """ai4bharat/IN22-Gen — IndicTrans2 evaluation set.
    Config: default. Split: test.
    Schema: {context, source, domain, eng_Latn, hin_Deva, tam_Taml, ...} (22 langs as columns).
    GATED: requires HF login.
    """
    log.info("[NEW] ai4bharat/IN22-Gen (GATED)")
    lang_cols = [
        ("hin_Deva", "Hindi"), ("tam_Taml", "Tamil"),
        ("tel_Telu", "Telugu"), ("ben_Beng", "Bengali"),
        ("kan_Knda", "Kannada"), ("mar_Deva", "Marathi"),
        ("mal_Mlym", "Malayalam"), ("guj_Gujr", "Gujarati"),
    ]
    ds = _safe_load("ai4bharat/IN22-Gen", "test", streaming=True, gated=True)
    if not ds:
        return []
    rows = _take(ds, n * 10)
    results = []
    count = 0
    lang_idx = 0
    for r in rows:
        if count >= n:
            break
        lang_code, lang_name = lang_cols[lang_idx % len(lang_cols)]
        lang_idx += 1
        en_text = r.get("eng_Latn", "")
        tgt_text = r.get(lang_code, "")
        if en_text and tgt_text and len(en_text) > 30:
            results.append({
                "id": f"in22_{lang_name.lower()}_{count+1}",
                "tag": f"indic_translation_{lang_name.lower()}_en",
                "text": _fmt(
                    f"Translate to {lang_name}:\n\n{en_text}",
                    tgt_text
                ),
            })
            _log_sample_preview(results[-1], count)
            count += 1
    log.info("  => %d samples collected", len(results))
    return results


def process_indicgenbench_xquad(n: int) -> List[Dict]:
    """google/IndicGenBench_xquad_in — cross-lingual QA.
    Load without field="examples" (causes UTF-8 errors). Each row has
    {canary: str, examples: list[{id, question, answers, context, title, lang}]}.
    We flatten the nested examples list ourselves.
    """
    log.info("[NEW] google/IndicGenBench_xquad_in")
    from datasets import load_dataset
    try:
        ds = load_dataset("google/IndicGenBench_xquad_in",
                          split="validation", streaming=True)
    except Exception as e:
        log.warning("  Could not load IndicGenBench_xquad_in: %s", e)
        return []
    target_langs = {"hi", "ta", "te", "bn", "kn", "ml", "mr", "gu"}
    lang_names = {"hi": "hindi", "ta": "tamil", "te": "telugu", "bn": "bengali",
                  "kn": "kannada", "ml": "malayalam", "mr": "marathi", "gu": "gujarati"}
    results = []
    count = 0
    for row in ds:
        if count >= n:
            break
        # Each row contains a nested list of examples
        examples = row.get("examples", [])
        if not isinstance(examples, list):
            continue
        for r in examples:
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
    Load without field="examples" (causes UTF-8 errors). Each row has
    {canary: str, examples: {lang, source_url, summary, target_url, text}}.
    We extract from the nested examples dict.
    """
    log.info("[NEW] google/IndicGenBench_crosssum_in")
    from datasets import load_dataset
    try:
        ds = load_dataset("google/IndicGenBench_crosssum_in",
                          split="validation", streaming=True)
    except Exception as e:
        log.warning("  Could not load IndicGenBench_crosssum_in: %s", e)
        return []
    target_langs = {"hi", "ta", "te", "bn", "kn", "ml"}
    lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
                  "kn": "Kannada", "ml": "Malayalam"}
    results = []
    count = 0
    for row in ds:
        if count >= n:
            break
        # Each row has a nested examples dict (not a list)
        r = row.get("examples", {})
        if not isinstance(r, dict):
            continue
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


# ===== DOMAIN 12: LINGUISTIC DIAGNOSTICS =====

def process_blimp(n: int) -> List[Dict]:
    """nyu-mll/blimp — Benchmark of Linguistic Minimal Pairs.
    Schema: {sentence_good: str, sentence_bad: str, field: str, linguistics_term: str, ...}
    """
    log.info("[NEW] nyu-mll/blimp")
    import hashlib
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
    acceptable = [r for r in rows if r.get("label") == 1 and len(r.get("sentence", "")) > 20]
    unacceptable = [r for r in rows if r.get("label") == 0 and len(r.get("sentence", "")) > 20]
    results = []
    count = 0
    a_idx, u_idx = 0, 0
    while count < n:
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
    # Include "olympiads" — the vast majority of this dataset uses that source tag
    target_sources = {"aime", "imo", "amc", "putnam", "olympiads"}
    for r in ds:
        if count >= n:
            break
        source = (r.get("source", "") or "").lower()
        synthetic = r.get("synthetic", True)
        solution = r.get("solution", "")
        problem = r.get("problem", "")
        valid = r.get("solution_is_valid", "")
        # solution_is_valid is a string like "Yes"/"No", not "true"/"false"
        if (source in target_sources and not synthetic
                and problem and solution and len(solution) > 200
                and str(valid).lower() not in ("no", "false")):
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
    """EleutherAI/hendrycks_math — MATH benchmark Level 5.
    The original hendrycks/competition_math is no longer available on HF Hub.
    This mirror has configs: algebra, counting_and_probability, geometry,
    intermediate_algebra, number_theory, prealgebra, precalculus.
    Schema: {problem: str, solution: str, level: str ("Level 5"), type: str}
    """
    log.info("[NEW] EleutherAI/hendrycks_math")
    configs = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    all_rows = []
    for cfg in configs:
        ds = _safe_load("EleutherAI/hendrycks_math", "train", streaming=True, name=cfg)
        if ds:
            rows = _take(ds, 800)
            all_rows.extend(rows)
    if not all_rows:
        return []
    log.info("  Loaded %d total rows across %d configs", len(all_rows), len(configs))
    # level is a string like "Level 5", not an integer
    level5 = [r for r in all_rows if r.get("level") == "Level 5"]
    level5.sort(key=lambda r: len(r.get("solution", "")), reverse=True)
    results = []
    seen_subjects = set()
    count = 0
    for r in level5:
        if count >= n:
            break
        subject = r.get("type", "")
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


# ---------------------------------------------------------------------------
# Dataset Registry + Allocator
# ---------------------------------------------------------------------------

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
    ("EleutherAI/hendrycks_math",     "Math",                   "math_olympiad",           process_competition_math,   False),
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
    # Domain 10: Indic Domain Knowledge (4 datasets)
    ("ryo0634/mgsm-reformatted",      "Indic Domain Knowledge", "indic_math",              process_mgsm,               False),
    ("sarvamai/mmlu-indic",           "Indic Domain Knowledge", "indic_knowledge",         process_mmlu_indic,         False),
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
for _ds_id, _domain, *_ in DATASET_REGISTRY:
    DATASETS_PER_DOMAIN.setdefault(_domain, []).append(_ds_id)
ALL_DATASET_IDS = [d[0] for d in DATASET_REGISTRY]
NUM_DATASETS = len(ALL_DATASET_IDS)

DEFAULT_DOMAIN_WEIGHTS = {d: 1.0 / len(DOMAIN_NAMES) for d in DOMAIN_NAMES}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
                    # Filter out Chinese / CJK samples globally
                    before = len(samples)
                    samples = [s for s in samples if not _has_cjk(s.get("text", ""))]
                    if before != len(samples):
                        log.info("  => filtered %d CJK samples", before - len(samples))
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

    # Tag distribution
    tags: Dict[str, int] = {}
    for s in all_samples:
        tag = s["tag"]
        tags[tag] = tags.get(tag, 0) + 1
    log.info("")
    log.info("Tag distribution (%d unique tags):", len(tags))
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        log.info("  %-35s %d", tag, count)

    if failed:
        log.warning("Failed datasets (%d): %s", len(failed), ", ".join(failed))
    if skipped:
        log.info("Skipped datasets (%d): %s", len(skipped), ", ".join(skipped))

    if len(all_samples) == N:
        log.info("Target of %d samples reached!", N)
    else:
        log.warning("Target of %d not reached (%d collected).", N, len(all_samples))


if __name__ == "__main__":
    main()
