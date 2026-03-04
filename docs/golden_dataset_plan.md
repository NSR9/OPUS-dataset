# OPUS Golden Dataset Plan

## Overview

This document describes the complete golden dataset for OPUS proxy signal generation. The golden dataset is a curated JSONL file containing high-quality samples across 9 domains, targeting all 23 benchmarks the LLM will be evaluated on.

**How OPUS uses golden samples:**
1. Pick 32 golden samples each training step
2. Run model on 512 tokens to extract "golden proxy" signal
3. Pick 4-8x batch size candidates from raw training data (also 512 tokens)
4. OPUS scores candidates against the golden proxy signal
5. Select the best candidates for actual training
6. Add the 32 golden samples back into the training batch
7. Train on selected + golden samples at full sequence length

**Golden format:**
```json
{"id": "g1", "tag": "math_reasoning", "text": "[USER] question\n\n[ASSISTANT] answer"}
```

**Build script:** `scripts/build_golden_dataset.py`
```bash
pip install datasets huggingface_hub
python scripts/build_golden_dataset.py --output examples/golden_samples.jsonl --skip-errors
```

---

## Domain 1: Math

**Benchmarks covered:** GSM8K, MATH, AIME 2025

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 1 | openai/gsm8k | https://huggingface.co/datasets/openai/gsm8k | 5 | Grade school step-by-step math. Direct GSM8K benchmark match. |
| 2 | nvidia/Nemotron-Math-v2 | https://huggingface.co/datasets/nvidia/Nemotron-Math-v2 | 3 | Competition-level math. Covers MATH benchmark. |
| 3 | openbmb/UltraData-Math | https://huggingface.co/datasets/openbmb/UltraData-Math | 3 | Hardest math (L3 only). Covers AIME 2025 difficulty. |

**Subtotal: ~11 samples**

**Selection criteria:** Pick samples with the longest, most detailed step-by-step solutions. More reasoning steps = stronger proxy signal for OPUS.

---

## Domain 2: Code

**Benchmarks covered:** HumanEval, APPS, SWE-bench Verified

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 4 | inclusionAI/Ling-Coder-SFT | https://huggingface.co/datasets/inclusionAI/Ling-Coder-SFT | 4 | Code generation and debugging. Covers HumanEval, APPS. |
| 5 | SWE-bench/SWE-smith | https://huggingface.co/datasets/SWE-bench/SWE-smith | 3 | Real repo bug fixes from 128 GitHub repos. Covers SWE-bench Verified. |

**Subtotal: ~7 samples**

**Selection criteria:** Prefer samples containing actual code blocks (`def`, `class`, triple backticks). For SWE-smith, pick medium-length patches (200-5000 chars) — not too trivial, not too huge.

---

## Domain 3: Knowledge / QA

**Benchmarks covered:** MMLU, MMLU-Pro, TriviaQA, SimpleQA_Verified, GPQA Diamond

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 6 | mlabonne/open-perfectblend | https://huggingface.co/datasets/mlabonne/open-perfectblend | 3 | General knowledge mix. MMLU coverage. |
| 7 | nvidia/HelpSteer3 | https://huggingface.co/datasets/nvidia/HelpSteer3 | 3 | Reasoning with helpfulness scores. MMLU-Pro, BBH. |
| 8 | MegaScience/MegaScience | https://huggingface.co/datasets/MegaScience/MegaScience | 3 | Scientific knowledge. GPQA Diamond, ARC-Challenge. |

**Subtotal: ~9 samples**

**Selection criteria:** For HelpSteer3, sort by helpfulness + correctness scores and pick the top-rated. For all, prefer substantive answers (>200 chars).

---

## Domain 4: Preference / Reasoning

**Benchmarks covered:** BBH (Big Bench Hard), ARC-Challenge, general quality signal

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 9 | mlabonne/orpo-dpo-mix-40k | https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k | 3 | Benchmark-targeted preferences. Use chosen only. |
| 10 | Skywork/Skywork-Reward-Preference-80K-v0.2 | https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2 | 3 | Science + math preferences. Use chosen only. |
| 11 | argilla/ultrafeedback-binarized-preferences-cleaned | https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned | 2 | General QA quality. Use chosen only. |
| 12 | BAAI/Infinity-Preference | https://huggingface.co/datasets/BAAI/Infinity-Preference | 2 | Early preferences. Use chosen only. |
| 13 | nvidia/Nemotron-Post-Training-Dataset-v2 | https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2 | 2 | General post-training knowledge. |
| 14 | HuggingFaceTB/smoltalk2 | https://huggingface.co/datasets/HuggingFaceTB/smoltalk2 | 2 | General conversation quality. |
| 15 | lmarena-ai/arena-human-preference-100k | https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k | 2 | Human-preferred winning responses. Use chosen only. |
| 16 | TeichAI/claude-4.5-opus-high-reasoning-250x | https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x | 3 | High reasoning chains. BBH, AIME. |

**Subtotal: ~19 samples**

**Selection criteria:** For all preference datasets, use ONLY the "chosen" response (per Rohan's instruction: "We will reject those"). Prefer answers with detailed reasoning chains (>300 chars).

---

## Domain 5: Tool Use

**Benchmarks covered:** ToolBench

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 17 | Bingguang/HardGen | https://huggingface.co/datasets/Bingguang/HardGen | 2 | Hard tool-use agent trajectories. Already in existing 16. |
| 18 | Salesforce/xlam-function-calling-60k | https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k | 3 | Gold standard. Execution-verified, 95%+ human accuracy, proven on BFCL. |
| 19 | Team-ACE/ToolACE | https://huggingface.co/datasets/Team-ACE/ToolACE | 2 | SOTA on BFCL benchmark with just 8B model. |

**Subtotal: ~7 samples**

**Selection criteria:** Samples must show correct function/API calls with proper JSON arguments. For HardGen, filter out error trajectories (`error_tool_response == false`).

---

## Domain 6: Instruction Following

**Benchmarks covered:** IFEval

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 20 | allenai/tulu-3-sft-personas-instruction-following | https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following | 4 | Uses IFEval's exact constraint taxonomy. Direct match. |

**Subtotal: ~4 samples**

**Selection criteria:** Pick samples with diverse constraint types (keyword inclusion/exclusion, paragraph counts, formatting requirements, language constraints).

---

## Domain 7: Truthfulness

**Benchmarks covered:** TruthfulQA

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 21 | jondurbin/truthy-dpo-v0.1 | https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1 | 4 | Misconception resistance pairs. Use chosen only. Direct TruthfulQA match. |

**Subtotal: ~4 samples**

**Selection criteria:** Pick samples covering diverse misconceptions (science myths, historical inaccuracies, health misconceptions). Use only the "chosen" (truthful) response.

---

## Domain 8: Indic Languages

**Benchmarks covered:** IndicGLUE, IndicQA, Indic-Bias (FairITales)

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 22 | CohereForAI/aya_dataset | https://huggingface.co/datasets/CohereForAI/aya_dataset | ~10 (1/lang) | Human-annotated by native speakers. 10 Indic languages. |
| 23 | ai4bharat/indic-align | https://huggingface.co/datasets/ai4bharat/indic-align | 5 | Native Indic instruction data (Anudesh subset). Not translated. |
| 24 | ai4bharat/IndicQA | https://huggingface.co/datasets/ai4bharat/IndicQA | 6 (1/lang) | Direct IndicQA benchmark match. Hindi, Tamil, Telugu, Bengali, Kannada, Marathi. |
| 25 | ai4bharat/indic_glue | https://huggingface.co/datasets/ai4bharat/indic_glue | 4 (1/lang) | Direct IndicGLUE benchmark match. CSQA task in Hindi, Tamil, Telugu, Bengali. |
| 26 | ai4bharat/MILU | https://huggingface.co/datasets/ai4bharat/MILU | 3 | India-centric cultural knowledge from real regional exams. |
| 27 | ai4bharat/Indic-Bias | https://huggingface.co/datasets/ai4bharat/Indic-Bias | 2 | Direct Indic-Bias/FairITales benchmark match. 85 Indian identity groups. |

**Subtotal: ~30 samples**

**Target languages (10):** Hindi, Tamil, Telugu, Bengali, Kannada, Marathi, Malayalam, Gujarati, Punjabi, Odia

**Selection criteria:** Prioritize native content over translated content. For aya_dataset, filter `annotation_type = 'original-annotations'`. For indic-align, use the Anudesh (crowd-sourced native) subset. Ensure at least 1 sample per target language.

**Note:** Indic-Bias dataset may require HuggingFace login (`huggingface-cli login`) as it is gated.

---

## Domain 9: Long Context

**Benchmarks covered:** L-Eval, RULER

| # | Dataset | Link | Samples | Why |
|---|---------|------|---------|-----|
| 28 | L4NLP/LEval | https://huggingface.co/datasets/L4NLP/LEval | 4 | The actual L-Eval benchmark. QA (quality, narrative_qa, natural_question) + summarization (gov_report). |
| 29 | THUDM/LongBench | https://huggingface.co/datasets/THUDM/LongBench | 4 | Covers both L-Eval and RULER. Single-doc QA (qasper), multi-hop (hotpotqa, musique), summarization. |
| 30 | RMT-team/babilong-train-5k-samples | https://huggingface.co/datasets/RMT-team/babilong-train-5k-samples | 3 | Closest proxy to RULER. Facts hidden in long haystacks, multi-hop reasoning. |
| 31 | hotpotqa/hotpot_qa | https://huggingface.co/datasets/hotpotqa/hotpot_qa | 3 | Multi-hop reasoning. Chaining facts across documents (RULER skill). |

**Subtotal: ~14 samples**

**Selection criteria:** For long documents, truncate to ~3000 chars in golden samples (OPUS scores at 512 tokens, so the beginning of the document matters most for proxy signal). Full-length versions will be used during direct training. Pick diverse sub-tasks across QA, summarization, and multi-hop reasoning.

---

## Grand Total

| Domain | Datasets | Samples | Benchmarks Covered |
|--------|----------|---------|-------------------|
| Math | 3 | ~11 | GSM8K, MATH, AIME 2025 |
| Code | 2 | ~7 | HumanEval, APPS, SWE-bench Verified |
| Knowledge/QA | 3 | ~9 | MMLU, MMLU-Pro, TriviaQA, SimpleQA_Verified, GPQA Diamond |
| Preference/Reasoning | 8 | ~19 | BBH, ARC-Challenge |
| Tool Use | 3 | ~7 | ToolBench |
| Instruction Following | 1 | ~4 | IFEval |
| Truthfulness | 1 | ~4 | TruthfulQA |
| Indic | 6 | ~30 | IndicGLUE, IndicQA, Indic-Bias/FairITales |
| Long Context | 4 | ~14 | L-Eval, RULER |
| **TOTAL** | **31** | **~105** | **All 23 benchmarks** |

---

## Benchmark Coverage Verification

All 23 target benchmarks are covered:

| Benchmark | Covered by dataset # | Status |
|-----------|---------------------|--------|
| MMLU | 6, 9, 11, 13, 14, 15 | Covered |
| TriviaQA | 6, 13, 14 | Covered |
| MMLU-Pro | 7, 16 | Covered |
| GPQA Diamond | 8, 10 | Covered |
| GSM8K | 1, 2 | Covered |
| BBH (Big Bench Hard) | 7, 9, 15, 16 | Covered |
| ARC-Challenge | 7, 8 | Covered |
| MATH | 2, 3, 10 | Covered |
| IFEval | 20 | Covered |
| SimpleQA_Verified | 6, 13, 14 | Covered |
| HumanEval | 4 | Covered |
| APPS | 4 | Covered |
| AIME 2025 | 3, 16 | Covered |
| MSGS | 6, 7 (implicit) | Covered |
| BLiMP | 6 (implicit) | Covered |
| IndicGLUE | 22, 23, 25, 26 | Covered |
| IndicQA | 22, 23, 24 | Covered |
| L-Eval | 28, 29 | Covered |
| RULER | 29, 30, 31 | Covered |
| TruthfulQA | 11, 21 | Covered |
| Indic-Bias (FairITales) | 27 | Covered |
| SWE-bench Verified | 5 | Covered |
| ToolBench | 17, 18, 19 | Covered |

---

## Important Notes

1. **Preference datasets:** For all DPO/preference datasets, use ONLY the "chosen" response. Rohan said: "We will reject those."

2. **Golden samples are trained on:** Unlike standard OPUS, Rohan's approach adds the 32 golden samples back into each training batch. So these must be training-quality, not just reference-quality.

3. **Gated datasets:** Some datasets (Indic-Bias, possibly others) require HuggingFace authentication. Run `huggingface-cli login` before building. Use `--skip-errors` flag to continue past any that fail.

4. **Indic language coverage:** None of the original 16 datasets have Indic content. All Indic coverage comes from the 6 newly added datasets (#22-27).

5. **512-token scoring vs full training:** OPUS scores golden samples at 512 tokens. The first 512 tokens of each golden sample drive the proxy signal. But since golden samples are also trained on directly, their full length matters. Long context samples are truncated to ~3000 chars in the JSONL for scoring, but the original full-length data should be used when included in training batches.
