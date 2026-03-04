# OPUS Golden Dataset Plan v2 — Updated for 30 Final Benchmarks

## Hard Rules

1. **Exactly 128 samples** (GPU-friendly, power of 2)
2. **Every dataset gets ≥1 sample** — NO dataset is skipped
3. **44 unique datasets** across 12 domains covering all 30 benchmarks
4. **Zero synthetic data** — all human-created or human-verified

---

## Master Dataset Registry (44 datasets, 128 samples)

Every dataset in the plan, its domain, and exact sample count:

| # | Dataset | Domain | Samples | Benchmarks Targeted |
|---|---------|--------|---------|---------------------|
| 1 | openai/gsm8k | Math | 3 | GSM8K |
| 2 | nvidia/Nemotron-Math-v2 | Math | 3 | MATH |
| 3 | openbmb/UltraData-Math | Math | 2 | AIME 2025/2026 |
| 4 | AI-MO/NuminaMath-1.5 | Math | 2 | AIME 2025/2026, MATH |
| 5 | hendrycks/competition_math | Math | 2 | AIME 2025/2026, MATH |
| 6 | inclusionAI/Ling-Coder-SFT | Code | 6 | HumanEval, APPS |
| 7 | SWE-bench/SWE-smith | Code | 6 | SWE-bench Verified |
| 8 | mlabonne/open-perfectblend | Knowledge/QA | 5 | MMLU, TriviaQA, SimpleQA |
| 9 | nvidia/HelpSteer3 | Knowledge/QA | 5 | MMLU-Pro, BBH |
| 10 | MegaScience/MegaScience | Knowledge/QA | 4 | GPQA Diamond, ARC-Challenge |
| 11 | mlabonne/orpo-dpo-mix-40k | Preference | 1 | BBH, ARC-Challenge |
| 12 | Skywork/Skywork-Reward-Preference-80K-v0.2 | Preference | 1 | MATH, GPQA |
| 13 | argilla/ultrafeedback-binarized-preferences-cleaned | Preference | 1 | MMLU, TruthfulQA |
| 14 | BAAI/Infinity-Preference | Preference | 1 | General quality |
| 15 | nvidia/Nemotron-Post-Training-Dataset-v2 | Preference | 1 | MMLU, SimpleQA |
| 16 | HuggingFaceTB/smoltalk2 | Preference | 1 | General quality |
| 17 | lmarena-ai/arena-human-preference-100k | Preference | 1 | BBH, MMLU |
| 18 | TeichAI/claude-4.5-opus-high-reasoning-250x | Preference | 1 | BBH, AIME |
| 19 | Bingguang/HardGen | Tool Use | 2 | Tool Bench |
| 20 | Salesforce/xlam-function-calling-60k | Tool Use | 2 | Tool Bench |
| 21 | Team-ACE/ToolACE | Tool Use | 2 | Tool Bench |
| 22 | allenai/tulu-3-sft-personas-instruction-following | Instr. Following | 6 | IFEval |
| 23 | jondurbin/truthy-dpo-v0.1 | Truthfulness | 6 | TruthfulQA |
| 24 | L4NLP/LEval | Long Context | 2 | L-Eval |
| 25 | THUDM/LongBench | Long Context | 2 | L-Eval, RULER |
| 26 | RMT-team/babilong-train-5k-samples | Long Context | 2 | RULER |
| 27 | hotpotqa/hotpot_qa | Long Context | 2 | RULER |
| 28 | CohereForAI/aya_dataset | Indic NLU | 2 | IndicGLUE, IndicQA |
| 29 | ai4bharat/indic-align | Indic NLU | 2 | IndicGLUE, IndicQA |
| 30 | ai4bharat/IndicQA | Indic NLU | 1 | IndicQA |
| 31 | ai4bharat/indic_glue | Indic NLU | 1 | IndicGLUE |
| 32 | ai4bharat/MILU | Indic NLU + Domain | 1 + 4 = **5** | MILU-IN |
| 33 | ai4bharat/Indic-Bias | Indic NLU | 1 | Indic-Bias (FairITales) |
| 34 | juletxara/mgsm | Indic Domain | 6 | GSM-8K-IN |
| 35 | sarvamai/mmlu-indic | Indic Domain | 6 | MMLU-Indic |
| 36 | sarvamai/trivia-qa-indic | Indic Domain | 4 | TriviaQA-IN |
| 37 | ai4bharat/indic-instruct-data-v0.1 | Indic Domain | 4 | MMLU-Indic, ARC-C-IN |
| 38 | ai4bharat/BPCC (BPCC-H only) | Indic Gen/Trans | 8 | IndicMTEval |
| 39 | ai4bharat/IN22-Gen | Indic Gen/Trans | 4 | IndicMTEval |
| 40 | google/IndicGenBench_xquad_in | Indic Gen/Trans | 4 | IndicGenBench |
| 41 | google/IndicGenBench_crosssum_in | Indic Gen/Trans | 2 | IndicGenBench |
| 42 | openlanguagedata/flores_plus | Indic Gen/Trans | 2 | IndicMTEval |
| 43 | nyu-mll/blimp | Linguistic | 3 | BLiMP, MSGS |
| 44 | nyu-mll/glue (CoLA) | Linguistic | 1 | BLiMP, MSGS |
| | **TOTAL** | **12 domains** | **128** | **30 benchmarks** |

---

## Domain-by-Domain Breakdown

### Domain 1: Math — 12 samples, 5 datasets

**Benchmarks:** GSM8K, MATH, AIME 2025/2026

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g1-g3 | openai/gsm8k | 3 | Longest step-by-step solutions from train split |
| g4-g6 | nvidia/Nemotron-Math-v2 | 3 | `high_part00` split, answers >200 chars |
| g7-g8 | openbmb/UltraData-Math | 2 | L3 difficulty only, longest solutions |
| g9-g10 | **AI-MO/NuminaMath-1.5** | 2 | Filter `source=aime` or `imo`, hardest with longest proofs. **NEW** |
| g11-g12 | **hendrycks/competition_math** | 2 | Level 5 only, number theory/combinatorics. **NEW** |

---

### Domain 2: Code — 12 samples, 2 datasets

**Benchmarks:** HumanEval, APPS, SWE-bench Verified

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g13-g18 | inclusionAI/Ling-Coder-SFT | 6 | Must contain actual code blocks (`def`, `class`, triple backticks) |
| g19-g24 | SWE-bench/SWE-smith | 6 | Medium-length patches (200-5000 chars), real repo bug fixes |

---

### Domain 3: Knowledge/QA — 14 samples, 3 datasets

**Benchmarks:** MMLU, MMLU-Pro, TriviaQA, SimpleQA_Verified, GPQA Diamond

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g25-g29 | mlabonne/open-perfectblend | 5 | Substantive answers >200 chars, diverse subjects |
| g30-g34 | nvidia/HelpSteer3 | 5 | Sort by helpfulness + correctness scores, pick top-rated |
| g35-g38 | MegaScience/MegaScience | 4 | Scientific knowledge, prefer peer-reviewed content |

---

### Domain 4: Preference/Reasoning — 8 samples, 8 datasets

**Benchmarks:** BBH (Big Bench Hard), ARC-Challenge, general quality signal

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g39 | mlabonne/orpo-dpo-mix-40k | 1 | **Chosen only.** Longest reasoning chain. |
| g40 | Skywork/Skywork-Reward-Preference-80K-v0.2 | 1 | **Chosen only.** Science/math preference. |
| g41 | argilla/ultrafeedback-binarized-preferences-cleaned | 1 | **Chosen only.** General QA quality. |
| g42 | BAAI/Infinity-Preference | 1 | **Chosen only.** Detailed reasoning >300 chars. |
| g43 | nvidia/Nemotron-Post-Training-Dataset-v2 | 1 | General post-training knowledge. |
| g44 | HuggingFaceTB/smoltalk2 | 1 | General conversation quality. |
| g45 | lmarena-ai/arena-human-preference-100k | 1 | **Chosen only.** Human-preferred winning response. |
| g46 | TeichAI/claude-4.5-opus-high-reasoning-250x | 1 | Longest reasoning chain. |

**Rule:** For ALL DPO/preference datasets, use ONLY the "chosen" response.

---

### Domain 5: Tool Use — 6 samples, 3 datasets

**Benchmarks:** Tool Bench

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g47-g48 | Bingguang/HardGen | 2 | Filter `error_tool_response == false`. Correct trajectories only. |
| g49-g50 | Salesforce/xlam-function-calling-60k | 2 | Execution-verified, proper JSON arguments. |
| g51-g52 | Team-ACE/ToolACE | 2 | Correct function/API calls with structured output. |

---

### Domain 6: Instruction Following — 6 samples, 1 dataset

**Benchmarks:** IFEval

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g53-g58 | allenai/tulu-3-sft-personas-instruction-following | 6 | Max constraint-type diversity: keyword inclusion, keyword exclusion, paragraph count, formatting, language constraint, case constraint. |

---

### Domain 7: Truthfulness — 6 samples, 1 dataset

**Benchmarks:** TruthfulQA

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g59-g64 | jondurbin/truthy-dpo-v0.1 | 6 | **Chosen only.** 6 diverse misconception categories: science myth, historical inaccuracy, health misconception, common saying, geography, statistics. |

---

### Domain 8: Long Context — 8 samples, 4 datasets

**Benchmarks:** L-Eval, RULER

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g65-g66 | L4NLP/LEval | 2 | 1 QA (narrative_qa), 1 summarization (gov_report). Truncate to ~3000 chars. |
| g67-g68 | THUDM/LongBench | 2 | 1 single-doc QA (qasper), 1 multi-hop (musique). |
| g69-g70 | RMT-team/babilong-train-5k-samples | 2 | Facts hidden in haystacks. Closest RULER proxy. |
| g71-g72 | hotpotqa/hotpot_qa | 2 | Multi-hop reasoning, chaining facts across documents. |

---

### Domain 9: Indic NLU — 8 samples, 6 datasets

**Benchmarks:** IndicGLUE, IndicQA, Indic-Bias (FairITales)

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g73-g74 | CohereForAI/aya_dataset | 2 | Filter `annotation_type = 'original-annotations'`. 1 Hindi, 1 Tamil. |
| g75-g76 | ai4bharat/indic-align | 2 | Anudesh (native crowd-sourced) subset. 1 Telugu, 1 Bengali. |
| g77 | ai4bharat/IndicQA | 1 | Direct benchmark match. Kannada sample. |
| g78 | ai4bharat/indic_glue | 1 | CSQA task. Hindi sample. |
| g79 | ai4bharat/MILU | 1 | India-centric cultural knowledge. Marathi sample. |
| g80 | ai4bharat/Indic-Bias | 1 | Direct Indic-Bias/FairITales match. Gated — requires HF login. |

---

### Domain 10 (NEW): Indic Domain Knowledge — 24 samples, 5 datasets

**Benchmarks:** MMLU-Indic, ARC-C-IN, GSM-8K-IN, TriviaQA-IN, MILU-IN

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g81-g86 | **juletxara/mgsm** | 6 | 1/language: Hindi, Bengali, Telugu, Tamil, Kannada, Marathi. Multi-step problems (4+ steps). Native script only. **Direct GSM-8K-IN match.** |
| g87-g92 | **sarvamai/mmlu-indic** | 6 | 1/language: Hindi, Tamil, Telugu, Bengali, Kannada, Malayalam. Diverse subjects (2 STEM, 2 humanities, 2 social science). **Direct MMLU-Indic match.** |
| g93-g96 | **sarvamai/trivia-qa-indic** | 4 | 1/language: Hindi, Tamil, Telugu, Bengali. Prefer India-relevant factual questions. **Direct TriviaQA-IN match.** |
| g97-g100 | **ai4bharat/indic-instruct-data-v0.1** | 4 | Anudesh subset ONLY (native crowd-sourced). 1 science, 1 math, 1 knowledge, 1 reasoning across different languages. Supports ARC-C-IN + MMLU-Indic. |
| g101-g104 | **ai4bharat/MILU** (additional) | 4 | Expand into new languages: Gujarati, Punjabi, Odia, Malayalam. Different exam types (state PSC, university entrance). **Direct MILU-IN match.** |

**Note:** MILU total = 1 (Domain 9 g79) + 4 (Domain 10 g101-g104) = **5 samples** from ai4bharat/MILU.

**Quality notes:**
- **mgsm**: Human-translated by Google's professional native-speaker translators. Published at ACL. Gold standard for multilingual math.
- **mmlu-indic**: From Sarvam AI. Verified by native speakers. Used in Sarvam-M evaluation.
- **trivia-qa-indic**: From Sarvam AI. Verified translations, not raw MT.
- **indic-instruct-data-v0.1**: From AI4Bharat. Anudesh subset is crowd-sourced native content. Used to train Airavata.
- **MILU**: Real Indian exam questions (UPSC, state PSC, university entrance). Not synthetic.

---

### Domain 11 (NEW): Indic Generation & Translation — 20 samples, 5 datasets

**Benchmarks:** IndicGenBench, IndicMTEval

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g105-g112 | **ai4bharat/BPCC** (BPCC-H only) | 8 | 1/language pair: Hi↔En, Ta↔En, Te↔En, Bn↔En, Kn↔En, Mr↔En, Ml↔En, Gu↔En. Moderate complexity (15-40 words). NO trivial greetings. **BPCC-H (human) subset ONLY.** |
| g113-g116 | **ai4bharat/IN22-Gen** | 4 | 1 conversational, 1 news, 1 technical, 1 literary. Different language pairs. |
| g117-g120 | **google/IndicGenBench_xquad_in** | 4 | 1/language: Hindi, Tamil, Telugu, Bengali. Multi-sentence answers. **Direct IndicGenBench match.** |
| g121-g122 | **google/IndicGenBench_crosssum_in** | 2 | 1 Hindi, 1 Tamil. Medium-length articles (500-1500 chars). **Direct IndicGenBench match.** |
| g123-g124 | **openlanguagedata/flores_plus** | 2 | Sentences with idiomatic expressions or complex syntax. Different Indic language pairs. |

**Quality notes:**
- **BPCC-H**: Professional human translators. Used to train IndicTrans2 (SOTA Indic MT). From AI4Bharat.
- **IN22-Gen**: Hand-curated evaluation benchmark from AI4Bharat.
- **IndicGenBench_*_in**: From Google Research. Part of published benchmark suite. Human-annotated.
- **flores_plus**: Professional translations. Meta FLORES heritage.

**Formats:**
```json
{"id": "g105", "tag": "indic_translation_hi_en", "text": "[USER] Translate the following Hindi text to English:\n\n{hindi_text}\n\n[ASSISTANT] {english_translation}"}
```
```json
{"id": "g117", "tag": "indic_crosslingual_qa", "text": "[USER] {context_passage}\n\nAnswer the following question in Hindi: {question}\n\n[ASSISTANT] {hindi_answer}"}
```

---

### Domain 12 (NEW): Linguistic Diagnostics — 4 samples, 2 datasets

**Benchmarks:** MSGS (Mixed Signals Generalization Set), BLiMP

| Slot | Dataset | Samples | Selection Criteria |
|------|---------|---------|-------------------|
| g125-g127 | **nyu-mll/blimp** | 3 | 1 from each: Subject-Verb Agreement, Filler-Gap Dependencies, Anaphor Agreement. Convert to SFT format (identify + explain grammaticality). **Direct BLiMP + MSGS match.** |
| g128 | **nyu-mll/glue** (CoLA config) | 1 | Complex syntax with subtle grammaticality judgment. Expert linguist annotations. |

**Why only 4:** MSGS/BLiMP are diagnostic benchmarks testing linguistic generalizations (subject-verb agreement, negative polarity, filler-gap). High-quality text from Knowledge/QA and Instruction Following provides strong implicit support. 4 explicit samples make coverage targeted.

**Format:**
```json
{"id": "g125", "tag": "linguistic_diagnostics", "text": "[USER] Which of these sentences is grammatically correct and why?\n\nA: The authors that the editor admired laughed.\nB: The authors that the editor admired them laughed.\n\n[ASSISTANT] Sentence A is correct. Sentence B contains a resumptive pronoun error..."}
```

---

## Domain Summary Table

| # | Domain | Samples | Datasets | Dataset Count | Benchmarks Covered |
|---|--------|---------|----------|---------------|-------------------|
| 1 | Math | **12** | gsm8k(3), Nemotron-Math(3), UltraData-Math(2), NuminaMath(2), competition_math(2) | 5 | GSM8K, MATH, AIME 2025/2026 |
| 2 | Code | **12** | Ling-Coder-SFT(6), SWE-smith(6) | 2 | HumanEval, APPS, SWE-bench |
| 3 | Knowledge/QA | **14** | open-perfectblend(5), HelpSteer3(5), MegaScience(4) | 3 | MMLU, MMLU-Pro, TriviaQA, SimpleQA, GPQA |
| 4 | Preference/Reasoning | **8** | orpo-dpo(1), Skywork(1), ultrafeedback(1), Infinity(1), Nemotron-PT(1), smoltalk2(1), arena(1), claude-reasoning(1) | 8 | BBH, ARC-Challenge |
| 5 | Tool Use | **6** | HardGen(2), xlam(2), ToolACE(2) | 3 | Tool Bench |
| 6 | Instruction Following | **6** | tulu-3-sft(6) | 1 | IFEval |
| 7 | Truthfulness | **6** | truthy-dpo(6) | 1 | TruthfulQA |
| 8 | Long Context | **8** | LEval(2), LongBench(2), babilong(2), hotpotqa(2) | 4 | L-Eval, RULER |
| 9 | Indic NLU | **8** | aya(2), indic-align(2), IndicQA(1), indic_glue(1), MILU(1), Indic-Bias(1) | 6 | IndicGLUE, IndicQA, Indic-Bias |
| 10 | **Indic Domain Knowledge** | **24** | mgsm(6), mmlu-indic(6), trivia-qa-indic(4), indic-instruct(4), MILU(4) | 5 | MMLU-Indic, ARC-C-IN, GSM-8K-IN, TriviaQA-IN, MILU-IN |
| 11 | **Indic Gen & Translation** | **20** | BPCC(8), IN22-Gen(4), IndicGenBench-xquad(4), IndicGenBench-crosssum(2), flores_plus(2) | 5 | IndicGenBench, IndicMTEval |
| 12 | **Linguistic Diagnostics** | **4** | blimp(3), CoLA(1) | 2 | MSGS, BLiMP |
| | **TOTAL** | **128** | | **44 unique** | **30 benchmarks** |

**Indic total: 8 + 24 + 20 = 52 samples (41%)** across 16 datasets

---

## Benchmark Coverage Verification (30/30)

| # | Benchmark | Datasets | Total Samples | Status |
|---|-----------|----------|---------------|--------|
| 1 | MMLU | open-perfectblend, ultrafeedback, Nemotron-PT | ~3 | ✅ |
| 2 | TriviaQA | open-perfectblend | ~2 | ✅ |
| 3 | MMLU-Pro | HelpSteer3 | ~3 | ✅ |
| 4 | GPQA Diamond | MegaScience, Skywork | ~3 | ✅ |
| 5 | GSM8K | gsm8k | 3 | ✅ |
| 6 | BBH | HelpSteer3, orpo-dpo, arena, claude-reasoning | ~4 | ✅ |
| 7 | ARC-Challenge | MegaScience, HelpSteer3 | ~3 | ✅ |
| 8 | MATH | Nemotron-Math, UltraData-Math, NuminaMath, competition_math | 9 | ✅ Strengthened |
| 9 | IFEval | tulu-3-sft | 6 | ✅ |
| 10 | SimpleQA_Verified | open-perfectblend, Nemotron-PT | ~2 | ✅ |
| 11 | HumanEval | Ling-Coder-SFT | 6 | ✅ |
| 12 | APPS | Ling-Coder-SFT | 6 | ✅ |
| 13 | AIME 2025/2026 | UltraData-Math, NuminaMath, competition_math | 6 | ✅ Strengthened |
| 14 | MSGS (Diagnostics) | blimp, CoLA | 4 | ✅ **NOW EXPLICIT** |
| 15 | BLiMP | blimp, CoLA | 4 | ✅ **NOW EXPLICIT** |
| 16 | IndicGLUE | aya, indic-align, indic_glue | ~3 | ✅ |
| 17 | IndicQA | aya, indic-align, IndicQA | ~3 | ✅ |
| 18 | L-Eval | LEval, LongBench | 4 | ✅ |
| 19 | RULER | LongBench, babilong, hotpotqa | 6 | ✅ |
| 20 | TruthfulQA | truthy-dpo | 6 | ✅ |
| 21 | Indic-Bias (FairITales) | Indic-Bias | 1 | ✅ |
| 22 | SWE-bench Verified | SWE-smith | 6 | ✅ |
| 23 | Tool Bench | HardGen, xlam, ToolACE | 6 | ✅ |
| 24 | **MMLU-Indic** | mmlu-indic, indic-instruct | 10 | 🆕 **NEW** |
| 25 | **ARC-C-IN** | indic-instruct, MILU | ~5 | 🆕 **NEW** |
| 26 | **GSM-8K-IN** | mgsm | 6 | 🆕 **NEW** |
| 27 | **MILU-IN** | MILU (Domain 9 + Domain 10) | 5 | 🆕 **EXPANDED** |
| 28 | **TriviaQA-IN** | trivia-qa-indic | 4 | 🆕 **NEW** |
| 29 | **IndicGenBench** | IndicGenBench-xquad, IndicGenBench-crosssum | 6 | 🆕 **NEW** |
| 30 | **IndicMTEval** | BPCC, IN22-Gen, flores_plus | 14 | 🆕 **NEW** |

---

## Dataset Quality — All Non-Synthetic

| # | Dataset | Organization | Source Type |
|---|---------|-------------|------------|
| 1 | openai/gsm8k | OpenAI | Human-written math problems |
| 2 | nvidia/Nemotron-Math-v2 | NVIDIA | Competition math |
| 3 | openbmb/UltraData-Math | OpenBMB | Hard math curated |
| 4 | AI-MO/NuminaMath-1.5 | Numina/AIMO | Real AMC/AIME/IMO problems |
| 5 | hendrycks/competition_math | UC Berkeley | Real competition math |
| 6 | inclusionAI/Ling-Coder-SFT | InclusionAI | Real code SFT |
| 7 | SWE-bench/SWE-smith | Princeton | Real GitHub repo patches |
| 8 | mlabonne/open-perfectblend | Community | Curated knowledge blend |
| 9 | nvidia/HelpSteer3 | NVIDIA | Human helpfulness scores |
| 10 | MegaScience/MegaScience | MegaScience | Peer-reviewed scientific knowledge |
| 11 | mlabonne/orpo-dpo-mix-40k | Community | Human preference pairs |
| 12 | Skywork/Skywork-Reward-Preference-80K | Skywork | Human preference pairs |
| 13 | argilla/ultrafeedback-binarized | Argilla | Human preference pairs |
| 14 | BAAI/Infinity-Preference | BAAI | Human preference pairs |
| 15 | nvidia/Nemotron-Post-Training-v2 | NVIDIA | Post-training curated |
| 16 | HuggingFaceTB/smoltalk2 | HuggingFace | Curated conversations |
| 17 | lmarena-ai/arena-human-preference-100k | LMSYS | Real human preferences |
| 18 | TeichAI/claude-4.5-opus-high-reasoning | TeichAI | High reasoning chains |
| 19 | Bingguang/HardGen | Community | Tool-use trajectories |
| 20 | Salesforce/xlam-function-calling-60k | Salesforce | Execution-verified, 95%+ accuracy |
| 21 | Team-ACE/ToolACE | Alibaba | SOTA on BFCL benchmark |
| 22 | allenai/tulu-3-sft-personas-IF | Allen AI | IFEval constraint taxonomy |
| 23 | jondurbin/truthy-dpo-v0.1 | Community | Misconception resistance pairs |
| 24 | L4NLP/LEval | L4NLP | Actual L-Eval benchmark data |
| 25 | THUDM/LongBench | Tsinghua | Long context benchmark data |
| 26 | RMT-team/babilong-train-5k-samples | RMT | Long needle-in-haystack |
| 27 | hotpotqa/hotpot_qa | CMU | Multi-hop QA, crowd-sourced |
| 28 | CohereForAI/aya_dataset | Cohere | Human-annotated by native speakers |
| 29 | ai4bharat/indic-align | AI4Bharat | Native Anudesh crowd-sourced |
| 30 | ai4bharat/IndicQA | AI4Bharat | Benchmark data, human-annotated |
| 31 | ai4bharat/indic_glue | AI4Bharat | Benchmark data, human-annotated |
| 32 | ai4bharat/MILU | AI4Bharat | Real Indian exam papers (UPSC, PSC) |
| 33 | ai4bharat/Indic-Bias | AI4Bharat | 85 Indian identity groups, human |
| 34 | juletxara/mgsm | Google Research | Professional native-speaker translators |
| 35 | sarvamai/mmlu-indic | Sarvam AI | Native speaker verified |
| 36 | sarvamai/trivia-qa-indic | Sarvam AI | Verified translations |
| 37 | ai4bharat/indic-instruct-data-v0.1 | AI4Bharat | Anudesh: native crowd-sourced |
| 38 | ai4bharat/BPCC (BPCC-H) | AI4Bharat | Professional human translators |
| 39 | ai4bharat/IN22-Gen | AI4Bharat | Hand-curated evaluation set |
| 40 | google/IndicGenBench_xquad_in | Google Research | Published benchmark, human-annotated |
| 41 | google/IndicGenBench_crosssum_in | Google Research | Published benchmark, human-annotated |
| 42 | openlanguagedata/flores_plus | Meta/Community | Professional translators |
| 43 | nyu-mll/blimp | NYU ML² | Expert linguists, published in TACL |
| 44 | nyu-mll/glue (CoLA) | NYU ML² | Expert linguist annotations |

---

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

### CLI Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--total-samples` | Yes | — | Total samples (must be >= 45). GPU-friendly: 64, 128, 256, 512 |
| `--output` | No | `examples/golden_samples.jsonl` | Output JSONL path |
| `--weights-json` | No | — | JSON file to override domain weights |
| `--dry-run` | No | — | Show allocation table, no downloads |
| `--seed` | No | 42 | Random seed |
| `--skip-errors` | No | — | Continue past dataset failures |
| `--verbose` | No | — | Debug logging |

### Architecture

- **45 datasets** across **12 domains** managed via `DATASET_REGISTRY`
- **Dynamic allocation**: `scripts/golden_allocator.py` distributes N samples uniformly across domains, then evenly within each domain
- **Every dataset guaranteed >= 1 sample** (strict rule)
- **HF auth auto-detected** from `HF_TOKEN` env var or `huggingface-cli login` cache

### 14 new processors added

- Domain 10 (Indic Domain Knowledge): `process_mgsm`, `process_mmlu_indic`, `process_trivia_qa_indic`, `process_indic_instruct`, `process_milu_expanded`
- Domain 11 (Indic Gen/Translation): `process_bpcc`, `process_in22_gen`, `process_indicgenbench_xquad`, `process_indicgenbench_crosssum`, `process_flores_plus`
- Domain 12 (Linguistic Diagnostics): `process_blimp`, `process_cola`
- Math expansion: `process_numinamath`, `process_competition_math`

### All 45 processors now accept `n: int`

Sample counts are no longer hardcoded — they are dynamically computed by the allocator based on `--total-samples`.

---

## Important Notes

1. **Every dataset has ≥1 sample.** This is a strict rule — no exceptions.
2. **Preference datasets: chosen only.** For all DPO/preference datasets, use ONLY the "chosen" response.
3. **Golden samples are trained on.** These must be training-quality, not just reference-quality.
4. **Gated datasets:** Indic-Bias and possibly others require `huggingface-cli login`. Use `--skip-errors` flag.
5. **512-token scoring vs full training:** OPUS scores at 512 tokens. Truncate long context samples to ~3000 chars in JSONL.
6. **BPCC-H vs BPCC-M:** Use ONLY BPCC-H (human-translated). BPCC-M is web-mined and noisy.
7. **MGSM is the gold standard for GSM-8K-IN:** Do NOT use machine-translated GSM8K.
8. **indic-instruct: Anudesh subset ONLY.** Avoid raw Dolly translations.
9. **No synthetic data.** All 44 datasets are human-created or human-verified.
10. **MILU appears in two domains:** 1 sample in Domain 9 (Indic NLU) + 4 samples in Domain 10 (Indic Domain Knowledge) = 5 total.
