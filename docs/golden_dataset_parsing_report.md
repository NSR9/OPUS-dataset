# OPUS Golden Dataset Builder — Parsing Logic Report

**Date:** 2026-03-02
**File:** `scripts/build_golden_dataset.py`
**Output:** `examples/golden_samples_256.jsonl` (256 samples)

---

## 1. Overview

The builder pulls samples from **44 HuggingFace datasets** across **12 domains**, parses each dataset's unique schema, and converts everything into a canonical format:

```json
{"id": "g1", "tag": "math_reasoning", "text": "<|user|> question...\n\n<|assistant|> answer..."}
```

- **Allocation:** Samples are distributed uniformly across 12 domains, then evenly within each domain's datasets.
- **Auth:** 8 datasets are gated (require HF token). Token is auto-detected from env var or `huggingface-cli login`.
- **Streaming:** All datasets are loaded in streaming mode (no full download).

---

## 2. Formatting Helpers

| Function | Purpose |
|----------|---------|
| `_fmt(user, assistant)` | Wraps text as `<|user|> ...\n\n<|assistant|> ...` for SFT loss masking |
| `_fmt_fim(prefix, suffix, middle)` | FIM format: `<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>...` (available, not yet used) |
| `_clean(s)` | Strips control chars except `\n`/`\t` |
| `_safe_load(...)` | Wraps `load_dataset()` with gated auth + error handling, returns `None` on failure |

---

## 3. All 44 Processors — Parsing Logic

### Domain 1: Math (5 datasets, budget=22)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 1 | `openai/gsm8k` | `process_gsm8k` | `{question, answer}` | Streams 500 rows, sorts by answer length desc (best step-by-step), takes top n | `math_reasoning` |
| 2 | `nvidia/Nemotron-Math-v2` | `process_nemotron_math` | `{problem, messages: [{role,content}], expected_answer}` | Extracts last assistant msg; filters `len(answer) > 200` | `math_competition` |
| 3 | `openbmb/UltraData-Math` | `process_ultradata_math` | `{uid, content}` (single combined field) | Heuristic split on markers (`\nSolution:`, `\nAnswer:`, `\nStep 1:`); filters `len >= 300` | `math_hard` |
| 4 | `AI-MO/NuminaMath-1.5` | `process_numinamath` | `{problem, solution, answer, source, synthetic, solution_is_valid}` | Filters: source in {aime,imo,amc,putnam,olympiads}, NOT synthetic, `len(solution)>200`, valid | `math_competition_hard` |
| 5 | `EleutherAI/hendrycks_math` | `process_competition_math` | `{problem, solution, level, type}` across 7 subject configs | Loads all 7 configs, filters `level=="Level 5"` only, sorts by solution length | `math_olympiad` |

### Domain 2: Code (2 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 6 | `inclusionAI/Ling-Coder-SFT` | `process_ling_coder` | `{mid, messages: [{content,role}], tags, languages}` | Skips Chinese (CJK char check); requires code in answer (backticks, `def`, `class`) | `code_generation` |
| 7 | `SWE-bench/SWE-smith` | `process_swe_smith` | `{instance_id, patch, problem_statement, repo}` | Filters: `200 < len(patch) < 5000`, `len(problem) > 100`; truncates problem to 2000 chars | `software_engineering` |

### Domain 3: Knowledge/QA (3 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 8 | `mlabonne/open-perfectblend` | `process_open_perfectblend` | `{conversations: [{from,value}]}` or `{messages: [{role,content}]}` | Handles both formats; filters `len(answer) > 200` | `general_knowledge` |
| 9 | `nvidia/HelpSteer3` | `process_helpsteer3` | `{context: [{role,content}], response1, response2, overall_preference}` | Sorts by `abs(preference)` desc; picks winning response; filters `len >= 100` | `reasoning` |
| 10 | `MegaScience/MegaScience` | `process_megascience` | `{question, answer, subject}` | Simple extraction; filters `len(answer) > 200` | `science` |

### Domain 4: Preference/Reasoning (8 datasets, budget=22)

All use a shared `_process_preference_dataset()` helper that extracts the `chosen` response from DPO pairs. Handles `chosen` as message list or string.

| # | Dataset | Processor | Key Schema Detail | Output Tag |
|---|---------|-----------|-------------------|------------|
| 11 | `mlabonne/orpo-dpo-mix-40k` | `process_orpo_dpo` | `chosen`: message list | `benchmark_qa` |
| 12 | `Skywork/Skywork-Reward-Preference-80K-v0.2` | `process_skywork_reward` | `chosen`: message list | `science_math` |
| 13 | `argilla/ultrafeedback-binarized-preferences-cleaned` | `process_ultrafeedback` | `chosen`: message list | `general_qa` |
| 14 | `BAAI/Infinity-Preference` | `process_infinity_pref` | `chosen`: message list | `general_preference` |
| 15 | `nvidia/Nemotron-Post-Training-Dataset-v2` **(GATED)** | `process_nemotron_post` | `messages` format; filters `len(asst) > 200` | `post_training` |
| 16 | `HuggingFaceTB/smoltalk2` | `process_smoltalk` | `messages` format; filters `len(asst) > 50` | `general_conversation` |
| 17 | `lmarena-ai/arena-human-preference-100k` | `process_arena_pref` | Has `conversation_a`/`b` + `winner`; picks winning conv, skips ties | `human_preferred` |
| 18 | `TeichAI/claude-4.5-opus-high-reasoning-250x` | `process_claude_reasoning` | `messages` format; filters `len(answer) > 300` | `high_reasoning` |

### Domain 5: Tool Use (3 datasets, budget=22)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 19 | `Bingguang/HardGen` | `process_hardgen` | Numbered keys `'0','1','2'...` each containing `{content,role}` | Iterates sorted keys, finds user→assistant pair; filters `len(asst) > 50` | `tool_use` |
| 20 | `Salesforce/xlam-function-calling-60k` **(GATED)** | `process_xlam_fc` | `{query, tools, answers}` | Serializes tools/answers to JSON; truncates tools to 1500 chars | `function_calling` |
| 21 | `Team-ACE/ToolACE` | `process_toolace` | `{system, conversations: [{from,value}]}` | Prepends system prompt (truncated to 1000 chars) | `function_calling` |

### Domain 6: Instruction Following (1 dataset, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 22 | `allenai/tulu-3-sft-personas-instruction-following` | `process_tulu_ifeval` | `{prompt, messages: [{role,content}], constraints}` | Uses `prompt` + first assistant message; filters `len(answer) > 100` | `instruction_following` |

### Domain 7: Truthfulness (1 dataset, budget=22)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 23 | `jondurbin/truthy-dpo-v0.1` | `process_truthy_dpo` | `{prompt, chosen, rejected}` (strings, not msg lists) | Direct extraction of `prompt` + `chosen`; filters `len(chosen) > 50` | `truthfulness` |

### Domain 8: Long Context (4 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 24 | `L4NLP/LEval` | `process_leval` | `{instructions, input, outputs}` via parquet | 4 configs (quality, narrative_qa, natural_question, gov_report_summ); handles JSON-encoded lists; truncates doc to 3000 chars | `long_context_qa`, `long_context_narrative`, `long_context_summarization` |
| 25 | `THUDM/LongBench-v2` | `process_longbench` | `{question, choice_A/B/C/D, answer, context, domain}` | Resolves answer key to text; one per domain for diversity; truncates to 3000 chars | `long_context_qa` |
| 26 | `RMT-team/babilong-train-5k-samples` | `process_babilong` | `{input, question, target}` across 4 config/split pairs | Diverse context lengths (4k/8k/16k); truncates to 3000 chars | `long_context_retrieval` |
| 27 | `hotpotqa/hotpot_qa` | `process_hotpotqa` | `{question, answer, supporting_facts, context: {title,[sentences]}}` | Constructs context from title/sentence pairs; truncates to 2000 chars; filters `len(answer) > 2` | `long_context_multihop` |

### Domain 9: Indic NLU (6 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 28 | `CohereForAI/aya_dataset` | `process_aya_dataset` | `{inputs, targets, language, language_code}` | Matches 10 Indic languages; balanced sampling; filters `len(targets) > 30` | `indic_{lang}` |
| 29 | `ai4bharat/indic-align` | `process_indic_align` | Anudesh: `{interactions: [[user,asst],...]}` / ShareLlama: lang columns | Two code paths by config; filters `len(asst) > 50` | `indic_instruction` |
| 30 | `ai4bharat/IndicQA` | `process_indicqa` | `{context, question, answers: {text:[str]}}` via parquet, 6 lang configs | 1 sample per language; truncates context to 500 chars | `indic_{lang}_qa` |
| 31 | `ai4bharat/indic_glue` | `process_indic_glue` | `{question, answer, options}` CSQA subset, 2 configs (hi, ta) | Formats options as numbered list | `indic_{lang}_nlu` |
| 32 | `ai4bharat/MILU` **(GATED)** | `process_milu` | `{question, option1-4, target}` where target=`"option2"` etc. | Resolves target string to answer text; formats as A/B/C/D | `indic_cultural_knowledge` |
| 33 | `ai4bharat/Indic-Bias` **(GATED)** | `process_indic_bias` | `{positive_template, negative_template, topic, concept}` | Constructs bias-analysis prompt from templates; **assistant response is template-generated** | `indic_fairness` |

### Domain 10: Indic Domain Knowledge (4 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 34 | `ryo0634/mgsm-reformatted` | `process_mgsm` | `{question, answer, answer_number, equation_solution}` (Bengali config) | Sorts by answer length desc, takes top n | `indic_math_bn` |
| 35 | `sarvamai/mmlu-indic` | `process_mmlu_indic` | `{question, choices: [str], answer: int}` across 10 lang configs | Round-robins languages; resolves int answer to text | `indic_knowledge_{lang}` |
| 36 | `ai4bharat/indic-instruct-data-v0.1` | `process_indic_instruct` | `{messages: [{content,role}], num_turns}` (Anudesh config, hi split) | Extracts first user/assistant pair; filters `len(asst) > 100` | `indic_instruction_native` |
| 37 | `ai4bharat/MILU` (expanded) **(GATED)** | `process_milu_expanded` | Same as MILU, 8 additional language configs | Covers Gujarati, Odia, Punjabi, Malayalam, Kannada, Marathi, Bengali, Telugu | `indic_cultural_{lang}` |

### Domain 11: Indic Gen/Translation (5 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 38 | `ai4bharat/BPCC` **(GATED)** | `process_bpcc` | `{src, tgt, src_lang, tgt_lang}` config=`bpcc-seed-latest`, splits=lang codes | 8 language splits; prompt: "Translate {lang} to English"; filters `len(src) > 30` | `indic_translation_{lang}_en` |
| 39 | `ai4bharat/IN22-Gen` **(GATED)** | `process_in22_gen` | `{eng_Latn, hin_Deva, tam_Taml, ...}` (22 lang columns), split=`test` | Round-robins 8 languages; prompt: "Translate to {lang}"; filters `len(en) > 30` | `indic_translation_{lang}_en` |
| 40 | `google/IndicGenBench_xquad_in` | `process_indicgenbench_xquad` | Nested: `{examples: [{question, answers, context, lang}]}` | Flattens nested examples; filters 8 Indic languages; truncates context to 2000 chars | `indic_crosslingual_qa_{lang}` |
| 41 | `google/IndicGenBench_crosssum_in` | `process_indicgenbench_crosssum` | Nested: `{examples: {lang, text, summary}}` | Filters 6 languages; filters `500 < len(text) < 3000` | `indic_summarization_{lang}` |
| 42 | `openlanguagedata/flores_plus` **(GATED)** | `process_flores_plus` | `{id, text}` per language config, split=`dev` | Loads English first, then joins Indic configs by ID for parallel pairs; filters `len > 50` | `indic_translation_{lang}_en` |

### Domain 12: Linguistic Diagnostics (2 datasets, budget=21)

| # | Dataset | Processor | Input Schema | Parsing / Selection | Output Tag |
|---|---------|-----------|--------------|---------------------|------------|
| 43 | `nyu-mll/blimp` | `process_blimp` | `{sentence_good, sentence_bad, field, linguistics_term}` across 6 configs | Deterministic A/B shuffle (MD5 hash); **assistant response is template-generated** explaining correct choice | `linguistic_diagnostics` |
| 44 | `nyu-mll/glue` (cola) | `process_cola` | `{sentence, label: 0/1, idx}` | Alternates acceptable/unacceptable; filters `len > 20`; **assistant response is template-generated** | `linguistic_acceptability` |

---

## 4. Allocation Engine

**File:** `scripts/golden_allocator.py`

Given N total samples:

1. **Domain-level:** Distribute N uniformly across 12 domains (default weights = 1/12 each). Floor + distribute fractional remainders.
2. **Dataset-level:** Within each domain, distribute evenly across its datasets. Every dataset gets >= 1 sample.
3. **Guarantee:** `sum(all allocations) == N` and every dataset gets at least 1.

Custom weights can be provided via `--weights-json`.

---

## 5. CLI Usage

```bash
# Dry run (shows allocation table, no downloads)
python scripts/build_golden_dataset.py --total-samples 256 --dry-run

# Full build
HF_TOKEN=hf_xxx python scripts/build_golden_dataset.py \
    --total-samples 256 \
    --output examples/golden_samples_256.jsonl \
    --skip-errors
```

| Argument | Description |
|----------|-------------|
| `--total-samples` | Target sample count (must be >= 44) |
| `--output` | Output JSONL path |
| `--weights-json` | Optional JSON to override domain weights |
| `--dry-run` | Show allocation only |
| `--skip-errors` | Continue past failed datasets |
| `--seed` | Random seed (default: 42) |

---

## 6. Notes for Review

1. **3 processors generate template responses** (not from dataset): `process_indic_bias`, `process_blimp`, `process_cola`. These provide consistent formatting but lower diversity.

2. **`_fmt_fim()` is defined but unused.** Available for future FIM code-completion processors.

3. **Context truncation varies:** Long Context processors use 3000 chars, IndicQA uses 500, HotpotQA uses 2000. No shared constant.

4. **All 8 gated datasets** require HF token auth. Token is auto-detected from `HF_TOKEN` env var or cached `huggingface-cli login`.

5. **Output uses `<|user|>`/`<|assistant|>` special tokens** (not `[USER]`/`[ASSISTANT]`) for proper SFT loss masking during training.
