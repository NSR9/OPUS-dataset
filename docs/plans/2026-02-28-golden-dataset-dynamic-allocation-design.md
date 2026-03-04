# Design: Dynamic Golden Dataset Builder

**Date:** 2026-02-28
**Status:** Approved
**Scope:** Refactor `scripts/build_golden_dataset.py` to support dynamic sample counts with 44 datasets across 12 domains

---

## Problem

The current `build_golden_dataset.py` is hardcoded for exactly 128 samples with fixed per-processor limits (`if count >= 7: break`). We need to:

1. Support any user-specified total (e.g., 64, 128, 256, 512)
2. Add 14 new datasets (13 new + 1 expanded) for 7 new benchmarks
3. Guarantee every dataset gets at least 1 sample (strict rule, 44 datasets)
4. Dynamically redistribute when total changes

---

## Architecture: Allocator + Registry

### Dataset Registry

A single `DATASET_REGISTRY` list defines all 44 datasets:

```python
DATASET_REGISTRY = [
    # (dataset_id, domain, tag, processor_fn, gated)
    ("openai/gsm8k",            "Math",                "math_reasoning",         process_gsm8k,           False),
    ("nvidia/Nemotron-Math-v2",  "Math",                "math_competition",       process_nemotron_math,   False),
    # ... all 44 entries
]
```

### Domain Weights

Default: uniform 1/12 per domain. Overridable via `--weights-json`.

```python
DEFAULT_DOMAIN_WEIGHTS = {
    "Math": 1/12, "Code": 1/12, "Knowledge/QA": 1/12,
    "Preference/Reasoning": 1/12, "Tool Use": 1/12,
    "Instruction Following": 1/12, "Truthfulness": 1/12,
    "Long Context": 1/12, "Indic NLU": 1/12,
    "Indic Domain Knowledge": 1/12, "Indic Gen/Translation": 1/12,
    "Linguistic Diagnostics": 1/12,
}
```

Weights JSON override format:
```json
{
  "domain_weights": {
    "Math": 0.083, "Code": 0.083, ...
  }
}
```

### Allocation Algorithm

```
INPUT: N (total samples), domain_weights, DATASET_REGISTRY

STEP 1: Domain-level allocation (uniform default)
  - 12 domains, each gets: floor(N * domain_weight) samples
  - But never less than num_datasets_in_domain (to honor >=1 rule)
  - Remainder (N - sum) distributed round-robin to domains with most datasets first

STEP 2: Within-domain distribution
  - Each dataset gets: floor(domain_budget / num_datasets_in_domain)
  - Remainder distributed round-robin across datasets in domain
  - Every dataset guaranteed >=1

STEP 3: Validation
  - Assert sum of all allocations == N
  - Assert every dataset has >=1
  - If N < 44: ERROR (impossible)

OUTPUT: Dict[dataset_id, int] — per-dataset sample count
```

### Processor Refactor

Every processor gets an `n: int` parameter replacing hardcoded limits:

```python
# Before:
def process_gsm8k() -> List[Dict]:
    ...
    if count >= 5:  break  # hardcoded

# After:
def process_gsm8k(n: int) -> List[Dict]:
    ...
    if count >= n:  break  # dynamic
```

### Main Loop

```python
allocation = compute_allocation(total_samples, domain_weights)

for dataset_id, domain, tag, processor_fn, gated in DATASET_REGISTRY:
    n = allocation[dataset_id]
    samples = processor_fn(n)
    # collect, validate, continue
```

---

## 14 New Processors — Verified Schemas

### 1. process_mgsm(n) — juletxara/mgsm

```
Schema:  {question: str, answer: str, answer_number: int/float, equation_solution: str}
Split:   test (250/lang)
Configs: bn (Bengali) — only Indic language available
Loading: load_dataset("juletxara/mgsm", "bn", split="test")
Format:  [USER] {question} -> [ASSISTANT] {answer}
Note:    Only Bengali available as Indic. Other Indic math coverage from indic-instruct.
```

### 2. process_mmlu_indic(n) — sarvamai/mmlu-indic

```
Schema:  {question: str, choices: list[str], answer: int (0-3)}
Split:   test (14042/lang)
Configs: hi, ta, te, bn, kn, ml, mr, or, pa, gu (native script)
Loading: load_dataset("sarvamai/mmlu-indic", "{lang}", split="test")
Format:  [USER] {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}
         -> [ASSISTANT] The correct answer is {chr(65+answer)}. {choices[answer]}
Parse:   Cycle through configs to get 1 sample per language, round-robin until n reached.
```

### 3. process_trivia_qa_indic(n) — sarvamai/trivia-qa-indic (GATED)

```
Schema:  {question: str, answer: {answer_aliases: list[str], normalized_aliases: list[str]}}
Split:   test/validation
Configs: per-language (behind gate)
Loading: load_dataset("sarvamai/trivia-qa-indic", "{lang}", split="test", token=True)
Format:  [USER] {question} -> [ASSISTANT] {answer.answer_aliases[0]}
Parse:   answer is a dict, extract answer_aliases[0] as the primary answer text.
```

### 4. process_indic_instruct(n) — ai4bharat/indic-instruct-data-v0.1

```
Schema:  VARIES by config:
  Config "anudesh", split "hi":  rows are messages with {content: str, role: str}
  Config "flan_v2", split "hi":  {instruction: str, context: str, response: str}
Loading: load_dataset("ai4bharat/indic-instruct-data-v0.1", "anudesh", split="hi")
Format (anudesh): Collect consecutive role=user/role=assistant pairs.
  [USER] {user_content} -> [ASSISTANT] {assistant_content}
Parse:   Iterate rows. Group consecutive user+assistant messages into pairs.
         Skip system messages. Filter for len(assistant_content) > 100.
Note:    Use "anudesh" config only (native crowd-sourced, not translated).
```

### 5. process_milu_expanded(n) — ai4bharat/MILU (GATED, expansion)

```
Schema:  {question: str, option1: str, option2: str, option3: str, option4: str, target: str}
Split:   test
Configs: Gujarati, Odia, Punjabi, Malayalam (new languages beyond existing Hindi/Tamil)
Loading: load_dataset("ai4bharat/MILU", data_dir="{lang}", split="test", token=True)
Format:  [USER] {question}\nA. {option1}\nB. {option2}\nC. {option3}\nD. {option4}
         -> [ASSISTANT] The answer is {resolve target to option text}.
Parse:   target is "option1"/"option2"/etc. Resolve: answer_text = row[target].
         Cycle through 4 new language configs, round-robin until n reached.
```

### 6. process_bpcc(n) — ai4bharat/BPCC (GATED)

```
Schema:  Parallel text files per language pair directory.
  Directory structure: eng_Latn-hin_Deva/train.eng_Latn, train.hin_Deva
Config:  "bpcc-seed-latest" or direct file loading
Loading: TBD — may need load_dataset with data_dir, or direct parquet/CSV download.
  Fallback: load_dataset("ai4bharat/BPCC", data_dir="eng_Latn-hin_Deva", token=True)
Format:  [USER] Translate the following {src_lang} text to {tgt_lang}:\n\n{source_sentence}
         -> [ASSISTANT] {target_sentence}
Parse:   Load source + target files for each language pair. Zip lines together.
         Language pairs: Hi, Ta, Te, Bn, Kn, Mr, Ml, Gu (all with eng_Latn).
         Cycle through pairs round-robin until n reached.
         Filter: len(source) > 30 chars (skip trivial greetings).
Note:    Non-standard format. If HF datasets API doesn't work, fall back to
         downloading parquet from refs/convert/parquet branch.
```

### 7. process_in22_gen(n) — ai4bharat/IN22-Gen (GATED)

```
Schema:  {id: int, sentence: str, context: str, source: str, url: str, domain: str,
          num_words: int, bucket: str}
  For paired configs: also has sentence_{lang_code} field.
Split:   "gen" (single split, 1024 sentences)
Configs: Language pair format "eng_Latn-hin_Deva" (506 total directions)
Loading: load_dataset("ai4bharat/IN22-Gen", "eng_Latn-hin_Deva", split="gen", token=True)
Format:  [USER] Translate to Hindi:\n\n{sentence_eng} -> [ASSISTANT] {sentence_hin}
Parse:   Load a paired config. The row has both source and target language sentences.
         Cycle through pairs: Hi, Ta, Te, Bn. Pick diverse domains (news, culture, etc.).
```

### 8. process_indicgenbench_xquad(n) — google/IndicGenBench_xquad_in

```
Schema:  {context: str, question: str, answers: list[{text: str}], lang: str}
Split:   validation
Config:  default, with field="examples"
Loading: load_dataset("google/IndicGenBench_xquad_in", field="examples", split="validation")
Format:  [USER] {context}\n\nQuestion: {question}
         -> [ASSISTANT] {answers[0]["text"]}
Parse:   answers is list of dicts. Use answers[0]["text"].
         Filter by lang in ["hi", "ta", "te", "bn", "kn", "ml"].
         Cycle through languages round-robin until n reached.
```

### 9. process_indicgenbench_crosssum(n) — google/IndicGenBench_crosssum_in

```
Schema:  {text: str, summary: str, lang: str}
Split:   validation
Config:  default, with field="examples"
Loading: load_dataset("google/IndicGenBench_crosssum_in", field="examples", split="validation")
Format:  [USER] Summarize the following article in {lang_name}:\n\n{text}
         -> [ASSISTANT] {summary}
Parse:   Filter by lang in Indic languages. Pick medium-length text (500-1500 chars).
         Cycle through languages round-robin until n reached.
```

### 10. process_flores_plus(n) — openlanguagedata/flores_plus (GATED)

```
Schema:  {id: str, text: str, iso_639_3: str, iso_15924: str, domain: str, topic: str, ...}
Split:   dev (997/lang)
Configs: per-language (e.g., "hin_Deva", "tam_Taml", "eng_Latn")
Loading: load_dataset("openlanguagedata/flores_plus", "hin_Deva", split="dev", token=True)
Format:  [USER] Translate Hindi to English:\n\n{text_hi}
         -> [ASSISTANT] {text_en}
Parse:   Load SAME id from two language configs to create parallel pairs.
         Load eng_Latn as reference. For each Indic lang, match by id.
         Pairs: Hi-En, Ta-En, Te-En, Bn-En. Round-robin until n reached.
         Prefer sentences with len > 50 chars.
```

### 11. process_blimp(n) — nyu-mll/blimp

```
Schema:  {sentence_good: str, sentence_bad: str, field: str, linguistics_term: str,
          UID: str, simple_LM_method: bool, one_prefix_method: bool, ...}
Split:   train (1000/config)
Configs: 67 configs. Use these 6 for diversity:
  "anaphor_number_agreement", "regular_plural_subject_verb_agreement_1",
  "wh_questions_subject_gap", "principle_A_domain_1",
  "sentential_negation_npi_scope", "tough_vs_raising_1"
Loading: load_dataset("nyu-mll/blimp", "{config}", split="train")
Format:  [USER] Which sentence is grammatically correct and why?
         A: {sentence_good}
         B: {sentence_bad}
         -> [ASSISTANT] Sentence A is correct. This tests {linguistics_term}...
Parse:   Cycle through 6 configs round-robin until n reached.
         Randomly swap A/B order (50%) so model doesn't learn positional bias.
```

### 12. process_cola(n) — nyu-mll/glue (CoLA)

```
Schema:  {sentence: str, label: int (0=unacceptable, 1=acceptable), idx: int}
Split:   train (8551)
Config:  "cola"
Loading: load_dataset("nyu-mll/glue", "cola", split="train")
Format:  [USER] Is this sentence grammatically acceptable? Explain why or why not.
         "{sentence}"
         -> [ASSISTANT] {"Yes, this sentence is grammatically acceptable." if label==1
                         else "No, this sentence is not grammatically acceptable."}
                        {brief explanation based on sentence structure}
Parse:   Alternate: pick 1 acceptable, 1 unacceptable, repeat until n reached.
         Filter: len(sentence) > 20 (skip trivially short).
Note:    test split labels are -1 (hidden). Use train split only.
```

### 13. process_numinamath(n) — AI-MO/NuminaMath-1.5

```
Schema:  {problem: str, solution: str, answer: str, source: str,
          synthetic: bool, problem_type: str, question_type: str,
          problem_is_valid: str, solution_is_valid: str}
Split:   train (896K, streaming required)
Config:  default
Loading: load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=True)
Format:  [USER] {problem} -> [ASSISTANT] {solution}
Filter:  source in ("aime", "imo", "amc", "putnam") AND synthetic == False
         AND solution_is_valid == "true" AND len(solution) > 200
Parse:   Stream, filter, pick top n by solution length.
```

### 14. process_competition_math(n) — hendrycks/competition_math

```
Schema:  {problem: str, solution: str, answer: str, subject: str, level: int (1-5),
          unique_id: str}
Split:   train (12000)
Config:  default
Loading: load_dataset("hendrycks/competition_math", split="train", streaming=True)
Format:  [USER] {problem} -> [ASSISTANT] {solution}
Filter:  level == 5 (hardest only). Prefer diverse subjects.
Parse:   Stream, filter level==5, pick top n by solution length.
         Cycle through subjects for diversity.
```

---

## HuggingFace Authentication

Auto-detect token with priority:
1. `HF_TOKEN` environment variable
2. Cached token from `huggingface-cli login` (`~/.cache/huggingface/token`)

```python
def _get_hf_token() -> Optional[str]:
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
```

At startup: if no token found, print warning listing 5 gated datasets that will be skipped. The `_safe_load` helper gains a `gated: bool` parameter — if `gated=True` and no token, returns None immediately.

Gated datasets: sarvamai/trivia-qa-indic, ai4bharat/BPCC, ai4bharat/IN22-Gen, openlanguagedata/flores_plus, ai4bharat/MILU.

---

## CLI Interface

```
python scripts/build_golden_dataset.py \
  --total-samples 128 \           # REQUIRED. Warns if not power-of-2.
  --output examples/golden_samples.jsonl \
  --weights-json configs/golden_weights.json \  # optional
  --dry-run \                      # optional: show allocation table, no downloads
  --skip-errors \
  --seed 42 \
  --verbose
```

### Dry-run Mode

`--dry-run` computes the allocation table and prints it, then exits without downloading any data. Shows:
- Per-domain budget
- Per-dataset sample count within each domain
- Gated dataset warnings
- HF token status
- Total validation (sum == N, all datasets >= 1)

---

## Validation & Safety

1. **N < 44** -> hard error ("Cannot allocate: need at least 44 samples for 44 datasets")
2. **N not power-of-2** -> warning ("N=100 is not a power of 2. GPU-friendly sizes: 64, 128, 256, 512")
3. **Every dataset >= 1** -> assertion after allocation
4. **Sum == N** -> assertion after allocation
5. **Post-build verification** -> after all processors run, check no dataset returned 0 samples. If any did (gated/broken), warn and list them.
6. **Sequential IDs** -> re-number g1, g2, ..., gN at the end (existing behavior preserved)
