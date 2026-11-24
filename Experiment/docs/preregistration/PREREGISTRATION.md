# E2 Rescue Configurations: Preregistration Plan

**Project:** Bangla LLM Calibration Thesis  
**Date:** 2026-11-01  
**Preregistration Scope:** Evaluation and rescue attempts for letter-biased / near-chance models (`titullm_1b`, `tigerllm_1b`).  
**Status:** Preregistered prior to running full evaluation inference.

---

## 1. Models Under Study

- **Target Models for Rescue:**
  - `titullm_1b` (huggingface id: `bangla-lm/titullm-1b`, base model)
  - `tigerllm_1b` (huggingface id: `muktai/TigerLLM-1B-it`, instruction-tuned model)
  - Citations: (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 38, cell 39).
- **Untouched Control Model:**
  - `llama32_3b` (`meta-llama/Llama-3.2-3B-Instruct`) is the untouched control model and is not subjected to rescue modifications; it serves as the gate-passing control benchmark.

---

## 2. Evaluation Stages (Cheapest First)

In accordance with compute efficiency and intervention cost, rescue interventions are executed in four sequential stages:

### Stage 1: Dev-Split Probability Priors
- **Type:** Offline label-free prior adjustment.
- **Scope:** Both `titullm_1b` and `tigerllm_1b` in both `bn` and `en`.
- **Implementation:** Evaluated via `run_dev_probabilities()` citing (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 39).
- **Mechanism:** Label-free prior probabilities are estimated on the 200-item dev split per language (`dev_bn.csv`, `dev_en.csv`), recording `prob_A`, `prob_B`, `prob_C`, `prob_D` to adjust evaluation prediction thresholds offline without full rerun inference.

### Stage 2: `tiger_chat`
- **Model:** `tigerllm_1b` only (`titullm_1b` is an un-instructed base model lacking a chat template).
- **Exact Configuration Key & Parameters:** Citing (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 39, `E2_CONFIGS["tiger_chat"]`):
  - `output_dir_name`: `"results_E2_chat_template"`
  - `description`: `"Chat template enabled for TigerLLM-1B-it only"`
  - `models`: `["tigerllm_1b"]`
  - `settings`:
    - `prompt_variant`: `"few_shot"`
    - `n_shots`: `4`
    - `use_chat_template`: `True`
    - `debias`: `"none"`

### Stage 3: `0shot_answer_cue`
- **Models:** `titullm_1b` and `tigerllm_1b`.
- **Exact Configuration Key & Parameters:** Citing (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 39, `E2_CONFIGS["0shot_answer_cue"]`):
  - `output_dir_name`: `"results_E2_0shot_answer_cue"`
  - `description`: `"Zero-shot with answer cue prompt variant"`
  - `models`: `["titullm_1b", "tigerllm_1b"]`
  - `settings`:
    - `prompt_variant`: `"answer_cue"`
    - `n_shots`: `0`
    - `use_chat_template`: `False`
    - `debias`: `"none"`

### Stage 4: `cyclic4`
- **Models:** `titullm_1b` and `tigerllm_1b`.
- **Exact Configuration Key & Parameters:** Citing (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 39, `E2_CONFIGS["cyclic4"]`):
  - `output_dir_name`: `"results_E2_cyclic4"`
  - `description`: `"Exemplar cyclic permutation (cyclic4 debiasing)"`
  - `models`: `["titullm_1b", "tigerllm_1b"]`
  - `settings`:
    - `prompt_variant`: `"few_shot"`
    - `n_shots`: `4`
    - `use_chat_template`: `False`
    - `debias`: `"cyclic4"`
- **Cost Note:** Involves 4 cyclic permutations of few-shot exemplars, requiring 4x evaluation passes per item.

---

## 3. Evaluation Items and Splits

- **Evaluation Benchmark:** 1500 Bengali items (`bn_subset.csv`) and 1500 English items (`en_subset.csv`), sampled under `seed: 42`. Citing (`Bangla-LLM-Calibration-results(v4e)/bn_subset.csv`, `Bangla-LLM-Calibration-results(v4e)/en_subset.csv`).
- **Dev Split:** 200 Bengali items (`dev_bn.csv`) and 200 English items (`dev_en.csv`). Citing (`Bangla-LLM-Calibration-results(v4e)/dev_bn.csv`, `Bangla-LLM-Calibration-results(v4e)/dev_en.csv`).
- **Disjointness Constraint:** The 200-item dev split is strictly reserved for screening and Stage 1 prior estimation. Zero overlap is verified against evaluation items (`bn_subset.csv`, `en_subset.csv`) and few-shot exemplars (`fewshot_exemplars.json`) citing (`Bangla-LLM-Calibration-Real-Data-v4e-translation-parallel.ipynb`, cell 39, assertions in `run_dev_probabilities`).

---

## 4. Dev Screening Rule

- Before running a full 1500-item evaluation for any model-config pair, inference is screened on the 200-item dev set.
- **Screening Criterion:**
  - If dev `top_share` > 0.40 in either language (Bengali or English), the configuration is marked as **"failed dev screen"**.
  - Any model-config failing the dev screen is skipped for the full 1500-item evaluation to preserve computational resources.
  - The failure on the dev screen must still be documented and reported in the full results summary.

---

## 5. Strict Gate Decision Rule

A configuration is preregistered to rescue a model **ONLY** if the full 1500-item evaluation run satisfies all strict gate criteria in **BOTH** languages (`bn` and `en`) simultaneously. Citing (`gate.yaml`, lines 4–22):

1. **Letter-Bias Gate:**
   - Maximum single-letter share: $\text{top\_share} \le 0.40$ citing (`gate.yaml`, key `top_share_max: 0.40`).
   - Minimum single-letter share: $\text{min\_letter\_share} \ge 0.10$ across all options (A, B, C, D) citing (`gate.yaml`, key `min_letter_share: 0.10`).
2. **Accuracy Gate:**
   - Accuracy must statistically exceed the language's majority-class baseline (derived from the gold answer distribution of the evaluation items; for the 1500 items, $\approx 27.53\%$).
   - Test: exact one-sided binomial test ($\alpha = 0.05$) citing (`gate.yaml`, lines 11–16, keys `majority-class baseline`, `alpha: 0.05`).
3. **Assessment Protocol:**
   - Letter shares and bias metrics must be computed strictly from per-item prediction files (e.g., predicted letters from CSV item outputs), never inferred solely from aggregate accuracy or Expected Calibration Error (ECE).
4. **Outcome Classification:**
   - If a model-config fails any of the gate criteria in either language, it is labeled:
     `"letter-biased / near-chance, excluded from calibration claims"` citing (`gate.yaml`, lines 20–22).

---

## 6. Preregistered Reporting Rules

- **Complete Transparency:** All tested configurations, whether passing or failing, must be fully reported in the final validity table.
- **Exploratory Labelling:** Any configuration, modification, or post-processing added after inspecting empirical results must be explicitly designated and labeled as **"post-hoc, exploratory"**.
- **Correlational Language:** In adherence with scientific calibration standards, all findings and descriptions are reported strictly in correlational terms, avoiding unwarranted causal assertions.

---

## 7. Preregistration Amendments

### Amendment 1: Authoritative Accuracy Gate Baseline Definition
- **Date:** 2026-11-02
- **Authoritative Source:** `gate.yaml` (lines 11–16).
- **Context & Correction:** The original text in Section 5.2 included an editorial error parenthetical equating the majority-class baseline to `"(25% chance level for 4-choice items)"`. `gate.yaml` is the sole authoritative pre-registered standard and explicitly specifies:
  > *"Accuracy must exceed the language's majority-class baseline. Test: exact binomial, one-sided (accuracy > baseline)."*
- **Clarification:** For the 1500-item evaluation subsets (`bn_subset.csv` and `en_subset.csv`), the gold answer majority class is letter B with 413 occurrences, establishing an empirical majority-class baseline of $413 / 1500 \approx 27.5333\%$ in both Bengali and English. All binomial tests and accuracy gating are strictly conducted against this empirical majority baseline (yielding $p = 8.74 \times 10^{-15}$ for `llama32_3b` BN and $p = 8.73 \times 10^{-137}$ for `llama32_3b` EN), exactly matching the pre-registered `gate.yaml` criteria and benchmark evaluation summaries.
