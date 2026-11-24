# Preregistration Amendment 2: E3 Extension for Expanded Tokenizer Models

**Project:** Bangla LLM Fragmentation and Calibration  
**Preregistration Protocol:** Amendment 2 to [PREREGISTRATION.md](preregistration/PREREGISTRATION.md)  
**Date:** [DATE: 2026-11-12]  
**Status:** Preregistered prior to running full evaluation inference for new candidate models.  
**Authoritative Baselines & Thresholds:** `gate.yaml`, `FRAGMENTATION_DEFINITION.md`

---

## 1. Motivation & Context
The initial preregistration plan **PREREGISTRATION.md** specified evaluation of three primary models:
1. `llama32_3b` (`meta-llama/Llama-3.2-3B-Instruct`, untouched reference control)
2. `titullm_1b` (`hishab/titulm-llama-3.2-1b-v1.1`, Bangla continued-pretraining of Llama 3.2)
3. `tigerllm_1b` (`md-nishat-008/TigerLLM-1B-it`, Gemma-3 derivative)

Task S8's all-seven tokenizer fragmentation and identity audit revealed that `llama32_3b` and `titullm_1b` share an **identical byte-level BPE tokenizer** (128,256 vocabulary, identity hash `460c071a5a2d`), yielding identical Bangla tokens-per-word ($\text{TPW} = 6.9687$). Consequently, the initial 3-model pool spanned only $N=2$ distinct tokenizer families (`460c071a5a2d` and Gemma `ddecfb23fef2`), presenting an empirical collinearity barrier for cross-tokenizer rank correlation tests of Hypothesis $H_1$ (tokenizer fragmentation predicting miscalibration).

To resolve this limitation and provide robust statistical power across diverse tokenizer designs, four additional 1–4B open models spanning distinct tokenizer families have been audited and approved for evaluation under the **E3 Extension Block**.

---

## 2. Added Models Under E3 Extension
The four approved new models are registered with exact Hugging Face repository IDs:

| Model Key | Exact Hugging Face Model ID | Architecture / Family | Size (Params) | Tokenizer Family | Vocab Size | Tokenizer Hash | Target Role |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :--- |
| `phi35_mini` | `microsoft/Phi-3.5-mini-instruct` | Phi-3.5 | 3.82B | `phi_family` (BPE) | 32,011 | `fa274a672937` | Low-vocab Latin-centric BPE (highest BN TPW = 7.12) |
| `qwen25_3b` | `Qwen/Qwen2.5-3B-Instruct` | Qwen-2.5 | 3.09B | `qwen_family` (tiktoken BPE) | 151,665 | `678cbc46a291` | Large multilingual BPE with native Asian coverage (BN TPW = 6.21) |
| `bloomz_3b` | `bigscience/bloomz-3b` | BLOOM | 3.00B | `bloom_family` (multilingual BPE) | 250,680 | `2f9a59b5b24c` | Script-efficient multilingual native BPE (lowest BN TPW = 2.21) |
| `llama32_3b_base` | `meta-llama/Llama-3.2-3B` | Llama-3.2 | 3.21B | `llama_family` (byte-level BPE) | 128,256 | `460c071a5a2d` | Base unaligned counterpart to instruct control |

---

## 3. Selection Criteria
The four candidate models were selected based on five pre-specified criteria:
1. **Tokenizer Diversity Expansion ($N=5$):** Expanding from the existing 2 distinct tokenizer levels to 5 distinct levels, enabling rank-correlation testing across vocabulary philosophies without collinearity collapse.
2. **Parameter Scale Comparability:** Models span 1.0B to 3.82B parameters, matched to the existing evaluation regime (1.0B to 3.21B) to avoid scale confounding.
3. **Hardware Feasibility (NVIDIA T4 16GB VRAM):** All four models require $\le 8.1$ GB VRAM in `float16` inference (including KV cache for sequence length 2048 at batch size 1), strictly respecting the Kaggle T4 usable memory ceiling ($\le 15.3$ GB).
4. **Hub Availability & Open License:** Publicly accessible on the Hugging Face Hub with verified architectures and loadable tokenizers.
5. **Script Efficiency Spectrum:** Encompasses the full spectrum of Bengali fertility, from script-efficient native multilingual tokenization (`bloomz_3b`: $\text{TPW} = 2.21$) to high-fertility subword fragmentation (`phi35_mini`: $\text{TPW} = 7.12$).

---

## 4. Invariant Experimental Settings
To prevent procedural drift and preserve cross-model comparability, all experimental settings remain strictly identical to the pre-registered protocol:
- **Few-Shot Exemplars:** Exactly 4 in-context exemplars (`n_shots=4`), drawn from the standardized held-out pool (`fewshot_exemplars.json`).
- **Prompt Format:** Standardized few-shot prompt with question, options, and natural completion cue (`prompt_variant="few_shot"`).
- **Chat Template Setting:** Strictly **OFF** (`use_chat_template=False`) across all models, ensuring prompt formatting invariance across models.
- **Random Seed:** Fixed to `seed=42`.
- **Position Debiasing:** Cyclic permutation debiasing (`debias="cyclic4"`), averaging probabilities over 4 cyclic option order shifts.
- **Evaluation Subsets:** Identical 1,500 question IDs per language (`bn_subset.csv`, `en_subset.csv`), disjoint from few-shot exemplars and dev splits.
- **Sample Size:** $N = 1500$ items per language (3,000 items total per model).
- **Batch Size:** $B = 1$ with adaptive OOM reduction guard.

---

## 5. Dev Screening, Gating, and Near-Chance Labeling Protocol
Prior to any full 1,500-item evaluation run, each candidate model undergoes a two-step validation on the held-out 200-item dev splits (`dev_bn.csv`, `dev_en.csv`):

### Step 1: Dev Screening & Throughput Projection
- Evaluated on $n=200$ items per language ($N_{\text{dev}} = 400$).
- Wall-clock runtime $T_{\text{dev}}$ is measured to compute actual throughput:
  $$\text{Throughput} = \frac{N_{\text{dev}}}{T_{\text{dev}}} \quad (\text{items/sec})$$
  $$\text{Projected Full Run Time} = \frac{3000}{\text{Throughput}} \quad (\text{sec})$$

### Step 2: Letter-Bias Collapse Gate
Evaluates prediction distribution across candidate options:
- $\text{top\_share} \le 0.40$
- $\text{min\_letter\_share} \ge 0.10$
Must pass in **both** Bengali and English.

### Step 3: Majority-Class Baseline Test (Preregistered Protocol)
- Empirical majority-class frequency on benchmark items: $p_{\text{majority}} = 413 / 1500 \approx 27.5333\%$ (letter B frequency).
- Exact one-sided binomial test of accuracy:
  $$H_0: p \le p_{\text{majority}} \quad \text{vs} \quad H_1: p > p_{\text{majority}} \quad (\alpha = 0.05)$$
- The model must statistically exceed the majority baseline in **both** languages.

### Step 4: Near-Chance Labeling and Full-Run Eligibility
- If a model fails either the collapse gate or the majority-class test in either language:
  - Labeled: **`"letter-biased / near-chance, excluded from calibration claims"`** per `gate.yaml` (lines 20–22).
  - The model is **ineligible** for the full 1,500-item evaluation run. Full evaluation is skipped to prevent wasted GPU compute and invalid calibration assertions.
- **Full evaluation is launched only for eligible models** that satisfy all gates in both languages.

---

## 6. Output Artifacts and Isolation
Each candidate model's outputs are saved in an isolated directory:
`results_E3_<model>/` (e.g. `results_E3_phi35_mini/`, `results_E3_qwen25_3b/`, `results_E3_bloomz_3b/`, `results_E3_llama32_3b_base/`).

Each directory contains:
- `bn_results.csv` & `en_results.csv`:
  - Per-item `pred_raw`, `confidence_raw`, `prob_raw_A`–`prob_raw_D`
  - Per-item `predicted_answer` (cyclic4), `confidence`, `prob_A`–`prob_D`
  - Per-item correctness indicators (`is_correct_raw`, `is_correct_cyclic4`)
- `shift_checkpoints/`: Intermediate shift outputs for interruption recovery.
- `summary.json`, `bn_summary.json`, `en_summary.json`: Top share, min letter share, raw distribution, gate status, and accuracy metrics.
- `assertion_status.json`: Complete PASS/FAIL record of all runtime assertions.
- `run_manifest.json`: Execution runtime, wall-clock seconds, batch size, max observed sequence length, and model provenance.
- `results_E3_<model>.zip`: Self-contained downloadable archive.

---

## 7. Untouched Control Invariance
- **LLaMA-3.2-3B-Instruct (`llama32_3b`):** Remains the untouched reference control. It is **not** re-evaluated under E3.
- **Existing Results:** All prior results (`results_E2_cyclic4_*` and Run A outputs) remain read-only and immutable.
- **Epistemic Framing:** All subsequent econometric models estimating relationships between fragmentation metrics ($\text{TPW}$, $\text{STRR}$, $\text{TPC}$) and calibration error ($\text{ECE}$, Brier score, residuals) will be reported strictly as **observational associations**, never causal effects.
