# Project Overview: Bangla LLM Fragmentation and Calibration

## 1. Research question

*Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? A Cross-Lingual Study.*

The project measures whether tokenization fragmentation (tokens per word TPW, tokens per character TPC, single-token retention rate STRR) is associated with confidence miscalibration (ECE, Brier, NLL, accuracy, mean confidence) when LLMs answer the same BnMMLU questions in Bangla (BN) and English (EN). All results are observational associations. No causal claim is made.

## 2. Dataset

- **Benchmark:** BnMMLU (train split, the only split published on the Hub).
- **Source file:** `bnmmlu_full_en.csv` (wide translation CSV with `question_bn` / `question_en` and four options per language).
- **Pairing:** English items are translations of the same Bangla questions, paired on `question_id`. The English side is therefore a translation, not a natural English benchmark.
- **Evaluation subsets:** `bn_subset.csv` and `en_subset.csv`, 1,500 items per language, seed 42, subject-balanced.
- **Dev subsets:** `dev_bn.csv` and `dev_en.csv`, 200 items per language, used for screening only.
- **Few-shot exemplars:** `fewshot_exemplars.json` (balanced across A-D), disjoint from evaluation and dev items.
- **Exclusions:** Bengali language-specific subjects, MathML items, untranslated identical pairs, Bangla script left on the English side, duplicates.
- **Tokenized text:** question plus options only (static few-shot prefix excluded), `add_special_tokens=False`, words counted by whitespace split.

## 3. Models

| Model | Hugging Face ID | Tokenizer | Role |
|---|---|---|---|
| `llama32_3b` | meta-llama/Llama-3.2-3B-Instruct | Llama-3.2 BPE (128,256) | main |
| `titullm_1b` | hishab/titulm-llama-3.2-1b-v1.1 | Llama-3.2 BPE (128,256) | main |
| `llama32_3b_base` | Llama-3.2-3B base | Llama-3.2 BPE (128,256) | main (base model) |
| `tigerllm_1b` | md-nishat-008/TigerLLM-1B-it | Gemma SentencePiece (262,145) | main |
| `qwen25_3b` | Qwen2.5-3B | Qwen BPE (151,665) | main |
| `bloomz_3b` | BLOOMZ-3B | BLOOM BPE (250,680) | main |
| `phi35_mini` | Phi-3.5-mini-instruct | Phi BPE (32,011) | exploratory |
| `mgpt_1_3b`, `olmo2_1b`, `smollm2_1_7b`, `sarvam1_2b` | various | various | E4 extension (none qualified) |

`llama32_3b`, `titullm_1b` and `llama32_3b_base` use the same tokenizer, so the six main models have **four distinct tokenizer levels** with calibration data (Llama, Gemma, Qwen, BLOOM).

## 4. Metrics

Definitions follow `FRAGMENTATION_DEFINITION.md` and `analysis_plan.yaml`.

- **TPW** = N_tokens / max(N_words, 1). **TPC** = N_tokens / max(characters, 1). **STRR** as defined in `FRAGMENTATION_DEFINITION.md`.
- **Confidence:** maximum normalized probability over option letters A-D, pooling single-token variants such as "A" and " A".
- **ECE:** 10 uniform bins (primary); 15-bin and equal-mass 10-bin variants also reported.
- **Brier:** 4-class multiclass Brier (uniform baseline 0.75). **NLL** baseline is ln 4 = 1.386.
- **Uncertainty:** item-level bootstrap, 2,000 resamples, seed 42.
- **Tests:** exact one-sided binomial test against 25% chance; exact McNemar test for paired BN vs EN correctness; Holm correction for multiple tests.

## 4a. Validity gates

Thresholds in `gate.yaml`: top-letter share at most 0.40, minimum letter share at least 0.10, alpha 0.05. Dev screening on 200 items per language runs before any full 1,500-item run. A majority-class check (exact one-sided binomial test against 413/1500, about 27.53%) is applied as well.

## 5. Experiment history

- **v4c / v4e:** original three-model runs with no debiasing (v4c used subject-matched MMLU English; v4e used the item-parallel translation). Raw outputs showed strong letter preference for `titullm_1b` and `tigerllm_1b`.
- **E2 rescue (`Bangla-LLM-Calibration-E2-rescue-v4h.ipynb` to v4i):** cyclic4 debiasing (four cyclic permutations of options, probabilities averaged back to canonical order). Cyclic4 brought all top-letter shares into 0.268-0.313.
- **E3 extension (`Bangla-LLM-Calibration-E2-rescue-v4i.ipynb`, Section 14g):** added `qwen25_3b`, `bloomz_3b`, `llama32_3b_base` (main set) and `phi35_mini` (exploratory), all with the same invariant settings. `PREREGISTRATION_AMENDMENT.md` records the amendment.
- **E4 (`Bangla-LLM-Calibration-E2-rescue-v4j.ipynb`):** `mgpt_1_3b`, `olmo2_1b`, `smollm2_1_7b`, `sarvam1_2b` screened under rules R1-R7 (`E4_CHECK_REPORT.md`, `e4_added_models_plan.json`). None qualified for the main extension.

Invariant settings across the main set: 4-shot prompts, no chat template, cyclic4, seed 42, 1,500 items per language, T4 GPU, float16 (float32 for `tigerllm_1b`).

## 6. Repository layout

```
bangla-llm-fragmentation-calibration/
├── README.md
├── pyproject.toml
├── .gitignore
│
├── docs/
│   ├── literature_review/
│   │   └── Paper and Literature Review Tracker(We_have_a_trust_issue)_06.xlsx
│   ├── preregistration/
│   │   ├── analysis_plan.yaml
│   │   ├── FRAGMENTATION_DEFINITION.md
│   │   ├── gate.yaml
│   │   ├── PREREGISTRATION_AMENDMENT.md
│   │   └── PREREGISTRATION.md
│   ├── project_overview.md
│   └── reports/
│       ├── AUDIT_REPORT.md
│       ├── FINAL_REPORT.md
│       ├── project_overview.md
│       └── report.md
│
├── notebooks/
│   ├── Bangla-LLM-Calibration-E2-rescue-v4j.ipynb
│   └── Bangla-LLM-Calibration-Real-Full-run-data-v4e-translation-parallel.ipynb
│
├── paper/
│   └── main.log
│
├── results/
│   ├── figures/
│   │   ├── bn_en_paired.png
│   │   ├── dev_screen_vs_full.png
│   │   ├── e4_tpw_calibration.png
│   │   ├── fragmentation_tpw_box.png
│   │   ├── quartile_ece.png
│   │   ├── raw_vs_cyclic4.png
│   │   ├── regression_coefs.png
│   │   └── reliability_diagrams.png
│   ├── report_work/
│   │   └── Analysis_Report_6models_plus_phi_pluss_e4.pdf
│   └── tables/
│
├── scripts/
│   ├── analysis_report_e4.py
│   ├── analysis_v2.py
│   ├── check_inputs.py
│   ├── discrimination.py
│   ├── run_s8_calculation.py
│   └── verify_parallel_subsets.py
│
├── src/
│   ├── data/
│   │   ├── bnmmlu_full_en.csv
│   │   ├── full/
│   │   │   ├── bnmmlu_full_failed.csv
│   │   │   ├── bnmmlu_full_translated_to_en_from_0k_to_70k_dataset.csv
│   │   │   └── bnmmlu_full_translated_to_en_from_70k_to_134k_dataset.csv
│   │   └── hard/
│   │       ├── bnmmlu_hard_failed.csv
│   │       └── bnmmlu_hard_translated_en_from_bn_from_0k_to_15k.csv
│   ├── docs/
│   │   └── bnmmlu_dataset_translation_summary.md
│   └── notebook/
│       ├── bangla_to_english_translation.ipynb
│       ├── bangla_to_english_translation.py
│       ├── data_exploration/
│       │   └── exploratory_data_analysis.py
│       └── dataset_translation/
│           ├── bangla_to_english_translation.ipynb
│           └── bangla_to_english_translation.py
│
└── tests/
    ├── test_s7_automated.py
    ├── test_s8_fragmentation.py
    ├── test_s9a_registry.py
    └── test_s9b_e3_block.py
```

Several scripts contain absolute local paths and must be parameterized before running in a clean checkout.

## 7. How to reproduce

1. Open the notebook on Kaggle (T4 GPU, about 15 GB VRAM, 9-hour session limit, about 20 GB disk).
2. Run the setup cells (environment, `MODEL_REGISTRY`, data loaders, inference backends, `BenchmarkEvaluator`, `run_one_model`).
3. Run **one model at a time**: dev screen first, then the full 1,500-item run only if the gates pass.
4. Package per-model results as `results_E3_<model>.zip` (or `results_E2_cyclic4_<model>`).
5. Run `analysis_v2.py` for metric tables and `analysis_report_e4.py` for the PDF report (`Analysis_Report_6models_plus_phi_pluss_e4.pdf`).
6. Run `python verify_parallel_subsets.py <results_dir> <wide_csv>` to check the translation-parallel subsets.

## 8. Known limitations

- Four tokenizer levels only (five with `phi35_mini`), so cross-model inference is limited and descriptive.
- Tokenizer family, model size, architecture and instruction tuning vary together with fragmentation.
- English is a translation; translation quality was not spot-checked.
- BnMMLU is evaluated on its train split.
- Cyclic4 changes the meaning of confidence, so raw and cyclic4 confidence are not directly comparable.
- Dev samples (n = 200) give noisy ECE and letter-share estimates.
- Single prompt configuration and a single seed.
- The dtype-sensitivity comparison (`dtype_sensitivity/dtype_sensitivity.csv`) has no data in the current run.
- Few subject clusters (22) make clustered standard errors approximate.