# Final Report: E3 Extension and E4 Screening

Project: Bangla LLM Fragmentation and Calibration
Thesis: *Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? A Cross-Lingual Study*

## 1. What was delivered

| Deliverable | File |
|---|---|
| E3 notebook (Section 14g: dev screen, collapse gate, majority-class test, full run, per-model ZIP) | `Bangla-LLM-Calibration-E2-rescue-v4i.ipynb` |
| E4 notebook | `Bangla-LLM-Calibration-E2-rescue-v4j.ipynb` |
| Preregistration amendment for the extension models | `PREREGISTRATION_AMENDMENT.md` |
| Fragmentation definition | `FRAGMENTATION_DEFINITION.md` |
| Analysis plan and gate thresholds | `analysis_plan.yaml`, `gate.yaml` |
| Fragmentation tables (7 and 11 models) | `tokenizer_fragmentation_per_item_all7.csv`, `tokenizer_fragmentation_per_item_all11.csv`, `tpw_strr_summary_all7.csv` |
| Dev-screen results | `dev_screen_report_e3.csv` |
| Analysis scripts | `analysis_v2.py`, `analysis_report_e4.py`, `run_s8_calculation.py` |
| Compiled results PDF | `Analysis_Report_6models_plus_phi_pluss_e4.pdf` |
| E4 pre-commitment and rule audit | `e4_added_models_plan.json`, `E4_CHECK_REPORT.md` |

## 2. Models and status

| Model | Role | Status |
|---|---|---|
| `titullm_1b`, `tigerllm_1b`, `llama32_3b` | main (E2 rescue, cyclic4) | full results, 1,500 items per language |
| `qwen25_3b`, `bloomz_3b`, `llama32_3b_base` | main (E3) | passed dev screen; full results, 1,500 items per language |
| `phi35_mini` | exploratory | dev majority-class gate not passed; full run done as exploratory second run |
| `mgpt_1_3b`, `olmo2_1b`, `smollm2_1_7b` | E4 | failed majority-class gate; dev only |
| `sarvam1_2b` | E4 | full run exists; invalid under rule R4 (BN option mass) |

Invariant settings for all main models: 4-shot, no chat template, `cyclic4`, seed 42, 1,500 items per language, dev set 200 per language, gate thresholds from `gate.yaml`.

## 3. Key results

- All 12 model-language cells in the main set are significantly above 25% chance.
- Cyclic4 top-letter shares are 0.268-0.313 (raw shares reached 0.784).
- BN accuracy ranges 30.5-40.7%; EN accuracy 39.8-63.4%.
- Best calibrated: `tigerllm_1b` (ECE 0.023 BN, 0.019 EN). Most overconfident: `qwen25_3b` (ECE 0.273 BN, 0.206 EN).
- BN mean TPW: about 7.0 (Llama family), 7.1 (`phi35_mini`), 6.2 (`qwen25_3b`), 2.3 (`tigerllm_1b`), 2.2 (`bloomz_3b`). EN mean TPW 1.6-2.0.
- Higher TPW is associated with lower confidence and correctness (13 of 24 item-level tests significant after Holm; pooled TPW_z coefficient -0.052 for correctness and -0.031 for confidence), most clearly in English.
- ECE by TPW quartile is not monotone.

Full tables are in `report.md`.

## 4. Statistical analysis stages

| Stage | Scope | Output |
|---|---|---|
| Core metrics | 6 models x 2 languages x 1,500 | accuracy, ECE (10 and 15 bins, equal-mass), Brier, NLL, overconfidence gap |
| Raw vs cyclic4 | 12 cells | `pred_raw`, `confidence_raw` vs cyclic4 outputs |
| BN vs EN | 6 models, paired on `question_id` | paired bootstrap, exact McNemar |
| Fragmentation vs calibration | 12 cells | quartiles, logistic and linear regressions, Spearman, squared-Brier proxy, Holm |
| Pooled sensitivity | 18,000 observations | TPW_z x language, model and subject fixed effects, clustered by `question_id` |
| Exploratory and extension | `phi35_mini`, E4 models | dev-only metrics or exploratory full-run metrics |

## 5. Kaggle execution order (v4i, one model per session)

1. **Setup (once per session):** environment detection, directory paths, HF cache, GPU diagnostics, `MODEL_REGISTRY` and base `CONFIG`, helpers, tokenization audit, prompt formatters, dataset loaders (`bn_subset.csv`, `en_subset.csv`, `dev_bn.csv`, `dev_en.csv`), inference backends, `BenchmarkEvaluator`, `run_one_model`, `run_e2_experiments`, `run_dev_probabilities`.
2. **E3 block (Section 14g):** defines `run_e3_extension_pipeline()` and the helper functions (dev screening, collapse gate, majority-class test, OOM-safe batching, shift checkpoints, per-model ZIP, cache cleanup).
3. **Run:** `run_e3_extension_pipeline(target_models=[<one model key>])`. Dev screening (200 per language) runs first; the full 3,000-item run launches only for models that pass both gates.
4. **Package:** `results_E3_<model>.zip` per model.
5. **Analyze:** `analysis_v2.py` and `analysis_report_e4.py`.

Constraints: T4 GPU, 9-hour session limit, about 20 GB disk. `llama32_3b_base` is gated and needs an `HF_TOKEN` secret.

## 6. Remaining limitations

1. Four tokenizer levels with calibration data (five with `phi35_mini`), so cross-model inference is descriptive.
2. Tokenizer family, model size, architecture and instruction tuning are confounded with fragmentation.
3. English is a translation; BnMMLU is evaluated on its train split.
4. Cyclic4 changes the meaning of confidence; raw and cyclic4 confidence are not directly comparable.
5. Dev sample size (200) makes dev-screen estimates noisy (for example `bloomz_3b` EN raw top-letter share 0.26 in dev vs 0.543 in the full run).
6. Dtype sensitivity (`dtype_sensitivity/dtype_sensitivity.csv`) has no data; single prompt format and single seed.
7. The v4i `run_manifest.json` is absent at the v4i root; E4 provenance is recorded separately.
8. All claims are associational, with no causal interpretation.