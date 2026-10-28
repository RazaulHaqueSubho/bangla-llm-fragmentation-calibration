# BnMMLU v3: full-dataset evaluation (Dipto Karmakar)

This folder has Dipto Karmakar's v3 run of the BnMMLU evaluation. It is separate from the root `bnmmlu_eval.py` run, which used a different prompt and a different set of models.

## What was done

Every valid row of the BnMMLU `full` config was scored by each local model. There is no sampling and no deduplication.

- Source rows: 134,382
- Valid rows scored per model: 134,375 (7 rows had an invalid schema and were skipped)
- Dataset: https://huggingface.co/datasets/samanjoy2/BnMMLU
- Seed: 20260916
- Inference is local on Kaggle (T4 GPU). No hosted API is used.

## Models

| Key | Model | Precision | Status |
|---|---|---|---|
| `qwen_1_5b` | Qwen/Qwen2.5-1.5B-Instruct | fp16 | done |
| `smol_1_7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | fp16 | done |
| `titu_3b` | hishab/titulm-llama-3.2-3b-v2.0 | fp16 | done |
| `tiger_1b` | md-nishat-008/TigerLLM-1B-it | fp32 | done |
| `llama3_3b` | meta-llama/Llama-3.2-3B-Instruct | fp16 | done |
| `mixtral_8x7b` | mistralai/Mixtral-8x7B-Instruct-v0.1 | nf4 | not run |

Mixtral was blocked by a resource check. It needs at least 26 GiB of free GPU memory and only 19.2 GiB was available, so it has no results.

## Method

1. Build a plain-completion prompt (no chat template): "Choose the correct answer. Reply with one letter: A, B, C, or D." followed by the question, the four options and `Answer:`.
2. Score the four answer labels (` A`, ` B`, ` C`, ` D`) with the model's log-probabilities and softmax over those four.
3. The highest probability is the predicted answer.
4. Calibration uses 5-fold cross-fitted temperature scaling. Questions with the same text always stay in the same fold, and each row's calibrated prediction uses a temperature fitted on the other folds.
5. Every row is saved right away to a SQLite checkpoint, so the run can resume across Kaggle sessions.

## Results

Raw results on 134,375 rows per model (coverage 99.995% of source rows).

| Model | Accuracy | Brier | NLL | ECE (15 bins) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 31.69% | 0.8903 | 1.7615 | 0.3002 |
| SmolLM2-1.7B-Instruct | 28.24% | 0.9387 | 1.8545 | 0.3279 |
| TituLLM-3B v2.0 | 26.65% | 0.8664 | 1.6819 | 0.2426 |
| TigerLLM-1B-it | 32.25% | 0.7559 | 1.4028 | 0.1108 |
| Llama-3.2-3B-Instruct | 38.69% | 0.7089 | 1.3185 | 0.0938 |

After cross-fitted temperature scaling (accuracy stays the same):

| Model | Brier | NLL | ECE (15 bins) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 0.7369 | 1.3628 | 0.0168 |
| SmolLM2-1.7B-Instruct | 0.7462 | 1.3788 | 0.0099 |
| TituLLM-3B v2.0 | 0.7493 | 1.3849 | 0.0011 |
| TigerLLM-1B-it | 0.7347 | 1.3581 | 0.0203 |
| Llama-3.2-3B-Instruct | 0.6934 | 1.2833 | 0.0184 |

## Findings

These were reported by Dipto from this run. The row-level files they come from are not in this repo.

1. Llama-3.2-3B is the best and most consistent model, with about 38.7% accuracy.
2. Several models pick one letter again and again: Tiger chose B on 66.40% of questions, Titu chose A on 66.19%, and Smol chose C on 58.71%.
3. Fewer tokens per word did not give better answers. Tiger used 3.06 tokens per whitespace-separated word and Titu used 3.14, but Llama used 8.56 and still scored the highest accuracy.
4. A low calibration error does not mean strong knowledge. Titu has the lowest calibrated ECE (0.0011) but only 26.65% accuracy.

## Files

| Path | What it is |
|---|---|
| `notebook/banglallm-v3-full-local.ipynb` | the Kaggle notebook that ran everything |
| `results/metrics_full_dataset.csv` | the metrics table (raw and calibrated) |
| `results/subject_metrics.csv`, `results/subject_macro_metrics.csv` | per-subject metrics |
| `results/unique_question_sensitivity.csv` | metrics on unique questions only |
| `results/crossfit_temperatures.csv` | temperature fitted for each fold |
| `results/common_rows_raw_comparison.csv` | comparison on the rows scored by all models |
| `results/model_progress.csv`, `results/run_events.csv`, `results/run_status.json` | run status and log |
| `results/data_audit.json`, `results/manifest.json` | dataset audit and protocol |
| `results/*_config.json`, `results/*_model_identity.json` | model and tokenizer details |

Not included because of size: the SQLite checkpoints (about 430-580 MB each) and the per-row prediction CSVs (about 70 MB each).

## Notes

- This is a descriptive full-dataset analysis, not a confirmatory test. The earlier smoke run had already seen some of the questions.
- Tiger runs in fp32 and the other models in fp16, which is a precision difference when comparing them.
- The plain prompt is a fixed format and not each model's native chat template.
- The Titu model here is the 3B version (`titulm-llama-3.2-3b-v2.0`). The root run used the 1B version.
