# Bangla LLM Fragmentation & Calibration

Evaluating small LLMs on Bangla multiple-choice questions (BnMMLU), looking at accuracy and how well calibrated the models are.

## What this repo does

Runs three models on the full BnMMLU dataset (134,382 questions) with the same pipeline:

| Key | Model |
|---|---|
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` |
| `titu` | `hishab/titulm-llama-3.2-1b-v2.0` |
| `tiger` | `md-nishat-008/TigerLLM-1B-it` |

Dataset: [`samanjoy2/BnMMLU`](https://huggingface.co/datasets/samanjoy2/BnMMLU)

## Method

1. Load the dataset and shuffle it with seed 42.
2. Build a prompt with the question and options A-D (chat template is used if the model has one).
3. Take the logits of the last token and compare only the tokens for `A`, `B`, `C`, `D`.
4. Softmax over those 4 logits gives the probability for each option.
5. Predicted answer = option with the highest probability.
6. Every question is saved to a `.jsonl` file right away, so the run can resume if Kaggle stops. A backup ZIP is updated every 5,000 questions.

No text is generated, the answer comes directly from the letter logits.

## Metrics

- **Accuracy**
- **Mean confidence** (average of the max probability)
- **Multiclass Brier score**
- **ECE** (expected calibration error, 10 bins)

## Files

| File | What it is |
|---|---|
| `bnmmlu_eval.py` | the evaluation script (runs on Kaggle) |
| `notebooks/` | Kaggle notebook(s) |
| `results/*_summary.json` | final metrics for each model |

Per-question results (`*_results.jsonl`) are big, so they are not in git.

## Output

For each model the script produces:

- `<model>_results.jsonl` : one line per question (`id`, `probs`, `pred`, `gold`)
- `<model>_full_backup.zip` : zipped copy of the results file
- `<model>_summary.json` : accuracy, mean confidence, Brier, ECE

## Results

| Model | Questions | Accuracy | Mean conf. | Brier | ECE |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 134,382 | TODO | TODO | TODO | TODO |
| TituLLM-1B v2.0 | 134,382 | TODO | TODO | TODO | TODO |
| TigerLLM-1B-it | 134,382 | TODO | TODO | TODO | TODO |

## How to run

1. Open a new Kaggle notebook, turn on GPU and Internet.
2. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`.
3. Paste `bnmmlu_eval.py` in a cell (set `LIMIT = 200` first to test).
4. Save Version -> Save & Run All for the full run.
5. Download the zip and summary files from the Output panel.

## Notes

- Dataset column names are auto-detected; if it fails, set them at the top of the script.
- Same prompt and same question order for all three models, so the numbers can be compared.
