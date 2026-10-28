# Bangla LLM Fragmentation & Calibration

In this project I test how well small LLMs answer Bangla multiple choice questions, and also how much they can trust their own answers (calibration).

## What I did

I use the BnMMLU dataset (134,382 questions, 4 options each) and run each model on the full dataset.

| Key | Model | Status |
|---|---|---|
| `llama` | `meta-llama/Llama-3.2-3B-Instruct` | done (full 134,382) |
| `titu` | `hishab/titulm-llama-3.2-1b-v2.0` | in progress |
| `tiger` | `md-nishat-008/TigerLLM-1B-it` | not run yet |

Dataset: https://huggingface.co/datasets/samanjoy2/BnMMLU

## Method

1. Load the dataset and shuffle it with seed 42.
2. Build a prompt with the question and options A-D (chat template is used if the model has one).
3. Take the logits of the last token and compare only the tokens for A, B, C, D.
4. Softmax over those 4 values gives the probability for each option.
5. The option with the highest probability is the predicted answer.
6. Every question is saved to a `.jsonl` file right away, so the run can resume if Kaggle stops. A backup zip is updated every 5,000 questions.

No text is generated, the answer comes directly from the letter logits.

## Metrics

- **Accuracy**
- **Mean confidence:** average of the highest probability the model gave
- **Multiclass Brier score:** lower is better (0.75 is what you get if you give 25% to every option)
- **ECE:** expected calibration error, 10 bins, lower is better

## Results

| Model | Questions | Accuracy | Mean conf. | Brier | ECE |
|---|---|---|---|---|---|
| Llama-3.2-3B-Instruct | 134,382 | 38.96% | 0.5889 | 0.7623 | 0.1993 |
| TituLLM-1B v2.0 | 134,382 | pending | pending | pending | pending |
| TigerLLM-1B-it | 134,382 | pending | pending | pending | pending |

### Llama-3.2-3B-Instruct

- Accuracy is 38.96% but mean confidence is 58.89%, so the model is about 20 points overconfident.
- ECE is 0.1993, which is almost the same as this gap. So the overconfidence is there in nearly all confidence bins, not only a few.
- Brier score is 0.7623, which is worse than the 0.75 baseline of giving every option 25%.
- Predictions by letter: A 35,961, B 44,507, C 32,874, D 21,040. The model picks B the most and D the least.

**Finding:** with 4-option scoring on all 134,382 BnMMLU questions, Llama-3.2-3B-Instruct gets 38.96% accuracy with 58.89% mean confidence, which shows aggregate overconfidence.

## Output

For each model the script produces:

- `<model>_results.jsonl`: one line per question (`id`, `probs`, `pred`, `gold`)
- `<model>_full_backup.zip`: zipped copy of the results file
- `<model>_summary.json`: accuracy, mean confidence, Brier, ECE

The jsonl and zip files are big, so they are not in git. Only the summary files are in `results/`.

## Files

| File | What it is |
|---|---|
| `bnmmlu_eval.py` | the evaluation script (runs on Kaggle) |
| `notebooks/` | Kaggle notebook(s) |
| `results/` | summary json for each model |

## How to run

1. Open a new Kaggle notebook, turn on GPU and Internet.
2. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`.
3. Paste `bnmmlu_eval.py` in a cell (set `LIMIT = 200` first to test).
4. Set `LIMIT = None` and do Save Version -> Save & Run All for the full run.
5. Download the zip and summary files from the Output panel.

## Notes

- The Llama numbers come from my first run with an earlier version of the code. The Titu and Tiger runs use `bnmmlu_eval.py`, so the prompt may be a bit different. For a fully fair comparison I will rerun Llama with the same script.
- An earlier Titu test used v1.1 by mistake. The results here are for v2.0 only.
- Dataset column names are auto-detected; if it fails, set them at the top of the script.
