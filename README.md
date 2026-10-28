# Bangla LLM Calibration

Research project investigating the relationship between tokenization fragmentation and confidence calibration in Bangla Large Language Models (LLMs). The analysis is non-causal: it examines associations between fragmentation metrics (e.g., tokens per word) and calibration errors (e.g., Expected Calibration Error, Brier Score) without claiming that fragmentation causes miscalibration.

## Research question

Is tokenization fragmentation in Bangla associated with confidence calibration errors in Bangla LLMs, relative to English?

## Project structure

```
bangla-llm-original/
├── README.md
├── .gitignore
├── Bangla-LLM-Calibration-Final.ipynb          # Final analysis notebook (root)
├── prompt_diagnostic_analysis.pdf               # Diagnostic report
├── code/
│   ├── Bangla-LLM-Calibration-Real-Data-v4a.ipynb   # Publication notebook v4a
│   ├── Bangla-LLM-Calibration-Real-Data-v4b.ipynb   # Publication notebook v4b
│   ├── Bangla-LLM-Calibration-Real-Data-v4b_backup_before_uniform.ipynb
│   ├── notebook_backup_before_step1.ipynb
│   ├── build_item_parallel.py
│   ├── candidate_tokenizer_audit.py
│   ├── make_v3.py
│   ├── make_v4a.py
│   ├── verify_nb.py
│   ├── tokenizer_audit.csv
│   ├── combined_results.csv
│   ├── bn_summary.json / en_summary.json
│   ├── bn_summary_tiger.json / bn_summary_titu.json
│   ├── en_summary_tiger.json / en_summary_titu.json
│   ├── fewshot_exemplars.json
│   ├── run_manifest.json
│   └── results/                                 # Model outputs and analysis artifacts
│       ├── llama32_3b/, tigerllm_1b/, titullm_1b/
│       ├── item_parallel/
│       ├── run_A_analysis/
│       ├── run_B/
│       ├── run_manifest.json
│       ├── statistical_analysis_report.json / .md
│       └── status.json
├── results/                                     # Baseline and extended evaluation outputs
│   ├── BnMMLU_*_baseline_*.csv / .jsonl / .json
│   ├── BnMMLU_*_extended_*.csv / .jsonl / .json
│   ├── bn_subset.csv / en_subset.csv
│   ├── sampled_question_ids.json
│   ├── fragmentation_vs_calibration.png
│   ├── reliability_diagram.png
│   ├── stratified_calibration_comparison.png
│   ├── statistical_analysis_report.json / .md
│   ├── run.log
│   ├── pilot_analysis/
│   ├── test_baseline/
│   ├── test_extended/
│   └── test_extended_analysis/
└── src/
    ├── data/bnmmlu_full_en.csv                  # Tracked BnMMLU English dataset
    └── notebook/bangla_to_english_translation.ipynb
    └── notebook/bangla_to_english_translation.py
```

## Retained notebooks

- `Bangla-LLM-Calibration-Final.ipynb` — final consolidated analysis notebook at the repository root.
- `code/Bangla-LLM-Calibration-Real-Data-v4a.ipynb` — publication candidate v4a.
- `code/Bangla-LLM-Calibration-Real-Data-v4b.ipynb` — publication candidate v4b.

Backup copies (`v4b_backup_before_uniform.ipynb`, `notebook_backup_before_step1.ipynb`) are retained for reference. No v4c/v4d notebooks exist in this project.

Obsolete notebook versions (v1, v2, v3*), temporary scripts, patch files, caches, logs, and browser artifacts have been removed. Older versions remain recoverable from `~/Bangla-LLM-Calibration`.

## Setup instructions

1. Clone the repository:
   ```
   git clone https://github.com/RazaulHaqueSubho/bangla-llm-fragmentation-calibration.git
   cd bangla-llm-fragmentation-calibration
   ```
2. Create a Python environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   (`requirements.txt` is not yet tracked; generate it from your environment if needed.)
3. Open the notebooks with Jupyter:
   ```
   jupyter notebook Bangla-LLM-Calibration-Final.ipynb
   ```

## Dataset and model information

- **Dataset:** BnMMLU (Bangla MMLU), tracked at `src/data/bnmmlu_full_en.csv`. Subsets (`bn_subset.csv`, `en_subset.csv`) and sampled question IDs are in `results/`.
- **Models evaluated:** Llama-3.2-3B-Instruct, TigerLLM-1B, TituLLM-1B. Per-model outputs are in `code/results/`.
- **Translation:** Bangla-to-English translation notebook/script at `src/notebook/`.

## Analysis plan and evaluation criteria

- **Metrics:** Tokens per Word (TPW), accuracy, Brier Score, Expected Calibration Error (ECE). STRR definition is to be verified.
- **Analysis:** Compare fragmentation and calibration between Bangla and English; assess whether fragmentation is associated with calibration errors.
- **Artifacts:** Statistical analysis reports, reliability diagrams, fragmentation-vs-calibration plots, and per-item results are stored in `results/` and `code/results/`.

## Results status

- **Exploratory results:** `results/pilot_analysis/`, `results/test_baseline/`, `results/test_extended/`, `results/test_extended_analysis/` — early pilot runs, not independently validated.
- **Validated results:** `results/statistical_analysis_report.*` and `code/results/statistical_analysis_report.*` — primary analysis outputs.
- **Incomplete experiments:** `code/results/run_A_analysis/` and `code/results/run_B/` contain partial or empty outputs (figures/stats_tables directories are empty).

No experiment claims are made here that are not supported by the artifacts in this repository.
