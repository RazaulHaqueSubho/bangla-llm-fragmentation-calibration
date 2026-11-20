# BnMMLU Dataset Translation: From Bengali to English

This codebase, specifically branch `data/4-question-pairs` contains the source
code and the translated dataset of the original dataset named
[BnMMLU](https://huggingface.co/datasets/samanjoy2/BnMMLU). The source code is
located on the `notebook/dataset_translation` folder where one Python and one
Jupyter Notebook is present. The Python script is mainly a marimo notebook which
is alternative to Jupyter Notebook, addressing the hidden state problem of
Jupyter Notebook.

## Repository Layout

```bash
.
├── README.md
├── pyproject.toml
└── src
    ├── data
    │   ├── full
    │   │   ├── bnmmlu_full_failed.csv
    │   │   ├── bnmmlu_full_translated_to_en_from_0k_to_70k_dataset.csv
    │   │   └── bnmmlu_full_translated_to_en_from_70k_to_134k_dataset.csv
    │   └── hard
    │       ├── bnmmlu_hard_failed.csv
    │       └── bnmmlu_hard_translated_en_from_bn_from_0k_to_15k.csv
    ├── docs
    │   └── bnmmlu_dataset_translation_from_bn_to_en.md
    └── notebook
        ├── data_exploration
        │   ├── __marimo__
        │   │   ├── exploratory_data_analysis.html
        │   │   └── session
        │   │       └── exploratory_data_analysis.py.json
        │   └── exploratory_data_analysis.py
        └── dataset_translation
            ├── bangla_to_english_translation.py
            └── bangla_to_english_translation.ipynb
```

## References

- [molab](https://molab.marimo.io)
- [marimo](https://marimo.io)
- [BnMMLU](https://huggingface.co/datasets/bangla-ai/bnmmlu)
- [https://huggingface.co/google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it)
