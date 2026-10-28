# BnMMLU Dataset Translation: From Bengali to English

This codebase, specifically branch `data/4-question-pairs` contains the source code and the translated dataset. The source code is located on the `notebook/dataset_translation` folder where one Python and one Jupyter Notebook is present. The Python script is mainly a marimo notebook which is alternative to Jupyter Notebook, addressing the hidden state problem of Jupyter Notebook.

## Repository Layout

```bash

bangla-llm-fragmentation-calibration
├── README.md
├── pyproject.toml
└── src
    ├── data
    │   └── full
    │       ├── bnmmlu_full_failed.csv                                  # contains the dataset [full] which was failed to translate
    │       ├── bnmmlu_translated_to_en_from_0k_to_70k_dataset.csv      # contains the dataset [full] which successfully translated from 0k to 70k
    │       └── bnmmlu_translated_to_en_from_70k_to_134k_dataset.csv    # contains the dataset [full] which successfully translated from 70k to 134k
    ├── docs
    │   └── bnmmlu_dataset_translation_from_bn_to_en.md                 # we are here! in the markdown file!
    └── notebook
        ├── data_exploration
        │   ├── __marimo__
        │   │   ├── exploratory_data_analysis.html                      # here, you can see the dataset in dataframe embeeded in html language
        │   │   └── session
        │   │       └── exploratory_data_analysis.py.json
        │   └── exploratory_data_analysis.py                            # marimo notebook to explore the translated and failed dataset
        └── dataset_translation
            ├── bangla_to_english_translation.ipynb                     # an exported copy of marimo notebook on jupyter notebook
            └── bangla_to_english_translation.py                        # marimo notebook that enables this translation or source pipeline

```
