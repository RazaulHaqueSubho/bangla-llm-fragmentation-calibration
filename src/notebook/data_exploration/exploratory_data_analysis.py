# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "accelerate==1.15.0",
#     "huggingface-hub==1.32.0",
#     "polar==0.0.127",
#     "tokenmonster==1.1.12",
#     "torch==2.14.0",
#     "transformers==5.17.0",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook translates the Bengali BnMMLU multiple choice benchmark into English with a Gemma model and saves the results to CSV files.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return (pd,)


@app.cell
def _(pd):
    bnmmlu_full_failed = pd.read_csv("/home/abdullah_al_azmi/bangla-llm-fragmentation-calibration/src/data/full/bnmmlu_full_failed.csv")
    return (bnmmlu_full_failed,)


@app.cell
def _(pd):
    bnmmlu_translated_to_en_from_0k_to_70k_dataset = pd.read_csv("/home/abdullah_al_azmi/bangla-llm-fragmentation-calibration/src/data/full/bnmmlu_translated_to_en_from_0k_to_70k_dataset.csv")
    return (bnmmlu_translated_to_en_from_0k_to_70k_dataset,)


@app.cell
def _(pd):
    bnmmlu_translated_to_en_from_70k_to_134k_dataset= pd.read_csv("/home/abdullah_al_azmi/bangla-llm-fragmentation-calibration/src/data/full/bnmmlu_translated_to_en_from_70k_to_134k_dataset.csv")
    return (bnmmlu_translated_to_en_from_70k_to_134k_dataset,)


@app.cell
def _(bnmmlu_full_failed):
    bnmmlu_full_failed
    return


@app.cell
def _(bnmmlu_translated_to_en_from_0k_to_70k_dataset):
    bnmmlu_translated_to_en_from_0k_to_70k_dataset
    return


@app.cell
def _(bnmmlu_translated_to_en_from_70k_to_134k_dataset):
    bnmmlu_translated_to_en_from_70k_to_134k_dataset
    return


if __name__ == "__main__":
    app.run()
