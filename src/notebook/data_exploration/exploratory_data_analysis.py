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


if __name__ == "__main__":
    app.run()
