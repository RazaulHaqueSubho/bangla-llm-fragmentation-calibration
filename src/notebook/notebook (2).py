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
    This notebook translates the Bengali BnMMLU multiple choice benchmark into English with a Gemma model and saves the results to CSV files. This first cell gathers every library the later cells rely on. The marimo module is imported as mo so that these explanatory text cells can be displayed. The os module is imported but no later cell uses it. The re module provides regular expressions, which parse_reply uses to strip markdown code fences from the model output. The csv module reads and writes the output spreadsheets in run_translation. The json module turns Python dictionaries into JSON text for the model prompt and turns the model's JSON reply back into Python objects. Path from pathlib gives run_translation a convenient way to handle file paths. load_dataset from the datasets library downloads the BnMMLU data from Hugging Face. login from huggingface_hub authenticates the session. AutoProcessor loads the tokenizer and chat template of the Gemma model, and AutoModelForMultimodalLM loads the model weights. Every name except os is returned so that the cells below can use it.
    """)
    return


@app.cell
def _():
    import os
    import re
    import csv
    import json
    import marimo as mo
    from pathlib import Path
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    return (
        AutoModelForMultimodalLM,
        AutoProcessor,
        Path,
        csv,
        json,
        load_dataset,
        login,
        mo,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Before anything is downloaded, this cell authenticates with the Hugging Face Hub. It calls the login function from the previous cell with an access token string, which lets the notebook fetch the dataset and the google/gemma-4-31B-it model under the account that owns the token. Nothing is returned because the only effect of this cell is a stored credential for the current session.
    """)
    return


@app.cell
def _(login):
    login(token="")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell defines system_prompt, the instruction text that tells the model how to behave. It is a template rather than a finished prompt: the word SUBJECT is a placeholder that build_messages will replace with the real subject name of each question, so the model is told to use the standard English terminology of that field, and the word NUM is a placeholder that will be replaced with the number of answer options. The prompt asks the model to translate the question and every option without solving or explaining anything, to keep the options in their original order without (a)/(b) labels, to keep numbers, formulas and units unchanged while writing Bengali digits as 0-9, and to reply with only one JSON object containing a question string and an options list of exactly NUM items. That strict output format is what makes the reply machine readable for parse_reply. The variable is returned so the message building cell can use it.
    """)
    return


@app.cell
def _():
    system_prompt = (
        "You are an expert Bengali-to-English translator for SUBJECT exam questions. "
        "Use the standard English terminology of SUBJECT. "
        "Translate the question and every option. Do not solve, explain or comment. "
        "Keep the options in the same order, with no (a)/(b) labels. "
        "Keep numbers, formulas and units unchanged; write Bengali digits as 0-9. "
        "Reply with ONLY one JSON object, no markdown, in exactly this format: "
        '{"question": "<English question>", "options": ["<option 1>", "<option 2>", ...]} '
        "The options list must have exactly NUM items."
    )
    return (system_prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell creates full_dataset, the complete BnMMLU collection. The load_dataset function downloads the full configuration of samanjoy2/BnMMLU and the train split is selected. The list comprehension uses the loop variable dataset_record to convert each record into an ordinary Python dictionary, so full_dataset ends up as a plain list of dictionaries. Each dictionary carries the fields used later in the notebook: Unique_Serial as the row identifier, subject_name, correct_answer, question and options, where question and options hold the Bengali text. This list is what run_translation iterates over for the full run.
    """)
    return


@app.cell
def _(load_dataset):
    full_dataset = [
        dict(dataset_record)
        for dataset_record in load_dataset("samanjoy2/BnMMLU", "full")["train"]
    ]
    return (full_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell creates hard_dataset in the same way as the previous cell, but it loads the hard configuration of samanjoy2/BnMMLU instead of the full one. The loop variable dataset_record converts each record into a plain dictionary, and the resulting list has the same fields as full_dataset: Unique_Serial, subject_name, correct_answer, question and options. Keeping it as a separate variable lets the notebook translate the hard set into its own CSV file.
    """)
    return


@app.cell
def _(load_dataset):
    hard_dataset = [
        dict(dataset_record)
        for dataset_record in load_dataset("samanjoy2/BnMMLU", "hard")["train"]
    ]
    return (hard_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell defines build_messages, a function that turns one dataset row into the chat messages the model will receive. The parameter dataset_row is a single dictionary from either dataset. Inside the function, translated_system_prompt is made from the system_prompt template by replacing SUBJECT with the subject_name of the row and NUM with the number of items in the row's options list, so every question gets instructions that match its own subject and option count. The variable user_payload is a JSON string holding the Bengali question and options, and ensure_ascii is set to False so the Bengali characters stay readable instead of being escaped. The function returns a list of two messages in chat format, a system message carrying translated_system_prompt and a user message carrying user_payload. The function and the json module are passed in from earlier cells and build_messages is returned for run_translation to use.
    """)
    return


@app.cell
def _(json, system_prompt):
    def build_messages(dataset_row):
        translated_system_prompt = system_prompt.replace(
            "SUBJECT", dataset_row["subject_name"]
        ).replace("NUM", str(len(dataset_row["options"])))
        user_payload = json.dumps(
            {
                "question": dataset_row["question"],
                "options": dataset_row["options"],
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": translated_system_prompt},
            {"role": "user", "content": user_payload},
        ]

    return (build_messages,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell defines parse_reply, a function that checks and unpacks the model's answer. The parameter model_reply is the raw text the model produced and expected_option_count is how many options the original question had. The variable reply_text is the reply with surrounding whitespace removed and with any markdown code fence at the start or end removed by a regular expression, because models sometimes wrap JSON in fences even when told not to. The variable parsed_reply is the result of json.loads on that text. The function then validates it step by step: parsed_reply must be a dictionary whose keys are exactly question and options, the question must be a non-empty string, and translated_options must be a list with exactly expected_option_count items, each a non-empty string, where option is the loop variable used in that check. If any check fails it raises a ValueError with a descriptive message, which run_translation catches and writes to its failure log. On success it returns the translated question and the list of translated options as two separate values.
    """)
    return


@app.cell
def _(json, re):
    def parse_reply(model_reply, expected_option_count):
        reply_text = re.sub(r"^```(?:json)?|```$", "", model_reply.strip()).strip()
        parsed_reply = json.loads(reply_text)
        if not isinstance(parsed_reply, dict) or set(parsed_reply) != {
            "question",
            "options",
        }:
            raise ValueError("keys must be exactly 'question' and 'options'")
        if (
            not isinstance(parsed_reply["question"], str)
            or not parsed_reply["question"].strip()
        ):
            raise ValueError("question must be a non-empty string")
        translated_options = parsed_reply["options"]
        if not (
            isinstance(translated_options, list)
            and len(translated_options) == expected_option_count
            and all(
                isinstance(option, str) and option.strip()
                for option in translated_options
            )
        ):
            raise ValueError(
                f"options must be a list of {expected_option_count} non-empty strings"
            )
        return parsed_reply["question"], translated_options

    return (parse_reply,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell defines run_translation, the function that drives the whole job and writes results to disk. The parameter dataset_config is a label such as full or hard that is only used in progress messages, dataset_rows is the list of questions to translate, generate_translation is the function that sends messages to the model and returns its text, and output_path is where the English CSV is written. The variable csv_output_path wraps output_path as a Path object. The variable header lists the column names of the CSV: the identifier, subject, correct answer, the English question and four English options, followed by the original Bengali question and four Bengali options. The variable completed_serials starts as an empty set and, if the output file already exists, is filled with the Unique_Serial values found in it, where file_handle is the open file and existing_row is each row read by csv.DictReader. This is what lets an interrupted run resume without redoing finished questions. The variable translation_error_log_path is a text file placed next to the CSV, named by removing the en suffix marker from the CSV name and adding failed.log, and it collects every failure. The variable new_file records whether the CSV is being created now, so the header row is written only once, and the file is opened in append mode with a byte order mark encoding on creation so spreadsheet programs read the Bengali text correctly. The variable writer is the csv writer for that file. The loop goes through dataset_rows with row_index as the position and dataset_row as the current question, and serial holds that question's Unique_Serial, which is skipped if it is already in completed_serials. The small function pad_options takes option_list and pads it with empty strings up to four items, so every row has the same number of columns even when a question has fewer than four options. The variable raw_translation holds the model's text and is set to an empty string first so the error log has something to print if generation itself fails. Inside the try block the function calls generate_translation with the output of build_messages and then parse_reply, which produces translated_question and translated_options. If anything goes wrong, translation_error is the caught exception, log is the opened failure log where the serial, the error and the first 300 characters of the raw reply are recorded, a FAILED line is printed, and the loop moves on, leaving that serial unfinished so a later run will retry it. On success it writes one row combining the identifier, subject, correct answer, English text and original Bengali text, flushes the file right away so a crash loses nothing, and prints an ok line with the progress count. The csv module and Path, build_messages and parse_reply come from earlier cells and run_translation is returned for the final cells.
    """)
    return


@app.cell
def _(Path, build_messages, csv, parse_reply):
    def run_translation(
        dataset_config, dataset_rows, generate_translation, output_path
    ):
        csv_output_path = Path(output_path)
        header = [
            "Unique_Serial",
            "subject_name",
            "correct_answer",
            "question_en",
            "option_a_en",
            "option_b_en",
            "option_c_en",
            "option_d_en",
            "question_bn",
            "option_a_bn",
            "option_b_bn",
            "option_c_bn",
            "option_d_bn",
        ]

        completed_serials = set()
        if csv_output_path.exists():
            with csv_output_path.open(encoding="utf-8-sig", newline="") as file_handle:
                completed_serials = {
                    existing_row["Unique_Serial"]
                    for existing_row in csv.DictReader(file_handle)
                }

        translation_error_log_path = csv_output_path.with_name(
            csv_output_path.stem.replace("_en", "") + "_failed.log"
        )
        new_file = not csv_output_path.exists()
        with csv_output_path.open(
            "a", encoding="utf-8-sig" if new_file else "utf-8", newline=""
        ) as file_handle:
            writer = csv.writer(file_handle)
            if new_file:
                writer.writerow(header)
            for row_index, dataset_row in enumerate(dataset_rows):
                serial = dataset_row["Unique_Serial"]
                if str(serial) in completed_serials:
                    continue
                pad_options = lambda option_list: (
                    list(option_list) + [""] * (4 - len(option_list))
                )
                raw_translation = ""
                try:
                    raw_translation = generate_translation(build_messages(dataset_row))
                    translated_question, translated_options = parse_reply(
                        raw_translation, len(dataset_row["options"])
                    )
                except Exception as translation_error:
                    with translation_error_log_path.open("a", encoding="utf-8") as log:
                        log.write(
                            f"serial={serial} error={translation_error} raw={raw_translation[:300]!r}\n"
                        )
                    print(
                        f"[{dataset_config}] {row_index + 1}/{len(dataset_rows)} serial={serial} FAILED (see {translation_error_log_path.name})"
                    )
                    continue
                writer.writerow(
                    [
                        serial,
                        dataset_row["subject_name"],
                        dataset_row["correct_answer"],
                        translated_question,
                        *pad_options(translated_options),
                        dataset_row["question"],
                        *pad_options(dataset_row["options"]),
                    ]
                )
                file_handle.flush()  # saved immediately, so a crash loses nothing
                print(
                    f"[{dataset_config}] {row_index + 1}/{len(dataset_rows)} serial={serial} ok"
                )

    return (run_translation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell loads the translation model and defines generate_translation. The variable processor is the AutoProcessor for google/gemma-4-31B-it and handles tokenization and the chat template. The variable model is the same model loaded with AutoModelForMultimodalLM, with dtype set to auto so it uses the precision stored in the checkpoint and device_map set to auto so the weights are spread across the available GPUs. The function generate_translation takes translation_messages, the list produced by build_messages, and returns the model's reply as text. Inside it, model_inputs is the result of applying the chat template to those messages: it tokenizes them, returns tensors, adds the generation prompt so the model knows to answer, turns thinking mode off so the output is only the JSON reply, and moves everything to the device the model lives on. The variable input_token_count is the length of the prompt in tokens. The variable generated_tokens is the output of model.generate, which uses greedy decoding by setting do_sample to False so the same input always gives the same translation, and allows up to 1024 new tokens. The function slices off the prompt tokens using input_token_count and decodes only the newly generated tokens, skipping special tokens, so what comes back is just the answer. The function generate_translation is returned to be passed into run_translation.
    """)
    return


@app.cell
def _(AutoModelForMultimodalLM, AutoProcessor):
    processor = AutoProcessor.from_pretrained("google/gemma-4-31B-it")
    model = AutoModelForMultimodalLM.from_pretrained(
        "google/gemma-4-31B-it", dtype="auto", device_map="auto"
    )

    def generate_translation(translation_messages):
        model_inputs = processor.apply_chat_template(
            translation_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(model.device)
        input_token_count = model_inputs["input_ids"].shape[-1]
        generated_tokens = model.generate(
            **model_inputs, max_new_tokens=1024, do_sample=False
        )
        return processor.decode(
            generated_tokens[0][input_token_count:], skip_special_tokens=True
        )

    return (generate_translation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell is the one that actually starts the work on the full dataset. It calls run_translation with the label full, the full_dataset list of questions, the generate_translation function that talks to the Gemma model, and the file name bnmmlu_full_en.csv as the output. Rows are translated one at a time and appended to that CSV, a failed log is written beside it for any rows that could not be parsed, and rerunning the cell skips every serial already saved. Nothing is returned because the result is the file on disk.
    """)
    return


@app.cell
def _(full_dataset, generate_translation, run_translation):
    # This is the cell that actually starts the translation.
    run_translation("full", full_dataset, generate_translation, "bnmmlu_full_en.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell repeats the process for the hard dataset. It calls run_translation with the label hard, the hard_dataset list, the same generate_translation function and the output file bnmmlu_hard_en.csv, so the hard questions get their own English CSV and their own failed log. The model is already loaded, so this cell reuses it, and as before a rerun resumes from the serials already saved. Nothing is returned because the result is the file on disk.
    """)
    return


@app.cell
def _(generate_translation, hard_dataset, run_translation):
    run_translation("hard", hard_dataset, generate_translation, "bnmmlu_hard_en.csv")
    return


if __name__ == "__main__":
    app.run()
