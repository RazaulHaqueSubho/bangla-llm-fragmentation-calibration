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


@app.cell
def _(login):
    login(token="")
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


@app.cell
def _(load_dataset):
    full_dataset = [
        dict(dataset_record)
        for dataset_record in load_dataset("samanjoy2/BnMMLU", "full")["train"]
    ]
    return (full_dataset,)


@app.cell
def _(load_dataset):
    hard_dataset = [
        dict(dataset_record)
        for dataset_record in load_dataset("samanjoy2/BnMMLU", "hard")["train"]
    ]
    return (hard_dataset,)


@app.cell
def _(json, system_prompt):
    def build_messages(dataset_row):
        translated_system_prompt = system_prompt.replace(
            "SUBJECT", dataset_row["subject_name"]
        ).replace("NUM", str(len(dataset_row["options"])))
        user_payload = json.dumps(
            {"question": dataset_row["question"], "options": dataset_row["options"]},
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": translated_system_prompt},
            {"role": "user", "content": user_payload},
        ]

    return (build_messages,)


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
                pad_options = lambda option_list: list(option_list) + [""] * (
                    4 - len(option_list)
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


@app.cell
def _(full_dataset, generate_translation, run_translation):
    # This is the cell that actually starts the translation.
    run_translation("full", full_dataset, generate_translation, "bnmmlu_full_en.csv")
    return


@app.cell
def _(generate_translation, hard_dataset, run_translation):
    run_translation("hard", hard_dataset, generate_translation, "bnmmlu_hard_en.csv")
    return


if __name__ == "__main__":
    app.run()
