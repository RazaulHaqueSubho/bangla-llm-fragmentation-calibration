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
    mo.md(
        r"""This notebook translates the Bengali BnMMLU multiple choice benchmark into English with a Gemma model and saves the results to CSV files."""
    )
    return


@app.cell
def _():
    import csv
    import json
    import os
    import re
    from pathlib import Path

    import marimo as mo
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
    def translation_paths(output_path):
        csv_output_path = Path(output_path)

        checkpoint_path = csv_output_path.with_name(
            csv_output_path.stem + ".checkpoint"
        )

        failed_csv_path = csv_output_path.with_name(
            csv_output_path.stem.replace("_en", "") + "_failed.csv"
        )

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

        failed_header = [
            "Unique_Serial",
            "subject_name",
            "correct_answer",
            "question_bn",
            "option_a_bn",
            "option_b_bn",
            "option_c_bn",
            "option_d_bn",
        ]
        return csv_output_path, checkpoint_path, failed_csv_path, header, failed_header

    def pad_options(option_list):
        return list(option_list) + [""] * (4 - len(option_list))

    def determine_start_index(
        dataset_config,
        dataset_rows,
        csv_output_path,
        checkpoint_path,
        failed_csv_path,
        failed_header,
    ):
        start_index = 0

        if checkpoint_path.exists():
            try:
                start_index = int(checkpoint_path.read_text().strip()) + 1
                print(
                    f"[{dataset_config}] Resuming from dataset index {start_index} (checkpoint)"
                )
            except ValueError:
                pass
        elif csv_output_path.exists():
            print(
                f"[{dataset_config}] No checkpoint found; scanning output CSV to find resume point..."
            )

            completed_serials = set()
            last_serial = None

            with csv_output_path.open(encoding="utf-8-sig", newline="") as fh:
                for existing_row in csv.DictReader(fh):
                    completed_serials.add(str(existing_row["Unique_Serial"]))
                    last_serial = existing_row["Unique_Serial"]

            if last_serial is not None:
                serial_to_idx = {
                    str(row["Unique_Serial"]): i for i, row in enumerate(dataset_rows)
                }
                last_idx = serial_to_idx.get(str(last_serial), -1)
                start_index = last_idx + 1
                checkpoint_path.write_text(str(last_idx))

                missed = [
                    dataset_rows[i]
                    for i in range(last_idx + 1)
                    if str(dataset_rows[i]["Unique_Serial"]) not in completed_serials
                ]

                if missed and not failed_csv_path.exists():
                    with failed_csv_path.open(
                        "w", encoding="utf-8-sig", newline=""
                    ) as fh:
                        writer = csv.writer(fh)
                        writer.writerow(failed_header)
                        for row in missed:
                            writer.writerow(
                                [
                                    row["Unique_Serial"],
                                    row["subject_name"],
                                    row["correct_answer"],
                                    row["question"],
                                    *pad_options(row["options"]),
                                ]
                            )
                    print(
                        f"[{dataset_config}] Captured {len(missed)} previously-failed rows → {failed_csv_path.name}"
                    )
                print(f"[{dataset_config}] Resuming from dataset index {start_index}")
        return start_index

    def write_translation_row(
        writer, dataset_row, translated_question, translated_options
    ):
        writer.writerow(
            [
                dataset_row["Unique_Serial"],
                dataset_row["subject_name"],
                dataset_row["correct_answer"],
                translated_question,
                *pad_options(translated_options),
                dataset_row["question"],
                *pad_options(dataset_row["options"]),
            ]
        )

    def write_failed_row(writer, dataset_row):
        writer.writerow(
            [
                dataset_row["Unique_Serial"],
                dataset_row["subject_name"],
                dataset_row["correct_answer"],
                dataset_row["question"],
                *pad_options(dataset_row["options"]),
            ]
        )

    def process_main_pass(
        dataset_config,
        dataset_rows,
        start_index,
        generate_translation,
        build_messages,
        parse_reply,
        csv_output_path,
        failed_csv_path,
        checkpoint_path,
        header,
        failed_header,
    ):
        output_is_new = not csv_output_path.exists()
        failed_is_new = not failed_csv_path.exists()

        with (
            csv_output_path.open(
                "a", encoding="utf-8-sig" if output_is_new else "utf-8", newline=""
            ) as out_fh,
            failed_csv_path.open(
                "a", encoding="utf-8-sig" if failed_is_new else "utf-8", newline=""
            ) as fail_fh,
        ):
            writer, fail_writer = csv.writer(out_fh), csv.writer(fail_fh)
            if output_is_new:
                writer.writerow(header)
            if failed_is_new:
                fail_writer.writerow(failed_header)

            for row_index in range(start_index, len(dataset_rows)):
                dataset_row = dataset_rows[row_index]
                serial = dataset_row["Unique_Serial"]
                try:
                    raw_translation = generate_translation(build_messages(dataset_row))

                    translated_question, translated_options = parse_reply(
                        raw_translation, len(dataset_row["options"])
                    )

                    write_translation_row(
                        writer, dataset_row, translated_question, translated_options
                    )

                    out_fh.flush()

                    print(
                        f"[{dataset_config}] {row_index + 1}/{len(dataset_rows)} serial={serial} ok"
                    )

                except Exception as translation_error:
                    write_failed_row(fail_writer, dataset_row)

                    fail_fh.flush()

                    print(
                        f"[{dataset_config}] {row_index + 1}/{len(dataset_rows)} serial={serial} FAILED: {translation_error}"
                    )

                checkpoint_path.write_text(str(row_index))

    def reconstruct_failed_row(failed_row):
        options_bn = [
            failed_row[k]
            for k in ("option_a_bn", "option_b_bn", "option_c_bn", "option_d_bn")
            if failed_row.get(k)
        ]
        return {
            "Unique_Serial": failed_row["Unique_Serial"],
            "subject_name": failed_row["subject_name"],
            "correct_answer": failed_row["correct_answer"],
            "question": failed_row["question_bn"],
            "options": options_bn,
        }

    def process_retry_pass(
        dataset_config,
        failed_rows,
        generate_translation,
        build_messages,
        parse_reply,
        csv_output_path,
    ):
        print(f"[{dataset_config}] Retrying {len(failed_rows)} failed rows...")

        still_failed = []

        with csv_output_path.open("a", encoding="utf-8", newline="") as out_fh:
            writer = csv.writer(out_fh)
            for failed_row in failed_rows:
                serial = failed_row["Unique_Serial"]
                retry_dataset_row = reconstruct_failed_row(failed_row)
                try:
                    raw_translation = generate_translation(
                        build_messages(retry_dataset_row)
                    )
                    translated_question, translated_options = parse_reply(
                        raw_translation, len(retry_dataset_row["options"])
                    )
                    write_translation_row(
                        writer,
                        retry_dataset_row,
                        translated_question,
                        translated_options,
                    )
                    out_fh.flush()
                    print(f"[{dataset_config}] retry serial={serial} ok")
                except Exception as retry_error:
                    still_failed.append(failed_row)
                    print(
                        f"[{dataset_config}] retry serial={serial} FAILED again: {retry_error}"
                    )
        return still_failed

    def cleanup_failed_file(
        dataset_config, failed_csv_path, failed_header, still_failed
    ):
        with failed_csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(failed_header)
            for row in still_failed:
                writer.writerow(
                    [row.get(header_name, "") for header_name in failed_header]
                )
        if not still_failed:
            failed_csv_path.unlink()
            print(f"[{dataset_config}] All failed rows retried successfully!")
        else:
            print(
                f"[{dataset_config}] {len(still_failed)} rows still failed after retry."
            )

    def run_translation(
        dataset_config, dataset_rows, generate_translation, output_path
    ):
        csv_output_path, checkpoint_path, failed_csv_path, header, failed_header = (
            translation_paths(output_path)
        )
        start_index = determine_start_index(
            dataset_config,
            dataset_rows,
            csv_output_path,
            checkpoint_path,
            failed_csv_path,
            failed_header,
        )

        process_main_pass(
            dataset_config,
            dataset_rows,
            start_index,
            generate_translation,
            build_messages,
            parse_reply,
            csv_output_path,
            failed_csv_path,
            checkpoint_path,
            header,
            failed_header,
        )

        if not failed_csv_path.exists():
            return

        with failed_csv_path.open(encoding="utf-8-sig", newline="") as fh:
            failed_rows = list(csv.DictReader(fh))

        if not failed_rows:
            return

        still_failed = process_retry_pass(
            dataset_config,
            failed_rows,
            generate_translation,
            build_messages,
            parse_reply,
            csv_output_path,
        )

        cleanup_failed_file(
            dataset_config, failed_csv_path, failed_header, still_failed
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
    run_translation("full", full_dataset, generate_translation, "bnmmlu_full_en.csv")
    return


@app.cell
def _(generate_translation, hard_dataset, run_translation):
    run_translation("hard", hard_dataset, generate_translation, "bnmmlu_hard_en.csv")
    return


if __name__ == "__main__":
    app.run()
