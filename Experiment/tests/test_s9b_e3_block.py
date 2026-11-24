"""
Unit and regression tests for TASK S9b: E3 Evaluation Block and Preregistration Amendment
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import pytest

NOTEBOOK_PATH = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/Bangla-LLM-Calibration-E2-rescue-v4i.ipynb"
AMENDMENT_PATH = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/PREREGISTRATION_AMENDMENT.md"

def _load_notebook_and_cell50():
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook not found at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell50_src = None
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "Section 14g: E3 Evaluation Block for Approved New Models (S9b)" in src and c.get("cell_type") == "code":
            cell50_src = src
            break
    assert cell50_src is not None, "Section 14g code cell not found in notebook"
    return nb, cell50_src

def test_e3_block_exists_and_compiles():
    _, src50 = _load_notebook_and_cell50()
    compile(src50, "<cell50>", "exec")

    # Execute in local namespace with dummy CONFIG
    class DummyConfig:
        output_dir = "results"
        seed = 42
        backend = "mock"
    loc = {"CONFIG": DummyConfig(), "MODEL_REGISTRY": {}, "logger": None}
    exec(compile(src50, "<cell50>", "exec"), loc, loc)

    assert "TARGET_E3_MODELS" in loc
    assert "E3_UNIFORM_SETTINGS" in loc
    assert "test_majority_class_baseline" in loc
    assert "run_e3_pipeline_for_model" in loc
    assert "run_e3_extension_pipeline" in loc

def test_e3_models_and_settings_invariance():
    _, src50 = _load_notebook_and_cell50()
    loc = {"CONFIG": None, "MODEL_REGISTRY": {}, "logger": None}
    exec(compile(src50, "<cell50>", "exec"), loc, loc)

    target_models = loc["TARGET_E3_MODELS"]
    base_models = loc["_E3_BASE_MODELS"]
    expected_models = ["qwen25_3b", "bloomz_3b", "phi35_mini", "llama32_3b_base"]
    assert base_models == expected_models, "registry order must be qwen25_3b, bloomz_3b, phi35_mini, llama32_3b_base"
    assert "llama32_3b" not in target_models, "Untouched control model must NOT be in target E3 models"

    settings = loc["E3_UNIFORM_SETTINGS"]
    assert settings["prompt_variant"] == "few_shot"
    assert settings["n_shots"] == 4
    assert settings["use_chat_template"] is False
    assert settings["debias"] == "cyclic4"
    assert settings["seed"] == 42
    assert settings["n_samples"] == 1500
    assert settings["dev_n"] == 200

def test_majority_class_baseline_logic():
    _, src50 = _load_notebook_and_cell50()
    loc = {"CONFIG": None, "MODEL_REGISTRY": {}, "logger": None}
    exec(compile(src50, "<cell50>", "exec"), loc, loc)
    test_maj = loc["test_majority_class_baseline"]

    # 1. High accuracy passing test: 80/200 = 40% vs 27.53%
    df_pass = pd.DataFrame({
        "predicted_answer": ["A"] * 80 + ["B"] * 120,
        "gold_answer": ["A"] * 80 + ["C"] * 120,
    })
    passed, acc, p_val, k, n = test_maj(df_pass, p_majority=0.275333, alpha=0.05)
    assert passed is True
    assert abs(acc - 0.40) < 1e-4
    assert p_val < 0.05
    assert k == 80 and n == 200

    # 2. Chance-level failing test: 50/200 = 25% vs 27.53%
    df_fail = pd.DataFrame({
        "predicted_answer": ["A"] * 50 + ["B"] * 150,
        "gold_answer": ["A"] * 50 + ["C"] * 150,
    })
    passed_f, acc_f, p_val_f, k_f, n_f = test_maj(df_fail, p_majority=0.275333, alpha=0.05)
    assert passed_f is False
    assert abs(acc_f - 0.25) < 1e-4
    assert p_val_f > 0.05
    assert k_f == 50 and n_f == 200

def test_throughput_and_runtime_estimation_formula():
    t_dev_elapsed = 40.0 # seconds for 400 items
    n_dev_total = 400
    throughput = n_dev_total / t_dev_elapsed # 10 items/sec
    assert throughput == 10.0

    # 3,000 items in full run
    est_full_time_sec = 3000.0 / throughput # 300 sec = 5 minutes
    est_full_time_min = est_full_time_sec / 60.0
    assert est_full_time_min == 5.0

def test_preregistration_amendment_document():
    assert os.path.exists(AMENDMENT_PATH), f"Missing {AMENDMENT_PATH}"
    with open(AMENDMENT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Required sections
    assert "Added Models" in text or "Approved Models" in text
    assert "phi35_mini" in text
    assert "qwen25_3b" in text
    assert "bloomz_3b" in text
    assert "llama32_3b_base" in text
    assert "Selection Criteria" in text
    assert "Experimental Settings" in text
    assert "llama32_3b" in text
    assert "untouched" in text.lower() or "control" in text.lower()
    assert "cyclic4" in text
    assert "4-shot" in text or "n_shots: 4" in text or "n_shots=4" in text
    assert "Date:" in text
