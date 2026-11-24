"""
Unit and regression tests for TASK S9a: Configuration and Pre-Run Checks for the Four New Models
"""

import json
import os
import sys
import numpy as np
import pytest

NOTEBOOK_PATH = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/Bangla-LLM-Calibration-E2-rescue-v4i.ipynb"

def _load_notebook_and_registry():
    assert os.path.exists(NOTEBOOK_PATH), f"Notebook not found at {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    src6 = "".join(nb["cells"][6]["source"])
    idx_dataclass = src6.find("@dataclass")
    loc = {"os": os, "Dict": dict, "Any": object}
    exec(compile(src6[:idx_dataclass], "<registry>", "exec"), loc, loc)
    reg = loc["MODEL_REGISTRY"]
    return nb, reg

def test_v4i_model_registry_contains_all_seven_models():
    _, reg = _load_notebook_and_registry()
    expected_models = {
        "llama32_3b", "titullm_1b", "tigerllm_1b",
        "phi35_mini", "qwen25_3b", "bloomz_3b", "llama32_3b_base"
    }
    assert set(reg.keys()) == expected_models
    assert len(reg) == 7

def test_three_existing_models_unaltered():
    _, reg = _load_notebook_and_registry()
    # Check llama32_3b
    assert reg["llama32_3b"]["hf_id"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert reg["llama32_3b"]["dtype"] == "float16"
    assert reg["llama32_3b"]["needs_hf_token"] is True

    # Check titullm_1b
    assert reg["titullm_1b"]["hf_id"] == "hishab/titulm-llama-3.2-1b-v1.1"
    assert reg["titullm_1b"]["dtype"] == "float16"
    assert reg["titullm_1b"]["prompt_variant"] == "answer_cue"

    # Check tigerllm_1b
    assert reg["tigerllm_1b"]["hf_id"] == "md-nishat-008/TigerLLM-1B-it"
    assert reg["tigerllm_1b"]["dtype"] == "float32"

def test_four_new_models_required_fields():
    _, reg = _load_notebook_and_registry()
    new_model_keys = ["phi35_mini", "qwen25_3b", "bloomz_3b", "llama32_3b_base"]

    expected_specs = {
        "phi35_mini": {
            "hf_id": "microsoft/Phi-3.5-mini-instruct",
            "tokenizer_family": "phi_family",
            "model_size": "3.82B",
            "gated_status": "ungated",
            "needs_hf_token": False,
        },
        "qwen25_3b": {
            "hf_id": "Qwen/Qwen2.5-3B-Instruct",
            "tokenizer_family": "qwen_family",
            "model_size": "3.09B",
            "gated_status": "ungated",
            "needs_hf_token": False,
        },
        "bloomz_3b": {
            "hf_id": "bigscience/bloomz-3b",
            "tokenizer_family": "bloom_family",
            "model_size": "3.00B",
            "gated_status": "ungated",
            "needs_hf_token": False,
        },
        "llama32_3b_base": {
            "hf_id": "meta-llama/Llama-3.2-3B",
            "tokenizer_family": "llama_family",
            "model_size": "3.21B",
            "gated_status": "gated",
            "needs_hf_token": True,
        },
    }

    for mkey in new_model_keys:
        info = reg[mkey]
        spec = expected_specs[mkey]

        # 1. Exact HF ID
        assert info["hf_id"] == spec["hf_id"]
        # 2. Tokenizer family
        assert info["tokenizer_family"] == spec["tokenizer_family"]
        # 3. Model size
        assert info["model_size"] == spec["model_size"]
        # 4. Dtype == float16
        assert info["dtype"] == "float16"
        # 5. Chat-template setting off
        assert info["use_chat_template"] is False
        # 6. Max sequence length
        assert info["max_seq_len"] == 2048
        # 7. Estimated GPU memory fits T4 (<= 15.3 GB)
        est_gb = float(info["estimated_gpu_memory"].split()[0])
        assert est_gb <= 15.3, f"{mkey} estimated memory exceeds T4 limit"
        # 8. Gated/ungated status
        assert info["gated_status"] == spec["gated_status"]
        # 9. HF_TOKEN requirement
        assert info["needs_hf_token"] == spec["needs_hf_token"]
        # 10. Model loading configuration
        cfg = info["model_loading_config"]
        assert isinstance(cfg, dict)
        assert cfg["torch_dtype"] == "float16"
        assert cfg["device_map"] == "auto"
        assert cfg["low_cpu_mem_usage"] is True
        assert cfg["trust_remote_code"] is True

def test_prerun_checks_finite_logits_detection():
    nb, _ = _load_notebook_and_registry()
    # Find Section 14f code cell
    cell14f = None
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "def verify_logits_finite" in src:
            cell14f = src
            break
    assert cell14f is not None, "Section 14f code cell not found in notebook"

    loc = {}
    exec(compile(cell14f, "<cell14f>", "exec"), loc, loc)
    verify_logits = loc["verify_logits_finite"]

    # Finite array passes
    assert verify_logits([0.1, -0.5, 3.2, 1.0], model_key="test") is True

    # NaN array fails loudly
    with pytest.raises(ValueError, match="Non-finite logits detected"):
        verify_logits([0.1, float("nan"), 3.2, 1.0], model_key="test")

    # Inf array fails loudly
    with pytest.raises(ValueError, match="Non-finite logits detected"):
        verify_logits([0.1, float("inf"), 3.2, 1.0], model_key="test")

def test_prerun_checks_smoke_prep():
    # Verify 5-item smoke test preparation on existing subset files
    bn_path = "/home/naeem/Downloads/Bangla-Calibration-Thesis/results/bn_subset.csv"
    en_path = "/home/naeem/Downloads/Bangla-Calibration-Thesis/results/en_subset.csv"

    import pandas as pd
    bn_df = pd.read_csv(bn_path)
    en_df = pd.read_csv(en_path)

    assert len(bn_df.head(5)) == 5
    assert len(en_df.head(5)) == 5

    for df, name in [(bn_df, "bn"), (en_df, "en")]:
        for idx, row in df.head(5).iterrows():
            assert str(row["question"]).strip() != ""
            ans = row.get("correct_answer", row.get("answer", ""))
            assert str(ans).strip() in ["A", "B", "C", "D"]
            opts = row.get("options")
            assert opts is not None
