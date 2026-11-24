"""
Validation tests for TASK S8: All-Seven Tokenizer Fragmentation
"""

import os
import pandas as pd
import numpy as np
import pytest

def test_s8_file_existence():
    tables_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tables"
    fig_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/figures"
    e2_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run"

    # CSV files
    assert os.path.exists(os.path.join(tables_dir, "tokenizer_fragmentation_per_item_all7.csv"))
    assert os.path.exists(os.path.join(tables_dir, "tpw_strr_summary_all7.csv"))
    assert os.path.exists(os.path.join(tables_dir, "tpw_strr_bn_en_ratio.csv"))
    assert os.path.exists(os.path.join(e2_dir, "FRAGMENTATION_DEFINITION.md"))

    # Figures
    assert os.path.exists(os.path.join(fig_dir, "tpw_comparison_by_model_and_language.png"))
    assert os.path.exists(os.path.join(fig_dir, "strr_comparison_by_model_and_language.png"))
    assert os.path.exists(os.path.join(fig_dir, "bangla_tpw_ranking.png"))

def test_s8_sample_sizes_and_no_duplicates():
    tables_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tables"
    df_per_item = pd.read_csv(os.path.join(tables_dir, "tokenizer_fragmentation_per_item_all7.csv"))

    # Exactly 7 models
    models = set(df_per_item["model"].unique())
    expected_models = {
        "llama32_3b", "titullm_1b", "tigerllm_1b",
        "phi35_mini", "qwen25_3b", "bloomz_3b", "llama32_3b_base"
    }
    assert models == expected_models

    # Exactly 2 languages
    assert set(df_per_item["lang"].unique()) == {"bn", "en"}

    # Exactly 1500 items per model per language = 21,000 total rows
    assert len(df_per_item) == 21000

    for m in expected_models:
        for lang in ["bn", "en"]:
            sub = df_per_item[(df_per_item["model"] == m) & (df_per_item["lang"] == lang)]
            assert len(sub) == 1500
            assert sub["question_id"].nunique() == 1500, f"Duplicate question_id in {m}/{lang}"
            assert not sub["tpw"].isna().any()
            assert not sub["tpc"].isna().any()
            assert not sub["strr"].isna().any()

def test_s8_existing_model_exact_match():
    tables_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tables"
    df_all = pd.read_csv(os.path.join(tables_dir, "tokenizer_fragmentation_per_item_all7.csv"))
    ref_path = "/home/naeem/Downloads/Bangla-Calibration-Thesis/anaysis-cyclic-run-and-sensitivity-analysis-v4h/tokenizer_fragmentation/tokenizer_fragmentation_per_item.csv"
    df_ref = pd.read_csv(ref_path)

    for m in ["llama32_3b", "titullm_1b", "tigerllm_1b"]:
        for lang in ["bn", "en"]:
            sub_calc = df_all[(df_all["model"] == m) & (df_all["lang"] == lang)].sort_values(by="question_id").reset_index(drop=True)
            sub_ref = df_ref[(df_ref["model_key"] == m) & (df_ref["lang"] == lang)].sort_values(by="question_id").reset_index(drop=True)

            assert len(sub_calc) == len(sub_ref) == 1500
            np.testing.assert_allclose(sub_calc["tpw"], sub_ref["tpw"], atol=1e-6)
            np.testing.assert_allclose(sub_calc["tpc"], sub_ref["tpc"], atol=1e-6)
            np.testing.assert_allclose(sub_calc["strr"], sub_ref["strr"], atol=1e-6)
            np.testing.assert_array_equal(sub_calc["n_words"], sub_ref["n_words"])
            np.testing.assert_array_equal(sub_calc["n_tokens"], sub_ref["n_tokens"])

def test_s8_tokenizer_identity_and_diversity():
    tables_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tables"
    df_sum = pd.read_csv(os.path.join(tables_dir, "tpw_strr_summary_all7.csv"))

    # Llama 3.2 3B, TituLLM 1B, and Llama 3.2 3B Base must have identical hash
    h_llama = df_sum[df_sum["model"] == "llama32_3b"]["tokenizer_identity_hash"].iloc[0]
    h_titu = df_sum[df_sum["model"] == "titullm_1b"]["tokenizer_identity_hash"].iloc[0]
    h_base = df_sum[df_sum["model"] == "llama32_3b_base"]["tokenizer_identity_hash"].iloc[0]
    assert h_llama == h_titu == h_base

    # Exactly 5 distinct tokenizer levels across the 7 models
    distinct_hashes = df_sum["tokenizer_identity_hash"].nunique()
    assert distinct_hashes == 5

    # Check targets for Bangla mean TPW
    bn_llama_tpw = df_sum[(df_sum["model"] == "llama32_3b") & (df_sum["language"] == "bn")]["mean_tpw"].iloc[0]
    bn_tiger_tpw = df_sum[(df_sum["model"] == "tigerllm_1b") & (df_sum["language"] == "bn")]["mean_tpw"].iloc[0]
    assert abs(bn_llama_tpw - 6.968736) < 1e-4
    assert abs(bn_tiger_tpw - 2.316911) < 1e-4
