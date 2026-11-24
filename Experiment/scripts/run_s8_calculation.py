#!/usr/bin/env python3
"""
TASK S8: Tokenization Fragmentation Across All Seven Models
- Verifies Hugging Face repository IDs, availability, license, gating
- Validates existing 3 models against tokenizer_fragmentation_per_item.csv
- Calculates per-item TPW, TPC, STRR for all 7 models across 1500 bn + 1500 en items
- Generates required CSVs:
    1. tokenizer_fragmentation_per_item_all7.csv
    2. tpw_strr_summary_all7.csv
    3. tpw_strr_bn_en_ratio.csv
- Computes diversity hashes, identifies identical groups, distinct tokenizer levels
- Ranks models by Bangla mean TPW
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 0. Setup & Hugging Face Token Authentication (never hard-code or print token)
# -----------------------------------------------------------------------------
def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    
    # Check known local token locations
    candidates = [
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
        "/home/naeem/Downloads/hugging face token.txt",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    for line in f:
                        clean = line.strip()
                        if clean.startswith("hf_"):
                            return clean
            except Exception:
                pass
    return ""

HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

# -----------------------------------------------------------------------------
# 1. Model Registry Definition & Hugging Face Verification Info
# -----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "llama32_3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "is_existing": True,
        "gated": True,
        "license": "Llama 3.2 Community License",
        "family": "llama_family",
        "description": "Llama 3.2 3B Instruct (untouched control)",
    },
    "titullm_1b": {
        "repo_id": "hishab/titulm-llama-3.2-1b-v1.1",
        "is_existing": True,
        "gated": False,
        "license": "Llama 3.2 Community License",
        "family": "llama_family",
        "description": "TituLLM 1B v1.1 (Bengali continued-pretraining of Llama 3.2)",
    },
    "tigerllm_1b": {
        "repo_id": "md-nishat-008/TigerLLM-1B-it",
        "is_existing": True,
        "gated": False,
        "license": "Gemma Terms of Use",
        "family": "gemma_family",
        "description": "TigerLLM 1B IT (Gemma 3 derivative)",
    },
    "phi35_mini": {
        "repo_id": "microsoft/Phi-3.5-mini-instruct",
        "is_existing": False,
        "gated": False,
        "license": "MIT",
        "family": "phi_family",
        "description": "Phi-3.5 Mini Instruct (Microsoft)",
    },
    "qwen25_3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct",
        "is_existing": False,
        "gated": False,
        "license": "Qwen Research License",
        "family": "qwen_family",
        "description": "Qwen 2.5 3B Instruct (Alibaba)",
    },
    "bloomz_3b": {
        "repo_id": "bigscience/bloomz-3b",
        "is_existing": False,
        "gated": False,
        "license": "BigScience OpenRAIL-M v1",
        "family": "bloom_family",
        "description": "BLOOMZ 3B (BigScience multilingual)",
    },
    "llama32_3b_base": {
        "repo_id": "meta-llama/Llama-3.2-3B",
        "is_existing": False,
        "gated": True,
        "license": "Llama 3.2 Community License",
        "family": "llama_family",
        "description": "Llama 3.2 3B Base (Meta unaligned base model)",
    },
}

_PROBE_TEXT = "বাংলাদেশের রাজধানী ঢাকা এবং আমাদের মাতৃভাষা বাংলা। The capital of Bangladesh is Dhaka."

# -----------------------------------------------------------------------------
# 2. Tokenization & Fragmentation Functions (Reusing Exact Existing Implementation)
# -----------------------------------------------------------------------------
def _tokenize_text(tokenizer: Any, text: str) -> List[Any]:
    if not text:
        return []
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(text)
    elif hasattr(tokenizer, "tokenize"):
        return tokenizer.tokenize(text)
    elif callable(tokenizer):
        return tokenizer(text)
    else:
        return text.strip().split()

def parse_options(opt_raw: Any) -> List[str]:
    if isinstance(opt_raw, list):
        return [str(x) for x in opt_raw]
    if isinstance(opt_raw, str):
        try:
            val = json.loads(opt_raw)
            if isinstance(val, list):
                return [str(x) for x in val]
        except Exception:
            pass
    return []

def item_to_text(question: str, options: List[str], lang: str) -> str:
    """Build the marginal per-item prompt text (excluding few-shot prefix)."""
    if lang == "bn":
        q_prefix = "প্রশ্ন:"
        opt_prefix = "বিকল্পগুলি:"
    else:
        q_prefix = "Question:"
        opt_prefix = "Options:"
    q = str(question or "").strip()
    opt_lines = "\n".join(
        f"{letter}. {str(opt).strip()}"
        for letter, opt in zip("ABCD", options[:4])
    )
    return f"{q_prefix} {q}\n{opt_prefix}\n{opt_lines}"

def compute_item_metrics(tokenizer: Any, text: str) -> Dict[str, Any]:
    """Reuses exact _item_metrics implementation from notebook."""
    words = text.split()
    n_words = len(words)
    try:
        toks = _tokenize_text(tokenizer, text)
        n_tokens = len(toks)
    except Exception:
        n_tokens = 0
    tpw = n_tokens / max(n_words, 1)
    tpc = n_tokens / max(len(text), 1)
    strr = 1.0 - (n_words / max(n_tokens, 1)) if n_tokens > 0 else 0.0
    return {
        "n_words": n_words,
        "n_tokens": n_tokens,
        "tpw": round(float(tpw), 6),
        "tpc": round(float(tpc), 6),
        "strr": round(float(strr), 6),
    }

# -----------------------------------------------------------------------------
# 3. Execution Pipeline
# -----------------------------------------------------------------------------
def run():
    from transformers import AutoTokenizer

    print("=================================================================")
    print("TASK S8: Tokenization Fragmentation Profiling across All 7 Models")
    print("=================================================================")
    
    # Locate dataset subsets
    base_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis"
    bn_sub_path = os.path.join(base_dir, "results/bn_subset.csv")
    en_sub_path = os.path.join(base_dir, "results/en_subset.csv")
    ref_frag_path = os.path.join(base_dir, "anaysis-cyclic-run-and-sensitivity-analysis-v4h/tokenizer_fragmentation/tokenizer_fragmentation_per_item.csv")

    assert os.path.exists(bn_sub_path), f"Missing {bn_sub_path}"
    assert os.path.exists(en_sub_path), f"Missing {en_sub_path}"
    assert os.path.exists(ref_frag_path), f"Missing {ref_frag_path}"

    df_bn = pd.read_csv(bn_sub_path)
    df_en = pd.read_csv(en_sub_path)
    df_ref = pd.read_csv(ref_frag_path)

    print(f"Loaded Bangla subset: {len(df_bn)} items")
    print(f"Loaded English subset: {len(df_en)} items")
    print(f"Loaded Reference fragmentation table: {len(df_ref)} items")

    # STEP 1: Verify tokenizers and compute identity hashes
    print("\n--- STEP 1 & 4: Loading Tokenizers & Diversity Audit ---")
    tokenizers = {}
    tok_hashes = {}
    probe_encodings = {}

    for mkey, minfo in MODEL_REGISTRY.items():
        repo = minfo["repo_id"]
        gated = minfo["gated"]
        tok_kwargs = {}
        if gated:
            if not HF_TOKEN:
                raise RuntimeError(f"Gated model {mkey} requires HF_TOKEN, but none found!")
            tok_kwargs["token"] = HF_TOKEN
        
        print(f"Loading tokenizer: {mkey:16s} ({repo})...", end="", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, **tok_kwargs)
        tokenizers[mkey] = tok
        vocab_size = len(tok)
        probe_toks = tok.encode(_PROBE_TEXT, add_special_tokens=False)
        probe_encodings[mkey] = probe_toks
        
        # Identity hash based on vocab size and probe text token sequence
        h = hashlib.sha256(f"{vocab_size}_{probe_toks}".encode()).hexdigest()[:12]
        tok_hashes[mkey] = h
        print(f" Done. Vocab: {vocab_size:,} | Probe tokens: {len(probe_toks)} | Hash: {h}")

    # STEP 2: Existing model validation against tokenizer_fragmentation_per_item.csv
    print("\n--- STEP 2: Existing Model Validation (llama32_3b, titullm_1b, tigerllm_1b) ---")
    existing_models = ["llama32_3b", "titullm_1b", "tigerllm_1b"]
    max_abs_diff = 0.0
    n_comparisons = 0

    for mkey in existing_models:
        tok = tokenizers[mkey]
        for lang, df_sub in [("bn", df_bn), ("en", df_en)]:
            for _, row in df_sub.iterrows():
                qid = int(row["question_id"])
                opts = parse_options(row["options"])
                text = item_to_text(row["question"], opts, lang)
                metrics = compute_item_metrics(tok, text)

                # Lookup reference
                ref_rows = df_ref[(df_ref["model_key"] == mkey) & (df_ref["lang"] == lang) & (df_ref["question_id"] == qid)]
                if len(ref_rows) != 1:
                    raise ValueError(f"Reference lookup failed for {mkey}/{lang}/{qid}: found {len(ref_rows)} rows")
                ref_row = ref_rows.iloc[0]

                # Assert exact words and tokens
                if metrics["n_words"] != int(ref_row["n_words"]):
                    raise AssertionError(f"n_words mismatch for {mkey}/{lang}/{qid}: {metrics['n_words']} vs {ref_row['n_words']}")
                if metrics["n_tokens"] != int(ref_row["n_tokens"]):
                    raise AssertionError(f"n_tokens mismatch for {mkey}/{lang}/{qid}: {metrics['n_tokens']} vs {ref_row['n_tokens']}")

                # Check tpw, tpc, strr within 1e-6
                diff_tpw = abs(metrics["tpw"] - float(ref_row["tpw"]))
                diff_tpc = abs(metrics["tpc"] - float(ref_row["tpc"]))
                diff_strr = abs(metrics["strr"] - float(ref_row["strr"]))

                max_abs_diff = max(max_abs_diff, diff_tpw, diff_tpc, diff_strr)
                n_comparisons += 3

                if diff_tpw > 1e-6 or diff_tpc > 1e-6 or diff_strr > 1e-6:
                    raise AssertionError(
                        f"Validation failed for {mkey}/{lang}/{qid}: "
                        f"tpw diff={diff_tpw}, tpc diff={diff_tpc}, strr diff={diff_strr}"
                    )

    print(f"✅ Exact match confirmed across all {n_comparisons:,} metrics! Max abs diff: {max_abs_diff:.2e} <= 1e-6")

    # Verify Llama / Titu identity
    llama_probe = probe_encodings["llama32_3b"]
    titu_probe = probe_encodings["titullm_1b"]
    assert llama_probe == titu_probe, "Llama-3.2 and TituLLM probe encodings differ!"
    assert tok_hashes["llama32_3b"] == tok_hashes["titullm_1b"], "Llama-3.2 and TituLLM hashes differ!"
    print(f"✅ llama32_3b and titullm_1b tokenizer identity CONFIRMED (hash={tok_hashes['llama32_3b']})")

    # STEP 3: All-seven tokenizer calculation
    print("\n--- STEP 3: Computing Per-Item Fragmentation for All Seven Models ---")
    all_rows = []
    
    all_models = [
        "llama32_3b",
        "titullm_1b",
        "tigerllm_1b",
        "phi35_mini",
        "qwen25_3b",
        "bloomz_3b",
        "llama32_3b_base",
    ]

    for mkey in all_models:
        tok = tokenizers[mkey]
        print(f"Processing model: {mkey}...", flush=True)
        for lang, df_sub in [("bn", df_bn), ("en", df_en)]:
            for _, row in df_sub.iterrows():
                qid = row["question_id"]
                subj = row.get("subject", row.get("subject_name", ""))
                opts = parse_options(row["options"])
                text = item_to_text(row["question"], opts, lang)
                m = compute_item_metrics(tok, text)
                all_rows.append({
                    "model": mkey,
                    "lang": lang,
                    "question_id": qid,
                    "subject_name": subj,
                    "tpw": m["tpw"],
                    "tpc": m["tpc"],
                    "strr": m["strr"],
                    "n_words": m["n_words"],
                    "n_tokens": m["n_tokens"],
                })

    df_all = pd.DataFrame(all_rows)
    print(f"Total rows calculated: {len(df_all):,} (expected 7 * 2 * 1500 = 21,000)")
    assert len(df_all) == 21000, f"Expected 21000 rows, got {len(df_all)}"

    # Save tokenizer_fragmentation_per_item_all7.csv
    tables_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tables"
    e2_dir = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run"
    out_per_item_tables = os.path.join(tables_dir, "tokenizer_fragmentation_per_item_all7.csv")
    out_per_item_e2 = os.path.join(e2_dir, "tokenizer_fragmentation_per_item_all7.csv")
    df_all.to_csv(out_per_item_tables, index=False)
    df_all.to_csv(out_per_item_e2, index=False)
    print(f"📄 Saved per-item all-seven: {out_per_item_tables}")

    # Build Summary Table: tpw_strr_summary_all7.csv
    summary_rows = []
    for mkey in all_models:
        tok = tokenizers[mkey]
        vocab_size = len(tok)
        thash = tok_hashes[mkey]
        for lang in ["bn", "en"]:
            sub = df_all[(df_all["model"] == mkey) & (df_all["lang"] == lang)]
            summary_rows.append({
                "model": mkey,
                "language": lang,
                "n": len(sub),
                "mean_tpw": round(float(sub["tpw"].mean()), 6),
                "std_tpw": round(float(sub["tpw"].std()), 6),
                "median_tpw": round(float(sub["tpw"].median()), 6),
                "mean_strr": round(float(sub["strr"].mean()), 6),
                "std_strr": round(float(sub["strr"].std()), 6),
                "mean_tpc": round(float(sub["tpc"].mean()), 6),
                "vocab_size": vocab_size,
                "tokenizer_identity_hash": thash,
            })
    
    df_summary = pd.DataFrame(summary_rows)
    out_sum_tables = os.path.join(tables_dir, "tpw_strr_summary_all7.csv")
    out_sum_e2 = os.path.join(e2_dir, "tpw_strr_summary_all7.csv")
    df_summary.to_csv(out_sum_tables, index=False)
    df_summary.to_csv(out_sum_e2, index=False)
    print(f"📄 Saved summary all-seven: {out_sum_tables}")

    # Build Ratio Table: tpw_strr_bn_en_ratio.csv
    ratio_rows = []
    for mkey in all_models:
        bn_row = df_summary[(df_summary["model"] == mkey) & (df_summary["language"] == "bn")].iloc[0]
        en_row = df_summary[(df_summary["model"] == mkey) & (df_summary["language"] == "en")].iloc[0]
        tpw_ratio = round(bn_row["mean_tpw"] / en_row["mean_tpw"], 6)
        strr_ratio = round(bn_row["mean_strr"] / en_row["mean_strr"], 6)
        ratio_rows.append({
            "model": mkey,
            "bn_mean_tpw": bn_row["mean_tpw"],
            "en_mean_tpw": en_row["mean_tpw"],
            "bn_en_tpw_ratio": tpw_ratio,
            "bn_mean_strr": bn_row["mean_strr"],
            "en_mean_strr": en_row["mean_strr"],
            "bn_en_strr_ratio": strr_ratio,
        })
    df_ratio = pd.DataFrame(ratio_rows)
    out_ratio_tables = os.path.join(tables_dir, "tpw_strr_bn_en_ratio.csv")
    out_ratio_e2 = os.path.join(e2_dir, "tpw_strr_bn_en_ratio.csv")
    df_ratio.to_csv(out_ratio_tables, index=False)
    df_ratio.to_csv(out_ratio_e2, index=False)
    print(f"📄 Saved BN/EN ratios: {out_ratio_tables}")

    # STEP 4: Tokenizer diversity analysis
    print("\n--- STEP 4: Tokenizer Diversity & Grouping ---")
    hash_groups = {}
    for mkey, h in tok_hashes.items():
        hash_groups.setdefault(h, []).append(mkey)
    
    print(f"Total distinct tokenizer levels: {len(hash_groups)}")
    for h, group in hash_groups.items():
        vsize = len(tokenizers[group[0]])
        print(f"  Level hash {h} (vocab={vsize:,}): {group}")

    # Rank by mean Bangla TPW
    bn_rank = df_summary[df_summary["language"] == "bn"].sort_values(by="mean_tpw", ascending=True)
    print("\n--- Model Ranking by Mean Bangla TPW (Lower = Better/Less Fragmented) ---")
    rank_idx = 1
    for _, r in bn_rank.iterrows():
        print(f"  {rank_idx}. {r['model']:16s}: Mean TPW = {r['mean_tpw']:.4f} | STRR = {r['mean_strr']:.4f} | TPC = {r['mean_tpc']:.4f} | Vocab = {r['vocab_size']:,}")
        rank_idx += 1

    # Check validation targets
    llama_mean = df_summary[(df_summary["model"] == "llama32_3b") & (df_summary["language"] == "bn")]["mean_tpw"].iloc[0]
    tiger_mean = df_summary[(df_summary["model"] == "tigerllm_1b") & (df_summary["language"] == "bn")]["mean_tpw"].iloc[0]
    print(f"\nTarget Validation Check:")
    print(f"  Llama-3.2 3B BN Mean TPW: {llama_mean:.4f} (target ~6.9687) -> Diff: {abs(llama_mean - 6.968736):.6f}")
    print(f"  TigerLLM 1B  BN Mean TPW: {tiger_mean:.4f} (target ~2.3169) -> Diff: {abs(tiger_mean - 2.316911):.6f}")
    assert abs(llama_mean - 6.968736) < 1e-4, f"Unexpected Llama mean TPW: {llama_mean}"
    assert abs(tiger_mean - 2.316911) < 1e-4, f"Unexpected TigerLLM mean TPW: {tiger_mean}"

    print("\n✅ All calculations, diversity audits, and target validations PASSED successfully!")

if __name__ == "__main__":
    run()
