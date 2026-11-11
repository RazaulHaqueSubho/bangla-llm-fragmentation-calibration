"""Step 2a: Candidate-model tokenizer-only audit.

Loads tokenizers ONLY (no model weights) via AutoTokenizer.from_pretrained.
Reports: tokenizer class, len(tokenizer), vocab/merges identity across models,
and token-id equality on the same 300 BN items from combined_results.csv.

No model evaluation, no model weights loaded.
"""

import sys
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

import pandas as pd
from transformers import AutoTokenizer


def load_tokenizer_only(model_id: str):
    """Load tokenizer only, no model weights."""
    tok = AutoTokenizer.from_pretrained(model_id)
    return tok


def get_vocab_dict(tok):
    """Return the tokenizer vocabulary as a dict."""
    return tok.get_vocab()


def main():
    candidates = [
        ("llama32_3b", "meta-llama/Llama-3.2-3B-Instruct"),
        ("titullm_1b", "hishab/titulm-llama-3.2-1b-v1.1"),
        ("tigerllm_1b", "md-nishat-008/TigerLLM-1B-it"),
        ("gemma-3-1b-it", "google/gemma-3-1b-it"),
        ("gemma-3-4b-it", "google/gemma-3-4b-it"),
        ("qwen2.5-3b-instruct", "Qwen/Qwen2.5-3B-Instruct"),
    ]

    # Load 300 BN items from bn_subset.csv for token-id identity test
    bn_subset_path = CODE_DIR / "results" / "bn_subset.csv"
    if not bn_subset_path.exists():
        bn_subset_path = CODE_DIR / "code" / "results" / "bn_subset.csv"
    bn_df = pd.read_csv(bn_subset_path)
    bn_items = bn_df[bn_df["language"] == "bn"].head(300)
    bn_texts = bn_items["question"].tolist()
    print(f"Loaded {len(bn_texts)} BN items for token-id identity test from {bn_subset_path}")

    tokenizers = {}
    results = []

    for key, model_id in candidates:
        try:
            tok = load_tokenizer_only(model_id)
            vocab = get_vocab_dict(tok)
            tok_class = type(tok).__name__
            vocab_size = len(tok)
            tokenizers[key] = (tok, vocab)

            results.append({
                "key": key,
                "model_id": model_id,
                "tokenizer_class": tok_class,
                "len_tokenizer": vocab_size,
                "vocab_size": len(vocab),
                "status": "LOADED",
                "error": "",
            })
            print(f"  {key}: {tok_class}, len={vocab_size}, vocab_size={len(vocab)}")
        except Exception as e:
            results.append({
                "key": key,
                "model_id": model_id,
                "tokenizer_class": "",
                "len_tokenizer": 0,
                "vocab_size": 0,
                "status": f"ERROR: {e}",
                "error": str(e),
            })
            print(f"  {key}: ERROR loading tokenizer: {e}")

    # Compare vocab equality across all pairs
    print("\n=== Vocabulary Equality (get_vocab() ==) ===")
    vocab_eq_rows = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ki, mi = candidates[i]
            kj, mj = candidates[j]
            if ki in tokenizers and kj in tokenizers:
                vi = tokenizers[ki][1]
                vj = tokenizers[kj][1]
                equal = vi == vj
                vocab_eq_rows.append({
                    "model_i": ki,
                    "model_j": kj,
                    "vocab_equal": equal,
                })
                print(f"  {ki} vs {kj}: vocab_equal={equal}")

    # Compare token ids on 300 BN items
    print("\n=== Token-ID Equality on 300 BN Items ===")
    token_eq_rows = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ki, mi = candidates[i]
            kj, mj = candidates[j]
            if ki in tokenizers and kj in tokenizers:
                toi = tokenizers[ki][0]
                toj = tokenizers[kj][0]
                all_equal = True
                for text in bn_texts:
                    ti = toi.encode(text, add_special_tokens=False)
                    tj = toj.encode(text, add_special_tokens=False)
                    if ti != tj:
                        all_equal = False
                        break
                token_eq_rows.append({
                    "model_i": ki,
                    "model_j": kj,
                    "token_ids_equal_on_300bn": all_equal,
                })
                print(f"  {ki} vs {kj}: token_ids_equal_on_300bn={all_equal}")

    # Summary table
    print("\n=== CANDIDATE TOKENIZER SUMMARY ===")
    print(f"{'Key':<20} {'Model ID':<45} {'Tokenizer Class':<25} {'Len':<8} {'Status'}")
    for r in results:
        print(f"{r['key']:<20} {r['model_id']:<45} {r['tokenizer_class']:<25} {r['len_tokenizer']:<8} {r['status']}")

    # Gemma vs TigerLLM tokenizer identity
    print("\n=== GEMMA vs TIGERLLM TOKENIZER IDENTITY ===")
    if "tigerllm_1b" in tokenizers:
        for gemma_key in ["gemma-3-1b-it", "gemma-3-4b-it"]:
            if gemma_key in tokenizers:
                vocab_eq = any(r["model_i"] == "tigerllm_1b" and r["model_j"] == gemma_key and r["vocab_equal"] for r in vocab_eq_rows) or \
                           any(r["model_i"] == gemma_key and r["model_j"] == "tigerllm_1b" and r["vocab_equal"] for r in vocab_eq_rows)
                tok_eq = any(r["model_i"] == "tigerllm_1b" and r["model_j"] == gemma_key and r["token_ids_equal_on_300bn"] for r in token_eq_rows) or \
                         any(r["model_i"] == gemma_key and r["model_j"] == "tigerllm_1b" and r["token_ids_equal_on_300bn"] for r in token_eq_rows)
                print(f"  TigerLLM vs {gemma_key}: vocab_equal={vocab_eq}, token_ids_equal={tok_eq}")
                if vocab_eq and tok_eq:
                    print(f"  => gemma-3 and TigerLLM SHARE a tokenizer")
                else:
                    print(f"  => gemma-3 and TigerLLM have DIFFERENT tokenizers")


if __name__ == "__main__":
    main()
