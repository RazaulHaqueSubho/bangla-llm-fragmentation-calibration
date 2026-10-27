"""Build ~1,000 stratified BN-EN item-parallel pairs from CohereLabs/Global-MMLU.

Reads the downloaded Global-MMLU bn/en test parquet files (already verified to be
one-to-one item-paired: identical sample_ids, subjects, subject counts, and
answers across languages). Selects a stratified sample across subjects with a
fixed seed, assigns a pair_id, and writes the paired items to
results/item_parallel/.

No model evaluation is performed here (tokenizer/dataset metadata only).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

SEED = 42  # matches run_manifest reproducibility_seed
TARGET_N = 1000

BASE_URL = "https://huggingface.co/datasets/CohereLabs/Global-MMLU/resolve/main/"
SOURCE = "CohereLabs/Global-MMLU"
SOURCE_LICENSE = "Apache 2.0"
OPTION_LETTERS = ["A", "B", "C", "D"]

# Workspace-relative output dir; source parquets cached under /tmp/kilo/e4
# (downloaded on demand from HF Hub if missing).
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "item_parallel")
SRC_DIR = "/tmp/kilo/e4"

_FILES = {
    "bn_test": "bn/test-00000-of-00001.parquet",
    "en_test": "en/test-00000-of-00001.parquet",
    "bn_dev": "bn/dev-00000-of-00001.parquet",
    "en_dev": "en/dev-00000-of-00001.parquet",
}


def _ensure_downloaded() -> None:
    """Fetch Global-MMLU parquet shards from HF Hub if not already cached."""
    os.makedirs(SRC_DIR, exist_ok=True)
    for key, rel in _FILES.items():
        dst = os.path.join(SRC_DIR, rel.replace("/", "_"))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = BASE_URL + rel
        urllib.request.urlretrieve(url, dst)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _local_path(rel: str) -> str:
    return os.path.join(SRC_DIR, rel.replace("/", "_"))


def verify_one_to_one(bn: pd.DataFrame, en: pd.DataFrame) -> None:
    """Assert bn and en test sets are item-paired one-to-one."""
    assert len(bn) == len(en), f"row count mismatch: {len(bn)} vs {len(en)}"
    assert list(bn["sample_id"]) == list(en["sample_id"]), "sample_id ordering mismatch"
    assert set(bn["sample_id"]) == set(en["sample_id"]), "sample_id set mismatch"
    assert (bn["subject"] == en["subject"]).all(), "subject mismatch"
    assert (bn["answer"] == en["answer"]).all(), "answer key mismatch"
    assert set(bn.columns) == set(en.columns), "column mismatch"


def verify_disjoint_from_dev(selected_ids: set) -> None:
    """Ensure no selected test sample_id appears in the dev split (disjointness)."""
    for lang, rel in (("bn", "bn/dev-00000-of-00001.parquet"), ("en", "en/dev-00000-of-00001.parquet")):
        dev = load_parquet(_local_path(rel))
        overlap = selected_ids & set(dev["sample_id"].tolist())
        assert not overlap, f"{lang}: {len(overlap)} selected items overlap dev split"


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sample across subjects (mirrors data_loader.sample_stratified_subset)."""
    if n >= len(df):
        return df
    rng = np.random.RandomState(seed)
    by_subject: Dict[str, List[int]] = {}
    for i, row in df.iterrows():
        by_subject.setdefault(str(row["subject"]), []).append(i)
    total = len(df)
    chosen: List[int] = []
    for subj, idxs in by_subject.items():
        k = int(round(len(idxs) / total * n))
        if k > 0 and idxs:
            k = min(k, len(idxs))
            chosen.extend(rng.choice(idxs, size=k, replace=False).tolist())
    if len(chosen) < n:
        remaining = list(set(range(len(df))) - set(chosen))
        if remaining:
            needed = min(n - len(chosen), len(remaining))
            chosen.extend(rng.choice(remaining, size=needed, replace=False).tolist())
    elif len(chosen) > n:
        chosen = rng.choice(chosen, size=n, replace=False).tolist()
    chosen.sort()
    return df.loc[chosen].copy()


def build() -> pd.DataFrame:
    _ensure_downloaded()
    bn = load_parquet(_local_path("bn/test-00000-of-00001.parquet"))
    en = load_parquet(_local_path("en/test-00000-of-00001.parquet"))
    verify_one_to_one(bn, en)

    # Stratified sample of pairs (sample by subject on the shared sample_id space).
    pairs = stratified_sample(bn, TARGET_N, SEED)
    pair_ids = set(pairs["sample_id"].tolist())
    verify_disjoint_from_dev(pair_ids)

    rows: List[Dict[str, Any]] = []
    for _, r in pairs.iterrows():
        pid = str(r["sample_id"])
        subj = str(r["subject"])
        gold_bn = str(r["answer"]).upper()
        # English row is the matching row by sample_id.
        er = en[en["sample_id"] == pid].iloc[0]
        gold_en = str(er["answer"]).upper()
        assert gold_bn == gold_en, f"{pid}: answer mismatch {gold_bn} vs {gold_en}"
        for lang, src in (("bn", r), ("en", er)):
            rows.append({
                "pair_id": pid,
                "question_id": f"{pid}#{lang}",
                "language": lang,
                "subject_group": subj,
                "question": str(src["question"]),
                "option_a": str(src["option_a"]),
                "option_b": str(src["option_b"]),
                "option_c": str(src["option_c"]),
                "option_d": str(src["option_d"]),
                "gold": gold_bn,
                "answer_idx": OPTION_LETTERS.index(gold_bn),
                "cultural_sensitivity_label": str(src.get("cultural_sensitivity_label", "")),
                "is_annotated": bool(src.get("is_annotated", False)),
                "source_dataset": SOURCE,
                "source_config": lang,
                "source_split": "test",
            })

    out = pd.DataFrame(rows)
    # Deterministic ordering: by pair_id then language.
    out = out.sort_values(["pair_id", "language"]).reset_index(drop=True)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = build()
    n_pairs = df["pair_id"].nunique()
    csv_path = os.path.join(OUT_DIR, "item_parallel_bn_en.csv")
    json_path = os.path.join(OUT_DIR, "item_parallel_bn_en.json")
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    manifest = {
        "source_dataset": SOURCE,
        "source_license": SOURCE_LICENSE,
        "source_url": "https://huggingface.co/datasets/CohereLabs/Global-MMLU",
        "configs_used": ["bn", "en"],
        "split_used": "test",
        "dev_split_excluded": True,
        "dev_items": {"bn": 285, "en": 285},
        "seed": SEED,
        "target_n_pairs": TARGET_N,
        "n_pairs": int(n_pairs),
        "n_rows": int(len(df)),
        "subjects": int(df["subject_group"].nunique()),
        "languages": sorted(df["language"].unique().tolist()),
        "pair_id_scheme": "Global-MMLU sample_id (bn and en share identical sample_id)",
        "one_to_one_verified": True,
        "disjoint_from_dev_verified": True,
        "model_evaluation": False,
        "files": {
            "csv": csv_path,
            "json": json_path,
            "manifest": manifest_path,
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {csv_path} ({len(df)} rows, {n_pairs} pairs)")
    print(f"Wrote {manifest_path}")
    print("Subjects:", int(df["subject_group"].nunique()))
    print("Per-language rows:", df["language"].value_counts().to_dict())
    print("Per-pair rows (should all be 2):", df.groupby("pair_id").size().value_counts().to_dict())


if __name__ == "__main__":
    main()
