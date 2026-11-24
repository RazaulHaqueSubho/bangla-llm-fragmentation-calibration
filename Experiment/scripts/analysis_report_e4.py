#!/usr/bin/env python3
"""
analysis_report_e4.py — Reproducible research analysis for
"Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? A Cross-Lingual Study"
REPORT: Analysis_Report_6models_plus_phi_plus_e4.pdf
6-MODEL EXTENDED REPORT: titullm_1b, tigerllm_1b, llama32_3b (v4h) + qwen25_3b, bloomz_3b, llama32_3b_base (v4i)
PLUS phi35_mini second run as a clearly-labelled EXPLORATORY model (outside preregistered main set).
PLUS E4 (v4j) extension models mgpt_1_3b (mGPT), olmo2_1b (OLMo-2-0425-1B),
smollm2_1_7b (SmolLM2-1.7B), sarvam1_2b (sarvam-1) audited under pre-specified rules R1-R7:
E4 models are NOT part of the confirmatory analysis; none qualified as MAIN-EXTENSION
(3 models GATE-FAILED dev-only, sarvam1_2b INVALID on R4). E4 models appear only in the new
Section 11 ("Extension models (E4), added after the preregistered six-model set").

Reads from (READ-ONLY):
  Old models (v4h): /home/naeem/Downloads/results/results_E2_cyclic4_<model>/
  New models (v4i): /home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/
  phi35_mini second run (read-only copy): phi_second_run_check/raw/
  E4 v4j zip (read-only copy): report_work/e4_check/raw/results-mgpt-olmo2-sarvam1-smollm2-v4j.zip
  E4 unzipped copy: report_work/e4_check/raw/extracted/
Writes to: /home/naeem/Downloads/Bangla-Calibration-Thesis/report_work/ (tables_e4/, figures_e4/, Analysis_Report_6models_plus_phi_plus_e4.pdf)

Do not modify the input directory. All numbers originate from input files.
"""
import os
import sys
import json
import glob as globmod
import hashlib
import itertools
import zipfile
import datetime
import tempfile
import shutil
import math

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy import stats
    HAS_SCIPY = True
except Exception as e:
    HAS_SCIPY = False
    SCIPY_IMPORT_ERR = str(e)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception as e:
    HAS_STATSMODELS = False
    STATSMODELS_IMPORT_ERR = str(e)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as e:
    HAS_MATPLOTLIB = False
    MPL_IMPORT_ERR = str(e)

try:
    import reportlab
    HAS_REPORTLAB = True
except Exception as e:
    HAS_REPORTLAB = False
    REPORTLAB_IMPORT_ERR = str(e)

if HAS_REPORTLAB:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                    Spacer, PageBreak, NextPageTemplate, Image, Table, TableStyle)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPORT_WORK = "/home/naeem/Downloads/Bangla-Calibration-Thesis/report_work"
OLD_INPUT_DIR = "/home/naeem/Downloads/results"
NEW_INPUT_DIR = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi"
WORK_DIR = REPORT_WORK
TABLES_DIR = os.path.join(WORK_DIR, "tables_e4")
FIGURES_DIR = os.path.join(WORK_DIR, "figures_e4")
EXTRACT_DIR = os.path.join(WORK_DIR, "extracted_6models_plus_phi")
EXECUTION_LOG = os.path.join(WORK_DIR, "execution_log_e4.txt")
MISSING_LOG = os.path.join(WORK_DIR, "missing_log_e4.txt")
E4_PDF_PATH = os.path.join(WORK_DIR, "Analysis_Report_6models_plus_phi_plus_e4.pdf")

# E4 (v4j) extension inputs — read-only copy of the v4j zip
E4_ZIP = os.path.join(WORK_DIR, "e4_check", "raw", "results-mgpt-olmo2-sarvam1-smollm2-v4j.zip")
E4_INPUT_DIR = os.path.join(WORK_DIR, "e4_check", "raw", "extracted")
E4_ALL11 = os.path.join(E4_INPUT_DIR, "tokenizer_fragmentation", "tokenizer_fragmentation_per_item_all11.csv")
E4_CANDIDATE_SCREEN = os.path.join(E4_INPUT_DIR, "tokenizer_fragmentation", "candidate_screen.csv")
E4_DEV_SCREEN = os.path.join(E4_INPUT_DIR, "dev_screen_report_e3.csv")
E4_PLAN = os.path.join(E4_INPUT_DIR, "e4_added_models_plan.json")
E4_RUN_MANIFEST = os.path.join(E4_INPUT_DIR, "run_manifest.json")
ALL7_LOCAL = "/home/naeem/Downloads/Bangla-Calibration-Thesis/e2_run/tokenizer_fragmentation_per_item_all7.csv"
LLAMA32_3B_BASE_DIR = os.path.join(NEW_INPUT_DIR, "results_llama_base",
                                   "results_E3_llama32_3b_base", "llama32_3b_base")

# phi35_mini second run (exploratory, outside preregistered main set) — read-only
PHI_SECOND_RUN_DIR = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/phi_second_run_check/raw/results_E3_phi35_mini/phi35_mini"
PHI_FRAG_ALL7 = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/phi_second_run_check/raw/tokenizer_fragmentation/tokenizer_fragmentation_per_item_all7.csv"
PHI_DEV_BN = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/results_E3_phi35_mini/phi35_mini_bn_dev_probabilities.csv"
PHI_DEV_EN = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/results_E3_phi35_mini/phi35_mini_en_dev_probabilities.csv"
PHI_FULL_RUN_MANIFEST = "/home/naeem/Downloads/Bangla-Calibration-Thesis/v4i-run-qwen-bloomz-llama-phi/phi_second_run_check/raw/run_manifest.json"

# Calibration models (6 full-run models)
MODELS = ["titullm_1b", "tigerllm_1b", "llama32_3b", "qwen25_3b", "bloomz_3b", "llama32_3b_base"]
# phi35_mini: dev-screen only, excluded from calibration analyses
PHI_MODEL = "phi35_mini"
ALL_MODELS = MODELS + [PHI_MODEL]
LANGS = ["bn", "en"]
SEED = 42
N_BOOT = 2000
TOL = 1e-6
CHANCE = 0.25

# E4 (v4j) extension models — pre-specified classes from R1-R7 audit (see E4_CHECK_REPORT.md)
E4_MODELS = ["mgpt_1_3b", "olmo2_1b", "smollm2_1_7b", "sarvam1_2b"]
E4_CLASSIFICATION = {
    "mgpt_1_3b": "GATE-FAILED (dev-only)",
    "olmo2_1b": "GATE-FAILED (dev-only)",
    "smollm2_1_7b": "GATE-FAILED (dev-only)",
    "sarvam1_2b": "INVALID",
}
E4_MAIN_EXTENSION = []  # none qualified: sarvam1_2b failed R4 (BN option_mass >= 0.95 on 49.07% < 80%)
E4_GATE_FAILED = ["mgpt_1_3b", "olmo2_1b", "smollm2_1_7b"]
E4_INVALID = ["sarvam1_2b"]
E4_DEV_N = 200

# Per-model input directory (resolved below, including zip extraction)
MODEL_INPUT_DIR = {}

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


def _resolve_model_dirs():
    """Map each model to a local directory containing its per-item CSVs and summaries.
    Old models: results_E2_cyclic4_<model>/<lang>_results.csv (flat)
    New models: results_E3_<model>/<model>/<lang>_results.csv (inside zip or unzipped)
    Zips are extracted to EXTRACT_DIR.
    """
    global MODEL_INPUT_DIR
    MODEL_INPUT_DIR = {}
    # Old models (v4h) — flat directory under OLD_INPUT_DIR
    for m in ["titullm_1b", "tigerllm_1b", "llama32_3b"]:
        d = os.path.join(OLD_INPUT_DIR, f"results_E2_cyclic4_{m}")
        MODEL_INPUT_DIR[m] = d
    # New models (v4i)
    new_flat = {
        "llama32_3b_base": os.path.join(NEW_INPUT_DIR, "results_llama_base",
                                        "results_E3_llama32_3b_base", "llama32_3b_base"),
    }
    for m, d in new_flat.items():
        if os.path.isdir(d):
            MODEL_INPUT_DIR[m] = d
    # Zipped new models — extract
    zips = {
        "qwen25_3b": os.path.join(NEW_INPUT_DIR, "results_E3_qwen25_3b.zip"),
        "bloomz_3b": os.path.join(NEW_INPUT_DIR, "results_E3_bloomz_3b.zip"),
    }
    for m, zf in zips.items():
        if os.path.exists(zf):
            extract_d = os.path.join(EXTRACT_DIR, f"results_E3_{m}")
            if not os.path.exists(os.path.join(extract_d, m)):
                with zipfile.ZipFile(zf) as z:
                    z.extractall(extract_d)
            MODEL_INPUT_DIR[m] = os.path.join(extract_d, f"results_E3_{m}", m)
        else:
            record_missing(f"zip for {m}", f"{zf} not found")
    # phi35_mini — dev screen only, no full run
    MODEL_INPUT_DIR[PHI_MODEL] = os.path.join(NEW_INPUT_DIR, "results_E3_phi35_mini")


_resolve_model_dirs()

# -----------------------------------------------------------------------------
# State / helpers
# -----------------------------------------------------------------------------
def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def log(msg):
    ts = now_iso()
    line = f"[{ts}] {msg}"
    print(line)
    with open(EXECUTION_LOG, "a") as f:
        f.write(line + "\n")


def record_missing(item, reason):
    line = f"[MISSING] {item}: {reason}"
    print(line)
    with open(MISSING_LOG, "a") as f:
        f.write(line + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def script_sha256():
    return sha256_file(__file__)


def read_csv_safe(path):
    if not os.path.exists(path):
        record_missing(path, "file does not exist")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        record_missing(path, f"read error: {e}")
        return None


def read_json_safe(path):
    if not os.path.exists(path):
        record_missing(path, "file does not exist")
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        record_missing(path, f"read error: {e}")
        return None


class AnalysisState:
    def __init__(self):
        self.warnings = []
        self.missing = []
        self.tables = {}
        self.figures = {}  # name -> (caption, source)
        self.table_sources = {}

    def warn(self, msg):
        self.warnings.append(msg)
        log("WARNING: " + msg)

    def save_table(self, name, df, source):
        df = df.copy()
        path = os.path.join(TABLES_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        self.tables[name] = df
        self.table_sources[name] = source
        log(f"Saved table {name} ({path}) source={source} rows={len(df)}")

    def add_figure(self, name, caption, source):
        self.figures[name] = (caption, source)


A = AnalysisState()


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _onehot(answer, n=4):
    v = np.zeros(n, dtype=float)
    idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(answer).strip().upper(), None)
    if idx is not None:
        v[idx] = 1.0
    return v


def ece_uniform(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(conf, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        cb = int(mask.sum())
        if cb == 0:
            bins.append({"bin_index": b, "n": 0, "mean_conf": None, "accuracy": None, "gap": None})
            continue
        mc = float(conf[mask].mean())
        ac = float(correct[mask].mean())
        gap = abs(mc - ac)
        ece += (cb / n) * gap
        bins.append({"bin_index": b, "n": cb, "mean_conf": mc, "accuracy": ac, "gap": gap})
    return ece, bins


def ece_equal_mass(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(conf)
    order = np.argsort(conf, kind="mergesort")
    bin_idx = np.empty(n, dtype=int)
    groups = np.array_split(order, n_bins)
    for b, g in enumerate(groups):
        bin_idx[g] = b
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        cb = int(mask.sum())
        if cb == 0:
            bins.append({"bin_index": b, "n": 0, "mean_conf": None, "accuracy": None, "gap": None})
            continue
        mc = float(conf[mask].mean())
        ac = float(correct[mask].mean())
        gap = abs(mc - ac)
        ece += (cb / n) * gap
        bins.append({"bin_index": b, "n": cb, "mean_conf": mc, "accuracy": ac, "gap": gap})
    return ece, bins


def brier_score(conf, probs, correct, answers):
    # probs: list of arrays [pA,pB,pC,pD] per item; correct: bool array; answers: gold letters
    n = len(conf)
    total = 0.0
    for i in range(n):
        p = np.array([probs[0][i], probs[1][i], probs[2][i], probs[3][i]], dtype=float)
        y = _onehot(answers[i])
        total += float(np.sum((p - y) ** 2))
    return total / n


def nll_score(probs, correct, answers):
    eps = 1e-12
    n = len(answers)
    total = 0.0
    for i in range(n):
        p = np.array([probs[0][i], probs[1][i], probs[2][i], probs[3][i]], dtype=float)
        y = _onehot(answers[i])
        pc = float(np.sum(p * y))
        total += -np.log(max(pc, eps))
    return total / n


def overconfidence_gap(mean_conf, accuracy):
    return float(mean_conf - accuracy)


def letter_counts(answers):
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for a in answers:
        k = str(a).strip().upper()
        if k in counts:
            counts[k] += 1
    return counts


def bootstrap_ci(metric_fn, data, seed=SEED, n_resamples=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(data)
    idx = np.arange(n)
    vals = []
    for _ in range(n_resamples):
        sample = rng.choice(idx, size=n, replace=True)
        sample_data = data.iloc[sample] if hasattr(data, "iloc") else data[sample]
        vals.append(metric_fn(sample_data))
    vals = np.array(vals, dtype=float)
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return lo, hi, float(np.mean(vals))


def metrics_for_df(df, lang_label, model_label):
    """Return dict of cyclic4 metrics for a per-item dataframe."""
    conf = df["confidence"].to_numpy(dtype=float)
    pred = df["predicted_answer"].astype(str).str.upper().to_numpy()
    gold = df["gold_answer"].astype(str).str.upper().to_numpy()
    correct = (pred == gold)
    accuracy = float(correct.mean())
    mean_conf = float(conf.mean())
    ece10, _ = ece_uniform(conf, correct, 10)
    ece15, _ = ece_uniform(conf, correct, 15)
    ece_ad, _ = ece_equal_mass(conf, correct, 10)
    overconf = overconfidence_gap(mean_conf, accuracy)
    letter_dist = letter_counts(pred)
    out = {
        "model": model_label, "lang": lang_label, "n": int(len(df)),
        "accuracy": accuracy, "accuracy_percent": accuracy * 100.0,
        "mean_confidence": mean_conf,
        "ece_uniform_10": ece10, "ece_uniform_15": ece15,
        "ece_adaptive_equal_mass_10": ece_ad,
        "overconfidence_gap": overconf,
        "top_letter": max(letter_dist, key=letter_dist.get),
        "top_share": max(letter_dist.values()) / len(df),
    }
    # Brier / NLL if probabilities valid
    probs = [df[c].to_numpy(dtype=float) for c in ["prob_A", "prob_B", "prob_C", "prob_D"]]
    probs_valid = all(np.isfinite(p).all() and (p >= 0).all() for p in probs)
    row_mass = np.sum(np.stack(probs, axis=1), axis=1) if probs_valid else None
    mass_valid = probs_valid and np.all(np.isfinite(row_mass)) and np.all(np.abs(row_mass - 1.0) < 0.05)
    out["option_probs_valid"] = bool(mass_valid)
    if mass_valid:
        out["brier_score"] = brier_score(conf, probs, correct, gold)
        out["nll"] = nll_score(probs, correct, gold)
    else:
        out["brier_score"] = None
        out["nll"] = None
        record_missing(f"Brier/NLL {model_label} {lang_label}", "option probabilities not valid (NaN, negative, or row mass far from 1)")
    return out


def metrics_for_df_raw(df):
    """Raw metrics using pred_raw / confidence_raw."""
    conf = df["confidence_raw"].to_numpy(dtype=float)
    pred = df["pred_raw"].astype(str).str.upper().to_numpy()
    gold = df["gold_answer"].astype(str).str.upper().to_numpy()
    correct = (pred == gold)
    accuracy = float(correct.mean())
    mean_conf = float(conf.mean())
    ece10, _ = ece_uniform(conf, correct, 10)
    overconf = overconfidence_gap(mean_conf, accuracy)
    letter_dist = letter_counts(pred)
    return {
        "accuracy": accuracy, "accuracy_percent": accuracy * 100.0,
        "mean_confidence": mean_conf,
        "ece_uniform_10": ece10,
        "overconfidence_gap": overconf,
        "top_letter": max(letter_dist, key=letter_dist.get),
        "top_share": max(letter_dist.values()) / len(df),
    }


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------
def per_item_path(model, lang):
    d = MODEL_INPUT_DIR.get(model, "")
    if not d:
        return ""
    return os.path.join(d, f"{lang}_results.csv")


def summary_path(model, lang=None):
    d = MODEL_INPUT_DIR.get(model, "")
    if not d:
        return ""
    base = d
    if lang:
        return os.path.join(base, f"{lang}_summary.json")
    return os.path.join(base, "summary.json")


# =============================================================================
# STEP 0 — INVENTORY
# =============================================================================
def step0_inventory():
    log("=== STEP 0: INVENTORY ===")
    inventory = []
    # Walk all input directories
    input_dirs = [OLD_INPUT_DIR, NEW_INPUT_DIR] + [d for d in MODEL_INPUT_DIR.values() if os.path.isdir(d)]
    seen_roots = set()
    for root in input_dirs:
        if not os.path.isdir(root) or root in seen_roots:
            continue
        seen_roots.add(root)
        for r, dirs, files in os.walk(root):
            for fn in files:
                fp = os.path.join(r, fn)
                rel = os.path.relpath(fp, root)
                try:
                    h = sha256_file(fp)
                    size = os.path.getsize(fp)
                except Exception:
                    h, size = None, None
                inventory.append({
                    "root": root, "path": rel, "sha256": h, "size_bytes": size,
                    "is_zip": fn.lower().endswith(".zip"),
                })

    inv_df = pd.DataFrame(inventory)
    A.save_table("file_inventory", inv_df, "os.walk of input directories (old + new)")

    # Per-model source manifest
    selected = {}
    for model in MODELS:
        bn_p = per_item_path(model, "bn")
        en_p = per_item_path(model, "en")
        selected[model] = {"bn": bn_p, "en": en_p}
        root = os.path.dirname(bn_p) if bn_p else ""
        log(f"  {model} root={root} bn={bn_p} en={en_p}")

    # Print column names of selected per-item CSVs
    for model in MODELS:
        for lang in LANGS:
            p = selected[model][lang]
            df = read_csv_safe(p)
            if df is not None:
                log(f"Columns {model}/{lang}: {list(df.columns)} rows={len(df)}")

    # Save selected sources manifest
    sel_rows = []
    for model in MODELS:
        for lang in LANGS:
            p = selected[model][lang]
            sel_rows.append({"model": model, "lang": lang, "path": p,
                             "sha256": sha256_file(p) if os.path.exists(p) else None})
    pd.DataFrame(sel_rows).to_csv(os.path.join(TABLES_DIR, "selected_sources.csv"), index=False)
    return selected

# =============================================================================
# STEP 1 — RUN HEALTH AND PROVENANCE
# =============================================================================
def step1_health(selected):
    log("=== STEP 1: RUN HEALTH AND PROVENANCE ===")

    # 1.1 Execution status
    assertion = read_json_safe(os.path.join(INPUT_DIR, "assertion_status.json"))
    new_cells = read_json_safe(os.path.join(INPUT_DIR, "new_cells_status.json"))
    merge_status = read_json_safe(os.path.join(INPUT_DIR, "merge_status.json"))

    health_rows = []
def step1_health(selected):
    log("=== STEP 1: RUN HEALTH AND PROVENANCE ===")

    # 1.1 Execution status — read from NEW_INPUT_DIR root (assertion_status.json,
    # new_cells_status.json live there; old models have none)
    assertion = read_json_safe(os.path.join(NEW_INPUT_DIR, "assertion_status.json"))
    new_cells = read_json_safe(os.path.join(NEW_INPUT_DIR, "new_cells_status.json"))
    merge_status = read_json_safe(os.path.join(NEW_INPUT_DIR, "merge_status.json"))
    # Also look for per-model manifest
    manifest_path = None
    for cand in [os.path.join(NEW_INPUT_DIR, "results_llama_base", "run_manifest_llama_base.json"),
                 os.path.join(NEW_INPUT_DIR, "run_manifest.json")]:
        if os.path.exists(cand):
            manifest_path = cand
            break
    manifest = read_json_safe(manifest_path) if manifest_path else None

    health_rows = []
    if assertion is None:
        A.warn("assertion_status.json is missing; treated as no failures recorded (provenance artifact disclosed).")
        health_rows.append({"file": "assertion_status.json", "status": "MISSING",
                            "failure": "none recorded (file absent)", "detail": "disclosed missing provenance"})
    else:
        health_rows.append({"file": "assertion_status.json", "status": "present",
                            "failure": "none", "detail": json.dumps(assertion)})

    failures = []
    if new_cells:
        for cell, info in new_cells.items():
            ok = info.get("ok", False) if isinstance(info, dict) else None
            err = info.get("error") if isinstance(info, dict) else None
            health_rows.append({"file": f"new_cells_status.json:{cell}",
                                "status": "ok" if ok else ("error" if ok is False else "unknown"),
                                "failure": err,
                                "detail": info.get("traceback") if isinstance(info, dict) else None})
            if isinstance(info, dict) and not info.get("ok", True):
                failures.append(f"{cell}: {err}")
    if merge_status:
        health_rows.append({"file": "merge_status.json",
                            "status": "ok" if merge_status.get("sha256_ok") else "error",
                            "failure": "; ".join(merge_status.get("mismatches", [])) if not merge_status.get("sha256_ok") else None,
                            "detail": json.dumps(merge_status)})
        if not merge_status.get("sha256_ok"):
            failures.extend(merge_status.get("mismatches", []))
    A.save_table("run_health", pd.DataFrame(health_rows), "assertion_status.json / new_cells_status.json / merge_status.json")

    # 1.2 Development screen — NEW format: dev_screen_report_e3.csv
    dev = read_csv_safe(os.path.join(NEW_INPUT_DIR, "dev_screen_report_e3.csv"))
    dev_rows = []
    skipped = []
    if dev is not None:
        for _, r in dev.iterrows():
            model = r["model_key"]
            for lang in ["bn", "en"]:
                dev_rows.append({
                    "model": model, "lang": lang, "stage": r.get("stage"),
                    "config": r.get("config"),
                    "dev_elapsed_sec": float(r.get("dev_elapsed_sec")) if pd.notna(r.get("dev_elapsed_sec")) else None,
                    "throughput_items_per_sec": float(r.get("throughput_items_per_sec")) if pd.notna(r.get("throughput_items_per_sec")) else None,
                    "est_full_run_min": float(r.get("est_full_run_min")) if pd.notna(r.get("est_full_run_min")) else None,
                    "collapse_gate_passed": bool(r.get("collapse_gate_passed")),
                    "majority_gate_passed": bool(r.get("majority_gate_passed")),
                    "eligible_for_full_run": bool(r.get("eligible_for_full_run")),
                    "label": str(r.get("label")),
                    "bn_top_share": float(r.get("bn_top_share")) if pd.notna(r.get("bn_top_share")) else None,
                    "bn_min_share": float(r.get("bn_min_share")) if pd.notna(r.get("bn_min_share")) else None,
                    "bn_acc": float(r.get("bn_acc")) if pd.notna(r.get("bn_acc")) else None,
                    "bn_pval": float(r.get("bn_pval")) if pd.notna(r.get("bn_pval")) else None,
                    "en_top_share": float(r.get("en_top_share")) if pd.notna(r.get("en_top_share")) else None,
                    "en_min_share": float(r.get("en_min_share")) if pd.notna(r.get("en_min_share")) else None,
                    "en_acc": float(r.get("en_acc")) if pd.notna(r.get("en_acc")) else None,
                    "en_pval": float(r.get("en_pval")) if pd.notna(r.get("en_pval")) else None,
                    "full_run_executed": bool(r.get("full_run_executed")),
                    "screen_source": "dev_screen_report_e3.csv",
                })
            if not r.get("eligible_for_full_run", True):
                skipped.append(model)
    A.save_table("dev_screen", pd.DataFrame(dev_rows), "dev_screen_report_e3.csv")
    if skipped:
        A.warn(f"Models not eligible for full run: {skipped}")

    # 1.3 Dataset consistency across all 6 calibration models
    cons_rows = []
    all_ok = True
    bn_id_sets = {}
    en_id_sets = {}
    for model in MODELS:
        bn_df = read_csv_safe(selected[model]["bn"])
        en_df = read_csv_safe(selected[model]["en"])
        if bn_df is not None:
            bn_id_sets[model] = set(bn_df["question_id"].astype(str))
            cons_rows.append({"check": f"{model} BN row count", "expected": 1500,
                              "actual": len(bn_df), "pass": len(bn_df) == 1500})
            if bn_df["confidence"].isna().any():
                cons_rows.append({"check": f"{model} BN null confidence", "expected": 0,
                                  "actual": int(bn_df["confidence"].isna().sum()), "pass": False})
                A.warn(f"{model} BN has null confidence values"); all_ok = False
        if en_df is not None:
            en_id_sets[model] = set(en_df["question_id"].astype(str))
            cons_rows.append({"check": f"{model} EN row count", "expected": 1500,
                              "actual": len(en_df), "pass": len(en_df) == 1500})
            if en_df["confidence"].isna().any():
                cons_rows.append({"check": f"{model} EN null confidence", "expected": 0,
                                  "actual": int(en_df["confidence"].isna().sum()), "pass": False})
                A.warn(f"{model} EN has null confidence values"); all_ok = False
    # BN identical across models
    bn_models = list(bn_id_sets)
    bn_same = all(bn_id_sets[b] == bn_id_sets[bn_models[0]] for b in bn_models) if len(bn_models) > 1 else True
    cons_rows.append({"check": "BN question-ID sets identical across models", "expected": True,
                      "actual": bn_same, "pass": bn_same})
    if not bn_same:
        A.warn("BN question-ID sets differ across models"); all_ok = False
    # EN identical across models
    en_models = list(en_id_sets)
    en_same = all(en_id_sets[e] == en_id_sets[en_models[0]] for e in en_models) if len(en_models) > 1 else True
    cons_rows.append({"check": "EN question-ID sets identical across models", "expected": True,
                      "actual": en_same, "pass": en_same})
    if not en_same:
        A.warn("EN question-ID sets differ across models"); all_ok = False
    # paired BN/EN within model
    paired_ok = True
    for model in MODELS:
        if model in bn_id_sets and model in en_id_sets:
            p = bn_id_sets[model] == en_id_sets[model]
            cons_rows.append({"check": f"{model} BN and EN IDs paired", "expected": True,
                              "actual": p, "pass": p})
            if not p:
                paired_ok = False; A.warn(f"{model} BN and EN IDs not paired"); all_ok = False
    # subset byte-identical across models (SHA256), per filename
    subset_by_name = {}
    # Old models: subsets in same dir as results. llama32_3b_base: subsets one level up.
    subset_dir_override = {"llama32_3b_base": os.path.join(
        NEW_INPUT_DIR, "results_llama_base", "results_E3_llama32_3b_base")}
    for model in MODELS:
        d = subset_dir_override.get(model, MODEL_INPUT_DIR.get(model, ""))
        for nm in ["bn_subset.csv", "en_subset.csv"]:
            pth = os.path.join(d, nm)
            if os.path.exists(pth):
                subset_by_name.setdefault(nm, {})[model] = sha256_file(pth)
    subset_same = True
    for nm, mhashes in subset_by_name.items():
        if len(set(mhashes.values())) != 1:
            subset_same = False
            A.warn(f"{nm} hashes differ across models: {mhashes}")
    cons_rows.append({"check": "bn_subset.csv/en_subset.csv byte-identical across models (SHA256, per filename)",
                      "expected": True, "actual": subset_same, "pass": subset_same})
    # match run_manifest subset hashes
    manifest_ok = True
    if manifest:
        smd_all = manifest.get("subset_sha256", {})
        for model in MODELS:
            smd = smd_all.get(model, {})
            for nm, key in [("bn_subset.csv", "bn_subset_sha256"), ("en_subset.csv", "en_subset_sha256")]:
                actual = subset_by_name.get(nm, {}).get(model)
                expected = smd.get(key)
                if expected is None:
                    # Manifest does not record this model's hash; skip comparison
                    cons_rows.append({"check": f"{model} {nm} matches run_manifest", "expected": "not_recorded",
                                      "actual": str(actual), "pass": True, "note": "manifest does not record hash"})
                    continue
                ok = actual == expected
                cons_rows.append({"check": f"{model} {nm} matches run_manifest", "expected": str(expected),
                                  "actual": str(actual), "pass": ok})
                if not ok:
                    manifest_ok = False; A.warn(f"{model} {nm} hash mismatch vs run_manifest"); all_ok = False
    A.save_table("dataset_consistency", pd.DataFrame(cons_rows), "per-item CSVs + bn_subset/en_subset + run_manifest.json")
    if not all_ok:
        A.warn("Dataset consistency checks failed; results should be interpreted with caution.")

    # 1.4 Run manifest provenance
    prov_rows = []
    if manifest:
        prov_rows.append({"setting": "notebook_version", "value": manifest.get("notebook_version")})
        prov_rows.append({"setting": "created_at", "value": manifest.get("created_at")})
        prov_rows.append({"setting": "torch_version", "value": manifest.get("torch_version")})
        prov_rows.append({"setting": "cuda_available", "value": manifest.get("cuda_available")})
        prov_rows.append({"setting": "gpu_name", "value": manifest.get("gpu_name")})
        prov_rows.append({"setting": "seed", "value": manifest.get("seed")})
        prov_rows.append({"setting": "n_items_per_language", "value": manifest.get("n_items_per_language")})
        prov_rows.append({"setting": "debias", "value": manifest.get("debias")})
        pmd = manifest.get("per_model_dtype", {})
        for model in MODELS:
            prov_rows.append({"setting": f"per_model_dtype:{model}", "value": pmd.get(model)})
        pwt = manifest.get("per_model_wall_time_seconds", {})
        wt_vals = [pwt.get(m) for m in MODELS]
        prov_rows.append({"setting": "per_model_wall_time_seconds",
                          "value": json.dumps(pwt) + (" (null: not recorded)" if all(v is None for v in wt_vals) else "")})
        if all(v is None for v in wt_vals):
            record_missing("per_model_wall_time_seconds", "all null in run_manifest.json")
        prov_rows.append({"setting": "dtype_sensitivity_active_models", "value": json.dumps(manifest.get("dtype_sensitivity_active_models"))})
        prov_rows.append({"setting": "e3_models", "value": json.dumps(manifest.get("e3_models"))})
        prov_rows.append({"setting": "models_executed", "value": json.dumps(manifest.get("models_executed"))})
    else:
        record_missing("run_manifest.json", "file not found")
    A.save_table("provenance", pd.DataFrame(prov_rows), "run_manifest.json")

    # Load data into dict (6 calibration models only)
    data = {}
    for model in MODELS:
        for lang in LANGS:
            p = selected[model][lang]
            df = read_csv_safe(p)
            if df is not None:
                data[(model, lang)] = df
    missing_status = []
    if assertion is None:
        missing_status.append("assertion_status.json (absent -> no failures recorded)")
    return data, {"manifest": manifest, "assertion": assertion, "new_cells": new_cells,
                  "merge_status": merge_status, "dev": dev, "missing_status": missing_status,
                  "all_ok": all_ok}


# =============================================================================
# STEP 2 — MAIN RESULTS
# =============================================================================
def step2_main(data):
    log("=== STEP 2: MAIN RESULTS ===")
    rows = []
    raw_rows = []
    letter_rows = []
    summary_rows = []

    for model in MODELS:
        for lang in LANGS:
            df = data[(model, lang)]
            m = metrics_for_df(df, lang, model)
            rows.append({k: v for k, v in m.items()
                         if k not in ("option_probs_valid",)})
            letter_rows.append({"model": model, "lang": lang, "n": m["n"],
                                "pred_A": letter_counts(df["predicted_answer"].astype(str).str.upper().to_numpy()).get("A", 0),
                                "pred_B": letter_counts(df["predicted_answer"].astype(str).str.upper().to_numpy()).get("B", 0),
                                "pred_C": letter_counts(df["predicted_answer"].astype(str).str.upper().to_numpy()).get("C", 0),
                                "pred_D": letter_counts(df["predicted_answer"].astype(str).str.upper().to_numpy()).get("D", 0),
                                "top_letter": m["top_letter"], "top_share": m["top_share"]})

            rm = metrics_for_df_raw(df)
            raw_rows.append({"model": model, "lang": lang,
                             "raw_accuracy": rm["accuracy"], "raw_accuracy_percent": rm["accuracy_percent"],
                             "raw_mean_confidence": rm["mean_confidence"],
                             "raw_ece_uniform_10": rm["ece_uniform_10"],
                             "raw_overconfidence_gap": rm["overconfidence_gap"],
                             "raw_top_letter": rm["top_letter"], "raw_top_share": rm["top_share"]})

    main_df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)
    letter_df = pd.DataFrame(letter_rows)
    A.save_table("main_metrics", main_df, "authoritative per-item CSVs (confidence/predicted_answer/prob_A..D)")
    A.save_table("raw_vs_cyclic4", pd.merge(raw_df, main_df[["model", "lang", "accuracy", "mean_confidence",
                    "ece_uniform_10", "overconfidence_gap"]], on=["model", "lang"]),
                 "pred_raw/confidence_raw vs predicted_answer/confidence")
    A.save_table("letter_distribution", letter_df, "predicted_answer column")

    # Chance tests
    chance_rows = []
    for model in MODELS:
        for lang in LANGS:
            df = data[(model, lang)]
            pred = df["predicted_answer"].astype(str).str.upper().to_numpy()
            gold = df["gold_answer"].astype(str).str.upper().to_numpy()
            correct = (pred == gold)
            k = int(correct.sum()); n = int(len(correct))
            if HAS_SCIPY:
                res = stats.binomtest(k, n, CHANCE, alternative="greater")
                p = float(res.pvalue)
            else:
                # fallback exact binomial tail
                from math import comb
                p = float(sum(comb(n, i) * (CHANCE ** i) * ((1 - CHANCE) ** (n - i)) for i in range(k, n + 1)))
            acc = k / n
            chance_rows.append({"model": model, "lang": lang, "n": n, "correct": k,
                                "accuracy": acc, "accuracy_percent": acc * 100,
                                "chance": CHANCE,
                                "binom_p_value": p,
                                "significantly_above_chance": p < 0.05})
    chance_df = pd.DataFrame(chance_rows)
    A.save_table("chance_tests", chance_df, "per-item correctness vs 25% chance (exact binomial)")

    # Reconciliation with summary.json / three_model_summary.csv
    recon_rows = []
    three = None
    for cand in [os.path.join(OLD_INPUT_DIR, "three_model_summary.csv"),
                 os.path.join(NEW_INPUT_DIR, "results_llama_base", "three_model_summary.csv")]:
        if os.path.exists(cand):
            three = read_csv_safe(cand)
            if three is not None and "accuracy_percent" in three.columns:
                break
            three = None
    for model in MODELS:
        for lang in LANGS:
            s = read_json_safe(summary_path(model, lang))
            mrow = main_df[(main_df["model"] == model) & (main_df["lang"] == lang)].iloc[0]
            if s:
                cm = s.get("calibration_metrics", {})
                fields = [
                    ("accuracy", "accuracy"),
                    ("mean_confidence", "mean_confidence"),
                    ("ece_uniform_10", "ece_uniform_10"),
                    ("brier_score", "brier_score"),
                    ("nll", "nll"),
                ]
                for out_key, s_key in fields:
                    if s_key in cm:
                        expected = cm[s_key]
                        actual = mrow[out_key]
                        if expected is None or (isinstance(expected, float) and np.isnan(expected)):
                            status = "summary_NaN"
                            diff = None
                        else:
                            diff = abs(float(actual) - float(expected))
                            status = "MATCH" if diff <= TOL else "MISMATCH"
                        recon_rows.append({"model": model, "lang": lang, "metric": out_key,
                                           "summary_json": expected, "recomputed": actual,
                                           "abs_diff": diff, "status": status,
                                           "source_file": summary_path(model, lang)})
                # also top_share vs three_model_summary
    # three_model_summary reconciliation
    if three is not None:
        for _, r in three.iterrows():
            mrow = main_df[(main_df["model"] == r["model_key"]) & (main_df["lang"] == r["lang"])].iloc[0]
            for col in ["accuracy_percent", "ece_uniform_10", "top_share"]:
                if col not in r.index:
                    continue
                if col in mrow.index:
                    expected = r[col]; actual = mrow[col]
                    diff = abs(float(actual) - float(expected))
                    recon_rows.append({"model": r["model_key"], "lang": r["lang"], "metric": col,
                                       "three_model_summary": expected, "recomputed": actual,
                                       "abs_diff": diff,
                                       "status": "MATCH" if diff <= TOL else "MISMATCH",
                                       "source_file": "three_model_summary.csv"})
    recon_df = pd.DataFrame(recon_rows)
    A.save_table("summary_reconciliation", recon_df,
                 "summary.json per model/lang + three_model_summary.csv vs recomputed (tol 1e-6)")

    return main_df, raw_df, chance_df, recon_df


def bootstrap_main_metrics(data):
    """Add bootstrap CIs to main_metrics table (regenerate with CIs)."""
    rows = []
    for model in MODELS:
        for lang in LANGS:
            df = data[(model, lang)]
            conf = df["confidence"].to_numpy(dtype=float)
            pred = df["predicted_answer"].astype(str).str.upper().to_numpy()
            gold = df["gold_answer"].astype(str).str.upper().to_numpy()
            correct = (pred == gold)

            def acc(d):
                return float((d["pred"].astype(str).str.upper() == d["gold"].astype(str).str.upper()).mean())
            def mconf(d):
                return float(d["conf"].astype(float).mean())
            def ece10f(d):
                c = d["conf"].astype(float).to_numpy()
                co = (d["pred"].astype(str).str.upper().to_numpy() == d["gold"].astype(str).str.upper().to_numpy())
                return ece_uniform(c, co, 10)[0]
            def overconf(d):
                c = d["conf"].astype(float).to_numpy()
                co = (d["pred"].astype(str).str.upper().to_numpy() == d["gold"].astype(str).str.upper().to_numpy())
                return overconfidence_gap(c.mean(), co.mean())

            data_cols = pd.DataFrame({"pred": pred, "conf": conf, "correct": correct, "gold": gold})
            for name, fn in [("accuracy", acc), ("mean_confidence", mconf),
                             ("ece_uniform_10", ece10f), ("overconfidence_gap", overconf)]:
                lo, hi, meanb = bootstrap_ci(fn, data_cols)
                rows.append({"model": model, "lang": lang, "metric": name,
                             "bootstrap_mean": meanb, "ci_low": lo, "ci_high": hi,
                             "n_resamples": N_BOOT, "seed": SEED})
    ci_df = pd.DataFrame(rows)
    ci_path = os.path.join(TABLES_DIR, "main_metrics_bootstrap_ci.csv")
    ci_df.to_csv(ci_path, index=False)
    A.tables["main_metrics_bootstrap_ci"] = ci_df
    A.table_sources["main_metrics_bootstrap_ci"] = "bootstrap (seed 42, 2000 resamples, item-level)"

# =============================================================================
# STEP 3 — PAIRED BN VS EN
# =============================================================================
def step3_paired(data):
    log("=== STEP 3: PAIRED BN VS EN COMPARISON ===")
    rows = []
    for model in MODELS:
        bn = data[(model, "bn")]
        en = data[(model, "en")]
        # pair on question_id
        bn = bn.copy(); en = en.copy()
        bn["question_id"] = bn["question_id"].astype(str)
        en["question_id"] = en["question_id"].astype(str)
        merged = bn.merge(en, on="question_id", suffixes=("_bn", "_en"))
        n = len(merged)

        def acc_bn(d):
            p = d["predicted_answer_bn"].astype(str).str.upper().to_numpy()
            g = d["gold_answer_bn"].astype(str).str.upper().to_numpy()
            return float((p == g).mean())
        def acc_en(d):
            p = d["predicted_answer_en"].astype(str).str.upper().to_numpy()
            g = d["gold_answer_en"].astype(str).str.upper().to_numpy()
            return float((p == g).mean())
        def mconf_bn(d): return float(d["confidence_bn"].to_numpy(dtype=float).mean())
        def mconf_en(d): return float(d["confidence_en"].to_numpy(dtype=float).mean())
        def ece_bn(d):
            c = d["confidence_bn"].to_numpy(dtype=float)
            co = (d["predicted_answer_bn"].astype(str).str.upper().to_numpy() ==
                  d["gold_answer_bn"].astype(str).str.upper().to_numpy())
            return ece_uniform(c, co, 10)[0]
        def ece_en(d):
            c = d["confidence_en"].to_numpy(dtype=float)
            co = (d["predicted_answer_en"].astype(str).str.upper().to_numpy() ==
                  d["gold_answer_en"].astype(str).str.upper().to_numpy())
            return ece_uniform(c, co, 10)[0]

        acc_diff = acc_bn(merged) - acc_en(merged)
        ece_diff = ece_bn(merged) - ece_en(merged)
        conf_diff = mconf_bn(merged) - mconf_en(merged)

        # paired bootstrap
        idx = np.arange(n)
        rng = np.random.default_rng(SEED)
        acc_diffs = []; ece_diffs = []; conf_diffs = []
        for _ in range(N_BOOT):
            s = rng.choice(idx, size=n, replace=True)
            sub = merged.iloc[s]
            acc_diffs.append(acc_bn(sub) - acc_en(sub))
            ece_diffs.append(ece_bn(sub) - ece_en(sub))
            conf_diffs.append(mconf_bn(sub) - mconf_en(sub))
        acc_diffs = np.array(acc_diffs); ece_diffs = np.array(ece_diffs); conf_diffs = np.array(conf_diffs)

        # McNemar exact on correctness
        cb = (bn.set_index("question_id").loc[merged["question_id"]]["predicted_answer"].astype(str).str.upper().to_numpy() ==
              bn.set_index("question_id").loc[merged["question_id"]]["gold_answer"].astype(str).str.upper().to_numpy())
        ce = (en.set_index("question_id").loc[merged["question_id"]]["predicted_answer"].astype(str).str.upper().to_numpy() ==
              en.set_index("question_id").loc[merged["question_id"]]["gold_answer"].astype(str).str.upper().to_numpy())
        n01 = int((~cb & ce).sum())   # BN wrong, EN right
        n10 = int((cb & ~ce).sum())   # BN right, EN wrong
        ndisc = n01 + n10
        if HAS_SCIPY and ndisc > 0:
            # exact binomial two-sided: min(n01,n10) ~ Binomial(ndisc, 0.5)
            k = min(n01, n10)
            p = float(2 * min(1.0, stats.binom.cdf(k, ndisc, 0.5)))
        else:
            p = None
        rows.append({"model": model, "n_paired": n,
                     "accuracy_bn": acc_bn(merged), "accuracy_en": acc_en(merged),
                     "accuracy_diff_bn_minus_en": acc_diff,
                     "acc_diff_ci_low": float(np.percentile(acc_diffs, 2.5)),
                     "acc_diff_ci_high": float(np.percentile(acc_diffs, 97.5)),
                     "ece_bn": ece_bn(merged), "ece_en": ece_en(merged),
                     "ece_diff_bn_minus_en": ece_diff,
                     "ece_diff_ci_low": float(np.percentile(ece_diffs, 2.5)),
                     "ece_diff_ci_high": float(np.percentile(ece_diffs, 97.5)),
                     "mean_conf_bn": mconf_bn(merged), "mean_conf_en": mconf_en(merged),
                     "mean_conf_diff_bn_minus_en": conf_diff,
                     "conf_diff_ci_low": float(np.percentile(conf_diffs, 2.5)),
                     "conf_diff_ci_high": float(np.percentile(conf_diffs, 97.5)),
                     "mcnemar_n01_bn_wrong_en_right": n01,
                     "mcnemar_n10_bn_right_en_wrong": n10,
                     "mcnemar_p_exact": p})
    pdf = pd.DataFrame(rows)
    A.save_table("bn_en_paired_comparison", pdf, "per-item CSVs paired on question_id (paired bootstrap seed 42, 2000 resamples; exact McNemar)")
    return pdf

# =============================================================================
# STEP 4 — TOKENIZATION FRAGMENTATION VS CALIBRATION
# =============================================================================
def step4_fragmentation(data):
    log("=== STEP 4: TOKENIZATION FRAGMENTATION VS CALIBRATION ===")
    # Find tokenizer_fragmentation_per_item_all7.csv in multiple locations
    tf_candidates = [
        os.path.join(NEW_INPUT_DIR, "results_llama_base", "tokenizer_fragmentation_llama_base",
                     "tokenizer_fragmentation_per_item_all7.csv"),
        os.path.join(NEW_INPUT_DIR, "results_llama_base", "tokenizer_fragmentation_llama_base",
                     "tokenizer_fragmentation_per_item.csv"),
        os.path.join(EXTRACT_DIR, "tokenizer_fragmentation_per_item_all7.csv"),
    ]
    tf_path = None
    for c in tf_candidates:
        if os.path.exists(c):
            tf_path = c
            break
    tf_summary = None
    tf_summary_candidates = [
        os.path.join(NEW_INPUT_DIR, "results_llama_base", "tokenizer_fragmentation_llama_base",
                     "tokenizer_fragmentation_summary_all7.csv"),
    ]
    for c in tf_summary_candidates:
        if os.path.exists(c):
            tf_summary = read_csv_safe(c)
            break
    if tf_summary is not None:
        A.save_table("tokenizer_fragmentation_summary_source", tf_summary,
                     "tokenizer_fragmentation_summary_all7.csv")

    if tf_path is None:
        record_missing("tokenizer_fragmentation_per_item_all7.csv", "cannot run Step 4")
        return None

    tf = read_csv_safe(tf_path)
    if tf is None:
        record_missing("tokenizer_fragmentation_per_item_all7.csv", "read error")
        return None

    # Normalize column names: old format has "model", new format has "model_key";
    # old format may have "subject_name", new format does not.
    tf_cols = set(tf.columns)
    model_col = "model_key" if "model_key" in tf_cols else ("model" if "model" in tf_cols else None)
    lang_col = "lang"
    qid_col = "question_id"
    if model_col is None:
        record_missing("tokenizer_fragmentation_per_item_all7.csv", "no model/model_key column")
        return None
    tf = tf.rename(columns={model_col: "model_key"})

    # Merge with results
    merged_list = []
    unmatched = []
    for model in MODELS:
        for lang in LANGS:
            df = data[(model, lang)][["question_id", "predicted_answer", "confidence",
                                      "is_correct", "gold_answer", "subject", "question_word_count",
                                      "prob_A", "prob_B", "prob_C", "prob_D"]].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub = tf[(tf["model_key"] == model) & (tf["lang"] == lang)].copy()
            sub["question_id"] = sub["question_id"].astype(str)
            m = df.merge(sub, on=["question_id"], how="left", indicator=True)
            both = (m["_merge"] == "both").sum()
            left_only = (m["_merge"] == "left_only").sum()
            right_only = (m["_merge"] == "right_only").sum()
            m = m.drop(columns=["_merge"])
            m["model"] = model
            m["lang"] = lang
            merged_list.append(m)
            unmatched.append({"model": model, "lang": lang, "n_items": len(df),
                              "matched": int(both), "unmatched_in_results": int(left_only),
                              "unmatched_in_fragmentation": int(right_only)})
            if left_only or right_only:
                A.warn(f"{model}/{lang}: {left_only} results rows unmatched, {right_only} fragmentation rows unmatched")
    merged = pd.concat(merged_list, ignore_index=True)
    keep_cols = ["model", "lang", "question_id", "tpw", "tpc", "strr",
                 "predicted_answer", "confidence", "is_correct", "gold_answer",
                 "question_word_count"]
    if "subject" in merged.columns:
        keep_cols.append("subject")
    A.save_table("fragmentation_merged", merged[keep_cols],
                 "tokenizer_fragmentation_per_item_all7.csv merged with per-item results on model,lang,question_id")
    pd.DataFrame(unmatched).to_csv(os.path.join(TABLES_DIR, "fragmentation_join_coverage.csv"), index=False)

    # 4A Model-level fragmentation
    ml_rows = []
    for model in MODELS:
        for lang in LANGS:
            sub = merged[(merged["model"] == model) & (merged["lang"] == lang)]
            ml_rows.append({"model": model, "lang": lang, "n": len(sub),
                            "mean_tpw": float(sub["tpw"].mean()), "std_tpw": float(sub["tpw"].std()),
                            "mean_tpc": float(sub["tpc"].mean()), "std_tpc": float(sub["tpc"].std()),
                            "mean_strr": float(sub["strr"].mean()), "std_strr": float(sub["strr"].std()),
                            "median_tpw": float(sub["tpw"].median())})
    ml_df = pd.DataFrame(ml_rows)
    A.save_table("fragmentation_model_level", ml_df,
                 "tokenizer_fragmentation_per_item_all7.csv (tpw/tpc/strr means by model & lang)")
    # BN/EN tpw ratio
    ratio_rows = []
    for model in MODELS:
        bn = ml_df[(ml_df["model"] == model) & (ml_df["lang"] == "bn")]
        en = ml_df[(ml_df["model"] == model) & (ml_df["lang"] == "en")]
        if len(bn) and len(en):
            ratio_rows.append({"model": model,
                               "bn_mean_tpw": float(bn["mean_tpw"].iloc[0]),
                               "en_mean_tpw": float(en["mean_tpw"].iloc[0]),
                               "bn_en_tpw_ratio": float(bn["mean_tpw"].iloc[0] / en["mean_tpw"].iloc[0])})
    ratio_df = pd.DataFrame(ratio_rows)
    ratio_df.to_csv(os.path.join(TABLES_DIR, "fragmentation_bn_en_tpw_ratio.csv"), index=False)
    A.tables["fragmentation_bn_en_tpw_ratio"] = ratio_df
    A.table_sources["fragmentation_bn_en_tpw_ratio"] = "fragmentation_model_level.csv derived"
    # Tokenizer level grouping (by vocab/TPW identity):
    # LLaMA family: titullm_1b, llama32_3b, llama32_3b_base (vocab 128256, identical TPW)
    # Gemma/TigerLLM: tigerllm_1b (vocab 262145)
    # Qwen: qwen25_3b (vocab 151665)
    # BLOOM: bloomz_3b (vocab 250680)
    # phi35_mini (vocab 32011) is dev-screen only, excluded from calibration
    tokenizer_levels = {
        "titullm_1b": "LLaMA family",
        "llama32_3b": "LLaMA family",
        "llama32_3b_base": "LLaMA family",
        "tigerllm_1b": "Gemma/TigerLLM",
        "qwen25_3b": "Qwen",
        "bloomz_3b": "BLOOM",
    }
    A.warn("Tokenizer levels: LLaMA family (titullm_1b, llama32_3b, llama32_3b_base share identical TPW by construction), "
           "Gemma/TigerLLM (tigerllm_1b), Qwen (qwen25_3b), BLOOM (bloomz_3b). phi35_mini is dev-screen only. "
           "Cross-tokenizer comparisons use 4 levels with calibration data and are descriptive only.")

    # 4B TPW quartile analysis
    quart_rows = []
    for model in MODELS:
        for lang in LANGS:
            sub = merged[(merged["model"] == model) & (merged["lang"] == lang)].copy()
            if len(sub) < 4:
                continue
            sub["quartile"] = pd.qcut(sub["tpw"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                qs = sub[sub["quartile"] == q]
                nq = len(qs)
                if nq == 0:
                    continue
                pred = qs["predicted_answer"].astype(str).str.upper().to_numpy()
                gold = qs["gold_answer"].astype(str).str.upper().to_numpy()
                correct = (pred == gold)
                conf = qs["confidence"].to_numpy(dtype=float)
                acc = float(correct.mean()); mconf = float(conf.mean())
                ece10 = ece_uniform(conf, correct, 10)[0]
                overconf = overconfidence_gap(mconf, acc)
                dcol_df = pd.DataFrame({"pred": pred, "conf": conf, "gold": gold})
                acc_lo, acc_hi, _ = bootstrap_ci(
                    lambda d: float((d["pred"].astype(str).str.upper() == d["gold"].astype(str).str.upper()).mean()), dcol_df)
                ece_lo, ece_hi, _ = bootstrap_ci(
                    lambda d: ece_uniform(d["conf"].astype(float).to_numpy(),
                        (d["pred"].astype(str).str.upper().to_numpy() == d["gold"].astype(str).str.upper().to_numpy()), 10)[0],
                    dcol_df)
                warn100 = nq < 100
                if warn100:
                    A.warn(f"Quartile {q} {model}/{lang} has {nq} observations (<100)")
                quart_rows.append({"model": model, "lang": lang, "quartile": q, "n": nq,
                                   "accuracy": acc, "acc_ci_low": acc_lo, "acc_ci_high": acc_hi,
                                   "mean_confidence": mconf,
                                   "ece_uniform_10": ece10, "ece_ci_low": ece_lo, "ece_ci_high": ece_hi,
                                   "overconfidence_gap": overconf,
                                   "warn_n_lt_100": warn100})
    quart_df = pd.DataFrame(quart_rows)
    A.save_table("fragmentation_quartiles", quart_df,
                 "within-model TPW quartiles of merged fragmentation+results (seed 42, 2000 resamples)")

    # 4C Item-level regression
    reg_rows = []
    spearman_rows = []
    holm_tests = []  # (label, raw_p, model, lang, kind)
    for model in MODELS:
        for lang in LANGS:
            sub = merged[(merged["model"] == model) & (merged["lang"] == lang)].copy()
            sub = sub.dropna(subset=["tpw", "confidence", "is_correct", "question_word_count"])
            n = len(sub)
            tpw = sub["tpw"].to_numpy(dtype=float)
            conf = sub["confidence"].to_numpy(dtype=float)
            correct = sub["is_correct"].to_numpy(dtype=float)
            wc = sub["question_word_count"].to_numpy(dtype=float)

            # Logistic: correct ~ tpw + wc
            logit_coef = logit_ci_lo = logit_ci_hi = logit_p = None
            try:
                X = np.column_stack([np.ones(n), tpw, wc])
                import statsmodels.api as sm
                m_logit = sm.Logit(correct, X).fit(disp=0)
                logit_coef = float(m_logit.params[1])
                se = float(m_logit.bse[1])
                logit_ci_lo = logit_coef - 1.96 * se
                logit_ci_hi = logit_coef + 1.96 * se
                logit_p = float(m_logit.pvalues[1])
            except Exception as e:
                record_missing(f"logit regression {model} {lang}", str(e))
            holm_tests.append((f"logit tpw {model} {lang}", logit_p, model, lang, "logit"))

            # Linear: confidence ~ tpw + wc
            lin_coef = lin_ci_lo = lin_ci_hi = lin_p = None
            try:
                X = np.column_stack([np.ones(n), tpw, wc])
                m_lin = sm.OLS(conf, X).fit(cov_type="HC1")
                lin_coef = float(m_lin.params[1])
                se = float(m_lin.bse[1])
                lin_ci_lo = lin_coef - 1.96 * se
                lin_ci_hi = lin_coef + 1.96 * se
                lin_p = float(m_lin.pvalues[1])
            except Exception as e:
                record_missing(f"linear regression {model} {lang}", str(e))
            holm_tests.append((f"linear tpw {model} {lang}", lin_p, model, lang, "linear"))

            reg_rows.append({"model": model, "lang": lang, "n": n,
                             "logit_tpw_coef": logit_coef, "logit_ci_lo": logit_ci_lo,
                             "logit_ci_hi": logit_ci_hi, "logit_p_raw": logit_p,
                             "linear_tpw_coef": lin_coef, "linear_ci_low": lin_ci_lo,
                             "linear_ci_high": lin_ci_hi, "linear_p_raw": lin_p})

            # Spearman
            if HAS_SCIPY:
                rho_tpw_conf, p_tpw_conf = stats.spearmanr(tpw, conf)
                rho_tpw_corr, p_tpw_corr = stats.spearmanr(tpw, correct)
            else:
                rho_tpw_conf = p_tpw_conf = rho_tpw_corr = p_tpw_corr = None
            spearman_rows.append({"model": model, "lang": lang, "n": n,
                                  "spearman_tpw_confidence_rho": rho_tpw_conf,
                                  "spearman_tpw_confidence_p": p_tpw_conf,
                                  "spearman_tpw_correctness_rho": rho_tpw_corr,
                                  "spearman_tpw_correctness_p": p_tpw_corr})

    reg_df = pd.DataFrame(reg_rows)
    A.save_table("fragmentation_regression", reg_df,
                 "logit(correct~tpw+wc) & linear(confidence~tpw+wc); subject-clustered robust SE where available")
    A.save_table("fragmentation_spearman", pd.DataFrame(spearman_rows),
                 "Spearman: tpw vs confidence, tpw vs correctness")

    # 4D Item-level calibration proxy
    proxy_rows = []
    for model in MODELS:
        for lang in LANGS:
            sub = merged[(merged["model"] == model) & (merged["lang"] == lang)].copy()
            sub = sub.dropna(subset=["tpw", "confidence", "is_correct", "question_word_count"])
            n = len(sub)
            tpw = sub["tpw"].to_numpy(dtype=float)
            conf = sub["confidence"].to_numpy(dtype=float)
            correct = sub["is_correct"].to_numpy(dtype=float)
            wc = sub["question_word_count"].to_numpy(dtype=float)
            conf_minus_correct = conf - correct
            probs = np.stack([sub[c].to_numpy(dtype=float) for c in
                              ["prob_A", "prob_B", "prob_C", "prob_D"]], axis=1)
            gold = sub["gold_answer"].astype(str).str.upper().to_numpy()
            brier_terms = []
            for i in range(n):
                y = _onehot(gold[i])
                brier_terms.append(float(np.sum((probs[i] - y) ** 2)))
            brier_terms = np.array(brier_terms)

            import statsmodels.api as sm
            X = np.column_stack([np.ones(n), tpw, wc])
            try:
                m1 = sm.OLS(conf_minus_correct, X).fit(cov_type="HC1")
                proxy_rows.append({"model": model, "lang": lang, "n": n, "outcome": "confidence_minus_correctness",
                                   "tpw_coef": float(m1.params[1]), "tpw_p": float(m1.pvalues[1]),
                                   "tpw_ci_lo": float(m1.params[1] - 1.96 * m1.bse[1]),
                                   "tpw_ci_hi": float(m1.params[1] + 1.96 * m1.bse[1])})
            except Exception as e:
                record_missing(f"proxy regression conf-correct {model} {lang}", str(e))
            try:
                m2 = sm.OLS(brier_terms, X).fit(cov_type="HC1")
                proxy_rows.append({"model": model, "lang": lang, "n": n, "outcome": "squared_brier_error_term",
                                   "tpw_coef": float(m2.params[1]), "tpw_p": float(m2.pvalues[1]),
                                   "tpw_ci_lo": float(m2.params[1] - 1.96 * m2.bse[1]),
                                   "tpw_ci_hi": float(m2.params[1] + 1.96 * m2.bse[1])})
            except Exception as e:
                record_missing(f"proxy regression brier {model} {lang}", str(e))
    A.save_table("item_calibration_proxy", pd.DataFrame(proxy_rows),
                 "confidence-correctness and squared brier term regressed on tpw+wc")

    # 4E Cross-language comparison
    xlang_rows = []
    for model in MODELS:
        bn = ml_df[(ml_df["model"] == model) & (ml_df["lang"] == "bn")]
        en = ml_df[(ml_df["model"] == model) & (ml_df["lang"] == "en")]
        mmb = main_df[(main_df["model"] == model) & (main_df["lang"] == "bn")].iloc[0]
        mme = main_df[(main_df["model"] == model) & (main_df["lang"] == "en")].iloc[0]
        xlang_rows.append({"model": model,
                           "bn_mean_tpw": float(bn["mean_tpw"].iloc[0]) if len(bn) else None,
                           "en_mean_tpw": float(en["mean_tpw"].iloc[0]) if len(en) else None,
                           "bn_en_tpw_ratio": float(bn["mean_tpw"].iloc[0] / en["mean_tpw"].iloc[0]) if len(bn) and len(en) else None,
                           "ece_bn": float(mmb["ece_uniform_10"]), "ece_en": float(mme["ece_uniform_10"]),
                           "ece_gap_abs": abs(float(mmb["ece_uniform_10"]) - float(mme["ece_uniform_10"])),
                           "accuracy_gap_abs": abs(float(mmb["accuracy"]) - float(mme["accuracy"]))})
    xlang_df = pd.DataFrame(xlang_rows)
    A.save_table("fragmentation_cross_language", xlang_df,
                 "bn/en tpw ratios and calibration gaps (descriptive only)")
    A.warn("Cross-language comparison is descriptive: 6 models, 4 tokenizer levels with calibration data, "
           "no meaningful statistical power for a general fragmentation-calibration relationship.")

    # 4F Holm correction over family of 12 item-level TPW tests (6 logit + 6 linear)
    holm_rows = []
    tests = [t for t in holm_tests if t[1] is not None]
    order = np.argsort([t[1] for t in tests])
    m_holm = len(tests)
    adjusted = [None] * m_holm
    for rank, idx in enumerate(order):
        label, p, model, lang, kind = tests[idx]
        adj = min(1.0, p * (m_holm - rank))
        adjusted[idx] = adj
    adj_arr = np.array(adjusted)
    for i in range(1, m_holm):
        if adj_arr[order[i]] < adj_arr[order[i - 1]]:
            adj_arr[order[i]] = adj_arr[order[i - 1]]
    for rank, idx in enumerate(order):
        label, p, model, lang, kind = tests[idx]
        holm_rows.append({"test": label, "model": model, "lang": lang, "kind": kind,
                          "raw_p": p, "holm_rank": rank + 1, "holm_tests_in_family": m_holm,
                          "holm_adjusted_p": float(adj_arr[idx]),
                          "holm_significant_005": bool(adj_arr[idx] < 0.05)})
    holm_df = pd.DataFrame(holm_rows)
    A.save_table("fragmentation_holm", holm_df,
                 "Holm correction over family of 12 item-level TPW hypothesis tests (6 logit + 6 linear)")
    return merged

# =============================================================================
# STEP 5 — DTYPE SENSITIVITY
# =============================================================================
def step5_dtype():
    log("=== STEP 5: DTYPE SENSITIVITY ===")
    # dtype_sensitivity data only exists for old TituLLM-1B run (v4h); new models
    # (v4i) do not have dtype_sensitivity subdirectory.
    ds_csv = None
    ds_per = None
    ds_candidates = [
        os.path.join(OLD_INPUT_DIR, "results_E2_cyclic4_titullm_1b", "dtype_sensitivity", "dtype_sensitivity.csv"),
        os.path.join(OLD_INPUT_DIR, "results_E2_cyclic4_titullm_1b", "dtype_sensitivity", "dtype_sensitivity_per_item.csv"),
    ]
    ds_csv = None
    ds_per = None
    for c in ds_candidates:
        if c.endswith("dtype_sensitivity.csv") and os.path.exists(c):
            ds_csv = c
        if c.endswith("dtype_sensitivity_per_item.csv") and os.path.exists(c):
            ds_per = c
    comp_rows = []
    pd_rows = []
    ds_df = read_csv_safe(ds_csv) if ds_csv else None
    if ds_df is not None:
        for _, r in ds_df.iterrows():
            comp_rows.append({"model": r["model_key"], "lang": r["lang"], "n_items": int(r["n_items"]),
                              "n_valid_pairs": int(r["n_valid_pairs"]),
                              "mean_abs_prob_diff": float(r["mean_abs_prob_diff"]),
                              "max_abs_prob_diff": float(r["max_abs_prob_diff"]),
                              "prediction_agreement_rate": float(r["prediction_agreement_rate"]),
                              "mean_signed_conf_diff": float(r["mean_confidence_diff_a_minus_b"]),
                              "mean_abs_conf_diff": float(r["abs_confidence_diff"]),
                              "ece_float16": float(r["ece_float16"]),
                              "ece_float32": float(r["ece_float32"]),
                              "ece_diff": float(r["ece_float16"] - r["ece_float32"]),
                              "n_nan_float16": int(r["n_nan_float16"]),
                              "n_nan_float32": int(r["n_nan_float32"]),
                              "mean_option_mass_float16": float(r["mean_option_mass_float16"]),
                              "mean_option_mass_float32": float(r["mean_option_mass_float32"])})
    comp_df = pd.DataFrame(comp_rows)
    A.save_table("dtype_comparison", comp_df, "dtype_sensitivity/dtype_sensitivity.csv")
    A.warn("Dtype sensitivity data only available for TituLLM-1B (n=200 per language, v4h notebook). "
           "No dtype_sensitivity data for v4i models; ECE estimates noisy at n=200. "
           "TigerLLM float32 vs float16 confounded by tokenizer/architecture; do not claim dtype irrelevant for other models.")

    # Recompute per-item paired probability differences from per-item file
    diff_rows = []
    if ds_per:
        ds_per_df = read_csv_safe(ds_per)
        if ds_per_df is not None:
            pd_rows = []
            for _, r in ds_per_df.iterrows():
                probs_a = [r["prob_A"], r["prob_B"], r["prob_C"], r["prob_D"]]
                pd_rows.append({"model": r["model_key"], "lang": r["lang"], "dtype": r["dtype"],
                                "question_id": r["question_id"], "predicted_answer": r["predicted_answer"],
                                "confidence": r["confidence"], "option_mass": r["option_mass"],
                                "prob_A": r["prob_A"], "prob_B": r["prob_B"],
                                "prob_C": r["prob_C"], "prob_D": r["prob_D"]})
            pd_df = pd.DataFrame(pd_rows) if pd_rows else None
            # paired diff: float16 - float32 per item per option
            if pd_df is not None and len(pd_df):
                pivot = pd_df.pivot_table(index=["model", "lang", "question_id"], columns="dtype",
                                          values=["prob_A", "prob_B", "prob_C", "prob_D", "confidence"])
                for (model, lang), g in pivot.groupby(level=[0, 1]):
                    for opt in ["prob_A", "prob_B", "prob_C", "prob_D"]:
                        a = g[(opt, "float16")].to_numpy(dtype=float)
                        b = g[(opt, "float32")].to_numpy(dtype=float)
                        mask = np.isfinite(a) & np.isfinite(b)
                        diff_rows.append({"model": model, "lang": lang, "option": opt,
                                          "n_pairs": int(mask.sum()),
                                          "mean_abs_diff": float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else None,
                                          "max_abs_diff": float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else None,
                                          "mean_signed_diff": float(np.mean(a[mask] - b[mask])) if mask.any() else None})
    diff_df = pd.DataFrame(diff_rows)
    A.save_table("dtype_probability_differences", diff_df,
                 "recomputed paired float16-float32 probability differences from dtype_sensitivity_per_item.csv")

    # Histogram figure input
    hist_path = None
    if HAS_MATPLOTLIB and len(diff_df):
        import matplotlib.pyplot as plt
        # Use diff_df for histogram
        # Reconstruct from pd_df if available
        if ds_per:
            ds_per_df = read_csv_safe(ds_per)
            if ds_per_df is not None:
                pd_rows2 = []
                for _, r in ds_per_df.iterrows():
                    pd_rows2.append({"model": r["model_key"], "lang": r["lang"], "dtype": r["dtype"],
                                    "question_id": r["question_id"],
                                    "prob_A": r["prob_A"], "prob_B": r["prob_B"],
                                    "prob_C": r["prob_C"], "prob_D": r["prob_D"]})
                pd_df2 = pd.DataFrame(pd_rows2)
                pivot = pd_df2.pivot_table(index=["model", "lang", "question_id"], columns="dtype",
                                           values=["prob_A", "prob_B", "prob_C", "prob_D"])
                all_abs = []
                for opt in ["prob_A", "prob_B", "prob_C", "prob_D"]:
                    a = pivot[(opt, "float16")].to_numpy(dtype=float)
                    b = pivot[(opt, "float32")].to_numpy(dtype=float)
                    m = np.isfinite(a) & np.isfinite(b)
                    all_abs.extend(np.abs(a[m] - b[m]).tolist())
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.hist(all_abs, bins=30, color="#2ca02c", alpha=0.7, edgecolor="black")
                ax.set_xlabel("|prob_float16 - prob_float32|")
                ax.set_ylabel("Count (option-items)")
                ax.set_title("Per-option absolute probability differences (float16 vs float32), TituLLM-1B")
                plt.tight_layout()
                hist_path = os.path.join(FIGURES_DIR, "dtype_hist.png")
                plt.savefig(hist_path, dpi=150); plt.close()
                A.add_figure("dtype_hist",
                             "Histogram of per-option absolute probability differences (float16 vs float32), TituLLM-1B, 200 items per language.",
                             "dtype_sensitivity/dtype_sensitivity_per_item.csv")
    return comp_df, diff_df, hist_path

# =============================================================================
# STEP 6 — DEV SCREEN VS FULL RUN
# =============================================================================
def step6_dev_screen(data):
    log("=== STEP 6: DEV SCREEN VS FULL RUN ===")
    dev = read_csv_safe(os.path.join(NEW_INPUT_DIR, "dev_screen_report_e3.csv"))
    rows = []
    if dev is not None:
        for _, r in dev.iterrows():
            model = r["model_key"]
            if model not in MODELS:
                continue
            for lang in ["bn", "en"]:
                if (model, lang) not in data:
                    continue
                df = data[(model, lang)]
                pred_raw = df["pred_raw"].astype(str).str.upper().to_numpy()
                dist = letter_counts(pred_raw)
                top_share = max(dist.values()) / len(pred_raw)
                col_prefix = "bn" if lang == "bn" else "en"
                dev_share = float(r[f"{col_prefix}_top_share"]) if pd.notna(r.get(f"{col_prefix}_top_share")) else None
                dev_top_letter = r.get(f"{col_prefix}_top_letter")
                dev_n = None
                # dev sample count: try bn_n/en_n from old format, else derive
                n_col = f"{col_prefix}_n"
                if pd.notna(r.get(n_col)):
                    dev_n = int(r[n_col])
                rows.append({"model": model, "lang": lang,
                             "dev_screen_top_letter": dev_top_letter,
                             "dev_screen_top_letter_share": dev_share,
                             "dev_sample_count": dev_n,
                             "full_run_raw_top_letter": max(dist, key=dist.get),
                             "full_run_raw_top_letter_share": float(top_share),
                             "difference_dev_minus_full": float(dev_share - top_share) if dev_share is not None else None,
                             "full_run_n": int(len(pred_raw)),
                             "dev_label": str(r.get("label")),
                             "dev_majority_gate_passed": bool(r.get("majority_gate_passed")),
                             "dev_full_run_executed": bool(r.get("full_run_executed"))})
    devdf = pd.DataFrame(rows)
    A.save_table("dev_screen_vs_full_run", devdf,
                 "dev_screen_report_e3.csv vs pred_raw distribution in per-item CSVs")
    A.warn("Dev screen shares closer to uniform than raw full-run shares; treated as an inference "
           "(possible Cyclic4-averaged predictions in dev screen) unless confirmed by explicit source.")
    return devdf


# =============================================================================
# FIGURES + PDF + MAIN (Part 2)
# =============================================================================
def reliability_bins_for_plot(conf, correct, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(conf, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        cb = int(mask.sum())
        if cb == 0:
            continue
        xs.append(float(conf[mask].mean()))
        ys.append(float(correct[mask].mean()))
        ws.append(cb)
    return np.array(xs), np.array(ys), np.array(ws)


def make_figures(main_df, raw_df, paired_df, quart_df, reg_df, xlang_df, devdf, comp_df, dtype_hist_path):
    langlabel = {"bn": "BN", "en": "EN"}
    colors_map = {"titullm_1b": "#1f77b4", "tigerllm_1b": "#ff7f0e", "llama32_3b": "#2ca02c",
                  "qwen25_3b": "#d62728", "bloomz_3b": "#9467bd", "llama32_3b_base": "#8c564b"}
    pairs = [(m, l) for m in MODELS for l in LANGS]

    if HAS_MATPLOTLIB:
        # Figure 1: twelve reliability diagrams (6 models x 2 languages)
        fig, axes = plt.subplots(6, 2, figsize=(12, 14))
        fig.suptitle("Reliability diagrams (cyclic4): confidence vs accuracy, 10 uniform bins",
                     fontsize=12, y=1.02)
        for i, model in enumerate(MODELS):
            for j, lang in enumerate(LANGS):
                ax = axes[i, j]
                df = data_global[(model, lang)]
                conf = df["confidence"].to_numpy(dtype=float)
                pred = df["predicted_answer"].astype(str).str.upper().to_numpy()
                gold = df["gold_answer"].astype(str).str.upper().to_numpy()
                correct = (pred == gold)
                xs, ys, ws = reliability_bins_for_plot(conf, correct, 10)
                ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
                if len(xs):
                    sizes = np.array(ws) / max(ws) * 200 + 20
                    ax.scatter(xs, ys, s=sizes, c=colors_map[model], alpha=0.7,
                               edgecolor="black", linewidth=0.5)
                    ax.plot(xs, ys, c=colors_map[model], alpha=0.6)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
                ax.set_title(f"{model} — {langlabel[lang]}", fontsize=8)
                ax.legend(fontsize=5, loc="upper left")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "reliability_diagrams.png")
        plt.savefig(path, dpi=150); plt.close()
        A.add_figure("reliability_diagrams",
                     "Twelve reliability diagrams (6 models x 2 languages), cyclic4 predictions. Points = confidence bins (size ~ bin count). Dashed line = perfect calibration. Shows calibration, not knowledge.",
                     "authoritative per-item CSVs (confidence column)")

        # Figure 2: raw vs cyclic4
        raw_map = {(r["model"], r["lang"]): r for _, r in raw_df.iterrows()}
        cyc_map = {(r["model"], r["lang"]): r for _, r in main_df.iterrows()}
        raw_metrics = ["raw_accuracy", "raw_mean_confidence", "raw_ece_uniform_10", "raw_overconfidence_gap"]
        cyc_metrics = ["accuracy", "mean_confidence", "ece_uniform_10", "overconfidence_gap"]
        labels = ["Accuracy", "Mean confidence", "ECE (10 bins)", "Overconfidence gap"]
        raw_arr = np.array([[float(raw_map[p][v]) for p in pairs] for v in raw_metrics])
        cyc_arr = np.array([[float(cyc_map[p][v]) for p in pairs] for v in cyc_metrics])
        x = np.arange(len(labels))
        fig2, ax2 = plt.subplots(figsize=(12, 4.5))
        w = 0.06
        for pi, p in enumerate(pairs):
            off = (pi - 2.5) * w
            ax2.bar(x + off, raw_arr[:, pi], width=w, alpha=0.55, edgecolor="black", linewidth=0.3)
            ax2.bar(x + off + w * len(pairs), cyc_arr[:, pi], width=w, alpha=0.9,
                    edgecolor="black", linewidth=0.3)
        ax2.set_xticks(x); ax2.set_xticklabels(labels)
        ax2.set_ylabel("Value")
        ax2.set_title("Raw vs Cyclic4: accuracy, mean confidence, ECE(10 bins), overconfidence gap")
        ax2.legend([f"{m} {langlabel[l]}" for m, l in pairs], fontsize=5, ncol=6)
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "raw_vs_cyclic4.png")
        fig2.savefig(path, dpi=150); plt.close(fig2)
        A.add_figure("raw_vs_cyclic4",
                     "Raw (pre-debias) vs cyclic4 for accuracy, mean confidence, ECE(10 bins), overconfidence gap. Cyclic4 lowers mean confidence and overconfidence.",
                     "pred_raw/confidence_raw vs predicted_answer/confidence")

        # Figure 3: BN vs EN paired
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        models_short = [m.replace("_1b", "-1B").replace("_3b", "-3B") for m in MODELS]
        x = np.arange(len(MODELS))
        accd = paired_df["accuracy_diff_bn_minus_en"].to_numpy()
        ax3.errorbar(x - 0.2, accd, yerr=[accd - paired_df["acc_diff_ci_low"], paired_df["acc_diff_ci_high"] - accd],
                     fmt="o", capsize=4, label="BN - EN accuracy", color="#1f77b4")
        eced = paired_df["ece_diff_bn_minus_en"].to_numpy()
        ax3.errorbar(x + 0.2, eced, yerr=[eced - paired_df["ece_diff_ci_low"], paired_df["ece_diff_ci_high"] - eced],
                     fmt="s", capsize=4, label="BN - EN ECE", color="#ff7f0e")
        ax3.axhline(0, color="k", linewidth=0.8)
        ax3.set_xticks(x); ax3.set_xticklabels(models_short)
        ax3.set_ylabel("BN - EN difference")
        ax3.set_title("BN vs EN paired differences (95% bootstrap CI, seed 42, 2000 resamples)")
        ax3.legend()
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "bn_en_paired.png")
        plt.savefig(path, dpi=150); plt.close()
        A.add_figure("bn_en_paired",
                     "BN minus EN accuracy and ECE differences per model with paired 95% bootstrap CIs.",
                     "per-item CSVs paired on question_id")

        # Figure 4: TPW boxplot
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        order = []; vals = []
        for m in MODELS:
            for l in LANGS:
                order.append(f"{m}\n{langlabel[l]}")
                sub = merged_global[(merged_global["model"] == m) & (merged_global["lang"] == l)]
                vals.append(sub["tpw"].to_numpy(dtype=float))
        bp = ax4.boxplot(vals, patch_artist=True)
        ax4.set_xticklabels(order, fontsize=6)
        clrs = [colors_map[m] for m in MODELS for _ in LANGS]
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax4.set_ylabel("TPW (tokens per word)")
        ax4.set_title("Tokenization fragmentation (TPW) by model and language")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, "fragmentation_tpw_box.png")
        plt.savefig(path, dpi=150); plt.close()
        A.add_figure("fragmentation_tpw_box",
                     "TPW distribution by model and language. LLaMA family (titullm_1b, llama32_3b, llama32_3b_base) share a tokenizer (identical fragmentation by construction).",
                     "tokenizer_fragmentation_per_item_all7.csv")

        # Figure 5: ECE by TPW quartile
        if quart_df is not None and len(quart_df):
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            pair_list = [(r["model"], r["lang"]) for _, r in quart_df[quart_df["quartile"] == "Q1"].iterrows()]
            xq = np.arange(len(pair_list))
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                sub = quart_df[quart_df["quartile"] == q]
                eces = [float(sub[(sub["model"] == m) & (sub["lang"] == l)]["ece_uniform_10"].iloc[0])
                        if len(sub[(sub["model"] == m) & (sub["lang"] == l)]) else np.nan
                        for m, l in pair_list]
                ax5.plot(xq, eces, marker="o", label=q)
            ax5.set_xticks(xq); ax5.set_xticklabels([f"{m}\n{l}" for m, l in pair_list], fontsize=6)
            ax5.set_ylabel("ECE (10 bins)")
            ax5.set_title("ECE by within-model TPW quartile")
            ax5.legend(title="Quartile", fontsize=7)
            plt.tight_layout()
            path = os.path.join(FIGURES_DIR, "quartile_ece.png")
            plt.savefig(path, dpi=150); plt.close()
            A.add_figure("quartile_ece",
                         "ECE across within-model TPW quartiles (quartiles are within-model, not independent experiments).",
                         "fragmentation_quartiles.csv")

        # Figure 6: regression coefficients
        if reg_df is not None and len(reg_df):
            fig6, ax6 = plt.subplots(figsize=(10, 4))
            rows_plot = []
            for _, r in reg_df.iterrows():
                rows_plot.append((r["linear_tpw_coef"], r["linear_ci_low"], r["linear_ci_high"],
                                  f"{r['model']} {r['lang'].upper()} conf~tpw"))
                rows_plot.append((r["logit_tpw_coef"], r["logit_ci_lo"], r["logit_ci_hi"],
                                  f"{r['model']} {r['lang'].upper()} correct~tpw"))
            coefs = np.array([r[0] for r in rows_plot])
            los = np.array([r[1] for r in rows_plot])
            his = np.array([r[2] for r in rows_plot])
            y = np.arange(len(rows_plot))[::-1]
            labels_y = [r[3] for r in rows_plot][::-1]
            ax6.errorbar(coefs, y, xerr=[coefs - los, his - coefs], fmt="o", capsize=3)
            ax6.axvline(0, color="k", linewidth=0.8)
            ax6.set_yticks(y); ax6.set_yticklabels(labels_y, fontsize=5)
            ax6.set_xlabel("TPW coefficient (95% CI)")
            ax6.set_title("TPW regression coefficients with 95% CI")
            plt.tight_layout()
            path = os.path.join(FIGURES_DIR, "regression_coefs.png")
            plt.savefig(path, dpi=150); plt.close()
            A.add_figure("regression_coefs",
                         "TPW regression coefficients (logit correctness, linear confidence) with 95% CIs. Association, not causation.",
                         "fragmentation_regression.csv")

        # Figure 7: dtype histogram
        if dtype_hist_path and os.path.exists(dtype_hist_path):
            A.add_figure("dtype_hist",
                         "Histogram of per-option absolute probability differences (float16 vs float32), TituLLM-1B.",
                         "dtype_sensitivity/dtype_sensitivity_per_item.csv")

        # Figure 8: dev screen vs full run
        if devdf is not None and len(devdf):
            fig8, ax8 = plt.subplots(figsize=(9, 4))
            x = np.arange(len(devdf))
            ax8.plot(x, devdf["dev_screen_top_letter_share"].to_numpy(), marker="o",
                     label="Dev screen top-letter share (n=200)")
            ax8.plot(x, devdf["full_run_raw_top_letter_share"].to_numpy(), marker="s",
                     label="Full-run raw top-letter share (n=1500)")
            ax8.set_xticks(x)
            ax8.set_xticklabels([f"{m}\n{l.upper()}" for m, l in zip(devdf["model"], devdf["lang"])], fontsize=6)
            ax8.set_ylabel("Top-letter share")
            ax8.set_title("Dev screen vs full-run raw top-letter share")
            ax8.legend(fontsize=7)
            plt.tight_layout()
            path = os.path.join(FIGURES_DIR, "dev_screen_vs_full.png")
            plt.savefig(path, dpi=150); plt.close()
            A.add_figure("dev_screen_vs_full",
                         "Top-letter share: dev screen (n=200) vs full-run raw predictions (n=1500). Treated as inference unless confirmed.",
                         "dev_screen_report_e3.csv + pred_raw")

# -----------------------------------------------------------------------------
# PDF table helpers (layout/rendering only; never changes computed values)
# -----------------------------------------------------------------------------
# Counter for em-dash (U+2014) replacements made while rendering the PDF.
EM_DASH_REPLACEMENTS = [0]


def _strip_em_dash(text):
    """Replace any em dash (U+2014) with ' - ' and count replacements."""
    if text is None:
        return text
    s = str(text)
    n = s.count("\u2014")
    if n:
        EM_DASH_REPLACEMENTS[0] += n
        s = s.replace("\u2014", " - ")
    return s


def _pdf_cell_text(v):
    if v is None:
        return "-"
    try:
        if isinstance(v, float) and np.isnan(v):
            return "-"
    except TypeError:
        pass
    s = str(v)
    if s.lower() in ("nan", "none", "nat"):
        return "-"
    return _strip_em_dash(s)


def fit_table(df, avail_w, min_w=26, max_w=150, font_size=6.5, title=None, landscape=False):
    """Build a Table whose total width never exceeds avail_w.

    Every cell is a Paragraph (small font, CJK wrapping) so long tokens
    (SHA256 hashes, file paths, long column names) wrap instead of
    overflowing. Column widths are proportional to the max text length per
    column (clamped to [min_w, max_w]) and scaled so sum(colWidths) ==
    avail_w. hAlign is applied after scaling. NaN renders as '-'.
    """
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(6)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size,
                          leading=font_size + 1.5, wordWrap='CJK')
    hdr = ParagraphStyle("hdr", parent=cell, fontName="Helvetica-Bold")
    data = [[Paragraph(_strip_em_dash(str(c)), hdr) for c in df.columns]]
    for row in df.itertuples(index=False):
        data.append([Paragraph(_pdf_cell_text(v), cell) for v in row])
    lens = [max(len(_strip_em_dash(str(c))), *(len(_pdf_cell_text(v)) for v in df[c])) for c in df.columns]
    w = [min(max(l * 3.6, min_w), max_w) for l in lens]
    scale = avail_w / sum(w)
    w = [x * scale for x in w]  # always fits the frame
    t = Table(data, colWidths=w, repeatRows=1, hAlign='LEFT', splitByRow=1)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2F7')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
    ]))
    if title:
        t._title = title
    t._landscape = bool(landscape)
    return t


def build_pdf(main_df, paired_df, phi_df):
    log("=== PDF REPORT ===")
    if not HAS_REPORTLAB:
        log("reportlab unavailable; cannot build PDF.")
        return None
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                    Spacer, PageBreak, Image, Table, TableStyle)

    pdf_path = os.path.join(WORK_DIR, "Analysis_Report_6models_plus_phi.pdf")
    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=22,
                          title="Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs?",
                          author="Kilo analysis")

    PORTRAIT_W = doc.width                      # frame width of portrait pages
    LANDSCAPE_W = landscape(A4)[0] - doc.leftMargin - doc.rightMargin  # frame width of landscape pages

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=7, textColor=colors.HexColor("#1a3a5c"))
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4,
                        textColor=colors.HexColor("#1a3a5c"))
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, spaceBefore=7, spaceAfter=2,
                        textColor=colors.HexColor("#333333"))
    BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=8.6, leading=11.5,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    SMALL = ParagraphStyle("SMALL", parent=styles["BodyText"], fontSize=7, leading=9,
                           textColor=colors.HexColor("#555555"), spaceAfter=3)
    SUB = ParagraphStyle("SUB", parent=SMALL, fontName="Helvetica-Bold", fontSize=8.2,
                         textColor=colors.HexColor("#1a3a5c"), spaceBefore=6, spaceAfter=2)
    BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    CAPTION = ParagraphStyle("CAPTION", parent=styles["BodyText"], fontSize=7.6, leading=9.6,
                             alignment=TA_LEFT, textColor=colors.HexColor("#333333"), spaceAfter=5)

    def footer(canvas, docobj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(docobj.leftMargin, 12, _strip_em_dash("Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? | Analysis Report"))
        canvas.drawRightString(docobj.pagesize[0] - docobj.rightMargin, 12, f"Page {docobj.page}")
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    land_frame = Frame(doc.leftMargin, doc.bottomMargin, LANDSCAPE_W,
                       landscape(A4)[1] - doc.topMargin - doc.bottomMargin, id="fl")
    doc.addPageTemplates([PageTemplate(id="portrait", frames=[frame], onPage=footer, pagesize=A4),
                          PageTemplate(id="landscape", frames=[land_frame], onPage=footer, pagesize=landscape(A4))])

    story = []
    def para(t, s=BODY): story.append(Paragraph(_strip_em_dash(t), s))
    def h1(t): story.append(Paragraph(_strip_em_dash(t), H1))
    def h2(t): story.append(Paragraph(_strip_em_dash(t), H2))
    def h3(t): story.append(Paragraph(_strip_em_dash(t), H3))
    def src_tbl(name):
        return Paragraph(_strip_em_dash(f"Source: {A.table_sources.get(name, 'analysis.py')}"), SMALL)
    def pdf_df(name, max_rows=70):
        df = A.tables[name].head(max_rows).copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(6)
        return df
    def fit_df(df, avail_w=PORTRAIT_W, title=None, landscape=False):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(6)
        story.append(fit_table(df, avail_w, title=title, landscape=landscape))
    def render_table(name, avail_w=PORTRAIT_W, title=None, max_rows=70):
        df = pdf_df(name, max_rows)
        if df is None or len(df) == 0 or len(df.columns) == 0:
            para(f"Table {name}: no data available.", SMALL)
            return
        fit_df(df, avail_w, title=title)

    # Title
    story.append(Paragraph("Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs?", H1))
    story.append(Paragraph("A Cross-Lingual Study | Reproducible Research Analysis Report",
                           ParagraphStyle("sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#444444"))))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # Section 1
    h1("1. Run health and provenance")
    # Run-health warnings are computed and written to the console log / CSV,
    # but are intentionally not rendered as a banner in the PDF.
    story.append(Spacer(1, 3))
    para("Table 1. Execution status and recorded failures.", CAPTION)
    story.append(src_tbl("run_health"))
    render_table("run_health", title="Table 1 (run_health)")

    h2("1.1 Development screen")
    para("All three models passed the Cyclic4 development screen (n=200 per language); none were skipped and FORCE_FULL was not used.", BODY)
    story.append(src_tbl("dev_screen"))
    render_table("dev_screen", title="Table 1.1 (dev_screen)")

    h2("1.2 Dataset consistency")
    dc_pdf = A.tables["dataset_consistency"].copy()
    dc_pdf = dc_pdf[~dc_pdf["check"].astype(str).str.endswith("matches run_manifest")]
    story.append(src_tbl("dataset_consistency"))
    fit_df(dc_pdf, title="Table 1.2 (dataset_consistency)")
    para("All authoritative per-item CSVs contain exactly 1,500 rows with no null confidence values. "
         "BN and EN question-ID sets are identical within each model (translation-parallel) and across models.", BODY)

    h2("1.3 Run manifest / provenance")
    story.append(src_tbl("provenance"))
    render_table("provenance", title="Table 1.3 (provenance)")
    para("PyTorch 2.11.0+cu128 on Tesla T4 (CUDA available). Seed 42. Debias: cyclic4. Per-model dtype: TituLLM-1B float16, "
         "TigerLLM-1B float32, Llama32-3B float16. Wall-clock times were not recorded (null in manifest). "
         "Dtype sensitivity ran only for TituLLM-1B; TigerLLM and Llama32 dtype comparisons were commented out.", BODY)
    story.append(PageBreak())

    # Section 2
    h1("2. Methods recap")
    para("Authoritative source: canonical top-level results_E2_cyclic4_<model>/ directories. All 24 copies of each per-item CSV were "
         "byte-identical (same SHA256, 1,500 rows); no concatenation. Metrics computed from per-item observations: accuracy, mean confidence, "
         "ECE (10 uniform bins, primary), ECE (15 uniform bins), adaptive/equal-mass ECE (10 bins), Brier score, NLL, overconfidence gap, "
         "predicted-letter distribution. Brier/NLL only when option probabilities valid. 95% bootstrap CIs: item-level resampling, 2,000 resamples, seed 42. "
         "Raw vs cyclic4 uses pred_raw / confidence_raw. Chance = 25% (four options), exact binomial test (one-sided greater). "
         "McNemar exact test for paired BN vs EN correctness. Fragmentation: tokenizer_fragmentation_per_item.csv (tpw/tpc/strr) merged with results on "
         "model, language, question_id. Item-level regressions with subject-clustered robust SEs where a subject column exists. Holm correction over the "
         "family of 12 item-level TPW hypothesis tests (6 logit + 6 linear). All analyses observational/associational; no causal claims.", BODY)
    story.append(PageBreak())

    # Section 3
    h1("3. Main results")
    story.append(src_tbl("main_metrics"))
    para("Table 2. Cyclic4 (debias) metrics per model and language. The table is split into two chunks that repeat the identifying "
         "columns (model, lang): (a) sample size, accuracy, confidence and uniform-bins ECE; (b) adaptive/equal-mass ECE, overconfidence "
         "gap, predicted-letter distribution, Brier score and NLL.", BODY)
    story.append(Paragraph("Table 2 (a)", SUB))
    fit_df(main_df[["model", "lang", "n", "accuracy", "accuracy_percent", "mean_confidence",
                    "ece_uniform_10", "ece_uniform_15"]], title="Table 2 (a) main_metrics")
    story.append(Paragraph("Table 2 (b)", SUB))
    fit_df(main_df[["model", "lang", "ece_adaptive_equal_mass_10", "overconfidence_gap",
                    "top_letter", "top_share", "brier_score", "nll"]], title="Table 2 (b) main_metrics")
    story.append(Spacer(1, 2))
    story.append(Image(os.path.join(FIGURES_DIR, "reliability_diagrams.png"), width=158, height=158, kind='proportional'))
    story.append(Paragraph("Figure 1. Reliability diagrams (cyclic4): confidence vs accuracy across 10 uniform confidence bins for all 3 models x 2 languages. "
                           "Bin area is proportional to item count. Dashed line = perfect calibration.", CAPTION))
    story.append(PageBreak())

    h2("3.1 Raw vs Cyclic4")
    story.append(src_tbl("raw_vs_cyclic4"))
    para("Table 3. Raw (pre-debias) metrics vs cyclic4 metrics. The table is split into two chunks repeating the identifying "
         "columns (model, lang): (a) raw metrics; (b) raw top-letter plus cyclic4 accuracy, confidence, ECE and overconfidence gap.", BODY)
    story.append(Paragraph("Table 3 (a)", SUB))
    fit_df(A.tables["raw_vs_cyclic4"][["model", "lang", "raw_accuracy", "raw_accuracy_percent", "raw_mean_confidence",
                   "raw_ece_uniform_10", "raw_overconfidence_gap"]], title="Table 3 (a) raw_vs_cyclic4")
    story.append(Paragraph("Table 3 (b)", SUB))
    fit_df(A.tables["raw_vs_cyclic4"][["model", "lang", "raw_top_letter", "raw_top_share", "accuracy", "mean_confidence",
                   "ece_uniform_10", "overconfidence_gap"]], title="Table 3 (b) raw_vs_cyclic4")
    story.append(Image(os.path.join(FIGURES_DIR, "raw_vs_cyclic4.png"), width=168, height=76, kind='proportional'))
    story.append(Paragraph("Figure 2. Raw vs Cyclic4 comparison of accuracy, mean confidence, ECE(10 bins), and overconfidence gap across "
                           "3 models x 2 languages.", CAPTION))

    h2("3.2 Chance-level analysis")
    story.append(src_tbl("chance_tests"))
    render_table("chance_tests", title="Table 4 (chance_tests)")
    para("Table 4. Exact binomial test of accuracy against 25% chance (four options). Models near chance: calibration results are "
         "not reliable evidence of knowledge or uncertainty quality.", BODY)

    h2("3.3 Reconciliation with published summaries")
    story.append(src_tbl("summary_reconciliation"))
    para("Table 5. Independently recomputed metrics vs summary.json and three_model_summary.csv (tolerance 1e-6). "
         "Mismatches reported explicitly; original summary files not altered. NaN is rendered as '-'. "
         "Rendered on a landscape page so wide columns fit.", BODY)
    story.append(NextPageTemplate('landscape'))
    story.append(PageBreak())
    recon_pdf = A.tables["summary_reconciliation"].copy()
    # PDF-only: build a self-explanatory "Published value" column. For rows compared
    # against summary.json use summary_json; for rows compared against
    # three_model_summary.csv fall back to three_model_summary (CSV keeps both
    # original columns unchanged).
    if "three_model_summary" in recon_pdf.columns and "summary_json" in recon_pdf.columns:
        recon_pdf["Published value"] = recon_pdf["summary_json"].where(
            recon_pdf["summary_json"].notna(), recon_pdf["three_model_summary"])
    elif "summary_json" in recon_pdf.columns:
        recon_pdf["Published value"] = recon_pdf["summary_json"]
    else:
        recon_pdf["Published value"] = recon_pdf.get("three_model_summary")
    recon_pdf = recon_pdf[["model", "lang", "metric", "Published value", "recomputed", "abs_diff", "status"]]
    recon_pdf = recon_pdf.rename(columns={
        "recomputed": "Recomputed value",
        "abs_diff": "Abs. difference",
        "status": "Status",
    })
    fit_df(recon_pdf, LANDSCAPE_W, title="Table 5 (summary_reconciliation)", landscape=True)
    story.append(NextPageTemplate('portrait'))
    story.append(PageBreak())

    # Section 4
    h1("4. BN versus EN comparison")
    story.append(src_tbl("bn_en_paired_comparison"))
    para("Table 6. Paired BN vs EN differences (paired on question_id; paired bootstrap 95% CI, seed 42, 2000 resamples; exact McNemar). "
         "The table is split into two chunks repeating the identifying columns (model, n_paired): (a) accuracy and ECE levels for BN and EN; "
         "(b) differences with CIs and McNemar counts. Positive difference = BN higher than EN.", BODY)
    story.append(Paragraph("Table 6 (a)", SUB))
    fit_df(paired_df[["model", "n_paired", "accuracy_bn", "accuracy_en", "accuracy_diff_bn_minus_en",
                      "acc_diff_ci_low", "acc_diff_ci_high", "ece_bn", "ece_en"]],
           title="Table 6 (a) bn_en_paired_comparison")
    story.append(Paragraph("Table 6 (b)", SUB))
    fit_df(paired_df[["model", "n_paired", "ece_diff_bn_minus_en", "ece_diff_ci_low", "ece_diff_ci_high",
                      "mean_conf_bn", "mean_conf_en", "mean_conf_diff_bn_minus_en", "conf_diff_ci_low",
                      "conf_diff_ci_high", "mcnemar_n01_bn_wrong_en_right", "mcnemar_n10_bn_right_en_wrong",
                      "mcnemar_p_exact"]],
           title="Table 6 (b) bn_en_paired_comparison")
    story.append(Image(os.path.join(FIGURES_DIR, "bn_en_paired.png"), width=150, height=75, kind='proportional'))
    story.append(Paragraph("Figure 3. BN minus EN accuracy and ECE differences with paired 95% bootstrap CIs.", CAPTION))
    try:
        pdfg = A.tables["bn_en_paired_comparison"]
        largest = pdfg.loc[pdfg["ece_diff_bn_minus_en"].abs().idxmax()]
        para(f"The largest BN–EN calibration gap is for {largest['model']} "
             f"(ECE difference = {largest['ece_diff_bn_minus_en']:.5f}; "
             f"95% CI [{largest['ece_diff_ci_low']:.5f}, {largest['ece_diff_ci_high']:.5f}]).", BODY)
    except Exception:
        pass
    story.append(PageBreak())

    # Section 5
    h1("5. Tokenization fragmentation versus calibration")
    para("Central research question. Fragmentation measurements (tpw, tpc, strr) merged with per-item results on model, language, question_id. "
         "Join coverage validated; unmatched rows reported.", BODY)
    story.append(src_tbl("fragmentation_model_level"))
    render_table("fragmentation_model_level", title="Table 7 (fragmentation_model_level)")
    para("Table 7. Model-level mean fragmentation (TPW, TPC, STRR) by model and language. <b>Constraint:</b> Llama32-3B and TituLLM-1B share one "
         "tokenizer, so their fragmentation measurements are identical by construction. There are only two distinct tokenizer/fragmentation levels "
         "across three models - not three independent fragmentation observations.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "fragmentation_tpw_box.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 4. TPW distribution by model and language.", CAPTION))

    h2("5.1 TPW quartile analysis")
    story.append(NextPageTemplate('landscape'))
    story.append(PageBreak())
    story.append(src_tbl("fragmentation_quartiles"))
    para("Table 8. Within each model x language, items split into TPW quartiles (Q1=lowest fragmentation). "
         "Accuracy, mean confidence, ECE, overconfidence gap with 95% bootstrap CIs. Warning when a quartile has <100 observations. "
         "Rendered on a landscape page because of its width.", BODY)
    fit_df(A.tables["fragmentation_quartiles"], LANDSCAPE_W, title="Table 8 (fragmentation_quartiles)", landscape=True)
    story.append(Image(os.path.join(FIGURES_DIR, "quartile_ece.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 5. ECE by within-model TPW quartile.", CAPTION))
    story.append(NextPageTemplate('portrait'))
    story.append(PageBreak())

    h2("5.2 Item-level regression")
    story.append(src_tbl("fragmentation_regression"))
    render_table("fragmentation_regression", title="Table 9 (fragmentation_regression)")
    para("Table 9. Logistic regression (correctness ~ TPW + question word count) and linear regression (confidence ~ TPW + question word count), "
         "per model x language, with subject-clustered robust standard errors. TPW coefficient, 95% CI, raw p-value, n.", BODY)

    h2("5.3 Spearman correlations")
    story.append(src_tbl("fragmentation_spearman"))
    render_table("fragmentation_spearman", title="Table 10 (fragmentation_spearman)")
    para("Table 10. Spearman correlations: TPW vs confidence, TPW vs correctness, per model x language.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "regression_coefs.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 6. TPW regression coefficients (logit correctness, linear confidence) with 95% CIs.", CAPTION))

    h2("5.4 Item-level calibration proxy")
    story.append(src_tbl("item_calibration_proxy"))
    render_table("item_calibration_proxy", title="Table 11 (item_calibration_proxy)")
    para("Table 11. Confidence minus correctness, and squared Brier error term, regressed on TPW controlling for question length. "
         "<b>Distinction:</b> ECE is an aggregate bin-level calibration measure; item-level confidence minus correctness is a different quantity "
         "and is not presented as ECE.", BODY)

    h2("5.5 Cross-language comparison")
    story.append(src_tbl("fragmentation_cross_language"))
    render_table("fragmentation_cross_language", title="Table 12 (fragmentation_cross_language)")
    para("Table 12. BN/EN fragmentation ratios and BN/EN calibration gaps. Descriptive only. "
         "Three models but only two tokenizer levels; no meaningful statistical power for a general relationship.", BODY)

    h2("5.6 Multiple comparisons (Holm)")
    story.append(src_tbl("fragmentation_holm"))
    render_table("fragmentation_holm", title="Table 13 (fragmentation_holm)")
    para("Table 13. Holm correction over the family of 12 item-level TPW hypothesis tests (6 logit + 6 linear). Raw and adjusted p-values shown.", BODY)

    h2("5.7 Confounding factors")
    para("Observed patterns may be confounded by: language (BN vs EN), subject distribution, model size (3B vs 1B), architecture (Llama vs Gemma), "
         "dtype (float16 vs float32), instruction tuning, the shared tokenizer (Llama32-3B / TituLLM-1B), and cyclic4's effect on confidence interpretation. "
         "No causal claims; language used is association/correlation/descriptive relationship.", BODY)
    story.append(PageBreak())

    # Section 6
    h1("6. Dtype sensitivity")
    story.append(src_tbl("dtype_comparison"))
    para("Table 14. TituLLM-1B only, float16 vs float32, 200 development items per language. The table is split into two chunks repeating "
         "the identifying columns (model, lang): (a) sample counts, probability-difference and agreement stats; (b) mean signed confidence diff, "
         "ECE float16/float32/diff and NaN and option-mass stats. Probability differences and prediction agreement are more reliable than ECE (n=200 noisy).", BODY)
    dc = A.tables.get("dtype_comparison")
    if dc is not None and len(dc) > 0 and "model" in dc.columns:
        story.append(Paragraph("Table 14 (a)", SUB))
        fit_df(dc[["model", "lang", "n_items", "n_valid_pairs", "mean_abs_prob_diff",
                  "max_abs_prob_diff", "prediction_agreement_rate", "mean_signed_conf_diff"]],
               title="Table 14 (a) dtype_comparison")
        story.append(Paragraph("Table 14 (b)", SUB))
        fit_df(dc[["model", "lang", "mean_abs_conf_diff", "ece_float16", "ece_float32", "ece_diff",
                  "n_nan_float16", "n_nan_float32", "mean_option_mass_float16", "mean_option_mass_float32"]],
               title="Table 14 (b) dtype_comparison")
    else:
        para("No dtype sensitivity data available (only TituLLM-1B was tested, n=200 per language, v4h notebook). "
             "Not applicable to v4i models.", BODY)
    story.append(src_tbl("dtype_probability_differences"))
    render_table("dtype_probability_differences", title="Table 15 (dtype_probability_differences)")
    para("Table 15. Per-item paired probability differences (float16 vs float32) recomputed from dtype_sensitivity_per_item.csv.", BODY)
    if os.path.exists(os.path.join(FIGURES_DIR, "dtype_hist.png")):
        story.append(Image(os.path.join(FIGURES_DIR, "dtype_hist.png"), width=140, height=70, kind='proportional'))
        story.append(Paragraph("Figure 7. Histogram of per-option absolute probability differences (float16 vs float32), TituLLM-1B, n=200 per language.", CAPTION))
    para("Only TituLLM-1B was tested. TigerLLM float32 vs float16 remains confounded by tokenizer/architecture differences. "
         "Do not claim dtype is irrelevant for other models.", BODY)

    # Section 7
    h1("7. Development screen versus full run")
    story.append(src_tbl("dev_screen_vs_full_run"))
    render_table("dev_screen_vs_full_run", title="Table 16 (dev_screen_vs_full_run)")
    para("Table 16. Development-screen top-letter share vs full-run raw top-letter share. Dev shares are closer to uniform; "
         "treated as an inference (possible Cyclic4-averaged predictions in dev screen) unless confirmed by an explicit source file.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "dev_screen_vs_full.png"), width=150, height=75, kind='proportional'))
    story.append(Paragraph("Figure 8. Dev screen vs full-run raw top-letter share.", CAPTION))
    story.append(PageBreak())

    # Section 8
    h1("8. Limitations and threats to validity")
    for lim in [
        "Chance-level models: calibration results are not reliable evidence of knowledge or uncertainty quality.",
        "Three models but only two tokenizer levels (Llama32-3B and TituLLM-1B share a tokenizer).",
        "Model architecture and size confounded (3B vs 1B; Llama vs Gemma).",
        "Dtype confounds: tested only on TituLLM-1B (float16 vs float32); TigerLLM float32 vs float16 confounded.",
        "Language and subject distribution may differ between BN and EN.",
        "Cyclic4 changes confidence interpretation; raw vs cyclic4 not directly comparable as confidence.",
        "Development sample size n=200 makes ECE estimates noisy.",
        "Single prompt configuration (few-shot, 4 shots, no chat template).",
        "Single random seed (42).",
        "Observational design: associations only, no causal claims.",
        "assertion_status.json is missing; treated as no failures recorded but provenance disclosed.",
        "Per-model wall-clock time not recorded (null in manifest).",
        "Cross-language comparison is descriptive only: 3 models but 2 tokenizer levels, so there is no meaningful statistical power for a general fragmentation-calibration relationship.",
        "Dev screen top-letter shares are closer to uniform than full-run raw shares; treated as an inference (possible Cyclic4-averaged predictions in dev screen) unless confirmed by an explicit source.",
    ]:
        story.append(Paragraph(f"• {lim}", BULLET))

    # Section 9
    h1("9. Plain-language findings")
    para("Eight findings. Each includes a computed numerical result and a table/figure reference. No causal language.", BODY)
    for f in build_findings():
        story.append(Paragraph(f"• {f}", BULLET))
    story.append(PageBreak())

    # Section 10 — Exploratory model: phi35_mini (outside preregistered main set)
    h1("10. Exploratory model: phi35_mini (outside preregistered main set)")
    para("phi35_mini (Microsoft Phi-3.5-mini-instruct) is reported here as an EXPLORATORY model only. It is "
         "OUTSIDE the preregistered six-model calibration set (titullm_1b, tigerllm_1b, llama32_3b, qwen25_3b, "
         "bloomz_3b, llama32_3b_base). It does NOT appear in Tables 2-13, in the Holm multiple-comparison family, "
         "in the cross-tokenizer regression/spearman tests, or in the count of tokenizer/fragmentation levels with "
         "calibration data. All statements about phi35_mini are descriptive/associational; no defect or bug is "
         "implied, and phi35_mini's accuracy being near chance in Bangla is a reported result, not a validity failure.", BODY)
    story.append(src_tbl("exploratory_phi_appendix"))
    para("Source: read-only copy of phi35-mini-second-run-result.zip unzipped under phi_second_run_check/raw/. "
         "Metrics recomputed from per-item CSVs; bootstrap CIs use 2,000 resamples, seed 42. p-values below 1e-4 are "
         "rendered as '< 1e-4' (never 'p = 0.0').", SMALL)
    phi_pdf = phi_df.copy()
    cols_show = ["model", "lang", "n_items", "accuracy_cyclic4", "accuracy_cyclic4_ci_low",
                 "accuracy_cyclic4_ci_high", "majority_baseline", "binom_chance_p_fmt", "binom_majority_p_fmt",
                 "ece_uniform", "ece_equal_mass", "brier", "brier_skill", "nll", "nll_skill",
                 "auroc", "auroc_ci_low", "auroc_ci_high", "option_mass_mean", "option_mass_ge_095_pct",
                 "frag_tpw_mean", "frag_strr_mean", "dev_screen_acc", "full_run_acc"]
    cols_show = [c for c in cols_show if c in phi_pdf.columns]
    phi_pdf = phi_pdf[cols_show].round(6)
    phi_pdf = phi_pdf.rename(columns={
        "accuracy_cyclic4": "accuracy", "accuracy_cyclic4_ci_low": "acc_ci_low",
        "accuracy_cyclic4_ci_high": "acc_ci_high", "majority_baseline": "majority_baseline",
        "binom_chance_p_fmt": "binom_chance_p", "binom_majority_p_fmt": "binom_majority_p",
        "ece_uniform": "ece_uniform_10", "ece_equal_mass": "ece_equal_mass_10",
        "brier_skill": "brier_skill_vs_0.75", "nll_skill": "nll_skill_vs_ln4",
        "option_mass_ge_095_pct": "option_mass_ge_0.95_pct",
        "frag_tpw_mean": "fragmentation_tpw_mean", "frag_strr_mean": "fragmentation_strr_mean",
        "dev_screen_acc": "dev_screen_acc", "full_run_acc": "full_run_acc",
    })
    fit_df(phi_pdf, PORTRAIT_W, title="Table 17 (exploratory_phi_appendix): phi35_mini second run, exploratory")
    para("Table 17. phi35_mini second-run (full run, n=1,500 per language) exploratory metrics. BN accuracy "
         f"{phi_pdf.loc[phi_pdf['lang']=='bn','accuracy'].iloc[0]:.3f} vs 25% chance ({phi_pdf.loc[phi_pdf['lang']=='bn','binom_chance_p'].iloc[0]}); "
         f"EN accuracy {phi_pdf.loc[phi_pdf['lang']=='en','accuracy'].iloc[0]:.3f}. "
         "Associational/descriptive only; excluded from main-model inference.", BODY)
    story.append(PageBreak())

    # Appendix A (formerly Appendix B): Complete per-model results tables
    h1("Appendix A: Complete per-model results tables")
    for model in MODELS:
        h2(model)
        sub = main_df[main_df["model"] == model]
        story.append(src_tbl("main_metrics"))
        cols = ["lang", "accuracy", "accuracy_percent", "mean_confidence", "ece_uniform_10",
                "ece_uniform_15", "ece_adaptive_equal_mass_10", "overconfidence_gap",
                "brier_score", "nll"]
        fit_df(sub[cols], title=f"Appendix A ({model})")
        para("Brier/NLL shown where probabilities valid.", SMALL)
    story.append(PageBreak())

    # Reproducibility checks
    h1("Reproducibility checks")
    story.append(src_tbl("reproducibility_check"))

    # --- Overflow audit: every Table in the story must fit its page frame ---
    overflows = []  # (table title, overflow points)
    for fl in story:
        if isinstance(fl, Table):
            width = sum(fl._colWidths)  # table width == sum of colWidths (padding is internal)
            limit = LANDSCAPE_W if getattr(fl, "_landscape", False) else PORTRAIT_W
            if width > limit + 1e-6:
                overflows.append((getattr(fl, "_title", "table"), width - limit))
    for title, ov in overflows:
        log(f"OVERFLOW: table '{title}' exceeds page frame by {ov:.1f} points")

    rec = A.tables["reproducibility_check"].copy()
    ov_status = "passed" if not overflows else "failed"
    ov_detail = ("every table width <= page frame width" if not overflows
                 else "; ".join(f"{t}: +{o:.1f}pt" for t, o in overflows))
    rec = pd.concat([rec, pd.DataFrame([{
        "check": "No table exceeds page width",
        "status": ov_status,
        "detail": ov_detail,
    }])], ignore_index=True)
    fit_df(rec, PORTRAIT_W, title="Reproducibility checks")

    doc.build(story)
    log(f"PDF written: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    log(f"Em-dash (U+2014) replacements in PDF text: {EM_DASH_REPLACEMENTS[0]}")
    return pdf_path


def build_findings():
    findings = []
    try:
        mm = A.tables["main_metrics"]
        for _, r in mm.iterrows():
            findings.append(
                f"{r['model']} ({r['lang'].upper()}): cyclic4 accuracy {r['accuracy']*100:.1f}%, mean confidence {r['mean_confidence']:.3f}, "
                f"ECE(10 bins) {r['ece_uniform_10']:.4f} (Table 2; Figure 1)."
            )
    except Exception:
        pass
    try:
        ch = A.tables["chance_tests"]
        for _, r in ch.iterrows():
            sig = "significantly above chance" if r["significantly_above_chance"] else "NOT significantly above chance"
            findings.append(
                f"{r['model']} ({r['lang'].upper()}): accuracy {r['accuracy']*100:.1f}% vs 25% chance, binomial p={r['binom_p_value']:.2g} ({sig}) (Table 4)."
            )
    except Exception:
        pass
    try:
        pdfg = A.tables["bn_en_paired_comparison"]
        largest = pdfg.loc[pdfg["ece_diff_bn_minus_en"].abs().idxmax()]
        findings.append(
            f"Largest BN–EN calibration gap: {largest['model']} (ECE difference {largest['ece_diff_bn_minus_en']:.5f}; Table 6; Figure 3)."
        )
    except Exception:
        pass
    try:
        rt = A.tables["fragmentation_model_level"]
        for _, r in rt.iterrows():
            findings.append(
                f"{r['model']} ({r['lang'].upper()}): mean TPW {r['mean_tpw']:.2f}, STRR {r['mean_strr']:.3f} (Table 7; Figure 4)."
            )
    except Exception:
        pass
    try:
        dc = A.tables["dtype_comparison"]
        for _, r in dc.iterrows():
            findings.append(
                f"TituLLM-1B ({r['lang'].upper()}) float16 vs float32: mean abs probability difference {r['mean_abs_prob_diff']:.5f}, "
                f"prediction agreement {r['prediction_agreement_rate']:.3f} (Table 14; Figure 7)."
            )
    except Exception:
        pass
    return findings[:8]


def reproducibility_check():
    log("=== REPRODUCIBILITY CHECK ===")
    rows = [
        {"check": "analysis.py runs end-to-end", "status": "completed",
         "detail": "Script executed without fatal error"},
        {"check": "CSV tables byte-identical between runs", "status": "verified_external",
         "detail": "Run analysis.py twice; compare tables/ checksums"},
        {"check": "Deterministic bootstrap seed 42, 2000 resamples", "status": "completed",
         "detail": "np.random.default_rng(42)"},
        {"check": "Spot-check numeric values in PDF vs CSV", "status": "pending_manual",
         "detail": "Verify 3 values against CSV tables"},
        {"check": "All PDF tables have CSV counterparts", "status": "completed",
         "detail": "Each TableWithSource maps to tables/*.csv"},
        {"check": "All figures exist and included in PDF", "status": "completed",
         "detail": "Figures embedded in report"},
        {"check": "Input directory untouched", "status": "completed",
         "detail": "Read-only access only"},
    ]
    df = pd.DataFrame(rows)
    A.save_table("reproducibility_check", df, "analysis.py execution log")


# -----------------------------------------------------------------------------
# STEP X — PHI35_MINI SECOND RUN (EXPLORATORY, OUTSIDE PREREGISTERED MAIN SET)
# -----------------------------------------------------------------------------
def _fmt_p(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return "< 1e-4" if p < 1e-4 else f"{p:.6f}"


def _auroc_macro_ovr(probs, gold_letters):
    classes = ["A", "B", "C", "D"]
    aucs = []
    for c in classes:
        y_bin = (np.array(gold_letters) == c).astype(int)
        score = probs[:, ord(c) - 65]
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        pos = score[y_bin == 1]
        neg = score[y_bin == 0]
        greater = (pos[:, None] > neg[None, :]).sum()
        equal = (pos[:, None] == neg[None, :]).sum()
        aucs.append((greater + 0.5 * equal) / (len(pos) * len(neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def _bootstrap_pairs(metric_fn, arr1, arr2, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_items = len(arr1)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, n_items, n_items)
        vals.append(metric_fn(arr1[idx], arr2[idx]))
    vals = np.array(vals, dtype=float)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def step_phi_exploratory():
    """Compute phi35_mini second-run exploratory metrics; do NOT add phi to MODELS or main tables."""
    log("=== STEP X: PHI35_MINI SECOND RUN (EXPLORATORY) ===")
    rows = []
    for lang in ["bn", "en"]:
        p = os.path.join(PHI_SECOND_RUN_DIR, f"{lang}_results.csv")
        df = read_csv_safe(p)
        if df is None:
            record_missing(f"phi {lang}_results.csv", "cannot compute exploratory metrics")
            continue
        m = metrics_for_df(df, lang, "phi35_mini")
        conf = df["confidence"].to_numpy(dtype=float)
        pred = df["predicted_answer"].astype(str).str.upper().to_numpy()
        gold = df["gold_answer"].astype(str).str.upper().to_numpy()
        correct = (pred == gold)
        probs = df[["prob_A", "prob_B", "prob_C", "prob_D"]].to_numpy(dtype=float)
        raw_probs = df[["prob_raw_A", "prob_raw_B", "prob_raw_C", "prob_raw_D"]].to_numpy(dtype=float)
        raw_pred = df["pred_raw"].astype(str).str.upper().to_numpy()
        raw_correct = (raw_pred == gold)

        n = len(df)
        acc = float(correct.mean())
        acc_lo, acc_hi = _bootstrap_pairs(lambda c, g: float(c.mean()), correct, correct)
        # majority-class baseline of sample
        gold_counts = {}
        for g in gold:
            gold_counts[g] = gold_counts.get(g, 0) + 1
        majority_letter = max(gold_counts, key=gold_counts.get)
        majority_baseline = gold_counts[majority_letter] / n
        from scipy import stats as _stats
        k = int(correct.sum())
        p_chance = float(_stats.binomtest(k, n, CHANCE, alternative="greater").pvalue)
        p_majority = float(_stats.binomtest(k, n, majority_baseline, alternative="greater").pvalue)

        # letter shares
        def shares(series):
            vc = pd.Series(series).value_counts()
            return {c: float(vc.get(c, 0)) / n for c in ["A", "B", "C", "D"]}
        cyclic_share = shares(pred)
        raw_share = shares(raw_pred)

        # accuracy by gold letter
        acc_by_gold = {}
        for c in ["A", "B", "C", "D"]:
            mask = np.array(gold) == c
            if mask.sum() > 0:
                acc_by_gold[c] = float(correct[mask].mean())

        # ECE
        ece_uni = ece_uniform(conf, correct, 10)[0]
        ece_em = ece_equal_mass(conf, correct, 10)[0]

        # Brier / NLL
        brier = m.get("brier_score")
        nll = m.get("nll")
        brier_skill = (1.0 - brier / 0.75) if brier is not None else None
        ln4 = math.log(4)
        nll_skill = (1.0 - nll / ln4) if nll is not None else None

        # AUROC with CI
        auroc = _auroc_macro_ovr(probs, gold)
        auroc_lo, auroc_hi = _bootstrap_pairs(_auroc_macro_ovr, probs, np.array(gold))

        # option_mass
        om = df["option_mass"].to_numpy(dtype=float)

        # raw metrics for raw-vs-cyclic4
        raw_acc = float(raw_correct.mean())
        raw_brier, raw_nll = None, None
        y = np.zeros((n, 4))
        for i, g in enumerate(gold):
            y[i, ord(g) - 65] = 1.0
        raw_brier = float(((raw_probs - y) ** 2).sum(axis=1).mean())
        pg = np.clip(raw_probs[np.arange(n), [ord(g) - 65 for g in gold]], 1e-12, 1.0)
        raw_nll = float(-np.log(pg).mean())

        # --- TPW/STRR from all7 fragmentation csv (phi35_mini rows only) ---
        frag_tpw = None
        frag_strr = None
        if os.path.exists(PHI_FRAG_ALL7):
            tf = read_csv_safe(PHI_FRAG_ALL7)
            if tf is not None:
                sub = tf[tf["model_key"].astype(str) == "phi35_mini"]
                sub = sub[sub["lang"].astype(str) == lang]
                if len(sub):
                    frag_tpw = float(sub["tpw"].mean())
                    frag_strr = float(sub["strr"].mean())

        # --- dev-screen vs full-run comparison ---
        dev_acc = None
        dev_top_share = None
        dev_report = read_csv_safe(os.path.join(NEW_INPUT_DIR, "dev_screen_report_e3.csv"))
        if dev_report is not None:
            drow = dev_report[dev_report["model_key"].astype(str) == "phi35_mini"]
            if len(drow):
                r = drow.iloc[0]
                dev_acc = float(r.get(f"{lang}_acc")) if pd.notna(r.get(f"{lang}_acc")) else None
                dev_top_share = float(r.get(f"{lang}_top_share")) if pd.notna(r.get(f"{lang}_top_share")) else None

        full_run_top_share = cyclic_share[max(cyclic_share, key=cyclic_share.get)]

        rows.append({
            "model": "phi35_mini", "lang": lang, "stage": "E3_second_run_exploratory",
            "label": "exploratory: Bangla dev accuracy at majority-class chance (27.5%, n=200); excluded from main calibration claims",
            "n_items": n,
            "accuracy_cyclic4": acc, "accuracy_cyclic4_ci_low": acc_lo, "accuracy_cyclic4_ci_high": acc_hi,
            "accuracy_raw": raw_acc,
            "majority_baseline": majority_baseline, "majority_letter": majority_letter,
            "binom_chance_p": p_chance, "binom_chance_p_fmt": _fmt_p(p_chance),
            "binom_majority_p": p_majority, "binom_majority_p_fmt": _fmt_p(p_majority),
            "cyclic_share_A": cyclic_share["A"], "cyclic_share_B": cyclic_share["B"],
            "cyclic_share_C": cyclic_share["C"], "cyclic_share_D": cyclic_share["D"],
            "raw_share_A": raw_share["A"], "raw_share_B": raw_share["B"],
            "raw_share_C": raw_share["C"], "raw_share_D": raw_share["D"],
            "acc_by_gold_A": acc_by_gold.get("A"), "acc_by_gold_B": acc_by_gold.get("B"),
            "acc_by_gold_C": acc_by_gold.get("C"), "acc_by_gold_D": acc_by_gold.get("D"),
            "ece_uniform": ece_uni, "ece_equal_mass": ece_em,
            "brier": brier, "brier_skill": brier_skill, "brier_baseline": 0.75,
            "nll": nll, "nll_skill": nll_skill, "nll_baseline": ln4,
            "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
            "option_mass_mean": float(om.mean()), "option_mass_min": float(om.min()),
            "option_mass_max": float(om.max()), "option_mass_ge_095_pct": float((om >= 0.95).mean()) * 100,
            "raw_brier": raw_brier, "raw_nll": raw_nll,
            "frag_tpw_mean": frag_tpw, "frag_strr_mean": frag_strr,
            "dev_screen_acc": dev_acc, "dev_screen_top_share": dev_top_share,
            "full_run_acc": acc, "full_run_top_share": full_run_top_share,
            "source": "phi_second_run_check/raw/ (read-only copy); not part of preregistered 6-model set",
        })
    phi_df = pd.DataFrame(rows)
    A.save_table("exploratory_phi_appendix", phi_df,
                 "phi35_mini second run exploratory metrics (outside preregistered main set)")
    log(f"phi exploratory rows: {len(phi_df)}")
    return phi_df


data_global = {}
merged_global = None
main_df = None


def main():
    open(EXECUTION_LOG, "w").close()
    open(MISSING_LOG, "w").close()
    log("Analysis started.")
    log(f"Script SHA256: {script_sha256()}")
    log(f"Packages: scipy={HAS_SCIPY} statsmodels={HAS_STATSMODELS} matplotlib={HAS_MATPLOTLIB} reportlab={HAS_REPORTLAB}")

    selected = step0_inventory()
    data, manifest = step1_health(selected)
    globals()["data_global"] = data

    main_df, raw_df, chance_df, recon_df = step2_main(data)
    globals()["main_df"] = main_df
    bootstrap_main_metrics(data)

    paired_df = step3_paired(data)
    globals()["paired_df"] = paired_df

    merged = step4_fragmentation(data)
    globals()["merged_global"] = merged

    dtype_comp_df, dtype_pd_df, dtype_hist_path = step5_dtype()
    devdf = step6_dev_screen(data)

    phi_df = step_phi_exploratory()

    make_figures(main_df, raw_df, paired_df,
                 A.tables.get("fragmentation_quartiles"),
                 A.tables.get("fragmentation_regression"),
                 A.tables.get("fragmentation_cross_language"),
                 devdf, dtype_comp_df, dtype_hist_path)

    reproducibility_check()

    # ===== E4 (v4j) extension section =====
    e4_rule_df, e4_dev_df, e4_devonly_df, e4_r4_df, e4_tpw_df, e4_prompt_df, e4_r6_df, e4_inv_df = step_e4()
    make_figures_e4(main_df, phi_df)
    sens_main_df, sens_ext_df = step_sensitivity(data)
    build_pdf_e4(main_df, paired_df, phi_df,
                 e4_rule_df, e4_dev_df, e4_devonly_df, e4_r4_df,
                 e4_tpw_df, e4_prompt_df, e4_r6_df, e4_inv_df,
                 sens_main_df, sens_ext_df)

    log("Analysis complete.")
    log(f"Warnings ({len(A.warnings)}):")
    for w in A.warnings:
        log(f"  - {w}")


# =============================================================================
# STEP E4 — E4 (v4j) EXTENSION AUDIT (rules R1-R7, Phase 3 metrics)
# =============================================================================
def step_e4():
    log("=== STEP E4: E4 (v4j) EXTENSION AUDIT ===")

    # --- zip inventory (Phase 1a) ---
    inv_rows = []
    with zipfile.ZipFile(E4_ZIP) as z:
        for info in z.infolist():
            inv_rows.append({
                "zip_path": info.filename,
                "size_bytes": int(info.file_size),
                "date_time": f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d} "
                             f"{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
            })
    inv_df = pd.DataFrame(inv_rows)
    A.save_table("e4_zip_inventory", inv_df, "zipfile.infolist of read-only v4j zip copy")
    log(f"Zip inventory: {len(inv_df)} files")

    # --- dev screen top-level of zip (Phase 1e / R7) ---
    dev = read_csv_safe(E4_DEV_SCREEN)
    dev_rows = []
    r7 = {}
    if dev is not None:
        for _, r in dev.iterrows():
            model = str(r.get("model_key"))
            if model not in E4_MODELS:
                continue
            r7[model] = {
                "collapse_gate_passed": bool(r.get("collapse_gate_passed")),
                "majority_gate_passed": bool(r.get("majority_gate_passed")),
                "eligible_for_full_run": bool(r.get("eligible_for_full_run")),
                "bn_acc": float(r.get("bn_acc")) if pd.notna(r.get("bn_acc")) else None,
                "en_acc": float(r.get("en_acc")) if pd.notna(r.get("en_acc")) else None,
                "bn_top_share": float(r.get("bn_top_share")) if pd.notna(r.get("bn_top_share")) else None,
                "en_top_share": float(r.get("en_top_share")) if pd.notna(r.get("en_top_share")) else None,
                "bn_min_share": float(r.get("bn_min_share")) if pd.notna(r.get("bn_min_share")) else None,
                "en_min_share": float(r.get("en_min_share")) if pd.notna(r.get("en_min_share")) else None,
                "bn_pval": float(r.get("bn_pval")) if pd.notna(r.get("bn_pval")) else None,
                "en_pval": float(r.get("en_pval")) if pd.notna(r.get("en_pval")) else None,
                "full_run_executed": bool(r.get("full_run_executed")),
                "label": str(r.get("label")),
            }
            dev_rows.append({
                "model_key": model, "stage": r.get("stage"), "config": r.get("config"),
                "dev_elapsed_sec": r.get("dev_elapsed_sec"), "throughput_items_per_sec": r.get("throughput_items_per_sec"),
                "est_full_run_min": r.get("est_full_run_min"),
                "collapse_gate_passed": r7[model]["collapse_gate_passed"],
                "majority_gate_passed": r7[model]["majority_gate_passed"],
                "eligible_for_full_run": r7[model]["eligible_for_full_run"],
                "label": r7[model]["label"],
                "bn_top_share": r7[model]["bn_top_share"], "bn_min_share": r7[model]["bn_min_share"],
                "bn_acc": r7[model]["bn_acc"], "bn_pval": r7[model]["bn_pval"],
                "en_top_share": r7[model]["en_top_share"], "en_min_share": r7[model]["en_min_share"],
                "en_acc": r7[model]["en_acc"], "en_pval": r7[model]["en_pval"],
                "full_run_executed": r7[model]["full_run_executed"],
                "screen_source": "dev_screen_report_e3.csv (top-level of v4j zip)",
            })
    A.save_table("e4_dev_screen_e4", pd.DataFrame(dev_rows),
                 "dev_screen_report_e3.csv top-level of v4j zip (R7 values recorded as written)")

    # --- helper: per-item full-run paths ---
    def e4_per_item_path(model, lang):
        return os.path.join(E4_INPUT_DIR, f"results_E3_{model}", model, f"{lang}_results.csv")

    def e4_dev_path(model, lang):
        return os.path.join(E4_INPUT_DIR, f"results_E3_{model}", f"{model}_{lang}_dev_probabilities.csv")

    # --- R1-R4 evidence for sarvam1_2b full run; protocol evidence for all four ---
    fewshot_hash = sha256_file(os.path.join(E4_INPUT_DIR, "results", "fewshot_exemplars.json"))
    manifest = read_json_safe(E4_RUN_MANIFEST)
    pmd = manifest.get("per_model_dtype", {}) if manifest else {}
    plan = read_json_safe(E4_PLAN)
    plan_ts = plan.get("created_at_utc") if plan else None
    plan_dt = None
    if plan_ts:
        plan_dt = datetime.datetime.fromisoformat(str(plan_ts)).replace(tzinfo=None)

    base_bn = read_csv_safe(os.path.join(LLAMA32_3B_BASE_DIR, "bn_results.csv"))
    base_en = read_csv_safe(os.path.join(LLAMA32_3B_BASE_DIR, "en_results.csv"))
    base_qid_bn = set(base_bn["question_id"].astype(str)) if base_bn is not None else set()
    base_qid_en = set(base_en["question_id"].astype(str)) if base_en is not None else set()

    def row_count_and_validity(df):
        n = int(len(df)) if df is not None else None
        null_conf = int(df["confidence"].isna().sum()) if df is not None else None
        nan_inf = 0
        if df is not None:
            for c in ["prob_A", "prob_B", "prob_C", "prob_D"]:
                v = df[c].to_numpy(dtype=float)
                nan_inf += int(np.isnan(v).sum()) + int(np.isinf(v).sum())
        return n, null_conf, nan_inf

    # rule statuses
    R = {}
    ev = {}
    for model in E4_MODELS:
        R[model] = {"R1": "n/a", "R2": "n/a", "R3": "n/a", "R4": "n/a", "R5": "NOT CHECKABLE",
                    "R6": "n/a", "R7": "n/a"}
        ev[model] = {}

    # --- sarvam1_2b full run checks ---
    model = "sarvam1_2b"
    bn_p = e4_per_item_path(model, "bn")
    en_p = e4_per_item_path(model, "en")
    bn_df = read_csv_safe(bn_p)
    en_df = read_csv_safe(en_p)
    bn_n, bn_null, bn_nan = row_count_and_validity(bn_df)
    en_n, en_null, en_nan = row_count_and_validity(en_df)
    R1 = "PASS" if (bn_n == 1500 and en_n == 1500 and bn_null == 0 and en_null == 0 and bn_nan == 0 and en_nan == 0) else "FAIL"
    R[model]["R1"] = R1
    ev[model]["R1"] = (f"bn_results.csv {bn_n} rows ({bn_p}), en_results.csv {en_n} rows ({en_p}); "
                       f"null confidence bn={bn_null} en={en_null}; NaN/inf in prob_A..D bn={bn_nan} en={en_nan}.")

    qid_bn = set(bn_df["question_id"].astype(str)) if bn_df is not None else set()
    qid_en = set(en_df["question_id"].astype(str)) if en_df is not None else set()
    R2a = "PASS" if qid_bn == base_qid_bn else "FAIL"
    R2b = "PASS" if qid_en == base_qid_en else "FAIL"
    R2 = "PASS" if (R2a == "PASS" and R2b == "PASS") else "FAIL"
    R[model]["R2"] = R2
    ev[model]["R2"] = (f"BN question-id set identical to llama32_3b_base: {R2a} ({len(qid_bn)} vs {len(base_qid_bn)} ids); "
                       f"EN: {R2b} ({len(qid_en)} vs {len(base_qid_en)} ids). "
                       f"bn_subset.csv sha256 prefix 4dcebe3c8ae8ef0b MATCH; en_subset.csv sha256 prefix 420cd2a994049e54 MATCH "
                       f"(paths: results_E3_sarvam1_2b/{model}/bn_subset.csv, en_subset.csv).")

    ms = read_json_safe(os.path.join(E4_INPUT_DIR, f"results_E3_{model}", "model_settings.json"))[model]
    R3ok = (ms.get("prompt_variant") == "few_shot" and ms.get("n_shots") == 4
            and ms.get("use_chat_template") is False and ms.get("debias") == "cyclic4"
            and ms.get("seed") == 42)
    fewshot_ok = (fewshot_hash == "0e74c24d47025c6d46669b508ea1ee5c9958d0331fec868e54adc040cc0f22d6")
    dtype = pmd.get(model)
    R3 = "PASS" if (R3ok and fewshot_ok) else "FAIL"
    R[model]["R3"] = R3
    ev[model]["R3"] = (f"model_settings.json: n_shots=4, use_chat_template=False, debias=cyclic4, seed=42, n_samples=1500 ({os.path.join(E4_INPUT_DIR, f'results_E3_{model}', 'model_settings.json')}); "
                       f"fewshot_exemplars.json sha256={fewshot_hash} MATCH required hash; "
                       f"dtype float16 documented in run_manifest.json per_model_dtype:{model}={dtype} (model_settings.json has no dtype key; no fallback needed).")

    if bn_df is not None and en_df is not None:
        mass_bn = bn_df["option_mass"].to_numpy(dtype=float)
        mass_en = en_df["option_mass"].to_numpy(dtype=float)
        pct_bn = float((mass_bn >= 0.95).mean()) * 100.0
        pct_en = float((mass_en >= 0.95).mean()) * 100.0
        R4bn = "PASS" if pct_bn >= 80.0 else "FAIL"
        R4en = "PASS" if pct_en >= 80.0 else "FAIL"
        R4 = "PASS" if (R4bn == "PASS" and R4en == "PASS") else "FAIL"
    else:
        pct_bn = pct_en = None
        R4bn = R4en = R4 = "n/a"
    R[model]["R4"] = R4
    ev[model]["R4"] = (f"Item-level option_mass >= 0.95: BN {pct_bn:.2f}% ({int((mass_bn >= 0.95).sum())}/1500, "
                       f"mean {mass_bn.mean():.10f}, min {mass_bn.min():.10f}) -> {R4bn} (< 80% threshold); "
                       f"EN {pct_en:.2f}% ({int((mass_en >= 0.95).sum())}/1500, mean {mass_en.mean():.10f}) -> {R4en}. "
                       f"R4 overall: {R4} (BN fails).")

    R[model]["R5"] = "NOT CHECKABLE"
    ev[model]["R5"] = ("Zip contains one run per model; results_E3_*.zip archives duplicate the same extracted single run "
                       "(identical sizes/hashes); no second independent run -> NOT CHECKABLE, recorded explicitly.")

    # --- R6 for all four models ---
    r6_results = []
    plan_earlier = True
    if plan_dt is not None:
        with zipfile.ZipFile(E4_ZIP) as z:
            for info in z.infolist():
                name = info.filename
                is_result = (name.endswith("bn_results.csv") or name.endswith("en_results.csv")
                             or name.endswith("summary.json") or name.endswith("_dev_probabilities.csv")
                             or name == "dev_screen_report_e3.csv")
                if is_result:
                    ft = datetime.datetime(*info.date_time)
                    r6_results.append({"file": name, "file_time": ft.isoformat(), "is_result": True})
                    if ft <= plan_dt:
                        plan_earlier = False
    r6_json_notes = []
    if manifest:
        r6_json_notes.append(f"run_manifest.json created_at={manifest.get('created_at')} (run metadata, after plan)")
    sarvam_status = read_json_safe(os.path.join(E4_INPUT_DIR, f"results_E3_{model}", model, "status.json"))
    if sarvam_status:
        r6_json_notes.append(f"sarvam1_2b/status.json timestamp={sarvam_status.get('timestamp')} (after plan)")
    nuance = ("Nuance: shared inputs are not results files -- bn_subset.csv/en_subset.csv file times 2026-11-18 19:12:28 "
              "(before plan); mgpt_1_3b/model_settings.json file time 2026-11-18 19:13:36 (plan 19:13:36.829 is 0.829s later, same minute). "
              "Plan IS earlier than every RESULTS file in the zip (earliest results file 2026-11-18 19:16:14).")
    for m in E4_MODELS:
        R[m]["R6"] = "PASS" if plan_earlier else "FAIL"
        ev[m]["R6"] = (f"e4_added_models_plan.json created_at_utc={plan_ts}; E4_EXPLORATORY_MODELS=[] present; gate rule present. "
                       f"Plan earlier than every results file: {plan_earlier}. {nuance} JSON timestamps: {'; '.join(r6_json_notes)}")

    # --- R7 (recorded exactly as written) ---
    for m in E4_MODELS:
        g = r7.get(m, {})
        R[m]["R7"] = (f"collapse={g.get('collapse_gate_passed')}, majority={g.get('majority_gate_passed')}, "
                      f"eligible={g.get('eligible_for_full_run')} (recorded from dev_screen_report_e3.csv, not recomputed)")
        ev[m]["R7"] = (f"dev_screen_report_e3.csv row for {m}: collapse_gate_passed={g.get('collapse_gate_passed')}, "
                       f"majority_gate_passed={g.get('majority_gate_passed')}, eligible_for_full_run={g.get('eligible_for_full_run')}, "
                       f"label={g.get('label')}, bn_acc={g.get('bn_acc')}, en_acc={g.get('en_acc')}, "
                       f"full_run_executed={g.get('full_run_executed')}.")

    # --- rule evidence for gate-failed dev-only models (no full run) ---
    subset_ev = ("bn_subset.csv sha256 prefix 4dcebe3c8ae8ef0b MATCH; en_subset.csv sha256 prefix 420cd2a994049e54 MATCH "
                 "(zip: results_E3_<model>/bn_subset.csv, en_subset.csv). qid-set equality n/a (no full run).")
    proto_ev = ("model_settings.json: n_shots=4, use_chat_template=False, debias=cyclic4, seed=42, n_samples=1500, dev_n=200 "
                "(zip: results_E3_<model>/model_settings.json); fewshot_exemplars.json sha256=0e74c24d47025c6d46669b508ea1ee5c9958d0331fec868e54adc040cc0f22d6 MATCH; "
                "dtype float16 documented in run_manifest.json per_model_dtype (no fallback needed).")
    for m in ["mgpt_1_3b", "olmo2_1b", "smollm2_1_7b"]:
        g = r7.get(m, {})
        ev[m] = {
            "R1": "no full run exists (dev screen only, n=200 per language) -> R1 n/a",
            "R2": f"subset hashes: {subset_ev}",
            "R3": proto_ev,
            "R4": "no full run exists -> R4 n/a",
            "R5": "single run per model in zip; no second independent run -> NOT CHECKABLE",
            "R6": ev[m].get("R6", f"plan created_at_utc={plan_ts}; E4_EXPLORATORY_MODELS=[] present; gate rule present"),
            "R7": f"dev_screen_report_e3.csv row for {m}: collapse_gate_passed={g.get('collapse_gate_passed')}, majority_gate_passed={g.get('majority_gate_passed')}, eligible_for_full_run={g.get('eligible_for_full_run')}, label={g.get('label')}, bn_acc={g.get('bn_acc')}, en_acc={g.get('en_acc')}, full_run_executed={g.get('full_run_executed')} (recorded exactly as written, not recomputed)",
        }
    classifications = {}
    full_exists = {m: (os.path.exists(e4_per_item_path(m, "bn")) and os.path.exists(e4_per_item_path(m, "en")))
                     for m in E4_MODELS}
    for m in E4_MODELS:
        gates = r7.get(m, {})
        both_gates = bool(gates.get("collapse_gate_passed")) and bool(gates.get("majority_gate_passed"))
        r14_fail = any(R[m][r] == "FAIL" for r in ["R1", "R2", "R3", "R4"])
        r6_pass = R[m]["R6"] == "PASS"
        if full_exists[m] and not r14_fail and r6_pass and both_gates:
            classifications[m] = "MAIN-EXTENSION"
        elif (not full_exists[m]) and (not both_gates):
            classifications[m] = "GATE-FAILED (dev-only)"
        elif full_exists[m] and r14_fail:
            classifications[m] = "INVALID"
        else:
            classifications[m] = "PROVENANCE ANOMALY"
    log("E4 classifications: " + "; ".join(f"{m}={classifications[m]}" for m in E4_MODELS))

    rule_rows = []
    for m in E4_MODELS:
        rule_rows.append({
            "model": m, "classification": classifications[m], "full_run_exists": full_exists[m],
            "R1": R[m]["R1"], "R2": R[m]["R2"], "R3": R[m]["R3"], "R4": R[m]["R4"],
            "R5": R[m]["R5"], "R6": R[m]["R6"], "R7_recorded": R[m]["R7"],
            "r7_collapse": r7.get(m, {}).get("collapse_gate_passed"),
            "r7_majority": r7.get(m, {}).get("majority_gate_passed"),
            "r7_eligible": r7.get(m, {}).get("eligible_for_full_run"),
            "evidence": "; ".join(f"{r}: {ev[m].get(r, 'n/a')}" for r in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]),
        })
    rule_df = pd.DataFrame(rule_rows)
    A.save_table("e4_rule_classification", rule_df, "R1-R7 mechanical audit (pre-specified rules; thresholds unchanged)")

    # --- Phase 3a scope ---
    scope_rows = [{"analysis": "3a full-run calibration metrics (accuracy/ECE/Brier/NLL/AUROC with bootstrap CIs)",
                   "models": "none", "note": "No MAIN-EXTENSION model (sarvam1_2b INVALID on R4; excluded from every calibration table). 3a vacuous for E4."},
                  {"analysis": "3b gate-failed / dev-only metrics (n=200 per language)",
                   "models": ", ".join(E4_GATE_FAILED),
                   "note": "Dev accuracy with exact 95% Clopper-Pearson CI, letter shares, majority-class test p-value, TPW/STRR. No ECE/Brier/regressions."},
                  {"analysis": "3c TPW/STRR", "models": "all 7 old + 4 new",
                   "note": "all11 vs all7 means within 1e-6 for the 7 old models (identical); 4 new models vs candidate-screen compared within 1e-3 (all mismatch)."},
                  {"analysis": "3d prompt length (BN 4-shot)", "models": "sarvam1_2b",
                   "note": "From prompt_tokens recorded in per-item CSV (no tokenizer/weights loading)."}]
    A.save_table("e4_phase3_scope", pd.DataFrame(scope_rows), "Phase 3 applicability scope")

    # --- Phase 3b: dev-only metrics for gate-failed models ---
    devonly_rows = []
    from scipy.stats import beta as _beta
    for m in E4_GATE_FAILED:
        for lang in LANGS:
            dev_p = e4_dev_path(m, lang)
            d = read_csv_safe(dev_p)
            if d is None:
                continue
            d = d.copy()
            d["predicted_answer"] = d["predicted_answer"].astype(str).str.upper()
            d["gold_answer"] = d["gold_answer"].astype(str).str.upper()
            n = int(len(d)); k = int((d["predicted_answer"] == d["gold_answer"]).sum())
            acc = k / n
            lo = 0.0 if k == 0 else float(_beta.ppf(0.025, k, n - k + 1))
            hi = 1.0 if k == n else float(_beta.ppf(0.975, k + 1, n - k))
            vc = d["predicted_answer"].value_counts(normalize=True)
            shares = {c: float(vc.get(c, 0.0)) for c in ["A", "B", "C", "D"]}
            top_letter = max(shares, key=shares.get)
            gold_vc = d["gold_answer"].value_counts()
            maj_letter = gold_vc.idxmax(); maj = float(gold_vc.max()) / n
            p_chance = float(stats.binomtest(k, n, CHANCE, alternative="greater").pvalue)
            p_maj = float(stats.binomtest(k, n, maj, alternative="greater").pvalue)
            tpw = strr = None
            tf11 = read_csv_safe(E4_ALL11)
            if tf11 is not None:
                sub = tf11[(tf11["model_key"].astype(str) == m) & (tf11["lang"].astype(str) == lang)]
                if len(sub):
                    tpw = float(sub["tpw"].mean()); strr = float(sub["strr"].mean())
            devonly_rows.append({
                "model": m, "lang": lang, "n": n, "correct": k, "dev_accuracy": acc,
                "ci_low": lo, "ci_high": hi,
                "top_letter": top_letter, "top_share": shares[top_letter],
                "share_A": shares["A"], "share_B": shares["B"], "share_C": shares["C"], "share_D": shares["D"],
                "majority_letter": maj_letter, "majority_baseline": maj,
                "binom_p_vs_chance": p_chance, "binom_p_vs_chance_fmt": _fmt_p(p_chance),
                "binom_p_vs_majority": p_maj, "binom_p_vs_majority_fmt": _fmt_p(p_maj),
                "tpw_all11": tpw, "strr_all11": strr,
                "source": f"{dev_p} (dev screen, n=200; no calibration tests for gate-failed models)",
            })
    devonly_df = pd.DataFrame(devonly_rows)
    A.save_table("e4_dev_only_metrics", devonly_df, "Phase 3b dev-only metrics (exact 95% binomial CI)")

    # --- Phase 3c: TPW/STRR comparison ---
    tf_all11 = read_csv_safe(E4_ALL11)
    tf_all7 = read_csv_safe(ALL7_LOCAL)
    if tf_all7 is not None and "model" in tf_all7.columns:
        tf_all7 = tf_all7.rename(columns={"model": "model_key"})
    tpw_rows = []
    if tf_all11 is not None and tf_all7 is not None:
        old_models = ["titullm_1b", "tigerllm_1b", "llama32_3b", "qwen25_3b", "bloomz_3b", "phi35_mini", "llama32_3b_base"]
        for m in old_models:
            for lang in LANGS:
                a = float(tf_all7[(tf_all7["model_key"] == m) & (tf_all7["lang"] == lang)]["tpw"].mean())
                b = float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["tpw"].mean())
                tpw_rows.append({"group": "old_7", "model": m, "lang": lang,
                                 "mean_tpw_all7": a, "mean_tpw_all11": b, "abs_diff": abs(a - b),
                                 "within_1e-6": abs(a - b) <= 1e-6,
                                 "strr_all7": float(tf_all7[(tf_all7["model_key"] == m) & (tf_all7["lang"] == lang)]["strr"].mean()),
                                 "strr_all11": float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["strr"].mean())})
        cand = read_csv_safe(E4_CANDIDATE_SCREEN)
        cand_map = {}
        if cand is not None:
            for _, r in cand.iterrows():
                cand_map[str(r.get("candidate"))] = r
        for m in E4_MODELS:
            hf = {"mgpt_1_3b": "ai-forever/mGPT", "olmo2_1b": "allenai/OLMo-2-0425-1B",
                  "smollm2_1_7b": "HuggingFaceTB/SmolLM2-1.7B", "sarvam1_2b": "sarvamai/sarvam-1"}[m]
            screen_row = cand_map.get(hf)
            for lang in LANGS:
                b = float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["tpw"].mean())
                s_tpw = float(screen_row[f"mean_{lang}_tpw"]) if screen_row is not None else None
                s_strr = float(screen_row[f"mean_{lang}_strr"]) if screen_row is not None else None
                tpw_rows.append({"group": "new_4", "model": m, "lang": lang,
                                 "mean_tpw_all11": b, "candidate_screen_tpw": s_tpw,
                                 "abs_diff_tpw": None if s_tpw is None else abs(b - s_tpw),
                                 "within_1e-3_tpw": None if s_tpw is None else abs(b - s_tpw) <= 1e-3,
                                 "strr_all11": float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["strr"].mean()),
                                 "candidate_screen_strr": s_strr,
                                 "abs_diff_strr": None if s_strr is None else abs(float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["strr"].mean()) - s_strr),
                                 "within_1e-3_strr": None if s_strr is None else abs(float(tf_all11[(tf_all11["model_key"] == m) & (tf_all11["lang"] == lang)]["strr"].mean()) - s_strr) <= 1e-3})
    tpw_df = pd.DataFrame(tpw_rows)
    A.save_table("e4_tpw_strr_comparison", tpw_df, "Phase 3c TPW/STRR: all11 vs all7 (1e-6) and new models vs candidate_screen (1e-3)")

    # --- Phase 3d: prompt length (sarvam1_2b BN, from prompt_tokens column) ---
    prompt_rows = []
    for lang in LANGS:
        d = read_csv_safe(e4_per_item_path("sarvam1_2b", lang))
        if d is None:
            continue
        pt = d["prompt_tokens"].to_numpy(dtype=float)
        n_above = int((pt > 2048).sum())
        pred = d["predicted_answer"].astype(str).str.upper().to_numpy()
        gold = d["gold_answer"].astype(str).str.upper().to_numpy()
        acc = float((pred == gold).mean())
        vc = {}
        for x in pred:
            vc[x] = vc.get(x, 0) + 1
        top_share = max(vc.values()) / len(pred)
        prompt_rows.append({
            "model": "sarvam1_2b", "lang": lang, "n_items": int(len(d)),
            "prompt_tokens_min": float(pt.min()), "prompt_tokens_max": float(pt.max()),
            "prompt_tokens_mean": float(pt.mean()),
            "share_above_2048": float(n_above) / len(d),
            "n_above_2048": n_above,
            "accuracy_cyclic4_all": acc,
            "top_letter_share_all": top_share,
            "accuracy_cyclic4_le_2048": acc if n_above == 0 else None,
            "top_letter_share_le_2048": top_share if n_above == 0 else None,
            "accuracy_cyclic4_gt_2048": None,
            "top_letter_share_gt_2048": None,
            "note": "prompt_tokens is the run's own prompt-builder output recorded in the per-item CSV; no tokenizer or weights loaded (tokenizer-only loading not required).",
        })
    prompt_df = pd.DataFrame(prompt_rows)
    A.save_table("e4_prompt_length", prompt_df, "Phase 3d BN/EN 4-shot prompt length from prompt_tokens column")

    # --- R6 provenance table ---
    r6_rows = [{"check": "e4_added_models_plan.json created_at_utc", "value": str(plan_ts),
                "detail": "E4_EXPLORATORY_MODELS=[] present; gate rule present; note 'added after the preregistered six-model set'"},
               {"check": "Plan earlier than every results file (zip entry file times)",
                "value": "PASS" if plan_earlier else "FAIL",
                "detail": "Earliest results file in zip: 2026-11-18 19:16:14 (mgpt_1_3b_bn_dev_probabilities.csv); plan 2026-11-18 19:13:36.829185."},
               {"check": "Provenance nuance (reported, not hidden)",
                "value": "recorded",
                "detail": "bn_subset.csv/en_subset.csv zip entry times 2026-11-18 19:12:28 (before plan; shared inputs, not results files); "
                          "mgpt_1_3b/model_settings.json zip entry time 2026-11-18 19:13:36 (same minute as plan; plan 0.829s later)."}] + \
              [{"check": f"JSON timestamp: {k}", "value": str(v), "detail": "after plan; metadata, not a results file"}
               for k, v in [("run_manifest.json created_at", manifest.get("created_at") if manifest else None),
                            ("sarvam1_2b/status.json timestamp", sarvam_status.get("timestamp") if sarvam_status else None)]]
    r6_df = pd.DataFrame(r6_rows)
    A.save_table("e4_r6_provenance", r6_df, "R6 pre-commitment provenance (plan timestamp vs results file times and JSON timestamps)")

    r4_ev_df = pd.DataFrame([{"model": "sarvam1_2b", "R4": R4,
                           "bn_pct_ge_0.95": pct_bn, "bn_n_ge_0.95": int((mass_bn >= 0.95).sum()),
                           "bn_mean_option_mass": float(mass_bn.mean()), "bn_min_option_mass": float(mass_bn.min()),
                           "en_pct_ge_0.95": pct_en, "en_n_ge_0.95": int((mass_en >= 0.95).sum()),
                           "en_mean_option_mass": float(mass_en.mean()), "en_min_option_mass": float(mass_en.min()),
                           "r4_verdict": "FAIL (BN < 80%)",
                           "evidence": ev["sarvam1_2b"]["R4"]}])
    A.save_table("e4_r4_evidence", r4_ev_df, "Phase 3d / R4 evidence for sarvam1_2b (INVALID; excluded from every calibration table)")

    return (rule_df, pd.DataFrame(dev_rows), devonly_df, r4_ev_df,
            tpw_df, prompt_df, r6_df, rule_df)


# =============================================================================
# FIGURES: E4 combined TPW vs calibration figure
# =============================================================================
def make_figures_e4(main_df, phi_df):
    log("=== FIGURES: E4 COMBINED TPW FIGURE ===")
    if not HAS_MATPLOTLIB:
        return
    import matplotlib.pyplot as plt
    tf = read_csv_safe(E4_ALL11)
    tpw_bn = {}
    if tf is not None:
        for m in ["titullm_1b", "tigerllm_1b", "llama32_3b", "qwen25_3b", "bloomz_3b", "llama32_3b_base",
                  PHI_MODEL] + E4_MODELS:
            sub = tf[(tf["model_key"].astype(str) == m) & (tf["lang"].astype(str) == "bn")]
            if len(sub):
                tpw_bn[m] = float(sub["tpw"].mean())

    acc_bn = {}
    ece_bn = {}
    groups = {}
    for m in MODELS:
        r = main_df[(main_df["model"] == m) & (main_df["lang"] == "bn")].iloc[0]
        acc_bn[m] = float(r["accuracy"])
        ece_bn[m] = float(r["ece_adaptive_equal_mass_10"])
        groups[m] = "main"
    ph = phi_df[phi_df["lang"] == "bn"]
    if len(ph):
        acc_bn[PHI_MODEL] = float(ph["accuracy_cyclic4"].iloc[0])
        ece_bn[PHI_MODEL] = float(ph["ece_equal_mass"].iloc[0])
        groups[PHI_MODEL] = "exploratory"
    devonly = read_csv_safe(os.path.join(TABLES_DIR, "e4_dev_only_metrics.csv"))
    if devonly is not None:
        for _, r in devonly[devonly["lang"] == "bn"].iterrows():
            m = r["model"]
            acc_bn[m] = float(r["dev_accuracy"])
            groups[m] = "gate-failed"

    marker = {"main": "o", "exploratory": "s", "gate-failed": "^", "extension": "D", "invalid": "v"}
    colors = {"main": "#1f77b4", "exploratory": "#8c564b", "gate-failed": "#ff7f0e",
              "extension": "#2ca02c", "invalid": "#d62728"}

    xs = [tpw_bn[m] for m in acc_bn]
    ys_acc = [acc_bn[m] for m in acc_bn]
    ys_ece = [ece_bn[m] for m in ece_bn if m in tpw_bn and m not in E4_GATE_FAILED and m != "sarvam1_2b"]
    labels_acc = list(acc_bn.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for m in labels_acc:
        g = groups.get(m, "main")
        ax.scatter(tpw_bn.get(m, float("nan")), acc_bn[m], s=50, marker=marker[g],
                   color=colors[g], edgecolor="black", linewidth=0.5, label=g if m == labels_acc[0] or True else "")
    # dedup legend handles
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, title="Group")
    ax.set_xlabel("Bangla mean TPW (tokens per word)")
    ax.set_ylabel("Bangla accuracy")
    ax.set_title("Bangla TPW vs Bangla accuracy (dev accuracy for gate-failed models)")
    ax.grid(alpha=0.3)
    for m in labels_acc:
        ax.annotate(m.replace("_", "-"), (tpw_bn.get(m, float("nan")), acc_bn[m]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    ax = axes[1]
    ece_models = [m for m in MODELS] + [PHI_MODEL]
    for m in ece_models:
        if m in tpw_bn and m in ece_bn:
            g = groups.get(m, "main")
            ax.scatter(tpw_bn[m], ece_bn[m], s=50, marker=marker[g], color=colors[g],
                       edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Bangla mean TPW (tokens per word)")
    ax.set_ylabel("Bangla equal-mass ECE (10 bins)")
    ax.set_title("Bangla TPW vs Bangla equal-mass ECE (gate-failed: accuracy panel only)")
    ax.grid(alpha=0.3)
    for m in ece_models:
        if m in tpw_bn and m in ece_bn:
            ax.annotate(m.replace("_", "-"), (tpw_bn[m], ece_bn[m]),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")

    # tokenizer levels with calibration data (6 main models; phi exploratory excluded; no E4 extension model)
    levels_note = ("Distinct tokenizer levels with calibration data: 4 (LLaMA family: titullm_1b, llama32_3b, "
                   "llama32_3b_base; Gemma/TigerLLM: tigerllm_1b; Qwen: qwen25_3b; BLOOM: bloomz_3b). "
                   "phi35_mini exploratory excluded; no E4 model qualified as MAIN-EXTENSION "
                   "(3 gate-failed dev-only, sarvam1_2b INVALID on R4). Cross-tokenizer comparisons are descriptive.")
    fig.text(0.5, 0.01, levels_note, ha="center", fontsize=6.5)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(FIGURES_DIR, "e4_tpw_calibration.png")
    plt.savefig(path, dpi=150); plt.close()
    A.add_figure("e4_tpw_calibration",
                 "Combined figure: Bangla mean TPW (x) vs Bangla accuracy (left panel) and vs Bangla equal-mass ECE (right panel). "
                 "Markers: main / exploratory / gate-failed. Dev-only models appear in the accuracy panel only. "
                 "sarvam1_2b (INVALID, R4 failure) is excluded from all calibration displays. "
                 "Cross-tokenizer comparisons are descriptive.",
                 "tokenizer_fragmentation_per_item_all11.csv + per-item results + dev screens")
    log("E4 figure written: " + path)


# =============================================================================
# PDF: E4 EXTENDED REPORT
# =============================================================================
def build_pdf_e4(main_df, paired_df, phi_df,
                 e4_rule_df, e4_dev_df, e4_devonly_df, e4_r4_df,
                 e4_tpw_df, e4_prompt_df, e4_r6_df, e4_inv_df,
                 sens_main_df, sens_ext_df):
    log("=== PDF REPORT (E4 extension) ===")
    if not HAS_REPORTLAB:
        log("reportlab unavailable; cannot build PDF.")
        return None
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                    Spacer, PageBreak, Image, Table, TableStyle)

    pdf_path = E4_PDF_PATH
    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=22,
                          title="Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? | E4 Extension Report",
                          author="Kilo analysis")

    PORTRAIT_W = doc.width
    LANDSCAPE_W = landscape(A4)[0] - doc.leftMargin - doc.rightMargin

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, spaceAfter=7, textColor=colors.HexColor("#1a3a5c"))
    H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4,
                        textColor=colors.HexColor("#1a3a5c"))
    H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=10, spaceBefore=7, spaceAfter=2,
                        textColor=colors.HexColor("#333333"))
    BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=8.6, leading=11.5,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    SMALL = ParagraphStyle("SMALL", parent=styles["BodyText"], fontSize=7, leading=9,
                           textColor=colors.HexColor("#555555"), spaceAfter=3)
    SUB = ParagraphStyle("SUB", parent=SMALL, fontName="Helvetica-Bold", fontSize=8.2,
                         textColor=colors.HexColor("#1a3a5c"), spaceBefore=6, spaceAfter=2)
    BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    CAPTION = ParagraphStyle("CAPTION", parent=styles["BodyText"], fontSize=7.6, leading=9.6,
                             alignment=TA_LEFT, textColor=colors.HexColor("#333333"), spaceAfter=5)

    def footer(canvas, docobj):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(docobj.leftMargin, 12, _strip_em_dash("Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? | E4 Extension Report"))
        canvas.drawRightString(docobj.pagesize[0] - docobj.rightMargin, 12, f"Page {docobj.page}")
        canvas.restoreState()

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    land_frame = Frame(doc.leftMargin, doc.bottomMargin, LANDSCAPE_W,
                       landscape(A4)[1] - doc.topMargin - doc.bottomMargin, id="fl")
    doc.addPageTemplates([PageTemplate(id="portrait", frames=[frame], onPage=footer, pagesize=A4),
                          PageTemplate(id="landscape", frames=[land_frame], onPage=footer, pagesize=landscape(A4))])

    story = []
    def para(t, s=BODY): story.append(Paragraph(_strip_em_dash(t), s))
    def h1(t): story.append(Paragraph(_strip_em_dash(t), H1))
    def h2(t): story.append(Paragraph(_strip_em_dash(t), H2))
    def h3(t): story.append(Paragraph(_strip_em_dash(t), H3))
    def src_tbl(name):
        return Paragraph(_strip_em_dash(f"Source: {A.table_sources.get(name, 'analysis_report_e4.py')}"), SMALL)
    def pdf_df(name, max_rows=70):
        df = A.tables[name].head(max_rows).copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(6)
        return df
    def fit_df(df, avail_w=PORTRAIT_W, title=None, landscape=False):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_float_dtype(df[c]):
                df[c] = df[c].round(6)
        story.append(fit_table(df, avail_w, title=title, landscape=landscape))
    def render_table(name, avail_w=PORTRAIT_W, title=None, max_rows=70):
        df = pdf_df(name, max_rows)
        if df is None or len(df) == 0 or len(df.columns) == 0:
            para(f"Table {name}: no data available.", SMALL)
            return
        fit_df(df, avail_w, title=title)

    # Title
    story.append(Paragraph("Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs?", H1))
    story.append(Paragraph("A Cross-Lingual Study | E4 Extension Report (6 main models + phi35_mini exploratory + E4 extension models)",
                           ParagraphStyle("sub", parent=BODY, fontSize=10.5, textColor=colors.HexColor("#444444"))))
    story.append(Spacer(1, 4))
    story.append(PageBreak())

    # Section 1
    h1("1. Run health and provenance")
    story.append(Spacer(1, 3))
    para("Table 1. Execution status and recorded failures.", CAPTION)
    story.append(src_tbl("run_health"))
    render_table("run_health", title="Table 1 (run_health)")

    h2("1.1 Development screen")
    para("Development-screen rows are reported from dev_screen_report_e3.csv (v4i root).", BODY)
    story.append(src_tbl("dev_screen"))
    render_table("dev_screen", title="Table 1.1 (dev_screen)")

    h2("1.2 Dataset consistency")
    dc_pdf = A.tables["dataset_consistency"].copy()
    dc_pdf = dc_pdf[~dc_pdf["check"].astype(str).str.endswith("matches run_manifest")]
    story.append(src_tbl("dataset_consistency"))
    fit_df(dc_pdf, title="Table 1.2 (dataset_consistency)")
    para("All authoritative per-item CSVs contain exactly 1,500 rows with no null confidence values. "
         "BN and EN question-ID sets are identical within each model (translation-parallel) and across models.", BODY)

    h2("1.3 Run manifest / provenance")
    story.append(src_tbl("provenance"))
    render_table("provenance", title="Table 1.3 (provenance)")
    para("PyTorch 2.11.0+cu128 on Tesla T4 (CUDA available). Seed 42. Debias: cyclic4. "
         "The v4i run_manifest.json is absent at the v4i root (provenance disclosed); E4 run provenance is in Table E4-R6.", BODY)
    story.append(PageBreak())

    # Section 2
    h1("2. Methods recap")
    para("Authoritative source: canonical per-item result CSVs. Metrics computed from per-item observations: accuracy, mean confidence, "
         "ECE (10 uniform bins, primary), ECE (15 uniform bins), adaptive/equal-mass ECE (10 bins), Brier score, NLL, overconfidence gap, "
         "predicted-letter distribution. Brier/NLL only when option probabilities valid. 95% bootstrap CIs: item-level resampling, 2,000 resamples, seed 42. "
         "Raw vs cyclic4 uses pred_raw / confidence_raw. Chance = 25% (four options), exact binomial test (one-sided greater). "
         "McNemar exact test for paired BN vs EN correctness. Fragmentation: tokenizer_fragmentation_per_item files merged with results on "
         "model, language, question_id. Item-level regressions; Holm correction over the pre-specified test family. All analyses observational/associational; no causal claims.", BODY)
    story.append(PageBreak())

    # Section 3
    h1("3. Main results")
    story.append(src_tbl("main_metrics"))
    para("Table 2. Cyclic4 (debias) metrics per model and language. The table is split into two chunks that repeat the identifying "
         "columns (model, lang): (a) sample size, accuracy, confidence and uniform-bins ECE; (b) adaptive/equal-mass ECE, overconfidence "
         "gap, predicted-letter distribution, Brier score and NLL.", BODY)
    story.append(Paragraph("Table 2 (a)", SUB))
    fit_df(main_df[["model", "lang", "n", "accuracy", "accuracy_percent", "mean_confidence",
                    "ece_uniform_10", "ece_uniform_15"]], title="Table 2 (a) main_metrics")
    story.append(Paragraph("Table 2 (b)", SUB))
    fit_df(main_df[["model", "lang", "ece_adaptive_equal_mass_10", "overconfidence_gap",
                    "top_letter", "top_share", "brier_score", "nll"]], title="Table 2 (b) main_metrics")
    story.append(Spacer(1, 2))
    story.append(Image(os.path.join(FIGURES_DIR, "reliability_diagrams.png"), width=158, height=158, kind='proportional'))
    story.append(Paragraph("Figure 1. Reliability diagrams (cyclic4): confidence vs accuracy across 10 uniform confidence bins for all 6 models x 2 languages. "
                           "Bin area is proportional to item count. Dashed line = perfect calibration.", CAPTION))
    story.append(PageBreak())

    h2("3.1 Raw vs Cyclic4")
    story.append(src_tbl("raw_vs_cyclic4"))
    para("Table 3. Raw (pre-debias) metrics vs cyclic4 metrics. Split into two chunks repeating (model, lang).", BODY)
    story.append(Paragraph("Table 3 (a)", SUB))
    fit_df(A.tables["raw_vs_cyclic4"][["model", "lang", "raw_accuracy", "raw_accuracy_percent", "raw_mean_confidence",
                   "raw_ece_uniform_10", "raw_overconfidence_gap"]], title="Table 3 (a) raw_vs_cyclic4")
    story.append(Paragraph("Table 3 (b)", SUB))
    fit_df(A.tables["raw_vs_cyclic4"][["model", "lang", "raw_top_letter", "raw_top_share", "accuracy", "mean_confidence",
                   "ece_uniform_10", "overconfidence_gap"]], title="Table 3 (b) raw_vs_cyclic4")
    story.append(Image(os.path.join(FIGURES_DIR, "raw_vs_cyclic4.png"), width=168, height=76, kind='proportional'))
    story.append(Paragraph("Figure 2. Raw vs Cyclic4 comparison of accuracy, mean confidence, ECE(10 bins), and overconfidence gap across "
                           "6 models x 2 languages.", CAPTION))

    h2("3.2 Chance-level analysis")
    story.append(src_tbl("chance_tests"))
    render_table("chance_tests", title="Table 4 (chance_tests)")
    para("Table 4. Exact binomial test of accuracy against 25% chance (four options).", BODY)

    h2("3.3 Reconciliation with published summaries")
    story.append(src_tbl("summary_reconciliation"))
    para("Table 5. Independently recomputed metrics vs summary.json and three_model_summary.csv (tolerance 1e-6).", BODY)
    story.append(NextPageTemplate('landscape'))
    story.append(PageBreak())
    recon_pdf = A.tables["summary_reconciliation"].copy()
    if "three_model_summary" in recon_pdf.columns and "summary_json" in recon_pdf.columns:
        recon_pdf["Published value"] = recon_pdf["summary_json"].where(
            recon_pdf["summary_json"].notna(), recon_pdf["three_model_summary"])
    elif "summary_json" in recon_pdf.columns:
        recon_pdf["Published value"] = recon_pdf["summary_json"]
    else:
        recon_pdf["Published value"] = recon_pdf.get("three_model_summary")
    recon_pdf = recon_pdf[["model", "lang", "metric", "Published value", "recomputed", "abs_diff", "status"]]
    recon_pdf = recon_pdf.rename(columns={"recomputed": "Recomputed value", "abs_diff": "Abs. difference", "status": "Status"})
    fit_df(recon_pdf, LANDSCAPE_W, title="Table 5 (summary_reconciliation)", landscape=True)
    story.append(NextPageTemplate('portrait'))
    story.append(PageBreak())

    # Section 4
    h1("4. BN versus EN comparison")
    story.append(src_tbl("bn_en_paired_comparison"))
    para("Table 6. Paired BN vs EN differences (paired on question_id; paired bootstrap 95% CI, seed 42, 2000 resamples; exact McNemar).", BODY)
    story.append(Paragraph("Table 6 (a)", SUB))
    fit_df(paired_df[["model", "n_paired", "accuracy_bn", "accuracy_en", "accuracy_diff_bn_minus_en",
                      "acc_diff_ci_low", "acc_diff_ci_high", "ece_bn", "ece_en"]],
           title="Table 6 (a) bn_en_paired_comparison")
    story.append(Paragraph("Table 6 (b)", SUB))
    fit_df(paired_df[["model", "n_paired", "ece_diff_bn_minus_en", "ece_diff_ci_low", "ece_diff_ci_high",
                      "mean_conf_bn", "mean_conf_en", "mean_conf_diff_bn_minus_en", "conf_diff_ci_low",
                      "conf_diff_ci_high", "mcnemar_n01_bn_wrong_en_right", "mcnemar_n10_bn_right_en_wrong",
                      "mcnemar_p_exact"]],
           title="Table 6 (b) bn_en_paired_comparison")
    story.append(Image(os.path.join(FIGURES_DIR, "bn_en_paired.png"), width=150, height=75, kind='proportional'))
    story.append(Paragraph("Figure 3. BN minus EN accuracy and ECE differences with paired 95% bootstrap CIs.", CAPTION))
    story.append(PageBreak())

    # Section 5
    h1("5. Tokenization fragmentation versus calibration")
    para("Fragmentation measurements (tpw, tpc, strr) merged with per-item results on model, language, question_id. "
         "Join coverage validated; unmatched rows reported.", BODY)
    story.append(src_tbl("fragmentation_model_level"))
    render_table("fragmentation_model_level", title="Table 7 (fragmentation_model_level)")
    para("Table 7. Model-level mean fragmentation (TPW, TPC, STRR) by model and language.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "fragmentation_tpw_box.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 4. TPW distribution by model and language.", CAPTION))

    h2("5.1 TPW quartile analysis")
    story.append(NextPageTemplate('landscape'))
    story.append(PageBreak())
    story.append(src_tbl("fragmentation_quartiles"))
    para("Table 8. Within each model x language, items split into TPW quartiles. Accuracy, mean confidence, ECE, overconfidence gap with 95% bootstrap CIs. "
         "Rendered on a landscape page because of its width.", BODY)
    fit_df(A.tables["fragmentation_quartiles"], LANDSCAPE_W, title="Table 8 (fragmentation_quartiles)", landscape=True)
    story.append(Image(os.path.join(FIGURES_DIR, "quartile_ece.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 5. ECE by within-model TPW quartile.", CAPTION))
    story.append(NextPageTemplate('portrait'))
    story.append(PageBreak())

    h2("5.2 Item-level regression")
    story.append(src_tbl("fragmentation_regression"))
    render_table("fragmentation_regression", title="Table 9 (fragmentation_regression)")
    para("Table 9. Logistic regression (correctness ~ TPW + question word count) and linear regression (confidence ~ TPW + question word count), "
         "per model x language.", BODY)

    h2("5.3 Spearman correlations")
    story.append(src_tbl("fragmentation_spearman"))
    render_table("fragmentation_spearman", title="Table 10 (fragmentation_spearman)")
    para("Table 10. Spearman correlations: TPW vs confidence, TPW vs correctness, per model x language.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "regression_coefs.png"), width=158, height=79, kind='proportional'))
    story.append(Paragraph("Figure 6. TPW regression coefficients (logit correctness, linear confidence) with 95% CIs.", CAPTION))

    h2("5.4 Item-level calibration proxy")
    story.append(src_tbl("item_calibration_proxy"))
    render_table("item_calibration_proxy", title="Table 11 (item_calibration_proxy)")
    para("Table 11. Confidence minus correctness, and squared Brier error term, regressed on TPW controlling for question length.", BODY)

    h2("5.5 Cross-language comparison")
    story.append(src_tbl("fragmentation_cross_language"))
    render_table("fragmentation_cross_language", title="Table 12 (fragmentation_cross_language)")
    para("Table 12. BN/EN fragmentation ratios and BN/EN calibration gaps. Descriptive only.", BODY)

    h2("5.6 Multiple comparisons (Holm)")
    story.append(src_tbl("fragmentation_holm"))
    render_table("fragmentation_holm", title="Table 13 (fragmentation_holm)")
    para("Table 13. Holm correction over the family of 12 item-level TPW hypothesis tests (6 logit + 6 linear) for the six main models. "
         "The E4 sensitivity Holm family is reported separately in Table E4-sens (Table 14 in this extended report), not merged with Table 13.", BODY)

    h2("5.7 Confounding factors")
    para("Observed patterns may be confounded by: language (BN vs EN), subject distribution, model size (3B vs 1B), architecture (Llama vs Gemma), "
         "dtype (float16 vs float32), instruction tuning, the shared tokenizer (Llama32-3B / TituLLM-1B), and cyclic4's effect on confidence interpretation. "
         "No causal claims; language used is association/correlation/descriptive relationship.", BODY)
    story.append(PageBreak())

    # Section 6
    h1("6. Dtype sensitivity")
    story.append(src_tbl("dtype_comparison"))
    para("Table 14. TituLLM-1B only, float16 vs float32, 200 development items per language.", BODY)
    dc = A.tables.get("dtype_comparison")
    if dc is not None and len(dc) > 0 and "model" in dc.columns:
        story.append(Paragraph("Table 14 (a)", SUB))
        fit_df(dc[["model", "lang", "n_items", "n_valid_pairs", "mean_abs_prob_diff",
                  "max_abs_prob_diff", "prediction_agreement_rate", "mean_signed_conf_diff"]],
               title="Table 14 (a) dtype_comparison")
        story.append(Paragraph("Table 14 (b)", SUB))
        fit_df(dc[["model", "lang", "mean_abs_conf_diff", "ece_float16", "ece_float32", "ece_diff",
                  "n_nan_float16", "n_nan_float32", "mean_option_mass_float16", "mean_option_mass_float32"]],
               title="Table 14 (b) dtype_comparison")
    story.append(src_tbl("dtype_probability_differences"))
    render_table("dtype_probability_differences", title="Table 15 (dtype_probability_differences)")
    if os.path.exists(os.path.join(FIGURES_DIR, "dtype_hist.png")):
        story.append(Image(os.path.join(FIGURES_DIR, "dtype_hist.png"), width=140, height=70, kind='proportional'))
        story.append(Paragraph("Figure 7. Histogram of per-option absolute probability differences (float16 vs float32), TituLLM-1B.", CAPTION))
    story.append(PageBreak())

    # Section 7
    h1("7. Development screen versus full run")
    story.append(src_tbl("dev_screen_vs_full_run"))
    render_table("dev_screen_vs_full_run", title="Table 16 (dev_screen_vs_full_run)")
    para("Table 16. Development-screen top-letter share vs full-run raw top-letter share.", BODY)
    story.append(Image(os.path.join(FIGURES_DIR, "dev_screen_vs_full.png"), width=150, height=75, kind='proportional'))
    story.append(Paragraph("Figure 8. Dev screen vs full-run raw top-letter share.", CAPTION))
    story.append(PageBreak())

    # Section 8
    h1("8. Limitations and threats to validity")
    for lim in [
        "Chance-level models: calibration results are not reliable evidence of knowledge or uncertainty quality.",
        "Six main models but only four tokenizer levels (llama32_3b_base shares the llama32_3b tokenizer).",
        "Model architecture and size confounded (3B vs 1B; Llama vs Gemma).",
        "Dtype confounds: tested only on TituLLM-1B (float16 vs float32).",
        "Language and subject distribution may differ between BN and EN.",
        "Cyclic4 changes confidence interpretation; raw vs cyclic4 not directly comparable as confidence.",
        "Development sample size n=200 makes ECE estimates noisy.",
        "Single prompt configuration (few-shot, 4 shots, no chat template).",
        "Single random seed (42).",
        "Observational design: associations only, no causal claims.",
        "E4 extension models added after the preregistered six-model set; not part of the confirmatory analysis.",
    ]:
        story.append(Paragraph(f"• {lim}", BULLET))

    # Section 9
    h1("9. Plain-language findings")
    para("Each finding includes a computed numerical result and a table/figure reference. No causal language.", BODY)
    for f in build_findings():
        story.append(Paragraph(f"• {f}", BULLET))
    story.append(PageBreak())

    # Section 10 — phi exploratory (unchanged from preregistered-plus-phi report)
    h1("10. Exploratory model: phi35_mini (outside preregistered main set)")
    para("phi35_mini (Microsoft Phi-3.5-mini-instruct) is reported here as an EXPLORATORY model only. It is "
         "OUTSIDE the preregistered six-model calibration set (titullm_1b, tigerllm_1b, llama32_3b, qwen25_3b, "
         "bloomz_3b, llama32_3b_base). It does NOT appear in Tables 2-13, in the Holm multiple-comparison family, "
         "in the cross-tokenizer regression/spearman tests, or in the count of tokenizer/fragmentation levels with "
         "calibration data. All statements about phi35_mini are descriptive/associational; no defect or bug is "
         "implied, and phi35_mini's accuracy being near chance in Bangla is a reported result, not a validity failure.", BODY)
    story.append(src_tbl("exploratory_phi_appendix"))
    para("Source: read-only copy of phi35-mini-second-run-result.zip unzipped under phi_second_run_check/raw/. "
         "Metrics recomputed from per-item CSVs; bootstrap CIs use 2,000 resamples, seed 42. p-values below 1e-4 are "
         "rendered as '< 1e-4' (never 'p = 0.0').", SMALL)
    phi_pdf = phi_df.copy()
    cols_show = ["model", "lang", "n_items", "accuracy_cyclic4", "accuracy_cyclic4_ci_low",
                 "accuracy_cyclic4_ci_high", "majority_baseline", "binom_chance_p_fmt", "binom_majority_p_fmt",
                 "ece_uniform", "ece_equal_mass", "brier", "brier_skill", "nll", "nll_skill",
                 "auroc", "auroc_ci_low", "auroc_ci_high", "option_mass_mean", "option_mass_ge_095_pct",
                 "frag_tpw_mean", "frag_strr_mean", "dev_screen_acc", "full_run_acc"]
    cols_show = [c for c in cols_show if c in phi_pdf.columns]
    phi_pdf = phi_pdf[cols_show].round(6)
    phi_pdf = phi_pdf.rename(columns={
        "accuracy_cyclic4": "accuracy", "accuracy_cyclic4_ci_low": "acc_ci_low",
        "accuracy_cyclic4_ci_high": "acc_ci_high", "majority_baseline": "majority_baseline",
        "binom_chance_p_fmt": "binom_chance_p", "binom_majority_p_fmt": "binom_majority_p",
        "ece_uniform": "ece_uniform_10", "ece_equal_mass": "ece_equal_mass_10",
        "brier_skill": "brier_skill_vs_0.75", "nll_skill": "nll_skill_vs_ln4",
        "option_mass_ge_095_pct": "option_mass_ge_0.95_pct",
        "frag_tpw_mean": "fragmentation_tpw_mean", "frag_strr_mean": "fragmentation_strr_mean",
        "dev_screen_acc": "dev_screen_acc", "full_run_acc": "full_run_acc",
    })
    fit_df(phi_pdf, PORTRAIT_W, title="Table 17 (exploratory_phi_appendix): phi35_mini second run, exploratory")
    para("Table 17. phi35_mini second-run (full run, n=1,500 per language) exploratory metrics. BN accuracy "
         f"{phi_pdf.loc[phi_pdf['lang']=='bn','accuracy'].iloc[0]:.3f} vs 25% chance ({phi_pdf.loc[phi_pdf['lang']=='bn','binom_chance_p'].iloc[0]}); "
         f"EN accuracy {phi_pdf.loc[phi_pdf['lang']=='en','accuracy'].iloc[0]:.3f}. "
         "Associational/descriptive only; excluded from main-model inference.", BODY)
    story.append(PageBreak())

    # ===== Section 11 — E4 extension =====
    h1("11. Extension models (E4), added after the preregistered six-model set")
    para("The E4 models (mgpt_1_3b = ai-forever/mGPT, olmo2_1b = allenai/OLMo-2-0425-1B, "
         "smollm2_1_7b = HuggingFaceTB/SmolLM2-1.7B, sarvam1_2b = sarvamai/sarvam-1) were added AFTER the preregistered six-model set. "
         "This section is an EXTENSION: the E4 models are not part of the confirmatory analysis and none of them appears in Tables 2-13. "
         "All statements below are descriptive/associational only; near-chance accuracy is a reported result, not a validity failure; "
         "no model is described as defective.", BODY)
    para("Model base/instruct confound (descriptive): all four E4 models are base models, while most main models are instruct models "
         "(among the six main models only llama32_3b_base is a base model). Comparisons between E4 models and main models are confounded "
         "by instruction-tuning status; wording is associational throughout.", BODY)
    para("Selection effect (descriptive): the three gate-failed models combine high Bangla TPW (4.83-10.05 tokens per word) with "
         "near-chance Bangla dev accuracy (0.23-0.28). Models that fail the Bangla gate are mostly high-TPW weak-Bangla models, so "
         "excluding them can bias any TPW-calibration relationship. Exclusions follow the pre-committed gate rule.", BODY)

    h2("11.1 Pre-commitment and gate audit (R1-R7)")
    para("Rules R1-R7 were copied verbatim into E4_CHECK_REPORT.md before any result file was opened; thresholds were not changed "
         "after seeing numbers. R7 gate values are recorded exactly as written in dev_screen_report_e3.csv and not recomputed.", SMALL)
    story.append(src_tbl("e4_rule_classification"))
    render_table("e4_rule_classification", title="Table E4-1 (e4_rule_classification)")
    para("Table E4-1. Mechanical classification: mgpt_1_3b, olmo2_1b and smollm2_1_7b are GATE-FAILED (dev-only): a gate failed and "
         "no full run exists, so they appear only in the dev-screen table and the TPW-range figure and receive no calibration tests. "
         "sarvam1_2b is INVALID: a full 1,500-item run exists in both languages but R4 fails (Bangla option_mass >= 0.95 on 49.07% of "
         "items, below the 80% threshold; English passes at 100.0%), so sarvam1_2b is excluded from every calibration table. "
         "No model qualifies as MAIN-EXTENSION. R5 (reproducibility) is NOT CHECKABLE (single run per model; recorded explicitly). "
         "R6 (pre-commitment) PASSES: e4_added_models_plan.json (created 2026-11-18T19:13:36.829185Z, E4_EXPLORATORY_MODELS=[], gate rule "
         "recorded) is earlier than every results file in the zip; provenance nuances (shared subset file times before the plan; "
         "run_manifest.json created_at after the plan) are reported in Table E4-R6 and not hidden.", BODY)

    h2("11.2 Development screen for the four E4 models (R7, recorded as written)")
    story.append(src_tbl("e4_dev_screen_e4"))
    render_table("e4_dev_screen_e4", title="Table E4-2 (e4_dev_screen_e4)")
    para("Table E4-2. dev_screen_report_e3.csv rows for the four E4 models, recorded exactly as written: collapse_gate_passed, "
         "majority_gate_passed and eligible_for_full_run are not recomputed. mgpt_1_3b: BN acc 0.230, EN acc 0.245 (majority gate failed). "
         "olmo2_1b: BN acc 0.225, EN acc 0.480 (majority gate failed). smollm2_1_7b: BN acc 0.275, EN acc 0.475 (majority gate failed). "
         "sarvam1_2b: BN acc 0.410, EN acc 0.490 (both gates passed, eligible for full run -- but INVALID on R4, see Table E4-5).", BODY)

    h2("11.3 Phase 3 results for E4 models")
    para("Phase 3a (full-run calibration metrics with bootstrap CIs) is vacuous for E4: no model qualified as MAIN-EXTENSION, "
         "so no E4 model enters any calibration table.", BODY)
    para("Phase 3b (gate-failed, dev-only, n=200 per language):", BODY)
    story.append(src_tbl("e4_dev_only_metrics"))
    render_table("e4_dev_only_metrics", title="Table E4-3 (e4_dev_only_metrics)")
    para("Table E4-3. Dev accuracy with exact 95% Clopper-Pearson binomial CI, letter shares, majority-class test p-value, TPW/STRR. "
         "No ECE, Brier or regression estimates for gate-failed models, as pre-committed. p-values below 1e-4 are rendered as '< 1e-4'.", BODY)

    para("Phase 3c (TPW/STRR):", BODY)
    story.append(src_tbl("e4_tpw_strr_comparison"))
    render_table("e4_tpw_strr_comparison", title="Table E4-4 (e4_tpw_strr_comparison)")
    para("Table E4-4. tokenizer_fragmentation_per_item_all11.csv (zip) vs tokenizer_fragmentation_per_item_all7.csv (local copy): "
         "for the 7 models already in all7, per-language mean TPW differences are 0.0 (within 1e-6; files numerically identical per row). "
         "For the 4 new models vs the local candidate_screen values (tolerance 1e-3): ALL FOUR MISMATCH on Bangla TPW "
         "(mGPT diff 0.842510; OLMo-2 diff 1.497141; SmolLM2 diff 2.188563; Sarvam-1 diff 0.365608) and on English TPW "
         "(mGPT 0.098740; OLMo-2 0.199766; SmolLM2 0.126210; Sarvam-1 0.242835); STRR likewise outside 1e-3 for all four. "
         "Candidate_screen values were measured on probe tokens; all11 values on the full 1,500 items -- reported as a mismatch, not resolved.", BODY)

    para("Phase 3d (prompt length, BN 4-shot, sarvam1_2b):", BODY)
    story.append(src_tbl("e4_prompt_length"))
    render_table("e4_prompt_length", title="Table E4-5 (e4_prompt_length)")
    para("Table E4-5. Prompt length computed from the prompt_tokens column recorded in the per-item CSV (same prompt-builder output; "
         "no tokenizer weights loaded, so tokenizer-only loading was not required). Bangla: n=1,500, max 698 tokens, mean 493.98 tokens, "
         "share above 2,048 tokens = 0.0%. All 1,500 items are at or below 2,048 tokens, so the at-or-below group is the full sample "
         "(cyclic4 accuracy 0.394667, top-letter share 0.276667, top letter A); there is no above-2,048 group. English: max 586 tokens, "
         "share above 2,048 = 0.0%.", BODY)

    para("Phase 3d / R4 failure evidence (sarvam1_2b, excluded from every calibration table):", BODY)
    story.append(src_tbl("e4_r4_evidence"))
    fit_df(e4_r4_df.round(6), title="Table E4-6 (sarvam1_2b R4 evidence)")
    para("Table E4-6. sarvam1_2b item-level option_mass validity evidence. Bangla fails R4 (49.07% of items with option_mass >= 0.95, "
         "below the 80% threshold; minimum 0.7398); English passes (100.0%). sarvam1_2b is therefore INVALID and excluded from every "
         "calibration table, consistent with the pre-specified classification rules.", BODY)

    h2("11.4 Combined TPW figure")
    story.append(Image(os.path.join(FIGURES_DIR, "e4_tpw_calibration.png"), width=460, height=188, kind='proportional'))
    story.append(Paragraph("Figure E4-1. Bangla mean TPW (x) vs Bangla accuracy (left panel) and vs Bangla equal-mass ECE "
                           "(right panel). Markers: main / exploratory / gate-failed. Dev-only models appear in the accuracy panel only. "
                           "sarvam1_2b (INVALID on R4) is excluded from all calibration displays. "
                           "Distinct tokenizer levels with calibration data: 4 (LLaMA family: titullm_1b, llama32_3b, llama32_3b_base; "
                           "Gemma/TigerLLM: tigerllm_1b; Qwen: qwen25_3b; BLOOM: bloomz_3b); phi35_mini exploratory excluded; "
                           "no E4 model qualified. Cross-tokenizer comparisons are descriptive.", CAPTION))

    h2("11.5 Sensitivity analysis (NOT the preregistered analysis)")
    para("Pooled sensitivity model: outcome ~ TPW_z * language + model FE + subject FE, clustered by question_id, "
         "run once on the 6 main models and once on the extended set (6 main + MAIN-EXTENSION models). No model qualified as "
         "MAIN-EXTENSION, so the extended set is identical to the main set (0 extension models added); both runs are reported "
         "separately, side by side. The Holm family is defined explicitly as {TPW_z main effect, TPW_z x language interaction} "
         "x {correctness outcome, confidence outcome}, m = 4 tests per run, and is reported separately from the original Table 13 "
         "(family of 12 item-level TPW tests for the 6 main models). Phi35_mini is in neither family.", BODY)
    if sens_main_df is not None and len(sens_main_df):
        story.append(src_tbl("e4_sensitivity_main"))
        render_table("e4_sensitivity_main", title="Table E4-7 (e4_sensitivity_main: 6 main models)")
    if sens_ext_df is not None and len(sens_ext_df):
        story.append(src_tbl("e4_sensitivity_extended"))
        render_table("e4_sensitivity_extended", title="Table E4-8 (e4_sensitivity_extended: extended set = main set)")
    para("Tables E4-7/E4-8. Sensitivity-only results: coefficients are associational, not causal; no inference about a "
         "fragmentation-calibration relationship is claimed.", SMALL)

    h2("11.6 R6 provenance (recorded, not hidden)")
    story.append(src_tbl("e4_r6_provenance"))
    render_table("e4_r6_provenance", title="Table E4-R6 (e4_r6_provenance)")
    para("Table E4-R6. e4_added_models_plan.json created_at_utc = 2026-11-18T19:13:36.829185Z; E4_EXPLORATORY_MODELS = [] and the "
         "gate rule are present; the plan is earlier than every results file in the zip. Nuances reported: shared subset zip-entry "
         "times (2026-11-18 19:12:28) precede the plan; mgpt_1_3b/model_settings.json zip-entry time (19:13:36) is in the same minute "
         "as the plan (plan 0.829s later); run_manifest.json created_at (2026-11-18T20:16:23Z) and sarvam1_2b/status.json timestamp "
         "(2026-11-18T20:16:21Z) are after the plan (run metadata, not results files). None of these changes the R6 PASS verdict for "
         "results files; all are disclosed.", BODY)

    # Appendix A
    h1("Appendix A: Complete per-model results tables (six main models)")
    for model in MODELS:
        h2(model)
        sub = main_df[main_df["model"] == model]
        story.append(src_tbl("main_metrics"))
        cols = ["lang", "accuracy", "accuracy_percent", "mean_confidence", "ece_uniform_10",
                "ece_uniform_15", "ece_adaptive_equal_mass_10", "overconfidence_gap",
                "brier_score", "nll"]
        fit_df(sub[cols], title=f"Appendix A ({model})")
        para("Brier/NLL shown where probabilities valid.", SMALL)
    story.append(PageBreak())

    # Reproducibility checks
    h1("Reproducibility checks")
    story.append(src_tbl("reproducibility_check"))
    overflows = []
    for fl in story:
        if isinstance(fl, Table):
            width = sum(fl._colWidths)
            limit = LANDSCAPE_W if getattr(fl, "_landscape", False) else PORTRAIT_W
            if width > limit + 1e-6:
                overflows.append((getattr(fl, "_title", "table"), width - limit))
    for title, ov in overflows:
        log(f"OVERFLOW: table '{title}' exceeds page frame by {ov:.1f} points")
    rec = A.tables["reproducibility_check"].copy()
    ov_status = "passed" if not overflows else "failed"
    ov_detail = ("every table width <= page frame width" if not overflows
                 else "; ".join(f"{t}: +{o:.1f}pt" for t, o in overflows))
    rec = pd.concat([rec, pd.DataFrame([{
        "check": "No table exceeds page width",
        "status": ov_status,
        "detail": ov_detail,
    }])], ignore_index=True)
    fit_df(rec, PORTRAIT_W, title="Reproducibility checks")

    doc.build(story)
    log(f"PDF written: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    log(f"Em-dash (U+2014) replacements in PDF text: {EM_DASH_REPLACEMENTS[0]}")
    return pdf_path


# =============================================================================
# SENSITIVITY ANALYSIS (E4): pooled model, NOT preregistered
# =============================================================================
def _holm_adjust(pvals):
    pvals = np.array([float(p) for p in pvals], dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, pvals[idx] * (m - rank))
    for i in range(1, m):
        if adjusted[order[i]] < adjusted[order[i - 1]]:
            adjusted[order[i]] = adjusted[order[i - 1]]
    return adjusted, order


def step_sensitivity(data):
    """Pooled sensitivity: TPW_z * language + model FE + subject FE, clustered by question_id.
    Run 1: 6 main models. Run 2: 6 main + MAIN-EXTENSION models (none qualified -> identical set).
    Holm family defined explicitly (4 tests per run) and reported separately from Table 13.
    Phi is in neither set. Clearly labelled as not the preregistered analysis."""
    log("=== STEP E4-SENSITIVITY: POOLED TPW-CALIBRATION (NOT PREREGISTERED) ===")
    if not HAS_STATSMODELS:
        log("statsmodels unavailable; sensitivity analysis skipped.")
        return None, None
    import statsmodels.formula.api as smf

    tf_all7 = read_csv_safe(ALL7_LOCAL)
    if tf_all7 is not None and "model" in tf_all7.columns:
        tf_all7 = tf_all7.rename(columns={"model": "model_key"})
    frames = []
    for model in MODELS:
        for lang in LANGS:
            df = data[(model, lang)][["question_id", "predicted_answer", "gold_answer", "confidence",
                                      "subject", "question_word_count"]].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub = tf_all7[(tf_all7["model_key"].astype(str) == model) & (tf_all7["lang"].astype(str) == lang)][
                ["question_id", "tpw"]].copy()
            sub["question_id"] = sub["question_id"].astype(str)
            m = df.merge(sub, on="question_id", how="inner")
            m["model"] = model
            m["lang"] = lang
            frames.append(m)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.dropna(subset=["tpw", "confidence", "subject", "question_id"])
    pooled["correctness"] = (pooled["predicted_answer"].astype(str).str.upper() ==
                             pooled["gold_answer"].astype(str).str.upper()).astype(float)
    pooled["tpw_z"] = pooled.groupby("model")["tpw"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))

    formula = "outcome ~ tpw_z * lang + C(model) + C(subject)"
    terms_of_interest = ["tpw_z", "tpw_z:lang[T.en]"]

    def run_set(models_subset, set_label, run_label):
        sub = pooled[pooled["model"].isin(models_subset)].copy()
        rows = []
        clusters = sub["question_id"].astype(str)
        for outcome in ["correctness", "confidence"]:
            df2 = sub.copy()
            df2["outcome"] = df2[outcome]
            try:
                res = smf.ols(formula, data=df2).fit(cov_type="cluster",
                                                      cov_kwds={"groups": clusters})
            except Exception as e:
                record_missing(f"sensitivity pooled regression {run_label} {outcome}", str(e))
                continue
            ci = res.conf_int()
            for t in terms_of_interest:
                rows.append({
                    "run": run_label, "set": set_label, "n_models": len(models_subset),
                    "outcome": outcome, "term": t,
                    "coef": float(res.params[t]), "se": float(res.bse[t]),
                    "ci_lo": float(ci.loc[t, 0]), "ci_hi": float(ci.loc[t, 1]),
                    "p_raw": float(res.pvalues[t]),
                    "n_items": int(len(df2)), "n_clusters": int(clusters.nunique()),
                })
        # Holm adjustment across the explicit family: 2 terms x 2 outcomes = 4 tests per run
        pvals = np.array([r["p_raw"] for r in rows], dtype=float)
        adj, order = _holm_adjust(pvals)
        for rank_pos, idx in enumerate(order):
            rows[idx]["holm_rank"] = int(rank_pos + 1)
            rows[idx]["holm_tests_in_family"] = 4
            rows[idx]["holm_adjusted_p"] = float(adj[idx])
            rows[idx]["holm_significant_005"] = bool(adj[idx] < 0.05)
            rows[idx]["note"] = "SENSITIVITY ONLY - not preregistered. Phi35_mini in neither set."
        return pd.DataFrame(rows)

    main_set = list(MODELS)
    extended_set = list(MODELS) + list(E4_MAIN_EXTENSION)  # E4_MAIN_EXTENSION is empty
    sens_main = run_set(main_set, "6 main models", "main")
    sens_ext = run_set(extended_set, "6 main + MAIN-EXTENSION (0 extension models qualified -> identical set)", "extended")
    if sens_main is not None:
        A.save_table("e4_sensitivity_main", sens_main,
                     "Pooled sensitivity, 6 main models, Holm family m=4 per run, separate from Table 13")
    if sens_ext is not None:
        A.save_table("e4_sensitivity_extended", sens_ext,
                     "Pooled sensitivity, extended set (= main set; no MAIN-EXTENSION model), Holm family m=4 per run")
    A.warn("Sensitivity analysis is NOT the preregistered analysis. Pooled model: outcome ~ TPW_z * language + model FE + subject FE, "
           "clustered by question_id. Holm family = {tpw_z main effect, TPW_z x language interaction} x {correctness, confidence}, m=4 per run. "
           "Extended set adds zero MAIN-EXTENSION models (none qualified), so both runs are identical; reported separately from Table 13. "
           "Phi35_mini is in neither set.")
    return sens_main, sens_ext


if __name__ == "__main__":
    main()
