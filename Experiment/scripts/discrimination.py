#!/usr/bin/env python3
"""
e2_run/discrimination.py — Discrimination and near-chance diagnostics.

Per model x language (six main models, BN and EN):
  accuracy, mean confidence, std of confidence,
  AUROC of confidence vs correctness (bootstrap 95% CI, 1000 resamples, seed 42),
  AURC and selective accuracy at 50% coverage,
  ECE (10 uniform bins primary; equal-mass secondary) with bootstrap 95% CI,
  binomial test of accuracy vs 0.25 with near_chance flag
    (near_chance = |accuracy - 0.25| < 0.05; NOT based on the binomial p-value,
     since n=1500 makes even 0.305 significantly above 0.25),
  majority-letter test (reuses report_work tables_6models_plus_phi/chance_tests.csv logic),
  |accuracy - mean_confidence|,
  constant-confidence ECE baseline (ECE if always outputting mean confidence),
  raw confidence (confidence_raw) vs averaged cyclic4 confidence (mean, std, ECE).

Per-permutation ECE is NOT possible: per-item CSVs store averaged cyclic4 probs plus ONE
raw value (pred_raw/confidence_raw); no per-permutation probability columns exist.

Outputs one consolidated table (fixes_2026-11-21/tables/task_c_discrimination.csv)
with accuracy, ECE (uniform + equal-mass), and AUROC side by side, plus anomaly flags.
No interpretation; anomalies flagged only.

Reads from read-only per-item CSVs; writes only under fixes_2026-11-21/.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

E2_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if E2_RUN not in sys.path:
    sys.path.insert(0, E2_RUN)
import analysis_v2  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATHS = {
    "titullm_1b":  os.path.join(ROOT, "anaysis-cyclic-run-and-sensitivity-analysis-v4h",
                                "results_E2_cyclic4_titullm_1b"),
    "tigerllm_1b": os.path.join(ROOT, "anaysis-cyclic-run-and-sensitivity-analysis-v4h",
                                "results_E2_cyclic4_tigerllm_1b"),
    "llama32_3b":  os.path.join(ROOT, "anaysis-cyclic-run-and-sensitivity-analysis-v4h",
                                "results_E2_cyclic4_llama32_3b"),
    "qwen25_3b":   os.path.join(ROOT, "report_work", "extracted_6models_plus_phi",
                                "results_E3_qwen25_3b", "results_E3_qwen25_3b", "qwen25_3b"),
    "bloomz_3b":   os.path.join(ROOT, "report_work", "extracted_6models_plus_phi",
                                "results_E3_bloomz_3b", "results_E3_bloomz_3b", "bloomz_3b"),
    "llama32_3b_base": os.path.join(ROOT, "v4i-run-qwen-bloomz-llama-phi",
                                    "results_llama_base", "results_E3_llama32_3b_base",
                                    "llama32_3b_base"),
}
LANGS = ["bn", "en"]
OPTIONS = ["A", "B", "C", "D"]
CHANCE = 0.25
NEAR_CHANCE_THRESHOLD = 0.05  # near_chance = |accuracy - 0.25| < 0.05
SEED = 42
N_BOOT = 1000
ECE_BINS = 10


def load_item_df(model_dir, lang):
    return pd.read_csv(os.path.join(model_dir, f"{lang}_results.csv"))


def majority_letter_test(df):
    """Reuse chance_tests.csv logic: exact binomial vs majority-class baseline."""
    golds = df["gold_answer"].astype(str).str.upper().str.strip().to_numpy()
    gold_counts = pd.Series(golds).value_counts().reindex(["A", "B", "C", "D"]).fillna(0)
    total = float(gold_counts.sum())
    majority_baseline = float(gold_counts.max() / total) if total > 0 else 0.25
    preds = df["predicted_answer"].astype(str).str.upper().str.strip().to_numpy()
    corrects = (preds == golds)
    k = int(corrects.sum()); n = int(len(corrects))
    res = stats.binomtest(k, n, majority_baseline, alternative="greater")
    return {"majority_baseline": majority_baseline, "majority_binom_p": float(res.pvalue),
            "majority_significant": bool(res.pvalue < 0.05), "k": k, "n": n}


def aurc_selective(conf, correct, coverage=0.5):
    """AURC (area under risk-coverage) and selective accuracy at coverage."""
    n = len(conf)
    order = np.argsort(-conf, kind="mergesort")
    conf_s = conf[order]; corr_s = correct[order]
    # cumulative accuracy at each coverage point k=1..n
    cum_correct = np.cumsum(corr_s)
    k_vals = np.arange(1, n + 1)
    acc_k = cum_correct / k_vals
    risk_k = 1.0 - acc_k
    # AURC = mean risk across coverage points (trapezoidal normalized by n)
    aurc = float(np.mean(risk_k))
    k50 = max(1, int(round(n * coverage)))
    sel_acc = float(acc_k[k50 - 1])
    return aurc, sel_acc


def constant_conf_ece(conf, correct):
    """ECE if predictor always outputs its mean confidence (single occupied bin)."""
    mean_c = float(np.mean(conf))
    const_conf = np.full_like(conf, mean_c)
    ece, _ = analysis_v2.compute_ece_uniform(const_conf, correct, n_bins=ECE_BINS)
    return ece


def bootstrap_metric(metric_fn, data, n_resamples=N_BOOT, seed=SEED):
    """Bootstrap 95% CI for a metric function over item-level data."""
    rng = np.random.default_rng(seed)
    n = len(data)
    idx = np.arange(n)
    vals = []
    for _ in range(n_resamples):
        sample = rng.choice(idx, size=n, replace=True)
        vals.append(metric_fn(data[sample]))
    vals = np.array(vals, dtype=float)
    lo = float(np.percentile(vals, 2.5))
    hi = float(np.percentile(vals, 97.5))
    return lo, hi, float(np.mean(vals))


def analyze(model_dir, model, lang):
    df = load_item_df(model_dir, lang)
    n = len(df)

    golds = df["gold_answer"].astype(str).str.upper().str.strip().to_numpy()
    preds = df["predicted_answer"].astype(str).str.upper().str.strip().to_numpy()
    corrects = (preds == golds).astype(float)

    conf = df["confidence"].to_numpy(dtype=float)
    conf_raw = df["confidence_raw"].to_numpy(dtype=float)
    probs = np.stack([df[f"prob_{c}"].to_numpy(dtype=float) for c in OPTIONS], axis=1)

    # Core point estimates
    acc = float(np.mean(corrects))
    mconf = float(np.mean(conf))
    conf_std = float(np.std(conf, ddof=1)) if n > 1 else 0.0
    mconf_raw = float(np.mean(conf_raw))
    conf_raw_std = float(np.std(conf_raw, ddof=1)) if n > 1 else 0.0

    # ECE (uniform primary, equal-mass secondary)
    ece_uni, _ = analysis_v2.compute_ece_uniform(conf, corrects, n_bins=ECE_BINS)
    ece_em, _ = analysis_v2.compute_ece_equal_mass(conf, corrects, n_bins=ECE_BINS)

    # AUROC
    auroc = analysis_v2.compute_auroc(conf, corrects)

    # Selective accuracy @50 + AURC
    aurc, sel_acc = aurc_selective(conf, corrects, 0.5)

    # |accuracy - mean_confidence|
    acc_conf_gap = abs(acc - mconf)

    # Constant-confidence ECE baseline
    const_ece = constant_conf_ece(conf, corrects)

    # Binomial vs chance + near_chance flag (threshold-based, not p-value)
    k = int(corrects.sum())
    binom_res = stats.binomtest(k, n, CHANCE, alternative="greater")
    binom_p = float(binom_res.pvalue)
    near_chance = bool(abs(acc - CHANCE) < NEAR_CHANCE_THRESHOLD)

    # Majority-letter test
    maj = majority_letter_test(df)

    # Raw vs cyclic4 confidence comparison
    ece_raw, _ = analysis_v2.compute_ece_uniform(conf_raw, corrects, n_bins=ECE_BINS)

    # Bootstrap CIs (1000 resamples, seed 42)
    data = np.column_stack([conf, corrects])
    acc_lo, acc_hi, acc_boot = bootstrap_metric(
        lambda d: float(np.mean(d[:, 1])), data)
    ece_uni_lo, ece_uni_hi, ece_uni_boot = bootstrap_metric(
        lambda d: analysis_v2.compute_ece_uniform(d[:, 0], d[:, 1], n_bins=ECE_BINS)[0], data)
    ece_em_lo, ece_em_hi, ece_em_boot = bootstrap_metric(
        lambda d: analysis_v2.compute_ece_equal_mass(d[:, 0], d[:, 1], n_bins=ECE_BINS)[0], data)
    conf_lo, conf_hi, conf_boot = bootstrap_metric(
        lambda d: float(np.mean(d[:, 0])), data)
    auroc_lo, auroc_hi, auroc_boot = bootstrap_metric(
        lambda d: analysis_v2.compute_auroc(d[:, 0], d[:, 1]), data)

    return {
        "model": model,
        "lang": lang,
        "n": n,
        "accuracy": acc,
        "accuracy_percent": acc * 100.0,
        "mean_confidence": mconf,
        "confidence_std": conf_std,
        "acc_conf_gap": acc_conf_gap,
        "ece_uniform_10": ece_uni,
        "ece_equal_mass_10": ece_em,
        "constant_conf_ece_baseline": const_ece,
        "auroc": auroc,
        "aurc": aurc,
        "selective_accuracy_50": sel_acc,
        "binom_p_vs_0.25": binom_p,
        "significantly_above_0.25": bool(binom_p < 0.05),
        "near_chance": near_chance,
        "near_chance_threshold": NEAR_CHANCE_THRESHOLD,
        "majority_baseline": maj["majority_baseline"],
        "majority_binom_p": maj["majority_binom_p"],
        "majority_significant": maj["majority_significant"],
        "mean_conf_raw": mconf_raw,
        "confidence_raw_std": conf_raw_std,
        "ece_uniform_10_raw": ece_raw,
        # Bootstrap CIs
        "acc_ci_low": acc_lo, "acc_ci_high": acc_hi,
        "ece_uniform_ci_low": ece_uni_lo, "ece_uniform_ci_high": ece_uni_hi,
        "ece_equal_mass_ci_low": ece_em_lo, "ece_equal_mass_ci_high": ece_em_hi,
        "mean_conf_ci_low": conf_lo, "mean_conf_ci_high": conf_hi,
        "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
        "n_bootstrap": N_BOOT,
        "bootstrap_seed": SEED,
    }


def anomaly_flags(row):
    """Flag anomalies without interpretation."""
    flags = []
    if row["near_chance"]:
        flags.append(f"near_chance(|acc-0.25|<{NEAR_CHANCE_THRESHOLD})")
    if row["ece_uniform_10"] > 0.10:
        flags.append("high_uniform_ECE(>0.10)")
    if row["auroc"] < 0.60:
        flags.append("low_AUROC(<0.60)")
    if row["acc_conf_gap"] > 0.10:
        flags.append("large_acc_conf_gap(>0.10)")
    if abs(row["ece_uniform_10"] - row["ece_uniform_10_raw"]) > 0.02:
        flags.append("raw_vs_cyclic4_ECE_shift(>0.02)")
    return ";".join(flags) if flags else "none"


def main():
    rows = []
    for model, model_dir in MODEL_PATHS.items():
        for lang in LANGS:
            rows.append(analyze(model_dir, model, lang))
    df = pd.DataFrame(rows)

    out_dir = os.path.join(ROOT, "fixes_2026-11-21", "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "task_c_discrimination.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")

    df["anomaly"] = df.apply(anomaly_flags, axis=1)
    df.to_csv(out_path, index=False)

    print("\n=== Consolidated table (accuracy | ECE uniform | ECE equal-mass | AUROC) ===")
    pd.set_option("display.width", 250)
    print(df[["model", "lang", "accuracy", "ece_uniform_10", "ece_equal_mass_10",
              "auroc", "near_chance", "anomaly"]].to_string(index=False))

    print("\n=== Anomaly flags ===")
    anom = df[df["anomaly"] != "none"]
    if len(anom):
        print(anom[["model", "lang", "anomaly"]].to_string(index=False))
    else:
        print("(none)")

    return df


if __name__ == "__main__":
    df = main()
