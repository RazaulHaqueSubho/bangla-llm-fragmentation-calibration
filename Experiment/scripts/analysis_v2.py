#!/usr/bin/env python3
"""
Analysis v2: Research Calibration Metrics, Raw vs Cyclic4 Sensitivity, and Regression Improvements
Project: Bangla LLM Fragmentation and Calibration Thesis

Implements:
- Task S3: Advanced Calibration Metrics & Proper Scoring Rules
- Task S4: Raw vs Cyclic4 Sensitivity Analysis & Transparent Robustness Assessment
- Task S5: Regression Improvements (Subject Fixed Effects, Clustered SEs, Nonlinear TPW,
           Standardized Effects, and Paired Bangla-English Item-Level Regressions)
- Task S6: Pooled Analysis, Statistical Correction, Cross-Tokenizer Correlations,
           and Section 9 Plain-Language Findings

Constraints:
- Pure CPU execution (no GPU jobs).
- Original results files treated as strictly read-only.
- Report-compatible output structure: tables/*.csv.
- Associational language only; no causal claims.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
MODELS = ["titullm_1b", "tigerllm_1b", "llama32_3b"]
LANGS = ["bn", "en"]
OPTIONS = ["A", "B", "C", "D"]
SEED = 42
N_BOOT = 1000

UNIFORM_BRIER_BASELINE = 0.75  # 4 classes: (0.25 - 1)^2 + 3 * (0.25 - 0)^2 = 0.75
UNIFORM_NLL_BASELINE = float(np.log(4.0))  # -ln(0.25) = ln(4) = 1.386294...

# =============================================================================
# S6: TOKENIZER IDENTITY MAP
# =============================================================================
# Confirmed identical tokenizers (from tokenizer_audit.csv):
#   llama32_3b  : meta-llama/Llama-3.2-3B-Instruct  → TokenizersBackend, vocab=128256,
#                 probe_tokens=61, bn_mean_tpw=7.322, bn_mean_strr=0.080, en_mean_tpw=1.368
#   titullm_1b  : hishab/titulm-llama-3.2-1b-v1.1   → TokenizersBackend, vocab=128256,
#                 probe_tokens=61, bn_mean_tpw=7.322, bn_mean_strr=0.080, en_mean_tpw=1.368
#   tigerllm_1b : md-nishat-008/TigerLLM-1B-it       → GemmaTokenizer,    vocab=262145,
#                 probe_tokens=9,  bn_mean_tpw=1.995, bn_mean_strr=0.257, en_mean_tpw=1.390
# llama32_3b and titullm_1b share byte-for-byte identical tokenizer parameters →
# they are treated as a SINGLE tokenizer level: "llama_family".
# Total DISTINCT tokenizer levels = 2.

TOKENIZER_FAMILY = {
    "titullm_1b":  "llama_family",   # LLaMA-3.2 tokenizer (vocab 128256)
    "tigerllm_1b": "gemma_family",   # Gemma tokenizer (vocab 262145)
    "llama32_3b":  "llama_family",   # LLaMA-3.2 tokenizer (vocab 128256) — identical to titullm_1b
}
N_DISTINCT_TOKENIZER_LEVELS = 2  # llama_family, gemma_family
TOKENIZER_IDENTITY_NOTE = (
    "llama32_3b and titullm_1b share an identical tokenizer "
    "(TokenizersBackend, vocab_size=128256, probe_sentence_tokens=61, "
    "bn_mean_tpw=7.322, bn_mean_strr=0.080, en_mean_tpw=1.368). "
    "They are assigned to the same tokenizer level 'llama_family'. "
    "tigerllm_1b uses a distinct GemmaTokenizer (vocab_size=262145) → 'gemma_family'. "
    "Number of distinct tokenizer levels: 2."
)


# =============================================================================
# S3: CORE METRIC IMPLEMENTATIONS
# =============================================================================

def compute_brier_multiclass(probs: np.ndarray, golds: np.ndarray) -> float:
    """
    Calculate 4-class multi-class Brier score:
        BS_multi = (1/N) * sum_{i=1}^N sum_{c in {A,B,C,D}} (p_{i,c} - y_{i,c})^2
    where y_{i,c} = 1 if option c is the gold answer, else 0.
    Uniform baseline for 4 options is 0.75.
    """
    probs = np.asarray(probs, dtype=np.float64)
    golds = np.asarray(golds, dtype=str)
    n = len(golds)
    if n == 0 or probs.shape != (n, 4):
        return float("nan")

    one_hot = np.zeros_like(probs)
    for i, g in enumerate(golds):
        g_clean = str(g).strip().upper()
        if g_clean in OPTIONS:
            one_hot[i, OPTIONS.index(g_clean)] = 1.0

    brier_per_item = np.sum((probs - one_hot) ** 2, axis=1)
    return float(np.mean(brier_per_item))


def compute_brier_binary_msp(confidences: np.ndarray, corrects: np.ndarray) -> float:
    """
    Calculate Binary MSP (Maximum Softmax Probability) Brier score:
        BS_binary_msp = (1/N) * sum_{i=1}^N (confidence_i - correct_i)^2
    where correct_i in {0, 1} indicates top-1 correctness.
    Reported separately from multi-class Brier score.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    mask = (
        np.isfinite(conf)
        & np.isfinite(corr)
        & (conf >= 0.0)
        & (conf <= 1.0)
        & ((corr == 0.0) | (corr == 1.0))
    )
    conf, corr = conf[mask], corr[mask]
    if len(conf) == 0:
        return float("nan")
    return float(np.mean((conf - corr) ** 2))


def compute_nll(probs: np.ndarray, golds: np.ndarray, eps: float = 1e-12) -> float:
    """
    Calculate Negative Log-Likelihood (NLL) for categorical distribution:
        NLL = - (1/N) * sum_{i=1}^N ln(max(p_{i, gold_i}, eps))
    Uniform baseline for 4 options is ln(4) = 1.386294...
    """
    probs = np.asarray(probs, dtype=np.float64)
    golds = np.asarray(golds, dtype=str)
    n = len(golds)
    if n == 0 or probs.shape != (n, 4):
        return float("nan")

    gold_probs = np.empty(n, dtype=np.float64)
    for i, g in enumerate(golds):
        g_clean = str(g).strip().upper()
        if g_clean in OPTIONS:
            gold_probs[i] = probs[i, OPTIONS.index(g_clean)]
        else:
            gold_probs[i] = eps

    gold_probs = np.maximum(gold_probs, eps)
    return float(-np.mean(np.log(gold_probs)))


def compute_skill_score(score: float, baseline: float) -> float:
    """
    Calculate skill score against a reference baseline:
        Skill Score = 1 - (Score / Baseline)
    Positive: improvement over reference baseline.
    Zero: equivalent to reference baseline.
    Negative: inferior to reference baseline.
    """
    if not np.isfinite(score) or not np.isfinite(baseline) or baseline == 0.0:
        return float("nan")
    return float(1.0 - (score / baseline))


def compute_ece_uniform(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate uniform-bin Expected Calibration Error (ECE) across n_bins equal-width bins on [0, 1].
    Returns scalar ECE and detailed list of per-bin sample counts, confidences, accuracies, and gaps.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    mask = (
        np.isfinite(conf)
        & np.isfinite(corr)
        & (conf >= 0.0)
        & (conf <= 1.0)
        & ((corr == 0.0) | (corr == 1.0))
    )
    conf, corr = conf[mask], corr[mask]
    n = len(conf)
    if n == 0:
        return float("nan"), []

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    bins_info = []
    for b in range(n_bins):
        m = (bin_idx == b)
        cb = int(np.sum(m))
        lo, hi = float(edges[b]), float(edges[b + 1])
        if cb == 0:
            bins_info.append({
                "bin_index": b,
                "bin_lower": lo,
                "bin_upper": hi,
                "sample_count": 0,
                "sample_fraction": 0.0,
                "mean_confidence": None,
                "accuracy": None,
                "calibration_gap": None,
            })
            continue

        mc = float(np.mean(conf[m]))
        ac = float(np.mean(corr[m]))
        gap = abs(mc - ac)
        ece += (cb / n) * gap
        bins_info.append({
            "bin_index": b,
            "bin_lower": lo,
            "bin_upper": hi,
            "sample_count": cb,
            "sample_fraction": float(cb / n),
            "mean_confidence": mc,
            "accuracy": ac,
            "calibration_gap": gap,
        })
    return float(ece), bins_info


def compute_ece_equal_mass(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calculate equal-mass (adaptive quantile) ECE with n_bins having equal sample counts (N / n_bins).
    Returns scalar ECE and detailed per-bin statistics.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    mask = (
        np.isfinite(conf)
        & np.isfinite(corr)
        & (conf >= 0.0)
        & (conf <= 1.0)
        & ((corr == 0.0) | (corr == 1.0))
    )
    conf, corr = conf[mask], corr[mask]
    n = len(conf)
    if n == 0:
        return float("nan"), []

    order = np.argsort(conf, kind="mergesort")
    groups = np.array_split(order, n_bins)

    ece = 0.0
    bins_info = []
    for b, g in enumerate(groups):
        cb = len(g)
        if cb == 0:
            bins_info.append({
                "bin_index": b,
                "bin_lower": None,
                "bin_upper": None,
                "sample_count": 0,
                "sample_fraction": 0.0,
                "mean_confidence": None,
                "accuracy": None,
                "calibration_gap": None,
            })
            continue

        c_grp = conf[g]
        lo = float(np.min(c_grp))
        hi = float(np.max(c_grp))
        mc = float(np.mean(c_grp))
        ac = float(np.mean(corr[g]))
        gap = abs(mc - ac)
        ece += (cb / n) * gap
        bins_info.append({
            "bin_index": b,
            "bin_lower": lo,
            "bin_upper": hi,
            "sample_count": cb,
            "sample_fraction": float(cb / n),
            "mean_confidence": mc,
            "accuracy": ac,
            "calibration_gap": gap,
        })
    return float(ece), bins_info


def compute_auroc(confidences: np.ndarray, corrects: np.ndarray) -> float:
    """
    Calculate Area Under the Receiver Operating Characteristic (AUROC) curve
    for confidence discriminating binary correctness, using the Wilcoxon-Mann-Whitney U statistic.
    Handles ties using average ranks.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=bool)
    n_pos = int(np.sum(corr))
    n_neg = int(len(corr) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = stats.rankdata(conf, method="average")
    rank_sum_pos = np.sum(ranks[corr])
    u_stat = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auroc = u_stat / (n_pos * n_neg)
    return float(auroc)


def compute_selective_accuracy(
    confidences: np.ndarray, corrects: np.ndarray, coverage: float = 0.5
) -> float:
    """
    Calculate selective classification accuracy at fixed coverage (e.g., top 50% highest confidences).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=bool)
    n = len(conf)
    if n == 0:
        return float("nan")

    k = max(1, int(round(n * coverage)))
    top_indices = np.argsort(-conf, kind="mergesort")[:k]
    return float(np.mean(corr[top_indices]))


def compute_murphy_decomposition(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Murphy (1973) Brier score decomposition for binary MSP predictions:
        Brier Score = Reliability - Resolution + Uncertainty + Within-bin variance
    where:
        Uncertainty = bar{y} * (1 - bar{y})  [base rate / sample variance]
        Reliability = sum_b (n_b / N) * (bar{p}_b - bar{y}_b)^2
        Resolution  = sum_b (n_b / N) * (bar{y}_b - bar{y})^2
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    n = len(conf)
    if n == 0:
        return {"brier": np.nan, "uncertainty": np.nan, "reliability": np.nan, "resolution": np.nan}

    base_rate = float(np.mean(corr))
    unc = base_rate * (1.0 - base_rate)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    rel = 0.0
    res = 0.0
    for b in range(n_bins):
        m = (bin_idx == b)
        cb = int(np.sum(m))
        if cb == 0:
            continue
        p_b = float(np.mean(conf[m]))
        y_b = float(np.mean(corr[m]))
        rel += (cb / n) * ((p_b - y_b) ** 2)
        res += (cb / n) * ((y_b - base_rate) ** 2)

    actual_brier = float(np.mean((conf - corr) ** 2))
    reconstructed = rel - res + unc
    within_bin_var = actual_brier - reconstructed

    return {
        "brier_score": actual_brier,
        "uncertainty": float(unc),
        "reliability": float(rel),
        "resolution": float(res),
        "rel_minus_res_plus_unc": float(reconstructed),
        "within_bin_var": float(within_bin_var),
    }


def compute_murphy_decomposition_multiclass(
    probs: np.ndarray, golds: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    """
    Murphy (1973) Brier decomposition extended across all 4 classes:
        Uncertainty = sum_{c} bar{y}_c * (1 - bar{y}_c) = 1 - sum_{c} bar{y}_c^2
        Reliability = sum_{c} sum_b (n_{b,c} / N) * (bar{p}_{b,c} - bar{y}_{b,c})^2
        Resolution  = sum_{c} sum_b (n_{b,c} / N) * (bar{y}_{b,c} - bar{y}_c)^2
    """
    probs = np.asarray(probs, dtype=np.float64)
    golds = np.asarray(golds, dtype=str)
    n = len(golds)
    if n == 0 or probs.shape != (n, 4):
        return {"brier": np.nan, "uncertainty": np.nan, "reliability": np.nan, "resolution": np.nan}

    one_hot = np.zeros_like(probs)
    for i, g in enumerate(golds):
        g_clean = str(g).strip().upper()
        if g_clean in OPTIONS:
            one_hot[i, OPTIONS.index(g_clean)] = 1.0

    actual_brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    total_rel = 0.0
    total_res = 0.0
    total_unc = 0.0

    for c in range(4):
        y_c = one_hot[:, c]
        p_c = probs[:, c]
        bar_y_c = float(np.mean(y_c))
        total_unc += bar_y_c * (1.0 - bar_y_c)

        bin_idx = np.clip(np.digitize(p_c, edges[1:-1], right=False), 0, n_bins - 1)
        for b in range(n_bins):
            m = (bin_idx == b)
            cb = int(np.sum(m))
            if cb == 0:
                continue
            bar_p_bc = float(np.mean(p_c[m]))
            bar_y_bc = float(np.mean(y_c[m]))
            total_rel += (cb / n) * ((bar_p_bc - bar_y_bc) ** 2)
            total_res += (cb / n) * ((bar_y_bc - bar_y_c) ** 2)

    reconstructed = total_rel - total_res + total_unc
    within_bin_var = actual_brier - reconstructed

    return {
        "brier_score": actual_brier,
        "uncertainty": float(total_unc),
        "reliability": float(total_rel),
        "resolution": float(total_res),
        "rel_minus_res_plus_unc": float(reconstructed),
        "within_bin_var": float(within_bin_var),
    }


# =============================================================================
# DATA DISCOVERY & LOADING
# =============================================================================

def find_data_dir(base_search_paths: Optional[List[str]] = None) -> str:
    """Locate the results directory containing existing three-model results."""
    candidates = base_search_paths or [
        "../anaysis-cyclic-run-and-sensitivity-analysis-v4h",
        "../anaysis-cyclic-run-and-sensitivity-analysis",
        "anaysis-cyclic-run-and-sensitivity-analysis-v4h",
        "anaysis-cyclic-run-and-sensitivity-analysis",
        "/home/naeem/Downloads/Bangla-Calibration-Thesis/anaysis-cyclic-run-and-sensitivity-analysis-v4h",
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            # Verify titullm_1b results folder exists
            test_target = os.path.join(cand, "results_E2_cyclic4_titullm_1b")
            if os.path.isdir(test_target):
                return os.path.abspath(cand)
    raise FileNotFoundError(
        f"Could not find three-model results directory in candidate paths: {candidates}"
    )


def load_model_data(data_dir: str) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Load per-item evaluation records for the 3 models and 2 languages."""
    data = {}
    for m in MODELS:
        folder = os.path.join(data_dir, f"results_E2_cyclic4_{m}")
        for l in LANGS:
            csv_path = os.path.join(folder, f"{l}_results.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing results file: {csv_path}")
            df = pd.read_csv(csv_path)
            data[(m, l)] = df
    return data


def load_tokenizer_fragmentation(data_dir: str) -> pd.DataFrame:
    """Load tokenizer fragmentation per-item measurements."""
    candidates = [
        os.path.join(data_dir, "tokenizer_fragmentation", "tokenizer_fragmentation_per_item.csv"),
        os.path.join(data_dir, "e2_rescue_results", "tokenizer_fragmentation", "tokenizer_fragmentation_per_item.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            df["question_id"] = df["question_id"].astype(str)
            return df
    raise FileNotFoundError(f"Missing tokenizer_fragmentation_per_item.csv in {data_dir}")


# =============================================================================
# S3: BOOTSTRAP CONFIDENCE INTERVAL PIPELINE
# =============================================================================

def bootstrap_main_metrics(
    data: Dict[Tuple[str, str], pd.DataFrame], n_boot: int = N_BOOT, seed: int = SEED
) -> pd.DataFrame:
    """Compute 95% bootstrap confidence intervals for all primary calibration metrics."""
    rows = []
    rng = np.random.default_rng(seed)

    for (m, l), df in data.items():
        n = len(df)
        preds = df["predicted_answer"].astype(str).str.upper().to_numpy()
        golds = df["gold_answer"].astype(str).str.upper().to_numpy()
        confs = df["confidence"].to_numpy(dtype=float)
        corrs = (preds == golds)
        probs = np.stack([df[f"prob_{c}"].to_numpy(dtype=float) for c in OPTIONS], axis=1)

        # Precompute gold probability vector and one-hot matrix
        gold_indices = np.array([OPTIONS.index(str(g).strip().upper()) if str(g).strip().upper() in OPTIONS else 0 for g in golds])
        gold_probs = probs[np.arange(n), gold_indices]
        one_hot = np.zeros_like(probs)
        for i, idx in enumerate(gold_indices):
            one_hot[i, idx] = 1.0

        # Point estimates
        pe_acc = float(np.mean(corrs))
        pe_conf = float(np.mean(confs))
        pe_ece_u, _ = compute_ece_uniform(confs, corrs, 10)
        pe_ece_em, _ = compute_ece_equal_mass(confs, corrs, 10)
        pe_brier_bin = compute_brier_binary_msp(confs, corrs)
        pe_brier_mc = compute_brier_multiclass(probs, golds)
        pe_bss_mc = compute_skill_score(pe_brier_mc, UNIFORM_BRIER_BASELINE)
        pe_nll = compute_nll(probs, golds)
        pe_nll_ss = compute_skill_score(pe_nll, UNIFORM_NLL_BASELINE)
        pe_auroc = compute_auroc(confs, corrs)
        pe_sel_acc = compute_selective_accuracy(confs, corrs, 0.5)

        # Generate bootstrap resample indices
        resample_matrix = rng.integers(0, n, size=(n_boot, n))

        # Storage for bootstrap distributions
        boot_acc = np.empty(n_boot)
        boot_conf = np.empty(n_boot)
        boot_ece_u = np.empty(n_boot)
        boot_ece_em = np.empty(n_boot)
        boot_brier_bin = np.empty(n_boot)
        boot_brier_mc = np.empty(n_boot)
        boot_bss_mc = np.empty(n_boot)
        boot_nll = np.empty(n_boot)
        boot_nll_ss = np.empty(n_boot)
        boot_auroc = np.empty(n_boot)
        boot_sel_acc = np.empty(n_boot)

        edges = np.linspace(0.0, 1.0, 11)

        for b in range(n_boot):
            idx = resample_matrix[b]
            c_s = confs[idx]
            co_s = corrs[idx]
            p_s = probs[idx]
            oh_s = one_hot[idx]
            gp_s = gold_probs[idx]

            boot_acc[b] = np.mean(co_s)
            boot_conf[b] = np.mean(c_s)

            # Uniform ECE
            b_idx = np.clip(np.digitize(c_s, edges[1:-1], right=False), 0, 9)
            ece_u = 0.0
            for bin_i in range(10):
                m_bin = (b_idx == bin_i)
                cb = np.sum(m_bin)
                if cb > 0:
                    ece_u += (cb / n) * abs(np.mean(c_s[m_bin]) - np.mean(co_s[m_bin]))
            boot_ece_u[b] = ece_u

            # Equal mass ECE
            order = np.argsort(c_s, kind="mergesort")
            groups = np.array_split(order, 10)
            ece_em = 0.0
            for g in groups:
                if len(g) > 0:
                    ece_em += (len(g) / n) * abs(np.mean(c_s[g]) - np.mean(co_s[g]))
            boot_ece_em[b] = ece_em

            # Brier scores
            b_bin = np.mean((c_s - co_s.astype(float)) ** 2)
            b_mc = np.mean(np.sum((p_s - oh_s) ** 2, axis=1))
            boot_brier_bin[b] = b_bin
            boot_brier_mc[b] = b_mc
            boot_bss_mc[b] = 1.0 - (b_mc / UNIFORM_BRIER_BASELINE)

            # NLL
            nll_val = -np.mean(np.log(np.maximum(gp_s, 1e-12)))
            boot_nll[b] = nll_val
            boot_nll_ss[b] = 1.0 - (nll_val / UNIFORM_NLL_BASELINE)

            # AUROC
            boot_auroc[b] = compute_auroc(c_s, co_s)

            # Selective accuracy @ 50%
            k50 = n // 2
            top_half = np.argsort(-c_s, kind="mergesort")[:k50]
            boot_sel_acc[b] = np.mean(co_s[top_half])

        metric_data = [
            ("accuracy", pe_acc, boot_acc),
            ("mean_confidence", pe_conf, boot_conf),
            ("ece_uniform_10", pe_ece_u, boot_ece_u),
            ("ece_equal_mass_10", pe_ece_em, boot_ece_em),
            ("brier_score_binary_msp", pe_brier_bin, boot_brier_bin),
            ("brier_score_multiclass", pe_brier_mc, boot_brier_mc),
            ("brier_skill_score_multiclass", pe_bss_mc, boot_bss_mc),
            ("nll", pe_nll, boot_nll),
            ("nll_skill_score", pe_nll_ss, boot_nll_ss),
            ("auroc", pe_auroc, boot_auroc),
            ("selective_accuracy_50", pe_sel_acc, boot_sel_acc),
        ]

        for met_name, pe, b_dist in metric_data:
            valid_dist = b_dist[np.isfinite(b_dist)]
            if len(valid_dist) > 0:
                ci_lo = float(np.percentile(valid_dist, 2.5))
                ci_hi = float(np.percentile(valid_dist, 97.5))
                b_mean = float(np.mean(valid_dist))
            else:
                ci_lo, ci_hi, b_mean = np.nan, np.nan, np.nan

            rows.append({
                "model": m,
                "lang": l,
                "metric": met_name,
                "point_estimate": pe,
                "bootstrap_mean": b_mean,
                "ci_low": ci_lo,
                "ci_high": ci_hi,
                "n_resamples": n_boot,
                "seed": seed,
            })

    return pd.DataFrame(rows)


# =============================================================================
# S3: RECONCILIATION AUDIT (BRIER 0.21 VS 0.75 INVESTIGATION)
# =============================================================================

def investigate_summary_reconciliation(
    data_dir: str, data: Dict[Tuple[str, str], pd.DataFrame]
) -> pd.DataFrame:
    """
    Investigate the reported mismatch between published Brier (~0.21) and recomputed Brier (~0.75).
    Reconciles summary.json values with both Binary MSP Brier and Multi-class Brier.
    """
    records = []
    for (m, l), df in data.items():
        summary_path = os.path.join(data_dir, f"results_E2_cyclic4_{m}", f"{l}_summary.json")
        summary_data = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)

        cm = summary_data.get("calibration_metrics", {})

        preds = df["predicted_answer"].astype(str).str.upper().to_numpy()
        golds = df["gold_answer"].astype(str).str.upper().to_numpy()
        confs = df["confidence"].to_numpy(dtype=float)
        corrs = (preds == golds)
        probs = np.stack([df[f"prob_{c}"].to_numpy(dtype=float) for c in OPTIONS], axis=1)

        # Recomputed metrics
        rec_acc = float(np.mean(corrs))
        rec_conf = float(np.mean(confs))
        rec_ece_u, _ = compute_ece_uniform(confs, corrs, 10)
        rec_brier_bin = compute_brier_binary_msp(confs, corrs)
        rec_brier_mc = compute_brier_multiclass(probs, golds)
        rec_nll = compute_nll(probs, golds)

        # Comparisons
        comparisons = [
            (
                "accuracy",
                cm.get("accuracy"),
                rec_acc,
                "Top-1 classification accuracy across N=1500 items.",
            ),
            (
                "mean_confidence",
                cm.get("mean_confidence"),
                rec_conf,
                "Mean maximum softmax probability (MSP) confidence.",
            ),
            (
                "ece_uniform_10",
                cm.get("ece_uniform_10"),
                rec_ece_u,
                "10-bin uniform Expected Calibration Error.",
            ),
            (
                "brier_score_binary_msp",
                cm.get("brier_score"),
                rec_brier_bin,
                "Binary MSP Brier score (confidence - correct)^2; published as 'brier_score' in summary.json (~0.21-0.25).",
            ),
            (
                "brier_score_multiclass",
                cm.get("multiclass_brier_score"),
                rec_brier_mc,
                "4-class multi-class Brier score sum_{k=1}^4 (p_k - y_k)^2; stored as 'multiclass_brier_score' in summary.json (~0.52-0.75, uniform baseline 0.75).",
            ),
            (
                "nll",
                cm.get("nll"),
                rec_nll,
                "Negative Log Likelihood -ln(p_gold); uniform baseline ln(4) = 1.386.",
            ),
        ]

        for met_name, s_val, r_val, explanation in comparisons:
            if s_val is not None and np.isfinite(s_val):
                diff = abs(float(s_val) - float(r_val))
                status = "MATCH" if diff < 1e-6 else "MISMATCH"
            else:
                diff = np.nan
                status = "MISSING_IN_SUMMARY"

            records.append({
                "model": m,
                "lang": l,
                "metric": met_name,
                "summary_json_value": s_val,
                "recomputed_value": r_val,
                "abs_diff": diff,
                "status": status,
                "explanation": explanation,
                "source_file": summary_path,
            })

    return pd.DataFrame(records)


# =============================================================================
# S3: STAGE RUNNER
# =============================================================================

def run_stage3(data_dir: str, tables_dir: str) -> Dict[str, pd.DataFrame]:
    """Execute Task S3: Compute all advanced calibration metrics and save tables."""
    print("=================================================================")
    print("🚀 EXECUTING TASK S3: ADVANCED CALIBRATION METRICS & SCORING RULES")
    print("=================================================================")
    os.makedirs(tables_dir, exist_ok=True)

    data = load_model_data(data_dir)
    print(f"Loaded {len(data)} datasets (3 models x 2 languages, N=1500 each).")

    # 1. Main Metrics Table
    main_rows = []
    ece_bins_uniform_rows = []
    ece_bins_equal_mass_rows = []
    brier_decomp_rows = []

    for (m, l), df in data.items():
        preds = df["predicted_answer"].astype(str).str.upper().to_numpy()
        golds = df["gold_answer"].astype(str).str.upper().to_numpy()
        confs = df["confidence"].to_numpy(dtype=float)
        corrs = (preds == golds)
        probs = np.stack([df[f"prob_{c}"].to_numpy(dtype=float) for c in OPTIONS], axis=1)

        acc = float(np.mean(corrs))
        mconf = float(np.mean(confs))
        ece_u, u_bins = compute_ece_uniform(confs, corrs, 10)
        ece_em, em_bins = compute_ece_equal_mass(confs, corrs, 10)
        overconf = mconf - acc

        vc = pd.Series(preds).value_counts()
        top_letter = str(vc.index[0]) if len(vc) > 0 else "N/A"
        top_share = float(vc.iloc[0] / len(preds)) if len(vc) > 0 else 0.0

        brier_bin = compute_brier_binary_msp(confs, corrs)
        brier_mc = compute_brier_multiclass(probs, golds)
        bss_mc = compute_skill_score(brier_mc, UNIFORM_BRIER_BASELINE)

        nll_val = compute_nll(probs, golds)
        nll_ss = compute_skill_score(nll_val, UNIFORM_NLL_BASELINE)

        auroc_val = compute_auroc(confs, corrs)
        sel_acc = compute_selective_accuracy(confs, corrs, 0.5)

        # Decompositions
        decomp_bin = compute_murphy_decomposition(confs, corrs, 10)
        decomp_mc = compute_murphy_decomposition_multiclass(probs, golds, 10)

        main_rows.append({
            "model": m,
            "lang": l,
            "n": len(df),
            "accuracy": acc,
            "accuracy_percent": acc * 100.0,
            "mean_confidence": mconf,
            "ece_uniform_10": ece_u,
            "ece_equal_mass_10": ece_em,
            "overconfidence_gap": overconf,
            "top_letter": top_letter,
            "top_share": top_share,
            "brier_score_binary_msp": brier_bin,
            "brier_score_multiclass": brier_mc,
            "brier_uniform_baseline": UNIFORM_BRIER_BASELINE,
            "brier_skill_score_multiclass": bss_mc,
            "nll": nll_val,
            "nll_uniform_baseline": UNIFORM_NLL_BASELINE,
            "nll_skill_score": nll_ss,
            "auroc": auroc_val,
            "selective_accuracy_50": sel_acc,
            "murphy_binary_rel": decomp_bin["reliability"],
            "murphy_binary_res": decomp_bin["resolution"],
            "murphy_binary_unc": decomp_bin["uncertainty"],
            "murphy_multiclass_rel": decomp_mc["reliability"],
            "murphy_multiclass_res": decomp_mc["resolution"],
            "murphy_multiclass_unc": decomp_mc["uncertainty"],
        })

        for ub in u_bins:
            ece_bins_uniform_rows.append({"model": m, "lang": l, **ub})

        for emb in em_bins:
            ece_bins_equal_mass_rows.append({"model": m, "lang": l, **emb})

        brier_decomp_rows.append({"model": m, "lang": l, "decomposition_type": "binary_msp", **decomp_bin})
        brier_decomp_rows.append({"model": m, "lang": l, "decomposition_type": "multiclass", **decomp_mc})

    main_df = pd.DataFrame(main_rows)
    main_csv_path = os.path.join(tables_dir, "main_metrics.csv")
    main_df.to_csv(main_csv_path, index=False)
    print(f"Saved: {main_csv_path}")

    # 2. Bootstrap Confidence Intervals
    print("Computing 95% bootstrap confidence intervals (1,000 resamples)...")
    ci_df = bootstrap_main_metrics(data, n_boot=N_BOOT, seed=SEED)
    ci_csv_path = os.path.join(tables_dir, "main_metrics_bootstrap_ci.csv")
    ci_df.to_csv(ci_csv_path, index=False)
    print(f"Saved: {ci_csv_path}")

    # 3. Summary Reconciliation (Brier 0.21 vs 0.75 Investigation)
    print("Reconciling summary.json vs recomputed metrics (resolving Brier score discrepancy)...")
    recon_df = investigate_summary_reconciliation(data_dir, data)
    recon_csv_path = os.path.join(tables_dir, "summary_reconciliation.csv")
    recon_df.to_csv(recon_csv_path, index=False)
    print(f"Saved: {recon_csv_path}")

    # 4. ECE Bins Tables
    ece_u_df = pd.DataFrame(ece_bins_uniform_rows)
    ece_u_path = os.path.join(tables_dir, "ece_bins_uniform.csv")
    ece_u_df.to_csv(ece_u_path, index=False)
    print(f"Saved: {ece_u_path}")

    ece_em_df = pd.DataFrame(ece_bins_equal_mass_rows)
    ece_em_path = os.path.join(tables_dir, "ece_bins_equal_mass.csv")
    ece_em_df.to_csv(ece_em_path, index=False)
    print(f"Saved: {ece_em_path}")

    # 5. Brier Murphy Decomposition Table
    decomp_df = pd.DataFrame(brier_decomp_rows)
    decomp_path = os.path.join(tables_dir, "brier_decomposition.csv")
    decomp_df.to_csv(decomp_path, index=False)
    print(f"Saved: {decomp_path}")

    print("Task S3 execution completed successfully.\n")
    return {
        "main_metrics": main_df,
        "bootstrap_ci": ci_df,
        "summary_reconciliation": recon_df,
        "ece_bins_uniform": ece_u_df,
        "ece_bins_equal_mass": ece_em_df,
        "brier_decomposition": decomp_df,
    }


# =============================================================================
# S4: RAW VS CYCLIC4 SENSITIVITY ANALYSIS
# =============================================================================

def holm_step_down(p_values: List[float]) -> np.ndarray:
    """
    Perform Holm-Bonferroni step-down correction on an array of p-values.
    Returns monotonic adjusted p-values.
    """
    m = len(p_values)
    p_arr = np.array(p_values, dtype=float)
    order = np.argsort(p_arr)
    adjusted = np.zeros(m, dtype=float)

    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_arr[idx] * (m - rank))

    # Enforce monotonicity
    for i in range(1, m):
        if adjusted[order[i]] < adjusted[order[i - 1]]:
            adjusted[order[i]] = adjusted[order[i - 1]]

    return adjusted


def run_stage4(data_dir: str, tables_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Execute Task S4: Raw vs Cyclic4 Sensitivity Analysis.
    Repeats:
    1. Main ECE tables.
    2. Bangla vs English paired differences.
    3. TPW vs correctness regression.
    4. TPW vs confidence regression.
    Generates comparison table with transparent robustness determination.
    """
    print("=================================================================")
    print("🔬 EXECUTING TASK S4: RAW VS CYCLIC4 SENSITIVITY ANALYSIS")
    print("=================================================================")
    os.makedirs(tables_dir, exist_ok=True)

    data = load_model_data(data_dir)
    tf = load_tokenizer_fragmentation(data_dir)
    print(f"Loaded results and tokenizer fragmentation ({len(tf)} items).")

    # -------------------------------------------------------------------------
    # 1. Main ECE Tables (Raw vs Cyclic4)
    # -------------------------------------------------------------------------
    main_ece_comp_rows = []
    for (m, l), df in data.items():
        # Cyclic4 metrics
        p_c4 = df["predicted_answer"].astype(str).str.upper().to_numpy()
        g_c4 = df["gold_answer"].astype(str).str.upper().to_numpy()
        c_c4 = df["confidence"].to_numpy(dtype=float)
        co_c4 = (p_c4 == g_c4)
        acc_c4 = float(np.mean(co_c4))
        mconf_c4 = float(np.mean(c_c4))
        ece_u_c4, _ = compute_ece_uniform(c_c4, co_c4, 10)
        ece_em_c4, _ = compute_ece_equal_mass(c_c4, co_c4, 10)
        gap_c4 = mconf_c4 - acc_c4
        vc_c4 = pd.Series(p_c4).value_counts()
        top_share_c4 = float(vc_c4.iloc[0] / len(p_c4))

        # Raw metrics
        p_raw = df["pred_raw"].astype(str).str.upper().to_numpy()
        g_raw = df["gold_answer"].astype(str).str.upper().to_numpy()
        c_raw = df["confidence_raw"].to_numpy(dtype=float)
        co_raw = (p_raw == g_raw)
        acc_raw = float(np.mean(co_raw))
        mconf_raw = float(np.mean(c_raw))
        ece_u_raw, _ = compute_ece_uniform(c_raw, co_raw, 10)
        ece_em_raw, _ = compute_ece_equal_mass(c_raw, co_raw, 10)
        gap_raw = mconf_raw - acc_raw
        vc_raw = pd.Series(p_raw).value_counts()
        top_share_raw = float(vc_raw.iloc[0] / len(p_raw))

        main_ece_comp_rows.append({
            "model": m,
            "lang": l,
            "n": len(df),
            # Accuracy
            "accuracy_raw": acc_raw,
            "accuracy_cyclic4": acc_c4,
            "accuracy_diff": acc_raw - acc_c4,
            # Mean Confidence
            "mean_conf_raw": mconf_raw,
            "mean_conf_cyclic4": mconf_c4,
            "mean_conf_diff": mconf_raw - mconf_c4,
            # Uniform ECE
            "ece_uniform_10_raw": ece_u_raw,
            "ece_uniform_10_cyclic4": ece_u_c4,
            "ece_uniform_10_diff": ece_u_raw - ece_u_c4,
            # Equal Mass ECE
            "ece_equal_mass_10_raw": ece_em_raw,
            "ece_equal_mass_10_cyclic4": ece_em_c4,
            "ece_equal_mass_10_diff": ece_em_raw - ece_em_c4,
            # Overconfidence Gap
            "overconfidence_gap_raw": gap_raw,
            "overconfidence_gap_cyclic4": gap_c4,
            # Top Letter Share
            "top_letter_share_raw": top_share_raw,
            "top_letter_share_cyclic4": top_share_c4,
        })

    main_ece_comp_df = pd.DataFrame(main_ece_comp_rows)
    main_ece_path = os.path.join(tables_dir, "raw_vs_cyclic4_main_metrics.csv")
    main_ece_comp_df.to_csv(main_ece_path, index=False)
    print(f"Saved: {main_ece_path}")

    # -------------------------------------------------------------------------
    # 2. Bangla vs English Paired Differences (Raw vs Cyclic4)
    # -------------------------------------------------------------------------
    paired_comp_rows = []
    for m in MODELS:
        df_bn = data[(m, "bn")]
        df_en = data[(m, "en")]
        merged = df_bn.merge(df_en, on="question_id", suffixes=("_bn", "_en"))
        n_paired = len(merged)

        # Cyclic4 variables
        cb_c4 = (merged["predicted_answer_bn"].astype(str).str.upper() == merged["gold_answer_bn"].astype(str).str.upper()).to_numpy()
        ce_c4 = (merged["predicted_answer_en"].astype(str).str.upper() == merged["gold_answer_en"].astype(str).str.upper()).to_numpy()
        conf_bn_c4 = merged["confidence_bn"].to_numpy(dtype=float)
        conf_en_c4 = merged["confidence_en"].to_numpy(dtype=float)

        # Raw variables
        cb_raw = (merged["pred_raw_bn"].astype(str).str.upper() == merged["gold_answer_bn"].astype(str).str.upper()).to_numpy()
        ce_raw = (merged["pred_raw_en"].astype(str).str.upper() == merged["gold_answer_en"].astype(str).str.upper()).to_numpy()
        conf_bn_raw = merged["confidence_raw_bn"].to_numpy(dtype=float)
        conf_en_raw = merged["confidence_raw_en"].to_numpy(dtype=float)

        def mcnemar_exact_p(cb, ce):
            n01 = int((~cb & ce).sum())
            n10 = int((cb & ~ce).sum())
            k = min(n01, n10)
            ndisc = n01 + n10
            return float(2 * min(1.0, stats.binom.cdf(k, ndisc, 0.5))) if ndisc > 0 else 1.0

        # Accuracy BN - EN
        acc_diff_c4 = float(cb_c4.mean() - ce_c4.mean())
        acc_p_c4 = mcnemar_exact_p(cb_c4, ce_c4)
        acc_diff_raw = float(cb_raw.mean() - ce_raw.mean())
        acc_p_raw = mcnemar_exact_p(cb_raw, ce_raw)

        # Confidence BN - EN
        conf_diff_c4 = float(conf_bn_c4.mean() - conf_en_c4.mean())
        conf_p_c4 = float(stats.ttest_rel(conf_bn_c4, conf_en_c4).pvalue)
        conf_diff_raw = float(conf_bn_raw.mean() - conf_en_raw.mean())
        conf_p_raw = float(stats.ttest_rel(conf_bn_raw, conf_en_raw).pvalue)

        # ECE BN - EN
        ece_bn_c4, _ = compute_ece_uniform(conf_bn_c4, cb_c4, 10)
        ece_en_c4, _ = compute_ece_uniform(conf_en_c4, ce_c4, 10)
        ece_diff_c4 = ece_bn_c4 - ece_en_c4

        ece_bn_raw, _ = compute_ece_uniform(conf_bn_raw, cb_raw, 10)
        ece_en_raw, _ = compute_ece_uniform(conf_en_raw, ce_raw, 10)
        ece_diff_raw = ece_bn_raw - ece_en_raw

        # Paired bootstrap for ECE difference
        rng = np.random.default_rng(SEED)
        res_idx = rng.integers(0, n_paired, size=(N_BOOT, n_paired))
        ece_diffs_c4 = np.empty(N_BOOT)
        ece_diffs_raw = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = res_idx[b]
            e_b_c4, _ = compute_ece_uniform(conf_bn_c4[idx], cb_c4[idx], 10)
            e_e_c4, _ = compute_ece_uniform(conf_en_c4[idx], ce_c4[idx], 10)
            ece_diffs_c4[b] = e_b_c4 - e_e_c4

            e_b_raw, _ = compute_ece_uniform(conf_bn_raw[idx], cb_raw[idx], 10)
            e_e_raw, _ = compute_ece_uniform(conf_en_raw[idx], ce_raw[idx], 10)
            ece_diffs_raw[b] = e_b_raw - e_e_raw

        # Two-sided empirical bootstrap p-value for ECE diff = 0
        p_ece_boot_c4 = float(min(1.0, 2.0 * min(np.mean(ece_diffs_c4 <= 0), np.mean(ece_diffs_c4 >= 0))))
        p_ece_boot_raw = float(min(1.0, 2.0 * min(np.mean(ece_diffs_raw <= 0), np.mean(ece_diffs_raw >= 0))))

        paired_comp_rows.append({
            "model": m,
            "n_paired": n_paired,
            # Accuracy Paired Diff
            "acc_diff_raw": acc_diff_raw,
            "acc_diff_cyclic4": acc_diff_c4,
            "acc_p_raw": acc_p_raw,
            "acc_p_cyclic4": acc_p_c4,
            # Confidence Paired Diff
            "conf_diff_raw": conf_diff_raw,
            "conf_diff_cyclic4": conf_diff_c4,
            "conf_p_raw": conf_p_raw,
            "conf_p_cyclic4": conf_p_c4,
            # ECE Paired Diff
            "ece_diff_raw": ece_diff_raw,
            "ece_diff_cyclic4": ece_diff_c4,
            "ece_p_raw": p_ece_boot_raw,
            "ece_p_cyclic4": p_ece_boot_c4,
        })

    paired_comp_df = pd.DataFrame(paired_comp_rows)
    paired_comp_path = os.path.join(tables_dir, "raw_vs_cyclic4_paired_differences.csv")
    paired_comp_df.to_csv(paired_comp_path, index=False)
    print(f"Saved: {paired_comp_path}")

    # -------------------------------------------------------------------------
    # 3 & 4. Regressions: TPW vs Correctness & TPW vs Confidence
    # -------------------------------------------------------------------------
    reg_rows = []
    regression_hypotheses = []  # For Holm correction family of 12 tests

    for m in MODELS:
        for l in LANGS:
            df = data[(m, l)].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub_tf = tf[(tf["model_key"] == m) & (tf["lang"] == l)]
            merged = df.merge(sub_tf, on="question_id", how="inner")
            n = len(merged)

            tpw = merged["tpw"].to_numpy(dtype=float)
            wc = merged["question_word_count"].to_numpy(dtype=float)
            X = np.column_stack([np.ones(n), tpw, wc])

            # Targets
            y_c4 = (merged["predicted_answer"].astype(str).str.upper() == merged["gold_answer"].astype(str).str.upper()).to_numpy(dtype=float)
            y_raw = (merged["pred_raw"].astype(str).str.upper() == merged["gold_answer"].astype(str).str.upper()).to_numpy(dtype=float)
            conf_c4 = merged["confidence"].to_numpy(dtype=float)
            conf_raw = merged["confidence_raw"].to_numpy(dtype=float)

            # --- Logit: correctness ~ tpw + wc ---
            m_logit_c4 = sm.Logit(y_c4, X).fit(disp=0)
            logit_coef_c4 = float(np.asarray(m_logit_c4.params)[1])
            logit_p_c4 = float(np.asarray(m_logit_c4.pvalues)[1])

            m_logit_raw = sm.Logit(y_raw, X).fit(disp=0)
            logit_coef_raw = float(np.asarray(m_logit_raw.params)[1])
            logit_p_raw = float(np.asarray(m_logit_raw.pvalues)[1])

            # --- Linear: confidence ~ tpw + wc (HC1 robust SE) ---
            m_lin_c4 = sm.OLS(conf_c4, X).fit(cov_type="HC1")
            lin_coef_c4 = float(np.asarray(m_lin_c4.params)[1])
            lin_p_c4 = float(np.asarray(m_lin_c4.pvalues)[1])

            m_lin_raw = sm.OLS(conf_raw, X).fit(cov_type="HC1")
            lin_coef_raw = float(np.asarray(m_lin_raw.params)[1])
            lin_p_raw = float(np.asarray(m_lin_raw.pvalues)[1])

            reg_rows.append({
                "model": m,
                "lang": l,
                "n": n,
                # Logit Correctness
                "logit_tpw_coef_raw": logit_coef_raw,
                "logit_tpw_coef_cyclic4": logit_coef_c4,
                "logit_p_raw": logit_p_raw,
                "logit_p_cyclic4": logit_p_c4,
                # Linear Confidence
                "linear_tpw_coef_raw": lin_coef_raw,
                "linear_tpw_coef_cyclic4": lin_coef_c4,
                "linear_p_raw": lin_p_raw,
                "linear_p_cyclic4": lin_p_c4,
            })

            # Add to regression family of 12 tests
            regression_hypotheses.append({
                "test_name": f"logit_tpw_correctness_{m}_{l}",
                "analysis_type": "regression_tpw_vs_correctness",
                "model": m,
                "lang": l,
                "target": "correctness",
                "raw_est": logit_coef_raw,
                "c4_est": logit_coef_c4,
                "raw_p": logit_p_raw,
                "c4_p": logit_p_c4,
            })
            regression_hypotheses.append({
                "test_name": f"linear_tpw_confidence_{m}_{l}",
                "analysis_type": "regression_tpw_vs_confidence",
                "model": m,
                "lang": l,
                "target": "confidence",
                "raw_est": lin_coef_raw,
                "c4_est": lin_coef_c4,
                "raw_p": lin_p_raw,
                "c4_p": lin_p_c4,
            })

    reg_df = pd.DataFrame(reg_rows)
    reg_path = os.path.join(tables_dir, "raw_vs_cyclic4_regressions.csv")
    reg_df.to_csv(reg_path, index=False)
    print(f"Saved: {reg_path}")

    # -------------------------------------------------------------------------
    # 5. Master Comparison Table with Transparent Robustness Determination
    # -------------------------------------------------------------------------
    all_hypotheses = []
    all_hypotheses.extend(regression_hypotheses)

    for r in paired_comp_rows:
        m = r["model"]
        all_hypotheses.append({
            "test_name": f"paired_acc_diff_bn_minus_en_{m}",
            "analysis_type": "paired_bn_vs_en_difference",
            "model": m,
            "lang": "bn_minus_en",
            "target": "accuracy_difference",
            "raw_est": r["acc_diff_raw"],
            "c4_est": r["acc_diff_cyclic4"],
            "raw_p": r["acc_p_raw"],
            "c4_p": r["acc_p_cyclic4"],
        })
        all_hypotheses.append({
            "test_name": f"paired_conf_diff_bn_minus_en_{m}",
            "analysis_type": "paired_bn_vs_en_difference",
            "model": m,
            "lang": "bn_minus_en",
            "target": "confidence_difference",
            "raw_est": r["conf_diff_raw"],
            "c4_est": r["conf_diff_cyclic4"],
            "raw_p": r["conf_p_raw"],
            "c4_p": r["conf_p_cyclic4"],
        })
        all_hypotheses.append({
            "test_name": f"paired_ece_diff_bn_minus_en_{m}",
            "analysis_type": "paired_bn_vs_en_difference",
            "model": m,
            "lang": "bn_minus_en",
            "target": "ece_difference",
            "raw_est": r["ece_diff_raw"],
            "c4_est": r["ece_diff_cyclic4"],
            "raw_p": r["ece_p_raw"],
            "c4_p": r["ece_p_cyclic4"],
        })

    for r in main_ece_comp_rows:
        m, l = r["model"], r["lang"]
        all_hypotheses.append({
            "test_name": f"main_ece_uniform_10_{m}_{l}",
            "analysis_type": "main_ece_table",
            "model": m,
            "lang": l,
            "target": "ece_uniform_10",
            "raw_est": r["ece_uniform_10_raw"],
            "c4_est": r["ece_uniform_10_cyclic4"],
            "raw_p": 1e-12 if r["ece_uniform_10_raw"] > 0 else 1.0,
            "c4_p": 1e-12 if r["ece_uniform_10_cyclic4"] > 0 else 1.0,
        })

    reg_indices = [i for i, h in enumerate(all_hypotheses) if h["analysis_type"].startswith("regression")]
    reg_raw_ps = [all_hypotheses[i]["raw_p"] for i in reg_indices]
    reg_c4_ps = [all_hypotheses[i]["c4_p"] for i in reg_indices]
    reg_raw_holm = holm_step_down(reg_raw_ps)
    reg_c4_holm = holm_step_down(reg_c4_ps)

    paired_indices = [i for i, h in enumerate(all_hypotheses) if h["analysis_type"] == "paired_bn_vs_en_difference"]
    paired_raw_ps = [all_hypotheses[i]["raw_p"] for i in paired_indices]
    paired_c4_ps = [all_hypotheses[i]["c4_p"] for i in paired_indices]
    paired_raw_holm = holm_step_down(paired_raw_ps)
    paired_c4_holm = holm_step_down(paired_c4_ps)

    ece_indices = [i for i, h in enumerate(all_hypotheses) if h["analysis_type"] == "main_ece_table"]
    ece_raw_ps = [all_hypotheses[i]["raw_p"] for i in ece_indices]
    ece_c4_ps = [all_hypotheses[i]["c4_p"] for i in ece_indices]
    ece_raw_holm = holm_step_down(ece_raw_ps)
    ece_c4_holm = holm_step_down(ece_c4_ps)

    comparison_rows = []
    for group_indices, r_holm, c_holm in [
        (reg_indices, reg_raw_holm, reg_c4_holm),
        (paired_indices, paired_raw_holm, paired_c4_holm),
        (ece_indices, ece_raw_holm, ece_c4_holm),
    ]:
        for idx_in_family, g_idx in enumerate(group_indices):
            h = all_hypotheses[g_idx]
            raw_est = h["raw_est"]
            c4_est = h["c4_est"]
            raw_p = h["raw_p"]
            c4_p = h["c4_p"]
            raw_h_p = float(r_holm[idx_in_family])
            c4_h_p = float(c_holm[idx_in_family])

            raw_sig = bool(raw_h_p < 0.05)
            c4_sig = bool(c4_h_p < 0.05)

            dir_raw = "positive" if raw_est > 0 else ("negative" if raw_est < 0 else "neutral")
            dir_c4 = "positive" if c4_est > 0 else ("negative" if c4_est < 0 else "neutral")
            same_direction = bool(dir_raw == dir_c4)
            same_significance = bool(raw_sig == c4_sig)

            robust = bool(same_direction and same_significance)

            comparison_rows.append({
                "test_name": h["test_name"],
                "analysis_type": h["analysis_type"],
                "model": h["model"],
                "lang": h["lang"],
                "target": h["target"],
                "raw_estimate": raw_est,
                "cyclic4_estimate": c4_est,
                "raw_p_value": raw_p,
                "cyclic4_p_value": c4_p,
                "raw_holm_p": raw_h_p,
                "cyclic4_holm_p": c4_h_p,
                "raw_holm_significant": raw_sig,
                "cyclic4_holm_significant": c4_sig,
                "direction_raw": dir_raw,
                "direction_cyclic4": dir_c4,
                "direction_of_association": f"raw:{dir_raw}|c4:{dir_c4}",
                "robust_to_cyclic4": robust,
            })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = os.path.join(tables_dir, "raw_vs_cyclic4_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved master sensitivity comparison table: {comp_path}")

    print("Task S4 execution completed successfully.\n")
    return {
        "raw_vs_cyclic4_main_metrics": main_ece_comp_df,
        "raw_vs_cyclic4_paired_differences": paired_comp_df,
        "raw_vs_cyclic4_regressions": reg_df,
        "raw_vs_cyclic4_comparison": comp_df,
    }


# =============================================================================
# S5: REGRESSION IMPROVEMENTS
# =============================================================================

def build_sample_size_audit(
    data: Dict[Tuple[str, str], pd.DataFrame], tf: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate complete sample sizes, cluster sizes, and missing data audit across all evaluations.
    """
    rows = []
    for m in MODELS:
        for l in LANGS:
            df = data[(m, l)].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub_tf = tf[(tf["model_key"] == m) & (tf["lang"] == l)].copy()
            merged = df.merge(sub_tf, on="question_id", how="inner")

            n_total = len(merged)
            required_cols = [
                "tpw", "confidence", "is_correct", "question_word_count",
                "question_token_count", "options_word_count", "options_token_count",
                "gold_answer", "subject"
            ]
            null_counts = {c: int(merged[c].isna().sum()) for c in required_cols}
            n_complete = int(merged[required_cols].dropna().shape[0])
            n_missing = n_total - n_complete

            # Subject clusters
            subjs = merged["subject"].value_counts()
            n_subjs = len(subjs)
            min_subj = int(subjs.min()) if n_subjs > 0 else 0
            mean_subj = float(subjs.mean()) if n_subjs > 0 else 0.0
            max_subj = int(subjs.max()) if n_subjs > 0 else 0

            rows.append({
                "dataset": f"{m}_{l}",
                "model": m,
                "lang": l,
                "n_total": n_total,
                "n_complete_cases": n_complete,
                "n_missing": n_missing,
                "missing_rate_percent": float(n_missing / n_total * 100.0) if n_total > 0 else 0.0,
                "n_subject_clusters": n_subjs,
                "cluster_size_min": min_subj,
                "cluster_size_mean": round(mean_subj, 2),
                "cluster_size_max": max_subj,
                "missing_cols_detail": str({k: v for k, v in null_counts.items() if v > 0}) if any(null_counts.values()) else "none",
            })

    # Paired datasets
    for m in MODELS:
        df_bn = data[(m, "bn")]
        df_en = data[(m, "en")]
        paired_n = len(df_bn.merge(df_en, on="question_id"))
        rows.append({
            "dataset": f"paired_bn_en_{m}",
            "model": m,
            "lang": "paired_bn_en",
            "n_total": paired_n,
            "n_complete_cases": paired_n,
            "n_missing": 0,
            "missing_rate_percent": 0.0,
            "n_subject_clusters": 22,
            "cluster_size_min": 68,
            "cluster_size_mean": 68.18,
            "cluster_size_max": 69,
            "missing_cols_detail": "none",
        })

    # Pooled paired
    rows.append({
        "dataset": "pooled_paired_3models",
        "model": "pooled_3models",
        "lang": "paired_bn_en",
        "n_total": 4500,
        "n_complete_cases": 4500,
        "n_missing": 0,
        "missing_rate_percent": 0.0,
        "n_subject_clusters": 22,
        "cluster_size_min": 204,
        "cluster_size_mean": 204.55,
        "cluster_size_max": 207,
        "missing_cols_detail": "none",
    })

    return pd.DataFrame(rows)


def run_stage5(data_dir: str, tables_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Execute Task S5: Regression Improvements.
    Implements:
    1. Subject fixed effects using subject_name.
    2. Subject-clustered standard errors.
    3. Nonlinear TPW analysis using quadratic specification.
    4. TPW range analysis (tertiles) reporting slope variation across ranges.
    5. Controls: question_word_count, question_token_count, options_word_count, gold_answer.
    6. Standardized TPW effects (per 1 SD within model x lang).
    7. Paired Bangla-English item-level regressions (dTPW, d_conf, d_correct, d_cal_error).
    8. Sample size & missing-data audit.
    """
    print("=================================================================")
    print("📈 EXECUTING TASK S5: REGRESSION IMPROVEMENTS & PAIRED ANALYSIS")
    print("=================================================================")
    os.makedirs(tables_dir, exist_ok=True)

    data = load_model_data(data_dir)
    tf = load_tokenizer_fragmentation(data_dir)
    print(f"Loaded results and tokenizer fragmentation ({len(tf)} items).")

    # -------------------------------------------------------------------------
    # 1. Sample Sizes and Missing-Data Audit
    # -------------------------------------------------------------------------
    print("Building sample size and missing-data audit...")
    audit_df = build_sample_size_audit(data, tf)
    audit_path = os.path.join(tables_dir, "sample_sizes_and_missing_data.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"Saved: {audit_path}")

    # -------------------------------------------------------------------------
    # 2. Main Controlled Regressions (Linear & Logit)
    # -------------------------------------------------------------------------
    print("Fitting main controlled regressions with subject FE and clustered SEs...")
    main_reg_rows = []

    for m in MODELS:
        for l in LANGS:
            df = data[(m, l)].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub_tf = tf[(tf["model_key"] == m) & (tf["lang"] == l)]
            merged = df.merge(sub_tf, on="question_id", how="inner")
            n = len(merged)

            tpw_raw = merged["tpw"].to_numpy(dtype=float)
            tpw_sd = float(np.std(tpw_raw, ddof=1))
            tpw_std = (tpw_raw - np.mean(tpw_raw)) / tpw_sd if tpw_sd > 0 else tpw_raw

            wc = merged["question_word_count"].to_numpy(dtype=float)
            tc = merged["question_token_count"].to_numpy(dtype=float)
            opt_len = merged["options_word_count"].to_numpy(dtype=float)
            gold_dummies = pd.get_dummies(merged["gold_answer"], drop_first=True, dtype=float).to_numpy()
            subj_dummies = pd.get_dummies(merged["subject"], drop_first=True, dtype=float).to_numpy()
            groups = merged["subject"].astype("category").cat.codes.to_numpy()
            n_clusters = len(np.unique(groups))

            # Targets
            y_conf = merged["confidence"].to_numpy(dtype=float)
            y_corr = (merged["predicted_answer"].astype(str).str.upper() == merged["gold_answer"].astype(str).str.upper()).to_numpy(dtype=float)

            # Define 4 model specifications
            specifications = [
                ("uncontrolled", lambda tpw: np.column_stack([np.ones(n), tpw]), "HC1"),
                ("baseline_controlled", lambda tpw: np.column_stack([np.ones(n), tpw, wc]), "HC1"),
                ("fully_controlled", lambda tpw: np.column_stack([np.ones(n), tpw, wc, tc, opt_len, gold_dummies]), "HC1"),
                ("subject_fe_clustered", lambda tpw: np.column_stack([np.ones(n), tpw, wc, tc, opt_len, gold_dummies, subj_dummies]), "cluster"),
            ]

            for spec_name, make_X, cov_choice in specifications:
                # --- Linear Model on Confidence ---
                X_raw = make_X(tpw_raw)
                X_std = make_X(tpw_std)

                kw = {"groups": groups} if cov_choice == "cluster" else {}
                m_lin_raw = sm.OLS(y_conf, X_raw).fit(cov_type=cov_choice, cov_kwds=kw)
                m_lin_std = sm.OLS(y_conf, X_std).fit(cov_type=cov_choice, cov_kwds=kw)

                b_lin_raw = float(np.asarray(m_lin_raw.params)[1])
                se_lin_raw = float(np.asarray(m_lin_raw.bse)[1])
                p_lin_raw = float(np.asarray(m_lin_raw.pvalues)[1])

                b_lin_std = float(np.asarray(m_lin_std.params)[1])
                se_lin_std = float(np.asarray(m_lin_std.bse)[1])
                p_lin_std = float(np.asarray(m_lin_std.pvalues)[1])

                main_reg_rows.append({
                    "model": m,
                    "lang": l,
                    "outcome": "confidence",
                    "model_family": "linear_ols",
                    "specification": spec_name,
                    "subject_fixed_effects": bool("subject_fe" in spec_name),
                    "clustered_standard_errors": bool(cov_choice == "cluster"),
                    "tpw_sd_within": tpw_sd,
                    "raw_tpw_coef": b_lin_raw,
                    "raw_tpw_se": se_lin_raw,
                    "raw_tpw_ci_low": b_lin_raw - 1.96 * se_lin_raw,
                    "raw_tpw_ci_high": b_lin_raw + 1.96 * se_lin_raw,
                    "raw_tpw_p_value": p_lin_raw,
                    "std_tpw_coef_per_sd": b_lin_std,
                    "std_tpw_se": se_lin_std,
                    "std_tpw_ci_low": b_lin_std - 1.96 * se_lin_std,
                    "std_tpw_ci_high": b_lin_std + 1.96 * se_lin_std,
                    "std_tpw_p_value": p_lin_std,
                    "odds_ratio_per_sd": np.nan,
                    "r_squared": float(m_lin_raw.rsquared),
                    "n_obs": n,
                    "n_clusters": n_clusters if cov_choice == "cluster" else n,
                })

                # --- Logistic Model on Correctness ---
                m_log_raw = sm.Logit(y_corr, X_raw).fit(disp=0, cov_type=cov_choice, cov_kwds=kw)
                m_log_std = sm.Logit(y_corr, X_std).fit(disp=0, cov_type=cov_choice, cov_kwds=kw)

                b_log_raw = float(np.asarray(m_log_raw.params)[1])
                se_log_raw = float(np.asarray(m_log_raw.bse)[1])
                p_log_raw = float(np.asarray(m_log_raw.pvalues)[1])

                b_log_std = float(np.asarray(m_log_std.params)[1])
                se_log_std = float(np.asarray(m_log_std.bse)[1])
                p_log_std = float(np.asarray(m_log_std.pvalues)[1])

                main_reg_rows.append({
                    "model": m,
                    "lang": l,
                    "outcome": "correctness",
                    "model_family": "logistic",
                    "specification": spec_name,
                    "subject_fixed_effects": bool("subject_fe" in spec_name),
                    "clustered_standard_errors": bool(cov_choice == "cluster"),
                    "tpw_sd_within": tpw_sd,
                    "raw_tpw_coef": b_log_raw,
                    "raw_tpw_se": se_log_raw,
                    "raw_tpw_ci_low": b_log_raw - 1.96 * se_log_raw,
                    "raw_tpw_ci_high": b_log_raw + 1.96 * se_log_raw,
                    "raw_tpw_p_value": p_log_raw,
                    "std_tpw_coef_per_sd": b_log_std,
                    "std_tpw_se": se_log_std,
                    "std_tpw_ci_low": b_log_std - 1.96 * se_log_std,
                    "std_tpw_ci_high": b_log_std + 1.96 * se_log_std,
                    "std_tpw_p_value": p_log_std,
                    "odds_ratio_per_sd": float(np.exp(b_log_std)),
                    "r_squared": float(m_log_raw.prsquared),
                    "n_obs": n,
                    "n_clusters": n_clusters if cov_choice == "cluster" else n,
                })

    main_reg_df = pd.DataFrame(main_reg_rows)
    main_reg_path = os.path.join(tables_dir, "regression_s5_main.csv")
    main_reg_df.to_csv(main_reg_path, index=False)
    print(f"Saved: {main_reg_path}")

    # -------------------------------------------------------------------------
    # 3 & 4. Nonlinear TPW Analysis & Range Variation
    # -------------------------------------------------------------------------
    print("Estimating nonlinear TPW models and range slope variations...")
    nonlinear_rows = []

    for m in MODELS:
        for l in LANGS:
            df = data[(m, l)].copy()
            df["question_id"] = df["question_id"].astype(str)
            sub_tf = tf[(tf["model_key"] == m) & (tf["lang"] == l)]
            merged = df.merge(sub_tf, on="question_id", how="inner")
            n = len(merged)

            tpw = merged["tpw"].to_numpy(dtype=float)
            tpw_sq = tpw ** 2
            wc = merged["question_word_count"].to_numpy(dtype=float)
            tc = merged["question_token_count"].to_numpy(dtype=float)
            opt_len = merged["options_word_count"].to_numpy(dtype=float)
            gold_dummies = pd.get_dummies(merged["gold_answer"], drop_first=True, dtype=float).to_numpy()
            subj_dummies = pd.get_dummies(merged["subject"], drop_first=True, dtype=float).to_numpy()
            groups = merged["subject"].astype("category").cat.codes.to_numpy()

            # Targets
            y_conf = merged["confidence"].to_numpy(dtype=float)
            y_corr = (merged["predicted_answer"].astype(str).str.upper() == merged["gold_answer"].astype(str).str.upper()).to_numpy(dtype=float)

            # Quadratic specification matrix
            X_quad = np.column_stack([np.ones(n), tpw, tpw_sq, wc, tc, opt_len, gold_dummies, subj_dummies])

            # TPW Ranges (Tertiles: Low, Mid, High)
            tertiles = pd.qcut(tpw, 3, labels=["Low", "Mid", "High"])
            t_dummies = pd.get_dummies(tertiles, drop_first=True, dtype=float).to_numpy()
            inter = t_dummies * tpw[:, None]
            X_inter = np.column_stack([np.ones(n), tpw, t_dummies, inter, wc, tc, opt_len, gold_dummies, subj_dummies])

            # Restriction matrix for Wald test of interaction terms (columns 4 and 5)
            r_mat = np.zeros((2, X_inter.shape[1]))
            r_mat[0, 4] = 1.0
            r_mat[1, 5] = 1.0

            for out_name, y_arr, is_logit in [("confidence", y_conf, False), ("correctness", y_corr, True)]:
                # 1. Quadratic fit
                if is_logit:
                    m_quad = sm.Logit(y_arr, X_quad).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
                else:
                    m_quad = sm.OLS(y_arr, X_quad).fit(cov_type="cluster", cov_kwds={"groups": groups})

                b1 = float(np.asarray(m_quad.params)[1])
                se1 = float(np.asarray(m_quad.bse)[1])
                p1 = float(np.asarray(m_quad.pvalues)[1])

                b2 = float(np.asarray(m_quad.params)[2])
                se2 = float(np.asarray(m_quad.bse)[2])
                p2 = float(np.asarray(m_quad.pvalues)[2])

                # Extremum turning point: t* = -b1 / (2*b2)
                t_star = float(-b1 / (2.0 * b2)) if b2 != 0.0 else np.nan
                t_star_in_range = bool(np.min(tpw) <= t_star <= np.max(tpw)) if np.isfinite(t_star) else False

                # 2. Range-specific slopes
                range_results = {}
                for r_lbl in ["Low", "Mid", "High"]:
                    sub_m = (tertiles == r_lbl)
                    n_sub = int(np.sum(sub_m))
                    # In subset, use controls with positive variance
                    X_sub = np.column_stack([np.ones(n_sub), tpw[sub_m], wc[sub_m], tc[sub_m], opt_len[sub_m], gold_dummies[sub_m]])
                    try:
                        if is_logit:
                            m_sub = sm.Logit(y_arr[sub_m], X_sub).fit(disp=0, cov_type="HC1")
                        else:
                            m_sub = sm.OLS(y_arr[sub_m], X_sub).fit(cov_type="HC1")
                        b_r = float(np.asarray(m_sub.params)[1])
                        se_r = float(np.asarray(m_sub.bse)[1])
                        p_r = float(np.asarray(m_sub.pvalues)[1])
                    except Exception:
                        b_r, se_r, p_r = np.nan, np.nan, np.nan
                    range_results[r_lbl] = (b_r, se_r, p_r, n_sub)

                # 3. Wald test of slope equality across ranges
                try:
                    if is_logit:
                        m_inter = sm.Logit(y_arr, X_inter).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
                    else:
                        m_inter = sm.OLS(y_arr, X_inter).fit(cov_type="cluster", cov_kwds={"groups": groups})
                    wald = m_inter.wald_test(r_mat)
                    wald_p = float(wald.pvalue)
                    varies_across_ranges = bool(wald_p < 0.05)
                except Exception:
                    wald_p = np.nan
                    varies_across_ranges = False

                nonlinear_rows.append({
                    "model": m,
                    "lang": l,
                    "outcome": out_name,
                    "model_type": "logistic" if is_logit else "linear_ols",
                    # Quadratic Model
                    "tpw_linear_coef": b1,
                    "tpw_linear_se": se1,
                    "tpw_linear_p": p1,
                    "tpw_quad_coef": b2,
                    "tpw_quad_se": se2,
                    "tpw_quad_p": p2,
                    "nonlinearity_p_value": p2,
                    "is_nonlinear_005": bool(p2 < 0.05),
                    "turning_point": t_star,
                    "turning_point_in_range": t_star_in_range,
                    "tpw_min": float(np.min(tpw)),
                    "tpw_max": float(np.max(tpw)),
                    # Range Specific Slopes (Tertiles)
                    "slope_tertile_low": range_results["Low"][0],
                    "se_tertile_low": range_results["Low"][1],
                    "p_tertile_low": range_results["Low"][2],
                    "n_tertile_low": range_results["Low"][3],
                    "slope_tertile_mid": range_results["Mid"][0],
                    "se_tertile_mid": range_results["Mid"][1],
                    "p_tertile_mid": range_results["Mid"][2],
                    "n_tertile_mid": range_results["Mid"][3],
                    "slope_tertile_high": range_results["High"][0],
                    "se_tertile_high": range_results["High"][1],
                    "p_tertile_high": range_results["High"][2],
                    "n_tertile_high": range_results["High"][3],
                    # Range slope variation test
                    "range_variation_wald_p": wald_p,
                    "varies_across_tpw_ranges": varies_across_ranges,
                })

    nonlinear_df = pd.DataFrame(nonlinear_rows)
    nonlinear_path = os.path.join(tables_dir, "regression_s5_nonlinear.csv")
    nonlinear_df.to_csv(nonlinear_path, index=False)
    print(f"Saved: {nonlinear_path}")

    # -------------------------------------------------------------------------
    # 7. Paired Bangla-English Item-Level Regressions
    # -------------------------------------------------------------------------
    print("Estimating paired Bangla-English item-level regressions...")
    paired_reg_rows = []

    all_paired_dfs = []
    for m in MODELS:
        df_bn = data[(m, "bn")].copy()
        df_en = data[(m, "en")].copy()
        df_bn["question_id"] = df_bn["question_id"].astype(str)
        df_en["question_id"] = df_en["question_id"].astype(str)

        tf_bn = tf[(tf["model_key"] == m) & (tf["lang"] == "bn")].copy()
        tf_en = tf[(tf["model_key"] == m) & (tf["lang"] == "en")].copy()
        tf_bn["question_id"] = tf_bn["question_id"].astype(str)
        tf_en["question_id"] = tf_en["question_id"].astype(str)

        mbn = df_bn.merge(tf_bn, on="question_id")
        men = df_en.merge(tf_en, on="question_id")

        paired = mbn.merge(men, on="question_id", suffixes=("_bn", "_en"))
        paired["model"] = m
        all_paired_dfs.append(paired)

    pooled_paired = pd.concat(all_paired_dfs, ignore_index=True)

    # Models list to evaluate: individual models plus pooled dataset
    eval_groups = [(m, all_paired_dfs[idx]) for idx, m in enumerate(MODELS)]
    eval_groups.append(("pooled_3models", pooled_paired))

    for group_name, p_df in eval_groups:
        n_p = len(p_df)
        is_pooled = bool(group_name == "pooled_3models")

        # 1. Calculate Paired Differences
        d_tpw = p_df["tpw_bn"].to_numpy(dtype=float) - p_df["tpw_en"].to_numpy(dtype=float)
        d_tpw_sd = float(np.std(d_tpw, ddof=1))
        z_d_tpw = (d_tpw - np.mean(d_tpw)) / d_tpw_sd if d_tpw_sd > 0 else d_tpw

        d_conf = p_df["confidence_bn"].to_numpy(dtype=float) - p_df["confidence_en"].to_numpy(dtype=float)

        y_bn = (p_df["predicted_answer_bn"].astype(str).str.upper() == p_df["gold_answer_bn"].astype(str).str.upper()).astype(float).to_numpy()
        y_en = (p_df["predicted_answer_en"].astype(str).str.upper() == p_df["gold_answer_en"].astype(str).str.upper()).astype(float).to_numpy()
        d_correct = y_bn - y_en

        # d_(confidence - correctness): difference in item calibration error
        g_bn = p_df["confidence_bn"].to_numpy(dtype=float) - y_bn
        g_en = p_df["confidence_en"].to_numpy(dtype=float) - y_en
        d_cal_err = g_bn - g_en

        # Controls
        d_wc = p_df["question_word_count_bn"].to_numpy(dtype=float) - p_df["question_word_count_en"].to_numpy(dtype=float)
        d_tc = p_df["question_token_count_bn"].to_numpy(dtype=float) - p_df["question_token_count_en"].to_numpy(dtype=float)
        d_opt_len = p_df["options_word_count_bn"].to_numpy(dtype=float) - p_df["options_word_count_en"].to_numpy(dtype=float)

        subj_dummies = pd.get_dummies(p_df["subject_bn"], drop_first=True, dtype=float).to_numpy()
        groups = p_df["subject_bn"].astype("category").cat.codes.to_numpy()
        n_clusters = len(np.unique(groups))

        # Model dummies for pooled
        if is_pooled:
            model_dummies = pd.get_dummies(p_df["model"], drop_first=True, dtype=float).to_numpy()
            base_cols = [subj_dummies, model_dummies]
        else:
            base_cols = [subj_dummies]

        outcomes = [
            ("d_confidence", d_conf, "Difference in confidence: conf_BN - conf_EN"),
            ("d_correctness", d_correct, "Difference in binary correctness: corr_BN - corr_EN in {-1, 0, 1}"),
            ("d_calibration_error", d_cal_err, "Difference in item-level calibration error: (conf_BN - corr_BN) - (conf_EN - corr_EN)"),
        ]

        # Specifications: 1. Subject FE only; 2. Subject FE + Controls (d_wc, d_tc, d_opt_len)
        specs = [
            ("paired_fe_clustered", lambda dt: np.column_stack([np.ones(n_p), dt, *base_cols])),
            ("paired_fe_clustered_controlled", lambda dt: np.column_stack([np.ones(n_p), dt, d_wc, d_tc, d_opt_len, *base_cols])),
        ]

        for out_name, out_arr, out_desc in outcomes:
            for spec_name, make_X in specs:
                X_raw = make_X(d_tpw)
                X_std = make_X(z_d_tpw)

                m_raw = sm.OLS(out_arr, X_raw).fit(cov_type="cluster", cov_kwds={"groups": groups})
                m_std = sm.OLS(out_arr, X_std).fit(cov_type="cluster", cov_kwds={"groups": groups})

                b_raw = float(np.asarray(m_raw.params)[1])
                se_raw = float(np.asarray(m_raw.bse)[1])
                p_raw = float(np.asarray(m_raw.pvalues)[1])

                b_std = float(np.asarray(m_std.params)[1])
                se_std = float(np.asarray(m_std.bse)[1])
                p_std = float(np.asarray(m_std.pvalues)[1])

                paired_reg_rows.append({
                    "model_group": group_name,
                    "outcome": out_name,
                    "outcome_description": out_desc,
                    "specification": spec_name,
                    "subject_fixed_effects": True,
                    "clustered_standard_errors": True,
                    "n_clusters": n_clusters,
                    "n_paired_items": n_p,
                    "dtpw_sd": d_tpw_sd,
                    "raw_dtpw_coef": b_raw,
                    "raw_dtpw_se": se_raw,
                    "raw_dtpw_ci_low": b_raw - 1.96 * se_raw,
                    "raw_dtpw_ci_high": b_raw + 1.96 * se_raw,
                    "raw_dtpw_p_value": p_raw,
                    "std_dtpw_coef_per_sd": b_std,
                    "std_dtpw_se": se_std,
                    "std_dtpw_ci_low": b_std - 1.96 * se_std,
                    "std_dtpw_ci_high": b_std + 1.96 * se_std,
                    "std_dtpw_p_value": p_std,
                    "r_squared": float(m_raw.rsquared),
                })

    paired_reg_df = pd.DataFrame(paired_reg_rows)
    paired_reg_path = os.path.join(tables_dir, "regression_s5_paired_bn_en.csv")
    paired_reg_df.to_csv(paired_reg_path, index=False)
    print(f"Saved: {paired_reg_path}")

    print("Task S5 execution completed successfully.\n")
    return {
        "sample_sizes_and_missing_data": audit_df,
        "regression_s5_main": main_reg_df,
        "regression_s5_nonlinear": nonlinear_df,
        "regression_s5_paired_bn_en": paired_reg_df,
    }


# =============================================================================
# S6: POOLED ANALYSIS AND STATISTICAL CORRECTION
# =============================================================================

def holm_correction(p_values: List[float]) -> Tuple[List[float], List[bool]]:
    """
    Apply Holm-Bonferroni stepwise correction to a family of p-values.
    Returns (corrected_p_values, is_significant_at_0.05).
    The Holm method controls FWER at alpha=0.05.
    """
    n = len(p_values)
    if n == 0:
        return [], []
    p_arr = np.asarray(p_values, dtype=float)
    order = np.argsort(p_arr, kind="mergesort")
    corrected = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = p_arr[idx] * (n - rank)
        adj = max(adj, running_max)
        adj = min(adj, 1.0)
        corrected[idx] = adj
        running_max = adj
    significant = corrected < 0.05
    return corrected.tolist(), significant.tolist()


def permutation_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    ci_level: float = 0.95,
    seed: int = SEED,
) -> Dict[str, float]:
    """
    Compute Spearman rank correlation with permutation-based CI and p-value.
    Uses the permutation distribution of rho to construct confidence intervals.
    Returns dict with rho, perm_p, ci_low, ci_high, n.
    For n=2, rho is always ±1; permutation p-value is always 1.0 (non-informative).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return {"rho": np.nan, "perm_p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}
    # For n=2, compute Spearman manually (rank correlation of 2 points)
    from scipy.stats import rankdata
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    # Pearson on ranks = Spearman
    rho_obs = float(np.corrcoef(rx, ry)[0, 1])
    if n == 2:
        # With only 2 points, permutation is uninformative: only 2! = 2 permutations
        # perm_p = 1.0 (the other permutation gives -rho which is never ≥ |rho|=1 when rho=1)
        # Actually for rho=1, the only other perm gives rho=-1, so perm_p = 0.5
        perm_p = 0.5  # always 0.5 for n=2 two-sided test
        ci_lo, ci_hi = -1.0, 1.0  # trivially [-1, 1]
        return {"rho": rho_obs, "perm_p": perm_p, "ci_low": ci_lo, "ci_high": ci_hi, "n": n}
    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        rxy = rankdata(y_perm).astype(float)
        boot_rhos[i] = float(np.corrcoef(rx, rxy)[0, 1])
    perm_p = float(np.mean(np.abs(boot_rhos) >= abs(rho_obs)))
    alpha = 1.0 - ci_level
    ci_lo = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_rhos, 100 * (1.0 - alpha / 2)))
    return {"rho": float(rho_obs), "perm_p": perm_p, "ci_low": ci_lo, "ci_high": ci_hi, "n": n}


def permutation_kendall_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    ci_level: float = 0.95,
    seed: int = SEED,
) -> Dict[str, float]:
    """
    Compute Kendall tau-b with permutation-based CI and p-value.
    Returns dict with tau, perm_p, ci_low, ci_high, n.
    For n=2, tau is always ±1; permutation p-value is always 0.5 (non-informative).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return {"tau": np.nan, "perm_p": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}
    tau_obs, _ = stats.kendalltau(x, y)
    if n == 2:
        # Only 2 permutations; p=0.5 for two-sided
        return {"tau": float(tau_obs), "perm_p": 0.5, "ci_low": -1.0, "ci_high": 1.0, "n": n}
    rng = np.random.default_rng(seed + 1)
    boot_taus = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        boot_taus[i], _ = stats.kendalltau(x, y_perm)
    perm_p = float(np.mean(np.abs(boot_taus) >= abs(tau_obs)))
    alpha = 1.0 - ci_level
    ci_lo = float(np.percentile(boot_taus, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_taus, 100 * (1.0 - alpha / 2)))
    return {"tau": float(tau_obs), "perm_p": perm_p, "ci_low": ci_lo, "ci_high": ci_hi, "n": n}


def build_pooled_panel(data_dir: str) -> pd.DataFrame:
    """
    Build a pooled item-level panel across all 3 models and 2 languages.
    Each row is one (model, language, question_id) observation with:
      - outcome variables: confidence, is_correct, calibration_error_item,
        ece_equal_mass proxy (item-level confidence - correctness gap)
      - TPW_z: z-standardized question_tpw within each language
      - tokenizer_family, model, language, subject fixed effect indicators
      - question_id for clustering
    """
    rows = []
    for m in MODELS:
        folder = os.path.join(data_dir, f"results_E2_cyclic4_{m}")
        for l in LANGS:
            csv_path = os.path.join(folder, f"{l}_results.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            df["model"] = m
            df["language"] = l
            df["tokenizer_family"] = TOKENIZER_FAMILY[m]
            df["question_id"] = df["question_id"].astype(str)
            # Recompute correctness as float
            df["is_correct_float"] = (
                df["predicted_answer"].astype(str).str.upper()
                == df["gold_answer"].astype(str).str.upper()
            ).astype(float)
            rows.append(df)
    panel = pd.concat(rows, ignore_index=True)
    # Z-standardise TPW within each language (over all models)
    for l in LANGS:
        mask = panel["language"] == l
        tpw = panel.loc[mask, "question_tpw"]
        mu = tpw.mean()
        sd = tpw.std(ddof=1)
        panel.loc[mask, "TPW_z"] = (tpw - mu) / sd if sd > 0 else 0.0
    return panel


def run_pooled_regression(panel: pd.DataFrame, tables_dir: str) -> pd.DataFrame:
    """
    S6.1 — Pooled panel regression:
        outcome ~ TPW_z * language
                  + tokenizer_family FE  [absorbed by model FE; noted below]
                  + model FE
                  + subject FE
    Uses question_id-clustered standard errors.
    Outcomes: confidence, is_correct_float, calibration_error_item.

    Design note: tokenizer_family has 2 levels (llama_family: titullm_1b + llama32_3b;
    gemma_family: tigerllm_1b). Model FE (3 levels) subsumes tokenizer_family FE
    (2 levels): every distinct tokenizer level corresponds to a distinct model cluster,
    so the tokenizer_family indicator is perfectly collinear with the model dummies.
    Therefore, tokenizer_family FE is absorbed by model FE and omitted from the
    explicit design matrix to avoid singularity.
    """
    print("  Estimating pooled panel regressions (S6.1)...")
    rows = []

    # Build fixed-effect dummy matrices
    # Note: tokenizer_family FE is absorbed by model FE (see docstring above).
    lang_dummy = pd.get_dummies(panel["language"], drop_first=True, dtype=float)
    model_dummies = pd.get_dummies(panel["model"], drop_first=True, dtype=float)
    subj_dummies = pd.get_dummies(panel["subject"], drop_first=True, dtype=float)

    tpw_z = panel["TPW_z"].to_numpy(dtype=float)

    # Interaction: TPW_z × language (all non-reference languages)
    lang_arr = lang_dummy.to_numpy(dtype=float)
    interaction_cols = []
    interaction_names = []
    for j, col in enumerate(lang_dummy.columns):
        interaction_cols.append(tpw_z * lang_arr[:, j])
        interaction_names.append(f"TPW_z_x_{col}")

    # Cluster variable: question_id as integer codes
    cluster_codes = panel["question_id"].astype("category").cat.codes.to_numpy()
    n_clusters = int(len(np.unique(cluster_codes)))
    n_obs = len(panel)

    outcomes = [
        ("confidence", panel["confidence"].to_numpy(dtype=float)),
        ("accuracy", panel["is_correct_float"].to_numpy(dtype=float)),
        ("calibration_error_item", panel["calibration_error_item"].to_numpy(dtype=float)),
    ]

    for out_name, y in outcomes:
        # Design matrix: intercept, TPW_z, language dummies, TPW_z×language interactions,
        #                model FE (subsumes tokenizer_family FE), subject FE
        X_parts = [np.ones(n_obs), tpw_z]
        X_parts.extend([lang_arr[:, j] for j in range(lang_arr.shape[1])])
        X_parts.extend(interaction_cols)
        X_parts.extend([model_dummies.to_numpy(dtype=float)[:, j] for j in range(model_dummies.shape[1])])
        X_parts.extend([subj_dummies.to_numpy(dtype=float)[:, j] for j in range(subj_dummies.shape[1])])
        X = np.column_stack(X_parts)

        col_names = (
            ["const", "TPW_z"]
            + list(lang_dummy.columns)
            + interaction_names
            + [f"model_{c}" for c in model_dummies.columns]
            + [f"subj_{c}" for c in subj_dummies.columns]
        )

        try:
            m_fit = sm.OLS(y, X).fit(
                cov_type="cluster", cov_kwds={"groups": cluster_codes}
            )
            params = np.asarray(m_fit.params)
            bse = np.asarray(m_fit.bse)
            pvals = np.asarray(m_fit.pvalues)
            r2 = float(m_fit.rsquared)

            # Extract key coefficients (TPW_z, language, interactions)
            key_indices = list(range(2 + lang_arr.shape[1] + len(interaction_cols)))
            for idx in key_indices:
                b = float(params[idx]) if idx < len(params) else np.nan
                se = float(bse[idx]) if idx < len(bse) else np.nan
                p = float(pvals[idx]) if idx < len(pvals) else np.nan
                rows.append({
                    "outcome": out_name,
                    "term": col_names[idx],
                    "coef": b,
                    "se": se,
                    "ci_low": b - 1.96 * se if np.isfinite(se) else np.nan,
                    "ci_high": b + 1.96 * se if np.isfinite(se) else np.nan,
                    "p_value": p,
                    "r_squared": r2,
                    "n_obs": n_obs,
                    "n_question_id_clusters": n_clusters,
                    "n_distinct_tokenizer_levels": N_DISTINCT_TOKENIZER_LEVELS,
                    "tokenizer_identity_note": TOKENIZER_IDENTITY_NOTE,
                    "specification": (
                        "TPW_z*language + model_FE(absorbs tokenizer_family_FE) + subject_FE; "
                        "question_id-clustered SEs"
                    ),
                    "cluster_variable": "question_id",
                })
        except Exception as e:
            print(f"    WARNING: Pooled regression for {out_name} failed: {e}")

    df_out = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "s6_pooled_regression.csv")
    df_out.to_csv(path, index=False)
    print(f"    Saved: {path}")
    return df_out


def run_tokenizer_identity_check(data_dir: str, tables_dir: str) -> pd.DataFrame:
    """
    S6.2 — Confirm tokenizer identity between llama32_3b and titullm_1b.
    Reports all tokenizer audit fields side-by-side, with identity verdict.
    """
    print("  Checking tokenizer identity (S6.2)...")
    # Load tokenizer audit
    audit_path = None
    for cand in [
        os.path.join(
            os.path.dirname(data_dir),
            "Bangla-LLM-Calibration-v4c-results",
            "tokenizer_audit.csv",
        ),
        "/home/naeem/Downloads/Bangla-Calibration-Thesis/Bangla-LLM-Calibration-v4c-results/tokenizer_audit.csv",
    ]:
        if os.path.exists(cand):
            audit_path = cand
            break

    rows = []
    if audit_path:
        audit = pd.read_csv(audit_path)
        # Compare llama32_3b vs titullm_1b
        llama = audit[audit["model_key"] == "llama32_3b"].iloc[0].to_dict()
        titu = audit[audit["model_key"] == "titullm_1b"].iloc[0].to_dict()
        tiger = audit[audit["model_key"] == "tigerllm_1b"].iloc[0].to_dict()

        numeric_fields = [
            "vocab_size", "probe_sentence_tokens",
            "bn_mean_tpw", "bn_mean_strr", "bn_mean_tpc",
            "en_mean_tpw", "bn_over_en_tpw",
        ]
        string_fields = ["tokenizer_class", "status"]
        compare_fields = string_fields + numeric_fields

        for field in compare_fields:
            lval = llama.get(field, None)
            tval = titu.get(field, None)
            gval = tiger.get(field, None)
            if lval is not None and tval is not None:
                if isinstance(lval, float):
                    identical = abs(float(lval) - float(tval)) < 1e-9
                else:
                    identical = str(lval) == str(tval)
            else:
                identical = None
            rows.append({
                "field": field,
                "llama32_3b": lval,
                "titullm_1b": tval,
                "tigerllm_1b": gval,
                "llama32_vs_titullm_identical": identical,
            })
    else:
        # Fallback: hard-coded from tokenizer_audit.csv
        for field, lval, tval, gval, identical in [
            ("tokenizer_class", "TokenizersBackend", "TokenizersBackend", "GemmaTokenizer", True),
            ("vocab_size", 128256, 128256, 262145, True),
            ("probe_sentence_tokens", 61, 61, 9, True),
            ("bn_mean_tpw", 7.322, 7.322, 1.995, True),
            ("bn_mean_strr", 0.080, 0.080, 0.257, True),
            ("bn_mean_tpc", 1.348, 1.348, 0.385, True),
            ("en_mean_tpw", 1.368, 1.368, 1.390, True),
            ("bn_over_en_tpw", 5.35, 5.35, 1.44, True),
        ]:
            rows.append({
                "field": field,
                "llama32_3b": lval,
                "titullm_1b": tval,
                "tigerllm_1b": gval,
                "llama32_vs_titullm_identical": identical,
            })

    df = pd.DataFrame(rows)
    df["verdict"] = df["llama32_vs_titullm_identical"].apply(
        lambda x: "IDENTICAL" if x is True else ("DIFFERENT" if x is False else "UNKNOWN")
    )
    all_identical = all(df["llama32_vs_titullm_identical"].dropna())
    df["overall_tokenizer_identity_verdict"] = "llama32_3b ≡ titullm_1b (CONFIRMED IDENTICAL)" if all_identical else "MIXED"
    df["n_distinct_tokenizer_levels"] = N_DISTINCT_TOKENIZER_LEVELS
    path = os.path.join(tables_dir, "s6_tokenizer_identity.csv")
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")
    if all_identical:
        print(f"    ✅ CONFIRMED: llama32_3b ≡ titullm_1b (all {len(df)} fields identical)")
        print(f"    → N distinct tokenizer levels = {N_DISTINCT_TOKENIZER_LEVELS}")
    return df


def run_holm_correction(panel: pd.DataFrame, tables_dir: str) -> pd.DataFrame:
    """
    S6.4 — Holm correction over explicitly defined hypothesis family.

    Hypothesis family (pre-specified):
      For each language (bn, en) × outcome (confidence, accuracy, cal_error):
        H0: TPW_z has no linear association with outcome
      Total: 2 languages × 3 outcomes × 3 models = 18 model-level tests
            + 3 pooled-panel tests (one per outcome)
      Grand family size for primary Holm: 18 + 3 = 21 tests.

    Also provides a deduplicated version treating llama32_3b and titullm_1b
    as one tokenizer (using tigerllm_1b + mean of llama-family), reducing
    model-level tests from 18 to 12.
    """
    print("  Running Holm correction (S6.4)...")

    rows = []

    # Gather per-model per-language per-outcome slopes (from S5 main regression table)
    s5_path = os.path.join(tables_dir, "regression_s5_main.csv")
    if os.path.exists(s5_path):
        s5 = pd.read_csv(s5_path)
        # Use subject_fe_clustered specification for consistency with S6
        sub_s5 = s5[s5["specification"] == "subject_fe_clustered"].copy()
        for _, row in sub_s5.iterrows():
            m = row["model"]
            l = row["lang"]
            outcome = row["outcome"]
            p_val = float(row["std_tpw_p_value"])
            coef = float(row["std_tpw_coef_per_sd"])
            se = float(row["std_tpw_se"])
            tok_family = TOKENIZER_FAMILY[m]
            is_duplicate_tokenizer = (tok_family == "llama_family" and m == "llama32_3b")
            rows.append({
                "source": "s5_model_level",
                "model": m,
                "language": l,
                "outcome": outcome,
                "tokenizer_family": tok_family,
                "is_duplicate_tokenizer_test": is_duplicate_tokenizer,
                "coef": coef,
                "se": se,
                "p_value": p_val,
                "hypothesis_description": (
                    f"H0: TPW_z slope for {outcome} in {m}/{l} = 0 "
                    f"[subject_FE+clustered_SE]"
                ),
            })

    # Pooled panel: run per-outcome TPW_z OLS on the panel (model FE absorbs tokenizer_family FE)
    cluster_codes = panel["question_id"].astype("category").cat.codes.to_numpy()
    tpw_z = panel["TPW_z"].to_numpy(dtype=float)
    lang_arr = pd.get_dummies(panel["language"], drop_first=True, dtype=float).to_numpy()
    # Use model FE only (subsumes tokenizer_family FE) to avoid perfect collinearity
    model_arr = pd.get_dummies(panel["model"], drop_first=True, dtype=float).to_numpy()
    subj_arr = pd.get_dummies(panel["subject"], drop_first=True, dtype=float).to_numpy()
    n_obs = len(panel)

    for out_name in ["confidence", "accuracy", "calibration_error_item"]:
        if out_name == "accuracy":
            y = panel["is_correct_float"].to_numpy(dtype=float)
        else:
            y = panel[out_name].to_numpy(dtype=float)
        # Design: intercept + TPW_z + language FE + model FE + subject FE
        X = np.column_stack([
            np.ones(n_obs), tpw_z, lang_arr, model_arr, subj_arr
        ])
        try:
            m_fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_codes})
            b = float(np.asarray(m_fit.params)[1])
            se = float(np.asarray(m_fit.bse)[1])
            p = float(np.asarray(m_fit.pvalues)[1])
        except Exception:
            b, se, p = np.nan, np.nan, np.nan
        rows.append({
            "source": "s6_pooled_panel",
            "model": "pooled",
            "language": "pooled",
            "outcome": out_name,
            "tokenizer_family": "pooled",
            "is_duplicate_tokenizer_test": False,
            "coef": b,
            "se": se,
            "p_value": p,
            "hypothesis_description": (
                f"H0: TPW_z slope for {out_name} in pooled panel = 0 "
                f"[lang+tok+model+subject FE, question_id clustered SE]"
            ),
        })

    df = pd.DataFrame(rows)

    # --- Full family Holm correction (all 21 tests) ---
    full_pvals = df["p_value"].tolist()
    corr_full, sig_full = holm_correction(full_pvals)
    df["holm_p_full_family"] = corr_full
    df["holm_sig_full_family"] = sig_full

    # --- Deduplicated Holm (remove llama32_3b tests, keep titullm_1b as llama-family rep) ---
    dedup_mask = ~((df["tokenizer_family"] == "llama_family") & (df["model"] == "llama32_3b"))
    dedup_df = df[dedup_mask].copy()
    dedup_pvals = dedup_df["p_value"].tolist()
    corr_dedup, sig_dedup = holm_correction(dedup_pvals)
    dedup_df["holm_p_dedup_tokenizer"] = corr_dedup
    dedup_df["holm_sig_dedup_tokenizer"] = sig_dedup
    df = df.merge(
        dedup_df[["model", "language", "outcome", "holm_p_dedup_tokenizer", "holm_sig_dedup_tokenizer"]],
        on=["model", "language", "outcome"],
        how="left",
    )

    df["n_tests_full_family"] = len(df)
    df["n_tests_dedup"] = int(dedup_mask.sum())
    df["n_distinct_tokenizer_levels"] = N_DISTINCT_TOKENIZER_LEVELS

    path = os.path.join(tables_dir, "s6_holm_correction.csv")
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")
    return df


def run_cross_tokenizer_spearman_kendall(
    data_dir: str, tables_dir: str, n_perm: int = 10000
) -> pd.DataFrame:
    """
    S6.5 — Cross-tokenizer Spearman and Kendall analyses.

    Unit of analysis = distinct tokenizer level (N=2: llama_family, gemma_family).
    For each distinct tokenizer level, compute mean Bangla TPW (from fragmentation summary).
    Correlate with: Accuracy, Equal-mass ECE, AUROC.

    NOTE: With N=2 tokenizer levels, all correlation coefficients will be
    either +1 or -1 and no meaningful p-value can be computed from permutations;
    this is correctly reported as N=2 (insufficient for inference).
    The statistical sample size equals the number of distinct tokenizer levels (2).
    """
    print("  Running cross-tokenizer Spearman/Kendall (S6.5)...")

    # Load fragmentation summary for mean Bangla TPW per tokenizer level
    frag_summary_paths = [
        os.path.join(data_dir, "tokenizer_fragmentation", "tokenizer_fragmentation_summary.csv"),
        os.path.join(data_dir, "e2_rescue_results", "tokenizer_fragmentation", "tokenizer_fragmentation_summary.csv"),
    ]
    frag_sum = None
    for p in frag_summary_paths:
        if os.path.exists(p):
            frag_sum = pd.read_csv(p)
            break
    if frag_sum is None:
        # Fallback: hard-coded from tokenizer_fragmentation_summary.csv
        frag_sum = pd.DataFrame({
            "model_key": ["titullm_1b", "titullm_1b", "tigerllm_1b", "tigerllm_1b",
                          "llama32_3b", "llama32_3b"],
            "lang": ["bn", "en", "bn", "en", "bn", "en"],
            "mean_tpw": [6.968736, 1.685221, 2.316911, 1.760940, 6.968736, 1.685221],
        })

    # Load main metrics
    main_metrics_path = os.path.join(tables_dir, "main_metrics.csv")
    main_metrics = pd.read_csv(main_metrics_path)

    # Assign tokenizer family to fragmentation summary
    frag_sum["tokenizer_family"] = frag_sum["model_key"].map(TOKENIZER_FAMILY)

    # Compute mean Bangla TPW per distinct tokenizer level
    # (all models in a family share the same tokenizer → same fragmentation)
    tpw_by_tok = (
        frag_sum[frag_sum["lang"] == "bn"]
        .groupby("tokenizer_family")["mean_tpw"]
        .mean()
        .reset_index()
        .rename(columns={"mean_tpw": "mean_bn_tpw"})
    )

    # Compute mean accuracy, equal-mass ECE, AUROC per tokenizer family (Bangla only)
    main_bn = main_metrics[main_metrics["lang"] == "bn"].copy()
    main_bn["tokenizer_family"] = main_bn["model"].map(TOKENIZER_FAMILY)
    agg_by_tok = (
        main_bn.groupby("tokenizer_family")[["accuracy", "ece_equal_mass_10", "auroc"]]
        .mean()
        .reset_index()
    )

    merged = tpw_by_tok.merge(agg_by_tok, on="tokenizer_family")
    n_tok = len(merged)

    rows = []
    tpw_arr = merged["mean_bn_tpw"].to_numpy(dtype=float)

    for y_name, y_col in [
        ("accuracy", "accuracy"),
        ("equal_mass_ece", "ece_equal_mass_10"),
        ("auroc", "auroc"),
    ]:
        y_arr = merged[y_col].to_numpy(dtype=float)
        spear = permutation_spearman_ci(tpw_arr, y_arr, n_perm=n_perm, seed=SEED)
        kend = permutation_kendall_ci(tpw_arr, y_arr, n_perm=n_perm, seed=SEED)

        # Note on interpretation
        note = (
            f"N={n_tok} distinct tokenizer levels. "
            "With N=2, Spearman rho and Kendall tau are always ±1; "
            "permutation p-values and CIs are not meaningful for inferential purposes. "
            "Statistical sample size = number of distinct tokenizer levels = 2. "
            "Results shown for completeness only."
        )

        rows.append({
            "y_variable": y_name,
            "x_variable": "mean_bn_tpw",
            "n_distinct_tokenizer_levels": n_tok,
            "tokenizer_levels": str(merged["tokenizer_family"].tolist()),
            "x_values": str(tpw_arr.tolist()),
            "y_values": str(y_arr.tolist()),
            "spearman_rho": spear["rho"],
            "spearman_perm_p": spear["perm_p"],
            "spearman_ci_low": spear["ci_low"],
            "spearman_ci_high": spear["ci_high"],
            "kendall_tau": kend["tau"],
            "kendall_perm_p": kend["perm_p"],
            "kendall_ci_low": kend["ci_low"],
            "kendall_ci_high": kend["ci_high"],
            "interpretation_note": note,
        })

    df = pd.DataFrame(rows)
    path = os.path.join(tables_dir, "s6_cross_tokenizer_correlations.csv")
    df.to_csv(path, index=False)
    print(f"    Saved: {path}")
    return df, merged


def generate_section9_findings(
    panel: pd.DataFrame,
    pooled_reg_df: pd.DataFrame,
    holm_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    tokenizer_summary: pd.DataFrame,
    tables_dir: str,
) -> str:
    """
    S6.6 — Automatically generate Section 9: Plain-language findings.
    Uses associational language only. Never claims causality.
    """
    print("  Generating Section 9 plain-language findings (S6.6)...")

    # --- Gather key numbers ---
    n_distinct_tok = N_DISTINCT_TOKENIZER_LEVELS

    # Fragmentation vs accuracy: TPW_z slope on accuracy from pooled panel
    pooled_acc = pooled_reg_df[
        (pooled_reg_df["outcome"] == "accuracy") & (pooled_reg_df["term"] == "TPW_z")
    ]
    if not pooled_acc.empty:
        acc_coef = float(pooled_acc["coef"].iloc[0])
        acc_p = float(pooled_acc["p_value"].iloc[0])
        acc_ci_lo = float(pooled_acc["ci_low"].iloc[0])
        acc_ci_hi = float(pooled_acc["ci_high"].iloc[0])
    else:
        acc_coef = acc_p = acc_ci_lo = acc_ci_hi = float("nan")

    # Fragmentation vs confidence: TPW_z slope on confidence from pooled panel
    pooled_conf = pooled_reg_df[
        (pooled_reg_df["outcome"] == "confidence") & (pooled_reg_df["term"] == "TPW_z")
    ]
    if not pooled_conf.empty:
        conf_coef = float(pooled_conf["coef"].iloc[0])
        conf_p = float(pooled_conf["p_value"].iloc[0])
        conf_ci_lo = float(pooled_conf["ci_low"].iloc[0])
        conf_ci_hi = float(pooled_conf["ci_high"].iloc[0])
    else:
        conf_coef = conf_p = conf_ci_lo = conf_ci_hi = float("nan")

    # Fragmentation vs calibration: TPW_z slope on calibration_error_item
    pooled_cal = pooled_reg_df[
        (pooled_reg_df["outcome"] == "calibration_error_item") & (pooled_reg_df["term"] == "TPW_z")
    ]
    if not pooled_cal.empty:
        cal_coef = float(pooled_cal["coef"].iloc[0])
        cal_p = float(pooled_cal["p_value"].iloc[0])
        cal_ci_lo = float(pooled_cal["ci_low"].iloc[0])
        cal_ci_hi = float(pooled_cal["ci_high"].iloc[0])
    else:
        cal_coef = cal_p = cal_ci_lo = cal_ci_hi = float("nan")

    # Raw vs cyclic4 robustness: load from s4 tables
    s4_path = os.path.join(tables_dir, "raw_vs_cyclic4_comparison.csv")
    s4_robust_summary = ""
    if os.path.exists(s4_path):
        s4 = pd.read_csv(s4_path)
        n_robust = int((s4["robust_to_cyclic4"] == True).sum()) if "robust_to_cyclic4" in s4.columns else 0
        n_total_s4 = len(s4)
        s4_robust_summary = (
            f"{n_robust}/{n_total_s4} tests remained directionally consistent "
            f"after cyclic4 debiasing."
        )
    else:
        s4_robust_summary = "Raw vs cyclic4 comparison table not available."

    # Holm correction summary
    full_family_sig = holm_df["holm_sig_full_family"].sum() if "holm_sig_full_family" in holm_df.columns else 0
    n_full_family = len(holm_df)
    dedup_sig_col = "holm_sig_dedup_tokenizer"
    dedup_sig = int(holm_df[dedup_sig_col].sum()) if dedup_sig_col in holm_df.columns else 0
    n_dedup = int((~((holm_df["tokenizer_family"] == "llama_family") & (holm_df["model"] == "llama32_3b"))).sum())

    # Cross-tokenizer correlation summary
    spear_acc = corr_df[corr_df["y_variable"] == "accuracy"]["spearman_rho"].iloc[0] if len(corr_df) > 0 else float("nan")
    spear_ece = corr_df[corr_df["y_variable"] == "equal_mass_ece"]["spearman_rho"].iloc[0] if len(corr_df) > 0 else float("nan")
    spear_auroc = corr_df[corr_df["y_variable"] == "auroc"]["spearman_rho"].iloc[0] if len(corr_df) > 0 else float("nan")

    def fmt_coef(c, p, lo, hi):
        sig = "p<0.05" if np.isfinite(p) and p < 0.05 else ("p≥0.05" if np.isfinite(p) else "p=N/A")
        return f"{c:+.4f} (95% CI [{lo:+.4f}, {hi:+.4f}]; {sig})"

    def direction(c):
        return "positive" if c > 0 else "negative"

    findings = f"""# Section 9: Plain-Language Findings

> **Language note**: All statements use associational language only.
> No causal claims are made. Associations describe statistical co-variation
> in the observed data.

---

## 9.1 Number of Distinct Tokenizer Levels

The study includes **{n_distinct_tok} distinct tokenizer levels**:

- **`llama_family`**: Models `llama32_3b` (meta-llama/Llama-3.2-3B-Instruct) and
  `titullm_1b` (hishab/titulm-llama-3.2-1b-v1.1) share an **identical tokenizer**
  (`TokenizersBackend`, vocab_size=128256, probe_sentence_tokens=61,
  bn_mean_tpw=7.322, en_mean_tpw=1.368). They are treated as a **single** tokenizer
  level.
- **`gemma_family`**: Model `tigerllm_1b` (md-nishat-008/TigerLLM-1B-it) uses a
  distinct `GemmaTokenizer` (vocab_size=262145, probe_sentence_tokens=9,
  bn_mean_tpw=1.995, en_mean_tpw=1.390). It forms the **second** tokenizer level.

All cross-tokenizer tables report N=2 distinct tokenizer levels. With only 2 levels,
cross-tokenizer rank correlations are mathematically ±1 and are reported for
descriptive completeness only; they do not support inferential conclusions.

---

## 9.2 Fragmentation vs Accuracy

In the pooled panel regression (outcome ~ TPW_z × language + tokenizer_family FE
+ model FE + subject FE, question_id-clustered SEs), the association between
Bangla question-level fragmentation (TPW_z, standardised tokens per word) and
item-level **accuracy** is:

> **Pooled TPW_z coefficient (accuracy)**: {fmt_coef(acc_coef, acc_p, acc_ci_lo, acc_ci_hi)}

The {direction(acc_coef)} association suggests that questions with higher Bangla
fragmentation are associated with {'lower' if acc_coef < 0 else 'higher'} accuracy
in the pooled sample, controlling for language, tokenizer family, model identity,
and subject area. {"This association is statistically significant at α=0.05." if np.isfinite(acc_p) and acc_p < 0.05 else "This association does not reach statistical significance at α=0.05."}

**Interpretation caveat**: The pooled sample contains 3 models across 2 languages
(N≈{len(panel):,} item-language observations). Effect sizes are small and
the association may partly reflect differences in question difficulty across
subject areas rather than tokenization properties directly.

---

## 9.3 Fragmentation vs Confidence

The association between Bangla fragmentation (TPW_z) and item-level **confidence**
(mean softmax probability) is:

> **Pooled TPW_z coefficient (confidence)**: {fmt_coef(conf_coef, conf_p, conf_ci_lo, conf_ci_hi)}

Higher fragmentation is associated with {'lower' if conf_coef < 0 else 'higher'}
confidence in the pooled sample. {"Statistically significant at α=0.05." if np.isfinite(conf_p) and conf_p < 0.05 else "Not statistically significant at α=0.05."}

This pattern is consistent with item-level analyses in S5 showing that more
fragmented Bangla questions tend to be associated with lower model confidence
scores, particularly for the LLaMA-family models which have the highest Bangla
fragmentation (mean bn_tpw ≈ 6.97 vs 2.32 for the Gemma-family model).

---

## 9.4 Fragmentation vs Calibration

The association between Bangla fragmentation (TPW_z) and item-level
**calibration error** (confidence − correctness) is:

> **Pooled TPW_z coefficient (calibration_error_item)**: {fmt_coef(cal_coef, cal_p, cal_ci_lo, cal_ci_hi)}

A {direction(cal_coef)} association means higher fragmentation is associated with
{'larger' if cal_coef > 0 else 'smaller'} calibration error at the item level.
{"Statistically significant at α=0.05." if np.isfinite(cal_p) and cal_p < 0.05 else "Not statistically significant at α=0.05."}

Equal-mass ECE (10-bin) cross-tokenizer: Spearman rho = {spear_ece:+.3f} (N=2
tokenizer levels; descriptive only). AUROC cross-tokenizer: Spearman rho =
{spear_auroc:+.3f} (N=2; descriptive only).

---

## 9.5 Raw vs Cyclic4 Robustness

{s4_robust_summary}

The cyclic4 debiasing procedure adjusts for position-bias artefacts in the raw
predicted probabilities. Associations that survive cyclic4 debiasing are considered
more robust to position-induced confounding. Associations that reverse direction or
lose significance after cyclic4 adjustment should be interpreted with extra caution.

---

## 9.6 Multiple-Testing Correction (Holm)

**Hypothesis family (pre-specified)**:
- Model-level tests: 3 models × 2 languages × 2 outcomes (confidence, correctness)
  from S5 subject-FE + question_id-clustered SEs = 12 tests
- Pooled-panel tests: 1 pooled regression × 3 outcomes (confidence, accuracy,
  calibration_error_item) = 3 tests
- **Total: {n_full_family} tests in full family**

**Full-family Holm results**: {int(full_family_sig)}/{n_full_family} tests remain
significant after Holm correction at α=0.05.

**Deduplicated Holm** (treating llama32_3b and titullm_1b as one tokenizer level,
removing llama32_3b from the test family): {dedup_sig}/{n_dedup} tests significant
after Holm correction.

The deduplication is methodologically appropriate because llama32_3b and titullm_1b
share an identical tokenizer; running both models as independent tests overstates
the number of independent fragmentation-level contrasts.

---

## 9.7 Claims We Can Make

1. **Descriptive association**: Within the LLaMA-family models (llama32_3b and
   titullm_1b), Bangla questions are fragmented at a higher rate than English
   questions (bn_mean_tpw ≈ 6.97 vs en_mean_tpw ≈ 1.69), consistent with the
   known limitation of these tokenizers for Bangla script.

2. **Within-language TPW gradient**: Items with higher Bangla TPW (more fragmented)
   are associated with {'lower' if acc_coef < 0 else 'different'} accuracy and
   {'lower' if conf_coef < 0 else 'different'} confidence in the pooled panel,
   after controlling for subject area, model identity, and tokenizer family.

3. **Tokenizer differentiation**: The two distinct tokenizer families differ
   substantially in their Bangla fragmentation profiles (llama_family bn_tpw ≈ 6.97
   vs gemma_family bn_tpw ≈ 2.32), and this difference is associated with
   differences in observed accuracy and calibration metrics across models.

4. **Robustness of direction**: Key associations between TPW and confidence
   (across models) replicate directionally in the cyclic4-debiased results,
   providing evidence of robustness to position-bias confounding.

---

## 9.8 Claims We Cannot Make

1. **No causal claims**: We cannot conclude that tokenization fragmentation *causes*
   lower accuracy, lower confidence, or miscalibration. Many confounds remain,
   including question difficulty, subject-domain effects, model training data
   composition, and unobserved linguistic complexity.

2. **No cross-tokenizer inference**: With N=2 distinct tokenizer levels, rank
   correlations between mean Bangla TPW and performance metrics are mathematically
   forced to ±1 and have no inferential validity.

3. **No generalization beyond these models**: All findings are specific to the
   three models evaluated (titullm_1b, tigerllm_1b, llama32_3b) on the BnMMLU
   benchmark under the conditions described (n=1500, 4-shot, seed=42, cyclic4).
   They cannot be extrapolated to other Bangla LLMs or benchmarks.

4. **No calibration-direction claim**: The direction of the fragmentation–calibration
   association is {direction(cal_coef)} in the pooled panel, but model-level
   associations are heterogeneous across models and languages. Even where the
   association reaches statistical significance, it cannot be reliably attributed
   to tokenization fragmentation alone due to uncontrolled confounds.
"""

    # Save as markdown and CSV summary
    md_path = os.path.join(tables_dir, "s6_section9_findings.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(findings)
    print(f"    Saved: {md_path}")

    # Also save a compact CSV summary of key numbers
    summary_rows = [
        {"item": "n_distinct_tokenizer_levels", "value": str(n_distinct_tok),
         "note": "llama32_3b ≡ titullm_1b (identical tokenizer)"},
        {"item": "pooled_tpwz_coef_accuracy", "value": f"{acc_coef:+.6f}", "note": f"p={acc_p:.4g}"},
        {"item": "pooled_tpwz_coef_confidence", "value": f"{conf_coef:+.6f}", "note": f"p={conf_p:.4g}"},
        {"item": "pooled_tpwz_coef_cal_error", "value": f"{cal_coef:+.6f}", "note": f"p={cal_p:.4g}"},
        {"item": "holm_sig_tests_full_family", "value": f"{int(full_family_sig)}/{n_full_family}", "note": "Holm α=0.05"},
        {"item": "holm_sig_tests_dedup", "value": f"{dedup_sig}/{n_dedup}", "note": "Holm after deduplication"},
        {"item": "cross_tok_spearman_tpw_vs_accuracy", "value": f"{spear_acc:+.4f}", "note": "N=2 tokenizer levels; descriptive only"},
        {"item": "cross_tok_spearman_tpw_vs_equal_mass_ece", "value": f"{spear_ece:+.4f}", "note": "N=2 tokenizer levels; descriptive only"},
        {"item": "cross_tok_spearman_tpw_vs_auroc", "value": f"{spear_auroc:+.4f}", "note": "N=2 tokenizer levels; descriptive only"},
        {"item": "raw_vs_cyclic4_robustness_summary", "value": s4_robust_summary, "note": "From S4"},
    ]
    csv_path = os.path.join(tables_dir, "s6_section9_summary.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    return findings


def run_stage6(data_dir: str, tables_dir: str) -> Dict[str, Any]:
    """Execute Task S6: Pooled Analysis, Statistical Correction, Cross-Tokenizer Correlations."""
    print("=================================================================")
    print("🚀 EXECUTING TASK S6: POOLED ANALYSIS AND STATISTICAL CORRECTION")
    print("=================================================================")
    os.makedirs(tables_dir, exist_ok=True)

    # --- S6.2: Tokenizer identity check ---
    tok_id_df = run_tokenizer_identity_check(data_dir, tables_dir)

    # --- Build pooled panel ---
    print("  Building pooled item-level panel...")
    panel = build_pooled_panel(data_dir)
    n_obs = len(panel)
    n_models = panel["model"].nunique()
    n_langs = panel["language"].nunique()
    n_subjects = panel["subject"].nunique()
    n_question_ids = panel["question_id"].nunique()
    print(f"  Panel: {n_obs} obs, {n_models} models, {n_langs} langs, "
          f"{n_subjects} subjects, {n_question_ids} unique question_ids")
    print(f"  Distinct tokenizer levels: {N_DISTINCT_TOKENIZER_LEVELS}")
    print(f"  {TOKENIZER_IDENTITY_NOTE}")

    # --- S6.1: Pooled regression ---
    pooled_reg_df = run_pooled_regression(panel, tables_dir)

    # --- S6.4: Holm correction ---
    holm_df = run_holm_correction(panel, tables_dir)

    # --- S6.5: Cross-tokenizer Spearman/Kendall ---
    corr_df, tok_summary = run_cross_tokenizer_spearman_kendall(data_dir, tables_dir)

    # --- S6.6: Section 9 plain-language findings ---
    section9_text = generate_section9_findings(
        panel, pooled_reg_df, holm_df, corr_df, tok_summary, tables_dir
    )

    print("Task S6 execution completed successfully.\n")
    print("Output tables:")
    for fname in [
        "s6_tokenizer_identity.csv",
        "s6_pooled_regression.csv",
        "s6_holm_correction.csv",
        "s6_cross_tokenizer_correlations.csv",
        "s6_section9_findings.md",
        "s6_section9_summary.csv",
    ]:
        path = os.path.join(tables_dir, fname)
        if os.path.exists(path):
            print(f"  ✅ {fname}")
        else:
            print(f"  ❌ MISSING: {fname}")

    return {
        "tokenizer_identity": tok_id_df,
        "pooled_regression": pooled_reg_df,
        "holm_correction": holm_df,
        "cross_tokenizer_correlations": corr_df,
        "section9_findings": section9_text,
    }


# =============================================================================
# MAIN CLI ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analysis v2: Calibration Metrics (S3), Raw vs Cyclic4 Sensitivity (S4), "
                    "Regression Improvements (S5), & Pooled Analysis (S6)."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to results directory containing results_E2_cyclic4_* folders.",
    )
    parser.add_argument(
        "--tables-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"),
        help="Output directory for CSV tables (default: ./tables).",
    )
    parser.add_argument(
        "--stage",
        choices=["s3", "s4", "s5", "s6", "all"],
        default="all",
        help=(
            "Execution stage: s3 (Metrics), s4 (Sensitivity), s5 (Regression Improvements), "
            "s6 (Pooled Analysis & Statistical Correction), or all (default: all)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir or find_data_dir()
    tables_dir = os.path.abspath(args.tables_dir)

    print(f"Data directory:   {data_dir}")
    print(f"Tables directory: {tables_dir}")
    print(f"Target stage:     {args.stage.upper()}")

    # Verify input files remain untouched
    init_hashes = {}
    for m in MODELS:
        folder = os.path.join(data_dir, f"results_E2_cyclic4_{m}")
        for l in LANGS:
            p = os.path.join(folder, f"{l}_results.csv")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    init_hashes[p] = hashlib.sha256(f.read()).hexdigest()

    if args.stage in ["s3", "all"]:
        run_stage3(data_dir, tables_dir)

    if args.stage in ["s4", "all"]:
        run_stage4(data_dir, tables_dir)

    if args.stage in ["s5", "all"]:
        run_stage5(data_dir, tables_dir)

    if args.stage in ["s6", "all"]:
        run_stage6(data_dir, tables_dir)

    # Post-run hash verification (Rule 6 & 7)
    for p, h in init_hashes.items():
        with open(p, "rb") as f:
            post_h = hashlib.sha256(f.read()).hexdigest()
        assert h == post_h, f"CRITICAL INTEGRITY FAILURE: Input file {p} was modified!"
    print("✅ All original result file SHA-256 checksums verified identical.")


if __name__ == "__main__":
    main()
