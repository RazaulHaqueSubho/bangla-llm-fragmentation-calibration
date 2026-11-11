# Statistical Analysis Report: Tokenization Fragmentation & Model Miscalibration

> Observational associations only. No causal claims are made. Pretraining corpus, architecture, and alignment remain potential confounders.

### 1. Sample Characteristics
- **Total items analyzed:** 180
- **Models:** llama32_3b, titullm_1b, tigerllm_1b
- **Languages:** bn, en
- **Mean accuracy:** 21.67%
- **Mean confidence (MSP):** 0.5056

### 2. Cross-Model Descriptive Table (descriptive only, no inference)
| Model | Lang | N | Acc (%) | Mean conf | ECE | MCE | Brier | MC-Brier | NLL | ECE (mass-filt) | Filt-Out (%) | Mean TPW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llama32_3b` | bn | 30 | 20.00 | 0.5160 | 0.3160 | 0.7526 | 0.2639 | 0.9702 | 2.06 | 0.316 | 0.0% | 1.08 |
| `llama32_3b` | en | 30 | 23.33 | 0.4952 | 0.3151 | 0.7413 | 0.2687 | 0.9451 | 1.9578 | 0.3151 | 0.0% | 1.08 |
| `tigerllm_1b` | bn | 30 | 20.00 | 0.5160 | 0.3160 | 0.7526 | 0.2639 | 0.9702 | 2.06 | 0.316 | 0.0% | 1.08 |
| `tigerllm_1b` | en | 30 | 23.33 | 0.4952 | 0.3151 | 0.7413 | 0.2687 | 0.9451 | 1.9578 | 0.3151 | 0.0% | 1.08 |
| `titullm_1b` | bn | 30 | 20.00 | 0.5160 | 0.3160 | 0.7526 | 0.2639 | 0.9702 | 2.06 | 0.316 | 0.0% | 1.08 |
| `titullm_1b` | en | 30 | 23.33 | 0.4952 | 0.3151 | 0.7413 | 0.2687 | 0.9451 | 1.9578 | 0.3151 | 0.0% | 1.08 |

### 3. Bangla minus English ECE (subject-cluster bootstrap)
- `llama32_3b`: mean delta ECE = 0.0023, 95% CI [-0.0951, 0.0517]
- `titullm_1b`: mean delta ECE = 0.0023, 95% CI [-0.0951, 0.0517]
- `tigerllm_1b`: mean delta ECE = 0.0023, 95% CI [-0.0951, 0.0517]

### 4. H2: Fragmentation vs Confidence / Correctness / Overconfidence
> Expected signs: TPW and TPC positive for fragmentation (higher fragmentation predicts miscalibration / overconfidence); STRR negative (higher subword token retention ratio = less fragmentation).

#### `llama32_3b` (bn), N=30
- Bootstrap 95% CI for ECE: [0.1814, 0.4461]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | 0.0068 | [-0.0608, 0.0745] | 8.429e-01 | 1.000e+00 | 0.0278 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0726 | [-0.2391, 0.0938] | 3.924e-01 | 1.000e+00 | 0.0796 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0795 | [-0.0689, 0.2279] | 2.939e-01 | 1.000e+00 | 0.0855 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | 0.0090 | [-0.0589, 0.0769] | 7.947e-01 | 7.947e-01 | 0.0285 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0618 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3695 | 0.2746 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.2942 | 0.2896 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.3541 | 0.2780 | 14.3 |
| Q4_highest | 8 | 1.08 | 0.2483 | 0.2182 | 25.0 |

#### `llama32_3b` (en), N=30
- Bootstrap 95% CI for ECE: [0.1994, 0.4581]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | -0.0125 | [-0.0459, 0.0209] | 4.625e-01 | 1.000e+00 | 0.0209 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0291 | [-0.2716, 0.2135] | 8.144e-01 | 1.000e+00 | 0.0972 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0166 | [-0.2342, 0.2673] | 8.971e-01 | 1.000e+00 | 0.1005 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | -0.0133 | [-0.0474, 0.0207] | 4.426e-01 | 4.426e-01 | 0.0220 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0151 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3894 | 0.2371 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.3002 | 0.2242 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.4104 | 0.2927 | 42.9 |
| Q4_highest | 8 | 1.08 | 0.4554 | 0.3182 | 12.5 |

#### `tigerllm_1b` (bn), N=30
- Bootstrap 95% CI for ECE: [0.1814, 0.4461]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | 0.0068 | [-0.0608, 0.0745] | 8.429e-01 | 1.000e+00 | 0.0278 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0726 | [-0.2391, 0.0938] | 3.924e-01 | 1.000e+00 | 0.0796 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0795 | [-0.0689, 0.2279] | 2.939e-01 | 1.000e+00 | 0.0855 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | 0.0090 | [-0.0589, 0.0769] | 7.947e-01 | 7.947e-01 | 0.0285 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0618 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3695 | 0.2746 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.2942 | 0.2896 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.3541 | 0.2780 | 14.3 |
| Q4_highest | 8 | 1.08 | 0.2483 | 0.2182 | 25.0 |

#### `tigerllm_1b` (en), N=30
- Bootstrap 95% CI for ECE: [0.1994, 0.4581]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | -0.0125 | [-0.0459, 0.0209] | 4.625e-01 | 1.000e+00 | 0.0209 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0291 | [-0.2716, 0.2135] | 8.144e-01 | 1.000e+00 | 0.0972 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0166 | [-0.2342, 0.2673] | 8.971e-01 | 1.000e+00 | 0.1005 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | -0.0133 | [-0.0474, 0.0207] | 4.426e-01 | 4.426e-01 | 0.0220 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0151 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3894 | 0.2371 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.3002 | 0.2242 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.4104 | 0.2927 | 42.9 |
| Q4_highest | 8 | 1.08 | 0.4554 | 0.3182 | 12.5 |

#### `titullm_1b` (bn), N=30
- Bootstrap 95% CI for ECE: [0.1814, 0.4461]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | 0.0068 | [-0.0608, 0.0745] | 8.429e-01 | 1.000e+00 | 0.0278 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0726 | [-0.2391, 0.0938] | 3.924e-01 | 1.000e+00 | 0.0796 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0795 | [-0.0689, 0.2279] | 2.939e-01 | 1.000e+00 | 0.0855 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | 0.0090 | [-0.0589, 0.0769] | 7.947e-01 | 7.947e-01 | 0.0285 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0618 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3695 | 0.2746 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.2942 | 0.2896 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.3541 | 0.2780 | 14.3 |
| Q4_highest | 8 | 1.08 | 0.2483 | 0.2182 | 25.0 |

#### `titullm_1b` (en), N=30
- Bootstrap 95% CI for ECE: [0.1994, 0.4581]

| Metric | Outcome Model | Coef (frag_z) | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpw` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `tpc` | `M1_confidence` | -0.0125 | [-0.0459, 0.0209] | 4.625e-01 | 1.000e+00 | 0.0209 | 30 | 3 | few_clusters |
| `tpc` | `M2_is_correct` | -0.0291 | [-0.2716, 0.2135] | 8.144e-01 | 1.000e+00 | 0.0972 | 30 | 3 | few_clusters |
| `tpc` | `M3_overconfidence` | 0.0166 | [-0.2342, 0.2673] | 8.971e-01 | 1.000e+00 | 0.1005 | 30 | 3 | few_clusters |
| `tpc` | `M4_confidence_cond` | -0.0133 | [-0.0474, 0.0207] | 4.426e-01 | 4.426e-01 | 0.0220 | 30 | 3 | few_clusters |
| `strr_ctx` | `M1_confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M2_is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M3_overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |
| `strr_ctx` | `M4_confidence_cond` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 30 | 3 | few_clusters |

Residual regression: `calibration_residual ~ frag_std + is_correct + total_tokens + C(subject_norm)` (SE: `clustered_by_subject_norm`), R2 = 0.0151 *(secondary / mechanically tied to correctness)*

| TPW quartile | N | Mean TPW | ECE | Brier | Acc (%) |
|---|---|---|---|---|---|
| Q1_lowest | 8 | 1.08 | 0.3894 | 0.2371 | 12.5 |
| Q2_low_mid | 7 | 1.08 | 0.3002 | 0.2242 | 28.6 |
| Q3_high_mid | 7 | 1.08 | 0.4104 | 0.2927 | 42.9 |
| Q4_highest | 8 | 1.08 | 0.4554 | 0.3182 | 12.5 |

### 5. H3: Language x Fragmentation Interaction
> Tests whether the within-language slope per 1 SD of fragmentation differs between Bangla and English. Note: samples are subject-matched, not item-parallel.

#### `llama32_3b` Interaction Models
| Metric | Outcome | Interaction Coef | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpc` | `confidence` | -0.0171 | [-0.0923, 0.0581] | 6.563e-01 | 1.000e+00 | 0.0341 | 60 | 3 | few_clusters |
| `tpc` | `is_correct` | 0.0479 | [-0.1378, 0.2337] | 6.129e-01 | 1.000e+00 | 0.1149 | 60 | 3 | few_clusters |
| `tpc` | `overconfidence` | -0.0650 | [-0.2985, 0.1684] | 5.852e-01 | 1.000e+00 | 0.1209 | 60 | 3 | few_clusters |

#### `tigerllm_1b` Interaction Models
| Metric | Outcome | Interaction Coef | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpc` | `confidence` | -0.0171 | [-0.0923, 0.0581] | 6.563e-01 | 1.000e+00 | 0.0341 | 60 | 3 | few_clusters |
| `tpc` | `is_correct` | 0.0479 | [-0.1378, 0.2337] | 6.129e-01 | 1.000e+00 | 0.1149 | 60 | 3 | few_clusters |
| `tpc` | `overconfidence` | -0.0650 | [-0.2985, 0.1684] | 5.852e-01 | 1.000e+00 | 0.1209 | 60 | 3 | few_clusters |

#### `titullm_1b` Interaction Models
| Metric | Outcome | Interaction Coef | 95% CI | Raw p | Holm p | HC3 SE | N | Clusters | Flags |
|---|---|---|---|---|---|---|---|---|---|
| `tpw` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpw` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `confidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `is_correct` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `strr` | `overconfidence` | 0.0000 | [0.0000, 0.0000] | nan | nan | 0.0000 | 60 | 3 | few_clusters |
| `tpc` | `confidence` | -0.0171 | [-0.0923, 0.0581] | 6.563e-01 | 1.000e+00 | 0.0341 | 60 | 3 | few_clusters |
| `tpc` | `is_correct` | 0.0479 | [-0.1378, 0.2337] | 6.129e-01 | 1.000e+00 | 0.1149 | 60 | 3 | few_clusters |
| `tpc` | `overconfidence` | -0.0650 | [-0.2985, 0.1684] | 5.852e-01 | 1.000e+00 | 0.1209 | 60 | 3 | few_clusters |

### 6. Robustness & Sensitivity Checks
#### Word-Count Length Control vs Prompt-Tokens Control (TPW Models)
| Model x Language | Outcome | Prompt-Tokens Coef | Total-Words Coef | Difference |
|---|---|---|---|---|
| `llama32_3b_bn` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `llama32_3b_bn` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `llama32_3b_bn` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |
| `llama32_3b_en` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `llama32_3b_en` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `llama32_3b_en` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_bn` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_bn` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_bn` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_en` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_en` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `tigerllm_1b_en` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_bn` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_bn` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_bn` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_en` | `M1_confidence` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_en` | `M2_is_correct` | 0.0000 | 0.0000 | 0.0000 |
| `titullm_1b_en` | `M3_overconfidence` | 0.0000 | 0.0000 | 0.0000 |

#### Standard Error Robustness (Cluster SE vs HC3)
- Evaluated side by side in Tables 4 & 5. If subject clusters < 30, HC3 standard errors serve as the primary sensitivity safeguard.

#### Option-Mass Filtering Robustness
- Evaluated in Table 2. Shows calibration error (ECE) when low-mass items (< 0.50 option mass) are removed.

### 7. Reliability Diagrams & Visualizations
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_llama32_3b_bn.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_llama32_3b_bn_raw.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_llama32_3b_en.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_llama32_3b_en_raw.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_tigerllm_1b_bn.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_tigerllm_1b_bn_raw.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_tigerllm_1b_en.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_tigerllm_1b_en_raw.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_titullm_1b_bn.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_titullm_1b_bn_raw.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_titullm_1b_en.png`
- Saved plot: `/home/naeem/Bangla-LLM-Calibration/code/results/reliability_titullm_1b_en_raw.png`

### 8. Caveats & Methodological Limitations
- **Observational associations:** All reported relationships reflect observational associations; no causal claims are warranted.
- **Mechanical dependency of calibration residual:** The item-level residual |confidence - correctness| is mechanically tied to correctness; interpret residual regressions cautiously.
- **Item difficulty and topic confounding:** Within-tokenizer fragmentation may be confounded with item difficulty, syntactic complexity, and subject area.
- **Length control trade-off:** `prompt_tokens` is itself a tokenizer output mechanically correlated with fragmentation, potentially absorbing part of the effect; `total_words` is reported as a sensitivity check.
- **Small cluster count:** In benchmarks with fewer than 30 subject categories, clustered standard errors can be downward biased; HC3 heteroskedasticity-robust SEs serve as the sensitivity check.
- **Subject-matched samples:** Bangla (BnMMLU) and English (MMLU) are subject-matched across broad curricula, not item-parallel translations.
- **Small model cohort:** With 3 evaluated models (2 of which share a tokenizer family), model-level comparisons (H1) are descriptive only.