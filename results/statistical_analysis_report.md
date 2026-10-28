# Statistical Analysis Report: Tokenization Fragmentation & Model Miscalibration

## Central Research Hypothesis:
> *Higher tokenization fragmentation predicts greater model miscalibration in Bangla LLMs, and this relationship may differ across languages.*

### 1. Sample Characteristics
- **Total items analyzed:** 50
- **Models:** meta-llama/Llama-3.2-3B-Instruct
- **Languages:** bn
- **Mean Accuracy:** 12.00%
- **Mean Confidence (MSP):** 0.5049
- **Mean Item Calibration Residual ($|	ext{conf} - 	ext{acc}|$):** 0.5064

### 2. Correlational Analysis
| Metric | Outcome Variable | Pearson r | p-value | Spearman rho | p-value |
|---|---|---|---|---|---|
| TPW | Calibration Residual | -0.0409 | 7.7820e-01 | 0.0469 | 7.4626e-01 |
| TPW | Accuracy | -0.0925 | 5.2310e-01 | -0.0469 | 7.4627e-01 |
| STRR | Calibration Residual | -0.1258 | 3.8392e-01 | -0.0729 | 6.1471e-01 |
| STRR | Accuracy | 0.0926 | 5.2241e-01 | 0.1111 | 4.4243e-01 |

### 3. Stratified Calibration (Median Split by Fragmentation)
**TPW Median Cutoff:** 2.487

| Stratum | Sample Count | Accuracy (%) | ECE (10 bins) | Brier Score | MCE |
|---|---|---|---|---|---|
| Low Fragmentation  | 25 | 12.00% | 0.3806 | 0.2805 | 0.7747 |
| High Fragmentation | 25 | 12.00% | 0.3892 | 0.2664 | 0.7452 |

### 4. Regression Modeling (Dependent Variable: Calibration Residual)
- **Formula:** `calibration_residual ~ frag_std + is_correct`
- **Standard Errors:** `clustered_by_subject`
- **R²:** 0.0017 (Adjusted R²: -0.0408)

| Variable | Coef (β) | Std Err | p-value | 95% CI Lower | 95% CI Upper |
|---|---|---|---|---|---|
| `Intercept` | 0.5066 | 0.0194 | 1.1179e-149 | 0.4685 | 0.5447 |
| `frag_std` | -0.0054 | 0.0163 | 7.3830e-01 | -0.0373 | 0.0265 |
| `is_correct` | -0.0017 | 0.0464 | 9.7042e-01 | -0.0926 | 0.0892 |

### 5. Multicollinearity Diagnostics (VIF)
| Feature | Variance Inflation Factor (VIF) |
|---|---|
| `tpw` | 1.01 |
| `is_correct` | 1.01 |

### 6. Non-Causal Epistemic Caveat
- All findings reflect observational associations on benchmark distributions. Pretraining tokens, architectural differences, and alignment stages serve as potential confounders.