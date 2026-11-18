# Work allocation and required handoff

## 1. Naeem — measurement and calibration methodology

**Write/prepare:**

1. Exact TPW definition and implementation rules: tokenizer input unit, special tokens, word segmentation, punctuation, whitespace, Unicode normalization, empty strings, and aggregation.
2. STRR: source citation and exact formula, including numerator, denominator, alignment procedure, and tested implementation. If the definition cannot be verified, recommend excluding STRR rather than guessing.
3. Confidence extraction protocol: exact prompt, answer-option scoring, probability normalization, answer prediction, invalid output handling, and model chat template.
4. Calibration metrics: accuracy, multiclass Brier score, ECE. Specify ECE bin count, bin edges, endpoint convention, empty-bin handling, and sample size.
5. Temperature scaling, if used: fitting split, validation procedure, and protection against test leakage.
6. Statistical analysis plan: define the observational unit and primary outcome. Explain how item-level TPW will be related to calibration behaviour while accounting for repeated questions, models, and subjects. State effect sizes, confidence intervals, clustering/mixed effects, multiple-testing correction, and sensitivity checks.
7. Reproducibility: code path, package versions, model/tokenizer revisions, seed, and exact commit.

**Deliverables:**

- Completed methodology subsections in `main.tex` (or a separate Markdown draft for review).
- One metric-definition/rules table.
- One model/tokenizer audit table with exact revisions.
- Calibration/reliability plots and the underlying CSV/JSON used to generate them, if the final protocol supports them.
- A short note identifying unresolved methodological decisions.

**Important:** ECE is an aggregate/group-level metric. Do not treat one global ECE value as a per-question outcome or correlate it naively with thousands of item-level TPW observations. Propose and justify a statistically valid item-level or grouped/hierarchical analysis.

## 2. Abdullah al Azmi — dataset, translation, and paired examples

**Write/prepare:**

1. Dataset source, exact repository revision, split/configuration, license, and retrieval date.
2. Bangla source items and their English translations: pairing IDs, translation method, quality review, option order, answer-label preservation, and adjudication.
3. Dataset filtering: original rows, valid rows, missing values, duplicates, invalid labels, exclusions, and final paired/unpaired counts.
4. Unicode normalization and text-cleaning steps actually applied.
5. Five checked Bangla-English examples for tokenizer demonstration, with stable pair IDs and exact source/translated text. Confirm the examples manually.
6. Clearly separate translated English counterparts of Bangla items from independent English MMLU questions.
7. Provide machine-readable final data and a data dictionary; do not silently edit benchmark answers.

**Deliverables:**

- Completed dataset subsection in `main.tex` (or separate Markdown draft).
- Dataset composition/filtering table.
- Five checked example pairs in CSV/JSON.
- Translation and preprocessing note.
- A data-quality summary with counts and unresolved issues.

## 3. Subho — coordination, literature, research framing, integration

**Write/prepare:**

1. Introduction: motivation, problem statement, research gap, research questions, contributions, paper organization.
2. Related work: tokenizer/fragmentation literature, Bangla LLM and benchmark literature, calibration literature, and synthesis of the gap.
3. Reconcile the final model panel and evaluation protocol across branches with Naeem and Abdullah.
4. Integrate methods, tables, plots, results, discussion, limitations, and conclusion after receiving the evidence.
5. Maintain references.bib and verify every citation.
6. Review all figures, tables, equations, and terminology for consistency.

**Deliverables:**

- Introduction and Related Work draft.
- Research questions and contribution statement.
- Final integrated manuscript and consistency review.
- Final abstract after results are frozen.
