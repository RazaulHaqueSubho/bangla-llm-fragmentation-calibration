# Paper workspace: Springer LNCS

**Working title:** Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? A Cross-Lingual Study

## Current writing team

The active writing team currently consists of:
- **Md. Razaul Haque Subho**
- **Iqramul Hasan Naeem** 
- **Abdullah al Azmi** 

## Files

- `main.tex`: detailed Springer LNCS manuscript structure with section/subsection prompts, equations, and a figure/table plan.
- `references.bib`: shared BibTeX bibliography. Add only verified references.

## Work allocation and required handoff

### 1. Naeem — measurement and calibration methodology

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

### 2. Abdullah — dataset, translation, and paired examples

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

### 3. Subho — coordination, literature, research framing, integration

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

## Paper section map

1. Introduction
   - Background and Motivation
   - Problem Statement
   - Research Gap
   - Research Questions and Contributions
   - Paper Organization
2. Related Work
   - Subword Tokenization and Multilingual Tokenizer Behaviour
   - Measuring Tokenization Fragmentation
   - Bangla LLMs and Bangla Benchmarking
   - Confidence Estimation and Calibration in LLMs
   - Summary of the Gap
3. Study Design and Research Questions
   - Study Overview
   - Hypotheses (only if pre-specified)
4. Models and Datasets
   - Dataset Sources and Versions
   - Bangla and English Evaluation Conditions
   - Question Pairing and Translation Quality
   - Model and Tokenizer Selection
   - Data Cleaning and Preprocessing
   - Ethics and Data Handling
5. Tokenization Fragmentation Measurement
   - Text Unit Being Tokenized
   - Tokenizer Execution and Reproducibility
   - Word Counting and Normalization
   - TPW
   - STRR (only if verified)
   - Worked Tokenization Examples
6. Confidence and Calibration Methodology
   - Prompt and Multiple-Choice Scoring Protocol
   - Confidence Definition
   - Accuracy
   - Brier Score
   - ECE
   - Reliability Diagrams
   - Calibration Variants and Data Leakage Prevention
7. Experimental Setup
   - Evaluation Pipeline
   - Primary and Secondary Outcomes
   - Statistical Analysis Plan
   - Robustness and Sensitivity Checks
8. Results
   - Dataset and Model Coverage
   - Tokenization Fragmentation by Language and Model
   - Accuracy and Calibration Performance
   - Association Between Fragmentation and Calibration
   - Robustness Checks
9. Discussion
   - Summary of Main Findings
   - Interpretation and Comparison with Prior Work
   - Implications for Bangla LLM Evaluation
10. Limitations and Threats to Validity
    - Construct Validity
    - Internal Validity
    - External Validity
    - Statistical Conclusion Validity
11. Conclusion and Future Work

## Planned tables

| No. | Table | Owner / source |
|---|---|---|
| 1 | Dataset composition, pairing, filtering, final counts | Abdullah |
| 2 | Five worked Bangla-English tokenizer examples | Abdullah supplies pairs; Naeem/Subho validate tokenizer outputs |
| 3 | Model/tokenizer identities and immutable revisions | Naeem + Subho |
| 4 | TPW/verified STRR descriptive statistics | Naeem |
| 5 | Accuracy, Brier, ECE, sample size and bin count | Naeem |
| 6 | Primary fragmentation-calibration association model | Naeem |
| 7 | Robustness/sensitivity analysis, if conducted | Naeem + Subho |

## Planned figures

| No. | Figure | Purpose |
|---|---|---|
| 1 | Study workflow diagram | Show data, paired languages, tokenizer, model scoring, calibration, association analysis |
| 2 | Tokenization example visualization (optional) | Illustrate segmentation of selected matched items |
| 3 | TPW distribution plot | Compare fragmentation by model and language |
| 4 | Reliability diagrams | Compare confidence with observed accuracy |
| 5 | Fragmentation-calibration association plot | Visualize the primary statistical relationship |

Use vector PDF for plots where possible. Every figure needs a number, caption, readable axis labels, units, legend, and an in-text reference. Every table needs a number, caption, clear column definitions, and an in-text reference. Never plot placeholder or unverified values.

## Critical decisions before final results

The current repository branches contain different model panels, prompts, ECE bin counts, and evaluation designs. These results must not be combined without a documented reconciliation.

- The Dipto v3 branch reports 134,375 valid rows per model after excluding 7 invalid rows, with five models and 15-bin ECE in its documented protocol.
- The evaluation-pipeline branch uses a different model/prompt setup and 10-bin ECE.
- The Naeem branch contains an exploratory 180-item run (90 Bangla and 90 English across three models), tokenizer audits, and statistical analysis. Treat it as exploratory until the item-level calculations and repeated metric rows are checked.
- The data/4-question-pairs branch contains translated BnMMLU material. Verify alignment, translation quality, exclusions, and exact snapshot before describing it as a matched cross-lingual dataset.

**Important statistical point:** TPW can be measured for each item and tokenizer, while ECE is an aggregate calibration metric over a group of predictions. Do not assign one global ECE to every question and run a naive item-level correlation. The analysis must define a valid item-level calibration outcome or a justified grouped/hierarchical approach, and account for repeated items within models and subjects.

## Collaboration rules

1. Each person should work on their own branch or a dedicated feature branch, not edit the same `main.tex` simultaneously.
2. Submit a pull request with the section draft plus supporting CSV/JSON, scripts, figures, and source references.
3. Every reported number must be traceable to a script, data snapshot, model revision, and output file.
4. Use `\cite{key}` in LaTeX and add verified BibTeX entries to `references.bib`.
5. Mark unresolved decisions with `TODO`; do not silently guess.
6. Merge the final text only after the team agrees on the shared protocol and terminology.

## Compile

Use the official Springer LNCS class and bibliography style. From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Follow the target venue's current LNCS instructions if they differ from this working template.
