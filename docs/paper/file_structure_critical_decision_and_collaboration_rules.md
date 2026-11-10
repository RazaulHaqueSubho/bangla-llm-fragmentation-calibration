# Files

- `main.tex`: detailed Springer LNCS manuscript structure with section/subsection prompts, equations, and a figure/table plan.
- `references.bib`: shared BibTeX bibliography. Add only verified references.

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
