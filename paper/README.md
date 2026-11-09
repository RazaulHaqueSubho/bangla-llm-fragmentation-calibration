# Paper workspace: Springer LNCS

This folder contains the shared manuscript skeleton for:

**Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs? A Cross-Lingual Study**

## Files

- **main.tex**: single-file LNCS manuscript skeleton. Each author can fill the relevant sections.
- **references.bib**: shared BibTeX bibliography. Add verified references here; do not invent bibliographic details.

## Suggested section ownership

| Paper section | Primary contributor(s) | Review |
|---|---|---|
| Introduction and research framing | Md. Razaul Haque Subho | All authors |
| Background: tokenization and fragmentation | Ayetullah Khomane / Subho | Naeem |
| Models and execution protocol | Dipto Karmakar | Subho, Naeem |
| Dataset and cross-lingual alignment | Abdullah al Azmi | Subho, Dipto |
| Confidence and calibration methodology | Naeem Hossain | Dipto, Subho |
| Results and statistical interpretation | Relevant analysis owners jointly | All authors |
| Discussion, limitations, conclusion | Subho, all authors | All authors |

This is a suggested division, not a claim that the sections are already complete.

## Important repository findings to reconcile before writing final results

The manuscript deliberately leaves empirical claims as TODOs because the current branches do not yet represent one unified experimental protocol.

1. **feature/dipto-bnmmlu-v3** reports a full BnMMLU evaluation of 134,375 valid rows per model (7 invalid rows excluded) for Qwen2.5-1.5B, SmolLM2-1.7B, TituLLM-3B v2.0, TigerLLM-1B-it, and Llama-3.2-3B-Instruct. Its README documents plain-completion scoring and 15-bin ECE, with cross-fitted temperature-scaled results also reported.
2. **feature/bnmmlu-eval-pipeline** uses a different model set/version and a different prompt construction. Its script calculates ECE with 10 bins and should not be silently combined with the v3 results.
3. **naeem** contains a 180-item exploratory run (90 Bangla and 90 English across three models), tokenizer audit outputs, and a statistical report. The report itself flags observational analysis and few clusters. Its tokenizer audit reports identical Bangla mean TPW values for Llama and TituLLM, and the report shows repeated identical metric rows across models; inspect the underlying item-level outputs and implementation before using these findings as publication evidence.
4. **data/4-question-pairs** contains a Bangla-to-English translation pipeline and translated BnMMLU files. This is potentially useful for matched cross-lingual comparisons, but translation quality, failed rows, item alignment, and review procedures must be documented. It is not the same as using independent MMLU English questions.
5. PRs #7, #8, #9, and #10 are separate work streams. Their methods, model revisions, prompts, samples, and metrics need a deliberate reconciliation before the paper reports a single result.

## Writing and reproducibility rules

- Keep one sentence per source line where practical.
- Use \\cite{key} for citations and add the corresponding verified BibTeX entry to references.bib.
- Record exact model repository IDs and immutable revisions.
- Label preliminary, exploratory, and final results accurately.
- Never claim that fragmentation causes miscalibration; the project investigates associations.
- Do not change the TPW word-counting rules or STRR definition independently in different sections.
- Freeze the prompt, probability extraction, calibration bins, and primary analysis before producing final tables.
- Keep figures in this folder and use descriptive filenames such as fig-reliability.pdf.

## Compile

Compile from this directory with the Springer LNCS class file (llncs.cls) and BibTeX available: pdflatex main.tex, bibtex main, pdflatex main.tex, pdflatex main.tex.

The conference's official LNCS instructions take precedence over this working skeleton.
