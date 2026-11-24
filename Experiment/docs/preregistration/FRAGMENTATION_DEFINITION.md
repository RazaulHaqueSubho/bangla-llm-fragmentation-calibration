# Tokenization Fragmentation Metrics: Formal Definitions & Implementation Audit

This document provides the formal mathematical definitions, exact algorithmic procedures, and preprocessing specifications for all tokenization fragmentation metrics computed across the evaluation subsets in the research project:
**"Does Tokenization Fragmentation Predict Miscalibration in Bangla LLMs?A cross Lingual Study."**

---

## 1. Metric Formulations

### 1.1 Tokens-per-Word (TPW / Fertility)
Measures the average number of subword tokens produced per whitespace-delimited word in the item prompt.

$$\text{TPW} = \frac{N_{\text{tokens}}}{\max(N_{\text{words}}, 1)}$$

- **Interpretation**: A higher TPW indicates severe subword fragmentation (the tokenizer splits single lexical words into multiple pieces, e.g., byte fallbacks).
- **Theoretical Minimum**: $1.0$ (when every word corresponds to exactly one token or fewer).

### 1.2 Tokens-per-Character (TPC / Character Fertility)
Measures the ratio of generated subword tokens to total character length of the item prompt.

$$\text{TPC} = \frac{N_{\text{tokens}}}{\max(\text{len}(\text{text}), 1)}$$

- **Implementation Details**: In the per-item production pipeline (`_item_metrics`), the denominator is the total character length $\text{len}(\text{text})$ including single spaces between words. (In isolated character fertility diagnostics, non-whitespace character count $\text{len}(\text{text.replace(' ', ''))}$ is also referenced; here the verified benchmark denominator is $\max(\text{len}(\text{text}), 1)$).
- **Interpretation**: TPC controls for varying average word lengths across languages and scripts.

### 1.3 Single-Token Retention Rate (STRR) / Subword Token Recognition Rate
In the empirical item-level fragmentation pipeline, STRR evaluates the fraction of generated tokens that represent fragmented subword units:

$$\text{STRR} = 1.0 - \frac{N_{\text{words}}}{\max(N_{\text{tokens}}, 1)} \quad (\text{if } N_{\text{tokens}} > 0, \text{ else } 0.0)$$

- **Algebraic Derivation**:
  $$\frac{N_{\text{tokens}} - N_{\text{words}}}{N_{\text{tokens}}} = 1.0 - \frac{N_{\text{words}}}{N_{\text{tokens}}}$$
- **Interpretation**: When every word is an atomic token ($N_{\text{tokens}} = N_{\text{words}}$), $\text{STRR} = 0.0$. As words fragment into multiple subword pieces ($N_{\text{tokens}} \gg N_{\text{words}}$), $\text{STRR} \to 1.0$.
- **Section 6 Atomic Word Definition Contrast**: Note that in exploratory lexicon audits, atomic retention is defined as $\frac{1}{|W|} \sum_{w \in W} \mathbb{I}(\text{len}(\text{tokenize}(w)) = 1)$. In the per-item empirical regression tables (`tokenizer_fragmentation_per_item.csv`), the exact formula implemented and verified across all runs is $1.0 - \frac{N_{\text{words}}}{\max(N_{\text{tokens}}, 1)}$.

---

## 2. Tokenization & Preprocessing Specifications

### 2.1 How Words Are Counted
Words are identified strictly via whitespace splitting:
```python
words = text.split()
n_words = len(words)
```
Consecutive whitespace characters (spaces, tabs, newlines) are treated as single delimiters, and leading/trailing whitespace is stripped.

### 2.2 How Roots Are Obtained for STRR
In this benchmark, **no external morphological stemmer, lemmatizer, or root analyzer** is applied. 
- In Bengali, rule-based or statistical stemmers often introduce language-specific noise and fail on technical/borrowed terms.
- The surface whitespace-delimited tokens from the standardized BnMMLU item prompts serve as the base lexical units.
- STRR measures subword fragmentation relative to these observed surface words.

### 2.3 Exact Text Field Tokenized
Tokenization is performed on the marginal prompt string that the model receives for each evaluation item, **excluding the static 4-shot exemplar prefix**:

- **Bangla (`bn`)**:
  ```text
  প্রশ্ন: <question>
  বিকল্পগুলি:
  A. <opt_a>
  B. <opt_b>
  C. <opt_c>
  D. <opt_d>
  ```
- **English (`en`)**:
  ```text
  Question: <question>
  Options:
  A. <opt_a>
  B. <opt_b>
  C. <opt_c>
  D. <opt_d>
  ```

Where:
- `<question>` is stripped of leading and trailing whitespace.
- Each option `<opt_a>` through `<opt_d>` is stripped of whitespace and prefixed by its uppercase label `A. `, `B. `, `C. `, `D. `.
- Options are joined by single newline characters `\n`.
- The static 4-shot prompt demonstrations are excluded because they are identical across all items and would dilute marginal item-level variance.

### 2.4 Special Tokens and BOS/EOS Handling
Special tokens are explicitly excluded:
```python
tokenizer.encode(text, add_special_tokens=False)
```
- Beginning-of-Sequence (`<s>`, `<|begin_of_text|>`), End-of-Sequence (`</s>`, `<|end_of_text|>`), and Chat-Template tokens are NOT included in $N_{\text{tokens}}$.
- Only the subword tokens directly encoding the prompt text are counted.

### 2.5 Language-Specific Preprocessing
- **No text alteration**: No Unicode normalization conversions (NFC/NFD), no transliteration, and no stop-word removals are performed.
- Authentic BnMMLU strings (Bengali script, UTF-8 encoded) and MMLU strings (ASCII/Latin UTF-8 encoded) are evaluated as presented in the frozen `bn_subset.csv` and `en_subset.csv`.
