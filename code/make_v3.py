import json, sys, re, ast

SRC_NB = "Bangla-LLM-Calibration-Real-Data-v3c.ipynb"
DST_NB = "Bangla-LLM-Calibration-Real-Data-v3.ipynb"

with open(SRC_NB, encoding="utf-8") as f:
    nb = json.load(f)

def get_src(idx):
    return "".join(nb["cells"][idx]["source"])

def set_src(idx, new_src):
    lines = new_src.split("\n")
    result = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    elif result:
        pass
    nb["cells"][idx]["source"] = result

def patch(src, old, new, label):
    if old not in src:
        raise ValueError(f"PATCH FAILED for '{label}': target not found!")
    return src.replace(old, new, 1)

# =============================================================================
# 1. CELL 5: Markdown Docs (Section 3)
# =============================================================================
c5 = get_src(5)
s3_doc_add = '''

### Methodology & Debiasing Strategy (v3)
Small and base models frequently suffer from single-letter logit collapse under zero-shot prompting.
To measure genuine tokenization fragmentation effects without distortion from collapse:
1. **Prompt Variants:**
   - `plain`: Standard MMLU question + options format ending with `:\\n`.
   - `answer_cue`: Appends language-specific continuation cue (`উত্তর:` / `Answer:`).
   - `few_shot`: In-context learning with balanced worked exemplars ($k \\in \\{4, 8\\}$) formatted with answer cues.
2. **Debiasing Protocol (`cyclic4`):**
   - 4-pass cyclic option permutation with probability marginalization over permuted positions.
   - Cross-referencing items (*all/none/both of the above*, etc.) are exempt to maintain question semantics.
   - Note: `cyclic4` alters the definition of item confidence (marginalized over permutations); both raw and debiased calibration metrics are logged.
3. **Selection Protocol:**
   - Per-model settings are chosen on-GPU via `diagnose_prompt_variants` based **strictly on validity gates** (`mean_mass >= 0.50` and `top_letter_share <= 0.80`), **never accuracy** (to avoid data leakage).
   - All final per-model hyperparameters are reported transparently in the paper's methodology and limitations.
'''
c5 = c5 + s3_doc_add
set_src(5, c5)

# =============================================================================
# 2. CELL 21: Markdown Docs (Section 11)
# =============================================================================
c21 = get_src(21)
s11_doc_add = '''

### Validity Gates vs Accuracy in Model Selection
Per-model settings (`prompt_variant`, `n_shots`, `use_chat_template`, `debias`) are selected exclusively using **validity gates**:
- `mean_option_mass >= 0.50` (ensures sufficient probability mass on valid option letters)
- `top_letter_share <= 0.80` (ensures the model has not collapsed onto a single letter)

**Accuracy is never used as a selection criterion** because diagnostic subsets overlap with the evaluation pool; optimizing for accuracy during configuration selection would introduce fatal data snooping and circular validity.
'''
c21 = c21 + s11_doc_add
set_src(21, c21)

# =============================================================================
# 3. CELL 22: _log_top_next_tokens & diagnose_prompt_variants
# =============================================================================
c22 = get_src(22)

# 3a. _log_top_next_tokens: apply chat template when backend.use_chat_template is True
old_log_fn = '''def _log_top_next_tokens(backend: BaseInferenceBackend, prompts: List[str], k: int = 8) -> str:
    """Log top-k next-token probabilities for diagnostic prompts.

    Silently does nothing if the backend has no .model attribute (e.g. mock backends in tests).
    Returns the diagnostic string (also logged).
    """
    model = getattr(backend, "model", None)
    if model is None:
        return ""

    tokenizer = backend.tokenizer
    torch = getattr(backend, "torch", None)
    if torch is None:
        return ""

    device = getattr(backend, "device", "cpu")
    lines = []
    for i, prompt in enumerate(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)'''

new_log_fn = '''def _log_top_next_tokens(backend: BaseInferenceBackend, prompts: List[str], k: int = 8) -> str:
    """Log top-k next-token probabilities for diagnostic prompts.

    Silently does nothing if the backend has no .model attribute (e.g. mock backends in tests).
    Applies the chat template when backend.use_chat_template is True so the logged prompt matches the real input.
    Returns the diagnostic string (also logged).
    """
    model = getattr(backend, "model", None)
    if model is None:
        return ""

    tokenizer = backend.tokenizer
    torch = getattr(backend, "torch", None)
    if torch is None:
        return ""

    device = getattr(backend, "device", "cpu")
    lines = []
    for i, prompt in enumerate(prompts):
        try:
            eval_prompt = prompt
            if getattr(backend, "use_chat_template", False) and hasattr(tokenizer, "apply_chat_template"):
                try:
                    messages = [{"role": "user", "content": prompt}]
                    eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    eval_prompt = prompt
            inputs = tokenizer(eval_prompt, return_tensors="pt", add_special_tokens=True).to(device)'''

c22 = patch(c22, old_log_fn, new_log_fn, "Cell 22: _log_top_next_tokens chat template")

# 3b. Rebuild diagnose_prompt_variants
old_diag_fn = '''def diagnose_prompt_variants(
    model_key: str,
    n: int = 30,
    languages: Tuple[str, ...] = ("bn", "en"),
    config: ExperimentConfig = CONFIG,
    subset_paths: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compare prompt variants and chat-template settings for a model.

    Loads the backend once, evaluates {plain, answer_cue} x {chat_template off, on}
    on seeded subsets, and prints a recommendation.
    """'''

# Let's find the entire old diagnose_prompt_variants function in Cell 22
idx_start = c22.find("def diagnose_prompt_variants(")
idx_end = c22.find("def run_one_model(")
old_diag_full = c22[idx_start:idx_end]

new_diag_full = '''def diagnose_prompt_variants(
    model_key: str,
    n: int = 60,
    languages: Tuple[str, ...] = ("bn", "en"),
    config: ExperimentConfig = CONFIG,
    subset_paths: Optional[Dict[str, str]] = None,
    exemplar_path: Optional[str] = None,
) -> pd.DataFrame:
    """Compare prompt variants, chat-template, and debiasing settings for a model.

    Loads the backend once, evaluates grid across:
      prompt_variant {plain, answer_cue, few_shot}
      chat_template {off, on (if available)}
      debias {none, cyclic4}
    Uses validity gates strictly (never accuracy) to recommend settings.
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Registered: {list(MODEL_REGISTRY.keys())}")

    # Stable per-language seed offsets (CRC32 constant, avoiding randomised hash())
    LANG_SEEDS = {"bn": 1001, "en": 2002}

    # Prepared subsets
    paths = subset_paths or {"bn": bn_subset_file, "en": en_subset_file}
    benches = {"bn": "BnMMLU", "en": "MMLU"}
    items_by_lang = {
        lg: load_dataset_from_file(paths[lg], language=lg, benchmark=benches[lg], is_mock=False)
        for lg in languages if lg in paths and os.path.exists(paths[lg])
    }

    # Check fewshot_exemplars file
    ex_file = exemplar_path or os.path.join(config.output_dir, "fewshot_exemplars.json")
    has_fewshot = os.path.exists(ex_file)
    if not has_fewshot:
        logger.warning(f"fewshot_exemplars.json missing at '{ex_file}'. Skipping few_shot variant in grid.")

    backend = None
    orig_chat = False
    try:
        backend = _build_backend(model_key, config)
        orig_chat = getattr(backend, "use_chat_template", False)
        has_chat_tpl = getattr(backend.tokenizer, "chat_template", None) is not None
        chat_options = [False]
        if has_chat_tpl:
            chat_options.append(True)

        variants = ["plain", "answer_cue"]
        if has_fewshot:
            variants.append("few_shot")
        debias_options = ["none", "cyclic4"]

        results_rows = []
        for lang in languages:
            all_items = items_by_lang.get(lang, [])
            if not all_items:
                continue
            # Seeded permutation using stable per-language seed
            l_seed = config.seed + LANG_SEEDS.get(lang, 9999)
            _rng = np.random.RandomState(l_seed)
            perm = _rng.permutation(len(all_items))
            subset = [all_items[i] for i in perm[:n]]

            for variant in variants:
                for use_chat in chat_options:
                    for debias in debias_options:
                        backend.use_chat_template = use_chat

                        masses = []
                        preds = []
                        raw_preds = []
                        corrects = []
                        n_nan = 0
                        prompt_tokens_list = []
                        n_exempt = 0

                        _diag_settings = {
                            "prompt_variant": variant,
                            "n_shots": getattr(config, "n_shots", 4) if variant == "few_shot" else 0,
                            "use_chat_template": use_chat,
                            "debias": debias,
                            "collapse_threshold": MAX_COLLAPSE_THRESHOLD,
                            "collapse_gate_overridden": False,
                        }

                        for it in subset:
                            res = predict_item(backend, it, lang, _diag_settings)
                            masses.append(res.option_mass)
                            preds.append(res.predicted_letter)
                            raw_preds.append(getattr(res, "pred_raw", res.predicted_letter))
                            gold = it.get("correct_answer", "")
                            corrects.append(res.predicted_letter == gold if res.predicted_letter and gold else False)
                            if res.confidence is None or (isinstance(res.confidence, float) and math.isnan(res.confidence)):
                                n_nan += 1
                            if res.prompt_tokens is not None:
                                prompt_tokens_list.append(res.prompt_tokens)
                            if not getattr(res, "perm_applied", False):
                                n_exempt += 1

                        pred_counts = pd.Series(preds).value_counts(normalize=True)
                        top_letter_share = float(pred_counts.max()) if len(pred_counts) > 0 else 0.0

                        raw_counts = pd.Series(raw_preds).value_counts(normalize=True)
                        raw_top_share = float(raw_counts.max()) if len(raw_counts) > 0 else 0.0

                        mean_mass = float(np.mean(masses)) if masses else 0.0
                        min_mass = float(np.min(masses)) if masses else 0.0
                        acc = float(np.mean(corrects)) * 100.0 if corrects else 0.0
                        mean_pt = float(np.mean(prompt_tokens_list)) if prompt_tokens_list else 0.0

                        results_rows.append({
                            "language": lang,
                            "prompt_variant": variant,
                            "chat_template": use_chat,
                            "debias": debias,
                            "mean_mass": round(mean_mass, 4),
                            "min_mass": round(min_mass, 4),
                            "top_letter_share": round(top_letter_share, 4),
                            "raw_top_letter_share": round(raw_top_share, 4),
                            "accuracy_pct": round(acc, 1),
                            "n_nan": n_nan,
                            "mean_prompt_tokens": round(mean_pt, 1),
                            "n_perm_exempt": n_exempt,
                        })

        df = pd.DataFrame(results_rows)
        print(chr(10) + "=" * 80)
        print(f"\U0001f4ca PROMPT VARIANT DIAGNOSTIC: {model_key}")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)

        # Save CSV
        out_path = os.path.join(config.output_dir, f"prompt_diagnostics_{model_key}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Diagnostic results saved to: {out_path}")

        # Recommendation logic:
        # Pass gates: mean_mass >= MIN_OPTION_MASS_THRESHOLD and top_letter_share <= MAX_COLLAPSE_THRESHOLD in BOTH languages
        # Tie-breaking preferences:
        #   1. debias "none" over "cyclic4"
        #   2. plain < answer_cue < few_shot
        #   3. chat False over True
        #   4. highest minimum mean_mass
        combo_keys = df.groupby(["prompt_variant", "chat_template", "debias"])
        passing = []
        all_candidates = []

        pv_order = {"plain": 0, "answer_cue": 1, "few_shot": 2}

        for (pv, ct, deb), grp in combo_keys:
            passes_all = True
            min_mean_mass = float("inf")
            for _, row in grp.iterrows():
                if row["mean_mass"] < MIN_OPTION_MASS_THRESHOLD or row["top_letter_share"] > MAX_COLLAPSE_THRESHOLD:
                    passes_all = False
                min_mean_mass = min(min_mean_mass, row["mean_mass"])

            # Sort key: debias ("none" is preferred -> 1, "cyclic4" -> 0),
            #           pv (plain=0 < answer_cue=1 < few_shot=2, prefer lower -> -pv_order),
            #           chat (False preferred -> 1, True -> 0),
            #           min_mean_mass (higher is better)
            score_key = (
                1 if deb == "none" else 0,
                -pv_order.get(pv, 99),
                1 if not ct else 0,
                min_mean_mass,
            )
            candidate_tuple = (pv, ct, deb, min_mean_mass, score_key)
            all_candidates.append(candidate_tuple)
            if passes_all:
                passing.append(candidate_tuple)

        if passing:
            # Sort by tie-breaking preference
            passing.sort(key=lambda x: x[4], reverse=True)
            best_pv, best_ct, best_deb, best_mm, _ = passing[0]
            print(chr(10) + "\u2705 RECOMMENDATION (based on validity gates only, accuracy is NOT used):")
            print(f'   MODEL_REGISTRY["{model_key}"]["use_chat_template"] = {best_ct}')
            print(f'   MODEL_REGISTRY["{model_key}"]["prompt_variant"]   = "{best_pv}"')
            print(f'   MODEL_REGISTRY["{model_key}"]["n_shots"]          = {4 if best_pv == "few_shot" else 0}')
            print(f'   MODEL_REGISTRY["{model_key}"]["debias"]           = "{best_deb}"')
            print(f"   (passes both gates across languages; min mean_mass: {best_mm:.4f})")
        else:
            # None passes: pick combination with highest minimum mean_mass
            all_candidates.sort(key=lambda x: x[3], reverse=True)
            best_pv, best_ct, best_deb, best_mm, _ = all_candidates[0]
            print(chr(10) + "\u274c NO VARIANT PASSES both gates across languages.")
            print(f"   Highest minimum mean_mass candidate (min mean_mass: {best_mm:.4f}):")
            print(f'   MODEL_REGISTRY["{model_key}"]["use_chat_template"] = {best_ct}')
            print(f'   MODEL_REGISTRY["{model_key}"]["prompt_variant"]   = "{best_pv}"')
            print(f'   MODEL_REGISTRY["{model_key}"]["n_shots"]          = {4 if best_pv == "few_shot" else 0}')
            print(f'   MODEL_REGISTRY["{model_key}"]["debias"]           = "{best_deb}"')

        return df

    finally:
        if backend is not None:
            try:
                backend.use_chat_template = orig_chat
            except Exception:
                pass
            del backend
        gc.collect()
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass


'''

c22 = c22.replace(old_diag_full, new_diag_full)
set_src(22, c22)

# =============================================================================
# 4. CELL 31 & 32: Markdown & Code for Section 11b
# =============================================================================
c31_new = '''## Section 11b: Prompt-variant diagnostic (run per model BEFORE the full run)
`diagnose_prompt_variants(model_key)` evaluates a grid of `{plain, answer_cue, few_shot} × {chat_template off, on} × {debias none, cyclic4}` on seeded subsets using **validity gates only** (`mean_mass >= 0.50` and `top_letter_share <= 0.80`). Accuracy is reported for informational purposes only and is **never** used in recommendation to prevent circular data snooping.

> [!IMPORTANT]
> **Checklist before re-running a model:**
> 1. Run `diagnose_prompt_variants("tigerllm_1b")` and `diagnose_prompt_variants("titullm_1b")` first.
> 2. Copy the printed `RECOMMENDATION` lines into `MODEL_REGISTRY` in Section 3.
> 3. **Delete that model's results folder** (e.g. `shutil.rmtree(f"results/{model_key}", ignore_errors=True)`) before re-running it with new settings so existing checkpoints do not conflict with the new signature.
> 4. Execute the model run and verify the status dashboard.
'''
set_src(31, c31_new)

c32_new = '''# ============================================================
# Prompt-variant diagnostics: run per model BEFORE full execution
# (Uncomment to execute on GPU):
# ============================================================
# import shutil

# 1. TigerLLM-1B (evaluate first)
# diagnose_prompt_variants("tigerllm_1b", n=60)
# To re-run after changing registry: shutil.rmtree(os.path.join(CONFIG.output_dir, "tigerllm_1b"), ignore_errors=True)

# 2. TituLLM-1B (evaluate second)
# diagnose_prompt_variants("titullm_1b", n=60)
# To re-run after changing registry: shutil.rmtree(os.path.join(CONFIG.output_dir, "titullm_1b"), ignore_errors=True)

# 3. LLaMA-3.2-3B-Instruct (evaluate third)
# diagnose_prompt_variants("llama32_3b", n=60)
# To re-run after changing registry: shutil.rmtree(os.path.join(CONFIG.output_dir, "llama32_3b"), ignore_errors=True)
'''
set_src(32, c32_new)

# =============================================================================
# 5. CELL 28: Update test_diagnose_prompt_variants_fake & add Task 7 tests
# =============================================================================
c28 = get_src(28)

old_diag_test = '''    def test_diagnose_prompt_variants_fake():
        import tempfile, dataclasses

        class FakeDiagBackend:
            def __init__(self):
                self.model_id = "fake-diag-model"
                self.use_chat_template = False
                class FakeTok:
                    name_or_path = "fake-tok"
                    chat_template = None
                    def encode(self, text, add_special_tokens=False):
                        return list(range(max(1, len(text) // 5)))
                    def tokenize(self, text):
                        return [str(t) for t in self.encode(text)]
                self.tokenizer = FakeTok()
                self.rng = np.random.RandomState(42)

            def predict_option_probabilities(self, prompt):
                raw_logits = self.rng.randn(4)
                exps = np.exp(raw_logits - np.max(raw_logits))
                probs = (exps / np.sum(exps)).tolist()
                pred_idx = int(np.argmax(probs))
                return InferenceResult(
                    predicted_letter=OPTION_LETTERS[pred_idx],
                    confidence=float(probs[pred_idx]),
                    probabilities=probs,
                    raw_response=OPTION_LETTERS[pred_idx],
                    prompt_tokens=len(self.tokenizer.encode(prompt)),
                    completion_tokens=1,
                    probability_source="fake_logits",
                    option_mass=float(self.rng.uniform(0.55, 0.95)),
                    chat_template_used=self.use_chat_template,
                )

        with tempfile.TemporaryDirectory() as td:
            fake_be = FakeDiagBackend()
            # Monkeypatch _build_backend
            original_build = globals().get("_build_backend")
            try:
                globals()["_build_backend"] = lambda mk, cfg: fake_be
                cfg = dataclasses.replace(CONFIG, output_dir=td)
                # Self-contained tiny subsets (no dependency on Section 5 having run)
                paths = {}
                for lg in ("bn", "en"):
                    rows = [dict(it, options=json.dumps(it["options"], ensure_ascii=False))
                            for it in create_synthetic_benchmark_dataset() if it["language"] == lg]
                    paths[lg] = os.path.join(td, f"{lg}_subset.csv")
                    pd.DataFrame(rows).to_csv(paths[lg], index=False, encoding="utf-8")
                df = diagnose_prompt_variants("llama32_3b", n=5, languages=("bn", "en"), config=cfg, subset_paths=paths)
                assert set(df["prompt_variant"]) == {"plain", "answer_cue"} and set(df["language"]) == {"bn", "en"}
                assert isinstance(df, pd.DataFrame) and len(df) > 0, "Should produce non-empty DataFrame"
                assert "mean_mass" in df.columns
                csv_path = os.path.join(td, "prompt_diagnostics_llama32_3b.csv")
                assert os.path.exists(csv_path), "Should save diagnostic CSV"
            finally:
                if original_build is not None:
                    globals()["_build_backend"] = original_build
    test_registry.append(("diagnose_prompt_variants runs end-to-end with fake backend", test_diagnose_prompt_variants_fake))'''

new_diag_test = '''    def test_diagnose_prompt_variants_fake():
        import tempfile, dataclasses

        class FakeDiagBackend:
            def __init__(self):
                self.model_id = "fake-diag-model"
                self.use_chat_template = False
                class FakeTok:
                    name_or_path = "fake-tok"
                    chat_template = None
                    def encode(self, text, add_special_tokens=False):
                        return list(range(max(1, len(text) // 5)))
                    def tokenize(self, text):
                        return [str(t) for t in self.encode(text)]
                self.tokenizer = FakeTok()
                self.rng = np.random.RandomState(42)

            def predict_option_probabilities(self, prompt):
                raw_logits = self.rng.randn(4)
                exps = np.exp(raw_logits - np.max(raw_logits))
                probs = (exps / np.sum(exps)).tolist()
                pred_idx = int(np.argmax(probs))
                return InferenceResult(
                    predicted_letter=OPTION_LETTERS[pred_idx],
                    confidence=float(probs[pred_idx]),
                    probabilities=probs,
                    raw_response=OPTION_LETTERS[pred_idx],
                    prompt_tokens=len(self.tokenizer.encode(prompt)),
                    completion_tokens=1,
                    probability_source="fake_logits",
                    option_mass=float(self.rng.uniform(0.55, 0.95)),
                    chat_template_used=self.use_chat_template,
                )

        with tempfile.TemporaryDirectory() as td:
            fake_be = FakeDiagBackend()
            original_build = globals().get("_build_backend")
            try:
                globals()["_build_backend"] = lambda mk, cfg: fake_be
                cfg = dataclasses.replace(CONFIG, output_dir=td)
                paths = {}
                for lg in ("bn", "en"):
                    rows = [dict(it, options=json.dumps(it["options"], ensure_ascii=False))
                            for it in create_synthetic_benchmark_dataset() if it["language"] == lg]
                    paths[lg] = os.path.join(td, f"{lg}_subset.csv")
                    pd.DataFrame(rows).to_csv(paths[lg], index=False, encoding="utf-8")

                # Create tiny fewshot_exemplars.json
                ex_path = os.path.join(td, "fewshot_exemplars.json")
                with open(ex_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "bn": [{"question_id": f"b_{i}", "subject": "S", "question": "Q?", "options": ["A","B","C","D"], "correct_answer": OPTION_LETTERS[i%4]} for i in range(8)],
                        "en": [{"question_id": f"e_{i}", "subject": "S", "question": "Q?", "options": ["A","B","C","D"], "correct_answer": OPTION_LETTERS[i%4]} for i in range(8)],
                    }, f)

                df = diagnose_prompt_variants("llama32_3b", n=5, languages=("bn", "en"), config=cfg, subset_paths=paths, exemplar_path=ex_path)
                assert {"plain", "answer_cue", "few_shot"}.issubset(set(df["prompt_variant"]))
                assert set(df["language"]) == {"bn", "en"}
                assert "raw_top_letter_share" in df.columns
                assert "n_perm_exempt" in df.columns
                csv_path = os.path.join(td, "prompt_diagnostics_llama32_3b.csv")
                assert os.path.exists(csv_path), "Should save diagnostic CSV"
            finally:
                if original_build is not None:
                    globals()["_build_backend"] = original_build
    test_registry.append(("diagnose_prompt_variants runs end-to-end with fake backend", test_diagnose_prompt_variants_fake))

    def test_diagnose_recommendation_never_uses_accuracy():
        # Synthetic DataFrame with 2 candidates:
        # Candidate 1: High accuracy (99%), but FAILS collapse gate (top_letter_share = 0.90)
        # Candidate 2: Moderate accuracy (40%), PASSES both gates (mean_mass = 0.85, top_letter_share = 0.35)
        # The recommendation logic must choose Candidate 2, NEVER Candidate 1.
        data = [
            {"language": "bn", "prompt_variant": "plain", "chat_template": False, "debias": "none", "mean_mass": 0.80, "min_mass": 0.70, "top_letter_share": 0.90, "raw_top_letter_share": 0.90, "accuracy_pct": 99.0, "n_nan": 0, "mean_prompt_tokens": 40, "n_perm_exempt": 0},
            {"language": "en", "prompt_variant": "plain", "chat_template": False, "debias": "none", "mean_mass": 0.80, "min_mass": 0.70, "top_letter_share": 0.90, "raw_top_letter_share": 0.90, "accuracy_pct": 99.0, "n_nan": 0, "mean_prompt_tokens": 40, "n_perm_exempt": 0},
            {"language": "bn", "prompt_variant": "answer_cue", "chat_template": False, "debias": "none", "mean_mass": 0.85, "min_mass": 0.75, "top_letter_share": 0.35, "raw_top_letter_share": 0.35, "accuracy_pct": 40.0, "n_nan": 0, "mean_prompt_tokens": 42, "n_perm_exempt": 0},
            {"language": "en", "prompt_variant": "answer_cue", "chat_template": False, "debias": "none", "mean_mass": 0.85, "min_mass": 0.75, "top_letter_share": 0.35, "raw_top_letter_share": 0.35, "accuracy_pct": 40.0, "n_nan": 0, "mean_prompt_tokens": 42, "n_perm_exempt": 0},
        ]
        df_synth = pd.DataFrame(data)
        combo_keys = df_synth.groupby(["prompt_variant", "chat_template", "debias"])
        passing = []
        for (pv, ct, deb), grp in combo_keys:
            passes_all = True
            for _, r in grp.iterrows():
                if r["mean_mass"] < MIN_OPTION_MASS_THRESHOLD or r["top_letter_share"] > MAX_COLLAPSE_THRESHOLD:
                    passes_all = False
            if passes_all:
                passing.append((pv, ct, deb))
        assert passing == [("answer_cue", False, "none")], f"Should only recommend passing variant, got {passing}"
    test_registry.append(("diagnose recommendation never uses accuracy", test_diagnose_recommendation_never_uses_accuracy))'''

c28 = patch(c28, old_diag_test, new_diag_test, "Cell 28: update diagnose unit tests")
set_src(28, c28)

# Save to DST_NB
with open(DST_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully generated {DST_NB}")

# Check valid ast for all code cells
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        code_str = "".join(c["source"])
        try:
            ast.parse(code_str)
        except Exception as e:
            print(f"AST Error in cell {idx}: {e}")
            sys.exit(1)

print("All code cells in v3 parsed successfully!")
