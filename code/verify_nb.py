#!/usr/bin/env python3
"""verify_nb.py <notebook.ipynb>
A) run the in-notebook unit tests (torch stubbed, nothing downloaded)
B) dry run of run_one_model for every MODEL_REGISTRY key with ONLY create_inference_backend patched (the real _build_backend is used)."""
import sys, os, types, json, tempfile, traceback
nb_path = os.path.abspath(sys.argv[1])
os.environ["HF_TOKEN"] = "dummy"
work = tempfile.mkdtemp(prefix="verify_")
os.chdir(work)
t = types.ModuleType("torch"); t.__version__ = "stub"
class _Cuda:
    @staticmethod
    def is_available(): return False
t.cuda = _Cuda(); t.float16 = "float16"; t.float32 = "float32"; t.manual_seed = lambda s: None
t.Tensor = type("Tensor", (), {}); t.is_tensor = lambda x: False; t.dtype = type("dtype", (), {})
sys.modules["torch"] = t
nb = json.load(open(nb_path, encoding="utf-8"))
ns = {"__name__": "__main__", "NOTEBOOK_PATH": nb_path}
code = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
test_idx = [i for i, s in code if "tests_ok = run_notebook_unit_tests()" in s]
assert len(test_idx) == 1, "unit-test cell not found"
ok = True
for i, s in code:
    if i > test_idx[0]:
        break
    if "# Execute subset preparation" in s:
        s = s[: s.index("# Execute subset preparation")]
    try:
        exec(compile(s, f"cell{i}", "exec"), ns)
    except BaseException as e:
        ok = False
        print(f"[A] cell {i} FAILED: {type(e).__name__}: {str(e)[:300]}")
        if i == test_idx[0]:
            break
print("[A] UNIT TESTS:", "PASS" if ok else "FAIL")
if ok:
    import pandas as pd
    CONFIG = ns["CONFIG"]; CONFIG.n_samples = 30
    out = os.path.join(work, "dry_results"); os.makedirs(out, exist_ok=True); CONFIG.output_dir = out
    N = 30
    def mk(lang, bench, q, opts):
        subj = ["physics", "biology", "chemistry"]
        rows = []
        for k in range(N):
            letter = "ABCD"[k % 4]
            rows.append({"question_id": f"{lang}_{k}", "subject": subj[k % 3], "subject_group": subj[k % 3], "question": f"{q} {k}",
                         "options": json.dumps([o + str(k) for o in opts], ensure_ascii=False), "correct_answer": letter,
                         "answer_idx": "ABCD".index(letter), "language": lang, "benchmark": bench, "has_markup": False, "raw_index": k})
        return pd.DataFrame(rows)
    bn_f, en_f = os.path.join(out, "bn_subset.csv"), os.path.join(out, "en_subset.csv")
    mk("bn", "BnMMLU", "bangla question", ["k", "kh", "g", "gh"]).to_csv(bn_f, index=False, encoding="utf-8")
    mk("en", "MMLU", "english question", ["a", "b", "c", "d"]).to_csv(en_f, index=False, encoding="utf-8")
    ns["bn_subset_file"], ns["en_subset_file"] = bn_f, en_f
    ex = [{"question_id": f"ex{i}", "subject": "s", "question": f"Example {i}", "options": ["o1", "o2", "o3", "o4"], "correct_answer": "ABCD"[i % 4]} for i in range(8)]
    with open(os.path.join(out, "fewshot_exemplars.json"), "w", encoding="utf-8") as f:
        json.dump({"bn": ex, "en": ex, "version": "t", "n_shots_max": 8}, f, ensure_ascii=False)
    Mock = ns["MockInferenceBackend"]
    seen = []
    def fake_create(backend_type, model_id, **kw):
        seen.append((model_id, kw.get("use_chat_template")))
        return Mock(model_id, seed=7)
    ns["create_inference_backend"] = fake_create
    for key in list(ns["MODEL_REGISTRY"].keys()):
        try:
            ns["run_one_model"](key, languages=("bn", "en"), config=CONFIG)
            st = json.load(open(os.path.join(out, key, "status.json")))
            print(f"[B] {key}: state={st.get('state')} n_bn={st.get('n_done_bn')} n_en={st.get('n_done_en')}")
        except BaseException as e:
            ok = False
            print(f"[B] {key} FAILED: {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
    print("[B] backends built (model_id, use_chat_template):", seen)
print("VERIFY RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
