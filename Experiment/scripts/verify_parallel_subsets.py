"""Independent check of the translation-parallel subsets (does NOT import the notebook).

Usage:  python verify_parallel_subsets.py <results_dir> <wide_csv>
        e.g. python verify_parallel_subsets.py /kaggle/working/results /kaggle/input/<ds>/bnmmlu_full_en.csv
Exit code 0 = all checks passed.
"""
import sys, os, json, re
import pandas as pd

res_dir, wide_csv = sys.argv[1], sys.argv[2]
N = int(os.environ.get("EXPECTED_N", "1500"))
DEV_N = int(os.environ.get("EXPECTED_DEV_N", "200"))
EXCLUDED = {"Bengali Language & Syntax", "Bengali Literature", "Bengali Poetry"}
MATHML = re.compile(r"</?(?:math|mi|mn|mo|mrow|msup|msub|msubsup|mfrac|msqrt|mroot|mtext|mstyle|mover|munder|munderover|mtable|mtr|mtd|mspace|mfenced|menclose|mpadded|mphantom|semantics|annotation)(?:\s[^<>]*)?/?>", re.I)
BN = re.compile(r"[\u0980-\u09FF]")
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

R = []
def check(name, ok, detail=""):
    R.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

rd = lambda f: pd.read_csv(os.path.join(res_dir, f), dtype=str, keep_default_na=False)
bn, en, dbn, den = rd("bn_subset.csv"), rd("en_subset.csv"), rd("dev_bn.csv"), rd("dev_en.csv")
ex = json.load(open(os.path.join(res_dir, "fewshot_exemplars.json"), encoding="utf-8"))
meta = json.load(open(os.path.join(res_dir, "subset_meta.json"), encoding="utf-8"))
wide = pd.read_csv(wide_csv, dtype=str, keep_default_na=False).set_index("Unique_Serial", drop=False)

check("sizes", len(bn) == len(en) == N and len(dbn) == len(den) == DEV_N, f"bn={len(bn)} en={len(en)} dev={len(dbn)}/{len(den)}")
check("same question_id order in bn and en (eval)", bn.question_id.tolist() == en.question_id.tolist())
check("same question_id order in bn and en (dev)", dbn.question_id.tolist() == den.question_id.tolist())
check("question_ids unique", bn.question_id.is_unique and dbn.question_id.is_unique)
check("gold identical across languages", (bn.correct_answer == en.correct_answer).all() and (dbn.correct_answer == den.correct_answer).all())
check("subject_group identical across languages", (bn.subject_group == en.subject_group).all())
check("language/benchmark labels", set(bn.language) == {"bn"} and set(en.language) == {"en"} and set(en.benchmark) == {"BnMMLU-en"}, f"{set(en.benchmark)}")

ev, dv = set(bn.question_id), set(dbn.question_id)
xv = {e["question_id"] for e in ex["bn"]}
check("eval/dev/exemplar disjoint by id", not (ev & dv) and not (ev & xv) and not (dv & xv))
txt_eval = set(bn.question) | set(en.question)
txt_other = set(dbn.question) | set(den.question) | {e["question"] for e in ex["bn"]} | {e["question"] for e in ex["en"]}
check("eval vs dev/exemplar disjoint by question text", not (txt_eval & txt_other), f"overlap={len(txt_eval & txt_other)}")
check("few-shot parallel + balanced", [e["question_id"] for e in ex["bn"]] == [e["question_id"] for e in ex["en"]] and len(ex["bn"]) == 8
      and all(sorted(e["correct_answer"] for e in ex["bn"][k:k + 4]) == list("ABCD") for k in (0, 4)))

# Rows must be exactly the wide-CSV rows (no text was altered), on BOTH sides
bad = 0
for df, lang in ((bn, "bn"), (en, "en"), (dbn, "bn"), (den, "en")):
    for r in df.itertuples():
        w = wide.loc[r.question_id]
        opts = json.loads(r.options)
        want = [w[f"option_{c}_{lang}"].strip() for c in "abcd"]
        if r.question != w[f"question_{lang}"].strip() or opts != want or r.correct_answer != w["correct_answer"].strip().upper():
            bad += 1
check("every row matches the wide CSV exactly (question, 4 options, gold)", bad == 0, f"mismatches={bad}")
check("4 non-empty options everywhere", all(len(json.loads(o)) == 4 and all(x.strip() for x in json.loads(o)) for o in pd.concat([bn.options, en.options, dbn.options, den.options])))

check("language-specific subjects excluded", not (set(wide.loc[bn.question_id].subject_name) & EXCLUDED))
check("no MathML in any eval/dev text (both languages)", not any(MATHML.search(t) for t in pd.concat([bn.question, en.question, dbn.question, den.question, bn.options, en.options])))
check("no Bangla script in English side", not any(BN.search(t) for t in pd.concat([en.question, en.options, den.question, den.options])))
check("no untranslated identical bn==en question", not (bn.question.str.strip() == en.question.str.strip()).any())

per = bn.subject_group.value_counts()
check("per-subject balance (max/min <= 1.5 and no subject > cap)", per.max() <= -(-N // len(per) * 3 // 2) and per.max() / per.min() <= 1.5, f"subjects={len(per)} min={per.min()} max={per.max()}")
g = bn.correct_answer.value_counts(normalize=True)
check("gold letters all present, none > 40%", set(g.index) == set("ABCD") and g.max() <= 0.40, g.round(3).to_dict())

# Translation-fidelity heuristic: digits in options must agree after mapping Bengali digits to ASCII
def digits(s): return re.findall(r"\d+(?:\.\d+)?", s.translate(BN_DIGITS))
mism = 0; both_num = 0
for b, e in zip(bn.itertuples(), en.itertuples()):
    for ob, oe in zip(json.loads(b.options), json.loads(e.options)):
        db_, de_ = digits(ob), digits(oe)
        if db_ or de_:
            both_num += 1; mism += (sorted(db_) != sorted(de_))
rate = mism / max(1, both_num)
check("numeric options agree across translation (heuristic, warn if > 5%)", rate <= 0.05, f"{mism}/{both_num} = {rate:.1%}")

check("meta records csv hash + en_source", meta.get("en_source") == "bnmmlu_translation" and len(meta.get("translation_csv_sha256", "")) == 64)
print("\nSUMMARY: %d/%d passed" % (sum(ok for _, ok, _ in R), len(R)))
sys.exit(0 if all(ok for _, ok, _ in R) else 1)
