import gc, json, time, zipfile
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# hf token from kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = None

# settings
DATASET_ID = "samanjoy2/BnMMLU"
MODELS = {
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
    "titu":  "hishab/titulm-llama-3.2-1b-v2.0",
    "tiger": "md-nishat-008/TigerLLM-1B-it",
}
RUN = ["titu", "tiger", "llama"]   # order of models to run
OUT = Path("/kaggle/working")
SEED = 42
BATCH = 16
CHUNK = 5000      # update backup zip after every 5000 questions
MAX_LEN = 1024
LIMIT = None      # put 200 here to test quickly
LETTERS = ["A", "B", "C", "D"]

# column names, if auto detect fails then write them here
COL_ID = COL_Q = COL_ANS = None
COL_OPTS = None

assert torch.cuda.is_available(), "gpu is off"

# same shuffle every time (seed 42)
ds = load_dataset(DATASET_ID, token=HF_TOKEN)["train"].shuffle(seed=SEED)
if LIMIT:
    ds = ds.select(range(LIMIT))
cols = ds.column_names
print("Columns:", cols)
low = {c.lower(): c for c in cols}


def find(cands):
    for c in cands:
        if c in low:
            return low[c]
    return None


# try to guess the column names
COL_ID = COL_ID or find(["unique_serial", "id"])
COL_Q = COL_Q or find(["question", "prompt", "query"])
COL_ANS = COL_ANS or find(["answer", "correct_answer", "label", "correct"])
if COL_OPTS is None:
    COL_OPTS = [find([f"option_{l.lower()}", f"option {l.lower()}", l.lower(), f"choice_{l.lower()}"]) for l in LETTERS]
assert COL_ID and COL_Q and COL_ANS and all(COL_OPTS), \
    f"could not find columns: {COL_ID, COL_Q, COL_ANS, COL_OPTS}"
print("Using:", COL_ID, COL_Q, COL_OPTS, COL_ANS)

ids = [str(x) for x in ds[COL_ID]]
assert len(set(ids)) == len(ids), "duplicate ids found"


def gold_index(row):
    # answer can be a letter, the option text or a number
    a = str(row[COL_ANS]).strip()
    if a.upper() in LETTERS:
        return LETTERS.index(a.upper())
    opts = [str(row[c]).strip() for c in COL_OPTS]
    if a in opts:
        return opts.index(a)
    if a.isdigit() and 0 <= int(a) <= 3:
        return int(a)
    raise ValueError(f"cant read answer: {a!r}")


def build_text(row):
    opts = "\n".join(f"{l}. {row[c]}" for l, c in zip(LETTERS, COL_OPTS))
    return ("The following is a multiple choice question in Bengali. "
            "Answer with only the letter (A, B, C, or D).\n\n"
            f"Question: {row[COL_Q]}\n{opts}\nAnswer:")


def make_prompt(tok, row):
    text = build_text(row)
    # use chat template if the model has one
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)
    return text


def letter_ids(tok):
    # token ids for A B C D, with and without space
    out = []
    for l in LETTERS:
        s = set()
        for v in (l, " " + l):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                s.add(t[-1])
        out.append(sorted(s))
    return out


def metrics(path):
    recs = [json.loads(x) for x in open(path, encoding="utf-8")]
    P = np.array([r["probs"] for r in recs])
    y = np.array([r["gold"] for r in recs])
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == y).astype(float)
    brier = ((P - np.eye(4)[y]) ** 2).sum(1).mean()
    # ece with 10 bins
    ece, bins = 0.0, np.linspace(0, 1, 11)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return dict(completed=len(recs), accuracy=float(correct.mean()),
                mean_confidence=float(conf.mean()), brier=float(brier), ece=float(ece))


def run_model(key):
    model_id = MODELS[key]
    res_path = OUT / f"{key}_results.jsonl"
    zip_path = OUT / f"{key}_full_backup.zip"
    print(f"\n=== {key}: {model_id} ===")

    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tok.padding_side = "left"   # needed to read the last token correctly
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=HF_TOKEN, dtype=torch.float16, device_map="auto").eval()
    lid = letter_ids(tok)

    # skip questions that are already done (resume)
    done = set()
    if res_path.exists():
        done = {json.loads(x)["id"] for x in open(res_path, encoding="utf-8")}
    todo = [i for i in range(len(ds)) if ids[i] not in done]
    print(f"Done: {len(done)} | Remaining: {len(todo)}")

    t0, new = time.time(), 0
    with open(res_path, "a", encoding="utf-8") as f:
        for c0 in range(0, len(todo), CHUNK):
            chunk = todo[c0:c0 + CHUNK]
            rows = {i: ds[i] for i in chunk}
            prompts = {i: make_prompt(tok, rows[i]) for i in chunk}
            chunk.sort(key=lambda i: len(prompts[i]))   # similar length together, less padding
            for b0 in range(0, len(chunk), BATCH):
                bi = chunk[b0:b0 + BATCH]
                enc = tok([prompts[i] for i in bi], return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_LEN, add_special_tokens=False
                          ).to(model.device)
                with torch.no_grad():
                    # only last token logits, saves memory
                    try:
                        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
                    except TypeError:
                        logits = model(**enc).logits[:, -1, :]
                logits = logits.float()
                scores = torch.stack([torch.logsumexp(logits[:, ids_], dim=-1) for ids_ in lid], dim=1)
                probs = torch.softmax(scores, dim=-1).cpu().numpy()
                for k, i in enumerate(bi):
                    f.write(json.dumps({"id": ids[i], "probs": probs[k].round(6).tolist(),
                                        "pred": int(probs[k].argmax()),
                                        "gold": gold_index(rows[i])}) + "\n")
                new += len(bi)
                f.flush()
            el = time.time() - t0
            print(f"Saved {len(done) + new}/{len(ds)} | {new / el:.1f} q/s | "
                  f"ETA {(len(todo) - new) / max(new / el, 1e-9) / 60:.0f} min")
            # backup
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                z.write(res_path, res_path.name)

    m = metrics(res_path)
    (OUT / f"{key}_summary.json").write_text(json.dumps(m, indent=2))
    print(f"--- FULL {key.upper()} SUMMARY ---")
    for k, v in m.items():
        print(f"{k}: {v}")

    # free gpu before next model
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return m


results = {k: run_model(k) for k in RUN}
print("\nALL DONE", json.dumps(results, indent=2))
