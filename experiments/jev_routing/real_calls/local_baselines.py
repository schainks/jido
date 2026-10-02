#!/usr/bin/env python3
"""Local baselines for next-tool prediction on prepare.py's output. Nothing leaves the machine.

  majority          the most common class in train
  same as previous  the label of the previous call in the same session and thread
  student           Jevstiller's encoder (bge-small, CPU) and LinearStudent, trained on the train split with the real labels
  student + said    the same with the assistant's narration in the text

Scored on the test split (sessions never seen in training): accuracy, top-3 accuracy and macro-F1 over classes
with at least 10 test rows. Embeddings are cached next to the input (private data, mode 600).

Run: python local_baselines.py PREPARED.jsonl [--json OUT.json] [--encoder small] [--max-train N]   (pip install jevstiller numpy)
"""
import argparse, collections, json, time
from pathlib import Path
import numpy as np


def metrics(pred, truth, classes):
    pred, truth = np.array(pred), np.array(truth)
    f1s = []
    for c in classes:
        tp = ((pred == c) & (truth == c)).sum(); fp = ((pred == c) & (truth != c)).sum(); fn = ((pred != c) & (truth == c)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return {"accuracy": round(float((pred == truth).mean()), 4), "macro_f1": round(float(np.mean(f1s)), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prepared"); ap.add_argument("--json"); ap.add_argument("--encoder", default="small"); ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()
    from jevstiller import load_encoder
    from jevstiller._student import LinearStudent
    rows = [json.loads(l) for l in open(a.prepared)]
    train = [r for r in rows if r["split"] == "train"]; test = [r for r in rows if r["split"] == "test"]
    if a.max_train:
        train = train[:a.max_train]
    classes = sorted({r["label"] for r in rows}); idx = {c: i for i, c in enumerate(classes)}
    big = [c for c in classes if sum(r["label"] == c for r in test) >= 10]
    truth = [r["label"] for r in test]
    out = {"n_train": len(train), "n_test": len(test), "classes_scored_in_macro_f1": big}
    maj = collections.Counter(r["label"] for r in train).most_common(1)[0][0]
    out["majority"] = {"class": maj, **metrics([maj] * len(test), truth, big)}
    prev, last = [], {}
    for r in rows:   # rows are in transcript order within a session
        key = (r["session"], r["sidechain"])
        if r["split"] == "test":
            prev.append(last.get(key, maj))
        last[key] = r["label"]
    out["same_as_previous"] = metrics(prev, truth, big)
    enc = load_encoder(a.encoder)
    for name, field in (("student", "text"), ("student_with_narration", "text_narrated")):
        t0 = time.time()
        cache = Path(a.prepared).with_suffix(f".{a.encoder}.{field}.npy")
        if cache.exists() and np.load(cache).shape[0] == len(rows):
            E = np.load(cache)
        else:
            E = np.vstack([enc.encode([r[field] for r in rows[i:i + 256]]) for i in range(0, len(rows), 256)]).astype(np.float32)
            np.save(cache, E); cache.chmod(0o600)
        pos = {r["id"]: i for i, r in enumerate(rows)}
        Xtr = E[[pos[r["id"]] for r in train]]; Xte = E[[pos[r["id"]] for r in test]]
        Y = np.eye(len(classes))[[idx[r["label"]] for r in train]]
        st = LinearStudent(enc.dim, len(classes)); info = st.fit(Xtr, Y, seed=0)
        P = Xte @ st.W + st.b
        pred = [classes[i] for i in P.argmax(1)]
        top3 = float(np.mean([idx[t] in np.argsort(-p)[:3] for p, t in zip(P, truth)]))
        out[name] = {**metrics(pred, truth, big), "top3_accuracy": round(top3, 4), "epochs": info["epochs"], "seconds": round(time.time() - t0)}
        print(name, out[name], flush=True)
    print({k: v for k, v in out.items() if k in ("majority", "same_as_previous")})
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
