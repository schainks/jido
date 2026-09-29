#!/usr/bin/env python3
"""Logistic regression on the frozen Qwen3-8B state embeddings, as a baseline for the fine-tuned head.

CLM's PR 13 reports this matching or beating the fine-tuned head on seen classes, so it shows whether
the encoder or the head is the limit. Same train/test rows as make_data.py, same embeddings (finetune.py's
cache), same state text (CLM's build_pairs). It sees only the state, so like the tuned head it can only
answer with the action set it was trained on. The regularization is picked by 5-fold cross-validation on
the training rows; the test split picks nothing.

Run: python3 logreg_baseline.py DATA_DIR EMBEDDING_CACHE.npz --clm ~/work/CLM   (numpy, pyarrow, scikit-learn)
"""
import argparse, hashlib, json, sys
import numpy as np


def load(path, clm):
    sys.path.insert(0, f"{clm}/src")
    from clm.schema import build_pairs
    import pyarrow.parquet as pq
    out = []
    for r in pq.read_table(path).to_pylist():
        (qid, (text, keys, _)), = build_pairs(json.loads(r["state"]), json.loads(r["questions"])).items()
        gold = json.loads(r["gold"])["tool"]["probabilities"]
        out.append((text, {k for k, p in gold.items() if p > 0}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data"); ap.add_argument("cache"); ap.add_argument("--clm", required=True)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    z = np.load(args.cache); vecs = dict(zip(z["keys"].tolist(), z["vecs"]))
    emb = lambda t: vecs[hashlib.sha1(t.encode()).hexdigest()].astype(np.float32)  # noqa: E731
    train, test = load(f"{args.data}/all/train.parquet", args.clm), load(f"{args.data}/all/test.parquet", args.clm)
    Xtr, Xte = np.stack([emb(t) for t, _ in train]), np.stack([emb(t) for t, _ in test])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = np.array([sorted(g)[0] for _, g in train])
    best = max(((cross_val_score(LogisticRegression(C=c, max_iter=2000), Xtr, ytr, cv=5).mean(), c) for c in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)))
    clf = LogisticRegression(C=best[1], max_iter=2000).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    ok = sum(p in g for p, (_, g) in zip(pred, test))
    print(f"train {len(train)} rows | C={best[1]} (5-fold CV accuracy {best[0]:.3f}) | test {ok}/{len(test)} = {ok/len(test):.2f}")
    print(json.dumps({"train": len(train), "C": best[1], "cv_acc": round(best[0], 4), "test_correct": int(ok), "test_n": len(test)}))


if __name__ == "__main__":
    main()
