#!/usr/bin/env python3
"""Jevstiller's student on Jido tool selection: its encoders and its LinearStudent, trained on gold labels.

Jevstiller (github.com/tomerglick57/Jevstiller) sits in front of Jev's Choice calls and distills Jev's
answers into a local student: a frozen sentence encoder (bge-small by default) plus a numpy logistic
regression trained on Jev's probability distributions, with an out-of-distribution gate, a permanent
audit slice and a calibrated routing threshold. This script tests the student half alone, with gold
labels standing in for a perfect teacher (Jev's own answers on the training requests would need a key):
same rows as make_data.py, same test sets (the 34 benchmark requests and holdout.py's 45), three seeds.

  --encoder small|base   Jevstiller's ONNX bge encoders, on CPU
  --encoder qwen         Qwen3-8B through a vLLM /v1/embeddings server, for comparison
  --input request        embed the request text only
  --input state          embed Jevstiller's own state_text() of the whole state, as its proxy does

Run: jevstiller_student.py DATA_DIR HOLDOUT_DIR --encoder small --input request     (pip install jevstiller pyarrow requests)
     DATA_DIR is make_data.py's 40-per-action output; HOLDOUT_DIR is holdout.py's output on top of it.
"""
import argparse, base64, collections, json
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data"); ap.add_argument("holdout"); ap.add_argument("--encoder", default="small"); ap.add_argument("--input", default="request", choices=["request", "state"])
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/embeddings")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from jevstiller._student import LinearStudent
    from jevstiller._task import state_text

    def rows(path):
        out = []
        for r in pq.read_table(path).to_pylist():
            state = json.loads(r["state"]); g = json.loads(r["gold"])["tool"]["probabilities"]
            out.append((state["request"] if a.input == "request" else state_text(state), {k for k, p in g.items() if p > 0}))
        return out
    train, bench, hold = rows(f"{a.data}/all/train.parquet"), rows(f"{a.data}/all/test.parquet"), rows(f"{a.holdout}/all/test.parquet")
    classes = sorted({sorted(g)[0] for _, g in train})
    if a.encoder == "qwen":
        import requests

        def encode(texts):
            out = []
            for i in range(0, len(texts), 16):
                r = requests.post(a.url, json={"model": "qwen3-8b", "input": texts[i:i + 16], "encoding_format": "base64"}, timeout=600).json()
                for d in sorted(r["data"], key=lambda d: d["index"]):
                    v = np.frombuffer(base64.b64decode(d["embedding"]), dtype=np.float32); out.append(v / np.linalg.norm(v))
            return np.stack(out)
    else:
        from jevstiller import load_encoder
        encode = load_encoder(a.encoder).encode
    X, Xb, Xh = encode([q for q, _ in train]), encode([q for q, _ in bench]), encode([q for q, _ in hold])
    print(f"encoder {a.encoder}, input {a.input}, dim {X.shape[1]}, {len(classes)} classes")

    def first(n):  # the first n requests per action, as make_data.py's PER_ACTION does; "none" scales as 57 per 20
        c, keep = collections.Counter(), []
        for i, (_, g) in enumerate(train):
            act = sorted(g)[0]; c[act] += 1
            if c[act] <= (n if act != "none" else -(-57 * n // 20)):
                keep.append(i)
        return keep
    for n in (5, 10, 20, 40):
        idx = first(n); Y = np.eye(len(classes))[[classes.index(sorted(train[i][1])[0]) for i in idx]]; res = []
        for seed in (0, 1, 2):
            st = LinearStudent(X.shape[1], len(classes)); st.fit(X[idx], Y, seed=seed)
            score = lambda Xe, ds: sum(classes[int((Xe[i] @ st.W + st.b).argmax())] in g for i, (_, g) in enumerate(ds))  # noqa: E731
            res.append((score(Xb, bench), score(Xh, hold)))
        print(f"  {n:2d}/action ({len(idx)} rows): benchmark {[r[0] for r in res]} of {len(bench)} | holdout {[r[1] for r in res]} of {len(hold)}")


if __name__ == "__main__":
    main()
