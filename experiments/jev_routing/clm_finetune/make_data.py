#!/usr/bin/env python3
"""Build the typed-decisions dataset for fine-tuning CLM's head on Jido tool selection.

Writes OUT/all/train.parquet (train_requests.py) and OUT/all/test.parquet (the 34 benchmark
requests from jev_eval.GOLD), the layout `train/finetune.py --task choice --data OUT` reads.
Rows use exactly what jev_eval.py sends: the same `state` dict, the same choice question and
the same action descriptions as criteria, so the head is trained and tested on the input
format Jev sees. Stdlib plus pyarrow.

Refuses to build if a training request is close to a benchmark request (token overlap or
sequence similarity), so the test split stays unseen.

Run: python3 make_data.py OUT [PER_ACTION]
     PER_ACTION keeps the first N requests of each action (and N/20 of the "none" requests)
     for a learning curve; the default keeps all of them.
"""
import difflib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jev_eval as J  # noqa: E402
from train_requests import REQUESTS  # noqa: E402

MAX_JACCARD, MAX_RATIO = 0.6, 0.8


def words(s):
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def too_close(a, b):
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) >= MAX_JACCARD or difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() >= MAX_RATIO


def main(out, per_action=None):
    acts = J.extract_actions()
    names = {a["name"] for a in acts} | {"none"}
    assert set(REQUESTS) == names, f"actions differ: {set(REQUESTS) ^ names}"
    tests = [q for q, _, _ in J.GOLD]
    dupes = [(t, q) for reqs in REQUESTS.values() for t in reqs for q in tests if too_close(t, q)]
    flat = [t for reqs in REQUESTS.values() for t in reqs]
    assert len(flat) == len(set(flat)), "duplicate training requests"
    if dupes:
        sys.exit("training requests too close to benchmark requests:\n" + "\n".join(f"  {t!r} ~ {q!r}" for t, q in dupes))
    best = max((difflib.SequenceMatcher(None, t.lower(), q.lower()).ratio(), t, q) for t in flat for q in tests)
    print(f"{len(flat)} training requests, {len(tests)} test requests; closest pair {best[0]:.2f}: {best[1]!r} ~ {best[2]!r}")

    question = {"tool": {"type": "choice", "instructions": J.questions(acts, "discovery")["tool"]["instructions"],
                         "criteria": J.criteria(acts, "discovery")}}

    def row(rid, query, gold):
        p = {g: 1 / len(gold) for g in gold}
        return {"id": rid, "workflow": "all", "state": json.dumps(J.state(acts, query)), "questions": json.dumps(question),
                "gold": json.dumps({"tool": {"label": sorted(gold)[0], "probabilities": p}})}

    def keep(a, reqs):
        n = len(reqs) if not per_action else -(-len(reqs) * per_action // 20) if a == "none" else per_action
        return reqs[:n]
    train = [row(f"train-{a}-{i}", q, {a}) for a, reqs in REQUESTS.items() for i, q in enumerate(keep(a, reqs))]
    test = [row(f"test-{i}", q, ok) for i, (q, ok, _) in enumerate(J.GOLD)]
    import pyarrow as pa, pyarrow.parquet as pq
    d = Path(out) / "all"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), d / "train.parquet")
    pq.write_table(pa.Table.from_pylist(test), d / "test.parquet")
    print(f"wrote {d}/train.parquet ({len(train)} rows) and test.parquet ({len(test)} rows)")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) == 3 else None)
