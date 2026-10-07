#!/usr/bin/env python3
"""Ask Jev which tool the assistant called next, for a random sample of prepare.py's rows, into Jevstiller's answer cache.

This sends scrubbed context text to TypeSafe's hosted Jev API: --n rows, one call each, each the text prepare.py built
(task, latest user message, previous steps, last result, optionally the assistant's narration) plus the class descriptions.
Only aggregates are printed. The key comes from TYPESAFE_API_KEY or ~/.typesafe_key and is never printed.

Run: python jev_label.py PREPARED.jsonl --cache CACHE.jsonl --split test --n 300 [--narrated] [--seed 0]     (pip install jevstiller)
"""
import argparse, collections, json, os, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import CLASSES, INSTRUCTIONS  # noqa: E402


REQUEST_INSTRUCTIONS = "Which tool will the assistant reach for first to start working on this request?"


def make_task(target=0.98, request_mode=False):
    from jevstiller import Task
    return Task(name="first_tool" if request_mode else "next_tool", instructions=REQUEST_INSTRUCTIONS if request_mode else INSTRUCTIONS, classes=CLASSES, target_agreement=target)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prepared"); ap.add_argument("--cache", required=True); ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--request-mode", action="store_true", help="only calls that follow a user message; the text is the request alone");
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--narrated", action="store_true")
    ap.add_argument("--model", default="jev-1.13.0")
    a = ap.parse_args()
    from jevstiller.teachers import CachedTeacher
    from jevstiller.teachers.jev import JevTeacher
    key_file = Path.home() / ".typesafe_key"
    key = os.environ.get("TYPESAFE_API_KEY") or (key_file.read_text().strip() if key_file.exists() else None)
    if not key:
        sys.exit("no key: set TYPESAFE_API_KEY or write it to ~/.typesafe_key")
    field = "text_request" if a.request_mode else "text_narrated" if a.narrated else "text"
    rows = [json.loads(l) for l in open(a.prepared)]
    pool = [r for r in rows if (a.split == "all" or r["split"] == a.split) and (r["first_after_user"] or not a.request_mode)]
    sample = random.Random(a.seed).sample(pool, min(a.n, len(pool)))
    task = make_task(request_mode=a.request_mode)
    teacher = CachedTeacher(JevTeacher(model=a.model, api_key=key), a.cache)
    outs, errors = {}, 0
    for i in range(0, len(sample), 100):
        chunk = sample[i:i + 100]
        for r, o in zip(chunk, teacher.classify([r[field] for r in chunk], task)):
            if isinstance(o, Exception):
                errors += 1
            else:
                outs[r["id"]] = o
        print(f"{min(i + 100, len(sample)):>6}/{len(sample)}  calls {teacher.misses}  cache hits {teacher.hits}  errors {errors}", flush=True)
    if errors:
        print(f"{errors} answers failed; rerun to fetch them")
    truth = {r["id"]: r["label"].replace(":", "_") for r in sample}
    ok = [i for i in outs if outs[i].label == truth[i]]
    top3 = [i for i in outs if truth[i] in sorted(outs[i].probs, key=outs[i].probs.get, reverse=True)[:3]]
    print(f"Jev on {len(outs)} {a.split} rows ({'with' if a.narrated else 'without'} narration): top-1 {len(ok) / len(outs):.1%}, top-3 {len(top3) / len(outs):.1%}")
    by = collections.defaultdict(lambda: [0, 0])
    for i, o in outs.items():
        by[truth[i]][1] += 1; by[truth[i]][0] += o.label == truth[i]
    print("recall by class (rows >= 10):", {c: f"{h}/{n}" for c, (h, n) in sorted(by.items(), key=lambda kv: -kv[1][1]) if n >= 10})
    lat = [o.latency_ms for o in outs.values() if o.latency_ms]
    print(f"cost ${sum(o.cost_usd for o in outs.values()):.4f} for {len(outs)} calls, median latency {statistics.median(lat):.0f} ms; cache {a.cache}")
    Path(a.cache).chmod(0o600)


if __name__ == "__main__":
    main()
