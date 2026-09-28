#!/usr/bin/env python3
"""Ask Jev which tier each task needs: one request per task, four questions.

  tier              (choice) - fast / capable / reasoning (common.TIER_CRITERIA), with probabilities
  difficulty        (score)  - how much careful reasoning a correct answer takes
  risk              (score)  - the blast-radius question from ../jev_routing/jev_eval.py, unchanged
  implicit_context  (noul)   - does the request lean on something it does not state

Jev only sees the request: the same `state` shape as the tool experiment (request,
available actions, context). It never sees a model's answer. Output: jev_routes.jsonl,
one line per task; a rerun skips tasks that already have an answer.

Run: python3 route_jev.py    (stdlib only)
Key: TYPESAFE_API_KEY or ~/.typesafe_key, read by jev_eval.py. Never printed.
"""
import argparse, time, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import HERE, TIER_CRITERIA, append_jsonl, read_jsonl  # first: puts ../jev_routing on sys.path
import jev_eval
from tasks import load_tasks


def questions():
    return {
        "tier": {"type": "choice",
                 "instructions": "What is the least capable model tier that can fully and correctly handle `request`?",
                 "criteria": dict(TIER_CRITERIA)},
        "difficulty": {"type": "score",
                       "instructions": "How much careful reasoning does a correct answer to `request` take?",
                       "criteria": ["None, or one obvious step",
                                    "A few steps of reasoning or arithmetic",
                                    "Long, exact, multi-step reasoning or a search over many possibilities"]},
        "risk": jev_eval.questions([], "discovery")["risk"],
        "implicit_context": {"type": "noul",
                             "instructions": "Does `request` refer to something it does not state, such as 'this signal', "
                                             "'the child you spawned' or 'whoever asked', that must be resolved from context?",
                             "criteria": {"true": "The request depends on a target, value or referent that is not in the request itself",
                                          "false": "Everything needed to act on the request is stated in it"}},
    }


def route_one(call, task):
    row = {"task": task["id"], "ts": round(time.time(), 3)}
    try:
        resp, dt = call({"state": task["jev_state"], "model": jev_eval.MODEL, "questions": questions()})
    except (urllib.error.URLError, OSError, ValueError) as e:  # HTTP errors after retries, timeouts, bad JSON
        return {**row, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    try:
        a = resp["answers"]
        return {**row, "ok": True, "tier": a["tier"]["choice"], "tier_confidence": a["tier"]["confidence"],
                "tier_probs": a["tier"]["probabilities"], "difficulty": a["difficulty"]["score"],
                "difficulty_confidence": a["difficulty"].get("confidence"), "risk": a["risk"]["score"],
                "implicit_context": a["implicit_context"]["noul"], "latency_s": dt,
                "usage": resp.get("usage"), "model": resp.get("model")}
    except (KeyError, TypeError) as e:
        keys = sorted(resp) if isinstance(resp, dict) else type(resp).__name__
        return {**row, "ok": False, "error": f"unexpected response shape ({e!r}); top-level keys: {keys}"}


def run(tasks, out, call, workers=4, log=print):
    path = Path(out) / "jev_routes.jsonl"
    done = {r["task"] for r in read_jsonl(path) if r.get("ok")}
    todo = [t for t in tasks if t["id"] not in done]
    log(f"{len(todo)} tasks to route ({len(done)} already done) -> {path}")

    def one(t):
        row = route_one(call, t)
        append_jsonl(path, row)
        if not row["ok"]:
            log(f"  ERROR {t['id']}: {row['error']}")
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, todo))
    ok = [r for r in rows if r["ok"]]
    if ok:
        tiers = {tier: sum(r["tier"] == tier for r in ok) for tier in TIER_CRITERIA}
        log(f"== routed {len(ok)}: {tiers}, median {1000 * sorted(r['latency_s'] for r in ok)[len(ok) // 2]:.0f} ms")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    jev_eval.key()  # exits with a message if no key, before any thread starts
    tasks, _ = load_tasks()
    run(tasks, args.out, jev_eval.call, args.workers)


if __name__ == "__main__":
    main()
