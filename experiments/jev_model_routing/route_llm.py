#!/usr/bin/env python3
"""Baseline router: ask a small model which tier each task needs.

The counterpart of the tool experiment's native tool-calling baseline. Same tier
descriptions and the same request information Jev gets; one short call per task.
Output: llm_routes.jsonl, one line per task; a rerun skips tasks already routed.

Run: .venv/bin/python route_llm.py [--model claude-haiku-4-5]
Key: as label_models.py. Never printed.
"""
import argparse, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import HERE, PRICE, TIER_CRITERIA, anthropic_client, api_errors, append_jsonl, cost_usd, read_jsonl, usage_dict
from tasks import load_tasks

ROUTER_SYSTEM = (
    "You route requests to one of three model tiers. Pick the least capable tier that can fully and correctly "
    "handle the request.\n"
    + "".join(f"{tier}: {desc}\n" for tier, desc in TIER_CRITERIA.items())
    + "Reply with exactly one word: fast, capable or reasoning."
)


def user_message(task):
    st = task["jev_state"]
    lines = [f"Context: {st['context']}"]
    if st["available_actions"]:
        lines.append("Available actions: " + ", ".join(st["available_actions"]))
    lines.append(f"Request: {st['request']}")
    return "\n".join(lines)


def route_one(create, task, model, errors):
    # Haiku 4.5 answers in a few tokens; the headroom is for Sonnet 5 / Opus 5, which think by default.
    req = {"model": model, "max_tokens": 2048, "system": ROUTER_SYSTEM,
           "messages": [{"role": "user", "content": user_message(task)}]}
    row = {"task": task["id"], "router_model": model, "ts": round(time.time(), 3)}
    t0 = time.perf_counter()
    try:
        r = create(**req)
    except errors as e:
        return {**row, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    dt = time.perf_counter() - t0
    text = " ".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\b(fast|capable|reasoning)\b", text, re.I)
    usage = usage_dict(r.usage)
    return {**row, "ok": True, "tier": m.group(1).lower() if m else "capable", "parsed": bool(m),
            "text": text[:200], "stop_reason": r.stop_reason, "usage": usage,
            "cost_usd": cost_usd(model, usage), "latency_s": dt}


def run(tasks, out, create, errors, model="claude-haiku-4-5", workers=4, log=print):
    path = Path(out) / "llm_routes.jsonl"
    done = {r["task"] for r in read_jsonl(path) if r.get("ok") and r.get("router_model") == model}
    todo = [t for t in tasks if t["id"] not in done]
    log(f"{len(todo)} tasks to route with {model} ({len(done)} already done) -> {path}")

    def one(t):
        row = route_one(create, t, model, errors)
        append_jsonl(path, row)
        if not row["ok"]:
            log(f"  ERROR {t['id']}: {row['error']}")
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, todo))
    ok = [r for r in rows if r["ok"]]
    if ok:
        tiers = {tier: sum(r["tier"] == tier for r in ok) for tier in TIER_CRITERIA}
        log(f"== routed {len(ok)}: {tiers}, unparsed {sum(not r['parsed'] for r in ok)}, "
            f"${sum(r['cost_usd'] for r in ok):.3f}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="claude-haiku-4-5", choices=sorted(PRICE))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    tasks, _ = load_tasks()
    run(tasks, args.out, anthropic_client().messages.create, api_errors(), args.model, args.workers)


if __name__ == "__main__":
    main()
