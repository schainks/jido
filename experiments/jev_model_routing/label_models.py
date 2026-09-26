#!/usr/bin/env python3
"""Labeling run: every model config on every task, N samples each.

Each (task, config, sample) becomes one line of runs.jsonl: graded outcome, the
visible reply (tail), tool calls, tokens, cost, latency, stop reason. The file is
append-only and the run resumes: a rerun skips calls that succeeded and retries
errors. analyze.py regrades from the stored replies, so a grader fix needs no new calls.

Configs (common.CONFIGS): Haiku 4.5, Sonnet 5, Opus 5 at API defaults, plus Opus 5
at effort medium and low. The tool slice reuses the pilot's selection-only prompt
and the 19 actions as tools; the open slice asks for a final `ANSWER:` line.

Server-side refusal fallbacks are deliberately not enabled: a fallback answers with
a different model, and that answer would be credited to the tier being measured.
A refusal is recorded as its own stop reason and graded as a miss.

Run: uv venv && uv pip install anthropic
     .venv/bin/python label_models.py --plan     # call count and rough cost, no API calls
     .venv/bin/python label_models.py --smoke    # one tool and one open task per config
     .venv/bin/python label_models.py            # everything; safe to interrupt and rerun
Key: ANTHROPIC_API_KEY or ~/.anthropic_key (else ANTHROPIC_AUTH_TOKEN or `ant auth login`). Never printed.
"""
import argparse, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from common import CONFIGS, HERE, PRICE, anthropic_client, api_errors, append_jsonl, cost_usd, read_jsonl, usage_dict
from tasks import OPEN_SYSTEM, SELECT, grade, load_tasks, tools_from_actions

MAX_TOKENS = {"tool": 4096, "open": 16000}
TEXT_KEEP = 4000  # chars of visible reply kept for regrading; the ANSWER line is at the end

# Rough (input, output) tokens per call, only for --plan. Tool slice from the pilot's
# means (Haiku 1,920/79, Opus 2,206/103); open-slice outputs are guesses.
EST_TOKENS = {
    "tool": {"haiku": (1920, 80), "sonnet": (2100, 150), "opus": (2210, 110), "opus-medium": (2210, 90), "opus-low": (2210, 70)},
    "open": {"haiku": (300, 500), "sonnet": (300, 1500), "opus": (300, 2000), "opus-medium": (300, 1300), "opus-low": (300, 700)},
}


def build_request(task, config, tools):
    c = CONFIGS[config]
    req = {"model": c["model"], "max_tokens": MAX_TOKENS[task["slice"]],
           "messages": [{"role": "user", "content": task["prompt"]}]}
    if task["slice"] == "tool":
        req["system"] = SELECT
        req["tools"] = tools
    else:
        req["system"] = OPEN_SYSTEM
    if c["effort"]:
        req["output_config"] = {"effort": c["effort"]}
    return req


def run_one(create, task, config, sample, tools, errors):
    req = build_request(task, config, tools)
    row = {"task": task["id"], "slice": task["slice"], "config": config, "sample": sample,
           "model": req["model"], "effort": CONFIGS[config]["effort"], "ts": round(time.time(), 3)}
    t0 = time.perf_counter()
    try:
        r = create(**req)
    except errors as e:  # after the SDK's own retries; not a label, a rerun retries it
        return {**row, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                "latency_s": time.perf_counter() - t0}
    dt = time.perf_counter() - t0
    calls = [b.name for b in r.content if b.type == "tool_use"]
    text = "\n".join(b.text for b in r.content if b.type == "text")
    usage = usage_dict(r.usage)
    passed, got = grade(task, calls, text)
    if r.stop_reason in ("refusal", "max_tokens"):
        passed = False  # an incomplete reply is not a pass, even if its last line parses
    return {**row, "ok": True, "passed": passed, "got": got, "calls": calls, "text": text[-TEXT_KEEP:],
            "stop_reason": r.stop_reason, "usage": usage, "cost_usd": cost_usd(req["model"], usage),
            "latency_s": dt, "response_model": getattr(r, "model", None),
            "request_id": getattr(r, "_request_id", None)}


def plan(tasks, configs, samples, done):
    total = 0.0
    print(f"{'config':12s} {'calls':>6s} {'est. $':>8s}")
    for c in configs:
        n, usd = 0, 0.0
        for t in tasks:
            todo = sum((t["id"], c, s) not in done for s in range(samples))
            tin, tout = EST_TOKENS[t["slice"]][c]
            pin, pout = PRICE[CONFIGS[c]["model"]]
            n += todo
            usd += todo * (tin * pin + tout * pout) / 1e6
        total += usd
        print(f"{c:12s} {n:6d} {usd:8.2f}")
    print(f"{'total':12s} {'':6s} {total:8.2f}   (rough: open-slice output lengths are guesses; thinking can run longer)")


def run(tasks, acts, configs, samples, out, create, errors, workers=4, log=print):
    path = Path(out) / "runs.jsonl"
    done = {(r["task"], r["config"], r["sample"]) for r in read_jsonl(path) if r.get("ok")}
    tools = tools_from_actions(acts)
    jobs = [(t, c, s) for t in tasks for c in configs for s in range(samples) if (t["id"], c, s) not in done]
    log(f"{len(jobs)} calls to make ({len(done)} already done) -> {path}")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, create, t, c, s, tools, errors) for t, c, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            append_jsonl(path, row)
            rows.append(row)
            if not row["ok"]:
                log(f"  ERROR {row['task']} {row['config']}#{row['sample']}: {row['error']}")
            if i % 25 == 0 or i == len(futs):
                spent = sum(r.get("cost_usd", 0) for r in rows)
                log(f"  {i}/{len(futs)} done, ${spent:.2f} this run")
    for c in configs:
        mine = [r for r in rows if r["config"] == c and r["ok"]]
        if mine:
            log(f"== {c}: passed {sum(r['passed'] for r in mine)}/{len(mine)}, "
                f"median {1000 * statistics.median(r['latency_s'] for r in mine):.0f} ms, "
                f"${sum(r['cost_usd'] for r in mine):.2f}")
    return rows


def select_tasks(tasks, spec):
    if not spec:
        return tasks
    want = set(spec.split(","))
    return [t for t in tasks if t["id"] in want or t["slice"] in want]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset of " + ",".join(CONFIGS))
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tasks", default="", help="task ids and/or slices (tool, open), comma-separated")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--plan", action="store_true", help="print call counts and a rough cost, make no calls")
    ap.add_argument("--smoke", action="store_true", help="one sample of the first tool and first open task per config")
    args = ap.parse_args(argv)

    configs = [c for c in args.configs.split(",") if c]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"unknown configs: {unknown}")
    tasks, acts = load_tasks()
    tasks = select_tasks(tasks, args.tasks)
    samples = args.samples
    if args.smoke:
        tasks = [next(t for t in tasks if t["slice"] == s) for s in ("tool", "open") if any(t["slice"] == s for t in tasks)]
        samples = 1
    if args.plan:
        done = {(r["task"], r["config"], r["sample"]) for r in read_jsonl(Path(args.out) / "runs.jsonl") if r.get("ok")}
        return plan(tasks, configs, samples, done)
    client = anthropic_client()
    rows = run(tasks, acts, configs, samples, args.out, client.messages.create, api_errors(), args.workers)
    if args.smoke:
        for r in rows:
            status = f"stop={r['stop_reason']} passed={r['passed']} out_tokens={r['usage']['output_tokens']}" if r["ok"] else r["error"]
            print(f"  {r['config']:12s} {r['task']:18s} {status}")


if __name__ == "__main__":
    main()
