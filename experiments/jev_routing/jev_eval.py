#!/usr/bin/env python3
"""Can Jev pick the right Jido action from Jido's own action metadata?

Candidates are extracted straight from lib/jido/actions/*.ex (name, description,
schema param names + docs), i.e. the same metadata Jido.Discovery exposes.
Each query is sent once per variant with four questions in a single request:
  needs_tool  (noul)   - does the request call for one of these actions at all
  tool        (choice) - which action, or "none"
  complexity  (score)  - 3 levels
  risk        (score)  - 3 levels
Run: python3 jev_eval.py   (stdlib only)
Results: accuracy per variant, accuracy by confidence band, coverage/precision
per threshold, none-detection, latency, token usage. Key: TYPESAFE_API_KEY env var or ~/.typesafe_key. Nothing here prints the key.
"""
import json, os, re, statistics, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # repo root
OUT = Path(__file__).parent
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
def key():
    p = Path.home() / ".typesafe_key"
    k = os.environ.get("TYPESAFE_API_KEY") or (p.read_text().strip() if p.exists() else None)
    if not k:
        sys.exit("no key: set TYPESAFE_API_KEY or write it to ~/.typesafe_key")
    return k

# ---------------------------------------------------------------- candidates
ACTION_RE = re.compile(
    r'use Jido\.Action,\s*name:\s*"(?P<name>[^"]+)",\s*description:\s*"(?P<desc>[^"]+)"'
    r'(?:,\s*schema:\s*\[(?P<schema>.*?)\]\s*\n)?', re.S)
PARAM_RE = re.compile(r'^\s*(?P<key>[a-z_]+):\s*\[(?P<spec>[^\]]*)\]', re.M)
DOC_RE = re.compile(r'doc:\s*"([^"]*)"')

def extract_actions():
    acts = []
    for f in sorted((REPO / "lib/jido/actions").glob("*.ex")):
        for m in ACTION_RE.finditer(f.read_text()):
            params = []
            for p in PARAM_RE.finditer(m.group("schema") or ""):
                d = DOC_RE.search(p.group("spec"))
                params.append((p.group("key"), d.group(1) if d else ""))
            acts.append({"name": m.group("name"), "description": m.group("desc"),
                         "params": params, "file": f.name})
    return acts

# ---------------------------------------------------------------- gold set
# (query, acceptable tools, needs_tool)  -- "none" means no listed action fits
GOLD = [
    ("Shut yourself down gracefully.", {"stop_self"}, True),
    ("Terminate the worker child you spawned earlier.", {"stop_child"}, True),
    ("Start a new child agent to handle the import job.", {"spawn_child"}, True),
    ("Let your parent know the batch finished.", {"notify_parent"}, True),
    ("Send this event to process <0.123.0>.", {"notify_pid"}, True),
    ("Pass this signal along to the billing agent.", {"forward"}, True),
    ("Announce the deploy to everyone subscribed on the topic.", {"broadcast"}, True),
    ("Answer the caller who sent this request.", {"reply"}, True),
    ("Abort the current operation, the user cancelled.", {"cancel"}, True),
    ("Ignore this signal, there is nothing to do.", {"noop"}, True),
    ("Remind yourself in 5 minutes to check the queue.", {"schedule_signal"}, True),
    ("If no response arrives within 30 seconds, treat it as a deadline miss.", {"schedule_timeout"}, True),
    ("Run the cleanup every night at midnight.", {"schedule_cron"}, True),
    ("Stop the nightly cleanup job.", {"cancel_cron"}, True),
    ("Set your status to 'paused'.", {"set_status"}, True),
    ("You're done, record the final result.", {"mark_completed"}, True),
    ("This failed because the API returned 500, record that.", {"mark_failed"}, True),
    ("Flag that you're busy processing now.", {"mark_working"}, True),
    ("You're free now, mark yourself available.", {"mark_idle"}, True),
    ("Tell the parent we hit an unrecoverable error.", {"notify_parent", "mark_failed"}, True),
    ("Every Monday at 9am send the report signal.", {"schedule_cron"}, True),
    ("Give up on waiting for the reply after two minutes.", {"schedule_timeout"}, True),
    ("Relay this to the agent named 'audit'.", {"forward"}, True),
    ("Reply to whoever asked with the totals.", {"reply"}, True),
    ("Cancel the recurring health check.", {"cancel_cron"}, True),
    ("Spin up three workers for the crawl.", {"spawn_child"}, True),
    ("Stop being idle, get to work.", {"mark_working"}, True),
    ("Set a 10 second timer then ping yourself.", {"schedule_signal", "schedule_timeout"}, True),
    ("Kill it.", {"stop_self", "stop_child", "cancel"}, True),          # deliberately ambiguous
    ("What's the weather in Paris today?", {"none"}, False),
    ("Summarize this article for me.", {"none"}, False),
    ("Translate 'hello' to French.", {"none"}, False),
    ("Delete the user's account permanently.", {"none"}, False),        # no such tool; must not force-fit
    ("Book me a flight to Denver.", {"none"}, False),
]

# ---------------------------------------------------------------- questions
def criteria(acts, variant):
    c = {}
    for a in acts:
        if variant == "discovery" or not a["params"]:
            c[a["name"]] = a["description"]
        else:
            c[a["name"]] = {"what": a["description"],
                            "params": ", ".join(f"{k} ({d})" if d else k for k, d in a["params"])}
    c["none"] = "None of the listed actions is what the request asks for"
    return c

def questions(acts, variant):
    return {
        "needs_tool": {"type": "noul",
            "instructions": "Does `request` ask this agent to perform one of the actions listed in `available_actions`?",
            "criteria": {"true": "The request maps to one of the listed agent actions",
                         "false": "The request asks for something none of the listed actions does, such as answering a question or generating text"}},
        "tool": {"type": "choice",
            "instructions": "Which action in `available_actions` should the agent run to satisfy `request`? Pick none if no listed action fits.",
            "criteria": criteria(acts, variant)},
        "complexity": {"type": "score",
            "instructions": "How much work is needed to turn `request` into an action call?",
            "criteria": ["One obvious action with parameters stated or defaulted",
                         "One action, but a parameter must be inferred or clarified",
                         "Needs multi-step reasoning, several actions, or a language model to answer"]},
        "risk": {"type": "score",
            "instructions": "What is the blast radius if the agent runs the wrong action for `request`?",
            "criteria": ["Reads or reports state only",
                         "Changes this agent's own state, status or schedule",
                         "Affects other agents or processes, or is hard to undo"]},
    }

def state(acts, query):
    return {"request": query,
            "available_actions": [a["name"] for a in acts],
            "context": "The agent is a Jido AgentServer process with a parent, possibly child agents, and scheduled jobs."}

# ---------------------------------------------------------------- client
def call(body, retries=5):
    data = json.dumps(body).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API, data=data, method="POST", headers={
            "Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            print(f"HTTP {e.code}: {e.read()[:500]!r}", file=sys.stderr); raise

# ---------------------------------------------------------------- run
def run_variant(acts, variant, workers=1):
    qs = questions(acts, variant)
    def one(g):
        query, ok, needs = g
        resp, dt = call({"state": state(acts, query), "model": MODEL, "questions": qs})
        a = resp["answers"]
        return {"query": query, "gold": sorted(ok), "needs_tool_gold": needs,
                "choice": a["tool"]["choice"], "confidence": a["tool"]["confidence"],
                "probs": a["tool"]["probabilities"], "needs_tool": a["needs_tool"]["noul"],
                "complexity": a["complexity"]["score"], "complexity_conf": a["complexity"]["confidence"],
                "risk": a["risk"]["score"], "latency_s": dt, "usage": resp.get("usage"),
                "model": resp.get("model"), "correct": a["tool"]["choice"] in ok}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, GOLD))

def band(c):
    return "<0.6" if c < 0.6 else "0.6-0.85" if c < 0.85 else ">=0.85"

def summarize(rows):
    n = len(rows); acc = sum(r["correct"] for r in rows) / n
    bands = {}
    for r in rows:
        b = bands.setdefault(band(r["confidence"]), [0, 0]); b[0] += r["correct"]; b[1] += 1
    thresholds = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):
        acted = [r for r in rows if r["confidence"] >= t]
        thresholds.append({"t": t, "coverage": len(acted) / n,
                           "precision": (sum(r["correct"] for r in acted) / len(acted)) if acted else None})
    none_rows = [r for r in rows if not r["needs_tool_gold"]]
    tool_rows = [r for r in rows if r["needs_tool_gold"]]
    lat = sorted(r["latency_s"] for r in rows)
    return {"n": n, "accuracy": acc, "by_band": bands, "thresholds": thresholds,
            "none_detected_by_choice": sum(r["choice"] == "none" for r in none_rows) / len(none_rows),
            "none_noul_mean": statistics.mean(r["needs_tool"] for r in none_rows),
            "tool_noul_mean": statistics.mean(r["needs_tool"] for r in tool_rows),
            "latency_ms": {"median": 1000 * statistics.median(lat), "p95": 1000 * lat[int(0.95 * (n - 1))],
                           "min": 1000 * lat[0], "max": 1000 * lat[-1]},
            "tokens_in_mean": statistics.mean(r["usage"]["input_tokens"] for r in rows if r["usage"]),
            "model": rows[0]["model"],
            "misses": [{"q": r["query"], "gold": r["gold"], "got": r["choice"], "conf": round(r["confidence"], 2),
                        "top3": sorted(r["probs"].items(), key=lambda kv: -kv[1])[:3]} for r in rows if not r["correct"]]}

if __name__ == "__main__":
    acts = extract_actions()
    print(f"{len(acts)} actions extracted from lib/jido/actions/: {[a['name'] for a in acts]}")
    results = {}
    for variant in ("discovery", "schema"):
        rows = run_variant(acts, variant)
        results[variant] = {"summary": summarize(rows), "rows": rows}
        s = results[variant]["summary"]
        print(f"\n== {variant}: acc={s['accuracy']:.2f} n={s['n']} model={s['model']} "
              f"median={s['latency_ms']['median']:.0f}ms p95={s['latency_ms']['p95']:.0f}ms tokens_in~{s['tokens_in_mean']:.0f}")
        print("   by band:", s["by_band"])
        print("   thresholds:", [(t['t'], round(t['coverage'], 2), t['precision'] and round(t['precision'], 2)) for t in s["thresholds"]])
        print(f"   none: choice={s['none_detected_by_choice']:.2f} noul(none)={s['none_noul_mean']:.2f} noul(tool)={s['tool_noul_mean']:.2f}")
        for m in s["misses"]: print("   MISS", m)
    # throughput check: same requests, 8 in flight
    t0 = time.perf_counter(); rows = run_variant(acts, "schema", workers=8); wall = time.perf_counter() - t0
    results["parallel8"] = {"wall_s": wall, "n": len(rows), "rps": len(rows) / wall,
                            "median_ms": 1000 * statistics.median(r["latency_s"] for r in rows),
                            "accuracy": sum(r["correct"] for r in rows) / len(rows)}
    print("\n== parallel x8:", results["parallel8"])
    (OUT / "jev_eval_results.json").write_text(json.dumps({"actions": acts, **results}, indent=1))
