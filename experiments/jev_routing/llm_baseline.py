#!/usr/bin/env python3
"""Baseline for jev_eval.py: what jido_ai does today.

Same 19 Jido actions, same 34 queries, but the LLM picks the tool via native
tool-calling (tool_choice auto), which is how jido_ai's ReAct loop works.
Default model list mirrors jido_ai's aliases: :fast = claude-haiku-4-5, plus
claude-opus-5 as the capable reference. Correct = called a gold tool, or made
no tool call when gold is "none". Reads the key from ANTHROPIC_API_KEY or
~/.anthropic_key; never prints it.
Run: uv venv && uv pip install anthropic && .venv/bin/python llm_baseline.py
     COND=select .venv/bin/python llm_baseline.py   # selection-only prompt
"""
import json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anthropic
from jev_eval import extract_actions, GOLD, OUT

MODELS = sys.argv[1:] or ["claude-haiku-4-5", "claude-opus-5"]
COND = os.environ.get("COND", "default")
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0)}  # $/MTok in, out
SYSTEM = ("You are a Jido agent runtime assistant. You have tools that correspond to the agent's "
          "available actions. If the request maps to one of your tools, call it. If no tool fits the "
          "request, do not call any tool; reply briefly in text instead.")

SELECT = ("You are the tool-selection step of a Jido agent. Your only job is to decide which one of your tools, "
          "if any, the request maps to, and call it. Missing parameters are fine: call the tool with whatever "
          "parameters you can infer and leave the rest out; do not ask clarifying questions. Only if no tool fits "
          "the request at all, reply in text without calling a tool.")

def tools_from_actions(acts):
    out = []
    for a in acts:
        props = {k: {"type": "string", "description": d or k} for k, d in a["params"]}
        out.append({"name": a["name"], "description": a["description"],
                    "input_schema": {"type": "object", "properties": props}})
    return out

def key():
    p = Path.home() / ".anthropic_key"
    return os.environ.get("ANTHROPIC_API_KEY") or (p.read_text().strip() if p.exists() else None)

def run_model(client, model, tools, workers=4):
    def one(g):
        query, ok, needs = g
        t0 = time.perf_counter()
        r = client.messages.create(model=model, max_tokens=512, system=(SELECT if COND == "select" else SYSTEM), tools=tools,
                                   messages=[{"role": "user", "content": query}])
        dt = time.perf_counter() - t0
        calls = [b.name for b in r.content if b.type == "tool_use"]
        text = " ".join(b.text for b in r.content if b.type == "text")[:160]
        got = calls[0] if calls else "none"
        pin, pout = PRICE.get(model, (0, 0))
        return {"query": query, "gold": sorted(ok), "needs_tool_gold": needs, "choice": got, "all_calls": calls,
                "correct": got in ok, "text": text, "stop_reason": r.stop_reason, "latency_s": dt,
                "usage": {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens},
                "cost_usd": (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, GOLD))

def summarize(rows):
    n = len(rows); lat = sorted(r["latency_s"] for r in rows)
    none_rows = [r for r in rows if not r["needs_tool_gold"]]
    return {"n": n, "accuracy": sum(r["correct"] for r in rows) / n,
            "none_correct": sum(r["choice"] == "none" for r in none_rows) / len(none_rows),
            "latency_ms": {"median": 1000 * statistics.median(lat), "p95": 1000 * lat[int(0.95 * (n - 1))]},
            "tokens_in_mean": statistics.mean(r["usage"]["input_tokens"] for r in rows),
            "tokens_out_mean": statistics.mean(r["usage"]["output_tokens"] for r in rows),
            "cost_per_call_usd": statistics.mean(r["cost_usd"] for r in rows),
            "misses": [{"q": r["query"], "gold": r["gold"], "got": r["choice"], "text": r["text"]} for r in rows if not r["correct"]]}

if __name__ == "__main__":
    k = key()
    if not k:
        sys.exit("no key: put it in ~/.anthropic_key or ANTHROPIC_API_KEY")
    ws = Path.home() / ".anthropic_workspace"
    headers = {"anthropic-workspace-id": ws.read_text().strip()} if ws.exists() else {}
    client = anthropic.Anthropic(api_key=k, default_headers=headers)
    acts = extract_actions(); tools = tools_from_actions(acts)
    results = {}
    for model in MODELS:
        rows = run_model(client, model, tools)
        s = summarize(rows); results[model] = {"summary": s, "rows": rows}
        print(f"\n== {model}: acc={s['accuracy']:.2f} none={s['none_correct']:.2f} median={s['latency_ms']['median']:.0f}ms "
              f"p95={s['latency_ms']['p95']:.0f}ms in~{s['tokens_in_mean']:.0f} out~{s['tokens_out_mean']:.0f} "
              f"${s['cost_per_call_usd']*1000:.2f}/1k calls")
        for m in s["misses"]: print("   MISS", m)
    (OUT / f"llm_baseline_{COND}.json").write_text(json.dumps(results, indent=1))
