#!/usr/bin/env python3
"""Cheap-alternative baseline for tool selection: embeddings and BM25.

Same 34 queries and 19 Jido actions as jev_eval.py. Each action becomes one
document ("name: description"); the request is matched by cosine similarity
(local ONNX models via fastembed, no API) or BM25 (lexical). Top-1 is the pick.

Because nearest-neighbour search has no native "none" answer, none-detection is
reported at the best possible similarity threshold (an upper bound that favours
the baseline). The margin between the top two scores stands in for confidence
so precision/coverage can be compared with Jev's confidence table.

Run: .venv/bin/python embed_baseline.py   (downloads ~100-400 MB of models on first run)
"""
import json, re, statistics, time
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from jev_eval import extract_actions, GOLD, OUT

MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]

def doc_text(a):
    return f"{a['name'].replace('_', ' ')}: {a['description']}"

def tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower().replace("_", " "))

def evaluate(name, scores, acts, latency_ms):
    """scores: (n_queries, n_actions) similarity matrix."""
    names = [a["name"] for a in acts]
    rows = []
    for (q, ok, needs), sc in zip(GOLD, scores):
        order = np.argsort(-sc)
        top, second = names[order[0]], names[order[1]]
        rows.append({"query": q, "gold": sorted(ok), "needs_tool": needs, "top": top, "second": second,
                     "top_score": float(sc[order[0]]), "margin": float(sc[order[0]] - sc[order[1]]),
                     "correct_if_tool": top in ok})
    tool_rows = [r for r in rows if r["needs_tool"]]
    none_rows = [r for r in rows if not r["needs_tool"]]
    tool_acc = sum(r["correct_if_tool"] for r in tool_rows) / len(tool_rows)
    # best-case none detection: pick the top_score threshold that maximises overall accuracy
    best = None
    for thr in sorted({r["top_score"] for r in rows}) + [float("inf")]:
        acc = sum((r["top_score"] < thr) if not r["needs_tool"] else (r["top_score"] >= thr and r["correct_if_tool"]) for r in rows) / len(rows)
        if best is None or acc > best[1]: best = (thr, acc)
    # precision/coverage by margin, at the coverage levels Jev reached (~0.97, 0.91, 0.79, 0.71, 0.65)
    ranked = sorted(tool_rows, key=lambda r: -r["margin"])
    pc = []
    for cov in (0.97, 0.91, 0.79, 0.71, 0.65):
        k = max(1, round(cov * len(ranked))); acted = ranked[:k]
        pc.append({"coverage": round(k / len(ranked), 2), "precision": round(sum(r["correct_if_tool"] for r in acted) / k, 2),
                   "margin_cut": round(acted[-1]["margin"], 3)})
    misses = [{"q": r["query"], "gold": r["gold"], "got": r["top"], "second": r["second"], "margin": round(r["margin"], 3)}
              for r in tool_rows if not r["correct_if_tool"]]
    return {"model": name, "tool_accuracy": tool_acc, "tool_n": len(tool_rows),
            "best_case_overall_accuracy": best[1], "best_threshold": best[0],
            "precision_by_coverage": pc, "latency_ms_per_query": latency_ms, "misses": misses, "rows": rows}

if __name__ == "__main__":
    acts = extract_actions(); docs = [doc_text(a) for a in acts]; queries = [g[0] for g in GOLD]
    results = []
    # BM25
    bm = BM25Okapi([tokens(d) for d in docs])
    t0 = time.perf_counter(); sc = np.array([bm.get_scores(tokens(q)) for q in queries]); dt = (time.perf_counter() - t0) / len(queries) * 1000
    results.append(evaluate("bm25", sc, acts, dt))
    # embeddings
    for m in MODELS:
        emb = TextEmbedding(model_name=m)
        D = np.array(list(emb.embed(docs)))
        t0 = time.perf_counter(); Q = np.array(list(emb.query_embed(queries))); dt = (time.perf_counter() - t0) / len(queries) * 1000
        D /= np.linalg.norm(D, axis=1, keepdims=True); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        results.append(evaluate(m, Q @ D.T, acts, dt))
    for r in results:
        print(f"\n== {r['model']}: tool-query accuracy {r['tool_accuracy']:.2f} ({int(r['tool_accuracy']*r['tool_n'])}/{r['tool_n']}), "
              f"best-case overall incl. none {r['best_case_overall_accuracy']:.2f}, {r['latency_ms_per_query']:.1f} ms/query")
        print("   precision@coverage:", [(p['coverage'], p['precision']) for p in r["precision_by_coverage"]])
        for m in r["misses"]: print("   MISS", m)
    (OUT / "embed_baseline_results.json").write_text(json.dumps(results, indent=1))
