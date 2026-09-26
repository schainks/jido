#!/usr/bin/env python3
"""Labels and the router comparison, computed offline from the raw files. Stdlib only.

Label: the cheapest tier whose config passes at least 2/3 of its samples on a task,
or "unsolved" when none does. Every task ran on every config, so each router is
scored counterfactually: on each task it gets the pass rate, cost and latency of
the config its tier maps to, plus its own latency and cost. Replies are regraded
with the current graders, so fixing a grader needs no new API calls.

Run: python3 analyze.py              # README tables to stdout, summary.json next to the data
     python3 analyze.py --pilot      # the pilot table, from ../jev_routing's existing results
     python3 analyze.py --jev-cost X # assumed $ per Jev call; the API did not report one last time
"""
import argparse, json, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import CONFIGS, HERE, MAPPINGS, TIERS, TOOL_EXPERIMENT, read_jsonl
from tasks import grade, load_tasks

PASS_BAR = 2 / 3
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95)
RISK_FLOOR = 1.5  # the risk Score runs 0-2; above 1.5 is nearer "affects other agents or is hard to undo"
VARIANT_TAU = 0.8  # the threshold the floor/bump variants use, fixed before seeing results


# ---------------------------------------------------------------- jido_ai's heuristic, ported
# Jido.AI.Reasoning.Adaptive.Strategy.calculate_complexity/1 and its default thresholds
# (agentjido/jido_ai@17c10b5, lib/jido_ai/reasoning/adaptive/strategy.ex). Adaptive uses
# the score to choose a reasoning strategy for one fixed model; here the same bands pick a tier.
ADAPTIVE_COMPLEX_KEYWORDS = "analyze explore consider multiple options alternatives compare contrast evaluate".split()


def adaptive_complexity(prompt):
    lower = prompt.lower()
    length_score = min(len(re.split(r"\s+", lower)) / 100, 1.0) * 0.3
    structure_score = min((len(re.split(r"[.!?]+", prompt)) - 1) / 5, 1.0) * 0.2
    keyword_score = min(sum(kw in lower for kw in ADAPTIVE_COMPLEX_KEYWORDS) / 3, 1.0) * 0.3
    constraints = len(re.findall(r"(must|should|need to|have to|require)", prompt, re.I))
    constraint_score = min((prompt.count("?") + constraints) / 5, 1.0) * 0.2
    return min(length_score + structure_score + keyword_score + constraint_score, 1.0)


def adaptive_tier(prompt, simple=0.3, complex_=0.7):
    s = adaptive_complexity(prompt)
    return "fast" if s < simple else "reasoning" if s > complex_ else "capable"


# ---------------------------------------------------------------- Jev decision rules
def cdf_tier(probs, tau):
    """Cheapest tier whose cumulative probability reaches tau, so an unsure answer fails up."""
    cum = 0.0
    for tier in TIERS:
        cum += probs.get(tier, 0.0)
        if cum >= tau:
            return tier
    return TIERS[-1]


def at_least_capable(tier):
    return "capable" if tier == "fast" else tier


def difficulty_tier(score):
    return "fast" if score < 2 / 3 else "capable" if score < 4 / 3 else "reasoning"


# ---------------------------------------------------------------- data
def latest_ok(rows, key):
    out = {}
    for r in sorted(rows, key=lambda r: r.get("ts", 0)):
        if r.get("ok"):
            out[key(r)] = r
    return out


def regrade(runs, tasks_by_id):
    changed = 0
    for r in runs.values():
        t = tasks_by_id.get(r["task"])
        if t is None:
            continue
        passed, got = grade(t, r.get("calls") or [], r.get("text") or "")
        passed = passed and r.get("stop_reason") not in ("refusal", "max_tokens")
        changed += passed != r.get("passed")
        r["passed"], r["got"] = passed, got
    return changed


def aggregate(runs):
    cells = defaultdict(list)
    for r in runs.values():
        cells[(r["task"], r["config"])].append(r)
    return {k: {"n": len(rs), "pass": statistics.mean(r["passed"] for r in rs),
                "cost": statistics.mean(r["cost_usd"] for r in rs),
                "latency": statistics.median(r["latency_s"] for r in rs)} for k, rs in cells.items()}


def label_for(agg, task_id, mapping):
    for tier in TIERS:
        a = agg.get((task_id, mapping[tier]))
        if a and a["pass"] >= PASS_BAR - 1e-9:
            return tier
    return "unsolved"


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def p95(xs):
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))]


# ---------------------------------------------------------------- routers
def build_routers(jev, llm, jev_cost):
    """name -> (route(task) -> (tier, fell_back), overhead(task) -> (usd, seconds), kind)."""
    def fixed(tier):
        return lambda t: (tier, False)

    def from_rows(rows, pick):
        return lambda t: (pick(rows[t["id"]]), False) if t["id"] in rows else ("capable", True)

    def jev_overhead(t):
        r = jev.get(t["id"])
        return (jev_cost or 0.0, r["latency_s"]) if r else (0.0, 0.0)

    def llm_overhead(t):
        r = llm.get(t["id"])
        return (r["cost_usd"], r["latency_s"]) if r else (0.0, 0.0)

    none = lambda t: (0.0, 0.0)  # noqa: E731
    R = {f"Always {tier}": (fixed(tier), none, "fixed") for tier in TIERS}
    R["jido_ai Adaptive heuristic"] = (lambda t: (adaptive_tier(t["prompt"]), False), none, "heuristic")
    if llm:
        R["Haiku 4.5 picks the tier"] = (from_rows(llm, lambda r: r["tier"]), llm_overhead, "llm")
    if jev:
        R["Jev choice (argmax)"] = (from_rows(jev, lambda r: r["tier"]), jev_overhead, "jev")
        for tau in TAUS:
            R[f"Jev cumulative >= {tau}"] = (from_rows(jev, lambda r, tau=tau: cdf_tier(r["tier_probs"], tau)), jev_overhead, "jev")
        R[f"Jev cumulative >= {VARIANT_TAU}, risk floor"] = (from_rows(jev, lambda r: at_least_capable(cdf_tier(r["tier_probs"], VARIANT_TAU)) if r["risk"] > RISK_FLOOR else cdf_tier(r["tier_probs"], VARIANT_TAU)), jev_overhead, "jev")
        R[f"Jev cumulative >= {VARIANT_TAU}, implicit-context bump"] = (from_rows(jev, lambda r: at_least_capable(cdf_tier(r["tier_probs"], VARIANT_TAU)) if r["implicit_context"] >= 0.5 else cdf_tier(r["tier_probs"], VARIANT_TAU)), jev_overhead, "jev")
        R["Jev difficulty Score, thirds"] = (from_rows(jev, lambda r: difficulty_tier(r["difficulty"])), jev_overhead, "jev")
    return R


def oracle(agg, mapping):
    def route(t):
        return max(TIERS, key=lambda tier: (agg[(t["id"], mapping[tier])]["pass"], -TIERS.index(tier))), False
    return route, (lambda t: (0.0, 0.0)), "oracle"


def evaluate(route, overhead, tasks, agg, mapping, labels):
    rows = []
    for t in tasks:
        tier, fell_back = route(t)
        a = agg[(t["id"], mapping[tier])]
        usd, secs = overhead(t)
        rows.append({"task": t["id"], "tier": tier, "label": labels[t["id"]], "pass": a["pass"],
                     "cost": a["cost"] + usd, "latency": a["latency"] + secs, "fell_back": fell_back})
    labeled = [r for r in rows if r["label"] in TIERS]
    idx = TIERS.index
    return {"n": len(rows), "pass": statistics.mean(r["pass"] for r in rows),
            "cost_per_1k": 1000 * statistics.mean(r["cost"] for r in rows),
            "p50_ms": 1000 * statistics.median(r["latency"] for r in rows), "p95_ms": 1000 * p95([r["latency"] for r in rows]),
            "under": sum(idx(r["tier"]) < idx(r["label"]) for r in labeled) / len(labeled) if labeled else None,
            "over": sum(idx(r["tier"]) > idx(r["label"]) for r in labeled) / len(labeled) if labeled else None,
            "tiers": dict(Counter(r["tier"] for r in rows)), "fell_back": sum(r["fell_back"] for r in rows), "rows": rows}


def frontier(results):
    names = [n for n, r in results.items() if r["kind"] != "oracle"]
    out = set()
    for n in names:
        a = results[n]
        dominated = any(results[m]["pass"] >= a["pass"] and results[m]["cost_per_1k"] <= a["cost_per_1k"]
                        and (results[m]["pass"] > a["pass"] or results[m]["cost_per_1k"] < a["cost_per_1k"])
                        for m in names if m != n)
        if not dominated:
            out.add(n)
    return out


# ---------------------------------------------------------------- report
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f} %"


def fmt_auc(x):
    return "n/a" if x is None else f"{x:.2f}"


def config_table(runs, errors, tasks, slices):
    lines = ["| Config | Slice | Pass rate | Cost per 1k tasks | Median latency | p95 latency | Output tokens (mean) | Refusals | max_tokens stops | Unresolved errors |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    slice_of = {t["id"]: t["slice"] for t in tasks}
    for c in CONFIGS:
        for s in slices:
            rs = [r for r in runs.values() if r["config"] == c and slice_of.get(r["task"]) == s]
            if not rs:
                continue
            per_task = defaultdict(list)
            for r in rs:
                per_task[r["task"]].append(r)
            pass_rate = statistics.mean(statistics.mean(x["passed"] for x in v) for v in per_task.values())
            cost = 1000 * statistics.mean(statistics.mean(x["cost_usd"] for x in v) for v in per_task.values())
            lat = [r["latency_s"] for r in rs]
            errs = sum(1 for k in errors if k[1] == c and slice_of.get(k[0]) == s)
            lines.append(f"| {c} | {s} | {pct(pass_rate)} | ${cost:.2f} | {1000 * statistics.median(lat):,.0f} ms | "
                         f"{1000 * p95(lat):,.0f} ms | {statistics.mean(r['usage']['output_tokens'] for r in rs):,.0f} | "
                         f"{sum(r['stop_reason'] == 'refusal' for r in rs)} | {sum(r['stop_reason'] == 'max_tokens' for r in rs)} | {errs} |")
    return "\n".join(lines)


def router_table(results, front, jev_cost):
    base = results.get("Always reasoning", {}).get("cost_per_1k")
    lines = ["| Router | Tasks | Pass rate | Cost per 1k tasks | vs always reasoning | Median latency | p95 latency | Under-routed | Over-routed | Frontier |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in results.items():
        cost = f"${r['cost_per_1k']:.2f}" + (" + Jev" if r["kind"] == "jev" and jev_cost is None else "")
        rel = f"{100 * r['cost_per_1k'] / base:.0f} %" if base else "n/a"
        mark = "ceiling" if r["kind"] == "oracle" else ("yes" if name in front else "")
        fb = f" ({r['fell_back']} fell back)" if r["fell_back"] else ""
        lines.append(f"| {name} | {r['n']}{fb} | {pct(r['pass'])} | {cost} | {rel} | {r['p50_ms']:,.0f} ms | {r['p95_ms']:,.0f} ms | "
                     f"{pct(r['under'])} | {pct(r['over'])} | {mark} |")
    return "\n".join(lines)


def label_table(labels_by_mapping, tasks):
    lines = ["| Mapping | Slice | fast | capable | reasoning | unsolved |", "| --- | --- | --- | --- | --- | --- |"]
    for mapping, labels in labels_by_mapping.items():
        for s in ("tool", "open", "all"):
            ids = [t["id"] for t in tasks if (s == "all" or t["slice"] == s) and t["id"] in labels]
            c = Counter(labels[i] for i in ids)
            lines.append(f"| {mapping} | {s} | {c['fast']} | {c['capable']} | {c['reasoning']} | {c['unsolved']} |")
    return "\n".join(lines)


def signal_table(tasks, labels, jev):
    sig = {"jido_ai Adaptive heuristic score": lambda t: adaptive_complexity(t["prompt"])}
    if jev:
        sig.update({
            "Jev P(tier above fast)": lambda t: 1 - jev[t["id"]]["tier_probs"].get("fast", 0.0),
            "Jev P(reasoning)": lambda t: jev[t["id"]]["tier_probs"].get("reasoning", 0.0),
            "Jev difficulty Score": lambda t: jev[t["id"]]["difficulty"],
            "Jev risk Score": lambda t: jev[t["id"]]["risk"],
            "Jev implicit-context Noul": lambda t: jev[t["id"]]["implicit_context"],
        })
    have = [t for t in tasks if t["id"] in labels and (not jev or t["id"] in jev)]
    lines = ["| Signal | AUC: needs more than fast | AUC: needs reasoning |", "| --- | --- | --- |"]
    out = {}
    for name, f in sig.items():
        above_fast = auc([f(t) for t in have if labels[t["id"]] in ("capable", "reasoning")],
                         [f(t) for t in have if labels[t["id"]] == "fast"])
        reasoning = auc([f(t) for t in have if labels[t["id"]] == "reasoning"],
                        [f(t) for t in have if labels[t["id"]] in ("fast", "capable")])
        out[name] = {"needs_more_than_fast": above_fast, "needs_reasoning": reasoning}
        lines.append(f"| {name} | {fmt_auc(above_fast)} | {fmt_auc(reasoning)} |")
    return "\n".join(lines), out


def confusion_table(tasks, labels, jev):
    lines = ["| Jev choice \\ label | fast | capable | reasoning | unsolved |", "| --- | --- | --- | --- | --- |"]
    c = Counter((jev[t["id"]]["tier"], labels[t["id"]]) for t in tasks if t["id"] in jev and t["id"] in labels)
    for tier in TIERS:
        lines.append(f"| {tier} | " + " | ".join(str(c[(tier, lab)]) for lab in TIERS + ["unsolved"]) + " |")
    return "\n".join(lines)


def analyze(out, jev_cost=None, log=print):
    out = Path(out)
    tasks, _ = load_tasks()
    by_id = {t["id"]: t for t in tasks}
    raw = read_jsonl(out / "runs.jsonl")
    if not raw:
        log(f"no {out / 'runs.jsonl'} yet: run label_models.py first")
        return None
    key = lambda r: (r["task"], r["config"], r["sample"])  # noqa: E731
    runs = latest_ok(raw, key)
    errors = {key(r) for r in raw if not r.get("ok")} - set(runs)
    changed = regrade(runs, by_id)
    agg = aggregate(runs)
    jev = latest_ok(read_jsonl(out / "jev_routes.jsonl"), lambda r: r["task"])
    llm = latest_ok(read_jsonl(out / "llm_routes.jsonl"), lambda r: r["task"])

    summary = {"regraded_changes": changed, "unresolved_errors": len(errors), "mappings": {}}
    sections, labels_by_mapping = [], {}
    for mapping_name, mapping in MAPPINGS.items():
        complete = [t for t in tasks if all((t["id"], mapping[tier]) in agg for tier in TIERS)]
        if not complete:
            continue
        labels = {t["id"]: label_for(agg, t["id"], mapping) for t in complete}
        labels_by_mapping[mapping_name] = labels
        routers = build_routers(jev, llm, jev_cost)
        routers["Perfect routing"] = oracle(agg, mapping)
        results = {}
        for name, (route, overhead, kind) in routers.items():
            results[name] = {**evaluate(route, overhead, complete, agg, mapping, labels), "kind": kind}
        front = frontier(results)
        summary["mappings"][mapping_name] = {
            "configs": mapping, "tasks": len(complete), "labels": labels,
            "routers": {n: {k: v for k, v in r.items() if k != "rows"} | {"frontier": n in front} for n, r in results.items()},
            "per_task": {n: r["rows"] for n, r in results.items()}}
        sections += ["", f"### Routers, {mapping_name} mapping ({', '.join(f'{k} = {v}' for k, v in mapping.items())}), {len(complete)} tasks",
                     "", router_table(results, front, jev_cost)]
        if mapping_name == "model":
            sig_md, sig = signal_table(complete, labels, jev)
            summary["signals"] = sig
            sections += ["", "### Which signals predict the label (model mapping)", "", sig_md]
            if jev:
                sections += ["", "### Jev's choice against the label (model mapping)", "", confusion_table(complete, labels, jev)]
    report = [f"Calls: {len(runs)} graded, {len(errors)} unresolved errors, {changed} grades changed on regrading.",
              "", "### Labels", "", label_table(labels_by_mapping, tasks),
              "", "### Per config", "", config_table(runs, errors, tasks, ("tool", "open"))] + sections
    report += ["", "Pass rate: mean over tasks of the share of samples that passed. Cost: mean per task, times 1,000. "
               "Latency: the chosen config's median on each task plus the router's own call. Under- and over-routed: "
               "share of labeled tasks sent below or above their label (unsolved tasks excluded). Perfect routing takes "
               "the highest pass rate per task, cheapest on ties, so it can sit above a label that only needs 2 of 3. "
               "Frontier: no other router matches or beats it on both pass rate and cost."]
    if jev_cost is None and jev:
        report += ["", "Jev rows exclude Jev's own per-call price (pass --jev-cost to include it); their latency includes it."]
    text = "\n".join(report)
    log(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


# ---------------------------------------------------------------- pilot
def pilot(log=print):
    sel = json.loads((TOOL_EXPERIMENT / "llm_baseline_select.json").read_text())
    jev = {r["query"]: r for r in json.loads((TOOL_EXPERIMENT / "jev_eval_results.json").read_text())["discovery"]["rows"]}
    H, O = sel["claude-haiku-4-5"]["rows"], sel["claude-opus-5"]["rows"]
    assert [h["query"] for h in H] == [o["query"] for o in O]
    perfect = [h if h["correct"] else o for h, o in zip(H, O)]
    lines = ["| Router | Correct | Cost per 1k calls |", "| --- | --- | --- |"]
    for name, rows in (("Always Haiku 4.5", H), ("Always Opus 5", O), ("Perfect routing (Haiku unless it fails)", perfect)):
        lines.append(f"| {name} | {sum(r['correct'] for r in rows)}/{len(rows)} | ${1000 * statistics.mean(r['cost_usd'] for r in rows):.2f} |")
    needs = [h["query"] for h, o in zip(H, O) if o["correct"] and not h["correct"]]
    fine = [h["query"] for h in H if h["correct"]]
    lines += ["", f"{len(needs)} requests needed Opus; {len(fine)} were fine on Haiku.", "",
              "| Jev signal (tool experiment) | AUC: needs Opus |", "| --- | --- |"]
    for name, f in (("complexity Score", lambda r: r["complexity"]), ("risk Score", lambda r: r["risk"]),
                    ("1 - tool-choice confidence", lambda r: 1 - r["confidence"])):
        lines.append(f"| {name} | {fmt_auc(auc([f(jev[q]) for q in needs], [f(jev[q]) for q in fine]))} |")
    lines += ["", "Haiku's replies on the requests that needed Opus:"]
    lines += [f"- {h['query']} -> {' '.join(h['text'].split())[:90]}..." for h in H if h["query"] in needs]
    log("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(HERE), help="directory holding runs.jsonl and the route files")
    ap.add_argument("--jev-cost", type=float, default=None, help="assumed $ per Jev call")
    ap.add_argument("--pilot", action="store_true", help="print the pilot table from ../jev_routing")
    args = ap.parse_args(argv)
    if args.pilot:
        return pilot()
    analyze(args.out, args.jev_cost)


if __name__ == "__main__":
    main()
