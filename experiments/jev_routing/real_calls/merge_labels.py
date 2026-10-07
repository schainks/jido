#!/usr/bin/env python3
"""Merge two labellers' passes (labels_A_*.jsonl, labels_B_*.jsonl) and measure how well they agree, and how far both are from habit.

Agreement is on the primary label; Cohen's kappa corrects for chance. Rows where A and B differ go to an adjudication pass
(adjudicate batches are written here). A row where the labellers agree is final; a row where one's `alt` is the other's `label`
is "soft agreement" and gets both labels as acceptable. Habit labels (the tool actually called) are compared after mapping them
to the rubric's function classes, to show how far habit is from the right tool. Private data: mode 600, outside any repo.

Run: python3 merge_labels.py LABEL_DIR [--adjudication-batch 60]
"""
import argparse, collections, json
from pathlib import Path

HABIT_TO_RUBRIC = {"Read": "read_file", "Bash_view": "read_file", "Bash_search": "search_files", "Search": "search_files", "Edit": "edit_file", "Write": "write_file", "Bash_git": "git", "Bash_gh": "github",
                   "Bash_python": "run_code", "Bash_js": "run_code", "Bash_build": "run_code", "Bash_modify": "shell_ops", "Bash_wait": "shell_ops", "Bash_other": "shell_ops",
                   "Bash_network": "remote_or_http", "WebFetch": "web_fetch", "WebSearch": "web_search", "mcp_browser": "browser_or_app_ui", "mcp_simulator": "browser_or_app_ui",
                   "Agent": "delegate", "mcp_other": "external_service", "mcp_memory": "external_service", "mcp_posthog": "external_service", "Task": "shell_ops",
                   "Skill": "use_skill", "AskUserQuestion": "ask_user"}


def load(d, who):
    out = {}
    for p in sorted(Path(d).glob(f"labels_{who}_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("dir"); ap.add_argument("--adjudication-batch", type=int, default=60)
    a = ap.parse_args()
    d = Path(a.dir)
    A, B = load(d, "A"), load(d, "B")
    key = {r["id"]: r for r in map(json.loads, (d / "label_key.jsonl").read_text().splitlines() if (d / "label_key.jsonl").exists() else [])}
    ids = sorted(set(A) & set(B))
    print(f"labelled by A: {len(A)}, by B: {len(B)}, both: {len(ids)}")
    la, lb = [A[i]["label"] for i in ids], [B[i]["label"] for i in ids]
    agree = sum(x == y for x, y in zip(la, lb))
    soft = sum(1 for i in ids if A[i]["label"] != B[i]["label"] and (A[i].get("alt") == B[i]["label"] or B[i].get("alt") == A[i]["label"]))
    print(f"exact agreement {agree}/{len(ids)} = {agree / len(ids):.1%}, kappa {kappa(la, lb):.3f}; of the {len(ids) - agree} disagreements, {soft} are soft (one's alt is the other's label)")
    conf = collections.Counter((A[i]["conf"], B[i]["conf"]) for i in ids if A[i]["label"] == B[i]["label"])
    print("agreement when both are 'high':", sum(v for (x, y), v in conf.items() if x == y == "high"), "rows")
    dis = [i for i in ids if A[i]["label"] != B[i]["label"] and not (A[i].get("alt") == B[i]["label"] or B[i].get("alt") == A[i]["label"])]
    print(f"hard disagreements to adjudicate: {len(dis)}")
    pairs = collections.Counter(tuple(sorted((A[i]["label"], B[i]["label"]))) for i in ids if A[i]["label"] != B[i]["label"])
    print("most confused pairs:", pairs.most_common(8))
    if key:
        hab = {i: ("answer_directly" if key[i]["habit_label"] is None else HABIT_TO_RUBRIC.get(key[i]["habit_label"], "run_code")) for i in ids}
        ha = sum(hab[i] == A[i]["label"] for i in ids); hb = sum(hab[i] == B[i]["label"] for i in ids)
        hs = sum(hab[i] in (A[i]["label"], B[i]["label"]) for i in ids)
        print(f"habit (mapped to the rubric) equals A: {ha / len(ids):.1%}, B: {hb / len(ids):.1%}, either: {hs / len(ids):.1%}")
    (d / "disagreements.json").write_text(json.dumps([{"id": i, "a": A[i], "b": B[i]} for i in dis], indent=0)); (d / "disagreements.json").chmod(0o600)
    items = {x["id"]: x for b in sorted(d.glob("batch_*.json")) for x in json.loads(b.read_text())}
    for k in range(0, len(dis), a.adjudication_batch):
        p = d / f"adjudicate_{k // a.adjudication_batch:02d}.json"
        p.write_text(json.dumps([{"id": i, "request": items[i]["request"], "session_task": items[i]["session_task"], "option_1": A[i]["label"], "option_2": B[i]["label"]} for i in dis[k:k + a.adjudication_batch]]))
        p.chmod(0o600)
    print("adjudication batches written:", -(-len(dis) // a.adjudication_batch))


if __name__ == "__main__":
    main()
