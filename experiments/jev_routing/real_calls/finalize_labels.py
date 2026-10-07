#!/usr/bin/env python3
"""Final labels from two labellers' passes, an adjudication file and an artifact list.

  both labellers agree         that label
  soft disagreement            the two primary labels are both acceptable (one's `alt` is the other's label)
  hard disagreement            the adjudication file's acceptable set for that id (an id in neither is dropped)
  artifact                     dropped: injected text that is not a request (session-continuation summaries, skill bodies, caveats)

Each final row has `acceptable` (a list, like the Jido gold sets), a primary `label` (the agreed one, else the more confident,
else A's) and the habit label (the tool actually called) mapped to the rubric, so the gap between habit and the right tool can be
measured. Private data: mode 600, outside any repo. Only counts are printed.

Run: python3 finalize_labels.py LABEL_DIR [--adjudication adjudication.json] [--artifacts artifacts.json]
"""
import argparse, collections, json
from pathlib import Path
from merge_labels import HABIT_TO_RUBRIC, load

RANK = {"high": 2, "med": 1, "low": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir"); ap.add_argument("--adjudication", default="adjudication.json"); ap.add_argument("--artifacts", default="artifacts.json")
    a = ap.parse_args()
    d = Path(a.dir)
    A, B = load(d, "A"), load(d, "B")
    adj = {int(k): v for k, v in json.loads((d / a.adjudication).read_text()).items()} if (d / a.adjudication).exists() else {}
    art = {int(k) for k in json.loads((d / a.artifacts).read_text())} if (d / a.artifacts).exists() else set()
    items = {x["id"]: x for b in sorted(d.glob("batch_*.json")) for x in json.loads(b.read_text())}
    key = {r["id"]: r for r in map(json.loads, (d / "label_key.jsonl").read_text().splitlines())}
    out, why = [], collections.Counter()
    for i in sorted(items):
        if i in art:
            why["artifact"] += 1; continue
        la, lb = A[i], B[i]
        if la["label"] == lb["label"]:
            acc, kind = [la["label"]], "agreed"
        elif la.get("alt") == lb["label"] or lb.get("alt") == la["label"]:
            acc, kind = [la["label"], lb["label"]], "soft"
        elif i in adj:
            acc, kind = adj[i], "adjudicated"
        else:
            why["hard disagreement not adjudicated"] += 1; continue
        prim = acc[0] if kind != "soft" else (la["label"] if RANK[la["conf"]] >= RANK[lb["conf"]] else lb["label"])
        why[kind] += 1
        habit = "answer_directly" if key[i]["habit_label"] is None else HABIT_TO_RUBRIC.get(key[i]["habit_label"], "run_code")
        out.append({"id": i, "request": items[i]["request"], "session_task": items[i]["session_task"], "sidechain": key[i]["sidechain"], "prepared_id": key[i]["prepared_id"], "session": key[i].get("session"),
                    "label": prim, "acceptable": acc, "kind": kind, "habit": habit, "habit_in_acceptable": habit in acc})
    p = d / "final_labels.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in out) + "\n"); p.chmod(0o600)
    print(f"{len(out)} final rows; {dict(why)}")
    print("primary label counts:", collections.Counter(r["label"] for r in out).most_common(17))
    print(f"rows with more than one acceptable label: {sum(len(r['acceptable']) > 1 for r in out)} ({sum(len(r['acceptable']) > 1 for r in out) / len(out):.1%})")
    print(f"habit label (mapped) is an acceptable answer: {sum(r['habit_in_acceptable'] for r in out) / len(out):.1%}"
          f" (main thread {sum(r['habit_in_acceptable'] for r in out if not r['sidechain']) / sum(not r['sidechain'] for r in out):.1%}, subagent {sum(r['habit_in_acceptable'] for r in out if r['sidechain']) / max(1, sum(r['sidechain'] for r in out)):.1%})")


if __name__ == "__main__":
    main()
