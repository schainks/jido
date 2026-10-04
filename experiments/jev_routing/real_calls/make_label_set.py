#!/usr/bin/env python3
"""Pick self-contained user requests for labelling with the right tool (TAXONOMY.md), in batches the labellers can read.

From prepare.py's rows that are the first tool call after a user message, keep requests that stand on their own (at least 25
characters, not a bare "yes" or "continue", no leftover injected tag), drop repeated request texts, and draw a random sample
with its main-thread and subagent shares kept. Labellers get only an id, the request and the session's opening task; the
habit label (the tool actually called) goes into a separate key file they never see. Private data: mode 600, outside any repo.

Run: python3 make_label_set.py PREPARED.jsonl OUT_DIR [--n 600] [--batch 100] [--seed 0]
"""
import argparse, collections, json, random, re
from pathlib import Path

BARE = re.compile(r"^\W*(yes|yep|yeah|no|nope|ok|okay|continue|go on|go ahead|do it|proceed|sure|thanks|thank you|please|lgtm|sounds good|done|next)\b[\W\w]{0,20}$", re.I)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prepared"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=600); ap.add_argument("--batch", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prepared)]
    seen, pool = set(), []
    for r in rows:
        if not r["first_after_user"]:
            continue
        req = r["text_request"][len("Request: "):].strip()
        if len(req) < 25 or req.startswith("<") or BARE.match(req) or req in seen:
            continue
        seen.add(req)
        pool.append((r, req))
    rng = random.Random(a.seed)
    main_rows = [x for x in pool if not x[0]["sidechain"]]; side_rows = [x for x in pool if x[0]["sidechain"]]
    n_side = round(a.n * len(side_rows) / len(pool))
    pick = rng.sample(main_rows, min(a.n - n_side, len(main_rows))) + rng.sample(side_rows, min(n_side, len(side_rows)))
    rng.shuffle(pick)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key, items = [], []
    for i, (r, req) in enumerate(pick):
        task = re.sub(r"^Task: ", "", r.get("text", "").split("\n")[0])[:200]
        items.append({"id": i, "request": req[:400], "session_task": "" if task.startswith(req[:60]) else task})
        key.append({"id": i, "prepared_id": r["id"], "habit_label": r["label"], "sidechain": r["sidechain"], "split": r["split"]})
    for k in range(0, len(items), a.batch):
        p = out / f"batch_{k // a.batch:02d}.json"
        p.write_text(json.dumps(items[k:k + a.batch], indent=0)); p.chmod(0o600)
    kp = out / "label_key.jsonl"
    kp.write_text("\n".join(json.dumps(x) for x in key) + "\n"); kp.chmod(0o600)
    print(f"{len(pool)} self-contained requests ({len(main_rows)} main, {len(side_rows)} subagent); sampled {len(items)} into {-(-len(items) // a.batch)} batches; "
          f"request length median {sorted(len(x['request']) for x in items)[len(items) // 2]} chars")
    print("habit labels in the sample (not shown to labellers):", collections.Counter(x["habit_label"] for x in key).most_common(8))


if __name__ == "__main__":
    main()
