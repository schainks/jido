#!/usr/bin/env python3
"""Split extract_requests.py's output into labelling batches (id, request, session_task) plus a separate key file.

The key file (habit label, thread kind, session) is never shown to labellers; merge_labels.py and finalize_labels.py read it
exactly as for the first sample. Private data: mode 600, outside any repo.

Run: python3 batch_requests.py REQUESTS.jsonl OUT_DIR [--batch 200]
"""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0]); ap.add_argument("requests"); ap.add_argument("out"); ap.add_argument("--batch", type=int, default=200)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.requests)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(rows), a.batch):
        p = out / f"batch_{k // a.batch:02d}.json"
        p.write_text(json.dumps([{"id": r["id"], "request": r["request"], "session_task": r["session_task"]} for r in rows[k:k + a.batch]], indent=0)); p.chmod(0o600)
    kp = out / "label_key.jsonl"
    kp.write_text("\n".join(json.dumps({"id": r["id"], "prepared_id": None, "habit_label": r["habit_label"], "sidechain": r["sidechain"], "session": r["session"]}) for r in rows) + "\n"); kp.chmod(0o600)
    print(f"{len(rows)} requests in {-(-len(rows) // a.batch)} batches of up to {a.batch} -> {out}")


if __name__ == "__main__":
    main()
