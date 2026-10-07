#!/usr/bin/env python3
"""Combine finalized label sets into one file with a session id and a `short` flag (request under 60 characters).

The first sample's rows carry a prepared_id (their session comes from prepare.py's output); later samples carry the session directly.
Private data: mode 600, outside any repo.

Run: python3 combine_labels.py OUT.jsonl PREPARED.jsonl FINAL_1.jsonl [FINAL_2.jsonl ...]
"""
import json, sys
from pathlib import Path


def main(out, prepared, *finals):
    sess = {}
    for l in open(prepared):
        r = json.loads(l)
        sess[r["id"]] = r["session"]
    rows, n = [], 0
    for k, f in enumerate(finals):
        for l in open(f):
            r = json.loads(l)
            r["session"] = r.get("session") or sess[r["prepared_id"]]
            r["set"] = k
            r["short"] = len(r["request"]) < 60
            r["id"] = n; n += 1
            rows.append(r)
    Path(out).write_text("\n".join(json.dumps(r) for r in rows) + "\n"); Path(out).chmod(0o600)
    print(f"{len(rows)} rows from {len(finals)} sets; {sum(r['short'] for r in rows)} short; sessions {len({r['session'] for r in rows})}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
