#!/usr/bin/env python3
"""Score Jev (or any System One endpoint, via JEV_API / JEV_MODEL as in jev_eval.py) on holdout.py's 45 requests.

Same request as the benchmark's "discovery" variant: the same state, the same four questions, the same
action descriptions; only the requests differ. 45 calls. Key handling is jev_eval.py's, never printed.

Run: python3 jev_holdout.py OUT.json
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import jev_eval as J  # noqa: E402
from holdout import HOLDOUT  # noqa: E402


def main(out):
    J.GOLD = HOLDOUT  # run_variant scores whatever GOLD holds
    acts = J.extract_actions()
    rows = J.run_variant(acts, "discovery", workers=4)
    s = J.summarize(rows)
    print(f"model {s['model']}: {sum(r['correct'] for r in rows)}/{len(rows)} correct, none detected {s['none_detected_by_choice']:.2f}, "
          f"median {s['latency_ms']['median']:.0f} ms")
    for r in rows:
        if not r["correct"]:
            print(f"  MISS {r['query']!r}: gold {r['gold']} got {r['choice']} ({r['confidence']:.2f})")
    Path(out).write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
