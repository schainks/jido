#!/usr/bin/env python3
"""Replay Jev's answers on the right-tool requests through Jevstiller's real loop, and score against the right-tool labels.

Sessions are split 85/15. The 85% stream through Jevstiller (inline training) with Jev's recorded answers as the teacher (from
rubric_eval.py's cache; no API calls); the 15% are held out and scored with Jevstiller.evaluate(), which never writes to its store.
For each target agreement and threshold arm it reports the share the student answers on the held-out requests, its agreement with
Jev, and the whole system's accuracy against the labels (student where it answers, Jev elsewhere) next to Jev alone. A prediction
counts as correct when it is one of a row's acceptable labels. Private data stays outside the repo; only aggregates are printed.

Run: python replay_jevstiller.py ALL_LABELS.jsonl --cache JEV_CACHE.jsonl [--json OUT.json] [--targets 0.98 0.95 0.9]     (pip install jevstiller)
"""
import argparse, json, tempfile
from pathlib import Path
import numpy as np
from rubric_eval import INSTRUCTIONS, rubric


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("labels"); ap.add_argument("--cache", required=True); ap.add_argument("--json"); ap.add_argument("--targets", type=float, nargs="+", default=[0.98, 0.95, 0.9])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--batch", type=int, default=50); ap.add_argument("--encoder", default="small")
    a = ap.parse_args()
    from jevstiller import Config, Jevstiller, Task, load_encoder
    from jevstiller.teachers import CachedTeacher, ReplayTeacher
    rows = [json.loads(l) for l in open(a.labels)]
    class NoLive:
        name = "cached"
    answers = {k.split("\t", 1)[1]: v for k, v in CachedTeacher(NoLive(), a.cache).cache.items()}
    rows = [r for r in rows if r["request"] in answers]
    sessions = sorted({r["session"] for r in rows}); rng = np.random.default_rng(a.seed); rng.shuffle(sessions)
    held = set(sessions[:round(0.15 * len(sessions))])
    stream = [r for r in rows if r["session"] not in held]; ev = [r for r in rows if r["session"] in held]
    order = rng.permutation(len(stream)); stream = [stream[i] for i in order]
    ok = lambda label, r: label in r["acceptable"]  # noqa: E731
    ev_texts = [r["request"] for r in ev]; ev_jev = np.array([answers[t].label for t in ev_texts])
    jev_ok = np.array([ok(l, r) for l, r in zip(ev_jev, ev)])
    print(f"stream {len(stream)} requests, held out {len(ev)} (sessions never seen); Jev alone on the held-out: {jev_ok.mean():.1%} right")
    enc = load_encoder(a.encoder)
    arms = {"library defaults": {}, "run.py protocol": dict(min_train_samples=500, min_samples_per_class=5, min_calib_samples=200, min_new_samples=1000, shadow_min_samples=300),
            "small-data": dict(min_train_samples=200, min_samples_per_class=10, min_calib_samples=100, min_new_samples=150, shadow_min_samples=100)}
    results = []
    for target in a.targets:
        for arm, kw in arms.items():
            task = Task(name="right_tool", instructions=INSTRUCTIONS, classes=rubric(), target_agreement=target)
            with tempfile.TemporaryDirectory() as d:
                js = Jevstiller(task, ReplayTeacher(answers), d, encoder=enc, config=Config(training="inline", seed=a.seed, **kw))
                seen = local = 0
                for s in range(0, len(stream), a.batch):
                    res = js.classify_batch([r["request"] for r in stream[s:s + a.batch]])
                    seen += len(res); local += sum(x.source != "teacher" for x in res)
                st = js.status(); row = {"target": target, "arm": arm, "production": st.production, "stream_student_share": round(local / seen, 4)}
                if st.production:
                    e = js.evaluate(ev_texts, ev_jev)
                    acc = e["accepted"]; sys_label = np.where(acc, e["student_label"], ev_jev)
                    sys_ok = np.array([ok(l, r) for l, r in zip(sys_label, ev)]); stu_ok = np.array([ok(l, r) for l, r in zip(e["student_label"], ev)])
                    row.update({"coverage": round(float(e["coverage"]), 4), "agreement_with_jev": round(float(e["system_agreement"]), 4), "system_correct": round(float(sys_ok.mean()), 4),
                                "jev_alone_correct": round(float(jev_ok.mean()), 4), "student_correct_when_it_answers": round(float(stu_ok[acc].mean()), 4) if acc.any() else None})
                js.close()
            results.append(row)
            print(f"target {target:.2f} | {arm:16s} | " + (f"production {row['production']}: answers {row['coverage']:.0%} of held-out, agrees with Jev {row['agreement_with_jev']:.1%}, system {row['system_correct']:.1%} vs Jev alone {row['jev_alone_correct']:.1%}, "
                                                          f"student right {row['student_correct_when_it_answers']:.1%} when it answers" if row["production"] else "no student yet"), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps({"stream": len(stream), "held_out": len(ev), "jev_alone_correct": round(float(jev_ok.mean()), 4), "runs": results}, indent=1))


if __name__ == "__main__":
    main()
