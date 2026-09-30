#!/usr/bin/env python3
"""Record Jev's answers on the Jido tool-selection requests, and replay them through Jevstiller.

Jevstiller (github.com/tomerglick57/Jevstiller) sits in front of Jev's Choice calls and trains a local student
from Jev's answers. This follows its own experiment protocol (experiments/record_answers.py, run.py) on our data:

  record   one Jev call per request, in its cache format (JSONL keyed by task version and text)
             --teacher jev    the real thing (needs a TypeSafe key: TYPESAFE_API_KEY, else ~/.typesafe_key; never printed)
             --teacher gold   our gold labels as a perfect teacher, to check the pipeline and give a ceiling
  replay   stream the training requests through the Jevstiller loop with recorded answers (free, offline), and score
           the held-out requests at checkpoints with Jevstiller.evaluate(), which never writes to its store

Requests: make_data.py's 40-per-action training rows (streamed and recorded), the 34 benchmark requests and
holdout.py's 45 (held out: recorded, never streamed). The state is the request text alone, as
jevstiller_student.py found best; the task is one Choice question over the action descriptions plus "none".

Run: python jevstiller_replay.py record DATA_DIR HOLDOUT_DIR --cache answers.jsonl [--teacher jev|gold]
     python jevstiller_replay.py replay DATA_DIR HOLDOUT_DIR --cache answers.jsonl [--encoder small] --json out.json
     (pip install jevstiller pyarrow; replay needs no key)
"""
import argparse, json, os, statistics, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import jev_eval as J  # noqa: E402

INSTRUCTIONS = "Which action should the agent run to satisfy this request? Pick none if no listed action fits."


def make_task(target=0.98):
    from jevstiller import Task
    return Task(name="jido_tools", instructions=INSTRUCTIONS, classes=J.criteria(J.extract_actions(), "discovery"), target_agreement=target)


def load_rows(data, holdout):
    import pyarrow.parquet as pq

    def rows(path):
        out = []
        for r in pq.read_table(path).to_pylist():
            g = json.loads(r["gold"])["tool"]["probabilities"]
            out.append((json.loads(r["state"])["request"], {k for k, p in g.items() if p > 0}))
        return out
    return rows(f"{data}/all/train.parquet"), rows(f"{data}/all/test.parquet"), rows(f"{holdout}/all/test.parquet")


class GoldTeacher:
    """Our gold label as the teacher's answer: 0.9 on it, the rest spread evenly."""
    name = "gold"

    def __init__(self, gold):
        self.gold = gold

    def classify(self, texts, task):
        from jevstiller.teachers import TeacherOutput
        out = []
        for t in texts:
            g = sorted(self.gold[t])[0]
            rest = 0.1 / (len(task.labels) - 1)
            out.append(TeacherOutput(label=g, probs={c: 0.9 if c == g else rest for c in task.labels}, confidence=0.9, input_tokens=0,
                                     cost_usd=0.0, latency_ms=0.0, request_id=None, model="gold"))
        return out


class NoLive:
    name = "cached"

    def classify(self, texts, task):
        raise KeyError(f"{len(texts)} requests are not in the recorded answers; run `record` first")


def api_key():
    k = os.environ.get("TYPESAFE_API_KEY")
    p = Path.home() / ".typesafe_key"
    return k or (p.read_text().strip() if p.exists() else None)


def record(a):
    from jevstiller.teachers import CachedTeacher
    train, bench, hold = load_rows(a.data, a.holdout)
    gold = {q: g for q, g in train + bench + hold}
    texts = list(dict.fromkeys(q for q, _ in train + bench + hold))
    task = make_task(a.target)
    if a.teacher == "gold":
        inner = GoldTeacher(gold)
    else:
        from jevstiller.teachers.jev import JevTeacher
        k = api_key()
        if not k:
            sys.exit("no key: set TYPESAFE_API_KEY or write it to ~/.typesafe_key")
        inner = JevTeacher(model=a.model, api_key=k)
    teacher = CachedTeacher(inner, a.cache)
    errors, outs = 0, {}
    for i in range(0, len(texts), 100):
        chunk = texts[i:i + 100]
        for t, o in zip(chunk, teacher.classify(chunk, task)):
            if isinstance(o, Exception):
                errors += 1
            else:
                outs[t] = o
        print(f"{min(i + 100, len(texts)):>5}/{len(texts)}  cache hits {teacher.hits}  calls {teacher.misses}  errors {errors}", flush=True)
    if errors:
        sys.exit(f"{errors} answers failed; run again to fetch them")
    print(f"complete: {len(teacher.cache)} answers in {a.cache}")
    for name, rows in (("benchmark", bench), ("holdout", hold)):
        ok = sum(outs[q].label in g for q, g in rows)
        print(f"  {a.teacher} alone on the {name}: {ok}/{len(rows)} correct (request text only as the state)")
    lat = [o.latency_ms for o in outs.values() if o.latency_ms]
    print(f"  cost ${sum(o.cost_usd for o in outs.values()):.4f}" + (f", median latency {statistics.median(lat):.0f} ms" if lat else ""))


def replay(a):
    import numpy as np
    from jevstiller import Config, Jevstiller, load_encoder
    from jevstiller.teachers import CachedTeacher, ReplayTeacher
    train, bench, hold = load_rows(a.data, a.holdout)
    cache = CachedTeacher(NoLive(), a.cache).cache
    answers = {k.split("\t", 1)[1]: v for k, v in cache.items()}   # keyed by text, so one recording serves every target
    ev = [(q, g, "benchmark") for q, g in bench] + [(q, g, "holdout") for q, g in hold]
    missing = [q for q, _ in train + [(q, g) for q, g, _ in ev] if q not in answers]
    if missing:
        sys.exit(f"{len(missing)} requests have no recorded answer; run `record` first")
    enc = load_encoder(a.encoder)
    ev_texts = [q for q, _, _ in ev]
    ev_teacher = np.array([answers[q].label for q in ev_texts])
    where = np.array([w for _, _, w in ev])
    gold_ok = lambda labels: np.array([l in g for l, (_, g, _) in zip(labels, ev)])  # noqa: E731
    jev_ok = gold_ok(ev_teacher)
    print(f"teacher alone vs gold: benchmark {int(jev_ok[where == 'benchmark'].sum())}/{(where == 'benchmark').sum()}, "
          f"holdout {int(jev_ok[where == 'holdout'].sum())}/{(where == 'holdout').sum()}")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(train))
    stream = [train[i][0] for i in order]
    arms = {"library defaults": {}, "run.py protocol": dict(min_train_samples=500, min_samples_per_class=5, min_calib_samples=200, min_new_samples=1000, shadow_min_samples=300),
            "small-data": dict(min_train_samples=200, min_samples_per_class=10, min_calib_samples=100, min_new_samples=150, shadow_min_samples=100)}
    results = []
    for target in a.targets:
        for arm, kw in arms.items():
            task = make_task(target)
            with tempfile.TemporaryDirectory() as d:
                js = Jevstiller(task, ReplayTeacher(answers), d, encoder=enc, config=Config(training="inline", seed=a.seed, **kw))
                seen, local = 0, 0
                rows = []
                for s in range(0, len(stream), a.batch):
                    res = js.classify_batch(stream[s:s + a.batch])
                    seen += len(res); local += sum(r.source != "teacher" for r in res)
                    if seen % a.checkpoint < a.batch or seen == len(stream):
                        st = js.status(); row = {"streamed": seen, "production": st.production, "stream_student_share": round(local / seen, 4)}
                        if st.production:
                            e = js.evaluate(ev_texts, ev_teacher)
                            acc = e["accepted"]; sys_label = np.where(acc, e["student_label"], ev_teacher)
                            sys_ok = gold_ok(sys_label)
                            row.update({"coverage": round(float(e["coverage"]), 4), "system_agreement_with_teacher": round(float(e["system_agreement"]), 4),
                                        "accepted": int(acc.sum()),
                                        "coverage_benchmark": round(float(acc[where == "benchmark"].mean()), 4), "coverage_holdout": round(float(acc[where == "holdout"].mean()), 4),
                                        "system_correct_benchmark": int(sys_ok[where == "benchmark"].sum()), "system_correct_holdout": int(sys_ok[where == "holdout"].sum()),
                                        "student_correct_on_accepted": int(gold_ok(e["student_label"])[acc].sum())})
                        rows.append(row)
                final = js.status()
                js.close()
            last = rows[-1]
            cov = f"coverage {last['coverage']:.1%} (benchmark {last['coverage_benchmark']:.0%}, holdout {last['coverage_holdout']:.0%}), agreement with teacher {last['system_agreement_with_teacher']:.1%}, " \
                  f"system correct {last['system_correct_benchmark']}/34 and {last['system_correct_holdout']}/45" if "coverage" in last else "no student yet"
            print(f"target {target:.2f} | {arm:16s} | production {str(final.production or '-'):11s} | {cov}")
            results.append({"target": target, "arm": arm, "thresholds": kw, "checkpoints": rows})
    if a.json:
        Path(a.json).write_text(json.dumps({"encoder": enc.id, "teacher_vs_gold": {"benchmark": int(jev_ok[where == "benchmark"].sum()), "holdout": int(jev_ok[where == "holdout"].sum())},
                                            "streamed": len(stream), "held_out": len(ev), "seed": a.seed, "runs": results}, indent=1))
        print(f"wrote {a.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["record", "replay"]); ap.add_argument("data"); ap.add_argument("holdout")
    ap.add_argument("--cache", required=True); ap.add_argument("--teacher", default="jev", choices=["jev", "gold"]); ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--target", type=float, default=0.98, help="record: the task version the answers are cached under")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.98, 0.95]); ap.add_argument("--encoder", default="small")
    ap.add_argument("--batch", type=int, default=50); ap.add_argument("--checkpoint", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    (record if a.mode == "record" else replay)(a)


if __name__ == "__main__":
    main()
