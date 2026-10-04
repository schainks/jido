#!/usr/bin/env python3
"""Score Jev zero-shot, a trained student and baselines against the right-tool labels (finalize_labels.py's output).

Jev gets the request alone, or the request with the session's opening task, one Choice question over the rubric's classes
(TAXONOMY.md descriptions). The student is Jevstiller's bge-small encoder and LinearStudent, 5-fold cross-validated with folds
grouped by session. A prediction counts as correct when it is one of the row's acceptable labels. Sends the scrubbed request texts
(at most 400 characters each) to TypeSafe's hosted Jev API; the key comes from TYPESAFE_API_KEY or ~/.typesafe_key, never printed.

Run: python rubric_eval.py FINAL_LABELS.jsonl PREPARED.jsonl --cache CACHE.jsonl [--json OUT.json]     (pip install jevstiller numpy)
"""
import argparse, collections, json, os, re, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
INSTRUCTIONS = "Which kind of tool is the right first action for this request? Judge what the request needs."


def rubric():
    text = (HERE / "TAXONOMY.md").read_text()
    return {m.group(1): re.sub(r"^right when the request needs the assistant to\.\.\. ?", "", m.group(2).strip()) for m in re.finditer(r"^\| `([a-z_]+)` \| (.+?) \|$", text, re.M)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("final"); ap.add_argument("prepared"); ap.add_argument("--cache", required=True); ap.add_argument("--json")
    a = ap.parse_args()
    from jevstiller import Task, load_encoder
    from jevstiller._student import LinearStudent
    from jevstiller.teachers import CachedTeacher
    from jevstiller.teachers.jev import JevTeacher
    rows = [json.loads(l) for l in open(a.final)]
    sess = {}
    for l in open(a.prepared):
        r = json.loads(l)
        sess[r["id"]] = r["session"]
    classes = rubric()
    labels = sorted(classes)
    ok = lambda pred, r: pred in r["acceptable"]  # noqa: E731
    out = {"n": len(rows), "main_thread": sum(not r["sidechain"] for r in rows), "subagent": sum(r["sidechain"] for r in rows)}
    top = collections.Counter(r["label"] for r in rows).most_common(1)[0][0]
    out["majority_class"] = {"class": top, "top1_vs_acceptable": round(sum(ok(top, r) for r in rows) / len(rows), 4)}
    out["habit_label_is_acceptable"] = {"all": round(sum(r["habit_in_acceptable"] for r in rows) / len(rows), 4),
                                        "main_thread": round(sum(r["habit_in_acceptable"] for r in rows if not r["sidechain"]) / out["main_thread"], 4),
                                        "subagent": round(sum(r["habit_in_acceptable"] for r in rows if r["sidechain"]) / out["subagent"], 4)}
    # Jev zero-shot
    key = os.environ.get("TYPESAFE_API_KEY") or (Path.home() / ".typesafe_key").read_text().strip()
    teacher = CachedTeacher(JevTeacher(api_key=key), a.cache)
    task = Task(name="right_tool", instructions=INSTRUCTIONS, classes=classes, target_agreement=0.98)
    for name, text in (("request_only", lambda r: r["request"]), ("with_session_task", lambda r: (f"Session task: {r['session_task']}\n" if r["session_task"] else "") + f"Request: {r['request']}")):
        outs = []
        for i in range(0, len(rows), 100):
            outs += teacher.classify([text(r) for r in rows[i:i + 100]], task)
        errs = sum(isinstance(o, Exception) for o in outs)
        good = [(r, o) for r, o in zip(rows, outs) if not isinstance(o, Exception)]
        top3 = lambda o: sorted(o.probs, key=o.probs.get, reverse=True)[:3]  # noqa: E731
        out[f"jev_{name}"] = {"errors": errs, "top1": round(sum(ok(o.label, r) for r, o in good) / len(good), 4),
                              "top1_main_thread": round(statistics.mean(ok(o.label, r) for r, o in good if not r["sidechain"]), 4),
                              "top1_subagent": round(statistics.mean(ok(o.label, r) for r, o in good if r["sidechain"]), 4),
                              "top3": round(sum(any(c in r["acceptable"] for c in top3(o)) for r, o in good) / len(good), 4),
                              "mean_confidence_right": round(statistics.mean(o.confidence for r, o in good if ok(o.label, r)), 3),
                              "mean_confidence_wrong": round(statistics.mean(o.confidence for r, o in good if not ok(o.label, r)), 3)}
        print(name, out[f"jev_{name}"], flush=True)
    # student, grouped 5-fold CV
    enc = load_encoder("small")
    E = np.vstack([enc.encode([r["request"] for r in rows[i:i + 64]]) for i in range(0, len(rows), 64)])
    ix = {c: i for i, c in enumerate(labels)}; y = np.array([ix[r["label"]] for r in rows])
    groups = sorted({sess[r["prepared_id"]] for r in rows}); rng = np.random.default_rng(0); rng.shuffle(groups)
    fold = {g: i % 5 for i, g in enumerate(groups)}; f = np.array([fold[sess[r["prepared_id"]]] for r in rows])
    hits, maj, top3h = [], [], []
    for k in range(5):
        tr, te = f != k, f == k
        st = LinearStudent(E.shape[1], len(labels)); st.fit(E[tr], np.eye(len(labels))[y[tr]], seed=0, epochs=800)
        P = E[te] @ st.W + st.b
        idx = np.where(te)[0]
        hits += [labels[int(P[j].argmax())] in rows[i]["acceptable"] for j, i in enumerate(idx)]
        top3h += [any(labels[c] in rows[i]["acceptable"] for c in np.argsort(-P[j])[:3]) for j, i in enumerate(idx)]
        m = labels[collections.Counter(y[tr]).most_common(1)[0][0]]
        maj += [m in rows[i]["acceptable"] for i in idx]
    out["bge_small_student_5fold_cv_by_session"] = {"top1": round(float(np.mean(hits)), 4), "top3": round(float(np.mean(top3h)), 4), "majority_in_folds": round(float(np.mean(maj)), 4)}
    print("student", out["bge_small_student_5fold_cv_by_session"])
    print({k: v for k, v in out.items() if k in ("n", "majority_class", "habit_label_is_acceptable")})
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
