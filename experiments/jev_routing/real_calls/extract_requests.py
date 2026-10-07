#!/usr/bin/env python3
"""Extract every substantial user message from Claude Code transcripts as a request to label, whether or not a tool call followed.

A request is a plain user message (not a tool result) of at least 25 characters that is not injected text (session-continuation
summaries, skill bodies, caveats, notification blocks), scrubbed and masked exactly as build_real_calls.py and prepare.py do,
de-duplicated, and not already in the labelled set. Rows keep the session hash, thread kind (typed vs delegated to a subagent),
the session's opening task, and, for comparison only, the habit label of the tool called first (None when the assistant answered
without a tool). Private data: mode 600, outside any repo. Only counts are printed.

Run: python3 extract_requests.py OUT.jsonl [--skip-session FILE ...] [--secret-file FILE ...] [--exclude-labelled FINAL_LABELS.jsonl ...]
"""
import argparse, collections, glob, hashlib, json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_real_calls import HOME, REMINDER, Scrubber, label_of, short, text_of  # noqa: E402
from prepare import mask  # noqa: E402

ART = [re.compile(r"^\s*(This session is being continued|Summary:)", re.I), re.compile(r"^\s*Base directory for this skill", re.I), re.compile(r"^\s*(Caveat:|\[Image|\[Request interrupted)", re.I)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out"); ap.add_argument("--skip-session", action="append", default=[]); ap.add_argument("--secret-file", action="append", default=[])
    ap.add_argument("--exclude-labelled", action="append", default=[])
    a = ap.parse_args()
    scrub = Scrubber([Path(p).read_text().strip() for p in a.secret_file if Path(p).exists()])
    done = {json.loads(l)["request"] for p in a.exclude_labelled for l in open(p)}
    skip = {os.path.realpath(p) for p in a.skip_session}
    files = sorted(f for f in glob.glob(os.path.join(HOME, ".claude/projects/**/*.jsonl"), recursive=True) if os.path.realpath(f) not in skip)
    stats, seen, rows = collections.Counter(), set(done), []
    for f in files:
        session = hashlib.sha1(f.encode()).hexdigest()[:10]
        first, pending = {}, {}
        for line in open(f, errors="ignore"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("message")
            if not isinstance(m, dict):
                continue
            thread = bool(d.get("isSidechain")); c = m.get("content")
            if d.get("type") == "user":
                if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                    continue
                raw = REMINDER.sub(" ", text_of(c) if c is not None else "")
                t = short(raw, 400)
                pending.pop(thread, None)
                if not t:
                    continue
                first.setdefault(thread, t)
                stats["user messages"] += 1
                if len(t) < 25 or t.startswith("<") or any(p.search(t) for p in ART):
                    stats["dropped: short, tag or injected text"] += 1; continue
                req, task = scrub(t), scrub(first[thread][:200])
                if req is None or task is None:
                    stats["dropped: scrubber"] += 1; continue
                req, task = mask(req), mask(task)
                if req in seen:
                    stats["dropped: duplicate or already labelled"] += 1; continue
                seen.add(req)
                rows.append({"session": session, "sidechain": thread, "request": req[:400], "session_task": "" if task.startswith(req[:60]) else task, "habit_label": None})
                pending[thread] = len(rows) - 1
            elif d.get("type") == "assistant" and isinstance(c, list) and thread in pending:
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_use" and rows[pending[thread]]["habit_label"] is None:
                        rows[pending[thread]]["habit_label"] = label_of(b.get("name"), b.get("input") or {})
    for i, r in enumerate(rows):
        r["id"] = i
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    os.chmod(a.out, 0o600)
    print(f"{len(files)} files; {dict(stats)}")
    print(f"{len(rows)} new requests ({sum(not r['sidechain'] for r in rows)} typed, {sum(r['sidechain'] for r in rows)} delegated); "
          f"{sum(r['habit_label'] is None for r in rows)} were answered without any tool call")


if __name__ == "__main__":
    main()
