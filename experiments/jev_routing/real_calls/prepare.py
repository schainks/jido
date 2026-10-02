#!/usr/bin/env python3
"""Turn build_real_calls.py's rows into the exact texts used for labelling and training.

A second, stricter mask runs over every field (IPv4 addresses, user@host strings, long digit runs, phone-like
numbers, any remaining long mixed letter-and-digit token), then the context is laid out as one text. Rows get a
split by session (a session is never split across train, validation and test): 10% test, 10% validation, the rest
train. Two texts per row: without and with the assistant's narration from the same response, which often names the
action it is about to take.

The output is private data and stays in OUT_DIR, outside any repository. Only counts are printed.

Run: python3 prepare.py CALLS.jsonl OUT.jsonl
"""
import collections, hashlib, json, re, sys
from pathlib import Path

CLASSES = {
    "Read": "Read the contents of a file", "Edit": "Edit an existing file by replacing text in it", "Write": "Create a new file or overwrite one with new content",
    "Bash_view": "Shell command that views or inspects files and directories: cat, head, tail, sed -n, ls, wc, diff, pwd",
    "Bash_search": "Shell command that searches: grep, rg, find", "Bash_git": "A git command", "Bash_gh": "A GitHub CLI (gh) command: pull requests, issues, API calls",
    "Bash_python": "Run Python, pip, uv or pytest", "Bash_js": "Run node, npm, pnpm, npx or yarn", "Bash_build": "Build or run a toolchain: cargo, make, mix, go, docker, xcodebuild",
    "Bash_network": "Network or remote shell command: curl, ssh, wget, scp", "Bash_modify": "Shell command that changes the filesystem: cp, mv, mkdir, rm, touch, in-place perl edits",
    "Bash_wait": "Wait or poll in a shell loop: sleep, for, until, while", "Bash_other": "Any other shell command",
    "WebFetch": "Fetch the contents of a web page or URL", "WebSearch": "Search the web", "Agent": "Launch a subagent to do a delegated task",
    "Task": "Create or update items in the task list", "Skill": "Invoke a named skill", "ToolSearch": "Look up a deferred tool's definition",
    "AskUserQuestion": "Ask the user a multiple-choice question", "SendMessage": "Send a message to another agent", "StructuredOutput": "Return the final structured result",
    "PlanMode": "Enter or exit plan mode", "Search": "The native search tool (Grep or Glob)",
    "mcp_browser": "Control a web browser: click, type, navigate, screenshot, run JavaScript", "mcp_simulator": "Control the iOS simulator",
    "mcp_memory": "Search memory of past conversations", "mcp_posthog": "Query PostHog analytics", "mcp_other": "Another MCP tool: Slack, GitHub, Google Drive, Sentry, documents and so on",
    "other": "Any other tool"}
INSTRUCTIONS = "Which tool will the assistant call next? Pick the one that best matches what it needs to do, given the task and the steps so far."

MASKS = [(re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"), (re.compile(r"\b[A-Za-z_][A-Za-z0-9_\-]*@[A-Za-z0-9.\-]+\b"), "<user@host>"),
         (re.compile(r"\+?\d[\d\s().\-]{8,}\d"), "<num>"), (re.compile(r"\b\d{9,}\b"), "<num>"),
         (re.compile(r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{20,}\b"), "<id>")]


def mask(t):
    for p, r in MASKS:
        t = p.sub(r, t)
    return t


def layout(r, narration):
    s = f"Task: {r['task']}\nLatest user message: {r['latest_user']}\nPrevious steps: {r['previous_steps'] or '(none)'}\nLast result: {r['last_result'] or '(none)'}"
    return s + (f"\nAssistant said: {r['narration']}" if narration and r["narration"] else "")


def split_of(session):
    h = int(hashlib.sha1(session.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def main(src, dst):
    rows, n = [], 0
    for line in open(src):
        r = json.loads(line)
        for k in ("task", "latest_user", "previous_steps", "last_result", "narration"):
            r[k] = mask(r[k])
        label = r["label"].replace(":", "_")
        if label not in CLASSES:
            label = "other"
        rows.append({"id": n, "session": r["session"], "split": split_of(r["session"]), "label": label, "sidechain": r["sidechain"], "text": layout(r, False), "text_narrated": layout(r, True),
                     "has_narration": bool(r["narration"]), "first_after_user": r.get("first_after_user", False),
                     "text_request": f"Request: {r['latest_user']}"})
        n += 1
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(dst).chmod(0o600)
    by = collections.Counter(r["split"] for r in rows)
    print(f"{n} rows -> {dst}; splits {dict(by)}; sessions {len({r['session'] for r in rows})}; with narration {sum(r['has_narration'] for r in rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
