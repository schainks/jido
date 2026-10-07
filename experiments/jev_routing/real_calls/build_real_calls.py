#!/usr/bin/env python3
"""Build a next-tool-prediction dataset from Claude Code session transcripts (~/.claude/projects/**/*.jsonl).

One row per tool call: the label is the tool Claude called, grouped into about 30 classes (Bash is split by its
first command word, MCP tools by server), and the context is what was known just before the call: the session's
first user prompt, the latest user message, the previous three steps (tool and a short argument, file names only),
the last tool result, and any narration that came earlier in the same response. Every text field is scrubbed
(secrets, tokens, emails, home paths, URL query strings) and a row that still looks like it holds a secret is dropped.

The output is private data: write it outside any repository. Only counts are printed. This script's own tests:
python3 build_real_calls.py --selftest

Run: python3 build_real_calls.py OUT_DIR [--skip-session FILE ...] [--secret-file FILE ...]
       --secret-file  a file whose whole content is a literal secret to drop any row containing it (never printed)
"""
import argparse, collections, glob, hashlib, json, os, re, sys
from pathlib import Path

HOME = os.path.expanduser("~")

# ---------------------------------------------------------------- labels
BASH_GROUPS = {
    "search": {"grep", "rg", "find", "ag", "fd"}, "view": {"cat", "head", "tail", "sed", "ls", "wc", "pwd", "diff", "awk", "less", "file", "stat", "tree"},
    "git": {"git"}, "gh": {"gh"}, "python": {"python", "python3", "pip", "uv", "pytest"}, "js": {"node", "npx", "pnpm", "npm", "yarn", "bun", "tsc"},
    "build": {"cargo", "make", "mix", "go", "xcrun", "xcodebuild", "swift", "docker"}, "network": {"curl", "ssh", "wget", "scp", "rsync", "nc"},
    "modify": {"cp", "mv", "mkdir", "rm", "perl", "touch", "chmod", "ln"}, "wait": {"for", "until", "while", "sleep"}}
CORE = {"Read": "Read", "Edit": "Edit", "MultiEdit": "Edit", "NotebookEdit": "Edit", "Write": "Write", "WebFetch": "WebFetch", "WebSearch": "WebSearch",
        "Agent": "Agent", "Task": "Agent", "Skill": "Skill", "ToolSearch": "ToolSearch", "TaskCreate": "Task", "TaskUpdate": "Task", "TaskGet": "Task",
        "TaskList": "Task", "TodoWrite": "Task", "AskUserQuestion": "AskUserQuestion", "SendMessage": "SendMessage", "StructuredOutput": "StructuredOutput",
        "EnterPlanMode": "PlanMode", "ExitPlanMode": "PlanMode", "Grep": "Search", "Glob": "Search"}
ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")


def first_word(cmd):
    for part in re.split(r"&&|;|\n", cmd.strip()):
        toks = part.strip().split()
        if not toks or toks[0].startswith("#"):
            continue
        while toks and ENV.match(toks[0]):
            toks = toks[1:]
        if not toks:
            continue
        w = toks[0].split("/")[-1]
        if w in ("cd", "export", "set", "source", "echo", "true", ":"):
            continue
        if w in ("sudo", "time", "nohup", "env", "timeout", "exec") and len(toks) > 1:
            w = toks[1].split("/")[-1]
        return w
    return ""


def label_of(name, inp):
    if name == "Bash":
        w = first_word((inp or {}).get("command", ""))
        return "Bash:" + next((g for g, ws in BASH_GROUPS.items() if w in ws), "other")
    if name in CORE:
        return CORE[name]
    if name and name.startswith("mcp__"):
        s = name.split("__")[1].lower()
        return "mcp:" + ("browser" if "chrome" in s or "browser" in s else "simulator" if "simulator" in s else "memory" if "memory" in s else "posthog" if "posthog" in s else "other")
    return "other"


# ---------------------------------------------------------------- scrubbing
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), None),                       # None: drop the whole row
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "<secret>"), (re.compile(r"\b(?:ghp|gho|ghs|ghu|ghr)_[A-Za-z0-9]{20,}"), "<secret>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), "<secret>"), (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), "<secret>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<secret>"), (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "<secret>"),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "<secret>"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{12,}=*"), "Bearer <secret>"),
    (re.compile(r"(?i)\b([A-Za-z0-9_\-]*(?:api[_-]?key|token|secret|passw(?:or)?d|passwd|pwd|auth|credential)[A-Za-z0-9_\-]*)(\s*[:=]\s*)(\"[^\"\n]*\"|'[^'\n]*'|[^\s\"',;]+)"), r"\1\2<secret>"),
    (re.compile(r"(?i)(--?(?:api[_-]?key|token|password|secret|auth)[= ])(\S+)"), r"\1<secret>"),
    (re.compile(r"://[^/\s:@]+:[^/\s@]+@"), "://<user>:<secret>@"),
]
GENERIC = [(re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), "<email>"), (re.compile(r"(https?://[^\s?#]+)\?[^\s#]*"), r"\1?<q>"),
           (re.compile(re.escape(HOME) + r"(?=/|\b)"), "~"), (re.compile(r"/Users/[A-Za-z0-9._\-]+"), "/Users/<u>"),
           (re.compile(r"\b(?=[A-Za-z0-9+/_\-]*[A-Za-z])(?=[A-Za-z0-9+/_\-]*\d)[A-Za-z0-9+/_\-]{32,}={0,2}"), "<secret>"),
           (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<hex>")]
DROP_MARKERS = re.compile(r"(?i)(private key|\bssh-rsa\b|\bssh-ed25519\b|BEGIN CERTIFICATE|aws_secret|client_secret)")


class Scrubber:
    def __init__(self, literals=()):
        self.literals = [l for l in literals if l and len(l) >= 8]
        self.counts = collections.Counter()

    def __call__(self, text):
        """-> scrubbed text, or None when the row must be dropped."""
        if any(l in text for l in self.literals):
            self.counts["dropped: literal secret"] += 1
            return None
        for pat, rep in SECRET_PATTERNS:
            if rep is None:
                if pat.search(text):
                    self.counts["dropped: private key"] += 1
                    return None
                continue
            text, n = pat.subn(rep, text)
            self.counts["redacted: " + rep.replace(r"\1\2", "kv").replace(r"\1", "")[:14]] += n
        for pat, rep in GENERIC:
            text, n = pat.subn(rep, text)
            self.counts["generic: " + rep[:8]] += n
        if DROP_MARKERS.search(text):
            self.counts["dropped: marker"] += 1
            return None
        return text


# ---------------------------------------------------------------- transcripts
REMINDER = re.compile(r"<(system-reminder|task-notification|command-[a-z-]+|local-command-[a-z-]+|ide_[a-z_]+|user-prompt-submit-hook)>.*?</\1>", re.S)


def text_of(content):
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def short(s, n):
    return " ".join(s.split())[:n]


def brief(name, inp):
    inp = inp or {}
    if name == "Bash":
        return "Bash: " + short(inp.get("command", ""), 50)
    if name in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit"):
        return f"{name} {os.path.basename(str(inp.get('file_path', '')))}"
    if name == "WebFetch":
        m = re.match(r"https?://([^/]+)", str(inp.get("url", "")))
        return f"WebFetch {m.group(1) if m else ''}"
    if name in ("WebSearch", "ToolSearch"):
        return f"{name} {short(str(inp.get('query', '')), 40)}"
    if name in ("Agent", "Task"):
        return f"Agent {short(str(inp.get('description', '')), 40)}"
    return name


def result_summary(block):
    c = block.get("content")
    t = text_of(c) if isinstance(c, (str, list)) else ""
    return ("error: " if block.get("is_error") else "") + (short(t, 150) or "ok")


def rows_from(path, scrub, stats):
    state = collections.defaultdict(lambda: {"first": "", "last": "", "calls": [], "result": "", "since_user": 0})
    text_by_msg = collections.defaultdict(str)
    session = hashlib.sha1(path.encode()).hexdigest()[:10]
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        st = state[bool(d.get("isSidechain"))]
        c = m.get("content")
        if d.get("type") == "user":
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        st["result"] = result_summary(b)
                continue
            t = short(REMINDER.sub(" ", text_of(c) if c is not None else ""), 400)
            if t and not t.startswith("[Request interrupted"):
                st["first"] = st["first"] or t
                st["last"] = t
                st["since_user"] = 0
        elif d.get("type") == "assistant" and isinstance(c, list):
            mid = m.get("id") or d.get("requestId")
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    text_by_msg[mid] += b.get("text", "")
                elif b.get("type") == "tool_use":
                    name, inp = b.get("name"), b.get("input") or {}
                    label = label_of(name, inp)
                    stats["calls"] += 1
                    if not st["last"] and not st["first"]:
                        stats["no user prompt"] += 1
                    else:
                        narration = short(text_by_msg.get(mid, ""), 200)
                        parts = {"task": st["first"][:300], "latest_user": st["last"][:300],
                                 "previous_steps": "; ".join(st["calls"][-3:]), "last_result": st["result"], "narration": narration}
                        first_after_user = st["since_user"] == 0
                        clean = {k: scrub(v) for k, v in parts.items()}
                        if any(v is None for v in clean.values()):
                            stats["rows dropped by the scrubber"] += 1
                        else:
                            yield {"session": session, "sidechain": bool(d.get("isSidechain")), "ts": d.get("timestamp"), "tool": name, "label": label,
                                   "first_after_user": first_after_user, **clean}
                    st["calls"].append(brief(name, inp))
                    st["since_user"] += 1


def selftest():
    s = Scrubber(literals=["LITERAL-SECRET-VALUE-123"])
    cases = {
        "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456": "sk-abc", "token: ghp_abcdefghijklmnopqrstuvwxyz0123456789": "ghp_abc",
        "Authorization: Bearer abcdef0123456789abcdef.xyz": "abcdef0123456789", "password = 'hunter2hunter2'": "hunter2",
        "mail me at someone@example.com": "someone@", f"{HOME}/work/secret-project/x.py": HOME, "curl https://api.x.com/v1?key=ABC123&u=me": "key=ABC123",
        "AKIAABCDEFGHIJKLMNOP is the id": "AKIAABCDEFGHIJKLMNOP", "postgres://admin:s3cr3tpw@db.host/x": "s3cr3tpw",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk": "eyJhbGci", "--api-key abc123def456ghi": "abc123def456ghi",
        "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8": "a1b2c3d4e5f6a7b8c9d0e1f2"}
    for text, leaked in cases.items():
        out = s(text)
        assert out is not None and leaked not in out, f"leaked {leaked!r} in {out!r}"
    for text in ("-----BEGIN RSA PRIVATE KEY-----", "has LITERAL-SECRET-VALUE-123 inside", "my ssh-ed25519 AAAA"):
        assert s(text) is None, text
    assert s("Edit the file src/foo.py and run the tests") == "Edit the file src/foo.py and run the tests"
    assert label_of("Bash", {"command": "cd x && git status"}) == "Bash:git" and label_of("Bash", {"command": "FOO=1 python3 a.py"}) == "Bash:python"
    assert label_of("mcp__Claude_Browser__navigate", {}) == "mcp:browser" and label_of("Read", {}) == "Read" and label_of("Frob", {}) == "other"
    print("selftest ok:", len(cases) + 3, "scrubber cases and the label rules")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?"); ap.add_argument("--skip-session", action="append", default=[]); ap.add_argument("--secret-file", action="append", default=[])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out:
        sys.exit(__doc__)
    literals = [Path(p).read_text().strip() for p in a.secret_file if Path(p).exists()]
    scrub, stats = Scrubber(literals), collections.Counter()
    skip = {os.path.realpath(p) for p in a.skip_session}
    files = sorted(f for f in glob.glob(os.path.join(HOME, ".claude/projects/**/*.jsonl"), recursive=True) if os.path.realpath(f) not in skip)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seen, kept = set(), 0
    with open(out / "calls.jsonl", "w") as f:
        for i, p in enumerate(files):
            for r in rows_from(p, scrub, stats):
                key = (r["label"], r["task"], r["latest_user"], r["previous_steps"], r["last_result"], r["narration"])
                if key in seen:
                    stats["exact duplicates dropped"] += 1
                    continue
                seen.add(key)
                f.write(json.dumps(r) + "\n"); kept += 1
    os.chmod(out / "calls.jsonl", 0o600)
    labels = collections.Counter(json.loads(l)["label"] for l in open(out / "calls.jsonl"))
    print(f"{len(files)} transcript files, {stats['calls']} tool calls, {kept} rows kept")
    print({k: v for k, v in stats.items() if k != "calls"})
    print("scrubber:", dict(scrub.counts.most_common(14)))
    print(f"{len(labels)} labels:", labels.most_common(40))


if __name__ == "__main__":
    main()
