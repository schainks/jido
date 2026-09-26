"""The task set: two slices, every task graded without a judge model.

- tool (34): the requests from ../jev_routing/jev_eval.py over Jido's 19 actions.
  The model gets the actions as tools and the selection-only prompt from
  llm_baseline.py; correct = it calls a gold tool (or no tool when gold is "none").
- open (34): self-contained tasks with one checkable answer, from lookups to puzzles.
  The model ends its reply with `ANSWER: ...`; a deterministic grader checks it.

`band` is a prior guess at difficulty, used only to break down results. The labels
come from running every model. Stdlib only.
"""
import json, re
from fractions import Fraction

import common  # noqa: F401  (puts ../jev_routing on sys.path)
from jev_eval import GOLD, extract_actions

# Identical to ../jev_routing/llm_baseline.py (selftest.py checks this when anthropic is installed).
SELECT = ("You are the tool-selection step of a Jido agent. Your only job is to decide which one of your tools, "
          "if any, the request maps to, and call it. Missing parameters are fine: call the tool with whatever "
          "parameters you can infer and leave the rest out; do not ask clarifying questions. Only if no tool fits "
          "the request at all, reply in text without calling a tool.")


def tools_from_actions(acts):
    out = []
    for a in acts:
        props = {k: {"type": "string", "description": d or k} for k, d in a["params"]}
        out.append({"name": a["name"], "description": a["description"],
                    "input_schema": {"type": "object", "properties": props}})
    return out


OPEN_SYSTEM = ("Solve the user's task. Take as much care as the task needs. End your reply with one final line "
               "of the form `ANSWER: <your answer>`, with just the answer on that line and nothing after it.")

AGENT_CONTEXT = "The agent is a Jido AgentServer process with a parent, possibly child agents, and scheduled jobs."
OPEN_CONTEXT = "A general assistant request. The model answers in text and has no tools."


# ---------------------------------------------------------------- answer parsing
_ANSWER_RE = re.compile(r"^[\s>*_`#]*answer[\s*_`]*[:：](.*)$", re.I | re.M)


def parse_answer(text):
    """Text after the last `ANSWER:` marker (or the next line if that is empty), else the last line."""
    lines = text.splitlines()
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]
        rest = m.group(1).strip()
        if not rest:
            after = text[m.end():].splitlines()
            rest = next((ln.strip() for ln in after if ln.strip() and not ln.strip().startswith("```")), "")
        return clean(rest)
    tail = [ln for ln in lines if ln.strip()]
    return clean(tail[-1]) if tail else ""


def clean(s):
    """Drop markdown wrappers (**x**, `x`, "x") but never a bare `*`, which cron answers need."""
    s = s.strip()
    if s.startswith("**") and s[2:3].isspace():  # closing half of **ANSWER:**
        s = s[2:].strip()
    for _ in range(3):
        for mark in ("**", "__", "`", '"', "'", "*", "_"):
            n = len(mark)
            if (len(s) > 2 * n and s.startswith(mark) and s.endswith(mark)
                    and not s[n].isspace() and not s[-n - 1].isspace()):
                s = s[n:-n].strip()
                break
        else:
            break
    return s


def norm(s):
    return re.sub(r"\s+", " ", s.strip().strip("`*\"'").rstrip(".").strip()).casefold()


# ---------------------------------------------------------------- graders: g(answer, full_text) -> bool
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?|[-+]?\.\d+")


def first_number(s):
    m = _NUM_RE.search(s.replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def exact(*accepted):
    """Equal after normalizing, or followed only by a qualifier: "Wednesday, 19 Feb" passes, "Wolfram" is not "W"."""
    ok = [norm(a) for a in accepted]

    def g(ans, text):
        n = norm(ans)
        return any(n == a or (n.startswith(a) and not n[len(a)].isalnum()) for a in ok)
    return g


def number(target, tol=1e-6):
    def g(ans, text):
        x = first_number(ans)
        return x is not None and abs(x - target) <= tol
    return g


def signed_percent(target):
    def g(ans, text):
        x = first_number(ans)
        if x is None:
            return False
        if x > 0 and re.search(r"\b(decrease|down|drop|fall|loss|lower)\b", ans, re.I) and not ans.lstrip().startswith("+"):
            x = -x  # "1.0% decrease"
        return abs(x - target) < 0.05
    return g


def fraction(target):
    def g(ans, text):
        m = re.search(r"(-?\d+)\s*/\s*(\d+)", ans)
        if m:
            return int(m.group(2)) != 0 and Fraction(int(m.group(1)), int(m.group(2))) == target
        x = first_number(ans)
        return x is not None and abs(x - float(target)) < 1e-4
    return g


def iso_date(target):
    def g(ans, text):
        m = re.search(r"\d{4}-\d{2}-\d{2}", ans)
        return bool(m) and m.group(0) == target
    return g


def date_list(targets):
    return lambda ans, text: re.findall(r"\d{4}-\d{2}-\d{2}", ans) == list(targets)


def clock(target):
    """HH:MM, 24-hour; also accepts 12-hour with am/pm."""
    th, tm = map(int, target.split(":"))

    def g(ans, text):
        m = re.search(r"(\d{1,2}):(\d{2})\s*([ap]\.?\s?m\.?)?", ans, re.I)
        if not m:
            return False
        h, mi = int(m.group(1)), int(m.group(2))
        if m.group(3):
            pm = m.group(3).lower().startswith("p")
            h = (h % 12) + (12 if pm else 0)
        return (h, mi) == (th, tm)
    return g


def int_list(target):
    return lambda ans, text: [int(x) for x in re.findall(r"-?\d+", ans)] == list(target)


def intervals(target):
    def g(ans, text):
        pairs = re.findall(r"[\[(]\s*(-?\d+)\s*,\s*(-?\d+)\s*[\])]", ans) or re.findall(r"(-?\d+)\s*[-–]\s*(-?\d+)", ans)
        return [(int(a), int(b)) for a, b in pairs] == list(target)
    return g


def _tokens(ans, vocab):
    return [t.casefold() for t in re.findall(r"[A-Za-z][A-Za-z_-]*", ans) if t.casefold() in vocab]


def sequence(target):
    vocab = {t.casefold() for t in target}
    return lambda ans, text: _tokens(ans, vocab) == [t.casefold() for t in target]


def set_of(target, universe):
    vocab = {t.casefold() for t in universe}

    def g(ans, text):
        if norm(ans) in ("none", "no one", "nobody"):
            return not target
        return set(_tokens(ans, vocab)) == {t.casefold() for t in target}
    return g


def json_fields(target):
    """Last JSON object in the whole reply (it may sit in a code block under ANSWER:)."""
    def g(ans, text):
        for blob in reversed(re.findall(r"\{[^{}]*\}", text)):
            try:
                obj = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            return all(_same(obj.get(k), v) for k, v in target.items())
        return False
    return g


def _same(got, want):
    if isinstance(want, (int, float)):
        try:
            return float(got) == float(want)
        except (TypeError, ValueError):
            return False
    return isinstance(got, str) and norm(got) == norm(want)


def topo_order(deps):
    """deps: {job: [prerequisites]}; the answer must list every job once, after its prerequisites."""
    jobs = set(deps) | {p for ps in deps.values() for p in ps}

    def g(ans, text):
        order = _tokens(ans, jobs)
        if sorted(order) != sorted(jobs):
            return False
        pos = {j: i for i, j in enumerate(order)}
        return all(pos[p] < pos[j] for j, ps in deps.items() for p in ps)
    return g


def cron(target):
    """Compares the sets of times two 5-field cron expressions select, so 1-5 == MON-FRI."""
    want = _cron_sets(target)

    def g(ans, text):
        toks = ans.replace("`", " ").split()
        for i in range(len(toks) - 4):
            got = _cron_sets(" ".join(toks[i:i + 5]))
            if got is not None:
                return got == want
        return False
    return g


_MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
_DAYS = {d: i for i, d in enumerate("sun mon tue wed thu fri sat".split())}
_FIELDS = [(0, 59, {}), (0, 23, {}), (1, 31, {}), (1, 12, _MONTHS), (0, 7, _DAYS)]


def _cron_sets(expr):
    parts = expr.split()
    if len(parts) != 5:
        return None
    out = []
    for field, (lo, hi, names) in zip(parts, _FIELDS):
        vals = _cron_field(field.lower(), lo, hi, names)
        if vals is None:
            return None
        if hi == 7:  # day of week: 7 is Sunday too
            vals = {0 if v == 7 else v for v in vals}
        out.append(frozenset(vals))
    return out


def _cron_field(field, lo, hi, names):
    def num(s):
        return names[s[:3]] if s[:3] in names else int(s)

    vals = set()
    try:
        for part in field.split(","):
            step = 1
            if "/" in part:
                part, s = part.split("/", 1)
                step = int(s)
            if part in ("*", "?"):
                a, b = lo, hi
            elif "-" in part:
                a, b = (num(x) for x in part.split("-", 1))
            else:
                a = num(part)
                b = hi if step != 1 else a
            if not (lo <= a <= b <= hi) or step < 1:
                return None
            vals.update(range(a, b + 1, step))
    except (ValueError, KeyError):
        return None
    return vals


REGEX_POS = ["abc-123", "q-007.US", "zz-999.GB", "a-000"]
REGEX_NEG = ["Abc-123", "abc-12", "abc-1234", "abc123", "-123", "abc-123.us", "abc-123.USA",
             "abc-123.", "abc-123 US", "abc-12a", "ab_c-123", "abc-123.U1", "abc-123.US.GB"]


def python_regex(ans, text):
    pat = ans.strip()
    m = re.fullmatch(r"r?(['\"])(.*)\1", pat)
    if m:
        pat = m.group(2)
    if len(pat) > 200:
        return False
    try:
        rx = re.compile(pat)
    except re.error:
        return False
    return all(rx.fullmatch(s) for s in REGEX_POS) and not any(rx.fullmatch(s) for s in REGEX_NEG)


# ---------------------------------------------------------------- open slice
CRON = "Write the standard 5-field cron expression (minute hour day-of-month month day-of-week) for: "

CODE = '''What does f(50) return?

def f(n):
    total = 0
    for i in range(1, n):
        if i % 3 == 0 or i % 5 == 0:
            total += i
        elif i % 7 == 0:
            total -= i
    return total'''

CSV = '''Here are some orders:

order_id,customer,date,qty,unit_price,status
1001,Acme,2026-02-27,3,19.99,shipped
1002,Birch,2026-03-01,1,249.00,shipped
1003,Acme,2026-03-04,10,4.50,refunded
1004,Acme,2026-03-09,2,120.00,shipped
1005,Cobalt,2026-03-15,5,12.00,shipped
1006,Acme,2026-03-22,1,89.95,cancelled
1007,Acme,2026-03-31,4,15.25,shipped
1008,Birch,2026-04-01,2,249.00,shipped
1009,Acme,2026-04-02,6,9.99,shipped

What is Acme's total revenue (qty times unit_price) from shipped orders dated in March 2026?'''

GRID = '''Four people (Ana, Ben, Cai, Dee) live in four houses in a row, numbered 1 to 4 from left to right. Each keeps a different pet (cat, dog, fish, bird) and drinks a different drink (tea, coffee, milk, water).
1. Ben drinks milk.
2. The dog lives in house 1.
3. Ana lives directly to the right of the cat's owner.
4. Coffee is drunk in house 4.
5. Dee lives next to the tea drinker.
6. The bird's owner drinks water.
7. Cai does not live in house 4.
8. The fish lives in a house to the right of Ben's house.
9. Ben does not live in house 1.
List the four people in house order, from house 1 to house 4, comma-separated.'''

KNIGHTS = ("On an island, knights always tell the truth and knaves always lie. A says: \"Exactly one of us three "
           "is a knight.\" B says: \"Exactly two of us three are knights.\" C says: \"At least one of A and B is a "
           "knave.\" Which of A, B and C are knights? Answer with their letters, comma-separated, or none.")

SLOTS = ("Five talks A, B, C, D and E fill slots 1 to 5, one talk per slot. C is before B. B is before D. A and C "
         "are in adjacent slots. E is in neither the first nor the last slot. D is exactly two slots after E. A is "
         "not in slot 1. List the talks from slot 1 to slot 5, comma-separated.")

KNAPSACK = ("A hiker can carry at most 20 kg. Items as (weight in kg, value): tent (11, 30), stove (4, 10), "
            "camera (3, 14), rope (5, 12), radio (6, 18), lamp (2, 7), book (1, 3), first-aid kit (7, 20). "
            "Each item can be taken at most once. What is the maximum total value that fits?")

GRAPH = ("An undirected graph has these weighted edges: A-B 4, A-C 2, B-C 5, B-D 10, C-E 3, E-D 4, D-F 11, "
         "E-F 12, C-F 16, B-F 15. What is the length of the shortest path from A to F?")

TOPO_DEPS = {"build": ["fetch"], "test": ["build"], "lint": ["fetch"], "package": ["test", "lint"],
             "deploy": ["package"], "docs": ["fetch"]}

# (id, band, prompt, grader, a correct reply's answer, a wrong one)
OPEN = [
    ("email", "easy", "Extract the email address from this message: \"Ping Dana at dana.ortiz@example.org or call 555-0142 after 3pm.\"",
     exact("dana.ortiz@example.org"), "dana.ortiz@example.org", "dana.ortiz@example.com"),
    ("seconds", "easy", "How many seconds are there in 2 hours and 45 minutes?", number(9900), "9900", "9600"),
    ("symbol", "easy", "What is the chemical symbol for tungsten?", exact("W"), "W", "Tu"),
    ("iso_date", "easy", "Rewrite the date 'March 7th, 2026' in ISO 8601 format (YYYY-MM-DD).",
     iso_date("2026-03-07"), "2026-03-07", "2026-07-03"),
    ("json_value", "easy", "What is the value of `retries` in this JSON?\n{\"name\": \"sync\", \"opts\": {\"retries\": 3, \"backoff\": \"exp\"}, \"retries_total\": 12}",
     number(3), "3", "12"),
    ("snake_case", "easy", "Convert the identifier 'maxRetryCount' to snake_case.", exact("max_retry_count"), "max_retry_count", "max_retrycount"),
    ("sentiment", "easy", "Classify the sentiment of this review as positive, negative, or mixed: \"The battery lasts forever and the camera is superb, but the screen scratched on day one.\"",
     exact("mixed"), "mixed", "positive"),
    ("cron_daily", "easy", CRON + "every day at 2:30 AM.", cron("30 2 * * *"), "30 2 * * *", "2 30 * * *"),
    ("cron_weekdays", "medium", CRON + "9:15 AM, Monday through Friday.", cron("15 9 * * 1-5"), "15 9 * * MON-FRI", "15 9 * * 0-4"),
    ("cron_complex", "medium", CRON + "every 15 minutes from 9:00 AM through 5:45 PM, Monday through Friday, in January and July only.",
     cron("*/15 9-17 * 1,7 1-5"), "0,15,30,45 9-17 * JAN,JUL 1-5", "*/15 9-17 * 1-7 1-5"),
    ("extract_json", "medium", "Return a JSON object with the keys name, age and current_city for the person described here: \"Our newest hire, Priya Raman (29), relocated from Pune to Austin last spring.\"",
     json_fields({"name": "Priya Raman", "age": 29, "current_city": "Austin"}),
     '{"name": "Priya Raman", "age": 29, "current_city": "Austin"}', '{"name": "Priya Raman", "age": 29, "current_city": "Pune"}'),
    ("topo_order", "medium", "Jobs and their prerequisites: build needs fetch; test needs build; lint needs fetch; package needs test and lint; deploy needs package; docs needs fetch. Give one order that runs every job after all of its prerequisites, as a comma-separated list.",
     topo_order(TOPO_DEPS), "fetch, build, lint, docs, test, package, deploy", "fetch, test, build, lint, package, deploy, docs"),
    ("percent", "medium", "A price rises 20%, then falls 25%, then rises 10%. What is the net percentage change from the original price? Answer as a signed percentage with one decimal place, like +2.5% or -2.5%.",
     signed_percent(-1.0), "-1.0%", "+5.0%"),
    ("intervals", "medium", "Merge the overlapping intervals and list the result in ascending order as [start,end] pairs: [8,10], [1,4], [15,18], [2,6], [21,22], [9,12], [17,20]",
     intervals([(1, 6), (8, 12), (15, 20), (21, 22)]), "[1,6], [8,12], [15,20], [21,22]", "[1,6], [8,12], [15,22]"),
    ("csv_revenue", "medium", CSV, number(301.00, tol=0.005), "$301.00", "$390.95"),
    ("list_trace", "medium", "Start with the Python list [5, 3, 8, 1]. Apply in order: sort ascending; append 4; reverse; remove the value 3; insert 9 at index 1; pop the last element. What is the final list?",
     int_list([4, 9, 8, 5]), "[4, 9, 8, 5]", "[4, 9, 8, 5, 1]"),
    ("date_add", "medium", "What date is 100 days after 2026-11-20? Answer as YYYY-MM-DD.", iso_date("2027-02-28"), "2027-02-28", "2027-03-01"),
    ("business_days", "medium", "How many weekdays (Monday to Friday) are there from 2026-03-02 through 2026-03-31, counting both ends?",
     number(22), "22", "21"),
    ("regex", "medium", "Write a Python regular expression that matches a whole string consisting of: one or more lowercase ASCII letters, a hyphen, exactly three digits, and optionally a dot followed by exactly two uppercase ASCII letters. It must match abc-123 and q-007.US, and must not match Abc-123, abc-12, abc-123.us or abc-123.USA. Give only the pattern as the answer.",
     python_regex, "[a-z]+-[0-9]{3}(\\.[A-Z]{2})?", "[a-z]+-\\d+(\\.[A-Z]+)?"),
    ("bit_count", "medium", "How many 1 bits are in the binary representation of 2026?", number(8), "8", "7"),
    ("weekday", "hard", "What day of the week is 19 February 2031?", exact("Wednesday", "Wed"), "Wednesday", "Thursday"),
    ("train", "hard", "A train departs at 14:05. It covers 212 km at an average of 80 km/h, stops for 12 minutes, then covers 95 km at an average of 60 km/h. At what time does it arrive? Answer as HH:MM in 24-hour time.",
     clock("18:31"), "18:31", "18:19"),
    ("code_trace", "hard", CODE, number(445), "445", "543"),
    ("mod_pow", "hard", "What are the last two digits of 7^2026?", number(49), "49", "43"),
    ("dice", "hard", "Two fair six-sided dice are rolled. What is the probability that the product of the two numbers is a multiple of 6? Answer as a reduced fraction.",
     fraction(Fraction(5, 12)), "5/12", "1/3"),
    ("no_adjacent", "hard", "How many 7-character strings of decimal digits (0-9, leading zeros allowed) contain exactly three 7s, with no two 7s next to each other?",
     number(65610), "65610", "229635"),
    ("second_tuesdays", "hard", "A meeting is held on the second Tuesday of every month. List its dates from January through June 2027 as comma-separated YYYY-MM-DD.",
     date_list(["2027-01-12", "2027-02-09", "2027-03-09", "2027-04-13", "2027-05-11", "2027-06-08"]),
     "2027-01-12, 2027-02-09, 2027-03-09, 2027-04-13, 2027-05-11, 2027-06-08",
     "2027-01-12, 2027-02-09, 2027-03-09, 2027-04-06, 2027-05-11, 2027-06-08"),
    ("timezone", "hard", "A call is scheduled for 09:30 on 29 March 2026, London time. What time is that in New York? Answer as HH:MM in 24-hour time.",
     clock("04:30"), "04:30", "05:30"),
    ("knapsack", "hard", KNAPSACK, number(62), "62", "60"),
    ("shortest_path", "hard", GRAPH, number(17), "17", "18"),
    ("cryptarithm", "hard", "In the addition BASE + BALL = GAMES, each letter stands for a different digit and no number starts with 0. What number is GAMES?",
     number(14938), "14938", "14928"),
    ("logic_grid", "hard", GRID, sequence(["Cai", "Dee", "Ben", "Ana"]), "Cai, Dee, Ben, Ana", "Ben, Dee, Cai, Ana"),
    ("knights", "hard", KNIGHTS, set_of({"B", "C"}, "ABC"), "B, C", "A"),
    ("talk_slots", "hard", SLOTS, sequence(["C", "A", "E", "B", "D"]), "C, A, E, B, D", "A, C, E, B, D"),
]


def load_tasks():
    acts = extract_actions()
    names = [a["name"] for a in acts]
    tasks = []
    for i, (query, gold, needs_tool) in enumerate(GOLD, 1):
        tasks.append({"id": f"tool{i:02d}", "slice": "tool", "band": "tool", "prompt": query,
                      "gold": sorted(gold), "needs_tool": needs_tool,
                      "jev_state": {"request": query, "available_actions": names, "context": AGENT_CONTEXT}})
    for tid, band, prompt, grader, ref, wrong in OPEN:
        tasks.append({"id": f"open_{tid}", "slice": "open", "band": band, "prompt": prompt, "grader": grader,
                      "ref": ref, "wrong": wrong,
                      "jev_state": {"request": prompt, "available_actions": [], "context": OPEN_CONTEXT}})
    return tasks, acts


def grade(task, tool_calls, text):
    if task["slice"] == "tool":
        got = tool_calls[0] if tool_calls else "none"
        return got in task["gold"], got
    ans = parse_answer(text)
    return bool(task["grader"](ans, text)), ans
