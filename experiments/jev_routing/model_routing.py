#!/usr/bin/env python3
"""Can Jev pick the cheapest model tier that will get a task right?

Ground truth is outcome-based: every task runs on every tier, is graded
(exact match, unit tests for code, or an Opus 5 judge for open tasks), and the
gold tier is the cheapest one that passes. Then several routing policies are
scored against that gold without re-running the models:

  always-fast / always-capable / always-reasoning   (floor and ceiling)
  heuristic     keyword + length rule in the spirit of jido_ai's AdaptiveAgent
  jev-choice    Jev Choice over the three tiers from the request alone
  jev-composed  code policy over Jev's reasoning-depth Score
  jev-cascade   run fast, ask Jev "is this answer wrong", escalate; sweep threshold

Tiers here are the Claude models that are byte-identical on Vertex AI and the
Anthropic API (claude-haiku-4-5, claude-sonnet-5, claude-opus-5). Gemini tiers
need Google ADC and are not run by this script.

Keys: ANTHROPIC_API_KEY or ~/.anthropic_key; TYPESAFE_API_KEY or ~/.typesafe_key.
Run: .venv/bin/python model_routing.py            (needs the anthropic package)
     .venv/bin/python model_routing.py --regrade  (offline: regrade stored replies)
Cost per full run: ~120 model calls, ~12 judge calls, ~120 Jev calls.
"""
import json, os, re, statistics, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import anthropic
from jev_eval import call as jev_call, MODEL as JEV_MODEL, OUT

TIERS = [("fast", "claude-haiku-4-5", 1.0, 5.0),
         ("capable", "claude-sonnet-5", 2.0, 10.0),
         ("reasoning", "claude-opus-5", 5.0, 25.0)]
TIER_RANK = {t[0]: i for i, t in enumerate(TIERS)}
JUDGE_MODEL = "claude-opus-5"
ANSWER_ONLY = " Reply with only the final answer, no explanation."
CODE_ONLY = " Reply with only the Python code in a single ```python block, no explanation."

# ------------------------------------------------------------------ tasks
# kind: exact (any alternative must appear in the normalized reply)
#       code  (extract ```python block, run TESTS)
#       judge (Opus 5 grades against RUBRIC)
T = []
def exact(q, *alts): T.append({"kind": "exact", "q": q + ANSWER_ONLY, "alts": [str(a) for a in alts]})
def code(q, tests): T.append({"kind": "code", "q": q + CODE_ONLY, "tests": tests})
def judge(q, rubric): T.append({"kind": "judge", "q": q, "rubric": rubric})

exact("What is 17 * 23?", 391)
exact("Convert 98.6 degrees Fahrenheit to Celsius, one decimal place.", "37.0", "37")
exact("Extract the email address from: 'Contact: Jane Doe <jane.doe@example.org>, ext 42'.", "jane.doe@example.org")
exact("How many days are in February 2024?", 29)
exact("What is the ISO 8601 date of the day after 2024-02-28?", "2024-02-29")
exact("Reverse the string 'jido'.", "odij")
exact("Sort these numbers ascending: 42, 7, 19, 3.", "3, 7, 19, 42", "3 7 19 42", "[3, 7, 19, 42]")
exact('Given the JSON {"a":{"b":[1,2,{"c":"x"}]}}, what is the value at a.b[2].c?', "x")
exact("Which number is larger, 9.11 or 9.9?", "9.9")
exact("How many times does the letter r appear in the word strawberry?", 3)
exact("How many words are in the sentence 'the quick brown fox jumps'?", 5)
exact("A train departs at 14:35 and the journey takes 2 hours 50 minutes. What is the arrival time in 24-hour format?", "17:25")
exact("What is the sum of all integers from 1 to 100 inclusive that are divisible by 3 or 5?", 2418)
exact("Two trains 300 km apart travel toward each other at 70 km/h and 80 km/h. After how many hours do they meet?", 2)
exact("Base64-decode the string aGVsbG8gamlkbw==", "hello jido")
exact("What is 2 to the power of 20?", 1048576, "1,048,576")
exact("In Elixir, is version 2.9.0 allowed by the requirement ~> 2.3? Is 3.0.0 allowed? Answer with two words, yes or no for each, in order.", "yes no", "yes, no")
exact("What is the greatest common divisor of 1071 and 462?", 21)
exact("In Elixir, what does Enum.reduce([1, 2, 3], 0, &+/2) return?", 6)
exact("How many trailing zeros does 25! (25 factorial) have?", 6)
exact("What is the sum of the decimal digits of 2 to the power of 15?", 26)
exact("How many positive integers less than 1000 are divisible by neither 5 nor 7?", 686)
exact("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?", 5)
exact("At what time between 3:00 and 4:00 are the hour and minute hands of a clock exactly together? Give minutes past 3:00 to two decimal places.", "16.36")
exact("What is the last digit of 7 to the power of 222?", 9)
exact("How many distinct arrangements are there of the letters in MISSISSIPPI?", 34650, "34,650")
exact("Solve for x: 3^(2x) = 81^(x-1).", 2, "x = 2", "x=2")
exact("Monty Hall with 4 doors: one car, three goats. You pick a door, the host opens one other door showing a goat, and offers a switch to one of the two remaining closed doors. What is the probability of winning by switching? Give a fraction.", "3/8", "0.375")
exact("What is the next term in the sequence 2, 6, 12, 20, 30?", 42)
exact("How many three-digit palindromes are divisible by 11?", 8)  # aba: b = 2a mod 11, a=5 needs b=10, so 8 not 9
exact("A number leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 2 when divided by 7. What is the smallest positive such number?", 23)
exact("If it takes 5 machines 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?", 5)

code("Write a Python function is_balanced(s) that returns True if the brackets (), [], {} in s are balanced and properly nested, else False.",
     "assert is_balanced('([]{})') is True; assert is_balanced('([)]') is False; assert is_balanced('') is True; assert is_balanced('((') is False; assert is_balanced('a(b)c') is True")
code("Write a Python function roman_to_int(s) converting a Roman numeral string to an integer.",
     "assert roman_to_int('III') == 3; assert roman_to_int('IV') == 4; assert roman_to_int('MCMXCIV') == 1994; assert roman_to_int('LVIII') == 58")
code("Write a Python function merge_intervals(intervals) that takes a list of [start, end] pairs and returns the merged, sorted list of non-overlapping intervals.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]; assert merge_intervals([[1,4],[4,5]]) == [[1,5]]; assert merge_intervals([]) == []; assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]")
code("Write a Python function parse_cron_field(field, lo, hi) that expands one cron field into a sorted list of ints within [lo, hi]. Support '*', a single value, 'a-b', comma lists, '*/n', and 'a-b/n'.",
     "assert parse_cron_field('*', 0, 5) == [0,1,2,3,4,5]; assert parse_cron_field('3', 0, 5) == [3]; assert parse_cron_field('1-3', 0, 5) == [1,2,3]; assert parse_cron_field('1,4', 0, 5) == [1,4]; assert parse_cron_field('*/2', 0, 5) == [0,2,4]; assert parse_cron_field('1-5/2', 0, 5) == [1,3,5]; assert parse_cron_field('0,1-2,*/5', 0, 10) == [0,1,2,5,10]")
code("Write a Python function longest_palindrome(s) returning the longest palindromic substring of s (return the leftmost on ties).",
     "assert longest_palindrome('babad') in ('bab','aba'); assert longest_palindrome('cbbd') == 'bb'; assert longest_palindrome('a') == 'a'; assert longest_palindrome('forgeeksskeegfor') == 'geeksskeeg'")
code("Write a Python function topo_sort(n, edges) where nodes are 0..n-1 and edges is a list of (u, v) meaning u must come before v. Return a valid ordering as a list, or raise ValueError if there is a cycle.",
     "o = topo_sort(4, [(0,1),(1,2),(0,3),(3,2)]); assert o.index(0) < o.index(1) < o.index(2) and o.index(3) < o.index(2)\ntry:\n    topo_sort(2, [(0,1),(1,0)]); assert False\nexcept ValueError:\n    pass\nassert sorted(topo_sort(3, [])) == [0,1,2]")

judge("In no more than three sentences, explain why an agent framework would want its core command function to be pure (same input, same output) and push side effects into separate descriptors.",
      "Passes if it mentions at least two of: testability/determinism, replayability or reasoning about state, isolating failures of side effects, and it stays within three sentences.")
judge("Summarize in one sentence: 'Jido separates decision from execution. An agent's cmd/2 returns a new agent state and a list of directives; the AgentServer runtime then executes the directives, such as emitting signals or spawning children, and feeds results back as new signals.'",
      "Passes if it is one sentence and conveys that cmd/2 decides (returns state plus directives) while the runtime executes the directives.")
judge("Write a polite two-sentence reply declining a meeting invitation for Thursday and proposing next Monday instead.",
      "Passes if it is exactly two sentences, declines Thursday politely, and proposes Monday.")
judge("A user writes: 'my deploy failed with exit code 137'. Reply in at most two sentences with the most likely cause and one concrete thing to check.",
      "Passes if it identifies out-of-memory / OOM kill (SIGKILL) as the likely cause and suggests checking memory limits or logs, in at most two sentences.")

# ------------------------------------------------------------------ grading
def norm(s):
    s = re.sub(r"[*_`#\"']+", "", s)  # markdown emphasis/headers, quotes
    return re.sub(r"[\s$,.]+", " ", s.strip().lower()).strip()

def grade_exact(t, reply):
    r = norm(reply)
    return any(norm(a) == r or f" {norm(a)} " in f" {r} " for a in t["alts"])

def grade_code(t, reply):
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    src = (m.group(1) if m else reply) + "\n\n" + t["tests"] + "\nprint('OK')\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"; p.write_text(src)
        try:
            out = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=10, cwd=d)
            return out.returncode == 0 and "OK" in out.stdout
        except subprocess.TimeoutExpired:
            return False

def grade_judge(client, t, reply):
    r = client.messages.create(model=JUDGE_MODEL, max_tokens=64,
        system="You grade a model's reply against a rubric. Output exactly PASS or FAIL.",
        messages=[{"role": "user", "content": f"Task:\n{t['q']}\n\nRubric:\n{t['rubric']}\n\nReply to grade:\n{reply}"}])
    return "PASS" in "".join(b.text for b in r.content if b.type == "text").upper()

# ------------------------------------------------------------------ model runs
def key(name, envvar):
    p = Path.home() / name
    return os.environ.get(envvar) or (p.read_text().strip() if p.exists() else None)

def run_tier(client, tier, model, pin, pout):
    def one(t):
        t0 = time.perf_counter()
        r = client.messages.create(model=model, max_tokens=16000, messages=[{"role": "user", "content": t["q"]}])
        dt = time.perf_counter() - t0
        text = "".join(b.text for b in r.content if b.type == "text")
        if t["kind"] == "exact": ok = grade_exact(t, text)
        elif t["kind"] == "code": ok = grade_code(t, text)
        else: ok = grade_judge(client, t, text)
        return {"pass": ok, "latency_s": dt, "reply": text, "stop_reason": r.stop_reason,
                "cost_usd": (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6,
                "out_tokens": r.usage.output_tokens}
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, T))

# ------------------------------------------------------------------ Jev
ROUTE_Q = {
    "tier": {"type": "choice",
        "instructions": "Which model tier is the cheapest one that will reliably get `request` right?",
        "criteria": {
            "fast": "Direct lookup, formatting, extraction, one-step arithmetic, short rewrite; a small fast model gets it right",
            "capable": "Several dependent steps, code with edge cases, or careful reading; a mid-size model is needed",
            "reasoning": "Long chain of dependent steps, a trap where the obvious answer is wrong, tricky math or probability, or algorithm design; only a top reasoning model is reliable"}},
    "depth": {"type": "score",
        "instructions": "How much reasoning does `request` need?",
        "criteria": ["The answer is directly retrievable or a single operation",
                     "A few dependent steps that a careful person does in their head",
                     "A long chain of dependent steps, or a well-known trap where the intuitive answer is wrong"]},
    "needs_code": {"type": "noul", "instructions": "Does `request` ask for code to be written?"},
    "small_model_error_prone": {"type": "noul",
        "instructions": "Is `request` the kind of question that small language models commonly get wrong?",
        "criteria": {"true": "Counting letters, comparing decimals, multi-step arithmetic, probability puzzles, modular arithmetic, or code with many edge cases",
                     "false": "Simple recall, formatting, extraction, or short writing"}},
}
VERIFY_Q = {"wrong": {"type": "noul",
    "instructions": "Is `answer` incorrect or incomplete for `request`?",
    "criteria": {"true": "The answer is wrong, missing part of what was asked, or violates a stated format constraint",
                 "false": "The answer fully and correctly satisfies the request"}}}

def jev_route(t):
    resp, dt = jev_call({"state": {"request": t["q"]}, "model": JEV_MODEL, "questions": ROUTE_Q})
    a = resp["answers"]
    return {"tier": a["tier"]["choice"], "tier_conf": a["tier"]["confidence"], "tier_probs": a["tier"]["probabilities"],
            "depth": a["depth"]["score"], "needs_code": a["needs_code"]["noul"],
            "error_prone": a["small_model_error_prone"]["noul"], "latency_s": dt}

def jev_verify(t, answer):
    resp, dt = jev_call({"state": {"request": t["q"], "answer": answer}, "model": JEV_MODEL, "questions": VERIFY_Q})
    return {"p_wrong": resp["answers"]["wrong"]["noul"], "latency_s": dt}

# ------------------------------------------------------------------ policies
KW = ["prove", "probability", "algorithm", "edge case", "cycle", "palindrom", "arrange", "power of", "divisible", "remainder", "clock"]
def heuristic(t):  # in the spirit of Jido.AI.Reasoning.Adaptive: length + keywords
    q = t["q"].replace(ANSWER_ONLY, "").replace(CODE_ONLY, "").lower(); words = len(q.split())
    sentences = len([x for x in re.split(r"[.?!]+", q) if x.strip()])
    score = min(words / 100, 0.3) + min(sentences / 5, 0.2) + min(0.1 * sum(k in q for k in KW), 0.3) + (0.1 if "?" in q else 0)
    return "fast" if score < 0.3 else ("reasoning" if score > 0.7 else "capable")

def composed(j):
    return "fast" if j["depth"] < 0.75 else ("capable" if j["depth"] < 1.5 else "reasoning")

def score_policy(chosen, results, gold):
    """chosen: list of tier per task. results[tier][i] has pass/cost."""
    rows = []
    for i, tier in enumerate(chosen):
        r = results[tier][i]
        rows.append({"pass": r["pass"], "cost": r["cost_usd"], "tier": tier, "gold": gold[i],
                     "under": (not r["pass"]) and gold[i] is not None,
                     "over": gold[i] is not None and TIER_RANK[tier] > TIER_RANK[gold[i]]})
    n = len(rows)
    return {"pass_rate": sum(r["pass"] for r in rows) / n, "cost_per_task_usd": sum(r["cost"] for r in rows) / n,
            "underroute": sum(r["under"] for r in rows) / n, "overroute": sum(r["over"] for r in rows) / n,
            "mix": {t: sum(r["tier"] == t for r in rows) for t, *_ in TIERS}}

def cascade(results, verify_fast, verify_cap, thr):
    """Run fast; escalate to capable if P(wrong) > thr; then to reasoning likewise. Costs add up."""
    chosen, extra = [], []
    for i in range(len(T)):
        if verify_fast[i]["p_wrong"] <= thr: chosen.append("fast"); extra.append(0.0); continue
        if verify_cap[i]["p_wrong"] <= thr: chosen.append("capable"); extra.append(results["fast"][i]["cost_usd"]); continue
        chosen.append("reasoning"); extra.append(results["fast"][i]["cost_usd"] + results["capable"][i]["cost_usd"])
    s = score_policy(chosen, results, GOLD)
    s["cost_per_task_usd"] += sum(extra) / len(T)  # wasted cheaper attempts
    s["threshold"] = thr
    return s

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ak = key(".anthropic_key", "ANTHROPIC_API_KEY")
    if not ak: sys.exit("no Anthropic key")
    client = anthropic.Anthropic(api_key=ak, timeout=180.0)
    print(f"{len(T)} tasks: {sum(t['kind']=='exact' for t in T)} exact, {sum(t['kind']=='code' for t in T)} code, {sum(t['kind']=='judge' for t in T)} judge")

    results = {}
    REGRADE = "--regrade" in sys.argv
    prev = json.loads((OUT / "model_routing_results.json").read_text()) if REGRADE else None
    for tier, model, pin, pout in TIERS:
        if REGRADE:  # offline: regrade exact/code from stored replies, keep judge verdicts
            results[tier] = prev["results"][tier]
            for i, r in enumerate(results[tier]):
                if T[i]["kind"] == "exact": r["pass"] = grade_exact(T[i], r["reply"])
                elif T[i]["kind"] == "code": r["pass"] = grade_code(T[i], r["reply"])
        else:
            results[tier] = run_tier(client, tier, model, pin, pout)
        rs = results[tier]
        print(f"  {tier:9s} {model:18s} pass={sum(r['pass'] for r in rs)}/{len(rs)} "
              f"median={1000*statistics.median(r['latency_s'] for r in rs):.0f}ms cost/task=${statistics.mean(r['cost_usd'] for r in rs):.4f}")

    GOLD = []
    for i in range(len(T)):
        g = next((t for t, *_ in TIERS if results[t][i]["pass"]), None)
        GOLD.append(g)
    print("  gold tier distribution:", {t: GOLD.count(t) for t, *_ in TIERS}, "unsolved:", GOLD.count(None))

    if REGRADE:
        routes, vfast, vcap = prev["routes"], prev["verify_fast"], prev["verify_capable"]
    else:
        with ThreadPoolExecutor(max_workers=8) as ex:
            routes = list(ex.map(jev_route, T))
            vfast = list(ex.map(lambda i: jev_verify(T[i], results["fast"][i]["reply"]), range(len(T))))
            vcap = list(ex.map(lambda i: jev_verify(T[i], results["capable"][i]["reply"]), range(len(T))))

    policies = {
        "always-fast": ["fast"] * len(T), "always-capable": ["capable"] * len(T), "always-reasoning": ["reasoning"] * len(T),
        "heuristic": [heuristic(t) for t in T],
        "jev-choice": [r["tier"] for r in routes],
        "jev-choice+bump<0.6": [TIERS[min(TIER_RANK[r["tier"]] + (r["tier_conf"] < 0.6), 2)][0] for r in routes],
        "jev-composed(depth)": [composed(r) for r in routes],
        "oracle(gold)": [g or "reasoning" for g in GOLD],
    }
    summary = {name: score_policy(ch, results, GOLD) for name, ch in policies.items()}
    for thr in (0.3, 0.5, 0.7, 0.9):
        summary[f"jev-cascade>{thr}"] = cascade(results, vfast, vcap, thr)

    print(f"\n{'policy':22s} pass   $/task  under  over   mix")
    for name, s in summary.items():
        print(f"{name:22s} {s['pass_rate']:.2f}  {s['cost_per_task_usd']:.4f}  {s['underroute']:.2f}   {s['overroute']:.2f}   {s['mix']}")

    # verification quality: does P(wrong) separate right from wrong fast answers?
    def auc_ish(v, tier):
        wrong = [x["p_wrong"] for x, r in zip(v, results[tier]) if not r["pass"]]
        right = [x["p_wrong"] for x, r in zip(v, results[tier]) if r["pass"]]
        return {"n_wrong": len(wrong), "mean_p_wrong|wrong": statistics.mean(wrong) if wrong else None,
                "mean_p_wrong|right": statistics.mean(right) if right else None,
                "pairs_ordered": (sum(w > r for w in wrong for r in right) / (len(wrong) * len(right))) if wrong and right else None}
    verify_quality = {"fast": auc_ish(vfast, "fast"), "capable": auc_ish(vcap, "capable")}
    print("\nverify quality:", json.dumps(verify_quality))
    print("jev route median latency: %.0f ms; verify: %.0f ms" % (
        1000 * statistics.median(r["latency_s"] for r in routes), 1000 * statistics.median(v["latency_s"] for v in vfast)))

    (OUT / "model_routing_results.json").write_text(json.dumps({
        "tasks": [{"kind": t["kind"], "q": t["q"]} for t in T], "tiers": TIERS, "results": results, "gold": GOLD,
        "routes": routes, "verify_fast": vfast, "verify_capable": vcap, "policies": policies, "summary": summary,
        "verify_quality": verify_quality}, indent=1))
