#!/usr/bin/env python3
"""Offline checks for the harness: no network, no keys, no spend.

1. Every open-slice answer is re-derived independently (brute force, datetime,
   or by executing the prompt's own code), and every grader accepts the reference
   answer and rejects the recorded wrong one.
2. Answer parsing, the cron grader, the ported jido_ai heuristic and the Jev rule
   on fixed cases.
3. The labeling run, both routers and the analysis end to end on fake clients in a
   temp directory, including resume after an error; the Jev client's endpoint override.
4. With `anthropic` installed: the same calls through the real SDK against a mocked
   transport, checking the request bodies the API would receive.

Run: python3 selftest.py              (parts 1-3)
     .venv/bin/python selftest.py     (all four)
"""
import csv, datetime as dt, heapq, io, itertools, json, os, re, statistics, subprocess, sys, tempfile, threading, unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import analyze, jev_eval, label_models, route_jev, route_llm, tasks as T
from common import CONFIGS, TIERS, TIER_CRITERIA, read_jsonl

TASKS, ACTS = T.load_tasks()
BY_ID = {t["id"]: t for t in TASKS}
jev_eval.MODEL = "jev-latest"  # the suite checks Jev's file names, whatever JEV_MODEL is set to

try:
    import anthropic
    import httpx2
except ImportError:
    anthropic = None


def accepts(task_id, value):
    t = BY_ID[task_id]
    return T.grade(t, [], f"Working.\nANSWER: {value}")[0]


class ReferenceAnswers(unittest.TestCase):
    def test_graders_accept_reference_and_reject_wrong(self):
        for t in TASKS:
            if t["slice"] != "open":
                continue
            for wrap in ("ANSWER: {}", "**ANSWER:** {}", "Answer: `{}`", "ANSWER:\n{}"):
                with self.subTest(task=t["id"], wrap=wrap):
                    self.assertTrue(T.grade(t, [], "Working.\n" + wrap.format(t["ref"]))[0])
                    self.assertFalse(T.grade(t, [], "Working.\n" + wrap.format(t["wrong"]))[0])

    def test_arithmetic_and_dates(self):
        self.assertTrue(accepts("open_seconds", 2 * 3600 + 45 * 60))
        self.assertTrue(accepts("open_date_add", dt.date(2026, 11, 20) + dt.timedelta(days=100)))
        d0 = dt.date(2026, 3, 2)
        self.assertTrue(accepts("open_business_days", sum((d0 + dt.timedelta(i)).weekday() < 5 for i in range(30))))
        self.assertTrue(accepts("open_weekday", dt.date(2031, 2, 19).strftime("%A")))
        arrive = dt.datetime(2026, 1, 1, 14, 5) + dt.timedelta(hours=212 / 80, minutes=12) + dt.timedelta(hours=95 / 60)
        self.assertTrue(accepts("open_train", arrive.strftime("%H:%M")))
        self.assertTrue(accepts("open_bit_count", bin(2026).count("1")))
        self.assertTrue(accepts("open_mod_pow", pow(7, 2026, 100)))
        self.assertTrue(accepts("open_percent", f"{(1.2 * 0.75 * 1.1 - 1) * 100:+.1f}%"))
        dice = Fraction(sum((a * b) % 6 == 0 for a in range(1, 7) for b in range(1, 7)), 36)
        self.assertTrue(accepts("open_dice", f"{dice.numerator}/{dice.denominator}"))
        tuesdays = []
        for month in range(1, 7):
            days = [dt.date(2027, month, d) for d in range(1, 15)]
            tuesdays.append([d for d in days if d.weekday() == 1][1].isoformat())
        self.assertTrue(accepts("open_second_tuesdays", ", ".join(tuesdays)))

    def test_timezone(self):
        try:
            from zoneinfo import ZoneInfo
            london = dt.datetime(2026, 3, 29, 9, 30, tzinfo=ZoneInfo("Europe/London"))
            ny = london.astimezone(ZoneInfo("America/New_York"))
        except Exception as e:  # no tz database on this machine
            self.skipTest(f"zoneinfo unavailable: {e}")
        self.assertTrue(accepts("open_timezone", ny.strftime("%H:%M")))

    def test_no_adjacent_sevens_by_dynamic_programming(self):
        # state: (sevens so far, last digit was 7) -> count of prefixes
        states = {(0, False): 1}
        for _ in range(7):
            nxt = {}
            for (k, last7), n in states.items():
                nxt[(k, False)] = nxt.get((k, False), 0) + 9 * n
                if not last7 and k < 3:
                    nxt[(k + 1, True)] = nxt.get((k + 1, True), 0) + n
            states = nxt
        self.assertTrue(accepts("open_no_adjacent", states.get((3, False), 0) + states.get((3, True), 0)))

    def test_prompt_code_and_list_trace(self):
        ns = {}
        exec(BY_ID["open_code_trace"]["prompt"].split("\n\n", 1)[1], ns)  # the prompt's own function
        self.assertTrue(accepts("open_code_trace", ns["f"](50)))
        xs = [5, 3, 8, 1]
        xs.sort(); xs.append(4); xs.reverse(); xs.remove(3); xs.insert(1, 9); xs.pop()
        self.assertTrue(accepts("open_list_trace", xs))

    def test_data_read_from_the_prompts(self):
        rows = csv.DictReader(io.StringIO(BY_ID["open_csv_revenue"]["prompt"].split("\n\n")[1]))
        revenue = sum(int(r["qty"]) * float(r["unit_price"]) for r in rows
                      if r["customer"] == "Acme" and r["date"].startswith("2026-03") and r["status"] == "shipped")
        self.assertTrue(accepts("open_csv_revenue", f"{revenue:.2f}"))

        pairs = sorted([int(a), int(b)] for a, b in re.findall(r"\[(\d+),(\d+)\]", BY_ID["open_intervals"]["prompt"]))
        merged = []
        for a, b in pairs:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.assertTrue(accepts("open_intervals", ", ".join(f"[{a},{b}]" for a, b in merged)))

        items = [(int(w), int(v)) for w, v in re.findall(r"\((\d+), (\d+)\)", BY_ID["open_knapsack"]["prompt"])]
        self.assertEqual(len(items), 8)
        best = max(sum(v for _, v in c) for r in range(9) for c in itertools.combinations(items, r) if sum(w for w, _ in c) <= 20)
        self.assertTrue(accepts("open_knapsack", best))

        graph = {}
        for a, b, w in re.findall(r"([A-F])-([A-F]) (\d+)", BY_ID["open_shortest_path"]["prompt"]):
            graph.setdefault(a, []).append((b, int(w)))
            graph.setdefault(b, []).append((a, int(w)))
        dist, heap = {"A": 0}, [(0, "A")]
        while heap:
            d, u = heapq.heappop(heap)
            for v, w in graph[u]:
                if d + w < dist.get(v, float("inf")):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))
        self.assertTrue(accepts("open_shortest_path", dist["F"]))

    def test_cryptarithm_is_unique(self):
        letters, found = "BASELGM", []
        for digits in itertools.permutations(range(10), len(letters)):
            m = dict(zip(letters, digits))
            if m["B"] and m["G"]:
                val = lambda w: int("".join(str(m[c]) for c in w))  # noqa: E731
                if val("BASE") + val("BALL") == val("GAMES"):
                    found.append(val("GAMES"))
        self.assertEqual(len(found), 1)
        self.assertTrue(accepts("open_cryptarithm", found[0]))

    def test_logic_puzzles_are_unique(self):
        # The clues as written in tasks.GRID, in order.
        names, pets, drinks = ("Ana", "Ben", "Cai", "Dee"), ("cat", "dog", "fish", "bird"), ("tea", "coffee", "milk", "water")
        solutions = []
        for N in itertools.permutations(names):
            for P in itertools.permutations(pets):
                for D in itertools.permutations(drinks):
                    n, p, d = N.index, P.index, D.index
                    if (n("Ben") == d("milk") and p("dog") == 0 and n("Ana") == p("cat") + 1 and d("coffee") == 3
                            and abs(n("Dee") - d("tea")) == 1 and p("bird") == d("water") and n("Cai") != 3
                            and p("fish") > n("Ben") and n("Ben") != 0):
                        solutions.append(N)
        self.assertEqual(len(solutions), 1)
        self.assertTrue(accepts("open_logic_grid", ", ".join(solutions[0])))

        knights = []
        for a, b, c in itertools.product((True, False), repeat=3):
            says = (a + b + c == 1, a + b + c == 2, not (a and b))  # A, B, C as in tasks.KNIGHTS
            if (a, b, c) == says:
                knights.append([x for x, k in zip("ABC", (a, b, c)) if k])
        self.assertEqual(len(knights), 1)
        self.assertTrue(accepts("open_knights", ", ".join(knights[0]) or "none"))

        orders = []
        for perm in itertools.permutations("ABCDE"):
            s = {t: i + 1 for i, t in enumerate(perm)}
            if (s["C"] < s["B"] < s["D"] and abs(s["A"] - s["C"]) == 1 and s["E"] not in (1, 5)
                    and s["D"] == s["E"] + 2 and s["A"] != 1):
                orders.append(perm)
        self.assertEqual(len(orders), 1)
        self.assertTrue(accepts("open_talk_slots", ", ".join(orders[0])))


class Parsing(unittest.TestCase):
    def test_parse_answer(self):
        self.assertEqual(T.parse_answer("x\nANSWER: 42"), "42")
        self.assertEqual(T.parse_answer("**ANSWER:** 42"), "42")
        self.assertEqual(T.parse_answer("Answer: `30 2 * * *`"), "30 2 * * *")
        self.assertEqual(T.parse_answer("ANSWER:\n```\n42\n```"), "42")
        self.assertEqual(T.parse_answer("ANSWER: 1\nmore\nANSWER: 2"), "2")
        self.assertEqual(T.parse_answer("The answer: is not a marker\nfinal line"), "final line")

    def test_cron_equivalence(self):
        g = T.cron("*/15 9-17 * 1,7 1-5")
        self.assertTrue(g("0,15,30,45 9-17 * jan,jul mon-fri", ""))
        self.assertTrue(g("0-59/15 9-17 ? 1,7 1-5", ""))
        self.assertFalse(g("*/15 9-18 * 1,7 1-5", ""))
        self.assertTrue(T.cron("0 0 * * 0")("0 0 * * 7", ""))  # 7 is Sunday too
        self.assertTrue(T.cron("30 2 * * *")("cron: 30 2 * * *", ""))  # leading words are skipped
        self.assertFalse(T.cron("30 2 * * *")("2 30 * * *", ""))  # hour 30 does not parse

    def test_tool_grading(self):
        t = next(t for t in TASKS if t["gold"] == ["stop_self"])
        self.assertTrue(T.grade(t, ["stop_self", "noop"], "")[0])
        self.assertFalse(T.grade(t, [], "Which agent?")[0])
        none = next(t for t in TASKS if t["gold"] == ["none"])
        self.assertTrue(T.grade(none, [], "No tool fits.")[0])

    def test_adaptive_heuristic_port(self):
        # 5 words -> 0.015; 1 sentence -> 0.04; no keywords, questions or constraints.
        self.assertAlmostEqual(analyze.adaptive_complexity("Set your status to 'paused'."), 0.055)
        # 15 words -> 0.045; 3 sentences -> 0.12; 6 keywords (cap 3) -> 0.3; 2 "?" + must/should -> 0.16.
        p = "Analyze and compare the alternatives. You must evaluate each option? Should we consider multiple paths?"
        self.assertAlmostEqual(analyze.adaptive_complexity(p), 0.625)
        self.assertEqual(analyze.adaptive_tier(p), "capable")
        self.assertEqual(analyze.adaptive_tier("Set your status to 'paused'."), "fast")

    def test_cumulative_rule(self):
        probs = {"fast": 0.6, "capable": 0.3, "reasoning": 0.1}
        self.assertEqual([analyze.cdf_tier(probs, tau) for tau in (0.5, 0.85, 0.95)], ["fast", "capable", "reasoning"])
        self.assertEqual(analyze.cdf_tier({"fast": 0.5, "capable": 0.49}, 0.999), "reasoning")  # rounding fails up


# ---------------------------------------------------------------- fakes for the end-to-end run
class FakeAPIError(Exception):
    pass


HARD = [t["id"] for t in TASKS if t["band"] == "hard"]
WEAK_TOOL = {"tool05", "tool06"}


def fake_passes(config, task, sample):
    """Chosen so every label occurs: easy/medium -> fast, weak tool tasks -> capable,
    hard tasks alternate capable/reasoning, one hard task is unsolved on the model mapping."""
    if task["slice"] == "tool" or task["band"] in ("easy", "medium"):
        return not (config == "haiku" and task["id"] in WEAK_TOOL)
    i = HARD.index(task["id"])
    return {"haiku": False, "sonnet": i % 2 == 0, "opus": i != 1, "opus-medium": True, "opus-low": sample != 0}[config]


def fake_create(fail_once=()):
    pending, lock = set(fail_once), threading.Lock()

    def create(**req):
        task = next(t for t in TASKS if t["prompt"] == req["messages"][0]["content"])
        config = next(c for c, v in CONFIGS.items() if v["model"] == req["model"] and v["effort"] == (req.get("output_config") or {}).get("effort"))
        with lock:  # nth call for this cell; run_one does not pass the sample index to the client
            sample = create.calls.get((task["id"], config), 0)
            create.calls[(task["id"], config)] = sample + 1
            fail = (task["id"], config) in pending
            pending.discard((task["id"], config))
        if fail:
            raise FakeAPIError("overloaded")
        ok = fake_passes(config, task, sample)
        if task["slice"] == "tool":
            gold = task["gold"][0]
            name = (gold if gold != "none" else None) if ok else ("reply" if gold == "none" else "noop" if gold != "noop" else "cancel")
            content = [SimpleNamespace(type="tool_use", name=name)] if name else [SimpleNamespace(type="text", text="No action fits.")]
        else:
            content = [SimpleNamespace(type="text", text="Working.\nANSWER: " + (task["ref"] if ok else task["wrong"]))]
        usage = SimpleNamespace(input_tokens=300, output_tokens={"haiku": 100, "sonnet": 400}.get(config, 800))
        return SimpleNamespace(content=content, usage=usage, stop_reason="end_turn", model=req["model"])
    create.calls = {}
    return create


def fake_jev(body):
    fake_jev.bodies.append(body)
    task = next(t for t in TASKS if t["jev_state"] == body["state"])
    probs = {"tool": (0.7, 0.25, 0.05), "easy": (0.8, 0.15, 0.05), "medium": (0.3, 0.6, 0.1), "hard": (0.05, 0.25, 0.7)}[task["band"]]
    answers = {"tier": {"choice": TIERS[probs.index(max(probs))], "confidence": max(probs), "probabilities": dict(zip(TIERS, probs))},
               "difficulty": {"score": {"tool": 0.3, "easy": 0.2, "medium": 1.0, "hard": 1.8}[task["band"]], "confidence": 0.5},
               "risk": {"score": 1.6 if task["id"] in WEAK_TOOL else 1.0, "confidence": 0.5},
               "implicit_context": {"noul": 0.9 if task["id"] in WEAK_TOOL else 0.1}}
    return {"answers": answers, "usage": {"input_tokens": 900, "output_tokens": 200}, "model": "jev-test"}, 0.15


fake_jev.bodies = []


def fake_router(**req):
    tier = "reasoning" if "must be exactly right" in req["messages"][0]["content"] else "capable"
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=tier)],
                           usage=SimpleNamespace(input_tokens=250, output_tokens=2), stop_reason="end_turn")


class EndToEnd(unittest.TestCase):
    def test_pipeline_on_fakes(self):
        quiet = lambda *a, **k: None  # noqa: E731
        with tempfile.TemporaryDirectory() as out:
            create = fake_create(fail_once={("open_email", "sonnet")})
            rows = label_models.run(TASKS, ACTS, list(CONFIGS), 3, out, create, (FakeAPIError,), workers=4, log=quiet)
            self.assertEqual(len(rows), len(TASKS) * len(CONFIGS) * 3)
            self.assertEqual(sum(not r["ok"] for r in rows), 1)
            again = label_models.run(TASKS, ACTS, list(CONFIGS), 3, out, create, (FakeAPIError,), log=quiet)
            self.assertEqual([(r["task"], r["config"], r["ok"]) for r in again], [("open_email", "sonnet", True)])
            self.assertEqual(label_models.run(TASKS, ACTS, list(CONFIGS), 3, out, create, (FakeAPIError,), log=quiet), [])

            self.assertEqual(len(route_jev.run(TASKS, out, fake_jev, log=quiet)), len(TASKS))
            self.assertEqual(route_jev.run(TASKS, out, fake_jev, log=quiet), [])
            self.assertEqual(len(route_llm.run(TASKS, out, fake_router, (FakeAPIError,), log=quiet)), len(TASKS))

            s = analyze.analyze(out, log=quiet)
            self.assertEqual(s["regraded_changes"], 0)
            self.assertEqual(s["unresolved_errors"], 0)
            model = s["mappings"]["model"]
            self.assertEqual(model["tasks"], len(TASKS))
            labels = model["labels"]
            self.assertEqual({lab for lab in labels.values()}, {"fast", "capable", "reasoning", "unsolved"})
            self.assertEqual(labels["tool05"], "capable")
            self.assertEqual(labels[HARD[0]], "capable")
            self.assertEqual(labels[HARD[1]], "unsolved")
            self.assertEqual(labels[HARD[3]], "reasoning")
            routers = model["routers"]
            best = routers["Perfect routing"]["pass"]
            self.assertTrue(all(r["pass"] <= best + 1e-12 for r in routers.values()))
            runs = [r for r in read_jsonl(Path(out) / "runs.jsonl") if r.get("ok") and r["config"] == "opus"]
            by_task = {}
            for r in runs:
                by_task.setdefault(r["task"], []).append(r["passed"])
            self.assertAlmostEqual(routers["Always reasoning"]["pass"], statistics.mean(statistics.mean(v) for v in by_task.values()))
            self.assertEqual(routers["Always fast"]["under"] > 0, True)
            self.assertEqual(routers["Always reasoning"]["under"], 0)
            self.assertIn("Jev cumulative >= 0.9", routers)
            self.assertTrue(any(r["frontier"] for r in routers.values()))
            self.assertEqual(s["mappings"]["effort"]["tasks"], len(TASKS))
            self.assertIn("Jev risk Score", s["signals"])
            self.assertTrue((Path(out) / "summary.json").exists())

    def test_jev_request_shape(self):
        fake_jev.bodies.clear()
        task = BY_ID["tool01"]
        row = route_jev.route_one(fake_jev, task)
        body = fake_jev.bodies[-1]
        self.assertTrue(row["ok"])
        self.assertEqual(body["model"], jev_eval.MODEL)
        self.assertEqual(body["state"], task["jev_state"])
        self.assertEqual(set(body["questions"]), {"tier", "difficulty", "risk", "implicit_context"})
        self.assertEqual(body["questions"]["risk"], jev_eval.questions(ACTS, "discovery")["risk"])  # unchanged from the pilot
        self.assertEqual(list(body["questions"]["tier"]["criteria"]), TIERS)
        self.assertEqual(body["state"]["available_actions"], [a["name"] for a in ACTS])

    def test_jev_errors_are_rows(self):
        def broken(body):
            return {"unexpected": True}, 0.1
        row = route_jev.route_one(broken, BY_ID["tool01"])
        self.assertFalse(row["ok"])
        self.assertIn("unexpected response shape", row["error"])

    def test_every_script_starts_on_its_own(self):
        for script in ("label_models.py", "route_jev.py", "route_llm.py", "analyze.py"):
            with self.subTest(script=script):
                r = subprocess.run([sys.executable, script, "--help"], cwd=Path(__file__).parent, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)

    def test_other_models_get_their_own_files(self):
        quiet = lambda *a, **k: None  # noqa: E731
        self.assertEqual(jev_eval.model_file("jev_routes", ".jsonl"), "jev_routes.jsonl")
        with mock.patch.object(jev_eval, "MODEL", "clm-latest"), tempfile.TemporaryDirectory() as out:
            self.assertEqual(jev_eval.model_file("jev_eval_results"), "jev_eval_results_clm-latest.json")
            label_models.run(TASKS, ACTS, ["haiku", "sonnet", "opus"], 1, out, fake_create(), (FakeAPIError,), log=quiet)
            route_jev.run(TASKS, out, fake_jev, log=quiet)
            self.assertEqual(fake_jev.bodies[-1]["model"], "clm-latest")
            report = []
            s = analyze.analyze(out, log=report.append)
            self.assertTrue(report[0].startswith("Every Jev row below is clm-latest, from jev_routes_clm-latest.jsonl."))
            self.assertIn("Jev choice (argmax)", s["mappings"]["model"]["routers"])
            self.assertEqual(sorted(p.name for p in Path(out).iterdir()),
                             ["jev_routes_clm-latest.jsonl", "runs.jsonl", "summary_clm-latest.json"])
        with mock.patch.object(jev_eval, "MODEL", "org/model:v1"):
            self.assertEqual(jev_eval.model_file("summary"), "summary_org_model_v1.json")

    def test_typesafe_key_only_goes_to_typesafe(self):
        sent = []
        def urlopen(req, timeout):
            sent.append(req)
            return io.BytesIO(b'{"answers": {}}')
        with mock.patch.object(jev_eval.urllib.request, "urlopen", urlopen), \
                mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": "typesafe-sentinel"}):
            os.environ.pop("JEV_API_KEY", None)
            with mock.patch.object(jev_eval, "API", jev_eval.TYPESAFE_API):
                jev_eval.call({})
            self.assertEqual(sent[-1].get_header("Authorization"), "Bearer typesafe-sentinel")
            with mock.patch.object(jev_eval, "API", "http://127.0.0.1:8700/v1/systemone"):
                jev_eval.call({})
                self.assertEqual(sent[-1].full_url, "http://127.0.0.1:8700/v1/systemone")
                self.assertIsNone(sent[-1].get_header("Authorization"))
                os.environ["JEV_API_KEY"] = "local-sentinel"
                jev_eval.call({})
                self.assertEqual(sent[-1].get_header("Authorization"), "Bearer local-sentinel")


@unittest.skipUnless(anthropic, "anthropic SDK not installed")
class RealSDK(unittest.TestCase):
    """The harness's own code through anthropic's client, against a mocked transport."""

    def client(self, status=200):
        self.seen = []

        def handler(request):
            body = json.loads(request.content)
            self.seen.append((request.url.path, body))
            if status != 200:
                return httpx2.Response(status, json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
            if body.get("tools"):
                content = [{"type": "tool_use", "id": "toolu_1", "name": "stop_self", "input": {}}]
            else:
                content = [{"type": "text", "text": "Working.\nANSWER: 9900"}]
            return httpx2.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                                              "content": content, "stop_reason": "end_turn", "stop_sequence": None,
                                              "usage": {"input_tokens": 1000, "output_tokens": 100}})
        transport = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
        return anthropic.Anthropic(api_key="dummy-not-a-key", http_client=transport, max_retries=0)

    def test_prompt_and_tools_match_the_pilot(self):
        import llm_baseline
        self.assertEqual(T.SELECT, llm_baseline.SELECT)
        self.assertEqual(T.tools_from_actions(ACTS), llm_baseline.tools_from_actions(ACTS))

    def test_open_task_request(self):
        c = self.client()
        tools = T.tools_from_actions(ACTS)
        row = label_models.run_one(c.messages.create, BY_ID["open_seconds"], "opus-low", 0, tools, (anthropic.APIStatusError,))
        path, body = self.seen[-1]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(body["model"], "claude-opus-5")
        self.assertEqual(body["output_config"], {"effort": "low"})
        self.assertEqual(body["max_tokens"], 16000)
        self.assertEqual(body["system"], T.OPEN_SYSTEM)
        self.assertNotIn("thinking", body)
        self.assertNotIn("tools", body)
        self.assertTrue(row["ok"] and row["passed"])
        self.assertAlmostEqual(row["cost_usd"], (1000 * 5 + 100 * 25) / 1e6)

    def test_tool_task_request(self):
        c = self.client()
        tools = T.tools_from_actions(ACTS)
        row = label_models.run_one(c.messages.create, BY_ID["tool01"], "haiku", 0, tools, (anthropic.APIStatusError,))
        body = self.seen[-1][1]
        self.assertEqual(body["model"], "claude-haiku-4-5")
        self.assertNotIn("output_config", body)
        self.assertEqual(body["system"], T.SELECT)
        self.assertEqual(len(body["tools"]), 19)
        self.assertEqual(body["max_tokens"], 4096)
        self.assertEqual(row["calls"], ["stop_self"])
        self.assertTrue(row["passed"])

    def test_api_error_is_recorded(self):
        c = self.client(status=400)
        row = label_models.run_one(c.messages.create, BY_ID["open_seconds"], "sonnet", 0, [], (anthropic.APIStatusError,))
        self.assertFalse(row["ok"])
        self.assertTrue(row["error"].startswith("BadRequestError"))

    def test_llm_router_request(self):
        c = self.client()
        row = route_llm.route_one(c.messages.create, BY_ID["open_knights"], "claude-haiku-4-5", (anthropic.APIStatusError,))
        body = self.seen[-1][1]
        self.assertEqual(body["system"], route_llm.ROUTER_SYSTEM)
        for tier, desc in TIER_CRITERIA.items():
            self.assertIn(f"{tier}: {desc}", body["system"])
        self.assertTrue(row["ok"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
