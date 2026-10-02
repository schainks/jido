# Experiments: Jev (TypeSafe System One) as a tool router and model router for Jido

Four experiments: tool selection versus native LLM tool-calling, model-tier routing, tool selection versus embeddings, and a live Jido AI agent with a Jev request transformer (Elixir).

**Question.** Can a calibrated judgment model pick the right Jido action from the
metadata Jido already exposes (name + description via `Jido.Discovery`), and how
does that compare with the native LLM tool-calling that `jido_ai`'s ReAct loop
uses today?

**Short answer.** On 34 hand-written requests over the 19 real actions in
`lib/jido/actions/*.ex`, Jev chose correctly 33/34 times from the Choice alone and
34/34 when gated by a Noul ("does any listed action fit"), at ~140 ms per request.
The same requests through native tool-calling scored 27–28/34 on Claude Haiku 4.5
(jido_ai's default `:fast` alias) and 28–33/34 on Claude Opus 5, at 1–2 s per call.
Nearly every LLM miss was a clarification reply instead of a tool call.

This is a smoke test, not a benchmark. See caveats at the bottom.

## Experiment 1: tool selection

Runs on 2026-09-24 (Jev) and 2026-09-25 (LLM baseline). Same 34 queries, same 19
actions, one request per query.

| Router | Prompt / criteria | Correct | No-action cases | Median latency | p95 latency | Input tokens | Cost per 1k calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Jev 1.13 (Choice only) | name + description | 33 / 34 | 4 / 5 | 134 ms | 199 ms | ~1,040 | not measured |
| Jev 1.13 (Noul gate + Choice) | name + description | 34 / 34 | 5 / 5 | 134 ms | 199 ms | ~1,040 | not measured |
| Jev 1.13 (Noul gate + Choice) | + param names and docs | 34 / 34 | 5 / 5 | 146 ms | 187 ms | ~1,370 | not measured |
| Claude Haiku 4.5, native tool-calling | assistant prompt | 27 / 34 | 5 / 5 | 1,100 ms | 1,779 ms | ~1,890 | $2.30 |
| Claude Haiku 4.5, native tool-calling | selection-only prompt | 28 / 34 | 5 / 5 | 1,054 ms | 2,134 ms | ~1,920 | $2.31 |
| Claude Opus 5, native tool-calling | assistant prompt | 28 / 34 | 5 / 5 | 1,969 ms | 5,476 ms | ~2,166 | $14.38 |
| Claude Opus 5, native tool-calling | selection-only prompt | 33 / 34 | 5 / 5 | 2,060 ms | 6,041 ms | ~2,206 | $13.61 |

Jev confidence versus precision (discovery variant, Choice only):

| Confidence threshold | Coverage | Precision |
| --- | --- | --- |
| 0.5 | 97 % | 97 % |
| 0.6 | 91 % | 97 % |
| 0.7 | 79 % | 96 % |
| 0.85 | 71 % | 100 % |
| 0.9 | 65 % | 100 % |

Jev with 8 concurrent requests: 34 requests in 0.83 s, no rate limiting, same accuracy.

## Observations

- **Noul gate is clean.** Requests with no matching action scored 0.05–0.12 on
  "does any listed action fit"; requests with one scored 0.59–0.99. The single Choice
  miss (`Translate 'hello' to French` chosen as `reply` at 0.83) had a Noul of 0.06.
- **Low Jev confidence lands on real ambiguity.** `notify_pid` vs `forward` (0.57),
  `mark_idle` vs `set_status` (0.55), a 10 s timer that could be `schedule_signal` or
  `schedule_timeout` (0.61). All were still correct.
- **One overconfident Jev answer.** `Kill it.` chose `stop_self` at 0.90 although
  `stop_child` and `cancel` are equally plausible. The blast-radius Score for that
  request was 1.85/2, so a "high risk requires confirmation" policy would still have
  paused it. Confidence alone is not a safety gate; pair it with a risk judgment.
- **LLM misses were clarification, not confusion.** On requests whose referent lives
  in agent state rather than the message ("pass *this* signal along to the billing
  agent"), Haiku and Opus replied asking for signal type / payload / target PID
  instead of calling `forward`. In ReAct that costs an extra turn. Haiku kept doing
  this on 6/34 even under a prompt that says "call the best tool, missing parameters
  are fine, never ask".
- **Opus 5 matches Jev's Choice accuracy when forced to select** (33/34). Its one miss
  was `notify_pid` vs `forward`, the same pair Jev flagged at 0.57. Opus gives no
  confidence signal, at ~15x the latency.
- **Every router refused the five no-tool requests** (weather, summarize, translate,
  delete an account, book a flight). The Noul's edge is refusing *with a number*.
- **Discovery metadata is enough.** Adding param names/docs to the Jev criteria did
  not change accuracy and cost ~30 % more input tokens.

## Caveats

- 34 queries written by one person; Jido's control actions are well named and
  mutually distinct. Application tool sets with overlapping purposes will be harder.
- Jev names the action; it does not fill arguments. Native tool-calling does both.
- Opus 5 ran with default adaptive thinking, so its latency includes reasoning.
- Jev per-request cost was not visible in the API response and is not reported.

## Experiment 2: Jev as a model-tier router

**Question.** Can Jev pick the cheapest model tier that will get a task right, from the
request alone, and how does that compare with always-X policies, a keyword heuristic,
and a run-then-verify cascade?

**Ground truth is outcome-based.** 42 tasks (32 exact-answer, 6 code graded by unit
tests, 4 open tasks graded by an Opus 5 judge) each ran once on three tiers that are
the same models on Vertex AI and the Anthropic API. Gold tier = cheapest tier that
passed. Policies are then scored against gold without re-running models.
Run on 2026-09-25; `model_routing.py`, results in `model_routing_results.json`.

| Tier | Model | Vertex ID | Passed | Median latency | Cost per task |
| --- | --- | --- | --- | --- | --- |
| fast | Claude Haiku 4.5 | `claude-haiku-4-5@20251001` | 31 / 42 | 671 ms | $0.0005 |
| capable | Claude Sonnet 5 | `claude-sonnet-5` | 39 / 42 | 1,314 ms | $0.0007 |
| reasoning | Claude Opus 5 (default adaptive thinking) | `claude-opus-5` | 38 / 42 | 1,856 ms | $0.0025 |

Gold distribution: 31 tasks solvable by fast, 8 need capable, 1 needs reasoning, 2 unsolved by all.

| Policy | Pass rate | Cost per task | Underrouted (sent to a tier that fails) | Overrouted (paid for more than needed) | Tier mix fast / capable / reasoning |
| --- | --- | --- | --- | --- | --- |
| always-fast | 0.74 | $0.0005 | 0.21 | 0.00 | 42 / 0 / 0 |
| always-capable | 0.93 | $0.0007 | 0.02 | 0.74 | 0 / 42 / 0 |
| always-reasoning | 0.90 | $0.0025 | 0.05 | 0.93 | 0 / 0 / 42 |
| heuristic (Adaptive-style length + keywords) | 0.90 | $0.0007 | 0.05 | 0.64 | 5 / 37 / 0 |
| Jev Choice over tiers | 0.86 | $0.0005 | 0.10 | 0.05 | 35 / 6 / 1 |
| Jev Choice, bump one tier when confidence < 0.6 | 0.90 | $0.0006 | 0.05 | 0.14 | 31 / 8 / 3 |
| Jev depth Score, code thresholds | 0.90 | $0.0008 | 0.05 | 0.40 | 18 / 20 / 4 |
| Jev cascade: run fast, escalate if P(wrong) > 0.5 | 0.88 | $0.0010 | 0.07 | 0.14 | 30 / 8 / 4 |
| oracle (gold) | 0.95 | $0.0008 | 0.00 | 0.00 | 31 / 8 / 3 |

Jev routing call: 160 ms median. Jev verification call: 152 ms median. Cascade cost
includes the wasted cheaper attempts.

### What it shows

- **Jev pre-routing recovers the fast tier's cost and latency for most tasks.** The
  Choice-with-bump policy matches the heuristic's pass rate (0.90) while sending 31
  tasks to Haiku instead of 5, at 14 % overrouting versus 64 %. Against always-fast it
  gains 16 points of pass rate at the same cost.
- **On this particular stack the savings are small in dollars.** Sonnet 5 costs only
  1.4x Haiku per task here, so always-capable is near-oracle for $0.0002 more per task.
  Routing pays in latency (Haiku is ~650 ms faster per call, which compounds over an
  agent loop) and in keeping Opus out of the loop (5x cost). The economics get much
  stronger with a wider spread, for example Gemini Flash-Lite versus Gemini Pro on
  Vertex, which this script does not run.
- **The remaining underroutes are not predictable from the request.** After the
  confidence bump, Jev's misses were Haiku mistyping a base64 decode (`hello jidw`),
  Haiku adding a Markdown header to a "one sentence" summary, and Haiku answering "write
  a function that checks bracket balance" with a function that calls the Claude API to
  do it. Jev rated those requests easy at 0.98 to 1.00 confidence, and they are easy;
  the small model just fumbled. Only post-hoc verification can catch that class.
- **Jev's verification Noul orders right and wrong answers reasonably but not sharply.**
  P(wrong) averaged 0.58 on Haiku's wrong answers versus 0.23 on its right ones, and
  ranked a wrong answer above a right one 83 % of the time. That is enough for a cascade
  to beat always-fast (0.88 versus 0.74) but not enough, on this set, to beat simply
  paying for Sonnet.
- **Two API behaviors worth knowing.** Opus 5 returned `stop_reason: refusal` on the
  cron-field parser task (a safety-classifier false positive; a client-side fallback is
  needed on Vertex, which has no server-side `fallbacks`). And the Opus 5 judge failed all
  three tiers on the "exactly two sentences" task, so open-task grading is the noisiest
  part of this setup.

### Caveats

- 42 tasks, one run each, no repeats; single-sample pass/fail is noisy.
- Task difficulty was chosen to give the fast tier something to fail, not to mirror any
  real workload. Route policies must be re-evaluated on real traffic.
- Grading is imperfect: the first run had three grader bugs (Markdown bold, quoted
  answers, and my own wrong gold on the palindrome task, which all three models got
  right). `--regrade` re-scores stored replies offline so fixes do not cost new calls.
- Gemini tiers were not run (need Google ADC). Tier names are what jido_ai's
  `model_aliases` would map to Vertex model IDs.

**Pending: Google Cloud leg.** The harness this is for runs on Google Cloud, so the
next run should use Gemini Flash-Lite / Flash / Pro (and optionally Claude via Vertex)
as the tiers. Their wider price spread is where pre-routing should pay in dollars, not
just latency. Needs a service account with `roles/aiplatform.user`; the script will take
the project, region and model IDs as arguments.

## Experiment 3: the cheap alternative for tool selection (embeddings, BM25)

**Question.** If nearest-neighbour matching over action descriptions is as good as Jev,
the tool-selection case shrinks to "Jev adds a confidence number". Same 34 queries,
same 19 actions, each action embedded as `name: description`. Local ONNX models via
`fastembed`, no API. Run on 2026-09-25; `embed_baseline.py`, results in
`embed_baseline_results.json`.

Nearest-neighbour has no native "none" answer, so the overall figure uses the single
best similarity threshold found after the fact, an upper bound that favours the
baseline. The margin between the top two scores stands in for confidence.

| Router | Tool queries correct (of 29) | Overall incl. 5 no-action queries (of 34) | Precision at ~71 % coverage | Latency per query |
| --- | --- | --- | --- | --- |
| Jev 1.13, Choice only | 29 | 33 | 1.00 | 134 ms |
| Jev 1.13, Noul gate + Choice | 29 | 34 | 1.00 | 134 ms |
| bge-small-en-v1.5 (cosine) | 24 | 26 (best-case threshold) | 0.81 | 4 ms |
| bge-base-en-v1.5 | 22 | 25 (best-case) | 0.86 | 14 ms |
| bge-large-en-v1.5 | 21 | 24 (best-case) | 0.90 | 44 ms |
| all-MiniLM-L6-v2 | 17 | 20 (best-case) | 0.62 | 2 ms |
| BM25 (lexical) | 12 | 13 (best-case) | 0.48 | 0.1 ms |

### What it shows

- **Embeddings lose exactly where the tool test is hard.** Every embedding model missed
  "Relay this to the agent named 'audit'" (forward), "Spin up three workers for the
  crawl" (spawn_child), "Give up on waiting for the reply after two minutes"
  (schedule_timeout, pulled toward `reply` by the word "reply"), and "Stop being idle,
  get to work" (mark_working, pulled toward `mark_idle` by the word "idle"). Those are
  paraphrases and negations, which is what a judgment model handles and a similarity
  score does not.
- **Bigger embedding models did not help.** bge-large scored below bge-small on this set;
  with 29 queries the differences are noise, but there is no trend toward Jev's 29/29.
- **Margin is a weak confidence.** At the coverage where Jev's confidence gave 100 %
  precision, embedding margins gave 81 % to 90 %. Several embedding misses had margins
  near zero, but so did several hits.
- **Embeddings are 30x faster and free**, which matters for a pre-filter. A hybrid that
  uses embeddings to cut 100 tools to 10 and Jev to choose among the 10 is the natural
  shape for large registries, and would also stay under Jev's 255-option limit.
- **BM25 is not a baseline worth keeping.** Jido's action descriptions are short and the
  requests are paraphrases, so lexical overlap fails.

### Caveats

- Same 34 hand-written queries as experiment 1; one author's phrasing.
- Documents were `name: description` only, the same information Jev received. Richer
  documents (parameter docs, example phrasings) would help embeddings and were not tried.
- The "none" threshold was tuned on the test set itself, so the overall column overstates
  what embeddings would do in production.

## Experiment 4: inside the loop (Elixir, live `Jido.AI.Agent`)

**Question.** Single-call accuracy is not the claim that matters. Does a Jev-backed
`request_transformer` change what a whole `Jido.AI.Agent` ReAct run costs and returns?

**Setup.** `elixir/` holds two Mix projects run in the `hexpm/elixir:1.20.4` image via
`run.sh`: `typesafe_client` (a standalone, publishable client for TypeSafe System One with
typed answers, a stub, and Req-based HTTP with retry) and `jev_routing` (the transformer,
24 pure tools, 30 graded tasks, and the bench). Resolved deps: jido_ai 2.3.0, jido 2.3.3,
jido_action 2.3.2, req_llm 1.25.0. Design in `elixir/SPEC.md`, plan in `elixir/PLAN.md`.

Three agents share the same 24 tools (9 jido_action built-ins plus 15 synthetic pure
actions with deliberate confusable pairs) and the same prompt, `max_iterations: 6`,
streaming off. `baseline-fast` runs Haiku 4.5 with all tools; `baseline-capable` runs
Sonnet 5 with all tools; `jev` starts on Haiku and before every LLM turn asks Jev three
questions in one call (does the next step need a tool, which one, how deep is the
reasoning), then overrides that turn's `tools` to the tools Jev scored above zero, at most
3 (or none when no tool is needed), and its `model` to `:fast` / `:capable` / `:reasoning`
by depth (0.75 / 1.5). Any Jev error fails open to the baseline behaviour. 30 tasks
(14 one-tool, 8 two-tool chains, 8 no-tool), one fresh agent process per run, graded by
normalized exact match. Final run on 2026-09-26 after the code review below.

| condition | pass | one / two / no-tool | mean turns | median ms | p95 ms | mean input tokens | $/task | Jev ms/task | tier mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-fast (Haiku 4.5, 24 tools) | 0.93 | 14/14 / 8/8 / 6/8 | 2.03 | 1,630 | 3,189 | 5,603 | $0.00605 | 0 | 30 fast |
| baseline-capable (Sonnet 5, 24 tools) | 1.00 | 14/14 / 8/8 / 8/8 | 2.00 | 2,621 | 6,146 | 6,536 | $0.01390 | 0 | 30 capable |
| jev (transformer) | 0.93 | 14/14 / 8/8 / 6/8 | 2.00 | 1,789 | 5,548 | 903 | $0.00210 | 209 | 22 fast, 8 capable |

Cost uses the notional per-MTok table from the spec (fast 1/5, capable 2/10, reasoning
5/25), not provider list prices, and prices every `jev` turn at the most expensive tier
chosen in that run (an upper bound). Models actually resolved by ReqLLM's registry:
`anthropic:claude-haiku-4-5-20251001`, `anthropic:claude-sonnet-5`, `anthropic:claude-opus-5`
(never chosen). Raw rows with every per-turn decision, including Jev's full tool
probability map and the state it was sent, are in `bench_results_elixir.json`.

### What it shows

- **Tool gating is the cost lever, not model choice.** Every tool schema goes into every
  LLM turn. Cutting 24 tools to 1 (or 0) took mean input tokens per task from 5,603 to
  903, a 6.2x reduction, which is why `jev` costs 35 % of `baseline-fast` even though it
  sent 8 of 30 tasks to Sonnet. With larger registries this gap widens; with 5 tools it
  would mostly vanish.
- **Same pass rate as Haiku, same number of turns.** 28/30 for both; the two misses are
  the same two no-tool tasks (one asks what language Jido is written in, the other has a
  too-narrow gold answer) and both are task-quality problems, not routing. Gating never
  removed a tool the agent needed: all 14 one-tool and 8 two-tool tasks passed with a
  one-tool list on the first turn in 21 of 30 runs.
- **Jev's per-turn behaviour is legible and now auditable.** On the first turn it named
  exactly one tool 21 times, two tools once, and none 8 times (all 8 no-tool tasks). On 21
  of 30 later turns it reported "needs_tool" below 0.5 and emptied the tool list so the
  model answered directly. On all 8 two-tool chains the depth Score landed near 1.0 and
  routed to Sonnet; on all 22 others it stayed on Haiku.
- **Latency is a wash overall, split by kind.** Jev adds ~100 ms per turn (209 ms/task).
  No-tool tasks got faster (738 vs 786 ms median) because the prompt shrank; one-tool tasks
  were 10 % slower; two-tool tasks were 92 % slower because they ran on Sonnet. Whether the
  depth-to-Sonnet routing is worth it is a policy question; Haiku passed all 8 two-tool
  chains on its own here, so on this task set the answer is no.
- **Two things for the maintainer.** (1) jido_ai's `use Jido.AI.Agent` reads `tools:` and
  `system_prompt:` from the raw AST, so they must be literals; a shared function call does
  not compile. (2) Every run logged 2-3 `[routing]: No route for signal` errors from the
  agent; the runs still completed. The signal type is not in the log line, so this is
  unexplained noise worth a look (see the jido core note about unmatched signals in the doc).

### What the code review caught

A fresh-context review of the branch found, among other things, that the first version of
the gate sorted a probability map in which most tools scored exactly 0.0, so "top 3" was
really the argmax plus two alphabetical tie-fillers (`add` appeared in 30 of 30 gated
turns). The gate now drops zero-probability tools, and the numbers above are from the
re-run. The first run's results were 28/30 at $0.00240 and 1,081 input tokens, so the
conclusion did not change, but the "gates to 2-3 tools" description did. The review also
had the runner stop putting API keys on the `sudo docker run` command line (now a
mode-600 env file), made `run.sh <proj> test` actually use the test env, isolated bench
crashes to a failed row instead of losing the batch, and closed two fail-open gaps
(a 200 with a null `answers` body; a client raise).

### Caveats

- 30 tasks, one run each. Pass-rate differences of one task are noise (Sonnet went 29/30
  then 30/30 across the two runs).
- Tasks were chosen so the fast model can do them; nothing here needed Opus. That makes
  this a test of gating and overhead, not of rescuing hard tasks.
- Cost for `jev` is an upper bound (all turns priced at the max tier chosen).
- The bench does not independently confirm which model served a turn; the routing claim
  rests on jido_ai honouring the `model` override, which the two-tool latency supports.
- Three bugs were found and fixed during the smoke tests, all on my side: a tool schema
  that rejected numbers the LLM passed as strings, a wrong gold answer that ignored the
  tool's rounding, and a request-id lookup that missed the decision log until the bench
  switched to `ask/3` + `await/2`.

### Running it

```sh
cd experiments/jev_routing/elixir
./run.sh typesafe_client test          # client suite (20 tests, Req.Test, no network)
./run.sh . test                         # transformer, tools, tasks, bench helpers (28 tests, stubbed Jev)
./run.sh . run -e 'JevRouting.Bench.main([])'                       # full bench, ~90 runs
./run.sh . run -e 'JevRouting.Bench.main(["--tasks","t01,t15","--conditions","jev"])'
```

Needs Docker with passwordless sudo, `~/.typesafe_key`, and `~/.anthropic_key` (read into
a temporary mode-600 env file; never on the command line).

## Files

| File | What |
| --- | --- |
| `jev_eval.py` | Extracts the 19 actions from `lib/jido/actions/*.ex`, sends each query with four questions (Noul, Choice, complexity Score, risk Score), reports accuracy by confidence band, thresholds, latency, tokens. `JEV_API` and `JEV_MODEL` point it at any System One endpoint. Python 3 stdlib only. |
| `jev_eval_results.json` | Raw per-query answers, probabilities, latencies for both criteria variants and the 8-way parallel run. |
| `clm_smoke.py` | Sends CLM's README quickstart to a local CLM server and checks the answers against what other stacks get for it, so a wrongly wired encoder shows up before the benchmark runs. It checks wiring, not accuracy. Stdlib only. |
| `ref_encoder.py` | Qwen3-8B through Hugging Face transformers, behind the same `/v1/embeddings` API as `vllm serve`: a reference to check a local encoder against. Needs `torch` and `transformers`. |
| `clm_finetune/` | Fine-tuning CLM's head on Jido tool selection. `train_requests.py` (874 training requests) and `make_data.py` (builds the train/test parquet `finetune.py` reads; refuses near-copies of the benchmark requests); `make_external_data.py` (ToolACE rows in the benchmark's shapes); `embed_cache.py` (resumable embedding into `finetune.py`'s cache); `replay_finetune.py` (trainer with optional Nemotron replay); `logreg_baseline.py`; `jevstiller_student.py` (Jevstiller's encoders and student on our data); `holdout.py` and `jev_holdout.py` (the 45-request holdout and its Jev scorer); `results/`. |
| `llm_baseline.py` | Same 19 actions as Anthropic tool definitions, same 34 queries, native tool-calling. `COND=select` switches to the selection-only prompt. Needs the `anthropic` package. |
| `llm_baseline_default.json`, `llm_baseline_select.json` | Raw per-query results for Haiku 4.5 and Opus 5 under each prompt. |
| `model_routing.py` | Experiment 2: 42 tasks on three tiers, outcome-based gold, Jev routing and verification, policy scoring. `--regrade` re-scores offline; `--reroute` has another System One model (a local CLM) route and verify the stored replies. |
| `model_routing_results.json` | Raw replies, grades, Jev answers, and policy summary for experiment 2. |
| `embed_baseline.py` | Experiment 3: BM25 and local embedding models (fastembed) on the experiment 1 queries. |
| `embed_baseline_results.json` | Per-query top-2 matches, margins, and summaries for each model. |
| `elixir/` | Experiment 4: `typesafe_client` package, `jev_routing` transformer + bench, `SPEC.md`, `PLAN.md`, `run.sh`. |
| `bench_results_elixir.json` | Raw per-run rows for experiment 4 including every per-turn Jev decision. |

## Running

```sh
# Jev: key in TYPESAFE_API_KEY or ~/.typesafe_key
python3 experiments/jev_routing/jev_eval.py

# LLM baseline: key in ANTHROPIC_API_KEY or ~/.anthropic_key
cd experiments/jev_routing
uv venv && uv pip install anthropic fastembed rank-bm25
.venv/bin/python llm_baseline.py                 # assistant prompt
COND=select .venv/bin/python llm_baseline.py     # selection-only prompt
.venv/bin/python llm_baseline.py claude-sonnet-5 # other models as args
```

About 100 Jev requests and 70 Anthropic requests per full run. Neither script
prints or stores a key.

### Against a local CLM

[CLM](https://github.com/Contrastive-LM/CLM) serves the same `POST /v1/systemone` API from a
frozen Qwen3-8B encoder and a 75 MB projection head, so the scripts here can point at it with
two variables: `JEV_API` (the endpoint) and `JEV_MODEL` (`clm-latest`). The TypeSafe key is
only ever sent to the TypeSafe API; `JEV_API_KEY` covers a CLM server started with
`CLM_API_KEY`. Any model other than `jev-latest` writes its own files
(`jev_eval_results_clm-latest.json`, `model_routing_results_clm-latest.json`,
`../jev_model_routing/jev_routes_clm-latest.jsonl`), so a local run never overwrites Jev's.

On Apple silicon (macOS 15+; the encoder's bf16 weights are about 16 GB):

```sh
# 1. The encoder: vLLM on Metal with last-token pooling, which is what the head was trained on
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b \
  --runner pooling --max-model-len 2048 --port 8090

# 2. The CLM API on :8700, in its own venv. --no-deps skips CLM's vllm requirement, which
#    has no macOS wheel; the heads run on CPU. The 75 MB head downloads on first start.
git clone https://github.com/Contrastive-LM/CLM && cd CLM
uv venv -p 3.12 && uv pip install --no-deps -e . && uv pip install numpy requests torch fastapi uvicorn
.venv/bin/clm-serve --device cpu --emb-url http://127.0.0.1:8090/v1/embeddings

# 3. From this repo: is the stack wired like everyone else's? Then the benchmark.
python3 experiments/jev_routing/clm_smoke.py
export JEV_API=http://127.0.0.1:8700/v1/systemone JEV_MODEL=clm-latest
python3 experiments/jev_routing/jev_eval.py      # -> jev_eval_results_clm-latest.json
python3 experiments/jev_routing/model_routing.py --reroute   # -> model_routing_results_clm-latest.json
```

`clm_smoke.py` sends the quickstart request from CLM's README and fails if an answer is off from
what independent stacks get for it (CUDA with vLLM 0.30, an RTX 4090, MLX, PyTorch MPS, Hugging Face
all agree: urgency 0.84, billing 0.99, frustration 2.0, 98 encoder tokens). The README's own
numbers (0.41, 0.94, 1.98, 106 tokens) don't reproduce for anyone, including at the commit it
shipped in ([CLM issue 15](https://github.com/Contrastive-LM/CLM/issues/15)), so they are not the
check.

When it reports MISMATCH, `ref_encoder.py` checks the encoder against an independent
implementation: the same last-token embedding, computed with Hugging Face transformers instead of
vLLM.

```sh
# stop vllm serve first; this needs the same ~16 GB. In the CLM checkout:
uv pip install transformers
.venv/bin/python <jido>/experiments/jev_routing/ref_encoder.py    # serves :8090, on MPS
# restart clm-serve, which keeps every embedding it has seen, then from this repo:
python3 experiments/jev_routing/clm_smoke.py
```

If the smoke check now passes and vllm serve's did not, the vllm serve encoder was wrong, and the
benchmark can run on the reference encoder instead (slower, one text per forward pass). If the numbers
are the same, the encoder is fine; on a Mac Studio, vllm-metal matched it to cosine 0.9998 or better.
With the two variables exported,
`../jev_model_routing/route_jev.py` and `analyze.py` use CLM too, and `analyze.py --pilot`
computes the Experiment 1 signal AUCs from CLM's answers.

For Experiment 2, `model_routing.py --reroute` keeps the model replies and grades stored in
`model_routing_results.json` and asks CLM only for the routes and the verification verdicts:
126 CLM calls, no Anthropic calls or key. Its policy table has the same rows as the Jev run's,
and only the jev-* rows can differ, because the replies, grades and gold tiers are identical.
Without the flag the script would rerun the ~120 model calls as well.

Reading the results next to Jev's:

- Accuracy, the confidence bands, none-detection and the coverage/precision curve compare
  directly. Both define confidence as the top probability minus the mean of the rest (Jev's
  recorded answers match that to within 0.02), but each model calibrates its probabilities
  differently, so compare the curves rather than a single threshold.
- Latency is a Mac against a hosted GPU. CLM also caches every text it embeds, so quote the
  discovery pass: the schema pass reuses its state texts, and the 8-way pass is all cache hits.
- CLM's `tokens_in` counts only encoder tokens spent on cache misses, so it doesn't compare
  with Jev's.

#### Result: the released CLM head does not do this task

Run on a Mac Studio (M2 Max): Qwen3-8B on vllm-metal, `clm-serve --device cpu`, head
`CLM_v0.1-8B.pt`. Raw files: `jev_eval_results_clm-latest.json`, `jev_eval_results_clm-raw.json`,
`model_routing_results_clm-latest.json`, `model_routing_results_clm-raw.json`. `clm-raw` is CLM's own
ablation: cosine in the encoder's embedding space, no head.

| Tool selection, 34 requests, 19 actions plus none | Jev | CLM head | CLM raw |
| --- | --- | --- | --- |
| Correct, action descriptions only | 33/34 | 8/34 | 4/34 |
| Correct, descriptions plus parameter docs | 33/34 | 2/34 | 1/34 |
| Requests that need an action, descriptions only | 29/29 | 6/29 | 4/29 |
| "None of these" detected (5 requests) | 4/5 | 2/5 | 0/5 |
| Answers at confidence 0.85 or more, descriptions only | 24, all right | 0 | 0 |
| Most common answer | none (4) | `mark_working` (17) | `mark_working` (24) |
| Median latency, descriptions only | 134 ms | 306 ms | not measured |

Always answering "none" scores 5/34, a uniform guess 1/20. With parameter docs the head answers
`stop_child` for 33 of 34 requests, at 0.5 to 0.95 confidence, including "Book me a flight to Denver."

| Model-tier routing, 42 stored tasks | Jev | CLM head | CLM raw |
| --- | --- | --- | --- |
| Choice policy: pass rate, $ per task | 0.86, $0.0005 | 0.95, $0.0008 | 0.90, $0.0025 |
| Tiers picked (fast / capable / reasoning) | 35 / 6 / 1 | 0 / 39 / 3 | 0 / 0 / 42 |
| Cascade at 0.5: pass rate, $ per task | 0.88, $0.0010 | 0.74, $0.0005 | 0.74, $0.0005 |
| "Is this answer wrong" ranks a wrong fast answer above a right one | 83% of pairs | 43% | 45% |

CLM's 0.95 in the first row is not routing: 39 of 42 tasks went to the capable tier, which is
the always-capable policy (0.93) plus three lucky picks. Below 50% on the last row is worse than
chance, so its cascade never escalates and lands on always-fast (0.74).

This is the released head, not the setup. Checked:
- **Encoder.** vllm-metal and an independent Hugging Face bf16 run of Qwen3-8B agree to cosine
  0.9998 or better on every text the smoke check sends, and CLM's own engine gives the same
  answers on either set of vectors. Token counts match the Qwen3 tokenizer exactly. Texts embedded
  together or one at a time give identical vectors.
- **Layout.** Four state layouts, fixed before running (as benchmarked, request last, request
  only, request plus context without the action list) give 8, 10, 4 and 2 of 34 correct.
- **Upstream.** [Issue 15](https://github.com/Contrastive-LM/CLM/issues/15) reports the same
  collapse on CUDA: the state vectors stay 0.947 cosine apart even after the head, so answers
  barely depend on the request. [Issue 3](https://github.com/Contrastive-LM/CLM/issues/3)
  reports score questions returning "Very angry" for every state, reproduced on CUDA, MLX and MPS.
  Neither had a fix or a maintainer reply when checked on 2026-09-29, and no open pull request
  touches the schema, engine or head.

This is the released `clm-latest` head zero-shot. CLM is meant to be fine-tuned on a domain's own
decisions, which is the next section.

#### Result: fine-tuned on tool selection

`train/finetune.py --task choice` (defaults, warm-started from the released head) on 437 requests
I wrote for the 19 actions: 20 per action plus 57 requests no action fits (`clm_finetune/`), later doubled
to 874 (40 per action, 114 that fit none). The
34 benchmark requests are the test split. They are not in the training set, and `make_data.py`
refuses to build if a training request is close to one (closest pair: 0.76 similarity, limit 0.8).
Nothing was chosen on the test split: every run used the script's defaults, early stopping used a 10%
validation split, and every run's test score is reported. Trained on CPU with the
Mac's embeddings in 2.6 minutes. The head file is not committed; `make_data.py` and the command
below rebuild it.

| Tool selection, 34 requests, descriptions only | Jev (zero-shot) | CLM released | CLM fine-tuned |
| --- | --- | --- | --- |
| Correct | 33/34 | 8/34 | 28/34 |
| Requests that need an action | 29/29 | 6/29 | 25/29 |
| "None of these" detected (5 requests) | 4/5 | 2/5 | 3/5 |
| Confidence 0.6 or more: share answered, precision | 91%, 97% | 0%, n/a | 32%, 100% |
| Confidence 0.85 or more: share answered, precision | 71%, 100% | 0%, n/a | 12%, 100% |

| Training requests per action | 0 (released) | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned CLM head, correct of 34 (`finetune.py`'s own metric) | 7 | 15 | 20 | 28, 28, 28 | 28, 25, 27 |
| Logistic regression on the frozen encoder's embeddings, correct of 34 | n/a | 25 | 30 | 31 | 31 |

20 and 40 per action show three seeds each (validation split and initialization change; the test
split does not). Validation accuracy on my own phrasing rose from 0.82-0.86 to 0.82-0.89 at 40, while
the benchmark's did not: the ceiling looks like my wording against the benchmark's, not data volume.

- Fine-tuning fixes the collapse. Answers spread over the actions, and confidence now means
  something: everything above 0.6 is right, where the released head never got past 0.85.
- With `finetune.py`'s defaults, doubling the data to 40 per action did not help (28, 25 and 27 of 34).
  With a differently configured trainer it did (28-30 at 20 per action, 30-31 at 40; see the next section).
- **A logistic regression on the same frozen embeddings does better than `finetune.py`'s tuned head, and
  as well as the best head I trained**: 31/34 at 20
  and 40 per action, 30/34 at 10, with no head and no contrastive training, from the state embedding
  alone (`clm_finetune/logreg_baseline.py`, regularization picked by cross-validation on the training
  rows). Its three misses are the same at both sizes ("Spin up three workers for the crawl", "Stop being
  idle, get to work", "Delete the user's account permanently"), and its confidence separates right from
  wrong (0.93 against 0.71 mean at 20 per action). So the frozen Qwen3-8B encoder carries the signal
  and CLM's head adds nothing over a linear probe here. A third-party benchmark reports the same ordering
  on Banking77 and CLM's own typed-decisions set ([CLM PR 13](https://github.com/Contrastive-LM/CLM/pull/13), open).
  The classifier only knows the 19 actions it was trained on, like the tuned head; PR 13 finds the
  fine-tuned head weak on intents it never trained on (0.40-0.46), so neither is a general router.
- Jev is ahead (33/34 against 28/34), but with 34 requests the gap is not conclusive: 95% intervals
  are roughly 85-99% and 66-92%.
- The two are not the same kind of result. Jev needed no examples. The tuned head needed about 440,
  written by me after seeing the benchmark requests, and it only works for this action list in this
  request format: on the version with parameter docs, which it never saw, it collapses again (2/34
  released, 11/34 tuned). Model-tier routing was not retrained; the tuned head does not apply there.
- Encoder cost is the same as the released head (about 0.3 s per request cold on this Mac). The
  head itself adds nothing measurable, and nothing here costs per call.

```sh
python3 experiments/jev_routing/clm_finetune/make_data.py /tmp/tooldata      # needs pyarrow
cd CLM && python train/finetune.py --task choice --data /tmp/tooldata --workflow all \
  --init-ckpt "$(clm-download)" --out-dir runs/tools --embed-url http://127.0.0.1:8090/v1/embeddings \
  --served-model-name qwen3-8b --embed-model Qwen/Qwen3-8B --max-len 2048     # needs torch, transformers, pyarrow
clm-serve --model clm-tuned=runs/tools/best_head.pt --port 8701 ...          # then JEV_MODEL=clm-tuned
```

Raw: `jev_eval_results_clm-tuned.json`, `clm_finetune/results/finetune_runs.json`.

What CLM's [announcement post](https://contrastive-lm.notion.site/) says that bears on this:
- The head is trained mostly on question-answer pairs (60M from Nemotron DQA, then 30M synthetic hard
  negatives, then 1M agent trajectories), and the post's own numbers say fine-tuning on one narrow
  domain forgets: with 40% Nemotron data replayed alongside the agentic data, hard-negative accuracy
  stays at 68.5%, and on agentic data alone it falls to 56.2%. My fine-tune had no replay.
- It reports the optimal head size growing with data at about 310 tokens per parameter. The 20M-parameter
  head against a few hundred examples is far from that, which fits a linear probe winning here.
- It claims zero-shot parity with Jev on tool-calling and computer use. Independent reproductions
  (issues 3 and 15) and this benchmark don't support that for the released head, and its own
  headline results are fine-tuned reward models on DeepSWE and Terminal-Bench, not zero-shot routing.
  Its claim that Jev fails as a long-horizon verifier is a different task from anything measured here.
#### More data: the paper's datasets, ToolACE and replay

Of the datasets in CLM's paper, I tried one and skipped the rest:
- **Nemotron DQA (60M question-answer pairs), as replay.** CLM publishes it as precomputed embeddings
  (`Contrastive-LM/CLM-v0.1-Pretrain-Nemotron`); I used one chunk of 100,000 pairs, 2,000 held out, for the
  40% replay the post describes.
- Not tried: the 30M synthetic hard negatives (a pre-training stage for a whole head, not a fine-tuning
  add-on); the agent trajectories (ADP, Endless-Terminals, LiteCoder-Terminal-SFT; step data with a
  different kind of action from choosing among named tools); `LocalLLaMA/typed-decisions` (CLM's own
  fine-tuning evaluation, a different domain).

Instead I built tool-selection rows from a public tool-calling corpus the paper doesn't use, ToolACE
(11,300 conversations, Apache-2.0), with `make_external_data.py`: 2,400 rows in the benchmark's shapes
(same state layout and question, 20 options padded with other rows' tools, 13% with the answer removed so
"none" is right). No Jido action or request is in it. `replay_finetune.py` trains with or without replay
(constant learning rate 5e-4, softmax over each row's own options). Test split as before; the test score
is reported for every run and the epoch is chosen on validation only. Correct of 34, each seed shown:

| In-domain requests per action | 0 | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- | --- |
| Jido examples only, this trainer | n/a | 10, 11, 14 | 11, 24, 27 | 28, 28, 30 | 30, 31, 31 |
| ToolACE plus Jido examples, replay 0.4 | 7 | 20, 24, 25 | 26, 26, 28 | 31, 31, 31 | 30, 32, 33 |
| Jido examples only, `finetune.py` defaults | n/a | 15 | 20 | 28, 28, 28 | 25, 27, 28 |
| Logistic regression, Jido examples only | n/a | 25 | 30 | 31 | 31 |

- **ToolACE alone does not transfer.** 81-85% on its own validation split, 7 of 34 on Jido (10 with
  `finetune.py`). The head learns the training tools, not tool selection: the same thing [CLM PR 13](https://github.com/Contrastive-LM/CLM/pull/13)
  saw on intents it never trained on.
- **As a supplement it helps, most where in-domain data is scarce**: 23 against 12 correct on average at 5
  per action, 27 against 21 at 10, and it removes the unstable runs (Jido-only at 10 had a seed stuck at
  11). At 20 it's 31 against 29 and at 40 it's 32 against 31, within noise.
- **The ceiling is about Jev's level.** The best heads (31-33 of 34 at 20-40 per action) match the logistic
  regression (31) and approach Jev (33). None passed it.
- **Replay does nothing for tool accuracy** (equal or one apart). It does hold the head's question-answer
  retrieval, measured as top-1 among 100 answers on held-out Nemotron pairs: 0.826 for the released head,
  0.809-0.8185 after fine-tuning without replay, 0.816-0.8265 with it. The forgetting the post describes
  is real and small at this scale, about one point.
- **The trainer matters and I don't know why.** This trainer beat `finetune.py` on the same rows
  (30-31 against 25-28 at 40 per action), and `finetune.py` with `--loss softce --targets hard` scored
  26-28 at 20 and 28 at 40, so it isn't the loss. `finetune.py` uses a OneCycle learning-rate schedule and
  gradient clipping; this trainer uses neither. I didn't isolate which.

Is a tool-selection dataset worth building? Public tool-calling data is not: alone it teaches nothing that
transfers, and as a supplement it saves perhaps half of the in-domain examples at small sizes. What pays is
in-domain data: about 20 examples per tool, in the wording your real requests use (the gap between my
phrasing and the benchmark's is the likely ceiling). Before building anything, fit the logistic regression:
it needed the same examples, matched the best head and runs with no training infrastructure. To know
whether any of it holds up, hold out requests from real traffic, not written ones.

#### A second test set: 45 requests written after everything else

`clm_finetune/holdout.py`: 45 hand-written requests (36 that need an action, every action at least once, and 9 near-misses
with none) in a register meant to differ from both the benchmark and the training sets, with a guard against
near-copies of either. Which models were scored, and how, was fixed before any of them saw it, and nothing was chosen on it.
It is more independent than the benchmark but I wrote it too, so it is not real traffic. Correct of 45:

| Model | Correct of 45 |
| --- | --- |
| Jevstiller's student, bge-base, gold labels, 40 per action | 41, 41, 41 |
| Jevstiller's student, bge-small | 38, 40, 36 |
| Jev, zero-shot (the request text as the state) | 39 |
| Logistic regression on Qwen3-8B embeddings, 40 per action | 39 |
| CLM head, ToolACE plus Jido examples | 37, 35, 33 |
| CLM head, Jido examples only | 36, 36, 32 |
| Jevstiller's student on Qwen3-8B, request text only | 33, 34, 34 |
| CLM released head | 5 |

- **Jev misses 6 of 45, and the field is within a few requests of it.** Three of Jev's misses send a "none" request to `reply`
  ("How much memory are you using?", "Explain what a cron expression is"), where `reply` is arguably defensible and my gold label is
  debatable. The rest ("Kill the worker named 'thumbnailer'", "let it go", "Ping <0.412.0>") are also missed by most other models.
  Its confidence is informative: 0.87 on right answers against 0.50 on wrong ones.
- **The tuned CLM heads drop about 10 points from the benchmark to here and the small-encoder students about 5.** The holdout's wording
  is further from the training data.
- With 45 requests a difference of two to four is noise. The students trained on gold labels would inherit Jev's errors if taught
  by Jev instead (see the replay below).

#### Jevstiller's student

[Jevstiller](https://github.com/tomerglick57/Jevstiller) (Apache-2.0, alpha) is a proxy in front of Jev's `Choice` calls. It
distills Jev's own answers, with their probability distributions, into a local student: a frozen sentence
encoder (bge-small by default, on CPU) plus a numpy logistic regression, with an out-of-distribution gate, a
permanent audit slice sent to Jev, and a routing threshold set by a finite-sample bound so that the student
disagrees with Jev on no more than a budget you choose (2% by default). Its teacher is Jev, so its ceiling is
Jev's accuracy, and it needs a few thousand real Jev calls to take over (Banking77: about 4,000). A new class list is
a new task, trained from scratch.

That student is the linear probe that did well above, so I ran its own encoders and its own `LinearStudent`
on our gold labels (a perfect teacher; Jev's answers on our training requests would need a key), three seeds,
at 40 requests per action. `clm_finetune/jevstiller_student.py`, raw numbers in `results/jevstiller_student.json`:

| Correct | Benchmark (34) | Holdout (45) |
| --- | --- | --- |
| bge-small (384-dim, 1.8 ms per request on CPU), request text only | 31, 31, 31 | 38, 40, 36 |
| bge-base (768-dim, 6.3 ms), request text only | 31, 31, 31 | 41, 41, 41 |
| Qwen3-8B (16 GB, GPU-class), request text only | 33, 33, 33 | 33, 34, 34 |
| bge-small, the whole state JSON as the proxy embeds it | 32, 31, 32 | 33, 33, 33 |
| bge-base, the whole state JSON | 31, 31, 31 | 34, 33, 34 |
| CLM head, ToolACE plus Jido examples (above) | 30, 32, 33 | 37, 35, 33 |

- **A small CPU encoder does as well as the 8B one, and better on differently-worded requests.** On the
  holdout bge-small and bge-base score 36-41 against 33-34 for Qwen3-8B through the same student, and at 5
  examples per action they hold at 31-32 and 35-37 where Qwen3-8B falls to 26-31 and 23-25. Sentence encoders are
  trained for meaning, not for next-token prediction.
- **The state must be the request, not the whole object.** Jevstiller embeds the canonical JSON of the state.
  Ours carried the constant 19-action list, and that costs 5-7 points on the holdout (38-41 down to 33-34).
  Send the request text alone; the class list is already part of the task's identity.
- **Not measured: a Jev teacher.** I trained on gold labels. With Jev as teacher the student inherits Jev's
  errors, so 33 of 34 on the benchmark is its ceiling, and the routing threshold decides how often it answers
  locally. That needs Jev's answers on the training requests (a key and a few thousand calls) and Jevstiller's own
  replay tooling (`experiments/run.py`).
- An earlier version of this section called the teacher Claude. It is Jev; I got that from a summary of the docs page
  and corrected it from the repository.

#### Replaying through Jevstiller

`clm_finetune/jevstiller_replay.py` follows Jevstiller's own protocol (`record_answers.py`, then a replay from its answer
cache): `record` makes one Jev call per request into its cache format, and `replay` streams the 874 training requests
through the real `Jevstiller` loop with inline training, then scores the 79 held-out requests (the 34 benchmark and
the 45 holdout) at checkpoints with `evaluate()`, which never writes to its store, so the held-out rows cannot leak
into training. The state is the request text alone. A replay needs no key: Jev's 953 recorded answers
(`results/jev_answers.jsonl.gz`, gunzip it to replay) cost $0.027, about 3 cents per thousand requests, at a median of 95 ms.
Jev alone, with the request text as the state, gets 33/34 on the benchmark and 39/45 on the holdout.

Replayed on Jev's real answers with bge-small on CPU (raw: `results/jevstiller_replay_jev.json`):

| Target agreement | Library defaults (1000 / 50 per class / 500 calibration rows) | `run.py`'s protocol (500 / 5 / 200) | Small-data (200 / 10 / 100) |
| --- | --- | --- | --- |
| 98% | no student | no student | no student |
| 95% | no student | no student | student answers 42% of held-out requests (59% benchmark-style, 29% holdout-style), 100% agreement with Jev, system 33/34 and 39/45 |
| 90% | no student | no student | student answers 73% (88%, 62%), 91% agreement, system 31/34 and 36/45 |

- **874 requests are too few for Jevstiller to take over under its guarantees.** With our gold labels as a perfect teacher the
  result was the same (only the small-data column at 95% produced a student, answering 73%), so volume is the limit, not
  Jev's noise: the routing threshold has to be certified on calibration rows and there are too few. Its own Banking77 run
  took about 4,000 requests.
- **It answers familiar wording and sends unfamiliar wording to Jev**, which is the point of its out-of-distribution gate:
  at the 95% target it handles 59% of benchmark-style requests and 29% of the differently-worded holdout.
- **Loosening the target buys coverage at the price of accuracy.** At 90% it answers 73% locally but the system falls to
  36/45 on the holdout against Jev's 39.
- **Money is not the reason to do this here.** Jev costs about 3 cents per thousand of these requests. The case for a local
  student is latency (a bge-small answer is a few milliseconds against Jev's ~95-300 ms), no dependence on one vendor's API
  and its rate limit, and offline use.

#### Real traffic: Claude Code's own tool calls

I built a next-tool-prediction dataset from Claude Code session transcripts (`~/.claude/projects`; the Claude Desktop logs
hold only 6 tool calls): 60,214 calls from 1,274 sessions, 57,557 rows after scrubbing and exact de-duplication, 29 classes (Bash split by
its first command word, MCP tools by server). The data is private and stays outside the repository; only the code
(`real_calls/`: `build_real_calls.py` with its scrubber and self-test, `prepare.py`, `local_baselines.py`, `jev_label.py`) and
aggregate numbers (`real_calls/results.json`) are committed. Scrubbing covers keys and tokens of common formats,
`key=value` secrets, emails, URL query strings, home paths, IPs, long identifiers, and drops any row with private-key markers or a
secret from this machine's environment or key files (checked by value, none found). Jev received 1,140 unique scrubbed
contexts (about 440,000 characters, $0.05), a sample, with the owner's permission.

| Top-1 accuracy | Next tool given the steps so far (3,602 test rows from unseen sessions) | First tool for a user request (2,545 rows) |
| --- | --- | --- |
| Majority class | 20.9% | 18.8% |
| Jev zero-shot | 25.7% (32.3% with the assistant's narration), 300 rows | 20.2% (top-3 37.8%), 1,000 rows |
| bge-small student trained on the real labels | 29.3% (top-3 62%) | 37.1% (top-3 62.5%), 5-fold CV by session |
| Same tool as the previous call | **36.4%** | n/a |
| Student on the previous two tools alone | **38.4%** (top-3 68.8%) | n/a |

- **This is not a tool-selection task in the way the Jido benchmark is.** The labels are one assistant's habits, and many classes overlap in meaning
  (`cat` through Bash against the Read tool, `grep` against Read), so nothing in the context says which the assistant will pick.
  Repeating the last tool beats every text model, and the previous two tools alone beat the student with the text added.
- **Jev zero-shot is close to the majority class** (20.2% against 18.8% on requests). It is right where a request names a domain (git: 67 of
  169; file reading: 60 of 124) and near zero elsewhere (deferred-tool loads: 0 of 69).
- **A student trained on your own history roughly doubles it** (37.1%), because the signal is a person's and an agent's habits, which no pretrained model knows.
  Some of that may be near-identical prompts repeated across sessions.
- **So Jevstiller's design is the wrong fit for this data**: its teacher is Jev, which is 20-26% here, so a student distilled from it would be too.
  Training directly on the real labels is the only route that learns anything, and the previous-tool sequence matters more than the text.

The 2,400 ToolACE rows and 2,000 Nemotron pairs are rebuilt from public files by the scripts in
`clm_finetune/` (`embed_cache.py` embeds them resumably, since a vllm-metal crash once lost a 10-minute run);
raw runs are in `clm_finetune/results/external_and_replay_runs.json`.

## Sources

- TypeSafe API: https://docs.typesafe.ai/api
- Function calling cookbook: https://docs.typesafe.ai/cookbooks/function_calling
- Confidence-gated routing: https://docs.typesafe.ai/patterns/confidence-routing
- jido_ai model aliases and ReAct request transformer: https://github.com/agentjido/jido_ai
