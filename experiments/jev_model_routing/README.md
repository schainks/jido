# Experiment: Jev (TypeSafe System One) as a model-tier router for jido_ai

**Question.** Can Jev pick the cheapest model that will still get a request right,
from the request alone, the way it picked the right Jido action in
[`../jev_routing`](../jev_routing/README.md)? How does it compare with jido_ai's
fixed routes, its Adaptive keyword heuristic, and a small LLM asked the same question?

**Status.** The harness is built and checked offline (`selftest.py`). The labeling
run has not been executed yet: it needs Anthropic and TypeSafe keys and costs about
$17 in Anthropic calls by `label_models.py --plan`'s estimate, plus 68 Jev calls.
The pilot below reuses the tool experiment's results and needs no new calls.

**Relation to [Experiment 2](../jev_routing/README.md#experiment-2-jev-as-a-model-tier-router)
in `../jev_routing`.** That run asks the same question on 42 tasks, one sample each,
graded by exact match, unit tests, and an Opus 5 judge. Its caveats name the single
samples and the judge noise. This harness is the follow-up. It adds:
- three samples per cell
- deterministic graders only
- the 34 tool-selection requests as a second slice
- an Opus 5 effort arm
- a Haiku-as-router baseline
- Jev's cumulative-probability rule with a threshold sweep

It does not repeat Experiment 2's run-then-verify cascade.

## What changes from tool routing

1. **Nobody can write the gold label.** The right answer is "the cheapest tier that
   gets it right", so every tier runs every task and the label is measured.
2. **Mistakes cost different amounts.** A tier that's too weak costs quality; one
   that's too strong only costs money. So an unsure Jev answer should route up.
   The main rule here sums Jev's probabilities cheapest-first and picks the first
   tier whose running total reaches a threshold τ.
3. **Jev judges the work, not the model.** The criteria describe what each tier is
   for; `common.MAPPINGS` turns a tier into a model (or an effort level). A model
   upgrade is a mapping change, not a prompt change.

## Pilot, from the tool experiment's data

`llm_baseline_select.json` has Claude Haiku 4.5 and Claude Opus 5 on the same 34
requests (one sample each). Taking "cheapest model that got it right" as the label
(`python3 analyze.py --pilot`):

| Router | Correct | Cost per 1k calls |
| --- | --- | --- |
| Always Haiku 4.5 | 28/34 | $2.31 |
| Always Opus 5 | 33/34 | $13.61 |
| Perfect routing (Haiku unless it fails) | 33/34 | $4.36 |

- **Perfect routing gets Opus accuracy for about a third of the cost.** Only 5 of
  the 34 requests needed Opus.
- **All 5 leave their target unstated:** "the worker child you spawned earlier",
  "pass *this* signal along", "whoever asked". Haiku replied asking for context on
  each one. The `capable` tier's description is written from that failure.
- **The tool experiment's `complexity` Score barely separates them** (AUC 0.62).
  Its `risk` Score does (0.94), but on only 5 positives. That's a hint to test, not
  a result. This run keeps `risk` unchanged and adds an implicit-context Noul aimed
  at exactly that failure.
- Jev alone beats both models at tool selection, so this slice mainly tests the
  method. The open slice below is where a model actually does the work.

## Design

### Tiers and configs

| Tier | Model mapping | Effort mapping |
| --- | --- | --- |
| fast | Claude Haiku 4.5 (API default: no thinking) | Claude Opus 5, effort low |
| capable | Claude Sonnet 5 (API default: adaptive thinking) | Claude Opus 5, effort medium |
| reasoning | Claude Opus 5 (API default: adaptive thinking, effort high) | Claude Opus 5, API default |

The effort mapping tests the alternative to switching models: keep one model and
route its effort. One model keeps one prompt cache, and a stronger model at lower
effort can cost less per solved task than a smaller one. Server-side refusal
fallbacks are left off on purpose: a fallback would answer with a different model
and credit that answer to the tier being measured. A refusal is recorded and graded
as a miss.

### Tasks (68)

| Slice | Tasks | Prompt | Graded by |
| --- | --- | --- | --- |
| tool | 34, the tool experiment's requests over the 19 actions in `lib/jido/actions/*.ex` | Selection-only prompt from `llm_baseline.py`, actions as tools, `max_tokens` 4,096 (the pilot's 512 truncated one Opus call) | First tool called is a gold tool, or no call when gold is "none" |
| open | 34 self-contained tasks: 8 easy, 12 medium, 14 hard by prior guess (lookups, reformatting, cron, dates, JSON extraction, regex, arithmetic, puzzles) | Asks for a final `ANSWER:` line, `max_tokens` 16,000 | Deterministic graders: exact, numeric, cron equivalence, JSON fields, dependency order, regex test strings, and so on |

No judge model. `selftest.py` re-derives every open-slice answer independently. It
brute-forces the puzzles and checks they have one solution, runs the prompt's own
code, and parses the CSV, graph and knapsack prompts. The prior bands only break
down results; no router sees them.

### Labels

Each config runs each task 3 times. **Label = the cheapest tier whose config passes
at least 2 of 3**, or `unsolved`. Labels are computed separately for each mapping.
Every task runs on every config, so routers are scored counterfactually. On each task
a router gets the pass rate, cost and latency of the config its tier maps to, plus
its own call.

### The Jev request

One request per task (`route_jev.py`), same `state` shape as the tool experiment:
`request`, `available_actions` (empty for the open slice), `context`. Four questions:

| Question | Type | Asks |
| --- | --- | --- |
| `tier` | choice | the least capable tier that can fully and correctly handle `request` |
| `difficulty` | score, 3 levels | how much careful reasoning a correct answer takes |
| `risk` | score, 3 levels | the tool experiment's blast-radius question, unchanged |
| `implicit_context` | noul | does `request` lean on a target or value it does not state |

Tier criteria (`common.TIER_CRITERIA`):

- **fast**: one obvious step with everything stated: answer directly, run one clearly named action, classify, reformat, or give a short reply
- **capable**: a few dependent steps, or a target or parameter that must be inferred from context rather than read from the request
- **reasoning**: planning, debugging, math or code that must be exactly right, search over many possibilities, synthesis across many sources, or a costly judgment call

### Routers compared (`analyze.py`)

- **Always fast / capable / reasoning.** Always capable is also what jido_ai's
  `ModelRouting` plugin does by default: every `chat.message` goes to `:capable`.
- **jido_ai Adaptive heuristic.** A port of
  `Jido.AI.Reasoning.Adaptive.Strategy.calculate_complexity/1`
  ([jido_ai@17c10b5](https://github.com/agentjido/jido_ai/blob/17c10b5/lib/jido_ai/reasoning/adaptive/strategy.ex)).
  It scores word count, sentence count, keywords, questions and must/should words;
  its default bands (0.3, 0.7) become tiers. Adaptive uses them to pick a strategy
  for one fixed model.
- **Haiku 4.5 picks the tier** (`route_llm.py`). Same tier descriptions and the
  same request information as Jev; the counterpart of the tool experiment's native
  tool-calling baseline.
- **Jev choice, argmax.**
- **Jev cumulative ≥ τ**, τ ∈ {0.5, 0.7, 0.8, 0.9, 0.95}.
- **Jev cumulative ≥ 0.8 with a risk floor.** `risk` > 1.5 means never fast.
- **Jev cumulative ≥ 0.8 with an implicit-context bump.** Noul ≥ 0.5 means never
  fast. Both variant settings were fixed before any data.
- **Jev difficulty Score, split into thirds.**
- **Perfect routing** (the ceiling): per task, the tier with the best pass rate,
  cheapest on ties.

### Reported (`analyze.py` prints these tables)

- Labels by slice.
- Per config: pass rate, cost per 1k tasks, median and p95 latency, output tokens,
  refusals and `max_tokens` stops.
- Per router under each mapping:
  - pass rate
  - cost per 1k tasks, and as a share of always-reasoning
  - median and p95 end-to-end latency
  - under- and over-route rates against the label
  - whether it's on the cost/quality frontier
- AUC of each signal for "needs more than fast" and "needs reasoning": Jev's tier
  probabilities, `difficulty`, `risk`, `implicit_context`, and the Adaptive score.
- Jev's choice against the label, as a confusion table.

## How it would plug into jido_ai

[Experiment 4](../jev_routing/README.md#experiment-4-inside-the-loop-elixir-live-jidoaiagent)
already implements this seam and benchmarks it in a live `Jido.AI.Agent`.
`JevRouting.Transformer` in `../jev_routing/elixir` asks Jev before every turn, gates
`tools`, and picks the model alias from a depth Score. The notes below still apply.
The one design this harness scores that the transformer does not use is a single
tier per run, chosen by the cumulative rule.

- **The ReAct `request_transformer` is the right place.**
  `Jido.AI.Reasoning.ReAct.RequestTransformer` documents per-turn model selection:
  return `model:` in the overrides and the next LLM turn uses that model. The hook
  runs in the task-based ReAct runner, off the agent's mailbox. It gets `run_id` in
  its runtime context, and the `:llm_started`/`:llm_completed` events record the
  model each turn used.
- **Not `Jido.AI.Plugins.ModelRouting`.** Its `handle_signal/2` runs synchronously
  inside the AgentServer process (`run_plugin_signal_hooks/2` in
  `lib/jido/agent_server.ex`), so a Jev call there would stall every signal. Keep
  its static table as the fallback when Jev is slow or down.
- **Overrides last one turn.** Every turn starts again from `config.model`, so the
  transformer re-sends the pinned tier each turn.
- **Route once, escalate one way.** Call Jev on turn 1 and pin the tier by
  `run_id`. Re-judging every turn costs a Jev call each time (Experiment 4 measured
  about 100 ms a turn). Each model switch also discards the prompt cache, which is
  per model. Per-turn calls still make sense for tool gating, which is what
  Experiment 4 does.
- **The default aliases collapse.** jido_ai maps `:capable`, `:thinking`,
  `:reasoning` and `:planning` all to `claude-sonnet-4-20250514`. Set
  `config :jido_ai, :model_aliases` first, or routing between them changes nothing.
- **Effort routing needs no model switch.** ReqLLM's Anthropic provider maps
  `llm_opts: [reasoning_effort: :low]` onto each model's thinking settings.

The rule this harness scores, as it would sit in a transformer like Experiment 4's
(untested). It picks the cheapest tier whose cumulative probability clears the
threshold, so an unsure answer fails up. Call it on the first turn, then pin the
result by `run_id` (for example in ETS). Re-send it as `model:` on every turn,
because overrides last one turn.

```elixir
@tiers [:fast, :capable, :reasoning]

defp cheapest_sufficient(probs, threshold) do
  @tiers
  |> Enum.scan(0.0, &(&2 + Map.get(probs, Atom.to_string(&1), 0.0)))
  |> Enum.zip(@tiers)
  |> Enum.find_value(List.last(@tiers), fn {cum, tier} -> cum >= threshold and tier end)
end
```

## Results

_Not run yet._ After the steps under [Running](#running), paste the output of
`python3 analyze.py` here and commit the raw files next to it.

## Caveats

- Written by one person. The open slice is short, self-contained puzzles and lookups;
  real agent traffic is longer, multi-turn and messier.
- One turn per task. In a ReAct loop a weak tier can cost extra turns or retries.
  This measures cost per call, which equals cost per completed task only for
  one-turn work.
- No prompt caching. Per-model caches favour staying on one model across a
  multi-turn run, which the effort mapping keeps.
- Three samples per cell: differences of a few points are noise. Jev is called once
  per task.
- Labels depend on the prompts, `max_tokens` and model versions. Re-run when any of
  them change; the Jev question does not need to.
- Jev's per-call price was not in its API response in the tool experiment, so Jev
  rows show "+ Jev". Pass `--jev-cost` once you know it.
- The Adaptive heuristic was designed to choose reasoning strategies, not models;
  it is here as the cheapest in-tree signal, not a straw man.
- Prices are first-party list prices per MTok: Haiku 4.5 $1/$5, Sonnet 5 $2/$10,
  Opus 5 $5/$25. Output tokens include thinking.
- Every router here sees only the request. Experiment 2 found that its remaining
  misses were the fast model fumbling easy tasks, which no request-only router can
  predict. Only its cascade (run fast, ask Jev whether the answer is wrong, then
  escalate) addresses that, and this harness does not repeat it.

## Files

| File | What |
| --- | --- |
| `common.py` | Tiers, tier criteria, configs, mappings, prices, JSONL and client helpers. |
| `tasks.py` | The 68 tasks, answer parsing and the graders. Reuses `../jev_routing/jev_eval.py` for the tool slice. Stdlib only. |
| `label_models.py` | Runs every config on every task N times. Appends `runs.jsonl` and resumes after interruption. `--plan` estimates cost, `--smoke` sends one request per config. Needs `anthropic`. |
| `route_jev.py` | One Jev request per task; writes `jev_routes.jsonl`. Stdlib only. |
| `route_llm.py` | Haiku 4.5 picks a tier per task; writes `llm_routes.jsonl`. Needs `anthropic`. |
| `analyze.py` | Regrades stored replies, computes labels, scores every router and prints the tables; writes `summary.json`. `--pilot` prints the pilot. Stdlib only. |
| `selftest.py` | Offline checks: answers, graders, heuristic port, request shapes, and the whole pipeline on fakes. No keys, about 2 s. |

## Running

```sh
cd experiments/jev_model_routing
python3 selftest.py                        # offline, no keys
uv venv && uv pip install anthropic
.venv/bin/python selftest.py               # adds request-shape checks through the real SDK
.venv/bin/python label_models.py --plan    # call count and rough cost, no calls
.venv/bin/python label_models.py --smoke   # one tool and one open task per config
.venv/bin/python label_models.py           # 1,020 calls; interrupt and rerun freely
python3 route_jev.py                       # 68 Jev calls
.venv/bin/python route_llm.py              # 68 short Haiku calls
python3 analyze.py                         # tables for Results; writes summary.json
```

Keys are read as in the tool experiment:
- Anthropic: `ANTHROPIC_API_KEY` or `~/.anthropic_key`, with the optional
  `~/.anthropic_workspace`. Without either key, the SDK falls back to
  `ANTHROPIC_AUTH_TOKEN` or an `ant auth login` profile.
- Jev: `TYPESAFE_API_KEY` or `~/.typesafe_key`.

To route with a local CLM instead of Jev, export `JEV_API` and `JEV_MODEL` before
`route_jev.py` and `analyze.py`. They then read and write `jev_routes_<model>.jsonl` and
`summary_<model>.json`, leaving Jev's files alone; setup is in
[`../jev_routing/README.md`](../jev_routing/README.md#against-a-local-clm).

No script prints or stores a key. `--configs haiku,sonnet,opus` skips the effort
arm and cuts cost by about 45 %.

## Sources

- Experiments 1-4: [`../jev_routing/README.md`](../jev_routing/README.md)
- TypeSafe API: https://docs.typesafe.ai/api
- Confidence-gated routing: https://docs.typesafe.ai/patterns/confidence-routing
- jido_ai at 17c10b5:
  - [`RequestTransformer`](https://github.com/agentjido/jido_ai/blob/17c10b5/lib/jido_ai/reasoning/react/request_transformer.ex)
  - [`ModelRouting`](https://github.com/agentjido/jido_ai/blob/17c10b5/lib/jido_ai/plugins/model_routing.ex)
  - [`ModelAliases`](https://github.com/agentjido/jido_ai/blob/17c10b5/lib/jido_ai/model_aliases.ex)
  - [Adaptive strategy](https://github.com/agentjido/jido_ai/blob/17c10b5/lib/jido_ai/reasoning/adaptive/strategy.ex)
- ReqLLM's Anthropic effort mapping: https://github.com/agentjido/req_llm/blob/main/lib/req_llm/providers/anthropic.ex
