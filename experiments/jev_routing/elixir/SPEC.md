# Spec: Jev routing prototype inside a live Jido AI agent

Date: 2026-09-26. Status: approved design, prototype.

## Purpose

Measure, at the level of a whole agent run rather than a single call, what a
Jev-backed `request_transformer` does to a `Jido.AI.Agent` running the ReAct loop:
correctness, LLM turns, wall time, tokens and dollars per completed task, and the
overhead Jev adds. The Python experiments one level up measured single calls; this
is the demo a maintainer can believe, and it is written so the transformer and the
Jev client could move into jido_ai and Hex respectively without rewriting.

## Non-goals

- No changes to jido or jido_ai. Everything uses public behaviours.
- No argument filling by Jev; the LLM still fills tool arguments.
- No Gemini / Vertex leg (needs Google credentials; parked).
- Not a benchmark. About 30 hand-written tasks, one run per condition.

## Layout

```
experiments/jev_routing/elixir/
  SPEC.md                      this file
  run.sh                       Docker entrypoint (sudo docker run ... mix ...)
  mix.exs                      app :jev_routing, deps: jido_ai ~> 2.3, typesafe_client (path)
  config/config.exs            model aliases, req_llm key from env
  lib/jev_routing/transformer.ex
  lib/jev_routing/tools/*.ex   synthetic pure Jido.Action modules
  lib/jev_routing/tasks.ex     task list + grader
  lib/jev_routing/bench.ex     conditions, runner, report
  lib/jev_routing/agent.ex     the Jido.AI.Agent under test
  test/                        ExUnit: transformer with stub client, grader, tools
  typesafe_client/             SEPARATE Mix project (publishable)
    mix.exs                    app :typesafe_client, deps: req, jason
    lib/typesafe_client.ex     public API + behaviour
    lib/typesafe_client/{http,stub,answer}.ex
    test/
```

## Component: `typesafe_client` (separate package)

Purpose: a small, provider-specific client for TypeSafe's System One API
(`POST https://api.typesafe.ai/v1/systemone`) with typed answers. No knowledge of
Jido, jido_ai, or this experiment. Depends only on `req` and `jason`.

Public API:

```elixir
@callback evaluate(state :: term(), questions :: map(), opts :: keyword()) ::
            {:ok, %{String.t() => TypesafeClient.Answer.t()}, meta :: map()} | {:error, term()}

TypesafeClient.evaluate(state, questions, opts)   # delegates to configured impl
```

- `TypesafeClient.HTTP` is the real implementation. Options: `api_key` (default
  `TYPESAFE_API_KEY` env), `model` (default `"jev-latest"`), `receive_timeout`
  (default 2_000 ms), `retry` (Req retry on 429/529, max 2). `meta` carries
  `model`, `usage`, and `latency_ms`.
- `TypesafeClient.Stub` returns canned answers from a function or map in `opts`,
  for tests.
- `TypesafeClient.Answer` structs: `%Choice{choice, probabilities, confidence}`,
  `%Noul{noul}`, `%Score{score, legend, probabilities, confidence}`. Question maps are
  passed through as-is (the API shape is the contract); a light validation rejects
  unknown `type` values before the request is sent.
- Implementation is selected by `opts[:client]` or app env
  `config :typesafe_client, client: TypesafeClient.HTTP`.
- Errors: `{:error, {:http, status, body}}`, `{:error, {:transport, reason}}`,
  `{:error, :timeout}`, `{:error, {:invalid_question, id, reason}}`. Never raises for
  network conditions. Never logs the key.

## Component: `JevRouting.Transformer`

Implements `Jido.AI.Reasoning.ReAct.RequestTransformer.transform_request/4`.

Input per turn: `request` (`messages`, `llm_opts`, `tools` as name => module map,
`model`), ReAct `state`, `config`, runtime `ctx` (`request_id`, `run_id`).

State sent to Jev (one request, three questions):

```
%{request: <latest user message text>,
  progress: <up to 3 most recent tool results as "name -> result" strings, or nil>,
  available_actions: [%{name, description}]}
```

- `needs_tool` (Noul): does the request, given progress so far, need one of the
  listed actions for the next step?
- `tool` (Choice): which action next, criteria = name => description plus `none`.
- `depth` (Score, 3 levels): direct / a few dependent steps / long chain or trap.

Overrides:

- `tools`: if `needs_tool < needs_tool_threshold` (default 0.5) then `[]` (the model
  answers directly); else the top `k` (default 3) tool modules by Choice probability,
  always including the argmax, `none` excluded.
- `model`: `:fast` if `depth < 0.75`, `:capable` if `< 1.5`, else `:reasoning`.
  Thresholds and aliases configurable via `config :jev_routing, ...`.
- Any client error or timeout: `{:ok, %{}}` (fail open to baseline behaviour) and a
  `Logger.warning`.

Observability: each decision is written to an ETS table `:jev_routing_decisions`
keyed by `request_id`, as a list of `%{turn, latency_ms, needs_tool, tool_probs,
depth, chosen_tools, chosen_model}`. The bench reads and clears it per run.

## Component: tools

Built-ins (pure): `Jido.Tools.Arithmetic.{Add, Subtract, Multiply, Divide, Square}`,
`Jido.Tools.Basic.{Increment, Decrement, Noop, Today}`.

Synthetic (pure, keyword schemas, `run/2` returns `{:ok, %{result: _}}`):
`ConvertUnit`, `DateAdd`, `DateDiff`, `CountWords`, `CountChars`, `Base64Encode`,
`Base64Decode`, `ReverseString`, `Gcd`, `SortNumbers`, `JsonGet`, `Sha256`,
`CapitalOf` (static table, ~20 countries), `ExchangeRate` (static table, 6 pairs),
`RegexMatch`. Confusable pairs are deliberate: add/increment, subtract/decrement,
date_add/date_diff, count_words/count_chars, encode/decode.

## Component: tasks and grader

About 30 tasks: `%{q, expect: [alternatives], kind: :one_tool | :two_tool | :no_tool}`.
Roughly 14 single-tool, 8 two-tool chains (e.g. "convert 5 miles to km, then square
it"), 8 no-tool (knowledge or rewriting). Grader = normalize (strip markdown, quotes,
`$`, commas, collapse whitespace, lowercase) then equality or word-bounded containment
of any alternative. Shared with tests.

## Component: bench

Conditions, each a fresh `Jido.AgentServer` per task:

| condition | model | tools | transformer |
| --- | --- | --- | --- |
| baseline-fast | `:fast` | all | none |
| baseline-capable | `:capable` | all | none |
| jev | chosen per turn | gated per turn | `JevRouting.Transformer` |

Per run capture: answer text, pass/fail, `iteration` and `usage` from the strategy
snapshot (`Jido.AgentServer.state/1` -> strategy snapshot details), wall time, cost
from usage x price table (`fast` 1/5, `capable` 2/10, `reasoning` 5/25 $ per MTok),
Jev decisions from ETS. Per condition report: pass rate, mean turns, median and p95
wall ms, mean cost per task, mean Jev overhead ms, model mix (jev only). Output:
`bench_results.json` plus a markdown table on stdout. Concurrency 2 to stay under
provider rate limits. `mix run -e 'JevRouting.Bench.main()'` (or `mix bench` alias)
with `--conditions` and `--tasks` filters.

Model aliases: `fast: "anthropic:claude-haiku-4-5"`, `capable: "anthropic:claude-sonnet-5"`,
`reasoning: "anthropic:claude-opus-5"`. If ReqLLM's bundled model registry rejects an
ID, use the nearest registered ID and record which in the results.

## Docker

`run.sh` runs `sudo -n docker run --rm -v <repo>:/work -w /work/experiments/jev_routing/elixir
-v jev_routing_deps:/work/experiments/jev_routing/elixir/deps -v jev_routing_build:/work/.../_build
-e TYPESAFE_API_KEY -e ANTHROPIC_API_KEY hexpm/elixir:1.20.4-... sh -c "mix local.hex --force
&& mix local.rebar --force && mix deps.get && <command>"`. Keys are read from
`~/.typesafe_key` and `~/.anthropic_key` into the environment by the script; nothing is
echoed. `run.sh test` runs `mix test`; `run.sh bench` runs the bench. The container has
no `git`; all deps are Hex packages.

## Testing

- `typesafe_client`: HTTP impl against a Req test adapter (`Req.Test`) for success,
  429 retry, timeout, and malformed answer; Stub returns canned answers; Answer parsing.
- `jev_routing`: transformer with Stub: top-k gating, none gate -> `[]`, depth thresholds
  -> model alias, error -> `%{}`, ETS record written. Grader cases. Each synthetic tool's
  `run/2` on two inputs.
- Bench is not run in `mix test`.

## Success criteria

The bench prints a table where the `jev` condition can be compared with both baselines
on pass rate, turns, wall time, cost, and shows Jev overhead per turn. Whatever the
numbers say, they go into the README and the doc unedited.
