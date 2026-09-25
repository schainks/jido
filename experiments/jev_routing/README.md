# Experiment: Jev (TypeSafe System One) as a tool router for Jido actions

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

## Results

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

## Files

| File | What |
| --- | --- |
| `jev_eval.py` | Extracts the 19 actions from `lib/jido/actions/*.ex`, sends each query with four questions (Noul, Choice, complexity Score, risk Score), reports accuracy by confidence band, thresholds, latency, tokens. Python 3 stdlib only. |
| `jev_eval_results.json` | Raw per-query answers, probabilities, latencies for both criteria variants and the 8-way parallel run. |
| `llm_baseline.py` | Same 19 actions as Anthropic tool definitions, same 34 queries, native tool-calling. `COND=select` switches to the selection-only prompt. Needs the `anthropic` package. |
| `llm_baseline_default.json`, `llm_baseline_select.json` | Raw per-query results for Haiku 4.5 and Opus 5 under each prompt. |

## Running

```sh
# Jev: key in TYPESAFE_API_KEY or ~/.typesafe_key
python3 experiments/jev_routing/jev_eval.py

# LLM baseline: key in ANTHROPIC_API_KEY or ~/.anthropic_key
cd experiments/jev_routing
uv venv && uv pip install anthropic
.venv/bin/python llm_baseline.py                 # assistant prompt
COND=select .venv/bin/python llm_baseline.py     # selection-only prompt
.venv/bin/python llm_baseline.py claude-sonnet-5 # other models as args
```

About 100 Jev requests and 70 Anthropic requests per full run. Neither script
prints or stores a key.

## Sources

- TypeSafe API: https://docs.typesafe.ai/api
- Function calling cookbook: https://docs.typesafe.ai/cookbooks/function_calling
- Confidence-gated routing: https://docs.typesafe.ai/patterns/confidence-routing
- jido_ai model aliases and ReAct request transformer: https://github.com/agentjido/jido_ai
