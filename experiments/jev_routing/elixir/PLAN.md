# Jev Routing Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a real `Jido.AI.Agent` ReAct loop with and without a Jev-backed `request_transformer` and produce a task-level comparison table (pass rate, turns, wall time, cost, Jev overhead).

**Architecture:** Two Mix projects under `experiments/jev_routing/elixir/`: `typesafe_client` (standalone, publishable: HTTP client for TypeSafe System One with typed answers, a stub, and a behaviour) and `jev_routing` (depends on `jido_ai` from Hex and on `typesafe_client` by path: a transformer implementing jido_ai's public `RequestTransformer` behaviour, pure tools, graded tasks, and a bench that starts a fresh agent per task per condition). Everything runs in the `hexpm/elixir:1.20.4` Docker image via `run.sh`.

**Tech Stack:** Elixir 1.20 / OTP 28 (Docker), `jido_ai ~> 2.3`, `req`, `jason`, ExUnit, `Req.Test` + `plug` for HTTP tests.

**Spec:** `experiments/jev_routing/elixir/SPEC.md`

## Global Constraints

- Elixir `~> 1.18`, OTP 27+ (repo AGENTS.md); the container is Elixir 1.20.4 / OTP 28.
- No changes to `jido` or `jido_ai`; only their public behaviours and functions.
- `typesafe_client` depends only on `req` and `jason` (plus `plug` in test). It must not reference `Jido*` or `JevRouting*`.
- Keys come from env vars `TYPESAFE_API_KEY` and `ANTHROPIC_API_KEY` (set by `run.sh` from `~/.typesafe_key` and `~/.anthropic_key`). Nothing prints, logs, or commits a key.
- Conventional Commits; every commit ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- All commands run through `run.sh` (Docker); there is no Elixir on the host. The container has no `git`; all deps are Hex packages.
- Model aliases: `fast: "anthropic:claude-haiku-4-5"`, `capable: "anthropic:claude-sonnet-5"`, `reasoning: "anthropic:claude-opus-5"`; if ReqLLM's registry rejects an ID, use the nearest registered one and record it in `bench_results.json` under `"models"`.

## Review Focus

1. **Jev returns HTTP 200 with a missing or malformed answer** (e.g. `"tool"` absent): the transformer must return `{:ok, %{}}`, not crash the agent. Pinned in Task 7 (`returns no overrides when an answer is missing`).
2. **The Choice names a tool that is not in `request.tools`** (Jev can only pick from criteria, but a stale stub or renamed tool could): gating must drop unknown names and never produce an empty map because of them. Pinned in Task 7 (`ignores unknown tool names`).
3. **A tool result that is not a string** (maps, tuples) in `request.messages`: state building must stringify without raising. Pinned in Task 7 (`builds progress from non-string tool results`).
4. **Task text or model answer with Unicode or Markdown** (`**42**`, `“quoted”`): grader must normalize. Pinned in Task 6 (`strips markdown and quotes`).
5. **Rate limiting (429) and overload (529) from Jev**: client retries twice then returns an error rather than hanging; transformer fails open. Pinned in Task 2 (`retries 429 twice then errors`) and Task 7 (`fails open on client error`).

---

### Task 1: `typesafe_client` project and typed answers

**Files:**
- Create: `experiments/jev_routing/elixir/typesafe_client/mix.exs`
- Create: `experiments/jev_routing/elixir/typesafe_client/lib/typesafe_client/answer.ex`
- Create: `experiments/jev_routing/elixir/typesafe_client/test/test_helper.exs`
- Create: `experiments/jev_routing/elixir/typesafe_client/test/typesafe_client/answer_test.exs`
- Create: `experiments/jev_routing/elixir/run.sh`

**Interfaces:**
- Produces: `TypesafeClient.Answer.parse(map()) :: {:ok, Answer.t()} | {:error, term()}`, structs `TypesafeClient.Answer.Choice{choice, probabilities, confidence}`, `Noul{noul}`, `Score{score, legend, probabilities, confidence}`, and `TypesafeClient.Answer.parse_all(%{String.t() => map()}) :: {:ok, %{String.t() => Answer.t()}} | {:error, term()}`.

- [ ] **Step 1: Create `run.sh`** (Docker entrypoint used by every later step)

```bash
#!/usr/bin/env bash
# Runs mix commands for the Jev routing prototype inside the hexpm/elixir image.
# Usage: ./run.sh <project-dir> <mix args...>     e.g. ./run.sh typesafe_client test
#        ./run.sh . test          (jev_routing project)
#        ./run.sh . bench         (alias for the bench)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PROJ="${1:?project dir}"; shift
IMAGE="hexpm/elixir:1.20.4-erlang-28.5.0.7-debian-bookworm-20260918-slim"
REL="experiments/jev_routing/elixir/${PROJ#./}"
REL="${REL%/.}"
TS_KEY="$(cat "$HOME/.typesafe_key" 2>/dev/null || true)"
AN_KEY="$(cat "$HOME/.anthropic_key" 2>/dev/null || true)"
exec sudo -n docker run --rm \
  -v "$REPO:/work" -w "/work/$REL" \
  -v jev_routing_deps:/work/experiments/jev_routing/elixir/deps \
  -v jev_routing_build:/work/experiments/jev_routing/elixir/_build \
  -v jev_routing_tc_deps:/work/experiments/jev_routing/elixir/typesafe_client/deps \
  -v jev_routing_tc_build:/work/experiments/jev_routing/elixir/typesafe_client/_build \
  -v jev_routing_hex:/root/.mix \
  -e TYPESAFE_API_KEY="$TS_KEY" -e ANTHROPIC_API_KEY="$AN_KEY" -e MIX_ENV="${MIX_ENV:-dev}" \
  "$IMAGE" sh -c "mix local.hex --force >/dev/null && mix local.rebar --force >/dev/null && mix deps.get >/dev/null && mix $*"
```

Run: `chmod +x experiments/jev_routing/elixir/run.sh`

- [ ] **Step 2: Create `typesafe_client/mix.exs`**

```elixir
defmodule TypesafeClient.MixProject do
  use Mix.Project

  def project do
    [
      app: :typesafe_client,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Client for TypeSafe's System One API (Jev): typed Choice, Noul and Score judgments.",
      package: [licenses: ["Apache-2.0"], links: %{}]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:req, "~> 0.5"},
      {:jason, "~> 1.4"},
      {:plug, "~> 1.16", only: :test}
    ]
  end
end
```

- [ ] **Step 3: Create `test/test_helper.exs`**

```elixir
ExUnit.start()
```

- [ ] **Step 4: Write the failing answer tests**

`test/typesafe_client/answer_test.exs`:

```elixir
defmodule TypesafeClient.AnswerTest do
  use ExUnit.Case, async: true
  alias TypesafeClient.Answer
  alias TypesafeClient.Answer.{Choice, Noul, Score}

  test "parses a choice answer" do
    assert {:ok, %Choice{choice: "add", probabilities: %{"add" => 0.7, "none" => 0.3}, confidence: 0.7}} =
             Answer.parse(%{"type" => "choice", "choice" => "add",
                            "probabilities" => %{"add" => 0.7, "none" => 0.3}, "confidence" => 0.7})
  end

  test "parses a noul answer" do
    assert {:ok, %Noul{noul: 0.93}} = Answer.parse(%{"type" => "noul", "noul" => 0.93})
  end

  test "parses a score answer" do
    assert {:ok, %Score{score: 1.43, confidence: 0.35, legend: %{"0" => "a"}, probabilities: %{"0" => 0.0, "1" => 0.57}}} =
             Answer.parse(%{"type" => "score", "score" => 1.43, "confidence" => 0.35,
                            "legend" => %{"0" => "a"}, "probabilities" => %{"0" => 0.0, "1" => 0.57}})
  end

  test "rejects unknown types and missing fields" do
    assert {:error, {:unknown_answer_type, "bogus"}} = Answer.parse(%{"type" => "bogus"})
    assert {:error, {:malformed_answer, "choice"}} = Answer.parse(%{"type" => "choice"})
  end

  test "parse_all keeps ids and fails on the first bad answer" do
    good = %{"a" => %{"type" => "noul", "noul" => 0.1}, "b" => %{"type" => "noul", "noul" => 0.9}}
    assert {:ok, %{"a" => %Noul{noul: 0.1}, "b" => %Noul{noul: 0.9}}} = Answer.parse_all(good)
    assert {:error, {"b", {:unknown_answer_type, "x"}}} = Answer.parse_all(Map.put(good, "b", %{"type" => "x"}))
  end
end
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `cd experiments/jev_routing/elixir && ./run.sh typesafe_client test`
Expected: compile error, `TypesafeClient.Answer` undefined.

- [ ] **Step 6: Implement `lib/typesafe_client/answer.ex`**

```elixir
defmodule TypesafeClient.Answer do
  @moduledoc "Typed answers returned by TypeSafe System One questions."

  defmodule Choice do
    @moduledoc "One option from a defined set, with the full probability distribution."
    @enforce_keys [:choice, :probabilities, :confidence]
    defstruct [:choice, :probabilities, :confidence]
    @type t :: %__MODULE__{choice: String.t(), probabilities: %{String.t() => float()}, confidence: float()}
  end

  defmodule Noul do
    @moduledoc "Probability that a condition holds."
    @enforce_keys [:noul]
    defstruct [:noul]
    @type t :: %__MODULE__{noul: float()}
  end

  defmodule Score do
    @moduledoc "Probability-weighted position on ordered levels."
    @enforce_keys [:score, :probabilities, :confidence]
    defstruct [:score, :legend, :probabilities, :confidence]
    @type t :: %__MODULE__{score: float(), legend: map() | nil, probabilities: %{String.t() => float()}, confidence: float()}
  end

  @type t :: Choice.t() | Noul.t() | Score.t()

  @spec parse(map()) :: {:ok, t()} | {:error, term()}
  def parse(%{"type" => "choice", "choice" => c, "probabilities" => p, "confidence" => conf})
      when is_binary(c) and is_map(p) and is_number(conf),
      do: {:ok, %Choice{choice: c, probabilities: p, confidence: conf / 1}}

  def parse(%{"type" => "noul", "noul" => n}) when is_number(n), do: {:ok, %Noul{noul: n / 1}}

  def parse(%{"type" => "score", "score" => s, "probabilities" => p, "confidence" => conf} = a)
      when is_number(s) and is_map(p) and is_number(conf),
      do: {:ok, %Score{score: s / 1, legend: Map.get(a, "legend"), probabilities: p, confidence: conf / 1}}

  def parse(%{"type" => type}) when type in ["choice", "noul", "score"], do: {:error, {:malformed_answer, type}}
  def parse(%{"type" => type}), do: {:error, {:unknown_answer_type, type}}
  def parse(_), do: {:error, :malformed_answer}

  @spec parse_all(%{String.t() => map()}) :: {:ok, %{String.t() => t()}} | {:error, {String.t(), term()}}
  def parse_all(answers) when is_map(answers) do
    Enum.reduce_while(answers, {:ok, %{}}, fn {id, raw}, {:ok, acc} ->
      case parse(raw) do
        {:ok, a} -> {:cont, {:ok, Map.put(acc, id, a)}}
        {:error, reason} -> {:halt, {:error, {id, reason}}}
      end
    end)
  end
end
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `./run.sh typesafe_client test`
Expected: `5 tests, 0 failures`.

- [ ] **Step 8: Commit**

```bash
git add experiments/jev_routing/elixir/run.sh experiments/jev_routing/elixir/typesafe_client
git commit -m "feat(experiments): typesafe_client package with typed answers and Docker runner

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `TypesafeClient.HTTP` with Req and retry

**Files:**
- Create: `typesafe_client/lib/typesafe_client/http.ex`
- Create: `typesafe_client/lib/typesafe_client.ex` (behaviour only in this task; dispatcher in Task 3)
- Test: `typesafe_client/test/typesafe_client/http_test.exs`

**Interfaces:**
- Consumes: `TypesafeClient.Answer.parse_all/1` (Task 1).
- Produces: behaviour `TypesafeClient` with `@callback evaluate(state :: term(), questions :: map(), opts :: keyword()) :: {:ok, %{String.t() => Answer.t()}, meta :: map()} | {:error, term()}`; `TypesafeClient.HTTP.evaluate/3`; `TypesafeClient.validate_questions(map()) :: :ok | {:error, {:invalid_question, id, reason}}`. Options for HTTP: `:api_key`, `:model` (default `"jev-latest"`), `:url`, `:receive_timeout` (default 2_000), `:max_retries` (default 2), `:plug` (tests). `meta` = `%{model: String.t() | nil, usage: map() | nil, latency_ms: non_neg_integer()}`.

- [ ] **Step 1: Write the failing HTTP tests**

`test/typesafe_client/http_test.exs`:

```elixir
defmodule TypesafeClient.HTTPTest do
  use ExUnit.Case, async: true
  alias TypesafeClient.Answer.{Choice, Noul}

  @questions %{"q" => %{"type" => "noul", "instructions" => "Is it urgent?"}}
  @opts [api_key: "test-key", plug: {Req.Test, __MODULE__}, max_retries: 2, retry_delay_ms: 0]

  test "posts state, model and questions with a bearer key and parses answers" do
    Req.Test.stub(__MODULE__, fn conn ->
      assert ["Bearer test-key"] = Plug.Conn.get_req_header(conn, "authorization")
      {:ok, body, conn} = Plug.Conn.read_body(conn)
      assert %{"state" => %{"x" => 1}, "model" => "jev-latest", "questions" => %{"q" => _}} = Jason.decode!(body)
      Req.Test.json(conn, %{"model" => "jev-1.13.0", "usage" => %{"input_tokens" => 10, "output_tokens" => 2},
                            "answers" => %{"q" => %{"type" => "noul", "noul" => 0.8}}})
    end)

    assert {:ok, %{"q" => %Noul{noul: 0.8}}, meta} = TypesafeClient.HTTP.evaluate(%{x: 1}, @questions, @opts)
    assert meta.model == "jev-1.13.0" and meta.usage["input_tokens"] == 10 and is_integer(meta.latency_ms)
  end

  test "retries 429 twice then errors" do
    {:ok, counter} = Agent.start_link(fn -> 0 end)
    Req.Test.stub(__MODULE__, fn conn ->
      Agent.update(counter, &(&1 + 1))
      Plug.Conn.send_resp(conn, 429, "slow down")
    end)
    assert {:error, {:http, 429, _}} = TypesafeClient.HTTP.evaluate("s", @questions, @opts)
    assert Agent.get(counter, & &1) == 3
  end

  test "returns http error with body on 4xx without retry" do
    Req.Test.stub(__MODULE__, fn conn -> Plug.Conn.send_resp(conn, 422, ~s({"error":"bad"})) end)
    assert {:error, {:http, 422, _}} = TypesafeClient.HTTP.evaluate("s", @questions, @opts)
  end

  test "returns transport error on connection failure" do
    Req.Test.stub(__MODULE__, fn conn -> Req.Test.transport_error(conn, :timeout) end)
    assert {:error, {:transport, :timeout}} = TypesafeClient.HTTP.evaluate("s", @questions, Keyword.put(@opts, :max_retries, 0))
  end

  test "rejects a question with an unknown type before sending" do
    bad = %{"q" => %{"type" => "guess", "instructions" => "?"}}
    assert {:error, {:invalid_question, "q", :unknown_type}} = TypesafeClient.HTTP.evaluate("s", bad, @opts)
  end

  test "returns an error when the key is missing" do
    assert {:error, :missing_api_key} = TypesafeClient.HTTP.evaluate("s", @questions, plug: {Req.Test, __MODULE__}, api_key: nil)
  end

  test "surfaces malformed answers" do
    Req.Test.stub(__MODULE__, fn conn -> Req.Test.json(conn, %{"answers" => %{"q" => %{"type" => "choice"}}}) end)
    assert {:error, {:malformed_answers, {"q", {:malformed_answer, "choice"}}}} = TypesafeClient.HTTP.evaluate("s", @questions, @opts)
    _ = Choice
  end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `./run.sh typesafe_client test test/typesafe_client/http_test.exs`
Expected: compile error, `TypesafeClient.HTTP` undefined.

- [ ] **Step 3: Implement the behaviour module `lib/typesafe_client.ex`** (dispatcher added in Task 3)

```elixir
defmodule TypesafeClient do
  @moduledoc """
  Client for TypeSafe's System One API (Jev).

  Ask typed questions (`noul`, `choice`, `score`) about arbitrary JSON state and get
  probabilities back. `TypesafeClient.HTTP` talks to the API; `TypesafeClient.Stub`
  returns canned answers for tests. Question maps follow the API shape verbatim; this
  library does not invent its own schema for them.
  """

  alias TypesafeClient.Answer

  @type answers :: %{String.t() => Answer.t()}
  @type meta :: %{model: String.t() | nil, usage: map() | nil, latency_ms: non_neg_integer()}

  @callback evaluate(state :: term(), questions :: map(), opts :: keyword()) ::
              {:ok, answers(), meta()} | {:error, term()}

  @valid_types ~w(noul choice score)

  @doc "Checks every question has a known `type`. Ids may be atoms or strings."
  @spec validate_questions(map()) :: :ok | {:error, {:invalid_question, String.t(), :unknown_type | :not_a_map}}
  def validate_questions(questions) when is_map(questions) do
    Enum.reduce_while(questions, :ok, fn {id, q}, :ok ->
      type = if is_map(q), do: to_string(Map.get(q, "type") || Map.get(q, :type) || ""), else: nil
      cond do
        not is_map(q) -> {:halt, {:error, {:invalid_question, to_string(id), :not_a_map}}}
        type in @valid_types -> {:cont, :ok}
        true -> {:halt, {:error, {:invalid_question, to_string(id), :unknown_type}}}
      end
    end)
  end
end
```

- [ ] **Step 4: Implement `lib/typesafe_client/http.ex`**

```elixir
defmodule TypesafeClient.HTTP do
  @moduledoc "Real `TypesafeClient` implementation over Req. Never logs the key."
  @behaviour TypesafeClient

  alias TypesafeClient.Answer

  @url "https://api.typesafe.ai/v1/systemone"
  @default_model "jev-latest"

  @impl true
  def evaluate(state, questions, opts \\ []) do
    with :ok <- TypesafeClient.validate_questions(questions),
         {:ok, key} <- api_key(opts) do
      body = %{state: state, model: Keyword.get(opts, :model, @default_model), questions: questions}
      max_retries = Keyword.get(opts, :max_retries, 2)
      delay = Keyword.get(opts, :retry_delay_ms, 250)

      req =
        Req.new(
          url: Keyword.get(opts, :url, @url),
          auth: {:bearer, key},
          json: body,
          receive_timeout: Keyword.get(opts, :receive_timeout, 2_000),
          retry: &retry?/2,
          max_retries: max_retries,
          retry_delay: fn n -> delay * Integer.pow(2, n) end,
          retry_log_level: false
        )
        |> maybe_plug(Keyword.get(opts, :plug))

      t0 = System.monotonic_time(:millisecond)

      case Req.post(req) do
        {:ok, %Req.Response{status: 200, body: %{"answers" => raw} = resp}} ->
          case Answer.parse_all(raw) do
            {:ok, answers} ->
              {:ok, answers,
               %{model: resp["model"], usage: resp["usage"], latency_ms: System.monotonic_time(:millisecond) - t0}}
            {:error, reason} -> {:error, {:malformed_answers, reason}}
          end

        {:ok, %Req.Response{status: 200}} -> {:error, {:malformed_answers, :no_answers}}
        {:ok, %Req.Response{status: status, body: body}} -> {:error, {:http, status, body}}
        {:error, %Req.TransportError{reason: reason}} -> {:error, {:transport, reason}}
        {:error, other} -> {:error, {:transport, other}}
      end
    end
  end

  defp retry?(_req, %Req.Response{status: s}) when s in [429, 529], do: true
  defp retry?(_req, %Req.TransportError{}), do: true
  defp retry?(_req, _), do: false

  defp maybe_plug(req, nil), do: req
  defp maybe_plug(req, plug), do: Req.merge(req, plug: plug)

  defp api_key(opts) do
    case Keyword.get(opts, :api_key, System.get_env("TYPESAFE_API_KEY")) do
      k when is_binary(k) and k != "" -> {:ok, k}
      _ -> {:error, :missing_api_key}
    end
  end
end
```

Note: `Req.Test.transport_error/2` is available in `req >= 0.5`; the `retry` function form receives `(request, response_or_exception)`. If `mix deps.get` resolves a Req where `retry_log_level: false` is rejected, remove that option.

- [ ] **Step 5: Run tests**

Run: `./run.sh typesafe_client test`
Expected: `12 tests, 0 failures` (5 from Task 1 + 7).

- [ ] **Step 6: Commit**

```bash
git add experiments/jev_routing/elixir/typesafe_client
git commit -m "feat(experiments): typesafe_client HTTP implementation with retry and question validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `TypesafeClient.Stub` and the `TypesafeClient.evaluate/3` dispatcher

**Files:**
- Create: `typesafe_client/lib/typesafe_client/stub.ex`
- Modify: `typesafe_client/lib/typesafe_client.ex` (add `evaluate/3`)
- Test: `typesafe_client/test/typesafe_client/stub_test.exs`

**Interfaces:**
- Produces: `TypesafeClient.evaluate(state, questions, opts)` which calls `opts[:client] || Application.get_env(:typesafe_client, :client, TypesafeClient.HTTP)`. `TypesafeClient.Stub.evaluate/3` reads `opts[:answers]`: either a map of id => `Answer.t()` (or raw API map) or a 2-arity function `(state, questions) -> map | {:error, term}`.

- [ ] **Step 1: Write the failing stub tests**

```elixir
defmodule TypesafeClient.StubTest do
  use ExUnit.Case, async: true
  alias TypesafeClient.Answer.{Noul, Choice}

  @q %{"a" => %{"type" => "noul", "instructions" => "?"}}

  test "returns canned struct answers" do
    assert {:ok, %{"a" => %Noul{noul: 0.2}}, %{latency_ms: 0}} =
             TypesafeClient.Stub.evaluate("s", @q, answers: %{"a" => %Noul{noul: 0.2}})
  end

  test "parses canned raw answers" do
    assert {:ok, %{"a" => %Choice{choice: "x"}}, _} =
             TypesafeClient.Stub.evaluate("s", @q, answers: %{"a" => %{"type" => "choice", "choice" => "x", "probabilities" => %{"x" => 1.0}, "confidence" => 1.0}})
  end

  test "calls a function with state and questions, and passes errors through" do
    fun = fn state, questions -> if state == :boom, do: {:error, :down}, else: %{"a" => %Noul{noul: map_size(questions) / 1}} end
    assert {:ok, %{"a" => %Noul{noul: 1.0}}, _} = TypesafeClient.Stub.evaluate(:ok, @q, answers: fun)
    assert {:error, :down} = TypesafeClient.Stub.evaluate(:boom, @q, answers: fun)
  end

  test "dispatcher uses opts[:client] then app env" do
    assert {:ok, %{"a" => %Noul{noul: 0.5}}, _} = TypesafeClient.evaluate("s", @q, client: TypesafeClient.Stub, answers: %{"a" => %Noul{noul: 0.5}})
    Application.put_env(:typesafe_client, :client, TypesafeClient.Stub)
    on_exit(fn -> Application.delete_env(:typesafe_client, :client) end)
    assert {:ok, _, _} = TypesafeClient.evaluate("s", @q, answers: %{"a" => %Noul{noul: 0.5}})
  end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `./run.sh typesafe_client test test/typesafe_client/stub_test.exs`
Expected: `TypesafeClient.Stub` undefined.

- [ ] **Step 3: Implement the stub and dispatcher**

`lib/typesafe_client/stub.ex`:

```elixir
defmodule TypesafeClient.Stub do
  @moduledoc "Test implementation of `TypesafeClient`: returns canned answers from `opts[:answers]`."
  @behaviour TypesafeClient
  alias TypesafeClient.Answer

  @impl true
  def evaluate(state, questions, opts) do
    case Keyword.fetch!(opts, :answers) do
      fun when is_function(fun, 2) -> fun.(state, questions) |> normalize()
      map when is_map(map) -> normalize(map)
    end
  end

  defp normalize({:error, reason}), do: {:error, reason}

  defp normalize(map) when is_map(map) do
    map
    |> Enum.reduce_while({:ok, %{}}, fn
      {id, %_{} = struct}, {:ok, acc} -> {:cont, {:ok, Map.put(acc, id, struct)}}
      {id, raw}, {:ok, acc} ->
        case Answer.parse(raw) do
          {:ok, a} -> {:cont, {:ok, Map.put(acc, id, a)}}
          {:error, r} -> {:halt, {:error, {:malformed_answers, {id, r}}}}
        end
    end)
    |> case do
      {:ok, answers} -> {:ok, answers, %{model: "stub", usage: nil, latency_ms: 0}}
      err -> err
    end
  end
end
```

Add to `lib/typesafe_client.ex` after `validate_questions/1`:

```elixir
  @doc "Evaluates questions with the configured client (`opts[:client]`, else app env `:client`, else HTTP)."
  @spec evaluate(term(), map(), keyword()) :: {:ok, answers(), meta()} | {:error, term()}
  def evaluate(state, questions, opts \\ []) do
    {client, opts} = Keyword.pop(opts, :client, Application.get_env(:typesafe_client, :client, TypesafeClient.HTTP))
    client.evaluate(state, questions, opts)
  end
```

- [ ] **Step 4: Run tests**

Run: `./run.sh typesafe_client test`
Expected: `16 tests, 0 failures`.

- [ ] **Step 5: Commit**

```bash
git add experiments/jev_routing/elixir/typesafe_client
git commit -m "feat(experiments): typesafe_client stub and client dispatcher

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `jev_routing` project scaffold, config, and model-ID check

**Files:**
- Create: `experiments/jev_routing/elixir/mix.exs`
- Create: `experiments/jev_routing/elixir/config/config.exs`
- Create: `experiments/jev_routing/elixir/config/runtime.exs`
- Create: `experiments/jev_routing/elixir/test/test_helper.exs`
- Create: `experiments/jev_routing/elixir/.gitignore`
- Create: `experiments/jev_routing/elixir/.formatter.exs`

**Interfaces:**
- Produces: app `:jev_routing` with deps `jido_ai ~> 2.3`, `typesafe_client` (path), config keys `config :jev_routing, top_k: 3, needs_tool_threshold: 0.5, depth_thresholds: {0.75, 1.5}, client_opts: []`; `mix bench` alias running `JevRouting.Bench.main/1`.

- [ ] **Step 1: Create `mix.exs`**

```elixir
defmodule JevRouting.MixProject do
  use Mix.Project

  def project do
    [
      app: :jev_routing,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: false,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      aliases: [bench: ["run -e 'JevRouting.Bench.main(System.argv())'"]]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:jido_ai, "~> 2.3"},
      {:typesafe_client, path: "typesafe_client"}
    ]
  end
end
```

- [ ] **Step 2: Create config files**

`config/config.exs`:

```elixir
import Config

config :jev_routing,
  top_k: 3,
  needs_tool_threshold: 0.5,
  depth_thresholds: {0.75, 1.5},
  client_opts: []

config :jido_ai,
  model_aliases: %{
    fast: "anthropic:claude-haiku-4-5",
    capable: "anthropic:claude-sonnet-5",
    reasoning: "anthropic:claude-opus-5"
  }

config :logger, level: :warning

if config_env() == :test do
  config :typesafe_client, client: TypesafeClient.Stub
end
```

`config/runtime.exs`:

```elixir
import Config

if key = System.get_env("ANTHROPIC_API_KEY") do
  config :req_llm, anthropic_api_key: key
end
```

`test/test_helper.exs`:

```elixir
ExUnit.start()
```

`.gitignore`:

```
/_build/
/deps/
/typesafe_client/_build/
/typesafe_client/deps/
erl_crash.dump
bench_results.json
```

`.formatter.exs`:

```elixir
[inputs: ["{mix,.formatter}.exs", "{config,lib,test}/**/*.{ex,exs}", "typesafe_client/{lib,test}/**/*.{ex,exs}"]]
```

- [ ] **Step 3: Fetch deps and compile in Docker; record resolved versions**

Run: `./run.sh . compile 2>&1 | tail -20`
Expected: compiles with no errors (warnings from deps are fine). If `mix deps.get` fails on `jido_ai ~> 2.3`, check `https://hex.pm/api/packages/jido_ai` for the latest 2.x and adjust.

Run: `./run.sh . run -e 'IO.inspect({Application.spec(:jido_ai, :vsn), Application.spec(:req_llm, :vsn), Application.spec(:jido, :vsn)})'`
Expected: three version strings; note them for the README.

- [ ] **Step 4: Check the model IDs against ReqLLM's registry**

Run:

```
./run.sh . run -e '
for alias <- [:fast, :capable, :reasoning] do
  spec = Jido.AI.resolve_model(alias)
  IO.inspect({alias, spec, ReqLLM.Model.from(spec)})
end'
```

Expected: each line ends with `{:ok, %ReqLLM.Model{}}`. If any is `{:error, _}`, list registered Anthropic IDs with `./run.sh . run -e 'IO.inspect(LLMDB.models("anthropic") |> Enum.map(& &1.id))'` (if `LLMDB.models/1` is not exported, try `ReqLLM.Provider.Registry.list_models(:anthropic)`), pick the nearest current-generation IDs, update `config.exs`, and record the substitution in the README.

- [ ] **Step 5: Commit**

```bash
git add experiments/jev_routing/elixir/mix.exs experiments/jev_routing/elixir/config experiments/jev_routing/elixir/test/test_helper.exs experiments/jev_routing/elixir/.gitignore experiments/jev_routing/elixir/.formatter.exs experiments/jev_routing/elixir/mix.lock
git commit -m "chore(experiments): scaffold jev_routing project on jido_ai and typesafe_client

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Synthetic pure tools

**Files:**
- Create: `lib/jev_routing/tools.ex` (all 15 modules in one file, each ~10 lines, plus `JevRouting.Tools.all/0`)
- Test: `test/jev_routing/tools_test.exs`

**Interfaces:**
- Produces: `JevRouting.Tools.all() :: [module()]` = the 9 built-ins + 15 synthetic modules, and each synthetic module `use Jido.Action` with `run/2 -> {:ok, %{result: term()}}`. Names: `convert_unit`, `date_add`, `date_diff`, `count_words`, `count_chars`, `base64_encode`, `base64_decode`, `reverse_string`, `gcd`, `sort_numbers`, `json_get`, `sha256`, `capital_of`, `exchange_rate`, `regex_match`.

- [ ] **Step 1: Write the failing tool tests**

```elixir
defmodule JevRouting.ToolsTest do
  use ExUnit.Case, async: true
  alias JevRouting.Tools, as: T

  test "all/0 lists 24 unique action modules with unique names" do
    mods = T.all()
    assert length(mods) == 24
    names = Enum.map(mods, & &1.name())
    assert length(Enum.uniq(names)) == 24
  end

  test "convert_unit" do
    assert {:ok, %{result: 8.05}} = T.ConvertUnit.run(%{value: 5, from: "mi", to: "km"}, %{})
    assert {:ok, %{result: 37.0}} = T.ConvertUnit.run(%{value: 98.6, from: "f", to: "c"}, %{})
    assert {:error, _} = T.ConvertUnit.run(%{value: 1, from: "mi", to: "kg"}, %{})
  end

  test "date_add and date_diff" do
    assert {:ok, %{result: "2024-02-29"}} = T.DateAdd.run(%{date: "2024-02-28", days: 1}, %{})
    assert {:ok, %{result: 366}} = T.DateDiff.run(%{from: "2024-01-01", to: "2025-01-01"}, %{})
  end

  test "count_words and count_chars" do
    assert {:ok, %{result: 5}} = T.CountWords.run(%{text: "the quick brown fox jumps"}, %{})
    assert {:ok, %{result: 3}} = T.CountChars.run(%{text: "strawberry", char: "r"}, %{})
  end

  test "base64 round trip and reverse" do
    assert {:ok, %{result: "aGVsbG8gamlkbw=="}} = T.Base64Encode.run(%{text: "hello jido"}, %{})
    assert {:ok, %{result: "hello jido"}} = T.Base64Decode.run(%{text: "aGVsbG8gamlkbw=="}, %{})
    assert {:error, _} = T.Base64Decode.run(%{text: "not base64!"}, %{})
    assert {:ok, %{result: "odij"}} = T.ReverseString.run(%{text: "jido"}, %{})
  end

  test "gcd, sort_numbers, json_get, sha256" do
    assert {:ok, %{result: 21}} = T.Gcd.run(%{a: 1071, b: 462}, %{})
    assert {:ok, %{result: [3, 7, 19, 42]}} = T.SortNumbers.run(%{numbers: [42, 7, 19, 3]}, %{})
    assert {:ok, %{result: "x"}} = T.JsonGet.run(%{json: ~s({"a":{"b":[1,2,{"c":"x"}]}}), path: "a.b.2.c"}, %{})
    assert {:ok, %{result: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}} = T.Sha256.run(%{text: "hello"}, %{})
  end

  test "lookups and regex" do
    assert {:ok, %{result: "Canberra"}} = T.CapitalOf.run(%{country: "Australia"}, %{})
    assert {:error, _} = T.CapitalOf.run(%{country: "Atlantis"}, %{})
    assert {:ok, %{result: 0.92}} = T.ExchangeRate.run(%{from: "USD", to: "EUR"}, %{})
    assert {:ok, %{result: ["42", "7"]}} = T.RegexMatch.run(%{text: "a42b7", pattern: "\\d+"}, %{})
  end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `./run.sh . test test/jev_routing/tools_test.exs`
Expected: `JevRouting.Tools` undefined.

- [ ] **Step 3: Implement `lib/jev_routing/tools.ex`**

```elixir
defmodule JevRouting.Tools do
  @moduledoc """
  Pure tools for the bench: 9 built-ins from jido_action plus 15 synthetic actions.
  Confusable pairs are deliberate: add/increment_action, subtract/decrement_action,
  date_add/date_diff, count_words/count_chars, base64_encode/base64_decode.
  """

  @builtins [
    Jido.Tools.Arithmetic.Add, Jido.Tools.Arithmetic.Subtract, Jido.Tools.Arithmetic.Multiply,
    Jido.Tools.Arithmetic.Divide, Jido.Tools.Arithmetic.Square,
    Jido.Tools.Basic.Increment, Jido.Tools.Basic.Decrement, Jido.Tools.Basic.Noop, Jido.Tools.Basic.Today
  ]

  def builtins, do: @builtins

  def synthetic do
    [__MODULE__.ConvertUnit, __MODULE__.DateAdd, __MODULE__.DateDiff, __MODULE__.CountWords, __MODULE__.CountChars,
     __MODULE__.Base64Encode, __MODULE__.Base64Decode, __MODULE__.ReverseString, __MODULE__.Gcd, __MODULE__.SortNumbers,
     __MODULE__.JsonGet, __MODULE__.Sha256, __MODULE__.CapitalOf, __MODULE__.ExchangeRate, __MODULE__.RegexMatch]
  end

  def all, do: @builtins ++ synthetic()

  defmodule ConvertUnit do
    @moduledoc false
    use Jido.Action,
      name: "convert_unit",
      description: "Converts a quantity between units: mi/km, kg/lb, c/f, m/ft",
      schema: [
        value: [type: {:or, [:integer, :float]}, required: true, doc: "Quantity to convert"],
        from: [type: :string, required: true, doc: "Source unit: mi, km, kg, lb, c, f, m, ft"],
        to: [type: :string, required: true, doc: "Target unit"]
      ]

    @factors %{{"mi", "km"} => 1.609344, {"km", "mi"} => 0.621371, {"kg", "lb"} => 2.20462, {"lb", "kg"} => 0.453592,
               {"m", "ft"} => 3.28084, {"ft", "m"} => 0.3048}

    def run(%{value: v, from: from, to: to}, _ctx) do
      key = {String.downcase(from), String.downcase(to)}
      case key do
        {"c", "f"} -> {:ok, %{result: Float.round(v * 9 / 5 + 32, 2)}}
        {"f", "c"} -> {:ok, %{result: Float.round((v - 32) * 5 / 9, 2)}}
        _ ->
          case Map.fetch(@factors, key) do
            {:ok, f} -> {:ok, %{result: Float.round(v * f, 2)}}
            :error -> {:error, "unsupported conversion #{from} -> #{to}"}
          end
      end
    end
  end

  defmodule DateAdd do
    @moduledoc false
    use Jido.Action,
      name: "date_add",
      description: "Adds a number of days to an ISO 8601 date and returns the new date",
      schema: [date: [type: :string, required: true, doc: "ISO date, e.g. 2024-02-28"],
               days: [type: :integer, required: true, doc: "Days to add (negative to subtract)"]]

    def run(%{date: d, days: n}, _ctx) do
      with {:ok, date} <- Date.from_iso8601(d), do: {:ok, %{result: Date.to_iso8601(Date.add(date, n))}}
    end
  end

  defmodule DateDiff do
    @moduledoc false
    use Jido.Action,
      name: "date_diff",
      description: "Number of days between two ISO 8601 dates (to minus from)",
      schema: [from: [type: :string, required: true, doc: "Start ISO date"], to: [type: :string, required: true, doc: "End ISO date"]]

    def run(%{from: f, to: t}, _ctx) do
      with {:ok, a} <- Date.from_iso8601(f), {:ok, b} <- Date.from_iso8601(t), do: {:ok, %{result: Date.diff(b, a)}}
    end
  end

  defmodule CountWords do
    @moduledoc false
    use Jido.Action, name: "count_words", description: "Counts whitespace-separated words in a text",
      schema: [text: [type: :string, required: true, doc: "Text to count"]]
    def run(%{text: t}, _ctx), do: {:ok, %{result: t |> String.split() |> length()}}
  end

  defmodule CountChars do
    @moduledoc false
    use Jido.Action, name: "count_chars", description: "Counts how many times a single character occurs in a text",
      schema: [text: [type: :string, required: true, doc: "Text to search"], char: [type: :string, required: true, doc: "Single character to count"]]
    def run(%{text: t, char: c}, _ctx), do: {:ok, %{result: t |> String.graphemes() |> Enum.count(&(&1 == c))}}
  end

  defmodule Base64Encode do
    @moduledoc false
    use Jido.Action, name: "base64_encode", description: "Encodes text as standard Base64",
      schema: [text: [type: :string, required: true, doc: "Plain text"]]
    def run(%{text: t}, _ctx), do: {:ok, %{result: Base.encode64(t)}}
  end

  defmodule Base64Decode do
    @moduledoc false
    use Jido.Action, name: "base64_decode", description: "Decodes standard Base64 text back to plain text",
      schema: [text: [type: :string, required: true, doc: "Base64 text"]]
    def run(%{text: t}, _ctx) do
      case Base.decode64(t) do
        {:ok, s} -> {:ok, %{result: s}}
        :error -> {:error, "invalid base64"}
      end
    end
  end

  defmodule ReverseString do
    @moduledoc false
    use Jido.Action, name: "reverse_string", description: "Reverses the characters of a string",
      schema: [text: [type: :string, required: true, doc: "Text to reverse"]]
    def run(%{text: t}, _ctx), do: {:ok, %{result: String.reverse(t)}}
  end

  defmodule Gcd do
    @moduledoc false
    use Jido.Action, name: "gcd", description: "Greatest common divisor of two integers",
      schema: [a: [type: :integer, required: true, doc: "First integer"], b: [type: :integer, required: true, doc: "Second integer"]]
    def run(%{a: a, b: b}, _ctx), do: {:ok, %{result: Integer.gcd(a, b)}}
  end

  defmodule SortNumbers do
    @moduledoc false
    use Jido.Action, name: "sort_numbers", description: "Sorts a list of numbers in ascending order",
      schema: [numbers: [type: {:list, {:or, [:integer, :float]}}, required: true, doc: "Numbers to sort"]]
    def run(%{numbers: ns}, _ctx), do: {:ok, %{result: Enum.sort(ns)}}
  end

  defmodule JsonGet do
    @moduledoc false
    use Jido.Action, name: "json_get", description: "Reads a value from a JSON document by dotted path; list indexes are numbers, e.g. a.b.2.c",
      schema: [json: [type: :string, required: true, doc: "JSON text"], path: [type: :string, required: true, doc: "Dotted path"]]
    def run(%{json: j, path: p}, _ctx) do
      with {:ok, doc} <- Jason.decode(j) do
        value =
          p |> String.split(".") |> Enum.reduce(doc, fn
            _k, nil -> nil
            k, list when is_list(list) -> case Integer.parse(k) do {i, ""} -> Enum.at(list, i); _ -> nil end
            k, map when is_map(map) -> Map.get(map, k)
            _k, _ -> nil
          end)
        {:ok, %{result: value}}
      end
    end
  end

  defmodule Sha256 do
    @moduledoc false
    use Jido.Action, name: "sha256", description: "Hex-encoded SHA-256 digest of a text",
      schema: [text: [type: :string, required: true, doc: "Text to hash"]]
    def run(%{text: t}, _ctx), do: {:ok, %{result: :crypto.hash(:sha256, t) |> Base.encode16(case: :lower)}}
  end

  defmodule CapitalOf do
    @moduledoc false
    use Jido.Action, name: "capital_of", description: "Capital city of a country from a small static table",
      schema: [country: [type: :string, required: true, doc: "Country name in English"]]
    @table %{"australia" => "Canberra", "canada" => "Ottawa", "brazil" => "Brasília", "japan" => "Tokyo", "germany" => "Berlin",
             "france" => "Paris", "india" => "New Delhi", "kenya" => "Nairobi", "mexico" => "Mexico City", "norway" => "Oslo",
             "turkey" => "Ankara", "switzerland" => "Bern", "nigeria" => "Abuja", "argentina" => "Buenos Aires",
             "egypt" => "Cairo", "poland" => "Warsaw", "vietnam" => "Hanoi", "new zealand" => "Wellington",
             "south africa" => "Pretoria", "chile" => "Santiago"}
    def run(%{country: c}, _ctx) do
      case Map.fetch(@table, String.downcase(String.trim(c))) do
        {:ok, cap} -> {:ok, %{result: cap}}
        :error -> {:error, "unknown country #{c}"}
      end
    end
  end

  defmodule ExchangeRate do
    @moduledoc false
    use Jido.Action, name: "exchange_rate", description: "Static exchange rate between two currency codes (USD, EUR, GBP, JPY)",
      schema: [from: [type: :string, required: true, doc: "Source currency code"], to: [type: :string, required: true, doc: "Target currency code"]]
    @rates %{{"USD", "EUR"} => 0.92, {"EUR", "USD"} => 1.09, {"USD", "GBP"} => 0.79, {"GBP", "USD"} => 1.27, {"USD", "JPY"} => 150.0, {"JPY", "USD"} => 0.0067}
    def run(%{from: f, to: t}, _ctx) do
      case Map.fetch(@rates, {String.upcase(f), String.upcase(t)}) do
        {:ok, r} -> {:ok, %{result: r}}
        :error -> {:error, "no rate for #{f}->#{t}"}
      end
    end
  end

  defmodule RegexMatch do
    @moduledoc false
    use Jido.Action, name: "regex_match", description: "All non-overlapping matches of a regular expression in a text",
      schema: [text: [type: :string, required: true, doc: "Text to scan"], pattern: [type: :string, required: true, doc: "Regular expression"]]
    def run(%{text: t, pattern: p}, _ctx) do
      with {:ok, re} <- Regex.compile(p), do: {:ok, %{result: Regex.scan(re, t) |> Enum.map(&hd/1)}}
    end
  end
end
```

- [ ] **Step 4: Run tests**

Run: `./run.sh . test test/jev_routing/tools_test.exs`
Expected: `7 tests, 0 failures`. If `use Jido.Action` rejects `{:or, [...]}` or `{:list, ...}` NimbleOptions types, replace with `:any` and coerce in `run/2`.

- [ ] **Step 5: Commit**

```bash
git add experiments/jev_routing/elixir/lib/jev_routing/tools.ex experiments/jev_routing/elixir/test/jev_routing/tools_test.exs
git commit -m "feat(experiments): pure synthetic tools with deliberate confusable pairs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Tasks and grader

**Files:**
- Create: `lib/jev_routing/tasks.ex`
- Test: `test/jev_routing/tasks_test.exs`

**Interfaces:**
- Produces: `JevRouting.Tasks.all() :: [%{id: String.t(), q: String.t(), expect: [String.t()], kind: :one_tool | :two_tool | :no_tool}]` (30 tasks), `JevRouting.Tasks.grade(task, reply :: String.t()) :: boolean()`, `JevRouting.Tasks.normalize(String.t()) :: String.t()`.

- [ ] **Step 1: Write the failing tests**

```elixir
defmodule JevRouting.TasksTest do
  use ExUnit.Case, async: true
  alias JevRouting.Tasks

  test "30 tasks, unique ids, every kind present" do
    tasks = Tasks.all()
    assert length(tasks) == 30
    assert tasks |> Enum.map(& &1.id) |> Enum.uniq() |> length() == 30
    assert Enum.frequencies_by(tasks, & &1.kind) == %{one_tool: 14, two_tool: 8, no_tool: 8}
  end

  test "strips markdown and quotes" do
    assert Tasks.normalize(~s(**"8.05"**)) == "8 05"
    assert Tasks.normalize("The answer is: 2,418.") == "the answer is 2 418"
  end

  test "grades by equality or word-bounded containment of any alternative" do
    t = %{expect: ["8.05", "8.0"]}
    assert Tasks.grade(t, "**8.05 km**")
    assert Tasks.grade(t, "8.0")
    refute Tasks.grade(t, "18.05")
    refute Tasks.grade(t, "8.055")
  end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `./run.sh . test test/jev_routing/tasks_test.exs`
Expected: `JevRouting.Tasks` undefined.

- [ ] **Step 3: Implement `lib/jev_routing/tasks.ex`**

```elixir
defmodule JevRouting.Tasks do
  @moduledoc "Graded tasks for the bench and the grader shared with tests."

  @suffix " Reply with only the final answer."

  @tasks [
    # one_tool (14)
    {"t01", "Convert 5 miles to kilometers, two decimals.", ["8.05"], :one_tool},
    {"t02", "What is the date 45 days after 2024-11-20?", ["2025-01-04"], :one_tool},
    {"t03", "How many days are there from 2024-01-01 to 2025-01-01?", ["366"], :one_tool},
    {"t04", "How many words are in: 'the quick brown fox jumps over the lazy dog'?", ["9"], :one_tool},
    {"t05", "How many times does the letter r appear in 'strawberry'?", ["3"], :one_tool},
    {"t06", "Base64-encode the text 'jido agent'.", ["amlkbyBhZ2VudA=="], :one_tool},
    {"t07", "Decode this Base64: c3lzdGVtIG9uZQ==", ["system one"], :one_tool},
    {"t08", "Reverse the string 'directive'.", ["evitcerid"], :one_tool},
    {"t09", "What is the greatest common divisor of 1071 and 462?", ["21"], :one_tool},
    {"t10", "Sort ascending: 42, 7, 19, 3, 88.", ["3 7 19 42 88", "3, 7, 19, 42, 88", "[3, 7, 19, 42, 88]"], :one_tool},
    {"t11", ~s(In the JSON {"a":{"b":[1,2,{"c":"x"}]}} what is the value at a.b[2].c?), ["x"], :one_tool},
    {"t12", "What is the capital of Kenya?", ["Nairobi"], :one_tool},
    {"t13", "Multiply 1234 by 5678.", ["7006652", "7,006,652"], :one_tool},
    {"t14", "Increment the value 41 by one.", ["42"], :one_tool},
    # two_tool (8): each needs two dependent tool calls
    {"t15", "Convert 10 miles to kilometers, then square the result. Two decimals.", ["258.99", "259.0", "259"], :two_tool},
    {"t16", "How many days from 2024-03-01 to 2024-12-25, then multiply that by 3?", ["897"], :two_tool},
    {"t17", "Count the words in 'a b c d e f g' and then add 100.", ["107"], :two_tool},
    {"t18", "Take the gcd of 84 and 36, then convert that many kilometers to miles. Two decimals.", ["7.46"], :two_tool},
    {"t19", "Decode the Base64 'Zm9ydHk=' and then count its characters that are the letter o.", ["1"], :two_tool},
    {"t20", "Convert 100 USD to EUR using the exchange rate tool, then subtract 2.", ["90", "90.0"], :two_tool},
    {"t21", "Reverse the string 'level up' and then count its words.", ["2"], :two_tool},
    {"t22", "What is the date 30 days after 2024-02-01, and how many days is that from 2024-01-01?", ["2024-03-02", "61"], :two_tool},
    # no_tool (8): knowledge or rewriting, no tool should be needed
    {"t23", "Which planet is known as the Red Planet?", ["Mars"], :no_tool},
    {"t24", "What is the chemical symbol for gold?", ["Au"], :no_tool},
    {"t25", "Give the plural of 'mouse' (the animal).", ["mice"], :no_tool},
    {"t26", "Who wrote 'Pride and Prejudice'?", ["Jane Austen", "Austen"], :no_tool},
    {"t27", "Translate 'thank you' into Spanish.", ["gracias"], :no_tool},
    {"t28", "What is the past tense of 'run'?", ["ran"], :no_tool},
    {"t29", "Which language is Jido written in?", ["Elixir"], :no_tool},
    {"t30", "Rewrite 'we will not be attending' as a single word meaning the same.", ["absent", "declining", "no"], :no_tool}
  ]

  @spec all() :: [map()]
  def all do
    for {id, q, expect, kind} <- @tasks, do: %{id: id, q: q <> @suffix, expect: expect, kind: kind}
  end

  @spec normalize(String.t()) :: String.t()
  def normalize(s) do
    s
    |> String.replace(~r/[*_`#"“”'‘’]+/u, "")
    |> String.downcase()
    |> String.replace(~r/[\s$,.:]+/u, " ")
    |> String.trim()
  end

  @spec grade(%{expect: [String.t()]}, String.t()) :: boolean()
  def grade(%{expect: alts}, reply) when is_binary(reply) do
    r = normalize(reply)
    Enum.any?(alts, fn a ->
      n = normalize(a)
      n == r or String.contains?(" #{r} ", " #{n} ")
    end)
  end

  def grade(_task, _non_binary), do: false
end
```

- [ ] **Step 4: Run tests**

Run: `./run.sh . test test/jev_routing/tasks_test.exs`
Expected: `3 tests, 0 failures`.

- [ ] **Step 5: Commit**

```bash
git add experiments/jev_routing/elixir/lib/jev_routing/tasks.ex experiments/jev_routing/elixir/test/jev_routing/tasks_test.exs
git commit -m "feat(experiments): graded task set and normalizing grader

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `JevRouting.Transformer` and `JevRouting.Decisions`

**Files:**
- Create: `lib/jev_routing/decisions.ex`
- Create: `lib/jev_routing/transformer.ex`
- Test: `test/jev_routing/transformer_test.exs`

**Interfaces:**
- Consumes: `TypesafeClient.evaluate/3`, `TypesafeClient.Answer.{Choice,Noul,Score}` (Tasks 1-3); `Jido.AI.Reasoning.ReAct.RequestTransformer` behaviour (`transform_request/4`).
- Produces: `JevRouting.Transformer.transform_request(request, state, config, ctx) :: {:ok, overrides}`; `JevRouting.Transformer.questions([{name, description}]) :: map()`; `JevRouting.Transformer.build_state(request) :: map()`; `JevRouting.Decisions.ensure/0`, `record(request_id, map())`, `take(request_id) :: [map()]`, `clear/0`.

- [ ] **Step 1: Write the failing tests**

```elixir
defmodule JevRouting.TransformerTest do
  use ExUnit.Case, async: false
  alias JevRouting.{Transformer, Decisions}
  alias TypesafeClient.Answer.{Choice, Noul, Score}

  @tools %{"add" => Jido.Tools.Arithmetic.Add, "increment_action" => Jido.Tools.Basic.Increment,
           "count_words" => JevRouting.Tools.CountWords, "gcd" => JevRouting.Tools.Gcd}
  @request %{messages: [%{role: :user, content: "add 2 and 3"}], llm_opts: [], tools: @tools, model: :fast}
  @ctx %{request_id: "req-1"}

  setup do
    Decisions.ensure()
    Decisions.clear()
    :ok
  end

  defp answers(needs, probs, depth) do
    {top, _} = Enum.max_by(probs, fn {_, p} -> p end)
    %{"needs_tool" => %Noul{noul: needs},
      "tool" => %Choice{choice: top, probabilities: probs, confidence: Map.fetch!(probs, top)},
      "depth" => %Score{score: depth, legend: nil, probabilities: %{}, confidence: 0.9}}
  end

  defp run(answers_or_fun, opts \\ []) do
    ctx = @ctx |> Map.merge(Map.new(opts)) |> Map.put(:client_opts, client: TypesafeClient.Stub, answers: answers_or_fun)
    Transformer.transform_request(@request, %{iteration: 0}, %{}, ctx)
  end

  test "gates tools to top-k by probability, excluding none, and picks :fast for shallow depth" do
    probs = %{"add" => 0.6, "increment_action" => 0.25, "gcd" => 0.1, "count_words" => 0.03, "none" => 0.02}
    assert {:ok, %{tools: tools, model: :fast}} = run(answers(0.9, probs, 0.3))
    assert Map.keys(tools) |> Enum.sort() == ["add", "gcd", "increment_action"]
  end

  test "top_k 1 keeps only the argmax" do
    probs = %{"add" => 0.6, "increment_action" => 0.4}
    assert {:ok, %{tools: %{"add" => Jido.Tools.Arithmetic.Add}}} = run(answers(0.9, probs, 0.3), top_k: 1)
  end

  test "empties tools when needs_tool is below threshold" do
    assert {:ok, %{tools: tools}} = run(answers(0.2, %{"add" => 0.5, "none" => 0.5}, 0.3))
    assert tools == %{}
  end

  test "depth thresholds pick :capable and :reasoning" do
    probs = %{"add" => 1.0}
    assert {:ok, %{model: :capable}} = run(answers(0.9, probs, 1.0))
    assert {:ok, %{model: :reasoning}} = run(answers(0.9, probs, 1.7))
  end

  test "ignores unknown tool names" do
    probs = %{"add" => 0.5, "ghost" => 0.4, "gcd" => 0.1}
    assert {:ok, %{tools: tools}} = run(answers(0.9, probs, 0.3), top_k: 2)
    assert Map.keys(tools) |> Enum.sort() == ["add", "gcd"]
  end

  test "fails open on client error" do
    assert {:ok, %{}} = run(fn _s, _q -> {:error, {:http, 529, ""}} end)
  end

  test "returns no overrides when an answer is missing" do
    assert {:ok, %{}} = run(%{"needs_tool" => %Noul{noul: 0.9}})
  end

  test "records a decision per turn in ETS" do
    run(answers(0.9, %{"add" => 1.0}, 0.3))
    assert [%{turn: 0, chosen_model: :fast, chosen_tools: ["add"], needs_tool: 0.9, depth: 0.3, latency_ms: _}] = Decisions.take("req-1")
    assert Decisions.take("req-1") == []
  end

  test "builds state from latest user message and stringifies tool results" do
    req = %{@request | messages: [%{role: :user, content: "first"}, %{role: :assistant, content: "calling"},
                                   %{role: :tool, content: %{result: 5}}, %{role: :user, content: "second"}]}
    state = Transformer.build_state(req)
    assert state.request == "second"
    assert [p] = state.progress
    assert p =~ "5"
    assert Enum.map(state.available_actions, & &1.name) |> Enum.sort() == ["add", "count_words", "gcd", "increment_action"]
  end

  test "questions include every tool name and none as choice criteria" do
    q = Transformer.questions([{"add", "Adds two numbers"}, {"gcd", "Greatest common divisor"}])
    assert %{"type" => "choice", "criteria" => %{"add" => "Adds two numbers", "gcd" => _, "none" => _}} = q["tool"]
    assert q["needs_tool"]["type"] == "noul" and q["depth"]["type"] == "score"
  end
end
```

- [ ] **Step 2: Run to verify failure**

Run: `./run.sh . test test/jev_routing/transformer_test.exs`
Expected: `JevRouting.Transformer` / `JevRouting.Decisions` undefined.

- [ ] **Step 3: Implement `lib/jev_routing/decisions.ex`**

```elixir
defmodule JevRouting.Decisions do
  @moduledoc "ETS log of routing decisions, keyed by request id. Public-table so the transformer can write from the agent process."
  @table :jev_routing_decisions

  def ensure do
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :public, :duplicate_bag, read_concurrency: true])
    end
    :ok
  end

  def record(request_id, %{} = decision) do
    if :ets.whereis(@table) != :undefined, do: :ets.insert(@table, {request_id, decision})
    :ok
  end

  @spec take(term()) :: [map()]
  def take(request_id) do
    if :ets.whereis(@table) == :undefined do
      []
    else
      rows = :ets.lookup(@table, request_id) |> Enum.map(fn {_, d} -> d end) |> Enum.sort_by(& &1.turn)
      :ets.delete(@table, request_id)
      rows
    end
  end

  def clear, do: if(:ets.whereis(@table) != :undefined, do: :ets.delete_all_objects(@table)) && :ok
end
```

- [ ] **Step 4: Implement `lib/jev_routing/transformer.ex`**

```elixir
defmodule JevRouting.Transformer do
  @moduledoc """
  jido_ai ReAct request transformer that asks Jev, before each LLM turn, which tool is
  needed next (gating `tools` to the top-k) and how deep the reasoning is (choosing the
  model alias). Any client failure returns no overrides, so the loop falls back to the
  agent's configured behaviour.
  """
  @behaviour Jido.AI.Reasoning.ReAct.RequestTransformer
  require Logger
  alias JevRouting.Decisions
  alias TypesafeClient.Answer.{Choice, Noul, Score}

  @impl true
  def transform_request(request, state, _config, ctx) do
    cfg = config(ctx)
    tools = request.tools || %{}
    pairs = tools |> Enum.map(fn {name, mod} -> {to_string(name), description(mod)} end) |> Enum.sort()
    client_opts = Map.get(ctx, :client_opts, Application.get_env(:jev_routing, :client_opts, []))

    case TypesafeClient.evaluate(build_state(request), questions(pairs), client_opts) do
      {:ok, %{"needs_tool" => %Noul{noul: needs}, "tool" => %Choice{probabilities: probs}, "depth" => %Score{score: depth}}, meta} ->
        chosen_tools = if needs < cfg.needs_tool_threshold, do: %{}, else: top_k(tools, probs, cfg.top_k)
        model = tier(depth, cfg.depth_thresholds)

        Decisions.record(Map.get(ctx, :request_id), %{
          turn: Map.get(state, :iteration, 0), latency_ms: meta.latency_ms, needs_tool: needs,
          tool_probs: probs, depth: depth, chosen_tools: Map.keys(chosen_tools) |> Enum.sort(), chosen_model: model
        })

        {:ok, %{tools: chosen_tools, model: model}}

      {:ok, _partial, _meta} ->
        Logger.warning("jev_routing: incomplete answers, no overrides")
        {:ok, %{}}

      {:error, reason} ->
        Logger.warning("jev_routing: client error #{inspect(reason)}, no overrides")
        {:ok, %{}}
    end
  end

  @doc "Jev state for one turn: latest user message, up to 3 recent tool results, tool names and descriptions."
  def build_state(%{messages: messages, tools: tools}) do
    user = messages |> Enum.filter(&(role(&1) == "user")) |> List.last()
    progress = messages |> Enum.filter(&(role(&1) == "tool")) |> Enum.take(-3) |> Enum.map(&content_text/1)
    %{request: content_text(user), progress: (if progress == [], do: nil, else: progress),
      available_actions: Enum.map(tools || %{}, fn {n, m} -> %{name: to_string(n), description: description(m)} end)}
  end

  @doc "The three questions, with tool criteria from `[{name, description}]`."
  def questions(pairs) do
    criteria = pairs |> Map.new() |> Map.put("none", "No listed action is needed for the next step")
    %{
      "needs_tool" => %{"type" => "noul",
        "instructions" => "Given `request` and any `progress` so far, does the next step need one of the actions in `available_actions`?",
        "criteria" => %{"true" => "The next step is a computation or lookup that one of the listed actions performs",
                        "false" => "The request is already answered by `progress`, or is general knowledge or rewriting that needs no listed action"}},
      "tool" => %{"type" => "choice",
        "instructions" => "Which action in `available_actions` should run next to make progress on `request`? Pick none if no listed action fits.",
        "criteria" => criteria},
      "depth" => %{"type" => "score",
        "instructions" => "How much reasoning does `request` need beyond calling the actions?",
        "criteria" => ["A single direct step or one action call", "A few dependent steps or two chained action calls",
                       "A long chain of dependent steps, or a well-known trap where the intuitive answer is wrong"]}
    }
  end

  defp top_k(tools, probs, k) do
    probs
    |> Enum.reject(fn {name, _} -> name == "none" end)
    |> Enum.sort_by(fn {_, p} -> -p end)
    |> Enum.map(fn {name, _} -> name end)
    |> Enum.filter(&Map.has_key?(tools, &1))
    |> Enum.take(k)
    |> then(&Map.take(tools, &1))
  end

  defp tier(depth, {low, _high}) when depth < low, do: :fast
  defp tier(depth, {_low, high}) when depth < high, do: :capable
  defp tier(_depth, _), do: :reasoning

  defp config(ctx) do
    %{top_k: Map.get(ctx, :top_k, Application.get_env(:jev_routing, :top_k, 3)),
      needs_tool_threshold: Map.get(ctx, :needs_tool_threshold, Application.get_env(:jev_routing, :needs_tool_threshold, 0.5)),
      depth_thresholds: Map.get(ctx, :depth_thresholds, Application.get_env(:jev_routing, :depth_thresholds, {0.75, 1.5}))}
  end

  defp description(mod) when is_atom(mod) do
    if function_exported?(mod, :description, 0), do: mod.description() || to_string(mod), else: to_string(mod)
  end

  defp role(%{role: r}), do: to_string(r)
  defp role(%{"role" => r}), do: to_string(r)
  defp role(_), do: ""

  defp content_text(nil), do: ""
  defp content_text(%{content: c}), do: content_text(c)
  defp content_text(%{"content" => c}), do: content_text(c)
  defp content_text(c) when is_binary(c), do: c
  defp content_text(list) when is_list(list), do: list |> Enum.map(&content_text/1) |> Enum.join(" ")
  defp content_text(%{text: t}) when is_binary(t), do: t
  defp content_text(%{"text" => t}) when is_binary(t), do: t
  defp content_text(other), do: inspect(other, limit: 50, printable_limit: 300)
end
```

Note on `client_opts`: the tests pass client options through `ctx` for isolation; in production they come from `config :jev_routing, client_opts`. The ReAct runner passes its own runtime context map as `ctx` (`request_id`, `run_id`, ...), and `Map.get(ctx, :client_opts, app_env)` handles both.

- [ ] **Step 5: Run tests**

Run: `./run.sh . test`
Expected: all `jev_routing` tests pass (tools 7, tasks 3, transformer 10). Per-test `top_k` arrives through `ctx`, which `config/1` reads before app env.

- [ ] **Step 6: Commit**

```bash
git add experiments/jev_routing/elixir/lib/jev_routing/decisions.ex experiments/jev_routing/elixir/lib/jev_routing/transformer.ex experiments/jev_routing/elixir/test/jev_routing/transformer_test.exs
git commit -m "feat(experiments): Jev request transformer with tool gating, model tiering and decision log

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Agents, bench, first live run, and write-up

**Files:**
- Create: `lib/jev_routing/agents.ex`
- Create: `lib/jev_routing/bench.ex`
- Modify: `experiments/jev_routing/README.md` (add Experiment 4 section)
- Modify: `experiments/jev_routing/elixir/SPEC.md` (only if the implementation had to deviate)

**Interfaces:**
- Consumes: `JevRouting.Tools.all/0`, `JevRouting.Tasks.{all/0, grade/2}`, `JevRouting.Transformer`, `JevRouting.Decisions`.
- Produces: `JevRouting.Agents.{Fast, Capable, Jev}` (each `use Jido.AI.Agent`); `JevRouting.Bench.main(argv) :: :ok` writing `bench_results.json` and printing a markdown table; `JevRouting.Bench.run_one(condition, task) :: map()`.

- [ ] **Step 1: Implement `lib/jev_routing/agents.ex`**

```elixir
defmodule JevRouting.Agents do
  @moduledoc "The three agents under test. Same tools and prompt; they differ in model and transformer."

  @prompt "You are a precise assistant with tools. Use a tool whenever one computes or looks up what is asked; " <>
          "chain tools when a task has two steps. When no tool applies, answer from knowledge. " <>
          "Finish with only the final answer."

  defmodule Fast do
    use Jido.AI.Agent, name: "bench_fast", model: :fast, tools: JevRouting.Tools.all(),
      system_prompt: @prompt, max_iterations: 6, streaming: false
  end

  defmodule Capable do
    use Jido.AI.Agent, name: "bench_capable", model: :capable, tools: JevRouting.Tools.all(),
      system_prompt: @prompt, max_iterations: 6, streaming: false
  end

  defmodule Jev do
    use Jido.AI.Agent, name: "bench_jev", model: :fast, tools: JevRouting.Tools.all(),
      system_prompt: @prompt, max_iterations: 6, streaming: false,
      request_transformer: JevRouting.Transformer
  end

  def prompt, do: @prompt
end
```

If `use Jido.AI.Agent` rejects a module attribute for `system_prompt` inside nested modules, inline the string in each module. If it rejects a function call for `tools:`, replace with the explicit module list from `JevRouting.Tools` (copy `@builtins ++ synthetic()` as a literal list).

- [ ] **Step 2: Implement `lib/jev_routing/bench.ex`**

```elixir
defmodule JevRouting.Bench do
  @moduledoc """
  Runs every task under every condition with a fresh agent process, grades the answer,
  reads iterations and usage from the strategy snapshot, and reports per-condition
  pass rate, turns, wall time, cost and Jev overhead.
  """
  alias JevRouting.{Agents, Decisions, Tasks}

  @conditions %{"baseline-fast" => Agents.Fast, "baseline-capable" => Agents.Capable, "jev" => Agents.Jev}
  # $ per MTok, input / output
  @price %{fast: {1.0, 5.0}, capable: {2.0, 10.0}, reasoning: {5.0, 25.0}}
  @timeout 180_000

  def main(argv \\ []) do
    {opts, _, _} = OptionParser.parse(argv, strict: [conditions: :string, tasks: :string, concurrency: :integer])
    conds = (opts[:conditions] || Map.keys(@conditions) |> Enum.join(",")) |> String.split(",")
    tasks = Tasks.all() |> filter_tasks(opts[:tasks])
    Decisions.ensure()

    rows =
      for c <- conds do
        Task.async_stream(tasks, &run_one(c, &1), max_concurrency: opts[:concurrency] || 2, timeout: @timeout + 30_000, ordered: true)
        |> Enum.map(fn {:ok, r} -> r end)
      end
      |> List.flatten()

    summary = Map.new(conds, fn c -> {c, summarize(Enum.filter(rows, &(&1.condition == c)))} end)
    models = Map.new([:fast, :capable, :reasoning], fn a -> {a, inspect(Jido.AI.resolve_model(a))} end)
    File.write!("bench_results.json", Jason.encode!(%{ran_at: DateTime.utc_now(), models: models, summary: summary, rows: rows}, pretty: true))
    IO.puts(table(summary))
    :ok
  end

  def run_one(condition, task) do
    mod = Map.fetch!(@conditions, condition)
    {:ok, pid} = Jido.AgentServer.start_link(agent: mod)
    t0 = System.monotonic_time(:millisecond)
    {status, reply} =
      case mod.ask_sync(pid, task.q, timeout: @timeout) do
        {:ok, result} -> {:ok, extract_text(result)}
        {:error, reason} -> {:error, inspect(reason)}
      end
    wall = System.monotonic_time(:millisecond) - t0
    {iterations, usage, request_id} = snapshot(pid, mod)
    decisions = if condition == "jev", do: Decisions.take(request_id), else: []
    GenServer.stop(pid, :normal, 5_000)

    tier = if condition == "jev", do: max_tier(decisions), else: (if condition == "baseline-capable", do: :capable, else: :fast)
    %{condition: condition, task_id: task.id, kind: task.kind, status: status, reply: String.slice(reply, 0, 300),
      pass: status == :ok and Tasks.grade(task, reply), iterations: iterations, usage: usage, wall_ms: wall,
      cost_usd: cost(usage, tier), cost_tier: tier, jev_ms: Enum.sum(Enum.map(decisions, & &1.latency_ms)),
      decisions: Enum.map(decisions, &Map.take(&1, [:turn, :needs_tool, :depth, :chosen_tools, :chosen_model, :latency_ms]))}
  end

  defp snapshot(pid, mod) do
    case Jido.AgentServer.state(pid) do
      {:ok, %{agent: agent}} ->
        snap = mod.strategy_snapshot(agent)
        details = snap.details || %{}
        strategy_state = get_in(agent.state, [:__strategy__]) || %{}
        {details[:iteration] || strategy_state[:iteration] || 0, details[:usage] || strategy_state[:usage] || %{},
         strategy_state[:last_request_id] || strategy_state[:active_request_id] || details[:request_id]}
      _ -> {0, %{}, nil}
    end
  end

  # jido_ai returns the final answer as a binary in the common case; be tolerant of maps.
  defp extract_text(r) when is_binary(r), do: r
  defp extract_text(%{result: r}) when is_binary(r), do: r
  defp extract_text(%{text: t}) when is_binary(t), do: t
  defp extract_text(%{"result" => r}) when is_binary(r), do: r
  defp extract_text(other), do: inspect(other, limit: 200)

  defp max_tier([]), do: :fast
  defp max_tier(decisions), do: decisions |> Enum.map(& &1.chosen_model) |> Enum.max_by(&Map.fetch!(%{fast: 0, capable: 1, reasoning: 2}, &1))

  defp cost(usage, tier) do
    {pin, pout} = Map.fetch!(@price, tier)
    input = usage[:input_tokens] || usage["input_tokens"] || 0
    output = usage[:output_tokens] || usage["output_tokens"] || 0
    (input * pin + output * pout) / 1_000_000
  end

  defp filter_tasks(tasks, nil), do: tasks
  defp filter_tasks(tasks, ids), do: (set = String.split(ids, ","); Enum.filter(tasks, &(&1.id in set)))

  defp summarize(rows) do
    n = length(rows)
    walls = rows |> Enum.map(& &1.wall_ms) |> Enum.sort()
    %{n: n, pass_rate: Enum.count(rows, & &1.pass) / max(n, 1), errors: Enum.count(rows, &(&1.status == :error)),
      mean_iterations: mean(Enum.map(rows, & &1.iterations)), median_wall_ms: Enum.at(walls, div(n, 2)) || 0,
      p95_wall_ms: Enum.at(walls, min(n - 1, round(0.95 * (n - 1)))) || 0, mean_cost_usd: mean(Enum.map(rows, & &1.cost_usd)),
      mean_jev_ms: mean(Enum.map(rows, & &1.jev_ms)), tier_mix: rows |> Enum.frequencies_by(& &1.cost_tier),
      pass_by_kind: rows |> Enum.group_by(& &1.kind) |> Map.new(fn {k, rs} -> {k, "#{Enum.count(rs, & &1.pass)}/#{length(rs)}"} end)}
  end

  defp mean([]), do: 0.0
  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp table(summary) do
    header = "| condition | pass | one/two/no-tool | mean turns | median ms | p95 ms | $/task | jev ms | tier mix |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    header <>
      Enum.map_join(summary, "\n", fn {c, s} ->
        k = s.pass_by_kind
        "| #{c} | #{Float.round(s.pass_rate, 2)} | #{k[:one_tool]} / #{k[:two_tool]} / #{k[:no_tool]} | #{Float.round(s.mean_iterations / 1, 2)} | " <>
          "#{s.median_wall_ms} | #{s.p95_wall_ms} | #{:erlang.float_to_binary(s.mean_cost_usd / 1, decimals: 5)} | #{round(s.mean_jev_ms)} | #{inspect(s.tier_mix)} |"
      end)
  end
end
```

- [ ] **Step 3: Compile and smoke-test one task per condition**

Run: `./run.sh . compile --warnings-as-errors 2>&1 | tail -5`
Expected: compiles. Fix any warning (unused variable, etc.).

Run: `./run.sh . bench --tasks t01,t23 --conditions baseline-fast`
Expected: a two-row result and a table. Inspect `bench_results.json`: `reply` should be the answer text, `iterations` >= 1, `usage` non-empty. If `reply` is `inspect` output of a struct, adjust `extract_text/1` to that struct's answer field and re-run. If `iterations` is 0 or `usage` empty, print `mod.strategy_snapshot(agent)` once with `IO.inspect` to find the right keys and adjust `snapshot/2`.

Run: `./run.sh . bench --tasks t01,t15,t23 --conditions jev`
Expected: `decisions` non-empty for each row, `jev_ms` > 0, `chosen_tools` a short list for t01 and t15, `[]` or short for t23.

- [ ] **Step 4: Full run**

Run: `./run.sh . bench 2>&1 | tee bench_run.log`
Expected: 90 rows (30 tasks x 3 conditions), a markdown table. Rough cost: under $2 for the LLM calls and ~90 Jev calls.

- [ ] **Step 5: Record results in the README and the doc**

Add an "Experiment 4: inside the loop (Elixir)" section to `experiments/jev_routing/README.md` with the table verbatim from stdout, the resolved dependency versions from Task 4, the model IDs actually used, per-kind pass rates, and the caveats (one run per task, hand-written tasks, cost for `jev` priced at the most expensive tier chosen in that run as an upper bound). Copy `bench_results.json` to `experiments/jev_routing/bench_results_elixir.json` so it is committed (the in-project file is gitignored).

- [ ] **Step 6: Run the whole test suite and format**

Run: `./run.sh . format --check-formatted || ./run.sh . format`
Run: `./run.sh . test` and `./run.sh typesafe_client test`
Expected: all green.

- [ ] **Step 7: Commit and push**

```bash
git add experiments/jev_routing/elixir experiments/jev_routing/README.md experiments/jev_routing/bench_results_elixir.json
git commit -m "feat(experiments): live ReAct bench comparing Jev-routed agent with baselines

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push fork experiment/jev-tool-routing
```
