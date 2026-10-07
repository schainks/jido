defmodule JevRouting.Bench do
  @moduledoc """
  Runs every task under every condition with a fresh agent process, grades the answer,
  reads iterations and usage from the strategy snapshot, and reports per-condition
  pass rate, turns, wall time, cost and Jev overhead.
  """
  alias JevRouting.{Agents, Decisions, Tasks}

  @conditions %{
    "baseline-fast" => Agents.Fast,
    "baseline-capable" => Agents.Capable,
    "jev" => Agents.Jev
  }
  # $ per MTok, input / output
  @price %{fast: {1.0, 5.0}, capable: {2.0, 10.0}, reasoning: {5.0, 25.0}}
  @rank %{fast: 0, capable: 1, reasoning: 2}
  @timeout 180_000

  def main(argv \\ []) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [conditions: :string, tasks: :string, concurrency: :integer]
      )

    {:ok, conds} =
      (opts[:conditions] || Enum.join(Map.keys(@conditions), ","))
      |> String.split(",")
      |> validate_conditions()

    tasks = Tasks.all() |> filter_tasks(opts[:tasks])
    Decisions.ensure()
    ensure_instance()

    rows =
      for c <- conds do
        tasks
        |> Task.async_stream(&safe_run_one(c, &1),
          max_concurrency: opts[:concurrency] || 2,
          timeout: @timeout + 30_000,
          on_timeout: :kill_task,
          ordered: true
        )
        |> Enum.zip(tasks)
        |> Enum.map(fn
          {{:ok, r}, _task} -> r
          {{:exit, reason}, task} -> error_row(c, task, {:exit, reason})
        end)
      end
      |> List.flatten()

    summary = Map.new(conds, fn c -> {c, summarize(Enum.filter(rows, &(&1.condition == c)))} end)

    models =
      Map.new([:fast, :capable, :reasoning], fn a ->
        spec = Jido.AI.resolve_model(a)

        resolved =
          case LLMDB.model(spec) do
            {:ok, m} -> "#{m.provider}:#{m.id}"
            _ -> "unresolved"
          end

        {a, %{alias: inspect(spec), resolved: resolved}}
      end)

    File.write!(
      "bench_results.json",
      Jason.encode!(%{ran_at: DateTime.utc_now(), models: models, summary: summary, rows: rows},
        pretty: true
      )
    )

    IO.puts(table(summary))
    :ok
  end

  @doc "Accepts only known condition names."
  def validate_conditions(names) do
    case Enum.reject(names, &Map.has_key?(@conditions, &1)) do
      [] -> {:ok, names}
      bad -> {:error, {:unknown_conditions, bad}}
    end
  end

  @doc "A failed row standing in for a run that crashed or timed out, so one crash cannot lose the batch."
  def error_row(condition, task, reason) do
    %{
      condition: condition,
      task_id: task.id,
      kind: task.kind,
      status: :error,
      reply: inspect(reason, limit: 50, printable_limit: 300),
      pass: false,
      iterations: 0,
      usage: %{},
      wall_ms: 0,
      cost_usd: 0.0,
      cost_tier: :fast,
      jev_ms: 0,
      decisions: []
    }
  end

  defp safe_run_one(condition, task) do
    run_one(condition, task)
  rescue
    e -> error_row(condition, task, {:raised, Exception.message(e)})
  catch
    kind, reason -> error_row(condition, task, {kind, reason})
  end

  def run_one(condition, task) do
    mod = Map.fetch!(@conditions, condition)
    {:ok, pid} = Jido.AgentServer.start_link(jido: JevRouting.Jido, agent: mod)
    t0 = System.monotonic_time(:millisecond)

    {request_id, {status, reply}} =
      case mod.ask(pid, task.q) do
        {:ok, handle} ->
          {handle.id,
           case mod.await(handle, timeout: @timeout) do
             {:ok, result} -> {:ok, extract_text(result)}
             {:error, reason} -> {:error, inspect(reason, limit: 50, printable_limit: 300)}
           end}

        {:error, reason} ->
          {nil, {:error, inspect(reason, limit: 50, printable_limit: 300)}}
      end

    wall = System.monotonic_time(:millisecond) - t0
    {iterations, usage} = snapshot(pid, mod)
    decisions = if condition == "jev", do: Decisions.take(request_id), else: []
    GenServer.stop(pid, :normal, 5_000)

    tier =
      cond do
        condition == "jev" -> max_tier(decisions)
        condition == "baseline-capable" -> :capable
        true -> :fast
      end

    %{
      condition: condition,
      task_id: task.id,
      kind: task.kind,
      status: status,
      reply: String.slice(reply, 0, 300),
      pass: status == :ok and Tasks.grade(task, reply),
      iterations: iterations,
      usage: usage,
      wall_ms: wall,
      cost_usd: cost(usage, tier),
      cost_tier: tier,
      jev_ms: Enum.sum(Enum.map(decisions, & &1.latency_ms)),
      decisions:
        Enum.map(
          decisions,
          &Map.take(&1, [
            :turn,
            :needs_tool,
            :depth,
            :chosen_tools,
            :chosen_model,
            :latency_ms,
            :tool_probs,
            :request,
            :progress
          ])
        )
    }
  end

  defp ensure_instance do
    case JevRouting.Jido.start_link([]) do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end
  end

  defp snapshot(pid, mod) do
    case Jido.AgentServer.state(pid) do
      {:ok, %{agent: agent}} ->
        snap = mod.strategy_snapshot(agent)
        details = snap.details || %{}
        strategy_state = Map.get(agent.state, :__strategy__) || %{}

        {details[:iteration] || strategy_state[:iteration] || 0,
         details[:usage] || strategy_state[:usage] || %{}}

      _ ->
        {0, %{}}
    end
  end

  # jido_ai returns the final answer as a binary in the common case; be tolerant of maps.
  defp extract_text(r) when is_binary(r), do: r
  defp extract_text(%{result: r}) when is_binary(r), do: r
  defp extract_text(%{text: t}) when is_binary(t), do: t
  defp extract_text(%{"result" => r}) when is_binary(r), do: r
  defp extract_text(other), do: inspect(other, limit: 200, printable_limit: 600)

  defp max_tier([]), do: :fast

  defp max_tier(decisions),
    do: decisions |> Enum.map(& &1.chosen_model) |> Enum.max_by(&Map.fetch!(@rank, &1))

  defp cost(usage, tier) do
    {pin, pout} = Map.fetch!(@price, tier)
    input = usage[:input_tokens] || usage["input_tokens"] || 0
    output = usage[:output_tokens] || usage["output_tokens"] || 0
    (input * pin + output * pout) / 1_000_000
  end

  defp filter_tasks(tasks, nil), do: tasks

  defp filter_tasks(tasks, ids) do
    set = String.split(ids, ",")
    Enum.filter(tasks, &(&1.id in set))
  end

  defp summarize(rows) do
    n = length(rows)
    walls = rows |> Enum.map(& &1.wall_ms) |> Enum.sort()

    %{
      n: n,
      pass_rate: Enum.count(rows, & &1.pass) / max(n, 1),
      errors: Enum.count(rows, &(&1.status == :error)),
      mean_iterations: mean(Enum.map(rows, & &1.iterations)),
      median_wall_ms: Enum.at(walls, div(n, 2)) || 0,
      p95_wall_ms: Enum.at(walls, min(n - 1, round(0.95 * (n - 1)))) || 0,
      mean_cost_usd: mean(Enum.map(rows, & &1.cost_usd)),
      mean_input_tokens:
        mean(Enum.map(rows, &(&1.usage[:input_tokens] || &1.usage["input_tokens"] || 0))),
      mean_jev_ms: mean(Enum.map(rows, & &1.jev_ms)),
      tier_mix: rows |> Enum.frequencies_by(& &1.cost_tier),
      pass_by_kind:
        rows
        |> Enum.group_by(& &1.kind)
        |> Map.new(fn {k, rs} -> {k, "#{Enum.count(rs, & &1.pass)}/#{length(rs)}"} end)
    }
  end

  defp mean([]), do: 0.0
  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp table(summary) do
    header =
      "| condition | pass | one/two/no-tool | mean turns | median ms | p95 ms | mean input tokens | $/task | jev ms | tier mix |\n" <>
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"

    header <>
      Enum.map_join(summary, "\n", fn {c, s} ->
        k = s.pass_by_kind

        "| #{c} | #{Float.round(s.pass_rate / 1, 2)} | #{k[:one_tool]} / #{k[:two_tool]} / #{k[:no_tool]} | " <>
          "#{Float.round(s.mean_iterations / 1, 2)} | #{s.median_wall_ms} | #{s.p95_wall_ms} | #{round(s.mean_input_tokens)} | " <>
          "#{:erlang.float_to_binary(s.mean_cost_usd / 1, decimals: 5)} | #{round(s.mean_jev_ms)} | #{inspect(s.tier_mix)} |"
      end)
  end
end
