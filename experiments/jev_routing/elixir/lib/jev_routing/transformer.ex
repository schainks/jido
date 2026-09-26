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
    tools = Map.get(request, :tools) || %{}

    pairs = Enum.map(tools, fn {name, mod} -> {to_string(name), description(mod)} end)

    client_opts = Map.get(ctx, :client_opts, Application.get_env(:jev_routing, :client_opts, []))

    jev_state = build_state(request)

    case safe_evaluate(jev_state, questions(pairs), client_opts) do
      {:ok,
       %{
         "needs_tool" => %Noul{noul: needs},
         "tool" => %Choice{probabilities: probs},
         "depth" => %Score{score: depth}
       }, meta} ->
        model = tier(depth, cfg.depth_thresholds)

        # tools override: %{} when no tool is needed; top-k of the tools Jev actually
        # scored above zero; no override at all if Jev named only unknown tools.
        overrides =
          cond do
            needs < cfg.needs_tool_threshold -> %{tools: %{}, model: model}
            (gated = top_k(tools, probs, cfg.top_k)) != %{} -> %{tools: gated, model: model}
            true -> %{model: model}
          end

        Decisions.record(Map.get(ctx, :request_id), %{
          turn: Map.get(state, :iteration, 0),
          latency_ms: meta.latency_ms,
          needs_tool: needs,
          tool_probs: probs,
          depth: depth,
          chosen_tools: overrides |> Map.get(:tools, tools) |> Map.keys() |> Enum.sort(),
          chosen_model: model,
          request: jev_state.request,
          progress: jev_state.progress
        })

        {:ok, overrides}

      {:ok, _partial, _meta} ->
        Logger.warning("jev_routing: incomplete answers, no overrides")
        {:ok, %{}}

      {:error, reason} ->
        Logger.warning("jev_routing: client error #{inspect(reason)}, no overrides")
        {:ok, %{}}
    end
  end

  @doc "Jev state for one turn: latest user message, up to 3 recent tool results, tool names and descriptions."
  def build_state(%{messages: messages} = request) do
    tools = Map.get(request, :tools) || %{}
    user = messages |> Enum.filter(&(role(&1) == "user")) |> List.last()

    progress =
      messages |> Enum.filter(&(role(&1) == "tool")) |> Enum.take(-3) |> Enum.map(&content_text/1)

    %{
      request: content_text(user),
      progress: if(progress == [], do: nil, else: progress),
      available_actions:
        Enum.map(tools, fn {n, m} -> %{name: to_string(n), description: description(m)} end)
    }
  end

  @doc "The three questions, with tool criteria from `[{name, description}]`."
  def questions(pairs) do
    criteria =
      pairs |> Map.new() |> Map.put("none", "No listed action is needed for the next step")

    %{
      "needs_tool" => %{
        "type" => "noul",
        "instructions" =>
          "Given `request` and any `progress` so far, does the next step need one of the actions in `available_actions`?",
        "criteria" => %{
          "true" =>
            "The next step is a computation or lookup that one of the listed actions performs",
          "false" =>
            "The request is already answered by `progress`, or is general knowledge or rewriting that needs no listed action"
        }
      },
      "tool" => %{
        "type" => "choice",
        "instructions" =>
          "Which action in `available_actions` should run next to make progress on `request`? Pick none if no listed action fits.",
        "criteria" => criteria
      },
      "depth" => %{
        "type" => "score",
        "instructions" => "How much reasoning does `request` need beyond calling the actions?",
        "criteria" => [
          "A single direct step or one action call",
          "A few dependent steps or two chained action calls",
          "A long chain of dependent steps, or a well-known trap where the intuitive answer is wrong"
        ]
      }
    }
  end

  # Never let the client take the agent down: any raise becomes a fail-open.
  defp safe_evaluate(state, questions, opts) do
    TypesafeClient.evaluate(state, questions, opts)
  rescue
    e -> {:error, {:raised, Exception.message(e)}}
  catch
    kind, reason -> {:error, {kind, reason}}
  end

  defp top_k(tools, probs, k) do
    probs
    |> Enum.reject(fn {name, p} -> name == "none" or not is_number(p) or p <= 0 end)
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
    %{
      top_k: Map.get(ctx, :top_k, Application.get_env(:jev_routing, :top_k, 3)),
      needs_tool_threshold:
        Map.get(
          ctx,
          :needs_tool_threshold,
          Application.get_env(:jev_routing, :needs_tool_threshold, 0.5)
        ),
      depth_thresholds:
        Map.get(
          ctx,
          :depth_thresholds,
          Application.get_env(:jev_routing, :depth_thresholds, {0.75, 1.5})
        )
    }
  end

  defp description(mod) when is_atom(mod) do
    if function_exported?(mod, :description, 0),
      do: mod.description() || to_string(mod),
      else: to_string(mod)
  end

  defp role(%{role: r}), do: to_string(r)
  defp role(%{"role" => r}), do: to_string(r)
  defp role(_), do: ""

  defp content_text(nil), do: ""
  defp content_text(%{content: c}), do: content_text(c)
  defp content_text(%{"content" => c}), do: content_text(c)
  defp content_text(c) when is_binary(c), do: c

  defp content_text(list) when is_list(list),
    do: list |> Enum.map(&content_text/1) |> Enum.join(" ")

  defp content_text(%{text: t}) when is_binary(t), do: t
  defp content_text(%{"text" => t}) when is_binary(t), do: t

  # ReqLLM-style content parts: name the tool and show its output
  defp content_text(%{} = part) when is_map_key(part, :tool_name) or is_map_key(part, :output) do
    name = Map.get(part, :tool_name) || Map.get(part, :name) || "tool"
    out = Map.get(part, :output) || Map.get(part, :result)
    "#{name} -> #{inspect(out, limit: 50, printable_limit: 300)}"
  end

  defp content_text(other), do: inspect(other, limit: 50, printable_limit: 300)
end
