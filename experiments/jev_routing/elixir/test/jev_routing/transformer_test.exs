defmodule JevRouting.TransformerTest do
  use ExUnit.Case, async: false
  @moduletag capture_log: true
  alias JevRouting.{Transformer, Decisions}
  alias TypesafeClient.Answer.{Choice, Noul, Score}

  @tools %{
    "add" => Jido.Tools.Arithmetic.Add,
    "increment_action" => Jido.Tools.Basic.Increment,
    "count_words" => JevRouting.Tools.CountWords,
    "gcd" => JevRouting.Tools.Gcd
  }
  @request %{
    messages: [%{role: :user, content: "add 2 and 3"}],
    llm_opts: [],
    tools: @tools,
    model: :fast
  }
  @ctx %{request_id: "req-1"}

  setup do
    Decisions.ensure()
    Decisions.clear()
    :ok
  end

  defp answers(needs, probs, depth) do
    {top, _} = Enum.max_by(probs, fn {_, p} -> p end)

    %{
      "needs_tool" => %Noul{noul: needs},
      "tool" => %Choice{choice: top, probabilities: probs, confidence: Map.fetch!(probs, top)},
      "depth" => %Score{score: depth, legend: nil, probabilities: %{}, confidence: 0.9}
    }
  end

  defp run(answers_or_fun, opts \\ []) do
    ctx =
      @ctx
      |> Map.merge(Map.new(opts))
      |> Map.put(:client_opts, client: TypesafeClient.Stub, answers: answers_or_fun)

    Transformer.transform_request(@request, %{iteration: 0}, %{}, ctx)
  end

  test "gates tools to top-k by probability, excluding none, and picks :fast for shallow depth" do
    probs = %{
      "add" => 0.6,
      "increment_action" => 0.25,
      "gcd" => 0.1,
      "count_words" => 0.03,
      "none" => 0.02
    }

    assert {:ok, %{tools: tools, model: :fast}} = run(answers(0.9, probs, 0.3))
    assert Map.keys(tools) |> Enum.sort() == ["add", "gcd", "increment_action"]
  end

  test "top_k 1 keeps only the argmax" do
    probs = %{"add" => 0.6, "increment_action" => 0.4}

    assert {:ok, %{tools: %{"add" => Jido.Tools.Arithmetic.Add} = tools}} =
             run(answers(0.9, probs, 0.3), top_k: 1)

    assert map_size(tools) == 1
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
    assert {:ok, overrides} = run(fn _s, _q -> {:error, {:http, 529, ""}} end)
    assert overrides == %{}
  end

  test "returns no overrides when an answer is missing" do
    assert {:ok, overrides} = run(%{"needs_tool" => %Noul{noul: 0.9}})
    assert overrides == %{}
  end

  test "records a decision per turn in ETS" do
    run(answers(0.9, %{"add" => 1.0}, 0.3))

    assert [
             %{
               turn: 0,
               chosen_model: :fast,
               chosen_tools: ["add"],
               needs_tool: 0.9,
               depth: 0.3,
               latency_ms: _
             }
           ] = Decisions.take("req-1")

    assert Decisions.take("req-1") == []
  end

  test "builds state from latest user message and stringifies tool results" do
    req = %{
      @request
      | messages: [
          %{role: :user, content: "first"},
          %{role: :assistant, content: "calling"},
          %{role: :tool, content: %{result: 5}},
          %{role: :user, content: "second"}
        ]
    }

    state = Transformer.build_state(req)
    assert state.request == "second"
    assert [p] = state.progress
    assert p =~ "5"

    assert Enum.map(state.available_actions, & &1.name) |> Enum.sort() ==
             ["add", "count_words", "gcd", "increment_action"]
  end

  test "questions include every tool name and none as choice criteria" do
    q = Transformer.questions([{"add", "Adds two numbers"}, {"gcd", "Greatest common divisor"}])

    assert %{"type" => "choice", "criteria" => %{"add" => "Adds two numbers", "gcd" => _, "none" => _}} =
             q["tool"]

    assert q["needs_tool"]["type"] == "noul" and q["depth"]["type"] == "score"
  end
end
