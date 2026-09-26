defmodule TypesafeClient.StubTest do
  use ExUnit.Case, async: false
  alias TypesafeClient.Answer.{Noul, Choice}

  @q %{"a" => %{"type" => "noul", "instructions" => "?"}}

  test "returns canned struct answers" do
    assert {:ok, %{"a" => %Noul{noul: 0.2}}, %{latency_ms: 0}} =
             TypesafeClient.Stub.evaluate("s", @q, answers: %{"a" => %Noul{noul: 0.2}})
  end

  test "parses canned raw answers" do
    assert {:ok, %{"a" => %Choice{choice: "x"}}, _} =
             TypesafeClient.Stub.evaluate("s", @q,
               answers: %{
                 "a" => %{
                   "type" => "choice",
                   "choice" => "x",
                   "probabilities" => %{"x" => 1.0},
                   "confidence" => 1.0
                 }
               }
             )
  end

  test "calls a function with state and questions, and passes errors through" do
    fun = fn state, questions ->
      if state == :boom,
        do: {:error, :down},
        else: %{"a" => %Noul{noul: map_size(questions) / 1}}
    end

    assert {:ok, %{"a" => %Noul{noul: 1.0}}, _} =
             TypesafeClient.Stub.evaluate(:ok, @q, answers: fun)

    assert {:error, :down} = TypesafeClient.Stub.evaluate(:boom, @q, answers: fun)
  end

  test "dispatcher uses opts[:client] then app env" do
    assert {:ok, %{"a" => %Noul{noul: 0.5}}, _} =
             TypesafeClient.evaluate("s", @q,
               client: TypesafeClient.Stub,
               answers: %{"a" => %Noul{noul: 0.5}}
             )

    Application.put_env(:typesafe_client, :client, TypesafeClient.Stub)
    on_exit(fn -> Application.delete_env(:typesafe_client, :client) end)
    assert {:ok, _, _} = TypesafeClient.evaluate("s", @q, answers: %{"a" => %Noul{noul: 0.5}})
  end
end
