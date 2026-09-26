defmodule JevRouting.BenchTest do
  use ExUnit.Case, async: true
  alias JevRouting.Bench

  test "validate_conditions accepts known names and rejects unknown ones" do
    assert {:ok, ["jev", "baseline-fast"]} = Bench.validate_conditions(["jev", "baseline-fast"])
    assert {:error, {:unknown_conditions, ["jevv"]}} = Bench.validate_conditions(["jev", "jevv"])
  end

  test "error_row turns a crash into a failed row instead of losing the run" do
    task = %{id: "t99", kind: :one_tool, expect: ["x"], q: "?"}
    row = Bench.error_row("jev", task, {:exit, :boom})
    assert row.status == :error and row.pass == false and row.task_id == "t99"
    assert row.reply =~ "boom"
  end
end
