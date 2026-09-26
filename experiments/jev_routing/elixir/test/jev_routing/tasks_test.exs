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
