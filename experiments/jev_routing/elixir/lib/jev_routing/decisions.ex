defmodule JevRouting.Decisions do
  @moduledoc """
  ETS log of routing decisions, keyed by request id. Public table so the transformer can
  write from the agent process while the bench reads from its own.
  """
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
      rows =
        :ets.lookup(@table, request_id)
        |> Enum.map(fn {_, d} -> d end)
        |> Enum.sort_by(& &1.turn)

      :ets.delete(@table, request_id)
      rows
    end
  end

  def clear do
    if :ets.whereis(@table) != :undefined, do: :ets.delete_all_objects(@table)
    :ok
  end
end
