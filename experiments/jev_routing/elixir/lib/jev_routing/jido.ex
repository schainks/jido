defmodule JevRouting.Jido do
  @moduledoc "Jido instance for the bench: owns the agent registry, agent supervisor and task supervisor."
  use Jido, otp_app: :jev_routing
end
