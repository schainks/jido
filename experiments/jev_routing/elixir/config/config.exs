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
