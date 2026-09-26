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
