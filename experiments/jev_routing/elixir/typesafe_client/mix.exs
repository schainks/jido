defmodule TypesafeClient.MixProject do
  use Mix.Project

  def project do
    [
      app: :typesafe_client,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Client for TypeSafe's System One API (Jev): typed Choice, Noul and Score judgments.",
      package: [licenses: ["Apache-2.0"], links: %{}]
    ]
  end

  def application, do: [extra_applications: [:logger]]

  defp deps do
    [
      {:req, "~> 0.5"},
      {:jason, "~> 1.4"},
      {:plug, "~> 1.16", only: :test}
    ]
  end
end
