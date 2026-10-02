defmodule JevRouting.Agents do
  @moduledoc """
  The three agents under test. Same tools and prompt; they differ in model and transformer.

  `use Jido.AI.Agent` reads its options from the raw AST at compile time, so `tools:` must
  be a literal list and `system_prompt:` a literal string. `JevRouting.Agents.Define`
  keeps that single source of truth and splices it into each agent.
  """

  defmodule Define do
    @moduledoc false

    # Literal list: must stay in sync with JevRouting.Tools.all/0 (a test checks this).
    @tools [
      Jido.Tools.Arithmetic.Add,
      Jido.Tools.Arithmetic.Subtract,
      Jido.Tools.Arithmetic.Multiply,
      Jido.Tools.Arithmetic.Divide,
      Jido.Tools.Arithmetic.Square,
      Jido.Tools.Basic.Increment,
      Jido.Tools.Basic.Decrement,
      Jido.Tools.Basic.Noop,
      Jido.Tools.Basic.Today,
      JevRouting.Tools.ConvertUnit,
      JevRouting.Tools.DateAdd,
      JevRouting.Tools.DateDiff,
      JevRouting.Tools.CountWords,
      JevRouting.Tools.CountChars,
      JevRouting.Tools.Base64Encode,
      JevRouting.Tools.Base64Decode,
      JevRouting.Tools.ReverseString,
      JevRouting.Tools.Gcd,
      JevRouting.Tools.SortNumbers,
      JevRouting.Tools.JsonGet,
      JevRouting.Tools.Sha256,
      JevRouting.Tools.CapitalOf,
      JevRouting.Tools.ExchangeRate,
      JevRouting.Tools.RegexMatch
    ]

    @prompt "You are a precise assistant with tools. Use a tool whenever one computes or looks up what is asked; " <>
              "chain tools when a task has two steps. When no tool applies, answer from knowledge. " <>
              "Finish with only the final answer."

    def tools, do: @tools
    def prompt, do: @prompt

    defmacro __using__(opts) do
      kw =
        [
          name: Keyword.fetch!(opts, :name),
          model: Keyword.fetch!(opts, :model),
          tools: @tools,
          system_prompt: @prompt,
          max_iterations: 6,
          streaming: false
        ] ++ Keyword.take(opts, [:request_transformer])

      quote do
        use Jido.AI.Agent, unquote(kw)
      end
    end
  end

  defmodule Fast do
    @moduledoc false
    use JevRouting.Agents.Define, name: "bench_fast", model: :fast
  end

  defmodule Capable do
    @moduledoc false
    use JevRouting.Agents.Define, name: "bench_capable", model: :capable
  end

  defmodule Jev do
    @moduledoc false
    use JevRouting.Agents.Define,
      name: "bench_jev",
      model: :fast,
      request_transformer: JevRouting.Transformer
  end
end
