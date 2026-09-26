defmodule JevRouting.Tasks do
  @moduledoc "Graded tasks for the bench and the grader shared with tests."

  @suffix " Reply with only the final answer."

  @tasks [
    # one_tool (14)
    {"t01", "Convert 5 miles to kilometers, two decimals.", ["8.05"], :one_tool},
    {"t02", "What is the date 45 days after 2024-11-20?", ["2025-01-04"], :one_tool},
    {"t03", "How many days are there from 2024-01-01 to 2025-01-01?", ["366"], :one_tool},
    {"t04", "How many words are in: 'the quick brown fox jumps over the lazy dog'?", ["9"],
     :one_tool},
    {"t05", "How many times does the letter r appear in 'strawberry'?", ["3"], :one_tool},
    {"t06", "Base64-encode the text 'jido agent'.", ["amlkbyBhZ2VudA=="], :one_tool},
    {"t07", "Decode this Base64: c3lzdGVtIG9uZQ==", ["system one"], :one_tool},
    {"t08", "Reverse the string 'directive'.", ["evitcerid"], :one_tool},
    {"t09", "What is the greatest common divisor of 1071 and 462?", ["21"], :one_tool},
    {"t10", "Sort ascending: 42, 7, 19, 3, 88.",
     ["3 7 19 42 88", "3, 7, 19, 42, 88", "[3, 7, 19, 42, 88]"], :one_tool},
    {"t11", ~s(In the JSON {"a":{"b":[1,2,{"c":"x"}]}} what is the value at a.b[2].c?), ["x"],
     :one_tool},
    {"t12", "What is the capital of Kenya?", ["Nairobi"], :one_tool},
    {"t13", "Multiply 1234 by 5678.", ["7006652", "7,006,652"], :one_tool},
    {"t14", "Increment the value 41 by one.", ["42"], :one_tool},
    # two_tool (8): each needs two dependent tool calls
    {"t15", "Convert 10 miles to kilometers, then square the result. Two decimals.",
     ["258.89", "258.99", "259.0", "259"], :two_tool},
    {"t16", "How many days from 2024-03-01 to 2024-12-25, then multiply that by 3?", ["897"],
     :two_tool},
    {"t17", "Count the words in 'a b c d e f g' and then add 100.", ["107"], :two_tool},
    {"t18",
     "Take the gcd of 84 and 36, then convert that many kilometers to miles. Two decimals.",
     ["7.46"], :two_tool},
    {"t19", "Decode the Base64 'Zm9ydHk=' and then count its characters that are the letter o.",
     ["1"], :two_tool},
    {"t20", "Convert 100 USD to EUR using the exchange rate tool, then subtract 2.",
     ["90", "90.0"], :two_tool},
    {"t21", "Reverse the string 'level up' and then count its words.", ["2"], :two_tool},
    {"t22",
     "What is the date 30 days after 2024-02-01, and how many days is that from 2024-01-01?",
     ["2024-03-02", "61"], :two_tool},
    # no_tool (8): knowledge or rewriting, no tool should be needed
    {"t23", "Which planet is known as the Red Planet?", ["Mars"], :no_tool},
    {"t24", "What is the chemical symbol for gold?", ["Au"], :no_tool},
    {"t25", "Give the plural of 'mouse' (the animal).", ["mice"], :no_tool},
    {"t26", "Who wrote 'Pride and Prejudice'?", ["Jane Austen", "Austen"], :no_tool},
    {"t27", "Translate 'thank you' into Spanish.", ["gracias"], :no_tool},
    {"t28", "What is the past tense of 'run'?", ["ran"], :no_tool},
    {"t29", "Which language is Jido written in?", ["Elixir"], :no_tool},
    {"t30", "Rewrite 'we will not be attending' as a single word meaning the same.",
     ["absent", "declining", "no"], :no_tool}
  ]

  @spec all() :: [map()]
  def all do
    for {id, q, expect, kind} <- @tasks,
        do: %{id: id, q: q <> @suffix, expect: expect, kind: kind}
  end

  @spec normalize(String.t()) :: String.t()
  def normalize(s) do
    s
    |> String.replace(~r/[*_`#"“”'‘’]+/u, "")
    |> String.downcase()
    |> String.replace(~r/[\s$,.:]+/u, " ")
    |> String.trim()
  end

  @spec grade(%{expect: [String.t()]}, String.t()) :: boolean()
  def grade(%{expect: alts}, reply) when is_binary(reply) do
    r = normalize(reply)

    Enum.any?(alts, fn a ->
      n = normalize(a)
      n == r or String.contains?(" #{r} ", " #{n} ")
    end)
  end

  def grade(_task, _non_binary), do: false
end
