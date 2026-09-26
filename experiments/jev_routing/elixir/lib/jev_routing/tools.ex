defmodule JevRouting.Tools do
  @moduledoc """
  Pure tools for the bench: 9 built-ins from jido_action plus 15 synthetic actions.
  Confusable pairs are deliberate: add/increment_action, subtract/decrement_action,
  date_add/date_diff, count_words/count_chars, base64_encode/base64_decode.
  """

  @builtins [
    Jido.Tools.Arithmetic.Add,
    Jido.Tools.Arithmetic.Subtract,
    Jido.Tools.Arithmetic.Multiply,
    Jido.Tools.Arithmetic.Divide,
    Jido.Tools.Arithmetic.Square,
    Jido.Tools.Basic.Increment,
    Jido.Tools.Basic.Decrement,
    Jido.Tools.Basic.Noop,
    Jido.Tools.Basic.Today
  ]

  def builtins, do: @builtins

  def synthetic do
    [
      __MODULE__.ConvertUnit,
      __MODULE__.DateAdd,
      __MODULE__.DateDiff,
      __MODULE__.CountWords,
      __MODULE__.CountChars,
      __MODULE__.Base64Encode,
      __MODULE__.Base64Decode,
      __MODULE__.ReverseString,
      __MODULE__.Gcd,
      __MODULE__.SortNumbers,
      __MODULE__.JsonGet,
      __MODULE__.Sha256,
      __MODULE__.CapitalOf,
      __MODULE__.ExchangeRate,
      __MODULE__.RegexMatch
    ]
  end

  def all, do: @builtins ++ synthetic()

  @doc "Coerces a number that an LLM may have passed as a string. Returns {:ok, number} | :error."
  def num(n) when is_number(n), do: {:ok, n}

  def num(s) when is_binary(s) do
    case Integer.parse(s) do
      {i, ""} ->
        {:ok, i}

      _ ->
        case Float.parse(s) do
          {f, ""} -> {:ok, f}
          _ -> :error
        end
    end
  end

  def num(_), do: :error

  defmodule ConvertUnit do
    @moduledoc false
    use Jido.Action,
      name: "convert_unit",
      description: "Converts a quantity between units: mi/km, kg/lb, c/f, m/ft",
      schema: [
        value: [type: :any, required: true, doc: "Quantity to convert (number)"],
        from: [type: :string, required: true, doc: "Source unit: mi, km, kg, lb, c, f, m, ft"],
        to: [type: :string, required: true, doc: "Target unit"]
      ]

    @factors %{
      {"mi", "km"} => 1.609344,
      {"km", "mi"} => 0.621371,
      {"kg", "lb"} => 2.20462,
      {"lb", "kg"} => 0.453592,
      {"m", "ft"} => 3.28084,
      {"ft", "m"} => 0.3048
    }

    def run(%{value: raw, from: from, to: to}, _ctx) do
      key = {String.downcase(from), String.downcase(to)}

      case {JevRouting.Tools.num(raw), key} do
        {:error, _} -> {:error, "value must be a number, got #{inspect(raw)}"}
        {{:ok, v}, key} -> convert(v, key, from, to)
      end
    end

    defp convert(v, key, from, to) do
      case key do
        {"c", "f"} ->
          {:ok, %{result: Float.round(v * 9 / 5 + 32, 2)}}

        {"f", "c"} ->
          {:ok, %{result: Float.round((v - 32) * 5 / 9, 2)}}

        _ ->
          case Map.fetch(@factors, key) do
            {:ok, f} -> {:ok, %{result: Float.round(v * f, 2)}}
            :error -> {:error, "unsupported conversion #{from} -> #{to}"}
          end
      end
    end
  end

  defmodule DateAdd do
    @moduledoc false
    use Jido.Action,
      name: "date_add",
      description: "Adds a number of days to an ISO 8601 date and returns the new date",
      schema: [
        date: [type: :string, required: true, doc: "ISO date, e.g. 2024-02-28"],
        days: [type: :integer, required: true, doc: "Days to add (negative to subtract)"]
      ]

    def run(%{date: d, days: n}, _ctx) do
      with {:ok, date} <- Date.from_iso8601(d),
           do: {:ok, %{result: Date.to_iso8601(Date.add(date, n))}}
    end
  end

  defmodule DateDiff do
    @moduledoc false
    use Jido.Action,
      name: "date_diff",
      description: "Number of days between two ISO 8601 dates (to minus from)",
      schema: [
        from: [type: :string, required: true, doc: "Start ISO date"],
        to: [type: :string, required: true, doc: "End ISO date"]
      ]

    def run(%{from: f, to: t}, _ctx) do
      with {:ok, a} <- Date.from_iso8601(f),
           {:ok, b} <- Date.from_iso8601(t),
           do: {:ok, %{result: Date.diff(b, a)}}
    end
  end

  defmodule CountWords do
    @moduledoc false
    use Jido.Action,
      name: "count_words",
      description: "Counts whitespace-separated words in a text",
      schema: [text: [type: :string, required: true, doc: "Text to count"]]

    def run(%{text: t}, _ctx), do: {:ok, %{result: t |> String.split() |> length()}}
  end

  defmodule CountChars do
    @moduledoc false
    use Jido.Action,
      name: "count_chars",
      description: "Counts how many times a single character occurs in a text",
      schema: [
        text: [type: :string, required: true, doc: "Text to search"],
        char: [type: :string, required: true, doc: "Single character to count"]
      ]

    def run(%{text: t, char: c}, _ctx),
      do: {:ok, %{result: t |> String.graphemes() |> Enum.count(&(&1 == c))}}
  end

  defmodule Base64Encode do
    @moduledoc false
    use Jido.Action,
      name: "base64_encode",
      description: "Encodes text as standard Base64",
      schema: [text: [type: :string, required: true, doc: "Plain text"]]

    def run(%{text: t}, _ctx), do: {:ok, %{result: Base.encode64(t)}}
  end

  defmodule Base64Decode do
    @moduledoc false
    use Jido.Action,
      name: "base64_decode",
      description: "Decodes standard Base64 text back to plain text",
      schema: [text: [type: :string, required: true, doc: "Base64 text"]]

    def run(%{text: t}, _ctx) do
      case Base.decode64(t) do
        {:ok, s} -> {:ok, %{result: s}}
        :error -> {:error, "invalid base64"}
      end
    end
  end

  defmodule ReverseString do
    @moduledoc false
    use Jido.Action,
      name: "reverse_string",
      description: "Reverses the characters of a string",
      schema: [text: [type: :string, required: true, doc: "Text to reverse"]]

    def run(%{text: t}, _ctx), do: {:ok, %{result: String.reverse(t)}}
  end

  defmodule Gcd do
    @moduledoc false
    use Jido.Action,
      name: "gcd",
      description: "Greatest common divisor of two integers",
      schema: [
        a: [type: :integer, required: true, doc: "First integer"],
        b: [type: :integer, required: true, doc: "Second integer"]
      ]

    def run(%{a: a, b: b}, _ctx), do: {:ok, %{result: Integer.gcd(a, b)}}
  end

  defmodule SortNumbers do
    @moduledoc false
    use Jido.Action,
      name: "sort_numbers",
      description: "Sorts a list of numbers in ascending order",
      schema: [
        numbers: [type: {:list, :any}, required: true, doc: "Numbers to sort"]
      ]

    def run(%{numbers: ns}, _ctx) do
      coerced = Enum.map(ns, &JevRouting.Tools.num/1)

      if Enum.all?(coerced, &match?({:ok, _}, &1)),
        do: {:ok, %{result: coerced |> Enum.map(fn {:ok, n} -> n end) |> Enum.sort()}},
        else: {:error, "numbers must all be numeric"}
    end
  end

  defmodule JsonGet do
    @moduledoc false
    use Jido.Action,
      name: "json_get",
      description:
        "Reads a value from a JSON document by dotted path; list indexes are numbers, e.g. a.b.2.c",
      schema: [
        json: [type: :string, required: true, doc: "JSON text"],
        path: [type: :string, required: true, doc: "Dotted path"]
      ]

    def run(%{json: j, path: p}, _ctx) do
      with {:ok, doc} <- Jason.decode(j) do
        value =
          p
          |> String.split(".")
          |> Enum.reduce(doc, fn
            _k, nil ->
              nil

            k, list when is_list(list) ->
              case Integer.parse(k) do
                {i, ""} -> Enum.at(list, i)
                _ -> nil
              end

            k, map when is_map(map) ->
              Map.get(map, k)

            _k, _ ->
              nil
          end)

        {:ok, %{result: value}}
      end
    end
  end

  defmodule Sha256 do
    @moduledoc false
    use Jido.Action,
      name: "sha256",
      description: "Hex-encoded SHA-256 digest of a text",
      schema: [text: [type: :string, required: true, doc: "Text to hash"]]

    def run(%{text: t}, _ctx),
      do: {:ok, %{result: :crypto.hash(:sha256, t) |> Base.encode16(case: :lower)}}
  end

  defmodule CapitalOf do
    @moduledoc false
    use Jido.Action,
      name: "capital_of",
      description: "Capital city of a country from a small static table",
      schema: [country: [type: :string, required: true, doc: "Country name in English"]]

    @table %{
      "australia" => "Canberra",
      "canada" => "Ottawa",
      "brazil" => "Brasília",
      "japan" => "Tokyo",
      "germany" => "Berlin",
      "france" => "Paris",
      "india" => "New Delhi",
      "kenya" => "Nairobi",
      "mexico" => "Mexico City",
      "norway" => "Oslo",
      "turkey" => "Ankara",
      "switzerland" => "Bern",
      "nigeria" => "Abuja",
      "argentina" => "Buenos Aires",
      "egypt" => "Cairo",
      "poland" => "Warsaw",
      "vietnam" => "Hanoi",
      "new zealand" => "Wellington",
      "south africa" => "Pretoria",
      "chile" => "Santiago"
    }

    def run(%{country: c}, _ctx) do
      case Map.fetch(@table, String.downcase(String.trim(c))) do
        {:ok, cap} -> {:ok, %{result: cap}}
        :error -> {:error, "unknown country #{c}"}
      end
    end
  end

  defmodule ExchangeRate do
    @moduledoc false
    use Jido.Action,
      name: "exchange_rate",
      description: "Static exchange rate between two currency codes (USD, EUR, GBP, JPY)",
      schema: [
        from: [type: :string, required: true, doc: "Source currency code"],
        to: [type: :string, required: true, doc: "Target currency code"]
      ]

    @rates %{
      {"USD", "EUR"} => 0.92,
      {"EUR", "USD"} => 1.09,
      {"USD", "GBP"} => 0.79,
      {"GBP", "USD"} => 1.27,
      {"USD", "JPY"} => 150.0,
      {"JPY", "USD"} => 0.0067
    }

    def run(%{from: f, to: t}, _ctx) do
      case Map.fetch(@rates, {String.upcase(f), String.upcase(t)}) do
        {:ok, r} -> {:ok, %{result: r}}
        :error -> {:error, "no rate for #{f}->#{t}"}
      end
    end
  end

  defmodule RegexMatch do
    @moduledoc false
    use Jido.Action,
      name: "regex_match",
      description: "All non-overlapping matches of a regular expression in a text",
      schema: [
        text: [type: :string, required: true, doc: "Text to scan"],
        pattern: [type: :string, required: true, doc: "Regular expression"]
      ]

    def run(%{text: t, pattern: p}, _ctx) do
      with {:ok, re} <- Regex.compile(p),
           do: {:ok, %{result: Regex.scan(re, t) |> Enum.map(&hd/1)}}
    end
  end
end
