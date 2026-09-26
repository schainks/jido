defmodule JevRouting.ToolsTest do
  use ExUnit.Case, async: true
  alias JevRouting.Tools, as: T

  test "all/0 lists 24 unique action modules with unique names" do
    mods = T.all()
    assert length(mods) == 24
    names = Enum.map(mods, & &1.name())
    assert length(Enum.uniq(names)) == 24
  end

  test "convert_unit" do
    assert {:ok, %{result: 8.05}} = T.ConvertUnit.run(%{value: 5, from: "mi", to: "km"}, %{})
    assert {:ok, %{result: 37.0}} = T.ConvertUnit.run(%{value: 98.6, from: "f", to: "c"}, %{})
    assert {:error, _} = T.ConvertUnit.run(%{value: 1, from: "mi", to: "kg"}, %{})
  end

  test "date_add and date_diff" do
    assert {:ok, %{result: "2024-02-29"}} = T.DateAdd.run(%{date: "2024-02-28", days: 1}, %{})
    assert {:ok, %{result: 366}} = T.DateDiff.run(%{from: "2024-01-01", to: "2025-01-01"}, %{})
  end

  test "count_words and count_chars" do
    assert {:ok, %{result: 5}} = T.CountWords.run(%{text: "the quick brown fox jumps"}, %{})
    assert {:ok, %{result: 3}} = T.CountChars.run(%{text: "strawberry", char: "r"}, %{})
  end

  test "base64 round trip and reverse" do
    assert {:ok, %{result: "aGVsbG8gamlkbw=="}} = T.Base64Encode.run(%{text: "hello jido"}, %{})
    assert {:ok, %{result: "hello jido"}} = T.Base64Decode.run(%{text: "aGVsbG8gamlkbw=="}, %{})
    assert {:error, _} = T.Base64Decode.run(%{text: "not base64!"}, %{})
    assert {:ok, %{result: "odij"}} = T.ReverseString.run(%{text: "jido"}, %{})
  end

  test "gcd, sort_numbers, json_get, sha256" do
    assert {:ok, %{result: 21}} = T.Gcd.run(%{a: 1071, b: 462}, %{})
    assert {:ok, %{result: [3, 7, 19, 42]}} = T.SortNumbers.run(%{numbers: [42, 7, 19, 3]}, %{})

    assert {:ok, %{result: "x"}} =
             T.JsonGet.run(%{json: ~s({"a":{"b":[1,2,{"c":"x"}]}}), path: "a.b.2.c"}, %{})

    assert {:ok, %{result: "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"}} =
             T.Sha256.run(%{text: "hello"}, %{})
  end

  test "lookups and regex" do
    assert {:ok, %{result: "Canberra"}} = T.CapitalOf.run(%{country: "Australia"}, %{})
    assert {:error, _} = T.CapitalOf.run(%{country: "Atlantis"}, %{})
    assert {:ok, %{result: 0.92}} = T.ExchangeRate.run(%{from: "USD", to: "EUR"}, %{})
    assert {:ok, %{result: ["42", "7"]}} = T.RegexMatch.run(%{text: "a42b7", pattern: "\\d+"}, %{})
  end
end
