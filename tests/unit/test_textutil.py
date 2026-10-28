from ragdoc.textutil import count_tokens, find_answer, tokens


def test_exact_and_numbers_exact():
    assert find_answer("temperature of 85 degrees", "a temperature of 85 degrees C")["mode"] == "exact"
    assert not find_answer("temperature of 85 degrees", "a temperature of 855 degrees C")["found"]
    assert not find_answer("current 3.5 A", "current 35 A")["found"]


def test_decimal_kept_together():
    assert "85.5" in tokens("limit 85.5 mm")


def test_fuzzy_and_scattered():
    assert find_answer("rated current 3.5 A", "Rated 3.5 current A")["found"]
    res = find_answer("rated current 3.5 A", "Rated current response time value 24 3.5 12 unit V A ms")
    assert res["mode"] == "scattered" and not res["found"]


def test_none():
    assert find_answer("alpha beta", "x y z")["mode"] == "none"
    assert not find_answer("", "text")["found"]


def test_count_tokens_includes_punctuation():
    assert count_tokens("Hello, world!") == 4
