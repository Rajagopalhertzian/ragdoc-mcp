from ragdoc.envelope import est_tokens, fit_budget, ok, shape


def test_small_unchanged():
    e = ok("s", {"a": 1}, "n")
    assert fit_budget(e, 2000) == e


def test_large_is_shortened_but_core_fields_kept():
    e = ok("summary", {"items": [{"text": "x" * 2000} for _ in range(40)]}, "next")
    out = fit_budget(e, 2000)
    assert est_tokens(out) <= 2300 and out["truncated"] and out["summary"] == "summary" and out["next_step"] == "next"


def test_protected_text_survives_first_pass():
    e = ok("s", {"chunks": [{"text": "keep " * 120, "meta": "m" * 3000} for _ in range(3)]}, "n")
    out = fit_budget(e, 2000, protect=("text",))
    assert out["data"]["chunks"][0]["text"] == "keep " * 120


def test_full_detail_larger_budget():
    e = ok("s", {"items": ["y" * 400 for _ in range(100)]}, "n")
    assert est_tokens(shape(e, "full")) > est_tokens(shape(e, "brief"))
