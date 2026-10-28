import json

import pytest

from ragdoc.errors import RagDocError
from ragdoc.eval import evaluate_cases
from ragdoc.eval.corpus import case_for
from ragdoc.eval.verify import verify_fix
from ragdoc.retrieval.pipeline import Pipeline


def failing_cases(coll, facts, ctype="prose", n=4):
    p = Pipeline(coll)
    out = [c for c in (case_for(f, ctype) for f in facts) if not evaluate_cases(p, [c], 8)[0]["hit"]]
    if len(out) < 2:
        pytest.skip("not enough failing cases")
    return out[:n]


@pytest.fixture()
def broken(corpus, builder):
    coll, _, _ = builder("ctx", chunking={"prepend_heading": False})
    return coll, failing_cases(coll, corpus[1])


def snapshot(coll):
    return coll.manifest_path.read_text(), coll.rows_path.read_bytes()


def test_good_fix_improves_without_regression_and_leaves_live_untouched(broken, base):
    coll, cases = broken
    before = snapshot(coll)
    v = verify_fix(coll, {"type": "rechunk", "chunking": {"prepend_heading": True}}, cases, apply=False)
    assert v["verdict"] == "improves_no_regression" and v["target_cases"]["fixed"] == len(cases) and not v["applied"]
    assert snapshot(coll) == before  # apply=false must not change the collection
    assert not list((base / ".ragdoc" / "shadow").glob("ctx-*"))  # shadow cleaned up


def test_harmful_fix_is_flagged_and_refused_even_with_apply(corpus, builder):
    coll, _, _ = builder("good")
    # a "fix" for a case that already works, that makes everything worse
    c = case_for(corpus[1][1], "prose")
    before = snapshot(coll)
    v = verify_fix(coll, {"type": "rechunk", "chunking": {"kind": "window", "max_tokens": 8, "overlap_pct": 0}}, [c], apply=True)
    assert v["verdict"] in ("worse", "no_improvement") and not v["applied"]
    assert snapshot(coll) == before


def test_apply_promotes_and_keeps_backup(broken):
    coll, cases = broken
    v = verify_fix(coll, {"type": "rechunk", "chunking": {"prepend_heading": True}}, cases, apply=True)
    assert v["applied"] and v["backup"]
    assert json.loads(coll.manifest_path.read_text())["plan"]["chunking"]["prepend_heading"] is True
    assert all(r["hit"] for r in evaluate_cases(Pipeline(coll), cases, 8))


def test_regression_set_ignores_sources_edited_on_disk(corpus, builder, base):
    coll, _, _ = builder()
    f = corpus[1][0]
    p = base / "corpus" / f["md"]
    p.write_text(p.read_text().replace(f"temperature of {f['t_max']} degrees", f"temperature of {f['t_max'] + 3} degrees"))
    c = case_for({**f, "t_max": f["t_max"] + 3}, "prose")
    v = verify_fix(coll, {"type": "reindex", "files": [f["md"]]}, [c])
    assert v["verdict"] == "improves_no_regression"


@pytest.mark.parametrize("fix", [
    {"type": "rm -rf"}, {"type": "rechunk", "chunking": {"evil": 1}}, {"type": "rechunk", "chunking": {"max_tokens": "big"}},
    {"type": "rechunk", "chunking": {"max_tokens": 9999999}}, {"type": "retrieval_config", "retrieval": {"candidates": 0}},
    {"type": "pack_config", "pack": {"max_tokens": 1}}, {"type": "reparse", "backend": "shell", "files": ["a"]},
])
def test_bad_fix_payloads_rejected(corpus, builder, fix):
    coll, _, _ = builder()
    with pytest.raises(RagDocError) as e:
        verify_fix(coll, fix, [case_for(corpus[1][0], "prose")])
    assert e.value.code == "BAD_FIX"


def test_unavailable_backend_reported_not_faked(corpus, builder):
    from ragdoc.parsers import docling_available

    if docling_available():
        pytest.skip("docling installed")
    coll, _, _ = builder()
    f = corpus[1][0]
    v = verify_fix(coll, {"type": "reparse", "backend": "docling", "files": [f["md"]]}, [case_for(f, "prose")])
    assert v["verdict"] == "fix_unavailable" and "Docling" in v["reason"] and not v["applied"]


def test_query_rewrite_is_verifiable_but_never_applied(corpus, builder):
    coll, _, _ = builder()
    c = {"question": "zzz", "expected_answer": case_for(corpus[1][1], "prose")["expected_answer"], "expected_source": corpus[1][1]["md"]}
    fix = {"type": "query_rewrite", "suggested_query": case_for(corpus[1][1], "prose")["question"]}
    v = verify_fix(coll, fix, [c], apply=True)
    assert v["target_cases"]["fixed"] == 1 and not v["applied"] and "nothing to apply" in v["apply_refused"]


def test_needs_cases(corpus, builder):
    coll, _, _ = builder()
    with pytest.raises(RagDocError):
        verify_fix(coll, {"type": "reindex", "files": []}, [])
