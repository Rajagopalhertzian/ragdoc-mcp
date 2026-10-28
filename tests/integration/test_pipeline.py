import os

import pytest

from ragdoc.errors import RagDocError
from ragdoc.retrieval.pipeline import Pipeline


def test_build_and_search_has_citations(corpus, builder):
    _, facts = corpus
    coll, rep, _ = builder()
    assert rep["files_total"] == 32 and rep["chunks"] > 60
    from ragdoc.eval.corpus import case_for

    c = case_for(facts[2], "table")
    tr = Pipeline(coll).run(c["question"], k=3)
    top = tr.packed[0]
    assert top["file"] == c["expected_source"] and top["ctype"] == "table" and top["heading_path"]


def test_resume_incremental_and_removal(corpus, builder, base):
    _, facts = corpus
    coll, r1, plan = builder()
    from ragdoc.build import build_collection

    r2 = build_collection(plan)
    assert r2["files_indexed"] == 0 and r2["files_unchanged"] == 32
    p = base / "corpus" / facts[0]["md"]
    p.write_text(p.read_text() + "\n## Extra\nA brand new paragraph about gaskets.\n")
    r3 = build_collection(plan)
    assert r3["files_indexed"] == 1 and r3["files_unchanged"] == 31
    (base / "corpus" / facts[1]["py"]).unlink()
    r4 = build_collection(plan)
    assert r4["files_removed"] == 1
    from ragdoc.collection import Collection

    assert not any(r["file"] == facts[1]["py"] for r in Collection.open("demo").read_rows())


def test_symlink_outside_base_is_skipped(corpus, builder, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\nTop secret launch codes 0000.")
    os.symlink(outside, corpus[0] / "docs" / "leak.md")
    coll, rep, _ = builder()
    assert any("outside base" in s["reason"] for s in rep["skipped"])
    assert not any("launch codes" in r["text"] for r in coll.read_rows())


def test_filters(corpus, builder):
    coll, _, _ = builder()
    p = Pipeline(coll)
    tr = p.run("rated current", k=5, filters={"ctype": "table"})
    assert tr.packed and all(c["ctype"] == "table" for c in tr.packed)
    tr = p.run("maximum operating temperature", k=5, filters={"file": "docs/pump"})
    assert tr.packed and all("pump" in c["file"] for c in tr.packed)
    assert p.run("x", filters={"file": "nonexistent_zzz"}).notes == ["filters matched no chunks"]
    with pytest.raises(RagDocError) as e:
        p.run("x", filters={"sql": "1=1"})
    assert e.value.code == "BAD_FILTER"


def test_token_budget_truncates_and_notes(corpus, builder):
    coll, _, _ = builder()
    tr = Pipeline(coll).run("maximum operating temperature", k=8, max_tokens=60)
    assert sum(c["packed_tokens"] for c in tr.packed) <= 60 and tr.notes


def test_empty_corpus_rejected(base):
    (base / "x").mkdir()
    (base / "x" / "a.xyz").write_text("nope")
    from ragdoc.planner import plan_rag

    with pytest.raises(RagDocError) as e:
        plan_rag("x")
    assert e.value.code == "EMPTY_CORPUS"
