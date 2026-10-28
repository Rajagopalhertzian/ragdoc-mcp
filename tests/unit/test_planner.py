import pytest

import ragdoc.planner as planner
from ragdoc.errors import RagDocError
from ragdoc.planner import load_plan, plan_rag, profile_corpus
from tests.unit.test_parsers import _pdf_scanned


def test_profile_signals_on_mixed_technical_corpus(corpus, base):
    prof = profile_corpus(base / "corpus")
    assert {"tables", "equations", "mixed"} <= set(prof["signals"])
    assert prof["files_total"] == 32 and prof["languages"].get("en", 0) > 0.5
    assert prof["content_mix"]["code"] > 0 and prof["content_mix"]["table"] > 0 and prof["est_total_pages"] > 0


def test_profile_detects_scanned_pdf(base):
    (base / "p").mkdir()
    _pdf_scanned(base / "p" / "s.pdf")
    prof = profile_corpus(base / "p")
    assert prof["scanned_share"] == 0.5 and "scanned" in prof["signals"]


def test_plan_has_reasons_estimates_and_is_persisted(corpus, base, monkeypatch):
    monkeypatch.setattr(planner, "neural_available", lambda: False)
    res = plan_rag("corpus", collection="x")
    p = res["plan"]
    assert p["embedder"]["name"] == "hashing-384" and p["chunking"]["keep_tables_whole"] and p["chunking"]["keep_formulas_whole"]
    assert all({"choice", "value", "reason"} <= set(d) for d in p["decision_log"])
    assert p["estimates"]["est_chunks"] > 0 and p["estimates"]["measured"]["workers"] == 1 and p["requires_approval"] is False
    assert load_plan(p["id"])["collection"] == "x"
    assert any("offline hashing" in w for w in p["warnings"])


def test_neural_is_opt_in_and_never_downloads_while_planning(corpus, monkeypatch):
    monkeypatch.setattr(planner, "neural_available", lambda: True)

    def no_network(name):
        if name.startswith("hashing"):
            from ragdoc.embedders import HashingEmbedder

            return HashingEmbedder()
        raise RagDocError("offline", code="MODEL_UNAVAILABLE")

    monkeypatch.setattr(planner, "get_embedder", no_network)
    assert plan_rag("corpus")["plan"]["embedder"]["name"] == "hashing-384"  # default stays offline even if fastembed is installed
    assert plan_rag("corpus", {"embedder": "neural"})["plan"]["embedder"]["name"].startswith("BAAI/")
    monkeypatch.setattr(planner, "neural_available", lambda: False)
    p = plan_rag("corpus", {"embedder": "neural"})["plan"]
    assert p["embedder"]["name"] == "hashing-384" and any("fastembed" in w for w in p["warnings"])


def test_cloud_parser_requires_opt_in_key_and_hard_pdfs(base, monkeypatch):
    (base / "p").mkdir()
    _pdf_scanned(base / "p" / "s.pdf")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "fake-key-not-used")
    monkeypatch.setattr(planner, "docling_available", lambda: False)
    local = plan_rag("p")["plan"]
    assert local["parsers"][".pdf"] == "fast" and local["consent"] == {}  # local_only never goes to the cloud
    assert any("Docling is not installed" in w for w in local["warnings"])
    cloud = plan_rag("p", {"privacy": "cloud_ok"})["plan"]
    assert cloud["parsers"][".pdf"] == "llamaparse" and cloud["consent"]["cloud_parser"] and cloud["requires_approval"]
    assert cloud["estimates"]["cloud_cost"]["usd"] is None  # no invented prices
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY")
    assert plan_rag("p", {"privacy": "cloud_ok"})["plan"]["parsers"][".pdf"] == "fast"


def test_docling_chosen_for_hard_pdfs_when_available(base, monkeypatch):
    (base / "p").mkdir()
    _pdf_scanned(base / "p" / "s.pdf")
    monkeypatch.setattr(planner, "docling_available", lambda: True)
    assert plan_rag("p")["plan"]["parsers"][".pdf"] == "docling"


def test_bad_plan_ids_and_paths(base):
    for bad in ("../x", "plan_zzzz", "", "plan_" + "0" * 11):
        with pytest.raises(RagDocError):
            load_plan(bad)
    with pytest.raises(RagDocError):
        plan_rag("/etc")
