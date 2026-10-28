from pathlib import Path

import pytest


@pytest.fixture()
def base(tmp_path, monkeypatch) -> Path:
    b = tmp_path / "base"
    b.mkdir()
    monkeypatch.setenv("RAGDOC_BASE_DIR", str(b))
    return b


@pytest.fixture()
def corpus(base):
    from ragdoc.eval.corpus import make_corpus

    facts = make_corpus(base / "corpus", seed=7, n_docs=16)
    return base / "corpus", facts


def build(base: Path, name: str = "demo", src: str = "corpus", **ov):
    from ragdoc.build import build_collection
    from ragdoc.collection import Collection
    from ragdoc.planner.plans import default_plan

    plan = default_plan(str(base / src), name, **ov)
    rep = build_collection(plan)
    return Collection.open(name), rep, plan


@pytest.fixture()
def builder(base):
    return lambda name="demo", src="corpus", **ov: build(base, name, src, **ov)
