"""Reproduce the fixture benchmark.

    python benchmarks/run_attribution.py --docs 60 --per-code 8 --verify-per-code 3

Builds the synthetic corpus and fault collections in a temporary directory, runs diagnose_failure on every labeled
case, and prints (and saves) the metrics. See docs/FIXTURES.md for how cases are labeled and what this does NOT prove."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--per-code", type=int, default=8)
    ap.add_argument("--verify-per-code", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).with_name("results_attribution.json")))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        os.environ["RAGDOC_BASE_DIR"] = td
        from ragdoc import __version__
        from ragdoc.eval.attribution import run_attribution
        from ragdoc.eval.fixtures import build_suite

        t0 = time.time()
        cases = build_suite(seed=a.seed, n_docs=a.docs, n_per_code=a.per_code)
        build_s = time.time() - t0
        rep = run_attribution(cases, verify_per_code=a.verify_per_code)
    rep["run"] = {"ragdoc_version": __version__, "seed": a.seed, "docs": a.docs, "per_code": a.per_code, "fixture_build_seconds": round(build_s, 1),
                  "python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count(),
                  "embedder": "hashing-384 (offline)", "data": "SYNTHETIC (ragdoc.eval.corpus); not real documents"}
    Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
