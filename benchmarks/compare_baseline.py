"""plan_rag vs a naive baseline (fixed 512-token windows, dense only, no rerank) on the SAME questions.

    python benchmarks/compare_baseline.py --docs 60 --per-type 40

Gate from the product spec: plan_rag must beat the baseline on at least 3 of 4 content types, otherwise do not
claim it helps. Synthetic corpus + keyword-style extractive questions: optimistic. Re-run on your own documents."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=60)
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).with_name("results_baseline.json")))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        os.environ["RAGDOC_BASE_DIR"] = td
        from ragdoc.build import build_collection
        from ragdoc.collection import Collection
        from ragdoc.eval import domain_eval, generate_questions, reference_rows
        from ragdoc.eval.corpus import make_corpus
        from ragdoc.planner import baseline_plan, load_plan, plan_rag

        make_corpus(Path(td) / "corpus", seed=a.seed, n_docs=a.docs)
        res = plan_rag("corpus", collection="bench_plan")
        plan = load_plan(res["plan"]["id"])
        build_collection(plan)
        build_collection(baseline_plan(str(Path(td) / "corpus"), "bench_base"))
        cp, cb = Collection.open("bench_plan"), Collection.open("bench_base")
        qs = generate_questions(reference_rows(cp), None, a.per_type, seed=3)
        rp = domain_eval(cp, questions=qs, k=a.k, diagnose_failures=0)
        rb = domain_eval(cb, questions=qs, k=a.k, diagnose_failures=0)
    wins, table = 0, {}
    for t in ("prose", "table", "equation", "code"):
        p, b = rp["by_type"].get(t), rb["by_type"].get(t)
        if not p or not b:
            continue
        win = (p["hit_at_k"], p["mrr"]) > (b["hit_at_k"], b["mrr"])
        wins += win
        table[t] = {"n": p["n"], "plan_hit@k": p["hit_at_k"], "base_hit@k": b["hit_at_k"], "plan_mrr": p["mrr"], "base_mrr": b["mrr"], "plan_wins": win}
    out = {"types_won": wins, "types_total": len(table), "gate_passed": wins >= 3, "by_type": table, "plan_overall": rp["overall"], "baseline_overall": rb["overall"],
           "plan": {"chunking": plan["chunking"], "retrieval": plan["retrieval"], "embedder": plan["embedder"]},
           "caveat": "SYNTHETIC corpus; keyword-style extractive questions; offline hashing embedder. Not evidence about real documents."}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
