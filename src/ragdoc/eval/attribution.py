"""Score diagnose_failure (and verify_fix) against the labeled fixture suite."""
from __future__ import annotations

import time
from collections import Counter, defaultdict

from ..collection import Collection
from ..diagnose import diagnose
from .verify import verify_fix


def run_attribution(cases: list[dict], verify_per_code: int = 0, regression_sample: int = 6) -> dict:
    rows, t_ms = [], []
    for c in cases:
        t0 = time.perf_counter()
        d = diagnose(Collection.open(c["collection"]), c["question"], c["expected_answer"], c["expected_source"])
        t_ms.append((time.perf_counter() - t0) * 1000)
        rows.append({"case": c, "diag": d})

    n = len(rows)
    stage_ok = [r["diag"]["first_failing_stage"] == r["case"]["true_stage"] for r in rows]
    code_ok = [r["diag"]["failure_code"] == r["case"]["true_code"] for r in rows]
    parse_rows = [r for r in rows if r["case"]["true_code"] == "PARSE_LOSS"]
    sub_ok = [r["diag"].get("subcause") == r["case"]["true_subcause"] for r in parse_rows]
    controls = [r for r in rows if r["case"]["true_code"] == "GENERATION_SIDE"]
    fixable = [r for r in rows if r["case"]["true_code"] != "GENERATION_SIDE"]
    false_blame = [r for r in controls if r["diag"]["first_failing_stage"] != "downstream"]
    wrong_on_fixable = [r for r in fixable if r["diag"]["first_failing_stage"] == "downstream"]  # told "not a retrieval problem" when it was

    per_code: dict[str, list[bool]] = defaultdict(list)
    for r, ok_ in zip(rows, code_ok):
        per_code[r["case"]["true_code"]].append(ok_)
    confusion = Counter((r["case"]["true_code"], r["diag"]["failure_code"]) for r in rows if r["diag"]["failure_code"] != r["case"]["true_code"])
    conf_right = [r["diag"]["confidence"] for r, ok_ in zip(rows, stage_ok) if ok_]
    conf_wrong = [r["diag"]["confidence"] for r, ok_ in zip(rows, stage_ok) if not ok_]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731

    report = {
        "n_cases": n,
        "stage_attribution_accuracy": round(sum(stage_ok) / n, 3) if n else None,
        "failure_code_accuracy": round(sum(code_ok) / n, 3) if n else None,
        "subcause_accuracy_parse": round(sum(sub_ok) / len(sub_ok), 3) if sub_ok else None,
        "false_blame_rate_on_controls": round(len(false_blame) / len(controls), 3) if controls else None,
        "missed_failures_called_generation_side": len(wrong_on_fixable),
        "per_code": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for k, v in sorted(per_code.items())},
        "confusion": [{"true": t, "diagnosed": g, "count": c} for (t, g), c in confusion.most_common(12)],
        "mean_confidence_when_right": mean(conf_right), "mean_confidence_when_wrong": mean(conf_wrong),
        "mean_ms_per_diagnosis": round(sum(t_ms) / n, 1) if n else None,
        "mismatches": [{"id": r["case"]["id"], "true": r["case"]["true_code"], "got": r["diag"]["failure_code"], "sub": r["diag"].get("subcause")}
                       for r in rows if r["diag"]["failure_code"] != r["case"]["true_code"]][:15],
    }

    if verify_per_code:
        by_code: dict[str, list[dict]] = defaultdict(list)
        for r in fixable:
            by_code[r["case"]["true_code"]].append(r)
        ver = []
        for code, rs in sorted(by_code.items()):
            for r in rs[:verify_per_code]:
                fixes = [f for f in r["diag"]["fixes"] if f["type"] != "advice" and f.get("verifiable", True)]
                if not fixes:
                    ver.append({"code": code, "id": r["case"]["id"], "verdict": "no_fix_offered"})
                    continue
                c = r["case"]
                case = {k: v for k, v in c.items() if k in ("question", "expected_answer", "expected_source", "ctype")}
                try:
                    v = verify_fix(Collection.open(c["collection"]), fixes[0], [case], regression_sample=regression_sample)
                    ver.append({"code": code, "id": c["id"], "fix": fixes[0]["id"], "verdict": v["verdict"],
                                "regressions": v.get("regression_check", {}).get("regressions")})
                except Exception as e:  # noqa: BLE001
                    ver.append({"code": code, "id": c["id"], "fix": fixes[0]["id"], "verdict": "error", "error": str(e)[:100]})
        good = [v for v in ver if v["verdict"] == "improves_no_regression"]
        report["verify_fix"] = {
            "n": len(ver), "improves_no_regression": len(good),
            "success_rate": round(len(good) / len(ver), 3) if ver else None,
            "by_verdict": dict(Counter(v["verdict"] for v in ver)),
            "failures": [v for v in ver if v["verdict"] != "improves_no_regression"][:12],
            "note": "End to end: the FIRST fix suggested by diagnose_failure is applied regardless of whether the diagnosis was right.",
        }
    return report
