"""Guards diagnosis quality on the synthetic fixture suite (small and fast version of benchmarks/run_attribution.py)."""
from ragdoc.eval.attribution import run_attribution
from ragdoc.eval.fixtures import build_suite


def test_attribution_accuracy_and_no_false_blame(base):
    cases = build_suite(seed=7, n_docs=24, n_per_code=3)
    assert len({c["family"] for c in cases}) >= 11  # most fault families produced cases
    rep = run_attribution(cases, verify_per_code=1, regression_sample=4)
    assert rep["stage_attribution_accuracy"] >= 0.85, rep["mismatches"]
    assert rep["false_blame_rate_on_controls"] == 0.0
    assert rep["missed_failures_called_generation_side"] == 0
    assert rep["subcause_accuracy_parse"] == 1.0
    assert rep["verify_fix"]["success_rate"] >= 0.7, rep["verify_fix"]["failures"]
