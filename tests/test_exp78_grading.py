"""exp78's verdict logic, under test before the job runs.

The precedent is commit ba9f009: exp77's grader broke ties by position, so seven tasks with identical numbers
would have reported a perfect rank correlation on no evidence, and the run shipped before anyone noticed. These
tests exercise the code that decides whether a prediction held, on constructed inputs, so a wrong verdict cannot
reach the record.
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("e78", os.path.join(ROOT, "exp", "exp78_error_rate_estimation.py"))
e78 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e78)          # module level is torch-free by design


def _rows(spec_):
    """spec_: {task: (error_rate, srs_cover, naive_cover, deff, base_half, best_half, best_cover)}"""
    out = {}
    for t, (er, srs, naive, deff, base, best, bcov) in spec_.items():
        out[t] = {"n_units": 10_000, "error_rate": er, "design_effect": deff, "budgets": {"300": {
            "D1/E1": {"coverage": srs, "half_width": base},
            "D4/E2_naive": {"coverage": naive, "half_width": base},
            "D2p/E1": {"coverage": bcov, "half_width": best},
            "D2c/E1": {"coverage": bcov, "half_width": best * 1.02},
            "D1+E3ps": {"coverage": bcov, "half_width": best * 1.05},
        }}}
    return out


def test_p1_fails_when_any_single_task_under_covers():
    ok = _rows({f"t{i}": (0.2, 0.95, 0.5, 4.0, 0.05, 0.045, 0.95) for i in range(7)})
    assert e78.verdicts(ok)["P1_the_interval_is_honest"]["holds"]
    bad = dict(ok)
    bad["t3"] = _rows({"t3": (0.2, 0.929, 0.5, 4.0, 0.05, 0.045, 0.95)})["t3"]
    v = e78.verdicts(bad)["P1_the_interval_is_honest"]
    assert not v["holds"] and v["tasks_failing"] == ["t3"], "P1 is the gate; one bad task fails it and is named"


def test_p2_needs_both_the_coverage_count_and_the_design_effect():
    """Either half alone must not carry it: a large design effect with adequate coverage is not the claim, and
    poor coverage with no clustering would mean something else is wrong."""
    good = _rows({f"t{i}": (0.2, 0.95, 0.70, 4.0, 0.05, 0.045, 0.95) for i in range(7)})
    assert e78.verdicts(good)["P2_the_naive_interval_is_badly_wrong"]["holds"]
    thin = _rows({f"t{i}": (0.2, 0.95, 0.70, 1.2, 0.05, 0.045, 0.95) for i in range(7)})
    assert not e78.verdicts(thin)["P2_the_naive_interval_is_badly_wrong"]["holds"], "no clustering, no claim"
    covered = _rows({f"t{i}": (0.2, 0.95, 0.93, 6.0, 0.05, 0.045, 0.95) for i in range(7)})
    assert not e78.verdicts(covered)["P2_the_naive_interval_is_badly_wrong"]["holds"], "coverage fine, no claim"


def test_p2_separates_piloted_tasks_from_out_of_sample_ones():
    """Three tasks informed the threshold, so they cannot also confirm it. The count must be reported both ways."""
    spec_ = {t: (0.2, 0.95, 0.70, 4.0, 0.05, 0.045, 0.95)
             for t in ("mados", "sen1floods11", "pastis_sentinel2", "m_cashew_plant", "m_sa_crop_type")}
    v = e78.verdicts(_rows(spec_))["P2_the_naive_interval_is_badly_wrong"]
    assert v["n"] == 5 and v["n_out_of_sample"] == 2
    assert set(v["out_of_sample_tasks"]) == {"m_cashew_plant", "m_sa_crop_type"}, \
        "only the tasks the pilot never saw may count as confirmation"


def test_p3_needs_the_pattern_and_not_merely_a_narrow_interval():
    """The prediction is that the gain concentrates on the good maps. A uniform gain everywhere must NOT pass,
    because it would not distinguish the stated mechanism from anything else."""
    patterned = _rows({"good1": (0.05, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.80, 0.95),
                       "good2": (0.08, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.85, 0.95),
                       "mid": (0.19, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.90, 0.95),
                       "bad1": (0.34, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.96, 0.95)})
    assert e78.verdicts(patterned)["P3_confidence_helps_most_where_the_map_is_good"]["holds"]
    uniform = _rows({"good1": (0.05, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.90, 0.95),
                     "good2": (0.08, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.90, 0.95),
                     "bad1": (0.34, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.90, 0.95)})
    assert not e78.verdicts(uniform)["P3_confidence_helps_most_where_the_map_is_good"]["holds"], \
        "a uniform gain is not the predicted pattern"


def test_p3_refuses_a_narrow_interval_bought_by_breaking_coverage():
    cheating = _rows({"good1": (0.05, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.50, 0.70),
                      "good2": (0.08, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.50, 0.70),
                      "bad1": (0.34, 0.95, 0.5, 4.0, 0.05, 0.05 * 0.96, 0.70)})
    assert not e78.verdicts(cheating)["P3_confidence_helps_most_where_the_map_is_good"]["holds"], \
        "width bought by under-covering is not a gain"


def test_p4_catches_a_saving_that_breaks_the_design_bound():
    fine = _rows({f"t{i}": (0.25, 0.95, 0.5, 4.0, 0.06, 0.06 * 0.90, 0.95) for i in range(7)})
    assert e78.verdicts(fine)["P4_no_order_of_magnitude"]["holds"]
    impossible = _rows({f"t{i}": (0.25, 0.95, 0.5, 4.0, 0.06, 0.06 * 0.50, 0.95) for i in range(7)})
    v = e78.verdicts(impossible)["P4_no_order_of_magnitude"]
    assert not v["holds"] and v["max_budget_saving"] == pytest.approx(4.0), \
        "a 2x narrower interval is a 4x budget saving, past the design bound; that means an assumption broke"


def test_the_sampling_machinery_is_honest_on_data_built_to_be_easy():
    """An end-to-end check of the estimator itself: on independent units with a known rate, the design-based
    interval must cover near its nominal level."""
    rng = np.random.default_rng(0)
    n = 20_000
    err = (rng.random(n) < 0.2).astype(float)
    theta = err.mean()
    margin = rng.random(n) - 0.6 * err
    p1 = 0.5 + margin / 4
    out = e78.run_designs(err, margin, p1, None, 300, np.random.default_rng(1), theta)
    assert 0.92 <= out["D1/E1"]["coverage"] <= 0.98, out["D1/E1"]
    assert 0.90 <= out["D2p/E1"]["coverage"] <= 0.99, "stratified must also cover"
    assert out["D2p/E1"]["half_width"] < out["D1/E1"]["half_width"], "a good ranker should narrow the interval"


def test_the_design_effect_is_one_without_clustering_and_large_with_it():
    rng = np.random.default_rng(2)
    n_tiles, per = 400, 16
    tile = np.repeat(np.arange(n_tiles), per)
    flat = (rng.random(n_tiles * per) < 0.2).astype(float)
    assert e78.design_effect(flat, tile) == pytest.approx(1.0, abs=0.25)
    rate = np.clip(0.2 + rng.normal(0, 0.18, n_tiles), 0.01, 0.99).repeat(per)
    clumped = (rng.random(n_tiles * per) < rate).astype(float)
    assert e78.design_effect(clumped, tile) > 3.0, "clustered errors must show a large design effect"
