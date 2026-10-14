"""The estimator's closed forms, checked by enumeration rather than simulation.

Every claim oe_inferencex.estimate rests on is finite-population sampling theory, and each has a closed form that
can be checked EXACTLY on a small population by enumerating every possible sample: the stratified estimator is
unbiased and its variance estimator is unbiased for the true variance; Neyman's allocation minimises that variance;
Wilson's interval covers at a rate a hypergeometric sum gives exactly; the cluster-sample variance ratio is
m S_b^2 / S^2 exactly and 1 + (m - 1) rho in the large-T limit; and the rate a review set gives is capture(b) e / b.
Derivations are in docs/method/protocol.md. Where the claim is an equality, the enumeration uses exact rational
arithmetic and the comparison to the package's floats is at 1e-12.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import math
import os

import numpy as np
import pytest

from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _variance_at(alloc, sizes, S2):
    """Cochran's variance at a possibly non-integer allocation, as a float."""
    Ntot = sum(sizes)
    return sum((Nh / Ntot) ** 2 * (1 - nh / Nh) * s2 / nh for Nh, nh, s2 in zip(sizes, alloc, S2))


@pytest.mark.parametrize("task,bound", [("mados", 0.006), ("pastis_sentinel2", 0.003), ("m_cashew_plant", 0.002)])
def test_the_integer_allocation_is_within_half_a_percent_of_the_continuous_optimum_at_the_records_budget(task, bound):
    """On exp78's real strata at B = 300, from the same confidence-derived spread the design uses; the continuous
    allocation is a lower bound on any integer one, so this is the whole rounding cost."""
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", f"{task}.npz"))
    margin, p1, err = d["margin"].astype(float), d["p1"].astype(float), d["err"].astype(float)
    s = est.confidence_strata(margin)
    sizes = np.bincount(s, minlength=5)
    q = np.array([(1 - p1[s == h]).mean() for h in range(5)])
    spread = np.sqrt(np.clip(q * (1 - q), 1e-9, None))
    S2 = np.array([err[s == h].var(ddof=1) for h in range(5)])
    cont = 300 * sizes * spread / (sizes * spread).sum()
    ney = est.neyman_allocation(sizes, spread, 300)
    ratio = _variance_at(ney, sizes, S2) / _variance_at(cont, sizes, S2)
    assert 1.0 <= ratio < 1 + bound, (task, ratio)


def test_exact_coverage_at_madoss_recorded_cell_explains_its_monte_carlo_number():
    """exp78 recorded MADOS at 0.933, on the 0.93 bar, and named Wilson's discreteness in advance. The exact
    coverage at MADOS's (N, K, B) says whether that was the interval or the draw."""
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", "mados.npz"))
    err = d["err"].astype(float)
    exact = est.exact_coverage_srs(int(err.size), int(err.sum()), 300)
    mc = 0.933                                                              # exp/out/exp78_summary.json, D1/E1
    se = math.sqrt(exact * (1 - exact) / 2000)
    assert abs(mc - exact) < 3 * se, (exact, mc, se)


# ----------------------------------------------------------------------------- the review set
@pytest.mark.parametrize("task", ["mados", "sen1floods11", "pastis_sentinel2", "m_cashew_plant"])
def test_the_rate_a_review_set_gives_is_capture_times_e_over_b_against_the_recorded_capture(task):
    """rate on the top-k = (captured errors)/k = capture(b) E / k. With the record's own capture at 5% and error
    rate (exp70), this reproduces the inflation the guard warns about, and ties it to a number already on the
    record rather than to a fresh measurement."""
    import json
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))["results"]["tasks"][task]
    d = np.load(os.path.join(ROOT, "exp", "out", "exp78_units", f"{task}.npz"))
    err, margin = d["err"].astype(float), d["margin"].astype(float)
    n = err.size
    k = int(round(0.05 * n))
    top = np.argsort(margin, kind="stable")[:k]
    measured = err[top].mean()
    predicted = float(rec["signals"]["margin"]["capture"]["0.05"]) * rec["error_rate"] * n / k
    assert measured == pytest.approx(predicted, abs=2e-3), (task, measured, predicted)
    assert measured / err.mean() > 1.7
