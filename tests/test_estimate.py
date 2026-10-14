"""oe_inferencex.estimate: the intervals against enumeration, the designs against their own definitions, and
the guard against the one thing a user must not do — on exp78's committed real units, not on synthetic data.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json
import math
import os

import numpy as np
import pytest

import oe_inferencex as ox
from oe_inferencex import estimate as est

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITS = os.path.join(ROOT, "exp", "out", "exp78_units")


def _units(task):
    d = np.load(os.path.join(UNITS, f"{task}.npz"), allow_pickle=False)
    ok = np.unpackbits(d["ok_packed"])[: int(d["ok_len"][0])].astype(bool)
    N, hw, ww = (int(v) for v in d["grid"])
    tile = np.broadcast_to(np.arange(N)[:, None, None], (N, hw, ww)).reshape(-1)[ok]
    return d["margin"].astype(float), d["p1"].astype(float), d["err"].astype(float), tile


def test_the_confidence_design_oversamples_the_least_confident_stratum_on_a_real_map():
    """That is what buys the width: on MADOS exp78 measured a ratio of 0.63 against a random sample."""
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1)
    alloc = np.array(s["allocation"])
    assert alloc[0] > alloc[4] and alloc.sum() == 300


# ----------------------------------------------------------------------------- estimates, on real units
@pytest.mark.parametrize("design", ["random", "proportional", "confidence"])
def test_estimate_covers_the_true_rate_on_mados_at_the_rate_exp78_recorded(design):
    """Two hundred reviewer draws on exp78's MADOS export: coverage near 0.95, the confidence design narrower."""
    margin, p1, err, tile = _units("mados")
    theta = err.mean()
    cov, width = 0, 0.0
    for seed in range(200):
        s = ox.sample_for_estimation(margin, 300, design=design, p1=p1, seed=seed)
        r = ox.estimate_error_rate(s, err[s["indices"]])
        cov += r["low"] <= theta <= r["high"]
        width += r["half_width"]
    assert cov / 200 >= 0.90, (design, cov / 200)
    assert 0.01 < width / 200 < 0.04


def test_the_confidence_design_is_narrower_than_random_on_mados():
    margin, p1, err, tile = _units("mados")
    w = {}
    for design in ("random", "confidence"):
        w[design] = np.mean([ox.estimate_error_rate(s := ox.sample_for_estimation(margin, 300, design=design, p1=p1, seed=k),
                                                   err[s["indices"]])["half_width"] for k in range(100)])
    assert w["confidence"] < 0.8 * w["random"], w                # exp78: 0.63


def test_the_tile_design_reports_the_cluster_interval_and_the_naive_one_beside_it():
    """On MADOS the naive interval covered on 0.506 of draws in exp78 against a population design effect of 9.77.
    MADOS tiles hold a median of six valid windows, so one draw's design effect is noisy (5% of draws fall under
    1.5); the test is over 100 draws, and the budget must be met on every one."""
    margin, p1, err, tile = _units("mados")
    wider, deffs = 0, []
    for seed in range(100):
        s = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tile, seed=seed)
        assert s["indices"].size == 300, seed
        r = ox.estimate_error_rate(s, err[s["indices"]])
        naive = r["naive_interval_if_treated_as_random"]
        assert r["method"].startswith("ratio estimator") and "warning" in r
        wider += (r["high"] - r["low"]) > (naive["high"] - naive["low"])
        deffs.append(r["design_effect"])
    # The corrected interval is wider than the naive one on 75% of MADOS draws. The old estimator's was wider on
    # over 90%, but only because one-window tiles, whose means are 0 or 1, inflated its spread.
    assert wider >= 60 and np.median(deffs) > 3


def test_the_tiles_warning_quotes_the_shipped_design_and_exp78_naive_interval():
    """Every number in the warning is read back from the artifact that measured it. "This interval" is the shipped
    design's (exp/out/exp78_shipped_tiles.json); the naive interval's 51 to 78% is exp78's own 18-tile design, on six of
    its seven tasks (recorded beside each task in the same file). 1.3.1 quoted exp78's design for both."""
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp78_shipped_tiles.json")))
    T = rec["tasks"]
    assert len(T) == 7 and rec["config"]["budget"] == 300 and rec["config"]["per_tile"] == est.M_PER_TILE
    cov = {t: v["shipped"]["coverage"] for t, v in T.items()}
    short = sorted(cov, key=cov.get)[:2]
    assert short == ["mados", "sen1floods11"]
    rest = [cov[t] for t in cov if t not in short]
    w = est.TILES_WARNING
    assert len(rest) == 5 and f"covered {100 * min(rest):.1f} to {100 * max(rest):.1f}% of the time on five," in w  # 94.5, 95.4
    assert f"{100 * cov['sen1floods11']:.1f}% on Sen1Floods11 and {100 * cov['mados']:.1f}% on MADOS" in w   # 84.3, 68.5
    # a tenth of the tiles hold most of the errors on those two, and on none of the other five
    tenth = {t: v["tiles"]["share_of_errors_in_top_tenth_of_tiles"] for t, v in T.items()}
    assert all(tenth[t] > 0.5 for t in short) and all(tenth[t] < 0.5 for t in T if t not in short)
    naive = {t: v["exp78_recorded"]["D4/E2_naive"] for t, v in T.items()}
    assert all(v["exp78_recorded"]["tiles"] == 300 // est.M_PER_TILE == 18 for v in T.values())
    inside = [x for x in naive.values() if x < 0.8]
    assert len(inside) == 6 and f"{round(100 * min(inside))} to {round(100 * max(inside))}%" == "51 to 78%"
    assert "18 tiles of 16 windows" in w and "on six of seven tasks while claiming 95%" in w
    assert "0.91 to 0.94" not in w and "0.60 on MADOS" not in w


# ----------------------------------------------------------------------------- the guard
@pytest.mark.parametrize("task", ["mados", "sen1floods11", "pastis_sentinel2", "m_cashew_plant"])
def test_labelling_the_review_set_and_dividing_is_refused_with_the_inflation_named(task):
    """The natural wrong thing, on the real units: the 5% review set gives 1.8 to 5.8 times the true rate."""
    margin, p1, err, tile = _units(task)
    n = margin.size
    k = int(round(0.05 * n))
    review = np.argsort(margin, kind="stable")[:k]                # least confident first: the review set
    chk = ox.review_set_check(review, margin)
    assert chk["looks_like_a_review_set"] and chk["mean_suspicion_percentile"] > 0.95
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(review, err[review], margin)
    assert err[review].mean() > 1.7 * err.mean()                  # and it would indeed have been inflated


def test_a_random_sample_passes_the_guard_and_is_estimated_as_one():
    margin, p1, err, tile = _units("mados")
    idx = np.random.default_rng(0).choice(margin.size, 300, replace=False)
    r = ox.estimate_from_indices(idx, err[idx], margin)
    assert 0.45 < r["review_set_check"]["mean_suspicion_percentile"] < 0.55
    # two hundred random draws: none refused
    for k in range(200):
        j = np.random.default_rng(k).choice(margin.size, 300, replace=False)
        assert not ox.review_set_check(j, margin)["looks_like_a_review_set"], k
    assert r["low"] <= err.mean() <= r["high"] and r["method"].startswith("exact hypergeometric")


def test_the_same_windows_are_estimated_with_their_design_and_refused_without_it():
    """The confidence design oversamples the suspect end on purpose: on MADOS its median suspicion percentile is
    about 0.72 and treating it as random would report 1.9 times the true rate. With its design it carries the
    weights that undo that and is estimated; handed over as bare indices it is refused. That is the guard's job,
    and at 0.75 the first threshold let it through: the threshold is four SDs above a random sample's mean
    percentile (the median until 2026-09-24, which heavy ties defeated)."""
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1)
    chk = ox.review_set_check(s["indices"], margin)
    assert chk["mean_suspicion_percentile"] > chk["threshold"] and abs(chk["threshold"] - 0.566) < 0.01
    assert err[s["indices"]].mean() > 1.5 * err.mean()
    r = ox.estimate_error_rate(s, err[s["indices"]])
    assert r["low"] <= err.mean() <= r["high"]
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(s["indices"], err[s["indices"]], margin)


def test_exp78_reads_its_estimators_from_the_package():
    """One implementation: the recorded run and the shipped code cannot drift apart."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("e78", os.path.join(ROOT, "exp", "exp78_error_rate_estimation.py"))
    e78 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e78)
    assert e78.wilson is est.wilson_interval and e78.stratified_interval is est.stratified_interval
    assert e78.neyman is est.neyman_allocation and e78.design_effect is est.design_effect
    assert e78.Z95 == est.Z95 and e78.MIN_PER_STRATUM == est.MIN_PER_STRATUM


def test_the_shipped_stratified_interval_agrees_with_the_graded_one_away_from_the_edges_and_covers_no_worse():
    """exp78 graded the Wald form. The Wilson-effective-n form the package ships must agree with it to the third
    decimal where both are defined, and cover at least as often, on the real units of the cleanest task."""
    margin, p1, err, tile = _units("mados")
    theta = err.mean()
    cov_w = cov_s = 0
    gaps = []
    pos = None
    for seed in range(200):
        s = ox.sample_for_estimation(margin, 300, p1=p1, seed=seed)
        if pos is None:                                           # the population is the same for every seed
            pos = np.full(margin.size, -1); pos[s["strata_of_population"]] = np.arange(s["strata_of_population"].size)
        local = pos[s["indices"]]
        e = err[s["strata_of_population"]]
        _, wlo, whi, _ = est.stratified_interval(e, s["strata"], local, s["sizes"], s["n_population"])
        _, slo, shi, _, n_eff = est.stratified_interval_wilson(e, s["strata"], local, s["sizes"], s["n_population"])
        cov_w += wlo <= theta <= whi
        cov_s += slo <= theta <= shi
        gaps.append(abs((shi - slo) - (whi - wlo)))
        assert 100 < n_eff < 5000
    assert cov_s >= cov_w and cov_s / 200 >= 0.93
    assert np.median(gaps) < 3e-3


# ----------------------------------------------------------------------------- the estimator audit, 2026-09-22
def test_the_tile_estimate_is_the_window_rate_not_the_average_tiles_rate_on_mados():
    """MADOS tiles hold 1 to 400 valid windows. The unweighted mean of tile means, which exp78 first recorded,
    targets the average tile's rate, 0.133, and returned 1.78 times the true 0.074; the ratio estimator weights
    each tile by its windows and is unbiased to within Monte Carlo noise over 150 draws."""
    margin, p1, err, tile = _units("mados")
    ests = []
    for seed in range(300):
        s = ox.sample_for_estimation(margin, 300, design="tiles", tiles=tile, seed=seed)
        ests.append(ox.estimate_error_rate(s, err[s["indices"]])["estimate"])
    # Over 1,000 draws the ratio is 1.029 with a Monte Carlo SE of 0.030; at 300 draws the SE is 0.055, so 0.2 is
    # 3.6 SE and still separates cleanly from the old estimator's 1.78. An 8% tolerance at 150 draws, the first
    # version of this test, was one SE and failed on noise.
    assert abs(np.mean(ests) / err.mean() - 1) < 0.2, np.mean(ests) / err.mean()
    ids = np.unique(tile)
    assert np.mean([err[tile == u].mean() for u in ids]) > 1.7 * err.mean()       # what the old estimator targeted


def test_nan_margins_are_not_the_suspect_end_so_the_review_set_cannot_hide_behind_no_data():
    """assess's confidence array is NaN at no-data; argsort put NaN at the suspect end, and at 40% no-data the
    tool's own review set passed the guard and was estimated at 5.5 times the true rate."""
    margin, p1, err, tile = _units("mados")
    nd = np.zeros(margin.size, bool); nd[: int(0.4 * margin.size)] = True
    m_nan = np.where(nd, np.nan, margin)
    vi = np.flatnonzero(~nd)
    review = vi[np.argsort(margin[vi], kind="stable")[: int(0.05 * vi.size)]]
    assert ox.review_set_check(review, m_nan)["looks_like_a_review_set"]        # without passing `valid`
    with pytest.raises(ValueError, match="not a sample"):
        ox.estimate_from_indices(review, err[review], m_nan)


def test_estimate_from_indices_refuses_no_data_windows_and_repeats():
    margin, p1, err, tile = _units("mados")
    valid = np.ones(margin.size, bool); valid[: int(0.3 * margin.size)] = False
    rng = np.random.default_rng(0)
    good = rng.choice(np.flatnonzero(valid), 150, replace=False)
    bad = rng.choice(np.flatnonzero(~valid), 150, replace=False)
    with pytest.raises(ValueError, match="outside the valid map"):
        ox.estimate_from_indices(np.concatenate([good, bad]), np.concatenate([err[good], np.zeros(150)]), margin, valid)
    with pytest.raises(ValueError, match="more than once"):
        ox.estimate_from_indices(np.tile(good[:100], 3), np.tile(err[good[:100]], 3), margin, valid)


def test_a_budget_too_small_to_floor_every_stratum_is_refused_rather_than_zeroing_the_suspect_end():
    """With a budget under 2 x 5 the allocation used to zero the least confident strata, and the estimate treated
    the most error-prone part of the map as error-free: 0.009 against a true 0.074 on MADOS."""
    with pytest.raises(ValueError, match="cannot give each"):
        est.neyman_allocation(np.array([1000] * 5), np.ones(5), 3)
    margin, p1, err, tile = _units("mados")
    with pytest.raises(ValueError, match="cannot give each"):
        ox.sample_for_estimation(margin, 4, p1=p1)


def test_a_stratified_sample_with_a_repeated_or_foreign_window_is_refused_with_a_message():
    margin, p1, err, tile = _units("mados")
    s = ox.sample_for_estimation(margin, 300, p1=p1, seed=0)
    w = err[s["indices"]]
    dup = {**s, "indices": np.concatenate([s["indices"], s["indices"][:10]])}
    with pytest.raises(ValueError, match="more than once"):
        ox.estimate_error_rate(dup, np.concatenate([w, np.zeros(10)]))
    valid = np.ones(margin.size, bool); valid[:100] = False
    s2 = ox.sample_for_estimation(margin, 300, p1=p1, valid=valid, seed=0)
    foreign = {**s2, "indices": np.concatenate([s2["indices"][:-1], [5]])}         # window 5 is not in the population
    with pytest.raises(ValueError, match="not in the population"):
        ox.estimate_error_rate(foreign, err[foreign["indices"]])


def test_the_exact_interval_is_the_tail_inversion_and_covers_at_least_nominal_by_enumeration():
    """Written out by scanning every K, not by bisection: the interval is every K whose two tails at the observed
    k exceed 2.5%. Its exact coverage is at least 0.95 on a grid where the finite-population Wilson form falls to
    0.80, and it is the point on a census and everything on an empty sample."""
    def by_scan(k, n, N):
        tail_up = lambda K: sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(n, K) + 1)) / math.comb(N, n)
        tail_dn = lambda K: sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(max(0, n - (N - K)), k + 1)) / math.comb(N, n)
        keep = [K for K in range(k, N - (n - k) + 1) if tail_up(K) > 0.025 and tail_dn(K) > 0.025]
        return min(min(keep) / N, k / n), max(max(keep) / N, k / n)      # widened to hold the sample share
    for k, n, N in ((0, 10, 50), (1, 10, 50), (5, 10, 50), (10, 10, 50), (29, 30, 100), (1, 30, 100), (13, 20, 60)):
        assert est.hypergeom_interval(k, n, N) == pytest.approx(by_scan(k, n, N), abs=1e-12)
    worst = 1.0
    for N in (50, 100, 300):
        for K in (1, 2, 3, 5, 10, 25):
            for n in (10, 20, 30, 45):
                if n < N:
                    worst = min(worst, est.exact_coverage_srs(N, K, n, interval=est.hypergeom_interval))
    assert worst >= 0.95
    assert est.exact_coverage_srs(50, 1, 10) == pytest.approx(0.8)          # the Wilson form's dip on the same grid
    assert est.hypergeom_interval(7, 40, 40) == (7 / 40, 7 / 40) and est.hypergeom_interval(0, 0, 40) == (0.0, 1.0)
    with pytest.raises(ValueError):
        est.hypergeom_interval(5, 3, 40)


# ----------------------------------------------------------------------------- the input-condition design on exp88's map
def test_the_condition_design_on_exp88s_mixed_map():
    """exp88's mixed PASTIS map (Base, probe seed 0): half the tiles read without the optical input, rebuilt from the
    recorded units with exp88's own mixed_map. The condition design with the cloud flag as the condition, 200 seeds
    at 300 labels: each part's exact interval and the whole-map interval, the parts' exact intervals at 97.5%
    weighted by share, cover their truth on at least 0.93 of draws, and the whole-map one on at least 0.95. A check
    of the tool on one real map, not a recorded claim."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    import exp88_missing_modality as e88
    d = np.load(os.path.join(ROOT, "exp", "out", "exp88_units.npz"))
    base = "olmoearth_base/pastis"
    units = {c: {k: d[f"{base}/seed0/{c}/{k}"] for k in ("margin", "p1", "err")} for c in ("full", "optical_missing")}
    n_tiles = json.load(open(os.path.join(ROOT, "exp", "out", "exp88_summary.json")))[
        "results"]["olmoearth_base"]["seeds"]["0"]["mixed_map"]["n_tiles"]
    mm = e88.mixed_map(units["full"], units["optical_missing"], d[f"{base}/tile"], n_tiles)
    assert np.array_equal(mm["cloudy_tiles"], d[f"{base}/cloudy_tiles"])        # the recorded mix, rebuilt
    err, cond = (mm["err"] > 0.5).astype(float), mm["cloudy"].astype(int)
    truth = {"whole": err.mean(), "clear": err[cond == 0].mean(), "cloudy": err[cond == 1].mean()}
    assert (round(truth["cloudy"], 3), round(truth["clear"], 3)) == (0.741, 0.197)   # exp88's 74.1% and 19.7%
    cover = {k: 0 for k in truth}
    for seed in range(200):
        s = ox.sample_for_estimation(mm["margin"], 300, design="condition", condition=cond,
                                     condition_names={0: "clear", 1: "cloudy"}, seed=seed)
        assert s["allocation"] == [150, 150] and s["condition"]["names"] == ["clear", "cloudy"]
        r = ox.estimate_error_rate(s, err[s["indices"]])
        assert r["conditions_in_interval"] == 2
        cover["whole"] += r["low"] <= truth["whole"] <= r["high"]
        for part in ("clear", "cloudy"):
            row = r["per_condition"][part]
            cover[part] += row["low"] <= truth[part] <= row["high"]
    for k, v in cover.items():
        assert v / 200 >= 0.93, (k, v / 200)
    # the whole-map interval holds at least 95% by construction, and it does here (0.995); the stratified Wilson
    # interval with the conditions as strata, without a floor, held 0.945 of these draws
    assert cover["whole"] / 200 >= 0.95, cover["whole"] / 200
