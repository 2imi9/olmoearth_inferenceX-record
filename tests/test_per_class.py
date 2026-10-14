"""Per-class accuracy from a labelled sample (docs/plan/per_class_assessment.md, exp81), checked by enumeration
where the claim is an equality and by a large seeded Monte Carlo where it is a variance.

Under the confidence design every per-class quantity is built from Horvitz-Thompson totals over the margin strata;
over EVERY stratified sample of a small population those totals are unbiased and their variance estimator is
unbiased for the exact variance. Under a random sample the user's accuracy interval must be the exact hypergeometric
interval on the labelled windows of that map class, the class shares the post-stratified form of Olofsson et al. 2014, and the
producer's accuracy their eq. 7, written out independently here."""
import itertools

import numpy as np
import pytest

from oe_inferencex import estimate as est

# a population of 12 windows: map class (rows) and reference class, three margin strata of sizes 5, 4, 3
MAP = np.array([0, 0, 1, 1, 2,   0, 1, 1, 2,   0, 2, 2])
REF = np.array([0, 1, 1, 1, 2,   0, 0, 1, 1,   0, 2, 2])
STRATA = np.repeat(np.arange(3), (5, 4, 3))
SIZES = [5, 4, 3]
N = 12
ALLOC = (2, 2, 2)


def _every_stratified_sample():
    groups = [np.flatnonzero(STRATA == h) for h in range(3)]
    for parts in itertools.product(*[itertools.combinations(g, a) for g, a in zip(groups, ALLOC)]):
        yield np.concatenate([np.array(p) for p in parts])


def _sample_obj(local):
    return {"design": "confidence", "indices": local, "n_population": N, "strata": STRATA,
            "strata_of_population": np.arange(N), "sizes": SIZES, "n_strata": 3, "budget": int(local.size)}


def test_ht_totals_are_unbiased_and_their_variance_estimator_is_unbiased_over_every_stratified_sample():
    both = ((MAP == 1) & (REF == 1)).astype(float)
    truth = both.sum()
    ests, vars_ = [], []
    for local in _every_stratified_sample():
        t, v = est._ht_total(both, STRATA, local, SIZES)
        ests.append(t)
        vars_.append(v)
    ests, vars_ = np.array(ests), np.array(vars_)
    assert abs(ests.mean() - truth) < 1e-12
    assert abs(vars_.mean() - ests.var()) < 1e-12


def test_per_class_under_the_confidence_design_returns_ratios_of_the_same_totals():
    local = np.array([0, 3, 5, 7, 9, 11])
    out = est.estimate_per_class(_sample_obj(local), REF[local], MAP)
    both = ((MAP == 1) & (REF == 1)).astype(float)
    ua, _ = est._ht_ratio(both, (MAP == 1).astype(float), STRATA, local, SIZES)
    pa, _ = est._ht_ratio(both, (REF == 1).astype(float), STRATA, local, SIZES)
    tot, _ = est._ht_total((REF == 1).astype(float), STRATA, local, SIZES)
    assert abs(out["per_class"][1]["user_accuracy"]["estimate"] - ua) < 1e-12
    assert abs(out["per_class"][1]["producer_accuracy"]["estimate"] - pa) < 1e-12
    assert abs(out["per_class"][1]["reference_share"]["estimate"] - tot / N) < 1e-12
    assert out["confusion_counts"][1][1] == int(((MAP[local] == 1) & (REF[local] == 1)).sum())
    assert "warning" in out["per_class"][1]                     # far fewer than 30 labelled


def _olofsson(conf, N_map):
    """Olofsson et al. 2014, eq. 4-7, written from the paper: rows map class, columns reference class."""
    C = conf.shape[0]
    n_i = conf.sum(1)
    W = N_map / N_map.sum()
    p = np.zeros((C, C))
    for i in range(C):
        if n_i[i] > 0:
            p[i] = W[i] * conf[i] / n_i[i]
    ua = np.array([p[i, i] / p[i].sum() if p[i].sum() > 0 else np.nan for i in range(C)])
    pa = np.array([p[j, j] / p[:, j].sum() if p[:, j].sum() > 0 else np.nan for j in range(C)])
    share = p.sum(0)
    v_share = np.zeros(C)
    for j in range(C):
        v_share[j] = sum(W[i] ** 2 * (conf[i, j] / n_i[i]) * (1 - conf[i, j] / n_i[i]) / (n_i[i] - 1) * (1 - n_i[i] / N_map[i])
                         for i in range(C) if n_i[i] > 1)
    v_pa = np.full(C, np.nan)
    fpc = 1 - n_i / N_map                                       # the record's form: eq. 7 with each class's FPC
    for j in range(C):
        Nhat = sum(N_map[i] * conf[i, j] / n_i[i] for i in range(C) if n_i[i] > 0)
        if Nhat > 0 and n_i[j] > 1:
            u = conf[j, j] / n_i[j]
            t1 = N_map[j] ** 2 * (1 - pa[j]) ** 2 * u * (1 - u) * fpc[j] / (n_i[j] - 1)
            t2 = pa[j] ** 2 * sum(N_map[i] ** 2 * (conf[i, j] / n_i[i]) * (1 - conf[i, j] / n_i[i]) * fpc[i] / (n_i[i] - 1)
                                  for i in range(C) if i != j and n_i[i] > 1)
            v_pa[j] = (t1 + t2) / Nhat ** 2
    return ua, pa, share, v_share, v_pa


def test_random_design_matches_the_exact_interval_per_class_and_olofssons_equations():
    """The user's accuracy under a random sample is the exact hypergeometric interval (exp81's sixth amendment;
    until then this test pinned the finite-population Wilson form, which fell to 0.70 on a one-error class)."""
    rng = np.random.default_rng(3)
    Npop, C, B = 3000, 4, 300
    mc = rng.integers(0, C, Npop)
    ref = np.where(rng.random(Npop) < 0.8, mc, rng.integers(0, C, Npop))
    idx = rng.choice(Npop, B, replace=False)
    sample = {"design": "random", "indices": idx, "n_population": Npop, "budget": B}
    out = est.estimate_per_class(sample, ref[idx], mc, interval="wald")       # the field's form, for the equations
    conf = np.array(out["confusion_counts"])
    N_map = np.bincount(mc, minlength=C).astype(float)
    ua, pa, share, v_share, v_pa = _olofsson(conf, N_map)
    for c in range(C):
        row = out["per_class"][c]
        lo, hi = est.hypergeom_interval(int(conf[c, c]), int(conf[c].sum()), int(N_map[c]))
        assert abs(row["user_accuracy"]["estimate"] - ua[c]) < 1e-12
        assert (row["user_accuracy"]["low"], row["user_accuracy"]["high"]) == (lo, hi)
        assert abs(row["producer_accuracy"]["estimate"] - pa[c]) < 1e-12
        assert abs(row["producer_accuracy"]["high"] - min(1.0, pa[c] + est.Z95 * np.sqrt(v_pa[c]))) < 1e-12
        assert abs(row["reference_share"]["estimate"] - share[c]) < 1e-12
        assert abs(row["reference_share"]["high"] - min(1.0, share[c] + est.Z95 * np.sqrt(v_share[c]))) < 1e-12
    # the overall accuracy is `estimate`'s: the simple proportion right with the exact interval; Olofsson's
    # post-stratified sum is kept beside it as a point so the table adds up
    assert abs(out["overall_accuracy_post_stratified"] - sum(N_map[c] / Npop * conf[c, c] / conf[c].sum() for c in range(C))) < 1e-12
    assert out["overall_accuracy"]["estimate"] == np.trace(conf) / B
    assert (out["overall_accuracy"]["low"], out["overall_accuracy"]["high"]) == est.hypergeom_interval(int(np.trace(conf)), B, Npop)


@pytest.mark.parametrize("design", ["random", "confidence"])
def test_the_variance_estimators_track_the_monte_carlo_variance(design):
    """On a population with an unbalanced map, the mean estimated variance of every per-class quantity is within
    15% of the Monte Carlo variance of its estimate over 3000 draws, and the estimates are unbiased to 3 SEs."""
    rng = np.random.default_rng(7)
    Npop, C, B, R = 4000, 3, 400, 3000
    margin = rng.random(Npop)
    mc = np.where(margin < 0.3, 0, np.where(margin < 0.8, 1, 2))                 # class depends on margin
    ref = np.where(rng.random(Npop) < 0.6 + 0.35 * margin, mc, rng.integers(0, C, Npop))
    p1 = 0.5 + 0.5 * margin
    N_map = np.bincount(mc, minlength=C)
    truth_ua = np.array([((mc == c) & (ref == c)).sum() / N_map[c] for c in range(C)])
    truth_share = np.bincount(ref, minlength=C) / Npop
    ests, vars_ = {k: [] for k in ("ua", "pa", "share")}, {k: [] for k in ("ua", "pa", "share")}
    for r in range(R):
        s = est.sample_for_estimation(margin, B, design=design, p1=p1, seed=r)
        out = est.estimate_per_class(s, ref[s["indices"]], mc)
        for c in range(1, 2):                                                   # the middle class, which cuts all strata
            row = out["per_class"][c]
            for k, key in (("ua", "user_accuracy"), ("pa", "producer_accuracy"), ("share", "reference_share")):
                ests[k].append(row[key]["estimate"])
                vars_[k].append(((row[key]["high"] - row[key]["low"]) / (2 * est.Z95)) ** 2)
    for k in ("ua", "pa", "share"):
        e, v = np.array(ests[k]), np.array(vars_[k])
        ratio = v.mean() / e.var()
        assert 0.85 < ratio < 1.15, (design, k, ratio)
    truth_pa1 = ((mc == 1) & (ref == 1)).sum() / (ref == 1).sum()
    for k, t in (("ua", truth_ua[1]), ("pa", truth_pa1), ("share", truth_share[1])):
        e = np.array(ests[k])
        assert abs(e.mean() - t) < 3 * e.std() / np.sqrt(R) + 1e-3, (design, k, e.mean(), t)


def test_refusals():
    rng = np.random.default_rng(0)
    mc = rng.integers(0, 3, 500)
    s = est.sample_for_estimation(rng.random(500), 60, design="random")
    with pytest.raises(ValueError, match="reference labels for"):
        est.estimate_per_class(s, np.zeros(10, int), mc)
    with pytest.raises(ValueError, match=">= 0"):
        est.estimate_per_class(s, np.full(60, -1), mc)
    with pytest.raises(ValueError, match="drawn from"):
        est.estimate_per_class(s, np.zeros(60, int), np.r_[mc, -1, 2])
    t = est.sample_for_estimation(rng.random(500), 64, design="tiles", tiles=np.arange(500) // 8)
    with pytest.raises(ValueError, match="tile sample"):
        est.estimate_per_class(t, np.zeros(64, int), mc)
    bad = mc.copy()
    bad[s["indices"][0]] = -1
    with pytest.raises(ValueError):
        est.estimate_per_class(s, np.zeros(60, int), bad)


def test_the_few_errors_note_gives_each_quantity_its_own_count():
    """The exp86 round 6 audit: a class with 3 omissions and 20 commissions had its producer's accuracy said to rest on
    "20 to 23" sampled errors; it rests on its 3 omissions (the share on 23, which is not few)."""
    N = 2000
    mc = np.repeat([0, 1], N // 2)
    sample = {"design": "random", "indices": np.arange(0, N, 5), "n_population": N, "budget": N // 5}
    idx = sample["indices"]
    ref = mc[idx].copy()
    ref[np.flatnonzero(mc[idx] == 1)[:20]] = 0              # the map calls 20 windows class 1 wrongly
    ref[np.flatnonzero(mc[idx] == 0)[:3]] = 1               # and misses 3 windows of class 1
    row = est.estimate_per_class(sample, ref, mc, n_classes=2)["per_class"][1]
    assert "few errors" in row["warning_codes"]
    assert "the producer's accuracy rests on 3 sampled errors" in row["warning"]
    assert "share" not in row["warning"].split("(")[0] and "20 to 23" not in row["warning"]
    assert "(20 the map calls this class wrongly, 3 of this class it misses)" in row["warning"]


def test_the_few_errors_warning_fires_on_one_to_four_sampled_errors_and_not_on_zero_or_five():
    """Seventh amendment: the producer's accuracy and the share rest on the sampled errors of the kind they count;
    one to four of them is warned, none (the interval falls back wide) and five or more are not."""
    N = 2000
    mc = np.repeat([0, 1], N // 2)
    sample = {"design": "random", "indices": np.arange(0, N, 5), "n_population": N, "budget": N // 5}
    idx = sample["indices"]
    for n_missed, expect in ((0, False), (1, True), (4, True), (5, False), (12, False)):
        ref = mc[idx].copy()
        cls1 = np.flatnonzero(ref == 1)
        ref[cls1[:n_missed]] = 0                          # class-0 windows the map calls 1: commissions of class 1
        ref_c0 = ref.copy()
        out = est.estimate_per_class(sample, ref_c0, mc, n_classes=2)
        codes = out["per_class"][1].get("warning_codes", [])
        assert ("few errors" in codes) is expect, (n_missed, codes)


# ----------------------------------------------------------------------------- the input-condition design
def _wilson(p, n):
    z = est.Z95
    d = 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def test_per_class_under_the_condition_design():
    """Per-class accuracy is allowed under the condition design, with the input conditions as the strata of the same
    Horvitz-Thompson estimators, and the method says it is not graded. The numbers are written out here per
    stratum, without the package's helpers; the wording names the design."""
    cond = np.array([0] * 5 + [1] * 4 + [-1] * 3)                     # the strata above, as conditions (the third unrecorded)
    s = est.sample_for_estimation(np.linspace(1, 0, N), 6, design="condition", condition=cond)
    assert s["allocation"] == list(ALLOC) and np.array_equal(s["strata"], STRATA)
    local = np.array([0, 3, 5, 7, 9, 11])
    out = est.estimate_per_class(dict(s, indices=local), REF[local], MAP)
    assert out["method"].startswith("stratified by input condition: ratios of Horvitz-Thompson totals")
    assert "not been graded" in out["method"] and "confidence margin" not in out["method"]
    c = 1
    num, den = ((MAP == c) & (REF == c)).astype(float), (MAP == c).astype(float)
    tot = lambda v: sum(Nh * v[local][STRATA[local] == h].mean() for h, Nh in enumerate(SIZES))
    R = tot(num) / tot(den)                                             # the user's accuracy, a ratio of HT totals
    z = (num - R * den) / tot(den)
    var = sum(Nh ** 2 * (1 - 2 / Nh) * z[local][STRATA[local] == h].var(ddof=1) / 2 for h, Nh in enumerate(SIZES))
    ua = out["per_class"][c]["user_accuracy"]
    assert abs(ua["estimate"] - R) < 1e-12
    n_eff = R * (1 - R) / var if var > 0 and 0 < R < 1 else float(((MAP[local] == c)).sum())
    assert (ua["low"], ua["high"]) == pytest.approx(_wilson(R, n_eff), abs=1e-12)
    is_ref = (REF == c).astype(float)
    assert abs(out["per_class"][c]["producer_accuracy"]["estimate"] - tot(num) / tot(is_ref)) < 1e-12
    assert abs(out["per_class"][c]["reference_share"]["estimate"] - tot(is_ref) / N) < 1e-12
    # the thin-strata note under this design says "strata", not "confidence strata": a condition three times the
    # others' size gets a third of the labels, under half the overall rate
    rng = np.random.default_rng(2)
    cond2 = np.r_[np.zeros(600, int), np.ones(60, int), np.full(60, 2)]
    mc = np.where(cond2 == 0, 0, rng.integers(0, 3, cond2.size))
    ref = np.where(rng.random(cond2.size) < 0.8, mc, rng.integers(0, 3, cond2.size))
    s2 = est.sample_for_estimation(np.linspace(1, 0, cond2.size), 90, design="condition", condition=cond2)
    row = est.estimate_per_class(s2, ref[s2["indices"]], mc)["per_class"][0]
    assert "thin strata" in row["warning_codes"]
    assert "of this class sits in strata the design samples at under half the overall rate" in row["warning"]
    assert "confidence strata" not in row["warning"]
    # under a random sample with a condition recorded, the per-class table is the random design's, unchanged
    r1 = est.sample_for_estimation(np.linspace(1, 0, cond2.size), 90, design="random", seed=4)
    r2 = est.sample_for_estimation(np.linspace(1, 0, cond2.size), 90, design="random", seed=4, condition=cond2)
    assert est.estimate_per_class(r1, ref[r1["indices"]], mc) == est.estimate_per_class(r2, ref[r2["indices"]], mc)


def test_per_class_with_one_condition_is_the_random_design():
    """With one input condition the condition design draws the random design's windows (condition_spec.md 2.9), so
    its per-class table is the random design's: the same numbers, the same method, the post-stratified overall
    accuracy beside the exact one. Only `design` records how the sample was drawn."""
    for seed in range(10):
        rng = np.random.default_rng(50 + seed)
        n = int(rng.integers(300, 900))
        margin = rng.random(n)
        mc = rng.integers(0, 3, n)
        mc[rng.random(n) < 0.05] = -1                                  # no-data windows are outside the population
        margin[mc < 0] = np.nan
        ref = np.where(rng.random(n) < 0.2, rng.integers(0, 3, n), mc)
        cond = np.full(n, int(rng.integers(0, 4)))
        B = int(rng.integers(60, 250))
        r = est.sample_for_estimation(margin, B, design="random", seed=seed)
        c = est.sample_for_estimation(margin, B, design="condition", condition=cond, seed=seed)
        assert np.array_equal(r["indices"], c["indices"])
        pr = est.estimate_per_class(r, ref[r["indices"]], mc)
        pc = est.estimate_per_class(c, ref[c["indices"]], mc)
        assert pc.pop("design") == "condition" and pr.pop("design") == "random"
        assert pc == pr, seed
        assert pc["method"].startswith("random sample: exact hypergeometric") and "graded" not in pc["method"]
