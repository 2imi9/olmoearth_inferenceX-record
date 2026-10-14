"""The model-assisted (difference / PPI++) estimators, checked by enumeration (docs/plan/model_assisted_estimation.md).

Over every simple random sample of a small population the difference estimator is exactly unbiased for any fixed
coefficient and its variance estimator, (1 - n/N) s^2(residual)/n written here from Cochran (1977, section 2.8),
is unbiased for the exact variance of the estimates; the stratified form the same over every stratified sample;
the tuned coefficient converges to the population covariance ratio and never widens the interval beyond tuning
noise."""
import itertools

import numpy as np

from oe_inferencex import estimate as est

ERR = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1], float)
G = np.array([0.9, 0.6, 0.2, 0.1, 0.7, 0.8, 0.3, 0.2, 0.1, 0.05, 0.15, 0.5])     # a predictor that tracks the errors
N = ERR.size


def _every_srs(n):
    for c in itertools.combinations(range(N), n):
        yield np.array(c)


def test_difference_estimator_is_unbiased_and_its_variance_estimator_unbiased_over_every_sample():
    n = 5
    for lam in (0.0, 0.5, 1.0):
        ests, vhat = [], []
        for i in _every_srs(n):
            e, lo, hi, l = est.model_assisted_interval(ERR, G, i, N, lam=lam)
            assert l == lam
            assert abs(e - (lam * G.mean() + (ERR[i] - lam * G[i]).mean())) < 1e-12
            ests.append(e)
            resid = ERR[i] - lam * G[i]
            vhat.append((1 - n / N) * resid.var(ddof=1) / n)
            if 0 < e < 1:                                            # the returned bounds are the Wald ones, clipped
                assert abs((hi - lo) / 2 - est.Z95 * np.sqrt(vhat[-1])) < 1e-12 or hi == 1.0 or lo == 0.0
        ests, vhat = np.array(ests), np.array(vhat)
        assert abs(ests.mean() - ERR.mean()) < 1e-12                    # unbiased for every fixed coefficient
        assert abs(vhat.mean() - ests.var()) < 1e-12                    # and the variance estimator is unbiased


def test_stratified_form_is_unbiased_over_every_stratified_sample_for_a_fixed_coefficient():
    strata = np.repeat(np.arange(3), (5, 4, 3))
    sizes = [5, 4, 3]
    groups = [np.flatnonzero(strata == h) for h in range(3)]
    for lam in (0.0, 1.0):
        ests, vhat = [], []
        for parts in itertools.product(*[itertools.combinations(g, 2) for g in groups]):
            i = np.concatenate([np.array(p) for p in parts])
            e, lo, hi, lams, starved = est.stratified_model_assisted_interval(ERR, G, strata, i, sizes, N, lam=lam)
            assert starved == 0 and all(v == lam for v in lams.values())
            ests.append(e)
            v = 0.0
            for h, Nh in enumerate(sizes):
                m = strata[i] == h
                resid = ERR[i][m] - lam * G[i][m]
                v += (Nh / N) ** 2 * (1 - m.sum() / Nh) * resid.var(ddof=1) / m.sum()
            vhat.append(v)
        ests, vhat = np.array(ests), np.array(vhat)
        assert abs(ests.mean() - ERR.mean()) < 1e-12
        assert abs(vhat.mean() - ests.var()) < 1e-12


def test_the_tuned_coefficient_converges_to_the_population_covariance_ratio_and_never_widens_much():
    rng = np.random.default_rng(0)
    Npop = 50000
    g = rng.random(Npop)
    e = (rng.random(Npop) < 0.5 * g).astype(float)
    lam_pop = np.cov(e, g)[0, 1] / g.var()
    lams, w_cls, w_tuned = [], [], []
    for r in range(300):
        i = rng.choice(Npop, 1000, replace=False)
        _, lo, hi, lam = est.model_assisted_interval(e, g, i, Npop)
        lams.append(lam)
        w_tuned.append(hi - lo)
        _, lo0, hi0, _ = est.model_assisted_interval(e, g, i, Npop, lam=0.0)
        w_cls.append(hi0 - lo0)
    assert abs(np.mean(lams) - lam_pop) < 0.05
    assert np.mean(w_tuned) < np.mean(w_cls)                                   # the predictor helps here
    assert max(np.array(w_tuned) / np.array(w_cls)) < 1.03                      # and never costs more than tuning noise
    assert est.tuned_coefficient(np.array([0.0, 1.0]), np.array([0.2, 0.9])) == 0.0   # too few labels to tune


def test_the_tuning_floor_keeps_small_strata_classical_and_the_stratified_tuned_form_is_exercised():
    """exp85's audit: with a floor of three labels the per-stratum coefficient reached 66 on a 26-label stratum
    whose predictor barely varied, and the Wald interval, which treats the coefficient as fixed, claimed half the
    true spread. Below MIN_FOR_TUNING labels the coefficient is 0; above it the tuned stratified estimate is
    unbiased over a Monte Carlo and its interval covers at nominal."""
    rng = np.random.default_rng(11)
    assert est.tuned_coefficient(rng.random(est.MIN_FOR_TUNING - 1), rng.random(est.MIN_FOR_TUNING - 1)) == 0.0
    Npop = 20000
    margin = rng.random(Npop)
    strata = est.confidence_strata(margin, 4)
    sizes = np.bincount(strata, minlength=4)
    g = 1 - (0.5 + 0.5 * margin)
    e = (rng.random(Npop) < 1.5 * g).astype(float)
    theta = e.mean()
    ests, cover, lam_seen = [], 0, []
    for r in range(600):
        picked = est.draw_stratified(np.random.default_rng(r), strata, sizes, np.array([100, 100, 100, 100]))
        est_, lo, hi, lams, starved = est.stratified_model_assisted_interval(e, g, strata, picked, sizes, Npop)
        ests.append(est_); cover += lo <= theta <= hi; lam_seen += list(lams.values())
    ests = np.array(ests)
    assert abs(ests.mean() - theta) < 3 * ests.std() / np.sqrt(ests.size) + 2e-3
    assert cover / 600 > 0.92 and 0.5 < np.median(lam_seen) < 2.0
