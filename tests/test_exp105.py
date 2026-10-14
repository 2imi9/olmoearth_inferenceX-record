"""exp105 (docs/plan/class_strata_baseline.md): the designs, the win rule, the verdicts and the variance formula,
each against a second route on synthetic units; the grade is refused while the page is a draft."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp105_class_strata as e105  # noqa: E402


def test_the_constants_are_the_pages():
    text = open(e105.PLAN, encoding="utf-8").read()
    assert "at least 14 of the 18" in text and "at most 0.90" in text and "at least 12 of the 18" in text
    assert "R = 2,000" in text and "seed 105" in text and "at least 0.93" in text
    assert (e105.P1_WINS, e105.P1_RATIO, e105.P2_WINS, e105.N_GRADABLE, e105.COVER) == (14, 0.90, 12, 18, 0.93)
    assert (e105.R_DRAWS, e105.SEED, e105.HEADLINE) == (2000, 105, 300)


def test_the_designs_allocate_the_budget_within_the_strata():
    u = e105.synthetic_units(seed=3)
    for B in (100, 300):
        plan = e105.designs(u, B)
        for arm, (s, z, alloc) in plan.items():
            alloc = np.asarray(alloc)
            assert alloc.sum() == B and (alloc <= z).all() and (alloc >= np.minimum(z, 2)).all(), arm
            assert z.sum() == u["err"].size and np.array_equal(np.bincount(s, minlength=z.size), z)
        sC, zC, aC = plan["C"]
        prop = B * zC / zC.sum()
        # the package's integer rule floors each share and hands the remainder, one label at a time, to the stratum
        # with the most units left, so a stratum can sit up to (number of strata - 1) labels above its floor
        assert (aC >= np.floor(prop)).all() and (aC <= np.floor(prop) + zC.size - 1).all(), "C is proportional"
        assert not np.array_equal(plan["D"][2], aC), "D's allocation follows the classes' confidence, not their size"
        assert np.array_equal(plan["C"][0], plan["D"][0]) and np.array_equal(plan["C"][0], plan["E"][0])
        aE = np.asarray(plan["E"][2])
        assert aE.max() - aE.min() <= 1, "E is an equal split where no class is labelled in full"


def test_b_is_the_shipped_confidence_design_also_on_a_saturated_map():
    """Second route: B's allocation is what `sample --design confidence` allocates, Q_FLOOR included, on a map whose
    most confident 40% have p1 = 1.0 (the case the floor exists for)."""
    rng = np.random.default_rng(11)
    N = 20000
    margin = rng.random(N)
    p1 = np.where(margin > np.quantile(margin, 0.6), 1.0, 0.55 + 0.4 * margin)
    u = {"margin": margin, "p1": p1, "err": (rng.random(N) < 0.05).astype(float), "dec": rng.integers(0, 3, N)}
    for B in (100, 300):
        shipped = e105.est.sample_for_estimation(margin, B, design="confidence", p1=p1, seed=0)
        assert list(e105.designs(u, B)["B"][2]) == shipped["allocation"]
        assert min(shipped["allocation"]) > 2, "the floor gives the saturated strata more than two labels"


def test_a_class_labelled_in_full_covers_exactly():
    """E labels a 3-unit class with one error in full: its user's accuracy interval is the truth on every draw."""
    rng = np.random.default_rng(5)
    N = 2000
    dec = np.r_[np.zeros(3, np.int64), rng.integers(1, 4, N - 3)]
    err = np.r_[[1.0, 0.0, 0.0], (rng.random(N - 3) < 0.2).astype(float)]
    u = {"err": err, "dec": dec, "margin": rng.random(N), "p1": 0.5 + rng.random(N) / 2}
    r = e105.run_task(u, 100, 50, np.random.default_rng(2))
    pc = r["arms"]["E"]["per_class_users_accuracy"]
    assert pc["n"][0] == pc["N"][0] == 3 and pc["coverage"][0] == 1.0 and pc["half_width"][0] == 0.0
    assert r["arms"]["E"]["starved_rate"] == 0.0


def test_the_win_rule():
    good, wide, poor = {"coverage": 0.95, "half_width": 0.02}, {"coverage": 0.95, "half_width": 0.03}, \
        {"coverage": 0.90, "half_width": 0.01}
    assert e105.win(good, wide) and not e105.win(wide, good)
    assert e105.win(good, dict(good)), "a tie in width is a win"
    assert e105.win(good, poor) and not e105.win(poor, good), "coverage first"


def _rows(widths_b, widths_x, cover=0.95, gradable=True, widths_d=None, cover_b=None):
    rows = {}
    for t, (wb, wx) in enumerate(zip(widths_b, widths_x)):
        arms = {a: {"coverage": cover, "half_width": wx} for a in e105.ARMS}
        arms["B"] = {"coverage": cover if cover_b is None else cover_b[t], "half_width": wb}
        if widths_d is not None:
            arms["D"] = {"coverage": cover, "half_width": widths_d[t]}
        rows[f"t{t}"] = {"budgets": {"300": {"gradable": gradable, "arms": arms}}}
    return rows


def test_the_verdicts_on_constructed_rows():
    v = e105.verdicts(_rows([0.8] * 14 + [1.2] * 4, [1.0] * 18))
    assert v["P1"]["wins"] == 14 and v["P1"]["holds"] and v["P2"]["wins"] == 14 and v["P2"]["holds"]
    assert v["P1"]["median_half_width_ratio"] == pytest.approx(0.8)
    v = e105.verdicts(_rows([0.95] * 18, [1.0] * 18))
    assert v["P1"]["wins"] == 18 and not v["P1"]["holds"], "every task won but the median ratio 0.95 > 0.90"
    v = e105.verdicts(_rows([0.8] * 13 + [1.2] * 5, [1.0] * 18))
    assert not v["P1"]["holds"] and v["P2"]["wins"] == 13 and v["P2"]["holds"]
    v = e105.verdicts(_rows([0.8] * 18, [1.0] * 18, widths_d=[0.5] * 18))
    assert v["P1"]["holds"] and v["P2"]["wins"] == 0 and not v["P2"]["holds"], "P1 reads C, P2 reads D"
    assert v["reverse"]["D_over_B"] == 18 and v["reverse"]["C_over_B"] == 0 and v["reverse"]["D_over_C"] == 18
    # coverage first in the median too: four narrow, under-covering B intervals must not carry the ratio
    # (counted at their face value 0.5 the median would be 0.875 and P1 would hold)
    v = e105.verdicts(_rows([0.8] * 5 + [0.95] * 9 + [0.5] * 4, [1.0] * 18, cover_b=[0.95] * 14 + [0.85] * 4))
    assert v["P1"]["wins"] == 14 and v["P1"]["median_half_width_ratio"] > 0.90 and not v["P1"]["holds"]
    v = e105.verdicts(_rows([0.8] * 17, [1.0] * 17))
    assert v["n_gradable"] == 17 and v["P1"]["holds"] is None and "verdicts not applied" in v["note"]


def test_the_variance_formula_and_the_run_against_brute_force():
    """Second route: Cochran's variances of the proportional class-stratified estimator and of the simple random
    mean, as predicted_ratio_binary computes them, against the variance of many independent draws written here."""
    rng = np.random.default_rng(7)
    N, B = 3000, 300
    dec = (rng.random(N) < 0.3).astype(np.int64)
    err = (rng.random(N) < np.where(dec == 1, 0.40, 0.05)).astype(np.float64)
    u = {"err": err, "dec": dec, "margin": rng.random(N), "p1": 0.5 + rng.random(N) / 2}
    ratio = e105.predicted_ratio_binary(u, B)
    z = np.bincount(dec)
    alloc = e105.est.neyman_allocation(z, np.ones(2), B)
    pools = [np.flatnonzero(dec == h) for h in range(2)]
    st, srs = [], []
    for _ in range(6000):
        st.append(sum(z[h] / N * err[rng.choice(pools[h], alloc[h], replace=False)].mean() for h in range(2)))
        srs.append(err[rng.choice(N, B, replace=False)].mean())
    assert ratio == pytest.approx(np.var(st, ddof=1) / np.var(srs, ddof=1), rel=0.08)
    r = e105.run_task(u, B, 1500, np.random.default_rng(1), per_class=False)
    assert r["arms"]["C"]["estimate_variance"] == pytest.approx(np.var(st, ddof=1), rel=0.12)
    assert r["arms"]["A"]["estimate_variance"] == pytest.approx(np.var(srs, ddof=1), rel=0.12)
    for arm in e105.ARMS:
        assert r["arms"][arm]["bias_ratio"] == pytest.approx(1.0, abs=0.03), arm
    # every stratified arm's measured variance against Cochran's at its own allocation (written here, not imported)
    for arm, (s, zz, al) in e105.designs(u, B).items():
        v = 0.0
        for h in range(zz.size):
            if al[h] >= zz[h]:
                continue
            p = err[s == h].mean()
            v += (zz[h] / N) ** 2 * (1 - al[h] / zz[h]) * zz[h] / (zz[h] - 1) * p * (1 - p) / al[h]
        assert r["arms"][arm]["estimate_variance"] == pytest.approx(v, rel=0.12), arm
    assert e105.predicted_ratio_binary({**u, "dec": rng.integers(0, 3, N)}, B) is None


def test_the_smoke_and_the_refusal_while_a_draft(tmp_path, monkeypatch):
    rec = e105.main(["--smoke", "--out-dir", str(tmp_path), "--draws", "20"])
    assert set(rec["tasks"]) == {"synthetic_0", "synthetic_1"}
    a = rec["tasks"]["synthetic_0"]["budgets"]["300"]["arms"]
    assert set(a) == set(e105.ARMS) and "per_class_users_accuracy" in a["C"] and "per_class_users_accuracy" in a["E"]
    monkeypatch.setattr(e105, "prereg_status", lambda path=e105.PLAN: "draft")
    with pytest.raises(SystemExit):
        e105.main([])


def test_the_suite_exports_load_as_exp78_exported_them():
    names = e105.tasks()
    assert len(names) == 24
    u = e105.load(names[0])
    assert len({v.size for v in u.values()}) == 1 and u["dec"].dtype == np.int64
    sizes = {t: e105.load(t)["err"].size for t in names}
    assert sum(e105.gradable(n, e105.HEADLINE) for n in sizes.values()) == e105.N_GRADABLE


def test_exp105_recorded_by_a_second_route():
    """The recorded verdicts and the claims' numbers recomputed from the summary's per-task rows with code written
    here (no import of exp105's win, ratio or verdicts)."""
    import json
    with open(os.path.join(ROOT, "exp", "out", "exp105_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    T = S["tasks"]
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("76b7a3c")
    G = sorted(t for t, r in T.items() if r["n_units"] >= 600)
    assert len(G) == 18 and G == sorted(S["verdicts"]["gradable_tasks"])

    def arms(t, b="300"):
        return T[t]["budgets"][b]["arms"]

    def beats(x, y):
        if x["coverage"] < 0.93:
            return False
        return y["coverage"] < 0.93 or x["half_width"] <= y["half_width"]

    def cf_ratio(x, y):
        return float("inf") if x["coverage"] < 0.93 else 0.0 if y["coverage"] < 0.93 else x["half_width"] / y["half_width"]

    p1 = [t for t in G if beats(arms(t)["B"], arms(t)["C"])]
    p2 = [t for t in G if beats(arms(t)["B"], arms(t)["D"])]
    assert (len(p1), len(p2)) == (14, 16)
    assert round(float(np.median([cf_ratio(arms(t)["B"], arms(t)["C"]) for t in G])), 3) == 0.853
    assert round(float(np.median([cf_ratio(arms(t)["B"], arms(t)["D"]) for t in G])), 3) == 0.863
    assert sum(beats(arms(t)["C"], arms(t)["B"]) for t in G) == 4 and sum(beats(arms(t)["D"], arms(t)["B"]) for t in G) == 2
    dc = {t: arms(t)["D"]["half_width"] / arms(t)["C"]["half_width"] for t in G}
    assert sum(abs(r - 1) < 0.01 for r in dc.values()) == 12 and 0.905 < dc["mados"] < 0.915
    assert all(dc[t] > 1 for t in G if t.startswith("cropharvest_Peoples_Republic_of_China_6"))
    # the report-only 100-label cells on the 200-window maps
    assert [round(arms(t, "100")["B"]["coverage"], 3) for t in ("awf_sentinel2", "awf_landsat")] == [0.913, 0.933]
