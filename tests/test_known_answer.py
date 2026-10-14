"""Known-answer checks of recorded estimators: build a population whose true value is known, draw the recorded
experiment's own design from it many times, and require the estimator to land on the truth.

A claim's check reads its number back from the artifact that produced it; that catches drift and cannot catch a
formula that was wrong from the start. These reach the same numbers by another route. They exist because exp78's
cluster arm covered 0.904 on MADOS while estimating 1.78 times the truth, and nothing then recorded bias."""
import os

import numpy as np
import pytest

from oe_inferencex import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_exp68_design_weighted_lead_is_unbiased_under_its_own_design():
    """exp68's P1 arm: 4,778 field-surveyed LUCAS polygons in the report regions, eight strata, Horvitz-Thompson
    weights from 6 to 42. The pseudo-population replicates each polygon by its weight, so its true lead of the
    margin over the pixel-variance control is known; redrawing exp68's stratified design from it must recover that
    truth. It does, to 0.2 Monte Carlo SE over 400 draws, while the unweighted lead misses by 57 SE; the unweighted
    minus weighted gap comes out at 0.021, as exp68's P4 recorded (0.0216)."""
    z = np.load(os.path.join(ROOT, "exp", "out", "exp68_masks.npz"))
    keep = z["report"] & ~z["no_data"] & (z["obs"] == "field")
    err, margin, var, w = (z[k][keep].astype(float) for k in ("err", "margin", "variance", "weight"))
    strata = np.array([f"{a}:{b}" for a, b in zip(z["y"][keep], z["obs"][keep])])
    assert keep.sum() == 4778
    lead_recorded = metrics.weighted_excess_aurc(var, err, w) - metrics.weighted_excess_aurc(-margin, err, w)
    assert round(lead_recorded, 4) == 0.0953                                        # exp68_summary.json, P1
    m = np.maximum(1, np.rint(w).astype(int))
    pe, pm, pv, ps = np.repeat(err, m), np.repeat(-margin, m), np.repeat(var, m), np.repeat(strata, m)
    truth = metrics.excess_aurc(pv, pe) - metrics.excess_aurc(pm, pe)
    groups = {h: np.flatnonzero(ps == h) for h in np.unique(ps)}
    n_h = {h: int((strata == h).sum()) for h in groups}
    rng = np.random.default_rng(0)
    wl, nl = [], []
    for _ in range(120):
        idx = [rng.choice(g, n_h[h], replace=False) for h, g in groups.items()]
        ww = np.concatenate([np.full(n_h[h], g.size / n_h[h]) for h, g in groups.items()])
        i = np.concatenate(idx)
        e = pe[i]
        wl.append(metrics.weighted_excess_aurc(pv[i], e, ww) - metrics.weighted_excess_aurc(pm[i], e, ww))
        nl.append(metrics.excess_aurc(pv[i], e) - metrics.excess_aurc(pm[i], e))
    wl, nl = np.array(wl), np.array(nl)
    se = wl.std() / np.sqrt(wl.size)
    assert abs(wl.mean() - truth) < 3 * se, (wl.mean(), truth, se)                # unbiased
    assert (nl.mean() - truth) / (nl.std() / np.sqrt(nl.size)) > 10                 # and the check has power
    assert nl.mean() - wl.mean() == pytest.approx(0.0216, abs=0.004)               # P4's recorded gap, another route


def test_exp65_cross_fitted_lead_has_no_optimism_and_reproduces():
    """exp65's held-out fusion leads come from cross-fitting by tile. The known answer for a leak-free pipeline is a
    null: scramble the four extra readings so they carry nothing about errors, and the held-out fusion must not beat
    the margin. Over 12 scrambles the largest null lead was +0.00005 on the test split and every Bolivia one was
    negative, against the recorded +0.0021 (20%) and +0.0029 (30%)."""
    from oe_inferencex.calibrate import fit_ranker
    z = np.load(os.path.join(ROOT, "exp", "out", "exp65_readings.npz"))
    names = ("confidence", "tile_phase", "boundary", "ndwi_level", "s2_variance")
    sig = {k: z[f"ranker/bolivia/{k}"].astype(np.float64) for k in names}
    err = z["ranker/bolivia/err"].astype(np.float64)
    tile = z["ranker/bolivia/tile"]
    ok = np.ones(err.size, bool)
    conf = metrics.excess_aurc(sig["confidence"], err)
    _, rep = fit_ranker(sig, err, ok, groups=tile, family="x", folds=5)
    assert round(conf, 4) == 0.0105 and round(rep["held_out"]["excess_aurc"], 4) == 0.0084     # the recorded pair
    rng = np.random.default_rng(0)
    for _ in range(3):
        perm = rng.permutation(err.size)
        s0 = {"confidence": sig["confidence"], **{k: sig[k][perm] for k in names[1:]}}
        _, r0 = fit_ranker(s0, err, ok, groups=tile, family="x", folds=5)
        assert conf - r0["held_out"]["excess_aurc"] < 0.0005                                # no lead from noise


def test_exp55_activation_clustered_figures_reproduce_from_the_per_event_records():
    """The clustered figures in geoid-capture-effect-size and geoid-exception-rate were computed by hand in a docs
    commit (297a6e0) with no generator and no check: activations as the unit, areas averaged within an activation
    first. Recomputed here from exp55's per-event records by an independent route, they reproduce exactly."""
    import json
    from math import comb
    R = json.load(open(os.path.join(ROOT, "exp", "out", "exp55_summary.json")))["results"]
    want = {"A": ("control NDWI level", 0.843, 16.9, (9, 0), 2.0e-03), "B": ("control S1 level", 0.607, 12.1, (9, 1), 1.1e-02)}
    for t, (ctl, cap_w, ratio_w, (w_w, l_w), p_w) in want.items():
        ev, scored = R[t]["per_event"], R[t]["across_events"]["events_scored"]
        conf = "averaged confidence"
        act = {e: e.rsplit("-", 1)[0] for e in scored}
        acts = sorted(set(act.values()))
        cap = {e: ev[e]["capture"][conf]["0.05"] for e in scored}
        ratio = {e: cap[e] / (max(1, round(0.05 * ev[e]["n_windows"])) / ev[e]["n_windows"]) for e in scored}
        gain = {e: ev[e]["pooled_eaurc"][ctl] - ev[e]["pooled_eaurc"][conf] for e in scored}
        by = lambda d: [np.mean([d[e] for e in scored if act[e] == a]) for a in acts]
        assert round(float(np.median(by(cap))), 3) == cap_w and round(float(np.median(by(ratio))), 1) == ratio_w
        g = by(gain)
        w, l = sum(x > 0 for x in g), sum(x < 0 for x in g)
        p = sum(comb(w + l, k) for k in range(w, w + l + 1)) / 2 ** (w + l)
        assert (w, l) == (w_w, l_w) and p == pytest.approx(p_w, rel=0.06)


def test_exp70_p3_p_value_is_a_rank_sum_over_24_tasks_not_a_sign_test_over_144_pairs():
    """exp70's P3 recorded p = 6.9e-15 from a sign test over 144 task pairs treated as independent. Recomputed here
    from the per-task results by brute force over a million relabellings of the 24 tasks, the p is about 0.0055, and
    the recorded rank-sum p must match it."""
    import json
    S = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))
    R = S["results"]["tasks"]
    adv = {t: r["signals"]["entropy"]["excess_aurc"] - r["signals"]["margin"]["excess_aurc"] for t, r in R.items() if "entropy" in r["signals"]}
    gaps = {t: R[t]["generalisation_gap"] for t in adv}
    med = float(np.median(list(gaps.values())))
    v = np.array([adv[t] for t in adv]); wide_m = np.array([gaps[t] > med for t in adv])
    stat = lambda m: ((v[m][:, None] < v[~m][None]).sum() + 0.5 * (v[m][:, None] == v[~m][None]).sum())
    obs = stat(wide_m)
    rng = np.random.default_rng(1)
    hits = sum(stat(rng.permutation(wide_m)) >= obs - 1e-9 for _ in range(20000))
    p_mc = hits / 20000
    p = S["verdicts"]["P3"]["rank_sum_p"]
    assert abs(p - p_mc) < 4 * np.sqrt(p * (1 - p) / 20000) and 0.004 < p < 0.007
    assert S["verdicts"]["P3"]["holds"] is True and "pairwise_p" not in S["verdicts"]["P3"]


def test_exp80_zone_guarantee_and_coverage_reproduce_from_a_fresh_draw():
    """exp80's headline cell (MADOS, alpha = theta/2, B = 300) re-drawn with a different seed and the package's
    zone rules: the prefix and Bonferroni rules violate on at most delta + 3 SE of draws, the plug-in far more, and
    the median certified coverage is the recorded one. The P1 bound is also recomputed from every recorded cell."""
    import json
    import sys
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    import exp78_error_rate_estimation as e78
    from oe_inferencex import estimate as est
    d = json.load(open(os.path.join(ROOT, "exp", "out", "exp80_summary.json")))
    delta, draws = d["config"]["delta"], 300
    bound = delta + 3 * np.sqrt(delta * (1 - delta) / draws)
    u = e78.load_units("mados")
    err, margin = u["err"], u["margin"]
    order, _ = est.zone_order(margin)
    err_o = err[order]
    N, theta = order.size, float(err.mean())
    alpha, B = theta / 2, 300
    cov, sizes, _ = est.zone_levels(N, B, alpha, delta)
    risk = np.cumsum(err_o)[np.array(sizes) - 1] / np.array(sizes)
    rng = np.random.default_rng(20260923)
    viol = {r: 0 for r in ("prefix", "bonferroni", "plugin")}
    picks = {r: [] for r in viol}
    for _ in range(draws):
        pos = rng.choice(N, B, replace=False)
        b, k = est.zone_counts(pos, err_o[pos], sizes)
        p = [est.zone_pvalue(kk, bb, n, alpha) for kk, bb, n in zip(k, b, sizes)]
        for rule in viol:
            _, best = est.apply_zone_rule(p, b, k, alpha, delta, rule)
            if best is not None:
                viol[rule] += risk[best] > alpha
                picks[rule].append(cov[best])
    rec = [c for c in d["tasks"]["mados"]["cells"] if c["alpha_kind"] == "half_theta" and c["budget"] == B][0]["rules"]
    assert viol["prefix"] / draws <= bound and viol["bonferroni"] / draws <= bound
    assert viol["plugin"] / draws > 0.25 and rec["plugin"]["violation_rate"] > 0.25
    # within one grid step (0.05): the median sits on a grid level and 300 draws can put it one step off
    assert abs(np.median(picks["prefix"]) - rec["prefix"]["median_coverage"]) <= 0.05 + 1e-9
    assert abs(np.median(picks["bonferroni"]) - rec["bonferroni"]["median_coverage"]) <= 0.05 + 1e-9
    # the recorded P1: recomputed over every cell from the per-cell violation rates, not read from the verdict
    bound_rec = delta + 3 * np.sqrt(delta * (1 - delta) / d["config"]["draws"])
    cells = [c for row in d["tasks"].values() for c in row["cells"] if c["monotone"] is not None]
    assert len(cells) == 110
    assert max(c["rules"]["bonferroni"]["violation_rate"] for c in cells) <= bound_rec
    assert max(c["rules"]["prefix"]["violation_rate"] for c in cells) <= bound_rec
    assert max(c["rules"]["plugin"]["violation_rate"] for c in cells) > 0.5
    # the calibration gap of one task, from the per-unit file
    gap = float(u["p1"].mean() - (1 - theta))
    assert abs(gap - d["tasks"]["mados"]["calibration_gap"]) < 1e-9 and 0 < gap < 0.02


def test_the_test_split_boundary_shares_quoted_with_exp18_are_in_the_committed_readings():
    """The record quotes 73% of errors against 18% of correct windows on the Sen1Floods11 test split beside exp18's
    Bolivia shares; the ledger noted that no committed artifact held them. exp65's readings (the v1 shift-averaged
    decision, 149,684 windows) do: 0.732 and 0.180, the same to the percent the record states."""
    z = np.load(os.path.join(ROOT, "exp", "out", "exp65_readings.npz"))
    b, e = z["ranker/test/boundary"], z["ranker/test/err"]
    ok = np.isfinite(b) & np.isfinite(e)
    b, e = b[ok], e[ok] > 0.5
    assert e.size == 149684
    assert round(float((b[e] > 0).mean()), 2) == 0.73 and round(float((b[~e] > 0).mean()), 2) == 0.18
    bb, ee = z["ranker/bolivia/boundary"], z["ranker/bolivia/err"] > 0.5
    assert round(float((bb[ee] > 0).mean()), 2) == 0.75 and round(float((bb[~ee] > 0).mean()), 2) == 0.21


def test_exp79_base_gate_seed_leads_and_engine_differences_recompute_from_the_exports():
    """exp79's first reading (OlmoEarth Base): the gate against exp70 recomputed from the export's seed-0 accuracies;
    every one of the 240 (task, seed) leads recomputed from the signals; the smallest leads named in the record; and
    the RTX-vs-B200 differences recomputed from the two exports by a second route."""
    import json
    lead = lambda sig: min(sig["ctl_embedding_distance"]["excess_aurc"], sig["ctl_class_rarity"]["excess_aurc"]) - sig["margin"]["excess_aurc"]
    b200 = json.load(open(os.path.join(ROOT, "exp", "out", "exp79_seeds", "olmoearth_base.json")))
    rtx = json.load(open(os.path.join(ROOT, "exp", "out", "exp79_engine", "olmoearth_base_rtx.json")))
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))["results"]["tasks"]
    assert set(b200["tasks"]) == set(rec) and len(rec) == 24
    assert max(abs(v["seeds"][0]["test_accuracy"] - rec[t]["test_accuracy"]) for t, v in b200["tasks"].items()) < 1e-4
    mins = {}
    for t, v in b200["tasks"].items():
        leads = [lead(s["signals"]) for s in v["seeds"]]
        assert len(leads) == 10 and all(abs(l - s["margin_lead"]) < 1e-12 for l, s in zip(leads, v["seeds"]))
        assert min(leads) > 0
        mins[t] = (min(leads), int(np.argmin(leads)), float(np.median(leads)))
    assert round(mins["cropharvest_Togo_12_sentinel1"][0], 4) == 0.0017 and mins["cropharvest_Togo_12_sentinel1"][1] == 5
    assert round(mins["cropharvest_Togo_12_sentinel1"][2], 3) == 0.018
    assert sorted(mins, key=lambda t: mins[t][0])[:4] == ["cropharvest_Togo_12_sentinel1", "nandi_sentinel1", "m_eurosat", "mados"]
    # the engine: accuracy differences per family and the RTX gate failures, from the two exports directly
    cls = {t: max(abs(a["test_accuracy"] - b["test_accuracy"]) for a, b in zip(rtx["tasks"][t]["seeds"], b200["tasks"][t]["seeds"]))
           for t in rec if b200["tasks"][t]["family"].startswith("cla")}
    seg = {t: max(abs(a["test_accuracy"] - b["test_accuracy"]) for a, b in zip(rtx["tasks"][t]["seeds"], b200["tasks"][t]["seeds"]))
           for t in rec if b200["tasks"][t]["family"].startswith("seg")}
    assert round(max(cls.values()), 3) == 0.015 and max(cls, key=cls.get) == "awf_sentinel1"
    assert max(seg.values()) < 1e-4
    rtx_fail = sorted(t for t, v in rtx["tasks"].items() if abs(v["seeds"][0]["test_accuracy"] - rec[t]["test_accuracy"]) >= 1e-4)
    assert rtx_fail == ["awf_sentinel1", "cropharvest_Peoples_Republic_of_China_6_sentinel1", "m_eurosat", "m_forestnet", "m_so2sat", "nandi_landsat", "nandi_sentinel1"]
    wins_same = all((lead(a["signals"]) > 0) == (lead(b["signals"]) > 0) for t in rec for a, b in zip(rtx["tasks"][t]["seeds"], b200["tasks"][t]["seeds"]))
    assert wins_same
    eng = json.load(open(os.path.join(ROOT, "exp", "out", "exp79_engine", "summary.json")))
    assert eng["classification"]["tasks_gate_failed_rtx"] == rtx_fail and eng["classification"]["tasks_gate_failed_b200"] == []


EXP79_ENCODERS = ["anysat", "clay_large", "copernicusfm", "croma_base", "croma_large", "galileo_base", "galileo_nano",
                  "galileo_tiny", "olmoearth_base", "olmoearth_large", "olmoearth_nano", "olmoearth_tiny", "panopticon",
                  "satlas_base", "terramind_base", "terramind_large"]


def _exp79_exports():
    import json
    out = {}
    for e in EXP79_ENCODERS:
        p = os.path.join(ROOT, "exp", "out", "exp79_seeds", f"{e}.json")
        if not os.path.exists(p):
            pytest.skip(f"exp79 export for {e} not present")
        out[e] = json.load(open(p))
    return out


def test_exp79_all_sixteen_gate_p1_and_p3_recompute_by_a_second_route():
    """exp79's final reading by a route the experiment does not use: the gate against both records, the exact
    one-sided sign test at every (encoder, seed) with integer binomials, and the per-seed median headroom with the
    perfect ranker's AURC from its harmonic-number closed form, AURC* = (k - (n - k)(H_n - H_{n-k})) / n, rather
    than the package's mean over ranks."""
    import json
    import math
    E = _exp79_exports()
    r74 = json.load(open(os.path.join(ROOT, "exp", "out", "exp74_summary.json")))["results"]["tasks"]
    r70 = json.load(open(os.path.join(ROOT, "exp", "out", "exp70_summary.json")))["results"]["tasks"]
    lead = lambda s: min(s["signals"]["ctl_embedding_distance"]["excess_aurc"], s["signals"]["ctl_class_rarity"]["excess_aurc"]) - s["signals"]["margin"]["excess_aurc"]
    # the gate
    failing = {}
    for e, d in E.items():
        rec = r70 if e == "olmoearth_base" else r74[e]
        assert set(d["tasks"]) == set(rec)
        diff = {t: abs(v["seeds"][0]["test_accuracy"] - rec[t]["test_accuracy"]) for t, v in d["tasks"].items()}
        bad = sorted(t for t, x in diff.items() if x >= 1e-4)
        if bad:
            failing[e] = bad
        else:
            assert max(diff.values()) < 1e-10, e
    assert set(failing) == {"copernicusfm", "croma_base", "croma_large", "terramind_base", "terramind_large"}
    assert {t for v in failing.values() for t in v} <= {"pastis_sentinel1", "pastis_sentinel1_sentinel2", "m_cashew_plant"}
    # P1 with the exact sign test
    for e, d in E.items():
        for s in range(10):
            L = [lead(v["seeds"][s]) for v in d["tasks"].values()]
            w, l = sum(x > 0 for x in L), sum(x < 0 for x in L)
            p = sum(math.comb(w + l, k) for k in range(w, w + l + 1)) / 2 ** (w + l)
            assert w / len(L) >= 0.75 and p < 0.05, (e, s, w, len(L), p)
    # P3 with an independent oracle
    nmax = max(v["n_units"] for d in E.values() for v in d["tasks"].values())
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, nmax + 1))])
    oracle = lambda n, k: (k - (n - k) * (H[n] - H[n - k])) / n
    assert abs(oracle(1000, 85) - metrics.oracle_aurc(1000, 85)) < 1e-12        # the closed form is the same quantity
    rec_h = json.load(open(os.path.join(ROOT, "exp", "out", "headroom_by_encoder.json")))["per_encoder_median_headroom"]
    ranks = lambda h: {e: sorted(h, key=h.get).index(e) for e in h}
    rows = []
    for s in range(10):
        med = {}
        for e, d in E.items():
            hs = []
            for v in d["tasks"].values():
                r = v["seeds"][s]
                n, er = r["n_units"], r["error_rate"]
                gap = er - oracle(n, int(round(er * n)))
                if gap > 0:
                    hs.append(1 - r["signals"]["margin"]["excess_aurc"] / gap)
            med[e] = float(np.median(hs))
        a, b = ranks(rec_h), ranks(med)
        rho = 1 - 6 * sum((a[e] - b[e]) ** 2 for e in med) / (16 * (16 ** 2 - 1))
        order = sorted(med, key=med.get, reverse=True)
        rows.append((rho, order[0], order[-1], med[order[0]] - med[order[1]]))
    assert round(min(r[0] for r in rows), 3) == 0.947
    assert [r[1] for r in rows].count("olmoearth_large") == 8 and [r[1] for r in rows].count("olmoearth_base") == 2
    assert all(r[2] == "satlas_base" for r in rows) and round(min(r[3] for r in rows), 4) == 0.0002


def test_exp79_monte_carlo_coverage_agrees_with_exact_enumeration_on_every_cell():
    """P5's 111 Monte Carlo coverages against the exact hypergeometric coverage of the same Wilson interval at each
    cell's (N, K, 300): within four standard errors of 2,000 draws on every cell, so the harness drew what it says.
    And the package's current random-draw interval, the exact hypergeometric one, covers at least 0.95 on each."""
    import json
    from oe_inferencex import estimate as est
    per = json.load(open(os.path.join(ROOT, "exp", "out", "exp79_summary.json")))["estimation"]["per_encoder"]
    cells = [(e, t, r) for e, rows in per.items() for t, r in rows.items()]
    assert len(cells) == 111
    for e, t, r in cells:
        N = r["n_units"]; K = int(round(r["error_rate"] * N))
        ex = est.exact_coverage_srs(N, K, 300)
        se = (ex * (1 - ex) / 2000) ** 0.5
        assert abs(r["srs_coverage"] - ex) <= 4 * se, (e, t, r["srs_coverage"], ex)
        assert est.exact_coverage_srs(N, K, 300, interval=est.hypergeom_interval) >= 0.95, (e, t)
