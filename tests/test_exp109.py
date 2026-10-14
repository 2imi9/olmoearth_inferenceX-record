"""exp109 (report-only): the confident third, the STRICT-only errors on the one contested pair, the AUROCs and the
in-distribution grass/shrub ranking, rebuilt from the committed units files by code written here (ranks rather than
quantiles, labels and classes by name rather than the crosswalk, pairwise AUROCs on p1 rather than the package's
metric on the suspicion); and the summary holds no identifier."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp109_transfer_diagnosis as e109  # noqa: E402

SUMMARY = os.path.join(ROOT, "exp", "out", "exp109_summary.json")


def _pair_auroc(p1, err, w):
    """Weighted AUROC of low p1 for the errors, by every error/correct pair (ties half)."""
    P, N = np.flatnonzero(err), np.flatnonzero(~err)
    s = -p1
    gt = (s[P][:, None] > s[N][None, :]) + 0.5 * (s[P][:, None] == s[N][None, :])
    return float((w[P][:, None] * w[N][None, :] * gt).sum() / (w[P].sum() * w[N].sum()))


def _plots(rel):
    d = np.load(os.path.join(ROOT, rel), allow_pickle=False)
    k = d["covered"].astype(bool)
    names = [str(x) for x in d["label_names"]]
    return {"label": np.array([names[i] for i in d["label"][k]]), "pred": d["pred"][k], "p1": d["p1"][k],
            "w": d["weight"][k], "country": d["country"][k],
            "es": d["wrong_strict"][k] > 0, "el": d["wrong_lenient"][k] > 0}


def test_the_confident_third_and_the_aurocs_on_constructed_plots():
    p1 = np.array([0.5, 0.9, 0.99, 0.6, 0.999, 0.7])
    conf, cut = e109.confident_third(p1)
    assert cut == pytest.approx(np.quantile(p1, 2 / 3)) and conf.tolist() == [False, False, True, False, True, False]
    rng = np.random.default_rng(1)
    n = 120
    es = rng.random(n) < 0.4
    el = es & (rng.random(n) < 0.5)                         # LENIENT's errors are a subset of STRICT's
    p1 = rng.choice(np.linspace(0.3, 1.0, 15), n)           # heavy ties
    t = {"u": -np.log(p1), "weight": rng.uniform(0.5, 2, n), "es": es, "el": el}
    a = e109.auroc_set(t)
    w = t["weight"]
    assert a["strict"] == pytest.approx(_pair_auroc(p1, es, w), abs=1e-12)
    keep = el | ~es
    assert a["lenient_errors_vs_strict_correct"] == pytest.approx(_pair_auroc(p1[keep], el[keep], w[keep]), abs=1e-12)
    keep = ~el
    assert a["strict_only_errors_vs_strict_correct"] == pytest.approx(_pair_auroc(p1[keep], es[keep], w[keep]), abs=1e-12)


def test_exp109_recorded_by_a_second_route():
    """The key counts, the weighted error rates of the confident third, the share at other cuts, the four AUROCs at the
    TimeSync plots with the STRICT-only one's within-country interval, the AUROCs at Ai2's points (all, and the
    grass/shrub boundary), and the two losses from those AUROCs, from the units files with code written here."""
    S = json.load(open(SUMMARY, encoding="utf-8"))
    assert S["preregistered"] is False
    expect = {"v1": (91, 20, 14, 11, 6), "v1.2": (91, 22, 20, 19, 2)}
    # other cuts, the plots by rank (top half, 40%, quarter, fifth) -> (STRICT errors, STRICT-only)
    by_cut = {"v1": {136: (32, 22), 109: (23, 15), 68: (15, 12), 55: (11, 10)},
              "v1.2": {136: (32, 24), 109: (24, 20), 68: (15, 14), 55: (12, 12)}}
    second = {}
    for m, rel in e109.TRANSFER.items():
        t = _plots(rel)
        n = t["p1"].size
        assert n == 271 and np.unique(t["p1"]).size == n, "distinct confidences: the third is a rank cut"
        k = n - (n - 1) * 2 // 3                            # the 2/3 quantile of 271 values is the 181st smallest
        conf = np.zeros(n, bool)
        conf[np.argsort(-t["p1"])[:k]] = True
        so = t["es"] & ~t["el"]
        og_shrub = conf & (t["label"] == "Open Grassland") & (t["pred"] == 2)   # channel 2: shrubland_savanna
        got = (int(conf.sum()), int((conf & t["es"]).sum()), int((conf & so).sum()), int(og_shrub.sum()),
               int((conf & t["el"]).sum()))
        assert got == expect[m], (m, got)
        assert not t["el"][og_shrub].any() and t["es"][og_shrub].all(), "the contested pair is STRICT-only"
        C = S["models"][m]["transfer_2017"]["confident_third"]
        assert (C["plots"], C["strict_errors"], C["strict_only"], C["shrubland_savanna_on_open_grassland"],
                C["lenient_errors"]) == got
        w = t["w"]
        for rule, e in (("strict", t["es"]), ("lenient", t["el"])):
            assert C["weighted_error_rate"][rule] == pytest.approx((w * e)[conf].sum() / w[conf].sum(), abs=1e-12)
        order = np.argsort(-t["p1"])
        recorded = {v["plots"]: (v["strict_errors"], v["strict_only"])
                    for v in S["models"][m]["transfer_2017"]["strict_only_by_cut"].values()}
        for kc, want_cut in by_cut[m].items():
            top = np.zeros(n, bool)
            top[order[:kc]] = True
            got_cut = (int((top & t["es"]).sum()), int((top & so).sum()))
            assert got_cut == want_cut == recorded[kc], (m, kc, got_cut)
        # the four AUROCs (pairwise on p1): STRICT and LENIENT over all plots; LENIENT errors, and STRICT-only errors,
        # each against the STRICT-correct plots
        A = S["models"][m]["transfer_2017"]["auroc_weighted"]
        keep_l = t["el"] | ~t["es"]
        keep = ~t["el"]
        two = {"strict": _pair_auroc(t["p1"], t["es"], w), "lenient": _pair_auroc(t["p1"], t["el"], w),
               "lenient_errors_vs_strict_correct": _pair_auroc(t["p1"][keep_l], t["el"][keep_l], w[keep_l]),
               "strict_only_errors_vs_strict_correct": _pair_auroc(t["p1"][keep], t["es"][keep], w[keep])}
        for key, v in two.items():
            assert A[key] == pytest.approx(v, abs=1e-12), (m, key)
        second[m] = two
        # the STRICT-only AUROC's within-country interval (the same resampling stream, the pairwise AUROC on p1)
        rng = np.random.default_rng(0)
        groups = [np.flatnonzero(t["country"] == g) for g in np.unique(t["country"])]
        draws = []
        for _ in range(2000):
            ix = np.concatenate([rng.choice(g, g.size, replace=True) for g in groups])
            kk = ~t["el"][ix]
            draws.append(_pair_auroc(t["p1"][ix][kk], t["es"][ix][kk], w[ix][kk]))
        lo, hi = np.quantile(draws, [0.05, 0.95])
        B = S["models"][m]["transfer_2017"]["bootstrap"]["strict_only_errors_vs_strict_correct"]
        assert B == pytest.approx([lo, hi], abs=1e-12)
    want = {"v1": (146, 8, 3, 5), "v1.2": (142, 8, 2, 6)}
    for m, rel in e109.AI2_POINTS.items():
        d = np.load(os.path.join(ROOT, rel), allow_pickle=False)
        k = d["covered"].astype(bool) & (d["split"] == "val")
        pred, lab, p1 = d["pred_deployed"][k], d["label"][k], d["p1"][k]
        assert k.sum() == 259
        gs = ((pred == 2) | (pred == 4)) & ((lab == 2) | (lab == 4))   # shrubland_savanna 2, grassland_barren 4
        e = pred != lab
        got = (int(gs.sum()), int((gs & e).sum()), int((gs & e & (pred == 2)).sum()), int((gs & e & (pred == 4)).sum()))
        assert got == want[m], (m, got)
        G = S["models"][m]["ai2_validation_points"]["grass_shrub"]
        assert (G["points"], G["errors"], G["shrubland_savanna_where_ai2_labels_grassland_barren"],
                G["grassland_barren_where_ai2_labels_shrubland_savanna"]) == got
        assert G["auroc_unweighted"] == pytest.approx(_pair_auroc(p1[gs], e[gs], np.ones(gs.sum())), abs=1e-12)
        ai2_all = _pair_auroc(p1, e, np.ones(e.size))
        assert S["models"][m]["ai2_validation_points"]["auroc_unweighted"] == pytest.approx(ai2_all, abs=1e-12)
        # the two losses, from the AUROCs recomputed here rather than from the summary's own fields
        L, two = S["models"][m]["losses"], second[m]
        assert L["transfer_and_point_placement"] == pytest.approx(
            ai2_all - two["lenient_errors_vs_strict_correct"], abs=1e-12)
        assert L["legend"] == pytest.approx(two["lenient_errors_vs_strict_correct"] - two["strict"], abs=1e-12)
    assert [round(second[m]["lenient_errors_vs_strict_correct"], 3) for m in ("v1", "v1.2")] == [0.698, 0.741]
    assert [round(S["models"][m]["losses"][k], 3) for m in ("v1", "v1.2")
            for k in ("transfer_and_point_placement", "legend")] == [0.151, 0.06, 0.073, 0.099]


def test_the_summary_holds_no_identifier():
    text = open(SUMMARY, encoding="utf-8").read()
    names = np.load(os.path.join(ROOT, "exp", "out", "exp98_units.npz"), allow_pickle=False)["names"]
    assert not any(str(x) in text for x in names)
    for key in ("lat", "lon", "pid", "plot_id", "coordinates"):
        assert f'"{key}"' not in text
