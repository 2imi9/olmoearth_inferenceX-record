"""The package reproduces numbers recorded in exp/out from the committed artifacts alone (no encoder needed)."""
import collections
import csv
import json
import os

import numpy as np
import pytest

from oe_inferencex.metrics import (aurc_expected, capture_at_budget, capture_at_budget_expected, excess_aurc,
                                   expected_calibration_error, oracle_aurc, risk_coverage, selective_accuracy)
from oe_inferencex.signals import boundary_indicator, confidence
from oe_inferencex.stats import cluster_bootstrap_difference, sign_test

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def test_exp02_kazungula_aurc_and_error_rate():
    """exp/NOTES.md, exp02: E_case AURC 0.0011 vs max-softmax baseline 0.0009; Base error rate 2.1%."""
    z = np.load(_need("exp02_cache.npz"), allow_pickle=True)
    p_nano, p_base, lab = z["p_nano"], z["p_base"], z["ev_labels"]
    errors = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
    e_case = np.abs(p_nano - p_base)
    baseline = 1 - np.maximum(p_base, 1 - p_base)
    assert round(risk_coverage(e_case, errors)[2], 4) == 0.0011
    assert round(risk_coverage(baseline, errors)[2], 4) == 0.0009
    assert round(float(errors.mean()), 3) == 0.021
    # recipe item 1: the negative absolute logit orders these patches exactly as the saturating form does here
    with np.errstate(divide="ignore"):
        p64 = p_base.astype(np.float64)
        logit = np.log(p64) - np.log1p(-p64)
    assert aurc_expected(confidence(logit), errors) == pytest.approx(aurc_expected(baseline, errors), abs=1e-9)
    assert 0.0 <= boundary_indicator(p_base).mean() <= 1.0


def test_exp13_table_is_internally_consistent_with_the_oracle():
    """Every exp13 row: aurc - eaurc equals the closed-form oracle for 1024 patches and that scene's error count."""
    rows = list(csv.DictReader(open(_need("exp13_corrected_stats.csv"))))
    assert len(rows) == 27 * 6
    for r in rows:
        assert float(r["aurc"]) - float(r["eaurc"]) == pytest.approx(oracle_aurc(1024, int(r["n_errors"])), abs=1e-9)


@pytest.mark.parametrize("name", ["exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv",
                                  "exp36_dihedral_lexicographic.csv"])
def test_constant_score_rows_equal_error_rate_minus_oracle(name):
    """A constant score ties every unit, so its tie-aware E-AURC is the error rate minus the oracle (parts A and B)."""
    rows = [r for r in csv.DictReader(open(_need(name))) if r["signal"] == "constant score"]
    assert len(rows) >= 27
    for r in rows:
        n, k = int(r["n_patches"]), int(r["n_errors"])
        assert float(r["eaurc"]) == pytest.approx(k / n - oracle_aurc(n, k), abs=1e-9)
        e = np.r_[np.ones(k), np.zeros(n - k)]
        assert float(r["aurc"]) == pytest.approx(aurc_expected(np.zeros(n), e), abs=1e-9)
        assert excess_aurc(np.zeros(n), e) == pytest.approx(float(r["eaurc"]), abs=1e-9)


def test_exp28_and_exp30_share_the_error_set_and_the_reference_signals():
    """exp30 and exp31 run on the exp28 scaffolding: identical errors and identical reference E-AURCs per scene."""
    tabs = {}
    for name in ("exp28_decoder_consistency.csv", "exp30_laplace_head.csv", "exp31_feature_typicality.csv"):
        tabs[name] = {(r["unit"], r["signal"]): (int(r["n_errors"]), float(r["eaurc"]))
                      for r in csv.DictReader(open(_need(name))) if r["part"] == "A"}
    for sig in ("confidence (baseline)", "tile-phase (aligned)", "boundary indicator", "control NDWI gradient",
                "constant score", "control S2 patch variance", "control NDWI level"):
        for unit in {u for u, s in tabs["exp28_decoder_consistency.csv"] if s == sig}:
            vals = [tabs[n][(unit, sig)] for n in tabs]
            assert vals[0][0] == vals[1][0] == vals[2][0]
            assert vals[0][1] == pytest.approx(vals[1][1], abs=1e-12) and vals[0][1] == pytest.approx(vals[2][1], abs=1e-12)


def test_recorded_prereg_river_tests_are_reproducible_from_the_summaries():
    """The one-sided exact sign test in each summary equals the package's on the stored per-river means."""
    from oe_inferencex.stats import clustered_sign_test
    for name in ("exp28_summary.json", "exp30_summary.json", "exp31_summary.json", "exp36_summary.json"):
        s = json.load(open(_need(name)))
        pr = s["part_a"]["prereg"]["combination_gain_over_confidence"]
        recomputed = clustered_sign_test(pr["per_river"], {})
        assert (recomputed["w"], recomputed["l"], recomputed["t"]) == (pr["w"], pr["l"], pr["t"])
        assert recomputed["p"] == pytest.approx(pr["one_sided_p"], abs=1e-12)



def test_exp21_fine_tuned_model_metrics_from_the_per_window_table():
    """exp21_summary.json (16-px crops) reproduced from exp21_finetuned_awf.csv: AURCs, ECE, selective accuracy,
    error capture at the review budgets and the cluster bootstrap of tiling instability against confidence."""
    rows = [r for r in csv.DictReader(open(_need("exp21_finetuned_awf.csv"))) if r["crop"] == "16"]
    ref = json.load(open(_need("exp21_summary.json")))["crops"]["16"]
    err = np.array([float(r["error"]) for r in rows])
    conf = -np.array([float(r["logit_margin"]) for r in rows])
    tile = np.array([float(r["tile_phase"]) for r in rows])
    bnd = np.array([float(r["boundary"]) for r in rows])
    assert len(rows) == 344 and int(err.sum()) == ref["n_errors"]
    assert aurc_expected(conf, err) == pytest.approx(ref["aurc"]["confidence (neg logit margin)"], abs=1e-12)
    assert aurc_expected(tile, err) == pytest.approx(ref["aurc"]["tiling instability (aligned)"], abs=1e-12)
    assert aurc_expected(bnd, err) == pytest.approx(ref["aurc"]["boundary indicator"], abs=1e-12)
    assert oracle_aurc(len(err), int(err.sum())) == pytest.approx(ref["aurc_oracle"], abs=1e-12)
    top1 = np.array([float(r["top1_prob"]) for r in rows])
    ece, reliability = expected_calibration_error(top1, 1 - err)
    assert ece == pytest.approx(ref["ece_10bins"], abs=1e-12) and round(ece, 3) == 0.080
    # the top bin of the reliability table: 299 windows claiming 0.99 and 0.93 accurate (accuracy-needs-coverage)
    assert reliability[-1][2] == 299 and round(reliability[-1][3], 2) == 0.99 and round(reliability[-1][4], 2) == 0.93
    assert round(1 - float(err.mean()), 3) == 0.881 == round(ref["accuracy"], 3)
    sel = selective_accuracy(conf, 1 - err)
    assert all(sel[c] == pytest.approx(ref["selective_accuracy_at_coverage"][str(c)], abs=1e-12) for c in (0.5, 0.8, 0.9, 1.0))
    assert round(sel[0.8], 3) == 0.945 and round(sel[0.9], 3) == 0.919
    cap = capture_at_budget(tile, err, budgets=(0.05, 0.1, 0.2))
    assert all(cap[b] == pytest.approx(ref["error_capture_at_budget"][str(b)]["tiling instability (aligned)"], abs=1e-12)
               for b in (0.05, 0.1, 0.2))
    clusters = np.array([r["task"] for r in rows])
    boots = ref["bootstrap_vs_confidence (lo, hi, P(signal better))"]
    lo, hi, p_better = cluster_bootstrap_difference(tile, conf, err, clusters)
    assert (lo, hi, p_better) == pytest.approx(boots["tiling instability (aligned)"], abs=1e-12)
    assert lo < 0 < hi                                                     # tiling instability is indistinguishable from confidence
    # and every other signal is significantly worse than confidence: its AURC and its bootstrap, from the same rows
    for col, name in (("boundary", "boundary indicator"), ("probe_disagree", "probe disagreement"), ("control_ndvi_tstd", "control (NDVI temporal std)")):
        sig = np.array([float(r[col]) for r in rows])
        assert aurc_expected(sig, err) == pytest.approx(ref["aurc"][name], abs=1e-12)
        b = cluster_bootstrap_difference(sig, conf, err, clusters)
        assert b == pytest.approx(boots[name], abs=1e-12) and b[0] > 0, name
    assert round(aurc_expected(conf, err), 4) == 0.0262 and round(aurc_expected(tile, err), 4) == 0.0235


def test_exp36_preregistered_budget_tests_are_reproducible_from_the_summary():
    """The lexicographic rule's one-sided per-tile sign tests at the 5% and 10% budgets (exp36, run of record 716511)."""
    s = json.load(open(os.path.join(OUT, "exp36_summary.json")))
    pr = s["part_b"]["prereg_lexicographic"]
    assert set(pr) == {"0.05", "0.1"}
    for b, r in pr.items():
        assert r["one_sided"] and r["w"] + r["l"] + r["t"] == s["part_b"]["n_tiles_scored"] == 351
        assert r["sign_p"] == pytest.approx(sign_test(r["w"], r["l"], "greater"), rel=1e-9)
        assert r["pooled"] > r["pooled_confidence"] and r["boot_lo"] > 0
    assert (pr["0.05"]["w"], pr["0.05"]["l"], pr["0.05"]["t"]) == (85, 31, 235)
    assert (pr["0.1"]["w"], pr["0.1"]["l"], pr["0.1"]["t"]) == (112, 48, 191)
    awf = s["part_awf"]["crops"]
    for crop in ("16", "32"):                                   # the two orders pick the same 5% review set on the AWF model
        r = awf[crop]["lexicographic_vs_confidence"]["0.05"]
        assert r["capture_lex"] == pytest.approx(r["capture_confidence"]) and r["boot_lo"] == pytest.approx(0.0)


def test_exp37_cue_shares_recompute_from_the_per_window_tables():
    """exp37's pooled shares among error and correct windows recompute from its per-window tables, with the cues
    derived as the experiment defines them (boundary > 0; |NDWI| < 0.1; top 20% per unit, ties included)."""
    from oe_inferencex.explain import cue_enrichment, top_fraction
    s = json.load(open(os.path.join(OUT, "exp37_summary.json")))
    for part, name, unit_col in (("part_a", "exp37_patches_scenes.npz", "scene"), ("part_b", "exp37_patches_bolivia.npz", "tile")):
        z = np.load(os.path.join(OUT, name))
        err = z["err"] > 0.5
        units = z[unit_col]
        cues = {"boundary": z["boundary"] > 0, "ndwi_ambiguous": z["ndwi_level"] > -0.1,
                "low_confidence": np.zeros(len(err), bool), "unstable": np.zeros(len(err), bool), "dihedral_disagree": np.zeros(len(err), bool)}
        for u in np.unique(units):
            sel = units == u
            for cue, col in (("low_confidence", "conf"), ("unstable", "tile_phase"), ("dihedral_disagree", "dihedral")):
                cues[cue][sel] = top_fraction(z[col][sel], 0.2)
        rec = s[part]["analysis"]["cues"]
        assert rec["boundary"]["n"] == len(err) and rec["boundary"]["n_errors"] == int(err.sum())
        for cue in rec:
            r = cue_enrichment(cues[cue], err, n_boot=0)
            assert r["share_errors"] == pytest.approx(rec[cue]["share_errors"], abs=1e-12)
            assert r["share_correct"] == pytest.approx(rec[cue]["share_correct"], abs=1e-12)
            assert rec[cue]["boot_lo"] <= rec[cue]["enrichment"] <= rec[cue]["boot_hi"]


def test_exp37_review_sets_cue_coverage_and_the_boundary_first_counts_recompute_from_the_per_window_table():
    """The numbers five claims pin on Bolivia hand labels, reached from exp37_patches_bolivia.npz alone rather than
    from the summaries that recorded them: the boundary shares 0.750 / 0.214, the four-cue table with its 7.2x NDWI
    enrichment, the 95% of error windows carrying a cue, the error rate inside confidence's review set (0.38, 0.33
    and 0.26 at 5, 10 and 20% against a base rate of 0.088) and the cue error rates inside the 5% set, and exp36's
    boundary-first order: its per-tile counts (85/31/235 and 112/48/191 on the 351 tiles with 3 <= errors <= n - 3,
    the exp18 rule) and its pooled captures at the three budgets, which exp38 had reached by the same route."""
    from oe_inferencex.assess import review_mask
    from oe_inferencex.explain import cue_enrichment, top_fraction
    from oe_inferencex.signals import midrank_pct
    from oe_inferencex.stats import wins_losses_ties
    z = np.load(os.path.join(OUT, "exp37_patches_bolivia.npz"))
    err = z["err"] > 0.5
    tiles = z["tile"]
    assert len(err) == 81984 and int(err.sum()) == 7248 and round(float(err.mean()), 3) == 0.088
    cues = {"boundary": z["boundary"] > 0, "ndwi_ambiguous": z["ndwi_level"] > -0.1,
            "low_confidence": np.zeros(len(err), bool), "unstable": np.zeros(len(err), bool), "dihedral_disagree": np.zeros(len(err), bool)}
    review = {b: np.zeros(len(err), bool) for b in (0.05, 0.1, 0.2)}
    caps = {}
    for u in np.unique(tiles):
        sel = tiles == u
        # the table lists a tile's valid windows in raster order, so the assessor's tie rule (descending raster
        # position) applies to the sub-array exactly as review_mask applied it to the tile
        assert np.all(np.diff(z["row"][sel] * 100 + z["col"][sel]) > 0)
        for cue, col in (("low_confidence", "conf"), ("unstable", "tile_phase"), ("dihedral_disagree", "dihedral")):
            cues[cue][sel] = top_fraction(z[col][sel], 0.2)
        for b in review:
            review[b][sel] = review_mask(z["conf"][sel], np.ones(int(sel.sum()), bool), b)
        e = err[sel].astype(float)
        if 3 <= e.sum() <= sel.sum() - 3:
            mr = midrank_pct(z["conf"][sel])
            caps[u] = (capture_at_budget_expected(mr, e, (0.05, 0.1, 0.2)),
                       capture_at_budget_expected(2.0 * (z["boundary"][sel] > 0) + mr, e, (0.05, 0.1, 0.2)))
    # errors-sit-on-boundaries and cue-enrichment-table
    table = {k: cue_enrichment(cues[k], err, n_boot=0) for k in cues}
    assert (round(table["boundary"]["share_errors"], 3), round(table["boundary"]["share_correct"], 3)) == (0.750, 0.214)
    assert {k: (round(v["share_errors"], 3), round(v["share_correct"], 3), round(v["enrichment"], 1), round(v["precision"], 2))
            for k, v in table.items() if k != "dihedral_disagree"} == {
        "boundary": (0.750, 0.214, 3.5, 0.25), "low_confidence": (0.589, 0.163, 3.6, 0.26),
        "unstable": (0.583, 0.164, 3.6, 0.26), "ndwi_ambiguous": (0.483, 0.067, 7.2, 0.41)}
    assert all(3.4 < table[k]["enrichment"] < 3.7 for k in ("boundary", "low_confidence", "unstable", "dihedral_disagree"))
    # explanation-cues-cover-errors: 95% of the error windows carry a cue; inside the 5% set the boundary and the
    # NDWI cue separate the error rates and the two instability cues do not
    n_cues = np.stack(list(cues.values())).sum(0)
    assert round(1 - float((n_cues[err] == 0).mean()), 2) == 0.95
    m = review[0.05]
    rate = lambda k, inside: float(err[(cues[k] if inside else ~cues[k]) & m].mean())
    assert round(rate("boundary", True), 2) == 0.46 and abs(rate("boundary", False) - 0.155) < 0.006
    assert round(rate("ndwi_ambiguous", True), 2) == 0.51 and round(rate("ndwi_ambiguous", False), 2) == 0.33
    assert all(abs(rate(k, True) - rate(k, False)) < 0.05 for k in ("unstable", "dihedral_disagree"))
    # review-set-error-rate
    assert [int(review[b].sum()) for b in (0.05, 0.1, 0.2)] == [4043, 8075, 16402]
    assert [round(float(err[review[b]].mean()), 2) for b in (0.05, 0.1, 0.2)] == [0.38, 0.33, 0.26]
    # boundary-first-review-order: exp36's per-tile counts, and its pooled captures under one global midrank
    assert len(caps) == 351
    counts = {b: wins_losses_ties(np.array([caps[u][1][b] - caps[u][0][b] for u in caps])) for b in (0.05, 0.1, 0.2)}
    assert counts == {0.05: (85, 31, 235), 0.1: (112, 48, 191), 0.2: (121, 77, 153)}
    assert sign_test(85, 31, "greater") < 1e-6 and sign_test(112, 48, "greater") < 1e-6
    mr = midrank_pct(z["conf"])
    lex = capture_at_budget_expected(2.0 * (z["boundary"] > 0) + mr, err, (0.05, 0.1, 0.2))
    conf = capture_at_budget_expected(mr, err, (0.05, 0.1, 0.2))
    assert [round(lex[b], 3) for b in (0.05, 0.1, 0.2)] == [0.274, 0.494, 0.732]
    assert [round(conf[b], 3) for b in (0.05, 0.1, 0.2)] == [0.259, 0.465, 0.749]
    assert lex[0.05] > conf[0.05] and lex[0.1] > conf[0.1] and lex[0.2] < conf[0.2]     # ahead at 5 and 10%, behind at 20%
    assert excess_aurc(2.0 * (z["boundary"] > 0) + mr, err) > excess_aurc(z["conf"], err)   # and a worse ranker over the whole set


def test_exp38_prereg_tests_and_exp36_consistency_from_the_summary():
    """exp38 (CPU, on exp37's tables): the exp36 consistency check holds and the preregistered per-tile p-values
    recompute from the counts; the pooled bootstrap at the preregistered budgets does not exclude zero (mixed)."""
    s = json.load(open(os.path.join(OUT, "exp38_summary.json")))
    chk = s["part_b"]["exp36_consistency"]
    assert all(chk[b]["match"] for b in ("0.05", "0.1"))
    assert tuple(chk["0.05"]["recomputed"]) == (85, 31, 235) and tuple(chk["0.1"]["recomputed"]) == (112, 48, 191)
    for b in ("0.05", "0.1"):
        pt, po = s["part_b"]["prereg"][b]["per_tile"], s["part_b"]["prereg"][b]["pooled"]
        assert pt["one_sided"] and pt["n_units"] == 351
        assert pt["sign_p"] == pytest.approx(sign_test(pt["w"], pt["l"], "greater"), rel=1e-9)
        assert pt["w"] > pt["l"] and pt["sign_p"] < 0.001
        assert po["boot_lo"] <= 0 <= po["boot_hi"]                           # the pooled gain is not established
    assert s["part_b"]["tests"]["NDWI-ambiguous, then boundary, then confidence vs boundary, then confidence"]["pooled"]["0.2"]["boot_lo"] > 0


def test_exp41_two_view_disagreement_is_a_recorded_negative():
    """exp41 (job 726248): outside partners' errors are as correlated with OlmoEarth's as the family's; the primary
    fails on both tasks."""
    s = json.load(open(os.path.join(OUT, "exp41_summary.json")))
    f = s["tasks"]["sen1floods11"]
    assert f["prereg"]["supported_vs_confidence"] is False and f["falsification_correlation"]["outside_below_family"] is False
    for m, c in f["error_correlation"].items():
        assert 0.75 < c["phi_test"] < 0.85 and c["p_err_given_oe_err_test"] > 0.79        # every model errs on the same windows
    pt = f["prereg"]["capture_10_per_unit"]
    assert (pt["w"], pt["l"], pt["t"]) == (281, 1263, 35) and pt["sign_p"] == pytest.approx(sign_test(281, 1263, "greater"), rel=1e-9)
    assert f["pooled_capture"]["weighted disagreement (selected partner)"]["0.1"] < f["pooled_capture"]["confidence (OlmoEarth)"]["0.1"]
    a = s["tasks"]["awf_sentinel2"]
    assert a["prereg"]["supported_vs_confidence"] is False and a["prereg"]["boot_p05"] < 0


def test_exp42_shift_averaged_window_is_supported_on_both_testbeds():
    """exp42 (job 726464): W1 beats the grid window on pixel accuracy per tile on Bolivia and on the test split."""
    s = json.load(open(os.path.join(OUT, "exp42_summary.json")))
    assert s["prereg"]["supported"] is True
    for name, counts in (("bolivia", (321, 66, 53)), ("test", (583, 78, 139))):
        r = s["results"][name]["prereg_W1_vs_W0_pixel"]
        assert r["one_sided"] and (r["w"], r["l"], r["t"]) == counts and r["sign_p"] == pytest.approx(sign_test(r["w"], r["l"], "greater"), rel=1e-9)
        acc = s["results"][name]["pixel_accuracy_mean"]
        assert acc["W1 shift-averaged"] - acc["W0 grid"] > 0.008
        pur = s["results"][name]["purity"]
        assert pur["share_of_windows_impure"] < 0.11 and pur["share_of_errors_on_impure_windows"] > 0.44


def test_exp43_segment_majority_breaks_more_than_it_corrects():
    """exp43 (job 727104): the corrected/broken accounting (E_g + C = E_f + B) on both testbeds; not supported."""
    s = json.load(open(os.path.join(OUT, "exp43_summary.json")))
    assert s["prereg"]["supported"] is False and s["prereg"]["complete"] is True
    for name, (C, B) in (("bolivia", (15093, 16273)), ("test", (24525, 30951))):
        r = s["results"][name]["primary_k8_hard"]
        assert (r["C_total"], r["B_total"]) == (C, B) and B > C and r["one_sided"] and r["sign_p"] > 0.5
        assert r["acc_candidate_mean"] - r["acc_w1_mean"] == pytest.approx(r["mean_gain"], abs=2e-3)
        pur = s["results"][name]["w1_errors_by_grid_window_purity"]
        assert 0.4 < pur["share_of_w1_errors_on_impure_windows"] < 0.6 and pur["share_of_pixels_in_impure_windows"] < 0.11


def test_exp43_mechanism_accounting_matches_the_theorem():
    """exp43 (job 727364): pure segments with a correct majority break nothing and pure segments with a wrong majority
    correct nothing, exactly as the Lean segment theorem says; mixed segments do most of the damage; the pooled and
    mean-tile endpoints are both recorded and differ."""
    s = json.load(open(os.path.join(OUT, "exp43_summary.json")))
    for name, broken_share_mixed in (("bolivia", 0.70), ("test", 0.85)):
        r = s["results"][name]
        m = r["mechanism_primary"]
        assert m["pure_correct"]["B"] == 0 and m["pure_wrong"]["C"] == 0
        p = r["primary_k8_hard"]
        assert m["mixed"]["B"] / p["B_total"] > broken_share_mixed
        assert p["pooled_gain"] == pytest.approx((p["C_total"] - p["B_total"]) / p["N_total"], rel=1e-9)
        assert p["pooled_gain"] != p["mean_gain"] and p["pooled_gain"] < 0
        assert r["secondary_k8_hard_twelve_bands_only"]["mean_gain"] < 0          # dropping NDWI does not rescue it
        assert 0.02 < r["block_constant_oracle_limit"]["rate"] < 0.04             # far below the grid's error rate
    assert s["results"]["bolivia"]["tiles_excluded_no_labelled_pixel_in_common_region"] == [47]


def test_exp45_v12_replication_and_cross_version_overlap():
    """exp45 (job 728864): on v1.2 Base the cues and the shift-averaged window replicate, confidence is no longer
    the best ranker on Bolivia, and a different architecture still errs on ~3/4 of v1's error windows."""
    s = json.load(open(os.path.join(OUT, "exp45_summary.json")))
    assert s["prereg"]["complete"] is True
    assert s["prereg"]["R3_cue_enrichment"] is True and s["prereg"]["R4_shift_averaged_window"] is True
    assert s["prereg"]["R1_best_ranker"] is False and s["prereg"]["R2_boundary_first"] is False
    bo = s["results"]["bolivia"]
    e = bo["R1_best_ranker"]["pooled_eaurc"]
    assert e["tile-phase"] < e["confidence"] and e["control NDWI level"] < e["confidence"]   # confidence is third
    assert s["results"]["test"]["R1_best_ranker"]["replicates"] is True                      # but still leads on the test split
    assert bo["R2_boundary_first"]["replicates"] is True
    for name in ("bolivia", "test"):
        r4 = s["results"][name]["R4_shift_averaged_window"]
        assert r4["replicates"] and r4["mean_gain"] > 0.009
        d = s["D1_cross_version_error_overlap"][name]
        assert 0.70 < d["p_v12_wrong_given_v1_wrong"] < 0.78
        assert d["p_v12_wrong_given_v1_right"] < 0.04 and 0.65 < d["phi"] < 0.75


def test_fine_tuning_moves_the_error_set_more_than_swapping_a_frozen_encoder():
    """The measurement that withdrew the "errors belong to the windows" gloss: on the same AWF points and labels,
    fine-tuning the encoder corrects most of the frozen probe's errors, while swapping a frozen encoder (exp41)
    barely moves the error set."""
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "exp21_finetuned_awf.csv"))) if r["crop"] == "16"]
    ft = np.array([float(r["error"]) for r in rows]) > 0.5
    pr = np.array([float(r["probe_error"]) for r in rows]) > 0.5
    n11, n10, n01, n00 = int((ft & pr).sum()), int((ft & ~pr).sum()), int((~ft & pr).sum()), int((~ft & ~pr).sum())
    assert (n11, n10, n01, n00) == (28, 13, 35, 268) and len(rows) == 344
    corrected = n01 / (n01 + n11)
    phi = (n11 * n00 - n10 * n01) / np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    assert corrected == pytest.approx(35 / 63, rel=1e-9) and corrected > 0.5
    assert phi == pytest.approx(0.4753, abs=5e-4)
    frozen = json.load(open(os.path.join(OUT, "exp41_summary.json")))["tasks"]["sen1floods11"]["error_correlation"]
    assert phi < min(c["phi_test"] for c in frozen.values()) - 0.25          # unfreezing moves it far more than swapping


def test_exp46_modality_moves_the_error_set_most():
    """exp46 (job 729376): the input modality dominates the readout and the backbone; impurity and chips do not explain it."""
    s = json.load(open(os.path.join(OUT, "exp46_summary.json")))
    assert s["prereg"]["P1_readout"] is True and s["prereg"]["P2_modality"] is True and s["prereg"]["complete"] is True
    for name in ("bolivia", "test"):
        r = s["results"][name]
        ref = r["prereg"]["backbone_swap_phi"]
        phi = {k.split("vs ")[1].split()[0]: v["phi"] for k, v in r["pairwise"].items() if k.startswith("A1")}
        assert phi["A2"] < 0.45 < ref                       # modality moves it far more than the backbone
        assert phi["A3"] < ref and phi["A4"] < ref          # readout, only just
        assert phi["A3"] > 0.6                              # and nowhere near fine-tuning's 0.475
        for st in r["purity_strata"].values():              # impurity does not explain the overlap
            assert abs(st["pure"] - st["mixed"]) < 0.2
    assert s["results"]["bolivia"]["accuracy"]["A0 no encoder"] > s["results"]["bolivia"]["accuracy"]["A1 S2 linear"]


def test_exp47_a_no_model_control_matches_confidence_on_bolivia():
    """exp47 (job 729376): on the shift-averaged decision the repository recommends, the NDWI-level control matches or
    beats the model's own confidence on Bolivia under both backbones; confidence always beats tiling instability."""
    s = json.load(open(os.path.join(OUT, "exp47_summary.json")))
    assert s["prereg"]["supported"] is False and s["prereg"]["complete"] is True
    for v in ("v1", "v1_2"):
        for n in ("bolivia", "test"):
            t = s["results"][v][n]["tests"]
            assert t["tile-phase"]["pooled_lead"] > 0.001 and t["tile-phase"]["sign_p"] < 0.05
            for k in ("U+ averaged confidence + tile-phase", "U+ averaged confidence + NDWI level"):
                assert t[k]["pooled_lead"] > 0                        # plain averaged confidence beats every combination
        assert s["results"][v]["bolivia"]["tests"]["control NDWI level"]["sign_p"] > 0.5   # loses the per-tile count
        e = s["results"][v]["bolivia"]["pooled_eaurc"]
        assert e["averaged confidence"] < e["grid confidence"]        # averaging helps ranking as well as accuracy
    assert s["results"]["v1_2"]["test"]["prereg_passes"] is True


def test_exp49_shrug_fm_signals_do_not_beat_confidence_but_a_label_fitted_fusion_does():
    """exp49 (job 731703): SHRUG-FM's signals under the window protocol. Preregistered and supported: under v1.2 the
    averaged confidence beats ensemble mutual information and -NCDD on both testbeds. Secondary: the bagged predictive
    entropy beats confidence on Bolivia under v1; the label-fitted linear fusion beats confidence on three of four arms;
    NCDD as the paper states it (cluster-normalized) is worse than the raw deficit everywhere."""
    s = json.load(open(os.path.join(OUT, "exp49_summary.json")))
    assert s["prereg"] == {"supported": True, "complete": True} and s["n_failures"] == 0
    for n in ("bolivia", "test"):
        t = s["results"]["v1_2"][n]["tests"]
        for k in ("ensemble mutual information", "embedding NCDD"):
            assert t[k]["pooled_lead"] >= 0.001 and t[k]["sign_p"] < 0.05 and t[k]["one_sided"]
    for v in ("v1", "v1_2"):
        for n in ("bolivia", "test"):
            e = s["results"][v][n]["pooled_eaurc"]
            assert e["embedding NCDD raw"] < e["embedding NCDD"]
            for k in ("embedding normalized distance", "input extremity max", "input extremity mean"):
                assert e[k] > 2.5 * e["averaged confidence"]
    b = s["results"]["v1"]["bolivia"]["tests"]["ensemble predictive entropy"]
    assert b["pooled_lead"] <= -0.001 and b["l"] > 2 * b["w"] and b["sign_p"] < 1e-6      # the bag ranks better under v1
    fus = {(v, n): s["results"][v][n]["tests"]["fusion linear (labelled)"]["beats_confidence"] for v in ("v1", "v1_2") for n in ("bolivia", "test")}
    assert fus == {("v1", "bolivia"): True, ("v1", "test"): True, ("v1_2", "bolivia"): True, ("v1_2", "test"): False}
    for v in ("v1", "v1_2"):
        tl = s["results"][v]["bolivia"]["tile_level"]
        assert abs(tl["failure_rate"] - 0.331) < 0.005                    # the paper's flood and burn-scar failure rates are 0.21 and 0.33
        assert tl["per_ranker"]["averaged confidence"]["aurc"] < tl["per_ranker"]["ensemble mutual information"]["aurc"]


def test_exp50_ndwi_carries_the_fusion_and_the_bag_does_not_replicate():
    """exp50 (job 736320): the label-fitted fusion's weight sits on NDWI level and dropping it costs the most on three
    of four arms; the exp49 bag result does not replicate with sixteen fresh members (P1 fails, sign flips across
    draws); a fusion fitted on tile failures beats confidence at the tile level on the multi-region split under both
    backbones but not on Bolivia (P2 fails); shift-label entropy loses to tile-phase on Bolivia (issue 5)."""
    s = json.load(open(os.path.join(OUT, "exp50_summary.json")))
    assert s["prereg"] == {"P1": False, "P2": False, "supported": False, "complete": True} and s["n_failures"] == 0
    for v in ("v1", "v1_2"):
        w = s["results"][v]["fusion_weights"]
        assert max(w, key=lambda k: abs(w[k])) == "control NDWI level" and w["control NDWI level"] > 0.8
        assert w["ensemble average entropy"] > 0.5 and w["ensemble predictive entropy"] > 0.5 and w["averaged confidence"] > 0.25
        for n in ("bolivia", "test"):
            r = s["results"][v][n]
            assert r["pooled_eaurc"]["fusion linear (labelled)"] < r["pooled_eaurc"]["averaged confidence"]
            assert r["pooled_eaurc"]["shift-label entropy"] >= r["pooled_eaurc"]["tile-phase"] - 0.0002
    for v, n in (("v1", "bolivia"), ("v1", "test"), ("v1_2", "bolivia")):
        dr = s["results"][v][n]["drop_one_pooled_eaurc"]
        assert max(dr, key=dr.get) == "control NDWI level"
    assert abs(s["results"]["v1_2"]["bolivia"]["drop_one_pooled_eaurc"]["control NDWI level"] - s["results"]["v1_2"]["bolivia"]["pooled_eaurc"]["averaged confidence"]) < 0.0005
    b1 = s["results"]["v1"]["bolivia"]["tests"]["bag16 predictive entropy"]
    b2 = s["results"]["v1_2"]["test"]["tests"]["bag16 predictive entropy"]
    assert b1["pooled_lead"] > 0 and b1["sign_p"] > 0.5                    # the exp49 bag gain is gone with a fresh draw
    assert b2["pooled_lead"] < 0 and b2["sign_p"] < 1e-4                   # and appears where exp49's bag had lost
    for v in ("v1", "v1_2"):
        assert s["results"][v]["test"]["tile_level"]["tile-fitted fusion (labelled)"]["diff_ci95"][1] < 0
    assert s["results"]["v1_2"]["bolivia"]["tile_level"]["tile-fitted fusion (labelled)"]["diff_ci95"][1] > 0


def test_exp51_their_probe_reproduces_and_the_exception_moves_with_the_sensor():
    """exp51 (job 761713): Ai2's own S1 probe recipe on their embeddings and on our S1 encodes. P1 holds (confidence beats
    the sensor-level control); P2 fails (v1.2 worse than v1 on Bolivia pooled, not per tile); the NDWI index beats the
    S1 probe's confidence on the multi-region split and loses on Bolivia; seven encoders share OlmoEarth's error
    windows at phi 0.78-0.82 (Satlas 0.62); their mIoU reproduces and our encode matches their embeddings."""
    s = json.load(open(os.path.join(OUT, "exp51_summary.json")))
    assert s["prereg"] == {"P1": True, "P2": False, "supported": False, "complete": True} and s["n_failures"] == 0
    p = s["P2_test"]
    assert p["lead_v1_2_minus_v1"] >= 0.001 and p["sign_p"] >= 0.05 and (p["w_v1_better"], p["l"]) == (173, 152)
    for a in ("B_v1", "C_v1_2"):
        t = s["results"][a]["test"]["tests"]["control NDWI level"]
        assert t["pooled_lead"] < -0.001 and t["l"] > t["w"]                      # the no-model index beats the S1 probe on the multi-region split
        assert s["results"][a]["bolivia"]["tests"]["control NDWI level"]["pooled_lead"] > 0
        assert s["results"][a]["test"]["tests"]["control S1 level"]["pooled_lead"] > 0.05
    a = s["results"]["A_olmoearth_base"]
    assert round(a["pixel_miou"], 3) == 0.789 and a["matched_tiles"] == 2419 and a["matched"]["phi_vs_arm_B_v1"] > 0.9
    ph = s["phi_vs_olmoearth_base"]
    assert all(0.77 <= ph[k]["phi"] <= 0.83 for k in ("galileo_base", "croma_base", "terramind_base", "clay_large", "anysat", "panopticon"))
    assert 0.55 <= ph["satlas_base"]["phi"] <= 0.70


def test_exp52_fine_tuning_corrects_the_frozen_errors_and_dissolves_the_bolivia_exception():
    """exp52 (job 762708): both preregistered tests pass; Sentinel-1 adds at most 0.1 points to the fine-tuned S2 model."""
    s = json.load(open(os.path.join(OUT, "exp52_summary.json")))
    assert s["prereg"] == {"P1": True, "P2": True, "supported": True, "complete": True} and s["n_failures"] == 0
    R = s["results"]
    for a in ("FT-S2 v1", "FT-S2 v1_2"):
        t = R[a]["bolivia"]["tests"]["control NDWI level"]
        assert t["pooled_lead"] >= 0.001 and t["sign_p"] < 0.05
    for n in ("bolivia", "test"):
        ct = R["FT-S2 v1"][n]["vs_frozen_head"]
        assert ct["share_corrected"] >= 0.40 and ct["corrected"] > ct["broken"]
        assert 0 <= R["FT-S1+S2 v1"][n]["window_accuracy"] - R["FT-S2 v1"][n]["window_accuracy"] <= 0.001
        assert R["FT-S1 v1"][n]["window_accuracy"] < R["FT-S1 v1"][n]["frozen_window_accuracy"]
    assert R["FT-S1 v1"]["test"]["pooled_eaurc"]["control NDWI level"] < R["FT-S1 v1"]["test"]["pooled_eaurc"]["averaged confidence"]


def test_exp56_audit_guided_label_selection_is_rejected():
    """exp56 (job 775302): the audit's tile selection is worse than random on the multi-region split at every budget."""
    s = json.load(open(os.path.join(OUT, "exp56_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is False and s["prereg"]["complete"] is True and s["n_failures"] == 0
    assert all(v < 0 for v in s["prereg"]["P1_detail"]["per_seed_lead"].values())
    A = s["aggregate"]
    for b in ("100", "300", "500", "1000"):
        assert A["audit"][b]["test"]["mean"] < A["random"][b]["test"]["mean"]
        assert A["audit"][b]["selected_frozen_error_rate_mean"] > 2.5 * A["random"][b]["selected_frozen_error_rate_mean"]
    assert s["labels_to_parity"]["audit"] is None


def test_exp54_multiclass_confidence_holds_and_shared_errors_are_task_dependent():
    """exp54 (job 775213): confidence beats the embedding-distance control on all five multi-class tasks; the cross-encoder
    phi is 0.26-0.44 on MADOS and 0.05-0.08 on PASTIS; boundary-first loses pooled and wins per tile at many classes."""
    s = json.load(open(os.path.join(OUT, "exp54_summary.json")))
    assert s["prereg"]["P1"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0 and len(s["prereg"]["P1_per_task"]) == 5
    r = s["results"]
    assert all(0.25 <= v["phi"] <= 0.45 for v in r["mados"]["phi_vs_olmoearth_base"].values())
    assert all(0.04 <= v["phi"] <= 0.09 for v in r["pastis_sentinel2"]["phi_vs_olmoearth_base"].values())
    bf = r["m_sa_crop_type"]["olmoearth_base"]["boundary_first_vs_confidence"]["0.05"]
    assert (bf["w"], bf["l"]) == (647, 196) and bf["pooled_boundary_first"] <= bf["pooled_confidence"]


def test_exp55_exceptions_are_rare_over_events_and_the_review_set_is_worth_ten_times_random():
    """exp55 (job 775649): over 45 GEOID-Flood events the no-model controls beat confidence on 2 (S2) and 1 (S1)
    events; the median event's 5% review set holds 88% / 64% of the errors; the sensor flip and the
    boundary-first majority do not generalise."""
    s = json.load(open(os.path.join(OUT, "exp55_summary.json")))
    assert s["prereg"]["P1"] is True and s["prereg"]["P2"] is True and s["prereg"]["P3"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0
    R = s["results"]
    assert R["A"]["across_events"]["n_events_scored"] == 45
    assert R["A"]["across_events"]["control NDWI level"]["control_wins"] == 2
    assert R["B"]["across_events"]["control S1 level"]["control_wins"] == 1
    assert all(R[t]["across_events"]["median_capture5_ratio_to_random"] >= 5 for t in ("A", "B"))
    d = s["prereg"]["descriptive"]
    assert d["B_ndwi_prior_beats_s1_confidence_on_majority"]["holds"] is False
    assert all(d[k]["holds"] is False for k in ("A_boundary_first_beats_confidence_at_5pct_on_majority", "B_boundary_first_beats_confidence_at_5pct_on_majority"))


def test_exp57_disagreement_is_boundary_located_and_the_differences_are_different():
    """exp57 (jobs 779972, 779973): on the disagreement windows of every pair of inferences the boundary cue is enriched
    more than twofold (P1, 16 of 16 pairs, at least 3.3x); the sensor and the backbone disagreement masks overlap at
    phi 0.27 / 0.18 pooled and below 0.5 on 257 of 327 and 386 of 489 tiles (P2); the module's cross-tabs reproduce
    exp52's corrected share and exp55's GEOID phi."""
    s = json.load(open(os.path.join(OUT, "exp57_summary.json")))
    assert s["prereg"]["P1"] is True and s["prereg"]["P2"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0
    assert s["prereg"]["P1_n_tested"] == 16 and min(s["prereg"]["P1_detail"].values()) > 3.3
    sb = s["results"]["cross_pairs"]["sensors_vs_backbones"]
    assert all(sb[t]["pooled_phi"] < 0.3 for t in ("bolivia", "test"))
    assert (sb["bolivia"]["tiles_below"], sb["test"]["tiles_below"]) == (257, 386)
    ft = s["results"]["finetune"]
    assert round(ft["bolivia"]["graded"]["crosstab"]["share_corrected"], 3) == 0.655
    assert round(ft["test"]["graded"]["crosstab"]["share_corrected"], 3) == 0.427
    g = s["results"]["geoid"]["graded"]
    assert round(g["crosstab_own_labels"]["phi"], 3) == 0.585 and round(g["disagreement_is"]["share_flooded_by_label"], 2) == 0.27


def test_exp58_tool_beats_the_raw_diff_only_modestly():
    """exp58 (job 787530; the GEOID half is job 789352): believing the larger margin beats the coin flip on the disagreement
    windows of all 15 pairs (per-tile p < 0.05) but misses the preregistered 55% on four of them (P1 fails); ordering the
    set by the first inference's confidence holds more than 60% of its errors in the least confident half only where the
    other side is much worse (P2 fails); the rule is less accurate on the persistent disagreement windows than on the
    transient ones on all six pair-testbeds. P3 (GEOID change) is added when that half lands."""
    s = json.load(open(_need("exp58_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is False and s["prereg"]["n_tested"] == 15
    d = s["prereg"]["detail"]
    assert len(d) == 15 and all(v["resolution_p"] < 0.05 for v in d.values())
    assert round(d["offsets/bolivia"]["share_right"], 2) == 0.51
    R = s["results"]
    assert round(R["sensors"]["test"]["resolution"]["share_right"], 2) == 0.70 and round(R["sensors"]["test"]["resolution"]["always_a"], 2) == 0.84
    assert round(R["sensors"]["bolivia"]["targeting"]["capture"]["0.5"], 2) == 0.74 and round(R["sensors"]["test"]["targeting"]["capture"]["0.5"], 2) == 0.92
    for p in ("offsets", "backbones", "sensors"):
        for t in ("bolivia", "test"):
            st = R[p][t]["stability"]
            assert st["resolution_persistent"] < st["resolution_transient"]
    assert s["prereg"]["P3"] is False and s["prereg"]["complete"] is True and s["n_failures"] == 0
    c = R["geoid"]["change"]
    assert round(c["ratio"], 2) == 1.92 and (c["over_events"]["w"], c["over_events"]["l"]) == (16, 9) and c["over_events"]["sign_p"] > 0.05
    assert R["geoid"]["resolution"]["share_right"] < 0.5 < R["geoid"]["resolution"]["always_b"]   # the S2 head predicts permanent water: wrong on every flooded window


def test_exp59_fitted_resolution_helps_except_on_the_fine_tuned_pair():
    """exp59 (jobs 791849, 791850): a label-fitted rule beats the raw margin rule on six of eight pair-testbeds and loses on the
    fine-tuned pair; on GEOID-Flood it is precise (80%) on 6% of the differing windows and not better event by event."""
    s = json.load(open(_need("exp59_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is False and s["prereg"]["complete"] is True and s["n_failures"] == 0
    D = s["prereg"]["P1_detail"]
    assert sum(v["gain"] > 0 for v in D.values()) == 6 and D["finetune/bolivia"]["gain"] < -0.2
    assert round(D["sensors/test"]["fitted"], 2) == 0.86 and round(D["sensors/test"]["margin"], 2) == 0.70
    c = s["results"]["geoid"]["change"]
    assert round(c["flooded_rule"], 2) == 0.80 and round(c["share_change"], 2) == 0.06 and (c["over_events_vs_raw_diff"]["w"], c["over_events_vs_raw_diff"]["l"]) == (11, 8)


def test_exp60_sensor_axis_isolates_and_time_axis_does_not_concentrate_the_flood():
    """exp60 (job 794821): the same-period cross-sensor difference carries none of the later flood (P2); the same-sensor
    difference across the event carries 26%, 1.33x the mixed pair, not the preregistered doubling (P1 fails)."""
    s = json.load(open(_need("exp60_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0
    R, T = s["results"]["pairs"], s["results"]["tests"]
    assert round(R["sensor_only"]["flooded_share"], 3) == 0.004 and (T["P2_sensor_vs_mixed"]["w"], T["P2_sensor_vs_mixed"]["l"]) == (23, 3)
    assert round(T["P1_time_vs_mixed"]["ratio_pooled"], 2) == 1.33 and (T["P1_time_vs_mixed"]["w"], T["P1_time_vs_mixed"]["l"]) == (13, 13)
    assert round(R["time_only"]["err_a_share"], 3) == 0.545
    z = np.load(_need("exp60_masks.npz"))
    d = (z["A_s2pre"] != z["A_s1pre"]) & z["ok"]
    flooded = z["y_after"] & ~z["y_permanent"]
    assert d.sum() == R["sensor_only"]["n_disagree"] and abs(flooded[d].mean() - R["sensor_only"]["flooded_share"]) < 1e-9


def test_exp62_fourth_cell_completed_and_the_pre_event_optical_head_fails_on_one_event():
    """exp62 (job 799540): the square is complete on 544 chips; both preregistered claims fail because the pre-event optical
    head misses the label's permanent water on the Ebro delta (393 of 544 chips), recomputed from the committed masks."""
    s = json.load(open(_need("exp62_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is False and s["prereg"]["complete"] is True and s["n_failures"] == 0
    m = np.load(_need("exp62_masks.npz"))
    z = np.load(_need("exp60_masks.npz"))
    idx, ok = m["chip_index"], m["ok"]
    ya, yb = z["y_permanent"][idx], z["y_after"][idx]
    assert len(idx) == 544 and abs(((m["B_s2post"] == yb)[ok]).mean() - s["results"]["fourth_cell"]["accuracy_water_after_label"]) < 1e-9
    a2 = z["A_s2pre"][idx]
    assert abs(((a2 == ya)[ok]).mean() - s["results"]["fourth_cell"]["accuracy_own_task_others"]["A_s2pre"]) < 1e-9
    tortosa = m["event"] == "EMSR279-11"
    assert tortosa.sum() == 393 and round(float(((a2 == ya)[tortosa[:, None, None] & ok]).mean()), 3) == 0.699
    assert round(float(a2[tortosa[:, None, None] & ok].mean()), 2) == 0.09 and round(float(ya[tortosa[:, None, None] & ok].mean()), 2) == 0.38


def test_exp63_boundary_cue_fails_on_parcels_and_the_draw_floor_is_tiny():
    """exp63 (job 800249): P1 fails on the seven PASTIS pairs and holds on MADOS, P2 holds; the head-draw floor recomputed from
    the committed decisions."""
    s = json.load(open(_need("exp63_summary.json")))
    assert s["prereg"]["P1"] is False and s["prereg"]["P2"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0
    d = s["prereg"]["P1_detail"]
    assert sum(v > 2 for v in d.values()) == 6 and all(v < 2 for k, v in d.items() if k.startswith("pastis"))
    z = np.load(_need("exp63_masks.npz"))
    for task in ("mados", "pastis_sentinel2"):
        ok = z[f"{task}/ok"]
        d01 = (z[f"{task}/olmoearth_base/0/dec"] != z[f"{task}/olmoearth_base/1/dec"]) & ok
        # the saved draw-1 decisions were aligned by label hash, which permutes tiles with identical labels: 21 windows on PASTIS
        assert abs(d01.sum() / ok.sum() - s["results"][task]["pairs"]["draw 0 vs draw 1"]["disagreement_rate"]) < 1e-4
        dg = (z[f"{task}/olmoearth_base/0/dec"] != z[f"{task}/galileo_base/0/dec"]) & ok
        assert dg.sum() > 20 * d01.sum()


def test_exp65_calibrate_refits_the_recorded_held_out_numbers_from_the_readings():
    """exp65 (job 801135): P1-P3 hold; fit_ranker and fit_side on the saved readings reproduce the recorded held-out numbers."""
    from oe_inferencex.calibrate import fit_ranker, fit_side
    s = json.load(open(_need("exp65_summary.json")))
    assert s["prereg"]["P1"] is True and s["prereg"]["P2"] is True and s["prereg"]["P3"] is True and s["prereg"]["complete"] is True and s["n_failures"] == 0
    z = np.load(_need("exp65_readings.npz"))
    for n in ("bolivia", "test"):
        sig = {k: z[f"ranker/{n}/{k}"] for k in ("confidence", "tile_phase", "boundary", "ndwi_level", "s2_variance")}
        err = z[f"ranker/{n}/err"]
        _, rep = fit_ranker(sig, err, np.ones_like(err, bool), groups=z[f"ranker/{n}/tile"], family="v1 S2 head")
        assert abs(rep["held_out"]["excess_aurc"] - s["results"]["ranker"][n]["held_out_excess_aurc"]) < 1e-9
    # every one of the eight pair-testbeds: the cross-fitted share right, the raw margin rule and their gain
    gain = {}
    for pair in ("offsets", "backbones", "sensors", "finetune"):
        for n in ("bolivia", "test"):
            pre = f"side/{pair}/{n}/"
            names = ("margin", "rank", "signed", "boundary", "tile_phase")
            fa = {k: z[pre + f"{k}:a"] for k in names}; fb = {k: z[pre + f"{k}:b"] for k in names}
            b_right, a_right = z[pre + "b_right"], z[pre + "a_right"]
            # a synthetic pair whose disagreement set is these windows: a = 0, b = 1, label = 1 where b is right and 0 where a is right
            a, b = np.zeros(len(b_right), int), np.ones(len(b_right), int)
            lab = np.where(b_right, 1, np.where(a_right, 0, 2))
            _, rep = fit_side(fa, fb, a, b, np.ones(len(b_right), bool), lab, groups=z[pre + "tile"], family="x", baseline="margin", shared={"ndwi": z[pre + "ndwi"]})
            rec = s["results"]["side"][pair][n]
            assert abs(rep["held_out"]["share_right"] - rec["cross_fit"]) < 1e-9, (pair, n)
            assert abs(rep["baseline"]["share_right"] - rec["margin_rule"]) < 1e-9, (pair, n)
            gain[pair, n] = rep["held_out"]["share_right"] - rep["baseline"]["share_right"]
            assert abs(gain[pair, n] - rec["gain"]) < 1e-9
    # calibrate-side-rule-held-out: the fitted rule beats the raw margin rule on all eight, by 8 to 24 points, and
    # refitting per family gains 14 and 17 on the fine-tuned pair where exp59's frozen-fitted rule had lost 26
    assert all(g >= 0.05 for g in gain.values()) and round(min(gain.values()), 2) == 0.08 and round(max(gain.values()), 2) == 0.24
    assert round(100 * gain["finetune", "bolivia"]) == 14 and round(100 * gain["finetune", "test"]) == 17


def test_exp66_dfc2020_sensor_axis_recomputes_from_the_masks():
    """exp66 (job 804312): P1 and P2 hold, P3 fails on all three arms; the sensor difference, the seed floor and the
    boundary enrichment recompute from the committed decisions of the first 200 test patches. The committed file holds
    no DFC2020 reference classes (removed 5 October 2026: the data are released to approved contest participants), so
    the which-side shares and the reference gap are checked against the summary only, not recomputed."""
    from oe_inferencex.explain import cue_enrichment
    s = json.load(open(_need("exp66_summary.json")))
    assert s["prereg"]["P1"] is True and s["prereg"]["P2"] is True and s["prereg"]["P3"] is False and s["prereg"]["complete"] is True and s["n_failures"] == 0
    assert s["prereg"]["P3_detail"]["arms_negative"] == 3 and s["prereg"]["P3_detail"]["arms_positive"] == 0
    z = np.load(_need("exp66_masks.npz"))
    assert "y_dfc" not in z.files and "y_lc" not in z.files                 # no DFC2020 labels in the repository
    ok = z["ok_dfc"]
    assert len(np.unique(z["patch"])) == 200 and ok.shape[0] == 3200        # 200 patches x 16 chips
    sens = (z["s2/dec"] != z["s1/dec"]) & ok
    floor = (z["s1/dec"] != z["s1/dec_seed1"]) & ok
    assert sens.sum() / ok.sum() > 0.35 and sens.sum() > 10 * floor.sum()   # the axis dominates the seed noise on this subsample too
    bnd = boundary_indicator(z["s2/dec"]) > 0        # the package's own cue; exp54.boundary_share is the same quantity
    enr = cue_enrichment(bnd[ok], sens[ok], n_boot=0)["enrichment"]
    assert 2.5 < enr < 4.5, enr                                            # the full run records 3.30x over all 1,200 patches
    g = s["results"]["reference_gap"]["boundary_first_lead_over_margin"]
    for arm in ("s2", "s1", "s1s2"):
        assert g[arm]["dfc"] < 0 and g[arm]["lc"] < g[arm]["dfc"], arm


def test_exp67_dynamic_world_recomputes_from_the_committed_windows():
    """exp67 (jobs 815613, 815860): all three preregistrations hold; the margin's advantage over the naive confidence, its
    tie counts and the under-confidence recompute from the committed windows of every eighth tile."""
    s = json.load(open(_need("exp67_summary.json")))
    assert all(s["prereg"][k] is True for k in ("P1", "P2", "P3")) and s["prereg"]["complete"] is True and s["n_failures"] == 0
    z = np.load(_need("exp67_windows.npz"))
    err = (z["dec"] != z["y"]).astype(np.float64)
    assert len(np.unique(z["tile"])) == 52 and len(err) == 548515
    margin, naive = -z["margin"].astype(np.float64), 1.0 - z["top1p"].astype(np.float64)
    assert excess_aurc(margin, err) < excess_aurc(naive, err)                      # the margin ranks better on the subsample too
    assert excess_aurc(margin, err) < excess_aurc(z["rarity"].astype(np.float64), err)
    ties = lambda v: len(v) - len(np.unique(v))
    assert ties(naive) > ties(margin)                                             # the saturation that exp13 chose a margin to avoid
    K = s["results"]["ranking"]["tied_values"]                                    # the ratio itself is a full-run quantity: tie counts are not scale-invariant
    assert K["1 - top-1 probability"] > 6 * K["margin"]
    assert float(z["top1p"].mean()) < float(1 - err.mean()) - 0.15                # under-confident by more than fifteen points
    from oe_inferencex.explain import cue_enrichment
    e = cue_enrichment(z["boundary"] > 0, err, n_boot=0)["enrichment"]
    assert 2.0 < e < 4.0, e


def test_exp68_lucas_recomputes_from_the_committed_polygons():
    """exp68 (jobs 817465, 820129): four of five preregistrations hold against a reference that never saw the imagery,
    and the load-bearing quantities recompute from the committed per-polygon arrays.

    The weighted estimators are exercised here rather than only in the experiment's own smoke, because the whole record
    rests on them: a design-weighted excess AURC must reduce to the package's unweighted one when every weight is one."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))
    import exp68_lucas as e68

    s = json.load(open(_need("exp68_summary.json")))
    v = s["verdicts"]
    assert all(v[k]["holds"] is True for k in ("P1", "P1b", "P4", "P3", "P5"))
    assert v["P2"]["holds"] is False                       # the published homogeneity filter flatters the tool
    z = np.load(_need("exp68_masks.npz"), allow_pickle=True)
    err = (z["dec"] != z["y"]).astype(np.float64)
    assert np.array_equal(err, z["err"])
    w, rep, obs = z["weight"], z["report"], z["obs"].astype(str)
    A = rep & (obs == "field")
    assert int(A.sum()) == s["results"]["part_a"]["n"] == 4778

    # the weighted estimator must agree with the package under unit weights, and with the record under the real ones
    margin = -z["margin"].astype(np.float64)
    assert abs(e68.w_excess_aurc(margin[A], err[A], np.ones(int(A.sum())))
               - excess_aurc(margin[A], err[A])) < 1e-9
    rec = s["results"]["part_a"]["signals"]
    assert abs(e68.w_excess_aurc(margin[A], err[A], w[A]) - rec["margin"]["excess_aurc_weighted"]) < 1e-9
    for name, arr in (("ctl_pixel_variance", z["variance"]), ("ctl_polygon_area", -z["area"]),
                      ("ctl_window_impurity", 1.0 - z["purity"]), ("entropy", z["entropy"])):
        got = e68.w_excess_aurc(np.asarray(arr, dtype=np.float64)[A], err[A], w[A])
        assert abs(got - rec[name]["excess_aurc_weighted"]) < 1e-9, (name, got, rec[name]["excess_aurc_weighted"])

    # P1: the margin beats both deployable controls, and the design weighting cuts the lead rather than creating it
    assert rec["margin"]["excess_aurc_weighted"] < rec["ctl_pixel_variance"]["excess_aurc_weighted"]
    assert rec["margin"]["excess_aurc_weighted"] < rec["ctl_class_rarity"]["excess_aurc_weighted"]
    assert s["results"]["part_a"]["margin_lead_naive"] > s["results"]["part_a"]["margin_lead_weighted"] > 0.01

    # the graded window must be the one holding the survey point, not the grid centre (the bug that moved P5)
    assert s["results"]["window_holds_the_survey_point"] == e68.POINT_WIN == 8
    assert e68.choose_window(np.zeros((e68.G, e68.G))) == (8, 8, 0.0)

    # the tuned probe: the chosen decay is interior to the swept grid and the default memorised the fit set
    T = s["results"]["probe_tuning"]
    assert T["chosen"]["weight_decay"] < max(g["weight_decay"] for g in T["grid"])
    default = [g for g in T["grid"] if g["weight_decay"] == 1e-4 and g["epochs"] == 80][0]
    assert default["inner_fit_accuracy"] > 0.99 > default["inner_val_accuracy"] + 0.4
    assert T["chosen"]["inner_val_accuracy"] > default["inner_val_accuracy"]

    # P3's gap is over half reproduced by a flag carrying no information, so it is not boundary-specific
    P = s["results"]["part_c_provenance"]
    assert v["P3"]["gap_from_a_rate_matched_null"] > 0.5 * v["P3"]["gap"]
    assert (P["arms"]["photo"]["rate_matched"]["variance_matched"]
            - P["arms"]["field"]["rate_matched"]["variance_matched"]) > v["P3"]["gap"]

    # part B: the two-date difference dwarfs the head-reseed floor and the near date is the better side
    B = s["results"]["part_b"]
    assert B["disagreement_rate"] > 50 * B["head_draw_floor"]
    assert B["which_side"]["a_right"] > B["which_side"]["b_right"] and B["near_beats_far_sign_test"]["p_greater"] < 1e-12
    assert B["per_class"]["cropland"]["change_rate"] >= 2 * B["per_class"]["woodland"]["change_rate"]

    # part C, the published homogeneity filter (P2, lucas-the-homogeneity-filter-flatters-the-tool): the units it
    # keeps (homogeneous plot fills the window, area >= 5,000 sqm) against the units it deletes, recomputed from the
    # committed polygons: the counts, the design-weighted AUROC of the margin, its weighted excess AURC and the
    # ceiling-free odds ratio of the boundary cue on each side
    from oe_inferencex.metrics import weighted_auroc
    filt = A & z["homog"] & (z["area"] >= 5000)
    deleted = A & ~(z["homog"] & (z["area"] >= 5000))
    F, P2 = s["results"]["part_c_filters"]["filtered_vs_deleted"], v["P2"]
    assert (int(filt.sum()), int(deleted.sum())) == (1145, 3633) == (F["n_filtered"], F["n_deleted"])
    auroc = {k: weighted_auroc(margin[m], err[m], w[m]) for k, m in (("filtered", filt), ("deleted", deleted))}
    assert abs(auroc["filtered"] - P2["auroc_filtered"]) < 1e-9 and abs(auroc["deleted"] - P2["auroc_deleted"]) < 1e-9
    assert (round(auroc["filtered"], 3), round(auroc["deleted"], 3)) == (0.774, 0.677)
    eaurc = {k: e68.w_excess_aurc(margin[m], err[m], w[m]) for k, m in (("filtered", filt), ("deleted", deleted))}
    assert abs(eaurc["filtered"] - F["margin_excess_aurc_filtered"]) < 1e-9 and abs(eaurc["deleted"] - F["margin_excess_aurc_deleted"]) < 1e-9
    assert (round(eaurc["filtered"], 4), round(eaurc["deleted"], 4)) == (0.0785, 0.1925)
    odds = {k: e68.cue_stats(z["boundary"][m] > 0, err[m], w[m])["odds_ratio_weighted"] for k, m in (("filtered", filt), ("deleted", deleted))}
    assert abs(odds["filtered"] - P2["odds_ratio_filtered"]) < 1e-9 and abs(odds["deleted"] - P2["odds_ratio_deleted"]) < 1e-9
    assert (round(odds["filtered"], 2), round(odds["deleted"], 2)) == (3.17, 3.04)


def test_exp69_eurocrops_recomputes_from_the_committed_windows():
    """exp69 (jobs 821061, 822320): all three preregistrations hold on farmers' declared crops in three regions, and
    part A and part B recompute from the committed per-window arrays.

    The load-bearing quantity is the one no other testbed here can produce: the rate at which the model changes its
    decision between two summers on ground whose DECLARED CROP DID NOT CHANGE, which is the floor a two-date difference
    must be read against."""
    s = json.load(open(_need("exp69_summary.json")))
    V, R = s["verdicts"], s["results"]["regions"]
    assert all(V[k]["holds"] is True for k in ("P1", "P2", "P3"))
    assert set(R) == {"at", "dk", "si"}
    assert sum(R[r]["n_graded_windows_year0"] for r in R) == 106274

    # the floor is far above the head-reseed rate exp68 had to use, and it tracks the model rather than the place
    floors = {r: R[r]["part_b"]["model_change_rate_declared_same"] for r in R}
    assert all(0.0 < f < 0.5 for f in floors.values())
    assert min(floors, key=floors.get) == max(R, key=lambda r: R[r]["window_accuracy_report"])
    assert all(R[r]["part_b"]["ratio"] >= 2.0 for r in R)
    # and none of it depends on how long the probe trained
    for r in R:
        assert abs(R[r]["reference_probe"]["margin_lead"] - R[r]["part_a"]["margin_lead"]) <= 0.003
        assert R[r]["reference_probe"]["ratio"] >= 2.0

    z = np.load(_need("exp69_windows.npz"), allow_pickle=True)
    region = z["region"].astype(str)
    assert set(np.unique(region)) == {"at", "dk", "si"}
    for r in np.unique(region):
        m = region == r
        # part A, on the graded windows of the report cells, recomputed from the arrays
        A = m & z["graded_y0"] & z["report"]
        assert A.sum() > 500, (r, int(A.sum()))
        err = (z["dec_y0"][A] != z["y_y0"][A]).astype(np.float64)
        margin = -z["margin_y0"][A].astype(np.float64)
        rec = R[r]["part_a"]["signals"]
        assert abs(excess_aurc(margin, err) - rec["margin"]["excess_aurc"]) < 0.02, r
        # the margin beats both no-model controls on the subsample too
        assert excess_aurc(margin, err) < excess_aurc(z["variance_y0"][A].astype(np.float64), err)
        # part B's two rates, and the sign of their difference, recomputed
        B = m & z["graded_y0"] & z["graded_y1"] & z["report"]
        moved = z["dec_y0"][B] != z["dec_y1"][B]
        declared_changed = z["y_y0"][B] != z["y_y1"][B]
        assert declared_changed.any() and (~declared_changed).any(), r
        got_changed, got_same = moved[declared_changed].mean(), moved[~declared_changed].mean()
        assert got_changed > 2 * got_same, (r, got_changed, got_same)
        assert abs(got_same - R[r]["part_b"]["model_change_rate_declared_same"]) < 0.06, r
        # a difference can be the model right about BOTH years, the case one labelled date cannot represent
        both = moved & (z["dec_y0"][B] == z["y_y0"][B]) & (z["dec_y1"][B] == z["y_y1"][B])
        assert both.sum() > 0, r


def test_exp70_every_task_of_a_suite_we_did_not_choose():
    """exp70 (job 837407): the margin beats the best no-model control on all 24 tasks of Ai2's published suite.

    This is the answer to the objection that the testbeds in this repository were chosen with knowledge of the answer,
    so the test checks the things that make it an answer: that every task won, that the win survives collapsing tasks
    that share a source, that both families and both ends of the accuracy range are covered, and that nothing was
    silently dropped."""
    s = json.load(open(_need("exp70_summary.json")))
    V, R = s["verdicts"], s["results"]["tasks"]
    assert not s["results"]["failures"], f"tasks failed to score: {s['results']['failures']}"
    assert len(R) == 24 and all(V[k]["holds"] is True for k in ("P1", "P2", "P3"))

    # every task, and every distinct source, since three AWF sensors are not three datasets
    assert all(R[t]["margin_lead"] > 0 for t in R)
    assert V["P1"]["distinct_sources"] == V["P1"]["source_wins"] == 14
    assert V["P1"]["p"] < 1e-6

    # both families, and the excluded task named rather than dropped
    fams = collections.Counter(R[t]["family"] for t in R)
    assert fams["classification"] == 17 and fams["segmentation"] == 7
    assert "m_bigearthnet" in s["config"]["excluded"] and "multi-label" in s["config"]["excluded"]["m_bigearthnet"]

    # the margin's lead is not an artifact of easy tasks: it wins at both ends of a wide accuracy range
    accs = {t: R[t]["test_accuracy"] for t in R}
    assert min(accs.values()) < 0.40 and max(accs.values()) > 0.97
    worst, best = min(accs, key=accs.get), max(accs, key=accs.get)
    assert R[worst]["margin_lead"] > 0 and R[best]["margin_lead"] > 0

    # the control it beats must be a control, not another confidence reading
    assert all(R[t]["best_control"].startswith("ctl_") for t in R)

    # P3 carries its own scope: exp68's direction, not exp68's magnitude
    assert V["P3"]["margin_over_entropy_wide_gap"] < V["P3"]["margin_over_entropy_tight_gap"]
    assert abs(V["P3"]["margin_over_entropy_tight_gap"] - V["P3"]["margin_over_entropy_wide_gap"]) < 0.001, \
        "the suite must not be quoted as reproducing exp68's inversion; it reproduces only the direction"


def test_exp70_csv_agrees_with_its_summary():
    """One row per task, and the row's numbers are the summary's."""
    s = json.load(open(_need("exp70_summary.json")))["results"]["tasks"]
    rows = list(csv.DictReader(open(_need("exp70_tasks.csv"))))
    assert len(rows) == len(s) == 24
    for r in rows:
        t = r["task"]
        assert t in s
        assert abs(float(r["margin_lead"]) - s[t]["margin_lead"]) < 1e-9
        assert abs(float(r["test_accuracy"]) - s[t]["test_accuracy"]) < 1e-9
        assert int(r["n_units"]) == s[t]["n_units"]


def test_exp78_coverages_reproduce_in_a_small_seeded_monte_carlo_from_the_per_unit_files():
    """exp78's recorded coverages re-drawn from the per-unit export (exp78_units/<task>.npz, read with the
    experiment's own load_units) with 300 fresh draws at seed 1, where the run used 2,000 at seed 0, through the
    package estimators exactly as run_designs builds each arm: a simple random sample of 300 under Wilson with the
    finite-population correction (D1/E1), 18 tiles of 16 windows under the ordinary formula (D4/E2_naive) and under
    the ratio estimator with the normal quantile (D4/E1_cluster), and the stratified draw allocated by Neyman from
    the model's own confidence (D2c/E1). On MADOS and m-cashew-plant every coverage must sit within three Monte
    Carlo standard errors of the record (the error of both runs combined), the design effect and the error rate must
    reproduce exactly, MADOS's naive interval must fall far below the 0.85 bar and its cluster interval below it,
    and MADOS's label saving at equal half-width must be the record's 2.51 within its own Monte Carlo error."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))
    import exp78_error_rate_estimation as e78
    from oe_inferencex import estimate as est
    S = json.load(open(_need("exp78_summary.json")))
    assert S["config"] == {"R": 2000, "headline_budget": 300, "strata": 5, "m_per_tile": 16, "piloted": S["config"]["piloted"]}
    R, B, M, K = 300, 300, 16, 5
    want = {"mados": {"D1/E1": 0.933, "D4/E2_naive": 0.506, "D4/E1_cluster": 0.598, "D2c/E1": 0.941},
            "m_cashew_plant": {"D1/E1": 0.948, "D4/E2_naive": 0.939, "D4/E1_cluster": 0.936, "D2c/E1": 0.947}}
    for task, rec in want.items():
        _need(os.path.join("exp78_units", f"{task}.npz"))
        u = e78.load_units(task)
        err, margin, p1, tile = u["err"], u["margin"], u["p1"], u["tile"]
        N, theta = err.size, float(err.mean())
        T = S["tasks"][task]
        assert theta == T["error_rate"] and N == T["n_units"]
        assert est.design_effect(err, tile) == pytest.approx(T["design_effect"], abs=1e-9)
        rng = np.random.default_rng(1)
        tiles = np.unique(tile)
        by = {t: np.flatnonzero(tile == t) for t in tiles}
        sizes = np.bincount(tile)
        strata = est.confidence_strata(margin)
        ssz = np.bincount(strata, minlength=K)
        q = np.array([(1 - p1[strata == h]).mean() if (strata == h).any() else 0.0 for h in range(K)])
        cov = {k: 0 for k in rec}
        hw = {"D1/E1": [], "D2c/E1": []}
        for _ in range(R):
            i = rng.choice(N, B, replace=False)
            lo, hi = est.wilson_interval(err[i].sum(), B, N)
            cov["D1/E1"] += lo <= theta <= hi
            hw["D1/E1"].append((hi - lo) / 2)
            pick = rng.choice(tiles, min(B // M, tiles.size), replace=False)
            i = np.concatenate([rng.choice(by[t], min(M, by[t].size), replace=False) for t in pick])
            lo, hi = est.wilson_interval(err[i].sum(), i.size, N)
            cov["D4/E2_naive"] += lo <= theta <= hi
            _, lo, hi, _, _ = est.cluster_interval(err, tile, i, tile_sizes=sizes, quantile="normal")
            cov["D4/E1_cluster"] += lo <= theta <= hi
            alloc = est.neyman_allocation(ssz, np.sqrt(np.clip(q * (1 - q), 1e-9, None)), B)
            _, lo, hi, _ = est.stratified_interval(err, strata, est.draw_stratified(rng, strata, ssz, alloc), ssz, N)
            cov["D2c/E1"] += lo <= theta <= hi
            hw["D2c/E1"].append((hi - lo) / 2)
        for arm, p in rec.items():
            got = cov[arm] / R
            assert round(T["budgets"]["300"][arm]["coverage"], 3) == p, (task, arm)
            se = np.sqrt(p * (1 - p) / R + p * (1 - p) / S["config"]["R"])
            assert abs(got - p) <= 3 * se, (task, arm, got, p, se)
        if task == "mados":
            se_naive = np.sqrt(0.85 * 0.15 / R)
            assert cov["D4/E2_naive"] / R < 0.85 - 3 * se_naive and cov["D4/E1_cluster"] / R < 0.85 - 3 * se_naive
            # confidence-saves-a-quarter-to-a-half-of-the-labels: the saving is the inverse square of the width ratio
            hs, hc = np.array(hw["D1/E1"]), np.array(hw["D2c/E1"])
            ratio = hc.mean() / hs.mean()
            saving = 1 / ratio ** 2
            var_ratio = ratio ** 2 * (hc.var() / hc.mean() ** 2 + hs.var() / hs.mean() ** 2) * (1 / R + 1 / S["config"]["R"])
            se_saving = 2 * saving / ratio * np.sqrt(var_ratio)
            assert abs(saving - 2.51) <= 3 * se_saving, (saving, se_saving)
            assert saving > 1.8
        else:
            assert theta > 0.20 and np.mean(hw["D1/E1"]) > 0.04          # no arm reaches +/-4 points where the map is a third wrong


def _enumerated_random_excess(n, k):
    """A random order's excess AURC written out: its expected AURC is the error rate k / n, and the perfect order's
    selective risk at coverage i / n is max(0, i - (n - k)) / i, averaged over i; no package function is used."""
    i = np.arange(1, n + 1, dtype=np.float64)
    return k / n - float(np.mean(np.maximum(0.0, i - (n - k)) / i))


def test_suite_controls_margin_and_ten_percent_review_by_a_second_route():
    """suite-controls-are-near-chance and suite-review-at-ten-percent by two other routes. exp79's OlmoEarth Base
    seed-0 record, a second run of exp70's probes, gives the control counts and the 10% capture; exp78's per-unit
    export gives the margin's share of the gap (its AURC with ties broken at random, seeded, against an enumerated
    perfect order) and a plain stable-sort 10% review set. The export stores margins in float32, so a few tied
    windows move the per-unit numbers slightly; the tolerances say how much."""
    rec = json.load(open(_need("exp70_summary.json")))["results"]["tasks"]
    b = json.load(open(_need(os.path.join("exp79_seeds", "olmoearth_base.json"))))["tasks"]
    assert set(b) == set(rec) and len(rec) == 24
    ctl = ("ctl_embedding_distance", "ctl_class_rarity")
    share, at_or_below, below_half, cap79 = [], 0, 0, []
    for t, v in b.items():
        s = v["seeds"][0]
        rnd = _enumerated_random_excess(s["n_units"], int(round(s["error_rate"] * s["n_units"])))
        best = min(s["signals"][c]["excess_aurc"] for c in ctl)
        share.append(1 - best / rnd)
        at_or_below += best >= rnd
        below_half += s["signals"]["ctl_embedding_distance"]["auroc"] < 0.5
        cap79.append(s["signals"]["margin"]["capture"]["0.1"])
    assert at_or_below == 9 and below_half == 21
    assert abs(np.median(share) - 0.094) < 1e-3 and abs(np.median(cap79) - 0.214) < 1e-3
    rng = np.random.default_rng(0)
    margin_share, cap, ceil = {}, {}, {}
    for t in rec:
        z = np.load(_need(os.path.join("exp78_units", f"{t}.npz")))
        m, e = z["margin"].astype(np.float64), z["err"].astype(np.float64)
        n, k = e.size, int(e.sum())
        order = np.lexsort((rng.random(n), -m))                  # most confident first, ties in a random order
        aurc = float(np.mean(np.cumsum(e[order]) / np.arange(1, n + 1)))
        rnd = _enumerated_random_excess(n, k)
        margin_share[t] = 1 - (aurc - (k / n - rnd)) / rnd
        kk = max(1, round(0.1 * n))
        cap[t] = e[np.argsort(m, kind="stable")[:kk]].sum() / k
        ceil[t] = min(kk, k) / k
    assert abs(np.median(list(margin_share.values())) - 0.680) < 5e-3
    assert all(v > 0.1 for v in cap.values()) and min(cap, key=cap.get) == "nandi_sentinel1"
    assert abs(np.median(list(cap.values())) - 0.214) < 3e-3 and abs(np.median(list(ceil.values())) - 0.410) < 3e-3
    assert abs(np.median([(cap[t] - 0.1) / (ceil[t] - 0.1) for t in rec]) - 0.40) < 5e-3


def test_exp79_margin_beats_random_in_every_cell_by_a_second_route():
    """exp79-margin-beats-random-everywhere by another statistic and another file. In all 3,560 cells the margin's
    recorded excess AURC is below a random order's, written out by enumeration here; and on OlmoEarth Base at seed 0,
    whose per-unit export is committed (exp78_units), a mid-rank AUROC computed from the units is above 0.5 on every
    task. The export's float32 margins tie on a few classification tasks (377 of m-eurosat's 1,000 windows at the
    top), which moves that AUROC by up to 0.023 from exp79's float64 reading."""
    import glob
    files = sorted(glob.glob(os.path.join(OUT, "exp79_seeds", "*.json")))
    assert len(files) == 16
    cells, cache = 0, {}
    for f in files:
        for t, r in json.load(open(f))["tasks"].items():
            for s in r["seeds"]:
                n, k = s["n_units"], int(round(s["error_rate"] * s["n_units"]))
                if (n, k) not in cache:
                    cache[(n, k)] = _enumerated_random_excess(n, k)
                assert s["signals"]["margin"]["excess_aurc"] < cache[(n, k)], (f, t, s["seed"])
                cells += 1
    assert cells == 3560
    b = json.load(open(os.path.join(OUT, "exp79_seeds", "olmoearth_base.json")))["tasks"]
    for t, v in b.items():
        z = np.load(_need(os.path.join("exp78_units", f"{t}.npz")))
        score, pos = -z["margin"].astype(np.float64), z["err"].astype(bool)
        _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
        ranks = (np.concatenate([[0], np.cumsum(counts)[:-1]]) + (counts + 1) / 2.0)[inv]
        n1, n0 = int(pos.sum()), int((~pos).sum())
        auroc = (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        assert auroc > 0.5, t
        assert abs(auroc - v["seeds"][0]["signals"]["margin"]["auroc"]) < 0.03, t


def test_exp64_declines_recount_from_the_recorded_answers():
    """agent-benchmark-decline-holds by a second route: the 27B answers themselves (exp64_answers.jsonl), read on
    the ten cards with a second inference (exp64_cards.csv), with the grader's own rule for what a pick is and what
    a decline is. Arm D, which has no rasters and no package, declines on every run, as arm A does."""
    cards = {r["card"] for r in csv.DictReader(open(_need("exp64_cards.csv"))) if r["has_second"] == "True"}
    assert len(cards) == 10
    picks = {"first", "a", "second", "b"}
    declines = ("decline", "neither", "none", "cannot", "can't", "unresolv", "not resolv", "without label",
                "no way to", "insufficient", "unknown", "not possible")           # exp64_arms.grade_comparison
    tally = collections.defaultdict(collections.Counter)
    for line in open(_need("exp64_answers.jsonl")):
        run = json.loads(line)
        if run["card"] not in cards:
            continue
        comp = (run.get("answer") or {}).get("comparison")
        if not isinstance(comp, dict):
            tally[run["arm"]]["unanswered"] += 1
            continue
        said = str(comp.get("believe", "")).strip().lower()
        kind = "pick" if said in picks else "decline" if said and any(w in said for w in declines) else "other"
        tally[run["arm"]][kind] += 1
    assert tally["A"] == {"decline": 30} and tally["D"] == {"decline": 30}
    assert tally["B"] == {"pick": 18, "decline": 3, "unanswered": 9}


def test_exp78_shipped_tile_design_coverage_by_an_independent_implementation():
    """exp78_shipped_tiles.json by a second route: the shipped design written out here from its description (tiles
    in a random order, up to 16 windows from each, until 300 labels; the ratio estimator weighted by each tile's valid
    windows, its with-replacement between-tile variance and scipy's t quantile on tiles - 1 df), none of the package's
    code, 800 fresh draws per task. Each coverage must sit within four Monte Carlo standard errors of the record, on
    the two tasks where the record says the design falls short and on one where it holds."""
    student_t = pytest.importorskip("scipy.stats").t         # scipy comes with the geo extra
    d = json.load(open(_need("exp78_shipped_tiles.json")))
    assert d["config"]["budget"] == 300 and d["config"]["per_tile"] == 16
    R, rng = 800, np.random.default_rng(20260930)
    for task in ("mados", "sen1floods11", "pastis_sentinel2"):
        z = np.load(_need(os.path.join("exp78_units", f"{task}.npz")))
        n, h, w = (int(v) for v in z["grid"])
        ok = np.unpackbits(z["ok_packed"])[:int(z["ok_len"][0])].astype(bool)
        tile = np.repeat(np.arange(n), h * w)[ok]
        err = z["err"].astype(np.float64)
        theta = err.mean()
        order = np.argsort(tile, kind="stable")
        ids, starts, counts = np.unique(tile[order], return_index=True, return_counts=True)
        members = [order[a:a + c] for a, c in zip(starts, counts)]
        cover = 0
        for _ in range(R):
            sizes, means, total = [], [], 0
            for j in rng.permutation(ids.size):
                if total >= 300:
                    break
                take = min(16, counts[j], 300 - total)
                pick = rng.choice(members[j], take, replace=False)
                sizes.append(counts[j])
                means.append(err[pick].mean())
                total += take
            sizes, means = np.array(sizes, float), np.array(means)
            k = sizes.size
            est = (sizes * means).sum() / sizes.sum()
            var = ((sizes * (means - est)) ** 2).sum() / (k * (k - 1)) / sizes.mean() ** 2
            half = student_t.ppf(0.975, k - 1) * np.sqrt(var)
            cover += max(0.0, est - half) <= theta <= min(1.0, est + half)
        rec = d["tasks"][task]["shipped"]
        assert abs(d["tasks"][task]["error_rate"] - theta) < 1e-12
        c = cover / R
        se = np.sqrt(rec["coverage"] * (1 - rec["coverage"]) * (1 / R + 1 / rec["draws"]))
        assert abs(c - rec["coverage"]) <= 4 * se, (task, c, rec["coverage"])
