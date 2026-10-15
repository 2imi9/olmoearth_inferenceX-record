"""exp110 (docs/plan/ftw_fields.md): the constants are the page's; the window reading, truth, controls and cues against
code written here on constructed chips; the chip bootstrap's pair matrix and the grouped statistics against the
package's metrics; certify's steps against certify_zone; the B3 stop rules and the verdicts on constructed rows; the
smoke run and the refusal while a draft; the summary, once written, holds no chip or tile identifier."""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))

import exp110_ftw_field_map as e110  # noqa: E402
from oe_inferencex import estimate as est, metrics  # noqa: E402

SUMMARY = os.path.join(ROOT, "exp", "out", e110.SUMMARY)


def _page():
    return " ".join(open(e110.PLAN, encoding="utf-8").read().split())


def test_the_constants_are_the_pages():
    t = _page()
    for phrase in (
            "the map's confidence ranks its window errors at AUROC ≥ 0.70, with capture at 10% ≥ 0.25",
            "its AUROC is at least 0.05 above the best no-model control's",
            "a country with fewer than 200 window errors is not determined for P1. If two or more countries fall below "
            "the floor, P1 cannot hold.",
            "In at least 3 of the 4 countries",
            "The exact coverage is at least 0.95 at B = 100 and 300 in each of the four countries and pooled (10 cells), "
            "and the Monte Carlo route agrees with it within 3 Monte Carlo standard errors.",
            "the share of draws that certify a zone wrong more often than α is at most 0.115 in each of the 5 populations",
            "at α = 0.05 and 300 labels, the median certified share over the draws is at least 0.25 of the windows in at "
            "least 2 of the 4 countries",
            "at n = 100 and 300, the tool names the worse-agreeing map with probability at most 0.025 in every country "
            "and pooled",
            "at n = 100, it names the better-agreeing map with probability at least 0.8 in at least 3 of the 4 countries",
            "2,000 draws per cell (seed 110)", "2,000 draws per population (seed 110)",
            "α = 0.05 (graded) and 0.10 (reported), δ = 0.10", "the prefix rule and the standard level cut",
            "The B = 300 random samples of arm 2", "8 x 8 mask pixels on each chip's own mask grid",
            "16 x 16 windows (331,520 in all) are reported, not graded", "(2,000 resamples by chip, seed 110)",
            "agrees with the 2e map on fewer than 70% of a country's windows",
            "B3 with its input bands reversed (B02, B03, B04, B08)", "(`--tile 32`) and 16 windows per chip",
            "dilated by 7 m, about 1 mask pixel N-S and 2 E-W", "Croatia 4, Sweden 9", "side by side in sorted aoi_id order",
            "Windows holding any pixel that is zero in every band of either S2 window are left out",
            "scaled by its standard deviation over the country's chips, and pixels that are zero in every band are left out",
            "the share of the country's windows the map puts in the window's class",
            "`exp/exp110_ftw_field_map.py`", "`exp/out/exp110_summary.json`"):
        assert phrase in t, phrase
    for c, (n, w) in {"belgium": (198, "202,752"), "croatia": (353, "361,472"), "slovenia": (228, "233,472"),
                      "sweden": (516, "528,384")}.items():
        assert f"| {n} | {w} |" in t and e110.CHIPS[c] == n and e110.WINDOWS[c] == int(w.replace(",", ""))
    assert "**1,295** | **1,326,080** |" in t and sum(e110.WINDOWS.values()) == 1_326_080
    assert sum(e110.CHIPS[c] * (e110.CHIP // e110.PATCH_REPORTED) ** 2 for c in e110.COUNTRIES) == 331_520
    assert (e110.P1_AUROC, e110.P1_CAPTURE, e110.P1_LEAD, e110.P1_COUNTRIES, e110.P1_ERROR_FLOOR) == (0.70, 0.25, 0.05, 3, 200)
    assert (e110.P2_COVERAGE, e110.P2_MC_SE, e110.P3_VIOLATION, e110.P3_SHARE, e110.P3_COUNTRIES) == (0.95, 3, 0.115, 0.25, 2)
    assert (e110.P4_WRONG, e110.P4_RIGHT, e110.P4_COUNTRIES, e110.P4_HEADLINE, e110.B3_AGREE_STOP) == (0.025, 0.8, 3, 100, 0.70)
    assert (e110.ALPHA, e110.ALPHA_REPORTED, e110.DELTA, e110.BUDGETS, e110.CERTIFY_BUDGET) == (0.05, 0.10, 0.10, (100, 300), 300)
    assert (e110.DRAWS, e110.SEED, e110.BOOT, e110.BOOT_LEVEL, e110.PATCH, e110.PATCH_REPORTED) == (2000, 110, 2000, 0.90, 8, 16)
    assert (e110.TILE_WINDOWS, e110.PER_TILE, e110.DILATE_NS, e110.DILATE_EW) == (32, 16, 1, 2)
    assert e110.SEAM_CHIPS == {"belgium": 0, "croatia": 4, "slovenia": 0, "sweden": 9}
    assert est.min_labels_to_certify(0.05, 0.10) == 45 and "(45 at α = 0.05)" in t
    assert e110.BANDS == ("B04", "B03", "B02", "B08") and [e110.BANDS[i] for i in e110.B3_SWAP] == ["B02", "B03", "B04", "B08"]


def _brute_window(p, patch):
    """Window confidence (mean of 2|p - 0.5|) and class (majority of p > 0.5, a tie to the larger margin sum)."""
    n, H, W = p.shape
    g, h = H // patch, W // patch
    conf, cls = np.zeros((n, g, h)), np.zeros((n, g, h), int)
    for i in range(n):
        for r in range(g):
            for c in range(h):
                q = p[i, r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
                m = 2 * np.abs(q - 0.5)
                conf[i, r, c] = m.mean()
                ones = int((q > 0.5).sum())
                if 2 * ones != q.size:
                    cls[i, r, c] = int(2 * ones > q.size)
                else:
                    cls[i, r, c] = int(m[q > 0.5].sum() > m[q <= 0.5].sum())
    return conf, cls


def test_the_reading_and_the_truth_on_constructed_chips():
    rng = np.random.default_rng(3)
    p = rng.choice([0.0, 0.1, 0.45, 0.55, 0.9, 1.0], size=(3, 32, 32))
    p[1, :8, :8] = 1.0                                       # a window wholly at 1: confidence 1
    p[2, :8, :4], p[2, :8, 4:8] = 0.95, 0.4                  # a 32 / 32 window: the tie goes to the larger margin
    r = e110.read_extent(p, 8)
    conf, cls = _brute_window(p, 8)
    assert np.allclose(r["conf"], conf, atol=1e-12) and np.array_equal(r["cls"], cls)
    assert r["conf"][1, 0, 0] == 1.0 and r["cls"][2, 0, 0] == 1
    rc = e110.read_extent(p, 8, chunk=2)
    assert np.array_equal(rc["conf"], r["conf"]) and np.array_equal(rc["cls"], r["cls"]), "chunks read the same windows"
    m = (rng.random((3, 32, 32)) < 0.5).astype(np.uint8)
    m[0, :8, :8] = 0
    m[0, :4, :8] = 1                                         # 32 against 32: no majority
    t = e110.truth_windows(m, 8, 2)
    assert t[0, 0, 0] == -1
    for i in range(3):
        for a in range(4):
            for b in range(4):
                s = int(m[i, a * 8:(a + 1) * 8, b * 8:(b + 1) * 8].sum())
                assert t[i, a, b] == (-1 if s == 32 else int(s > 32))
    # the mosaic's flat order is the row mosaic's raster order
    x = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    assert e110.flat(x).tolist() == [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11]
    assert np.array_equal(e110.unmosaic(e110.row_mosaic(x), 3), x)


def test_the_controls_and_cues_on_constructed_chips():
    rng = np.random.default_rng(5)
    s2 = rng.integers(1, 500, size=(2, 8, 16, 16)).astype(np.uint16)
    s2[0, :, :8, :8] = 0                                     # a window with no pixel left
    s2[1, :, 0, :3] = 0                                      # three pixels zero in every band
    s2[1, :4, 5, 5] = 0                                      # zero in one S2 window's bands only: kept
    var, empty, sds = e110.pixel_variance(s2, 8)
    keep = (s2 != 0).any(axis=1)
    want_sd = [s2[:, j][keep].astype(float).std() for j in range(8)]
    assert np.allclose(sds, want_sd) and empty == 1
    for i in range(2):
        for r in range(2):
            for c in range(2):
                k = keep[i, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8]
                v = [np.var(s2[i, j, r * 8:(r + 1) * 8, c * 8:(c + 1) * 8][k] / want_sd[j]) if k.any() else 0.0
                     for j in range(8)]
                assert var[i, r, c] == pytest.approx(np.mean(v), abs=1e-12)
    cls = rng.integers(0, 2, size=(2, 5, 5))
    rare = e110.rarity(cls)
    assert np.allclose(rare, -np.where(cls == 1, (cls == 1).mean(), (cls == 0).mean()))
    nb = e110.neighbour_disagreement(cls)
    for i in range(2):
        for r in range(5):
            for c in range(5):
                ns = [cls[i, r + dy, c + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if (dy or dx) and 0 <= r + dy < 5 and 0 <= c + dx < 5]
                assert nb[i, r, c] == pytest.approx(np.mean([v != cls[i, r, c] for v in ns]))
    m = np.zeros((2, 9, 9), np.uint8)
    m[0, 4, 4] = 1
    d = e110.dilate_field(m)
    assert d[1].sum() == 0 and d[0].sum() == 3 * 5 and d[0, 3:6, 2:7].all(), "a box of 1 row N-S and 2 columns E-W"


def test_the_chip_bootstrap_matrix_is_the_weighted_auroc():
    rng = np.random.default_rng(7)
    n, C = 3000, 23
    u = np.round(rng.normal(size=n), 1)                      # heavy ties
    chip = rng.integers(0, C, n)
    e = rng.random(n) < 0.2 + 0.1 * (u > 0)
    M, E, R = e110.pair_matrix(u, e, chip, C)
    assert e110.matrix_auroc(M, E, R, np.ones(C)) == pytest.approx(metrics.weighted_auroc(u, e, np.ones(n)), abs=1e-12)
    for _ in range(5):
        w = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        assert e110.matrix_auroc(M, E, R, w) == pytest.approx(metrics.weighted_auroc(u, e, w[chip]), abs=1e-12)


def test_the_grouped_statistics_are_the_packages():
    rng = np.random.default_rng(11)
    u = np.round(rng.random(5000), 2)
    e = rng.random(5000) < 0.1 + 0.3 * u
    auc, cap, th = e110.grouped_auroc_capture(e110.grouped_table(u, e))
    assert auc == pytest.approx(metrics.weighted_auroc(u, e, np.ones(u.size)), abs=1e-12)
    assert cap == pytest.approx(metrics.capture_at_budget_expected(u, e, (0.10,))[0.10], abs=1e-12)
    assert th == pytest.approx(e.mean())
    merged = e110.merge_tables([e110.grouped_table(u[:1700], e[:1700]), e110.grouped_table(u[1700:], e[1700:])])
    assert e110.grouped_auroc_capture(merged) == pytest.approx((auc, cap, th), abs=1e-12)


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_certify_steps_are_certify_zone(alpha):
    rng = np.random.default_rng(13)
    N = 4000
    margin = np.round(rng.random(N), 2)                      # ties at the zone edges
    wrong = (rng.random(N) < np.where(margin > 0.6, 0.01, 0.25)).astype(float)
    Z = e110.zone_setup(margin, wrong)
    certified = 0
    for r in range(40):
        s = est.sample_for_estimation(margin, 300, design="random", seed=110 + r)
        i = s["indices"]
        got = e110.certify_steps(Z, i, wrong[i], alpha)
        ref = est.certify_zone(margin, i, wrong[i], alpha, 0.10)
        assert (got["coverage"], got.get("n_zone")) == (ref["coverage"], ref["n_zone"])
        if ref["coverage"] is not None:
            certified += 1
            order, _ = est.zone_order(margin)
            assert got["zone_error"] == pytest.approx(wrong[order[:ref["n_zone"]]].mean())
    assert certified >= 10
    review = np.argsort(margin, kind="stable")[:300]          # the least confident windows: refused by both
    with pytest.raises(ValueError):
        est.certify_zone(margin, review, wrong[review], alpha, 0.10)
    assert e110.certify_steps(Z, review, wrong[review], alpha).get("refused")


def test_the_b3_stop_rules():
    inv = {c: {"b3_agreement_stated": 0.85, "b3_agreement_reversed": 0.6} for c in e110.COUNTRIES}
    assert e110.b3_stop_rules(inv)["arm4_runs"]
    inv["croatia"]["b3_agreement_stated"] = 0.69
    r = e110.b3_stop_rules(inv)
    assert not r["arm4_runs"] and "croatia" in r["reason"]
    inv["croatia"].update(b3_agreement_stated=0.9, b3_agreement_reversed=0.9)
    r = e110.b3_stop_rules(inv)
    assert not r["arm4_runs"] and "reversed" in r["reason"], "a tie is not fewer windows"


def _rows(auroc=(0.8,) * 4, capture=(0.3,) * 4, lead=(0.1,) * 4, errors=(500,) * 4, exact=0.96, mc=0.96, viol=0.05,
          share=(0.3,) * 4, wrong=0.0, right=(0.9,) * 4):
    rows = {}
    for j, pop in enumerate(e110.COUNTRIES + ("pooled",)):
        k = min(j, 3)
        se = np.sqrt(exact * (1 - exact) / 2000)
        rows[pop] = {"arm1": {"auroc": auroc[k], "capture_10": capture[k], "lead_over_best_control": lead[k],
                              "errors": errors[k]},
                     "arm2": {"cells": {str(B): {"exact_coverage": exact, "mc_coverage": mc, "mc_se": se}
                                        for B in e110.BUDGETS}},
                     "arm3": {"by_alpha": {"0.05": {"violation_share": viol, "median_certified_share": share[k]}}},
                     "arm4": {"budgets": {str(n): {"p_wrong": wrong, "p_right": right[k]} for n in e110.BUDGETS}}}
    return rows


def test_the_verdicts_on_constructed_rows():
    v = e110.verdicts(_rows())
    assert all(v[k]["holds"] for k in ("P1", "P2", "P3a", "P3b", "P4a", "P4b"))
    v = e110.verdicts(_rows(auroc=(0.8, 0.8, 0.69, 0.8)))
    assert v["P1"]["holds"] and v["P1"]["by_country"]["slovenia"] == "does not meet"
    v = e110.verdicts(_rows(auroc=(0.8, 0.8, 0.69, 0.8), capture=(0.24, 0.3, 0.3, 0.3)))
    assert not v["P1"]["holds"] and v["P1"]["verdict"] == "fails"
    v = e110.verdicts(_rows(lead=(0.049, 0.1, 0.1, 0.1), errors=(500, 500, 199, 500)))
    assert not v["P1"]["holds"] and v["P1"]["below_floor"] == ["slovenia"], "an undetermined country does not count"
    v = e110.verdicts(_rows(errors=(500, 199, 199, 500)))
    assert v["P1"]["verdict"].startswith("cannot hold")
    v = e110.verdicts(_rows(errors=(500, 500, 199, 500)))
    assert v["P1"]["holds"], "three of the four meet; the fourth is not determined"
    assert not e110.verdicts(_rows(exact=0.949, mc=0.949))["P2"]["holds"]
    assert not e110.verdicts(_rows(exact=0.96, mc=0.96 + 3.1 * np.sqrt(0.96 * 0.04 / 2000)))["P2"]["holds"]
    assert e110.verdicts(_rows(exact=0.96, mc=0.96 + 2.9 * np.sqrt(0.96 * 0.04 / 2000)))["P2"]["holds"]
    assert e110.verdicts(_rows(viol=0.115))["P3a"]["holds"] and not e110.verdicts(_rows(viol=0.1151))["P3a"]["holds"]
    v = e110.verdicts(_rows(share=(0.25, 0.2, 0.25, 0.0)))
    assert v["P3b"]["holds"] and v["P3b"]["countries_meeting"] == ["belgium", "slovenia"]
    assert not e110.verdicts(_rows(share=(0.25, 0.2, 0.2, 0.0)))["P3b"]["holds"]
    assert e110.verdicts(_rows(wrong=0.025))["P4a"]["holds"] and not e110.verdicts(_rows(wrong=0.026))["P4a"]["holds"]
    assert e110.verdicts(_rows(right=(0.8, 0.8, 0.8, 0.1)))["P4b"]["holds"]
    assert not e110.verdicts(_rows(right=(0.8, 0.79, 0.8, None)))["P4b"]["holds"]
    v = e110.verdicts(_rows(), arm4_runs=False)
    assert v["P4a"]["holds"] is None and v["P4b"]["verdict"].startswith("not run")


def test_the_smoke_and_the_refusal_while_a_draft(tmp_path, monkeypatch):
    rec = e110.main(["--smoke", "--out-dir", str(tmp_path), "--work", str(tmp_path), "--draws", "8", "--boot", "8"])
    assert set(rec["verdicts"]) == {"P1", "P2", "P3a", "P3b", "P4a", "P4b"}
    assert rec["label_free"]["b3_stop_rules"]["arm4_runs"]
    for pop in e110.COUNTRIES + ("pooled",):
        row = rec["populations"][pop]
        assert row["arm1"]["auroc"] > 0.6 and row["arm1"]["N"] == row["arm2"]["N"]
        assert all(c["exact_coverage"] >= 0.95 for c in row["arm2"]["cells"].values())
        assert row["arm4"]["K_A"] + row["arm4"]["K_B"] == row["arm4"]["D"], "two classes: one map is right"
    assert all(c["agrees_with_python_arrays"] for c in rec["checks"]["command_line_arm1"].values())
    assert os.path.exists(tmp_path / "exp110_summary_smoke.json")
    monkeypatch.setattr(e110, "prereg_status", lambda path=e110.PLAN: "draft")
    with pytest.raises(SystemExit):
        e110.main(["--data", str(tmp_path)])
    monkeypatch.setattr(e110, "prereg_status", lambda path=e110.PLAN: "frozen")
    with pytest.raises(SystemExit):
        e110.main(["--data", str(tmp_path), "--draws", "10"])


@pytest.mark.skipif(not os.path.exists(SUMMARY), reason="exp110 has not been run")
def test_the_summary_holds_no_identifier():
    text = open(SUMMARY, encoding="utf-8").read()
    assert not re.search(r"g\d+(-\d+)?_\d{5}_\d+", text), "an FTW aoi_id"
    assert not re.search(r"\b\d{2}[A-Z]{3}(_\d_\d)?\b", text), "a Sentinel-2 or 2e tile name"
    assert "/Users/" not in text and "s2_images/" not in text

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    assert longest(json.loads(text)) <= 64, "no per-window or per-chip list"


CROSSCHECK = os.path.join(ROOT, "exp", "out", "exp110_crosscheck.json")
CROSSCHECK_SCRIPT = os.path.join(ROOT, "exp", "exp110_crosscheck.py")


def _no_identifier(text):
    assert not re.search(r"g\d+(-\d+)?_\d{5}_\d+", text), "an FTW aoi_id"
    assert not re.search(r"\b\d{2}[A-Z]{3}(_\d_\d)?\b", text), "a Sentinel-2 or 2e tile name"
    assert "/Users/" not in text and "s2_images/" not in text


def test_exp110_recorded_by_a_second_route():
    """The run's graded numbers against exp/exp110_crosscheck.py, written from the independent audit with numpy, scipy
    and rasterio (its own windowing, ranks, hypergeometric tails, certify steps and sign test), importing neither the
    run's script nor the package: the inputs' sha256, the label-free and label checks, and per country and pooled every
    count exactly and every rate to 1e-9 or tighter (the exact coverage to 1e-8: scipy's tails against the package's
    lgamma), the Monte Carlo cells draw for draw from their seeds, the chip bootstrap's ends to its Monte Carlo error,
    and the six verdicts. The field + boundary reading, which only the crosscheck computes, is checked against the
    package's own exact functions from its counts."""
    import ast
    import hashlib

    from oe_inferencex import plan

    src = open(CROSSCHECK_SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp110_ftw_field_map"}, "the second route imports neither"
    S = json.load(open(SUMMARY, encoding="utf-8"))
    text = open(CROSSCHECK, encoding="utf-8").read()
    _no_identifier(text)
    C = json.loads(text)

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    assert longest(C) <= 64, "aggregates only: no per-window or per-chip list"
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest(), "the committed crosscheck ran"
    with open(os.path.join(ROOT, "exp", "exp110_ftw_field_map.py"), "rb") as f:
        assert S["script_sha256"] == hashlib.sha256(f.read()).hexdigest(), "the committed run script is the one that ran"
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("cf59be4")

    # inputs: the same files, and they match the acquisition manifest's list
    SI, CI = S["inputs"], C["inputs"]
    assert CI["npz_and_index_sha256"] == SI["npz_and_index_sha256"] and CI["npz_and_index_match_manifest"]
    assert CI["s2_files"] == {k: SI["s2_files"][k] for k in CI["s2_files"]} and CI["s2_files"]["all_match_manifest"]
    assert CI["manifest_files_csv_sha256"] == SI["manifest_files_csv_sha256"]

    # label-free and label checks, per country
    for c in e110.COUNTRIES:
        r, lf, ck = C["countries"][c], S["label_free"]["countries"][c], S["checks"]["labels"][c]
        assert (r["chips"], r["windows"], r["seam_chips"]) == (lf["chips"], lf["windows"], lf["seam_chips"])
        assert r["sorted_ids"] and r["mask2_eq_mask3_field"] and not r["mask2_eq_mask3_field_or_boundary"]
        assert r["boundary_px_coded_background_in_mask2"] == ck["mask2_differs_from_mask3_field_or_boundary_pixels"]
        assert (r["truth_tied"], r["truth_tied_16"], r["truth3_tied"]) == (
            ck["windows_truth_tied"], ck["windows_truth_tied_16px"], ck["windows_truth3_tied"])
        assert (r["mask2_values"], r["mask3_values"]) == (ck["mask2_values"], ck["mask3_values"])
        assert r["truth_field_share"] == pytest.approx(ck["truth_class_share"]["1"], abs=1e-12)
        assert (r["zero_imagery_windows"], r["map_windows_conf_1"], r["D_all_windows"]) == (
            lf["zero_imagery_windows"], lf["map_windows_at_confidence_1"], lf["D_all_windows"])
        ctl = lf["controls"]
        assert (r["variance_windows_no_pixel"], r["px_zero_one_window_only"], r["px_zero_all8"]) == (
            ctl["pixel_variance_windows_with_no_pixel"], ctl["pixels_zero_in_one_window_only"],
            ctl["pixels_zero_in_every_band"])
        assert r["band_sd"] == pytest.approx(ctl["band_sd"], rel=1e-9)
        assert r["map_class_share_field"] == pytest.approx(lf["map_class_share"]["1"], abs=1e-12)
        assert r["map_conf_median"] == pytest.approx(lf["map_confidence_quantiles"]["0.5"], abs=1e-12)
        assert r["b3_agree_stated"] == pytest.approx(lf["b3_agreement_stated"], abs=1e-12)
        assert r["b3_agree_reversed"] == pytest.approx(lf["b3_agreement_reversed"], abs=1e-12)
        assert r["b3_zero_flag_matches_s2"] and r["swap_band_order"] == ["B02", "B03", "B04", "B08"]

    def close(a, b, tol, what):
        assert a == pytest.approx(b, abs=tol), what

    for pop in e110.COUNTRIES + ("pooled",):
        R, Q = S["populations"][pop], C["populations"][pop]
        # arm 1
        r1, q1 = R["arm1"], Q["arm1"]
        assert (q1["N"], q1["errors"], q1["best_control"]) == (r1["N"], r1["errors"], r1["best_control"]), pop
        close(q1["theta"], r1["theta"], 1e-12, (pop, "theta"))
        for k in ("auroc", "capture_10", "excess_aurc", "oracle_capture_10", "lead_over_best_control"):
            close(q1[k], r1[k], 1e-9, (pop, k))
        close(q1["confident_error_share"], r1["confident_errors"]["share_of_errors"], 1e-12, (pop, "confident"))
        for k in ("ctl_pixel_variance", "ctl_class_rarity"):
            for m in ("auroc", "capture_10"):
                close(q1["controls"][k][m], r1["controls"][k][m], 1e-9, (pop, k, m))
        # the chip bootstrap: 400 resamples (150 pooled) against the run's 2,000, so its ends agree to Monte Carlo error
        for k in ("auroc", "lead_over_best_control"):
            assert np.allclose(q1["bootstrap"][k], r1["bootstrap"][k], atol=0.006), (pop, k)
        # arm 2: the exact coverage, and the run's Monte Carlo draw for draw
        for B in ("100", "300"):
            cell = R["arm2"]["cells"][B]
            close(Q["arm2"][B]["exact_coverage"], cell["exact_coverage"], 1e-8, (pop, B, "exact"))
            assert Q["arm2"][B]["mc_coverage_same_seeds"] == cell["mc_coverage"], (pop, B)
        # arm 3: certify on the run's draws, both alphas, count for count
        for a in ("0.05", "0.1"):
            q3, r3 = Q["arm3"][a]["same_seeds"], R["arm3"]["by_alpha"][a]
            assert (q3["violations"], q3["refused_as_review_set"], q3["certified_share_counts"]) == (
                r3["violations"], r3["refused_as_review_set"], r3["certified_share_counts"]), (pop, a)
            for k in ("p_certify_any", "violation_share", "median_certified_share"):
                close(q3[k], r3[k], 1e-12, (pop, a, k))
            close(q3["zone_true_error_median"], r3["zone_true_error"]["median"], 1e-12, (pop, a, "zone error"))
        # arm 4: the counts, the exact probabilities and the Monte Carlo draw for draw
        q4, r4 = Q["arm4"], R["arm4"]
        assert (q4["N"], q4["D"], q4["K_A"], q4["K_B"], q4["better_agreeing"]) == (
            r4["N"], r4["D"], r4["K_A"], r4["K_B"], r4["better_agreeing"]), pop
        for n in ("100", "300"):
            for k in ("p_right", "p_wrong"):
                close(q4["budgets"][n][k], r4["budgets"][n][k], 1e-9, (pop, n, k))
                assert q4["budgets"][n]["mc_same_seeds"][k] == r4["budgets"][n]["mc"][k], (pop, n, k)
        # reported readings
        for k in ("windows_16px", "three_class_8px", "lenient_extent_8px", "without_seam_chips"):
            if k not in Q["reported"]:
                assert pop == "pooled" and k != "without_seam_chips"
                continue
            q, r = Q["reported"][k], R["reported"][k]
            assert (q["N"], q["errors"]) == (r["N"], r["errors"]), (pop, k)
            close(q["theta"], r["theta"], 1e-12, (pop, k))
            close(q["auroc"], r["auroc"], 1e-9, (pop, k))
    for k in ("P1", "P2", "P3a", "P3b", "P4a", "P4b"):
        assert C["verdicts"][k]["holds"] is S["verdicts"][k]["holds"] is True, k
    assert C["verdicts"]["P1"]["countries_meeting"] == S["verdicts"]["P1"]["countries_meeting"]
    assert C["verdicts"]["P3b"]["countries_meeting"] == S["verdicts"]["P3b"]["countries_meeting"]
    assert C["verdicts"]["P4b"]["countries_meeting"] == S["verdicts"]["P4b"]["countries_meeting"]

    # the field + boundary reading (beside the graded result): its arithmetic against the package's exact functions
    F = C["field_plus_boundary"]
    for c in e110.COUNTRIES:
        assert F[c]["arm1"]["N"] == C["countries"][c]["windows"] - C["countries"][c]["field_plus_boundary_truth_tied"]
    assert F["pooled"]["arm1"]["N"] == sum(F[c]["arm1"]["N"] for c in e110.COUNTRIES)
    assert F["pooled"]["arm1"]["errors"] == sum(F[c]["arm1"]["errors"] for c in e110.COUNTRIES)
    for pop in e110.COUNTRIES + ("pooled",):
        f1, f4 = F[pop]["arm1"], F[pop]["arm4"]
        close(f1["theta"], f1["errors"] / f1["N"], 1e-15, (pop, "fb theta"))
        close(f1["lead_over_best_control"], f1["auroc"] - f1["controls"][f1["best_control"]]["auroc"], 1e-15, pop)
        for B in ("100", "300"):
            pkg = est.exact_coverage_srs(f1["N"], f1["errors"], int(B), interval=est.hypergeom_interval)
            close(F[pop]["arm2"][B]["exact_coverage"], pkg, 1e-8, (pop, B, "fb exact coverage"))
        assert f4["K_A"] + f4["K_B"] == f4["D"], "two classes: one map is right"
        for n in ("100", "300"):
            pa, pb = plan.which_map_probability(f4["N"], f4["D"], int(n), f4["K_A"], f4["K_B"])
            close(f4["budgets"][n]["p_names_2e"], pa, 1e-9, (pop, n, "fb 2e"))
            close(f4["budgets"][n]["p_names_b3"], pb, 1e-9, (pop, n, "fb b3"))
