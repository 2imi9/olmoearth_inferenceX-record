"""exp112 (docs/plan/v13_embeddings.md): a map from OlmoEarth v1.3 embeddings at LCMAP's probability sample. The run's
outputs against the independent second route, the outputs' privacy, and the level-cut limit that decides P3(b)."""
import ast
import csv
import hashlib
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")
SUMMARY = os.path.join(OUT, "exp112_summary.json")
CROSSCHECK = os.path.join(OUT, "exp112_crosscheck.json")
PLOTS = os.path.join(OUT, "exp112_plots.csv")
CROSSCHECK_SCRIPT = os.path.join(ROOT, "exp", "exp112_crosscheck.py")


def _load():
    with open(SUMMARY, encoding="utf-8") as f:
        S = json.load(f)
    with open(CROSSCHECK, encoding="utf-8") as f:
        C = json.load(f)
    return S, C


def test_exp112_recorded_by_a_second_route():
    """Every deterministic graded number of the run equals the second route's (written by an agent that saw neither the
    run's script nor its outputs, importing neither), its Monte Carlo cells agree within their errors, and P1 to P4
    come out the same."""
    src = open(CROSSCHECK_SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp112_v13_lcmap"}, "the second route imports neither"
    S, C = _load()
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest(), "the committed crosscheck ran"
    assert C["script_uncommitted"] is False
    assert S["reference_csv_sha256"] == C["inputs"]["reference"]["sha256"]
    assert C["inputs"]["reference"]["zip"]["sha256"].startswith("eb935d96")
    assert C["S"]["vectors_file_equals_S"] is True and C["S"]["without_embedding"] == S["no_embedding"] == 0

    c = S["by_fit"]["1000"]["128"]
    a1, x1 = c["arm1"], C["arm1"]
    assert (a1["n"], a1["errors"]) == (x1["N"], x1["errors"]) == (11000, 1290)
    for run, cross in ((a1["own"]["auroc"], x1["auroc"]), (a1["own"]["capture_10"], x1["capture_10"]),
                       (a1["own"]["excess_aurc"], x1["excess_aurc"]),
                       (a1["controls"]["rarity"]["auroc"], x1["controls"]["ctl_class_rarity"]["auroc"]),
                       (a1["controls"]["distance"]["auroc"], x1["controls"]["ctl_distance_from_F_mean"]["auroc"]),
                       (a1["lead"], x1["lead_over_best_control"]),
                       (a1["confident_error_share"], x1["confident_error_share"])):
        assert abs(run - cross) < 1e-9
    for B in ("100", "300"):
        assert abs(c["arm2"]["cells"][B]["exact_coverage"] - C["arm2"][B]["exact_coverage"]) < 1e-8
        se = c["arm2"]["cells"][B]["mc_se"]
        assert abs(c["arm2"]["cells"][B]["mc_coverage"] - C["arm2"][B]["mc_coverage"]) < 3 * se * 2 ** 0.5
    v_run = c["arm3"]["by_alpha"]["0.05"]["violation_share"]
    v_cross = C["arm3"]["0.05"]["large_mc"]["violation_share"]
    assert abs(v_run - v_cross) < 3 * (v_cross * (1 - v_cross) / 2000) ** 0.5
    for m, key in zip(c["arm4"], ("lcmap_collection_1_3", "esri")):
        x = C["arm4"][key]
        assert (m["N"], m["D"], m["K_A"], m["K_B"]) == (x["N"], x["D"], x["K_A"], x["K_B"])
        for n in ("100", "300"):
            assert abs(m["n"][n]["p_names_A"] - x["budgets"][n]["p_names_A"]) < 1e-9
    for nfit in ("300", "3000"):
        run = S["by_fit"][nfit]["128"]
        cross = C["reported_fit_sizes_and_dims"][f"dims128_fit{nfit}"]
        assert abs(run["theta"] - cross["theta"]) < 1e-12 and abs(run["ranking"]["auroc"] - cross["auroc"]) < 1e-9
    V, W = S["verdicts"], C["verdicts"]
    assert V["P1"]["holds"] is W["P1"]["holds"] is True
    assert V["P2"] is W["P2"]["holds"] is True
    assert V["P3"]["a"] is W["P3a"]["holds"] is True
    assert V["P3"]["b"] is W["P3b"]["holds"] is False
    assert V["P4"] is W["P4"]["holds"] is True


def test_exp112_outputs_hold_no_position_or_embedding():
    """The per-plot table has the six named columns and one row per plot of S; the license notice sits beside it."""
    with open(PLOTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["plotid", "in_fit_1000", "ref", "has_embedding", "pred", "top_probability"]
    assert len(rows) == 12000 and len({r["plotid"] for r in rows}) == 12000
    notice = open(os.path.join(OUT, "exp112_NOTICE.md"), encoding="utf-8").read()
    assert "Ai2" in notice and "military" in notice and "extractive" in notice
    S, C = _load()

    def longest(o):
        if isinstance(o, dict):
            return max([longest(v) for v in o.values()] + [0])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    assert longest(S) <= 64 and longest(C) <= 64, "aggregates only"


def test_exp112_p3b_limit_is_the_level_cut():
    """At 300 labels and alpha 0.05 the standard cut first tests the 15% zone, which expects exactly b_min = 45 labels,
    so even a perfect map certifies a zone with probability at most 0.5255 on exp112's 11,000 plots; the second route's
    exact value for this map is below it."""
    stats = pytest.importorskip("scipy.stats")
    from oe_inferencex import estimate as est, plan
    S, C = _load()
    z = C["arm3"]["0.05"]
    N, n = S["by_fit"]["1000"]["128"]["n_T"], 300
    assert est.min_labels_to_certify(0.05, 0.10) == z["levels"]["b_min"] == 45
    first = z["levels"]["sizes"][0]
    assert (z["levels"]["coverage"][0], first) == (0.15, 1650) and n * first / N == 45
    limit = float(stats.hypergeom.sf(44, N, first, n))
    assert abs(limit - 0.5255) < 5e-5
    assert abs(plan.zone_level_probability(N, n, 0.15, 0.05, 0.0) - limit) < 1e-4
    assert z["exact_p_certify_any_guard_aside"] < limit
    assert abs(plan.zone_level_probability(N, n, 0.15, 0.05, z["first_level"]["zone_true_error"])
               - z["exact_p_certify_any_guard_aside"]) < 1e-3


def test_exp112_p3b_knife_edge_and_the_400_label_budget():
    """The bar passes only a map whose most confident 15% holds at most one error: a 2,000-draw median reaches the
    first level when one draw certifies with probability above one half, which 0 and 1 errors give and 2 do not; a
    perfect map then passes with probability about 0.99. At 400 labels the same cut certifies this map's 15% zone with
    probability 0.920 and a perfect map's with 0.989."""
    stats = pytest.importorskip("scipy.stats")
    from oe_inferencex import plan
    N, nz = 11000, 1650
    p = [plan.zone_level_probability(N, 300, 0.15, 0.05, k / nz) for k in (0, 1, 2)]
    assert [round(x, 3) for x in p] == [0.525, 0.51, 0.494] and p[1] > 0.5 > p[2]
    assert round(float(stats.binom.sf(1000, 2000, p[0])), 2) == 0.99
    assert round(plan.zone_level_probability(N, 400, 0.15, 0.05, 2 / nz), 3) == 0.92
    assert round(plan.zone_level_probability(N, 400, 0.15, 0.05, 0.0), 3) == 0.989

