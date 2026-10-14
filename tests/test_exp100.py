"""exp100 (docs/plan/alignment.md): the shift check. Written before the page was frozen and before any shifted reading
of exp98's or exp99's maps: the peak and its refinement against surfaces whose maximum is known exactly, the rule's
three calls, the resampler, the reading of accepted classes, the refusal of a reading that does not give the recorded
errors, and the smoke (a synthetic map with a planted offset, and exp98's and exp99's own smoke inputs). No network.
Needs rasterio, shapely and pyproj (the geo extra); skipped without them."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp100_alignment as e100   # noqa: E402
import exp98_awf_deployment as e98   # noqa: E402

S = e100.SHIFT


def quadratic(south, east, s=S):
    """A surface that is exactly a paraboloid with its vertex at (south, east) px from the labelled pixel."""
    i, j = np.mgrid[0:2 * s + 1, 0:2 * s + 1]
    return 1.0 - (i - s - south) ** 2 - 0.5 * (j - s - east) ** 2


@pytest.mark.parametrize("south,east", [(0.0, 0.0), (0.3, -0.2), (-0.45, 0.4), (1.25, -2.4), (2.0, 1.0)])
def test_peak_recovers_a_paraboloids_vertex(south, east):
    p = e100.peak(quadratic(south, east))
    assert p["south_px"] == pytest.approx(south, abs=1e-12)
    assert p["east_px"] == pytest.approx(east, abs=1e-12)
    assert (p["best_south_px"], p["best_east_px"]) == (round(south), round(east))
    assert not p["at_edge"]


def test_peak_on_the_edge_is_unrefined_and_called_an_offset():
    p = e100.peak(quadratic(3.4, 0.0))
    assert p["at_edge"] and p["best_south_px"] == 3 and p["south_px"] == 3.0
    assert e100.call(p, {"south_px": [3.0, 3.0], "east_px": [0.0, 0.0]}) == "offset"


def test_ties_take_the_position_nearest_the_labelled_pixel():
    M = np.zeros((2 * S + 1, 2 * S + 1))
    M[S - 2, S] = M[S, S + 1] = M[S + 3, S + 3] = 1.0
    p = e100.peak(M)
    assert (p["best_south_px"], p["best_east_px"]) == (0, 1)
    flat = e100.peak(np.ones((2 * S + 1, 2 * S + 1)))
    assert (flat["south_px"], flat["east_px"]) == (0.0, 0.0)          # no peak: no refinement


def test_refine_is_zero_without_a_peak_and_half_at_a_tied_neighbour():
    assert e100._refine(1.0, 1.0, 1.0) == 0.0
    assert e100._refine(0.0, 1.0, 1.0) == pytest.approx(0.5)
    assert e100._refine(1.0, 1.0, 0.0) == pytest.approx(-0.5)


@pytest.mark.parametrize("iv,want", [
    ({"south_px": [-0.5, 0.5], "east_px": [-0.1, 0.2]}, "aligned"),
    ({"south_px": [0.51, 1.2], "east_px": [-0.1, 0.2]}, "offset"),
    ({"south_px": [-0.1, 0.1], "east_px": [-1.4, -0.6]}, "offset"),
    ({"south_px": [0.2, 0.7], "east_px": [-0.1, 0.1]}, "not determined"),
    ({"south_px": [-0.6, 0.6], "east_px": [0.0, 0.0]}, "not determined"),
])
def test_the_rule(iv, want):
    assert e100.call({"at_edge": False}, iv) == want


def test_the_resampler_keeps_each_strata_count_of_clusters():
    clusters = np.array([0, 0, 1, 2, 2, 2, 3, 4])
    strata = np.array(["a", "a", "a", "a", "a", "a", "b", "b"])
    draw = e100._resampler(clusters, strata)
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = draw(rng)
        picked = [(strata[k], clusters[k]) for k in idx]
        assert all(s == "b" for s, c in picked if c in (3, 4)) and all(s == "a" for s, c in picked if c < 3)
        sizes = {0: 2, 1: 1, 2: 3, 3: 1, 4: 1}
        n_a = sum(1 for k in idx if strata[k] == "a")
        n_b = sum(1 for k in idx if strata[k] == "b")
        # three clusters drawn in a, two in b: the counts are sums of three and two cluster sizes
        assert n_a in {sizes[x] + sizes[y] + sizes[z] for x in range(3) for y in range(3) for z in range(3)}
        assert n_b == 2


def test_surfaces_read_the_accepted_classes_and_never_channel_nine():
    B = np.zeros((2, e98.N_OUT, 2 * S + 1, 2 * S + 1))
    B[:, 2] = 0.5
    B[:, 4] = 0.3
    B[:, 9] = 0.2
    B[1, :, 0, 0] = np.nan
    score, agree, full, _ = e100.surfaces(B, [(2,), (4, 9)])
    assert np.allclose(score[0], 0.5) and np.allclose(score[1][1:, 1:], 0.5)
    assert agree[0].all() and not agree[1][1:, 1:].any()
    assert list(full) == [True, False]
    B[0, 9] = 0.6
    _, agree, _, _ = e100.surfaces(B, [(9,), (4,)])
    assert agree[0].all()                       # accepted only when a label accepts it; the crosswalk never does


def test_a_reading_that_misses_the_recorded_errors_is_refused():
    score = np.ones((3, 2 * S + 1, 2 * S + 1))
    agree = np.ones_like(score)
    agree[0, S, S] = 0
    with pytest.raises(ValueError, match="not the graded reading"):
        e100.set_record("V", score, agree, np.ones(3, bool), np.arange(3), recorded={"n": 3, "errors": 2})
    r = e100.set_record("V", score, agree, np.ones(3, bool), np.arange(3), n_boot=20, recorded={"n": 3, "errors": 1})
    assert r["reproduces_record"] and r["errors_at_labelled_pixel"] == 1


def test_prereg_status_reads_the_status_line():
    assert e100.prereg_status("# t\n\n**Status: frozen on 11 October 2026.**") == "frozen"
    assert e100.prereg_status("# t\n\n**Status: draft, written 10 October 2026.**") == "draft"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    return e100.smoke(out_dir=str(tmp_path_factory.mktemp("exp100")), log=lambda *a: None)


def test_smoke_recovers_the_planted_offsets(smoke):
    for key, r in smoke["planted"].items():
        want = r["planted"]
        assert r["n_full_block"] == e100.SMOKE_PLANTED_N
        assert r["estimate"]["south_px"] == pytest.approx(want["south_px"], abs=0.25)
        assert r["estimate"]["east_px"] == pytest.approx(want["east_px"], abs=0.25)
        assert r["call"] == ("aligned" if (want["south_px"], want["east_px"]) == (0, 0) else "offset"), key
        best = r["mean_agreement_surface"][want["south_px"] + S][want["east_px"] + S]
        assert best == pytest.approx(r["agreement_best_over_surface"])


def test_smoke_reads_exp98s_and_exp99s_points_as_they_did(smoke):
    sets = smoke["sets"]
    assert (sets["T"]["n"], sets["V"]["n"]) == (10, 40)        # exp98's smoke: 10 training, 40 graded points
    assert sets["S_strict"]["n"] == sets["S_lenient"]["n"] == 22
    assert sets["T"]["errors_at_labelled_pixel"] == 1          # one training label set wrong on purpose
    for r in sets.values():
        assert r["call"] in e100.CALLS
        assert r["n_full_block"] + r["n_excluded_partial_block"] == r["n"]
    assert smoke["mapping"]["exp98"]["reprojected"] == 1       # exp98's smoke puts one point in another CRS
    assert smoke["mapping"]["exp99"]["reprojected"] == 22      # every TimeSync plot is reprojected from WGS84
    e98.check_no_coordinates(smoke)


def test_the_by_task_follow_up_finds_a_planted_offset_in_every_task(tmp_path):
    """exp100_by_task (descriptive, after the run): on a map with a planted (2, -1) offset most tasks' own best
    position is the planted one (ten points per task, on a map of 4-px blocks, can tie between neighbours), every one
    lies west of the labelled pixel, and leaving any task out keeps the estimate at the planted offset."""
    import exp100_by_task as bt
    path, points, labels, clusters = e100.make_planted(str(tmp_path), 2, -1)
    entries, _, _ = e98.find_outputs(path)
    grids = [e98.open_grid(e) for e in entries]
    gi, rows, cols, *_ = e98.read_points(grids, e98.locate_all(points, grids))
    blocks = e100.read_blocks(grids, gi, rows, cols)
    r = bt.by_task(blocks, [(int(c),) for c in labels], [f"t{c:02d}" for c in clusters])
    assert r["n_tasks"] == e100.SMOKE_PLANTED_CLUSTERS and r["n"] == e100.SMOKE_PLANTED_N
    assert sum((t["best_south_px"], t["best_east_px"]) == (2, -1) for t in r["tasks"]) >= 15
    assert r["tasks_best_east_of_pixel"] == 0
    lo, hi = r["leave_one_task_out_east_px_range"]
    assert -1.25 < lo <= hi < -0.75
    e98.check_no_coordinates(r)


def test_the_by_task_smoke_runs_on_exp98s_inputs(tmp_path):
    import exp100_by_task as bt
    rec = bt.smoke(out_dir=str(tmp_path), log=lambda *a: None)
    assert rec["V"]["n"] <= 40 and rec["T"]["n"] <= 10 and rec["not_preregistered"] is True
    e98.check_no_coordinates(rec)


# ----------------------------------------------------------------------------- the recorded run, by a second route
# The claim ledger's crosschecks for exp100: the committed summary's estimates and calls recomputed from its own
# surfaces and intervals with plain numpy, and the follow-up's counts from its per-task rows.
SUMMARY = os.path.join(ROOT, "exp", "out", "exp100_summary.json")
BY_TASK = os.path.join(ROOT, "exp", "out", "exp100_by_task.json")


def _vertex(a, b, c):
    """The parabola through (-1, a), (0, b), (1, c): its vertex, 0 when the three make no peak."""
    curvature = (a + c) / 2 - b
    return 0.0 if curvature >= 0 else (a - c) / (4 * curvature)


def _rule(iv, at_edge):
    lo_s, hi_s = iv["south_px"]
    lo_e, hi_e = iv["east_px"]
    if at_edge:
        return "offset"
    if -0.5 <= lo_s and hi_s <= 0.5 and -0.5 <= lo_e and hi_e <= 0.5:
        return "aligned"
    if lo_s > 0.5 or hi_s < -0.5 or lo_e > 0.5 or hi_e < -0.5:
        return "offset"
    return "not determined"


@pytest.mark.skipif(not os.path.exists(SUMMARY), reason="exp100 not recorded")
def test_exp100_estimates_and_calls_by_a_second_route():
    import json
    rec = json.load(open(SUMMARY))
    assert rec["prereg_status"] == "frozen" and rec["dirty"] is False
    want = {"T": (783, 16), "V": (259, 33), "S_strict": (271, 91), "S_lenient": (271, 37)}
    for name, r in rec["sets"].items():
        assert (r["n"], r["errors_at_labelled_pixel"]) == want[name] and r["reproduces_record"] is True
        M = np.array(r["mean_score_surface"])
        A = np.array(r["mean_agreement_surface"])
        top = np.argwhere(M >= M.max() - 1e-12)
        i, j = top[np.argmin(((top - S) ** 2).sum(1))]
        assert (i - S, j - S) == (r["estimate"]["best_south_px"], r["estimate"]["best_east_px"])
        south = i - S + (_vertex(M[i - 1, j], M[i, j], M[i + 1, j]) if 0 < i < 2 * S else 0.0)
        east = j - S + (_vertex(M[i, j - 1], M[i, j], M[i, j + 1]) if 0 < j < 2 * S else 0.0)
        # the summary keeps the surfaces to 6 decimals; near a flat top that moves the vertex by up to about 1e-4 px
        assert south == pytest.approx(r["estimate"]["south_px"], abs=2e-3)
        assert east == pytest.approx(r["estimate"]["east_px"], abs=2e-3)
        assert _rule(r["interval"], i in (0, 2 * S) or j in (0, 2 * S)) == r["call"]
        assert A[S, S] == pytest.approx(r["agreement_at_labelled_pixel"]) and A[i, j] == pytest.approx(
            r["agreement_at_best_position"])
    V = rec["sets"]["V"]
    assert V["call"] == "offset" and round(V["agreement_at_labelled_pixel"] * 259) == 226
    assert round(V["agreement_at_best_position"] * 259) == 231
    assert [rec["sets"][k]["call"] for k in ("T", "S_strict", "S_lenient")] == ["not determined"] * 3
    T = np.array(rec["sets"]["T"]["mean_agreement_surface"])
    assert T[S, S] == T.max()                               # T's agreement peaks at the labelled pixel


@pytest.mark.skipif(not os.path.exists(BY_TASK), reason="exp100's follow-up not recorded")
def test_exp100_follow_up_counts_by_a_second_route():
    import json
    rec = json.load(open(BY_TASK))
    assert rec["not_preregistered"] is True and rec["dirty"] is False
    for name, n, tasks, east, west in (("V", 259, 20, 13, 3), ("T", 780, 21, 8, 8)):
        r = rec[name]
        rows = r["tasks"]
        assert (r["n"], r["n_tasks"], len(rows)) == (n, tasks, tasks) and sum(t["n"] for t in rows) == n
        assert sum(t["best_east_px"] > 0 for t in rows) == east == r["tasks_best_east_of_pixel"]
        assert sum(t["best_east_px"] < 0 for t in rows) == west == r["tasks_best_west_of_pixel"]
        loo = [x["east_px"] for x in r["leave_one_task_out"]]
        assert [min(loo), max(loo)] == r["leave_one_task_out_east_px_range"]
    assert [round(x, 2) for x in rec["V"]["leave_one_task_out_east_px_range"]] == [1.28, 2.09]
    assert [round(x, 2) for x in rec["T"]["leave_one_task_out_east_px_range"]] == [-0.25, 0.72]
