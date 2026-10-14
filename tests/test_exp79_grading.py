"""exp79's verdicts on synthetic inputs, before the cluster runs: each preregistered bar must fail for its own
reason and for nothing else, and the exact coverage that P5 is judged against must equal a brute-force enumeration."""
import importlib.util
import itertools
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("exp79", os.path.join(ROOT, "exp", "exp79_seed_floors.py"))
e79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e79)

ENCS = list(e79.ENCODERS)                 # the sixteen, by name; "all" in the grading means all of these


def _rec(per):
    """The synthetic record (task sets, seed-0 headroom, ordering) a synthetic export is graded against."""
    r = e79.synthetic_record(per)
    return r["tasks"], r["headroom"], (r["headroom"], r["order"])


def _gate(per, **kw):
    tasks, h, _ = _rec(per)
    return e79.gate(per, recorded=kw.pop("recorded", tasks), headroom_ref=kw.pop("headroom_ref", h))


def _p3(per, n_seeds, **kw):
    return e79.grade_p3(per, n_seeds, record=kw.pop("record", _rec(per)[2]), **kw)


def test_gate_names_the_encoder_whose_seed0_misses_the_record():
    ok = _gate(e79.synthetic_export(ENCS))
    assert ok["holds"] and ok["encoders_failing"] == [] and ok["encoders_missing"] == []
    bad = _gate(e79.synthetic_export(ENCS, off_gate=("satlas_base",)))
    assert not bad["holds"] and bad["encoders_failing"] == ["satlas_base"]
    assert bad["per_encoder"]["satlas_base"]["max_abs_diff"] >= e79.GATE_TOL > bad["per_encoder"]["anysat"]["max_abs_diff"]


def test_gate_refuses_an_encoder_with_no_recorded_accuracy_to_compare():
    per = e79.synthetic_export(ENCS)
    for v in per["anysat"]["tasks"].values():
        v["accuracy_minus_recorded"] = None
    g = _gate(per)
    assert not g["per_encoder"]["anysat"]["holds"] and g["encoders_failing"] == ["anysat"]


def test_gate_fails_when_a_recorded_task_is_missing_from_the_export_and_names_it():
    """The export files any load failure under `absent`; without this the lost task would just shrink every count."""
    per = e79.synthetic_export(ENCS)
    tasks = _rec(per)[0]                                 # the record's task sets, before the loss
    per["croma_base"]["tasks"].pop("t2")
    per["croma_base"]["absent"] = [{"task": "t2", "reason": "RuntimeError: CUDA out of memory"}]
    h = e79.headroom_per_seed(per, 0)                    # headroom taken after the loss, so only the task-set clause can catch it
    g = e79.gate(per, recorded=tasks, headroom_ref=h)
    c = g["per_encoder"]["croma_base"]
    assert not g["holds"] and g["encoders_failing"] == ["croma_base"]
    assert c["missing_from_export"] == ["t2"] and c["n_tasks_recorded"] == 6 and c["n_tasks_exported"] == 5
    assert c["absent_reasons"][0]["reason"].startswith("RuntimeError")


def test_gate_fails_when_seed0_headroom_does_not_reproduce_the_record():
    per = e79.synthetic_export(ENCS)
    tasks, h, _ = _rec(per)
    h = dict(h); h["panopticon"] += 5e-3               # the record says something else for this encoder
    g = e79.gate(per, recorded=tasks, headroom_ref=h)
    assert not g["holds"] and g["encoders_failing"] == ["panopticon"]
    assert g["per_encoder"]["panopticon"]["headroom_abs_diff"] == pytest.approx(5e-3)


def test_gate_fails_on_a_partial_run_and_names_the_missing_encoders():
    per = e79.synthetic_export(ENCS[:5])
    g = _gate(per)
    assert not g["holds"] and g["encoders_failing"] == [] and g["encoders_missing"] == sorted(ENCS[5:])


def test_p1_fails_on_one_encoder_one_seed_and_says_which():
    per = e79.synthetic_export(ENCS, n_seeds=3, n_tasks=8)
    assert e79.grade_p1(per, 3)["holds"]
    # push one encoder under the 75% bar on the last seed only: 3 of 8 losses
    flips = tuple(("galileo_tiny", f"t{j}") for j in range(3))
    bad = e79.grade_p1(e79.synthetic_export(ENCS, n_seeds=3, n_tasks=8, flip=flips), 3)
    assert not bad["holds"] and bad["failing_encoder_seed"] == [("galileo_tiny", 2)]
    assert bad["min_share"]["galileo_tiny"] == pytest.approx(5 / 8) and bad["min_wins"]["galileo_tiny"] == 5


def test_p1_fails_on_a_partial_run_or_a_short_seed_count_even_when_every_row_passes():
    """'All 16 encoders under all 10 seeds' cannot hold on three encoders, or on an export run with --seeds 5."""
    partial = e79.grade_p1(e79.synthetic_export(ENCS[:3], n_seeds=3, n_tasks=8), 3)
    assert not partial["holds"] and partial["failing_encoder_seed"] == [] and len(partial["encoders_missing"]) == 13
    per = e79.synthetic_export(ENCS, n_seeds=3, n_tasks=8)
    for v in per["clay_large"]["tasks"].values():
        v["seeds"] = v["seeds"][:2]
    short = e79.grade_p1(per, 3)
    assert not short["holds"] and short["encoders_short_of_seeds"] == ["clay_large"]


def test_p1_needs_the_sign_test_as_well_as_the_share():
    """Two tasks won of two is a share of 1.0 and a sign test p of 0.25; the bar needs both."""
    per = e79.synthetic_export(["only"], n_seeds=1, n_tasks=2)
    r = e79.grade_p1(per, 1)
    assert not r["holds"] and r["min_share"]["only"] == 1.0


def test_p2_counts_robust_wins_and_names_every_flipper_with_its_spread():
    per = e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24)
    r = e79.grade_p2(per, n_seeds=4)
    assert r["holds"] and r["n_robust"] == 24 and r["flips"] == {}
    flips = tuple(("olmoearth_base", f"t{j}") for j in range(5))
    r = e79.grade_p2(e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24, flip=flips), n_seeds=4)
    assert not r["holds"] and r["n_robust"] == 19 and sorted(r["flips"]) == [f"t{j}" for j in range(5)]
    f = r["flips"]["t0"]
    assert f["wins"] == 3 and f["of"] == 4 and f["min"] < 0 < f["mean"] and f["sd"] > 0
    # exactly 20 robust is the bar, and it is inclusive
    flips4 = tuple(("olmoearth_base", f"t{j}") for j in range(4))
    assert e79.grade_p2(e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24, flip=flips4), n_seeds=4)["holds"]


def test_p2_says_so_when_base_has_not_run():
    assert e79.grade_p2(e79.synthetic_export(["anysat"]))["holds"] is None


def test_p2_needs_every_seed_on_every_task_and_the_records_task_count():
    """A --seeds 5 export or a lost task must not pass as 'all 10 seeds on at least 20 of 24'."""
    per = e79.synthetic_export(["olmoearth_base"], n_seeds=4, n_tasks=24)
    assert e79.grade_p2(per, n_seeds=4)["holds"]
    fewer = e79.grade_p2(per, n_seeds=10)                              # asked for ten, the export has four
    assert not fewer["holds"] and not fewer["complete"] and fewer["n_robust"] == 24
    per["olmoearth_base"]["tasks"].pop("t23")                          # 23 tasks, all robust
    lost = e79.grade_p2(per, n_seeds=4)
    assert not lost["holds"] and not lost["complete"] and lost["of"] == 23 and lost["of_recorded"] == 24


def _with_headroom(per, enc, seed, value):
    """Set one encoder's margin excess AURC at one seed so its median headroom becomes `value`."""
    for v in per[enc]["tasks"].values():
        s = v["seeds"][seed]
        e, n = s["error_rate"], s["n_units"]
        gap = e - e79.hbe.metrics.oracle_aurc(n, int(round(e * n)))
        s["signals"]["margin"]["excess_aurc"] = (1 - value) * gap


def test_p3_holds_when_the_top_two_swap_and_fails_when_the_middle_reaches_the_top():
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    rec = _rec(per)[2]
    r = e79.grade_p3(per, 2, record=rec)
    first, second, third = rec[1][:3]
    assert r["holds"] and r["top_set"] == sorted([first, second, third]) and r["encoders_missing"] == []
    # swap the top two on seed 1: still inside the top set, rho stays high
    h = rec[0]
    _with_headroom(per, first, 1, h[second] - 1e-3)
    _with_headroom(per, second, 1, h[first] + 1e-3)
    r2 = e79.grade_p3(per, 2, record=rec)
    assert r2["holds"] and r2["per_seed"][1]["best"] == second
    # rank 8 of 16 reaching rank 1 on seed 1 keeps Spearman above 0.9, so only the set clause catches it
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    eighth = rec[1][7]
    _with_headroom(per, eighth, 1, 0.99)
    r3 = e79.grade_p3(per, 2, record=rec)
    assert r3["per_seed"][1]["rho"] >= e79.P3_RHO, r3["per_seed"][1]["rho"]
    assert not r3["holds"] and r3["per_seed"][1]["best"] == eighth and not r3["per_seed"][1]["best_in_top"]


def test_p3_takes_its_named_sets_from_the_record_not_from_the_rerun():
    """The plan names the record's top three. A rerun whose seed 0 would put the fourth encoder third must not
    quietly change the set it is tested against; the gate is what catches a seed 0 that disagrees."""
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    h, order = _rec(per)[2]
    third, fourth = order[2], order[3]
    _with_headroom(per, fourth, 0, h[third] + 1e-4)     # the rerun's seed 0 ranks the fourth above the third
    _with_headroom(per, fourth, 1, h[third] + 1e-4)
    r = e79.grade_p3(per, 2, record=(h, order))
    assert r["top_set"] == sorted(order[:3]) and fourth not in r["top_set"]


def test_p3_fails_on_a_partial_run():
    per = e79.synthetic_export(ENCS[:8], n_seeds=2, n_tasks=6)
    r = e79.grade_p3(per, 2, record=_rec(per)[2])
    assert not r["holds"] and len(r["encoders_missing"]) == 8


def test_p3_fails_on_a_scrambled_ordering():
    per = e79.synthetic_export(ENCS, n_seeds=2, n_tasks=6)
    rec = _rec(per)[2]
    vals = [rec[0][e] for e in ENCS]
    for enc, v in zip(ENCS, vals[::-1]):            # reverse the ordering on seed 1
        _with_headroom(per, enc, 1, v)
    r = e79.grade_p3(per, 2, record=rec)
    assert not r["holds"] and r["min_rho"] < 0


def _rows(deffs, naive, srs=0.95, exact=None):
    return {f"t{i}": {"n_units": 10000, "error_rate": 0.1, "design_effect": d, "naive_coverage": c,
                      "srs_coverage": srs, "exact_srs_coverage": exact, "cluster_coverage": 0.93, "srs_half_width": 0.03}
            for i, (d, c) in enumerate(zip(deffs, naive))}


def _carried(per):
    return {enc: set(rows) for enc, rows in per.items()}


def test_p4_allows_two_exceptions_per_encoder_and_needs_thirteen_encoders():
    good = _rows([3.0] * 5 + [1.1, 1.2], [0.5] * 5 + [0.94, 0.95])           # 5 of 7 under, two exceptions
    per = {f"e{i}": good for i in range(13)}
    r = e79.grade_p4(per, carried=_carried(per))
    assert r["holds"] and r["n_encoders_holding"] == 13 and r["per_encoder"]["e0"]["exceptions"] == ["t5", "t6"]
    three = _rows([3.0] * 4 + [1.1] * 3, [0.5] * 4 + [0.94] * 3)               # three exceptions: fails
    per["e0"] = three
    r = e79.grade_p4(per, carried=_carried(per))
    assert not r["holds"] and r["failing"] == ["e0"] and r["n_encoders_holding"] == 12


def test_p4_needs_the_design_effect_as_well_as_the_coverage_count():
    weak = _rows([1.5] * 7, [0.8] * 7)                                         # all under-cover, but median deff 1.5
    per = {f"e{i}": weak for i in range(16)}
    r = e79.grade_p4(per, carried=_carried(per))
    assert not r["holds"] and r["n_encoders_holding"] == 0


def test_p4_counts_per_encoder_over_the_tasks_it_carries():
    five = _rows([3.0] * 3 + [1.1] * 2, [0.5] * 3 + [0.95] * 2)                # carries 5, two exceptions: holds
    four = _rows([3.0] * 2 + [1.1] * 2, [0.5] * 2 + [0.95] * 2)                # carries 4, two exceptions: holds
    per = {**{f"e{i}": five for i in range(7)}, **{f"f{i}": four for i in range(6)}}
    r = e79.grade_p4(per, carried=_carried(per))
    assert r["holds"] and r["per_encoder"]["f0"]["carried"] == 4


def test_p4_counts_a_task_the_record_carries_but_the_export_lost_as_an_exception():
    """Two exceptions plus one lost task is three: a lost task cannot be shown to under-cover."""
    good = _rows([3.0] * 5 + [1.1, 1.2], [0.5] * 5 + [0.94, 0.95])
    per = {f"e{i}": dict(good) for i in range(13)}
    carried = _carried(per)
    per["e0"].pop("t0")                                                        # lost, but the record carries it
    r = e79.grade_p4(per, carried=carried)
    assert not r["holds"] and r["failing"] == ["e0"]
    assert r["per_encoder"]["e0"]["missing"] == ["t0"] and r["per_encoder"]["e0"]["carried"] == 7


def test_p5_judges_a_low_cell_against_the_exact_coverage_not_the_round_number():
    one = {("a", "t0")}
    fine = {"a": _rows([3.0], [0.5], srs=0.95)}
    assert e79.grade_p5(fine, expected=one)["holds"]
    discrete = {"a": _rows([3.0], [0.5], srs=0.925, exact=0.930)}              # within tolerance of what Wilson can do
    r = e79.grade_p5(discrete, expected=one)
    assert r["holds"] and r["below_bar_but_within_exact"] == [("a", "t0", 0.925, 0.930)]
    bug = {"a": _rows([3.0], [0.5], srs=0.900, exact=0.930)}                   # 0.03 below, past any tolerance: a defect
    r = e79.grade_p5(bug, expected=one)
    assert not r["holds"] and r["failing"] == [("a", "t0")]
    no_exact = {"a": _rows([3.0], [0.5], srs=0.925, exact=None)}               # low with nothing to compare: fails
    assert not e79.grade_p5(no_exact, expected=one)["holds"]


def test_p5_tolerance_is_three_monte_carlo_standard_errors_never_below_a_hundredth():
    """At 0.93 and R = 2,000 one SE is 0.0057; a flat 0.01 would be 1.75 SE and fail ~4 of 112 honest cells."""
    se = math.sqrt(0.93 * 0.07 / e79.e78.R_DRAWS)
    assert se == pytest.approx(0.0057, abs=2e-4)
    assert e79.p5_tolerance(0.93) == pytest.approx(3 * se) and e79.p5_tolerance(0.93) > 0.01
    assert e79.p5_tolerance(0.999) == 0.01                                     # the floor, where the SE is tiny
    cell = {"a": _rows([3.0], [0.5], srs=0.930 - 2.5 * se, exact=0.930)}       # 2.5 SE below: chance, not a bug
    assert e79.grade_p5(cell, expected={("a", "t0")})["holds"]


def test_p5_fails_when_a_recorded_cell_is_missing():
    r = e79.grade_p5({"a": _rows([3.0], [0.5], srs=0.95)}, expected={("a", "t0"), ("a", "t1")})
    assert not r["holds"] and r["cells_missing"] == [("a", "t1")] and r["of"] == 2


@pytest.mark.parametrize("N,K,B", [(10, 3, 4), (12, 5, 5), (9, 1, 3), (15, 7, 6)])
def test_exact_srs_coverage_equals_brute_force_enumeration(N, K, B):
    """Every subset of size B, counted by hand; the hypergeometric sum must match it exactly."""
    theta = K / N
    hits = 0
    for combo in itertools.combinations(range(N), B):
        k = sum(1 for i in combo if i < K)
        lo, hi = e79.e78.wilson(k, B, N)
        hits += lo <= theta <= hi
    brute = hits / math.comb(N, B)
    assert e79.exact_srs_coverage(N, K, B) == pytest.approx(brute, abs=1e-12)


def test_exact_srs_coverage_is_one_at_a_census_and_below_one_before_it():
    """A census has no sampling error. Before 2026-09-22 exp78's wilson returned a zero-width interval around
    Wilson's shrunk centre at B == N, so this coverage was 0.0; the branch is unreachable in exp78 and exp79."""
    assert e79.exact_srs_coverage(50, 10, 50) == pytest.approx(1.0)
    assert e79.e78.wilson(10, 50, 50) == (0.2, 0.2)
    assert 0.85 < e79.exact_srs_coverage(2000, 150, 300) < 1.0
