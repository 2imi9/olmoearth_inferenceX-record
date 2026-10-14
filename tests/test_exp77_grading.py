"""The logic that will grade exp77's preregistered predictions, checked before the run's numbers exist.

The experiment itself needs torch and the cached embeddings; these functions are numpy only, and they are the
part that must not be wrong: they decide whether a prediction held.
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("e77", os.path.join(ROOT, "exp", "exp77_scene_contamination.py"))
e77 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e77)          # module level is torch-free by design; torch is imported inside the run


def _rows(pairs, floor=0.002, margin_best=True):
    return [{"task": f"t{i}", "contamination": {"gap": g}, "gain": gn, "seed_floor": floor,
             "excess_aurc": {"margin": 0.01, "scene_typicality": 0.02 if margin_best else 0.001}}
            for i, (g, gn) in enumerate(pairs)]


def test_the_contamination_gap_is_measured_inside_the_confident_half_only():
    """P1 asks about confident windows. A signal present only among the UNCONFIDENT errors must not count."""
    rng = np.random.default_rng(0)
    n = 600
    margin = np.linspace(0, 1, n)                      # the top half is the confident half
    err = (rng.random(n) < 0.25).astype(float)
    typ = rng.random(n)
    typ[(margin < 0.5) & (err > 0)] += 5.0             # a huge effect, entirely outside the confident half
    g = e77.contamination_gap(typ, err, margin)
    assert abs(g["gap"]) < 0.2, "an effect among unconfident errors must not be read as P1's effect"
    typ2 = rng.random(n)
    typ2[(margin >= 0.5) & (err > 0)] += 1.0           # the same effect where P1 looks for it
    assert e77.contamination_gap(typ2, err, margin)["gap"] > 0.5


def test_the_gap_reports_how_much_of_the_problem_it_covers_and_declines_when_too_thin():
    rng = np.random.default_rng(1)
    err = (rng.random(400) < 0.2).astype(float)
    g = e77.contamination_gap(rng.random(400), err, rng.random(400))
    assert 0.0 <= g["share_of_errors_confident"] <= 1.0 and g["n_confident_errors"] >= 5
    assert e77.contamination_gap(np.zeros(20), np.zeros(20), np.zeros(20)) is None, "no errors, no claim"
    almost = np.zeros(400); almost[:4] = 1.0
    assert e77.contamination_gap(rng.random(400), almost, rng.random(400)) is None, "four errors cannot carry P1"


def test_a_gain_inside_the_seed_floor_is_not_a_gain():
    """The failure mode this guard exists for: probe noise read as an improvement."""
    pairs = [(0.05, 0.02), (0.04, 0.02), (0.03, 0.02), (0.02, 0.02), (0.01, 0.0), (-0.01, 0.0), (-0.02, 0.0)]
    assert e77.verdicts(_rows(pairs, floor=0.002))["P2_decontamination_improves_accuracy"]["holds"]
    wide = e77.verdicts(_rows(pairs, floor=0.05))["P2_decontamination_improves_accuracy"]
    assert not wide["holds"] and wide["n"] == 0, "every gain sits inside the floor, so none of them counts"


def test_a_gain_below_the_stated_size_is_not_a_gain():
    just_under = [(0.05, e77.MIN_GAIN - 1e-9)] * 7
    assert e77.verdicts(_rows(just_under))["P2_decontamination_improves_accuracy"]["n"] == 0
    just_over = [(0.05, e77.MIN_GAIN)] * 7
    assert e77.verdicts(_rows(just_over))["P2_decontamination_improves_accuracy"]["n"] == 7


def test_p3_is_a_rank_correlation_and_fails_when_the_diagnosis_points_the_wrong_way():
    rising = _rows([(0.01, 0.001), (0.02, 0.005), (0.03, 0.01), (0.04, 0.02), (0.05, 0.03), (0.06, 0.04), (0.07, 0.05)])
    assert e77.verdicts(rising)["P3_the_diagnosis_predicts_where_it_helps"]["spearman"] == pytest.approx(1.0)
    inverted = _rows([(0.01, 0.05), (0.02, 0.04), (0.03, 0.03), (0.04, 0.02), (0.05, 0.01), (0.06, 0.005), (0.07, 0.001)])
    p3 = e77.verdicts(inverted)["P3_the_diagnosis_predicts_where_it_helps"]
    assert p3["spearman"] == pytest.approx(-1.0) and not p3["holds"]
    flat = _rows([(0.01, 0.01)] * 7)
    assert e77.verdicts(flat)["P3_the_diagnosis_predicts_where_it_helps"]["spearman"] is None, "no spread, no correlation"


def test_p4_catches_a_new_reading_that_beats_the_margin():
    """If the record's main result were overturned here, the run must say so rather than pass quietly."""
    kept = e77.verdicts(_rows([(0.01, 0.01)] * 7, margin_best=True))["P4_no_new_reading_beats_the_margin"]
    assert kept["holds"] and kept["n_tasks_beaten"] == 0
    lost = e77.verdicts(_rows([(0.01, 0.01)] * 7, margin_best=False))["P4_no_new_reading_beats_the_margin"]
    assert not lost["holds"] and lost["n_tasks_beaten"] == 7


def test_every_prediction_states_the_threshold_it_was_graded_against():
    v = e77.verdicts(_rows([(0.01, 0.01)] * 7))
    assert len(v) == 4 and all("holds" in d and "threshold" in d for d in v.values())


def test_aligning_rows_by_a_label_hash_needs_a_permutation_check():
    """exp54's cross-encoder block aligned rows by a hash of the label tile. Identical tiles collide to one key, so
    where they are not adjacent the reorder permutes rows even when a model is aligned against itself, while every
    other condition in the guard still passes. That is what corrupted its phi on PASTIS (reported 2026-09-20)."""
    def reorder(k0, dec, guard_is_permutation):
        pos0 = {k: i for i, k in enumerate(k0)}
        idx = np.array([pos0.get(x, -1) for x in k0])
        passes = (idx >= 0).all() and (len(np.unique(idx)) == len(idx) == len(k0) if guard_is_permutation
                                       else len(idx) == len(k0))
        return passes, (dec[np.argsort(idx)] if passes else None)

    dec = np.array([10, 20, 30, 40])
    scattered = ["a", "b", "a", "c"]          # identical tiles, not adjacent
    passes, out = reorder(scattered, dec, guard_is_permutation=False)
    assert passes and not np.array_equal(out, dec), "the old guard passes and scrambles a model against itself"
    assert not reorder(scattered, dec, guard_is_permutation=True)[0], "the permutation check refuses it"

    adjacent = ["a", "a", "b", "c"]           # identical tiles side by side: the reorder happens to be the identity
    passes, out = reorder(adjacent, dec, guard_is_permutation=False)
    assert passes and np.array_equal(out, dec), "which is why only some tasks were corrupted"

    unique = ["a", "b", "c", "d"]
    passes, out = reorder(unique, dec, guard_is_permutation=True)
    assert passes and np.array_equal(out, dec), "unique keys pass the permutation check and are unchanged"
