"""exp91 (a vision-language model as the reviewer on Sen1Floods11 Bolivia) by a second route.

The recorded summary is recomputed from the committed labels and key with code written here: the agreement and the
reviewer's error counts by direct counting, the one-sided Clopper-Pearson bounds by their defining tail property in
exact rational arithmetic, and the hand-label interval from the hypergeometric interval at the population's size.
Nothing here needs the imagery or torch."""
import csv
import json
import os
from fractions import Fraction
from math import comb

from oe_inferencex.estimate import hypergeom_interval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "exp", "out")


def _load():
    labels = {d["id"]: d["label"] for d in json.load(open(os.path.join(OUT, "exp91_labels.json")))}
    key = list(csv.DictReader(open(os.path.join(OUT, "exp91_key.csv"))))
    summary = json.load(open(os.path.join(OUT, "exp91_summary.json")))
    return labels, key, summary


def _binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), p a Fraction, exactly."""
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))


def test_exp91_by_a_second_route():
    labels, key, s = _load()
    assert len(labels) == len(key) == 210 and set(labels) == {k["id"] for k in key}
    S = [k for k in key if k["set"] == "S"]
    C = [k for k in key if k["set"] == "C"]
    assert len(S) == 150 and sum(k["err"] == "1" for k in C) == 30 and sum(k["err"] == "0" for k in C) == 30
    as_class = {"water": "1", "land": "0"}

    # A: agreement with the hand labels on the windows the reviewer judged
    judged = [k for k in S if labels[k["id"]] != "?"]
    agree = sum(as_class[labels[k["id"]]] == k["hand"] for k in judged)
    assert len(judged) == s["S"]["judged"] == 104
    assert abs(agree / len(judged) - s["S"]["class_agreement"]) < 1e-12 and agree == 89

    # B: the reviewer's errors on the calibration set, over judged windows
    def says_wrong(k):
        return as_class[labels[k["id"]]] != k["map"]
    right_judged = [k for k in C if k["err"] == "0" and labels[k["id"]] != "?"]
    wrong_judged = [k for k in C if k["err"] == "1" and labels[k["id"]] != "?"]
    false_alarms = sum(says_wrong(k) for k in right_judged)
    misses = sum(not says_wrong(k) for k in wrong_judged)
    assert (len(right_judged), false_alarms, len(wrong_judged), misses) == (22, 0, 19, 14)
    assert (s["C"]["false_alarms"], s["C"]["misses"]) == (false_alarms, misses)

    # the Clopper-Pearson upper bound U solves P(X <= k | n, U) = 0.05: check it brackets that point exactly
    for k, n, u in ((false_alarms, len(right_judged), s["C"]["U0"]), (misses, len(wrong_judged), s["C"]["U1"])):
        lo, hi = Fraction(u) - Fraction(1, 10 ** 6), Fraction(u) + Fraction(1, 10 ** 6)
        assert _binom_cdf(k, n, lo) > Fraction(1, 20) > _binom_cdf(k, n, hi), (k, n, u)

    # C: the hand-label interval on S, from the hypergeometric interval at the population's size
    N = s["population"]["windows"]
    k_hand = sum(k["err"] == "1" for k in S)
    lo, hi = hypergeom_interval(k_hand, 150, N, 0.95)
    assert abs(lo - s["intervals"]["hand_labels"]["low"]) < 1e-12 and abs(hi - s["intervals"]["hand_labels"]["high"]) < 1e-12
    # the reviewer-label interval: ? counted right at the lower end and wrong at the upper end
    flagged = sum(labels[k["id"]] != "?" and says_wrong(k) for k in S)
    unjudged = sum(labels[k["id"]] == "?" for k in S)
    assert (flagged, unjudged) == (12, 46)
    v_lo, _ = hypergeom_interval(flagged, 150, N, 0.95)
    _, v_hi = hypergeom_interval(flagged + unjudged, 150, N, 0.95)
    assert abs(v_lo - s["intervals"]["vlm_labels"]["low"]) < 1e-12 and abs(v_hi - s["intervals"]["vlm_labels"]["high"]) < 1e-12
    # widened by the calibrated bounds: (low - U0) / (1 - U0) and high / (1 - U1), clipped to [0, 1]
    w_lo = max(0.0, (v_lo - s["C"]["U0"]) / (1 - s["C"]["U0"])); w_hi = min(1.0, v_hi / (1 - s["C"]["U1"]))
    assert (w_lo, w_hi) == (s["intervals"]["vlm_widened"]["low"], s["intervals"]["vlm_widened"]["high"]) == (0.0, 1.0)

    # the descriptive numbers found after the result
    d = s["descriptive_added_after_the_result"]
    assert sum(k["hand"] == k["map"] for k in judged) == 99 and abs(d["S_map_agreement_on_vlm_judged"] - 99 / 104) < 1e-12
    assert sum(k["hand"] == k["map"] for k in S) == 136
    assert sum(labels[k["id"]] == "?" or as_class[labels[k["id"]]] == k["hand"] for k in S) == 135
    wrong_all = [k for k in key if k["err"] == "1"]; right_all = [k for k in key if k["err"] == "0"]
    assert (sum(labels[k["id"]] == "?" for k in wrong_all), len(wrong_all)) == (20, 44) == tuple(d["pooled_unjudged_wrong"])
    assert (sum(labels[k["id"]] == "?" for k in right_all), len(right_all)) == (45, 166) == tuple(d["pooled_unjudged_correct"])
    errs = [k for k in judged if as_class[labels[k["id"]]] != k["hand"]]
    assert len(errs) == 15 and sum(labels[k["id"]] == "water" and k["hand"] == "0" for k in errs) == 13

    # the predictions as recorded
    assert s["prereg"] == {"P1": {"holds": False}, "P2": {"holds": True}, "P3": {"holds": True}}
    assert d["S_true_errors"] == k_hand == 14
    assert d["S_true_errors_flagged"] == sum(k["err"] == "1" and labels[k["id"]] != "?" and says_wrong(k) for k in S)


def test_exp91_key_matches_the_window_table_and_the_seed():
    """The key's map errors are exp37's, window by window, and S is the package's seed-91 random draw over the table."""
    import numpy as np
    from oe_inferencex.estimate import sample_for_estimation
    _, key, _ = _load()
    t = np.load(os.path.join(OUT, "exp37_patches_bolivia.npz"))
    err = (t["err"] > 0.5).astype(int)
    for k in key:
        i = int(k["index"])
        assert int(k["err"]) == err[i] and (int(k["chip"]), int(k["row"]), int(k["col"])) == (int(t["tile"][i]), int(t["row"][i]), int(t["col"][i]))
    draw = sample_for_estimation(t["conf"], 150, design="random", valid=np.ones(len(err), bool), seed=91)["indices"]
    assert sorted(int(k["index"]) for k in key if k["set"] == "S") == sorted(int(i) for i in draw)
    recorded = json.load(open(os.path.join(OUT, "exp91_key_sample.json")))["sample"]["indices"]
    assert sorted(recorded) == sorted(int(i) for i in draw)
