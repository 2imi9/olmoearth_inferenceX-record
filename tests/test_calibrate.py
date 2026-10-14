"""oe_inferencex.calibrate on synthetic fixtures and on the committed exp60 decisions: held-out gains, the family lock,
JSON round trips, and the logistic fit itself.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import os

import numpy as np
import pytest

from oe_inferencex.calibrate import fit_side, side_features

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def _need(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not committed")
    return path


def test_fit_side_on_the_recorded_exp60_decisions():
    """The radar pair across the event from the committed masks: margins of both sides, cross-fitted by event; the
    fitted rule beats the coin and does not lose to the raw margin rule."""
    z = np.load(_need("exp60_masks.npz"))
    a, b, ok, lab = z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"]
    fa, fb = {"margin": z["margin_A_s1pre"]}, {"margin": z["margin_B_s1post"]}
    fusion, rep = fit_side(fa, fb, a, b, ok, lab, groups=z["event"], family="OlmoEarth v1 frozen S1 head")
    assert rep["n_disagree"] == 29700
    assert rep["held_out"]["share_right"] > 0.5
    assert rep["held_out"]["share_right"] >= rep["baseline"]["share_right"] - 0.02
    with pytest.raises(ValueError):
        fusion.score(side_features(fa, fb), family="FT-S2 v1")
