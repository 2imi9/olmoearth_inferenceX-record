"""The defects an adversarial audit of the package found on 2026-09-21, each pinned so it cannot return.

Every one of these shipped in 1.1.2 and produced a plausible wrong answer with exit 0, which is the failure mode
this project exists to prevent in the maps it audits. Each test states the user-visible consequence.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import os

import numpy as np
import pytest
from oe_inferencex.calibrate import fit_ranker


def test_fit_side_with_one_group_is_refused_rather_than_reported_as_always_side_a():
    """The high finding the 21 September audit left open. One group id collapses every fold; each held-out logit is
    NaN, NaN > 0 is False, and fit_side reported the "always a" rate as its fitted rule's held-out accuracy: 0.196
    on exp60 against an honest 0.830 cross-fitted by event, reversing the verdict against a 0.690 baseline."""
    import os
    from oe_inferencex.calibrate import fit_ranker, fit_side
    z = np.load(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "exp60_masks.npz"))
    fa, fb = {"margin": z["margin_A_s1pre"]}, {"margin": z["margin_B_s1post"]}
    one = np.zeros_like(z["event"])
    with pytest.raises(ValueError, match="at least two groups"):
        fit_side(fa, fb, z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"], groups=one, family="x")
    _, r = fit_side(fa, fb, z["A_s1pre"], z["B_s1post"], z["ok"], z["y_after"], groups=z["event"], family="x")
    assert round(r["held_out"]["share_right"], 3) == 0.830 and r["n_unscored_rows"] == 0
    with pytest.raises(ValueError, match="at least two groups"):
        fit_ranker({"m": z["margin_A_s1pre"]}, z["A_s1pre"] != z["y_after"], z["ok"], groups=one, family="x")


def test_reference_unstable_cue_quotes_exp23_as_recorded():
    """Finding 14: the cue quoted 14.3x on 27 scenes; exp23 recorded 13.7x on 24."""
    import json, os
    from oe_inferencex.explain import CUES
    t1 = json.load(open(os.path.join(os.path.dirname(__file__), "..", "exp", "out", "exp23_summary.json")))["tests"]["T1_enrichment"]
    c = CUES["reference_unstable"]
    assert c.share_errors == pytest.approx(t1["median_share_errors"], abs=5e-4)
    assert c.share_correct == pytest.approx(t1["median_share_correct"], abs=5e-5)
    assert round(c.enrichment, 1) == 13.7 and "24" in c.reference
