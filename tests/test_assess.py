"""The assessor (oe_inferencex.assess): recipe items 1, 2 and 6 on a synthetic prediction, its JSON view, and the
input-condition layer.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json
import os
import re

import numpy as np
import pytest

from oe_inferencex.assess import SCOPE_ASSESS, SCOPE_ASSESS_K, boundary_first_score


def test_boundary_first_order_matches_exp36_construction_and_recorded_captures():
    """assess.boundary_first_score is exp36's lexicographic rule, and per-tile captures with it reproduce exp38's
    recorded 'boundary, then confidence' column from exp37's per-window table."""
    import csv
    import os

    import numpy as np
    import pytest

    from oe_inferencex.assess import boundary_first_score
    from oe_inferencex.metrics import capture_at_budget_expected
    from oe_inferencex.signals import midrank_pct

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")
    conf = np.array([[0.1, 0.5], [0.5, 0.9]])
    bnd = np.array([[0.0, 0.25], [0.0, 0.5]])
    s = boundary_first_score(-conf, bnd)
    assert s.tolist() == (np.where(bnd > 0, 2.0, 0.0) + midrank_pct(-conf).reshape(2, 2)).tolist()
    assert s[0, 1] > s[1, 1] > s[0, 0] > s[1, 0]                       # boundary windows first (least confident of them first), then the interior by confidence

    z = np.load(os.path.join(out, "exp37_patches_bolivia.npz"))
    rec = {r["unit"]: r for r in csv.DictReader(open(os.path.join(out, "exp38_ndwi_first.csv"))) if r["part"] == "B"}
    for tile in list(rec)[:40]:
        m = z["tile"] == int(tile)
        cap = capture_at_budget_expected(boundary_first_score(z["conf"][m], z["boundary"][m]), z["err"][m], (0.05, 0.1, 0.2))
        for b in (0.05, 0.1, 0.2):
            assert cap[b] == pytest.approx(float(rec[tile][f"boundary, then confidence @{b}"]), abs=1e-12)


# --------------------------------------------------------------------------- the input-condition layer (1.4.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_scope_notes_quote_exp88_as_recorded():
    """Every number the scope notes quote is exp88's, read from its summary (as test_explain reads exp37's), or its
    matched-head follow-up's, read from exp/out/exp88_matched_head.json. The follow-up's rows for the probe trained on
    radar plus optical are exp88's own numbers, and the test checks that they agree."""
    from oe_inferencex.estimate import CONFIDENCE_REFUSAL, SCOPE_CERTIFY, SCOPE_ESTIMATE
    rec = json.load(open(os.path.join(ROOT, "exp", "out", "exp88_summary.json")))
    mh = json.load(open(os.path.join(ROOT, "exp", "out", "exp88_matched_head.json")))
    pre = rec["prereg"]
    assert rec["prereg_status"] == "frozen" and pre["graded_on"] == "olmoearth_base, probe seed 0"
    base = rec["results"]["olmoearth_base"]["seeds"]["0"]["families"]["pastis"]["conditions"]
    assert mh["probe_seed"] == 0
    rows = mh["encoders"]["olmoearth_base"]["families"]["pastis"]["rows"]
    mismatched, matched, full = rows["s1s2_head_on_s1"], rows["s1_head_on_s1"], rows["s1s2_head_on_s1s2"]
    # the probe trained on radar plus optical: the follow-up's rows are exp88's, on the same windows
    for row, cond in ((mismatched, "optical_missing"), (full, "full")):
        assert row["confident_share"] == base[cond]["confident_errors"]["share"]
        assert row["margin_auroc"] == pytest.approx(base[cond]["ranking"]["margin_auroc"], abs=1e-6)
        assert row["n_units"] == mh["encoders"]["olmoearth_base"]["families"]["pastis"]["n_units"]
    assert matched["source"]["path"] == "exp/out/exp78_units/pastis_sentinel1.npz" and matched["source"]["committed"]

    p2 = pre["P2"]                                  # errors at least as confident as the typical correct full-input window
    assert p2["confident_share_optical_missing"] == base["optical_missing"]["confident_errors"]["share"]
    assert p2["confident_share_full"] == base["full"]["confident_errors"]["share"]
    # the graded numbers are OlmoEarth Base's (graded_on above), and the note says so
    assert (f"OlmoEarth Base's probe trained on radar plus optical was {100 * mismatched['error_rate']:.1f}% wrong when "
            "run on radar alone, as under cloud.") in SCOPE_ASSESS                                          # 73.6%
    assert f"Of its errors, {100 * p2['confident_share_optical_missing']:.1f}% were at least as confident" in SCOPE_ASSESS  # 59.8%
    assert f"against {100 * p2['confident_share_full']:.1f}% of its errors with full input" in SCOPE_ASSESS  # 6.0%
    # the probe trained on radar alone, on the same windows: its AUROC first, since the confident share is read at the
    # threshold of the probe trained on both, not at its own
    assert (f"A probe trained on radar alone was {100 * matched['error_rate']:.1f}% wrong and ranked its errors with an "
            f"AUROC of {matched['margin_auroc']:.2f}, against {base['full']['ranking']['margin_auroc']:.2f} for the probe "
            f"trained on both with full input; {100 * matched['confident_share']:.1f}% of its errors reached the same "
            "threshold, which is set by the probe trained on both.") in SCOPE_ASSESS  # 28.4%, 0.79, 0.83, 3.9%
    assert mh["encoders"]["olmoearth_base"]["families"]["pastis"]["threshold_confident"] == pytest.approx(
        mh["encoders"]["olmoearth_base"]["families"]["pastis"]["threshold_recorded_by_exp88"], abs=1e-6)
    # OlmoEarth Large, over its five probe seeds: the full-input share is one number to a tenth of a point, and the
    # optical-missing share a range
    large = [v["families"]["pastis"]["conditions"] for v in rec["results"]["olmoearth_large"]["seeds"].values()]
    full_l = {f"{100 * c['full']['confident_errors']['share']:.1f}" for c in large}
    miss_l = [c["optical_missing"]["confident_errors"]["share"] for c in large]
    assert len(large) == 5 and len(full_l) == 1
    assert (f"For OlmoEarth Large's probe trained on both inputs, the share rose only from {full_l.pop()}% to "
            f"{100 * min(miss_l):.1f}-{100 * max(miss_l):.1f}%.") in SCOPE_ASSESS                         # 5.7%, 12.8-13.8%
    china = pre["replication_china6"]["P2"]         # the direction on China 6: the confident share fell
    assert china["holds"] is False and china["confident_share_optical_missing"] < china["confident_share_full"]
    assert "On CropHarvest China 6, Base's probe trained on both showed no such rise (exp88)." in SCOPE_ASSESS
    assert "China_6" in rec["config"]["families"]["china6"]["optical_missing"]
    assert "though not on CropHarvest China 6 (exp88)" in CONFIDENCE_REFUSAL

    full_auc = base["full"]["ranking"]["margin_auroc"]
    missing_auc = pre["P3"]["per_family"]["pastis"]["margin_auroc"]
    assert missing_auc == base["optical_missing"]["ranking"]["margin_auroc"]
    assert (f"was {missing_auc:.2f} for a probe trained on radar plus optical and run on radar alone, against "
            f"{full_auc:.2f} on both inputs and {matched['margin_auroc']:.2f} for a probe trained on radar alone "
            "(exp88)") in SCOPE_ASSESS_K                                                                    # 0.59, 0.83, 0.79

    # the mixed map: each part's truth, and the pooled estimate of random samples of 300. The record's number is
    # the mean over its draws, so the note says "on average": one sample of 300 gives k/300, and 46.9% is not one
    p4 = pre["P4"]
    draws = rec["results"]["olmoearth_base"]["seeds"]["0"]["mixed_map"]["estimation"]["pooled"]
    assert p4["pooled_mean_estimate"] == draws["mean_estimate"] and rec["config"]["draws"] > 1
    assert f"{100 * p4['truth']['cloudy']:.1f}% against {100 * p4['truth']['clear']:.1f}% on the rest" in SCOPE_ESTIMATE
    assert (f"random samples of {rec['config']['sample']} estimated {100 * p4['pooled_mean_estimate']:.1f}% on average"
            in SCOPE_ESTIMATE)
    assert "a random sample of" not in SCOPE_ESTIMATE
    # the same map with the cloudy half read by the probe trained on radar alone: the rates still differ
    mm = mh["encoders"]["olmoearth_base"]["mixed_map"]
    assert mm["mismatched"]["error_rate"]["cloudy"] == pytest.approx(p4["truth"]["cloudy"], abs=1e-12)
    assert mm["matched"]["error_rate"]["clear"] == pytest.approx(p4["truth"]["clear"], abs=1e-12)
    assert mm["matched"]["cloudy_part"] == "s1_head_on_s1" and mm["matched"]["n_cloudy_windows"] == mm["mismatched"]["n_cloudy_windows"]
    assert (f"Read by a probe trained on radar alone, that half was still {100 * mm['matched']['error_rate']['cloudy']:.1f}% "
            f"wrong against {100 * mm['matched']['error_rate']['clear']:.1f}% (exp88).") in SCOPE_ESTIMATE  # 28.3%, 19.7%
    assert "(exp88)" in SCOPE_CERTIFY and not any(ch.isdigit() for ch in SCOPE_CERTIFY.replace("exp88", ""))
    # every percentage the notes quote is one of the numbers checked above
    quoted = {m for t in (SCOPE_ASSESS, SCOPE_ESTIMATE, SCOPE_ASSESS_K) for m in re.findall(r"\d+\.\d+", t)}
    assert quoted == {"73.6", "59.8", "6.0", "28.4", "3.9", "5.7", "12.8", "13.8", "0.59", "0.83", "0.79", "74.1", "19.7",
                      "46.9", "28.3"}
