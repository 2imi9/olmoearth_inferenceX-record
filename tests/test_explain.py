"""The explanation layer (oe_inferencex.explain): cues, enrichment, and the review-set explanation.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json

import numpy as np
import pytest

import os
from oe_inferencex.explain import CUES, library_table

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp", "out")


def test_library_numbers_trace_to_the_recorded_shares():
    """The expert-label cues carry exp37's shares on identical Bolivia windows (exp/out/exp37_summary.json); the
    boundary shares also equal exp36's run-of-record boundary_share."""
    rec = json.load(open(os.path.join(OUT, "exp37_summary.json")))["part_b"]["analysis"]["cues"]
    for name in ("boundary", "low_confidence", "unstable", "ndwi_ambiguous", "dihedral_disagree"):
        assert CUES[name].share_errors == pytest.approx(rec[name]["share_errors"], abs=5e-4), name
        assert CUES[name].share_correct == pytest.approx(rec[name]["share_correct"], abs=5e-4), name
    b36 = json.load(open(os.path.join(OUT, "exp36_summary.json")))["part_b"]["boundary_share"]
    assert CUES["boundary"].share_errors == pytest.approx(b36["errors"], abs=5e-4)
    assert CUES["boundary"].share_correct == pytest.approx(b36["correct"], abs=5e-4)
    assert CUES["ndwi_ambiguous"].enrichment == pytest.approx(0.483 / 0.067)
    rows = library_table(quantile=0.2)
    assert {r["name"] for r in rows} == set(CUES) and all(r["source"] and r["quote"] for r in rows)
    assert "not yet measured" in CUES["osm_disagrees"].quote()


def test_the_library_carries_exp82s_per_task_verifications_and_reports_the_maps_boundary_prevalence():
    """The verified shares in the library equal exp82's artifact, the quoted range is the artifact's, and the
    review-set explanation states the map's own boundary prevalence beside that range."""
    import json
    import os
    from oe_inferencex import explain
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = json.load(open(os.path.join(root, "exp", "out", "exp82_summary.json")))["tasks"]
    # the boundary cue is exp82's own; the low-confidence cue is exp82's per-tile cut, the scene-level cut the tool
    # draws (review of 2026-09-23: the pooled task-wide cut enriched 2.5x to 5.1x, the per-scene one 1.4x to 3.4x)
    for cue, key in (("boundary", "boundary"), ("low_confidence", "low_confidence_per_tile")):
        v = explain.CUES[cue].verified
        assert set(v) == set(d)
        for t, (se, sc) in v.items():
            assert abs(se - d[t]["cues"][key]["share_errors"]) < 1e-4 and abs(sc - d[t]["cues"][key]["share_correct"]) < 1e-4
    lo_c, hi_c = explain.CUES["low_confidence"].verified_range
    assert round(lo_c, 1) == 1.4 and round(hi_c, 1) == 3.4
    lo, hi = explain.CUES["boundary"].verified_range
    assert round(lo, 2) == 1.24 and round(hi, 2) == 8.13
    assert "1.2x to 8.1x" in explain.CUES["boundary"].quote()
    rng = np.random.default_rng(0)
    conf = rng.random((16, 16))
    bnd = rng.random((16, 16)) < 0.3
    assessment = {"arrays": {"confidence": conf, "boundary": bnd.astype(float), "valid": np.ones((16, 16), bool)}, "review_sets": {}}
    out = explain.explain_review_set(assessment)
    assert "boundary_prevalence_note" in out and f"{100 * bnd.mean():.0f}%" in out["boundary_prevalence_note"]
