"""`oe-inferencex demo`: the first run a stranger makes. It must work with numpy alone, write what it says it writes,
say what the tool is worth and what it does not do, and keep telling the truth about its own sample.

Record copy. Only the tests of this file that use the record (exp/ code or exp/out) are kept here;
the others stay with the package in 2imi9/olmoearth_inferenceX, where the file was split at
0d1161a. These run against the released package.
"""
import json

import numpy as np
import pytest

from oe_inferencex.cli import main
from oe_inferencex.demo import SAMPLE


def test_the_real_sample_is_the_one_the_selection_record_chose_and_the_run_reports_its_pinned_numbers(tmp_path, capsys):
    """The tile was chosen by a rule so that the demo cannot flatter; the record of that choice and the bundled file must
    name the same tile, and what the run prints must be what exp/out pins."""
    meta = json.loads(str(np.load(SAMPLE)["meta"]))
    sel = json.load(open("exp/out/demo_sample_selection.json"))
    assert meta["tile"] == sel["chosen"]["tile"] and meta["n_candidates"] == sel["n_candidates"] == 18
    caps = sorted(c["capture_05"] for c in sel["candidates"])
    assert sel["chosen"]["capture_05"] == caps[(len(caps) - 1) // 2], "the lower median, not a better tile"
    main(["demo", "--out", str(tmp_path / "d")])
    said = capsys.readouterr().out
    got = json.load(open(tmp_path / "d" / "audit" / "assessment.json"))["against_reference"]
    pinned = json.load(open("exp/out/demo_sample_audit.json"))
    assert got["error_rate"] == pytest.approx(pinned["error_rate"])
    assert got["error_capture_at_budget"]["0.05"]["precision_in_set"] == pytest.approx(pinned["review_sets"]["0.05"]["precision_in_set"])
    assert "Of the 5% the tool flags, 67% are really wrong" in said and "picks at random (19%)" in said
    assert "the most any review could" in said and "no review of 5% can cover a map that is 19%" in said, \
        "a reader who sees red outside the flagged windows must be told what any 5% could hold"
