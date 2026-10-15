"""exp111 (docs/plan/blind_transfer.md), written at the phase-1 commit, before any cluster job: the page states the
numbers the code holds (bars, floors, gate, sizes, failure allowance, confident cuts); the committed sizing record's
pool is the one the select step refuses without; the selection, window, gate and units smoke and the grader's smoke
(synthetic units with known answers, the known answer at exp99's 271) pass. No network. Needs rasterio, shapely and
pyproj (the geo extra); skipped without them."""
import json
import os
import re
import sys

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp111_blind_transfer as b111   # noqa: E402
import exp111_grade as g111             # noqa: E402
import exp111_jobs as j111              # noqa: E402

E111_SH = os.path.join(ROOT, "exp", "jobs", "e111.sh")
POWER = os.path.join(ROOT, "exp", "out", "exp111_power.json")


def _page():
    return re.sub(r"\s+", " ", open(b111.PLAN, encoding="utf-8").read())


def test_the_page_states_the_codes_numbers():
    t, sh = _page(), open(E111_SH, encoding="utf-8").read()
    f, bars, gate = g111.FLOORS, g111.BARS, b111.GATE_BARS
    for phrase in (
            "binding at the phase-1 commit",
            f"at least {f['with_input']} of the 900",
            f"at least {f['errors_per_rule']} errors and {f['correct_per_rule']} correct",
            f"at least {f['confident_strict_errors']} confident STRICT errors and {f['strict_only_errors']} STRICT-only",
            f"at least {f['zone_plots']} plots in shrubland_savanna",
            f"**{bars['P1a_strict_below']:.2f}** and its 90% interval lies above **{bars['P1a_lower_above']:.2f}**",
            f"at most **{bars['P2b_strict_only_max']:.2f}**", f"at least **{bars['P2b_lenient_min']:.2f}**",
            f"at least **{bars['P2a_pair_share_min']:.2f}**", f"P(note) at least **{bars['P3_p_note_min']}**",
            f"{int(gate['same_class_min_share'] * 100)}% or more",
            f"at least {gate['floor_with_input_both']['gate23']} of the 259",
            f"at least {gate['floor_with_input_both']['gate17']} of the 271",
            f"at least {gate['spearman_suspicion_min']}",
            "{O}/{W}/{X}".format(**b111.SIZES),
            f"MAX_FAILED_SHARE` = {re.search(r'^MAX_FAILED_SHARE=([0-9.]+)$', sh, re.M).group(1)}",
            f"at most {j111.MAX_FAILED_SHARE_CAP}"):
        assert phrase in t, phrase
    assert all(str(c) in t for c in b111.CONFIDENT_CUT.values())
    assert b111.SIZES == {"O": 300, "W": 300, "X": 300} and b111.SEED == 111
    assert f["with_input"] == int(0.8 * sum(b111.SIZES.values()))
    assert re.search(r"^MAX_FAILED_SHARE=0$", sh, re.M)


def test_the_sizing_record_holds_the_select_steps_pool():
    p = json.load(open(POWER, encoding="utf-8"))
    assert {k.replace("|", ":"): v for k, v in p["pool_by_country_group"].items()} == b111.POOL_CELLS
    assert sum(p["pool_by_group"].values()) == b111.POOL_N


def test_the_selection_window_gate_and_units_smoke(tmp_path):
    assert b111.smoke(out_dir=str(tmp_path), log=lambda *a: None)["ok"]


def test_the_graders_smoke(tmp_path):
    assert g111.smoke(out_dir=str(tmp_path), log=lambda *a: None)["ok"]
