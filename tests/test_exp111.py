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
            f"at most {j111.MAX_FAILED_SHARE_CAP}",
            f"within {j111.LEGACY_MAX} of the formula", "on-the-fly encoding within 1e-5"):
        assert phrase in t, phrase
    assert all(str(c) in t for c in b111.CONFIDENT_CUT.values())
    assert b111.SIZES == {"O": 300, "W": 300, "X": 300} and b111.SEED == 111
    assert f["with_input"] == int(0.8 * sum(b111.SIZES.values()))
    assert re.search(r"^MAX_FAILED_SHARE=0$", sh, re.M)
    assert (j111.LEGACY_MAX, j111.LEGACY_EQUAL) == (0.01, 1e-5)
    # the amendment of 16 October 2026: v1.2 is the graded model, v1 report-only
    assert b111.GRADED_MODELS == ("v12",) and b111.GRADED_CHECKS == {"A": ("A3", "A4", "A_floor"), "B": ("B2", "B_floor")}
    for phrase in ("v1.2 is the graded model.", "A3, A4, gate A's floor, B2 (all four checks) and gate B's floor",
                   "graded for v1.2; v1's verdict is reported beside"):
        assert phrase in t, phrase


def test_the_sizing_record_holds_the_select_steps_pool():
    p = json.load(open(POWER, encoding="utf-8"))
    assert {k.replace("|", ":"): v for k, v in p["pool_by_country_group"].items()} == b111.POOL_CELLS
    assert sum(p["pool_by_group"].values()) == b111.POOL_N


def test_the_selection_window_gate_and_units_smoke(tmp_path):
    # the smoke's select, windows and redo paths are a draft page's; the page is frozen from 16 October 2026, and the
    # module is guarded since the gate (8dc3a01), so the test, not the module, sets the draft status
    with b111._pretend_status("draft"):
        assert b111.smoke(out_dir=str(tmp_path), log=lambda *a: None)["ok"]


def test_the_graders_smoke(tmp_path):
    assert g111.smoke(out_dir=str(tmp_path), log=lambda *a: None)["ok"]


def test_the_v1_diagnostic_compares_each_variant_with_exp98(tmp_path, monkeypatch):
    """The report-only v1 route diagnostic (amendment of 16 October 2026): each variant's read is compared with exp98's
    known answer as A1 and A2 compare it; the temporary read names leave nothing behind."""
    import numpy as np
    u = np.load(b111.KNOWN[("gate23", "v1")])
    sel = (u["split"] == "val") & u["covered"]
    with open(tmp_path / "units_gate23.json", "w") as f:
        json.dump({"units": [{"key": str(n)} for n in u["names"][sel]]}, f)

    def fake_read(priv, root, model, redo=False, log=print):
        assert root == "gate23" and b111.MODELS[model] == "output_d_" + model.split("_", 1)[1]
        return {"ok": True}

    def fake_load(priv, root, model):
        p1, pred = u["p1"][sel].astype(float), u["pred_deployed"][sel].copy()
        if model == "v1diag_table":
            pred[:10] = (pred[:10] + 1) % 9
        return {"covered": np.ones(len(pred), bool), "p1": p1, "pred": pred, "suspicion": -np.log(p1)}

    monkeypatch.setattr(b111, "run_read", fake_read)
    monkeypatch.setattr(b111, "_load_read", fake_load)
    out = b111.run_v1diag(str(tmp_path), log=lambda *a: None)
    assert list(out["variants"]) == list(b111.V1DIAG_VARIANTS)
    f, t = out["variants"]["formula"], out["variants"]["table"]
    assert f["same_class"] == f["covered_both"] == int(sel.sum()) and f["would_pass_A1_A2"]
    assert t["same_class"] == int(sel.sum()) - 10 and not t["would_pass_A1_A2"]
    assert not any(k.startswith("v1diag_") for k in b111.MODELS)
