"""exp111 (docs/plan/blind_transfer.md), written at the phase-1 commit, before any cluster job: the page states the
numbers the code holds (bars, floors, gate, sizes, failure allowance, confident cuts); the committed sizing record's
pool is the one the select step refuses without; the selection, window, gate and units smoke and the grader's smoke
(synthetic units with known answers, the known answer at exp99's 271) pass. No network. Needs rasterio, shapely and
pyproj (the geo extra); skipped without them.

Added with the run's outputs (16 October 2026): the committed records, logs and units hold no identifier or position,
and the units files are the run's without the cgls_raw column (the owner's decision), the run's sha256 kept beside
the committed files' own."""
import hashlib
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


# ----------------------------------------------------------------------------- the run's outputs (16 October 2026)
OUT = os.path.join(ROOT, "exp", "out")
E111_OUT = os.path.join(OUT, "exp111")
SUMMARY = os.path.join(OUT, "exp111_summary.json")
UNITS_RECORD = os.path.join(OUT, "exp111_units_record.json")
UNITS_COMMITTED = os.path.join(OUT, "exp111_units_committed.json")
UNITS = {m: os.path.join(OUT, f"exp111_units_{m}.npz") for m in ("v1", "v12")}
RUN_COLUMNS = ["country", "country_names", "label", "label_names", "covered", "inside_awf", "weight", "stratum",
               "stratum_names", "band", "cgls_raw", "in_gate17_cell", "pred", "suspicion", "p1", "condition",
               "majority_3x3", "wrong_strict", "wrong_lenient"]
recorded = pytest.mark.skipif(not os.path.exists(SUMMARY), reason="exp111's outputs are not committed")


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@recorded
def test_exp111_outputs_hold_no_identifier_or_position():
    """Every committed record passes the pack jobs' key check (exp111_jobs.pack_check, the set the cluster ran before
    anything left it), no committed text names a TimeSync plot id, and the units files hold no identifier, position or
    CGLS class: the run's cgls_raw column is dropped (7 rows were unique among exp94's Kenya and Tanzania 2017 rows
    through country, label and cgls_raw, so each could be linked to a public plot id)."""
    import numpy as np
    bad, n = j111.pack_check([E111_OUT])
    assert not bad and n == 83, (bad, n)
    texts = []
    for p in (SUMMARY, UNITS_RECORD, UNITS_COMMITTED):
        j111.check_record_keys(_json(p), os.path.basename(p))
        texts.append(open(p, encoding="utf-8").read())
    for dp, _, fs in os.walk(E111_OUT):
        texts += [open(os.path.join(dp, fn), encoding="utf-8").read() for fn in fs]
    assert len(texts) == 3 + 83 + 9                     # three top-level records, 83 records, 9 Slurm logs
    for t in texts:
        assert not re.search(r"\b(kenya|tanzania|ethiopia|uganda|rwanda|burundi|malawi):\d", t, re.I), "a plot id"
        assert "POINT" not in t and '"coordinates"' not in t
    for m, p in UNITS.items():
        z = np.load(p, allow_pickle=False)
        assert list(z.files) == [k for k in RUN_COLUMNS if k != "cgls_raw"]
        assert not [k for k in z.files if k.lower() in j111.RECORD_FORBIDDEN_KEYS or "pid" in k.lower() or k == "id"]
        assert all(z[k].shape == (900,) for k in z.files if not k.endswith("_names"))


@recorded
def test_exp111_committed_units_are_the_runs_without_cgls_raw():
    """The committed units files are the run's minus one column: the grade read the run's files (their sha256 in the
    units record and the summary), the committed record is the run's (its sha256 in the summary), and
    exp111_units_committed.json pairs each run sha256 with the committed file's. The rows' counts are the run's."""
    import numpy as np
    S, R, C = _json(SUMMARY), _json(UNITS_RECORD), _json(UNITS_COMMITTED)
    page_sha = b111.page_draw_sha()
    assert page_sha == S["draw_sha256"] == R["draw_sha256"] == "3e5506e51f4d66620665fab17939a298fdc85bc5b1ed7b1c499e1a2095c0731c"
    assert S["units_sha256"] == R["sha256"] and S["units_record_sha256"] == _sha256(UNITS_RECORD)
    assert C["dropped_column"] == "cgls_raw" and sorted(C["files"]) == sorted(os.path.basename(p) for p in UNITS.values())
    for m, p in UNITS.items():
        f = C["files"][os.path.basename(p)]
        assert f["run_sha256"] == R["sha256"][os.path.basename(p)] != f["committed_sha256"] == _sha256(p)
        assert f["run_columns"] == RUN_COLUMNS and f["committed_columns"] == [k for k in RUN_COLUMNS if k != "cgls_raw"]
        z = np.load(p, allow_pickle=False)
        cov = z["covered"]
        assert len(cov) == R["n_plots"] == 900
        assert (int(cov.sum()), int(np.nansum(z["wrong_strict"][cov])), int(np.nansum(z["wrong_lenient"][cov]))) == (
            R["models"][m]["covered"], R["models"][m]["errors_strict"], R["models"][m]["errors_lenient"])
        assert int(z["in_gate17_cell"].sum()) == R["in_gate17_cell"] == 0
        names = [f"{c}:{g}" for c in z["country_names"] for g in z["stratum_names"]]
        drawn = {k: int(((z["country"] == i // 3) & (z["stratum"] == i % 3)).sum()) for i, k in enumerate(names)}
        assert drawn == R["drawn_by_cell"]
