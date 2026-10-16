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


# ----------------------------------------------------------------------------- recorded numbers (16 October 2026)
CROSSCHECK = os.path.join(OUT, "exp111_crosscheck.json")
CROSSCHECK_SCRIPT = os.path.join(ROOT, "exp", "exp111_crosscheck.py")
RECORDS = os.path.join(E111_OUT, "records")
GATE_3DD770D = os.path.join(E111_OUT, "gate_3dd770d")


def _close(a, b, tol, path="."):
    """a and b equal in shape, strings and booleans equal, numbers within tol."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (path, sorted(set(a) ^ set(b)))
        for k in a:
            _close(a[k], b[k], tol, f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, tol, f"{path}[{i}]")
    elif isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b, (path, a, b)
    else:
        assert b == pytest.approx(a, abs=tol), (path, a, b)


@recorded
def test_exp111_recorded_by_a_second_route(tmp_path):
    """The grade (exp111_summary.json) against exp/exp111_crosscheck.py, written from the independent audit with numpy
    and scipy (its own weights, weighted AUROC, bootstrap, Clopper-Pearson k* and exact multinomial P(note)), importing
    neither the grader, the run's modules nor the package: the crosscheck reruns here to its committed output, which
    agrees with the summary in every count exactly, every point value and interval to 1e-12 (the region's error-rate
    estimate, the package's own route, to 1e-9) and every verdict; k* agrees with the package's interval. R1 is not
    recomputed: the committed units do not hold cgls_raw."""
    import ast
    import importlib.util

    from oe_inferencex.estimate import hypergeom_interval, interval_verdict

    src = open(CROSSCHECK_SCRIPT, encoding="utf-8").read()
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"oe_inferencex", "exp111_grade", "exp111_blind_transfer", "exp111_jobs"}, imported
    C, S = _json(CROSSCHECK), _json(SUMMARY)
    assert C["script_sha256"] == hashlib.sha256(src.encode("utf-8")).hexdigest(), "the committed crosscheck ran"
    spec = importlib.util.spec_from_file_location("exp111_crosscheck", CROSSCHECK_SCRIPT)
    xc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(xc)
    _close(xc.main(["--out", str(tmp_path / "again.json")]), C, 1e-12)

    # the inputs: the committed units, which are the run's without cgls_raw, and the run's units record
    for fn, f in C["inputs"]["units"].items():
        assert f["is_the_committed_file"] and f["run_sha256"] == S["units_sha256"][fn]
    assert C["inputs"]["units_record_sha256"] == S["units_record_sha256"]
    assert C["inputs"]["draw_sha256"] == S["draw_sha256"]
    ch = C["checks"]
    assert ch["rows"] == 900 and ch["no_cgls_raw_column"] and ch["shared_columns_equal_across_models"]
    assert ch["label_group_is_the_stratum"] and ch["drawn_by_cell_is_the_records"] and ch["in_gate17_cell"] == 0
    assert ch["weight_column_is_the_region_weight_over_its_sum_max_abs"] < 1e-15 and ch["weight_column_zero_without_input"]
    assert ch["crosswalk_sha256"] == S["crosswalk_sha256"]
    assert ch["with_input_by_cell"] == {k: v["with_input"] for k, v in S["report_only"]["R5_inventory"]["by_label_cell"].items()}
    for m in ("v1", "v12"):
        assert ch[f"{m}_wrong_strict_is_the_crosswalk"] and ch[f"{m}_wrong_lenient_is_the_crosswalk"]
        assert ch[f"{m}_suspicion_is_minus_log_p1_max_abs"] < 1e-12 and ch[f"{m}_ties_in_suspicion"] == 0

    for m in ("v1", "v12"):
        x, s = C["models"][m], S["models"][m]
        pt = s["point"]
        for k in ("n", "errors_strict", "errors_lenient", "correct_strict", "correct_lenient", "strict_only",
                  "confident", "confident_strict_errors", "confident_strict_errors_pair", "zone_plots"):
            assert x["counts"][k] == pt[k], (m, k)
        assert x["floors"] == {**s["floors"], "P2_counts_if_strict_only_were_counted_among_the_confident":
                               x["floors"]["P2_counts_if_strict_only_were_counted_among_the_confident"]}
        P1 = x["P1"]
        for k in ("auc_strict", "auc_lenient", "diff_lenient_minus_strict", "auc_strict_unweighted", "auc_lenient_unweighted"):
            assert P1[k] == pytest.approx(pt[k], abs=1e-12), (m, k)
        assert P1["auc_strict_by_every_pair"] == pytest.approx(P1["auc_strict"], abs=1e-12)
        assert P1["auc_lenient_by_every_pair"] == pytest.approx(P1["auc_lenient"], abs=1e-12)
        for rule in ("strict", "lenient"):
            for wn in ("weighted", "unweighted"):
                c, g = P1["capture_10"][f"{rule}_{wn}"], s["report_only"]["ranking"][rule][wn]
                assert (c["capture"], c["gap_closed"], c["error_rate"]) == pytest.approx(
                    (g["capture"]["0.1"], g["gap_closed"]["0.1"], g["error_rate"]), abs=1e-12), (m, rule, wn)
        st, rw = x["P2"]["standardised"], x["P2"]["region_weighted"]
        for k, sk in (("pair_share", "pair_share_std"), ("reverse_pair_share", "reverse_pair_share_std"),
                      ("auc_lenient_vs_correct", "auc_lenient_vs_correct_std"),
                      ("auc_strict_only_vs_correct", "auc_strict_only_vs_correct_std"), ("gap", "gap_std"),
                      ("strict_only_share_of_confident_strict_errors", "strict_only_share_of_confident_strict_errors_std")):
            assert st[k] == pytest.approx(pt[sk], abs=1e-12), (m, k)
        for k in ("pair_share", "auc_lenient_vs_correct", "auc_strict_only_vs_correct", "gap"):
            assert rw[k] == pytest.approx(pt["region_weighted"][k], abs=1e-12), (m, k)
        assert (x["P3"]["zone_eS"], x["P3"]["zone_eL"]) == pytest.approx((s["P3"]["zone_eS"], s["P3"]["zone_eL"]), abs=1e-12)
        for n in ("15", "20", "30"):
            mine, theirs = x["P3"]["labels"][n], s["report_only"]["P3"][f"labels_{n}"]
            assert mine["k_star"] == theirs["k_star"]
            for k in ("p_note", "p_strict_above", "p_lenient_above"):
                assert mine[k] == pytest.approx(theirs[k], abs=1e-12), (m, n, k)
            k = mine["k_star"]                     # the package's interval at N = 10^6 calls k* above alpha, k* - 1 not
            assert interval_verdict(*hypergeom_interval(k, int(n), 10 ** 6, 0.95), 0.05) == "above"
            assert interval_verdict(*hypergeom_interval(k - 1, int(n), 10 ** 6, 0.95), 0.05) != "above"
        for k, v in s["intervals90"].items():
            if k in x["intervals90"]:
                assert x["intervals90"][k] == pytest.approx(v, abs=1e-12), (m, k)   # the grade's resamples, exactly
        assert s["intervals90"]["resamples"] == 2000 and not any(s["intervals90"]["resamples_undefined"].values())
        v = x["verdicts"]
        assert {k: v[k] for k in ("P1a", "P1b", "P1", "P2a", "P2b", "P2", "P3")} == {
            k: s[k]["verdict"] for k in ("P1a", "P1b", "P1", "P2a", "P2b", "P2", "P3")}
        assert v["lenient_call"] == s["lenient_call"]["call"] == "not determined"
        assert v["P2b_arms"] == {k: s["P2b"][k] for k in v["P2b_arms"]}
        lead = s["report_only"]["class_with_most_confident_strict_only_errors"]
        mine = x["class_with_most_confident_strict_only_errors"]["region_weighted"]
        assert mine["class"] == lead["class"] and mine["share"] == pytest.approx(lead["weighted_share"], abs=1e-12)
        for b in ("0", "1", "2"):
            r, g = x["R2_bands"][b], s["report_only"]["R2_distance_bands"][f"band_{b}"]
            assert r["plots"] == g["plots"]
            for rule in ("strict", "lenient"):
                assert (r[rule]["error_rate"], r[rule]["auroc"]) == pytest.approx(
                    (g[rule]["error_rate"], g[rule]["auroc"]), abs=1e-12), (m, b, rule)
        for rule in ("strict", "lenient"):
            g = s["report_only"]["R4_region_error_rate"][rule]
            for k in ("estimate", "envelope_missing_right", "envelope_missing_wrong"):
                assert C["R4"][m][rule][k] == pytest.approx(g[k], abs=1e-9), (m, rule, k)
        assert x["R5_condition_codes"] == S["report_only"]["R5_inventory"]["condition_codes"][m]
    assert S["graded_models"] == ["v12"]
    assert C["verdicts_graded"] == {k: v["v12"] for k, v in S["verdicts_graded"].items()}
    assert C["models_same_class_share"] == pytest.approx(S["report_only"]["R5_inventory"]["models_same_class_share"], abs=1e-15)
    assert S["report_only"]["without_plots_in_a_gate17_cell"] == {"plots_in_a_gate17_cell_with_input": 0}


@recorded
def test_exp111_v1_does_not_reproduce_exp98_under_rslearn_0117():
    """Three jobs ran v1 with the formula's temporal table at gate A's 259 points, each against exp98's committed known
    answer, and agree: the failed gate (1365118 at 3dd770d), the passing gate (1369594 at 8dc3a01) and the v1 route
    diagnostic's formula variant (1369140); the diagnostic's three other routes do no better. v1.2 on the same route
    reproduces exp103. The gate that passed counted v1.2's checks only (the amendment of 16 October 2026), and the
    blind prediction ran on the gate's GPU model at the frozen commit."""
    failed, passed = _json(os.path.join(GATE_3DD770D, "e111_gate.json")), _json(os.path.join(RECORDS, "e111_gate.json"))
    diag = _json(os.path.join(RECORDS, "e111_v1diag.json"))
    assert (failed["device"]["job"], failed["predict_commit"][:7], failed["pass"]) == ("1365118", "3dd770d", False)
    assert (passed["device"]["job"], passed["predict_commit"][:7], passed["commit"][:7], passed["pass"]) == (
        "1369594", "8dc3a01", "8dc3a01", True)
    assert failed["A"]["checks"] == passed["A"]["checks"] == {"A1": False, "A2": False, "A3": True, "A4": True, "A_floor": True}
    assert "B" not in failed and passed["A"]["graded_checks"] == ["A3", "A4", "A_floor"]
    assert passed["B"]["graded_checks"] == ["B2", "B_floor"] and passed["B"]["B2"] and not passed["B"]["B1"]
    runs = [failed["A"]["models"]["v1"], passed["A"]["models"]["v1"], diag["variants"]["formula"]]
    assert {(r["covered_both"], r["same_class"], r["dp1_above_tol"], round(r["spearman_suspicion"], 3)) for r in runs} == {
        (259, 245, 39, 0.916)}
    assert round(1 - 245 / 259, 3) == 0.054
    V = diag["variants"]
    assert [(k, V[k]["same_class"], V[k]["dp1_above_tol"], V[k]["would_pass_A1_A2"]) for k in V] == [
        ("formula", 245, 39, False), ("table", 245, 40, False), ("table_fp32buf", 245, 40, False),
        ("table_fp32", 244, 39, False)] and diag["variants_failed"] == []
    for g in (failed, passed):
        v12 = g["A"]["models"]["v12"]
        assert (v12["same_class"], v12["dp1_above_tol"], round(v12["dp1_max"], 3)) == (259, 0, 0.044)
    b1, b2 = passed["B"]["models"]["v1"], passed["B"]["models"]["v12"]
    assert (b1["covered_both"], b1["same_class"], b1["dp1_above_tol"], round(b1["spearman_suspicion"], 3)) == (263, 237, 89, 0.84)
    assert (b1["confident_zone"]["known"], b1["confident_zone"]["kept"]) == (91, 76)
    assert (b2["same_class"], b2["dp1_above_tol"], round(b2["spearman_suspicion"], 4)) == (262, 1, 0.9995)
    assert (b2["confident_zone"]["known"], b2["confident_zone"]["kept"]) == (90, 89)
    pred = _json(os.path.join(RECORDS, "e111_device_predict.json"))
    assert pred["name"] == passed["device"]["name"] == failed["device"]["name"] == "NVIDIA B200"
    assert (pred["job"], pred["commit"][:7]) == ("1369838", "6d80ab7")
    for f in ("e111_versions_gate.json", "e111_versions_predict.json", "e111_versions_v1diag.json"):
        assert _json(os.path.join(RECORDS, f))["pins"] == {"rslearn": "0.1.17", "olmoearth-pretrain-minimal": "0.1.2",
                                                            "torch": "2.14.1", "lightning": "2.6.6", "jsonargparse": "4.52.0"}
    C = _json(CROSSCHECK)["gate"]
    assert C["v1_formula_route_the_same_in_three_jobs"] and round(C["v1_share_changing_class_at_gate_A"], 3) == 0.054
