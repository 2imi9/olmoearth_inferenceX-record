"""exp98's measures and grades before the preregistration is frozen and before any real run
(docs/plan/awf_deployment.md). The synthetic smoke writes a map read in two overlapping parts (two from-olmoearth
directories, two deployment dataset roots, as the real run is read), its condition layer, AWF-style label windows
(one in another CRS, two outside the map, two on uncovered pixels, two dropped by the label rule, eleven in both
parts), exp89's replica units, and runs every part on them; the values below are computed by hand from that design
(exp98_awf_deployment.py, the comment above SMOKE_GRID). The grades are also checked on planted numbers at their
thresholds and floors, the readers on small cases, the guard that refuses the run while the page is a draft, and that
nothing written carries a position. The last three tests recompute the recorded run's graded numbers from its committed
per-unit files by a second route, with plain numpy and no exp98 helper (the claim ledger's crosschecks for exp98).
Needs rasterio (the geo extra); skipped without it."""
import json
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("shapely")      # Part I (exp/timesync_awf_crosswalk.py)
pytest.importorskip("pyproj")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp98_awf_deployment as e98   # noqa: E402

X = e98.SMOKE_EXPECTED


@pytest.fixture(scope="module")
def out_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("exp98"))


@pytest.fixture(scope="module")
def smoke(out_dir):
    return e98.smoke(out_dir=out_dir, log=lambda *a: None)


# ----------------------------------------------------------------------------- structure
def test_the_summary_holds_every_part(smoke, out_dir):
    for k in ("experiment", "prereg_page", "inputs", "labels", "mapping", "graded_validation", "train_in_sample",
              "where_the_points_sit", "plan_reported", "inputs_reported", "prereg", "inventory"):
        assert k in smoke, k
    for p in e98.PREDICTIONS:
        row = smoke["prereg"][p]
        assert {"rule", "threshold", "value", "graded"} <= set(row), p
        assert row["threshold"] == e98.THRESHOLDS[p]
    g = smoke["graded_validation"]
    for k in ("accuracy", "ai2", "ranking", "replica", "by_condition", "assess_package_path", "boundary_reported"):
        assert k in g, k
    with open(os.path.join(out_dir, "exp98_summary_smoke.json")) as f:
        assert json.load(f)["prereg"]["P3"]["value"] == pytest.approx(X["auroc"])
    z = np.load(os.path.join(out_dir, "exp98_units_smoke.npz"))
    assert set(z.files) == {"names", "split", "label", "covered", "pred_deployed", "suspicion", "p1", "condition",
                            "burned", "boundary", "input_identical"}
    assert int(z["covered"].sum()) == 50


def test_nothing_written_carries_a_position(smoke, out_dir):
    e98.check_no_coordinates(smoke)
    with open(os.path.join(out_dir, "exp98_summary_smoke.json")) as f:
        written = json.load(f)

    def numbers(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from numbers(v)
        elif isinstance(o, list):
            for v in o:
                yield from numbers(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield o
    # the synthetic grid's origin in metres and in projection pixels, and its far corner; no number written is one
    origin = {500000.0, 9000000.0, 50000.0, -900000.0, 501600.0, 8998800.0, 50160.0, -899880.0}
    assert not origin & set(float(v) for v in numbers(written))
    with pytest.raises(ValueError, match="position"):
        e98.check_no_coordinates({"grids": [{"transform": [10, 0, 1, 0, -10, 2]}]})
    with pytest.raises(ValueError, match="position"):
        e98.check_no_coordinates({"a": {"lonlat": [1, 2]}})


# ----------------------------------------------------------------------------- hand-computed values
def test_labels_and_mapping(smoke):
    lab, mp = smoke["labels"], smoke["mapping"]
    assert lab["by_split"] == {"val": 44, "train": 10}
    assert sum(lab["dropped"].values()) == X["dropped"]
    assert lab["dropped"] == {"2 labelled pixels": 1, "0 labelled pixels": 1}
    assert lab["time_range_matches_request"]["n_matching"] == 54
    assert mp["by_split"]["val"] == X["mapping_val"]
    assert mp["by_split"]["train"] == {"n": 10, "outside_every_grid": 0, "in_grid_uncovered": 0, "covered": 10}
    assert mp["reprojected"] == X["reprojected"] and mp["same_crs"] == 51
    assert mp["largest_offset_from_pixel_middle_px"] < 1e-6, "every label pixel's centre is a score pixel's centre"
    assert smoke["inventory"]["mapping"] == mp, "the inventory maps the points as the run does"


def test_two_parts_read_as_one_map(smoke):
    o = smoke["mapping"]["overlapping_grids"]
    assert {k: o[k] for k in X["overlap"] if k in o} == {k: v for k, v in X["overlap"].items() if k in o}
    assert o["covered_only_in_a_later_grid"] == 5, "a point on a pixel NaN in part 0 is read from part 1"
    w = smoke["where_the_points_sit"]
    assert w["n_map_pixels"] == 16000, "a pixel covered in both parts counts once"
    assert w["n_pixels_counted_once_of_overlaps"] == X["overlap"]["pixels_counted_once"]
    assert w["n_grid_overlaps_not_deduplicated"] == 0
    assert [g["label"] for g in smoke["inputs"]["grids"]] == ["0:32737", "1:32737"]
    assert sum(g["n_val_covered"] for g in smoke["inputs"]["grids"]) == X["n_val"]


def test_condition_names_that_conflict_across_parts_are_refused(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    dirs = []
    for k, name in enumerate(("sentinel2:all:clear", "sentinel2:none")):
        d = tmp_path / f"p{k}"
        d.mkdir()
        with rasterio.open(d / "scores_32737.tif", "w", driver="GTiff", height=2, width=2, count=10, dtype="float32",
                           crs="EPSG:32737", transform=from_origin(0, 0, 10, 10)) as dst:
            dst.write(np.full((10, 2, 2), 0.1, np.float32))
        (d / "olmoearth_conditions.json").write_text(json.dumps({"codes": {"0": name}}))
        dirs.append(str(d))
    with pytest.raises(ValueError, match="not read with the same inputs"):
        e98.find_outputs(dirs)
    entries, names, _ = e98.find_outputs(dirs[:1])
    assert names == {0: "sentinel2:all:clear"} and entries[0]["label"] == "32737"


def test_part_a_against_the_replica(smoke):
    g = smoke["graded_validation"]
    assert g["n"] == X["n_val"] and g["n_clusters"] == 6
    assert g["accuracy"]["accuracy_window"] == pytest.approx(X["accuracy"])
    assert g["accuracy"]["n_errors"] == X["n_errors"]
    assert g["readings"]["n_pred_untrained"] == X["n_pred_untrained"]
    r = g["replica"]
    assert r["n_matched"] == 40 and r["label_mismatches"] == 0 and r["cluster_mismatches"] == 0
    assert r["accuracy_replica"] == pytest.approx(X["replica_accuracy"])
    assert r["difference"] == pytest.approx(X["accuracy"] - X["replica_accuracy"])
    assert r["agreement"] == pytest.approx(X["agreement"]) and r["n_same_class"] == 36
    d = r["discordant"]
    assert (d["deployed_right_replica_wrong"], d["replica_right_deployed_wrong"]) == (1, 3)
    assert d["sign_test_two_sided_p"] == pytest.approx(2 * (1 + 4) / 16)        # 1 against 3 of 4, two-sided
    assert g["ai2"]["matched"] is False and g["ai2"]["difference"] == pytest.approx(0.80 - 0.895)
    same = g["replica_same_crs_reported"]                      # the reprojected point (rank 39, an error) left out
    assert same["n_matched"] == 39 and same["n_left_out"] == 1
    assert same["accuracy_deployed"] == pytest.approx(32 / 39) and same["agreement"] == pytest.approx(35 / 39)
    iv = r["p1_equivalence_interval"]
    assert iv["level"] == 0.90 and iv["n_clusters"] == 6
    assert iv["lo"] <= r["difference"] + 1e-12 <= iv["hi"] + 1e-12
    assert iv["max_abs_end"] == pytest.approx(max(abs(iv["lo"]), abs(iv["hi"])))
    assert r["within_2_points_reported"] is False


def test_part_b_ranking(smoke):
    rk = smoke["graded_validation"]["ranking"]
    assert rk["auroc"] == pytest.approx(X["auroc"])
    assert rk["capture"] == pytest.approx(X["capture"])
    assert rk["capture_random"] == pytest.approx({"0.05": 2 / 40, "0.1": 4 / 40, "0.2": 8 / 40})
    assert rk["ceiling"] == pytest.approx({"0.05": 2 / 8, "0.1": 4 / 8, "0.2": 1.0})


def test_the_package_path_differs_only_by_the_untrained_channel(smoke):
    a = smoke["graded_validation"]["assess_package_path"]
    assert a["n_windows_scored"] == 40
    assert a["error_rate"] == pytest.approx(X["assess_errors"] / 40)
    assert a["same_order_as_ranking"] is True


def test_part_c_conditions(smoke):
    bc = smoke["graded_validation"]["by_condition"]
    got = {v["code"]: (v["n_points"], v["n_errors"]) for v in bc.values()}
    assert got == X["by_condition"]
    assert bc["sentinel2:all:clear"]["share_of_map"] == pytest.approx(50 * 160 / 16000)
    assert bc["unrecorded"]["share_of_map"] == pytest.approx(2 / 16000)


def test_train_points_are_reported_apart(smoke):
    t = smoke["train_in_sample"]
    assert t["n"] == 10 and t["accuracy"] == pytest.approx(X["train_accuracy"])
    assert smoke["prereg"]["floors"]["n_validation_points"] == 40, "training points never enter the graded set"


def test_part_e_where_the_points_sit(smoke):
    w = smoke["where_the_points_sit"]
    assert w["n_map_pixels"] == 16000
    assert w["points_mean_percentile"] == pytest.approx(X["mean_percentile"])
    assert w["map_class_share"]["shrubland_savanna"] + w["map_class_share"]["grassland_barren"] > 0.99
    assert w["reweighted_reported"]["estimate"] is None, "the tied background leaves strata without points"


def test_part_f_plan_is_reported_not_graded(smoke):
    pl = smoke["plan_reported"]
    assert pl["n_windows"] == 16000 and pl["width"] == 0.10 and pl["alpha"] == 0.10
    assert set(pl["error_rate"]) == {"points", "exp89_replica", "default_0.5"}
    assert pl["error_rate"]["points"]["error_rate"] == pytest.approx(0.2)
    assert pl["zone"]["1"]["refusal"], "the points' 20% error is above alpha, so the whole map cannot be planned"
    assert pl["certify_min_labels"] == 22
    assert "--design sequential" in pl["sequential"]["command"]
    assert all("holds" not in v for v in pl["zone"].values())


def test_part_g_inputs(smoke):
    g = smoke["inputs_reported"]
    assert g["n_compared"] == X["inputs"]["n_compared"]
    assert g["n_identical_all_groups"] == X["inputs"]["identical"]
    assert g["n_points_without_deployed_window"] == X["inputs"]["no_window"]
    assert g["n_points_without_label_imagery"] == X["inputs"]["no_label_imagery"]
    assert g["n_points_without_deployed_imagery_read"] == 0 and g["n_points_incomplete"] == 0
    assert g["n_points_in_two_deployed_windows"] == X["inputs"]["in_two_windows"]
    assert g["config"]["n_deployment_roots"] == 2 and g["config"]["deployment_roots_share_one_config"] is True
    assert g["n_identical_any_order"] == 37 and g["n_identical_reversed_order"] == 0
    assert g["share_identical_by_group"][5] == pytest.approx(37 / 39)
    assert [d["key"] for d in g["config"]["differences"]] == ["band_sets"]
    assert g["validation_points_by_input"]["not_identical"] == {"n_val": 2, "error_rate": 0.0,
                                                                "agreement_with_replica": 1.0}
    inv = smoke["inventory"]["inputs_reported"]
    assert "validation_points_by_input" not in inv, "the inventory reads no outcome"
    assert inv["n_identical_all_groups"] == X["inputs"]["identical"]


def test_part_h_burned_area(smoke, out_dir):
    h = smoke["burned_area_reported"]
    assert h["status"].startswith("report-only") and "holds" not in json.dumps(h)
    names = {c: e98.burn_name(c, e98.SMOKE_BURN_NAMES) for c in (-1, 0, 1)}
    assert names[0] == "mcd64a1:no-burn-2023:sep-not-read" and h["months_missing"] == ["2023-09"]
    pts = h["validation_points"]
    assert {c: (pts[names[c]]["n_points"], pts[names[c]]["n_errors"]) for c in names} == X["burned"]["points"]
    w = h["windows"]
    assert {c: w["per_code"][names[c]]["n_windows"] for c in names} == X["burned"]["windows"]
    assert w["n_windows"] == 1000
    assert w["n_windows_dropped_overlapping_an_earlier_grid"] == X["burned"]["dropped_overlap"]
    assert {c: h["pixels"][names[c]]["n"] for c in names} == X["burned"]["pixels"]
    # 17 windows hold a point less confident than the background's 0.6 (ranks 0-16: four burned, ten unrecorded,
    # three not burned); the other 83 of the least confident 100 come from the 960 windows tied at the background
    tied = {-1: 70, 0: 420, 1: 470}
    above = {-1: 10, 0: 3, 1: 4}
    assert w["n_tied_at_cutoff"] == 960 and w["n_tied_taken"] == 83
    for c in names:
        share = (above[c] + 83 * tied[c] / 960) / 100
        assert w["per_code"][names[c]]["share_of_least_confident"] == pytest.approx(share)
        assert w["per_code"][names[c]]["confidence_quartiles"]["0.5"] == pytest.approx(0.6)
    z = np.load(os.path.join(out_dir, "exp98_units_smoke.npz"))
    v = (z["split"] == "val") & z["covered"]
    assert sorted(np.unique(z["burned"][v]).tolist()) == [-1, 0, 1]


def _keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _keys(v)


def test_part_i_timesync_plots(smoke, tmp_path):
    """Part I on the smoke's TimeSync plots (SMOKE_TS_PLOTS): counts, disagreements by rule and country, the
    stratified rate from the countries' shares, and no verdict; the inventory reads no class."""
    t = smoke["timesync_reported"]
    XT = e98.SMOKE_TS_EXPECTED
    assert t["status"].startswith("report-only") and "holds" not in set(_keys(t))
    assert {c: t["plots"][c]["inside_geometry"] for c in XT["inside"]} == XT["inside"]
    assert {c: t["plots"][c]["covered"] for c in XT["covered"]} == XT["covered"]
    assert t["label_changes"]["plots_with_two_or_more_labels_2015_2017"] == XT["changed_2015_2017"]
    assert t["overlapping_grids"]["covered_only_in_a_later_grid"] == 1, "a plot on a pixel NaN in part 0"
    share = t["areas"]["share_by_country"]
    assert sum(share.values()) == pytest.approx(1.0) and t["areas"]["share_outside_both_countries"] < 1e-9
    for rule in ("strict", "lenient"):
        r = t["rules"][rule]
        assert {c: r["by_country"][c]["disagreements"] for c in XT[rule]} == XT[rule]
        assert r["stratified"]["weights"] == pytest.approx(share, rel=1e-4), "10 m pixels of each country's part"
        theta = sum(r["stratified"]["weights"][c] * XT[rule][c] / XT["covered"][c] for c in share)
        assert r["stratified"]["estimate"] == pytest.approx(theta)
        assert r["stratified"]["low"] <= theta <= r["stratified"]["high"]
        assert r["auroc_unweighted"] == pytest.approx(0.5), "every plot sits on a background pixel at 0.6"
        assert r["three_by_three_majority_disagreements"] == r["disagreements"]
    assert t["crosswalk"]["sha256"] == e98.tsx.crosswalk_record()["sha256"]
    inv = smoke["inventory"]["timesync_inventory"]
    assert inv["mode"].startswith("inventory") and "rules" not in inv and inv["plots"] == t["plots"]
    # no plot's position is written
    paths = e98.make_smoke_timesync(str(tmp_path))
    plots = e98.tsx.read_plots(paths["timesync"], 2017)
    coords = set(np.round(plots["lon"], 6)) | set(np.round(plots["lat"], 6))

    def numbers(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from numbers(v)
        elif isinstance(o, list):
            for v in o:
                yield from numbers(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield round(float(o), 6)
    assert not coords & set(numbers(smoke))


def test_part_i_never_stops_the_graded_run(tmp_path):
    paths = e98.make_smoke_inputs(str(tmp_path))
    with open(paths["timesync"]["timesync"], "w") as f:
        f.write("not,a,timesync,file\n1,2,3,4\n")
    rec, units = e98.analyse(paths["scores"], paths["labels"], replica_path=paths["replica"],
                             n_boot=e98.SMOKE_BOOT, zone_draws=e98.SMOKE_ZONE_DRAWS,
                             max_labels=e98.SMOKE_MAX_LABELS, log=lambda *a: None, geometry_sha256=None,
                             **paths["timesync"], **e98.SMOKE_FLOORS)
    assert "error" in rec["timesync_reported"]
    assert {p: rec["prereg"][p]["holds"] for p in e98.PREDICTIONS} == X["holds"]


def _analyse_smoke(tmp_path):
    paths = e98.make_smoke_inputs(str(tmp_path))
    rec, units = e98.analyse(paths["scores"], paths["labels"], replica_path=paths["replica"],
                             n_boot=e98.SMOKE_BOOT, zone_draws=e98.SMOKE_ZONE_DRAWS,
                             max_labels=e98.SMOKE_MAX_LABELS, log=lambda *a: None, **e98.SMOKE_FLOORS)
    return paths, rec, units


def _graded_as_designed(rec, units):
    assert {p: rec["prereg"][p]["holds"] for p in e98.PREDICTIONS} == X["holds"]
    assert all(rec["prereg"][p]["graded"] for p in e98.PREDICTIONS)
    assert (units["burned"] == -2).all(), "Part H left out: no point carries a burn code"
    e98.check_no_coordinates(rec)


@pytest.mark.parametrize("fault", ["unreadable_file", "names_json", "window_refusal", "point_read"])
def test_a_failing_burned_layer_never_stops_the_graded_run(tmp_path, monkeypatch, fault):
    """Part H is report-only (docs/plan/awf_deployment.md, step 10): an unreadable burned_<EPSG>.tif, an unreadable
    names file, a refusal from the window pooling or a failed point read leaves Part H out and P1-P5 are graded."""
    if fault == "unreadable_file":
        orig = e98.make_smoke_inputs

        def broken(root):
            paths = orig(root)
            f = os.path.join(paths["scores"][1], f"burned_{e98.SMOKE_GRID['epsg']}.tif")
            with open(f, "wb") as fh:
                fh.write(b"not a tiff")
            return paths
        monkeypatch.setattr(e98, "make_smoke_inputs", broken)
    elif fault == "names_json":
        orig = e98.make_smoke_inputs

        def broken(root):
            paths = orig(root)
            with open(os.path.join(paths["scores"][0], e98.eb.NAMES_JSON), "w") as fh:
                fh.write("{not json")
            return paths
        monkeypatch.setattr(e98, "make_smoke_inputs", broken)
    elif fault == "window_refusal":
        def refuse(*a, **k):
            raise ValueError("planted refusal")
        monkeypatch.setattr(e98, "burn_windows", refuse)      # as burn_windows re-raises pool_condition's
    else:
        def fail(*a, **k):
            raise OSError("planted read failure")
        monkeypatch.setattr(e98, "codes_at", fail)
    _, rec, units = _analyse_smoke(tmp_path)
    _graded_as_designed(rec, units)
    h = rec["burned_area_reported"]
    if fault == "point_read":
        assert "planted read failure" in h["error"]
    else:
        assert "windows" not in h and "validation_points" not in h, h
        assert {"unreadable_file": "could not be opened", "names_json": "names the codes differently",
                "window_refusal": "planted refusal"}[fault] in h["note"]


# ----------------------------------------------------------------------------- the grades
def test_the_smoke_grades_as_designed(smoke):
    pr = smoke["prereg"]
    assert {p: pr[p]["holds"] for p in e98.PREDICTIONS} == X["holds"]
    assert all(pr[p]["graded"] for p in e98.PREDICTIONS)
    iv = smoke["graded_validation"]["replica"]["p1_equivalence_interval"]
    assert pr["P1"]["value"] == pytest.approx(iv["max_abs_end"]) and iv["max_abs_end"] > e98.THRESHOLDS["P1"]
    assert pr["P1"]["point_difference_reported"] == pytest.approx(X["accuracy"] - X["replica_accuracy"])
    bs = smoke["where_the_points_sit"]["points_mean_percentile_bootstrap"]
    assert pr["P5"]["lower_one_sided_95"] == pytest.approx(bs["lower_one_sided_95"]) and bs["n_clusters"] == 6
    assert bs["lower_one_sided_95"] <= pr["P5"]["value"] <= bs["hi_one_sided_95"]


def _planted(**kw):
    m = {"n_val": 250, "n_errors": 24, "p1_interval_max_abs": 0.05, "p1_difference": 0.0, "p2_agreement": 0.90,
         "p3_auroc": 0.75, "p4_capture_10": 0.25, "p5_mean_percentile": 0.55, "p5_lower_95": 0.5001}
    m.update(kw)
    return e98.grade(m)


def test_grades_at_their_thresholds():
    g = _planted()
    assert all(g[p]["graded"] and g[p]["holds"] for p in e98.PREDICTIONS), "a value at its threshold holds"
    g = _planted(p1_interval_max_abs=0.0501, p2_agreement=0.8999, p3_auroc=0.7499, p4_capture_10=0.2499,
                 p5_mean_percentile=0.5499)
    assert not any(g[p]["holds"] for p in e98.PREDICTIONS)
    g = _planted(p1_interval_max_abs=0.051, p1_difference=0.0)
    assert g["P1"]["holds"] is False, "P1 is an equivalence test: a point difference of 0 with a wide interval fails"
    g = _planted(p5_mean_percentile=0.60, p5_lower_95=0.50)
    assert g["P5"]["holds"] is False, "P5 needs its lower bound above 0.5, not only its point value"


def test_grades_below_their_floors_are_reported_not_graded():
    g = _planted(n_val=e98.MIN_VAL - 1)
    assert not any(g[p]["graded"] for p in e98.PREDICTIONS)
    assert all("holds" not in g[p] and g[p]["value"] is not None for p in e98.PREDICTIONS)
    g = _planted(n_errors=e98.MIN_ERRORS - 1)
    assert [p for p in e98.PREDICTIONS if g[p]["graded"]] == ["P1", "P2", "P5"]
    g = _planted(p1_interval_max_abs=None, p5_mean_percentile=float("nan"))
    assert g["P1"]["graded"] is False and g["P1"]["reason"].startswith("not computable")
    assert g["P5"]["graded"] is False
    assert _planted(p5_lower_95=None)["P5"]["graded"] is False


def test_the_floors_match_the_in_area_count():
    # the page's "What is known before the run": 250 validation points inside the request geometry, 24 replica errors
    assert e98.MIN_VAL < 250 and e98.MIN_ERRORS < 24
    assert e98.THRESHOLDS["P1"] == 0.05 and e98.P1_LEVEL == 0.90


def test_the_run_is_refused_while_the_page_is_a_draft(tmp_path, capsys):
    assert e98.prereg_status("# exp98\n\n**Status: DRAFT, not frozen.** Written 8 October 2026.\n") == "draft"
    assert e98.prereg_status("# exp98\n\n**Status: frozen on 9 October 2026, before any run.**\n") == "frozen"
    if e98.prereg_status() != "frozen":
        rc = e98.main(["--scores", str(tmp_path), "--labels", str(tmp_path), "--out-dir", str(tmp_path)])
        assert rc == 3 and "refused" in capsys.readouterr().out
        assert not os.listdir(tmp_path)


# ----------------------------------------------------------------------------- the readers on small cases
def test_trained_confidence_is_the_renormalised_softmax():
    P = np.array([[0.1, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0.8],       # predicted as the untrained channel
                  [0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0.0]])
    r = e98.readings(P)
    assert r["pred"].tolist() == [9, 0] and r["n_pred_untrained"] == 1
    assert r["p1"] == pytest.approx([0.1 / 0.2, 0.7])
    assert r["n_saturated_float32"] == 0


def test_percentile_counts_ties_half():
    sample = np.array([1.0, 2.0, 2.0, 3.0])                      # suspicion: higher is less confident
    assert e98.percentile_in_map(sample, np.array([2.0, 0.5, 3.5])) == pytest.approx([0.5, 1.0, 0.0])


def test_reweighting_by_the_maps_confidence_quintiles():
    sample = np.linspace(0, 1, 1000)
    u = np.array([0.05, 0.1, 0.25, 0.3, 0.45, 0.5, 0.65, 0.7, 0.85, 0.9])
    err = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1], dtype=float)
    r = e98.reweighted(err, u, sample, n_map=1_000_000)
    assert r["map_share"] == pytest.approx([0.2] * 5)
    assert r["points_per_stratum"] == [2] * 5
    assert r["estimate"] == pytest.approx(0.2 * (0 + 0 + 0.5 + 0.5 + 1))
    assert r["accuracy"] == pytest.approx(0.6)


def test_a_label_pixel_maps_to_the_score_pixel_holding_its_centre():
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    grid = {"crs": CRS.from_epsg(32737), "transform": from_origin(1000.0, 2000.0, 10, 10), "shape": (50, 60)}
    pts = [{"crs": CRS.from_epsg(32737), "X": 1000.0 + 10 * 7 + 5, "Y": 2000.0 - 10 * 3 - 5},   # pixel (3, 7)
           {"crs": CRS.from_epsg(32737), "X": 1000.0 - 5, "Y": 2000.0 - 5},                     # outside
           {"crs": CRS.from_epsg(32637), "X": 1000.0 + 10 * 2 + 5, "Y": 2000.0 - 10 * 9 - 5 - 10_000_000}]
    gi, rows, cols, how, off = e98.locate(pts, [grid])
    assert gi.tolist() == [0, -1, 0]
    assert (rows[0], cols[0]) == (3, 7) and (rows[2], cols[2]) == (9, 2)
    assert how.tolist() == ["same_crs", "outside", "reprojected"]
    assert off[0] == 0.0 and off[2] < 1e-6


# ----------------------------------------------------------------------------- the recorded run, by a second route
OUT = os.path.join(ROOT, "exp", "out")


def _recorded():
    """The recorded run (exp/out/exp98_*), read with plain numpy: V is the covered validation rows of the units file;
    the deployed class is the units file's; the replica's class is the argmax of its ten logits in exp89's units file,
    matched by window name, and its confidence the top-1 softmax over the nine trained channels; the task is the
    window name before `_point_`. Only names, labels, clusters and logits are read from exp89's file."""
    with open(os.path.join(OUT, "exp98_summary.json")) as f:
        S = json.load(f)
    z = np.load(os.path.join(OUT, "exp98_units.npz"))
    r = np.load(os.path.join(OUT, "exp89_units_awf.npz"))
    V = z["covered"] & (z["split"] == "val")
    names, y = z["names"][V].tolist(), z["label"][V]
    task = np.array([n.split("_point_")[0] for n in names])
    row = {n: i for i, n in enumerate(r["names"].tolist())}
    k = np.array([row[n] for n in names])
    assert (r["label"][k] == y).all() and (r["clusters"][k] == task).all()
    logits = r["logits"][k].astype(np.float64)
    t = logits[:, :9]
    p = np.exp(t - t.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return {"S": S, "z": z, "V": V, "y": y, "task": task, "pred": z["pred_deployed"][V], "u": z["suspicion"][V],
            "p1": z["p1"][V], "rpred": logits.argmax(axis=1), "ru": -np.log(p.max(axis=1))}


def _pairs_auroc(score, err):
    """The AUROC for errors by counting every (error, correct) pair, a tie counting one half."""
    d = score[err][:, None] - score[~err][None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def test_exp98_p1_by_a_second_route():
    """P1 recomputed: the errors and accuracies by counting; the discordant points and the exact two-sided sign test
    from binomial coefficients; the 90% task-cluster interval through a weight matrix (each resample's task counts
    times the per-task error sums), drawn as the script draws (seed 98, 20 task indices per resample, 2,000
    resamples). Equivalence within 5 points is not shown, and the interval contains 0."""
    R = _recorded()
    rec, P = R["S"]["graded_validation"]["replica"], R["S"]["prereg"]["P1"]
    e_d, e_r = R["pred"] != R["y"], R["rpred"] != R["y"]
    assert (e_d.size, int(e_d.sum()), int(e_r.sum())) == (259, 33, 27)
    assert (1 - e_d.mean()) - (1 - e_r.mean()) == pytest.approx(rec["difference"], abs=1e-12)
    a, b = int((~e_d & e_r).sum()), int((e_d & ~e_r).sum())
    assert (a, b) == (7, 13)
    p = min(1.0, 2 * sum(math.comb(a + b, i) for i in range(min(a, b) + 1)) / 2 ** (a + b))
    assert p == pytest.approx(rec["discordant"]["sign_test_two_sided_p"], abs=1e-15)
    ids = np.unique(R["task"])
    T = (R["task"][None, :] == ids[:, None]).astype(np.float64)          # tasks x points
    pick = np.random.default_rng(98).integers(0, ids.size, (2000, ids.size))
    W = np.zeros((2000, ids.size))
    np.add.at(W, (np.repeat(np.arange(2000), ids.size), pick.ravel()), 1)
    d = (W @ (T @ e_r) - W @ (T @ e_d)) / (W @ T.sum(axis=1))           # deployed accuracy minus the replica's
    lo, hi = float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))
    I = rec["p1_equivalence_interval"]
    assert lo == pytest.approx(I["lo"], abs=1e-12) and hi == pytest.approx(I["hi"], abs=1e-12)
    assert max(abs(lo), abs(hi)) == pytest.approx(P["value"], abs=1e-12)
    assert max(abs(lo), abs(hi)) > 0.05 and P["holds"] is False          # equivalence within 5 points not shown
    assert lo < 0 < hi                                                   # and no difference shown either

    def larger_end(seed, n):                                             # the same draw, any seed and size
        pk = np.random.default_rng(seed).integers(0, ids.size, (n, ids.size))
        Wn = np.zeros((n, ids.size))
        np.add.at(Wn, (np.repeat(np.arange(n), ids.size), pk.ravel()), 1)
        dn = (Wn @ (T @ e_r) - Wn @ (T @ e_d)) / (Wn @ T.sum(axis=1))
        return float(np.quantile(dn, 0.05)), float(np.quantile(dn, 0.95))

    # the record's seed sensitivity: the 200 seeds 0 to 200 other than 98, and 400,000 resamples at seed 98
    ends = np.array([max(-a_, b_) for a_, b_ in (larger_end(s, 2000) for s in range(201) if s != 98)])
    assert ends.size == 200 and (round(100 * ends.min(), 1), round(100 * ends.max(), 1)) == (4.8, 5.3)
    assert round(100 * float(np.median(ends)), 2) == 5.07 and int((ends <= 0.05).sum()) == 40
    lo4, hi4 = larger_end(98, 400_000)
    assert (round(100 * lo4, 2), round(100 * hi4, 2)) == (-5.07, 0.77) and -lo4 > 0.05


def test_exp98_classes_and_ranking_by_a_second_route():
    """P2 to P4 recomputed: the agreement with the replica by counting; the AUROC of the deployed confidence, and of
    the replica's, by counting every (error, correct) pair; the 10% review as the round(0.1 n) least confident points
    by a stable sort (no ties, so the order is unique); the confidence written is exp(-suspicion). Also the counts the
    record cites beside them: the train points, the condition codes, the burn codes and the identical 10 m inputs."""
    R = _recorded()
    S, y, pred = R["S"], R["y"], R["pred"]
    g = S["graded_validation"]
    err = pred != y
    assert int((pred == R["rpred"]).sum()) == 238
    assert (pred == R["rpred"]).mean() == pytest.approx(S["prereg"]["P2"]["value"], abs=1e-12)
    assert np.abs(R["p1"] - np.exp(-R["u"])).max() < 1e-12 and np.unique(R["u"]).size == R["u"].size
    assert _pairs_auroc(R["u"], err) == pytest.approx(S["prereg"]["P3"]["value"], abs=1e-12)
    assert _pairs_auroc(R["ru"], R["rpred"] != y) == pytest.approx(g["replica"]["auroc_replica"], abs=1e-9)
    order = np.argsort(-R["u"], kind="stable")                           # least confident first
    for b, n_err in ((0.05, 7), (0.1, 14), (0.2, 19)):
        k = int(round(b * err.size))
        assert int(err[order][:k].sum()) == n_err
        assert err[order][:k].sum() / err.sum() == pytest.approx(g["ranking"]["capture"][str(b)], abs=1e-12)
    assert (int(round(0.1 * err.size)), int(err[order][:25].sum())) == (26, 14)   # 25 points give the same 14
    assert S["prereg"]["P4"]["value"] == pytest.approx(14 / 33, abs=1e-12)
    assert all(S["prereg"][q]["holds"] is True for q in ("P2", "P3", "P4")) and int((pred == 9).sum()) == 0
    z, V = R["z"], R["V"]
    T = z["covered"] & (z["split"] == "train")
    assert (int(T.sum()), int((z["pred_deployed"][T] != z["label"][T]).sum())) == (783, 16)
    cond, burn, same = z["condition"][V], z["burned"][V], z["input_identical"][V]
    assert [(int((cond == c).sum()), int(err[cond == c].sum())) for c in (0, 1)] == [(20, 2), (239, 31)]
    assert (burn == 0).all() and int((same == 1).sum()) == 258 and int((same == 0).sum()) == 1


def test_exp98_p5_bracket_from_the_map_quantiles():
    """P5 cannot be recomputed exactly: the map's confidence sample is not saved. The summary's five map quantiles
    bracket each point's percentile (the share of the map less confident than the point lies between the quantile
    levels on either side of its confidence), so the recorded mean must lie inside the bracket's mean, and the
    bracket's lower edge, bootstrapped over the tasks as P5's bound is (2,000 resamples, seed 98), must keep the bound
    above 0.5. P5's first condition (mean at least 0.55) is checked only from the summary: the bracket starts at 0.547."""
    R = _recorded()
    w = R["S"]["where_the_points_sit"]
    q = w["map_confidence_quantiles"]
    levels = np.array([0.0] + [float(x) for x in q] + [1.0])
    j = np.searchsorted(np.array([q[x] for x in q]), R["p1"], side="right")
    lo_edge, hi_edge = levels[j], levels[j + 1]
    assert (round(float(lo_edge.mean()), 3), round(float(hi_edge.mean()), 3)) == (0.547, 0.741)
    assert lo_edge.mean() <= w["points_mean_percentile"] <= hi_edge.mean()
    ids = np.unique(R["task"])
    rows = {c: np.flatnonzero(R["task"] == c) for c in ids}
    rng = np.random.default_rng(98)
    v = [lo_edge[np.concatenate([rows[c] for c in ids[rng.integers(0, ids.size, ids.size)]])].mean()
         for _ in range(2000)]
    assert np.quantile(v, 0.05) > 0.5
    assert w["points_mean_percentile_bootstrap"]["lower_one_sided_95"] > 0.5 and R["S"]["prereg"]["P5"]["holds"] is True
