"""exp104 (docs/plan/forest_loss_2026.md): Ai2's Forest Loss Driver model graded at 2026 MapBiomas Alerta alerts.
Written before the page was frozen and before the model predicted any sampled alert. The frame rule and its hash on a
synthetic snapshot; the sampler's sizes and weights; the crosswalk; the window (centre, size, CRS, time ranges against
Ai2's offsets, rslearn's metadata.json); the configs against Ai2's pinned files; P1, P2 and P3 on constructed cases
against independent computations; the refusal before freezing; and nothing written carries an alert code or a
position. No network; numpy only (one check against pyproj is skipped without it)."""
import itertools
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp104_forest_loss as e104       # noqa: E402
from oe_inferencex import estimate as est   # noqa: E402


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("exp104"))
    rec, units, truth = e104.smoke(out_dir=out, log=lambda *a: None)
    return {"out": out, "rec": rec, "units": units, "truth": truth}


def props(code, driver="agriculture", biomes="Amazônia", status="published", date="2026-03-01", area=3.0):
    return {"alert_code": code, "area_ha": area, "biomes": biomes, "detected_at": date, "image_after_at": date,
            "deforestation_driver": driver, "status_name": status}


# ----------------------------------------------------------------------------- the page's numbers
def test_the_constants_are_the_pages():
    with open(e104.PLAN, encoding="utf-8") as f:
        page = f.read()
    assert e104.FRAME_SHA_PREFIX in page and "4,113" in page
    assert "3,565 agriculture, 380 garimpo, 9 mining, 7 roads, 151 others" in page and "1 without a driver" in page
    assert sum(e104.FRAME_DRIVERS.values()) == e104.FRAME_N == 4113
    assert e104.SEED == 104 and "seed 104" in page
    assert e104.SAMPLE_SIZE == {"agriculture": 300} and "300 of 3,565" in page and "all 389" in page
    assert e104.THRESHOLDS == {"P1": 0.70, "P2": 0.25}
    assert e104.FLOORS == {"min_with_prediction": 500, "min_errors": 20, "min_correct": 20}
    assert "at least 500 alerts" in page and "at least 20 errors and 20 correct" in page
    assert e104.AI2_ACCURACY == 0.761 and "76.1%" in page
    assert math.isclose(3565 / 300, 11.883333, rel_tol=1e-6) and "11.88" in page


# ----------------------------------------------------------------------------- the frame
def test_the_frame_rule():
    assert e104.in_frame(props(1))
    assert e104.in_frame(props(1, biomes="Amazônia,Cerrado")), "a multi-biome alert holding Amazônia is in"
    assert not e104.in_frame(props(1, biomes="Cerrado"))
    assert not e104.in_frame(props(1, biomes="Cerrado,Pantanal"))
    assert not e104.in_frame(props(1, status="cancelled"))
    assert e104.in_frame(props(1, date="2026-01-01")) and e104.in_frame(props(1, date="2026-06-30"))
    assert not e104.in_frame(props(1, date="2025-12-31")) and not e104.in_frame(props(1, date="2026-07-01"))


def test_the_frame_hash_and_its_refusals():
    ps = [props(1_600_000 + 3 * k, driver=d) for k, d in enumerate(["agriculture"] * 5 + ["garimpo", "roads", None])]
    ps += [props(1_700_000, biomes="Cerrado"), props(1_700_001, status="cancelled")]
    codes = sorted(p["alert_code"] for p in ps[:8])
    import hashlib
    want = hashlib.sha256(",".join(map(str, codes)).encode()).hexdigest()[:16]
    assert e104.sha16(reversed(codes)) == want, "the hash is over the SORTED codes"
    drivers = {"agriculture": 5, "garimpo": 1, "roads": 1, "none": 1}
    frame, rec = e104.build_frame(ps, want, 8, drivers)
    assert [p["alert_code"] for p in frame] == codes and rec["by_driver"] == dict(sorted(drivers.items()))
    with pytest.raises(ValueError, match="hash"):
        e104.build_frame(ps, "0" * 16, 8, drivers)
    with pytest.raises(ValueError, match="holds 8 alerts"):
        e104.build_frame(ps, want, 9, drivers)
    with pytest.raises(ValueError, match="drivers"):
        e104.build_frame(ps, want, 8, dict(drivers, agriculture=4))
    with pytest.raises(ValueError, match="twice"):
        e104.build_frame(ps + [ps[0]], None, None, None)


def test_the_frame_on_the_pages_snapshot_when_present(tmp_path):
    """The snapshot of 11 October 2026 is not committed (it holds bounding boxes); where it is on disk, the rule
    reproduces the page's count and hash."""
    snap = os.environ.get("E104_SNAPSHOT")
    if not snap or not os.path.isfile(snap):
        pytest.skip("E104_SNAPSHOT not set")
    rec, units = e104.run_frame(str(tmp_path / "private"), snapshot=snap, log=lambda *a: None)
    assert rec["frame"]["n_frame"] == 4113 and rec["frame"]["sha256_prefix"] == e104.FRAME_SHA_PREFIX
    assert len(units) == 300 + 389 + 7


# ----------------------------------------------------------------------------- the sample
def test_the_sampler_sizes_weights_and_reproducibility():
    frame = [props(2_000_000 + k, driver=d) for k, d in
             enumerate(["agriculture"] * 50 + ["garimpo"] * 7 + ["mining"] * 2 + ["roads"] * 3 + ["others"] * 4)]
    units, rec = e104.draw_sample(frame, sizes={"agriculture": 10})
    by = {s: [u for u in units if u["stratum"] == s] for s in e104.STRATUM_NAMES}
    assert [len(by[s]) for s in e104.STRATUM_NAMES] == [10, 9, 3]
    assert rec["not_in_any_stratum"] == 4 and rec["n_sample"] == 22
    assert all(u["weight"] == 5.0 for u in by["agriculture"]) and all(u["weight"] == 1.0 for u in by["mining"])
    ag = sorted(2_000_000 + k for k in range(50))
    pick = np.sort(np.random.default_rng(104).choice(50, 10, replace=False))
    assert [u["alert_code"] for u in by["agriculture"]] == [ag[i] for i in pick], "default_rng(104) over sorted codes"
    units2, _ = e104.draw_sample(list(reversed(frame)), sizes={"agriculture": 10})
    assert units2 == units, "the sample does not depend on the snapshot's order"
    assert [u["window"] for u in units] == [f"u{k:04d}" for k in range(22)]
    assert all(str(u["alert_code"]) not in u["window"] for u in units)
    with pytest.raises(ValueError):
        e104.draw_sample(frame, sizes={"agriculture": 51})


# ----------------------------------------------------------------------------- the crosswalk and the classes
def test_the_crosswalk_and_ai2s_class_order():
    import exp89_finetuned_checkpoints as e89
    assert e104.FLD_CLASSES == e89.FLD_CLASSES and e104.FLD_LAYERS == e89.FLD_LAYERS
    assert e104.MODEL["revision"] == e89.ARMS["fld"]["model_revision"]
    assert e104.MODEL["sha256"] == e89.ARMS["fld"]["model_sha256"]
    with open(os.path.join(e104.CONFIG_DIR, "model.yaml"), encoding="utf-8") as f:
        text = f.read()
    assert '["agriculture", "mining", "airstrip", "road", "logging", "burned", "landslide", "hurricane", "river", ' \
           '"none"]' in text, "the probabilities come in the order of the task's classes"
    assert e104.CROSSWALK == {"agriculture": "agriculture", "garimpo": "mining", "mining": "mining", "roads": "road"}
    for s, drivers in e104.STRATA.items():
        assert {e104.CROSSWALK[d] for d in drivers} == {e104.STRATUM_CLASS[s]}
    for d in ("others", "urban_expansion", "reservoir_or_dam", "renewable_energy_project", "natural_cause",
              "aquaculture", "none"):
        assert e104.stratum_of(d) is None, f"{d} has no model class and is never sampled"


# ----------------------------------------------------------------------------- the window
def test_mercator_and_the_pixel_size():
    assert math.isclose(e104.PIXEL_SIZE, 9.554628535647032, rel_tol=0, abs_tol=1e-12)
    x, y = e104.mercator(0.0, 0.0)
    assert x == 0 and abs(y) < 1e-9
    x, y = e104.mercator(180.0, 0.0)
    assert math.isclose(float(x), math.pi * e104.EARTH_R)
    pyproj = pytest.importorskip("pyproj")
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    for lon, lat in ((-60.1, -8.3), (-45.0, 2.5), (-72.9, -11.0)):
        a, b = t.transform(lon, lat)
        x, y = e104.mercator(lon, lat)
        assert abs(a - float(x)) < 1e-6 and abs(b - float(y)) < 1e-6


def test_centroid_inside_and_holes():
    sq = {"type": "Polygon", "coordinates": [[[0, 0], [0.01, 0], [0.01, 0.01], [0, 0.01], [0, 0]]]}
    f = e104.polygon_facts(sq)
    cx, cy = e104.mercator(0.005, 0.005)
    assert abs(f["cx"] - float(cx)) < 0.01 and abs(f["cy"] - float(cy)) < 1.0 and f["inside"]
    rev = {"type": "Polygon", "coordinates": [list(reversed(sq["coordinates"][0]))]}
    assert abs(e104.polygon_facts(rev)["area_m2"] - f["area_m2"]) < 1e-6, "ring orientation does not matter"
    holed = {"type": "Polygon", "coordinates": [sq["coordinates"][0],
                                                [[0.004, 0.004], [0.006, 0.004], [0.006, 0.006], [0.004, 0.006],
                                                 [0.004, 0.004]]]}
    h = e104.polygon_facts(holed)
    assert not h["inside"] and math.isclose(h["area_m2"], f["area_m2"] * 0.96, rel_tol=1e-3)
    multi = {"type": "MultiPolygon", "coordinates": [sq["coordinates"],
                                                     [[[0.02, 0], [0.03, 0], [0.03, 0.01], [0.02, 0.01], [0.02, 0]]]]}
    m = e104.polygon_facts(multi)
    assert not m["inside"] and abs(m["cx"] - float(e104.mercator(0.015, 0)[0])) < 0.01
    with pytest.raises(ValueError):
        e104.polygon_facts({"type": "Point", "coordinates": [0, 0]})


def test_the_second_route_to_the_windows_centroid_and_the_letter_of_ai2s_code():
    """The dataset check's gated route (ring_centre_px: pyproj, and shapely per ring) against the windows' own centroid
    (polygon_facts, numpy in EPSG:3857) on squares, holes, multi-part and self-intersecting polygons in either axis
    order; Ai2's code's route (shapely's centroid of the unrepaired polygon) on the valid ones; and make_valid moving a
    self-intersecting polygon's centroid, which is why rslearn's route is gated on valid polygons only. Job 1286487
    found the letter of Ai2's code 2 to about 1,870 px off on 24 of 343 invalid multi-part MapBiomas polygons."""
    import shapely
    import shapely.geometry
    R = e104.PIXEL_SIZE
    lon0, lat0, d = -55.0, -6.0, 0.004
    sq = {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon0 + d, lat0], [lon0 + d, lat0 + d], [lon0, lat0 + d],
                                              [lon0, lat0]]]}
    holed = {"type": "Polygon", "coordinates": [sq["coordinates"][0],
                                                [[lon0 + d / 4, lat0 + d / 4], [lon0 + d / 2, lat0 + d / 4],
                                                 [lon0 + d / 2, lat0 + d / 2], [lon0 + d / 4, lat0 + d / 2],
                                                 [lon0 + d / 4, lat0 + d / 4]]]}
    multi = {"type": "MultiPolygon", "coordinates": [sq["coordinates"],
                                                     [[[lon0 + 2 * d, lat0], [lon0 + 4 * d, lat0],
                                                       [lon0 + 4 * d, lat0 + d], [lon0 + 2 * d, lat0 + d],
                                                       [lon0 + 2 * d, lat0]]]]}
    bowtie = {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon0 + 3 * d, lat0 + 3 * d], [lon0 + 3 * d, lat0],
                                                  [lon0, lat0 + d], [lon0, lat0]]]}
    eight = [[lon0 + d * x, lat0 + d * y] for x, y in ((0, 0), (2, 1), (2.5, 2), (3, 1), (2, 0), (0, 1.5), (0, 0))]
    multi_bad = {"type": "MultiPolygon", "coordinates": [[eight], [[[lon0 + 5 * d, lat0], [lon0 + 6 * d, lat0],
                                                                   [lon0 + 6 * d, lat0 + d], [lon0 + 5 * d, lat0 + d],
                                                                   [lon0 + 5 * d, lat0]]]]}
    swap = lambda c: [c[1], c[0]] if isinstance(c[0], float) else [swap(x) for x in c]   # noqa: E731
    for g in (sq, holed, multi, bowtie, multi_bad):
        f = e104.polygon_facts(g)
        ours = (f["cx"] / R, -f["cy"] / R)
        for got in (e104.ring_centre_px(g),
                    e104.ring_centre_px({"type": g["type"], "coordinates": swap(g["coordinates"])}, swap=True)):
            assert abs(got[0] - ours[0]) < 0.05 and abs(got[1] - ours[1]) < 0.05
        if shapely.geometry.shape(g).is_valid:
            a = e104.ai2_centre_px(g)
            assert abs(a[0] - ours[0]) < 0.05 and abs(a[1] - ours[1]) < 0.05
    for g in (bowtie, multi_bad):
        assert not shapely.geometry.shape(g).is_valid
        m = shapely.make_valid(shapely.geometry.shape(g)).centroid
        mx, my = (v / R for v in e104.mercator(m.x, m.y))
        f = e104.polygon_facts(g)
        assert math.hypot(mx - f["cx"] / R, my - f["cy"] / R) > 5, "make_valid moves this centroid by many pixels"


def test_the_window_centre_size_crs_and_time_ranges():
    cx, cy = e104.mercator(-60.123, -8.456)
    b = e104.window_bounds(float(cx), float(cy))
    assert b[2] - b[0] == 128 and b[3] - b[1] == 128
    col, row = float(cx) / e104.PIXEL_SIZE, float(cy) / -e104.PIXEL_SIZE
    assert b[0] + 64 <= col < b[0] + 65 and b[1] + 64 <= row < b[1] + 65, "the centroid is pixel (64, 64)"
    t = e104.event_time("2026-05-17")
    assert t == datetime(2026, 5, 17, tzinfo=timezone.utc)
    meta = e104.window_metadata(b, t)
    assert list(meta) == ["projection", "bounds", "time_range", "options"]
    assert meta["projection"] == {"crs": "EPSG:3857", "x_resolution": e104.PIXEL_SIZE, "y_resolution": -e104.PIXEL_SIZE}
    assert meta["time_range"] == ["2026-05-17T00:00:00+00:00"] * 2, "Ai2's oe_start_time = oe_end_time = the date"
    # Ai2's offsets, read from the committed dataset config, and rslearn's rule (offset, then end = start + duration)
    per = e104.dataset_periods()
    assert per["pre_sentinel2"] == (-300, 180, 4, 4, "CONTAINS") and per["post_sentinel2"] == (7, 180, 4, 4, "CONTAINS")
    assert {k: v[:2] for k, v in per.items()} == e104.PERIODS
    pre = e104.request_range(t, t, -300, 180)
    post = e104.request_range(t, t, 7, 180)
    assert pre == (t - timedelta(days=300), t - timedelta(days=120))
    assert post == (t + timedelta(days=7), t + timedelta(days=187))
    assert pre[1] < t < post[0], "the pre period ends before the event, the post period starts after it"


def test_the_windows_on_synthetic_polygons_either_axis_order(tmp_path):
    frame = [props(3_000_000 + k, driver=d, area=4.0) for k, d in enumerate(["agriculture"] * 6 + ["garimpo"] * 4)]
    units, _ = e104.draw_sample(frame, sizes={"agriculture": 6})
    got = {}
    for latlon in (False, True):
        priv = str(tmp_path / f"p{int(latlon)}")
        e104.private_dir(priv)
        e104.write_private_json({"units": units, "strata": {}}, os.path.join(priv, "e104_sample.json"))
        rec = e104.run_windows(priv, log=lambda *a: None, get=e104.smoke_wfs(units, latlon=latlon))
        assert rec["axis_order"] == ("latlon" if latlon else "lonlat")
        assert rec["polygons_missing"] == 1 and rec["n_windows"] == 9, "unit 0 has no polygon in the stand-in WFS"
        assert rec["config_is_ai2_mpc_version"]
        ds = os.path.join(priv, "dataset")
        got[latlon] = {n: json.load(open(os.path.join(ds, "windows", "e104", n, "metadata.json")))
                       for n in sorted(os.listdir(os.path.join(ds, "windows", "e104")))}
        with open(os.path.join(ds, "config.json"), "rb") as f, \
                open(os.path.join(e104.CONFIG_DIR, "dataset.json"), "rb") as g:
            assert f.read() == g.read()
    assert got[False] == got[True], "the same windows whichever axis order the WFS answers in"


def test_a_private_directory_inside_a_checkout_is_refused():
    with pytest.raises(ValueError, match="git checkout"):
        e104.private_dir(os.path.join(ROOT, "exp", "out", "e104_private"), create=False)


# ----------------------------------------------------------------------------- the configs
def test_the_configs_reverse_to_ai2s_pinned_files():
    rec = e104.check_configs()
    assert rec["ok"], rec
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(e104.CONFIG_DIR, "model.yaml"), encoding="utf-8") as f:
        der = yaml.safe_load(f)
    ai2 = yaml.safe_load(e104.reverse_model_yaml(open(os.path.join(e104.CONFIG_DIR, "model.yaml")).read()))
    assert der["trainer"]["logger"]["class_path"] == "lightning.pytorch.loggers.CSVLogger"
    der["trainer"].pop("logger"), ai2["trainer"].pop("logger")
    assert der == ai2, "nothing but the logger differs"
    with open(os.path.join(e104.CONFIG_DIR, "dataset.json"), encoding="utf-8") as f:
        d = json.load(f)
    for name in e104.S2_LAYERS:
        src = d["layers"][name]["data_source"]
        assert src["class_path"] == "rslearn.data_sources.planetary_computer.Sentinel2"
        assert src["init_args"]["sort_by"] == "eo:cloud_cover" and src["init_args"]["harmonize"] is True
        assert src["query_config"] == {"max_matches": 4, "min_matches": 4, "space_mode": "CONTAINS"}
        assert src["ingest"] is False
    back = e104.reverse_dataset_config(d)
    for name in e104.S2_LAYERS:
        src = back["layers"][name]["data_source"]
        assert src["init_args"]["query"] == {"sort_by": "CLOUD_COVER", "sort_direction": "ASC"}, \
            "Ai2's ordering: least cloud first, as eo:cloud_cover ascending"
    assert e104.derive_dataset_config(back) == d


# ----------------------------------------------------------------------------- the measures on constructed cases
def brute_auroc(u, e, w):
    num = den = 0.0
    for i, j in itertools.product(np.flatnonzero(e == 1), np.flatnonzero(e == 0)):
        ww = w[i] * w[j]
        den += ww
        num += ww * (1.0 if u[i] > u[j] else 0.5 if u[i] == u[j] else 0.0)
    return num / den


def test_p1_p2_on_constructed_cases():
    u = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    e = np.array([1, 0, 1, 0, 0, 0, 0, 1, 0, 0], float)
    w = np.array([2.0, 2, 2, 2, 2, 1, 1, 1, 1, 1])
    st = np.array(["agriculture"] * 5 + ["mining"] * 5)
    r = e104.ranking(u, e, w, st, n_boot=0)
    assert math.isclose(r["weighted"]["auroc"], brute_auroc(u, e, w), rel_tol=1e-12)
    # the least confident 10% of the weight (1.5 of 15) is three quarters of unit 0, an error: capture 1.5 / 5
    theta = (2 + 2 + 1) / 15
    assert math.isclose(r["weighted"]["error_rate"], theta)
    assert math.isclose(r["weighted"]["capture"]["0.1"], 1.5 / 5)
    ceil = min(1, 0.1 / theta)
    assert math.isclose(r["weighted"]["gap_closed"]["0.1"], (1.5 / 5 - 0.1) / (ceil - 0.1))
    perfect = e104.ranking(e + 0.0, e, w, st, n_boot=0)
    assert perfect["weighted"]["auroc"] == 1.0 and math.isclose(perfect["weighted"]["gap_closed"]["0.1"], 1.0)
    worst = e104.ranking(-e, e, w, st, n_boot=0)
    assert worst["weighted"]["auroc"] == 0.0 and worst["weighted"]["gap_closed"]["0.1"] < 0
    flat = e104.ranking(np.ones_like(u), e, w, st, n_boot=0)
    assert flat["weighted"]["auroc"] == 0.5 and abs(flat["weighted"]["gap_closed"]["0.1"]) < 1e-12
    b = e104.ranking(u, e, w, st, n_boot=300)
    bs = b["auroc_weighted_bootstrap"]
    assert bs["n_resamples"] + bs["skipped"] == 300 and bs["lo90"] <= r["weighted"]["auroc"] <= bs["hi90"]
    assert e104.ranking(u, np.zeros(10), w, st, 10)["note"].startswith("no ranking")


def test_p3_the_stratified_exact_interval():
    r = e104.stratified({"agriculture": 70, "mining": 40}, {"agriculture": 290, "mining": 380},
                        {"agriculture": 3565, "mining": 389})
    theta, lo, hi, L = est._union_interval([70, 40], [290, 380], [3565, 389])
    assert (r["estimate"], r["low"], r["high"], r["strata_in_interval"]) == (theta, lo, hi, L) and L == 2
    assert math.isclose(theta, 3565 / 3954 * 70 / 290 + 389 / 3954 * 40 / 380)
    m = r["by_stratum"]["mining"]
    assert (m["low"], m["high"]) == est.hypergeom_interval(40, 380, 389)
    census = e104.stratified({"agriculture": 70, "mining": 40}, {"agriculture": 290, "mining": 389},
                             {"agriculture": 3565, "mining": 389})
    assert census["strata_in_interval"] == 1, "a stratum labelled in full enters at its exact rate"
    a, b = est.hypergeom_interval(70, 290, 3565)
    w = 3565 / 3954
    assert math.isclose(census["low"], w * a + (1 - w) * 40 / 389, rel_tol=1e-12)
    assert math.isclose(census["high"], w * b + (1 - w) * 40 / 389, rel_tol=1e-12)


def test_grades_at_thresholds_and_floors():
    base = {"n_with_prediction": 600, "n_errors": 150, "n_correct": 450}
    g = e104.grade(dict(base, p1_auroc=0.70, p2_gap_closed=0.25))
    assert g["P1"]["holds"] and g["P2"]["holds"] and g["reading"].startswith("P1 and P2 hold")
    g = e104.grade(dict(base, p1_auroc=0.6999, p2_gap_closed=0.3))
    assert not g["P1"]["holds"] and g["P2"]["holds"] and g["reading"].startswith("one holds")
    g = e104.grade(dict(base, p1_auroc=0.5, p2_gap_closed=0.1))
    assert g["reading"].startswith("both fail")
    for k, v in (("n_with_prediction", 499), ("n_errors", 19), ("n_correct", 19)):
        g = e104.grade(dict(base, **{k: v}, p1_auroc=0.9, p2_gap_closed=0.9))
        assert not g["P1"]["graded"] and not g["P2"]["graded"] and "without verdicts" in g["reading"]
    assert e104.grade(dict(base, p1_auroc=None, p2_gap_closed=0.5))["P1"]["graded"] is False


# ----------------------------------------------------------------------------- the smoke, end to end
def test_the_smoke_end_to_end(smoke):
    rec, U, truth = smoke["rec"], smoke["units"], smoke["truth"]
    s = rec["smoke"]
    assert s["frame"]["n_frame"] == 162 and s["frame"]["multi_biome_in_frame"] == 2
    assert s["frame"]["excluded"] == {"other_biomes": 10, "not_published": 2, "detected_outside_dates": 4}
    assert {k: v["n"] for k, v in s["sample"]["strata"].items()} == {"agriculture": 40, "mining": 30, "roads": 3}
    assert s["windows"]["n_windows"] == 72 and s["windows"]["polygons_missing"] == 1
    assert s["configs"]["ok"] and s["reader"]["class_property"] == "new_label"
    assert s["imagery"]["total"]["all_eight"] == 69 == s["reader"]["windows_read"]
    assert rec["units"]["agriculture"]["with_prediction"] == 37 and rec["units"]["mining"]["with_prediction"] == 29
    # the npz against the planted truth, by a second route
    has, g = U["has_prediction"], U["graded"]
    assert int(has.sum()) == len(truth) == 69
    assert int((U["pred"][has] != U["label"][has]).sum()) == s["planted"]["errors"]
    G = has & g
    w = U["weight"][G]
    assert np.allclose(w[U["stratum"][G] == 0], 120 / 37) and np.allclose(w[U["stratum"][G] == 1], 30 / 29)
    err = (U["pred"][G] != U["label"][G]).astype(float)
    u = -np.log(U["probs"][G].astype(np.float64).max(1))
    assert np.allclose(u, U["suspicion"][G])
    assert math.isclose(rec["prereg"]["P1"]["value"], brute_auroc(u, err, w), rel_tol=1e-9)
    k = {s_: int(err[U["stratum"][G] == i].sum()) for i, s_ in enumerate(("agriculture", "mining"))}
    theta = 120 / 150 * k["agriculture"] / 37 + 30 / 150 * k["mining"] / 29
    assert math.isclose(rec["p3_error_rate"]["estimate"], theta, rel_tol=1e-12)
    assert rec["roads_reported"]["n"] == 3 and not U["graded"][U["stratum"] == 2].any()
    ci = rec["centroid_inside_polygon"]      # report-only; the smoke's 'u', 'holed' and 'multi' polygons lie outside
    assert ci["inside"]["n"] + ci["outside"]["n"] == int(G.sum()) and ci["inside"]["n"] > 0 and ci["outside"]["n"] > 0
    assert ci["inside"]["errors"] + ci["outside"]["errors"] == int(err.sum())


def test_the_smoke_cli(tmp_path, capsys):
    assert e104.main(["--smoke", "--out-dir", str(tmp_path)]) == 0
    for name in ("exp104_summary_smoke.json", "exp104_units_smoke.npz"):
        assert os.path.isfile(tmp_path / name)


# ----------------------------------------------------------------------------- refusals and guards
def test_the_grade_is_refused_for_a_draft_page(tmp_path, capsys, monkeypatch):
    """The refusal itself, with the page's status set to draft (the page has been frozen since 54b1dfe)."""
    monkeypatch.setattr(e104, "prereg_status", lambda text=None: "draft")
    rc = e104.main(["--grade", "--private-dir", str(tmp_path), "--probs", str(tmp_path), "--out-dir", str(tmp_path)])
    assert rc == 3 and "refused" in capsys.readouterr().out
    with pytest.raises(PermissionError):
        e104.run_grade(str(tmp_path), str(tmp_path), str(tmp_path), status="draft")
    monkeypatch.undo()
    assert e104.prereg_status("**Status: frozen on 12 October 2026.**") == "frozen"
    assert e104.prereg_status("**Status: draft, written 11 October 2026.** not frozen") == "draft"
    assert e104.prereg_status() == "frozen"


def test_predictions_out_of_ai2s_order_are_refused(tmp_path):
    P = np.zeros((10, 1, 2), np.float32)
    P[0, 0, 0], P[1, 0, 1] = 1.0, 1.0
    np.save(tmp_path / "window_probs.npy", P)
    with open(tmp_path / "window_probs.csv", "w") as f:
        f.write("window_col,window,feature,crs,x,y,argmax,class\n0,e104/u0000,0,,,,0,agriculture\n"
                "1,e104/u0001,0,,,,1,road\n")
    with pytest.raises(ValueError, match="class order"):
        e104.read_probs(str(tmp_path))


def test_nothing_written_carries_a_code_or_a_position(smoke, tmp_path):
    codes = list(range(e104.SMOKE_CODE0, e104.SMOKE_CODE0 + 7 * 400, 7))
    with open(os.path.join(smoke["out"], "exp104_summary_smoke.json"), encoding="utf-8") as f:
        text = f.read()
    rec = json.loads(text)
    e104.check_no_codes(rec, codes)
    import exp98_awf_deployment as e98
    e98.check_no_coordinates(rec)
    assert "alert_code" not in text and "u0000" not in text
    z = np.load(os.path.join(smoke["out"], "exp104_units_smoke.npz"))
    assert set(z.files) == {"stratum", "stratum_names", "graded", "has_prediction", "weight", "label", "pred", "probs",
                            "suspicion", "top1", "error", "class_names"}
    for k in z.files:
        a = z[k]
        if a.dtype.kind in "iu":
            assert not np.isin(a, codes).any()
    with pytest.raises(ValueError, match="alert code"):
        e104.check_no_codes({"n": 3, "deep": [{"k": codes[5]}]}, codes)
    with pytest.raises(ValueError, match="alert code"):
        e104.check_no_codes({str(codes[0]): 1}, codes)
    with pytest.raises(ValueError, match="alert code"):
        e104.dump({"a": float(codes[1])}, str(tmp_path / "r.json"), codes=codes)
    with pytest.raises(ValueError, match="position"):
        e104.dump({"bounds": [1, 2, 3, 4]}, str(tmp_path / "r.json"))
    with pytest.raises(ValueError):
        e104.save_units({"alert_code": np.array([1])}, str(tmp_path / "u.npz"))
    with pytest.raises(ValueError):
        e104.save_units({"pred": np.array([codes[2]])}, str(tmp_path / "u.npz"), codes=codes)


def test_the_units_rows_are_shuffled_and_the_name_lists_kept(tmp_path):
    """shuffle_units on a dict shaped like the grade's (per-row arrays plus the two name lists): every row array is
    permuted by one permutation, kept privately; the name lists are untouched. Until the fix the name lists made it
    return the rows in alert-code order (exp104's first grade output)."""
    n = 200
    stratum = np.repeat([0, 1, 2], [80, 110, 10])
    probs = np.random.default_rng(0).random((n, 10)).astype(np.float32)
    units = {"stratum": stratum, "stratum_names": np.array(["a", "b", "c"]), "label": np.arange(n),
             "probs": probs, "class_names": np.array([f"k{i}" for i in range(10)])}
    out = e104.shuffle_units(units, priv=str(tmp_path))
    perm = np.load(tmp_path / "units_permutation.npy")
    assert sorted(perm.tolist()) == list(range(n)) and not np.array_equal(perm, np.arange(n))
    assert np.array_equal(out["label"], perm) and np.array_equal(out["stratum"], stratum[perm])
    assert np.array_equal(out["probs"], probs[perm])
    assert out["stratum_names"].tolist() == ["a", "b", "c"] and out["class_names"].size == 10
    assert (np.diff(out["stratum"]) != 0).sum() > 10, "not in stratum blocks"
    with pytest.raises(ValueError):
        e104.shuffle_units({"label": np.arange(3)})
    with pytest.raises(ValueError):
        e104.shuffle_units({"stratum": np.arange(3), "label": np.arange(4)})


def test_the_smokes_units_are_shuffled(smoke):
    st = np.asarray(smoke["units"]["stratum"])
    assert (np.diff(st) != 0).sum() > 3, "the smoke's units must not come out in stratum blocks"


def test_exp104_recorded_by_a_second_route():
    """The recorded grade and the post-hoc decomposition recomputed from the units file with code written here (pair
    counting, no import of the package's metrics or of exp104's grade): the design-weighted AUROC overall, by stratum
    and by kind of disagreement, the stratified disagreement rate, and that the rows are not in stratum blocks."""
    out = os.path.join(ROOT, "exp", "out")
    S = json.load(open(os.path.join(out, "exp104_summary.json"), encoding="utf-8"))
    H = json.load(open(os.path.join(out, "exp104_posthoc.json"), encoding="utf-8"))
    u = np.load(os.path.join(out, "exp104_units.npz"), allow_pickle=False)
    assert S["prereg_status"] == "frozen" and S["dirty"] is False and S["commit"].startswith("deace9d")
    st, w, e, s, pred = u["stratum"], u["weight"], u["error"] == 1, u["suspicion"], u["pred"]
    G = u["graded"].astype(bool)
    assert (np.diff(st) != 0).sum() > 100, "the rows are shuffled"

    def pairs(pos, neg):
        P, N = np.flatnonzero(pos), np.flatnonzero(neg)
        num = sum(w[i] * (w[N] * ((s[i] > s[N]) + 0.5 * (s[i] == s[N]))).sum() for i in P)
        return num / (w[P].sum() * w[N].sum())

    ok = G & ~e
    assert pairs(G & e, ok) == pytest.approx(S["prereg"]["P1"]["value"], abs=1e-9) == pytest.approx(0.7704, abs=1e-4)
    names = [str(c) for c in u["class_names"]]
    lab = np.isin(pred, [names.index("agriculture"), names.index("mining")])
    assert pairs(G & e & ~lab, ok) == pytest.approx(
        H["groups"]["to_a_class_mapbiomas_does_not_record"]["weighted_auroc_against_correct"], abs=1e-9)
    assert pairs(G & e & lab, ok) == pytest.approx(
        H["groups"]["between_agriculture_and_mining"]["weighted_auroc_against_correct"], abs=1e-9)
    assert ((G & e & ~lab).sum(), (G & e & lab).sum()) == (53, 18)
    for k, name in enumerate(("agriculture", "mining")):
        m = G & (st == k)
        assert pairs(m & e, m & ~e) == pytest.approx(H["by_stratum"][name]["auroc"], abs=1e-9)
    k = {name: int((G & e & (st == i)).sum()) for i, name in enumerate(("agriculture", "mining"))}
    assert k == {"agriculture": 33, "mining": 38}
    theta = (3565 * 33 / 300 + 38) / (3565 + 389)
    assert S["p3_error_rate"]["estimate"] == pytest.approx(theta, rel=1e-12)
