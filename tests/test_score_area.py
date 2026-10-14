"""scripts/score_area.py without torch or network: the geometry, the imagery plan, the windowing and the files, with a
stub model in place of the fine-tuned one. The written raster must round-trip through `oe-inferencex assess` and
`sample`, which is the point of the script."""
import csv
import datetime as dt
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="the scores provider writes GeoTIFFs")
pytest.importorskip("shapely", reason="the imagery plan intersects footprints")
import shapely.geometry  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("score_area", os.path.join(ROOT, "scripts", "score_area.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)

UTC = dt.timezone.utc
NAMES = ["woodland_forest", "open_water", "shrubland_savanna", "herbaceous_wetland", "grassland_barren",
         "agriculture_settlement", "montane_forest", "lava_forest", "urban_dense_development"]
BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
# The fields of the real AWF card (docs/method/taskcards.md) that the script reads; keys as JSON leaves them.
CARD = {"name": "awf", "kind": "project", "goal": "land use and land cover in southern Kenya",
        "encoder": {"model_id": "OLMOEARTH_V1_BASE", "patch_size": 4},
        "task": {"type": "segmentation (dense per-pixel classes)", "num_classes": 10, "nodata_value": 9,
                 "classes": {str(i): n for i, n in enumerate(NAMES)}},
        "inputs": {"sentinel2_l2a": {"layers": ["sentinel2"], "n_timesteps": 12, "bands": BANDS, "is_target": False}},
        "windows": {"window_resolution": 10.0, "window_size_px": 63}, "outputs": {"decoder_out_channels": [10]},
        "audit": {"n_classes": 10}, "sources": ["https://example.invalid/awf/dataset.json"], "warnings": []}
RECIPE = {"source": "https://example.invalid/awf/dataset.json", "max_matches": 12, "period_days": 30,
          **sa.EXPECTED_SOURCE, **sa.EXPECTED_QUERY}
MODEL_INFO = {"repo": sa.MODEL_REPO, "revision": "0" * 40, "device": "stub", "dtype": "float32"}

_rng = np.random.default_rng(0)
W_STUB = _rng.normal(0, 2e-3, (10, 12)).astype(np.float32)
B_STUB = _rng.normal(0, 1, 10).astype(np.float32)


def linear_stub(crops):
    """A per-pixel linear head on the temporal mean of the bands: translation-equivariant, so any correct tiling of
    the area gives exactly the map computed on the whole stack at once."""
    return np.einsum("nhwb,cb->nchw", crops.mean(axis=3), W_STUB) + B_STUB[None, :, None, None]


def _cli(*args):
    # the entry point of the Python running the tests first: it is the package under test, installed or editable
    # (the release check runs the suite against the wheel, where `uv run` has no project to run in)
    exe = os.path.join(os.path.dirname(sys.executable), "oe-inferencex")
    uv = shutil.which("uv")
    if os.path.exists(exe):
        cmd = [exe, *args]
    elif uv:
        cmd = [uv, "run", "--no-sync", "oe-inferencex", *args]
    else:
        pytest.skip("neither the oe-inferencex entry point nor uv is on this machine")
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)


# ----------------------------------------------------------------------------- windowing
def test_crop_starts_are_rslearns():
    # rslearn get_window_crop_options' own worked example: bounds [0, 15], crop 8, overlap 2 -> 0, 6, 7
    assert sa.crop_starts(15, 8, 2) == [0, 6, 7]
    assert sa.crop_starts(14, 8, 2) == [0, 6]
    assert sa.crop_starts(16) == [0]
    s = sa.crop_starts(512)
    assert s[0] == 0 and s[-1] == 512 - sa.CROP and all(b - a == 12 for a, b in zip(s[:-2], s[1:-1]))


def test_merge_keeps_each_pixel_from_the_crop_rslearn_would():
    """Band 0 holds the column index and band 1 the row; the stub answers with its crop's origin, so the merged map
    says which crop each pixel came from. For 40 px, crops at 0, 12, 24 and a 2 px trim on inner edges: crop 0 owns
    [0, 14), crop 12 owns [14, 26), crop 24 owns [26, 40)."""
    H = W = 40
    stack = np.zeros((H, W, 2, 2), np.float32)
    stack[..., 0] = np.arange(W)[None, :, None]
    stack[..., 1] = np.arange(H)[:, None, None]

    def origin_stub(crops):
        x0 = crops[..., 0].min(axis=(1, 2, 3))
        y0 = crops[..., 1].min(axis=(1, 2, 3))
        return np.broadcast_to(np.stack([x0, y0], 1)[:, :, None, None], (len(crops), 2, 16, 16)).copy()

    merged, n = sa.predict_area(stack, origin_stub, batch=4)
    owner = np.where(np.arange(40) < 14, 0, np.where(np.arange(40) < 26, 12, 24))
    assert n == 9
    assert np.array_equal(merged[0], np.broadcast_to(owner[None, :], (H, W)))
    assert np.array_equal(merged[1], np.broadcast_to(owner[:, None], (H, W)))


def test_tiling_reproduces_the_map_computed_at_once():
    stack = _rng.uniform(0, 5000, (50, 37, 12, 12)).astype(np.float32)
    merged, _ = sa.predict_area(stack, linear_stub, batch=5)
    direct = linear_stub(stack[None])[0]
    assert np.isfinite(merged).all()
    np.testing.assert_allclose(merged, direct, rtol=1e-5, atol=1e-4)


# ----------------------------------------------------------------------------- imagery
def test_periods_reproduce_the_awf_training_window():
    """The item dates of Ai2's training window task_20546e37-..._point_32 (items.json in dataset.tar, item group 0
    first) each fall in the period of the same index: most recent first, 30 days back from 2023-12-31."""
    dates = ["2023-12-09", "2023-11-29", "2023-10-20", "2023-09-25", "2023-08-26", "2023-07-27", "2023-06-07",
             "2023-05-23", "2023-04-18", "2023-03-09", "2023-02-07", "2023-01-28"]
    per = sa.periods(sa.parse_day("2023-01-01"), sa.parse_day("2023-12-31"), 12, 30)
    assert len(per) == 12
    assert per[0] == (dt.datetime(2023, 12, 1, tzinfo=UTC), dt.datetime(2023, 12, 31, tzinfo=UTC))
    assert per[-1][0] == dt.datetime(2023, 1, 5, tzinfo=UTC)
    for (ps, pe), d in zip(per, dates):
        t = dt.datetime.fromisoformat(d).replace(hour=7, tzinfo=UTC)
        assert ps < t < pe, (d, ps, pe)
    assert len(sa.periods(sa.parse_day("2023-06-01"), sa.parse_day("2023-12-31"), 12, 30)) == 7


def test_harmonize_is_rslearns_callback():
    raw = np.array([0, 1, 999, 1000, 1001, 5000], np.uint16)
    assert sa.harmonize(raw, 1000).tolist() == [0, 1, 1, 1, 1, 4000]      # a valid pixel never becomes no-data
    assert sa.harmonize(raw, 0).tolist() == raw.tolist()
    when = dt.datetime(2023, 3, 1, tzinfo=UTC)
    assert sa.harmonize_offset({"baseline": "05.10", "datetime": when})[0] == 1000
    assert sa.harmonize_offset({"baseline": "03.01", "datetime": when})[0] == 0
    assert sa.harmonize_offset({"baseline": None, "datetime": dt.datetime(2021, 6, 1, tzinfo=UTC)})[0] == 0
    assert sa.harmonize_offset({"baseline": None, "datetime": when})[0] == 1000


def _fake_imagery(grid, drop_period=None):
    """(search, read): per period one full-cover scene, except the most recent period, where a clearer scene covers
    the west half and a cloudier one the rest, a still clearer one lies outside the date window, and period 5,
    whose only scene covers the north half. The value of a scene is 1000 (the offset) plus a code."""
    box = sa.grid_box(grid)
    x0, y0, x1, y1 = box.bounds
    west = shapely.geometry.box(x0, y0, (x0 + x1) / 2, y1)
    north = shapely.geometry.box(x0, (y0 + y1) / 2, x1, y1)
    per = sa.periods(sa.parse_day("2023-01-01"), sa.parse_day("2023-12-31"), 12, 30)

    def item(name, shp, when, cloud, code):
        return {"id": name, "datetime": when, "cloud": cloud, "baseline": "05.10", "code": code,
                "geometry": shapely.geometry.mapping(sa.to_wgs84(shp, grid["crs"]))}

    items = [item("A_west_clear", west, per[0][0] + dt.timedelta(days=3), 5.0, 111),
             item("B_full_cloudy", box, per[0][0] + dt.timedelta(days=9), 10.0, 222),
             item("C_outside_window", box, dt.datetime(2022, 12, 20, tzinfo=UTC), 0.0, 999)]
    for t in range(1, 12):
        if t != drop_period:
            items.append(item(f"D{t}", north if t == 5 else box, per[t][0] + dt.timedelta(days=10), 1.0, 300 + t))
    by_id = {it["id"]: it for it in items}

    def search(aoi_wgs84, start, end):
        assert aoi_wgs84.intersects(sa.to_wgs84(box, grid["crs"]))
        return [dict(it) for it in items]

    def read(it, bands, g):
        cols, rows = np.meshgrid(np.arange(g["width"]), np.arange(g["height"]))
        xs, ys = rasterio.transform.xy(g["transform"], rows.ravel(), cols.ravel(), offset="center")
        xs, ys = np.reshape(xs, rows.shape), np.reshape(ys, rows.shape)
        inside = shapely.contains_xy(sa.from_wgs84(by_id[it["id"]]["geometry"], g["crs"]), xs, ys)
        return np.where(inside[None], np.uint16(1000 + by_id[it["id"]]["code"]), np.uint16(0)).repeat(len(bands), 0)

    return search, read


def test_fetch_plans_mosaics_fills_and_marks_missing_periods():
    grid = sa.area_grid(-2.55, 36.81, 32)
    search, read = _fake_imagery(grid)
    stack, valid, imagery = sa.fetch_stack(grid, sa.parse_day("2023-01-01"), sa.parse_day("2023-12-31"), 12, 30,
                                           BANDS, search=search, read=read, log=lambda *_: None)
    assert stack.shape == (32, 32, 12, 12) and stack.dtype == np.float32
    assert [r["id"] for r in imagery[0]["items"]] == ["A_west_clear", "B_full_cloudy"]     # C is outside the window
    assert (stack[:, :16, 0, :] == 111).all() and (stack[:, 16:, 0, :] == 222).all()       # harmonized, B fills only
    assert all(r["harmonize_offset"] == 1000 for p in imagery for r in p["items"])
    assert [p["t"] for p in imagery] == list(range(12))
    assert imagery[0]["period"][0] > imagery[1]["period"][0]                              # most recent first
    assert (stack[:, :, 3, 0] == 303).all()
    assert valid[:16].all() and not valid[16:].any()                                       # period 5 covers the north
    search, read = _fake_imagery(grid, drop_period=7)
    with pytest.raises(SystemExit, match="period 7"):
        sa.fetch_stack(grid, sa.parse_day("2023-01-01"), sa.parse_day("2023-12-31"), 12, 30, BANDS,
                       search=search, read=read, log=lambda *_: None)


def test_area_grid_is_utm_on_the_10m_lattice():
    g = sa.area_grid(-2.55, 36.81, 512)
    assert g["crs"] == "EPSG:32737" and g["width"] == g["height"] == 512
    t = g["transform"]
    assert t.a == 10 and t.e == -10 and t.c % 10 == 0 and t.f % 10 == 0
    lon0, lat0, lon1, lat1 = sa.to_wgs84(sa.grid_box(g), g["crs"]).bounds
    assert lon0 < 36.81 < lon1 and lat0 < -2.55 < lat1
    assert sa.utm_epsg(51.5, -0.1) == 32630
    with pytest.raises(SystemExit):
        sa.area_grid(-2.55, 36.81, 8)


def test_card_facts_name_every_channel_or_refuse():
    names, bands, n_t = sa.card_facts(CARD)
    assert names == NAMES + ["nodata_value_9"] and bands == BANDS and n_t == 12
    bad = json.loads(json.dumps(CARD))
    bad["task"]["nodata_value"] = None
    with pytest.raises(SystemExit, match="channel 9"):
        sa.card_facts(bad)


# ----------------------------------------------------------------------------- files, and the package reading them
def _scene(size=64):
    grid = sa.area_grid(-2.55, 36.81, size)
    stack = _rng.uniform(100, 4000, (size, size, 12, 12)).astype(np.float32)
    valid = np.ones((size, size), bool)
    valid[:8, :8] = False
    imagery = [{"t": t, "period": ["p", "q"], "items": [{"id": f"S2_{t}"}], "valid_fraction": 1.0} for t in range(12)]
    request = {"lat": -2.55, "lon": 36.81, "size_px": size, "start": "2023-01-01", "end": "2023-12-31"}
    return grid, stack, valid, imagery, request


def test_logits_raster_carries_the_grid_and_round_trips_through_assess_and_sample(tmp_path):
    grid, stack, valid, imagery, request = _scene()
    out = tmp_path / "scores"
    m = sa.produce(str(out), grid, stack, valid, imagery, linear_stub, MODEL_INFO, CARD, RECIPE, request,
                   log=lambda *_: None)
    path = out / "scores.tif"
    with rasterio.open(path) as src:
        a = src.read()
        assert src.count == 10 and src.dtypes[0] == "float32"
        assert src.crs.to_string() == grid["crs"] and src.transform == grid["transform"]
        assert np.isnan(src.nodata)
        assert list(src.descriptions) == NAMES + ["nodata_value_9"]
        assert src.tags()["values"] == "logits"
    assert np.isnan(a[:, :8, :8]).all() and np.isfinite(a[:, valid]).all()
    np.testing.assert_allclose(a[:, valid], linear_stub(stack[None])[0][:, valid], rtol=1e-5, atol=1e-4)

    man = json.loads((out / "manifest.json").read_text())
    for key in ("scores", "classes", "model", "task_card", "area", "date_window", "imagery_recipe", "imagery",
                "inference", "software", "next", "caveats"):
        assert man[key], key
    assert man["scores"]["values"] == "logits" and man["scores"]["sha256"] == sa.sha256(path)
    assert man["scores"]["n_nodata_pixels"] == 64
    assert man["classes"]["1"] == "open_water" and "never a training target" in man["class_note"]
    assert man["model"]["repo"] == sa.MODEL_REPO and man["model"]["revision"] == "0" * 40
    assert man["area"]["crs"] == "EPSG:32737" and man["area"]["transform"][:3] == [10.0, 0.0, grid["transform"].c]
    assert man["date_window"]["start"] == "2023-01-01" and len(man["imagery"]) == 12
    assert man["inference"]["crop_px"] == 16 and man["inference"]["overlap_px"] == 4
    assert man["software"]["numpy"] and "--logits" in man["next"]["assess"]
    assert sum(man["argmax_pixel_counts"].values()) == int(valid.sum())
    assert m == man

    r = _cli("assess", str(path), "--logits", "--out", str(out / "assess"))
    assert r.returncode == 0, r.stderr
    s = json.loads((out / "assess" / "assessment.json").read_text())
    assert s["n_windows"] == 16 * 16 - 4                     # 4-px windows; the no-data corner holds four
    x0, y0, x1, y1 = man["area"]["bounds"]
    with open(out / "assess" / "review_set_05pct.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(x0 < float(w["x"]) < x1 and y0 < float(w["y"]) < y1 for w in rows)
    r = _cli("sample", str(path), "--logits", "--budget", "20", "--out", str(out / "sample" / "sample.csv"))
    assert r.returncode == 0, r.stderr
    assert (out / "sample" / "sample.csv").exists()


def test_probabilities_raster_is_a_distribution_assess_takes_as_is(tmp_path):
    grid, stack, valid, imagery, request = _scene()
    out = tmp_path / "probs"
    sa.produce(str(out), grid, stack, valid, imagery, linear_stub, MODEL_INFO, CARD, RECIPE, request,
               probabilities=True, log=lambda *_: None)
    with rasterio.open(out / "scores.tif") as src:
        p = src.read()
        assert src.tags()["values"] == "probabilities"
    v = p[:, valid]
    assert (v >= 0).all() and (v <= 1).all()
    np.testing.assert_allclose(v.sum(0), 1.0, atol=1e-5)
    man = json.loads((out / "manifest.json").read_text())
    assert man["scores"]["values"] == "probabilities" and "--logits" not in man["next"]["assess"]
    r = _cli("assess", str(out / "scores.tif"), "--out", str(out / "assess"))
    assert r.returncode == 0, r.stderr


def test_produce_refuses_a_model_that_disagrees_with_the_card(tmp_path):
    grid, stack, valid, imagery, request = _scene(32)
    with pytest.raises(SystemExit, match="channels"):
        sa.produce(str(tmp_path), grid, stack, valid, imagery, lambda c: linear_stub(c)[:, :9], MODEL_INFO, CARD,
                   RECIPE, request, log=lambda *_: None)
    with pytest.raises(SystemExit, match="task card"):
        sa.produce(str(tmp_path), grid, stack[:, :, :6], valid, imagery, linear_stub, MODEL_INFO, CARD, RECIPE,
                   request, log=lambda *_: None)


def test_main_end_to_end_with_the_network_and_the_model_stubbed(tmp_path, monkeypatch):
    """The glue the cluster job runs: card, recipe, grid, imagery, model, files, in that order."""
    monkeypatch.setattr(sa, "load_task_card", lambda project="awf": json.loads(json.dumps(CARD)))
    monkeypatch.setattr(sa, "load_query_config", lambda card: dict(RECIPE))
    grid = sa.area_grid(-2.55, 36.81, 48)
    search, read = _fake_imagery(grid)
    monkeypatch.setattr(sa, "search_items", search)
    monkeypatch.setattr(sa, "read_item", read)
    monkeypatch.setattr(sa, "build_awf_model", lambda bands, n_t: (linear_stub, MODEL_INFO))
    out = tmp_path / "run"
    assert sa.main(["--out", str(out), "--size", "48"]) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["area"]["request"]["size_px"] == 48 and man["area"]["width"] == 48
    assert [r["id"] for r in man["imagery"][0]["items"]] == ["A_west_clear", "B_full_cloudy"]
    assert man["scores"]["n_nodata_pixels"] == 48 * 24                   # period 5 covers the north half only
    with rasterio.open(out / "scores.tif") as src:
        assert src.transform == grid["transform"]
