"""exp98's burned-area stage (exp/exp98_burned.py) and its Part H (exp/exp98_awf_deployment.py), offline.

MODIS tiles are synthetic GeoTIFFs on the MODIS sinusoidal lattice (the sphere of radius 6371007.181 m, cells of
463.3 m), placed at a synthetic spot far from the AWF area where the sinusoidal grid is strongly sheared against UTM,
so that a misplaced nearest-neighbour lookup would show. The score grid is a synthetic from-olmoearth part. Checked:
the item selection on STAC-shaped dicts, the code mapping cell by cell (burned, unburned, water, unmapped, no tile,
special conditions, a burn date outside its month) and its refusals, the warp against an independent per-pixel
transform of every pixel centre, the command end to end (rasters, names, counts, the refusals), and Part H's pooling
and tables on hand-checked cases. No test touches the network. Needs rasterio (the geo extra); skipped without it."""
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("rasterio")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp98_burned as eb             # noqa: E402
import exp98_awf_deployment as e98    # noqa: E402

SINU = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
CELL = 1111950.5196666666 / 2400                          # MODIS's 500 m cell, 463.3127 m
UL_X, UL_Y = -20015109.355798, 10007554.677899            # the MODIS sinusoidal grid's corner
UTM = "EPSG:32641"
X0, Y0 = 400000.0, 4500000.0                              # synthetic, far from the AWF area; never written
H, W, NAN_ROWS = 180, 240, 60                             # the score grid, 10 m; its last 60 rows uncovered

QA_LAND = 3                                               # land, valid data
QA_WATER = 0
QA_UNMAPPED = 1                                           # land, no valid data


# ----------------------------------------------------------------------------- synthetic inputs
def _transform_pts(src, dst, xs, ys):
    from rasterio.warp import transform
    a, b = transform(src, dst, list(map(float, xs)), list(map(float, ys)))
    return np.asarray(a), np.asarray(b)


def _write(path, arr, crs, transform, dtype, nodata=None):
    import rasterio
    arr = np.asarray(arr)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[-2], width=arr.shape[-1],
                       count=1 if arr.ndim == 2 else arr.shape[0], dtype=dtype, crs=crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(arr if arr.ndim == 3 else arr[None])


def _score_grid(d):
    """A from-olmoearth part: scores_32641.tif, ten bands summing to 1, the last NAN_ROWS rows NaN."""
    from rasterio.transform import from_origin
    os.makedirs(d, exist_ok=True)
    s = np.full((10, H, W), 0.1, np.float32)
    s[:, H - NAN_ROWS:] = np.nan
    path = os.path.join(d, "scores_32641.tif")
    _write(path, s, UTM, from_origin(X0, Y0, 10, 10), "float32", float("nan"))
    return path


def _lattice_box():
    """(row0, col0, nrows, ncols) of MODIS lattice cells that hold the score grid with a margin of three."""
    from rasterio.warp import transform_bounds
    left, bottom, right, top = transform_bounds(UTM, SINU, X0, Y0 - 10 * H, X0 + 10 * W, Y0, densify_pts=21)
    c0, c1 = int(np.floor((left - UL_X) / CELL)) - 3, int(np.ceil((right - UL_X) / CELL)) + 3
    r0, r1 = int(np.floor((UL_Y - top) / CELL)) - 3, int(np.ceil((UL_Y - bottom) / CELL)) + 3
    return r0, c0, r1 - r0, c1 - c0


def _cell_of(px_rows, px_cols, box):
    """Independent of the stage: the lattice cell (row, col within box) that holds each 10 m pixel's centre."""
    r0, c0, _, _ = box
    xs = X0 + 10 * (np.asarray(px_cols) + 0.5)
    ys = Y0 - 10 * (np.asarray(px_rows) + 0.5)
    sx, sy = _transform_pts(UTM, SINU, xs, ys)
    return (np.floor((UL_Y - sy) / CELL).astype(int) - r0, np.floor((sx - UL_X) / CELL).astype(int) - c0)


def _design(box):
    """Per lattice cell of `box`, the year's design: left half burned on day 70 (March), right half unburned, one cell
    water, one unmapped in June (and otherwise unburned), one unmapped in June but burned on day 220 (August), one
    unburned with a special-condition code; the cells are those holding chosen pixels of the grid."""
    _, _, nr, nc = box
    split = nc // 2
    rr, cc = np.meshgrid(np.arange(H - NAN_ROWS), np.arange(W), indexing="ij")     # the covered pixels
    cr, ccol = _cell_of(rr.ravel(), cc.ravel(), box)
    right = [(int(r), int(c)) for r, c in zip(cr, ccol) if c >= split]
    cand, n = np.unique(np.array(right), axis=0, return_counts=True)
    pick = [tuple(int(v) for v in cand[i]) for i in np.argsort(-n, kind="stable")[:4]]   # the four fullest cells
    assert len(pick) == 4 and np.sort(n)[-4] > 50, "four cells of the unburned half hold covered pixels"
    assert any(c < split for c in ccol), "the burned half holds covered pixels too"
    cells = dict(zip(("water", "unmapped", "unmapped_then_burned", "special"), pick))
    months = {}
    for m in range(1, 13):
        b = np.zeros((nr, nc), np.int16)
        q = np.full((nr, nc), QA_LAND, np.uint8)
        if m == 3:
            b[:, :split] = 70
        r, c = cells["water"]
        b[r, c], q[r, c] = eb.WATER, QA_WATER
        if m == 6:
            for k in ("unmapped", "unmapped_then_burned"):
                r, c = cells[k]
                b[r, c], q[r, c] = eb.UNMAPPED, QA_UNMAPPED
        if m == 8:
            r, c = cells["unmapped_then_burned"]
            b[r, c] = 220
        r, c = cells["special"]
        q[r, c] = QA_LAND | (2 << 5)
        months[m] = (b, q)
    return months, cells, split


def _write_tiles(d, box, months, skip=(9,)):
    """Two tiles side by side on the lattice per month (the box split at a column), Burn_Date and QA each; the months
    in `skip` are not written. Returns --item triples."""
    from rasterio.transform import from_origin
    os.makedirs(d, exist_ok=True)
    r0, c0, nr, nc = box
    cut = nc // 3                                         # the tile edge, away from the design's own split
    items = []
    for m, (b, q) in months.items():
        if m in skip:
            continue
        for t, (a, z) in enumerate(((0, cut), (cut, nc))):
            tr = from_origin(UL_X + (c0 + a) * CELL, UL_Y - r0 * CELL, CELL, CELL)
            bp, qp = os.path.join(d, f"b_{m:02d}_{t}.tif"), os.path.join(d, f"q_{m:02d}_{t}.tif")
            _write(bp, b[:, a:z], SINU, tr, "int16")
            _write(qp, q[:, a:z], SINU, tr, "uint8")
            items.append((f"2023-{m:02d}", bp, qp))
    return items


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("burn")
    part = str(root / "scores" / "r00_p00")
    spath = _score_grid(part)
    box = _lattice_box()
    months, cells, split = _design(box)
    items = _write_tiles(str(root / "tiles"), box, months)
    out_json = str(root / "out" / "e98_burned.json")
    rc = eb.main(["--scores", str(root / "scores"), "--out-json", out_json, "--no-stac", "--allow-missing", "2023-09",
                  "--threads", "1"] + [x for it in items for x in ["--item", *it]])
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cr, ccol = _cell_of(rr.ravel(), cc.ravel(), box)
    cr, ccol = cr.reshape(H, W), ccol.reshape(H, W)
    expect = np.where(ccol < split, 1, 0)
    for k in ("water", "unmapped"):
        expect[(cr == cells[k][0]) & (ccol == cells[k][1])] = -1
    expect[(cr == cells["unmapped_then_burned"][0]) & (ccol == cells["unmapped_then_burned"][1])] = 1
    return {"root": root, "part": part, "scores": spath, "box": box, "items": items, "out_json": out_json, "rc": rc,
            "expect": expect, "cells": (cr, ccol), "cell_ids": cells, "split": split}


# ----------------------------------------------------------------------------- the items
def _feature(month, prod="2024000000001", tile="h10v05", year=2023, doy=None, assets=("Burn_Date", "QA")):
    doy = doy if doy is not None else eb.month_doys(year, month)[0]
    return {"id": f"MCD64A1.A{year}{doy:03d}.{tile}.061.{prod}",
            "properties": {"start_datetime": f"{year}-{month:02d}-01T00:00:00Z"},
            "assets": {a: {"href": f"https://example.invalid/{a}_{month}_{prod}.tif"} for a in assets}}


def test_items_are_selected_by_month_with_the_later_production_kept():
    feats = [_feature(m) for m in range(1, 13) if m != 9]
    feats += [_feature(3, prod="2025000000000"),                       # a reprocessed March: kept
              _feature(4, prod="2020000000000"),                       # an older April: dropped
              _feature(5, tile="h11v05"),                              # a second tile in May: kept
              _feature(6, year=2022),                                  # another year
              _feature(7, doy=200),                                    # day of year not its month's first
              _feature(8, assets=("Burn_Date",)),                      # no QA
              {"id": "something else", "properties": {}, "assets": {}}]
    by, notes = eb.select_items(feats)
    assert sorted(by) == [f"2023-{m:02d}" for m in range(1, 13) if m != 9]
    assert by["2023-03"][0]["production"] == "2025000000000" and len(by["2023-03"]) == 1
    assert by["2023-04"][0]["production"] == "2024000000001"
    assert len(by["2023-05"]) == 2 and len(by["2023-08"]) == 1
    assert notes["duplicates_dropped"] == 2 and notes["items_kept"] == 12
    assert notes["skipped"] == {"another year": 1, "day of year in the id is not its month's first day": 1,
                                "Burn_Date or QA asset missing": 1, "id not MCD64A1 v061": 1}
    assert [eb.month_doys(2023, m)[0] for m in (1, 2, 3, 9, 12)] == [1, 32, 60, 244, 335]   # the guide's Table 2


def test_names_say_which_months_were_not_read():
    assert eb.code_names(2023, []) == {0: "mcd64a1:no-burn-2023", 1: "mcd64a1:burned-2023"}
    names = eb.code_names(2023, ["2023-09"])
    assert names[0] == "mcd64a1:no-burn-2023:sep-not-read"
    assert eb.condition_names_arg(names) == ("--condition-names 0=mcd64a1:no-burn-2023:sep-not-read "
                                             "1=mcd64a1:burned-2023")
    assert eb.code_names(2023, ["2023-09", "2023-10"])[0].endswith(":sep+oct-not-read")


# ----------------------------------------------------------------------------- the code mapping
def _months(cols):
    """12 monthly (Burn_Date, QA) rows of one cell each, from {month: (burn, qa)} per column."""
    burns, qas = [], []
    for m in range(1, 13):
        burns.append(np.array([[c.get(m, (0, QA_LAND))[0] for c in cols]], np.int16))
        qas.append(np.array([[c.get(m, (0, QA_LAND))[1] for c in cols]], np.uint8))
    return burns, qas, [eb.month_doys(2023, m) for m in range(1, 13)]


def test_codes_cell_by_cell():
    F = eb.FILL
    cols = [{},                                                         # 0: unburned all year
            {3: (70, QA_LAND)},                                         # 1: burned in March
            {m: (eb.WATER, QA_WATER) for m in range(1, 13)},           # 2: water
            {6: (eb.UNMAPPED, QA_UNMAPPED)},                            # 3: unmapped in June, else unburned
            {6: (eb.UNMAPPED, QA_UNMAPPED), 8: (220, QA_LAND)},         # 4: unmapped in June, burned in August
            {5: (F, 0)},                                                # 5: no tile in May
            {m: (0, QA_LAND | (2 << 5)) for m in range(1, 13)},        # 6: unburned, special condition 2
            {2: (40, QA_LAND), 10: (280, QA_LAND)},                     # 7: burned twice
            {1: (100, QA_LAND)},                                        # 8: a burn date outside January
            {m: (F, 0) for m in range(1, 13)},                          # 9: no tile in any month
            {4: (0, QA_LAND | 4)}]                                      # 10: shortened mapping period in April
    burns, qas, ranges = _months(cols)
    code, bits, flags, ck = eb.combine(burns, qas, ranges)
    assert code[0].tolist() == [0, 1, -1, -1, 1, -1, 0, 1, 1, F, 0]
    assert bits[0, 1] == 1 << 2 and bits[0, 4] == 1 << 7 and bits[0, 7] == (1 << 1) | (1 << 9)
    assert flags[0, 2] & eb.FLAG_WATER and flags[0, 3] & eb.FLAG_UNMAPPED and flags[0, 5] & eb.FLAG_NOTILE
    assert flags[0, 6] & eb.FLAG_SPECIAL and not flags[0, 0] and flags[0, 8] & eb.FLAG_OFF_MONTH
    assert flags[0, 10] & eb.FLAG_SHORT
    assert ck["burn_dates_outside_month"] == 1 and ck["qa_agrees"] is True
    assert ck["water_qa_disagree"] == 0 and ck["unmapped_qa_disagree"] == 0
    assert ck["cells_read"] == 12 * 11 - 1 - 12                 # cell 5 misses May, cell 9 every month


def test_codes_refuse_an_unknown_value_and_a_disagreeing_qa():
    burns, qas, ranges = _months([{3: (400, QA_LAND)}])
    with pytest.raises(ValueError, match="outside -2..366"):
        eb.combine(burns, qas, ranges)
    burns, qas, ranges = _months([{m: (eb.WATER, QA_LAND) for m in range(1, 13)}])     # water, but QA says land
    with pytest.raises(ValueError, match="QA disagrees"):
        eb.combine(burns, qas, ranges)
    burns, qas, ranges = _months([{6: (eb.UNMAPPED, QA_LAND)}])                         # unmapped, QA says valid
    with pytest.raises(ValueError, match="QA disagrees"):
        eb.combine(burns, qas, ranges)


# ----------------------------------------------------------------------------- the warp
def test_each_pixel_takes_the_cell_that_holds_its_centre():
    from rasterio.transform import from_origin
    box = _lattice_box()
    r0, c0, nr, nc = box
    assert nr * nc < 32767
    ids = np.arange(nr * nc, dtype=np.int16).reshape(nr, nc)
    src_tr = from_origin(UL_X + c0 * CELL, UL_Y - r0 * CELL, CELL, CELL)
    out = eb.to_grid(ids[None], src_tr, SINU, from_origin(X0, Y0, 10, 10), UTM, (H, W), threads=1)[0]
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cr, ccol = _cell_of(rr.ravel(), cc.ravel(), box)
    assert np.array_equal(out.ravel(), ids[cr, ccol]), "every pixel, exactly"
    assert len(np.unique(out)) >= 8, "the grid spans several sheared cells"


# ----------------------------------------------------------------------------- the command
def test_the_command_writes_the_layer_on_the_score_grid(synth):
    import rasterio
    assert synth["rc"] == 0
    bpath = os.path.join(synth["part"], "burned_32641.tif")
    with rasterio.open(bpath) as b, rasterio.open(synth["scores"]) as s:
        assert (b.height, b.width, b.crs, b.transform) == (s.height, s.width, s.crs, s.transform)
        assert b.dtypes[0] == "int32" and b.nodata == -1 and b.count == 1
        code = b.read(1)
    assert np.array_equal(code, synth["expect"]), "the codes, pixel by pixel, against an independent lookup"
    assert set(np.unique(code).tolist()) == {-1, 0, 1}


def test_the_names_record(synth):
    with open(os.path.join(synth["part"], eb.NAMES_JSON)) as f:
        n = json.load(f)
    assert n["codes"] == {"0": "mcd64a1:no-burn-2023:sep-not-read", "1": "mcd64a1:burned-2023"}
    assert n["condition_names"] == "--condition-names 0=mcd64a1:no-burn-2023:sep-not-read 1=mcd64a1:burned-2023"
    assert n["source"]["months_missing"] == ["2023-09"] and n["source"]["collection"] == "modis-64A1-061"
    assert len(n["source"]["months_from_files"]) == 11
    assert "--condition burned_32641.tif" in n["grids"]["32641"]["assess"]
    assert "463 m" in n["rule"] and "No code 2" in n["code_2"]


def test_the_counts_record(synth):
    with open(synth["out_json"]) as f:
        r = json.load(f)
    (e,) = r["parts"]
    cov = np.zeros((H, W), bool)
    cov[:H - NAN_ROWS] = True
    x = synth["expect"][cov]
    assert e["part"] == "r00_p00" and e["grid"] == "32641" and e["pixels"] == H * W
    assert e["pixels_covered"] == int(cov.sum())
    assert e["pixels_covered_by_code"] == {str(k): int((x == k).sum()) for k in (-1, 0, 1)}
    assert e["share_of_covered_by_code"]["1"] == pytest.approx(float((x == 1).mean()))
    cr, cc = synth["cells"]
    in_left = (cc < synth["split"])[cov]
    assert e["burned_share_by_month"]["2023-03"] == pytest.approx(float(in_left.mean()))
    aug = synth["cell_ids"]["unmapped_then_burned"]
    assert e["burned_share_by_month"]["2023-08"] == pytest.approx(
        float(((cr == aug[0]) & (cc == aug[1]))[cov].mean()))
    assert e["burned_share_by_month"]["2023-09"] is None and e["months_missing"] == ["2023-09"]
    sp = synth["cell_ids"]["special"]
    in_sp = ((cr == sp[0]) & (cc == sp[1]))[cov]
    assert in_sp.any()
    assert e["not_burned_with_special_condition_share"] == pytest.approx(float(in_sp[x == 0].mean()))
    why = e["unrecorded_why"]
    assert why["water_in_a_month_read"] + why["unmapped_in_a_month_read"] == int((x == -1).sum())
    assert why["no_tile_in_a_month_read"] == 0
    assert e["qa_check"]["qa_agrees"] is True and e["qa_check"]["tiles_read"] == 2
    assert r["names"]["0"].endswith("sep-not-read") and r["source"]["months_missing"] == ["2023-09"]
    assert r["totals"]["share_of_covered_by_code"] == e["share_of_covered_by_code"]


def test_nothing_written_carries_a_position(synth):
    r0, c0, nr, nc = synth["box"]
    origins = {X0, Y0, UL_X + c0 * CELL, UL_Y - r0 * CELL}
    for path in (synth["out_json"], os.path.join(synth["part"], eb.NAMES_JSON)):
        with open(path) as f:
            text = f.read()
        obj = json.loads(text)
        eb.check_no_coordinates(obj)
        assert "example.invalid" not in text and str(synth["root"]) not in text, "no href and no local path"

        def numbers(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from numbers(v)
            elif isinstance(o, list):
                for v in o:
                    yield from numbers(v)
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                yield float(o)
        assert not any(abs(v - o) < 1 for v in numbers(obj) for o in origins)
    with pytest.raises(ValueError, match="position"):
        eb.check_no_coordinates({"parts": [{"bbox": [1, 2, 3, 4]}]})


def test_the_command_refuses(synth, tmp_path):
    items = synth["items"]
    with pytest.raises(FileExistsError, match="--redo"):
        eb.run([str(synth["root"] / "scores")], str(tmp_path / "x.json"), ["2023-09"], items, stac=False)
    d = str(tmp_path / "scores" / "r00_p00")
    _score_grid(d)
    with pytest.raises(ValueError, match=r"no MCD64A1 item for \['2023-09'\]"):
        eb.run([d], str(tmp_path / "x.json"), [], items, stac=False)
    with pytest.raises(ValueError, match="YYYY-MM"):
        eb.run([d], str(tmp_path / "x.json"), ["2023-9"], items, stac=False)
    assert not os.path.exists(os.path.join(d, "burned_32641.tif")) and not os.path.exists(tmp_path / "x.json")


# ----------------------------------------------------------------------------- part H on hand-checked cases
def test_windows_take_the_majority_code_of_their_covered_pixels():
    patch = 4
    u = np.zeros((4, 12))
    u[:, 0:4], u[:, 4:8], u[:, 8:12] = 1.0, 2.0, 3.0
    cov = np.ones((4, 12), bool)
    cov[:, 6:8] = False                                   # window 1: 8 of 16 covered, still valid
    cov[:, 8:10] = False
    cov[0, 10] = False                                    # window 2: 7 of 16 covered, not valid
    burned = np.zeros((4, 12), np.int32)
    burned[:, 0:4] = 1
    burned[3, 0:4] = 0
    burned[0:2, 0:2] = -1                                 # window 0: 8 votes for 1, 4 for 0, 4 unrecorded -> 1
    burned[0:2, 4:6], burned[2:4, 4:6] = 1, 0             # window 1: its 8 covered pixels split 4 to 4 -> -1
    dup = np.zeros((4, 12), bool)
    s, c, nd = e98.burn_windows(u, cov, dup, burned, patch)
    assert s.tolist() == [1.0, 2.0] and c.tolist() == [1, -1] and nd == 0
    dup[0, 4] = True                                      # a pixel of window 1 was covered in an earlier grid
    cov2 = cov & ~dup
    s, c, nd = e98.burn_windows(u, cov2, dup, burned, patch)
    assert s.tolist() == [1.0] and c.tolist() == [1] and nd == 1
    s, c, nd = e98.burn_windows(u, cov, np.zeros_like(dup), np.full((4, 12), -1, np.int32), patch)
    assert c.tolist() == [-1, -1], "no window with a code: every window unrecorded, not a refusal"


def test_part_h_map_table_by_hand():
    s = np.array([5, 4, 3, 3, 3, 2, 1, 1, 0, 0], float)
    c = np.array([1, 1, 0, 1, -1, 0, 0, 0, 0, 1])
    t = e98.burned_map_block(s, c, {0: "no", 1: "yes"}, budget=0.3)
    assert t["n_least_confident"] == 3 and t["n_tied_at_cutoff"] == 3 and t["n_tied_taken"] == 1
    pc = t["per_code"]
    assert pc["yes"]["share_of_least_confident"] == pytest.approx(7 / 9)       # 2 above the cut, 1/3 of a tie
    assert pc["no"]["share_of_least_confident"] == pytest.approx(1 / 9)
    assert pc["unrecorded"]["share_of_least_confident"] == pytest.approx(1 / 9)
    assert pc["yes"]["least_confident_over_all_windows"] == pytest.approx((7 / 9) / 0.4)
    assert (pc["no"]["n_windows"], pc["no"]["share_of_windows"]) == (5, 0.5)
    q = pc["no"]["confidence_quartiles"]                  # code 0's suspicions 3, 2, 1, 1, 0
    assert q == pytest.approx({"0.25": np.exp(-2), "0.5": np.exp(-1), "0.75": np.exp(-1)})
    t = e98.burned_map_block(s, c, {0: "no", 1: "yes"}, budget=0.2)
    assert t["n_tied_at_cutoff"] == 1 and t["per_code"]["yes"]["share_of_least_confident"] == 1.0
    assert e98.burned_map_block(np.zeros(0), np.zeros(0, int), {})["n_windows"] == 0


def test_part_h_points_table_by_hand():
    t = e98.burned_points_block([1, 1, 0, -1, 0], [1, 0, 0, 1, 0], ["a", "b", "a", "c", "c"], {0: "no", 1: "yes"},
                                {-1: 10, 0: 60, 1: 30})
    assert t["yes"] == {"code": 1, "n_points": 2, "n_errors": 1, "error_rate": 0.5, "share_of_points": 0.4,
                        "share_of_map": 0.3, "n_tasks": 2}
    assert t["no"]["error_rate"] == 0.0 and t["no"]["n_tasks"] == 2
    assert t["unrecorded"]["n_errors"] == 1 and t["unrecorded"]["share_of_map"] == 0.1


def test_part_h_is_skipped_with_a_note_without_its_layers():
    ok = {"codes": {"0": "a", "1": "b"}}
    grids = [{"label": "0", "burned": "x.tif"}, {"label": "1", "burned": "y.tif"}]
    assert e98.burned_inputs(grids, [{"burned_conditions": ok}] * 2) == ({0: "a", 1: "b"}, None)
    names, note = e98.burned_inputs([grids[0], {"label": "1", "burned": None}], [{"burned_conditions": ok}] * 2)
    assert names is None and "1 of 2 grids" in note
    names, note = e98.burned_inputs(grids, [{"burned_conditions": ok}, {"burned_conditions": {"codes": {"0": "z"}}}])
    assert names is None and "differently" in note
    blk = e98.burned_block(None, "absent", [], np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0), [])
    assert blk["note"] == "absent" and "windows" not in blk and blk["status"].startswith("report-only")
