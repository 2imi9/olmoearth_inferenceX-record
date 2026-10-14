"""exp103's job helpers (exp/jobs/e103.sh; docs/plan/v12_regrade.md is the specification). They do the bookkeeping
around `rslearn model predict` (rslearn 0.1.17, exp102's venv) and `oe-inferencex from-olmoearth` on exp98's and exp99's
deployment datasets on the cluster's scratch; nothing here grades.

    count         windows, windows with input, with twelve item groups, with a layer (counts only)
    root-path     the real path of the i-th dataset root (for the job's environment; never echoed)
    remove-layer  remove output_v12 or output_v12_smoke (no other name) from every window of the dataset
    precheck      rslearn 0.1.17 parses each root's config.json and loads its windows; how many windows its own
                  check_window keeps for model.yaml's inputs (the windows `model predict` will write)
    smoke-config  model.yaml with predict_config restricted to two windows of one group (groups + names)
    check         the written output layer, as e98_predict.sh checks it (counts only)
    read          a view of the predicted windows (symlinks: metadata.json, the output layer and the collected SCL
                  layers), then `oe-inferencex from-olmoearth --layer L` in parts: e98_read.sh's grid plan or
                  e99_read.sh's one window per part; with --burned-from, exp98's burned layer beside identical parts
    verify-breadth  the TimeSync sample and Natural Earth's countries against their pins (no download)
    mask          stdin to stdout with the dataset's root, group and window names and long numbers masked

Nothing printed or recorded holds a window name, a group name, a real root name, a bound or a coordinate: the 2017
windows each hold a TimeSync plot. Window ids go only to files on scratch (the smoke's id list, parts.json).
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
P = "_".join(f"p{i}" for i in range(10))          # rslearn's band-set directory of p0..p9 (raster_format.py:32-47)
N_GROUPS = 12
INPUT = "sentinel2"
SCL = "sentinel2_scl"
REMOVABLE = ("output_v12", "output_v12_smoke")
# the deployment datasets as exp98's and exp99's records hold them (exp/out/exp98/e98_prepare_inventory.json,
# exp/out/exp99/e99_prepare_inventory.json): a different count means a purge or another dataset, so everything refuses
EXPECTED = {"23": {"windows": 270, "with_input": 270, "with_12_groups": 270},
            "17": {"windows": 313, "with_input": 271, "with_12_groups": 271}}
BYTES_PER_PX = 4 * 10 + 4                         # the reader's count for 10 float32 bands and an int32 owner


# ----------------------------------------------------------------------------- the datasets
def load_roots(inv_path, run_dir, year):
    """The inventory and [{root_id, path}] in the inventory's order (exp98's and exp99's read jobs number roots so)."""
    with open(inv_path) as f:
        inv = json.load(f)
    if inv.get("pilot", False):
        raise SystemExit(f"the {year} inventory is a pilot's")
    want = "exp99" if year == "17" else None
    if inv.get("experiment") != want:
        raise SystemExit(f"the {year} inventory says experiment {inv.get('experiment')!r}, expected {want!r}")
    roots = []
    for r in inv["roots"]:
        path = os.path.join(run_dir, r["root"])
        if not os.path.isdir(os.path.join(path, "windows")):
            raise SystemExit(f"{r['root_id']} of the {year} run holds no windows/ (the 30-day scratch purge?)")
        roots.append({"root_id": r["root_id"], "path": path})
    return inv, roots


def list_windows(root):
    """[(group, name, path)] sorted by group then name, hidden entries left out (as glob windows/*/* does)."""
    wd = os.path.join(root, "windows")
    out = []
    for g in sorted(os.listdir(wd)):
        gd = os.path.join(wd, g)
        if g.startswith(".") or not os.path.isdir(gd):
            continue
        for n in sorted(os.listdir(gd)):
            p = os.path.join(gd, n)
            if not n.startswith(".") and os.path.isdir(p):
                out.append((g, n, p))
    return out


def groups_completed(path):
    """How many of the twelve item groups of the input layer are marked completed (layers/sentinel2[.k]/completed)."""
    return sum(os.path.exists(os.path.join(path, "layers", INPUT if k == 0 else f"{INPUT}.{k}", "completed"))
               for k in range(N_GROUPS))


def has_input(path):
    """rslearn 0.1.17's test for an input read with load_all_item_groups: item group 0 completed
    (rslearn/train/dataset.py:865-879)."""
    return os.path.exists(os.path.join(path, "layers", INPUT, "completed"))


def layer_done(path, layer):
    return os.path.exists(os.path.join(path, "layers", layer, "completed"))


def counts(roots, layer=None):
    c = {"windows": 0, "with_input": 0, "with_12_groups": 0, "with_partial_groups": 0}
    if layer:
        c.update({"with_layer_dir": 0, "with_layer_completed": 0})
    for r in roots:
        for _, _, p in list_windows(r["path"]):
            c["windows"] += 1
            g = groups_completed(p)
            c["with_input"] += has_input(p)
            c["with_12_groups"] += g == N_GROUPS
            c["with_partial_groups"] += 0 < g < N_GROUPS
            if layer:
                c["with_layer_dir"] += os.path.isdir(os.path.join(p, "layers", layer))
                c["with_layer_completed"] += layer_done(p, layer)
    return c


def check_expected(year, c):
    e = EXPECTED[year]
    bad = {k: (c[k], v) for k, v in e.items() if c[k] != v}
    if bad or c["with_partial_groups"]:
        raise SystemExit(f"the {year} dataset is not the one exp98/exp99 predicted on: (found, expected) {bad}, "
                         f"windows with 1 to 11 item groups {c['with_partial_groups']}")


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- subcommands
def cmd_count(a):
    _, roots = load_roots(a.inv, a.run, a.year)
    c = counts(roots, a.layer)
    check_expected(a.year, c)
    print(json.dumps(c))
    if a.refuse_layer and c["with_layer_dir"]:
        print(f"{c['with_layer_dir']} window(s) already hold layers/{a.layer}/. The writer would write over them, and a "
              f"config with output_layer_name_skip_inference_if_exists would skip them and write nothing: rerun with "
              f"E103_REDO=1, which removes layers/{a.layer}/ from every window first")
        return 3
    return 0


def cmd_root_path(a):
    _, roots = load_roots(a.inv, a.run, a.year)
    print(roots[a.index]["path"])
    return 0


def cmd_remove_layer(a):
    if a.layer not in REMOVABLE:
        raise SystemExit(f"refused: only {', '.join(REMOVABLE)} may be removed")
    _, roots = load_roots(a.inv, a.run, a.year)
    n = 0
    for r in roots:
        for _, _, p in list_windows(r["path"]):
            d = os.path.join(p, "layers", a.layer)
            if os.path.lexists(d):
                shutil.rmtree(d)
                n += 1
    print(f"removed layers/{a.layer}/ from {n} window(s) of the {a.year} dataset")
    return 0


def model_inputs(config_path):
    """model.yaml's data inputs as rslearn DataInputs, targets left out (predict_config sets skip_targets, which drops
    them before the windows are checked: rslearn/train/dataset.py:996-999)."""
    import yaml
    from rslearn.train.dataset import DataInput
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    out = {}
    for k, v in cfg["data"]["init_args"]["inputs"].items():
        if v.get("is_target"):
            continue
        out[k] = DataInput(data_type=v["data_type"], layers=list(v["layers"]), bands=v.get("bands"),
                           required=v.get("required", True), passthrough=v.get("passthrough", False),
                           load_all_layers=v.get("load_all_layers", False),
                           load_all_item_groups=v.get("load_all_item_groups", False))
    return out


def cmd_precheck(a):
    """rslearn 0.1.17 itself: Dataset() parses config.json (pydantic, extra="forbid"), load_windows() reads every
    metadata.json, and check_window() (rslearn/train/dataset.py:905-943) says which windows `model predict` keeps."""
    import importlib.metadata as md
    from upath import UPath
    from rslearn.dataset.dataset import Dataset
    from rslearn.train.dataset import check_window
    _, roots = load_roots(a.inv, a.run, a.year)
    fs = counts(roots)
    check_expected(a.year, fs)
    inputs = model_inputs(a.config)
    rec = {"year": a.year, "rslearn": md.version("rslearn"), "filesystem": fs, "roots": []}
    kept_ids = set()
    for r in roots:
        ds = Dataset(UPath(r["path"]))
        wins = ds.load_windows(workers=a.workers)
        kept = [w for w in wins if check_window(inputs, w)[0] is not None]
        kept_ids |= {(r["root_id"], w.group, w.name) for w in kept}
        rec["roots"].append({"root_id": r["root_id"], "config_layers": sorted(ds.layers), "windows_loaded": len(wins),
                             "windows_rslearn_predicts": len(kept)})
    fs_ids = {(r["root_id"], g, n) for r in roots for g, n, p in list_windows(r["path"]) if has_input(p)}
    rec["windows_rslearn_predicts"] = len(kept_ids)
    rec["same_windows_as_filesystem_rule"] = kept_ids == fs_ids
    rec["ok"] = bool(rec["same_windows_as_filesystem_rule"] and len(kept_ids) == EXPECTED[a.year]["with_input"]
                     and sum(x["windows_loaded"] for x in rec["roots"]) == EXPECTED[a.year]["windows"])
    write_json(rec, a.out)
    print(json.dumps(rec))
    if not rec["ok"]:
        print("precheck failed: rslearn 0.1.17 does not see these windows as exp98/exp99 recorded them")
        return 1
    return 0


def cmd_smoke_config(a):
    """Two windows with all twelve item groups, from the first group (sorted) that has two, so that groups + names
    select exactly them (FileWindowStorage.get_windows looks each name up in each group:
    rslearn/dataset/storage/file.py:121-150). The derived config differs from model.yaml in nothing else (checked)."""
    import copy
    import yaml
    _, roots = load_roots(a.inv, a.run, a.year)
    check_expected(a.year, counts(roots))
    r = roots[0]
    by_group = {}
    for g, n, p in list_windows(r["path"]):
        if has_input(p) and groups_completed(p) == N_GROUPS:
            by_group.setdefault(g, []).append(n)
    groups = sorted(by_group)
    pick = next((g for g in groups if len(by_group[g]) >= 2), None)
    if pick is None:
        raise SystemExit("no group of root_0 holds two windows with input")
    names = by_group[pick][:2]
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    derived = copy.deepcopy(cfg)
    pc = derived["data"]["init_args"].setdefault("predict_config", {})
    pc["groups"] = [pick]
    pc["names"] = list(names)
    back = copy.deepcopy(derived)
    del back["data"]["init_args"]["predict_config"]["groups"], back["data"]["init_args"]["predict_config"]["names"]
    if back != cfg:
        raise SystemExit("the derived config differs from model.yaml beyond predict_config groups and names")
    with open(a.out, "w") as f:
        f.write("# exp103 smoke: exp/jobs/e102_config/model.yaml with data.init_args.predict_config.groups and .names "
                "set to two windows (exp/exp103_jobs.py smoke-config); nothing else differs\n")
        yaml.safe_dump(derived, f, sort_keys=False)
    # the same check on what rslearn parses: both files after its ${VAR} substitution (rslearn/arg_parser.py,
    # rslearn/template_params.py), the derived one without the two keys
    from rslearn.template_params import substitute_env_vars_in_string as subst
    with open(a.config) as f1, open(a.out) as f2:
        orig, der = yaml.safe_load(subst(f1.read())), yaml.safe_load(subst(f2.read()))
    dpc = der["data"]["init_args"]["predict_config"]
    if dpc["groups"] != [pick] or dpc["names"] != list(names):
        raise SystemExit("the derived config does not read back the two windows' group and names as strings")
    del dpc["groups"], dpc["names"]
    if der != orig:
        raise SystemExit("after rslearn's substitution the derived config differs from model.yaml")
    write_json({"root_id": r["root_id"], "windows": [f"{pick}/{n}" for n in names]}, a.ids_out)
    print(f"smoke windows: 2 of group index {groups.index(pick)} (of {len(groups)} groups with input) in "
          f"{r['root_id']}; derived config written")
    return 0


def read_ids(path):
    with open(path) as f:
        d = json.load(f)
    return d["root_id"], list(d["windows"])


def cmd_check(a):
    """e98_predict.sh's check of the output layer, for layer a.layer: every window's input and output, every output's
    band set and GeoTIFF header (10 float32 bands, the window's size), and up to eight sampled windows (or the smoke's
    two) read in full: finite, within [0, 1], the bands summing to 1 within 1e-3 at covered pixels."""
    import numpy as np
    import rasterio
    inv, roots = load_roots(a.inv, a.run, a.year)
    ids = None
    if a.ids:
        rid, wl = read_ids(a.ids)
        ids = {(rid, w) for w in wl}
    res = {"job": os.environ.get("SLURM_JOB_ID"), "year": a.year, "layer": a.layer, "windows": 0,
           "windows_without_input": 0, "windows_with_input": 0, "windows_with_12_groups": 0, "windows_with_output": 0,
           "missing_output": 0, "output_without_input": 0, "bad_bandset": 0, "bad_header": 0,
           "area_complete": inv.get("complete"), "allow_incomplete": bool(inv.get("allow_incomplete")), "sampled": []}
    outs, out_ids = [], set()
    for r in roots:
        for g, n, p in list_windows(r["path"]):
            hin, hout = has_input(p), layer_done(p, a.layer)
            res["windows"] += 1
            res["windows_without_input"] += not hin
            res["windows_with_input"] += hin
            res["windows_with_12_groups"] += groups_completed(p) == N_GROUPS
            res["windows_with_output"] += hout
            res["output_without_input"] += hout and not hin
            if ids is None:
                res["missing_output"] += hin and not hout
            if not hout:
                continue
            out_ids.add((r["root_id"], f"{g}/{n}"))
            ldir = os.path.join(p, "layers", a.layer)
            sets = sorted(d for d in os.listdir(ldir) if os.path.isfile(os.path.join(ldir, d, "geotiff.tif")))
            if sets != [P]:
                res["bad_bandset"] += 1
                continue
            tif = os.path.join(ldir, P, "geotiff.tif")
            with open(os.path.join(p, "metadata.json")) as f:
                b = json.load(f)["bounds"]
            with rasterio.open(tif) as src:
                hdr_ok = src.count == 10 and all(d == "float32" for d in src.dtypes) and \
                    (src.width, src.height) == (b[2] - b[0], b[3] - b[1])
            res["bad_header"] += not hdr_ok
            outs.append(((r["root_id"], f"{g}/{n}"), tif))
    if ids is not None:
        res["smoke_windows_written"] = len(ids & out_ids)
        res["other_windows_written"] = len(out_ids - ids)
        picks = [i for i, (k, _) in enumerate(outs) if k in ids]
    else:
        picks = sorted(set(np.linspace(0, len(outs) - 1, min(8, len(outs))).round().astype(int).tolist())) \
            if outs else []
    ok = True
    for i in picks:
        with rasterio.open(outs[i][1]) as src:
            arr = src.read()
        s = arr.sum(axis=0, dtype=np.float64)
        covered = s > 0                       # RasterMerger leaves 0 where no crop wrote
        e = {"window_index": int(i), "bands": int(arr.shape[0]), "dtype": str(arr.dtype),
             "finite": bool(np.isfinite(arr).all()), "min": float(arr.min()), "max": float(arr.max()),
             "uncovered_share": float(1 - covered.mean()),
             "max_abs_sum_minus_1": float(np.abs(s[covered] - 1).max()) if covered.any() else None,
             "argmax_is_9_share": float((arr.argmax(axis=0)[covered] == 9).mean()) if covered.any() else None,
             "max_p9": float(arr[9].max()), "mean_top1": float(arr.max(axis=0)[covered].mean()) if covered.any() else None}
        e["ok"] = bool(e["bands"] == 10 and e["dtype"] == "float32" and e["finite"] and e["min"] >= -1e-6
                       and e["max"] <= 1 + 1e-6 and e["max_abs_sum_minus_1"] is not None
                       and e["max_abs_sum_minus_1"] <= 1e-3)
        ok &= e["ok"]
        res["sampled"].append(e)
    if ids is not None:
        res["ok"] = bool(ok and len(picks) == len(ids) and res["smoke_windows_written"] == len(ids)
                         and res["other_windows_written"] == 0 and res["bad_bandset"] == 0 and res["bad_header"] == 0)
    else:
        exp = EXPECTED[a.year]
        res["ok"] = bool(ok and picks and res["missing_output"] == 0 and res["bad_bandset"] == 0
                         and res["bad_header"] == 0 and res["output_without_input"] == 0
                         and res["windows"] == exp["windows"] and res["windows_with_output"] == exp["with_input"]
                         and res["windows_with_12_groups"] == exp["with_12_groups"])
    write_json(res, a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "sampled"}))
    for e in res["sampled"]:
        print(json.dumps(e))
    if not res["ok"]:
        print(f"the output check failed; see {os.path.basename(a.out)}")
        return 1
    return 0


# ----------------------------------------------------------------------------- read
def esc(s):
    """A window id as a literal fnmatch pattern (e98_read.sh)."""
    return "".join(f"[{ch}]" if ch in "[]*?" else ch for ch in s)


def build_view(roots, collected, layer, view, ids=None, conditions=True):
    """VIEW/<root_id>/ as an rslearn dataset the reader reads: config.json (the collected copy's, which declares the
    SCL sidecar's sentinel2_scl layer: e98_collect.sh), and per window with layers/<layer>/ completed (or the ids
    given) symlinks to the deployment window's metadata.json and layers/<layer>/ and to the collected copy's
    layers/sentinel2_scl[.N]/. The collected metadata.json must equal the deployment's byte for byte."""
    if os.path.lexists(view):
        shutil.rmtree(view)
    c = {"windows": 0, "with_scl": 0, "missing_collected": 0, "metadata_differs": 0, "roots": []}
    for r in roots:
        vroot = os.path.join(view, r["root_id"])
        croot = os.path.join(collected, r["root_id"]) if collected else None
        os.makedirs(os.path.join(vroot, "windows"))
        cfg = os.path.join(croot, "config.json") if croot else None
        if cfg and os.path.isfile(cfg):
            shutil.copyfile(cfg, os.path.join(vroot, "config.json"))
        elif conditions:
            raise SystemExit(f"no collected config.json for {r['root_id']}: the condition layer needs the collected "
                             "copy (E103_CONDITIONS=0 reads without it)")
        n = 0
        for g, name, p in list_windows(r["path"]):
            wid = f"{g}/{name}"
            if (ids is not None and (r["root_id"], wid) not in ids) or (ids is None and not layer_done(p, layer)):
                continue
            vw = os.path.join(vroot, "windows", g, name)
            os.makedirs(os.path.join(vw, "layers"))
            os.symlink(os.path.join(p, "metadata.json"), os.path.join(vw, "metadata.json"))
            os.symlink(os.path.join(p, "layers", layer), os.path.join(vw, "layers", layer))
            n += 1
            if not conditions:
                continue
            cw = os.path.join(croot, "windows", g, name)
            if not os.path.isdir(cw):
                c["missing_collected"] += 1
                continue
            with open(os.path.join(cw, "metadata.json"), "rb") as f1, open(os.path.join(p, "metadata.json"), "rb") as f2:
                c["metadata_differs"] += f1.read() != f2.read()
            scl = sorted(d for d in glob.glob(os.path.join(glob.escape(cw), "layers", SCL + "*"))
                         if re.fullmatch(re.escape(SCL) + r"(\.\d+)?", os.path.basename(d)))
            for d in scl:
                os.symlink(d, os.path.join(vw, "layers", os.path.basename(d)))
            c["with_scl"] += bool(scl)
        c["windows"] += n
        c["roots"].append({"root_id": r["root_id"], "windows": n})
    if conditions and (c["missing_collected"] or c["metadata_differs"] or c["with_scl"] != c["windows"]):
        raise SystemExit(f"the collected copy does not match the predicted windows: {json.dumps(c)}")
    return c


def plan_grid(view_root, ri, limit):
    """e98_read.sh's plan, unchanged: per CRS, windows sorted by their pixel bounds' (x0, y0), greedily packed into parts
    of at most `limit` bytes of grid whose windows do not overlap one another."""
    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    def union(a, b):
        return b if a is None else [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

    def meet(a, b):
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    wins = []
    for g, n, p in list_windows(view_root):
        with open(os.path.join(p, "metadata.json")) as f:
            m = json.load(f)
        pr = m["projection"]
        wins.append({"id": f"{g}/{n}", "crs": (pr["crs"], pr["x_resolution"], pr["y_resolution"]),
                     "b": [int(v) for v in m["bounds"]]})
    parts = []
    for crs in sorted({w["crs"] for w in wins}, key=str):
        left = sorted((w for w in wins if w["crs"] == crs), key=lambda w: (w["b"][0], w["b"][1]))
        while left:
            part, box, rest = [], None, []
            for w in left:
                nb = union(box, w["b"])
                if (part and area(nb) * BYTES_PER_PX > limit) or any(meet(w["b"], q["b"]) for q in part):
                    rest.append(w)
                    continue
                part.append(w)
                box = nb
            if area(box) * BYTES_PER_PX > limit:
                raise SystemExit(f"a single window of root {ri} is larger than the part limit")
            parts.append({"windows": [w["id"] for w in part], "grid_gib": area(box) * BYTES_PER_PX / 2 ** 30})
            left = rest
    return len(wins), parts


def mask_text(text, secrets):
    for s, tag in secrets:
        text = text.replace(s, tag)
    return re.sub(r"-?[0-9]{4,}", "<n>", re.sub(r"-?[0-9]+\.[0-9]{2,}", "<n>", text))


def secrets_of(roots, view=None):
    """(string, placeholder) pairs, the longer first: the roots' real paths, the view's path, groups and windows."""
    out = []
    for r in roots:
        out += [(r["path"], f"<{r['root_id']}>"), (os.path.basename(r["path"]), f"<{r['root_id']}>")]
        for g, n, _ in list_windows(r["path"]):
            out += [(n, "<window>"), (g, "<group>")]
    if view:
        out.append((view, "<view>"))
    return sorted(set(out), key=lambda x: -len(x[0]))


def burned_from_exp98(scores, plan, exp98):
    """exp98's MODIS burned layer (exp/exp98_burned.py) is a property of the grid, not of the model: it is copied
    beside each part only when exp98 read a part of the same name from the same windows and every scores_<label>.tif
    of ours has exp98's on the identical grid (CRS, transform, size) with burned_<label>.tif beside it; all parts or
    none."""
    import rasterio
    pj = os.path.join(exp98, "parts.json")
    if not os.path.isfile(pj):
        return {"copied_parts": 0, "note": "exp98's parts.json is absent"}
    with open(pj) as f:
        theirs = {p["dir"]: p for p in json.load(f)}
    todo = []
    for p in plan:
        t = theirs.get(p["dir"])
        if t is None or t["root"] != p["root"] or list(t["windows"]) != list(p["windows"]):
            return {"copied_parts": 0, "note": f"part {p['dir']} differs from exp98's"}
        mine = os.path.join(scores, p["dir"])
        for s in sorted(glob.glob(os.path.join(glob.escape(mine), "scores_*.tif"))):
            lab = re.sub(r"^scores_|\.tif$", "", os.path.basename(s))
            ts, tb = os.path.join(exp98, p["dir"], f"scores_{lab}.tif"), os.path.join(exp98, p["dir"], f"burned_{lab}.tif")
            if not (os.path.isfile(ts) and os.path.isfile(tb)):
                return {"copied_parts": 0, "note": f"exp98's part {p['dir']} lacks scores_ or burned_ for one grid"}
            with rasterio.open(s) as a, rasterio.open(ts) as b:
                same = a.crs == b.crs and a.transform == b.transform and (a.width, a.height) == (b.width, b.height)
            if not same:
                return {"copied_parts": 0, "note": f"part {p['dir']}: a grid differs from exp98's"}
            todo.append((tb, os.path.join(mine, os.path.basename(tb))))
        names = os.path.join(exp98, p["dir"], "burned_conditions.json")
        if not os.path.isfile(names):
            return {"copied_parts": 0, "note": f"exp98's part {p['dir']} lacks burned_conditions.json"}
        todo.append((names, os.path.join(mine, "burned_conditions.json")))
    for src, dst in todo:
        shutil.copyfile(src, dst)
    return {"copied_parts": len(plan), "files": len(todo)}


def cmd_read(a):
    _, roots = load_roots(a.inv, a.run, a.year)
    ids = None
    if a.ids:
        rid, wl = read_ids(a.ids)
        ids = {(rid, w) for w in wl}
    rec = {"job": os.environ.get("SLURM_JOB_ID"), "year": a.year, "layer": a.layer, "plan": a.plan,
           "conditions": a.conditions}
    rec["view"] = build_view(roots, a.collected, a.layer, a.view, ids, a.conditions)
    secrets = secrets_of(roots, a.view)
    if os.path.lexists(a.scores):
        raise SystemExit(f"{os.path.basename(a.scores)} exists (the job removes it only with E103_REREAD=1)")
    part_dir = a.scores + ".partial"
    if os.path.lexists(part_dir):
        shutil.rmtree(part_dir)
    os.makedirs(part_dir)
    limit = int(a.part_gib * 2 ** 30)
    plan, rec["roots"] = [], []
    for ri, r in enumerate(roots):
        vroot = os.path.join(a.view, r["root_id"])
        if a.plan == "grid":
            n, parts = plan_grid(vroot, ri, limit)
            rec["roots"].append({"root_id": r["root_id"], "windows": n, "parts": [
                {"windows": len(p["windows"]), "grid_gib": round(p["grid_gib"], 2)} for p in parts]})
            plan += [{"root": r["root_id"], "dir": f"r{ri:02d}_p{pi:02d}", "windows": p["windows"]}
                     for pi, p in enumerate(parts)]
        else:
            wins = [f"{g}/{n}" for g, n, _ in list_windows(vroot)]
            rec["roots"].append({"root_id": r["root_id"], "windows": len(wins)})
            plan += [{"root": r["root_id"], "dir": f"r{ri:02d}_w{k:04d}", "windows": [w]} for k, w in enumerate(wins)]
    write_json(plan, os.path.join(part_dir, "parts.json"))
    print(json.dumps({k: rec[k] for k in ("roots", "view")}))
    t0 = time.time()
    windows_read, max_dev = 0, 0.0
    for i, p in enumerate(plan):
        out = os.path.join(part_dir, p["dir"])
        cmd = ["oe-inferencex", "from-olmoearth", os.path.join(a.view, p["root"]), "--out", out, "--layer", a.layer]
        cmd += ["--conditions", "--inputs", SCL] if a.conditions else []
        cmd += ["--window"] + [esc(w) for w in p["windows"]]
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if rc.returncode != 0:
            last = (rc.stderr.strip().splitlines() or [""])[-1][:300]
            last = mask_text(last.replace(part_dir, "<scores>"), secrets)
            print(f"from-olmoearth failed on part {p['dir']} (exit {rc.returncode}); the last line of its error, "
                  f"masked: {last}", flush=True)
            return 1
        with open(os.path.join(out, "olmoearth_output.json")) as f:
            so = json.load(f)
        n = len([f for f in os.listdir(out) if f.startswith("scores_")])
        bad = so.get("bands") != 10 or so.get("windows_read") != len(p["windows"]) or n < 1 or \
            (a.plan == "window" and n != 1) or (a.conditions and not os.path.isfile(os.path.join(out, "olmoearth_conditions.json")))
        if bad:
            print(f"part {p['dir']}: bands {so.get('bands')}, windows read {so.get('windows_read')} of "
                  f"{len(p['windows'])}, {n} scores raster(s), conditions "
                  f"{os.path.isfile(os.path.join(out, 'olmoearth_conditions.json'))}")
            return 1
        windows_read += so["windows_read"]
        max_dev = max(max_dev, float(so.get("largest_sum_deviation") or 0.0))
        if (i + 1) % 25 == 0 or i + 1 == len(plan):
            print(f"{i + 1} of {len(plan)} parts read ({time.time() - t0:.0f} s)", flush=True)
    rec.update(parts_read=len(plan), windows_read=windows_read, largest_sum_deviation=max_dev,
               seconds=round(time.time() - t0, 1))
    if a.burned_from:
        rec["burned"] = burned_from_exp98(part_dir, plan, a.burned_from) if a.plan == "grid" else \
            {"copied_parts": 0, "note": "only the grid plan matches exp98's parts"}
    os.replace(part_dir, a.scores)
    write_json(rec, a.record)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("roots", "view")}))
    return 0


def cmd_verify_breadth(a):
    sys.path.insert(0, EXP_DIR)
    import timesync_awf_crosswalk as tsx
    ok = True
    for src in (tsx.TIMESYNC, tsx.BOUNDARIES):
        good, got = tsx.verify(os.path.join(a.dir, src["file"]), src)
        print(f"{src['file']}: {'verified' if good else 'NOT verified'}"
              + (f", sha256 {got['sha256']}, git blob {got['git_blob_sha1']}" if isinstance(got, dict) else f" ({got})"))
        ok &= good
    if a.request_geometry:
        good = os.path.isfile(a.request_geometry) and tsx.sha256_file(a.request_geometry) == tsx.AWF_GEOMETRY_SHA256
        print(f"Ai2's request geometry: {'verified' if good else 'NOT verified'}")
        ok &= good
    return 0 if ok else 1


def cmd_mask(a):
    _, roots = load_roots(a.inv, a.run, a.year)
    secrets = secrets_of(roots)
    for line in sys.stdin:
        sys.stdout.write(mask_text(line, secrets))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def ds(p):
        p.add_argument("--inv", required=True, help="the scratch inventory (real root names)")
        p.add_argument("--run", required=True, help="olmoearth_run's scratch_path holding the dataset roots")
        p.add_argument("--year", required=True, choices=tuple(EXPECTED))
        return p

    p = ds(sub.add_parser("count"))
    p.add_argument("--layer", default=None)
    p.add_argument("--refuse-layer", action="store_true", help="exit 3 if any window holds layers/<layer>/")
    p.set_defaults(func=cmd_count)
    p = ds(sub.add_parser("root-path"))
    p.add_argument("--index", type=int, default=0)
    p.set_defaults(func=cmd_root_path)
    p = ds(sub.add_parser("remove-layer"))
    p.add_argument("--layer", required=True)
    p.set_defaults(func=cmd_remove_layer)
    p = ds(sub.add_parser("precheck"))
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_precheck)
    p = ds(sub.add_parser("smoke-config"))
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids-out", required=True)
    p.set_defaults(func=cmd_smoke_config)
    p = ds(sub.add_parser("check"))
    p.add_argument("--layer", required=True)
    p.add_argument("--ids", default=None, help="the smoke's id list: check exactly those windows")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_check)
    p = ds(sub.add_parser("read"))
    p.add_argument("--layer", required=True)
    p.add_argument("--collected", default=None, help="exp98's or exp99's collected copy (config.json and SCL layers)")
    p.add_argument("--view", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--plan", choices=("grid", "window"), required=True)
    p.add_argument("--part-gib", type=float, default=3.0)
    p.add_argument("--ids", default=None)
    p.add_argument("--conditions", action="store_true")
    p.add_argument("--burned-from", default=None, help="exp98's scores directory (parts.json, burned_<EPSG>.tif)")
    p.add_argument("--record", required=True)
    p.set_defaults(func=cmd_read)
    p = sub.add_parser("verify-breadth")
    p.add_argument("--dir", required=True)
    p.add_argument("--request-geometry", default=None)
    p.set_defaults(func=cmd_verify_breadth)
    p = ds(sub.add_parser("mask"))
    p.set_defaults(func=cmd_mask)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
