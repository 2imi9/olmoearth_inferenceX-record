#!/usr/bin/env python
"""exp120: estimate --valid and the per-profile verdicts on fully labelled real maps (preregistered 29 October 2026,
docs/plan/valid_and_profiles_on_real_maps.md; the predictions are there and were frozen before this ran).

    python exp/exp120_valid_profiles.py units --units DIR --exp116 FILE --out FILE [--workers K]   # arm A
    python exp/exp120_valid_profiles.py tile --out FILE --work DIR [--workers K]                    # arm B
    python exp/exp120_valid_profiles.py summary --units-out FILE --tile-out FILE --out FILE        # P1 to P3
    python exp/exp120_valid_profiles.py smoke                                                       # synthetic units

Runs against the installed package (1.20.0 or later: `estimate_error_rate(valid=True)`, `build_profiles`,
`profile_verdicts`), whose version is recorded. Arm A calls the package's functions as the command line calls them
(estimate/sample_command.py: the class design is the condition design with the map class as the condition and the
class floor; the profile design the condition design with build_profiles' index as the condition). Arm B runs the
command line end to end on the package's Dynamic World tile. Every number is a count or share of draws; there is no
fit.
"""
import argparse
import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import sys
import tempfile
import zlib
from multiprocessing import Pool

import numpy as np

import oe_inferencex
from oe_inferencex import estimate as est

D = 400                                   # draws per cell
VALID_CELLS = [("random", 300), ("random", 1000), ("confidence", 300), ("confidence", 1000),
               ("proportional", 300), ("proportional", 1000), ("class", 400), ("class", 1000)]
UNIT_SPECS = ("confidence:4", "class")
TILE_SPECS = ("confidence:4", "class", "grid:2,confidence:2")
UNIT_ALPHA_FACTORS = (0.5, 1.0, 2.0)
TILE_ALPHAS = (0.1, 0.2, 0.3)
PROFILE_BUDGET = 800
DELTA = 0.1
COVER_BAR = 0.917                         # 0.95 - 3 binomial standard errors at 400 draws
WRONG_BAR = 0.083                         # delta / 2 + 3 standard errors
MISS_BAR = 0.145                          # delta + 3 standard errors
SHARE_BAR = 0.99
TILE_SEED0 = 120_000_000


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seeds(name, cell):
    rng = np.random.default_rng(np.random.SeedSequence([120, zlib.crc32(name.encode()), cell]))
    return [int(s) for s in rng.integers(0, 2 ** 31 - 1, D)]


def _refusal(exc):
    return f"{type(exc).__name__}: {str(exc)[:160]}"


# ----------------------------------------------------------------------------- grading, shared by both arms
def _new_valid_cell():
    return {"draws": 0, "refused": 0, "refusals": [], "valid_covers": 0, "valid_not_computed": 0, "plain_covers": 0,
            "plain_none": 0, "valid_width_sum": 0.0, "plain_width_sum": 0.0}


def _grade_valid(cell, res, true):
    cell["draws"] += 1
    v = res["valid_interval"]
    if v.get("low") is None or v.get("high") is None:
        cell["valid_not_computed"] += 1          # counts as not covering (the plan)
    else:
        cell["valid_covers"] += int(v["low"] <= true <= v["high"])
        cell["valid_width_sum"] += v["high"] - v["low"]
    if res.get("low") is None or res.get("high") is None:
        cell["plain_none"] += 1
    else:
        cell["plain_covers"] += int(res["low"] <= true <= res["high"])
        cell["plain_width_sum"] += res["high"] - res["low"]


def _new_profile_cell():
    return {"draws": 0, "refused": 0, "refusals": [], "any_wrong": 0, "any_miss": 0, "decided_tested": 0,
            "tested": 0, "census": 0, "n_profiles": None}


def _grade_profiles(cell, rows, truth_of, alpha):
    """rows: profile_verdicts' rows ({name: row}); truth_of(index) -> the profile's true error rate."""
    cell["draws"] += 1
    wrong = miss = False
    for row in rows.values():
        true = truth_of(int(row["index"]))
        v = row["verdict"]
        wrong |= (v == "good" and true > alpha) or (v == "bad" and true <= alpha)
        lo, hi = row.get("low"), row.get("high")
        miss |= lo is None or hi is None or not (lo <= true <= hi)
        if row.get("census"):
            cell["census"] += 1
        elif row.get("tested"):
            cell["tested"] += 1
            cell["decided_tested"] += int(v != "undecided")
    cell["any_wrong"] += int(wrong)
    cell["any_miss"] += int(miss)
    cell["n_profiles"] = len(rows)


def _refuse(cell, exc):
    cell["refused"] += 1
    msg = _refusal(exc)
    if msg not in cell["refusals"] and len(cell["refusals"]) < 3:
        cell["refusals"].append(msg)


# ----------------------------------------------------------------------------- arm A
def _load_unit(path):
    z = np.load(path, allow_pickle=False)
    margin = z["margin"].astype(np.float64).ravel()
    p1 = z["p1"].astype(np.float64).ravel()
    dec = z["dec"].astype(np.int64).ravel()
    err = z["err"].astype(np.int64).ravel()
    pop = np.isfinite(margin)
    return margin, p1, dec, err, pop


def _unit_job(job):
    path, name = job
    margin, p1, dec, err, pop = _load_unit(path)
    N = int(pop.sum())
    e = float(err[pop].mean())
    out = {"map": name, "n_units": N, "error_rate": e, "valid": [], "profiles": []}
    for ci, (design, n) in enumerate(VALID_CELLS):
        if n > N // 2:
            continue
        cell = dict(_new_valid_cell(), design=design, n=n)
        for seed in _seeds(name, ci):
            try:
                if design == "class":
                    names = {int(c): f"class {int(c)}" for c in np.unique(dec[pop])}
                    s = est.sample_for_estimation(margin, n, design="condition", p1=p1, valid=pop, seed=seed,
                                                  condition=np.where(pop, dec, -1), condition_names=names,
                                                  class_floor=est.CLASS_FLOOR)
                else:
                    s = est.sample_for_estimation(margin, n, design=design, p1=p1, valid=pop, seed=seed)
                res = est.estimate_error_rate(s, err[np.asarray(s["indices"], int)], valid=True)
            except ValueError as exc:
                _refuse(cell, exc)
                continue
            _grade_valid(cell, res, e)
        out["valid"].append(cell)
    if not (0.02 <= e <= 0.5 and N >= 2 * PROFILE_BUDGET):
        return out
    for si, spec in enumerate(UNIT_SPECS):
        try:
            prof = est.build_profiles(margin.reshape(1, -1), pop.reshape(1, -1), est.parse_profile_by(spec),
                                      map_class=dec.reshape(1, -1))
        except ValueError as exc:
            for f in UNIT_ALPHA_FACTORS:
                cell = dict(_new_profile_cell(), spec=spec, alpha=f * e, factor=f)
                _refuse(cell, exc)
                cell["refused"] = D
                out["profiles"].append(cell)
            continue
        index = prof["index"]
        rates = {h: float(err[index == h].mean()) for h in range(len(prof["names"]))}
        names = dict(enumerate(prof["names"]))
        for fi, f in enumerate(UNIT_ALPHA_FACTORS):
            alpha = f * e
            cell = dict(_new_profile_cell(), spec=spec, alpha=alpha, factor=f)
            for seed in _seeds(name, 100 + 10 * si + fi):
                try:
                    s = est.sample_for_estimation(margin, PROFILE_BUDGET, design="condition", p1=p1, valid=pop,
                                                  seed=seed, condition=index, condition_names=names)
                    s["profile"] = prof["block"]
                    idx = np.asarray(s["indices"], int)
                    res = est.profile_verdicts(s, err[idx], alpha, delta=DELTA)
                except ValueError as exc:
                    _refuse(cell, exc)
                    continue
                _grade_profiles(cell, res["profiles"], rates.__getitem__, alpha)
            out["profiles"].append(cell)
    return out


def units(units_dir, exp116, out_path, workers):
    planned = json.load(open(exp116))["input_sha256"]
    planned = {k: v for k, v in planned.items() if not k.startswith("olmoearth_base_rtx/")}
    if len(planned) != 356:
        raise SystemExit(f"exp116's planned population has {len(planned)} files, not 356")
    files = sorted(glob.glob(os.path.join(units_dir, "*", "*.npz")))
    have = {os.path.relpath(f, units_dir): f for f in files}
    missing = sorted(set(planned) - set(have))
    if missing:
        raise SystemExit(f"{len(missing)} planned unit files are missing, e.g. {missing[:3]}")
    bad = [n for n in sorted(planned) if _sha256(have[n]) != planned[n]]
    if bad:
        raise SystemExit(f"{len(bad)} unit files differ from exp116's sha256, e.g. {bad[:3]}")
    jobs = [(have[n], n) for n in sorted(planned)]
    with Pool(workers) as pool:
        rows = pool.map(_unit_job, jobs, chunksize=1)
    result = {"experiment": "exp120 arm A", "package_version": oe_inferencex.__version__, "draws": D,
              "population": "exp116's 356 planned unit files (sha256 checked)", "maps": rows}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"arm A: {len(rows)} maps written to {out_path}")


# ----------------------------------------------------------------------------- arm B
def _tile(work):
    z = np.load(os.path.join(os.path.dirname(oe_inferencex.__file__), "sample", "dynamic_world_tile.npz"))
    probs, expert = z["probs"].astype(np.float32), z["expert"].astype(np.int64)
    probs[:, expert < 0] = np.nan                       # no expert label: no-data
    path = os.path.join(work, "dw.npy")
    if not os.path.exists(path):
        np.save(path, probs)
    dec = np.nanargmax(np.where(np.isnan(probs), -np.inf, probs), axis=0)
    valid = expert >= 0
    return path, expert, dec, valid, (dec != expert) & valid


def _cli(argv):
    from oe_inferencex.cli import main
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        try:
            return main(argv), ""
        except SystemExit as exc:
            return 2, f"{exc} {err.getvalue().strip()[-200:]}".strip()


def _label(csv_path, expert, dec):
    """The expert's class as the reference; returns the rows whose CSV map class is not the script's own."""
    rows = list(csv.DictReader(open(csv_path)))
    off = 0
    for r in rows:
        i, j = int(r["pixel_row"]), int(r["pixel_col"])
        off += int(int(r["map_class"]) != int(dec[i, j]))
        r["wrong"] = int(int(r["map_class"]) != int(expert[i, j]))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return off


def _tile_job(job):
    work, kind, ci, key = job
    path, expert, dec, valid, wrong = _tile(work)
    true = float(wrong[valid].mean())
    cell = dict(_new_valid_cell() if kind == "valid" else _new_profile_cell(), map_class_mismatch=0,
                population_mismatch=0)
    if kind == "valid":
        design, n = key
        cell.update(design=design, n=n)
    else:
        spec, alpha = key
        cell.update(spec=spec, alpha=alpha)
    for k in range(D):
        seed = TILE_SEED0 + 1000 * ci + k
        with tempfile.TemporaryDirectory(dir=work) as d:
            s = os.path.join(d, "s.csv")
            argv = ["sample", path, "--patch", "1", "--seed", str(seed), "--out", s]
            argv += (["--design", key[0], "--budget", str(key[1])] if kind == "valid" else
                     ["--design", "profile", "--profile-by", key[0], "--budget", str(PROFILE_BUDGET)])
            code, msg = _cli(argv)
            if code != 0:
                _refuse(cell, RuntimeError(f"sample: {msg}"))
                continue
            cell["map_class_mismatch"] += _label(s, expert, dec)
            side = json.load(open(os.path.join(d, "s.json")))
            cell["population_mismatch"] += int(side.get("n_population") != int(valid.sum()))
            if kind == "valid":
                code, msg = _cli(["estimate", s, "--valid", "--out", os.path.join(d, "e.json")])
                if code != 0:
                    _refuse(cell, RuntimeError(f"estimate: {msg}"))
                    continue
                _grade_valid(cell, json.load(open(os.path.join(d, "e.json"))), true)
            else:
                code, msg = _cli(["certify", s, "--alpha", str(key[1])])
                if code != 0:
                    _refuse(cell, RuntimeError(f"certify: {msg}"))
                    continue
                res = json.load(open(os.path.join(d, "s_profiles.json")))
                grid = np.asarray(side["condition_grid"], dtype=np.int64).ravel()
                cell["population_mismatch"] += int(int((grid >= 0).sum()) != int(valid.sum()))
                w = wrong.ravel()
                _grade_profiles(cell, res["profiles"], lambda h: float(w[grid == h].mean()), key[1])
    return cell


def tile(out_path, work, workers):
    os.makedirs(work, exist_ok=True)
    path, expert, dec, valid, wrong = _tile(work)
    jobs = [(work, "valid", ci, c) for ci, c in enumerate(VALID_CELLS)]
    jobs += [(work, "profile", 100 + 10 * si + ai, (spec, a)) for si, spec in enumerate(TILE_SPECS)
             for ai, a in enumerate(TILE_ALPHAS)]
    with Pool(workers) as pool:
        cells = pool.map(_tile_job, jobs, chunksize=1)
    result = {"experiment": "exp120 arm B", "package_version": oe_inferencex.__version__, "draws": D,
              "map": "Dynamic World expert tile (oe_inferencex/sample/dynamic_world_tile.npz)",
              "n_windows": int(valid.sum()), "error_rate": float(wrong[valid].mean()),
              "valid": [c for c in cells if "design" in c], "profiles": [c for c in cells if "spec" in c]}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"arm B written to {out_path}")


# ----------------------------------------------------------------------------- the predictions
def _share(cells, ok):
    return (sum(map(ok, cells)) / len(cells)) if cells else None


def _cover(c):
    graded = c["draws"]
    return None if not graded else c["valid_covers"] / graded


def analyze(a, b):
    a_valid = [dict(c, map=m["map"]) for m in a["maps"] for c in m["valid"] if c["draws"]]
    a_prof = [dict(c, map=m["map"]) for m in a["maps"] for c in m["profiles"] if c["draws"]]
    b_valid = [c for c in b["valid"] if c["draws"]]
    b_prof = [c for c in b["profiles"] if c["draws"]]
    cov = lambda c: _cover(c) >= COVER_BAR
    wr = lambda c: c["any_wrong"] / c["draws"] <= WRONG_BAR
    mi = lambda c: c["any_miss"] / c["draws"] <= MISS_BAR
    p1 = {"cells_a": len(a_valid), "share_a_at_bar": _share(a_valid, cov),
          "cells_b": len(b_valid), "all_b_at_bar": all(map(cov, b_valid)) and bool(b_valid),
          "min_coverage_a": min((_cover(c) for c in a_valid), default=None),
          "min_coverage_b": min((_cover(c) for c in b_valid), default=None),
          "cells_a_below": [{k: c[k] for k in ("map", "design", "n")} | {"coverage": _cover(c)}
                            for c in a_valid if not cov(c)]}
    p1["held"] = (p1["share_a_at_bar"] or 0) >= SHARE_BAR and p1["all_b_at_bar"]
    p2 = {"cells_a": len(a_prof), "share_a_at_bar": _share(a_prof, wr), "cells_b": len(b_prof),
          "all_b_at_bar": all(map(wr, b_prof)) and bool(b_prof),
          "max_share_a": max((c["any_wrong"] / c["draws"] for c in a_prof), default=None),
          "max_share_b": max((c["any_wrong"] / c["draws"] for c in b_prof), default=None)}
    p2["held"] = (p2["share_a_at_bar"] or 0) >= SHARE_BAR and p2["all_b_at_bar"]
    p3 = {"cells_a": len(a_prof), "share_a_at_bar": _share(a_prof, mi), "cells_b": len(b_prof),
          "all_b_at_bar": all(map(mi, b_prof)) and bool(b_prof),
          "max_share_a": max((c["any_miss"] / c["draws"] for c in a_prof), default=None),
          "max_share_b": max((c["any_miss"] / c["draws"] for c in b_prof), default=None)}
    p3["held"] = (p3["share_a_at_bar"] or 0) >= SHARE_BAR and p3["all_b_at_bar"]
    by_design = {}
    for design, n in VALID_CELLS:
        cs = [c for c in a_valid if c["design"] == design and c["n"] == n]
        ws = [c["valid_width_sum"] / c["draws"] / (c["plain_width_sum"] / (c["draws"] - c["plain_none"]))
              for c in cs if c["plain_width_sum"] > 0 and c["draws"] > c["plain_none"]]
        by_design[f"{design}_{n}"] = {
            "cells": len(cs), "median_valid_coverage": float(np.median([_cover(c) for c in cs])) if cs else None,
            "median_plain_coverage": float(np.median([c["plain_covers"] / c["draws"] for c in cs])) if cs else None,
            "median_width_valid_over_plain": float(np.median(ws)) if ws else None,
            "refused_draws": sum(c["refused"] for c in cs)}
    refused = {"a_valid_cells_all_refused": sum(1 for m in a["maps"] for c in m["valid"] if not c["draws"]),
               "a_profile_cells_all_refused": sum(1 for m in a["maps"] for c in m["profiles"] if not c["draws"]),
               "b_refused_draws": sum(c["refused"] for c in b["valid"] + b["profiles"])}
    decided = {"a_mean_decided_share_of_tested": (sum(c["decided_tested"] for c in a_prof)
                                                  / max(1, sum(c["tested"] for c in a_prof))),
               "b_mean_decided_share_of_tested": (sum(c["decided_tested"] for c in b_prof)
                                                  / max(1, sum(c["tested"] for c in b_prof)))}
    guards = {"b_map_class_mismatch": sum(c.get("map_class_mismatch", 0) for c in b["valid"] + b["profiles"]),
              "b_population_mismatch": sum(c.get("population_mismatch", 0) for c in b["valid"] + b["profiles"])}
    return {"P1": p1, "P2": p2, "P3": p3, "valid_by_design_a": by_design, "refused": refused, "decided": decided,
            "guards": guards, "bars": {"cover": COVER_BAR, "wrong": WRONG_BAR, "miss": MISS_BAR, "share": SHARE_BAR}}


def summary(units_out, tile_out, out_path):
    a, b = json.load(open(units_out)), json.load(open(tile_out))
    res = {"experiment": "exp120", "package_version": a["package_version"], "tile_package_version":
           b["package_version"], "draws": D, "maps_a": len(a["maps"]), "tile_error_rate": b["error_rate"],
           "tile_windows": b["n_windows"], **analyze(a, b)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("P1", "P2", "P3")}, indent=1, default=float))


# ----------------------------------------------------------------------------- smoke
def smoke():
    global D
    D = 6
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(d, "units", "enc"))
    shas = {}
    for t, (N, C) in enumerate([(3000, 3), (2500, 5), (900, 2)]):
        y = rng.integers(0, C, N)
        margin = rng.random(N)
        err = (rng.random(N) < 0.35 * (1 - margin)).astype(np.uint8)
        dec = np.where(err == 1, (y + 1) % C, y)
        p = os.path.join(d, "units", "enc", f"t{t}.npz")
        np.savez(p, margin=margin, p1=0.5 + margin / 2, dec=dec, y=y, err=err, n_classes=np.array([C]))
        shas[f"enc/t{t}.npz"] = _sha256(p)
    shas.update({f"x{i}/y.npz": "0" for i in range(353)})
    ex = os.path.join(d, "exp116_ab.json")
    json.dump({"input_sha256": shas}, open(ex, "w"))
    rows = [_unit_job((os.path.join(d, "units", n), n)) for n in sorted(k for k in shas if k.startswith("enc/"))]
    a = {"package_version": oe_inferencex.__version__, "maps": rows}
    work = os.path.join(d, "tile")
    os.makedirs(work)
    cells = [_tile_job((work, "valid", 0, ("random", 300))), _tile_job((work, "valid", 6, ("class", 400))),
             _tile_job((work, "profile", 100, ("confidence:4", 0.2)))]
    b = {"package_version": oe_inferencex.__version__, "valid": cells[:2], "profiles": cells[2:], "error_rate": 0.0,
         "n_windows": 0}
    res = analyze(a, b)
    print(json.dumps({"maps": [(m["map"], len(m["valid"]), len(m["profiles"])) for m in rows], "tile_cells":
                      [(c.get("design") or c.get("spec"), c["draws"], c["refused"], c["refusals"]) for c in cells],
                      "P1": res["P1"]["held"], "P2": res["P2"]["held"], "P3": res["P3"]["held"],
                      "guards": res["guards"]}, indent=1, default=float))
    assert all(c["draws"] == D and not c["refused"] for c in cells), "a tile cell refused or lost draws"
    assert res["guards"] == {"b_map_class_mismatch": 0, "b_population_mismatch": 0}
    assert all(c["draws"] == D for m in rows for c in m["valid"]), "a unit cell lost draws"
    print("smoke ok")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("units")
    u.add_argument("--units", required=True)
    u.add_argument("--exp116", required=True)
    u.add_argument("--out", required=True)
    u.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("tile")
    t.add_argument("--out", required=True)
    t.add_argument("--work", required=True)
    t.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("summary")
    s.add_argument("--units-out", required=True)
    s.add_argument("--tile-out", required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("smoke")
    a = ap.parse_args()
    if a.cmd == "units":
        units(a.units, a.exp116, a.out, a.workers)
    elif a.cmd == "tile":
        tile(a.out, a.work, a.workers)
    elif a.cmd == "summary":
        summary(a.units_out, a.tile_out, a.out)
    else:
        smoke()


if __name__ == "__main__":
    sys.exit(main())
