#!/usr/bin/env python
"""exp120, second route: `estimate --valid` and the per-profile verdicts on fully labelled real maps, implemented a
second time from the preregistration (docs/plan/valid_and_profiles_on_real_maps.md) and the package's source alone,
without reading the first implementation (exp/exp120_valid_profiles.py). Its counts are compared cell by cell with
the first route's.

    python exp/exp120_crosscheck.py units --units DIR --exp116 FILE --out FILE [--workers K]   # arm A
    python exp/exp120_crosscheck.py tile --out FILE --work DIR [--workers K]                   # arm B
    python exp/exp120_crosscheck.py smoke                                                      # synthetic, few draws

The package (olmoearth-inferencex 1.20.3, its version recorded) is called only for what the plan says it does: in arm
A `estimate.sample_for_estimation`, `estimate.estimate_error_rate(..., valid=True)`, `estimate.parse_profile_by`,
`estimate.build_profiles` and `certify.localized.profile_verdicts`; in arm B the command line (`oe_inferencex.cli.main`)
with sample, estimate --valid and certify. Every graded quantity is this script's own: the truth of a map (the mean of
`err` over its units; in arm B of map class != expert class over the expert-labelled windows), the truth of each
profile, the coverage tests and the wrong-verdict tests.

Grading conventions (the plan does not fix them; chosen here and stated):
- an interval covers when low <= truth <= high in floats, truth = K / N, the nearest float to the map's rate (the
  package rounds exact ends to the nearest float, so an end equal to the truth as a fraction is the same float);
- a verdict is wrong when it is good and the profile's rate k / N_h is above alpha, or bad and the rate is at most
  alpha, compared exactly with alpha as written (Fraction(repr(alpha))), as profile_verdicts compares;
- a profile's interval misses when its row's low is above k / N_h or its high below, exactly (the package rounds
  those ends outward);
- a ValueError from the package (arm A), or a non-zero exit of a command (arm B), is a refusal: counted, never graded;
  a valid interval absent or without finite ends counts as not computed and not covering.
"""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zlib
from fractions import Fraction
from multiprocessing import Pool

import numpy as np

from oe_inferencex import __version__
from oe_inferencex import estimate as est
from oe_inferencex.certify.localized import profile_verdicts

DRAWS = 400
# the --valid cells 0 to 7, in the plan's order
VALID_CELLS = (("random", 300), ("random", 1000), ("confidence", 300), ("confidence", 1000),
               ("proportional", 300), ("proportional", 1000), ("class", 400), ("class", 1000))
CLASS_FLOOR = 50                          # the class design's labels per map class at least
PROFILE_BUDGET = 800
DELTA = 0.1
A_FIELDS = ("confidence:4", "class")      # field index 0 and 1
A_ALPHA_FACTORS = (0.5, 1.0, 2.0)         # alpha = e/2, e, 2e: alpha index 0, 1, 2
B_FIELDS = ("confidence:4", "class", "grid:2,confidence:2")
B_ALPHAS = (0.1, 0.2, 0.3)
E_LOW, E_HIGH = 0.02, 0.5                 # verdict cells only on maps with e in [0.02, 0.5] ...
MIN_UNITS_VERDICT = 1600                  # ... and at least 1,600 units
B_SEED_BASE = 120000000
EXCLUDED = "olmoearth_base_rtx/"
N_PLANNED = 356
TILE_NAME = "dynamic_world_tile"


# ----------------------------------------------------------------------------- shared
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verdict_cell_number(field_index, alpha_index):
    return 100 + 10 * field_index + alpha_index


def cell_seeds(name, cell):
    """Arm A: a cell's 400 seeds, as the plan writes them; a run with fewer draws takes the first ones."""
    ss = np.random.SeedSequence([120, zlib.crc32(name.encode("utf-8")), cell])
    return np.random.default_rng(ss).integers(0, 2**31 - 1, 400)


def as_written(alpha):
    """alpha as written: 0.1 is 1/10, not the binary float nearest to it."""
    return Fraction(repr(float(alpha)))


def new_valid_cell():
    return {"draws": 0, "refused": 0, "valid_covers": 0, "valid_not_computed": 0, "plain_covers": 0}


def new_verdict_cell():
    return {"draws": 0, "refused": 0, "any_wrong": 0, "any_miss": 0}


def grade_valid(cell, res, truth):
    """One graded --valid draw: the interval without --valid (low, high) and the valid one (valid_interval)."""
    cell["draws"] += 1
    if res["low"] <= truth <= res["high"]:
        cell["plain_covers"] += 1
    v = res.get("valid_interval")
    lo = v.get("low") if isinstance(v, dict) else None
    hi = v.get("high") if isinstance(v, dict) else None
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        cell["valid_not_computed"] += 1               # counts as not covering
    elif lo <= truth <= hi:
        cell["valid_covers"] += 1


def grade_verdicts(cell, profiles, truths, sizes, alpha_exact):
    """One graded verdict draw. `profiles` is profile_verdicts' rows by name; truths[h] and sizes[h] are this script's
    exact rate and window count of the profile whose row index is h."""
    cell["draws"] += 1
    wrong = miss = False
    for row in profiles.values():
        h = int(row["index"])
        if int(row["n_population"]) != sizes[h]:
            raise RuntimeError(f"profile {h}: the package counts {row['n_population']} windows, this script {sizes[h]}")
        t = truths[h]
        if Fraction(row["low"]) > t or Fraction(row["high"]) < t:
            miss = True
        if (row["verdict"] == "good" and t > alpha_exact) or (row["verdict"] == "bad" and t <= alpha_exact):
            wrong = True
    cell["any_wrong"] += int(wrong)
    cell["any_miss"] += int(miss)


def check_cell(cell, draws):
    """What every cell must satisfy, whatever the numbers."""
    assert cell["draws"] + cell["refused"] == draws, cell
    assert all(isinstance(v, int) and v >= 0 for v in cell.values()), cell
    for k in ("valid_covers", "valid_not_computed", "plain_covers", "any_wrong", "any_miss"):
        if k in cell:
            assert cell[k] <= cell["draws"], cell
    if "valid_covers" in cell:
        assert cell["valid_covers"] + cell["valid_not_computed"] <= cell["draws"], cell


# ----------------------------------------------------------------------------- arm A
def draw_a1(margin, p1, dec, design, n, seed, class_names):
    """One sample of a --valid cell, through sample_for_estimation as the command line calls it: the class design is
    the condition design with the map class as the condition, named "class c", and class_floor 50."""
    if design == "class":
        s = est.sample_for_estimation(margin, n, design="condition", seed=seed, condition=dec,
                                      condition_names=class_names, class_floor=CLASS_FLOOR)
        s["condition"]["by"] = "map class"              # as the sidecar records it; no number reads it
        return s
    if design == "confidence":
        return est.sample_for_estimation(margin, n, design="confidence", p1=p1, seed=seed)
    return est.sample_for_estimation(margin, n, design=design, seed=seed)


def a2_cells(name, margin, dec, err, e, draws):
    """The verdict cells of one map: for each field, the profiles of build_profiles (the package's defaults: at most
    12 profiles, cells under 100 units folded into other), 800 labels split equally, delta 0.1, alpha e/2, e, 2e."""
    N = margin.size
    out = {}
    for fi, spec in enumerate(A_FIELDS):
        try:
            prof = est.build_profiles(margin.reshape(1, N), np.ones((1, N), bool), est.parse_profile_by(spec),
                                      map_class=dec.reshape(1, N))
        except ValueError:
            prof = None                                 # e.g. more than 12 profiles: every draw is refused
        if prof is not None:
            pid = np.asarray(prof["index"], dtype=np.int64).ravel()
            if (pid < 0).any():
                raise RuntimeError(f"{name}: a unit lies outside every profile")
            P = len(prof["names"])
            n_p = np.bincount(pid, minlength=P)
            k_p = np.bincount(pid[err == 1], minlength=P)
            truth_p = [Fraction(int(k), int(n)) for k, n in zip(k_p, n_p)]
            names = dict(enumerate(prof["names"]))
        for ai, factor in enumerate(A_ALPHA_FACTORS):
            cell = verdict_cell_number(fi, ai)
            alpha = e * factor
            a_exact = as_written(alpha)
            c = new_verdict_cell()
            for seed in cell_seeds(name, cell)[:draws]:
                if prof is None:
                    c["refused"] += 1
                    continue
                try:
                    s = est.sample_for_estimation(margin, PROFILE_BUDGET, design="condition", seed=int(seed),
                                                  condition=pid, condition_names=names)
                    s["condition"]["by"] = "profile"    # as the sidecar records it
                    s["profile"] = prof["block"]
                    res = profile_verdicts(s, err[s["indices"]], alpha, delta=DELTA)
                except ValueError:
                    c["refused"] += 1
                    continue
                values = s["condition"]["values"]       # row index h -> profile id
                grade_verdicts(c, res["profiles"], [truth_p[int(v)] for v in values],
                               [int(n_p[int(v)]) for v in values], a_exact)
            out[str(cell)] = c
    return out


def run_map(job):
    """Arm A on one unit file: (name, {n_units, error_rate, cells})."""
    path, name, draws = job
    z = np.load(path, allow_pickle=False)
    margin = np.asarray(z["margin"], dtype=np.float64).ravel()
    p1 = np.asarray(z["p1"], dtype=np.float64).ravel()
    dec = np.asarray(z["dec"]).astype(np.int64).ravel()
    err = np.asarray(z["err"]).astype(np.int64).ravel()
    N = int(margin.size)
    if not p1.size == dec.size == err.size == N:
        raise RuntimeError(f"{name}: margin, p1, dec and err differ in length")
    if not np.isin(err, (0, 1)).all():
        raise RuntimeError(f"{name}: err is not 0 or 1 per unit")
    if not np.isfinite(margin).all():
        # the plan's truth is over every unit, the package's population over finite margins; they would differ
        raise RuntimeError(f"{name}: {int((~np.isfinite(margin)).sum())} unit(s) have no finite margin")
    K = int(err.sum())
    e = K / N                                           # the truth of the map, the mean of err over its units
    cells = {}
    class_names = {int(c): f"class {int(c)}" for c in np.unique(dec)}
    for cell, (design, n) in enumerate(VALID_CELLS):
        if 2 * n > N:                                   # run only where n is at most half the map's units
            continue
        c = new_valid_cell()
        for seed in cell_seeds(name, cell)[:draws]:
            try:
                s = draw_a1(margin, p1, dec, design, n, int(seed), class_names)
                res = est.estimate_error_rate(s, err[s["indices"]], valid=True)
            except ValueError:
                c["refused"] += 1
                continue
            grade_valid(c, res, e)
        cells[str(cell)] = c
    if E_LOW <= e <= E_HIGH and N >= MIN_UNITS_VERDICT:
        cells.update(a2_cells(name, margin, dec, err, e, draws))
    cells = {k: cells[k] for k in sorted(cells, key=int)}
    return name, {"n_units": N, "error_rate": e, "cells": cells}


def unit_jobs(units, exp116, expect=N_PLANNED):
    """exp116's planned files: the keys of its input_sha256 not under olmoearth_base_rtx/, each checked by sha256."""
    with open(exp116) as f:
        shas = json.load(f)["input_sha256"]
    names = sorted(k for k in shas if not k.startswith(EXCLUDED))
    if expect is not None and len(names) != expect:
        raise SystemExit(f"{exp116} lists {len(names)} unit files outside {EXCLUDED}; the plan fixes {expect}")
    bad, jobs = [], []
    for name in names:
        path = os.path.join(units, *name.split("/"))
        if not os.path.isfile(path):
            bad.append(f"{name}: missing")
        elif sha256(path) != shas[name]:
            bad.append(f"{name}: sha256 differs from exp116's")
        jobs.append((path, name))
    if bad:
        raise SystemExit(f"{len(bad)} unit file(s) refused, first {bad[0]}")
    return jobs


def run_units(units, exp116, out, workers, draws=DRAWS, expect=N_PLANNED):
    jobs = unit_jobs(units, exp116, expect)
    jobs = sorted(jobs, key=lambda j: -os.path.getsize(j[0]))       # the largest first, for the pool's tail
    maps = {}
    with Pool(workers) as pool:
        for name, row in pool.imap_unordered(run_map, [(p, n, draws) for p, n in jobs], chunksize=1):
            maps[name] = row
            print(f"{name}: {row['n_units']} units, e {row['error_rate']:.4f}, {len(row['cells'])} cells",
                  file=sys.stderr, flush=True)
    result = {"route": "second", "package_version": __version__, "draws": draws,
              "maps": {n: maps[n] for n in sorted(maps)}}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}: {len(maps)} maps, {sum(len(m['cells']) for m in maps.values())} cells")


# ----------------------------------------------------------------------------- arm B
def tile_arrays():
    """The package's Dynamic World tile, as exp116's tile() writes it: probabilities float32, NaN where the expert gave
    no label (no-data); the map class is each window's argmax."""
    import oe_inferencex
    z = np.load(os.path.join(os.path.dirname(oe_inferencex.__file__), "sample", "dynamic_world_tile.npz"),
                allow_pickle=False)
    probs = z["probs"].astype(np.float32)
    expert = z["expert"].astype(np.int64)
    probs[:, expert < 0] = np.nan
    dec = np.argmax(np.where(np.isfinite(probs), probs, -np.inf), axis=0)
    return probs, expert, dec


def run_cli(argv):
    """The command line in this process, its output kept quiet; True when it exits 0."""
    from oe_inferencex.cli import main
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            code = main([str(a) for a in argv])
    except SystemExit as exc:
        code = exc.code
    return code is None or code == 0


def write_labels(csv_path, expert_flat, dec_flat):
    """The labels written from the expert map: wrong = 1 where the CSV's map_class is not the expert's class. Returns
    how many rows show a map_class that is not the window's own argmax."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    off = 0
    for r in rows:
        i = int(r["index"])
        shown = int(r["map_class"])
        off += int(shown != int(dec_flat[i]))
        r["wrong"] = "1" if shown != int(expert_flat[i]) else "0"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return off


def population_is_labelled(side, labelled, shape):
    """Whether a sample's population is exactly the expert-labelled windows, as far as its sidecar records it."""
    if [int(v) for v in side.get("grid", [])] != list(shape):
        return False
    if int(side["n_population"]) != int(labelled.sum()):
        return False
    idx = np.asarray(side["indices"], dtype=np.int64)
    if idx.size == 0 or idx.min() < 0 or idx.max() >= labelled.size or not labelled[idx].all():
        return False
    if side.get("strata_of_population") is not None:
        pop = np.sort(np.asarray(side["strata_of_population"], dtype=np.int64))
        if not np.array_equal(pop, np.flatnonzero(labelled)):
            return False
    if side.get("condition_grid") is not None:
        if not np.array_equal(np.asarray(side["condition_grid"], dtype=np.int64) >= 0, labelled):
            return False
    return True


def run_tile_cell(job):
    """One cell of arm B, end to end through the command line: (cell, counts, map-class mismatches, draws whose
    population is not the expert-labelled windows)."""
    cell, kind, what, value, dw, work, draws = job
    _, expert, dec = tile_arrays()
    shape = expert.shape
    expert_f, dec_f = expert.ravel(), dec.ravel()
    labelled = expert_f >= 0
    wrong_true = (dec_f != expert_f) & labelled
    truth = int(wrong_true.sum()) / int(labelled.sum())
    c = new_valid_cell() if kind == "valid" else new_verdict_cell()
    bad_class = bad_pop = 0
    for k in range(draws):
        seed = B_SEED_BASE + 1000 * cell + k
        d = os.path.join(work, f"cell{cell}", f"draw{k}")
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
        csv_path = os.path.join(d, "sample.csv")
        if kind == "valid":
            argv = ["sample", dw, "--budget", value, "--patch", 1, "--seed", seed, "--design", what, "--out", csv_path]
        else:
            argv = ["sample", dw, "--budget", PROFILE_BUDGET, "--patch", 1, "--seed", seed, "--design", "profile",
                    "--profile-by", what, "--out", csv_path]
        if not run_cli(argv):
            c["refused"] += 1
            shutil.rmtree(d)
            continue
        with open(csv_path[:-4] + ".json", encoding="utf-8") as f:
            side = json.load(f)
        bad_pop += int(not population_is_labelled(side, labelled, shape))
        bad_class += write_labels(csv_path, expert_f, dec_f)
        if kind == "valid":
            res_path = os.path.join(d, "estimate.json")
            if run_cli(["estimate", csv_path, "--valid", "--out", res_path]):
                with open(res_path) as f:
                    grade_valid(c, json.load(f), truth)
            else:
                c["refused"] += 1
        else:
            res_path = os.path.join(d, "profiles.json")
            if run_cli(["certify", csv_path, "--alpha", repr(float(value)), "--out", res_path]):
                with open(res_path) as f:
                    res = json.load(f)
                grid = np.asarray(side["condition_grid"], dtype=np.int64)
                truths, sizes = [], []
                for h in range(len(side["condition"]["names"])):
                    m = grid == h
                    if not m.any():
                        raise RuntimeError(f"cell {cell}, draw {k}: profile {h} holds no window")
                    sizes.append(int(m.sum()))
                    truths.append(Fraction(int(wrong_true[m].sum()), int(m.sum())))
                grade_verdicts(c, res["profiles"], truths, sizes, as_written(value))
            else:
                c["refused"] += 1
        shutil.rmtree(d)
    return cell, c, bad_class, bad_pop


def run_tile(out, work, workers, draws=DRAWS):
    os.makedirs(work, exist_ok=True)
    probs, expert, dec = tile_arrays()
    dw = os.path.join(work, "dw.npy")
    np.save(dw, probs)
    labelled = expert >= 0
    N = int(labelled.sum())
    K = int(((dec != expert) & labelled).sum())
    # the plan lists arm B's cells: A1's eight and the verdict cells of three fields at alpha 0.1, 0.2 and 0.3
    jobs = [(cell, "valid", design, n, dw, work, draws) for cell, (design, n) in enumerate(VALID_CELLS)]
    jobs += [(verdict_cell_number(fi, ai), "verdict", spec, alpha, dw, work, draws)
             for fi, spec in enumerate(B_FIELDS) for ai, alpha in enumerate(B_ALPHAS)]
    with Pool(workers) as pool:
        results = pool.map(run_tile_cell, jobs, chunksize=1)
    results.sort(key=lambda r: r[0])
    cells = {str(cell): c for cell, c, _, _ in results}
    bad_class = sum(r[2] for r in results)
    bad_pop = sum(r[3] for r in results)
    result = {"route": "second", "package_version": __version__, "draws": draws,
              "maps": {TILE_NAME: {"n_units": N, "error_rate": K / N, "cells": cells}}}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}: {len(cells)} cells; sampled windows whose CSV map class is not their argmax: {bad_class}; "
          f"draws whose population is not the expert-labelled windows: {bad_pop}")
    if bad_class or bad_pop:
        raise SystemExit("arm B: both counts must be 0 (the plan); the output is written, but the run does not hold")


# ----------------------------------------------------------------------------- smoke
def smoke():
    """Synthetic units and the real tile at a few draws: every path runs (a1 designs, refusals of the class floors
    and of more than 12 profiles, the sha256 check, arm B's commands) and every cell adds up."""
    d = tempfile.mkdtemp(prefix="exp120_crosscheck_smoke_")
    units = os.path.join(d, "units")
    os.makedirs(os.path.join(units, "enc"))
    rng = np.random.default_rng(0)
    # name: (units, classes, error scale); many has 14 classes of ~143 units: the class design at 400 cannot give
    # each its 50 labels, and the class field gives 14 profiles, more than 12
    specs = {"enc/three.npz": (2400, 3, 0.3), "enc/many.npz": (2000, 14, 0.2), "enc/small.npz": (500, 3, 0.3),
             "enc/clean.npz": (1800, 3, 0.01)}
    shas = {}
    for name, (N, C, scale) in specs.items():
        margin = rng.random(N)
        p1 = 0.5 + margin / 2
        dec = rng.permutation(np.arange(N) % C)
        err = (rng.random(N) < scale * (1 - margin)).astype(np.uint8)
        y = np.where(err == 1, (dec + 1) % C, dec)
        path = os.path.join(units, *name.split("/"))
        np.savez(path, margin=margin.astype(np.float32), p1=p1.astype(np.float32), dec=dec.astype(np.uint8),
                 y=y.astype(np.uint8), err=err, n_classes=np.array([C]))
        shas[name] = sha256(path)
    shas[EXCLUDED + "three.npz"] = "0" * 64           # excluded by the plan: never read, never checked
    exp116 = os.path.join(d, "exp116_ab.json")
    with open(exp116, "w") as f:
        json.dump({"input_sha256": shas}, f)
    draws = 3
    out = os.path.join(d, "units.json")
    run_units(units, exp116, out, workers=2, draws=draws, expect=None)
    with open(out) as f:
        r = json.load(f)
    assert list(r) == ["route", "package_version", "draws", "maps"] and r["route"] == "second" and r["draws"] == draws
    assert sorted(r["maps"]) == sorted(k for k in shas if not k.startswith(EXCLUDED))
    for m in r["maps"].values():
        assert set(m) == {"n_units", "error_rate", "cells"}
        for c in m["cells"].values():
            check_cell(c, draws)
    every = {str(c) for c in range(8)} | {str(verdict_cell_number(f, a)) for f in range(2) for a in range(3)}
    assert set(r["maps"]["enc/three.npz"]["cells"]) == every
    assert set(r["maps"]["enc/many.npz"]["cells"]) == every
    assert r["maps"]["enc/small.npz"]["cells"] == {}
    assert set(r["maps"]["enc/clean.npz"]["cells"]) == {"0", "2", "4", "6"}
    many = r["maps"]["enc/many.npz"]["cells"]
    assert many["6"]["refused"] == draws and many["7"]["refused"] == 0
    assert all(many[str(verdict_cell_number(1, a))]["refused"] == draws for a in range(3))
    assert all(r["maps"]["enc/three.npz"]["cells"][k]["refused"] == 0 for k in every)
    bad = dict(shas)
    bad["enc/three.npz"] = "f" * 64
    bad_path = os.path.join(d, "exp116_bad.json")
    with open(bad_path, "w") as f:
        json.dump({"input_sha256": bad}, f)
    try:
        unit_jobs(units, bad_path, expect=None)
    except SystemExit:
        pass
    else:
        raise AssertionError("a unit file whose sha256 differs was not refused")
    tile_out = os.path.join(d, "tile.json")
    run_tile(tile_out, os.path.join(d, "tile_work"), workers=2, draws=2)
    with open(tile_out) as f:
        t = json.load(f)
    cells = t["maps"][TILE_NAME]["cells"]
    assert set(cells) == {str(c) for c in range(8)} | {str(verdict_cell_number(f, a)) for f in range(3)
                                                        for a in range(3)}
    for c in cells.values():
        check_cell(c, 2)
    print(f"smoke: ok ({d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="exp120, second route, written from the plan alone")
    sp = ap.add_subparsers(dest="cmd", required=True)
    u = sp.add_parser("units", help="arm A: exp116's 356 unit files")
    u.add_argument("--units", required=True, help="exp/out/exp79_units")
    u.add_argument("--exp116", required=True, help="exp/out/exp116_ab.json")
    u.add_argument("--out", required=True)
    u.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    t = sp.add_parser("tile", help="arm B: the package's Dynamic World tile through the command line")
    t.add_argument("--out", required=True)
    t.add_argument("--work", required=True)
    t.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    sp.add_parser("smoke", help="synthetic units and the tile at a few draws")
    a = ap.parse_args()
    if __version__ != "1.20.3":
        print(f"warning: the plan runs olmoearth-inferencex 1.20.3; this is {__version__}", file=sys.stderr)
    if a.cmd == "units":
        run_units(a.units, a.exp116, a.out, a.workers)
    elif a.cmd == "tile":
        run_tile(a.out, a.work, a.workers)
    else:
        smoke()
