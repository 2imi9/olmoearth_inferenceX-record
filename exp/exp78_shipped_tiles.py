"""exp78 follow-up: the coverage of the tiles design as the package ships it, on exp78's seven segmentation tasks.

exp78 graded its own tile design, D4: exactly budget // per_tile tiles (18 at 300 labels and 16 per tile), under the
ordinary formula (D4/E2_naive) and under a cluster ratio estimator with a normal quantile (D4/E1_cluster). The package
ships a different design. `sample_for_estimation(design="tiles")` takes tiles in a random order, up to `per_tile`
windows from each, until the budget is met, and `estimate_error_rate` gives the ratio estimator's interval with a t
quantile on (tiles - 1) degrees of freedom. This script grades that shipped path, unchanged, on the same per-unit
files exp78 graded (exp/out/exp78_units, OlmoEarth Base, exp70's probes).

    python exp/exp78_shipped_tiles.py [--draws 4000]     # writes exp/out/exp78_shipped_tiles.json

For each task and draw r in 0 .. draws-1: s = sample_for_estimation(margin, 300, design="tiles", tiles=tile,
per_tile=16, seed=r), then estimate_error_rate(s, err[s["indices"]]); the draw covers when low <= the map's true error
rate <= high. Reported per task: the coverage with its Monte Carlo standard error, the mean half-width, the mean
estimate over the truth, the tiles and labels used, the tile sizes, how unevenly the errors sit across tiles, and
exp78's recorded D4 coverages beside it. Not preregistered: a check of the shipped code, prompted by the red team
of 30 September 2026. numpy only, a few minutes on a laptop CPU.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import oe_inferencex                                   # noqa: E402
from oe_inferencex import estimate as est              # noqa: E402
import exp78_error_rate_estimation as e78              # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
TASKS = ("mados", "sen1floods11", "pastis_sentinel1", "pastis_sentinel2", "pastis_sentinel1_sentinel2",
         "m_cashew_plant", "m_sa_crop_type")
BUDGET, PER_TILE = 300, 16


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grade(task, draws, budget=BUDGET, per_tile=PER_TILE):
    u = e78.load_units(task)
    err, margin, tile = u["err"], u["margin"], np.ascontiguousarray(u["tile"])
    theta = float(err.mean())
    sizes = np.bincount(tile)
    sizes = sizes[sizes > 0]
    tile_err = np.bincount(tile, weights=err)
    tile_err = tile_err[np.bincount(tile) > 0]
    top = np.sort(tile_err)[::-1][:max(1, int(round(0.1 * tile_err.size)))]
    cover, half, estimate, n_tiles, n_lab = 0, [], [], [], []
    t0 = time.time()
    for r in range(draws):
        s = est.sample_for_estimation(margin, budget, design="tiles", tiles=tile, per_tile=per_tile, seed=r)
        o = est.estimate_error_rate(s, err[s["indices"]])
        cover += o["low"] <= theta <= o["high"]
        half.append(o["half_width"])
        estimate.append(o["estimate"])
        n_tiles.append(o["n_tiles"])
        n_lab.append(o["n_labelled"])
    c = cover / draws
    return {"n_units": int(err.size), "error_rate": theta,
            "tiles": {"n": int(sizes.size), "valid_windows_min": int(sizes.min()),
                      "valid_windows_median": float(np.median(sizes)), "valid_windows_max": int(sizes.max()),
                      "share_under_per_tile": float((sizes < per_tile).mean()),
                      "share_with_no_error": float((tile_err == 0).mean()),
                      "share_of_errors_in_top_tenth_of_tiles": float(top.sum() / tile_err.sum())},
            "shipped": {"coverage": c, "covered": int(cover), "draws": draws,
                        "mc_standard_error": float(np.sqrt(c * (1 - c) / draws)),
                        "mean_half_width": float(np.mean(half)),
                        "mean_estimate_over_truth": float(np.mean(estimate) / theta),
                        "tiles_used_median": float(np.median(n_tiles)), "tiles_used_min": int(min(n_tiles)),
                        "tiles_used_max": int(max(n_tiles)),
                        "labels_min": int(min(n_lab)), "labels_max": int(max(n_lab))},
            "seconds": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    a = ap.parse_args(argv)
    rec = json.load(open(os.path.join(OUT, "exp78_summary.json")))["tasks"]
    out = {"experiment": "exp78 follow-up: the shipped tiles design graded on exp78's per-unit files",
           "config": {"budget": BUDGET, "per_tile": PER_TILE, "draws": a.draws, "seeds": f"0 to {a.draws - 1}",
                      "design": "sample_for_estimation(design='tiles'), tiles in a random order until the budget is met",
                      "interval": "estimate_error_rate: ratio estimator over tiles, ultimate-cluster variance, t quantile",
                      "package_version": oe_inferencex.__version__,
                      "estimate_py_sha256": _sha256(os.path.join(ROOT, "oe_inferencex", "estimate.py")),
                      "units": "exp/out/exp78_units (OlmoEarth Base, exp70's probes, seed 0)"},
           "tasks": {}}
    for t in a.tasks:
        r = grade(t, a.draws)
        b = rec[t]["budgets"][str(BUDGET)]
        r["exp78_recorded"] = {"D4/E1_cluster": b["D4/E1_cluster"]["coverage"], "D4/E2_naive": b["D4/E2_naive"]["coverage"],
                               "tiles": BUDGET // PER_TILE, "draws": 2000}
        out["tasks"][t] = r
        s = r["shipped"]
        print(f"{t:28s} coverage {s['coverage']:.4f} (se {s['mc_standard_error']:.4f}), half-width {s['mean_half_width']:.4f}, "
              f"tiles {s['tiles_used_median']:.0f} [{s['tiles_used_min']}, {s['tiles_used_max']}], labels "
              f"{s['labels_min']}-{s['labels_max']}; exp78 D4 cluster {r['exp78_recorded']['D4/E1_cluster']:.3f}, "
              f"naive {r['exp78_recorded']['D4/E2_naive']:.3f}; {r['seconds']} s", flush=True)
    path = os.path.join(OUT, "exp78_shipped_tiles.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
