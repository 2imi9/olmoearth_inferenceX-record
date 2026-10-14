#!/usr/bin/env python
"""exp97's pre-record audit, made reproducible (8 October 2026; after the result, not preregistered).

    python exp/exp97_audit.py reseed DIR SEED [SEED ...]   # exp97's grading at other master seeds, one JSON per seed
    python exp/exp97_audit.py digest DIR                   # P1, P3 and P4 at each reseeded run, beside exp97's own

**Why.** exp96's ramp passed its preregistered bar at its seed and failed it at 4 of 8 others (exp96_audit.py). P4,
exp97's cost prediction, is a median over cells of a ratio read off mean curves, so it is checked the same way:
exp97's own run_cell, at master seeds other than 97 (the cell seed is SeedSequence([master, crc32(cell name)])), all
else as preregistered, without the implementation check (P2), which exp97 ran. The reseeded files stay local; their
sha256 is recorded in exp/out/exp97_audit.json.
"""
import glob
import hashlib
import json
import os
import sys
import zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))
import exp95_level_cut as e95  # noqa: E402
import exp97_sequential as e97  # noqa: E402

OUT = os.path.join(ROOT, "exp", "out")
AUDIT = os.path.join(OUT, "exp97_audit.json")


def _items():
    held, _ = e97.heldout_as_recorded()
    return [(f"exp95:{k}", v) for k, v in e95.cells().items()] + [(f"exp96:{k}", v) for k, v in held.items()]


def _job(arg):
    master, name, err_o = arg
    return name, e97.run_cell(err_o, seed=np.random.SeedSequence([master, zlib.crc32(name.encode())]), check=0)


def reseed(directory, seeds):
    os.makedirs(directory, exist_ok=True)
    items = _items()
    for master in seeds:
        rows = {}
        with Pool(int(os.environ.get("EXP97_WORKERS", "6"))) as pool:
            for name, r in pool.imap_unordered(_job, [(master, k, v) for k, v in items]):
                rows[name] = r
        rows = dict(sorted(rows.items()))
        path = os.path.join(directory, f"seed{master}.json")
        json.dump({"master_seed": master, "cells": rows}, open(path, "w"), default=float)
        print(master, "written", path, flush=True)


def _digest_rows(rows):
    a = e97.analyze(rows)
    keep = {"P1": {k: a["P1"][k] for k in ("holds", "max_violation", "rates_above_3se")},
            "P3": {k: a["P3"][k] for k in ("holds", "cells_F_pathwise_above_bound", "of")},
            "P4": a["P4"],
            "median_area": {k: v["median_area"] for k, v in a["descriptive"].items()}}
    return keep


def digest(directory):
    out = json.load(open(AUDIT)) if os.path.exists(AUDIT) else {}
    main_path = os.path.join(OUT, "exp97_summary.json")
    seeds = {"97": {**_digest_rows(json.load(open(main_path))["cells"]), "file": "exp97_summary.json",
                    "sha256": hashlib.sha256(open(main_path, "rb").read()).hexdigest()}}
    for f in sorted(glob.glob(os.path.join(directory, "seed*.json"))):
        s = os.path.basename(f)[4:-5]
        seeds[s] = {**_digest_rows(json.load(open(f))["cells"]), "sha256": hashlib.sha256(open(f, "rb").read()).hexdigest()}
    out["by_seed"] = seeds
    out["p4_holds_on"] = f"{sum(v['P4']['holds'] for v in seeds.values())} of {len(seeds)} seeds"
    json.dump(out, open(AUDIT, "w"), indent=1, default=float)
    for k, v in seeds.items():
        print(k, "P1", v["P1"]["holds"], round(v["P1"]["max_violation"], 4), "P4 median rho", v["P4"]["median_rho"],
              "holds", v["P4"]["holds"], "cells", v["P4"]["cells"], "quartiles", v["P4"]["rho_quartiles_inverted_cdf"])


if __name__ == "__main__":
    if sys.argv[1:2] == ["reseed"]:
        reseed(sys.argv[2], [int(x) for x in sys.argv[3:]])
    elif sys.argv[1:2] == ["digest"]:
        digest(sys.argv[2])
    else:
        raise SystemExit(__doc__)
