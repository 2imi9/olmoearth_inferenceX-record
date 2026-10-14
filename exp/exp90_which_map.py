"""exp90: which of two maps is more accurate, labelling only the windows where they differ.

Not preregistered: a check of `estimate.compare_from_disagreement`'s design on the record's own maps before it was
documented. For every task of exp79 and every pair of encoders run on the same units, the full truth gives the true
accuracy difference; over R draws of n labels, the union-bound interval's coverage, how often it names a map and how
often the wrong one, and its median width. The comparison, "random", draws n windows from the whole map at random and
applies the same interval to the differing windows the draw happens to hold (they are a random sample of the D, given
their number), which is what those labels can support; "random_naive" is the earlier analysis, each map's count over
all N windows, kept for the record (review of 3 October 2026: it understated what random labels allow).
The maps are the record's linear probes on Ai2's published embeddings, not map products.

    python exp/exp90_which_map.py          # about ten minutes on a laptop CPU; writes exp/out/exp90_which_map.json
"""
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from oe_inferencex import estimate as est  # noqa: E402

UNITS = os.path.join(ROOT, "exp", "out", "exp79_units")
OUT = os.path.join(ROOT, "exp", "out", "exp90_which_map.json")
NS, R, SEED = (50, 100, 200), 400, 90


def table(n, M):
    """Exact 97.5% intervals for every count a of n drawn from M (the union bound's level for two counts)."""
    return np.array([est.hypergeom_interval(a, n, M, conf=0.975) for a in range(n + 1)])


def cell(rng, delta, N, D, KA, KB, n, design):
    """Coverage, decisions and width over R draws of n labels: among the D differing windows ("disagree"), or from the
    whole map, analysed on the differing windows drawn ("random") or with each count over all N ("random_naive")."""
    if design == "random":
        m = min(n, N)
        draws = rng.multivariate_hypergeometric([KA, KB, D - KA - KB, N - D], m, size=R)
        a, b, k = draws[:, 0], draws[:, 1], draws[:, 0] + draws[:, 1] + draws[:, 2]
        ends = np.array([[*est.hypergeom_interval(int(x), int(kk), D, conf=0.975),
                          *est.hypergeom_interval(int(y), int(kk), D, conf=0.975)] for x, y, kk in zip(a, b, k)])
        lo = (ends[:, 0] - ends[:, 3]) * D / N
        hi = (ends[:, 1] - ends[:, 2]) * D / N
    else:
        M, c0 = (D, D - KA - KB) if design == "disagree" else (N, N - KA - KB)
        m = min(n, M)
        tab = table(m, M)
        draws = rng.multivariate_hypergeometric([KA, KB, c0], m, size=R)
        a, b = draws[:, 0], draws[:, 1]
        lo = (tab[a, 0] - tab[b, 1]) * M / N
        hi = (tab[a, 1] - tab[b, 0]) * M / N
    cover = (lo <= delta + 1e-12) & (delta - 1e-12 <= hi)
    decide = (lo > 0) | (hi < 0)
    right = ((lo > 0) & (delta > 0)) | ((hi < 0) & (delta < 0))
    return {"labels": int(m), "coverage": float(cover.mean()), "decides": float(decide.mean()),
            "decides_right": float(right.mean()), "decides_wrong": float((decide & ~right).mean()),
            "median_width_points": float(np.median(hi - lo) * 100)}


def main():
    rng = np.random.default_rng(SEED)
    encs = sorted(os.listdir(UNITS))
    tasks = sorted(f[:-4] for f in os.listdir(os.path.join(UNITS, "olmoearth_base")))
    rows, t0, inputs = [], time.time(), {}
    for task in tasks:
        maps = {}
        for e in encs:
            p = os.path.join(UNITS, e, task + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                maps[e] = (z["dec"].astype(np.int32), z["y"].astype(np.int32))
                with open(p, "rb") as f:                   # exp79_units is not committed: a reader checks the bytes
                    inputs[f"{e}/{task}"] = hashlib.sha256(f.read()).hexdigest()
        sizes = {e: v[0].size for e, v in maps.items()}
        common = max(set(sizes.values()), key=list(sizes.values()).count)     # anysat runs a coarser grid on two tasks
        maps = {e: v for e, v in maps.items() if sizes[e] == common}
        y = next(iter(maps.values()))[1]
        assert all(np.array_equal(v[1], y) for v in maps.values()), task       # one truth for every encoder
        N = int(y.size)
        for ea, eb in itertools.combinations(sorted(maps), 2):
            da, db = maps[ea][0], maps[eb][0]
            dis = da != db
            D = int(dis.sum())
            KA, KB = int(((da == y) & dis).sum()), int(((db == y) & dis).sum())
            delta = (KA - KB) / N
            row = {"task": task, "a": ea, "b": eb, "N": N, "D": D, "KA": KA, "KB": KB, "delta": delta,
                   "acc_a": float((da == y).mean()), "acc_b": float((db == y).mean())}
            if D:
                for n in NS:
                    for design in ("disagree", "random", "random_naive"):
                        row[f"{design}_{n}"] = cell(rng, delta, N, D, KA, KB, n, design)
            rows.append(row)
        print(f"{task}: {len(maps)} encoders, {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for n in NS:
        for design in ("disagree", "random", "random_naive"):
            k = f"{design}_{n}"
            got = [r[k] for r in rows if k in r]
            summary[k] = {"cells": len(got),
                          "coverage_min": min(g["coverage"] for g in got),
                          "coverage_median": float(np.median([g["coverage"] for g in got])),
                          "decides_right_mean": float(np.mean([g["decides_right"] for g in got])),
                          "decides_wrong_mean": float(np.mean([g["decides_wrong"] for g in got])),
                          "decides_wrong_max": max(g["decides_wrong"] for g in got),
                          "median_width_points": float(np.median([g["median_width_points"] for g in got]))}
    for n in NS:                                   # pairs whose differing windows are all labelled at this budget
        census = [r for r in rows if r["D"] and r["D"] <= n]
        rest = [r for r in rows if r["D"] > n]
        summary[f"census_{n}"] = {
            "pairs_labelled_in_full": len(census),
            "decides_right_mean_in_full": float(np.mean([r[f"disagree_{n}"]["decides_right"] for r in census])),
            "other_pairs": len(rest),
            "decides_right_mean_other": float(np.mean([r[f"disagree_{n}"]["decides_right"] for r in rest])),
            "random_decides_right_mean_other": float(np.mean([r[f"random_{n}"]["decides_right"] for r in rest])),
            "median_width_points_other": float(np.median([r[f"disagree_{n}"]["median_width_points"] for r in rest])),
            "random_median_width_points_other": float(np.median([r[f"random_{n}"]["median_width_points"] for r in rest]))}
    ratio = [r["N"] / r["D"] for r in rows if r["D"]]
    summary["labels_saved_factor_median"] = float(np.median(ratio))
    summary["disagree_share_median"] = float(np.median([r["D"] / r["N"] for r in rows]))
    summary["abs_delta_points_median"] = float(np.median([abs(r["delta"]) for r in rows]) * 100)
    summary["encoders"] = len({r["a"] for r in rows} | {r["b"] for r in rows})
    out = {"experiment": "exp90: which of two maps is more accurate, labelling only where they differ",
           "preregistered": False, "seed": SEED, "draws": R, "budgets": list(NS), "summary": summary, "cells": rows,
           "inputs_sha256": inputs}
    with open(OUT, "w") as f:
        json.dump(out, f)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
