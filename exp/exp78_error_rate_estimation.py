"""exp78: how wrong is this map? A design-based error rate under a review budget.

Preregistration is docs/plan/map_error_estimation.md; this file is the run.

Two stages, deliberately split.

  --stage export    needs the cluster and the embedding cache. Fits exp70's probe on the suite and writes per-unit
                    margin, p1, decision, label, error and packed validity to exp/out/exp78_units/<task>.npz, with
                    a manifest that carries each task's recomputed numbers beside exp70's recorded ones. This is
                    the only part that needs torch, and it ends a standing cost: exp70 committed only summaries,
                    so every reanalysis of the suite has needed a fresh 792 GB embedding load.

  --stage estimate  runs anywhere, numpy only, no torch. The whole Monte Carlo: sampling designs, estimators,
                    coverage and width, and the four preregistered predictions. Re-runnable and auditable.

  --smoke           synthetic units, both stages, no cache and no network.

The estimand is the finite-population window error rate of one map against one reference, so the randomness is the
reviewer's draw and the intervals are design-based. Every unit on these testbeds is labelled, which is the only way
an interval can be graded; labels are withheld and revealed only for sampled units.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import estimate as est, metrics, stats  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
UNITS = os.path.join(OUT, "exp78_units")
R_DRAWS, BUDGETS, HEADLINE, N_STRATA, M_PER_TILE = 2000, (100, 300, 1000), 300, 5, 16
Z95, MIN_PER_STRATUM = est.Z95, est.MIN_PER_STRATUM
# Preregistered thresholds. See docs/plan/map_error_estimation.md; P2's 0.85 was set with a disclosed pilot.
P1_COVER, P2_COVER, P2_DEFF, P3_RATIO, P3_GOOD, P3_BAD, P4_SAVING, P4_HALFWIDTH = 0.93, 0.85, 2.5, 0.92, 0.88, 0.94, 1.8, 0.04
PILOTED = ("mados", "sen1floods11", "pastis_sentinel1", "pastis_sentinel2", "pastis_sentinel1_sentinel2")


# ----------------------------------------------------------------------------- stage: export
def export_task(task, cache, seg):
    """exp70's probe and its per-unit quantities, plus the top probability, without the window embeddings."""
    import torch
    import exp54_multiclass_embeddings as e54
    import exp70_task_suite as e70
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    if not seg:
        import exp51_their_probe as e51
        C = int(max(int(ytr.max()), int(yte.max())) + 1)
        torch.manual_seed(0)
        probe = e51.LinearProbe(in_dim=xtr.shape[-1], out_dim=C).to(e54.DEV).float()
        opt = torch.optim.AdamW(probe.parameters(), lr=e51.LR)
        X, Y = xtr.to(torch.float32), ytr.to(torch.int64)
        n, steps = len(Y), int(np.ceil(len(Y) / e51.BATCH))
        g = torch.Generator().manual_seed(0)
        probe.train()
        for epoch in range(e51.EPOCHS):
            order = torch.randperm(n, generator=g)
            for i in range(steps):
                idx = order[i * e51.BATCH:(i + 1) * e51.BATCH]
                torch.nn.functional.cross_entropy(probe(X[idx].to(e54.DEV))["logits"], Y[idx].to(e54.DEV)).backward()
                e51.adjust_learning_rate(optimizer=opt, epoch=epoch + i / steps, total_epochs=e51.EPOCHS,
                                         warmup_epochs=int(e51.EPOCHS * 0.1), max_lr=e51.LR, min_lr=1.0e-5)
                opt.step(); opt.zero_grad()
        probe.eval()
        P = []
        with torch.no_grad():
            for i in range(0, len(xte), 4096):
                P.append(torch.softmax(probe(xte[i:i + 4096].to(torch.float32).to(e54.DEV))["logits"], 1).cpu().numpy())
        P = np.concatenate(P)
        srt = np.sort(P, axis=1)
        y = yte.numpy().astype(np.int64)
        return {"margin": (srt[:, -1] - srt[:, -2]).astype(np.float32), "p1": srt[:, -1].astype(np.float32),
                "dec": P.argmax(1).astype(np.uint8), "y": y.astype(np.uint8),
                "err": (P.argmax(1) != y).astype(np.uint8), "ok": np.ones(len(y), bool),
                "grid": np.array([len(y)]), "n_classes": C, "patch_px": 1, "family": "classification"}

    # segmentation: exp70's own path, so the baseline reproduces by construction
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))
    probe = e54.train_probe(xtr, ytr, pp, C, e54.TASK_LR.get(task, 0.1), seed=0)
    q = e54.window_quantities(probe, xte, yte, pp, C)
    ok = q["ok"]
    # the top window-mean probability, the one quantity window_quantities does not return
    N, H, W = yte.shape
    Wn = e54.WIN
    hw, ww = H // Wn, W // Wn
    p1 = np.zeros((N, hw, ww), np.float32)
    with torch.no_grad():
        for i in range(0, N, 64):
            x = xte[i:i + 64].to(e54.DEV, dtype=torch.float32)
            n_, h_, w_ = x.shape[:3]
            lg = probe(x)["logits"].reshape(n_, h_, w_, C, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(n_, C, h_ * pp, w_ * pp)
            if lg.shape[-2:] != (H, W):
                lg = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=True)
            p = torch.softmax(lg, dim=1)[..., :hw * Wn, :ww * Wn].reshape(n_, C, hw, Wn, ww, Wn).mean(dim=(3, 5))
            p1[i:i + n_] = p.amax(1).cpu().numpy()
    flat = ok.ravel()
    return {"margin": q["margin"].ravel()[flat].astype(np.float32), "p1": p1.ravel()[flat],
            "dec": q["dec"].ravel()[flat].astype(np.uint8),
            "y": np.zeros(int(flat.sum()), np.uint8),          # the reference class is not needed; err carries it
            "err": q["err"].ravel()[flat].astype(np.uint8), "ok": ok.reshape(-1),
            "grid": np.array([N, hw, ww]), "n_classes": C, "patch_px": pp, "family": "segmentation",
            "pixel_accuracy": q["pixel_accuracy"]}


def cmd_export(args):
    import exp70_task_suite as e70
    os.makedirs(UNITS, exist_ok=True)
    tasks = args.tasks or (e70.TASKS_SEG + e70.TASKS_CLS)
    man = {"model": e70.MODEL, "tasks": {}}
    for t in tasks:
        t0 = time.time()
        seg = t in e70.TASKS_SEG
        print(f"[exp78 export] {t} ({'seg' if seg else 'cls'})", flush=True)
        try:
            d = export_task(t, args.cache, seg)
        except Exception as ex:  # noqa: BLE001
            print(f"  FAILED {t}: {ex!r}", flush=True)
            man["tasks"][t] = {"failed": repr(ex)}
            continue
        np.savez_compressed(os.path.join(UNITS, f"{t}.npz"),
                            margin=d["margin"], p1=d["p1"], dec=d["dec"], y=d["y"], err=d["err"],
                            ok_packed=np.packbits(d["ok"]), ok_len=np.array([d["ok"].size]),
                            grid=d["grid"], n_classes=np.array([d["n_classes"]]),
                            patch_px=np.array([d["patch_px"]]), family=np.array([d["family"]]))
        err = d["err"].astype(float)
        rec = e70_recorded(t)
        man["tasks"][t] = {"n_units": int(err.size), "error_rate": float(err.mean()),
                           "accuracy": float(1 - err.mean()),
                           "margin_excess_aurc": float(metrics.excess_aurc(-d["margin"].astype(np.float64), err)),
                           "exp70_accuracy": rec.get("test_accuracy"), "exp70_n_units": rec.get("n_units"),
                           "exp70_margin_excess_aurc": rec.get("excess_aurc"),
                           "seconds": time.time() - t0}
        m = man["tasks"][t]
        if m["exp70_accuracy"] is not None:
            m["accuracy_minus_exp70"] = m["accuracy"] - m["exp70_accuracy"]
            m["excess_aurc_minus_exp70"] = m["margin_excess_aurc"] - m["exp70_margin_excess_aurc"]
        print(f"  n={m['n_units']} err={m['error_rate']:.4f} acc-exp70={m.get('accuracy_minus_exp70', float('nan')):+.2e}", flush=True)
    with open(os.path.join(UNITS, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(f"wrote {UNITS}/manifest.json for {len(man['tasks'])} tasks")
    return 0


def e70_recorded(task):
    p = os.path.join(OUT, "exp70_summary.json")
    if not os.path.exists(p):
        return {}
    T = json.load(open(p))["results"]["tasks"]
    if task not in T:
        return {}
    r = T[task]
    return {"test_accuracy": r["test_accuracy"], "n_units": r["n_units"],
            "excess_aurc": r["signals"]["margin"]["excess_aurc"]}


# ----------------------------------------------------------------------------- stage: estimate
# The estimators live in the package since 2026-09-22 (oe_inferencex.estimate), so the code a user runs is the
# code this experiment graded; the names below are the ones this file and exp79 use. Re-running --stage estimate
# after the move reproduced every coverage, half-width and verdict in exp/out/exp78_summary.json byte for byte.
# One design effect moved: MADOS, 9.774639 -> 9.774605, because the package's intra-cluster correlation takes its
# grand mean over the units in tiles of two or more windows, the same units its between- and within-tile sums use,
# where the copy that lived here took it over all units; MADOS has single-window tiles. No recorded digit changes
# (the record cites 9.77), and the committed summary is the package's output.
wilson = est.wilson_interval
strata_of = est.confidence_strata
stratified_interval = est.stratified_interval
neyman = est.neyman_allocation
draw_stratified = est.draw_stratified


def run_designs(err, margin, p1, tile, B, rng, theta):
    """One task, one budget: every design and estimator, R draws each. Returns coverage and mean half-width."""
    N = err.size
    s = strata_of(margin)
    sizes = np.bincount(s, minlength=N_STRATA)
    out = {}

    def tally(name, fn):
        # Bias beside coverage. The truth is known here, so the mean estimate costs nothing, and coverage alone
        # cannot tell a sound estimator from a biased one inside an inflated interval: the cluster arm covered
        # 0.904 on MADOS while estimating 1.78 times the truth, and nothing here recorded that until 2026-09-22.
        cov = wid = tot = 0.0
        starve = 0
        for _ in range(R_DRAWS):
            est_, lo, hi, st = fn()
            cov += lo <= theta <= hi
            wid += (hi - lo) / 2
            tot += est_
            starve += st
        out[name] = {"coverage": cov / R_DRAWS, "half_width": wid / R_DRAWS,
                     "starved_strata_rate": starve / R_DRAWS,
                     "mean_estimate": tot / R_DRAWS, "bias_ratio": (tot / R_DRAWS) / theta if theta > 0 else float("nan")}

    def srs():
        i = rng.choice(N, B, replace=False)
        lo, hi = wilson(err[i].sum(), B, N)
        return float(err[i].mean()), lo, hi, 0

    tally("D1/E1", srs)

    def prop():
        alloc = neyman(sizes, np.ones(N_STRATA), B)
        i = draw_stratified(rng, s, sizes, alloc)
        return stratified_interval(err, s, i, sizes, N)

    tally("D2p/E1", prop)

    # Neyman from the model's own confidence: no labels spent estimating the stratum rates
    q = np.array([(1 - p1[s == h]).mean() if (s == h).any() else 0.0 for h in range(N_STRATA)])
    tally("D2c/E1", lambda: stratified_interval(
        err, s, draw_stratified(rng, s, sizes, neyman(sizes, np.sqrt(np.clip(q * (1 - q), 1e-9, None)), B)), sizes, N))

    # the ceiling: Neyman from the true stratum rates. Not deployable, reported so a shortfall can be decomposed
    ptrue = np.array([err[s == h].mean() if (s == h).any() else 0.0 for h in range(N_STRATA)])
    tally("D2star/E1", lambda: stratified_interval(
        err, s, draw_stratified(rng, s, sizes, neyman(sizes, np.sqrt(np.clip(ptrue * (1 - ptrue), 1e-9, None)), B)), sizes, N))

    # post-stratification: a plain random draw, re-weighted to the known stratum sizes
    def postrat():
        i = rng.choice(N, B, replace=False)
        return stratified_interval(err, s, i, sizes, N)

    tally("D1+E3ps", postrat)

    # the reviewer who labels whole tiles, then computes the obvious interval
    if tile is not None:
        tiles = np.unique(tile)
        by = {t: np.flatnonzero(tile == t) for t in tiles}
        T = max(1, B // M_PER_TILE)
        tile_sizes = np.bincount(tile)                            # valid windows per tile, the ratio estimator's weights

        def cluster(naive):
            pick = rng.choice(tiles, min(T, tiles.size), replace=False)
            i = np.concatenate([rng.choice(by[t], min(M_PER_TILE, by[t].size), replace=False) for t in pick])
            if naive:
                lo, hi = wilson(err[i].sum(), i.size, N)
                return float(err[i].mean()), lo, hi, 0
            # The ratio estimator, weighted by each tile's valid windows, with the normal quantile this arm was
            # graded under. Until 2026-09-22 it was the unweighted mean of tile means: identical on six tasks, whose
            # tiles are of equal size, and 1.78 times the true rate on MADOS, whose tiles hold 1 to 400 windows.
            est_, lo, hi, _, _ = est.cluster_interval(err, tile, i, tile_sizes=tile_sizes, quantile="normal")
            return est_, lo, hi, 0

        tally("D4/E2_naive", lambda: cluster(True))
        tally("D4/E1_cluster", lambda: cluster(False))
    return out


design_effect = est.design_effect


def load_units(task, units_dir=UNITS):
    """One task's per-unit export; `units_dir` lets exp79 read another encoder's export in the same format."""
    d = np.load(os.path.join(units_dir, f"{task}.npz"), allow_pickle=False)
    ok = np.unpackbits(d["ok_packed"])[:int(d["ok_len"][0])].astype(bool)
    grid = d["grid"]
    tile = None
    if grid.size == 3:
        N, hw, ww = (int(v) for v in grid)
        tile = np.broadcast_to(np.arange(N)[:, None, None], (N, hw, ww)).reshape(-1)[ok]
    return {"margin": d["margin"].astype(np.float64), "p1": d["p1"].astype(np.float64),
            "err": d["err"].astype(np.float64), "tile": tile, "family": str(d["family"][0])}


def cmd_estimate(args):
    import exp70_task_suite as e70
    man = json.load(open(os.path.join(UNITS, "manifest.json")))
    tasks = args.tasks or [t for t in e70.TASKS_SEG if "failed" not in man["tasks"].get(t, {"failed": 1})]
    rows = {}
    for t in tasks:
        u = load_units(t)
        err, theta, N = u["err"], float(u["err"].mean()), u["err"].size
        rng = np.random.default_rng(0)
        rec = {"n_units": N, "error_rate": theta, "budgets": {},
               "design_effect": design_effect(err, u["tile"]) if u["tile"] is not None else float("nan"),
               "exp70_accuracy": man["tasks"][t].get("exp70_accuracy"),
               "accuracy_minus_exp70": man["tasks"][t].get("accuracy_minus_exp70")}
        for B in (BUDGETS if args.all_budgets else (HEADLINE,)):
            if B >= N:
                continue
            rec["budgets"][str(B)] = run_designs(err, u["margin"], u["p1"], u["tile"], B, rng, theta)
        rows[t] = rec
        b = rec["budgets"].get(str(HEADLINE), {})
        print(f"[exp78] {t:28s} theta {theta:.4f} deff {rec['design_effect']:5.2f} "
              f"SRS cover {b.get('D1/E1', {}).get('coverage', float('nan')):.3f} "
              f"naive cover {b.get('D4/E2_naive', {}).get('coverage', float('nan')):.3f}", flush=True)
    summary = {"experiment": "exp78", "config": {"R": R_DRAWS, "headline_budget": HEADLINE, "strata": N_STRATA,
                                                 "m_per_tile": M_PER_TILE, "piloted": list(PILOTED)},
               "tasks": rows, "prereg": verdicts(rows)}
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp78_summary{tag}.json"), "w") as f:
        json.dump(jsonable(summary), f, indent=1)
    print(json.dumps(jsonable(summary["prereg"]), indent=1))
    return 0


def jsonable(o):
    """JSON-safe view: numpy scalars become Python ones and NaN becomes null, as assess.summary does."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def verdicts(rows):
    n = len(rows)
    h = str(HEADLINE)
    get = lambda r, k, f: r["budgets"].get(h, {}).get(k, {}).get(f, float("nan"))
    p1 = [t for t, r in rows.items() if get(r, "D1/E1", "coverage") >= P1_COVER]
    p2 = [t for t, r in rows.items() if get(r, "D4/E2_naive", "coverage") < P2_COVER]
    p2_oos = [t for t in p2 if t not in PILOTED]
    deffs = [r["design_effect"] for r in rows.values() if np.isfinite(r["design_effect"])]
    ratios, cover_ok = {}, {}
    for t, r in rows.items():
        base = get(r, "D1/E1", "half_width")
        cands = {k: get(r, k, "half_width") for k in ("D2p/E1", "D2c/E1", "D1+E3ps")}
        best = min(cands, key=lambda k: cands[k]) if cands else None
        ratios[t] = cands[best] / base if best and base > 0 else float("nan")
        cover_ok[t] = get(r, best, "coverage") >= P1_COVER if best else False
    good = [t for t, r in rows.items() if r["error_rate"] < 0.10]
    bad = [t for t, r in rows.items() if r["error_rate"] > 0.30]
    med_ratio = float(np.median([v for v in ratios.values() if np.isfinite(v)])) if ratios else float("nan")
    savings = {t: (1 / v ** 2 if np.isfinite(v) and v > 0 else float("nan")) for t, v in ratios.items()}
    halfw = {t: get(r, "D1/E1", "half_width") for t, r in rows.items()}
    return {
        "P1_the_interval_is_honest": {"holds": len(p1) == n, "n": len(p1), "of": n, "tasks_failing": [t for t in rows if t not in p1],
                                      "threshold": f"coverage >= {P1_COVER} on every task at B = {HEADLINE}"},
        "P2_the_naive_interval_is_badly_wrong": {
            "holds": len(p2) >= 5 and float(np.median(deffs)) >= P2_DEFF if deffs else False,
            "n": len(p2), "of": n, "n_out_of_sample": len(p2_oos), "out_of_sample_tasks": p2_oos,
            "median_design_effect": float(np.median(deffs)) if deffs else float("nan"),
            "threshold": f"naive coverage < {P2_COVER} on at least 5 of 7 and median design effect >= {P2_DEFF}; "
                         f"piloted tasks are not confirmation"},
        "P3_confidence_helps_most_where_the_map_is_good": {
            "holds": bool(med_ratio <= P3_RATIO and all(cover_ok.values())
                          and all(ratios.get(t, 9) <= P3_GOOD for t in good) and any(ratios.get(t, 0) > P3_BAD for t in bad)),
            "median_width_ratio": med_ratio, "per_task_ratio": ratios,
            "low_error_tasks": good, "high_error_tasks": bad, "coverage_kept": cover_ok,
            "threshold": f"median ratio <= {P3_RATIO}, <= {P3_GOOD} on tasks under 10% error, > {P3_BAD} on one over 30%"},
        "P4_no_order_of_magnitude": {
            "holds": bool(all(not np.isfinite(v) or v <= P4_SAVING for v in savings.values())
                          and all(halfw[t] >= P4_HALFWIDTH for t, r in rows.items() if r["error_rate"] > 0.20)),
            "max_budget_saving": float(np.nanmax(list(savings.values()))) if savings else float("nan"),
            "half_widths": halfw,
            "threshold": f"no saving above {P4_SAVING}x and no half-width below {P4_HALFWIDTH} where error rate > 0.20"},
    }


def cmd_smoke(args):
    """Synthetic units through both stages: no cache, no network, no torch."""
    os.makedirs(UNITS, exist_ok=True)
    rng = np.random.default_rng(0)
    for t, (N, hw, ww, rate) in {"smoke_a": (60, 8, 8, 0.08), "smoke_b": (40, 8, 8, 0.33)}.items():
        n = N * hw * ww
        tile_effect = rng.normal(0, 1.0, N).repeat(hw * ww)          # real clustering, so the deff is not 1
        err = (rng.random(n) < np.clip(rate + 0.10 * tile_effect, 0.01, 0.95)).astype(np.uint8)
        margin = rng.random(n) - 0.8 * err
        np.savez_compressed(os.path.join(UNITS, f"{t}.npz"),
                            margin=margin.astype(np.float32), p1=(0.5 + margin / 4).astype(np.float32),
                            dec=np.zeros(n, np.uint8), y=np.zeros(n, np.uint8), err=err,
                            ok_packed=np.packbits(np.ones(n, bool)), ok_len=np.array([n]),
                            grid=np.array([N, hw, ww]), n_classes=np.array([2]), patch_px=np.array([4]),
                            family=np.array(["segmentation"]))
    with open(os.path.join(UNITS, "manifest.json"), "w") as f:
        json.dump({"model": "smoke", "tasks": {t: {"n_units": 0} for t in ("smoke_a", "smoke_b")}}, f)
    args.tasks = ["smoke_a", "smoke_b"]
    return cmd_estimate(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("export", "estimate"), default="estimate")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME"))
    ap.add_argument("--all-budgets", action="store_true")
    ap.add_argument("--units-dir", default=None, help="read the per-unit export from here instead of exp/out/exp78_units")
    args = ap.parse_args()
    if args.units_dir:
        globals()["UNITS"] = os.path.abspath(args.units_dir)
    if args.smoke:
        return cmd_smoke(args)
    return cmd_export(args) if args.stage == "export" else cmd_estimate(args)


if __name__ == "__main__":
    sys.exit(main())
