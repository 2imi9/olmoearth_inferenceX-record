#!/usr/bin/env python
"""exp50: what carries the fusion that beat confidence, does the bag replicate, and does the fusion reach the tile level.

Why. exp49 left three loose ends. (1) A linear logistic combiner of ten label-free signals, fitted on the head's own
training split, beat the model's confidence on three of four arms (E-AURC 0.0067 vs 0.0094, 0.0068 vs 0.0109,
0.0089 vs 0.0116), the first fusion in this repository to do so, but its weights were not saved, so which signals
carry it is unknown. (2) The predictive entropy of eight tile-bootstrap heads beat confidence on Bolivia under v1
(210/103 tiles, p = 1.5e-9) with one bootstrap draw; a single seed is not a finding. (3) The fusion's window-level
gain did not carry to the tile level, where it was never fitted. Issue 5 (shift-label entropy, the semantic-entropy
port of the tiling perturbation) is also closed here at zero marginal cost: the four shifted hard decisions of the
reference head are painted to pixels like tile-phase and their per-pixel binary entropy pooled to the window.

Design. exp49's data path, heads, W1 error set, signals and scoring, imported unchanged (same seeds, so the
exp49 rankers reproduce to the digit), plus:
  weights      the linear fusion's standardized coefficients per signal, recorded;
  drop-one     the linear fusion refitted ten times leaving one signal out, pooled E-AURC per arm: a signal whose
               removal costs the most carries the fusion;
  bag16        sixteen tile-bootstrap heads with a fresh seed (50) beside exp49's eight (seed 49); rankers
               "bag16 predictive entropy" and "bag16 mutual information";
  shift-label entropy   the issue-5 ranker, graded like every other;
  tile-fitted fusion    the same ten signals averaged over each training tile's predicted-water ROI, a linear
               logistic head fitted against the tile failure (water F1 < 0.6, exp49's rule), applied to the
               testbeds' tiles; tile AURC and Risk@0.5 / 0.75, with a 2000-draw tile bootstrap of the AURC
               difference against confidence's tile score.

Preregistered, one-sided, stated before the run:
  P1  under v1 on Bolivia, bag16 predictive entropy beats the averaged confidence: pooled lead <= -0.001 and the
      per-tile exact sign test (bag better) p < 0.05. Falsification: exp49's bag result was a seed artefact and its
      ledger row becomes "not supported".
  P2  under v1.2 on Bolivia, the tile-fitted fusion has a lower tile-level AURC than confidence's ROI-averaged
      score, with the 95% tile-bootstrap interval of the difference excluding zero. Falsification: the fusion's
      gain is a window-level property and the tile-level claim is dropped.
Secondary, descriptive: the weights and the drop-one table; shift-label entropy against tile-phase; bag16 mutual
information; the tile-fitted fusion on the other three arms.

Inputs: data/floods/, exp/out/exp18_feats.npz. Outputs: exp/out/exp50_summary.json, exp/out/exp50_fusion_ablation.csv.
Run with ~/oe12/.venv. --smoke: CPU, 4 tiles per split, v1 only, _smoke outputs.
"""
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp49_shrug_signals as e49  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected, oracle_aurc, selective_accuracy  # noqa: E402
from oe_inferencex.reliability import bootstrap_tiles, ensemble_uncertainty, kmeans, standardize  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS, PAD = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS, exp18.PAD
W1C, SIGNALS, FUSE_LIN = e49.W1C, e49.SIGNALS, e49.FUSE_LIN
BAG_PE, BAG_MI, SLE = "bag16 predictive entropy", "bag16 mutual information", "shift-label entropy"
TILE_FUSE = "tile-fitted fusion (labelled)"
N_BAG, BAG_SEED, MIN_LEAD, N_BOOT = 16, 50, 0.001, 2000


def member_heads(tr_feats, tr_lab, n, seed):
    """exp49's ensemble with the member count and bootstrap seed as parameters (member i: torch seed 1000 * seed + i)."""
    y, ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    flat, heads = tr_feats.reshape(-1, tr_feats.shape[-1]), []
    for i, idx in enumerate(bootstrap_tiles(len(tr_feats), n, seed=seed)):
        rows, keep = (idx[:, None] * (G * G) + np.arange(G * G)[None, :]).ravel(), ok[idx].ravel()
        torch.manual_seed(1000 * seed + i)
        heads.append(train_logistic_head(torch.tensor(flat[rows[keep]]), y[idx].ravel()[keep]))
    return heads


def shift_label_entropy(p_shift):
    """(S, N, G, G) reference maps -> per-window mean of the per-pixel binary entropy of the S aligned hard decisions."""
    S, N = p_shift.shape[:2]
    canvas = np.full((S, N, CROP + PAD, CROP + PAD), np.nan, dtype=np.float32)
    for s in range(S):
        canvas[s, :, s:s + CROP, s:s + CROP] = np.kron((p_shift[s] > 0.5).astype(np.float32), np.ones((PATCH, PATCH), dtype=np.float32))
    f = np.clip(np.nanmean(canvas, axis=0), 1e-7, 1 - 1e-7)                 # fraction of tilings calling the pixel water
    h = -(f * np.log(f) + (1 - f) * np.log(1 - f))
    return h[:, :CROP, :CROP].reshape(N, G, PATCH, G, PATCH).mean(axis=(2, 4))


def fit_linear(sig, target, ok, keys):
    """A linear logistic combiner of the named signals on the rows `ok`, standardized there; returns mu, sd, w, b."""
    Z, mu, sd = standardize(np.stack([np.asarray(sig[k])[ok] for k in keys], 1))
    torch.manual_seed(0)
    w, b = train_logistic_head(torch.tensor(Z, dtype=torch.float32), np.asarray(target)[ok])
    return {"keys": list(keys), "mu": mu, "sd": sd, "w": w.numpy(), "b": float(b)}


def apply_linear(sig, fit):
    shape = np.asarray(sig[fit["keys"][0]]).shape
    Z, _, _ = standardize(np.stack([np.asarray(sig[k]) for k in fit["keys"]], -1).reshape(-1, len(fit["keys"])), fit["mu"], fit["sd"])
    return (Z @ fit["w"] + fit["b"]).reshape(shape)


def tile_rows(sig, w1, yy, ok, keys):
    """Per tile with a labelled window: the ROI-averaged signals (exp49's rule) and the failure indicator (water F1 < 0.6)."""
    X, fail = [], []
    for t in range(len(w1)):
        m = ok[t]
        if not m.any():
            continue
        pred, truth = w1[t][m] > 0.5, yy[t][m] > 0.5
        tp, fp, fn = float((pred & truth).sum()), float((pred & ~truth).sum()), float((~pred & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1.0
        roi = m & (w1[t] >= 0.5)
        roi = roi if roi.any() else m
        X.append([float(np.mean(np.asarray(sig[k][t])[roi])) for k in keys])
        fail.append(float(f1 < 0.6))
    return np.array(X), np.array(fail)


def tile_scores(score, fail, ref, rng):
    """Tile AURC, Risk@0.5/0.75 and the tile-bootstrap interval of AURC(score) - AURC(ref)."""
    acc = selective_accuracy(score, 1 - fail, coverages=(0.5, 0.75))
    n = len(fail)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if 0 < fail[i].sum() < n:
            diffs.append(aurc_expected(score[i], fail[i]) - aurc_expected(ref[i], fail[i]))
    diffs = np.array(diffs)
    return {"aurc": aurc_expected(score, fail), "risk_at_0.5": 1 - acc[0.5], "risk_at_0.75": 1 - acc[0.75],
            "aurc_diff_vs_confidence": aurc_expected(score, fail) - aurc_expected(ref, fail),
            "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if len(diffs) else None, "n_boot_valid": int(len(diffs))}


def pooled_eaurc(v, err, ok):
    return aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum()))


def per_tile_test(sig_a, sig_b, err, ok, alternative):
    """Per-tile excess AURC of a against b (positive: a better) with an exact sign test."""
    tiles = [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]
    g = np.array([(aurc_expected(np.asarray(sig_b[t])[ok[t]], err[t][ok[t]]) - aurc_expected(np.asarray(sig_a[t])[ok[t]], err[t][ok[t]])) for t in tiles])
    w, l, t_ = wins_losses_ties(g)
    return {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, alternative), "median_gain": float(np.median(g)) if len(g) else None, "n_tiles": len(tiles)}


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp50 fusion anatomy, bag replication, shift-label entropy, tile-fitted fusion", "smoke": args.smoke,
               "config": {"fusion_inputs": SIGNALS, "bag_members": N_BAG, "bag_seed": BAG_SEED, "min_lead": MIN_LEAD, "n_boot": N_BOOT,
                          "prereg": "P1 v1/bolivia: bag16 predictive entropy beats averaged confidence (pooled lead <= -0.001, one-sided per-tile sign test); "
                                    "P2 v1_2/bolivia: tile-fitted fusion beats confidence's ROI-averaged score on tile AURC with a 95% tile-bootstrap interval excluding zero"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, _ = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
    n_tr = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr_s2, tr_lab = hb.load_floods_split(paths["valid"], n_tr)
    bo = hb.load_floods_split(paths["bolivia"])
    te_path = os.path.join(floods_dir, "flood_test_data.pt")
    te = hb.load_floods_split(paths["valid"], args.smoke_tiles, seed=7) if (args.smoke and not os.path.exists(te_path)) else hb.load_floods_split(te_path, n=exp18.N_TEST_TILES, seed=1)
    if args.smoke:
        bo = (bo[0][:args.smoke_tiles], bo[1][:args.smoke_tiles]); te = (te[0][:args.smoke_tiles], te[1][:args.smoke_tiles])
    splits = {"bolivia": bo, "test": te}

    versions = {"v1": None} if args.smoke else {"v1": None, "v1_2": None}
    try:
        from olmoearth_pretrain.model_loader import ModelID
        has_v12 = hasattr(ModelID, "OLMOEARTH_V1_2_BASE")
    except Exception:
        has_v12 = False
    if not has_v12:
        if not args.smoke:
            raise SystemExit("OLMOEARTH_V1_2_BASE is unavailable: run the full experiment with ~/oe12/.venv.")
        versions.pop("v1_2", None)
        summary["note"] = "v1.2 identifiers unavailable in this environment; only the v1 arm ran"

    cache = np.load(os.path.join(hb.OUT, "exp18_feats.npz")) if not args.smoke else np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz"))
    rng = np.random.default_rng(50)
    for version in list(versions):
        try:
            t0 = time.time()
            if version == "v1":
                model = hb.load_model()
                trf = np.asarray(cache["tr_base"], dtype=np.float32)
                if len(trf) != len(tr_s2):
                    trf = np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32)
                    print("  train cache length mismatch, re-encoded", flush=True)
            else:
                from olmoearth_pretrain.model_loader import load_model_from_id
                model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(exp18.DEV).eval().float()
                trf = np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32)

            def feats_at(name, s2):
                key = f"{'bolivia' if name == 'bolivia' else 'test'}_base"
                cached = version == "v1" and name != "train" and all(f"{key}{s}" in cache.files and len(cache[f"{key}{s}"]) == len(s2) for s in SHIFTS)

                def get(s):
                    if name == "train" and s == 0:
                        return trf
                    if cached:
                        return np.asarray(cache[f"{key}{s}"], dtype=np.float32)
                    return np.asarray(exp18.embed(model, s2, s)[0], dtype=np.float32)
                return get

            # exp49's reference head and eight members (seed 49), then the fresh bag of sixteen (seed 50)
            heads = [e49.head_from(trf, tr_lab)] + e49.member_heads(trf, tr_lab) + member_heads(trf, tr_lab, N_BAG, BAG_SEED)
            n49 = 1 + e49.M_MEMBERS
            centroids, _, intra_mean = kmeans(trf.reshape(-1, trf.shape[-1]), k=e49.K_CENTROIDS, iters=25, seed=0)
            train_stats = e49.window_stats(tr_s2).reshape(-1, e49.N_STATS)

            def signals(name, s2, lab):
                p, f0 = e49.predict_shifts(feats_at(name, s2), heads)
                sig, err, ok, w1, yy = e49.arm(p[:n49], f0, s2, lab, centroids, intra_mean, train_stats)
                bag = ensemble_uncertainty(np.stack([e49.w1_windows(q) for q in p[n49:]]))
                sl = slice(1, G)
                sig[BAG_PE], sig[BAG_MI] = bag["predictive entropy"][:, sl, sl], bag["mutual information"][:, sl, sl]
                sig[SLE] = shift_label_entropy(p[0])[:, sl, sl]
                return sig, err, ok, w1, yy

            sig_tr, err_tr, ok_tr, w1_tr, yy_tr = signals("train", tr_s2, tr_lab)
            full = fit_linear(sig_tr, err_tr, ok_tr, SIGNALS)
            drop = {k: fit_linear(sig_tr, err_tr, ok_tr, [s for s in SIGNALS if s != k]) for k in SIGNALS}
            Xt, ft = tile_rows(sig_tr, w1_tr, yy_tr, ok_tr, SIGNALS)
            tile_fit = fit_linear({k: Xt[:, i] for i, k in enumerate(SIGNALS)}, ft, np.ones(len(ft), dtype=bool), SIGNALS)
            res = {"train_seconds": time.time() - t0, "train_windows": int(ok_tr.sum()), "train_errors": int(err_tr[ok_tr].sum()),
                   "train_tiles": int(len(ft)), "train_tile_failures": int(ft.sum()),
                   "fusion_weights": {k: float(v) for k, v in zip(SIGNALS, full["w"])}, "fusion_bias": full["b"],
                   "tile_fusion_weights": {k: float(v) for k, v in zip(SIGNALS, tile_fit["w"])}}
            summary["results"][version] = res
            print(f"{version}: heads and fusions fitted in {res['train_seconds']:.0f}s | fusion weights " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in sorted(res["fusion_weights"].items(), key=lambda kv: -abs(kv[1]))), flush=True)
            del sig_tr
            for name, (s2, lab) in splits.items():
                sig, err, ok, w1, yy = signals(name, s2, lab)
                sig[FUSE_LIN] = apply_linear(sig, full)
                pooled = {k: pooled_eaurc(sig[k], err, ok) for k in [W1C, e49.TILE, e49.ENS_PE, e49.ENS_MI, BAG_PE, BAG_MI, SLE, FUSE_LIN]}
                pooled_drop = {k: pooled_eaurc(apply_linear(sig, drop[k]), err, ok) for k in SIGNALS}
                tests = {BAG_PE: per_tile_test(sig[BAG_PE], sig[W1C], err, ok, "greater"),
                         BAG_MI: per_tile_test(sig[BAG_MI], sig[W1C], err, ok, "two-sided"),
                         SLE: per_tile_test(sig[SLE], sig[e49.TILE], err, ok, "two-sided"),
                         FUSE_LIN: per_tile_test(sig[FUSE_LIN], sig[W1C], err, ok, "two-sided")}
                for k in tests:
                    tests[k]["pooled_lead"] = pooled[k] - pooled[W1C if k != SLE else e49.TILE]
                X, fail = tile_rows(sig, w1, yy, ok, SIGNALS)
                Xc = tile_rows(sig, w1, yy, ok, [W1C])[0][:, 0]
                tf_score = apply_linear({k: X[:, i] for i, k in enumerate(SIGNALS)}, tile_fit)
                tile = {"n_tiles": int(len(fail)), "failure_rate": float(fail.mean()),
                        "averaged confidence": tile_scores(Xc, fail, Xc, rng), TILE_FUSE: tile_scores(tf_score, fail, Xc, rng)}
                p1 = version == "v1" and name == "bolivia" and tests[BAG_PE]["pooled_lead"] <= -MIN_LEAD and tests[BAG_PE]["sign_p"] < 0.05
                ci = tile[TILE_FUSE]["diff_ci95"]
                p2 = version == "v1_2" and name == "bolivia" and ci is not None and ci[1] < 0
                r = {"n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()), "pooled_eaurc": pooled, "drop_one_pooled_eaurc": pooled_drop,
                     "tests": tests, "tile_level": tile, "P1": bool(p1), "P2": bool(p2)}
                summary["results"][version][name] = r
                worst = max(pooled_drop, key=pooled_drop.get)
                print(f"{version}/{name}: confidence {pooled[W1C]:.4f}, fusion {pooled[FUSE_LIN]:.4f}, bag16 PE {pooled[BAG_PE]:.4f} "
                      f"(w/l {tests[BAG_PE]['w']}/{tests[BAG_PE]['l']}, p={tests[BAG_PE]['sign_p']:.2g}), bag16 MI {pooled[BAG_MI]:.4f}, "
                      f"shift-label entropy {pooled[SLE]:.4f} vs tile-phase {pooled[e49.TILE]:.4f} | drop-one costs most: {worst} -> {pooled_drop[worst]:.4f} | "
                      f"tile AURC confidence {tile['averaged confidence']['aurc']:.3f}, tile-fitted fusion {tile[TILE_FUSE]['aurc']:.3f}, "
                      f"diff {tile[TILE_FUSE]['aurc_diff_vs_confidence']:+.3f} CI {ci} | P1 {p1} P2 {p2}", flush=True)
                for k in SIGNALS:
                    rows.append({"version": version, "testbed": name, "dropped": k, "pooled_eaurc": pooled_drop[k], "cost_vs_full": pooled_drop[k] - pooled[FUSE_LIN]})
            del model
            if exp18.DEV == "cuda":
                torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": version, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{version} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    r1 = summary["results"].get("v1", {}).get("bolivia", {}).get("P1")
    r2 = summary["results"].get("v1_2", {}).get("bolivia", {}).get("P2")
    summary["prereg"] = {"P1": r1, "P2": r2, "supported": bool(r1 and r2) if (r1 is not None and r2 is not None) else None, "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp50_fusion_ablation{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp50_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp50_summary{suffix}.json; prereg {summary['prereg']}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
