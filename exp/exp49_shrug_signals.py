#!/usr/bin/env python
"""exp49: SHRUG-FM's reliability signals under the label-free window protocol.

Why. SHRUG-FM (Gonzalez-Calabuig et al., arXiv 2511.10370; CVPR 2026 EarthVision workshop, Best Paper Award) is
selective prediction for geospatial foundation models, the closest published relative of this repository's
question. It ranks images by three signal families: input-space extremity U_I, the percentile rank of
geophysical covariates against the pretraining data's empirical CDFs plus a KDE spatial density of the
pretraining locations; embedding-space OOD U_E, the normalized distance to the nearest of k = 64 k-means
centroids of pretraining embeddings and the Nearest Centroid Distance Deficit NCDD = sum_{j != c} d_j - (k - 1) d_c
over the cluster-normalized distances d_j / d-bar_j (their eq. 5, after Pokhrel et al. 2024; higher = more
in-distribution); and task uncertainty U_T, the ensemble mutual information
(epistemic) and average entropy (aleatoric) of the downstream model, averaged over the predicted-event region of
interest to an image-level score. A depth-3 decision tree trained on labels (failure = image F1 < 0.6) on the
train+val splits fuses them; evaluation is AURC and Risk@coverage on test. Their baselines are single signals
from the ensemble; a single downstream model's own confidence, the reference every ranker in this repository
must beat, is never compared. Our protocol (exp47) grades rankers at WINDOW level on the shift-averaged (W1)
decision's own errors, with the model's own confidence as the reference and no-model controls beside it. This
run puts their signals under that protocol.

Substitutions, stated plainly. The head's training split (Sen1Floods11 valid, 600 tiles) stands in for
pretraining data in U_I and U_E; there is no spatial-density term because the flood chips carry no coordinates;
an ensemble of 8 tile-bootstrap linear heads on the frozen features stands in for their CNN ensembles; the
labelled fusion is a logistic combiner (linear, and degree-2 polynomial), which the paper itself reports as
performing comparably to the tree ("low-degree polynomial models ... yield comparable performance"); every
signal is aggregated to the window, not the image, with a separate tile-level bridge that follows their ROI
aggregation and F1 < 0.6 failure definition.

Design. Per backbone version (v1 Base from the cached exp/out/exp18_feats.npz, v1.2 Base encoded here), exp47's
arm exactly: both testbeds (Bolivia, 441 tiles; the test split as exp18 sampled it, 800 tiles, seed 1), the exp18
reference head (valid split, 600 tiles, seed 0), the four crop offsets, the shift-averaged probability pooled to
the 4-px window grid at offset 0 and cropped to the windows every tiling covers, its hard decision defining ONE
error set per arm. Rankers graded on it:
  averaged confidence (the reference), tile-phase and the NDWI-level control (exp47's primaries);
  U_T  ensemble mutual information, average entropy and predictive entropy of the 8 members' own W1 maps
       (member i: seed 100 + i on the labelled windows of a with-replacement resample of the training tiles);
  U_E  embedding normalized distance and -NCDD against k = 64 centroids of ALL training shift-0 windows,
       labelled or not, as the paper clusters unlabelled pretraining embeddings; NCDD is computed as their eq. 5
       states it, over cluster-normalized distances, which is not monotone in distance when cluster spreads differ
       (a broad nearest cluster makes far points look in-distribution), so the raw-distance deficit of Pokhrel et
       al., which vanishes far from every centroid, is graded beside it as "embedding NCDD raw" (secondary, not a
       fusion input);
  U_I  input extremity max and mean over 14 per-window statistics of the shift-0 crop (twelve band means, NDWI
       mean and std) against the training windows' empirical CDFs;
  fusion linear / poly2 (labelled): logistic combiners of the ten signals above, standardized on the training
       tiles' windows and fitted there against the in-sample W1 decision's errors, as SHRUG-FM fits its tree on
       train+val; applied to the testbeds with the training mean and sd, the logit is the ranker.
Tile-level bridge (their granularity): per tile with any labelled window, ROI = labelled windows with W1 >= 0.5
(all labelled windows if none), tile score = mean of the window ranker over the ROI, failure = water-class F1
over the tile's labelled windows below 0.6 (no water predicted and none labelled counts as F1 = 1); AURC and
Risk@0.5 / Risk@0.75 per ranker.

Preregistered, one-sided, on BOTH testbeds under v1.2; each must hold on both:
  P1  the averaged decision's confidence has a lower pooled excess AURC than ensemble mutual information, and
      wins the per-tile exact sign test against it (tiles with 3 <= errors <= n - 3); pooled lead >= 0.001.
  P2  the same against -NCDD.
Secondary, descriptive: the two labelled fusions against confidence (a fusion beats confidence when its pooled
lead is <= -0.001 with a two-sided sign p < 0.05), and the tile-level bridge. Falsification: if mutual information
beats confidence on both testbeds the ledger's confidence row changes and head ensembles join the audit; if a
labelled fusion beats confidence on a testbed, that testbed has headroom for signal fusion that exp47's U+ did
not find.

Inputs: data/floods/, exp/out/exp18_feats.npz. Outputs: exp/out/exp49_summary.json, exp/out/exp49_shrug_signals.csv.
Run the v1.2 arm with ~/oe12/.venv. --smoke: CPU, 4 tiles per split, v1 only, _smoke outputs.
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
import harness_ab as hb  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected, oracle_aurc, selective_accuracy  # noqa: E402
from oe_inferencex.reliability import bootstrap_tiles, centroid_signals, ensemble_uncertainty, input_extremity, kmeans, poly2, standardize  # noqa: E402
from oe_inferencex.signals import ndwi, ndwi_level, pool_to_windows, shift_averaged_probability  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

PATCH, CROP, G, SHIFTS = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS
MIN_LEAD, M_MEMBERS, K_CENTROIDS, N_STATS = 0.001, 8, 64, 14
W1C, TILE, CTRL_L = "averaged confidence", "tile-phase", "control NDWI level"
ENS_MI, ENS_AE, ENS_PE = "ensemble mutual information", "ensemble average entropy", "ensemble predictive entropy"
EMB_ND, EMB_NCDD, EMB_NCDD_RAW = "embedding normalized distance", "embedding NCDD", "embedding NCDD raw"
INP_MAX, INP_MEAN = "input extremity max", "input extremity mean"
FUSE_LIN, FUSE_POLY = "fusion linear (labelled)", "fusion poly2 (labelled)"
SIGNALS = [W1C, TILE, CTRL_L, ENS_MI, ENS_AE, ENS_PE, EMB_ND, EMB_NCDD, INP_MAX, INP_MEAN]      # the fusions' inputs
FUSIONS = [FUSE_LIN, FUSE_POLY]
RANKERS = SIGNALS + [EMB_NCDD_RAW] + FUSIONS
PRIMARY_AGAINST = [ENS_MI, EMB_NCDD]


def head_from(tr_feats, tr_lab):
    y, ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    torch.manual_seed(0)
    return train_logistic_head(torch.tensor(tr_feats.reshape(-1, tr_feats.shape[-1])[ok.flatten()]), y.flatten()[ok.flatten()])


def member_heads(tr_feats, tr_lab):
    """The ensemble: M heads, member i trained (seed 100 + i) on the labelled windows of a with-replacement resample of the training tiles."""
    y, ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    flat, heads = tr_feats.reshape(-1, tr_feats.shape[-1]), []
    for i, idx in enumerate(bootstrap_tiles(len(tr_feats), M_MEMBERS, seed=49)):
        rows, keep = (idx[:, None] * (G * G) + np.arange(G * G)[None, :]).ravel(), ok[idx].ravel()
        torch.manual_seed(100 + i)
        heads.append(train_logistic_head(torch.tensor(flat[rows[keep]]), y[idx].ravel()[keep]))
    return heads


def predict_shifts(feats_at, heads):
    """Every head's probability map at every crop offset, (len(heads), S, N, G, G); the shift-0 features are kept, the others dropped."""
    p, f0 = [[] for _ in heads], None
    for s in SHIFTS:
        f = feats_at(s)
        for i, h in enumerate(heads):
            p[i].append(exp18.head_prob_logit(f, *h)[0])
        f0 = f if s == 0 else f0
    return np.stack([np.stack(q) for q in p]), f0


def w1_windows(p_shift):
    """(S, N, G, G) shift maps -> the shift-averaged decision pooled back to the shift-0 window grid, (N, G, G)."""
    return np.stack([pool_to_windows(shift_averaged_probability(p_shift[:, t], patch=PATCH)[0], patch=PATCH, offset=0) for t in range(p_shift.shape[1])])


def window_stats(s2):
    """(N, G, G, 14) per-window input statistics on the shift-0 grid: twelve band means over each PATCH x PATCH block of the crop, NDWI mean and std."""
    N = len(s2)
    x = s2[:, :12, :CROP, :CROP].astype(np.float64)
    bands = x.reshape(N, 12, G, PATCH, G, PATCH).mean(axis=(3, 5)).transpose(0, 2, 3, 1)
    nd = ndwi(s2[:, :, :CROP, :CROP]).reshape(N, G, PATCH, G, PATCH)
    return np.concatenate([bands, nd.mean(axis=(2, 4))[..., None], nd.std(axis=(2, 4))[..., None]], axis=-1)


def arm(p, f0, s2, lab, centroids, intra_mean, train_stats):
    """One backbone on one tile set: the averaged decision's error set (exp47's construction) and every label-free ranker graded on it."""
    N, D = len(s2), f0.shape[-1]
    y, ok = exp18.patch_labels(lab[:, :CROP, :CROP])
    # As exp47: pool the averaged map at offset 0 so the windows coincide with the shift-0 label grid, then keep the
    # windows lying wholly inside the region every tiling covers, i = 1..G-1.
    sl = slice(1, G)
    w1 = w1_windows(p[0])[:, sl, sl]
    yy, okk = y[:, sl, sl], ok[:, sl, sl]
    err = ((w1 > 0.5) != (yy > 0.5)).astype(np.float64)
    ens = ensemble_uncertainty(np.stack([w1_windows(q) for q in p[1:]]))
    emb = centroid_signals(f0.reshape(-1, D), centroids, intra_mean)
    inp = input_extremity(train_stats, window_stats(s2).reshape(-1, N_STATS))
    grid = lambda v: np.asarray(v).reshape(N, G, G)[:, sl, sl]            # noqa: E731
    sig = {W1C: -np.abs(w1 - 0.5),
           TILE: exp18.aligned_tile_phase(p[0])[:, sl, sl],
           CTRL_L: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])[:, sl, sl],
           ENS_MI: ens["mutual information"][:, sl, sl], ENS_AE: ens["average entropy"][:, sl, sl], ENS_PE: ens["predictive entropy"][:, sl, sl],
           EMB_ND: grid(emb["normalized distance"]), EMB_NCDD: -grid(emb["NCDD"]),                # higher = more suspect
           EMB_NCDD_RAW: -grid(emb["NCDD raw"]),
           INP_MAX: grid(inp["max"]), INP_MEAN: grid(inp["mean"])}
    return sig, err, okk, w1, yy


def fit_fusion(sig, err, ok):
    """SHRUG-FM's labelled step on the training windows: logistic combiners of the standardized signals, linear and degree-2 polynomial."""
    Z, mu, sd = standardize(np.stack([sig[k][ok] for k in SIGNALS], 1))
    heads = {}
    for name, X in ((FUSE_LIN, Z), (FUSE_POLY, poly2(Z))):
        torch.manual_seed(0)
        heads[name] = train_logistic_head(torch.tensor(X, dtype=torch.float32), err[ok])
    return {"mu": mu, "sd": sd, "heads": heads}


def apply_fusion(sig, fus):
    """Adds the two fusion rankers, the combiners' logits under the training mean and sd (higher = more suspect)."""
    shape = sig[W1C].shape
    Z, _, _ = standardize(np.stack([sig[k] for k in SIGNALS], -1).reshape(-1, len(SIGNALS)), fus["mu"], fus["sd"])
    for name, X in ((FUSE_LIN, Z), (FUSE_POLY, poly2(Z))):
        w, b = fus["heads"][name]
        sig[name] = (torch.tensor(X, dtype=torch.float32) @ w + b).numpy().reshape(shape)


def score(sig, err, ok):
    tiles = [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]
    pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum())) for k, v in sig.items()}
    per = {k: [aurc_expected(np.asarray(sig[k][t])[ok[t]], err[t][ok[t]]) - oracle_aurc(int(ok[t].sum()), int(err[t][ok[t]].sum())) for t in tiles] for k in sig}
    tests = {}
    for k in sig:
        if k == W1C:
            continue
        g = np.array(per[k]) - np.array(per[W1C])                          # positive: the averaged confidence is better
        w, l, t_ = wins_losses_ties(g)
        one = k in PRIMARY_AGAINST
        tests[k] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                    "median_gain": float(np.median(g)) if len(g) else None, "pooled_lead": pooled[k] - pooled[W1C]}
        if k in FUSIONS:                                                    # a labelled combiner beating the label-free reference
            tests[k]["beats_confidence"] = bool(tests[k]["pooled_lead"] <= -MIN_LEAD and tests[k]["sign_p"] < 0.05)
    passes = all(tests[k]["sign_p"] < 0.05 and tests[k]["pooled_lead"] >= MIN_LEAD for k in PRIMARY_AGAINST)
    return {"n_tiles_scored": len(tiles), "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()),
            "pooled_eaurc": pooled, "tests": tests, "prereg_passes": bool(passes)}


def tile_level(sig, w1, yy, ok):
    """SHRUG-FM's granularity: each ranker averaged over the tile's predicted-water ROI; failure = water-class F1 over the tile's labelled windows below 0.6."""
    scores, fail = {k: [] for k in sig}, []
    for t in range(len(w1)):
        m = ok[t]
        if not m.any():
            continue
        pred, truth = w1[t][m] > 0.5, yy[t][m] > 0.5
        tp, fp, fn = float((pred & truth).sum()), float((pred & ~truth).sum()), float((~pred & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1.0     # no water predicted and none labelled: not a failure
        fail.append(float(f1 < 0.6))
        roi = m & (w1[t] >= 0.5)
        roi = roi if roi.any() else m
        for k in sig:
            scores[k].append(float(np.mean(np.asarray(sig[k][t])[roi])))
    fail = np.array(fail)
    per = {}
    for k in sig:
        s = np.array(scores[k])
        acc = selective_accuracy(s, 1 - fail, coverages=(0.5, 0.75))      # most confident = lowest ranker value kept first
        per[k] = {"aurc": aurc_expected(s, fail), "risk_at_0.5": 1 - acc[0.5], "risk_at_0.75": 1 - acc[0.75]}
    return {"n_tiles": int(len(fail)), "n_failures": int(fail.sum()), "failure_rate": float(fail.mean()) if len(fail) else float("nan"), "per_ranker": per}


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp49 SHRUG-FM's reliability signals under the label-free window protocol", "smoke": args.smoke,
               "config": {"rankers": RANKERS, "fusion_inputs": SIGNALS, "primary_against": PRIMARY_AGAINST, "min_lead": MIN_LEAD,
                          "ensemble_members": M_MEMBERS, "kmeans_k": K_CENTROIDS, "n_input_stats": N_STATS, "shifts": list(SHIFTS),
                          "prereg": "the averaged decision's own confidence beats ensemble mutual information and -NCDD, pooled lead >= 0.001 and a one-sided per-tile sign test, on BOTH testbeds under v1.2; "
                                    "secondary: each labelled fusion against confidence (beats_confidence = pooled lead <= -0.001 and two-sided sign p < 0.05) and the tile-level bridge"},
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
            raise SystemExit("OLMOEARTH_V1_2_BASE is unavailable: run the full experiment with ~/oe12/.venv, whose "
                             "olmoearth_pretrain is built from upstream main. The v1.2 arm is the preregistered one.")
        versions.pop("v1_2", None)
        summary["note"] = "v1.2 identifiers unavailable in this environment; only the v1 arm ran"

    cache = np.load(os.path.join(hb.OUT, "exp18_feats.npz")) if not args.smoke else np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz"))
    for version in list(versions):
        try:
            t0 = time.time()
            if version == "v1":
                model = hb.load_model()                                    # the training tiles' shifts 1..3 are not in the cache
                trf = np.asarray(cache["tr_base"], dtype=np.float32)
                if len(trf) != len(tr_s2):                                # never slice a cache to fit: the seed-0 sample is not prefix-stable
                    trf = np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32)
                    print("  train cache length mismatch, re-encoded", flush=True)
            else:
                from olmoearth_pretrain.model_loader import load_model_from_id
                model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(exp18.DEV).eval().float()
                trf = np.asarray(exp18.embed(model, tr_s2, 0)[0], dtype=np.float32)

            def feats_at(name, s2):
                """Features at offset s for one tile set: the exp18 cache where it holds every shift of the split under v1, else the encoder."""
                key = f"{'bolivia' if name == 'bolivia' else 'test'}_base"
                cached = version == "v1" and name != "train" and all(f"{key}{s}" in cache.files and len(cache[f"{key}{s}"]) == len(s2) for s in SHIFTS)

                def get(s):
                    if name == "train" and s == 0:
                        return trf
                    if cached:
                        return np.asarray(cache[f"{key}{s}"], dtype=np.float32)
                    return np.asarray(exp18.embed(model, s2, s)[0], dtype=np.float32)
                return get

            heads = [head_from(trf, tr_lab)] + member_heads(trf, tr_lab)
            centroids, assign, intra_mean = kmeans(trf.reshape(-1, trf.shape[-1]), k=K_CENTROIDS, iters=25, seed=0)
            train_stats = window_stats(tr_s2).reshape(-1, N_STATS)
            p_tr, _ = predict_shifts(feats_at("train", tr_s2), heads)     # in-sample, as SHRUG-FM fits its fusion on train+val
            sig_tr, err_tr, ok_tr, _, _ = arm(p_tr, trf, tr_s2, tr_lab, centroids, intra_mean, train_stats)
            fus = fit_fusion(sig_tr, err_tr, ok_tr)
            del sig_tr, p_tr
            sizes = np.bincount(assign, minlength=K_CENTROIDS)
            summary["results"][version] = {"train_seconds": time.time() - t0, "train_windows": int(ok_tr.sum()), "train_errors": int(err_tr[ok_tr].sum()),
                                           "kmeans_cluster_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())}}
            print(f"{version}: reference head, {M_MEMBERS} members, {K_CENTROIDS} centroids (cluster sizes {sizes.min()}-{sizes.max()}) and the two labelled fusions "
                  f"fitted on {int(ok_tr.sum())} training windows ({int(err_tr[ok_tr].sum())} in-sample errors) in {time.time() - t0:.0f}s", flush=True)
            for name, (s2, lab) in splits.items():
                t1 = time.time()
                p, f0 = predict_shifts(feats_at(name, s2), heads)
                sig, err, ok, w1, yy = arm(p, f0, s2, lab, centroids, intra_mean, train_stats)
                del p, f0
                apply_fusion(sig, fus)
                r = score(sig, err, ok)
                r["tile_level"] = tile_level(sig, w1, yy, ok)
                r["seconds"] = time.time() - t1
                summary["results"][version][name] = r
                lead = {k: r["tests"][k]["pooled_lead"] for k in PRIMARY_AGAINST}
                tl = r["tile_level"]
                print(f"{version}/{name}: {r['n_windows']} windows, {r['n_errors']} errors of the averaged decision, {r['n_tiles_scored']} tiles | "
                      f"pooled E-AURC " + ", ".join(f"{k} {v:.4f}" for k, v in r["pooled_eaurc"].items()) +
                      f" | lead over MI {lead[ENS_MI]:+.4f} (p={r['tests'][ENS_MI]['sign_p']:.2g}), over -NCDD {lead[EMB_NCDD]:+.4f} (p={r['tests'][EMB_NCDD]['sign_p']:.2g}) -> {r['prereg_passes']}"
                      f" | fusion beats confidence: " + ", ".join(f"{k.split()[1]} {r['tests'][k]['beats_confidence']}" for k in FUSIONS), flush=True)
                print(f"  tile level: {tl['n_tiles']} tiles, {tl['n_failures']} failures (rate {tl['failure_rate']:.3f}) | AURC " +
                      ", ".join(f"{k} {tl['per_ranker'][k]['aurc']:.4f}" for k in RANKERS) +
                      " | risk@0.5 " + ", ".join(f"{k} {tl['per_ranker'][k]['risk_at_0.5']:.3f}" for k in (W1C, ENS_MI, EMB_NCDD, FUSE_LIN, FUSE_POLY)), flush=True)
                rows.append({"version": version, "testbed": name, "n_windows": r["n_windows"], "n_errors": r["n_errors"],
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()},
                             **{f"tile_aurc {k}": tl["per_ranker"][k]["aurc"] for k in RANKERS}, "prereg_passes": r["prereg_passes"]})
            del model
            if exp18.DEV == "cuda":
                torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": version, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{version} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    if "v1_2" in summary["results"] and all(k in summary["results"]["v1_2"] for k in ("bolivia", "test")):
        summary["prereg"] = {"supported": bool(all(summary["results"]["v1_2"][n]["prereg_passes"] for n in ("bolivia", "test"))),
                             "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp49_shrug_signals{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp49_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp49_summary{suffix}.json; prereg {summary.get('prereg')}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
