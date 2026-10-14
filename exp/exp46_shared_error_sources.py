#!/usr/bin/env python
"""exp46: what makes the errors shared — the readout, the input modality, or the window?

Why. exp41 found that six frozen encoders read by the same token-local linear probe err on 80-82% of the same
Sen1Floods11 windows, and the repository briefly concluded that "the errors belong to the windows, not the
model". That conclusion was withdrawn on 2026-09-09: the one end-to-end fine-tuned model we hold corrects 55.6%
of a frozen probe's errors on identical AWF points at phi 0.475 (exp/out/exp21_finetuned_awf.csv), against
0.77-0.81 across six frozen encoders, so changing the readout moves the error set about twice as much as
changing the frozen backbone. exp41 held three things constant that could each produce the shared errors on
their own: one token-local linear probe, one input modality (its Sen1Floods11 embeddings are Sentinel-1), and
one hard-thresholded majority label per 4-px window. This run varies them one at a time on identical windows.

Arms, all with the same chips, windows, labels and fit split (the valid split's 600 tiles, seed 0), the frozen
OlmoEarth v1 Base encoder except where stated:
  A0  no encoder      logistic head on per-window pixel statistics only (twelve Sentinel-2 band means, NDWI mean
                      and standard deviation, and the two Sentinel-1 backscatter means): how much of the
                      accuracy the foundation model is actually buying.
  A1  S2 linear       token-local linear head on the cached Sentinel-2 features. The reference arm.
  A2  S1 linear       the same head class on Sentinel-1 features encoded here. Modality varied, readout fixed.
  A3  S2 MLP          a two-layer head (256 hidden units) on the SAME features as A1. Readout capacity varied,
                      features fixed.
  A4  S2 3x3 linear   a linear head on the 3-by-3 token neighbourhood (9 x 768 features), edge-padded. Spatial
                      context varied, features and head class otherwise fixed.

Reference points already measured on these same windows, for comparison rather than re-measurement: swapping the
backbone from v1 to v1.2 at fixed modality and readout gives phi 0.697 (Bolivia) and 0.705 (test split, exp45);
swapping the frozen encoder across families gives phi 0.77-0.81 (exp41, Sentinel-1); fine-tuning the encoder
gives phi 0.475 (exp21, AWF).

Preregistered, before the run, on both testbeds; each must hold on both.
  P1 readout: changing the readout moves the error set at least as much as changing the backbone, that is
     phi(A1, A3) < 0.697 on Bolivia and < 0.705 on the test split, and the same for phi(A1, A4). One-sided.
  P2 modality: changing the input modality moves it at least as much, that is phi(A1, A2) below the same
     thresholds. One-sided.
  Falsification: if phi(A1, A3) and phi(A1, A4) are at or above the backbone-swap value while the arms reach
  comparable accuracy, readout capacity is not the shared factor, and the window-and-label explanation the
  repository withdrew is strengthened rather than refuted.

Descriptive alongside: the accuracy of every arm (an MLP that does not improve accuracy makes P1 uninformative,
so it is reported first); the full pairwise overlap matrix; phi stratified by window label purity (windows whose
labelled pixels are strictly one class, against mixed ones); and the per-chip median phi beside the pooled value,
since a pooled correlation over half a million windows mixes chips being hard with windows being hard.

Inputs: data/floods/*.pt (both sensors and the labels), exp/out/exp18_feats.npz (the cached Sentinel-2
features). Outputs: exp/out/exp46_summary.json, exp/out/exp46_shared_error_sources.csv.
--smoke: CPU, 4 tiles per split, _smoke outputs.
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
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from oe_inferencex.evidence import train_logistic_head  # noqa: E402
from oe_inferencex.compare import phi  # noqa: E402,F401  (the recorded copy lived here; NaN now where it returned None)
from oe_inferencex.signals import ndwi  # noqa: E402

PATCH, CROP, G = exp18.PATCH, exp18.CROP, exp18.G
DEV = exp18.DEV
BACKBONE_SWAP_PHI = {"bolivia": 0.6968365816742046, "test": 0.7046557397695807}   # exp45 D1, same windows
A0, A1, A2, A3, A4 = "A0 no encoder", "A1 S2 linear", "A2 S1 linear", "A3 S2 MLP", "A4 S2 3x3 linear"
ARMS = [A0, A1, A2, A3, A4]


def embed_s1(model, tiles, batch=32):
    """Sentinel-1 analogue of exp18.embed: one band set (vv, vh), pooled tokens (N, G, G, D)."""
    norm = exp18._norm
    out = []
    for i in range(0, len(tiles), batch):
        x = tiles[i:i + batch, :, :CROP, :CROP]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)      # (B, H, W, 1, 2)
        x = norm.normalize(Modality.SENTINEL1, x)
        b = x.shape[0]
        sample = MaskedOlmoEarthSample(
            sentinel1=torch.tensor(x, dtype=torch.float32, device=DEV),
            sentinel1_mask=torch.ones((b, CROP, CROP, 1, 1), device=DEV) * MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor([1, 5, 2020], device=DEV)[None, None, :].repeat(b, 1, 1),
        )
        with torch.no_grad():
            tok = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel1
        out.append(tok.mean(dim=[3, 4]).float().cpu().numpy())
    return np.concatenate(out)


def pixel_stats(s2, s1):
    """(N, G, G, 15) per-window statistics: twelve band means, NDWI mean and std, and the two backscatter means."""
    N = len(s2)
    x = s2[:, :12, :CROP, :CROP].astype(np.float64)
    bands = x.reshape(N, 12, G, PATCH, G, PATCH).mean(axis=(3, 5)).transpose(0, 2, 3, 1)
    nd = ndwi(s2[:, :, :CROP, :CROP]).reshape(N, G, PATCH, G, PATCH)
    nd_mean, nd_std = nd.mean(axis=(2, 4))[..., None], nd.std(axis=(2, 4))[..., None]
    r = np.nan_to_num(s1[:, :, :CROP, :CROP], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    rad = r.reshape(N, 2, G, PATCH, G, PATCH).mean(axis=(3, 5)).transpose(0, 2, 3, 1)
    return np.concatenate([bands, nd_mean, nd_std, rad], axis=-1)


def neighbourhood(feats):
    """(N, G, G, D) -> (N, G, G, 9D): the 3x3 token neighbourhood, edge-padded."""
    p = np.pad(feats, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
    return np.concatenate([p[:, i:i + G, j:j + G, :] for i in range(3) for j in range(3)], axis=-1)


class MLP:
    """Two-layer head, 256 hidden units, trained with Adam on the same rows as the linear probe."""

    def __init__(self, X, y, hidden=256, epochs=60, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32, device=DEV)
        yt = torch.tensor(y, dtype=torch.float32, device=DEV)
        pos = float((yt == 0).sum() / max(float((yt == 1).sum()), 1.0))
        self.net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos], device=DEV))
        n = len(Xt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, 8192):
                idx = perm[i:i + 8192]
                opt.zero_grad()
                lossf(self.net(Xt[idx]).squeeze(-1), yt[idx]).backward()
                opt.step()
        self.net.eval()

    @torch.no_grad()
    def logit(self, X, chunk=200000):
        out = []
        for i in range(0, len(X), chunk):
            Xt = torch.tensor((X[i:i + chunk] - self.mu) / self.sd, dtype=torch.float32, device=DEV)
            out.append(self.net(Xt).squeeze(-1).cpu().numpy())
        return np.concatenate(out)


class Linear:
    """The established protocol (evidence.train_logistic_head, seed 0), fitted once and applied to both testbeds.

    `standardise` is used only for A0, whose raw per-window statistics span four orders of magnitude; the encoder
    arms keep exp18's protocol untouched so A1 reproduces the published head exactly."""

    def __init__(self, X, y, standardise=False):
        self.mu, self.sd = (X.mean(0), X.std(0) + 1e-6) if standardise else (0.0, 1.0)
        torch.manual_seed(0)
        self.w, self.b = train_logistic_head(torch.tensor(np.asarray((X - self.mu) / self.sd, dtype=np.float32)), y)

    def logit(self, X, chunk=200000):
        out = []
        for i in range(0, len(X), chunk):
            x = torch.tensor(np.asarray((X[i:i + chunk] - self.mu) / self.sd, dtype=np.float32))
            out.append((x @ self.w + self.b).numpy())
        return np.concatenate(out)


def overlap(ea, eb):
    return {"phi": phi(ea, eb), "p_b_wrong_given_a_wrong": float(eb[ea].mean()) if ea.any() else None,
            "p_b_wrong_given_a_right": float(eb[~ea].mean()) if (~ea).any() else None,
            "p_a_wrong_given_b_wrong": float(ea[eb].mean()) if eb.any() else None}


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp46 what makes the errors shared: readout, modality or window", "smoke": args.smoke,
               "config": {"arms": ARMS, "backbone_swap_phi_reference": BACKBONE_SWAP_PHI,
                          "prereg": "P1 phi(A1,A3) and phi(A1,A4) below the backbone-swap phi on both testbeds; P2 phi(A1,A2) likewise; falsification if the readout arms are at or above it at comparable accuracy"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    paths, _ = hb.ensure_floods(floods_dir, allow_download=not args.smoke)
    n_tr = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES

    def load(path, n=None, seed=0):
        d = torch.load(path, weights_only=True)
        s2 = d["s2"].numpy().astype(np.float32)[:, exp18.BAND_IDX]
        s1 = d["s1"].numpy().astype(np.float32)
        lab = d["labels"].numpy()[:, 0]
        if n is not None and n < len(s2):
            idx = np.random.default_rng(seed).choice(len(s2), n, replace=False)
            s2, s1, lab = s2[idx], s1[idx], lab[idx]
        return s2, s1, lab

    tr = load(paths["valid"], n_tr)
    bo = load(paths["bolivia"])
    te_path = os.path.join(floods_dir, "flood_test_data.pt")
    if not os.path.exists(te_path) and args.smoke:
        te = load(paths["valid"], args.smoke_tiles, seed=7)          # smoke stand-in: the test split lives on the cluster
    else:
        te = load(te_path, n=exp18.N_TEST_TILES, seed=1)
    if args.smoke:
        bo = tuple(a[:args.smoke_tiles] for a in bo); te = tuple(a[:args.smoke_tiles] for a in te)
    splits = {"train": tr, "bolivia": bo, "test": te}

    model = hb.load_model()
    cache = np.load(os.path.join(hb.OUT, "exp18_feats.npz" if not args.smoke else "exp42_floods_feats_smoke.npz"))
    key = {"train": "tr_base", "bolivia": "bolivia_base0", "test": "test_base0"}
    feats = {}
    for name, (s2, s1, lab) in splits.items():
        cached_ok = key[name] in cache.files and len(cache[key[name]]) == len(s2)   # exact length, never a prefix
        f2 = np.asarray(cache[key[name]], dtype=np.float32) if cached_ok else np.asarray(exp18.embed(model, s2, 0)[0], dtype=np.float32)
        if not cached_ok:
            print(f"  {name}: cache miss or length mismatch, re-encoded", flush=True)
        t0 = time.time()
        f1 = np.asarray(embed_s1(model, s1), dtype=np.float32)
        feats[name] = {"s2": f2, "s1": f1, "px": pixel_stats(s2, s1), "y": exp18.patch_labels(lab[:, :CROP, :CROP])}
        print(f"  {name}: S2 {f2.shape}, S1 {f1.shape} ({time.time()-t0:.0f}s), pixel stats {feats[name]['px'].shape}", flush=True)
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    ty, tok = feats["train"]["y"]
    sel = tok.flatten()
    fit = {"s2": feats["train"]["s2"].reshape(-1, feats["train"]["s2"].shape[-1])[sel],
           "s1": feats["train"]["s1"].reshape(-1, feats["train"]["s1"].shape[-1])[sel],
           "px": feats["train"]["px"].reshape(-1, feats["train"]["px"].shape[-1])[sel],
           "nb": neighbourhood(feats["train"]["s2"]).reshape(-1, feats["train"]["s2"].shape[-1] * 9)[sel]}
    fy = ty.flatten()[sel]
    heads = {A0: Linear(fit["px"], fy, standardise=True), A1: Linear(fit["s2"], fy), A2: Linear(fit["s1"], fy),
             A3: MLP(fit["s2"], fy), A4: Linear(fit["nb"], fy)}
    del fit
    print(f"  five heads fitted once on {len(fy)} windows", flush=True)

    for name in ("bolivia", "test"):
        try:
            f = feats[name]
            y, ok = f["y"]
            nb = neighbourhood(f["s2"]).reshape(-1, f["s2"].shape[-1] * 9)
            lg = {A0: heads[A0].logit(f["px"].reshape(-1, f["px"].shape[-1])),
                  A1: heads[A1].logit(f["s2"].reshape(-1, f["s2"].shape[-1])),
                  A2: heads[A2].logit(f["s1"].reshape(-1, f["s1"].shape[-1])),
                  A3: heads[A3].logit(f["s2"].reshape(-1, f["s2"].shape[-1])),
                  A4: heads[A4].logit(nb)}
            del nb
            m = ok.flatten()
            truth = (y.flatten() > 0.5)[m]
            err = {k: ((v.reshape(-1) > 0)[m] != truth) for k, v in lg.items()}
            acc = {k: float(1 - e.mean()) for k, e in err.items()}
            pair = {f"{a} vs {b}": overlap(err[a], err[b]) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
            # purity strata and per-chip decomposition
            lab = splits[name][2]
            l = lab[:, :G * PATCH, :G * PATCH].reshape(len(lab), G, PATCH, G, PATCH)
            nv = (l >= 0).sum(axis=(2, 4)); frac = np.where(nv > 0, (l == 1).sum(axis=(2, 4)) / np.maximum(nv, 1), np.nan)
            impure = ((frac > 0) & (frac < 1)).flatten()[m]        # mixed = not strictly one class among labelled pixels
            chips = np.repeat(np.arange(len(lab)), G * G)[m]
            strata, per_chip = {}, {}
            for k in (A2, A3, A4):
                strata[f"{A1} vs {k}"] = {"pure": phi(err[A1][~impure], err[k][~impure]), "mixed": phi(err[A1][impure], err[k][impure])}
                v = [phi(err[A1][chips == c], err[k][chips == c]) for c in np.unique(chips)]
                v = [x for x in v if x is not None and np.isfinite(x)]
                per_chip[f"{A1} vs {k}"] = {"median_within_chip_phi": float(np.median(v)) if v else None, "n_chips": len(v)}
            ref = BACKBONE_SWAP_PHI[name]
            def below(k):
                v = pair[f"{A1} vs {k}"]["phi"]
                return None if v is None else bool(v < ref)
            p13, p14, p12 = (pair[f"{A1} vs {k}"]["phi"] for k in (A3, A4, A2))
            prereg = {"backbone_swap_phi": ref,
                      "P1_readout": {"phi_A1_A3": p13, "phi_A1_A4": p14,
                                     "passes": None if None in (p13, p14) else bool(p13 < ref and p14 < ref)},
                      "P2_modality": {"phi_A1_A2": p12, "passes": below(A2)}}
            summary["results"][name] = {"n_windows": int(m.sum()), "accuracy": acc, "pairwise": pair, "purity_strata": strata,
                                        "per_chip": per_chip, "prereg": prereg, "impure_share": float(impure.mean())}
            print(f"{name}: acc " + ", ".join(f"{k.split()[0]} {v:.4f}" for k, v in acc.items()), flush=True)
            def fmt(v):
                return "n/a" if v is None else f"{v:.3f}"
            print(f"  phi vs {A1}: A2(modality) {fmt(p12)}, A3(MLP) {fmt(p13)}, A4(3x3) {fmt(p14)}, "
                  f"A0(no encoder) {fmt(pair[f'{A0} vs {A1}']['phi'])} | backbone swap {ref:.3f} | "
                  f"P1 {prereg['P1_readout']['passes']}, P2 {prereg['P2_modality']['passes']}", flush=True)
            print(f"  homogeneous/mixed phi: " + ", ".join(f"{k.split('vs ')[1]} {fmt(v['pure'])}/{fmt(v['mixed'])}" for k, v in strata.items())
                  + " | within-chip median phi: " + ", ".join(f"{k.split('vs ')[1]} {fmt(v['median_within_chip_phi'])}" for k, v in per_chip.items()), flush=True)
            rows.append({"testbed": name, "n_windows": int(m.sum()), **{f"acc {k}": v for k, v in acc.items()},
                         **{f"phi {k}": v["phi"] for k, v in pair.items()}, "backbone_swap_phi": ref})
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    if len(summary["results"]) == 2:
        summary["prereg"] = {"P1_readout": all(summary["results"][n]["prereg"]["P1_readout"]["passes"] is True for n in ("bolivia", "test")),
                             "P2_modality": all(summary["results"][n]["prereg"]["P2_modality"]["passes"] is True for n in ("bolivia", "test")),
                             "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp46_shared_error_sources{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp46_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp46_summary{suffix}.json; prereg {summary.get('prereg')}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
