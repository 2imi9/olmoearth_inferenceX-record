"""Exp 19: OlmoEarth v1 vs v1.2 (rotary position encoding) on the 27 scenes.

v1.2-Base uses rotary position encodings (rope_3d_mixed, verified from the
loaded model), introduced by Ai2 to fix the v1 striping artifact. Runs in
the isolated environment built on the current olmoearth_pretrain main
(.venv-main), which loads both versions. Questions:

  1. Does tiling instability shrink under RoPE? Mean aligned tile-phase
     magnitude per scene, v1 vs v1.2, same head protocol.
  2. Does the tile-phase signal still rank errors under v1.2? Tie-aware
     E-AURC vs the v1.2 head's own confidence, W/L/T over scenes.
  3. Is cross-version disagreement |p_v1.2 - p_v1| a usable signal for
     either model's errors?
  4. Consistency check: v1-Base features recomputed with the new code are
     compared with the cached exp11 features (same weights, refactored code).

Heads: water heads trained on the Katima Mulilo training scene for each
version separately (each version's own embedding space). Errors: each
version's head vs WorldCover on the shift-0 view. Band-set disagreement is
computed for both versions as well.
"""
import csv
import math
import os
from collections import Counter

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.evidence import train_logistic_head, aurc_expected

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE, PATCH, PAD = 128, 4, 4
G = SIZE // PATCH
SHIFTS = (0, 1, 2, 3)
NON_RULE = ("kafue", "luangwa")
CACHE = "exp/out/exp19_feats.npz"
_norm = Normalizer(Strategy.COMPUTED)


def embed(model, img, date, shift):
    x = img[:, shift:shift + SIZE, shift:shift + SIZE].transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = _norm.normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = date
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=DEV),
        sentinel2_l2a_mask=torch.ones((1, SIZE, SIZE, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=torch.tensor([d, m0, y], device=DEV)[None, None, :],
    )
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a[0]
    return out.mean(dim=[2, 3]).cpu().numpy(), out[:, :, 0].permute(2, 0, 1, 3).cpu().numpy()  # (G,G,D), (3,G,G,D)


def prob_logit(feats, w, b):
    x = torch.tensor(np.asarray(feats, dtype=np.float32)).reshape(-1, feats.shape[-1])
    logit = (x @ w + b).reshape(feats.shape[:-1]).numpy()
    return 1 / (1 + np.exp(-logit)), logit


def aligned_tile_phase(p_shift):
    canvas = np.full((len(p_shift), SIZE + PAD, SIZE + PAD), np.nan)
    for s, p in enumerate(p_shift):
        canvas[s, s:s + SIZE, s:s + SIZE] = np.kron(p, np.ones((PATCH, PATCH)))
    pix = np.nanstd(canvas, axis=0)[:SIZE, :SIZE].reshape(G, PATCH, G, PATCH)
    return np.nanmean(pix, axis=(1, 3))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def oracle(n, k):
    i = np.arange(1, n + 1)
    return float((np.maximum(0, i - (n - k)) / i).mean())


def eaurc(sig, err):
    return aurc_expected(sig, err) - oracle(len(err), int(err.sum()))


def main():
    scenes = dict(np.load("exp/out/exp11_scenes.npz", allow_pickle=True))
    old = dict(np.load("exp/out/exp11_feats.npz", allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    names = [n for n in names if f"{n}_base0" in old and n not in NON_RULE]

    versions = {"v1": ModelID.OLMOEARTH_V1_BASE, "v1_2": ModelID.OLMOEARTH_V1_2_BASE}
    if os.path.exists(CACHE):
        z = dict(np.load(CACHE)); print("loaded cached features")
    else:
        z = {}
        for ver, mid in versions.items():
            model = load_model_from_id(mid).to(DEV).eval()
            z[f"{ver}_tr"], z[f"{ver}_tr_bs"] = embed(model, tr_img, tr_date, 0)
            for n in names:
                img = scenes[f"{n}_img"]; date = tuple(int(v) for v in scenes[f"{n}_date"])
                for s in SHIFTS:
                    pooled, bs = embed(model, img, date, s)
                    z[f"{ver}_{n}_s{s}"] = pooled
                    if s == 0:
                        z[f"{ver}_{n}_bs"] = bs
            print(f"  embedded {ver}", flush=True)
            del model; torch.cuda.empty_cache()
        np.savez(CACHE, **z)

    # consistency of v1 features between old checkout and new code
    diffs = [np.abs(z[f"v1_{n}_s0"] - old[f"{n}_base0"]).max() for n in names]
    print(f"v1 feature max |diff| new code vs cached: median {np.median(diffs):.2e}, max {np.max(diffs):.2e}")

    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    heads, heads_bs = {}, {}
    for ver in versions:
        torch.manual_seed(0); heads[ver] = train_logistic_head(f32(z[f"{ver}_tr"]), tr_labels)
        heads_bs[ver] = []
        n_bs = z[f"{ver}_tr_bs"].shape[0]
        print(f"{ver}: band-set tokens per patch = {n_bs}")
        for s in range(n_bs):
            torch.manual_seed(0); heads_bs[ver].append(train_logistic_head(f32(z[f"{ver}_tr_bs"][s]), tr_labels))

    rows, per = [], {ver: {} for ver in versions}
    mag = {ver: [] for ver in versions}
    acc = {ver: [] for ver in versions}
    cross = {ver: [] for ver in versions}
    p_all = {}
    for n in names:
        lab = scenes[f"{n}_lab"].astype(bool)
        for ver in versions:
            p_shift = [prob_logit(z[f"{ver}_{n}_s{s}"], *heads[ver])[0] for s in SHIFTS]
            p, logit = prob_logit(z[f"{ver}_{n}_s0"], *heads[ver])
            p_all[(ver, n)] = p
            err = ((p > 0.5) != lab).astype(np.float64)
            acc[ver].append(1 - err.mean())
            tp = aligned_tile_phase(p_shift)
            mag[ver].append(float(tp.mean()))
            sigs = {"confidence": -np.abs(logit), "tile-phase": tp}
            if len(heads_bs[ver]) > 1:
                p_bs = np.stack([prob_logit(z[f"{ver}_{n}_bs"][s], *heads_bs[ver][s])[0] for s in range(len(heads_bs[ver]))])
                sigs["band-set"] = p_bs.std(0)
            e = err.flatten()
            if e.sum() >= 8:
                per[ver][n] = {k: eaurc(v.flatten(), e) for k, v in sigs.items()}
                per[ver][n]["_err"] = e
        # cross-version disagreement, scored against each version's errors
        d = np.abs(p_all[("v1_2", n)] - p_all[("v1", n)])
        for ver in versions:
            if n in per[ver]:
                per[ver][n]["cross-version"] = eaurc(d.flatten(), per[ver][n]["_err"])
        rows.append({"scene": n, "v1_acc": f"{acc['v1'][-1]:.4f}", "v1_2_acc": f"{acc['v1_2'][-1]:.4f}",
                     "v1_tilephase_mean": f"{mag['v1'][-1]:.5f}", "v1_2_tilephase_mean": f"{mag['v1_2'][-1]:.5f}"})

    print(f"\nhead accuracy vs WorldCover, mean over {len(names)} scenes: v1 {np.mean(acc['v1']):.4f}  v1.2 {np.mean(acc['v1_2']):.4f}")
    dm = np.array(mag["v1"]) - np.array(mag["v1_2"])
    print(f"tile-phase magnitude (mean per-patch std across 0-3 px shifts): v1 {np.mean(mag['v1']):.4f}  v1.2 {np.mean(mag['v1_2']):.4f}; "
          f"v1.2 smaller on {int((dm > 0).sum())}/{len(dm)} scenes (sign p={sign_p(int((dm > 0).sum()), int((dm < 0).sum())):.1e})")
    for ver in versions:
        sn = sorted(per[ver]); N = len(sn)
        print(f"\n{ver}: {N} scenes. Signals vs own confidence (E-AURC):")
        for k in ("tile-phase", "band-set", "cross-version"):
            if k not in per[ver][sn[0]]:
                print(f"  {k:<14} n/a (single band-set token per patch in this version)"); continue
            d = np.array([per[ver][s]["confidence"] - per[ver][s][k] for s in sn])
            w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            print(f"  {k:<14} W/L/T {w_:>2}/{l_:>2}/{N - w_ - l_:<2} sign p={sign_p(w_, l_):.1e}  median gain {np.median(d):+.4f}")
        best = Counter(min((k for k in per[ver][s] if not k.startswith('_')), key=lambda k: per[ver][s][k]) for s in sn)
        print("  best per scene:", dict(best))
    with open("exp/out/exp19_v1_vs_v12.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote exp/out/exp19_v1_vs_v12.csv")


if __name__ == "__main__":
    main()
