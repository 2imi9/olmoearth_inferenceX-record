"""Exp 18: the signals against dense expert water labels (Sen1Floods11).

Every water-task result so far was scored against ESA WorldCover, a weak
reference. Sen1Floods11 provides hand-labelled flood-water masks
(LabelHand: -1 no data, 0 non-water, 1 water) on 512x512 Sentinel-2 chips,
cut by Ai2's evaluation code into 64x64 tiles and mirrored in the public
research_benchmarks bucket. The Bolivia split is a geographically held-out
region; the valid and test splits share regions with train.

Protocol. A water head is trained on frozen v1-Base tokens of tiles from the
valid split; signals are scored on Bolivia (primary, spatial hold-out) and on
a subsample of the test split. All signals are computed on a 60x60 crop of
each tile (15x15 patches) so that aligned tile-phase (crops at 0-3 px
offsets) and the other signals share one patch grid. Patch labels pool the
hand labels: a patch counts if at least half its pixels are labelled, and is
water if more than half of the labelled pixels are water. Sentinel-2 here is
Level-1C; it is fed through the encoder's L2A path with the L2A normalizer
(a documented mismatch; the head is trained on the same inputs, so relative
rankings of signals are unaffected in kind, but absolute head accuracy is
not comparable to L2A results).

Signals: confidence (negative absolute logit), aligned tile-phase, band-set
disagreement, E_case (Nano head), boundary indicator, E_dist (k-NN to train
patches), and the NDWI-gradient control. Scoring: tie-aware excess AURC per
tile (tiles with at least 3 errors and both classes present), W/L/T versus
confidence with exact sign tests, and a pooled ranking over all patches of a
split. Errors are Base-head disagreements with the hand labels.
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
PATCH, CROP, PAD = 4, 60, 4
G = CROP // PATCH  # 15
SHIFTS = (0, 1, 2, 3)
N_TRAIN_TILES, N_TEST_TILES = 600, 800
BATCH = 32
CACHE = "exp/out/exp18_feats.npz"
PROC_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
OE_BANDS = Modality.SENTINEL2_L2A.band_order  # B02..B09 naming
ALIAS = {"B01": "B1", "B02": "B2", "B03": "B3", "B04": "B4", "B05": "B5", "B06": "B6", "B07": "B7",
         "B08": "B8", "B8A": "B8A", "B09": "B9", "B11": "B11", "B12": "B12"}
BAND_IDX = [PROC_BANDS.index(ALIAS[b]) for b in OE_BANDS]
_norm = Normalizer(Strategy.COMPUTED)


def load_split(name, n=None, seed=0):
    d = torch.load(f"data/floods/flood_{name}_data.pt", weights_only=True)
    s2 = d["s2"].numpy().astype(np.float32)[:, BAND_IDX]  # (N,12,64,64)
    lab = d["labels"].numpy()[:, 0]                        # (N,64,64) in {-1,0,1}
    if n is not None and n < len(s2):
        idx = np.random.default_rng(seed).choice(len(s2), n, replace=False)
        s2, lab = s2[idx], lab[idx]
    return s2, lab


def patch_labels(lab):
    """(N,CROP,CROP) -> water label (N,G,G) float and valid mask (N,G,G) bool."""
    l = lab.reshape(len(lab), G, PATCH, G, PATCH)
    valid = (l >= 0)
    n_valid = valid.sum(axis=(2, 4))
    n_water = (l == 1).sum(axis=(2, 4))
    ok = n_valid >= (PATCH * PATCH) / 2
    water = np.where(ok, n_water > n_valid / 2, False)
    return water.astype(np.float32), ok


def embed(model, tiles, shift):
    """tiles (N,12,64,64) -> pooled (N,G,G,D) and band-set (N,3,G,G,D) tokens at a crop offset."""
    pooled, bsets = [], []
    for i in range(0, len(tiles), BATCH):
        x = tiles[i:i + BATCH, :, shift:shift + CROP, shift:shift + CROP]
        x = x.transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)  # (B,H,W,1,12)
        x = _norm.normalize(Modality.SENTINEL2_L2A, x)
        b = x.shape[0]
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=DEV),
            sentinel2_l2a_mask=torch.ones((b, CROP, CROP, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor([1, 5, 2020], device=DEV)[None, None, :].repeat(b, 1, 1),
        )
        with torch.no_grad():
            out = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a
        pooled.append(out.mean(dim=[3, 4]).half().cpu().numpy())
        bsets.append(out[:, :, :, 0].permute(0, 3, 1, 2, 4).half().cpu().numpy())
    return np.concatenate(pooled), np.concatenate(bsets)


def head_prob_logit(feats, w, b):
    x = torch.tensor(np.asarray(feats, dtype=np.float32)).reshape(-1, feats.shape[-1])
    logit = (x @ w + b).reshape(feats.shape[:-1]).numpy()
    return 1 / (1 + np.exp(-logit)), logit


def aligned_tile_phase(p_shift):
    """p_shift: (S,N,G,G) probability maps from crops at offsets 0..S-1 px."""
    S, N = p_shift.shape[:2]
    canvas = np.full((S, N, CROP + PAD, CROP + PAD), np.nan, dtype=np.float32)
    for s in range(S):
        canvas[s, :, s:s + CROP, s:s + CROP] = np.kron(p_shift[s], np.ones((PATCH, PATCH), dtype=np.float32))
    pix = np.nanstd(canvas, axis=0)  # (N,CROP+PAD,CROP+PAD)
    pix = pix[:, :CROP, :CROP].reshape(N, G, PATCH, G, PATCH)
    return np.nanmean(pix, axis=(2, 4))


def boundary(p):
    hard = (p > 0.5).astype(int)
    pad = np.pad(hard, ((0, 0), (1, 1), (1, 1)), mode="edge")
    nb = np.zeros(hard.shape, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[:, 1 + di:1 + di + G, 1 + dj:1 + dj + G] != hard)
    return nb / 8.0


def ndwi_gradient(tiles):
    x = tiles[:, :, :CROP, :CROP].astype(np.float64)
    g, nir = x[:, OE_BANDS.index("B03")], x[:, OE_BANDS.index("B08")]
    nd = (g - nir) / np.clip(g + nir, 1, None)
    gy, gx = np.gradient(nd, axis=(1, 2))
    return np.hypot(gx, gy).reshape(len(x), G, PATCH, G, PATCH).mean(axis=(2, 4))


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
    torch.manual_seed(0)
    tr_s2, tr_lab = load_split("valid", N_TRAIN_TILES)
    ev = {"bolivia": load_split("bolivia"), "test": load_split("test", N_TEST_TILES, seed=1)}

    if os.path.exists(CACHE):
        z = dict(np.load(CACHE)); print("loaded cached features")
    else:
        base = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval()
        nano = load_model_from_id(ModelID.OLMOEARTH_V1_NANO).to(DEV).eval()
        z = {}
        z["tr_base"], z["tr_bs"] = embed(base, tr_s2, 0)
        z["tr_nano"], _ = embed(nano, tr_s2, 0)
        for name, (s2, _) in ev.items():
            for s in SHIFTS:
                z[f"{name}_base{s}"], bs = embed(base, s2, s)
                if s == 0:
                    z[f"{name}_bs"] = bs
            z[f"{name}_nano"], _ = embed(nano, s2, 0)
            print(f"  embedded {name}", flush=True)
        np.savez(CACHE, **z)

    tr_y, tr_ok = patch_labels(tr_lab[:, :CROP, :CROP])
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    sel = tr_ok.flatten()
    D = z["tr_base"].shape[-1]
    torch.manual_seed(0); hb = train_logistic_head(f32(z["tr_base"]).reshape(-1, D)[sel], tr_y.flatten()[sel])
    torch.manual_seed(0); hn = train_logistic_head(f32(z["tr_nano"]).reshape(-1, z["tr_nano"].shape[-1])[sel], tr_y.flatten()[sel])
    hbs = []
    for s in range(3):
        torch.manual_seed(0)
        hbs.append(train_logistic_head(f32(z["tr_bs"][:, s]).reshape(-1, D)[sel], tr_y.flatten()[sel]))
    a_tr = torch.nn.functional.normalize(f32(z["tr_base"]).reshape(-1, D)[sel], dim=-1).to(DEV)

    rows, summary = [], {}
    for name, (s2, lab) in ev.items():
        y, ok = patch_labels(lab[:, :CROP, :CROP])
        p_shift = np.stack([head_prob_logit(z[f"{name}_base{s}"], *hb)[0] for s in SHIFTS])
        p, logit = head_prob_logit(z[f"{name}_base0"], *hb)
        p_nano, _ = head_prob_logit(z[f"{name}_nano"], *hn)
        p_bs = np.stack([head_prob_logit(z[f"{name}_bs"][:, s], *hbs[s])[0] for s in range(3)])
        err = ((p > 0.5) != (y > 0.5)).astype(np.float64)
        a_ev = torch.nn.functional.normalize(f32(z[f"{name}_base0"]).reshape(-1, D), dim=-1).to(DEV)
        knn = []
        for i in range(0, len(a_ev), 4096):
            knn.append(torch.topk(1 - a_ev[i:i + 4096] @ a_tr.T, k=5, largest=False).values.mean(1).cpu().numpy())
        sigs = {
            "confidence (baseline)": -np.abs(logit),
            "tile-phase (aligned)": aligned_tile_phase(p_shift),
            "band-set disagreement": p_bs.std(0),
            "E_case |Nano-Base|": np.abs(p_nano - p),
            "boundary indicator": boundary(p),
            "E_dist knn-to-train": np.concatenate(knn).reshape(p.shape),
            "control NDWI gradient": ndwi_gradient(s2),
        }
        acc = 1 - err[ok].mean()
        print(f"\n{name}: tiles {len(y)}, valid patches {int(ok.sum())}, Base head accuracy {acc:.3f}, error rate {err[ok].mean():.3f}")
        # pooled ranking over all valid patches of the split
        pooled = {k: eaurc(v[ok], err[ok]) for k, v in sigs.items()}
        # per-tile
        per = {k: [] for k in sigs}
        n_tiles = 0
        for t in range(len(y)):
            m = ok[t]
            e = err[t][m]
            if e.sum() < 3 or e.sum() > len(e) - 3:
                continue
            n_tiles += 1
            for k, v in sigs.items():
                per[k].append(eaurc(v[t][m], e))
        base_t = np.array(per["confidence (baseline)"])
        print(f"  pooled E-AURC / per-tile (n={n_tiles}) W/L vs confidence:")
        summary[name] = {}
        for k in sigs:
            d = base_t - np.array(per[k])
            w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            summary[name][k] = dict(pooled=pooled[k], w=w_, l=l_, p=sign_p(w_, l_), median=float(np.median(d)))
            print(f"    {k:<24} pooled {pooled[k]:.4f}   tiles {w_:>3}/{l_:<3} sign p={sign_p(w_, l_):.1e}  median gain {np.median(d):+.4f}")
        best = Counter(min(sigs, key=lambda k: per[k][i]) for i in range(n_tiles))
        print("    best per tile:", dict(best))
        on_b = sigs["boundary indicator"] > 0
        print(f"    errors on boundary patches {on_b[ok & (err > 0)].mean():.1%} vs correct {on_b[ok & (err == 0)].mean():.1%}")
        rows.append({"split": name, "tiles": len(y), "valid_patches": int(ok.sum()), "head_acc": f"{acc:.4f}", "n_tiles_scored": n_tiles,
                     **{f"{k}|pooled": f"{v['pooled']:.5f}" for k, v in summary[name].items()},
                     **{f"{k}|W/L": f"{v['w']}/{v['l']}" for k, v in summary[name].items()},
                     **{f"{k}|sign_p": f"{v['p']:.2e}" for k, v in summary[name].items()}})

    with open("exp/out/exp18_sen1floods.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote exp/out/exp18_sen1floods.csv")


if __name__ == "__main__":
    main()
