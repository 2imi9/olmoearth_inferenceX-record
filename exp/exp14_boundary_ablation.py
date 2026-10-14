"""Exp 14: is aligned tile-phase a perturbation signal or a boundary detector?

Aligned tile-phase (exp13) is by construction largest where neighboring
patches disagree. This ablation asks whether a zero-cost proxy computed from
the model's own shift-0 prediction map alone matches it:

  pred-gradient   - gradient magnitude of the Base probability map on the
                    patch grid (no perturbation, no rerun).
  pred-boundary   - fraction of a patch's 8 neighbors whose hard label
                    differs from its own (a discrete boundary indicator).

All are scored with tie-aware E-AURC on the 27 rule-selected exp11 scenes
(two cache leftovers excluded) against the same seed-0 errors, alongside
the pixel control, with per-scene head-to-head counts, exact sign tests, and the
rank correlation between tile-phase and pred-gradient within each scene.
If pred-gradient matches tile-phase, the perturbation adds nothing beyond
boundary proximity and the claim should be restated accordingly.
"""
import csv
import math

import numpy as np
import torch

from oe_inferencex.evidence import train_logistic_head, predict_head, predict_logit
from exp13_stat_corrections import (
    aligned_tile_phase, eaurc, sign_test_p, ndwi_gradient, SHIFTS, GRID, NON_RULE,
)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    scenes = dict(np.load("exp/out/exp11_scenes.npz", allow_pickle=True))
    feats = dict(np.load("exp/out/exp11_feats.npz", allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    torch.manual_seed(0)
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), z3["tr_labels"])

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    rows, per = [], {}
    for name in names:
        if f"{name}_base0" not in feats or name in NON_RULE:
            continue
        lab = scenes[f"{name}_lab"]
        p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
        p = p_shift[0]
        logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
        err = ((p > 0.5) != lab.astype(bool)).astype(np.float64)
        if err.sum() < 8:
            continue
        gy, gx = np.gradient(p)
        grad = np.hypot(gx, gy)
        hard = (p > 0.5).astype(int)
        pad = np.pad(hard, 1, mode="edge")
        nb = np.zeros_like(p)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    nb += (pad[1 + di:1 + di + GRID, 1 + dj:1 + dj + GRID] != hard)
        boundary = nb / 8.0
        sigs = {
            "baseline": -np.abs(logit),
            "tile-phase (aligned)": aligned_tile_phase(p_shift),
            "pred-gradient": grad,
            "pred-boundary": boundary,
            "control": ndwi_gradient(scenes[f"{name}_img"]),
        }
        e = err.flatten()
        val = {k: eaurc(v.flatten(), e) for k, v in sigs.items()}
        rho = spearman(sigs["tile-phase (aligned)"].flatten(), grad.flatten())
        per[name] = (val, rho)
        rows.append({"scene": name, "n_errors": int(err.sum()), **{k: f"{v:.5f}" for k, v in val.items()},
                     "spearman_tile_vs_grad": f"{rho:.3f}"})
        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()) + f", rho={rho:.2f}")

    with open("exp/out/exp14_boundary_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    sn = sorted(per); n = len(sn)
    print(f"\nscenes: {n}")
    for a, b in (("tile-phase (aligned)", "pred-gradient"), ("tile-phase (aligned)", "pred-boundary"),
                 ("pred-gradient", "baseline"), ("pred-boundary", "baseline"), ("tile-phase (aligned)", "baseline"),
                 ("pred-boundary", "control"), ("tile-phase (aligned)", "control")):
        d = np.array([per[s][0][b] - per[s][0][a] for s in sn])  # >0 = a better
        wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        ties = n - wins - losses
        p_sign = sign_test_p(wins, wins + losses) if wins + losses else 1.0
        print(f"{a:<22} < {b:<14}: W/L/T {wins:>2}/{losses:>2}/{ties:<2} sign p={p_sign:.3f}  median E-AURC gain {float(np.median(d)):+.4f}")
    best = {s: min(per[s][0], key=per[s][0].get) for s in sn}
    from collections import Counter
    print("best signal per scene (E-AURC):", dict(Counter(best.values())))
    rhos = [per[s][1] for s in sn]
    print(f"\nwithin-scene Spearman(tile-phase, pred-gradient): median {np.median(rhos):.2f}, "
          f"min {min(rhos):.2f}, max {max(rhos):.2f}")
    print("wrote exp/out/exp14_boundary_ablation.csv")


if __name__ == "__main__":
    main()
