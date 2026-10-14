"""Exp 13: methodological corrections to the 29-scene comparison (exp11).

Three corrections, all recomputed from the cached exp11 features and scenes
(no new inference):

1. Aligned tile-phase. exp05/exp09/exp11 took the std across shifted patch
   grids without aligning them, so a patch at shift 3 px covers different
   ground than at shift 0. Here each shifted prediction map is upsampled to
   pixels, placed at its true offset on a common canvas, and the per-pixel
   std across shifts is pooled back to the shift-0 patch grid (the exp03
   method).
2. Excess AURC (E-AURC, Geifman et al. 2018). Raw AURC scales with a
   scene's error rate, so cross-scene averages and permutation tests are
   dominated by high-error scenes. E-AURC subtracts the oracle AURC (all
   errors ranked last), giving a scale-comparable quantity. Cross-scene
   tests are then run on E-AURC differences and on scale-free ranks: exact
   two-sided sign test and sign-flip permutation on per-scene E-AURC
   differences.
3. Block bootstrap. Patches are spatially autocorrelated, so an i.i.d.
   patch bootstrap understates variance. Here 4x4-patch blocks (64 per
   scene) are resampled with replacement, B=1000, and the 95% interval of
   each signal's E-AURC difference from the baseline is reported per scene.

Errors and heads are those of exp11 (seed 0). Scene set: the rule-selected
exp11 scenes minus two cache leftovers (kafue, luangwa), 27 scenes. Later
corrections (post-audit): tie-aware AURC, negative-|logit| baseline, and
tie-excluding sign tests.
"""
import csv
import math

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from oe_inferencex.evidence import train_logistic_head, predict_head, predict_logit, aurc_expected

SIZE, PATCH, PAD = 128, 4, 4
GRID = SIZE // PATCH
SHIFTS = (0, 1, 2, 3)
BLOCK = 4  # patches per block side -> 8x8 blocks of 4x4 patches = 64 blocks
B_BOOT = 1000
N_PERM = 10000
SIGS = ["baseline", "E_case", "tile-phase (aligned)", "E_dist", "control"]


def aurc(sig, err):
    """Tie-aware AURC (expected value under random tie-breaking)."""
    return aurc_expected(sig, err)


NON_RULE = ("kafue", "luangwa")  # exp09 first-attempt AOIs that entered exp11 via the cache import; excluded


def oracle_aurc(n, k):
    i = np.arange(1, n + 1)
    return float((np.maximum(0, i - (n - k)) / i).mean())


def eaurc(sig, err):
    return aurc(sig, err) - oracle_aurc(len(err), int(err.sum()))


def aligned_tile_phase(shift_probs):
    """shift_probs: list over shifts of (GRID, GRID) maps from views offset
    by s px. Returns (GRID, GRID) std pooled to the shift-0 grid."""
    canvas = np.full((len(shift_probs), SIZE + PAD, SIZE + PAD), np.nan)
    for s, p in enumerate(shift_probs):
        canvas[s, s:s + SIZE, s:s + SIZE] = np.kron(p, np.ones((PATCH, PATCH)))
    pix_std = np.nanstd(canvas, axis=0)
    out = np.zeros((GRID, GRID))
    for i in range(GRID):
        for j in range(GRID):
            blk = pix_std[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]
            out[i, j] = np.nanmean(blk) if np.isfinite(blk).any() else 0.0
    return out


def ndwi_gradient(img):
    bo = Modality.SENTINEL2_L2A.band_order
    x = img[:, :SIZE, :SIZE].astype(np.float64)
    nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(
        x[bo.index("B03")] + x[bo.index("B08")], 1, None)
    gy, gx = np.gradient(nd)
    return np.hypot(gx, gy).reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def sign_test_p(wins, n):
    """Exact two-sided binomial sign test, p=0.5."""
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def block_indices(rng):
    """Resample 4x4-patch blocks with replacement; return flat patch indices."""
    nb = GRID // BLOCK
    blocks = rng.integers(0, nb * nb, nb * nb)
    idx = []
    for b in blocks:
        bi, bj = divmod(int(b), nb)
        rows = np.arange(bi * BLOCK, (bi + 1) * BLOCK)
        cols = np.arange(bj * BLOCK, (bj + 1) * BLOCK)
        idx.extend((r * GRID + c) for r in rows for c in cols)
    return np.array(idx)


def main():
    scenes = dict(np.load("exp/out/exp11_scenes.npz", allow_pickle=True))
    feats = dict(np.load("exp/out/exp11_feats.npz", allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    tr_labels = z3["tr_labels"]
    torch.manual_seed(0)
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), tr_labels)
    hn = train_logistic_head(torch.tensor(feats["tr_nano"]), tr_labels)
    a_tr = torch.nn.functional.normalize(
        torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    rng = np.random.default_rng(0)
    rows, per_scene = [], {}
    for name in names:
        if f"{name}_base0" not in feats or name in NON_RULE:
            continue
        lab = scenes[f"{name}_lab"]
        p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
        p_base = p_shift[0]
        logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
        p_nano = predict_head(torch.tensor(feats[f"{name}_nano"]), *hn)
        err = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
        if err.sum() < 8:
            continue
        f0 = feats[f"{name}_base0"]
        a_ev = torch.nn.functional.normalize(torch.tensor(f0).reshape(-1, f0.shape[-1]), dim=-1)
        knn = torch.topk(1 - (a_ev @ a_tr.T), k=5, largest=False).values.mean(1).reshape(GRID, GRID).numpy()
        sigs = {
            "baseline": -np.abs(logit),  # monotone in max-softmax, tie-free where the sigmoid saturates
            "E_case": np.abs(p_nano - p_base),
            "tile-phase (aligned)": aligned_tile_phase(p_shift),
            "tile-phase (unaligned, exp11)": np.stack(p_shift).std(0),
            "E_dist": knn,
            "control": ndwi_gradient(scenes[f"{name}_img"]),
        }
        e_flat = err.flatten()
        s_flat = {k: v.flatten() for k, v in sigs.items()}
        point = {k: eaurc(v, e_flat) for k, v in s_flat.items()}
        boot_diff = {k: [] for k in sigs if k != "baseline"}
        for _ in range(B_BOOT):
            idx = block_indices(rng)
            eb = e_flat[idx]
            if eb.sum() == 0:
                continue
            base_b = eaurc(s_flat["baseline"][idx], eb)
            for k in boot_diff:
                boot_diff[k].append(eaurc(s_flat[k][idx], eb) - base_b)
        per_scene[name] = point
        for k in sigs:
            row = {"scene": name, "n_errors": int(err.sum()), "signal": k,
                   "aurc": aurc(s_flat[k], e_flat), "eaurc": point[k]}
            if k != "baseline":
                bd = np.array(boot_diff[k])
                row.update({"diff_vs_baseline": point[k] - point["baseline"],
                            "diff_ci_lo": float(np.percentile(bd, 2.5)),
                            "diff_ci_hi": float(np.percentile(bd, 97.5)),
                            "frac_better": float((bd < 0).mean())})
            rows.append(row)
        print(f"{name}: errors {int(err.sum())}, "
              + ", ".join(f"{k}={point[k]:.4f}" for k in sigs))

    keys = ["scene", "n_errors", "signal", "aurc", "eaurc", "diff_vs_baseline",
            "diff_ci_lo", "diff_ci_hi", "frac_better"]
    with open("exp/out/exp13_corrected_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    sn = sorted(per_scene)
    n = len(sn)
    print(f"\nscenes: {n}. Cross-scene tests on E-AURC (scale-comparable):")
    print(f"{'signal':<30}{'median diff':>12}{'W/L/T':>10}{'sign p':>9}{'perm p':>9}{'CI excl 0':>11}")
    print("(sign test on untied pairs; permutation is the mean-difference sign-flip test, which the")
    print(" oracle subtraction cannot change and which is driven by the high-error scenes)")
    rng2 = np.random.default_rng(1)
    for k in ["E_case", "tile-phase (aligned)", "tile-phase (unaligned, exp11)", "E_dist", "control"]:
        d = np.array([per_scene[s]["baseline"] - per_scene[s][k] for s in sn])  # >0 = signal better
        wins, losses = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        ties = n - wins - losses
        perm = np.array([(d * rng2.choice([-1, 1], n)).mean() for _ in range(N_PERM)])
        p_perm = float((np.abs(perm) >= abs(d.mean())).mean())
        sig_rows = [r for r in rows if r["signal"] == k]
        excl = sum(1 for r in sig_rows if r["diff_ci_hi"] < 0) if sig_rows and "diff_ci_hi" in sig_rows[0] else 0
        excl_worse = sum(1 for r in sig_rows if r["diff_ci_lo"] > 0) if sig_rows and "diff_ci_lo" in sig_rows[0] else 0
        p_sign = sign_test_p(wins, wins + losses) if wins + losses else 1.0
        print(f"{k:<30}{np.median(d):>+12.4f}{wins:>3}/{losses}/{ties:<3}{p_sign:>9.3f}{p_perm:>9.3f}"
              f"{excl:>6}b/{excl_worse}w")
    best = {s: min(per_scene[s], key=per_scene[s].get) for s in sn}
    from collections import Counter
    print("\nbest signal per scene (E-AURC):", dict(Counter(best.values())))
    print("wrote exp/out/exp13_corrected_stats.csv")


if __name__ == "__main__":
    main()
