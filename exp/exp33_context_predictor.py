"""exp33: a latent-MIM predictor with a whitened target on frozen features (context-prediction residual).

exp28 read the shipped decoder's masked-token residual and found nothing;
exp32 showed why: its targets are frozen random projections of the raw patch
with effective rank 2, so the pretraining loss can mostly tell bright from
dark. This experiment keeps the latent-MIM reading and replaces the target.
The frozen encoder's pooled 768-d patch tokens are PCA-whitened in their top
d directions (fitted on the predictor's training units, so the target has
unit variance in every retained direction), and a small transformer
predictor is trained label-free to reconstruct hidden patches from the rest
of the unit. At inference K = 8 masks hide 25% of the patches each (two
rounds of four disjoint quarters, every patch hidden exactly twice, as
exp28) and a patch's score is its mean squared residual in the whitened
space over the masks that hid it. Two companions from the same forward:
the residual measured as cosine distance, and the residual of the trivial
predictor that outputs the mean, which is the top-d Mahalanobis distance of
exp31, and the residual of the same trained predictor with every patch
hidden, its position-only prior; the per-unit context gain
1 - MSE(25% hidden) / MSE(all hidden) says whether context predicts anything
beyond position, which the zero baseline cannot tell (a position code alone
learns spatially varying means).

Preregistered: primary score = the whitened MSE residual (K = 8); U+ = mean
of the within-unit midrank percentiles of confidence and of that score;
inference = per-river mean gain of U+ over confidence, one-sided exact sign
test over the 8 rivers (7/8 gives p = 0.035). Null hypothesis: the residual
is a boundary detector (a patch unlike its neighbours), so it wins against
WorldCover and loses on hand labels; its Spearman with the boundary
indicator and the S2 patch-variance control is recorded.

Training sets. Part A: the 27 rule scenes plus the katima training scene,
cross-fitted over three river-disjoint folds (Zambezi + Luangwa; Cuando +
Kafue + Okavango; Rovuma + Save + Shire): each scene is scored by a
predictor that never saw its river; katima, which lies on the Zambezi, is in
the two pools that do not hold the Zambezi out. The
training pools take every rule scene's features whether or not the scene
passes the eight-error scoring rule, so no evaluation label shapes a fit. Part
B: the 600 valid-split tiles (the head's training tiles, label-free here),
scored on Bolivia. No evaluation label enters any fit. Predictor: tokens
projected to width 128 with a sinusoidal 2-d position code, a learned mask
token, two pre-norm transformer layers with four heads, an output head to
the d whitened dimensions, MSE on hidden positions, Adam 1e-3, 400 steps
with a fresh 25% random mask per unit per step; seed = fold index in part
A, 0 in part B. fp32.

Evaluation, controls and scaffolding: exp/harness_ab.py (exp28 pattern).
Inputs: exp/out/exp11_feats.npz and exp/out/exp18_feats.npz (cluster
caches; recomputed when missing), exp/out/exp11_scenes.npz,
exp/out/exp03_cache.npz, data/floods/. Outputs:
exp/out/exp33_context_predictor.csv, exp33_summary.json, exp33_cache.npz.
--smoke: CPU, 2 scenes at 64 px, 4 flood tiles, 60 steps, _smoke files.
"""
import math
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

import harness_ab as hb
from harness_ab import CONF, TILE, BOUND, CTRL, DEV, RIVER  # noqa: F401

D_WHITE = 64
WIDTH, HEADS, LAYERS, FF = 128, 4, 2, 256
STEPS, LR, MASK_FRAC, BATCH_UNITS = 400, 1e-3, 0.25, 64
K_ROUNDS = 2                       # 2 rounds x 4 quarters = 8 masks, every patch hidden twice
FOLDS_A = ({"Zambezi", "Luangwa"}, {"Cuando", "Kafue", "Okavango"}, {"Rovuma", "Save", "Shire"})
RES, RES_COS, MAHA, NOCTX = ("predictor residual, whitened MSE (K=8)", "predictor residual, cosine (K=8)",
                             "mean-predictor residual (Mahalanobis top-d)", "position-only residual (all patches hidden)")
NEW_NAMES = [RES, RES_COS, MAHA, NOCTX]
PRIMARY = RES
COMBO = "combination conf+predictor (prereg)"


# ----------------------------------------------------------------------------- whitening
def fit_whitener(X, d=D_WHITE):
    """PCA whitening of (n, D) float32 features in the top-d directions; returns a dict and the variance kept."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    cov = Xc.T @ Xc / len(X)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:d]
    lam, P = np.clip(evals[order], 1e-8, None), evecs[:, order]
    return {"mu": mu, "P": P, "scale": 1 / np.sqrt(lam), "var_kept": float(evals[order].sum() / evals.sum()), "d": d}


def whiten(w, X):
    return ((np.asarray(X, dtype=np.float64) - w["mu"]) @ w["P"] * w["scale"]).astype(np.float32)


# ----------------------------------------------------------------------------- predictor
def pos_code(G, width):
    """Sinusoidal 2-d position code (G*G, width)."""
    ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    q = width // 4
    freqs = 1.0 / (10000 ** (np.arange(q) / q))
    code = np.concatenate([np.sin(ii.reshape(-1, 1) * freqs), np.cos(ii.reshape(-1, 1) * freqs),
                           np.sin(jj.reshape(-1, 1) * freqs), np.cos(jj.reshape(-1, 1) * freqs)], axis=1)
    return torch.tensor(code, dtype=torch.float32)


class Predictor(nn.Module):
    def __init__(self, d, G):
        super().__init__()
        self.inp = nn.Linear(d, WIDTH)
        self.mask_token = nn.Parameter(torch.zeros(WIDTH))
        self.register_buffer("pos", pos_code(G, WIDTH))
        layer = nn.TransformerEncoderLayer(WIDTH, HEADS, FF, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, LAYERS)
        self.out = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, d))

    def forward(self, z, hidden):
        """z (B, G*G, d) whitened tokens; hidden (B, G*G) bool -> predictions (B, G*G, d)."""
        h = self.inp(z)
        h = torch.where(hidden[..., None], self.mask_token.expand_as(h), h) + self.pos[None]
        return self.out(self.blocks(h))


def train_predictor(Z, G, steps, seed=0, log=""):
    """Z (n_units, G*G, d) whitened tokens -> trained predictor and the training-loss trace."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Predictor(Z.shape[-1], G).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Zt = torch.tensor(Z, device=DEV)
    n = len(Z)
    trace = []
    t0 = time.time()
    for step in range(steps):
        idx = rng.choice(n, min(BATCH_UNITS, n), replace=False)
        hidden = torch.tensor(rng.random((len(idx), Z.shape[1])) < MASK_FRAC, device=DEV)
        pred = model(Zt[idx], hidden)
        loss = ((pred - Zt[idx]) ** 2).mean(-1)[hidden].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        trace.append(float(loss))
        if log and (step % 100 == 0 or step == steps - 1):
            print(f"  {log} step {step}: loss {float(loss):.4f} ({time.time() - t0:.0f}s)", flush=True)
    return model.eval(), trace


def quarter_masks(n, rounds, rng):
    masks = []
    for _ in range(rounds):
        perm = rng.permutation(n)
        for chunk in np.array_split(perm, 4):
            m = np.zeros(n, dtype=bool)
            m[chunk] = True
            masks.append(m)
    return np.stack(masks)


@torch.no_grad()
def score_units(model, Z, G, rng):
    """Z (n_units, G*G, d) -> per-patch residuals (MSE, cosine) averaged over the masks that hid the patch, and the
    mean-predictor residual; plus per-unit context gain."""
    n_units, n_tok, d = Z.shape
    mse = np.zeros((n_units, n_tok))
    cosd = np.zeros((n_units, n_tok))
    cnt = np.zeros((n_units, n_tok))
    Zt = torch.tensor(Z, device=DEV)
    all_hidden = torch.ones((n_units, n_tok), dtype=torch.bool, device=DEV)
    nocontext = ((model(Zt, all_hidden) - Zt) ** 2).mean(-1).cpu().numpy()       # position-only prior: no context at all
    for u in range(n_units):
        masks = quarter_masks(n_tok, K_ROUNDS, rng)
        hidden = torch.tensor(masks, device=DEV)
        pred = model(Zt[u][None].expand(len(masks), -1, -1), hidden)
        tgt = Zt[u][None].expand(len(masks), -1, -1)
        e = ((pred - tgt) ** 2).mean(-1).cpu().numpy()
        c = (1 - torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)).cpu().numpy()
        mse[u] += (e * masks).sum(0)
        cosd[u] += (c * masks).sum(0)
        cnt[u] += masks.sum(0)
    mse /= cnt
    cosd /= cnt
    maha = (Z ** 2).mean(-1)                                   # the mean predictor (zero in whitened space)
    gain = 1 - mse.mean(1) / np.maximum(nocontext.mean(1), 1e-12)   # against the position-only prior, not the zero predictor
    return {"mse": mse.reshape(n_units, G, G), "cos": cosd.reshape(n_units, G, G), "maha": maha.reshape(n_units, G, G),
            "nocontext": nocontext.reshape(n_units, G, G), "context_gain_per_unit": gain,
            "gain_vs_zero_per_unit": 1 - mse.mean(1) / np.maximum(maha.mean(1), 1e-12)}


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    G = ctx["G"]
    D = ctx["tr_feats"].shape[-1]
    units = {}
    for name in ctx["names"]:
        try:
            units[name] = hb.scene_unit(ctx, model, name, args, summary)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
    # training pools use every rule scene's shift-0 features (scene_unit computes them before its error filter), so
    # membership never depends on the evaluation labels; the error filter applies only to which scenes are scored.
    # Cache-only scenes outside the rule set (no river cluster) are excluded from both pools and scoring.
    all_names = [n for n in ctx["names"] if n in hb.RULE_SCENES and f"{n}_base0" in ctx["feats"]]   # rule scenes only
    names = [n for n in all_names if units.get(n) is not None]
    feats = {n: np.asarray(ctx["feats"][f"{n}_base0"], dtype=np.float32).reshape(G * G, D) for n in all_names}
    feats["__katima__"] = ctx["tr_feats"].reshape(G * G, D)
    folds = FOLDS_A if not args.smoke else tuple({RIVER[n]} for n in all_names)
    rng = np.random.default_rng(0)
    scores, diag = {}, {}
    for f, rivers in enumerate(folds):
        held = [n for n in names if RIVER[n] in rivers]                              # scored scenes of this fold
        train = [n for n in all_names if RIVER[n] not in rivers]                       # every other river, labels unseen
        if "Zambezi" not in rivers:                                                    # katima lies on the Zambezi
            train.append("__katima__")
        if not held:
            continue
        try:
            Xtr = np.concatenate([feats[n] for n in train])
            w = fit_whitener(Xtr, args.d)
            Ztr = np.stack([whiten(w, feats[n]) for n in train])
            pred, trace = train_predictor(Ztr, G, args.steps, seed=f, log=f"fold {f} ({len(train)} units)")
            Zev = np.stack([whiten(w, feats[n]) for n in held])
            out = score_units(pred, Zev, G, rng)
            for i, n in enumerate(held):
                scores[n] = {k: out[k][i] for k in ("mse", "cos", "maha", "nocontext")}
                diag[n] = {"fold": f, "context_gain_vs_position_only": float(out["context_gain_per_unit"][i]),
                           "gain_vs_zero_predictor": float(out["gain_vs_zero_per_unit"][i])}
            summary["part_a"].setdefault("folds", {})[str(f)] = {
                "held": held, "n_train_units": len(train), "var_kept_by_whitening": w["var_kept"],
                "train_loss_first_last": [trace[0], trace[-1]], "seed": f,
                "mean_context_gain_vs_position_only_held": float(out["context_gain_per_unit"].mean()),
                "mean_gain_vs_zero_predictor_held": float(out["gain_vs_zero_per_unit"].mean())}
            print(f"fold {f}: whitening keeps {w['var_kept']:.3f} of variance; train loss {trace[0]:.3f} -> {trace[-1]:.3f}; "
                  f"held-out context gain {out['context_gain_per_unit'].mean():.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": f"fold{f}", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"fold {f}: FAILED {ex!r}", flush=True)
    per, rho_per = {}, {}
    for name in names:
        if name not in scores:
            continue
        try:
            unit = units[name]
            sigs = hb.base_signals_a(unit, ctx)
            sigs[RES], sigs[RES_COS], sigs[MAHA], sigs[NOCTX] = (scores[name]["mse"], scores[name]["cos"],
                                                                  scores[name]["maha"], scores[name]["nocontext"])
            sigs[COMBO] = hb.combination(sigs, PRIMARY)
            for k, key in ((RES, "mse"), (RES_COS, "cos"), (MAHA, "maha"), (NOCTX, "nocontext")):
                cache[f"{name}_{key}"] = np.asarray(sigs[k], dtype=np.float32)
            cache[f"{name}_err"] = unit["err"].astype(np.float32)
            val = hb.score_scene(unit, sigs, rows, per, rho_per, NEW_NAMES)
            diag[name]["spearman_residual_vs_s2_variance"] = hb.exp14.spearman(sigs[RES].flatten(), sigs[hb.CTRL_VAR].flatten())
            print(f"{name}: {unit['n_err']} errors, conf {val[CONF]:.4f} tile {val[TILE]:.4f} residual {val[RES]:.4f} "
                  f"maha {val[MAHA]:.4f} U+ {val[COMBO]:.4f} | context gain {diag[name]['context_gain_vs_position_only']:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    summary["part_a"]["diagnostics"] = diag
    hb.finish_part_a(summary, per, rho_per, NEW_NAMES, PRIMARY, COMBO)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp33")
    if ctx is None:
        return
    N, G = ctx["N"], ctx["G"]
    D = ctx["ev_feats"].shape[-1]
    z = ctx["z"]
    tr_units = np.asarray(z["tr_base"], dtype=np.float32).reshape(-1, G * G, D)     # all patches of the training tiles, label-free
    w = fit_whitener(tr_units.reshape(-1, D), args.d)
    Ztr = np.stack([whiten(w, u) for u in tr_units])
    pred, trace = train_predictor(Ztr, G, args.steps, seed=0, log=f"part B ({len(Ztr)} tiles)")
    Zev = np.stack([whiten(w, u) for u in ctx["ev_feats"].reshape(N, G * G, D)])
    out = score_units(pred, Zev, G, np.random.default_rng(0))
    summary["part_b"]["predictor"] = {"n_train_tiles": int(len(Ztr)), "var_kept_by_whitening": w["var_kept"],
                                      "train_loss_first_last": [trace[0], trace[-1]],
                                      "seed": 0, "d": args.d,
                                      "mean_context_gain_vs_position_only_bolivia": float(out["context_gain_per_unit"].mean()),
                                      "median_context_gain_vs_position_only_bolivia": float(np.median(out["context_gain_per_unit"])),
                                      "mean_gain_vs_zero_predictor_bolivia": float(out["gain_vs_zero_per_unit"].mean())}
    print(f"part B: whitening keeps {w['var_kept']:.3f}; train loss {trace[0]:.3f} -> {trace[-1]:.3f}; "
          f"Bolivia context gain vs position-only {out['context_gain_per_unit'].mean():.3f}", flush=True)
    sigs = hb.base_signals_b(ctx)
    sigs[RES], sigs[RES_COS], sigs[MAHA], sigs[NOCTX] = out["mse"], out["cos"], out["maha"], out["nocontext"]
    for k, key in ((RES, "mse"), (RES_COS, "cos"), (MAHA, "maha"), (NOCTX, "nocontext")):
        cache[f"bolivia_{key}"] = np.asarray(sigs[k], dtype=np.float32)
    cache["bolivia_err"] = ctx["err"].astype(np.float32)
    cache["bolivia_ok"] = ctx["ok"]
    ok = ctx["ok"]
    summary["part_b"]["diagnostics"] = {"spearman_residual_vs_s2_variance_pooled": hb.exp14.spearman(sigs[RES][ok], sigs[hb.CTRL_VAR][ok]),
                                        "spearman_residual_vs_boundary_pooled": hb.exp14.spearman(sigs[RES][ok], sigs[BOUND][ok])}
    hb.finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO)


def main():
    def extra(ap):
        ap.add_argument("--steps", type=int, default=None, help=f"predictor training steps (default {STEPS}; smoke 60)")
        ap.add_argument("--d", type=int, default=D_WHITE, help=f"whitened dimensions (default {D_WHITE})")
    args = hb.make_parser(__doc__, extra).parse_args()
    args.steps = args.steps or (60 if args.smoke else STEPS)
    config = {"whitening": f"PCA whitening in the top {args.d} directions, fitted on the predictor's training units",
              "d": args.d, "fold_seeds_a": "fold index (0, 1, 2)", "seed_b": 0,
              "predictor": f"width {WIDTH}, {LAYERS} pre-norm transformer layers, {HEADS} heads, ff {FF}, sinusoidal 2-d positions, learned mask token",
              "training": f"MSE on hidden positions, {MASK_FRAC} random mask per unit per step, Adam {LR}, {args.steps} steps, batch {BATCH_UNITS} units",
              "scoring": f"{K_ROUNDS} rounds of 4 disjoint quarter masks (K = {4 * K_ROUNDS}), every patch hidden {K_ROUNDS} times",
              "folds_a": [sorted(f) for f in FOLDS_A], "primary_score": PRIMARY, "combination": COMBO,
              "null": "the residual is a boundary detector: wins against WorldCover, loses on hand labels"}
    hb.run("exp33", "exp33 context-prediction residual with a whitened target", config, part_a, part_b, args, "exp33_context_predictor")


if __name__ == "__main__":
    main()
