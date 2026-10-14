"""exp30: last-layer Laplace uncertainty of the probe head as a label-free error signal.

Every probe in this repository is a logistic head on frozen 768-d pooled
tokens, and its confidence (negative absolute logit) has been the best error
ranker on every expert-labelled testbed. Confidence is the head's point
estimate; it says nothing about how well determined the head's weights are
in the direction of a given feature. A last-layer Laplace approximation
(MacKay 1992; Kristiadi, Hein and Hennig 2020; Daxberger et al. 2021) puts a
Gaussian posterior on the head, N(theta_MAP, (H + lambda I)^-1), with H the
Hessian of the training objective at the trained weights and lambda a prior
precision, and gives every patch a logit variance v(x) = phi(x)^T Sigma
phi(x), phi = [x, 1]. Large v means the training patches did not pin the
head down along x: an epistemic term that confidence cannot contain. The
head's objective is the balanced binary cross-entropy of
oe_inferencex.evidence.train_logistic_head (positive weight n_neg / n_pos),
so H = sum_i w_i p_i (1 - p_i) phi_i phi_i^T with w_i the positive weight for
water patches and 1 otherwise. lambda is chosen post hoc by maximising the
Laplace marginal likelihood on a log grid with the trained weights held
fixed (the laplace-torch default). Because the head was trained without a
prior, the trained weights are not the stationary point of the regularised
objective, so this is an evidence surrogate rather than a Laplace evidence at
a MAP; the residual gradient ||grad NLL(theta) + lambda theta|| is recorded
next to ||grad NLL(theta)||, and the sensitivity of the ranking to lambda is
reported. Because the training set of part A is a single 32x32 scene
(1024 patches in 768 dimensions) the head fits it perfectly and H is nearly
singular, so there v is dominated by the prior term and measures how far x
leaves the subspace spanned by the training scene: the Bayesian form of
exp13's E_dist. Part B trains on 135k patches, where H is well conditioned.

Signals (higher = more uncertain), all from one fit per part:
  Laplace logit variance          v(x); the preregistered primary score.
  Laplace moderated confidence    -|mu| / sqrt(1 + pi v / 8), the probit
                                  approximation to the posterior predictive
                                  (MacKay); a principled fusion of
                                  confidence and variance.
  Laplace mutual information      H[E p] - E[H p] under l ~ N(mu, v), by
                                  numerical integration over the logit on a
                                  4001-point grid covering the Gaussian mass
                                  within the window where the entropy is
                                  non-zero (|l| < 45), with the tail mass
                                  added in closed form; a fixed Gauss-Hermite
                                  rule mis-orders patches once v exceeds a
                                  few hundred, which part A produces. The
                                  BALD epistemic score.
  Laplace predictive entropy      H[E p], the same integration.
  bootstrap-head logit std        the frequentist counterpart: 16 heads
                                  retrained on bootstrap resamples of the
                                  training patches, std of their logits.
  diagnostic feature norm         ||x||, to show whether a score is norm.
Controls and references as exp28: confidence (negative absolute logit),
aligned tile-phase, boundary indicator, NDWI gradient, constant score, S2
within-patch variance, NDWI level. Preregistered inference: U+ = mean of the
within-unit midrank percentiles of confidence and of the logit variance;
gain of U+ over confidence averaged within river, one-sided exact sign test
over the eight river clusters (7/8 gives p = 0.035). Everything else is
secondary and reported with the same tests. Errors are always those of the
original head; no evaluation label enters any fit.

Evaluation. (A) The 27 rule-selected exp11 scenes against the exp13
disagreement set (seed-0 Base head trained on katima, errors against
WorldCover 2021, scenes with fewer than 8 errors skipped). (B) Sen1Floods11
Bolivia hand labels exactly as exp18 (600 valid-split training tiles, 60x60
crops, same head, same patch-label pooling, same per-tile rule), both through
exp/harness_ab.py.

Inputs. exp/out/exp11_scenes.npz, exp/out/exp03_cache.npz (committed);
exp/out/exp11_feats.npz and exp/out/exp18_feats.npz (gitignored caches;
recomputed with the same code paths when missing); the Base checkpoint from
the HF cache; data/floods/flood_{valid,bolivia}_data.pt.

Outputs. exp/out/exp30_laplace_head.csv (one row per unit x signal),
exp30_summary.json (tests, prereg, Laplace diagnostics: lambda, marginal
likelihood curve, effective parameters, lambda sensitivity), cache
exp30_cache.npz (per-unit score arrays). --smoke runs on CPU with 2 scenes
at 64 px and 4 flood tiles and writes _smoke files.

fp32 only for the encoder: no autocast, no TF32, no compile. The Laplace
algebra runs in float64.
"""
import time
import traceback

import numpy as np
import torch

import harness_ab as hb
from harness_ab import CONF, TILE, BOUND, CTRL, CONST, CTRL_VAR, CTRL_LVL, REFERENCES  # noqa: F401
from oe_inferencex.evidence import train_logistic_head

LAMBDA_GRID = np.logspace(-4, 6, 51)
INT_NODES = 4001          # logit-grid points per patch for the predictive integrals
INT_HALF_WINDOW = 45.0    # |l| beyond which the Bernoulli entropy is below 1e-18
N_BOOT = 16
LAM_SENS = (0.01, 100.0)   # lambda multipliers for the ranking-sensitivity diagnostic

VAR, MOD, MI, ENT, BOOT, NORM = ("Laplace logit variance", "Laplace moderated confidence (probit)",
                                 "Laplace mutual information", "Laplace predictive entropy",
                                 f"bootstrap-head logit std (B={N_BOOT})", "diagnostic feature norm")
NEW_NAMES = [VAR, MOD, MI, ENT, BOOT, NORM]
PRIMARY = VAR
COMBO = "combination conf+Laplace (prereg)"


def _phi(x):
    """(n, D) float64 features -> (n, D+1) with the bias column."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def _bce_weighted_sum(logit, y, pos_w):
    """Summed balanced BCE, the objective train_logistic_head minimises (times n)."""
    lp = -np.logaddexp(0.0, -logit)   # log sigmoid
    ln = -np.logaddexp(0.0, logit)    # log (1 - sigmoid)
    return float(-(pos_w * y * lp + (1 - y) * ln).sum())


def fit_laplace(x_tr, y_tr, w, b):
    """Last-layer Laplace posterior of the head (w, b) on its training set.

    Returns a dict with the eigendecomposition of the weighted Hessian, the
    prior precision chosen by the marginal likelihood, and diagnostics."""
    phi = _phi(x_tr)
    y = np.asarray(y_tr, dtype=np.float64).ravel()
    theta = np.concatenate([np.asarray(w, dtype=np.float64).ravel(), [float(b)]])
    pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))
    logit = phi @ theta
    p = 0.5 * (1 + np.tanh(logit / 2))
    wt = np.where(y == 1, pos_w, 1.0) * p * (1 - p)
    H = (phi * wt[:, None]).T @ phi                               # (P, P) Hessian of the summed objective
    h, Q = np.linalg.eigh(H)
    h = np.clip(h, 0.0, None)
    P = len(theta)
    nll = _bce_weighted_sum(logit, y, pos_w)
    lml = np.array([-nll - 0.5 * lam * theta @ theta + 0.5 * P * np.log(lam) - 0.5 * np.log(h + lam).sum()
                    for lam in LAMBDA_GRID])
    lam = float(LAMBDA_GRID[int(np.argmax(lml))])
    gamma = float(P - lam * (1.0 / (h + lam)).sum())              # effective number of parameters
    g_nll = phi.T @ (np.where(y == 1, pos_w, 1.0) * (p - y))       # gradient of the summed weighted BCE at theta
    return {"theta": theta, "Q": Q, "h": h, "lam": lam, "pos_w": pos_w, "P": P,
            "diag": {"lambda": lam, "lambda_grid_edge": bool(lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1])),
                     "effective_params": gamma, "train_nll_sum": nll, "n_train": int(len(y)),
                     "train_acc": float(((logit > 0) == (y > 0.5)).mean()),
                     "hessian_eig_max": float(h.max()), "hessian_eig_min": float(h.min()),
                     "hessian_rank_1e-8": int((h > 1e-8 * h.max()).sum()),
                     "nll_grad_norm": float(np.linalg.norm(g_nll)),
                     "posterior_grad_norm": float(np.linalg.norm(g_nll + lam * theta)),
                     "theta_norm": float(np.linalg.norm(theta)),
                     "lml_curve": {"lambda": LAMBDA_GRID.tolist(), "lml": lml.tolist()},
                     "pos_weight": pos_w}}


def laplace_predict(lap, x, lam=None):
    """Mean logit and logit variance of every row of x under the Laplace posterior (optionally at another lambda)."""
    phi = _phi(x)
    lam = lap["lam"] if lam is None else lam
    mu = phi @ lap["theta"]
    proj = phi @ lap["Q"]
    v = (proj ** 2 / (lap["h"] + lam)[None, :]).sum(1)
    return mu, v


def _entropy(q):
    eps = 1e-300
    return -(q * np.log(q + eps) + (1 - q) * np.log(1 - q + eps))


def _ndtr(z):
    """Standard normal CDF."""
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))


def gh_scores(mu, v, chunk=2048):
    """Predictive entropy and mutual information of a Bernoulli with logit l ~ N(mu, v).

    E[sigmoid(l)] and E[H(sigmoid(l))] are integrated on a uniform grid of
    INT_NODES points over [max(mu - 8 s, -W), min(mu + 8 s, W)], s = sqrt(v),
    W = INT_HALF_WINDOW, by the trapezoid rule; outside |l| < W the entropy
    is numerically zero and sigmoid is 0 or 1, so the mass above the window
    is added to E[sigmoid] in closed form. This resolves the sigmoid
    transition for any v, which a fixed Gauss-Hermite rule does not once v
    exceeds a few hundred."""
    mu = np.asarray(mu, dtype=np.float64).ravel()
    s = np.sqrt(np.clip(np.asarray(v, dtype=np.float64).ravel(), 1e-300, None))
    W = INT_HALF_WINDOW
    pred_ent, mi = np.empty(len(mu)), np.empty(len(mu))
    u = np.linspace(0.0, 1.0, INT_NODES)
    for i0 in range(0, len(mu), chunk):
        m, sd = mu[i0:i0 + chunk], s[i0:i0 + chunk]
        lo, hi = np.maximum(m - 8 * sd, -W), np.minimum(m + 8 * sd, W)
        ok = hi > lo
        lo_c, hi_c = np.where(ok, lo, m), np.where(ok, hi, m + 1e-6)
        l = lo_c[:, None] + (hi_c - lo_c)[:, None] * u[None, :]
        z = (l - m[:, None]) / sd[:, None]
        pdf = np.exp(-0.5 * z * z) / (sd[:, None] * np.sqrt(2 * np.pi))
        p = 0.5 * (1 + np.tanh(l / 2))
        w = np.full(INT_NODES, 1.0); w[0] = w[-1] = 0.5
        dl = (hi_c - lo_c) / (INT_NODES - 1)
        p_bar = (pdf * p * w[None, :]).sum(1) * dl + (1 - _ndtr((hi_c - m) / sd))
        exp_ent = (pdf * _entropy(p) * w[None, :]).sum(1) * dl
        p_bar = np.where(ok, p_bar, 1 - _ndtr((W - m) / sd))   # all mass outside the window
        exp_ent = np.where(ok, exp_ent, 0.0)
        p_bar = np.clip(p_bar, 0.0, 1.0)
        pe = _entropy(p_bar)
        pred_ent[i0:i0 + chunk] = pe
        mi[i0:i0 + chunk] = pe - exp_ent
    return pred_ent, mi


def bootstrap_heads(x_tr, y_tr, n_boot, seed=0):
    """Heads retrained on bootstrap resamples of the training patches (the head's own recipe)."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(np.asarray(x_tr, dtype=np.float32)).reshape(-1, np.shape(x_tr)[-1])
    y = np.asarray(y_tr).ravel()
    heads = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        heads.append(train_logistic_head(x[idx], y[idx]))
    return heads


def bootstrap_std(heads, x):
    xt = torch.tensor(np.asarray(x, dtype=np.float32)).reshape(-1, np.shape(x)[-1])
    logits = np.stack([(xt @ w + b).numpy() for w, b in heads])
    return logits.std(0)


def laplace_signals(lap, heads, x, shape):
    """All exp30 signals for features x (n, D), reshaped to `shape`."""
    mu, v = laplace_predict(lap, x)
    pred_ent, mi = gh_scores(mu, v)
    xn = np.linalg.norm(np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1]), axis=1)
    sigs = {VAR: v, MOD: -np.abs(mu) / np.sqrt(1 + np.pi * v / 8), MI: mi, ENT: pred_ent,
            BOOT: bootstrap_std(heads, x), NORM: xn}
    sens = {}
    for m in LAM_SENS:
        _, v2 = laplace_predict(lap, x, lam=lap["lam"] * m)
        sens[f"spearman_var_at_{m}x_lambda"] = hb.exp14.spearman(v, v2)
    sens["spearman_var_vs_norm"] = hb.exp14.spearman(v, xn)
    sens["mean_logit_variance"] = float(v.mean())
    return {k: np.asarray(val).reshape(shape) for k, val in sigs.items()}, sens


def head_check(lap, heads, x_tr, y_tr, w, b):
    """Consistency of the Laplace MAP with the head and the variance on the training patches (diagnostic)."""
    mu, v = laplace_predict(lap, x_tr)
    xt = torch.tensor(np.asarray(x_tr, dtype=np.float32)).reshape(-1, np.shape(x_tr)[-1])
    ref = (xt @ w + b).numpy()
    return {"max_abs_logit_diff_vs_head": float(np.abs(mu - ref).max()), "train_mean_logit_variance": float(v.mean()),
            "train_bootstrap_mean_std": float(bootstrap_std(heads, x_tr).mean()),
            "bootstrap_train_acc_mean": float(np.mean([(((xt @ w2 + b2).numpy() > 0) == (np.asarray(y_tr).ravel() > 0.5)).mean() for w2, b2 in heads]))}


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    G = ctx["G"]
    w, b = ctx["hb"]
    x_tr, y_tr = ctx["tr_feats"].reshape(-1, ctx["tr_feats"].shape[-1]), ctx["tr_lab"].ravel()
    t0 = time.time()
    lap = fit_laplace(x_tr, y_tr, w.numpy(), float(b))
    heads = bootstrap_heads(x_tr, y_tr, N_BOOT)
    summary["part_a"]["laplace"] = {**lap["diag"], **head_check(lap, heads, x_tr, y_tr, w, b), "fit_s": time.time() - t0}
    d = summary["part_a"]["laplace"]
    print(f"part A Laplace: lambda {d['lambda']:.3g} (grid edge {d['lambda_grid_edge']}), effective params {d['effective_params']:.1f}/{lap['P']}, "
          f"Hessian rank {d['hessian_rank_1e-8']}, train acc {d['train_acc']:.3f}, train mean var {d['train_mean_logit_variance']:.3g}, "
          f"max |logit diff| {d['max_abs_logit_diff_vs_head']:.2e}, {d['fit_s']:.1f}s", flush=True)
    per, rho_per, diag = {}, {}, {}
    for name in ctx["names"]:
        t0 = time.time()
        try:
            unit = hb.scene_unit(ctx, model, name, args, summary)
            if unit is None:
                continue
            sigs = hb.base_signals_a(unit, ctx)
            new, sens = laplace_signals(lap, heads, unit["feats0"].reshape(-1, unit["feats0"].shape[-1]), (G, G))
            sigs.update(new)
            sigs[COMBO] = hb.combination(sigs, PRIMARY)
            diag[name] = sens
            val = hb.score_scene(unit, sigs, rows, per, rho_per, NEW_NAMES)
            for k in NEW_NAMES:
                cache[f"{name}_{k.split(' ')[0]}_{k.split(' ')[1]}"] = sigs[k].astype(np.float32)
            cache[f"{name}_err"] = unit["err"].astype(np.float32)
            print(f"{name}: {unit['n_err']} errors, {time.time() - t0:.1f}s, conf {val[CONF]:.4f} tile {val[TILE]:.4f} "
                  f"var {val[VAR]:.4f} mod {val[MOD]:.4f} MI {val[MI]:.4f} boot {val[BOOT]:.4f} U+ {val[COMBO]:.4f} "
                  f"| mean var {sens['mean_logit_variance']:.3g}, rho(var, norm) {sens['spearman_var_vs_norm']:+.2f}", flush=True)
        except Exception as ex:  # noqa: BLE001 - record and continue; the summary must always be written
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    summary["part_a"]["diagnostics"] = diag
    hb.finish_part_a(summary, per, rho_per, NEW_NAMES, PRIMARY, COMBO)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp30")
    if ctx is None:
        return
    N, G = ctx["N"], ctx["G"]
    w, b = ctx["hb"]
    x_tr, y_tr = ctx["tr_feats"], ctx["tr_y"]
    t0 = time.time()
    lap = fit_laplace(x_tr, y_tr, w.numpy(), float(b))
    heads = bootstrap_heads(x_tr, y_tr, N_BOOT)
    summary["part_b"]["laplace"] = {**lap["diag"], **head_check(lap, heads, x_tr, y_tr, w, b), "fit_s": time.time() - t0}
    d = summary["part_b"]["laplace"]
    print(f"part B Laplace: lambda {d['lambda']:.3g} (grid edge {d['lambda_grid_edge']}), effective params {d['effective_params']:.1f}/{lap['P']}, "
          f"Hessian rank {d['hessian_rank_1e-8']}, train acc {d['train_acc']:.3f}, train mean var {d['train_mean_logit_variance']:.3g}, "
          f"max |logit diff| {d['max_abs_logit_diff_vs_head']:.2e}, {d['fit_s']:.1f}s", flush=True)
    sigs = hb.base_signals_b(ctx)
    x_ev = ctx["ev_feats"].reshape(-1, ctx["ev_feats"].shape[-1])
    new, sens = laplace_signals(lap, heads, x_ev, (N, G, G))
    sigs.update(new)
    summary["part_b"]["diagnostics"] = sens
    for k in NEW_NAMES:
        cache[f"bolivia_{k.split(' ')[0]}_{k.split(' ')[1]}"] = sigs[k].astype(np.float32)
    cache["bolivia_err"] = ctx["err"].astype(np.float32)
    cache["bolivia_ok"] = ctx["ok"]
    hb.finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"lambda_grid": [float(LAMBDA_GRID[0]), float(LAMBDA_GRID[-1]), len(LAMBDA_GRID)],
              "lambda_selection": "post-hoc Laplace marginal likelihood on the grid, trained weights fixed",
              "hessian": "weighted BCE Hessian at the trained head, positive weight n_neg/n_pos as in train_logistic_head, bias included",
              "predictive_integration": f"trapezoid, {INT_NODES} logit-grid points within [mu - 8 s, mu + 8 s] clipped to |l| < {INT_HALF_WINDOW}, tail mass closed form",
              "n_bootstrap_heads": N_BOOT,
              "primary_score": PRIMARY, "combination": COMBO,
              "prereg": "U+ = mean of within-unit midrank percentiles of confidence and of the Laplace logit variance; "
                        "per-river mean gain over confidence, one-sided exact sign test over the 8 rivers"}
    hb.run("exp30", "exp30 last-layer Laplace on the probe head", config, part_a, part_b, args, "exp30_laplace_head")


if __name__ == "__main__":
    main()
