"""Exp 16: the boundary signal on the AWF point-label task.

Background. exp04 found the model's own confidence beats every other signal
on AWF, and the docs explained this as "point labels carry no boundary
context". That explanation had never been tested. This experiment applies
the Base head densely to all 8x8 patches of each validation window's 32x32
crop, computes the exp14 boundary indicator at the labelled patch (fraction
of its 8 neighbours whose argmax differs from its own), and asks two
separate questions with explicit tests:

  (a) Are labelled patches interior? The labelled patch's score is compared
      with the scores of the other 63 patches of the same prediction map
      (within-window quantile; paired sign test against a seeded random
      non-label patch; pooled interior reference). Note the score is
      derived from the head's own prediction map, not from ground-truth
      land-cover boundaries.
  (b) Does the boundary score carry error information beyond confidence?
      Error rate per boundary level, Fisher exact test on error x (score>0),
      Mann-Whitney and permutation tests (descriptive of the marginal
      association), then the conditional test: logistic regression of error
      on the standardized logit margin with and without the boundary score
      (likelihood-ratio test), the margin's AURC inside score==0 and
      score>0, and a granularity control (margin re-quantized to the
      boundary score's own tie-group sizes).

Ranking comparison. Tie-aware AURC of the negative logit margin (confidence),
the boundary score, and the per-window tile-phase of exp04, against the
exp04 errors. Uncertainty: a cluster bootstrap that resamples annotation
tasks (val windows cluster by task: 30 tasks, up to 40 windows each, sharing
annotator, landscape and season), reported alongside the i.i.d. window
bootstrap for comparison; the head is held fixed across replicates, so the
intervals condition on the trained head. Caveat carried from exp04: the AWF
split is by point, not by task; every val task also contributes training
windows.
"""
import csv
import json
import math
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.awf import list_windows, load_window_full, crop_stack, stacks_to_sample
from oe_inferencex.evidence import train_softmax_head, aurc_expected

PATCH = 4
N_CLASSES = 9
BATCH = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DENSE_CACHE = "exp/out/exp16_dense_val.npz"
B = 2000


def embed_dense(windows):
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval()
    grids, locs = [], []
    for i in range(0, len(windows), BATCH):
        chunk = windows[i:i + BATCH]
        stacks = []
        for wdir, _, r, c, _ in chunk:
            cr, (pr, pc) = crop_stack(load_window_full(wdir), r, c, 0)
            stacks.append(cr)
            locs.append((min(pr // PATCH, 7), min(pc // PATCH, 7)))
        sample = stacks_to_sample(stacks, DEV)
        with torch.no_grad():
            out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
        grids.append(out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4]).cpu().numpy())
        if (i // BATCH) % 10 == 9:
            print(f"  {i + len(chunk)}/{len(windows)}", flush=True)
    return np.concatenate(grids), np.array(locs)


def boundary_map(hard):
    """8-neighbour argmax-disagreement fraction for every patch of an (n,8,8) map."""
    pad = np.pad(hard, ((0, 0), (1, 1), (1, 1)), mode="edge")
    nb = np.zeros(hard.shape, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[:, 1 + di:9 + di, 1 + dj:9 + dj] != hard)
    return nb / 8.0


def sign_test_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fisher_exact_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]; two-sided p by summing hypergeometric
    probabilities no larger than the observed one."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    def pmf(x):
        return math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n, k)
    p_obs = pmf(a)
    lo, hi = max(0, k - n2), min(k, n1)
    return sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9))


def ranks(x):
    """Average ranks with ties."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    s = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def mann_whitney_z(x1, x0):
    x = np.concatenate([x1, x0]); r = ranks(x)
    n1, n0 = len(x1), len(x0); n = n1 + n0
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    mu = n1 * n0 / 2
    _, cnt = np.unique(x, return_counts=True)
    tie = (cnt ** 3 - cnt).sum()
    var = n1 * n0 / 12 * ((n + 1) - tie / (n * (n - 1)))
    return (u - mu) / math.sqrt(var)


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def logistic_ll(X, y, iters=100, ridge=1e-6):
    """Newton-fitted logistic regression; returns (log-likelihood, coefficients)."""
    Xb = np.c_[np.ones(len(y)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (y - p) - ridge * w
        H = -(Xb.T * (p * (1 - p))) @ Xb - ridge * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    p = np.clip(1 / (1 + np.exp(-Xb @ w)), 1e-12, 1 - 1e-12)
    return float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum()), w


def chi2_1_sf(x):
    return math.erfc(math.sqrt(max(x, 0) / 2))


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def main():
    torch.manual_seed(0)
    windows = list_windows()
    labels = np.array([w[4] for w in windows])
    is_val = np.array([w[1] == "val" for w in windows])
    z = np.load("exp/out/exp04_feats.npz")
    w, b = train_softmax_head(z["base_s0"][~is_val], labels[~is_val], N_CLASSES)

    val_windows = [wd for wd, m in zip(windows, is_val) if m]
    if os.path.exists(DENSE_CACHE):
        d = np.load(DENSE_CACHE)
        grids, locs = d["grids"], d["locs"]
        print("loaded cached dense grids")
    else:
        print(f"embedding {len(val_windows)} validation windows densely on {DEV}")
        grids, locs = embed_dense(val_windows)
        np.savez(DENSE_CACHE, grids=grids, locs=locs)

    y = labels[is_val]
    n = len(val_windows)
    logits = (torch.tensor(grids.reshape(n * 64, -1)).float() @ w + b).numpy().reshape(n, 8, 8, N_CLASSES)
    probs = np.exp(logits - logits.max(-1, keepdims=True)); probs /= probs.sum(-1, keepdims=True)
    hard = probs.argmax(-1)
    bmap = boundary_map(hard)
    ii = np.arange(n); pi, pj = locs[:, 0], locs[:, 1]
    lg = np.sort(logits[ii, pi, pj], axis=-1)
    base = -(lg[:, -1] - lg[:, -2])            # negative top-1 margin: higher = less confident
    bnd = bmap[ii, pi, pj]
    err = (hard[ii, pi, pj] != y).astype(float)
    p04 = torch.softmax(torch.tensor(z["base_s0"][is_val]).float() @ w + b, dim=-1).numpy()
    err04 = (p04.argmax(1) != y).astype(float)
    print(f"errors: dense-at-label {int(err.sum())}, exp04 label-patch {int(err04.sum())}, agreement {(err == err04).mean():.3f}")
    sp = np.stack([torch.softmax(torch.tensor(z[f"base_s{s}"][is_val]).float() @ w + b, dim=-1).numpy() for s in range(4)])
    tile = sp.max(2).std(0) + 0.5 * np.abs(sp - sp.mean(0)).sum(2).mean(0)

    S = {"n": n, "n_err": int(err.sum())}

    # ---- ranking comparison ----
    sigs = {"baseline": base, "boundary": bnd, "tile-phase": tile}
    S["aurc"] = {k: aurc_expected(v, err) for k, v in sigs.items()}
    for k, v in S["aurc"].items():
        print(f"AURC {k}: {v:.4f}")
    task = np.array([wd[0].split("/")[-1].split("_point_")[0] for wd in val_windows])
    tasks = np.unique(task)
    members = {t: np.flatnonzero(task == t) for t in tasks}
    S["n_tasks"] = int(len(tasks))
    rng = np.random.default_rng(0)
    S["ci"] = {}
    for scheme in ("cluster", "iid"):
        diffs = {k: [] for k in ("boundary", "tile-phase")}
        for _ in range(B):
            if scheme == "cluster":
                idx = np.concatenate([members[t] for t in rng.choice(tasks, len(tasks))])
            else:
                idx = rng.integers(0, n, n)
            if err[idx].sum() == 0:
                continue
            b0 = aurc_expected(base[idx], err[idx])
            for k in diffs:
                diffs[k].append(aurc_expected(sigs[k][idx], err[idx]) - b0)
        S["ci"][scheme] = {k: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)),
                                    hi=float(np.percentile(v, 97.5)), p_better=float(np.mean(np.array(v) < 0)))
                           for k, v in diffs.items()}
        for k, c in S["ci"][scheme].items():
            print(f"  [{scheme}] {k} minus baseline: {c['mean']:+.4f}  95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  P(better)={c['p_better']:.3f}")

    # ---- (a) are labelled patches interior? ----
    mask = np.ones((n, 8, 8), bool); mask[ii, pi, pj] = False
    other = bmap[mask].reshape(n, 63)
    quant = ((other < bnd[:, None]).sum(1) + 0.5 * (other == bnd[:, None]).sum(1)) / 63
    rnd = other[ii, rng.integers(0, 63, n)]
    wl = (int((bnd > rnd).sum()), int((bnd < rnd).sum()))
    S["interior"] = dict(label_zero_frac=float((bnd == 0).mean()), label_mean=float(bnd.mean()),
                         other_zero_frac=float((other == 0).mean()), other_mean=float(other.mean()),
                         label_quantile_mean=float(quant.mean()), label_gt_random=wl[0], label_lt_random=wl[1],
                         sign_p=sign_test_p(*wl), share_of_map_same_class=float(np.mean([(hard[i] == hard[i, pi[i], pj[i]]).mean() for i in range(n)])))
    a = S["interior"]
    print(f"\n(a) interior test (score is prediction-derived): labelled patch zero-fraction {a['label_zero_frac']:.3f}, mean {a['label_mean']:.3f}; "
          f"other patches zero-fraction {a['other_zero_frac']:.3f}, mean {a['other_mean']:.3f}; labelled patch within-window quantile mean {a['label_quantile_mean']:.3f}; "
          f"label > random patch on {wl[0]}, < on {wl[1]} (sign p={a['sign_p']:.3f}); share of map sharing the label patch's class {a['share_of_map_same_class']:.3f}")

    # ---- (b) error information beyond confidence ----
    levels = np.unique(bnd)
    S["per_level"] = {f"{lv:.3f}": dict(err=int(err[bnd == lv].sum()), n=int((bnd == lv).sum())) for lv in levels}
    print("(b) error count / n per boundary level: " + ", ".join(f"{lv}: {v['err']}/{v['n']}" for lv, v in S["per_level"].items()))
    a_, b_ = int(((bnd > 0) & (err == 1)).sum()), int(((bnd == 0) & (err == 1)).sum())
    c_, d_ = int(((bnd > 0) & (err == 0)).sum()), int(((bnd == 0) & (err == 0)).sum())
    S["fisher"] = dict(table=[a_, b_, c_, d_], p=fisher_exact_two_sided(a_, b_, c_, d_))
    S["mwu_z"] = mann_whitney_z(bnd[err == 1], bnd[err == 0])
    obs = bnd[err == 1].mean() - bnd[err == 0].mean()
    perm = np.array([(lambda e: bnd[e == 1].mean() - bnd[e == 0].mean())(rng.permutation(err)) for _ in range(20000)])
    S["perm_p"] = float((perm >= obs).mean())
    S["spearman_bnd_margin"] = spearman(bnd, base)
    ll1, w1 = logistic_ll(zscore(base)[:, None], err)
    ll2, w2 = logistic_ll(np.c_[zscore(base), zscore(bnd)], err)
    S["lrt"] = dict(chi2=2 * (ll2 - ll1), p=chi2_1_sf(2 * (ll2 - ll1)), coef_margin=float(w2[1]), coef_bnd=float(w2[2]))
    S["margin_aurc_within"] = {"bnd==0": aurc_expected(base[bnd == 0], err[bnd == 0]) if err[bnd == 0].sum() else None,
                               "bnd>0": aurc_expected(base[bnd > 0], err[bnd > 0])}
    order = np.argsort(base, kind="stable"); _, sizes = np.unique(np.sort(bnd), return_counts=True)
    q = np.empty(n); start = 0
    for gi, sz in enumerate(sizes):
        q[order[start:start + sz]] = gi; start += sz
    S["granularity_control_aurc"] = aurc_expected(q, err)
    print(f"    Fisher exact err x (score>0) table {S['fisher']['table']} p={S['fisher']['p']:.2e}; Mann-Whitney z={S['mwu_z']:.2f}; permutation p={S['perm_p']:.2e}")
    print(f"    Spearman(score, neg. margin)={S['spearman_bnd_margin']:.3f}; logistic err ~ margin (+score): LRT chi2={S['lrt']['chi2']:.2f} p={S['lrt']['p']:.3f}, "
          f"standardized coefs margin {S['lrt']['coef_margin']:.2f}, score {S['lrt']['coef_bnd']:.2f}")
    print(f"    margin AURC within score==0: {S['margin_aurc_within']['bnd==0']}; within score>0: {S['margin_aurc_within']['bnd>0']:.4f}; "
          f"margin re-quantized to the score's tie-group sizes: AURC {S['granularity_control_aurc']:.4f}")

    json.dump(S, open("exp/out/exp16_summary.json", "w"), indent=1)
    with open("exp/out/exp16_awf_boundary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["window", "task", "label", "pred", "error", "baseline_neg_margin", "boundary", "tile_phase", "top1_prob", "label_quantile_in_window", "other_patches_mean_boundary"])
        for i in range(n):
            wr.writerow([val_windows[i][0].split("/")[-1], task[i], int(y[i]), int(hard[i, pi[i], pj[i]]), int(err[i]),
                         f"{base[i]:.5f}", f"{bnd[i]:.3f}", f"{tile[i]:.5f}", f"{probs[i, pi[i], pj[i]].max():.4f}", f"{quant[i]:.3f}", f"{other[i].mean():.3f}"])
    print("wrote exp/out/exp16_awf_boundary.csv and exp16_summary.json")


if __name__ == "__main__":
    main()
