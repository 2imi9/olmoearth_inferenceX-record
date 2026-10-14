"""exp105: confidence strata against map-class strata for a map's overall error rate.

Preregistration: docs/plan/class_strata_baseline.md (the page is the specification). Numpy only.

On exp78's per-unit exports of the embedding suite (24 tasks, OlmoEarth Base probe maps, every unit labelled), five
designs draw a budget of windows R times each and give a 95% interval for the map's overall window error rate:

  A  a simple random sample, the package's exact hypergeometric interval
  B  five margin quintiles, Neyman allocation from the model's confidence floored at Q_FLOOR (the shipped
     `confidence` design, as `sample_for_estimation` allocates it)
  C  the map's classes as strata, proportional allocation (the practitioner's design; Olofsson et al. 2014)
  D  the map's classes as strata, Neyman allocation from the model's confidence (B's allocation, C's strata)
  E  the map's classes as strata, equal allocation by water-filling (report-only)

B to E use the package's allocation rules, its stratified estimator and `stratified_interval_wilson`. The grade is
refused until the page is frozen; --smoke runs on synthetic units anywhere.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, ROOT)

from oe_inferencex import estimate as est  # noqa: E402

PLAN = os.path.join(ROOT, "docs", "plan", "class_strata_baseline.md")
OUT = os.path.join(EXP_DIR, "out")
UNITS = os.path.join(OUT, "exp78_units")
R_DRAWS, BUDGETS, HEADLINE, SEED = 2000, (100, 300, 1000), 300, 105
COVER, P1_WINS, P1_RATIO, P2_WINS, N_GRADABLE = 0.93, 14, 0.90, 12, 18      # the page's thresholds
REVERSE_MAJORITY = 10                                                         # the page's branches
ARMS = ("A", "B", "C", "D", "E")
GRADED = {"P1": ("B", "C"), "P2": ("B", "D")}


def prereg_status(path=PLAN):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return "frozen" if "**Status: frozen" in text else "draft"


def tasks(units_dir=UNITS):
    with open(os.path.join(units_dir, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    return sorted(t for t, v in man["tasks"].items() if "failed" not in v)


def load(task, units_dir=UNITS):
    """One task's units as exp78 graded them: margin, top-1 probability, error and the map's class, per valid unit."""
    d = np.load(os.path.join(units_dir, f"{task}.npz"), allow_pickle=False)
    u = {"margin": d["margin"].astype(np.float64), "p1": d["p1"].astype(np.float64),
         "err": d["err"].astype(np.float64), "dec": d["dec"].astype(np.int64)}
    n = {k: v.size for k, v in u.items()}
    if len(set(n.values())) != 1:
        raise ValueError(f"{task}: the per-unit arrays differ in length {n}")
    return u


def spread_from_confidence(strata, p1, H):
    """Neyman's spread per stratum from the model's own confidence: sqrt(q(1 - q)), q = max(Q_FLOOR, the mean of
    1 - p1), as `sample_for_estimation` computes it (estimate.py; the floor dates from the audit of 2026-09-22: a
    saturated stratum otherwise gets two labels, and exp78's and exp85's recorded B had no floor)."""
    q = np.array([max(est.Q_FLOOR, (1 - p1[strata == h]).mean()) if (strata == h).any() else 0.0 for h in range(H)])
    return np.sqrt(np.clip(q * (1 - q), 1e-9, None))


def designs(u, B):
    """The strata, their sizes and the allocation of each stratified arm at budget B (label-free)."""
    margin, p1, dec = u["margin"], u["p1"], u["dec"]
    s_conf = est.confidence_strata(margin)
    H = int(s_conf.max()) + 1
    z_conf = np.bincount(s_conf, minlength=H)
    _, s_cls = np.unique(dec, return_inverse=True)
    K = int(s_cls.max()) + 1
    z_cls = np.bincount(s_cls, minlength=K)
    return {
        "B": (s_conf, z_conf, est.neyman_allocation(z_conf, spread_from_confidence(s_conf, p1, H), B)),
        "C": (s_cls, z_cls, est.neyman_allocation(z_cls, np.ones(K), B)),
        "D": (s_cls, z_cls, est.neyman_allocation(z_cls, spread_from_confidence(s_cls, p1, K), B)),
        "E": (s_cls, z_cls, est.equal_allocation(z_cls, B)),
    }


def gradable(N, B):
    return N >= 2 * B


def run_task(u, B, R, rng, per_class=True):
    """Every arm at one budget: coverage, mean half-width, bias ratio, the variance of the estimates, starved
    strata; for C and E each class's user's accuracy interval from that class's labels (report-only)."""
    err = u["err"]
    N = err.size
    theta = float(err.mean())
    plan = designs(u, B)
    pools = {a: [np.flatnonzero(s == h) for h in range(z.size)] for a, (s, z, _) in plan.items()}
    hyper = {}
    rows = {}
    for arm in ARMS:
        ests, cov, half, starved = np.empty(R), 0, 0.0, 0
        cls_acc = None
        if per_class and arm in ("C", "E"):
            s, z, alloc = plan[arm]
            # the estimate's own arithmetic, so a class labelled in full matches its census exactly
            cls_true = np.array([(z[h] - err[s == h].sum()) / z[h] for h in range(z.size)])
            cls_acc = {"cover": np.zeros(z.size), "half": np.zeros(z.size), "n": np.asarray(alloc).tolist(),
                       "N": z.tolist()}
        for r in range(R):
            if arm == "A":
                i = rng.choice(N, B, replace=False)
                k = int(err[i].sum())
                if k not in hyper:
                    hyper[k] = est.hypergeom_interval(k, B, N)
                e_, (lo, hi) = k / B, hyper[k]
            else:
                s, z, alloc = plan[arm]
                i = np.concatenate([rng.choice(pools[arm][h], int(n_h), replace=False)
                                    for h, n_h in enumerate(alloc) if n_h > 0])
                e_, lo, hi, st, _ = est.stratified_interval_wilson(err, s, i, z, N)
                # a one-unit class labelled in full is counted as starved by the estimator; it is a census
                starved += st > int((z == 1).sum())
                if cls_acc is not None:
                    for h, n_h in enumerate(alloc):
                        if n_h <= 0:
                            continue
                        m = s[i] == h
                        k_h = int(n_h - err[i][m].sum())
                        a_lo, a_hi = est.wilson_interval(k_h, int(n_h), int(z[h]))
                        cls_acc["cover"][h] += a_lo <= cls_true[h] <= a_hi
                        cls_acc["half"][h] += (a_hi - a_lo) / 2
            ests[r] = e_
            cov += lo <= theta <= hi
            half += (hi - lo) / 2
        rows[arm] = {"coverage": cov / R, "half_width": half / R, "mean_estimate": float(ests.mean()),
                     "bias_ratio": float(ests.mean() / theta) if theta > 0 else None,
                     "estimate_variance": float(ests.var(ddof=1)), "starved_rate": starved / R}
        if arm != "A":
            s, z, alloc = plan[arm]
            rows[arm]["n_strata"] = int(z.size)
            rows[arm]["allocation"] = np.asarray(alloc).tolist()
        if cls_acc is not None:
            rows[arm]["per_class_users_accuracy"] = {
                "coverage": (cls_acc["cover"] / R).tolist(), "half_width": (cls_acc["half"] / R).tolist(),
                "n": cls_acc["n"], "N": cls_acc["N"], "reading": "report-only: each map class's labels, Wilson"}
    return {"n_units": N, "error_rate": theta, "arms": rows}


def predicted_ratio_binary(u, B):
    """Report-only, binary maps: Var(C) / Var(A) predicted from the population's class sizes and class error rates
    (Cochran's stratified variance, with finite-population corrections, at C's integer allocation; on a binary map
    the same per-stratum variance as Skakun's area case, but the simple random denominator here is the error rate's
    theta(1 - theta), not the class share's P(1 - P))."""
    err, dec = u["err"], u["dec"]
    N = err.size
    _, s = np.unique(dec, return_inverse=True)
    z = np.bincount(s)
    if z.size != 2:
        return None
    alloc = est.neyman_allocation(z, np.ones(2), B)
    v_c = 0.0
    for h in range(2):
        if alloc[h] >= z[h]:
            continue                                          # a class labelled in full adds no variance
        p = err[s == h].mean()
        S2 = z[h] / (z[h] - 1) * p * (1 - p)
        v_c += (z[h] / N) ** 2 * (1 - alloc[h] / z[h]) * S2 / alloc[h]
    th = err.mean()
    v_a = (1 - B / N) * (N / (N - 1) * th * (1 - th)) / B
    return float(v_c / v_a) if v_a > 0 else None


def win(a, b):
    """Arm a wins a task against arm b: a covers at least COVER and either b does not or a is no wider."""
    return a["coverage"] >= COVER and (b["coverage"] < COVER or a["half_width"] <= b["half_width"])


def ratio(a, b):
    """B's half-width over the other's, coverage first as the win rule: infinite where a covers below COVER, zero
    where only b does."""
    if a["coverage"] < COVER:
        return float("inf")
    if b["coverage"] < COVER:
        return 0.0
    return a["half_width"] / b["half_width"]


def verdicts(rows, budget=HEADLINE):
    graded = sorted(t for t, r in rows.items() if str(budget) in r["budgets"] and r["budgets"][str(budget)]["gradable"])
    out = {"budget": budget, "gradable_tasks": graded, "n_gradable": len(graded),
           "n_gradable_expected": N_GRADABLE}
    arm = lambda t, k: rows[t]["budgets"][str(budget)]["arms"][k]      # noqa: E731
    for p, (a, b) in GRADED.items():
        wins = [t for t in graded if bool(win(arm(t, a), arm(t, b)))]
        ratios = [ratio(arm(t, a), arm(t, b)) for t in graded]
        out[p] = {"arms": [a, b], "wins": len(wins), "of": len(graded), "won": wins,
                  "median_half_width_ratio": float(np.median(ratios)) if ratios else None}
    out["P1"]["bar"] = f"wins >= {P1_WINS} and median ratio <= {P1_RATIO} (coverage first)"
    out["P1"]["holds"] = bool(out["P1"]["wins"] >= P1_WINS and (out["P1"]["median_half_width_ratio"] or 9) <= P1_RATIO)
    out["P2"]["bar"] = f"wins >= {P2_WINS}"
    out["P2"]["holds"] = bool(out["P2"]["wins"] >= P2_WINS)
    out["reverse"] = {f"{a}_over_{b}": int(sum(bool(win(arm(t, a), arm(t, b))) for t in graded))
                      for a, b in (("C", "B"), ("D", "B"), ("D", "C"), ("C", "D"))}
    out["reverse"]["reading"] = (f"the page's branches: a design is recommended over B only if it wins against B on "
                                 f"at least {REVERSE_MAJORITY} of the gradable tasks")
    if len(graded) != N_GRADABLE:
        out["note"] = f"{len(graded)} tasks gradable at {budget}, the page expects {N_GRADABLE}: verdicts not applied"
        out["P1"]["holds"] = out["P2"]["holds"] = None
    return out


def run(task_units, R=R_DRAWS, budgets=BUDGETS, seed=SEED, log=print):
    rows = {}
    for t, u in task_units.items():
        rng = np.random.default_rng(seed)
        rec = {"n_units": int(u["err"].size), "error_rate": float(u["err"].mean()),
               "map_classes": int(np.unique(u["dec"]).size), "budgets": {}}
        for B in budgets:
            if B >= u["err"].size:
                rec["budgets"][str(B)] = {"gradable": False, "skipped": "budget at or above the units"}
                continue
            t0 = time.time()
            r = run_task(u, B, R, rng)
            r["gradable"] = gradable(u["err"].size, B)
            r["predicted_var_ratio_C_over_A_binary"] = predicted_ratio_binary(u, B)
            va = r["arms"]["A"]["estimate_variance"]
            r["measured_var_ratio_C_over_A"] = r["arms"]["C"]["estimate_variance"] / va if va > 0 else None
            r["seconds"] = round(time.time() - t0, 1)
            rec["budgets"][str(B)] = r
            a = r["arms"]
            log(f"[exp105] {t:58s} B={B:5d} " + " ".join(
                f"{k} {a[k]['coverage']:.3f}/{a[k]['half_width']:.4f}" for k in ARMS) + f" {r['seconds']} s")
        rows[t] = rec
    return rows


def synthetic_units(seed=0, N=6000, K=4):
    """Classes with different error rates; the margin tracks the error, more in some classes than others."""
    rng = np.random.default_rng(seed)
    dec = rng.choice(K, N, p=np.linspace(1, 2, K) / np.linspace(1, 2, K).sum())
    rate = np.linspace(0.05, 0.35, K)[dec]
    err = (rng.random(N) < rate).astype(np.float64)
    margin = np.clip(rng.normal(0.6 - 0.35 * err, 0.2), 0, 1)
    p1 = np.clip(0.5 + margin / 2, 0, 1)
    return {"margin": margin, "p1": p1, "err": err, "dec": dec.astype(np.int64)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="synthetic units, few draws, anywhere")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--draws", type=int, default=R_DRAWS)
    a = ap.parse_args(argv)
    if a.smoke:
        units = {f"synthetic_{k}": synthetic_units(seed=k) for k in range(2)}
        rows = run(units, R=min(a.draws, 60), budgets=(100, 300))
        rec = {"experiment": "exp105 smoke (synthetic units)", "tasks": rows, "verdicts": verdicts(rows)}
        os.makedirs(a.out_dir, exist_ok=True)
        with open(os.path.join(a.out_dir, "exp105_summary_smoke.json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print(json.dumps(rec["verdicts"]["P1"]))
        return rec
    status = prereg_status()
    if status != "frozen":
        raise SystemExit(f"{os.path.relpath(PLAN, ROOT)} is {status}: no arm is graded on the suite before it is frozen")
    import exp89_finetuned_checkpoints as e89
    commit, dirty = e89.git_state()
    units = {t: load(t) for t in tasks()}
    rows = run(units, R=a.draws)
    rec = {"experiment": "exp105: confidence strata against map-class strata for a map's overall error rate",
           "prereg_page": os.path.relpath(PLAN, ROOT), "prereg_status": status, "commit": commit, "dirty": dirty,
           "config": {"R": a.draws, "budgets": list(BUDGETS), "headline": HEADLINE, "seed": SEED, "cover": COVER,
                      "arms": {"A": "simple random, exact hypergeometric",
                               "B": "margin quintiles, Neyman from confidence, Wilson at n_eff",
                               "C": "map classes, proportional, Wilson at n_eff",
                               "D": "map classes, Neyman from confidence, Wilson at n_eff",
                               "E": "map classes, equal (water-filling), Wilson at n_eff; report-only"}},
           "tasks": rows, "verdicts": verdicts(rows)}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "exp105_summary.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({p: rec["verdicts"][p] for p in ("P1", "P2")}))
    return rec


if __name__ == "__main__":
    main()
