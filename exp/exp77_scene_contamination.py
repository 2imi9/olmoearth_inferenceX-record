"""exp77: is a confident error a window that agrees with its scene instead of its own evidence?

Preregistration is docs/plan/scene_contamination.md; this file is the run.

Why. The record's open gap is the errors the model makes confidently: no label-free reading tried has seen them
(exp70's headroom, exp73's alternatives, exp76's forms). LaSt-ViT (arXiv 2602.22394) names a mechanism that would
produce exactly such errors in a ViT: under coarse supervision and global attention, scene-level semantics diffuse
into tokens that do not carry them, so a token can be confidently labelled by what its neighbourhood is rather than
by what it shows. Their fix is a pre-training change and their probe needs a CLS token, which OlmoEarth does not
have; but the mechanism is testable on frozen tokens, because every token of a tile can be compared with that
tile's own mean.

What this run does, on the seven segmentation tasks of Ai2's published suite (the ones whose embeddings are token
grids, so the tokens are ours to read):

  scene typicality  s_i = cos(x_i, m),  m = mean token of the tile          a label-free reading per token
  decontaminated    x_i - lam * (x_i . m_hat) m_hat                         lam chosen on a train holdout only

Arm A is exp70's recipe unchanged (Ai2's probe on the raw tokens, two seeds, the second only to measure the floor).
Arm B is the same recipe on decontaminated tokens. Nothing here trains an encoder: the tokens are frozen and the
probe is the linear one the suite already uses.

    python exp/exp77_scene_contamination.py --smoke      # synthetic grids, no cache, no network
    python exp/exp77_scene_contamination.py              # the seven tasks, CPU

Outputs exp/out/exp77_summary.json and exp77_tasks.csv.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics  # noqa: E402
from oe_inferencex.signals import midrank_pct  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
LAMBDAS = (0.5, 1.0)          # how much of the scene direction to remove; 0 is arm A
HOLDOUT = 0.2                 # share of TRAIN tiles that choose lambda; the test split never does
CONFIDENT = 0.5               # "confident" means the more confident half of the scored windows
MIN_GAIN = 0.005              # 0.5 points of window accuracy, and it must also clear the seed floor


def scene_direction(emb):
    """Unit mean token per tile, (N, D), and the cosine of every token to it, (N, h, w)."""
    import torch
    x = emb.to(torch.float32)
    m = x.mean(dim=(1, 2))
    m_hat = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos = (x * m_hat[:, None, None, :]).sum(-1) / x.norm(dim=-1).clamp_min(1e-12)
    return m_hat, cos


def decontaminate(emb, m_hat, lam):
    """Remove lam of each token's component along its own tile's mean direction. lam=0 returns the input."""
    import torch
    x = emb.to(torch.float32)
    proj = (x * m_hat[:, None, None, :]).sum(-1, keepdim=True) * m_hat[:, None, None, :]
    return (x - lam * proj).to(emb.dtype)


def window_typicality(cos, pp, win=4):
    """Token cosines (N, h, w) averaged onto the 4-px windows everything here is graded on."""
    import torch
    n, h, w = cos.shape
    per_px = cos[:, :, None, :, None].expand(n, h, pp, w, pp).reshape(n, h * pp, w * pp)
    hw, ww = (h * pp) // win, (w * pp) // win
    return per_px[:, :hw * win, :ww * win].reshape(n, hw, win, ww, win).mean(dim=(2, 4)).to(torch.float32).numpy()


def fit_and_score(emb_tr, lab_tr, emb_te, lab_te, pp, C, lr, seed):
    """One arm: train the suite's probe, return the per-window error, margin, decision and validity on the test side."""
    import exp54_multiclass_embeddings as e54
    probe = e54.train_probe(emb_tr, lab_tr, pp, C, lr, seed=seed)
    q = e54.window_quantities(probe, emb_te, lab_te, pp, C)
    ok = q["ok"].ravel()
    return {"ok": ok, "err": q["err"].ravel()[ok].astype(np.float64), "dec": q["dec"].ravel()[ok],
            "margin": q["margin"].ravel()[ok].astype(np.float64), "accuracy": float(1.0 - q["err"].ravel()[ok].mean()),
            "pixel_accuracy": q["pixel_accuracy"], "pixel_miou": q["pixel_miou"]}


def contamination_gap(typ, err, margin):
    """P1's quantity: inside the confident half, mean scene typicality of error windows minus that of correct ones."""
    if err.sum() < 5:
        return None
    keep = margin >= np.quantile(margin, 1.0 - CONFIDENT)      # the more confident half
    e, t = err[keep] > 0, typ[keep]
    if e.sum() < 5 or (~e).sum() < 5:
        return None
    return {"gap": float(t[e].mean() - t[~e].mean()), "n_confident": int(keep.sum()),
            "n_confident_errors": int(e.sum()), "share_of_errors_confident": float(e.sum() / err.sum())}


def excess_aurc(sig, err):
    a = metrics.aurc_expected(sig, err)
    return float(a - metrics.aurc_expected(err, err))


def run_task(task, cache, smoke=False):
    import torch
    import exp54_multiclass_embeddings as e54
    import exp70_task_suite as e70
    t0 = time.time()
    if smoke:
        g = torch.Generator().manual_seed(0)
        N, h, w, D, C, pp = 24, 6, 6, 16, 3, 2
        emb_tr = torch.randn(N, h, w, D, generator=g)
        emb_te = torch.randn(N, h, w, D, generator=g)
        lab_tr = torch.randint(0, C, (N, h * pp, w * pp), generator=g)
        lab_te = torch.randint(0, C, (N, h * pp, w * pp), generator=g)
        lr = 0.1
    else:
        emb_tr, lab_tr = e54.load_theirs(e70.MODEL, task, "train", cache)
        emb_te, lab_te = e54.load_theirs(e70.MODEL, task, "test", cache)
        C = int(max(int(lab_tr[lab_tr >= 0].max()), int(lab_te[lab_te >= 0].max())) + 1)
        pp = int(round(lab_tr.shape[-1] / emb_tr.shape[1]))
        lr = e54.TASK_LR.get(task, 0.1)
    if smoke:
        C, pp = 3, 2

    m_tr, cos_tr = scene_direction(emb_tr)
    m_te, cos_te = scene_direction(emb_te)
    typ = window_typicality(cos_te, pp)

    # Arm A: exp70's recipe unchanged, plus a second seed that measures the floor a gain has to clear.
    A = fit_and_score(emb_tr, lab_tr, emb_te, lab_te, pp, C, lr, seed=0)
    A1 = fit_and_score(emb_tr, lab_tr, emb_te, lab_te, pp, C, lr, seed=1)
    seed_floor = abs(A["accuracy"] - A1["accuracy"])

    # lambda is chosen on a holdout of the TRAIN tiles; the test split never sees the choice.
    n_tr = len(lab_tr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_tr)
    n_hold = max(2, int(round(HOLDOUT * n_tr)))
    hold, keep = perm[:n_hold], perm[n_hold:]
    chosen, holdout_scores = None, {}
    for lam in LAMBDAS:
        a = fit_and_score(decontaminate(emb_tr[keep], m_tr[keep], lam), lab_tr[keep],
                          decontaminate(emb_tr[hold], m_tr[hold], lam), lab_tr[hold], pp, C, lr, seed=0)
        holdout_scores[str(lam)] = a["accuracy"]
        if chosen is None or a["accuracy"] > holdout_scores[str(chosen)]:
            chosen = lam
    B = fit_and_score(decontaminate(emb_tr, m_tr, chosen), lab_tr, decontaminate(emb_te, m_te, chosen),
                      lab_te, pp, C, lr, seed=0)

    ok = A["ok"]
    typ_ok = typ.ravel()[ok]
    gap = contamination_gap(typ_ok, A["err"], A["margin"])
    same = bool(A["ok"].shape == B["ok"].shape and (A["ok"] == B["ok"]).all())
    readings = {"margin": excess_aurc(-A["margin"], A["err"]),
                "scene_typicality": excess_aurc(typ_ok, A["err"]),
                "scene_atypicality": excess_aurc(-typ_ok, A["err"])}
    if same:
        readings["decontamination_flip"] = excess_aurc((A["dec"] != B["dec"]).astype(np.float64), A["err"])
    rec = {"task": task, "n_classes": C, "patch_px": pp, "n_windows": int(ok.sum()),
           "accuracy_A": A["accuracy"], "accuracy_A_seed1": A1["accuracy"], "seed_floor": float(seed_floor),
           "accuracy_B": B["accuracy"], "gain": float(B["accuracy"] - A["accuracy"]), "lambda": float(chosen),
           "holdout_accuracy": holdout_scores, "pixel_accuracy_A": A["pixel_accuracy"], "pixel_accuracy_B": B["pixel_accuracy"],
           "pixel_miou_A": A["pixel_miou"], "pixel_miou_B": B["pixel_miou"],
           "error_rate_A": float(A["err"].mean()), "contamination": gap, "excess_aurc": readings,
           "same_valid_windows": same, "seconds": time.time() - t0}
    return rec


def verdicts(rows):
    """The four preregistered predictions, graded as written."""
    n = len(rows)
    p1 = [r for r in rows if r["contamination"] and r["contamination"]["gap"] > 0]
    gains = [r for r in rows if r["gain"] >= MIN_GAIN and r["gain"] > r["seed_floor"]]
    pairs = [(r["contamination"]["gap"], r["gain"]) for r in rows if r["contamination"]]
    rho = None
    if len(pairs) >= 3:
        # Midranks, not argsort-of-argsort: the latter breaks ties by position, so seven tasks with the same gap
        # and the same gain would rank 0..6 on both axes and report a perfect correlation on no evidence.
        a = midrank_pct([p[0] for p in pairs])
        b = midrank_pct([p[1] for p in pairs])
        sa, sb = a.std(), b.std()
        rho = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else None
    beats = [r for r in rows if any(v < r["excess_aurc"]["margin"] for k, v in r["excess_aurc"].items() if k != "margin")]
    return {
        "P1_confident_errors_are_more_scene_typical": {"holds": len(p1) >= 5, "n": len(p1), "of": n,
                                                       "threshold": "at least 5 of 7 tasks with a positive gap"},
        "P2_decontamination_improves_accuracy": {"holds": len(gains) >= 4, "n": len(gains), "of": n,
                                                 "threshold": f"at least 4 of 7 tasks gaining >= {MIN_GAIN} and more than that task's seed floor",
                                                 "tasks": [r["task"] for r in gains]},
        "P3_the_diagnosis_predicts_where_it_helps": {"holds": rho is not None and rho > 0, "spearman": rho,
                                                     "threshold": "positive rank correlation between the P1 gap and the P2 gain"},
        "P4_no_new_reading_beats_the_margin": {"holds": len(beats) <= 2, "n_tasks_beaten": len(beats), "of": n,
                                               "tasks": [r["task"] for r in beats],
                                               "threshold": "the margin keeps the lead on at least 5 of 7 tasks"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME"))
    ap.add_argument("--regrade", action="store_true",
                    help="re-evaluate the predictions over an existing summary with this file's grader and write it back; "
                         "the run of 2026-09-20 carried the pre-ba9f009 grader, whose rank correlation broke ties by position")
    args = ap.parse_args()
    if args.regrade:
        path = os.path.join(OUT, "exp77_summary_smoke.json" if args.smoke else "exp77_summary.json")
        with open(path) as f:
            summary = json.load(f)
        done = [r for r in summary["tasks"] if "failed" not in r]
        summary.setdefault("prereg_as_run", summary.get("prereg", {}))
        summary["prereg"] = verdicts(done)
        summary["regraded_by"] = "exp/exp77_scene_contamination.py verdicts(), tie-aware midranks"
        with open(path, "w") as f:
            json.dump(summary, f, indent=1)
        changed = {k: (summary["prereg_as_run"].get(k, {}).get("holds"), v["holds"])
                   for k, v in summary["prereg"].items() if summary["prereg_as_run"].get(k, {}).get("holds") != v["holds"]}
        print(f"regraded {len(done)} tasks in {path}")
        print("changed:", json.dumps(changed) if changed else "nothing; the grader's fix did not move a verdict")
        print(json.dumps(summary["prereg"], indent=1))
        return
    import exp70_task_suite as e70
    tasks = args.tasks or (["smoke_a", "smoke_b", "smoke_c"] if args.smoke else e70.TASKS_SEG)
    rows = []
    for t in tasks:
        print(f"[exp77] {t}", flush=True)
        try:
            rows.append(run_task(t, args.cache, smoke=args.smoke))
        except Exception as ex:  # noqa: BLE001
            print(f"  FAILED {t}: {ex!r}", flush=True)
            rows.append({"task": t, "failed": repr(ex)})
        print(f"  {json.dumps({k: v for k, v in rows[-1].items() if k in ('accuracy_A', 'accuracy_B', 'gain', 'seed_floor', 'lambda')})}", flush=True)
    done = [r for r in rows if "failed" not in r]
    summary = {"model": e70.MODEL, "tasks": rows, "n_tasks": len(done),
               "config": {"lambdas": list(LAMBDAS), "holdout": HOLDOUT, "confident_share": CONFIDENT, "min_gain": MIN_GAIN,
                          "probe": "exp54/exp51 linear probe, Ai2's segmentation recipe, fp32", "smoke": bool(args.smoke)},
               "prereg": verdicts(done) if done else {}}
    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp77_summary{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    import csv
    fields = ["task", "n_classes", "n_windows", "accuracy_A", "accuracy_B", "gain", "seed_floor", "lambda", "error_rate_A"]
    with open(os.path.join(OUT, f"exp77_tasks{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["contamination_gap"])
        w.writeheader()
        for r in done:
            row = {k: r.get(k) for k in fields}
            row["contamination_gap"] = r["contamination"]["gap"] if r.get("contamination") else None
            w.writerow(row)
    print(json.dumps(summary["prereg"], indent=1))


if __name__ == "__main__":
    main()
