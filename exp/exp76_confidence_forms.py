"""exp76: which form of the model's own confidence, and which aggregator, on the suite this project did not choose.

Why. A search of the 2024 to 2026 literature (docs/related_work.md, section 7) found three published challenges to
the record's choice of score that are cheap to answer. (1) Traub et al. 2024: AURC ranks methods wrongly and AUGRC
should be used. For a fixed set of errors AUGRC is a decreasing function of the failure AUROC, which exp70 already
recorded, so that one was answered from exp70's artifact before this run: the margin beats the best no-model control
on 23 of 24 tasks by AUROC, and among the model's own readings one minus top-1 beats top-1 minus top-2 on 8 of the
10 multi-class classification tasks where both were recorded, under either statistic. That second fact is what this
experiment follows up. (2) Whole-vector scores: a published workshop claim that the full distribution of class
margins beats top-1 minus top-2 (its definition could not be retrieved, so it is not tested under its name). (3)
Guarino et al. 2026: the aggregator from pixels to a region is a first-class choice; the record's window score is the
margin of the window-mean probabilities, chosen without a test.

Design. exp70's 24 tasks, OlmoEarth Base, Ai2's linear probe at seed 0 with exp70's recipe, so the errors are
exp70's errors; the consistency check is that the probability margin's excess AURC reproduces exp70's per task.
Readings, all label-free, higher = more suspect, from the same probe:
  margin            minus (top-1 minus top-2 probability), the record's score
  one_minus_top1    one minus the top probability, now also on the 7 segmentation tasks (exp70 could not separate it)
  entropy           of the probabilities
  logit_margin      minus (top-1 minus top-2 logit), the form oe_inferencex.signals.confidence uses
  margin_to_mean    minus (top logit minus the mean of the others)
  margin_cv         std over mean of the class margins (top logit minus each other logit), three classes or more:
                    one small margin among large ones is a two-way confusion, so a high ratio is suspect; the sign
                    is fixed here, before the run
  energy            minus logsumexp of the logits
  max_logit         minus the top logit
On the segmentation path logits and probabilities are averaged over the 4-px window first, as the record does.
Aggregators, segmentation only, against the record's margin of window-mean probabilities:
  pix_margin_mean   minus the mean over the window's pixels of the pixel margin
  pix_margin_min    minus the smallest pixel margin in the window
  pix_low_share     share of the window's pixels with a margin below 0.5
  pix_omt_mean      mean over pixels of one minus top-1
  nbr_margin_mean   minus the mean of the window margin over the 3x3 windows around it, the spatial-context form
Every reading is scored with exp70's statistics: excess AURC, AUROC (hence AUGRC's ordering), capture at 5/10/20%.

Preregistered, before the run, on numbers not yet seen.
  P1  the pattern extends to segmentation: on the multi-class segmentation tasks one minus top-1 has the lower
      excess AURC than the margin on a majority of them.
  P2  no whole-vector score (logit_margin, margin_to_mean, margin_cv, energy, max_logit) beats the better of margin
      and one_minus_top1 on 18 or more of the 24 tasks, the record's bar for a win.
  P3  no alternative aggregator beats the margin of window-mean probabilities on 6 or more of the 7 segmentation
      tasks.
  P4  descriptive: on how many tasks excess AURC and AUROC name the same best reading.
Falsification. If P2 fails, the winning score replaces the margin as the record's representative confidence and the
front page says so. If P3 fails, the aggregator becomes an option in oe_inferencex.assess. Whatever P1 says, the docs
state which member of the confidence family is best on multi-class tasks, and oe_inferencex.signals.confidence says
which it uses and why.
Outputs: exp/out/exp76_summary.json, exp76_readings.csv; per-task checkpoints under exp76_parts/ (gitignored).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp70_task_suite as e70                    # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PARTS = os.path.join(OUT, "exp76_parts")
FORMS = ("margin", "one_minus_top1", "entropy", "logit_margin", "margin_to_mean", "margin_cv", "energy", "max_logit")
VECTOR = ("logit_margin", "margin_to_mean", "margin_cv", "energy", "max_logit")
AGGS = ("pix_margin_mean", "pix_margin_min", "pix_low_share", "pix_omt_mean", "nbr_margin_mean")
LOW, WIN_BAR, AGG_BAR = 0.5, 18, 6


# ----------------------------------------------------------------------------- readings from probabilities and logits
def forms(P, Z):
    """The eight confidence forms from (N, C) probabilities and logits; higher = more suspect."""
    P = np.asarray(P, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
    sp = np.sort(P, axis=1); sz = np.sort(Z, axis=1)
    C = P.shape[1]
    out = {"margin": -(sp[:, -1] - sp[:, -2]), "one_minus_top1": 1.0 - sp[:, -1],
           "entropy": -(np.clip(P, 1e-12, 1) * np.log(np.clip(P, 1e-12, 1))).sum(1),
           "logit_margin": -(sz[:, -1] - sz[:, -2]),
           "margin_to_mean": -(sz[:, -1] - sz[:, :-1].mean(1)),
           "energy": -(np.log(np.exp(Z - Z.max(1, keepdims=True)).sum(1)) + Z.max(1)),
           "max_logit": -sz[:, -1]}
    if C >= 3:
        m = sz[:, -1:] - sz[:, :-1]                                  # the C - 1 class margins, all >= 0
        out["margin_cv"] = m.std(1) / np.maximum(m.mean(1), 1e-12)
    return out


def neighbour_mean(a):
    """Mean of a (N, h, w) map over the 3x3 windows around each window, over the neighbours that exist."""
    N, h, w = a.shape
    pad = np.full((N, h + 2, w + 2), np.nan)
    pad[:, 1:-1, 1:-1] = a
    stack = np.stack([pad[:, i:i + h, j:j + w] for i in range(3) for j in range(3)])
    return np.nanmean(stack, axis=0)


# ----------------------------------------------------------------------------- fits, exp70's recipe, logits kept
def fit_cls(task, cache):
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    C = int(max(int(ytr.max()), int(yte.max())) + 1)
    dev = e54.DEV
    torch.manual_seed(0)
    probe = e51.LinearProbe(in_dim=xtr.shape[-1], out_dim=C).to(dev).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=e51.LR)
    Xtr, Ytr = xtr.to(torch.float32), ytr.to(torch.int64)
    n = len(Ytr); steps = int(np.ceil(n / e51.BATCH))
    g = torch.Generator().manual_seed(0)
    probe.train()
    for epoch in range(e51.EPOCHS):
        order = torch.randperm(n, generator=g)
        for i in range(steps):
            idx = order[i * e51.BATCH:(i + 1) * e51.BATCH]
            lg = probe(Xtr[idx].to(dev))["logits"]
            torch.nn.functional.cross_entropy(lg, Ytr[idx].to(dev)).backward()
            e51.adjust_learning_rate(optimizer=opt, epoch=epoch + i / steps, total_epochs=e51.EPOCHS,
                                     warmup_epochs=int(e51.EPOCHS * 0.1), max_lr=e51.LR, min_lr=1.0e-5)
            opt.step(); opt.zero_grad()
    probe.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, len(xte), 4096):
            Z.append(probe(xte[i:i + 4096].to(torch.float32).to(dev))["logits"].cpu().numpy())
    Z = np.concatenate(Z).astype(np.float64)
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    etr, ete = xtr.to(torch.float32).numpy(), xte.to(torch.float32).numpy()
    return P, Z, yte.numpy(), etr, ete, C, {}


def fit_seg(task, cache):
    import torch
    import exp54_multiclass_embeddings as e54
    import exp74_suite_encoders as e74
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))
    probe = e54.train_probe(xtr, ytr, pp, C, e54.TASK_LR.get(task, 0.1), seed=0)
    N, H, W = yte.shape
    Wn = e54.WIN
    hw, ww = H // Wn, W // Wn
    wp = np.zeros((N, C, hw, ww), dtype=np.float32); wz = np.zeros_like(wp)
    agg = {k: np.zeros((N, hw, ww), dtype=np.float32) for k in ("pix_margin_mean", "pix_margin_min", "pix_low_share", "pix_omt_mean")}
    blk = lambda t: t[..., :hw * Wn, :ww * Wn].reshape(*t.shape[:-2], hw, Wn, ww, Wn)
    with torch.no_grad():
        for i in range(0, N, 64):
            x = xte[i:i + 64].to(e54.DEV, dtype=torch.float32)
            n, h, w = x.shape[:3]
            lg = probe(x)["logits"].reshape(n, h, w, C, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(n, C, h * pp, w * pp)
            if lg.shape[-2:] != (H, W):
                lg = torch.nn.functional.interpolate(lg, size=(H, W), mode="bilinear", align_corners=True)
            p = torch.softmax(lg, dim=1)
            wp[i:i + n] = blk(p).mean(dim=(3, 5)).cpu().numpy()
            wz[i:i + n] = blk(lg).mean(dim=(3, 5)).cpu().numpy()
            top2 = torch.topk(p, 2, dim=1).values
            pm = blk(top2[:, 0] - top2[:, 1])                                   # (n, hw, Wn, ww, Wn) pixel margins
            agg["pix_margin_mean"][i:i + n] = (-pm.mean(dim=(2, 4))).cpu().numpy()
            agg["pix_margin_min"][i:i + n] = (-pm.amin(dim=(2, 4))).cpu().numpy()
            agg["pix_low_share"][i:i + n] = (pm < LOW).float().mean(dim=(2, 4)).cpu().numpy()
            agg["pix_omt_mean"][i:i + n] = blk(1.0 - top2[:, 0]).mean(dim=(2, 4)).cpu().numpy()
    l = yte[:, :hw * Wn, :ww * Wn].numpy().reshape(N, hw, Wn, ww, Wn).transpose(0, 1, 3, 2, 4).reshape(N, hw, ww, Wn * Wn)
    valid = l >= 0
    ok = (valid.sum(-1) >= (Wn * Wn) // 2).reshape(-1)
    counts = np.zeros((N, hw, ww, C), dtype=np.int32)
    for c in range(C):
        counts[..., c] = ((l == c) & valid).sum(-1)
    y = counts.argmax(-1).reshape(-1)[ok]
    P = wp.transpose(0, 2, 3, 1).reshape(-1, C)[ok]
    Z = wz.transpose(0, 2, 3, 1).reshape(-1, C)[ok]
    srt = np.sort(wp, axis=1)
    extra = {k: v.reshape(-1)[ok].astype(np.float64) for k, v in agg.items()}
    extra["nbr_margin_mean"] = -neighbour_mean(srt[:, -1] - srt[:, -2]).reshape(-1)[ok].astype(np.float64)
    ete = e74.window_embeddings(xte.to(torch.float32).numpy(), hw, ww, H, W, Wn).reshape(-1, xte.shape[-1])[ok]
    etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
    return P, Z, y, etr, ete, C, extra


def run_task(task, cache):
    import exp73_alternatives_suite as e73
    seg = task in e70.TASKS_SEG
    P, Z, y, etr, ete, C, extra = (fit_seg if seg else fit_cls)(task, cache)
    sig = forms(P, Z)
    dec = P.argmax(1)
    err = (dec != y).astype(np.float64)
    sig.update(extra)
    sig.update(e73.controls_chunked(ete, etr, dec))
    sc = e70.score_task(sig, err, C)
    cname, cval = e70.best_control(sc)
    return {"family": "segmentation" if seg else "classification", "n_units": int(len(err)), "n_classes": int(C),
            "test_accuracy": float(1 - err.mean()), "error_rate": float(err.mean()), "signals": sc,
            "best_control": cname, "margin_lead": float(cval - sc["margin"]["excess_aurc"]), "source": e70.source_of(task)}


# ----------------------------------------------------------------------------- verdicts
def _better(r, a, b, stat="excess_aurc"):
    """+1 if reading a beats b on this task, -1 if it loses, 0 on a tie or if either is absent."""
    s = r["signals"]
    if a not in s or b not in s:
        return 0
    x, y = s[a][stat], s[b][stat]
    if abs(x - y) < 1e-12:
        return 0
    return (1 if x < y else -1) if stat == "excess_aurc" else (1 if x > y else -1)


def verdicts(results, e70_tasks=None):
    tasks = sorted(results)
    multi_seg = [t for t in tasks if results[t]["family"] == "segmentation" and results[t]["n_classes"] > 2]
    w = sum(1 for t in multi_seg if _better(results[t], "one_minus_top1", "margin") > 0)
    l = sum(1 for t in multi_seg if _better(results[t], "one_minus_top1", "margin") < 0)
    v = {"P1": {"holds": bool(multi_seg) and w > len(multi_seg) / 2, "one_minus_top1_better": w, "margin_better": l, "of": len(multi_seg), "tasks": multi_seg}}
    ref = {t: ("margin" if _better(results[t], "margin", "one_minus_top1") >= 0 else "one_minus_top1") for t in tasks}
    wins = {f: sum(1 for t in tasks if _better(results[t], f, ref[t]) > 0) for f in VECTOR}
    v["P2"] = {"holds": all(x < WIN_BAR for x in wins.values()), "wins_over_the_better_of_margin_and_omt": wins, "of": len(tasks)}
    seg = [t for t in tasks if results[t]["family"] == "segmentation"]
    awins = {a: sum(1 for t in seg if _better(results[t], a, "margin") > 0) for a in AGGS}
    v["P3"] = {"holds": all(x < AGG_BAR for x in awins.values()), "wins_over_margin": awins, "of": len(seg)}
    cand = lambda r: [k for k in r["signals"] if not k.startswith("ctl_")]
    same = sum(1 for t in tasks if min(cand(results[t]), key=lambda k: results[t]["signals"][k]["excess_aurc"]) ==
               max(cand(results[t]), key=lambda k: results[t]["signals"][k]["auroc"]))
    v["P4"] = {"note": "descriptive", "same_best_reading_under_both_statistics": same, "of": len(tasks)}
    multi = [t for t in tasks if results[t]["n_classes"] > 2]
    v["descriptive"] = {
        "one_minus_top1_vs_margin_multiclass": {s: [sum(1 for t in multi if _better(results[t], "one_minus_top1", "margin", s) > 0),
                                                    sum(1 for t in multi if _better(results[t], "one_minus_top1", "margin", s) < 0)] for s in ("excess_aurc", "auroc")},
        "n_multiclass": len(multi),
        "median_eaurc": {f: float(np.median([results[t]["signals"][f]["excess_aurc"] for t in tasks if f in results[t]["signals"]])) for f in FORMS + AGGS},
        "best_reading_by_eaurc": {t: min(cand(results[t]), key=lambda k: results[t]["signals"][k]["excess_aurc"]) for t in tasks},
        "margin_beats_best_control_by_auroc": sum(1 for t in tasks if results[t]["signals"]["margin"]["auroc"] > max(results[t]["signals"][c]["auroc"] for c in e70.CONTROLS))}
    if e70_tasks:
        dev = [abs(results[t]["signals"]["margin"]["excess_aurc"] - e70_tasks[t]["signals"]["margin"]["excess_aurc"]) for t in tasks if t in e70_tasks]
        v["consistency"] = {"max_abs_deviation_of_margin_eaurc_from_exp70": float(max(dev)) if dev else None, "tasks_compared": len(dev)}
    return v


# ----------------------------------------------------------------------------- driver
def cmd_all(args):
    cache = os.environ.get("HF_HOME")
    os.makedirs(PARTS, exist_ok=True)
    results, rows, failures = {}, [], {}
    for task in e70.TASKS_CLS + e70.TASKS_SEG:
        if args.only and task not in args.only:
            continue
        part = os.path.join(PARTS, f"{task}.json")
        if os.path.exists(part) and not args.redo:
            with open(part) as fh:
                r = json.load(fh)
        else:
            t0 = time.time()
            try:
                r = run_task(task, cache)
            except Exception as exc:
                failures[task] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  {task:44s} FAILED {failures[task][:100]}", flush=True)
                continue
            r["seconds"] = round(time.time() - t0, 1)
            with open(part, "w") as fh:
                json.dump(r, fh, default=float)
        results[task] = r
        s = r["signals"]
        for k, x in s.items():
            rows.append({"task": task, "family": r["family"], "n_classes": r["n_classes"], "reading": k,
                         "excess_aurc": x["excess_aurc"], "auroc": x["auroc"], **{f"capture_{b}": c for b, c in x["capture"].items()}})
        best = min((k for k in s if not k.startswith("ctl_")), key=lambda k: s[k]["excess_aurc"])
        print(f"  {task:44s} {r['family'][:3]} C={r['n_classes']:2d} n={r['n_units']:7d} acc {r['test_accuracy']:.3f} | margin {s['margin']['excess_aurc']:.4f} "
              f"omt {s['one_minus_top1']['excess_aurc']:.4f} | best {best} {s[best]['excess_aurc']:.4f} ({r['seconds']:.0f}s)", flush=True)
    e70_tasks = None
    p70 = os.path.join(OUT, "exp70_summary.json")
    if os.path.exists(p70):
        with open(p70) as fh:
            e70_tasks = json.load(fh)["results"]["tasks"]
    summary = {"experiment": "exp76 which form of confidence and which aggregator, on Ai2's published suite",
               "config": {"model": e70.MODEL, "forms": list(FORMS), "aggregators": list(AGGS), "low_margin": LOW,
                          "bars": {"P2_wins": WIN_BAR, "P3_wins": AGG_BAR}, "budgets": list(e70.BUDGETS), "controls": list(e70.CONTROLS)},
               "results": {"tasks": results, "failures": failures},
               "verdicts": verdicts(results, e70_tasks) if results else {}}
    with open(os.path.join(OUT, "exp76_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, "exp76_readings.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for k, x in summary["verdicts"].items():
        print(f"{k}: {x.get('holds', '')}  {json.dumps({a: b for a, b in x.items() if a != 'holds'})[:260]}", flush=True)
    if failures:
        print(f"failed: {sorted(failures)}", flush=True)
    return summary


# ----------------------------------------------------------------------------- smokes
def smoke(args):
    """Torch-free: the forms, the neighbour mean and the verdicts on planted data."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((500, 5)) * 2
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    f = forms(P, Z)
    assert set(f) == set(FORMS) and all(v.shape == (500,) for v in f.values())
    assert np.all(f["margin"] <= 0) and np.all(f["one_minus_top1"] >= 0) and np.all(f["margin_cv"] >= 0)
    assert np.allclose(f["energy"], -np.log(np.exp(Z).sum(1)))
    f2 = forms(P[:, :2] / P[:, :2].sum(1, keepdims=True), Z[:, :2])
    assert "margin_cv" not in f2, "the dispersion needs three classes"
    assert np.allclose(np.argsort(f2["margin"]), np.argsort(f2["one_minus_top1"])), "binary: the two are the same ranking"
    a = np.arange(12, dtype=float).reshape(1, 3, 4)
    nm = neighbour_mean(a)
    assert nm.shape == a.shape and abs(nm[0, 0, 0] - np.mean([0, 1, 4, 5])) < 1e-12 and abs(nm[0, 1, 1] - np.mean([0, 1, 2, 4, 5, 6, 8, 9, 10])) < 1e-12

    def task(fam, C, m, omt, vec, agg):
        s = {"margin": {"excess_aurc": m, "auroc": 0.8}, "one_minus_top1": {"excess_aurc": omt, "auroc": 0.81},
             "entropy": {"excess_aurc": m + 0.01, "auroc": 0.7}, "ctl_embedding_distance": {"excess_aurc": 0.3, "auroc": 0.5},
             "ctl_class_rarity": {"excess_aurc": 0.3, "auroc": 0.5}}
        s.update({k: {"excess_aurc": vec, "auroc": 0.6} for k in VECTOR})
        if fam == "segmentation":
            s.update({k: {"excess_aurc": agg, "auroc": 0.6} for k in AGGS})
        return {"family": fam, "n_classes": C, "signals": s}
    R = {f"c{i}": task("classification", 4, 0.10, 0.09, 0.2, None) for i in range(17)}
    R.update({f"s{i}": task("segmentation", 6, 0.10, 0.09, 0.2, 0.2) for i in range(7)})
    v = verdicts(R)
    assert v["P1"]["holds"] and v["P1"]["one_minus_top1_better"] == 7 and v["P2"]["holds"] and v["P3"]["holds"]
    R2 = {k: task(r["family"], r["n_classes"], 0.10, 0.11, 0.05, 0.05) for k, r in R.items()}
    v2 = verdicts(R2)
    assert not v2["P1"]["holds"] and not v2["P2"]["holds"] and not v2["P3"]["holds"]
    assert v2["P2"]["wins_over_the_better_of_margin_and_omt"]["energy"] == 24 and v2["P3"]["wins_over_margin"]["pix_margin_min"] == 7
    print("smoke OK: eight forms, binary degeneracy, neighbour mean, and the verdicts hold and fail as planted")


def smoke_torch(args):
    """Both fits through the real code on synthetic data; the window readings agree with exp54's window quantities."""
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    e51.EPOCHS, e54.EPOCHS = 2, 2
    rng = np.random.default_rng(0)
    D, C, pp, h = 16, 3, 4, 8
    emb = torch.tensor(rng.standard_normal((6, h, h, D)), dtype=torch.float32)
    lab = torch.tensor(rng.integers(0, C, (6, h * pp, h * pp)), dtype=torch.int64); lab[:, :4, :4] = -1
    store = {("m", "t", "train"): (emb, lab), ("m", "t", "test"): (emb, lab)}
    e54.load_theirs = lambda model, task, split, cache: store[("m", "t", split)]
    e70.MODEL = "m"; e70.TASKS_SEG = ["t"]
    P, Z, y, etr, ete, C2, extra = fit_seg("t", None)
    probe_q = None
    assert P.shape == Z.shape and P.shape[1] == C and set(extra) == set(AGGS) and all(v.shape == (len(y),) for v in extra.values())
    assert np.all(extra["pix_margin_min"] >= extra["pix_margin_mean"] - 1e-6), "minus the smallest pixel margin is the more suspect of the two"
    assert np.all((extra["pix_low_share"] >= 0) & (extra["pix_low_share"] <= 1))
    print("smoke-torch OK: the segmentation fit returns window probabilities, logits and the five aggregators")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-torch", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args); return
    if args.smoke_torch:
        smoke_torch(args); return
    cmd_all(args)


if __name__ == "__main__":
    main()
