#!/usr/bin/env python
"""exp70: the ranking protocol across the whole of Ai2's published task suite, on tasks this project did not choose.

Why. The strongest objection to everything in this repository is that we picked our own testbeds. Sen1Floods11, the AWF
points, GEOID-Flood and the four references of exp66 to exp69 were each chosen here, one at a time, for a reason we
also wrote. A reader is entitled to ask whether the margin ranks errors on tasks nobody selected to make it look good.

Ai2's `olmoearth-paper-embeddings` answers that objection cheaply and completely. It is the embedding set behind their
own paper: 25 tasks for OlmoEarth Base, chosen by them for their evaluation and not by us, with the splits fixed in the
files. Seven carry the `m_` prefix because they are GEO-Bench tasks, a third-party benchmark from ServiceNow Research
with its own paper and published baselines. Because the embeddings are published there is no encoder pass at all: the
whole suite is 68 GB of tensors and the protocol runs on top of them.

What this is not. GEO-Bench and Ai2's suite both measure ACCURACY, and no public benchmark measures label-free error
ranking, so nothing here is a leaderboard result and nothing competes with a published number. The suite is used as a
task set, which is exactly what is needed: it converts "the margin ranks errors on the testbeds we chose" into "the
margin ranks errors on a suite somebody else fixed".

Design. 24 of the 25 tasks, in two families the protocol treats differently because the data do.

  Classification, 17 tasks, one pooled embedding and one label per sample, so the graded unit is the sample:
  m_eurosat, m_brick_kiln, m_forestnet, m_so2sat (GEO-Bench), awf_landsat/sentinel1/sentinel2, nandi_landsat/
  sentinel1/sentinel2, six cropharvest variants, breizhcrops. Two to seventeen classes, 200 to 122,614 samples.
  Segmentation, 7 tasks, dense embeddings on a patch grid, so the graded unit is the 4-px window as everywhere here:
  mados, sen1floods11, pastis_sentinel1/sentinel2/sentinel1_sentinel2, m_cashew_plant, m_sa_crop_type (GEO-Bench).

  m_bigearthnet is excluded and the reason is recorded rather than left out: its labels are multi-label, 43 binary
  tags per sample, so a top-1-minus-top-2 margin is not defined on it without inventing a convention this project has
  never tested.

  The probe is Ai2's own linear recipe at their published settings, fitted on their train split and reported on their
  test split. Rankers are the margin, one minus top-1 and predictive entropy; the controls are the distance of a
  sample's embedding from the training set's mean embedding, which uses no labels and no probe, and the rarity of the
  predicted class, which uses the decision and nothing else.

  The unit of the headline is the TASK. Per task the protocol produces an excess AURC for every ranker and every
  control; across tasks a one-sided exact sign test asks on how many of the 24 the margin beats the best control. That
  is the generality claim, and it is the one a reader who suspects testbed selection actually wants.

Preregistered (one-sided, decided before the first run):
  P1  the margin beats the best no-model control on at least 18 of the 24 tasks (exact sign test over tasks, p < 0.05).
  P2  it is not a property of one task family: the margin beats the best control on at least 13 of the 17
      classification tasks and at least 6 of the 7 segmentation tasks, each tested separately.
  P3  exp68's lesson generalises beyond the one task it was found on. There the ordering of the model signals inverted
      when the probe had memorised its fit set: the margin went from last of four to the front once regularisation was
      chosen on held-out ground. So across these 24 tasks the margin's advantage over predictive entropy is smaller on
      the tasks whose probe has the larger generalisation gap, measured as train accuracy minus test accuracy, with the
      tasks split at the median gap and compared by an exact sign test.
  Falsification. P1 fails if the margin loses to a control that sees no model on a third or more of a suite this
  project did not assemble, which would confine every ranking result here to testbeds chosen with knowledge of the
  answer. P2 fails if the result belongs to one family, in which case the claim must be narrowed to that family by
  name. P3 fails if the signal ordering is unrelated to how well the probe generalises, which would make exp68's
  finding a property of LUCAS rather than of probes, and the warning now on the front page would have to come off it.
  Stated predictions, not tested: accuracy spans a wide range across the suite and the margin's lead is not a function
  of it; the two-class tasks give the least room for a margin and are where the ranking should be weakest.

Caveats carried into the record. The embeddings are Ai2's, extracted with their settings, so this inherits whatever
their extraction did and is not an independent implementation of their models. The splits are theirs, and where a task
has a small test split the per-task excess AURC is noisy, which is why the headline is a count over tasks and not a
pooled number. Several tasks share a source (three AWF sensors, three nandi sensors, six cropharvest variants, three
PASTIS variants), so the 24 are not 24 independent datasets and the sign test over them overstates independence; the
count by distinct SOURCE is reported beside it.

Inputs: `allenai/olmoearth-paper-embeddings`, about 68 GB for olmoearth_base, cached under $HF_HOME.
Outputs: exp/out/exp70_summary.json, exp/out/exp70_tasks.csv.
--smoke: synthetic embeddings, no download, _smoke outputs.
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
HUB = "allenai/olmoearth-paper-embeddings"
MODEL = "olmoearth_base"
BUDGETS = (0.05, 0.10, 0.20)

# one pooled embedding and one label per sample; the graded unit is the sample
TASKS_CLS = ["m_eurosat", "m_brick_kiln", "m_forestnet", "m_so2sat",
             "awf_landsat", "awf_sentinel1", "awf_sentinel2",
             "nandi_landsat", "nandi_sentinel1", "nandi_sentinel2",
             "cropharvest_Togo_12_sentinel1", "cropharvest_Togo_12_sentinel2",
             "cropharvest_Togo_12_sentinel2_sentinel1",
             "cropharvest_Peoples_Republic_of_China_6",
             "cropharvest_Peoples_Republic_of_China_6_sentinel1",
             "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
             "breizhcrops"]
# dense embeddings on a patch grid; the graded unit is the 4-px window
TASKS_SEG = ["mados", "sen1floods11", "pastis_sentinel1", "pastis_sentinel2",
             "pastis_sentinel1_sentinel2", "m_cashew_plant", "m_sa_crop_type"]
EXCLUDED = {"m_bigearthnet": "multi-label (43 binary tags per sample); a top-1 minus top-2 margin is undefined "
                             "without a convention this project has never tested"}
GEOBENCH = {"m_eurosat", "m_brick_kiln", "m_forestnet", "m_so2sat", "m_bigearthnet",
            "m_cashew_plant", "m_sa_crop_type"}
# tasks sharing a source, so the sign test over tasks overstates independence
SOURCE = {**{t: "awf" for t in TASKS_CLS if t.startswith("awf")},
          **{t: "nandi" for t in TASKS_CLS if t.startswith("nandi")},
          **{t: "cropharvest_togo" for t in TASKS_CLS if t.startswith("cropharvest_Togo")},
          **{t: "cropharvest_china" for t in TASKS_CLS if t.startswith("cropharvest_Peoples")},
          **{t: "pastis" for t in TASKS_SEG if t.startswith("pastis")}}


def source_of(task):
    return SOURCE.get(task, task)


# ----------------------------------------------------------------------------- signals
def readings_from_probs(p):
    """Margin, one minus top-1 and entropy from a (N, C) probability matrix; higher = more suspect."""
    p = np.asarray(p, dtype=np.float64)
    srt = np.sort(p, axis=1)
    top1 = srt[:, -1]
    margin = top1 - srt[:, -2] if p.shape[1] > 1 else top1
    ent = -(np.clip(p, 1e-12, 1) * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return {"margin": -margin, "one_minus_top1": 1.0 - top1, "entropy": ent}, p.argmax(1)


def controls(emb_test, emb_train, dec):
    """Two controls. The embedding distance uses no labels and no probe: how far a sample sits from the training set's
    mean embedding, which is the nearest thing to a no-model reading these files allow, since the imagery is not
    published with them. The class rarity uses the decision and nothing else."""
    mu = np.asarray(emb_train, dtype=np.float64).mean(0)
    d = np.linalg.norm(np.asarray(emb_test, dtype=np.float64) - mu, axis=1)
    freq = collections.Counter(np.asarray(dec).tolist())
    rare = np.array([-np.log(max(freq[int(c)], 1) / max(len(dec), 1)) for c in dec])
    return {"ctl_embedding_distance": d, "ctl_class_rarity": rare}


def score_task(sig, err, n_classes):
    """Excess AURC and capture for every signal on one task's graded units."""
    out = {}
    for name, u in sig.items():
        out[name] = {"excess_aurc": float(metrics.excess_aurc(u, err)),
                     "capture": {str(b): float(v) for b, v in
                                 metrics.capture_at_budget_expected(u, err, BUDGETS).items()},
                     "auroc": float(metrics.weighted_auroc(u, err, np.ones(len(err))))}
    return out


# ----------------------------------------------------------------------------- the two families
def run_classification(task, cache, seed=0):
    """One pooled embedding and one label per sample: Ai2's linear recipe, graded on their test split."""
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54

    xtr, ytr = e54.load_theirs(MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(MODEL, task, "test", cache)
    if ytr.ndim != 1:
        raise ValueError(f"{task}: labels have shape {tuple(ytr.shape)}, not one label per sample")
    C = int(max(int(ytr.max()), int(yte.max())) + 1)
    D = xtr.shape[-1]
    dev = e54.DEV
    torch.manual_seed(seed)
    probe = e51.LinearProbe(in_dim=D, out_dim=C).to(dev).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=e51.LR)
    Xtr = xtr.to(torch.float32)
    Ytr = ytr.to(torch.int64)
    n = len(Ytr)
    steps = int(np.ceil(n / e51.BATCH))
    g = torch.Generator().manual_seed(seed)
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

    def probs(x):
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                out.append(torch.softmax(probe(x[i:i + 4096].to(torch.float32).to(dev))["logits"], dim=1).cpu().numpy())
        return np.concatenate(out)

    p_te, p_tr = probs(xte), probs(xtr)
    sig, dec = readings_from_probs(p_te)
    sig.update(controls(xte.to(torch.float32).numpy(), xtr.to(torch.float32).numpy(), dec))
    y = yte.numpy()
    err = (dec != y).astype(np.float64)
    return {"family": "classification", "n_units": int(len(y)), "n_classes": C,
            "test_accuracy": float((dec == y).mean()),
            "train_accuracy": float((p_tr.argmax(1) == ytr.numpy()).mean()),
            "error_rate": float(err.mean()), "signals": score_task(sig, err, C)}


def run_segmentation(task, cache, seed=0):
    """Dense embeddings on a patch grid: Ai2's segmentation probe, graded on the 4-px windows used everywhere here."""
    import torch
    import exp54_multiclass_embeddings as e54

    xtr, ytr = e54.load_theirs(MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(MODEL, task, "test", cache)
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))          # pixels per patch on this task's grid
    lr = e54.TASK_LR.get(task, 0.1)
    probe = e54.train_probe(xtr, ytr, pp, C, lr, seed=seed)
    q_te = e54.window_quantities(probe, xte, yte, pp, C)
    q_tr = e54.window_quantities(probe, xtr, ytr, pp, C)
    ok = q_te["ok"].ravel()
    err = q_te["err"].ravel()[ok]
    dec = q_te["dec"].ravel()[ok]
    # the window-level probabilities are not returned, so the two confidence readings come from what is:
    # the margin and the entropy window_quantities computes, and one minus top-1 is not separable here
    sig = {"margin": -q_te["margin"].ravel()[ok].astype(np.float64),
           "entropy": q_te["entropy"].ravel()[ok].astype(np.float64)}
    # the embedding control on the window grid: distance of each window's embedding from the training mean
    etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
    ete = xte.to(torch.float32).numpy().reshape(-1, xte.shape[-1])[ok]
    sig.update(controls(ete, etr, dec))
    ok_tr = q_tr["ok"]
    return {"family": "segmentation", "n_units": int(ok.sum()), "n_classes": C, "patch_px": pp,
            "test_accuracy": float(1.0 - err.mean()),
            "train_accuracy": float(1.0 - q_tr["err"][ok_tr].mean()),
            "pixel_accuracy": q_te["pixel_accuracy"], "pixel_miou": q_te["pixel_miou"],
            "error_rate": float(err.mean()), "signals": score_task(sig, err, C)}


# ----------------------------------------------------------------------------- driver
CONTROLS = ("ctl_embedding_distance", "ctl_class_rarity")


def best_control(sig):
    c = {k: v["excess_aurc"] for k, v in sig.items() if k in CONTROLS}
    name = min(c, key=c.get)
    return name, c[name]


def cmd_run(args):
    cache = os.environ.get("HF_HOME")
    results, rows, failures = {}, [], {}
    for task in TASKS_CLS + TASKS_SEG:
        if args.only and task not in args.only:
            continue
        fn = run_classification if task in TASKS_CLS else run_segmentation
        t0 = time.time()
        try:
            r = fn(task, cache, seed=args.seed)
        except Exception as exc:                       # a task that will not load must be named, not silently dropped
            failures[task] = f"{type(exc).__name__}: {str(exc)[:200]}"
            print(f"  {task:48s} FAILED {failures[task][:90]}", flush=True)
            continue
        cname, cval = best_control(r["signals"])
        r["best_control"] = cname
        r["margin_lead"] = float(cval - r["signals"]["margin"]["excess_aurc"])
        r["generalisation_gap"] = float(r["train_accuracy"] - r["test_accuracy"])
        r["geobench"] = task in GEOBENCH
        r["source"] = source_of(task)
        r["seconds"] = round(time.time() - t0, 1)
        results[task] = r
        rows.append({"task": task, "family": r["family"], "source": r["source"], "geobench": r["geobench"],
                     "n_units": r["n_units"], "n_classes": r["n_classes"],
                     "test_accuracy": r["test_accuracy"], "train_accuracy": r["train_accuracy"],
                     "generalisation_gap": r["generalisation_gap"], "error_rate": r["error_rate"],
                     "margin_excess_aurc": r["signals"]["margin"]["excess_aurc"],
                     "best_control": cname, "control_excess_aurc": cval, "margin_lead": r["margin_lead"],
                     "margin_capture_10": r["signals"]["margin"]["capture"]["0.1"],
                     "margin_auroc": r["signals"]["margin"]["auroc"],
                     "entropy_excess_aurc": r["signals"]["entropy"]["excess_aurc"],
                     "seconds": r["seconds"]})
        print(f"  {task:48s} {r['family'][:3]} n={r['n_units']:7d} C={r['n_classes']:2d} "
              f"acc {r['test_accuracy']:.3f} gap {r['generalisation_gap']:+.3f} "
              f"lead {r['margin_lead']:+.4f} over {cname[4:]} ({r['seconds']:.0f}s)", flush=True)
    return results, rows, failures


def verdicts(results):
    """The three preregistrations, decided on the tasks that ran."""
    tasks = sorted(results)
    wins = [t for t in tasks if results[t]["margin_lead"] > 0]
    cls = [t for t in tasks if results[t]["family"] == "classification"]
    seg = [t for t in tasks if results[t]["family"] == "segmentation"]
    w_cls = [t for t in cls if results[t]["margin_lead"] > 0]
    w_seg = [t for t in seg if results[t]["margin_lead"] > 0]
    p_all = float(stats.sign_test(len(wins), len(tasks) - len(wins), alternative="greater"))

    # the sources, because three AWF sensors are not three independent datasets
    by_src = collections.defaultdict(list)
    for t in tasks:
        by_src[results[t]["source"]].append(results[t]["margin_lead"])
    src_wins = sum(1 for v in by_src.values() if float(np.mean(v)) > 0)

    v = {"P1": {"holds": bool(len(wins) >= 18 and p_all < 0.05),
                "wins": len(wins), "of": len(tasks), "p": p_all,
                "distinct_sources": len(by_src), "source_wins": src_wins,
                "note": "the count by source is beside the count by task because several tasks share a source"},
         "P2": {"holds": bool(len(w_cls) >= 13 and len(w_seg) >= 6),
                "classification": f"{len(w_cls)}/{len(cls)}", "segmentation": f"{len(w_seg)}/{len(seg)}"}}

    # P3: exp68 found the signal ordering inverts when the probe has memorised its fit set. Split the tasks at the
    # median generalisation gap and ask whether the margin's advantage over entropy is smaller on the worse half.
    adv = {t: results[t]["signals"]["entropy"]["excess_aurc"] - results[t]["signals"]["margin"]["excess_aurc"]
           for t in tasks if "entropy" in results[t]["signals"]}
    gaps = {t: results[t]["generalisation_gap"] for t in adv}
    if len(adv) >= 6:
        med = float(np.median(list(gaps.values())))
        wide = [adv[t] for t in adv if gaps[t] > med]
        tight = [adv[t] for t in adv if gaps[t] <= med]
        w = sum(1 for a in wide for b in tight if a < b)
        l = sum(1 for a in wide for b in tight if a > b)
        # The 144 wide-by-tight pairs come from 24 tasks, so they are not 144 independent trials. Until 2026-09-22
        # this ran a binomial sign test over them and recorded p = 6.9e-15. The test whose statistic is this pair
        # count is the rank-sum test, exact over every relabelling of the 24 tasks: p = 0.0055. The verdict holds.
        rs = stats.rank_sum_test(wide, tight, alternative="less")
        v["P3"] = {"holds": bool(float(np.median(wide)) < float(np.median(tight)) and rs["p"] < 0.05),
                   "median_gap": med,
                   "margin_over_entropy_wide_gap": float(np.median(wide)),
                   "margin_over_entropy_tight_gap": float(np.median(tight)),
                   "n_wide": len(wide), "n_tight": len(tight),
                   "pairs_wide_below_tight": w, "pairs_wide_above_tight": l,
                   "rank_sum_p": rs["p"], "rank_sum_method": rs["method"],
                   "note": "exp68 found the ordering of the model signals inverts when the probe memorises its fit "
                           "set; this asks whether that is a property of probes or of that one task"}
    else:
        v["P3"] = {"holds": None, "note": "too few tasks scored to split at the median gap"}
    return v


def cmd_all(args):
    os.makedirs(OUT, exist_ok=True)
    print(f"{len(TASKS_CLS)} classification + {len(TASKS_SEG)} segmentation tasks, "
          f"{len(GEOBENCH & set(TASKS_CLS + TASKS_SEG))} of them GEO-Bench", flush=True)
    results, rows, failures = cmd_run(args)
    summary = {"experiment": "exp70 the ranking protocol across Ai2's published task suite, on tasks we did not choose",
               "config": {"hub": HUB, "model": MODEL, "budgets": list(BUDGETS),
                          "n_classification": len(TASKS_CLS), "n_segmentation": len(TASKS_SEG),
                          "geobench_tasks": sorted(GEOBENCH & set(TASKS_CLS + TASKS_SEG)),
                          "excluded": EXCLUDED, "controls": list(CONTROLS),
                          "caveats": ["the embeddings are Ai2's, extracted with their settings, so this inherits "
                                      "their extraction and is not an independent implementation of their models",
                                      "several tasks share a source, so the 24 are not 24 independent datasets and "
                                      "the count by distinct source is reported beside the count by task",
                                      "the suite measures accuracy and no public benchmark measures label-free error "
                                      "ranking, so this is a task set and not a leaderboard result"]},
               "results": {"tasks": results, "failures": failures},
               "verdicts": verdicts(results) if results else {}}
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp70_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, f"exp70_tasks{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for k, x in summary["verdicts"].items():
        print(f"{k}: {x.get('holds')}  {json.dumps({a: b for a, b in x.items() if a not in ('holds', 'note')})[:150]}",
              flush=True)
    if failures:
        print(f"failed to score {len(failures)} tasks: {sorted(failures)}", flush=True)
    return summary


def smoke(args):
    """Synthetic embeddings through the scoring path, with a planted signal and a planted null."""
    rng = np.random.default_rng(0)
    n, C = 3000, 5
    y = rng.integers(0, C, n)
    # a probability matrix whose margin is informative about the error, and an uninformative embedding
    p = rng.dirichlet(np.ones(C) * 0.6, size=n)
    dec = p.argmax(1)
    err = (dec != y).astype(np.float64)
    sig, dec2 = readings_from_probs(p)
    assert np.array_equal(dec, dec2)
    assert sig["margin"].max() <= 0 + 1e-12, "suspicion is the negated margin, so it must be non-positive"
    assert np.all(sig["one_minus_top1"] >= 0) and np.all(sig["entropy"] >= 0)
    # a degenerate one-class problem must not divide by nothing
    one, _ = readings_from_probs(np.ones((10, 1)))
    assert np.isfinite(one["margin"]).all()

    ctl = controls(rng.standard_normal((n, 16)), rng.standard_normal((500, 16)), dec)
    assert set(ctl) == set(CONTROLS) and all(len(v) == n for v in ctl.values())
    assert (ctl["ctl_class_rarity"] >= 0).all()

    sig.update(ctl)
    sc = score_task(sig, err, C)
    assert set(sc) == set(sig)
    for k, v in sc.items():
        assert np.isfinite(v["excess_aurc"]) and 0 <= v["capture"]["0.1"] <= 1

    # the verdict machinery on planted per-task results
    fake = {}
    for i in range(24):
        lead = 0.05 if i < 20 else -0.02                       # 20 of 24 win
        fam = "classification" if i < 17 else "segmentation"
        fake[f"t{i}"] = {"family": fam, "margin_lead": lead, "source": f"s{i}",
                         "train_accuracy": 0.9, "test_accuracy": 0.9 - 0.01 * i,
                         "generalisation_gap": 0.01 * i,
                         "signals": {"margin": {"excess_aurc": 0.10 + 0.001 * i},
                                     "entropy": {"excess_aurc": 0.12}}}
    v = verdicts(fake)
    assert v["P1"]["wins"] == 20 and v["P1"]["of"] == 24 and v["P1"]["holds"] is True
    assert v["P1"]["distinct_sources"] == 24
    assert v["P3"]["holds"] in (True, False)
    assert v["P3"]["n_wide"] + v["P3"]["n_tight"] == 24
    # planted: the wide-gap half has a SMALLER margin-over-entropy advantage, so P3 must hold
    assert v["P3"]["margin_over_entropy_wide_gap"] < v["P3"]["margin_over_entropy_tight_gap"]
    assert v["P3"]["holds"] is True
    print("smoke OK: readings, controls, scoring, and the three verdicts on planted task results")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None, help="run only these tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    cmd_all(args)


if __name__ == "__main__":
    main()
