"""exp73: the strong alternatives on the suite this project did not choose.

exp70 showed that the margin beats the best NO-MODEL control on 24 of 24 of Ai2's published tasks. That is the
bar a signal must clear to exist, not the bar a practitioner cares about, which is whether the alternatives the
literature actually proposes would have ranked the errors better: an ensemble's disagreement, a nearest-neighbour
typicality score, a Mahalanobis distance in feature space. Those were tested here before and lost, but on the
flood and water testbeds this project chose, and a reader is entitled to ask whether the loss travels to
twenty-four tasks chosen by someone else. This runs them there: the same published embeddings, the same probe
recipe, the same graded units and the same scoring as exp70, so the only thing that changes is the signal.

The signals, all label-free at inference.
  ensemble        K=5 probes differing only in seed. Classification, where per-unit probabilities exist: mutual
                  information (entropy of the mean minus the mean entropy), vote disagreement (one minus the modal
                  vote share), and the bag margin (the margin of the averaged probabilities, exp49's bag).
                  Segmentation, where the window path yields decisions and margins but not probabilities: vote
                  disagreement, and the standard deviation of the margin across seeds.
  knn_dist        distance to the 10th nearest training embedding (Sun et al. 2022), fp32, against a seeded
                  bank of at most 20,000 training units, the size recorded per task.
  mahalanobis     the minimum over classes of the Mahalanobis distance to the class mean with a shared, shrunk
                  covariance (Lee et al. 2018), fitted on the bank's PREDICTED classes rather than its labels, so
                  that labels grade and never train, as everywhere in this repository.
Alongside: the margin and entropy of the seed-0 probe and exp70's two no-model controls, unchanged.

Preregistered, one-sided, on the same 18-of-24 and p < 0.05 bar exp70 set.
  P1  the margin beats the primary ensemble signal (mutual information on classification, vote disagreement on
      segmentation) by excess AURC on at least 18 of the 24 tasks, sign test p < 0.05.
  P2  the same against knn_dist.
  P3  the same against mahalanobis.
  P4  descriptive, not a prediction: on how many tasks each alternative beats the best no-model control at all,
      because a signal that cannot clear that bar is not a competitor, and one that clears it and still loses to
      the margin is the interesting case.
Falsification. If an alternative wins on more tasks than not, the claim "no encoder-internal or ensemble signal
beats confidence" is confined to the water testbeds and the front page must say so. If the bag margin wins,
exp49's bag result under v1 was the rule and not the exception.

What this is not. Not new probes: the recipe, learning rate and epochs are Ai2's, as in exp70. Not a search over
K, k or the shrinkage; each is fixed once here and a different value is a different experiment.
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
import exp70_task_suite as e70                    # noqa: E402
from oe_inferencex import stats                   # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
K_SEEDS = 5
KNN_K = 10
BANK = 20000
SHRINK = 0.1
ALTS = ("ensemble", "knn_dist", "mahalanobis")


# ----------------------------------------------------------------------------- the signals
def _entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1)


def ensemble_from_probs(prob_list):
    """Classification: K probability matrices (N, C) into three disagreement readings, higher = more suspect."""
    P = np.stack([np.asarray(p, dtype=np.float64) for p in prob_list])       # K, N, C
    mean = P.mean(0)
    mi = _entropy(mean) - _entropy(P).mean(0)
    modal = mean.argmax(-1)
    vote_dis = 1.0 - (P.argmax(-1) == modal[None, :]).mean(0)
    srt = np.sort(mean, axis=-1)
    bag_margin = -(srt[:, -1] - srt[:, -2]) if mean.shape[1] > 1 else -srt[:, -1]
    return {"ens_mi": mi, "ens_vote_dis": vote_dis, "bag_margin": bag_margin}


def ensemble_from_decisions(dec_list, margin_list, n_classes):
    """Segmentation: K decision and margin vectors into two disagreement readings, higher = more suspect."""
    D = np.stack([np.asarray(d, dtype=np.int64) for d in dec_list])            # K, N
    M = np.stack([np.asarray(m, dtype=np.float64) for m in margin_list])
    n = D.shape[1]
    counts = np.zeros((n, n_classes), dtype=np.int32)
    rows = np.arange(n)
    for k in range(D.shape[0]):
        np.add.at(counts, (rows, np.clip(D[k], 0, n_classes - 1)), 1)
    vote_dis = 1.0 - counts.max(1) / D.shape[0]
    return {"ens_vote_dis": vote_dis, "ens_margin_std": M.std(0)}


def draw_bank(n_train, rng):
    """The seeded subset of training units the two feature-space signals see; its size is recorded."""
    if n_train <= BANK:
        return np.arange(n_train)
    return np.sort(rng.choice(n_train, size=BANK, replace=False))


def knn_distance(test, bank, k=KNN_K, chunk=4096):
    """Euclidean distance to the k-th nearest bank row, fp32, in chunks; torch on the GPU when there is one."""
    bank = np.asarray(bank, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    k = min(k, len(bank))
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        B = torch.from_numpy(bank).to(dev)
        bb = (B * B).sum(1)
        out = np.empty(len(test), dtype=np.float64)
        for i in range(0, len(test), chunk):
            T = torch.from_numpy(test[i:i + chunk]).to(dev)
            d2 = (T * T).sum(1)[:, None] + bb[None, :] - 2.0 * (T @ B.T)
            kth = torch.topk(d2, k, dim=1, largest=False).values[:, -1]
            out[i:i + chunk] = torch.sqrt(torch.clamp(kth, min=0.0)).cpu().numpy()
        return out
    except ImportError:
        bb = (bank * bank).sum(1)
        out = np.empty(len(test), dtype=np.float64)
        for i in range(0, len(test), chunk):
            T = test[i:i + chunk]
            d2 = (T * T).sum(1)[:, None] + bb[None, :] - 2.0 * (T @ bank.T)
            out[i:i + chunk] = np.sqrt(np.clip(np.partition(d2, k - 1, axis=1)[:, k - 1], 0, None))
        return out


def mahalanobis_min(test, bank, bank_dec, n_classes, shrink=SHRINK, chunk=65536):
    """Minimum over classes of the Mahalanobis distance to the class mean, shared shrunk covariance.

    Double precision throughout, as the first 23 tasks were scored; the quadratic form runs on the GPU in
    float64 when there is one, chunked, because m_sa_crop_type's 4,096,000 windows put the numpy version
    past both the memory and the time of a job. The test set is never copied whole to float64."""
    bank = np.asarray(bank, dtype=np.float64)
    d = bank.shape[1]
    means, resid = [], []
    for c in range(n_classes):
        sel = bank[bank_dec == c]
        if len(sel) == 0:
            continue
        mu = sel.mean(0)
        means.append(mu)
        resid.append(sel - mu)
    means = np.stack(means)
    R = np.concatenate(resid)
    cov = R.T @ R / max(len(R) - 1, 1)
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / d * np.eye(d)
    prec = np.linalg.inv(cov)
    out = np.full(len(test), np.inf)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        P = torch.from_numpy(prec).to(dev)
        M = torch.from_numpy(means).to(dev)
        for i in range(0, len(test), chunk):
            T = torch.from_numpy(np.asarray(test[i:i + chunk], dtype=np.float64)).to(dev)
            best = torch.full((len(T),), float("inf"), dtype=torch.float64, device=dev)
            for mu in M:
                diff = T - mu
                best = torch.minimum(best, ((diff @ P) * diff).sum(1))
            out[i:i + chunk] = best.cpu().numpy()
    except ImportError:
        for i in range(0, len(test), chunk):
            T = np.asarray(test[i:i + chunk], dtype=np.float64)
            for mu in means:
                diff = T - mu
                out[i:i + chunk] = np.minimum(out[i:i + chunk], np.einsum("ij,jk,ik->i", diff, prec, diff))
    return np.sqrt(np.clip(out, 0, None))


def controls_chunked(emb_test, emb_train, dec, chunk=262144):
    """exp70's two no-model controls with the same arithmetic in double precision, accumulated in chunks so a
    task of four million windows is never copied whole to float64; the only difference from exp70.controls is
    the order of summation."""
    n, d = emb_train.shape
    acc = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        acc += np.asarray(emb_train[i:i + chunk], dtype=np.float64).sum(0)
    mu = acc / n
    dist = np.empty(len(emb_test), dtype=np.float64)
    for i in range(0, len(emb_test), chunk):
        dist[i:i + chunk] = np.linalg.norm(np.asarray(emb_test[i:i + chunk], dtype=np.float64) - mu, axis=1)
    freq = collections.Counter(np.asarray(dec).tolist())
    rare = np.array([-np.log(max(freq[int(c)], 1) / max(len(dec), 1)) for c in dec])
    return {"ctl_embedding_distance": dist, "ctl_class_rarity": rare}


# ----------------------------------------------------------------------------- the two families
def _train_cls_probe(xtr, ytr, C, seed):
    """Ai2's linear recipe, exactly as exp70 runs it, returning a probability function."""
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54
    dev = e54.DEV
    torch.manual_seed(seed)
    probe = e51.LinearProbe(in_dim=xtr.shape[-1], out_dim=C).to(dev).float()
    opt = torch.optim.AdamW(probe.parameters(), lr=e51.LR)
    Xtr, Ytr = xtr.to(torch.float32), ytr.to(torch.int64)
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
                out.append(torch.softmax(probe(x[i:i + 4096].to(torch.float32).to(dev))["logits"], 1).cpu().numpy())
        return np.concatenate(out)
    return probs


def run_classification(task, cache):
    import torch
    import exp54_multiclass_embeddings as e54
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    if ytr.ndim != 1:
        raise ValueError(f"{task}: labels have shape {tuple(ytr.shape)}, not one label per sample")
    C = int(max(int(ytr.max()), int(yte.max())) + 1)
    prob_fns = [_train_cls_probe(xtr, ytr, C, seed) for seed in range(K_SEEDS)]
    p_te = [f(xte) for f in prob_fns]
    sig, dec = e70.readings_from_probs(p_te[0])
    y = yte.numpy()
    err = (dec != y).astype(np.float64)
    etr = xtr.to(torch.float32).numpy()
    ete = xte.to(torch.float32).numpy()
    sig.update(controls_chunked(ete, etr, dec))
    sig.update(ensemble_from_probs(p_te))
    rng = np.random.default_rng(0)
    bank = draw_bank(len(etr), rng)
    bank_dec = prob_fns[0](xtr[bank]).argmax(1)
    sig["knn_dist"] = knn_distance(ete, etr[bank])
    sig["mahalanobis"] = mahalanobis_min(ete, etr[bank], bank_dec, C)
    return {"family": "classification", "n_units": int(len(y)), "n_classes": C, "bank": int(len(bank)),
            "ensemble_primary": "ens_mi", "test_accuracy": float((dec == y).mean()),
            "error_rate": float(err.mean()), "signals": e70.score_task(sig, err, C)}


def run_segmentation(task, cache):
    import torch
    import exp54_multiclass_embeddings as e54
    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    C = int(max(int(ytr[ytr >= 0].max()), int(yte[yte >= 0].max())) + 1)
    pp = int(round(ytr.shape[-1] / xtr.shape[1]))
    lr = e54.TASK_LR.get(task, 0.1)
    qs, q_tr0 = [], None
    for seed in range(K_SEEDS):
        probe = e54.train_probe(xtr, ytr, pp, C, lr, seed=seed)
        qs.append(e54.window_quantities(probe, xte, yte, pp, C))
        if seed == 0:
            q_tr0 = e54.window_quantities(probe, xtr, ytr, pp, C)
    q0 = qs[0]
    ok = q0["ok"].ravel()
    err = q0["err"].ravel()[ok]
    dec = q0["dec"].ravel()[ok]
    sig = {"margin": -q0["margin"].ravel()[ok].astype(np.float64),
           "entropy": q0["entropy"].ravel()[ok].astype(np.float64)}
    etr = xtr.to(torch.float32).numpy().reshape(-1, xtr.shape[-1])
    ete = xte.to(torch.float32).numpy().reshape(-1, xte.shape[-1])[ok]
    sig.update(controls_chunked(ete, etr, dec))
    sig.update(ensemble_from_decisions([q["dec"].ravel()[ok] for q in qs],
                                       [q["margin"].ravel()[ok] for q in qs], C))
    ok_tr = q_tr0["ok"].ravel()
    etr_ok = etr[ok_tr]
    dec_tr = q_tr0["dec"].ravel()[ok_tr]
    rng = np.random.default_rng(0)
    bank = draw_bank(len(etr_ok), rng)
    sig["knn_dist"] = knn_distance(ete, etr_ok[bank])
    sig["mahalanobis"] = mahalanobis_min(ete, etr_ok[bank], dec_tr[bank], C)
    return {"family": "segmentation", "n_units": int(ok.sum()), "n_classes": C, "patch_px": pp,
            "bank": int(len(bank)), "ensemble_primary": "ens_vote_dis",
            "test_accuracy": float(1.0 - err.mean()), "error_rate": float(err.mean()),
            "signals": e70.score_task(sig, err, C)}


# ----------------------------------------------------------------------------- driver
def leads(r):
    """Excess AURC of each alternative minus the margin's: positive means the margin ranks better."""
    s = r["signals"]
    m = s["margin"]["excess_aurc"]
    out = {"ensemble": s[r["ensemble_primary"]]["excess_aurc"] - m,
           "knn_dist": s["knn_dist"]["excess_aurc"] - m,
           "mahalanobis": s["mahalanobis"]["excess_aurc"] - m}
    for k in ("ens_mi", "ens_vote_dis", "bag_margin", "ens_margin_std"):
        if k in s:
            out[k] = s[k]["excess_aurc"] - m
    ctl = min(s[c]["excess_aurc"] for c in e70.CONTROLS)
    out["beats_control"] = {a: bool(s[k]["excess_aurc"] < ctl)
                            for a, k in (("ensemble", r["ensemble_primary"]), ("knn_dist", "knn_dist"),
                                         ("mahalanobis", "mahalanobis"))}
    return out


def verdicts(results):
    tasks = sorted(results)
    v = {}
    for i, alt in enumerate(ALTS, start=1):
        wins = [t for t in tasks if results[t]["lead"][alt] > 0]
        losses = [t for t in tasks if results[t]["lead"][alt] < 0]
        p = float(stats.sign_test(len(wins), len(losses), alternative="greater"))
        by_src = collections.defaultdict(list)
        for t in tasks:
            by_src[results[t]["source"]].append(results[t]["lead"][alt])
        v[f"P{i}"] = {"alternative": alt, "holds": bool(len(wins) >= 18 and p < 0.05),
                      "wins": len(wins), "losses": len(losses), "of": len(tasks), "p": p,
                      "median_lead": float(np.median([results[t]["lead"][alt] for t in tasks])),
                      "source_wins": sum(1 for x in by_src.values() if float(np.mean(x)) > 0),
                      "distinct_sources": len(by_src),
                      "lost_on": sorted(losses)}
    v["P4"] = {"note": "descriptive: tasks on which the alternative beats the best no-model control",
               **{alt: sum(1 for t in tasks if results[t]["lead"]["beats_control"][alt]) for alt in ALTS},
               "of": len(tasks)}
    return v


PARTS = os.path.join(OUT, "exp73_parts")


def cmd_all(args):
    cache = os.environ.get("HF_HOME")
    os.makedirs(PARTS, exist_ok=True)
    results, rows, failures = {}, [], {}
    for task in e70.TASKS_CLS + e70.TASKS_SEG:
        if args.only and task not in args.only:
            continue
        # One task's result is written the moment it finishes and reused on a rerun: the PASTIS tasks take the
        # better part of an hour each, and a job cut at its time limit must not lose the tasks it did finish.
        part = os.path.join(PARTS, f"{task}.json")
        if os.path.exists(part) and not args.redo:
            with open(part) as fh:
                r = json.load(fh)
            print(f"  {task:48s} from checkpoint ({r['seconds']:.0f}s when run)", flush=True)
        else:
            fn = run_classification if task in e70.TASKS_CLS else run_segmentation
            t0 = time.time()
            try:
                r = fn(task, cache)
            except Exception as exc:
                failures[task] = f"{type(exc).__name__}: {str(exc)[:200]}"
                print(f"  {task:48s} FAILED {failures[task][:90]}", flush=True)
                continue
            r["lead"] = leads(r)
            r["source"] = e70.source_of(task)
            r["seconds"] = round(time.time() - t0, 1)
            with open(part, "w") as fh:
                json.dump(r, fh, default=float)
        results[task] = r
        s = r["signals"]
        rows.append({"task": task, "family": r["family"], "source": r["source"], "n_units": r["n_units"],
                     "n_classes": r["n_classes"], "bank": r["bank"], "test_accuracy": r["test_accuracy"],
                     "margin_eaurc": s["margin"]["excess_aurc"],
                     "ensemble_eaurc": s[r["ensemble_primary"]]["excess_aurc"],
                     "knn_eaurc": s["knn_dist"]["excess_aurc"], "mahalanobis_eaurc": s["mahalanobis"]["excess_aurc"],
                     "lead_ensemble": r["lead"]["ensemble"], "lead_knn": r["lead"]["knn_dist"],
                     "lead_mahalanobis": r["lead"]["mahalanobis"], "seconds": r["seconds"]})
        if os.path.exists(part) and not args.redo and "seconds" in r and r.get("_printed"):
            continue
        print(f"  {task:48s} {r['family'][:3]} n={r['n_units']:7d} acc {r['test_accuracy']:.3f} | margin lead over "
              f"ens {r['lead']['ensemble']:+.4f} knn {r['lead']['knn_dist']:+.4f} maha {r['lead']['mahalanobis']:+.4f} "
              f"({r['seconds']:.0f}s)", flush=True)
    summary = {"experiment": "exp73 the strong alternatives on Ai2's published suite",
               "config": {"model": e70.MODEL, "k_seeds": K_SEEDS, "knn_k": KNN_K, "bank": BANK, "shrink": SHRINK,
                          "budgets": list(e70.BUDGETS), "controls": list(e70.CONTROLS),
                          "caveats": ["the segmentation ensemble has decisions and margins but no probabilities, "
                                      "so its primary reading is vote disagreement over five seeds, which ties often; "
                                      "the scoring is tie-aware",
                                      "the feature-space signals see a seeded bank of at most 20,000 training units",
                                      "the Mahalanobis class means are fitted on predicted classes, not labels"]},
               "results": {"tasks": results, "failures": failures},
               "verdicts": verdicts(results) if results else {}}
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp73_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    if rows:
        with open(os.path.join(OUT, f"exp73_tasks{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    for k, x in summary["verdicts"].items():
        print(f"{k}: {x.get('holds', x.get('note', ''))}  "
              f"{json.dumps({a: b for a, b in x.items() if a not in ('holds', 'note', 'lost_on')})[:160]}", flush=True)
    if failures:
        print(f"failed: {sorted(failures)}", flush=True)
    return summary


def smoke(args):
    """The signals and the verdicts on synthetic data, torch-free."""
    rng = np.random.default_rng(0)
    n, C, d = 2000, 4, 16
    y = rng.integers(0, C, n)
    probs = [rng.dirichlet(np.ones(C) * 0.7, size=n) for _ in range(K_SEEDS)]
    ens = ensemble_from_probs(probs)
    assert set(ens) == {"ens_mi", "ens_vote_dis", "bag_margin"} and all(len(v) == n for v in ens.values())
    assert (ens["ens_mi"] >= -1e-9).all() and (0 <= ens["ens_vote_dis"]).all() and (ens["ens_vote_dis"] <= 1).all()
    same = ensemble_from_probs([probs[0]] * K_SEEDS)
    assert np.allclose(same["ens_mi"], 0) and np.allclose(same["ens_vote_dis"], 0), "identical members disagree on nothing"
    seg = ensemble_from_decisions([rng.integers(0, C, n) for _ in range(K_SEEDS)],
                                  [rng.random(n) for _ in range(K_SEEDS)], C)
    assert set(seg) == {"ens_vote_dis", "ens_margin_std"}
    bank = rng.standard_normal((300, d)); test = rng.standard_normal((50, d))
    kd = knn_distance(test, bank, k=5)
    assert kd.shape == (50,) and (kd > 0).all()
    assert knn_distance(bank[:5], bank, k=1).max() < 1e-4, "a bank row is its own nearest neighbour"
    md = mahalanobis_min(test, bank, rng.integers(0, C, 300), C)
    assert md.shape == (50,) and np.isfinite(md).all()
    far = mahalanobis_min(test + 50.0, bank, rng.integers(0, C, 300), C)
    assert (far > md).all(), "a point far from every class mean is farther"
    assert len(draw_bank(10, rng)) == 10 and len(draw_bank(BANK + 5, rng)) == BANK
    dec_s = rng.integers(0, C, 50)
    a, b = controls_chunked(test, bank, dec_s, chunk=7), e70.controls(test, bank, dec_s)
    assert np.allclose(a["ctl_embedding_distance"], b["ctl_embedding_distance"]) and np.array_equal(a["ctl_class_rarity"], b["ctl_class_rarity"])
    # the verdicts on planted leads: the margin wins 20 of 24 against each alternative
    fake = {}
    for i in range(24):
        lead = 0.03 if i < 20 else -0.01
        fake[f"t{i}"] = {"source": f"s{i % 14}", "lead": {"ensemble": lead, "knn_dist": lead, "mahalanobis": -lead,
                                                       "beats_control": {"ensemble": True, "knn_dist": i % 2 == 0,
                                                                         "mahalanobis": False}}}
    v = verdicts(fake)
    assert v["P1"]["holds"] is True and v["P1"]["wins"] == 20 and v["P1"]["distinct_sources"] == 14
    assert v["P3"]["holds"] is False and v["P3"]["wins"] == 4
    assert v["P4"]["ensemble"] == 24 and v["P4"]["knn_dist"] == 12 and v["P4"]["mahalanobis"] == 0
    print("smoke OK: ensemble readings, knn and mahalanobis, the bank, and the four verdicts on planted leads")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--redo", action="store_true", help="recompute tasks that have a checkpoint")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        smoke(args)
        return
    cmd_all(args)


if __name__ == "__main__":
    main()
