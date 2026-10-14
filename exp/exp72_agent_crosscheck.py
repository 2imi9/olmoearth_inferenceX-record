"""exp72: does the agent skill compute the same thing as the experiments, on REAL logits?

Skill #18 of 2imi9/OlmoEarth-Agent (``olmoearth-review-set``) re-implements this repository's ranking
statistics in dependency-free pure Python so the agent can carry them without numpy or torch. A unit
test proved the port faithful on synthetic input. That is arithmetic, not evidence: a port can agree
on random matrices and still disagree where real inference lives, because real probability matrices
are not random. They are near-degenerate on easy samples (top-1 at 1 - 1e-7, so top1 minus top2 lands
in float noise), they carry exact ties where a probe saturates, and their error vectors are strongly
ordered rather than exchangeable, which is exactly where a tie-handling convention or a prefix-sum
order can diverge.

So this runs both implementations over the SAME real probability matrices: Ai2's published
olmoearth_base embeddings, their linear-probe recipe, their test splits, which is exp70's path
exactly. No encoder pass, no GPU, because the embeddings are published. For every task it compares,
between the numpy path (``oe_inferencex.metrics``, what every recorded number in this repository was
computed with) and the pure-Python path (the agent skill):

  the margin per sample, the arg-max decision per sample, excess AURC, capture at three budgets,
  and AUROC.

What this can and cannot show. It is a DIFFERENTIAL test, not a finding: agreement says the agent
reports what the experiments report, so the evidence in the skill's citations transfers to it. It
says nothing about whether the ranking itself is good; that is exp70's question and exp70 answered it.
Disagreement is a bug in the port, and the tolerance is tight enough to catch one: the two paths share
no code, only a specification.

Preregistered. P1 the decisions and margins agree to 1e-9 on every task, so the agent reads the same
model. P2 the three statistics agree to 1e-9 on every task and budget. P3 at least one task exhibits
exact ties in the margin, so the tie-handling convention is actually exercised rather than assumed;
if no task ties, P3 is recorded as untested rather than passed, because a tolerance met on untied
input proves nothing about ties.
"""
import argparse, hashlib, json, os, sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

import exp70_task_suite as e70                      # noqa: E402
from oe_inferencex import metrics                   # noqa: E402
import agent_review_set as agent                    # noqa: E402  the skill, copied in verbatim

OUT = os.path.join(EXP_DIR, "out")
TOL = 1e-9
BUDGETS = e70.BUDGETS


def compare_task(task, cache, seed=0):
    """Both implementations over one task's real probability matrix."""
    import torch
    import exp51_their_probe as e51
    import exp54_multiclass_embeddings as e54

    xtr, ytr = e54.load_theirs(e70.MODEL, task, "train", cache)
    xte, yte = e54.load_theirs(e70.MODEL, task, "test", cache)
    if ytr.ndim != 1:
        raise ValueError(f"{task}: labels have shape {tuple(ytr.shape)}, not one label per sample")
    C = int(max(int(ytr.max()), int(yte.max())) + 1)
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
    out = []
    with torch.no_grad():
        for i in range(0, len(xte), 4096):
            out.append(torch.softmax(probe(xte[i:i + 4096].to(torch.float32).to(dev))["logits"], 1).cpu().numpy())
    p_te = np.concatenate(out)                       # the real probability matrix, (N, C)

    # The probe emits float32. The numpy path subtracts in float32; the agent receives .tolist(), i.e. float64.
    # So compute the numpy side BOTH ways: native dtype (what every recorded number used) and promoted to float64.
    # The pair separates a dtype difference from a logic difference -- only the second would be a bug in the port.
    y = yte.numpy()

    def numpy_side(mat):
        srt = np.sort(mat, axis=1)
        m = srt[:, -1] - srt[:, -2]
        d = mat.argmax(1)
        e = (d != y).astype(np.float64)
        return m, d, e, -m                           # higher = more suspect, exp70's convention

    np_margin, np_dec, err, u_np = numpy_side(p_te)
    f64_margin, f64_dec, f64_err, u_f64 = numpy_side(p_te.astype(np.float64))

    # ---- pure-Python path: the agent skill, fed the identical matrix as lists
    rows = p_te.tolist()
    ag_margin = agent.margins(rows)
    ag_dec = agent.predicted_classes(rows)
    u_ag = [-m for m in ag_margin]
    err_ag = [float(d != int(t)) for d, t in zip(ag_dec, y.tolist())]

    d_margin = float(np.abs(np_margin - np.asarray(ag_margin)).max())
    d_margin_f64 = float(np.abs(f64_margin - np.asarray(ag_margin)).max())
    n_dec_diff = int((np_dec != np.asarray(ag_dec)).sum())

    def numpy_stats(u, e):
        return {"excess_aurc": float(metrics.excess_aurc(u, e)),
                "auroc": float(metrics.weighted_auroc(u, e, np.ones(len(e)))),
                "capture": {f"{b:.2f}": float(v) for b, v in
                            metrics.capture_at_budget_expected(u, e, BUDGETS).items()}}

    np_stats = numpy_stats(u_np, err)
    f64_stats = numpy_stats(u_f64, f64_err)
    ag_cap = agent.capture_at_budget(u_ag, err_ag, BUDGETS)   # takes the whole sequence, returns a dict
    ag_stats = {"excess_aurc": agent.excess_aurc(u_ag, err_ag),
                "auroc": agent.auroc(u_ag, err_ag),
                "capture": {f"{b:.2f}": float(ag_cap[b]) for b in BUDGETS}}

    def stat_deltas(ref):
        d = {"excess_aurc": abs(ref["excess_aurc"] - ag_stats["excess_aurc"]),
             "auroc": abs(ref["auroc"] - (ag_stats["auroc"] if ag_stats["auroc"] is not None else np.nan))}
        for b in BUDGETS:
            k = f"{b:.2f}"
            d[f"capture@{k}"] = abs(ref["capture"][k] - ag_stats["capture"][k])
        return d

    deltas = stat_deltas(np_stats)
    deltas_f64 = stat_deltas(f64_stats)

    n_tied = int(len(np_margin) - len(set(np.round(np_margin, 15).tolist())))
    return {"task": task, "n": int(len(y)), "n_classes": C,
            "accuracy": float((np_dec == y).mean()), "error_rate": float(err.mean()),
            "margin_min": float(np_margin.min()), "margin_max": float(np_margin.max()),
            "n_exact_margin_ties": n_tied,
            "max_abs_margin_delta": d_margin, "max_abs_margin_delta_f64": d_margin_f64,
            "n_decision_disagreements": n_dec_diff,
            "probs_dtype": str(p_te.dtype),
            "numpy": np_stats, "numpy_float64": f64_stats, "agent": ag_stats,
            "max_stat_delta": float(max(deltas.values())), "deltas": deltas,
            "max_stat_delta_f64": float(max(deltas_f64.values())), "deltas_f64": deltas_f64}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(e70.TASKS_CLS))
    ap.add_argument("--cache", default=os.environ.get("HF_HOME"),
                    help="hf_hub_download cache_dir holding allenai/olmoearth-paper-embeddings")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    # Pin exactly which build of the skill was tested; a later edit invalidates this run.
    skill_sha = hashlib.sha256(open(agent.__file__, "rb").read()).hexdigest()
    print(f"skill under test: {agent.__file__}\n  sha256 {skill_sha}\n", flush=True)

    rows, failures = [], {}
    for task in [t for t in args.tasks.split(",") if t]:
        if task in e70.EXCLUDED:
            failures[task] = f"excluded by exp70: {e70.EXCLUDED[task]}"
            print(f"  {task:<52} EXCLUDED", flush=True)
            continue
        try:
            r = compare_task(task, args.cache)
        except Exception as exc:
            failures[task] = f"{type(exc).__name__}: {exc}"
            print(f"  {task:<52} FAILED {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            continue
        rows.append(r)
        print(f"  {task:<52} n={r['n']:<7} | native: margin {r['max_abs_margin_delta']:.1e} "
              f"stat {r['max_stat_delta']:.1e} | float64: margin {r['max_abs_margin_delta_f64']:.1e} "
              f"stat {r['max_stat_delta_f64']:.1e} | ties={r['n_exact_margin_ties']}", flush=True)

    p1 = bool(rows) and all(r["max_abs_margin_delta_f64"] <= TOL and r["n_decision_disagreements"] == 0
                            for r in rows)
    p2 = bool(rows) and all(r["max_stat_delta_f64"] <= TOL for r in rows)
    p1_native = bool(rows) and all(r["max_abs_margin_delta"] <= TOL for r in rows)
    p2_native = bool(rows) and all(r["max_stat_delta"] <= TOL for r in rows)
    worst_native = max((r["max_stat_delta"] for r in rows), default=float("nan"))
    any_ties = any(r["n_exact_margin_ties"] > 0 for r in rows)
    p3 = "holds" if any_ties else "untested (no task produced an exact margin tie)"

    summary = {"experiment": "exp72 agent-skill cross-check on real logits",
               "what": "oe_inferencex.metrics (numpy) against OlmoEarth-Agent skill #18 (pure Python), "
                       "over identical real probability matrices from Ai2's published embeddings",
               "skill_sha256": skill_sha, "skill_source":
                   "2imi9/OlmoEarth-Agent src/olmoearth_agent/analysis/review_set.py, copied verbatim",
               "tolerance": TOL, "budgets": list(BUDGETS), "n_tasks": len(rows),
               "results": rows, "failures": failures,
               "verdicts": {"P1_decisions_and_margins_agree": p1,
                            "P2_statistics_agree": p2,
                            "P3_ties_exercised": p3},
               "dtype_note": {
                   "at_the_probe_dtype_float32": {"P1": p1_native, "P2": p2_native,
                                                  "worst_stat_delta": worst_native},
                   "reading": "P1/P2 are judged against the float64 numpy path. The probe emits float32, so the "
                              "native numpy path subtracts in float32 while the agent, receiving .tolist(), works "
                              "in float64. Any gap that closes when numpy is promoted is a precision difference, "
                              "not a difference in what the two implementations compute."}}
    with open(os.path.join(OUT, "exp72_agent_crosscheck.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"\nP1 decisions+margins agree (<= {TOL:g}): {p1}")
    print(f"P2 statistics agree (<= {TOL:g}): {p2}")
    print(f"  at the probe's native float32, the same tests read P1={p1_native} P2={p2_native} "
          f"(worst stat delta {worst_native:.2e}) -- precision, not logic")
    print(f"P3 tie handling exercised: {p3}")
    if failures:
        print(f"failures: {json.dumps(failures, indent=1)[:800]}")


if __name__ == "__main__":
    main()
