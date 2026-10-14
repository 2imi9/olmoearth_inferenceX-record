"""exp100, descriptive follow-up (not preregistered, written after the run): where does V's eastward best position come
from? The frozen run (docs/plan/alignment.md) called exp98's validation points (V) an offset, about 1.9 px east, while
its training points (T) showed no east trend. This reads the same points the same way (exp100_alignment's exp98_sets)
and reports, per annotation task: its points, the integer best position of its own mean score surface, and the
agreement at the labelled pixel and one and two pixels east; and, for V, the estimate with each task left out in turn.
Tasks are numbered in sorted order of their names; no name, position or identifier is written.

    python exp/exp100_by_task.py --smoke
    python exp/exp100_by_task.py --exp98-scores DIR [DIR ...] --labels WINDOWS

Writes exp/out/exp100_by_task.json (exp100_by_task_smoke.json with --smoke).
"""
import argparse
import json
import os
import sys
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)
import exp100_alignment as e100   # noqa: E402
import exp89_finetuned_checkpoints as e89   # noqa: E402
import exp98_awf_deployment as e98   # noqa: E402

S = e100.SHIFT


def by_task(blocks, accept, tasks):
    """Per task (numbered in sorted order of the names), and the leave-one-task-out estimates, from one set."""
    score, agree, full, _ = e100.surfaces(blocks, accept)
    F = np.flatnonzero(full)
    score, agree, tasks = score[F], agree[F], np.asarray(tasks)[F]
    names = sorted(set(tasks.tolist()))
    rows, loo = [], []
    for i, t in enumerate(names):
        m = tasks == t
        p = e100.peak(score[m].mean(0))
        A = agree[m].mean(0)
        rows.append({"task": i, "n": int(m.sum()), "errors_at_labelled_pixel": int(m.sum() - agree[m][:, S, S].sum()),
                     "best_south_px": p["best_south_px"], "best_east_px": p["best_east_px"],
                     "agreement_at_labelled_pixel": float(A[S, S]), "agreement_1px_east": float(A[S, S + 1]),
                     "agreement_2px_east": float(A[S, S + 2])})
        if (~m).any():                            # a set with one task has nothing left without it
            q = e100.peak(score[~m].mean(0))
            loo.append({"left_out": i, "n": int((~m).sum()), "south_px": q["south_px"], "east_px": q["east_px"]})
    if not F.size:
        return {"n": 0, "n_tasks": 0, "note": "no point with a covered 7 x 7 block"}
    full_peak = e100.peak(score.mean(0))
    east = [r["east_px"] for r in loo]
    return {"n": int(F.size), "n_tasks": len(names), "estimate_all": full_peak, "tasks": rows,
            "leave_one_task_out": loo,
            "leave_one_task_out_east_px_range": [float(min(east)), float(max(east))] if east else None,
            "tasks_best_east_of_pixel": int(sum(r["best_east_px"] > 0 for r in rows)),
            "tasks_best_west_of_pixel": int(sum(r["best_east_px"] < 0 for r in rows)),
            "points_in_tasks_best_east": int(sum(r["n"] for r in rows if r["best_east_px"] > 0))}


def analyse(scores, labels):
    sets, mapping = e100.exp98_sets(scores, labels)
    commit, dirty = e89.git_state()
    rec = {"experiment": "exp100 follow-up (descriptive, written after the run): V's and T's best positions per "
                         "annotation task, and V's estimate with each task left out",
           "not_preregistered": True, "utc": e89.utc_now(), "commit": commit, "dirty": dirty, "mapping": mapping}
    for name in ("V", "T"):
        blocks, accept, tasks, _ = sets[name]
        if e100.RECORDED.get(name) and len(accept) != e100.RECORDED[name]["n"]:
            raise ValueError(f"{name}: {len(accept)} points, the record has {e100.RECORDED[name]['n']}")
        rec[name] = by_task(blocks, accept, tasks)
    return rec


def smoke(out_dir=None, log=print):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "e98"))
        p = e98.make_smoke_inputs(os.path.join(tmp, "e98"))
        saved = dict(e100.RECORDED)
        e100.RECORDED.clear()                     # the smoke's counts are not the record's
        try:
            rec = analyse(p["scores"], p["labels"])
        finally:
            e100.RECORDED.update(saved)
    rec["mode"] = "smoke: exp98's synthetic inputs"
    out_dir = out_dir or e100.OUT
    os.makedirs(out_dir, exist_ok=True)
    e98.dump(rec, os.path.join(out_dir, "exp100_by_task_smoke.json"))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--exp98-scores", nargs="+")
    ap.add_argument("--labels")
    ap.add_argument("--out-dir", default=e100.OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir)
        print(json.dumps({k: rec[k]["n_tasks"] for k in ("V", "T")}))
        return 0
    if not (a.exp98_scores and a.labels):
        ap.error("--exp98-scores and --labels are required outside --smoke")
    rec = analyse(a.exp98_scores, a.labels)
    os.makedirs(a.out_dir, exist_ok=True)
    e98.dump(rec, os.path.join(a.out_dir, "exp100_by_task.json"))
    print(json.dumps({"V": rec["V"]["leave_one_task_out_east_px_range"], "V_tasks_east": rec["V"]["tasks_best_east_of_pixel"],
                      "V_tasks_west": rec["V"]["tasks_best_west_of_pixel"]}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
