"""exp104, post hoc and report-only (not preregistered): where the Forest Loss Driver model's confidence ranks its
disagreements with MapBiomas's driver, and where it does not.

Read from the recorded units file (exp/out/exp104_units.npz, rows in a random order, no code, no position) after the
independent audit of exp104's grade asked for it: the design-weighted AUROC of the suspicion within each graded
stratum, and separately for the disagreements where the model picks a class MapBiomas Alerta does not record (road,
hurricane, none, burned, airstrip, ...) and for those between the two labelled classes (agriculture and mining), each
against all the correct alerts; and the top-1 probabilities of each group. Writes exp/out/exp104_posthoc.json.
"""
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics  # noqa: E402

UNITS = os.path.join(EXP_DIR, "out", "exp104_units.npz")
OUT = os.path.join(EXP_DIR, "out", "exp104_posthoc.json")


def posthoc(path=UNITS):
    u = np.load(path, allow_pickle=False)
    names = [str(c) for c in u["class_names"]]
    graded = u["graded"].astype(bool) & u["has_prediction"].astype(bool)
    err = u["error"] == 1
    w, s, top, pred, st = u["weight"], u["suspicion"], u["top1"], u["pred"], u["stratum"]
    labelled = np.isin(pred, [names.index("agriculture"), names.index("mining")])
    groups = {"all": graded & err, "to_a_class_mapbiomas_does_not_record": graded & err & ~labelled,
              "between_agriculture_and_mining": graded & err & labelled}
    correct = graded & ~err

    def auroc(m):
        keep = m | correct
        return float(metrics.weighted_auroc(s[keep], err[keep].astype(float), w[keep]))

    rec = {"reading": "post hoc, report-only (not preregistered); asked for by the audit of exp104's grade",
           "source": os.path.relpath(path, os.path.dirname(EXP_DIR)),
           "groups": {k: {"n": int(m.sum()), "weighted_auroc_against_correct": auroc(m),
                          "median_top1": float(np.median(top[m]))} for k, m in groups.items()},
           "correct": {"n": int(correct.sum()), "median_top1": float(np.median(top[correct]))},
           "by_stratum": {}}
    for k, name in enumerate(str(x) for x in u["stratum_names"]):
        m = graded & (st == k)
        if m.any():
            rec["by_stratum"][name] = {"n": int(m.sum()), "errors": int((m & err).sum()),
                                       "auroc": float(metrics.weighted_auroc(s[m], err[m].astype(float), w[m]))}
    return rec


if __name__ == "__main__":
    rec = posthoc()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
