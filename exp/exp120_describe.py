#!/usr/bin/env python
"""exp120: descriptive counts beside the predictions, read from the run's cells (no prediction rests on them).

    python exp/exp120_describe.py --units FILE --tile FILE --out FILE

How many cells sit below the nominal levels the bars allow noise around (coverage 0.95, a wrong verdict 0.05, an
interval miss 0.10), and the interval given without --valid by design, which no prediction grades. Standard library.
"""
import argparse
import json
import statistics


def main():
    ap = argparse.ArgumentParser()
    for name in ("--units", "--tile", "--out"):
        ap.add_argument(name, required=True)
    a = ap.parse_args()
    A, B = json.load(open(a.units)), json.load(open(a.tile))
    av = [c for m in A["maps"] for c in m["valid"] if c["draws"]]
    ap_ = [c for m in A["maps"] for c in m["profiles"] if c["draws"]]
    out = {"a_valid_cells": len(av), "a_profile_cells": len(ap_)}
    out["a_valid_cells_below_0.95"] = sum(c["valid_covers"] / c["draws"] < 0.95 for c in av)
    out["a_profile_cells_wrong_above_0.05"] = sum(c["any_wrong"] / c["draws"] > 0.05 for c in ap_)
    out["a_profile_cells_miss_above_0.10"] = sum(c["any_miss"] / c["draws"] > 0.10 for c in ap_)
    out["a_profile_mean_miss"] = statistics.fmean(c["any_miss"] / c["draws"] for c in ap_)
    out["a_profile_mean_wrong"] = statistics.fmean(c["any_wrong"] / c["draws"] for c in ap_)
    out["a_valid_pooled_coverage"] = sum(c["valid_covers"] for c in av) / sum(c["draws"] for c in av)
    out["a_valid_not_computed"] = sum(c["valid_not_computed"] for c in av)
    plain = {}
    for c in av:
        k = f"{c['design']}_{c['n']}"
        if c["draws"] > c["plain_none"]:
            plain.setdefault(k, []).append(c["plain_covers"] / (c["draws"] - c["plain_none"]))
    out["a_plain_by_design"] = {k: {"cells": len(v), "min": min(v), "below_0.917": sum(x < 0.917 for x in v),
                                    "below_0.95": sum(x < 0.95 for x in v)} for k, v in sorted(plain.items())}
    out["b_valid"] = [{"design": c["design"], "n": c["n"], "valid": c["valid_covers"] / c["draws"],
                       "plain": c["plain_covers"] / c["draws"]} for c in B["valid"]]
    out["b_profiles"] = [{"spec": c["spec"], "alpha": c["alpha"], "wrong": c["any_wrong"] / c["draws"],
                          "miss": c["any_miss"] / c["draws"], "decided_share_of_tested":
                          c["decided_tested"] / max(1, c["tested"])} for c in B["profiles"]]
    refusals = {}
    for m in A["maps"]:
        for c in m["valid"] + m["profiles"]:
            for r in c["refusals"]:
                key = r.split("(")[0][:90]
                refusals[key] = refusals.get(key, 0) + 1
    out["a_refusal_kinds"] = refusals
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
