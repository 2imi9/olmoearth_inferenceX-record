"""exp88 follow-up: is the confident-error effect the missing input, or a head read on an input it was not trained on?

exp88 trained a probe on the S1+S2 embeddings and read it, with no refit, on the S1-only and S2-only embeddings of the
same test windows. This script puts beside each of those readings the probe the record already TRAINED on that input
(exp70's probes, exported per unit by exp78 for OlmoEarth Base and by exp79 for the other encoders), on the same
windows in the same order. No probe is fitted here; every number is read from committed per-unit files.

    python exp/exp88_matched_head.py            # writes exp/out/exp88_matched_head.json

Inputs, and how they line up.
  exp/out/exp88_units.npz      exp88's per-unit margins and errors at probe seed 0 (the graded seed), its labels, its
                               tiles and the half-cloudy map's cloudy tiles.
  exp/out/exp78_units/*.npz    OlmoEarth Base, exp70's probe per task at seed 0 (committed). On PASTIS its S1+S2 file is
                               exp88's full condition unit for unit (margins and errors equal), its tiles equal exp88's,
                               and it stores no label (y is zero); each file's decision equals exp88's label exactly
                               where the file records no error and differs from it exactly where it records one, which
                               pins the unit order. China 6 stores the label, equal to exp88's.
  exp/out/exp79_units/<enc>/   OlmoEarth Large, the same probes from exp79's seed-0 export. The directory is not
                               committed (about 45 MB per encoder); each file is used only when its sha256 equals the
                               one exp/out/exp79_seeds/<enc>.json committed, and is reported absent otherwise. Large's
                               S1+S2 export is a second run of the probe exp88 fitted and differs from exp88's full
                               condition on a few units (counted below), so Large's full-input row is exp88's.

Per head and input (PASTIS and CropHarvest China 6; OlmoEarth Base, and Large when its files verify):
  s1s2_head_on_s1s2   exp88's full condition
  s1s2_head_on_s1     exp88's optical-missing condition (the head was not trained on this input)
  s1_head_on_s1       the probe trained on the S1 embeddings (exp70's pastis_sentinel1 / China 6 sentinel1 task)
  s1s2_head_on_s2     exp88's radar-missing condition
  s2_head_on_s2       the probe trained on the S2 embeddings
each with the error rate, the confident share of its errors at exp88's threshold (the median margin of the windows
the S1+S2 head gets right on S1+S2, per encoder and family), the margin's AUROC for errors and its excess AURC (the
package's metrics.weighted_auroc and metrics.excess_aurc on -margin), the share of the most frequent decision, and
the probe learning rate.

The half-cloudy PASTIS map, exp88's measure 5a with its cloudy tiles: the cloudy tiles take the optical-missing
units of either the S1+S2 head (exp88's map, "mismatched") or the S1 head ("matched"); the rest keep the S1+S2 head
on full input. Per part: the error rate and the share of that part's errors inside the 5% review set by margin
(exp88.review_shares, ties broken at random in expectation); for the whole map: the share of all errors inside it.

Checks written into the output: the recomputed exp88 rows agree with exp/out/exp88_summary.json, and the matched map's
arithmetic reproduces exp88's recorded mixed map when fed exp88's own units.
"""
import ast
import hashlib
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (os.path.join(ROOT, "scripts"), ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from oe_inferencex import metrics                      # noqa: E402
import exp78_error_rate_estimation as e78              # noqa: E402
import exp88_missing_modality as e88                   # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
UNITS88 = os.path.join(OUT, "exp88_units.npz")
SUMMARY88 = os.path.join(OUT, "exp88_summary.json")
EXP79_UNITS = os.path.join(OUT, "exp79_units")
SEED = "0"
TASKS = {"pastis": {"s1s2": "pastis_sentinel1_sentinel2", "s1": "pastis_sentinel1", "s2": "pastis_sentinel2"},
         "china6": {"s1s2": "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2",
                    "s1": "cropharvest_Peoples_Republic_of_China_6_sentinel1",
                    "s2": "cropharvest_Peoples_Republic_of_China_6"}}
EXP88_ROWS = {"s1s2_head_on_s1s2": "full", "s1s2_head_on_s1": "optical_missing", "s1s2_head_on_s2": "radar_missing"}
MATCHED_ROWS = {"s1_head_on_s1": "s1", "s2_head_on_s2": "s2"}
ROW_ORDER = ("s1s2_head_on_s1s2", "s1s2_head_on_s1", "s1_head_on_s1", "s1s2_head_on_s2", "s2_head_on_s2")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exports(enc):
    """{task: (path, provenance)} for the encoder's per-unit exports of the six tasks, or (None, reason)."""
    out = {}
    tasks = [t for fam in TASKS.values() for t in fam.values()]
    if enc == "olmoearth_base":
        for t in tasks:
            p = os.path.join(OUT, "exp78_units", f"{t}.npz")
            out[t] = (p, {"path": os.path.relpath(p, ROOT), "sha256": _sha256(p), "committed": True})
        return out
    rec_path = os.path.join(OUT, "exp79_seeds", f"{enc}.json")
    rec = json.load(open(rec_path))["tasks"]
    for t in tasks:
        p = os.path.join(EXP79_UNITS, enc, f"{t}.npz")
        want = rec.get(t, {}).get("units", {}).get("sha256")
        if not os.path.exists(p):
            out[t] = (None, {"path": os.path.relpath(p, ROOT), "absent": "file not present (exp79_units is not committed)"})
            continue
        got = _sha256(p)
        if got != want:
            out[t] = (None, {"path": os.path.relpath(p, ROOT), "absent": f"sha256 {got} differs from the committed {want}"})
            continue
        out[t] = (p, {"path": os.path.relpath(p, ROOT), "sha256": got, "committed": False,
                      "sha256_matches": os.path.relpath(rec_path, ROOT)})
    return out


def exp54_task_lr():
    """exp54.TASK_LR read from its source, since importing exp54 needs torch."""
    tree = ast.parse(open(os.path.join(EXP_DIR, "exp54_multiclass_embeddings.py"), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TASK_LR" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("exp54.TASK_LR not found")


def probe_lr(enc, task):
    if enc == "olmoearth_base":
        # exp78's export and exp70 train segmentation probes at exp54.TASK_LR.get(task, 0.1); classification at 0.1
        return float(exp54_task_lr().get(task, 0.1)) if task.startswith("pastis") else 0.1
    return float(json.load(open(os.path.join(OUT, "exp79_seeds", f"{enc}.json")))["probe_lrs"].get(task, float("nan")))


def row(margin, err, dec, thr):
    m = np.asarray(margin, dtype=np.float64)
    e = np.asarray(err) > 0.5
    d = np.asarray(dec).astype(np.int64)
    counts = np.bincount(d)
    return {"n_units": int(e.size), "n_errors": int(e.sum()), "error_rate": float(e.mean()),
            "n_confident_errors": int((m[e] >= thr).sum()), "confident_share": float((m[e] >= thr).mean()),
            "margin_auroc": float(metrics.weighted_auroc(-m, e.astype(np.float64), np.ones(e.size))),
            "margin_excess_aurc": float(metrics.excess_aurc(-m, e.astype(np.float64))),
            "top_decision": int(counts.argmax()), "top_decision_share": float(counts.max() / d.size)}


def family(enc, fam, z, files, rec88):
    B = f"{enc}/{fam}/"
    y = z[B + "y"].astype(np.int64)
    tile = z[B + "tile"] if B + "tile" in z.files else None
    base = B + f"seed{SEED}/"
    full_err = z[base + "full/err"] > 0
    thr = float(np.median(z[base + "full/margin"][~full_err].astype(np.float64)))
    rows, units, prov, align = {}, {}, {}, {}
    for name, cond in EXP88_ROWS.items():
        units[name] = {k: z[base + f"{cond}/{k}"] for k in ("margin", "err", "dec")}
        prov[name] = {"source": f"exp/out/exp88_units.npz {base}{cond}", "probe_lr": probe_lr(enc, TASKS[fam]["s1s2"])}
    for name, key in MATCHED_ROWS.items():
        task = TASKS[fam][key]
        path, p = files[task]
        prov[name] = {"source": p, "probe_lr": probe_lr(enc, task)}
        if path is None:
            continue
        d = np.load(path)
        if d["err"].size != y.size:
            raise RuntimeError(f"{enc} {task}: {d['err'].size} units against exp88's {y.size}")
        dec, err = d["dec"].astype(np.int64), d["err"] > 0
        # the unit order: the export's decision is exp88's label exactly where the export records no error
        consistent = bool(np.array_equal(dec == y, ~err))
        if not consistent:
            raise RuntimeError(f"{enc} {task}: decisions and errors do not line up with exp88's labels")
        align[task] = {"decision_equals_exp88_label_exactly_where_correct": consistent}
        if tile is not None and d["grid"].size == 3:
            t78 = e78.load_units(task, os.path.dirname(path))["tile"]
            if not np.array_equal(t78, tile):
                raise RuntimeError(f"{enc} {task}: tiles differ from exp88's")
            align[task]["tiles_equal_exp88"] = True
        units[name] = {"margin": d["margin"], "err": d["err"], "dec": dec}
    # the same-input S1+S2 export beside exp88's full condition: identical for Base, a second run for Large
    path, p = files[TASKS[fam]["s1s2"]]
    if path is not None:
        d = np.load(path)
        align[TASKS[fam]["s1s2"]] = {
            "units_whose_error_differs_from_exp88_full": int((d["err"].astype(np.uint8) != z[base + "full/err"]).sum()),
            "max_abs_margin_difference": float(np.abs(d["margin"].astype(np.float64) - z[base + "full/margin"]).max()),
            "source": p}
    for name in ROW_ORDER:
        if name in units:
            u = units[name]
            rows[name] = {**row(u["margin"], u["err"], u["dec"], thr), **prov[name]}
        else:
            rows[name] = {"absent": prov[name]["source"].get("absent", "no export"), **prov[name]}
    # agreement with what exp88 recorded, on its own three rows
    cond = rec88["conditions"]
    agree = {}
    for name, c in EXP88_ROWS.items():
        r, rc = rows[name], cond[c]
        agree[name] = {"error_rate": abs(r["error_rate"] - rc["error_rate"]),
                       "confident_share": abs(r["confident_share"] - rc["confident_errors"]["share"]),
                       "margin_auroc": abs(r["margin_auroc"] - rc["ranking"]["margin_auroc"]),
                       "margin_excess_aurc": abs(r["margin_excess_aurc"] - rc["ranking"]["margin_excess_aurc"])}
    out = {"n_units": int(y.size), "threshold_confident": thr,
           "threshold_recorded_by_exp88": rec88["threshold_confident"],
           "label_share_of_top_decision": {n: float((y == rows[n]["top_decision"]).mean()) for n in rows
                                          if "top_decision" in rows[n]},
           "rows": rows, "alignment": align,
           "max_abs_difference_from_exp88_summary": {k: max(v[k] for v in agree.values())
                                                     for k in next(iter(agree.values()))}}
    return out, units, tile


def mixed(units, tile, cloudy_tiles, cloud_row):
    cloudy = np.isin(tile, cloudy_tiles)
    m = np.where(cloudy, units[cloud_row]["margin"], units["s1s2_head_on_s1s2"]["margin"]).astype(np.float64)
    e = np.where(cloudy, units[cloud_row]["err"], units["s1s2_head_on_s1s2"]["err"]) > 0
    rs = e88.review_shares(m, e.astype(np.float64), cloudy)
    return {"cloudy_part": cloud_row, "clear_part": "s1s2_head_on_s1s2",
            "n_windows": int(e.size), "n_cloudy_windows": int(cloudy.sum()),
            "n_errors": {"cloudy": int((e & cloudy).sum()), "clear": int((e & ~cloudy).sum()), "whole": int(e.sum())},
            "error_rate": {"cloudy": float(e[cloudy].mean()), "clear": float(e[~cloudy].mean()), "whole": float(e.mean())},
            "review_set": rs}


def main():
    z = np.load(UNITS88)
    s88 = json.load(open(SUMMARY88))
    out = {"experiment": "exp88 matched-head follow-up: each input read by the head trained on it, beside exp88's head",
           "probe_seed": int(SEED), "rows": list(ROW_ORDER),
           "threshold_rule": "the median margin of the windows the S1+S2 head gets right on S1+S2 (exp88's)",
           "review_budget": e88.REVIEW_BUDGET, "encoders": {}, "not_run": {}}
    for enc in ("olmoearth_base", "olmoearth_large"):
        files = exports(enc)
        rec = s88["results"][enc]["seeds"][SEED]
        E = {"families": {}}
        for fam in TASKS:
            res, units, tile = family(enc, fam, z, files, rec["families"][fam])
            E["families"][fam] = res
            if fam == "pastis" and "s1_head_on_s1" in units:
                ct = z[f"{enc}/pastis/cloudy_tiles"]
                E["mixed_map"] = {"n_cloudy_tiles": int(ct.size), "mix_seed": e88.MIX_SEED,
                                  "mismatched": mixed(units, tile, ct, "s1s2_head_on_s1"),
                                  "matched": mixed(units, tile, ct, "s1_head_on_s1")}
                recm = rec.get("mixed_map")
                if recm:
                    mm = E["mixed_map"]["mismatched"]
                    E["mixed_map"]["mismatched_agrees_with_exp88_summary"] = bool(
                        abs(mm["error_rate"]["cloudy"] - recm["error_rate"]["cloudy"]) < 1e-12
                        and abs(mm["error_rate"]["clear"] - recm["error_rate"]["clear"]) < 1e-12
                        and all(abs(mm["review_set"][k] - recm["review_set"][k]) < 1e-12
                                for k in ("cloudy_errors_in_review_share", "clear_errors_in_review_share")))
            elif fam == "pastis":
                out["not_run"][f"{enc}/mixed_map"] = "the S1 head's export is absent"
        out["encoders"][enc] = E
    path = os.path.join(OUT, "exp88_matched_head.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for enc, E in out["encoders"].items():
        for fam, F in E["families"].items():
            print(f"{enc} {fam} (threshold {F['threshold_confident']:.4f})")
            for n in ROW_ORDER:
                r = F["rows"][n]
                if "error_rate" in r:
                    print(f"  {n:20s} err {r['error_rate']:.3f}  confident {r['confident_share']:.3f}  "
                          f"AUROC {r['margin_auroc']:.3f}  E-AURC {r['margin_excess_aurc']:.4f}  "
                          f"top decision {r['top_decision_share']:.3f}  lr {r['probe_lr']}")
                else:
                    print(f"  {n:20s} absent: {r['absent']}")
        for k in ("mismatched", "matched"):
            if "mixed_map" in E:
                m = E["mixed_map"][k]
                print(f"  mixed {k:10s} err cloudy {m['error_rate']['cloudy']:.3f} clear {m['error_rate']['clear']:.3f}; "
                      f"5% review holds cloudy {m['review_set']['cloudy_errors_in_review_share']:.3f} "
                      f"clear {m['review_set']['clear_errors_in_review_share']:.3f} "
                      f"all {m['review_set']['all_errors_in_review_share']:.3f}")
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
