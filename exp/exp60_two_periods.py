#!/usr/bin/env python
"""exp60: two periods, both sensors. Does the time-only difference carry the flood and the sensor-only difference the error?

Why. The comparison the README shows (exp57's GEOID pair, the pre-event Sentinel-2 head against the post-event Sentinel-1
head) mixes two axes: time (the flood) and sensor (optics against radar). Its difference was 27% flood and the rest error,
and no label-free reading could split the two well (exp58, exp59). GEOID-Flood holds three of the four cells of the
two-period, two-sensor design: Sentinel-1 before and after the event, Sentinel-2 before (no post-event optical pass; the
scene is under cloud during a flood). Three cells separate the axes: the time-only pair (S1 before against S1 after,
one head) should hold the flood; the sensor-only pair (S2 before against S1 before, both permanent water, same period)
holds no flood by construction, only sensor error; the mixed pair (exp57's) holds both.

Design. exp55's chips (64 px at stride 256, the per-AoI cap of 400, seed 0) on the tiles of the first four test shards
and the first two val shards that carry all five layers: the S2 composite, S1 pre, S1 post, label, cloud mask. The
pre-event S1 pass is extracted from the same s1grd shards exp55 fetched (exp55 kept the post pass only). Heads, fitted
on the val chips as in exp55: the permanent-water head on the S2 composite (clear chips, task A), and ONE Sentinel-1
water head fitted on the val chips' pre-event pass with the permanent-water label and their post-event pass with the
water-after label, stacked, then applied to both dates of the test chips. Inferences on the clear test chips, four crop
offsets averaged (W1, windows 1..14): A_s2pre, A_s1pre (permanent water), B_s1post (water after the event). Pairs
through oe_inferencex.compare on identical windows: time-only (A_s1pre, B_s1post), sensor-only (A_s2pre, A_s1pre), mixed
(A_s2pre, B_s1post). Per pair: the disagreement rate pooled and per event, the share of the differing windows flooded by
the label (flooded after, not permanent), each side's error share on its own task, the boundary enrichment, the
both-confident share (margin >= 0.25 each) and its flooded share; per-event tests with compare.over_groups on the events
with at least MIN_EVENT_WINDOWS differing windows in both pairs of a test. Caveat: the two S1 passes may come from
different orbits and incidence angles, so "time only" is time plus acquisition geometry.

Preregistered (one-sided):
  P1  the time-only pair's differing windows are flooded by the label at least TWICE as often as the mixed pair's, pooled
      over the events, and more often on more events than not (p < 0.05).
  P2  the sensor-only pair's differing windows are flooded by the label at most HALF as often as the mixed pair's, pooled,
      and less often on more events than not (p < 0.05).
  Falsification: P1 fails if the ratio is below 2 or the event test misses (removing the sensor axis would not concentrate
  the flood in the difference); P2 fails if the ratio is above 0.5 or the event test misses (a same-period cross-sensor
  difference would carry the later flood as much as the mixed pair, so the sensor axis could not be isolated). Stated
  predictions, not tested: the sensor-only pair has fewer differing windows than the mixed pair on a majority of events;
  the both-confident reading is flooded on the time-only pair more often than on the mixed pair (exp58's 52%).

Inputs. $HF_HOME/geoid_flood (exp55's tree; the pre pass is added), the OlmoEarth v1 Base checkpoint; the pinned
~/olmoearth_inferenceX/.venv (rasterio). Outputs. exp/out/exp60_summary.json (per pair pooled and per event, prereg),
exp/out/exp60_two_periods.csv (one row per pair). Smoke: the sample tree in the local Hub cache; the pre-event S1 sample
files are downloaded when the network allows, else the post pass stands in (recorded in the summary).
"""
import csv
import json
import os
import sys
import tarfile
import time

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp55_geoid_flood as e55  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import compare_inferences, over_groups  # noqa: E402
from oe_inferencex.explain import cue_enrichment  # noqa: E402
from oe_inferencex.signals import boundary_indicator  # noqa: E402

SHIFTS, DEV, SL = e55.SHIFTS, e55.DEV, e57.SL
LAYERS = ("s2l2a", "s1grd_pre", "s1grd_post", "label", "cloudmask")
MIN_EVENT_WINDOWS, CONFIDENT_MARGIN, CAP = 20, 0.25, 400
P1_RATIO, P2_RATIO = 2.0, 0.5
PAIRS = {"time_only": ("A_s1pre", "B_s1post"), "sensor_only": ("A_s2pre", "A_s1pre"), "mixed": ("A_s2pre", "B_s1post")}
TASK_OF = {"A_s2pre": "A", "A_s1pre": "A", "B_s1post": "B"}
fmt = e57.fmt


# ----------------------------------------------------------------------------- the tree with both S1 passes
def wanted(name):
    d = e55.parse_name(name)
    return d is not None and d["layer"] in e55.LAYERS and (d["layer"] == "s1grd" or e55.PASS[d["layer"]] is None or d["pas"] == e55.PASS[d["layer"]])


def tile_table(paths):
    """{file name: path} -> {tile: {"aoi", s2l2a, s1grd_pre, s1grd_post, label, cloudmask}} for tiles with all five; s1grd = the post
    pass, so exp55's readers work unchanged."""
    tiles = {}
    for name in sorted(paths):
        if not wanted(name):
            continue
        d = e55.parse_name(name)
        key = f"s1grd_{d['pas']}" if d["layer"] == "s1grd" else d["layer"]
        tiles.setdefault(f"{d['aoi']}-{d['idx']}", {"aoi": d["aoi"]}).setdefault(key, paths[name])
    out = {k: v for k, v in sorted(tiles.items()) if all(layer in v for layer in LAYERS)}
    for v in out.values():
        v["s1grd"] = v["s1grd_post"]
    return out


def scan_split(root, split):
    paths = {}
    for dirpath, _, fnames in os.walk(os.path.join(root, split)):
        for fn in fnames:
            if fn.endswith(".tif"):
                paths[fn] = os.path.join(dirpath, fn)
    return tile_table(paths)


def fetch_pre_s1(files, split, n_shards, local_dir, root):
    """The pre-event members of the first n_shards s1grd shards of a split (exp55 extracted the post pass only)."""
    from huggingface_hub import hf_hub_download
    dest = os.path.join(root, split)
    done_dir = os.path.join(dest, ".done")
    os.makedirs(done_dir, exist_ok=True)
    names = sorted(f for f in files if f.startswith(f"{e55.TREE}/shards/{split}/s1grd/") and f.endswith(".tar"))[:n_shards]
    got = []
    for sh in names:
        marker = os.path.join(done_dir, os.path.basename(sh) + ".pre.done")
        if os.path.exists(marker):
            got.append({"shard": sh, "cached": True})
            continue
        t0 = time.time()
        path = hf_hub_download(e55.REPO, sh, repo_type="dataset", local_dir=local_dir)
        with tarfile.open(path) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and "_s1grd_pre_" in os.path.basename(m.name)]
            try:
                tar.extractall(path=dest, members=members, filter="data")
            except TypeError:
                tar.extractall(path=dest, members=members)
        os.remove(path)
        with open(marker, "w") as f:
            f.write(str(len(members)))
        got.append({"shard": sh, "members_extracted": len(members), "seconds": time.time() - t0})
        print(f"  {sh}: {len(members)} pre-event members extracted in {time.time() - t0:.0f}s", flush=True)
    return got


def tiles_of(args, data_dir, summary):
    if not args.smoke:
        root = os.path.join(data_dir, "tree")
        base = e57.geoid_tiles(args, data_dir)                              # exp55's post-pass tree, fetched if absent
        files = e55.hub_files()
        if any(not base[s] for s in ("test", "val")):                       # emptied tree with surviving markers (exp58's case)
            import shutil
            for split, n in (("test", 4), ("val", 2)):
                shutil.rmtree(os.path.join(root, split, ".done"), ignore_errors=True)
                e55.fetch_split(files, split, n, data_dir, root)
        summary["config"]["pre_s1_shards"] = {split: fetch_pre_s1(files, split, n, data_dir, root) for split, n in (("test", 4), ("val", 2))}
        return {split: scan_split(root, split) for split in ("test", "val")}
    base = e57.geoid_tiles(args, data_dir)
    if base is None:
        return None
    out = {}
    for split, table in base.items():
        for tid, t in table.items():
            t["s1grd_post"] = t["s1grd"]
            pre = None
            snap = os.path.dirname(os.path.dirname(t["s1grd"]))
            for dirpath, _, fnames in os.walk(snap):
                for fn in fnames:
                    if fn.startswith(tid + "_s1grd_pre_"):
                        pre = os.path.join(dirpath, fn)
            if pre is None and not os.environ.get("HF_HUB_OFFLINE"):
                try:
                    from huggingface_hub import hf_hub_download
                    cand = [f for f in e55.hub_files() if f"/{tid}_s1grd_pre_" in f and f.startswith("sample/")]
                    if cand:
                        pre = hf_hub_download(e55.REPO, cand[0], repo_type="dataset")
                except Exception as ex:  # noqa: BLE001
                    print(f"  smoke: pre-event S1 download failed for {tid}: {ex!r}", flush=True)
            if pre is None:
                summary["config"].setdefault("smoke_stand_ins", []).append(tid)
                pre = t["s1grd"]
            t["s1grd_pre"] = pre
        out[split] = table
    return out


def read_s1(tiles, chips, key):
    """exp55's S1 reader (nodata rule, dB) on the chosen pass."""
    s1 = np.zeros((len(chips), 2, e55.CHIP, e55.CHIP), np.float32)
    by_tile = {}
    for i, x in enumerate(chips):
        by_tile.setdefault(x["tile"], []).append(i)
    import rasterio
    for tid, idx in by_tile.items():
        with rasterio.open(tiles[tid][key]) as src:
            for i in idx:
                x = e55.read_chip(src, chips[i]["r"], chips[i]["c"]).astype(np.float32)
                x = np.where(np.isfinite(x) & (x <= e55.S1_FILL_MAX), x, 0.0)
                s1[i] = 10.0 * np.log10(np.maximum(x, e55.S1_FLOOR))
    return s1


# ----------------------------------------------------------------------------- the readings
def w1_of(model, sensor, x, head):
    return e57.w1(np.stack([exp18.head_prob_logit(e55.features(model, sensor, x, s), *head)[0] for s in SHIFTS]))


def pair_reading(name, pa, pb, ok, labels, events):
    """One pair on identical windows: compare's label-free summary, then what the difference is by the label."""
    sa, sb = PAIRS[name]
    a, b = pa > 0.5, pb > 0.5
    out = compare_inferences(a, b, ok, groups=events)
    d = out["arrays"]["disagree"]
    flooded = labels["B"] & ~labels["A"]
    err_a, err_b = (a != labels[TASK_OF[sa]]), (b != labels[TASK_OF[sb]])
    both = (np.abs(pa - 0.5) >= CONFIDENT_MARGIN) & (np.abs(pb - 0.5) >= CONFIDENT_MARGIN)
    bnd = boundary_indicator(a) > 0
    per = {}
    for ev in np.unique(events):
        m = d & (events == ev)[:, None, None]
        per[ev.item()] = {"n_disagree": int(m.sum()), "n_windows": int((ok & (events == ev)[:, None, None]).sum()),
                          "flooded_share": float(flooded[m].mean()) if m.any() else None,
                          "both_confident_flooded": float(flooded[m & both].mean()) if (m & both).any() else None}
    r = {"sides": [sa, sb], "n_windows": out["n_windows"], "n_disagree": out["n_disagree"], "disagreement_rate": out["disagreement_rate"],
         "flooded_share": float(flooded[d].mean()) if d.any() else None,
         "err_a_share": float(err_a[d].mean()) if d.any() else None, "err_b_share": float(err_b[d].mean()) if d.any() else None,
         "neither_share": float((~flooded & ~err_a & ~err_b)[d].mean()) if d.any() else None,
         "boundary_enrichment": cue_enrichment(bnd[ok], d[ok], n_boot=0)["enrichment"],
         "both_confident_share": float(both[d].mean()) if d.any() else None,
         "both_confident_flooded": float(flooded[d & both].mean()) if (d & both).any() else None,
         "per_event": per}
    print(f"{name} ({sa} vs {sb}): {r['n_disagree']} of {r['n_windows']} windows differ ({fmt(100 * r['disagreement_rate'], '.1f')}%) | flooded by the label {fmt(r['flooded_share'], '.3f')}, "
          f"{sa} wrong {fmt(r['err_a_share'], '.3f')}, {sb} wrong {fmt(r['err_b_share'], '.3f')}, neither {fmt(r['neither_share'], '.3f')} | boundary {fmt(r['boundary_enrichment'], '.1f')}x | "
          f"both confident {fmt(r['both_confident_share'], '.3f')}, flooded {fmt(r['both_confident_flooded'], '.3f')} of those", flush=True)
    return r


def event_test(R, x, y, stat="flooded_share", direction=1):
    """over_groups of (stat[x] - stat[y]) * direction over the events with enough differing windows in both pairs."""
    diffs = {}
    for ev, rx in R[x]["per_event"].items():
        ry = R[y]["per_event"].get(ev)
        if ry and rx["n_disagree"] >= MIN_EVENT_WINDOWS and ry["n_disagree"] >= MIN_EVENT_WINDOWS and rx[stat] is not None and ry[stat] is not None:
            diffs[ev] = direction * (rx[stat] - ry[stat])
    return over_groups(diffs)


def prereg(summary):
    R = summary["results"]["pairs"]
    T = summary["results"]["tests"] = {}
    p1 = p2 = None
    if all(k in R for k in PAIRS):
        r1 = R["time_only"]["flooded_share"] / R["mixed"]["flooded_share"] if R["mixed"]["flooded_share"] else float("nan")
        T["P1_time_vs_mixed"] = {"ratio_pooled": r1, **event_test(R, "time_only", "mixed")}
        p1 = bool(np.isfinite(r1) and r1 >= P1_RATIO and T["P1_time_vs_mixed"]["sign_p"] < 0.05)
        r2 = R["sensor_only"]["flooded_share"] / R["mixed"]["flooded_share"] if R["mixed"]["flooded_share"] else float("nan")
        T["P2_sensor_vs_mixed"] = {"ratio_pooled": r2, **event_test(R, "mixed", "sensor_only")}
        p2 = bool(np.isfinite(r2) and r2 <= P2_RATIO and T["P2_sensor_vs_mixed"]["sign_p"] < 0.05)
        T["prediction_sensor_fewer_differences"] = event_test(R, "mixed", "sensor_only", stat="n_disagree")
        T["prediction_both_confident_time_vs_mixed"] = event_test(R, "time_only", "mixed", stat="both_confident_flooded")
    summary["prereg"] = {"P1": p1, "P2": p2, "p1_ratio": P1_RATIO, "p2_ratio": P2_RATIO, "min_event_windows": MIN_EVENT_WINDOWS,
                         "complete": bool(p1 is not None and p2 is not None and not summary["failures"])}
    summary["n_failures"] = len(summary["failures"])
    if p1 is not None:
        print(f"prereg: P1 {p1} (time-only / mixed flooded share {fmt(T['P1_time_vs_mixed']['ratio_pooled'])}x, events {T['P1_time_vs_mixed']['w']}/{T['P1_time_vs_mixed']['l']}, p={T['P1_time_vs_mixed']['sign_p']:.2g}) | "
              f"P2 {p2} (sensor-only / mixed {fmt(T['P2_sensor_vs_mixed']['ratio_pooled'])}x, events {T['P2_sensor_vs_mixed']['w']}/{T['P2_sensor_vs_mixed']['l']}, p={T['P2_sensor_vs_mixed']['sign_p']:.2g}) | "
              f"sensor-only fewer differences on {T['prediction_sensor_fewer_differences']['w']}/{T['prediction_sensor_fewer_differences']['l']} events | complete {summary['prereg']['complete']}", flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp60 two periods, both sensors: the time-only, sensor-only and mixed differences on GEOID-Flood", "smoke": args.smoke, "device": DEV,
               "config": {"pairs": PAIRS, "cap_per_aoi": CAP, "confident_margin": CONFIDENT_MARGIN, "min_event_windows": MIN_EVENT_WINDOWS, "shifts": list(SHIFTS),
                          "s1_head": "one head fitted on the val chips' pre pass (permanent-water label) and post pass (water-after label), stacked",
                          "prereg": "P1 time-only flooded share >= 2x the mixed pair's pooled and higher on more events than not (p < 0.05); "
                                    "P2 sensor-only flooded share <= 0.5x the mixed pair's pooled and lower on more events than not (p < 0.05)"},
               "results": {"pairs": {}}, "failures": []}
    rows = []
    try:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        tiles = tiles_of(args, os.path.join(hf_home, "geoid_flood"), summary)
        if tiles is None:
            raise RuntimeError("smoke: GEOID sample tiles not in the local Hub cache")
        data = {}
        for split in ("val", "test"):
            sel = e55.select_chips(e55.chip_candidates(tiles[split], split), CAP, smoke_chips=4 if args.smoke else None)
            s2, s1post, lab = e55.read_imagery(tiles[split], sel)
            data[split] = {"s2": s2, "s1post": s1post, "s1pre": read_s1(tiles[split], sel, "s1grd_pre"), "lab": lab,
                           "aoi": np.array([x["aoi"] for x in sel]), "clear": np.array([x["clear"] >= e55.MIN_CLEAR for x in sel], bool)}
            summary["config"][f"{split}_chips"] = {"tiles": len(tiles[split]), "selected": len(sel), "clear": int(data[split]["clear"].sum()), "events": len(set(data[split]["aoi"].tolist()))}
            print(f"{split}: {len(tiles[split])} tiles with all five layers, {len(sel)} chips, {int(data[split]['clear'].sum())} clear, {len(set(data[split]['aoi'].tolist()))} events", flush=True)
        model = hb.load_model()
        v = data["val"]
        t0 = time.time()
        head_s2 = e55.fit_head(e55.features(model, "s2", v["s2"][v["clear"]], 0), e55.task_labels(v["lab"][v["clear"]], "A"))
        f_pre, f_post = e55.features(model, "s1", v["s1pre"], 0), e55.features(model, "s1", v["s1post"], 0)
        head_s1 = e55.fit_head(np.concatenate([f_pre, f_post]), np.concatenate([e55.task_labels(v["lab"], "A"), e55.task_labels(v["lab"], "B")]))
        del f_pre, f_post
        summary["config"]["head_seconds"] = time.time() - t0
        te = data["test"]
        idx = np.flatnonzero(te["clear"])
        t0 = time.time()
        P = {"A_s2pre": w1_of(model, "s2", te["s2"][idx], head_s2), "A_s1pre": w1_of(model, "s1", te["s1pre"][idx], head_s1), "B_s1post": w1_of(model, "s1", te["s1post"][idx], head_s1)}
        summary["config"]["encode_seconds"] = time.time() - t0
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        lab = te["lab"][idx]
        ya, oka = e57.labels_of(e55.task_labels(lab, "A"))
        yb, okb = e57.labels_of(e55.task_labels(lab, "B"))
        ok, events, labels = oka & okb, te["aoi"][idx], {"A": ya, "B": yb}
        summary["results"]["accuracy_own_task"] = {k: float((((p > 0.5) == labels[TASK_OF[k]])[ok]).mean()) for k, p in P.items()}
        print("accuracy on its own task: " + ", ".join(f"{k} {v:.4f}" for k, v in summary["results"]["accuracy_own_task"].items()), flush=True)
        for name, (sa, sb) in PAIRS.items():
            try:
                r = pair_reading(name, P[sa], P[sb], ok, labels, events)
                summary["results"]["pairs"][name] = r
                rows.append({"pair": name, "a": sa, "b": sb, **{k: v for k, v in r.items() if k not in ("sides", "per_event")}})
            except Exception as ex:  # noqa: BLE001
                e57.fail(summary, name, ex)
        np.savez_compressed(os.path.join(hb.OUT, f"exp60_masks{suffix}.npz"), ok=ok, y_permanent=ya, y_after=yb, event=events,
                            **{k: (p > 0.5) for k, p in P.items()}, **{f"margin_{k}": np.abs(p - 0.5).astype(np.float32) for k, p in P.items()})
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp60_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        with open(os.path.join(hb.OUT, f"exp60_two_periods{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
