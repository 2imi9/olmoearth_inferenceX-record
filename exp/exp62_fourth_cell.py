#!/usr/bin/env python
"""exp62: the fourth cell. Post-event Sentinel-2 from WorldFloods v2 completes the two-period, two-sensor square on GEOID-Flood.

Why. exp60 held three of the four cells (S2 before, S1 before, S1 after) and showed the sensor axis isolates before the
event while the same-sensor radar difference across the event is only 26% flood, 54.5% of it the pre-event radar calling
water that neither the label nor a date-matched optical product holds (exp61). GEOID has no optical pass after the
event; WorldFloods v2 (isp-uv-es/WorldFloodsv2, Sentinel-2 L1C at 10 m with cloud and water masks, CC BY-NC 4.0) holds
26 post-event scenes on four of the eleven activations behind exp60's 55 events, most of them clear. With the fourth
cell the two time-only differences can be compared across sensors and the sensor-only difference after the event
against the one before.

Design. A water head for post-event optical imagery: OlmoEarth v1 Base features of WorldFloods L1C chips (the 13 bands
mapped to the encoder's twelve, B10 dropped, digital numbers unchanged as for GEOID's composites) and a balanced logistic
head (exp55.fit_head), fitted on chips cut at stride FIT_STRIDE from WorldFloods train scenes of activations that are
NOT among exp60's eleven, at least FIT_CLEAR clear, labels from the water mask (invalid and cloud pixels ignored), capped
at FIT_CHIPS. The fourth cell on exp60's chips: every WorldFloods scene of the four shared activations is reprojected
(nearest) onto each exp60 chip it covers; chips at least CHIP_CLEAR clear on the scene's cloud mask keep the scene with
the most clear pixels; the head runs at the four crop offsets and the shift-averaged decision B_s2post lands on exp60's
14 x 14 windows. exp60's chips and decisions are reproduced from the same selection and asserted equal to the committed
masks; margins rebuild each side's probability. Pairs on the chips with all four cells, through compare on identical
windows: time-only optical (A_s2pre, B_s2post), time-only radar (A_s1pre, B_s1post), sensor-only before (A_s2pre,
A_s1pre), sensor-only after (B_s2post, B_s1post), mixed (A_s2pre, B_s1post) and its mirror (A_s1pre, B_s2post). Per
pair: disagreement rate, the share flooded by the label, each side's error on its own task, the both-confident share
and its flooded share, per event with compare.over_groups (events with at least MIN_EVENT_WINDOWS differing windows in
both pairs of a test). B_s2post is also graded on GEOID's water-after label and against WorldFloods' own water mask
at the S2 date. Caveat: the fitting imagery is top-of-atmosphere L1C while exp60's heads read surface-reflectance
composites; the fourth cell's head is self-consistent on L1C, and the two optical cells differ in processing level.

Preregistered (one-sided):
  P1  the optical time-only difference is flooded by the label at least as often as the radar time-only difference on
      the same chips: pooled share higher, and higher on more events than not (p < 0.05). exp61's reason: the radar's
      pre-event false alarms are the larger contaminant.
  P2  the sensor-only difference after the event is flooded less often than the optical time-only difference: pooled
      share lower, and lower on more events than not (p < 0.05). A cross-sensor disagreement is sensor error in either
      period; a same-sensor difference across the event is where the flood is.
  Falsification: P1 fails if the optical time-only pair is flooded less often pooled or the event test misses (the
  optical pre-event head would then contaminate its temporal difference at least as much as the radar's); P2 fails if
  the post-event sensor difference carries the flood as much as the temporal one (the sensor axis would not isolate
  after the event). Stated prediction, not tested: the sensor-only-before pair reproduces exp60's near-zero flooded share
  on this subset.

Inputs. exp/out/exp60_masks.npz, $HF_HOME/geoid_flood (exp60's tree), WorldFloods v2 (fetched under $HF_HOME), the v1
Base checkpoint. Outputs. exp/out/exp62_summary.json, exp/out/exp62_fourth_cell.csv (one row per pair),
exp/out/exp62_masks.npz (B_s2post decisions and margins, WorldFloods water and clear per window, chip tile / row /
column, for the figure). --smoke: WorldFloods only, three small scenes, the head graded on the third scene's own mask.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp55_geoid_flood as e55  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import exp60_two_periods as e60  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import compare_inferences, over_groups  # noqa: E402

WF = "isp-uv-es/WorldFloodsv2"
WF_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
BAND_IDX = [WF_BANDS.index(b) for b in exp18.OE_BANDS]
CHIP, SHIFTS, DEV, SL = e55.CHIP, e55.SHIFTS, e55.DEV, e57.SL
FIT_STRIDE, FIT_CLEAR, FIT_CHIPS, FIT_SCENES, CHIP_CLEAR = 128, 0.9, 2000, 40, 0.9
MIN_EVENT_WINDOWS, CONFIDENT_MARGIN = 20, e60.CONFIDENT_MARGIN
PAIRS = {"time_only_s2": ("A_s2pre", "B_s2post"), "time_only_s1": ("A_s1pre", "B_s1post"), "sensor_only_pre": ("A_s2pre", "A_s1pre"),
         "sensor_only_post": ("B_s2post", "B_s1post"), "mixed": ("A_s2pre", "B_s1post"), "mixed_mirror": ("A_s1pre", "B_s2post")}
TASK_OF = {"A_s2pre": "A", "A_s1pre": "A", "B_s1post": "B", "B_s2post": "B"}
fmt = e57.fmt


# ----------------------------------------------------------------------------- WorldFloods
def wf_meta():
    from huggingface_hub import hf_hub_download
    return pd.read_csv(hf_hub_download(WF, "dataset_metadata.csv", repo_type="dataset"))


def wf_paths(row):
    from huggingface_hub import hf_hub_download
    return {k: hf_hub_download(WF, f"{row['split']}/{k}/{row['event id']}.tif", repo_type="dataset") for k in ("S2", "gt")}


def wf_label(gt):
    """WorldFloods gt (2, H, W): band 1 {0 invalid, 1 clear, 2 cloud}, band 2 {0 invalid, 1 land, 2 water} -> {-1, 0, 1}."""
    clear = gt[0] == 1
    return np.where(clear & (gt[1] > 0), (gt[1] == 2).astype(np.int64), -1), clear


def fit_chips(df, exclude_acts, rng):
    """Chips for the head from clear WorldFloods train scenes of other activations: (N, 12, 64, 64) DN and (N, 64, 64) labels."""
    cand = df[(df["split"] == "train") & ~df["ems_code"].astype(str).str.upper().isin(exclude_acts)].copy()
    cand = cand.iloc[rng.permutation(len(cand))]
    xs, ys, used = [], [], []
    for _, row in cand.iterrows():
        if len(used) >= FIT_SCENES or sum(len(x) for x in xs) >= FIT_CHIPS:
            break
        try:
            p = wf_paths(row)
        except Exception as ex:  # noqa: BLE001
            print(f"  fit scene {row['event id']} skipped: {ex!r}", flush=True)
            continue
        with rasterio.open(p["gt"]) as g:
            gt = g.read()
        lab, clear = wf_label(gt)
        if clear.mean() < FIT_CLEAR:
            continue
        with rasterio.open(p["S2"]) as s:
            H, W = s.height, s.width
            pos = [(r, c) for r in range(0, H - CHIP + 1, FIT_STRIDE) for c in range(0, W - CHIP + 1, FIT_STRIDE)
                   if clear[r:r + CHIP, c:c + CHIP].mean() >= FIT_CLEAR and (lab[r:r + CHIP, c:c + CHIP] >= 0).mean() >= FIT_CLEAR]
            rng.shuffle(pos)
            pos = pos[:max(1, FIT_CHIPS // FIT_SCENES)]
            for r, c in pos:
                x = s.read(window=Window(c, r, CHIP, CHIP)).astype(np.float32)[BAND_IDX]
                xs.append(x[None]); ys.append(lab[r:r + CHIP, c:c + CHIP][None])
        used.append({"event id": row["event id"], "ems_code": row["ems_code"], "chips": len(pos), "water_share": float((lab == 1)[lab >= 0].mean())})
        print(f"  fit scene {row['event id']}: {len(pos)} chips, water {used[-1]['water_share']:.3f}", flush=True)
    return np.concatenate(xs), np.concatenate(ys), used


def scene_chip(scene, tile_path, r, c):
    """A WorldFloods scene's 15 bands and gt (2 bands) on one GEOID chip's grid, nearest; None when the chip lies outside."""
    with rasterio.open(tile_path) as t:
        win = Window(c, r, CHIP, CHIP)
        dst_transform, dst_crs = t.window_transform(win), t.crs
        b = transform_bounds(t.crs, scene["crs"], *t.window_bounds(win), densify_pts=5)
    sb = scene["bounds"]
    if b[0] < sb.left or b[2] > sb.right or b[1] < sb.bottom or b[3] > sb.top:
        return None
    out = {}
    for key, src in (("S2", scene["s2"]), ("gt", scene["gt"])):
        w = src.window(*b).round_offsets().round_lengths()
        w = Window(max(int(w.col_off) - 2, 0), max(int(w.row_off) - 2, 0), int(w.width) + 4, int(w.height) + 4)
        data = src.read(window=w)
        arr = np.zeros((src.count, CHIP, CHIP), data.dtype)
        for i in range(src.count):
            reproject(data[i], arr[i], src_transform=src.window_transform(w), src_crs=src.crs, dst_transform=dst_transform, dst_crs=dst_crs, resampling=Resampling.nearest)
        out[key] = arr
    return out


# ----------------------------------------------------------------------------- readings
def reading(name, P, ok, labels, events):
    sa, sb = PAIRS[name]
    pa, pb = P[sa], P[sb]
    a, b = pa > 0.5, pb > 0.5
    out = compare_inferences(a, b, ok, groups=events)
    d = out["arrays"]["disagree"]
    flooded = labels["B"] & ~labels["A"]
    err_a, err_b = a != labels[TASK_OF[sa]], b != labels[TASK_OF[sb]]
    both = (np.abs(pa - 0.5) >= CONFIDENT_MARGIN) & (np.abs(pb - 0.5) >= CONFIDENT_MARGIN)
    per = {}
    for ev in np.unique(events):
        m = d & (events == ev)[:, None, None]
        per[ev.item()] = {"n_disagree": int(m.sum()), "flooded_share": float(flooded[m].mean()) if m.any() else None}
    r = {"sides": [sa, sb], "n_windows": out["n_windows"], "n_disagree": out["n_disagree"], "disagreement_rate": out["disagreement_rate"],
         "flooded_share": float(flooded[d].mean()) if d.any() else None, "err_a_share": float(err_a[d].mean()) if d.any() else None,
         "err_b_share": float(err_b[d].mean()) if d.any() else None, "both_confident_share": float(both[d].mean()) if d.any() else None,
         "both_confident_flooded": float(flooded[d & both].mean()) if (d & both).any() else None, "per_event": per}
    print(f"{name} ({sa} vs {sb}): {r['n_disagree']} of {r['n_windows']} differ ({fmt(100 * r['disagreement_rate'], '.1f')}%) | flooded {fmt(r['flooded_share'], '.3f')}, "
          f"{sa} off {fmt(r['err_a_share'], '.3f')}, {sb} off {fmt(r['err_b_share'], '.3f')} | both confident {fmt(r['both_confident_share'], '.3f')}, flooded {fmt(r['both_confident_flooded'], '.3f')}", flush=True)
    return r


def event_test(R, x, y, direction=1):
    diffs = {}
    for ev, rx in R[x]["per_event"].items():
        ry = R[y]["per_event"].get(ev)
        if ry and rx["n_disagree"] >= MIN_EVENT_WINDOWS and ry["n_disagree"] >= MIN_EVENT_WINDOWS and rx["flooded_share"] is not None and ry["flooded_share"] is not None:
            diffs[ev] = direction * (rx["flooded_share"] - ry["flooded_share"])
    return over_groups(diffs)


def prereg(summary):
    R = summary["results"].get("pairs", {})
    T = summary["results"]["tests"] = {}
    p1 = p2 = None
    if all(k in R for k in ("time_only_s2", "time_only_s1", "sensor_only_post")):
        T["P1_optical_vs_radar_time_only"] = event_test(R, "time_only_s2", "time_only_s1")
        p1 = bool(R["time_only_s2"]["flooded_share"] >= R["time_only_s1"]["flooded_share"] and T["P1_optical_vs_radar_time_only"]["sign_p"] < 0.05)
        T["P2_sensor_post_vs_time_only_s2"] = event_test(R, "time_only_s2", "sensor_only_post")
        p2 = bool(R["sensor_only_post"]["flooded_share"] < R["time_only_s2"]["flooded_share"] and T["P2_sensor_post_vs_time_only_s2"]["sign_p"] < 0.05)
    summary["prereg"] = {"P1": p1, "P2": p2, "min_event_windows": MIN_EVENT_WINDOWS, "complete": bool(p1 is not None and p2 is not None and not summary["failures"])}
    summary["n_failures"] = len(summary["failures"])
    if p1 is not None:
        print(f"prereg: P1 {p1} (optical time-only flooded {fmt(R['time_only_s2']['flooded_share'], '.3f')} vs radar {fmt(R['time_only_s1']['flooded_share'], '.3f')}; events {T['P1_optical_vs_radar_time_only']['w']}/{T['P1_optical_vs_radar_time_only']['l']}, p={T['P1_optical_vs_radar_time_only']['sign_p']:.2g}) | "
              f"P2 {p2} (sensor-only after {fmt(R['sensor_only_post']['flooded_share'], '.3f')}; events {T['P2_sensor_post_vs_time_only_s2']['w']}/{T['P2_sensor_post_vs_time_only_s2']['l']}, p={T['P2_sensor_post_vs_time_only_s2']['sign_p']:.2g}) | complete {summary['prereg']['complete']}", flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    rng = np.random.default_rng(0)
    summary = {"experiment": "exp62 the fourth cell: WorldFloods v2 post-event Sentinel-2 completes the two-period, two-sensor square on GEOID-Flood", "smoke": args.smoke, "device": DEV,
               "config": {"worldfloods": WF, "band_idx": BAND_IDX, "fit_stride": FIT_STRIDE, "fit_clear": FIT_CLEAR, "fit_chips": FIT_CHIPS, "fit_scenes": FIT_SCENES, "chip_clear": CHIP_CLEAR,
                          "pairs": PAIRS, "min_event_windows": MIN_EVENT_WINDOWS, "confident_margin": CONFIDENT_MARGIN,
                          "prereg": "P1 optical time-only flooded share >= radar time-only pooled and higher on more events than not (p < 0.05); "
                                    "P2 sensor-only-after flooded share < optical time-only pooled and lower on more events than not (p < 0.05)"},
               "results": {}, "failures": []}
    rows = []
    try:
        df = wf_meta()
        model = hb.load_model()
        if args.smoke:
            pick = ["EMSR279_07ALFARO_GRA_v1", "EMSR279_08ALCALADEEBRO_GRA_v1", "EMSR273_01GRILE_DEL_MONIT04_v2"]   # two train scenes, one test scene, all small and clear
            sub = df.set_index("event id").loc[pick].reset_index()
            X, Y, used = fit_chips(sub.iloc[:2], set(), rng)
            head = e55.fit_head(e55.features(model, "s2", X[:64], 0), Y[:64])
            p = wf_paths(sub.iloc[2])
            with rasterio.open(p["S2"]) as s, rasterio.open(p["gt"]) as g:
                H, W = s.height, s.width
                pos = [(r, c) for r in range(0, H - CHIP + 1, FIT_STRIDE) for c in range(0, W - CHIP + 1, FIT_STRIDE)][:16]
                x = np.stack([s.read(window=Window(c, r, CHIP, CHIP)).astype(np.float32)[BAND_IDX] for r, c in pos])
                gt = np.stack([g.read(window=Window(c, r, CHIP, CHIP)) for r, c in pos])
            prob = e57.w1(np.stack([exp18.head_prob_logit(e55.features(model, "s2", x, sh), *head)[0] for sh in SHIFTS]))
            lab, _ = wf_label(gt.transpose(1, 0, 2, 3)) if False else (np.stack([wf_label(gg)[0] for gg in gt]), None)
            y, ok = e57.labels_of(lab)
            ndwi = (x[:, 1] - x[:, 3]) / np.maximum(x[:, 1] + x[:, 3], 1)                      # (B03 - B08) / (B03 + B08) in the encoder band order
            ndwi_w = e57.w1(np.stack([np.stack([nd[sh:sh + 60, sh:sh + 60].reshape(15, 4, 15, 4).mean(axis=(1, 3)) for nd in ndwi]) for sh in SHIFTS]) > 0)
            summary["results"]["smoke"] = {"fit": used, "held_out_windows": int(ok.sum()), "held_out_agreement": float(((prob > 0.5) == y)[ok].mean()) if ok.any() else None,
                                           "prob_on_water_windows": float(prob[ok & y].mean()) if (ok & y).any() else None, "prob_on_land_windows": float(prob[ok & ~y].mean()) if (ok & ~y).any() else None,
                                           "ndwi_positive_agreement": float(((ndwi_w > 0.5) == y)[ok].mean()) if ok.any() else None}
            print("smoke: head fitted on", sum(u["chips"] for u in used), "chips;", {k: v for k, v in summary["results"]["smoke"].items() if k != "fit"}, flush=True)
        else:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            tiles = e60.tiles_of(args, os.path.join(hf_home, "geoid_flood"), summary)
            sel = e55.select_chips(e55.chip_candidates(tiles["test"], "test"), e60.CAP)
            chips = [x for x in sel if x["clear"] >= e55.MIN_CLEAR]
            lab = np.stack([x["lab"] for x in chips])
            ya, oka = e57.labels_of(e55.task_labels(lab, "A")); yb, okb = e57.labels_of(e55.task_labels(lab, "B"))
            z = np.load(os.path.join(hb.OUT, "exp60_masks.npz"))
            events = np.array([x["aoi"] for x in chips])
            if not (len(chips) == len(z["event"]) and (events == z["event"]).all() and (ya == z["y_permanent"]).all() and (yb == z["y_after"]).all()):
                raise RuntimeError("chip selection does not reproduce exp60's")
            acts = sorted({e.split("-")[0] for e in set(events.tolist())})
            # the head
            t0 = time.time()
            X, Y, used = fit_chips(df, set(acts), rng)
            feats = np.concatenate([e55.features(model, "s2", X[i:i + 256], 0) for i in range(0, len(X), 256)])
            head = e55.fit_head(feats, Y)
            del feats
            summary["config"]["fit"] = {"scenes": used, "chips": int(len(X)), "water_share": float((Y == 1)[Y >= 0].mean()), "seconds": time.time() - t0}
            print(f"head fitted on {len(X)} chips from {len(used)} scenes ({time.time() - t0:.0f}s)", flush=True)
            # the fourth cell on exp60's chips
            t0 = time.time()
            scenes = []
            for _, row in df[df["ems_code"].astype(str).str.upper().isin(acts)].iterrows():
                p = wf_paths(row)
                s, g = rasterio.open(p["S2"]), rasterio.open(p["gt"])
                scenes.append({"row": row, "s2": s, "gt": g, "crs": s.crs, "bounds": s.bounds})
            summary["config"]["scenes"] = [{"event id": sc["row"]["event id"], "s2_date": str(sc["row"]["s2_date"])[:10]} for sc in scenes]
            N = len(chips)
            x4 = np.zeros((N, 12, CHIP, CHIP), np.float32); have = np.zeros(N, bool); wf_water = np.full((N, CHIP, CHIP), -1, np.int64); scene_of = np.full(N, -1)
            for i, ch in enumerate(chips):
                best = None
                for k, sc in enumerate(scenes):
                    if sc["row"]["ems_code"] != ch["aoi"].split("-")[0]:
                        continue
                    got = scene_chip(sc, tiles["test"][ch["tile"]]["s2l2a"], ch["r"], ch["c"])
                    if got is None:
                        continue
                    clear = float((got["gt"][0] == 1).mean())
                    if clear >= CHIP_CLEAR and (best is None or clear > best[0]):
                        best = (clear, k, got)
                if best is not None:
                    _, k, got = best
                    x4[i] = got["S2"].astype(np.float32)[BAND_IDX]; have[i] = True; scene_of[i] = k
                    wf_water[i], _ = wf_label(got["gt"])
            for sc in scenes:
                sc["s2"].close(); sc["gt"].close()
            idx = np.flatnonzero(have)
            summary["config"]["fourth_cell"] = {"chips_with_clear_post_optical": int(len(idx)), "of": N, "events": sorted(set(events[idx].tolist())), "cut_seconds": time.time() - t0}
            print(f"fourth cell on {len(idx)} of {N} chips ({len(set(events[idx].tolist()))} events), cut in {time.time() - t0:.0f}s", flush=True)
            if len(idx) == 0:
                raise RuntimeError("no exp60 chip is covered by a clear WorldFloods scene")
            t0 = time.time()
            p4 = e57.w1(np.stack([np.concatenate([exp18.head_prob_logit(e55.features(model, "s2", x4[idx][j:j + 256], sh), *head)[0] for j in range(0, len(idx), 256)]) for sh in SHIFTS]))
            summary["config"]["encode_seconds"] = time.time() - t0
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
            ok = (oka & okb)[idx]
            labels = {"A": ya[idx], "B": yb[idx]}
            P = {"B_s2post": p4}
            for k in ("A_s2pre", "A_s1pre", "B_s1post"):
                P[k] = np.where(z[k][idx], 0.5 + z[f"margin_{k}"][idx], 0.5 - z[f"margin_{k}"][idx]).astype(np.float64)
            wf_y, wf_ok = e57.labels_of(wf_water[idx])
            summary["results"]["fourth_cell"] = {"accuracy_water_after_label": float((((p4 > 0.5) == labels["B"])[ok]).mean()),
                                                 "agreement_with_worldfloods_water": float((((p4 > 0.5) == wf_y)[ok & wf_ok]).mean()) if (ok & wf_ok).any() else None,
                                                 "worldfloods_vs_geoid_water_after": float(((wf_y == labels["B"])[ok & wf_ok]).mean()) if (ok & wf_ok).any() else None,
                                                 "accuracy_own_task_others": {k: float((((P[k] > 0.5) == labels[TASK_OF[k]])[ok]).mean()) for k in ("A_s2pre", "A_s1pre", "B_s1post")}}
            print("fourth cell: accuracy on GEOID's water-after label %.4f, agreement with WorldFloods' water %s, WorldFloods vs GEOID labels %s" % (
                summary["results"]["fourth_cell"]["accuracy_water_after_label"], fmt(summary["results"]["fourth_cell"]["agreement_with_worldfloods_water"], ".4f"), fmt(summary["results"]["fourth_cell"]["worldfloods_vs_geoid_water_after"], ".4f")), flush=True)
            summary["results"]["pairs"] = {}
            ev = events[idx]
            for name in PAIRS:
                try:
                    r = reading(name, P, ok, labels, ev)
                    summary["results"]["pairs"][name] = r
                    rows.append({"pair": name, **{k: v for k, v in r.items() if k not in ("sides", "per_event")}})
                except Exception as ex:  # noqa: BLE001
                    e57.fail(summary, name, ex)
            np.savez_compressed(os.path.join(hb.OUT, f"exp62_masks{suffix}.npz"), chip_index=idx, event=ev, tile=np.array([chips[i]["tile"] for i in idx]),
                                row=np.array([chips[i]["r"] for i in idx]), col=np.array([chips[i]["c"] for i in idx]), scene=np.array([summary["config"]["scenes"][scene_of[i]]["event id"] for i in idx]),
                                B_s2post=(p4 > 0.5), margin_B_s2post=np.abs(p4 - 0.5).astype(np.float32), wf_water=wf_y, wf_ok=wf_ok, ok=ok)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
    try:
        prereg(summary)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp62_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        with open(os.path.join(hb.OUT, f"exp62_fourth_cell{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
