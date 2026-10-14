"""Choose and export the real map that `oe-inferencex demo` audits.

Source: the expert-annotated test tiles of Dynamic World (Zenodo record 4766508, `dw_test.zip`, CC BY 4.0), the archive
exp67 audited: a served global land-cover product this project had no hand in, with the nine probabilities it
publishes about itself and an expert annotation of the same pixels.

The tile is chosen by a rule written before any tile was looked at, so that the demo is representative and not
flattering. Among the tiles that an expert marked almost fully (at least 90% of windows), that hold at least three
classes with 5% of the windows each, and whose error rate lies between 5% and 35%, take the tile whose error capture
at the 5% review budget is the MEDIAN of those tiles (the lower median when their number is even). Every candidate
and its numbers go to exp/out/demo_sample_selection.json, so the choice can be checked.

Exported, at window level to keep the package small: the mean probability per 4 x 4 pixel window (float16, nine
classes) and the expert's class per window (-1 where under half of the window was marked).

    python scripts/make_demo_sample.py            # needs rasterio and the archive under $HF_HOME/dw_test
    python scripts/make_demo_sample.py --smoke    # synthetic tiles, no archive, checks the selection logic
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from oe_inferencex.assess import assess_prediction  # noqa: E402

WIN, BUDGET = 4, 0.05
MIN_MARKED, MIN_CLASSES, CLASS_SHARE, ERR_RANGE = 0.90, 3, 0.05, (0.05, 0.35)
CLASSES = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice")


def to_windows(lulc, p, win=WIN):
    """(mean probability per window, valid window, expert class per window or -1)."""
    C, H, W = p.shape
    h, w = H // win, W // win
    fin = np.isfinite(p).all(0)[:h * win, :w * win].reshape(h, win, w, win)
    pw = np.where(np.isfinite(p), p, 0.0)[:, :h * win, :w * win].reshape(C, h, win, w, win).mean((2, 4))
    valid = fin.all((1, 3))
    marked = (lulc > 0)[:h * win, :w * win].reshape(h, win, w, win)
    counts = np.stack([((lulc - 1) == c)[:h * win, :w * win].reshape(h, win, w, win).sum((1, 3)) for c in range(C)], 0)
    y = np.where(marked.sum((1, 3)) >= win * win / 2, counts.argmax(0), -1)
    return np.clip(pw, 0, 1), valid, y


def tile_stats(pw, valid, y):
    ref = np.where(valid, y, -1)
    a = assess_prediction(pw, is_logit=False, patch=1, nodata_mask=~valid, reference=ref, budgets=(BUDGET, 0.10))
    r = a["against_reference"]
    dec = pw.argmax(0)
    shares = np.array([(dec[valid] == c).mean() for c in range(pw.shape[0])])
    return {"marked_share": float((ref >= 0).sum() / max(valid.sum(), 1)), "error_rate": r["error_rate"],
            "capture_05": r["error_capture_at_budget"][BUDGET]["errors_captured_fraction"],
            "capture_10": r["error_capture_at_budget"][0.10]["errors_captured_fraction"],
            "n_classes_5pct": int((shares >= CLASS_SHARE).sum()), "n_windows": int(valid.sum())}


def choose(stats):
    """Index of the chosen tile among `stats`, and the indices of the candidates."""
    cand = [i for i, s in enumerate(stats) if s["marked_share"] >= MIN_MARKED and s["n_classes_5pct"] >= MIN_CLASSES
            and ERR_RANGE[0] <= s["error_rate"] <= ERR_RANGE[1]]
    if not cand:
        raise SystemExit("no tile passes the rule")
    order = sorted(cand, key=lambda i: (stats[i]["capture_05"], i))
    return order[(len(order) - 1) // 2], cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--audit", action="store_true", help="audit the bundled sample and write exp/out/demo_sample_audit.json (numpy only)")
    args = ap.parse_args()
    if args.audit:
        z = np.load(os.path.join(ROOT, "oe_inferencex", "sample", "dynamic_world_tile.npz"))
        pw, y = z["probs"].astype(np.float32), z["expert"].astype(int)
        nodata = ~np.isfinite(pw).all(0)
        a = assess_prediction(pw, is_logit=False, patch=1, nodata_mask=nodata, reference=np.where(nodata, -1, y), budgets=(0.01, BUDGET, 0.10, 0.20, 0.50))
        r = a["against_reference"]
        out = {"tile": json.loads(str(z["meta"]))["tile"], "n_windows": a["n_windows"], "n_windows_scored": r["n_windows_scored"],
               "error_rate": r["error_rate"], "review_sets": {str(b): {"n_windows": a["review_sets"][b]["n_windows"],
                                                                       "boundary_share_in_set": a["review_sets"][b]["boundary_share_in_set"],
                                                                       **r["error_capture_at_budget"][b]} for b in (0.01, BUDGET, 0.10, 0.20, 0.50)}}
        with open(os.path.join(ROOT, "exp", "out", "demo_sample_audit.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out)[:400])
        return
    tiles = []
    if args.smoke:
        rng = np.random.default_rng(0)
        for k in range(7):
            yy, xx = np.mgrid[:128, :128]
            truth = ((yy // 32 + xx // 32 + k) % 4).astype(np.int16)
            logits = rng.normal(0, 1.2 + 0.2 * k, (9, 128, 128)); logits[truth, yy, xx] += 2.0
            p = np.exp(logits) / np.exp(logits).sum(0)
            tiles.append((f"smoke_{k}", truth + 1, p.astype(np.float32), (0.0, 0.0)))
    else:
        import io
        import rasterio
        import rasterio.warp
        path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "dw_test", "dw_test.zip")
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist() if "/label_dw_" in n)
            for i, nm in enumerate(names):
                with rasterio.open(io.BytesIO(zf.read(nm))) as src:
                    d = {n: j for j, n in enumerate(src.descriptions)}
                    lulc = src.read()[d["lulc"]].astype(np.int16)
                    cx, cy = src.transform * (src.width / 2, src.height / 2)
                    lon, lat = rasterio.warp.transform(src.crs, "EPSG:4326", [cx], [cy])
                with rasterio.open(io.BytesIO(zf.read(nm.replace("/label_dw_", "/probs_dw_")))) as src:
                    p = src.read().astype(np.float32)
                tiles.append((os.path.basename(nm), lulc, p, (float(lon[0]), float(lat[0]))))
                if (i + 1) % 50 == 0:
                    print(f"  read {i + 1}/{len(names)}", flush=True)

    stats, arrays = [], []
    for name, lulc, p, lonlat in tiles:
        pw, valid, y = to_windows(lulc, p)
        s = tile_stats(pw, valid, y) if valid.any() and (y[valid] >= 0).any() else None
        if s is None or not np.isfinite(s["error_rate"]):
            continue
        s.update(tile=name, lon=lonlat[0], lat=lonlat[1])
        stats.append(s); arrays.append((pw, valid, y))
    k, cand = choose(stats)
    chosen = stats[k]
    caps = sorted(stats[i]["capture_05"] for i in cand)
    print(f"{len(stats)} tiles scored, {len(cand)} pass the rule; capture at 5% among them: min {caps[0]:.3f}, "
          f"median {caps[(len(caps) - 1) // 2]:.3f}, max {caps[-1]:.3f}")
    print("chosen:", json.dumps(chosen))

    rule = (f"marked_share >= {MIN_MARKED}, at least {MIN_CLASSES} classes with {CLASS_SHARE:.0%} of windows, error rate in "
            f"{list(ERR_RANGE)}; the lower median of capture at the {BUDGET:.0%} budget")
    out = {"source": {"zenodo_record": 4766508, "archive": "dw_test.zip", "licence": "CC BY 4.0", "product": "Dynamic World"},
           "rule": rule, "n_tiles_scored": len(stats), "n_candidates": len(cand), "chosen": chosen,
           "candidates": [stats[i] for i in cand], "smoke": bool(args.smoke)}
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(ROOT, "exp", "out", f"demo_sample_selection{tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    if not args.smoke:
        pw, valid, y = arrays[k]
        os.makedirs(os.path.join(ROOT, "oe_inferencex", "sample"), exist_ok=True)
        np.savez_compressed(os.path.join(ROOT, "oe_inferencex", "sample", "dynamic_world_tile.npz"),
                            probs=np.where(valid, pw, np.nan).astype(np.float16), expert=np.where(valid, y, -1).astype(np.int8),
                            meta=json.dumps({"tile": chosen["tile"], "lon": chosen["lon"], "lat": chosen["lat"], "classes": CLASSES,
                                             "window_m": 40, "rule": rule, "n_candidates": len(cand), "zenodo_record": 4766508,
                                             "licence": "CC BY 4.0"}))
        print("wrote oe_inferencex/sample/dynamic_world_tile.npz", os.path.getsize(os.path.join(ROOT, "oe_inferencex", "sample", "dynamic_world_tile.npz")), "bytes")


if __name__ == "__main__":
    main()
