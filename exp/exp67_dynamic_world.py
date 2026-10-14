#!/usr/bin/env python
"""exp67: auditing a production model with its own published probabilities: Dynamic World against the expert consensus.

Why. Every ranking result in this repository so far scores a model we or Ai2 fitted: a linear probe on frozen OlmoEarth
features (exp13 to exp66), a head we fine-tuned (exp52), or Ai2's published probes (exp51, exp54, exp63). None scores a
production land-cover product that someone else trained, deployed globally and published the per-pixel probabilities of.
Dynamic World is exactly that, and Ai2 (Patrick Johnson, 2026-09-11) suggested its expert-annotated validation labels as
an evaluation set. Our earlier judgement was that they were reachable only through Earth Engine; that judgement was
wrong. The expert test tiles are on Zenodo, ungated, CC BY 4.0, and they ship three things on one grid: the expert
consensus annotation, Dynamic World's own nine-class probability raster, and Dynamic World's Top-1 decision.

So this experiment needs no imagery and no encoder. It asks whether the recipe this repository recommends survives
contact with a served product it had no hand in: does a production model's own margin rank its own errors, do those
errors carry the explanation layer's cues, and are the published probabilities calibrated.

Data. Zenodo record 4766508, `dw_test.zip`, 3,897,915,689 bytes, md5 9bff6f3fc346cf458ab21a5478f5a9fe, CC BY 4.0: 409
label rasters and 409 probability rasters of 512 x 512 pixels at 10 m in UTM, plus a metadata CSV naming the Sentinel-2
granule each tile was annotated from. Conventions verified against the record's own description and five sampled tiles,
because the band order and the class indexing are both traps. In `label_*.tif` the band named `lulc` is the EXPERT
annotation and is ONE-indexed with 0 meaning "no markup", while the band named `label` is Dynamic World's Top-1 and is
ZERO-indexed; comparing them directly gives 0.3% agreement and comparing `lulc - 1` to `label` gives 52% to 80%, which
is the real number. In `probs_*.tif` the nine float32 bands are water, trees, grass, flooded_vegetation, crops,
shrub_and_scrub, built, bare and snow_and_ice; their argmax reproduces the published Top-1 exactly on every tile sampled,
and some pixels are not finite, so validity has to be checked per pixel.

Design. Pixels are pooled to the 4 px windows this repository scores on: the window's class probabilities are the mean
of its sixteen pixel vectors, its decision their argmax, its expert label the majority of its annotated pixels, and a
window is valid when at least half its pixels carry an annotation and all sixteen probability vectors are finite. An
error is a window whose decision differs from its expert label. Rankers, all label-free and all computed from what
Dynamic World published: the model's own margin (top-1 minus top-2 window probability, the quantity this repository
recommends), one minus the top-1 probability (the naive confidence, which ties wherever probabilities saturate),
predictive entropy over the nine classes, the prediction-boundary indicator, the boundary-first order (exp36), and a
control that uses no imagery and no probabilities at all, the rarity of the window's predicted class within its own
tile. Scored as everywhere here: pooled excess AURC, a one-sided exact sign test over tiles, error capture at 5, 10 and
20 percent. The explanation layer is measured on the error windows as cue enrichment (boundary, bottom margin quintile,
top entropy quintile), and the published top-1 probability is scored for calibration against correctness at BOTH scales:
on the pooled windows and, on every sixteenth annotated pixel, at Dynamic World's own native resolution, because averaging
sixteen probability vectors lowers the maximum and a window-level calibration number alone would confound the product
with the pooling.

Preregistered (one-sided):
  P1  Dynamic World's own margin ranks its own errors: its pooled excess AURC is at least 0.01 below the no-imagery
      class-rarity control, and lower on more tiles than not (p < 0.05).
  P2  the margin beats the naive confidence: its pooled excess AURC is lower, and it has strictly fewer tied values,
      which is the reason this repository scores a margin rather than one minus a maximum probability (exp13).
  P3  the errors are boundary-enriched above twofold, so the explanation layer transfers from our probes to a served
      product at nine classes.
  Falsification. P1 fails if a control that never sees the imagery ranks a production model's errors as well as the
  model's own confidence does, which would be the strongest negative result this repository could record about its own
  recipe. P2 fails if the two confidences are indistinguishable, retiring a design choice made in exp13. P3 fails if the
  cue does not concentrate, which at nine classes would follow exp63's parcel finding rather than exp66's land-cover one.
  Stated predictions, not tested: Dynamic World's agreement with the expert consensus lands near the 0.74 its paper
  reports; the boundary-first order loses to the margin, as at eight, fifteen and nineteen classes; the published
  probabilities are overconfident.

Limits carried into the record. The expert consensus is released only as a consensus, not per annotator, so the question
this dataset most invites, whether the model is unsure where the annotators disagreed, cannot be asked of it. Annotated
coverage varies by tile (0.57 to 0.95 on the sampled tiles), so tiles are weighted by their valid windows and every
per-tile test uses only tiles with enough errors to score. The nine classes include no cloud class, and cloud is one
reason a pixel may be unannotated.

Inputs: the Zenodo archive alone, cached under $HF_HOME/dw_test. CPU only, no encoder, no imagery. Outputs:
exp/out/exp67_summary.json, exp/out/exp67_dynamic_world.csv (one row per ranker), exp/out/exp67_windows.npz (the
per-window margin, entropy, top-1 probability, decision, label, validity, boundary and tile id, so every number here
recomputes on a laptop). --smoke: synthetic tiles, no download.
"""
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp57_difference_atlas as e57  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.explain import cue_enrichment, top_fraction  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, expected_calibration_error, oracle_aurc  # noqa: E402
from oe_inferencex.signals import boundary_indicator  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

URL = "https://zenodo.org/api/records/4766508/files/dw_test.zip/content"
MD5, NBYTES = "9bff6f3fc346cf458ab21a5478f5a9fe", 3897915689
CLASSES = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub", "built", "bare", "snow_and_ice")
WIN, TILE_PX = 4, 512
MARGIN, NAIVE, ENT, BND, LEX, CTRL = "margin", "1 - top-1 probability", "entropy", "boundary indicator", "boundary first, then margin", "control class rarity"
BUDGETS = (0.05, 0.10, 0.20)
MIN_LEAD, MIN_ENRICH, MIN_TILE_ERRORS = 0.01, 2.0, 3
PIXSUB = 16                                                                    # every 16th annotated pixel enters the calibration check
fmt = e57.fmt


# ----------------------------------------------------------------------------- data
def fetch(data_dir):
    """The Zenodo archive, resumed if interrupted and checked against the published md5."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "dw_test.zip")
    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have != NBYTES:
        for attempt in range(6):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have >= NBYTES:
                break
            req = urllib.request.Request(URL, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=180) as r, open(path, "ab" if have else "wb") as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as ex:  # noqa: BLE001
                print(f"  download attempt {attempt + 1} interrupted at {os.path.getsize(path) if os.path.exists(path) else 0} bytes: {ex!r}", flush=True)
        size = os.path.getsize(path)
        if size != NBYTES:
            raise RuntimeError(f"archive is {size} bytes, expected {NBYTES}")
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    if h.hexdigest() != MD5:
        raise RuntimeError(f"md5 {h.hexdigest()} != published {MD5}")
    return path


def read_tile(zf, name):
    """One tile: the expert annotation (1-indexed, 0 = no markup), Dynamic World's Top-1 (0-indexed) and its nine probabilities."""
    import rasterio
    with rasterio.open(io.BytesIO(zf.read(name))) as src:
        d = {n: i for i, n in enumerate(src.descriptions)}
        a = src.read()
    lulc, top1 = a[d["lulc"]].astype(np.int16), a[d["label"]].astype(np.int16)
    with rasterio.open(io.BytesIO(zf.read(name.replace("/label_dw_", "/probs_dw_")))) as src:
        p = src.read().astype(np.float32)
    return lulc, top1, p


def windows(lulc, top1, p, win=WIN):
    """Pixels -> 4 px windows: mean class probabilities, argmax decision, majority expert label, validity."""
    C, H, W = p.shape
    h, w = H // win, W // win
    finite = np.isfinite(p).all(0)
    pw = np.where(finite, p, 0.0).reshape(C, h, win, w, win).sum(axis=(2, 4))
    n_fin = finite.reshape(h, win, w, win).sum(axis=(1, 3))
    pw = pw / np.maximum(n_fin, 1)[None]
    marked = lulc > 0
    n_marked = marked.reshape(h, win, w, win).sum(axis=(1, 3))
    counts = np.stack([((lulc - 1) == c).reshape(h, win, w, win).sum(axis=(1, 3)) for c in range(C)], 0)
    y = counts.argmax(0)
    ok = (n_marked >= (win * win) // 2) & (n_fin == win * win)
    srt = np.sort(pw, axis=0)
    dec = pw.argmax(0)
    t1 = np.stack([(top1 == c).reshape(h, win, w, win).sum(axis=(1, 3)) for c in range(C)], 0)
    top1_win = t1.argmax(0)                                                     # the majority of the published per-pixel decisions, for the agreement check
    return {"dec": dec, "y": y, "ok": ok, "margin": srt[-1] - srt[-2], "top1p": srt[-1],
            "entropy": -(np.clip(pw, 1e-7, 1) * np.log(np.clip(pw, 1e-7, 1))).sum(0), "dw_top1": top1_win,
            "marked_share": n_marked / (win * win)}


def rarity(dec, ok):
    """The no-imagery control: how rare the window's predicted class is inside its own tile (higher = rarer = more suspect)."""
    out = np.zeros(dec.shape, np.float64)
    if not ok.any():
        return out
    vals, counts = np.unique(dec[ok], return_counts=True)
    freq = {int(v): c / counts.sum() for v, c in zip(vals, counts)}
    for v, f in freq.items():
        out[dec == v] = -np.log(max(f, 1e-9))
    return out


def synthetic(n_tiles=6, seed=0):
    """Smoke stand-in: tiles whose probabilities carry real signal about the label, with unmarked and non-finite regions."""
    rng = np.random.default_rng(seed)
    out = []
    yy, xx = np.mgrid[:TILE_PX, :TILE_PX]
    for k in range(n_tiles):
        truth = ((yy // 64 + xx // 64 + k) % len(CLASSES)).astype(np.int16)
        logits = rng.normal(0, 1.6, (len(CLASSES), TILE_PX, TILE_PX)).astype(np.float32)
        logits[truth, yy, xx] += 1.4                                            # weak enough that the model errs, so the ranking path is exercised
        p = np.exp(logits - logits.max(0)); p /= p.sum(0)
        p[:, :40, :40] = np.nan
        lulc = (truth + 1).astype(np.int16)
        lulc[rng.random((TILE_PX, TILE_PX)) < 0.25] = 0
        out.append((lulc, p.argmax(0).astype(np.int16), p))
    return out


# ----------------------------------------------------------------------------- scoring
def score(sig, err, tile, primary=(CTRL, NAIVE)):
    pooled = {k: aurc_expected(v, err) - oracle_aurc(len(err), int(err.sum())) for k, v in sig.items()}
    per = {k: [] for k in sig}
    tiles = []
    for t in np.unique(tile):
        m = tile == t
        e = err[m]
        if MIN_TILE_ERRORS <= e.sum() <= len(e) - MIN_TILE_ERRORS:
            tiles.append(int(t))
            for k in sig:
                per[k].append(aurc_expected(sig[k][m], e) - oracle_aurc(int(m.sum()), int(e.sum())))
    tests = {}
    for k in sig:
        if k == MARGIN:
            continue
        g = np.array(per[k]) - np.array(per[MARGIN])                            # positive: the margin is better
        w, l, t_ = wins_losses_ties(g)
        one = k in primary
        tests[k] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                    "median_gain": float(np.median(g)) if len(g) else None, "pooled_lead": pooled[k] - pooled[MARGIN]}
    cap = {k: {str(b): v for b, v in capture_at_budget_expected(sig[k], err, BUDGETS).items()} for k in sig}
    ties = {k: int(len(sig[k]) - len(np.unique(sig[k]))) for k in sig}
    return {"n_windows": int(len(err)), "n_errors": int(err.sum()), "accuracy": float(1 - err.mean()),
            "n_tiles_scored": len(tiles), "pooled_excess_aurc": pooled, "tests": tests, "capture": cap, "tied_values": ties}


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--tiles", type=int, default=0, help="cap the number of tiles (0 = all 409)")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    summary = {"experiment": "exp67 auditing Dynamic World with its own published probabilities against the expert consensus",
               "smoke": args.smoke,
               "config": {"zenodo_record": 4766508, "url": URL, "md5": MD5, "bytes": NBYTES, "licence": "CC BY 4.0",
                          "classes": list(CLASSES), "window_px": WIN, "budgets": list(BUDGETS), "min_tile_errors": MIN_TILE_ERRORS,
                          "conventions": "label_*.tif band 'lulc' is the expert annotation, 1-indexed, 0 = no markup; band 'label' is Dynamic World's Top-1, 0-indexed; probs_*.tif holds the nine class probabilities",
                          "prereg": "P1 the model's own margin beats the no-imagery class-rarity control by >= 0.01 pooled and on more tiles than not; "
                                    "P2 the margin beats one minus the top-1 probability and has strictly fewer ties; "
                                    "P3 the error windows are boundary-enriched above twofold"},
               "results": {}, "failures": []}
    rows = []
    try:
        if args.smoke:
            tiles_data = [(f"synthetic_{i}", *t) for i, t in enumerate(synthetic())]
            summary["config"]["n_tiles"] = len(tiles_data)
        else:
            data_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "dw_test")
            t0 = time.time()
            path = fetch(data_dir)
            summary["config"]["fetch_seconds"] = time.time() - t0
            print(f"archive verified in {time.time() - t0:.0f}s", flush=True)
            zf = zipfile.ZipFile(path)
            names = sorted(n for n in zf.namelist() if "/label_dw_" in n)
            if args.tiles:
                names = names[:args.tiles]
            summary["config"]["n_tiles"] = len(names)
            tiles_data = []
            t0 = time.time()
            for i, nm in enumerate(names):
                tiles_data.append((nm.split("/")[-1], *read_tile(zf, nm)))
                if i % 100 == 0:
                    print(f"  read {i + 1}/{len(names)} tiles ({time.time() - t0:.0f}s)", flush=True)
            summary["config"]["read_seconds"] = time.time() - t0
        # ---- windows
        acc = {k: [] for k in ("dec", "y", "margin", "top1p", "entropy", "bnd", "rar", "tile", "marked")}
        pix = {"top1p": [], "correct": [], "margin": []}                      # every PIXSUB-th annotated pixel, for calibration at the native resolution
        agree_published = []
        for t, (name, lulc, top1, p) in enumerate(tiles_data):
            q = windows(lulc, top1, p)
            ok = q["ok"]
            if not ok.any():
                continue
            b = boundary_indicator(q["dec"])
            r = rarity(q["dec"], ok)
            agree_published.append(float((q["dec"] == q["dw_top1"])[ok].mean()))
            for k, v in (("dec", q["dec"]), ("y", q["y"]), ("margin", q["margin"]), ("top1p", q["top1p"]),
                         ("entropy", q["entropy"]), ("bnd", b), ("rar", r), ("marked", q["marked_share"])):
                acc[k].append(np.asarray(v)[ok])
            acc["tile"].append(np.full(int(ok.sum()), t, np.int32))
            fin = np.isfinite(p).all(0) & (lulc > 0)
            if fin.any():
                srt = np.sort(p, axis=0)
                sel = np.zeros(fin.shape, bool); sel.reshape(-1)[::PIXSUB] = True
                m = fin & sel
                pix["top1p"].append(srt[-1][m]); pix["margin"].append((srt[-1] - srt[-2])[m])
                pix["correct"].append(((lulc - 1) == p.argmax(0))[m])
        A = {k: np.concatenate(v) for k, v in acc.items()}
        err = (A["dec"] != A["y"]).astype(np.float64)
        summary["results"]["windows"] = {"n_windows": int(len(err)), "n_tiles": len(agree_published),
                                         "accuracy_against_expert_consensus": float(1 - err.mean()),
                                         "window_decision_agrees_with_published_top1": float(np.mean(agree_published)),
                                         "mean_annotated_share_of_a_window": float(A["marked"].mean()),
                                         "class_share": {CLASSES[c]: float((A["y"] == c).mean()) for c in range(len(CLASSES))}}
        print(f"{len(err)} windows over {len(agree_published)} tiles | Dynamic World agrees with the expert consensus on {1 - err.mean():.4f} "
              f"| the pooled window decision reproduces its published Top-1 on {np.mean(agree_published):.4f}", flush=True)
        # ---- rankers
        sig = {MARGIN: -A["margin"], NAIVE: 1.0 - A["top1p"], ENT: A["entropy"], BND: A["bnd"], CTRL: A["rar"]}
        sig[LEX] = np.concatenate([boundary_first_score(sig[MARGIN][A["tile"] == t], A["bnd"][A["tile"] == t]) for t in np.unique(A["tile"])])
        order = np.concatenate([np.flatnonzero(A["tile"] == t) for t in np.unique(A["tile"])])
        lex = np.empty_like(sig[LEX]); lex[order] = sig[LEX]; sig[LEX] = lex
        r = score(sig, err, A["tile"])
        summary["results"]["ranking"] = r
        for k in sig:
            rows.append({"ranker": k, "pooled_excess_aurc": r["pooled_excess_aurc"][k], "lead_over_margin": r["tests"].get(k, {}).get("pooled_lead", 0.0),
                         "tiles_w": r["tests"].get(k, {}).get("w"), "tiles_l": r["tests"].get(k, {}).get("l"), "sign_p": r["tests"].get(k, {}).get("sign_p"),
                         "tied_values": r["tied_values"][k], **{f"capture {b}": v for b, v in r["capture"][k].items()}})
            print(f"  {k:32s} E-AURC {r['pooled_excess_aurc'][k]:.4f} | lead over margin {fmt(r['tests'].get(k, {}).get('pooled_lead', 0.0), '+.4f')} "
                  f"(tiles {r['tests'].get(k, {}).get('w', '-')}/{r['tests'].get(k, {}).get('l', '-')}, p={fmt(r['tests'].get(k, {}).get('sign_p'), '.2g')}) | ties {r['tied_values'][k]} | capture@5% {r['capture'][k]['0.05']:.3f}", flush=True)
        # ---- explanation layer and calibration
        cues = {"boundary": A["bnd"] > 0, "low_margin": top_fraction(-A["margin"], 0.2), "high_entropy": top_fraction(A["entropy"], 0.2)}
        summary["results"]["explanation"] = {k: {kk: vv for kk, vv in cue_enrichment(v, err, n_boot=0).items()} for k, v in cues.items()}
        Px = {k: np.concatenate(v) for k, v in pix.items()} if pix["top1p"] else None
        ece, reliability = expected_calibration_error(A["top1p"], 1 - err)
        summary["results"]["calibration"] = {"ece_of_top1_probability": ece, "mean_top1_probability": float(A["top1p"].mean()),
                                             "accuracy": float(1 - err.mean()), "overconfidence": float(A["top1p"].mean() - (1 - err.mean())),
                                             "reliability": [{"lo": a, "hi": b, "n": n, "confidence": c, "accuracy": d} for a, b, n, c, d in reliability]}
        if Px is not None:
            pece, prel = expected_calibration_error(Px["top1p"], Px["correct"].astype(np.float64))
            summary["results"]["calibration"]["per_pixel"] = {
                "n_pixels": int(len(Px["correct"])), "subsample": PIXSUB,
                "mean_top1_probability": float(Px["top1p"].mean()), "accuracy": float(Px["correct"].mean()),
                "overconfidence": float(Px["top1p"].mean() - Px["correct"].mean()), "ece": pece,
                "reliability": [{"lo": a, "hi": b, "n": n, "confidence": c, "accuracy": d} for a, b, n, c, d in prel]}
            print(f"calibration per pixel ({len(Px['correct'])} annotated pixels, every {PIXSUB}th): mean top-1 probability {Px['top1p'].mean():.4f} against accuracy {Px['correct'].mean():.4f}, "
                  f"overconfidence {Px['top1p'].mean() - Px['correct'].mean():+.4f}, ECE {pece:.4f}", flush=True)
        print("cues on the error windows: " + ", ".join(f"{k} {v['enrichment']:.2f}x ({100*v['share_errors']:.0f}% vs {100*v['share_correct']:.0f}%)" for k, v in summary["results"]["explanation"].items()), flush=True)
        print(f"calibration: mean published top-1 probability {A['top1p'].mean():.4f} against accuracy {1 - err.mean():.4f}, overconfidence {A['top1p'].mean() - (1 - err.mean()):+.4f}, ECE {ece:.4f}", flush=True)
        np.savez_compressed(os.path.join(hb.OUT, f"exp67_windows{suffix}.npz"),
                            dec=A["dec"].astype(np.int8), y=A["y"].astype(np.int8), margin=A["margin"].astype(np.float32),
                            top1p=A["top1p"].astype(np.float32), entropy=A["entropy"].astype(np.float32),
                            boundary=A["bnd"].astype(np.float32), rarity=A["rar"].astype(np.float32), tile=A["tile"])
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
    # ---- prereg
    try:
        R = summary["results"]["ranking"]; E = summary["results"]["explanation"]
        p1 = bool(R["tests"][CTRL]["pooled_lead"] >= MIN_LEAD and R["tests"][CTRL]["sign_p"] < 0.05)
        p2 = bool(R["tests"][NAIVE]["pooled_lead"] > 0 and R["tied_values"][MARGIN] < R["tied_values"][NAIVE])
        p3 = bool(E["boundary"]["enrichment"] > MIN_ENRICH)
        summary["prereg"] = {"P1": p1, "P2": p2, "P3": p3,
                             "P1_detail": {"control_lead": R["tests"][CTRL]["pooled_lead"], "tiles": [R["tests"][CTRL]["w"], R["tests"][CTRL]["l"]], "sign_p": R["tests"][CTRL]["sign_p"]},
                             "P2_detail": {"naive_lead": R["tests"][NAIVE]["pooled_lead"], "ties_margin": R["tied_values"][MARGIN], "ties_naive": R["tied_values"][NAIVE]},
                             "P3_detail": {"boundary_enrichment": E["boundary"]["enrichment"]},
                             "complete": not summary["failures"]}
        print(f"prereg: P1 {p1} (control lead {R['tests'][CTRL]['pooled_lead']:+.4f}, tiles {R['tests'][CTRL]['w']}/{R['tests'][CTRL]['l']}) | "
              f"P2 {p2} (naive lead {R['tests'][NAIVE]['pooled_lead']:+.4f}, ties {R['tied_values'][MARGIN]} vs {R['tied_values'][NAIVE]}) | "
              f"P3 {p3} (boundary {E['boundary']['enrichment']:.2f}x) | complete {summary['prereg']['complete']}", flush=True)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["prereg"] = {"P1": None, "P2": None, "P3": None, "complete": False}
    summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp67_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        fields = []
        for r_ in rows:
            fields += [k for k in r_ if k not in fields]
        with open(os.path.join(hb.OUT, f"exp67_dynamic_world{suffix}.csv"), "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=fields)
            w_.writeheader()
            w_.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
