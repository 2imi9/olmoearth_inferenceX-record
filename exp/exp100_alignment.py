"""exp100: does the map line up with its labels? exp98's and exp99's maps read at their label points moved by whole
pixels, to see where the map and the labels agree best. Preregistered in docs/plan/alignment.md; read that page first,
this file is the run.

    python exp/exp100_alignment.py --smoke                                  # synthetic inputs end to end
    python exp/exp100_alignment.py --exp98-scores DIR [DIR ...] --labels WINDOWS \\
        --exp99-scores DIR [DIR ...] --timesync CSV --request-geometry AI2.geojson   # the run, once the page is frozen

The sets (the page's names): T, exp98's training points on the map (in-sample; tests the record's pixel-finding code);
V, exp98's graded validation points; S, exp99's plots with 2017 input (STRICT primary, LENIENT reported). A point is
found on its map exactly as exp98 and exp99 found it (exp98's locate_all and read_points, which pick the grid that
covers the point's pixel); its 7 x 7 block of the ten bands is then read around that pixel in the same grid. Before
anything is estimated, the classes at the labelled pixels must give the errors exp98 and exp99 recorded (RECORDED), so
the check reads the pixels that were graded.

Writes exp/out/exp100_summary.json (exp100_summary_smoke.json with --smoke). Offsets are relative, in pixels, east and
south positive; nothing written holds a coordinate, a transform, a bound or a point's identity.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
for _p in (ROOT, EXP_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import exp89_finetuned_checkpoints as e89          # noqa: E402
import exp98_awf_deployment as e98                 # noqa: E402
import exp99_transfer as e99                       # noqa: E402
import timesync_awf_crosswalk as tsx               # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
PLAN = os.path.join(ROOT, "docs", "plan", "alignment.md")

SHIFT = 3                     # whole pixels in each direction: 7 x 7 = 49 positions
SIDE = 2 * SHIFT + 1
LEVEL = 0.90                  # the bootstrap percentile interval of each axis's estimate
HALF = 0.5                    # aligned: both intervals within +-HALF px; offset: one entirely beyond it
BOOT = 2000
SMOKE_BOOT = 200
SEED = 100
TIE = 1e-12                   # positions within this of the maximum tie; the one nearest the labelled pixel is taken
EPS = 1e-9

# what exp98 and exp99 recorded at the labelled pixels (exp/out/exp98_summary.json: train_in_sample and
# graded_validation.accuracy; exp/out/exp99_summary.json: rules); the run refuses a reading that does not give them
RECORDED = {"T": {"n": 783, "errors": 16}, "V": {"n": 259, "errors": 33},
            "S_strict": {"n": 271, "errors": 91}, "S_lenient": {"n": 271, "errors": 37}}
CALLS = ("aligned", "offset", "not determined")
WHAT = {"T": "exp98's training points on the map (in-sample: the model was fitted to these labels at these pixels, so "
             "the best position is where the record's code puts them; tests that code, not the imagery)",
        "V": "exp98's graded validation points (held out, the same experts: tests whether the map and AWF's labels "
             "line up where the model was not fitted)",
        "S_strict": "exp99's plots with 2017 input, under STRICT (primary): independent labels at 30 m plots, "
                    "reprojected from WGS84 (tests the reprojection and the plots' registration)",
        "S_lenient": "the same plots under LENIENT (reported beside STRICT)"}
CANNOT = ("A constant offset between Sentinel-2 and AWF's labels already present in the training data could be learned "
          "by the model and so be invisible in V.",
          "An offset that differs from point to point (geolocation noise) flattens the surface without moving its "
          "maximum: the check sees systematic offsets only.",
          "The map's classes are smooth over a few pixels, so offsets well under a pixel are not resolved; a TimeSync "
          "plot spans three pixels.")

SMOKE_PLANTED = ((0, 0), (2, -1))          # (south, east) offsets planted in the synthetic map
SMOKE_PLANTED_N = 200
SMOKE_PLANTED_SIZE = 160
SMOKE_PLANTED_BLOCK = 4
SMOKE_PLANTED_CLUSTERS = 20


def prereg_status(text=None):
    if text is None:
        with open(PLAN, encoding="utf-8") as f:
            text = f.read()
    return e89.prereg_status(text)


# ----------------------------------------------------------------------------- reading
def read_blocks(grids, gi, rows, cols, s=SHIFT):
    """(N, 10, 2s+1, 2s+1): the ten bands around each point's pixel in the grid read_points chose for it, NaN outside
    the grid or for a point outside every grid (gi -1)."""
    import rasterio
    from rasterio.windows import Window
    side = 2 * s + 1
    out = np.full((len(gi), e98.N_OUT, side, side), np.nan)
    for g, G in enumerate(grids):
        idx = np.flatnonzero(np.asarray(gi) == g)
        if not idx.size:
            continue
        with rasterio.open(G["scores"]) as src:
            for k in idx:
                out[k] = src.read(window=Window(int(cols[k]) - s, int(rows[k]) - s, side, side), boundless=True,
                                  fill_value=np.nan).astype(np.float64)
    return out


def surfaces(blocks, accept):
    """Per point and position: the probability on the classes the label accepts, whether the map's class (the argmax
    over all ten channels, as rslearn takes it) is accepted, and whether the whole block is covered. `accept` holds one
    collection of channel indices per point."""
    B = np.asarray(blocks, dtype=np.float64)
    n = len(B)
    cov = np.isfinite(B).all(1) & (np.nan_to_num(B) != 0).any(1)
    mask = np.zeros((n, e98.N_OUT), bool)
    for k, a in enumerate(accept):
        mask[k, sorted(int(c) for c in a)] = True
    Z = np.nan_to_num(B)
    score = np.einsum("nckl,nc->nkl", Z, mask.astype(np.float64))
    agree = mask[np.arange(n)[:, None, None], Z.argmax(1)].astype(np.float64)
    return score, agree, cov.all((1, 2)), cov


def _refine(a, b, c):
    """The vertex of the parabola through (-1, a), (0, b), (1, c), b the largest; 0 where the three make no peak."""
    den = a - 2.0 * b + c
    return 0.0 if den >= 0 else float((a - c) / (2.0 * den))


def peak(M, s=SHIFT, tie=TIE):
    """The maximum of a (2s+1) x (2s+1) mean surface: the integer position (ties: the nearest the labelled pixel, then
    the first in row order), refined on each axis by a parabola through it and its two neighbours, unrefined on an axis
    where it sits on the edge of the search."""
    M = np.asarray(M, dtype=np.float64)
    side = M.shape[0]
    cand = np.argwhere(M >= M.max() - tie)
    i, j = (int(v) for v in cand[np.argmin(((cand - s) ** 2).sum(1))])
    di = _refine(M[i - 1, j], M[i, j], M[i + 1, j]) if 0 < i < side - 1 else 0.0
    dj = _refine(M[i, j - 1], M[i, j], M[i, j + 1]) if 0 < j < side - 1 else 0.0
    return {"south_px": float(i - s + di), "east_px": float(j - s + dj), "best_south_px": i - s,
            "best_east_px": j - s, "at_edge": bool(i in (0, side - 1) or j in (0, side - 1))}


def _resampler(clusters, strata):
    """A function rng -> indices: the clusters resampled with replacement within each stratum."""
    cl = np.asarray(clusters).astype(str)
    st = np.zeros(len(cl), int).astype(str) if strata is None else np.asarray(strata).astype(str)
    groups = []
    for s_ in np.unique(st):
        ids = np.unique(cl[st == s_])
        groups.append([np.flatnonzero((cl == c) & (st == s_)) for c in ids])

    def draw(rng):
        return np.concatenate([g[i] for g in groups for i in rng.integers(0, len(g), len(g))])
    return draw


def estimate(score, clusters, strata=None, n_boot=BOOT, seed=SEED, level=LEVEL):
    """(the full sample's peak, {axis: [lo, hi]} percentile intervals, the share of resamples whose integer maximum
    is the labelled pixel, the share whose maximum is on the edge)."""
    full = peak(score.mean(0))
    draw = _resampler(clusters, strata)
    rng = np.random.default_rng(seed)
    south, east, at_pixel, at_edge = [], [], 0, 0
    for _ in range(n_boot):
        p = peak(score[draw(rng)].mean(0))
        south.append(p["south_px"])
        east.append(p["east_px"])
        at_pixel += (p["best_south_px"], p["best_east_px"]) == (0, 0)
        at_edge += p["at_edge"]
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    iv = {"south_px": [float(v) for v in np.quantile(south, q)], "east_px": [float(v) for v in np.quantile(east, q)]}
    return full, iv, at_pixel / n_boot, at_edge / n_boot


def call(full, iv, half=HALF):
    """The page's rule: aligned, offset or not determined."""
    if full["at_edge"]:
        return "offset"
    lo = [iv["south_px"][0], iv["east_px"][0]]
    hi = [iv["south_px"][1], iv["east_px"][1]]
    if all(a >= -half - EPS and b <= half + EPS for a, b in zip(lo, hi)):
        return "aligned"
    if any(a > half + EPS or b < -half - EPS for a, b in zip(lo, hi)):
        return "offset"
    return "not determined"


def set_record(name, score, agree, full, clusters, strata=None, n_boot=BOOT, seed=SEED, recorded=None):
    """One set's record. `recorded` ({n, errors}) is checked against the classes at the labelled pixels of all its
    points (full block or not) before anything else; a mismatch refuses the run."""
    n = len(score)
    errors = int(n - agree[:, SHIFT, SHIFT].sum()) if n else 0
    rec = {"what": WHAT.get(name, name), "n": int(n), "errors_at_labelled_pixel": errors}
    if recorded is not None:
        if (n, errors) != (recorded["n"], recorded["errors"]):
            raise ValueError(f"set {name}: {n} points and {errors} errors at the labelled pixels, where the record has "
                             f"{recorded['n']} and {recorded['errors']}: not the graded reading; nothing is estimated")
        rec["reproduces_record"] = True
    F = np.flatnonzero(full)
    rec["n_full_block"] = int(F.size)
    rec["n_excluded_partial_block"] = int(n - F.size)
    if F.size < 2:
        rec["call"] = "not determined"
        rec["note"] = "fewer than two points with a covered 7 x 7 block"
        return rec
    sc, ag = score[F], agree[F]
    cl = np.asarray(clusters)[F]
    st = None if strata is None else np.asarray(strata)[F]
    fp, iv, share_pixel, share_edge = estimate(sc, cl, st, n_boot, seed)
    M, A = sc.mean(0), ag.mean(0)
    rec.update({
        "mean_score_surface": np.round(M, 6).tolist(),
        "mean_agreement_surface": np.round(A, 6).tolist(),
        "surface_layout": f"[south][east], from -{SHIFT} to +{SHIFT} px; the labelled pixel in the middle",
        "estimate": fp, "interval": dict(iv, level=LEVEL),
        "bootstrap": {"n_resamples": n_boot, "share_maximum_at_labelled_pixel": share_pixel,
                      "share_maximum_on_edge": share_edge,
                      "resampled": "annotation tasks" if strata is None else "plots within country"},
        "n_clusters": int(np.unique(cl).size),
        "agreement_at_labelled_pixel": float(A[SHIFT, SHIFT]),
        "agreement_at_best_position": float(A[fp["best_south_px"] + SHIFT, fp["best_east_px"] + SHIFT]),
        "agreement_best_over_surface": float(A.max()),
        "call": call(fp, iv)})
    return rec


# ----------------------------------------------------------------------------- the two experiments' points
def exp98_sets(scores, labels):
    """{T, V}: (blocks, accept, clusters) for exp98's covered training and validation points, and a mapping record."""
    entries, _, _ = e98.find_outputs(scores)
    grids = [e98.open_grid(e) for e in entries]
    points, _ = e98.read_label_windows(labels)
    cands = e98.locate_all(points, grids)
    gi, rows, cols, how, off, P, _, _, _ = e98.read_points(grids, cands)
    cov = e98.covered(P)
    split = np.array([p["split"] for p in points], dtype=object)
    tasks = np.array([p["task"] for p in points], dtype=object)
    lab = np.array([p["label"] for p in points])
    blocks = read_blocks(grids, gi, rows, cols)
    out = {}
    for name, sp in (("T", "train"), ("V", "val")):
        K = np.flatnonzero(cov & (split == sp))
        out[name] = (blocks[K], [(int(c),) for c in lab[K]], tasks[K], None)
    used = cov & np.isin(split, ("train", "val"))
    mapping = {"n_grids": len(grids), "same_crs": int((how[used] == "same_crs").sum()),
               "reprojected": int((how[used] == "reprojected").sum()),
               "largest_offset_from_pixel_middle_px": float(np.nanmax(off[used])) if used.any() else None}
    return out, mapping


def exp99_sets(scores, timesync, request_geometry, fixed=e99.FIXED_RULE, geometry_sha256=tsx.AWF_GEOMETRY_SHA256,
               min_plots=e99.MIN_PLOTS):
    """{S_strict, S_lenient}: (blocks, accept, clusters, strata) for exp99's covered plots, and a mapping record."""
    P, _, _ = e99.select(timesync, request_geometry, None, geometry_sha256, min_plots, fixed)
    grids, _, _ = e99.read_map(scores)
    cands = e98.locate_all(e98.timesync_points(P), grids)
    gi, rows, cols, how, off, Pr, _, _, _ = e98.read_points(grids, cands)
    cov = e98.covered(Pr)
    K = np.flatnonzero(cov)
    lab = np.asarray(P["label"])[K]
    ctry = np.asarray(P["country"])[K]
    blocks = read_blocks(grids, gi, rows, cols)[K]
    out = {}
    for rule in ("strict", "lenient"):
        acc = tsx.accepted(rule)
        out[f"S_{rule}"] = (blocks, [acc[str(t)] for t in lab], np.arange(K.size), ctry)
    mapping = {"n_grids": len(grids), "crs": sorted({G["crs"].to_string() for G in grids}),
               "same_crs": int((how[K] == "same_crs").sum()), "reprojected": int((how[K] == "reprojected").sum())}
    return out, mapping


def analyse_sets(sets, n_boot=BOOT, recorded=RECORDED, log=print):
    recs = {}
    for name, (blocks, accept, clusters, strata) in sets.items():
        score, agree, full, _ = surfaces(blocks, accept)
        recs[name] = set_record(name, score, agree, full, clusters, strata, n_boot,
                                recorded=None if recorded is None else recorded[name])
        r = recs[name]
        log(f"{name}: {r['n']} points, {r['n_full_block']} with a full block, call {r['call']}"
            + (f", estimate {r['estimate']['south_px']:+.2f} south {r['estimate']['east_px']:+.2f} east"
               if "estimate" in r else ""))
    return recs


def analyse(exp98_scores, labels, exp99_scores, timesync, request_geometry, n_boot=BOOT, recorded=RECORDED,
            fixed=e99.FIXED_RULE, geometry_sha256=tsx.AWF_GEOMETRY_SHA256, min_plots=e99.MIN_PLOTS, log=print):
    t0 = time.time()
    s98, m98 = exp98_sets(exp98_scores, labels)
    s99, m99 = exp99_sets(exp99_scores, timesync, request_geometry, fixed, geometry_sha256, min_plots)
    commit, dirty = e89.git_state()
    rec = {"experiment": "exp100: exp98's and exp99's maps read at their label points moved by whole pixels; where do "
                         "the map and the labels agree best",
           "prereg_page": os.path.relpath(PLAN, ROOT), "utc": e89.utc_now(), "commit": commit, "dirty": dirty,
           "rule": {"shift_px": SHIFT, "level": LEVEL, "half_px": HALF, "calls": list(CALLS),
                    "score": "the probability on the classes the label accepts; agreement reported beside it"},
           "mapping": {"exp98": m98, "exp99": m99},
           "sets": analyse_sets({**s98, **s99}, n_boot, recorded, log),
           "cannot_show": list(CANNOT)}
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ----------------------------------------------------------------------------- smoke
def make_planted(root, south, east, n=SMOKE_PLANTED_N, size=SMOKE_PLANTED_SIZE, block=SMOKE_PLANTED_BLOCK, seed=SEED):
    """A synthetic map whose pixel (r, c) shows the truth at (r - south, c - east), and n points labelled with the
    truth at their own pixel: the map agrees with them best at (south, east). Returns (grids, points, labels,
    clusters); the points are exp98's point records, located by its code."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    rng = np.random.default_rng(seed)
    m = SHIFT + 2
    big = size + 2 * m
    coarse = rng.integers(0, 9, size=(big // block + 1, big // block + 1))
    truth = np.kron(coarse, np.ones((block, block), int))[:big, :big]
    shown = truth[m - south:m - south + size, m - east:m - east + size]
    scores = np.full((e98.N_OUT, size, size), 0.04, np.float32)
    scores[9] = 0.0
    np.put_along_axis(scores, shown[None], 0.68, axis=0)
    X0, Y0, epsg = 500000.0, 9000000.0, 32737              # synthetic; never written
    path = os.path.join(root, f"planted_{south}_{east}")
    os.makedirs(path, exist_ok=True)
    with rasterio.open(os.path.join(path, f"scores_{epsg}.tif"), "w", driver="GTiff", height=size, width=size,
                       count=e98.N_OUT, dtype="float32", crs=f"EPSG:{epsg}", nodata=float("nan"),
                       transform=from_origin(X0, Y0, 10, 10)) as dst:
        dst.write(scores)
    rc = rng.integers(m, size - m, size=(n, 2))
    crs = CRS.from_epsg(epsg)
    points = [{"crs": crs, "X": X0 + (c + 0.5) * 10, "Y": Y0 - (r + 0.5) * 10} for r, c in rc]
    labels = truth[rc[:, 0] + m, rc[:, 1] + m]
    return path, points, labels, np.arange(n) % SMOKE_PLANTED_CLUSTERS


def planted_set(root, south, east, n_boot):
    path, points, labels, clusters = make_planted(root, south, east)
    entries, _, _ = e98.find_outputs(path)
    grids = [e98.open_grid(e) for e in entries]
    gi, rows, cols, *_ = e98.read_points(grids, e98.locate_all(points, grids))
    score, agree, full, _ = surfaces(read_blocks(grids, gi, rows, cols), [(int(c),) for c in labels])
    return set_record(f"planted_{south}_{east}", score, agree, full, clusters, None, n_boot)


def smoke(out_dir=None, n_boot=SMOKE_BOOT, log=print):
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        planted = {f"{s}_{e}": dict(planted_set(tmp, s, e, n_boot), planted={"south_px": s, "east_px": e})
                   for s, e in SMOKE_PLANTED}
        for d in ("e98", "e99"):
            os.makedirs(os.path.join(tmp, d))
        p98 = e98.make_smoke_inputs(os.path.join(tmp, "e98"))
        p99 = e99.make_smoke_inputs(os.path.join(tmp, "e99"))
        rec = analyse(p98["scores"], p98["labels"], p99["scores"], p99["timesync"], p99["request_geometry"],
                      n_boot=n_boot, recorded=None, fixed=None, geometry_sha256=None, min_plots=e99.SMOKE_MIN_PLOTS,
                      log=log)
    rec["mode"] = "smoke: synthetic inputs; the recorded counts are not checked"
    rec["planted"] = planted
    out_dir = out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)
    e98.dump(rec, os.path.join(out_dir, "exp100_summary_smoke.json"))
    return rec


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true", help="synthetic inputs end to end")
    ap.add_argument("--exp98-scores", nargs="+", help="exp98's from-olmoearth part directories")
    ap.add_argument("--labels", help="the AWF windows root (holding spatial_split/), exp98's --labels")
    ap.add_argument("--exp99-scores", nargs="+", help="exp99's from-olmoearth part directories")
    ap.add_argument("--timesync", help="the East Africa TimeSync yearly CSV (checked against its pin by exp99's select)")
    ap.add_argument("--request-geometry", help="Ai2's AWF prediction_request_geometry.geojson (its sha256 is checked)")
    ap.add_argument("--out-dir", default=OUT)
    a = ap.parse_args(argv)
    if a.smoke:
        rec = smoke(a.out_dir)
        print(json.dumps({k: v["call"] for k, v in rec["planted"].items()}))
        return 0
    if not (a.exp98_scores and a.labels and a.exp99_scores and a.timesync and a.request_geometry):
        ap.error("--exp98-scores, --labels, --exp99-scores, --timesync and --request-geometry are required outside "
                 "--smoke")
    status = prereg_status()
    if status != "frozen":
        print(f"refused: {os.path.relpath(PLAN, ROOT)} is {status}, not frozen; only --smoke runs before it is frozen")
        return 3
    rec = analyse(a.exp98_scores, a.labels, a.exp99_scores, a.timesync, a.request_geometry)
    rec["prereg_status"] = status
    os.makedirs(a.out_dir, exist_ok=True)
    e98.dump(rec, os.path.join(a.out_dir, "exp100_summary.json"))
    print(json.dumps({k: v["call"] for k, v in rec["sets"].items()}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001  (a job log needs the traceback, and the exit code must fail the job)
        traceback.print_exc()
        sys.exit(1)
