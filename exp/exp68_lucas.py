#!/usr/bin/env python
"""exp68: the protocol against ground observation. LUCAS Copernicus 2022, the first reference here that never saw a pixel.

Why. Every reference this repository has graded against is image interpretation. Sen1Floods11 and GEOID-Flood are
analysts drawing on the same Sentinel imagery the model reads; DFC2020's "hand-labelled" test set is an iterated Earth
Engine random forest (exp66); Dynamic World's consensus is expert annotators labelling Sentinel-2 chips (exp67); MADOS
and PASTIS arrive as Ai2's embeddings. That shared provenance is this project's oldest unresolved worry, stated in exp18
and tested only indirectly since: when the reference is drawn from the pixels the model reads, reference error and model
error are correlated, because a hazy pixel or an ambiguous edge misleads the annotator and the network the same way and
in the same direction. A window we score as "captured error" may be a window where the annotator, not the model, was
wrong. Nothing in exp18, exp23 or exp66 can separate the two, because every reference in them is photointerpreted.

LUCAS breaks the correlation. It is an in-situ survey: a surveyor stood on the ground, at the point, and recorded the
land cover in front of them. The reference is causally independent of the imagery. If the margin ranking still finds
errors against a grader that never saw a pixel, the effect is not annotator agreement. If it does not, then a large part
of what this repository has measured is annotator agreement, and that is the headline, not a footnote.

LUCAS carries two further instruments that no other testbed here has.

  The provenance contrast. Of the 137,966 Copernicus polygons, 131,195 were observed in the field and 6,771 were
  photo-interpreted in the field, under one protocol, one class nomenclature and one year. So the same dataset holds a
  ground-observed arm and an image-interpreted arm, and exp18's caveat can be tested on reference *provenance* directly
  rather than on the reference *resolution* proxy exp66 had to use. The two arms are not exchangeable: photo-interpreted
  points concentrate in inaccessible terrain, 3,242 of them woodland, so the contrast is run only within matched
  (class x country) strata and is reported with that confound named.

  The filter contrast. LUCAS ships `survey_homplot_fills_extwin`, the surveyor's own record of whether the homogeneous
  plot fills the 51 m extended window: Yes for 64,788 polygons, No for 73,178. Published LUCAS practice filters to large
  homogeneous units, which deletes exactly the mixed and edge units where the boundary and low-margin cues concentrate.
  Here the filter is a published field, not one we invent, so the effect of the convention can be measured rather than
  argued.

Design. The graded unit is the polygon, not the window, and that is forced by the data: the median polygon is 2,746 sqm
and a 4-px window is 1,600 sqm, so most polygons hold at most one window and a tenth are smaller than a single 10 m
pixel. One polygon carries one surveyed class, which is also the unit LUCAS is designed to support. Each polygon gets the
4-px window with the largest overlap with it (ties to the window nearest the centroid); that window's decision is the
model's answer at that place and its margin is the model's own confidence there, with the 5 x 5 window neighbourhood kept
so the boundary indicator can be computed from the model's own decision map.

  Sample. Stratified by (level-1 class x observation type), sixteen strata, with the photo-interpreted strata
  deliberately oversampled so the provenance contrast has power. Inclusion probabilities are therefore known exactly and
  set by us, which is what makes the Horvitz-Thompson estimator exact here rather than approximate.
  Imagery. Sentinel-2 L2A from the Planetary Computer, 64 px at 10 m on the polygon centroid, the encoder reading the
  60 px crop for a 15 x 15 grid of 4-px windows. Two acquisitions per polygon: the least-cloudy scene within 30 days of
  the survey (near) and the least-cloudy scene within five months but at least 90 days away (far). Digital numbers are
  harmonised to the pre-baseline-04.00 convention by subtracting the 1,000 BOA offset whenever the item's processing
  baseline is 04.00 or later, which is every 2022 scene; the encoder's normaliser and every other testbed here use the
  older convention, and the probe measured a floor near 1,000 DN on all twelve sampled chips before the correction.
  Model. OlmoEarth v1 Base, frozen, fp32; a linear probe over the eight LUCAS level-1 classes fitted on the centre-window
  embeddings of field-surveyed polygons in the fit regions only, two seeds for the head-draw floor. NUTS2 regions are
  split into fit and report halves, so no region contributes to both.

  Part A, ranking. The margin (top-1 minus top-2 window probability) against predictive entropy, one minus top-1, the
  boundary-first order, the neighbourhood-disagreement fraction on its own, and four controls that see no model
  confidence: two deployable ones, the within-window pixel variance and the rarity of the predicted class, and two
  oracle-side ones an operator would not hold at inference time, the polygon's area and the chosen window's purity.
  P1 is judged on the deployable pair; the oracle pair is the secondary bar P1b. The disagreement fraction is scored as
  a ranker and not as a control, because it reads the model's argmax at nine windows and so is a model output, but it
  uses no confidence at all and an audit found it competitive, which is worth the record knowing. Every ranker is also
  tested against the margin per region and by cluster bootstrap, which is exploratory and labelled as such. The
  per-region sign tests use the design-weighted E-AURC, the estimator the preregistrations name; the unweighted version
  is recorded beside it because the two differ.
  Part B, two dates. The same polygon, the same model, two acquisitions: oe_inferencex.compare on the near and far
  decisions, with the surveyed class saying which side is right. This is the difference measurement the package exists
  for, run for the first time with a ground-observed arbiter.
  Part C, provenance and filters. Every part-A number recomputed on the field-surveyed and photo-interpreted arms within
  matched strata, and on the filtered population, the unfiltered one, and the units the published filter deletes, which
  is the disjoint contrast an interval can be put on. Two corrections an audit forced. The boundary-first order works by
  coarsening the suspicion order into two blocks, and that coarsening has a cost or a benefit of its own set by its fire
  rate and by how much headroom the suspicion order has left, nothing to do with boundaries; so each arm's gain is
  reported beside two rate-matched nulls, a random flag and the top slice of pixel variance, and only the excess over
  the random null is evidence about boundaries. And error capture at a fixed budget is bounded above by the budget over
  the error rate, so two subsets with different error rates do not have the same attainable maximum: the capture
  comparison is reported with both ceilings, and the ceiling-free readings, a design-weighted AUROC and an odds ratio,
  carry the direction.
  Two covariates are carried through because they are the obvious ways this could be fooled. The purity of the chosen
  window, since the sample's median polygon covers only a third of one, so part A is also reported within four purity
  bands: label noise attenuates every ranker equally and cannot manufacture a lead, but the record should show where the
  lead lives. And the cloud state of the chosen window from the scene classification band, since a cloudier far
  acquisition would change decisions for reasons that are not phenology, so part B's class-dependent change rate is
  reported again on the polygons whose window is cloud-clear in both acquisitions.
  Estimators. Everything is reported twice, as a naive unweighted count and as a Horvitz-Thompson design-weighted share
  over the polygon population, with a cluster bootstrap over NUTS2 regions. The weight is the stratum's population over
  the number of polygons actually in hand, not over the number drawn: 217 of 12,073 returned no usable scene and that
  loss is not missing at random across Europe, so the intended-sample weights are recorded per stratum beside the
  adjusted ones. Polygons whose graded window is entirely scene-classification nodata are dropped rather than scored,
  because a window with no observation cannot be graded, and the count is recorded. The weighted estimators are checked against
  the package's unweighted ones under uniform weights to 1e-9 in the smoke test.

Preregistered (one-sided):
  P1  the ranking survives a reference that never saw the imagery: on field-surveyed polygons in held-out regions the
      margin's excess AURC is at least 0.01 below the best DEPLOYABLE no-model control, pooled, and lower on more NUTS2
      regions than not (sign test p < 0.05), under the design-weighted estimator. Deployable means computable at
      inference time from the imagery and the decision alone: the within-window pixel variance and the rarity of the
      predicted class. The polygon's area and the chosen window's purity are reference-side metadata an operator does
      not hold, so they are scored as a separate and harder secondary bar (P1b), reported either way; losing to an
      oracle control says the units are hard, not that the ranking fails, and conflating the two would let a bar no
      deployment can face decide the protocol's fate.
  P2  the standard homogeneity filter understates the tool: error capture at a 10% budget is higher on the unfiltered
      population than on the filtered one (homogeneous plot fills the window and area >= 5,000 sqm), and the boundary
      cue's enrichment on error polygons is higher unfiltered.
  P3  exp18's caveat holds on provenance, not only on resolution: within matched (class x country) strata the
      boundary-first order's lead over the margin is larger on photo-interpreted polygons than on field-surveyed ones.
  P1b the harder bar, reported whichever way it falls: the margin also beats the best oracle-side control, polygon area
      or window purity, by 0.01 weighted. This is not a falsification of the protocol, because no deployment holds
      these; it says whether the model's own confidence adds anything over simply knowing how small and mixed the unit
      is, which is the question a sceptic would ask next.
  P4  the finding survives the sampling design: the part-A lead of P1 keeps its sign and stays at least 0.005 when
      estimated with Horvitz-Thompson weights, with a cluster bootstrap interval over NUTS2 excluding zero.
  P5  the two-date difference is class-dependent in the direction land cover implies: the share of polygons whose
      decision changes between the near and far acquisition is at least twice as high for cropland (B) as for the two
      classes whose cover is stable within a year, artificial land (A) and woodland (C).
  Falsification. P1 fails if the margin does not beat a no-model control against ground observation, which would say the
  ranking this repository recommends works only where the grader shares the model's inputs, and would put every capture
  number here in question; that is the most damaging result this project can produce and it is to be reported first if it
  happens. P2 fails if capture is equal or lower unfiltered, which would say the advantage comes from easy homogeneous
  units rather than the mixed ones the convention deletes. P3 fails at a zero or negative gap, retiring the caveat on
  this axis. P4 fails if the lead is an artifact of ignoring unequal inclusion probabilities, which would be a
  methodological error in every review-set number reported here on a sampled reference. P5 fails if the change rate is
  flat across classes, which would say the difference is noise rather than phenology.
  Stated predictions, not tested: overall level-1 accuracy between 0.55 and 0.75, well under the dense-testbed numbers
  here, because a linear probe on frozen features with one 40 m window per polygon is a hard setting; water (G) and
  artificial (A) the most accurate classes and shrubland (D) the least.

Caveats carried into the record. The surveyed class describes the surveyor's plot, and the window is 40 m across, so a
polygon smaller than the window carries a label for only part of what the model saw; the purity of the chosen window is
recorded per polygon and is the covariate for it, never silently filtered away. LUCAS's own design weights over EU area
are not shipped in this file, so the population estimand here is the LUCAS Copernicus polygon population, not EU area;
the Horvitz-Thompson correction is exact for our own stratified subsample of that population and is not claimed to be
more. The survey ran 2022-03-11 to 2023-06-26 and imagery is matched per polygon to its own date. Level-1 classes are
coarse: a cropland error between two crops is invisible here, which makes this a weaker test of the model than PASTIS
and a stronger test of the reference. 167 of every 3,000 points are photo-interpreted and they sit in harder terrain, so
the part-C contrast is within matched strata and is still confounded by anything terrain drives beyond class and country. The probe is fitted on the
class-balanced stratified sample, so its class prior does not match the population: the design-weighted error rate is
therefore higher than a probe fitted on the population would give, and both rates are recorded. That mismatch shifts the
level of every ranker equally, because all of them are scored on identical units.

Inputs: the LUCAS Copernicus 2022 GeoPackage (1.12 GB, CC BY 4.0, JRC) and Sentinel-2 L2A from the Planetary Computer.
Run with ~/olmoearth_inferenceX/.venv, which carries rasterio, pyogrio, the STAC clients and the encoder.
Stages: --stage fetch (cpu partition, threaded, caches chips to scratch), --stage analyze (GPU, encodes and scores), all.
Outputs: exp/out/exp68_summary.json, exp/out/exp68_lucas.csv, exp/out/exp68_masks.npz.
--smoke: synthetic polygons and chips, no network, _smoke outputs.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))

from oe_inferencex import metrics, stats  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.explain import cue_enrichment  # noqa: E402

GPKG = "/scratch/qi_zim_neu/olmoearth_inferenceX/hf/lucas/l2022_survey_cop_radpoly_attr.gpkg"
CHIPDIR = "/scratch/qi_zim_neu/olmoearth_inferenceX/hf/lucas/chips"
OUT = os.path.join(EXP_DIR, "out")
LAYER = "l2022_cop_radpoly_radqs_allatr_wcol"

CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H"]
CLASS_NAME = {"A": "artificial", "B": "cropland", "C": "woodland", "D": "shrubland",
              "E": "grassland", "F": "bare", "G": "water", "H": "wetland"}
N_CLASS = len(CLASSES)
SIZE, CROP, PATCH = 64, 60, 4
G = CROP // PATCH                      # 15 windows across
NB = 5                                 # stored neighbourhood, 5 x 5 windows around the chosen one
BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
BOA_OFFSET = 1000                      # processing baseline 04.00+ adds this; every other testbed here predates it
FIELD, PHOTO = "field", "photo"
PHOTO_CAP, FIELD_CAP = 900, 1000
BUDGETS = (0.05, 0.10, 0.20)
SHARD = 500
SCL_CLEAR = (4, 5, 6, 7, 11)      # vegetation, bare, water, unclassified, snow; not nodata, shadow, cloud or cirrus
PURITY_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))
POINT_WIN = (SIZE // 2) // PATCH   # the chip is centred on the survey point at pixel 32, and the 60 px crop is taken
                                   # from the top-left, so the point sits in window 8 of 15 and NOT in the grid centre 7


# ----------------------------------------------------------------------------- design-weighted estimators
# The design-weighted estimators now live in the package, where they are unit-tested against their unweighted
# counterparts; these names are kept because the recorded test and three commits of history refer to them.
w_aurc = metrics.weighted_aurc
w_excess_aurc = metrics.weighted_excess_aurc
w_mean = metrics.weighted_mean


def w_capture(u, e, w, budgets=BUDGETS):
    return metrics.weighted_capture_at_budget(u, e, w, budgets)


def cluster_boot(fn, groups, n_boot=2000, seed=0):
    """Cluster bootstrap over NUTS2 regions: resample regions with replacement, recompute fn on the pooled draw."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[k] for k in pick])
        v = fn(sel)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n": 0}
    v = np.sort(np.asarray(vals))
    return {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975)), "n": len(v)}


def region_sign_test(a, b, groups, min_n=8, weights=None):
    """One-sided sign test over NUTS2 regions on the per-region E-AURC gap (a better than b => positive win).

    `weights` makes each region's vote use the design-weighted E-AURC, which is what the preregistrations mean when they
    say "under the design-weighted estimator". Left unweighted the votes ignore Horvitz-Thompson weights that vary from
    1.0 to 42.0 inside a single region, which is not the estimator that was preregistered; both are recorded so the
    difference is visible."""
    wins = losses = 0
    per, dropped_small, dropped_degenerate = [], 0, 0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < min_n:
            dropped_small += 1
            continue
        e = a["err"][m]
        if e.sum() == 0 or e.sum() == m.sum():
            dropped_degenerate += 1
            continue
        if weights is None:
            ga, gb = metrics.excess_aurc(a["u"][m], e), metrics.excess_aurc(b["u"][m], e)
        else:
            w = weights[m]
            ga, gb = w_excess_aurc(a["u"][m], e, w), w_excess_aurc(b["u"][m], e, w)
        per.append((str(g), gb - ga))
        if gb > ga:
            wins += 1
        elif gb < ga:
            losses += 1
    p = stats.sign_test(wins, losses, alternative="greater")
    return {"wins": wins, "losses": losses, "p": float(p), "n_regions": len(per),
            "regions_dropped_too_small": dropped_small, "regions_dropped_degenerate": dropped_degenerate,
            "estimator": "design-weighted" if weights is not None else "unweighted"}


# ----------------------------------------------------------------------------- sample design
def strata_key(letter, obs_type):
    return f"{letter}:{PHOTO if str(obs_type).startswith('Photo') else FIELD}"


def build_sample(df, seed=0):
    """Stratified subsample by (level-1 class x observation type) with exact inclusion probabilities.

    The photo-interpreted strata are oversampled on purpose, which is why the Horvitz-Thompson weights matter and why
    P4 is worth preregistering: naive counts over this sample would over-represent photo-interpreted woodland by an
    order of magnitude."""
    rng = np.random.default_rng(seed)
    keys = np.array([strata_key(l, o) for l, o in zip(df["letter_group"], df["survey_obs_type"])])
    take, meta = [], {}
    for k in sorted(set(keys)):
        pool = np.flatnonzero(keys == k)
        cap = PHOTO_CAP if k.endswith(PHOTO) else FIELD_CAP
        n = int(min(cap, len(pool)))
        pick = rng.choice(pool, size=n, replace=False)
        take.append(pick)
        meta[k] = {"N_stratum": int(len(pool)), "n_sampled": n,
                   "inclusion_probability": n / len(pool), "ht_weight": len(pool) / n}
    take = np.sort(np.concatenate(take))
    return take, meta


def region_split(nuts2, salt=""):
    """Two halves by a stable hash of the NUTS2 code, so no region contributes to both.

    `salt` gives an independent split of the same regions, which is what the inner validation split needs: hashing the
    fit regions with the outer salt would put every one of them on the same side again."""
    h = np.array([int(hashlib.md5((salt + str(r)).encode()).hexdigest()[:8], 16) % 2 for r in nuts2])
    return h == 0, h == 1


# ----------------------------------------------------------------------------- imagery
_TLS = None


def _sign_catalog():
    """One signed catalog per worker thread: pystac_client holds a requests session and is not thread-safe."""
    global _TLS
    import threading
    import planetary_computer
    import pystac_client
    if _TLS is None:
        _TLS = threading.local()
    if getattr(_TLS, "cat", None) is None:
        _TLS.cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                             modifier=planetary_computer.sign_inplace)
    return _TLS.cat


def pick_items(cat, lon, lat, day):
    """Least-cloudy scene within 30 days of the survey (near) and within five months but >= 90 days away (far)."""
    import datetime as dt
    d0 = dt.date(int(day[:4]), int(day[5:7]), int(day[8:10]))
    lo, hi = d0 - dt.timedelta(days=150), d0 + dt.timedelta(days=150)
    items = list(cat.search(collections=["sentinel-2-l2a"],
                            intersects={"type": "Point", "coordinates": [lon, lat]},
                            datetime=f"{lo.isoformat()}/{hi.isoformat()}",
                            query={"eo:cloud_cover": {"lt": 20}}).items())
    if not items:
        return None, None
    def gap(it):
        s = it.properties["datetime"][:10]
        return abs((dt.date(int(s[:4]), int(s[5:7]), int(s[8:10])) - d0).days)
    near = [it for it in items if gap(it) <= 30]
    far = [it for it in items if gap(it) >= 90]
    cc = lambda it: it.properties.get("eo:cloud_cover", 100.0)
    return (min(near, key=cc) if near else None), (min(far, key=cc) if far else None)


def read_chip(item, lon, lat, geom_3035):
    """(12, 64, 64) harmonised DN, the SCL band and the polygon's sub-pixel coverage on the same grid."""
    import rasterio
    import pyproj
    from rasterio.enums import Resampling
    from rasterio.windows import Window, bounds as win_bounds, from_bounds
    from rasterio.features import rasterize
    from shapely.ops import transform as sh_transform

    with rasterio.open(item.assets["B04"].href) as src:
        crs = src.crs
        tr = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = tr.transform(lon, lat)
        row, col = src.index(x, y)
        win = Window(col - SIZE // 2, row - SIZE // 2, SIZE, SIZE)
        bnds = win_bounds(win, src.transform)
        chip_tr = rasterio.windows.transform(win, src.transform)
    stack = []
    for b in BANDS:
        with rasterio.open(item.assets[b].href) as s:
            w = from_bounds(*bnds, transform=s.transform)
            stack.append(s.read(1, window=w, out_shape=(SIZE, SIZE), resampling=Resampling.bilinear,
                                boundless=True, fill_value=0))
    with rasterio.open(item.assets["SCL"].href) as s:
        w = from_bounds(*bnds, transform=s.transform)
        scl = s.read(1, window=w, out_shape=(SIZE, SIZE), resampling=Resampling.nearest,
                     boundless=True, fill_value=0)
    img = np.stack(stack).astype(np.float32)
    if str(item.properties.get("s2:processing_baseline", "00.00")) >= "04.00":
        img = np.clip(img - BOA_OFFSET, 0, None)          # back to the pre-offset convention the encoder was normalised on
    # sub-pixel coverage: rasterise at 4x and average into the 10 m grid
    to_crs = pyproj.Transformer.from_crs("EPSG:3035", crs, always_xy=True).transform
    geom = sh_transform(to_crs, geom_3035)
    fine = rasterize([(geom, 1)], out_shape=(SIZE * 4, SIZE * 4),
                     transform=chip_tr * rasterio.Affine.scale(0.25, 0.25), dtype="uint8")
    cover = fine.reshape(SIZE, 4, SIZE, 4).mean(axis=(1, 3)).astype(np.float32)
    return img.astype(np.uint16), scl.astype(np.uint8), cover


def window_coverage(cover):
    """Per 4-px window coverage over the 15 x 15 grid the encoder emits from the 60 px crop."""
    c = cover[:CROP, :CROP].reshape(G, PATCH, G, PATCH).mean(axis=(1, 3))
    return c


def choose_window(cov_win):
    """The window with the largest overlap; ties, and the no-overlap fallback, to the window holding the survey point.

    The anchor is POINT_WIN, not the grid centre. read_chip centres the 64 px chip on the survey point, so the point is
    at chip pixel (32, 32); the encoder reads the 60 px crop taken from the TOP-LEFT, so the point lands at crop pixel
    (32, 32), which is window 32 // 4 = 8 of 15. Using G // 2 = 7 put the fallback one window (40 m) away from the
    point, silently grading a twelfth of the polygons on ground the surveyor never looked at."""
    best = cov_win.max()
    if best <= 0:
        return POINT_WIN, POINT_WIN, 0.0
    cand = np.argwhere(cov_win >= best - 1e-9)
    d = ((cand[:, 0] - POINT_WIN) ** 2 + (cand[:, 1] - POINT_WIN) ** 2)
    r, q = cand[int(np.argmin(d))]
    return int(r), int(q), float(best)


# ----------------------------------------------------------------------------- stage 1, fetch
def fetch_stage(args):
    import pyogrio
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(CHIPDIR, exist_ok=True)
    cols = ["point_id", "point_nuts0", "nuts2", "point_lat", "point_long", "survey_date", "survey_obs_type",
            "letter_group", "poly_area_sqm", "survey_homplot_fills_extwin"]
    t0 = time.time()
    df = pyogrio.read_dataframe(GPKG, layer=LAYER, columns=cols, read_geometry=False)
    df = df[(df["poly_area_sqm"].astype(float) > 0) & (df["letter_group"].isin(CLASSES))].reset_index(drop=True)
    take, strata = build_sample(df, seed=args.seed)
    print(f"population {len(df)}, sampled {len(take)} over {len(strata)} strata in {time.time()-t0:.1f}s", flush=True)
    sub = df.iloc[take].reset_index(drop=True)
    assert sub["point_id"].is_unique, ("duplicate point_id in the sample: load_shards and part B key metadata by "
                                       "point_id, so a duplicate would attach another polygon's class and weight")
    g = pyogrio.read_dataframe(GPKG, layer=LAYER, columns=["point_id"], read_geometry=True)
    if g["point_id"].duplicated().any():
        print(f"warning: {int(g['point_id'].duplicated().sum())} duplicate point_id, keeping the first", flush=True)
        g = g.drop_duplicates(subset="point_id", keep="first")
    geoms = g.set_index("point_id")["geometry"]
    with open(os.path.join(OUT, "exp68_strata.json"), "w") as fh:
        json.dump(strata, fh, indent=1)

    from olmoearth_pretrain.data.constants import Modality
    assert BANDS == list(Modality.SENTINEL2_L2A.band_order), \
        f"band order drifted: encoder wants {list(Modality.SENTINEL2_L2A.band_order)}, this fetches {BANDS}"
    _sign_catalog()
    todo = [i for i in range(0, len(sub), SHARD)
            if not _shard_ok(os.path.join(CHIPDIR, f"shard_{i:06d}.npz"))]
    print(f"{len(todo)} shards to fetch of {int(np.ceil(len(sub)/SHARD))}", flush=True)
    for s0 in todo:
        rows = sub.iloc[s0:s0 + SHARD]
        t1 = time.time()

        def one(rec):
            i, r = rec
            try:
                lon, lat = float(r["point_long"]), float(r["point_lat"])
                near, far = pick_items(_sign_catalog(), lon, lat, str(r["survey_date"])[:10])
                if near is None:
                    return None
                g = geoms.loc[r["point_id"]]
                out = {"point_id": int(r["point_id"])}
                for tag, it in (("near", near), ("far", far)):
                    if it is None:
                        continue
                    img, scl, cover = read_chip(it, lon, lat, g)
                    out[f"img_{tag}"] = img
                    out[f"scl_{tag}"] = scl
                    out[f"cov_{tag}"] = cover
                    out[f"id_{tag}"] = it.id
                    out[f"date_{tag}"] = it.properties["datetime"][:10]
                    out[f"cloud_{tag}"] = float(it.properties.get("eo:cloud_cover", np.nan))
                return out
            except Exception as exc:                       # a single unreachable asset must not kill a shard
                return {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}

        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            got = list(ex.map(one, list(rows.iterrows())))
        keep = [g for g in got if g and "img_near" in g]
        errs = collections.Counter(g["error"].split(":")[0] for g in got if g and "error" in g)
        _write_shard(os.path.join(CHIPDIR, f"shard_{s0:06d}.npz"), keep, rows)
        print(f"  shard {s0:06d}: {len(keep)}/{len(rows)} chips in {time.time()-t1:.0f}s"
              f"{' errors ' + str(dict(errs)) if errs else ''}", flush=True)


def _shard_ok(path):
    """exp58's lesson: a marker is not proof, the file must open and carry rows."""
    if not os.path.exists(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return int(z["n"]) >= 0
    except Exception:
        return False


def _write_shard(path, keep, rows):
    meta = {c: rows[c].to_numpy() for c in rows.columns}
    pack = {"n": len(keep), "meta_cols": np.array(list(meta.keys()))}
    for c, v in meta.items():
        pack[f"meta_{c}"] = v.astype(str) if v.dtype == object else v
    for tag in ("near", "far"):
        have = [g for g in keep if f"img_{tag}" in g]
        pack[f"pid_{tag}"] = np.array([g["point_id"] for g in have], dtype=np.int64)
        if have:
            pack[f"img_{tag}"] = np.stack([g[f"img_{tag}"] for g in have])
            pack[f"scl_{tag}"] = np.stack([g[f"scl_{tag}"] for g in have])
            pack[f"cov_{tag}"] = np.stack([g[f"cov_{tag}"] for g in have])
            pack[f"date_{tag}"] = np.array([g[f"date_{tag}"] for g in have])
            pack[f"cloud_{tag}"] = np.array([g[f"cloud_{tag}"] for g in have], dtype=np.float32)
            pack[f"scene_{tag}"] = np.array([g[f"id_{tag}"] for g in have])
    np.savez_compressed(path + ".tmp.npz", **pack)
    os.replace(path + ".tmp.npz", path)


# ----------------------------------------------------------------------------- stage 2, encode
def embed_chips(model, imgs, dates, rc, batch=32):
    """(N, 12, 64, 64) harmonised DN, (N, 3) day/month0/year and the chosen window -> (N, 5, 5, D) fp16 tokens.

    Same encoder path as exp18.embed, with two differences. The timestamp is the scene's own date rather than exp18's
    fixed one, because part B compares two acquisitions of one place and the date must be the thing that differs. And
    only the 5 x 5 neighbourhood of the chosen window is kept, per batch: the full 15 x 15 stack for this many chips is
    4.2 GB and nothing downstream reads outside the neighbourhood."""
    import torch
    import exp18_sen1floods_expert as exp18
    out, clamped = [], np.zeros(len(imgs), dtype=bool)
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch][:, :, :CROP, :CROP].transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
        x = exp18._norm.normalize(exp18.Modality.SENTINEL2_L2A, x)
        b = x.shape[0]
        sample = exp18.MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=exp18.DEV),
            sentinel2_l2a_mask=torch.ones((b, CROP, CROP, 1, 3), device=exp18.DEV) * exp18.MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor(dates[i:i + batch], dtype=torch.long, device=exp18.DEV)[:, None, :])
        with torch.no_grad():
            o = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a
        f = o.mean(dim=[3, 4]).half().cpu().numpy()                      # (b, 15, 15, D)
        keep, cl = neighbourhood(f, rc[i:i + b])
        clamped[i:i + b] = cl
        out.append(keep)
    return np.concatenate(out), clamped


def neighbourhood(feats, rc):
    """(N, 15, 15, D) -> (N, 5, 5, D) around each chosen window, edge-clamped, plus the clamp flag."""
    n, _, _, d = feats.shape
    half = NB // 2
    out = np.zeros((n, NB, NB, d), dtype=feats.dtype)
    clamped = np.zeros(n, dtype=bool)
    for i in range(n):
        r, q = rc[i]
        rs = np.clip(np.arange(r - half, r + half + 1), 0, G - 1)
        qs = np.clip(np.arange(q - half, q + half + 1), 0, G - 1)
        clamped[i] = (r - half < 0) or (r + half >= G) or (q - half < 0) or (q + half >= G)
        out[i] = feats[i][np.ix_(rs, qs)]
    return out, clamped


def fit_probe(X, y, seed=0, epochs=80, lr=2e-3, wd=1e-4, batch=256):
    """A linear probe over the eight level-1 classes, fp32, AdamW with a cosine schedule (e51's recipe, one window)."""
    import math
    import torch
    import exp18_sen1floods_expert as exp18
    dev = exp18.DEV
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(X.shape[1], N_CLASS).to(dev).float()
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=dev)
    yt = torch.tensor(np.asarray(y, dtype=np.int64), device=dev)
    n = len(y)
    for ep in range(epochs):
        order = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            torch.nn.functional.cross_entropy(lin(Xt[idx]), yt[idx]).backward()
            opt.step(); opt.zero_grad()
        for pg in opt.param_groups:
            pg["lr"] = 1e-5 + 0.5 * (lr - 1e-5) * (1 + math.cos(math.pi * (ep + 1) / epochs))
    return lin.eval()


WD_GRID = (1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 10000.0)
EPOCH_GRID = (10, 30, 80)


def tune_probe(X, y, groups, seed=0):
    """Pick weight decay and epochs on NUTS2 regions held out of the fit set, then refit on all of it.

    The first run of this experiment fitted 6,152 probe parameters on 3,037 polygons and reached 0.943 accuracy on its
    own fit set against 0.480 on the held-out regions. A probe that has memorised its fit set is not a fair model to
    audit, and part of that gap is genuine spatial shift across Europe rather than plain overfitting, so the validation
    split is by region and not by row: a within-region split would not see the shift the report split imposes.
    The grid is deliberately wide, to 10,000 in weight decay and down to ten epochs, and it was extended upward once
    after a first sweep chose its largest value: a grid whose winner sits on its own boundary has not shown the optimum,
    only that it lies at least that far out. The recorded sweep lets a reader check that the chosen setting is interior,
    and that the remaining fit-to-report gap is spatial shift across Europe rather than regularisation left on the
    table.

    Returns the refitted probe and the whole sweep, so the record shows what regularisation bought."""
    inner_fit, inner_val = region_split(groups, salt="inner")
    if inner_fit.sum() < 200 or inner_val.sum() < 200:
        return fit_probe(X, y, seed=seed), {"note": "too few units to tune; defaults used", "chosen": None}
    sweep = []
    for wd in WD_GRID:
        for ep in EPOCH_GRID:
            lin = fit_probe(X[inner_fit], y[inner_fit], seed=seed, epochs=ep, wd=wd)
            pv = probe_probs(lin, X[inner_val][:, None, None, :])[:, 0, 0, :]
            pf = probe_probs(lin, X[inner_fit][:, None, None, :])[:, 0, 0, :]
            sweep.append({"weight_decay": wd, "epochs": ep,
                          "inner_fit_accuracy": float((pf.argmax(-1) == y[inner_fit]).mean()),
                          "inner_val_accuracy": float((pv.argmax(-1) == y[inner_val]).mean())})
            print(f"  tune wd={wd:g} epochs={ep}: inner fit {sweep[-1]['inner_fit_accuracy']:.4f} "
                  f"val {sweep[-1]['inner_val_accuracy']:.4f}", flush=True)
    best = max(sweep, key=lambda d: d["inner_val_accuracy"])
    print(f"  chosen wd={best['weight_decay']:g} epochs={best['epochs']} "
          f"(inner val {best['inner_val_accuracy']:.4f})", flush=True)
    return (fit_probe(X, y, seed=seed, epochs=best["epochs"], wd=best["weight_decay"]),
            {"grid": sweep, "chosen": {k: best[k] for k in ("weight_decay", "epochs", "inner_val_accuracy")},
             "n_inner_fit": int(inner_fit.sum()), "n_inner_val": int(inner_val.sum())})


def probe_probs(lin, E, batch=512):
    """(N, 5, 5, D) -> (N, 5, 5, 8) class probabilities, in batches so the whole stack never sits on the device."""
    import torch
    import exp18_sen1floods_expert as exp18
    out = []
    with torch.no_grad():
        for i in range(0, len(E), batch):
            x = torch.tensor(np.asarray(E[i:i + batch], dtype=np.float32), device=exp18.DEV)
            out.append(torch.softmax(lin(x), dim=-1).cpu().numpy())
    return np.concatenate(out)


def readings(p):
    """Centre-window decision, margin, top-1, entropy, and the boundary indicator from the 5 x 5 decision map."""
    c = NB // 2
    dec_map = p.argmax(-1)
    centre = p[:, c, c, :]
    srt = np.sort(centre, axis=-1)
    ent = -(np.clip(centre, 1e-7, 1) * np.log(np.clip(centre, 1e-7, 1))).sum(-1)
    d0 = dec_map[:, c, c]
    nb = np.stack([dec_map[:, c - 1, c - 1], dec_map[:, c - 1, c], dec_map[:, c - 1, c + 1],
                   dec_map[:, c, c - 1], dec_map[:, c, c + 1],
                   dec_map[:, c + 1, c - 1], dec_map[:, c + 1, c], dec_map[:, c + 1, c + 1]], axis=1)
    return {"dec": d0, "margin": srt[:, -1] - srt[:, -2], "top1": srt[:, -1], "entropy": ent,
            "boundary": (nb != d0[:, None]).mean(axis=1)}


# ----------------------------------------------------------------------------- scoring
def signal_table(r, meta, weights):
    """Every ranker, higher = more suspect. Named by what each is allowed to see."""
    sus = -r["margin"]
    rare = np.zeros(len(sus))
    freq = collections.Counter(r["dec"].tolist())
    for k, v in freq.items():
        rare[r["dec"] == k] = -np.log(max(v, 1) / len(sus))
    return {"margin": sus,
            "one_minus_top1": 1.0 - r["top1"],
            "entropy": r["entropy"],
            "boundary_first": boundary_first_score(sus, r["boundary"]),
            "boundary_only": r["boundary"].astype(np.float64),   # the neighbourhood-disagreement fraction alone, no
                                                                # confidence at all; it reads the model's argmax at nine
                                                                # windows, so it is a model output and not a control

            "ctl_pixel_variance": meta["variance"],
            "ctl_class_rarity": rare,
            "ctl_polygon_area": -meta["area"],
            "ctl_window_impurity": 1.0 - meta["purity"]}


def score_arm(sigs, err, weights, groups, name, rows):
    """Naive and design-weighted E-AURC and capture for every signal on one arm."""
    out = {}
    for s, u in sigs.items():
        naive_e = float(metrics.excess_aurc(u, err))
        wgt_e = float(w_excess_aurc(u, err, weights))
        nc = metrics.capture_at_budget_expected(u, err, BUDGETS)
        wc = w_capture(u, err, weights, BUDGETS)
        out[s] = {"excess_aurc_naive": naive_e, "excess_aurc_weighted": wgt_e,
                  "capture_naive": {str(b): float(v) for b, v in nc.items()},
                  "capture_weighted": {str(b): float(v) for b, v in wc.items()}}
        rows.append({"arm": name, "signal": s, "n": len(err), "n_errors": int(err.sum()),
                     "error_rate_naive": float(err.mean()), "error_rate_weighted": w_mean(err, weights),
                     "excess_aurc_naive": naive_e, "excess_aurc_weighted": wgt_e,
                     "capture_05_naive": float(nc[0.05]), "capture_10_naive": float(nc[0.10]),
                     "capture_20_naive": float(nc[0.20]), "capture_05_weighted": float(wc[0.05]),
                     "capture_10_weighted": float(wc[0.10]), "capture_20_weighted": float(wc[0.20])})
    return out


DEPLOYABLE_CONTROLS = ("ctl_pixel_variance", "ctl_class_rarity")
ORACLE_CONTROLS = ("ctl_polygon_area", "ctl_window_impurity")


def cue_stats(cue, err, w):
    """Design-weighted cue enrichment beside the odds ratio, which has no ceiling.

    The enrichment ratio is bounded above by 1 / (share among the correct units), so when a cue fires on almost every
    correct unit the ratio cannot be large however informative the cue is, and comparing enrichments across subsets with
    different fire rates partly compares headroom. The odds ratio is free of that ceiling, so both are recorded."""
    cue = np.asarray(cue, dtype=np.float64).ravel()
    err = np.asarray(err, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    we, wc = w * err, w * (1.0 - err)
    se = float((we * cue).sum() / max(we.sum(), 1e-300))
    sc = float((wc * cue).sum() / max(wc.sum(), 1e-300))
    odds = float((se / max(1 - se, 1e-300)) / max(sc / max(1 - sc, 1e-300), 1e-300)) if 0 < sc < 1 else float("nan")
    return {"share_errors_weighted": se, "share_correct_weighted": sc,
            "enrichment_weighted": float(se / sc) if sc > 0 else float("nan"),
            "enrichment_ceiling": float(1.0 / sc) if sc > 0 else float("nan"),
            "odds_ratio_weighted": odds, "fire_rate_weighted": w_mean(cue, w)}


w_auroc = metrics.weighted_auroc


def flag_lead(sus, flag, err, w):
    """How much a two-block re-ordering at this flag's fire rate gains over the plain suspicion order, weighted."""
    return float(w_excess_aurc(sus, err, w) - w_excess_aurc(boundary_first_score(sus, flag), err, w))


def rate_matched_leads(sus, boundary, variance, err, w, n_draw=200, seed=0):
    """The boundary order's gain beside two null orders that fire at the same rate but carry no boundary information.

    boundary_first_score coarsens the suspicion order into two blocks. That coarsening has a cost or a benefit of its
    own, set by the fire rate and by how much headroom the suspicion order has left, and it is nothing to do with
    boundaries. So the boundary gain is only evidence about boundaries to the extent it exceeds what a rate-matched
    partition carrying no information at all achieves: a random flag, and, as a second reference, the top slice of the
    within-window pixel variance, which is one of this experiment's own controls."""
    rate = float((np.asarray(boundary) > 0).mean())
    rng = np.random.default_rng(seed)
    rnd = [flag_lead(sus, rng.random(len(err)) < rate, err, w) for _ in range(n_draw)]
    v = np.asarray(variance, dtype=np.float64)
    k = max(1, int(round(rate * len(v))))
    var_flag = np.zeros(len(v), dtype=bool)
    var_flag[np.argsort(-v, kind="stable")[:k]] = True
    b = flag_lead(sus, np.asarray(boundary) > 0, err, w)
    return {"fire_rate": rate, "boundary": b, "random_mean": float(np.mean(rnd)),
            "random_sd": float(np.std(rnd)), "variance_matched": flag_lead(sus, var_flag, err, w),
            "boundary_minus_random": float(b - np.mean(rnd))}


def paired_cluster_boot(fn_a, groups_a, fn_b, groups_b, n_boot=1000, seed=0):
    return stats.paired_cluster_bootstrap(fn_a, groups_a, fn_b, groups_b, n_boot=n_boot, seed=seed)


def best_control(scored, family=DEPLOYABLE_CONTROLS, key="excess_aurc_weighted"):
    """The strongest control of one family, by weighted E-AURC (lower is a better ranker, so the strongest is the min).

    The families are kept apart on purpose: `DEPLOYABLE_CONTROLS` are computable at inference time from the imagery and
    the decision, `ORACLE_CONTROLS` read the reference's own geometry and no operator holds them. P1 is judged on the
    first, P1b on the second."""
    ctl = {k: v for k, v in scored.items() if k in family}
    name = min(ctl, key=lambda k: ctl[k][key])
    return name, ctl[name][key]


# ----------------------------------------------------------------------------- stage 2 driver
def load_shards():
    """Every fetched shard back into flat arrays, aligned on point_id."""
    files = sorted(f for f in os.listdir(CHIPDIR) if f.startswith("shard_") and f.endswith(".npz"))
    acc = collections.defaultdict(list)
    for f in files:
        with np.load(os.path.join(CHIPDIR, f), allow_pickle=True) as z:
            if int(z["n"]) == 0:
                continue
            cols = [str(c) for c in z["meta_cols"]]
            mpid = np.asarray(z["meta_point_id"]).astype(np.int64)
            look = {int(p): i for i, p in enumerate(mpid)}
            for tag in ("near", "far"):
                key = f"img_{tag}"
                if key not in z:
                    continue
                pid = np.asarray(z[f"pid_{tag}"]).astype(np.int64)
                acc[f"pid_{tag}"].append(pid)
                for k in ("img", "scl", "cov", "date", "cloud", "scene"):
                    acc[f"{k}_{tag}"].append(np.asarray(z[f"{k}_{tag}"]))
                sel = np.array([look[int(p)] for p in pid])
                for c in cols:
                    acc[f"{tag}_meta_{c}"].append(np.asarray(z[f"meta_{c}"])[sel])
    return {k: np.concatenate(v) for k, v in acc.items() if v}


def prepare_arm(d, tag, model):
    """Chosen window, its purity, its pixel variance, and the 5 x 5 embedding neighbourhood for one acquisition."""
    img, cov, scl = d[f"img_{tag}"], d[f"cov_{tag}"], d[f"scl_{tag}"]
    n = len(img)
    rc = np.zeros((n, 2), dtype=np.int64)
    purity = np.zeros(n, dtype=np.float64)
    var = np.zeros(n, dtype=np.float64)
    clear = np.zeros(n, dtype=np.float64)
    for i in range(n):
        cw = window_coverage(cov[i])
        r, q, best = choose_window(cw)
        rc[i] = (r, q)
        purity[i] = best
        sl = slice(r * PATCH, (r + 1) * PATCH), slice(q * PATCH, (q + 1) * PATCH)
        blk = img[i][:, sl[0], sl[1]].astype(np.float64)
        var[i] = float(blk.std(axis=(1, 2)).mean())
        clear[i] = float(np.isin(scl[i][sl[0], sl[1]], SCL_CLEAR).mean())
    dates = np.array([[int(s[8:10]), int(s[5:7]) - 1, int(s[:4])] for s in d[f"date_{tag}"].astype(str)])
    E, clamped = embed_chips(model, img, dates, rc)
    return {"E": E, "rc": rc, "purity": purity, "variance": var, "clear": clear, "clamped": clamped, "dates": dates}


def analyze_stage(args):
    import exp18_sen1floods_expert as exp18
    from oe_inferencex import compare as cmp_mod

    t0 = time.time()
    d = load_shards()
    n = len(d["pid_near"])
    print(f"loaded {n} near chips, {len(d.get('pid_far', []))} far chips in {time.time()-t0:.0f}s", flush=True)
    # harness_ab.load_model's construction, restated so this experiment does not import the Sen1Floods11 harness
    t_load = time.time()
    model = exp18.load_model_from_id(exp18.ModelID.OLMOEARTH_V1_BASE).to(exp18.DEV).eval().float()
    print(f"loaded OlmoEarth v1 Base on {exp18.DEV} in {time.time() - t_load:.1f}s", flush=True)
    near = prepare_arm(d, "near", model)
    print(f"encoded near in {time.time()-t0:.0f}s, D={near['E'].shape[-1]}", flush=True)

    y = np.array([CLASSES.index(str(c)) for c in d["near_meta_letter_group"]])
    nuts2 = d["near_meta_nuts2"].astype(str)
    nuts0 = d["near_meta_point_nuts0"].astype(str)
    obs = np.array([PHOTO if str(o).startswith("Photo") else FIELD for o in d["near_meta_survey_obs_type"]])
    area = d["near_meta_poly_area_sqm"].astype(np.float64)
    homog = np.array([str(v) == "Yes" for v in d["near_meta_survey_homplot_fills_extwin"]])
    with open(os.path.join(OUT, "exp68_strata.json")) as fh:
        strata = json.load(fh)
    skey = np.array([f"{CLASSES[c]}:{o}" for c, o in zip(y, obs)])
    wt_intended = np.array([strata[k]["ht_weight"] for k in skey])
    # Nonresponse: 217 of the 12,073 sampled polygons returned no usable scene, and that loss is not missing at random
    # across Europe, so the weight is the stratum's population size over the number of polygons actually in hand, not
    # over the number we meant to draw. Both are recorded.
    realized = collections.Counter(skey.tolist())
    wt = np.array([strata[k]["N_stratum"] / max(realized[k], 1) for k in skey])
    summary_nonresponse = {k: {"N_stratum": strata[k]["N_stratum"], "n_sampled": strata[k]["n_sampled"],
                               "n_realized": int(realized.get(k, 0)),
                               "response_rate": realized.get(k, 0) / max(strata[k]["n_sampled"], 1),
                               "ht_weight_intended": strata[k]["ht_weight"],
                               "ht_weight_adjusted": strata[k]["N_stratum"] / max(realized.get(k, 1), 1)}
                           for k in sorted(strata)}

    # A window whose sixteen pixels are all scene-classification nodata carries no observation and cannot be graded;
    # read_chip fills outside-granule reads with zeros, so these exist and must be dropped rather than scored.
    no_data = (near["clear"] <= 0) & (near["variance"] <= 0)
    fit_m, rep_m = region_split(nuts2)
    rep_m = rep_m & ~no_data
    fit_sel = fit_m & (obs == FIELD) & ~no_data
    c = NB // 2
    Xc = near["E"][:, c, c, :].astype(np.float32)
    mu, sd = Xc[fit_sel].mean(0), Xc[fit_sel].std(0) + 1e-6
    Xn = (near["E"] - mu) / sd

    summary = {"experiment": "exp68 the protocol against ground observation: LUCAS Copernicus 2022",
               "config": {"n_polygons": int(n), "n_fit": int(fit_sel.sum()), "n_report": int(rep_m.sum()),
                          "embedding_dim": int(near["E"].shape[-1]), "window_px": PATCH, "chip_px": SIZE,
                          "crop_px": CROP, "grid": G, "neighbourhood": NB, "classes": CLASSES,
                          "boa_offset_removed": BOA_OFFSET, "strata": strata,
                          "caveats": ["the reference is in-situ observation, causally independent of the imagery",
                                      "the probe is fitted on about 3,000 field-surveyed polygons in the fit regions "
                                      "for 768 features and eight classes, which is thin; weight decay and epochs are "
                                      "tuned on NUTS2 regions held out of the fit set, never on the report regions, "
                                      "and the fit, inner-validation and report accuracies are all recorded so the "
                                      "remaining gap is visible and attributable",
                                      "the graded unit is the polygon; the median polygon is smaller than the 40 m window",
                                      "LUCAS design weights over EU area are not shipped; the estimand is the Copernicus polygon population",
                                      "photo-interpreted points sit in harder terrain, so part C matches on class and country"]},
               "results": {}, "verdicts": {}}
    summary["config"]["nonresponse_by_stratum"] = summary_nonresponse
    summary["config"]["n_sampled_intended"] = int(sum(v["n_sampled"] for v in strata.values()))
    summary["config"]["n_realized"] = int(len(y))
    summary["config"]["n_dropped_no_valid_pixels"] = int(no_data.sum())
    summary["config"]["weights"] = ("Horvitz-Thompson, stratum population over the number of polygons actually in hand; "
                                   "the intended-sample weights are recorded per stratum beside them")
    rows = []

    # ---- probes, two seeds
    arms = {}
    tuned = None
    for seed in (0, 1):
        if seed == 0:
            lin, tuned = tune_probe(Xn[:, c, c, :][fit_sel], y[fit_sel], nuts2[fit_sel], seed=seed)
            summary["results"]["probe_tuning"] = tuned
        else:
            ch = (tuned or {}).get("chosen") or {}
            lin = fit_probe(Xn[:, c, c, :][fit_sel], y[fit_sel], seed=seed,
                            epochs=ch.get("epochs", 80), wd=ch.get("weight_decay", 1e-4))
        p = probe_probs(lin, Xn)
        arms[seed] = readings(p)
        acc = float((arms[seed]["dec"][rep_m] == y[rep_m]).mean())
        acc_fit = float((arms[seed]["dec"][fit_sel] == y[fit_sel]).mean())
        print(f"probe seed {seed}: fit accuracy {acc_fit:.4f}, report accuracy {acc:.4f}", flush=True)
        summary["results"][f"probe_seed_{seed}_report_accuracy"] = acc
        summary["results"][f"probe_seed_{seed}_fit_accuracy"] = acc_fit
    r = arms[0]
    err = (r["dec"] != y).astype(np.float64)
    summary["results"]["per_class_accuracy"] = {
        CLASS_NAME[CLASSES[k]]: float((r["dec"][rep_m & (y == k)] == k).mean()) if (rep_m & (y == k)).any() else float("nan")
        for k in range(N_CLASS)}
    summary["results"]["head_draw_floor_decision_change"] = float((arms[0]["dec"] != arms[1]["dec"])[rep_m].mean())

    # ---- part A, ranking on the held-out regions, field-surveyed arm
    meta = {"variance": near["variance"], "area": area, "purity": near["purity"]}
    def sub(m):
        return {k: v[m] for k, v in meta.items()}
    A = rep_m & (obs == FIELD)
    sigs_a = signal_table({k: v[A] for k, v in r.items()}, sub(A), wt[A])
    scored_a = score_arm(sigs_a, err[A], wt[A], nuts2[A], "field_report", rows)
    ctl_name, ctl_val = best_control(scored_a, DEPLOYABLE_CONTROLS)
    orc_name, orc_val = best_control(scored_a, ORACLE_CONTROLS)
    lead_w = ctl_val - scored_a["margin"]["excess_aurc_weighted"]
    lead_orc = orc_val - scored_a["margin"]["excess_aurc_weighted"]
    st = region_sign_test({"u": sigs_a["margin"], "err": err[A]},
                          {"u": sigs_a[ctl_name], "err": err[A]}, nuts2[A], weights=wt[A])
    st_unw = region_sign_test({"u": sigs_a["margin"], "err": err[A]},
                              {"u": sigs_a[ctl_name], "err": err[A]}, nuts2[A])
    boot = cluster_boot(lambda s: (w_excess_aurc(sigs_a[ctl_name][s], err[A][s], wt[A][s])
                                   - w_excess_aurc(sigs_a["margin"][s], err[A][s], wt[A][s])), nuts2[A], seed=args.seed)
    summary["results"]["part_a"] = {"n": int(A.sum()), "n_errors": int(err[A].sum()),
                                    "error_rate_weighted": w_mean(err[A], wt[A]),
                                    "signals": scored_a, "best_control": ctl_name, "best_oracle_control": orc_name,
                                    "margin_lead_over_oracle_weighted": lead_orc,
                                    "margin_lead_weighted": lead_w,
                                    "margin_lead_naive": float(scored_a[ctl_name]["excess_aurc_naive"]
                                                               - scored_a["margin"]["excess_aurc_naive"]),
                                    "region_sign_test": st, "region_sign_test_unweighted": st_unw,
                                    "cluster_bootstrap": boot,
                                    "per_class": {CLASS_NAME[CLASSES[k]]: (lambda mk: (
                                        {"n": int(mk.sum()),
                                         "weight_share": float(wt[A][mk].sum() / wt[A].sum()),
                                         "error_rate_weighted": w_mean(err[A][mk], wt[A][mk]),
                                         "margin_excess_aurc_weighted": w_excess_aurc(sigs_a["margin"][mk], err[A][mk], wt[A][mk]),
                                         "margin_lead_weighted": float(w_excess_aurc(sigs_a[ctl_name][mk], err[A][mk], wt[A][mk])
                                                                       - w_excess_aurc(sigs_a["margin"][mk], err[A][mk], wt[A][mk]))}
                                        if mk.sum() >= 40 and 0 < err[A][mk].sum() < mk.sum()
                                        else {"n": int(mk.sum()), "note": "too few units or degenerate"}))(y[A] == k)
                                        for k in range(N_CLASS)}}
    summary["verdicts"]["P1"] = {"holds": bool(lead_w >= 0.01 and st["p"] < 0.05 and st["wins"] > st["losses"]),
                                 "lead_weighted": lead_w, "threshold": 0.01, "against": ctl_name,
                                 "sign_test": st, "sign_test_unweighted": st_unw,
                                 "note": "the sign test uses the design-weighted per-region E-AURC, which is the "
                                         "estimator the preregistration names; the unweighted test is beside it"}
    # Not preregistered, and added because part A raised it: the margin is this repository's recommended ranker, and
    # on this task three other model signals score below it. Whether that ordering is real or noise is a question the
    # record must answer rather than leave to the eye, so every ranker is tested against the margin on the same units.
    pair = {}
    for name, u in sigs_a.items():
        if name == "margin":
            continue
        st_p = region_sign_test({"u": u, "err": err[A]}, {"u": sigs_a["margin"], "err": err[A]}, nuts2[A])
        bt = cluster_boot(lambda idx_, u_=u: (w_excess_aurc(sigs_a["margin"][idx_], err[A][idx_], wt[A][idx_])
                                              - w_excess_aurc(u_[idx_], err[A][idx_], wt[A][idx_])),
                          nuts2[A], n_boot=2000, seed=args.seed)
        pair[name] = {"beats_margin_by_weighted": float(scored_a["margin"]["excess_aurc_weighted"]
                                                        - scored_a[name]["excess_aurc_weighted"]),
                      "region_sign_test_vs_margin": st_p, "cluster_bootstrap": bt,
                      "significant": bool(bt["lo"] > 0 or bt["hi"] < 0)}
    summary["results"]["part_a_vs_margin"] = pair

    bands = {}
    for lo, hi in PURITY_BANDS:
        m = A & (near["purity"] >= lo) & (near["purity"] < hi)
        if m.sum() < 40 or err[m].sum() < 5:
            bands[f"{lo:.2f}-{min(hi,1.0):.2f}"] = {"n": int(m.sum()), "note": "too few units or errors to score"}
            continue
        sg = signal_table({k: v[m] for k, v in r.items()}, sub(m), wt[m])
        sq = score_arm(sg, err[m], wt[m], nuts2[m], f"purity_{lo:.2f}", rows)
        cn, cv = best_control(sq, DEPLOYABLE_CONTROLS)
        bands[f"{lo:.2f}-{min(hi,1.0):.2f}"] = {
            "n": int(m.sum()), "error_rate_weighted": w_mean(err[m], wt[m]),
            "margin_excess_aurc_weighted": sq["margin"]["excess_aurc_weighted"],
            "best_control": cn, "margin_lead_weighted": float(cv - sq["margin"]["excess_aurc_weighted"]),
            "capture_10_weighted": sq["margin"]["capture_weighted"]["0.1"]}
    summary["results"]["part_a_by_purity"] = bands
    summary["results"]["purity_quantiles"] = {q: float(np.quantile(near["purity"][A], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
    summary["results"]["scl_clear_quantiles"] = {q: float(np.quantile(near["clear"][A], q)) for q in (0.05, 0.5, 0.95)}
    summary["results"]["n_neighbourhood_clamped"] = int(near["clamped"].sum())
    summary["results"]["window_holds_the_survey_point"] = int(POINT_WIN)
    summary["verdicts"]["P1b"] = {"holds": bool(lead_orc >= 0.01), "lead_weighted": lead_orc, "threshold": 0.01,
                                  "against": orc_name,
                                  "note": "oracle-side control; failing this is informative, not a falsification"}
    summary["verdicts"]["P4"] = {"holds": bool(lead_w >= 0.005 and boot["lo"] > 0),
                                 "lead_weighted": lead_w, "bootstrap": boot,
                                 "naive_minus_weighted": float(summary["results"]["part_a"]["margin_lead_naive"] - lead_w)}

    # ---- part C, filters (P2) and provenance (P3)
    filt = A & homog & (area >= 5000)
    unfilt = A
    deleted = A & ~(homog & (area >= 5000))        # exactly the units the published convention throws away
    cap = {}
    for tag, m in (("filtered", filt), ("unfiltered", unfilt), ("deleted_by_filter", deleted)):
        s = signal_table({k: v[m] for k, v in r.items()}, sub(m), wt[m])
        sc = score_arm(s, err[m], wt[m], nuts2[m], f"{tag}_field_report", rows)
        enr = cue_enrichment(r["boundary"][m] > 0, err[m], clusters=nuts2[m], n_boot=400, seed=args.seed)
        cue_w = cue_stats(r["boundary"][m] > 0, err[m], wt[m])
        cap[tag] = {"n": int(m.sum()), "capture_10_weighted": sc["margin"]["capture_weighted"]["0.1"],
                    "boundary_enrichment": float(enr["enrichment"]),
                    "boundary_cue_weighted": cue_w,
                    "error_rate_weighted": w_mean(err[m], wt[m]),
                    "capture_10_ceiling": float(min(1.0, 0.10 / max(w_mean(err[m], wt[m]), 1e-9))),
                    "margin_auroc_weighted": w_auroc(-r["margin"][m], err[m], wt[m]),
                    "signals": sc}
    # filtered against deleted is a contrast between disjoint sets, so the direction can carry an interval; filtered
    # against unfiltered cannot, the one being a subset of the other.
    def _cap10(mask, idx_):
        sub_i = np.flatnonzero(mask)[idx_]
        return w_capture(-r["margin"][sub_i], err[sub_i], wt[sub_i], (0.10,))[0.10]
    def _auroc(mask, idx_):
        sub_i = np.flatnonzero(mask)[idx_]
        return w_auroc(-r["margin"][sub_i], err[sub_i], wt[sub_i])
    boot_filt = cluster_boot(lambda ii: _cap10(filt, ii), nuts2[filt], n_boot=1000, seed=args.seed)
    boot_del = cluster_boot(lambda ii: _cap10(deleted, ii), nuts2[deleted], n_boot=1000, seed=args.seed)
    paired_cap = paired_cluster_boot(lambda ii: _cap10(filt, ii), nuts2[filt],
                                     lambda ii: _cap10(deleted, ii), nuts2[deleted], n_boot=1000, seed=args.seed)
    paired_auc = paired_cluster_boot(lambda ii: _auroc(filt, ii), nuts2[filt],
                                     lambda ii: _auroc(deleted, ii), nuts2[deleted], n_boot=1000, seed=args.seed)
    cap["filtered_vs_deleted"] = {
        "n_filtered": int(filt.sum()), "n_deleted": int(deleted.sum()),
        "capture_10_filtered": cap["filtered"]["capture_10_weighted"],
        "capture_10_deleted": cap["deleted_by_filter"]["capture_10_weighted"],
        "margin_excess_aurc_filtered": cap["filtered"]["signals"]["margin"]["excess_aurc_weighted"],
        "margin_excess_aurc_deleted": cap["deleted_by_filter"]["signals"]["margin"]["excess_aurc_weighted"],
        "bootstrap_capture_10_filtered": boot_filt, "bootstrap_capture_10_deleted": boot_del,
        "intervals_disjoint": bool(boot_filt["lo"] > boot_del["hi"] or boot_del["lo"] > boot_filt["hi"]),
        # Capture at a fixed budget is bounded above by budget / error rate, so the two subsets do not have the same
        # attainable maximum and the raw comparison is partly a base-rate effect; AUROC has no such ceiling.
        "capture_10_ceiling_filtered": cap["filtered"]["capture_10_ceiling"],
        "capture_10_ceiling_deleted": cap["deleted_by_filter"]["capture_10_ceiling"],
        "capture_10_share_of_ceiling_filtered": float(cap["filtered"]["capture_10_weighted"]
                                                      / max(cap["filtered"]["capture_10_ceiling"], 1e-9)),
        "capture_10_share_of_ceiling_deleted": float(cap["deleted_by_filter"]["capture_10_weighted"]
                                                     / max(cap["deleted_by_filter"]["capture_10_ceiling"], 1e-9)),
        "margin_auroc_filtered": cap["filtered"]["margin_auroc_weighted"],
        "margin_auroc_deleted": cap["deleted_by_filter"]["margin_auroc_weighted"],
        "paired_bootstrap_capture_10": paired_cap, "paired_bootstrap_auroc": paired_auc}
    summary["results"]["part_c_filters"] = cap
    summary["verdicts"]["P2"] = {
        "holds": bool(cap["unfiltered"]["capture_10_weighted"] > cap["filtered"]["capture_10_weighted"]
                      and cap["unfiltered"]["boundary_enrichment"] > cap["filtered"]["boundary_enrichment"]),
        "capture_10_unfiltered": cap["unfiltered"]["capture_10_weighted"],
        "capture_10_filtered": cap["filtered"]["capture_10_weighted"],
        "boundary_enrichment_unfiltered": cap["unfiltered"]["boundary_enrichment"],
        "boundary_enrichment_filtered": cap["filtered"]["boundary_enrichment"],
        # the base-rate-free reading of the same contrast, on disjoint sets
        "auroc_filtered": cap["filtered"]["margin_auroc_weighted"],
        "auroc_deleted": cap["deleted_by_filter"]["margin_auroc_weighted"],
        "auroc_difference_bootstrap": paired_auc,
        "odds_ratio_filtered": cap["filtered"]["boundary_cue_weighted"]["odds_ratio_weighted"],
        "odds_ratio_deleted": cap["deleted_by_filter"]["boundary_cue_weighted"]["odds_ratio_weighted"],
        "note": ("enrichment is bounded by 1 / share-among-correct, which differs between the subsets, so the odds "
                 "ratio and the AUROC are the ceiling-free readings of the same contrast")}

    key = np.array([f"{CLASSES[c_]}|{k}" for c_, k in zip(y, nuts0)])
    both = {k for k in set(key[rep_m]) if (rep_m & (key == k) & (obs == FIELD)).sum() >= 5
            and (rep_m & (key == k) & (obs == PHOTO)).sum() >= 5}
    matched = rep_m & np.isin(key, list(both))
    prov = {}
    for tag in (FIELD, PHOTO):
        m = matched & (obs == tag)
        if m.sum() < 30 or err[m].sum() == 0:
            prov[tag] = {"n": int(m.sum()), "lead_boundary_first_over_margin": float("nan")}
            continue
        s = signal_table({k: v[m] for k, v in r.items()}, sub(m), wt[m])
        sc = score_arm(s, err[m], wt[m], nuts2[m], f"matched_{tag}", rows)
        prov[tag] = {"n": int(m.sum()), "error_rate_weighted": w_mean(err[m], wt[m]),
                     "lead_boundary_first_over_margin": float(sc["margin"]["excess_aurc_weighted"]
                                                              - sc["boundary_first"]["excess_aurc_weighted"]),
                     "rate_matched": rate_matched_leads(-r["margin"][m], r["boundary"][m], near["variance"][m],
                                                        err[m], wt[m], seed=args.seed),
                     "median_polygon_area": float(np.median(area[m])),
                     "median_window_purity": float(np.median(near["purity"][m])),
                     "signals": sc}
    # The photo arm is harder than the field arm even within matched class-country strata, so the gap could be a
    # difficulty effect rather than a provenance effect. Recompute it on high-purity units, where the two arms' error
    # rates are closest, as the nearest thing to matching on difficulty that this design allows.
    prov_hi = {}
    for tag in (FIELD, PHOTO):
        m = matched & (obs == tag) & (near["purity"] >= 0.75)
        if m.sum() < 30 or err[m].sum() == 0:
            prov_hi[tag] = {"n": int(m.sum()), "lead_boundary_first_over_margin": float("nan")}
            continue
        sg = signal_table({k: v[m] for k, v in r.items()}, sub(m), wt[m])
        sq = score_arm(sg, err[m], wt[m], nuts2[m], f"matched_highpurity_{tag}", rows)
        prov_hi[tag] = {"n": int(m.sum()), "error_rate_weighted": w_mean(err[m], wt[m]),
                        "lead_boundary_first_over_margin": float(sq["margin"]["excess_aurc_weighted"]
                                                                - sq["boundary_first"]["excess_aurc_weighted"])}
    gap_hi = prov_hi[PHOTO]["lead_boundary_first_over_margin"] - prov_hi[FIELD]["lead_boundary_first_over_margin"]
    gap = prov[PHOTO]["lead_boundary_first_over_margin"] - prov[FIELD]["lead_boundary_first_over_margin"]
    rm_gap = (prov[PHOTO]["rate_matched"]["boundary_minus_random"]
              - prov[FIELD]["rate_matched"]["boundary_minus_random"]) if "rate_matched" in prov[PHOTO] and "rate_matched" in prov[FIELD] else float("nan")
    null_gap = (prov[PHOTO]["rate_matched"]["random_mean"] - prov[FIELD]["rate_matched"]["random_mean"]) \
        if "rate_matched" in prov[PHOTO] and "rate_matched" in prov[FIELD] else float("nan")
    summary["results"]["part_c_provenance"] = {"n_matched_strata": len(both), "arms": prov, "gap": float(gap),
                                              "high_purity_arms": prov_hi, "high_purity_gap": float(gap_hi),
                                              "gap_from_a_rate_matched_null": float(null_gap),
                                              "boundary_specific_gap": float(rm_gap)}
    summary["verdicts"]["P3"] = {"holds": bool(np.isfinite(gap) and gap > 0), "gap": float(gap),
                                 "photo_lead": prov[PHOTO]["lead_boundary_first_over_margin"],
                                 "field_lead": prov[FIELD]["lead_boundary_first_over_margin"],
                                 "high_purity_gap": float(gap_hi),
                                 "high_purity_holds": bool(np.isfinite(gap_hi) and gap_hi > 0),
                                 "gap_from_a_rate_matched_null": float(null_gap),
                                 "boundary_specific_gap": float(rm_gap),
                                 "boundary_specific_holds": bool(np.isfinite(rm_gap) and rm_gap > 0),
                                 "note": ("a two-block re-ordering has a cost of its own set by its fire rate and by "
                                          "the suspicion order's remaining headroom, so only the part of the gap that "
                                          "exceeds a rate-matched null carrying no information is about boundaries")}

    # ---- part B, two acquisitions of one place
    if "pid_far" in d and len(d["pid_far"]):
        far = prepare_arm(d, "far", model)
        Xf = (far["E"] - mu) / sd
        ch0 = (tuned or {}).get("chosen") or {}
        lin0 = fit_probe(Xn[:, c, c, :][fit_sel], y[fit_sel], seed=0,
                         epochs=ch0.get("epochs", 80), wd=ch0.get("weight_decay", 1e-4))
        rf = readings(probe_probs(lin0, Xf))
        pos = {int(p): i for i, p in enumerate(d["pid_near"])}
        idx = np.array([pos[int(p)] for p in d["pid_far"]])
        keep = rep_m[idx] & (far["clear"] > 0)
        a, b = r["dec"][idx][keep], rf["dec"][keep]
        lab, ok = y[idx][keep], np.ones(int(keep.sum()), dtype=bool)
        cmpres = cmp_mod.compare_inferences(a, b, ok, groups=nuts2[idx][keep], labels=lab,
                                            cues={"boundary_near": r["boundary"][idx][keep] > 0,
                                                  "low_margin_near": r["margin"][idx][keep] <= np.quantile(r["margin"][idx][keep], 0.2)})
        changed = (a != b)
        per_class = {}
        yy, wwb = y[idx][keep], wt[idx][keep]
        for k, cl in enumerate(CLASSES):
            m = yy == k
            per_class[CLASS_NAME[cl]] = {"n": int(m.sum()),
                                         "change_rate": float(changed[m].mean()) if m.any() else float("nan"),
                                         "change_rate_weighted": w_mean(changed[m], wwb[m]) if m.any() else float("nan")}
        gapdays = np.abs(d["date_far"].astype(str).astype("datetime64[D]").astype(int)[keep]
                         - d["date_near"].astype(str).astype("datetime64[D]").astype(int)[idx][keep])
        summary["results"]["part_b"] = {
            "n": int(keep.sum()), "median_day_gap": float(np.median(gapdays)),
            "disagreement_rate": cmpres["disagreement_rate"],
            "head_draw_floor": summary["results"]["head_draw_floor_decision_change"],
            "which_side": cmpres["graded"]["which_side"], "crosstab": cmpres["graded"]["crosstab"],
            "near_beats_far_sign_test": {
                "a_right": int(cmpres["graded"]["which_side"]["a_right"]),
                "b_right": int(cmpres["graded"]["which_side"]["b_right"]),
                "p_greater": float(stats.sign_test(int(cmpres["graded"]["which_side"]["a_right"]),
                                                  int(cmpres["graded"]["which_side"]["b_right"]),
                                                  alternative="greater"))},
            "where": cmpres["where"], "per_class": per_class}
        clear_both = (near["clear"][idx][keep] >= 0.99) & (far["clear"][keep] >= 0.99)
        per_class_clear = {}
        for k, cl in enumerate(CLASSES):
            m = (yy == k) & clear_both
            per_class_clear[CLASS_NAME[cl]] = {"n": int(m.sum()),
                                               "change_rate": float(changed[m].mean()) if m.sum() >= 20 else float("nan")}
        summary["results"]["part_b"]["per_class_cloud_clear"] = per_class_clear
        summary["results"]["part_b"]["n_cloud_clear_both"] = int(clear_both.sum())
        floor = min(v["change_rate"] for v in per_class.values() if np.isfinite(v["change_rate"]))
        floor_class = min((v["change_rate"], k) for k, v in per_class.items() if np.isfinite(v["change_rate"]))[1]
        summary["results"]["part_b"]["class_independent_floor"] = {
            "lowest_change_rate": float(floor), "class": floor_class,
            "note": ("the probe-reseed rate is the floor for a different question, whether the head is stable; the "
                     "floor for a two-date change rate is the rate on cover that did not change, and the most stable "
                     "class is the closest observable bound on it")}
        cb = per_class[CLASS_NAME["B"]]["change_rate"]
        stable = max(per_class[CLASS_NAME["A"]]["change_rate"], per_class[CLASS_NAME["C"]]["change_rate"])
        summary["verdicts"]["P5"] = {"holds": bool(np.isfinite(cb) and np.isfinite(stable) and cb >= 2 * stable),
                                     "cropland_change_rate": cb, "max_stable_change_rate": stable,
                                     "artificial": per_class[CLASS_NAME["A"]]["change_rate"],
                                     "woodland": per_class[CLASS_NAME["C"]]["change_rate"],
                                     "cropland_over_lowest_class": float(cb / max(floor, 1e-9)),
                                     "lowest_class": floor_class,
                                     "cropland_change_rate_weighted": per_class[CLASS_NAME["B"]]["change_rate_weighted"],
                                     "cloud_clear_only": {
                                         "cropland": per_class_clear[CLASS_NAME["B"]]["change_rate"],
                                         "artificial": per_class_clear[CLASS_NAME["A"]]["change_rate"],
                                         "woodland": per_class_clear[CLASS_NAME["C"]]["change_rate"]}}
    else:
        summary["verdicts"]["P5"] = {"holds": None, "note": "no far acquisitions fetched"}

    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(OUT, f"exp68_summary{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    with open(os.path.join(OUT, f"exp68_lucas{tag}.csv"), "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    np.savez_compressed(os.path.join(OUT, f"exp68_masks{tag}.npz"),
                        dec=r["dec"], y=y, err=err, margin=r["margin"], top1=r["top1"], entropy=r["entropy"],
                        boundary=r["boundary"], purity=near["purity"], variance=near["variance"], clear=near["clear"],
                        area=area, weight=wt, weight_intended=wt_intended, report=rep_m, obs=obs, homog=homog,
                        nuts2=nuts2, clamped=near["clamped"], no_data=no_data, point_id=d["pid_near"],
                        fit=fit_sel, nuts0=nuts0)
    for k, v in summary["verdicts"].items():
        print(f"{k}: {v.get('holds')}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return summary


# ----------------------------------------------------------------------------- smoke
def smoke(args):
    """Synthetic end-to-end check of everything that is not the network or the encoder.

    The first block is the one that matters: the design-weighted estimators this experiment reports must reduce to the
    package's unweighted ones when every weight is one, ties included, or a Horvitz-Thompson number here would not be
    comparable with any other number in this repository."""
    rng = np.random.default_rng(0)
    n = 2000
    e = (rng.random(n) < 0.2).astype(np.float64)
    ones = np.ones(n)
    for name, u in (("continuous", rng.random(n)), ("tied", np.round(rng.random(n), 1)),
                    ("informative", rng.random(n) - 0.6 * e)):
        a, b = w_aurc(u, e, ones), metrics.aurc_expected(u, e)
        assert abs(a - b) < 1e-9, f"w_aurc {name}: {a} vs {b}"
        a, b = w_excess_aurc(u, e, ones), metrics.excess_aurc(u, e)
        assert abs(a - b) < 1e-9, f"w_excess_aurc {name}: {a} vs {b}"
        wc, nc = w_capture(u, e, ones), metrics.capture_at_budget_expected(u, e, BUDGETS)
        tol = 2.0 / max(e.sum(), 1) + 1e-9
        for bgt in BUDGETS:
            assert abs(wc[bgt] - nc[bgt]) <= tol, f"w_capture {name} @{bgt}: {wc[bgt]} vs {nc[bgt]}"
    # rescaling every weight must change nothing at all
    u = rng.random(n)
    assert abs(w_aurc(u, e, np.full(n, 3.7)) - w_aurc(u, e, ones)) < 1e-12, "w_aurc is not scale invariant"
    # a weight of two must agree with duplicating the row, but only to the discretisation of AURC itself: AURC is a
    # right-endpoint average over units, so it is not exactly replication invariant and neither is the package's
    # aurc_expected, which moves by 1.7e-5 on this data when every unit is duplicated. 1e-4 is that order, not 1e-9.
    w2 = np.where(np.arange(n) < 500, 2.0, 1.0)
    dup = np.r_[np.arange(n), np.arange(500)]
    assert abs(w_aurc(u, e, w2) - w_aurc(u[dup], e[dup], np.ones(len(dup)))) < 1e-4, "weights do not act as replication"

    # the weighted AUROC must equal the Mann-Whitney statistic under unit weights
    for name, u in (("continuous", rng.random(n)), ("tied", np.round(rng.random(n), 1))):
        pos, neg = u[e > 0], u[e == 0]
        ref = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
        got = w_auroc(u, e, ones)
        assert abs(got - ref) < 1e-9, f"w_auroc {name}: {got} vs Mann-Whitney {ref}"
    assert abs(w_auroc(u, e, np.full(n, 4.2)) - w_auroc(u, e, ones)) < 1e-9, "w_auroc is not scale invariant"
    assert np.isnan(w_auroc(rng.random(10), np.ones(10), np.ones(10))), "w_auroc must be NaN with no correct units"

    # cue statistics: the enrichment ceiling is exactly 1 / share-among-correct, and a useless cue gives 1 and 1
    cue = (rng.random(n) < 0.3)
    cs = cue_stats(cue, e, ones)
    assert abs(cs["enrichment_ceiling"] - 1.0 / cs["share_correct_weighted"]) < 1e-9
    assert abs(cs["enrichment_weighted"] - 1.0) < 0.2 and abs(cs["odds_ratio_weighted"] - 1.0) < 0.6
    perfect = cue_stats(e > 0, e, ones)                      # a cue that fires exactly on the errors
    assert perfect["share_errors_weighted"] == 1.0 and perfect["share_correct_weighted"] == 0.0
    assert np.isnan(perfect["enrichment_weighted"]) and np.isnan(perfect["odds_ratio_weighted"]), \
        "a cue with no false positives has no finite ratio and must report NaN rather than a number"

    # a rate-matched null must find a nonzero cost or benefit for a two-block re-ordering that carries no information
    sus0 = rng.random(n) - 0.5 * e
    rm = rate_matched_leads(sus0, (rng.random(n) < 0.4).astype(float), rng.random(n), e, ones, n_draw=40, seed=0)
    assert 0.3 < rm["fire_rate"] < 0.5 and np.isfinite(rm["random_mean"]) and np.isfinite(rm["boundary_minus_random"])
    assert abs(rm["boundary"] - (rm["random_mean"] + rm["boundary_minus_random"])) < 1e-12

    # the paired bootstrap must return an interval on the difference of two disjoint subsets
    g = np.array([f"R{i % 24}" for i in range(n)])
    half = np.arange(n) < n // 2
    pb = paired_cluster_boot(lambda ii: float(e[half][ii].mean()), g[half],
                             lambda ii: float(e[~half][ii].mean()), g[~half], n_boot=120, seed=0)
    assert pb["n"] > 50 and pb["difference_lo"] <= pb["difference_mean"] <= pb["difference_hi"]

    # window choice on a known polygon
    cov = np.zeros((SIZE, SIZE), dtype=np.float32)
    cov[20:28, 36:44] = 1.0                       # windows rows 5-6, cols 9-10 of the 15 x 15 grid
    r_, q_, best = choose_window(window_coverage(cov))
    assert best == 1.0 and r_ in (5, 6) and q_ in (9, 10), f"choose_window gave {(r_, q_, best)}"
    # the survey point sits at chip pixel (32, 32) and the crop is taken from the top-left, so its window is 8, not 7
    assert POINT_WIN == 8 and POINT_WIN * PATCH <= SIZE // 2 < (POINT_WIN + 1) * PATCH, "POINT_WIN does not hold the point"
    assert choose_window(window_coverage(np.zeros((SIZE, SIZE), np.float32))) == (POINT_WIN, POINT_WIN, 0.0), \
        "the no-overlap fallback must be the window holding the survey point"
    one_px = np.zeros((SIZE, SIZE), dtype=np.float32)
    one_px[SIZE // 2, SIZE // 2] = 1.0
    assert choose_window(window_coverage(one_px))[:2] == (POINT_WIN, POINT_WIN), "a polygon on the point must pick its window"

    # readings and the neighbourhood
    m = 300
    feats = rng.standard_normal((m, G, G, 16)).astype(np.float16)
    rc = rng.integers(0, G, size=(m, 2))
    E, clamped = neighbourhood(feats, rc)
    assert E.shape == (m, NB, NB, 16) and clamped.any()
    mid = (rc[:, 0] >= 2) & (rc[:, 0] <= G - 3) & (rc[:, 1] >= 2) & (rc[:, 1] <= G - 3)
    i = int(np.flatnonzero(mid)[0])
    assert np.array_equal(E[i, NB // 2, NB // 2], feats[i, rc[i, 0], rc[i, 1]]), "neighbourhood is off-centre"
    p = rng.dirichlet(np.ones(N_CLASS), size=(m, NB, NB)).astype(np.float32)
    rd = readings(p)
    assert rd["margin"].min() >= 0 and rd["top1"].max() <= 1 and 0 <= rd["boundary"].min() <= rd["boundary"].max() <= 1
    assert np.array_equal(rd["dec"], p[:, NB // 2, NB // 2].argmax(-1))

    # a planted ranking: the margin must beat every control it is scored against
    err = (rng.random(m) < 0.25).astype(np.float64)
    rd["margin"] = np.clip(0.8 - 0.6 * err + 0.12 * rng.standard_normal(m), 0, 1)
    rd["top1"] = np.clip(rd["margin"] * 0.5 + 0.5, 0, 1)
    meta = {"variance": rng.random(m) * 100, "area": rng.random(m) * 8000, "purity": rng.random(m)}
    sigs = signal_table(rd, meta, np.ones(m))
    rows = []
    sc = score_arm(sigs, err, rng.random(m) * 2 + 0.5, rng.integers(0, 12, m), "smoke", rows)
    for fam in (DEPLOYABLE_CONTROLS, ORACLE_CONTROLS):
        name, val = best_control(sc, fam)
        assert name in fam, f"best_control crossed families: {name} not in {fam}"
        assert sc["margin"]["excess_aurc_weighted"] < val, f"planted margin lost to {name}"
    assert len(rows) == len(sigs) and all(set(r0) == set(rows[0]) for r0 in rows)

    # sampling design and the region split
    import pandas as pd
    df = pd.DataFrame({"letter_group": rng.choice(CLASSES, 5000),
                       "survey_obs_type": rng.choice(["Field Survey, point visible <= 100 m",
                                                      "Photo-interpretation in the field"], 5000, p=[0.95, 0.05])})
    take, strata = build_sample(df, seed=0)
    assert len(take) == len(set(take.tolist()))
    for k, v in strata.items():
        assert abs(v["ht_weight"] * v["n_sampled"] - v["N_stratum"]) < 1e-6, f"HT weight wrong for {k}"
        assert v["n_sampled"] <= v["N_stratum"]
    fitm, repm = region_split(np.array([f"R{i%40}" for i in range(400)]))
    assert not (fitm & repm).any() and (fitm | repm).all()
    assert np.array_equal(fitm, region_split(np.array([f"R{i%40}" for i in range(400)]))[0]), "split is not stable"

    # compare on per-polygon decisions
    from oe_inferencex import compare as cmp_mod
    a = rng.integers(0, N_CLASS, m)
    b = a.copy()
    flip = rng.random(m) < 0.15
    b[flip] = (a[flip] + 1) % N_CLASS
    res = cmp_mod.compare_inferences(a, b, np.ones(m, bool), groups=rng.integers(0, 9, m),
                                     labels=rng.integers(0, N_CLASS, m), cues={"boundary": rd["boundary"] > 0.2})
    assert abs(res["disagreement_rate"] - flip.mean()) < 1e-12, "disagreement rate disagrees with the planted flips"
    assert res["graded"] is not None and res["where"] is not None
    print("smoke OK: weighted estimators reduce to the package's, AUROC matches Mann-Whitney, cue ceiling, "
          "rate-matched null, paired bootstrap, window anchored on the survey point, readings, scoring, design, compare")


# ----------------------------------------------------------------------------- entry
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("fetch", "analyze", "all"), default="all")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.smoke:
        smoke(args)
        return
    if args.stage in ("fetch", "all"):
        fetch_stage(args)
    if args.stage in ("analyze", "all"):
        analyze_stage(args)


if __name__ == "__main__":
    main()
