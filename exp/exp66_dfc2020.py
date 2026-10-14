#!/usr/bin/env python
"""exp66: the protocol on DFC2020, eight-class land cover, both sensors, our own encoder, and two references of different quality.

Why. Ai2 (Patrick Johnson, 2026-09-11) suggested DFC2020 as a labelled evaluation set: Sentinel-1 and Sentinel-2 at 10 m
over the same scenes with a 10 m reference, where every other dense testbed here is binary flood water (Sen1Floods11,
GEOID-Flood) or comes as Ai2's published embeddings rather than our own encoder pass (MADOS, PASTIS in exp54 and exp63).
It adds three things at once: a new dataset our encoder has never seen, eight land-cover classes, and the sensor axis on a
land task rather than on water (exp60's sensor axis was flood water; exp63's was PASTIS through their embeddings).

It also carries something no other testbed here does, and it is the reason this experiment has a third part. The DFC2020
reference is NOT hand-drawn, despite how the contest is usually described: it is an iterated Google Earth Engine random
forest over Sentinel-1, Sentinel-2, spectral indices, the 500 m MODIS map and FROM-GLC10, with a published overall
accuracy of 82.4% and average precision 76.3% (Yokoya, Ghamisi, Hansch, Schmitt; verified 2026-09-11). The same patches
also ship the 500 m MODIS labels the contest called weak. So one dataset carries two references of KNOWN and different
quality over identical pixels, which is the clean instrument this repository has never had for its oldest caveat: that
reference-product labels flatter boundary-type signals (exp18, docs/method/protocol.md). Every number here is therefore
graded twice, against the 10 m reference and against the 500 m one, and the third preregistration is about the gap.

Design. Data: the ungated HuggingFace mirror `125oii/dfc2020` (byte counts reproduce the IEEE DataPort manifest; IEEE
DataPort itself is not scriptable, single sign-on plus manual approval). The validation split (986 patches) fits the
probes, the test split (5,128) is reported, which is the convention in the literature; both carry the 10 m reference, and
the two splits differ in class composition, so this is a shift, not an i.i.d. split, and it is stated not corrected.
Patches are 256 x 256 at 10 m, cut into sixteen 64 px chips; the encoder reads the 60 px crop of each, giving 15 x 15
windows of 4 px. Sentinel-2 is L1C top-of-atmosphere, 13 bands, gathered into the encoder's twelve (B10 dropped) with
digital numbers unchanged; Sentinel-1 is already sigma-nought in decibels as float64, cast and passed through. The
encoder is OlmoEarth v1 Base, frozen; the probe is their linear recipe in fp32 (exp54's port), fitted per arm and per
seed. Arms: Sentinel-2 alone, Sentinel-1 alone, and both together (the pooled tokens of the two modalities concatenated),
two probe seeds each, the second seed giving the noise floor of a difference rate.

  Part A, ranking. Windows are ranked by the model's own margin (top-1 minus top-2 window probability), against
  predictive entropy, the boundary-first order (exp36), the no-model NDWI level control computed from the pixels, and
  cross-arm disagreement. Scored as everywhere here: pooled excess AURC, a one-sided per-patch sign test, error capture
  at 5, 10 and 20 percent.
  Part B, comparison. oe_inferencex.compare on identical windows: Sentinel-2 against Sentinel-1 (the sensor axis),
  Sentinel-2 against the joint arm, and seed 0 against seed 1 within each arm (the floor). Cue enrichment on the
  differing windows (boundary, bottom margin quintile, top entropy quintile), which side is right with the multi-class
  neither, the cross-tab, per patch with compare.over_groups.
  Part C, the two references. Every ranker of part A is scored again against the 500 m MODIS reference on the same
  windows, after the contest's IGBP to DFC remapping.

Preregistered (one-sided):
  P1  on every arm the model's own margin beats the no-model NDWI control against the 10 m reference: pooled excess AURC
      lead at least 0.001 and a per-patch sign test p < 0.05 (three tests).
  P2  the sensor difference is real and located: Sentinel-2 against Sentinel-1 differs on at least ten times the share of
      windows that a second probe seed does, and its differing windows carry the boundary cue more than twice as often as
      the agreeing ones.
  P3  the weaker reference flatters the boundary order: the boundary-first order's lead over the margin is larger when
      graded against the 500 m MODIS reference than against the 10 m reference, pooled, and larger on more patches than
      not (p < 0.05).
  Falsification. P1 fails if any arm's margin does not beat a pixel index, which would say the ranking does not transfer
  to eight-class land cover under our own encoder. P2 fails if the sensor difference is within ten times the seed floor
  (it would then be noise) or is not boundary-located (the atlas finding would not hold on land cover). P3 fails if the
  gap is zero or negative, which would retire exp18's caveat rather than confirm it; a negative gap would be the more
  interesting result and must be reported as such.
  Stated predictions, not tested: the joint arm beats both single-sensor arms on accuracy; Sentinel-1 alone trails
  Sentinel-2 alone by roughly ten points of mean intersection over union; the boundary cue is weaker here than on
  Sen1Floods11 and stronger than on PASTIS, since eight classes sit between two and nineteen.

Caveats carried into the record. The imagery is L1C top-of-atmosphere while the encoder's modality is Sentinel-2 L2A
surface reflectance: the inputs are off-manifold and a linear probe absorbs only a constant offset. The split zips carry
no real georeferencing (identity transform, nominal EPSG:4326), so windows are addressed by patch and pixel, never by
map coordinates. The reference's own errors are correlated with what a linear probe on these features learns, so a
"captured error" here includes reference error; that is the point of part C, not a flaw to hide. Publishing a DFC2020
number requires case-by-case approval from the IEEE GRSS IADF technical committee and TUM under the contest terms:
running and recording internally is fine, and any paper use needs that approval first.

Inputs: the mirror (about 10.4 GB, cached under $HF_HOME/dfc2020). Run with ~/oe12/.venv (rasterio and the encoder).
Outputs: exp/out/exp66_summary.json, exp/out/exp66_dfc2020.csv (one row per arm and reference, one per compared pair),
exp/out/exp66_masks.npz (window decisions, margins, entropy and where each reference is valid, for a capped sample of
patches; no reference classes, which are DFC2020 data and stay out of the repository).
--smoke: synthetic patches on CPU, no download, _smoke outputs.
"""
import csv
import json
import os
import sys
import time
import zipfile

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp46_shared_error_sources as e46  # noqa: E402
import exp54_multiclass_embeddings as e54  # noqa: E402
import exp57_difference_atlas as e57  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.assess import boundary_first_score  # noqa: E402
from oe_inferencex.compare import compare_inferences, stability  # noqa: E402
from oe_inferencex.explain import top_fraction  # noqa: E402
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import ndwi_level  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402

REPO = "125oii/dfc2020"
ZIPS = ["s2_validation.zip", "s1_validation.zip", "dfc_validation.zip", "lc_validation.zip",
        "s2_0.zip", "s1_0.zip", "dfc_0.zip", "lc_0.zip"]
PATCH_PX, CHIP, CROP, PATCH, G = 256, 64, exp18.CROP, exp18.PATCH, exp18.G
# DFC 1-10: Forest, Shrubland, Savanna, Grassland, Wetlands, Croplands, Urban, Snow/Ice, Barren, Water.
DFC_NAMES = {1: "forest", 2: "shrubland", 3: "savanna", 4: "grassland", 5: "wetlands", 6: "croplands", 7: "urban", 8: "snow_ice", 9: "barren", 10: "water"}
PRESENT = (1, 2, 4, 5, 6, 7, 9, 10)                                  # savanna and snow/ice have no pixels in either split
CODE_TO_CLASS = {c: i for i, c in enumerate(PRESENT)}
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])   # the contest's simplification of IGBP
ARMS = ("s2", "s1", "s1s2")
SEEDS = (0, 1)
REFS = ("dfc", "lc")
CONF, ENT, LEX, CTRL, DIS = "margin", "entropy", "boundary first, then margin", "control NDWI level", "cross-arm disagreement"
BUDGETS = (0.05, 0.10, 0.20)
MIN_LEAD, MIN_ENRICH, FLOOR_RATIO = 0.001, 2.0, 10.0
fmt = e57.fmt
DEV = exp18.DEV


# ----------------------------------------------------------------------------- data
def fetch(data_dir, smoke=False):
    """The mirror's eight zips, extracted once into data_dir/tree; markers skip finished archives."""
    from huggingface_hub import snapshot_download
    root = os.path.join(data_dir, "tree")
    done = os.path.join(root, ".done")
    os.makedirs(done, exist_ok=True)
    if all(os.path.exists(os.path.join(done, z + ".done")) for z in ZIPS):
        return root, {"cached": True}
    snap = snapshot_download(REPO, repo_type="dataset", allow_patterns=ZIPS, local_dir=os.path.join(data_dir, "zips"), max_workers=8)
    got = {}
    for z in ZIPS:
        marker = os.path.join(done, z + ".done")
        if os.path.exists(marker):
            continue
        t0 = time.time()
        with zipfile.ZipFile(os.path.join(snap, z)) as zf:
            members = [m for m in zf.namelist() if m.endswith(".tif") and "__MACOSX" not in m and not os.path.basename(m).startswith("._")]
            zf.extractall(root, members)
        got[z] = {"members": len(members), "seconds": time.time() - t0}
        open(marker, "w").write(str(len(members)))
        print(f"  {z}: {len(members)} rasters in {time.time() - t0:.0f}s", flush=True)
    return root, got


def index_split(root, split):
    """{patch id: {modality: path}} for a split; the archives unzip flat or one level down, so walk."""
    want = {"s1": f"_s1_", "s2": f"_s2_", "dfc": f"_dfc_", "lc": f"_lc_"}
    out = {}
    for dirpath, _, names in os.walk(root):
        for fn in names:
            if not fn.endswith(".tif") or split not in fn:
                continue
            for mod, tag in want.items():
                if tag in fn:
                    pid = fn.split(tag)[-1].replace(".tif", "")
                    out.setdefault(pid, {})[mod] = os.path.join(dirpath, fn)
    return {k: v for k, v in sorted(out.items()) if all(m in v for m in ("s1", "s2", "dfc", "lc"))}


def read_patch(paths, s2_gather):
    """One 256-px patch -> s2 (12, 256, 256) float32 DN in encoder order, s1 (2, 256, 256) float32 dB, dfc and lc codes."""
    import rasterio
    with rasterio.open(paths["s2"]) as src:
        s2 = src.read(s2_gather).astype(np.float32)
    with rasterio.open(paths["s1"]) as src:
        s1 = np.nan_to_num(src.read([1, 2]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    with rasterio.open(paths["dfc"]) as src:
        dfc = src.read(1).astype(np.int32)
    with rasterio.open(paths["lc"]) as src:
        lc = IGBP2DFC[np.clip(src.read(1).astype(np.int32), 0, 17)]
    return s2, s1, dfc, lc


def to_class(codes):
    """DFC codes -> contiguous classes with -1 for absent or unlabelled."""
    out = np.full(codes.shape, -1, np.int64)
    for c, i in CODE_TO_CLASS.items():
        out[codes == c] = i
    return out


def chips_of(arr):
    """(C, 256, 256) -> (16, C, 64, 64), row-major over the 4 x 4 grid."""
    c = arr.shape[0]
    return arr.reshape(c, 4, CHIP, 4, CHIP).transpose(1, 3, 0, 2, 4).reshape(16, c, CHIP, CHIP)


def label_chips(arr):
    return arr.reshape(4, CHIP, 4, CHIP).transpose(0, 2, 1, 3).reshape(16, CHIP, CHIP)


def load_chips(index, ids, s2_gather):
    """The chips of a list of patches: s2, s1, the two references as contiguous classes, and the patch id per chip."""
    s2, s1, dfc, lc, pid = [], [], [], [], []
    for k, p in enumerate(ids):
        a, b, d, l = read_patch(index[p], s2_gather)
        s2.append(chips_of(a)); s1.append(chips_of(b))
        dfc.append(label_chips(to_class(d))); lc.append(label_chips(to_class(l)))
        pid.append(np.full(16, k, np.int32))
    return (np.concatenate(s2), np.concatenate(s1), np.concatenate(dfc), np.concatenate(lc), np.concatenate(pid))


# ----------------------------------------------------------------------------- encoder
def embed_joint(model, s2, s1, batch=32):
    """Both modalities in one pass, the pooled tokens concatenated: (N, G, G, 2D)."""
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    out = []
    for i in range(0, len(s2), batch):
        x2 = s2[i:i + batch, :, :CROP, :CROP].transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
        x2 = exp18._norm.normalize(Modality.SENTINEL2_L2A, x2)
        x1 = np.nan_to_num(s1[i:i + batch, :, :CROP, :CROP], nan=0.0).transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
        x1 = exp18._norm.normalize(Modality.SENTINEL1, x1)
        b = x2.shape[0]
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.tensor(x2, dtype=torch.float32, device=DEV),
            sentinel2_l2a_mask=torch.ones((b, CROP, CROP, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
            sentinel1=torch.tensor(x1, dtype=torch.float32, device=DEV),
            sentinel1_mask=torch.ones((b, CROP, CROP, 1, 1), device=DEV) * MaskValue.ONLINE_ENCODER.value,
            timestamps=torch.tensor([1, 5, 2020], device=DEV)[None, None, :].repeat(b, 1, 1))
        with torch.no_grad():
            tok = model.encoder(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"]
        out.append(torch.cat([tok.sentinel2_l2a.mean(dim=[3, 4]), tok.sentinel1.mean(dim=[3, 4])], dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def features(model, arm, s2, s1, chunk=256):
    """(N, G, G, D) float16: half precision keeps the joint arm's 1536-d features inside memory; the probe casts to fp32."""
    if arm == "s2":
        return np.concatenate([np.asarray(exp18.embed(model, s2[i:i + chunk], 0)[0], np.float16) for i in range(0, len(s2), chunk)])
    if arm == "s1":
        return np.concatenate([np.asarray(e46.embed_s1(model, s1[i:i + chunk]), np.float16) for i in range(0, len(s1), chunk)])
    return np.concatenate([embed_joint(model, s2[i:i + chunk], s1[i:i + chunk]).astype(np.float16) for i in range(0, len(s2), chunk)])


def synthetic(n_patches, seed=0):
    """The smoke's stand-in for the mirror: patches whose classes follow a smooth field, so a probe can learn something."""
    rng = np.random.default_rng(seed)
    s2 = rng.random((n_patches, 12, PATCH_PX, PATCH_PX), np.float32) * 3000 + 500
    s1 = rng.normal(-14, 3, (n_patches, 2, PATCH_PX, PATCH_PX)).astype(np.float32)
    yy, xx = np.mgrid[:PATCH_PX, :PATCH_PX]
    dfc, lc = [], []
    for k in range(n_patches):
        f = ((yy + 40 * np.sin(xx / 30 + k)) // 48 % len(PRESENT)).astype(np.int64)
        dfc.append(f)
        coarse = f.reshape(PATCH_PX // 32, 32, PATCH_PX // 32, 32)
        lc.append(np.repeat(np.repeat(coarse[:, 0, :, 0], 32, 0), 32, 1))          # a 500 m analogue: block-constant
        s2[k, 1] += 400 * f; s2[k, 3] -= 300 * f; s1[k, 0] += 1.5 * f
    return s2, s1, np.stack(dfc), np.stack(lc)


# ----------------------------------------------------------------------------- scoring
def score(sig, err, ok, ref_name, primary=(CTRL,)):
    """exp47's protocol per patch: pooled excess AURC, one-sided sign tests against the margin, capture at budgets."""
    per_patch = {}
    pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum())) for k, v in sig.items()}
    pat = np.unique(PATCH_OF[ok])
    for k in sig:
        vals = []
        for p in pat:
            m = ok & (PATCH_OF == p)
            e = err[m]
            if 3 <= e.sum() <= len(e) - 3:
                vals.append(aurc_expected(np.asarray(sig[k])[m], e) - oracle_aurc(int(m.sum()), int(e.sum())))
        per_patch[k] = vals
    tests = {}
    for k in sig:
        if k == CONF:
            continue
        g = np.array(per_patch[k]) - np.array(per_patch[CONF])                    # positive: the margin is better
        w, l, t = wins_losses_ties(g)
        one = k in primary
        tests[k] = {"w": w, "l": l, "t": t, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                    "median_gain": float(np.median(g)) if len(g) else None, "pooled_lead": pooled[k] - pooled[CONF]}
    cap = {k: {str(b): v for b, v in capture_at_budget_expected(np.asarray(sig[k])[ok], err[ok], BUDGETS).items()} for k in sig}
    return {"reference": ref_name, "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()), "accuracy": float(1 - err[ok].mean()),
            "n_patches_scored": len(per_patch[CONF]), "pooled_excess_aurc": pooled, "tests": tests, "capture": cap}


PATCH_OF = None          # set once the windows exist; the per-patch unit of every test


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--fit-patches", type=int, default=400, help="validation patches used to fit the probes")
    ap.add_argument("--test-patches", type=int, default=1200, help="test patches reported")
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=ARMS)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    t_start = time.time()
    global PATCH_OF
    summary = {"experiment": "exp66 the protocol on DFC2020: eight-class land cover, both sensors, our own encoder, two references",
               "smoke": args.smoke, "device": DEV,
               "config": {"repo": REPO, "arms": list(args.arms), "seeds": list(SEEDS), "classes": {int(c): DFC_NAMES[c] for c in PRESENT},
                          "fit_patches": args.fit_patches, "test_patches": args.test_patches, "chip_px": CHIP, "crop": CROP, "patch": PATCH,
                          "budgets": list(BUDGETS), "references": {"dfc": "the contest's 10 m reference, an iterated GEE random forest, published OA 0.824",
                                                                   "lc": "the 500 m MODIS labels remapped by the contest's IGBP2DFC"},
                          "caveats": ["Sentinel-2 is L1C top-of-atmosphere; the encoder's modality is L2A surface reflectance",
                                      "the split zips carry no real georeferencing; windows are addressed by patch and pixel",
                                      "validation and test differ in class composition; this is a shift, not an i.i.d. split",
                                      "publishing a DFC2020 number needs IEEE GRSS IADF and TUM approval under the contest terms"],
                          "prereg": "P1 the margin beats the NDWI control against the 10 m reference on every arm; "
                                    "P2 the sensor difference exceeds ten times the probe-seed floor and is boundary-enriched above 2x; "
                                    "P3 the boundary-first order's lead over the margin is larger against the 500 m reference than against the 10 m one"},
               "results": {"arms": {}, "pairs": {}, "reference_gap": {}}, "failures": []}
    rows, store = [], {}
    try:
        # DFC2020 stores 13 bands B01..B12 with B8A ninth; gather into the encoder's order, dropping B10 (index 11, 1-based)
        stored = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        s2_gather = [stored.index(b) + 1 for b in exp18.OE_BANDS]
        summary["config"]["s2_gather_1based"] = s2_gather
        if args.smoke:
            n_fit, n_test = 3, 4
            s2_f, s1_f, dfc_f, lc_f = synthetic(n_fit, 0)
            s2_t, s1_t, dfc_t, lc_t = synthetic(n_test, 1)
            fit = (np.concatenate([chips_of(x) for x in s2_f]), np.concatenate([chips_of(x) for x in s1_f]),
                   np.concatenate([label_chips(x) for x in dfc_f]), np.concatenate([label_chips(x) for x in lc_f]),
                   np.concatenate([np.full(16, k, np.int32) for k in range(n_fit)]))
            test = (np.concatenate([chips_of(x) for x in s2_t]), np.concatenate([chips_of(x) for x in s1_t]),
                    np.concatenate([label_chips(x) for x in dfc_t]), np.concatenate([label_chips(x) for x in lc_t]),
                    np.concatenate([np.full(16, k, np.int32) for k in range(n_test)]))
            summary["config"]["patches"] = {"fit": n_fit, "test": n_test, "synthetic": True}
        else:
            data_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "dfc2020")
            t0 = time.time()
            root, got = fetch(data_dir)
            summary["config"]["fetch"] = {"seconds": time.time() - t0, "archives": got}
            idx_fit, idx_test = index_split(root, "validation"), index_split(root, "test")
            ids_fit = sorted(idx_fit)[:args.fit_patches]
            ids_test = sorted(idx_test)[:args.test_patches] if args.test_patches else sorted(idx_test)
            summary["config"]["patches"] = {"validation_available": len(idx_fit), "test_available": len(idx_test), "fit": len(ids_fit), "test": len(ids_test)}
            print(f"patches: {len(idx_fit)} validation ({len(ids_fit)} used to fit), {len(idx_test)} test ({len(ids_test)} reported)", flush=True)
            t0 = time.time()
            fit, test = load_chips(idx_fit, ids_fit, s2_gather), load_chips(idx_test, ids_test, s2_gather)
            summary["config"]["read_seconds"] = time.time() - t0
        s2_f, s1_f, dfc_f, _, _ = fit
        s2_t, s1_t, dfc_t, lc_t, pid_t = test
        C = len(PRESENT)
        print(f"chips: {len(s2_f)} to fit, {len(s2_t)} to report; {C} classes", flush=True)
        model = hb.load_model()
        held = {}
        for arm in args.arms:
            t0 = time.time()
            f_fit = features(model, arm, s2_f, s1_f)
            f_test = features(model, arm, s2_t, s1_t)
            enc_s = time.time() - t0
            for seed in SEEDS:
                probe = e54.train_probe(torch.tensor(f_fit), torch.tensor(dfc_f[:, :CROP, :CROP]), PATCH, C, 0.1, seed=seed)
                q = e54.window_quantities(probe, torch.tensor(f_test), torch.tensor(dfc_t[:, :CROP, :CROP]), PATCH, C)
                held[(arm, seed)] = q
                if seed == 0:
                    summary["results"]["arms"][arm] = {"encode_seconds": enc_s, "pixel_accuracy": q["pixel_accuracy"], "pixel_miou": q["pixel_miou"], "n_windows": int(q["ok"].sum())}
                    print(f"{arm}: pixel acc {q['pixel_accuracy']:.4f}, mIoU {q['pixel_miou']:.3f}, {int(q['ok'].sum())} windows ({enc_s:.0f}s encode)", flush=True)
            del f_fit, f_test
            if DEV == "cuda":
                torch.cuda.empty_cache()
        del model
        ref0 = held[(args.arms[0], 0)]
        PATCH_OF = np.broadcast_to(pid_t[:, None, None], ref0["dec"].shape)
        # the second reference pooled onto the same windows
        hw = ref0["dec"].shape[1]
        l = lc_t[:, :CROP, :CROP][:, :hw * PATCH, :hw * PATCH].reshape(len(lc_t), hw, PATCH, hw, PATCH).transpose(0, 1, 3, 2, 4).reshape(len(lc_t), hw, hw, PATCH * PATCH)
        valid = l >= 0
        counts = np.stack([((l == c) & valid).sum(-1) for c in range(C)], -1)
        yl, okl = counts.argmax(-1), valid.sum(-1) >= (PATCH * PATCH) // 2
        summary["results"]["reference_gap"]["agreement_of_the_two_references"] = float((yl == ref0["y"])[ref0["ok"] & okl].mean())
        print(f"the two references agree on {summary['results']['reference_gap']['agreement_of_the_two_references']:.3f} of the windows both label", flush=True)
        # ---- Part A and C: rankers against each reference
        ndwi = np.stack([ndwi_level(x, patch=PATCH, size=CROP) for x in s2_t])
        for arm in args.arms:
            q = held[(arm, 0)]
            others = [held[(a, 0)]["dec"] for a in args.arms if a != arm]
            sig = {CONF: -q["margin"], ENT: q["entropy"], CTRL: ndwi,
                   DIS: np.mean([(o != q["dec"]) for o in others], axis=0) if others else np.zeros_like(q["margin"])}
            sig[LEX] = np.stack([boundary_first_score(sig[CONF][t], e54.boundary_share(q["dec"])[t]) for t in range(len(q["dec"]))])
            for ref, y, ok in (("dfc", q["y"], q["ok"]), ("lc", yl, okl & q["ok"])):
                err = ((q["dec"] != y) & ok).astype(np.float64)
                r = score(sig, err, ok, ref)
                summary["results"]["arms"].setdefault(arm, {}).setdefault("scores", {})[ref] = r
                rows.append({"kind": "rank", "arm": arm, "reference": ref, "n_windows": r["n_windows"], "accuracy": r["accuracy"],
                             **{f"eaurc {k}": v for k, v in r["pooled_excess_aurc"].items()},
                             "lead over control": r["tests"][CTRL]["pooled_lead"], "control p": r["tests"][CTRL]["sign_p"],
                             "lead boundary-first": r["tests"][LEX]["pooled_lead"], "capture5 margin": r["capture"][CONF]["0.05"]})
                print(f"{arm}/{ref}: acc {r['accuracy']:.4f}, {r['n_errors']} errors | E-AURC margin {r['pooled_excess_aurc'][CONF]:.4f}, control {r['pooled_excess_aurc'][CTRL]:.4f} "
                      f"(lead {r['tests'][CTRL]['pooled_lead']:+.4f}, p={r['tests'][CTRL]['sign_p']:.2g}) | boundary-first lead {r['tests'][LEX]['pooled_lead']:+.4f} | capture@5% {r['capture'][CONF]['0.05']:.3f}", flush=True)
        # ---- Part B: comparison
        pairs = {}
        if "s2" in args.arms and "s1" in args.arms:
            pairs["sensors"] = (("s2", 0), ("s1", 0))
        if "s2" in args.arms and "s1s2" in args.arms:
            pairs["s2 vs joint"] = (("s2", 0), ("s1s2", 0))
        for arm in args.arms:
            pairs[f"{arm} draw 0 vs 1"] = ((arm, 0), (arm, 1))
        dis = {}
        for name, (ka, kb) in pairs.items():
            A, B = held[ka], held[kb]
            ok = A["ok"] & B["ok"]
            cues = {"boundary": e54.boundary_share(A["dec"]) > 0, "low_margin": top_fraction(-A["margin"], 0.2, ok), "high_entropy": top_fraction(A["entropy"], 0.2, ok)}
            out = compare_inferences(A["dec"], B["dec"], ok, groups=pid_t, labels=A["y"], cues=cues)
            dis[name] = out["arrays"]["disagree"]
            r = {k: v for k, v in out.items() if k != "arrays"}
            r["accuracy_a"] = float((A["dec"] == A["y"])[ok].mean()); r["accuracy_b"] = float((B["dec"] == A["y"])[ok].mean())
            summary["results"]["pairs"][name] = r
            w, g = r["where"], r["graded"]; ws = g["which_side"]
            rows.append({"kind": "pair", "arm": name, "reference": "dfc", "n_windows": r["n_windows"], "n_disagree": r["n_disagree"], "rate": r["disagreement_rate"],
                         "boundary_enrichment": w["boundary"]["enrichment"], "low_margin_enrichment": w["low_margin"]["enrichment"],
                         "share_a_right": ws["share_a_right"], "share_b_right": ws["share_b_right"], "errors_phi": g["crosstab"]["phi"]})
            print(f"pair {name}: differ {r['n_disagree']} of {r['n_windows']} ({fmt(100 * r['disagreement_rate'], '.2f')}%) | boundary {fmt(w['boundary']['enrichment'], '.2f')}x, low margin {fmt(w['low_margin']['enrichment'], '.2f')}x | "
                  f"a right {fmt(ws['share_a_right'], '.3f')}, b right {fmt(ws['share_b_right'], '.3f')} | acc {r['accuracy_a']:.4f}/{r['accuracy_b']:.4f}", flush=True)
        if len(dis) > 1:
            summary["results"]["stability_across_pairs"] = stability(dis)
        for arm in args.arms:
            q = held[(arm, 0)]
            store[f"{arm}/dec"] = q["dec"].astype(np.int8); store[f"{arm}/margin"] = q["margin"].astype(np.float16); store[f"{arm}/entropy"] = q["entropy"].astype(np.float16)
            store[f"{arm}/dec_seed1"] = held[(arm, 1)]["dec"].astype(np.int8)
        # no DFC2020 labels are committed: the files are released to approved contest participants, so the store keeps
        # the model outputs and the references' validity masks, never the reference classes (removed 5 October 2026)
        store["ok_dfc"], store["ok_lc"], store["patch"] = ref0["ok"], okl, pid_t
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "pipeline", ex)
    # ---- prereg
    try:
        A = summary["results"]["arms"]; P = summary["results"]["pairs"]
        p1 = [(A[a]["scores"]["dfc"]["tests"][CTRL]["pooled_lead"] >= MIN_LEAD and A[a]["scores"]["dfc"]["tests"][CTRL]["sign_p"] < 0.05)
              for a in A if "scores" in A[a]]
        floor = max((P[k]["disagreement_rate"] for k in P if "draw 0 vs 1" in k), default=None)
        sens = P.get("sensors")
        p2 = None if not (sens and floor) else bool(sens["disagreement_rate"] >= FLOOR_RATIO * floor and sens["where"]["boundary"]["enrichment"] > MIN_ENRICH)
        gaps = {}
        for a in A:
            if "scores" not in A[a]:
                continue
            gaps[a] = {r: -A[a]["scores"][r]["tests"][LEX]["pooled_lead"] for r in REFS if r in A[a]["scores"]}   # positive: boundary-first better than the margin
            gaps[a]["gap_lc_minus_dfc"] = gaps[a].get("lc", float("nan")) - gaps[a].get("dfc", float("nan"))
        summary["results"]["reference_gap"]["boundary_first_lead_over_margin"] = gaps
        vals = [v["gap_lc_minus_dfc"] for v in gaps.values() if np.isfinite(v.get("gap_lc_minus_dfc", np.nan))]
        w, l, t = wins_losses_ties(np.array(vals)) if vals else (0, 0, 0)
        p3 = bool(vals and all(v > 0 for v in vals))
        summary["prereg"] = {"P1": all(p1) if len(p1) == len(args.arms) else None, "P2": p2, "P3": p3 if vals else None,
                             "P3_detail": {"per_arm_gap": {k: v["gap_lc_minus_dfc"] for k, v in gaps.items()}, "arms_positive": w, "arms_negative": l},
                             "seed_floor_rate": floor, "sensor_rate": sens["disagreement_rate"] if sens else None,
                             "complete": bool(len(p1) == len(args.arms) and p2 is not None and vals and not summary["failures"])}
        print(f"prereg: P1 {summary['prereg']['P1']} | P2 {summary['prereg']['P2']} (sensor rate {fmt(summary['prereg']['sensor_rate'], '.4f')} vs seed floor {fmt(floor, '.4f')}) | "
              f"P3 {summary['prereg']['P3']} (gap per arm { {k: round(v['gap_lc_minus_dfc'], 4) for k, v in gaps.items()} }) | complete {summary['prereg']['complete']}", flush=True)
    except Exception as ex:  # noqa: BLE001
        e57.fail(summary, "prereg", ex)
        summary["prereg"] = {"P1": None, "P2": None, "P3": None, "complete": False}
    summary["n_failures"] = len(summary["failures"])
    summary["seconds"] = time.time() - t_start
    os.makedirs(hb.OUT, exist_ok=True)
    if store:
        np.savez_compressed(os.path.join(hb.OUT, f"exp66_masks{suffix}.npz"), **store)
    with open(os.path.join(hb.OUT, f"exp66_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    if rows:
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        with open(os.path.join(hb.OUT, f"exp66_dfc2020{suffix}.csv"), "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=fields)
            w_.writeheader()
            w_.writerows(rows)
    print(f"done in {summary['seconds']:.0f}s, {summary['n_failures']} failures", flush=True)


if __name__ == "__main__":
    main()
