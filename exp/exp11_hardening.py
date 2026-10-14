"""Exp 11: statistical hardening of the multi-scene comparison.

PRE-REGISTERED SCENE RULE (committed before any new scene was fetched):
Candidate scene centers are sampled deterministically along OSM
waterway=river ways whose name matches one of eight named rivers (Zambezi,
Kafue, Luangwa, Okavango, Shire, Cuando, Rovuma, Save) inside the bounding
box 20-37E, 20-8S. For each river, all matching way geometries are fetched
from Overpass, node coordinates are concatenated in the order returned, and
candidates are taken at fixed fractional positions 0.2, 0.5, 0.8 of the node
list. Candidates closer than 0.2 degrees to a previously used AOI or to an
earlier candidate are dropped. Each surviving candidate is fetched (dry
season 2024, <5% cloud, 132 px window) and enters the evaluation set if and
only if the seed-0 Base head commits at least 8 errors against WorldCover on
it. The seven exp09 scenes are retained. No candidate may be added or
removed on any other ground.

STATISTICS:
- Per scene: paired bootstrap over patches (B=1000): 95% CI for each
  signal's AURC and for each signal-minus-baseline difference; the fraction
  of resamples in which the signal outranks the baseline.
- Across scenes: for each signal vs the baseline, a sign-flip permutation
  test (10000 permutations, two-sided) on per-scene AURC differences.
- Head-seed variance: heads retrained with five seeds; AURC spread reported
  per signal (embeddings fixed; only head initialization varies).
Signals: baseline max-softmax, E_case |Nano-Base|, E_system tile-phase,
E_dist knn-to-train, NDWI-gradient control. Errors are defined by the
seed-0 Base head throughout, as in exp09.
"""
# NOTE (post-hoc, 2026-09-01): the five-seed head test above is vacuous. Heads are
# zero-initialised and trained full-batch (evidence.py train_logistic_head), so
# seed_std in exp11_stats.csv is float noise. Also, the exp09 cache import below
# swept in two unsuffixed exp09 first-attempt AOIs (kafue, luangwa) that were not
# selected by the rule; exp13-exp15 exclude them (NON_RULE). Retained for the record.
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window
from oe_inferencex.evidence import train_logistic_head, predict_head, pool_to_patches

SIZE, PATCH, PAD = 128, 4, 4
GRID = SIZE // PATCH
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCENE_CACHE = "exp/out/exp11_scenes.npz"
FEAT_CACHE = "exp/out/exp11_feats.npz"
RIVERS = ["Zambezi", "Kafue", "Luangwa", "Okavango", "Shire", "Cuando", "Rovuma", "Save"]
BBOX = (-20.0, 20.0, -8.0, 37.0)  # S, W, N, E
FRACS = (0.2, 0.5, 0.8)
MIN_SEP = 0.2
SEEDS = (0, 1, 2, 3, 4)
B_BOOT = 1000
N_PERM = 10000

EXISTING = {
    "kazungula": (25.263, -17.788), "barotse": (23.090, -15.400),
    "delta": (36.180, -18.850), "luangwa_conf": (30.418, -15.617),
    "okavango_sep": (22.185, -18.735), "shire_liwonde": (35.230, -15.060),
    "vicfalls_up": (25.840, -17.880),
}
MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]


def overpass(query):
    for url in MIRRORS:
        try:
            req = urllib.request.Request(
                url, data=("data=" + urllib.parse.quote(query)).encode(),
                headers={"User-Agent": "oe-inferencex-exp11"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())["elements"]
        except Exception as exc:
            print(f"  overpass {url} failed: {type(exc).__name__}")
    raise RuntimeError("all overpass mirrors failed")


CANDIDATE_CACHE = "exp/out/rule_candidates.json"


def candidate_centers():
    """Rule-selected scene centres. Cached in exp/out/rule_candidates.json the
    first time every river's OSM geometry is retrieved, so later experiments
    use the same scene set regardless of Overpass mirror availability."""
    if os.path.exists(CANDIDATE_CACHE):
        return [tuple(c) for c in json.load(open(CANDIDATE_CACHE))]
    for attempt in range(3):
        kept, complete = _candidate_centers_once()
        if complete:
            json.dump(kept, open(CANDIDATE_CACHE, "w"))
            return kept
        print(f"candidate rule incomplete (attempt {attempt + 1}); retrying")
    return kept


def _candidate_centers_once():
    s, w, n, e = BBOX
    cands, complete = [], True
    for river in RIVERS:
        try:
            ways = overpass(
                f'[out:json][timeout:90];way["waterway"="river"]["name"~"{river}"]({s},{w},{n},{e});out geom;')
        except RuntimeError:
            print(f"{river}: overpass failed entirely, skipped")
            complete = False
            continue
        coords = [(nd["lon"], nd["lat"]) for way in ways for nd in way.get("geometry", [])]
        print(f"{river}: {len(ways)} ways, {len(coords)} nodes")
        if len(coords) < 10:
            complete = False
            continue
        for frac in FRACS:
            lon, lat = coords[round(frac * (len(coords) - 1))]
            cands.append((f"{river.lower()}_{int(frac*100)}", lon, lat))
    used = list(EXISTING.values())
    kept = []
    for name, lon, lat in cands:
        if all(max(abs(lon - a), abs(lat - b)) >= MIN_SEP for a, b in used):
            kept.append((name, lon, lat))
            used.append((lon, lat))
        else:
            print(f"{name}: within {MIN_SEP} deg of an existing AOI, dropped")
    return kept, complete


def embed_gpu(model, img, date):
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    x = img.transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = date
    xs = torch.tensor(x, dtype=torch.float32, device=DEV)
    ts = torch.tensor([d, m0, y], device=DEV)[None, None, :]
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=xs,
        sentinel2_l2a_mask=torch.ones((1, img.shape[1], img.shape[2], 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts)
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])[0].cpu().numpy()


def aurc(sig, err):
    order = np.argsort(sig.flatten(), kind="stable")
    e = err.flatten()[order]
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def ndwi_gradient(img):
    bo = Modality.SENTINEL2_L2A.band_order
    x = img[:, :SIZE, :SIZE].astype(np.float64)
    nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(
        x[bo.index("B03")] + x[bo.index("B08")], 1, None)
    gy, gx = np.gradient(nd)
    return np.hypot(gx, gy).reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3))


def main():
    scenes = dict(np.load(SCENE_CACHE, allow_pickle=True)) if os.path.exists(SCENE_CACHE) else {}
    if not scenes:
        z9 = np.load("exp/out/exp09_cache.npz", allow_pickle=True)
        for k in z9.files:
            scenes[k] = z9[k]
    names_existing = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    print(f"existing scenes: {names_existing}")

    if "rule_candidates_done" not in scenes:
        for name, lon, lat in candidate_centers():
            key = f"{name}_img"
            if key in scenes:
                continue
            try:
                img, date, (crs, transform) = fetch_s2_window(lon, lat, SIZE + PAD)
                wc = fetch_worldcover_window(lon, lat, crs, transform, SIZE + PAD)
            except Exception as exc:
                print(f"{name}: fetch failed ({type(exc).__name__})")
                continue
            lab = (pool_to_patches(wc[:SIZE, :SIZE] == 80, PATCH) > 0.5).astype(np.float32)
            scenes[f"{name}_img"], scenes[f"{name}_date"], scenes[f"{name}_lab"] = img, np.array(date), lab
            np.savez(SCENE_CACHE, **scenes)
            print(f"{name}: fetched ({lon:.3f},{lat:.3f})")
        scenes["rule_candidates_done"] = np.array(1)
        np.savez(SCENE_CACHE, **scenes)

    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    models = {k: load_model_from_id(m).to(DEV).eval()
              for k, m in (("nano", ModelID.OLMOEARTH_V1_NANO), ("base", ModelID.OLMOEARTH_V1_BASE))}
    z3 = np.load("exp/out/exp03_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]

    feats = dict(np.load(FEAT_CACHE, allow_pickle=True)) if os.path.exists(FEAT_CACHE) else {}
    if "tr_base" not in feats:
        feats["tr_base"] = embed_gpu(models["base"], tr_img, tr_date)
        feats["tr_nano"] = embed_gpu(models["nano"], tr_img, tr_date)
    for name in names:
        if f"{name}_base0" in feats:
            continue
        img = scenes[f"{name}_img"]
        date = tuple(int(v) for v in scenes[f"{name}_date"])
        for s in range(4):
            feats[f"{name}_base{s}"] = embed_gpu(models["base"], img[:, s:s + SIZE, s:s + SIZE], date)
        feats[f"{name}_nano"] = embed_gpu(models["nano"], img[:, :SIZE, :SIZE], date)
        np.savez(FEAT_CACHE, **feats)
        print(f"{name}: features cached")

    heads = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        heads[seed] = {
            "base": train_logistic_head(torch.tensor(feats["tr_base"]), tr_labels),
            "nano": train_logistic_head(torch.tensor(feats["tr_nano"]), tr_labels),
        }

    a_tr = torch.nn.functional.normalize(torch.tensor(feats["tr_base"]).reshape(-1, feats["tr_base"].shape[-1]), dim=-1)
    rng = np.random.default_rng(0)
    rows, per_scene_sig = [], {}
    for name in names:
        lab = scenes[f"{name}_lab"]
        sig_by_seed = {}
        for seed in SEEDS:
            hb, hn = heads[seed]["base"], heads[seed]["nano"]
            p_base = predict_head(torch.tensor(feats[f"{name}_base0"]), *hb)
            p_nano = predict_head(torch.tensor(feats[f"{name}_nano"]), *hn)
            err = ((p_base > 0.5) != lab.astype(bool)).astype(np.float64)
            tile = np.stack([p_base] + [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in (1, 2, 3)]).std(0)
            a_ev = torch.nn.functional.normalize(torch.tensor(feats[f"{name}_base0"]).reshape(-1, feats[f"{name}_base0"].shape[-1]), dim=-1)
            knn = torch.topk(1 - (a_ev @ a_tr.T), k=5, largest=False).values.mean(1).reshape(GRID, GRID).numpy()
            sig_by_seed[seed] = ({
                "baseline": 1 - np.maximum(p_base, 1 - p_base),
                "E_case": np.abs(p_nano - p_base),
                "tile-phase": tile,
                "E_dist": knn,
                "control": ndwi_gradient(scenes[f"{name}_img"]),
            }, err)
        sigs0, err0 = sig_by_seed[0]
        if err0.sum() < 8:
            print(f"{name}: {int(err0.sum())} errors, excluded by rule")
            continue
        per_scene_sig[name] = sig_by_seed
        boot = {sn: [] for sn in sigs0}
        n = err0.size
        e_flat = err0.flatten()
        s_flat = {sn: s.flatten() for sn, s in sigs0.items()}
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, n)
            for sn in sigs0:
                boot[sn].append(aurc(s_flat[sn][idx], e_flat[idx]))
        seed_aurcs = {sn: [aurc(sig_by_seed[sd][0][sn], sig_by_seed[sd][1]) for sd in SEEDS] for sn in sigs0}
        for sn in sigs0:
            bs = np.array(boot[sn])
            bb = np.array(boot["baseline"])
            rows.append({
                "scene": name, "signal": sn, "n_errors": int(err0.sum()),
                "aurc": aurc(sigs0[sn], err0),
                "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                "frac_beats_baseline": float((bs < bb).mean()),
                "seed_std": float(np.std(seed_aurcs[sn])),
            })
        print(f"{name}: {int(err0.sum())} errors, "
              + ", ".join(f"{sn}={aurc(sigs0[sn], err0):.4f}" for sn in sigs0))

    import csv
    with open("exp/out/exp11_stats.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    scene_names = sorted(per_scene_sig)
    print(f"\nscenes in final set: {len(scene_names)}")
    print("\nsign-flip permutation tests vs baseline (per-scene AURC differences):")
    rng2 = np.random.default_rng(1)
    base_by_scene = {nm: aurc(per_scene_sig[nm][0][0]["baseline"], per_scene_sig[nm][0][1]) for nm in scene_names}
    for sn in ("E_case", "tile-phase", "E_dist", "control"):
        d = np.array([base_by_scene[nm] - aurc(per_scene_sig[nm][0][0][sn], per_scene_sig[nm][0][1]) for nm in scene_names])
        obs = d.mean()
        perm = np.array([(d * rng2.choice([-1, 1], len(d))).mean() for _ in range(N_PERM)])
        p = float((np.abs(perm) >= abs(obs)).mean())
        wins = int((d > 0).sum())
        print(f"  {sn:<12} mean diff {obs:+.4f}  better on {wins}/{len(d)} scenes  p={p:.4f}")
    n_best_base = sum(1 for nm in scene_names
                      if min(per_scene_sig[nm][0][0], key=lambda s: aurc(per_scene_sig[nm][0][0][s], per_scene_sig[nm][0][1])) == "baseline")
    print(f"\nbaseline best on {n_best_base}/{len(scene_names)} scenes")
    print("wrote exp/out/exp11_stats.csv")


if __name__ == "__main__":
    main()
