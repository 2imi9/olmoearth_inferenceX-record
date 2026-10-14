"""exp36: dihedral consistency, and boundary-first-then-confidence at fixed budgets.

Two label-free candidates the ledger points at, each with its own
preregistered test.

(1) Dihedral consistency. OlmoEarth v1 was pretrained with flip-and-rotate
    augmentation (train_module.transform_config flip_and_rotate in the
    checkpoint config), so its predictions on the eight flips and rotations
    of a window should agree. The window is transformed, encoded, scored by
    the same head, and the probability map is transformed back; the signal
    is the standard deviation of the eight probability maps at each patch.
    It perturbs the tokenisation and not the content (the exp08 rule) and
    covers the half of the perturbation family that aligned tile-phase
    (translations only) does not. Preregistered test: dihedral consistency
    against confidence, one vote per river on the 27 scenes and per-tile on
    Sen1Floods11 Bolivia; the exp13-style null is that it behaves like
    tile-phase (wins against WorldCover, loses on hand labels); its Spearman
    with tile-phase and the boundary indicator is recorded.

(2) Boundary first, then confidence. exp35 found the boundary indicator
    captures more errors than confidence at a 5% review budget on hand
    labels and fewer at 20%. The reviewer's natural rule is lexicographic:
    boundary patches first, ordered by confidence within, then the interior
    by confidence. No parameter. Preregistered test: capture at the 5% and
    10% budgets against confidence on Bolivia (per-tile one-sided exact
    sign tests) and on the fine-tuned AWF model (exp21's table, bootstrap
    over its 30 tasks); 20% and the WorldCover scenes reported alongside.
    The midrank combination U+ of confidence and dihedral consistency is
    scored as the exp28-pattern combination.

Everything else is the exp28 pattern (exp/harness_ab.py): the seed-0 head,
the exp13 error set, the exp18 protocol, confidence, aligned tile-phase,
boundary and the pixel controls on identical patches, tie-aware excess
AURC, capture at budget under random tie-breaking. Inputs: the exp11 and
exp18 feature caches for the unperturbed signals, the Base checkpoint for
the eight extra forward passes per unit (27 scenes at 128 px; 441 Bolivia
tiles at 60 px; the head's training tiles are not transformed), exp21's
per-window table. Outputs: exp/out/exp36_dihedral_lexicographic.csv,
exp36_summary.json, exp36_cache.npz. --smoke: CPU, 2 scenes at 64 px, 4
flood tiles. fp32.
"""
import csv
import os
import traceback

import numpy as np
import torch

import harness_ab as hb
from harness_ab import CONF, TILE, BOUND, CTRL, DEV, OUT, RIVER, SHIFTS  # noqa: F401
from oe_inferencex.evidence import predict_head
from oe_inferencex.metrics import aurc_expected, capture_at_budget_expected
from oe_inferencex.stats import sign_test, wins_losses_ties

DIH, LEX = "dihedral consistency (8 transforms)", "boundary first, then confidence"
NEW_NAMES = [DIH, LEX]
PRIMARY = DIH
COMBO = "combination conf+dihedral (prereg)"
BUDGETS = (0.05, 0.10, 0.20)
LEX_BUDGETS = (0.05, 0.10)
N_BOOT = 2000
TRANSFORMS = [(k, f) for f in (False, True) for k in range(4)]     # (rot90 count, flip left-right)


# ----------------------------------------------------------------------------- dihedral group on (…, H, W) grids
def apply_t(a, k, f, axes):
    """Rotate k quarter turns and optionally flip along the last of `axes`; works for pixel and patch grids."""
    out = np.rot90(a, k, axes=axes)
    return np.flip(out, axis=axes[1]) if f else out


def invert_t(a, k, f, axes):
    out = np.flip(a, axis=axes[1]) if f else a
    return np.rot90(out, -k, axes=axes)


@torch.no_grad()
def dihedral_probs_scene(model, img, date, size, hb_head):
    """(8, G, G) probability maps of the transformed scene, each mapped back to the original grid."""
    maps = []
    crop = np.asarray(img)[:, :size, :size]
    for k, f in TRANSFORMS:
        x, ts = hb.scene_tensor(apply_t(crop, k, f, axes=(1, 2)), date, size)
        feats = hb.embed_pooled(model, x, ts)
        p = predict_head(torch.tensor(feats), *hb_head)
        maps.append(invert_t(p, k, f, axes=(0, 1)))
    return np.stack(maps)


@torch.no_grad()
def dihedral_probs_tiles(model, tiles, hb_head, crop_px):
    """(8, N, G, G) probability maps of the transformed 60-px crops of Sen1Floods11 tiles, mapped back."""
    import exp18_sen1floods_expert as exp18
    maps = []
    for k, f in TRANSFORMS:
        t = apply_t(np.asarray(tiles)[:, :, :crop_px, :crop_px], k, f, axes=(2, 3))
        pooled = []
        for i in range(0, len(t), exp18.BATCH):
            x = hb.normalize_s2(t[i:i + exp18.BATCH].transpose(0, 2, 3, 1))
            b = x.shape[0]
            from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
            sample = MaskedOlmoEarthSample(sentinel2_l2a=x, sentinel2_l2a_mask=torch.ones((b, crop_px, crop_px, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
                                           timestamps=torch.tensor([1, 5, 2020], device=DEV)[None, None, :].repeat(b, 1, 1))
            out = model.encoder(sample, fast_pass=True, patch_size=hb.PATCH)["tokens_and_masks"].sentinel2_l2a
            pooled.append(out.mean(dim=[3, 4]).cpu().numpy())
        p, _ = exp18.head_prob_logit(np.concatenate(pooled), *hb_head)
        maps.append(invert_t(p, k, f, axes=(1, 2)))
    return np.stack(maps)


def lexicographic(conf, boundary):
    """Boundary patches first ordered by confidence (higher = more suspect), then the interior by confidence.

    The midrank is taken over whatever set is passed in: one unit for the per-unit scores, the pooled set for pooled
    estimates (per-unit midranks would not order confidence across units)."""
    c, b = np.asarray(conf, dtype=np.float64), np.asarray(boundary) > 0
    return np.where(b, 2.0, 0.0) + hb.midrank_pct(c).reshape(c.shape)


def capture_rows(sigs, err, mask=None):
    m = np.ones(np.shape(err), bool) if mask is None else mask
    return {k: capture_at_budget_expected(np.asarray(v)[m], err[m], BUDGETS) for k, v in sigs.items()}


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    ctx = hb.load_part_a(model, args, summary)
    per, rho_per, cap = {}, {}, {}
    for name in ctx["names"]:
        try:
            unit = hb.scene_unit(ctx, model, name, args, summary)
            if unit is None:
                continue
            sigs = hb.base_signals_a(unit, ctx)
            maps = dihedral_probs_scene(model, unit["img"], unit["date"], ctx["size"], ctx["hb"])
            cache[f"{name}_dihedral"] = maps.astype(np.float32)
            sigs[DIH] = maps.std(0)
            sigs[LEX] = lexicographic(sigs[CONF], sigs[BOUND])
            sigs[COMBO] = hb.combination(sigs, PRIMARY)
            val = hb.score_scene(unit, sigs, rows, per, rho_per, NEW_NAMES, extra_cols={"identity_max_abs_dp": float(np.abs(maps[0] - unit["p"]).max())})
            cap[name] = capture_rows(sigs, unit["err"])
            print(f"{name}: {unit['n_err']} errors, conf {val[CONF]:.4f} tile {val[TILE]:.4f} dihedral {val[DIH]:.4f} lex {val[LEX]:.4f} U+ {val[COMBO]:.4f}"
                  f" | identity check max|dp| {np.abs(maps[0] - unit['p']).max():.1e}", flush=True)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)
    hb.finish_part_a(summary, per, rho_per, NEW_NAMES, PRIMARY, COMBO)
    sn = sorted(cap)
    if sn:
        summary["part_a"]["capture_tests_vs_confidence"] = {
            k: {str(b): hb.river_test({s: cap[s][k][b] - cap[s][CONF][b] for s in sn}) for b in BUDGETS} for k in (LEX, DIH, BOUND, TILE)}
        summary["part_a"]["median_capture"] = {k: {str(b): float(np.median([cap[s][k][b] for s in sn])) for b in BUDGETS} for k in cap[sn[0]]}
        r = summary["part_a"]["capture_tests_vs_confidence"][LEX]
        print("  lexicographic vs confidence, capture, rivers W/L: " + ", ".join(f"{b}: {r[str(b)]['w']}/{r[str(b)]['l']}" for b in BUDGETS), flush=True)


# ----------------------------------------------------------------------------- part B
def part_b(model, args, summary, rows, cache):
    ctx = hb.load_part_b(model, args, summary, "exp36")
    if ctx is None:
        return
    N, G, ok, err = ctx["N"], ctx["G"], ctx["ok"], ctx["err"]
    sigs = hb.base_signals_b(ctx)
    maps = dihedral_probs_tiles(model, ctx["bo_s2"], ctx["hb"], ctx["CROP"])
    cache["bolivia_dihedral"] = maps.astype(np.float32)
    summary["part_b"]["identity_max_abs_dp"] = float(np.abs(maps[0] - ctx["p"]).max())
    sigs[DIH] = maps.std(0)
    sigs[LEX] = np.stack([lexicographic(sigs[CONF][t], sigs[BOUND][t]) for t in range(N)])
    for k in NEW_NAMES:
        cache[f"bolivia_{NEW_NAMES.index(k)}"] = np.asarray(sigs[k], dtype=np.float32)
    per_b, tiles = hb.finish_part_b(summary, rows, ctx, sigs, NEW_NAMES, PRIMARY, COMBO)
    # capture at budgets: per-tile tests and pooled bootstrap over every tile with valid patches
    cap = {k: [] for k in (LEX, DIH, BOUND, TILE, CONF)}
    all_sig, all_err = {k: [] for k in cap}, []
    for t in range(N):
        m = ok[t]
        if not m.any():
            continue
        all_err.append(err[t][m])
        for k in cap:
            all_sig[k].append(np.asarray(sigs[k][t])[m])
        if t in tiles:
            c = capture_rows({k: sigs[k][t] for k in cap}, err[t], m)
            for k in cap:
                cap[k].append(c[k])
    rng = np.random.default_rng(1)
    tests = {}

    def pooled_score(k, idx):
        """Pooled score of signal k over the tiles in idx; the lexicographic rule is rebuilt from the pooled confidence
        with one global midrank, since per-tile midranks do not order confidence across tiles."""
        if k == LEX:
            c_ = np.concatenate([all_sig[CONF][i] for i in idx])
            b_ = np.concatenate([all_sig[BOUND][i] for i in idx])
            return lexicographic(c_, b_)
        return np.concatenate([all_sig[k][i] for i in idx])

    for k in (LEX, DIH, BOUND, TILE):
        tests[k] = {}
        for b in BUDGETS:
            gains = np.array([cap[k][i][b] - cap[CONF][i][b] for i in range(len(tiles))])
            w, l, t_ = wins_losses_ties(gains)
            one_sided = (k == LEX and b in LEX_BUDGETS)
            boot = []
            for _ in range(N_BOOT):
                pick = rng.integers(0, len(all_err), len(all_err))
                e = np.concatenate([all_err[i] for i in pick])
                if e.sum() == 0:
                    continue
                s_, c_ = pooled_score(k, pick), np.concatenate([all_sig[CONF][i] for i in pick])
                boot.append(capture_at_budget_expected(s_, e, (b,))[b] - capture_at_budget_expected(c_, e, (b,))[b])
            g = np.array(boot)
            everything = np.arange(len(all_err))
            pooled_sig, pooled_err = pooled_score(k, everything), np.concatenate(all_err)
            pooled_conf = np.concatenate(all_sig[CONF])
            tests[k][str(b)] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one_sided else "two-sided"), "one_sided": one_sided,
                                "pooled": capture_at_budget_expected(pooled_sig, pooled_err, (b,))[b],
                                "pooled_confidence": capture_at_budget_expected(pooled_conf, pooled_err, (b,))[b],
                                "boot_lo": float(np.percentile(g, 2.5)), "boot_hi": float(np.percentile(g, 97.5)), "p_better": float((g > 0).mean())}
    summary["part_b"]["capture_tests_vs_confidence"] = tests
    summary["part_b"]["prereg_lexicographic"] = {b: tests[LEX][str(b)] for b in LEX_BUDGETS}
    for k in (LEX, DIH):
        print(f"  {k}: capture vs confidence " + " | ".join(f"{b}: {tests[k][str(b)]['w']}/{tests[k][str(b)]['l']}/{tests[k][str(b)]['t']} p={tests[k][str(b)]['sign_p']:.2g} "
                                                              f"pooled {tests[k][str(b)]['pooled']:.3f} vs {tests[k][str(b)]['pooled_confidence']:.3f}" for b in BUDGETS), flush=True)


# ----------------------------------------------------------------------------- AWF fine-tuned model: lexicographic at budgets
def part_awf(summary):
    path = os.path.join(OUT, "exp21_finetuned_awf.csv")
    table = list(csv.DictReader(open(path)))
    summary["part_awf"] = {"source": path, "crops": {}}
    rng = np.random.default_rng(0)
    for crop in ("16", "32"):
        R = [r for r in table if r["crop"] == crop]
        err = np.array([float(r["error"]) for r in R])
        conf = -np.array([float(r["logit_margin"]) for r in R])
        bnd = np.array([float(r["boundary"]) for r in R])
        lex = lexicographic(conf, bnd)
        clusters = np.array([r["task"] for r in R])
        ids = np.unique(clusters)
        by = {c: np.flatnonzero(clusters == c) for c in ids}
        res = {}
        for b in BUDGETS:
            gains = []
            for _ in range(N_BOOT):
                pick = rng.choice(ids, size=len(ids), replace=True)
                sel = np.concatenate([by[c] for c in pick])
                if err[sel].sum() == 0:
                    continue
                gains.append(capture_at_budget_expected(lex[sel], err[sel], (b,))[b] - capture_at_budget_expected(conf[sel], err[sel], (b,))[b])
            g = np.array(gains)
            res[str(b)] = {"capture_lex": capture_at_budget_expected(lex, err, (b,))[b], "capture_confidence": capture_at_budget_expected(conf, err, (b,))[b],
                           "boot_lo": float(np.percentile(g, 2.5)), "boot_hi": float(np.percentile(g, 97.5)), "p_better": float((g > 0).mean())}
        summary["part_awf"]["crops"][crop] = {"n_windows": len(R), "n_errors": int(err.sum()), "lexicographic_vs_confidence": res,
                                             "aurc": {"confidence": aurc_expected(conf, err), "lexicographic": aurc_expected(lex, err)}}
        print(f"AWF crop {crop}: lexicographic vs confidence capture " + ", ".join(f"{b}: {res[str(b)]['capture_lex']:.3f} vs {res[str(b)]['capture_confidence']:.3f} (P {res[str(b)]['p_better']:.2f})" for b in BUDGETS), flush=True)


def main():
    args = hb.make_parser(__doc__).parse_args()
    config = {"transforms": "8 dihedral transforms (4 rotations x flip) of the input window; probability maps mapped back; std across the 8",
              "lexicographic": "boundary patches (indicator > 0) first ordered by confidence midrank, then interior by confidence midrank",
              "budgets": list(BUDGETS), "prereg": {"dihedral": "vs confidence: one vote per river (A), per-tile sign test (B); U+ = midrank mean of confidence and dihedral",
                                                    "lexicographic": "capture at 5% and 10% vs confidence: per-tile one-sided sign tests on Bolivia, task bootstrap on AWF"},
              "n_boot": N_BOOT, "primary_score": PRIMARY, "combination": COMBO}

    def part_a_then_awf(model, args, summary, rows, cache):
        part_awf(summary)
        part_a(model, args, summary, rows, cache)
    hb.run("exp36", "exp36 dihedral consistency and boundary-first-then-confidence", config, part_a_then_awf, part_b, args, "exp36_dihedral_lexicographic")


if __name__ == "__main__":
    main()
