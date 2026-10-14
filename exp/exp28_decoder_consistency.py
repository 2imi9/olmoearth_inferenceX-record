"""exp28: decoder self-consistency as a label-free error signal.

exp08 perturbed pixel content (occlusion with mean fill) and was rejected: it
measured context reliance, not error. This experiment uses the perturbation
the model was pretrained on instead: native token masking. OlmoEarth v1 is a
latent-MIM model; in pretraining the public v1 recipe and the installed code
use modality_cross_random: random token masking over space, time and band
set, NOT grouped across band sets (encode ratio 0.5), which marks tokens as
ONLINE_ENCODER or DECODER; the encoder sees only the encoded tokens, the Predictor decodes the
masked ones from that context, and the loss is a cosine patch-discrimination
loss (tau 0.1) between each decoded token and the frozen target projection of
the true patch (target_encoder.patch_embeddings, token_exit_cfg 0: no
attention). Maps (worldcover, srtm, osm, canopy, cdl, worldcereal) were
decode-only targets and are never fed here; the evaluation label is never an
input.

Signal. For a scene, K random structured masks each hide 25% of the patches;
every hidden patch has all three Sentinel-2 band-set tokens (10 m, 20 m, 60 m)
masked together. This whole-spectrum geometry is a deliberate intervention,
not the pretraining geometry: it removes the trivially easy within-patch
cross-band query at the cost of a distribution shift (the decoder never saw
25% of complete patches hidden at once). A pretraining-consistent variant is
scored from the same code path: token-level complementary-pair masks (each
of `pairs` random permutations of the G*G*3 tokens is split into two halves,
so 50% of tokens are hidden per mask, matching the base sampler's marginal,
and every token is hidden exactly `pairs` times). Whole-spectrum masks
are built as rounds of four disjoint quarters of a random patch permutation,
so with K = 8 (two rounds) every patch is hidden exactly twice and no patch is
left unscored. Per hidden token the error is measured three ways from the same
forward: (1) cosine distance to the target projection of the true patch (the
spec'd primary), (2) the same after subtracting the scene-mean target and the
mask-mean prediction per band set (the raw target space has a large shared
direction: targets of unrelated patches have cosine about 0.99, so (1) is
offset-dominated), and (3) the per-token patch-discrimination NLL over the
decoded tokens of the same mask minus log(n), i.e. the pretraining loss
itself. The per-patch signal is the mean over the masks in which the patch
was hidden and over its three band-set tokens. A K = 32 variant (eight
rounds) is scored from the same cache to show the effect of averaging.

Scores are realized mismatches under a masking intervention, not predictive
uncertainty; errors are always those of the ORIGINAL unmasked probe.
Alongside each decoder score the same forward yields a target-only control:
the crowding NLL of the true target among the mask's other targets with no
decoder involved, so a decoder "advantage" that the crowding control
reproduces is input distinctiveness, not decoding. Also scored: a constant
score (its tie-aware AURC is the error rate), S2 within-patch variance and
NDWI level as observed-input controls, and a preregistered fixed
combination U+ = mean of the within-scene midrank percentiles of confidence
and of the primary decoder score, whose gain over confidence alone is the
primary test of added ranking utility. Statistics: scene-level wins/losses
are descriptive; the primary inference is a one-sided exact sign test over
the eight independent river clusters of the per-river mean gain.

Evaluation. (A) The 27 rule-selected exp11 scenes against the exp13
disagreement set (seed-0 Base head trained on katima, errors against
WorldCover 2021; scenes with fewer than 8 errors skipped, as exp13). (B)
Sen1Floods11 Bolivia hand labels exactly as exp18: same 600 valid-split
training tiles, same 60x60 crops, same head, same patch-label pooling, same
per-tile scoring rule (at least three errors and three correct patches),
tie-aware per-tile E-AURC and exact sign tests. Both parts compare the
decoder signals with the model's own confidence (negative absolute logit),
aligned tile-phase (exp13 / exp18 code), the boundary indicator (exp14) and
the no-model NDWI-gradient control on identical errors, with wins/losses/ties,
exact sign tests, a sign-flip permutation test on mean E-AURC differences
(part A) and per-scene Spearman rank correlations of the decoder signals with
tile-phase and confidence (is it new information?). Nothing is trained on the
labels that are evaluated against.

Inputs. exp/out/exp11_scenes.npz (committed), exp/out/exp03_cache.npz
(committed, training scene), exp/out/exp11_feats.npz (gitignored; regenerated
on the cluster; if missing the pooled features are recomputed here with the
same code path and the summary says so), the Base checkpoint from the HF
cache, and for part B data/floods/flood_valid_data.pt and
flood_bolivia_data.pt (downloaded from the public research_benchmarks bucket
into data/floods/ when missing; ~65 MB and ~364 MB). exp/out/exp18_feats.npz
is used for the Bolivia head and confidence if present; otherwise the
features are recomputed with exp18.embed and cached as
exp28_floods_feats.npz.

Outputs. exp/out/exp28_decoder_consistency.csv (one row per scene x signal
for part A, per split and per tile x signal for part B),
exp28_summary.json (cross-scene and cross-tile tests, diagnostics,
failures), cache exp28_errors.npz (per-mask, per-band-set token errors) and
exp28_floods_feats.npz when computed. --smoke runs on CPU with 2 scenes at
64 px crops and writes the same files with a _smoke suffix; it skips part B
unless the flood files exist (or --floods-dir points at fixtures).

fp32 only: no autocast, no TF32, no compile.
"""
import argparse
import warnings
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import Counter

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402
from olmoearth_pretrain.nn.utils import unpack_encoder_output  # noqa: E402

from oe_inferencex.evidence import predict_head, predict_logit, train_logistic_head  # noqa: E402
from oe_inferencex.metrics import aurc_expected  # noqa: E402
import exp13_stat_corrections as exp13  # noqa: E402
import exp14_boundary_ablation as exp14  # noqa: E402
import exp18_sen1floods_expert as exp18  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

OUT = os.path.join(EXP_DIR, "out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 4
SHIFTS = exp13.SHIFTS
N_BANDSETS = Modality.SENTINEL2_L2A.num_band_sets  # 3
TAU = 0.1                 # pretraining patch-discrimination temperature
QUARTERS = 4              # masks per round: disjoint quarters -> 25% hidden each
PRIMARY_K = 8             # two rounds: every patch hidden exactly twice
DEFAULT_ROUNDS = 8        # 32 masks cached; the K=32 variant is scored from them
MIN_ERRORS_A = 8          # exp13 scene rule
N_PERM = 10000
FLOODS_URL = ("https://storage.googleapis.com/ai2-olmoearth-projects-public-data/"
              "research_benchmarks/floods/flood_{split}_data.pt")
FLOODS_SPLITS = ("valid", "bolivia")
RULE_SCENES = [
    "barotse", "cuando_20", "cuando_50", "cuando_80", "delta", "kafue_20", "kafue_50", "kafue_80",
    "kazungula", "luangwa_conf", "okavango_50", "okavango_80", "okavango_sep", "rovuma_20", "rovuma_50",
    "rovuma_80", "save_20", "save_50", "save_80", "shire_20", "shire_50", "shire_80", "shire_liwonde",
    "vicfalls_up", "zambezi_20", "zambezi_50", "zambezi_80",
]
SMOKE_SCENES = ("barotse", "okavango_80")
SMOKE_SIZE = 64
CONF, TILE, BOUND, CTRL = ("confidence (baseline)", "tile-phase (aligned)", "boundary indicator",
                            "control NDWI gradient")
REFERENCES = (CONF, TILE, BOUND, CTRL)
CONST, CTRL_VAR, CTRL_LVL = ("constant score", "control S2 patch variance", "control NDWI level")
EXTRA_CONTROLS = (CONST, CTRL_VAR, CTRL_LVL)
TOKEN_PAIRS = PRIMARY_K // 2      # token-level complementary pairs -> PRIMARY_K masks, every token hidden TOKEN_PAIRS times
PRIMARY_DEC = f"decoder cos-dist (K={PRIMARY_K})"
COMBO = "combination conf+decoder (prereg)"
RIVER = {"barotse": "Zambezi", "delta": "Zambezi", "kazungula": "Zambezi", "vicfalls_up": "Zambezi",
         "zambezi_20": "Zambezi", "zambezi_50": "Zambezi", "zambezi_80": "Zambezi",
         "cuando_20": "Cuando", "cuando_50": "Cuando", "cuando_80": "Cuando",
         "kafue_20": "Kafue", "kafue_50": "Kafue", "kafue_80": "Kafue", "luangwa_conf": "Luangwa",
         "okavango_50": "Okavango", "okavango_80": "Okavango", "okavango_sep": "Okavango",
         "rovuma_20": "Rovuma", "rovuma_50": "Rovuma", "rovuma_80": "Rovuma",
         "save_20": "Save", "save_50": "Save", "save_80": "Save",
         "shire_20": "Shire", "shire_50": "Shire", "shire_80": "Shire", "shire_liwonde": "Shire"}
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(nanvar|Degrees of freedom|Mean of empty slice|All-NaN).*")


# ----------------------------------------------------------------------------- helpers
def normalize_s2(x_dn):
    """(N, H, W, 12) DN -> normalized float32 tensor (N, H, W, 1, 12) on DEV (exp18/exp17 path)."""
    x = exp18._norm.normalize(Modality.SENTINEL2_L2A, x_dn.astype(np.float64)[:, :, :, None, :])
    return torch.tensor(x, dtype=torch.float32, device=DEV)


def scene_tensor(img, date, size, shift=0):
    """(12, H, W) DN scene -> (x (1, size, size, 1, 12), timestamps (1, 1, 3))."""
    crop = img[:, shift:shift + size, shift:shift + size].transpose(1, 2, 0)[None]
    d, m0, y = (int(v) for v in date)
    return normalize_s2(crop), torch.tensor([d, m0, y], device=DEV)[None, None, :]


@torch.no_grad()
def embed_pooled(model, x, ts):
    """Pooled (G, G, D) features exactly as exp11.embed_gpu (fast pass, mean over T and band sets)."""
    b = x.shape[0]
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=x,
        sentinel2_l2a_mask=torch.ones((b, x.shape[1], x.shape[2], 1, N_BANDSETS), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts)
    out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])[0].cpu().numpy()


def quarter_masks(n_patches, rounds, rng):
    """rounds x QUARTERS boolean masks (n_patches,), each hiding one quarter of a random permutation.

    Within a round the quarters are disjoint and cover every patch, so after
    `rounds` rounds every patch is hidden exactly `rounds` times."""
    masks = []
    for _ in range(rounds):
        perm = rng.permutation(n_patches)
        for chunk in np.array_split(perm, QUARTERS):
            m = np.zeros(n_patches, dtype=bool)
            m[chunk] = True
            masks.append(m)
    return masks


def token_pair_masks(n_tokens, pairs, rng):
    """2*pairs boolean masks (n_tokens,): each pair is a random permutation split into two complementary halves,
    so every mask hides 50% of the tokens (the pretraining marginal) and every token is hidden exactly `pairs` times."""
    masks = []
    for _ in range(pairs):
        perm = rng.permutation(n_tokens)
        for half in np.array_split(perm, 2):
            m = np.zeros(n_tokens, dtype=bool)
            m[half] = True
            masks.append(m)
    return masks


def pixel_mask(patch_masks, size):
    """(B, G, G) patch masks or (B, G, G, 3) token masks (bool) -> (B, size, size, 1, 3) float mask with DECODER where hidden."""
    m = np.where(patch_masks, float(MaskValue.DECODER.value), float(MaskValue.ONLINE_ENCODER.value))
    m = np.repeat(np.repeat(m, PATCH, axis=1), PATCH, axis=2)[:, :size, :size]
    if m.ndim == 3:
        m = np.repeat(m[:, :, :, None, None], N_BANDSETS, axis=4)
    else:
        m = m[:, :, :, None, :]
    return torch.tensor(m, dtype=torch.float32, device=DEV)


@torch.no_grad()
def decoder_errors(model, x, ts, masks, batch, rng):
    """Per-mask, per-band-set token errors of the pretrained decoder.

    x: (1, size, size, 1, 12) normalized S2, ts: (1, 1, 3), masks: list of (G*G,) bool.
    Returns dict with 'cos' (raw cosine distance), 'cos_c' (centred cosine
    distance), 'nll' (patch-discrimination NLL minus log n), each (K, G, G, 3)
    with NaN where the token was not decoded, 'hidden' (K, G, G) bool, and
    scalar diagnostics."""
    size = x.shape[1]
    G = size // PATCH
    K = len(masks)
    full = MaskedOlmoEarthSample(
        sentinel2_l2a=x,
        sentinel2_l2a_mask=torch.ones((1, size, size, 1, N_BANDSETS), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts)
    # frozen target projection of every true patch: (G, G, 3, D)
    tgt = model.target_encoder.patch_embeddings(full, PATCH)["sentinel2_l2a"][0, :, :, 0]
    D = tgt.shape[-1]
    tgt_mean = tgt.reshape(-1, N_BANDSETS, D).mean(0)  # (3, D) scene-mean target per band set
    tgt_n = torch.nn.functional.normalize(tgt, dim=-1)
    pm = np.stack(masks)
    pm = pm.reshape(K, G, G) if pm.shape[1] == G * G else pm.reshape(K, G, G, N_BANDSETS)
    out = {k: np.full((K, G, G, N_BANDSETS), np.nan, dtype=np.float32) for k in ("cos", "cos_c", "nll", "nll_self")}
    cos_shuf, pair_cos = [], []
    counts = np.array([m.sum() for m in masks])
    for c in np.unique(counts):  # equal token counts per batch: the eval-mode encoder has no padding mask
        idx = np.flatnonzero(counts == c)
        for i0 in range(0, len(idx), batch):
            sel = idx[i0:i0 + batch]
            B = len(sel)
            sample = MaskedOlmoEarthSample(
                sentinel2_l2a=x.expand(B, -1, -1, -1, -1),
                sentinel2_l2a_mask=pixel_mask(pm[sel], size),
                timestamps=ts.expand(B, -1, -1))
            enc = model.encoder(sample, patch_size=PATCH, fast_pass=False)
            latent, _pooled, dec_kwargs = unpack_encoder_output(enc)
            decoded = model.decoder(latent, timestamps=sample.timestamps, patch_size=PATCH, **dec_kwargs)
            pred = decoded.sentinel2_l2a[:, :, :, 0]                                  # (B, G, G, 3, D)
            dm = decoded.sentinel2_l2a_mask[:, :, :, 0] == MaskValue.DECODER.value     # (B, G, G, 3)
            cos = torch.nn.functional.cosine_similarity(pred, tgt[None], dim=-1)      # (B, G, G, 3)
            pred_n = torch.nn.functional.normalize(pred, dim=-1)
            for j, k in enumerate(sel):
                hid = dm[j]
                e = torch.where(hid, 1 - cos[j], torch.nan)
                out["cos"][k] = e.cpu().numpy()
                # centred: subtract the mean prediction over this mask's decoded tokens and the scene-mean target
                pred_mean = torch.stack([pred[j][..., s, :][hid[..., s]].mean(0) for s in range(N_BANDSETS)])
                cos_c = torch.nn.functional.cosine_similarity(pred[j] - pred_mean, tgt - tgt_mean, dim=-1)
                out["cos_c"][k] = torch.where(hid, 1 - cos_c, torch.nan).cpu().numpy()
                # pretraining loss per token: softmax over the decoded targets of the same mask
                p_dec, t_dec = pred_n[j][hid], tgt_n[hid]                              # (n, D)
                n = p_dec.shape[0]
                scores = (p_dec @ t_dec.T) / TAU
                nll = torch.nn.functional.cross_entropy(scores, torch.arange(n, device=DEV), reduction="none") - math.log(n)
                full_nll = torch.full((G, G, N_BANDSETS), float("nan"), device=DEV)
                full_nll[hid] = nll
                out["nll"][k] = full_nll.cpu().numpy()
                # target-only crowding control: the true target queried against the same candidates, no decoder
                nll_self = torch.nn.functional.cross_entropy((t_dec @ t_dec.T) / TAU, torch.arange(n, device=DEV), reduction="none") - math.log(n)
                full_self = torch.full((G, G, N_BANDSETS), float("nan"), device=DEV)
                full_self[hid] = nll_self
                out["nll_self"][k] = full_self.cpu().numpy()
                # diagnostics: cosine to a shuffled target, and target-target similarity
                perm = torch.tensor(rng.permutation(G * G), device=DEV)
                t_sh = tgt.reshape(G * G, N_BANDSETS, D)[perm].reshape(G, G, N_BANDSETS, D)
                cos_sh = torch.nn.functional.cosine_similarity(pred[j], t_sh, dim=-1)
                cos_shuf.append(float(cos_sh[hid].mean()))
                pair_cos.append(float((tgt_n.reshape(G * G, N_BANDSETS, D)[perm] * tgt_n.reshape(G * G, N_BANDSETS, D)).sum(-1).mean()))
    out["hidden"] = pm
    out["diag"] = {
        "mean_cos_true": float(1 - np.nanmean(out["cos"])),
        "mean_cos_shuffled_target": float(np.mean(cos_shuf)),
        "target_pair_cos": float(np.mean(pair_cos)),
        "mean_nll_excess": float(np.nanmean(out["nll"])),
    }
    return out


def patch_signal(err, k):
    """(K, G, G, 3) token errors -> (G, G) mean over the first k masks and the band sets."""
    return np.nanmean(err[:k], axis=(0, 3))


def decoder_signals(err, n_masks, err_tok=None):
    k = min(PRIMARY_K, n_masks)
    sigs = {
        f"decoder cos-dist (K={k})": patch_signal(err["cos"], k),
        f"decoder cos-dist centred (K={k})": patch_signal(err["cos_c"], k),
        f"decoder NLL (K={k})": patch_signal(err["nll"], k),
        f"target crowding NLL, no decoder (K={k})": patch_signal(err["nll_self"], k),
    }
    if n_masks > k:
        sigs[f"decoder cos-dist (K={n_masks})"] = patch_signal(err["cos"], n_masks)
        sigs[f"decoder NLL (K={n_masks})"] = patch_signal(err["nll"], n_masks)
    if err_tok is not None:
        kt = err_tok["cos"].shape[0]
        sigs[f"decoder cos-dist token-mask (K={kt})"] = patch_signal(err_tok["cos"], kt)
        sigs[f"decoder NLL token-mask (K={kt})"] = patch_signal(err_tok["nll"], kt)
        sigs[f"target crowding NLL token-mask (K={kt})"] = patch_signal(err_tok["nll_self"], kt)
    return sigs


def s2_patch_variance(img, size):
    """Observed-input control: mean over bands of the within-patch pixel std (raw DN)."""
    G = size // PATCH
    x = np.asarray(img)[:, :size, :size].astype(np.float64)
    return x.reshape(x.shape[0], G, PATCH, G, PATCH).std(axis=(2, 4)).mean(axis=0)


def ndwi_level(img, size):
    """Observed-input control: -|patch-mean NDWI| (near zero = spectrally ambiguous water/land)."""
    bo = Modality.SENTINEL2_L2A.band_order
    G = size // PATCH
    x = np.asarray(img)[:, :size, :size].astype(np.float64)
    nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(x[bo.index("B03")] + x[bo.index("B08")], 1, None)
    return -np.abs(nd.reshape(G, PATCH, G, PATCH).mean(axis=(1, 3)))


def midrank_pct(v):
    """Within-unit midrank percentile in [0, 1], ties averaged."""
    v = np.asarray(v, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    s = v[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / max(len(v) - 1, 1)


def one_sided_sign_p(w, n):
    """P(X >= w), X ~ Binomial(n, 1/2); the preregistered one-sided exact sign test."""
    return float(sum(math.comb(n, i) for i in range(w, n + 1)) / 2 ** n) if n else 1.0


def river_test(per_scene):
    """per_scene: {scene: gain}; average within river cluster, then a one-sided exact sign test (> 0 predicted)."""
    groups = {}
    for s, d in per_scene.items():
        groups.setdefault(RIVER.get(s, s), []).append(float(d))
    means = {g: float(np.mean(v)) for g, v in groups.items()}
    w = sum(m > 1e-12 for m in means.values())
    l = sum(m < -1e-12 for m in means.values())
    return {"per_river": means, "w": w, "l": l, "t": len(means) - w - l, "one_sided_p": one_sided_sign_p(w, w + l),
            "n_rivers": len(means)}


def aligned_tile_phase_general(shift_probs, size, pad=exp13.PAD):
    """exp13.aligned_tile_phase for an arbitrary crop size (equal to exp13's at 128; checked in --smoke)."""
    G = size // PATCH
    canvas = np.full((len(shift_probs), size + pad, size + pad), np.nan)
    for s, p in enumerate(shift_probs):
        canvas[s, s:s + size, s:s + size] = np.kron(p, np.ones((PATCH, PATCH)))
    pix_std = np.nanstd(canvas, axis=0)
    out = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            blk = pix_std[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]
            out[i, j] = np.nanmean(blk) if np.isfinite(blk).any() else 0.0
    return out


def tile_phase_a(shift_probs, size):
    return exp13.aligned_tile_phase(shift_probs) if size == exp13.SIZE else aligned_tile_phase_general(shift_probs, size)


def ndwi_gradient_general(img, size):
    """exp13.ndwi_gradient for an arbitrary crop size (equal to exp13's at 128; checked in --smoke)."""
    bo = Modality.SENTINEL2_L2A.band_order
    G = size // PATCH
    x = img[:, :size, :size].astype(np.float64)
    nd = (x[bo.index("B03")] - x[bo.index("B08")]) / np.clip(x[bo.index("B03")] + x[bo.index("B08")], 1, None)
    gy, gx = np.gradient(nd)
    return np.hypot(gx, gy).reshape(G, PATCH, G, PATCH).mean(axis=(1, 3))


def ndwi_a(img, size):
    return exp13.ndwi_gradient(img) if size == exp13.SIZE else ndwi_gradient_general(img, size)


def boundary_indicator(p):
    """exp14's pred-boundary: fraction of a patch's 8 neighbours whose hard label differs (edge padding)."""
    G0, G1 = p.shape
    hard = (p > 0.5).astype(int)
    pad = np.pad(hard, 1, mode="edge")
    nb = np.zeros_like(p, dtype=float)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                nb += (pad[1 + di:1 + di + G0, 1 + dj:1 + dj + G1] != hard)
    return nb / 8.0


def paired_stats(diffs, rng):
    """diffs: per-unit (reference E-AURC - signal E-AURC); > 0 = signal better."""
    d = np.asarray(diffs, dtype=np.float64)
    n = len(d)
    w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    res = {"n": n, "w": w, "l": l, "t": n - w - l,
           "sign_p": exp13.sign_test_p(w, w + l) if w + l else 1.0,
           "median_gain": float(np.median(d)) if n else float("nan"),
           "mean_gain": float(d.mean()) if n else float("nan")}
    if n and rng is not None:
        perm = np.array([(d * rng.choice([-1, 1], n)).mean() for _ in range(N_PERM)])
        res["perm_p"] = float((np.abs(perm) >= abs(d.mean())).mean())
    return res


def json_ready(o):
    if isinstance(o, dict):
        return {str(k): json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return json_ready(o.tolist())
    return o


def smoke_selfcheck():
    """The size-generic helpers must equal the exp13/exp18 originals on the real grids."""
    rng = np.random.default_rng(123)
    ps = [rng.random((exp13.GRID, exp13.GRID)) for _ in SHIFTS]
    assert np.allclose(aligned_tile_phase_general(ps, exp13.SIZE), exp13.aligned_tile_phase(ps)), "tile-phase mismatch"
    img = rng.integers(0, 6000, (12, exp13.SIZE + exp13.PAD, exp13.SIZE + exp13.PAD)).astype(np.int32)
    assert np.allclose(ndwi_gradient_general(img, exp13.SIZE), exp13.ndwi_gradient(img)), "ndwi mismatch"
    p = rng.random((1, exp18.G, exp18.G))
    assert np.allclose(boundary_indicator(p[0]), exp18.boundary(p)[0]), "boundary mismatch"
    ps60 = np.stack([rng.random((1, exp18.G, exp18.G)) for _ in SHIFTS])
    assert np.allclose(aligned_tile_phase_general([q[0] for q in ps60], exp18.CROP), exp18.aligned_tile_phase(ps60)[0]), "tile-phase (60) mismatch"
    print("smoke self-check: size-generic helpers equal exp13/exp18 originals", flush=True)


# ----------------------------------------------------------------------------- part A
def part_a(model, args, summary, rows, cache):
    size = SMOKE_SIZE if args.smoke else exp13.SIZE
    G = size // PATCH
    rounds = args.rounds
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    z3 = np.load(os.path.join(OUT, "exp03_cache.npz"))
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    feat_path = os.path.join(OUT, "exp11_feats.npz")
    feats, feats_source = {}, None
    if args.smoke:
        names = [n for n in SMOKE_SCENES if f"{n}_img" in scenes][:2]
        feats_source = f"recomputed on {size}-px crops (smoke)"
    elif os.path.exists(feat_path):
        feats = dict(np.load(feat_path, allow_pickle=True))
        names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
        names = [n for n in names if f"{n}_base0" in feats and n not in exp13.NON_RULE]   # exp13's scene set
        feats_source = "exp/out/exp11_feats.npz"
        if set(names) != set(RULE_SCENES):
            summary["part_a"]["scene_set_warning"] = {"missing": sorted(set(RULE_SCENES) - set(names)),
                                                       "extra": sorted(set(names) - set(RULE_SCENES))}
            print(f"WARNING: exp11_feats scene set differs from the 27 rule scenes: {summary['part_a']['scene_set_warning']}")
    else:
        names = [n for n in RULE_SCENES if f"{n}_img" in scenes]
        feats_source = "recomputed (exp11_feats.npz missing): same code path as exp11.embed_gpu, fp32"
        print("WARNING: exp11_feats.npz missing; recomputing pooled features", flush=True)
    summary["part_a"].update({"crop_size": size, "grid": G, "feats_source": feats_source, "scenes_requested": names})
    tr_lab = tr_labels[:G, :G]
    if "tr_base" not in feats:
        x, ts = scene_tensor(tr_img, tr_date, size)
        feats["tr_base"] = embed_pooled(model, x, ts)
    torch.manual_seed(0)   # exp13: seed 0, Base head trained first
    hb = train_logistic_head(torch.tensor(feats["tr_base"]), tr_lab)

    rng = np.random.default_rng(0)
    per, rho_per, diag, strata = {}, {}, {}, {}
    for name in names:
        t0 = time.time()
        try:
            img, date, lab = scenes[f"{name}_img"], scenes[f"{name}_date"], scenes[f"{name}_lab"][:G, :G]
            for s in SHIFTS:
                if f"{name}_base{s}" not in feats:
                    x_s, ts_s = scene_tensor(img, date, size, shift=s)
                    feats[f"{name}_base{s}"] = embed_pooled(model, x_s, ts_s)
            p_shift = [predict_head(torch.tensor(feats[f"{name}_base{s}"]), *hb) for s in SHIFTS]
            p = p_shift[0]
            logit = predict_logit(torch.tensor(feats[f"{name}_base0"]), *hb)
            err = ((p > 0.5) != lab.astype(bool)).astype(np.float64)
            n_err = int(err.sum())
            if n_err < (1 if args.smoke else MIN_ERRORS_A):
                summary["part_a"]["skipped"].append({"scene": name, "reason": f"{n_err} errors"})
                print(f"{name}: {n_err} errors, skipped", flush=True)
                continue
            x, ts = scene_tensor(img, date, size)
            masks = quarter_masks(G * G, rounds, rng)
            e = decoder_errors(model, x, ts, masks, args.batch, rng)
            et = decoder_errors(model, x, ts, token_pair_masks(G * G * N_BANDSETS, TOKEN_PAIRS, rng), args.batch, rng)
            for k in ("cos", "cos_c", "nll", "nll_self", "hidden"):
                cache[f"{name}_{k}"] = e[k]
            for k in ("cos", "nll", "nll_self", "hidden"):
                cache[f"{name}_tok_{k}"] = et[k]
            cache[f"{name}_err"] = err.astype(np.float32)
            diag[name] = {**e["diag"], "token_mask": et["diag"]}
            sigs = {
                CONF: -np.abs(logit),
                TILE: tile_phase_a(p_shift, size),
                BOUND: boundary_indicator(p),
                CTRL: ndwi_a(img, size),
                CONST: np.zeros((G, G)),
                CTRL_VAR: s2_patch_variance(img, size),
                CTRL_LVL: ndwi_level(img, size),
            }
            sigs.update(decoder_signals(e, len(masks), et))
            sigs[COMBO] = ((midrank_pct(sigs[CONF]) + midrank_pct(sigs[PRIMARY_DEC])) / 2).reshape(G, G)
            ef = err.flatten()
            val = {k: exp13.eaurc(v.flatten(), ef) for k, v in sigs.items()}
            per[name] = val
            lab_b, onb, prim = lab.astype(bool), sigs[BOUND] > 0, sigs[PRIMARY_DEC]
            strata[name] = {}
            for wn, wm in (("ref-water", lab_b), ("ref-nonwater", ~lab_b)):
                for bn, bm in (("pred-boundary", onb), ("pred-interior", ~onb)):
                    m = wm & bm
                    me, mc = m & (err > 0), m & (err == 0)
                    strata[name][f"{wn}/{bn}"] = {"n": int(m.sum()), "n_err": int(me.sum()),
                                                  "mean_primary_err": float(prim[me].mean()) if me.any() else None,
                                                  "mean_primary_ok": float(prim[mc].mean()) if mc.any() else None}
            rho_per[name] = {k: {r: exp14.spearman(sigs[k].flatten(), sigs[r].flatten()) for r in REFERENCES}
                             for k in sigs if k.startswith("decoder")}
            for k in sigs:
                row = {"part": "A", "unit": name, "n_patches": int(ef.size), "n_errors": n_err, "signal": k,
                       "aurc": aurc_expected(sigs[k].flatten(), ef), "eaurc": val[k],
                       "mean_value": float(np.mean(sigs[k]))}
                for r in REFERENCES:
                    row[f"gain_vs_{r.split(' (')[0].split(' ')[0]}"] = val[r] - val[k]
                    if k.startswith("decoder"):
                        row[f"spearman_vs_{r.split(' (')[0].split(' ')[0]}"] = rho_per[name][k][r]
                rows.append(row)
            print(f"{name}: {n_err} errors, {time.time() - t0:.1f}s, " + ", ".join(f"{k}={v:.4f}" for k, v in val.items())
                  + f" | cos true {e['diag']['mean_cos_true']:.3f} vs shuffled {e['diag']['mean_cos_shuffled_target']:.3f}", flush=True)
        except Exception as ex:  # noqa: BLE001 - record and continue; the summary must always be written
            summary["failures"].append({"part": "A", "unit": name, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{name}: FAILED {ex!r}", flush=True)

    sn = sorted(per)
    summary["part_a"]["n_scenes"] = len(sn)
    summary["part_a"]["scenes"] = sn
    summary["part_a"]["diagnostics"] = diag
    if not sn:
        return
    sig_names = list(per[sn[0]].keys())
    rng2 = np.random.default_rng(1)
    tests = {}
    for k in sig_names:
        tests[k] = {}
        refs = REFERENCES + (EXTRA_CONTROLS if k.startswith(("decoder", "target crowding", "combination")) else ())
        for r in refs:
            if r == k:
                continue
            d = [per[s][r] - per[s][k] for s in sn]
            tests[k][f"vs {r}"] = paired_stats(d, rng2)
            tests[k][f"vs {r}"]["river"] = river_test({s: per[s][r] - per[s][k] for s in sn})
    summary["part_a"]["tests"] = tests
    summary["part_a"]["strata"] = strata
    summary["part_a"]["prereg"] = {
        "primary_decoder_score": PRIMARY_DEC,
        "combination": "U+ = mean of within-scene midrank percentiles of confidence and of the primary decoder score",
        "combination_gain_over_confidence": river_test({s: per[s][CONF] - per[s][COMBO] for s in sn}),
        "primary_decoder_vs_confidence": river_test({s: per[s][CONF] - per[s][PRIMARY_DEC] for s in sn}),
        "tile_phase_vs_confidence": river_test({s: per[s][CONF] - per[s][TILE] for s in sn}),
        "note": "river-level one-sided exact sign tests are the inference; scene-level W/L/T are descriptive",
    }
    pr = summary["part_a"]["prereg"]
    print(f"  prereg: U+ vs confidence rivers {pr['combination_gain_over_confidence']['w']}/{pr['combination_gain_over_confidence']['l']}/{pr['combination_gain_over_confidence']['t']} p={pr['combination_gain_over_confidence']['one_sided_p']:.3f}"
          f" | decoder vs confidence rivers {pr['primary_decoder_vs_confidence']['w']}/{pr['primary_decoder_vs_confidence']['l']}/{pr['primary_decoder_vs_confidence']['t']} p={pr['primary_decoder_vs_confidence']['one_sided_p']:.3f}", flush=True)
    summary["part_a"]["best_tally"] = dict(Counter(min(per[s], key=per[s].get) for s in sn))
    summary["part_a"]["median_eaurc"] = {k: float(np.median([per[s][k] for s in sn])) for k in sig_names}
    summary["part_a"]["spearman"] = {
        k: {r: {"median": float(np.median([rho_per[s][k][r] for s in sn])),
                "min": float(min(rho_per[s][k][r] for s in sn)), "max": float(max(rho_per[s][k][r] for s in sn))}
            for r in REFERENCES}
        for k in sig_names if k.startswith("decoder")}
    print(f"\npart A: {len(sn)} scenes. E-AURC head-to-head (W/L/T, exact sign p, perm p, median gain):")
    for k in sig_names:
        for r, t in tests[k].items():
            print(f"  {k:<34} {r:<30} {t['w']:>2}/{t['l']:>2}/{t['t']:<2} sign p={t['sign_p']:.3g} perm p={t.get('perm_p', float('nan')):.3g} median {t['median_gain']:+.4f}")
    print("  best per scene:", summary["part_a"]["best_tally"])
    for k, v in summary["part_a"]["spearman"].items():
        print(f"  Spearman {k}: " + ", ".join(f"{r.split(' (')[0]} {vv['median']:+.2f} [{vv['min']:+.2f},{vv['max']:+.2f}]" for r, vv in v.items()))


# ----------------------------------------------------------------------------- part B
def ensure_floods(floods_dir, allow_download):
    paths = {s: os.path.join(floods_dir, f"flood_{s}_data.pt") for s in FLOODS_SPLITS}
    missing = [s for s, p in paths.items() if not os.path.exists(p)]
    if missing and not allow_download:
        return None, missing
    os.makedirs(floods_dir, exist_ok=True)
    import urllib.request
    for s in missing:
        url = FLOODS_URL.format(split=s)
        tmp = paths[s] + ".part"
        print(f"downloading {url} -> {paths[s]}", flush=True)
        last = [-1]

        def hook(blocks, bs, total, last=last):
            pct = int(100 * blocks * bs / max(total, 1))
            if pct // 10 != last[0]:
                last[0] = pct // 10
                print(f"  {min(pct, 100)}%", flush=True)
        urllib.request.urlretrieve(url, tmp, reporthook=hook)
        os.replace(tmp, paths[s])
    return paths, []


def load_floods_split(path, n=None, seed=0):
    """exp18.load_split with an absolute path (same band reorder, same subsampling)."""
    d = torch.load(path, weights_only=True)
    s2 = d["s2"].numpy().astype(np.float32)[:, exp18.BAND_IDX]
    lab = d["labels"].numpy()[:, 0]
    if n is not None and n < len(s2):
        idx = np.random.default_rng(seed).choice(len(s2), n, replace=False)
        s2, lab = s2[idx], lab[idx]
    return s2, lab


def part_b(model, args, summary, rows, cache):
    floods_dir = args.floods_dir or os.path.join(ROOT, "data", "floods")
    paths, missing = ensure_floods(floods_dir, allow_download=not args.smoke)
    if paths is None:
        summary["part_b"]["skipped"] = f"flood files missing in smoke mode: {missing}"
        print(f"part B skipped: {summary['part_b']['skipped']}", flush=True)
        return
    n_train = args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES
    tr_s2, tr_lab = load_floods_split(paths["valid"], n_train)
    bo_s2, bo_lab = load_floods_split(paths["bolivia"])
    if args.smoke:
        bo_s2, bo_lab = bo_s2[:args.smoke_tiles], bo_lab[:args.smoke_tiles]
    CROP, G = exp18.CROP, exp18.G
    summary["part_b"].update({"floods_dir": floods_dir, "train_tiles": int(len(tr_s2)), "bolivia_tiles": int(len(bo_s2)), "crop": CROP})
    # features: exp18's cache when present (real runs), else recomputed with exp18.embed and cached here
    keys = ["tr_base"] + [f"bolivia_base{s}" for s in SHIFTS]
    z = {}
    exp18_cache = os.path.join(OUT, "exp18_feats.npz")
    own_cache = os.path.join(OUT, f"exp28_floods_feats{args.suffix}.npz")
    if not args.smoke and os.path.exists(exp18_cache):
        zz = np.load(exp18_cache)
        z = {k: zz[k] for k in keys}
        summary["part_b"]["feats_source"] = exp18_cache
    elif os.path.exists(own_cache):
        zz = np.load(own_cache)
        z = {k: zz[k] for k in keys}
        summary["part_b"]["feats_source"] = own_cache
    else:
        t0 = time.time()
        z["tr_base"], _ = exp18.embed(model, tr_s2, 0)
        for s in SHIFTS:
            z[f"bolivia_base{s}"], _ = exp18.embed(model, bo_s2, s)
        np.savez(own_cache, **z)
        summary["part_b"]["feats_source"] = f"recomputed with exp18.embed in {time.time() - t0:.0f}s, cached {own_cache}"
    print(f"part B features: {summary['part_b']['feats_source']}", flush=True)

    tr_y, tr_ok = exp18.patch_labels(tr_lab[:, :CROP, :CROP])
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    sel = tr_ok.flatten()
    D = z["tr_base"].shape[-1]
    torch.manual_seed(0)
    hb = train_logistic_head(f32(z["tr_base"]).reshape(-1, D)[sel], tr_y.flatten()[sel])

    y, ok = exp18.patch_labels(bo_lab[:, :CROP, :CROP])
    p_shift = np.stack([exp18.head_prob_logit(z[f"bolivia_base{s}"], *hb)[0] for s in SHIFTS])
    p, logit = exp18.head_prob_logit(z["bolivia_base0"], *hb)
    err = ((p > 0.5) != (y > 0.5)).astype(np.float64)
    acc = 1 - err[ok].mean()
    summary["part_b"]["head_acc"] = float(acc)
    print(f"bolivia: tiles {len(y)}, valid patches {int(ok.sum())}, Base head accuracy {acc:.3f}", flush=True)

    # decoder errors per tile on the shift-0 crop (exp18 normalization and timestamps)
    N = len(bo_s2)
    rng = np.random.default_rng(0)
    n_masks = args.rounds * QUARTERS
    err_c = {k: np.full((N, n_masks, G, G, N_BANDSETS), np.nan, dtype=np.float32) for k in ("cos", "cos_c", "nll", "nll_self")}
    err_t = {k: np.full((N, PRIMARY_K, G, G, N_BANDSETS), np.nan, dtype=np.float32) for k in ("cos", "nll", "nll_self")}
    hidden = np.zeros((N, n_masks, G, G), dtype=bool)
    diag = []
    ts = torch.tensor([1, 5, 2020], device=DEV)[None, None, :]
    t0 = time.time()
    failed = []
    for i in range(N):
        try:
            x = normalize_s2(bo_s2[i:i + 1, :, :CROP, :CROP].transpose(0, 2, 3, 1))
            masks = quarter_masks(G * G, args.rounds, rng)
            e = decoder_errors(model, x, ts, masks, args.batch, rng)
            for k in err_c:
                err_c[k][i] = e[k]
            et = decoder_errors(model, x, ts, token_pair_masks(G * G * N_BANDSETS, TOKEN_PAIRS, rng), args.batch, rng)
            for k in err_t:
                err_t[k][i] = et[k]
            hidden[i] = e["hidden"]
            diag.append(e["diag"])
        except Exception as ex:  # noqa: BLE001
            failed.append(i)
            summary["failures"].append({"part": "B", "unit": f"bolivia/tile{i}", "error": repr(ex), "traceback": traceback.format_exc()})
        if i % 50 == 0 or i == N - 1:
            print(f"  decoder errors: tile {i + 1}/{N}, {time.time() - t0:.0f}s", flush=True)
    for k in err_c:
        cache[f"bolivia_{k}"] = err_c[k]
    for k in err_t:
        cache[f"bolivia_tok_{k}"] = err_t[k]
    cache["bolivia_hidden"] = hidden
    cache["bolivia_err"] = err.astype(np.float32)
    cache["bolivia_ok"] = ok
    summary["part_b"]["diagnostics"] = {k: float(np.mean([d[k] for d in diag])) for k in diag[0]} if diag else {}
    summary["part_b"]["tiles_failed"] = failed

    sigs = {
        CONF: -np.abs(logit),
        TILE: exp18.aligned_tile_phase(p_shift),
        BOUND: exp18.boundary(p),
        CTRL: exp18.ndwi_gradient(bo_s2),
        CONST: np.zeros((N, G, G)),
        CTRL_VAR: np.stack([s2_patch_variance(tile, CROP) for tile in bo_s2]),
        CTRL_LVL: np.stack([ndwi_level(tile, CROP) for tile in bo_s2]),
    }
    k_primary = min(PRIMARY_K, n_masks)
    dec = {f"decoder cos-dist (K={k_primary})": np.nanmean(err_c["cos"][:, :k_primary], axis=(1, 4)),
           f"decoder cos-dist centred (K={k_primary})": np.nanmean(err_c["cos_c"][:, :k_primary], axis=(1, 4)),
           f"decoder NLL (K={k_primary})": np.nanmean(err_c["nll"][:, :k_primary], axis=(1, 4)),
           f"target crowding NLL, no decoder (K={k_primary})": np.nanmean(err_c["nll_self"][:, :k_primary], axis=(1, 4)),
           f"decoder cos-dist token-mask (K={PRIMARY_K})": np.nanmean(err_t["cos"], axis=(1, 4)),
           f"decoder NLL token-mask (K={PRIMARY_K})": np.nanmean(err_t["nll"], axis=(1, 4)),
           f"target crowding NLL token-mask (K={PRIMARY_K})": np.nanmean(err_t["nll_self"], axis=(1, 4))}
    if n_masks > k_primary:
        dec[f"decoder cos-dist (K={n_masks})"] = np.nanmean(err_c["cos"], axis=(1, 4))
        dec[f"decoder NLL (K={n_masks})"] = np.nanmean(err_c["nll"], axis=(1, 4))
    sigs.update(dec)
    valid_tiles = np.array([i not in failed for i in range(N)])
    ok = ok & valid_tiles[:, None, None] & np.all(np.isfinite(np.stack([v for v in dec.values()])), axis=0)
    combo = np.zeros((N, G, G))
    for tt in range(N):
        m = ok[tt]
        if m.any():
            combo[tt][m] = (midrank_pct(sigs[CONF][tt][m]) + midrank_pct(sigs[PRIMARY_DEC][tt][m])) / 2
    sigs[COMBO] = combo
    pooled = {k: exp18.eaurc(v[ok], err[ok]) for k, v in sigs.items()}
    per = {k: [] for k in sigs}
    rho = {k: {r: [] for r in REFERENCES} for k in dec}
    tiles_scored = []
    for t in range(N):
        m = ok[t]
        e = err[t][m]
        if e.sum() < 3 or e.sum() > len(e) - 3:   # exp18 rule
            continue
        tiles_scored.append(t)
        for k, v in sigs.items():
            per[k].append(exp18.eaurc(v[t][m], e))
        for k in dec:
            for r in REFERENCES:
                rho[k][r].append(exp14.spearman(sigs[k][t][m], sigs[r][t][m]))
    for k in per:
        per[k] = np.array(per[k])
    n_tiles = len(tiles_scored)
    summary["part_b"].update({"valid_patches": int(ok.sum()), "n_tiles_scored": n_tiles, "pooled_eaurc": pooled})
    tests = {}
    for k in sigs:
        tests[k] = {}
        refs = REFERENCES + (EXTRA_CONTROLS if k.startswith(("decoder", "target crowding", "combination")) else ())
        for r in refs:
            if r == k:
                continue
            tests[k][f"vs {r}"] = paired_stats(per[r] - per[k], None)
    summary["part_b"]["tests"] = tests
    summary["part_b"]["prereg"] = {"combination_gain_over_confidence_pooled": float(pooled[CONF] - pooled[COMBO]),
                                   "combination_tiles_better": int(((per[CONF] - per[COMBO]) > 1e-12).sum()),
                                   "combination_tiles_worse": int(((per[CONF] - per[COMBO]) < -1e-12).sum()),
                                   "note": "Bolivia is one flood event; per-tile signs are descriptive, not an exact test"}
    summary["part_b"]["best_tally"] = dict(Counter(min(sigs, key=lambda k: per[k][i]) for i in range(n_tiles))) if n_tiles else {}
    summary["part_b"]["spearman"] = {k: {r: {"median": float(np.median(v)) if v else None, "min": float(min(v)) if v else None,
                                            "max": float(max(v)) if v else None} for r, v in rr.items()} for k, rr in rho.items()}
    on_b = sigs[BOUND] > 0
    summary["part_b"]["boundary_share"] = {"errors": float(on_b[ok & (err > 0)].mean()) if (ok & (err > 0)).any() else None,
                                           "correct": float(on_b[ok & (err == 0)].mean()) if (ok & (err == 0)).any() else None}
    print(f"  pooled E-AURC / per-tile (n={n_tiles}) W/L vs references:")
    for k in sigs:
        line = f"    {k:<34} pooled {pooled[k]:.4f}"
        for r, t in tests[k].items():
            line += f" | {r.split(' (')[0][:12]:<12} {t['w']:>3}/{t['l']:<3} p={t['sign_p']:.1e}"
        print(line)
    print("    best per tile:", summary["part_b"]["best_tally"])
    for k, rr in summary["part_b"]["spearman"].items():
        print(f"    Spearman {k}: " + ", ".join(f"{r.split(' (')[0]} {v['median']:+.2f}" for r, v in rr.items() if v["median"] is not None))
    rows.append({"part": "B", "unit": "bolivia (pooled)", "n_patches": int(ok.sum()), "n_errors": int(err[ok].sum()), "signal": "",
                 "aurc": "", "eaurc": "", "mean_value": "", "head_acc": float(acc), "n_tiles_scored": n_tiles})
    for k in sigs:
        row = {"part": "B", "unit": "bolivia (pooled)", "n_patches": int(ok.sum()), "n_errors": int(err[ok].sum()), "signal": k,
               "aurc": aurc_expected(sigs[k][ok], err[ok]), "eaurc": pooled[k], "mean_value": float(np.mean(sigs[k][ok])),
               "n_tiles_scored": n_tiles}
        for r in REFERENCES:
            tag = r.split(' (')[0].split(' ')[0]
            row[f"gain_vs_{tag}"] = pooled[r] - pooled[k]
            if k in dec:
                row[f"W/L_vs_{tag}"] = f"{tests[k][f'vs {r}']['w']}/{tests[k][f'vs {r}']['l']}"
                row[f"sign_p_vs_{tag}"] = tests[k][f"vs {r}"]["sign_p"]
                row[f"spearman_vs_{tag}"] = summary["part_b"]["spearman"][k][r]["median"]
        rows.append(row)
    for i, t in enumerate(tiles_scored):
        m = ok[t]
        for k in sigs:
            row = {"part": "B", "unit": f"bolivia/tile{t}", "n_patches": int(m.sum()), "n_errors": int(err[t][m].sum()), "signal": k,
                   "aurc": aurc_expected(sigs[k][t][m], err[t][m]), "eaurc": float(per[k][i]), "mean_value": float(np.mean(sigs[k][t][m]))}
            for r in REFERENCES:
                tag = r.split(' (')[0].split(' ')[0]
                row[f"gain_vs_{tag}"] = float(per[r][i] - per[k][i])
                if k in dec:
                    row[f"spearman_vs_{tag}"] = rho[k][r][i]
            rows.append(row)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="CPU smoke: 2 scenes, 64-px crops, _smoke outputs")
    ap.add_argument("--rounds", type=int, default=None, help=f"rounds of {QUARTERS} disjoint quarter masks (default {DEFAULT_ROUNDS}; smoke 2)")
    ap.add_argument("--batch", type=int, default=None, help="masks per forward (default 8; smoke 4)")
    ap.add_argument("--floods-dir", default=None, help="directory holding flood_{valid,bolivia}_data.pt (default data/floods)")
    ap.add_argument("--smoke-tiles", type=int, default=4, help="tiles per split in smoke part B")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    args = ap.parse_args()
    args.rounds = args.rounds or (2 if args.smoke else DEFAULT_ROUNDS)
    args.batch = args.batch or (4 if args.smoke else 8)
    args.suffix = "_smoke" if args.smoke else ""
    os.makedirs(OUT, exist_ok=True)
    t_start = time.time()
    summary = {"experiment": "exp28 decoder self-consistency", "device": DEV, "smoke": args.smoke,
               "config": {"patch": PATCH, "mask_fraction": 1.0 / QUARTERS, "rounds": args.rounds, "n_masks": args.rounds * QUARTERS,
                          "primary_k": PRIMARY_K, "tau": TAU, "batch": args.batch, "tf32": False, "autocast": False, "compile": False,
                          "masking": "primary: whole patches, all three S2 band-set tokens together (a deliberate intervention; pretraining used modality_cross_random token masking, not grouped across band sets); variant: token-level complementary pairs, 50% hidden, every token hidden TOKEN_PAIRS times",
                          "token_pairs": TOKEN_PAIRS, "primary_decoder_score": PRIMARY_DEC, "combination": COMBO,
                          "target": "target_encoder.patch_embeddings (token_exit_cfg 0), cosine distance / centred / patch-disc NLL"},
               "part_a": {"skipped": []}, "part_b": {}, "failures": []}
    rows, cache = [], {}
    csv_path = os.path.join(OUT, f"exp28_decoder_consistency{args.suffix}.csv")
    sum_path = os.path.join(OUT, f"exp28_summary{args.suffix}.json")
    cache_path = os.path.join(OUT, f"exp28_errors{args.suffix}.npz")
    try:
        if args.smoke:
            smoke_selfcheck()
        t0 = time.time()
        model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval().float()
        print(f"loaded Base on {DEV} in {time.time() - t0:.1f}s; rounds {args.rounds} -> {args.rounds * QUARTERS} masks, primary K={PRIMARY_K}", flush=True)
        if not args.skip_a:
            try:
                part_a(model, args, summary, rows, cache)
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": "A", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"part A FAILED: {ex!r}", flush=True)
        if not args.skip_b:
            try:
                part_b(model, args, summary, rows, cache)
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": "B", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"part B FAILED: {ex!r}", flush=True)
    except Exception as ex:  # noqa: BLE001
        summary["failures"].append({"part": "setup", "unit": "*", "error": repr(ex), "traceback": traceback.format_exc()})
        print(f"setup FAILED: {ex!r}", flush=True)
    finally:
        if rows:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys:
                        keys.append(k)
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in keys})
            print(f"wrote {csv_path} ({len(rows)} rows)")
        if cache:
            np.savez(cache_path, **cache)
            print(f"wrote {cache_path}")
        summary["runtime_s"] = time.time() - t_start
        summary["n_failures"] = len(summary["failures"])
        with open(sum_path, "w") as f:
            json.dump(json_ready(summary), f, indent=1)
        print(f"wrote {sum_path}; failures: {len(summary['failures'])}; runtime {summary['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
