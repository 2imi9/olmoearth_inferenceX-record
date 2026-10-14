"""Exp 17: evidence from inside the encoder.

Every earlier signal treated the encoder as a black box producing one pooled
vector per patch. This experiment hooks the OlmoEarth v1-Base encoder (12
pre-norm ViT blocks, 768-d, 12 heads, no register tokens; token sequence laid
out H, W, T, S with S = the three Sentinel-2 band-set tokens per patch,
verified empirically) and derives per-patch signals from its internals, with
one forward pass per scene and no second model:

  depth-probe disagreement  - the same logistic water head trained on each
                              block's tokens (train scene); std of the
                              per-layer probabilities over the last six
                              blocks at evaluation. A within-model ensemble
                              decorrelated by depth rather than by size.
  decision settling         - the final head applied to every block's
                              tokens (logit-lens); std of that trajectory
                              over the last six blocks: how late the
                              decision settles.
  representation drift      - mean cosine distance between a patch's tokens
                              at consecutive blocks over the last three
                              blocks (INSIDE-style internal-state signal).
  band-set disagreement     - heads trained separately on the 10 m, 20 m and
                              60 m band-set tokens; std of their three
                              probabilities. An intra-model ensemble whose
                              members see different inputs.
  attention entropy         - mean over heads of the entropy of the last
                              block's attention distribution for the
                              patch's tokens, recomputed from hooked q/k.

All scored against the exp13 errors (seed-0 Base head on pooled final
tokens, 27 rule-selected scenes) with tie-aware E-AURC, alongside the
confidence baseline (negative absolute logit), aligned tile-phase (from the
cached shifted features) and the pixel control. Cross-scene: W/L/T vs
baseline with exact sign tests on untied pairs. Also reports the error
correlation (phi) between the final head and each internal rater, next to
Nano's, to test whether depth or band-set views decorrelate errors better
than a smaller model does (exp07/exp10 question).
"""
import csv
import math
import os
from collections import Counter

import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.evidence import train_logistic_head, predict_head, predict_logit, aurc_expected
from exp13_stat_corrections import aligned_tile_phase, oracle_aurc, ndwi_gradient, sign_test_p, NON_RULE, SHIFTS

SIZE, PATCH = 128, 4
GRID = SIZE // PATCH
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "exp/out/exp17_internals.npz"
LAST_K = 6  # blocks used for depth signals


def make_sample(img, date):
    x = img[:, :SIZE, :SIZE].transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = date
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32, device=DEV),
        sentinel2_l2a_mask=torch.ones((1, SIZE, SIZE, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value,
        timestamps=torch.tensor([d, m0, y], device=DEV)[None, None, :],
    )


def hooked_forward(enc, sample):
    """Return per-block pooled tokens (L, G, G, D) float16, final band-set
    tokens (3, G, G, D), and last-block attention entropy per patch (G, G)."""
    caps, qk = {}, {}
    hs = [blk.register_forward_hook(lambda m, i, o, k=k: caps.__setitem__(k, o.detach())) for k, blk in enumerate(enc.blocks)]
    last = enc.blocks[-1].attn
    hs.append(last.q.register_forward_hook(lambda m, i, o: qk.__setitem__("q", o.detach())))
    hs.append(last.k.register_forward_hook(lambda m, i, o: qk.__setitem__("k", o.detach())))
    with torch.no_grad():
        out = enc(sample, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a[0]  # (G,G,1,3,D)
    for h in hs:
        h.remove()
    L = len(enc.blocks)
    D = out.shape[-1]
    layers = []
    with torch.no_grad():
        for k in range(L):
            t = enc.norm(caps[k])[0].reshape(GRID, GRID, 1, 3, D).mean(dim=(2, 3))  # (G,G,D)
            layers.append(t.half().cpu().numpy())
        bandsets = out[:, :, 0, :, :].permute(2, 0, 1, 3).half().cpu().numpy()  # (3,G,G,D)
        # attention entropy of the last block, recomputed from q/k
        H = last.num_heads
        q = qk["q"][0].reshape(-1, H, D // H).transpose(0, 1)  # (H,N,d)
        k = qk["k"][0].reshape(-1, H, D // H).transpose(0, 1)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * last.scale, dim=-1)  # (H,N,N)
        ent = -(attn * torch.log(attn + 1e-12)).sum(-1).mean(0)  # (N,)
        ent = ent.reshape(GRID, GRID, 1, 3).mean(dim=(2, 3)).float().cpu().numpy()
    return np.stack(layers), bandsets, ent


def phi(a, b):
    a, b = a.astype(bool), b.astype(bool)
    n11 = (a & b).sum(); n10 = (a & ~b).sum(); n01 = (~a & b).sum(); n00 = (~a & ~b).sum()
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / den) if den else float("nan")


def main():
    scenes = dict(np.load("exp/out/exp11_scenes.npz", allow_pickle=True))
    feats11 = dict(np.load("exp/out/exp11_feats.npz", allow_pickle=True))
    z3 = np.load("exp/out/exp03_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    names = sorted({k.rsplit("_", 1)[0] for k in scenes if k.endswith("_img")})
    names = [n for n in names if f"{n}_base0" in feats11 and n not in NON_RULE]

    if os.path.exists(CACHE):
        cache = dict(np.load(CACHE))
        print("loaded cached internals")
    else:
        enc = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval().encoder
        cache = {}
        lay, bs, ent = hooked_forward(enc, make_sample(tr_img, tr_date))
        cache["train_layers"], cache["train_bandsets"] = lay, bs
        for n in names:
            img = scenes[f"{n}_img"]; date = tuple(int(v) for v in scenes[f"{n}_date"])
            lay, bs, ent = hooked_forward(enc, make_sample(img, date))
            cache[f"{n}_layers"], cache[f"{n}_bandsets"], cache[f"{n}_attn_ent"] = lay, bs, ent
            print(f"  captured {n}", flush=True)
        np.savez(CACHE, **cache)

    torch.manual_seed(0)
    L = cache["train_layers"].shape[0]
    f32 = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    # heads: final pooled (same as exp13), per layer, per band set
    head_final = train_logistic_head(f32(feats11["tr_base"]), tr_labels)
    head_nano = train_logistic_head(f32(feats11["tr_nano"]), tr_labels)
    torch.manual_seed(0); heads_layer = [train_logistic_head(f32(cache["train_layers"][k]), tr_labels) for k in range(L)]
    torch.manual_seed(0); heads_bs = [train_logistic_head(f32(cache["train_bandsets"][s]), tr_labels) for s in range(3)]
    a_tr = torch.nn.functional.normalize(f32(feats11["tr_base"]).reshape(-1, 768), dim=-1)

    rows, per, corr = [], {}, {"nano": [], "depth L-4 probe": [], "bandset 20m probe": [], "layer-9 logit-lens": []}
    for n in names:
        lab = scenes[f"{n}_lab"]
        f0 = feats11[f"{n}_base0"]
        p = predict_head(f32(f0), *head_final)
        logit = predict_logit(f32(f0), *head_final)
        err = ((p > 0.5) != lab.astype(bool)).astype(np.float64)
        if err.sum() < 8:
            continue
        layers = cache[f"{n}_layers"].astype(np.float32)          # (L,G,G,D)
        bsets = cache[f"{n}_bandsets"].astype(np.float32)         # (3,G,G,D)
        p_layer = np.stack([predict_head(f32(layers[k]), *heads_layer[k]) for k in range(L)])      # (L,G,G)
        p_lens = np.stack([predict_head(f32(layers[k]), *head_final) for k in range(L)])           # (L,G,G)
        p_bs = np.stack([predict_head(f32(bsets[s]), *heads_bs[s]) for s in range(3)])            # (3,G,G)
        drift = np.mean([1 - np.sum(layers[k] * layers[k + 1], -1) / (np.linalg.norm(layers[k], axis=-1) * np.linalg.norm(layers[k + 1], axis=-1) + 1e-8)
                         for k in range(L - 3, L - 1)], axis=0)
        p_shift = [predict_head(f32(feats11[f"{n}_base{s}"]), *head_final) for s in SHIFTS]
        a_ev = torch.nn.functional.normalize(f32(f0).reshape(-1, 768), dim=-1)
        sigs = {
            "baseline": -np.abs(logit),
            "tile-phase (aligned)": aligned_tile_phase(p_shift),
            "control": ndwi_gradient(scenes[f"{n}_img"]),
            "depth-probe disagreement": p_layer[L - LAST_K:].std(0),
            "decision settling (logit-lens)": p_lens[L - LAST_K:].std(0),
            "representation drift": drift,
            "band-set disagreement": p_bs.std(0),
            "attention entropy": cache[f"{n}_attn_ent"],
            "E_case |Nano-Base|": np.abs(predict_head(f32(feats11[f"{n}_nano"]), *head_nano) - p),
        }
        e = err.flatten()
        orc = oracle_aurc(e.size, int(e.sum()))
        val = {k: aurc_expected(v.flatten(), e) - orc for k, v in sigs.items()}
        per[n] = val
        rows.append({"scene": n, "n_errors": int(e.sum()), **{k: f"{v:.5f}" for k, v in val.items()}})
        # error-correlation of alternative raters with the final head (phi on error indicators)
        e_nano = ((predict_head(f32(feats11[f"{n}_nano"]), *head_nano) > 0.5) != lab.astype(bool)).flatten()
        e_depth = ((p_layer[L - 5] > 0.5) != lab.astype(bool)).flatten()
        e_bs = ((p_bs[1] > 0.5) != lab.astype(bool)).flatten()
        e_lens = ((p_lens[L - 4] > 0.5) != lab.astype(bool)).flatten()
        for key, ee in (("nano", e_nano), ("depth L-4 probe", e_depth), ("bandset 20m probe", e_bs), ("layer-9 logit-lens", e_lens)):
            corr[key].append(phi(e, ee))
        print(f"{n}: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items()))

    with open("exp/out/exp17_internal_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    sn = sorted(per); N = len(sn)
    print(f"\nscenes: {N}. Each signal vs confidence baseline (E-AURC, tie-aware):")
    print(f"{'signal':<32}{'W/L/T':>10}{'sign p':>9}{'median gain':>13}{'vs control W':>13}")
    for k in [k for k in per[sn[0]] if k != "baseline"]:
        d = np.array([per[s]["baseline"] - per[s][k] for s in sn]); w_, l_ = int((d > 1e-12).sum()), int((d < -1e-12).sum())
        dc = np.array([per[s]["control"] - per[s][k] for s in sn]); wc = int((dc > 1e-12).sum())
        print(f"{k:<32}{w_:>3}/{l_}/{N - w_ - l_:<3}{sign_test_p(w_, w_ + l_):>9.3f}{np.median(d):>+13.4f}{wc:>10}/{N}")
    best = Counter(min(per[s], key=per[s].get) for s in sn)
    print("best signal per scene:", dict(best))
    print("\nerror correlation (phi) with the final head, mean over scenes:")
    for k, v in corr.items():
        print(f"  {k:<22} {np.nanmean(v):.3f}")
    print("wrote exp/out/exp17_internal_evidence.csv")


if __name__ == "__main__":
    main()
