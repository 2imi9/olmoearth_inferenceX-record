"""exp32: what the latent-MIM target space of OlmoEarth v1-Base actually is (CPU, committed scenes).

Read from the installed code and the checkpoint config: the target encoder is
a copy of the online encoder that is never updated (train module
ContrastiveLatentMIM, ema_decay (1.0, 1.0), reinit_targets False), every
modality exits the target at block 0 (token_exit_cfg), so a target is the
target encoder's patch embedding of the raw patch with no attention; the loss
is patch discrimination (pred and target L2-normalised, softmax at tau 0.1
over the masked tokens of the same sample) plus a 0.1-weighted contrastive
term on pooled tokens. Two questions follow. (1) Is the shipped target
encoder still at its random initialisation? A random init leaves every
LayerNorm weight at exactly 1 and every bias at exactly 0, and its patch
projection drifts away from the trained online one. (2) How degenerate are
the targets on real Sentinel-2 scenes? For each of four committed 128-px
scenes and each of the three S2 band sets, the 1024 target tokens and the
1024 online-encoder tokens are compared: mean cosine between random pairs,
share of the energy on the top singular direction (raw and after centring),
effective rank exp(H) of the centred spectrum, and the coefficient of
variation of the token norm. A random linear projection preserves the raw
patch covariance, which is dominated by brightness, and the targets are not
normalised per patch, so a low effective rank is the expected outcome; it is
the mechanism behind exp27's class aliasing and exp28's uninformative
residual, and the number issue #11 asks for.

Outputs. exp/out/exp32_target_space.csv (one row per scene x band set),
exp32_summary.json (the init signature and the medians). No GPU, no labels,
a minute on CPU.
"""
import csv
import json
import os
import sys

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(EXP_DIR))
sys.path.insert(0, EXP_DIR)

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue  # noqa: E402
import harness_ab as hb  # noqa: E402

OUT = os.path.join(EXP_DIR, "out")
SCENES = ("barotse", "okavango_80", "shire_80", "kazungula")
SIZE, PATCH, N_PAIRS = 128, 4, 4000


def init_signature(m):
    """Random-init fingerprint: LayerNorm weights all exactly 1, biases all exactly 0, init-scale matrices."""
    ln_w, ln_b, mats = [], [], []
    for n, p in m.named_parameters():
        if "norm" in n.lower() and n.endswith("weight"):
            ln_w.append(p.detach().flatten())
        elif "norm" in n.lower() and n.endswith("bias"):
            ln_b.append(p.detach().flatten())
        elif p.dim() >= 2:
            mats.append(p.detach().flatten())
    lw, lb, mt = torch.cat(ln_w), torch.cat(ln_b), torch.cat(mats)
    return {"layernorm_weights_all_exactly_1": bool((lw == 1).all()), "layernorm_weight_std": float(lw.std()),
            "layernorm_biases_all_exactly_0": bool((lb == 0).all()), "matrix_weight_std": float(mt.std()),
            "n_params": int(sum(p.numel() for p in m.parameters()))}


def spectrum(X):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    raw = np.linalg.svd(X, compute_uv=False)
    return {"raw_top_share": float(raw[0] ** 2 / (raw ** 2).sum()), "centred_top_share": float(p[0]),
            "centred_effective_rank": float(np.exp(-(p * np.log(p + 1e-300)).sum()))}


def pair_cos(X, rng):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    i, j = rng.integers(0, len(X), N_PAIRS), rng.integers(0, len(X), N_PAIRS)
    return float((Xn[i] * Xn[j]).sum(1).mean())


def main():
    model = hb.load_model()
    te, en = model.target_encoder, model.encoder
    summary = {"experiment": "exp32 latent-MIM target space", "device": "cpu",
               "config_read": {"ema_decay": [1.0, 1.0], "reinit_targets": False, "token_exit_cfg": "0 for every modality",
                               "loss": "patch_discrimination, tau 0.1, negatives = masked tokens of the same sample; + contrastive 0.1",
                               "masking": "encode 0.5 / decode 0.5; maps decode-only"},
               "target_encoder": init_signature(te), "online_encoder": init_signature(en)}
    pe_t, pe_e = dict(te.patch_embeddings.named_parameters()), dict(en.patch_embeddings.named_parameters())
    summary["s2_patch_projection"] = {}
    for n in [k for k in pe_t if "sentinel2" in k and k.endswith("proj.weight")]:
        a, b = pe_t[n].detach().flatten(), pe_e[n].detach().flatten()
        summary["s2_patch_projection"][n.split(".")[-3]] = {"target_std": float(a.std()), "online_std": float(b.std()),
                                                            "cosine_target_online": float(torch.dot(a, b) / (a.norm() * b.norm()))}
    print("target encoder:", json.dumps(summary["target_encoder"]))
    print("online encoder:", json.dumps(summary["online_encoder"]))
    scenes = dict(np.load(os.path.join(OUT, "exp11_scenes.npz"), allow_pickle=True))
    rng = np.random.default_rng(0)
    rows = []
    for name in SCENES:
        x, ts = hb.scene_tensor(scenes[f"{name}_img"], scenes[f"{name}_date"], SIZE)
        full = MaskedOlmoEarthSample(sentinel2_l2a=x, sentinel2_l2a_mask=torch.ones((1, SIZE, SIZE, 1, 3)) * MaskValue.ONLINE_ENCODER.value,
                                     timestamps=ts)
        with torch.no_grad():
            tgt = te.patch_embeddings(full, PATCH)["sentinel2_l2a"][0, :, :, 0].numpy()                       # (G, G, 3, D)
            enc = en(full, fast_pass=True, patch_size=PATCH)["tokens_and_masks"].sentinel2_l2a[0, :, :, 0].numpy()
        for bs in range(3):
            T, E = tgt[:, :, bs].reshape(-1, tgt.shape[-1]), enc[:, :, bs].reshape(-1, enc.shape[-1])
            st, se = spectrum(T), spectrum(E)
            norms = np.linalg.norm(T, axis=1)
            rows.append({"scene": name, "bandset": bs, "n_tokens": len(T),
                         "target_pair_cos": pair_cos(T, rng), "encoder_pair_cos": pair_cos(E, rng),
                         **{f"target_{k}": v for k, v in st.items()}, **{f"encoder_{k}": v for k, v in se.items()},
                         "target_norm_cv": float(norms.std() / norms.mean())})
        print(f"{name}: target pair cos {np.mean([r['target_pair_cos'] for r in rows[-3:]]):.3f}, "
              f"centred effective rank {np.mean([r['target_centred_effective_rank'] for r in rows[-3:]]):.1f} "
              f"(encoder {np.mean([r['encoder_centred_effective_rank'] for r in rows[-3:]]):.1f})", flush=True)
    keys = list(rows[0])
    with open(os.path.join(OUT, "exp32_target_space.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary["medians"] = {k: float(np.median([r[k] for r in rows])) for k in keys if k not in ("scene", "bandset", "n_tokens")}
    summary["scenes"] = list(SCENES)
    with open(os.path.join(OUT, "exp32_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("medians:", json.dumps({k: round(v, 4) for k, v in summary["medians"].items()}))
    print("wrote exp/out/exp32_target_space.csv and exp32_summary.json")


if __name__ == "__main__":
    main()
