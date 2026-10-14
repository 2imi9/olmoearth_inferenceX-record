"""Exp 03: verify four more techniques on the Katima/Kazungula pair.

  a) tri-model E_case: Nano + Tiny + Base, disagreement = std of water probs
  b) E_system tile-phase: Base probs at pixel shifts 0-3, per-patch flip std
  c) E_dist minimal: k-NN cosine distance from eval patches to train patches
  d) v1 vs v1_2: attempt load_model_from_repo_id("allenai/OlmoEarth-v1_2-Base")

All scored against Base-head errors vs WorldCover, AURC vs max-softmax baseline.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_repo_id
from oe_inferencex.data import fetch_s2_window, fetch_worldcover_window, s2_to_sample, embed
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage, pool_to_patches

TRAIN = (24.302, -17.485)
EVAL = (25.263, -17.788)
SIZE, PATCH, PAD = 128, 4, 4  # eval fetched at SIZE+PAD for tile-phase shifts
GRID = SIZE // PATCH
CACHE = "exp/out/exp03_cache.npz"


def get_windows():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return (z["tr_img"], tuple(z["tr_date"]), z["tr_labels"],
                z["ev_img"], tuple(z["ev_date"]), z["ev_labels"])
    print("train AOI:")
    tr_img, tr_date, (crs, transform) = fetch_s2_window(*TRAIN, SIZE)
    wc = fetch_worldcover_window(*TRAIN, crs, transform, SIZE)
    tr_labels = (pool_to_patches(wc == 80, PATCH) > 0.5).astype(np.float32)
    print("eval AOI (padded):")
    ev_img, ev_date, (crs2, t2) = fetch_s2_window(*EVAL, SIZE + PAD)
    wc2 = fetch_worldcover_window(*EVAL, crs2, t2, SIZE + PAD)
    ev_labels = (pool_to_patches(wc2[:SIZE, :SIZE] == 80, PATCH) > 0.5).astype(np.float32)
    np.savez(CACHE, tr_img=tr_img, tr_date=np.array(tr_date), tr_labels=tr_labels,
             ev_img=ev_img, ev_date=np.array(ev_date), ev_labels=ev_labels)
    return tr_img, tr_date, tr_labels, ev_img, ev_date, ev_labels


def head_probs(model, tr_img, tr_date, tr_labels, ev_view, ev_date, return_feats=False):
    torch.manual_seed(0)
    f_tr = embed(model, s2_to_sample(tr_img, *tr_date), PATCH)
    f_ev = embed(model, s2_to_sample(ev_view, *ev_date), PATCH)
    w, b = train_logistic_head(f_tr, tr_labels)
    p = predict_head(f_ev, w, b)
    return (p, f_tr, f_ev) if return_feats else p


def main():
    tr_img, tr_date, tr_labels, ev_img, ev_date, ev_labels = get_windows()
    base_view = ev_img[:, :SIZE, :SIZE]

    # --- heads for Nano / Tiny / Base ---
    probs, feats = {}, {}
    for mid in (ModelID.OLMOEARTH_V1_NANO, ModelID.OLMOEARTH_V1_TINY, ModelID.OLMOEARTH_V1_BASE):
        model = load_model_from_id(mid)
        p, f_tr, f_ev = head_probs(model, tr_img, tr_date, tr_labels, base_view, ev_date, True)
        probs[mid.value] = p
        feats[mid.value] = (f_tr, f_ev)
        acc = ((p > 0.5) == ev_labels.astype(bool)).mean()
        print(f"{mid.value}: eval acc = {acc:.3f}")
        if mid == ModelID.OLMOEARTH_V1_BASE:
            base_model = model

    p_base = probs["OlmoEarth-v1-Base"]
    errors = ((p_base > 0.5) != ev_labels.astype(bool)).astype(np.float64)
    baseline = 1 - np.maximum(p_base, 1 - p_base)
    print(f"Base error rate: {errors.mean():.3f} ({int(errors.sum())} patches)")

    signals = {"baseline max-softmax": baseline}

    # --- (a) tri-model disagreement ---
    stack = np.stack(list(probs.values()))
    signals["E_case tri-model std"] = stack.std(axis=0)
    signals["E_case |Nano-Base|"] = np.abs(probs["OlmoEarth-v1-Nano"] - p_base)

    # --- (b) tile-phase stability, Base only ---
    canvas = np.full((PAD, SIZE + PAD, SIZE + PAD), np.nan)
    for s in range(PAD):
        view = ev_img[:, s:s + SIZE, s:s + SIZE]
        p = head_probs(base_model, tr_img, tr_date, tr_labels, view, ev_date)
        up = np.kron(p, np.ones((PATCH, PATCH)))  # patch grid -> pixels
        canvas[s, s:s + SIZE, s:s + SIZE] = up
    pix_std = np.nanstd(canvas, axis=0)
    phase = np.full((GRID, GRID), np.nan)
    for i in range(GRID):
        for j in range(GRID):
            block = pix_std[i * PATCH:(i + 1) * PATCH, j * PATCH:(j + 1) * PATCH]
            phase[i, j] = np.nanmean(block)
    valid = ~np.isnan(phase)
    signals["E_system tile-phase"] = np.where(valid, phase, 0)
    print(f"tile-phase: mean std={np.nanmean(phase):.4f} max={np.nanmax(phase):.4f}")

    # --- (c) E_dist: k-NN cosine distance to train patches (Base features) ---
    f_tr, f_ev = feats["OlmoEarth-v1-Base"]
    a = torch.nn.functional.normalize(f_ev.reshape(-1, f_ev.shape[-1]), dim=-1)
    b = torch.nn.functional.normalize(f_tr.reshape(-1, f_tr.shape[-1]), dim=-1)
    d = 1 - (a @ b.T)  # cosine distance
    knn = torch.topk(d, k=5, largest=False).values.mean(dim=1)
    signals["E_dist knn-to-train"] = knn.reshape(GRID, GRID).numpy()

    # --- (d) v1_2 attempt ---
    try:
        m12 = load_model_from_repo_id("allenai/OlmoEarth-v1_2-Base")
        p12 = head_probs(m12, tr_img, tr_date, tr_labels, base_view, ev_date)
        acc12 = ((p12 > 0.5) == ev_labels.astype(bool)).mean()
        print(f"v1_2-Base: LOADED, eval acc = {acc12:.3f}")
        signals["E_system |v1_2-v1|"] = np.abs(p12 - p_base)
    except Exception as exc:
        print(f"v1_2-Base: FAILED to load with current checkout: {type(exc).__name__}: {exc}")

    # --- score everything ---
    results = {}
    for name, sig in signals.items():
        cov, risk, aurc = risk_coverage(sig, errors)
        results[name] = (cov, risk, aurc)
        print(f"AURC {name}: {aurc:.5f}")

    from oe_inferencex.figstyle import setup, map_panel, rc_panel
    import matplotlib.pyplot as plt
    setup()
    cbar_labels = {
        "E_case tri-model std": "std of water prob across Nano/Tiny/Base",
        "E_case |Nano-Base|": "|p_Nano - p_Base|",
        "E_system tile-phase": "mean per-pixel std of water prob\nacross 0-3 px grid shifts",
        "E_dist knn-to-train": "mean cosine distance to 5 nearest\ntraining patches (Base embeddings)",
        "E_system |v1_2-v1|": "|p_v1.2 - p_v1|",
    }
    names = [n for n in signals if n != "baseline max-softmax"]
    n = len(names)
    ncol = (n + 2) // 2 + 1
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 9))
    axes = axes.flat
    map_panel(fig, axes[0], errors, "Base head vs WorldCover\n(disagreement, counted as error)",
              "disagreement (binary)", cmap="Reds", idx=0, vmin=0, vmax=1)
    for k, name in enumerate(names):
        map_panel(fig, axes[k + 1], signals[name], name, cbar_labels.get(name, "signal value"),
                  cmap="magma", idx=k + 1)
    rc_panel(axes[n + 1], results, "Kazungula scene (n=1024 patches, 18 errors)", idx=n + 1)
    for a_ in axes[n + 2:]:
        a_.set_visible(False)
    fig.suptitle("Signals on the Kazungula water task; heads trained at Katima Mulilo (110 km away)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("exp/out/exp03_more_channels.png", bbox_inches="tight")
    print("wrote exp/out/exp03_more_channels.png")


if __name__ == "__main__":
    main()
