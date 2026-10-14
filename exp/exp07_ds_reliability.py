"""Exp 07: label-free rater-reliability estimation via Dawid-Skene.

Tests whether per-model reliability can be estimated from agreement structure
alone. Three raters (Nano, Tiny, Base heads) vote on the AWF validation
windows; Dawid-Skene EM estimates each rater's expected accuracy without
touching the labels; the estimates are then compared against the accuracies
measured with the expert labels. Agreement between the two would support
label-free accuracy estimation for regions where no labels exist.

Also evaluates DS-posterior-based signals against the naive aggregations
from exp03/exp04 (the weak-rater problem), on the same AURC harness.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.awf import list_windows, load_window_full, crop_stack, stacks_to_sample
from oe_inferencex.evidence import (
    train_softmax_head, predict_softmax_head, risk_coverage, dawid_skene,
)

PATCH = 4
N_CLASSES = 9
BATCH = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "exp/out/exp04_feats.npz"
TINY_CACHE = "exp/out/exp07_tiny_feats.npy"


def embed_tiny(windows):
    model = load_model_from_id(ModelID.OLMOEARTH_V1_TINY).to(DEV)
    model.eval()
    feats = []
    for i in range(0, len(windows), BATCH):
        chunk = windows[i:i + BATCH]
        stacks, locs = [], []
        for wdir, _, r, c, _ in chunk:
            full = load_window_full(wdir)
            cr, (pr, pc) = crop_stack(full, r, c, 0)
            stacks.append(cr)
            locs.append((min(pr // PATCH, 7), min(pc // PATCH, 7)))
        sample = stacks_to_sample(stacks, DEV)
        with torch.no_grad():
            out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
        f = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4]).cpu()
        for bi, (pi, pj) in enumerate(locs):
            feats.append(f[bi, pi, pj].numpy())
        if (i // BATCH) % 40 == 39:
            print(f"  {i + len(chunk)}/{len(windows)}", flush=True)
    return np.stack(feats)


def main():
    torch.manual_seed(0)
    windows = list_windows()
    labels = np.array([w[4] for w in windows])
    is_val = np.array([w[1] == "val" for w in windows])

    z = np.load(CACHE)
    feats = {"nano": z["nano_s0"], "base": z["base_s0"]}
    if os.path.exists(TINY_CACHE):
        feats["tiny"] = np.load(TINY_CACHE)
        print("loaded cached tiny features")
    else:
        print("embedding Tiny over AWF windows")
        feats["tiny"] = embed_tiny(windows)
        np.save(TINY_CACHE, feats["tiny"])

    tr, va = ~is_val, is_val
    y = labels

    probs, acc_true = {}, {}
    for name in ("nano", "tiny", "base"):
        w, b = train_softmax_head(feats[name][tr], y[tr], N_CLASSES)
        probs[name] = predict_softmax_head(feats[name][va], w, b)
        acc_true[name] = float((probs[name].argmax(1) == y[va]).mean())
        print(f"{name}: measured val accuracy = {acc_true[name]:.3f}")

    votes = np.stack([probs[n].argmax(1) for n in ("nano", "tiny", "base")], axis=1)
    post, conf, reliab = dawid_skene(votes, N_CLASSES)
    print("\nDawid-Skene estimates (no labels used):")
    for i, name in enumerate(("nano", "tiny", "base")):
        print(f"  {name}: estimated reliability = {reliab[i]:.3f}  "
              f"(measured {acc_true[name]:.3f}, gap {reliab[i] - acc_true[name]:+.3f})")

    # DS posterior as aggregated prediction; compare aggregations on AURC
    p_base = probs["base"]
    errors = (p_base.argmax(1) != y[va]).astype(np.float64)
    mean_p = np.mean([probs[n] for n in probs], axis=0)
    signals = {
        "baseline max-softmax (Base)": 1 - p_base.max(1),
        "equal-weight mean prob entropy-ish": 1 - mean_p.max(1),
        "DS posterior uncertainty": 1 - post.max(1),
        "DS-Base disagreement": 1 - post[np.arange(len(post)), p_base.argmax(1)],
    }
    print(f"\nBase errors on val: {int(errors.sum())}/{len(errors)}")
    for sn, sig in signals.items():
        _, _, aurc = risk_coverage(sig, errors)
        print(f"AURC {sn}: {aurc:.4f}")

    ds_pred_acc = (post.argmax(1) == y[va]).mean()
    maj_acc = np.mean([
        np.bincount(votes[i], minlength=N_CLASSES).argmax() == y[va][i]
        for i in range(len(votes))
    ])
    print(f"\naggregated prediction accuracy: DS posterior {ds_pred_acc:.3f}, "
          f"majority vote {maj_acc:.3f}, Base alone {acc_true['base']:.3f}")


if __name__ == "__main__":
    main()
