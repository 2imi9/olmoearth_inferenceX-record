"""Exp 04: audit signals validated against AWF expert labels (real partner truth).

Uses the official rslearn spatial split shipped in the dataset (1115 train /
344 val). Frozen Nano + Base embeddings of the 12-month S2 stack, multiclass
heads, then on val: max-softmax baseline, Nano-Base disagreement (total
variation), tile-phase instability (Base, shifts 0-3 px). AURC against
expert-label errors of the Base head.
"""
import os

import numpy as np
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.awf import CROP, list_windows, load_window_full, crop_stack, stacks_to_sample
from oe_inferencex.evidence import train_softmax_head, predict_softmax_head, risk_coverage

PATCH = 4
N_CLASSES = 9
SHIFTS = (0, 1, 2, 3)
BATCH = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CACHE = "exp/out/exp04_feats.npz"


PASSES = ["nano_s0", "base_s0", "base_s1", "base_s2", "base_s3"]
CKPT = "exp/out/exp04_ckpt.npz"


def embed_all(windows):
    """One disk pass per batch; all five model/shift passes from memory.
    Checkpoints every 20 batches so a killed run resumes."""
    models = {"nano": load_model_from_id(ModelID.OLMOEARTH_V1_NANO).to(DEV),
              "base": load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV)}
    for m in models.values():
        m.eval()
    feats = {k: [] for k in PASSES}
    start = 0
    if os.path.exists(CKPT):
        z = np.load(CKPT)
        start = int(z["done"])
        for k in PASSES:
            feats[k] = list(z[k])
        print(f"resuming at window {start}")
    for i in range(start, len(windows), BATCH):
        chunk = windows[i:i + BATCH]
        fulls = [(load_window_full(w[0]), w[2], w[3]) for w in chunk]
        for key in PASSES:
            mname, s = key.split("_s")
            stacks, locs = [], []
            for full, r, c in fulls:
                cr, (pr, pc) = crop_stack(full, r, c, int(s))
                stacks.append(cr)
                locs.append((min(pr // PATCH, CROP // PATCH - 1), min(pc // PATCH, CROP // PATCH - 1)))
            sample = stacks_to_sample(stacks, DEV)
            with torch.no_grad():
                out = models[mname].encoder(sample, fast_pass=True, patch_size=PATCH)
            f = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4]).cpu()
            for bi, (pi, pj) in enumerate(locs):
                feats[key].append(f[bi, pi, pj].numpy())
        if (i // BATCH) % 5 == 4:
            np.savez(CKPT, done=i + len(chunk), **{k: np.stack(v) for k, v in feats.items()})
            print(f"  {i + len(chunk)}/{len(windows)} (checkpointed)", flush=True)
    return {k: np.stack(v) for k, v in feats.items()}


def main():
    torch.manual_seed(0)
    windows = list_windows()
    labels = np.array([w[4] for w in windows])
    is_val = np.array([w[1] == "val" for w in windows])
    print(f"windows: {len(windows)}, train {np.sum(~is_val)}, val {np.sum(is_val)}")

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        feats = {k: z[k] for k in z.files}
        print("loaded cached features")
    else:
        feats = embed_all(windows)
        np.savez(CACHE, **feats)

    tr, va = ~is_val, is_val
    y = labels

    # heads trained on shift-0 features
    heads, probs = {}, {}
    for name in ("nano_s0", "base_s0"):
        w, b = train_softmax_head(feats[name][tr], y[tr], N_CLASSES)
        heads[name] = (w, b)
        probs[name] = predict_softmax_head(feats[name][va], w, b)
        acc = (probs[name].argmax(1) == y[va]).mean()
        print(f"{name}: val acc = {acc:.3f}")

    p_base = probs["base_s0"]
    errors = (p_base.argmax(1) != y[va]).astype(np.float64)
    print(f"Base head error rate on expert labels: {errors.mean():.3f} ({int(errors.sum())}/{len(errors)})")

    signals = {"baseline max-softmax": 1 - p_base.max(1)}
    signals["E_case TV(Nano,Base)"] = 0.5 * np.abs(probs["nano_s0"] - p_base).sum(1)

    # tile-phase: same Base head applied to shifted features
    wb, bb = heads["base_s0"]
    shift_probs = [p_base] + [
        predict_softmax_head(feats[f"base_s{s}"][va], wb, bb) for s in SHIFTS[1:]
    ]
    sp = np.stack(shift_probs)  # (S, N, C)
    signals["E_system tile-phase"] = sp.max(2).std(0) + 0.5 * np.abs(sp - sp.mean(0)).sum(2).mean(0)

    results = {}
    for name, sig in signals.items():
        cov, risk, aurc = risk_coverage(sig, errors)
        results[name] = (cov, risk, aurc)
        print(f"AURC {name}: {aurc:.4f}")

    from oe_inferencex.figstyle import setup, rc_panel, letter, AWF_CLASSES
    import matplotlib.pyplot as plt
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rc_panel(axes[0], results,
             "AWF expert labels, official spatial split\n(n=344 val windows, 51 errors)", idx=0)
    per_class = [((p_base.argmax(1) == y[va]) & (y[va] == k)).sum() / max((y[va] == k).sum(), 1)
                 for k in range(N_CLASSES)]
    counts = [int((y[va] == k).sum()) for k in range(N_CLASSES)]
    axes[1].bar(range(N_CLASSES), per_class, color="#4878a8")
    axes[1].set_xticks(range(N_CLASSES))
    axes[1].set_xticklabels([f"{c}\n(n={n})" for c, n in zip(AWF_CLASSES, counts)],
                            rotation=45, ha="right", fontsize=6.5)
    axes[1].set_ylabel("recall of Base-head prediction vs expert label")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis="y", alpha=0.25, lw=0.5)
    axes[1].set_title("Per-class recall\n(class indices mapped via olmoearth_projects awf model.yaml)")
    letter(axes[1], 1)
    fig.tight_layout()
    fig.savefig("exp/out/exp04_awf_expert.png", bbox_inches="tight")
    print("wrote exp/out/exp04_awf_expert.png")


if __name__ == "__main__":
    main()
