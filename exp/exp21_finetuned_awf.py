"""exp21: the fine-tuned production-style AWF model, run end to end.

Ai2 publishes the fine-tuned checkpoints (allenai/OlmoEarth-v1-FT-AWF-Base:
a fully fine-tuned v1-Base encoder plus a 1x1 convolution head, trained by
rslearn from olmoearth_projects/olmoearth_run_data/awf/model.yaml). This
experiment replicates that model without rslearn: the encoder weights are
loaded into olmoearth_pretrain's v1-Base encoder, tokens are mean-pooled over
timesteps and band sets as rslearn's OlmoEarth wrapper does, and the 1x1
convolution is applied to the pooled patch features (bilinear upsampling
commutes with a 1x1 convolution, so pixel logits equal upsampled patch
logits). Legacy timestamps (month index only) are what the wrapper used.

Evaluated on the AWF validation split (344 expert-labelled points, official
spatial split; the train split is the model's own training data). Two crop
sizes around the label pixel: 16 px (the training regime, Pad 31 + Crop 16)
and 32 px (the regime of the probe experiments). Signals at the label patch:
the model's own confidence (logit margin), the boundary indicator of its own
prediction map, aligned tiling instability over 0-3 px shifts, disagreement
with the frozen-encoder probe of exp16, and a no-model control (temporal
standard deviation of NDVI at the label pixel). Also reported: accuracy,
per-class recall, calibration (expected calibration error, reliability
curve), selective accuracy at fixed coverage, and error capture at review
budgets. Errors are windows where the argmax at the label pixel differs from
the expert label. Paired comparisons use a cluster bootstrap over annotation
tasks (windows cluster by task).

Outputs: exp/out/exp21_finetuned_awf.csv, exp21_summary.json, exp21_finetuned_awf.png.
Cache: exp/out/exp21_stacks.npz (ignored by git).
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download  # noqa: E402
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id  # noqa: E402

from oe_inferencex import awf  # noqa: E402
from oe_inferencex.figstyle import AWF_CLASSES  # noqa: E402
from oe_inferencex.metrics import aurc_expected  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PATCH = 4
SHIFTS = (0, 1, 2, 3)
CROPS = (16, 32)
N_CLASSES = 10
BUDGETS = (0.05, 0.10, 0.20)
REPO = "allenai/OlmoEarth-v1-FT-AWF-Base"


def load_finetuned():
    sd = torch.load(hf_hub_download(REPO, "model.ckpt"), map_location="cpu", weights_only=False)["state_dict"]
    pre = "model.encoder.0.model."
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
    model.encoder.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
    w = sd["model.decoders.segment.1.layer.weight"][:, :, 0, 0].float()   # (10, 768)
    b = sd["model.decoders.segment.1.layer.bias"].float()
    return model.to(DEV).eval(), w.to(DEV), b.to(DEV)


def crop_at(full, r, c, size, shift):
    r0 = min(max(r - size // 2 + shift, 0), 63 - size)
    c0 = min(max(c - size // 2 + shift, 0), 63 - size)
    return full[r0:r0 + size, c0:c0 + size], (r - r0, c - c0)


def load_stacks(windows):
    cache = os.path.join(OUT, "exp21_stacks.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["stacks"]
    t0 = time.time()
    stacks = np.zeros((len(windows), 63, 63, awf.N_MONTHS, 12), dtype=np.float32)
    for i, (wdir, *_rest) in enumerate(windows):
        stacks[i] = awf.load_window_full(wdir)
        if i % 50 == 0:
            print(f"  loaded {i}/{len(windows)} windows, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, stacks=stacks)
    return stacks


@torch.no_grad()
def logits_grid(model, w, b, stacks):
    """Batch of raw crops (B, S, S, T, 12) -> patch logits (B, S/4, S/4, 10) and
    pixel logits (B, S, S, 10), the latter by bilinear x4 upsampling exactly as
    rslearn's Upsample(scale_factor=4) does before the 1x1 convolution."""
    sample = awf.stacks_to_sample(stacks, device=DEV)
    out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    feat = out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])          # (B, H', W', 768)
    patch_logits = (feat @ w.T + b).float()                                 # (B, H', W', 10)
    pix = torch.nn.functional.interpolate(patch_logits.permute(0, 3, 1, 2), scale_factor=PATCH, mode="bilinear", align_corners=False)
    return patch_logits.cpu().numpy(), pix.permute(0, 2, 3, 1).cpu().numpy()


def boundary_indicator(hard, pr, pc):
    G0, G1 = hard.shape
    n, diff = 0, 0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                i, j = pr + di, pc + dj
                if 0 <= i < G0 and 0 <= j < G1:
                    n += 1; diff += int(hard[i, j] != hard[pr, pc])
    return diff / max(n, 1)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            total += m.mean() * gap
            rows.append((float(lo), float(hi), int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return float(total), rows


def cluster_bootstrap_diff(sig_a, sig_b, err, clusters, n_boot=2000, seed=0):
    """Bootstrap over clusters of the AURC difference (a - b); negative favours a."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in ids}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        sel = np.concatenate([idx_by[c] for c in pick])
        if err[sel].sum() == 0:
            continue
        diffs.append(aurc_expected(sig_a[sel], err[sel]) - aurc_expected(sig_b[sel], err[sel]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs < 0).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    windows = [w for w in awf.list_windows() if w[1] == "val"]
    print(f"{len(windows)} validation windows; device {DEV}")
    probe = {os.path.basename(r["window"].replace("\\", "/")): r for r in csv.DictReader(open(os.path.join(OUT, "exp16_awf_boundary.csv")))}
    stacks = load_stacks(windows)
    model, w, b = load_finetuned()

    rows = []
    for crop in CROPS:
        t0 = time.time()
        per_shift = {}
        for shift in SHIFTS:
            crops, locs = [], []
            for i, (wdir, split, r, c, cat) in enumerate(windows):
                cr, (pr, pc) = crop_at(stacks[i], r, c, crop, shift)
                crops.append(cr); locs.append((pr, pc))
            outs = [logits_grid(model, w, b, crops[j:j + 32]) for j in range(0, len(crops), 32)]
            lg = np.concatenate([o[0] for o in outs]); px = np.concatenate([o[1] for o in outs])
            per_shift[shift] = (lg, px, locs)
        print(f"  crop {crop}: {len(SHIFTS)} shifts in {time.time() - t0:.0f}s", flush=True)
        lg0, px0, locs0 = per_shift[0]
        for i, (wdir, split, r, c, cat) in enumerate(windows):
            yr, xc = locs0[i]                            # label pixel within the crop
            pr, pc = yr // PATCH, xc // PATCH
            g = lg0[i]                                   # (G, G, 10) patch logits, for the boundary map
            lp = px0[i][yr, xc]                          # (10,) pixel logits at the label pixel (bilinear, as served)
            p = softmax(lp)
            pred = int(lp.argmax())
            srt = np.sort(lp)
            hard = g.argmax(-1)
            # aligned tiling instability: probability of the shift-0 predicted class at the label pixel, across shifts
            probs, flips = [], []
            for shift in SHIFTS:
                _, px_s, locs_s = per_shift[shift]
                ps = softmax(px_s[i][locs_s[i][0], locs_s[i][1]])
                probs.append(ps[pred]); flips.append(int(ps.argmax() != pred))
            probs = np.array(probs)
            name = os.path.basename(wdir)
            pb = probe.get(name)
            stack = stacks[i]
            red, nir = stack[r, c, :, 2], stack[r, c, :, 3]     # B04, B08 in OlmoEarth band order
            ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
            rows.append({
                "crop": crop, "window": name, "task": pb["task"] if pb else name.split("_point_")[0], "label": cat, "pred": pred,
                "error": int(pred != cat), "class_name": AWF_CLASSES[cat] if cat < len(AWF_CLASSES) else str(cat),
                "logit_margin": float(srt[-1] - srt[-2]), "top1_prob": float(p.max()),
                "boundary": boundary_indicator(hard, pr, pc), "tile_phase": float(probs.std()),
                "tile_phase_flip": float(np.mean(flips)),
                "probe_pred": int(pb["pred"]) if pb else -1, "probe_error": int(pb["error"]) if pb else -1,
                "probe_disagree": int(int(pb["pred"]) != pred) if pb else -1,
                "control_ndvi_tstd": float(np.nanstd(ndvi)),
                "grid": g.shape[0],
            })

    with open(os.path.join(OUT, "exp21_finetuned_awf.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    summary = {"n_val": len(windows), "device": DEV, "checkpoint": REPO, "crops": {}}
    for crop in CROPS:
        R = [r for r in rows if r["crop"] == crop]
        err = np.array([r["error"] for r in R], float)
        lab = np.array([r["label"] for r in R]); prd = np.array([r["pred"] for r in R])
        clusters = np.array([r["task"] for r in R])
        signals = {
            "confidence (neg logit margin)": -np.array([r["logit_margin"] for r in R]),
            "boundary indicator": np.array([r["boundary"] for r in R]),
            "tiling instability (aligned)": np.array([r["tile_phase"] for r in R]),
            "probe disagreement": np.array([r["probe_disagree"] for r in R], float),
            "control (NDVI temporal std)": np.array([r["control_ndvi_tstd"] for r in R]),
        }
        aurc = {k: aurc_expected(v, err) for k, v in signals.items()}
        base = signals["confidence (neg logit margin)"]
        boots = {k: cluster_bootstrap_diff(v, base, err, clusters) for k, v in signals.items() if not k.startswith("confidence")}
        top1 = np.array([r["top1_prob"] for r in R]); correct = 1 - err
        e, rel = ece(top1, correct)
        order = np.argsort(base, kind="stable")   # most confident first
        sel_acc = {str(c): float(correct[order[:max(1, int(round(c * len(R))))]].mean()) for c in (0.5, 0.8, 0.9, 1.0)}
        cap = {}
        for bud in BUDGETS:
            k = max(1, int(round(bud * len(R))))
            cap[str(bud)] = {kk: float(err[np.argsort(v, kind="stable")[::-1][:k]].sum() / max(err.sum(), 1)) for kk, v in signals.items()}
        recall = {AWF_CLASSES[c]: {"n": int((lab == c).sum()), "recall": float((prd[lab == c] == c).mean()) if (lab == c).any() else None} for c in range(len(AWF_CLASSES))}
        probe_err = np.array([r["probe_error"] for r in R], float)
        summary["crops"][str(crop)] = {
            "accuracy": float(correct.mean()), "n_errors": int(err.sum()), "probe_accuracy_exp16": float(1 - probe_err.mean()),
            "aurc": aurc, "aurc_oracle": aurc_expected(err, err), "aurc_random": float(err.mean()),
            "bootstrap_vs_confidence (lo, hi, P(signal better))": boots,
            "ece_10bins": e, "reliability": rel, "selective_accuracy_at_coverage": sel_acc,
            "error_capture_at_budget": cap, "per_class_recall": recall,
            "boundary_share_errors": float((signals["boundary indicator"][err > 0] > 0).mean()),
            "boundary_share_correct": float((signals["boundary indicator"][err == 0] > 0).mean()),
            "tile_flip_share_errors": float(np.array([r["tile_phase_flip"] for r in R])[err > 0].mean()),
            "tile_flip_share_correct": float(np.array([r["tile_phase_flip"] for r in R])[err == 0].mean()),
            "ft_vs_probe": {"both_wrong": int(((err > 0) & (probe_err > 0)).sum()), "ft_only_wrong": int(((err > 0) & (probe_err == 0)).sum()),
                            "probe_only_wrong": int(((err == 0) & (probe_err > 0)).sum()), "disagreement_rate": float(signals["probe disagreement"].mean())},
        }
        print(f"crop {crop}: acc {correct.mean():.3f} ({int(err.sum())} errors; probe {1 - probe_err.mean():.3f}) | ECE {e:.3f} | " +
              " | ".join(f"{k.split(' (')[0]} {v:.4f}" for k, v in aurc.items()) + f" | oracle {aurc_expected(err, err):.4f} random {err.mean():.4f}")
    with open(os.path.join(OUT, "exp21_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    make_figure(rows, summary)


def make_figure(rows, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from oe_inferencex import figstyle
    figstyle.setup()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    crop = "16"
    R = [r for r in rows if r["crop"] == 16]
    err = np.array([r["error"] for r in R], float)
    S = summary["crops"][crop]
    ax = axes[0]
    for name, s, color in (("confidence (neg logit margin)", -np.array([r["logit_margin"] for r in R]), "#1f77b4"),
                           ("boundary indicator", np.array([r["boundary"] for r in R]), "#ff7f0e"),
                           ("tiling instability (aligned)", np.array([r["tile_phase"] for r in R]), "#2ca02c"),
                           ("probe disagreement", np.array([r["probe_disagree"] for r in R], float), "#9467bd"),
                           ("control (NDVI temporal std)", np.array([r["control_ndvi_tstd"] for r in R]), "#7f7f7f")):
        order = np.argsort(s, kind="stable"); ke = np.cumsum(err[order]); cov = np.arange(1, len(err) + 1) / len(err)
        ax.plot(cov, ke / np.arange(1, len(err) + 1), color=color, label=f"{name} (AURC {S['aurc'][name]:.4f})")
    ax.axhline(err.mean(), color="k", linestyle="--", linewidth=0.8, label=f"random ({err.mean():.3f})")
    ax.set_xlabel("coverage (fraction of windows kept)"); ax.set_ylabel("risk (error rate among kept)")
    ax.set_title(f"(a) fine-tuned AWF model, 16-px crops, val n={len(R)}, {int(err.sum())} errors"); ax.legend(fontsize=7)
    ax = axes[1]
    rel = S["reliability"]
    ax.bar([(lo + hi) / 2 for lo, hi, *_ in rel], [acc for *_, acc in rel], width=0.09, color="#1f77b4", alpha=0.8, label="observed accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    for lo, hi, n, cm, acc in rel:
        ax.text((lo + hi) / 2, acc + 0.02, str(n), ha="center", fontsize=7)
    ax.set_xlabel("top-1 probability (bin)"); ax.set_ylabel("accuracy in bin"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title(f"(b) reliability, ECE {S['ece_10bins']:.3f}; counts above bars"); ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    names = [k for k, v in S["per_class_recall"].items() if v["n"] > 0]
    ax.barh(names, [S["per_class_recall"][k]["recall"] for k in names], color="#ff7f0e")
    for i, k in enumerate(names):
        ax.text(0.01, i, f"n={S['per_class_recall'][k]['n']}", va="center", fontsize=7, color="white" if S["per_class_recall"][k]["recall"] > 0.15 else "black")
    ax.set_xlim(0, 1); ax.set_xlabel("recall on validation points"); ax.set_title(f"(c) per-class recall, accuracy {S['accuracy']:.3f}")
    ax.invert_yaxis()
    fig.suptitle("exp21: allenai/OlmoEarth-v1-FT-AWF-Base run end to end on the AWF validation split", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "exp21_finetuned_awf.png"), dpi=150)
    print("figure written")


if __name__ == "__main__":
    main()
