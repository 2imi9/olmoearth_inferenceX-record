"""Exp 08: masking-perturbation ensemble (GPU).

The sampling-ensemble port of SelfCheckGPT for a deterministic encoder:
occlude a random 15% of patch cells with mean-fill (zeros in normalized
space), rerun inference, repeat N times; the per-patch standard deviation of
the predicted water probability across the runs in which that patch was
visible is the instability signal. Evaluated on the three cached scenes
against the same Base-head errors, alongside the baseline, E_case, and the
4-shift tile-phase signal for reference.
"""
import numpy as np
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from oe_inferencex.evidence import train_logistic_head, predict_head, risk_coverage

SIZE, PATCH = 128, 4
GRID = SIZE // PATCH
N_MASK = 32
MASK_FRAC = 0.15
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_sample(img, date, device):
    x = img[:, :SIZE, :SIZE].transpose(1, 2, 0)[None, :, :, None, :].astype(np.float64)
    x = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, x)
    d, m0, y = date
    return (torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor([d, m0, y], device=device)[None, None, :])


def embed_batch(model, xs, ts):
    b = xs.shape[0]
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=xs,
        sentinel2_l2a_mask=torch.ones((b, SIZE, SIZE, 1, 3), device=xs.device) * MaskValue.ONLINE_ENCODER.value,
        timestamps=ts.repeat(b, 1, 1),
    )
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4])  # (b, GRID, GRID, D)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z3 = np.load("exp/out/exp03_cache.npz")
    z5 = np.load("exp/out/exp05_cache.npz")
    tr_img, tr_date, tr_labels = z3["tr_img"], tuple(int(v) for v in z3["tr_date"]), z3["tr_labels"]
    scenes = {
        "kazungula": (z3["ev_img"], tuple(int(v) for v in z3["ev_date"]), z3["ev_labels"]),
        "hard_barotse": (z5["hard_barotse_img"], tuple(int(v) for v in z5["hard_barotse_date"]), z5["hard_barotse_lab"]),
        "ood_delta": (z5["ood_delta_img"], tuple(int(v) for v in z5["ood_delta_date"]), z5["ood_delta_lab"]),
    }

    models = {k: load_model_from_id(m).to(DEV).eval()
              for k, m in (("nano", ModelID.OLMOEARTH_V1_NANO), ("base", ModelID.OLMOEARTH_V1_BASE))}

    xs_tr, ts_tr = make_sample(tr_img, tr_date, DEV)
    heads = {}
    for k, m in models.items():
        f = embed_batch(m, xs_tr, ts_tr)[0].cpu()
        heads[k] = train_logistic_head(f, tr_labels)

    print(f"device={DEV}, N_MASK={N_MASK}, mask_frac={MASK_FRAC}")
    results = {}
    for name, (img, date, labels) in scenes.items():
        xs, ts = make_sample(img, date, DEV)
        p = {k: predict_head(embed_batch(m, xs, ts)[0].cpu(), *heads[k]) for k, m in models.items()}
        errors = ((p["base"] > 0.5) != labels.astype(bool)).astype(np.float64)

        # tile-phase reference (4 diagonal shifts on the padded image)
        shift_p = []
        for s in range(4):
            xv, tv = make_sample(img[:, s:s + SIZE, s:s + SIZE], date, DEV)
            shift_p.append(predict_head(embed_batch(models["base"], xv, tv)[0].cpu(), *heads["base"]))
        tile_phase = np.stack(shift_p).std(0)

        # masking ensemble, batched on GPU
        masks = rng.random((N_MASK, GRID, GRID)) < MASK_FRAC  # True = occluded
        xb = xs.repeat(N_MASK, 1, 1, 1, 1)
        mask_px = torch.tensor(
            np.kron(masks, np.ones((PATCH, PATCH), dtype=bool)), device=DEV
        )[:, :, :, None, None]
        xb = xb.masked_fill(mask_px, 0.0)  # 0 = per-band mean in normalized space
        feats = []
        for i in range(0, N_MASK, 8):
            feats.append(embed_batch(models["base"], xb[i:i + 8], ts).cpu())
        feats = torch.cat(feats)  # (N_MASK, GRID, GRID, D)
        probs = np.stack([predict_head(feats[i], *heads["base"]) for i in range(N_MASK)])
        vis = ~masks  # only count runs where the patch itself was visible
        pm = np.where(vis, probs, np.nan)
        mask_sig = np.nanstd(pm, axis=0)

        signals = {
            "baseline max-softmax": 1 - np.maximum(p["base"], 1 - p["base"]),
            "E_case |Nano-Base|": np.abs(p["nano"] - p["base"]),
            "E_system tile-phase (4 shifts)": tile_phase,
            f"E_system mask-perturb (N={N_MASK})": mask_sig,
        }
        res = {}
        print(f"\n{name}: {int(errors.sum())} errors")
        for sn, sig in signals.items():
            cov, risk, aurc = risk_coverage(sig, errors)
            res[sn] = (cov, risk, aurc)
            print(f"  AURC {sn}: {aurc:.4f}")
        results[name] = res

    from oe_inferencex.figstyle import setup, rc_panel
    import matplotlib.pyplot as plt
    setup()
    titles = {
        "kazungula": "Kazungula (in-domain, 18 errors)",
        "hard_barotse": "Barotse floodplain (ambiguous margins, 97 errors)",
        "ood_delta": "Zambezi delta (geographic shift, 29 errors)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5))
    for i, (name, res) in enumerate(results.items()):
        rc_panel(axes[i], res, titles[name], idx=i)
    fig.suptitle(f"Masking-perturbation ensemble (occlude {int(MASK_FRAC*100)}% of patches, "
                 f"N={N_MASK} reruns, GPU) vs prior signals; identical errors per scene", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("exp/out/exp08_mask_perturb.png", bbox_inches="tight")
    print("\nwrote exp/out/exp08_mask_perturb.png")


if __name__ == "__main__":
    main()
