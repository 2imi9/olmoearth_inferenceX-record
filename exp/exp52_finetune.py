#!/usr/bin/env python
"""exp52: fine-tune OlmoEarth ourselves on Sen1Floods11 and grade the fine-tuned models under the window protocol.

Why. Every fine-tuned-model claim here rests on Ai2's AWF model (exp21: fine-tuning corrects 55.6% of the frozen
head's errors, phi 0.475). A fine-tune we run ourselves, on the flood task where every other result lives, verifies
that with our own hands: does training the encoder correct half the frozen head's errors here too, does the
fine-tuned model's confidence rank its own errors better, and does the Bolivia exception, a no-model spectral index
matching or beating the frozen S2 head's confidence, survive once the head is no longer frozen. Three sensors settle
exp46's lever with a trained model instead of a linear probe: Sentinel-2 (ours), Sentinel-1 (Ai2's eval sensor),
and both in one pass.

Recipe. Ai2's fine-tuning harness (olmoearth_pretrain/evals/finetune) restated in fp32 without olmo_core: the
encoder with a linear per-pixel head (two classes x 4 x 4 logits per token, the pooled band-set token per window,
the two sensors' tokens concatenated in the joint arm), AdamW at LR, the backbone frozen for the first 20% of the
epochs then unfrozen at 0.1 x LR, ReduceLROnPlateau on validation mIoU (factor 0.2, patience 2, cool-down 10),
cross-entropy ignoring -1, the checkpoint with the best validation mIoU kept. Training tiles: the bucket's train
split (6,790 tiles) cropped at offset 0 to 60 px; validation: the valid split at offset 0; grading: Bolivia (441
tiles) and the test split as exp18 sampled it (800 tiles, seed 1), never trained on. LR 3e-4, EPOCHS epochs, batch 32,
seed 0. Not Ai2's exact hyper-parameters, which the paper does not state; ours, stated here.

Grading, per arm, exp47's protocol: the four crop offsets, per-pixel probabilities pooled to the 4-px window grid,
the shift-averaged (W1) decision's own error set, the averaged confidence against aligned tile-phase and the
no-model NDWI-level control (pooled excess AURC, per-tile sign tests); pixel accuracy of the W1 decision; and the
cross-tab against the frozen exp18 head's W1 decision on identical windows (share of the frozen head's errors the
fine-tuned model corrects, share it breaks, phi), as exp21 did with Ai2's AWF model.

Preregistered, one-sided, stated before the run:
  P1  FT-S2 on v1, Bolivia: the fine-tuned model's averaged confidence beats the NDWI-level control (pooled lead
      >= 0.001, per-tile sign test p < 0.05). Falsification: the exception is the event's, not the frozen encoder's.
  P2  FT-S2 on v1 corrects at least 40% of the frozen S2 head's W1 errors on both testbeds. Falsification: on this
      task fine-tuning mostly re-labels the same windows.
Secondary, descriptive: FT-S1+S2 against FT-S2 (accuracy, shared errors); FT-S1 against FT-S2 (the sensor lever
after training); FT-S2 on v1.2; pixel accuracy against the frozen W1 (exp42: 0.9071 / 0.9503).

Inputs: data/floods/ (train downloaded from the bucket when absent), exp/out/exp18_feats.npz. Outputs:
exp/out/exp52_summary.json, exp/out/exp52_finetune.csv. Run with ~/oe12/.venv. --smoke: CPU, 4 tiles, 1 epoch,
FT-S2 on v1 only, _smoke outputs.
"""
import copy
import csv
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp47_served_ranker as e47  # noqa: E402
import exp49_shrug_signals as e49  # noqa: E402
import exp51_their_probe as e51  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import crosstab as compare_crosstab  # noqa: E402
from oe_inferencex.signals import ndwi_level  # noqa: E402

PATCH, CROP, G, SHIFTS, DEV = exp18.PATCH, exp18.CROP, exp18.G, exp18.SHIFTS, exp18.DEV
LR, EPOCHS, BATCH, FREEZE_FRAC, UNFREEZE_LR_FACTOR = 3e-4, 12, 32, 0.2, 0.1
ARMS = [("v1", "s2"), ("v1", "s1"), ("v1", "s1s2"), ("v1_2", "s2")]
W1C, TILE, CTRL_L = e47.W1C, e47.TILE, e47.CTRL_L


def prep(s2, s1, shift):
    """Normalised inputs at a crop offset: S2 (N, 60, 60, 1, 12) and S1 (N, 60, 60, 1, 2), float32 numpy."""
    x2 = s2[:, :, shift:shift + CROP, shift:shift + CROP].transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
    x2 = exp18._norm.normalize(Modality.SENTINEL2_L2A, x2).astype(np.float32)
    x1 = np.nan_to_num(s1[:, :, shift:shift + CROP, shift:shift + CROP], nan=0.0, posinf=0.0, neginf=0.0).transpose(0, 2, 3, 1)[:, :, :, None, :].astype(np.float64)
    x1 = exp18._norm.normalize(Modality.SENTINEL1, x1).astype(np.float32)
    return x2, x1


class FineTuner(torch.nn.Module):
    """Encoder + per-pixel linear head on the pooled window tokens of the chosen sensors."""

    def __init__(self, model, mods):
        super().__init__()
        self.model, self.mods = model, mods
        self.head = torch.nn.Linear(768 * len(mods), 2 * PATCH * PATCH)

    def features(self, x2, x1):
        b = x2.shape[0]
        kw = {"timestamps": torch.tensor([1, 5, 2020], device=DEV)[None, None, :].repeat(b, 1, 1)}
        if "s2" in self.mods:
            kw["sentinel2_l2a"] = x2; kw["sentinel2_l2a_mask"] = torch.ones((b, CROP, CROP, 1, 3), device=DEV) * MaskValue.ONLINE_ENCODER.value
        if "s1" in self.mods:
            kw["sentinel1"] = x1; kw["sentinel1_mask"] = torch.ones((b, CROP, CROP, 1, 1), device=DEV) * MaskValue.ONLINE_ENCODER.value
        out = self.model.encoder(MaskedOlmoEarthSample(**kw), fast_pass=True, patch_size=PATCH)["tokens_and_masks"]
        parts = []
        if "s2" in self.mods:
            parts.append(out.sentinel2_l2a.mean(dim=[3, 4]))
        if "s1" in self.mods:
            parts.append(out.sentinel1.mean(dim=[3, 4]))
        return torch.cat(parts, dim=-1)                                            # (b, G, G, D * n_mods)

    def forward(self, x2, x1):
        f = self.features(x2, x1)
        b = f.shape[0]
        return self.head(f).reshape(b, G, G, 2, PATCH, PATCH).permute(0, 3, 1, 4, 2, 5).reshape(b, 2, CROP, CROP)


def set_backbone(ft, trainable):
    for p in ft.model.parameters():
        p.requires_grad_(trainable)


def batches(n, bs, order=None):
    idx = np.arange(n) if order is None else order
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def predict(ft, x2, x1, bs=64):
    """(N, 60, 60) water probability, fp32."""
    ft.eval(); out = []
    with torch.no_grad():
        for idx in batches(len(x2), bs):
            lg = ft(torch.tensor(x2[idx], device=DEV), torch.tensor(x1[idx], device=DEV))
            out.append(torch.softmax(lg, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def finetune(model, mods, tr, va, epochs, seed=0, log=print):
    """Their recipe restated: freeze, unfreeze at 0.1 x LR, plateau scheduler on validation mIoU, best checkpoint kept."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    ft = FineTuner(model, mods).to(DEV)
    x2, x1, lab = tr
    vx2, vx1, vlab = va
    opt = torch.optim.AdamW([p for p in ft.parameters()], lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.2, patience=2, cooldown=10, min_lr=0.0)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    freeze_epochs = int(np.ceil(FREEZE_FRAC * epochs)) if epochs > 0 else 0
    set_backbone(ft, freeze_epochs == 0)
    best, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        if epoch == freeze_epochs and freeze_epochs > 0:
            set_backbone(ft, True)
            for g in opt.param_groups:
                g["lr"] = LR * UNFREEZE_LR_FACTOR
        ft.train(); t0 = time.time(); tot = 0.0
        for idx in batches(len(x2), BATCH, rng.permutation(len(x2))):
            lg = ft(torch.tensor(x2[idx], device=DEV), torch.tensor(x1[idx], device=DEV))
            loss = loss_fn(lg, torch.tensor(lab[idx], device=DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        p = predict(ft, vx2, vx1)
        miou, acc = e51.segmentation_miou((p > 0.5).astype(np.int64), vlab)
        sched.step(miou)
        history.append({"epoch": epoch, "train_loss": tot / len(x2), "val_miou": miou, "val_acc": acc, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        log(f"    epoch {epoch}: loss {tot / len(x2):.4f} val mIoU {miou:.4f} acc {acc:.4f} lr {opt.param_groups[0]['lr']:.1e} ({time.time() - t0:.0f}s)")
        if miou > best:
            best, best_state = miou, copy.deepcopy({k: v.detach().cpu() for k, v in ft.state_dict().items()})
    ft.load_state_dict(best_state)
    return ft, history, best


def crosstab(err_a, err_b, ok):
    """Windows where the frozen head (a) errs and the fine-tuned model (b) does not, and the reverse
    (oe_inferencex.compare.crosstab since exp57, under the key names this experiment recorded)."""
    ct = compare_crosstab(err_a, err_b, ok)
    return {"frozen_errors": ct["errors_a"], "corrected": ct["corrected"], "broken": ct["broken"], "both": ct["both"],
            "share_corrected": ct["share_corrected"], "phi": ct["phi"]}


def main():
    args = hb.make_parser(__doc__).parse_args()
    suffix = "_smoke" if args.smoke else ""
    epochs = 1 if args.smoke else EPOCHS
    summary = {"experiment": "exp52 fine-tune OlmoEarth on Sen1Floods11 and grade under the window protocol", "smoke": args.smoke,
               "config": {"lr": LR, "epochs": epochs, "batch": BATCH, "freeze_fraction": FREEZE_FRAC, "unfreeze_lr_factor": UNFREEZE_LR_FACTOR, "arms": ARMS, "crop": CROP,
                          "prereg": "P1 FT-S2 v1 Bolivia: averaged confidence beats NDWI level (lead >= 0.001, one-sided per-tile sign test); "
                                    "P2 FT-S2 v1 corrects >= 40% of the frozen S2 head's W1 errors on both testbeds"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    n = args.smoke_tiles if args.smoke else None
    tr_s1, tr_s2, tr_lab = e51.load_ours(floods_dir, "valid" if args.smoke else "train", n, seed=3)
    va_s1, va_s2, va_lab = e51.load_ours(floods_dir, "valid", n, seed=0)
    bo = e51.load_ours(floods_dir, "bolivia", n, seed=1)
    te = e51.load_ours(floods_dir, "valid" if args.smoke else "test", n if args.smoke else exp18.N_TEST_TILES, seed=7 if args.smoke else 1)
    splits = {"bolivia": bo} if args.smoke else {"bolivia": bo, "test": te}
    tr = (*prep(tr_s2, tr_s1, 0), tr_lab[:, :CROP, :CROP])
    va = (*prep(va_s2, va_s1, 0), va_lab[:, :CROP, :CROP])
    print(f"train {len(tr_s2)} tiles, valid {len(va_s2)}, grading " + ", ".join(f"{k} {len(v[0])}" for k, v in splits.items()), flush=True)

    # the frozen reference: exp18's head on the cached v1 features, its W1 error set on the same windows (exp47's arm)
    frozen = {}
    try:
        cache = np.load(os.path.join(hb.OUT, "exp42_floods_feats_smoke.npz" if args.smoke else "exp18_feats.npz"))
        trf = np.asarray(cache["tr_base"], dtype=np.float32)
        head_tr_s2, head_tr_lab = hb.load_floods_split(os.path.join(floods_dir, "flood_valid_data.pt"), args.smoke_tiles if args.smoke else exp18.N_TRAIN_TILES)
        head = e47.head_from(trf, head_tr_lab)
        for name, (s1, s2, lab) in splits.items():
            key = f"{'bolivia' if name == 'bolivia' else 'test'}_base"
            if all(f"{key}{s}" in cache.files and len(cache[f"{key}{s}"]) == len(s2) for s in SHIFTS):
                pshift = np.stack([exp18.head_prob_logit(np.asarray(cache[f"{key}{s}"], dtype=np.float32), *head)[0] for s in SHIFTS])
                w1 = e49.w1_windows(pshift)[:, 1:G, 1:G]
                y, ok = exp18.patch_labels(lab[:, :CROP, :CROP]); y, ok = y[:, 1:G, 1:G], ok[:, 1:G, 1:G]
                frozen[name] = {"err": ((w1 > 0.5) != (y > 0.5)).astype(np.float64), "ok": ok}
            else:
                summary["failures"].append({"part": f"frozen reference {name}", "error": "cache does not cover this split's tiles"})
    except Exception as ex:  # noqa: BLE001
        summary["failures"].append({"part": "frozen reference", "error": repr(ex), "traceback": traceback.format_exc()})

    arms = [("v1", "s2")] if args.smoke else ARMS
    for version, mod in arms:
        arm = f"FT-{mod.upper().replace('S1S2', 'S1+S2')} {version}"
        t0 = time.time()
        try:
            if version == "v1":
                model = hb.load_model()
            else:
                from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
                model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(DEV).float()
            mods = ["s2", "s1"] if mod == "s1s2" else [mod]
            ft, history, best = finetune(model, mods, tr, va, epochs, log=lambda m: print(f"  {arm}{m}", flush=True))
            res = {"train_seconds": time.time() - t0, "best_val_miou": best, "history": history}
            for name, (s1, s2, lab) in splits.items():
                pshift = []
                for s in SHIFTS:
                    x2, x1 = prep(s2, s1, s)
                    p = predict(ft, x2, x1)
                    pshift.append(p.reshape(len(p), G, PATCH, G, PATCH).mean(axis=(2, 4)))   # per-pixel probabilities pooled to the window grid
                    if s == 0:
                        pix_ok = lab[:, :CROP, :CROP] >= 0
                        pix_acc0 = float(((p > 0.5) == (lab[:, :CROP, :CROP] == 1))[pix_ok].mean())
                pshift = np.stack(pshift)
                w1 = e49.w1_windows(pshift)[:, 1:G, 1:G]
                y, ok = exp18.patch_labels(lab[:, :CROP, :CROP]); y, ok = y[:, 1:G, 1:G], ok[:, 1:G, 1:G]
                err = ((w1 > 0.5) != (y > 0.5)).astype(np.float64)
                sig = {W1C: -np.abs(w1 - 0.5), TILE: exp18.aligned_tile_phase(pshift)[:, 1:G, 1:G],
                       CTRL_L: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])[:, 1:G, 1:G]}
                r = e47.score(sig, err, ok)
                r["window_accuracy"] = float(1 - err[ok].mean()); r["pixel_accuracy_shift0"] = pix_acc0
                if name in frozen:
                    r["vs_frozen_head"] = crosstab(frozen[name]["err"], err, ok & frozen[name]["ok"])
                    r["frozen_window_accuracy"] = float(1 - frozen[name]["err"][frozen[name]["ok"]].mean())
                res[name] = r
                ct = r.get("vs_frozen_head", {})
                print(f"{arm}/{name}: window acc {r['window_accuracy']:.4f} (frozen {r.get('frozen_window_accuracy', float('nan')):.4f}), pixel acc {pix_acc0:.4f} | "
                      f"E-AURC confidence {r['pooled_eaurc'][W1C]:.4f}, tile-phase {r['pooled_eaurc'][TILE]:.4f}, NDWI level {r['pooled_eaurc'][CTRL_L]:.4f} | "
                      f"lead over NDWI {r['tests'][CTRL_L]['pooled_lead']:+.4f} (p={r['tests'][CTRL_L]['sign_p']:.2g}) | "
                      + (f"corrects {ct['share_corrected']:.3f} of the frozen head's errors ({ct['corrected']}/{ct['frozen_errors']}), breaks {ct['broken']}, phi {ct['phi']:.3f}" if ct else ""), flush=True)
                rows.append({"arm": arm, "split": name, "window_acc": r["window_accuracy"], "frozen_window_acc": r.get("frozen_window_accuracy"), "pixel_acc_shift0": pix_acc0,
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()}, "share_frozen_errors_corrected": ct.get("share_corrected"), "phi_vs_frozen": ct.get("phi")})
            res["_err"] = {name: (res[name].pop("_err", None)) for name in splits}   # placeholder, error sets kept below
            summary["results"][arm] = {k: v for k, v in res.items() if k != "_err"}
            summary.setdefault("_errsets", {})[arm] = {name: None for name in splits}
            del ft, model
            if DEV == "cuda":
                torch.cuda.empty_cache()
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": arm, "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"{arm} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
    summary.pop("_errsets", None)

    r = summary["results"].get("FT-S2 v1", {})
    p1 = p2 = None
    if "bolivia" in r:
        t = r["bolivia"]["tests"][CTRL_L]
        p1 = bool(t["pooled_lead"] >= e47.MIN_LEAD and t["sign_p"] < 0.05)
    if all(n in r and "vs_frozen_head" in r[n] for n in splits):
        p2 = bool(all(r[n]["vs_frozen_head"]["share_corrected"] >= 0.40 for n in splits))
    summary["prereg"] = {"P1": p1, "P2": p2, "supported": bool(p1 and p2) if (p1 is not None and p2 is not None) else None, "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        with open(os.path.join(hb.OUT, f"exp52_finetune{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp52_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp52_summary{suffix}.json; prereg {summary['prereg']}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
