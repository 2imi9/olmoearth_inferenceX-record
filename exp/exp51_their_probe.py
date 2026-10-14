#!/usr/bin/env python
"""exp51: Ai2's own Sen1Floods11 probe under the window protocol, on their embeddings and on ours.

Why. Every statement this repository makes about OlmoEarth v1.2's error ranking rests on our head (a balanced
logistic probe on pooled Sentinel-2 tokens) and our sensor. Ai2's Sen1Floods11 evaluation is a different object:
a Sentinel-1 linear probe (olmoearth_pretrain/evals/datasets/configs.py, sen1floods11 -> [SENTINEL1]; the v1.2
paper's Table 1 marks the task S1) whose head emits sixteen pixel logits per token (LinearProbe, out_dim =
classes x 4 x 4, no batchnorm), trained with AdamW at lr 0.1, ten percent warm-up then half-cycle cosine to 1e-5,
cross-entropy with ignore index -1, fifty epochs. Ai2 also publishes the embeddings that fed their Table 2
(allenai/olmoearth-paper-embeddings: {model}/sen1floods11/{train,valid,test}.pt, bf16 (N, 16, 16, 768) at the
4-px token grid on 64-px tiles, labels (N, 64, 64), with the paper-best settings in eval_settings/). So their
readout can be run here, on their embeddings and on our own encodes, and graded like every other ranker.

Design.
  A  their embeddings, their probe: OlmoEarth Base (run phase2.0_base_lr0.0001_wd0.02, the v1 Base taskcard's run)
     trained on their train split, graded on their test split; pixel mIoU and accuracy as they report, then the
     window protocol (token = 4-px window, majority label, the probe's own confidence = the window-mean water
     probability's distance from 0.5). Their tiles are matched to ours by exact equality of the 64 x 64 label
     tile, which lets the no-model NDWI-level control and our S2 head's error set sit on the same windows.
  B  our Sentinel-1 encode of v1 Base (exp46.embed_s1, 60-px crop, shift 0), their probe recipe, trained on the
     bucket's train split (research_benchmarks/floods/flood_train_data.pt), graded on Bolivia and the test split.
  C  the same with v1.2 Base (~/oe12/.venv). B against A checks that our encode path reproduces their probe;
     C against B is the v1.2 question asked with their readout and their sensor, on Bolivia, which their splits
     do not contain.
  D  the other encoders whose Sen1Floods11 embeddings they publish (galileo_base, croma_base, terramind_base,
     clay_large, satlas_base, anysat, panopticon), same probe, same tiles: accuracy and the phi of window errors
     against OlmoEarth Base, exp41's question at scale with no encoder passes.
Controls: the S1-level control -|window-mean VV - median over training windows| (the sensor's analogue of NDWI
level) and, on matched tiles, NDWI level from the bucket's Sentinel-2. fp32 throughout; their harness's bf16
autocast is not used.

Preregistered, one-sided, stated before the run:
  P1  arm B, test split: the probe's own confidence beats the S1-level control (pooled excess AURC lead >= 0.001
      and a per-tile exact sign test, confidence better, p < 0.05).
  P2  arm C against B, Bolivia: v1.2's confidence ranks its own errors worse than v1's, pooled excess AURC higher
      by >= 0.001, with a per-tile sign test (v1 better) p < 0.05 on tiles both arms score. This is exp47's
      Bolivia finding asked with their readout and sensor; if it fails, that finding belonged to our S2 head.
Secondary, descriptive: A's pixel mIoU against the paper (79.2 for v1 Base); A against B in accuracy (the encode
path); the NDWI-level control on matched tiles; D's accuracies and phi table.

Inputs: data/floods/ (train downloaded from the bucket when absent), the Hub dataset allenai/olmoearth-paper-embeddings
(downloaded into HF_HOME). Outputs: exp/out/exp51_summary.json, exp/out/exp51_their_probe.csv.
Run with ~/oe12/.venv. --smoke: CPU, 4 tiles per split, arm B on v1 only, no downloads, _smoke outputs.
"""
import csv
import hashlib
import json
import os
import sys
import time
import traceback
import urllib.request

import numpy as np
import torch

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.dirname(EXP_DIR))
import exp18_sen1floods_expert as exp18  # noqa: E402
import exp46_shared_error_sources as e46  # noqa: E402
import harness_ab as hb  # noqa: E402
from oe_inferencex.compare import phi  # noqa: E402,F401  (the float-safe copy lived here until exp57 promoted it)
from oe_inferencex.metrics import aurc_expected, oracle_aurc  # noqa: E402
from oe_inferencex.signals import ndwi_level  # noqa: E402
from oe_inferencex.stats import sign_test, wins_losses_ties  # noqa: E402


def adjust_learning_rate(optimizer, epoch, warmup_epochs, total_epochs, max_lr, min_lr):
    """Their schedule (evals/utils.py), restated: linear warm-up then half-cycle cosine to min_lr."""
    if epoch < warmup_epochs:
        lr = max_lr * epoch / warmup_epochs
    else:
        lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def segmentation_miou(pred, lab):
    """Their segmentation metric (evals/metrics.py imports sklearn, absent in the inference environments), restated:
    mean over the two classes of tp / (tp + fp + fn) over labelled pixels, and overall pixel accuracy."""
    m = lab >= 0
    p, y = pred[m], lab[m]
    ious = []
    for c in (0, 1):
        tp = float(((p == c) & (y == c)).sum()); fp = float(((p == c) & (y != c)).sum()); fn = float(((p != c) & (y == c)).sum())
        if tp + fp + fn > 0:
            ious.append(tp / (tp + fp + fn))
    return float(np.mean(ious)) if ious else float("nan"), float((p == y).mean()) if len(p) else float("nan")


class LinearProbe(torch.nn.Module):
    """olmoearth_pretrain.evals.linear_probe.LinearProbe with use_batchnorm=False, restated here because that module imports
    olmo_core, which the inference environments do not carry: one linear layer, logits in a dict."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return {"logits": self.linear(x)}

PATCH, CROP, G = exp18.PATCH, exp18.CROP, exp18.G
DEV = exp18.DEV
HUB = "allenai/olmoearth-paper-embeddings"
THEIRS = "olmoearth_base"
OTHERS = ["galileo_base", "croma_base", "terramind_base", "clay_large", "satlas_base", "anysat", "panopticon"]
LR, EPOCHS, BATCH, MIN_LEAD = 0.1, 50, 64, 0.001                   # eval_settings/base_settings: probe_lr 0.1
CONF, CTRL_S1, CTRL_NDWI = "probe confidence", "control S1 level", "control NDWI level"
PAPER_MIOU_V1_BASE = 79.2


# ----------------------------------------------------------------------------- their probe, fp32
def train_probe(emb, lab, pp, seed=0):
    """Their segmentation probe recipe: LinearProbe (no batchnorm) emitting classes x pp x pp logits per token,
    AdamW at LR with warm-up and half-cycle cosine, cross-entropy ignoring -1, EPOCHS epochs. emb (N, h, w, D) any
    float dtype on CPU; lab (N, H, W) int64 with H = h * pp."""
    torch.manual_seed(seed)
    N, h, w, D = emb.shape
    probe = LinearProbe(in_dim=D, out_dim=2 * pp * pp).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    steps = int(np.ceil(N / BATCH))
    g = torch.Generator().manual_seed(seed)
    probe.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(N, generator=g)
        for i in range(steps):
            idx = order[i * BATCH:(i + 1) * BATCH]
            x = emb[idx].to(DEV, dtype=torch.float32)
            y = lab[idx].to(DEV)
            logits = probe(x)["logits"].reshape(len(idx), h, w, 2, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(len(idx), 2, h * pp, w * pp)
            if logits.shape[-2:] != y.shape[-2:]:                                   # their harness: bilinear to the label size
                logits = torch.nn.functional.interpolate(logits, size=tuple(y.shape[-2:]), mode="bilinear", align_corners=True)
            loss = loss_fn(logits, y)
            loss.backward()
            adjust_learning_rate(optimizer=opt, epoch=epoch + i / steps, total_epochs=EPOCHS, warmup_epochs=int(EPOCHS * 0.1), max_lr=LR, min_lr=1.0e-5)
            opt.step()
            opt.zero_grad()
    return probe.eval()


def predict_pixels(probe, emb, pp, out_hw):
    """(N, H, W) water probability at pixel level, fp32; logits interpolated to out_hw when the token grid does not tile it."""
    out = []
    with torch.no_grad():
        for i in range(0, len(emb), 256):
            x = emb[i:i + 256].to(DEV, dtype=torch.float32)
            n, h, w = x.shape[:3]
            lg = probe(x)["logits"].reshape(n, h, w, 2, pp, pp).permute(0, 3, 1, 4, 2, 5).reshape(n, 2, h * pp, w * pp)
            if lg.shape[-2:] != tuple(out_hw):
                lg = torch.nn.functional.interpolate(lg, size=tuple(out_hw), mode="bilinear", align_corners=True)
            out.append(torch.softmax(lg, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def window_view(p_pix, lab, pp):
    """Pixel probabilities and labels -> the window grid: mean probability, majority label, validity, error, confidence."""
    N, H, W = p_pix.shape
    h, w = H // pp, W // pp
    wp = p_pix.reshape(N, h, pp, w, pp).mean(axis=(2, 4))
    l = lab.reshape(N, h, pp, w, pp)
    valid = l >= 0
    n_valid, n_water = valid.sum(axis=(2, 4)), (l == 1).sum(axis=(2, 4))
    ok = n_valid >= (pp * pp) / 2
    y = np.where(ok, n_water > n_valid / 2, False)
    err = ((wp > 0.5) != y).astype(np.float64)
    return wp, y, ok, err, -np.abs(wp - 0.5)


def s1_level(s1, med, pp, size):
    """The sensor's level control: -|window-mean VV - median VV of the training windows|; higher = closer to the boundary level."""
    x = np.nan_to_num(s1[:, 0, :size, :size], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    N = len(x)
    return -np.abs(x.reshape(N, size // pp, pp, size // pp, pp).mean(axis=(2, 4)) - med)


def score(sig, err, ok, ref, primary):
    tiles = [t for t in range(len(err)) if ok[t].any() and 3 <= err[t][ok[t]].sum() <= ok[t].sum() - 3]
    pooled = {k: aurc_expected(np.asarray(v)[ok], err[ok]) - oracle_aurc(int(ok.sum()), int(err[ok].sum())) for k, v in sig.items()}
    per = {k: [aurc_expected(np.asarray(sig[k][t])[ok[t]], err[t][ok[t]]) - oracle_aurc(int(ok[t].sum()), int(err[t][ok[t]].sum())) for t in tiles] for k in sig}
    tests = {}
    for k in sig:
        if k == ref:
            continue
        g = np.array(per[k]) - np.array(per[ref])                           # positive: the reference is better
        w, l, t_ = wins_losses_ties(g)
        one = k in primary
        tests[k] = {"w": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater" if one else "two-sided"), "one_sided": one,
                    "median_gain": float(np.median(g)) if len(g) else None, "pooled_lead": pooled[k] - pooled[ref]}
    return {"n_tiles_scored": len(tiles), "n_windows": int(ok.sum()), "n_errors": int(err[ok].sum()), "accuracy": float(1 - err[ok].mean()),
            "pooled_eaurc": pooled, "tests": tests, "per_tile_eaurc_ref": per[ref], "tiles": tiles}


# ----------------------------------------------------------------------------- data
def label_keys(lab):
    """One hash per 64 x 64 label tile, for matching their tiles to ours."""
    return [hashlib.sha1(np.ascontiguousarray(t.astype(np.int8)).tobytes()).hexdigest() for t in lab]


def load_theirs(model, split, cache_dir):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(HUB, f"{model}/sen1floods11/{split}.pt", repo_type="dataset", cache_dir=cache_dir)
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["embeddings"], d["labels"].to(torch.int64)


def load_ours(floods_dir, split, n=None, seed=0):
    """(s1 (N,2,64,64) float32, s2 (N,12,64,64) reordered, labels (N,64,64)) from the bucket's .pt files."""
    path = os.path.join(floods_dir, f"flood_{split}_data.pt")
    if not os.path.exists(path):
        url = hb.FLOODS_URL.format(split=split)
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    d = torch.load(path, weights_only=True)
    s1 = d["s1"].numpy().astype(np.float32)
    s2 = d["s2"].numpy().astype(np.float32)[:, exp18.BAND_IDX]
    lab = d["labels"].numpy()[:, 0].astype(np.int64)
    if n is not None and n < len(s1):
        idx = np.random.default_rng(seed).choice(len(s1), n, replace=False)
        s1, s2, lab = s1[idx], s2[idx], lab[idx]
    return s1, s2, lab


def checkpoint(summary, suffix):
    """Write the summary so far (without the in-memory arrays), so a late failure keeps the finished arms."""
    slim = json.loads(json.dumps(hb.json_ready({k: v for k, v in summary.items()}), default=str)) if False else None
    keep = {}
    for k, v in summary["results"].items():
        keep[k] = {kk: ({x: y for x, y in vv.items() if x not in ("err", "ok", "per_tile_eaurc_ref", "tiles")} if isinstance(vv, dict) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v
    os.makedirs(hb.OUT, exist_ok=True)
    with open(os.path.join(hb.OUT, f"exp51_summary{suffix}.partial.json"), "w") as f:
        json.dump(hb.json_ready({**{k: v for k, v in summary.items() if k != "results"}, "results": keep}), f, indent=1)


def main():
    ap = hb.make_parser(__doc__)
    ap.add_argument("--others", nargs="*", default=OTHERS, help="other encoders' published embeddings to grade (arm D)")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    summary = {"experiment": "exp51 Ai2's Sen1Floods11 probe under the window protocol", "smoke": args.smoke,
               "config": {"probe": "LinearProbe classes x 4 x 4 per token, no batchnorm, AdamW lr 0.1, warmup 10% + half-cycle cosine to 1e-5, CE ignore -1, 50 epochs, fp32",
                          "hub": HUB, "theirs": THEIRS, "others": args.others, "min_lead": MIN_LEAD,
                          "prereg": "P1 arm B test split: probe confidence beats the S1-level control (lead >= 0.001, one-sided per-tile sign test); "
                                    "P2 arm C vs B on Bolivia: v1.2's probe confidence has higher pooled excess AURC than v1's by >= 0.001 with a one-sided per-tile sign test (v1 better)"},
               "results": {}, "failures": []}
    rows = []
    floods_dir = args.floods_dir or os.path.join(hb.ROOT, "data", "floods")
    cache_dir = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "paper_embeddings")
    n_smoke = args.smoke_tiles if args.smoke else None

    # ---- our tiles (arms B, C; controls and matching for A)
    ours = {}
    for split in (["valid", "bolivia"] if args.smoke else ["train", "valid", "bolivia", "test"]):
        try:
            ours[split] = load_ours(floods_dir, split, n_smoke, seed={"valid": 0, "bolivia": 1, "test": 2, "train": 3}[split])
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": f"load {split}", "error": repr(ex)}); print(f"load {split} FAILED: {ex!r}", flush=True)
    tr_key = "valid" if args.smoke else "train"
    grade_splits = ["bolivia"] if args.smoke else ["bolivia", "test"]
    med_vv = float(np.nanmedian(np.nan_to_num(ours[tr_key][0][:, 0, :CROP, :CROP])))

    # ---- arms B and C: our S1 encode, their probe
    versions = {"v1": None} if args.smoke else {"v1": None, "v1_2": None}
    try:
        from olmoearth_pretrain.model_loader import ModelID
        has_v12 = hasattr(ModelID, "OLMOEARTH_V1_2_BASE")
    except Exception:
        has_v12 = False
    if not has_v12 and not args.smoke:
        raise SystemExit("OLMOEARTH_V1_2_BASE is unavailable: run with ~/oe12/.venv; arm C is preregistered.")
    conf_by_version = {}
    for version in list(versions):
        t0 = time.time()
        try:
            if version == "v1":
                model = hb.load_model()
            else:
                from olmoearth_pretrain.model_loader import load_model_from_id
                model = load_model_from_id(ModelID.OLMOEARTH_V1_2_BASE).to(DEV).eval().float()
            emb = {k: torch.tensor(e46.embed_s1(model, ours[k][0])) for k in ours}      # (N, G, G, D) at shift 0, 60-px crop
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()
            probe = train_probe(emb[tr_key], torch.tensor(ours[tr_key][2][:, :CROP, :CROP]), PATCH)
            res = {"encode_and_train_seconds": time.time() - t0, "train_tiles": int(len(emb[tr_key]))}
            for split in grade_splits:
                s1, s2, lab = ours[split]
                p = predict_pixels(probe, emb[split], PATCH, (CROP, CROP))
                wp, y, ok, err, conf = window_view(p, lab[:, :CROP, :CROP], PATCH)
                pix_ok = lab[:, :CROP, :CROP] >= 0
                sig = {CONF: conf, CTRL_S1: s1_level(s1, med_vv, PATCH, CROP),
                       CTRL_NDWI: np.stack([ndwi_level(t, patch=PATCH, size=CROP) for t in s2])}
                r = score(sig, err, ok, CONF, [CTRL_S1])
                r["pixel_accuracy"] = float(((p > 0.5) == (lab[:, :CROP, :CROP] == 1))[pix_ok].mean())
                r["pixel_miou"], _ = segmentation_miou((p > 0.5).astype(np.int64), lab[:, :CROP, :CROP])
                r["err"] = err; r["ok"] = ok                                    # kept in memory for the cross-arm tests
                res[split] = r
                conf_by_version.setdefault(version, {})[split] = r
                print(f"{version}/{split} (our S1 encode, their probe): windows {r['n_windows']}, errors {r['n_errors']}, window acc {r['accuracy']:.4f}, "
                      f"pixel acc {r['pixel_accuracy']:.4f}, mIoU {r['pixel_miou']:.3f} | pooled E-AURC " + ", ".join(f"{k} {v:.4f}" for k, v in r["pooled_eaurc"].items()) +
                      f" | lead over S1 level {r['tests'][CTRL_S1]['pooled_lead']:+.4f} (p={r['tests'][CTRL_S1]['sign_p']:.2g})", flush=True)
                rows.append({"arm": f"{version} our S1 encode", "split": split, "window_acc": r["accuracy"], "pixel_acc": r["pixel_accuracy"], "miou": r["pixel_miou"],
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()}})
            summary["results"][f"B_{version}" if version == "v1" else "C_v1_2"] = {k: v for k, v in res.items()}
            checkpoint(summary, suffix)
        except Exception as ex:  # noqa: BLE001
            summary["failures"].append({"part": f"arm {version}", "error": repr(ex), "traceback": traceback.format_exc()})
            print(f"arm {version} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)

    # P1 / P2
    try:
        b = summary["results"].get("B_v1", {})
        if "test" in b:
            t = b["test"]["tests"][CTRL_S1]
            summary["P1"] = bool(t["pooled_lead"] >= MIN_LEAD and t["sign_p"] < 0.05)
        c = summary["results"].get("C_v1_2", {})
        if "bolivia" in b and "bolivia" in c:
            rb, rc = b["bolivia"], c["bolivia"]
            common = sorted(set(rb["tiles"]) & set(rc["tiles"]))
            ib, ic = {t: i for i, t in enumerate(rb["tiles"])}, {t: i for i, t in enumerate(rc["tiles"])}
            g = np.array([rc["per_tile_eaurc_ref"][ic[t]] - rb["per_tile_eaurc_ref"][ib[t]] for t in common])   # positive: v1 better
            w, l, t_ = wins_losses_ties(g)
            lead = rc["pooled_eaurc"][CONF] - rb["pooled_eaurc"][CONF]
            summary["P2_test"] = {"pooled_eaurc_v1": rb["pooled_eaurc"][CONF], "pooled_eaurc_v1_2": rc["pooled_eaurc"][CONF], "lead_v1_2_minus_v1": lead,
                                  "w_v1_better": w, "l": l, "t": t_, "sign_p": sign_test(w, l, "greater"), "n_tiles": len(common),
                                  "phi_errors_v1_vs_v1_2": phi(rb["err"][rb["ok"]], rc["err"][rc["ok"]])}
            summary["P2"] = bool(lead >= MIN_LEAD and summary["P2_test"]["sign_p"] < 0.05)
    except Exception as ex:  # noqa: BLE001
        summary["failures"].append({"part": "prereg", "error": repr(ex), "traceback": traceback.format_exc()})

    # ---- arm A and D: their embeddings, their probe (full run only)
    if not args.smoke:
        their_err = {}
        for model in [THEIRS] + list(args.others):
            t0 = time.time()
            try:
                emb_tr, lab_tr = load_theirs(model, "train", cache_dir)
                emb_te, lab_te = load_theirs(model, "test", cache_dir)
                pp = lab_tr.shape[-1] // emb_tr.shape[1]
                probe = train_probe(emb_tr, lab_tr, pp)
                p = predict_pixels(probe, emb_te, pp, tuple(lab_te.shape[-2:]))
                lab_np = lab_te.numpy()
                wp, y, ok, err, conf = window_view(p, lab_np, PATCH)                  # 4-px windows for every model, whatever its token grid
                miou, _ = segmentation_miou((p > 0.5).astype(np.int64), lab_np)
                r = {"seconds": time.time() - t0, "train_tiles": int(len(emb_tr)), "test_tiles": int(len(emb_te)), "token_grid": list(emb_tr.shape[1:3]), "pixels_per_token_side": int(pp),
                     "pixel_miou": miou, "pixel_accuracy": float(((p > 0.5) == (lab_np == 1))[lab_np >= 0].mean())}
                sig = {CONF: conf}
                if model == THEIRS and "test" in ours:
                    # match their test tiles to ours by the label tile, then add the controls and our S2 head's errors
                    theirs_k, ours_k = label_keys(lab_np), label_keys(ours["test"][2])
                    pos = {k: i for i, k in enumerate(ours_k)}
                    match = np.array([pos.get(k, -1) for k in theirs_k])
                    r["matched_tiles"] = int((match >= 0).sum())
                    if (match >= 0).sum() >= 50:
                        mi = np.where(match >= 0)[0]
                        s1m, s2m = ours["test"][0][match[mi]], ours["test"][1][match[mi]]
                        sig_m = {CONF: conf[mi], CTRL_S1: s1_level(s1m, med_vv, PATCH, lab_np.shape[-1]),
                                 CTRL_NDWI: np.stack([ndwi_level(t, patch=PATCH, size=lab_np.shape[-1]) for t in s2m])}
                        r["matched"] = score(sig_m, err[mi], ok[mi], CONF, [CTRL_S1, CTRL_NDWI])
                        for k in ("per_tile_eaurc_ref", "tiles"):
                            r["matched"].pop(k)
                        r["matched"]["phi_vs_arm_B_v1"] = None
                        if "B_v1" in summary["results"] and "test" in summary["results"]["B_v1"]:
                            rb = summary["results"]["B_v1"]["test"]
                            # arm B graded the 60-px crop (15 x 15 windows); compare on the shared 15 x 15 block of matched tiles
                            eb, ob = rb["err"][match[mi]][:, :G, :G], rb["ok"][match[mi]][:, :G, :G]
                            ea, oa = err[mi][:, :G, :G], ok[mi][:, :G, :G]
                            both = ob & oa
                            r["matched"]["phi_vs_arm_B_v1"] = phi(ea[both], eb[both])
                            r["matched"]["acc_arm_A_on_shared"] = float(1 - ea[both].mean()); r["matched"]["acc_arm_B_on_shared"] = float(1 - eb[both].mean())
                r_score = score(sig, err, ok, CONF, [])
                r["window_accuracy"] = r_score["accuracy"]; r["pooled_eaurc"] = r_score["pooled_eaurc"]; r["n_windows"] = r_score["n_windows"]; r["n_errors"] = r_score["n_errors"]
                their_err[model] = (err, ok, theirs_k if model == THEIRS else label_keys(lab_np))
                summary["results"][f"A_{model}" if model == THEIRS else f"D_{model}"] = r
                checkpoint(summary, suffix)
                print(f"{model} (their embeddings, their probe): train {r['train_tiles']} test {r['test_tiles']} tiles, grid {r['token_grid']} | pixel mIoU {r['pixel_miou']:.3f} "
                      f"acc {r['pixel_accuracy']:.4f} | window acc {r['window_accuracy']:.4f}, E-AURC confidence {r['pooled_eaurc'][CONF]:.4f}" +
                      (f" | matched {r.get('matched_tiles')} tiles: " + ", ".join(f"{k} {v:.4f}" for k, v in r["matched"]["pooled_eaurc"].items()) + f", phi vs our encode {r['matched']['phi_vs_arm_B_v1']}" if "matched" in r else ""), flush=True)
                rows.append({"arm": f"their embeddings {model}", "split": "test", "window_acc": r["window_accuracy"], "pixel_acc": r["pixel_accuracy"], "miou": r["pixel_miou"],
                             **{f"eaurc {k}": v for k, v in r["pooled_eaurc"].items()}})
                del emb_tr, emb_te
            except Exception as ex:  # noqa: BLE001
                summary["failures"].append({"part": f"their embeddings {model}", "error": repr(ex), "traceback": traceback.format_exc()})
                print(f"{model} FAILED: {ex!r}\n{traceback.format_exc()}", flush=True)
        # phi of window errors against OlmoEarth Base on their tiles (rows aligned by label tile)
        if THEIRS in their_err:
            e0, o0, k0 = their_err[THEIRS]
            pos0 = {k: i for i, k in enumerate(k0)}
            summary["phi_vs_olmoearth_base"] = {}
            for model, (e, o, k) in their_err.items():
                if model == THEIRS:
                    continue
                m = np.array([pos0.get(x, -1) for x in k]); mi = np.where(m >= 0)[0]
                both = o[mi] & o0[m[mi]]
                summary["phi_vs_olmoearth_base"][model] = {"phi": phi(e[mi][both], e0[m[mi]][both]), "aligned_tiles": int(len(mi)),
                                                           "p_base_wrong_given_model_wrong": float(e0[m[mi]][both][e[mi][both] > 0].mean()) if (e[mi][both] > 0).any() else None}
            print("phi vs olmoearth_base: " + ", ".join(f"{m} {v['phi']:.3f}" for m, v in summary["phi_vs_olmoearth_base"].items()), flush=True)

    for k in list(summary["results"]):
        for split in ("bolivia", "test"):
            if isinstance(summary["results"][k], dict) and split in summary["results"][k] and isinstance(summary["results"][k][split], dict):
                for drop in ("err", "ok", "per_tile_eaurc_ref", "tiles"):
                    summary["results"][k][split].pop(drop, None)
    summary["prereg"] = {"P1": summary.get("P1"), "P2": summary.get("P2"),
                         "supported": bool(summary.get("P1") and summary.get("P2")) if (summary.get("P1") is not None and summary.get("P2") is not None) else None,
                         "complete": not summary["failures"]}
    summary["n_failures"] = len(summary["failures"])
    os.makedirs(hb.OUT, exist_ok=True)
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda s: (s not in ("arm", "split"), s))
        with open(os.path.join(hb.OUT, f"exp51_their_probe{suffix}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(hb.OUT, f"exp51_summary{suffix}.json"), "w") as f:
        json.dump(hb.json_ready(summary), f, indent=1)
    print(f"wrote exp51_summary{suffix}.json; prereg {summary['prereg']}; failures {summary['n_failures']}", flush=True)


if __name__ == "__main__":
    main()
