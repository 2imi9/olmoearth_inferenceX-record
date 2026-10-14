"""Encoder throughput and numerical drift under fp32 / TF32 / bf16 / compile.

Answers "is the encoder using the GPU well?" for OlmoEarth-v1-Base at the
shapes this repository runs: a 128x128 single-date scene (3072 tokens, the
river-scene regime), the same batched 16-wide, and 32-wide batches of
32x32 twelve-month stacks (the AWF regime). Each setting reports ms per
window and, against the fp32 run, the drift of the pooled features and the
Spearman correlation of a random linear head's scores, since ranking is
what the audit consumes. It then forces the flash SDPA kernel per dtype to
show which one can run, and profiles the top CUDA kernels of one forward.

Random input; no data, no network beyond the checkpoint. Run inside a GPU
job:  .venv/bin/python exp/bench_encoder_b200.py
Writes exp/out/bench_encoder_b200.json.

First run 2026-09-05 on AICR (job 697668, one B200, torch 2.7.1+cu128): in
fp32 attention executes on an sm80 memory-efficient kernel and the GEMMs on
SIMT sgemm, so the tensor cores are idle; TF32 halves the time with head
score Spearman 1.000000; bf16 autocast is 7-10x faster on the flash kernel
with feature drift of 2e-3 to 6e-3 relative; torch.compile adds 10-25%.
"""
import contextlib
import json
import os
import time

import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DEV = "cuda"
SHAPES = {
    "scene 128x128 T=1, B=1": (1, 128, 128, 1),
    "scene 128x128 T=1, B=16": (16, 128, 128, 1),
    "AWF 32x32 T=12, B=32": (32, 32, 32, 12),
}


def make(B, H, W, T):
    x = torch.randn(B, H, W, T, 12, device=DEV)
    m = torch.ones(B, H, W, T, 3, device=DEV) * MaskValue.ONLINE_ENCODER.value
    ts = torch.tensor([[[15, mo % 12, 2024] for mo in range(T)]] * B, device=DEV)
    return MaskedOlmoEarthSample(sentinel2_l2a=x, sentinel2_l2a_mask=m, timestamps=ts)


@torch.no_grad()
def fwd(enc, s):
    return enc(s, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a


def timed(enc, s, iters, ctx):
    with ctx:
        for _ in range(3):
            out = fwd(enc, s)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            out = fwd(enc, s)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
    return dt, out.float().mean(dim=[3, 4])  # pooled (B, H', W', D), as data.embed does


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm() + 1e-12))


def precision(mode):
    torch.set_float32_matmul_precision(mode)
    torch.backends.cudnn.allow_tf32 = mode != "highest"


def main():
    torch.manual_seed(0)
    print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| sm", torch.cuda.get_device_capability(0))
    model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(DEV).eval()
    enc = model.encoder
    bf16 = torch.autocast("cuda", dtype=torch.bfloat16)
    configs = [
        ("fp32 (repo default)", "highest", contextlib.nullcontext(), enc),
        ("tf32", "high", contextlib.nullcontext(), enc),
        ("bf16 autocast", "highest", bf16, enc),
    ]
    try:
        configs.append(("bf16 + torch.compile", "highest", bf16, torch.compile(enc, dynamic=False)))
    except Exception as e:  # noqa: BLE001
        print("torch.compile unavailable:", repr(e)[:120])

    results = {}
    for sname, (B, H, W, T) in SHAPES.items():
        s = make(B, H, W, T)
        iters = 20 if B == 1 else 8
        ref = head = None
        print(f"\n== {sname}  ({H // 4 * W // 4 * T * 3} tokens/window) ==")
        for cname, mode, ctx, e in configs:
            precision(mode)
            try:
                dt, pooled = timed(e, s, iters, ctx)
            except Exception as ex:  # noqa: BLE001
                print(f"  {cname:22s} FAILED: {repr(ex)[:140]}")
                continue
            if ref is None:
                ref, head = pooled, torch.randn(pooled.shape[-1], device=DEV)
            d = (pooled - ref).abs()
            rel = float(d.mean() / ref.abs().mean())
            rho = spearman((ref @ head).flatten(), (pooled @ head).flatten())
            results[f"{sname} | {cname}"] = dict(ms_per_window=dt * 1000 / B, windows_per_s=B / dt,
                                                  max_abs_diff=float(d.max()), rel_mean_diff=rel,
                                                  head_score_spearman=rho)
            print(f"  {cname:22s} {dt * 1000 / B:8.2f} ms/window  {B / dt:8.1f} windows/s"
                  f"   vs fp32: max|d|={float(d.max()):.2e} rel={rel:.2e} head-score spearman={rho:.6f}")
    precision("highest")

    print("\n== which attention kernel can run? (force flash SDPA) ==")
    s = make(1, 128, 128, 1)
    for label, ctx in [("fp32", contextlib.nullcontext()), ("bf16", bf16)]:
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]), ctx:
                fwd(enc, s)
            print(f"  {label}: flash kernel OK")
        except Exception as ex:  # noqa: BLE001
            print(f"  {label}: flash kernel NOT usable -> {repr(ex)[:110]}")

    print("\n== top CUDA kernels, one forward (scene 128x128) ==")
    for label, ctx in [("fp32", contextlib.nullcontext()), ("bf16", bf16)]:
        with ctx:
            fwd(enc, s)
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p, ctx:
            fwd(enc, s)
            torch.cuda.synchronize()
        print(f"  -- {label} --")
        for row in sorted(p.key_averages(), key=lambda r: -r.device_time_total)[:7]:
            print(f"    {row.device_time_total / 1000:8.2f} ms  {row.key[:95]}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "bench_encoder_b200.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nBENCH DONE")


if __name__ == "__main__":
    main()
