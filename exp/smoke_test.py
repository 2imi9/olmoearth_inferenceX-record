"""Smoke test: load OlmoEarth checkpoints, run synthetic inference, compare embeddings.

First sanity check for E_case: verifies that Nano and Base produce
embeddings on the same input and that their per-patch representations differ
measurably.
No real imagery yet. CPU is fine.
"""
import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue


def embed(model_id: ModelID, sample: MaskedOlmoEarthSample) -> torch.Tensor:
    model = load_model_from_id(model_id)
    model.eval()
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=4)
    return out["tokens_and_masks"].sentinel2_l2a


def main() -> None:
    torch.manual_seed(0)
    dummy_image = torch.randn(1, 64, 64, 1, 12)
    dummy_mask = torch.ones(1, 64, 64, 1, 3) * MaskValue.ONLINE_ENCODER.value
    dummy_timestamps = torch.tensor([[[15, 6, 2024]]])
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=dummy_image,
        sentinel2_l2a_mask=dummy_mask,
        timestamps=dummy_timestamps,
    )

    feats = {}
    for mid in (ModelID.OLMOEARTH_V1_NANO, ModelID.OLMOEARTH_V1_BASE):
        feats[mid.value] = embed(mid, sample)
        print(f"{mid.value}: {tuple(feats[mid.value].shape)}")

    # Different widths, so compare structure not values: per-patch L2-normalized
    # self-similarity within each model, then correlate the two similarity maps.
    sims = []
    for name, f in feats.items():
        v = f.flatten(1, -2).squeeze(0)  # (patches, D)
        v = torch.nn.functional.normalize(v.flatten(1), dim=-1)
        sims.append(v @ v.T)
    corr = torch.corrcoef(torch.stack([s.flatten() for s in sims]))[0, 1]
    print(f"cross-model similarity-structure correlation: {corr:.4f}")


if __name__ == "__main__":
    main()

# Result 2026-08-31 (CPU, synthetic noise input):
#   Nano (1,16,16,1,3,128), Base (1,16,16,1,3,768)
#   cross-model similarity-structure correlation: 0.5855
# NOTE: ~0.59 on pure noise = the noise floor for cross-model structural
# agreement (shared patchification/normalization). Real-imagery agreement
# must be measured against this floor, not against zero.
