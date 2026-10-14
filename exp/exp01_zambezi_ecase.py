"""Exp 01: first embedding-level E_case map, Zambezi at Kazungula (Zambia-Botswana border).

One Sentinel-2 L2A scene from Planetary Computer, Nano + Base embeddings,
per-patch local similarity-structure agreement between the two models.
Low agreement = the two models represent that patch's relationship to its
neighborhood differently = candidate suspect window.

CPU only. No Studio, no labels, no GPU.
"""
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
import torch
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

# Zambezi at Kazungula, Zambia-Botswana border.
LON, LAT = 25.263, -17.788  # Kazungula: Zambezi ~400 m wide
SIZE = 128  # pixels at 10 m
PATCH = 4


def fetch_scene():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [LON, LAT]},
        datetime="2024-06-01/2024-09-30",
        query={"eo:cloud_cover": {"lt": 5}},
    )
    items = sorted(search.items(), key=lambda i: i.properties["eo:cloud_cover"])
    item = items[0]
    print(f"scene: {item.id}  cloud={item.properties['eo:cloud_cover']:.2f}%  {item.datetime.date()}")

    band_order = Modality.SENTINEL2_L2A.band_order
    print(f"band order: {band_order}")

    # Reference grid from the first band (10 m).
    ref_href = item.assets[band_order[0]].href
    with rasterio.open(ref_href) as src:
        crs, transform = src.crs, src.transform
        # Pixel coords of AOI center on the 10 m grid.
        xs, ys = rasterio.warp.transform("EPSG:4326", crs, [LON], [LAT])
        row, col = src.index(xs[0], ys[0])
    r0, c0 = row - SIZE // 2, col - SIZE // 2
    win_transform = transform * rasterio.Affine.translation(c0, r0)

    image = np.zeros((len(band_order), SIZE, SIZE), dtype=np.int32)
    for bi, band in enumerate(band_order):
        with rasterio.open(item.assets[band].href) as src:
            with WarpedVRT(src, crs=crs, transform=win_transform,
                           width=SIZE, height=SIZE,
                           resampling=Resampling.bilinear) as vrt:
                image[bi] = vrt.read(1)
    dt = item.datetime
    return image, (dt.day, dt.month - 1, dt.year)


def embed(model_id, sample):
    model = load_model_from_id(model_id)
    model.eval()
    with torch.no_grad():
        out = model.encoder(sample, fast_pass=True, patch_size=PATCH)
    return out["tokens_and_masks"].sentinel2_l2a.mean(dim=[3, 4]).squeeze(0)  # (H', W', D)


def local_agreement(fa, fb, radius=4):
    """Per-patch correlation of local cosine-similarity structure between models."""
    h, w, _ = fa.shape
    na = torch.nn.functional.normalize(fa, dim=-1)
    nb = torch.nn.functional.normalize(fb, dim=-1)
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            i0, i1 = max(0, i - radius), min(h, i + radius + 1)
            j0, j1 = max(0, j - radius), min(w, j + radius + 1)
            va = (na[i0:i1, j0:j1] @ na[i, j]).flatten()
            vb = (nb[i0:i1, j0:j1] @ nb[i, j]).flatten()
            out[i, j] = torch.corrcoef(torch.stack([va, vb]))[0, 1].item()
    return out


def main():
    image, (day, month0, year) = fetch_scene()
    x = image.transpose(1, 2, 0)[None, :, :, None, :]
    normalizer = Normalizer(Strategy.COMPUTED)
    x = normalizer.normalize(Modality.SENTINEL2_L2A, x)

    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.tensor(x, dtype=torch.float32),
        sentinel2_l2a_mask=torch.ones((1, SIZE, SIZE, 1, 3)) * MaskValue.ONLINE_ENCODER.value,
        timestamps=torch.tensor([day, month0, year])[None, None, :],
    )

    feats = {m: embed(m, sample) for m in (ModelID.OLMOEARTH_V1_NANO, ModelID.OLMOEARTH_V1_BASE)}
    for m, f in feats.items():
        print(f"{m.value}: {tuple(f.shape)}")

    agree = local_agreement(*feats.values())
    print(f"agreement: mean={agree.mean():.3f} min={agree.min():.3f} "
          f"p5={np.percentile(agree, 5):.3f} max={agree.max():.3f}")

    # Quicklook: RGB (B04,B03,B02 are indices per band_order) + agreement map.
    from oe_inferencex.figstyle import setup, map_panel
    import matplotlib.pyplot as plt
    setup()
    band_order = Modality.SENTINEL2_L2A.band_order
    rgb_idx = [band_order.index(b) for b in ("B04", "B03", "B02")]
    rgb = image[rgb_idx].transpose(1, 2, 0).astype(np.float32)
    rgb = np.clip((rgb - 1000) / 2000, 0, 1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    map_panel(fig, axes[0], rgb, "Sentinel-2 RGB, Zambezi at Kazungula", None, rgb=True, idx=0)
    map_panel(fig, axes[1], agree,
              "Nano-Base local similarity-structure\ncorrelation (radius-4 neighborhood)",
              "Pearson r of local cosine-similarity\nvectors (noise floor: 0.59)",
              cmap="RdYlGn", idx=1, vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig("exp/out/exp01_zambezi_agreement.png", bbox_inches="tight")
    np.save("exp/out/exp01_agreement.npy", agree)
    print("wrote exp/out/exp01_zambezi_agreement.png")


if __name__ == "__main__":
    main()
