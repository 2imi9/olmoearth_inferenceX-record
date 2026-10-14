# exp98 configs: what differs from Ai2's AWF deployment

Base: `allenai/olmoearth_projects` at `f3c9b0c89c7670b3525647dda4c14b76245d682d`, directory `olmoearth_run_data/awf/`.
Line numbers below are rslearn v0.0.27 (`09564335`), the version `olmoearth-runner==0.1.14` pins, unless stated.

| File (Ai2's sha256) | Here | Change |
|---|---|---|
| `model.yaml` (`08e04320...a7ea`) | `model.yaml` | two keys added, listed below; nothing else |
| `dataset.json` (`7da20e92...1b4b`) | `dataset.json` | the `output` layer's band set: bands `["output"]` become `["p0", ..., "p9"]`, dtype float32 kept |
| `olmoearth_run.yaml` (`1e1c8f70...05ba`) | not copied | unchanged; the jobs take it from the pinned clone and check its sha256 |
| `prediction_request_geometry.geojson` (`fee3ce01...3784`) | not copied | unchanged (the AWF project's own request geometry, 2023-01-01 to 2023-12-31); taken from the pinned clone and checked by sha256, so no geometry is stored in this repository |
| none | `scl_dataset.json` | new: the config of a sidecar rslearn dataset that reads each window's Sentinel-2 SCL band from the same scenes the model read |

`e98_prepare.sh` refuses to run unless the parsed `model.yaml` differs from Ai2's by exactly the two additions and the
parsed `dataset.json` by exactly the output bands.

## model.yaml

```diff
             init_args:
               num_classes: 10
               zero_is_invalid: false
               nodata_value: 9
+              output_probs: true
...
         merger:
           class_path: rslearn.train.prediction_writer.RasterMerger
           init_args:
             padding: 2
+        layer_config:
+          type: RASTER
+          band_sets:
+            - bands: ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]
+              dtype: FLOAT32
```

1. `output_probs: true`. Without it `SegmentationTask.process_output` returns one band, the argmax over the 10 channels
   (`rslearn/train/tasks/segmentation.py:187-188`); with it, the CHW softmax from `SegmentationHead`
   (`segmentation.py:343`) is returned unchanged (`segmentation.py:44, 79-80, 107, 179-185`). `prob_scales` is left
   unset (`segmentation.py:171-177`), so the 10 bands sum to 1. Channel 9 is AWF's nodata class, which no label
   trains; it is written like the others. rslearn 0.0.23, which Ai2's `uv.lock` resolves through `olmoearth-runner`
   0.1.12, has no `output_probs` (it entered between v0.0.23 and v0.0.27, commit `3eaaeaa9`).
2. `layer_config` on the `RslearnWriter`. The writer otherwise reads the output layer's config from the dataset's
   `config.json` (`rslearn/train/prediction_writer.py:292-305`); given here, it is used instead (`:195, 209-211,
   225-229`). The merged raster takes `band_sets[0].dtype` (`:130-137`): float32, as Ai2's layer already is, since an
   integer type would truncate every probability. The band-set directory is the names joined by `_`
   (`:424-426`, `rslearn/utils/raster_format.py:25-40`), so names carry no `_` and the GeoTIFF lands at
   `layers/output/p0_p1_p2_p3_p4_p5_p6_p7_p8_p9/geotiff.tif`. In YAML jsonargparse takes the enum names (`RASTER`,
   `FLOAT32`), not the values `config.json` uses; Ai2 writes it that way in `allenai/rslearn_projects`
   `data/landslide/model.yaml:119-124` (`199cfcef`), and `e98_env.sh` parses and exercises this callback with the
   deployment's own packages before any real run.

The model's input is untouched: `sentinel2_l2a` reads the 12 named bands of layer `sentinel2` with all item groups
(Ai2's `model.yaml:36-43`), and `read_raster_layer_for_data_input` reads only the band sets that hold a named band
(`rslearn/train/dataset.py:243-270`).

## dataset.json

The output layer becomes 10 float32 bands `p0` to `p9`, the same band set as the writer's `layer_config`, so the
dataset's `config.json` (which `olmoearth_run` is expected to write from this file; checked by `e98_prepare.sh`) agrees
with what the writer writes, whichever of the two the writer reads. The `sentinel2` input layer is byte-for-byte Ai2's
(after JSON parsing): no SCL band set is added to it. Why not:

- rslearn 0.0.27's Planetary Computer `Sentinel2` source has no SCL asset: its `BANDS` lists B01 to B12, B8A and
  `visual` only (`rslearn/data_sources/planetary_computer.py:312-326`), and it selects assets by intersecting each
  band set with `BANDS` (`:347-357`), so an `SCL` band set would have no asset to read. SCL entered later, with
  `NON_REFLECTANCE_ASSETS` so that harmonization skips it (v0.1.17 `planetary_computer.py:323-326, 378-380`). In 0.0.27
  `harmonize: true` would also apply the reflectance offset callback to any non-`visual` asset (`:421-426, 462-467`).
- A band set on the model's input layer would share the layer's `resampling_method` (bilinear by default,
  `rslearn/config/dataset.py:490-493`, used for every band set at `rslearn/dataset/materialize.py:481`), which
  interpolates class codes; changing it to nearest would change how the 20 m and 60 m bands reach the model.

## scl_dataset.json (new): the SCL sidecar

`e98_prepare.sh` builds, after `build_dataset`, one sidecar rslearn dataset per dataset root of the run, under
`deploy/awf_scl/`. Each window of the run that holds a `sentinel2` layer gets a sidecar window with the same
`metadata.json` (projection and pixel bounds) and an `items.json` whose `sentinel2_scl` entry is the run's `sentinel2`
item groups, copied as they are (`rslearn/dataset/storage/file.py:133-156` for the format). Then
`rslearn dataset materialize` writes `layers/sentinel2_scl[.N]/SCL/geotiff.tif` per item group. So:

- **Same scenes, same order.** Materialize reads the item groups from `items.json` (`rslearn/dataset/manage.py:372-391`)
  and composites each group first-valid in item order (`materialize.py:159-225`), one band at a time where the band is
  still nodata (`materialize.py:106-112`). Group N of the sidecar is group N of the model's input, and a pixel's SCL
  comes from the first scene in the group that covers it, as its reflectance does.
- **Nearest neighbour, on the window's 10 m grid.** The layer's `resampling_method` is `nearest`; every read goes
  through rasterio's WarpedVRT with that method (`rslearn/data_sources/direct_materialize_data_source.py:194-212`).
- **No harmonization.** The generic `PlanetaryComputer` source (`planetary_computer.py:126-205`) has no harmonize
  callback (`direct_materialize_data_source.py:96-114` returns none).
- **The asset is there.** STAC items keep every asset's URL (`rslearn/data_sources/stac.py:123-126`), and the
  sidecar re-reads each item by name with its own cache (`stac.py:155-191`).
- **Nodata.** No `nodata_vals`, so 0 (SCL's NO_DATA class) marks a pixel no scene covered (`materialize.py:469-471`).

`query_config` repeats Ai2's for the record only: the sidecar never runs `prepare`.
