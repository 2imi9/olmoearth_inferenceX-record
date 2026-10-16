# Established facts

Non-signal findings: upstream sources, export formats, transfer
properties. Index at [TECHNIQUES.md](../TECHNIQUES.md).

### Heads and spatial transfer
- Linear heads on frozen embeddings transfer spatially: trained on one river
  reach, 97-98% accuracy against WorldCover on a reach ~110 km away.
  (exp/exp02)
- WorldCover-as-reference carries temporal drift (2021 labels vs 2024
  scenes), so a fraction of measured "model error" is label noise.

### Production inference exports (allenai/olmoearth_lcc)
- Source: the
  [olmoearth_lcc dataset](https://huggingface.co/datasets/allenai/olmoearth_lcc)
  backing the [OlmoEarth LCC product](https://olmoearth-lcc.allen.ai).
- The published continent-scale LCC run (encoder OlmoEarth-v1.2-Base,
  32768x32768-pixel UTM tiles) provides 9-band uint8 summary rasters: band 1
  binary-change probability (scaled 0-255), bands 2-5 argmax classes, bands
  6-7 the probability of the argmax class, bands 8-9 month-encoded change
  dates. The export format therefore carries partial probability information
  (top-1 score and one binary-head probability), not full per-class
  distributions. (Documented in the dataset README.)
- The rasters are cloud-optimized GeoTIFFs on public storage; windows can be
  read over HTTP without bulk download.
- Caveat for any validation against the olmoearth_lcc annotations: most
  collection phases used output-based labeling (model proposes candidates,
  annotators verify), so label locations are correlated with model beliefs.

### Infrastructure

The Slurm scripts behind the served-model and multi-encoder runs (exp64, exp72, exp73, exp74) are kept
verbatim under [exp/jobs/](../../exp/jobs/), with a
README that maps each script to its job ids and artifacts.
- Encoders, loader, and eval wrappers come from
  [allenai/olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain);
  dataset tooling and the window format come from
  [allenai/rslearn](https://github.com/allenai/rslearn); per-project task
  configs live in
  [allenai/olmoearth_projects](https://github.com/allenai/olmoearth_projects).
- The olmoearth_pretrain base dependency set suffices for inference (torch,
  einops, huggingface_hub, numpy); CPU handles 128x128-pixel windows at
  patch size 4, and a single consumer GPU (tested: RTX 5090 laptop, torch
  2.7.1+cu128) makes perturbation ensembles and multi-scene sweeps
  interactive. (exp/smoke_test.py, exp/exp08, and the archived exp09 in the
  [v1.9.0 tree](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/exp/exp09_multiscene.py))
- All encoder checkpoints are public: v1 Nano/Tiny/Base/Large, v1_1
  Nano/Tiny/Base, v1_2 Nano/Tiny/Small/Base, plus fine-tuned variants (AWF,
  LFMC, Mangrove, ForestLossDriver, EcosystemTypeMapping). Multi-model and
  cross-version signals require no private infrastructure.
- Planetary Computer Sentinel-2 L2A and ESA WorldCover can be read onto a
  shared 10 m grid; OSM Overpass requires mirror fallback.

### Encoder internals (verified empirically, exp17 preparation)
- OlmoEarth v1-Base encoder: 12 pre-norm ViT blocks (norm1, attention,
  layer-scale, norm2, MLP, layer-scale), 768-d, 12 heads, no register
  tokens, no qk-norm; attention runs through PyTorch scaled-dot-product
  attention, so weights are not returned but the q and k projections are
  plain linear layers whose outputs can be hooked and the weights recomputed.
- The token sequence entering the blocks is laid out H, W, T, S
  (outer to inner), where S indexes the three Sentinel-2 band-set tokens per
  patch (10 m bands B02/B03/B04/B08; 20 m bands B05/B06/B07/B8A/B11/B12;
  60 m bands B01/B09). Verified to zero error against the official output
  by reshaping a hooked last-block output through the final LayerNorm.
  Every earlier experiment mean-pooled these three tokens; they are
  separately addressable.
- Per-block outputs are hookable with standard forward hooks under
  fast_pass=True (no masked-token removal, no packing), which is what makes
  layer-wise probes, logit-lens trajectories and attention statistics
  available from a single forward pass.

### Findings from the olmoearth_pretrain repository (2026-09-01 dig)
- Dense expert-labelled water truth exists in the evaluation set:
  Sen1Floods11 (hand-labelled flood masks, LabelHand: -1 no data, 0
  non-water, 1 water) processed into 64x64 tiles and mirrored in the public
  bucket gs://ai2-olmoearth-projects-public-data/research_benchmarks/floods
  (flood_{train,valid,test,bolivia}_data.pt; 13-band Sentinel-2 Level-1C
  chips, int16, plus Sentinel-1). Bolivia (441 tiles) is a geographically
  held-out region. The upstream benchmark uses the Sentinel-1 channel only.
  The v1 sample type carries only sentinel2_l2a, so Level-1C chips must go
  through the L2A path with the L2A normalizer (a documented mismatch).
- OlmoEarth v1.2 uses rotary position encodings: origin/main
  nn/encodings.py defines 2D and 3D RoPE variants (axial and mixed). This
  confirms in source the "RoPE fix in 1.2" statement from the project
  kickoff. The local checkout used for exp01-exp17 is 411 commits behind
  origin/main (April vs August 2026); v1.2 checkpoints load only with the
  newer code. To avoid disturbing the environment those results depend on,
  origin/main is checked out as a git worktree (../olmoearth_pretrain_main)
  with its own virtual environment (.venv-main).
- OlmoEarth v1.2-Base (loaded on current main, exp19): 12 blocks, 768-d,
  position encoding rope_3d_mixed, and a single Sentinel-2 band-set token
  per patch (v1 has three), so the exp17 band-set signal has no v1.2
  counterpart. v1 features recomputed with the new code match the cached
  ones exactly.
- Other evaluation sets wrapped in olmoearth_pretrain/evals with dense
  labels: MADOS (marine debris, 15 classes), PASTIS-R (crop segmentation,
  19 classes), GeoBench m-cashew-plant and m-sa-crop-type. Baseline model
  wrappers in evals/models include Galileo, Satlas, TerraMind, Prithvi v2,
  Panopticon, CROMA and AnySat: candidate out-of-family raters.
- olmoearth_pretrain ships an MCP server (olmoearth_pretrain/mcp) exposing
  model loading, config analysis, modality description and inference-code
  generation as tools; relevant to agent integration, not used here.
- Environment note: torch is index-pinned in this repository's
  `pyproject.toml` (previously inherited from the sibling
  olmoearth_pretrain checkout) — Linux resolves the cu128 build, every other
  platform the CPU build. On Windows and macOS a GPU run therefore still
  needs the cu128 wheel installed over the top, and `uv run` will revert it,
  so invoke the venv's Python directly after doing so.

### Task cards (oe_inferencex/taskcard.py)
- A Layer-2 resolver that reads the authoritative configuration sources and
  returns one structured card per model: the encoder's HuggingFace
  config.json (depth, width, heads, register tokens, position encoding,
  Sentinel-2 band groups per patch), the project's model.yaml and
  olmoearth_run.yaml in allenai/olmoearth_projects (task type, class
  legend from per-class metric definitions or explicit class lists, nodata
  value, inputs and timesteps, window size and resolution, split protocol),
  the project docs (stated goal), and the olmoearth_lcc README (export band
  table and class legends). Audit settings are derived from the card:
  whether outputs are dense, class count, whether band-set disagreement
  exists for the encoder version, and how to score confidence.
- Resolved for both encoders, eleven projects (twelve models: kenya_lulc_croptype
  publishes a cropland and a maize model) and the LCC product
  (docs/method/taskcards.md, exp/out/taskcards.json). Fleet-level facts that
  fell out: eight of the twelve models use 63-px windows at 10 m, seven of them
  with a 128-px spatial splitter grid; they stack 4 to 12 Sentinel-2 time steps
  (12 for AWF, Nandi, mangrove and LFMC; 8 for Mozambique, Togo, Fields of the
  World and forest loss), read from each project's dataset.json; class counts
  range from 2 to 110 (ecosystem_type_mapping, whose legend names 60); nodata
  conventions differ per project (9, 10, 54, 3, -1); v1.2 tokenizes all twelve
  Sentinel-2 bands as one group where v1 uses three.
- Corrected 2026-09-22 after the module's first audit: the cards had counted
  layers as time steps (so a 12-month model read as one), taken the class
  legend only from metric names, reported a partition grid in degrees as the
  split grid, skipped inputs written as DataInput specs, called mangrove's
  pooled output dense, missed kenya_lulc_croptype's two models, and this
  paragraph had said v1.2 drops B01/B09.

## Served production rasters (exp20)

- allenai/olmoearth_lcc model_outputs: one file per 32768-px UTM tile, named
  EPSG:<code>_<col>_<row>.tif, but served in EPSG:3857 at about 9.55 m (the
  Kazungula tile is 36588 x 36794 px, 1.2 GB); BigTIFF, 256-px tiles,
  deflate, pixel-interleaved, 9 bands uint8. Legends and band semantics are
  in the dataset card and mirrored in oe_inferencex/lcc.py.
- HuggingFace serves the file through a signed CDN redirect (xet bridge)
  whose URL has no .tif suffix; GDAL's vsicurl stalls on it. The reader in
  oe_inferencex/lcc.py parses the first IFD, fetches the tile tables, and
  range-requests only the tiles that cover a window: a 128-px window in
  under a second, a 512-px window in about three seconds.
- ESA WorldCover 2021 is warped onto the same 3857 window grid with a
  WarpedVRT (oe_inferencex/data.py); water agreement peaks at zero shift
  (IoU 0.905 at Kazungula), so the two grids are aligned.
- The olmoearth_lcc dataset also carries annotated change points
  (pre_category, post_category and fine-grained change categories). They
  are the LCC model's training labels (dataset card), collected largely by
  output-based labelling of the model's own predictions, so they are not a
  held-out reference for the served product and are not used here.

## Fine-tuned checkpoints (exp21)

- Fine-tuned models on HuggingFace: allenai/OlmoEarth-v1-FT-
  {AWF, Mangrove, ForestLossDriver, LFMC, EcosystemTypeMapping}-Base, each a
  Lightning checkpoint (model.ckpt) trained by rslearn from the project's
  model.yaml in olmoearth_projects. The AWF one holds state-dict keys
  model.encoder.0.model.* (the pretrain encoder, 231 tensors, 203 changed by
  fine-tuning; the backbone was unfrozen at epoch 20 of 55) and
  model.decoders.segment.1.layer.* (a 1x1 convolution, 768 to 10 classes).
- Replica without rslearn: load the encoder keys strictly into
  olmoearth_pretrain's v1-Base encoder; mean-pool tokens over timesteps and
  band sets (rslearn's OlmoEarth wrapper, token_pooling=True); legacy
  timestamps (day 1, month index 0-11, year 2024; only the month is used);
  bilinear x4 upsampling (torch.nn.Upsample default, align_corners=False)
  then the 1x1 convolution; softmax. The training pipeline pads to 31 px and
  crops 16 px around the point, so 16-px crops are the training regime.
- The replica reaches 0.881 on the validation split against the reported
  0.895; the remaining gap is within crop-alignment and interpolation
  choices (0.898 when the containing patch's logits are read directly).
- Library version matters (exp111): Ai2's FT-AWF v1 checkpoint does not
  reproduce its rslearn 0.0.27 predictions (exp98) under rslearn 0.1.17
  (olmoearth-pretrain-minimal 0.1.2, torch 2.14.1): 245 of exp98's 259
  points keep their class. The stored temporal table, the encodings'
  precision and autocast are ruled out; the cause is not isolated. The
  record's v1.2 fine-tune, whose encoder does not use that table, repeats
  exp103 on the same route (259 of 259).
