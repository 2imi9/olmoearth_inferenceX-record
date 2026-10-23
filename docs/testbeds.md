# Test sets

What each function of the package was tested on, and what each test set is. The reference behind a test decides
what the test can show: a probability sample of the map supports an estimate for the map; a benchmark split or a
hand-picked set supports a comparison between methods on that set, not a statement about a map. The per-experiment
record is [Comparisons](results/comparisons.md); every number cited there is a claim in the
[ledger](method/claims.md).

## By function

| Function (command) | Question it answers | Test sets that graded it | Main experiments |
|---|---|---|---|
| Ranking and review set (`assess`) | Which windows should a reviewer check first? | Sen1Floods11; AWF; ESA WorldCover on 27 scenes; the OlmoEarth paper embedding suite (24 tasks, 16 encoders); GEOID-Flood; DFC2020; Dynamic World; LUCAS; EuroCrops; Ai2's fine-tuned models; published products' own confidences (LCMAP, CGLS-LC100, ODSE-LULC) against five probability or validation samples; FT-AWF's deployment maps and the record's v1.2 fine-tune at Ai2's points and the East Africa TimeSync plots, and blind at 890 fresh TimeSync plots; FTW's 2nd-edition field map at the FTW benchmark's test chips of four European countries; a logistic regression on OlmoEarth v1.3 embeddings at LCMAP's reference plots; logistic regressions on three embeddings fitted on other territories, at PLANTEUR's held-out territories | exp04, exp13, exp18, exp55, exp66 to exp70, exp79, exp89, exp92, exp94, exp98, exp99, exp101, exp103, exp104, exp109, exp110, exp111, exp112, exp113 |
| Comparison of two maps without labels (`compare`) | Where do two inferences of one area differ, and why? | Sen1Floods11 (v1 against v1.2, eight encoders, heads); GEOID-Flood (dates and sensors); WorldFloods v2; the embedding suite (encoder pairs on MADOS and PASTIS); DFC2020 (sensor arms); LUCAS; EuroCrops (two years); LCMAP against Esri | exp57 to exp63 (some archived), exp65, exp66, exp68, exp69, exp83, exp92 |
| Error rate from a labelled sample (`sample`, `estimate`) | How wrong is the map, with an interval? | The embedding suite (every unit labelled, so each design is graded exactly); LCMAP's reference sample; Sen1Floods11 Bolivia (with a vision-language reviewer); Ai2's fine-tuned models (report-only); 18 product-years against NLCD's, GFC2020's, East Africa's and S2GLC's reference points; FT-AWF's 2017 map and the v1.2 fine-tune's at the East Africa TimeSync plots, near the AWF area and at fresh plots (report-only); FTW's field map at the FTW benchmark's test chips (every window labelled); the v1.3 embedding map at LCMAP's reference plots; 18 embedding maps at PLANTEUR's crop-type patches (every window labelled) | exp78, exp79, exp81, exp85, exp89, exp91, exp92, exp94, exp99, exp103, exp110, exp111, exp112, exp113 |
| Which of two maps is more accurate (`sample --other`, `estimate`) | Which map is better, from few labels? | Probe maps of 16 encoders (2,514 pairs); LCMAP and Esri against LCMAP's reference sample; eight more pairs of published products (NLCD and LCMAP, GFC2020 and WorldCover, CGLS-LC100 and Esri, ODSE-LULC and Esri); FTW's field map and FTW's PRUE B3 model at the FTW benchmark's test chips; the v1.3 embedding map against LCMAP Collection 1.3 and Esri; maps from OlmoEarth v1.3, TESSERA v2 and AlphaEarth embeddings on PLANTEUR's held-out territories | exp90, exp92, exp94, exp110, exp112, exp113 |
| Certified zone (`certify`) | Which part of the map is wrong at most a stated share, with a guarantee? | The embedding suite (24 tasks, five encoders); Ai2's fine-tuned models (report-only); LCMAP's own confidence; CGLS-LC100's and ODSE-LULC's, and LCMAP's on NLCD's points; FTW's field map at the FTW benchmark's test chips; the v1.3 embedding map at LCMAP's reference plots; 18 embedding maps at PLANTEUR, at 400 labels | exp80, exp89, exp93, exp94, exp110, exp112, exp113 |
| Input conditions (`--condition`) | Does the confidence fall with accuracy when an input is missing? | The embedding suite: PASTIS (S1, S2, both) and CropHarvest China | exp88 |
| Agent use (`mcp`) | Does a language-model agent answer map questions correctly with the package? | Forty cards from GEOID-Flood and Sen1Floods11, with hidden labels | exp64 |

## The test sets

"Probability sample" asks whether the labelled units are a random sample of the map with known inclusion
probabilities, the condition for an estimate of the map's accuracy. Licences are as stated by each publisher;
"Host" is where the record read the data.

| Test set | Reference produced by | Probability sample of the map? | What the record used | Host | Licence | Experiments |
|---|---|---|---|---|---|---|
| Sen1Floods11 (Bonafilia et al. 2020) | Analysts drawing water on the same Sentinel-1/2 imagery | No: chips chosen per event | valid (600 tiles, to fit heads), test (2,419 tiles), Bolivia (a held-out event), train (fine-tuning) | Ai2's mirror of the Cloud to Street bucket | None stated by the publisher | exp18 to exp53 (some archived), exp64, exp65, exp91 |
| AWF (`allenai/olmoearth_projects_awf`) | Ai2's expert-labelled points | No | 1,459 points, 344 in the validation split | Hugging Face | Apache-2.0 | exp04, exp21, exp35, exp89, exp98, exp101, exp103, exp109, exp111 |
| ESA WorldCover 2020 and 2021 | Another product's map, used as a weak reference; in exp94 a graded product | No: 27 scenes on 8 rivers chosen by a rule | water class at Sentinel-2 L2A scenes; forest (tree cover, mangroves) at GFC2020's validation units | Planetary Computer; AWS | CC BY 4.0 | exp02 to exp37 (some archived), exp94 |
| OlmoEarth paper embeddings (`allenai/olmoearth-paper-embeddings`) | The benchmarks' own labels, in splits fixed by the model authors; every unit labelled | No: benchmark splits, 24 tasks from 14 sources | 6,435,473 graded units (exp70); 16 encoders (exp79) | Hugging Face, revision pinned | CC BY 4.0 (embeddings); the source benchmarks keep their own | exp54, exp63, exp70 to exp90 |
| GEOID-Flood (`links-ads/geoid-flood`) | Copernicus EMS flood delineations, manually validated | No: a shard subset of the test split | 55 test areas; the 45 scored come from 9 activations | Hugging Face | CC BY 4.0 | exp55 to exp61 (some archived), exp64 |
| WorldFloods v2 (`isp-uv-es/WorldFloodsv2`) | Water and cloud masks shipped with the scenes | No | 544 chips in 6 events with a clear post-event optical scene | Hugging Face | CC BY-NC 4.0 | exp62 |
| JRC Global Surface Water | A water-history product, used as an arbiter and a control | No | seasonality, occurrence and monthly history | Planetary Computer, JRC | CC BY 4.0 | exp25, exp61 (archived; [v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#is-the-pre-event-radars-departure-the-seasonal-water-of-its-date-exp61)) |
| DFC2020 (IEEE GRSS Data Fusion Contest 2020) | An iterated random forest over Sentinel-1/2 and other maps (10 m), and MODIS labels (500 m) | No | probe fitted on 400 validation patches, graded on 1,200 test patches; no reference classes are committed | An unofficial Hugging Face mirror (`125oii/dfc2020`) | IEEE GRSS contest terms: files for approved participants; publication subject to approval by the IEEE GRSS IADF committee and TU Munich | exp66 |
| Dynamic World expert test tiles (Zenodo 4766508) | Expert consensus annotation, beside Dynamic World's own probabilities | No | 409 tiles, 4,348,526 windows | Zenodo | CC BY 4.0 | exp67; a sample tile ships with the package |
| LUCAS Copernicus 2022 (JRC) | Surveyors in the field | Of the LUCAS Copernicus polygons, by our stratified draw with known inclusion probabilities; not of EU area | 12,073 polygons drawn, 4,778 field-surveyed in held-out regions | JRC open data | CC BY 4.0 | exp68 |
| EuroCrops V2 (JRC) | Farmers' parcel declarations to paying agencies | No | Austria 2020 and 2021, Denmark 2018 and 2019, Slovenia 2020 and 2021 | JRC open data; class mappings on GitHub | CC BY 4.0 (data), CC BY-SA 4.0 (mappings) | exp69 |
| Ai2's fine-tuned models (FT-AWF, FT-Forest Loss Driver) | Ai2's validation labels | No | AWF 344 points; Forest Loss Driver 109 windows | Hugging Face (models), Ai2's public bucket (Forest Loss Driver data) | OlmoEarth Artifact License (models); CC BY 4.0 (Forest Loss Driver data) | exp21, exp89, exp98, exp103, exp111 |
| OlmoEarth LCC (`allenai/olmoearth_lcc`) | No reference: served rasters checked without labels | n/a | land-cover-change rasters | Hugging Face | Apache-2.0 | exp20, exp22 (archived; [v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#periodic-artifacts-in-the-served-product-exp22)) |
| LCMAP Reference Data Product v1.2 (USGS) | Analysts interpreting each plot for LCMAP's programme | **Yes**: a simple random sample of the 30 m pixels of the conterminous United States | 25,000 plots, 2018 | USGS ScienceBase | US public domain | exp92, exp93, exp112 |
| NLCD accuracy-assessment points, 2011 and 2016 editions (USGS; Wickham et al. 2017, 2021) | Analysts interpreting each point for NLCD's own assessment | **Yes**: stratified random samples of the conterminous United States, with weights | 8,000 points (2001, 2006, 2011) and 4,629 (2011, 2016), with NLCD's map classes in the file | USGS ScienceBase | US public domain | exp94 |
| JRC GFC2020 validation set v2 (Colditz et al.) | Interpreters labelling forest or not for 2020 | **Yes**: stratified by 149 continental strata, areas published | 21,612 units, with GFC2020 V2 in the file | JRC open data | CC BY 4.0 | exp94 |
| East Africa TimeSync sample (Bullock et al. 2021) | Analysts interpreting each plot's Landsat history | **Yes, within each country**: 2,000 random plots in each of seven countries | 14,000 plots, labels for 2015, 2016, 2017; 900 fresh Kenya and Tanzania 2017 plots more than 100 km from the AWF area drawn for exp111 | GitHub (bullocke/eastafrica) | CC0 | exp94, exp99, exp101, exp103, exp109, exp111 |
| MapBiomas Alerta (published alerts, Amazon biome, detected January to June 2026) | Analysts refining each alert on high-resolution imagery and recording its probable driver | **Yes, stratified**: agriculture a simple random 300 of 3,565, mining (garimpo and mining) all 389, roads all 7, with weights; alert codes and polygons stay off the repository | 696 alerts, 689 graded | MapBiomas's public GeoServer (no account) | CC BY | exp104 |
| FTW benchmark v1 (Fields of The World, Kerner et al.), test chips of Belgium (Flanders), Croatia, Slovenia and Sweden | National parcel data rasterised by FTW: farmers' declarations (Belgium and Slovenia, via EuroCrops), photo-interpreted parcels (Croatia) and LPIS blocks that can hold several fields (Sweden); a 2-class mask, which is field only (exp110), and a 3-class mask (background, field, boundary) | No: the benchmark's test split | 1,295 test chips, 1,326,080 windows of 8 x 8 mask pixels; no chip identifier, mask or per-window value is committed | Source Cooperative (`kerner-lab/fields-of-the-world`) | Per country: Belgium the Flemish model licence for free reuse ("Bron: Dept. LV"); Croatia "Open Data", linking to terms in a .docx the record did not open; Slovenia CC BY 4.0; Sweden "No restrictions on public access" in FTW's table, the provider's record "CCBY" and elsewhere "conditions unknown"; none non-commercial | exp110 |
| FTW Global Data 2nd edition | The graded product: per-pixel field and boundary probabilities (uint8 / 255), 2.5 m, made by FTW's ftw-q model | n/a | the label year's edition (2021, Croatia 2023), area-averaged onto each chip's mask grid | Source Cooperative (`ftw/global-data-2e`) | CC BY 4.0 | exp110 |
| FTW PRUE U-Net EfficientNet-B3, CC-BY checkpoint (ftw-baselines release v3.1) | The second map of exp110's arm 4, run by the record on the benchmark's two Sentinel-2 windows | n/a | a 3-class softmax on the 1,295 chips, kept outside the repository | GitHub (`fieldsoftheworld/ftw-baselines` releases) | CC BY 4.0 | exp110 |
| S2GLC validation set (Jenerowicz et al., PANGAEA 934197; design in Malinowski et al. 2020) | Interpreters labelling Sentinel-2 points for S2GLC | No: tiles drawn by country, points by class within each tile, classes under 0.95% of a tile left out, no weights | 52,024 points in 55 tiles, 2017 | PANGAEA | CC BY 4.0 | exp94 |
| CGLS-LC100 v3 (Copernicus) | The graded product: discrete class and its probability, 100 m | n/a | read at the East Africa plots, 2015 to 2017 | Zenodo | CC BY 4.0 | exp94 |
| ODSE-LULC v0.2 (EcoDataCube) | The graded product: dominant CORINE class and per-class probabilities, 30 m | n/a | read at the S2GLC points, 2017 | EcoDataCube S3 | CC BY-SA 4.0 | exp94 |
| OlmoEarth v1.3 embeddings (Ai2) | The input of the graded map: 128-dimensional annual 10 m vectors, 2018 at LCMAP's reference plots, 2019 on PLANTEUR's label grid | n/a | 12,000 plots read; 2,750 PLANTEUR patches | Source Cooperative; Hugging Face (with PLANTEUR) | OlmoEarth Artifact License (derivatives carry its use restrictions) | exp112, exp113 |
| PLANTEUR crop-type benchmark (Ai2) | France's parcel registry (Registre parcellaire graphique): seven crop classes, the rest Background | No: patches chosen by the benchmark; each held-out territory's windows are graded in full | 2,750 patches of 128 x 128 pixels at 10 m in six overseas territories, 2019, with leave-one-island-out and spatial splits | Hugging Face (`allenai/planteur-benchmark`) | ODC-BY | exp113 |
| TESSERA v2 and AlphaEarth Foundations embeddings | Inputs of the compared maps: 128 and 64 dimensions, 2019, on PLANTEUR's label grid | n/a | 2,750 patches each | Hugging Face (with PLANTEUR) | CC0 (TESSERA v2), CC BY 4.0 (AlphaEarth) | exp113 |
| LCMAP CONUS Collection 1.3 (USGS) | The graded product: primary land cover and its confidence | n/a | read at LCMAP's reference plots and NLCD's assessment points | Planetary Computer | CC0 | exp92, exp93, exp94, exp112 |
| Esri 10 m Annual Land Use Land Cover v2 | The graded product, crosswalked to LCMAP's classes | n/a | read at LCMAP's plots (2018), and at the East Africa and S2GLC points (2017) | Planetary Computer | CC BY 4.0 | exp92, exp94, exp112 |
| Sentinel-2 L2A | Imagery, not a reference | n/a | scenes for the encoders | Planetary Computer | Copernicus data terms | exp01 to exp53 (some archived), exp68, exp69 |
| OpenStreetMap river centrelines | Used to choose scenes | n/a | `waterway=river` | Overpass | ODbL | exp02 to exp13 |

The private OlmoEarth Studio project used by the archived agent trials exp86 and exp87 is described in the [v1.9.0
copy of this page](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/testbeds.md); the kept Studio trial of 24 September 2026
(`exp/out/agent_trial_2026-09-24.md`, [agent integration](method/agent_integration.md)) used the same project.

## What the test sets do not cover

- Few probability samples of maps: LCMAP's (2018), NLCD's two assessment samples (2001 to 2016), GFC2020's (forest,
  2020) and East Africa's (random within seven countries, 2015 to 2017). The points stand in for the maps, so these
  grade the procedures exactly on the points; products are compared through crosswalks, and several pairs are related
  (NLCD and LCMAP, GFC2020 and WorldCover).
- Most of the evidence is linear probes on frozen embeddings, graded against benchmark splits that are not samples
  of any map. The hand-labelled image testbeds are mostly binary water.
- The fine-tuned evidence is two Ai2 models graded on their own validation sets; FT-AWF also through a re-run of its
  deployment configuration (exp98) and on 2017 imagery at the East Africa TimeSync plots that had imagery (exp99), and
  the record's own v1.2 fine-tune of the AWF task under the same rules (exp103) and, blind, at 890 fresh TimeSync plots
  more than 100 km from the AWF area, with Ai2's v1 report-only there (exp111); Forest Loss Driver also through its
  deployment configuration at 2026 MapBiomas Alerta alerts, in two of its ten classes (exp104).
- One deployed product of another organisation outside land cover: FTW's field map, at the FTW benchmark's test chips in four
  European countries, a benchmark split whose train chips trained the map's model (exp110).

## Why the data are not re-hosted

The checks rerun from the per-unit values committed in `exp/out/` (for example `exp92_plots.csv`, `exp78_units/`),
which hold no coordinates. The raw data stay with their publishers, at the revisions pinned in
`exp/out/upstream_revisions.json`, and their licences differ: WorldFloods v2 is non-commercial, DFC2020 is released
to approved contest participants, and Sen1Floods11's labels carry no stated licence. The DFC2020 reference classes
once committed in `exp/out/exp66_masks.npz` were removed on 5 October 2026; exp66's results stand as an internal record. exp94's per-point files hold values derived from the publishers' data under the licences in the table above; the ODSE-LULC columns of `exp/out/exp94_europe_points.csv` (`odse_corine`, `odse`, `odse_prob`) derive from a CC BY-SA 4.0 product and are shared under CC BY-SA 4.0. exp110 commits no per-window or per-chip file, a departure from that habit: its labels' terms are weakly documented for Croatia, so the record holds aggregates only, with the inputs' sha256. A copy on Hugging Face would add
nothing the record needs and could breach those terms.
