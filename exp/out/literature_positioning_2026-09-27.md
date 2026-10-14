# Literature positioning, 27 September 2026

A search of arXiv, Semantic Scholar and OpenAlex in four directions:

1. uncertainty in Earth-observation foundation models;
2. design-based accuracy assessment and prediction-powered inference;
3. geospatial LLM agents;
4. tool grounding and self-verification.

A synthesis then confirmed every paper it relies on. It also read the 19 pages of docs.olmoearth.allenai.org. The
search is not exhaustive. Summaries are in the searchers' and the synthesis's own words.

A1 to A4 are the package's four parts: the review order, the error estimate, certified zones and map comparison. B is
the agent trial, and C is whether OlmoEarth already assesses map credibility.

**Literature positioning for olmoearth-inferencex (A1–A4), the agent trial (B) and the Ai2 question (C), 27 Sep 2026**

**Verification of the searchers' claims**

I checked all 32 distinct papers marked already_does_it or does_part.

- **arXiv papers (22):** one arXiv API batch call returned all 22. Title, first author and date match in every case, and each abstract supports what the searcher said.
- **Journal papers (10):** the Semantic Scholar batch call returned 429 (rate limited). I got the records from OpenAlex by DOI plus three single Semantic Scholar lookups. All 10 exist with the stated authors, venue, volume and pages.
  - Abstracts retrieved and consistent: Stehman 2014, Tyukavina 2025, Foody 2004, McIver & Friedl 2001, Fritz & See 2008.
  - Abstracts withheld by the publisher: Olofsson 2014, Stehman 2009, Stehman 1997, Khatami 2017, Steele 2005. For these only the metadata is confirmed (Khatami's co-authors are Mountrakis and Stehman). Semantic Scholar's machine summaries for Olofsson and Khatami agree with the searchers' summaries. The rest rests on the papers' well-known content.
- **Dropped:** none.

**Corrections to the searchers' notes**

1. **Kumar & Raj 2018 (arXiv 1607.02665) is stronger prior art than reported.** I read the full text. Section 3 builds the strata on "the probability of the predicted class" (or the score's absolute value for SVMs) and compares proportional with optimal allocation. That is A2's confidence-stratified design without the map or the clustering. For the design idea it should count as already_does_it.
2. **SHRUG-FM v2 abstract:** it now reports "outperforming established single-signal baselines like predictive entropy". The memory and docs line saying it has no single-model confidence baseline is out of date.
3. **Lehmann 2026:** the claim that "confidence-based abstention cannot defer around confidently wrong predictions" is made under corruption and shift. The abstract does not make it for clean data. It sharpens A1's scope (in-distribution) but does not directly contradict A1.
4. **StratPPI:** the arXiv title is "Stratified Prediction-Powered Inference for Hybrid Language Model Evaluation".
5. **Shirota 2026:** single author. Its main result goes beyond the ~10% smaller standard errors. When map errors are spatially dependent and the labelling propensities are wrong, coverage can get *worse* as labels accumulate.
6. **TerraBench:** the abstract confirms the three tracks and the tolerance-aware numeric scoring. The claim that provenance is enforced only on reference traces rests on the searcher's full-text reading; I did not re-check it.
7. **Basu 2026:** confirmed that it is evaluated on injected hallucinations, not natural answers.

**Not re-checked:** papers marked "related", namely Valle 2023, Dayan 2026, Pontius & Millones 2011, Huang 2024, Kamoi 2024, Stechly 2024, Hsieh 2023, Hasan 2026, Munir 2026, Dynamic World, WorldCereal and Singh 2024. Where I cite them below, they rest on the searchers' reading.

---

**A1. Label-free review by confidence margin**

- **Already known**
  - The model's own confidence separates errors from correct predictions. This is the standard baseline (Hendrycks & Gimpel 2017).
  - In land cover, it goes back to per-pixel confidence maps where errors carry low confidence (McIver & Friedl 2001) and to maps of per-pixel predicted accuracy, fitted with reference data (Khatami, Mountrakis & Stehman 2017).
  - In EO segmentation, per-pixel uncertainty is benchmarked by how well it finds errors (Rey 2025).
  - Selective prediction for GeoFMs exists: SHRUG-FM 2025 is per image, with a decision tree fitted on labels, and now beats predictive entropy.
  - Lehmann 2026 tests 16 frozen encoders and finds GeoFMs grow more overconfident under shift, and that confidence-based abstention fails there.
- **Partly new:** a label-free, within-map ranking framed as review order, reported as the share of the random-to-perfect gap closed. That metric is close to a normalised risk–coverage curve, and reviewers will want it stated in selective-classification terms.
- **Appears new:**
  - A head-to-head of the model's margin against no-model controls (embedding distance and similar) on one suite of 16 GeoFM encoders × 24 tasks with 10 probe seeds.
  - The finding that the ceiling tracks the task more than the encoder.
- **Confidence:** high that the idea is not new, medium that the controlled comparison at this scale is. Lehmann is the paper most likely to have overlapped, and it did not do this.
- **Reviewer will ask:** does the ranking survive shift? This is Lehmann's setting; the LUCAS results are a partial answer at best.

**A2. Design-based error estimation**

- **Already known**
  - The estimator is standard. Olofsson et al. 2014 give the good-practice recipe. Stehman 2014 gives estimators when strata are not map classes, which is exactly A2's confidence strata. Stehman 1997 shows that clustered reference samples understate the standard error, the mechanism behind A2's design effect. Stehman & Foody 2019 set the standard for rigour. Tyukavina 2025 adds area-weighted estimators and code.
  - Wadoux 2021 argues that a validation-split score is not map accuracy.
  - Stratifying by the classifier's own confidence to estimate its accuracy is in Kumar & Raj 2018 (non-spatial).
  - Model-assisted and prediction-powered estimators do the same job, possibly with tighter intervals: Stehman 2009, PPI (Angelopoulos 2023), StratPPI (Fisch 2024), active inference (Zrnic 2024), PPAT (2026).
  - Shirota 2026 does design-based PPI for spatial maps: LUCAS data, stratified designs, and spatially correlated errors that break the guarantees.
  - Steele 2005 is the model-based alternative; it depends on calibration, which Lehmann finds fails.
- **Partly new:** measuring interval coverage empirically on 111 encoder-task cells for foundation-model maps, and quantifying the design effect of tile sampling (about 3) for such maps. The clustering phenomenon itself is known.
- **Appears new at method level:** nothing. What is new is joining margin strata to the Olofsson/Stehman estimator inside a tool.
- **Confidence:** high that the estimator is not new. Medium that no EO paper stratifies by a foundation model's margin and audits coverage at scale; the search was not exhaustive.
- **Reviewer will ask**
  - Why not a difference or PPI estimator (Stehman 2009, Shirota 2026)?
  - What is the response design, i.e. who labels a window and how (Stehman & Foody 2019)?
  - Are unequal window areas weighted (Tyukavina 2025)?
  - Why is coverage 0.93 against a nominal 0.95?

**A3. Certified zones**

- **Already known**
  - The procedure: Geifman & El-Yaniv 2017 keep the most-confident part and bound its error at a set risk with probability 1−δ, using a binomial tail bound.
  - A prefix rule over nested thresholds is fixed-sequence testing, justified in general by Learn then Test (Angelopoulos 2021), which also avoids Geifman's union bound.
  - Nearby EO guarantees use different contracts: conformal prediction sets per pixel (Valle 2023, Singh 2024), conformal risk-controlled wildfire zones (Dayan 2026), and conformal intervals plus survey-based aid allocation with a provable exclusion-risk bound (Pettersson 2026).
- **Partly new:** the calibration sample is a probability design over a map's windows, and the output is a certified region of a map rather than an accepted set of test inputs.
- **Appears new:** no EO paper found that certifies a map zone's error rate from a random design. The procedure itself is Geifman's.
- **Confidence:** high that the procedure is not new, medium that the EO application is.
- **Reviewer will ask**
  - The exact binomial test is valid only if the sampled windows are the independent units. If windows are drawn in tiles, A2's design effect applies and the test is anti-conservative.
  - Was the prefix order fixed before the labels were seen?

**A4. Comparing two maps**

- **Already known**
  - Pixel-by-pixel disagreement maps between land-cover products, used to point reviewers at places to check (Fritz & See 2008).
  - Paired tests of accuracy on a shared sample, not independent kappas (Foody 2004).
  - Using a sample stratified on one map to assess another (Stehman 2014).
  - The land-change literature (Olofsson 2014) already treats a difference between dates as change that must be told apart from map error with a reference sample.
  - Pontius & Millones 2011 split disagreement into quantity and allocation.
- **Partly new:** attaching each model's confidence to where two foundation-model maps differ, across models, sensors or dates, in one tool.
- **Appears new:** little that can be defended.
- **Confidence:** low. The search here was thin, with no targeted search for comparisons of foundation-model maps.
- **Reviewer will ask:** when claiming one map is better, is a paired test run on a shared sample?

**B. LLM agent with tools that state their conclusions, limits and a capability card**

- **Already known**
  - Tool-augmented models misjudge whether a task can be solved at all (ToolBeHonest 2024). Geospatial benchmarks score refusal of unsolvable or unanswerable tasks (GeoBenchX 2025, OGD4All 2026).
  - Claim-level factuality rates follow FActScore (Min 2023).
  - Tool-side signalling controls fabrication (Sethi 2026). A named status in the tool output, or a required emitted status line, cuts dishonesty from 14.1% to 0.87%, and that flag can be checked with a regex. This explains why our regexes on free wording catch only about half.
  - Tool descriptions affect agent behaviour (Hsieh 2023; Hasan 2026 on MCP descriptions, measured by task success).
  - Stronger grounding architectures exist: an evidence ledger (LEDGERMIND 2026) and signed tool receipts (Basu 2026, on injected errors only).
  - Our claim-check result is the expected one: zero-shot LLM judges reach at most 0.22 span-F1 on tool-output grounding against 0.69 for a fine-tuned 2B detector (Kovács 2026), and intrinsic self-correction does not work (Huang 2024, Kamoi 2024).
- **Partly new:** a capability card and per-tool conclusion/limits statements as a no-training intervention, with a hallucination endpoint instead of task success.
- **Appears new**
  - For EO agents, a blind adversarial audit of material false statements per sentence in open-ended answers. The geospatial agent papers found (Earth-Agent, ThinkGeo, GeoBenchX, TerraBench, Earth AI) score task success, trajectories or LLM-judge accuracy.
  - An agent whose answers carry credibility figures for the foundation-model maps they rely on. Munir 2026 states this need only as a formula.
- **Confidence:** medium. The agent literature moves fast and only arXiv was searched.
- **Reviewer will ask**
  - The per-sentence Wilson intervals treat sentences as independent. Sentences cluster within answers and briefs, so the non-overlap overstates the evidence; the per-answer comparison (16 → 9 of 30, intervals overlapping) is the more honest unit. This is A2's design-effect lesson applied to our own endpoint.
  - The 8 briefs are the ones the fixes were built against, and the held-out briefs have not been run.
  - Materiality is the auditor's judgement.

**C. Does OlmoEarth already assess map credibility "as part of the model"?**

No public evidence that it does. What it provides:

- **Benchmark and validation-split scores.** The project docs report, for example, 89.5% for AWF land use on a spatially hashed validation split. That is a model score, not a design-based accuracy of a deployed map (Wadoux 2021).
- **Optional softmax probabilities** in rslearn (`prob_property`, `output_probs`, a fixed temperature). This is the ingredient for A1's margin, not an assessment.
- **Label-quality checks on training labels** in olmoearth_projects.
- **A human review and correction loop.**
  - The OlmoEarth paper's platform workflow ends with "review and publish the final maps".
  - I fetched all 19 pages of docs.olmoearth.allenai.org from its sitemap today (not checked by any searcher). None mentions confidence, uncertainty, probability, calibration, credibility or error rate for predictions. The prediction postprocessors are vectorize and value_transform. The model config reference exposes no probability output.
  - The only related passages are the fine-tuning page ("confirming or correcting predictions with additional annotations can help reduce false positives"), a loop that feeds back into training, and advice about label confidence in training data.

The searchers found nothing in either OlmoEarth paper (v1, v1.2) or the Hugging Face model cards.

The owner's impression plausibly comes from the review/correct step or rslearn's probability output. Neither ranks errors, estimates a map's error rate or certifies zones.

**Confidence:** medium-high for public material. The closed Studio UI cannot be inspected. One concrete question for Ai2: does the Studio review step show per-pixel probabilities, or estimate map accuracy from a sample?

---

**Papers a reviewer would most expect us to cite**

1. Olofsson et al. 2014, RSE 148:42–57, for A2 and A4 (with Stehman 1997 for the clustering).
2. Geifman & El-Yaniv 2017, NeurIPS, for A3 (with Learn then Test for the prefix rule).
3. Hendrycks & Gimpel 2017, ICLR, for A1.
4. Lehmann et al. 2026, arXiv 2608.16614, the direct challenge to A1 at the same 16-encoder scale.
5. Sethi et al. 2026, arXiv 2609.14758, the closest prior work for B.

The "why not" question a reviewer will raise: prediction-powered inference (Angelopoulos 2023) and its spatial design-based version (Shirota 2026, arXiv 2608.10356).

**What is new, in one sentence**

The statistical parts (confidence ranking, confidence-stratified design-based error estimates, selective classification with guaranteed risk, map comparison) all exist. What is new is joining them in one after-the-fact credibility tool for foundation-model maps and testing it at scale: the margin ranks errors better than no-model controls on 16 encoders × 24 tasks, the intervals keep their coverage on 111 encoder-task cells, and tile sampling under-covers with a measured design effect near 3. The agent trial adds a development-stage finding (8 briefs, held-out set not yet run): making tools state their conclusions, limits and capabilities reduced audited false statements, while the model checking its own answer did not help.

Working files were kept outside the repository.
- arxiv_batch.xml
- kumar.txt
- oedocs/ (the 19 docs pages as text)

## Papers found, by search direction

### eo_uncertainty

- Maria Gonzalez-Calabuig (2025). SHRUG-FM: Reliability-Aware Foundation Models for Earth Observation. arXiv:2511.10370 (accepted for proceedings, CVPR EarthVision 2026). https://arxiv.org/abs/2511.10370. Bears on A1, A3; does part.
- Nils Lehmann (2026). Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts. arXiv:2608.16614. https://arxiv.org/abs/2608.16614. Bears on A1, A3; does part.
- Melanie Rey (2025). Uncertainty evaluation of segmentation models for Earth observation. arXiv:2510.19586. https://arxiv.org/abs/2510.19586. Bears on A1; does part.
- Yonatan Geifman (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017; arXiv:1705.08500. https://arxiv.org/abs/1705.08500. Bears on A3, A1; already does it.
- Dan Hendrycks (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. ICLR 2017; arXiv:1610.02136. https://arxiv.org/abs/1610.02136. Bears on A1; does part.
- Douglas K. McIver (2001). Estimating pixel-scale land cover classification confidence using nonparametric machine learning methods. IEEE Transactions on Geoscience and Remote Sensing 39(9):1959-1968. https://doi.org/10.1109/36.951086. Bears on A1; does part.
- Reza Khatami (2017). Mapping per-pixel predicted accuracy of classified remote sensing images. Remote Sensing of Environment 191:156-167. https://doi.org/10.1016/j.rse.2017.01.025. Bears on A1, A2; does part.
- Brian M. Steele (2005). Maximum posterior probability estimators of map accuracy. Remote Sensing of Environment 99:254-270. https://doi.org/10.1016/j.rse.2005.09.001. Bears on A2, A1; does part.
- Christopher F. Brown (2022). Dynamic World, Near real-time global 10 m land use land cover mapping. Scientific Data 9. https://doi.org/10.1038/s41597-022-01307-4. Bears on A1, C; related.
- Kristof Van Tricht (2023). WorldCereal: a dynamic open-source system for global-scale, seasonal, and reproducible crop and irrigation mapping. Earth System Science Data 15:5491-5515. https://doi.org/10.5194/essd-15-5491-2023. Bears on A1, A2; related.
- Steffen Fritz (2008). Identifying and quantifying uncertainty and spatial disagreement in the comparison of Global Land Cover for different applications. Global Change Biology 14(5):1057-1075. https://doi.org/10.1111/j.1365-2486.2007.01519.x. Bears on A4, A1; does part.
- Geethen Singh (2024). Uncertainty quantification for probabilistic machine learning in earth observation using conformal prediction. arXiv:2401.06421. https://arxiv.org/abs/2401.06421. Bears on A2, A3, C; related.
- Markus B. Pettersson (2026). Beyond Point Predictions: Uncertainty-Aware Satellite Poverty Mapping for Public Policy. arXiv:2608.23322 (under consideration at PNAS). https://arxiv.org/abs/2608.23322. Bears on A2, A3; does part.
- Spyros Kondylatos (2025). On the Generalization of Representation Uncertainty in Earth Observation. ICCV 2025; arXiv:2503.07082. https://arxiv.org/abs/2503.07082. Bears on A1; related.
- Burak Ekim (2024). Distribution Shifts at Scale: Out-of-distribution Detection in Earth Observation. arXiv:2412.13394. https://arxiv.org/abs/2412.13394. Bears on A1; related.
- Syed Roshaan Ali Shah (2026). Embeddings based Anomaly Detection for Cleaning Global Crop Type Reference Datasets. arXiv:2607.23908 (ECCV 2026 TerraBytes workshop, oral). https://arxiv.org/abs/2607.23908. Bears on C, A1; related.
- Raul Ramos-Pollan (2024). Uncertainty and Generalizability in Foundation Models for Earth Observation. arXiv:2409.08744. https://arxiv.org/abs/2409.08744. Bears on A2; related.
- Yuanyuan Wang (2024). How Certain are Uncertainty Estimates? Three Novel Earth Observation Datasets for Benchmarking Uncertainty Quantification in Machine Learning. arXiv:2412.06451 (submitted to IEEE GRSM). https://arxiv.org/abs/2412.06451. Bears on A1; related.
- Max Gaber (2026). Uncertainty-aware tree height change regression. arXiv:2607.00638. https://arxiv.org/abs/2607.00638. Bears on A4; related.

### accuracy_assessment

- Pontus Olofsson (2014). Good practices for estimating area and assessing accuracy of land change. Remote Sensing of Environment 148:42-57, doi:10.1016/j.rse.2014.02.015. https://doi.org/10.1016/j.rse.2014.02.015. Bears on A2; already does it.
- Stephen V. Stehman (2014). Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. International Journal of Remote Sensing 35:4923-4939, doi:10.1080/01431161.2014.930207. https://doi.org/10.1080/01431161.2014.930207. Bears on A2, A4; already does it.
- Stephen V. Stehman (2019). Key issues in rigorous accuracy assessment of land cover products. Remote Sensing of Environment 231:111199, doi:10.1016/j.rse.2019.05.018. https://doi.org/10.1016/j.rse.2019.05.018. Bears on A2; related.
- Stephen V. Stehman (2009). Model-assisted estimation as a unifying framework for estimating the area of land cover and land-cover change from remote sensing. Remote Sensing of Environment 113:2455-2462, doi:10.1016/j.rse.2009.07.006. https://doi.org/10.1016/j.rse.2009.07.006. Bears on A2; does part.
- Stephen V. Stehman (1997). Estimating standard errors of accuracy assessment statistics under cluster sampling. Remote Sensing of Environment 60:258-269, doi:10.1016/S0034-4257(96)00176-9. https://doi.org/10.1016/s0034-4257(96)00176-9. Bears on A2; does part.
- Alexandra Tyukavina (2025). Practical global sampling methods for estimating area and map accuracy of land cover and change. Remote Sensing of Environment 324:114714, doi:10.1016/j.rse.2025.114714. https://doi.org/10.1016/j.rse.2025.114714. Bears on A2; does part.
- Sergii V. Skakun (2025). The impact of map accuracy on area estimation with remotely sensed data within the stratified random sampling design. Remote Sensing of Environment 326:114805, doi:10.1016/j.rse.2025.114805. https://doi.org/10.1016/j.rse.2025.114805. Bears on A2; related.
- Alexandre M.J.-C. Wadoux (2021). Spatial cross-validation is not the right way to evaluate map accuracy. Ecological Modelling 457:109692, doi:10.1016/j.ecolmodel.2021.109692. https://doi.org/10.1016/j.ecolmodel.2021.109692. Bears on A2; related.
- Reza Khatami (2017). Mapping per-pixel predicted accuracy of classified remote sensing images. Remote Sensing of Environment 191:156-167, doi:10.1016/j.rse.2017.01.025. https://doi.org/10.1016/j.rse.2017.01.025. Bears on A1; does part.
- Giles M. Foody (2004). Thematic map comparison: evaluating the statistical significance of differences in classification accuracy. Photogrammetric Engineering & Remote Sensing 70(5):627-633, doi:10.14358/PERS.70.5.627. https://doi.org/10.14358/pers.70.5.627. Bears on A4; does part.
- Denis Valle (2023). Quantifying uncertainty in land-use land-cover classification using conformal statistics. Remote Sensing of Environment 295:113682, doi:10.1016/j.rse.2023.113682. https://doi.org/10.1016/j.rse.2023.113682. Bears on A3, A1; related.
- Anastasios N. Angelopoulos (2023). Prediction-powered inference. Science 382, doi:10.1126/science.adi6000 (arXiv:2301.09633). https://arxiv.org/abs/2301.09633. Bears on A2; does part.
- Adam Fisch (2024). Stratified Prediction-Powered Inference for Effective Hybrid Evaluation of Language Models. NeurIPS 2024 (Advances in Neural Information Processing Systems 37:111489-111514), arXiv:2406.04291. https://arxiv.org/abs/2406.04291. Bears on A2; does part.
- Shinichiro Shirota (2026). Design-Based Prediction-Powered Inference for Spatial Data. arXiv:2608.10356. https://arxiv.org/abs/2608.10356. Bears on A2; does part.
- Kerri Lu (2025). Regression coefficient estimation from remote sensing maps. Remote Sensing of Environment 330:114949, doi:10.1016/j.rse.2025.114949 (arXiv:2407.13659). https://arxiv.org/abs/2407.13659. Bears on A2; related.
- Tijana Zrnic (2024). Active Statistical Inference. ICML 2024, arXiv:2403.03208. https://arxiv.org/abs/2403.03208. Bears on A2, A1; does part.
- Kianoosh Ashouritaklimi (2026). Prediction-Powered Active Testing. arXiv:2607.08347. https://arxiv.org/abs/2607.08347. Bears on A2, A1; does part.
- Anurag Kumar (2018). Classifier Risk Estimation under Limited Labeling Resources. PAKDD 2018, arXiv:1607.02665. https://arxiv.org/abs/1607.02665. Bears on A2; does part.
- Yonatan Geifman (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017, arXiv:1705.08500. https://arxiv.org/abs/1705.08500. Bears on A3; already does it.
- Anastasios N. Angelopoulos (2021). Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. arXiv:2110.01052. https://arxiv.org/abs/2110.01052. Bears on A3; does part.

### agents

- Aaron Bell (2025). Earth AI: Unlocking Geospatial Insights with Foundation Models and Cross-Modal Reasoning. arXiv 2510.18318 (Google Research technical report). https://arxiv.org/abs/2510.18318. Bears on B, A1; related.
- Muhammad Akhtar Munir (2026). Agentic AI for Remote Sensing: Technical Challenges and Research Directions. arXiv 2604.24919 (position paper). https://arxiv.org/abs/2604.24919. Bears on A1, A2, B; related.
- Varvara Krechetova (2025). GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks. arXiv 2503.18129. https://arxiv.org/abs/2503.18129. Bears on B; does part.
- Akashah Shabbir (2025). ThinkGeo: Evaluating Tool-Augmented Agents for Remote Sensing Tasks. arXiv 2505.23752. https://arxiv.org/abs/2505.23752. Bears on B; related.
- Peilin Feng (2025). Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents. arXiv 2509.23141. https://arxiv.org/abs/2509.23141. Bears on B; related.
- Zhutao Lv (2026). Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents. arXiv 2609.12533. https://arxiv.org/abs/2609.12533. Bears on B; related.
- Dat Tien Nguyen (2026). TerraBench: Can Agents Reason Over Heterogeneous Earth-System Data?. arXiv 2606.13148. https://arxiv.org/abs/2606.13148. Bears on B; does part.
- Abhinav Pothuri (2026). GISAgentBench: A Practitioner-Sourced Benchmark for Evaluating LLM Agents on GIS Tasks. arXiv 2608.01645. https://arxiv.org/abs/2608.01645. Bears on B; related.
- Maram Hasan (2026). GeoDisaster: Benchmarking Orchestrated Agents for Operational Disaster Geo-Intelligence. arXiv 2606.17246. https://arxiv.org/abs/2606.17246. Bears on B, A4; related.
- Chia Hsiang Kao (2025). Towards LLM Agents for Earth Observation. arXiv 2504.12110 (ICML 2025 TerraBytes workshop). https://arxiv.org/abs/2504.12110. Bears on B; related.
- Simranjit Singh (2024). GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots. arXiv 2404.15500 (EarthVision 2024, CVPR workshop). https://arxiv.org/abs/2404.15500. Bears on B; related.
- Zhenlong Li (2023). Autonomous GIS: the next-generation AI-powered GIS. arXiv 2305.06453. https://arxiv.org/abs/2305.06453. Bears on B; related.
- Niloufar Alipour Talemi (2026). Agentic AI in Remote Sensing: Foundations, Taxonomy, and Emerging Systems. WACV 2026 GeoCV Workshop, pp. 786-799 (arXiv 2601.01891). https://arxiv.org/abs/2601.01891. Bears on B; related.
- Michael Siebenmann (2026). OGD4All: A Framework for Accessible Interaction with Geospatial Open Government Data Based on Large Language Models. IEEE CAI 2026, pp. 882-888 (arXiv 2602.00012). https://arxiv.org/abs/2602.00012. Bears on B; does part.
- Meenu Ravi (2026). GeoRisk-RAG: A Hierarchy-Aware Risk Framework for Improving RAG Reliability through Selective Answering. CIKM 2026 (arXiv 2608.22634). https://arxiv.org/abs/2608.22634. Bears on B; related.
- Zihui Zhou (2026). RSHallu: Dual-Mode Hallucination Evaluation for Remote-Sensing Multimodal Large Language Models with Domain-Tailored Mitigation. arXiv 2602.10799. https://arxiv.org/abs/2602.10799. Bears on B; related.
- Yuxiang Zhang (2024). ToolBeHonest: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented Large Language Models. arXiv 2406.20015. https://arxiv.org/abs/2406.20015. Bears on B; related.
- Sewon Min (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023 (arXiv 2305.14251). https://arxiv.org/abs/2305.14251. Bears on B; does part.
- Jie Huang (2024). Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024 (arXiv 2310.01798). https://arxiv.org/abs/2310.01798. Bears on B; related.
- Ryo Kamoi (2024). When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs. TACL 12:1417-1440 (arXiv 2406.01297). https://arxiv.org/abs/2406.01297. Bears on B; related.

### grounding

- Arham Sethi (2026). Fabrication After Tool Failure: Tool-Augmented Agents Assert Values Their Tools Did Not Return. arXiv 2609.14758 (preprint, Sep 2026). https://arxiv.org/abs/2609.14758. Bears on B; does part.
- Mohammed Mehedi Hasan (2026). Model Context Protocol (MCP) Tool Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool Descriptions. arXiv 2602.14878 (preprint). https://arxiv.org/abs/2602.14878. Bears on B; related.
- Cheng-Yu Hsieh (2023). Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models. arXiv 2308.00675. https://arxiv.org/abs/2308.00675. Bears on B; related.
- Yuxiang Zhang (2024). ToolBeHonest: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented Large Language Models. arXiv 2406.20015. https://arxiv.org/abs/2406.20015. Bears on B; does part.
- Hongshen Xu (2024). Reducing Tool Hallucination via Reliability Alignment. arXiv 2412.04141. https://arxiv.org/abs/2412.04141. Bears on B; related.
- Abhinaba Basu (2026). Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents. arXiv 2603.10060 (preprint). https://arxiv.org/abs/2603.10060. Bears on B; does part.
- Enjun Du (2026). LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning with a Structured Evidence Ledger. arXiv 2607.28374 (preprint). https://arxiv.org/abs/2607.28374. Bears on B; does part.
- Ádám Kovács (2026). Beyond Document Grounding: Span-Level Hallucination Detection over Code, Tool Output, and Documents. arXiv 2607.00895 (preprint). https://arxiv.org/abs/2607.00895. Bears on B; does part.
- Jie Huang (2024). Large Language Models Cannot Self-Correct Reasoning Yet. ICLR 2024 (arXiv 2310.01798). https://arxiv.org/abs/2310.01798. Bears on B; related.
- Ryo Kamoi (2024). When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey of Self-Correction of LLMs. TACL 2024 (arXiv 2406.01297). https://arxiv.org/abs/2406.01297. Bears on B; related.
- Kaya Stechly (2024). On the Self-Verification Limitations of Large Language Models on Reasoning and Planning Tasks. arXiv 2402.08115 (ICLR per Semantic Scholar). https://arxiv.org/abs/2402.08115. Bears on B; related.
- Gladys Tyen (2024). LLMs cannot find reasoning errors, but can correct them given the error location. ACL 2024 Findings (arXiv 2311.08516). https://arxiv.org/abs/2311.08516. Bears on B; related.
- Arjun Panickssery (2024). LLM Evaluators Recognize and Favor Their Own Generations. NeurIPS 2024 (arXiv 2404.13076). https://arxiv.org/abs/2404.13076. Bears on B; related.
- Ken Tsui (2025). Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in Large Language Models. COLM 2026 (arXiv 2507.02778). https://arxiv.org/abs/2507.02778. Bears on B; related.
- Zhibin Gou (2024). CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. ICLR 2024 (arXiv 2305.11738). https://arxiv.org/abs/2305.11738. Bears on B; related.
- Shehzaad Dhuliawala (2023). Chain-of-Verification Reduces Hallucination in Large Language Models. arXiv 2309.11495. https://arxiv.org/abs/2309.11495. Bears on B; related.
- Sewon Min (2023). FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. EMNLP 2023 (arXiv 2305.14251). https://arxiv.org/abs/2305.14251. Bears on B; related.
- Miriam Wanner (2024). A Closer Look at Claim Decomposition. arXiv 2403.11903. https://arxiv.org/abs/2403.11903. Bears on B; related.
- Liyan Tang (2024). MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents. EMNLP 2024 (arXiv 2404.10774). https://arxiv.org/abs/2404.10774. Bears on B; related.
- Margaret Mitchell (2019). Model Cards for Model Reporting. FAT* 2019 (arXiv 1810.03993). https://arxiv.org/abs/1810.03993. Bears on B, C; related.

