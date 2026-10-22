# Related work

What this repository builds on, grouped by the question each group answers.
Each entry says what the work contributes and which experiment or claim here
touches it. Nothing on this page is claimed as ours; its purpose is the
opposite, to say where each idea comes from. Every entry was checked against
arXiv or Crossref before it went in.

## 1. Ranking a model's own errors

The core question here is selective classification: order predictions so the
least trustworthy come first, and measure the trade-off between coverage and
risk. That plain confidence is hard to beat is the field's default finding,
not a surprise, and the results here should be read in that light.

- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE
  Transactions on Information Theory* 16(1):41–46.
  [doi:10.1109/TIT.1970.1054406](https://doi.org/10.1109/TIT.1970.1054406).
  The reject option; a review budget is a reject rate.
- Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep
  neural networks. *NeurIPS*. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500).
  Risk-coverage curves for deep networks.
- Geifman, Y., Uziel, G. and El-Yaniv, R. (2019). Bias-reduced uncertainty
  estimation for deep neural classifiers. *ICLR*.
  [arXiv:1805.08206](https://arxiv.org/abs/1805.08206). Introduces excess AURC,
  the statistic `oe_inferencex.metrics.excess_aurc` implements.
- Hendrycks, D. and Gimpel, K. (2017). A baseline for detecting misclassified
  and out-of-distribution examples in neural networks. *ICLR*.
  [arXiv:1610.02136](https://arxiv.org/abs/1610.02136). Maximum softmax
  probability as the baseline; the margin used here is the same family.
- Jaeger, P. F., Lüth, C. T., Klein, L. and Bungert, T. J. (2023). A call to
  reflect on evaluation practices for failure detection in image
  classification. *ICLR*. [arXiv:2211.15259](https://arxiv.org/abs/2211.15259).
  Finds that most proposed failure-detection methods do not beat maximum
  softmax response under AURC. The general-ML version of what exp49, exp50 and
  exp70 found on Earth-observation maps.
- Corbière, C., Thome, N., Bar-Hen, A., Cord, M. and Pérez, P. (2019).
  Addressing failure prediction by learning model confidence. *NeurIPS*.
  [arXiv:1910.04851](https://arxiv.org/abs/1910.04851). A learned confidence
  head; the label-fitted fusion of exp65 is the same move with a linear model
  and a family lock.
- Scheffer, T., Decomain, C. and Wrobel, S. (2001). Active hidden Markov
  models for information extraction. *Advances in Intelligent Data Analysis*,
  LNCS 2189. [doi:10.1007/3-540-44816-0_31](https://doi.org/10.1007/3-540-44816-0_31).
  Margin sampling, the top-1 minus top-2 rule, originates here.
- Lewis, D. D. and Gale, W. A. (1994). A sequential algorithm for training
  text classifiers. *SIGIR*.
  [doi:10.1007/978-1-4471-2099-5_1](https://doi.org/10.1007/978-1-4471-2099-5_1).
  Uncertainty sampling; reviewing by confidence is uncertainty sampling with
  the labels used to grade rather than to train.
- Rottmann, M., Colling, P., Hack, T. P., Chan, R., Hüger, F., Schlicht, P.
  and Gottschalk, H. (2020). Prediction error meta classification in semantic
  segmentation: detection via aggregated dispersion measures of softmax
  probabilities. *IJCNN*. [arXiv:1811.00648](https://arxiv.org/abs/1811.00648),
  [doi:10.1109/IJCNN48605.2020.9206659](https://doi.org/10.1109/IJCNN48605.2020.9206659).
  Segment-level error prediction from softmax dispersion; the closest prior
  framing to the window-level protocol on segmentation.
- Madras, D., Pitassi, T. and Zemel, R. (2018). Predict responsibly: improving
  fairness and accuracy by learning to defer. *NeurIPS*.
  [arXiv:1711.06664](https://arxiv.org/abs/1711.06664). Deferral to a person.
- Mozannar, H. and Sontag, D. (2020). Consistent estimators for learning to
  defer to an expert. *ICML*. [arXiv:2006.01862](https://arxiv.org/abs/2006.01862).
  The comparison task's "decline" answer is a deferral.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Traub et al. (2024). Overcoming Common Flaws in the Evaluation of Selective Classification Systems. *NeurIPS*.
  [arXiv:2407.01032](https://arxiv.org/abs/2407.01032). Five requirements for a multi-threshold selective-classification metric, which AURC fails, and AUGRC, the average risk of undetected failures, which reorders methods on five of six datasets. This record reports excess AURC and capture at a budget; AUGRC is the statistic it should add before any ranking claim is called settled.
- Rabanser et al. (2025). What Does It Take to Build a Performant Selective Classifier?. *NeurIPS*.
  [arXiv:2510.20242](https://arxiv.org/abs/2510.20242). Decomposes the gap to a perfect ordering into Bayes noise, approximation, ranking, statistical and shift terms, and shows monotone post-hoc rescalings cannot move the ranking term. The theory behind what exp70 and exp73 found: every rescaling of confidence tried here was doomed a priori, and only feature-aware reorderings could have won.
- Heng et al. (2026). Know When to Abstain: Optimal Selective Classification with Likelihood Ratios. *ICLR*.
  [arXiv:2505.15008](https://arxiv.org/abs/2505.15008). The Neyman-Pearson view: the optimal selector is a likelihood ratio, of which the usual baselines are special cases, with gains reported under covariate shift. A likelihood-ratio selector is the one post-hoc family this record never scored against the margin; a candidate test, see section 7.
- Xia et al. (2025). Towards Understanding Why Label Smoothing Degrades Selective Classification and How to Fix It. *ICLR*.
  [arXiv:2403.14715](https://arxiv.org/abs/2403.14715). Label smoothing degrades selective classification by suppressing the top logit more on correct predictions than on wrong ones; a logit normalisation restores the ordering. The mechanism behind the warning here that the readout head's training regime decides whether its margin ranks errors.
- Zhou et al. (2025). A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators. *ICML*.
  [arXiv:2410.15361](https://arxiv.org/abs/2410.15361). A population definition of AURC with finite-sample bias and convergence rates for its estimator. This record puts intervals on excess AURC by block and cluster bootstrap and cited no estimator theory; this is the reference.
- Nguyen et al. (2025). Interpretable Failure Detection with Human-Level Concepts. *AAAI*.
  [arXiv:2502.05275](https://arxiv.org/abs/2502.05275). Argues category-level logit signals stay overconfident on failures and scores instead by the ordinal ranking of concept activations. Needs a concept model this pipeline does not have; noted as the concept-level challenge to a logit margin.
- Huang et al. (2026). Revisiting Confidence Calibration for Misclassification Detection in VLMs. *ICLR*.
  [OpenReview d8WMoi571f](https://openreview.net/forum?id=d8WMoi571f). Calibration and misclassification detection are different, partly opposed objectives. The record uses the distinction throughout without citing it: the margin is scored as a ranker, and expected calibration error is reported apart.
- Judeson et al. (2026). CMD: Class Margin Dispersion for Post-hoc Misclassification Detection. *ECCV 2026 workshop (DriveX)*.
  [OpenReview vXoAglgsmX](https://openreview.net/forum?id=vXoAglgsmX). Class margin dispersion: the whole margin distribution rather than top-1 minus top-2, reported to win on AURC. A workshop result, but the only published attack on this record's exact score on its exact metric; cheap to test on the 24 tasks, see section 7.

## 2. The signals tested against confidence

Every family below was scored here against the model's own margin and a
no-model control, and lost or tied
([Signals](results/signals.md), [Findings: what was tried and rejected](Findings.md#what-was-tried-and-rejected)).
The references say where each signal comes from; they are not evidence for it.

- Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
  representing model uncertainty in deep learning. *ICML*.
  [arXiv:1506.02142](https://arxiv.org/abs/1506.02142).
- Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). Simple and
  scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
  [arXiv:1612.01474](https://arxiv.org/abs/1612.01474). Ensembles; rejected here
  as bag and probe-seed disagreement (exp49, exp50).
- Kendall, A. and Gal, Y. (2017). What uncertainties do we need in Bayesian
  deep learning for computer vision? *NeurIPS*.
  [arXiv:1703.04977](https://arxiv.org/abs/1703.04977). Aleatoric and epistemic
  uncertainty for segmentation.
- Ovadia, Y. et al. (2019). Can you trust your model's uncertainty? Evaluating
  predictive uncertainty under dataset shift. *NeurIPS*.
- Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017). On calibration of
  modern neural networks. *ICML*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).
  Calibration is not ranking; a temperature does not change the order.
- Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M. and
  Hennig, P. (2021). Laplace redux: effortless Bayesian deep learning.
  *NeurIPS*. [arXiv:2106.14806](https://arxiv.org/abs/2106.14806). Laplace
  posterior over the head; rejected.
- Lee, K., Lee, K., Lee, H. and Shin, J. (2018). A simple unified framework for
  detecting out-of-distribution samples and adversarial attacks. *NeurIPS*.
  [arXiv:1807.03888](https://arxiv.org/abs/1807.03888). Mahalanobis feature
  typicality; rejected.
- Sun, Y., Ming, Y., Zhu, X. and Li, Y. (2022). Out-of-distribution detection
  with deep nearest neighbors. *ICML*.
  [arXiv:2204.06507](https://arxiv.org/abs/2204.06507). kNN typicality; rejected.
- Jiang, Y., Nagarajan, V., Baek, C. and Kolter, J. Z. (2022). Assessing
  generalization of SGD via disagreement. *ICLR*.
  [arXiv:2106.13799](https://arxiv.org/abs/2106.13799). Disagreement tracks the
  error rate in aggregate; it does not rank errors per window here.
- Milani Fard, M., Cormier, Q., Canini, K. and Gupta, M. (2016). Launch and
  iterate: reducing prediction churn. *NeurIPS*. Cross-version disagreement,
  the v1 against v1.2 comparison (exp19).
- Shanmugam, D., Blalock, D., Sahoo, G. and Guttag, J. (2021). Better
  aggregation in test-time augmentation. *ICCV*.
  [doi:10.1109/ICCV48922.2021.00125](https://doi.org/10.1109/ICCV48922.2021.00125).
  TTA; rejected (exp53, archived;
  [v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/docs/results/comparisons.md#test-time-adaptation-of-the-head-by-shift-consistency-exp53)).
- Bahat, Y. and Shakhnarovich, G. (2020). Classification confidence estimation
  with test-time data-augmentation. [arXiv:2006.16705](https://arxiv.org/abs/2006.16705).
  Augmentation consistency as confidence; the dihedral signal, rejected.
- Dawid, A. P. and Skene, A. M. (1979). Maximum likelihood estimation of
  observer error-rates using the EM algorithm. *Applied Statistics*
  28(1):20–28. [doi:10.2307/2346806](https://doi.org/10.2307/2346806).
  Label-free reliability from several raters; rejected because heads of one
  family err together.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Guarino et al. (2026). Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance. *CVPR*.
  [arXiv:2603.29941](https://arxiv.org/abs/2603.29941). Pixel uncertainty must be aggregated before it can flag a region, and aggregators that use spatial structure beat the global average on failure detection across ten datasets. This record's window margin is the margin of the window-mean probabilities, an aggregator chosen without a test; the first thing to try if the review set is to improve, see section 7.
- Borges et al. (2026). Soft Dice Confidence: A Near-Optimal Confidence Estimator for Selective Prediction in Semantic Segmentation. *Machine Learning*.
  [doi:10.1007/s10994-026-07096-w](https://doi.org/10.1007/s10994-026-07096-w). Soft Dice Confidence, a near-optimal confidence aggregator for selective prediction in segmentation. The second candidate aggregator for the 4-px window, with an optimality argument the current mean lacks.
- Rey et al. (2025). Uncertainty evaluation of segmentation models for Earth observation. *arXiv preprint*.
  [arXiv:2510.19586](https://arxiv.org/abs/2510.19586). Scores per-pixel uncertainty for Earth-observation segmentation by whether it finds errors and corrupted input, on PASTIS and ForTy, with stochastic segmentation networks and ensembles. Runs this record's evaluation on one of its own tasks with a method family it never included.
- Lehmann et al. (2026). Beyond Accuracy: Assessing Calibration of Geospatial Foundation Models and Their Sensitivity to Distribution Shifts. *arXiv preprint*.
  [arXiv:2608.16614](https://arxiv.org/abs/2608.16614). Sixteen frozen encoders, nine datasets, two shift axes, scored on calibration: every encoder degrades and the ranking of encoders changes under shift, and deep ensembles do not help. The confidence-side twin of exp74; it agrees that ensembles add nothing and predicts where the margin should weaken; of the two shifts of year and place applied here, the first (exp99, exp103) saw a fine-tuned model's confidence miss its ranking bar, consistent in direction, and the second, a blind test at fresh plots farther away (exp111), did not repeat the miss; no corruption was applied.
- Johnson et al. (2026). Calibrated spatial uncertainty for Earth observation foundation models via Matérn-motivated latent stochastic regularization. *Remote Sensing of Environment*.
  [doi:10.1016/j.rse.2026.115610](https://doi.org/10.1016/j.rse.2026.115610). Calibrated spatial uncertainty for Earth-observation foundation models with a Matérn-motivated latent. A spatially aware uncertainty estimator for this model class; whether it ranks errors better than the margin is untested and would need the same protocol.
- Liu et al. (2026). Adaptive Confidence Regularization for Multimodal Failure Detection. *CVPR*.
  [arXiv:2603.02200](https://arxiv.org/abs/2603.02200). Multimodal failure detection from the gap between the fused head's confidence and each branch's, with a regularizer that trains the gap to be informative. Names the mechanism exp75 measured post hoc: heads on other sensors disagree informatively and, without that training, do not pay.
- Doorenbos et al. (2026). Modality-Aware Out-of-Distribution Detection for Multi-modal Action Recognition. *ECCV*.
  [arXiv:2606.24404](https://arxiv.org/abs/2606.24404). Out-of-distribution detection from the relationship between multi-modal and uni-modal predictions, combined with a feature-space score. The closest published statement of exp75's hypothesis, on action recognition rather than maps.
- Mena et al. (2025). Multi-modal Co-learning for Earth Observation: Enhancing single-modality models via modality collaboration. *Machine Learning*.
  [arXiv:2510.19579](https://arxiv.org/abs/2510.19579). Multi-modal co-learning for Earth observation: single-modality models improved by modality collaboration, with a split into modality-shared and modality-specific parts. The construction exp75 uses, heads on other sensors of the same units, and a hypothesis for why a weak sensor head's disagreement is noise.
- Goel et al. (2025). Great Models Think Alike and this Undermines AI Oversight. *ICML*.
  [arXiv:2502.04313](https://arxiv.org/abs/2502.04313). CAPA, a chance-adjusted agreement statistic defined on the overlap of two models' mistakes, correcting for the agreement accuracy alone produces. The statistic exp75's vote disagreement should have been, since raw disagreement is confounded by the heads' accuracies; see section 7.
- Kim et al. (2025). Correlated Errors in Large Language Models. *ICML*.
  [arXiv:2506.07962](https://arxiv.org/abs/2506.07962). Errors of 350 language models are correlated, more so as accuracy rises. The large-sample version of this record's finding that errors are shared across models and come from the input.
- Klein et al. (2025). Quantifying Uncertainty in Error Consistency: Towards Reliable Behavioral Comparison of Classifiers. *NeurIPS*.
  [doi:10.52202/085713-1702](https://doi.org/10.52202/085713-1702). Confidence intervals for error consistency between classifiers. The inference this record lacks for its shared-error finding and for exp75's disagreement readings.
- Plas et al. (2026). Better Together: Evaluating the Complementarity of Earth Embedding Models. *arXiv preprint*.
  [arXiv:2605.18667](https://arxiv.org/abs/2605.18667). Complementarity of Earth embedding models across families. This record established shared errors within one family; cross-family complementarity is the untested half.
- Shah et al. (2026). Embeddings based Anomaly Detection for Cleaning Global Crop Type Reference Datasets. *ECCV 2026 workshop (TerraBytes)*.
  [arXiv:2607.23908](https://arxiv.org/abs/2607.23908). Locality-aware embedding anomaly detection to clean crop-type reference labels for WorldCereal. The nearest-neighbour typicality this record rejected as a ranker of model errors, pointed at the reference instead, where it works; the rejection is of one use, not of the reading.
- Opravil et al. (2026). Consensus land-cover mapping improves grassland classification in European mountain landscapes. *Scientific Reports*.
  [doi:10.1038/s41598-026-39197-w](https://doi.org/10.1038/s41598-026-39197-w). Consensus of independent land-cover maps improves grassland classification in mountain landscapes. This record rejected multi-rater fusion for raters of one model family; with independent raters the fusion pays, so the rejection is scoped to the family.
- Paplhám et al. (2026). Evaluating Epistemic Uncertainty: Beyond OOD Detection and Active Learning. *arXiv preprint*.
  [arXiv:2607.14817](https://arxiv.org/abs/2607.14817). Evaluates epistemic-uncertainty estimators beyond OOD detection and active learning. A framework in which every signal rejected here is an epistemic estimator scored on a task it was not built for.
- Landgraf et al. (2026). A Critical Synthesis of Uncertainty Quantification and Foundation Models for Semantic Segmentation. *ISPRS Annals*.
  [doi:10.5194/isprs-annals-XI-2-2026-673-2026](https://doi.org/10.5194/isprs-annals-XI-2-2026-673-2026). A synthesis of uncertainty quantification for foundation-model segmentation, run on a frozen encoder with a small decoder: Monte Carlo dropout, sub-ensembles and test-time augmentation. The same readout regime as here, with the same families that lost to confidence.
- Choi et al. (2026). Uncertainty-Based Quality-Control Prioritization for High-Resolution Land-Cover Mapping Under Spatially Disjoint Evaluation. *Remote Sensing (MDPI)*.
  [doi:10.3390/rs18183197](https://doi.org/10.3390/rs18183197). Uncertainty-based quality-control prioritisation for land-cover mapping: plain confidence ties a 20-pass Monte Carlo ensemble at finding errors, and suspicion-ordered selection can underperform random. An independent replication of two results recorded here, published the day this page was updated.

## 3. Why signals from inside the encoder carry no error information

The retargeted-residual line asked whether the pretraining objective's own
error could locate task errors. It could not, and the representation-learning
literature says why: a reconstruction or masking objective tracks the data's
variance subspace, which need not align with what a task needs.

- Esser, P. M., Fleissner, M. and Ghoshdastidar, D. (2025). Theoretical
  foundations of representation learning using unlabeled data: statistics and
  optimization. [arXiv:2509.18997](https://arxiv.org/abs/2509.18997). A survey:
  reconstruction learns principal components, which need not be the
  task-relevant directions; every downstream bound assumes a linear probe on
  the frozen representation, which makes the probe's own regularisation a
  separate factor (the exp68 warning).
- Balestriero, R. and LeCun, Y. (2024). Learning by reconstruction produces
  uninformative features for perception.
  [arXiv:2402.11337](https://arxiv.org/abs/2402.11337).
- Jing, L., Vincent, P., LeCun, Y. and Tian, Y. (2022). Understanding
  dimensional collapse in contrastive self-supervised learning. *ICLR*.
  [arXiv:2110.09348](https://arxiv.org/abs/2110.09348). The vocabulary for the
  frozen target's effective rank of 2.
- Torralba, A., Isola, P. and Freeman, W. T. (2024). *Foundations of Computer
  Vision*, chapter 30, Representation learning. MIT Press.
  [visionbook.mit.edu](https://visionbook.mit.edu/representation_learning.html).
  Compression against prediction; a representation good for one task can be
  bad for another.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P. and Girshick, R. (2022).
  Masked autoencoders are scalable vision learners. *CVPR*.
  [arXiv:2111.06377](https://arxiv.org/abs/2111.06377).
- Wei, Y. et al. (2024). Towards latent masked image modeling for
  self-supervised visual representation learning.
  [arXiv:2407.15837](https://arxiv.org/abs/2407.15837).
- Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J. and Auli, M. (2022).
  data2vec: a general framework for self-supervised learning in speech, vision
  and language. *ICML*. [arXiv:2202.03555](https://arxiv.org/abs/2202.03555).
- Hsu, W.-N. et al. (2021). HuBERT: self-supervised speech representation
  learning by masked prediction of hidden units. *IEEE/ACM TASLP*.
  [arXiv:2106.07447](https://arxiv.org/abs/2106.07447). The last three are the
  retarget recipes tried and rejected.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Kondylatos et al. (2025). On the Generalization of Representation Uncertainty in Earth Observation. *ICCV*.
  [arXiv:2503.07082](https://arxiv.org/abs/2503.07082). Pretrained representation uncertainty, an encoder-internal zero-shot reading, transplanted to Earth observation, where it behaves unlike its natural-image counterpart. The live challenger to this record's rejected encoder-internal bucket; it was not among the readings exp17 tested.
- Corley et al. (2026). From Pixels to Patches: Pooling Strategies for Earth Embeddings. *ICLR 2026 workshop (ML4RS)*.
  [arXiv:2603.02080](https://arxiv.org/abs/2603.02080). Eleven training-free pooling methods over downloaded embedding fields (AlphaEarth, OlmoEarth, TESSERA): mean pooling is the worst under a geographic split. This record pools patches into 4-px windows by mean; the choice is a variable, not a given.

## 4. Reliability and evaluation of spatial and Earth-observation models

- Gonzalez-Calabuig, M. et al. (2025). SHRUG-FM: reliability-aware foundation
  models for Earth observation. [arXiv:2511.10370](https://arxiv.org/abs/2511.10370).
  Image-level reliability signals with a label-fitted gate; accepted for the
  CVPR EarthVision 2026 proceedings. Ported to the window and scored here
  (exp49). Its revised abstract reports beating single-signal baselines such
  as predictive entropy, so the earlier note here, that it runs no
  single-model confidence baseline, is out of date (literature search of 27
  September 2026).
- Meyer, H. and Pebesma, E. (2021). Predicting into unknown space? Estimating
  the area of applicability of spatial prediction models. *Methods in Ecology
  and Evolution* 12:1620–1633.
  [doi:10.1111/2041-210X.13650](https://doi.org/10.1111/2041-210X.13650).
  Distance-to-training as a warning flag; the OlmoEarth Agent's uncertainty
  skill implements it.
- Meyer, H. and Pebesma, E. (2022). Machine learning-based global maps of
  ecological variables and the challenge of assessing them. *Nature
  Communications* 13:2208.
  [doi:10.1038/s41467-022-29838-9](https://doi.org/10.1038/s41467-022-29838-9).
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive
  performance of large-scale ecological mapping models. *Nature
  Communications* 11:4540.
  [doi:10.1038/s41467-020-18321-y](https://doi.org/10.1038/s41467-020-18321-y).
  Why the splits here are spatial.
- Milà, C., Mateu, J., Pebesma, E. and Meyer, H. (2022). Nearest neighbour
  distance matching leave-one-out cross-validation for map validation.
  *Methods in Ecology and Evolution* 13:1304–1316.
  [doi:10.1111/2041-210X.13851](https://doi.org/10.1111/2041-210X.13851).
- Olofsson, P., Foody, G. M., Herold, M., Stehman, S. V., Woodcock, C. E. and
  Wulder, M. A. (2014). Good practices for estimating area and assessing
  accuracy of land change. *Remote Sensing of Environment* 148:42–57.
  [doi:10.1016/j.rse.2014.02.015](https://doi.org/10.1016/j.rse.2014.02.015).
  The design-weighted estimators in `oe_inferencex.metrics` follow this
  practice.
- Stehman, S. V. and Foody, G. M. (2019). Key issues in rigorous accuracy
  assessment of land cover products. *Remote Sensing of Environment*
  231:111199. [doi:10.1016/j.rse.2019.05.018](https://doi.org/10.1016/j.rse.2019.05.018).
- Tuia, D., Volpi, M., Copa, L., Kanevski, M. and Muñoz-Marí, J. (2011). A
  survey of active learning algorithms for supervised remote sensing image
  classification. *IEEE Journal of Selected Topics in Signal Processing*
  5(3):606–617. [doi:10.1109/JSTSP.2011.2139193](https://doi.org/10.1109/JSTSP.2011.2139193).
  Uncertainty sampling in Earth observation.
- Kampffmeyer, M., Salberg, A.-B. and Jenssen, R. (2016). Semantic
  segmentation of small objects and modeling of uncertainty in urban remote
  sensing images using deep convolutional neural networks. *CVPR Workshops*.
  [doi:10.1109/CVPRW.2016.90](https://doi.org/10.1109/CVPRW.2016.90). Early
  per-pixel uncertainty maps in Earth observation.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Xu et al. (2024). Comparative validation of recent 10 m-resolution global land cover maps. *Remote Sensing of Environment*.
  [doi:10.1016/j.rse.2024.114316](https://doi.org/10.1016/j.rse.2024.114316). Independent validation of ESRI, WorldCover and Dynamic World in 47 countries, with five ways of handling reference-data uncertainty from geolocation and labelling error. The measured reason WorldCover cannot serve as truth at 10 m, which is where this record's open question lives.
- Tyukavina et al. (2025). Practical global sampling methods for estimating area and map accuracy of land cover and change. *Remote Sensing of Environment*.
  [doi:10.1016/j.rse.2025.114714](https://doi.org/10.1016/j.rse.2025.114714). Practical global sampling for area and accuracy of land cover and change maps, the current successor to the 2014 good-practice recommendations this record's design-weighted estimators follow.
- Johnson et al. (2025). From pixels to parcels: Flexible, practical small-area uncertainty estimation for spatial averages obtained from aboveground biomass maps. *Remote Sensing of Environment*.
  [doi:10.1016/j.rse.2025.114951](https://doi.org/10.1016/j.rse.2025.114951). Small-area uncertainty for spatial averages of map products: at small scales the spatially correlated residual dominates. The check this record's block and cluster bootstraps at window scale should be held to.
- Biase et al. (2025). Design-based mapping of errors in remote sensing-based land use/land cover maps. *Stochastic Environmental Research and Risk Assessment*.
  [doi:10.1007/s00477-025-02908-2](https://doi.org/10.1007/s00477-025-02908-2). Design-based mapping of where a land-cover map's errors are. The labelled counterpart of this record's product, and the baseline a label-free ranking should be shown beside.
- Shimizu et al. (2025). Quantifying consistency among interpreters of reference data for estimation of harvest area through visual interpretation of remote sensing data. *Forestry*.
  [doi:10.1093/forestry/cpaf077](https://doi.org/10.1093/forestry/cpaf077). Consistency among interpreters of reference data: a single-interpreter reference makes design-based variances optimistic. The current source for this record's imperfect-reference caveat.
- Xiao et al. (2024). The illusion of success: Test set disproportion causes inflated accuracy in remote sensing mapping research. *Int. J. Applied Earth Observation and Geoinformation*.
  [doi:10.1016/j.jag.2024.104256](https://doi.org/10.1016/j.jag.2024.104256). Test sets with curated class composition inflate reported accuracy relative to landscape composition. A caveat on the 24-task suite: its splits are the model authors', not a landscape sample.
- Matos et al. (2025). Accounting for alternation in temporal quality analysis in MapBiomas Brazil. *International Journal of Digital Earth*.
  [doi:10.1080/17538947.2025.2528604](https://doi.org/10.1080/17538947.2025.2528604). Alternation in MapBiomas: the map changing where the ground did not, given a name, an estimator and a probability sample. The published form of the 6 to 37% movement this record measured on crops.
- Ricci et al. (2026). Mitigating Negative Flips via Margin Preserving Training. *AAAI*.
  [doi:10.1609/aaai.v40i11.37825](https://doi.org/10.1609/aaai.v40i11.37825). Negative flips, predictions a new model version gets wrong where the old one was right, and margin-preserving training to reduce them. The vocabulary for the version axis of the compare module.
- Jin et al. (2025). Confidence on the focal: conformal prediction with selection-conditional coverage. *Journal of the Royal Statistical Society B*.
  [doi:10.1093/jrsssb/qkaf016](https://doi.org/10.1093/jrsssb/qkaf016). Conformal coverage conditional on a unit having been selected by a data-driven rule. A review set is such a selection, so any guarantee attached to it is selection-conditional; the construction to use if this record ever states one.
- Farinhas et al. (2024). Non-Exchangeable Conformal Risk Control. *ICLR*.
  [arXiv:2310.01262](https://openreview.net/forum?id=j511LaqEeP). Conformal risk control without exchangeability, with weights for dependent data. The route to a false-negative-rate guarantee on a review set drawn from spatially dependent windows.
- Schweden et al. (2026). Assessing Conformal Prediction for Remote Sensing: Application to Local Climate Zone Classification. *IEEE TGRS*.
  [doi:10.1109/tgrs.2026.3719780](https://doi.org/10.1109/tgrs.2026.3719780). Conformal prediction assessed on local climate zone classification. The one uncertainty contract this record never scored against the margin, on a task of the same kind.
- Li et al. (2026). Quantifying and Communicating Uncertainty in SAR-Based Flood Mapping via Density-Aware Neural Networks and Conformal Risk Control. *IEEE TGRS*.
  [doi:10.1109/tgrs.2026.3661208](https://doi.org/10.1109/tgrs.2026.3661208). Density-aware networks for SAR flood mapping with quantified, communicated uncertainty. The closest published answer to turning a ranked flood map into a risk-controlled one.
- Kuronen et al. (2025). Uncertainty quantification for forest attribute maps with conformal prediction and k-nearest neighbor method. *Remote Sensing of Environment*.
  [doi:10.1016/j.rse.2025.114758](https://doi.org/10.1016/j.rse.2025.114758). Conformal prediction with k-nearest neighbours for forest attribute maps. The precedent that distribution-free sets are an accepted uncertainty contract for maps.
- Boyeau et al. (2025). AutoEval Done Right: Using Synthetic Data for Model Evaluation. *ICML*.
  [arXiv:2403.07008](https://arxiv.org/abs/2403.07008). Prediction-powered evaluation: a wall-to-wall model output combined with a small gold-standard sample for unbiased estimates. The machine-learning source for what this record's design-weighted estimators already do.
- Mozer et al. (2026). PPI is the Difference Estimator: Recognizing the Survey Sampling Roots of Prediction-Powered Inference. *arXiv preprint*.
  [arXiv:2603.19160](https://arxiv.org/abs/2603.19160). Prediction-powered inference is the survey-sampling difference estimator. The one sentence that ties this record's estimators to the autoeval literature.
- Hamilton et al. (2026). Scalable Model-Assisted Multi-Target Estimation in Large Image Collections. *arXiv preprint*.
  [arXiv:2607.17581](https://arxiv.org/abs/2607.17581). Model-assisted estimation of many population quantities from one image collection under an annotation budget: uniform sampling with control variates wins at many targets or few labels. Current practice for the estimator side of a review budget.
- Yu et al. (2025). ODP-Bench: Benchmarking Out-Of-Distribution Performance Prediction. *ICCV*.
  [arXiv:2510.27263](https://arxiv.org/abs/2510.27263). ODP-Bench, a benchmark for predicting out-of-distribution accuracy without labels. Dataset-level, where this record's claim is window-level; the framing statement the page needed.
- Marsocci et al. (2025). PANGAEA: Assessing Geospatial Foundation Models Capabilities through a Global and Inclusive Benchmark. *IEEE Geoscience and Remote Sensing Magazine*.
  [doi:10.1109/MGRS.2025.3628194](https://doi.org/10.1109/MGRS.2025.3628194). PANGAEA: geospatial foundation-model evaluation is narrow and geographically biased, and foundation models do not consistently beat supervised baselines. The external-validity caveat on the 24-task suite's spatial support.
- Corley et al. (2026). No One Knows the State of the Art in Geospatial Foundation Models. *arXiv preprint*.
  [arXiv:2605.12678](https://arxiv.org/abs/2605.12678). No one knows the state of the art in geospatial foundation models: reported results are not comparable across papers. The justification for this record's preregistration and claim ledger.
- Zhu et al. (2026). On the foundations of Earth foundation models. *Communications Earth & Environment*.
  [doi:10.1038/s43247-025-03127-x](https://doi.org/10.1038/s43247-025-03127-x). On the foundations of Earth foundation models: what their evaluation should cover. Auditing a map without labels is one axis it names and does not fill.
- Li et al. (2025). REOBench: Benchmarking Robustness of Earth Observation Foundation Models. *NeurIPS Datasets and Benchmarks*.
  [arXiv:2505.16793](https://arxiv.org/abs/2505.16793). REOBench: Earth-observation foundation models under twelve corruptions, including geometric ones, with drops from under 1 to over 20 points. The benchmark next to which this record's rejected tiling-instability reading belongs.
- Sialelli et al. (2026). From Machine Learning to Large-Scale EO Products: Best Practices for Making Maps. *ECCV 2026 workshop (TerraBytes)*.
  [arXiv:2607.24532](https://arxiv.org/abs/2607.24532). Best practices for turning models into large-scale Earth-observation products, citing OlmoEarth and the good-practice validation literature. The community statement of the problem this repository works on.
- Romero et al. (2026). How do Self-Supervised Remote Sensing Vision Models Transfer to Downstream Tasks?. *arXiv preprint*.
  [arXiv:2606.13896](https://arxiv.org/abs/2606.13896). How self-supervised remote-sensing models transfer, on PASTIS and Sen1Floods11 among others. External support for the readout-gap warning on two of this record's own tasks.

### Added 13 October 2026

From a prior-art check of 13 October 2026 (titles and abstracts; each entry checked against its publisher, ASPRS,
Crossref or arXiv record).

- Gopal, S. and Woodcock, C. (1994). Theory and methods for accuracy assessment of thematic maps using fuzzy sets.
  *Photogrammetric Engineering and Remote Sensing* 60(2):181–188.
  [ASPRS](https://asprs.org/wp-content/uploads/pers/1994journal/feb/1994_feb_181-188.pdf). Fuzzy accuracy
  assessment: a map location may belong acceptably to more than one class. The package's `--also-accept` reading
  (the per-class audit, `estimate` and `certify`) is this idea applied as a fixed lenient crosswalk.
- Valle, D., Izbicki, R. and Leite, R. V. (2023). Quantifying uncertainty in land-use land-cover classification using
  conformal statistics. *Remote Sensing of Environment* 295:113682.
  [doi:10.1016/j.rse.2023.113682](https://doi.org/10.1016/j.rse.2023.113682). Per-pixel conformal prediction sets
  for land cover with marginal coverage; the certified zone instead bounds the error rate inside one map's zone.
- Lu, K., Kluger, D. M., Bates, S. and Wang, S. (2024). Regression coefficient estimation from remote sensing maps.
  [arXiv:2407.13659](https://arxiv.org/abs/2407.13659). Prediction-powered inference with remote-sensing maps and a
  small labelled sample, compared with design-based estimation; asymptotic, where the package's intervals are exact
  over the finite population.

## 5. The models and the testbeds

- Herzog, H. et al. (2025). OlmoEarth: stable latent image modeling for
  multimodal Earth observation. [arXiv:2511.13655](https://arxiv.org/abs/2511.13655).
  The encoder, and the published task suite exp70 runs on.
- Lacoste, A. et al. (2023). GEO-Bench: toward foundation models for Earth
  monitoring. *NeurIPS Datasets and Benchmarks*.
  [arXiv:2306.03831](https://arxiv.org/abs/2306.03831). Six of exp70's tasks.
- Bonafilia, D., Tellman, B., Anderson, T. and Issenberg, E. (2020).
  Sen1Floods11: a georeferenced dataset to train and test deep learning flood
  algorithms for Sentinel-1. *CVPR Workshops*.
  [doi:10.1109/CVPRW50498.2020.00113](https://doi.org/10.1109/CVPRW50498.2020.00113).
- Chiriaco, F. et al. (2026). GEOID-Flood: a large-scale multi-modal benchmark
  dataset for flood segmentation. [arXiv:2608.02315](https://arxiv.org/abs/2608.02315).
  Copernicus EMS flood extents as chips; exp55, exp57, exp60, exp62.
- Mateo-Garcia, G. et al. (2021). Towards global flood mapping onboard low
  cost satellites with machine learning. *Scientific Reports* 11:7249.
  [doi:10.1038/s41598-021-86650-z](https://doi.org/10.1038/s41598-021-86650-z).
  WorldFloods.
- Portalés-Julià, E., Mateo-García, G., Purcell, C. and Gómez-Chova, L.
  (2023). Global flood extent segmentation in optical satellite images.
  *Scientific Reports* 13.
  [doi:10.1038/s41598-023-47595-7](https://doi.org/10.1038/s41598-023-47595-7).
  WorldFloods v2, the fourth cell of exp62.
- Yokoya, N., Ghamisi, P., Hänsch, R. and Schmitt, M. (2020). 2020 IEEE GRSS
  Data Fusion Contest: global land cover mapping with weak supervision. *IEEE
  Geoscience and Remote Sensing Magazine* 8(1).
  [doi:10.1109/MGRS.2020.2970124](https://doi.org/10.1109/MGRS.2020.2970124).
  DFC2020, exp66.
- Brown, C. F. et al. (2022). Dynamic World, near real-time global 10 m land
  use land cover mapping. *Scientific Data* 9:251.
  [doi:10.1038/s41597-022-01307-4](https://doi.org/10.1038/s41597-022-01307-4).
  The served product of exp67.
- d'Andrimont, R. et al. (2021). LUCAS Copernicus 2018: Earth-observation-
  relevant in situ data on land cover and use throughout the European Union.
  *Earth System Science Data* 13:1119–1133.
  [doi:10.5194/essd-13-1119-2021](https://doi.org/10.5194/essd-13-1119-2021).
  The 2022 release is exp68's reference.
- Schneider, M., Schelte, T., Schmitz, F. and Körner, M. (2023). EuroCrops:
  the largest harmonized open crop dataset across the European Union.
  *Scientific Data* 10. [doi:10.1038/s41597-023-02517-0](https://doi.org/10.1038/s41597-023-02517-0).
  exp69.
- Astruc, G., Gonthier, N., Mallet, C. and Landrieu, L. (2025). AnySat: one
  Earth observation model for many resolutions, scales, and modalities.
  *CVPR*. [arXiv:2412.14123](https://arxiv.org/abs/2412.14123). The outside
  witness, rejected.
- Zanaga, D. et al. (2021, 2022). ESA WorldCover 10 m 2020 and 2021. Zenodo.
  The reference of exp18 and exp23.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Jakubik et al. (2025). TerraMind: Large-Scale Generative Multimodality for Earth Observation. *ICCV*.
  [arXiv:2504.11171](https://arxiv.org/abs/2504.11171). TerraMind, an any-to-any generative multimodal Earth-observation model that can synthesise a missing sensor. It would make exp75 general: a second sensor for any cell, not only where one was published.
- Doerksen et al. (2026). EarthShift: a benchmark for measuring robustness to real-world distribution shifts in Earth observation. *arXiv preprint*.
  [arXiv:2605.29330](https://arxiv.org/abs/2605.29330). EarthShift, paired datasets measuring robustness to real-world distribution shift. A paired-reference construction of the kind the exp27 verdict asked for (exp27 is archived; [v1.9.0](https://github.com/2imi9/olmoearth_inferenceX/blob/v1.9.0/exp/NOTES.md#exp27-oracle-gate-2026-09-06)), aimed at shift.
- Gordon et al. (2026). MMEarth-Bench: Global Model Adaptation via Multimodal Test-Time Training. *ECCV*.
  [arXiv:2602.06285](https://arxiv.org/abs/2602.06285). MMEarth-Bench, geographically split multimodal test-time adaptation with label budgets. A testbed on which the label-budget results here could be replicated.
- Kaushik et al. (2026). Assessing Geo-Foundational Models for Flood Inundation Mapping: Benchmarking Models for Sentinel-1, Sentinel-2, and Planetscope. *IEEE JSTARS*.
  [doi:10.1109/JSTARS.2026.3656855](https://doi.org/10.1109/JSTARS.2026.3656855). Geo-foundation models benchmarked for Sentinel-1 flood mapping. Reproduces, on the same task, that swapping foundation models buys little.

### Added 22 October 2026

Embedding products, after Ai2 released OlmoEarth v1.3 (each entry checked against its arXiv record or publisher page).

- Brown, C. F. et al. (2025). AlphaEarth Foundations: an embedding field model for accurate and efficient global
  mapping from sparse label data. [arXiv:2507.22291](https://arxiv.org/abs/2507.22291). Google DeepMind's annual
  64-dimensional embedding field. Not evaluated in this record; Corley et al. (2026, section 3 above) pool over it.
- Team OlmoEarth (2026). OlmoEarth v1.3 Embeddings. Technical report, Allen Institute for AI.
  [allenai.org/papers/olmoearth-v1-3](https://allenai.org/papers/olmoearth-v1-3). Precomputed annual 10 m
  embeddings, 128 or 64 dimensions, on [Source Cooperative](https://source.coop/olmoearth/olmoearth-v1-3-embeddings).
  The report evaluates the embeddings by readout accuracy on class-balanced draws, against AlphaEarth and TESSERA v2;
  it gives no confidence layer and no accuracy of a map made from them. Its section 4.3 notes that 2025, most of 2024
  and small parts of 2019 and 2023 were computed before a fix for a seam every 20.48 km. The package's Usage page
  shows how to check a classifier map made from them; this record has not graded that route.

## 6. Whether the tool helps a person

- Bansal, G., Wu, T., Zhou, J., Fok, R., Nushi, B., Kamar, E., Ribeiro, M. T.
  and Weld, D. S. (2021). Does the whole exceed its parts? The effect of AI
  explanations on complementary team performance. *CHI*.
  [doi:10.1145/3411764.3445717](https://doi.org/10.1145/3411764.3445717). Why
  exp64 grades grounding and task accuracy separately.

### Added 18 September 2026

From a search of the 2025 and 2026 literature; each entry was checked against its arXiv, Crossref or OpenAlex record before it went in.

- Kao et al. (2026). Towards LLM Agents for Earth Observation. *Findings of ACL*.
  [arXiv:2504.12110](https://arxiv.org/abs/2504.12110). UnivEARTH: agents answering Earth-observation questions through Earth Engine reach 33%, and the code fails to run more than 58% of the time. Explains the exp64 shape: much of a sandbox arm's gap is execution failure, which a curated tool removes and a stronger model closes.
- Feng et al. (2026). Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents. *ICLR*.
  [arXiv:2509.23141](https://arxiv.org/abs/2509.23141). Earth-Agent: agents given real Earth-observation tools rather than a code sandbox. The anchor for the tool arm of exp64.
- Yu et al. (2026). Benchmarking LLM Tool-Use in the Wild (WildToolBench). *ICLR*.
  [arXiv:2604.06185](https://arxiv.org/abs/2604.06185). WildToolBench: tool-use progress on clean single-intent prompts is spurious in the wild. exp64's cards are clean and single-intent; the caveat on its external validity.
- Qiao et al. (2026). Scaling Generalist Data-Analytic Agents (DataMind). *ICLR*.
  [arXiv:2509.25084](https://arxiv.org/abs/2509.25084). DataMind: unstable multi-turn code rollouts are a distinct, fixable failure of data-analytic agents. The confound to name in exp64's sandbox arm.
- Zhao et al. (2026). OpenEarth-Agent: From Tool Calling to Tool Creation for Open-Environment Earth Observation. *arXiv preprint*.
  [arXiv:2603.22148](https://arxiv.org/abs/2603.22148). OpenEarth-Agent: from tool calling to tool creation. A sandbox that persists its output as a tool is a third arm exp64 did not have.
- Shen et al. (2026). SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents. *ICML*.
  [arXiv:2602.12984](https://arxiv.org/abs/2602.12984). SciAgentGym: tiered scientific tool use, elementary actions then composed workflows. A template for recutting exp64 by difficulty.
- Manzini et al. (2026). Looks Can be Deceiving: Annotator and Reviewer Performance Across Imagery Sources in Crowd-Sourced Aerial Damage Assessment. *arXiv preprint*.
  [arXiv:2608.14942](https://arxiv.org/abs/2608.14942). Annotator and reviewer performance differs by imagery source in crowd-sourced mapping. A measured human prior for the review budget: the reviewer's own error rate depends on what the ranking sends them to.
- Lüth et al. (2025). nnActive: A Framework for Evaluation of Active Learning in 3D Biomedical Segmentation. *TMLR*.
  [arXiv:2511.19183](https://arxiv.org/abs/2511.19183). nnActive: in 3D biomedical segmentation every active-learning method beats naive random and none reliably beats a task-adapted random baseline. The strongest corroboration of exp56, and the counter-argument to pre-empt: the random baseline must be a fair one.
- Lüth et al. (2026). Finally Outshining the Random Baseline: A Simple and Effective Solution for Active Learning in 3D Biomedical Imaging. *TMLR*.
  [arXiv:2601.13677](https://arxiv.org/abs/2601.13677). Uncertainty-driven selection beats random once class imbalance and early-round redundancy are handled. The sharpest challenge to exp56's rejection; the fix it prescribes was not applied there.
- Machnio et al. (2025). To Label or Not to Label: PALM - a Predictive Model for Evaluating Sample Efficiency in Active Learning Models. *ICCV*.
  [doi:10.1109/ICCV51701.2025.00385](https://doi.org/10.1109/ICCV51701.2025.00385). PALM, a predictive model of active-learning sample efficiency. A way to state exp56 as a fitted curve rather than a per-budget table.
- Kay et al. (2025). Consensus-Driven Active Model Selection. *ICCV*.
  [arXiv:2507.23771](https://arxiv.org/abs/2507.23771). Consensus-driven active model selection with a label budget. A relative of the compare module, and a challenge to the finding that a second model of the same family adds nothing.

## 7. Published challenges to this record, and what it answers

Added 18 September 2026. The search above was run in part to find published
claims that contradict a recorded finding. Each row names the challenge, what
the record already says, and whether the question is answered, open or
untested. Nothing here changes a claim; the untested rows are the experiments
the record would need next, with their cost.

| Challenge | What it claims | What the record says | Status |
|---|---|---|---|
| Traub et al. 2024, AUGRC | AURC ranks methods wrongly on five of six datasets; AUGRC is the interpretable statistic | Every ranking claim here is excess AURC plus capture at a budget; AUGRC was never computed | Answered (exp76 and exp70's artifact): for fixed errors AUGRC orders readings as the failure AUROC does; the margin beats the best control on 23 of 24 by AUROC, and the best reading is the same under both statistics on 22 of 24 |
| Rabanser and Papernot 2025 | Only feature-aware, reordering scores can close the ranking gap | Every rejected signal was a monotone rescaling or a feature-space score that lost; the theory predicts the record | Answered, and now cited |
| Heng and Soh 2026, likelihood ratios | The optimal selector is a likelihood ratio, with gains under covariate shift | Never scored here; the Mahalanobis and typicality scores tested are not likelihood ratios of correct against wrong | Untested: a likelihood-ratio selector on the 24 tasks, one CPU job |
| Class margin dispersion, ECCV 2026 workshop | The whole margin distribution beats top-1 minus top-2 on AURC | The record's score is top-1 minus top-2 only | Tested in part (exp76): the published definition could not be retrieved; a margin-dispersion score defined here beat the better probability form on 0 of 24 tasks, and no whole-vector score on more than 3 |
| Guarino et al. 2026; Soft Dice Confidence 2026 | The aggregator from pixels to regions is a first-class choice; spatially aware aggregation wins | The 4-px window score is the margin of window-mean probabilities, chosen without a test | Tested (exp76): five aggregators on the seven segmentation tasks; the form matters, the aggregator does not, since the same form on window-mean probabilities beats its pixel-level aggregate on 6 of 7 |
| Kondylatos et al. 2025 | Pretrained representation uncertainty works zero-shot in Earth observation | exp17 tested band-set disagreement, depth probes, logit lens and attention entropy; not this reading | Untested: needs their pretrained uncertainty head, an encoder pass |
| Lehmann et al. 2026 | Confidence of frozen geospatial encoders miscalibrates under corruption and shift; ensembles do not help | exp74 scored ranking under 16 encoders on clean splits; ensembles lost in exp73 | Half answered: agrees on ensembles; of two shifts of year and place applied here, graded at random samples, the first (exp99, exp103) saw a fine-tuned model's confidence miss its ranking bar, consistent in direction with their finding, and the second, a blind test at fresh plots farther away (exp111), did not repeat the miss; neither run attributes the result to the shift, and no corruption was applied |
| Johnson et al. 2026, RSE | A spatially aware, calibrated uncertainty for this model class | Not scored | Untested, encoder-side |
| Consensus mapping 2026; CODA 2025 | Fusion of independent raters pays; consensus selects models | Dawid-Skene fusion and a second same-family model were rejected | Answered with a scope: the rejection is within one family; cross-family raters were never tried |
| Nguyen et al. 2025, concepts | Logit-level signals stay overconfident; concept rankings do better | No concept model exists for these encoders | Not applicable without one |
| nnActive 2025; Outshining the Random Baseline 2026 | Active learning beats random only against a naive random baseline, or only after fixing imbalance and redundancy | exp56 found suspicion-ordered selection worse than random at every budget | Half answered: exp56's random baseline was plain random; the imbalance fix was not applied |
| Xu et al. 2024, RSE; illusion of success 2024; PANGAEA 2025 | 10 m references carry geolocation and label error; curated test sets inflate accuracy; benchmarks are geographically biased | The five-reference design and the design-weighted estimators address the first; the suite's splits are the model authors', not a landscape sample | Answered in part; the spatial support of the 24 tasks is not characterised |
| Goel et al. 2025, CAPA | Raw disagreement between models is confounded by their accuracies | exp75 reports raw vote disagreement and KL | Untested and cheap: recompute exp75's readings as chance-adjusted agreement |
| Kao et al. 2026, UnivEARTH; DataMind 2026 | Most agent failure is execution failure in a sandbox, not reasoning | exp64's sandbox arm produced no gradeable answer in 51 of 120 runs at 7B | Answered: the record already separates unanswered from wrong |

Tests this suggests, in order of cost: AUGRC, a margin-dispersion score and five
window aggregators were run as exp76. Still not run: CAPA for exp75, a
likelihood-ratio selector on the 24 tasks, and representation uncertainty and
the Matérn estimator, which need encoder passes.

**Lazy aggregation** (Shi, Yu and Yang, *Vision Transformers Need More Than Registers*, arXiv 2602.22394, Apr 2026). The challenge: a ViT's global representation is assembled from background patches, so a token can be confidently labelled by its neighbourhood rather than by its content, which would be a mechanism for the confident errors this record cannot see. Their probe needs a CLS token and their fix changes pre-training; OlmoEarth has neither a CLS token nor register tokens in the released checkpoints, and its pre-training is Ai2's. **Tested on the surrogate available on frozen tokens** (a token against its own tile's mean), exp77: the signature is present on 5 of 7 segmentation tasks but is confounded with class frequency and needs labels to compute; as a ranker it loses to the margin on 7 of 7; and removing the scene direction costs accuracy on all seven. The mechanism is closed on the surrogate, not on their probe. <!-- claim:scene-typicality-loses-to-the-margin -->
