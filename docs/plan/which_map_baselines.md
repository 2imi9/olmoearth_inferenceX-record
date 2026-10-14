# Which of two maps is more accurate: the tool's design against McNemar's test on a whole-map sample (exp107 preregistration)

**Status: frozen on 12 October 2026, before any arm's probabilities were computed on the nine pairs.** Written 12
October 2026. The owner confirmed the predictions and their bars on 12 October 2026 (P1:
the tool's design names the better-agreeing map at least as often as practice on at least 7 of 9 pairs at 100 labels,
both naming the wrong map on at most 5% of draws on every pair; P2: if an exact test on the tool's own sample names it
at least 5 points more often than the tool's interval on at least 5 of 9 pairs, and names the wrong map on at most 5% of
draws on every pair, it replaces the interval's verdict in the package). Changed after two reviews and before freezing,
with the bars' numbers unchanged: practice is graded in two forms, McNemar's exact test and Foody's uncorrected z, and
P1 asks the tool's design to meet both; the readings and limits below were sharpened. The page is frozen before any
arm's probabilities are computed.

**Result, 12 October 2026 (added at recording; the frozen text is unchanged).** Run once at 1daa6bb (job 1298649),
recorded after an independent audit ([comparisons](../results/comparisons.md#which-map-is-more-accurate-the-tools-design-against-mcnemars-test-exp107)).
P1 does not hold (8 of 9 against the exact test, 6 of 9 against Foody's z form; the three losses on pairs under 1.2
points). P2 holds (5 of 9); the owner chose to adopt the sign test only up to a fifth of the differing windows, since
past a quarter to two fifths it names the better map less often than the interval. The package change is made for
1.10.0 (`estimate.which_map_rule`).

- **Not blind for the populations:** each pair's counts are on the record (exp92 and exp94, Q1): N, the reference rows
  where both maps have a class; D, those where the maps differ; K_A and K_B, those among D where only map A, or only
  map B, agrees with the reference. Every arm's operating characteristic is a function of (N, D, K_A, K_B), so the
  outcome is arithmetic on known counts; what has never been computed is arms A and C below, and the bars were set
  with the pairs' differences known (four of 4 to 6 points, one of 2.7, one of 1.1, three under 1). The arms' order is
  predictable to first order from the one-right rows each design draws, n·(K_A + K_B)/N for practice against
  n·(K_A + K_B)/D for the tool's sample (8 to 34 against 84 to 100 at n = 100); none of the arms' probabilities was
  computed before the bars were set. exp92 and exp94 recorded arm
  B's naming rates by Monte Carlo at 50, 100 and 200 labels (and a whole-sample union-bound arm), and exp90 the same
  design on 2,514 probe-map pairs.
- **Allowed before freezing:** reading the recorded counts; the smoke on constructed counts.
- **Not allowed before freezing:** computing any arm's naming probabilities on the nine pairs (including with
  `plan.which_map_probability`).

## The question

Practice compares two maps' accuracy from one sample of reference points with McNemar's test (Foody 2004): a simple
random sample of the whole area, the test on the points where exactly one map agrees with the reference. The tool
instead draws all its labels where the two maps differ (`sample --other`) and names a map when its interval on the
accuracy difference excludes zero (`estimate`); that interval is a union bound of each map's exact interval at 97.5%,
which is conservative. Does the tool's design name the better-agreeing map more often than practice at the same
budget, and would an exact test on the tool's own sample name it more often still?

## The nine pairs (counts as recorded)

| # | Pair (A against B), reference | N | D | K_A | K_B |
|---|---|---|---|---|---|
| 1 | LCMAP against Esri 2018, LCMAP's reference plots (exp92) | 24,968 | 5,387 | 2,728 | 2,062 |
| 2 | NLCD against LCMAP, NLCD points set A 2001 (exp94) | 8,000 | 1,467 | 452 | 782 |
| 3 | NLCD against LCMAP, set A 2006 | 8,000 | 1,812 | 598 | 976 |
| 4 | NLCD against LCMAP, set A 2011 | 8,000 | 1,733 | 727 | 817 |
| 5 | NLCD against LCMAP, set B 2011 | 4,629 | 982 | 454 | 417 |
| 6 | NLCD against LCMAP, set B 2016 | 4,629 | 1,115 | 636 | 364 |
| 7 | GFC2020 against WorldCover, GFC2020's validation set | 21,612 | 1,770 | 962 | 808 |
| 8 | CGLS-LC100 against Esri, East Africa TimeSync 2017 | 13,975 | 5,479 | 2,388 | 2,297 |
| 9 | ODSE-LULC against Esri, S2GLC points 2017 | 49,897 | 9,274 | 5,553 | 3,181 |

The better-agreeing map is the one with the larger K (A when K_A > K_B); "right" and "wrong" below are relative to it.
Every row counts once (the NLCD and GFC samples are stratified and read unweighted, as exp94 read them).

## The arms (exact probabilities, no Monte Carlo)

- **A, practice:** n labels by simple random sampling from the N rows; McNemar's exact test (the two-sided exact
  binomial sign test, `stats.sign_test`) on b, the drawn rows where only A agrees, and c, where only B agrees; names
  the map with more when p < 0.05. The count m = b + c is hypergeometric over the N rows and b given m hypergeometric
  over the K_A + K_B one-right rows.
- **Az, practice in Foody's form:** the same sample; names the map with more when (b − c)² / (b + c) > 1.96² (the
  uncorrected z, more powerful than the exact form at small b + c). **Acc** (McNemar's chi-square with continuity
  correction, R's default) is reported only.
- **B, the tool:** n labels drawn among the D differing rows; the package's interval (`estimate`,
  `compare_from_disagreement`), whose naming probabilities `plan.which_map_probability` gives exactly; at n ≥ D a
  census, which names the map with more.
- **C, the tool's sample with an exact test:** B's sample; the sign test on b and c as in A; at n ≥ D the census answer.

Each arm gives P(names the better-agreeing map), P(names the other) and P(names none) at each budget n.

## Predictions (confirmed by the owner)

- **P1 (the design against practice):** at n = 100, B names the better-agreeing map at least as often as A on at least
  7 of the 9 pairs and at least as often as Az on at least 7 of the 9, and A, Az and B each name the other map with
  probability at most 0.05 on every pair.
- **P2 (the decision):** at n = 100, if C names the better-agreeing map at least 0.05 more often than B on at least 5 of
  the 9 pairs, and C names the other map with probability at most 0.05 on every pair, C's verdict replaces the
  interval's verdict in the package (the interval is still printed); otherwise the package is unchanged. The new
  verdict would be the exact sign test on the windows where only one map is right, at alpha = 1 − conf: it says only
  which map is more accurate (it names the less accurate one at most (1 − conf)/2 of the time, for every population;
  the difference (K_A − K_B)/N is exactly the two maps' accuracy difference, rows where neither is right only shrinking
  m), printed with its p-value and counts; the union-bound interval stays as the size, with a note that it can include
  0 when the test names a map. With u windows marked '?', a is named only if the test on (a, b + u) names it, and b
  only if the test on (a + u, b) does. `plan.which_map_probability`, `plan`'s which-map advice, `decide`, the CLI and
  MCP texts would follow.
- **What the wrong-map conditions check:** the exact test and the interval each name the worse map at most 0.025 of the
  time by construction (without replacement, given m the count b is hypergeometric over the one-right rows; at K_A =
  K_B its ratio to Binomial(m, 1/2) is symmetric and decreasing in |b − m/2|, so its tails are no heavier, and it is
  stochastically increasing in K_A), so these conditions check the code, not a prediction; the record reports each
  arm's largest wrong rate over all pairs and budgets against 0.025.
- **Reported, not graded:** n = 50, 200, 400 and 800 (capped at D for B and C); P(names none); Acc; the labels each arm
  needs to name the better-agreeing map with probability 0.8 and 0.9 on each pair, on a ladder to 6,400 (capped at N
  for practice, at D for the tool's sample); each arm's largest wrong rate; arm B's exact rates beside exp92's and
  exp94's recorded Monte Carlo rates of the same design.

## What would change the conclusion

- **P1 holds:** on these nine pairs (not independent), at 100 new labels, labels drawn where the maps differ, read with
  the package's interval, name the better-agreeing map at least as often as McNemar's test in either form on a simple
  random sample of the same size. This is computed from recorded counts and follows to first order from the one-right
  rows each design draws; it is for the one question of which map is better, at equal label count.
- **P1 does not hold:** on at least 3 of the 9 pairs practice's test names the better-agreeing map more often at 100
  labels; the record names those pairs, gives both arms' rates, and says on which neither arm names it with probability
  above 0.1, where the comparison is between rates near no difference that the union bound keeps lower.
- **P2 holds:** the tool adopts the exact test on its sample (a change to the package, with tests), keeping its
  interval as the size of the difference.
- **P2 does not hold:** the union-bound interval's verdict stays.

## What it cannot show

- "Better-agreeing" is agreement with each reference through its crosswalk, not accuracy on the ground; several
  references belong to one of the two programmes (LCMAP's plots, NLCD's points, GFC2020's validation set).
- The nine pairs are not nine independent tests: LCMAP is in six, Esri in three, and the NLCD set-years share points.
- Pairs related by construction (LCMAP trained on NLCD 2001; GFC2020 built partly from WorldCover) share errors, which
  shrinks D; this changes the size of the design's gain, not any arm's validity.
- Sampling is without replacement from a finite set; the sign test assumes independent draws, which here makes it
  conservative (the argument above; checked exactly on a grid by the tests).
- Equal label count, not equal effort: the rows where two maps differ are boundaries and mixed cells, harder to label,
  and that cost is not measured. Practice's sample also gives each map's own accuracy, which the tool's design needs a
  second random sample for, and a reference set that already exists costs no new labels.
- On the stratified reference sets (NLCD sets A and B, GFC2020) the share of rows where the maps differ is the sample's,
  not the map's; the design's gain grows with N/D, so its size here is the reference sets'.
- On the pairs whose difference is under about a point, neither design may name the better map at 100 labels, and the
  comparison there is between rates near no difference.

## Outputs

`exp/out/exp107_summary.json`. Script `exp/exp107_which_map_baselines.py` (numpy), run as a cluster CPU job pinned to
a commit.
