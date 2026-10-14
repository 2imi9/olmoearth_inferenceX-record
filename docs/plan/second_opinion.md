# A second opinion from an independent product: does disagreement with another map find a product's confident errors? (exp106 preregistration)

**Status: frozen on 12 October 2026, before anything using a reference label together with the disagreement or the
confidence was computed.** Written 12 October 2026. The owner chose the predictions and their bars on the night of 11 to 12
October 2026 (P1: the combined order's AUROC at least 0.02 above the product's own confidence's on at least 2 of 3
product cells; P2: the flag at least twice as frequent among confident errors as among confident correct rows, on all
3). "Weighted AUROC" in the owner's wording is the package's `weighted_auroc` with every row weighted 1. Changed before
freezing, with the bars unchanged: the third cell was proposed as LCMAP against NLCD; LCMAP Collection 1.3 was trained
on NLCD's 2001 map, so the two are related by construction (consistent with that, lcpconf predicts disagreement with
NLCD at AUROC 0.76 to 0.83, a stratified sample read unweighted, against 0.638 with Esri at cell 3), and LCMAP's second
opinion is Esri at LCMAP's own reference plots, with the NLCD pair a report-only control. **Cell 3 is not blind for
P1**: exp92 graded both scores on its rows (below), and AUROC(c) − AUROC(u) = r(1 − q)(1 − A10) − (1 − r) q A01 (r and
q the flag's shares among errors and correct rows, A10 the own confidence's AUROC for disputed errors against
undisputed correct rows, A01 for undisputed errors against disputed correct rows), so cell 3's part of P1 is largely
implied; P1 among the two blind cells is reported beside the verdict. Two reviews before freezing; the bars are the
owner's, the readings below are mine, to be shown to the owner. The page is frozen before anything that uses a
reference label together with the disagreement or the confidence is computed.

**Result, 12 October 2026 (added at recording; the frozen text is unchanged).** Run once at 5169e46 (job 1292574),
recorded after an independent audit ([comparisons](../results/comparisons.md#a-second-opinion-from-another-product-disagreement-with-esri-at-three-published-products-exp106)).
P1 holds as frozen: the combined order gains +0.077 (Europe) and +0.079 (CONUS), −0.002 in East Africa; one of the two
blind cells meets the bar, and 0.035 of CONUS's gain is tie-breaking. P2 does not hold: f_err / f_ok is 4.11 (Europe),
4.96 (CONUS) and 1.45 (East Africa, interval 1.31 to 1.59). The reading "P1 holds, P2 does not" applies; the related
control meets the ratio too (2.67 to 10.51), so it does not measure independence.

- **Allowed before freezing:** the label-free inventory (rows, disagreement shares, the confidence's coupling with the
  disagreement; done on 12 October 2026, below) and the smoke on synthetic rows.
- **Not allowed before freezing:** any error, AUROC, capture or cross-tabulation that uses the reference labels with
  the disagreement or the confidence, beyond what is already recorded (below).
- **Already known (seen):** exp94 recorded each product's own-confidence AUROC on cells 1 and 2 (CGLS 0.720, ODSE 0.787)
  and, on its own populations, how many of each pair's disagreements each map gets right (exp94 Q1), from which the
  binary disagreement's precision and recall for a product's errors can be derived; exp92 recorded on cell 3 that
  disagreement with Esri alone ranks LCMAP's errors at AUROC 0.725, against lcpconf's 0.753. Row identities: exp94's
  ODSE AUROC is on exactly cell 2's 4,807 rows; exp92's disagreement AUROC on exactly cell 3's 4,794 rows and its
  lcpconf AUROC on 4,796; exp94's CGLS AUROC on 12,598 rows, 24 more than cell 1. exp93 (cell 3's rows), exp94's Q2
  and Q4, and exp95 to exp97 (all three cells) graded the products' own confidence, its confident zones and the error
  rates against the labels; none used the disagreement. The combined order and the confident errors have never been
  graded. On 11 October 2026 the literature scouting computed, unaudited, a cross-tabulation of CGLS's raw shrub (20)
  and herbaceous (30) classes against TimeSync's Wooded and Open Grassland at the Kenya and Tanzania 2017 plots (1,472
  rows of cell 1), and FT-AWF numbers at exp99's 271 plots (also cell 1 rows); both stay inside exp94's merged
  Grass/Shrub class, so neither shows a cell 1 error, the disagreement or the confidence. FT-AWF is not graded here.

## The question

The record's largest open limit: a model's confident errors are checked last, and no label-free reading tried has found
them (exp99, exp101, exp104). Every reading tried came from the same model or its inputs. Does a second opinion from an
independent map product, one made by another programme from other data, flag a product's confident errors more often
than its confident correct rows, and does putting the disputed rows first improve the review order?

## The cells (every row weighted equally; the results describe the rows)

| Cell | Product (own confidence) | Second opinion | Reference and rows |
|---|---|---|---|
| 1 East Africa 2017 | CGLS-LC100 2017 (`cgls`, `cgls_proba`, 0 to 100) | Esri 10 m LULC 2017, 3 x 3 majority | TimeSync 2017 plots, exp94's 7-class legend; year 2017, ref, cgls and esri present, cgls_proba at most 100: 12,574 rows (seven countries of about 2,000 each) |
| 2 Europe 2017 | ODSE-LULC 2017 (`odse`, `odse_prob`) | Esri 10 m LULC 2017, one pixel | S2GLC validation points, exp94's 8-class legend; the random 5,000 with a probability, ref, odse and esri present: 4,807 rows (55 tiles; not a probability sample) |
| 3 CONUS 2018 | LCMAP (`lcmap`, `lcpconf` 1 to 100) | Esri 10 m LULC 2018, 3 x 3 majority, in LCMAP's legend | LCMAP's reference plots (a simple random sample), the random 5,000 with a confidence, ref, lcmap and esri present: 4,794 rows |
| Control | LCMAP (`lcpconf`) | NLCD (related by construction) | NLCD's accuracy-assessment points, five set-years, unweighted as exp94 read them (LCMAP, NLCD and the reference in 8 classes by exp94's `NLCD_TO_LCMAP`, lcpconf 1 to 100): 7,337, 7,193, 7,201, 4,217 and 4,240 rows; report-only |

Columns and crosswalks are exp94's and exp92's (`exp/out/exp94_eastafrica_points.csv`, `exp94_europe_points.csv`,
`exp94_nlcd_points.csv`, `exp92_plots.csv`); nothing is fetched. Label-free inventory: the products disagree with
Esri on 41.5% (cell 1), 18.2% (cell 2) and 21.2% (cell 3) of the rows; low confidence predicts disagreement at AUROC
0.687, 0.641 and (control) 0.76 to 0.83. The disputed share falls inside the upper confidence range in cell 1 (28.8% of
the rows at or above the median confidence, 18.3% at or above the upper quartile), less in cell 2 (12.1%, 12.1%) and
cell 3 (15.1%, 15.1%). lcpconf is 99 on 2,645 of cell 3's 4,794 rows, so 31.5% of its row pairs tie in u (2.4% and 1.8%
in cells 1 and 2); where u ties, c decides by d alone.

## The scores

- **Own:** u = 101 − confidence (higher = more suspect), as exp92 and exp94 rank.
- **Disagreement:** d = 1 where the product's class and the second product's differ in the common legend.
- **Combined:** c = d + u / 102: every disputed row before every undisputed one, and within each group the least
  confident first.
- An **error** is a row where the product's class differs from the reference's. AUROC is of a score for the errors,
  ties counted half.

## Predictions (confirmed by the owner)

- **P1 (the order):** AUROC(c) − AUROC(u) ≥ 0.02 on at least 2 of the 3 cells.
- **P2 (the confident errors):** in each cell, take the median confidence m of its correct rows; a confident error is
  an error with confidence ≥ m, a confident correct row a correct row with confidence ≥ m. f_err is the share of
  confident errors with d = 1, f_ok the share of confident correct rows with d = 1. P2 holds when f_err ≥ 2 f_ok on all
  3 cells (and f_err > 0 there, so a flag that never fires among confident rows does not pass). Floor: at least 20
  confident errors in a cell; a cell below it is not determined and P2 cannot hold. Among confident correct rows d = 1
  exactly where the second product is wrong, and among confident errors d = 0 exactly where it makes the same mistake:
  f_ok is the second product's error rate on the confident correct rows and f_err one minus its share of repeated
  errors, so P2 mainly measures the second product against the product's confident errors, and it asks most of cell 1
  (Esri wrong on 39.8% of exp94's rows there).
- **Reported, not graded:** 90% bootstrap intervals (2,000 resamples, seed 106; cell 1 within each country, cell 2 by
  tile, cell 3 simple) for AUROC(c) − AUROC(u) and for f_err / f_ok; AUROC of d alone; capture at 10% of u, d and c;
  the precision and recall of d for the errors, and among the confident rows the precision of d (the share of flagged
  confident rows that are errors) and its recall of the confident errors (f_err); P1 among the two blind cells; the
  identity's terms r, q, A10 and A01 and the part of the gain from error/correct pairs that u ties; cell 1 by country;
  the control's five set-years.

## What would change the conclusion

- **P1 and P2 hold:** disagreement with Esri, a product made by another programme from other data, flags these
  products' confident errors at least twice as often as their confident correct rows and improves the review order
  (cell 3's part of P1 largely implied by exp92 and by lcpconf's ties); the flag's precision among confident rows is
  reported beside it. A reference-product option for `assess` is worth building and testing on a deployed OlmoEarth
  map, with a second product whose legend separates the classes that map confuses.
- **P2 holds, P1 does not:** the flag finds confident errors but does not improve the whole order; it is worth reporting
  as "confident but disputed", not as a ranking.
- **P1 holds, P2 does not:** the flag lifts errors the confidence ranked below correct rows, but in at least one cell it
  does not single out those at or above the correct rows' median confidence, or the second product is wrong too often
  on that cell's confident correct rows; read with each cell's f_err and f_ok. A cell under the floor makes P2 not
  determined, which is not read as a failure of the flag.
- **Neither holds:** disagreement with Esri does not help these three products here; the confident-error limit stands,
  now tested this way.

## What it cannot show

- The second opinion is the same product (Esri) in all three cells, and it is itself wrong on 39.8% (cell 1) and 14.9%
  (cell 2) of exp94's rows; the three cells are three products, not three independent second opinions.
- Much disagreement is legend and support (Esri's rangeland class, a 10 m map against 100 m and 30 m products).
- Products, not a deployed OlmoEarth model: FT-AWF's 2017 map at the TimeSync plots is not graded here (its join needs
  the cluster's pinned geometry, and its plots were looked at in the literature scouting).
- Labels taken as right.
- Cell 3's reference is LCMAP's own programme's, in LCMAP's legend (whether any plot trained LCMAP was not checked);
  Esri's crosswalk into that legend adds disagreement on rows LCMAP gets right.
- Cell 1's legend merges Wooded and Open Grassland (and Esri's rangeland), so it cannot test the grassland/shrubland
  confusions behind most of exp99's confident errors.
- Cell 3 is not blind for P1 (exp92), and its confidence is 99 on 55% of its rows, so part of a gain there is
  tie-breaking.

## Outputs

`exp/out/exp106_summary.json`. Script `exp/exp106_second_opinion.py` (numpy), run as a cluster CPU job pinned to a
commit.
