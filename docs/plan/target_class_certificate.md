# The target rule, the class design and the certificate on fully labelled real maps (exp116 preregistration)

**Status: frozen (24 October 2026), by me, at the owner's request ("fix all", after the final three-rater score);
the owner has not reviewed the predictions.** Written and frozen before any statistic of this page was computed.

## The question

The final score of 1.17.0 (24 October 2026, three independent raters, median 80) left three of the package's newest
parts with no real-map evidence:

- the target rule (1.16.0), checked only by enumeration on small synthetic maps;
- the class design, a sample stratified by map class (Olofsson et al. 2014), new in 1.17.0, with exact per-class
  intervals checked only by exact computation on small synthetic maps;
- the map certificate (1.17.0), checked only by tests.

exp116 grades them on fully labelled real maps, where every quantity a statement is about is known.

## Populations (fixed before any statistic of this page)

1. **The embedding suite's unit files, `exp/out/exp79_units`** (16 encoders x the OlmoEarth paper's tasks, 356
   files, every one with the map's class `dec`, the reference `y`, the error `err` and the confidence `margin` of
   every unit). The file list and each file's sha256 are recorded in the output. A unit is the file's own unit. These
   maps are linear probes on benchmark splits, not landscape maps: the report says so where it cites them.
2. **The package's Dynamic World expert tile** (`oe_inferencex/sample/dynamic_world_tile.npz`, 128 x 128 windows, nine
   classes, Dynamic World's probabilities beside the expert consensus labels, Zenodo 4766508, CC BY 4.0), for P5 only.
   Windows with no expert label are no-data.

## Arms

**A. The target rule against prefix and Bonferroni (P1, P2).** Maps of population 1 whose error rate lies in [0.02,
0.5]; alpha half the map's error rate, delta 0.1, the package's grid and standard level cut. Budgets n = 300 and n =
1,000, each only where n is at most half the map's units. Targets 0.25, 0.5 and 0.75. A cell is a map, a budget and,
for the target rule, a target. 400 draws per map and budget, simple random samples of n units, from
SeedSequence([116, crc32(map), n]); every rule reads the same draws. On each draw each rule's certified zone is
computed with the package's own functions (`zone_order`, `zone_levels`, `zone_counts`, `zone_pvalue`,
`apply_zone_rule`, `target_weights`, in certify_zone's order; the script checks on the first draws of the first map
that `certify_zone` gives the same zones). A draw violates when the zone certified is wrong more than alpha over all
its units.

**B. The class design against a simple random sample (P3, P4).** Maps of population 1 with at least two map classes
and at least 800 units. Budget 400; the floor min(50, floor(400 / (2 K))) with K the map classes present, so the
floors take at most half the budget; allocation `estimate.class_allocation`. 400 draws per map of the class design and
400 simple random samples of 400, from SeedSequence([116, crc32(map), 1]) and ([116, crc32(map), 2]). On each class
draw: each class's user's accuracy interval, the class shares and producer's accuracies (`estimate._class_design_exact`,
the code `estimate --per-class` runs) and the overall accuracy (`estimate._union_interval`, the code `estimate`
runs). On each simple random sample: each class's user's accuracy by `estimate.hypergeom_interval` on the labelled
units the map calls that class (width 1 when none is labelled).

**C. Certificates on a real tile (P5).** The Dynamic World tile written as a (9, 128, 128) probability raster
(float32 .npy) and its expert labels; alpha 0.10. Five samples `sample --design random --target 0.5 --seed-from-map
--alpha 0.1 --patch 1` at budgets 200, 300, 400, 500 and 600, and one `sample --design class --seed-from-map --alpha
0.1 --patch 1 --budget 600`; each labelled from the expert labels (wrong where the map's class differs from the
expert's), certified, packed with `certificate` and checked with `verify --scores`.

## Predictions (bars fixed now)

- **P1, validity.** On every cell of arm A, the target rule's violation rate is at most 0.145 (delta 0.1 plus three
  binomial standard errors at 400 draws). Prefix and Bonferroni are reported beside it as controls under the same
  bar.
- **P2, the target rule's benefit.** Among the cells of arm A where the target zone's own error rate is at most alpha
  (the target is right), the share of draws on which the target rule certifies at least the target is at least the
  share on which Bonferroni certifies at least the target's level, on at least 80% of those cells.
- **P3, the class design's exact intervals hold.** Over arm B, with 0.917 (0.95 less three binomial standard errors
  at 400 draws) as each coverage's bar:
  - (a) each class's user's accuracy covers the class's true user's accuracy on at least 0.917 of draws, for at least
    99% of the (map, class) pairs;
  - (b) all the class shares and producer's accuracies hold together on at least 0.917 of draws, on at least 99% of
    the maps;
  - (c) the overall accuracy interval covers on at least 0.917 of draws, on at least 99% of the maps.
- **P4, what the class design buys.** For the (map, class) pairs whose class is under 5% of the map and holds at
  least the floor's units, the class design's mean user's-accuracy interval width is below the simple random
  sample's, on at least 80% of those pairs.
- **P5, certificates on a real map.** All six certificates are written and verify with the map given (`verify
  --scores`). Each certified zone's true error rate against the expert labels is reported, beside alpha, as a
  description: six draws grade nothing.

## What is reported beside the predictions

For A: per cell the violation rates of the three rules, the share of draws certifying at least each target, and the
mean certified coverage. For B: the coverage distributions and the width ratio. For C: each certificate's zone,
statement and true error. Every number in the output is a count or a share of the draws above; there is no fit.

## Reading the result

P1 failing on a cell would be a defect of the target rule as shipped and is reported as such, with the cell. P2
failing means the target rule's label saving does not reach real maps at these budgets. P3 failing would be a
defect of the class design's intervals. P4 failing means Olofsson's allocation does not help these maps' rare
classes at 400 labels.
