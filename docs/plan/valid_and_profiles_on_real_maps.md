# estimate --valid and the per-profile verdicts on fully labelled real maps (exp120 preregistration)

**Status: frozen (29 October 2026), by me, at the owner's request ("记录": record it); the owner has not reviewed the
predictions.** Written and frozen before any statistic of this page was computed on the population of arm A and before
the draws of arm B were taken.

## The question

The Fable score of 1.20.0 (27 October 2026, three raters, median 83) left two of 1.20.0's parts with no real-map
evidence:

- `estimate --valid`, the interval valid at any sample size under every design but tiles, measured only on the demo
  tile, and only for its cost (exp119), never for its coverage on a map whose every window is labelled;
- the per-profile verdicts (`sample --design profile`, then `certify`: good, bad or undecided per profile), never run
  on a real map.

This experiment grades both where every quantity a statement is about is known.

## What was seen before this plan

On 29 October 2026 an unregistered first check (aicr job 1580392, package 1.20.2 with the mask fix of 1.20.3, script
not in this repository) ran both on the Dynamic World expert tile of population 2, at 500 draws per cell: `--valid`
covered on 95.2% to 100% of draws per cell, a wrong verdict on at most 0.6% and an interval miss on at most 7.8%. Arm B
repeats that tile with fresh seeds and the bars below; it is not independent of what was seen. Population 1 has never
been run for this question.

## Populations (fixed before any statistic of this page)

1. **The embedding suite's unit files, exactly exp116's 356** (`exp/out/exp79_units`, 16 encoders x the OlmoEarth
   paper's tasks; the file list and each sha256 are those exp116 recorded in `exp/out/exp116_ab.json`, which excludes
   `olmoearth_base_rtx`; the script refuses a file whose sha256 differs). Each file gives every unit's confidence
   margin `margin`, top-1 probability `p1`, map class `dec`, reference `y` and error `err`. A unit is the file's own
   unit, laid out as a (1, N) grid. These maps are linear probes on benchmark splits, not landscape maps: the
   report says so where it cites them.
2. **The package's Dynamic World expert tile** (`oe_inferencex/sample/dynamic_world_tile.npz`, 128 x 128 windows of one
   pixel, nine classes, Dynamic World's probabilities beside the expert consensus labels, Zenodo 4766508, CC BY 4.0).
   Windows with no expert label are no-data.

The package is olmoearth-inferencex 1.20.3 from PyPI (its version is recorded); `--valid`, the profile design and
the verdicts are unchanged since 1.20.0.

## Arms

**A. Population 1, through the package's functions, as the command line calls them.** On each draw:
`estimate.sample_for_estimation` on the unit's margin and `p1` (the condition design with the map class as the
condition and `class_floor` 50 for the class design; with the profiles of `estimate.build_profiles` as the condition
for the profile design), the labels `err` of the drawn units, then `estimate.estimate_error_rate(..., valid=True)` or
`certify.localized.profile_verdicts`. The truth of a map is the mean of `err` over its units; of a profile, over the
profile's units.

- A1, `--valid`: designs random, confidence and proportional at n = 300 and 1,000, the class design at n = 400 and
  1,000; a cell is a map, a design and n, run only where n is at most half the map's units.
- A2, verdicts: profiles `confidence:4` and `class` (at most 12 profiles, cells under 100 units folded into `other`,
  the package's defaults), 800 labels, delta 0.1 (the package's default), alpha at half, one and twice the map's
  error rate e, on the maps with e in [0.02, 0.5] and at least 1,600 units. A cell is a map, a profile field and an
  alpha.

**B. Population 2, through the command line end to end** (`sample --patch 1 --seed S`, the labels written from the
expert map, `estimate --valid` or `certify --alpha A`), as a user runs it: the cells of A1 (the class design at 400
and 1,000, since 8 classes x 50 labels need 360), and of A2 with the fields `confidence:4`, `class` and
`grid:2,confidence:2` and alpha 0.1, 0.2 and 0.3 (the first check's), 800 labels.

400 draws per cell. Cells are numbered: the `--valid` cells 0 to 7 in the order random 300, random 1,000, confidence
300, confidence 1,000, proportional 300, proportional 1,000, class 400, class 1,000; the verdict cells 100 + 10 x the
field's index (`confidence:4` 0, `class` 1, `grid:2,confidence:2` 2) + the alpha's index (in arm A e/2 0, e 1, 2e 2;
in arm B 0.1 0, 0.2 1, 0.3 2). In arm A a cell's 400 seeds are
`numpy.random.default_rng(SeedSequence([120, crc32(name), cell])).integers(0, 2**31 - 1, 400)`, with name the unit
file's path relative to the units directory (for example `anysat/breizhcrops.npz`), each passed as
`sample_for_estimation`'s seed; in arm B draw k of a cell runs `sample --seed 120000000 + 1000 * cell + k`. None is
one of the first check's seeds. A draw the package refuses (a budget the class floors
cannot meet, more than 12 profiles) is counted as refused and graded nothing; a refusal is not an error. A draw whose
valid interval is not computed counts as not covering. In arm B the script also counts the sampled windows whose map
class in the CSV is not its own argmax, and draws whose population is not the expert-labelled windows; both must be 0.

## Predictions (bars fixed now)

The bars allow three binomial standard errors at 400 draws.

- **P1, `--valid` holds.** The valid interval covers the map's true error rate on at least 0.917 of draws (0.95 less
  three standard errors) on at least 99% of arm A's A1 cells, and on every cell of arm B.
- **P2, the verdicts hold.** The share of draws on which some profile's verdict is wrong (good where the profile's
  rate is above alpha, bad where it is at most alpha) is at most 0.083 (delta / 2 = 0.05 plus three standard errors)
  on at least 99% of arm A's A2 cells, and on every cell of arm B.
- **P3, the profile intervals hold together.** The share of draws on which some profile's interval misses its rate is
  at most 0.145 (delta 0.1 plus three standard errors) on at least 99% of arm A's A2 cells, and on every cell of arm
  B.

## What is reported beside the predictions

For each A1 and B cell: the coverage of the valid interval and of the interval given without `--valid`, and their
mean widths. For each A2 and B cell: the two shares above, the mean number of profiles decided among those labelled
in part (a profile labelled in full is reported apart), and the refusals. Every number is a count or a share of the
draws above; there is no fit.

## Second route

Before any number of this run becomes a claim, a second implementation written from this page alone draws arm A's
cells again with the same seeds and the package's functions, grades them with its own code, and must give the same
counts; and an independent audit reads the claims against the outputs.

## Reading the result

P1 failing on more than 1% of cells, or on a cell of arm B, would be a defect of `--valid` as shipped and is
reported as such, with the cells. P2 or P3 failing would be a defect of the per-profile verdicts. A pass is evidence
on these maps and designs only: linear-probe maps and one Dynamic World tile, at these budgets.
