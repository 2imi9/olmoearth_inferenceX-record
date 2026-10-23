# Per class, exactly: does a few hundred random labels show which class a map draws wrong, with guarantees that hold? (exp115 preregistration)

**Status: frozen (23 October 2026), by me, on the owner's overnight approval; the owner has not reviewed the
predictions.** Written and frozen before any statistic of this page was computed.

## The question

exp113's held-out maps ranked their errors well, yet three of them drew almost no crops. Their overall error rate
said little: Mayotte's map erred on 0.5% of its windows while missing every crop parcel. The literature's answer is to
report each class separately from the random labels a user already draws (Olofsson et al. 2014; Stehman and Foody
2019). The package does this with `estimate --per-class`, but not by default, and not with exact intervals for every
quantity: producer's accuracy and the class shares use normal-theory intervals. It also returns no producer's accuracy
when no labelled window falls in a class the map rarely draws.

exp115 grades a per-class report in which every interval is exact. All of it rests on one fact: in a simple random
sample of n of the N units, the count of units with any fixed property is hypergeometric, so the package's
`estimate.hypergeom_interval` covers that property's share with at least its nominal probability, on every population.
The report, for each class c:

- **Its true share p_c = K_c / N, and the share the map draws m_c = M_c / N.** m_c is known without labels. The
  interval for p_c is `hypergeom_interval(k_c, n, N, conf = 1 − 0.05/C)` from the k_c labelled units whose reference is
  c. With C classes this is Bonferroni, so all C intervals hold together with probability at least 0.95.
  - The **flag**: the map draws c too little when m_c is below the interval, and too much when it is above.
- **Its producer's accuracy PA_c = A_c / K_c**, where A_c counts units that are c and drawn c. A_c and K_c each get an
  exact interval at level 1 − 0.05/(2C), from the a_c labelled units that are c and drawn c, and from k_c. Then:

      PA_c in [ A_lo / K_hi ,  min(1, A_hi / K_lo, M_c / K_lo) ]

  The last term uses A_c ≤ M_c, which holds on every map. Under Bonferroni over the two counts and the C classes, all
  C intervals hold together with probability at least 0.95. When K_lo is 0 the upper end is 1.
  - The **collapse case is covered.** A class the map rarely draws gets a small upper bound through M_c / K_lo even
    when no labelled unit is drawn c.

Every probability in this experiment is computed exactly, from the counts of each population, with no Monte Carlo.
- A share interval's coverage and the flag's probability come from the hypergeometric distribution of k_c.
- A PA interval's coverage and the chance its upper end falls below a bar come from the joint distribution of
  (a_c, k_c − a_c, n − k_c), a three-category multivariate hypergeometric, summed over every possible sample.
- The chance that any share interval misses comes by convolution over the classes, as in exp114.

## Populations (fixed before any statistic of this page)

Every population below has a reference class for every unit, so each is graded in full. A unit is the population's
own unit.

1. **PLANTEUR, 18 maps.** exp113's maps: five held-out territories and the default split, three embeddings each.
   Units are 4 × 4 pixel windows by exp113's rules (map class: the package's pooled class; reference: the majority of
   non-void pixels). Eight classes. Read from exp113's private window arrays on the cluster's scratch.
2. **LCMAP's reference plots, 3 maps.**
   - LCMAP Collection 1.3 and Esri (crosswalked) at the 25,000 plots of `exp/out/exp92_plots.csv`;
   - the OlmoEarth v1.3 map at the 11,000 plots of `exp/out/exp112_plots.csv` outside its fit set.

   LCMAP's legend, eight classes.
3. **The embedding suite, 19 maps.** `exp/out/exp78_units`:
   - the 18 classification tasks, whose files carry the reference class `y` (`err` = `dec` ≠ `y` on every unit);
   - Sen1Floods11, whose truth is the map's class where `err` is 0 and the other class where it is 1.

   The five other segmentation files store no reference class and are left out.

40 maps in all (39: see the amendment before the run).

## Arms

For each map, class c with K_c > 0, and n in {100, 400, 1000}, with n capped at N (a census is reported, not graded):

1. **Share coverage:** the exact probability that the share interval covers p_c.
2. **Joint share coverage:** the exact probability that all of the map's share intervals cover together.
3. **PA coverage:** the exact probability that the PA interval covers PA_c.
4. **The flag:** the exact probability that 400 labels flag c as drawn too little, or too much.
5. **The PA bound:** the exact probability that 400 labels put PA_c's upper end at or below 0.25.

Reported per class: K_c, M_c, A_c, p_c, m_c, PA_c, and the median interval widths at each n.

## Predictions (written by me; see the decision at freezing)

**P1, the guarantees (a check).** On every class of every map at every graded n:
- the share interval's exact coverage is at least 1 − 0.05/C;
- the PA interval's exact coverage is at least 1 − 0.05/C.

A failure is a defect: stop, find it, record the defect.

**P2, the joint guarantee (a check).** On every map at every graded n, the exact probability that all share intervals
cover is at least 0.95.

**P3, the flag finds a class drawn at half its share or less.** For every class with p_c ≥ 0.05 and m_c ≤ p_c / 2, 400
labels flag it as drawn too little with exact probability at least 0.8. If no class qualifies, P3 is not determined. P3 to P5 use the maps with N > 400, where 400 labels are a
sample, not a census.

**P4, the flag is quiet on classes drawn about right.** For every class with |m_c − p_c| ≤ 0.02, 400 labels flag it,
either way, with exact probability at most 0.10.

**P5, the PA bound shows a collapsed class.** For every class with p_c ≥ 0.05 and PA_c ≤ 0.10, 400 labels put the upper
end of its PA interval at or below 0.25 with exact probability at least 0.8. If no class qualifies, P5 is not
determined.

## What would change the conclusion

- **P1 and P2 hold** (they should, by construction). The report's guarantees are exact on 40 real maps.
- **P3 to P5 hold.** 400 random labels, the budget the README already recommends, show a class drawn far too little,
  stay quiet on classes drawn about right, and expose a collapsed class's producer's accuracy. Then `estimate` can
  print this report by default.
- **P3 or P5 fails.** 400 labels are not enough for the cases they name, and the report must say how many are.
- **P4 fails.** The flag fires on near-right classes too often to be printed without its interval.

## What it cannot show

- Simple random sampling only. The confidence-stratified design, which might find rare-class omissions with fewer
  labels, is not graded here.
- The references are taken as right.
- exp113's PLANTEUR outputs were read before this page, including which classes collapse. That is what P3 and P5
  test, so their PLANTEUR cells were expected to qualify.

## What would invalidate the run

- Any change, after the first statistic of this page is computed, to the populations, intervals, levels, bars or
  budgets.
- Grading before the page is frozen.

## Independent check before anything is recorded

An independent second route recomputes the graded numbers. It is written without the run's script or outputs.

## Outputs

`exp/out/exp115_summary.json`: per map and class, counts and exact probabilities. It holds no unit-level list and no
coordinate.

## Cost

No download. One CPU job on the cluster, where the PLANTEUR windows are; minutes to an hour.

## Decisions taken at freezing

- 23 October 2026, about 03:00. The owner asked for maths-trial experiments to run overnight, in rounds, with results
  by the morning ("I approve what you can do now"), and went to sleep. Earlier that night the owner agreed to the
  direction: per-class from random labels, exact intervals, the collapsed case shown. The populations, intervals,
  bars and predictions above are mine, frozen without the owner's review, and the owner reviews them in the morning.
- Read before freezing: the unit files' formats (which files carry a reference class) and exp113's recorded
  outputs. No count of this page was computed.

## Amendment before the run (23 October 2026, about 03:45; no statistic of this page had been computed)

The pre-run review and the second route's author each found that the page's counts do not match its own rule for
choosing files. The rule stands; the counts are corrected. They also found how exp92's map columns code a plot with no
class.

- **The suite's files.** `exp/out/exp78_units` holds 17 classification files with a reference class, not 18, and six
  other segmentation files with none, not five: `m_cashew_plant`, `m_sa_crop_type`, `mados` and the three `pastis`
  files. So the suite contributes 18 maps, the 17 and Sen1Floods11, and the experiment has **39 maps**, not 40.
- **Plots with no map class.** In `exp92_plots.csv`, code 0 in the `lcmap` and `esri` columns marks a plot where that
  product has no class. That is 29 plots for LCMAP and 3 for Esri, never both on one plot; the reference has a class on
  all of them. The page keeps all 25,000 plots. Such a plot stays in the population and counts toward no class: it is
  in N, in no class's M, and never correct. Its count is reported. exp92 had dropped those plots, which would change N.
- **Computation.**
  - Probabilities are normalised after their total is checked to within 1e-6, since the log-gamma terms carry about
    1e-9 error at N near a million.
  - Interval ends are compared with the counts as whole numbers.
  - The joint coverage is computed by a convolution that does not underflow. The first version gave 0.03 where the
    exact value is 0.97, on synthetic counts at N 820,531.
  - P1 allows a shortfall of 1e-9 as floating-point noise, and the margin is reported.
  - P4 is not determined if no class qualifies.

  None of these changes a quantity the page defines.

## Result (recorded 23 October 2026; the text above, with its amendment before the run, is the frozen page)

Run once: job 1472015 at 7e0a71a, 41 s by the job log; the summary records no commit or run time.
- The second route's job (1472003) ran first, at the same commit.
- A first submission of the run, job 1472002, failed before reading any data. It and the second route's job cloned
  into the same new folder at the same moment.

The page has 39 maps. LCMAP Collection 1.3 has 29 plots with no class and Esri 3. Recorded after the independent audit
and a review below.

- **P1 holds.** All 706 graded cells are at or above 1 − 0.05/C. That is 1,412 intervals: a share and a producer's
  accuracy in each cell.
  - Smallest margin for a share interval: 0.00013 (the suite's ForestNet, class 0, n = 100).
  - Smallest margin for a producer's-accuracy interval: 0.0024.
  - The 1e-9 tolerance was not needed.
- **P2 holds.** The lowest joint coverage of all share intervals is 0.963, over 98 cells (So2Sat, n = 400).
- **P3 does not hold, by one class.** 17 classes qualify: a true share of at least 5%, drawn at half of it or less.
  - 16 are flagged with probability at least 0.8, 13 of them with probability 1.000.
  - The miss is AlphaEarth's Banana in Martinique, drawn at 3.0% against a true 6.1%, flagged with probability 0.644.
  - Its share is the reason: at 400 labels about 24 of its windows are labelled, and the gap is 3.1 points. Classes
    drawn at about the same ratio (0.478, 0.485) but with larger shares are flagged at 0.9997 or more.
- **P4 does not hold.** 69 of the 164 classes within 0.02 of their true share are flagged with probability above 0.10.
  The largest is 0.9996.
- **P5 does not hold, by one class.** 10 classes qualify: a true share of at least 5% and producer's accuracy at most
  0.10. 9 get an upper bound at or below 0.25 with probability at least 0.8. The miss is TESSERA v2's Meadow in
  Martinique (producer's accuracy 0.090), at probability 0.684.
- **Reported.**
  - **The collapsed classes of exp113's v1.3 maps.**
    - Guyane's Meadow (drawn on 0.1% against 14.3%) and Réunion's Sugarcane (0.1% against 16.0%) are flagged with
      probability 1.000. Their producer's-accuracy upper ends fall to 0.25 or below with probability 1.000. For
      Réunion's Meadow it is 0.963.
    - Smaller collapsed crops are flagged less often. In Guyane, Fruits, vegetables and flowers at 0.645 and Tropical
      tuber at 0.462. In Réunion, Orchard at 0.790.
    - Mayotte's crops are 0.5% of its windows together (742 of 142,535); its largest, Banana, is flagged at 0.716.
  - **Over-drawn classes.** Guyane's and Réunion's v1.3 Background is flagged as drawn too much with probability 1.000. Réunion's
    v1.3 map draws Pineapple on 3.7% of the land against 0.3% and is flagged with probability 0.999. LCMAP Collection
    1.3 draws Cropland on 24.0% against the reference's 17.4% and is flagged with probability 0.651.
  - **Over every class with 400 labels, beyond the predictions' conditions.** Rare classes need more labels than 400.
    - Of 101 classes drawn at half their share or less, 31 are flagged with probability at least 0.8. Of the 84 among
      them with a share under 5%, 15 are.
    - Of 93 classes with producer's accuracy at most 0.10, 19 get a bound of 0.25 or less with probability at least
      0.8.
  - **Interval widths.** The median, over the 244 classes graded at 400 labels, of each class's median width:
    - 0.056 for a share;
    - 0.713 for a producer's accuracy (PLANTEUR 0.973, LCMAP 0.755, suite 0.560).

    Exact, simultaneous producer's-accuracy intervals bound a collapsed class well and a working one loosely.
  - **Census cells.** The page says a census is reported. The run marks it only on the joint coverage; the second route
    reports it per class.

## Notes at recording (23 October 2026, after the run; the frozen text, with its amendment, is unchanged)

- **The times on the page are wrong.** The decision at freezing says "about 03:00", and the amendment "about 03:45".
  The commits say otherwise:
  - the page was frozen at dfb5053, 02:36 EDT;
  - the amendment and the script's fixes are 7e0a71a, 03:03 EDT;
  - both jobs ran after it: the second route from 03:03:50 to 03:04:38, the run from 03:05:01 to 03:05:42.

  The amendment came before any statistic, as it says; only its stated time is wrong.
- **The amendment before the run.** The pre-run review and the second route's author found that the page's own file
  rule gives 39 maps, not 40, and that code-0 plots have no map class. The review also found defects in the
  computation, all fixed before the run (7e0a71a):
  - the probability totals' tolerance and normalisation;
  - whole-count comparisons, including the qualifying conditions;
  - a joint coverage that underflowed at large N (0.03 where the exact value is 0.97);
  - the P1 tolerance;
  - P4's undetermined case.
- **What P3 and P5 failing means, as the page set it.** "The report must say how many [labels] are" needed for the
  cases they name. Read with the over-all-classes numbers above: 400 labels expose a collapsed class of 5% or more,
  about half of those from 1% to 5%, and rarely one under 1%.
- **What P4 failing means, as the page set it.** The flag is not to be printed without its interval. The bar itself
  admitted rare classes drawn at a small fraction of their share, since it qualifies a class by an absolute difference
  (|m_c − p_c| ≤ 0.02). The 164 qualifying classes, grouped after the run by the ratio of drawn to true share M/K:

  | Group | Classes | Flagged above 0.10 |
  |---|---|---|
  | M/K < 0.5 or > 2 | 86 | 62 (a flag is right there) |
  | within a factor of 2, outside 0.8 to 1.25 | 19 | 4, at most 0.206 |
  | 0.8 ≤ M/K ≤ 1.25 | 59 | 3: Mayotte's Background, drawn at 99.9% to 100% against 99.5% |

  Mayotte's case is a real over-draw that absorbs nearly every crop.

  Over every class, not only P4's, 23 of the 83 drawn within 0.8 to 1.25 of their share are flagged above 0.10. They
  are mostly Background over-draws, which are real but small in relative terms: Martinique v1.3 draws 0.913 against
  0.816 (0.999), and Guyane's draws are flagged at 1.000. A few others: Sen1Floods11's two classes (0.206), Esri's
  Cropland (0.138), ForestNet's class 0 (0.123).

  So the flag's power follows the gap measured in sampling errors, which grows with the class's share. A large class
  drawn 10% off is flagged; a rare class drawn at half may not be. Neither relative nor absolute error alone predicts
  it.
- **What was read before freezing.** exp113's outputs, including which classes collapse, were read before this page,
  as it says. 13 of P3's 17 qualifying classes and all 10 of P5's are PLANTEUR's.
- **The audit.** The second route (`exp/exp115_crosscheck.py`, job 1472003, 28 s) was written by an agent that opened
  neither the run's script nor its outputs. It builds the 39 maps itself, with its own exact interval inversion and its
  own sums.
  - 831 class counts (K, M and A of 277 classes) are identical.
  - 3,530 class-level probabilities (706 cells × 5) agree within 5.7e-10.
  - 98 joint coverages agree within 8.6e-11.
  - Median widths differ by 0.005 on the three 200-unit AWF maps, which share one reference, where the two routes
    define a median differently.
  - All five verdicts are the same.
- **The review of this write-up.** A review before it was committed corrected several statements in the first draft:
  - the units counted, 706 cells against intervals;
  - Mayotte's crop share;
  - the predictions' conditions, left out of short summaries;
  - which collapsed crops reach probability 1.000;
  - the reading of P4.
- **What it shows.** On 39 real maps the per-class report's guarantees hold exactly, as they must by construction.
  - At 400 random labels it shows a class of 5% or more that the map draws at half its share or less, and bounds such
    a class's producer's accuracy, each with one miss among the graded classes.
  - From 1% to 5% it is less reliable. 10 of 17 such classes drawn at half their share or less are flagged with
    probability at least 0.8, and 5 of 15 with producer's accuracy at most 0.10 are bounded.
  - It flags real over-draws, including small ones on large classes. It is not reliable for classes under 1% of the
    map.
  - Simple random sampling only. Whether stratifying by confidence finds a rare class's omission with fewer labels is
    not graded.
