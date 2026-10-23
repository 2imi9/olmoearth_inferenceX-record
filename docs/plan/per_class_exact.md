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

40 maps in all.

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
