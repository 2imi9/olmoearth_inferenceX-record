# grade's intervals by a second route, and their coverage at the agency designs' shapes (exp118 preregistration)

**Status: frozen (26 October 2026), by me, at the owner's direction ("one by one, follow the prerequisite
principle", after the 1.18.0 score); the owner has not reviewed the predictions.** Written and frozen before any
statistic of this page was computed.

## The question

The three raters of 1.18.0 (26 October 2026, median 81) found two weaknesses in the evidence for `grade` (exp117):

- the check of its intervals was not independent: the package's interval code is a port of the design study's
  scripts, so "every interval equals the design study's" shows a faithful port, not a correct interval;
- the simulation behind the coverage figures had 300 draws per case, too few to tell normal theory's 0.937 from 0.95.

exp118 answers both. Arm A recomputes every interval exp117 recorded with a second implementation written from the
specification below alone. Arm B measures the coverage of the shipped intervals (1.18.1) and of normal theory with
thousands of draws, at the shapes of NLCD 2016's and GFC2020's designs.

## The specification (the second route is written from this section alone)

A design has sampled strata h = 1..H. Stratum h has a share W_h of the population: its units over all units for
sizes in units, its area over all area for sizes as areas. It has N_h units (infinite for a point frame) and n_h
labels drawn at random without replacement. A stratum read as a point frame but drawn as units has N_lo,h, the fewest
units it holds: floor(area / unit area) for sizes as areas with `--unit-area`. The shares of the strata with no label
add up to W_u. Sizes come from the weights as N_h = the rounded sum of the stratum's weights (exp117's NLCD weights all
give one whole number), or from the strata file.

**A linear quantity** is theta = sum_h W_h mu_h, with mu_h stratum h's mean of a value x in [0, 1] taking at most
three distinct values in a stratum. Its interval at level alpha:

1. c_lo = sqrt(2 ln(2/alpha) / V), with V = sum_h W_h^2 / (4 n_h); the mixture is K = 12 values c_k, geometrically
   spaced from c_lo to 10 c_lo (numpy.geomspace), with weight 1/12 each.
2. For one order of stratum h's labels X_0 .. X_{n-1} and one c, with lambda0 = c W_h / n_h,
   l_t = lambda0 (N_h - t) / N_h (lambda0 for a point frame), S_t = X_0 + .. + X_{t-1},
   centre_t = min((1/2 + S_t) / (t + 1), 0.9 / l_t), the stratum's log-term is
   sum_t [ ln(1 + l_t (X_t - centre_t)) + l_t (centre_t + S_t / (N_h - t)) ]
   (for a point frame the S_t / (N_h - t) term is 0). Its value for the stratum is the log of the AVERAGE of exp(log-term)
   over every distinct order of the stratum's labels, all orders counted with equal weight (the labels are
   exchangeable); it must be computed exactly, not by sampling orders.
3. A_k = sum_h (stratum h's value at c_k), minus, for every point-frame stratum with N_lo,h,
   c_k W_h (n_h - 1) / (2 (N_lo,h - n_h + 1)).
4. The lower end is the largest theta0 in [0, sum_h W_h] at which ln( (1/12) sum_k exp(A_k - c_k theta0) ) is at
   least ln(2/alpha); 0 when it is below at theta0 = 0, sum_h W_h when it is at least there. Locate it to 1e-12.
5. The upper end is sum_h W_h minus step 4's end for the values 1 - x, plus W_u, at most 1.
6. A difference of two maps' accuracies, d in {-1, 0, 1} per point, is the linear quantity x = (1 + d) / 2, mapped
   back as 2 theta - 1.

**A user's or producer's accuracy** is R = theta_y / (theta_y + theta_w), with y = 1 where the point counts for the
class and is right, w = 1 where it is in the denominator and wrong. (L_y, U_y) and (L_w, U_w) are the linear intervals
of y and w, each at level alpha/2. The interval is [L_y / (L_y + U_w), U_y / (U_y + L_w)], with 0 and 1 for empty
denominators, clipped to [0, 1].

**The pure-stratum rule** applies to the user's accuracy of a class whose map area fills exactly one declared stratum
o, when every stratum's size is a whole number of units and no stratum is unsampled. M is the number of strata declared
pure for no class and not labelled in full; gamma = alpha / 10 if M > 0, else 0. Stratum o's right share gets the exact
two-sided hypergeometric interval at confidence 1 - (alpha - gamma). Each such mixed stratum's count of labels mapped as
the class and right, and of those mapped as it and wrong, gets the same interval at confidence 1 - gamma / (2M); a mixed
stratum labelled in full uses its sample shares. With a_h, b_h those shares and p_o stratum o's,
lower = (W_o p_o,lo + sum W_h a_h,lo) / (W_o + sum W_h a_h,lo + sum W_h b_h,hi), and
upper = min(1, (W_o p_o,hi + sum W_h a_h,hi) / (W_o + sum W_h a_h,hi + sum W_h b_h,lo)).
The hypergeometric interval for k marked among n drawn from N: the smallest K with P(X >= k | K) > (1 - conf) / 2 and
the largest K with P(X <= k | K) > (1 - conf) / 2, divided by N, then widened to contain k / n.

**Normal theory** (Stehman 2014): theta-hat = sum_h W_h xbar_h with variance sum_h W_h^2 (1 - n_h/N_h) s_h^2 / n_h
(s_h^2 with divisor n_h - 1; a stratum of one label adds 0; 1 - n_h/N_h is 1 for a point frame). A ratio uses
u = y - R z with z = y + w and divides that variance by (sum_h W_h zbar_h)^2. The interval is estimate +/- 1.959963984540054
standard errors, unclipped.

**The rules.** A point is right when the map's class equals its reference label or an alternate label; the reference
class is the map's class where it equals an alternate, else the reference label.

## Arm A: the second route (P1)

An agent writes `exp/exp118_crosscheck/second_route.py` from the specification above and the data's layout. It may
read this plan, `exp/out/exp94_nlcd_points.csv`, `exp/out/exp94_gfc_points.csv`, `exp/out/exp117_gfc2020_strata_size.csv`
and the command lines in `exp/exp117_agency/e2e_run.sh`. It may not read `oe_inferencex/` (the package), `exp/exp117_agency/*.py` or
any other implementation of these intervals, and it imports neither the package nor those scripts; a test checks its
imports. It runs on the cluster, never on the laptop. It computes, for exp117's runs nlcd2016_l2_alt,
nlcd2016_l2_primary, nlcd2011_l2_alt, nlcd2016_8_alt (both maps and their difference) and gfc2020 (both maps and their
difference): every overall accuracy, difference, per-class user's and producer's accuracy and reference share, the
valid interval and the normal-theory one, and compares them with the JSON files in `exp/out/exp117_e2e/`.

## Arm B: coverage (P2, P3)

The package as released (1.18.1, from PyPI) on populations built from the points, so that the truth is known:

- **B1, NLCD (finite).** Each stratum of set B, 2016, a population of ten copies of each of its labelled points (10 n_h
  units); samples of n_h drawn without replacement in each stratum; the design read from the weights' sizes scaled to
  10 n_h (`Design` given N_h = 10 n_h and the published W_h). Quantities: overall accuracy by the primary-or-alternate
  rule; every class's user's accuracy (the pure-stratum rule, the 15 no-change strata declared) and producer's accuracy
  (the box) and reference share; NLCD minus LCMAP at eight classes (primary-or-alternate). 5,000 draws.
- **B2, GFC2020 (point frame).** In each stratum, points drawn independently from its labelled points (with
  replacement), n_h of them; the design as exp117 read it (areas, `--unit-area 1`). Quantities: both maps' overall
  accuracies, their difference, and each map's user's and producer's accuracy of both classes and the forest share.
  2,000 draws.

Draws from numpy SeedSequence([118, arm, quantity block, draw]); the population's own value of each quantity is its
truth. Coverage is the share of draws whose interval contains the truth; the Monte Carlo standard error at 0.95 is
0.0031 at 5,000 draws and 0.0049 at 2,000.

## Predictions (bars fixed now)

- **P1, the second route.** Every valid interval of arm A equals exp117's to within 1e-9 (linear quantities and
  boxes) and 1e-6 (the pure-stratum rule, whose hypergeometric tails may be decided in floating point near their level),
  and every normal-theory interval to within 1e-9. If not, the difference is traced to one of the two implementations
  before anything is recorded, and recorded either way.
- **P2, validity.** Every valid interval of arm B covers its truth on at least 0.95 minus three Monte Carlo standard
  errors of its draws (0.9407 at 5,000 draws, 0.9354 at 2,000), for every quantity.
- **P3, normal theory falls short somewhere.** On at least one quantity of arm B, normal theory covers below the same
  bar. Its coverage is reported for every quantity, P3 holding or not.
- **Descriptive, no bar.** Each quantity's mean width of the valid interval over normal theory's.

## What it cannot show

Arm B's populations are built from the samples, so they have the samples' shares and joint structure, not the agencies'
populations, which no one has labelled in full. Ten copies per point make each stratum's population small next to the
agency's (10 n_h, not millions of pixels), so the without-replacement draw is closer to a census than the agency's;
B2's draws with replacement take the other extreme.
