# Readout design: theorems, predictions, preregistration for exp48

**Status, 17 September 2026.** Preregistered, not run; issue 13 closed as not planned when the evidence program closed at v1.0.0.

A theory note written before exp48 (issue 13) is run: derivations over recorded numbers (exp42-exp47), not results. Its published anchor for the cascade result is Jitkrittum et al., NeurIPS 2023 (arXiv 2307.02764). The numbers quoted are from the artifacts named; the predictions are the ones exp48 will be graded against.

Every number is from exp/out/ (exp42, exp43, exp46, exp47). Notation: finite index set I (windows or pixels), labels y : I -> {0,1},
decision d : I -> {0,1}, err(d) = {i : d(i) != y(i)}. Two heads h_z (token) and h_x (pixel). Fix = err(h_z) \ err(h_x),
Brk = err(h_x) \ err(h_z), Both = err(h_z) ∩ err(h_x). n = |I|, e_z = |err(h_z)|/n, F = |Fix|/n, B = |Brk|/n.

## T1  Reorder invariance
err(d) is a function of (d, y) only. A review order σ enters only through capture(σ, b) = |err(d) ∩ σ^{-1}[0, b n)|.
So every U+ combination of exp47 changed capture and could not change err. Proof: definitional.

## T2  Cascade identity
For G ⊆ I let d_G(i) = h_x(i) if i ∈ G else h_z(i). Then err(d_G) = (err(h_z) \ G) ⊔ (err(h_x) ∩ G), so
  |err(d_G)| = |err(h_z)| − |err(h_z) ∩ G| + |err(h_x) ∩ G|
             = |err(h_z)| − (|G∩Fix| + |G∩Both|) + (|G∩Brk| + |G∩Both|)
             = |err(h_z)| − |G∩Fix| + |G∩Brk|.                                                        ∎
(a) |err(d_G)| ≥ |err(h_z)| − |Fix|, equality iff Fix ⊆ G and G∩Brk = ∅  (oracle gate).
(b) the cascade helps iff |G∩Fix| > |G∩Brk|  (exact enrichment condition).
(c) G = I gives |err(h_x)| = |err(h_z)| − |Fix| + |Brk|  (full replacement realises F − B).
(d) uniform random G of size bn: E|G∩Fix| = b|Fix|, E|G∩Brk| = b|Brk|, so EΔ = b(F − B).
(e) with F(b) = |G∩Fix|/n, B(b) = |G∩Brk|/n and e(b) = (F(b)/B(b)) / (F/B):  Δ(b) > 0  iff  e(b) > B/F.
Instantiated (exp46, window level): B/F = 0.69 (Bolivia, A0), 0.52 (Bolivia, A4), 1.60 (test, A0), 0.91 (test, A4).
On the test split a pixel-statistics head needs the gate to enrich fixable-over-breakable 1.6×; the 3×3 head needs 0.91×, so
there even a random gate helps.

## T5  Marginal gain and the optimal threshold gate
Condition on c:  P(Fix | c) − P(Brk | c) = [p_z(c) − P(Both|c)] − [p_x(c) − P(Both|c)] = p_z(c) − p_x(c),
with p_z(c) = P(h_z wrong | c), p_x(c) = P(h_x wrong | c). The Both term cancels: the marginal value of gating a window at
confidence c depends only on the two conditional error rates, not on their dependence (φ). Hence among all gates measurable
in c the optimum is G* = {c : p_z(c) > p_x(c)}. If p_z is non-increasing and p_x non-decreasing in c, G* = {c < τ*} with
p_z(τ*) = p_x(τ*) (G* = ∅ if p_x ≥ p_z everywhere). Finite sample: sort by c; the running sum of 1[Fix] − 1[Brk] has increments
of expectation p_z − p_x, unimodal under the monotonicity, maximal at the crossing.
This is the published cascade result: Jitkrittum et al., "When does confidence-based cascade deferral suffice?" (NeurIPS 2023, arXiv 2307.02764), Prop. 3.1: defer iff η_{h2}(x) − η_{h1}(x) > cost, i.e. iff p_1(x) − p_2(x) > cost; Lemma 4.1: confidence-based deferral is optimal iff η_1 and η_1 − η_2 order instances the same way, e.g. when the second model's error is uniform. Their three failure modes — a specialist second model, label noise, distribution shift — all apply here: h_x is a specialist (wins Bolivia, loses the test split), the hand labels carry noise, and Bolivia is a single event for a head trained on the multi-region valid split.
Assumption to check in exp48, not a theorem: the shape of p_x(c). R5 is useless iff p_x(c) ≥ p_z(c) for every c, i.e. the pixel
head is at least as wrong as the token head wherever the token head is unsure.

### Prediction under conditional independence (h_x correctness ⊥ c | h_z correctness)
Then f_G := P(h_x right | h_z wrong, G) = F/e_z and k_G := P(h_x wrong | h_z right, G) = B/(1−e_z). With capture κ(b) = P(G | h_z wrong)
(exp42, W1 confidence): F(b) = κ(b) F,  B(b) = (b − κ(b) e_z) · B/(1−e_z),  Δ(b) = F(b) − B(b).

| testbed | h_x | b | capture(b) | F(b) pp | B(b) pp | Δ(b) pp, independence | enrichment implied | enrichment needed (B/F) | R5 helps iff k_G < α·f_G, α = | f_G under independence | k_G under independence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bolivia | A0 no encoder | 0.05 | 0.308 | 0.81 | 0.05 | +0.76 | 12.3× | 0.69× | 1.19 | 0.297 | 0.020 |
| bolivia | A0 no encoder | 0.10 | 0.517 | 1.36 | 0.11 | +1.25 | 8.7× | 0.69× | 0.84 | 0.297 | 0.020 |
| bolivia | A0 no encoder | 0.20 | 0.772 | 2.03 | 0.26 | +1.77 | 5.3× | 0.69× | 0.52 | 0.297 | 0.020 |
| bolivia | A4 S2 3x3 linear | 0.05 | 0.308 | 1.12 | 0.05 | +1.07 | 12.3× | 0.52× | 1.19 | 0.412 | 0.021 |
| bolivia | A4 S2 3x3 linear | 0.10 | 0.517 | 1.88 | 0.11 | +1.77 | 8.7× | 0.52× | 0.84 | 0.412 | 0.021 |
| bolivia | A4 S2 3x3 linear | 0.20 | 0.772 | 2.81 | 0.27 | +2.54 | 5.3× | 0.52× | 0.52 | 0.412 | 0.021 |
| test | A0 no encoder | 0.05 | 0.444 | 0.76 | 0.08 | +0.68 | 14.6× | 1.60× | 0.72 | 0.365 | 0.029 |
| test | A0 no encoder | 0.10 | 0.660 | 1.14 | 0.20 | +0.94 | 9.1× | 1.60× | 0.45 | 0.365 | 0.029 |
| test | A0 no encoder | 0.20 | 0.819 | 1.41 | 0.47 | +0.95 | 4.8× | 1.60× | 0.24 | 0.365 | 0.029 |
| test | A4 S2 3x3 linear | 0.05 | 0.444 | 0.66 | 0.04 | +0.62 | 14.6× | 0.91× | 0.72 | 0.316 | 0.014 |
| test | A4 S2 3x3 linear | 0.10 | 0.660 | 0.99 | 0.10 | +0.89 | 9.1× | 0.91× | 0.45 | 0.316 | 0.014 |
| test | A4 S2 3x3 linear | 0.20 | 0.819 | 1.22 | 0.23 | +0.99 | 4.8× | 0.91× | 0.24 | 0.316 | 0.014 |

Break-even frontier, no independence assumed:  Δ(b) = κ(b) e_z f_G − (b − κ(b) e_z) k_G, so
  R5 helps  iff  k_G < α(b) · f_G,   α(b) = κ(b) e_z / (b − κ(b) e_z):  α(0.10) = 0.84 Bolivia / 0.45 test;  α(0.20) = 0.52 / 0.24.
Under independence f_G ≈ 0.30–0.41 and k_G ≈ 0.02, an order of magnitude inside the frontier, which is why the independence
prediction is +0.9 to +2.5 pp. The whole question is the assumption. What breaks it: the errors confidence catches are the
ambiguous windows where both heads fail (f_G ≪ F/e_z) and where the pixel head also breaks correct windows (k_G ≫ B/(1−e_z)).
φ = 0.71 / 0.55 says the marginal dependence is strong; the dependence conditional on c is exactly what exp48 measures (f_G, k_G at
b = 0.05 / 0.10 / 0.20).

### Q2(v)  A gate that loses as a ranker can still win as a gate
With r_G = |G ∩ err(h_z)|/n (what exp31 scored typicality on): Δ_G = r_G f_G − (|G|/n − r_G) k_G. A typicality gate G_t with
r_t < r_c beats the confidence gate iff r_t f_t − (b − r_t) k_t > r_c f_c − (b − r_c) k_c. exp31 measured r only, never f or k, so it
does not rule this out. Prior is low: typicality tracked the NDWI-gradient control (Spearman ≤ 0.58), i.e. texture, which is where
both heads fail (low f). One extra column in exp48 settles it.

## T3  Block-constant floor, and a correction
Partition I into blocks β with labelled counts n0(β), n1(β). If d is constant on every β then |err(d) ∩ β| ≥ min(n0, n1), so
|err(d)| ≥ Σ_β min(n0(β), n1(β)) =: floor, attained by the per-block majority. d = y has |err| = 0, so the floor does not constrain
per-pixel rules. Recorded: floor = 3.04 % Bolivia, 2.90 % test (exp43).
Correction to what I said earlier ("9.3 = 6.3 + 3.0"): W1 is not block-constant. A pixel (r, c) lies in window (⌊(r−s)/4⌋, ⌊(c−s)/4⌋)
at shift s ∈ {0,1,2,3}; two pixels share all four windows only if ⌊(r−s)/4⌋ agrees for every s, and four 1-D tilings with shifts
0..3 have a breakpoint at every integer, so the common refinement is single pixels. W1 is already a per-pixel rule; T3 binds W0 and
any single-tiling rule, not W1. What T3 licenses: R2 removes the tiling from the decision altogether, so R2 vs W1 is "read x
directly" against "resolve sub-token structure through four overlapping block decisions". No theorem orders those; exp44 says
the second route is saturated (+0.04 pp for twelve more offsets). The measured sub-token problem after W1 is the impure-window
stratum: 42.6 % / 56.5 % of W1 errors at 34.5 % / 27.8 % error rate.

## T4  Hypothesis-class nesting
Lin_z = {sign(w·z + b)} embeds in Lin_zx = {sign(w·z + v·x + b)} by v = 0, so the empirical-risk minimum over Lin_zx is ≤ that over
Lin_z and ≤ that over Lin_x. In-sample only; out of sample the joint head has 12 more parameters against ≤ 135,000 windows.
Cascade ∉ Lin_zx. Take z, x ∈ R and the cascade d = sign(z) if |z| > 1 else sign(x). Six points:
(2, −100)+, (−2, 100)−, (0.5, 0.01)+, (0.5, −0.01)−, (−0.5, 0.01)+, (−0.5, −0.01)−. A linear rule w1 z + w2 x + b needs w2 > 0 (sign
flips in x at fixed z), |0.5 w1 + b| < 0.01 w2 and |−0.5 w1 + b| < 0.01 w2, hence |w1| < 0.02 w2 and |b| < 0.01 w2; then at
(2, −100): 2 w1 − 100 w2 + b < (0.04 − 100 + 0.01) w2 < 0, contradicting its + label.                                        ∎
A two-layer ReLU head represents the cascade. With a gate indicator g ∈ {0,1} as input and |z|, |x| ≤ M:
z·g = ReLU(z + M(g−1)) − ReLU(−z + M(g−1)),  x·(1−g) = ReLU(x − M g) − ReLU(−x − M g); four hidden units compute
s = g z + (1−g) x and sign(s) is the cascade. With c instead of g, 1[c < τ] = ReLU(k(τ−c) + 1) − ReLU(k(τ−c)) exactly on any finite
sample with a margin at τ, for k large enough. So MLP-R2 ⊇ R5 in expressivity. Finite sample: R5 inherits its gate from h_z's margin
for free; the MLP must learn the switch from the 10–20 % of windows in the gated region (13–27 k windows), at which size a
two-layer head's excess risk is far below 0.002 in expectation. A loss of MLP-R2 to R5 would therefore point to optimisation or
regularisation, not to sample size.

## Q4  Information bound
Binary Y, posteriors η_Z = P(Y=1|Z), η_ZX = P(Y=1|Z,X):
e*(Z) − e*(Z,X) = E[min(η_Z, 1−η_Z) − min(η_ZX, 1−η_ZX)] ≤ E|η_Z − η_ZX| = E[TV(P(Y|Z,X), P(Y|Z))] ≤ E√(KL/2) ≤ √(E[KL]/2) = √(I(Y;X|Z)/2)
(min(·, 1−·) is 1-Lipschitz; Pinsker; Jensen; nats). Estimator: the held-out cross-entropy drop CE(h_z) − CE(h_zx) — a point
estimate, neither an upper nor a lower bound on I with restricted heads. Threshold: the bound falls below the 0.002 minimum effect
only if I < 2·(0.002)² = 8×10⁻⁶ nats, which no measurable cross-entropy drop will be under. The information bound never blocks
the run; the binding questions are T5's enrichment and T4's out-of-sample behaviour.

## Q5  The Bolivia puzzle
h_x (fifteen pixel statistics, no encoder) beats h_z by 0.83 pp at window level on Bolivia, and ~90 % of windows are pure, so the
gain is not sub-token: for one flood event the raw reflectance statistics are a better input to a linear head than the frozen
768-d token. Two readings, separable in exp48: (i) the token has lost event-relevant spectral-level information (an NDWI-like
threshold is recoverable from x, not from z); (ii) the head trained on the multi-region valid split transfers to Bolivia better
through x than through z (shift in z). Either way the joint head leans on x on Bolivia, so its confidence ranks like NDWI level
there: exp47's Bolivia exception does not dissolve under R2, it is explained — a confidence that beats NDWI on Bolivia would have
to come from a head that ignores x, which is a worse head. On the test split (A1 > A0 by 1.03 pp) z dominates and R2's confidence
keeps beating NDWI level. Stratified: R2's gain on Bolivia sits mainly on pure windows; on test, mainly on impure windows.

## Q6  Preregistration for exp48 (pixel level, W1 baseline, min effect 0.002, exact sign test over tiles, both testbeds)
Primary: R2-linear vs W1, pixel accuracy, one-sided, both testbeds. Expected + on both: Bolivia +0.4 to +0.9 pp (pure-window
driven), test +0.1 to +0.4 pp (impure-window driven). Design falsified if R2-linear is below W1 by > 0.002 on either testbed.
Secondary:
- R1 (3×3 context): Bolivia +0.8 to +1.7 pp, test below the minimum effect (window-level +0.13). Prediction: passes Bolivia, fails test.
- R2-MLP over R2-linear: +0.2 to +0.5 pp (A3 − A1 = +0.85 / +0.39 at window level).
- R5 (b = 0.10, pixel head): positive on Bolivia; positive on test only if k_G < 0.45 f_G. Independence point prediction
  +1.3 / +0.9 pp, discounted to +0.3 to +1.2 (Bolivia) and −0.3 to +0.9 (test). R2-MLP ≥ R5 (T4).
- Typicality gate (Q2(v)): one column; expected to lose to the confidence gate.
- Ranker check (exp47 protocol) on each arm's own errors against NDWI level: R2's confidence ties NDWI on Bolivia, beats it on test.
Record per pixel: each arm's decision, c, purity of the shift-0 window, NDWI level; per window: p_z(c), p_x(c) in deciles of c;
f_G, k_G at b = 0.05 / 0.10 / 0.20; held-out cross-entropy of every head.
Framework falsifier: exp46's window-level F − B for A0 on Bolivia (+0.83 pp) failing to reproduce in sign at pixel level — the
window accounting not transferring to the shipped per-pixel decision, the exp45→exp47 failure mode.
Cost: heads on cached features, no new encoder passes; one optional 2× upsampled arm. A job of exp44's size.
