# Corrections to `coupling-project-brief.md`

Read alongside the existing brief. Where the two conflict, **this document wins**. The existing
brief is accurate about *what the code does*; it is wrong about *what the code should do* in the
places listed below.

Decision: **modify in place, do not re-fork.** Most of the diff is correct and expensive to
reproduce. The core error is localised to where Sinkhorn is applied, not to how it is implemented.

---

## 1. The central error: Sinkhorn is on the wrong axis

### The two-axis framing

The current brief treats Sinkhorn and MCMC as "two tractable estimators" of one object. They are
not — or rather, they should be, but only if both sit on the same axis. There are two independent
axes:

| Axis | What it decides | Legal values | Why |
|---|---|---|---|
| **Coupling** | which (x₀, x₁) pair, hence which x_t | genuine permutations only | must preserve marginals |
| **Target** | given x_t, what to regress toward | hard / soft-averaged / sampled | blending is correct here |

**Sinkhorn currently sits on the coupling axis. It must move to the target axis.**
**MCMC currently sits on the coupling axis. It must also move to the target axis.**

After the move, the coupling axis holds only `none` and `hungarian` (both hard permutations), and
the target axis holds `hard`, `sinkhorn`, `mcmc`.

### Why blending the noise is invalid (not merely suboptimal)

An alignment that uses x₁ to choose a transformation preserves the marginals **iff** the
transformation is a group element and p₀ is invariant under that group. Permutation matrices are
group elements; permuting i.i.d. Gaussian noise gives i.i.d. Gaussian noise, for *any* permutation,
including an x₁-dependent one. Doubly stochastic matrices are **not** group elements — they are
convex combinations of them (interior points of the Birkhoff polytope), and `x₀ ↦ P x₀` averages
independent Gaussians, so `Var = Σⱼ Pᵢⱼ² < 1`.

This is a **specification error, not an approximation error**. With infinite capacity and perfect
optimisation the model would still be wrong, because it is trained to transport a contracted,
x₁-correlated noise distribution while inference supplies plain N(0, I).

Rescaling by `1/sqrt(Σⱼ Pᵢⱼ²)` does **not** fix it. P is built from the cost between x₀ and x₁, so
the blend deliberately pulls the noise toward the data. Rescaling restores the variance but not the
x₁-correlation.

### Why blending the target IS correct

The existing brief §5 states that blending the data "gives non-physical targets (averaged atom
positions can land inside bonds)". **This is wrong and should be deleted.**

The model uses an x₁-prediction parameterisation (`fm.py:745`). The Bayes-optimal x₁-predictor is
`E[x₁ | x_t]` — a *posterior mean*, i.e. a blend. Squared-error regression converges to it
regardless. So:

- Blending the **input** (x_t) corrupts the distribution the model is trained on. Pure damage.
- Blending the **target** moves the regression target toward the quantity the loss is already
  estimating. Variance reduction, not corruption.

The "non-physical molecule" worry dissolves because ε = (1−t)² → 0 sharpens P onto a single
permutation as t→1, so the endpoint target is always a real molecule.

### Why MCMC belongs on the target axis too

Hungarian minimises transport cost **by definition**. Any softening on the coupling axis therefore
*increases* transport cost and can only make trajectories less straight. This exactly predicts the
observed result — MCMC significantly less straight than Hungarian across all three seeds
(p 1.7e-6 to 6e-17). That is not a finding about MCMC; it is a consequence of putting a
finite-temperature sampler where an argmin belongs.

On the target axis, MCMC becomes the **unbiased single-sample** counterpart to Sinkhorn's
**mean-field deterministic average** of the same posterior `p(π' | x_t)`. That is a genuine
like-for-like comparison and removes the confound §5 identifies.

Justification for using a single sample: the Bregman loss is linear in the target, so the expected
gradient depends on the target only through its mean. One draw from the correct posterior gives the
same expected gradient as the intractable exact average.

---

## 2. The current results are consistent with the artifact, not the hypothesis

Reported: Sinkhorn straighter (1.11–1.14 vs 1.16–1.21, p~1e-195), lower X̂₁ movement, **but worse**
Wasserstein bond-length (0.0138 vs 0.0102) and bond-angle (0.859 vs 0.623).

Noise contraction predicts exactly this pair of symptoms:

- **Straighter paths, mechanically.** Shrinking the prior toward its mean makes trajectories more
  radial. In the limit P = J/n, every path is a perfectly straight line from the origin — maximum
  straightness score, useless model. Straightness is a metric contraction can game.
- **Worse geometry**, from the train/test mismatch.

The posterior-mean hypothesis predicts straighter paths *and* equal-or-better geometry. So the
current evidence points at the bug.

**Treat all existing Sinkhorn numbers as uninterpretable.** Keep the runs, relabel them
exploratory. Do not delete.

### The control that settles it (cheap, one run)

Add an arm: Hungarian coupling, hard target, with the noise **artificially scaled** to match the
measured contraction profile — `x₀ → c(t)·x₀` with `c(t) = sqrt(Σⱼ Pᵢⱼ²)`, i.e. ~0.62 at t=0
rising to ~0.95 at t=0.99 (from the table in §5 of the existing brief).

- Reproduces the straightness gain → the effect was contraction. Coupling interpretation dead.
- Does not reproduce it → something real was happening, and it should survive into the fixed
  implementation.

Either outcome is informative and it costs one training run.

---

## 3. What to change in the code

### 3.1 Move Sinkhorn out of the coupling dispatch

In `interpolate.py`, the `sinkhorn` branch should no longer soft-permute the prior. x_t must be
built from a genuine hard permutation — reuse the existing `hungarian` path (or `none`, per the
coupling flag).

`_sinkhorn_couple` and `_mcmc_couple` as *coupling* methods go away. `COUPLING_TYPES` becomes
`["none", "hungarian"]`.

`molrepr.py:soft_permute` need not be deleted — the same blend operation is what the target side
needs. But the call site moves, and the target-side version blends `to_mols`, not `from_mols`.

### 3.2 Add a target axis

New flag `--target {hard,sinkhorn,mcmc}`, default `hard`. Independent of `--coupling`.

The soft target is computed **in the loss**, where x_t is available:

```
w(π') ∝ exp( −‖x_t − t·π'(x₁)‖² / (2(1−t)²) )
```

Two changes from the current Sinkhorn wiring:

1. **The cost must be computed from x_t, not from raw x₀/x₁.** `interpolate.py:377` currently uses
   `inter_distances(to_coords, from_coords)`, which is t-independent, with t bolted on afterwards
   as a temperature. That is not the posterior. The correct cost is between the *current state* x_t
   and each candidate `t·π'(x₁)`; ε = (1−t)² then arises naturally as the conditional path's own
   variance rather than being imposed.
2. **The target becomes `P @ x₁`** — coordinates, and one-hot atomics / bond types as soft labels.
   Soft labels are legitimate for cross-entropy by the same Bregman-linearity argument.

Note this makes the `CLAUDE.md` line the existing brief §5 flags as wrong ("discrete targets become
soft labels (P @ onehot)") **correct** under the fixed implementation. Update rather than delete.

### 3.3 Self-consistency caveat to instrument

x_t is built from π, and the posterior is then computed from x_t, so the posterior will be somewhat
peaked on π by construction. Not fatal — that is the structure of any posterior-mean estimator —
but it means the soft target resembles the hard one more when the coupling was already good.

**Log the entropy of P as a function of t** so the amount of actual blending is visible rather than
assumed.

### 3.4 Investigate `COUPLING_MIN_EPS`

At t=0.99 the measured ε reads 1e-3 where (1−0.99)² = 1e-4 — the clamp is binding, and
`Σⱼ Pᵢⱼ² = 0.896` shows the plan is still mixing ~1.12 atoms where it should be essentially hard.
The intended t→1 sharpening is not fully happening.

The solver is already log-space (`functional.py:460`), which is exactly what removes the underflow
motivating such a clamp. Determine whether the clamp is still needed; if it can be lowered or
removed, the designed schedule actually takes effect. As it stands it is an undocumented
hyperparameter doing real work, and reviewers will ask.

---

## 4. Evaluation corrections

### 4.1 MMFF → GFN2-xTB (the single biggest evaluation change)

`energy`, `strain`, `opt-rmsd`, `opt-energy-validity`, `energy-per-atom`, `strain-per-atom` are all
MMFF-derived and all correlated. Nikitin et al. (arXiv 2505.00169) show reference GEOM-Drugs
conformers score mean ΔE_relax ≈ 16 kcal/mol under MMFF but ≈ 0 under GFN2-xTB, because the dataset
was *built* by GFN2-xTB optimisation. MMFF's 15–20 kcal/mol error is larger than the effect being
measured, so MMFF comparisons "mask meaningful differences between models".

**Primary energy metric should be ΔE_relax under GFN2-xTB**, reported as **both median and mean** —
the distribution is heavily right-skewed (their SemlaFlow numbers: median 32.3, mean 91.0 ± 21.7),
and the two capture different failure modes. Median = typical geometry quality; mean = rate of
catastrophic failures.

Add the three geometry-deviation metrics from the same paper: mean **bond length**, **bond angle**,
and **torsion** differences between each generated molecule and *its own* GFN2-xTB-optimised
counterpart. These are more interpretable than distribution-level Wasserstein and are the field's
emerging standard.

Keep MMFF only as a coarse outlier filter. Demote, do not delete.

**Practical:** xTB is CPU work and embarrassingly parallel. The `b35bs.3.isambard` /
`b35bs.macs3.isambard` CPU allocation is currently unused — run xTB evaluation there and keep the
b5bg GPU hours for training. Also: 5000 molecules for the headline table, ~1000 per point for the
NFE sweep, to keep the cost tractable.

### 4.2 The valency table fix is a partial approximation

`metrics.py` was hand-corrected to `"C": {0: 4}`, `"N": {0: 3}`. Right direction, but Nikitin et al.
show that **neither 1 nor 1.5 is a universally correct aromatic bond order** — their solution is a
table indexed by `(element, n_aromatic_bonds, formal_charge, valency)` where the value is the
allowed *non-aromatic* bond order, or alternatively retraining on a kekulised dataset.

Use the reference implementation from `github.com/isayevlab/geom-drugs-3dgen-evaluation` rather
than a hand-patched table. Correcting the metric properly matters because this codebase is named
in that paper as affected.

### 4.3 PoseBusters — keep, but demote

Earlier advice was to skip it as redundant. Since it is already built and the `energy_ratio` trap
is already debugged, keeping it costs nothing. But it is not a replacement for RDKit validity
(different question: 3D geometric plausibility vs 2D graph chemistry), and the GFN2-xTB geometry
deviations are strictly more sensitive for this purpose. Secondary metric.

### 4.4 Statistics: the pairing is weaker than it looks

The `compare_arms.py` protocol matches arms on the **size sequence** drawn from the test set. That
is a legitimate blocking variable and worth keeping — but it is *not* true pairing. Arm A's
molecule i and arm B's molecule i are different molecules that happen to have the same atom count.
Wilcoxon signed-rank assumes genuinely paired observations.

Recommended: report the size-matched comparison as a variance-reduced descriptive, and use
**unpaired** tests (Mann-Whitney U, or bootstrap CIs on the difference of medians) for the formal
claim. Also note that p ~ 1e-195 at n=2000 reflects a systematic difference of unknown size, not
importance — report **effect sizes** alongside.

With 3 seeds, formal testing across seeds has almost no power. Better: show all three seed-level
values per arm so readers can see whether arms separate cleanly. If Sinkhorn's worst seed beats
Hungarian's best seed, that is persuasive without a p-value.

### 4.5 Straightness needs a companion metric

`trajectory-straightness` (path length / chord) is a reasonable measure, but as §2 above shows, it
is gameable by contraction. Pair it with **coupling transport cost** `E‖x₁ − x₀^π‖²` — a
training-time property of the pairing that involves no model at all. Together they distinguish
"the coupling changed" from "the learned paths straightened".

### 4.6 Target variance replaces permutation flip rate

The existing brief lists "target-variance diagnostic — how often the hard permutation flips" as
not-built. Note that flip rate only applies to hard-permutation arms; there is no π to flip under
Sinkhorn. The version that works across all arms: fix x₁ and t, resample x₀ many times, measure the
**variance of the resulting regression target**. Directly measures the gradient noise the method
claims to reduce, and is comparable across every arm.

---

## 5. Corrected experimental design

Kabsch **on** for all arms — do not run the Kabsch on/off 2×2. Reasons: it is the published
baseline (turning it off means the Hungarian arm is no longer Klein et al.); it is the harder test
(the effect must show up on top of rotation alignment); and under the two-axis framing Kabsch is a
coupling-side alignment while Sinkhorn is a target-side variance reduction, so overlap is unlikely
a priori. State the held-fixed factor explicitly in the paper.

Core factorial:

| | Hard target | Sinkhorn target | MCMC target |
|---|---|---|---|
| **No coupling** | floor | isolates target effect | isolates target effect |
| **Hungarian coupling** | Klein et al. baseline | **proposed** | **proposed** |

Plus the contraction control from §2. Three seeds throughout.

Expected interaction, worth stating as a prediction: the soft target is a *local* average over
permutations near the one that built x_t. A good coupling puts that neighbourhood in a sensible
place, so Hungarian + soft should beat none + soft by more than the Hungarian effect alone. The
competing prediction is that better couplings make the posterior *more* peaked, shrinking the
soft-vs-hard gap. Which wins is the empirical question and is why the factorial is worth running.

---

## 6. What to keep unchanged

Explicitly: the `--optimal_transport equivariant` → `--coupling` + `--kabsch_align` decomposition;
`sinkhorn` / `sinkhorn_batched` in `functional.py` (correct log-space solver, called from the wrong
place); `mcmc_permutation` (correct, applies a hard permutation, preserves the prior); the
`__init__.py` OpenMP/aarch64 fix (do not move or lazily import); `--seed` / `--run_name` /
per-run checkpoint dirs / `--resume_ckpt_path` / `--num_workers`; `record_trajectory` in
`_generate`; `geometry_metrics.py`; the 86 tests; the property that `paired_eval.py` calls the same
RDKit helpers as the aggregate metrics.

---

## 7. Still not built, in priority order

1. **NFE sweep** {1,2,5,10,20,50,100} — highest value. The mechanism predicts the gap *widens* at
   low NFE, and nothing tests that. Present as: (a) metric vs NFE per arm; (b) the *difference*
   vs NFE with a zero line, so "the gap widens" is the literal shape of the curve; (c) "NFE
   required to reach threshold τ" as a single interpretable number.
2. **GFN2-xTB pipeline** — see §4.1.
3. **The contraction control run** — see §2.
4. **Ryser estimator-bias comparison at n ≤ 12** — exact permanent marginals vs Sinkhorn vs MCMC on
   real cost matrices from the training loop. Cheap (CPU-seconds), disproportionate credibility:
   it quantifies the mean-field bias the argument rests on instead of asserting it.
5. **Target-variance diagnostic** — see §4.6.
6. **GEOM-Drugs** — everything so far is QM9.
