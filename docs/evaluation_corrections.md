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

Three seeds throughout.

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
3. **Ryser estimator-bias comparison at n ≤ 12** — exact permanent marginals vs Sinkhorn vs MCMC on
   real cost matrices from the training loop. Cheap (CPU-seconds), disproportionate credibility:
   it quantifies the mean-field bias the argument rests on instead of asserting it.
4. **Target-variance diagnostic** — see §4.6.
5. **GEOM-Drugs** — everything so far is QM9.
