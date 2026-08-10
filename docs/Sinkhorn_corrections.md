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

