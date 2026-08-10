# Project context

## Research goal

Testing whether a **soft (posterior-mean) regression target over permutations beats the hard
Hungarian target** in flow matching for de novo 3D molecule generation.

Background: Klein et al. (Equivariant Flow Matching, NeurIPS 2023) align noise to data using
the Hungarian algorithm (hard argmin permutation) plus Kabsch (rotation). This is now standard
and is what SemlaFlow uses.

The hypothesis: the hard argmin is a **biased point estimate** of a posterior mean over
permutations. The exact soft version requires computing permanent marginals of the cost
matrix, which is #P-hard. Two tractable estimators:

1. **Sinkhorn** — deterministic, O(n^2) batched matmuls, differentiable, mean-field biased
   (systematically more diffuse than truth). This is the primary method.
2. **MCMC over permutations** — Metropolis with transposition proposals, unbiased single
   sample from the same posterior. Secondary/comparison method.

Structural precedent: "Efficient Molecular Conformer Generation with SO(3)-Averaged Flow
Matching and Reflow" (Cao et al., ICML 2025) showed soft rotation averaging beats hard Kabsch.
This project is the permutation analogue. Their case had a closed form (matrix Fisher on a
compact Lie group); the permutation case does not, hence the estimators above.

## THE TWO AXES — read this before touching coupling code

The single most important structural fact about this project. There are two **independent**
axes, and the code currently conflates them (see "Known defects" below):

| Axis | What it decides | Legal values | Why |
|---|---|---|---|
| **Coupling** | which (x0, x1) pair, hence which x_t | genuine permutations ONLY | must preserve marginals |
| **Target** | given x_t, what to regress toward | hard / soft-averaged / sampled | blending is correct here |

**Sinkhorn and MCMC belong on the target axis, not the coupling axis.**
After the fix: coupling axis holds `{none, hungarian}`; target axis holds `{hard, sinkhorn, mcmc}`.

### Why blending the noise is invalid (specification error, not approximation error)

An x1-dependent alignment preserves the marginals **iff** the transformation is a group element
and p_0 is invariant under that group. Permutation matrices are group elements — permuting
i.i.d. Gaussian noise gives i.i.d. Gaussian noise for *any* permutation, including an
x1-dependent one. Doubly stochastic matrices are **not** group elements; they are convex
combinations of them (interior of the Birkhoff polytope). So `x0 -> P x0` averages independent
Gaussians and gives `Var = sum_j P_ij^2 < 1`.

With infinite capacity and perfect optimisation the model would still be wrong: it is trained to
transport a contracted, x1-correlated noise distribution while inference supplies plain N(0, I).

Rescaling by `1/sqrt(sum_j P_ij^2)` does **not** fix this. P is built from the x0-x1 cost, so the
blend deliberately pulls noise toward the data; rescaling restores the variance but not the
x1-correlation.

### Why blending the target IS correct

The model uses an x1-prediction parameterisation (`fm.py:745`). The Bayes-optimal x1-predictor is
`E[x1 | x_t]` — a posterior mean, i.e. a blend. Squared-error regression converges to it anyway.

- Blending the **input** (x_t) corrupts the distribution the model trains on. Pure damage.
- Blending the **target** moves the target toward what the loss is already estimating.
  Variance reduction, not corruption.

The "averaged atom positions land inside bonds / non-physical molecule" worry is **wrong** — do
not reintroduce it. `eps = (1-t)^2 -> 0` sharpens P onto a single permutation as t -> 1, so the
endpoint target is always a real molecule.

### Why MCMC belongs on the target axis too

Hungarian minimises transport cost **by definition**, so *any* softening on the coupling axis
increases transport cost and can only make trajectories less straight. That is exactly the
observed MCMC result (less straight than Hungarian, all 3 seeds, p 1.7e-6 to 6e-17) — a
consequence of putting a finite-temperature sampler where an argmin belongs, not a finding.

On the target axis, MCMC is the **unbiased single-sample** counterpart to Sinkhorn's
**mean-field deterministic average** of the same posterior `p(pi' | x_t)`. Single-sample is
justified because the Bregman loss is linear in the target, so the expected gradient depends on
the target only through its mean.

### The posterior, stated correctly

```
w(pi') ∝ exp( -||x_t - t * pi'(x1)||^2 / (2 (1-t)^2) )
```

Two things this fixes versus the current wiring:

1. **Cost must be computed from x_t**, not from raw x0/x1. `interpolate.py:377` currently uses
   `inter_distances(to_coords, from_coords)`, which is t-independent, with t bolted on afterwards
   as a temperature. That is not the posterior. The correct cost is between the *current state*
   x_t and each candidate `t * pi'(x1)`; `eps = (1-t)^2` then arises naturally as the conditional
   path's own variance rather than being imposed. No free hyperparameter either way.
2. **The target becomes `P @ x1`** — coordinates, plus one-hot atomics / bond types as soft labels.

## Important correctness constraints

- **Must be de novo generation, not conformer generation.** Alignment over a group G is only
  valid if BOTH p_0 and p_1 are G-invariant. For conformer generation the graph pins atom
  identities, so p_1 is NOT S_n-invariant and full-permutation alignment is invalid (only
  Aut(G) is legitimate, which is small enough to enumerate exactly). In de novo generation
  coordinates+types+bonds are permuted jointly, so p_1 IS S_n-invariant.
- **Prior must be i.i.d. Gaussian per node.** A harmonic prior correlates noise along bonds,
  breaks exchangeability, and invalidates the alignment.
- **The coupling must apply a hard permutation**, jointly to coordinates, atom types, bond types
  and charges. Never a doubly stochastic matrix — see the two-axis section above.
- **Soft targets are legitimate.** Under a soft target, discrete targets become soft labels
  (P @ onehot). Cross-entropy is the Bregman divergence with Phi = u log u - u, so the target
  enters linearly. State explicitly and ablate.
- **Self-consistency caveat.** x_t is built from pi, and the posterior is then computed from x_t,
  so the posterior is somewhat peaked on pi by construction. Not fatal — that is the structure of
  any posterior-mean estimator — but the soft target resembles the hard one more when the coupling
  was already good. **Log the entropy of P as a function of t** so the amount of actual blending
  is visible rather than assumed.

## Known defects in the current code (not yet fixed)

Authoritative source: `docs/Sinkhorn_corrections.md` (`docs/evaluation_corrections.md` is a
verbatim copy of its sections 4-7). Where anything conflicts with the older
`docs/coupling-project-brief.md`, the corrections document wins. Decision: **modify in place, do
not re-fork** — most of the diff is correct and expensive to reproduce, and the error is localised
to *where* Sinkhorn is applied, not to how it is implemented.

1. **[FIXED] Sinkhorn soft-permuted the prior.** `_sinkhorn_couple` called `soft_permute` on the
   noise — invalid, see the two-axis section. Both `_sinkhorn_couple` and `_mcmc_couple` have been
   removed from the coupling dispatch and `COUPLING_TYPES` is now `["none", "hungarian"]`. The
   solvers in `functional.py` and `molrepr.py:soft_permute` are untouched — they are correct and
   the target side needs them.
2. **[TODO] There is no target axis yet.** Add `--target {hard,sinkhorn,mcmc}` (default `hard`),
   independent of `--coupling`, computed **in the loss** where x_t is available. `soft_permute`
   is then applied to `to_mols`, not `from_mols`. This is where the four removed hyperparameters
   (`sinkhorn_n_iters`, `mcmc_n_iters`, `mcmc_proposal`, `mcmc_knn_k`) come back — they were
   deleted with the dispatch rather than left dangling; recover them from
   `git show HEAD~1:semlaflow/data/interpolate.py` when wiring the target side.
3. **[TODO] The cost must be computed from x_t.** The removed code used
   `inter_distances(to_coords, from_coords)`, which is t-independent, with t bolted on afterwards
   as a temperature. See "The posterior, stated correctly" above. Do not reproduce that when
   rebuilding on the target axis.
4. **[TODO] The eps clamp was binding.** `COUPLING_MIN_EPS = 1e-3` was removed with the dead code,
   but the question it raises must be settled when the schedule is rebuilt: at t=0.99 the measured
   eps read 1e-3 where (1-0.99)^2 = 1e-4, and `sum_j P_ij^2 = 0.896` showed the plan still mixing
   ~1.12 atoms where it should be essentially hard, so the intended t->1 sharpening was not
   happening. The solver is already log-space (`functional.py:460`), which is exactly what removes
   the underflow that motivates such a clamp. Do not reintroduce a clamp without establishing it
   is needed — an undocumented hyperparameter doing real work is something reviewers will ask
   about.

### All existing Sinkhorn/MCMC numbers are uninterpretable

Reported so far: Sinkhorn straighter (1.11-1.14 vs Hungarian 1.16-1.21, p~1e-195), lower X-hat_1
movement, **but worse** Wasserstein bond-length (0.0138 vs 0.0102) and bond-angle (0.859 vs 0.623).

Noise contraction predicts exactly this pair of symptoms. Shrinking the prior toward its mean makes
trajectories more radial — in the limit P = J/n every path is a perfectly straight line from the
origin, i.e. maximum straightness and a useless model. **Straightness is a metric that contraction
can game.** The geometry regression then follows from the train/test mismatch. The posterior-mean
hypothesis, by contrast, predicts straighter paths *and* equal-or-better geometry.

**Keep the runs, relabel them exploratory, do not delete, do not cite as evidence.**

### The control that settles it (one training run)

Hungarian coupling, hard target, noise artificially scaled to match the measured contraction:
`x0 -> c(t) * x0` with `c(t) = sqrt(sum_j P_ij^2)`, i.e. ~0.62 at t=0 rising to ~0.95 at t=0.99.

- Reproduces the straightness gain -> the effect was contraction, coupling interpretation dead.
- Does not reproduce it -> something real was happening and should survive into the fixed
  implementation.

Either outcome is informative.

### What to keep unchanged

The `--optimal_transport equivariant` -> `--coupling` + `--kabsch_align` decomposition;
`sinkhorn` / `sinkhorn_batched` in `functional.py` (correct log-space solver, called from the
wrong place); `mcmc_permutation` (correct, applies a hard permutation, preserves the prior);
the `__init__.py` OpenMP/aarch64 fix (do NOT move or lazily import); `--seed` / `--run_name` /
per-run checkpoint dirs / `--resume_ckpt_path` / `--num_workers`; `record_trajectory` in
`_generate`; `geometry_metrics.py`; the 86 tests; the property that `paired_eval.py` calls the
same RDKit helpers as the aggregate metrics.

## Codebase

SemlaFlow: https://github.com/rssrwn/semla-flow (Irwin et al., AISTATS 2025)

Chosen because it is flow-matching (not diffusion), E(3)-equivariant, fast (20 sampling steps),
has published numbers to reproduce, and already contains the Hungarian alignment as an
isolated step ("scale optimal transport") — so the change is a function swap.

- Scripts: `python -m semlaflow.<script>` where script is preprocess / train / evaluate / predict
- Args are documented at the bottom of each script; defaults at the top
- Defaults target GEOM Drugs. For QM9: bond_loss_weight 0.5, warm_up_steps 2000, epochs ~300
- NOT pip-installable (no setup.py/pyproject.toml) — run from repo root or set PYTHONPATH
- Data: preprocessed `smol` folders from the authors' Google Drive; pass `--data_path .../smol`
- Pretrained checkpoints also on the Drive; `--ckpt_path .../qm9.ckpt`
- Tests: `python -m unittest -v tests/*.py` (CPU only, no GPU needed)

One breaking CLI change in this fork: **`--optimal_transport equivariant` no longer exists.**
Upstream bundled permutation and rotation behind one `equivariant_ot` boolean; this fork splits
them. `--optimal_transport` now takes only `none`/`batch`/`scale`. Upstream's default is
reproduced by `--optimal_transport none --coupling hungarian --kabsch_align`, which is the fork
default. Upstream branch point is `3f43103`; `git diff 3f43103 HEAD` shows the full diff.

## Development workflow: three tiers

Iterate fast locally, only pay the Isambard queue cost for what genuinely needs a GPU.

1. **Local Mac** — fastest iteration. Write/edit code, unit test the Sinkhorn coupling
   against `scipy.optimize.linear_sum_assignment`, run `tests/`, run tiny CPU forward passes
   to catch wiring bugs (shapes, missing args, exceptions). Mac is ARM (Apple Silicon), same
   architecture family as GH200's Grace CPU, so most wheels that work here also work there.
2. **Isambard login node** — no GPU, but the real linux-aarch64 env. Sanity check that the
   package imports and tests pass in the actual production environment. Zero Slurm/queue cost.
3. **Isambard compute node (batch job)** — only for things that need a real GPU: actual
   training speed, memory footprint, real experiments. Push here only once tiers 1-2 are clean.

Sync between Mac and Isambard via **git**, not manual copying:
```bash
# Mac
git add . && git commit -m "..." && git push
# Isambard
cd /projects/b5bg/barkern.b5bg/semla-flow && git pull
```
Working on a fork of rssrwn/semla-flow, branch `sinkhorn-coupling`.

## Local Mac environment

Local venv is a **sanity-checking mirror only** — not where real numbers come from, so no
need to match exact versions here, just need things importable enough to catch code bugs
before pushing to Isambard.

Gotcha hit and fixed: default `python3` on this Mac was 3.13, which has no prebuilt
macOS-arm64 wheel for `scipy==1.11.4` (predates 3.13), causing a from-source build to fail
(`Compiler cython cannot compile programs`). Fix: use Python 3.11 to match what the repo's
`environment.yaml` targets, and drop exact version pins locally (let pip pick current
compatible versions):

```bash
cd ~/Desktop/semla-flow   # local clone location
python3.11 -m venv venv   # brew install python@3.11 if not present
source venv/bin/activate
pip install torch numpy pandas scipy rdkit lightning torchmetrics typing_extensions tqdm wandb ipython
```

Also note: if a conda `base` env auto-activates in new shells (shows as `(base)` alongside
`(venv)` in the prompt), it's cosmetic as long as `which python` / `which pip` point inside
`venv/bin/`, but cleaner to `conda deactivate` before `source venv/bin/activate`, or run
`conda config --set auto_activate_base false` once to stop it happening automatically.

Verify:
```bash
python -c "import torch, numpy, pandas, scipy, rdkit, lightning, torchmetrics; print('ok')"
python -m unittest -v tests/*.py
```

## Isambard environment (Isambard-AI Phase 2, Bristol)

GH200 = **linux-aarch64** (ARM) + Hopper GPU. No conda module available; using a venv (NOT a
mirror this time — this is where real results come from, exact pins matter).

```bash
module load cray-python/3.11.7
source /projects/b5bg/barkern.b5bg/venvs/equinv/bin/activate
cd /projects/b5bg/barkern.b5bg/semla-flow
```

Note: the repo's environment.yaml is named `equinv` (legacy name) though the README says
`semlaflow`. `pytorch-cuda=12.1` in that yaml is x86-only and must be skipped; torch is
installed from the PyTorch wheel index instead (`pip install torch --index-url
https://download.pytorch.org/whl/cu124`, or cu126/cu128 if that has no aarch64 build).

Dependencies are unusually light for this domain — no DGL, no torch-geometric, no
torch-scatter. Semla is pure PyTorch with dense attention. This is why ARM works cleanly.

### Slurm

- Account: `brics.b5bg` (NOT `b5bg`) — GPU allocation, use for training only
- CPU allocation `b35bs.3.isambard` / `b35bs.macs3.isambard` is currently unused. **GFN2-xTB
  evaluation is CPU work and embarrassingly parallel — run it there** and keep b5bg GPU hours
  for training.
- Partition: `workq`
- Accounting: 1 node = 4 GH200, so 1 NHR = 4 GPU-hours; single-GPU job = 0.25 NHR/hour
- Cluster is heavily loaded; interactive `--pty` jobs get stuck on `Reason: Priority`.
  **Prefer short batch jobs over interactive sessions.**
- Login node has no GPU — `torch.cuda.is_available()` correctly returns False there

Batch job template:
```bash
#!/bin/bash
#SBATCH --account=brics.b5bg
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/projects/b5bg/barkern.b5bg/runs/%x-%j.out

module load cray-python/3.11.7
source /projects/b5bg/barkern.b5bg/venvs/equinv/bin/activate
cd /projects/b5bg/barkern.b5bg/semla-flow
python -m semlaflow.train --data_path ... <args>
```

### Paths (Isambard)

- Repo: `/projects/b5bg/barkern.b5bg/semla-flow`
- Venv: `/projects/b5bg/barkern.b5bg/venvs/equinv`
- Data: `/projects/b5bg/barkern.b5bg/data`
- Job outputs: `/projects/b5bg/barkern.b5bg/runs`

## Plan / status

Environment, coupling implementation and evaluation harness are built. QM9, 300 epochs,
`--optimal_transport none --kabsch_align`, seeds 12345/23456/34567 — all four old coupling arms
trained, checkpoints in `checkpoints/<run_name>/` (naming is inconsistent for historical reasons;
normalise with `--label_a`/`--label_b` at the call site). Those Sinkhorn/MCMC results are
exploratory only — see "Known defects".

Next, in priority order:

1. [done] **Sinkhorn and MCMC removed from the coupling dispatch.**
       `COUPLING_TYPES = ["none","hungarian"]`; solvers and `soft_permute` kept.
2. [ ] **Add the target axis.** `--target {hard,sinkhorn,mcmc}` computed in the loss, cost from
       x_t, eps schedule with the clamp question settled. Unit-test as before: as eps -> 0 the
       soft plan converges to scipy's `linear_sum_assignment`; rows/cols sum to 1; log-space
       stable at small eps. Add an entropy-of-P-vs-t log.
3. [ ] **NFE sweep** {1,2,5,10,20,50,100}. Highest-value missing experiment: the mechanism
       predicts the gap *widens* at low NFE and nothing tests that. Everything so far ran at a
       fixed 100 steps; `--integration_steps` already exists on both scripts, so this is a sweep
       harness, not new metric code. Present as (a) metric vs NFE per arm; (b) the *difference*
       vs NFE with a zero line, so "the gap widens" is the literal shape of the curve; (c) "NFE
       required to reach threshold tau" as a single interpretable number.
4. [ ] **GFN2-xTB pipeline** (see Metrics). New dependency, unverified on aarch64.
5. [ ] **The contraction control run** (see Known defects).
6. [ ] **Ryser estimator-bias comparison at n <= 12** — exact permanent marginals vs Sinkhorn vs
       MCMC on real cost matrices from the training loop. Cheap (CPU-seconds), disproportionate
       credibility: quantifies the mean-field bias the argument rests on instead of asserting it.
7. [ ] **Target-variance diagnostic** (see Key plots).
8. [ ] **GEOM-Drugs** — everything so far is QM9.

## Experimental design

Fix architecture, data, optimizer, seeds, sampler. Hold SemlaFlow's "scale OT" (size handling)
fixed across all arms; it is orthogonal to the claim.

**Kabsch ON for all arms — do NOT run the Kabsch on/off 2x2.** It is the published baseline
(turning it off means the Hungarian arm is no longer Klein et al.); it is the harder test (the
effect must show up on top of rotation alignment); and under the two-axis framing Kabsch is a
coupling-side alignment while Sinkhorn is a target-side variance reduction, so overlap is
unlikely a priori. State the held-fixed factor explicitly in the paper.

Core factorial:

| | Hard target | Sinkhorn target | MCMC target |
|---|---|---|---|
| **No coupling** | floor | isolates target effect | isolates target effect |
| **Hungarian coupling** | Klein et al. baseline | **proposed** | **proposed** |

Plus the contraction control. Three seeds throughout.

Expected interaction, worth pre-stating as a prediction: the soft target is a *local* average over
permutations near the one that built x_t. A good coupling puts that neighbourhood somewhere
sensible, so Hungarian + soft should beat none + soft by more than the Hungarian effect alone. The
competing prediction is that better couplings make the posterior *more* peaked, shrinking the
soft-vs-hard gap. Which wins is the empirical question and is why the factorial is worth running.

### Key plots

1. Convergence curves (metric vs epoch, all arms) — the headline
2. Quality vs NFE {1,2,5,10,20,50,100} — gap should WIDEN at low NFE if the mechanism
   (straighter paths) is real
3. Estimator bias at small n (<= 12): exact permanent marginals via Ryser vs Sinkhorn vs MCMC
4. Straightness diagnostics. `trajectory-straightness` (path length / chord) is reasonable but
   **gameable by contraction**, so always pair it with **coupling transport cost**
   `E||x1 - x0^pi||^2` — a training-time property of the pairing involving no model at all.
   Together they distinguish "the coupling changed" from "the learned paths straightened".
5. Target-variance diagnostic. NOT permutation flip rate — flip rate only applies to
   hard-permutation arms, and under a soft target there is no pi to flip. The version that works
   across every arm: fix x1 and t, resample x0 many times, measure the **variance of the resulting
   regression target**. Directly measures the gradient noise the method claims to reduce.
6. Performance stratified by molecule size — gap should grow with n
7. Wall-clock per step — Sinkhorn O(n^2) batched should beat Hungarian O(n^3) sequential
8. Entropy of P vs t — shows how much blending is actually happening

### Metrics

Not just validity/uniqueness (these saturate and hide differences).

**Primary energy metric is Delta-E_relax under GFN2-xTB**, reported as **both median and mean** —
the distribution is heavily right-skewed (Nikitin et al.'s SemlaFlow numbers: median 32.3, mean
91.0 +/- 21.7) and the two capture different failure modes: median = typical geometry quality,
mean = rate of catastrophic failures.

**MMFF is demoted, not deleted** — keep only as a coarse outlier filter. `energy`, `strain`,
`opt-rmsd`, `opt-energy-validity`, `energy-per-atom`, `strain-per-atom` are all MMFF-derived and
all correlated. Nikitin et al. (arXiv 2505.00169) show reference GEOM-Drugs conformers score mean
Delta-E_relax ~16 kcal/mol under MMFF but ~0 under GFN2-xTB, because the dataset was *built* by
GFN2-xTB optimisation. MMFF's 15-20 kcal/mol error is larger than the effect being measured.

Add the three **geometry-deviation** metrics from the same paper: mean bond length, bond angle and
torsion differences between each generated molecule and *its own* GFN2-xTB-optimised counterpart.
More interpretable than distribution-level Wasserstein and the field's emerging standard. Keep the
Wasserstein distances against reference distributions as secondary.

**Valency table:** the hand-patch to `"C": {0: 4}`, `"N": {0: 3}` in `metrics.py` is the right
direction but a partial approximation — neither 1 nor 1.5 is a universally correct aromatic bond
order. Use the reference implementation from `github.com/isayevlab/geom-drugs-3dgen-evaluation`
(table indexed by `(element, n_aromatic_bonds, formal_charge, valency)` giving the allowed
*non-aromatic* bond order), or retrain on a kekulised dataset. Worth doing properly: this codebase
is named in that paper as affected.

**PoseBusters:** keep but secondary. It is already built and the `energy_ratio` trap is already
debugged, so it costs nothing, but it is not a replacement for RDKit validity (different question:
3D geometric plausibility vs 2D graph chemistry) and the GFN2-xTB geometry deviations are strictly
more sensitive.

Scale: 5000 molecules for the headline table, ~1000 per point for the NFE sweep.

### Statistics

**Do not use Wilcoxon signed-rank.** `compare_arms.py` matches arms on the **size sequence** drawn
from the test set. That is a legitimate blocking variable and worth keeping, but it is *not* true
pairing — arm A's molecule i and arm B's molecule i are different molecules that happen to have the
same atom count, and Wilcoxon assumes genuinely paired observations.

- Report the size-matched comparison as a variance-reduced descriptive.
- Use **unpaired** tests for the formal claim: Mann-Whitney U, or bootstrap CIs on the difference
  of medians.
- Always report **effect sizes**. p ~ 1e-195 at n=2000 reflects a systematic difference of unknown
  size, not importance.
- With 3 seeds, formal testing across seeds has almost no power. Show all three seed-level values
  per arm instead. If Sinkhorn's worst seed beats Hungarian's best seed, that is persuasive
  without a p-value.

### Falsification criterion (pre-committed)

If hard ~= Sinkhorn ~= MCMC targets across the factorial, the hypothesis is dead. The honest paper
is then "soft targets do not help for molecular flow matching, and here is the estimator-quality
analysis showing why" — still worth publishing given how widely equivariant-OT coupling is used.
