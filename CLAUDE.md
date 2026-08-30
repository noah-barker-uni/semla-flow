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
axes. The original implementation conflated them; both are now separate flags, `--coupling` and
`--target`.

| Axis | What it decides | Legal values | Why |
|---|---|---|---|
| **Coupling** | which (x0, x1) pair, hence which x_t | genuine permutations ONLY | must preserve marginals |
| **Target** | given x_t, what to regress toward | hard / soft-averaged / sampled | blending is correct here |

**Sinkhorn and MCMC belong on the target axis, not the coupling axis.**
Coupling axis holds `{none, hungarian}`; target axis holds `{hard, sinkhorn, mcmc}`.

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

The model uses an x1-prediction parameterisation (`fm.py:_loss`). The Bayes-optimal x1-predictor is
`E[x1 | x_t]` — a posterior mean, i.e. a blend. Squared-error regression converges to it anyway.

- Blending the **input** (x_t) corrupts the distribution the model trains on. Pure damage.
- Blending the **target** moves the target toward what the loss is already estimating.
  Variance reduction, not corruption.

The "averaged atom positions land inside bonds / non-physical molecule" worry is **wrong** — do
not reintroduce it. The temperature falls with t, sharpening P onto a single permutation, so the
endpoint target is a real molecule.

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
w(pi') ∝ exp( -||x_t - t * pi'(x1)||^2 / (2 * var) ),    var = Var(x_t | x1, t)
```

1. **The cost is computed from x_t**, not from raw x0/x1: `cost[i,j] = ||x_t[i] - t*x1[j]||^2`,
   rows indexing x_t slots and columns x1 candidates. A t-independent cost with t bolted on as a
   temperature is not the posterior.
2. **The temperature is the conditional path's own variance**, so there is no free hyperparameter.
   `x_t = (1-t)x0 + t*x1 + sigma*z` gives `var = (1-t)^2 + sigma^2`, and since sinkhorn is
   parameterised by `P ~ exp(-cost/eps)`, **`eps = 2 * var`**. The corrections doc's prose says
   `(1-t)^2`, dropping both the factor of 2 its own formula implies and the interpolant's
   `coord_noise_std`. Consequence: with the default sigma=0.2, eps floors at 0.08 rather than
   reaching 0, so blending vanishes at t->1 not because eps->0 but because the cost gap between
   distinct atoms (~1 in scaled units) swamps it. **`--coord_noise_std_dev` now enters the
   schedule and must be held fixed across all arms.**
3. **The target becomes `P @ x1`** — coordinates, plus atomics / bonds / charges as soft labels.

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

## Defects found in the original implementation, and how each was settled

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
2. **[FIXED] The target axis now exists.** `--target {hard,sinkhorn,mcmc}` (default `hard`),
   independent of `--coupling`, computed in the loss via `fm.py:permutation_target`. Both soft
   estimators share one cost, one eps schedule and one application path — `mcmc` returns a hard
   permutation which `functional.py:permutation_to_plan` turns into a permutation matrix so it
   goes through the same `fm.py:apply_plan` as Sinkhorn, rather than a parallel gather that could
   drift. `molrepr.py:soft_permute` stays as the single-molecule reference and `tests/functional.py`
   pins `apply_plan` against it so their conventions cannot drift either.
3. **[FIXED] The cost is computed from x_t.** `cost[b,i,j] = ||x_t[b,i] - t*x1[b,j]||^2`, rows
   indexing x_t slots and columns x1 candidates. Not the old t-independent
   `inter_distances(to_coords, from_coords)` with t bolted on as a temperature.
4. **[SETTLED] The eps clamp is gone and is not coming back.** The schedule is now
   `eps = 2((1-t)^2 + sigma^2)`, the conditional path's own variance (sigma = `coord_noise_std`),
   with the factor of 2 the corrections doc's prose dropped. `TARGET_MIN_EPS = 1e-5` in `fm.py` is
   **not** the old clamp: a floored eps only keeps `sinkhorn_batched`'s `eps > 0` check legal, and
   the resulting plan is then *discarded* in favour of the identity — which is the exact eps -> 0
   limit, since x_t -> x1 as t -> 1 makes the identity the argmin. So nothing is ever blended at a
   floored temperature, which is precisely what the old clamp did wrong. Its value comes from
   float32 conditioning, not taste, and `train-target-eps` / `train-target-hard-fallback-frac` are
   logged so a binding floor would be visible immediately rather than inferred years later.
5. **[FIXED, worth knowing] Sinkhorn's rows were the unconverged marginal.** The solver loop ends
   on the column update, so `sum_i P_ij = 1` holds identically at any iteration count while the
   ROW sums — the ones a posterior-mean target reads — are only approximate. At the 100 iterations
   a training step affords, the worst row is off by ~1e-2, which would have shrunk that atom's
   coordinate target toward the origin by 1% — the contraction artifact, reintroduced on the target
   side through a convergence bug. `functional.py:plan_from_sinkhorn` renormalises rows and logs
   the pre-normalisation deviation as `train-target-row-sum-dev`.
6. **[FIXED] The bond diagonal needed the single-index marginal.** `(P B P^T)_ii` treats sigma(i)
   as two independent draws, but on the diagonal both indices are the *same* draw, so the exact
   answer is `sum_j P_ij B_jj` — the self-bond row transforms like any node feature. `_bond_loss`
   does train on the diagonal (`adj_from_node_mask(..., self_connect=True)`), so this matters.

### All existing Sinkhorn/MCMC numbers are uninterpretable

Reported so far: Sinkhorn straighter (1.11-1.14 vs Hungarian 1.16-1.21, p~1e-195), lower X-hat_1
movement, **but worse** Wasserstein bond-length (0.0138 vs 0.0102) and bond-angle (0.859 vs 0.623).

Noise contraction predicts exactly this pair of symptoms. Shrinking the prior toward its mean makes
trajectories more radial — in the limit P = J/n every path is a perfectly straight line from the
origin, i.e. maximum straightness and a useless model. **Straightness is a metric that contraction
can game.** The geometry regression then follows from the train/test mismatch. The posterior-mean
hypothesis, by contrast, predicts straighter paths *and* equal-or-better geometry.

**Keep the runs, relabel them exploratory, do not delete, do not cite as evidence.**

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
- CPU allocation `b35bs.3.isambard` / `b35bs.macs3.isambard` is unused, and GFN2-xTB evaluation is
  CPU work — but **do not run xtb on Isambard**, see below. It runs on the Mac instead. At
  `--n_workers 8` a 5000-molecule QM9 set takes ~2-3 min for well-formed geometries.

### xtb on aarch64: works on the Mac, silently WRONG on Isambard

Getting a binary at all is non-obvious, because the usual routes do not exist: grimme-lab publishes
**no aarch64 binary** (only linux-x86_64 and windows), and there is **no linux-aarch64 wheel for
`tblite`** either. The PyPI `xtb` package fails to build. conda-forge builds xtb 6.7.1 for both
`linux-aarch64` and `osx-arm64`, and extracting the `.conda` by hand does NOT work — the binary
needs its transitive conda deps (`libmctc-lib` and friends), so let the solver do it.

**The osx-arm64 build is correct; every linux-aarch64 build tried is not.** The conda-forge
`linux-aarch64` xtb returns *positive* single-point energies (water: +0.0785 Eh, against
-0.0074518 Eh on the Mac), and two other aarch64 builds segfault. It does not error — it prints
plausible-looking output with wrong numbers, which is the worst possible failure mode for a
primary metric. So `semlaflow/util/xtb.py:validate_xtb_binary()` runs a two-part guard before any
evaluation: relax a distorted geometry (must *lower* the energy) and re-relax its own minimum (must
give ~0). A good run announces itself, e.g. `xtb 6.7.1 validated: distorted geometry relaxed by
26.5451 kcal/mol, its own minimum by 0.000000`. **Never disable this guard.**

Consequence: generation happens on Isambard, xTB scoring happens locally on the Mac against the
SDFs pulled down, and the resulting `xtb/*.json` are pushed back up. `SEMLAFLOW_XTB_BINARY` (or
`--xtb_binary`) points at the prefix so it need not be merged into any training venv:

```bash
conda create -p <prefix>/xtb -c conda-forge xtb      # osx-arm64
export SEMLAFLOW_XTB_BINARY=<prefix>/xtb/bin/xtb
```
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
- Data: `/projects/b5bg/barkern.b5bg/data/qm9/smol` and `/projects/b5bg/barkern.b5bg/data/geom-drugs/smol`
- Job outputs: `/projects/b5bg/barkern.b5bg/runs`
- Checkpoints: `checkpoints_v2/<run_name>/` for the corrected runs. The pre-corrections runs are
  preserved at `checkpoints_v1_exploratory/` — keep, do not cite (see "All existing Sinkhorn/MCMC
  numbers are uninterpretable").
- Results: `/projects/b5bg/barkern.b5bg/results/` — `generated/` (`predict.py`), `analysis/`
  (`analyse_generated.py`), `xtb/` (`xtb_eval.py`, produced on the **Mac**, see below),
  `diagnostics/` (`target_variance.py`), and `summary.{json,md}` from `collect_results.py`.
  Mirrored locally at `output/results/`, which is gitignored — the summaries live on disk only.

### SSH: use the Clifton host alias, not a hostname

`ssh barkern.b5bg@ai.login.isambard.ac.uk` fails with `Permission denied (publickey)` and
`ai-p2.access.isambard.ac.uk` does not resolve from off-cluster at all. Neither is the way in.
`~/.ssh/config` includes a Clifton-managed `config_clifton` defining **`b5bg.aip2.isambard`**,
which sets the user, the short-lived certificate in `~/Library/Caches/clifton/`, and a ProxyJump
through the login node. So the whole invocation is:

```bash
ssh b5bg.aip2.isambard
rsync -av output/results/xtb/ b5bg.aip2.isambard:/projects/b5bg/barkern.b5bg/results/xtb/
```

The certificate expires, and an expired one fails with the same `Permission denied (publickey)` as
a wrong hostname — re-run the Clifton login rather than debugging the host. The other allocations
have their own aliases in the same file (`b35bs.3.isambard`, `b35bs.macs3.isambard`).

## Plan / status

Environment, both axes, the evaluation harness and the wandb instrumentation are built. The
pre-corrections QM9 runs (all four old coupling arms x 3 seeds) are preserved at
`checkpoints_v1_exploratory/` on Isambard and in the old `equinv-qm9` wandb project — exploratory
only, see "Defects found in the original implementation".

**The corrected runs are a clean slate.** `DEFAULT_RUN_SERIES = "v2"` in `train.py` sends them to
the `equinv-<dataset>-v2` wandb project and `checkpoints_v2/`, so nothing mixes with the old
experiments. Override with `--wandb_project` / `--checkpoint_dir`. Run names are
`<coupling>_<target>_seed<seed>`, the arm is also in `wandb.config` as separate fields, and runs
are grouped by arm so "mean +/- band across seeds" is two clicks rather than a CSV export.

### QM9 results, 1 seed, 5000 molecules per arm — the soft target FAILS

`results/summary.md`. Baseline `none_hard`; ΔE_relax kcal/mol, lengths Angstrom, angles degrees.

| arm | validity | dE_relax med/mean | bond len dev | bond angle dev | xtb RMSD |
|---|---|---|---|---|---|
| hungarian_hard | 0.9924 | **9.29 / 13.5** | **0.0234** | **1.59** | **0.094** |
| none_hard | 0.9938 | 10.01 / 15.2 | 0.0246 | 1.71 | 0.109 |
| none_mcmc | 0.9672 | 14.28 / 53.3 | 0.0294 | 1.86 | 0.139 |
| none_sinkhorn | **0.0000** | — | — | — | — |
| none_sinkhorn-hardcat | 0.9496 | **768.7 / 2803** | **0.252** | **16.4** | **0.808** |

Read this before designing anything further:

- **Both channels of the soft target fail independently.** Soft *categoricals* make the bond graph
  unconstructible (0% valid). Soft *coordinates* leave a plausible graph wrapped around garbage
  geometry — that is the `-hardcat` arm, which is 95% valid but 75x worse in dE_relax, Cliff's
  delta 0.995 against the baseline. **Validity did not catch this and neither did Rg** (2.291 vs
  2.310, `collapsed_fraction` 0). It took GFN2-xTB. Do not judge an arm on validity alone.
- **Not geometric collapse.** The earlier prediction — soft target ⇒ every atom at the centroid ⇒
  visibly collapsed molecules — is **wrong** as stated for the generated samples. `none_sinkhorn`'s
  outputs have Rg 2.08 A and nearest-neighbour distance 1.126 A, both normal. The damage is in the
  bond graph and in fine geometry, not in the overall size of the molecule.
- **The training target does collapse, and that is arithmetic rather than a bug.** From
  `diagnostics/target_collapse.json`, `||P x1|| / ||x1||` under `sinkhorn` is 0.043 at t=0.05
  (17.3 effective atoms averaged), 0.622 at t=0.5, 0.994 at t=0.95. Near-uniform P makes `P @ x1`
  the row-mean of x1 = the centroid = the ORIGIN for zero-COM molecules, so the model is regressed
  toward zero over the whole low-t half of every trajectory.
- **The conceptual gap this exposes.** The Bayes-optimal x1-target is `E[x1 | x_t]` under the joint
  (x0, x1) law. Sinkhorn instead averages over permutations of the *specific* x1 in the batch,
  conditioning on information the model does not have. That moves the target away from x1 without
  moving it toward `E[x1 | x_t]` — it moves it toward that one molecule's centroid. Bregman
  linearity licenses substituting a conditional mean, but only over the correct conditioning set.
  This is the thing to think hardest about before spending more GPU hours on the target axis.
- **Against the pre-committed falsification criterion** ("if hard ~= sinkhorn ~= mcmc, the
  hypothesis is dead") the outcome is stronger: soft is dramatically worse, MCMC mildly worse
  (delta 0.13-0.31). The only thing that helps is the Hungarian **coupling** — hungarian_hard beats
  none_hard on all five geometry metrics, consistently but with small effect (delta -0.07 to -0.11,
  dE_relax -0.7, CI [-1.09, -0.41]).
- **One seed**, so these compare trained models rather than methods. The hardcat and sinkhorn
  effects are far too large for seed noise; the hungarian-vs-none effect is not, and needs the
  other two seeds before it is claimed.

Next, in priority order:

1. [done] **Sinkhorn and MCMC removed from the coupling dispatch.**
       `COUPLING_TYPES = ["none","hungarian"]`; solvers and `soft_permute` kept.
2. [done] **Target axis added** — `--target {hard,sinkhorn,mcmc}`, cost from x_t,
       `eps = 2((1-t)^2 + sigma^2)`, entropy-of-P-vs-t logged per epoch. MCMC uses `init_perm` =
       identity (x_t was built from x1 in index order, so the identity IS the permutation that
       generated it — no scipy call, no GPU->CPU sync in the loss) and `to_coords` = x_t coords for
       the knn proposal. **Still owed: a GPU step-time profile on Isambard before any real run.**
       On CPU at B=64/N=25 the target costs ~13 ms (sinkhorn, 100 it) vs ~9 ms (mcmc, 100 it), but
       MCMC is kernel-launch-bound so the ordering may invert on GPU.
3. [done] **NFE sweep harness** — `python -m semlaflow.nfe_sweep`, {1,2,5,10,20,50,100}.
       Prints (a) metric vs NFE per arm, (b) the *difference* vs NFE against a zero line so "the
       gap widens" is the literal shape of the curve, (c) "NFE to reach threshold tau" as one
       number. **Still owed: run it** — this is the highest-value missing *experiment*, since the
       mechanism predicts the gap widens at low NFE and everything so far ran at a fixed 100.
4. [done] **GFN2-xTB pipeline** — `semlaflow/util/xtb.py` + `python -m semlaflow.xtb_eval`,
       reading the SDF that `predict.py` already writes. Protocol matches
       `isayevlab/geom-drugs-3dgen-evaluation` (`xtb <xyz> --opt --charge q --gfn 2`, dE_relax
       read from the "total energy gain" line and negated) so numbers are comparable to their
       table. **Still owed: run it on Isambard** — see the install note below.
5. [done] **Ryser estimator-bias comparison at n <= 12** — `python -m semlaflow.estimator_bias`.
       Exact permanent marginals (`semlaflow/util/permanent.py`) vs Sinkhorn vs MCMC on cost
       matrices built the way the loss builds them. On synthetic molecules the mean-field claim
       already holds at every t (sinkhorn entropy > exact, eg. 0.496 vs 0.319 at t=0.5) and the
       hard argmin is far the crudest estimate at low t (dev 0.187 vs sinkhorn 0.003 at t=0.1).
       **Re-run on real QM9 cost matrices before citing.**
6. [done] **Target-variance diagnostic** — `python -m semlaflow.target_variance`. Note the
       quantity the corrections doc names is degenerate here: under `--target hard` the target is
       x1 regardless of x0, so its variance is exactly 0. The comparable quantity, and the one the
       gradient actually sees, is the variance of `target - x_t`; both are printed.
7. [ ] **GEOM-Drugs** — everything so far is QM9. Nothing to build: `--dataset geom-drugs` already
       works throughout. The valency table is GEOM-Drugs-derived, so stability numbers become
       exactly comparable to Nikitin et al. once this runs.

## Submitting runs on Isambard

Batch scripts and job outputs both live in `/projects/b5bg/barkern.b5bg/runs/` (not in the repo —
they hardcode cluster paths). Everything from the pre-corrections work is archived in `runs_old/`.

```
runs/
  smoke_qm9.sh              Gate. 1 epoch x 3 targets on QM9; exits non-zero if any arm fails.
  smoke_geom-drugs.sh       Timing probe -- steps_per_sec and memory, not a finished epoch.
  qm9/                      One self-contained script per arm, named <coupling>_<target>.sh
    none_hard.sh  none_sinkhorn.sh  none_mcmc.sh
    hungarian_hard.sh  hungarian_sinkhorn.sh  hungarian_mcmc.sh
  geom-drugs/               Same six names; different parameters (see below)
```

One script per arm with the settings baked in, rather than one parametrised script driven by
`--export`: the arm then exists on disk rather than only in the submit command, and
`sbatch runs/qm9/hungarian_sinkhorn.sh` is the whole invocation. Per-dataset subdirectories so the
filename can be just coupling and target.

```bash
cd /projects/b5bg/barkern.b5bg/runs
SMOKE=$(sbatch --parsable smoke_qm9.sh)                       # gate first
sbatch --dependency=afterok:$SMOKE qm9/hungarian_sinkhorn.sh  # arm starts only if the gate passes
```

**Parameters differ by dataset and this is easy to get wrong.** Upstream's README: *"The default
arguments in the training script are for GEOM Drugs. To train on QM9 we use a `bond_loss_weight`
of 0.5, 2000 `warm_up_steps` and usually 300 `epochs`."* So QM9 must override three arguments, and
**GEOM-Drugs must override none of them** — 200 epochs, `bond_loss_weight` 1.0, `warm_up_steps`
10000. Copying the QM9 flags onto a GEOM run is wrong in all three.

Three traps, all hit already:

- **A smoke job that loops over configs must track failures and `exit $FAILED`.** A bare loop
  exits with the status of the *last* command, so `afterok` fires even when an arm crashed — the
  gate is then worthless.
- **Deploy before you submit.** A fix that is written, tested and pushed is still not running
  until the cluster checkout is pulled. Six queued runs died on an `AttributeError` that had
  already been fixed three commits earlier. Every job script now echoes
  `RUNNING COMMIT: <sha> <subject>` as its first line and `train.py` records `git_revision` in
  `wandb.config`, so a stale checkout announces itself.
- **The login node reaps long-running and memory-heavy processes.** Detached `nohup`/`setsid`
  downloads were killed three times with an empty log, and loading GEOM's 915 MB `train.smol`
  was OOM-killed (exit 137). Long downloads need a kept-alive foreground SSH session with a
  resumable tool; anything that loads the full GEOM train split needs a compute node.

Held fixed across every arm, deliberately: `--kabsch_align`, `--optimal_transport none`,
`--coord_noise_std_dev 0.2` (it now enters the soft-target temperature schedule, so varying it
would make arms incomparable), and `--seed`.

**Wall time: QM9 300 epochs takes ~6-7 hours** (measured on earlier runs; `sacct` retention is too
short to recover this, so it is recorded here). Request ~10h for hard/sinkhorn and ~14h for mcmc,
whose per-step cost is the one genuine unknown. Do NOT pad to 24h "to be safe": Slurm bills
elapsed rather than requested time, so over-requesting costs nothing directly, but the backfill
scheduler can only slot a job into a gap at least as long as its requested limit — on a cluster
sitting at ~1275/1320 nodes allocated, a 24h request skips every 7-24h gap and can delay the
start substantially.

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

Three seeds throughout.

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
   Built: `interpolate.py:coupling_transport_cost`, logged every step as `train-transport-cost`,
   and reported per arm by `compare_arms.py` (which rebuilds the interpolant from the
   checkpoint's own recorded training hparams, so no model is loaded).
5. Target-variance diagnostic. NOT permutation flip rate — flip rate only applies to
   hard-permutation arms, and under a soft target there is no pi to flip. Built as
   `semlaflow/target_variance.py`; see the plan note above for why it reports the variance of
   `target - x_t` alongside the variance of the target itself.
6. Performance stratified by molecule size — gap should grow with n
7. Wall-clock per step — Sinkhorn O(n^2) batched should beat Hungarian O(n^3) sequential
8. Entropy of P vs t — shows how much blending is actually happening

### wandb instrumentation

Spec: `docs/wandb_corrections.md`. **Nothing logged during training is a reported result** — every
paper number comes from the post-hoc pipeline (GFN2-xTB, corrected valency table) on the CPU
allocation. Training logging exists to catch a broken implementation in the first few hundred
steps rather than after six hours.

Everything in the method-diagnostics group is logged per step AND accumulated into 5 t-bins
flushed each epoch (`<key>_t0`..`_t4`), because almost all of it is t-dependent and a scalar mean
over a batch spanning every t hides the shape, which is the only thing these are for.

| Key | Fires when | What a bad value means |
|---|---|---|
| `coupling/transport_cost` | every arm | straightness's un-gameable companion |
| `coupling/frac_reassigned` | every arm | 0 on a hungarian arm ⇒ the coupling is not wired up |
| `sinkhorn/plan_entropy` | sinkhorn | not falling with t ⇒ the eps schedule is not taking effect |
| `sinkhorn/sum_p_squared` | sinkhorn | comparable to the 0.896 contraction figure in the old brief |
| `sinkhorn/target_delta` | sinkhorn | ~0 ⇒ **the soft target is a no-op**; catch this on day one |
| `sinkhorn/marginal_violation` | sinkhorn | large ⇒ P not doubly stochastic, everything downstream suspect |
| `mcmc/acceptance_rate` | mcmc | ~0 ⇒ knn proposal or temperature is wrong |
| `mcmc/hamming_from_init` | mcmc | ~0 ⇒ the chain never moved; report it as a hard arm |
| `train/{loss,coord_loss,type_loss,bond_loss,charge_loss}` | every arm | per-modality, since a change could help coords and hurt types |
| `train/grad_norm`, `train/lr`, `perf/steps_per_sec` | every arm | steps_per_sec supports the O(n^2) vs O(n^3) wall-clock claim, and cannot be retrofitted without rerunning |

**Validation is deliberately trimmed** to `validity`, `fc-validity`, `energy-validity`,
`strain-per-atom` plus atom/molecule stability. Dropped: `uniqueness` and `novelty` (measured noise
around 0.99 across every arm and seed — they cannot discriminate on QM9, and dropping novelty also
removes a full train-set SMILES pass at startup), and three of the four MMFF metrics (`energy`,
`strain`, `opt-rmsd`, `opt-energy-validity` are correlated, not four independent confirmations, and
the optimising ones cost a forcefield minimisation per molecule per validation). **No GFN2-xTB
during training** — seconds per molecule, it belongs post-hoc on `b35bs`.

**Nothing influences training.** There is no `EarlyStopping` and no `ReduceLROnPlateau` anywhere.
`ModelCheckpoint(monitor="val-validity")` only decides which extra file is kept as `best.ckpt`;
every arm trains a fixed number of epochs and evaluation uses `last.ckpt`, so checkpoint selection
is identical across arms. Preserve that — arms stopping at different epochs would silently break
the comparison.

### Metrics

Not just validity/uniqueness (these saturate and hide differences).

**Primary energy metric is Delta-E_relax under GFN2-xTB** (built: `semlaflow/xtb_eval.py`),
reported as **both median and mean** —
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

**Valency table: [done]** `semlaflow/util/valency.py` + the vendored reference table in
`semlaflow/util/valency_tables/`. Aromatic bonds are *counted* and non-aromatic orders summed
separately, keyed `(element, charge) -> {(n_aromatic, non_aromatic_valence)}`, so no aromatic bond
order is ever assigned and the 1-vs-1.5 question does not arise. This fixed a real bug: the
training-time path in `fm.py` summed bond orders and truncated, so a **pyrrole-type N-H
(1.5+1.5+1 = 4.0 -> 4, vs neutral N allowing only 3) was scored unstable** — that is indole,
imidazole and pyrazole, so any previously reported `val-molecule-stability` was penalised for them.
Caveat: the table is GEOM-Drugs-derived and QM9's NH+/NH2+ are not in it; the removed hand-patch
allowed them. `load_valency_table(allow_legacy_qm9=True)` restores them explicitly, but numbers
computed that way are no longer the reference protocol's.

**PoseBusters:** keep but secondary (marked as such in `paired_eval.py`). It is already built and the `energy_ratio` trap is already
debugged, so it costs nothing, but it is not a replacement for RDKit validity (different question:
3D geometric plausibility vs 2D graph chemistry) and the GFN2-xTB geometry deviations are strictly
more sensitive.

Scale: 5000 molecules for the headline table, ~1000 per point for the NFE sweep.

### Statistics

Built as `semlaflow/util/stats.py`; `compare_arms.py` uses it and no longer imports `wilcoxon`.

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
