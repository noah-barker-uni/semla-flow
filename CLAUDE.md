# Project context

## Research goal

Testing whether **soft (entropic) permutation coupling beats hard Hungarian coupling** in
flow matching for de novo 3D molecule generation.

Background: Klein et al. (Equivariant Flow Matching, NeurIPS 2023) align noise to data using
the Hungarian algorithm (hard argmin permutation) plus Kabsch (rotation). This is now standard
and is what SemlaFlow uses.

The hypothesis: the hard argmin is a **biased point estimate** of a posterior mean over
permutations. The exact soft version requires computing permanent marginals of the cost
matrix, which is #P-hard. Two tractable estimators:

1. **Sinkhorn** — deterministic, O(n^2) batched matmuls, differentiable, mean-field biased
   (systematically more diffuse than truth). This is the primary method.
2. **MCMC over permutations** — Metropolis with transposition proposals, asymptotically
   unbiased, finite-chain biased. Secondary/comparison method.

Temperature is set by the conditional path's own variance: `eps = (1-t)^2`. So the coupling
is diffuse early in the trajectory (many permutations plausible) and sharpens onto the
Hungarian answer as t -> 1. No free hyperparameter.

Structural precedent: "Efficient Molecular Conformer Generation with SO(3)-Averaged Flow
Matching and Reflow" (Cao et al., ICML 2025) showed soft rotation averaging beats hard Kabsch.
This project is the permutation analogue. Their case had a closed form (matrix Fisher on a
compact Lie group); the permutation case does not, hence the estimators above.

## Important correctness constraints

- **Must be de novo generation, not conformer generation.** Alignment over a group G is only
  valid if BOTH p_0 and p_1 are G-invariant. For conformer generation the graph pins atom
  identities, so p_1 is NOT S_n-invariant and full-permutation alignment is invalid (only
  Aut(G) is legitimate, which is small enough to enumerate exactly). In de novo generation
  coordinates+types+bonds are permuted jointly, so p_1 IS S_n-invariant.
- **Prior must be i.i.d. Gaussian per node.** A harmonic prior correlates noise along bonds,
  breaks exchangeability, and invalidates the alignment.
- **Permutation must be applied jointly** to coordinates, atom types, bond types, and charges.
- Under soft coupling, discrete targets become soft labels (P @ onehot). This is legitimate
  (cross-entropy is the Bregman divergence with Phi = u log u - u; target enters linearly),
  but should be stated explicitly and ablated.

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

- Account: `brics.b5bg` (NOT `b5bg`)
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

1. [done] Isambard environment built (venv, torch, rdkit, lightning etc.)
2. [done] Local Mac mirror environment built (Python 3.11 venv)
3. [ ] Verify GPU visibility via Isambard batch job; run `tests/` on both Mac and Isambard
4. [ ] `pip install -r extra_requirements.txt` (needed for QM9 data-prep notebook)
5. [ ] Download QM9 `smol` folder + `qm9.ckpt` from the authors' Google Drive (Isambard)
6. [ ] Run `evaluate` on their checkpoint — validates the eval pipeline reproduces published numbers
7. [ ] **Gate:** train QM9 from scratch with default Hungarian coupling, reproduce published
       numbers. Do not proceed until this passes — otherwise later deltas are just setup bugs.
8. [ ] Implement Sinkhorn coupling on Mac (branch `sinkhorn-coupling`).
       Find insertion point: `grep -rn "linear_sum_assignment" semlaflow/`
       Add flag `--coupling {none,random,hungarian,sinkhorn}` + eps schedule, keeping the
       existing path as default. Unit-test Sinkhorn against scipy's linear_sum_assignment:
       as eps -> 0 the soft plan must converge to the hard assignment; rows/cols sum to 1;
       must be numerically stable in log-space for small eps.
9. [ ] Push to Isambard, run short batch-job sanity check, then full ablation; then scale to
       GEOM Drugs

## Experimental design

Fix architecture, data, optimizer, seeds, sampler. Vary ONLY the coupling:

| Arm | Role |
|---|---|
| none | floor |
| random permutation | negative control — must be worse |
| Hungarian | the baseline to beat (Klein et al.) |
| Sinkhorn, sweep over eps | proposed |
| MCMC-sampled | proposed (unbiased-in-limit) |

Cross with Kabsch on/off as a 2x2 — controls for whether the gain duplicates what rotation
alignment already does. Hold SemlaFlow's "scale OT" (size handling) fixed across all arms;
it is orthogonal to the claim.

3 seeds minimum. Paired per-molecule Wilcoxon signed-rank tests (test set is shared across arms).

### Key plots

1. Convergence curves (metric vs epoch, all arms) — the headline
2. Quality vs NFE {1,2,5,10,20,50,100} — gap should WIDEN at low NFE if the mechanism
   (straighter paths) is real
3. Estimator bias at small n (<= 12): exact permanent marginals via Ryser vs Sinkhorn vs MCMC
4. Straightness diagnostics: coupling transport cost, inference trajectory curvature
5. Target-variance diagnostic: how often does the hard permutation flip between visits to the
   same molecule? (Hungarian's argmin is discontinuous in x_0; Sinkhorn's P is smooth. This
   instability argument is specific to permutations and does not apply to Kabsch, whose argmin
   is generically smooth.)
6. Performance stratified by molecule size — gap should grow with n
7. Wall-clock per step — Sinkhorn O(n^2) batched should beat Hungarian O(n^3) sequential

### Metrics

Not just validity/uniqueness (these saturate and hide differences). Use Wasserstein distances
between generated and reference bond-length / bond-angle / torsion distributions, plus
energy-based metrics (xTB or MMFF single-point, RMSD after relaxation). See the benchmark
papers the SemlaFlow README points to: Nikitin et al. "GEOM-Drugs Revisited" (2505.00169) and
Buttenschoen et al. (2505.00518), plus PoseBusters.

### Falsification criterion (pre-committed)

If Hungarian ~= Sinkhorn ~= MCMC across all eps, the hypothesis is dead. The honest paper is
then "soft coupling does not help for molecular flow matching, and here is the estimator-quality
analysis showing why" — still worth publishing given how widely equivariant-OT coupling is used.
