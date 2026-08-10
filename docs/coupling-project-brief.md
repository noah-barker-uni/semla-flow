# Soft vs hard permutation coupling in SemlaFlow — implementation and evaluation brief

Single entry point for a fresh session. Covers the research claim, how all four coupling arms are
implemented, how evaluation works, what has actually been run, and the traps that have already
cost real debugging time.

Two focused deep-dives exist alongside this and are absorbed into it — `docs/evaluation.md`
(evaluation surfaces in more detail) and `docs/sinkhorn-coupling-variant.md` (the (a)/(b)
analysis). This document is the one to read first.

Repo: fork of `rssrwn/semla-flow` → `noah-barker-uni/semla-flow`, branch `sinkhorn-coupling`.

---

## 1. The claim being tested

Klein et al. (Equivariant Flow Matching, NeurIPS 2023) align noise to data with the Hungarian
algorithm — a hard argmin over permutations — plus Kabsch for rotation. This is now standard, and
is what stock SemlaFlow does.

The hypothesis: **the hard argmin is a biased point estimate of a posterior mean over
permutations.** The exact soft version needs permanent marginals of the cost matrix, which is
#P-hard. Two tractable estimators are implemented:

1. **Sinkhorn** — deterministic, O(n²) batched matmuls, differentiable, mean-field biased
   (systematically more diffuse than truth). Primary method.
2. **MCMC over permutations** — Metropolis with transposition proposals, asymptotically unbiased,
   finite-chain biased. Secondary/comparison method.

Temperature comes from the conditional path's own variance, `eps = (1-t)²`, so the coupling is
diffuse early and sharpens onto the Hungarian answer as t→1. **No free hyperparameter** — this is
a selling point of the design, so resist adding an eps knob.

Structural precedent: Cao et al., "Efficient Molecular Conformer Generation with SO(3)-Averaged
Flow Matching and Reflow" (ICML 2025), showed soft rotation averaging beats hard Kabsch. This is
the permutation analogue. Their case had a closed form on a compact Lie group; permutations do not.

**Pre-committed falsification criterion.** If Hungarian ≈ Sinkhorn ≈ MCMC across all eps, the
hypothesis is dead, and the honest paper is "soft coupling does not help for molecular flow
matching, and here is the estimator-quality analysis showing why". That is still publishable given
how widely equivariant-OT coupling is used. Do not quietly move the goalposts.

### Correctness constraints that must not be violated

- **Must be de novo generation, not conformer generation.** Alignment over a group G is valid only
  if *both* p₀ and p₁ are G-invariant. In conformer generation the graph pins atom identities, so
  p₁ is not Sₙ-invariant and full-permutation alignment is invalid. In de novo generation
  coordinates + types + bonds are permuted jointly, so p₁ *is* Sₙ-invariant.
- **The prior must be i.i.d. Gaussian per node.** A harmonic prior correlates noise along bonds,
  breaks exchangeability, and invalidates the alignment.
- **The permutation must be applied jointly** to coordinates, atom types, bond types and charges.

---

## 2. How coupling is wired

Everything routes through `GeometricInterpolant` in `semlaflow/data/interpolate.py`. The relevant
constant and enum, at the top of that file:

```python
COUPLING_MIN_EPS = 1e-3
COUPLING_TYPES = ["none", "hungarian", "sinkhorn", "mcmc"]
```

### CLI surface (`train.py`)

| Flag | Default | Notes |
|---|---|---|
| `--coupling` | `hungarian` | one of `none`/`hungarian`/`sinkhorn`/`mcmc` |
| `--kabsch_align` | `True` | `BooleanOptionalAction`, so `--no-kabsch_align` disables |
| `--sinkhorn_n_iters` | 100 | |
| `--mcmc_n_iters` | 100 | |
| `--mcmc_proposal` | `knn` | or `uniform` |
| `--mcmc_knn_k` | 8 | |
| `--seed` | — | drives everything, including the eval pairing |
| `--run_name` | — | names the wandb run and `checkpoints/<run_name>/` |

Kabsch is a **separate, orthogonal flag** from the coupling method, deliberately, so the 2×2
(coupling × rotation-alignment) ablation in the experimental design is possible.

### Dispatch (`interpolate.py:253-292`)

```python
times = self.time_dist.sample((batch_size,))   # sampled BEFORE matching -- sinkhorn needs t
if self.coupling == "mcmc":
    from_mols = self._mcmc_couple(from_mols, to_mols, times_list)
elif self.coupling == "sinkhorn":
    from_mols = self._sinkhorn_couple(from_mols, to_mols, times_list)
elif self.batch_ot:
    ...
else:
    from_mols = [self._match_mols(...) for ...]     # none / hungarian
interp_mols = [self._interpolate_mol(from_mol, to_mol, t) for ...]
```

Note the ordering: times are sampled *before* coupling, because both soft methods need `t` to set
their temperature. `none` and `hungarian` go through the per-molecule `_match_mols` path;
`sinkhorn` and `mcmc` each get one batched pass over the whole minibatch.

**Guard:** neither `sinkhorn` nor `mcmc` may be combined with `batch_ot=True` — both raise in
`__init__`. `batch_ot` needs a per-pair coupling callable evaluated over all B² candidate pairings,
which is incompatible with a single batched pass over the current pairing.

---

## 3. Sinkhorn implementation

### The solver — `semlaflow/util/functional.py:460`

`sinkhorn(cost_matrix, eps, n_iters)` solves for the doubly-stochastic P minimising
`<P, cost> − eps·H(P)` by log-space Sinkhorn–Knopp:

```python
for _ in range(n_iters):
    f = -eps * torch.logsumexp((g.unsqueeze(0) - cost_matrix) / eps, dim=1)
    g = -eps * torch.logsumexp((f.unsqueeze(1) - cost_matrix) / eps, dim=0)
log_plan = (f.unsqueeze(1) + g.unsqueeze(0) - cost_matrix) / eps
return torch.exp(log_plan)
```

Two things to know:

- **Log-space is not optional.** At the small eps reached near t→1, `exp(-cost/eps)` alone
  under/overflows.
- **Marginals are all-ones, not the probability simplex.** Row and column sums target 1 (a
  permutation-like matrix), which is *not* the usual OT convention. Do not "fix" this.

As eps→0 P converges to the hard assignment from `linear_sum_assignment` (assuming a unique
optimum); as eps→∞ it converges to uniform 1/n. Both limits are unit-tested against scipy.

### The batched solver — `functional.py:505`

`sinkhorn_batched(cost, node_mask, eps, n_iters)` runs all molecules at once. `eps` is per-batch-
element, shape `[B]`. Molecules differ in size, so it pads to the batch max and masks any
(row, col) pair touching padding by adding a large finite cost:

```python
SINKHORN_MASK_COST = 1e6
masked_cost = cost.masked_fill(~valid, SINKHORN_MASK_COST)
```

Large enough that `exp(-1e6/eps)` underflows to 0 even at the largest eps used, but **finite**, so
logsumexp never sees inf/nan. Values touching padding in the returned plan are meaningless and must
be sliced away (`plan[b, :n_b, :n_b]`), not used directly.

Batching was not just a speed optimisation — see §7, the aarch64 segfault.

### Wiring — `interpolate.py:353`

```python
cost = smolF.inter_distances(to_coords, from_coords, sqrd=True)     # raw x1 vs raw x0
eps = torch.clamp((1.0 - torch.tensor(times)) ** 2, min=COUPLING_MIN_EPS)
plan = smolF.sinkhorn_batched(cost, node_mask, eps, n_iters=self.sinkhorn_n_iters)
result = [trunc_from_mols[b].soft_permute(plan[b, :n_b, :n_b]) for b, n_b in ...]
```

### Applying the plan — `molrepr.py:548`, `GeometricMol.soft_permute`

Mixes each new atom slot as a convex combination weighted by `P[i, :]`:

```python
coords = P @ self.coords
atomics = P @ self.atomics
adj = torch.einsum("ij,jkc,lk->ilc", P, raw_adj, P)      # bonds, both axes
```

Requires atomics and bond_types to already be distributions (not class indices) and bonds to be
fully dense — both hold for a freshly-sampled prior molecule, which is the only intended caller.
It validates all three and raises otherwise.

---

## 4. MCMC implementation

### The sampler — `functional.py:549`

`mcmc_permutation(cost, node_mask, eps, n_iters, init_perm, proposal, knn_k, to_coords)` targets
`p(perm) ∝ exp(−cost(perm)/eps)` by batched Metropolis with transposition (swap) proposals, on the
**same cost matrix and same eps schedule** as Sinkhorn — deliberately, so the two are directly
comparable.

Proposal modes:

- **`knn`** (default, k=8) — restrict swaps to spatially-nearby position pairs.
- **`uniform`** — any valid pair. Near-zero acceptance for large n, so `knn` is the default. If
  someone reports MCMC "doing nothing", check this flag first.

### Wiring — `interpolate.py:394`

Chains are **initialised at the Hungarian solution**, one scipy call per molecule (the same
one-time cost `coupling="hungarian"` already pays), then all molecules are refined together as
batched tensor ops:

```python
for b, n_b in enumerate(seq_lengths.tolist()):
    _, col_ind = linear_sum_assignment(cost[b, :n_b, :n_b].numpy())
    init_perm[b, :n_b] = torch.as_tensor(col_ind)
eps = torch.clamp((1.0 - torch.tensor(times)) ** 2, min=COUPLING_MIN_EPS)
final_perm = smolF.mcmc_permutation(cost, node_mask, eps, self.mcmc_n_iters, init_perm=init_perm, ...)
result = [trunc_from_mols[b].permute(final_perm[b, :n_b].tolist()) for b, n_b in ...]
```

**Note the last line: `.permute`, a hard reindex — not `soft_permute`.** This is the single most
important structural difference between the two arms. See §5.

---

## 5. The two soft arms are not the same kind of object

This is the most important thing to understand before interpreting any result.

There are two genuinely different ways to soften the coupling:

- **(a) Soft coupling** — replace hard π with plan P and blend an endpoint. Blending the noise,
  `x0_blend = P x0`, breaks the prior: each blended coordinate has variance `Σⱼ Pᵢⱼ² < 1`, so
  training noise is a contracted Gaussian while inference samples plain N(0, I). Blending the
  data, `P x1`, gives non-physical targets (averaged atom positions can land inside bonds).
- **(b) Soft target** — sample `x_t` under a genuine single permutation, so both marginals are
  exactly preserved, and average **only the regression target** over the posterior
  `p(π' | x_t)`, weights `∝ exp(−‖x_t − t·π'(x1)‖² / 2(1−t)²)`. Blending vanishes as t→1.

**The Sinkhorn arm implements (a), blending the noise endpoint.** Verified chain:

1. The plan is applied to `from_mols`, the prior (`interpolate.py:274`, `:385`), via
   `soft_permute` — `to_mols` is never soft-permuted.
2. `x_t` is then built from the blended noise (`interpolate.py:462`):
   `coords_mean = (from_mol.coords * (1 - t)) + (to_mol.coords * t)`, i.e.
   `x_t = (1−t)(P x₀) + t·x₁`. The blend sits **inside** x_t, which is what (b) avoids.
3. The target is raw unpermuted x₁ (`fm.py:745`, `F.mse_loss(pred_coords, data["coords"])`).
   The model uses an **x₁-prediction parameterisation**, not velocity; velocity is derived at
   inference (`fm.py:64`).
4. The cost is **t-independent**, computed from raw x₀/x₁ (`interpolate.py:377`). `t` enters only
   as temperature. In (b) the weights come from `x_t` itself; this implementation never uses `x_t`
   when forming P.

Measured plan diffuseness, running the real `sinkhorn_batched` on the real cost construction
(B=512, N=12). `Σⱼ Pᵢⱼ² = 1` iff P is a true permutation matrix:

| t | ε=(1−t)² | Σⱼ Pᵢⱼ² | effective atoms mixed per slot |
|---|---|---|---|
| 0.00 | 1.0000 | 0.380 | ~2.6 |
| 0.25 | 0.5625 | 0.503 | ~2.0 |
| 0.50 | 0.2500 | 0.671 | ~1.5 |
| 0.75 | 0.0625 | 0.870 | ~1.15 |
| 0.99 | 0.0010 | 0.896 | ~1.12 |

**Mitigation that is present:** the standard criticism of (a) assumes a *fixed* ε. Here
`eps = (1−t)²` is the same schedule (b) uses, so P does sharpen toward a single permutation as
t→1 and the blending does vanish at the endpoint — the model is never trained to emit blended
molecules at t=1. The contraction is concentrated at low t, where it is severe.

**The MCMC arm has none of this.** It applies a hard sampled permutation, so it preserves the
prior exactly. So: **MCMC is a sampled genuine permutation; Sinkhorn is a mean-field blend.** They
are not two estimators of the same object. This is a confound in any Sinkhorn-vs-MCMC comparison
and may bear on why MCMC's straightness tracks Hungarian's while Sinkhorn's sits distinctly lower.

**`CLAUDE.md` is wrong on one related point.** It says "Under soft coupling, discrete targets
become soft labels (P @ onehot)". The code does the opposite — it soft-permutes the *noise*'s
atomics and bond types; the target's one-hots are untouched and the loss regresses raw data
labels. Not yet corrected in `CLAUDE.md`.

---

## 6. Evaluation

Three surfaces. They share metric implementations but answer different questions.

| Surface | Question | Output |
|---|---|---|
| Training-time (`train.py`) | Is this run learning? | wandb curves, every `--val_check_epochs` (default 10) |
| `evaluate.py` | How good is one checkpoint, absolutely? | mean ± std over replicates |
| `compare_arms.py` | Is arm A different from arm B, significantly? | Wilcoxon + size strata |

### `evaluate.py`

```bash
python -m semlaflow.evaluate --ckpt_path checkpoints/<run>/last.ckpt \
  --data_path <data>/qm9/smol --dataset qm9
```

Defaults `--n_molecules 10000`, `--n_replicates 3`, `--integration_steps 100`,
`--ode_sampling_strategy log`, `--seed 12345`. Metrics built in `scriptutil.py:135-148`:

`validity`, `connected-validity`, `uniqueness`, `novelty`, `energy-validity`,
`opt-energy-validity`, `energy`, `energy-per-atom`, `strain`, `strain-per-atom`, `opt-rmsd`,
`atom-stability`, `molecule-stability`.

It does **not** touch PoseBusters, straightness or X̂₁ movement — those live only in
`compare_arms.py`.

### `compare_arms.py`

```bash
python -m semlaflow.compare_arms \
  --ckpt_path_a checkpoints/hungarian/last.ckpt \
  --ckpt_path_b checkpoints/sinkhorn_seed1_envfix/last.ckpt \
  --label_a hungarian_seed1 --label_b sinkhorn_seed1 \
  --data_path <data>/qm9/smol --dataset qm9
```

Defaults `--n_molecules 2000`, `--integration_steps 100`, `--seed 12345`. ~9 min on one GH200.

Per-molecule metrics, each getting a Wilcoxon signed-rank test plus size-stratified means:
`validity`, `posebusters-valid`, `atom-stable-frac`, `mol-stable`, `energy-per-atom`,
`strain-energy-per-atom`, `opt-rmsd`, `trajectory-straightness`, `x1-movement`.

Plus three distribution-level Wasserstein distances to reference test-set geometry (no Wilcoxon):
`bond-length`, `bond-angle`, `torsion-angle`.

### How the pairing works — read before trusting any p-value

Unconditional generation has no natural correspondence between arm A's molecule *i* and arm B's.
Each is freely sampled from noise by an independently trained model. The pairing is constructed:

> Both arms run with the same `--seed`, `--n_molecules` and `--dataset_split`.
> `GeometricDataset.sample()` draws its size sequence via `np.random.choice`, which
> `L.seed_everything` covers. Both arms therefore condition on the **same sequence of real
> test-set molecule sizes**; slot *i* in each is sized to match the same real molecule.

`_generate_arm()` reseeds and rebuilds the datamodule per arm specifically to preserve this.
**Change seed handling and every p-value silently becomes meaningless.**

Known caveat: size buckets use each *generated* molecule's own atom count, not the target size that
seeded it. Intended size would be more principled if someone threads it through.

### The two trajectory metrics differ

- **`trajectory-straightness`** — the *realised* integrated ODE path. Whole atom configuration at a
  step treated as one point in R^(3n); sum of step-to-step displacement norms ÷ first-to-last
  chord. 1.0 is perfectly straight. From `curr["coords"]` *after* each integrator step.
- **`x1-movement`** — the *model's own prediction* of the endpoint (X̂₁), from `coords` *before*
  the integrator step. How fast the denoiser's belief settles. Averaged per atom then per step, so
  it does not scale with molecule size. From FlowMol3 (arXiv 2508.12629).

Both need `record_trajectory=True`, giving `predicted["trajectory"]` and
`predicted["x1_trajectory"]`, both `[B, T, N, 3]`. `x1_trajectory` has one **fewer** step — there
is no X̂₁ prediction before the first model call. Raises `NotImplementedError` on distilled models.

---

## 7. Traps

**PoseBusters `energy_ratio` is deliberately excluded** (`_POSEBUSTERS_EXCLUDED_FUNCTIONS` in
`paired_eval.py`). It asserts every requested ETKDG conformer embeds (`energy_ratio.py:158`).
Generated QM9 molecules are often strained fused cages RDKit cannot embed, so it returned NaN for
~8% of them; it also scored water and methane as outright failures. It measured RDKit's embedding
success, not molecule quality. Its signal is already covered by the MMFF energy/strain metrics. It
was also the dominant cost — `mol.yml` lists it twice, 50 conformers each. Do not add it back.

**A NaN check is a skip, not a failure.** `per_molecule_posebusters` uses `.all(skipna=True)`; an
all-NaN row returns `None`. Counting "could not evaluate" as "implausible" produced the original
bad numbers.

**PoseBusters re-enables RDKit's logger globally.** `posebusters/modules/sanity.py:22-24` calls
`EnableLog("rdApp.*")`, undoing `scriptutil.disable_lib_stdout()` and flooding job logs.
`per_molecule_posebusters` restores suppression afterwards. If logs get noisy, look here first.

**PoseBusters does not mutate its inputs** (verified). Matters because `posebusters-valid` is
computed *before* energy/strain/opt-rmsd in the same dict.

**All energy metrics are MMFF-derived.** `energy`, `strain`, `opt-rmsd`, `opt-energy-validity` are
correlated, not four independent confirmations.

**Multiple comparisons.** A full sweep is ~9 metrics × 6 arm-pairs × 3 seeds. Expect several
p<0.05 by chance. Treat a result as real only if it holds across essentially all comparisons and
all three seeds.

**Per-molecule functions are positionally aligned.** Every function in `paired_eval.py` returns one
entry per input molecule, `None` where undefined. Callers match across arms by slot index. Preserve
this in anything new.

**The valency table is intentionally non-standard.** `ALLOWED_VALENCIES` in `metrics.py` was
corrected per Nikitin et al. ("GEOM-Drugs Revisited", arXiv 2505.00169), which names SemlaFlow as
having an aromatic-bond-order rounding bug allowing neutral C at valence 3 and neutral N at
valence 2. Consequence: stability numbers will **not** exactly reproduce SemlaFlow's published
figures. Say so explicitly wherever reported — a deliberate correction, not a reproduction failure.

**The aarch64 OpenMP segfault (fixed, do not regress).** On GH200, torch CPU matmul routes through
oneDNN → ARM Compute Library GEMM → `GOMP_parallel`. libgomp is not fork-safe, so a forked
DataLoader worker building an OpenMP thread team inside `soft_permute`'s matmul/einsum segfaulted.
`torch.set_num_threads(1)` does **not** reach ACL, which sizes its pool from environment variables
at library load. The fix is `semlaflow/__init__.py`, which sets `OMP_NUM_THREADS` and friends to 1
**before torch is imported** — it works only because scripts run as `python -m semlaflow.<script>`,
so the package `__init__` executes first. Do not move or lazily-import that. `--num_workers 0`
exists as an independent fallback but is much slower (it timed out at 9h where the env fix
finished in ~6h).

---

## 8. Status

QM9, 300 epochs, `--optimal_transport none --kabsch_align`, seeds 12345 / 23456 / 34567.

All four arms × 3 seeds have trained. Checkpoints in `checkpoints/<run_name>/`:

| Arm | seed1 dir | seed2 | seed3 |
|---|---|---|---|
| none | `none_seed1` | `none_seed2` | `none_seed3` |
| hungarian | `hungarian` | `hungarian_seed2` | `hungarian_seed3` |
| mcmc | `mcmc` | `mcmc_seed2` | `mcmc_seed3` |
| sinkhorn | `sinkhorn_seed1_envfix` | `sinkhorn_seed2` | `sinkhorn_seed3` |

Naming is inconsistent for historical reasons (seed-1 sinkhorn was the run confirming the OpenMP
fix; `sinkhorn_seed1_noworkers` is the abandoned `--num_workers 0` twin — ignore it). Normalise
with `--label_a`/`--label_b` at the call site.

### Results so far — treat as provisional

- **Trajectory straightness is the strongest signal.** Sinkhorn ~1.11–1.14 vs Hungarian ~1.16–1.21
  across every size bucket (seed 1, p ~1e-195). X̂₁ movement 0.038–0.043 vs 0.050–0.054
  (p ~1e-231). MCMC is significantly *less* straight than Hungarian across all 3 seeds
  (p 1.7e-6 to 6e-17).
- **Standard quality metrics show no robust difference.** Validity, stability, energy all
  non-significant or inconsistent across comparisons.
- **Sinkhorn is worse on Wasserstein geometry** — bond-length 0.0138 vs 0.0102, bond-angle 0.859
  vs 0.623 — but better on torsion (24.9 vs 26.2).
- **`posebusters-valid` numbers from the first sinkhorn run are invalid** (the `energy_ratio` bug)
  and are being regenerated. Straightness, X̂₁ movement and Wasserstein from that same run *are*
  valid — PoseBusters does not mutate the molecules.

So the honest current reading: **straighter paths and a more settled X̂₁ prediction, with no
corresponding quality win, and a geometry regression.** Note §5 — some of the straightness
difference may be a mechanical consequence of Sinkhorn contracting the prior rather than evidence
for the posterior-mean hypothesis. That confound is unresolved and matters for the paper.

---

## 9. Not built

1. **NFE sweep** — quality vs steps ∈ {1,2,5,10,20,50,100}. The highest-value missing piece: the
   mechanism predicts the straightness gap should *widen* at low NFE, and everything currently runs
   at a fixed 100 steps, so the prediction is untested. `--integration_steps` already exists on
   both scripts, so this is a sweep harness, not new metric code.
2. **GFN2-xTB energies** — all energy metrics are MMFF. Nikitin et al. argue MMFF stops
   discriminating on GEOM-Drugs. Needs a new dependency, unverified on aarch64.
3. **Estimator-bias comparison at small n (≤12)** — exact permanent marginals via Ryser vs Sinkhorn
   vs MCMC. Directly measures the mean-field bias the whole argument rests on.
4. **Target-variance diagnostic** — how often the hard permutation flips between visits to the same
   molecule. Specific to permutations; does not apply to Kabsch, whose argmin is generically smooth.
5. **The Kabsch on/off 2×2** — the flag exists and works, but only `--kabsch_align` (on) has been run.
6. **GEOM-Drugs** — everything so far is QM9.

---

## 10. File map

| Path | Contents |
|---|---|
| `semlaflow/util/functional.py` | `sinkhorn`, `sinkhorn_batched`, `mcmc_permutation`, `inter_distances`, `adj_from_node_mask` |
| `semlaflow/data/interpolate.py` | `GeometricInterpolant`, `_sinkhorn_couple`, `_mcmc_couple`, `_match_mols`, `_kabsch_align` |
| `semlaflow/util/molrepr.py` | `GeometricMol.permute` / `.soft_permute` |
| `semlaflow/util/metrics.py` | aggregate torchmetrics classes, `ALLOWED_VALENCIES` |
| `semlaflow/util/paired_eval.py` | per-molecule metrics, PoseBusters, straightness, X̂₁ movement |
| `semlaflow/util/geometry_metrics.py` | bond/angle/torsion extraction + Wasserstein |
| `semlaflow/compare_arms.py` | paired comparison script |
| `semlaflow/evaluate.py` | absolute single-checkpoint evaluation |
| `semlaflow/models/fm.py` | `_generate` (trajectory recording), loss, training-time metrics |
| `semlaflow/__init__.py` | the OpenMP env-var fix — must run before torch import |
| `docs/evaluation.md`, `docs/sinkhorn-coupling-variant.md` | focused deep-dives |

Tests: 86 total, CPU-only, `python -m unittest -v tests/*.py`. Breakdown: `functional.py` 34,
`paired_eval.py` 25, `interpolate.py` 13, `molrepr.py` 7, `geometry_metrics.py` 7.

`paired_eval.py` mirrors the aggregate `Metric` classes 1:1 and calls the *same* underlying RDKit
helpers, so per-molecule numbers cannot drift from aggregates; tests assert the mean of the
per-molecule list equals the aggregate. Keep that property.

---

## 11. Environment

**Local Mac** — sanity-checking mirror only, not where numbers come from. Python 3.11 venv at
`~/Desktop/semla-flow/venv` (3.13 has no `scipy==1.11.4` wheel). Run unit tests and tiny CPU
forward passes here.

**Isambard-AI Phase 2** (GH200, linux-aarch64), `ssh b5bg.aip2.isambard`:

```bash
module load cray-python/3.11.7
source /projects/b5bg/barkern.b5bg/venvs/equinv/bin/activate
cd /projects/b5bg/barkern.b5bg/semla-flow
```

Paths: repo `/projects/b5bg/barkern.b5bg/semla-flow`, data `.../data/qm9/smol`, job scripts and
outputs `.../runs`. Slurm account `brics.b5bg` (not `b5bg`), partition `workq`. 1 node = 4 GH200,
so a single-GPU job is 0.25 NHR/hour. There is no budget limit configured in Slurm; usage shows via
`sreport`, allocation lives in the BriCS portal.

**Sync via git, never manual copying.** Dependencies are unusually light — no DGL, no
torch-geometric, no torch-scatter; Semla is pure PyTorch with dense attention, which is why ARM
works cleanly.

### Two workflow rules learned the hard way

- **Do not queue short smoke tests.** Queue latency, not runtime, is the bottleneck — a 30-minute
  job and a 9-hour job both come back tomorrow morning. Queue the full run.
- **Use `--dependency=afterok:<jobid>`** to chain evaluation behind training so it fires
  unattended. Jobs auto-cancel if the dependency fails. `scontrol top` is permission-denied here.
