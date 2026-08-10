# How validation and evaluation work in this fork

Handoff note for a fresh session. Covers what is measured, where it lives, how to run it, and
the traps. Assumes the research context in `CLAUDE.md` (soft Sinkhorn/MCMC permutation coupling
vs hard Hungarian coupling, on QM9).

There are **three separate evaluation surfaces**. They share the underlying metric
implementations but are invoked differently and answer different questions.

| Surface | Question | Output |
|---|---|---|
| Training-time validation (`train.py`) | Is this run learning? | wandb curves, per epoch |
| `evaluate.py` | How good is one checkpoint, in absolute terms? | mean ± std table |
| `compare_arms.py` | Is arm A different from arm B, and is it significant? | Wilcoxon table + size strata |

---

## 1. Training-time validation

Runs automatically during `python -m semlaflow.train`, every `--val_check_epochs` (default 10).
Metrics are constructed in `semlaflow/models/fm.py:466-499` and logged to wandb.

Generative metrics: `validity`, `fc-validity` (fully-connected), `uniqueness`, `novelty`,
`energy-validity`, `opt-energy-validity`, `energy`, `energy-per-atom`, `strain`,
`strain-per-atom`, `opt-rmsd`, `atom-stability`, `molecule-stability`.

Loss components: `train-loss`, `coord-loss`, `type-loss`, `bond-loss`, `charge-loss`.

`MolecularAccuracy` / `MolecularPairRMSD` also exist in that file but only activate on the
distillation path, which this project does not use.

---

## 2. `evaluate.py` — absolute quality of one checkpoint

```bash
python -m semlaflow.evaluate \
  --ckpt_path checkpoints/<run_name>/last.ckpt \
  --data_path <data>/qm9/smol --dataset qm9
```

Defaults: `--n_molecules 10000`, `--n_replicates 3`, `--integration_steps 100`,
`--ode_sampling_strategy log`, `--seed 12345`, `--dataset_split test`.

Generates N molecules × 3 replicates and reports mean ± std over replicates. Metric set is built
in `scriptutil.py:135-148` — the same 13 as training-time validation, minus the loss components:

| Metric | Meaning |
|---|---|
| `validity` | RDKit-sanitisable |
| `connected-validity` | valid **and** a single connected component |
| `uniqueness` | fraction of distinct canonical SMILES |
| `novelty` | fraction absent from the training set |
| `energy-validity` | MMFF force field can be set up |
| `opt-energy-validity` | same, after MMFF geometry optimisation |
| `energy` / `energy-per-atom` | MMFF single-point energy |
| `strain` / `strain-per-atom` | E(generated) − E(MMFF-relaxed) |
| `opt-rmsd` | RMSD between generated and MMFF-relaxed geometry |
| `atom-stability` | fraction of atoms with an allowed valence |
| `molecule-stability` | fraction of molecules where *all* atoms are stable |

`evaluate.py` does **not** touch PoseBusters, trajectory straightness, or X̂₁ movement. Those
exist only in `compare_arms.py`.

---

## 3. `compare_arms.py` — paired statistical comparison of two checkpoints

```bash
python -m semlaflow.compare_arms \
  --ckpt_path_a checkpoints/hungarian/last.ckpt \
  --ckpt_path_b checkpoints/sinkhorn_seed1_envfix/last.ckpt \
  --label_a hungarian_seed1 --label_b sinkhorn_seed1 \
  --data_path <data>/qm9/smol --dataset qm9
```

Defaults: `--n_molecules 2000`, `--n_reference_molecules 2000`, `--integration_steps 100`,
`--seed 12345`. Takes roughly 9 minutes on one GH200.

### How the pairing works — read this before trusting any p-value

Unconditional generation has no natural correspondence between "arm A's molecule *i*" and
"arm B's molecule *i*". Each is freely sampled from noise by an independently trained model,
not a reconstruction of a shared reference. So a paired test needs a deliberately constructed
pairing, and this is it:

> Both arms are generated with the same `--seed`, `--n_molecules` and `--dataset_split`.
> `GeometricDataset.sample()` draws its molecule-size sequence via `np.random.choice`, which
> `L.seed_everything` covers. Both arms therefore condition on the **same sequence of real
> test-set molecule sizes**. Slot *i* in each arm is sized to match the same real molecule.

`_generate_arm()` reseeds and rebuilds the datamodule per arm rather than reusing one across
arms, specifically to keep this guarantee. **If you change the seed handling, the pairing
silently breaks and every p-value becomes meaningless.**

Caveat already in the code comments: size-stratified buckets use each *generated* molecule's own
atom count, not the target size that seeded it. Intended size would be the more principled
stratification variable if someone threads it through later.

### Per-molecule metrics (each gets a Wilcoxon signed-rank test + size-stratified means)

From `_collect_per_molecule` (`compare_arms.py:105`) plus two added in `compare()`:

| Metric | Notes |
|---|---|
| `validity` | connected validity, per molecule |
| `posebusters-valid` | 11 physical-plausibility checks, all must pass — see below |
| `atom-stable-frac` | per-molecule fraction of valence-stable atoms |
| `mol-stable` | all atoms stable |
| `energy-per-atom` | MMFF |
| `strain-energy-per-atom` | MMFF |
| `opt-rmsd` | MMFF |
| `trajectory-straightness` | realised ODE path length ÷ straight-line chord |
| `x1-movement` | mean per-atom, per-step change in the model's own X̂₁ prediction |

Plus three **distribution-level** Wasserstein distances against reference test-set geometry —
`bond-length`, `bond-angle`, `torsion-angle`. These are not per-molecule, so they get no
Wilcoxon test, only a side-by-side number.

### The two trajectory metrics are not the same thing

Easy to conflate; they measure different objects.

- **`trajectory-straightness`** — the *realised* integrated ODE path. Treats a molecule's whole
  atom configuration at a step as one point in R^(3n), sums step-to-step displacement norms, and
  divides by the first-to-last chord. 1.0 is perfectly straight. Recorded from `curr["coords"]`
  *after* each integrator step.
- **`x1-movement`** — the *model's own prediction* of the endpoint (X̂₁), recorded from `coords`
  *before* the integrator step. Measures how much the denoiser's belief about the final structure
  changes between successive calls, i.e. how quickly it settles. Averaged per atom then per step,
  so it does not scale with molecule size. From FlowMol3 (arXiv 2508.12629).

Both need `record_trajectory=True`, which makes `MolecularCFM._generate` return
`predicted["trajectory"]` and `predicted["x1_trajectory"]`, both `[B, T, N, 3]`. Note
`x1_trajectory` has one *fewer* step than `trajectory` — there is no X̂₁ prediction before the
first model call. `record_trajectory` raises `NotImplementedError` on distilled models.

---

## Traps and things already fixed

**PoseBusters `energy_ratio` is deliberately excluded.** Do not add it back without reading
`_POSEBUSTERS_EXCLUDED_FUNCTIONS` in `semlaflow/util/paired_eval.py`. It scores a conformer
against an ETKDG reference ensemble and asserts *every* requested conformer embeds
(`energy_ratio.py:158`). Generated QM9 molecules are often strained fused cages RDKit cannot
embed, so it returned NaN for ~8% of them; it also scored water and methane as outright
failures. It was measuring RDKit's embedding success, not molecule quality. Its intended signal
is already covered better by the MMFF energy/strain metrics. It was also the dominant cost —
`mol.yml` lists the module twice and each copy embeds 50 conformers per molecule.

The 11 surviving checks: `mol_pred_loaded`, `sanitization`, `inchi_convertible`,
`all_atoms_connected`, `no_radicals`, `bond_lengths`, `bond_angles`, `internal_steric_clash`,
`aromatic_ring_flatness`, `non-aromatic_ring_non-flatness`, `double_bond_flatness`.

**A NaN check is a skip, not a failure.** `per_molecule_posebusters` uses `.all(skipna=True)`;
a row that is entirely NaN returns `None`. Counting "could not evaluate" as "implausible" is
what produced the original bad numbers.

**PoseBusters re-enables RDKit's logger globally.** `posebusters/modules/sanity.py:22-24`
disables `rdApp.*` for its own check then calls `EnableLog`, undoing
`scriptutil.disable_lib_stdout()` and flooding the rest of the job log with RDKit warnings.
`per_molecule_posebusters` restores the suppression afterwards. If logs get noisy again, look
here first.

**PoseBusters does not mutate its inputs** (verified: atom counts, conformer coordinates and
SMILES all identical before/after). This matters because `posebusters-valid` is computed *before*
energy/strain/opt-rmsd in the same dict, so a mutating call would have corrupted them.

**All energy metrics are MMFF-derived.** `energy`, `strain`, `opt-rmsd` and
`opt-energy-validity` are correlated, not independent evidence. Do not count them as four
separate confirmations.

**Multiple comparisons.** A full sweep is ~9 metrics × 6 arm-pairs × 3 seeds. Expect several
p < 0.05 hits by chance alone. Treat a result as real only if it holds across essentially all
comparisons and all three seeds — which, so far, only `trajectory-straightness` and
`x1-movement` do.

**Per-molecule functions are positionally aligned.** Every function in `paired_eval.py` returns
exactly one entry per input molecule, with `None` where a value is undefined. Callers depend on
this to match molecules across arms by slot index. Preserve it in anything new.

**The valency table is intentionally non-standard.** `ALLOWED_VALENCIES` in
`semlaflow/util/metrics.py` was corrected per Nikitin et al. ("GEOM-Drugs Revisited",
arXiv 2505.00169), which names SemlaFlow as having an aromatic-bond-order rounding bug that let
neutral carbon sit at valence 3 and neutral nitrogen at valence 2. Consequence: stability numbers
will **not** exactly reproduce SemlaFlow's published figures. Say so explicitly wherever those
numbers are reported — it is a deliberate correction, not a reproduction failure.

---

## Not built yet

Both are in the `CLAUDE.md` plan and absent from every run so far:

1. **NFE sweep** — quality vs integration steps ∈ {1,2,5,10,20,50,100}. This is the most valuable
   missing piece. Straightness is currently the strongest signal, and the mechanism predicts the
   gap should *widen* at low NFE. Everything runs at a fixed 100 steps, so that prediction is
   untested. `--integration_steps` already exists on both `evaluate.py` and `compare_arms.py`, so
   this is a sweep harness, not new metric code.
2. **GFN2-xTB energies** — needs a new dependency, unverified on aarch64/GH200. Nikitin et al.
   argue MMFF stops discriminating on GEOM-Drugs; less pressing for QM9.

Also unbuilt from the plan: estimator-bias comparison at small n (exact permanent marginals via
Ryser vs Sinkhorn vs MCMC), and the target-variance diagnostic (how often the hard permutation
flips between visits to the same molecule).

---

## File map

| Path | Contents |
|---|---|
| `semlaflow/util/metrics.py` | torchmetrics aggregate `Metric` classes, `ALLOWED_VALENCIES` |
| `semlaflow/util/paired_eval.py` | per-molecule versions, PoseBusters, straightness, X̂₁ movement |
| `semlaflow/util/geometry_metrics.py` | bond length/angle/torsion extraction + Wasserstein |
| `semlaflow/compare_arms.py` | the paired-comparison script |
| `semlaflow/evaluate.py` | absolute single-checkpoint evaluation |
| `semlaflow/scriptutil.py` | `init_metrics`, `generate_molecules`, `disable_lib_stdout` |
| `semlaflow/models/fm.py` | `_generate` (trajectory recording), training-time metrics |
| `tests/paired_eval.py` | tests for all per-molecule metrics |

`paired_eval.py` deliberately mirrors the aggregate `Metric` classes 1:1 and calls the *same*
underlying RDKit helpers, so per-molecule numbers cannot drift from aggregate ones. Tests assert
that the mean of the per-molecule list equals the aggregate. Keep that property.

Run tests with `python -m unittest -v tests/*.py` (CPU only, ~86 tests).

---

## Experimental arms

Four coupling methods, set by `--coupling {none,hungarian,sinkhorn,mcmc}` on `train.py`, crossed
with `--kabsch_align`. `none` is the floor, `hungarian` is the baseline to beat (Klein et al.),
`sinkhorn` and `mcmc` are the proposals. Entropic temperature is scheduled as
`eps = max((1-t)^2, COUPLING_MIN_EPS)` with `COUPLING_MIN_EPS = 1e-3`, shared by both soft
methods so they are directly comparable. No free hyperparameter.

Three seeds: 12345, 23456, 34567, passed as `--seed` with a matching `--run_name`. Checkpoints
land in `checkpoints/<run_name>/`.

One naming wrinkle: the seed-1 sinkhorn checkpoint directory is `sinkhorn_seed1_envfix` (it was
the run that confirmed the aarch64 OpenMP fix), while seeds 2 and 3 are plain `sinkhorn_seed2` /
`sinkhorn_seed3`. Seed-1 hungarian and mcmc are just `hungarian` and `mcmc`, with `_seed2` /
`_seed3` suffixes for the others. Label them uniformly at the `compare_arms.py` call site with
`--label_a` / `--label_b`.
