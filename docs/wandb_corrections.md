# wandb logging — what to track

Companion to `corrections-brief.md`. Covers what should be logged during training after the
coupling/target refactor. Existing runs are being discarded, so this is a clean-slate spec rather
than a migration.

**Scope note.** Nothing here is a reported result. All paper numbers come from the post-hoc
evaluation pipeline (GFN2-xTB, corrected valency table) on the CPU allocation. This document is
about *instrumentation* — catching a broken implementation within the first few hundred steps
rather than after six hours of training.

---

## 0. First, verify validation metrics do not affect training

Validation metrics are computed under `no_grad` and never enter the loss. They can still influence
training indirectly through three mechanisms:

- `ModelCheckpoint(monitor=...)` — selects which epoch is saved as "best"
- `EarlyStopping` — halts on a plateau
- `ReduceLROnPlateau` — cuts LR based on a monitored metric

Check with:

```bash
grep -n "monitor=\|EarlyStopping\|ReduceLROnPlateau" semlaflow/train.py
```

Current setup appears safe: all arms train a fixed 300 epochs and evaluation uses
`checkpoints/<run>/last.ckpt`, not `best.ckpt`, so checkpoint selection is deterministic and
identical across arms. **Preserve that property.** If any of the three mechanisms are active,
either disable them for the ablation or monitor *validation loss* (arm-appropriate, unaffected by
the valency-table issue) rather than a downstream chemistry metric. Different arms stopping at
different epochs would silently break the comparison.

---

## 1. Method diagnostics — the important part

These are cheap (the quantities are already computed) and they are what tell you whether the
mechanism is doing what the paper claims. None of them currently exist.

**Bucket everything in this section by `t`** — 5 bins is enough. Almost all the interesting
behaviour is t-dependent and a scalar mean over the batch hides it entirely.

### Coupling axis (log for every arm)

| Key | Definition | What it catches |
|---|---|---|
| `coupling/transport_cost` | `E‖x₁ − x₀^π‖²` | the un-gameable companion to straightness |
| `coupling/frac_reassigned` | fraction of atoms whose assignment ≠ identity | whether Hungarian is doing anything at all |

`transport_cost` matters because trajectory straightness can be inflated by anything that shrinks
the prior, whereas transport cost is a property of the *pairing* alone and involves no model.
Always read the two together.

### Sinkhorn target arm

| Key | Definition | What it catches |
|---|---|---|
| `sinkhorn/plan_entropy` | entropy of P, by t-bin | **most important.** Must fall toward 0 as t→1 |
| `sinkhorn/sum_p_squared` | `Σⱼ Pᵢⱼ²`, by t-bin | directly comparable to the contraction table in the old brief — confirms the fix landed |
| `sinkhorn/target_delta` | `‖P x₁ − π(x₁)‖` | **if ≈ 0 the method is a no-op.** Catch this on day one |
| `sinkhorn/marginal_violation` | max row/col sum deviation from 1 after `n_iters` | Sinkhorn not converged ⇒ P not doubly stochastic ⇒ everything downstream suspect |

### MCMC target arm

| Key | Definition | What it catches |
|---|---|---|
| `mcmc/acceptance_rate` | fraction of proposals accepted | near-zero ⇒ k-NN proposal or temperature is wrong |
| `mcmc/hamming_from_init` | distance between final and Hungarian-init permutation | zero ⇒ `K` too small; you have reimplemented Hungarian |

---

## 2. Training health

| Key | Note |
|---|---|
| `train/loss` | plus **per-modality components**: coords, atomics, bonds, charges |
| `train/grad_norm` | catches instability early |
| `train/lr` | |
| `perf/steps_per_sec` | needed for the wall-clock claim — see below |

Per-modality loss split matters because a coupling/target change could plausibly help coordinates
and hurt atom types, and an aggregate loss would hide that.

`perf/steps_per_sec` supports the paper claim that Sinkhorn (O(n²), batched matmuls) is cheaper
than Hungarian (O(n³), sequential scipy calls). Retrofitting it later means rerunning, so log it
from the start.

---

## 3. Validation quality — trim hard

### Keep

- `validity`, `connected-validity`
- `atom-stability`, `molecule-stability` (corrected valency table)
- **one** MMFF metric — `strain-per-atom` is the most informative — as a coarse "is geometry sane"
  signal
- `trajectory-straightness` (trajectory recording already exists), always read alongside
  `coupling/transport_cost`

### Drop

- **`uniqueness` and `novelty`** — measured noise around 0.99 across all arms and seeds in the
  previous runs. They cannot discriminate between reasonable models on QM9.
- **Three of the four MMFF metrics.** `energy`, `strain`, `opt-rmsd`, `opt-energy-validity` are
  correlated, not four independent confirmations. `opt-rmsd` and `opt-energy-validity` additionally
  require an optimisation step, costing validation time for no extra signal.

### Do not add

**No GFN2-xTB during training.** It is seconds per molecule and would dominate training time. It
belongs in post-hoc evaluation on the `b35bs` CPU allocation.

Rationale for the trimming: validity saturates near 0.97 for every arm and uniqueness is pure
noise, which is exactly what Nikitin et al. (arXiv 2505.00169) warn about. These metrics answer
"is this run learning / has it diverged", which is all they are for here. The discriminating
metrics are ΔE_relax and geometry deviations, computed post-hoc.

---

## 4. Run organisation

This matters more than it sounds with 12+ runs.

Put the arm definition into `wandb.config` as **separate fields**, not baked into the run name:

```python
wandb.config.update({
    "coupling": args.coupling,          # none | hungarian
    "target": args.target,              # hard | sinkhorn | mcmc
    "seed": args.seed,
    "kabsch": args.kabsch_align,
    "sinkhorn_n_iters": args.sinkhorn_n_iters,
    "mcmc_n_iters": args.mcmc_n_iters,
})
```

and set:

```python
group = f"{args.coupling}_{args.target}"
tags  = [f"seed{seed_index}"]
```

With `group` set, "mean ± band per arm across seeds" is a two-click operation in the wandb UI, and
the convergence-curve figure for the paper comes almost directly out of it. Without it, you are
exporting CSVs and re-plotting by hand.

Keep the existing per-run checkpoint directories (`checkpoints/<run_name>/`) — without them
parallel arms overwrite each other.

---

## 5. Why this set

The diagnostics in §1 are chosen so that a broken implementation announces itself immediately:

- `sinkhorn/target_delta` flat at ~0 → the soft target is not actually soft
- `sinkhorn/plan_entropy` not falling with t → the ε = (1−t)² schedule is not taking effect (check
  whether `COUPLING_MIN_EPS` is binding — see `corrections-brief.md` §3.4)
- `sinkhorn/sum_p_squared` appearing anywhere it shouldn't → P is being applied to the prior again
- `mcmc/acceptance_rate` ~0 → proposal restriction or temperature is wrong
- `mcmc/hamming_from_init` ~0 → K too small

Any one of these would have surfaced the previous implementation's problems in minutes rather than
after a full set of runs.
