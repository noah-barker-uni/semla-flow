# Which soft-coupling variant is actually implemented?

Check performed against the code on branch `sinkhorn-coupling`. Nothing was changed.

**Question.** There are two genuinely different ways to make equivariant-OT coupling soft:

- **(a) Soft coupling** — replace the hard permutation π with Sinkhorn's plan P and blend one of
  the endpoints. Blending the noise, `x0_blend = P x0`, breaks the prior: each blended coordinate
  has variance `Σⱼ Pᵢⱼ² < 1`, so training noise is a contracted Gaussian while inference samples
  plain N(0, I). Blending the data, `x1_blend = P x1`, gives non-physical target molecules.
- **(b) Soft target** — sample `x_t` using a genuine single permutation, so both marginals are
  exactly preserved, and average only the regression target over the posterior `p(π' | x_t)`,
  with weights `∝ exp(−‖x_t − t·π'(x1)‖² / 2(1−t)²)`. The blending vanishes as t→1 because
  ε = (1−t)² → 0 sharpens P onto a single permutation.

**Answer: the implementation is (a), not (b).** It blends the noise endpoint.

---

## The chain, with line references

### 1. The plan is applied to the noise, x₀

`interpolate.py:274` dispatches the coupling on `from_mols` (the prior):

```python
elif self.coupling == "sinkhorn":
    from_mols = self._sinkhorn_couple(from_mols, to_mols, times_list)
```

and inside `_sinkhorn_couple` (`interpolate.py:384-387`):

```python
result = [
    trunc_from_mols[b].soft_permute(plan[b, :n_b, :n_b])
    for b, n_b in enumerate(seq_lengths.tolist())
]
```

`soft_permute` is literally a matmul against the coordinates (`molrepr.py:578`):

```python
coords = P @ self.coords
```

`to_mols` is never passed through `soft_permute`.

### 2. x_t is then built from the blended noise

`interpolate.py:462`:

```python
coords_mean = (from_mol.coords * (1 - t)) + (to_mol.coords * t)
```

Since `from_mol` is the soft-permuted prior, this is `x_t = (1−t)·(P x₀) + t·x₁`. The blend sits
**inside** x_t — exactly what (b) is designed to avoid.

### 3. The target is raw, unpermuted x₁

`fm.py:745` — an x₁-prediction parameterisation, not velocity:

```python
coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
```

where `coords = data["coords"]` and `data` is the untouched `to_mols`.

### 4. The cost matrix is t-independent

`interpolate.py:377`, computed from raw x₀ / x₁ coordinates:

```python
cost = smolF.inter_distances(to_coords, from_coords, sqrd=True)
```

`t` enters **only** as temperature (`interpolate.py:380`):

```python
eps = torch.clamp((1.0 - torch.tensor(times)) ** 2, min=COUPLING_MIN_EPS)
```

In (b), the posterior weights are computed from `x_t` itself. This implementation never uses
`x_t` when forming P.

---

## The variance contraction is real and measurable

Running the actual `sinkhorn_batched` on the actual cost construction, B=512, N=12:

| t | ε = (1−t)² | Σⱼ Pᵢⱼ² | effective atoms mixed per slot |
|---|---|---|---|
| 0.00 | 1.0000 | 0.380 | ~2.6 |
| 0.25 | 0.5625 | 0.503 | ~2.0 |
| 0.50 | 0.2500 | 0.671 | ~1.5 |
| 0.75 | 0.0625 | 0.870 | ~1.15 |
| 0.90 | 0.0100 | 0.877 | ~1.14 |
| 0.99 | 0.0010 | 0.896 | ~1.12 |

`Σⱼ Pᵢⱼ² = 1` if and only if P is a true permutation matrix. It never is. At t=0 each blended
slot is averaging roughly 2.6 atoms.

`Σⱼ Pᵢⱼ²` is the right diagnostic here because it is a property of the plan alone. Absolute
variance numbers from the same run are less trustworthy, since the cost-dependent plan correlates
the blended noise with x₁ and the synthetic x₁ scale used was arbitrary.

---

## One mitigation that is present

The criticism of (a) was specifically "(a) **with a fixed ε** doesn't have the saving grace".
ε here is not fixed — `eps = (1−t)²` is the same schedule (b) uses. P therefore does sharpen
toward a single permutation as t→1, and the blending does vanish at the endpoint. The model is
not being trained to emit blended molecules at t=1.

The contraction is concentrated at low t, where the table above shows it is severe.

---

## Two consequences worth noting

**MCMC and Sinkhorn are structurally different arms.** `interpolate.py:439` applies a *hard*
single sampled permutation:

```python
trunc_from_mols[b].permute(final_perm[b, :n_b].tolist())
```

So the MCMC arm preserves the prior exactly and has no contraction at all. The two "soft" arms
are therefore not two estimators of the same object: MCMC is a sampled genuine permutation,
Sinkhorn is a mean-field blend. That asymmetry is a confound in any Sinkhorn-vs-MCMC comparison,
and it may bear on why MCMC's trajectory straightness tracks Hungarian's while Sinkhorn's sits
distinctly lower.

**`CLAUDE.md` is inaccurate on this point.** It states:

> Under soft coupling, discrete targets become soft labels (P @ onehot).

The code does the opposite. It soft-permutes the *noise*'s atomics and bond types; the target's
one-hots are untouched, and the loss regresses against raw data labels. That line should be
corrected, since it would mislead anyone reasoning about the loss.
