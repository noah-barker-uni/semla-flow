"""Estimator bias at small n: exact permanent marginals vs Sinkhorn vs MCMC.

The project's argument has two assertions in it that nothing has ever checked:

  1. the hard Hungarian assignment is a biased point estimate of a posterior mean over permutations;
  2. Sinkhorn is a mean-field approximation to that posterior and is "systematically more diffuse
     than the truth".

For n <= 12 the truth is computable -- the exact marginals P(sigma(i) = j) follow from permanents,
and Ryser's formula gets those in CPU-seconds (see semlaflow/util/permanent.py). So this script
computes the truth on real cost matrices taken from the training path and measures how far each
estimator sits from it, instead of asserting the direction of the bias.

What to look for:

  entropy(sinkhorn) > entropy(exact)   confirms the mean-field diffuseness claim
  hard deviation >> sinkhorn deviation confirms the argmin is the cruder estimate
  mcmc deviation ~ sinkhorn deviation  says the two estimators bracket the same object

Costs come from the same construction the loss uses -- cost[i,j] = ||x_t[i] - t*x1[j]||^2 with
eps = 2((1-t)^2 + sigma^2) -- so the numbers describe the regime training actually runs in, not an
arbitrary random matrix.

    python -m semlaflow.estimator_bias --data_path <smol dir> --dataset qm9

With no --data_path it runs on synthetic molecules from the prior sampler, which is enough to
exercise the machinery but is NOT a substitute for real cost matrices.
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import semlaflow.scriptutil as util
import semlaflow.util.functional as smolF
import semlaflow.util.permanent as perm_util
from semlaflow.data.datamodules import geometric_batch_to_dict
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.util.molrepr import GeometricMolBatch

DEFAULT_TIMES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_N_MOLECULES = 32
DEFAULT_MAX_ATOMS = 12
DEFAULT_MCMC_CHAINS = 512
DEFAULT_MCMC_ITERS = 100
DEFAULT_SINKHORN_ITERS = 100
DEFAULT_SEED = 12345
_SPLIT_FILES = {"train": "train.smol", "val": "val.smol", "test": "test.smol"}


def build_cost_matrices(mols, t, coord_noise_std, coupling="hungarian", kabsch_align=True):
    """Cost matrices exactly as the loss builds them, one per molecule.

    Returns:
        tuple: (list of [n, n] float64 cost matrices, eps scalar for this t).
    """

    prior_sampler = GeometricNoiseSampler(
        mols[0].atomics.size(-1),
        mols[0].bond_types.size(-1),
        coord_noise="gaussian",
        type_noise="uniform-sample",
        bond_noise="uniform-sample",
        scale_ot=False,
        zero_com=True,
    )
    interpolant = GeometricInterpolant(
        prior_sampler,
        coord_noise_std=coord_noise_std,
        coupling=coupling,
        kabsch_align=kabsch_align,
        fixed_time=t,
    )

    _, to_mols, interp_mols, _ = interpolant.interpolate(list(mols))

    n_atoms = max(mol.seq_length for mol in to_mols)
    data = geometric_batch_to_dict(GeometricMolBatch.from_list(list(to_mols)), n_atoms)
    interpolated = geometric_batch_to_dict(GeometricMolBatch.from_list(list(interp_mols)), n_atoms)

    scaled = t * data["coords"]
    cost = smolF.inter_distances(interpolated["coords"], scaled, sqrd=True)

    eps = 2.0 * ((1.0 - t) ** 2 + coord_noise_std ** 2)

    matrices = []
    for b, mol in enumerate(to_mols):
        n = mol.seq_length
        matrices.append(cost[b, :n, :n].double().numpy())

    return matrices, eps


def hard_plan(cost):
    """The Hungarian argmin as a 0/1 plan -- the estimator the baseline actually uses."""

    n = cost.shape[0]
    _, col = linear_sum_assignment(cost)
    plan = np.zeros((n, n))
    plan[np.arange(n), col] = 1.0
    return plan


def sinkhorn_plan(cost, eps, n_iters):
    cost_t = torch.from_numpy(cost).float().unsqueeze(0)
    mask = torch.ones((1, cost.shape[0]), dtype=torch.long)
    raw = smolF.sinkhorn_batched(cost_t, mask, torch.tensor([eps]), n_iters=n_iters)
    plan, _ = smolF.plan_from_sinkhorn(raw, mask)
    return plan[0].double().numpy()


def mcmc_plan(cost, eps, n_chains, n_iters, to_coords=None):
    """Empirical marginals from independent chains, each started at the identity like training."""

    n = cost.shape[0]
    cost_t = torch.from_numpy(cost).float().unsqueeze(0).expand(n_chains, n, n).contiguous()
    mask = torch.ones((n_chains, n), dtype=torch.long)
    init = torch.arange(n).unsqueeze(0).expand(n_chains, n).contiguous()
    coords = (
        to_coords.unsqueeze(0).expand(n_chains, n, 3).contiguous()
        if to_coords is not None
        else torch.zeros((n_chains, n, 3))
    )

    perms = smolF.mcmc_permutation(
        cost_t,
        mask,
        torch.full((n_chains,), eps),
        n_iters,
        init_perm=init,
        proposal="uniform" if to_coords is None else "knn",
        to_coords=None if to_coords is None else coords,
    )

    plan = np.zeros((n, n))
    for chain in perms.numpy():
        plan[np.arange(n), chain] += 1.0
    return plan / n_chains


def compare_estimators(cost, eps, args):
    """Deviation of each estimator from the exact permanent marginals, for one cost matrix."""

    weights = perm_util.weights_from_cost(cost, eps)
    exact = perm_util.permanent_marginals(weights)

    plans = {
        "hard": hard_plan(cost),
        "sinkhorn": sinkhorn_plan(cost, eps, args.sinkhorn_iters),
        "mcmc": mcmc_plan(cost, eps, args.mcmc_chains, args.mcmc_iters),
    }

    row = {"n": cost.shape[0], "exact_entropy": perm_util.normalised_entropy(exact)}
    for name, plan in plans.items():
        row[f"{name}_dev"] = float(np.abs(plan - exact).mean())
        row[f"{name}_entropy"] = perm_util.normalised_entropy(plan)

    return row


def run(mols, args):
    rows = []
    for t in args.times:
        torch.manual_seed(args.seed)
        matrices, eps = build_cost_matrices(mols, t, args.coord_noise_std_dev, coupling=args.coupling)

        per_t = []
        for cost in matrices:
            if cost.shape[0] > perm_util.MAX_EXACT_N:
                continue
            per_t.append(compare_estimators(cost, eps, args))

        if not per_t:
            continue

        summary = {"t": t, "eps": eps, "n_molecules": len(per_t)}
        for key in per_t[0]:
            if key == "n":
                continue
            summary[key] = float(np.mean([row[key] for row in per_t]))
        rows.append(summary)

    return rows


def print_rows(rows):
    print()
    header = (
        f"{'t':>6}{'eps':>8}{'mols':>6}"
        f"{'hard dev':>11}{'sink dev':>11}{'mcmc dev':>11}"
        f"{'exact H':>10}{'sink H':>10}{'mcmc H':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['t']:>6.2f}{row['eps']:>8.3f}{row['n_molecules']:>6}"
            f"{row['hard_dev']:>11.5f}{row['sinkhorn_dev']:>11.5f}{row['mcmc_dev']:>11.5f}"
            f"{row['exact_entropy']:>10.4f}{row['sinkhorn_entropy']:>10.4f}{row['mcmc_entropy']:>10.4f}"
        )

    print()
    print("dev = mean |estimator - exact permanent marginals|. H = normalised row entropy.")
    print("The mean-field claim predicts sink H > exact H at every t; if it does not hold, the")
    print("'systematically more diffuse than truth' line in the argument needs rewriting.")
    print()
    print("Read the mcmc column carefully: it uses many chains, so it measures the estimator's")
    print("BIAS (marginals converge to exact, because the sampler is unbiased). Training draws a")
    print("SINGLE sample, whose cost is variance, not bias -- that is semlaflow/target_variance.py.")
    print("The two scripts together are the bias/variance split: sinkhorn trades bias for low")
    print("variance, mcmc trades variance for no bias.")


def _load_mols(args, vocab):
    if args.data_path is None:
        print("No --data_path given: using synthetic molecules from the prior sampler.")
        print("These exercise the machinery but are NOT real cost matrices -- do not cite them.")
        torch.manual_seed(args.seed)
        sampler = GeometricNoiseSampler(vocab.size, 5, zero_com=True)
        return [sampler.sample_molecule(args.max_atoms) for _ in range(args.n_molecules)]

    coord_std = util.QM9_COORDS_STD_DEV if args.dataset == "qm9" else util.GEOM_COORDS_STD_DEV
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    dataset_path = Path(args.data_path) / _SPLIT_FILES[args.dataset_split]
    dataset = GeometricDataset.load(dataset_path, transform=transform)

    # Only molecules small enough for the exact permanent are usable
    mols = [dataset[i] for i in range(len(dataset))]
    mols = [mol for mol in mols if mol.seq_length <= args.max_atoms]
    if not mols:
        raise SystemExit(f"No molecules with <= {args.max_atoms} atoms in {dataset_path}.")

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(mols), size=min(args.n_molecules, len(mols)), replace=False)
    return [mols[i] for i in chosen]


def main(args):
    util.disable_lib_stdout()
    util.configure_fs()
    vocab = util.build_vocab()

    mols = _load_mols(args, vocab)
    print(f"Comparing estimators on {len(mols)} molecules of at most {args.max_atoms} atoms")
    print_rows(run(mols, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--categorical_strategy", type=str, default="uniform-sample")
    parser.add_argument("--coupling", type=str, default="hungarian")
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--coord_noise_std_dev", type=float, default=0.2)
    parser.add_argument("--sinkhorn_iters", type=int, default=DEFAULT_SINKHORN_ITERS)
    parser.add_argument("--mcmc_chains", type=int, default=DEFAULT_MCMC_CHAINS)
    parser.add_argument("--mcmc_iters", type=int, default=DEFAULT_MCMC_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)
