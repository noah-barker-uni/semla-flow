"""Target-variance diagnostic: how noisy is the regression target each arm trains against?

This replaces the "how often does the hard permutation flip" diagnostic from the original brief.
Flip rate only applies to hard-permutation arms -- under a soft target there is no pi to flip -- so
it cannot compare the thing the project is actually varying. The version that works across every
arm: fix x1 and t, resample x0 many times, and measure how much the resulting regression target
moves.

## Two quantities, and why both are reported

**target variance** -- Var over x0 draws of the target itself.

    Under `--target hard` this is EXACTLY ZERO by construction, and that is not a bug. The coupling
    permutes the PRIOR, leaving x1 in its original index order, so the hard target is x1 whatever
    x0 was drawn. Reported anyway because it is the quantity the corrections doc names, and because
    a reader needs to see the zero to understand why the second quantity exists.

**displacement variance** -- Var over x0 draws of (target - x_t).

    This is the comparable one, and it is what the loss gradient actually pushes on: the model at
    x_t is pulled towards the target, so the gradient is proportional to (prediction - target) and
    the noise that matters is in the displacement the model must learn to predict from that state.
    The hard arm is not degenerate here -- its target is fixed while x_t wanders with x0, so the
    displacement wanders too. A soft target computed FROM x_t tracks x_t, so if the posterior-mean
    claim is right this should be smaller. That is the variance reduction the method claims.

Neither number involves a trained model: this is a property of the coupling and target
construction alone, so it can be run before committing GPU hours to an arm.

    python -m semlaflow.target_variance --data_path <smol dir> --dataset qm9
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import geometric_batch_to_dict
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.models.fm import TARGET_TYPES, permutation_target
from semlaflow.util.molrepr import GeometricMolBatch

DEFAULT_N_MOLECULES = 32
DEFAULT_N_DRAWS = 64
DEFAULT_TIMES = [0.1, 0.3, 0.5, 0.7, 0.9]
DEFAULT_COUPLINGS = ["none", "hungarian"]
DEFAULT_SEED = 12345
_SPLIT_FILES = {"train": "train.smol", "val": "val.smol", "test": "test.smol"}


def _batch_from_mols(mols):
    n_atoms = max(mol.seq_length for mol in mols)
    return geometric_batch_to_dict(GeometricMolBatch.from_list(list(mols)), n_atoms)


def draw_targets(mols, coupling, target, t, n_draws, coord_noise_std, kabsch_align=True, **target_kwargs):
    """Resample x0 n_draws times at fixed x1 and t, returning the targets and states each produced.

    Returns:
        tuple: (targets [D,B,N,3], states [D,B,N,3], mask [B,N]) -- coordinates only, since the
            categorical channels are permuted by the same plan and carry the same information.
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

    targets, states = [], []
    mask = None
    for _ in range(n_draws):
        _, to_mols, interp_mols, times, _ = interpolant.interpolate(list(mols))

        data = _batch_from_mols(to_mols)
        interpolated = _batch_from_mols(interp_mols)
        times_t = torch.stack(list(times)).float()

        target_batch, _ = permutation_target(
            data, interpolated, times_t, target, noise_std=coord_noise_std, **target_kwargs
        )
        targets.append(target_batch["coords"])
        states.append(interpolated["coords"])
        mask = data["mask"]

    return torch.stack(targets), torch.stack(states), mask


def variance_summary(targets, states, mask):
    """Per-atom variance over draws, averaged over real atoms.

    Returns:
        dict: target_variance and displacement_variance, both summed over xyz so they are in the
            same units as a squared coordinate.
    """

    mask_f = mask.to(targets.dtype).unsqueeze(-1)
    n_real = mask.sum().item() or 1

    target_var = targets.var(dim=0, unbiased=False).sum(dim=-1)
    displacement = targets - states
    displacement_var = displacement.var(dim=0, unbiased=False).sum(dim=-1)

    return {
        "target_variance": float((target_var * mask_f.squeeze(-1)).sum().item() / n_real),
        "displacement_variance": float((displacement_var * mask_f.squeeze(-1)).sum().item() / n_real),
    }


def run(mols, args):
    rows = []
    for coupling in args.couplings:
        for target in args.targets:
            for t in args.times:
                torch.manual_seed(args.seed)
                targets, states, mask = draw_targets(
                    mols,
                    coupling,
                    target,
                    t,
                    args.n_draws,
                    args.coord_noise_std_dev,
                    sinkhorn_iters=args.target_sinkhorn_iters,
                    mcmc_iters=args.target_mcmc_iters,
                )
                summary = variance_summary(targets, states, mask)
                rows.append({"coupling": coupling, "target": target, "t": t, **summary})

    return rows


def print_rows(rows):
    print()
    print(f"{'coupling':<12}{'target':<10}{'t':>6}{'target var':>14}{'displacement var':>20}")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['coupling']:<12}{row['target']:<10}{row['t']:>6.2f}"
            f"{row['target_variance']:>14.5f}{row['displacement_variance']:>20.5f}"
        )

    print()
    print("target var is 0 for target=hard by construction -- the coupling permutes the PRIOR, so")
    print("the hard target is x1 whatever x0 was drawn. displacement var is the comparable column:")
    print("lower means the target moves less with the noise draw, ie. less gradient noise.")


def main(args):
    util.disable_lib_stdout()
    util.configure_fs()
    vocab = util.build_vocab()

    coord_std = util.QM9_COORDS_STD_DEV if args.dataset == "qm9" else util.GEOM_COORDS_STD_DEV
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    dataset_path = Path(args.data_path) / _SPLIT_FILES[args.dataset_split]
    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(min(args.n_molecules, len(dataset)), replacement=False)
    mols = [dataset[i] for i in range(len(dataset))]

    print(f"Measuring target variance over {args.n_draws} noise draws for {len(mols)} molecules")
    print_rows(run(mols, args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--categorical_strategy", type=str, default="uniform-sample")
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_draws", type=int, default=DEFAULT_N_DRAWS)
    parser.add_argument("--times", type=float, nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--couplings", type=str, nargs="+", default=DEFAULT_COUPLINGS)
    parser.add_argument("--targets", type=str, nargs="+", default=TARGET_TYPES)
    parser.add_argument("--coord_noise_std_dev", type=float, default=0.2)
    parser.add_argument("--target_sinkhorn_iters", type=int, default=100)
    parser.add_argument("--target_mcmc_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    args = parser.parse_args()
    main(args)
