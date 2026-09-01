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
import json
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

    targets, states, diagnostics = [], [], []
    mask = None
    data_coords = None
    for _ in range(n_draws):
        _, to_mols, interp_mols, times, _ = interpolant.interpolate(list(mols))

        data = _batch_from_mols(to_mols)
        interpolated = _batch_from_mols(interp_mols)
        times_t = torch.stack(list(times)).float()

        target_batch, diag = permutation_target(
            data, interpolated, times_t, target, noise_std=coord_noise_std, **target_kwargs
        )
        targets.append(target_batch["coords"])
        states.append(interpolated["coords"])
        diagnostics.append(diag)
        mask = data["mask"]
        data_coords = data["coords"]

    return torch.stack(targets), torch.stack(states), mask, data_coords, diagnostics


def variance_summary(targets, states, mask, data_coords=None, diagnostics=None):
    """Per-atom variance over draws, plus the target-collapse curve.

    The collapse quantities are the mechanism behind the soft-target failure and are the reason
    this reports more than variance. As the plan goes uniform, P @ x1 becomes the row-mean of x1,
    which is the molecular centroid -- and QM9 molecules are zero-COM, so that centroid is the
    ORIGIN. norm_ratio = ||target|| / ||x1|| therefore falls to ~0 at low t: the regression label
    is all zeros. eff_atoms = 1 / sum_j P_ij^2 says how many atoms are being averaged to get there.

    Returns:
        dict: target_variance, displacement_variance (both summed over xyz, so squared-coordinate
            units), plus target_norm, data_norm, norm_ratio and eff_atoms where available.
    """

    mask_f = mask.to(targets.dtype).unsqueeze(-1)
    n_real = mask.sum().item() or 1

    target_var = targets.var(dim=0, unbiased=False).sum(dim=-1)
    displacement = targets - states
    displacement_var = displacement.var(dim=0, unbiased=False).sum(dim=-1)

    summary = {
        "target_variance": float((target_var * mask_f.squeeze(-1)).sum().item() / n_real),
        "displacement_variance": float((displacement_var * mask_f.squeeze(-1)).sum().item() / n_real),
    }

    # Mean per-atom distance from the origin, for the target and for x1 itself
    def _norm(coords):
        per_atom = (coords * mask_f).pow(2).sum(dim=-1).sqrt()
        return float(per_atom.sum().item() / n_real)

    summary["target_norm"] = float(np.mean([_norm(t) for t in targets]))
    if data_coords is not None:
        summary["data_norm"] = _norm(data_coords)
        summary["norm_ratio"] = (
            summary["target_norm"] / summary["data_norm"] if summary["data_norm"] > 0 else None
        )

    if diagnostics:
        # sum_p_squared is already per MOLECULE (mean over that molecule's rows), so invert before
        # averaging over molecules rather than after: n_eff is a per-molecule quantity and
        # mean(1/s) != 1/mean(s). The earlier version collapsed the batch first, which is why the
        # number moved slightly when this was corrected.
        sums = [d["sinkhorn/sum_p_squared"] for d in diagnostics if "sinkhorn/sum_p_squared" in d]
        ents = [d["sinkhorn/plan_entropy"] for d in diagnostics if "sinkhorn/plan_entropy" in d]
        if sums:
            stacked = torch.stack(sums).to(targets.dtype)              # [draws, batch]
            n_real_per_mol = mask.sum(dim=1).to(targets.dtype).clamp_min(1.0)
            summary["eff_atoms"] = float((1.0 / stacked).mean().item())

            # The size-free version: what FRACTION of its own molecule each target row averages.
            # n_eff alone is not comparable across molecules of different sizes -- 17 atoms is the
            # whole of a QM9 molecule but half of a GEOM-Drugs one -- so this is the number to
            # report when molecules vary in size.
            summary["eff_atoms_frac"] = float((1.0 / (stacked * n_real_per_mol)).mean().item())
        if ents:
            summary["plan_entropy"] = float(torch.stack(ents).mean().item())

    return summary


def run(mols, args):
    rows = []
    for coupling in args.couplings:
        for target in args.targets:
            for t in args.times:
                torch.manual_seed(args.seed)
                targets, states, mask, data_coords, diags = draw_targets(
                    mols,
                    coupling,
                    target,
                    t,
                    args.n_draws,
                    args.coord_noise_std_dev,
                    sinkhorn_iters=args.target_sinkhorn_iters,
                    mcmc_iters=args.target_mcmc_iters,
                )
                summary = variance_summary(targets, states, mask, data_coords, diags)
                rows.append({"coupling": coupling, "target": target, "t": t, **summary})

    return rows


def print_rows(rows):
    def fmt(value, spec=">10.4f"):
        return format(value, spec) if isinstance(value, (int, float)) else f"{'n/a':>10}"

    print()
    header = (
        f"{'coupling':<11}{'target':<10}{'t':>5}{'tgt var':>10}{'disp var':>10}"
        f"{'||tgt||':>10}{'||x1||':>10}{'ratio':>10}{'eff atoms':>10}{'eff/N':>10}{'plan H':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['coupling']:<11}{row['target']:<10}{row['t']:>5.2f}"
            f"{fmt(row.get('target_variance'))}{fmt(row.get('displacement_variance'))}"
            f"{fmt(row.get('target_norm'))}{fmt(row.get('data_norm'))}{fmt(row.get('norm_ratio'))}"
            f"{fmt(row.get('eff_atoms'))}{fmt(row.get('eff_atoms_frac'))}{fmt(row.get('plan_entropy'))}"
        )

    print()
    print("tgt var is 0 for target=hard by construction -- the coupling permutes the PRIOR, so the")
    print("hard target is x1 whatever x0 was drawn. disp var is the comparable column: lower means")
    print("the target moves less with the noise draw, ie. less gradient noise.")
    print()
    print("ratio = ||target|| / ||x1||. THE COLLAPSE: as the plan goes uniform, P @ x1 becomes the")
    print("row-mean of x1 = the molecular centroid, and QM9 molecules are zero-COM, so that is the")
    print("ORIGIN. ratio -> 0 at low t means the regression label is all zeros. eff atoms is how")
    print("many atoms are averaged to get there (1 = a single atom, n = the whole molecule).")


def main(args):
    util.disable_lib_stdout()
    util.configure_fs()
    vocab = util.build_vocab()

    coord_std = util.QM9_COORDS_STD_DEV if args.dataset == "qm9" else util.GEOM_COORDS_STD_DEV
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    dataset_path = Path(args.data_path) / _SPLIT_FILES[args.dataset_split]
    dataset = GeometricDataset.load(dataset_path, transform=transform)

    # Seed BEFORE sampling the molecules, not just before each row's noise draws. Without this the
    # subset differs between invocations and the reported numbers move by a percent or two, which
    # is invisible unless you happen to rerun with a different --times list and compare.
    torch.manual_seed(args.seed)
    dataset = dataset.sample(min(args.n_molecules, len(dataset)), replacement=False)
    mols = [dataset[i] for i in range(len(dataset))]

    print(f"Measuring target variance over {args.n_draws} noise draws for {len(mols)} molecules")
    rows = run(mols, args)
    print_rows(rows)

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(
                {"n_molecules": len(mols), "n_draws": args.n_draws, "rows": rows}, indent=2
            )
        )
        print(f"\nWrote {save_path}")


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
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
