"""
Paired comparison between two trained checkpoints (eg. two coupling arms).

Unconditional generation has no natural pairing between "arm A's molecule i" and "arm B's
molecule i" -- each is freely sampled from noise by an independently-trained model, not a
reconstruction of a shared reference. The pairing used here is by test-set size-slot: both
arms are run with the same --seed, --n_molecules and --dataset_split, so both draw the exact
same sequence of real test-set molecule sizes to condition generation on (see
GeometricDataset.sample, which uses np.random.choice -- covered by L.seed_everything). Slot i
in both arms is therefore sized to match the same real molecule, which is what makes a paired
Wilcoxon signed-rank test meaningful here.

Note: size-stratified buckets below use each *generated* molecule's own atom count, not the
target size that seeded it -- simpler for this first pass, though the intended-size would be
a more principled stratification variable if threaded through separately later.
"""

import argparse
from pathlib import Path

import lightning as L
import numpy as np
from scipy.stats import wilcoxon

import semlaflow.scriptutil as util
import semlaflow.util.geometry_metrics as geometry_metrics
from semlaflow.data.datasets import GeometricDataset
from semlaflow.evaluate import dm_from_ckpt, load_model
from semlaflow.util.paired_eval import (
    per_molecule_energy,
    per_molecule_opt_rmsd,
    per_molecule_posebusters,
    per_molecule_stability,
    per_molecule_strain_energy,
    per_molecule_trajectory_straightness,
    per_molecule_validity,
    per_molecule_x1_movement,
)

DEFAULT_DATASET_SPLIT = "test"
DEFAULT_N_MOLECULES = 2000
DEFAULT_N_REFERENCE_MOLECULES = 2000
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"
DEFAULT_SEED = 12345

_SPLIT_FILES = {"train": "train.smol", "val": "val.smol", "test": "test.smol"}


def _label_for(ckpt_path, override):
    if override is not None:
        return override

    import torch

    hparams = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]
    return hparams.get("coupling", ckpt_path)


def _generate_arm(args, ckpt_path, vocab):
    """Build a datamodule/model for one checkpoint and generate its molecules + trajectories.

    Reseeds and rebuilds the datamodule per arm (rather than reusing one across arms) so that
    the same --seed reliably reproduces the same test-set size-draw sequence for every arm --
    this is what the slot-based pairing depends on.
    """

    L.seed_everything(args.seed)

    arm_args = argparse.Namespace(**vars(args))
    arm_args.ckpt_path = ckpt_path

    dm = dm_from_ckpt(arm_args, vocab)
    model = load_model(arm_args, vocab)

    molecules, outputs = util.generate_molecules(
        model, dm, args.integration_steps, args.ode_sampling_strategy, record_trajectory=True
    )

    straightness = []
    x1_movement = []
    for output in outputs:
        straightness.extend(per_molecule_trajectory_straightness(output["trajectory"], output["mask"]))
        x1_movement.extend(per_molecule_x1_movement(output["x1_trajectory"], output["mask"]))

    return molecules, straightness, x1_movement


def _load_reference_mols(args, vocab):
    """Real molecules (with their real 3D coordinates) for the Wasserstein geometry comparison.

    Loaded with transform=None so coordinates/atomics/charges stay in their raw, untransformed
    (real Angstrom units, real formal charges) form -- what GeometricMol.to_rdkit expects.
    """

    dataset_path = Path(args.data_path) / _SPLIT_FILES[args.dataset_split]
    dataset = GeometricDataset.load(dataset_path, transform=None)
    dataset = dataset.sample(min(args.n_reference_molecules, len(dataset)), replacement=False)
    return [dataset[i].to_rdkit(vocab) for i in range(len(dataset))]


def _collect_per_molecule(molecules):
    atom_stable_frac, mol_stable = per_molecule_stability(molecules)
    return {
        "validity": per_molecule_validity(molecules, connected=True),
        "posebusters-valid": per_molecule_posebusters(molecules),
        "atom-stable-frac": atom_stable_frac,
        "mol-stable": mol_stable,
        "energy-per-atom": per_molecule_energy(molecules, per_atom=True),
        "strain-energy-per-atom": per_molecule_strain_energy(molecules, per_atom=True),
        "opt-rmsd": per_molecule_opt_rmsd(molecules),
    }


def _paired_diffs(values_a, values_b):
    return [a - b for a, b in zip(values_a, values_b) if a is not None and b is not None]


def _run_wilcoxon(values_a, values_b):
    diffs = _paired_diffs(values_a, values_b)
    if len(diffs) < 2 or not any(diffs):
        return None, None, len(diffs)

    try:
        statistic, p_value = wilcoxon(diffs)
    except ValueError:
        return None, None, len(diffs)

    return statistic, p_value, len(diffs)


def _size_buckets(molecules, bucket_limits):
    sizes = [mol.GetNumAtoms() if mol is not None else None for mol in molecules]
    buckets = []
    for size in sizes:
        if size is None:
            buckets.append(None)
            continue
        bucket = next((limit for limit in bucket_limits if size <= limit), bucket_limits[-1])
        buckets.append(bucket)
    return buckets


def _stratified_means(values, buckets, bucket_limits):
    result = {}
    for limit in bucket_limits:
        bucket_values = [v for v, b in zip(values, buckets) if b == limit and v is not None]
        result[limit] = np.mean(bucket_values) if bucket_values else None
    return result


def compare(args, vocab):
    print(f"Generating {args.n_molecules} molecules for arm A ({args.label_a})...")
    molecules_a, straightness_a, x1_movement_a = _generate_arm(args, args.ckpt_path_a, vocab)

    print(f"Generating {args.n_molecules} molecules for arm B ({args.label_b})...")
    molecules_b, straightness_b, x1_movement_b = _generate_arm(args, args.ckpt_path_b, vocab)

    per_mol_a = _collect_per_molecule(molecules_a)
    per_mol_a["trajectory-straightness"] = straightness_a
    per_mol_a["x1-movement"] = x1_movement_a
    per_mol_b = _collect_per_molecule(molecules_b)
    per_mol_b["trajectory-straightness"] = straightness_b
    per_mol_b["x1-movement"] = x1_movement_b

    bucket_limits = util.QM9_BUCKET_LIMITS if args.dataset == "qm9" else util.GEOM_DRUGS_BUCKET_LIMITS
    buckets_a = _size_buckets(molecules_a, bucket_limits)

    print()
    print(f"{'Metric':<24}{'n_paired':<10}{'Wilcoxon stat':<16}{'p-value':<12}")
    print("-" * 62)

    for metric in per_mol_a:
        stat, p_value, n_paired = _run_wilcoxon(per_mol_a[metric], per_mol_b[metric])
        stat_str = f"{stat:.4f}" if stat is not None else "n/a"
        p_str = f"{p_value:.4g}" if p_value is not None else "n/a"
        print(f"{metric:<24}{n_paired:<10}{stat_str:<16}{p_str:<12}")

    print()
    print("Size-stratified means (generated-molecule atom count, bucketed):")
    for metric in per_mol_a:
        print(f"\n{metric}:")
        strat_a = _stratified_means(per_mol_a[metric], buckets_a, bucket_limits)
        strat_b = _stratified_means(per_mol_b[metric], buckets_a, bucket_limits)
        print(f"  {'bucket<=':<10}{args.label_a:<18}{args.label_b:<18}")
        for limit in bucket_limits:
            val_a = f"{strat_a[limit]:.4f}" if strat_a[limit] is not None else "n/a"
            val_b = f"{strat_b[limit]:.4f}" if strat_b[limit] is not None else "n/a"
            print(f"  {limit:<10}{val_a:<18}{val_b:<18}")

    print()
    print(f"Loading up to {args.n_reference_molecules} reference molecules from {args.dataset_split}.smol...")
    reference_mols = _load_reference_mols(args, vocab)

    print()
    print("Wasserstein distance to reference geometry (lower = closer to real data):")
    print(f"{'Metric':<16}{args.label_a:<18}{args.label_b:<18}")
    wasserstein_fns = {
        "bond-length": geometry_metrics.wasserstein_bond_length,
        "bond-angle": geometry_metrics.wasserstein_bond_angle,
        "torsion-angle": geometry_metrics.wasserstein_torsion_angle,
    }
    for name, fn in wasserstein_fns.items():
        dist_a = fn(molecules_a, reference_mols)
        dist_b = fn(molecules_b, reference_mols)
        print(f"{name:<16}{dist_a:<18.4f}{dist_b:<18.4f}")


def main(args):
    util.disable_lib_stdout()
    util.configure_fs()

    vocab = util.build_vocab()

    args.label_a = _label_for(args.ckpt_path_a, args.label_a)
    args.label_b = _label_for(args.ckpt_path_b, args.label_b)

    compare(args, vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path_a", type=str, required=True)
    parser.add_argument("--ckpt_path_b", type=str, required=True)
    parser.add_argument("--label_a", type=str, default=None)
    parser.add_argument("--label_b", type=str, default=None)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_reference_molecules", type=int, default=DEFAULT_N_REFERENCE_MOLECULES)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--n_layers", type=int, default=None)

    args = parser.parse_args()
    main(args)
