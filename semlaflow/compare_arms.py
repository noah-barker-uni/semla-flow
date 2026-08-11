"""
Comparison between two trained checkpoints (eg. two arms of the coupling x target factorial).

Unconditional generation has no natural pairing between "arm A's molecule i" and "arm B's
molecule i" -- each is freely sampled from noise by an independently-trained model, not a
reconstruction of a shared reference. What IS shared is the size sequence: both arms run with the
same --seed, --n_molecules and --dataset_split, so both draw the same sequence of real test-set
molecule sizes to condition generation on (GeometricDataset.sample uses np.random.choice, covered
by L.seed_everything). Slot i in each arm is therefore sized to match the same real molecule.

That is blocking, NOT pairing, and the distinction decides which tests are legal. Arm A's molecule
i and arm B's molecule i are different molecules that happen to have the same atom count, so
Wilcoxon signed-rank -- which assumes genuinely paired observations -- does not apply and has been
removed. Instead:

  - the size-matched difference is reported as a variance-reduced DESCRIPTIVE;
  - the formal claim uses UNPAIRED Mann-Whitney U plus a bootstrap CI on the difference of medians;
  - every comparison carries an EFFECT SIZE (Cliff's delta), because p ~ 1e-195 at n=2000 reflects
    a systematic difference of unknown size, not an important one.

See semlaflow/util/stats.py. Across seeds, three points have almost no power for a formal test --
report the three seed-level values per arm and let them speak.

Note: size-stratified buckets below use each *generated* molecule's own atom count, not the
target size that seeded it -- simpler for this first pass, though the intended-size would be
a more principled stratification variable if threaded through separately later.
"""

import argparse
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch

import semlaflow.scriptutil as util
import semlaflow.util.geometry_metrics as geometry_metrics
import semlaflow.util.stats as stats
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler, coupling_transport_cost
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


def _transport_cost_for_arm(args, ckpt_path, vocab, n_batches=20, batch_size=64):
    """Measure E||x1 - x0^pi||^2 for the coupling this checkpoint was TRAINED with.

    Straightness's companion metric (see coupling_transport_cost). No model is loaded and no
    generation happens -- the interpolant is rebuilt from the checkpoint's own recorded training
    hparams and run over real molecules, so this is a property of the coupling config alone.
    """

    hparams = torch.load(ckpt_path, map_location="cpu")["hyper_parameters"]
    coupling = hparams.get("train-coupling", hparams.get("coupling"))
    kabsch = hparams.get("train-kabsch-align", hparams.get("kabsch_align", False))

    if coupling is None:
        return None, None

    dataset_path = Path(args.data_path) / _SPLIT_FILES[args.dataset_split]
    coord_std = util.QM9_COORDS_STD_DEV if args.dataset == "qm9" else util.GEOM_COORDS_STD_DEV
    n_bond_types = util.get_n_bond_types(hparams.get("train-bond-interpolation", "unmask"))
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)
    dataset = GeometricDataset.load(dataset_path, transform=transform)

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise="uniform-sample",
        bond_noise="uniform-sample",
        scale_ot=False,
        zero_com=True,
    )
    interpolant = GeometricInterpolant(
        prior_sampler, coupling=coupling, kabsch_align=bool(kabsch), fixed_time=0.5
    )

    L.seed_everything(args.seed)
    costs = []
    for start in range(0, min(n_batches * batch_size, len(dataset)), batch_size):
        to_mols = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        from_mols, to_out, _, _ = interpolant.interpolate(to_mols)
        costs.append(coupling_transport_cost(from_mols, to_out))

    return (float(np.mean(costs)) if costs else None), f"{coupling}/kabsch={bool(kabsch)}"


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


def _fmt(value, spec=".4f"):
    return format(value, spec) if value is not None and np.isfinite(value) else "n/a"


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
    print("Unpaired comparison (Mann-Whitney U). The arms are size-BLOCKED, not paired, so the")
    print("formal test must be unpaired -- see semlaflow/util/stats.py.")
    print()
    cliffs = "Cliff's d"
    header = (
        f"{'Metric':<24}{'median A':>11}{'median B':>11}{'mean A':>11}{'mean B':>11}"
        f"{'p':>11}{cliffs:>11}{'95% CI on median diff':>26}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for metric in per_mol_a:
        result = stats.compare_metric(per_mol_a[metric], per_mol_b[metric], seed=args.seed)
        results[metric] = result

        test, boot = result["test"], result["bootstrap"]
        ci = (
            f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}]"
            if boot["ci_low"] is not None
            else "n/a"
        )
        print(
            f"{metric:<24}"
            f"{_fmt(result['a']['median']):>11}{_fmt(result['b']['median']):>11}"
            f"{_fmt(result['a']['mean']):>11}{_fmt(result['b']['mean']):>11}"
            f"{_fmt(test['p'], '.3g'):>11}{_fmt(test['cliffs_delta'], '.3f'):>11}"
            f"{ci:>26}"
        )

    print()
    print("Effect size guide: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.")
    print("A tiny p with a negligible d means a real but unimportant difference.")

    print()
    print("Size-matched difference (A - B), DESCRIPTIVE ONLY -- blocking reduces variance but does")
    print("not license a paired test:")
    print(f"{'Metric':<24}{'n slots':>10}{'mean diff':>14}{'median diff':>14}")
    for metric in per_mol_a:
        matched = stats.size_matched_difference(per_mol_a[metric], per_mol_b[metric])
        print(
            f"{metric:<24}{matched['n_slots']:>10}"
            f"{_fmt(matched['mean_difference']):>14}{_fmt(matched['median_difference']):>14}"
        )

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
    print("Coupling transport cost E||x1 - x0^pi||^2 -- straightness's companion. No model is")
    print("involved, so a straightness gain WITHOUT a transport-cost change means the learned")
    print("paths straightened; a gain WITH one means the coupling moved and straightness may")
    print("simply be tracking that.")
    for label, ckpt in [(args.label_a, args.ckpt_path_a), (args.label_b, args.ckpt_path_b)]:
        cost, config = _transport_cost_for_arm(args, ckpt, vocab)
        print(f"  {label:<18}{_fmt(cost):>12}   (trained with coupling={config})")

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
