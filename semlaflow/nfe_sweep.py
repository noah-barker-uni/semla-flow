"""Quality vs number of function evaluations, for one or two checkpoints.

The highest-value missing experiment. The mechanism this project proposes -- a soft target gives a
better-conditioned regression, hence straighter learned paths -- predicts that the gap between arms
**widens as NFE falls**: with 100 integration steps a curved path is followed accurately anyway, so
the advantage of a straight one only shows when there are few steps to spend. Every result so far
was measured at a single fixed step count, so that prediction has never been tested.

Three presentations, because they answer different questions:

  (a) metric vs NFE per arm       -- the raw curves
  (b) the DIFFERENCE vs NFE       -- with a zero line, so "the gap widens" is the literal shape of
                                     the curve rather than something the reader has to infer
  (c) NFE to reach threshold tau  -- one interpretable number: "arm A needs 10 steps to hit the
                                     validity arm B needs 50 for"

--integration_steps already exists on evaluate.py and predict.py, so this is a sweep harness over
the existing generation path, not new metric code. Scale note from the corrections doc: ~1000
molecules per point is enough here; save the 5000-molecule runs for the headline table.

    python -m semlaflow.nfe_sweep --ckpt_path_a hard.ckpt --ckpt_path_b sinkhorn.ckpt \\
        --data_path <smol dir> --dataset qm9 --save_path sweep.json
"""

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np

import semlaflow.scriptutil as util
from semlaflow.evaluate import dm_from_ckpt, load_model

DEFAULT_NFE_VALUES = [1, 2, 5, 10, 20, 50, 100]
DEFAULT_N_MOLECULES = 1000
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"
DEFAULT_SEED = 12345

# Metrics worth plotting against NFE. Higher-is-better decides which side of a threshold counts as
# "reached", so it has to be declared rather than guessed from the name.
HIGHER_IS_BETTER = {
    "validity": True,
    "connected-validity": True,
    "uniqueness": True,
    "atom-stability": True,
    "molecule-stability": True,
    "energy-validity": True,
    "energy": False,
    "strain": False,
    "opt-rmsd": False,
}


def _sweep_arm(args, ckpt_path, vocab, label):
    """Generate and score at every NFE value for one checkpoint.

    The model and datamodule are built once and reused across NFE values, but the seed is reset
    before each point so that every NFE value sees the same test-set size draw -- otherwise the
    curve would confound step count with which molecules were asked for.
    """

    arm_args = argparse.Namespace(**vars(args))
    arm_args.ckpt_path = ckpt_path

    dm = dm_from_ckpt(arm_args, vocab)
    model = load_model(arm_args, vocab)
    metrics, stab_metrics = util.init_metrics(args.data_path, model)

    results = {}
    for steps in args.nfe_values:
        L.seed_everything(args.seed)
        print(f"  [{label}] generating {args.n_molecules} molecules at NFE={steps}...")

        molecules, stabilities = util.generate_molecules(
            model, dm, steps, args.ode_sampling_strategy, stabilities=True
        )
        scored = util.calc_metrics_(molecules, metrics, stab_metrics=stab_metrics, mol_stabs=stabilities)
        results[steps] = {name: float(value.item()) for name, value in scored.items()}

    return results


def nfe_to_threshold(curve, metric, threshold, nfe_values):
    """Smallest NFE at which `metric` reaches `threshold`, or None if it never does.

    Presentation (c): a single number that answers "how many steps does this arm need to be good
    enough", which is the question a practitioner choosing a sampler budget actually has.
    """

    higher_is_better = HIGHER_IS_BETTER.get(metric, True)
    for steps in sorted(nfe_values):
        value = curve.get(steps, {}).get(metric)
        if value is None:
            continue
        if (value >= threshold) if higher_is_better else (value <= threshold):
            return steps

    return None


def print_report(results, args):
    labels = list(results)
    metrics = sorted({m for curve in results.values() for point in curve.values() for m in point})

    print()
    print("(a) Metric vs NFE")
    for metric in metrics:
        print(f"\n{metric}:")
        header = f"  {'NFE':>6}" + "".join(f"{label:>16}" for label in labels)
        print(header)
        for steps in args.nfe_values:
            row = f"  {steps:>6}"
            for label in labels:
                value = results[label].get(steps, {}).get(metric)
                row += f"{value:>16.4f}" if value is not None else f"{'n/a':>16}"
            print(row)

    if len(labels) != 2:
        return

    label_a, label_b = labels
    print()
    print(f"(b) Difference vs NFE ({label_a} - {label_b}). The mechanism predicts |difference|")
    print("    grows as NFE falls; a flat line means NFE is not where the arms differ.")
    for metric in metrics:
        diffs = []
        for steps in args.nfe_values:
            value_a = results[label_a].get(steps, {}).get(metric)
            value_b = results[label_b].get(steps, {}).get(metric)
            diffs.append(None if value_a is None or value_b is None else value_a - value_b)

        if all(diff is None for diff in diffs):
            continue

        cells = "".join(f"{d:>+10.4f}" if d is not None else f"{'n/a':>10}" for d in diffs)
        print(f"  {metric:<22}{cells}")

    print(f"  {'NFE':<22}" + "".join(f"{steps:>10}" for steps in args.nfe_values))
    print("  (zero line = arms indistinguishable at that step count)")

    print()
    print("(c) NFE required to reach threshold")
    for metric, threshold in args.thresholds.items():
        cells = ""
        for label in labels:
            reached = nfe_to_threshold(results[label], metric, threshold, args.nfe_values)
            cells += f"{reached if reached is not None else 'never':>16}"
        print(f"  {metric:<22}{'tau=' + format(threshold, '.3f'):<12}{cells}")


def main(args):
    util.disable_lib_stdout()
    util.configure_fs()
    vocab = util.build_vocab()

    results = {}
    arms = [(args.label_a, args.ckpt_path_a)]
    if args.ckpt_path_b is not None:
        arms.append((args.label_b, args.ckpt_path_b))

    for label, ckpt_path in arms:
        print(f"Sweeping {label} over NFE {args.nfe_values}...")
        results[label] = _sweep_arm(args, ckpt_path, vocab, label)

    print_report(results, args)

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps({"nfe_values": args.nfe_values, "n_molecules": args.n_molecules, "results": results}, indent=2)
        )
        print(f"\nWrote {save_path}")


def _parse_thresholds(values):
    thresholds = {}
    for item in values:
        metric, _, threshold = item.partition("=")
        thresholds[metric] = float(threshold)
    return thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path_a", type=str, required=True)
    parser.add_argument("--ckpt_path_b", type=str, default=None)
    parser.add_argument("--label_a", type=str, default="arm-a")
    parser.add_argument("--label_b", type=str, default="arm-b")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="qm9")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--nfe_values", type=int, nargs="+", default=DEFAULT_NFE_VALUES)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=str,
        nargs="+",
        default=["validity=0.900", "molecule-stability=0.900"],
        help="metric=value pairs for the 'NFE to reach tau' table",
    )

    args = parser.parse_args()
    args.thresholds = _parse_thresholds(args.threshold)
    main(args)
