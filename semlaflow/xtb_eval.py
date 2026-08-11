"""GFN2-xTB evaluation of generated molecules.

Reads an SDF (as written by `python -m semlaflow.predict`), GFN2-xTB relaxes every molecule, and
reports how far each one had to move to reach its own local minimum. This is the primary energy
metric for the project; the MMFF numbers in evaluate.py are a coarse outlier filter only, because
MMFF's own 15-20 kcal/mol error against GEOM-Drugs reference conformers is larger than the effect
being measured (Nikitin et al., arXiv 2505.00169).

Metrics reported, each as BOTH median and mean -- the distribution is heavily right-skewed, and
the two answer different questions: median = typical geometry quality, mean = rate of catastrophic
failures. Reporting only one hides half the story.

  dE_relax          kcal/mol, >= 0, larger is worse. The headline number.
  bond-length-dev   Angstrom, mean per molecule over its bonds
  bond-angle-dev    degrees, mean per molecule
  torsion-dev       degrees, mean per molecule
  xtb-rmsd          Angstrom, how far the geometry physically moved

This is CPU work and embarrassingly parallel, so it belongs on a CPU allocation rather than
burning GPU hours. On Isambard that is b35bs.3.isambard / b35bs.macs3.isambard, not brics.b5bg.

    #SBATCH --account=b35bs.3.isambard
    #SBATCH --partition=workq
    #SBATCH --time=04:00:00
    python -m semlaflow.xtb_eval --sdf_path gen.sdf --save_path gen.xtb.json --n_workers 64

Per-molecule results are written as JSON so the aggregate table, the size-stratified breakdown and
any arm-vs-arm comparison all read the same numbers. Note the corrections doc's statistics rule:
use unpaired tests (Mann-Whitney U, or bootstrap CIs on the difference of medians) on these, not
Wilcoxon signed-rank -- arms are matched on size, which is blocking, not true pairing.
"""

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

from rdkit import Chem, RDLogger

import semlaflow.util.geometry_metrics as geom
import semlaflow.util.xtb as smolXTB

DEFAULT_N_WORKERS = 8
DEFAULT_TIMEOUT = smolXTB.DEFAULT_TIMEOUT_SECONDS

# Scale from the corrections doc: 5000 for the headline table, ~1000 per point for the NFE sweep.
DEFAULT_N_MOLECULES = None

_WORKER_STATE = {}


def _init_worker(binary, timeout):
    RDLogger.DisableLog("rdApp.*")
    _WORKER_STATE["binary"] = binary
    _WORKER_STATE["timeout"] = timeout


def _mean_or_none(values):
    return float(statistics.fmean(values)) if values else None


def evaluate_molecule(mol, binary=None, timeout=DEFAULT_TIMEOUT):
    """Relax one molecule and measure how far every internal coordinate moved.

    Returns a record with None values (and an `error`) rather than raising, so one pathological
    molecule cannot take down a 5000-molecule sweep.
    """

    record = {
        "n_atoms": None,
        "smiles": None,
        "delta_e_relax": None,
        "xtb_rmsd": None,
        "recomputed_rmsd": None,
        "bond_length_dev": None,
        "bond_angle_dev": None,
        "torsion_dev": None,
        "converged": False,
        "error": None,
    }

    if mol is None:
        record["error"] = "mol is None"
        return record

    record["n_atoms"] = mol.GetNumAtoms()
    try:
        record["smiles"] = Chem.MolToSmiles(mol)
    except Exception:
        pass

    result = smolXTB.relax_molecule(mol, binary=binary, timeout=timeout)

    record["delta_e_relax"] = result["delta_e_relax"]
    record["xtb_rmsd"] = result["xtb_rmsd"]
    record["converged"] = bool(result["converged"])
    record["error"] = result["error"]

    opt_mol = result["opt_mol"]
    if opt_mol is None:
        return record

    record["recomputed_rmsd"] = smolXTB.conformer_rmsd(mol, opt_mol)
    record["bond_length_dev"] = _mean_or_none(geom.conformer_bond_length_deviations(mol, opt_mol))
    record["bond_angle_dev"] = _mean_or_none(geom.conformer_bond_angle_deviations(mol, opt_mol))
    record["torsion_dev"] = _mean_or_none(geom.conformer_torsion_deviations(mol, opt_mol))
    return record


def _evaluate_one(mol):
    return evaluate_molecule(mol, binary=_WORKER_STATE["binary"], timeout=_WORKER_STATE["timeout"])


def load_molecules(sdf_path, n_molecules=None):
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols = []
    for mol in supplier:
        mols.append(mol)
        if n_molecules is not None and len(mols) >= n_molecules:
            break

    return mols


def run(mols, binary=None, timeout=DEFAULT_TIMEOUT, n_workers=DEFAULT_N_WORKERS):
    if n_workers <= 1:
        _init_worker(binary, timeout)
        return [_evaluate_one(mol) for mol in mols]

    # spawn rather than fork: xtb is an external process but rdkit/BLAS state in a forked worker is
    # the same aarch64 hazard the OpenMP note in semlaflow/__init__.py describes.
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers, initializer=_init_worker, initargs=(binary, timeout)) as pool:
        return list(pool.imap(_evaluate_one, mols, chunksize=1))


def summarise(records):
    """Median AND mean for every metric, plus how much of the set actually produced a number."""

    summary = {"n_molecules": len(records)}

    n_ok = sum(1 for r in records if r["delta_e_relax"] is not None)
    summary["n_succeeded"] = n_ok
    summary["success_rate"] = n_ok / len(records) if records else 0.0
    summary["converged_rate"] = (
        sum(1 for r in records if r["converged"]) / len(records) if records else 0.0
    )

    for key in ["delta_e_relax", "bond_length_dev", "bond_angle_dev", "torsion_dev", "xtb_rmsd"]:
        values = [r[key] for r in records if r[key] is not None]
        if not values:
            summary[f"{key}-median"] = None
            summary[f"{key}-mean"] = None
            summary[f"{key}-std"] = None
            continue

        summary[f"{key}-median"] = float(statistics.median(values))
        summary[f"{key}-mean"] = float(statistics.fmean(values))
        summary[f"{key}-std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

    return summary


def main(args):
    RDLogger.DisableLog("rdApp.*")

    version = smolXTB.xtb_version(args.xtb_binary)
    if version is None:
        binary = smolXTB.resolve_xtb_binary(args.xtb_binary)
        raise SystemExit(
            f"xtb executable '{binary}' not found or not runnable.\n"
            f"There is no prebuilt aarch64 xtb release, but conda-forge builds it for "
            f"linux-aarch64 and osx-arm64:\n"
            f"    conda create -p <prefix> -c conda-forge xtb\n"
            f"then either put <prefix>/bin on PATH or set "
            f"{smolXTB.XTB_BINARY_ENV_VAR}=<prefix>/bin/xtb (or pass --xtb_binary)."
        )

    print(f"Using xtb version {version}")

    mols = load_molecules(args.sdf_path, args.n_molecules)
    print(f"Loaded {len(mols)} molecules from {args.sdf_path}")

    records = run(mols, binary=args.xtb_binary, timeout=args.timeout, n_workers=args.n_workers)
    summary = summarise(records)

    payload = {
        "xtb_version": version,
        "sdf_path": str(args.sdf_path),
        "summary": summary,
        "records": records,
    }

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))

    print()
    print(f"{'Metric':<28}{'Median':>12}{'Mean':>12}{'Std':>12}")
    print("-" * 64)
    for key in ["delta_e_relax", "bond_length_dev", "bond_angle_dev", "torsion_dev", "xtb_rmsd"]:
        median, mean, std = summary[f"{key}-median"], summary[f"{key}-mean"], summary[f"{key}-std"]
        if median is None:
            print(f"{key:<28}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
        else:
            print(f"{key:<28}{median:>12.4f}{mean:>12.4f}{std:>12.4f}")

    print("-" * 64)
    print(f"{'success rate':<28}{summary['success_rate']:>12.4f}")
    print(f"{'converged rate':<28}{summary['converged_rate']:>12.4f}")
    print()
    print(f"Wrote per-molecule results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_path", type=str, required=True, help="SDF written by semlaflow.predict")
    parser.add_argument("--save_path", type=str, required=True, help="Where to write the JSON results")
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--n_workers", type=int, default=DEFAULT_N_WORKERS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--xtb_binary",
        type=str,
        default=None,
        help=f"xtb executable; defaults to ${smolXTB.XTB_BINARY_ENV_VAR} or 'xtb' on PATH",
    )

    args = parser.parse_args()
    main(args)
