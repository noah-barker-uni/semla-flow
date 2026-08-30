"""Per-molecule analysis of a generated set, read from the raw .smol rather than the SDF.

Why the .smol and not the SDF: `predict.py` writes the SDF with invalid molecules dropped, so its
length is already conditioned on success and its indices do not line up with the generated batch.
The raw batch has every molecule, valid or not, which is what a validity denominator needs -- and
it is the only place the coordinates of a *failed* generation survive at all.

That last point is the whole reason this script exists. The soft-target collapse predicts that the
model learns to place every atom at the molecular centroid, so the generated molecules should be
physically collapsed rather than merely mis-bonded. Radius of gyration measures that directly:

    Rg = sqrt( mean_i || x_i - centroid ||^2 )

A real QM9 molecule has Rg of roughly 1.5 A. Rg -> 0 means the atoms are piled on one point, which
no validity or stability metric would tell you, because those read the bond graph rather than the
geometry. Reported alongside the largest pairwise distance (the molecule's extent) and the mean
nearest-neighbour distance, which is the one that goes to zero first when atoms overlap.

Coordinates are in real Angstroms: MolecularCFM._generate multiplies by coord_scale before the
batch is saved.

    python -m semlaflow.analyse_generated --smol_path gen.smol --label none_hard \\
        --save_path results/analysis/none_hard.json
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

import semlaflow.scriptutil as util
from semlaflow.util.molrepr import GeometricMolBatch
from semlaflow.util.paired_eval import per_molecule_stability, per_molecule_validity

# Below this radius of gyration a "molecule" is a pile of atoms rather than a structure. Real QM9
# molecules sit near 1.5 A, so this is a wide margin and only fires on genuine collapse.
COLLAPSE_RG_THRESHOLD = 0.5


def _geometry(coords: np.ndarray) -> dict:
    """Radius of gyration, extent and nearest-neighbour spacing for one molecule, in Angstrom."""

    n = coords.shape[0]
    if n == 0:
        return {"radius_of_gyration": None, "max_extent": None, "mean_nn_distance": None}

    centred = coords - coords.mean(axis=0, keepdims=True)
    rg = float(np.sqrt((centred ** 2).sum(axis=1).mean()))

    if n < 2:
        return {"radius_of_gyration": rg, "max_extent": 0.0, "mean_nn_distance": None}

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=-1))
    max_extent = float(dists.max())

    np.fill_diagonal(dists, np.inf)
    mean_nn = float(dists.min(axis=1).mean())

    return {"radius_of_gyration": rg, "max_extent": max_extent, "mean_nn_distance": mean_nn}


def _to_rdkit_or_none(mol, vocab):
    """Convert to RDKit, or None if RDKit refuses the molecule outright.

    A collapsed generation can emit a bond list RDKit will not even build -- upstream's
    mol_from_atoms raises "bond already exists" when the predicted adjacency contains a duplicate
    edge. That is a validity failure, not an analysis failure: the whole point of this script is to
    characterise generations that went wrong, so it must not itself die on them.
    """

    try:
        return mol.to_rdkit(vocab)
    except Exception:
        return None


def analyse(mols, vocab) -> list[dict]:
    """One record per generated molecule: geometry plus the graph-level metrics.

    Geometry is computed from the raw coordinates and never depends on the molecule being valid,
    which is the point -- a collapsed generation has meaningful coordinates and meaningless bonds.
    """

    rdkit_mols = [_to_rdkit_or_none(mol, vocab) for mol in mols]
    validity = per_molecule_validity(rdkit_mols, connected=False)
    fc_validity = per_molecule_validity(rdkit_mols, connected=True)

    # Stability reads the bond graph, which can be malformed enough to raise even when the molecule
    # was constructible. Degrade to "unknown" rather than losing the whole run.
    try:
        atom_stable, mol_stable = per_molecule_stability(rdkit_mols)
    except Exception:
        atom_stable = [None] * len(rdkit_mols)
        mol_stable = [None] * len(rdkit_mols)

    records = []
    for idx, mol in enumerate(mols):
        coords = np.asarray(mol.coords.detach().cpu(), dtype=np.float64)
        record = {
            "index": idx,
            "n_atoms": int(mol.seq_length),
            "rdkit_constructible": rdkit_mols[idx] is not None,
            "valid": bool(validity[idx]),
            "fc_valid": bool(fc_validity[idx]),
            "atom_stable_frac": atom_stable[idx],
            "mol_stable": mol_stable[idx],
            **_geometry(coords),
        }
        records.append(record)

    return records


def summarise(records: list[dict]) -> dict:
    """Aggregate, reporting median AND mean for the continuous quantities."""

    n = len(records)
    summary = {"n_molecules": n}
    if n == 0:
        return summary

    summary["rdkit_constructible_fraction"] = sum(1 for r in records if r.get("rdkit_constructible")) / n
    summary["validity"] = sum(1 for r in records if r["valid"]) / n
    summary["fc_validity"] = sum(1 for r in records if r["fc_valid"]) / n
    summary["molecule_stability"] = sum(1 for r in records if r["mol_stable"]) / n

    atom_fracs = [r["atom_stable_frac"] for r in records if r["atom_stable_frac"] is not None]
    summary["atom_stability"] = float(statistics.fmean(atom_fracs)) if atom_fracs else None

    for key in ["radius_of_gyration", "max_extent", "mean_nn_distance", "n_atoms"]:
        values = [r[key] for r in records if r[key] is not None]
        if not values:
            summary[f"{key}-median"] = summary[f"{key}-mean"] = None
            continue
        summary[f"{key}-median"] = float(statistics.median(values))
        summary[f"{key}-mean"] = float(statistics.fmean(values))

    # The collapse signature: how much of the set is a pile of atoms rather than a structure
    rgs = [r["radius_of_gyration"] for r in records if r["radius_of_gyration"] is not None]
    summary["collapsed_fraction"] = (
        sum(1 for rg in rgs if rg < COLLAPSE_RG_THRESHOLD) / len(rgs) if rgs else None
    )
    summary["collapse_rg_threshold"] = COLLAPSE_RG_THRESHOLD
    return summary


def load_generated(smol_path: Path):
    return GeometricMolBatch.from_bytes(Path(smol_path).read_bytes()).to_list()


def main(args):
    RDLogger.DisableLog("rdApp.*")
    util.disable_lib_stdout()
    util.configure_fs()
    vocab = util.build_vocab()

    mols = load_generated(args.smol_path)
    if args.n_molecules is not None:
        mols = mols[: args.n_molecules]

    print(f"[{args.label}] analysing {len(mols)} generated molecules from {args.smol_path}")
    records = analyse(mols, vocab)
    summary = summarise(records)

    payload = {"label": args.label, "smol_path": str(args.smol_path), "summary": summary, "records": records}
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))

    print(f"  RDKit-constructible {summary.get('rdkit_constructible_fraction')}")
    print(f"  validity          {summary.get('validity')}")
    print(f"  Rg median/mean    {summary.get('radius_of_gyration-median')} / {summary.get('radius_of_gyration-mean')}")
    print(f"  mean NN distance  {summary.get('mean_nn_distance-median')}")
    print(f"  collapsed frac    {summary.get('collapsed_fraction')} (Rg < {COLLAPSE_RG_THRESHOLD} A)")
    print(f"  wrote {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smol_path", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_molecules", type=int, default=None)

    args = parser.parse_args()
    main(args)
