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


def geometry_records(mols) -> list[dict]:
    """Per-molecule geometry, straight from the coordinates.

    Deliberately does NOT go through RDKit. GeometricMol.to_rdkit cannot be used on a batch saved
    by predict.py: _smol_from_tensors stores the FULL dense directed adjacency
    (torch.ones((n, n)).nonzero(), so both (i, j) and (j, i) plus the diagonal), whereas to_rdkit
    expects a dataset-style deduplicated bond list. Feeding it the dense form yields an invalid
    molecule for a perfectly good generation, and raises "bond already exists" outright when the
    predicted types make a reversed pair non-zero. Graph metrics come from the SDF instead, which
    predict.py builds through MolBuilder's proper sanitising path.
    """

    records = []
    for idx, mol in enumerate(mols):
        coords = np.asarray(mol.coords.detach().cpu(), dtype=np.float64)
        records.append({"index": idx, "n_atoms": int(mol.seq_length), **_geometry(coords)})

    return records


def graph_metrics(sdf_path: Path, n_generated: int) -> dict:
    """Validity and stability from the SDF, with the number GENERATED as the denominator.

    predict.py writes only the molecules RDKit could build, so the SDF length is already
    conditioned on success. Dividing by n_generated rather than by len(sdf) is what makes these
    comparable to the validity logged during training.
    """

    from rdkit import Chem

    if not Path(sdf_path).exists() or n_generated == 0:
        return {}

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mols = [mol for mol in supplier if mol is not None]

    validity = per_molecule_validity(mols, connected=False)
    fc_validity = per_molecule_validity(mols, connected=True)
    atom_stable, mol_stable = per_molecule_stability(mols)

    atom_fracs = [a for a in atom_stable if a is not None]
    return {
        "n_in_sdf": len(mols),
        "rdkit_constructible_fraction": len(mols) / n_generated,
        "validity": sum(1 for v in validity if v) / n_generated,
        "fc_validity": sum(1 for v in fc_validity if v) / n_generated,
        "molecule_stability": sum(1 for v in mol_stable if v) / n_generated,
        "atom_stability": float(statistics.fmean(atom_fracs)) if atom_fracs else None,
    }


def summarise(records: list[dict], graph: dict = None) -> dict:
    """Aggregate, reporting median AND mean for the continuous quantities."""

    n = len(records)
    summary = {"n_molecules": n}
    if n == 0:
        return summary

    summary.update(graph or {})

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
    records = geometry_records(mols)

    sdf_path = Path(args.sdf_path) if args.sdf_path else Path(str(args.smol_path) + ".sdf")
    graph = graph_metrics(sdf_path, len(mols))
    if not graph:
        print(f"  (no SDF at {sdf_path} -- geometry only; zero valid molecules is itself a result)")
    summary = summarise(records, graph)

    payload = {"label": args.label, "smol_path": str(args.smol_path), "summary": summary, "records": records}
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2))

    print(f"  RDKit-constructible {summary.get('rdkit_constructible_fraction')}")
    print(f"  n in sdf          {summary.get('n_in_sdf')}")
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
    parser.add_argument("--sdf_path", type=str, default=None)
    parser.add_argument("--n_molecules", type=int, default=None)

    args = parser.parse_args()
    main(args)
