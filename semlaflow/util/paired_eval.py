"""Per-molecule (as opposed to batch-aggregate) generative metrics.

The Metric classes in semlaflow/util/metrics.py accumulate-then-reduce (torchmetrics style),
which is right for training-time validation logging but throws away individual molecule
results. Offline paper analysis -- paired Wilcoxon signed-rank tests, size-stratified
breakdowns -- needs the per-molecule values themselves. These functions call the exact same
underlying rdkit helpers the Metric classes use, just without the accumulation step, so the
per-molecule numbers can't drift from the aggregate ones.

Every function here returns one entry per input molecule (None where a value can't be
computed, eg. an invalid molecule) -- callers rely on this positional alignment to match
molecules across independently-generated arms by slot index.
"""

from typing import Optional

import torch
from rdkit import Chem

import semlaflow.util.rdkit as smolRD
from semlaflow.util.metrics import calc_atom_stabilities


def per_molecule_validity(mols: list[Chem.rdchem.Mol], connected: bool = False) -> list[bool]:
    return [mol is not None and smolRD.mol_is_valid(mol, connected=connected) for mol in mols]


def per_molecule_energy(
    mols: list[Chem.rdchem.Mol], optimise: bool = False, per_atom: bool = False
) -> list[Optional[float]]:
    results = []
    for mol in mols:
        if mol is None:
            results.append(None)
            continue

        target_mol = smolRD.optimise_mol(mol) if optimise else mol
        if target_mol is None:
            results.append(None)
            continue

        results.append(smolRD.calc_energy(target_mol, per_atom=per_atom))

    return results


def per_molecule_strain_energy(mols: list[Chem.rdchem.Mol], per_atom: bool = False) -> list[Optional[float]]:
    results = []
    for mol in mols:
        if mol is None:
            results.append(None)
            continue

        opt_mol = smolRD.optimise_mol(mol)
        if opt_mol is None:
            results.append(None)
            continue

        opt_energy = smolRD.calc_energy(opt_mol, per_atom=per_atom)
        if opt_energy is None:
            results.append(None)
            continue

        orig_energy = smolRD.calc_energy(mol, per_atom=per_atom)
        if orig_energy is None:
            results.append(None)
            continue

        results.append(orig_energy - opt_energy)

    return results


def per_molecule_opt_rmsd(mols: list[Chem.rdchem.Mol]) -> list[Optional[float]]:
    results = []
    for mol in mols:
        if mol is None:
            results.append(None)
            continue

        opt_mol = smolRD.optimise_mol(mol)
        if opt_mol is None:
            results.append(None)
            continue

        results.append(smolRD.conf_distance(mol, opt_mol))

    return results


def per_molecule_stability(mols: list[Chem.rdchem.Mol]) -> tuple[list[Optional[float]], list[Optional[bool]]]:
    """Returns (per-molecule fraction of stable atoms, per-molecule "is fully stable" bool)"""

    atom_stable_fracs = []
    mol_stables = []

    for mol in mols:
        if mol is None or mol.GetNumAtoms() == 0:
            atom_stable_fracs.append(None)
            mol_stables.append(None)
            continue

        stabilities = calc_atom_stabilities(mol)
        atom_stable_fracs.append(sum(stabilities) / len(stabilities))
        mol_stables.append(all(stabilities))

    return atom_stable_fracs, mol_stables


def per_molecule_trajectory_straightness(trajectory: torch.Tensor, mask: torch.Tensor) -> list[Optional[float]]:
    """Ratio of ODE path length to straight-line (chord) length, per molecule.

    Treats each molecule's full atom configuration at a given step as one point in R^(3*n_atoms)
    (not per-atom) -- the path length is the sum of step-to-step displacement norms in that
    space, the chord length is the displacement norm between the first and last step. A ratio of
    1 is a perfectly straight path; higher means more curved. None where the chord length is 0
    (degenerate, eg. a single-atom molecule that never had anywhere to move).

    Args:
        trajectory (torch.Tensor): Recorded coordinates at each step, shape [B, T, N, 3] (see
            MolecularCFM._generate's record_trajectory option).
        mask (torch.Tensor): Shape [B, N], 1 for real (non-padding) atoms.
    """

    if trajectory.shape[0] != mask.shape[0] or trajectory.shape[2] != mask.shape[1]:
        raise ValueError(
            f"trajectory shape {tuple(trajectory.shape)} and mask shape {tuple(mask.shape)} are inconsistent."
        )

    results = []
    for b in range(trajectory.size(0)):
        n_b = int(mask[b].sum().item())
        if n_b == 0:
            results.append(None)
            continue

        points = trajectory[b, :, :n_b, :].reshape(trajectory.size(1), -1)
        step_lengths = torch.norm(points[1:] - points[:-1], dim=-1)
        path_length = step_lengths.sum().item()
        chord_length = torch.norm(points[-1] - points[0]).item()

        results.append(path_length / chord_length if chord_length > 0 else None)

    return results
