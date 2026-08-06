"""Distributional (Wasserstein) comparison of 3D geometry between two sets of molecules.

Unlike semlaflow/util/paired_eval.py (per-molecule, positionally paired), these functions pool
a geometric quantity (bond length / bond angle / torsion angle) across every molecule in a set
into one distribution, then compare two such distributions -- typically a generated set against
a reference set of real molecules -- via Wasserstein (earth-mover's) distance. There is no
per-molecule pairing here; this answers "does the generated geometry distribution match the
real one overall", not "is molecule i better than molecule i".
"""

from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.stats import wasserstein_distance


def mol_bond_lengths(mol: Chem.rdchem.Mol) -> list[float]:
    if mol is None or mol.GetNumConformers() == 0:
        return []

    conf = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        try:
            lengths.append(rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        except Exception:
            continue

    return lengths


def mol_bond_angles(mol: Chem.rdchem.Mol) -> list[float]:
    if mol is None or mol.GetNumConformers() == 0:
        return []

    conf = mol.GetConformer()
    angles = []
    for atom in mol.GetAtoms():
        neighbour_idxs = [n.GetIdx() for n in atom.GetNeighbors()]
        for i, k in combinations(neighbour_idxs, 2):
            try:
                angles.append(rdMolTransforms.GetAngleDeg(conf, i, atom.GetIdx(), k))
            except Exception:
                continue

    return angles


def mol_torsion_angles(mol: Chem.rdchem.Mol) -> list[float]:
    """One representative dihedral per non-terminal bond (first available neighbour on each side)"""

    if mol is None or mol.GetNumConformers() == 0:
        return []

    conf = mol.GetConformer()
    torsions = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        j_neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        k_neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

        if not j_neighbours or not k_neighbours:
            continue

        try:
            torsions.append(rdMolTransforms.GetDihedralDeg(conf, j_neighbours[0], j, k, k_neighbours[0]))
        except Exception:
            continue

    return torsions


def _pooled(mols: list[Chem.rdchem.Mol], extract_fn) -> list[float]:
    pooled = []
    for mol in mols:
        pooled.extend(extract_fn(mol))
    return pooled


def wasserstein_bond_length(mols_a: list[Chem.rdchem.Mol], mols_b: list[Chem.rdchem.Mol]) -> float:
    return wasserstein_distance(_pooled(mols_a, mol_bond_lengths), _pooled(mols_b, mol_bond_lengths))


def wasserstein_bond_angle(mols_a: list[Chem.rdchem.Mol], mols_b: list[Chem.rdchem.Mol]) -> float:
    return wasserstein_distance(_pooled(mols_a, mol_bond_angles), _pooled(mols_b, mol_bond_angles))


def wasserstein_torsion_angle(mols_a: list[Chem.rdchem.Mol], mols_b: list[Chem.rdchem.Mol]) -> float:
    return wasserstein_distance(_pooled(mols_a, mol_torsion_angles), _pooled(mols_b, mol_torsion_angles))
