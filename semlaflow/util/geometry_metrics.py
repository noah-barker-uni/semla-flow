"""3D geometry comparison, at two different levels of pairing.

**Distributional (the wasserstein_* functions).** Pool a geometric quantity (bond length / bond
angle / torsion angle) across every molecule in a set into one distribution, then compare two such
distributions -- typically a generated set against a reference set of real molecules -- via
Wasserstein (earth-mover's) distance. No per-molecule pairing: this answers "does the generated
geometry distribution match the real one overall", not "is molecule i better than molecule i".

**Same-molecule paired (the conformer_*_deviations functions).** Compare two conformers OF THE SAME
molecule over the same internal coordinates, and report how far each one moved. Used against each
molecule's own GFN2-xTB-optimised counterpart (see semlaflow/util/xtb.py), which is the field's
emerging standard and is strictly more sensitive than the distributional version: a set can match
the reference bond-length distribution while every individual molecule is badly strained.

Both differ again from semlaflow/util/paired_eval.py, which pairs molecule i of one arm against
molecule i of another arm.
"""

from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.stats import wasserstein_distance

# Matches github.com/isayevlab/geom-drugs-3dgen-evaluation's torsion enumeration: any bond between
# two non-terminal atoms that is not part of a triple bond. Deliberately NOT the same enumeration
# as mol_torsion_angles below, which takes one representative dihedral per non-terminal bond -- the
# deviation metric follows the reference implementation so the numbers are comparable to theirs.
_TORSION_BOND_SMARTS = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"


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


# --- Same-molecule paired deviations -------------------------------------------------------------
#
# These take two conformers of the SAME molecule -- in practice a generated geometry and its own
# GFN2-xTB-optimised counterpart -- and measure how far each internal coordinate moved. Both
# conformers must share the atom order, which holds because xtb writes xyz in the order it was
# given and semlaflow/util/xtb.py drops the optimised coordinates onto a copy of the input.


def _angular_difference(first: float, second: float) -> float:
    """Absolute difference between two angles in degrees, respecting 360-degree periodicity."""

    raw = abs(first - second)
    return min(raw, 360.0 - raw)


def _paired_conformers(mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol):
    if mol_a is None or mol_b is None:
        return None, None

    if mol_a.GetNumConformers() == 0 or mol_b.GetNumConformers() == 0:
        return None, None

    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return None, None

    return mol_a.GetConformer(), mol_b.GetConformer()


def conformer_bond_length_deviations(mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol) -> list[float]:
    """Per-bond |length_a - length_b|, in Angstrom. Topology is read from mol_a."""

    conf_a, conf_b = _paired_conformers(mol_a, mol_b)
    if conf_a is None:
        return []

    deviations = []
    for bond in mol_a.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            deviations.append(
                abs(rdMolTransforms.GetBondLength(conf_a, i, j) - rdMolTransforms.GetBondLength(conf_b, i, j))
            )
        except Exception:
            continue

    return deviations


def conformer_bond_angle_deviations(mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol) -> list[float]:
    """Per-angle difference over every neighbour pair at every atom, in degrees."""

    conf_a, conf_b = _paired_conformers(mol_a, mol_b)
    if conf_a is None:
        return []

    deviations = []
    for atom in mol_a.GetAtoms():
        neighbour_idxs = [n.GetIdx() for n in atom.GetNeighbors()]
        for i, k in combinations(neighbour_idxs, 2):
            j = atom.GetIdx()
            try:
                deviations.append(
                    _angular_difference(
                        rdMolTransforms.GetAngleDeg(conf_a, i, j, k),
                        rdMolTransforms.GetAngleDeg(conf_b, i, j, k),
                    )
                )
            except Exception:
                continue

    return deviations


def conformer_torsion_deviations(mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol) -> list[float]:
    """Per-dihedral difference in degrees, over every dihedral around every torsion bond.

    Enumerates ALL i-j-k-l dihedrals around each matching j-k bond, not one representative -- this
    follows the reference implementation rather than mol_torsion_angles' convention.
    """

    conf_a, conf_b = _paired_conformers(mol_a, mol_b)
    if conf_a is None:
        return []

    pattern = Chem.MolFromSmarts(_TORSION_BOND_SMARTS)
    if pattern is None:
        return []

    deviations = []
    for j, k in mol_a.GetSubstructMatches(pattern):
        j_neighbours = [n.GetIdx() for n in mol_a.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        k_neighbours = [n.GetIdx() for n in mol_a.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

        for i in j_neighbours:
            for l in k_neighbours:
                if i == l:
                    continue
                try:
                    deviations.append(
                        _angular_difference(
                            rdMolTransforms.GetDihedralDeg(conf_a, i, j, k, l),
                            rdMolTransforms.GetDihedralDeg(conf_b, i, j, k, l),
                        )
                    )
                except Exception:
                    continue

    return deviations
