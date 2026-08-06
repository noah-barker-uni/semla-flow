import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

import semlaflow.util.geometry_metrics as geometry_metrics


def _mol_with_conformer(smiles, seed=0xF00D):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


class GeometryExtractionTests(unittest.TestCase):
    def setUp(self):
        self.ethanol = _mol_with_conformer("CCO")

    def test_bond_lengths_count_and_range(self):
        lengths = geometry_metrics.mol_bond_lengths(self.ethanol)

        self.assertEqual(self.ethanol.GetNumBonds(), len(lengths))
        for length in lengths:
            self.assertGreater(length, 0.5)
            self.assertLess(length, 1.8)

    def test_bond_angles_count(self):
        angles = geometry_metrics.mol_bond_angles(self.ethanol)

        expected = 0
        for atom in self.ethanol.GetAtoms():
            degree = atom.GetDegree()
            expected += degree * (degree - 1) // 2

        self.assertEqual(expected, len(angles))
        for angle in angles:
            self.assertGreater(angle, 0.0)
            self.assertLessEqual(angle, 180.0)

    def test_torsion_angles_one_per_non_terminal_bond(self):
        torsions = geometry_metrics.mol_torsion_angles(self.ethanol)

        expected = sum(
            1
            for bond in self.ethanol.GetBonds()
            if bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1
        )
        self.assertEqual(expected, len(torsions))

    def test_none_and_no_conformer_return_empty(self):
        self.assertEqual([], geometry_metrics.mol_bond_lengths(None))
        self.assertEqual([], geometry_metrics.mol_bond_angles(None))
        self.assertEqual([], geometry_metrics.mol_torsion_angles(None))

        no_conf_mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        self.assertEqual([], geometry_metrics.mol_bond_lengths(no_conf_mol))


class WassersteinDistanceTests(unittest.TestCase):
    def test_distance_to_self_is_zero(self):
        mols = [_mol_with_conformer("CCO", seed=1), _mol_with_conformer("CCO", seed=2)]

        self.assertAlmostEqual(0.0, geometry_metrics.wasserstein_bond_length(mols, mols), places=6)
        self.assertAlmostEqual(0.0, geometry_metrics.wasserstein_bond_angle(mols, mols), places=6)
        self.assertAlmostEqual(0.0, geometry_metrics.wasserstein_torsion_angle(mols, mols), places=6)

    def test_distance_between_different_molecules_is_nonnegative(self):
        mols_a = [_mol_with_conformer("CCO", seed=1)]
        mols_b = [_mol_with_conformer("c1ccccc1", seed=1)]

        self.assertGreaterEqual(geometry_metrics.wasserstein_bond_length(mols_a, mols_b), 0.0)
        self.assertGreaterEqual(geometry_metrics.wasserstein_bond_angle(mols_a, mols_b), 0.0)

    def test_none_mols_are_skipped_not_crashing(self):
        mols = [_mol_with_conformer("CCO"), None]
        distance = geometry_metrics.wasserstein_bond_length(mols, mols)
        self.assertGreaterEqual(distance, 0.0)


if __name__ == "__main__":
    unittest.main()
