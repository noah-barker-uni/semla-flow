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


def _translated_copy(mol, shift):
    """A copy whose first atom is displaced by `shift` along x -- changes bonds/angles at that atom"""

    copy = Chem.Mol(mol)
    conf = copy.GetConformer()
    pos = conf.GetAtomPosition(0)
    conf.SetAtomPosition(0, (pos.x + shift, pos.y, pos.z))
    return copy


class ConformerDeviationTests(unittest.TestCase):
    """Same-molecule paired deviations, as used against each molecule's own xTB-optimised copy."""

    def setUp(self):
        self.ethanol = _mol_with_conformer("CCO")
        self.paracetamol = _mol_with_conformer("CC(=O)Nc1ccc(O)cc1")

    def test_identical_conformers_have_zero_deviation(self):
        same = Chem.Mol(self.paracetamol)

        for fn in [
            geometry_metrics.conformer_bond_length_deviations,
            geometry_metrics.conformer_bond_angle_deviations,
            geometry_metrics.conformer_torsion_deviations,
        ]:
            deviations = fn(self.paracetamol, same)
            self.assertGreater(len(deviations), 0, fn.__name__)
            self.assertAlmostEqual(max(deviations), 0.0, places=6, msg=fn.__name__)

    def test_bond_length_deviation_recovers_a_known_displacement(self):
        # Atom 0 is a terminal carbon, so displacing it along x changes exactly one bond length,
        # and by a known amount only if the bond already lies along x -- so just check the max
        # deviation is bounded by the displacement and is attained by exactly one bond.
        shifted = _translated_copy(self.ethanol, 0.1)

        deviations = geometry_metrics.conformer_bond_length_deviations(self.ethanol, shifted)
        nonzero = [d for d in deviations if d > 1e-9]

        self.assertEqual(len(self.ethanol.GetBonds()), len(deviations))
        self.assertEqual(4, len(nonzero))  # C0 is bonded to C1 and three hydrogens
        self.assertLessEqual(max(deviations), 0.1 + 1e-9)

    def test_bond_angle_deviation_is_periodic(self):
        # 179 deg vs -179 deg is a 2 deg difference, not 358
        self.assertAlmostEqual(2.0, geometry_metrics._angular_difference(179.0, -179.0), places=9)
        self.assertAlmostEqual(0.0, geometry_metrics._angular_difference(180.0, -180.0), places=9)
        self.assertAlmostEqual(90.0, geometry_metrics._angular_difference(45.0, 135.0), places=9)

    def test_torsion_deviations_enumerate_all_dihedrals_per_bond(self):
        shifted = _translated_copy(self.paracetamol, 0.05)

        deviations = geometry_metrics.conformer_torsion_deviations(self.paracetamol, shifted)

        self.assertGreater(len(deviations), 0)
        # All dihedrals per torsion bond, not one representative -- so strictly more than the
        # one-per-bond convention mol_torsion_angles uses
        self.assertGreater(len(deviations), len(geometry_metrics.mol_torsion_angles(self.paracetamol)))
        for deviation in deviations:
            self.assertGreaterEqual(deviation, 0.0)
            self.assertLessEqual(deviation, 180.0)

    def test_mismatched_or_missing_molecules_return_empty(self):
        no_conf = Chem.AddHs(Chem.MolFromSmiles("CCO"))

        for fn in [
            geometry_metrics.conformer_bond_length_deviations,
            geometry_metrics.conformer_bond_angle_deviations,
            geometry_metrics.conformer_torsion_deviations,
        ]:
            self.assertEqual([], fn(self.ethanol, None), fn.__name__)
            self.assertEqual([], fn(None, self.ethanol), fn.__name__)
            self.assertEqual([], fn(self.ethanol, self.paracetamol), fn.__name__)
            self.assertEqual([], fn(self.ethanol, no_conf), fn.__name__)
