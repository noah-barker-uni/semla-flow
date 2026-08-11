import unittest

from rdkit import Chem

import semlaflow.util.metrics as metrics
import semlaflow.util.valency as valency


class ValencyTableTests(unittest.TestCase):
    def setUp(self):
        self.table = valency.load_valency_table()

    def test_table_is_keyed_by_element_and_charge(self):
        self.assertIn(("C", 0), self.table)
        self.assertIn(("N", 1), self.table)
        self.assertIn((0, 4), self.table[("C", 0)])

    def test_unknown_element_and_charge_are_unstable(self):
        self.assertFalse(valency.is_stable_atom("Xx", 0, 0, 1, self.table))
        self.assertFalse(valency.is_stable_atom("C", 7, 0, 4, self.table))

    def test_fractional_non_aromatic_valence_is_unstable(self):
        # A non-integral, non-aromatic bond order cannot match any table entry
        self.assertFalse(valency.is_stable_atom("C", 0, 0, 3.5, self.table))

    def test_split_bond_orders_counts_aromatics_separately(self):
        n_aromatic, non_aromatic = valency.split_bond_orders([1.5, 1.5, 1.0, 2.0])

        self.assertEqual(2, n_aromatic)
        self.assertEqual(3.0, non_aromatic)

    def test_legacy_qm9_extras_are_off_by_default(self):
        """QM9's NH+/NH2+ are not in the GEOM-Drugs reference table, and must not be silently added."""

        reference = valency.load_valency_table()
        legacy = valency.load_valency_table(allow_legacy_qm9=True)

        self.assertFalse(valency.is_stable_atom("N", 1, 0, 3, reference))
        self.assertTrue(valency.is_stable_atom("N", 1, 0, 3, legacy))
        # The reference behaviour is unaffected by having loaded the legacy variant
        self.assertFalse(valency.is_stable_atom("N", 1, 0, 3, valency.load_valency_table()))


class AromaticRoundingTests(unittest.TestCase):
    """The bug this table exists to fix.

    Summing bond orders with aromatic = 1.5 and truncating gives 1.5+1.5+1 = 4 for a pyrrole-type
    N-H, which fails a "neutral N has valence 3" check even though the atom is perfectly ordinary.
    Counting aromatic bonds separately removes the rounding entirely.
    """

    def setUp(self):
        self.table = valency.load_valency_table()

    def test_pyrrole_type_nitrogen_is_stable(self):
        orders = [1.5, 1.5, 1.0]

        # The trap: the naive sum is 4.0, which fails a "neutral N has valence 3" check
        self.assertEqual(4.0, sum(orders))

        n_aromatic, non_aromatic = valency.split_bond_orders(orders)
        self.assertEqual((2, 1.0), (n_aromatic, non_aromatic))
        self.assertTrue(valency.is_stable_atom("N", 0, n_aromatic, non_aromatic, self.table))

    def test_pyridine_type_nitrogen_is_stable(self):
        n_aromatic, non_aromatic = valency.split_bond_orders([1.5, 1.5])

        self.assertTrue(valency.is_stable_atom("N", 0, n_aromatic, non_aromatic, self.table))

    def test_aromatic_carbon_variants_are_stable(self):
        for orders in ([1.5, 1.5, 1.0], [1.5, 1.5, 1.5], [1.5, 1.5, 2.0]):
            n_aromatic, non_aromatic = valency.split_bond_orders(orders)
            self.assertTrue(
                valency.is_stable_atom("C", 0, n_aromatic, non_aromatic, self.table), str(orders)
            )

    def test_overvalent_carbon_is_unstable(self):
        n_aromatic, non_aromatic = valency.split_bond_orders([1.0, 1.0, 1.0, 1.0, 1.0])

        self.assertFalse(valency.is_stable_atom("C", 0, n_aromatic, non_aromatic, self.table))


class CalcAtomStabilitiesTests(unittest.TestCase):
    """The RDKit path, which paired_eval and the aggregate metrics both go through."""

    def _stabilities(self, smiles):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        return metrics.calc_atom_stabilities(mol)

    def test_common_valid_molecules_are_fully_stable(self):
        for smiles in [
            "CCO",
            "c1ccccc1",
            "c1ccncc1",
            "c1cc[nH]c1",
            "CC(=O)Nc1ccc(O)cc1",
            "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
            "C[N+](C)(C)C",
            "C1=CC=C(C=C1)[N+](=O)[O-]",
            "O=S(=O)(O)O",
        ]:
            stabilities = self._stabilities(smiles)
            self.assertTrue(all(stabilities), f"{smiles}: {stabilities}")

    def test_implicit_hydrogens_are_counted(self):
        """Without AddHs the hydrogens carry no Bond object and must come from GetTotalNumHs."""

        mol = Chem.MolFromSmiles("CCO")

        self.assertTrue(all(metrics.calc_atom_stabilities(mol)))

    def test_hypervalent_carbon_is_flagged(self):
        mol = Chem.MolFromSmiles("C[C](C)(C)(C)C", sanitize=False)
        mol.UpdatePropertyCache(strict=False)

        stabilities = metrics.calc_atom_stabilities(mol)

        self.assertFalse(all(stabilities))

    def test_returns_one_entry_per_atom(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))

        self.assertEqual(mol.GetNumAtoms(), len(metrics.calc_atom_stabilities(mol)))


if __name__ == "__main__":
    unittest.main()
