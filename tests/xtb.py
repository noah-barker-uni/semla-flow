import json
import tempfile
import unittest
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

import semlaflow.util.xtb as smolXTB
import semlaflow.xtb_eval as xtb_eval

# Captured verbatim from xtb 6.7.1 (conda-forge, osx-arm64) optimising a distorted water molecule.
# Pinning the real text rather than a hand-written approximation is the point: the two numbers this
# project depends on are read out of a human-readable log that has no stable machine format, so the
# parser has to be tested against what xtb actually prints.
_REAL_XTB_OUTPUT = """
      -----------------------------------------------------------
     |                   =====================                   |
     |                        A N C O P T                        |
      -----------------------------------------------------------

   *** GEOMETRY OPTIMIZATION CONVERGED AFTER 5 ITERATIONS ***

------------------------------------------------------------------------
 total energy gain   :        -0.0074518 Eh       -4.6761 kcal/mol
 total RMSD          :         0.0841523 a0        0.0445 Å
 total power (kW/mol):        -3.9129264 (step)-1615.3228 (real)
------------------------------------------------------------------------

normal termination of xtb
"""

_FAILED_XTB_OUTPUT = """
   *** FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN 200 ITERATIONS ***

[ERROR] Program stopped due to fatal error
"""

# xtb writes the optimised geometry in the same atom order it was given, which is what lets the
# coordinates be dropped straight onto the input molecule's conformer.
_REAL_XTBOPT_XYZ = """3
 energy: -5.070544376395 gnorm: 0.000312583698 xtb: 6.7.1 (edcfbbe)
O           -0.00000000000000       -0.00000000026153        0.09578797903856
H           -0.00000000000000        0.77224440388470       -0.47289398962559
H           -0.00000000000000       -0.77224440362317       -0.47289398941297
"""


def _water():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    return mol


def _xtb_available():
    return smolXTB.xtb_version() is not None


class XtbOutputParsingTests(unittest.TestCase):
    def test_parses_real_output(self):
        parsed = smolXTB.parse_xtb_output(_REAL_XTB_OUTPUT)

        self.assertAlmostEqual(-4.6761, parsed["energy_gain_kcal"], places=4)
        self.assertAlmostEqual(0.0445, parsed["xtb_rmsd"], places=4)
        self.assertTrue(parsed["converged"])
        self.assertTrue(parsed["normal_termination"])

    def test_delta_e_relax_is_the_negated_gain_and_positive(self):
        """xtb reports the downhill energy change; the metric is how strained the input was.

        Getting this sign backwards would silently invert the entire headline result, so it is
        asserted directly rather than left to the reader of the docstring.
        """

        parsed = smolXTB.parse_xtb_output(_REAL_XTB_OUTPUT)

        self.assertAlmostEqual(4.6761, parsed["delta_e_relax"], places=4)
        self.assertGreater(parsed["delta_e_relax"], 0.0)
        self.assertAlmostEqual(-parsed["energy_gain_kcal"], parsed["delta_e_relax"], places=9)

    def test_reads_the_kcal_column_not_the_hartree_one(self):
        # The line carries both Eh and kcal/mol; picking the wrong column is a 627x error
        parsed = smolXTB.parse_xtb_output(_REAL_XTB_OUTPUT)

        self.assertNotAlmostEqual(0.0074518, parsed["delta_e_relax"], places=4)

    def test_failed_optimisation_yields_no_energy(self):
        parsed = smolXTB.parse_xtb_output(_FAILED_XTB_OUTPUT)

        self.assertIsNone(parsed["delta_e_relax"])
        self.assertFalse(parsed["converged"])
        self.assertFalse(parsed["normal_termination"])

    def test_empty_output_does_not_crash(self):
        parsed = smolXTB.parse_xtb_output("")

        self.assertIsNone(parsed["delta_e_relax"])
        self.assertIsNone(parsed["xtb_rmsd"])


class XyzRoundTripTests(unittest.TestCase):
    def test_xyz_block_has_a_line_per_atom_in_order(self):
        mol = _water()

        block = smolXTB.mol_to_xyz_block(mol)
        lines = block.splitlines()

        self.assertEqual("3", lines[0].strip())
        self.assertEqual(3, len(lines) - 2)
        self.assertEqual(
            [atom.GetSymbol() for atom in mol.GetAtoms()], [line.split()[0] for line in lines[2:]]
        )

    def test_rejects_molecule_without_conformer(self):
        with self.assertRaises(ValueError):
            smolXTB.mol_to_xyz_block(Chem.AddHs(Chem.MolFromSmiles("O")))

    def test_reads_optimised_geometry_onto_the_template(self):
        mol = _water()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xtbopt.xyz"
            path.write_text(_REAL_XTBOPT_XYZ)

            opt = smolXTB.read_xtbopt_xyz(path, mol)

        # Graph preserved, coordinates replaced
        self.assertEqual(mol.GetNumAtoms(), opt.GetNumAtoms())
        self.assertEqual(mol.GetNumBonds(), opt.GetNumBonds())
        pos = opt.GetConformer().GetAtomPosition(1)
        self.assertAlmostEqual(0.77224440388470, pos.y, places=8)
        self.assertAlmostEqual(-0.47289398962559, pos.z, places=8)

    def test_atom_count_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "xtbopt.xyz"
            path.write_text(_REAL_XTBOPT_XYZ)

            with self.assertRaises(ValueError):
                smolXTB.read_xtbopt_xyz(path, Chem.AddHs(Chem.MolFromSmiles("CCO")))


class BinaryResolutionTests(unittest.TestCase):
    def test_explicit_binary_wins(self):
        self.assertEqual("/some/xtb", smolXTB.resolve_xtb_binary("/some/xtb"))

    def test_missing_binary_reports_no_version(self):
        self.assertIsNone(smolXTB.xtb_version("/definitely/not/a/real/xtb"))

    def test_missing_binary_raises_xtb_error_rather_than_returning_a_bad_number(self):
        with self.assertRaises(smolXTB.XtbError):
            smolXTB.relax_molecule(_water(), binary="/definitely/not/a/real/xtb")

    def test_molecule_without_conformer_fails_softly(self):
        result = smolXTB.relax_molecule(Chem.AddHs(Chem.MolFromSmiles("O")))

        self.assertIsNone(result["delta_e_relax"])
        self.assertIsNotNone(result["error"])


class SummariseTests(unittest.TestCase):
    def _record(self, **kwargs):
        record = {
            "delta_e_relax": None,
            "bond_length_dev": None,
            "bond_angle_dev": None,
            "torsion_dev": None,
            "xtb_rmsd": None,
            "converged": False,
        }
        record.update(kwargs)
        return record

    def test_reports_both_median_and_mean(self):
        """The distribution is heavily right-skewed, so one number is not enough.

        With [1, 2, 3, 100] the median is 2.5 and the mean is 26.5 -- median tracks typical
        geometry quality, mean tracks catastrophic failures.
        """

        records = [self._record(delta_e_relax=v, converged=True) for v in [1.0, 2.0, 3.0, 100.0]]

        summary = xtb_eval.summarise(records)

        self.assertAlmostEqual(2.5, summary["delta_e_relax-median"], places=6)
        self.assertAlmostEqual(26.5, summary["delta_e_relax-mean"], places=6)

    def test_failed_molecules_are_excluded_but_counted(self):
        records = [
            self._record(delta_e_relax=10.0, converged=True),
            self._record(delta_e_relax=None),
        ]

        summary = xtb_eval.summarise(records)

        self.assertEqual(2, summary["n_molecules"])
        self.assertEqual(1, summary["n_succeeded"])
        self.assertAlmostEqual(0.5, summary["success_rate"], places=6)
        self.assertAlmostEqual(10.0, summary["delta_e_relax-mean"], places=6)

    def test_all_failed_gives_none_not_a_crash(self):
        summary = xtb_eval.summarise([self._record(), self._record()])

        self.assertIsNone(summary["delta_e_relax-median"])
        self.assertEqual(0.0, summary["success_rate"])

    def test_empty_input(self):
        summary = xtb_eval.summarise([])

        self.assertEqual(0, summary["n_molecules"])
        self.assertEqual(0.0, summary["success_rate"])


@unittest.skipUnless(_xtb_available(), "xtb binary not available")
class LiveXtbTests(unittest.TestCase):
    """Only runs where an xtb binary is present -- the rest of the suite stays CPU/dependency-free.

    Install with `conda create -p <prefix> -c conda-forge xtb` (there is no prebuilt aarch64
    release), then set SEMLAFLOW_XTB_BINARY=<prefix>/bin/xtb.
    """

    def test_relaxing_an_already_relaxed_geometry_costs_nothing(self):
        """The definition check: dE_relax must be ~0 at a minimum, and never negative."""

        mol = _water()
        first = smolXTB.relax_molecule(mol)
        self.assertIsNone(first["error"])

        second = smolXTB.relax_molecule(first["opt_mol"])

        self.assertIsNone(second["error"])
        self.assertGreaterEqual(second["delta_e_relax"], -1e-6)
        self.assertLess(second["delta_e_relax"], 0.1)

    def test_distorted_geometry_costs_something_and_moves(self):
        mol = _water()
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(1)
        conf.SetAtomPosition(1, (pos.x + 0.25, pos.y, pos.z))

        result = smolXTB.relax_molecule(mol)

        self.assertIsNone(result["error"])
        self.assertGreater(result["delta_e_relax"], 0.0)
        self.assertGreater(result["xtb_rmsd"], 0.0)

    def test_parsed_rmsd_matches_an_independent_recomputation(self):
        """Cross-check on the parser: xtb's own RMSD against one computed from the coordinates."""

        mol = _water()
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(2)
        conf.SetAtomPosition(2, (pos.x + 0.15, pos.y - 0.1, pos.z))

        result = smolXTB.relax_molecule(mol)
        recomputed = smolXTB.conformer_rmsd(mol, result["opt_mol"])

        self.assertAlmostEqual(result["xtb_rmsd"], recomputed, places=3)

    def test_end_to_end_record_is_json_serialisable(self):
        record = xtb_eval.evaluate_molecule(_water())

        self.assertIsNone(record["error"])
        self.assertIsNotNone(record["delta_e_relax"])
        self.assertIsNotNone(record["bond_length_dev"])
        json.dumps(record)


if __name__ == "__main__":
    unittest.main()
