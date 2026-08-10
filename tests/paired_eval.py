import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import semlaflow.util.metrics as metrics
import semlaflow.util.paired_eval as paired_eval


def _mol_with_conformer(smiles, seed=0xF00D):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


class PairedEvalTests(unittest.TestCase):
    def setUp(self):
        # ethanol, water, methane -- small, valid, embeddable molecules
        self.mols = [_mol_with_conformer("CCO"), _mol_with_conformer("O"), _mol_with_conformer("C")]

    def test_per_molecule_validity_matches_validity_aggregate(self):
        mols_with_none = [self.mols[0], None, self.mols[1]]

        per_mol = paired_eval.per_molecule_validity(mols_with_none, connected=True)

        metric = metrics.Validity(connected=True)
        metric.update(mols_with_none)
        aggregate = metric.compute().item()

        self.assertEqual([True, False, True], per_mol)
        self.assertAlmostEqual(aggregate, np.mean(per_mol), places=5)

    def test_per_molecule_validity_preserves_position_of_none(self):
        mols_with_none = [self.mols[0], None, self.mols[1]]
        per_mol = paired_eval.per_molecule_validity(mols_with_none)
        self.assertEqual(3, len(per_mol))
        self.assertFalse(per_mol[1])

    def test_per_molecule_energy_matches_average_energy_aggregate(self):
        per_mol = paired_eval.per_molecule_energy(self.mols, per_atom=False)

        metric = metrics.AverageEnergy(per_atom=False)
        metric.update(self.mols)
        aggregate = metric.compute().item()

        valid_values = [v for v in per_mol if v is not None]
        self.assertEqual(len(self.mols), len(per_mol))
        self.assertAlmostEqual(aggregate, np.mean(valid_values), places=4)

    def test_per_molecule_energy_none_for_none_input(self):
        per_mol = paired_eval.per_molecule_energy([self.mols[0], None])
        self.assertIsNone(per_mol[1])

    def test_per_molecule_strain_energy_matches_aggregate(self):
        per_mol = paired_eval.per_molecule_strain_energy(self.mols, per_atom=True)

        metric = metrics.AverageStrainEnergy(per_atom=True)
        metric.update(self.mols)
        aggregate = metric.compute().item()

        valid_values = [v for v in per_mol if v is not None]
        self.assertGreater(len(valid_values), 0)
        self.assertAlmostEqual(aggregate, np.mean(valid_values), places=4)

    def test_per_molecule_opt_rmsd_matches_aggregate(self):
        per_mol = paired_eval.per_molecule_opt_rmsd(self.mols)

        metric = metrics.AverageOptRmsd()
        metric.update(self.mols)
        aggregate = metric.compute().item()

        valid_values = [v for v in per_mol if v is not None]
        self.assertGreater(len(valid_values), 0)
        self.assertAlmostEqual(aggregate, np.mean(valid_values), places=4)
        for value in valid_values:
            self.assertGreaterEqual(value, 0.0)

    def test_per_molecule_posebusters_passes_clean_molecules(self):
        # Ethanol, water and methane are chemically unimpeachable and must all pass. They did not
        # before energy_ratio was excluded: that module scored water and methane as failures and
        # returned NaN for strained cages, so it measured RDKit's conformer-embedding success rather
        # than the molecule's plausibility.
        per_mol = paired_eval.per_molecule_posebusters(self.mols)
        self.assertEqual([True, True, True], per_mol)

    def test_posebusters_config_excludes_energy_ratio(self):
        functions = [module.get("function") for module in paired_eval._posebusters_config()["modules"]]
        self.assertNotIn("energy_ratio", functions)
        # The remaining plausibility checks must survive the filtering
        for expected in ["rdkit_sanity", "atoms_connected", "distance_geometry", "flatness"]:
            self.assertIn(expected, functions)

    def test_per_molecule_posebusters_config_is_not_mutated_between_calls(self):
        before = len(paired_eval._posebusters_config()["modules"])
        paired_eval.per_molecule_posebusters(self.mols)
        self.assertEqual(before, len(paired_eval._posebusters_config()["modules"]))

    def test_per_molecule_posebusters_treats_unevaluated_check_as_skip_not_failure(self):
        report = pd.DataFrame({"check_a": [True, True], "check_b": [np.nan, False]})
        with mock.patch.object(paired_eval.PoseBusters, "bust", return_value=report):
            per_mol = paired_eval.per_molecule_posebusters(self.mols[:2])

        # First molecule passed everything that could be evaluated -> pass, despite the NaN
        self.assertTrue(per_mol[0])
        # Second genuinely failed a check -> fail
        self.assertFalse(per_mol[1])

    def test_per_molecule_posebusters_returns_none_when_nothing_evaluated(self):
        report = pd.DataFrame({"check_a": [np.nan], "check_b": [np.nan]})
        with mock.patch.object(paired_eval.PoseBusters, "bust", return_value=report):
            per_mol = paired_eval.per_molecule_posebusters(self.mols[:1])

        self.assertIsNone(per_mol[0])

    def test_per_molecule_posebusters_preserves_position_of_none(self):
        per_mol = paired_eval.per_molecule_posebusters([self.mols[0], None, self.mols[1]])
        self.assertEqual(3, len(per_mol))
        self.assertIsNone(per_mol[1])
        self.assertTrue(per_mol[0])
        self.assertTrue(per_mol[2])

    def test_per_molecule_posebusters_all_none_input(self):
        per_mol = paired_eval.per_molecule_posebusters([None, None])
        self.assertEqual([None, None], per_mol)

    def test_per_molecule_posebusters_restores_rdkit_log_suppression(self):
        # posebusters.modules.sanity re-enables RDKit's logger globally as a side effect, which
        # otherwise floods the rest of a job's log with RDKit warnings.
        RDLogger.DisableLog("rdApp.*")
        paired_eval.per_molecule_posebusters(self.mols)

        with mock.patch.object(paired_eval.RDLogger, "DisableLog") as disable_log:
            paired_eval.per_molecule_posebusters(self.mols)
        disable_log.assert_called_with("rdApp.*")

    def test_per_molecule_stability_matches_calc_atom_stabilities(self):
        mols_with_none = [self.mols[0], None, self.mols[2]]

        atom_stable_frac, mol_stable = paired_eval.per_molecule_stability(mols_with_none)

        self.assertIsNone(atom_stable_frac[1])
        self.assertIsNone(mol_stable[1])

        for idx, mol in enumerate(mols_with_none):
            if mol is None:
                continue
            expected_stabs = metrics.calc_atom_stabilities(mol)
            self.assertAlmostEqual(sum(expected_stabs) / len(expected_stabs), atom_stable_frac[idx], places=5)
            self.assertEqual(all(expected_stabs), mol_stable[idx])


class TrajectoryStraightnessTests(unittest.TestCase):
    def test_straight_line_trajectory_has_ratio_one(self):
        torch.manual_seed(0)
        n_atoms, n_steps = 4, 10
        start = torch.randn(n_atoms, 3)
        end = torch.randn(n_atoms, 3)

        alphas = torch.linspace(0, 1, n_steps).view(-1, 1, 1)
        trajectory = start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas
        trajectory = trajectory.unsqueeze(0)  # [1, T, N, 3]
        mask = torch.ones(1, n_atoms, dtype=torch.long)

        straightness = paired_eval.per_molecule_trajectory_straightness(trajectory, mask)

        self.assertEqual(1, len(straightness))
        self.assertAlmostEqual(1.0, straightness[0], places=4)

    def test_curved_trajectory_has_ratio_greater_than_one(self):
        n_atoms = 2
        # Out-and-part-way-back path: strictly longer path length than the direct start->end chord
        trajectory = torch.tensor(
            [[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
              [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
              [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
              [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]]]
        )  # [1, T=4, N=2, 3]
        mask = torch.ones(1, n_atoms, dtype=torch.long)

        straightness = paired_eval.per_molecule_trajectory_straightness(trajectory, mask)

        self.assertGreater(straightness[0], 1.0)

    def test_padding_atoms_do_not_affect_result(self):
        torch.manual_seed(1)
        n_real, n_pad, n_steps = 3, 2, 5
        n_total = n_real + n_pad

        real_traj = torch.randn(1, n_steps, n_real, 3)
        pad_traj = torch.randn(1, n_steps, n_pad, 3) * 1000  # garbage, should be ignored
        full_traj = torch.cat([real_traj, pad_traj], dim=2)

        mask = torch.zeros(1, n_total, dtype=torch.long)
        mask[0, :n_real] = 1

        straightness_full = paired_eval.per_molecule_trajectory_straightness(full_traj, mask)
        straightness_real_only = paired_eval.per_molecule_trajectory_straightness(
            real_traj, torch.ones(1, n_real, dtype=torch.long)
        )

        self.assertAlmostEqual(straightness_real_only[0], straightness_full[0], places=5)

    def test_zero_chord_length_returns_none(self):
        trajectory = torch.zeros(1, 5, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.long)

        straightness = paired_eval.per_molecule_trajectory_straightness(trajectory, mask)

        self.assertIsNone(straightness[0])

    def test_mismatched_shapes_raise(self):
        trajectory = torch.zeros(2, 5, 3, 3)
        mask = torch.ones(3, 3, dtype=torch.long)
        with self.assertRaises(ValueError):
            paired_eval.per_molecule_trajectory_straightness(trajectory, mask)


class X1MovementTests(unittest.TestCase):
    def test_constant_prediction_has_zero_movement(self):
        n_atoms, n_steps = 4, 6
        prediction = torch.randn(1, 1, n_atoms, 3).expand(1, n_steps, n_atoms, 3)
        mask = torch.ones(1, n_atoms, dtype=torch.long)

        movement = paired_eval.per_molecule_x1_movement(prediction, mask)

        self.assertEqual(1, len(movement))
        self.assertAlmostEqual(0.0, movement[0], places=5)

    def test_known_per_step_displacement_averages_correctly(self):
        # Single atom, moves by a vector of norm 1 at each of 3 steps -> mean displacement 1.0
        step_vec = torch.tensor([1.0, 0.0, 0.0])
        x1_trajectory = torch.stack([torch.zeros(3), step_vec, step_vec * 2, step_vec * 3])
        x1_trajectory = x1_trajectory.view(1, 4, 1, 3)
        mask = torch.ones(1, 1, dtype=torch.long)

        movement = paired_eval.per_molecule_x1_movement(x1_trajectory, mask)

        self.assertAlmostEqual(1.0, movement[0], places=5)

    def test_padding_atoms_do_not_affect_result(self):
        torch.manual_seed(2)
        n_real, n_pad, n_steps = 3, 2, 5
        n_total = n_real + n_pad

        real_pred = torch.randn(1, n_steps, n_real, 3)
        pad_pred = torch.randn(1, n_steps, n_pad, 3) * 1000
        full_pred = torch.cat([real_pred, pad_pred], dim=2)

        mask = torch.zeros(1, n_total, dtype=torch.long)
        mask[0, :n_real] = 1

        movement_full = paired_eval.per_molecule_x1_movement(full_pred, mask)
        movement_real_only = paired_eval.per_molecule_x1_movement(real_pred, torch.ones(1, n_real, dtype=torch.long))

        self.assertAlmostEqual(movement_real_only[0], movement_full[0], places=5)

    def test_fewer_than_two_steps_returns_none(self):
        prediction = torch.randn(1, 1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.long)

        movement = paired_eval.per_molecule_x1_movement(prediction, mask)

        self.assertIsNone(movement[0])

    def test_mismatched_shapes_raise(self):
        prediction = torch.zeros(2, 5, 3, 3)
        mask = torch.ones(3, 3, dtype=torch.long)
        with self.assertRaises(ValueError):
            paired_eval.per_molecule_x1_movement(prediction, mask)


if __name__ == "__main__":
    unittest.main()
