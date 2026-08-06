import unittest

import numpy as np
import torch
from rdkit import Chem
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


if __name__ == "__main__":
    unittest.main()
