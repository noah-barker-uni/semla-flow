import unittest

import torch

import semlaflow.util.functional as smolF
from semlaflow.data.interpolate import GeometricNoiseSampler
from semlaflow.util.molrepr import GeometricMol


def _sample_mol(n_atoms=5, vocab_size=6, n_bond_types=4, seed=0, zero_com=False):
    torch.manual_seed(seed)
    sampler = GeometricNoiseSampler(vocab_size, n_bond_types, zero_com=zero_com)
    return sampler.sample_molecule(n_atoms)


class GeometricMolSoftPermuteTests(unittest.TestCase):
    def test_soft_permute_identity_is_noop(self):
        mol = _sample_mol()
        identity = torch.eye(mol.seq_length)

        permuted = mol.soft_permute(identity)

        self.assertTrue(torch.allclose(mol.coords, permuted.coords, atol=1e-5))
        self.assertTrue(torch.allclose(mol.atomics, permuted.atomics, atol=1e-5))
        self.assertTrue(torch.allclose(mol.adjacency, permuted.adjacency, atol=1e-5))

    def test_soft_permute_with_hard_permutation_matrix_matches_permute(self):
        mol = _sample_mol(n_atoms=5)
        indices = [2, 0, 3, 1, 4]

        hard = mol.permute(indices)

        P = torch.zeros((5, 5))
        for i, j in enumerate(indices):
            P[i, j] = 1.0
        soft = mol.soft_permute(P)

        self.assertTrue(torch.allclose(hard.coords, soft.coords, atol=1e-5))
        self.assertTrue(torch.allclose(hard.atomics, soft.atomics, atol=1e-5))

        # Bond content check independent of GeometricMol.permute()'s own (differently-ordered)
        # bond relabeling: new_raw[i, i'] should be exactly raw[indices[i], indices[i']].
        idx = torch.tensor(indices)
        raw = mol.bond_types.view(5, 5, -1)
        expected_raw = raw[idx][:, idx]
        soft_raw = soft.bond_types.view(5, 5, -1)
        self.assertTrue(torch.allclose(expected_raw, soft_raw, atol=1e-5))

    def test_soft_permute_produces_convex_combination_of_coords(self):
        mol = _sample_mol(n_atoms=4, seed=7)
        cost_matrix = smolF.inter_distances(torch.rand((4, 3)), mol.coords, sqrd=True)
        P = smolF.sinkhorn(cost_matrix, eps=0.5, n_iters=100)

        soft = mol.soft_permute(P)
        expected_coords = P @ mol.coords

        self.assertTrue(torch.allclose(expected_coords, soft.coords, atol=1e-5))

    def test_soft_permute_preserves_center_of_mass(self):
        mol = _sample_mol(n_atoms=6, seed=11, zero_com=False)
        cost_matrix = smolF.inter_distances(torch.rand((6, 3)), mol.coords, sqrd=True)
        P = smolF.sinkhorn(cost_matrix, eps=0.3, n_iters=200)

        soft = mol.soft_permute(P)

        self.assertTrue(torch.allclose(mol.com, soft.com, atol=1e-3))

    def test_soft_permute_rejects_wrong_shaped_matrix(self):
        mol = _sample_mol(n_atoms=4)
        with self.assertRaises(ValueError):
            mol.soft_permute(torch.eye(3))

    def test_soft_permute_rejects_class_index_atomics(self):
        coords = torch.rand((3, 3))
        atomics = torch.tensor([0, 1, 0])
        bond_indices = torch.ones((3, 3)).nonzero()
        bond_types = smolF.one_hot_encode_tensor(torch.zeros(9, dtype=torch.long), 2)
        mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)

        with self.assertRaises(ValueError):
            mol.soft_permute(torch.eye(3))

    def test_soft_permute_rejects_non_dense_bonds(self):
        coords = torch.rand((3, 3))
        atomics = smolF.one_hot_encode_tensor(torch.zeros(3, dtype=torch.long), 2)
        bond_indices = torch.tensor([[0, 1], [1, 0]])
        bond_types = smolF.one_hot_encode_tensor(torch.zeros(2, dtype=torch.long), 2)
        mol = GeometricMol(coords, atomics, bond_indices=bond_indices, bond_types=bond_types)

        with self.assertRaises(ValueError):
            mol.soft_permute(torch.eye(3))


if __name__ == "__main__":
    unittest.main()
