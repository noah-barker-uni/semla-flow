import unittest

import torch

from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler


def _build_interpolant(coupling="none", kabsch_align=False, batch_ot=False, fixed_time=None, **kwargs):
    prior_sampler = GeometricNoiseSampler(vocab_size=6, n_bond_types=4, zero_com=True)
    return GeometricInterpolant(
        prior_sampler,
        coupling=coupling,
        kabsch_align=kabsch_align,
        batch_ot=batch_ot,
        fixed_time=fixed_time,
        **kwargs,
    )


def _sample_to_mols(n_mols=3, n_atoms=5, seed=0):
    torch.manual_seed(seed)
    sampler = GeometricNoiseSampler(vocab_size=6, n_bond_types=4, zero_com=True)
    return [sampler.sample_molecule(n_atoms) for _ in range(n_mols)]


def _sample_to_mols_varied(sizes, seed=0):
    torch.manual_seed(seed)
    sampler = GeometricNoiseSampler(vocab_size=6, n_bond_types=4, zero_com=True)
    return [sampler.sample_molecule(n_atoms) for n_atoms in sizes]


class GeometricInterpolantCouplingTests(unittest.TestCase):
    def test_rejects_unknown_coupling(self):
        with self.assertRaises(ValueError):
            _build_interpolant(coupling="not-a-real-coupling")

    def test_none_coupling_only_truncates(self):
        interpolant = _build_interpolant(coupling="none", kabsch_align=True, fixed_time=0.5)
        to_mols = _sample_to_mols()

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)

    def test_hungarian_coupling_with_kabsch_runs(self):
        interpolant = _build_interpolant(coupling="hungarian", kabsch_align=True, fixed_time=0.5)
        to_mols = _sample_to_mols()

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)

    def test_sinkhorn_coupling_requires_t_and_runs(self):
        interpolant = _build_interpolant(coupling="sinkhorn", kabsch_align=True, fixed_time=0.7)
        to_mols = _sample_to_mols()

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)
            self.assertFalse(torch.isnan(from_mol.coords).any().item())

    def test_sinkhorn_coupling_with_batch_ot_raises(self):
        with self.assertRaises(ValueError):
            _build_interpolant(coupling="sinkhorn", batch_ot=True)

    def test_sinkhorn_coupling_handles_varied_molecule_sizes(self):
        # Exercises the padded/masked batching in _sinkhorn_couple, not just the same-size case
        interpolant = _build_interpolant(coupling="sinkhorn", kabsch_align=True, fixed_time=0.5)
        to_mols = _sample_to_mols_varied([4, 6, 5])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)
            self.assertFalse(torch.isnan(from_mol.coords).any().item())

    def test_sinkhorn_coupling_without_kabsch_runs(self):
        interpolant = _build_interpolant(coupling="sinkhorn", kabsch_align=False, fixed_time=0.5)
        to_mols = _sample_to_mols_varied([3, 5])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))

    def test_mcmc_coupling_with_batch_ot_raises(self):
        with self.assertRaises(ValueError):
            _build_interpolant(coupling="mcmc", batch_ot=True)

    def test_rejects_unknown_mcmc_proposal(self):
        with self.assertRaises(ValueError):
            _build_interpolant(coupling="mcmc", mcmc_proposal="not-a-real-proposal")

    def test_mcmc_coupling_uniform_proposal_runs(self):
        interpolant = _build_interpolant(coupling="mcmc", kabsch_align=True, fixed_time=0.5, mcmc_proposal="uniform")
        to_mols = _sample_to_mols_varied([4, 6, 5])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)
            self.assertFalse(torch.isnan(from_mol.coords).any().item())

    def test_mcmc_coupling_knn_proposal_runs(self):
        interpolant = _build_interpolant(
            coupling="mcmc", kabsch_align=True, fixed_time=0.5, mcmc_proposal="knn", mcmc_knn_k=3
        )
        to_mols = _sample_to_mols_varied([4, 6, 5])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)
            self.assertFalse(torch.isnan(from_mol.coords).any().item())

    def test_mcmc_coupling_handles_tiny_molecules_with_large_knn_k(self):
        interpolant = _build_interpolant(coupling="mcmc", mcmc_proposal="knn", mcmc_knn_k=50, fixed_time=0.5)
        to_mols = _sample_to_mols_varied([1, 2, 6])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)

    def test_mcmc_coupling_without_kabsch_runs(self):
        interpolant = _build_interpolant(coupling="mcmc", kabsch_align=False, fixed_time=0.5)
        to_mols = _sample_to_mols_varied([3, 5])

        from_mols, returned_to_mols, interp_mols, times = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))


if __name__ == "__main__":
    unittest.main()
