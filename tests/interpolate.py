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

        from_mols, returned_to_mols, interp_mols, times, _ = interpolant.interpolate(to_mols)

        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)

    def test_hungarian_coupling_with_kabsch_runs(self):
        interpolant = _build_interpolant(coupling="hungarian", kabsch_align=True, fixed_time=0.5)
        to_mols = _sample_to_mols()

        from_mols, returned_to_mols, interp_mols, times, _ = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)

    def test_hungarian_coupling_handles_varied_molecule_sizes(self):
        interpolant = _build_interpolant(coupling="hungarian", kabsch_align=True, fixed_time=0.5)
        to_mols = _sample_to_mols_varied([4, 6, 5])

        from_mols, returned_to_mols, interp_mols, times, _ = interpolant.interpolate(to_mols)

        self.assertEqual(len(from_mols), len(to_mols))
        for from_mol, to_mol in zip(from_mols, returned_to_mols):
            self.assertEqual(from_mol.seq_length, to_mol.seq_length)
            self.assertFalse(torch.isnan(from_mol.coords).any().item())

    def test_rejects_soft_couplings(self):
        # Soft (doubly stochastic) plans are not group elements, so applying one to the prior
        # contracts the noise and breaks the marginal -- they belong on the target axis instead.
        # See docs/Sinkhorn_corrections.md.
        for coupling in ["sinkhorn", "mcmc"]:
            with self.assertRaises(ValueError):
                _build_interpolant(coupling=coupling)


if __name__ == "__main__":
    unittest.main()
