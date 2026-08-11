import unittest

import numpy as np
import torch
import torch.nn.functional as F

import semlaflow.util.functional as smolF
from semlaflow.models.fm import TARGET_MIN_EPS, apply_plan, permutation_target


def _fixture(sizes=(5, 3, 6), t=0.7, seed=0, vocab_size=4, n_bond_types=3, n_charges=7, sigma=None):
    """Build a (data, interpolated, times, sigma) fixture with a KNOWN optimal permutation.

    As the temperature -> 0 the plan converges to the argmin permutation of the cost, which is only
    the identity if x_t happened to be built that way. So "the soft target must equal the hard
    target" is only a well-posed statement once the argmin is pinned down.

    This builds x_t[i] = t * x1[sigma(i)] exactly, which makes cost[i, sigma(i)] = 0 and
    cost[i, j] = t^2 ||x1[sigma(i)] - x1[j]||^2 > 0 for every other j. So sigma is the argmin, it is
    unique, and the target it implies is a well-defined object to compare against.

    Tensors match what datamodules._batch_to_dict produces: float one-hot atomics, dense symmetric
    one-hot bonds, int64 one-hot charges, and zeroed padding rows.
    """

    torch.manual_seed(seed)
    batch_size = len(sizes)
    n = max(sizes)
    seq_lengths = torch.tensor(sizes)
    mask = (torch.arange(n).unsqueeze(0) < seq_lengths.unsqueeze(1)).long()
    mask_f = mask.float()

    coords = torch.randn((batch_size, n, 3)) * mask_f.unsqueeze(2)

    atomics = F.one_hot(torch.randint(0, vocab_size, (batch_size, n)), vocab_size).float()
    atomics = atomics * mask_f.unsqueeze(2)

    raw_bonds = torch.randint(0, n_bond_types, (batch_size, n, n))
    raw_bonds = torch.minimum(raw_bonds, raw_bonds.transpose(1, 2))
    bonds = F.one_hot(raw_bonds, n_bond_types).float()
    edge_mask = (mask_f.unsqueeze(2) * mask_f.unsqueeze(1)).unsqueeze(-1)
    bonds = bonds * edge_mask

    charges = F.one_hot(torch.randint(0, n_charges, (batch_size, n)), n_charges).long()
    charges = charges * mask.unsqueeze(2)

    # A per-molecule permutation of the real atoms only, identity on padding
    if sigma is None:
        sigma = torch.arange(n).unsqueeze(0).repeat(batch_size, 1)
        for b, n_b in enumerate(sizes):
            sigma[b, :n_b] = torch.randperm(n_b)

    data = {"coords": coords, "atomics": atomics, "bonds": bonds, "charges": charges, "mask": mask}

    interp_coords = t * torch.gather(coords, 1, sigma.unsqueeze(2).expand(-1, -1, 3))
    interp_coords = interp_coords * mask_f.unsqueeze(2)
    interpolated = {
        "coords": interp_coords,
        "atomics": atomics.clone(),
        "bonds": bonds.clone(),
        "charges": charges.clone(),
        "mask": mask,
    }

    times = torch.full((batch_size,), float(t))
    return data, interpolated, times, sigma


def _gather_hard(data, sigma):
    """The hard target implied by sigma -- x1 permuted, all four channels jointly."""

    n = sigma.size(1)
    coords = torch.gather(data["coords"], 1, sigma.unsqueeze(2).expand(-1, -1, 3))
    atomics = torch.gather(data["atomics"], 1, sigma.unsqueeze(2).expand(-1, -1, data["atomics"].size(-1)))
    charges = torch.gather(data["charges"], 1, sigma.unsqueeze(2).expand(-1, -1, data["charges"].size(-1)))

    bonds = torch.stack([data["bonds"][b][sigma[b]][:, sigma[b]] for b in range(sigma.size(0))])
    return {"coords": coords, "atomics": atomics, "bonds": bonds, "charges": charges.float()}


class PermutationTargetTests(unittest.TestCase):
    def test_hard_target_returns_the_data_object(self):
        data, interpolated, times, _ = _fixture()

        target, diagnostics = permutation_target(data, interpolated, times, "hard")

        self.assertIs(target, data)
        self.assertEqual(diagnostics, {})

    def test_rejects_unknown_target(self):
        data, interpolated, times, _ = _fixture()
        with self.assertRaises(ValueError):
            permutation_target(data, interpolated, times, "not-a-real-target")

    def test_zero_temperature_soft_target_equals_hard_target(self):
        data, interpolated, times, sigma = _fixture(t=0.7, seed=1)
        eps = torch.full((3,), 1e-3)

        target, _ = permutation_target(
            data, interpolated, times, "sinkhorn", sinkhorn_iters=2000, eps_override=eps
        )
        expected = _gather_hard(data, sigma)

        for key in ["coords", "atomics", "bonds", "charges"]:
            np.testing.assert_almost_equal(
                target[key].numpy(), expected[key].numpy(), decimal=4, err_msg=f"channel {key}"
            )

    def test_zero_temperature_plan_recovers_the_known_permutation(self):
        data, interpolated, times, sigma = _fixture(t=0.7, seed=2)
        eps = torch.full((3,), 1e-3)

        target, _ = permutation_target(
            data, interpolated, times, "sinkhorn", sinkhorn_iters=2000, eps_override=eps
        )

        # Recover the plan implicitly: the coord target must be x1 gathered by sigma
        expected = _gather_hard(data, sigma)
        np.testing.assert_almost_equal(target["coords"].numpy(), expected["coords"].numpy(), decimal=4)

    def test_target_rows_are_a_proper_convex_combination(self):
        # 10 iterations is deliberately unconverged -- the convex-combination property must hold at
        # any iteration count, otherwise n_iters becomes an undocumented knob controlling a
        # systematic target-magnitude bias
        data, interpolated, times, _ = _fixture(t=0.3, seed=3)

        target, _ = permutation_target(data, interpolated, times, "sinkhorn", sinkhorn_iters=10)

        mask = data["mask"]
        for b in range(mask.size(0)):
            n_b = int(mask[b].sum().item())

            self.assertTrue((target["atomics"][b, :n_b] >= 0).all().item())
            np.testing.assert_almost_equal(
                target["atomics"][b, :n_b].sum(dim=-1).numpy(),
                np.ones(n_b),
                decimal=5,
            )

            lo = data["coords"][b, :n_b].min(dim=0).values
            hi = data["coords"][b, :n_b].max(dim=0).values
            self.assertTrue((target["coords"][b, :n_b] >= lo - 1e-5).all().item())
            self.assertTrue((target["coords"][b, :n_b] <= hi + 1e-5).all().item())

    def test_padding_rows_are_untouched(self):
        data, interpolated, times, _ = _fixture(sizes=(5, 3, 6), t=0.4, seed=4)

        target, _ = permutation_target(data, interpolated, times, "sinkhorn", sinkhorn_iters=50)

        mask = data["mask"]
        for b in range(mask.size(0)):
            n_b = int(mask[b].sum().item())
            for key in ["coords", "atomics", "bonds"]:
                self.assertTrue(torch.equal(target[key][b, n_b:], data[key][b, n_b:].float()))

    def test_blending_vanishes_as_t_approaches_one(self):
        # Identity sigma, which is the real training situation: the coupling permutes the PRIOR, so
        # x_t is built from x1 in its original index order. "Blending vanishes" then means the
        # target converges back onto x1 itself, which is what target-coord-shift measures.
        # noise_std=0 so the limit is exact rather than floored by the coord noise.
        entropies = []
        shifts = []
        for t in [0.1, 0.5, 0.9, 0.99]:
            identity = torch.arange(6).unsqueeze(0).repeat(3, 1)
            data, interpolated, times, _ = _fixture(t=t, seed=5, sigma=identity)
            target, diag = permutation_target(
                data, interpolated, times, "sinkhorn", noise_std=0.0, sinkhorn_iters=200
            )
            entropies.append(diag["target-plan-entropy"].item())
            shifts.append(diag["target-coord-shift"].item())

        for earlier, later in zip(entropies, entropies[1:]):
            self.assertLessEqual(later, earlier + 1e-6)

        self.assertLess(shifts[-1], 1e-3)

    def test_blending_is_near_zero_at_t_one_with_shipped_coord_noise(self):
        # With the shipped sigma=0.2 the temperature floors at 2*sigma^2 = 0.08 rather than
        # reaching 0, so the t -> 1 limit is approximate. It still holds because the cost gap
        # between distinct atoms swamps the floor.
        identity = torch.arange(6).unsqueeze(0).repeat(3, 1)
        data, interpolated, times, sigma = _fixture(t=1.0, seed=6, sigma=identity)

        target, diag = permutation_target(
            data, interpolated, times, "sinkhorn", noise_std=0.2, sinkhorn_iters=200
        )

        self.assertGreater(diag["target-eps"].item(), 0.0)
        self.assertLess(diag["target-coord-shift"].item(), 1e-2)
        np.testing.assert_almost_equal(
            target["coords"].numpy(), _gather_hard(data, sigma)["coords"].numpy(), decimal=2
        )

    def test_target_becomes_diffuse_as_t_approaches_zero(self):
        # The mirror image of the t -> 1 test. Documented explicitly so nobody later "fixes" it.
        data, interpolated, times, _ = _fixture(t=1e-3, seed=7)

        _, diag = permutation_target(
            data, interpolated, times, "sinkhorn", noise_std=0.0, sinkhorn_iters=200
        )

        self.assertGreater(diag["target-plan-entropy"].item(), 0.9)
        self.assertGreater(diag["target-eff-atoms"].item(), 2.5)

    def test_eps_schedule_is_twice_the_conditional_variance(self):
        for t, sigma_val in [(0.0, 0.0), (0.5, 0.0), (0.9, 0.2), (1.0, 0.2)]:
            data, interpolated, times, _ = _fixture(t=t, seed=8)
            _, diag = permutation_target(
                data, interpolated, times, "sinkhorn", noise_std=sigma_val, sinkhorn_iters=5
            )
            expected = 2.0 * ((1.0 - t) ** 2 + sigma_val ** 2)
            self.assertAlmostEqual(diag["target-eps"].item(), expected, places=5)

    def test_hard_fallback_when_temperature_underflows(self):
        data, interpolated, times, _ = _fixture(t=1.0, seed=9)
        eps = torch.full((3,), TARGET_MIN_EPS / 10.0)

        target, diag = permutation_target(
            data, interpolated, times, "sinkhorn", eps_override=eps
        )

        self.assertIs(target, data)
        self.assertEqual(diag["target-hard-fallback-frac"].item(), 1.0)

    def test_no_nan_or_inf_at_extreme_times(self):
        for t in [0.0, 1e-8, 1.0 - 1e-8, 1.0]:
            for sigma_val in [0.0, 0.2]:
                data, interpolated, times, _ = _fixture(sizes=(5, 1, 6), t=t, seed=10)
                target, _ = permutation_target(
                    data, interpolated, times, "sinkhorn", noise_std=sigma_val, sinkhorn_iters=100
                )
                for key in ["coords", "atomics", "bonds", "charges"]:
                    tensor = target[key]
                    self.assertFalse(
                        torch.isnan(tensor).any().item(), f"nan in {key} at t={t}, sigma={sigma_val}"
                    )
                    self.assertFalse(
                        torch.isinf(tensor).any().item(), f"inf in {key} at t={t}, sigma={sigma_val}"
                    )


class McmcTargetTests(unittest.TestCase):
    def test_target_is_a_genuine_permutation_of_the_data(self):
        """Unlike sinkhorn, the mcmc target must be a hard permutation -- one sample, not a mean.

        So every channel must be a rearrangement of x1's, with the multisets preserved and the
        discrete channels still exactly one-hot.
        """

        torch.manual_seed(0)
        data, interpolated, times, _ = _fixture(sizes=(5, 3, 6), t=0.3, seed=20)

        target, _ = permutation_target(data, interpolated, times, "mcmc", mcmc_iters=50)

        mask = data["mask"]
        for b in range(mask.size(0)):
            n_b = int(mask[b].sum().item())

            got = target["coords"][b, :n_b].sort(dim=0).values
            want = data["coords"][b, :n_b].sort(dim=0).values
            np.testing.assert_almost_equal(got.numpy(), want.numpy(), decimal=5)

            for key in ["atomics", "charges"]:
                rows = target[key][b, :n_b]
                self.assertTrue(((rows == 0) | (rows == 1)).all().item(), f"{key} not one-hot")
                self.assertTrue((rows.sum(dim=-1) == 1).all().item(), f"{key} rows not one-hot")
                got = target[key][b, :n_b].sum(dim=0)
                want = data[key][b, :n_b].to(got.dtype).sum(dim=0)
                np.testing.assert_almost_equal(got.numpy(), want.numpy(), decimal=5)

    def test_bonds_stay_one_hot_and_symmetric(self):
        torch.manual_seed(1)
        data, interpolated, times, _ = _fixture(t=0.3, seed=21)

        target, _ = permutation_target(data, interpolated, times, "mcmc", mcmc_iters=50)

        bonds = target["bonds"]
        self.assertTrue(((bonds == 0) | (bonds == 1)).all().item())
        np.testing.assert_almost_equal(bonds.numpy(), bonds.transpose(1, 2).numpy(), decimal=6)

    def test_tiny_temperature_stays_at_the_identity(self):
        """The chain starts at the identity, which is already optimal when x_t = t*x1.

        Mirrors tests/functional.py's tiny-eps mcmc test, but through the target path: at a
        temperature that accepts no uphill move, the target must be x1 untouched.
        """

        torch.manual_seed(2)
        identity = torch.arange(6).unsqueeze(0).repeat(3, 1)
        data, interpolated, times, _ = _fixture(t=0.9, seed=22, sigma=identity)
        eps = torch.full((3,), 1e-4)

        target, diag = permutation_target(
            data, interpolated, times, "mcmc", mcmc_iters=100, eps_override=eps
        )

        self.assertTrue(torch.equal(target["coords"], data["coords"]))
        self.assertAlmostEqual(diag["target-move-frac"].item(), 0.0, places=6)

    def test_move_fraction_is_reported_and_grows_as_t_falls(self):
        """The diagnostic that decides whether this arm means anything.

        At low t the posterior is near-uniform so the chain should wander; at high t it should sit
        still. If move-frac were ~0 everywhere the arm would be a hard arm in disguise.
        """

        fracs = []
        for t in [0.05, 0.95]:
            torch.manual_seed(3)
            identity = torch.arange(6).unsqueeze(0).repeat(3, 1)
            data, interpolated, times, _ = _fixture(t=t, seed=23, sigma=identity)
            _, diag = permutation_target(
                data, interpolated, times, "mcmc", noise_std=0.0, mcmc_iters=100
            )
            fracs.append(diag["target-move-frac"].item())

        self.assertGreater(fracs[0], fracs[1])
        self.assertGreater(fracs[0], 0.0)

    def test_padding_rows_are_untouched(self):
        torch.manual_seed(4)
        data, interpolated, times, _ = _fixture(sizes=(5, 3, 6), t=0.4, seed=24)

        target, _ = permutation_target(data, interpolated, times, "mcmc", mcmc_iters=50)

        mask = data["mask"]
        for b in range(mask.size(0)):
            n_b = int(mask[b].sum().item())
            for key in ["coords", "atomics", "bonds"]:
                self.assertTrue(torch.equal(target[key][b, n_b:], data[key][b, n_b:].float()))

    def test_hard_fallback_when_temperature_underflows(self):
        data, interpolated, times, _ = _fixture(t=1.0, seed=25)
        eps = torch.full((3,), TARGET_MIN_EPS / 10.0)

        target, diag = permutation_target(data, interpolated, times, "mcmc", eps_override=eps)

        self.assertIs(target, data)
        self.assertEqual(diag["target-hard-fallback-frac"].item(), 1.0)

    def test_uniform_proposal_also_works(self):
        torch.manual_seed(5)
        data, interpolated, times, _ = _fixture(t=0.3, seed=26)

        target, _ = permutation_target(
            data, interpolated, times, "mcmc", mcmc_iters=50, mcmc_proposal="uniform"
        )

        mask = data["mask"]
        for b in range(mask.size(0)):
            n_b = int(mask[b].sum().item())
            got = target["coords"][b, :n_b].sort(dim=0).values
            want = data["coords"][b, :n_b].sort(dim=0).values
            np.testing.assert_almost_equal(got.numpy(), want.numpy(), decimal=5)

    def test_no_nan_or_inf_at_extreme_times(self):
        for t in [0.0, 1e-8, 1.0 - 1e-8, 1.0]:
            for sigma_val in [0.0, 0.2]:
                torch.manual_seed(6)
                data, interpolated, times, _ = _fixture(sizes=(5, 1, 6), t=t, seed=27)
                target, _ = permutation_target(
                    data, interpolated, times, "mcmc", noise_std=sigma_val, mcmc_iters=50
                )
                for key in ["coords", "atomics", "bonds", "charges"]:
                    tensor = target[key]
                    self.assertFalse(
                        torch.isnan(tensor).any().item(), f"nan in {key} at t={t}, sigma={sigma_val}"
                    )
                    self.assertFalse(
                        torch.isinf(tensor).any().item(), f"inf in {key} at t={t}, sigma={sigma_val}"
                    )


class ApplyPlanTests(unittest.TestCase):
    def _plan_from_perm(self, sigma, n):
        return torch.stack([F.one_hot(sigma[b], n).float() for b in range(sigma.size(0))])

    def test_hard_permutation_plan_matches_a_gather(self):
        data, _, _, sigma = _fixture(sizes=(5, 5), seed=11)
        plan = self._plan_from_perm(sigma, sigma.size(1))

        target = apply_plan(plan, data)
        expected = _gather_hard(data, sigma)

        for key in ["coords", "atomics", "bonds", "charges"]:
            np.testing.assert_almost_equal(
                target[key].numpy(), expected[key].numpy(), decimal=5, err_msg=f"channel {key}"
            )

    def _soft_plan(self, sizes, seed):
        torch.manual_seed(seed)
        n = max(sizes)
        mask = (torch.arange(n).unsqueeze(0) < torch.tensor(sizes).unsqueeze(1)).long()
        cost = torch.rand((len(sizes), n, n))
        raw = smolF.sinkhorn_batched(cost, mask, torch.full((len(sizes),), 0.5), n_iters=200)
        plan, _ = smolF.plan_from_sinkhorn(raw, mask)
        return plan

    def test_bond_target_diagonal_uses_the_single_index_marginal(self):
        """The self-bond row is a single-index quantity, so its exact marginal is sum_j P_ij B_jj.

        The naive (P B P^T)_ii = sum_{j,k} P_ij P_ik B_jk treats sigma(i) as two independent draws
        and smears real bond types onto the self-bond entry, which _bond_loss does train on.
        """

        sizes = (5, 5)
        data, _, _, _ = _fixture(sizes=sizes, seed=12)
        plan = self._soft_plan(sizes, seed=12)

        target = apply_plan(plan, data)
        actual = target["bonds"].diagonal(dim1=1, dim2=2).transpose(1, 2)

        data_diag = data["bonds"].diagonal(dim1=1, dim2=2).transpose(1, 2)
        expected = torch.einsum("bij,bjc->bic", plan, data_diag)
        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), decimal=5)

        # And it is genuinely different from what the naive pairwise form would have produced
        naive = torch.einsum("bij,bjkc->bikc", plan, data["bonds"])
        naive = torch.einsum("blk,bikc->bilc", plan, naive)
        naive_diag = naive.diagonal(dim1=1, dim2=2).transpose(1, 2)
        self.assertGreater((naive_diag - expected).abs().max().item(), 1e-3)

    def test_bond_target_diagonal_is_a_gather_for_a_hard_plan(self):
        data, _, _, sigma = _fixture(sizes=(5, 5), seed=15)
        plan = self._plan_from_perm(sigma, sigma.size(1))

        target = apply_plan(plan, data)

        actual = target["bonds"].diagonal(dim1=1, dim2=2)
        expected = _gather_hard(data, sigma)["bonds"].diagonal(dim1=1, dim2=2)
        np.testing.assert_almost_equal(actual.numpy(), expected.numpy(), decimal=5)

    def test_bond_target_stays_symmetric(self):
        data, interpolated, times, _ = _fixture(t=0.3, seed=13)
        target, _ = permutation_target(data, interpolated, times, "sinkhorn", sinkhorn_iters=50)

        bonds = target["bonds"]
        np.testing.assert_almost_equal(
            bonds.numpy(), bonds.transpose(1, 2).numpy(), decimal=5
        )

    def test_mask_is_passed_through_unchanged(self):
        data, _, _, sigma = _fixture(sizes=(5, 3), seed=14)
        plan = self._plan_from_perm(sigma, sigma.size(1))

        target = apply_plan(plan, data)

        self.assertIs(target["mask"], data["mask"])


class SoftCrossEntropyTests(unittest.TestCase):
    def test_soft_ce_on_one_hot_equals_hard_ce(self):
        """The soft-label branch must reduce to the hard branch on hard labels.

        Guards both the PyTorch class-probabilities API and the claim that target=sinkhorn only
        differs from target=hard through the target itself.
        """

        torch.manual_seed(0)
        logits = torch.randn((32, 7))
        indices = torch.randint(0, 7, (32,))
        one_hot = F.one_hot(indices, 7).float()

        hard = F.cross_entropy(logits, indices, reduction="none")
        soft = F.cross_entropy(logits, one_hot, reduction="none")

        self.assertEqual(hard.shape, soft.shape)
        self.assertTrue(torch.allclose(hard, soft, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
