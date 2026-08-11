from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

import semlaflow.util.functional as smolF
import semlaflow.util.metrics as Metrics
import semlaflow.util.rdkit as smolRD
from semlaflow.models.semla import MolecularGenerator
from semlaflow.util.molrepr import GeometricMol
from semlaflow.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]

# "sinkhorn" is the mean-field deterministic average of the posterior over permutations; "mcmc" is
# an unbiased single sample from the same posterior. Single-sample is legitimate because the loss
# is linear in the target, so the expected gradient depends on the target only through its mean.
TARGET_TYPES = ["hard", "sinkhorn", "mcmc"]

# Number of t buckets the plan-entropy-vs-t curve is accumulated into over an epoch.
TARGET_T_BINS = 10

# Temperature below which the entropic plan is replaced by its exact zero-temperature limit.
#
# This is NOT the removed COUPLING_MIN_EPS clamp (see CLAUDE.md, defect 4). That clamp
# floored eps and then kept blending at the floored value, so the intended t -> 1 sharpening
# silently never happened. Here a floored eps exists only to keep sinkhorn_batched's eps > 0 check
# legal, and the resulting plan is THROWN AWAY and replaced by the identity -- which is the exact
# eps -> 0 limit of the plan, because as t -> 1 we have x_t -> x1 and the argmin permutation of the
# cost is then the identity (atoms of a real molecule are never coincident). So the schedule is
# exact everywhere it is used, and exactly right where it is not.
#
# The value is set by float32 conditioning, not by taste. The solver forms (f_i + g_j - c_ij) / eps
# with f, g = O(max cost) ~ O(1e2), so that exponent carries absolute error ~ 2^-24 * 1e2 = 1e-5,
# and exp() overflows once 1e-5 / eps > 88, ie around eps ~ 1e-7. 1e-5 leaves two orders of
# margin. Under the default schedule (which includes the coord noise floor) it never fires at all;
# train-target-hard-fallback-frac is logged so that stays visible rather than assumed.
TARGET_MIN_EPS = 1e-5


@torch.no_grad()
def permutation_target(
    data: _BatchT,
    interpolated: _BatchT,
    times: _T,
    target: str,
    noise_std: float = 0.0,
    sinkhorn_iters: int = 100,
    mcmc_iters: int = 100,
    mcmc_proposal: str = "knn",
    mcmc_knn_k: int = 8,
    eps_override: Optional[_T] = None,
) -> tuple[_BatchT, dict[str, _T]]:
    """Build the regression target the model is trained against, given the current state x_t.

    The hard target is x1 itself. The soft target is the posterior mean over permutations,

        w(pi') ~ exp( -||x_t - t * pi'(x1)||^2 / (2 * var) ),    var = Var(x_t | x1, t)

    estimated either by sinkhorn (the mean-field deterministic average) or by mcmc (an unbiased
    single sample). This is legitimate precisely where soft-permuting the PRIOR was not: the model
    is x1-parameterised, so the Bayes-optimal target is E[x1 | x_t], a posterior mean. See
    docs/Sinkhorn_corrections.md.

    A single mcmc sample is enough because the loss is linear in the target, so its expected
    gradient depends on the target only through its mean. Note the two estimators are not
    interchangeable at low t: the chain starts at the identity and only explores a neighbourhood of
    it, so the mcmc target stays closer to the hard one than the sinkhorn average does. Whether
    that neighbourhood is big enough for the arm to mean anything is what the move-fraction
    diagnostic is for.

    The temperature is not a free hyperparameter -- it is the conditional path's own variance. With
    x_t = (1-t) x0 + t x1 + noise_std * z, that is var = (1-t)^2 + noise_std^2, and since sinkhorn
    is parameterised by P ~ exp(-cost / eps), eps = 2 * var.

    Two documented approximations:
      - The likelihood uses coordinates only, though x_t's categorical channels are also informative
        about pi' under unmask interpolation. This biases the plan more diffuse. The hungarian
        coupling is coords-only for the same reason.
      - The posterior is exact only under coupling="none". Under a coupling, x0 was selected to be
        close to x1, so the true conditional residual is smaller than var predicts and the analytic
        temperature is too large -- again more diffuse. Measured by the eps-ratio diagnostic rather
        than corrected.

    Args:
        data (_BatchT): The data molecules x1, in their original (un-permuted) atom order.
        interpolated (_BatchT): The interpolated state x_t.
        times (_T): Interpolation times, shape [B].
        target (str): One of TARGET_TYPES.
        noise_std (float): The interpolant's coord_noise_std, part of the conditional variance.
        sinkhorn_iters (int): Solver iterations, target="sinkhorn" only.
        mcmc_iters (int): Metropolis iterations, target="mcmc" only.
        mcmc_proposal (str): "uniform" or "knn", target="mcmc" only.
        mcmc_knn_k (int): Neighbours the knn proposal may swap with, target="mcmc" only.
        eps_override (Optional[_T]): Shape [B]. Bypasses the schedule entirely. Exists so tests can
            drive the temperature directly instead of inverting the schedule; not used in training.

    Returns:
        tuple[_BatchT, dict[str, _T]]: (target batch, per-batch diagnostics). For target="hard" the
            data object itself is returned unchanged, with no tensor ops and no RNG consumed.
            target="mcmc" DOES consume global RNG, so its stream diverges from the other two arms.
    """

    if target not in TARGET_TYPES:
        raise ValueError(f"target '{target}' not supported. Accepted values: {TARGET_TYPES}.")

    if target == "hard":
        return data, {}

    mask = data["mask"]
    batch_size, n = mask.shape

    if eps_override is not None:
        eps = eps_override
    else:
        var = (1.0 - times).clamp_min(0.0).pow(2) + (noise_std ** 2)
        eps = 2.0 * var

    eps = eps.to(data["coords"].dtype)
    soft_ok = eps >= TARGET_MIN_EPS

    # Whole batch sits at the hard limit, so the limit IS the answer -- no solver call at all
    if not bool(soft_ok.any()):
        return data, _target_diagnostics(data, data, interpolated, None, times, eps, soft_ok, None)

    # Rows index x_t slots, columns index x1 candidates. That orientation is what makes
    # target = P @ x1 match GeometricMol.soft_permute's convention.
    scaled_data_coords = times.view(-1, 1, 1) * data["coords"]
    cost = smolF.inter_distances(interpolated["coords"], scaled_data_coords, sqrd=True)

    eps_safe = eps.clamp_min(TARGET_MIN_EPS)

    if target == "sinkhorn":
        raw_plan = smolF.sinkhorn_batched(cost, mask, eps_safe, n_iters=sinkhorn_iters)
        plan, row_dev = smolF.plan_from_sinkhorn(raw_plan, mask)

    else:
        # The coupling permuted the PRIOR, leaving x1 in its original index order, so x_t[i] was
        # built from x1[i]: the identity IS the permutation that generated x_t. That makes it a
        # draw from the posterior's own high-probability region -- the right place to start a short
        # chain -- and it removes the per-molecule scipy call that a hungarian init would need,
        # which in the loss would mean a GPU->CPU sync on the critical path.
        init_perm = torch.arange(n, device=cost.device).unsqueeze(0).expand(batch_size, n).contiguous()

        perm = smolF.mcmc_permutation(
            cost,
            mask,
            eps_safe,
            mcmc_iters,
            init_perm=init_perm,
            proposal=mcmc_proposal,
            knn_k=mcmc_knn_k,
            # The row index of cost is a slot of x_t, and a transposition of slots i, i' is only
            # ever accepted when x_t[i] is near x_t[i'], so the knn graph lives in x_t space. Using
            # x1 coords here would be nearly right as t -> 1 and badly wrong at small t.
            to_coords=interpolated["coords"],
        )
        plan = smolF.permutation_to_plan(perm, mask, dtype=cost.dtype)
        row_dev = None

    # Discard any plan computed at a clamped temperature in favour of its exact limit
    eye = torch.eye(n, dtype=plan.dtype, device=plan.device).expand(batch_size, n, n)
    plan = torch.where(soft_ok.view(-1, 1, 1), plan, eye)

    target_batch = apply_plan(plan, data)
    diagnostics = _target_diagnostics(
        target_batch, data, interpolated, plan, times, eps, soft_ok, row_dev
    )
    return target_batch, diagnostics


def apply_plan(plan: _T, data: _BatchT) -> _BatchT:
    """Apply a row-stochastic plan to every channel of a data batch jointly.

    A permutation acts on coordinates, atom types, bond types and charges together, so a soft
    permutation must too. Single-index channels are exact functions of the row marginals; the
    pairwise bond channel additionally needs pairwise marginals, and P B P^T is the mean-field
    approximation to those -- except on the diagonal, where the exact marginal is a point mass and
    is restored below.

    Args:
        plan (_T): Row-stochastic plan, shape [B, N, N].
        data (_BatchT): Batch to permute. Its mask is passed through unchanged.

    Returns:
        _BatchT: Soft-permuted batch. Discrete channels become soft labels.
    """

    coords = plan @ data["coords"]
    atomics = plan @ data["atomics"]
    charges = plan @ data["charges"].to(plan.dtype)

    # Two chained two-operand einsums, never the three-operand form: torch.einsum does not
    # guarantee a contraction order for three operands, and the naive order materialises
    # [B, N, N, N, E] -- several GB at GEOM-Drugs sizes. molrepr.py's soft_permute gets away with
    # the three-operand form only because it runs on one molecule at a time.
    bonds = torch.einsum("bij,bjkc->bikc", plan, data["bonds"])
    bonds = torch.einsum("blk,bikc->bilc", plan, bonds)

    # (P B P^T)_ii = sum_{j,k} P_ij P_ik B_jk treats sigma(i) as two independent draws, but on the
    # diagonal both indices are the SAME draw, so the exact marginal is just sum_j P_ij B_jj -- the
    # self-bond row is a single-index quantity and transforms like any other node feature. Restore
    # it exactly: _bond_loss does train on the diagonal, since adj_from_node_mask is built with
    # self_connect=True.
    diag = torch.einsum("bij,bjc->bic", plan, data["bonds"].diagonal(dim1=1, dim2=2).transpose(1, 2))
    bonds = bonds.diagonal_scatter(diag.transpose(1, 2), dim1=1, dim2=2)

    return {
        "coords": coords,
        "atomics": atomics,
        "bonds": bonds,
        "charges": charges,
        "mask": data["mask"],
    }


def _target_diagnostics(
    target: _BatchT,
    data: _BatchT,
    interpolated: _BatchT,
    plan: Optional[_T],
    times: _T,
    eps: _T,
    soft_ok: _T,
    row_dev: Optional[_T],
) -> dict[str, _T]:
    """Per-step scalars describing how much blending actually happened.

    x_t is built from the coupling's permutation and the posterior is then computed from x_t, so
    the plan is peaked on that permutation by construction. That makes "is anything actually
    blending" an empirical question rather than an assumption, which is what these measure.

    Every value is a scalar except target-plan-entropy-per-sample, which is [B] so the caller can
    accumulate the entropy-vs-t curve.
    """

    mask = data["mask"]
    dtype = data["coords"].dtype
    mask_f = mask.to(dtype)
    n_real = mask_f.sum(dim=1).clamp_min(1.0)

    # How far the analytic temperature is from the residual actually observed. ~1 means the
    # gaussian posterior is well specified; < 1 means the coupling already pulled x0 towards x1, so
    # the analytic variance overstates the real one and the plan is over-diffuse.
    resid = (interpolated["coords"] - times.view(-1, 1, 1) * data["coords"]) * mask_f.unsqueeze(2)
    measured_var = resid.pow(2).sum(dim=(1, 2)) / (3.0 * n_real)

    diagnostics = {
        "target-eps": eps.mean(),
        "target-hard-fallback-frac": (~soft_ok).to(dtype).mean(),
        "target-eps-ratio": (measured_var / (0.5 * eps).clamp_min(1e-12)).mean(),
    }

    if plan is None:
        return diagnostics

    if row_dev is not None:
        diagnostics["target-row-sum-dev"] = row_dev.mean()

    log_plan = plan.clamp_min(1e-12).log()
    row_entropy = -(plan * log_plan).sum(dim=2)
    norm_entropy = (row_entropy * mask_f).sum(dim=1) / (n_real * n_real.clamp_min(2.0).log())
    eff_atoms = (mask_f / plan.pow(2).sum(dim=2).clamp_min(1e-12)).sum(dim=1) / n_real
    plan_diag = (plan.diagonal(dim1=1, dim2=2) * mask_f).sum(dim=1) / n_real

    # For a sampled hard permutation the entropy diagnostics are all degenerate (entropy 0, one
    # effective atom), so the only thing that says whether the chain went anywhere is how many
    # slots moved off the identity it started from. If this sits near 0 the arm is a hard arm in
    # disguise and has to be reported as one, not as a soft-target result.
    diagnostics["target-move-frac"] = 1.0 - plan_diag.mean()

    coord_shift = ((target["coords"] - data["coords"]) * mask_f.unsqueeze(2)).pow(2).sum(dim=(1, 2))
    coord_norm = (data["coords"] * mask_f.unsqueeze(2)).pow(2).sum(dim=(1, 2)).clamp_min(1e-12)

    # CE against a soft label q is H(q) + KL(q||p). H(q) is a target-only constant that inflates
    # the reported type/bond losses without touching gradients, so log it to keep the reported
    # numbers comparable across arms.
    label_entropy = -(target["atomics"] * target["atomics"].clamp_min(1e-12).log()).sum(dim=-1)
    label_entropy = (label_entropy * mask_f).sum(dim=1) / n_real

    diagnostics.update({
        "target-plan-entropy": norm_entropy.mean(),
        "target-eff-atoms": eff_atoms.mean(),
        "target-plan-diag": plan_diag.mean(),
        "target-coord-shift": (coord_shift / coord_norm).sqrt().mean(),
        "target-label-entropy": label_entropy.mean(),
        "target-plan-entropy-per-sample": norm_entropy,
    })
    return diagnostics


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        eps=1e-5
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.eps = eps

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level
        }

    def step(self, curr: _BatchT, predicted: _BatchT, prior: _BatchT, t: _T, step_size: float) -> _BatchT:
        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)

        # *** Coord update step ***
        coord_velocity = (predicted["coords"] - curr["coords"]) / (1 - t.view(-1, 1, 1))
        coord_velocity += (torch.randn_like(coord_velocity) * self.coord_noise_std)
        coords = curr["coords"] + (step_size * coord_velocity)

        # *** Atom type update step ***
        if self.type_strategy == "linear":
            one_hots = torch.eye(vocab_size, device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Dirichlet refers to sampling from a dirichlet dist, not dirichlet FM
        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(predicted["atomics"] + self.eps).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size
            )

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(curr["atomics"], predicted["atomics"], t, step_size)

        # *** Bond update step ***
        if self.type_strategy == "linear":
            one_hots = torch.eye(n_bonds, device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(predicted["bonds"] + self.eps).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(curr["bonds"], predicted["bonds"], t, self.bond_mask_index, step_size)

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(curr["bonds"], predicted["bonds"], t, step_size)

        updated = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
            "mask": curr["mask"]
        }
        return updated

    # TODO test with mask sampling
    def _mask_sampling_step(self, curr_dist, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = (step_size * (1 + (self.cat_noise_level * times)) / (1 - times))
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return smolF.one_hot_encode_tensor(curr, n_categories)

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        # Note that there was a small bug in the original impl here, where mult was defined as:
        # mult = ((1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times))
        # In practice the original code was slightly worse than the fixed vesion, but we leave the correct version below
        mult = ((1 + noise + (noise * (n_categories - 1) * times)) / (1 - times))
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(f"{name}_mask_index must be provided if using the mask sampling strategy.")


class MolBuilder:
    def __init__(self, vocab, n_workers=16):
        self.vocab = vocab
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs) for smi in smiles]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None, sanitise=True):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )

        # mol_dicts = {}
        # for mol_coords, atom_dist, bond_dist, charge_dist in extracted:
        #     mol = {
        #         "coords": mol_coords,
        #         "atomics": atom_dist,
        #         "bonds": bond_dist,
        #         "charges": charge_dist
        #     }
        #     mol_dicts.append(mol)

        self._startup()
        build_fn = partial(self._smol_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(self, coords, atom_dists, bond_dists=None, charge_dists=None, sanitise=True):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = self._mol_extract_charges(charge_dists) if charge_dists is not None else None
        return smolRD.mol_from_atoms(coords.numpy(), tokens, bonds=bonds, charges=charges, sanitise=sanitise)

    def _smol_from_tensors(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.size(0)

        charges = torch.tensor(self._mol_extract_charges(charge_dists))
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]

        mol = GeometricMol(coords, atom_dists, bond_indices, bond_types, charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords,
            atom_dists,
            mask,
            bond_dists=bond_dists,
            charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices, bond_types, n_atoms, symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(self, coords, atom_dists, mask, bond_dists=None, charge_dists=None):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class MolecularCFM(L.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_strategy: str = "linear",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        target: str = "hard",
        target_noise_std: float = 0.0,
        target_sinkhorn_iters: int = 100,
        target_mcmc_iters: int = 100,
        target_mcmc_proposal: str = "knn",
        target_mcmc_knn_k: int = 8,
        **kwargs
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'. "
                             + "Supported are `mse`, `ce` or `mask`.")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'. Supported are `ce` or `mask`.")

        if lr_schedule not in ["constant", "one-cycle"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported. Supported are `constant` or `one-cycle`.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError("total_steps must be provided when using the one-cycle LR scheduler.")

        if distill and (type_strategy == "mask" or bond_strategy == "mask"):
            raise ValueError("Distilled training with masking strategy is not supported.")

        if target not in TARGET_TYPES:
            raise ValueError(f"target '{target}' not supported. Accepted values: {TARGET_TYPES}.")

        # _distill_loss builds its own targets and uses KL rather than the shared CE path, so a
        # soft target would only apply to half of it
        if distill and target != "hard":
            raise ValueError("Distilled training is only supported with target='hard'.")

        if lr_schedule == "one-cycle" and warm_up_steps is not None:
            print("Note: warm_up_steps is currently ignored if schedule is one-cycle")

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.target = target
        self.target_noise_std = target_noise_std
        self.target_sinkhorn_iters = target_sinkhorn_iters
        self.target_mcmc_iters = target_mcmc_iters
        self.target_mcmc_proposal = target_mcmc_proposal
        self.target_mcmc_knn_k = target_mcmc_knn_k

        # persistent=False is load-bearing: these must never enter state_dict, or a checkpoint
        # written by a soft arm stops loading in an arm that does not define them
        self.register_buffer("_plan_entropy_sum", torch.zeros(TARGET_T_BINS), persistent=False)
        self.register_buffer("_plan_entropy_count", torch.zeros(TARGET_T_BINS), persistent=False)

        builder = MolBuilder(vocab)

        if use_ema:
            avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)
            ema_gen = torch.optim.swa_utils.AveragedModel(gen, multi_avg_fn=avg_fn)

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.ema_gen = ema_gen if use_ema else None

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "coord_scale": coord_scale,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "distill": distill,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_strategy,
            "use_ema": use_ema,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            "target": target,
            "target_noise_std": target_noise_std,
            "target_sinkhorn_iters": target_sinkhorn_iters,
            "target_mcmc_iters": target_mcmc_iters,
            "target_mcmc_proposal": target_mcmc_proposal,
            "target_mcmc_knn_k": target_mcmc_knn_k,
            **gen.hparams,
            **integrator.hparams,
            **kwargs
        }
        self.save_hyperparameters(hparams)

        stability_metrics = {
            "atom-stability": Metrics.AtomStability(),
            "molecule-stability": Metrics.MoleculeStability()
        }
        gen_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd()
        }

        if train_smiles is not None:
            print("Creating RDKit mols from training SMILES...")
            train_mols = self.builder.mols_from_smiles(train_smiles, explicit_hs=True)
            train_mols = [mol for mol in train_mols if mol is not None]

            print("Initialising novelty metric...")
            gen_metrics["novelty"] = Metrics.Novelty(train_mols)
            print("Novelty metric complete.")

        self.stability_metrics = MetricCollection(stability_metrics, compute_groups=False)
        self.gen_metrics = MetricCollection(gen_metrics, compute_groups=False)

        if pairwise_metrics:
            pair_metrics = {
                "mol-accuracy": Metrics.MolecularAccuracy(),
                "pair-rmsd": Metrics.MolecularPairRMSD()
            }
            self.pair_metrics = MetricCollection(pair_metrics, compute_groups=False)

        self._init_params()

    def forward(self, batch, t, training=False, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"]

        # Prepare invariant atom features
        times = t.view(-1, 1, 1).expand(-1, coords.size(1), -1)
        features = torch.cat((times, atom_types), dim=2)

        # Whether to use the EMA version of the model or not
        if not training and self.ema_gen is not None:
            model = self.ema_gen
        else:
            model = self.gen

        if cond_batch is not None:
            out = model(
                coords,
                features,
                edge_feats=bonds,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                atom_mask=mask
            )

        else:
            out = model(coords, features, edge_feats=bonds, atom_mask=mask)

        return out

    def training_step(self, batch, b_idx):
        _, data, interpolated, times = batch

        if self.distill:
            return self._distill_training_step(batch)

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"])
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        interpolated,
                        times,
                        training=True,
                        cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1)
                    }

        coords, types, bonds, charges = self(
            interpolated,
            times,
            training=True,
            cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }

        losses = self._loss(data, interpolated, predicted, times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, b_idx):
        if self.ema_gen is not None:
            self.ema_gen.update_parameters(self.gen)

    def on_train_epoch_end(self):
        """Flush the plan-entropy-vs-t curve accumulated over the epoch.

        Each step averages over a batch spanning many different t, so the entropy schedule is only
        visible once binned by t -- which is the whole point of logging it.
        """

        if self.target == "hard":
            return

        means = self._plan_entropy_sum / self._plan_entropy_count.clamp_min(1.0)
        for k in range(TARGET_T_BINS):
            self.log(f"train-plan-entropy-t{k}", means[k], on_epoch=True, logger=True, sync_dist=True)

        self._plan_entropy_sum.zero_()
        self._plan_entropy_count.zero_()

    def _regression_target(self, data, interpolated, times):
        """Build the regression target and log how much blending it involved.

        For target="hard" this returns the data object itself, having done no tensor work and
        consumed no RNG -- so the hard arm stays bit-identical to the pre-target-axis baseline.
        """

        if self.target == "hard":
            return data

        target, diagnostics = permutation_target(
            data,
            interpolated,
            times,
            self.target,
            noise_std=self.target_noise_std,
            sinkhorn_iters=self.target_sinkhorn_iters,
            mcmc_iters=self.target_mcmc_iters,
            mcmc_proposal=self.target_mcmc_proposal,
            mcmc_knn_k=self.target_mcmc_knn_k,
        )

        entropy = diagnostics.pop("target-plan-entropy-per-sample", None)
        if entropy is not None:
            bins = (times * TARGET_T_BINS).long().clamp_(0, TARGET_T_BINS - 1)
            self._plan_entropy_sum.index_add_(0, bins, entropy.to(self._plan_entropy_sum.dtype))
            self._plan_entropy_count.index_add_(0, bins, torch.ones_like(entropy, dtype=self._plan_entropy_count.dtype))

        for name, value in diagnostics.items():
            self.log(f"train-{name}", value, on_step=True, logger=True)

        return target

    def validation_step(self, batch, b_idx):
        prior, data, interpolated, times = batch

        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        stabilities = self._generate_stabilities(gen_batch)
        gen_mols = self._generate_mols(gen_batch)

        self.stability_metrics.update(stabilities)
        self.gen_metrics.update(gen_mols)

        # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
        if self.pairwise_metrics:
            gen_interp_steps = max(1, int((1 - times[0].item()) * self.integrator.steps))
            gen_interp_batch = self._generate(interpolated, gen_interp_steps)
            gen_interp_mols = self._generate_mols(gen_interp_batch)
            data_mols = self._generate_mols(data)
            self.pair_metrics.update(gen_interp_mols, data_mols)

    def on_validation_epoch_end(self):
        stability_metrics_results = self.stability_metrics.compute()
        gen_metrics_results = self.gen_metrics.compute()
        pair_metrics_results = self.pair_metrics.compute() if self.pairwise_metrics else {}

        metrics = {
            **stability_metrics_results,
            **gen_metrics_results,
            **pair_metrics_results
        }

        for metric, value in metrics.items():
            progbar = True if metric == "validity" else False
            self.log(f"val-{metric}", value, on_epoch=True, logger=True, prog_bar=progbar)

        self.stability_metrics.reset()
        self.gen_metrics.reset()

        if self.pairwise_metrics:
            self.pair_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        prior, _, _, _ = batch
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=0.0
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3)

        else:
            raise ValueError("Only `constant` or `one-cycle` LR schedules are supported.")

        config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
        return config

    def _distill_training_step(self, batch):
        prior, data, interpolated, times = batch

        input_batch = prior
        cond_batch = None
        input_times = torch.zeros_like(times)

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"])
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _ = self(
                        input_batch,
                        input_times,
                        training=True,
                        cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1)
                    }

                input_batch = interpolated
                input_times = times

        coords, types, bonds, charges = self(
            input_batch,
            input_times,
            training=True,
            cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges
        }

        losses = self._distill_loss(data, predicted, times)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(f"train-{name}", loss_val, on_step=True, logger=True)

        self.log("train-loss", loss, prog_bar=True, on_step=True, logger=True)

        return loss

    def _compile_model(self, model):
        return torch.compile(model, dynamic=False, fullgraph=True, mode="reduce-overhead")

    def _loss(self, data, interpolated, predicted, times):
        target = self._regression_target(data, interpolated, times)

        pred_coords = predicted["coords"]
        coords = target["coords"]

        # From data, not target -- the mask is not permuted, and reading it from the original says so
        mask = data["mask"].unsqueeze(2)

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        coord_loss = (coord_loss * mask).mean(dim=(1, 2))

        type_loss = self._type_loss(target, interpolated, predicted)
        bond_loss = self._bond_loss(target, interpolated, predicted)
        charge_loss = self._charge_loss(target, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        return losses

    def _distill_loss(self, data, predicted, eps=1e-3):
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        mask = data["mask"].unsqueeze(2)

        pred_coords = predicted["coords"]
        pred_atomic_logits = predicted["atomics"]
        pred_bond_logits = predicted["bonds"]

        pred_atomic_dists = F.log_softmax(pred_atomic_logits, dim=-1)
        pred_bond_dists = F.log_softmax(pred_bond_logits, dim=-1)

        # When distilling data should already be given as a dist so use KL div for categoricals
        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        type_loss = F.kl_div(pred_atomic_dists, atomics, reduction="none")
        bond_loss = F.kl_div(pred_bond_dists, bonds, reduction="none")

        adj_matrix = smolF.adj_from_node_mask(mask.squeeze(-1), self_connect=True)
        n_atoms = mask.sum(dim=(1, 2)) + eps
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        coord_loss = (coord_loss * mask).mean(dim=(1, 2))
        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        bond_loss = (bond_loss * adj_matrix.unsqueeze(-1)).sum(dim=(1, 2, 3)) / n_bonds
        charge_loss = self._charge_loss(data, predicted)

        coord_loss = coord_loss.mean()
        type_loss = type_loss.mean() * self.type_loss_weight
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        charge_loss = charge_loss.mean() * self.charge_loss_weight

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss
        }
        return losses

    def _type_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["atomics"]
        atomics_dist = data["atomics"]
        mask = data["mask"].unsqueeze(2)
        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        else:
            atomics = self._ce_target(atomics_dist, flatten_end=1)
            type_loss = F.cross_entropy(pred_logits.flatten(0, 1), atomics, reduction="none")
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            masked_types = torch.argmax(interpolated["atomics"], dim=-1) == self.type_mask_index
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]
        mask = data["mask"]
        bonds = self._ce_target(data["bonds"], flatten_end=2)
        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(pred_logits.flatten(0, 2), bonds, reduction="none")
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            masked_bonds = torch.argmax(interpolated["bonds"], dim=-1) == self.bond_mask_index
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]
        charges = data["charges"]
        mask = data["mask"]
        batch_size, num_atoms, _ = pred_logits.size()

        charges = self._ce_target(charges, flatten_end=1)
        charge_loss = F.cross_entropy(pred_logits.flatten(0, 1), charges, reduction="none")
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps
        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        return charge_loss

    @property
    def _soft_labels(self):
        """Only the sinkhorn target produces genuinely soft labels.

        target="mcmc" is a single sampled permutation, so P @ onehot is still exactly one-hot and
        argmax(P @ onehot) == perm(argmax(onehot)). Routing it through the class-index path is not
        an approximation, just the cheaper way to compute the same number -- it gathers instead of
        summing over the class axis, which matters for bonds at [B*N*N, E].
        """

        return self.target == "sinkhorn"

    def _ce_target(self, dist, flatten_end):
        """Flatten a one-hot/soft target into whatever F.cross_entropy should receive.

        Under a soft target the label is a distribution over classes, not a class index. Passing it
        straight through is not a relaxation of the loss -- cross-entropy is the Bregman divergence
        generated by Phi(u) = u log u - u, so the target enters linearly and a posterior-mean label
        is exactly the right thing to regress on.

        Note the reported value is then not comparable across arms: CE against a soft label q is
        H(q) + KL(q||p), and the H(q) term is a target-only constant that inflates the number
        without touching gradients. It is logged as train-target-label-entropy so it can be
        subtracted.

        The hard path keeps the original argmax so the pre-target-axis baseline is untouched.
        """

        if not self._soft_labels:
            return torch.argmax(dist, dim=-1).flatten(0, flatten_end)

        return dist.flatten(0, flatten_end).float()

    def _generate(self, prior, steps, strategy="linear", record_trajectory=False):
        if self.distill:
            if record_trajectory:
                raise NotImplementedError("record_trajectory is not supported for distilled (self.distill) models.")
            return self._distill_generate(prior)

        if strategy == "linear":
            time_points = np.linspace(0, 1, steps + 1).tolist()

        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
            time_points.reverse()

        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        curr = {k: v.clone() for k, v in prior.items()}

        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"])
        }

        # Coordinates at each step, physical units applied at the end alongside predicted["coords"]
        # (straightness itself is scale-invariant, but keeping units consistent with the final output
        # makes an eventual absolute path-length number meaningful too)
        trajectory = [curr["coords"].clone()] if record_trajectory else None

        # The model's own raw per-step prediction of the endpoint (X-hat_1), ie. "coords" below
        # before the integrator moves curr towards it -- distinct from `trajectory`, which is the
        # *realized*, integrated path. One entry per step (not steps+1: there's no X-hat_1 prediction
        # before the first model call).
        x1_trajectory = [] if record_trajectory else None

        with torch.no_grad():
            for step_size in step_sizes:
                cond = cond_batch if self.self_condition else None
                coords, type_logits, bond_logits, charge_logits = self(curr, times, training=False, cond_batch=cond)

                if record_trajectory:
                    x1_trajectory.append(coords.clone())

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                cond_batch = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs
                }
                predicted = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "mask": curr["mask"]
                }

                curr = self.integrator.step(curr, predicted, prior, times, step_size)
                times = times + step_size

                if record_trajectory:
                    trajectory.append(curr["coords"].clone())

        predicted["coords"] = predicted["coords"] * self.coord_scale
        if record_trajectory:
            predicted["trajectory"] = torch.stack(trajectory, dim=1) * self.coord_scale
            predicted["x1_trajectory"] = torch.stack(x1_trajectory, dim=1) * self.coord_scale

        return predicted

    def _distill_generate(self, prior):
        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"])
        }

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        curr = {k: v.clone() for k, v in prior.items()}
        cond = cond_batch if self.self_condition else None

        coords, type_logits, bond_logits, charge_logits = self(curr, times, training=False, cond_batch=cond)

        type_probs = F.softmax(type_logits, dim=-1)
        bond_probs = F.softmax(bond_logits, dim=-1)
        charge_probs = F.softmax(charge_logits, dim=-1)

        predicted = {
            "coords": coords,
            "atomics": type_probs,
            "bonds": bond_probs,
            "charges": charge_probs,
            "mask": curr["mask"]
        }

        if self.self_condition:
            curr = self.integrator.step(curr, predicted, prior, times, 0.5)
            times = times + 0.5
            cond_batch = {
                "coords": coords,
                "atomics": type_probs,
                "bonds": bond_probs
            }
            coords, type_logits, bond_logits, charge_logits = self(
                curr,
                times,
                training=False,
                cond_batch=cond
            )

            type_probs = F.softmax(type_logits, dim=-1)
            bond_probs = F.softmax(bond_logits, dim=-1)
            charge_probs = F.softmax(charge_logits, dim=-1)

            predicted = {
                "coords": coords,
                "atomics": type_probs,
                "bonds": bond_probs,
                "charges": charge_probs,
                "mask": curr["mask"]
            }

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _generate_mols(self, generated, sanitise=True):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise
        )
        return mols

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(coords, atom_dists, masks, bond_dists, charge_dists)
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
