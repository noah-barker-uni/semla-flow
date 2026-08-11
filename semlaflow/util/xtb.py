"""GFN2-xTB relaxation of generated molecules, via the xtb command line program.

Why this exists: every energy metric in semlaflow/util/rdkit.py is MMFF-derived. Nikitin et al.
(arXiv 2505.00169) show reference GEOM-Drugs conformers score mean dE_relax ~16 kcal/mol under
MMFF but ~0 under GFN2-xTB, because the dataset was *built* by GFN2-xTB optimisation. MMFF's
15-20 kcal/mol error is larger than the effect this project is trying to measure, so MMFF
comparisons mask exactly the differences we care about. MMFF is kept as a coarse outlier filter;
GFN2-xTB is the primary energy metric.

The protocol deliberately mirrors github.com/isayevlab/geom-drugs-3dgen-evaluation
(scripts/energy_benchmark/xtb_optimization.py) so numbers are comparable to their published
table -- same program, same flags, same parsed quantity:

    xtb <mol.xyz> --opt --charge <q> --gfn 2

`--gfn 2` is xtb's own default and is passed explicitly only so the protocol is self-documenting
and cannot silently change if a future xtb release moves the default.

dE_relax is read from xtb's "total energy gain" line and NEGATED. xtb reports the energy change
during optimisation, which is negative (the geometry relaxes downhill); the reported metric is
how far above its own relaxed minimum the generated geometry sat, so it is positive and larger
is worse.

This is CPU work and embarrassingly parallel over molecules -- see semlaflow/xtb_eval.py, which
is meant to run on a CPU allocation rather than burning GPU hours.

Getting the binary: there is no prebuilt aarch64 xtb release and no aarch64 tblite wheel, but
conda-forge does build xtb for linux-aarch64 and osx-arm64. Point SEMLAFLOW_XTB_BINARY at it, or
put it on PATH.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem

# Environment override for installations where xtb is not on PATH (the common case on a cluster
# where it comes from a conda-forge prefix rather than the module system).
XTB_BINARY_ENV_VAR = "SEMLAFLOW_XTB_BINARY"
DEFAULT_XTB_BINARY = "xtb"

# Per-molecule wall-clock ceiling. Some generated geometries are strained enough that the
# optimiser thrashes; without a cap one molecule can stall a whole sweep.
DEFAULT_TIMEOUT_SECONDS = 300

BOHR_TO_ANGSTROM = 0.529177210903

# " total energy gain   :        -0.0074518 Eh       -4.6761 kcal/mol"
_ENERGY_GAIN_RE = re.compile(
    r"total energy gain\s*:\s*(?P<hartree>[-\d.Ee+]+)\s*Eh\s+(?P<kcal>[-\d.Ee+]+)\s*kcal/mol"
)

# " total RMSD          :         0.0841523 a0        0.0445 Å"
_RMSD_RE = re.compile(r"total RMSD\s*:\s*(?P<bohr>[-\d.Ee+]+)\s*a0\s+(?P<angstrom>[-\d.Ee+]+)")


class XtbError(RuntimeError):
    """Raised when xtb cannot be run at all, as opposed to failing on one molecule."""


def resolve_xtb_binary(binary: Optional[str] = None) -> str:
    """Resolve which xtb executable to use, without checking that it works."""

    if binary is not None:
        return binary

    return os.environ.get(XTB_BINARY_ENV_VAR, DEFAULT_XTB_BINARY)


def xtb_version(binary: Optional[str] = None) -> Optional[str]:
    """Return the xtb version string, or None if xtb is not runnable.

    Used both as an availability check and to record the exact program version alongside results,
    since energies are only comparable within one xtb version.
    """

    binary = resolve_xtb_binary(binary)
    if shutil.which(binary) is None and not Path(binary).exists():
        return None

    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None

    match = re.search(r"xtb version\s+(\S+)", proc.stdout + proc.stderr)
    return match.group(1) if match else None


def mol_to_xyz_block(mol: Chem.rdchem.Mol) -> str:
    """Write a molecule's first conformer as an XYZ block.

    XYZ carries no bond information, which is exactly what we want -- xtb determines bonding from
    the geometry itself, so the relaxation is not conditioned on RDKit's perception of the graph.
    """

    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError("mol must have at least one conformer to write as xyz.")

    conf = mol.GetConformer()
    lines = [str(mol.GetNumAtoms()), ""]
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<4}{pos.x:>20.12f}{pos.y:>20.12f}{pos.z:>20.12f}")

    return "\n".join(lines) + "\n"


def parse_xtb_output(text: str) -> dict:
    """Pull the relaxation energy and RMSD out of an xtb --opt log.

    Returns dE_relax already negated into "how strained the input geometry was", so it is >= 0 for
    a converged optimisation and larger is worse.
    """

    result = {"delta_e_relax": None, "energy_gain_kcal": None, "xtb_rmsd": None, "converged": False}

    energy_match = _ENERGY_GAIN_RE.search(text)
    if energy_match is not None:
        gain = float(energy_match.group("kcal"))
        result["energy_gain_kcal"] = gain
        result["delta_e_relax"] = -gain

    rmsd_match = _RMSD_RE.search(text)
    if rmsd_match is not None:
        result["xtb_rmsd"] = float(rmsd_match.group("angstrom"))

    result["converged"] = "GEOMETRY OPTIMIZATION CONVERGED" in text
    result["normal_termination"] = "normal termination of xtb" in text
    return result


def read_xtbopt_xyz(path: Path, template: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Read xtb's optimised geometry back onto a copy of the input molecule.

    xtb writes xyz in the same atom order it was given, so the coordinates can be dropped straight
    onto the template's conformer. Keeping the template's graph (rather than re-perceiving bonds
    from the optimised xyz) is what makes the geometry deviations well defined: both conformers
    are then measured over the same internal coordinates.
    """

    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())

    if n_atoms != template.GetNumAtoms():
        raise ValueError(f"xtb returned {n_atoms} atoms, expected {template.GetNumAtoms()}.")

    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    opt_mol = Chem.Mol(template)
    conf = opt_mol.GetConformer()
    for idx, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))

    return opt_mol


def relax_molecule(
    mol: Chem.rdchem.Mol,
    binary: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    keep_dir: Optional[Path] = None,
) -> dict:
    """GFN2-xTB relax one molecule and report how far it moved.

    Each call runs in its own temporary working directory: xtb scatters .charges / .wbo /
    .xtbrestart / .xtbopt.* files into the CWD, so concurrent calls sharing a directory would
    overwrite each other's output.

    Args:
        mol (Chem.Mol): Molecule with a conformer to relax. Not modified.
        binary (Optional[str]): xtb executable; defaults to $SEMLAFLOW_XTB_BINARY or "xtb".
        timeout (int): Per-molecule wall-clock ceiling in seconds.
        keep_dir (Optional[Path]): If given, run there and leave the xtb files behind for
            debugging instead of using a temporary directory.

    Returns:
        dict: delta_e_relax (kcal/mol, >= 0, larger is worse), xtb_rmsd (Angstrom), opt_mol
            (Chem.Mol or None), converged/normal_termination flags, and error (None on success).
    """

    failure = {
        "delta_e_relax": None,
        "energy_gain_kcal": None,
        "xtb_rmsd": None,
        "opt_mol": None,
        "converged": False,
        "normal_termination": False,
    }

    if mol is None or mol.GetNumConformers() == 0:
        return {**failure, "error": "no conformer"}

    binary = resolve_xtb_binary(binary)
    charge = Chem.GetFormalCharge(mol)

    context = tempfile.TemporaryDirectory() if keep_dir is None else None
    work_dir = Path(context.name) if context is not None else Path(keep_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        xyz_path = work_dir / "input.xyz"
        xyz_path.write_text(mol_to_xyz_block(mol))

        # OMP threads are pinned to 1 because parallelism here is over molecules, not within one:
        # letting each xtb spawn a full thread team would oversubscribe the node badly under a
        # process pool. semlaflow/__init__.py sets the same default for the same reason.
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")

        try:
            proc = subprocess.run(
                [binary, xyz_path.name, "--opt", "--charge", str(charge), "--gfn", "2"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except FileNotFoundError as err:
            raise XtbError(
                f"xtb binary '{binary}' not found. Put it on PATH or set {XTB_BINARY_ENV_VAR}."
            ) from err
        except subprocess.TimeoutExpired:
            return {**failure, "error": f"timeout after {timeout}s"}

        parsed = parse_xtb_output(proc.stdout)

        if parsed["delta_e_relax"] is None:
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-1:]
            return {**failure, "error": f"no energy in xtb output ({tail})"}

        opt_path = work_dir / "xtbopt.xyz"
        if not opt_path.exists():
            return {**parsed, "opt_mol": None, "error": "xtb wrote no optimised geometry"}

        try:
            opt_mol = read_xtbopt_xyz(opt_path, mol)
        except (ValueError, IndexError) as err:
            return {**parsed, "opt_mol": None, "error": f"could not read optimised geometry: {err}"}

        return {**parsed, "opt_mol": opt_mol, "error": None}

    finally:
        if context is not None:
            context.cleanup()


def conformer_rmsd(mol_a: Chem.rdchem.Mol, mol_b: Chem.rdchem.Mol) -> Optional[float]:
    """RMSD between two conformers of the same molecule, in the given atom order, no alignment.

    xtb's own "total RMSD" is reported too; this recomputes it independently so a parsing change
    cannot silently pass unnoticed.
    """

    if mol_a is None or mol_b is None:
        return None

    if mol_a.GetNumAtoms() != mol_b.GetNumAtoms():
        return None

    coords_a = np.array(mol_a.GetConformer().GetPositions())
    coords_b = np.array(mol_b.GetConformer().GetPositions())
    return float(np.sqrt(((coords_a - coords_b) ** 2).sum(axis=1).mean()))
