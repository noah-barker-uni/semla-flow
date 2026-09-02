"""One generated molecule from each arm, drawn side by side as a single figure.

Everything you are likely to want to change lives in the three config blocks below -- panel
contents and rotations, figure geometry, and drawing style. Edit and re-run; nothing else needs
touching.

    python -m semlaflow.plot_molecules                  # write the figure
    python -m semlaflow.plot_molecules --contact none_hard --n 24
                                                        # contact sheet to choose an index from

Rotation is three Euler angles in DEGREES, applied about x, then y, then z, to coordinates that
have already been centred. `(0, 0, 0)` draws the molecule as stored, looking down -z. Change one
angle at a time; 15-30 degree steps are usually enough to find a readable orientation.

Why this is a hand-rolled projection rather than matplotlib's 3D axes: mplot3d has no real depth
buffer, so atoms and bonds interleave wrongly and the result looks broken for anything but a
scatter. Here the molecule is rotated, projected orthographically, and every primitive is drawn
back-to-front in depth order, which occludes correctly and gives clean publication output.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

# ----------------------------------------------------------------------------------------------
# 1. PANELS -- what to draw. One dict per panel, drawn left to right.
#
#    index : which molecule in the SDF. Use --contact to find one you like.
#    rot   : (rx, ry, rz) in degrees, applied in that order. This is the orientation control.
#    zoom  : >1 makes the molecule bigger in its panel, <1 smaller.
# ----------------------------------------------------------------------------------------------

GEN = "output/results/generated"

PANELS = [
    dict(label="None",       sdf=f"{GEN}/none_hard.smol.sdf",      index=0, rot=(0, 0, 0), zoom=1.0),
    dict(label="Hungarian",  sdf=f"{GEN}/hungarian_hard.smol.sdf", index=0, rot=(0, 0, 0), zoom=1.0),
    dict(label="Sinkhorn",   sdf=f"{GEN}/none_sinkhorn.smol.sdf",  index=0, rot=(0, 0, 0), zoom=1.0),
    dict(label="Metropolis", sdf=f"{GEN}/none_mcmc.smol.sdf",      index=0, rot=(0, 0, 0), zoom=1.0),
]

# ----------------------------------------------------------------------------------------------
# 2. FIGURE -- geometry and output.
# ----------------------------------------------------------------------------------------------

FIGURE = dict(
    nrows=1,
    ncols=4,
    panel_size=(2.4, 2.6),      # inches per panel (width, height)
    dpi=300,
    out="figures/generated_molecules.png",   # .pdf works too, and is better for LaTeX
    label_fontsize=11,
    label_pad=6,                # points between the panel and its label
    wspace=0.02,                # horizontal gap between panels, as a fraction of panel width
    hspace=0.14,
    transparent=False,
    facecolor="white",
)

# ----------------------------------------------------------------------------------------------
# 3. STYLE -- how the molecules are drawn.
# ----------------------------------------------------------------------------------------------

STYLE = dict(
    show_hydrogens=True,
    atom_scale=1.0,             # multiplies the radii below
    bond_width=2.8,             # points
    bond_colour="#3c3c3c",
    bond_gap=0.20,              # separation of the parallel lines in a double/triple bond, Angstrom.
                                # Must exceed the drawn line width or multiple bonds merge into one
                                # thick line: at bond_width 2.8pt a line is about 0.11 A across here.
    split_bond_colours=False,   # colour each half of a bond by the atom it touches
    atom_edge_width=0.9,
    atom_edge_colour="#2b2b2b",
    margin=0.55,                # blank space around the molecule, Angstrom
    shrink_bonds=0.72,          # pull bond ends back inside the atom circles, as a fraction of radius
)

# Display radii in Angstrom. Deliberately smaller than van der Waals radii, which would overlap
# and hide the bonds; these are chosen to read clearly at figure size.
ATOM_RADII = {"H": 0.16, "C": 0.28, "N": 0.27, "O": 0.26, "F": 0.24,
              "S": 0.32, "Cl": 0.31, "Br": 0.34, "I": 0.37, "P": 0.32}
DEFAULT_RADIUS = 0.28

# CPK-ish, with a light grey for hydrogen so it does not disappear on white.
ATOM_COLOURS = {"H": "#e8e8e8", "C": "#4a4a4a", "N": "#3050f8", "O": "#ff2010", "F": "#90e050",
                "S": "#ffff30", "Cl": "#1ff01f", "Br": "#a62929", "I": "#940094", "P": "#ff8000"}
DEFAULT_COLOUR = "#ff1493"      # deliberately garish: an unstyled element should be obvious


def rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler rotation about x, then y, then z. Angles in degrees."""

    rx, ry, rz = (math.radians(a) for a in (rx, ry, rz))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def load_molecule(sdf_path: str, index: int):
    """Read one molecule from an SDF, keeping hydrogens and tolerating imperfect valences.

    sanitize=False because these are generated structures: some will not pass RDKit's valence
    model, and a figure of the failures is often exactly what you want to show. Aromaticity is
    perceived separately and allowed to fail quietly, since it only affects how bonds are drawn.
    """

    path = Path(sdf_path)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    if path.stat().st_size == 0:
        raise SystemExit(f"{path} is empty -- that arm produced no constructible molecules")

    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mols = []
    for i, mol in enumerate(supplier):
        if mol is not None:
            mols.append(mol)
        if len(mols) > index:
            break

    if len(mols) <= index:
        raise SystemExit(f"{path} has fewer than {index + 1} readable molecules")

    mol = mols[index]
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def molecule_geometry(mol, show_hydrogens: bool):
    """Coordinates, element symbols and bonds, with hydrogens optionally dropped."""

    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    keep = [i for i, s in enumerate(symbols) if show_hydrogens or s != "H"]
    remap = {old: new for new, old in enumerate(keep)}

    bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in remap and j in remap:
            bonds.append((remap[i], remap[j], bond.GetBondTypeAsDouble()))

    return coords[keep], [symbols[i] for i in keep], bonds


def draw_molecule(ax, mol, rot, zoom, style):
    """Rotate, project orthographically, and draw back-to-front so occlusion is correct."""

    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    coords, symbols, bonds = molecule_geometry(mol, style["show_hydrogens"])
    coords = coords - coords.mean(axis=0)
    xyz = coords @ rotation_matrix(*rot).T
    xy, depth = xyz[:, :2], xyz[:, 2]

    radii = np.array([ATOM_RADII.get(s, DEFAULT_RADIUS) for s in symbols]) * style["atom_scale"]

    # Every primitive is (depth, draw_fn). Sorting on depth and drawing in order is what makes
    # atoms correctly hide the bonds behind them.
    prims = []

    for i, j, order in bonds:
        a, b = xy[i], xy[j]
        vec = b - a
        length = float(np.hypot(*vec))
        if length < 1e-9:
            continue
        unit = vec / length
        normal = np.array([-unit[1], unit[0]])

        # Pull the ends back inside the atom circles so the line does not cross them.
        start = a + unit * radii[i] * style["shrink_bonds"]
        end = b - unit * radii[j] * style["shrink_bonds"]

        n_lines = {1.0: 1, 1.5: 2, 2.0: 2, 3.0: 3}.get(order, 1)
        offsets = np.linspace(-(n_lines - 1) / 2, (n_lines - 1) / 2, n_lines) * style["bond_gap"]
        z = float((depth[i] + depth[j]) / 2)

        for k, off in enumerate(offsets):
            p, q = start + normal * off, end + normal * off
            dashed = order == 1.5 and k == 1
            if style["split_bond_colours"]:
                mid = (p + q) / 2
                segments = [(p, mid, ATOM_COLOURS.get(symbols[i], DEFAULT_COLOUR)),
                            (mid, q, ATOM_COLOURS.get(symbols[j], DEFAULT_COLOUR))]
            else:
                segments = [(p, q, style["bond_colour"])]
            for u, v, colour in segments:
                prims.append((z, Line2D([u[0], v[0]], [u[1], v[1]],
                                        color=colour, linewidth=style["bond_width"],
                                        solid_capstyle="round",
                                        linestyle=(0, (2, 1.6)) if dashed else "-",
                                        zorder=1)))

    for i, symbol in enumerate(symbols):
        prims.append((float(depth[i]), Circle(
            (xy[i, 0], xy[i, 1]), radii[i],
            facecolor=ATOM_COLOURS.get(symbol, DEFAULT_COLOUR),
            edgecolor=style["atom_edge_colour"], linewidth=style["atom_edge_width"], zorder=2)))

    for _, artist in sorted(prims, key=lambda p: p[0]):
        ax.add_line(artist) if isinstance(artist, Line2D) else ax.add_patch(artist)

    half = (np.abs(xy).max() + radii.max() + style["margin"]) / max(zoom, 1e-6)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")


def build_figure(panels, figure, style):
    import matplotlib.pyplot as plt

    nrows, ncols = figure["nrows"], figure["ncols"]
    pw, ph = figure["panel_size"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows),
                             dpi=figure["dpi"], facecolor=figure["facecolor"])
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(panels):]:
        ax.axis("off")

    for ax, panel in zip(axes, panels):
        mol = load_molecule(panel["sdf"], panel["index"])
        draw_molecule(ax, mol, panel["rot"], panel.get("zoom", 1.0), style)
        ax.set_title(panel["label"], fontsize=figure["label_fontsize"], pad=figure["label_pad"])
        print(f"  {panel['label']:<12} {Path(panel['sdf']).name}  index {panel['index']}  "
              f"{mol.GetNumAtoms()} atoms  rot {panel['rot']}")

    fig.subplots_adjust(wspace=figure["wspace"], hspace=figure["hspace"])
    out = Path(figure["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=figure["dpi"],
                transparent=figure["transparent"], facecolor=figure["facecolor"])
    print(f"\nwrote {out}")
    return fig


def contact_sheet(arm, n, figure, style, cols=6):
    """Render the first n molecules of one arm, numbered, so an index can be chosen by eye."""

    import matplotlib.pyplot as plt

    sdf = next((p["sdf"] for p in PANELS if p["label"].lower() == arm.lower()), None)
    if sdf is None:
        sdf = f"{GEN}/{arm}.smol.sdf"

    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 1.9 * rows),
                             dpi=140, facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False)
    shown = 0
    for mol in supplier:
        if shown >= n:
            break
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        draw_molecule(axes[shown], mol, (0, 0, 0), 1.0, style)
        axes[shown].set_title(f"index {shown}  ({mol.GetNumAtoms()} atoms)", fontsize=7)
        shown += 1

    out = Path(figure["out"]).parent / f"contact_{arm}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"wrote {out}  ({shown} molecules, all at rot (0,0,0))")
    return fig


def main():
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contact", type=str, default=None,
                        help="render a numbered grid for one arm instead, to pick an index from")
    parser.add_argument("--n", type=int, default=24, help="molecules in the contact sheet")
    parser.add_argument("--out", type=str, default=None, help="override the output path")
    args = parser.parse_args()

    figure = dict(FIGURE)
    if args.out:
        figure["out"] = args.out

    if args.contact:
        contact_sheet(args.contact, args.n, figure, STYLE)
    else:
        build_figure(PANELS, figure, STYLE)


if __name__ == "__main__":
    main()
