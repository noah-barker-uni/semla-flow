"""Aromatic-aware atom stability, using the reference GEOM-Drugs valency table.

The problem this fixes. Atom stability is normally checked by summing bond orders and comparing to
a per-element allowed valence. That requires assigning a number to an aromatic bond, and Nikitin
et al. ("GEOM-Drugs Revisited", arXiv 2505.00169) show **neither 1 nor 1.5 is universally correct**
-- so any single choice mislabels some fraction of aromatic atoms. That paper names this codebase
as one of the affected implementations. An earlier hand-patch here (neutral C -> 4, N -> 3) moved
in the right direction but is still a single-number approximation.

The fix is to stop rounding at all. Count aromatic bonds and sum the non-aromatic bond orders
*separately*, then look the pair up:

    (element, formal_charge) -> [(n_aromatic_bonds, non_aromatic_valence), ...]

An atom is stable iff its (n_aromatic, non_aromatic_valence) pair appears in the list. Aromatic
carbon in benzene is (2, 1) -- two aromatic ring bonds plus one single bond to H -- and never has
to be resolved to a valence of 4 or 4.5.

The table in valency_tables/ is vendored verbatim from
github.com/isayevlab/geom-drugs-3dgen-evaluation (valency_tables/geom_drugs_h_tuple_valencies.json)
rather than re-derived, so stability numbers are comparable to that paper's.

CAVEAT, and it matters for this project: that table is derived from GEOM-Drugs. Everything here so
far is QM9. The element coverage is fine (QM9 is H/C/N/O/F, all present), but the allowed charge
and valence combinations are whatever GEOM-Drugs contains. The removed hand-patched table allowed
N+ at valence 2 and 3 with the comment "in QM9, N+ seems to be present in the form NH+ and NH2+";
the reference table allows N+ only at (0, 4) or with aromatic bonds. If a QM9 arm's stability drops
noticeably after this change, that is the reason, and it is a real difference in what is being
counted rather than a bug -- see LEGACY_QM9_EXTRA_VALENCIES below.
"""

import json
from functools import lru_cache
from pathlib import Path

# RDKit reports aromatic bonds as 1.5 from GetBondTypeAsDouble(); the interpolant encodes them as
# bond type index 4. Both are normalised to this before counting.
AROMATIC_BOND_ORDER = 1.5

DEFAULT_TABLE = "geom_drugs_h_tuple_valencies"

# Combinations the removed hand-patched table allowed which the GEOM-Drugs reference table does
# not. Only consulted when a table is loaded with allow_legacy_qm9=True. Kept as an explicit,
# switchable list rather than silently merged, so any number computed with it is identifiable as
# not-the-reference-protocol.
LEGACY_QM9_EXTRA_VALENCIES = {
    ("N", 1): [(0, 2), (0, 3)],
}


@lru_cache(maxsize=4)
def load_valency_table(name: str = DEFAULT_TABLE, allow_legacy_qm9: bool = False) -> dict:
    """Load a vendored valency table into {(element, charge): {(n_aromatic, valence), ...}}.

    Args:
        name (str): Table filename stem under valency_tables/.
        allow_legacy_qm9 (bool): Additionally allow LEGACY_QM9_EXTRA_VALENCIES. Off by default --
            on means the numbers are no longer the reference protocol's.

    Returns:
        dict: Maps (element symbol, formal charge) to a set of allowed
            (n_aromatic_bonds, non_aromatic_valence) pairs.
    """

    path = Path(__file__).parent / "valency_tables" / f"{name}.json"
    with open(path, encoding="utf-8") as table_file:
        raw = json.load(table_file)

    table = {}
    for element, by_charge in raw["valency_table"].items():
        for charge, combos in by_charge.items():
            table[(element, int(charge))] = {(int(n_arom), int(valence)) for n_arom, valence in combos}

    if allow_legacy_qm9:
        for key, combos in LEGACY_QM9_EXTRA_VALENCIES.items():
            table.setdefault(key, set()).update(combos)

    return table


def is_stable_atom(
    element: str,
    charge: int,
    n_aromatic: int,
    non_aromatic_valence: float,
    table: dict = None,
) -> bool:
    """Whether one atom's bonding pattern appears in the valency table.

    Args:
        element (str): Element symbol.
        charge (int): Formal charge.
        n_aromatic (int): Number of aromatic bonds on this atom.
        non_aromatic_valence (float): Sum of bond orders over the non-aromatic bonds.
        table (dict): As returned by load_valency_table; loaded on demand if omitted.

    Returns:
        bool: True if the (n_aromatic, non_aromatic_valence) pair is allowed. Unknown elements and
            unknown charge states are unstable, matching the previous behaviour.
    """

    if table is None:
        table = load_valency_table()

    # A fractional non-aromatic valence means a bond order that is neither integral nor aromatic,
    # which no table entry can match
    if float(non_aromatic_valence) != int(non_aromatic_valence):
        return False

    allowed = table.get((element, int(charge)))
    if allowed is None:
        return False

    return (int(n_aromatic), int(non_aromatic_valence)) in allowed


def split_bond_orders(bond_orders) -> tuple[int, float]:
    """Split an atom's bond orders into (aromatic count, non-aromatic order sum).

    Args:
        bond_orders (iterable[float]): Bond orders for one atom, aromatic bonds as 1.5.

    Returns:
        tuple[int, float]: (n_aromatic, non_aromatic_valence).
    """

    n_aromatic = 0
    non_aromatic = 0.0
    for order in bond_orders:
        if order == AROMATIC_BOND_ORDER:
            n_aromatic += 1
        else:
            non_aromatic += order

    return n_aromatic, non_aromatic
