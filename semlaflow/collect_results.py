"""Merge every results artefact into one summary, as JSON and as a readable table.

The pieces are produced by separate jobs on separate hardware -- generation needs a GPU, the xTB
relaxation is CPU work, the target-collapse curve needs no model at all -- so this walks the
results tree and assembles them. Each piece also stays on disk in its own file; this is the
single place that reads all of them together.

Expected layout, all optional -- whatever is present gets merged:

    <results_dir>/
      generated/<arm>.smol, <arm>.smol.sdf     from predict.py
      xtb/<arm>.json                           from xtb_eval.py
      analysis/<arm>.json                      from analyse_generated.py
      diagnostics/target_collapse.json         from target_variance.py --save_path
      summary.json / summary.md                written here

Pairwise significance follows the corrections doc: UNPAIRED Mann-Whitney U with Cliff's delta and
a bootstrap CI on the difference of medians, computed on the per-molecule records. The arms are
blocked on molecule size, not paired, so a signed-rank test does not apply. Note also that with a
single seed these compare two *trained models*, not two *methods* -- run-to-run variation is not
separated out, so read them alongside the effect sizes rather than the p-values.

    python -m semlaflow.collect_results --results_dir results
"""

import argparse
import json
from pathlib import Path

import semlaflow.util.stats as stats

# Metrics carried into the headline table, with the source file each comes from and whether higher
# is better. dE_relax and the geometry deviations are all "lower is better".
_XTB_METRICS = ["delta_e_relax", "bond_length_dev", "bond_angle_dev", "torsion_dev", "xtb_rmsd"]
_ANALYSIS_METRICS = ["radius_of_gyration", "mean_nn_distance"]


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def gather(results_dir: Path) -> dict:
    """Collect every per-arm artefact, keyed by arm label."""

    arms = {}
    for sub, key in [("xtb", "xtb"), ("analysis", "analysis")]:
        for path in sorted((results_dir / sub).glob("*.json")):
            payload = _load_json(path)
            if payload is None:
                continue
            label = payload.get("label") or path.stem
            arms.setdefault(label, {})[key] = payload

    diagnostics = {}
    diag_dir = results_dir / "diagnostics"
    if diag_dir.is_dir():
        for path in sorted(diag_dir.glob("*.json")):
            payload = _load_json(path)
            if payload is not None:
                diagnostics[path.stem] = payload

    return {"arms": arms, "diagnostics": diagnostics}


def _records(arm: dict, source: str) -> list:
    payload = arm.get(source)
    return payload.get("records", []) if payload else []


def _values(arm: dict, source: str, metric: str) -> list:
    return [r.get(metric) for r in _records(arm, source)]


def headline_rows(arms: dict) -> list[dict]:
    """One row per arm: generation quality, then energy, then geometry."""

    rows = []
    for label in sorted(arms):
        arm = arms[label]
        analysis = (arm.get("analysis") or {}).get("summary", {})
        xtb = (arm.get("xtb") or {}).get("summary", {})

        row = {
            "arm": label,
            "n_generated": analysis.get("n_molecules"),
            "validity": analysis.get("validity"),
            "fc_validity": analysis.get("fc_validity"),
            "atom_stability": analysis.get("atom_stability"),
            "molecule_stability": analysis.get("molecule_stability"),
            "rg_median": analysis.get("radius_of_gyration-median"),
            "collapsed_fraction": analysis.get("collapsed_fraction"),
            "n_xtb_scored": xtb.get("n_succeeded"),
        }
        for metric in _XTB_METRICS:
            row[f"{metric}_median"] = xtb.get(f"{metric}-median")
            row[f"{metric}_mean"] = xtb.get(f"{metric}-mean")
        rows.append(row)

    return rows


def pairwise(arms: dict, baseline: str, n_bootstrap: int = 10000) -> list[dict]:
    """Every arm against the baseline, on the metrics with per-molecule records."""

    if baseline not in arms:
        return []

    comparisons = []
    for label in sorted(arms):
        if label == baseline:
            continue
        for source, metrics in [("xtb", _XTB_METRICS), ("analysis", _ANALYSIS_METRICS)]:
            for metric in metrics:
                a = _values(arms[label], source, metric)
                b = _values(arms[baseline], source, metric)
                if not a or not b:
                    continue
                result = stats.compare_metric(a, b, n_bootstrap=n_bootstrap)
                comparisons.append({
                    "arm": label,
                    "baseline": baseline,
                    "metric": metric,
                    "arm_median": result["a"]["median"],
                    "baseline_median": result["b"]["median"],
                    "p": result["test"]["p"],
                    "cliffs_delta": result["test"]["cliffs_delta"],
                    "ci_low": result["bootstrap"]["ci_low"],
                    "ci_high": result["bootstrap"]["ci_high"],
                })

    return comparisons


def _fmt(value, spec=".4g"):
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def render_markdown(summary: dict) -> str:
    lines = ["# Results summary", ""]

    lines += ["## Generation quality", "",
              "| arm | n | validity | fc-validity | atom stab | mol stab | Rg median (A) | collapsed frac |",
              "|---|---|---|---|---|---|---|---|"]
    for row in summary["headline"]:
        lines.append(
            f"| {row['arm']} | {_fmt(row['n_generated'], 'd') if row['n_generated'] else 'n/a'} "
            f"| {_fmt(row['validity'])} | {_fmt(row['fc_validity'])} | {_fmt(row['atom_stability'])} "
            f"| {_fmt(row['molecule_stability'])} | {_fmt(row['rg_median'])} "
            f"| {_fmt(row['collapsed_fraction'])} |"
        )

    lines += ["", "## GFN2-xTB energy and geometry (median / mean)", "",
              "dE_relax in kcal/mol, bond length in A, angles in degrees. Lower is better throughout.", "",
              "| arm | n scored | dE_relax | bond length dev | bond angle dev | torsion dev | xtb RMSD |",
              "|---|---|---|---|---|---|---|"]
    for row in summary["headline"]:
        cells = []
        for metric in _XTB_METRICS:
            cells.append(f"{_fmt(row[f'{metric}_median'])} / {_fmt(row[f'{metric}_mean'])}")
        n = row["n_xtb_scored"]
        lines.append(f"| {row['arm']} | {n if n is not None else 'n/a'} | " + " | ".join(cells) + " |")

    if summary.get("pairwise"):
        lines += ["", f"## Against baseline `{summary['baseline']}` (unpaired Mann-Whitney U)", "",
                  "Cliff's delta: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.",
                  "One seed, so these compare trained models rather than methods.", "",
                  "| arm | metric | arm median | baseline median | p | Cliff's d | 95% CI on median diff |",
                  "|---|---|---|---|---|---|---|"]
        for c in summary["pairwise"]:
            ci = f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]" if c["ci_low"] is not None else "n/a"
            lines.append(
                f"| {c['arm']} | {c['metric']} | {_fmt(c['arm_median'])} | {_fmt(c['baseline_median'])} "
                f"| {_fmt(c['p'], '.3g')} | {_fmt(c['cliffs_delta'], '.3f')} | {ci} |"
            )

    collapse = summary.get("diagnostics", {}).get("target_collapse")
    if collapse:
        lines += ["", "## Target collapse against t (no model involved)", "",
                  "ratio = ||P x1|| / ||x1||. The plan going uniform makes the target the molecular",
                  "centroid, which is the ORIGIN for zero-COM molecules -- so the label is all zeros.", "",
                  "| coupling | target | t | ratio | eff atoms averaged | plan entropy | disp var |",
                  "|---|---|---|---|---|---|---|"]
        for row in collapse.get("rows", []):
            lines.append(
                f"| {row['coupling']} | {row['target']} | {_fmt(row['t'], '.2f')} "
                f"| {_fmt(row.get('norm_ratio'))} | {_fmt(row.get('eff_atoms'))} "
                f"| {_fmt(row.get('plan_entropy'))} | {_fmt(row.get('displacement_variance'))} |"
            )

    return "\n".join(lines) + "\n"


def main(args):
    results_dir = Path(args.results_dir)
    gathered = gather(results_dir)

    summary = {
        "baseline": args.baseline,
        "headline": headline_rows(gathered["arms"]),
        "pairwise": pairwise(gathered["arms"], args.baseline, n_bootstrap=args.n_bootstrap),
        "diagnostics": gathered["diagnostics"],
    }

    json_path = results_dir / "summary.json"
    md_path = results_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2))
    md_path.write_text(render_markdown(summary))

    print(render_markdown(summary))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--baseline", type=str, default="none_hard")
    parser.add_argument("--n_bootstrap", type=int, default=10000)

    args = parser.parse_args()
    main(args)
