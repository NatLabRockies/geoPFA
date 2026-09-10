"""Hierarchical-pooling diagnostics report writer.

Writes per-component pooling diagnostics (shrinkage, pooling factor,
global hyperprior posteriors, per-region β summaries) as both a
machine-readable JSON and a human-readable Markdown file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_hierarchical_diagnostics(  # noqa: PLR0914
    per_component: dict[str, dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write pooling diagnostics as JSON + Markdown.

    Parameters
    ----------
    per_component
        Dict mapping component name to an explicitly supplied pooling-
        diagnostics record. The writer does not fit a model.
    output_dir
        Directory to write into (created if absent).

    Returns
    -------
    (json_path, md_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = output_dir / "hierarchical_diagnostics.json"
    json_path.write_text(
        json.dumps(per_component, indent=2, default=str), encoding="utf-8"
    )

    # --- Markdown ---
    lines: list[str] = [
        "# Hierarchical pooling diagnostics",
        "",
        "Per-component Bayesian partial-pooling report.",
        "",
    ]
    for comp_name, diag in per_component.items():
        lines += [f"## {comp_name}", ""]
        n_reg = diag.get("n_regions", "?")
        rhat = diag.get("r_hat_max", float("nan"))
        ess = diag.get("ess_min", float("nan"))
        lines += [
            f"- Regions: {n_reg}",
            f"- max r̂: {rhat:.3f}",
            f"- min ESS: {ess:.0f}",
            "",
        ]

        # Global hyperpriors
        mu_mean = diag.get("global_mu_mean")
        sig_mean = diag.get("global_sigma_mean")
        feat = None
        per_region = diag.get("per_region", {})
        if per_region:
            first = next(iter(per_region.values()))
            feat = first.get("feature_names", [])
        if mu_mean and feat:
            lines.append("### Global hyperprior posteriors")
            lines.append("")
            lines.append("| Layer | mu_k (mean) | sigma_k (mean) |")
            lines.append("|---|---|---|")
            for i, name in enumerate(feat):
                mu = mu_mean[i] if i < len(mu_mean) else float("nan")
                sig = (
                    sig_mean[i]
                    if sig_mean and i < len(sig_mean)
                    else float("nan")
                )
                lines.append(f"| `{name}` | {mu:.3f} | {sig:.3f} |")
            lines.append("")

        # Pooling factor
        pf = diag.get("pooling_factor")
        if pf and feat:
            lines.append("### Pooling factor (0 = complete pool, 1 = no pool)")
            lines.append("")
            lines.append("| Layer | pooling_factor |")
            lines.append("|---|---|")
            for i, name in enumerate(feat):
                pfi = pf[i] if i < len(pf) else float("nan")
                lines.append(f"| `{name}` | {pfi:.3f} |")
            lines.append("")

        # Per-region shrinkage
        if per_region:
            lines.append("### Per-region β and shrinkage")
            lines.append("")
            header = (
                "| Region | "
                + " | ".join(f"`{f}` β" for f in (feat or []))
                + " |"
            )
            sep = "|---|" + "---|" * len(feat or [])
            lines += [header, sep]
            for reg_name, rd in per_region.items():
                bm = rd.get("beta_mean", [])
                row = (
                    f"| {reg_name} | "
                    + " | ".join(
                        f"{bm[i]:.3f}" if i < len(bm) else "—"
                        for i in range(len(feat or []))
                    )
                    + " |"
                )
                lines.append(row)
            lines.append("")

    md_path = output_dir / "hierarchical_diagnostics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


__all__ = ["write_hierarchical_diagnostics"]
