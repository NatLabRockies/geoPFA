"""Region-agnostic 2D plotter helpers for the probabilistic method.

Every helper takes a ``style: PlotStyle | None`` for color / theme overrides
so downstream applications (e.g., NREL-themed presentation figures) can
re-skin the outputs without touching the plotting logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from .calibration import equal_frequency_reliability_table, wilson_ci
from .decision_metrics import (
    DEFAULT_DECISION_LABELS,
    confusion_at_threshold,
    decision_class_table,
    top_n_targeting,
)

_DEFAULT_FIGSIZE = (12, 10)


@dataclass(frozen=True)
class PlotStyle:
    """Color / theme overrides for plotter helpers."""

    probability_cmap: str = "cividis"
    evidence_cmap: str = "viridis"
    diff_cmap: str = "coolwarm"
    confusion_cmap: str = "Oranges"
    positive_color: str = "tab:blue"
    negative_color: str = "tab:red"
    ci_color: str = "tab:orange"
    extra: dict = field(default_factory=dict)


def _setup_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _scatter_with_probability(  # noqa: PLR0913
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    *,
    value_col: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    title: str,
) -> None:
    xs = gdf.geometry.x.to_numpy()
    ys = gdf.geometry.y.to_numpy()
    vals = gdf[value_col].to_numpy()
    sc = ax.scatter(
        xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=12, marker="s"
    )
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)


def plot_component_panel(
    surface: gpd.GeoDataFrame,
    component_name: str,
    path: Path,
    *,
    style: PlotStyle | None = None,
) -> Path:
    """Render prior, non-spatial fit, spatial residual, and final probability.

    An unavailable intermediate is shown explicitly as unavailable. The final
    probability is never reused as a proxy for a missing model decomposition.
    """
    style = style or PlotStyle()
    has_prior = "prior_probability_spatial" in surface.columns
    has_spatial = (
        "spatial_u" in surface.columns or "spatial_u_std" in surface.columns
    )
    fig, axes = plt.subplots(2, 2, figsize=_DEFAULT_FIGSIZE)
    if has_prior:
        _scatter_with_probability(
            axes[0, 0],
            surface,
            value_col="prior_probability_spatial",
            cmap=style.probability_cmap,
            vmin=0.0,
            vmax=1.0,
            title=f"{component_name}: prior alpha_c(s)",
        )
    else:
        axes[0, 0].axis("off")
        axes[0, 0].text(
            0.5, 0.5, "no spatial prior column", ha="center", va="center"
        )
    if "nonspatial_probability" in surface.columns:
        _scatter_with_probability(
            axes[0, 1],
            surface,
            value_col="nonspatial_probability",
            cmap=style.probability_cmap,
            vmin=0.0,
            vmax=1.0,
            title=(
                f"{component_name}: fitted prior + evidence (no spatial field)"
            ),
        )
    else:
        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.5,
            0.5,
            "non-spatial fitted surface unavailable",
            ha="center",
            va="center",
        )
    if has_spatial and "spatial_u" in surface.columns:
        _scatter_with_probability(
            axes[1, 0],
            surface,
            value_col="spatial_u",
            cmap=style.diff_cmap,
            vmin=None,
            vmax=None,
            title=f"{component_name}: spatial residual u_c (logit)",
        )
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, "no spatial field", ha="center", va="center")
    _scatter_with_probability(
        axes[1, 1],
        surface,
        value_col="probability",
        cmap=style.probability_cmap,
        vmin=0.0,
        vmax=1.0,
        title=f"{component_name}: final probability",
    )
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_well_overlay(
    surface: gpd.GeoDataFrame,
    wells: gpd.GeoDataFrame,
    *,
    label_col: str,
    path: Path,
    style: PlotStyle | None = None,
) -> Path:
    """Plot a probability surface with labelled wells overlaid by label."""
    style = style or PlotStyle()
    fig, ax = plt.subplots(figsize=(8, 7))
    _scatter_with_probability(
        ax,
        surface,
        value_col="probability",
        cmap=style.probability_cmap,
        vmin=0.0,
        vmax=1.0,
        title="combined probability + wells",
    )
    wells_proj = (
        wells.to_crs(surface.crs) if surface.crs is not None else wells
    )
    labels = wells_proj[label_col].astype(float).to_numpy()
    pos_mask = labels == 1
    neg_mask = labels == 0
    if pos_mask.any():
        ax.scatter(
            wells_proj.loc[pos_mask].geometry.x,
            wells_proj.loc[pos_mask].geometry.y,
            edgecolors=style.positive_color,
            facecolors="none",
            marker="o",
            s=50,
            linewidths=1.5,
            label="positive",
        )
    if neg_mask.any():
        ax.scatter(
            wells_proj.loc[neg_mask].geometry.x,
            wells_proj.loc[neg_mask].geometry.y,
            color=style.negative_color,
            marker="x",
            s=50,
            label="negative",
        )
    ax.legend(loc="upper right")
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_reliability_diagram(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    path: Path,
    n_bins: int = 5,
    style: PlotStyle | None = None,
) -> Path:
    """Plot a reliability diagram with 95% Wilson CI error bars per bin."""
    style = style or PlotStyle()
    rows = equal_frequency_reliability_table(y, p, n_bins=n_bins)
    means = np.array([r.mean_predicted_p for r in rows])
    obs = np.array([r.observed_fraction for r in rows])
    lo = np.array([r.wilson_ci_lo for r in rows])
    hi = np.array([r.wilson_ci_hi for r in rows])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.errorbar(
        means,
        obs,
        yerr=[obs - lo, hi - obs],
        fmt="o",
        color=style.ci_color,
        capsize=4,
        label="observed (95% Wilson CI)",
    )
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title(f"reliability diagram ({n_bins} equal-frequency bins)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_decision_class_bar(
    method_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    path: Path,
    style: PlotStyle | None = None,
) -> Path:
    """Bar chart of intra-bin accuracy per decision class for each method."""
    style = style or PlotStyle()
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.8 / max(1, len(method_probs))
    x = np.arange(len(DEFAULT_DECISION_LABELS))
    for i, (method, probs) in enumerate(method_probs.items()):
        rows = decision_class_table(
            surface_name=method,
            surface_values=probs,
            well_scores=probs,
            well_labels=labels,
        )
        pct_correct = np.array(
            [
                r.intuitive_pct_correct
                if np.isfinite(r.intuitive_pct_correct)
                else 0.0
                for r in rows
            ]
        )
        ax.bar(x + i * bar_width, pct_correct, width=bar_width, label=method)
    ax.set_xticks(x + bar_width * (len(method_probs) - 1) / 2)
    ax.set_xticklabels(DEFAULT_DECISION_LABELS, rotation=20, ha="right")
    ax.set_ylabel("% correct in bin")
    ax.set_title("decision-class accuracy")
    ax.legend()
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_n_curve(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    path: Path,
    style: PlotStyle | None = None,
) -> Path:
    """Cumulative top-N targeting curve with random + oracle baselines."""
    style = style or PlotStyle()
    rows = top_n_targeting(y, p)
    n = np.array([r.n_picked for r in rows])
    hits = np.array([r.hits for r in rows])
    random = np.array([r.random_expected_hits for r in rows])
    oracle = np.array([r.max_possible_hits for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, hits, label="model", color=style.positive_color)
    ax.plot(n, random, "--", color="gray", label="random expected")
    ax.plot(n, oracle, ":", color="black", label="oracle")
    ax.set_xlabel("n picked")
    ax.set_ylabel("positive hits")
    ax.set_title("top-N targeting curve")
    ax.legend()
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    path: Path,
    style: PlotStyle | None = None,
) -> Path:
    """2x2 confusion matrix heatmap at the given threshold."""
    style = style or PlotStyle()
    cm = confusion_at_threshold(y, p, threshold=threshold)
    matrix = np.array([[cm.tn, cm.fp], [cm.fn, cm.tp]])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap=style.confusion_cmap)
    ax.set_xticks([0, 1], ["Predicted Negative", "Predicted Positive"])
    ax.set_yticks([0, 1], ["True Negative", "True Positive"])
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, str(int(val)), ha="center", va="center", color="black")
    ax.set_title(f"confusion matrix @ threshold = {threshold}")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3D plotter helpers
# ---------------------------------------------------------------------------


def plot_depth_slices(
    surface: gpd.GeoDataFrame,
    component_name: str,
    path: Path,
    *,
    n_slices: int = 4,
    style: PlotStyle | None = None,
) -> Path:
    """Render a grid of depth-slice probability panels for a 3D surface.

    Parameters
    ----------
    surface
        3D GeoDataFrame with Point(x, y, z) geometry and a ``probability``
        column.
    component_name
        Used as a title prefix.
    path
        Output PNG file path.
    n_slices
        How many evenly-spaced depth slices to show (default 4).
    style
        Optional :class:`PlotStyle` override.

    Returns
    -------
    Path
        The written PNG file.
    """
    style = style or PlotStyle()
    if len(surface) == 0 or not surface.geometry.has_z.any():
        raise ValueError("surface must have 3D Point(x, y, z) geometries")

    z_vals = surface.geometry.z.to_numpy()
    z_edges = np.linspace(z_vals.min(), z_vals.max(), n_slices + 1)
    ncols = min(n_slices, 4)
    nrows = (n_slices + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = np.asarray(axes).ravel()

    for i in range(n_slices):
        ax = axes_flat[i]
        mask = (z_vals >= z_edges[i]) & (z_vals < z_edges[i + 1])
        if mask.sum() == 0:
            ax.axis("off")
            continue
        subset = surface[mask]
        xs = subset.geometry.x.to_numpy()
        ys = subset.geometry.y.to_numpy()
        probs = subset["probability"].to_numpy()
        ax.scatter(
            xs,
            ys,
            c=probs,
            cmap=style.probability_cmap,
            vmin=0.0,
            vmax=1.0,
            s=10,
            marker="s",
        )
        mid_z = (z_edges[i] + z_edges[i + 1]) / 2.0
        ax.set_title(f"{component_name} z≈{mid_z:.0f}")
        ax.set_aspect("equal")

    for ax in axes_flat[n_slices:]:
        ax.axis("off")

    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vertical_cross_section(  # noqa: PLR0913
    surface: gpd.GeoDataFrame,
    component_name: str,
    path: Path,
    *,
    axis: str = "x",
    n_slices: int = 4,
    style: PlotStyle | None = None,
) -> Path:
    """Render vertical cross-sections (y-z or x-z) of a 3D probability surface.

    Parameters
    ----------
    surface
        3D GeoDataFrame with ``probability`` column.
    component_name
        Title prefix.
    path
        Output PNG path.
    axis
        Which horizontal axis to slice along (``"x"`` produces x-z panels
        at different y; ``"y"`` produces y-z panels at different x).
    n_slices
        Number of cross-sections to render.
    style
        Optional theme override.
    """
    style = style or PlotStyle()
    if len(surface) == 0 or not surface.geometry.has_z.any():
        raise ValueError("surface must have 3D Point(x, y, z) geometries")

    if axis == "x":
        slice_vals = surface.geometry.x.to_numpy()
        horiz_vals = surface.geometry.y.to_numpy()
        xlabel, slicename = "y", "x"
    else:
        slice_vals = surface.geometry.y.to_numpy()
        horiz_vals = surface.geometry.x.to_numpy()
        xlabel, slicename = "x", "y"

    z_vals = surface.geometry.z.to_numpy()
    probs = surface["probability"].to_numpy()
    edges = np.linspace(slice_vals.min(), slice_vals.max(), n_slices + 1)

    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    for i, ax in enumerate(np.asarray(axes).ravel()[:n_slices]):
        mask = (slice_vals >= edges[i]) & (slice_vals < edges[i + 1])
        if mask.sum() == 0:
            ax.axis("off")
            continue
        ax.scatter(
            horiz_vals[mask],
            z_vals[mask],
            c=probs[mask],
            cmap=style.probability_cmap,
            vmin=0,
            vmax=1,
            s=15,
            marker="s",
        )
        mid = (edges[i] + edges[i + 1]) / 2.0
        ax.set_title(f"{component_name} {slicename}≈{mid:.0f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("z")

    plt.tight_layout()
    _setup_dir(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


__all__ = [
    "PlotStyle",
    "plot_component_panel",
    "plot_confusion_matrix",
    "plot_decision_class_bar",
    "plot_depth_slices",
    "plot_reliability_diagram",
    "plot_top_n_curve",
    "plot_vertical_cross_section",
    "plot_well_overlay",
]
