"""Tests for the 2D plotter helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402

import pytest

from geopfa.prob import plotting as plotting_module

from geopfa.prob.plotting import (
    PlotStyle,
    plot_component_panel,
    plot_confusion_matrix,
    plot_decision_class_bar,
    plot_reliability_diagram,
    plot_top_n_curve,
    plot_well_overlay,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _fitted_surface():
    from geopfa.prob.fitting import fit_component_probability

    fixture = make_synthetic_pfa(grid_n=10, n_wells=25, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_component_probability(
            comp,
            prior_probability=0.55,
            prior_layer_name="prior_layer_a",
            include_spatial=True,
            labeled_wells=fixture.wells,
            label_column="heat_label",
        )
    return result.probability, fixture.wells


def test_plot_component_panel_writes_png(tmp_path: Path) -> None:
    surface, _ = _fitted_surface()
    out_path = tmp_path / "comp.png"
    plot_component_panel(surface, "component_a", out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_well_overlay_writes_png(tmp_path: Path) -> None:
    surface, wells = _fitted_surface()
    out_path = tmp_path / "overlay.png"
    plot_well_overlay(surface, wells, label_col="heat_label", path=out_path)
    assert out_path.exists()


def test_plot_reliability_diagram_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=80)
    y = (rng.uniform(0.0, 1.0, size=80) < p).astype(int)
    out_path = tmp_path / "reliability.png"
    plot_reliability_diagram(y, p, path=out_path)
    assert out_path.exists()


def test_plot_decision_class_bar_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    # Simulate two methods on the same wells
    probs_a = rng.uniform(0.0, 1.0, size=40)
    probs_b = rng.uniform(0.0, 1.0, size=40)
    labels = (rng.uniform(0.0, 1.0, size=40) < 0.4).astype(int)
    out_path = tmp_path / "decision.png"
    plot_decision_class_bar(
        {"probability": probs_a, "favorability": probs_b},
        labels,
        path=out_path,
    )
    assert out_path.exists()


def test_plot_top_n_curve_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 1.0, size=50)
    y = (rng.uniform(0.0, 1.0, size=50) < 0.5).astype(int)
    out_path = tmp_path / "topn.png"
    plot_top_n_curve(y, p, path=out_path)
    assert out_path.exists()


def test_plot_confusion_matrix_writes_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.0, 1.0, size=100)
    y = (rng.uniform(0.0, 1.0, size=100) < 0.5).astype(int)
    out_path = tmp_path / "confusion.png"
    plot_confusion_matrix(y, p, threshold=0.5, path=out_path)
    assert out_path.exists()


def test_plot_style_overrides_apply(tmp_path: Path) -> None:
    surface, _ = _fitted_surface()
    out_path = tmp_path / "comp_styled.png"
    style = PlotStyle(
        probability_cmap="viridis",
        positive_color="green",
        negative_color="purple",
    )
    plot_component_panel(surface, "component_a", out_path, style=style)
    assert out_path.exists()


def test_plot_component_panel_handles_no_prior_column(tmp_path: Path) -> None:
    """A surface without prior_probability_spatial should still plot."""
    surface, _ = _fitted_surface()
    surface = surface.drop(
        columns=[
            c
            for c in ["prior_probability_spatial", "spatial_u"]
            if c in surface.columns
        ]
    )
    out_path = tmp_path / "no_prior.png"
    plot_component_panel(surface, "component_a", out_path)
    assert out_path.exists()


def test_plot_component_panel_never_substitutes_final_for_an_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing evidence-only surface must be shown as missing, not fabricated."""
    surface, _ = _fitted_surface()
    calls: list[tuple[str, str]] = []
    original = plotting_module._scatter_with_probability  # noqa: SLF001

    def record(*args, value_col: str, title: str, **kwargs) -> None:
        calls.append((value_col, title))
        original(*args, value_col=value_col, title=title, **kwargs)

    monkeypatch.setattr(plotting_module, "_scatter_with_probability", record)
    plot_component_panel(surface, "component_a", tmp_path / "panel.png")

    assert [value for value, _ in calls].count("probability") == 1


def test_plot_component_panel_uses_declared_evidence_only_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    surface, _ = _fitted_surface()
    surface["nonspatial_probability"] = 0.4
    calls: list[tuple[str, str]] = []
    original = plotting_module._scatter_with_probability  # noqa: SLF001

    def record(*args, value_col: str, title: str, **kwargs) -> None:
        calls.append((value_col, title))
        original(*args, value_col=value_col, title=title, **kwargs)

    monkeypatch.setattr(plotting_module, "_scatter_with_probability", record)
    plot_component_panel(surface, "component_a", tmp_path / "panel.png")

    assert (
        "nonspatial_probability",
        "component_a: fitted prior + evidence (no spatial field)",
    ) in calls
