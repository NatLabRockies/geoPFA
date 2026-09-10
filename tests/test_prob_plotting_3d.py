"""Tests for 3D plotter helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import numpy as np
import pytest

from geopfa.prob.plotting import plot_depth_slices, plot_vertical_cross_section
from tests.fixtures.synthetic_prob_3d import make_synthetic_pfa_3d


def _surface_3d():
    fixture = make_synthetic_pfa_3d(grid_n=5, grid_nz=4, n_wells=15, seed=0)
    from geopfa.prob.fitting import fit_component_probability  # noqa: PLC0415
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_component_probability(
            comp,
            prior_probability=0.55,
            include_spatial=False,
            labeled_wells=fixture.wells,
            label_column="heat_label",
        )
    return result.probability


def test_plot_depth_slices_creates_png(tmp_path: Path) -> None:
    surface = _surface_3d()
    out = tmp_path / "depth_slices.png"
    plot_depth_slices(surface, "component_a", out, n_slices=4)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_vertical_cross_section_x_axis(tmp_path: Path) -> None:
    surface = _surface_3d()
    out = tmp_path / "cross_x.png"
    plot_vertical_cross_section(surface, "component_a", out, axis="x", n_slices=3)
    assert out.exists()


def test_plot_vertical_cross_section_y_axis(tmp_path: Path) -> None:
    surface = _surface_3d()
    out = tmp_path / "cross_y.png"
    plot_vertical_cross_section(surface, "component_a", out, axis="y", n_slices=3)
    assert out.exists()


def test_plot_depth_slices_raises_on_2d_surface(tmp_path: Path) -> None:
    from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: PLC0415
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from geopfa.prob.fitting import fit_component_probability  # noqa: PLC0415
        result = fit_component_probability(
            comp, prior_probability=0.5, include_spatial=False,
            labeled_wells=fixture.wells, label_column="heat_label"
        )
    out = tmp_path / "bad.png"
    with pytest.raises(ValueError, match="3D"):
        plot_depth_slices(result.probability, "comp_a", out)
