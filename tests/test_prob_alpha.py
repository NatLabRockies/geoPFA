"""Tests for the alpha_c (prior offset) builder."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from geopfa.prob.alpha import (
    AlphaCResult,
    build_alpha_c,
    write_alpha_provenance,
)
from geopfa.prob.config import AlphaModeConfig
from tests.fixtures.synthetic_prob import make_synthetic_pfa
from tests.fixtures.synthetic_prob_3d import make_synthetic_pfa_3d


def _component_grid_a():
    fixture = make_synthetic_pfa(grid_n=8, n_wells=10, seed=0)
    return fixture.pfa["criteria"]["geologic"]["components"]["component_a"]


# ---------------------------------------------------------------------------
# Scalar mode
# ---------------------------------------------------------------------------


def test_scalar_mode_returns_uniform_logit() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.4)
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    assert isinstance(result, AlphaCResult)
    expected = np.log(0.4 / 0.6)
    assert np.allclose(result.grid_offset, expected)
    assert result.scalar_fallback == pytest.approx(expected)
    assert result.excluded_layer_names == set()
    assert result.provenance["mode"] == "scalar"


# ---------------------------------------------------------------------------
# layer_logit mode
# ---------------------------------------------------------------------------


def test_layer_logit_mode_uses_layer_values() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="layer_logit",
        layer="prior_layer_a",
        p_min=0.2,
        p_max=0.8,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    assert "prior_layer_a" in result.excluded_layer_names
    # Offset is logit-rescaled so it cannot exceed logit(0.8) or fall below logit(0.2):
    lo, hi = np.log(0.2 / 0.8), np.log(0.8 / 0.2)
    assert result.grid_offset.min() >= lo - 1e-9
    assert result.grid_offset.max() <= hi + 1e-9
    assert result.provenance["mode"] == "layer_logit"
    assert result.provenance["layer"] == "prior_layer_a"


def test_layer_logit_mode_missing_layer_fails_closed() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="layer_logit",
        layer="layer_that_does_not_exist",
        scalar_fallback_pr0=0.5,
    )
    with pytest.raises(ValueError, match="layer_that_does_not_exist"):
        build_alpha_c(comp, cfg, grid_gdf=grid_gdf)


def test_layer_logit_rejects_partial_missing_coverage() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    comp["layers"]["prior_layer_a"]["model"].loc[0, "value_interpolated"] = (
        np.nan
    )
    cfg = AlphaModeConfig(mode="layer_logit", layer="prior_layer_a")

    with pytest.raises(ValueError, match="finite value at every grid cell"):
        build_alpha_c(comp, cfg, grid_gdf=grid_gdf)


def test_layer_logit_rejects_constant_layer_scaling() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    comp["layers"]["prior_layer_a"]["model"].loc[:, "value_interpolated"] = 7.0
    cfg = AlphaModeConfig(mode="layer_logit", layer="prior_layer_a")

    with pytest.raises(ValueError, match="constant"):
        build_alpha_c(comp, cfg, grid_gdf=grid_gdf)


# ---------------------------------------------------------------------------
# thermal_exceedance mode
# ---------------------------------------------------------------------------


def _write_thermal_raster(
    path: Path,
    *,
    value: float = 250.0,
    n: int = 12,
    crs: str | None = "EPSG:32611",
) -> None:
    # Build a raster that overlaps and extends past the synthetic 2D fixture's
    # extent (500_000 - 600_000 E, 4_300_000 - 4_400_000 N).
    transform = from_origin(
        west=490_000.0, north=4_410_000.0, xsize=10_000.0, ysize=10_000.0
    )
    data = np.full((n, n), value, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=n,
        width=n,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_thermal_exceedance_step_function(tmp_path: Path) -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    thermal_path = tmp_path / "thermal.tif"
    _write_thermal_raster(thermal_path, value=250.0)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        threshold=200.0,
        p_min=0.05,
        p_max=0.95,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    # Every cell has T=250 > 200 → exceedance = 1 → clipped to p_max=0.95 → logit(0.95) ≈ 2.94
    expected = np.log(0.95 / 0.05)
    assert np.allclose(result.grid_offset, expected, atol=1e-6)
    assert result.provenance["mode"] == "thermal_exceedance"


def test_thermal_exceedance_below_threshold_is_clipped_low(
    tmp_path: Path,
) -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    thermal_path = tmp_path / "thermal.tif"
    _write_thermal_raster(thermal_path, value=150.0)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        threshold=200.0,
        p_min=0.05,
        p_max=0.95,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    expected = np.log(0.05 / 0.95)
    assert np.allclose(result.grid_offset, expected, atol=1e-6)


def test_thermal_exceedance_with_uncertainty_raster_returns_smooth_prob(
    tmp_path: Path,
) -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    thermal_path = tmp_path / "thermal.tif"
    uncert_path = tmp_path / "thermal_sd.tif"
    # Mean T=200, sigma_T=50 → P(T>200) = 0.5 exactly under N(200, 50).
    _write_thermal_raster(thermal_path, value=200.0)
    _write_thermal_raster(uncert_path, value=50.0)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        threshold=200.0,
        uncertainty_raster=str(uncert_path),
        p_min=0.01,
        p_max=0.99,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    # P(T>200) = 0.5 → logit(0.5) = 0 (within rescale window)
    assert np.allclose(result.grid_offset, 0.0, atol=1e-3)
    np.testing.assert_allclose(result.latent_mean, 200.0)
    np.testing.assert_allclose(result.latent_sd, 50.0)
    assert result.event_threshold == pytest.approx(200.0)


def test_thermal_exceedance_missing_raster_raises(tmp_path: Path) -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(tmp_path / "missing.tif"),
        threshold=200.0,
    )
    with pytest.raises(FileNotFoundError):
        build_alpha_c(comp, cfg, grid_gdf=grid_gdf)


def test_thermal_exceedance_rejects_raster_without_crs(tmp_path: Path) -> None:
    comp = _component_grid_a()
    thermal_path = tmp_path / "thermal_no_crs.tif"
    _write_thermal_raster(thermal_path, crs=None)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        threshold=200.0,
    )

    with pytest.raises(ValueError, match="declare a CRS"):
        build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])


def test_thermal_exceedance_rejects_partial_raster_coverage(
    tmp_path: Path,
) -> None:
    comp = _component_grid_a()
    thermal_path = tmp_path / "partial_thermal.tif"
    _write_thermal_raster(thermal_path, n=6)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        threshold=200.0,
    )

    with pytest.raises(ValueError, match="finite value at every grid cell"):
        build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])


def test_thermal_exceedance_rejects_nonpositive_uncertainty(
    tmp_path: Path,
) -> None:
    comp = _component_grid_a()
    thermal_path = tmp_path / "thermal.tif"
    uncertainty_path = tmp_path / "thermal_sd.tif"
    _write_thermal_raster(thermal_path, value=220.0)
    _write_thermal_raster(uncertainty_path, value=0.0)
    cfg = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster=str(thermal_path),
        uncertainty_raster=str(uncertainty_path),
        threshold=200.0,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])


# ---------------------------------------------------------------------------
# multi_layer mode
# ---------------------------------------------------------------------------


def test_multi_layer_mode_sums_logits() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=10, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="multi_layer",
        layers=("prior_layer_a", "gradient"),
        p_min=0.2,
        p_max=0.8,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    # Both layers excluded from regression
    assert "prior_layer_a" in result.excluded_layer_names
    assert "gradient" in result.excluded_layer_names
    assert result.provenance["mode"] == "multi_layer"
    assert set(result.provenance["layers"]) == {"prior_layer_a", "gradient"}


def test_multi_layer_mode_missing_declared_layer_fails_closed() -> None:
    comp = _component_grid_a()
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="multi_layer",
        layers=("prior_layer_a", "missing_layer"),
        p_min=0.2,
        p_max=0.8,
    )
    with pytest.raises(ValueError, match="missing_layer"):
        build_alpha_c(comp, cfg, grid_gdf=grid_gdf)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------


def test_layer_logit_works_on_3d_voxel_grid() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=5, grid_nz=4, n_wells=15, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"]
    cfg = AlphaModeConfig(
        mode="layer_logit",
        layer="prior_layer_a",
        p_min=0.2,
        p_max=0.8,
    )
    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)
    # One offset per voxel
    assert len(result.grid_offset) == len(grid_gdf)
    lo, hi = np.log(0.2 / 0.8), np.log(0.8 / 0.2)
    assert result.grid_offset.min() >= lo - 1e-9
    assert result.grid_offset.max() <= hi + 1e-9


def test_layer_logit_preserves_depth_variation_on_3d_voxel_grid() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=4, n_wells=15, seed=3)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"]
    layer_values = comp["layers"]["prior_layer_a"]["model"][
        "value_interpolated"
    ].to_numpy(dtype=float)
    cfg = AlphaModeConfig(
        mode="layer_logit",
        layer="prior_layer_a",
        p_min=0.2,
        p_max=0.8,
    )

    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)

    scaled = cfg.p_min + (layer_values - layer_values.min()) / (
        layer_values.max() - layer_values.min()
    ) * (cfg.p_max - cfg.p_min)
    expected = np.log(scaled / (1.0 - scaled))
    np.testing.assert_allclose(result.grid_offset, expected)


def test_thermal_layer_exceedance_preserves_voxel_depth_variation() -> None:
    fixture = make_synthetic_pfa_3d(grid_n=3, grid_nz=4, n_wells=10, seed=4)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"]
    thermal = comp["layers"]["prior_layer_a"]["model"]
    z = np.asarray([point.z for point in thermal.geometry], dtype=float)
    z_scaled = (z - z.min()) / (z.max() - z.min())
    thermal["value_interpolated"] = 350.0 + 100.0 * z_scaled
    thermal["temperature_sd_c"] = 25.0
    cfg = AlphaModeConfig(
        mode="thermal_layer_exceedance",
        layer="prior_layer_a",
        threshold=400.0,
        uncertainty_column="temperature_sd_c",
        p_min=0.001,
        p_max=0.999,
    )

    result = build_alpha_c(comp, cfg, grid_gdf=grid_gdf)

    probability = 1.0 / (1.0 + np.exp(-result.grid_offset))
    np.testing.assert_allclose(
        result.latent_mean,
        thermal["value_interpolated"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(result.latent_sd, 25.0)
    assert result.event_threshold == pytest.approx(400.0)
    assert result.excluded_layer_names == {"prior_layer_a"}
    assert np.ptp(probability) > 0.9
    for level in np.unique(z):
        level_values = probability[z == level]
        np.testing.assert_allclose(level_values, level_values[0])


def test_thermal_layer_exceedance_requires_positive_voxel_uncertainty() -> (
    None
):
    fixture = make_synthetic_pfa_3d(grid_n=3, grid_nz=3, n_wells=10, seed=5)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = comp["layers"]["prior_layer_a"]["model"]
    thermal["temperature_sd_c"] = 20.0
    thermal.loc[thermal.index[0], "temperature_sd_c"] = 0.0
    cfg = AlphaModeConfig(
        mode="thermal_layer_exceedance",
        layer="prior_layer_a",
        threshold=400.0,
        uncertainty_column="temperature_sd_c",
    )

    with pytest.raises(ValueError, match="strictly positive"):
        build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])


def test_thermal_layer_exceedance_retains_rare_normal_tail_probability() -> (
    None
):
    fixture = make_synthetic_pfa_3d(grid_n=2, grid_nz=2, n_wells=8, seed=6)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = comp["layers"]["prior_layer_a"]["model"]
    thermal["value_interpolated"] = 320.0
    thermal["temperature_sd_c"] = 10.0
    cfg = AlphaModeConfig(
        mode="thermal_layer_exceedance",
        layer="prior_layer_a",
        threshold=400.0,
        uncertainty_column="temperature_sd_c",
        p_min=1e-18,
        p_max=1.0 - 1e-12,
    )

    result = build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])

    probability = 1.0 / (1.0 + np.exp(-result.grid_offset))
    # sf(8) is positive (~6.22e-16); 1-cdf(8) loses avoidable precision.
    assert np.all(probability > cfg.p_min)
    np.testing.assert_allclose(probability, 6.22096057427174e-16, rtol=1e-8)


# ---------------------------------------------------------------------------
# Provenance writer
# ---------------------------------------------------------------------------


def test_write_alpha_provenance_creates_json(tmp_path: Path) -> None:
    cfg = AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.3)
    comp = _component_grid_a()
    result = build_alpha_c(comp, cfg, grid_gdf=comp["pr_norm"])
    out_path = tmp_path / "alpha_provenance.json"
    write_alpha_provenance({"heat": result}, out_path)
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert "heat" in payload
    assert payload["heat"]["mode"] == "scalar"


def test_alpha_mode_config_rejects_scalar_pr0_zero() -> None:
    """scalar_fallback_pr0=0.0 would produce logit(0)=-inf."""
    import pytest as _p

    with _p.raises(ValueError, match="scalar_fallback_pr0"):
        AlphaModeConfig.from_dict(
            {"mode": "scalar", "scalar_fallback_pr0": 0.0},
            component_name="heat",
        )


def test_alpha_mode_config_rejects_scalar_pr0_one() -> None:
    """scalar_fallback_pr0=1.0 would produce logit(1)=+inf."""
    import pytest as _p

    with _p.raises(ValueError, match="scalar_fallback_pr0"):
        AlphaModeConfig.from_dict(
            {"mode": "scalar", "scalar_fallback_pr0": 1.0},
            component_name="heat",
        )


def test_alpha_mode_config_accepts_valid_scalar_pr0() -> None:
    cfg = AlphaModeConfig.from_dict(
        {"mode": "scalar", "scalar_fallback_pr0": 0.3}, component_name="heat"
    )
    assert cfg.scalar_fallback_pr0 == pytest.approx(0.3)
