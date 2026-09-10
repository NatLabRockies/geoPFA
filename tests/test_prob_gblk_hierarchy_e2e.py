"""End-to-end tests for hierarchical regional GBLK pooling (P7-S02).

Verifies that:
- A zero-label region receives shrinkage-based predictions pulled toward
  the play-type pool (shrinkage near 1).
- A data-rich region departs from the pool (lower shrinkage than
  zero-label region).
- The returned ProbabilisticResult satisfies the standard schema.

Requires the ``dev-gblk`` pixi environment (``latticekrigx``).
Skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

pytest.importorskip("latticekrigx.glk.joint")

from geopfa.prob.config import (  # noqa: E402
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GridConfig,
    InferenceConfig,
    LabelsConfig,
    OutputsConfig,
    ProbabilisticConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability  # noqa: E402
from geopfa.prob.gblk_runner import run_gblk_hierarchical_regional  # noqa: E402
from geopfa.prob.runner import ProbabilisticResult  # noqa: E402
from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_GRID_N = 10
_N_WELLS = 60
_NC_SMALL = 4
_OUTER_SMALL = 100
_IRLS_SMALL = 50

_X_MIN = 500_000.0
_X_MAX = 600_000.0


def _region_for_x(xs: np.ndarray) -> np.ndarray:
    """Assign 'west'/'centre'/'east' based on x-coordinate thirds."""
    third = (_X_MAX - _X_MIN) / 3.0
    regions = np.empty(len(xs), dtype=object)
    regions[xs < _X_MIN + third] = "west"
    regions[(xs >= _X_MIN + third) & (xs < _X_MIN + 2 * third)] = "centre"
    regions[xs >= _X_MIN + 2 * third] = "east"
    return regions


def _make_multiregion_fixture(
    tmp_path: Path,
) -> tuple[dict, ProbabilisticConfig, np.ndarray, np.ndarray, dict[str, str]]:
    """Build a three-region synthetic fixture.

    Regions:
      - ``"west"``   (data-rich, high prevalence): all positive labels
      - ``"centre"`` (data-rich, low prevalence): all negative labels
      - ``"east"``   (zero-label): wells exist but all labels set to NaN

    All three regions share the ``"extensional"`` play type so they pool
    together. Extreme label contrast between ``"west"`` and ``"centre"``
    ensures the between-region variance (tau2) is positive, which makes
    ``"west"`` shrinkage < 1 (verifiably lower than the zero-label
    ``"east"`` shrinkage which is near 1).
    """
    fixture = make_synthetic_pfa(grid_n=_GRID_N, n_wells=_N_WELLS, seed=42)

    well_xs = fixture.wells.geometry.x.to_numpy(dtype=float)
    region_per_well = _region_for_x(well_xs)

    wells = fixture.wells.copy()
    west_mask = region_per_well == "west"
    centre_mask = region_per_well == "centre"
    east_mask = region_per_well == "east"

    wells.loc[west_mask, "heat_label"] = 1.0
    wells.loc[west_mask, "reservoir_label"] = 1.0
    wells.loc[centre_mask, "heat_label"] = 0.0
    wells.loc[centre_mask, "reservoir_label"] = 0.0
    wells.loc[east_mask, "heat_label"] = np.nan
    wells.loc[east_mask, "reservoir_label"] = np.nan

    wells_path = tmp_path / "wells_hier.gpkg"
    wells.to_file(wells_path, layer="wells", driver="GPKG")

    grid_gdf: gpd.GeoDataFrame = next(
        iter(fixture.pfa["criteria"]["geologic"]["components"].values())
    )["pr_norm"]
    grid_xs = grid_gdf.geometry.x.to_numpy(dtype=float)
    region_per_grid = _region_for_x(grid_xs)

    play_type_per_region: dict[str, str] = {
        "west": "extensional",
        "centre": "extensional",
        "east": "extensional",
    }

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=tmp_path / "out",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns={
                "component_a": "heat_label",
                "component_b": "reservoir_label",
            },
            layer="wells",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.50,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="gblk"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=(),
        ),
    )

    return fixture.pfa, cfg, region_per_well, region_per_grid, play_type_per_region


# ---------------------------------------------------------------------------
# Return-type and schema tests
# ---------------------------------------------------------------------------


def test_returns_probabilistic_result(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    assert isinstance(result, ProbabilisticResult)


def test_result_has_both_components(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    assert set(result.components) == {"component_a", "component_b"}
    for cp in result.components.values():
        assert isinstance(cp, ComponentProbability)


def test_result_combined_surface_present(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert "probability" in result.combined.columns


# ---------------------------------------------------------------------------
# Acceptance: zero-label region receives finite shrinkage-based predictions
# ---------------------------------------------------------------------------


def test_zero_label_region_predictions_finite(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    combined = result.combined
    east_cell_mask = _region_for_x(
        combined.geometry.x.to_numpy(dtype=float)
    ) == "east"
    east_probs = combined.loc[east_cell_mask, "probability"].to_numpy(dtype=float)
    assert east_cell_mask.sum() > 0, "expected east grid cells"
    assert np.all(np.isfinite(east_probs)), (
        f"zero-label region contains non-finite predictions: {east_probs}"
    )
    assert np.all((east_probs > 0.0) & (east_probs < 1.0))


def test_zero_label_region_shrinkage_near_one(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    diag = next(iter(result.components.values())).diagnostics
    shrinkage = np.array(diag["pool_shrinkage"])  # (R, Q)
    region_names: list[str] = diag["region_names"]
    east_idx = region_names.index("east")
    east_shrinkage = shrinkage[east_idx]
    assert np.all(east_shrinkage > 0.99), (
        f"zero-label region shrinkage should be near 1, got {east_shrinkage}"
    )


# ---------------------------------------------------------------------------
# Acceptance: data-rich regions depart from the pool
# ---------------------------------------------------------------------------


def test_data_rich_region_lower_shrinkage_than_zero_label(
    tmp_path: Path,
) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    diag = next(iter(result.components.values())).diagnostics
    shrinkage = np.array(diag["pool_shrinkage"])  # (R, Q)
    region_names: list[str] = diag["region_names"]
    west_idx = region_names.index("west")
    east_idx = region_names.index("east")
    west_shrinkage_mean = float(shrinkage[west_idx].mean())
    east_shrinkage_mean = float(shrinkage[east_idx].mean())
    assert west_shrinkage_mean < east_shrinkage_mean, (
        f"data-rich 'west' shrinkage {west_shrinkage_mean:.4f} should be "
        f"less than zero-label 'east' shrinkage {east_shrinkage_mean:.4f}"
    )


def test_data_rich_region_predictions_differ_from_pool_mean(
    tmp_path: Path,
) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    combined = result.combined
    probs = combined["probability"].to_numpy(dtype=float)
    xs = combined.geometry.x.to_numpy(dtype=float)
    west_mask = _region_for_x(xs) == "west"
    east_mask = _region_for_x(xs) == "east"

    west_mean = float(probs[west_mask].mean())
    east_mean = float(probs[east_mask].mean())
    pool_mean = float(probs.mean())

    west_dev = abs(west_mean - pool_mean)
    east_dev = abs(east_mean - pool_mean)

    assert west_dev > east_dev or not np.isclose(west_mean, east_mean, atol=0.05), (
        "data-rich 'west' region should differ from pool; "
        f"west_mean={west_mean:.4f}, east_mean={east_mean:.4f}, "
        f"pool_mean={pool_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# Hierarchical diagnostics present
# ---------------------------------------------------------------------------


def test_diagnostics_contain_hierarchical_fields(tmp_path: Path) -> None:
    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    result = run_gblk_hierarchical_regional(
        pfa, cfg, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    diag = next(iter(result.components.values())).diagnostics
    assert diag.get("hierarchical") is True
    assert "pool_shrinkage" in diag
    assert "d_post" in diag
    assert "region_names" in diag


# ---------------------------------------------------------------------------
# Validation: cfg.enabled=False returns skipped result
# ---------------------------------------------------------------------------


def test_disabled_cfg_returns_skipped(tmp_path: Path) -> None:
    import dataclasses  # noqa: PLC0415

    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    cfg_off = dataclasses.replace(cfg, enabled=False)
    result = run_gblk_hierarchical_regional(
        pfa, cfg_off, rpw, rpg, ptp,
        nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
    )
    assert result.skipped is True


# ---------------------------------------------------------------------------
# Validation: unknown region label raises
# ---------------------------------------------------------------------------


def test_unknown_region_in_grid_raises(tmp_path: Path) -> None:
    from geopfa.exceptions import GEOPFAValueError  # noqa: PLC0415

    pfa, cfg, rpw, rpg, ptp = _make_multiregion_fixture(tmp_path)
    bad_rpg = rpg.copy()
    bad_rpg[0] = "unknown_region"
    with pytest.raises(GEOPFAValueError, match="region_per_grid"):
        run_gblk_hierarchical_regional(
            pfa, cfg, rpw, bad_rpg, ptp,
            nc=_NC_SMALL, max_outer_iter=_OUTER_SMALL, irls_max_iter=_IRLS_SMALL,
        )
