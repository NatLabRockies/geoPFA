"""Tests for 3D probabilistic-method end-to-end."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from geopfa.prob.config import (
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
from geopfa.prob.runner import ProbabilisticResult, run_probabilistic
from tests.fixtures.synthetic_prob_3d import make_synthetic_pfa_3d


def _3d_config(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
        dimensions="3d",
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
        spatial_field=SpatialFieldConfig(enabled=True, backend="rbf"),
        inference=InferenceConfig(backend="sequential"),
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
            format=("csv",),
        ),
    )


def test_run_probabilistic_3d_end_to_end(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa_3d(grid_n=5, grid_nz=4, n_wells=25, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    cfg = _3d_config(wells_path, tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)

    assert isinstance(result, ProbabilisticResult)
    assert set(result.components.keys()) == {"component_a", "component_b"}
    for surface in result.components.values():
        gdf = surface.probability
        assert gdf.geometry.iloc[0].has_z
        prob = gdf["probability"].to_numpy()
        assert (prob >= 0).all() and (prob <= 1).all()
    # Combined surface preserves Z geometry too
    assert result.combined.geometry.iloc[0].has_z


def test_run_probabilistic_3d_writes_csv_with_z_column(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=15, seed=1)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    out_dir = tmp_path / "out"
    cfg = _3d_config(wells_path, out_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)

    df = pd.read_csv(out_dir / "component_a_probability.csv")
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns
    assert "probability" in df.columns
    # Z values should be within the 3D fixture's depth range
    z_min, z_max = -3000.0, -500.0
    assert df["z"].min() >= z_min
    assert df["z"].max() <= z_max
