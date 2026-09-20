"""End-to-end 3D probabilistic workflow integration test.

Uses the synthetic 3D fixture (no real data) so it runs in CI.
Validates the full pipeline: alpha_c → sequential fitting → calibration → IO.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
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
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob_3d import make_synthetic_pfa_3d


@pytest.fixture(scope="module")
def fixture_3d():
    return make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=30, seed=99)


def _make_3d_cfg(wells_path: Path, out_dir: Path) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=out_dir,
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
            pu_mode="off",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            ),
            "component_b": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.4
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )


def test_run_probabilistic_3d_returns_result(
    tmp_path: Path, fixture_3d
) -> None:
    """Full 3D pipeline runs without errors and returns valid probabilities."""
    wells_path = tmp_path / "wells.gpkg"
    fixture_3d.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _make_3d_cfg(wells_path, tmp_path / "out")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture_3d.pfa, cfg)

    assert not result.skipped
    assert len(result.components) >= 1
    for name, comp in result.components.items():
        probs = comp.probability["probability"].to_numpy()
        assert (probs >= 0).all() and (probs <= 1).all(), (
            f"{name}: probabilities out of [0,1]"
        )
        assert np.all(np.isfinite(probs)), f"{name}: non-finite probabilities"


def test_run_probabilistic_3d_combined_surface(
    tmp_path: Path, fixture_3d
) -> None:
    """Combined surface is non-trivial when multiple components are fit."""
    wells_path = tmp_path / "wells.gpkg"
    fixture_3d.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _make_3d_cfg(wells_path, tmp_path / "out2")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture_3d.pfa, cfg)

    assert result.combined is not None
    combined_probs = result.combined["probability"].to_numpy()
    assert (combined_probs >= 0).all() and (combined_probs <= 1).all()
    # Combined surface should vary (not all the same value)
    assert combined_probs.std() > 0.001


def test_embedded_probabilistic_config_integration(
    tmp_path: Path, fixture_3d
) -> None:
    """An embedded config parses and runs through the canonical API."""
    wells_path = tmp_path / "wells.gpkg"
    fixture_3d.wells.to_file(wells_path, layer="wells", driver="GPKG")

    # Embed config inside pfa dict
    pfa_with_config = dict(fixture_3d.pfa)
    pfa_with_config["probabilistic"] = {
        "enabled": True,
        "output_dir": str(tmp_path / "out3"),
        "dimensions": "3d",
        "labels": {
            "source": str(wells_path),
            "id_col": "well_id",
            "label_columns": {"component_a": "heat_label"},
            "layer": "wells",
        },
        "alpha": {
            "component_a": {"mode": "scalar", "scalar_fallback_pr0": 0.5}
        },
        "calibration": {"method": "none"},
        "combination": {"rule": "product"},
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(
            pfa_with_config,
            ProbabilisticConfig.from_pfa(pfa_with_config),
        )

    assert not result.skipped
    assert "component_a" in result.components
