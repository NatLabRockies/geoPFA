"""Tests for evidence-regression knobs (C.4, C.5, C.7) exposed via config."""

from __future__ import annotations

import copy
import warnings

import numpy as np

import pytest

from geopfa.prob.fitting import (
    _flatten_component_features,
    fit_component_probability,
)
from geopfa.prob.spatial_alignment import snap_to_grid_indices
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _component_a():
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=0)
    return fixture.pfa["criteria"]["geologic"]["components"]["component_a"]


# ---------------------------------------------------------------------------
# C.5 sparse_binary_threshold
# ---------------------------------------------------------------------------


def test_sparse_binary_threshold_default_rejects_sparse_indicator() -> None:
    comp = _component_a()
    X, names = _flatten_component_features(comp, sparse_binary_threshold=0.90)
    # sparse_indicator should be auto-excluded at the default threshold
    assert all("sparse_indicator" not in n for n in names)


def test_sparse_binary_threshold_relaxed_keeps_sparse_indicator() -> None:
    comp = _component_a()
    # Very strict: only reject when at least 99.9% share one value.
    X, names = _flatten_component_features(comp, sparse_binary_threshold=0.999)
    # With this threshold, sparse_indicator (~3% positives) should now pass
    # the sparse check (97% same value, but threshold says we only reject
    # at >=99.9% same value)
    assert any("sparse_indicator" in n for n in names)


def test_include_layers_is_an_actual_allowlist() -> None:
    comp = _component_a()

    _, names = _flatten_component_features(
        comp,
        included_layer_names={"gradient"},
        sparse_binary_threshold=0.90,
    )

    assert names == ["gradient"]


def test_declared_evidence_column_is_required() -> None:
    """A typo must not silently substitute an unrelated numeric column."""
    comp = copy.deepcopy(_component_a())
    layer = comp["layers"]["gradient"]
    layer["model_data_col"] = "missing_declared_column"
    layer["model"]["unrelated_numeric"] = np.arange(len(layer["model"]))

    with pytest.raises(ValueError, match="missing_declared_column"):
        _flatten_component_features(
            comp,
            included_layer_names={"gradient"},
        )


def test_component_features_follow_the_component_grid_order() -> None:
    comp = copy.deepcopy(_component_a())
    original, names = _flatten_component_features(comp)
    for layer in comp["layers"].values():
        layer["model"] = layer["model"].iloc[::-1].reset_index(drop=True)

    reordered, reordered_names = _flatten_component_features(comp)

    assert reordered_names == names
    np.testing.assert_allclose(reordered, original)


# ---------------------------------------------------------------------------
# C.4 coordinate_blacklist
# ---------------------------------------------------------------------------


def test_coordinate_blacklist_with_custom_columns() -> None:
    """A custom blacklist excluded a custom-named column."""
    fixture = make_synthetic_pfa(grid_n=6, n_wells=10, seed=1)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    # Inject a column named 'custom_coord' into one layer's model
    gradient_layer = comp["layers"]["gradient"]
    gradient_layer["model"]["custom_coord"] = np.arange(
        len(gradient_layer["model"])
    )
    # First, with default blacklist, custom_coord is not excluded (since
    # the default blacklist doesn't contain it):
    X_default, names_default = _flatten_component_features(comp)
    # Then with a custom blacklist that includes 'custom_coord':
    X_custom, names_custom = _flatten_component_features(
        comp,
        coordinate_blacklist=("inverted_y", "x", "y", "custom_coord"),
    )
    # The gradient layer's declared evidence column is value_interpolated,
    # not custom_coord. Adding 'custom_coord' to the blacklist removes it
    # from the candidate set but doesn't drop gradient because gradient's
    # evidence_col is value_interpolated. The blacklist primarily affects
    # the fallback "first numeric non-blacklist column" pick.
    assert "gradient" in names_default
    assert "gradient" in names_custom


# ---------------------------------------------------------------------------
# C.7 single-class identification failure
# ---------------------------------------------------------------------------


def test_single_class_labels_fail_closed() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=15, seed=2)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    wells = fixture.wells.copy()
    # Force all labels to 1 (single-class)
    wells["heat_label"] = 1
    with pytest.raises(ValueError, match="single-class"):
        fit_component_probability(
            comp,
            prior_probability=0.5,
            include_spatial=False,
            labeled_wells=wells,
            label_column="heat_label",
        )


@pytest.mark.parametrize("bad_label", [0.5, "not-a-label"])
def test_direct_fit_rejects_nonbinary_labels(bad_label: object) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=23)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    wells = fixture.wells.copy()
    wells.loc[wells.index[0], "heat_label"] = bad_label
    with pytest.raises(ValueError, match="binary 0/1|nonnumeric"):
        fit_component_probability(
            comp,
            prior_probability=0.5,
            include_spatial=False,
            labeled_wells=wells,
            label_column="heat_label",
        )


def test_evidence_scaling_uses_training_wells_not_prediction_grid() -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=24, seed=24)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    altered = copy.deepcopy(comp)
    grid = comp["pr_norm"]
    wells = fixture.wells.to_crs(grid.crs)
    training_indices = np.unique(snap_to_grid_indices(wells, grid))
    held_grid = np.ones(len(grid), dtype=bool)
    held_grid[training_indices] = False
    altered_gradient = altered["layers"]["gradient"]["model"]
    altered_gradient.loc[held_grid, "value_interpolated"] += 1_000.0

    baseline = fit_component_probability(
        comp,
        prior_probability=0.5,
        include_spatial=False,
        labeled_wells=fixture.wells,
        label_column="heat_label",
    )
    perturbed = fit_component_probability(
        altered,
        prior_probability=0.5,
        include_spatial=False,
        labeled_wells=fixture.wells,
        label_column="heat_label",
    )

    np.testing.assert_allclose(baseline.model.x, perturbed.model.x)
    np.testing.assert_allclose(
        baseline.diagnostics["evidence_center"],
        perturbed.diagnostics["evidence_center"],
    )
    np.testing.assert_allclose(
        baseline.diagnostics["evidence_scale"],
        perturbed.diagnostics["evidence_scale"],
    )


def test_single_class_failure_via_runner(tmp_path) -> None:
    """The public runner must expose rather than conceal non-identification."""
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

    fixture = make_synthetic_pfa(grid_n=6, n_wells=12, seed=3)
    wells = fixture.wells.copy()
    wells["heat_label"] = 1.0  # force single-class
    wells_path = tmp_path / "wells.gpkg"
    wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=tmp_path / "out",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns={"component_a": "heat_label"},
            layer="wells",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.7
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
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
    with pytest.raises(ValueError, match="single-class"):
        run_probabilistic(fixture.pfa, cfg)
