"""Unit tests for :mod:`geopfa.prob.fitting` using a synthetic two-component fixture.

Covers:

* explicit prior-predictive path (no labelled wells)
* labelled-well fit path (the recommended Stage-1 mode)
* automatic exclusion of sparse-binary and coordinate-like layer columns
* spatial residual smoother attached at well points and projected to grid
* spatial-prior offset path (`prior_layer_name`) and double-counting protection
* combined probability via the product rule
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from shapely.geometry import Point

from geopfa.prob import calibration
from geopfa.prob.fitting import (
    ComponentProbability,
    _flatten_component_features,
    _is_sparse_binary,
    combine_probability_surfaces,
    fit_component_probability,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# Helpers and small inline fixtures
# ---------------------------------------------------------------------------


def _toy_component(extra_cols: dict | None = None) -> dict:
    """Tiny 4-cell component used by quick legacy / sanity tests."""
    geometry = [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)]
    base = gpd.GeoDataFrame(
        {"favorability": [0.2, 1.5, 3.0, 4.5], "geometry": geometry},
        geometry="geometry",
    )
    layer_a_df = gpd.GeoDataFrame(
        {"value_interpolated": [0.1, 0.4, 0.7, 1.0], "geometry": geometry},
        geometry="geometry",
    )
    if extra_cols:
        for k, v in extra_cols.items():
            layer_a_df[k] = v
    layer_b_df = gpd.GeoDataFrame(
        {"value_interpolated": [1.0, 0.5, 0.2, 0.0], "geometry": geometry},
        geometry="geometry",
    )
    return {
        "pr0": 0.4,
        "pr_norm": base,
        "layers": {
            "layer_a": {
                "model": layer_a_df,
                "model_data_col": "value_interpolated",
            },
            "layer_b": {
                "model": layer_b_df,
                "model_data_col": "value_interpolated",
            },
        },
    }


# ---------------------------------------------------------------------------
# Sparse-binary / coordinate-leak protection
# ---------------------------------------------------------------------------


def test_is_sparse_binary_detects_dominant_value() -> None:
    sparse = np.concatenate([np.zeros(95), np.ones(5)])
    assert bool(_is_sparse_binary(sparse)) is True


def test_is_sparse_binary_rejects_continuous_values() -> None:
    rng = np.random.default_rng(0)
    continuous = rng.uniform(0.0, 1.0, size=100)
    assert bool(_is_sparse_binary(continuous)) is False


def test_is_sparse_binary_rejects_balanced_binary() -> None:
    balanced = np.concatenate([np.zeros(50), np.ones(50)])
    assert bool(_is_sparse_binary(balanced)) is False


def test_flatten_component_features_excludes_coord_columns() -> None:
    component_data = _toy_component(
        extra_cols={
            "inverted_y": [10.0, 20.0, 30.0, 40.0],
            "X": [0.0, 1.0, 0.0, 1.0],
        }
    )
    X, names = _flatten_component_features(component_data)
    # Coordinate columns must never enter the regression matrix.
    for n in names:
        assert "inverted_y" not in n
    # Each layer contributes its declared `model_data_col` only.
    assert X.shape[1] == len(names) == 2
    assert names == ["layer_a", "layer_b"]


def test_flatten_component_features_excludes_sparse_binary_layer() -> None:
    """Layers whose declared evidence column is >90% the same value should drop out."""
    n_cells = 100
    geometry = [Point(i % 10, i // 10) for i in range(n_cells)]
    base = gpd.GeoDataFrame(
        {"favorability": np.linspace(0.0, 5.0, n_cells), "geometry": geometry},
        geometry="geometry",
    )
    layer_dense = gpd.GeoDataFrame(
        {
            "value_interpolated": np.linspace(0.0, 1.0, n_cells),
            "geometry": geometry,
        },
        geometry="geometry",
    )
    sparse_values = np.zeros(n_cells)
    sparse_values[:5] = 1.0  # 95% zeros, 5% ones → sparse-binary
    layer_sparse = gpd.GeoDataFrame(
        {"value_interpolated": sparse_values, "geometry": geometry},
        geometry="geometry",
    )
    comp = {
        "pr0": 0.4,
        "pr_norm": base,
        "layers": {
            "dense_layer": {
                "model": layer_dense,
                "model_data_col": "value_interpolated",
            },
            "sparse_layer": {
                "model": layer_sparse,
                "model_data_col": "value_interpolated",
            },
        },
    }
    X, names = _flatten_component_features(comp)
    assert all("sparse_layer" not in n for n in names)
    assert X.shape[1] == 1


# ---------------------------------------------------------------------------
# Missing-label and explicit prior-predictive contracts
# ---------------------------------------------------------------------------


def test_missing_labels_fail_closed() -> None:
    with pytest.raises(
        ValueError, match="labeled_wells and label_column are required"
    ):
        fit_component_probability(_toy_component(), prior_probability=0.4)


def test_explicit_prior_predictive_returns_unit_interval_probabilities() -> (
    None
):
    result = fit_component_probability(
        _toy_component(), prior_probability=0.4, force_prior_predictive=True
    )
    assert isinstance(result, ComponentProbability)
    p = result.probability["probability"].to_numpy()
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_prior_predictive_does_not_claim_a_spatial_fit() -> None:
    result = fit_component_probability(
        _toy_component(), prior_probability=0.4, force_prior_predictive=True
    )
    assert result.model is None
    assert result.spatial_field is None
    assert result.diagnostics["inference_role"] == "prior_predictive"


# ---------------------------------------------------------------------------
# Combine helper
# ---------------------------------------------------------------------------


def test_combine_uses_product_rule() -> None:
    out = combine_probability_surfaces(
        [np.array([0.5, 0.25]), np.array([0.2, 0.4])]
    )
    np.testing.assert_allclose(out, [0.1, 0.1])


def test_combine_raises_when_no_components_provided() -> None:
    with pytest.raises(ValueError):
        combine_probability_surfaces([])


@given(
    st.integers(min_value=1, max_value=20),
    st.integers(min_value=1, max_value=4),
    st.data(),
)
@settings(max_examples=25, deadline=None)
def test_combine_probability_surfaces_stays_bounded(
    n_cells: int, n_components: int, data
) -> None:
    surfaces = [
        np.asarray(
            data.draw(
                st.lists(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=n_cells,
                    max_size=n_cells,
                ),
                label=f"surface_{i}",
            ),
            dtype=float,
        )
        for i in range(n_components)
    ]
    out = combine_probability_surfaces(surfaces)
    assert out.shape == (n_cells,)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1),
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
        ),
        min_size=1,
        max_size=40,
    ),
)
@settings(max_examples=25, deadline=None)
def test_fit_temperature_preserves_unit_interval(
    pairs: list[tuple[int, float]],
) -> None:
    y = np.asarray([pair[0] for pair in pairs], dtype=int)
    p = np.asarray([pair[1] for pair in pairs], dtype=float)
    T, p_T = calibration.fit_temperature(y, p)
    assert T > 0.0
    assert p_T.shape == p.shape
    assert np.all(p_T >= 0.0)
    assert np.all(p_T <= 1.0)


# ---------------------------------------------------------------------------
# Labelled-well fit path on the synthetic fixture
# ---------------------------------------------------------------------------


def test_labelled_well_fit_returns_probabilities_per_grid_cell() -> None:
    syn = make_synthetic_pfa(seed=11)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    result = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        include_spatial=True,
        labeled_wells=syn.wells,
        label_column="heat_label",
    )
    p = result.probability["probability"].to_numpy()
    assert len(p) == len(comp_data["pr_norm"])
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert "spatial:rbf_residual_at_wells" in result.feature_names


def test_labelled_well_fit_outperforms_explicit_prior_at_wells() -> None:
    syn = make_synthetic_pfa(seed=12, n_wells=120, grid_n=14)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    prior = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        force_prior_predictive=True,
    )
    labelled = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        include_spatial=True,
        labeled_wells=syn.wells,
        label_column="heat_label",
    )
    wells = syn.wells.to_crs(comp_data["pr_norm"].crs)
    prior_join = gpd.sjoin_nearest(
        wells[["heat_label", "geometry"]],
        prior.probability[["probability", "geometry"]],
        how="inner",
    )
    labelled_join = gpd.sjoin_nearest(
        wells[["heat_label", "geometry"]],
        labelled.probability[["probability", "geometry"]],
        how="inner",
    )
    from geopfa.prob.decision_metrics import auc_tie_safe

    auc_prior = auc_tie_safe(
        prior_join["heat_label"].astype(int), prior_join["probability"]
    )
    auc_labelled = auc_tie_safe(
        labelled_join["heat_label"].astype(int), labelled_join["probability"]
    )
    # A fitted model should beat the explicitly identified prior-only baseline.
    assert auc_labelled >= auc_prior - 1e-3
    assert auc_labelled > 0.7  # synthetic data is highly separable


# ---------------------------------------------------------------------------
# Spatial-prior offset path
# ---------------------------------------------------------------------------


def test_spatial_prior_layer_excluded_from_regression_features() -> None:
    syn = make_synthetic_pfa(seed=13)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    result = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        labeled_wells=syn.wells,
        label_column="heat_label",
        prior_layer_name="prior_layer_a",
    )
    feature_names = [
        n for n in result.feature_names if not n.startswith("spatial:")
    ]
    assert all("prior_layer_a" not in n for n in feature_names)
    assert "prior_probability_spatial" in result.probability.columns


def test_spatial_prior_offset_is_in_expected_range() -> None:
    syn = make_synthetic_pfa(seed=14)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    result = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        labeled_wells=syn.wells,
        label_column="heat_label",
        prior_layer_name="prior_layer_a",
        prior_p_min=0.2,
        prior_p_max=0.8,
    )
    prior_p = result.probability["prior_probability_spatial"].to_numpy(
        dtype=float
    )
    assert prior_p.min() >= 0.2 - 1e-9
    assert prior_p.max() <= 0.8 + 1e-9


def test_spatial_prior_missing_layer_fails_closed() -> None:
    syn = make_synthetic_pfa(seed=15)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    with pytest.raises(ValueError, match="nonexistent_layer"):
        fit_component_probability(
            comp_data,
            prior_probability=float(comp_data["pr0"]),
            labeled_wells=syn.wells,
            label_column="heat_label",
            prior_layer_name="nonexistent_layer",
        )


def test_spatial_prior_clipped_p_bounds_remain_finite() -> None:
    syn = make_synthetic_pfa(seed=16)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    # Aggressive [0.01, 0.99] bounds; the result must still be finite.
    result = fit_component_probability(
        comp_data,
        prior_probability=float(comp_data["pr0"]),
        labeled_wells=syn.wells,
        label_column="heat_label",
        prior_layer_name="prior_layer_a",
        prior_p_min=0.01,
        prior_p_max=0.99,
    )
    assert np.all(
        np.isfinite(result.probability["probability"].to_numpy(dtype=float))
    )


# ---------------------------------------------------------------------------
# Insufficient labels raise informative error
# ---------------------------------------------------------------------------


def test_labelled_fit_fails_when_too_few_wells_overlap_grid() -> None:
    """With fewer than four wells, a fitted model is not identified."""
    syn = make_synthetic_pfa(seed=17)
    comp_data = syn.pfa["criteria"]["geologic"]["components"]["component_a"]
    one_well = syn.wells.iloc[:1].copy()
    with pytest.raises(ValueError, match="fewer than 4 labelled wells"):
        fit_component_probability(
            comp_data,
            prior_probability=float(comp_data["pr0"]),
            labeled_wells=one_well,
            label_column="heat_label",
        )
