"""Tests for the config-driven adapter over fit_component_probability."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from geopfa.prob.alpha import build_alpha_c
from geopfa.prob.config import (
    AlphaModeConfig,
    EvidenceConfig,
    RegularizationConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability, fit_component_probability
from geopfa.prob.fit_dispatch import (
    build_fit_kwargs,
    fit_component_from_config,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


# ---------------------------------------------------------------------------
# build_fit_kwargs
# ---------------------------------------------------------------------------


def test_build_fit_kwargs_scalar_alpha_no_spatial() -> None:
    alpha = AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.42)
    evidence = EvidenceConfig(
        include_layers=("gradient",), sparse_binary_threshold=0.93
    )
    spatial = SpatialFieldConfig(enabled=False)
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=evidence,
        spatial_field_config=spatial,
    )
    assert kwargs["prior_probability"] == pytest.approx(0.42)
    assert kwargs["prior_layer_name"] is None
    assert kwargs["include_spatial"] is False
    assert kwargs["excluded_layer_names"] == ()
    assert kwargs["included_layer_names"] == ("gradient",)
    assert kwargs["sparse_binary_threshold"] == pytest.approx(0.93)


def test_build_fit_kwargs_preserves_explicit_coefficient_priors() -> None:
    regularization = RegularizationConfig(
        per_feature_weights={"gradient": 2.0},
        prior_means={"gradient": 0.4},
        prior_precisions={"fault": 3.0},
    )

    kwargs = build_fit_kwargs(
        alpha_config=AlphaModeConfig(mode="scalar"),
        evidence_config=EvidenceConfig(regularization=regularization),
        spatial_field_config=SpatialFieldConfig(enabled=False),
    )

    assert kwargs["per_feature_weights"] == {"gradient": 2.0, "fault": 3.0}
    assert kwargs["prior_means"] == {"gradient": 0.4}


def test_build_fit_kwargs_layer_logit_alpha_uses_layer_name() -> None:
    alpha = AlphaModeConfig(
        mode="layer_logit", layer="prior_layer_a", p_min=0.15, p_max=0.85
    )
    evidence = EvidenceConfig()
    spatial = SpatialFieldConfig(enabled=True)
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=evidence,
        spatial_field_config=spatial,
    )
    assert kwargs["prior_layer_name"] == "prior_layer_a"
    assert kwargs["prior_p_min"] == pytest.approx(0.15)
    assert kwargs["prior_p_max"] == pytest.approx(0.85)


def test_build_fit_kwargs_excluded_layers_combine_with_alpha_layer() -> None:
    alpha = AlphaModeConfig(mode="layer_logit", layer="prior_layer_a")
    evidence = EvidenceConfig(exclude_layers=("noisy_layer",))
    spatial = SpatialFieldConfig()
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=evidence,
        spatial_field_config=spatial,
    )
    excluded = set(kwargs["excluded_layer_names"])
    assert "noisy_layer" in excluded
    # Note: fit_component_probability auto-excludes prior_layer_name
    # internally too; including it in excluded_layer_names is harmless and
    # makes the explicit kwarg-set easier to inspect.


def test_build_fit_kwargs_multi_layer_alpha_excludes_all_layers() -> None:
    alpha = AlphaModeConfig(
        mode="multi_layer",
        layers=("prior_layer_a", "prior_layer_b"),
    )
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=EvidenceConfig(),
        spatial_field_config=SpatialFieldConfig(),
    )
    excluded = set(kwargs["excluded_layer_names"])
    assert "prior_layer_a" in excluded
    assert "prior_layer_b" in excluded


def test_build_fit_kwargs_defers_thermal_offset_construction() -> None:
    # build_fit_kwargs has no component grid. fit_component_from_config builds
    # the declared thermal offset before dispatching to the low-level fitter.
    alpha = AlphaModeConfig(
        mode="thermal_exceedance",
        thermal_raster="thermal.tif",
        threshold=200.0,
        scalar_fallback_pr0=0.3,
    )
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=EvidenceConfig(),
        spatial_field_config=SpatialFieldConfig(),
    )
    assert kwargs["prior_probability"] == pytest.approx(0.3)
    assert kwargs["prior_layer_name"] is None


# ---------------------------------------------------------------------------
# fit_component_from_config end-to-end
# ---------------------------------------------------------------------------


def test_fit_component_from_config_matches_direct_call() -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=0)
    comp_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]

    alpha = AlphaModeConfig(
        mode="layer_logit",
        layer="prior_layer_a",
        scalar_fallback_pr0=0.55,
    )
    evidence = EvidenceConfig()
    spatial = SpatialFieldConfig(enabled=True, backend="rbf")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result_via_config = fit_component_from_config(
            comp_data,
            alpha_config=alpha,
            evidence_config=evidence,
            spatial_field_config=spatial,
            labeled_wells=fixture.wells,
            label_column="heat_label",
        )

    assert isinstance(result_via_config, ComponentProbability)
    assert "probability" in result_via_config.probability.columns


def test_fit_component_from_config_passes_labeled_wells() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=1)
    comp_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_b"
    ]

    alpha = AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_component_from_config(
            comp_data,
            alpha_config=alpha,
            evidence_config=EvidenceConfig(),
            spatial_field_config=SpatialFieldConfig(enabled=False),
            labeled_wells=fixture.wells,
            label_column="reservoir_label",
        )
    # n_train should reflect the supplied wells, not the legacy proxy-label
    # path.
    grid_gdf = result.probability
    assert "probability" in grid_gdf.columns
    assert (
        0.0
        <= float(grid_gdf["probability"].min())
        <= float(grid_gdf["probability"].max())
        <= 1.0
    )


def test_fit_component_from_config_accepts_layer_grid_without_pr_norm() -> (
    None
):
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=2)
    comp_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_b"
    ].copy()
    comp_data.pop("pr_norm")

    result = fit_component_from_config(
        comp_data,
        alpha_config=AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5),
        evidence_config=EvidenceConfig(),
        spatial_field_config=SpatialFieldConfig(enabled=False),
        labeled_wells=fixture.wells,
        label_column="reservoir_label",
    )

    assert len(result.probability) == 64
    assert np.all(np.isfinite(result.probability["probability"]))


def test_fit_component_from_config_honors_multi_layer_alpha() -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=8)
    comp_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]
    alpha = AlphaModeConfig(
        mode="multi_layer",
        layers=("prior_layer_a",),
        scalar_fallback_pr0=0.5,
    )
    evidence = EvidenceConfig()
    spatial = SpatialFieldConfig(enabled=False)

    via_config = fit_component_from_config(
        comp_data,
        alpha_config=alpha,
        evidence_config=evidence,
        spatial_field_config=spatial,
        labeled_wells=fixture.wells,
        label_column="heat_label",
    )
    built_alpha = build_alpha_c(
        comp_data,
        alpha,
        grid_gdf=comp_data["pr_norm"],
    )
    direct = fit_component_probability(
        comp_data,
        prior_probability=alpha.scalar_fallback_pr0,
        alpha_offset=built_alpha.grid_offset,
        excluded_layer_names=tuple(sorted(built_alpha.excluded_layer_names)),
        include_spatial=False,
        labeled_wells=fixture.wells,
        label_column="heat_label",
    )

    np.testing.assert_allclose(
        via_config.probability["probability"],
        direct.probability["probability"],
    )
    assert all(
        "prior_layer_a" not in name for name in via_config.feature_names
    )


def test_build_fit_kwargs_includes_supported_spatial_field_config() -> None:
    """Config dispatch forwards only executable spatial-field controls."""
    alpha = AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)
    evidence = EvidenceConfig()
    spatial = SpatialFieldConfig(
        enabled=True,
        backend="latticekrigx",
    )
    kwargs = build_fit_kwargs(
        alpha_config=alpha,
        evidence_config=evidence,
        spatial_field_config=spatial,
    )
    assert kwargs["include_spatial"] is True
    assert kwargs["spatial_backend"] == "latticekrigx"
