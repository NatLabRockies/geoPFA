"""Tests for the optional Bayesian GBLK path (P7-S01).

Requires the ``dev-gblk`` pixi environment (``latticekrigx`` +
``scikit-sparse``).  Skipped otherwise.
"""

from __future__ import annotations

from copy import deepcopy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit, ndtr

pytest.importorskip("latticekrigx.glk.bayes.paige")

from geopfa.exceptions import GEOPFAValueError  # noqa: E402
from geopfa.prob import gblk_runner  # noqa: E402
from geopfa.prob.config import (  # noqa: E402
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GBLKBayesianConfig,
    GridConfig,
    InferenceConfig,
    KleiberProfileConfig,
    LabelsConfig,
    ObservationModelConfig,
    OutputsConfig,
    PredictiveStackingConfig,
    ProbabilisticConfig,
    RegularizationConfig,
    ScenarioConfig,
    SpatialFieldConfig,
)
from geopfa.prob.fitting import ComponentProbability  # noqa: E402
from geopfa.prob.gblk_backend import (  # noqa: E402
    GBLKBayesianFitResult,
    GBLKBayesianPosteriorState,
    GBLKGaussianFitResult,
    build_lkinfo_2d,
    build_paige_prior,
    fit_gblk_bayesian_joint,
    fit_gblk_gaussian_bayesian_joint,
    project_gblk_bayesian_draw_block,
)
from geopfa.prob.gblk_runner import (  # noqa: E402
    _GaussianPredictiveResponseState,
    _apply_componentwise_stacking,
    _assemble_likelihood_groups,
    _blocked_family_predictions,
    _estimate_predictive_stacking,
    _gaussian_prior_exceedance_probability,
    _gaussian_predictive_exceedance_draws,
    _gaussian_predictive_log_density,
    _gaussian_predictive_response_summary,
    _grouped_spatial_folds,
    _prepare_joint_evidence_arrays,
    _prior_predictive_evidence_state,
    _select_component_stacking,
    _spawn_child_seeds,
    _stacking_training_minimum,
    _stacking_validation_mask,
    run_gblk_probabilistic,
)
from geopfa.prob.predictive_stacking import (  # noqa: E402
    PredictiveStackingResult,
)
from geopfa.prob.runner import (  # noqa: E402
    ProbabilisticResult,
    run_probabilistic,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: E402


_COMPONENTS = ("component_a", "component_b")
_NC_SMALL = 3
_OUTER_SMALL = 100
_IRLS_SMALL = 50
_N_DRAWS = 8


def test_spawned_bayesian_seeds_fit_legacy_numpy_seed_domain() -> None:
    seeds = _spawn_child_seeds(73_000, 100)

    assert seeds == _spawn_child_seeds(73_000, 100)
    assert len(set(seeds)) == 100
    assert min(seeds) >= 0
    assert max(seeds) <= np.iinfo(np.uint32).max


def test_fitted_and_prior_components_share_one_child_seed_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=61)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        alpha={
            "component_a": base.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.5,
                force_prior_predictive=True,
                use_evidence_prior=True,
            ),
        },
        evidence=EvidenceConfig(
            include_layers=("gradient",),
            regularization=RegularizationConfig(
                prior_means={"component_b:gradient": 0.5},
                prior_precisions={"component_b:gradient": 4.0},
            ),
        ),
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
        ),
    )
    captured: dict[str, int] = {}
    delegated_backend = gblk_runner.fit_gblk_bayesian_joint

    def capture_fitted_seed(*args, bayes_config, **kwargs):
        captured["fitted"] = bayes_config.seed
        return delegated_backend(*args, bayes_config=bayes_config, **kwargs)

    def capture_prior_seed(
        _adapter,
        _name,
        _alpha,
        prior_cfg,
        *,
        seed,
        reference_grid,
    ):
        captured["prior"] = seed
        return SimpleNamespace(
            probability_draws=np.full(
                (
                    prior_cfg.inference.gblk_bayesian.n_draws,
                    len(reference_grid),
                ),
                0.5,
            ),
            feature_names=("gradient",),
            diagnostics={},
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint",
        capture_fitted_seed,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._prior_predictive_evidence_draws",
        capture_prior_seed,
    )

    run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    expected = _spawn_child_seeds(base.inference.gblk_bayesian.seed, 2)
    assert captured == {"fitted": expected[0], "prior": expected[1]}


def test_streamed_fitted_and_prior_components_share_one_child_seed_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=62)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        alpha={
            "component_a": base.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="scalar",
                scalar_fallback_pr0=0.5,
                force_prior_predictive=True,
                use_evidence_prior=True,
            ),
        },
        evidence=EvidenceConfig(
            include_layers=("gradient",),
            regularization=RegularizationConfig(
                prior_means={"component_b:gradient": 0.5},
                prior_precisions={"component_b:gradient": 4.0},
            ),
        ),
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                cluster_effect=False,
                kleiber_profiles={},
            ),
        ),
        outputs=replace(
            base.outputs,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
        ),
    )
    captured: dict[str, int] = {}
    delegated_backend = gblk_runner.fit_gblk_bayesian_posterior_state
    delegated_prior = _prior_predictive_evidence_state

    def capture_fitted_seed(*args, bayes_config, **kwargs):
        captured["fitted"] = bayes_config.seed
        return delegated_backend(*args, bayes_config=bayes_config, **kwargs)

    def capture_prior_seed(*args, seed, **kwargs):
        captured["prior"] = seed
        return delegated_prior(*args, seed=seed, **kwargs)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_posterior_state",
        capture_fitted_seed,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._prior_predictive_evidence_state",
        capture_prior_seed,
    )

    run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    expected = _spawn_child_seeds(base.inference.gblk_bayesian.seed, 2)
    assert captured == {"fitted": expected[0], "prior": expected[1]}


def test_blocked_stacking_requires_an_explicit_fold_support_rule(
    tmp_path: Path,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(base, labels=replace(base.labels, min_wells_for_fit=4))

    assert _stacking_training_minimum(cfg) == 4

    cfg = replace(
        cfg,
        inference=replace(
            cfg.inference,
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                minimum_training_wells=3,
            ),
        ),
    )
    assert _stacking_training_minimum(cfg) == 3


def test_grouped_spatial_folds_never_split_one_well_across_rows() -> None:
    well_ids = np.repeat(np.array(["a", "b", "c", "d"]), 2)
    coordinates = np.array(
        [
            [0.0, 0.0],
            [50.0, 0.0],
            [1_000.0, 0.0],
            [1_050.0, 0.0],
            [0.0, 1_000.0],
            [50.0, 1_000.0],
            [1_000.0, 1_000.0],
            [1_050.0, 1_000.0],
        ]
    )

    folds = list(
        _grouped_spatial_folds(
            coordinates,
            well_ids,
            n_folds=2,
            block_type="grid",
            grid_size=2,
            seed=7,
            block_size_km=None,
            buffer_distance=0.0,
        )
    )

    for train_mask, test_mask in folds:
        for well_id in np.unique(well_ids):
            rows = well_ids == well_id
            assert not (np.any(train_mask[rows]) and np.any(test_mask[rows]))
            assert np.all(test_mask[rows]) or not np.any(test_mask[rows])


def test_grouped_spatial_buffer_excludes_an_entire_nearby_deviated_well(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    well_ids = np.repeat(np.array(["a", "b", "c", "d"]), 2)
    coordinates = np.array(
        [
            [0.0, 0.0],
            [1_000.0, 0.0],
            [1_100.0, 0.0],
            [10_000.0, 0.0],
            [20_000.0, 0.0],
            [21_000.0, 0.0],
            [30_000.0, 0.0],
            [31_000.0, 0.0],
        ]
    )

    def fixed_group_folds(group_coords, **kwargs):
        assert len(group_coords) == 4
        assert kwargs["buffer_distance"] == 0.0
        yield (
            np.array([False, True, True, True]),
            np.array([True, False, False, False]),
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.spatial_block_cv",
        fixed_group_folds,
    )

    ((train_mask, test_mask),) = _grouped_spatial_folds(
        coordinates,
        well_ids,
        n_folds=2,
        block_type="grid",
        grid_size=2,
        seed=7,
        block_size_km=None,
        buffer_distance=500.0,
    )

    np.testing.assert_array_equal(test_mask, well_ids == "a")
    assert not np.any(train_mask[well_ids == "b"])
    assert np.all(train_mask[well_ids == "c"])
    assert np.all(train_mask[well_ids == "d"])


def test_joint_evidence_support_is_counted_by_unique_well(
    tmp_path: Path,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(base, labels=replace(base.labels, min_wells_for_fit=3))
    well_ids = np.repeat(np.array(["a", "b"]), 4)
    evidence = np.arange(8.0)[:, np.newaxis]

    with pytest.raises(GEOPFAValueError, match="fewer than 3 observed wells"):
        _prepare_joint_evidence_arrays(
            component_names=("component_a",),
            train_evidence={"component_a": evidence},
            prediction_evidence={"component_a": np.zeros((2, 1))},
            layer_names={"component_a": ["feature"]},
            y_train=np.arange(8.0)[:, np.newaxis],
            observed_train=np.ones((8, 1), dtype=bool),
            well_ids=well_ids,
            cfg=cfg,
        )


def test_gaussian_event_draws_remain_open_probabilities() -> None:
    fit = GBLKGaussianFitResult(
        component_names=("heat",),
        response_grid=np.zeros((2, 1)),
        response_interval=np.zeros((2, 2, 1)),
        response_draws=np.array([[[-1e6], [1e6]]]),
        likelihood_precision_draws=np.ones((1, 1)),
        fixed_coef_draws=None,
        fit=SimpleNamespace(inference="inla"),
        diagnostics={},
    )

    probability = _gaussian_predictive_exceedance_draws(
        fit,
        component_index=0,
        threshold_scaled=0.0,
    )

    assert np.all((probability > 0.0) & (probability < 1.0))


def test_gaussian_prior_event_probability_uses_continuous_distribution() -> (
    None
):
    probability = _gaussian_prior_exceedance_probability(
        mean=np.array([0.0, 10.0]),
        sd=np.array([1.0, 2.0]),
        threshold=5.0,
    )

    np.testing.assert_allclose(probability, ndtr(np.array([-5.0, 2.5])))
    assert probability[0] < 0.01
    assert probability[1] > 0.99


def test_gaussian_stacking_uses_configured_prior_probability_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    @dataclass(frozen=True)
    class FakeAssembled:
        component_names: tuple[str, ...]
        prior_response_mean_grid: np.ndarray
        prior_response_sd_grid: np.ndarray
        prior_response_mean_well: np.ndarray
        prior_response_sd_well: np.ndarray
        prior_probability_grid: np.ndarray | None
        prior_probability_well: np.ndarray | None
        y: np.ndarray

    base = _cfg_bayesian(tmp_path / "unused.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "temperature_c"},
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="thermal_prior",
                threshold=0.0,
                uncertainty_column="temperature_sd_c",
                p_min=0.2,
                p_max=0.8,
            )
        },
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(enabled=True),
        ),
    )
    fit_alphas = {
        "component_a": gblk_runner.AlphaCResult(
            grid_offset=np.zeros(2),
            scalar_fallback=0.0,
            provenance={"p_min": 0.2, "p_max": 0.8},
            latent_mean=np.array([-1_000.0, 1_000.0]),
            latent_sd=np.ones(2),
            event_threshold=0.0,
        )
    }
    assembled = FakeAssembled(
        component_names=("component_a",),
        prior_response_mean_grid=np.array([[-1_000.0], [1_000.0]]),
        prior_response_sd_grid=np.ones((2, 1)),
        prior_response_mean_well=np.array([[-1_000.0], [1_000.0]]),
        prior_response_sd_well=np.ones((2, 1)),
        prior_probability_grid=None,
        prior_probability_well=None,
        y=np.array([[0.0], [1.0]]),
    )
    monkeypatch.setattr(
        gblk_runner,
        "assemble_gblk_inputs",
        lambda *_args, **_kwargs: assembled,
    )
    labels = gblk_runner.LoadedLabels(
        gdf=gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32611"),
        config=cfg.labels,
    )

    result = _assemble_likelihood_groups(
        None,
        labels,
        fit_alphas,
        cfg,
        reference_grid=gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32611"),
    )["gaussian"]

    np.testing.assert_array_equal(
        result.prior_probability_grid[:, 0], [0.2, 0.8]
    )
    np.testing.assert_array_equal(
        result.prior_probability_well[:, 0], [0.2, 0.8]
    )


def test_gaussian_predictive_density_averages_paired_draws() -> None:
    fit = GBLKGaussianFitResult(
        component_names=("heat",),
        response_grid=np.zeros((1, 1)),
        response_interval=np.zeros((2, 1, 1)),
        response_draws=np.array([[[0.0]], [[2.0]]]),
        likelihood_precision_draws=np.ones((2, 1)),
        fixed_coef_draws=None,
        fit=SimpleNamespace(inference="inla"),
        diagnostics={},
    )

    log_density = _gaussian_predictive_log_density(
        fit,
        outcomes_scaled=np.array([1.0]),
        component_index=0,
        response_scale=1.0,
    )

    expected = -0.5 * (np.log(2.0 * np.pi) + 1.0)
    np.testing.assert_allclose(log_density, expected)


def test_gaussian_predictive_density_pairs_heterogeneous_precision_draws() -> (
    None
):
    fit = GBLKGaussianFitResult(
        component_names=("heat",),
        response_grid=np.zeros((1, 1)),
        response_interval=np.zeros((2, 1, 1)),
        response_draws=np.array([[[0.0]], [[3.0]]]),
        likelihood_precision_draws=np.array([[1.0], [4.0]]),
        fixed_coef_draws=None,
        fit=SimpleNamespace(inference="inla"),
        diagnostics={},
    )

    log_density = _gaussian_predictive_log_density(
        fit,
        outcomes_scaled=np.array([1.0]),
        component_index=0,
        response_scale=1.0,
    )

    paired_log_densities = np.array(
        [
            -0.5 * (np.log(2.0 * np.pi) + 1.0),
            0.5 * (np.log(4.0) - np.log(2.0 * np.pi)) - 8.0,
        ]
    )
    expected = np.log(np.exp(paired_log_densities).mean())
    np.testing.assert_allclose(log_density, expected)


def test_gaussian_predictive_density_is_reported_in_physical_units() -> None:
    physical_fit = GBLKGaussianFitResult(
        component_names=("heat",),
        response_grid=np.zeros((1, 1)),
        response_interval=np.zeros((2, 1, 1)),
        response_draws=np.array([[[0.0]], [[2.0]]]),
        likelihood_precision_draws=np.ones((2, 1)),
        fixed_coef_draws=None,
        fit=SimpleNamespace(inference="inla"),
        diagnostics={},
    )
    scale = 10.0
    scaled_fit = replace(
        physical_fit,
        response_draws=physical_fit.response_draws / scale,
        likelihood_precision_draws=(
            physical_fit.likelihood_precision_draws * scale**2
        ),
    )

    physical = _gaussian_predictive_log_density(
        physical_fit,
        outcomes_scaled=np.array([1.0]),
        component_index=0,
        response_scale=1.0,
    )
    rescaled = _gaussian_predictive_log_density(
        scaled_fit,
        outcomes_scaled=np.array([0.1]),
        component_index=0,
        response_scale=scale,
    )

    np.testing.assert_allclose(rescaled, physical)


def test_gaussian_predictive_response_summary_uses_one_coherent_mixture() -> (
    None
):
    n_draws = 4_000
    state = _GaussianPredictiveResponseState(
        latent_mean_draws=np.full((n_draws, 1), 100.0),
        likelihood_sd_draws=np.full(n_draws, 20.0),
        prior_mean=np.array([0.0]),
        prior_sd=np.array([5.0]),
        seed=91,
    )

    full_mean, full_interval = _gaussian_predictive_response_summary(
        state,
        weight=1.0,
        ci_level=0.90,
    )
    prior_mean, prior_interval = _gaussian_predictive_response_summary(
        state,
        weight=0.0,
        ci_level=0.90,
    )
    mixed_mean, mixed_interval = _gaussian_predictive_response_summary(
        state,
        weight=0.25,
        ci_level=0.90,
    )

    np.testing.assert_allclose(full_mean, [100.0])
    np.testing.assert_allclose(prior_mean, [0.0])
    np.testing.assert_allclose(mixed_mean, [25.0])
    assert full_interval[0, 0] < 80.0 < full_interval[1, 0]
    assert prior_interval[0, 0] < 0.0 < prior_interval[1, 0]
    assert mixed_interval[0, 0] < 0.0
    assert mixed_interval[1, 0] > 100.0


def test_gaussian_stacking_replaces_response_summary_with_selected_mixture() -> (
    None
):
    n_draws = 4_000
    geometry = gpd.points_from_xy([0.0], [0.0])
    probability = gpd.GeoDataFrame(
        {
            "probability": [0.8],
            "probability_lo": [0.7],
            "probability_hi": [0.9],
            "response_predictive_mean": [100.0],
            "response_predictive_lo": [60.0],
            "response_predictive_hi": [140.0],
        },
        geometry=geometry,
    )
    components = {
        "heat": ComponentProbability(
            probability=probability,
            model=None,
            feature_names=(),
            diagnostics={
                "response_summary_estimand": "posterior_predictive_response"
            },
        )
    }
    component_draws = {"heat": np.full((n_draws, 1), 0.8)}
    assembled = SimpleNamespace(
        component_names=("heat",),
        prior_probability_grid=np.array([[0.2]]),
    )
    stacking = {
        "heat": PredictiveStackingResult(
            weight=0.0,
            prior_log_score=1.0,
            full_log_score=2.0,
            selected_log_score=1.0,
            n_observations=20,
            n_wells=20,
            status="estimated",
        )
    }
    response_states = {
        "heat": _GaussianPredictiveResponseState(
            latent_mean_draws=np.full((n_draws, 1), 100.0),
            likelihood_sd_draws=np.full(n_draws, 20.0),
            prior_mean=np.array([0.0]),
            prior_sd=np.array([5.0]),
            seed=91,
        )
    }

    _apply_componentwise_stacking(
        components,
        component_draws,
        {"gaussian": assembled},
        stacking,
        response_states,
        ci_level=0.90,
    )

    response = components["heat"].probability
    np.testing.assert_allclose(response["probability"], 0.2)
    np.testing.assert_allclose(response["response_predictive_mean"], 0.0)
    assert response["response_predictive_lo"].iloc[0] < 0.0
    assert response["response_predictive_hi"].iloc[0] > 0.0
    assert (
        components["heat"].diagnostics["response_summary_estimand"]
        == "stacked_posterior_predictive_response"
    )
    assert components["heat"].diagnostics[
        "response_interval_includes_likelihood_variance"
    ]


def test_gaussian_stacking_uses_continuous_temperature_density(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_wells = 12
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "temperature_c"},
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
            )
        },
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(enabled=True),
        ),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        observed_mask=np.ones((n_wells, 1), dtype=bool),
        prior_probability_well=np.full((n_wells, 1), 0.01),
        prior_response_mean_well=np.zeros((n_wells, 1)),
        prior_response_sd_well=np.full((n_wells, 1), 50.0),
        well_coords=np.column_stack([np.arange(n_wells), np.zeros(n_wells)]),
        well_ids=np.array([f"well-{index}" for index in range(n_wells)]),
        well_depths_m=None,
        y=np.full((n_wells, 1), 2.0),
    )

    def fake_predictions(*_args, **_kwargs):
        return (
            np.full((n_wells, 1), 0.99),
            np.full((n_wells, 1), -10.0),
            np.zeros(n_wells, dtype=bool),
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._blocked_family_predictions",
        fake_predictions,
    )

    selection = _estimate_predictive_stacking(
        {"gaussian": assembled}, cfg, nc=3, a_wght=None
    )["component_a"]

    assert selection.weight == pytest.approx(0.0)
    assert selection.prior_log_score == pytest.approx(
        0.5 * np.log(2.0 * np.pi) + np.log(50.0) + 2.0
    )
    assert selection.full_log_score == pytest.approx(10.0)
    assert selection.n_wells == n_wells
    assert selection.evidence is not None
    np.testing.assert_array_equal(
        selection.evidence.well_ids,
        assembled.well_ids,
    )
    np.testing.assert_allclose(selection.evidence.outcomes, 100.0)
    np.testing.assert_allclose(
        selection.evidence.prior_log_density,
        -0.5 * np.log(2.0 * np.pi) - np.log(50.0) - 2.0,
    )
    np.testing.assert_allclose(selection.evidence.full_log_density, -10.0)


def test_target_depth_stacking_retains_prior_with_too_few_wells() -> None:
    selection = _select_component_stacking(
        outcomes=np.array([0.0, 1.0]),
        prior_probability=np.array([0.2, 0.8]),
        full_probability=np.array([0.8, 0.2]),
        validation_coordinates=np.array([[1.0, 2.0], [100.0, 200.0]]),
        validation_well_ids=np.array(["same", "same"]),
        minimum_wells=2,
    )

    assert selection.weight == 0.0
    assert selection.selected_log_score == selection.prior_log_score
    assert selection.n_observations == 2
    assert selection.n_wells == 1
    assert selection.status == "prior_retained_insufficient_validation_wells"


def test_target_depth_stacking_retains_prior_without_validation_rows() -> None:
    selection = _select_component_stacking(
        outcomes=np.array([]),
        prior_probability=np.array([]),
        full_probability=np.array([]),
        validation_coordinates=np.empty((0, 2)),
        validation_well_ids=np.array([], dtype=str),
        minimum_wells=2,
    )

    assert selection.weight == 0.0
    assert selection.prior_log_score is None
    assert selection.full_log_score is None
    assert selection.selected_log_score is None
    assert selection.n_observations == 0
    assert selection.n_wells == 0
    assert selection.status == "prior_retained_no_validation_wells"


def test_stacking_validation_depth_is_component_specific() -> None:
    observed = np.array([True, True, True, False])
    depths = np.array([3_000.0, 4_000.0, 4_000.0, 4_000.0])

    heat = _stacking_validation_mask(
        observed_mask=observed,
        well_depths_m=depths,
        component_name="heat",
        validation_depths_m={"heat": 4_000.0},
    )
    hydraulic = _stacking_validation_mask(
        observed_mask=observed,
        well_depths_m=depths,
        component_name="hydraulic",
        validation_depths_m={"heat": 4_000.0},
    )

    np.testing.assert_array_equal(heat, [False, True, True, False])
    np.testing.assert_array_equal(hydraulic, observed)


@pytest.mark.parametrize("family", ["bernoulli", "gaussian"])
def test_target_depth_stacking_retains_prior_when_no_rows_match(
    family: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_wells = 4
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    labels = base.labels
    if family == "gaussian":
        labels = replace(
            labels,
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        )
    labels = replace(labels, depth_col="depth_m")
    cfg = replace(
        base,
        dimensions="3d",
        labels=labels,
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                validation_depths_m={"component_a": 4_000.0},
            ),
        ),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        observed_mask=np.ones((n_wells, 1), dtype=bool),
        prior_probability_well=np.full((n_wells, 1), 0.25),
        prior_response_mean_well=np.full((n_wells, 1), 150.0),
        prior_response_sd_well=np.full((n_wells, 1), 25.0),
        well_coords=np.column_stack([np.arange(n_wells), np.zeros(n_wells)]),
        well_ids=np.array([f"well-{index}" for index in range(n_wells)]),
        well_depths_m=np.array([2_000.0, 3_000.0, 2_000.0, 3_000.0]),
        y=np.full((n_wells, 1), 3.0),
    )

    def fake_predictions(*_args, **_kwargs):
        probability = np.full((n_wells, 1), 0.75)
        log_density = (
            np.full((n_wells, 1), -2.0) if family == "gaussian" else None
        )
        return probability, log_density, np.zeros(n_wells, dtype=bool)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._blocked_family_predictions",
        fake_predictions,
    )

    selection = _estimate_predictive_stacking(
        {family: assembled}, cfg, nc=3, a_wght=None
    )["component_a"]

    assert selection.weight == 0.0
    assert selection.prior_log_score is None
    assert selection.full_log_score is None
    assert selection.selected_log_score is None
    assert selection.n_observations == 0
    assert selection.n_wells == 0
    assert selection.status == "prior_retained_no_validation_wells"
    assert selection.validation_depth_m == 4_000.0
    assert selection.evidence is not None
    assert selection.evidence.family == family
    assert selection.evidence.outcomes.size == 0


def test_blocked_predictions_reuse_the_full_model_spatial_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(base.labels, min_wells_for_fit=1),
        cross_validation=replace(
            base.cross_validation, n_folds=2, grid_size=2
        ),
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                minimum_training_wells=1,
            ),
        ),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((4, 1), dtype=bool),
        well_offsets=np.zeros((4, 1)),
        well_coords=np.array(
            [[0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0], [1_000.0, 1_000.0]]
        ),
        well_ids=np.array(["a", "b", "c", "d"]),
        grid_coords=np.array([[-100.0, -200.0], [2_000.0, 2_500.0]]),
        evidence={"component_a": np.zeros((4, 0))},
        layer_names={"component_a": []},
    )
    captured_domains: list[np.ndarray] = []

    def capture_backend(
        _coords,
        _outcomes,
        grid,
        *,
        component_names,
        bayes_config,
        spatial_domain,
        **_kwargs,
    ):
        captured_domains.append(np.asarray(spatial_domain, dtype=float))
        draws = np.full((bayes_config.n_draws, len(grid), 1), 0.5)
        return GBLKBayesianFitResult(
            component_names=tuple(component_names),
            p_q_grid=draws.mean(axis=0),
            p_q_interval=np.quantile(draws, [0.05, 0.95], axis=0),
            p_q_draws=draws,
            p_joint_grid=draws[:, :, 0].mean(axis=0),
            p_joint_interval=np.quantile(draws[:, :, 0], [0.05, 0.95], axis=0),
            p_joint_draws=draws[:, :, 0],
            fixed_coef_draws=None,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={},
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint",
        capture_backend,
    )

    _blocked_family_predictions(
        assembled,
        "bernoulli",
        cfg,
        nc=3,
        a_wght=None,
    )

    assert len(captured_domains) == 2
    for domain in captured_domains:
        np.testing.assert_allclose(
            domain,
            np.array([[-100.0, 2_000.0], [-200.0, 2_500.0]]),
        )


def test_blocked_predictions_standardize_against_canonical_grid_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(base.labels, min_wells_for_fit=1),
        evidence=replace(
            base.evidence,
            standardization="prediction_support",
        ),
        cross_validation=replace(
            base.cross_validation, n_folds=2, grid_size=2
        ),
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                minimum_training_wells=1,
            ),
        ),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((4, 1), dtype=bool),
        well_offsets=np.zeros((4, 1)),
        well_coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        well_ids=np.array(["a", "b", "c", "d"]),
        grid_coords=np.array([[0.0, 0.0], [3.0, 0.0]]),
        evidence={"component_a": np.array([[1.0], [3.0], [7.0], [9.0]])},
        grid_evidence={"component_a": np.array([[0.0], [10.0]])},
        layer_names={"component_a": ["feature"]},
    )
    folds = (
        (
            np.array([False, False, True, True]),
            np.array([True, True, False, False]),
        ),
        (
            np.array([True, True, False, False]),
            np.array([False, False, True, True]),
        ),
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._grouped_spatial_folds",
        lambda *_args, **_kwargs: iter(folds),
    )
    captured_prediction_designs: list[np.ndarray] = []

    def capture_backend(
        _coords,
        _outcomes,
        grid,
        *,
        component_names,
        bayes_config,
        fixed_effects_grid,
        **_kwargs,
    ):
        captured_prediction_designs.append(fixed_effects_grid.copy())
        draws = np.full((bayes_config.n_draws, len(grid), 1), 0.5)
        return GBLKBayesianFitResult(
            component_names=tuple(component_names),
            p_q_grid=draws.mean(axis=0),
            p_q_interval=np.quantile(draws, [0.05, 0.95], axis=0),
            p_q_draws=draws,
            p_joint_grid=draws[:, :, 0].mean(axis=0),
            p_joint_interval=np.quantile(draws[:, :, 0], [0.05, 0.95], axis=0),
            p_joint_draws=draws[:, :, 0],
            fixed_coef_draws=None,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={},
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint",
        capture_backend,
    )

    _blocked_family_predictions(
        assembled,
        "bernoulli",
        cfg,
        nc=3,
        a_wght=None,
    )

    np.testing.assert_allclose(
        captured_prediction_designs[0][:, 0, 0],
        np.array([-0.8, -0.4]),
    )
    np.testing.assert_allclose(
        captured_prediction_designs[1][:, 0, 0],
        np.array([0.4, 0.8]),
    )


def test_incomplete_buffered_cv_retains_prior_without_fitting_unsupported_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "heat_label"},
            min_wells_for_fit=4,
        ),
        alpha={"component_a": base.alpha["component_a"]},
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                minimum_training_wells=3,
            ),
        ),
        cross_validation=replace(base.cross_validation, n_folds=2),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((4, 1), dtype=bool),
        well_offsets=np.zeros((4, 1)),
        well_coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        well_ids=np.array(["a", "b", "c", "d"]),
        well_depths_m=None,
        grid_coords=np.array([[0.0, 0.0], [3.0, 0.0]]),
        evidence={"component_a": np.array([[1.0], [2.0], [3.0], [4.0]])},
        layer_names={"component_a": ["feature"]},
        prior_probability_well=np.full((4, 1), 0.4),
    )
    folds = (
        (
            np.array([False, False, True, True]),
            np.array([True, True, False, False]),
        ),
        (
            np.array([True, True, False, False]),
            np.array([False, False, True, True]),
        ),
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._grouped_spatial_folds",
        lambda *_args, **_kwargs: iter(folds),
    )

    def fail_if_fit(*_args, **_kwargs):
        raise AssertionError("unsupported buffered fold must not be fitted")

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint", fail_if_fit
    )

    selection = _estimate_predictive_stacking(
        {"bernoulli": assembled}, cfg, nc=3, a_wght=None
    )["component_a"]

    assert selection.weight == 0.0
    assert selection.prior_log_score is None
    assert selection.full_log_score is None
    assert selection.selected_log_score is None
    assert selection.n_observations == 4
    assert selection.n_wells == 4
    assert selection.status == "prior_retained_incomplete_spatial_cv"
    assert selection.evidence is not None
    np.testing.assert_allclose(selection.evidence.prior_probability, 0.4)
    assert np.all(np.isnan(selection.evidence.full_probability))


def test_incomplete_gaussian_cv_preserves_physical_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "temperature_c"},
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
            min_wells_for_fit=4,
        ),
        alpha={"component_a": base.alpha["component_a"]},
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
            predictive_stacking=PredictiveStackingConfig(
                enabled=True,
                minimum_training_wells=3,
            ),
        ),
        cross_validation=replace(base.cross_validation, n_folds=2),
    )
    scaled_temperature = np.array([2.8, 3.2, 3.6, 4.0])
    assembled = SimpleNamespace(
        component_names=("component_a",),
        y=scaled_temperature[:, None],
        observed_mask=np.ones((4, 1), dtype=bool),
        well_offsets=np.zeros((4, 1)),
        well_coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        well_ids=np.array(["a", "b", "c", "d"]),
        well_depths_m=None,
        grid_coords=np.array([[0.0, 0.0], [3.0, 0.0]]),
        evidence={"component_a": np.array([[1.0], [2.0], [3.0], [4.0]])},
        layer_names={"component_a": ["feature"]},
        prior_probability_well=np.full((4, 1), 0.4),
        prior_response_mean_well=np.full((4, 1), 170.0),
        prior_response_sd_well=np.full((4, 1), 20.0),
    )
    folds = (
        (
            np.array([False, False, True, True]),
            np.array([True, True, False, False]),
        ),
        (
            np.array([True, True, False, False]),
            np.array([False, False, True, True]),
        ),
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._grouped_spatial_folds",
        lambda *_args, **_kwargs: iter(folds),
    )

    def fail_if_fit(*_args, **_kwargs):
        raise AssertionError("unsupported buffered fold must not be fitted")

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_gaussian_bayesian_joint",
        fail_if_fit,
    )

    selection = _estimate_predictive_stacking(
        {"gaussian": assembled}, cfg, nc=3, a_wght=None
    )["component_a"]

    assert selection.status == "prior_retained_incomplete_spatial_cv"
    assert selection.evidence is not None
    np.testing.assert_allclose(
        selection.evidence.outcomes,
        scaled_temperature * 50.0,
    )
    assert np.all(np.isfinite(selection.evidence.prior_log_density))
    assert np.all(np.isnan(selection.evidence.full_log_density))


def test_nonfinite_backend_predictions_are_not_mislabeled_as_incomplete_cv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _cfg_bayesian(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "heat_label"},
        ),
        alpha={"component_a": base.alpha["component_a"]},
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
        ),
    )
    assembled = SimpleNamespace(
        component_names=("component_a",),
        y=np.array([[0.0], [1.0], [0.0], [1.0]]),
        observed_mask=np.ones((4, 1), dtype=bool),
        prior_probability_well=np.full((4, 1), 0.4),
        well_coords=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        well_ids=np.array(["a", "b", "c", "d"]),
        well_depths_m=None,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._blocked_family_predictions",
        lambda *_args, **_kwargs: (
            np.array([[0.5], [np.nan], [0.5], [0.5]]),
            None,
            np.zeros(4, dtype=bool),
        ),
    )

    with pytest.raises(
        RuntimeError, match="nonfinite out-of-fold predictions"
    ):
        _estimate_predictive_stacking(
            {"bernoulli": assembled}, cfg, nc=3, a_wght=None
        )


def test_bayesian_projection_blocks_equal_one_shot_projection() -> None:
    from latticekrigx.basis.assembly import compute_basis

    grid = np.array(
        [[0.0, 0.0], [0.25, 0.75], [0.75, 0.25], [1.0, 1.0]], dtype=float
    )
    lkinfo = build_lkinfo_2d(grid, nc=3, nlevel=1)
    basis = compute_basis(grid, lkinfo, normalize=False)
    rng = np.random.default_rng(108)
    coefficient_draws = rng.normal(size=(5, basis.shape[1], 2))
    fixed_draws = rng.normal(size=(5, 1, 2))
    fixed_design = rng.normal(size=(len(grid), 2, 1))
    state = GBLKBayesianPosteriorState(
        component_names=("heat", "reservoir"),
        coefficient_draws=coefficient_draws,
        fixed_coef_draws=fixed_draws,
        grid_model=grid,
        grid_offsets=np.full((len(grid), 2), -0.25),
        fixed_design_grid=fixed_design,
        lkinfo=lkinfo,
        fit=SimpleNamespace(inference="inla"),
        diagnostics={},
    )

    one_shot = project_gblk_bayesian_draw_block(state, 0, 5)
    first = project_gblk_bayesian_draw_block(state, 0, 2)
    second = project_gblk_bayesian_draw_block(state, 2, 5)

    for field in ("component_probability", "evidence_logit", "spatial_logit"):
        np.testing.assert_allclose(
            np.concatenate([getattr(first, field), getattr(second, field)]),
            getattr(one_shot, field),
        )
    reconstructed = expit(
        one_shot.prior_logit[np.newaxis, :, :]
        + one_shot.evidence_logit
        + one_shot.spatial_logit
    )
    np.testing.assert_allclose(one_shot.component_probability, reconstructed)


@pytest.fixture(autouse=True)
def _isolate_runner_from_external_inla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep runner packaging tests independent of the external INLA runtime."""

    def fake_backend(
        coords,
        _labels,
        grid,
        *,
        component_names,
        bayes_config,
        fixed_effects=None,
        **_kwargs,
    ):
        q = len(component_names)
        draw_axis = np.linspace(0.25, 0.75, bayes_config.n_draws)
        p_q_draws = np.broadcast_to(
            draw_axis[:, None, None],
            (bayes_config.n_draws, len(grid), q),
        ).copy()
        p_joint_draws = np.prod(p_q_draws, axis=-1)
        tail = (1.0 - bayes_config.ci_level) / 2.0
        fixed_draws = (
            None
            if fixed_effects is None
            else np.zeros(
                (bayes_config.n_draws, fixed_effects.shape[1], q),
                dtype=float,
            )
        )
        return GBLKBayesianFitResult(
            component_names=tuple(component_names),
            p_q_grid=p_q_draws.mean(axis=0),
            p_q_interval=np.quantile(p_q_draws, [tail, 1.0 - tail], axis=0),
            p_q_draws=p_q_draws,
            p_joint_grid=p_joint_draws.mean(axis=0),
            p_joint_interval=np.quantile(
                p_joint_draws, [tail, 1.0 - tail], axis=0
            ),
            p_joint_draws=p_joint_draws,
            fixed_coef_draws=fixed_draws,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={
                "backend": "gblk_bayesian",
                "estimator": "paige_inla",
                "n": len(coords),
                "n_components": q,
            },
        )

    def fake_state_backend(
        coords,
        _labels,
        grid,
        *,
        component_names,
        bayes_config,
        grid_offsets=None,
        fixed_effects_grid=None,
        **_kwargs,
    ):
        q = len(component_names)
        lkinfo = build_lkinfo_2d(np.asarray(grid, dtype=float), nc=3, nlevel=1)
        from latticekrigx.basis.assembly import compute_basis

        n_basis = compute_basis(grid, lkinfo, normalize=False).shape[1]
        fixed_draws = (
            None
            if fixed_effects_grid is None
            else np.zeros(
                (bayes_config.n_draws, fixed_effects_grid.shape[1], q),
                dtype=float,
            )
        )
        return GBLKBayesianPosteriorState(
            component_names=tuple(component_names),
            coefficient_draws=np.zeros((bayes_config.n_draws, n_basis, q)),
            fixed_coef_draws=fixed_draws,
            grid_model=np.asarray(grid, dtype=float),
            grid_offsets=np.asarray(grid_offsets, dtype=float),
            fixed_design_grid=fixed_effects_grid,
            lkinfo=lkinfo,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={
                "backend": "gblk_bayesian",
                "estimator": "paige_inla",
                "n": len(coords),
                "n_components": q,
            },
            projected_mean_draws=None,
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint", fake_backend
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_posterior_state",
        fake_state_backend,
    )


def _cfg_bayesian(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
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
        spatial_field=SpatialFieldConfig(
            enabled=True,
            n_levels=1,
            lattice_centers_per_dimension=_NC_SMALL,
        ),
        inference=InferenceConfig(
            backend="gblk",
            gblk_bayesian=GBLKBayesianConfig(
                enabled=True,
                n_draws=_N_DRAWS,
                seed=42,
                ci_level=0.9,
                kleiber_profiles={
                    "bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)
                },
            ),
        ),
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


def _mismatched_gaussian_prior_case(
    tmp_path: Path,
    *,
    streamed: bool,
) -> tuple[dict, ProbabilisticConfig]:
    """Return a coarse fixed Gaussian prior on a finer canonical grid."""
    fixture = make_synthetic_pfa(grid_n=3, n_wells=8, seed=913)
    components = fixture.pfa["criteria"]["geologic"]["components"]
    coarse_component = components["component_b"]
    coarse_grid = coarse_component["pr_norm"]
    x = coarse_grid.geometry.x.to_numpy()
    y = coarse_grid.geometry.y.to_numpy()
    corners = np.isin(x, [x.min(), x.max()]) & np.isin(y, [y.min(), y.max()])
    coarse_grid = coarse_grid.loc[corners].reset_index(drop=True).copy()
    coarse_component["pr_norm"] = coarse_grid.copy()
    thermal_model = coarse_grid[["geometry"]].copy()
    thermal_model["value_interpolated"] = [0.0, 10.0, 0.0, 10.0]
    thermal_model["temperature_sd_c"] = [1.0, 10.0, 1.0, 10.0]
    coarse_component["layers"]["prior_layer_b"]["model"] = thermal_model

    suffix = "streamed" if streamed else "materialized"
    wells_path = tmp_path / f"wells_{suffix}.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / f"gaussian_{suffix}")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            observation_models={
                "component_b": ObservationModelConfig(family="gaussian")
            },
        ),
        alpha={
            "component_a": base.alpha["component_a"],
            "component_b": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_b",
                threshold=5.0,
                uncertainty_column="temperature_sd_c",
                p_min=0.05,
                p_max=0.95,
                force_prior_predictive=True,
            ),
        },
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                cluster_effect=not streamed,
                kleiber_profiles={},
            ),
        ),
        outputs=replace(
            base.outputs,
            posterior_draw_blocks=streamed,
            posterior_draw_block_size=3,
        ),
    )
    return fixture.pfa, cfg


def _cfg_standard(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
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
        spatial_field=SpatialFieldConfig(
            enabled=False,
            lattice_centers_per_dimension=_NC_SMALL,
        ),
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


def _run_bayesian(tmp_path: Path) -> ProbabilisticResult:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=40, seed=11)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg_bayesian(wells_path, tmp_path / "out")
    return run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )


def _run_standard(tmp_path: Path) -> ProbabilisticResult:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=40, seed=11)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg_standard(wells_path, tmp_path / "out")
    return run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_p7_bayesian_gblk_returns_result(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    assert isinstance(result, ProbabilisticResult)
    assert not result.skipped
    assert result.config is not None


def test_p7_bayesian_gblk_has_q_components(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    assert set(result.components.keys()) == set(_COMPONENTS)
    for name in _COMPONENTS:
        assert isinstance(result.components[name], ComponentProbability)


def test_p7_bayesian_gblk_has_combined_surface(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert "probability" in result.combined.columns


def test_gaussian_heat_stage_feeds_existing_component_combination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=31)
    fixture.wells["temperature_c"] = np.linspace(140.0, 280.0, 30)
    heat = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = heat["layers"]["prior_layer_a"]["model"]
    thermal["value_interpolated"] = np.linspace(150.0, 250.0, len(thermal))
    thermal["temperature_sd_c"] = 25.0
    reservoir = fixture.pfa["criteria"]["geologic"]["components"][
        "component_b"
    ]
    reservoir["pr_norm"] = (
        reservoir["pr_norm"].iloc[::-1].reset_index(drop=True)
    )
    for layer in reservoir["layers"].values():
        layer["model"] = layer["model"].iloc[::-1].reset_index(drop=True)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={
                "component_a": "temperature_c",
                "component_b": "reservoir_label",
            },
            observation_models={
                "component_a": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
                p_min=0.001,
                p_max=0.999,
            ),
            "component_b": base.alpha["component_b"],
        },
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
        ),
    )
    captured: dict[str, np.ndarray] = {}
    assembled_grid_coordinates: list[np.ndarray] = []
    original_assemble = gblk_runner.assemble_gblk_inputs

    def capture_assembly(*args, **kwargs):
        assembled = original_assemble(*args, **kwargs)
        assembled_grid_coordinates.append(assembled.grid_coords.copy())
        return assembled

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.assemble_gblk_inputs",
        capture_assembly,
    )

    def fake_gaussian(
        _coords,
        responses,
        grid,
        *,
        component_names,
        bayes_config,
        grid_offsets,
        **_kwargs,
    ):
        captured["responses"] = np.asarray(responses)
        captured["grid_offsets"] = np.asarray(grid_offsets)
        draws = np.empty((bayes_config.n_draws, len(grid), 1), dtype=float)
        draws[::2, :, 0] = 3.0
        draws[1::2, :, 0] = 5.0
        interval = np.quantile(draws, [0.05, 0.95], axis=0)
        return GBLKGaussianFitResult(
            component_names=tuple(component_names),
            response_grid=draws.mean(axis=0),
            response_interval=interval,
            response_draws=draws,
            likelihood_precision_draws=np.ones(
                (bayes_config.n_draws, 1), dtype=float
            ),
            fixed_coef_draws=None,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={"backend": "gblk_bayesian"},
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_gaussian_bayesian_joint",
        fake_gaussian,
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    expected_grid = np.column_stack(
        [heat["pr_norm"].geometry.x, heat["pr_norm"].geometry.y]
    )
    assert len(assembled_grid_coordinates) == 2
    for coordinates in assembled_grid_coordinates:
        np.testing.assert_allclose(coordinates, expected_grid)
    assert result.combined.geometry.equals(heat["pr_norm"].geometry)
    for component in result.components.values():
        assert component.probability.geometry.equals(heat["pr_norm"].geometry)
    np.testing.assert_allclose(
        captured["responses"][:, 0], fixture.wells["temperature_c"] / 50.0
    )
    np.testing.assert_allclose(
        captured["grid_offsets"][:, 0],
        thermal["value_interpolated"].to_numpy(dtype=float) / 50.0,
    )
    heat_draws = result.component_probability_draws["component_a"]
    expected_heat_draws = ndtr(
        np.resize(np.array([3.0, 5.0]), heat_draws.shape[0]) - 4.0
    )
    np.testing.assert_allclose(heat_draws[:, 0], expected_heat_draws)
    assert np.all((heat_draws > 0.0) & (heat_draws < 1.0))
    np.testing.assert_allclose(
        result.components["component_a"].probability["probability"], 0.5
    )
    response = result.components["component_a"].probability
    assert "response_predictive_mean" in response
    assert "response_mean" not in response
    assert np.all(
        response["response_predictive_lo"] < response["response_predictive_hi"]
    )
    assert (
        result.components["component_a"].diagnostics[
            "response_summary_estimand"
        ]
        == "posterior_predictive_response"
    )
    expected_joint = (
        heat_draws * result.component_probability_draws["component_b"]
    ).mean(axis=0)
    np.testing.assert_allclose(
        result.combined["probability"].to_numpy(), expected_joint
    )


def test_mixed_family_runner_fits_two_bernoulli_and_one_gaussian_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=317)
    components = fixture.pfa["criteria"]["geologic"]["components"]
    components["temperature"] = deepcopy(components["component_a"])
    thermal = components["temperature"]["layers"]["prior_layer_a"]["model"]
    thermal["value_interpolated"] = np.linspace(150.0, 250.0, len(thermal))
    thermal["temperature_sd_c"] = 25.0
    fixture.wells["temperature_c"] = np.linspace(140.0, 280.0, 30)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={
                **base.labels.label_columns,
                "temperature": "temperature_c",
            },
            observation_models={
                "temperature": ObservationModelConfig(
                    family="gaussian", response_scale=50.0
                )
            },
        ),
        alpha={
            **base.alpha,
            "temperature": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
            ),
        },
    )
    captured: dict[str, tuple[str, ...]] = {}
    delegated_bernoulli = gblk_runner.fit_gblk_bayesian_joint

    def capture_bernoulli(*args, component_names, bayes_config, **kwargs):
        captured["bernoulli"] = tuple(component_names)
        assert bayes_config.kleiber_profiles == {
            "bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)
        }
        return delegated_bernoulli(
            *args,
            component_names=component_names,
            bayes_config=bayes_config,
            **kwargs,
        )

    def fake_gaussian(
        _coords,
        _responses,
        grid,
        *,
        component_names,
        bayes_config,
        **_kwargs,
    ):
        captured["gaussian"] = tuple(component_names)
        assert "gaussian" not in bayes_config.kleiber_profiles
        draws = np.full((bayes_config.n_draws, len(grid), 1), 4.0)
        return GBLKGaussianFitResult(
            component_names=tuple(component_names),
            response_grid=draws.mean(axis=0),
            response_interval=np.quantile(draws, [0.05, 0.95], axis=0),
            response_draws=draws,
            likelihood_precision_draws=np.ones((bayes_config.n_draws, 1)),
            fixed_coef_draws=None,
            fit=SimpleNamespace(inference="inla"),
            diagnostics={"backend": "gblk_bayesian"},
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_joint",
        capture_bernoulli,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_gaussian_bayesian_joint",
        fake_gaussian,
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    assert captured == {
        "bernoulli": ("component_a", "component_b"),
        "gaussian": ("temperature",),
    }
    assert set(result.components) == {
        "component_a",
        "component_b",
        "temperature",
    }


def test_bayesian_fixed_gaussian_prior_reports_response_distribution(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=131)
    heat = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = heat["layers"]["prior_layer_a"]["model"]
    thermal_mean = np.linspace(150.0, 250.0, len(thermal))
    thermal_sd = np.linspace(15.0, 30.0, len(thermal))
    thermal["value_interpolated"] = thermal_mean
    thermal["temperature_sd_c"] = thermal_sd
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            observation_models={
                "component_a": ObservationModelConfig(family="gaussian")
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
                p_min=1e-12,
                p_max=1.0 - 1e-12,
                force_prior_predictive=True,
            ),
            "component_b": base.alpha["component_b"],
        },
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                kleiber_profiles={},
            ),
        ),
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    component = result.components["component_a"]
    response = component.probability
    np.testing.assert_allclose(
        response["response_predictive_mean"], thermal_mean
    )
    expected_z = 1.6448536269514722
    np.testing.assert_allclose(
        response["response_predictive_lo"],
        thermal_mean - expected_z * thermal_sd,
    )
    np.testing.assert_allclose(
        response["response_predictive_hi"],
        thermal_mean + expected_z * thermal_sd,
    )
    expected_probability = ndtr((thermal_mean - 200.0) / thermal_sd)
    np.testing.assert_allclose(response["probability"], expected_probability)
    np.testing.assert_allclose(
        result.component_probability_draws["component_a"],
        np.broadcast_to(
            expected_probability,
            (_N_DRAWS, expected_probability.size),
        ),
    )
    assert (
        component.diagnostics["event_probability_estimand"]
        == "clipped_configured_prior_predictive_response_exceedance"
    )
    assert (
        component.diagnostics["response_summary_estimand"]
        == "prior_predictive_response"
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_bayesian_fixed_bernoulli_prior_interval_is_exactly_degenerate(
    tmp_path: Path,
    streamed: bool,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=132)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / f"out_{streamed}")
    cfg = replace(
        base,
        alpha={
            "component_a": base.alpha["component_a"],
            "component_b": replace(
                base.alpha["component_b"], force_prior_predictive=True
            ),
        },
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                n_draws=7,
                cluster_effect=not streamed,
                kleiber_profiles={},
            ),
        ),
        outputs=replace(
            base.outputs,
            posterior_draw_blocks=streamed,
            posterior_draw_block_size=3,
        ),
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    component = result.components["component_b"]
    probability = component.probability
    assert component.diagnostics["inference_role"] == "fixed_prior_predictive"
    np.testing.assert_array_equal(
        probability["probability_lo"], probability["probability"]
    )
    np.testing.assert_array_equal(
        probability["probability_hi"], probability["probability"]
    )


@pytest.mark.parametrize("streamed", [False, True])
def test_bayesian_fixed_gaussian_prior_resampling_preserves_one_estimand(
    tmp_path: Path,
    streamed: bool,
) -> None:
    pfa, cfg = _mismatched_gaussian_prior_case(tmp_path, streamed=streamed)

    result = run_gblk_probabilistic(pfa, cfg)

    response = result.components["component_b"].probability
    mean = response["response_predictive_mean"].to_numpy()
    sd = (
        response["response_predictive_hi"].to_numpy()
        - response["response_predictive_lo"].to_numpy()
    ) / (2.0 * 1.6448536269514722)
    expected = np.clip(ndtr((mean - 5.0) / sd), 0.05, 0.95)
    np.testing.assert_allclose(response["probability"], expected)
    assert mean[4] == pytest.approx(5.0)
    assert sd[4] == pytest.approx(5.5)
    assert response["probability"].iloc[4] == pytest.approx(0.5)
    component_a = (
        result.components["component_a"].probability["probability"].to_numpy()
    )
    np.testing.assert_allclose(
        result.combined["probability"], component_a * expected
    )
    if not streamed:
        np.testing.assert_allclose(
            result.component_probability_draws["component_b"],
            np.broadcast_to(expected, (_N_DRAWS, expected.size)),
        )


def test_componentwise_stacking_shrinks_only_the_harmful_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=41)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(enabled=True),
        ),
    )

    stacking_inputs: dict[str, object] = {}

    def fake_stacking(groups, *_args, **_kwargs):
        stacking_inputs["groups"] = groups
        return {
            "component_a": PredictiveStackingResult(
                0.0, 0.2, 0.8, 0.2, 20, 20, "estimated"
            ),
            "component_b": PredictiveStackingResult(
                1.0, 0.8, 0.2, 0.2, 20, 20, "estimated"
            ),
        }

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner._estimate_predictive_stacking",
        fake_stacking,
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    heat_prior = (
        result.components["component_a"].probability["probability"].to_numpy()
    )
    groups = stacking_inputs["groups"]
    expected_prior = groups["bernoulli"].prior_probability_grid[:, 0]
    np.testing.assert_allclose(heat_prior, expected_prior)
    assert result.components["component_a"].diagnostics[
        "predictive_stacking_weight"
    ] == pytest.approx(0.0)
    assert result.components["component_b"].diagnostics[
        "predictive_stacking_weight"
    ] == pytest.approx(1.0)
    assert set(result.predictive_stacking) == {"component_a", "component_b"}


def test_componentwise_stacking_uses_blocked_out_of_fold_predictions(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=49)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        inference=replace(
            base.inference,
            predictive_stacking=PredictiveStackingConfig(enabled=True),
        ),
        cross_validation=replace(
            base.cross_validation, n_folds=3, grid_size=2
        ),
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    for component in result.components.values():
        diagnostics = component.diagnostics
        assert diagnostics["predictive_stacking_validation"] == (
            "blocked_out_of_fold"
        )
        assert diagnostics["predictive_stacking_n"] == 30
        assert (
            diagnostics["predictive_stacking_score"] == "bernoulli_log_score"
        )
        assert 0.0 <= diagnostics["predictive_stacking_weight"] <= 1.0


def test_bayesian_gblk_combines_paired_draws_by_product(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=23)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg_bayesian(wells_path, tmp_path / "out")

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    ordered = np.stack(
        [
            result.component_probability_draws[name]
            for name in sorted(result.component_probability_draws)
        ],
        axis=2,
    )
    expected = np.prod(ordered, axis=2)
    np.testing.assert_allclose(result.combined_probability_draws, expected)
    np.testing.assert_allclose(
        result.combined["probability"].to_numpy(), expected.mean(axis=0)
    )


def test_top_level_bayesian_runner_persists_audited_draw_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=19)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    output_dir = tmp_path / "out"
    base_cfg = _cfg_bayesian(wells_path, output_dir)
    cfg = replace(
        base_cfg,
        inference=replace(
            base_cfg.inference,
            gblk_bayesian=replace(
                base_cfg.inference.gblk_bayesian,
                cluster_effect=False,
            ),
        ),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
            format=(),
        ),
    )

    result = run_probabilistic(fixture.pfa, cfg)

    index_path = output_dir / "posterior_draws" / "index.json"
    manifest_path = output_dir / "manifest.json"
    assert index_path.is_file()
    assert manifest_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert len(index["blocks"]) == 3
    assert index["n_draws"] == _N_DRAWS
    assert index["seed"] == 42
    persisted = []
    for block in index["blocks"]:
        with np.load(
            index_path.parent / block["path"], allow_pickle=False
        ) as payload:
            persisted.append(payload["combined_probability"])
    assert result.combined_probability_draws is None
    assert result.component_probability_draws == {}
    assert result.posterior_draw_index == index_path
    np.testing.assert_allclose(
        np.concatenate(persisted).mean(axis=0),
        result.combined["probability"].to_numpy(),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    assert "posterior_draws/index.json" in manifest_paths
    assert {
        f"posterior_draws/{block['path']}" for block in index["blocks"]
    }.issubset(manifest_paths)

    def fail_if_refit(*_args, **_kwargs):
        raise AssertionError(
            "a completed persisted posterior must not be refitted"
        )

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_posterior_state",
        fail_if_refit,
    )
    with pytest.raises(GEOPFAValueError, match="completed manifested run"):
        run_probabilistic(fixture.pfa, cfg)

    assert index_path.is_file()
    assert manifest_path.is_file()


def test_prior_predictive_streaming_run_is_not_labeled_as_posterior(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=41)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "prior")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            label_columns={"component_a": "heat_label"},
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar",
                force_prior_predictive=True,
                use_evidence_prior=True,
            )
        },
        evidence=EvidenceConfig(
            include_layers=("gradient",),
            regularization=RegularizationConfig(
                prior_means={"component_a:gradient": 0.5},
                prior_precisions={"component_a:gradient": 4.0},
            ),
        ),
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                cluster_effect=False,
                kleiber_profiles={},
            ),
        ),
        outputs=replace(
            base.outputs,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
        ),
    )

    result = run_probabilistic(fixture.pfa, cfg)
    index = json.loads(result.posterior_draw_index.read_text())

    assert index["uncertainty_semantics"] == (
        "paired_prior_predictive_probability_draws"
    )
    assert index["state"]["metadata"]["model"] == (
        "geopfa_probabilistic_prior_predictive"
    )


def test_streamed_fixed_gaussian_prior_reports_response_distribution(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=141)
    heat = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    thermal = heat["layers"]["prior_layer_a"]["model"]
    thermal_mean = np.linspace(150.0, 250.0, len(thermal))
    thermal_sd = np.linspace(15.0, 30.0, len(thermal))
    thermal["value_interpolated"] = thermal_mean
    thermal["temperature_sd_c"] = thermal_sd
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _cfg_bayesian(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        labels=replace(
            base.labels,
            observation_models={
                "component_a": ObservationModelConfig(family="gaussian")
            },
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="thermal_layer_exceedance",
                layer="prior_layer_a",
                threshold=200.0,
                uncertainty_column="temperature_sd_c",
                p_min=1e-12,
                p_max=1.0 - 1e-12,
                force_prior_predictive=True,
            ),
            "component_b": base.alpha["component_b"],
        },
        inference=replace(
            base.inference,
            gblk_bayesian=replace(
                base.inference.gblk_bayesian,
                cluster_effect=False,
                kleiber_profiles={},
            ),
        ),
        outputs=replace(
            base.outputs,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
        ),
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    component = result.components["component_a"]
    response = component.probability
    np.testing.assert_allclose(
        response["response_predictive_mean"], thermal_mean
    )
    expected_z = 1.6448536269514722
    np.testing.assert_allclose(
        response["response_predictive_lo"],
        thermal_mean - expected_z * thermal_sd,
    )
    np.testing.assert_allclose(
        response["response_predictive_hi"],
        thermal_mean + expected_z * thermal_sd,
    )
    assert component.diagnostics["response_interval_method"] == (
        "analytic_normal_quantile"
    )
    assert not component.diagnostics[
        "response_interval_includes_likelihood_variance"
    ]


def test_top_level_bayesian_runner_resumes_missing_block_without_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=31)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    output_dir = tmp_path / "out"
    base_cfg = _cfg_bayesian(wells_path, output_dir)
    cfg = replace(
        base_cfg,
        inference=replace(
            base_cfg.inference,
            gblk_bayesian=replace(
                base_cfg.inference.gblk_bayesian,
                cluster_effect=False,
            ),
        ),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
            format=(),
        ),
    )
    final_dir = output_dir / "posterior_draws"
    work_dir = output_dir / ".posterior_draws.incomplete"
    marker_path = output_dir / ".probabilistic_run.incomplete.json"
    delegated_projection = project_gblk_bayesian_draw_block
    attempted_ranges: list[tuple[int, int]] = []

    def interrupt_last_projection(state, draw_start, draw_stop):
        attempted_ranges.append((draw_start, draw_stop))
        if draw_start == 6:  # three blocks: [0,3), [3,6), [6,8)
            raise RuntimeError("injected ordinary projection failure")
        return delegated_projection(state, draw_start, draw_stop)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.project_gblk_bayesian_draw_block",
        interrupt_last_projection,
    )

    with pytest.raises(RuntimeError, match="ordinary projection failure"):
        run_probabilistic(fixture.pfa, cfg)

    assert attempted_ranges == [(0, 3), (3, 6), (6, 8)]
    assert marker_path.is_file()
    assert work_dir.is_dir()
    assert not final_dir.exists()
    assert not (output_dir / "manifest.json").exists()
    progress = json.loads(
        (work_dir / "progress.json").read_text(encoding="utf-8")
    )
    assert [
        (block["draw_start"], block["draw_stop"])
        for block in progress["blocks"]
    ] == [(0, 3), (3, 6)]

    def fail_if_refit(*_args, **_kwargs):
        raise AssertionError(
            "an interrupted persisted posterior must not refit"
        )

    projected_ranges: list[tuple[int, int]] = []

    def record_projection(state, draw_start, draw_stop):
        projected_ranges.append((draw_start, draw_stop))
        return delegated_projection(state, draw_start, draw_stop)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_posterior_state",
        fail_if_refit,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.project_gblk_bayesian_draw_block",
        record_projection,
    )

    resumed = run_probabilistic(fixture.pfa, cfg)

    assert projected_ranges == [(6, 8)]
    assert not marker_path.exists()
    assert not work_dir.exists()
    index_path = final_dir / "index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    combined_blocks = []
    for block in index["blocks"]:
        with np.load(final_dir / block["path"], allow_pickle=False) as payload:
            combined_blocks.append(payload["combined_probability"])
    np.testing.assert_allclose(
        resumed.combined["probability"],
        np.concatenate(combined_blocks).mean(axis=0),
    )


def test_streamed_bayesian_restart_rejects_changed_analysis_inputs(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=37)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    output_dir = tmp_path / "out"
    base_cfg = _cfg_bayesian(wells_path, output_dir)
    cfg = replace(
        base_cfg,
        inference=replace(
            base_cfg.inference,
            gblk_bayesian=replace(
                base_cfg.inference.gblk_bayesian,
                cluster_effect=False,
            ),
        ),
        outputs=replace(
            base_cfg.outputs,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
        ),
    )
    run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    changed_wells = fixture.wells.copy()
    changed_wells.loc[0, "heat_label"] = (
        1.0 - changed_wells.loc[0, "heat_label"]
    )
    wells_path.unlink()
    changed_wells.to_file(wells_path, layer="wells", driver="GPKG")
    with pytest.raises(ValueError, match="analysis inputs"):
        run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    wells_path.unlink()
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    gradient = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]["layers"]["gradient"]["model"]
    gradient.loc[0, "value_interpolated"] += 0.25
    with pytest.raises(ValueError, match="analysis inputs"):
        run_gblk_probabilistic(fixture.pfa, cfg, nc=3)


def test_streamed_bayesian_restart_rejects_changed_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=38)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    output_dir = tmp_path / "out"
    base_cfg = _cfg_bayesian(wells_path, output_dir)
    cfg = replace(
        base_cfg,
        inference=replace(
            base_cfg.inference,
            gblk_bayesian=replace(
                base_cfg.inference.gblk_bayesian,
                cluster_effect=False,
            ),
        ),
        outputs=replace(
            base_cfg.outputs,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
        ),
    )
    run_gblk_probabilistic(fixture.pfa, cfg, nc=3)

    monkeypatch.setattr(
        gblk_runner,
        "_probabilistic_implementation_hash",
        lambda: "changed-implementation",
        raising=False,
    )

    with pytest.raises(ValueError, match="implementation"):
        run_gblk_probabilistic(fixture.pfa, cfg, nc=3)


def test_top_level_bayesian_runner_persists_separate_scenario_namespace(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=29)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    output_dir = tmp_path / "out"
    base_cfg = _cfg_bayesian(wells_path, output_dir)
    cfg = replace(
        base_cfg,
        inference=replace(
            base_cfg.inference,
            gblk_bayesian=replace(
                base_cfg.inference.gblk_bayesian, cluster_effect=False
            ),
        ),
        scenarios=(
            ScenarioConfig(name="no_gradient", drop_layers=("gradient",)),
        ),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=True,
            posterior_draw_blocks=True,
            posterior_draw_block_size=3,
            format=(),
        ),
    )

    run_probabilistic(fixture.pfa, cfg)

    baseline = json.loads(
        (output_dir / "posterior_draws/index.json").read_text(encoding="utf-8")
    )
    scenario = json.loads(
        (
            output_dir / "scenarios/no_gradient/posterior_draws/index.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline["scope"] == "baseline"
    assert baseline["cross_scenario_pairing"] == "within_scope_only"
    assert scenario["scope"] == "scenario:no_gradient"
    assert scenario["cross_scenario_pairing"] == "not_identified"
    assert baseline["state"]["fingerprint"] != scenario["state"]["fingerprint"]


# ---------------------------------------------------------------------------
# Credible interval columns present
# ---------------------------------------------------------------------------


def test_p7_bayesian_gblk_component_ci_columns(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    for name in _COMPONENTS:
        prob_gdf = result.components[name].probability
        assert "probability_lo" in prob_gdf.columns, name
        assert "probability_hi" in prob_gdf.columns, name


def test_p7_bayesian_gblk_combined_ci_columns(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    assert "probability_lo" in result.combined.columns
    assert "probability_hi" in result.combined.columns


# ---------------------------------------------------------------------------
# Probability values in [0, 1]
# ---------------------------------------------------------------------------


def test_p7_bayesian_gblk_probability_in_range(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    for name in _COMPONENTS:
        gdf = result.components[name].probability
        for col in ("probability", "probability_lo", "probability_hi"):
            vals = gdf[col].to_numpy()
            assert np.all(vals >= 0.0) and np.all(vals <= 1.0), (
                f"{name}.{col} out of [0,1]"
            )
    for col in ("probability", "probability_lo", "probability_hi"):
        vals = result.combined[col].to_numpy()
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0), (
            f"combined.{col} out of [0,1]"
        )


# ---------------------------------------------------------------------------
# CI ordering: lo ≤ mean ≤ hi
# ---------------------------------------------------------------------------


def test_p7_bayesian_gblk_ci_ordering_components(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    for name in _COMPONENTS:
        gdf = result.components[name].probability
        lo = gdf["probability_lo"].to_numpy()
        mean = gdf["probability"].to_numpy()
        hi = gdf["probability_hi"].to_numpy()
        assert np.all(lo <= mean + 1e-9), f"{name}: lo > mean"
        assert np.all(mean <= hi + 1e-9), f"{name}: mean > hi"


def test_p7_bayesian_gblk_ci_ordering_combined(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    gdf = result.combined
    lo = gdf["probability_lo"].to_numpy()
    mean = gdf["probability"].to_numpy()
    hi = gdf["probability_hi"].to_numpy()
    assert np.all(lo <= mean + 1e-9)
    assert np.all(mean <= hi + 1e-9)


# ---------------------------------------------------------------------------
# Diagnostics carry backend label
# ---------------------------------------------------------------------------


def test_p7_bayesian_gblk_diagnostics_backend(tmp_path: Path) -> None:
    result = _run_bayesian(tmp_path)
    for name in _COMPONENTS:
        diag = result.components[name].diagnostics or {}
        assert diag.get("backend") == "gblk_bayesian"


# ---------------------------------------------------------------------------
# Scalable (non-Bayesian) path unaffected when bayesian flag is off
# ---------------------------------------------------------------------------


def test_p7_standard_gblk_unaffected(tmp_path: Path) -> None:
    result = _run_standard(tmp_path)
    assert isinstance(result, ProbabilisticResult)
    assert not result.skipped
    for name in _COMPONENTS:
        gdf = result.components[name].probability
        assert "probability" in gdf.columns
        assert "probability_lo" not in gdf.columns
        assert "probability_hi" not in gdf.columns


def test_p7_standard_gblk_combined_no_ci(tmp_path: Path) -> None:
    result = _run_standard(tmp_path)
    assert "probability_lo" not in result.combined.columns
    assert "probability_hi" not in result.combined.columns


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_p7_gblk_bayesian_config_roundtrip() -> None:
    cfg = GBLKBayesianConfig(
        enabled=True,
        n_draws=50,
        seed=7,
        ci_level=0.95,
        cor_scale_median=0.15,
        spatial_sd_u=1.2,
        spatial_sd_tail_probability=0.1,
        dirichlet_concentration=3.0,
        separate_ranges=True,
        kleiber_profiles={"bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)},
        cluster_effect=False,
        validate_inla=False,
    )
    restored = GBLKBayesianConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_p7_retired_nested_laplace_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="range_prior_u"):
        GBLKBayesianConfig.from_dict({"range_prior_u": 0.2})


def test_p7_gblk_bayesian_uses_published_paige_prior() -> None:
    from latticekrigx.glk.bayes import PaigeELKPrior

    cfg = GBLKBayesianConfig()
    prior = build_paige_prior(cfg)

    assert isinstance(prior, PaigeELKPrior)
    assert prior.cor_scale_median == cfg.cor_scale_median
    assert prior.spatial_sd_u == cfg.spatial_sd_u
    assert prior.spatial_sd_alpha == cfg.spatial_sd_tail_probability
    assert prior.dirichlet_concentration == cfg.dirichlet_concentration


def test_p7_gblk_bayesian_requires_bivariate_kleiber_profile() -> None:
    rng = np.random.default_rng(13)
    coords = rng.uniform(size=(12, 2))
    labels = rng.binomial(1, 0.5, size=(12, 2)).astype(float)
    with pytest.raises(ValueError, match="kleiber_profiles.bernoulli"):
        fit_gblk_bayesian_joint(
            coords,
            labels,
            coords[:3],
            component_names=("a", "b"),
            bayes_config=GBLKBayesianConfig(enabled=True, n_draws=2),
            nc=3,
        )


def test_p7_gblk_bayesian_forwards_component_specific_coefficient_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(14)
    coords = rng.uniform(size=(12, 2))
    labels = rng.binomial(1, 0.5, size=(12, 1)).astype(float)
    design = np.ones((12, 2, 1))
    design[:, 1, 0] = 2.0
    captured: dict[str, object] = {}

    def fake_fit_joint(*args, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            inference="inla",
            extra={
                "c_draws": np.zeros((2, args[1].shape[1], 1)),
                "fixed_draws": np.array([[[-1.0], [1.0]], [[0.5], [0.25]]]),
            },
        )

    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)
    result = fit_gblk_bayesian_joint(
        coords,
        labels,
        coords[:3],
        component_names=("a",),
        bayes_config=GBLKBayesianConfig(
            enabled=True, n_draws=2, cluster_effect=False
        ),
        fixed_effects=design,
        fixed_effects_grid=design[:3],
        fixed_precision=np.array([[2.0], [3.0]]),
        fixed_prior_mean=np.array([[0.75], [-0.25]]),
        nc=3,
    )

    prior = captured["kwargs"]["coefficient_prior"]
    np.testing.assert_allclose(prior.mean, [0.75, -0.25])
    np.testing.assert_allclose(prior.precision.toarray(), np.diag([2.0, 3.0]))
    assert result.fixed_coef_draws.shape == (2, 2, 1)
    np.testing.assert_allclose(result.p_q_draws[0, :, 0], expit(1.0))
    np.testing.assert_allclose(result.p_q_draws[1, :, 0], expit(1.0))


def test_gaussian_bayesian_stage_uses_identity_response_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    )
    responses = np.array([[1.0], [1.5], [2.0], [2.5]])
    grid = np.array([[0.25, 0.25], [0.75, 0.75]], dtype=float)
    captured: dict[str, object] = {}

    def fake_fit_joint(*args, **kwargs):
        captured["family"] = kwargs["family"]
        captured["link"] = kwargs["link"]
        coefficient_draws = np.zeros((2, args[1].shape[1], 1))
        coefficient_draws[0, :, 0] = 0.25
        coefficient_draws[1, :, 0] = 0.75
        return SimpleNamespace(
            inference="inla",
            extra={
                "c_draws": coefficient_draws,
                "fixed_draws": None,
                "likelihood_precision_draws": np.array([[4.0], [9.0]]),
            },
        )

    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)

    result = fit_gblk_gaussian_bayesian_joint(
        coords,
        responses,
        grid,
        component_names=("heat",),
        bayes_config=GBLKBayesianConfig(
            enabled=True, n_draws=2, cluster_effect=False
        ),
        offsets=np.ones_like(responses),
        grid_offsets=np.ones((len(grid), 1)),
        nc=3,
    )

    assert isinstance(result, GBLKGaussianFitResult)
    assert captured["family"].name == "gaussian"
    assert captured["link"].name == "identity"
    assert result.response_draws.shape == (2, 2, 1)
    np.testing.assert_allclose(
        result.likelihood_precision_draws, [[4.0], [9.0]]
    )
    assert np.all(np.isfinite(result.response_draws))
    assert np.all(result.response_draws > 1.0)
    np.testing.assert_allclose(
        result.response_grid, result.response_draws.mean(axis=0)
    )


def test_p7_gblk_bayesian_uses_lkbox_for_3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(15)
    coords = rng.uniform(size=(12, 3))
    labels = rng.binomial(1, 0.5, size=(12, 1)).astype(float)
    captured: dict[str, object] = {}

    def fake_fit_joint(*args, **_kwargs):
        captured["lkinfo"] = args[2]
        return SimpleNamespace(
            inference="inla",
            extra={
                "c_draws": np.zeros((2, args[1].shape[1], 1)),
                "fixed_draws": None,
            },
        )

    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)
    result = fit_gblk_bayesian_joint(
        coords,
        labels,
        coords[:3],
        component_names=("a",),
        bayes_config=GBLKBayesianConfig(
            enabled=True, n_draws=2, cluster_effect=False
        ),
        nc=3,
        a_wght=6.5,
    )

    assert captured["lkinfo"].geometry.geometry_type == "LKBox"
    assert result.diagnostics["spatial_dimension"] == 3


def test_p7_inference_config_gblk_bayesian_default_off() -> None:
    ic = InferenceConfig()
    assert not ic.gblk_bayesian.enabled


def test_p7_inference_config_from_dict_gblk_bayesian() -> None:
    ic = InferenceConfig.from_dict(
        {
            "backend": "gblk",
            "gblk_bayesian": {"enabled": True, "n_draws": 20},
        }
    )
    assert ic.gblk_bayesian.enabled
    assert ic.gblk_bayesian.n_draws == 20


def test_p7_public_bayesian_fitter_is_the_canonical_array_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(521)
    coords = rng.uniform(0.0, 1.0, size=(48, 2))
    grid = rng.uniform(0.0, 1.0, size=(12, 2))
    offsets = np.column_stack(
        [np.full(len(coords), -0.4), np.full(len(coords), -0.2)]
    )
    grid_offsets = np.column_stack(
        [np.full(len(grid), -0.4), np.full(len(grid), -0.2)]
    )
    latent = np.column_stack(
        [
            0.9 * np.sin(np.pi * coords[:, 0]),
            0.8 * np.cos(np.pi * coords[:, 1]),
        ]
    )
    labels = rng.binomial(1, 1.0 / (1.0 + np.exp(-(offsets + latent))))
    captured: dict[str, object] = {}
    basis_calls: list[np.ndarray] = []
    coefficient_draws = np.zeros((6, 2, 2), dtype=float)
    coefficient_draws[:, 0, 0] = np.linspace(-0.3, 0.3, 6)
    coefficient_draws[:, 0, 1] = np.linspace(0.4, -0.2, 6)
    coefficient_draws[:, 1, :] = [[0.5, -0.25]]

    def fake_compute_basis(locations, _lkinfo, *, normalize):
        assert normalize is False
        locations = np.asarray(locations, dtype=float)
        basis_calls.append(locations.copy())
        return sp.csr_matrix(
            np.column_stack([np.ones(len(locations)), locations[:, 0]])
        )

    def fake_fit_joint(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            c_matrix=np.zeros((args[1].shape[1], 2)),
            fixed_coef=None,
            inference="inla",
            extra={"c_draws": coefficient_draws, "fixed_draws": None},
        )

    monkeypatch.setattr(
        "latticekrigx.basis.assembly.compute_basis", fake_compute_basis
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_backend._BAYESIAN_PREDICTION_CHUNK_SIZE", 5
    )
    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)

    result = fit_gblk_bayesian_joint(
        coords,
        labels,
        grid,
        component_names=("heat", "permeability"),
        offsets=offsets,
        grid_offsets=grid_offsets,
        bayes_config=GBLKBayesianConfig(
            enabled=True,
            n_draws=6,
            seed=19,
            ci_level=0.8,
            kleiber_profiles={
                "bernoulli": KleiberProfileConfig(r0=0.25, r1=0.10)
            },
            cluster_effect=False,
        ),
        nc=3,
        nlevel=1,
        a_wght=4.5,
    )

    assert isinstance(result, GBLKBayesianFitResult)
    assert result.p_q_grid.shape == (12, 2)
    assert result.p_q_interval.shape == (2, 12, 2)
    assert result.p_q_draws.shape == (6, 12, 2)
    assert result.p_joint_grid.shape == (12,)
    assert result.p_joint_interval.shape == (2, 12)
    assert result.p_joint_draws.shape == (6, 12)
    np.testing.assert_allclose(
        result.p_joint_draws,
        np.prod(result.p_q_draws, axis=-1),
    )
    grid_basis = sp.vstack(
        [
            sp.csr_matrix(np.column_stack([np.ones(len(chunk)), chunk[:, 0]]))
            for chunk in basis_calls[1:]
        ]
    )
    expected_eta = np.empty((6, len(grid), 2), dtype=float)
    for draw_index in range(6):
        expected_eta[draw_index] = (
            np.asarray(grid_basis @ coefficient_draws[draw_index])
            + grid_offsets
        )
    np.testing.assert_allclose(result.p_q_draws, expit(expected_eta))
    assert [len(chunk) for chunk in basis_calls] == [len(coords), 5, 5, 2]
    assert result.diagnostics["backend"] == "gblk_bayesian"
    assert result.diagnostics["estimator"] == "paige_inla"
    assert result.diagnostics["posterior_scope"] == "spatial_posterior"
    assert result.diagnostics["fixed_effects_in_joint_likelihood"] is False
    assert captured["kwargs"]["inference"] == "inla"
    assert captured["kwargs"]["r_0"] == 0.25
    assert captured["kwargs"]["r_1"] == 0.10
    assert "Phi_pred" not in captured["kwargs"]
    assert "X_pred" not in captured["kwargs"]
    assert "offsets_pred" not in captured["kwargs"]
