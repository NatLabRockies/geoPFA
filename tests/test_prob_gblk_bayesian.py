"""Tests for the optional Bayesian GBLK path (P7-S01).

Requires the ``dev-gblk`` pixi environment (``latticekrigx`` +
``scikit-sparse``).  Skipped otherwise.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit, ndtr

pytest.importorskip("latticekrigx.glk.bayes.paige")

from geopfa.prob.config import (  # noqa: E402
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GBLKBayesianConfig,
    GridConfig,
    InferenceConfig,
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
from geopfa.prob.gblk_runner import run_gblk_probabilistic  # noqa: E402
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
        spatial_field=SpatialFieldConfig(enabled=True, n_levels=1),
        inference=InferenceConfig(
            backend="gblk",
            gblk_bayesian=GBLKBayesianConfig(
                enabled=True,
                n_draws=_N_DRAWS,
                seed=42,
                ci_level=0.9,
                kleiber_r0=0.25,
                kleiber_r1=0.10,
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
    )
    captured: dict[str, np.ndarray] = {}

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
    assert "response_mean" in result.components["component_a"].probability
    expected_joint = (
        heat_draws * result.component_probability_draws["component_b"]
    ).mean(axis=0)
    np.testing.assert_allclose(
        result.combined["probability"].to_numpy(), expected_joint
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
            "component_a": PredictiveStackingResult(0.0, 0.2, 0.8, 0.2, 20),
            "component_b": PredictiveStackingResult(1.0, 0.8, 0.2, 0.2, 20),
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
        assert 0.0 <= diagnostics["predictive_stacking_weight"] <= 1.0


def test_bayesian_gblk_honors_geometric_mean_combination(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=30, seed=23)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = replace(
        _cfg_bayesian(wells_path, tmp_path / "out"),
        combination=CombinationConfig(rule="geometric_mean"),
    )

    result = run_gblk_probabilistic(fixture.pfa, cfg)

    ordered = np.stack(
        [
            result.component_probability_draws[name]
            for name in sorted(result.component_probability_draws)
        ],
        axis=2,
    )
    expected = np.exp(np.mean(np.log(np.clip(ordered, 1e-12, 1.0)), axis=2))
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
    resumed = run_probabilistic(fixture.pfa, cfg)
    assert resumed.posterior_draw_index == index_path
    np.testing.assert_allclose(
        resumed.combined["probability"], result.combined["probability"]
    )


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
                base.inference.gblk_bayesian, cluster_effect=False
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
    expected = run_probabilistic(fixture.pfa, cfg)

    final_dir = output_dir / "posterior_draws"
    index = json.loads((final_dir / "index.json").read_text(encoding="utf-8"))
    completed_blocks = index["blocks"][:-1]
    missing_block = index["blocks"][-1]
    work_dir = output_dir / ".posterior_draws.incomplete"
    final_dir.replace(work_dir)
    (work_dir / missing_block["path"]).unlink()
    (work_dir / "index.json").unlink()
    progress = {
        "schema_version": index["schema_version"],
        "state_fingerprint": index["state"]["fingerprint"],
        "scope": index["scope"],
        "component_names": index["component_names"],
        "n_draws": index["n_draws"],
        "n_cells": index["n_cells"],
        "n_components": index["n_components"],
        "block_size": index["block_size"],
        "seed": index["seed"],
        "combination_rule": index["combination_rule"],
        "coordinate_columns": index["coordinate_columns"],
        "crs": index["crs"],
        "coordinates": index["coordinates"],
        "state_metadata": index["state"]["metadata"],
        "state_arrays": index["state"]["arrays"],
        "blocks": completed_blocks,
    }
    (work_dir / "progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )

    def fail_if_refit(*_args, **_kwargs):
        raise AssertionError(
            "an interrupted persisted posterior must not refit"
        )

    projected_ranges: list[tuple[int, int]] = []

    def record_projection(state, draw_start, draw_stop):
        projected_ranges.append((draw_start, draw_stop))
        return project_gblk_bayesian_draw_block(state, draw_start, draw_stop)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.fit_gblk_bayesian_posterior_state",
        fail_if_refit,
    )
    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.project_gblk_bayesian_draw_block",
        record_projection,
    )

    resumed = run_probabilistic(fixture.pfa, cfg)

    assert projected_ranges == [
        (missing_block["draw_start"], missing_block["draw_stop"])
    ]
    assert not work_dir.exists()
    assert (final_dir / "index.json").is_file()
    np.testing.assert_allclose(
        resumed.combined["probability"], expected.combined["probability"]
    )


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
        kleiber_r0=0.25,
        kleiber_r1=0.10,
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
    with pytest.raises(ValueError, match="kleiber_r0.*kleiber_r1"):
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
            kleiber_r0=0.25,
            kleiber_r1=0.10,
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
