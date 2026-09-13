"""Tests for ProbabilisticConfig dataclass and config parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geopfa.prob.config import (
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
    ProbabilisticConfig,
    PredictiveStackingConfig,
    RegularizationConfig,
    ScenarioConfig,
    SpatialFieldConfig,
    load_probabilistic_config,
)


def _minimal_config_dict() -> dict:
    return {
        "enabled": True,
        "output_dir": "outputs/probabilistic/",
        "dimensions": "2d",
        "labels": {
            "source": "data/wells.gpkg",
            "id_col": "well_id",
            "label_columns": {
                "heat": "heat_label",
                "reservoir": "reservoir_label",
            },
        },
        "alpha": {
            "heat": {"mode": "scalar", "scalar_fallback_pr0": 0.5},
            "reservoir": {"mode": "scalar", "scalar_fallback_pr0": 0.4},
        },
    }


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_minimal_config_parses_with_defaults() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert cfg.enabled is True
    assert cfg.dimensions == "2d"
    assert cfg.output_dir == Path("outputs/probabilistic/")
    assert cfg.labels.id_col == "well_id"
    assert cfg.labels.label_columns["heat"] == "heat_label"
    assert cfg.alpha["heat"].mode == "scalar"
    # Defaults applied:
    assert cfg.evidence.sparse_binary_threshold == pytest.approx(0.90)
    assert cfg.spatial_field.backend == "latticekrigx"
    assert cfg.inference.backend == "gblk"
    assert cfg.calibration.method == "none"
    assert cfg.cross_validation.n_folds == 5
    assert cfg.combination.rule == "product"
    assert cfg.outputs.probability_rasters is True
    assert cfg.outputs.posterior_draw_blocks is False
    assert cfg.outputs.posterior_draw_block_size == 20


@pytest.mark.parametrize("reserved_name", ["combined", "Combined"])
def test_component_name_cannot_collide_with_combined_output(
    reserved_name: str,
) -> None:
    raw = _minimal_config_dict()
    raw["labels"]["label_columns"] = {reserved_name: "heat_label"}
    raw["alpha"] = {
        reserved_name: {"mode": "scalar", "scalar_fallback_pr0": 0.5}
    }

    with pytest.raises(ValueError, match="reserved.*combined"):
        ProbabilisticConfig.from_dict(raw)


def test_component_names_must_be_portable() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["label_columns"] = {"../heat": "heat_label"}
    raw["alpha"] = {"../heat": {"mode": "scalar", "scalar_fallback_pr0": 0.5}}

    with pytest.raises(ValueError, match="portable surface name"):
        ProbabilisticConfig.from_dict(raw)


def test_component_names_must_not_have_case_normalized_collisions() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["label_columns"] = {"Heat": "heat_label"}
    raw["alpha"] = {"heat": {"mode": "scalar", "scalar_fallback_pr0": 0.5}}

    with pytest.raises(ValueError, match="case-insensitive collision"):
        ProbabilisticConfig.from_dict(raw)


def test_posterior_draw_output_config_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True, "cluster_effect": False},
    }
    raw["outputs"] = {
        "posterior_draw_blocks": True,
        "posterior_draw_block_size": 7,
        "format": [],
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.outputs.posterior_draw_blocks is True
    assert cfg.outputs.posterior_draw_block_size == 7
    assert cfg.to_dict()["outputs"] == raw["outputs"] | {
        "probability_rasters": True,
        "uncertainty_rasters": True,
        "calibration_artifacts": True,
        "decision_artifacts": True,
        "scenarios": True,
    }


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_posterior_draw_block_size_rejects_nonpositive_or_coerced_values(
    value: object,
) -> None:
    raw = _minimal_config_dict()
    raw["outputs"] = {"posterior_draw_block_size": value}

    with pytest.raises(ValueError, match="posterior_draw_block_size"):
        ProbabilisticConfig.from_dict(raw)


def test_posterior_draw_blocks_reject_truthy_string() -> None:
    raw = _minimal_config_dict()
    raw["outputs"] = {"posterior_draw_blocks": "true"}

    with pytest.raises(ValueError, match="posterior_draw_blocks"):
        ProbabilisticConfig.from_dict(raw)


def test_posterior_draw_blocks_require_bayesian_gblk() -> None:
    raw = _minimal_config_dict()
    raw["outputs"] = {"posterior_draw_blocks": True}

    with pytest.raises(
        ValueError, match="posterior_draw_blocks.*Bayesian GBLK"
    ):
        ProbabilisticConfig.from_dict(raw)


def test_bayesian_mode_rejects_only_fixed_prior_components() -> None:
    raw = _minimal_config_dict()
    for component in raw["alpha"].values():
        component["force_prior_predictive"] = True
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }

    with pytest.raises(ValueError, match="no stochastic components"):
        ProbabilisticConfig.from_dict(raw)


def test_incremental_posterior_draw_blocks_reject_inla_grid_materialization() -> (
    None
):
    raw = _minimal_config_dict()
    raw["outputs"] = {"posterior_draw_blocks": True}
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True, "cluster_effect": True},
    }

    with pytest.raises(ValueError, match="cluster_effect=false"):
        ProbabilisticConfig.from_dict(raw)


def test_thermal_layer_exceedance_config_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
        "p_min": 0.001,
        "p_max": 0.999,
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    heat = cfg.alpha["heat"]
    assert heat.mode == "thermal_layer_exceedance"
    assert heat.uncertainty_column == "temperature_sd_c"
    assert cfg.to_dict()["alpha"]["heat"]["threshold"] == 400.0


def test_gaussian_component_observation_model_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.observation_model_for("heat") == ObservationModelConfig(
        family="gaussian", response_scale=50.0
    )
    assert cfg.labels.observation_model_for("reservoir").family == "bernoulli"
    assert (
        cfg.to_dict()["labels"]["observation_models"]
        == raw["labels"]["observation_models"]
    )


def test_gaussian_component_requires_explicit_response_scale() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}

    with pytest.raises(ValueError, match="response_scale"):
        ProbabilisticConfig.from_dict(raw)


def test_labels_csv_coordinate_contract_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["labels"].update(
        {
            "source": "data/wells.csv",
            "source_crs": "EPSG:32611",
            "x_col": "easting_m",
            "y_col": "northing_m",
            "depth_col": "depth_m",
        }
    )

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.source_crs == "EPSG:32611"
    assert cfg.labels.x_col == "easting_m"
    assert cfg.labels.y_col == "northing_m"
    assert cfg.labels.z_col is None
    assert cfg.labels.depth_col == "depth_m"
    labels_payload = cfg.to_dict()["labels"]
    assert labels_payload["source_crs"] == "EPSG:32611"
    assert labels_payload["x_col"] == "easting_m"
    assert labels_payload["y_col"] == "northing_m"
    assert labels_payload["depth_col"] == "depth_m"


def test_labels_allow_distinct_cartesian_z_and_scientific_depth_columns() -> (
    None
):
    raw = _minimal_config_dict()
    raw["labels"].update(
        {
            "source": "data/wells.csv",
            "source_crs": "EPSG:32611",
            "x_col": "x_m",
            "y_col": "y_m",
            "z_col": "z_m",
            "depth_col": "depth_m",
        }
    )

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.z_col == "z_m"
    assert cfg.labels.depth_col == "depth_m"
    assert cfg.to_dict()["labels"]["z_col"] == "z_m"
    assert cfg.to_dict()["labels"]["depth_col"] == "depth_m"


def test_explicit_prior_only_gaussian_does_not_require_bayesian_fit() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]["label_columns"]["heat"]
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
        "force_prior_predictive": True,
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.observation_model_for("heat").family == "gaussian"
    assert cfg.labels.observation_model_for("heat").response_scale is None
    assert cfg.alpha["heat"].force_prior_predictive is True
    assert cfg.inference.gblk_bayesian.enabled is False


def test_labelled_gaussian_can_explicitly_bypass_outcome_fitting() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
        "force_prior_predictive": True,
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.observation_model_for("heat").family == "gaussian"
    assert cfg.alpha["heat"].force_prior_predictive is True
    assert cfg.inference.gblk_bayesian.enabled is False


def test_all_prior_only_gaussian_config_does_not_require_label_mapping() -> (
    None
):
    raw = _minimal_config_dict()
    raw["labels"]["label_columns"] = {}
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}
    raw["alpha"] = {
        "heat": {
            "mode": "thermal_layer_exceedance",
            "layer": "temperature_model",
            "threshold": 400.0,
            "uncertainty_column": "temperature_sd_c",
            "force_prior_predictive": True,
        }
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.label_columns == {}
    assert cfg.labels.observation_model_for("heat").family == "gaussian"


def test_all_fixed_prior_gblk_config_does_not_require_labels_block() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]
    raw["alpha"] = {
        "heat": {
            "mode": "scalar",
            "scalar_fallback_pr0": 0.4,
            "force_prior_predictive": True,
        }
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.source is None
    assert cfg.labels.id_col is None
    assert cfg.labels.label_columns == {}


def test_prior_only_gaussian_can_declare_only_observation_model() -> None:
    raw = _minimal_config_dict()
    raw["labels"] = {"observation_models": {"heat": {"family": "gaussian"}}}
    raw["alpha"] = {
        "heat": {
            "mode": "thermal_layer_exceedance",
            "layer": "temperature_model",
            "threshold": 400.0,
            "uncertainty_column": "temperature_sd_c",
            "force_prior_predictive": True,
        }
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.labels.source is None
    assert cfg.labels.id_col is None
    assert cfg.labels.label_columns == {}
    assert cfg.labels.observation_model_for("heat").family == "gaussian"


def test_data_informed_config_requires_complete_labels_contract() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]
    raw["alpha"] = {"heat": {"mode": "scalar"}}

    with pytest.raises(
        ValueError,
        match=(
            "data-informed components require labels.source, labels.id_col, "
            "and non-empty labels.label_columns"
        ),
    ):
        ProbabilisticConfig.from_dict(raw)


def test_site_selection_requires_complete_labels_contract() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]
    raw["alpha"] = {
        "heat": {
            "mode": "scalar",
            "force_prior_predictive": True,
        }
    }
    raw["site_selection"] = {
        "mode": "joint_binary",
        "candidate_source": "candidates.csv",
        "id_col": "candidate_id",
        "outcome_feature_columns": ["temperature"],
        "selection_feature_columns": ["road_distance"],
    }

    with pytest.raises(
        ValueError,
        match=(
            "site_selection requires labels.source, labels.id_col, and "
            "non-empty labels.label_columns"
        ),
    ):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    "mode", ["thermal_exceedance", "thermal_layer_exceedance"]
)
def test_thermal_alpha_defaults_to_near_open_probability_bounds(
    mode: str,
) -> None:
    direct_kwargs = (
        {"thermal_raster": "temperature.tif"}
        if mode == "thermal_exceedance"
        else {"layer": "temperature_model"}
    )
    direct = AlphaModeConfig(
        mode=mode,
        threshold=350.0,
        force_prior_predictive=True,
        **direct_kwargs,
    )
    parsed = AlphaModeConfig.from_dict(
        {
            "mode": mode,
            "threshold": 350.0,
            "force_prior_predictive": True,
            **direct_kwargs,
        },
        "heat",
    )

    assert direct.p_min == pytest.approx(1e-12)
    assert direct.p_max == pytest.approx(1.0 - 1e-12)
    assert parsed.p_min == direct.p_min
    assert parsed.p_max == direct.p_max


def test_layer_logit_retains_material_probability_bound_defaults() -> None:
    direct = AlphaModeConfig(mode="layer_logit", layer="favorability")
    parsed = AlphaModeConfig.from_dict(
        {"mode": "layer_logit", "layer": "favorability"},
        "heat",
    )

    assert direct.p_min == pytest.approx(0.2)
    assert direct.p_max == pytest.approx(0.8)
    assert parsed.p_min == direct.p_min
    assert parsed.p_max == direct.p_max


def test_csv_labels_reject_unused_vector_layer_selector() -> None:
    with pytest.raises(ValueError, match="labels.layer.*CSV"):
        LabelsConfig(
            source="wells.csv",
            id_col="well_id",
            label_columns={"heat": "heat_label"},
            source_crs="EPSG:32611",
            x_col="x",
            y_col="y",
            layer="wells",
        )


@pytest.mark.parametrize(
    ("mode_config", "unused_field"),
    [
        ({"mode": "scalar", "threshold": 350.0}, "threshold"),
        ({"mode": "scalar", "p_min": 0.1}, "p_min"),
        (
            {
                "mode": "layer_logit",
                "layer": "favorability",
                "thermal_raster": "temperature.tif",
            },
            "thermal_raster",
        ),
        (
            {
                "mode": "multi_layer",
                "layers": ["a", "b"],
                "uncertainty_column": "temperature_sd_c",
            },
            "uncertainty_column",
        ),
        (
            {
                "mode": "thermal_exceedance",
                "thermal_raster": "temperature.tif",
                "threshold": 350.0,
                "layer": "temperature",
            },
            "layer",
        ),
        (
            {
                "mode": "thermal_layer_exceedance",
                "layer": "temperature",
                "threshold": 350.0,
                "uncertainty_raster": "temperature_sd.tif",
            },
            "uncertainty_raster",
        ),
    ],
)
def test_alpha_modes_reject_fields_they_do_not_use(
    mode_config: dict[str, object], unused_field: str
) -> None:
    with pytest.raises(ValueError, match=unused_field):
        AlphaModeConfig.from_dict(mode_config, "heat")


def test_direct_scalar_alpha_rejects_unused_probability_bounds() -> None:
    with pytest.raises(ValueError, match="scalar.*p_min"):
        AlphaModeConfig(mode="scalar", p_min=0.1)


@pytest.mark.parametrize(
    ("mode", "source_field", "source_value", "uncertainty_field"),
    [
        (
            "thermal_layer_exceedance",
            "layer",
            "temperature_model",
            "uncertainty_column",
        ),
        (
            "thermal_exceedance",
            "thermal_raster",
            "temperature.tif",
            "uncertainty_raster",
        ),
    ],
)
def test_prior_only_gaussian_requires_declared_response_uncertainty(
    mode: str,
    source_field: str,
    source_value: str,
    uncertainty_field: str,
) -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}
    raw["alpha"]["heat"] = {
        "mode": mode,
        source_field: source_value,
        "threshold": 400.0,
        "force_prior_predictive": True,
    }

    with pytest.raises(
        ValueError,
        match=rf"Gaussian prior-only.*{uncertainty_field}",
    ):
        ProbabilisticConfig.from_dict(raw)


def test_prior_only_gaussian_rejects_logit_evidence_prior() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {"heat": {"family": "gaussian"}}
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
        "force_prior_predictive": True,
        "use_evidence_prior": True,
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }

    with pytest.raises(
        ValueError,
        match="Gaussian prior-only.*use_evidence_prior",
    ):
        ProbabilisticConfig.from_dict(raw)


def test_prior_only_gaussian_rejects_unused_response_scale() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
        "force_prior_predictive": True,
    }

    with pytest.raises(
        ValueError,
        match="Gaussian prior-only.*response_scale",
    ):
        ProbabilisticConfig.from_dict(raw)


def test_unlabelled_observation_model_requires_explicit_prior_only_alpha() -> (
    None
):
    raw = _minimal_config_dict()
    del raw["labels"]["label_columns"]["heat"]
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
        "uncertainty_column": "temperature_sd_c",
    }

    with pytest.raises(ValueError, match="force_prior_predictive"):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_config_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {"enabled": True},
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.inference.predictive_stacking == PredictiveStackingConfig(
        enabled=True
    )
    assert cfg.to_dict()["inference"]["predictive_stacking"] == {
        "enabled": True
    }


def test_predictive_stacking_minimum_training_wells_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "minimum_training_wells": 3,
        },
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.inference.predictive_stacking.minimum_training_wells == 3
    assert cfg.to_dict()["inference"]["predictive_stacking"] == {
        "enabled": True,
        "minimum_training_wells": 3,
    }


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_predictive_stacking_minimum_training_wells_must_be_positive_integer(
    value: object,
) -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "minimum_training_wells": value,
        },
    }

    with pytest.raises(ValueError, match="minimum_training_wells"):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_target_depth_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = "3d"
    raw["labels"]["depth_col"] = "depth_m"
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "validation_depths_m": {"heat": 3_000.0},
        },
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.inference.predictive_stacking.validation_depths_m == {
        "heat": 3_000.0
    }
    assert cfg.to_dict()["inference"]["predictive_stacking"] == {
        "enabled": True,
        "validation_depths_m": {"heat": 3_000.0},
    }


def test_predictive_stacking_target_depth_requires_3d() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "validation_depths_m": {"heat": 3_000.0},
        },
    }

    with pytest.raises(ValueError, match="validation_depths_m.*3d"):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_target_depth_rejects_unknown_component() -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = "3d"
    raw["labels"]["depth_col"] = "depth_m"
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "validation_depths_m": {"not_a_component": 3_000.0},
        },
    }

    with pytest.raises(
        ValueError, match="validation_depths_m.*not_a_component"
    ):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_target_depth_requires_scientific_depth() -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = "3d"
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {
            "enabled": True,
            "validation_depths_m": {"heat": 3_000.0},
        },
    }

    with pytest.raises(ValueError, match="validation_depths_m.*depth_col"):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_requires_bayesian_gblk() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {"predictive_stacking": {"enabled": True}}

    with pytest.raises(ValueError, match="predictive_stacking.*Bayesian GBLK"):
        ProbabilisticConfig.from_dict(raw)


def test_predictive_stacking_rejects_incremental_draw_storage() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True, "cluster_effect": False},
        "predictive_stacking": {"enabled": True},
    }
    raw["outputs"] = {"posterior_draw_blocks": True}

    with pytest.raises(ValueError, match="posterior_draw_blocks=false"):
        ProbabilisticConfig.from_dict(raw)


def test_gaussian_component_requires_bayesian_thermal_gblk() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }

    with pytest.raises(ValueError, match="Gaussian.*Bayesian GBLK"):
        ProbabilisticConfig.from_dict(raw)


def test_gaussian_component_requires_thermal_prior_mean() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }

    with pytest.raises(ValueError, match="thermal.*alpha"):
        ProbabilisticConfig.from_dict(raw)


def test_gaussian_component_rejects_incremental_draw_storage() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True, "cluster_effect": False},
    }
    raw["outputs"] = {"posterior_draw_blocks": True}

    with pytest.raises(ValueError, match="Gaussian.*posterior_draw_blocks"):
        ProbabilisticConfig.from_dict(raw)


def test_gaussian_stacking_requires_prior_predictive_uncertainty() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    raw["alpha"]["heat"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model",
        "threshold": 400.0,
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {"enabled": True},
    }

    with pytest.raises(ValueError, match="predictive stacking.*uncertainty"):
        ProbabilisticConfig.from_dict(raw)


def test_observation_model_rejects_unknown_component() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "ghost": {"family": "gaussian", "response_scale": 50.0}
    }

    with pytest.raises(ValueError, match="ghost"):
        ProbabilisticConfig.from_dict(raw)


def test_explicit_gaussian_evidence_priors_roundtrip_without_layer_weights() -> (
    None
):
    raw = _minimal_config_dict()
    raw["alpha"]["reservoir"].update(
        {"force_prior_predictive": True, "use_evidence_prior": True}
    )
    raw["labels"]["label_columns"] = {"heat": "heat_label"}
    raw["evidence"] = {
        "regularization": {
            "prior_means": {"reservoir:fault_distance": 0.7},
            "prior_precisions": {"reservoir:fault_distance": 4.0},
        }
    }
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.alpha["reservoir"].use_evidence_prior is True
    assert cfg.evidence.regularization.prior_means == {
        "reservoir:fault_distance": 0.7
    }
    assert cfg.evidence.regularization.prior_precisions == {
        "reservoir:fault_distance": 4.0
    }
    assert cfg.to_dict()["evidence"]["regularization"] == raw["evidence"][
        "regularization"
    ] | {"C": 1.0, "per_feature_weights": {}}


def test_regularization_rejects_removed_play_type_key() -> None:
    raw = _minimal_config_dict()
    raw["evidence"] = {"regularization": {"play_type": "extensional"}}

    with pytest.raises(
        ValueError,
        match=r"unknown evidence\.regularization config key.*play_type",
    ):
        ProbabilisticConfig.from_dict(raw)


def test_prediction_support_evidence_standardization_roundtrips() -> None:
    raw = _minimal_config_dict()
    raw["evidence"] = {"standardization": "prediction_support"}

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.evidence.standardization == "prediction_support"
    assert cfg.to_dict()["evidence"]["standardization"] == "prediction_support"


def test_unknown_evidence_standardization_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["evidence"] = {"standardization": "global_magic"}

    with pytest.raises(ValueError, match="evidence.standardization"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize("precision", [0.0, -1.0])
def test_explicit_evidence_prior_precision_must_be_positive(
    precision: float,
) -> None:
    raw = _minimal_config_dict()
    raw["evidence"] = {
        "regularization": {"prior_precisions": {"heat:density": precision}}
    }

    with pytest.raises(ValueError, match="prior_precisions"):
        ProbabilisticConfig.from_dict(raw)


def test_full_config_roundtrips_via_to_dict() -> None:
    raw = {
        "enabled": True,
        "output_dir": "out/",
        "dimensions": "3d",
        "grid": {},
        "labels": {
            "source": "wells.gpkg",
            "layer": "great_basin",
            "id_col": "well_id",
            "label_columns": {"heat": "heat_label"},
            "min_wells_for_fit": 6,
            "pu_mode": "off",
        },
        "alpha": {
            "heat": {
                "mode": "thermal_exceedance",
                "thermal_raster": "thermal.tif",
                "threshold": 200.0,
                "uncertainty_raster": "thermal_sd.tif",
                "p_min": 0.1,
                "p_max": 0.9,
                "scalar_fallback_pr0": 0.3,
            },
        },
        "evidence": {
            "regularization": {
                "C": 0.5,
                "per_feature_weights": {"layer_a": 2.0},
            },
            "include_layers": ["layer_a", "layer_b"],
            "exclude_layers": ["bad_layer"],
            "sparse_binary_threshold": 0.95,
            "coordinate_blacklist": ["X", "Y"],
        },
        "spatial_field": {
            "enabled": True,
            "backend": "latticekrigx",
            "n_levels": 3,
            "lattice_centers_per_dimension": 4,
            "coordinate_scaling": "physical_isotropic",
        },
        "inference": {
            "backend": "gblk",
            "gblk_bayesian": {
                "enabled": True,
                "n_draws": 500,
                "seed": 42,
                "ci_level": 0.9,
            },
        },
        "calibration": {
            "method": "none",
            "fit_on": "block_cv",
            "n_bins": 8,
        },
        "cross_validation": {
            "n_folds": 8,
            "block_type": "grid",
            "block_size_km": 25.0,
            "grid_size": 5,
        },
        "combination": {"rule": "product"},
        "scenarios": [
            {
                "name": "full",
                "include_priors": True,
                "include_spatial": True,
                "drop_layers": [],
            },
            {
                "name": "no_priors",
                "include_priors": False,
                "include_spatial": True,
                "drop_layers": [],
            },
        ],
        "outputs": {
            "probability_rasters": True,
            "uncertainty_rasters": True,
            "calibration_artifacts": False,
            "decision_artifacts": True,
            "scenarios": True,
            "format": ["parquet", "vtk"],
        },
    }
    cfg = ProbabilisticConfig.from_dict(raw)
    out = cfg.to_dict()
    # Round-trip preserves user-visible structure (Path objects become strings):
    assert out["dimensions"] == raw["dimensions"]
    assert out["grid"]["nx"] is None
    assert out["labels"]["pu_mode"] == "off"
    assert out["spatial_field"]["n_levels"] == 3
    assert out["spatial_field"]["lattice_centers_per_dimension"] == 4
    assert out["spatial_field"]["coordinate_scaling"] == "physical_isotropic"
    assert out["alpha"]["heat"]["mode"] == "thermal_exceedance"
    assert (
        out["evidence"]["regularization"]["per_feature_weights"]["layer_a"]
        == 2.0
    )
    assert out["spatial_field"]["backend"] == "latticekrigx"
    assert out["inference"]["gblk_bayesian"]["n_draws"] == 500
    assert out["calibration"]["method"] == "none"
    assert out["cross_validation"]["block_type"] == "grid"
    assert out["combination"]["rule"] == "product"
    assert out["outputs"]["format"] == ["parquet", "vtk"]
    assert len(out["scenarios"]) == 2


def test_nondefault_grid_override_fails_closed() -> None:
    raw = _minimal_config_dict()
    raw["grid"] = {"nx": 100, "ny": 80, "crs": "EPSG:32611"}

    with pytest.raises(ValueError, match="grid overrides are not implemented"):
        ProbabilisticConfig.from_dict(raw)


def test_gblk_rejects_non_latticekrigx_spatial_backend() -> None:
    raw = _minimal_config_dict()
    raw["spatial_field"] = {"enabled": True, "backend": "rbf"}

    with pytest.raises(ValueError, match="backend='latticekrigx'"):
        ProbabilisticConfig.from_dict(raw)


def test_spatial_field_rejects_unknown_coordinate_scaling() -> None:
    with pytest.raises(ValueError, match="coordinate_scaling"):
        SpatialFieldConfig.from_dict({"coordinate_scaling": "vertical_magic"})


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("spatial_field", "kernel", "matern32"),
        ("spatial_field", "n_inducing", 100),
        ("spatial_field", "lengthscale_lower_frac", 0.05),
        ("spatial_field", "lengthscale_upper_frac", 0.3),
        ("spatial_field", "optimize_restarts", 2),
        ("labels", "label_quality_col", "quality"),
        ("labels", "label_source_col", "source"),
        ("calibration", "report_temperature", True),
        ("combination", "barrier_inverse", False),
    ],
)
def test_retired_no_effect_config_fields_fail_closed(
    block: str,
    key: str,
    value: object,
) -> None:
    """A config key must not survive when no execution path honors it."""
    raw = _minimal_config_dict()
    raw.setdefault(block, {})[key] = value

    with pytest.raises(ValueError, match=key):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize("fit_on", ["holdout", "in_sample"])
def test_calibration_rejects_unimplemented_fit_populations(
    fit_on: str,
) -> None:
    with pytest.raises(ValueError, match="fit_on"):
        CalibrationConfig.from_dict({"fit_on": fit_on})


def test_gblk_config_rejects_unimplemented_posthoc_calibration() -> None:
    raw = _minimal_config_dict()
    raw["calibration"] = {"method": "isotonic", "fit_on": "block_cv"}

    with pytest.raises(ValueError, match="GBLK.*calibration.method='none'"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("dimensions", "formats", "match_text"),
    [
        ("3d", ["geotiff", "csv"], "GeoTIFF.*3-D"),
        ("2d", ["vtk", "csv"], "VTK.*2-D"),
    ],
)
def test_output_formats_must_match_model_dimension(
    dimensions: str,
    formats: list[str],
    match_text: str,
) -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = dimensions
    raw["outputs"] = {"format": formats}

    with pytest.raises(ValueError, match=match_text):
        ProbabilisticConfig.from_dict(raw)


def test_3d_config_uses_dimension_appropriate_default_outputs() -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = "3d"

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.outputs.format == ("vtk", "csv")


@pytest.mark.parametrize(
    ("factory", "match_text"),
    [
        (lambda: AlphaModeConfig(mode="vibes"), "alpha.mode"),
        (
            lambda: SpatialFieldConfig(backend="ordinary_kriging"),
            "spatial_field.backend",
        ),
        (lambda: CalibrationConfig(method="beta"), "calibration.method"),
        (
            lambda: CrossValidationConfig(block_type="random"),
            "cross_validation.block_type",
        ),
        (lambda: OutputsConfig(format=("png",)), "outputs.format"),
        (
            lambda: OutputsConfig(probability_rasters="false"),
            "output toggles",
        ),
    ],
)
def test_direct_subconfig_construction_rejects_unknown_modes(
    factory, match_text: str
) -> None:
    """Programmatic callers receive the same enum validation as JSON callers."""
    with pytest.raises(ValueError, match=match_text):
        factory()


# ---------------------------------------------------------------------------
# JSON file loading
# ---------------------------------------------------------------------------


def test_load_probabilistic_config_reads_top_level_block(
    tmp_path: Path,
) -> None:
    pfa_config = {
        "criteria": {"geologic": {"weight": 1.0, "components": {}}},
        "probabilistic": _minimal_config_dict(),
    }
    cfg_path = tmp_path / "pfa_config.json"
    cfg_path.write_text(json.dumps(pfa_config))

    cfg = load_probabilistic_config(cfg_path)
    assert isinstance(cfg, ProbabilisticConfig)
    assert cfg.enabled is True


def test_load_prior_only_config_without_labels_block(tmp_path: Path) -> None:
    raw = _minimal_config_dict()
    del raw["labels"]
    raw["alpha"] = {
        "heat": {
            "mode": "scalar",
            "force_prior_predictive": True,
        }
    }
    cfg_path = tmp_path / "pfa_config.json"
    cfg_path.write_text(json.dumps({"probabilistic": raw}))

    cfg = load_probabilistic_config(cfg_path)

    assert cfg.labels.source is None
    assert cfg.labels.id_col is None
    assert cfg.labels.label_columns == {}


def test_load_probabilistic_config_resolves_declared_paths_from_config_dir(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "study" / "config"
    config_dir.mkdir(parents=True)
    raw = _minimal_config_dict()
    raw["output_dir"] = "../outputs/run"
    raw["labels"]["source"] = "../data/wells.gpkg"
    raw["alpha"]["heat"].update(
        {
            "mode": "thermal_exceedance",
            "thermal_raster": "../data/temperature.tif",
            "uncertainty_raster": "../data/temperature_sd.tif",
            "threshold": 150.0,
        }
    )
    raw["site_selection"] = {
        "mode": "joint_binary",
        "candidate_source": "../data/candidates.csv",
        "id_col": "well_id",
        "outcome_feature_columns": ["temperature"],
        "selection_feature_columns": ["roads"],
    }
    cfg_path = config_dir / "pfa_config.json"
    cfg_path.write_text(json.dumps({"probabilistic": raw}))

    cfg = load_probabilistic_config(cfg_path)

    expected_data = (config_dir / "../data").resolve()
    assert cfg.output_dir == (config_dir / "../outputs/run").resolve()
    assert Path(cfg.labels.source) == expected_data / "wells.gpkg"
    assert Path(cfg.alpha["heat"].thermal_raster) == (
        expected_data / "temperature.tif"
    )
    assert Path(cfg.alpha["heat"].uncertainty_raster) == (
        expected_data / "temperature_sd.tif"
    )
    assert Path(cfg.site_selection.candidate_source) == (
        expected_data / "candidates.csv"
    )


def test_load_probabilistic_config_missing_block_raises(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "no_prob.json"
    cfg_path.write_text(json.dumps({"criteria": {}}))
    with pytest.raises(ValueError, match="probabilistic"):
        load_probabilistic_config(cfg_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_dimensions_value_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["dimensions"] = "4d"
    with pytest.raises(ValueError, match="dimensions"):
        ProbabilisticConfig.from_dict(raw)


def test_unknown_calibration_method_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["calibration"] = {"method": "magic"}
    with pytest.raises(ValueError, match="calibration.method"):
        ProbabilisticConfig.from_dict(raw)


def test_unknown_inference_backend_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {"backend": "psychic"}
    with pytest.raises(ValueError, match="inference.backend"):
        ProbabilisticConfig.from_dict(raw)


def test_inference_default_is_gblk() -> None:
    cfg = InferenceConfig()
    assert cfg.backend == "gblk"


def test_inference_from_dict_defaults_to_gblk() -> None:
    cfg = InferenceConfig.from_dict({})
    assert cfg.backend == "gblk"

    cfg_none = InferenceConfig.from_dict(None)
    assert cfg_none.backend == "gblk"


def test_unknown_alpha_mode_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["alpha"] = {"heat": {"mode": "vibes"}}
    with pytest.raises(ValueError, match="alpha.heat.mode"):
        ProbabilisticConfig.from_dict(raw)


def test_thermal_exceedance_requires_thermal_raster() -> None:
    raw = _minimal_config_dict()
    raw["alpha"] = {"heat": {"mode": "thermal_exceedance", "threshold": 200.0}}
    with pytest.raises(ValueError, match="thermal_raster"):
        ProbabilisticConfig.from_dict(raw)


def test_layer_logit_requires_layer_name() -> None:
    raw = _minimal_config_dict()
    raw["alpha"] = {"heat": {"mode": "layer_logit"}}
    with pytest.raises(ValueError, match="layer"):
        ProbabilisticConfig.from_dict(raw)


def test_retired_pymc_backend_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {"backend": "bayesian"}
    with pytest.raises(ValueError, match="inference.backend"):
        ProbabilisticConfig.from_dict(raw)


def test_bayesian_gblk_flag_cannot_be_ignored_by_sequential_backend() -> None:
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "sequential",
        "gblk_bayesian": {"enabled": True},
    }
    with pytest.raises(ValueError, match="gblk_bayesian.enabled"):
        ProbabilisticConfig.from_dict(raw)


def test_retired_shared_field_combination_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["combination"] = {"rule": "shared_field"}
    with pytest.raises(ValueError, match="shared_field"):
        ProbabilisticConfig.from_dict(raw)


def test_geometric_mean_combination_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["combination"] = {"rule": "geometric_mean"}
    with pytest.raises(ValueError, match="geometric_mean"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("patch", "match_text"),
    [
        ({"cross_validation": {"n_folds": 1}}, "n_folds"),
        ({"cross_validation": {"grid_size": 0}}, "grid_size"),
        ({"calibration": {"n_bins": 0}}, "n_bins"),
        ({"inference": {"gblk_bayesian": {"ci_level": 1.0}}}, "ci_level"),
        (
            {
                "inference": {
                    "gblk_bayesian": {"cor_scale_median": float("nan")}
                }
            },
            "cor_scale_median",
        ),
        (
            {"evidence": {"sparse_binary_threshold": 1.01}},
            "sparse_binary_threshold",
        ),
        (
            {
                "evidence": {
                    "regularization": {"per_feature_weights": {"heat": -0.1}}
                }
            },
            "per_feature_weights",
        ),
        (
            {"cross_validation": {"block_size_km": 0.0}},
            "block_size_km",
        ),
        (
            {
                "cross_validation": {
                    "block_type": "kmeans",
                    "block_size_km": 20.0,
                }
            },
            "block_size_km",
        ),
        ({"grid": {"nx": 0}}, "grid.nx"),
        ({"grid": {"extent": [0.0, 1.0, 2.0]}}, "grid.extent"),
    ],
)
def test_from_dict_rejects_invalid_numeric_ranges(
    patch: dict, match_text: str
) -> None:
    """from_dict should fail fast on invalid numeric hyperparameter ranges."""
    from geopfa.exceptions import GEOPFAValueError

    raw = _minimal_config_dict()
    for key, value in patch.items():
        raw[key] = value
    with pytest.raises(GEOPFAValueError, match=match_text):
        ProbabilisticConfig.from_dict(raw)


def test_p_min_must_be_less_than_p_max() -> None:
    raw = _minimal_config_dict()
    raw["alpha"]["heat"] = {"mode": "scalar", "p_min": 0.8, "p_max": 0.2}
    with pytest.raises(ValueError, match="p_min"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("p_min", "p_max"),
    [(0.0, 0.8), (0.2, 1.0)],
)
def test_direct_alpha_construction_rejects_infinite_logit_bounds(
    p_min: float, p_max: float
) -> None:
    with pytest.raises(ValueError, match="strictly in \\(0, 1\\)"):
        AlphaModeConfig(
            mode="layer_logit",
            layer="favorability",
            p_min=p_min,
            p_max=p_max,
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_scenarios_default_to_empty_list() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert cfg.scenarios == ()


def test_grid_default_to_inferred_from_pfa() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert cfg.grid.nx is None
    assert cfg.grid.ny is None
    assert cfg.grid.nz is None
    assert cfg.grid.extent is None
    assert cfg.grid.crs is None


def test_evidence_defaults() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert cfg.evidence.regularization.C == pytest.approx(1.0)
    assert cfg.evidence.regularization.per_feature_weights == {}
    assert cfg.evidence.include_layers is None
    assert cfg.evidence.exclude_layers == ()
    assert "inverted_y" in cfg.evidence.coordinate_blacklist


def test_outputs_defaults_to_geotiff_and_csv() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert "geotiff" in cfg.outputs.format
    assert "csv" in cfg.outputs.format


# ---------------------------------------------------------------------------
# Dataclass nesting + immutability semantics
# ---------------------------------------------------------------------------


def test_subdataclasses_are_typed() -> None:
    cfg = ProbabilisticConfig.from_dict(_minimal_config_dict())
    assert isinstance(cfg.labels, LabelsConfig)
    assert isinstance(cfg.alpha["heat"], AlphaModeConfig)
    assert isinstance(cfg.evidence, EvidenceConfig)
    assert isinstance(cfg.spatial_field, SpatialFieldConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert isinstance(cfg.inference.gblk_bayesian, GBLKBayesianConfig)
    assert isinstance(cfg.calibration, CalibrationConfig)
    assert isinstance(cfg.cross_validation, CrossValidationConfig)
    assert isinstance(cfg.combination, CombinationConfig)
    assert isinstance(cfg.outputs, OutputsConfig)
    assert isinstance(cfg.grid, GridConfig)


def test_scenarios_are_typed() -> None:
    raw = _minimal_config_dict()
    raw["scenarios"] = [
        {
            "name": "s1",
            "include_priors": True,
            "include_spatial": True,
            "drop_layers": [],
        },
    ]
    cfg = ProbabilisticConfig.from_dict(raw)
    assert len(cfg.scenarios) == 1
    assert isinstance(cfg.scenarios[0], ScenarioConfig)
    assert cfg.scenarios[0].name == "s1"


def test_scenario_names_must_be_unique_and_nonempty() -> None:
    raw = _minimal_config_dict()
    raw["scenarios"] = [{"name": "repeat"}, {"name": "repeat"}]
    with pytest.raises(ValueError, match="unique"):
        ProbabilisticConfig.from_dict(raw)

    with pytest.raises(ValueError, match="non-empty string"):
        ScenarioConfig(name="   ")

    with pytest.raises(ValueError, match="path component"):
        ScenarioConfig(name="../escape")


def test_scenario_json_rejects_coerced_names_and_layer_strings() -> None:
    raw = _minimal_config_dict()
    raw["scenarios"] = [{"name": 3}]
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        ProbabilisticConfig.from_dict(raw)

    raw["scenarios"] = [{"name": "bad", "drop_layers": "gradient"}]
    with pytest.raises(ValueError, match="drop_layers must be an array"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("patch", "match_text"),
    [
        (
            {"alpha": {"heat": {"mode": "multi_layer", "layers": "faults"}}},
            "layers must be an array",
        ),
        (
            {"evidence": {"include_layers": "gradient"}},
            "include_layers must be an array",
        ),
        (
            {"evidence": {"exclude_layers": ["gradient", 3]}},
            "exclude_layers must contain only",
        ),
        (
            {"evidence": {"coordinate_blacklist": "xy"}},
            "coordinate_blacklist must be an array",
        ),
        (
            {"site_selection": {"outcome_feature_columns": "gradient"}},
            "outcome_feature_columns must be an array",
        ),
        (
            {"site_selection": {"selection_feature_columns": [""]}},
            "selection_feature_columns must contain only",
        ),
        (
            {"site_selection": {"outcome_selection_log_odds": 0.0}},
            "outcome_selection_log_odds must be an array",
        ),
        ({"site_selection": {"mode": True}}, "mode must be a JSON string"),
        (
            {"site_selection": {"candidate_source": True}},
            "candidate_source must be a JSON string or null",
        ),
        (
            {"site_selection": {"id_col": True}},
            "id_col must be a JSON string or null",
        ),
        (
            {"site_selection": {"selected_col": True}},
            "selected_col must be a JSON string",
        ),
        ({"outputs": {"format": "csv"}}, "format must be an array"),
        ({"scenarios": {"name": "full"}}, "scenarios must be an array"),
    ],
)
def test_json_array_fields_reject_scalar_or_malformed_values(
    patch: dict, match_text: str
) -> None:
    """JSON arrays must not be inferred from iterable scalar values."""
    raw = _minimal_config_dict()
    raw.update(patch)

    with pytest.raises(ValueError, match=match_text):
        ProbabilisticConfig.from_dict(raw)


def test_bayesian_scenario_cannot_silently_disable_required_spatial_field() -> (
    None
):
    raw = _minimal_config_dict()
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
    }
    raw["scenarios"] = [{"name": "no_spatial", "include_spatial": False}]

    with pytest.raises(ValueError, match="Bayesian.*include_spatial"):
        ProbabilisticConfig.from_dict(raw)


def test_unknown_top_level_key_is_rejected() -> None:
    raw = _minimal_config_dict()
    raw["mystery_block"] = {"foo": "bar"}
    with pytest.raises(ValueError, match="mystery_block"):
        ProbabilisticConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# ProbabilisticConfig.validate() tests
# ---------------------------------------------------------------------------


def test_validate_returns_empty_for_valid_config() -> None:
    """A correctly constructed config should pass validation."""
    from tests.fixtures.synthetic_prob import make_synthetic_pfa
    from pathlib import Path

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=Path("/tmp/test_output"),
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="wells.gpkg",
            id_col="well_id",
            label_columns={"component_a": "heat_label"},
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            )
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
    assert cfg.validate() == []


def test_validate_requires_alpha_for_every_labeled_component() -> None:
    raw = _minimal_config_dict()
    del raw["alpha"]["reservoir"]

    with pytest.raises(ValueError, match="reservoir"):
        ProbabilisticConfig.from_dict(raw)


def test_gaussian_stacking_reports_missing_alpha_as_config_error() -> None:
    raw = _minimal_config_dict()
    raw["labels"]["observation_models"] = {
        "heat": {"family": "gaussian", "response_scale": 50.0}
    }
    del raw["alpha"]["heat"]
    raw["inference"] = {
        "backend": "gblk",
        "gblk_bayesian": {"enabled": True},
        "predictive_stacking": {"enabled": True},
    }

    with pytest.raises(ValueError, match="missing: heat"):
        ProbabilisticConfig.from_dict(raw)


def test_validate_requires_unlabeled_alpha_to_be_prior_predictive() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]["label_columns"]["reservoir"]

    with pytest.raises(ValueError, match="force_prior_predictive"):
        ProbabilisticConfig.from_dict(raw)


def test_validate_allows_explicit_gblk_prior_only_component() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]["label_columns"]["reservoir"]
    raw["alpha"]["reservoir"]["force_prior_predictive"] = True

    cfg = ProbabilisticConfig.from_dict(raw)

    assert cfg.alpha["reservoir"].force_prior_predictive is True


def test_validate_rejects_sequential_prior_only_component() -> None:
    raw = _minimal_config_dict()
    del raw["labels"]["label_columns"]["reservoir"]
    raw["alpha"]["reservoir"]["force_prior_predictive"] = True
    raw["inference"] = {"backend": "sequential"}

    with pytest.raises(ValueError, match="sequential"):
        ProbabilisticConfig.from_dict(raw)


def test_validate_catches_n_folds_too_small() -> None:
    from pathlib import Path

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=Path("/tmp/test"),
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="w.gpkg", id_col="id", label_columns={"a": "lbl"}
        ),
        alpha={"a": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)},
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(n_folds=1),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    errors = cfg.validate()
    assert any("n_folds" in e for e in errors)


def test_validate_raise_raises_on_invalid_config() -> None:
    from pathlib import Path

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=Path("/tmp/test"),
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="w.gpkg", id_col="id", label_columns={"a": "lbl"}
        ),
        alpha={"a": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)},
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(n_folds=1),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    import pytest as _pytest

    with _pytest.raises(ValueError, match="n_folds"):
        cfg.validate_raise()


def test_validate_catches_retired_shared_field() -> None:
    from pathlib import Path
    import pytest as _pytest

    with _pytest.raises(ValueError, match="shared_field"):
        ProbabilisticConfig(
            enabled=True,
            output_dir=Path("/tmp/test"),
            dimensions="2d",
            grid=GridConfig(),
            labels=LabelsConfig(
                source="w.gpkg", id_col="id", label_columns={"a": "lbl"}
            ),
            alpha={
                "a": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)
            },
            evidence=EvidenceConfig(),
            spatial_field=SpatialFieldConfig(enabled=False),
            inference=InferenceConfig(backend="sequential"),
            calibration=CalibrationConfig(method="none"),
            cross_validation=CrossValidationConfig(),
            combination=CombinationConfig(rule="shared_field"),
            scenarios=(),
            outputs=OutputsConfig(format=("csv",)),
        )


def test_validate_catches_empty_output_dir() -> None:
    from pathlib import Path

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=Path(""),
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="w.gpkg", id_col="id", label_columns={"a": "lbl"}
        ),
        alpha={"a": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)},
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    errors = cfg.validate()
    assert any("output_dir" in e for e in errors)


def test_calibration_config_rejects_unknown_key() -> None:
    """A misspelled calibration key must not silently change the analysis."""
    with pytest.raises(ValueError, match="fit_on"):
        CalibrationConfig.from_dict({"method": "isotonic", "fit": "block_cv"})


def test_cross_validation_config_rejects_unknown_key() -> None:
    """A misspelled CV key must not silently change the validation design."""
    with pytest.raises(ValueError, match="n_folds"):
        CrossValidationConfig.from_dict({"n_splits": 3, "block_type": "grid"})


@pytest.mark.parametrize(
    ("block", "payload"),
    [
        ("grid", {"unknown_grid_key": 1}),
        (
            "labels",
            {
                "source": "wells.gpkg",
                "id_col": "well_id",
                "label_columns": {"heat": "heat_label"},
                "unknown_labels_key": 1,
            },
        ),
        ("alpha", {"heat": {"mode": "scalar", "unknown_alpha_key": 1}}),
        (
            "evidence",
            {"regularization": {"C": 1.0, "unknown_regularization_key": 1}},
        ),
        ("evidence", {"unknown_evidence_key": 1}),
        ("spatial_field", {"unknown_spatial_key": 1}),
        ("site_selection", {"unknown_selection_key": 1}),
        (
            "inference",
            {"gblk_bayesian": {"unknown_bayesian_key": 1}},
        ),
        ("combination", {"unknown_combination_key": 1}),
        ("scenarios", [{"name": "full", "unknown_scenario_key": 1}]),
        ("outputs", {"unknown_outputs_key": 1}),
    ],
)
def test_unknown_nested_config_keys_fail_closed(
    block: str, payload: object
) -> None:
    raw = _minimal_config_dict()
    raw[block] = payload
    with pytest.raises(ValueError, match="unknown"):
        ProbabilisticConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# Range validation — P2-S01
# ---------------------------------------------------------------------------


def _make_valid_cfg(**overrides):
    """Return a valid ProbabilisticConfig with optional field overrides."""
    from pathlib import Path

    defaults = {
        "enabled": True,
        "output_dir": Path("/tmp/test_output"),
        "dimensions": "2d",
        "grid": GridConfig(),
        "labels": LabelsConfig(
            source="w.gpkg", id_col="id", label_columns={"a": "lbl"}
        ),
        "alpha": {
            "a": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5)
        },
        "evidence": EvidenceConfig(),
        "spatial_field": SpatialFieldConfig(enabled=False),
        "inference": InferenceConfig(backend="sequential"),
        "calibration": CalibrationConfig(method="none"),
        "cross_validation": CrossValidationConfig(),
        "combination": CombinationConfig(rule="product"),
        "scenarios": (),
        "outputs": OutputsConfig(format=("csv",)),
    }
    defaults.update(overrides)
    return ProbabilisticConfig(**defaults)


def test_validate_catches_regularization_c_zero() -> None:
    from geopfa.exceptions import GEOPFAValueError

    cfg = _make_valid_cfg(
        evidence=EvidenceConfig(regularization=RegularizationConfig(C=0.0))
    )
    errors = cfg.validate()
    assert any("C" in e for e in errors)
    with pytest.raises(GEOPFAValueError, match="C"):
        cfg.validate_raise()


def test_validate_catches_regularization_c_negative() -> None:
    cfg = _make_valid_cfg(
        evidence=EvidenceConfig(regularization=RegularizationConfig(C=-1.0))
    )
    errors = cfg.validate()
    assert any("C" in e for e in errors)


def test_validate_catches_gblk_bayesian_n_draws_too_small() -> None:
    cfg = _make_valid_cfg(
        inference=InferenceConfig(
            backend="gblk",
            gblk_bayesian=GBLKBayesianConfig(enabled=True, n_draws=0),
        )
    )
    errors = cfg.validate()
    assert any("n_draws" in e for e in errors)


@pytest.mark.parametrize("ci_level", [0.0, 1.0])
def test_validate_catches_gblk_bayesian_invalid_ci_level(
    ci_level: float,
) -> None:
    cfg = _make_valid_cfg(
        inference=InferenceConfig(
            backend="gblk",
            gblk_bayesian=GBLKBayesianConfig(enabled=True, ci_level=ci_level),
        )
    )
    errors = cfg.validate()
    assert any("ci_level" in e for e in errors)


def test_validate_catches_nonpositive_lattice_centers() -> None:
    cfg = _make_valid_cfg(
        spatial_field=SpatialFieldConfig(
            enabled=True,
            lattice_centers_per_dimension=0,
        )
    )
    errors = cfg.validate()
    assert any("lattice_centers_per_dimension" in error for error in errors)


def test_validate_catches_alpha_p_min_too_large() -> None:
    cfg = _make_valid_cfg(
        alpha={
            "a": AlphaModeConfig(
                mode="layer_logit",
                layer="favorability",
                scalar_fallback_pr0=0.5,
                p_min=0.6,
                p_max=0.9,
            )
        }
    )
    errors = cfg.validate()
    assert any("p_min" in e for e in errors)


def test_validate_catches_alpha_p_max_too_small() -> None:
    cfg = _make_valid_cfg(
        alpha={
            "a": AlphaModeConfig(
                mode="layer_logit",
                layer="favorability",
                scalar_fallback_pr0=0.5,
                p_min=0.2,
                p_max=0.4,
            )
        }
    )
    errors = cfg.validate()
    assert any("p_max" in e for e in errors)


def test_validate_catches_alpha_p_min_not_less_than_p_max() -> None:
    with pytest.raises(ValueError, match="p_min < p_max"):
        AlphaModeConfig(
            mode="layer_logit",
            layer="favorability",
            scalar_fallback_pr0=0.5,
            p_min=0.3,
            p_max=0.3,
        )


def test_validate_raise_uses_geopfa_value_error() -> None:
    """validate_raise must raise GEOPFAValueError (a ValueError subclass)."""
    from geopfa.exceptions import GEOPFAValueError

    cfg = _make_valid_cfg(
        evidence=EvidenceConfig(regularization=RegularizationConfig(C=-1.0))
    )
    with pytest.raises(GEOPFAValueError):
        cfg.validate_raise()
    with pytest.raises(ValueError):
        cfg.validate_raise()


def test_gblk_inference_backend_is_accepted() -> None:
    """The latticekrigx GBLK joint backend is a recognized inference backend."""
    from geopfa.prob.config import ALLOWED_INFERENCE_BACKENDS, InferenceConfig

    assert "gblk" in ALLOWED_INFERENCE_BACKENDS
    cfg = InferenceConfig.from_dict({"backend": "gblk"})
    assert cfg.backend == "gblk"


def _assign_nested(
    raw: dict, path: tuple[str | int, ...], value: object
) -> None:
    """Assign a test value inside the JSON-shaped configuration."""
    cursor: object = raw
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(
    "path",
    [
        ("enabled",),
        ("alpha", "heat", "force_prior_predictive"),
        ("spatial_field", "enabled"),
        ("inference", "gblk_bayesian", "enabled"),
        ("inference", "gblk_bayesian", "separate_ranges"),
        ("scenarios", 0, "include_priors"),
        ("scenarios", 0, "include_spatial"),
        ("outputs", "probability_rasters"),
        ("outputs", "uncertainty_rasters"),
        ("outputs", "calibration_artifacts"),
        ("outputs", "decision_artifacts"),
        ("outputs", "scenarios"),
    ],
)
def test_json_boolean_fields_reject_truthy_strings(
    path: tuple[str | int, ...],
) -> None:
    """The string ``\"false\"`` must never silently activate an analysis."""
    raw = _minimal_config_dict()
    raw.update(
        {
            "spatial_field": {},
            "inference": {"gblk_bayesian": {}},
            "calibration": {},
            "combination": {},
            "scenarios": [{"name": "audit"}],
            "outputs": {},
        }
    )
    _assign_nested(raw, path, "false")

    with pytest.raises(ValueError, match="JSON boolean"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("labels", "min_wells_for_fit"), 4.5),
        (("spatial_field", "n_levels"), "2"),
        (("spatial_field", "lattice_centers_per_dimension"), 2.5),
        (("inference", "gblk_bayesian", "n_draws"), "100"),
        (("inference", "gblk_bayesian", "seed"), 1.25),
        (("calibration", "n_bins"), False),
        (("cross_validation", "n_folds"), 3.5),
        (("cross_validation", "grid_size"), "4"),
    ],
)
def test_json_integer_fields_reject_coercible_nonintegers(
    path: tuple[str | int, ...], value: object
) -> None:
    raw = _minimal_config_dict()
    raw.update(
        {
            "spatial_field": {},
            "inference": {"gblk_bayesian": {}},
            "calibration": {},
            "cross_validation": {},
        }
    )
    _assign_nested(raw, path, value)

    with pytest.raises(ValueError, match="integer"):
        ProbabilisticConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evidence", "regularization", "C"), float("nan")),
        (("evidence", "sparse_binary_threshold"), float("inf")),
        (("inference", "gblk_bayesian", "ci_level"), float("nan")),
        (("cross_validation", "block_size_km"), "20"),
        (("cross_validation", "buffer_km"), float("nan")),
        (("site_selection", "outcome_penalty"), float("inf")),
    ],
)
def test_json_real_fields_reject_non_numeric_or_nonfinite_values(
    path: tuple[str | int, ...], value: object
) -> None:
    raw = _minimal_config_dict()
    raw.update(
        {
            "evidence": {"regularization": {}},
            "spatial_field": {},
            "inference": {"gblk_bayesian": {}},
            "cross_validation": {},
            "site_selection": {},
        }
    )
    _assign_nested(raw, path, value)

    with pytest.raises(ValueError, match="finite real"):
        ProbabilisticConfig.from_dict(raw)


def test_json_loader_rejects_nonfinite_constants(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text('{"probabilistic": {"enabled": NaN}}\n')

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_probabilistic_config(config_path)
