"""Contracts for likelihood-family-specific Kleiber profiles."""

from __future__ import annotations

import pytest

from geopfa.prob.config import (
    GBLKBayesianConfig,
    KleiberProfileConfig,
    ProbabilisticConfig,
)
from geopfa.prob.gblk_backend import _resolve_kleiber_profile


def _mixed_family_config() -> dict:
    return {
        "enabled": True,
        "output_dir": "outputs/mixed-family",
        "dimensions": "2d",
        "labels": {
            "source": "data/wells.gpkg",
            "id_col": "well_id",
            "label_columns": {
                "heat_proxy": "heat_label",
                "reservoir": "reservoir_label",
                "temperature": "temperature_c",
            },
            "observation_models": {
                "temperature": {
                    "family": "gaussian",
                    "response_scale": 50.0,
                }
            },
        },
        "alpha": {
            "heat_proxy": {"mode": "scalar", "scalar_fallback_pr0": 0.5},
            "reservoir": {"mode": "scalar", "scalar_fallback_pr0": 0.4},
            "temperature": {
                "mode": "thermal_layer_exceedance",
                "layer": "temperature_model",
                "threshold": 350.0,
            },
        },
        "inference": {
            "backend": "gblk",
            "gblk_bayesian": {
                "enabled": True,
                "kleiber_profiles": {"bernoulli": {"r0": 0.25, "r1": 0.10}},
            },
        },
    }


def test_mixed_family_config_round_trips_one_bivariate_profile() -> None:
    raw = _mixed_family_config()

    config = ProbabilisticConfig.from_dict(raw)

    profile = config.inference.gblk_bayesian.kleiber_profiles["bernoulli"]
    assert profile == KleiberProfileConfig(r0=0.25, r1=0.10)
    assert (
        config.inference.gblk_bayesian.kleiber_profiles.get("gaussian") is None
    )
    assert config.to_dict()["inference"]["gblk_bayesian"][
        "kleiber_profiles"
    ] == {"bernoulli": {"r0": 0.25, "r1": 0.10}}


def test_bivariate_family_requires_its_own_kleiber_profile() -> None:
    raw = _mixed_family_config()
    raw["inference"]["gblk_bayesian"]["kleiber_profiles"] = {}

    with pytest.raises(
        ValueError, match="bivariate bernoulli.*kleiber_profiles.bernoulli"
    ):
        ProbabilisticConfig.from_dict(raw)


def test_univariate_family_rejects_unused_kleiber_profile() -> None:
    raw = _mixed_family_config()
    raw["inference"]["gblk_bayesian"]["kleiber_profiles"]["gaussian"] = {
        "r0": 0.1,
        "r1": 0.2,
    }

    with pytest.raises(
        ValueError, match="kleiber_profiles.gaussian.*exactly two"
    ):
        ProbabilisticConfig.from_dict(raw)


def test_bayesian_family_rejects_more_than_two_fitted_components() -> None:
    raw = _mixed_family_config()
    raw["labels"]["label_columns"]["insulation"] = "insulation_label"
    raw["alpha"]["insulation"] = {
        "mode": "scalar",
        "scalar_fallback_pr0": 0.3,
    }

    with pytest.raises(
        ValueError, match="at most two fitted bernoulli components"
    ):
        ProbabilisticConfig.from_dict(raw)


def test_two_bivariate_families_keep_distinct_profiles() -> None:
    raw = _mixed_family_config()
    raw["labels"]["label_columns"]["temperature_secondary"] = (
        "temperature_secondary_c"
    )
    raw["labels"]["observation_models"]["temperature_secondary"] = {
        "family": "gaussian",
        "response_scale": 50.0,
    }
    raw["alpha"]["temperature_secondary"] = {
        "mode": "thermal_layer_exceedance",
        "layer": "temperature_model_secondary",
        "threshold": 350.0,
    }
    raw["inference"]["gblk_bayesian"]["kleiber_profiles"]["gaussian"] = {
        "r0": -0.15,
        "r1": 0.35,
    }

    config = ProbabilisticConfig.from_dict(raw)

    bayes = config.inference.gblk_bayesian
    assert _resolve_kleiber_profile(
        bayes, response_family="bernoulli", n_components=2
    ) == (0.25, 0.10)
    assert _resolve_kleiber_profile(
        bayes, response_family="gaussian", n_components=2
    ) == (-0.15, 0.35)


def test_univariate_fit_omits_other_family_profile() -> None:
    config = ProbabilisticConfig.from_dict(_mixed_family_config())

    assert _resolve_kleiber_profile(
        config.inference.gblk_bayesian,
        response_family="gaussian",
        n_components=1,
    ) == (None, None)


def test_retired_global_kleiber_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="kleiber_r0"):
        GBLKBayesianConfig.from_dict({"kleiber_r0": 0.25, "kleiber_r1": 0.10})
