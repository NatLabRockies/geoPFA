"""Contracts for explicit fixed evidence coefficients."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("latticekrigx.glk.joint")

from geopfa.prob.alpha import build_alpha_c  # noqa: E402
from geopfa.prob.config import (  # noqa: E402
    AlphaModeConfig,
    EvidenceConfig,
    GBLKBayesianConfig,
    InferenceConfig,
    RegularizationConfig,
)
from geopfa.prob.gblk_runner import _prior_predictive_evidence_draws  # noqa: E402
from geopfa.prob.pfa_grid import PFAGridAdapter  # noqa: E402
from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: E402


def test_fixed_coefficients_parse_and_roundtrip() -> None:
    """The config contract must retain a fixed, auditable exclusion."""
    regularization = RegularizationConfig(
        fixed_coefficients={"insulation:temperature_model_500m": 0.0}
    )

    assert regularization.to_dict()["fixed_coefficients"] == {
        "insulation:temperature_model_500m": 0.0
    }


def test_prior_predictive_component_does_not_resample_fixed_layer() -> None:
    """A fixed zero rule excludes its layer from every evidence draw."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=50, seed=333)
    adapter = PFAGridAdapter(fixture.pfa, criteria="geologic", dimensions="2d")
    alpha_cfg = AlphaModeConfig(
        mode="scalar",
        scalar_fallback_pr0=0.23,
        force_prior_predictive=True,
        use_evidence_prior=True,
    )
    cfg = _prior_only_config(
        alpha_cfg,
        RegularizationConfig(
            prior_means={"component_b:gradient": 0.8},
            prior_precisions={"component_b:gradient": 4.0},
            fixed_coefficients={"component_b:prior_layer_b": 0.0},
        ),
    )
    alpha = build_alpha_c(
        adapter.component_data("component_b"),
        alpha_cfg,
        grid_gdf=adapter.pr_norm("component_b"),
    )

    draws = _prior_predictive_evidence_draws(
        adapter, "component_b", alpha, cfg, seed=91
    )

    np.testing.assert_array_equal(draws.coefficient_draws[:, 0], 0.0)
    assert draws.diagnostics["coefficient_fixed_value"] == [0.0, None]


def _prior_only_config(
    alpha: AlphaModeConfig, regularization: RegularizationConfig
):
    """Construct the minimal configuration consumed by the draw helper."""
    from geopfa.prob.config import ProbabilisticConfig

    return ProbabilisticConfig.from_dict(
        {
            "enabled": True,
            "output_dir": str(Path("unused")),
            "dimensions": "2d",
            "alpha": {
                "component_a": {
                    "mode": "scalar",
                    "scalar_fallback_pr0": 0.5,
                    "force_prior_predictive": True,
                },
                "component_b": alpha.to_dict(),
            },
            "labels": {"label_columns": {"component_a": "heat_label"}},
            "evidence": {"regularization": regularization.to_dict()},
            "inference": {
                "backend": "gblk",
                "gblk_bayesian": GBLKBayesianConfig(
                    enabled=True, n_draws=16, seed=91
                ).to_dict(),
            },
        }
    )
