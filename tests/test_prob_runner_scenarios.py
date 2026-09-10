"""Tests for multi-scenario ablation runs via run_probabilistic."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

from geopfa.prob.config import load_probabilistic_config
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _fixture_and_config(tmp_path: Path, scenarios: list[dict] | None = None) -> tuple:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    out_dir = tmp_path / "out"
    cfg_dict = {
        "criteria": {},
        "probabilistic": {
            "enabled": True,
            "output_dir": str(out_dir),
            "dimensions": "2d",
            "labels": {
                "source": str(wells_path),
                "id_col": "well_id",
                "layer": "wells",
                "label_columns": {
                    "component_a": "heat_label",
                    "component_b": "reservoir_label",
                },
            },
            "alpha": {
                "component_a": {"mode": "layer_logit", "layer": "prior_layer_a", "scalar_fallback_pr0": 0.55},
                "component_b": {"mode": "layer_logit", "layer": "prior_layer_b", "scalar_fallback_pr0": 0.50},
            },
            "spatial_field": {"enabled": True, "backend": "rbf"},
            "inference": {"backend": "sequential"},
            "calibration": {"method": "none"},
            "scenarios": scenarios or [],
            "outputs": {
                "probability_rasters": False,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": True,
                "format": ["csv"],
            },
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg_dict))
    return fixture, load_probabilistic_config(cfg_path)


def test_no_scenarios_gives_empty_scenarios_dict(tmp_path):
    fixture, cfg = _fixture_and_config(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert result.scenarios == {}


def test_two_scenarios_produces_both_results(tmp_path):
    scenarios = [
        {"name": "full", "include_priors": True, "include_spatial": True, "drop_layers": []},
        {"name": "no_spatial", "include_priors": True, "include_spatial": False, "drop_layers": []},
    ]
    fixture, cfg = _fixture_and_config(tmp_path, scenarios)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert set(result.scenarios.keys()) == {"full", "no_spatial"}
    for scenario_result in result.scenarios.values():
        assert "component_a" in scenario_result


def test_no_spatial_scenario_differs_from_full(tmp_path):
    scenarios = [
        {"name": "full", "include_priors": True, "include_spatial": True, "drop_layers": []},
        {"name": "no_spatial", "include_priors": True, "include_spatial": False, "drop_layers": []},
    ]
    fixture, cfg = _fixture_and_config(tmp_path, scenarios)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    p_full = result.scenarios["full"]["component_a"].probability["probability"].to_numpy()
    p_nospatial = result.scenarios["no_spatial"]["component_a"].probability["probability"].to_numpy()
    # Spatial field changes the surface — must differ somewhere.
    assert not (p_full == p_nospatial).all()


def test_drop_layers_scenario_excludes_layer(tmp_path):
    scenarios = [
        {"name": "no_sparse", "include_priors": True, "include_spatial": False,
         "drop_layers": ["sparse_indicator"]},
    ]
    fixture, cfg = _fixture_and_config(tmp_path, scenarios)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert "no_sparse" in result.scenarios
    # Component should still fit with remaining features.
    comp = result.scenarios["no_sparse"]["component_a"]
    assert "probability" in comp.probability.columns


def test_scenario_output_toggle_writes_named_probability_surfaces(tmp_path):
    scenarios = [
        {
            "name": "no_spatial",
            "include_priors": True,
            "include_spatial": False,
            "drop_layers": [],
        },
    ]
    fixture, cfg = _fixture_and_config(tmp_path, scenarios)
    cfg = replace(
        cfg,
        outputs=replace(cfg.outputs, probability_rasters=True, scenarios=True),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)

    scenario_dir = cfg.output_dir / "scenarios" / "no_spatial"
    assert (scenario_dir / "component_a_probability.csv").is_file()
    assert (scenario_dir / "component_b_probability.csv").is_file()
    assert (scenario_dir / "combined_probability.csv").is_file()
