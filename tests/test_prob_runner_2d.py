"""Tests for runner 2D integration (config-driven, end-to-end)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from geopfa.prob.config import load_probabilistic_config
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _write_fixture(tmp_path: Path, *, seed: int = 0):
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=seed)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    return fixture, wells_path


def _write_config(wells_path: Path, output_dir: Path, extra: dict | None = None) -> Path:
    cfg = {
        "criteria": {},
        "probabilistic": {
            "enabled": True,
            "output_dir": str(output_dir),
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
            "outputs": {
                "probability_rasters": False,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": False,
                "format": ["csv"],
            },
            **(extra or {}),
        },
    }
    p = output_dir.parent / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_2d_runner_produces_component_surfaces(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(wells_path, out_dir)
    cfg = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert set(result.components.keys()) == {"component_a", "component_b"}
    for surface in result.components.values():
        prob = surface.probability["probability"].to_numpy()
        assert (prob >= 0).all() and (prob <= 1).all()


def test_2d_runner_writes_csv_outputs(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=1)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(wells_path, out_dir)
    cfg = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    assert (out_dir / "component_a_probability.csv").exists()
    assert (out_dir / "component_b_probability.csv").exists()
    assert (out_dir / "combined_probability.csv").exists()
    df = pd.read_csv(out_dir / "combined_probability.csv")
    assert "probability" in df.columns


def test_2d_runner_writes_geotiff(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=2)
    out_dir = tmp_path / "out"
    cfg = {
        "criteria": {},
        "probabilistic": {
            "enabled": True,
            "output_dir": str(out_dir),
            "dimensions": "2d",
            "labels": {
                "source": str(wells_path),
                "id_col": "well_id",
                "layer": "wells",
                "label_columns": {"component_a": "heat_label", "component_b": "reservoir_label"},
            },
            "alpha": {
                "component_a": {"mode": "scalar", "scalar_fallback_pr0": 0.5},
                "component_b": {"mode": "scalar", "scalar_fallback_pr0": 0.5},
            },
            "spatial_field": {"enabled": False},
            "calibration": {"method": "none"},
            "outputs": {
                "probability_rasters": True,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": False,
                "format": ["geotiff", "csv"],
            },
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    from geopfa.prob.config import load_probabilistic_config
    cfg_obj = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg_obj)
    assert (out_dir / "component_a_probability.tif").exists()
    assert (out_dir / "combined_probability.tif").exists()


def test_2d_runner_probability_raster_toggle_suppresses_geotiff(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=12)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(
        wells_path,
        out_dir,
        extra={
            "outputs": {
                "probability_rasters": False,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": False,
                "format": ["geotiff", "csv"],
            }
        },
    )
    cfg = load_probabilistic_config(cfg_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)

    assert (out_dir / "component_a_probability.csv").is_file()
    assert not (out_dir / "component_a_probability.tif").exists()
    assert not (out_dir / "combined_probability.tif").exists()


def test_2d_runner_alpha_provenance_json(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=3)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(wells_path, out_dir)
    cfg = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    prov = json.loads((out_dir / "alpha_provenance.json").read_text())
    assert "component_a" in prov
    assert prov["component_a"]["mode"] == "layer_logit"


def test_2d_runner_manifest_json(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=4)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(wells_path, out_dir)
    cfg = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert "files" in manifest
    assert manifest["config_hash"]
    assert len(manifest["files"]) > 0


def test_2d_runner_disabled_is_noop(tmp_path):
    fixture, wells_path = _write_fixture(tmp_path, seed=5)
    out_dir = tmp_path / "out"
    cfg_path = _write_config(wells_path, out_dir, extra={"enabled": False})
    cfg = load_probabilistic_config(cfg_path)
    result = run_probabilistic(fixture.pfa, cfg)
    assert result.skipped is True
    assert result.components == {}
