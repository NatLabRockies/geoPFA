"""Tests for the geopfa-prob CLI."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from geopfa.prob.cli import main as cli_main
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _make_config_with_pfa_pickle(tmp_path: Path) -> Path:
    """Write a real PFA pickle + config JSON; the CLI loads the pickle path."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    pfa_pickle = tmp_path / "pfa.pkl"
    with pfa_pickle.open("wb") as fh:
        pickle.dump(fixture.pfa, fh)

    out_dir = tmp_path / "out"
    config = {
        "criteria": {},
        "pfa_pickle": str(pfa_pickle),
        "probabilistic": {
            "enabled": True,
            "output_dir": str(out_dir),
            "dimensions": "2d",
            "grid": {},
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
                "component_a": {
                    "mode": "layer_logit",
                    "layer": "prior_layer_a",
                    "scalar_fallback_pr0": 0.55,
                },
                "component_b": {
                    "mode": "layer_logit",
                    "layer": "prior_layer_b",
                    "scalar_fallback_pr0": 0.50,
                },
            },
            "spatial_field": {"enabled": True, "backend": "rbf"},
            "calibration": {"method": "none"},
            "outputs": {
                "probability_rasters": False,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": False,
                "format": ["csv"],
            },
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))
    return cfg_path


def test_cli_main_with_pfa_pickle(tmp_path: Path) -> None:
    cfg_path = _make_config_with_pfa_pickle(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(["run", "--config", str(cfg_path)])
    assert rc == 0
    out_dir = json.loads(cfg_path.read_text())["probabilistic"]["output_dir"]
    assert (Path(out_dir) / "component_a_probability.csv").exists()
    assert (Path(out_dir) / "component_b_probability.csv").exists()
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    assert {record["name"] for record in manifest["inputs"]} >= {
        "config",
        "pfa_pickle",
        "labels.source",
    }


def test_cli_resolves_relative_paths_from_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "study" / "config"
    config_dir.mkdir(parents=True)
    cfg_path = _make_config_with_pfa_pickle(config_dir)
    config = json.loads(cfg_path.read_text())
    config["pfa_pickle"] = "pfa.pkl"
    config["probabilistic"]["labels"]["source"] = "wells.gpkg"
    config["probabilistic"]["output_dir"] = "outputs"
    cfg_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(["run", "--config", str(cfg_path)])

    assert rc == 0
    assert (config_dir / "outputs/component_a_probability.csv").is_file()


def test_cli_main_returns_nonzero_on_missing_config(tmp_path: Path) -> None:
    rc = cli_main(["run", "--config", str(tmp_path / "missing.json")])
    assert rc != 0


def test_cli_main_returns_nonzero_when_no_pfa_pickle_key(
    tmp_path: Path,
) -> None:
    """Config missing 'pfa_pickle' key should fail cleanly."""
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    out_dir = tmp_path / "out"
    config = {
        "probabilistic": {
            "enabled": True,
            "output_dir": str(out_dir),
            "dimensions": "2d",
            "labels": {
                "source": str(wells_path),
                "id_col": "well_id",
                "layer": "wells",
                "label_columns": {"component_a": "heat_label"},
            },
            "alpha": {
                "component_a": {"mode": "scalar", "scalar_fallback_pr0": 0.5}
            },
        },
        # no pfa_pickle key
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))
    rc = cli_main(["run", "--config", str(cfg_path)])
    assert rc != 0


def test_cli_invoke_via_module_entrypoint(tmp_path: Path) -> None:
    """Smoke-check that ``python -m geopfa.prob`` exposes the CLI."""
    cfg_path = _make_config_with_pfa_pickle(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "geopfa.prob",
            "run",
            "--config",
            str(cfg_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
