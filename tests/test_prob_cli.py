"""Tests for the geopfa-prob CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from geopfa.prob.cli import main as cli_main
from geopfa.io.data_writers import GeospatialDataWriters
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _make_processed_study(tmp_path: Path) -> tuple[Path, Path]:
    """Write a processed PFA layer tree and matching probabilistic config."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    data_dir = tmp_path / "processed"
    data_dir.mkdir()
    GeospatialDataWriters.save_processed_layers(fixture.pfa, data_dir)

    out_dir = tmp_path / "out"
    config = {
        "criteria": json.loads(
            json.dumps(
                fixture.pfa["criteria"],
                default=lambda _value: None,
            )
        ),
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
            "spatial_field": {
                "enabled": True,
                "backend": "latticekrigx",
            },
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
    for component in config["criteria"]["geologic"]["components"].values():
        component.pop("pr_norm", None)
        for layer in component["layers"].values():
            layer.pop("model", None)
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))
    return cfg_path, data_dir


def test_cli_main_with_processed_layers(tmp_path: Path) -> None:
    cfg_path, data_dir = _make_processed_study(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(
            [
                "run",
                "--config",
                str(cfg_path),
                "--processed-data-dir",
                str(data_dir),
                "--crs",
                "EPSG:32611",
            ]
        )
    assert rc == 0
    out_dir = json.loads(cfg_path.read_text())["probabilistic"]["output_dir"]
    assert (Path(out_dir) / "component_a_probability.csv").exists()
    assert (Path(out_dir) / "component_b_probability.csv").exists()
    manifest = json.loads((Path(out_dir) / "manifest.json").read_text())
    assert {record["name"] for record in manifest["inputs"]} >= {
        "processed_config",
        "labels.source",
    }
    assert any(
        record["name"].startswith("processed_data:")
        for record in manifest["inputs"]
    )


def test_cli_resolves_relative_paths_from_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "study" / "config"
    config_dir.mkdir(parents=True)
    cfg_path, data_dir = _make_processed_study(config_dir)
    config = json.loads(cfg_path.read_text())
    config["probabilistic"]["labels"]["source"] = "wells.gpkg"
    config["probabilistic"]["output_dir"] = "outputs"
    cfg_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = cli_main(
            [
                "run",
                "--config",
                str(cfg_path),
                "--processed-data-dir",
                str(data_dir),
                "--crs",
                "EPSG:32611",
            ]
        )

    assert rc == 0
    assert (config_dir / "outputs/component_a_probability.csv").is_file()


def test_cli_main_returns_nonzero_on_missing_config(tmp_path: Path) -> None:
    rc = cli_main(
        [
            "run",
            "--config",
            str(tmp_path / "missing.json"),
            "--processed-data-dir",
            str(tmp_path),
            "--crs",
            "EPSG:32611",
        ]
    )
    assert rc != 0


def test_cli_main_rejects_empty_processed_data_directory(
    tmp_path: Path,
) -> None:
    cfg_path, _data_dir = _make_processed_study(tmp_path / "study")
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = cli_main(
        [
            "run",
            "--config",
            str(cfg_path),
            "--processed-data-dir",
            str(empty),
            "--crs",
            "EPSG:32611",
        ]
    )
    assert rc != 0


def test_cli_invoke_via_module_entrypoint(tmp_path: Path) -> None:
    """Smoke-check that ``python -m geopfa.prob`` exposes the CLI."""
    cfg_path, data_dir = _make_processed_study(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "geopfa.prob",
            "run",
            "--config",
            str(cfg_path),
            "--processed-data-dir",
            str(data_dir),
            "--crs",
            "EPSG:32611",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
