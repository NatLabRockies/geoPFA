"""Tests for :mod:`geopfa.prob.hierarchical_diag` — pooling diagnostics writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geopfa.prob.hierarchical_diag import write_hierarchical_diagnostics


def _minimal_diag(n_regions: int = 2) -> dict:
    feat = ["heat_flux", "fault_density"]
    per_region = {
        f"region_{i}": {
            "beta_mean": [0.5 + i * 0.1, -0.3 + i * 0.05],
            "feature_names": feat,
        }
        for i in range(n_regions)
    }
    return {
        "n_regions": n_regions,
        "r_hat_max": 1.01,
        "ess_min": 312.0,
        "global_mu_mean": [0.6, -0.25],
        "global_sigma_mean": [0.15, 0.10],
        "pooling_factor": [0.35, 0.52],
        "per_region": per_region,
    }


def test_write_creates_json_and_md(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag()}
    json_path, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    assert json_path.exists()
    assert md_path.exists()


def test_json_is_valid_and_round_trips(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag()}
    json_path, _ = write_hierarchical_diagnostics(diag, tmp_path)
    loaded = json.loads(json_path.read_text())
    assert "heat" in loaded
    assert loaded["heat"]["n_regions"] == 2


def test_md_contains_component_heading(tmp_path: Path) -> None:
    diag = {"reservoir": _minimal_diag(n_regions=3)}
    _, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    content = md_path.read_text()
    assert "## reservoir" in content


def test_md_contains_rhat_and_ess(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag()}
    _, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    content = md_path.read_text()
    assert "r̂" in content or "r_hat" in content.lower()
    assert "ESS" in content or "ess" in content.lower()


def test_md_contains_global_hyperprior_table(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag()}
    _, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    content = md_path.read_text()
    assert "heat_flux" in content
    assert "fault_density" in content


def test_md_contains_pooling_factor_table(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag()}
    _, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    content = md_path.read_text()
    assert "pooling" in content.lower()


def test_md_contains_per_region_beta_table(tmp_path: Path) -> None:
    diag = {"heat": _minimal_diag(n_regions=2)}
    _, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    content = md_path.read_text()
    assert "region_0" in content
    assert "region_1" in content


def test_output_dir_created_if_absent(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    write_hierarchical_diagnostics({"heat": _minimal_diag()}, nested)
    assert nested.exists()


def test_multiple_components_all_written(tmp_path: Path) -> None:
    diag = {
        "heat": _minimal_diag(n_regions=2),
        "reservoir": _minimal_diag(n_regions=3),
        "barrier": _minimal_diag(n_regions=2),
    }
    json_path, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    loaded = json.loads(json_path.read_text())
    assert set(loaded.keys()) == {"heat", "reservoir", "barrier"}
    content = md_path.read_text()
    for comp in ("heat", "reservoir", "barrier"):
        assert f"## {comp}" in content


def test_empty_per_region_does_not_crash(tmp_path: Path) -> None:
    diag = {
        "heat": {
            "n_regions": 0,
            "r_hat_max": float("nan"),
            "ess_min": float("nan"),
            "per_region": {},
        }
    }
    json_path, md_path = write_hierarchical_diagnostics(diag, tmp_path)
    assert json_path.exists()
    assert md_path.exists()


def test_returns_path_tuple(tmp_path: Path) -> None:
    result = write_hierarchical_diagnostics({"heat": _minimal_diag()}, tmp_path)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(p, Path) for p in result)
