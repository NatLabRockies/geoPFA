"""Regression checks for the probabilistic-runtime dependency contract."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_probabilistic_runtime_pins_current_latticekrigx_source() -> None:
    """The scientific runtime must not resolve the retired INLA dependency pin."""
    metadata = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )
    dependency = metadata["tool"]["pixi"]["pypi-dependencies"]["latticekrigx"]
    assert dependency == {
        "git": "https://github.com/dhetting/latticekrigx.git",
        "rev": "8ba4f3869db3d59420535dc6b5c3690ba44e0ee2",
    }
