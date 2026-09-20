"""Minimal runtime constraints for reproducible releases."""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest() -> dict:
    return tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )


def test_latticekrigx_source_is_checkout_independent_and_immutable() -> None:
    source = _manifest()["tool"]["pixi"]["pypi-dependencies"]["latticekrigx"]

    assert "path" not in source
    assert source["git"] == "https://github.com/NatLabRockies/latticekrigx.git"
    assert re.fullmatch(r"[0-9a-f]{40}", source["rev"])


def test_python_support_matches_latticekrigx_runtime() -> None:
    project = _manifest()["project"]

    assert project["requires-python"] == ">=3.11,<3.13"
    assert not any(
        classifier.endswith(("3.13", "3.14"))
        for classifier in project["classifiers"]
    )
