"""Release-environment contracts that must hold outside the source workspace."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]


def _example_paths() -> list[Path]:
    return [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "examples").rglob("*")
    ]


def test_pixi_latticekrigx_source_is_checkout_independent_and_immutable() -> (
    None
):
    manifest = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    source = manifest["tool"]["pixi"]["pypi-dependencies"]["latticekrigx"]

    assert "path" not in source
    assert source["git"] == "https://github.com/NatLabRockies/latticekrigx.git"
    assert re.fullmatch(r"[0-9a-f]{40}", source["rev"])


def test_python_support_matches_required_latticekrigx_runtime() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = manifest["project"]

    assert project["requires-python"] == ">=3.11,<3.13"
    assert not any(
        classifier.endswith(("3.13", "3.14"))
        for classifier in project["classifiers"]
    )


def test_local_gate_checks_ruff_formatting_like_ci() -> None:
    gate = (REPO_ROOT / "test_repo.sh").read_text(encoding="utf-8")

    assert "ruff format --check geopfa tests" in gate


def test_generated_version_module_is_excluded_from_formatting() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    excluded = manifest["tool"]["ruff"]["format"]["exclude"]

    assert "geopfa/_version.py" in excluded


def test_examples_track_notebooks_but_no_generated_visual_artifacts() -> None:
    prohibited_suffixes = {
        ".gif",
        ".jpeg",
        ".jpg",
        ".pdf",
        ".png",
        ".svg",
        ".tif",
        ".tiff",
    }
    prohibited = [
        str(path)
        for path in _example_paths()
        if path.parts
        and path.parts[0] == "examples"
        and (
            ".ipynb_checkpoints" in path.parts
            or path.suffix.lower() in prohibited_suffixes
        )
    ]

    assert prohibited == []


def test_tracked_example_notebooks_are_output_free() -> None:
    violations: list[str] = []
    notebooks = [
        path
        for path in _example_paths()
        if path.parts
        and path.parts[0] == "examples"
        and path.suffix == ".ipynb"
    ]
    for relative_path in notebooks:
        notebook = json.loads(
            (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        )
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                violations.append(f"{relative_path}:cell-{cell_index}")

    assert violations == []
