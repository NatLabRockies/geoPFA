"""Release-environment contracts that must hold outside the source workspace."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUP_PIXI_REVISION = "d3f436a425481402e6a95a1d1fc10331c708cd9e"
SETUP_PIXI_WORKFLOWS = (
    "codecov.yml",
    "docs.yml",
    "publish_to_pypi.yml",
    "pylint.yml",
    "unit_test.yml",
)
PROBABILISTIC_DEMO_NOTEBOOKS = (
    Path("examples/Newberry/3D/notebooks/5-newberry_superhot_400c.ipynb"),
    Path("examples/Nevada/2D/notebooks/5-nevada_conventional_150c_3km.ipynb"),
    Path("examples/Nevada/2D/notebooks/6-nevada_superhot_350c_7km.ipynb"),
)


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


def test_ci_uses_current_pinned_setup_pixi_action() -> None:
    workflows = REPO_ROOT / ".github" / "workflows"

    for workflow_name in SETUP_PIXI_WORKFLOWS:
        source = (workflows / workflow_name).read_text(encoding="utf-8")
        assert f"prefix-dev/setup-pixi@{SETUP_PIXI_REVISION}" in source, (
            workflow_name
        )


def test_documentation_gate_treats_warnings_as_errors() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    sphinx_options = manifest["tool"]["pixi"]["feature"]["doc"]["tasks"][
        "python-docs"
    ]["env"]["SPHINXOPTS"]

    assert "--fail-on-warning" in sphinx_options.split()


def test_generated_version_module_is_excluded_from_formatting() -> None:
    manifest = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    excluded = manifest["tool"]["ruff"]["format"]["exclude"]

    assert "geopfa/_version.py" in excluded


def test_examples_do_not_track_notebook_checkpoints() -> None:
    prohibited = [
        str(path)
        for path in _example_paths()
        if ".ipynb_checkpoints" in path.parts
    ]

    assert prohibited == []


def test_probabilistic_demo_notebooks_are_output_free() -> None:
    violations: list[str] = []
    for relative_path in PROBABILISTIC_DEMO_NOTEBOOKS:
        notebook = json.loads(
            (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        )
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("outputs") or cell.get("execution_count") is not None:
                violations.append(f"{relative_path}:cell-{cell_index}")

    assert violations == []


def test_probabilistic_demo_notebooks_have_portable_sources() -> None:
    violations: list[str] = []
    for relative_path in PROBABILISTIC_DEMO_NOTEBOOKS:
        notebook = json.loads(
            (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        )
        source = "".join(
            line
            for cell in notebook.get("cells", [])
            for line in cell.get("source", [])
        ).lower()
        if "/users/" in source or "c:\\users\\" in source:
            violations.append(str(relative_path))

    assert violations == []


def test_class_docs_do_not_index_members_twice() -> None:
    template = (
        REPO_ROOT / "docs/source/_templates/autosummary/class.rst"
    ).read_text(encoding="utf-8")

    assert ":members:" in template
    assert ":exclude-members: {{ attributes | join(', ') }}" in template
    assert ".. autosummary::" not in template
