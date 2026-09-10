"""Release contracts for the three config-driven study notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "newberry_superhot_400c": (
        REPO_ROOT
        / "examples"
        / "Newberry"
        / "3D"
        / "notebooks"
        / "5-newberry_superhot_400c.ipynb"
    ),
    "nevada_conventional_150c_3km": (
        REPO_ROOT
        / "examples"
        / "Nevada"
        / "2D"
        / "notebooks"
        / "5-nevada_conventional_150c_3km.ipynb"
    ),
    "nevada_superhot_350c_7km": (
        REPO_ROOT
        / "examples"
        / "Nevada"
        / "2D"
        / "notebooks"
        / "6-nevada_superhot_350c_7km.ipynb"
    ),
}


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def test_study_notebooks_are_output_free() -> None:
    for path in NOTEBOOKS.values():
        notebook = _load_notebook(path)
        assert notebook["nbformat"] == 4
        assert all(cell.get("id") for cell in notebook["cells"])
        assert len({cell["id"] for cell in notebook["cells"]}) == len(
            notebook["cells"]
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_study_notebooks_use_only_public_geopfa_interfaces() -> None:
    for path in NOTEBOOKS.values():
        source = _source(_load_notebook(path))
        assert "ProbabilisticConfig.from_dict" in source
        assert "run_probabilistic" in source
        assert "VoterVeto.do_voter_veto" in source
        assert "config_dict" in source
        assert "from _helpers" not in source
        assert "sys.path" not in source
        assert "load_probabilistic_config" not in source
        assert "/Users/" not in source
        assert "GEOPFA_DEMO_OUTPUT_ROOT" in source
        assert 'config_dict["output_dir"] = str(output_dir)' in source


def test_study_notebooks_keep_runtime_checks_active_under_optimization() -> None:
    for path in NOTEBOOKS.values():
        notebook = _load_notebook(path)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(
                "".join(cell.get("source", [])), filename=str(path)
            )
            assert not any(
                isinstance(node, ast.Assert) for node in ast.walk(tree)
            )


def test_study_targets_and_validation_scope_are_explicit() -> None:
    sources = {
        name: _source(_load_notebook(path))
        for name, path in NOTEBOOKS.items()
    }
    newberry = sources["newberry_superhot_400c"]
    assert "'threshold': 400.0" in newberry
    assert "'dimensions': '3d'" in newberry
    assert "'force_prior_predictive': True" in newberry
    assert "No population calibration claim" in newberry

    conventional = sources["nevada_conventional_150c_3km"]
    assert "150.0" in conventional
    assert "3KM" in conventional
    assert "run_gblk_calibration_cv" in conventional
    assert "metric_distributions" in conventional
    assert "Replicate-level" in conventional

    superhot = sources["nevada_superhot_350c_7km"]
    assert "'threshold': 350.0" in superhot
    assert "7KM" in superhot
    assert "'force_prior_predictive': True" in superhot
    assert "No target-matched labels exist" in superhot
