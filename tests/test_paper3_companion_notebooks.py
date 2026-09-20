"""Behavioral checks for the three probabilistic study notebooks."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from geopfa.prob import ProbabilisticConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    REPO_ROOT
    / "examples/Newberry/3D/notebooks/5-newberry_superhot_400c.ipynb",
    REPO_ROOT
    / "examples/Nevada/2D/notebooks/5-nevada_conventional_150c_3km.ipynb",
    REPO_ROOT
    / "examples/Nevada/2D/notebooks/6-nevada_superhot_350c_7km.ipynb",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _config(notebook: dict) -> dict:
    assignments = [
        node
        for node in ast.walk(ast.parse(_code(notebook)))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "config_dict"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    return ast.literal_eval(assignments[0].value)


def test_study_notebooks_are_valid_and_output_free() -> None:
    for path in NOTEBOOKS:
        notebook = _load(path)
        assert notebook["nbformat"] == 4
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids) and len(ids) == len(set(ids))
        ast.parse(_code(notebook), filename=str(path))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_study_notebook_configs_validate() -> None:
    for path in NOTEBOOKS:
        ProbabilisticConfig.from_dict(_config(_load(path))).validate_raise()


def test_study_notebooks_use_processed_inputs_and_final_surfaces() -> None:
    for path in NOTEBOOKS:
        source = _code(_load(path))
        normalized = source.casefold()

        assert "load_processed_pfa" in source
        assert "run_probabilistic" in source
        assert 'pfa_vv["pr_norm"]' in source
        assert "model_result.combined" in source
        assert "/users/" not in normalized
        assert "pickle" not in normalized
        assert "fpa.pkl" not in normalized
        assert "pfa.pkl" not in normalized


def test_newberry_demo_does_not_persist_full_grid_draw_blocks() -> None:
    config = _config(_load(NOTEBOOKS[0]))
    assert config["outputs"]["posterior_draw_blocks"] is False


def test_nevada_final_maps_use_raster_coordinates() -> None:
    for path in NOTEBOOKS[1:]:
        source = _code(_load(path))
        assert "surface_on_raster_support" in source
        assert ".to_crs(raster_crs)" in source
        assert "probability_grid" not in source
        assert "proxy_grid" not in source
