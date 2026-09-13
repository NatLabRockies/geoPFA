"""Release contracts for the three config-driven study notebooks."""

from __future__ import annotations

import ast
import importlib
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


def test_gblk_calibration_runner_is_public() -> None:
    prob = importlib.import_module("geopfa.prob")
    assert callable(prob.run_gblk_calibration_cv)
    assert "run_gblk_calibration_cv" in prob.__all__


def test_documented_bayesian_config_is_public() -> None:
    prob = importlib.import_module("geopfa.prob")
    assert prob.GBLKBayesianConfig.__module__ == "geopfa.prob.config"
    assert "GBLKBayesianConfig" in prob.__all__


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )


def _code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _config_dict(notebook: dict) -> dict:
    """Return the one literal probabilistic config declared by a notebook."""
    tree = ast.parse(_code_source(notebook))
    configs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "config_dict"
            for target in node.targets
        ):
            configs.append(ast.literal_eval(node.value))
    assert len(configs) == 1
    return configs[0]


def _run_input_artifact_names(notebook: dict) -> set[str]:
    """Return manual artifact names supplied to the model-run call."""
    tree = ast.parse(_code_source(notebook))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_probabilistic"
    ]
    assert len(calls) == 1
    artifacts = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "input_artifacts"
    )
    assert isinstance(artifacts, ast.Dict)
    return {
        key.value
        for key in artifacts.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }


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


def test_study_notebook_configs_validate() -> None:
    config_type = importlib.import_module("geopfa.prob").ProbabilisticConfig
    for path in NOTEBOOKS.values():
        config_type.from_dict(
            _config_dict(_load_notebook(path))
        ).validate_raise()


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
        assert "geopfa.prob.gblk_runner" not in source
        assert "/Users/" not in source
        assert "GEOPFA_DEMO_OUTPUT_ROOT" in source
        assert 'config_dict["output_dir"] = str(output_dir)' in source


def test_study_notebooks_keep_runtime_checks_active_under_optimization() -> (
    None
):
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
        name: _source(_load_notebook(path)) for name, path in NOTEBOOKS.items()
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
    assert (
        "from geopfa.prob import ProbabilisticConfig, "
        "run_gblk_calibration_cv, run_probabilistic"
    ) in conventional
    assert "metric_distributions" in conventional
    assert "Replicate-level" in conventional

    superhot = sources["nevada_superhot_350c_7km"]
    assert "'threshold': 350.0" in superhot
    assert "7KM" in superhot
    assert "'force_prior_predictive': True" in superhot
    assert "No target-matched labels exist" in superhot


def test_prior_only_notebooks_do_not_require_unused_label_sources() -> None:
    for name in ("newberry_superhot_400c", "nevada_superhot_350c_7km"):
        notebook = _load_notebook(NOTEBOOKS[name])
        source = _source(notebook)
        labels = _config_dict(notebook)["labels"]

        assert labels == {
            "observation_models": {"heat": {"family": "gaussian"}}
        }
        assert "labels_path" not in source
        assert "_LABELS" not in source


def test_notebooks_manually_bind_only_the_unconfigured_pfa_pickle() -> None:
    for path in NOTEBOOKS.values():
        notebook = _load_notebook(path)
        assert _run_input_artifact_names(notebook) == {"pfa_pickle"}


def test_nevada_maps_use_real_coordinates_on_checked_raster_support() -> None:
    for name in (
        "nevada_conventional_150c_3km",
        "nevada_superhot_350c_7km",
    ):
        source = _source(_load_notebook(NOTEBOOKS[name]))
        assert "def surface_on_raster_support" in source
        assert ".to_crs(raster_crs)" in source
        assert "raster CRS must be projected" in source
        assert "falls outside the thermal raster support" in source
        assert "probability_grid" not in source
        assert "proxy_grid" not in source


def test_user_docs_match_config_and_provenance_entry_points() -> None:
    method = (REPO_ROOT / "docs" / "probabilistic_method.md").read_text()
    readme = (REPO_ROOT / "README.md").read_text()

    assert '"validation_depths_m": {"heat": 3000.0}' not in method
    assert "run_probabilistic_pfa(pfa)" not in method
    assert 'run_probabilistic_pfa("outputs/pfa.pkl")' in method
    assert (
        "writes only the formats selected by `outputs.format`"
        in method.casefold()
    )
    assert "When `outputs.scenarios=true`" in method
    assert "Play-type regularization" not in readme
    assert "regularization.play_type" not in method
    assert "geometric mean" not in method.casefold()
    assert "resume verified completed blocks" not in readme
    assert "A completed directory can be reused" not in readme
    assert "compatible completed run" not in method
    for document in (readme, method):
        normalized = " ".join(document.split())
        assert "Completed output namespaces are immutable" in normalized
        assert "never accepted for another execution" in normalized
        assert "Only a verified incomplete posterior workspace" in normalized
    assert "When `vtk` is selected for a 3-D run" in method
    assert "2D/3D pipeline from labeled wells to GeoTIFF outputs" not in readme


def test_migration_guide_python_examples_parse() -> None:
    guide = (REPO_ROOT / "docs" / "migration_guide.md").read_text()
    blocks = guide.split("```python\n")[1:]
    assert blocks
    for index, block in enumerate(blocks):
        source, separator, _tail = block.partition("```")
        assert separator
        ast.parse(source, filename=f"migration_guide.py#{index}")
