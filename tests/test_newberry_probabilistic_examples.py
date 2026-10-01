"""Contracts for the canonical runnable Newberry probability examples."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from geopfa.prob import ProbabilisticConfig
from geopfa.prob.evidence_prior_profiles import (
    evidence_prior_profile_from_dict,
)
from geopfa.prob.evidence_priors import derive_evidence_coefficient_priors


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "150c": REPO_ROOT
    / "examples/Newberry/2D/notebooks/3-newberry_probabilistic_150c.ipynb",
    "400c": REPO_ROOT
    / "examples/Newberry/3D/notebooks/5-newberry_superhot_400c.ipynb",
}
PROFILE = (
    REPO_ROOT / "examples/Newberry/config/newberry_literature_priors.json"
)


def _load(path: Path) -> dict:
    """Load one output-free notebook."""
    return json.loads(path.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    """Join the executable cells of one notebook."""
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def _config(notebook: dict) -> dict:
    """Read the literal base configuration declared by one notebook."""
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


def test_newberry_probability_notebooks_are_runnable_and_output_free() -> None:
    """The two canonical notebooks have stable, clean execution sources."""
    for path in NOTEBOOKS.values():
        notebook = _load(path)
        assert notebook["nbformat"] == 4
        ids = [cell.get("id") for cell in notebook["cells"]]
        assert all(ids) and len(ids) == len(set(ids))
        source = _code(notebook)
        ast.parse(source, filename=str(path))
        assert "setup_newberry_tutorial_data(data_dir)" in source
        assert "ensure_newberry_tutorial_processed_data" in source
        assert "load_processed_pfa" in source
        assert "run_probabilistic" in source
        assert "voterveto" not in source.casefold()
        assert "layer_combination" not in source
        assert "/users/" not in source.casefold()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs") == []


def test_newberry_probability_notebooks_declare_their_targets() -> None:
    """Both public examples retain explicit threshold and draw semantics."""
    config_150c = _config(_load(NOTEBOOKS["150c"]))
    config_400c = _config(_load(NOTEBOOKS["400c"]))

    ProbabilisticConfig.from_dict(config_150c).validate_raise()
    ProbabilisticConfig.from_dict(config_400c).validate_raise()

    assert config_150c["dimensions"] == "2d"
    assert config_150c["alpha"]["heat"]["threshold"] == 150.0
    assert config_400c["dimensions"] == "3d"
    assert config_400c["alpha"]["heat"]["threshold"] == 400.0
    assert config_400c["inference"]["gblk_bayesian"]["n_draws"] == 64


def test_newberry_400c_profile_preserves_the_actual_component_rules() -> None:
    """The compact example uses the reviewed literature-profile assumptions."""
    profile = evidence_prior_profile_from_dict(
        json.loads(PROFILE.read_text(encoding="utf-8"))
    )
    resolved = derive_evidence_coefficient_priors(
        profile,
        {
            "reservoir": (
                "density_joint_inv",
                "mt_resistivity_joint_inv",
                "earthquakes",
                "ring_faults",
                "lineaments",
            ),
            "insulation": (
                "density_joint_inv",
                "mt_resistivity_joint_inv",
                "earthquakes",
                "temperature_model_500m",
            ),
        },
    )

    assert resolved.prior_means["reservoir:ring_faults"] > 0.0
    assert resolved.prior_precisions["reservoir:ring_faults"] == 1 / 0.6**2
    assert resolved.fixed_coefficients == {
        "insulation:temperature_model_500m": 0.0
    }


def test_legacy_empty_probabilistic_example_tree_is_absent() -> None:
    """Newberry is the sole canonical probability-example location."""
    assert not (REPO_ROOT / "examples/probabilistic").exists()
