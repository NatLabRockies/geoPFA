"""Real-data smoke test for the probabilistic workflow.

Gated by the ``realdata`` pytest marker — NOT run in default CI.
Requires the Newberry dataset to be present in the repo at the path
expected by the existing Newberry processing pipeline.

Run locally with::

    pixi run -e test python -m pytest tests/integration/test_prob_workflow.py -m realdata -v

This test does NOT assert exact numeric values — it is a smoke test that
confirms the end-to-end workflow runs to completion on real geologic data
without crashes, and that the output files have the expected structure.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PFA_PICKLE = Path("examples/Newberry/2D/notebooks/fpa.pkl")
_WELLS_FILE = Path("data/raw/wells.gpkg")  # update to your local path


@pytest.mark.realdata
def test_prob_workflow_on_newberry_pfa(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """Smoke test: run run_probabilistic on the cached Newberry PFA dict."""
    if not _PFA_PICKLE.exists():
        pytest.skip(f"Newberry PFA pickle not found at {_PFA_PICKLE}")

    from geopfa.prob import (
        AlphaModeConfig,
        CalibrationConfig,
        CombinationConfig,
        CrossValidationConfig,
        EvidenceConfig,
        GridConfig,
        InferenceConfig,
        LabelsConfig,
        OutputsConfig,
        ProbabilisticConfig,
        SpatialFieldConfig,
        run_probabilistic,
    )

    with _PFA_PICKLE.open("rb") as fh:
        pfa = pickle.load(fh)  # noqa: S301

    # Determine the first two component names from the PFA dict.
    components = list(pfa["criteria"]["geologic"]["components"].keys())
    if len(components) < 1:
        pytest.skip("Newberry PFA has no components")

    out_dir = tmp_path / "prob_outputs"

    # Locate a usable labeled wells file. Requires a GeoPackage (self-describing CRS).
    _nb_wells_gpkg = _PFA_PICKLE.parent / "wells.gpkg"
    if _nb_wells_gpkg.exists():
        wells_source = str(_nb_wells_gpkg)
    elif _WELLS_FILE.exists():
        wells_source = str(_WELLS_FILE)
    else:
        pytest.skip(
            "No labeled wells GeoPackage found for Newberry integration test"
        )

    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=out_dir,
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=wells_source,
            id_col="well_id",
            label_columns={components[0]: "heat_label"},
        ),
        alpha={
            components[0]: AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            )
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=True, backend="rbf"),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    result = run_probabilistic(pfa, cfg)

    # Smoke assertions — structure only, no exact values.
    assert not result.skipped
    assert len(result.components) >= 1
    for surface in result.components.values():
        probs = surface.probability["probability"].to_numpy()
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    assert (out_dir / "manifest.json").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert len(manifest["files"]) > 0
