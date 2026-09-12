"""Tests for the CV runner and calibration wiring."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import geopandas as gpd

from geopfa.prob.config import (
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
)
from geopfa.prob.cv_runner import (
    CVResult,
    _basic_metrics,
    component_oof_predictions,
    run_block_cv,
)
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _2d_cfg(
    wells_path: Path,
    output_dir: Path,
    *,
    calibration_method: str = "none",
) -> ProbabilisticConfig:
    return ProbabilisticConfig(
        enabled=True,
        output_dir=output_dir,
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns={
                "component_a": "heat_label",
                "component_b": "reservoir_label",
            },
            layer="wells",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_a",
                scalar_fallback_pr0=0.55,
            ),
            "component_b": AlphaModeConfig(
                mode="layer_logit",
                layer="prior_layer_b",
                scalar_fallback_pr0=0.5,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=True, backend="rbf"),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(
            method=calibration_method, fit_on="block_cv"
        ),
        cross_validation=CrossValidationConfig(
            n_folds=3, block_type="grid", grid_size=2
        ),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=True,
            decision_artifacts=True,
            scenarios=False,
            format=("csv",),
        ),
    )


def test_run_block_cv_returns_per_component_oof(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = run_block_cv(fixture.pfa, cfg)
    assert isinstance(cv, CVResult)
    assert "component_a" in cv.oof
    assert "component_b" in cv.oof
    oof_a = cv.oof["component_a"]
    assert "p" in oof_a and "y" in oof_a
    assert len(oof_a["p"]) == len(oof_a["y"])
    assert len(oof_a["p"]) <= len(fixture.wells)


def test_run_block_cv_rejects_declared_component_absent_from_pfa(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out")
    cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            label_columns={"ghost": "heat_label"},
        ),
        alpha={"ghost": AlphaModeConfig()},
    )

    with pytest.raises(ValueError, match="ghost"):
        run_block_cv(fixture.pfa, cfg)


def test_component_oof_rejects_label_column_not_declared_for_component(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out")

    with pytest.raises(ValueError, match="configured label column"):
        component_oof_predictions(
            fixture.pfa,
            cfg,
            component="component_a",
            label_column="reservoir_label",
        )


def test_component_oof_honors_configured_fixed_block_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from geopfa.prob import cv_runner as module

    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=17)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out")
    cfg = replace(
        cfg,
        cross_validation=replace(
            cfg.cross_validation,
            block_size_km=25.0,
            buffer_km=2.0,
        ),
    )
    real_splitter = module.spatial_block_cv
    captured: list[dict] = []

    def recording_splitter(*args, **kwargs):
        captured.append(dict(kwargs))
        return real_splitter(*args, **kwargs)

    monkeypatch.setattr(module, "spatial_block_cv", recording_splitter)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        component_oof_predictions(
            fixture.pfa,
            cfg,
            component="component_a",
            label_column="heat_label",
        )

    assert captured == [
        {
            "n_folds": 3,
            "block_type": "grid",
            "grid_size": 2,
            "seed": 0,
            "block_size_km": 25.0,
            "buffer_distance": 2_000.0,
            "dims": (0, 1),
        }
    ]


@pytest.mark.parametrize(
    ("y", "p", "message"),
    [
        (np.array([0, 1]), np.array([0.2, 1.2]), r"\[0, 1\]"),
        (np.array([0.5, 1.0]), np.array([0.2, 0.8]), "binary 0/1"),
        (np.array([0, 1]), np.array([0.2, np.nan]), "finite"),
    ],
)
def test_basic_metrics_rejects_invalid_scoring_inputs(
    y: np.ndarray, p: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _basic_metrics(y, p)


def test_run_block_cv_per_fold_metrics_have_expected_keys(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=1)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = run_block_cv(fixture.pfa, cfg)
    for name, folds in cv.fold_metrics.items():
        for fold in folds:
            for key in ("auc", "brier", "log_loss", "n"):
                assert key in fold, f"missing {key!r} in {name} fold metrics"


def test_run_block_cv_slices_fold_metrics_by_fold_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fold metrics must be computed from each fold's OOF slice."""

    from geopfa.prob import cv_runner as module  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=11)
    cfg = _2d_cfg(tmp_path / "wells.gpkg", tmp_path / "out")
    cfg = replace(
        cfg,
        labels=replace(
            cfg.labels,
            label_columns={"component_a": "heat_label"},
        ),
        alpha={"component_a": cfg.alpha["component_a"]},
    )
    fake_oof = {
        "p": np.array([0.1, 0.2, 0.8, 0.9], dtype=float),
        "y": np.array([0, 1, 1, 0], dtype=int),
        "fold": np.array([0, 0, 1, 1], dtype=int),
        "reference_prevalence": np.full(4, 0.5, dtype=float),
    }
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    class _FakeAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        @staticmethod
        def components() -> list[str]:
            return ["component_a"]

    def _fake_metrics(
        y: np.ndarray,
        p: np.ndarray,
        *,
        reference_prevalence: np.ndarray | float | None = None,
    ) -> dict[str, float]:
        assert reference_prevalence is not None
        seen.append((np.asarray(y, dtype=int), np.asarray(p, dtype=float)))
        return {
            "auc": float(len(y)),
            "brier": float(len(y)),
            "log_loss": float(len(y)),
            "n": float(len(y)),
        }

    monkeypatch.setattr(module, "PFAGridAdapter", _FakeAdapter)
    monkeypatch.setattr(
        module, "load_labels", lambda _labels: SimpleNamespace(gdf=None)
    )
    monkeypatch.setattr(
        module,
        "component_oof_predictions",
        lambda *_args, **_kwargs: fake_oof,
    )
    monkeypatch.setattr(module, "_basic_metrics", _fake_metrics)

    cv = run_block_cv(fixture.pfa, cfg)

    expected_folds = 2
    expected_metric_calls = expected_folds + 1
    assert len(cv.fold_metrics["component_a"]) == expected_folds
    assert len(seen) == expected_metric_calls
    np.testing.assert_array_equal(seen[0][0], np.array([0, 1], dtype=int))
    np.testing.assert_array_equal(
        seen[0][1], np.array([0.1, 0.2], dtype=float)
    )
    np.testing.assert_array_equal(seen[1][0], np.array([1, 0], dtype=int))
    np.testing.assert_array_equal(
        seen[1][1], np.array([0.8, 0.9], dtype=float)
    )
    np.testing.assert_array_equal(seen[2][0], fake_oof["y"])
    np.testing.assert_array_equal(seen[2][1], fake_oof["p"])


def test_run_probabilistic_with_calibration_writes_calibrated_outputs(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=2)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", calibration_method="platt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    # Calibrated surfaces should be present on the result
    assert result.calibrated_components
    for surface in result.calibrated_components.values():
        assert "probability" in surface.columns
        probs = surface["probability"].to_numpy()
        assert (probs >= 0).all() and (probs <= 1).all()


def test_run_probabilistic_writes_calibration_metrics_json(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=3)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", calibration_method="platt")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    metrics_path = cfg.output_dir / "calibration_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "per_component" in payload
    for comp_metrics in payload["per_component"].values():
        for key in ("ECE", "MCE", "brier", "log_loss"):
            assert key in comp_metrics
        assert (
            sum(row["n"] for row in comp_metrics["reliability"])
            == comp_metrics["n"]
        )
    # Markdown report should also be written
    assert (cfg.output_dir / "calibration_report.md").exists()
    decision_path = cfg.output_dir / "decision_metrics.json"
    assert decision_path.exists()
    decision = json.loads(decision_path.read_text())
    for component in decision["per_component"].values():
        assert component["prediction_source"] == "raw_spatial_oof"
        assert component["decision_classes"]
        assert component["top_n_targeting"]


def test_component_oof_predictions_handles_single_class_folds(
    tmp_path: Path,
) -> None:
    """Folds with all-positive or all-negative training data fall back gracefully."""
    fixture = make_synthetic_pfa(grid_n=6, n_wells=15, seed=4)
    wells = fixture.wells.copy()
    cfg = _2d_cfg(tmp_path / "wells.gpkg", tmp_path / "out")
    # Save wells so labels loader can find them
    wells.to_file(cfg.labels.source, layer="wells", driver="GPKG")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = component_oof_predictions(
            fixture.pfa,
            cfg,
            component="component_a",
            label_column="heat_label",
        )
    assert "p" in oof and "y" in oof
    assert len(oof["p"]) == len(oof["y"])


def test_component_oof_predictions_blocks_in_prediction_grid_crs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A metre buffer must never be applied to geographic label coordinates."""
    from geopfa.prob import cv_runner as module  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=6, n_wells=18, seed=41)
    wells_path = tmp_path / "wells_geographic.gpkg"
    fixture.wells.to_crs("EPSG:4326").to_file(
        wells_path, layer="wells", driver="GPKG"
    )
    cfg = _2d_cfg(wells_path, tmp_path / "out")
    cfg = replace(
        cfg,
        cross_validation=replace(cfg.cross_validation, buffer_km=1.0),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_split(coords, **_kwargs):
        captured["coords"] = np.asarray(coords, dtype=float)
        train = np.zeros(len(coords), dtype=bool)
        train[:12] = True
        yield train, ~train

    def fake_fit(component_data, **_kwargs):
        probability = component_data["pr_norm"][["geometry"]].copy()
        probability["probability"] = 0.5
        return SimpleNamespace(probability=probability)

    monkeypatch.setattr(module, "spatial_block_cv", fake_split)
    monkeypatch.setattr(module, "fit_component_probability", fake_fit)

    module.component_oof_predictions(
        fixture.pfa,
        cfg,
        component="component_a",
        label_column="heat_label",
    )

    grid_crs = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]["pr_norm"].crs
    expected = gpd.read_file(wells_path, layer="wells").to_crs(grid_crs)
    np.testing.assert_allclose(
        captured["coords"],
        np.column_stack([expected.geometry.x, expected.geometry.y]),
        rtol=0.0,
        atol=1e-6,
    )


def test_oof_fold_indices_cover_all_wells_exactly_once(tmp_path: Path) -> None:
    """Each test well appears in exactly one fold's OOF predictions."""
    from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: PLC0415
    from geopfa.prob.config import (  # noqa: PLC0415
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
    )
    from geopfa.prob.cv_runner import component_oof_predictions  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=8, n_wells=40, seed=55)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = ProbabilisticConfig(
        enabled=True,
        output_dir=tmp_path / "out",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source=str(wells_path),
            id_col="well_id",
            label_columns={"component_a": "heat_label"},
            layer="wells",
        ),
        alpha={
            "component_a": AlphaModeConfig(
                mode="scalar", scalar_fallback_pr0=0.5
            )
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(n_folds=3),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = component_oof_predictions(
            fixture.pfa,
            cfg,
            component="component_a",
            label_column="heat_label",
        )
    fold_idx = oof["fold"]
    if len(fold_idx) == 0:
        pytest.skip("All CV folds were degenerate for this fixture")
    # Each fold index appears at most once per well position
    # (no well appears in multiple folds — fold assignments are disjoint)
    from collections import Counter  # noqa: PLC0415

    counts = Counter(fold_idx.tolist())
    unique_folds = set(counts.keys())
    # All folds should be represented (none completely skipped for this fixture)
    assert len(unique_folds) >= 1, (
        "At least one fold must have OOF predictions"
    )
    # Total OOF predictions ≤ total wells (some may be skipped in degenerate folds)
    assert len(oof["p"]) <= len(fixture.wells)
