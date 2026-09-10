"""Tests for the run_probabilistic top-level entry point."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from geopfa.exceptions import GEOPFAValueError
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
    ScenarioConfig,
    SpatialFieldConfig,
)
from geopfa.prob.runner import ProbabilisticResult, run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _save_fixture_wells_as_gpkg(tmp_path: Path) -> Path:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=0)
    path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(path, layer="wells", driver="GPKG")
    return path


def _minimal_config(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
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
                scalar_fallback_pr0=0.50,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=True, backend="rbf"),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=__import__(
            "geopfa.prob.config", fromlist=["CrossValidationConfig"]
        ).CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=("csv",),
        ),
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_run_probabilistic_returns_result_with_component_surfaces(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=0)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert isinstance(result, ProbabilisticResult)
    assert set(result.components.keys()) == {"component_a", "component_b"}
    for surface in result.components.values():
        assert "probability" in surface.probability.columns
        prob = surface.probability["probability"].to_numpy()
        assert (prob >= 0).all() and (prob <= 1).all()
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert "probability" in result.combined.columns
    assert result.config is cfg


def test_run_probabilistic_writes_csv_when_requested(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=1)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    out_dir = tmp_path / "out"
    cfg = _minimal_config(wells_path, out_dir)
    # Enable CSV output
    cfg = ProbabilisticConfig(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=cfg.alpha,
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=OutputsConfig(
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=("csv",),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    # CSVs for per-component and combined surfaces should be present
    assert (out_dir / "component_a_probability.csv").exists()
    assert (out_dir / "component_b_probability.csv").exists()
    assert (out_dir / "combined_probability.csv").exists()


def test_run_probabilistic_disabled_short_circuits(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=2)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=False,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=cfg.alpha,
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=cfg.outputs,
    )
    result = run_probabilistic(fixture.pfa, cfg)
    assert result.skipped is True
    assert result.components == {}


def test_run_probabilistic_validates_before_disabled_short_circuit(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    invalid = replace(cfg, enabled=False, output_dir=Path("."))

    with pytest.raises(GEOPFAValueError, match="output_dir"):
        run_probabilistic({}, invalid)


def test_run_probabilistic_rejects_unmanifested_existing_outputs(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=2)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale_probability.csv").write_text(
        "probability\n0.9\n", encoding="utf-8"
    )
    cfg = _minimal_config(wells_path, output_dir)

    with pytest.raises(GEOPFAValueError, match="not a fresh or resumable"):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_rejects_outputs_from_different_config(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=2)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config_hash": "not-the-current-config",
                "implementation_sha256": "not-the-current-code",
                "inputs": [],
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = _minimal_config(wells_path, output_dir)

    with pytest.raises(GEOPFAValueError, match="different effective config"):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_rejects_outputs_from_different_implementation(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=2)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    output_dir = tmp_path / "out"
    cfg = _minimal_config(wells_path, output_dir)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_sha256"] = "not-the-current-code"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        GEOPFAValueError, match="different probabilistic implementation"
    ):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_allows_explicit_component_subset(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=4)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    # Configure only component_a; unrelated PFA components stay outside the
    # declared analysis rather than being inferred from an intersection.
    cfg = ProbabilisticConfig(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=LabelsConfig(
            source=cfg.labels.source,
            id_col=cfg.labels.id_col,
            label_columns={"component_a": "heat_label"},
            layer=cfg.labels.layer,
        ),
        alpha={"component_a": cfg.alpha["component_a"]},
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=cfg.outputs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = run_probabilistic(fixture.pfa, cfg)
    assert "component_a" in result.components
    assert "component_b" not in result.components


def test_run_probabilistic_rejects_declared_component_absent_from_pfa(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=4)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    ghost = "misspelled_component"
    cfg = ProbabilisticConfig(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=LabelsConfig(
            source=cfg.labels.source,
            id_col=cfg.labels.id_col,
            label_columns={**cfg.labels.label_columns, ghost: "heat_label"},
            layer=cfg.labels.layer,
        ),
        alpha={
            **cfg.alpha,
            ghost: AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.5),
        },
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=cfg.outputs,
    )

    with pytest.raises(GEOPFAValueError, match=ghost):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_runs_scenarios_when_configured(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=5)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    cfg = _minimal_config(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=cfg.alpha,
        evidence=cfg.evidence,
        spatial_field=cfg.spatial_field,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=(
            ScenarioConfig(
                name="full", include_priors=True, include_spatial=True
            ),
            ScenarioConfig(
                name="no_spatial", include_priors=True, include_spatial=False
            ),
        ),
        outputs=cfg.outputs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert set(result.scenarios.keys()) == {"full", "no_spatial"}
    for scenario_result in result.scenarios.values():
        assert "component_a" in scenario_result
        assert "component_b" in scenario_result


def test_run_probabilistic_rejects_unknown_scenario_drop_layer(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=51)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    base = _minimal_config(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        scenarios=(
            ScenarioConfig(
                name="typo", drop_layers=("not_a_real_evidence_layer",)
            ),
        ),
    )

    with pytest.raises(GEOPFAValueError, match="not_a_real_evidence_layer"):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_loads_from_json_config(tmp_path: Path) -> None:
    """Round-trip via load_probabilistic_config + run_probabilistic on disk."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=25, seed=6)
    wells_path = _save_fixture_wells_as_gpkg(tmp_path)
    out_dir = tmp_path / "out"
    pfa_cfg = {
        "criteria": {},  # ignored by the prob runner; just satisfies the schema
        "probabilistic": {
            "enabled": True,
            "output_dir": str(out_dir),
            "dimensions": "2d",
            "labels": {
                "source": str(wells_path),
                "id_col": "well_id",
                "layer": "wells",
                "label_columns": {
                    "component_a": "heat_label",
                    "component_b": "reservoir_label",
                },
            },
            "alpha": {
                "component_a": {
                    "mode": "layer_logit",
                    "layer": "prior_layer_a",
                    "scalar_fallback_pr0": 0.55,
                },
                "component_b": {
                    "mode": "layer_logit",
                    "layer": "prior_layer_b",
                    "scalar_fallback_pr0": 0.50,
                },
            },
            "spatial_field": {"enabled": True, "backend": "rbf"},
            "calibration": {"method": "none"},
            "outputs": {
                "probability_rasters": False,
                "uncertainty_rasters": False,
                "calibration_artifacts": False,
                "decision_artifacts": False,
                "scenarios": False,
                "format": ["csv"],
            },
        },
    }
    cfg_path = tmp_path / "pfa_config.json"
    cfg_path.write_text(json.dumps(pfa_cfg))

    from geopfa.prob.config import load_probabilistic_config

    cfg = load_probabilistic_config(cfg_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert isinstance(result, ProbabilisticResult)
    assert set(result.components.keys()) == {"component_a", "component_b"}


def test_nnpu_changes_probabilities(tmp_path: Path) -> None:
    """nnPU risk estimation should differ from treating unlabeled rows as negatives."""
    import copy
    from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=8, n_wells=60, seed=77)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    def _run(pu_mode):
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

        cfg = ProbabilisticConfig(
            enabled=True,
            output_dir=tmp_path / f"out_{pu_mode}",
            dimensions="2d",
            grid=GridConfig(),
            labels=LabelsConfig(
                source=str(wells_path),
                id_col="well_id",
                label_columns={"component_a": "heat_label"},
                layer="wells",
                pu_mode=pu_mode,
                pu_class_prior=0.5 if pu_mode == "nnpu" else None,
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
            cross_validation=CrossValidationConfig(),
            combination=CombinationConfig(rule="product"),
            scenarios=(),
            outputs=OutputsConfig(format=("csv",)),
        )
        import warnings as _w  # noqa: PLC0415

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            from geopfa.prob.runner import run_probabilistic  # noqa: PLC0415

            return run_probabilistic(fixture.pfa, cfg)

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        res_off = _run("off")
        res_pu = _run("nnpu")

    p_off = (
        res_off.components["component_a"].probability["probability"].to_numpy()
    )
    p_pu = (
        res_pu.components["component_a"].probability["probability"].to_numpy()
    )
    # The nnPU risk estimator should produce measurably different probabilities
    assert not np.allclose(p_off, p_pu, atol=1e-6), (
        "nnPU risk estimation produced identical results to off — "
        "correction is not being applied"
    )


def test_pu_mode_nnpu_differs_from_off(tmp_path: Path) -> None:
    """nnPU estimation must produce combined probabilities different from off.

    Uses identical seeds and fixture to isolate the effect of pu_mode.
    Verifies both outputs are valid probability surfaces and that they differ.
    """
    fixture = make_synthetic_pfa(grid_n=8, n_wells=40, seed=42)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    def _run(pu_mode: str) -> ProbabilisticResult:
        cfg = ProbabilisticConfig(
            enabled=True,
            output_dir=tmp_path / f"out_{pu_mode}",
            dimensions="2d",
            grid=GridConfig(),
            labels=LabelsConfig(
                source=str(wells_path),
                id_col="well_id",
                label_columns={"component_a": "heat_label"},
                layer="wells",
                pu_mode=pu_mode,
                pu_class_prior=0.5 if pu_mode == "nnpu" else None,
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
            cross_validation=CrossValidationConfig(),
            combination=CombinationConfig(rule="product"),
            scenarios=(),
            outputs=OutputsConfig(format=("csv",)),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_probabilistic(fixture.pfa, cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_off = _run("off")
        res_pu = _run("nnpu")

    combined_off = res_off.combined["probability"].to_numpy()
    combined_pu = res_pu.combined["probability"].to_numpy()

    assert ((combined_off >= 0) & (combined_off <= 1)).all(), (
        "off mode combined probabilities outside [0, 1]"
    )
    assert ((combined_pu >= 0) & (combined_pu <= 1)).all(), (
        "nnpu mode combined probabilities outside [0, 1]"
    )
    assert not np.allclose(combined_off, combined_pu, atol=1e-6), (
        "nnPU estimation produced identical combined probabilities to off"
    )


def test_run_probabilistic_disabled_config_skips_without_validation(
    tmp_path: Path,
) -> None:
    """enabled=False must return skipped=True without triggering validate_raise().

    A skeleton/minimal disabled config should not fail validation — only an
    active config (enabled=True) needs to be fully valid.
    """
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
    from geopfa.prob.runner import run_probabilistic  # noqa: PLC0415

    # Minimal disabled config — output_dir is empty which validate() would reject
    cfg = ProbabilisticConfig(
        enabled=False,
        output_dir=tmp_path / "never_written",
        dimensions="2d",
        grid=GridConfig(),
        labels=LabelsConfig(
            source="irrelevant.gpkg",
            id_col="well_id",
            label_columns={"component_a": "label"},
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
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(format=("csv",)),
    )
    # Should NOT raise even though output_dir doesn't exist
    result = run_probabilistic({}, cfg)
    assert result.skipped is True


# ---------------------------------------------------------------------------
# _fit_spatial_field RBF fallback warning (P1-S01)
# ---------------------------------------------------------------------------


def test_fit_spatial_field_rbf_failure_is_not_hidden() -> None:
    """A failed spatial optimizer must abort instead of emitting a flat map."""
    from unittest.mock import patch  # noqa: PLC0415

    from geopfa.prob.fitting import fit_component_probability  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=7)
    component_data = fixture.pfa["criteria"]["geologic"]["components"][
        "component_a"
    ]

    with (
        pytest.raises(RuntimeError, match="RBF spatial field fit failed"),
        patch(
            "geopfa.prob.fitting.Rbf",
            side_effect=ValueError("injected RBF failure"),
        ),
    ):
        fit_component_probability(
            component_data,
            prior_probability=0.55,
            include_spatial=True,
            spatial_backend="rbf",
            labeled_wells=fixture.wells,
            label_column="heat_label",
        )


# ---------------------------------------------------------------------------
# GBLK backend dispatch (P5-S02)
# ---------------------------------------------------------------------------


def test_run_probabilistic_gblk_backend_returns_result(tmp_path: Path) -> None:
    """inference.backend='gblk' must route to run_gblk_probabilistic.

    The returned ProbabilisticResult must carry per-component probability
    surfaces (one per configured component), a combined surface, and have
    skipped=False.
    """
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=10)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _minimal_config(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=cfg.alpha,
        evidence=cfg.evidence,
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="gblk"),
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=cfg.scenarios,
        outputs=cfg.outputs,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_probabilistic(fixture.pfa, cfg)
    assert isinstance(result, ProbabilisticResult)
    assert result.skipped is False
    assert set(result.components.keys()) == {"component_a", "component_b"}
    for surface in result.components.values():
        assert "probability" in surface.probability.columns
        prob = surface.probability["probability"].to_numpy()
        assert (prob >= 0).all() and (prob <= 1).all()
    assert isinstance(result.combined, gpd.GeoDataFrame)
    assert "probability" in result.combined.columns
    assert result.config is cfg


def test_run_probabilistic_gblk_executes_configured_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace

    fixture = make_synthetic_pfa(grid_n=8, n_wells=24, seed=101)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _minimal_config(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        inference=InferenceConfig(backend="gblk"),
        spatial_field=SpatialFieldConfig(enabled=False),
        scenarios=(
            ScenarioConfig(
                name="no_priors",
                include_priors=False,
                include_spatial=False,
            ),
        ),
        outputs=replace(base.outputs, format=()),
    )
    received: list[ProbabilisticConfig] = []

    def fake_gblk(pfa, config, *, criteria, nc, posterior_scope="baseline"):
        del pfa, criteria, nc, posterior_scope
        received.append(config)
        return ProbabilisticResult(config=config, skipped=False)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.run_gblk_probabilistic",
        fake_gblk,
    )

    result = run_probabilistic(fixture.pfa, cfg)

    assert len(received) == 2
    assert received[0] is cfg
    assert received[1].scenarios == ()
    assert set(result.scenarios) == {"no_priors"}


def test_run_probabilistic_gblk_scenario_really_drops_evidence(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    fixture = make_synthetic_pfa(grid_n=7, n_wells=30, seed=102)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _minimal_config(wells_path, tmp_path / "out")
    cfg = replace(
        base,
        inference=InferenceConfig(backend="gblk"),
        spatial_field=SpatialFieldConfig(enabled=False),
        scenarios=(
            ScenarioConfig(name="no_gradient", drop_layers=("gradient",)),
        ),
        outputs=replace(base.outputs, format=()),
    )

    result = run_probabilistic(fixture.pfa, cfg)

    assert result.components["component_a"].feature_names == ("gradient",)
    assert result.scenarios["no_gradient"]["component_a"].feature_names == ()


def test_run_probabilistic_forwards_configured_lattice_centers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=13)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _minimal_config(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=base.enabled,
        output_dir=base.output_dir,
        dimensions=base.dimensions,
        grid=base.grid,
        labels=base.labels,
        alpha=base.alpha,
        evidence=base.evidence,
        spatial_field=SpatialFieldConfig(
            enabled=False,
            lattice_centers_per_dimension=3,
        ),
        inference=InferenceConfig(backend="gblk"),
        calibration=base.calibration,
        cross_validation=base.cross_validation,
        combination=base.combination,
        scenarios=base.scenarios,
        outputs=base.outputs,
    )
    received: dict[str, int] = {}

    def fake_gblk(pfa, config, *, criteria, nc):
        del pfa, criteria
        received["nc"] = nc
        return ProbabilisticResult(config=config, skipped=True)

    monkeypatch.setattr(
        "geopfa.prob.gblk_runner.run_gblk_probabilistic",
        fake_gblk,
    )

    run_probabilistic(fixture.pfa, cfg)

    assert received == {"nc": 3}


def test_run_probabilistic_gblk_rejects_silently_ignored_calibration(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=12)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    base = _minimal_config(wells_path, tmp_path / "out")
    cfg = ProbabilisticConfig(
        enabled=base.enabled,
        output_dir=base.output_dir,
        dimensions=base.dimensions,
        grid=base.grid,
        labels=base.labels,
        alpha=base.alpha,
        evidence=base.evidence,
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="gblk"),
        calibration=CalibrationConfig(method="platt", fit_on="block_cv"),
        cross_validation=base.cross_validation,
        combination=base.combination,
        scenarios=base.scenarios,
        outputs=base.outputs,
    )

    with pytest.raises(
        GEOPFAValueError, match="does not apply post-hoc calibration"
    ):
        run_probabilistic(fixture.pfa, cfg)


def test_run_probabilistic_sequential_backend_emits_deprecation_warning(
    tmp_path: Path,
) -> None:
    """inference.backend='sequential' must emit a DeprecationWarning."""
    fixture = make_synthetic_pfa(grid_n=8, n_wells=30, seed=12)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _minimal_config(wells_path, tmp_path / "out")
    with pytest.warns(DeprecationWarning, match="sequential"):
        run_probabilistic(fixture.pfa, cfg)
