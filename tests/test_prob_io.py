"""Tests for the probability-output writer."""

from __future__ import annotations

import json
import hashlib
import warnings
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

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
    SiteSelectionConfig,
    SpatialFieldConfig,
)
from geopfa.prob.io import (
    PosteriorDrawBlockWriter,
    _gdf_to_raster,
    load_posterior_draw_state,
    write_geotiff_outputs,
    write_manifest,
    write_parquet_outputs,
    write_probability_outputs,
    write_vtk_outputs,
    verify_posterior_draw_bundle,
)
from geopfa.prob.runner import run_probabilistic
from tests.fixtures.synthetic_prob import make_synthetic_pfa
from tests.fixtures.synthetic_prob_3d import make_synthetic_pfa_3d


def _2d_cfg(
    wells_path: Path, output_dir: Path, formats: tuple[str, ...]
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
                scalar_fallback_pr0=0.50,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=True, backend="rbf"),
        inference=InferenceConfig(backend="sequential"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters="geotiff" in formats,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=formats,
        ),
    )


# ---------------------------------------------------------------------------
# Direct writer tests (unit)
# ---------------------------------------------------------------------------


def test_write_geotiff_outputs_creates_tif_per_component(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=20, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"].copy()
    grid_gdf["probability"] = np.linspace(0.0, 1.0, len(grid_gdf))
    write_geotiff_outputs(
        {"component_a": grid_gdf}, tmp_path / "out", value_col="probability"
    )
    assert (tmp_path / "out" / "component_a_probability.tif").exists()


def test_unified_geotiff_writer_emits_bayesian_interval_rasters(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=12, seed=13)
    grid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    grid["probability"] = 0.5
    grid["probability_lo"] = 0.25
    grid["probability_hi"] = 0.75

    write_probability_outputs(
        {"component_a": grid},
        tmp_path / "out",
        formats=("geotiff",),
        include_uncertainty=True,
    )

    assert (tmp_path / "out/component_a_probability.tif").is_file()
    assert (tmp_path / "out/component_a_probability_lo.tif").is_file()
    assert (tmp_path / "out/component_a_probability_hi.tif").is_file()


def test_write_parquet_outputs_round_trips(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=12, seed=1)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"].copy()
    grid_gdf["probability"] = np.full(len(grid_gdf), 0.7)
    write_parquet_outputs({"component_a": grid_gdf}, tmp_path / "out")
    parquet_path = tmp_path / "out" / "component_a_probability.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert "probability" in df.columns
    assert "x" in df.columns and "y" in df.columns
    np.testing.assert_allclose(df["probability"].to_numpy(), 0.7)


def test_write_parquet_outputs_uses_active_geometry_name(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=42)
    grid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    grid["probability"] = 0.7
    grid = grid.rename_geometry("grid_point")

    written = write_parquet_outputs({"component_a": grid}, tmp_path / "out")

    table = pd.read_parquet(written[0])
    assert "grid_point" not in table.columns
    assert {"x", "y", "probability"} <= set(table.columns)


def test_direct_geotiff_writer_rejects_3d_before_creating_output(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        {"probability": [0.5]},
        geometry=gpd.points_from_xy([0.0], [0.0], z=[-1.0]),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="GeoTIFF output requires 2-D"):
        write_geotiff_outputs({"heat": grid}, output_dir)

    assert not output_dir.exists()


def test_direct_vtk_writer_rejects_2d_before_creating_output(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        {"probability": [0.5]},
        geometry=gpd.points_from_xy([0.0], [0.0]),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="VTK output requires 3-D"):
        write_vtk_outputs({"heat": grid}, output_dir)

    assert not output_dir.exists()


def test_unified_writer_preflights_every_surface_before_writing(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=43)
    valid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    valid["probability"] = 0.5
    invalid = valid.drop(columns="probability")
    output_dir = tmp_path / "out"

    with pytest.raises(
        ValueError, match="missing requested column 'probability'"
    ):
        write_probability_outputs(
            {"first": valid, "second": invalid},
            output_dir,
            formats=("csv", "geotiff"),
        )

    assert not output_dir.exists()


def test_unified_writer_rejects_incompatible_dimension_before_any_format(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        {"probability": [0.5]},
        geometry=gpd.points_from_xy([0.0], [0.0], z=[-1.0]),
    )
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="GeoTIFF output requires 2-D"):
        write_probability_outputs(
            {"heat": grid}, output_dir, formats=("csv", "geotiff")
        )

    assert not output_dir.exists()


def test_unified_csv_writer_emits_coordinates_and_attributes(
    tmp_path: Path,
) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "probability": [0.25, 0.75],
            "probability_lo": [0.10, 0.60],
        },
        geometry=gpd.points_from_xy(
            [500_000.0, 500_100.0],
            [4_300_000.0, 4_300_100.0],
            z=[-3_000.0, -2_900.0],
        ),
        crs="EPSG:32611",
    )

    written = write_probability_outputs(
        {"heat": gdf}, tmp_path / "out", formats=("csv",)
    )

    assert written == [tmp_path / "out" / "heat_probability.csv"]
    table = pd.read_csv(written[0])
    assert set(table.columns) == {
        "probability",
        "probability_lo",
        "x",
        "y",
        "z",
    }
    assert "geometry" not in table.columns
    np.testing.assert_allclose(table["z"], [-3_000.0, -2_900.0])


def test_unified_csv_writer_skips_empty_surfaces(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    written = write_probability_outputs(
        {"empty": gpd.GeoDataFrame()}, output_dir, formats=("csv",)
    )

    assert written == []
    assert output_dir.is_dir()
    assert not (output_dir / "empty_probability.csv").exists()


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escaped", "nested/name", r"nested\name", "heat map", "café", "CON"],
)
def test_unified_writer_rejects_nonportable_surface_names_before_writing(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=25)
    grid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    grid["probability"] = 0.5
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="portable surface name"):
        write_probability_outputs(
            {"valid": grid, unsafe_name: grid},
            output_dir,
            formats=("geotiff", "csv"),
        )

    assert not output_dir.exists()
    assert not (tmp_path / "escaped_probability.tif").exists()


def test_unified_writer_rejects_case_normalized_name_collisions_before_writing(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=26)
    grid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    grid["probability"] = 0.5
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="case-insensitive collision"):
        write_probability_outputs(
            {"Heat": grid, "heat": grid},
            output_dir,
            formats=("parquet",),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "writer",
    [write_geotiff_outputs, write_parquet_outputs, write_vtk_outputs],
)
def test_direct_format_writers_reject_unsafe_surface_names_before_writing(
    tmp_path: Path,
    writer,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=27)
    grid = fixture.pfa["criteria"]["geologic"]["components"]["component_a"][
        "pr_norm"
    ].copy()
    grid["probability"] = 0.5
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="portable surface name"):
        writer({"../escaped": grid}, output_dir)

    assert not output_dir.exists()


def test_gdf_to_raster_rejects_irregular_point_support() -> None:
    rng = np.random.default_rng(42)
    xs = rng.uniform(500_000.0, 600_000.0, 7)
    ys = rng.uniform(4_300_000.0, 4_400_000.0, 7)
    vals = rng.uniform(0.0, 1.0, 7)
    gdf = gpd.GeoDataFrame(
        {"probability": vals},
        geometry=gpd.points_from_xy(xs, ys),
        crs="EPSG:32611",
    )
    with pytest.raises(ValueError, match="complete rectilinear"):
        _gdf_to_raster(gdf, value_col="probability")


def test_gdf_to_raster_rejects_ambiguous_cell_collision() -> None:
    gdf = gpd.GeoDataFrame(
        {"probability": [0.1, 0.2, 0.9]},
        geometry=gpd.points_from_xy([0.0, 0.1, 1.0], [0.0, 0.0, 1.0]),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="complete rectilinear"):
        _gdf_to_raster(gdf, value_col="probability")


def test_write_probability_outputs_rejects_unknown_format(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown output format"):
        write_probability_outputs({}, tmp_path / "out", formats=("typo",))


def test_write_vtk_outputs_requires_pyvista_for_3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    gdf = gpd.GeoDataFrame(
        {"probability": [0.25, 0.75]},
        geometry=gpd.points_from_xy([0.0, 1.0], [0.0, 1.0], z=[0.0, 1.0]),
        crs="EPSG:32611",
    )
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(ImportError, match="PyVista"):
        write_vtk_outputs({"heat": gdf}, tmp_path / "out")


def test_write_vtk_outputs_creates_vti_for_3d(tmp_path: Path) -> None:
    pyvista = pytest.importorskip("pyvista")  # noqa: F841
    fixture = make_synthetic_pfa_3d(grid_n=4, grid_nz=3, n_wells=8, seed=0)
    comp = fixture.pfa["criteria"]["geologic"]["components"]["component_a"]
    grid_gdf = comp["pr_norm"].copy()
    grid_gdf["probability"] = np.random.default_rng(0).uniform(
        0, 1, size=len(grid_gdf)
    )
    write_vtk_outputs({"component_a": grid_gdf}, tmp_path / "out")
    assert (tmp_path / "out" / "component_a_probability.vtp").exists()


def test_write_manifest_records_files_and_hashes(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", formats=("csv",))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "fake_artifact.csv").write_text(
        "hello,world\n", encoding="utf-8"
    )
    config_source = tmp_path / "config.json"
    config_source.write_text('{"probabilistic": {}}\n')
    write_manifest(
        out_dir,
        config=cfg,
        run_id="test",
        input_artifacts={"config": config_source},
    )
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert "files" in data
    assert any(
        entry["path"].endswith("fake_artifact.csv") for entry in data["files"]
    )
    assert "config_hash" in data
    assert "run_id" in data
    assert data["schema_version"] == 1
    assert data["producer"]["package"] == "geoPFA"
    assert data["producer"]["version"]
    assert "source" not in data["producer"]
    dependency = data["producer"]["dependencies"][0]
    assert dependency["package"] == "latticekrigx"
    assert dependency["version"] == "0.1.0.dev0"
    assert len(dependency["implementation_sha256"]) == 64
    assert "source" not in dependency
    assert data["config"] == cfg.to_dict()
    config_record = next(
        record for record in data["inputs"] if record["name"] == "config"
    )
    assert config_record["sha256"] == _sha256(config_source)
    assert config_record["size_bytes"] == config_source.stat().st_size


def test_manifest_omits_absent_prior_only_label_source(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=41)
    cfg = ProbabilisticConfig.from_dict(
        {
            "enabled": True,
            "output_dir": str(tmp_path / "out"),
            "dimensions": "2d",
            "alpha": {
                "component_a": {
                    "mode": "scalar",
                    "scalar_fallback_pr0": 0.4,
                    "force_prior_predictive": True,
                }
            },
            "outputs": {"format": []},
        }
    )

    result = run_probabilistic(fixture.pfa, cfg)

    assert set(result.components) == {"component_a"}
    manifest = json.loads(
        (cfg.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert {record["name"] for record in manifest["inputs"]} == set()


def test_manifest_binds_complete_shapefile_input_bundles(
    tmp_path: Path,
) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=0)
    labels_path = tmp_path / "labels.shp"
    candidates_path = tmp_path / "candidates.shp"
    supplied_path = tmp_path / "supplied.shp"
    shape_frame = fixture.wells[["well_id", "heat_label", "geometry"]]
    shape_frame.to_file(labels_path, driver="ESRI Shapefile")
    shape_frame.to_file(candidates_path, driver="ESRI Shapefile")
    shape_frame.to_file(supplied_path, driver="ESRI Shapefile")

    cfg = replace(
        _2d_cfg(labels_path, tmp_path / "out", formats=("csv",)),
        site_selection=SiteSelectionConfig(
            mode="joint_binary",
            candidate_source=str(candidates_path),
            id_col="well_id",
            outcome_feature_columns=("prior_layer_a",),
            selection_feature_columns=("prior_layer_b",),
        ),
    )
    cfg.output_dir.mkdir()
    (cfg.output_dir / "result.csv").write_text("value\n1\n", encoding="utf-8")

    write_manifest(
        cfg.output_dir,
        config=cfg,
        input_artifacts={"caller.supplied": supplied_path},
    )

    manifest = json.loads(
        (cfg.output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records = {record["name"]: record for record in manifest["inputs"]}
    required_suffixes = (".dbf", ".prj", ".shx")
    for logical_name, source in (
        ("labels.source", labels_path),
        ("site_selection.candidate_source", candidates_path),
        ("caller.supplied", supplied_path),
    ):
        assert records[logical_name]["path"] == str(source.resolve())
        for suffix in required_suffixes:
            member_name = f"{logical_name}{suffix}"
            member_path = source.with_suffix(suffix).resolve()
            assert records[member_name]["path"] == str(member_path)
            assert records[member_name]["sha256"] == _sha256(member_path)

    cpg_names = {
        f"{logical_name}.cpg"
        for logical_name in (
            "labels.source",
            "site_selection.candidate_source",
            "caller.supplied",
        )
        if Path(records[logical_name]["path"]).with_suffix(".cpg").is_file()
    }
    assert cpg_names <= records.keys()
    supplied_path.with_suffix(".shx").unlink()
    with pytest.raises(FileNotFoundError, match=r"supplied\.shx"):
        write_manifest(
            tmp_path / "missing-sidecar-output",
            config=cfg,
            input_artifacts={"caller.supplied": supplied_path},
        )


def test_write_manifest_rejects_runtime_source_change(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=4, n_wells=8, seed=0)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", formats=("csv",))

    with pytest.raises(
        RuntimeError, match="runtime source changed during run"
    ):
        write_manifest(
            cfg.output_dir,
            config=cfg,
            expected_implementation_sha256="0" * 64,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _posterior_metadata(
    component_names: tuple[str, ...], **metadata: object
) -> dict[str, object]:
    return {
        **metadata,
        "component_roles": dict.fromkeys(component_names, "joint_posterior"),
    }


def test_incremental_posterior_writer_rejects_geometric_mean(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )

    with pytest.raises(ValueError, match="geometric_mean"):
        PosteriorDrawBlockWriter(
            grid,
            tmp_path,
            component_names=("heat", "reservoir"),
            n_draws=1,
            block_size=1,
            seed=1,
            combination_rule="geometric_mean",
            scope="baseline",
            state_arrays={"coefficient_draws": np.zeros((1, 1))},
            state_metadata=_posterior_metadata(("heat", "reservoir")),
        )


def test_incremental_posterior_writer_persists_state_and_decomposed_blocks(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]),
        crs="EPSG:32610",
    )
    prior = np.array([[0.1, -0.2], [0.2, -0.1], [0.3, 0.0]])
    coefficient_draws = np.arange(8, dtype=float).reshape(4, 2)
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat", "reservoir"),
        n_draws=4,
        block_size=2,
        seed=17,
        combination_rule="product",
        scope="scenario:physical_isotropic",
        state_arrays={"coefficient_draws": coefficient_draws},
        state_metadata={
            "model": "test",
            "coordinate_transform": "physical_isotropic",
            "component_roles": {
                "heat": "joint_posterior",
                "reservoir": "joint_posterior",
            },
        },
    )

    all_components = []
    for start in (0, 2):
        evidence = np.full((2, 3, 2), 0.05 * (start + 1))
        spatial = np.full((2, 3, 2), -0.02 * start)
        logits = prior[np.newaxis, :, :] + evidence + spatial
        components = 1.0 / (1.0 + np.exp(-logits))
        all_components.append(components)
        writer.write_block(
            start,
            component_probability=components,
            prior_logit=prior,
            evidence_logit=evidence,
            spatial_logit=spatial,
        )

    summary = writer.finalize(ci_level=0.9)
    index = json.loads(summary.index_path.read_text(encoding="utf-8"))
    expected = np.concatenate(all_components)
    expected_combined = np.prod(expected, axis=2)

    assert index["schema_version"] == 1
    assert index["combination_rule"] == "product"
    assert index["combination_estimand"] == "within_draw_component_product"
    assert index["scope"] == "scenario:physical_isotropic"
    assert index["cross_scenario_pairing"] == "not_identified"
    assert index["state"]["metadata"]["coordinate_transform"] == (
        "physical_isotropic"
    )
    assert (
        _sha256(
            summary.index_path.parent / index["state"]["arrays"][0]["path"]
        )
        == (index["state"]["arrays"][0]["sha256"])
    )
    np.testing.assert_allclose(summary.component_mean, expected.mean(axis=0))
    np.testing.assert_allclose(
        summary.combined_mean, expected_combined.mean(axis=0)
    )
    np.testing.assert_allclose(
        summary.component_interval,
        np.quantile(expected, [0.05, 0.95], axis=0),
    )
    np.testing.assert_allclose(
        summary.combined_interval,
        np.quantile(expected_combined, [0.05, 0.95], axis=0),
    )
    for record in index["blocks"]:
        block_path = summary.index_path.parent / record["path"]
        assert _sha256(block_path) == record["sha256"]
        with np.load(block_path, allow_pickle=False) as payload:
            np.testing.assert_allclose(
                payload["component_probability"],
                1.0
                / (
                    1.0
                    + np.exp(
                        -(
                            payload["prior_logit"][np.newaxis, :, :]
                            + payload["evidence_logit"]
                            + payload["spatial_logit"]
                        )
                    )
                ),
            )


def test_incremental_writer_persists_mixed_family_predictive_blocks(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0], [2.0, 3.0]),
        crs="EPSG:32610",
    )
    prior = np.array([[7.5, -0.2], [8.5, 0.2]])
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat", "reservoir"),
        component_models=(
            {"family": "gaussian", "event_threshold_scaled": 8.0},
            {"family": "bernoulli"},
        ),
        n_draws=2,
        block_size=2,
        seed=17,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_linear_predictor": prior},
        state_metadata=_posterior_metadata(("heat", "reservoir")),
    )
    evidence = np.array(
        [
            [[0.1, 0.2], [0.2, -0.1]],
            [[-0.1, 0.1], [0.3, -0.2]],
        ]
    )
    spatial = np.zeros_like(evidence)
    precision = np.array([[4.0, 1.0], [9.0, 1.0]])
    eta = prior[np.newaxis, :, :] + evidence
    probability = np.empty_like(eta)
    probability[:, :, 0] = ndtr(
        (eta[:, :, 0] - 8.0) * np.sqrt(precision[:, 0, np.newaxis])
    )
    probability[:, :, 1] = 1.0 / (1.0 + np.exp(-eta[:, :, 1]))

    writer.write_predictive_block(
        0,
        component_probability=probability,
        prior_linear_predictor=prior,
        evidence_linear_predictor=evidence,
        spatial_linear_predictor=spatial,
        likelihood_precision=precision,
    )
    summary = writer.finalize(ci_level=0.9)
    index = json.loads(summary.index_path.read_text(encoding="utf-8"))
    audit = verify_posterior_draw_bundle(summary.index_path)

    assert index["schema_version"] == 1
    assert index["component_models"] == [
        {"family": "gaussian", "event_threshold_scaled": 8.0},
        {"family": "bernoulli"},
    ]
    assert index["decomposition"] == (
        "prior_linear_predictor + evidence_linear_predictor + "
        "spatial_linear_predictor"
    )
    assert audit["schema_version"] == 1
    np.testing.assert_allclose(
        summary.component_mean, probability.mean(axis=0)
    )


def test_incremental_writer_materializes_only_selected_cells_but_summarizes_full_grid(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]),
        crs="EPSG:32610",
    )
    prior = np.array([[-0.5, 0.1], [0.0, 0.2], [0.5, 0.3]], dtype=np.float64)
    probability = np.array(
        [
            [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
            [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        ],
        dtype=np.float64,
    )
    eta = np.log(probability) - np.log1p(-probability)
    evidence = eta - prior[np.newaxis, :, :]
    spatial = np.zeros_like(evidence)
    selected = np.array([0, 2], dtype=np.int64)
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat", "reservoir"),
        n_draws=2,
        block_size=1,
        seed=17,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": prior},
        state_metadata=_posterior_metadata(("heat", "reservoir")),
        draw_cell_indices=selected,
    )
    writer.write_block(
        0,
        component_probability=probability[:1],
        prior_logit=prior,
        evidence_logit=evidence[:1],
        spatial_logit=spatial[:1],
    )

    resumed = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat", "reservoir"),
        n_draws=2,
        block_size=1,
        seed=17,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": prior},
        state_metadata=_posterior_metadata(("heat", "reservoir")),
        draw_cell_indices=selected,
    )
    resumed.write_block(
        1,
        component_probability=probability[1:],
        prior_logit=prior,
        evidence_logit=evidence[1:],
        spatial_logit=spatial[1:],
    )
    summary = resumed.finalize(ci_level=0.5)
    index = json.loads(summary.index_path.read_text(encoding="utf-8"))

    assert index["n_cells"] == 3
    assert index["materialized_n_cells"] == 2
    assert index["draw_materialization"]["mode"] == "cell_subset"
    stored_selection = np.load(
        summary.index_path.parent
        / index["draw_materialization"]["cell_indices"]["path"],
        allow_pickle=False,
    )
    np.testing.assert_array_equal(stored_selection, selected)
    with np.load(
        summary.index_path.parent / index["blocks"][0]["path"],
        allow_pickle=False,
    ) as block:
        assert block["component_probability"].shape == (1, 2, 2)
        np.testing.assert_array_equal(block["prior_logit"], prior[selected])
    np.testing.assert_allclose(
        summary.component_mean, probability.mean(axis=0)
    )
    np.testing.assert_allclose(
        summary.combined_mean, np.prod(probability, axis=2).mean(axis=0)
    )
    assert set(index["full_grid_summaries"]) == {
        "component_mean",
        "component_interval",
        "combined_mean",
        "combined_interval",
    }
    audit = verify_posterior_draw_bundle(summary.index_path)
    assert audit["n_cells"] == 3
    assert audit["materialized_n_cells"] == 2


def test_incremental_draw_writer_names_prior_predictive_uncertainty_truthfully(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=2,
        block_size=2,
        seed=1,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": np.zeros((1, 1))},
        state_metadata={
            "model": "geopfa_probabilistic_prior_predictive",
            "component_roles": {
                "heat": "evidence_coefficient_prior_predictive"
            },
        },
    )
    evidence = np.array([[[0.0]], [[1.0]]])
    writer.write_block(
        0,
        component_probability=1.0 / (1.0 + np.exp(-evidence)),
        prior_logit=np.zeros((1, 1)),
        evidence_logit=evidence,
        spatial_logit=np.zeros_like(evidence),
    )

    index_path = writer.finalize(ci_level=0.9).index_path
    index = json.loads(index_path.read_text())

    assert index["uncertainty_semantics"] == (
        "paired_prior_predictive_probability_draws"
    )
    assert (
        verify_posterior_draw_bundle(index_path)["uncertainty_semantics"]
        == "paired_prior_predictive_probability_draws"
    )


def test_incremental_draw_writer_rejects_replicated_fixed_probabilities(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )

    with pytest.raises(ValueError, match="fixed prior"):
        PosteriorDrawBlockWriter(
            grid,
            tmp_path,
            component_names=("heat",),
            n_draws=2,
            block_size=2,
            seed=1,
            combination_rule="product",
            scope="baseline",
            state_arrays={"prior_logit": np.zeros((1, 1))},
            state_metadata={
                "model": "geopfa_probabilistic_prior_predictive",
                "component_roles": {"heat": "fixed_prior_predictive"},
            },
        )


def test_incremental_draw_verifier_rejects_false_uncertainty_semantics(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=1,
        block_size=1,
        seed=1,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": np.zeros((1, 1))},
        state_metadata=_posterior_metadata(("heat",)),
    )
    writer.write_block(
        0,
        component_probability=np.full((1, 1, 1), 0.5),
        prior_logit=np.zeros((1, 1)),
        evidence_logit=np.zeros((1, 1, 1)),
        spatial_logit=np.zeros((1, 1, 1)),
    )
    index_path = writer.finalize(ci_level=0.9).index_path
    index = json.loads(index_path.read_text())
    index["uncertainty_semantics"] = (
        "paired_prior_predictive_probability_draws"
    )
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="uncertainty semantics"):
        verify_posterior_draw_bundle(index_path)


def test_incremental_draw_verifier_rejects_record_path_traversal(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=1,
        block_size=1,
        seed=1,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": np.zeros((1, 1))},
        state_metadata=_posterior_metadata(("heat",)),
    )
    writer.write_block(
        0,
        component_probability=np.full((1, 1, 1), 0.5),
        prior_logit=np.zeros((1, 1)),
        evidence_logit=np.zeros((1, 1, 1)),
        spatial_logit=np.zeros((1, 1, 1)),
    )
    index_path = writer.finalize(ci_level=0.9).index_path
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((1, 2)), allow_pickle=False)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["coordinates"] = {
        "path": "../outside.npy",
        "size_bytes": outside.stat().st_size,
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="inside its posterior namespace"):
        verify_posterior_draw_bundle(index_path)


def test_incremental_posterior_writer_resumes_only_identical_state(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0], [2.0, 3.0]),
        crs="EPSG:32610",
    )
    kwargs = {
        "component_names": ("heat",),
        "n_draws": 3,
        "block_size": 2,
        "seed": 9,
        "combination_rule": "product",
        "scope": "baseline",
        "state_arrays": {"coefficient_draws": np.arange(3.0)[:, None]},
        "state_metadata": _posterior_metadata(("heat",), model="test"),
    }
    first = PosteriorDrawBlockWriter(grid, tmp_path, **kwargs)
    probabilities = np.full((2, 2, 1), 0.6)
    zero = np.zeros_like(probabilities)
    first.write_block(
        0,
        component_probability=probabilities,
        prior_logit=np.full((2, 1), np.log(0.6 / 0.4)),
        evidence_logit=zero,
        spatial_logit=zero,
    )

    resumed = PosteriorDrawBlockWriter(grid, tmp_path, **kwargs)
    assert resumed.completed_draw_ranges == ((0, 2),)
    resumed.write_block(
        2,
        component_probability=np.full((1, 2, 1), 0.7),
        prior_logit=np.full((2, 1), np.log(0.7 / 0.3)),
        evidence_logit=np.zeros((1, 2, 1)),
        spatial_logit=np.zeros((1, 2, 1)),
    )
    assert resumed.finalize(ci_level=0.8).index_path.is_file()

    changed = dict(kwargs)
    changed["state_arrays"] = {
        "coefficient_draws": np.arange(3.0)[:, None] + 1.0
    }
    with pytest.raises(ValueError, match="posterior state fingerprint"):
        PosteriorDrawBlockWriter(grid, tmp_path, **changed)


def test_incremental_posterior_writer_rejects_false_logit_decomposition(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=1,
        block_size=1,
        seed=1,
        combination_rule="product",
        scope="baseline",
        state_arrays={"coefficient_draws": np.zeros((1, 1))},
        state_metadata=_posterior_metadata(("heat",)),
    )
    with pytest.raises(ValueError, match="logit decomposition"):
        writer.write_block(
            0,
            component_probability=np.full((1, 1, 1), 0.9),
            prior_logit=np.zeros((1, 1)),
            evidence_logit=np.zeros((1, 1, 1)),
            spatial_logit=np.zeros((1, 1, 1)),
        )


def test_incremental_posterior_writer_rejects_prior_state_substitution(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=1,
        block_size=1,
        seed=1,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": np.zeros((1, 1))},
        state_metadata=_posterior_metadata(("heat",)),
    )
    substituted_prior = np.ones((1, 1))
    with pytest.raises(ValueError, match="persisted prior_logit state"):
        writer.write_block(
            0,
            component_probability=1.0
            / (1.0 + np.exp(-substituted_prior))[None],
            prior_logit=substituted_prior,
            evidence_logit=np.zeros((1, 1, 1)),
            spatial_logit=np.zeros((1, 1, 1)),
        )


def test_incremental_posterior_writer_detects_corrupt_resume_block(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    kwargs = {
        "component_names": ("heat",),
        "n_draws": 2,
        "block_size": 1,
        "seed": 1,
        "combination_rule": "product",
        "scope": "baseline",
        "state_arrays": {"prior_logit": np.zeros((1, 1))},
        "state_metadata": _posterior_metadata(("heat",)),
    }
    writer = PosteriorDrawBlockWriter(grid, tmp_path, **kwargs)
    writer.write_block(
        0,
        component_probability=np.full((1, 1, 1), 0.5),
        prior_logit=np.zeros((1, 1)),
        evidence_logit=np.zeros((1, 1, 1)),
        spatial_logit=np.zeros((1, 1, 1)),
    )
    block_path = next(
        (tmp_path / ".posterior_draws.incomplete/blocks").iterdir()
    )
    block_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="payload (size|hash) mismatch"):
        PosteriorDrawBlockWriter(grid, tmp_path, **kwargs)


def test_load_posterior_draw_state_reopens_incomplete_exact_state(
    tmp_path: Path,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0, 1.0], [2.0, 3.0]),
        crs="EPSG:32610",
    )
    prior = np.zeros((2, 1))
    coefficients = np.arange(6.0).reshape(3, 2)
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=3,
        block_size=2,
        seed=3,
        combination_rule="product",
        scope="baseline",
        state_arrays={
            "prior_logit": prior,
            "field_coefficient_draws": coefficients,
        },
        state_metadata=_posterior_metadata(
            ("heat",),
            config_hash="abc",
            analysis_input_sha256="1" * 64,
            implementation_sha256="2" * 64,
            model="test",
        ),
    )
    writer.write_block(
        0,
        component_probability=np.full((2, 2, 1), 0.5),
        prior_logit=prior,
        evidence_logit=np.zeros((2, 2, 1)),
        spatial_logit=np.zeros((2, 2, 1)),
    )

    restored = load_posterior_draw_state(
        tmp_path,
        grid,
        expected_config_hash="abc",
        expected_analysis_input_sha256="1" * 64,
        expected_implementation_sha256="2" * 64,
        expected_scope="baseline",
    )

    assert restored is not None
    assert restored.complete is False
    assert restored.completed_draw_ranges == ((0, 2),)
    assert restored.component_names == ("heat",)
    np.testing.assert_array_equal(
        restored.arrays["field_coefficient_draws"], coefficients
    )
    with pytest.raises(ValueError, match="config hash"):
        load_posterior_draw_state(
            tmp_path,
            grid,
            expected_config_hash="changed",
            expected_analysis_input_sha256="1" * 64,
            expected_implementation_sha256="2" * 64,
            expected_scope="baseline",
        )
    with pytest.raises(ValueError, match="analysis inputs"):
        load_posterior_draw_state(
            tmp_path,
            grid,
            expected_config_hash="abc",
            expected_analysis_input_sha256="2" * 64,
            expected_implementation_sha256="2" * 64,
            expected_scope="baseline",
        )
    with pytest.raises(ValueError, match="implementation"):
        load_posterior_draw_state(
            tmp_path,
            grid,
            expected_config_hash="abc",
            expected_analysis_input_sha256="1" * 64,
            expected_implementation_sha256="3" * 64,
            expected_scope="baseline",
        )

    restored.close()
    assert all(
        array._mmap.closed  # type: ignore[attr-defined]  # noqa: SLF001
        for array in restored.arrays.values()
    )


def test_posterior_summary_closes_memmap_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([0.0], [1.0]), crs="EPSG:32610"
    )
    writer = PosteriorDrawBlockWriter(
        grid,
        tmp_path,
        component_names=("heat",),
        n_draws=1,
        block_size=1,
        seed=3,
        combination_rule="product",
        scope="baseline",
        state_arrays={"prior_logit": np.zeros((1, 1))},
        state_metadata=_posterior_metadata(("heat",), model="test"),
    )
    writer.write_block(
        0,
        component_probability=np.full((1, 1, 1), 0.5),
        prior_logit=np.zeros((1, 1)),
        evidence_logit=np.zeros((1, 1, 1)),
        spatial_logit=np.zeros((1, 1, 1)),
    )
    real_memmap = np.memmap
    mappings: list[np.memmap] = []

    def tracked_memmap(*args, **kwargs):
        array = real_memmap(*args, **kwargs)
        mappings.append(array)
        return array

    real_unlink = Path.unlink

    def reject_open_mapping(path: Path, *args, **kwargs):
        if path.name.startswith(".posterior-summary-") and any(
            not array._mmap.closed  # type: ignore[attr-defined]  # noqa: SLF001
            for array in mappings
        ):
            raise PermissionError("cannot unlink an open Windows mapping")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr("geopfa.prob.io.np.memmap", tracked_memmap)
    monkeypatch.setattr(Path, "unlink", reject_open_mapping)

    summary = writer.finalize(ci_level=0.9)

    assert summary.index_path.is_file()
    assert mappings
    assert all(
        array._mmap.closed  # type: ignore[attr-defined]  # noqa: SLF001
        for array in mappings
    )


# ---------------------------------------------------------------------------
# End-to-end via run_probabilistic
# ---------------------------------------------------------------------------


def test_run_probabilistic_writes_geotiff(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=2)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", formats=("csv", "geotiff"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    out_dir = tmp_path / "out"
    assert (out_dir / "component_a_probability.tif").exists()
    assert (out_dir / "component_b_probability.tif").exists()
    assert (out_dir / "combined_probability.tif").exists()


def test_run_probabilistic_writes_manifest(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=12, seed=3)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _2d_cfg(wells_path, tmp_path / "out", formats=("csv",))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, cfg)
    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert len(data["files"]) > 0
    assert data["config_hash"]


# ---------------------------------------------------------------------------
# GBLK backend output parity
# ---------------------------------------------------------------------------


def _gblk_cfg(
    wells_path: Path, output_dir: Path, formats: tuple[str, ...]
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
                scalar_fallback_pr0=0.50,
            ),
        },
        evidence=EvidenceConfig(),
        spatial_field=SpatialFieldConfig(enabled=False),
        inference=InferenceConfig(backend="gblk"),
        calibration=CalibrationConfig(method="none"),
        cross_validation=CrossValidationConfig(),
        combination=CombinationConfig(rule="product"),
        scenarios=(),
        outputs=OutputsConfig(
            probability_rasters="geotiff" in formats,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=formats,
        ),
    )


def test_gblk_run_probabilistic_writes_geotiff(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=8, n_wells=20, seed=4)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _gblk_cfg(wells_path, tmp_path / "out", formats=("geotiff",))
    run_probabilistic(fixture.pfa, cfg)
    out_dir = tmp_path / "out"
    assert (out_dir / "component_a_probability.tif").exists()
    assert (out_dir / "component_b_probability.tif").exists()
    assert (out_dir / "combined_probability.tif").exists()


def test_gblk_run_probabilistic_writes_csv(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=5)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _gblk_cfg(wells_path, tmp_path / "out", formats=("csv",))
    run_probabilistic(fixture.pfa, cfg)
    out_dir = tmp_path / "out"
    assert (out_dir / "component_a_probability.csv").exists()
    assert (out_dir / "component_b_probability.csv").exists()
    assert (out_dir / "combined_probability.csv").exists()


def test_gblk_run_probabilistic_writes_manifest(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=6)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _gblk_cfg(wells_path, tmp_path / "out", formats=("csv",))
    run_probabilistic(fixture.pfa, cfg)
    out_dir = tmp_path / "out"
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert len(data["files"]) > 0
    assert data["config_hash"]


def test_gblk_output_schema_matches_sequential(tmp_path: Path) -> None:
    """GBLK and sequential backends produce CSV files with the same core columns."""
    fixture = make_synthetic_pfa(grid_n=6, n_wells=20, seed=7)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")

    seq_cfg = _2d_cfg(wells_path, tmp_path / "seq_out", formats=("csv",))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_probabilistic(fixture.pfa, seq_cfg)

    gblk_cfg = _gblk_cfg(wells_path, tmp_path / "gblk_out", formats=("csv",))
    run_probabilistic(fixture.pfa, gblk_cfg)

    seq_csv = tmp_path / "seq_out" / "component_a_probability.csv"
    gblk_csv = tmp_path / "gblk_out" / "component_a_probability.csv"
    assert seq_csv.exists()
    assert gblk_csv.exists()

    seq_df = pd.read_csv(seq_csv)
    gblk_df = pd.read_csv(gblk_csv)
    required_cols = {"probability", "x", "y"}
    assert required_cols.issubset(set(seq_df.columns))
    assert required_cols.issubset(set(gblk_df.columns))
    assert len(gblk_df) == len(seq_df)
