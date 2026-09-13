"""Tests for the GBLK label/offset/evidence assembler."""

from __future__ import annotations

import numpy as np
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from geopfa.prob.alpha import AlphaCResult
from geopfa.prob.config import EvidenceConfig, LabelsConfig
from geopfa.prob.gblk_assemble import (
    AssembledInputs,
    assemble_gblk_inputs as _assemble_gblk_inputs,
    build_component_grid_evidence,
)
from geopfa.prob.labels import LoadedLabels
from geopfa.prob.pfa_grid import PFAGridAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assemble_gblk_inputs(
    adapter: PFAGridAdapter,
    loaded: LoadedLabels,
    alpha: dict[str, AlphaCResult],
    *,
    evidence_config: EvidenceConfig | None = None,
) -> AssembledInputs:
    """Test helper that makes the evidence contract explicit."""
    return _assemble_gblk_inputs(
        adapter,
        loaded,
        alpha,
        evidence_config=(
            EvidenceConfig() if evidence_config is None else evidence_config
        ),
    )


def _make_grid_gdf(
    n: int = 10, *, crs: str = "EPSG:32611"
) -> gpd.GeoDataFrame:
    xs = np.linspace(500_000.0, 600_000.0, n)
    ys = np.linspace(4_300_000.0, 4_400_000.0, n)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    return gpd.GeoDataFrame(
        {"geometry": [Point(x, y) for x, y in coords]},
        crs=crs,
    )


def _make_pfa(
    components: list[str],
    grid_gdf: gpd.GeoDataFrame,
    n_layers: int = 2,
    *,
    rng: np.random.Generator,
) -> dict:
    n = len(grid_gdf)

    def _layer(vals: np.ndarray) -> dict:
        df = grid_gdf.copy()
        df["value_interpolated"] = vals
        return {
            "model": df,
            "model_data_col": "value_interpolated",
        }

    comp_dicts = {}
    for comp in components:
        layers = {
            f"layer_{i}_{comp}": _layer(rng.uniform(0.0, 1.0, size=n))
            for i in range(n_layers)
        }
        comp_dicts[comp] = {
            "pr0": 0.5,
            "pr_norm": grid_gdf.copy(),
            "layers": layers,
        }

    return {
        "criteria": {
            "geologic": {
                "components": comp_dicts,
            }
        }
    }


def _make_wells_gdf(
    n: int,
    components: list[str],
    *,
    rng: np.random.Generator,
    unlabeled_count: int = 5,
    crs: str = "EPSG:32611",
) -> gpd.GeoDataFrame:
    xs = rng.uniform(500_000.0, 600_000.0, size=n)
    ys = rng.uniform(4_300_000.0, 4_400_000.0, size=n)
    data: dict = {
        "well_id": [f"W{i:04d}" for i in range(n)],
        "geometry": [Point(x, y) for x, y in zip(xs, ys)],
    }
    for comp in components:
        col = f"{comp}_label"
        labels = rng.integers(0, 2, size=n).astype(float)
        labels[:unlabeled_count] = np.nan
        data[col] = labels
    return gpd.GeoDataFrame(data, crs=crs)


def _make_alpha_results(
    components: list[str],
    grid_gdf: gpd.GeoDataFrame,
    *,
    rng: np.random.Generator,
) -> dict[str, AlphaCResult]:
    G = len(grid_gdf)
    results = {}
    for comp in components:
        offset = rng.uniform(-1.0, 1.0, size=G).astype(np.float64)
        results[comp] = AlphaCResult(
            grid_offset=offset,
            scalar_fallback=float(np.mean(offset)),
        )
    return results


def _make_labels_config(components: list[str]) -> LabelsConfig:
    return LabelsConfig(
        source="/synthetic/wells.gpkg",
        id_col="well_id",
        label_columns={comp: f"{comp}_label" for comp in components},
    )


def _assemble(
    components: list[str] | None = None,
    *,
    n_wells: int = 30,
    grid_n: int = 8,
    n_layers: int = 2,
    unlabeled_count: int = 5,
    seed: int = 0,
    evidence_config: EvidenceConfig | None = None,
) -> tuple[AssembledInputs, dict]:
    if components is None:
        components = ["comp_a", "comp_b", "comp_c"]
    rng = np.random.default_rng(seed)
    grid_gdf = _make_grid_gdf(grid_n)
    pfa = _make_pfa(components, grid_gdf, n_layers=n_layers, rng=rng)
    wells_gdf = _make_wells_gdf(
        n_wells, components, rng=rng, unlabeled_count=unlabeled_count
    )
    alpha = _make_alpha_results(components, grid_gdf, rng=rng)
    cfg = _make_labels_config(components)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    result = assemble_gblk_inputs(
        adapter,
        loaded,
        alpha,
        evidence_config=EvidenceConfig()
        if evidence_config is None
        else evidence_config,
    )
    return result, {
        "components": components,
        "n_wells": n_wells,
        "G": grid_n * grid_n,
        "n_layers": n_layers,
        "unlabeled_count": unlabeled_count,
        "alpha": alpha,
        "wells_gdf": wells_gdf,
        "grid_gdf": grid_gdf,
    }


def test_p_gblk_assemble_honors_evidence_layer_allowlist() -> None:
    result, _ = _assemble(
        components=["comp_a"],
        n_layers=2,
        evidence_config=EvidenceConfig(include_layers=("layer_0_comp_a",)),
    )

    assert result.layer_names == {"comp_a": ["layer_0_comp_a"]}
    assert result.evidence["comp_a"].shape[1] == 1


def test_p_gblk_assemble_applies_configured_nonaffine_transformation() -> None:
    rng = np.random.default_rng(123)
    grid = _make_grid_gdf(4)
    pfa = _make_pfa(["comp_a"], grid, n_layers=1, rng=rng)
    layer = pfa["criteria"]["geologic"]["components"]["comp_a"]["layers"][
        "layer_0_comp_a"
    ]
    layer["transformation_method"] = "valley"
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    alpha = AlphaCResult(
        grid_offset=np.zeros(len(grid)),
        scalar_fallback=0.0,
    )

    evidence, names = build_component_grid_evidence(
        adapter,
        "comp_a",
        alpha,
        EvidenceConfig(),
    )

    raw = layer["model"]["value_interpolated"].to_numpy(dtype=float)
    median = np.nanmedian(raw)
    mad = np.nanmedian(np.abs(raw - median))
    expected = 1.0 - np.exp(-np.square(raw - median) / (2.0 * mad**2))
    assert names == ["layer_0_comp_a"]
    np.testing.assert_allclose(evidence[:, 0], expected)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_y_shape():
    result, meta = _assemble()
    Q = len(meta["components"])
    n = meta["n_wells"]
    assert result.y.shape == (n, Q)


def test_p_gblk_assemble_labeled_mask_shape():
    result, meta = _assemble()
    assert result.labeled_mask.shape == (meta["n_wells"],)
    assert result.labeled_mask.dtype == bool


def test_p_gblk_assemble_preserves_componentwise_missingness():
    """A label observed for one component must not create a false zero for another."""
    rng = np.random.default_rng(101)
    components = ["comp_a", "comp_b"]
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(6, components, rng=rng, unlabeled_count=0)
    wells_gdf.loc[0, "comp_a_label"] = np.nan
    wells_gdf.loc[1, "comp_b_label"] = np.nan
    alpha = _make_alpha_results(components, grid_gdf, rng=rng)
    loaded = LoadedLabels(
        gdf=wells_gdf, config=_make_labels_config(components)
    )
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(adapter, loaded, alpha)

    assert result.observed_mask.shape == result.y.shape
    assert not result.observed_mask[0, 0]
    assert result.observed_mask[0, 1]
    assert result.observed_mask[1, 0]
    assert not result.observed_mask[1, 1]
    assert not result.labeled_mask[0]
    assert not result.labeled_mask[1]


def test_p_gblk_assemble_rejects_nonnumeric_observed_label():
    rng = np.random.default_rng(104)
    components = ["comp_a"]
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(8, components, rng=rng, unlabeled_count=0)
    wells_gdf.loc[0, "comp_a_label"] = "not-a-label"
    loaded = LoadedLabels(
        gdf=wells_gdf, config=_make_labels_config(components)
    )
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    with pytest.raises(ValueError, match="nonnumeric"):
        assemble_gblk_inputs(
            adapter,
            loaded,
            _make_alpha_results(components, grid_gdf, rng=rng),
        )


def test_p_gblk_assemble_rejects_alpha_component_without_label_contract():
    rng = np.random.default_rng(102)
    components = ["observed", "prior_only"]
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(8, components, rng=rng, unlabeled_count=0)
    alpha = _make_alpha_results(components, grid_gdf, rng=rng)
    labels_cfg = LabelsConfig(
        source="/synthetic/wells.gpkg",
        id_col="well_id",
        label_columns={"observed": "observed_label"},
    )
    loaded = LoadedLabels(gdf=wells_gdf, config=labels_cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    with pytest.raises(ValueError, match="prior_only"):
        assemble_gblk_inputs(adapter, loaded, alpha)


def test_p_gblk_assemble_well_offsets_shape():
    result, meta = _assemble()
    Q = len(meta["components"])
    n = meta["n_wells"]
    assert result.well_offsets.shape == (n, Q)


def test_p_gblk_assemble_grid_offsets_shape():
    result, meta = _assemble()
    Q = len(meta["components"])
    G = meta["G"]
    assert result.grid_offsets.shape == (G, Q)


def test_p_gblk_assemble_well_coords_shape():
    result, meta = _assemble()
    assert result.well_coords.shape == (meta["n_wells"], 2)


def test_p_gblk_assemble_grid_coords_shape():
    result, meta = _assemble()
    assert result.grid_coords.shape == (meta["G"], 2)


def test_p_gblk_assemble_evidence_shape():
    components = ["comp_a", "comp_b"]
    n_layers = 3
    result, meta = _assemble(components=components, n_layers=n_layers)
    n = meta["n_wells"]
    for comp in components:
        assert result.evidence[comp].shape == (n, n_layers), (
            f"evidence[{comp!r}] shape mismatch"
        )
        assert result.grid_evidence[comp].shape == (meta["G"], n_layers)


def test_p_gblk_assemble_convenience_properties():
    result, meta = _assemble()
    assert result.n == meta["n_wells"]
    assert result.n_components == len(meta["components"])
    assert result.n_grid == meta["G"]


# ---------------------------------------------------------------------------
# Component order tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_component_order_sorted():
    components = ["comp_c", "comp_a", "comp_b"]
    result, _ = _assemble(components=components)
    assert result.component_names == ("comp_a", "comp_b", "comp_c")


def test_p_gblk_assemble_component_order_stable_repeated_calls():
    components = ["z_comp", "a_comp", "m_comp"]
    r1, _ = _assemble(components=components, seed=0)
    r2, _ = _assemble(components=components, seed=1)
    assert r1.component_names == r2.component_names


# ---------------------------------------------------------------------------
# Labeled-mask tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_unlabeled_wells_get_false_mask():
    unlabeled = 7
    result, meta = _assemble(unlabeled_count=unlabeled)
    assert not result.labeled_mask[:unlabeled].any(), (
        "first unlabeled_count wells should all have mask=False"
    )


def test_p_gblk_assemble_labeled_wells_get_true_mask():
    unlabeled = 5
    n_wells = 30
    result, _ = _assemble(n_wells=n_wells, unlabeled_count=unlabeled)
    assert result.labeled_mask[unlabeled:].all(), (
        "wells with finite labels should have mask=True"
    )


def test_p_gblk_assemble_all_unlabeled_gives_false_mask():
    rng = np.random.default_rng(42)
    components = ["c1", "c2"]
    grid_n = 5
    grid_gdf = _make_grid_gdf(grid_n)
    n_wells = 8
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    xs = rng.uniform(500_000.0, 600_000.0, size=n_wells)
    ys = rng.uniform(4_300_000.0, 4_400_000.0, size=n_wells)
    wells_gdf = gpd.GeoDataFrame(
        {
            "well_id": [f"W{i}" for i in range(n_wells)],
            "geometry": [Point(x, y) for x, y in zip(xs, ys)],
            "c1_label": np.full(n_wells, np.nan),
            "c2_label": np.full(n_wells, np.nan),
        },
        crs="EPSG:32611",
    )
    alpha = _make_alpha_results(components, grid_gdf, rng=rng)
    cfg = _make_labels_config(components)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    result = assemble_gblk_inputs(adapter, loaded, alpha)
    assert not result.labeled_mask.any()


# ---------------------------------------------------------------------------
# Offset correctness tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_grid_offsets_equal_alpha_logits():
    components = ["comp_a", "comp_b"]
    rng = np.random.default_rng(7)
    grid_gdf = _make_grid_gdf(6)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    alpha = _make_alpha_results(components, grid_gdf, rng=rng)
    cfg = _make_labels_config(components)
    wells_gdf = _make_wells_gdf(20, components, rng=rng)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    result = assemble_gblk_inputs(adapter, loaded, alpha)

    sorted_comps = sorted(components)
    for q, comp in enumerate(sorted_comps):
        expected = alpha[comp].grid_offset
        np.testing.assert_array_equal(
            result.grid_offsets[:, q],
            expected,
            err_msg=f"grid_offsets column {q} ({comp}) != alpha logit",
        )


def test_p_gblk_assemble_retains_prior_event_probabilities_at_grid_and_wells():
    result, meta = _assemble(components=["comp_a"], seed=18)

    expected_grid = 1.0 / (1.0 + np.exp(-meta["alpha"]["comp_a"].grid_offset))
    np.testing.assert_allclose(
        result.prior_probability_grid[:, 0], expected_grid
    )
    assert result.prior_probability_well.shape == (meta["n_wells"], 1)
    assert np.all(
        (result.prior_probability_well >= 0.0)
        & (result.prior_probability_well <= 1.0)
    )


def test_p_gblk_assemble_retains_prior_response_moments():
    components = ["comp_a"]
    rng = np.random.default_rng(181)
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    latent_mean = np.linspace(100.0, 200.0, len(grid_gdf))
    latent_sd = np.linspace(20.0, 30.0, len(grid_gdf))
    alpha = {
        "comp_a": AlphaCResult(
            grid_offset=np.zeros(len(grid_gdf)),
            scalar_fallback=0.0,
            latent_mean=latent_mean,
            latent_sd=latent_sd,
            event_threshold=150.0,
        )
    }
    cfg = _make_labels_config(components)
    wells_gdf = _make_wells_gdf(8, components, rng=rng)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(adapter, loaded, alpha)

    np.testing.assert_allclose(
        result.prior_response_mean_grid[:, 0], latent_mean
    )
    np.testing.assert_allclose(result.prior_response_sd_grid[:, 0], latent_sd)
    assert result.prior_response_mean_well.shape == (8, 1)
    assert result.prior_response_sd_well.shape == (8, 1)


def test_p_gblk_assemble_well_offsets_are_snapped_from_grid():
    rng = np.random.default_rng(3)
    components = ["c1"]
    grid_gdf = _make_grid_gdf(6)
    G = len(grid_gdf)

    constant_offset = 1.23
    alpha = {
        "c1": AlphaCResult(
            grid_offset=np.full(G, constant_offset),
            scalar_fallback=constant_offset,
        )
    }
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(15, components, rng=rng)
    cfg = _make_labels_config(components)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    result = assemble_gblk_inputs(adapter, loaded, alpha)

    np.testing.assert_allclose(
        result.well_offsets[:, 0],
        constant_offset,
        err_msg="well offsets should equal the (uniform) alpha logit",
    )


def test_p_gblk_assemble_reprojects_wells_to_prediction_grid_crs():
    """Label coordinates and snapped evidence must share the grid CRS."""
    rng = np.random.default_rng(31)
    components = ["c1"]
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, n_layers=1, rng=rng)
    layer = pfa["criteria"]["geologic"]["components"]["c1"]["layers"][
        "layer_0_c1"
    ]["model"]
    layer["value_interpolated"] = np.arange(len(layer), dtype=float)

    selected = np.array([0, 5, 10, 15])
    wells_projected = gpd.GeoDataFrame(
        {
            "well_id": [f"W{i}" for i in selected],
            "c1_label": [0.0, 1.0, 0.0, 1.0],
            "geometry": grid_gdf.geometry.iloc[selected].to_list(),
        },
        crs=grid_gdf.crs,
    )
    wells_geographic = wells_projected.to_crs("EPSG:4326")
    alpha = {
        "c1": AlphaCResult(
            grid_offset=np.arange(len(grid_gdf), dtype=float),
            scalar_fallback=0.0,
        )
    }
    loaded = LoadedLabels(
        gdf=wells_geographic,
        config=_make_labels_config(components),
    )
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(adapter, loaded, alpha)

    np.testing.assert_allclose(
        result.well_coords,
        np.column_stack(
            [wells_projected.geometry.x, wells_projected.geometry.y]
        ),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(result.well_offsets[:, 0], selected)
    np.testing.assert_array_equal(result.evidence["c1"][:, 0], selected)


def test_p_gblk_assemble_rejects_one_sided_missing_crs():
    rng = np.random.default_rng(32)
    components = ["c1"]
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(components, grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(8, components, rng=rng)
    wells_gdf = wells_gdf.set_crs(None, allow_override=True)
    loaded = LoadedLabels(
        gdf=wells_gdf,
        config=_make_labels_config(components),
    )
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    with pytest.raises(
        ValueError,
        match="point and prediction-grid geometries must both declare a CRS",
    ):
        assemble_gblk_inputs(
            adapter,
            loaded,
            _make_alpha_results(components, grid_gdf, rng=rng),
        )


# ---------------------------------------------------------------------------
# Evidence / layer name tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_layer_names_match_evidence_columns():
    components = ["ca", "cb"]
    n_layers = 3
    result, meta = _assemble(components=components, n_layers=n_layers)
    for comp in components:
        assert len(result.layer_names[comp]) == n_layers
        assert result.evidence[comp].shape[1] == n_layers


def test_p_gblk_assemble_excluded_layers_absent_from_evidence():
    rng = np.random.default_rng(9)
    components = ["comp_a"]
    grid_gdf = _make_grid_gdf(5)
    G = len(grid_gdf)
    pfa = _make_pfa(components, grid_gdf, n_layers=2, rng=rng)

    excluded = "layer_0_comp_a"
    alpha = {
        "comp_a": AlphaCResult(
            grid_offset=rng.uniform(-1.0, 1.0, size=G),
            scalar_fallback=0.0,
            excluded_layer_names={excluded},
        )
    }
    wells_gdf = _make_wells_gdf(10, components, rng=rng)
    cfg = _make_labels_config(components)
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")
    result = assemble_gblk_inputs(adapter, loaded, alpha)

    assert excluded not in result.layer_names["comp_a"]
    assert result.evidence["comp_a"].shape[1] == 1


def test_p_gblk_assemble_linearly_resamples_rectilinear_grid_evidence():
    """Offset regular evidence grids are sampled on the canonical support."""
    rng = np.random.default_rng(91)
    components = ["comp_a"]
    reference = _make_grid_gdf(4)
    pfa = _make_pfa(components, reference, n_layers=1, rng=rng)
    source_axis_x = np.linspace(490_000.0, 610_000.0, 4)
    source_axis_y = np.linspace(4_290_000.0, 4_410_000.0, 4)
    source_x, source_y = np.meshgrid(source_axis_x, source_axis_y)
    source = gpd.GeoDataFrame(
        {
            "value_interpolated": (
                2.0 * source_x.ravel() + 3.0 * source_y.ravel()
            ),
            "geometry": [
                Point(x, y) for x, y in zip(source_x.ravel(), source_y.ravel())
            ],
        },
        crs=reference.crs,
    ).sample(frac=1.0, random_state=7)
    pfa["criteria"]["geologic"]["components"]["comp_a"]["layers"][
        "layer_0_comp_a"
    ]["model"] = source
    wells = _make_wells_gdf(10, components, rng=rng, unlabeled_count=0)
    loaded = LoadedLabels(gdf=wells, config=_make_labels_config(components))
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(
        adapter,
        loaded,
        _make_alpha_results(components, reference, rng=rng),
    )

    expected = (
        2.0 * reference.geometry.x.to_numpy()
        + 3.0 * reference.geometry.y.to_numpy()
    )
    np.testing.assert_allclose(
        result.grid_evidence["comp_a"][:, 0],
        expected,
        rtol=1e-12,
        atol=1e-8,
    )


def test_p_gblk_assemble_tolerates_sub_per_mille_boundary_roundoff():
    rng = np.random.default_rng(93)
    components = ["comp_a"]
    reference = _make_grid_gdf(4)
    spacing = float(np.diff(np.unique(reference.geometry.x)).min())
    shift = 0.0005 * spacing
    source = reference.copy()
    source.geometry = source.translate(xoff=shift, yoff=shift)
    source["value_interpolated"] = (
        2.0 * source.geometry.x.to_numpy() + 3.0 * source.geometry.y.to_numpy()
    )
    pfa = _make_pfa(components, reference, n_layers=1, rng=rng)
    pfa["criteria"]["geologic"]["components"]["comp_a"]["layers"][
        "layer_0_comp_a"
    ]["model"] = source
    wells = _make_wells_gdf(10, components, rng=rng, unlabeled_count=0)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(
        adapter,
        LoadedLabels(gdf=wells, config=_make_labels_config(components)),
        _make_alpha_results(components, reference, rng=rng),
    )

    expected_x = np.maximum(
        reference.geometry.x.to_numpy(), source.geometry.x.min()
    )
    expected_y = np.maximum(
        reference.geometry.y.to_numpy(), source.geometry.y.min()
    )
    np.testing.assert_allclose(
        result.grid_evidence["comp_a"][:, 0],
        2.0 * expected_x + 3.0 * expected_y,
        rtol=1e-12,
        atol=1e-8,
    )


def test_p_gblk_assemble_resamples_component_offsets_to_reference_grid():
    """Component grids with serialization drift share one canonical support."""
    rng = np.random.default_rng(94)
    components = ["comp_a", "comp_b"]
    reference = _make_grid_gdf(4)
    spacing = float(np.diff(np.unique(reference.geometry.x)).min())
    shift = 0.0005 * spacing
    shifted = reference.copy()
    shifted.geometry = shifted.translate(xoff=shift, yoff=shift)
    pfa = _make_pfa(components, reference, n_layers=1, rng=rng)
    comp_b = pfa["criteria"]["geologic"]["components"]["comp_b"]
    comp_b["pr_norm"] = shifted.copy()
    comp_b["layers"]["layer_0_comp_b"]["model"] = shifted.assign(
        value_interpolated=(
            2.0 * shifted.geometry.x.to_numpy()
            + 3.0 * shifted.geometry.y.to_numpy()
        )
    )
    alpha = _make_alpha_results(components, reference, rng=rng)
    shifted_offset = (
        shifted.geometry.x.to_numpy() + shifted.geometry.y.to_numpy()
    )
    alpha["comp_b"] = AlphaCResult(
        grid_offset=shifted_offset,
        scalar_fallback=0.0,
    )
    wells = _make_wells_gdf(10, components, rng=rng, unlabeled_count=0)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    result = assemble_gblk_inputs(
        adapter,
        LoadedLabels(gdf=wells, config=_make_labels_config(components)),
        alpha,
    )

    expected_x = np.maximum(
        reference.geometry.x.to_numpy(), shifted.geometry.x.min()
    )
    expected_y = np.maximum(
        reference.geometry.y.to_numpy(), shifted.geometry.y.min()
    )
    np.testing.assert_allclose(
        result.grid_offsets[:, 1],
        expected_x + expected_y,
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        result.grid_evidence["comp_b"][:, 0],
        2.0 * expected_x + 3.0 * expected_y,
        rtol=1e-12,
        atol=1e-8,
    )


def test_p_gblk_assemble_rejects_evidence_grid_without_full_coverage():
    rng = np.random.default_rng(92)
    components = ["comp_a"]
    reference = _make_grid_gdf(4)
    pfa = _make_pfa(components, reference, n_layers=1, rng=rng)
    source = reference.copy()
    source.geometry = source.translate(xoff=20_000.0, yoff=20_000.0)
    source["value_interpolated"] = np.arange(len(source), dtype=float)
    pfa["criteria"]["geologic"]["components"]["comp_a"]["layers"][
        "layer_0_comp_a"
    ]["model"] = source
    wells = _make_wells_gdf(10, components, rng=rng, unlabeled_count=0)
    loaded = LoadedLabels(gdf=wells, config=_make_labels_config(components))
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    with pytest.raises(ValueError, match="outside the grid cell footprint"):
        assemble_gblk_inputs(
            adapter,
            loaded,
            _make_alpha_results(components, reference, rng=rng),
        )


# ---------------------------------------------------------------------------
# Error tests
# ---------------------------------------------------------------------------


def test_p_gblk_assemble_raises_when_declared_component_has_no_alpha():
    rng = np.random.default_rng(0)
    grid_gdf = _make_grid_gdf(4)
    pfa = _make_pfa(["comp_a"], grid_gdf, rng=rng)
    wells_gdf = _make_wells_gdf(5, ["comp_a"], rng=rng)
    cfg = _make_labels_config(["comp_a"])
    loaded = LoadedLabels(gdf=wells_gdf, config=cfg)
    adapter = PFAGridAdapter(pfa, criteria="geologic", dimensions="2d")

    with pytest.raises(ValueError, match="missing alpha: comp_a"):
        assemble_gblk_inputs(adapter, loaded, {})
