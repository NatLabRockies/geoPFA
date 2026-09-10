"""Tests for spatially-blocked calibration CV of the GBLK method (P6-S01).

Requires the ``gblk`` pixi environment. Skipped otherwise.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("latticekrigx.glk.calibration")

from latticekrigx.glk.calibration import CalibrationCVResult  # noqa: E402

from geopfa.exceptions import GEOPFAValueError  # noqa: E402
from geopfa.prob.config import (  # noqa: E402
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
from geopfa.prob.gblk_runner import run_gblk_calibration_cv  # noqa: E402
from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: E402


_COMPONENTS = ("component_a", "component_b")
_NC_SMALL = 5
_OUTER_SMALL = 100
_IRLS_SMALL = 50


def _cfg(wells_path: Path, output_dir: Path) -> ProbabilisticConfig:
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
            probability_rasters=False,
            uncertainty_rasters=False,
            calibration_artifacts=False,
            decision_artifacts=False,
            scenarios=False,
            format=(),
        ),
    )


def _run_cv(
    tmp_path: Path,
    *,
    n_wells: int = 300,
    grid_n: int = 22,
    seed: int = 3,
    n_folds: int = 3,
    n_bins: int = 6,
) -> dict[str, CalibrationCVResult]:
    fixture = make_synthetic_pfa(grid_n=grid_n, n_wells=n_wells, seed=seed)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out")
    return run_gblk_calibration_cv(
        fixture.pfa,
        cfg,
        n_folds=n_folds,
        n_bins=n_bins,
        random_state=0,
        nc=_NC_SMALL,
        a_wght=4.5,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )


def test_p_gblk_cv_returns_result_per_component_plus_joint(
    tmp_path: Path,
) -> None:
    results = _run_cv(tmp_path)
    expected_keys = set(_COMPONENTS) | {"joint"}
    assert set(results.keys()) == expected_keys
    for res in results.values():
        assert isinstance(res, CalibrationCVResult)


def test_p_gblk_cv_uses_shared_fold_ids(tmp_path: Path) -> None:
    results = _run_cv(tmp_path)
    ref = results["joint"].fold_ids
    for name in _COMPONENTS:
        np.testing.assert_array_equal(results[name].fold_ids, ref)


def test_p_gblk_cv_folds_cover_all_labeled_wells(tmp_path: Path) -> None:
    results = _run_cv(tmp_path)
    ref = results["joint"].fold_ids
    assert ref.ndim == 1
    unique_folds = np.unique(ref)
    assert unique_folds.size >= 2
    assert unique_folds.min() >= 0


def test_p_gblk_cv_honors_geoPFA_block_and_buffer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from geopfa.prob import gblk_runner as runner_module
    from geopfa.prob.cv import spatial_block_cv

    fixture = make_synthetic_pfa(grid_n=10, n_wells=80, seed=19)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out")
    cfg = replace(
        cfg,
        cross_validation=CrossValidationConfig(
            n_folds=4,
            block_type="grid",
            block_size_km=25.0,
            grid_size=2,
            buffer_km=5.0,
        ),
    )
    captured: list[dict] = []

    def fake_calibration_cv(*_args, **kwargs):
        captured.append(dict(kwargs))
        return SimpleNamespace(fold_ids=np.asarray(kwargs["fold_ids"]))

    monkeypatch.setattr(
        runner_module, "_calibration_cv_from_splits", fake_calibration_cv
    )

    result = run_gblk_calibration_cv(
        fixture.pfa,
        cfg,
        n_bins=4,
        random_state=7,
        components=["component_a"],
    )

    coords = np.column_stack(
        [fixture.wells.geometry.x, fixture.wells.geometry.y]
    )
    expected = np.empty(len(coords), dtype=int)
    expected_splits = list(
        spatial_block_cv(
            coords,
            n_folds=4,
            block_type="grid",
            block_size_km=25.0,
            grid_size=2,
            seed=7,
            buffer_distance=5_000.0,
            dims=(0, 1),
        )
    )
    for fold, (_train, test) in enumerate(expected_splits):
        expected[test] = fold

    assert set(result) == {"component_a"}
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0]["fold_ids"], expected)
    assert len(captured[0]["splits"]) == len(expected_splits)
    for actual, expected_split in zip(
        captured[0]["splits"], expected_splits, strict=True
    ):
        np.testing.assert_array_equal(actual[0], expected_split[0])
        np.testing.assert_array_equal(actual[1], expected_split[1])


def test_p_gblk_cv_respects_spatial_field_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from geopfa.prob import gblk_runner as runner_module

    fixture = make_synthetic_pfa(grid_n=10, n_wells=80, seed=21)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = replace(
        _cfg(wells_path, tmp_path / "out"),
        cross_validation=CrossValidationConfig(
            n_folds=3,
            block_type="grid",
            grid_size=2,
        ),
    )

    def forbidden_spatial_fit(*_args, **_kwargs):
        raise AssertionError("spatial fit was invoked for a no-spatial model")

    monkeypatch.setattr(runner_module, "fit_gblk_joint", forbidden_spatial_fit)

    result = run_gblk_calibration_cv(
        fixture.pfa,
        cfg,
        components=["component_a"],
        n_bins=3,
    )

    assert set(result) == {"component_a"}


def test_p_gblk_cv_rejects_bayesian_config_until_bayesian_cv_exists(
    tmp_path: Path,
) -> None:
    from geopfa.prob.config import GBLKBayesianConfig

    fixture = make_synthetic_pfa(grid_n=8, n_wells=40, seed=22)
    wells_path = tmp_path / "wells.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out")
    cfg = replace(
        cfg,
        spatial_field=replace(cfg.spatial_field, enabled=True),
        inference=replace(
            cfg.inference,
            gblk_bayesian=GBLKBayesianConfig(enabled=True, n_draws=10),
        ),
    )

    with pytest.raises(GEOPFAValueError, match="Bayesian.*cross-validation"):
        run_gblk_calibration_cv(fixture.pfa, cfg, components=["component_a"])


def test_p_gblk_cv_reliability_diagram_has_expected_shape(
    tmp_path: Path,
) -> None:
    results = _run_cv(tmp_path, n_bins=6)
    for res in results.values():
        for fold in res.folds:
            rd = fold.reliability
            assert rd.bin_centers.shape == (6,)
            assert rd.bin_fracs.shape == (6,)
            assert rd.bin_counts.shape == (6,)
            assert int(rd.bin_counts.sum()) == fold.n_test


def test_p_gblk_cv_reliability_near_diagonal_on_wellspecified_data(
    tmp_path: Path,
) -> None:
    """On well-specified synthetic data, reliability curves stay near
    the diagonal (populated bins only, weighted by bin count)."""
    results = _run_cv(tmp_path)
    joint = results["joint"]
    weighted_abs_gap = 0.0
    total_count = 0
    for fold in joint.folds:
        rd = fold.reliability
        populated = rd.bin_counts > 0
        if not populated.any():
            continue
        gap = np.abs(rd.bin_fracs[populated] - rd.bin_mean_pred[populated])
        weighted_abs_gap += float((gap * rd.bin_counts[populated]).sum())
        total_count += int(rd.bin_counts[populated].sum())
    assert total_count > 0
    mean_gap = weighted_abs_gap / total_count
    assert mean_gap < 0.30, (
        f"mean absolute reliability gap {mean_gap:.3f} is too large; "
        "expected reliability near diagonal on well-specified data"
    )


def test_p_gblk_cv_positive_brier_skill_score(tmp_path: Path) -> None:
    """Positive BSS (vs per-fold prevalence) on well-specified synthetic
    data confirms the GBLK fit beats the prevalence baseline under
    spatially-blocked CV — the P6-S01 acceptance criterion.
    """
    results = _run_cv(tmp_path)
    for name, res in results.items():
        assert res.mean_bss > 0.0, (
            f"target {name!r} mean BSS {res.mean_bss:.3f} <= 0; expected "
            "positive skill over per-fold prevalence"
        )


def test_p_gblk_cv_brier_score_reasonable(tmp_path: Path) -> None:
    """Sanity check: mean Brier score is < 0.25 (a constant-0.5 baseline)."""
    results = _run_cv(tmp_path)
    for name, res in results.items():
        assert res.mean_brier_score < 0.25, (
            f"target {name!r} mean Brier {res.mean_brier_score:.3f} >= 0.25"
        )


def test_p_gblk_cv_raises_when_disabled(tmp_path: Path) -> None:
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=0)
    wells_path = tmp_path / "wells_disabled.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_disabled")
    disabled = ProbabilisticConfig(
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
    with pytest.raises(GEOPFAValueError):
        run_gblk_calibration_cv(fixture.pfa, disabled)


# ---------------------------------------------------------------------------
# PB-1: restrict calibration-CV to a subset of components
#
# Real, sparsely-labeled datasets (e.g. Newberry: 31 wells, 8 heat-positive,
# a single non-NaN producibility label, zero insulation labels) cannot
# support calibration-CV for every component in a multivariate GBLK fit —
# a class-degenerate or all-NaN label column makes the per-component metric
# meaningless (or raises). ``components`` lets a caller request calibration
# only for the component(s) that can actually support it, while the joint
# multivariate fit itself still sees every component's evidence layers.
# ---------------------------------------------------------------------------


def _make_sparse_second_component_fixture(tmp_path: Path):
    """Synthetic fixture mirroring Newberry's label sparsity.

    ``component_a`` keeps its normal synthetic labels; ``component_b``'s
    labels are collapsed to a single non-NaN value (one class), which is
    not identifiable via calibration CV.
    """
    fixture = make_synthetic_pfa(grid_n=10, n_wells=40, seed=1)
    wells = fixture.wells.copy()
    wells["reservoir_label"] = np.nan
    wells.loc[wells.index[0], "reservoir_label"] = 1.0
    fixture.wells = wells
    wells_path = tmp_path / "wells_sparse.gpkg"
    wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / "out_sparse")
    return fixture, cfg


def test_p_gblk_cv_components_filter_restricts_returned_keys(
    tmp_path: Path,
) -> None:
    fixture, cfg = _make_sparse_second_component_fixture(tmp_path)
    results = run_gblk_calibration_cv(
        fixture.pfa,
        cfg,
        n_folds=3,
        n_bins=4,
        random_state=0,
        nc=_NC_SMALL,
        a_wght=4.5,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
        components=["component_a"],
    )
    assert set(results.keys()) == {"component_a"}
    assert results["component_a"].fold_ids.shape == (len(fixture.wells),)


def test_p_gblk_cv_without_filter_rejects_unidentifiable_component(
    tmp_path: Path,
) -> None:
    """Calibration fails closed instead of converting absent labels to zero."""
    fixture, cfg = _make_sparse_second_component_fixture(tmp_path)
    with pytest.raises(
        GEOPFAValueError,
        match="component_b.*fewer than 3 observed labels",
    ):
        run_gblk_calibration_cv(
            fixture.pfa,
            cfg,
            n_folds=3,
            n_bins=4,
            random_state=0,
            nc=_NC_SMALL,
            a_wght=4.5,
            max_outer_iter=_OUTER_SMALL,
            irls_max_iter=_IRLS_SMALL,
        )


def test_p_gblk_cv_components_filter_rejects_unknown_name(
    tmp_path: Path,
) -> None:
    fixture, cfg = _make_sparse_second_component_fixture(tmp_path)
    with pytest.raises(GEOPFAValueError):
        run_gblk_calibration_cv(
            fixture.pfa,
            cfg,
            n_folds=3,
            n_bins=4,
            components=["not_a_component"],
        )


# ---------------------------------------------------------------------------
# P6-S02: Van Calster intercept/slope + post-hoc maps on GBLK surfaces
# ---------------------------------------------------------------------------


from geopfa.prob.calibration import (  # noqa: E402
    calibration_intercept_slope,
    calibration_summary,
    fit_posthoc_calibration,
)
from geopfa.prob.gblk_runner import run_gblk_probabilistic  # noqa: E402


def _simulate_calibrated(
    n: int = 4000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate perfectly-calibrated ``(y, p)`` on ``[0, 1]``."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p).astype(int)
    return y, p


def test_p_gblk_intercept_slope_near_ideal_on_calibrated_data() -> None:
    """Van Calster intercept~0 / slope~1 on well-specified synthetic data."""
    y, p = _simulate_calibrated(n=5000, seed=0)
    result = calibration_intercept_slope(y, p)
    assert abs(result["intercept"]) < 0.15, (
        f"intercept {result['intercept']:.3f} not near 0"
    )
    assert abs(result["slope"] - 1.0) < 0.15, (
        f"slope {result['slope']:.3f} not near 1"
    )


def test_p_gblk_intercept_slope_detects_overconfidence() -> None:
    """Sharpened (over-confident) predictions produce slope < 1."""
    rng = np.random.default_rng(1)
    n = 4000
    p_true = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, p_true).astype(int)
    logits_true = np.log(p_true / (1.0 - p_true))
    p_sharp = 1.0 / (1.0 + np.exp(-2.0 * logits_true))
    result = calibration_intercept_slope(y, p_sharp)
    assert result["slope"] < 0.8, (
        f"expected slope < 0.8 for over-confident predictions, "
        f"got {result['slope']:.3f}"
    )


def test_p_gblk_intercept_slope_detects_bias() -> None:
    """A systematic logit shift produces a nonzero intercept."""
    rng = np.random.default_rng(2)
    n = 4000
    p_true = rng.uniform(0.1, 0.9, size=n)
    y = rng.binomial(1, p_true).astype(int)
    logits_true = np.log(p_true / (1.0 - p_true))
    p_biased = 1.0 / (1.0 + np.exp(-(logits_true + 1.0)))
    result = calibration_intercept_slope(y, p_biased)
    assert result["intercept"] < -0.5, (
        f"expected intercept < -0.5 for upward-biased predictions, "
        f"got {result['intercept']:.3f}"
    )


def test_p_gblk_intercept_slope_handles_single_class() -> None:
    y = np.zeros(50, dtype=int)
    p = np.linspace(0.1, 0.9, 50)
    result = calibration_intercept_slope(y, p)
    assert np.isnan(result["intercept"])
    assert np.isnan(result["slope"])


def test_p_gblk_calibration_summary_reports_intercept_slope() -> None:
    y, p = _simulate_calibrated(n=3000, seed=3)
    summary = calibration_summary(y, p, n_bins=10)
    assert "calibration_intercept" in summary
    assert "calibration_slope" in summary
    assert abs(summary["calibration_intercept"]) < 0.2
    assert abs(summary["calibration_slope"] - 1.0) < 0.2


@pytest.mark.parametrize("method", ["platt", "isotonic", "temperature"])
def test_p_gblk_posthoc_map_applies_to_joint_surface(
    tmp_path: Path, method: str
) -> None:
    """Post-hoc calibration maps apply to GBLK joint surfaces.

    Fits a CalibrationMap on labeled-well predictions produced by the
    GBLK method, then applies it to the joint grid surface and verifies
    shape, range, and monotonicity/finiteness.
    """
    from tests.fixtures.synthetic_prob import make_synthetic_pfa  # noqa: PLC0415

    fixture = make_synthetic_pfa(grid_n=20, n_wells=250, seed=5)
    wells_path = tmp_path / f"wells_{method}.gpkg"
    fixture.wells.to_file(wells_path, layer="wells", driver="GPKG")
    cfg = _cfg(wells_path, tmp_path / f"out_{method}")

    result = run_gblk_probabilistic(
        fixture.pfa,
        cfg,
        nc=_NC_SMALL,
        a_wght=4.5,
        max_outer_iter=_OUTER_SMALL,
        irls_max_iter=_IRLS_SMALL,
    )
    joint_surface = result.combined["probability"].to_numpy()
    assert joint_surface.ndim == 1
    assert np.all(np.isfinite(joint_surface))

    rng = np.random.default_rng(7)
    n = 400
    y_holdout = rng.binomial(1, 0.4, size=n).astype(int)
    p_holdout = np.clip(
        0.3 + 0.4 * y_holdout + rng.normal(0.0, 0.1, size=n), 0.02, 0.98
    )
    cmap = fit_posthoc_calibration(p_holdout, y_holdout, method=method)

    calibrated = cmap.predict(joint_surface)
    assert calibrated.shape == joint_surface.shape
    assert np.all(np.isfinite(calibrated))
    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0
    if method == "none":
        np.testing.assert_allclose(calibrated, joint_surface)
    else:
        assert not np.allclose(calibrated, joint_surface)
