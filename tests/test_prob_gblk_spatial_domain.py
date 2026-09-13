"""Regression tests for the frozen GBLK spatial-domain contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("latticekrigx.glk.bayes.paige")

from geopfa.prob.config import GBLKBayesianConfig  # noqa: E402
from geopfa.prob.gblk_backend import (  # noqa: E402
    fit_gblk_bayesian_posterior_state,
    fit_gblk_joint,
    scale_spatial_coordinates,
)


def _lattice_signature(lkinfo):
    """Return the domain and level geometry that must be fixed across folds."""
    return (
        tuple(lkinfo.geometry.domain_lower),
        tuple(lkinfo.geometry.domain_upper),
        tuple(
            (level.NC, level.NC_total, level.delta) for level in lkinfo.levels
        ),
    )


def test_explicit_spatial_domain_freezes_map_geometry_across_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spatial_domain = np.array([[0.0, 10.0], [-5.0, 15.0]])
    folds = (
        (
            np.array([[0.0, -5.0], [2.0, 1.0], [5.0, 5.0]]),
            np.array([[8.0, 10.0], [10.0, 15.0]]),
        ),
        (
            np.array([[1.0, -4.0], [6.0, 6.0], [9.0, 14.0]]),
            np.array([[2.0, 2.0], [7.0, 12.0]]),
        ),
    )
    captured_lattices = []

    def fake_fit_joint(*args, **kwargs):
        y, basis, lkinfo = args[:3]
        captured_lattices.append(lkinfo)
        n_grid = kwargs["Phi_pred"].shape[0]
        n_components = y.shape[1]
        p_q = np.full((n_grid, n_components), 0.5)
        return SimpleNamespace(
            p_q=p_q,
            p_joint=np.prod(p_q, axis=1),
            params=SimpleNamespace(omega=np.eye(n_components)),
            c_matrix=np.zeros((basis.shape[1], n_components)),
            fixed_coef=None,
        )

    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)

    results = []
    for coords, grid in folds:
        results.append(
            fit_gblk_joint(
                coords,
                np.array([[0.0], [1.0], [0.0]]),
                grid,
                component_names=("heat",),
                spatial_domain=spatial_domain,
                nc=3,
                nlevel=2,
            )
        )

    assert _lattice_signature(captured_lattices[0]) == _lattice_signature(
        captured_lattices[1]
    )
    for result in results:
        np.testing.assert_allclose(
            result.diagnostics["coordinate_lower"], [0, -5]
        )
        np.testing.assert_allclose(
            result.diagnostics["coordinate_span"], [10, 20]
        )
        np.testing.assert_allclose(
            result.diagnostics["coordinate_scale"], [10, 20]
        )
        np.testing.assert_allclose(
            result.diagnostics["spatial_domain"], spatial_domain
        )
        assert result.diagnostics["spatial_domain_explicit"] is True


def test_explicit_spatial_domain_freezes_bayesian_geometry_across_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spatial_domain = np.array([[100.0, 300.0], [1_000.0, 1_100.0]])
    folds = (
        (
            np.array([[110.0, 1_010.0], [150.0, 1_030.0], [200.0, 1_050.0]]),
            np.array([[250.0, 1_080.0]]),
        ),
        (
            np.array([[120.0, 1_020.0], [220.0, 1_060.0], [280.0, 1_090.0]]),
            np.array([[175.0, 1_040.0]]),
        ),
    )
    captured_lattices = []

    def fake_fit_joint(*args, **_kwargs):
        y, basis, lkinfo = args[:3]
        captured_lattices.append(lkinfo)
        return SimpleNamespace(
            inference="inla",
            extra={
                "c_draws": np.zeros((2, basis.shape[1], y.shape[1])),
                "fixed_draws": None,
            },
        )

    monkeypatch.setattr("latticekrigx.glk.joint.fit_joint", fake_fit_joint)
    config = GBLKBayesianConfig(
        enabled=True,
        n_draws=2,
        cluster_effect=False,
        validate_inla=False,
    )

    states = []
    for coords, grid in folds:
        states.append(
            fit_gblk_bayesian_posterior_state(
                coords,
                np.array([[0.0], [1.0], [0.0]]),
                grid,
                component_names=("heat",),
                bayes_config=config,
                spatial_domain=spatial_domain,
                coordinate_scaling="physical_isotropic",
                nc=3,
                nlevel=2,
            )
        )

    assert _lattice_signature(captured_lattices[0]) == _lattice_signature(
        captured_lattices[1]
    )
    for state in states:
        np.testing.assert_allclose(
            state.diagnostics["spatial_domain"], spatial_domain
        )
        np.testing.assert_allclose(
            state.diagnostics["coordinate_lower"], [100, 1_000]
        )
        np.testing.assert_allclose(
            state.diagnostics["coordinate_span"], [200, 100]
        )
        np.testing.assert_allclose(
            state.diagnostics["coordinate_scale"], [200, 200]
        )
        assert state.diagnostics["spatial_domain_explicit"] is True


def test_spatial_domain_rejects_zero_span() -> None:
    spatial_domain = np.array([[0.0, 0.0], [-1.0, 1.0]])
    with pytest.raises(
        ValueError, match="upper bounds must be strictly greater"
    ):
        scale_spatial_coordinates(
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.5]]),
            spatial_domain=spatial_domain,
        )


def test_spatial_domain_rejects_fit_or_prediction_points_outside() -> None:
    spatial_domain = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="outside spatial_domain"):
        scale_spatial_coordinates(
            np.array([[0.25, 0.25], [1.01, 0.5]]),
            np.array([[0.75, 0.75]]),
            spatial_domain=spatial_domain,
        )
    with pytest.raises(ValueError, match="outside spatial_domain"):
        scale_spatial_coordinates(
            np.array([[0.25, 0.25], [0.5, 0.5]]),
            np.array([[-0.01, 0.75]]),
            spatial_domain=spatial_domain,
        )
