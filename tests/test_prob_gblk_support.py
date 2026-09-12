"""Tests for linear change-of-support helpers in gblk_assemble."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from latticekrigx.basis.assembly import compute_basis
from latticekrigx.glk.support import Support
from latticekrigx.model.config import LKGeometryInfo, LKInfo, LKLevelInfo

from geopfa.exceptions import GEOPFAValueError
from geopfa.prob.gblk_assemble import (
    build_observation_basis,
    make_areal_support,
    make_interval_support,
    make_point_support,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _make_lkinfo(ndim: int = 2) -> LKInfo:
    """Minimal LKInfo for a unit-hypercube domain."""
    domain_lower = [0.0] * ndim
    domain_upper = [1.0] * ndim
    if ndim == 2:
        geometry_type = "LKRectangle"
        nc = 4
        a_wght = [4.5]
    else:
        geometry_type = "LKBox"
        nc = 3
        a_wght = [6.5]
    geom = LKGeometryInfo(
        geometry_type=geometry_type,
        ndim=ndim,
        domain_lower=domain_lower,
        domain_upper=domain_upper,
    )
    lv = LKLevelInfo(
        level=1,
        NC=nc,
        NC_total=nc**ndim,
        delta=1.0 / nc,
        overlap=2.5,
    )
    return LKInfo(
        geometry=geom,
        nlevel=1,
        levels=[lv],
        alpha=[1.0],
        a_wght=a_wght,
        normalize=False,
    )


# ---------------------------------------------------------------------------
# make_point_support
# ---------------------------------------------------------------------------


def test_p_gblk_support_point_returns_support():
    x = np.array([0.3, 0.7])
    s = make_point_support(x)
    assert isinstance(s, Support)
    assert s.kind == "point"
    assert s.x.shape == (1, 2)


def test_p_gblk_support_point_2d_1d_input():
    x = np.array([0.5, 0.5])
    s = make_point_support(x)
    assert s.x.shape == (1, 2)
    assert s.w.shape == (1,)
    np.testing.assert_allclose(s.w, [1.0])


def test_p_gblk_support_point_3d():
    x = np.array([0.2, 0.4, 0.6])
    s = make_point_support(x)
    assert s.x.shape == (1, 3)


# ---------------------------------------------------------------------------
# make_interval_support
# ---------------------------------------------------------------------------


def test_p_gblk_support_interval_returns_support():
    a = np.array([0.1, 0.5])
    b = np.array([0.9, 0.5])
    s = make_interval_support(a, b, n_quad=5)
    assert isinstance(s, Support)
    assert s.kind == "interval"
    assert s.x.shape == (5, 2)


def test_p_gblk_support_interval_equal_weights():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    s = make_interval_support(a, b, n_quad=7)
    assert s.w.shape == (7,)
    np.testing.assert_allclose(s.w, np.ones(7))


def test_p_gblk_support_interval_endpoints():
    a = np.array([0.1, 0.2])
    b = np.array([0.8, 0.6])
    s = make_interval_support(a, b, n_quad=3)
    np.testing.assert_allclose(s.x[0], a)
    np.testing.assert_allclose(s.x[-1], b)


def test_p_gblk_support_interval_default_n_quad():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    s = make_interval_support(a, b)
    assert s.x.shape[0] == 10


def test_p_gblk_support_interval_single_point():
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    s = make_interval_support(a, b, n_quad=1)
    assert s.x.shape == (1, 2)


def test_p_gblk_support_interval_raises_on_n_quad_zero():
    with pytest.raises(GEOPFAValueError):
        make_interval_support(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]), n_quad=0
        )


# ---------------------------------------------------------------------------
# make_areal_support
# ---------------------------------------------------------------------------


def test_p_gblk_support_areal_equal_weights_default():
    pts = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    s = make_areal_support(pts)
    assert isinstance(s, Support)
    assert s.kind == "areal"
    assert s.x.shape == (3, 2)
    np.testing.assert_allclose(s.w, np.ones(3))


def test_p_gblk_support_areal_explicit_weights():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    w = np.array([1.0, 2.0, 3.0])
    s = make_areal_support(pts, weights=w)
    np.testing.assert_allclose(s.w, w)


def test_p_gblk_support_areal_single_point_1d_input():
    pts = np.array([0.5, 0.5])
    s = make_areal_support(pts)
    assert s.x.shape == (1, 2)


# ---------------------------------------------------------------------------
# build_observation_basis — acceptance: point equals plain basis row
# ---------------------------------------------------------------------------


def test_p_gblk_support_point_basis_row_equals_compute_basis():
    """Point support must equal the plain compute_basis row (atol 1e-8)."""
    lkinfo = _make_lkinfo(ndim=2)
    x = np.array([[0.3, 0.6]])
    s = make_point_support(x[0])
    Phi_support = build_observation_basis([s], lkinfo)
    Phi_direct = compute_basis(x, lkinfo, normalize=False)
    np.testing.assert_allclose(
        Phi_support.toarray(),
        Phi_direct.toarray(),
        atol=1e-8,
    )


def test_p_gblk_support_multiple_points_match_compute_basis():
    """Stack of point supports must match compute_basis row-by-row."""
    lkinfo = _make_lkinfo(ndim=2)
    xs = np.array([[0.1, 0.2], [0.5, 0.5], [0.8, 0.3]])
    supports = [make_point_support(x) for x in xs]
    Phi_support = build_observation_basis(supports, lkinfo)
    Phi_direct = compute_basis(xs, lkinfo, normalize=False)
    assert Phi_support.shape == Phi_direct.shape
    np.testing.assert_allclose(
        Phi_support.toarray(),
        Phi_direct.toarray(),
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# build_observation_basis — acceptance: interval/areal = weighted mean
# ---------------------------------------------------------------------------


def test_p_gblk_support_interval_basis_row_is_weighted_mean():
    """Interval basis row must equal the weighted mean of member rows (1e-8)."""
    lkinfo = _make_lkinfo(ndim=2)
    pts = np.array([[0.2, 0.5], [0.4, 0.5], [0.6, 0.5], [0.8, 0.5]])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    s = Support(x=pts, w=w, kind="interval")
    row = build_observation_basis([s], lkinfo).toarray()

    Phi_members = compute_basis(pts, lkinfo, normalize=False).toarray()
    w_norm = w / w.sum()
    expected = (w_norm[:, np.newaxis] * Phi_members).sum(axis=0).reshape(1, -1)
    np.testing.assert_allclose(row, expected, atol=1e-8)


def test_p_gblk_support_areal_basis_row_is_weighted_mean():
    """Areal basis row must equal the weighted mean of member rows (1e-8)."""
    lkinfo = _make_lkinfo(ndim=2)
    pts = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])
    w = np.array([1.0, 2.0, 3.0])
    s = make_areal_support(pts, weights=w)
    row = build_observation_basis([s], lkinfo).toarray()

    Phi_members = compute_basis(pts, lkinfo, normalize=False).toarray()
    w_norm = w / w.sum()
    expected = (w_norm[:, np.newaxis] * Phi_members).sum(axis=0).reshape(1, -1)
    np.testing.assert_allclose(row, expected, atol=1e-8)


def test_p_gblk_support_interval_equal_weights_is_arithmetic_mean():
    """Equal-weight interval basis row equals the unweighted row mean."""
    lkinfo = _make_lkinfo(ndim=2)
    a = np.array([0.1, 0.3])
    b = np.array([0.7, 0.3])
    s = make_interval_support(a, b, n_quad=5)
    row = build_observation_basis([s], lkinfo).toarray()

    Phi_members = compute_basis(s.x, lkinfo, normalize=False).toarray()
    expected = Phi_members.mean(axis=0).reshape(1, -1)
    np.testing.assert_allclose(row, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# build_observation_basis — shape and mixed supports
# ---------------------------------------------------------------------------


def test_p_gblk_support_output_shape():
    lkinfo = _make_lkinfo(ndim=2)
    n_obs = 5
    pts = np.random.default_rng(42).uniform(0, 1, (n_obs, 2))
    supports = [make_point_support(pt) for pt in pts]
    Phi = build_observation_basis(supports, lkinfo)
    assert Phi.shape[0] == n_obs
    assert Phi.shape[1] == lkinfo.levels[0].NC_total


def test_p_gblk_support_output_is_sparse_csr():
    lkinfo = _make_lkinfo(ndim=2)
    s = make_point_support(np.array([0.5, 0.5]))
    result = build_observation_basis([s], lkinfo)
    assert scipy.sparse.issparse(result)
    assert result.format == "csr"


def test_p_gblk_support_mixed_supports_shape():
    """Mixed point + interval + areal supports stack correctly."""
    lkinfo = _make_lkinfo(ndim=2)
    s_pt = make_point_support(np.array([0.5, 0.5]))
    s_int = make_interval_support(
        np.array([0.1, 0.3]), np.array([0.8, 0.3]), n_quad=4
    )
    s_areal = make_areal_support(
        np.array([[0.2, 0.2], [0.6, 0.2], [0.4, 0.7]])
    )
    Phi = build_observation_basis([s_pt, s_int, s_areal], lkinfo)
    assert Phi.shape == (3, lkinfo.levels[0].NC_total)


# ---------------------------------------------------------------------------
# build_observation_basis — error handling
# ---------------------------------------------------------------------------


def test_p_gblk_support_empty_supports_raises():
    lkinfo = _make_lkinfo(ndim=2)
    with pytest.raises(GEOPFAValueError):
        build_observation_basis([], lkinfo)


# ---------------------------------------------------------------------------
# 3-D support (trajectory/depth interval)
# ---------------------------------------------------------------------------


def test_p_gblk_support_3d_point_basis_row_equals_compute_basis():
    """3-D point support equals compute_basis row (atol 1e-8)."""
    lkinfo = _make_lkinfo(ndim=3)
    x = np.array([[0.3, 0.4, 0.6]])
    s = make_point_support(x[0])
    Phi_support = build_observation_basis([s], lkinfo)
    Phi_direct = compute_basis(x, lkinfo, normalize=False)
    np.testing.assert_allclose(
        Phi_support.toarray(),
        Phi_direct.toarray(),
        atol=1e-8,
    )


def test_p_gblk_support_3d_interval_basis_row_is_weighted_mean():
    """3-D interval basis row equals weighted mean of member rows (1e-8)."""
    lkinfo = _make_lkinfo(ndim=3)
    a = np.array([0.5, 0.5, 0.1])
    b = np.array([0.5, 0.5, 0.9])
    s = make_interval_support(a, b, n_quad=6)
    row = build_observation_basis([s], lkinfo).toarray()

    Phi_members = compute_basis(s.x, lkinfo, normalize=False).toarray()
    expected = Phi_members.mean(axis=0).reshape(1, -1)
    np.testing.assert_allclose(row, expected, atol=1e-8)
