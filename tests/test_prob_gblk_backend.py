"""Tests for the GBLK (latticekrigx) inference backend adapter.

Requires the dedicated ``gblk`` pixi environment (``pixi run -e gblk ...``),
which provides ``latticekrigx`` + ``scikit-sparse``.  Skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("latticekrigx.glk.joint")

from geopfa.prob.gblk_backend import (  # noqa: E402
    GBLKFitResult,
    build_lkinfo,
    build_lkinfo_2d,
    fit_gblk_joint,
    scale_spatial_coordinates,
)


def _make_coupled_data(n=80, q=3, seed=0, unlabeled_frac=0.25):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 1.0, size=(n, 2))
    # smooth per-component logit offsets + shared latent factor => coupling
    smooth = np.sin(3.0 * coords[:, 0]) * np.cos(2.5 * coords[:, 1])
    offsets = np.column_stack(
        [0.6 * smooth, 0.4 * smooth + 0.2, -0.5 * smooth]
    )[:, :q]
    latent = offsets + 0.3 * rng.standard_normal((n, q))
    latent += 0.8 * rng.standard_normal((n, 1))  # strong shared factor
    probs = 1.0 / (1.0 + np.exp(-latent))
    labels = (rng.random((n, q)) < probs).astype(float)
    labeled = rng.random(n) > unlabeled_frac
    ax = np.linspace(0.1, 0.9, 8)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    g_smooth = np.sin(3.0 * grid[:, 0]) * np.cos(2.5 * grid[:, 1])
    grid_off = np.column_stack(
        [0.6 * g_smooth, 0.4 * g_smooth + 0.2, -0.5 * g_smooth]
    )[:, :q]
    names = tuple(f"c{i}" for i in range(q))
    return coords, labels, labeled, offsets, grid, grid_off, names


def test_p_gblk_backend_build_lkinfo_covers_domain():
    coords = np.array([[0.0, 0.0], [1.0, 2.0], [0.5, 1.0]])
    lk = build_lkinfo_2d(coords, nc=5, a_wght=4.5)
    assert lk.nlevel == 1
    assert lk.levels[0].NC == 5
    lo = lk.geometry.domain_lower
    hi = lk.geometry.domain_upper
    assert lo[0] <= 0.0 and hi[0] >= 1.0
    assert lo[1] <= 0.0 and hi[1] >= 2.0


def test_p_gblk_backend_build_lkinfo_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"coords_xy must be \(n, 2\)"):
        build_lkinfo_2d(np.zeros((4, 3)))


def test_p_gblk_backend_build_lkinfo_uses_box_for_3d():
    coords = np.array([[0.0, 0.0, -3.0], [1.0, 2.0, -1.0], [0.5, 1.0, -2.0]])
    lk = build_lkinfo(coords, nc=4, a_wght=6.5)
    assert lk.geometry.geometry_type == "LKBox"
    assert lk.geometry.ndim == 3
    assert lk.levels[0].NC_total == 4**3


def test_p_gblk_backend_rejects_3d_invalid_awght():
    coords = np.array([[0.0, 0.0, -3.0], [1.0, 2.0, -1.0]])
    with pytest.raises(ValueError, match="a_wght must be greater than 6"):
        build_lkinfo(coords, nc=4, a_wght=4.5)


@pytest.mark.parametrize("invalid_nlevel", [1.5, "2", True, np.bool_(False)])
def test_p_gblk_backend_rejects_noninteger_nlevel(invalid_nlevel):
    coords = np.array([[0.0, 0.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="nlevel must be a positive integer"):
        build_lkinfo(coords, nlevel=invalid_nlevel)


def test_p_gblk_physical_coordinate_scaling_preserves_axis_ratios():
    coords = np.array([[0.0, 0.0, 0.0], [100.0, 200.0, 20.0]])
    grid = np.array([[50.0, 100.0, 10.0]])

    axis_train, _, _, axis_scale = scale_spatial_coordinates(
        coords, grid, mode="axis_range"
    )
    physical_train, _, _, physical_scale = scale_spatial_coordinates(
        coords, grid, mode="physical_isotropic"
    )

    np.testing.assert_allclose(np.ptp(axis_train, axis=0), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(np.ptp(physical_train, axis=0), [0.5, 1.0, 0.1])
    np.testing.assert_allclose(axis_scale, [100.0, 200.0, 20.0])
    np.testing.assert_allclose(physical_scale, [200.0, 200.0, 200.0])


def test_p_gblk_backend_fit_shapes_and_ranges():
    coords, labels, labeled, offsets, grid, grid_off, names = (
        _make_coupled_data()
    )
    res = fit_gblk_joint(
        coords,
        labels,
        grid,
        component_names=names,
        labeled_mask=labeled,
        offsets=offsets,
        grid_offsets=grid_off,
        nc=5,
        max_outer_iter=100,
        irls_max_iter=50,
    )
    assert isinstance(res, GBLKFitResult)
    g, q = grid.shape[0], labels.shape[1]
    assert res.p_q_grid.shape == (g, q)
    assert res.p_joint_grid.shape == (g,)
    assert res.omega.shape == (q, q)
    assert res.coef.shape[1] == q
    # probabilities are valid
    assert np.all((res.p_q_grid >= 0.0) & (res.p_q_grid <= 1.0))
    assert np.all((res.p_joint_grid >= 0.0) & (res.p_joint_grid <= 1.0))
    # joint <= each marginal (product of [0,1] values)
    assert np.all(res.p_joint_grid <= res.p_q_grid.min(axis=1) + 1e-9)
    assert res.diagnostics["n_components"] == q
    assert res.diagnostics["n_labeled"] == int(labeled.sum())


def test_p_gblk_backend_estimates_nontrivial_coupling():
    # Strong shared latent factor => off-diagonal Omega should be non-zero.
    coords, labels, labeled, offsets, grid, grid_off, names = (
        _make_coupled_data(n=120, q=3, seed=1, unlabeled_frac=0.1)
    )
    res = fit_gblk_joint(
        coords,
        labels,
        grid,
        component_names=names,
        labeled_mask=labeled,
        offsets=offsets,
        grid_offsets=grid_off,
        nc=5,
        max_outer_iter=100,
        irls_max_iter=50,
    )
    omega = res.omega
    # symmetric, unit diagonal correlation matrix
    assert np.allclose(np.diag(omega), 1.0, atol=1e-6)
    assert np.allclose(omega, omega.T, atol=1e-6)
    off_diag = omega[np.triu_indices_from(omega, k=1)]
    assert np.max(np.abs(off_diag)) > 0.1  # detected coupling


def test_p_gblk_backend_joint_differs_from_naive_product_of_marginals():
    # p_joint (product within the joint fit) should differ from the product of
    # independently-averaged marginals when components are coupled.
    coords, labels, labeled, offsets, grid, grid_off, names = (
        _make_coupled_data(n=100, q=2, seed=2, unlabeled_frac=0.1)
    )
    res = fit_gblk_joint(
        coords,
        labels,
        grid,
        component_names=names,
        labeled_mask=labeled,
        offsets=offsets,
        grid_offsets=grid_off,
        nc=5,
        max_outer_iter=100,
        irls_max_iter=50,
    )
    naive = np.prod(res.p_q_grid.mean(axis=0))
    joint_mean = res.p_joint_grid.mean()
    assert np.isfinite(joint_mean)
    assert abs(joint_mean - naive) > 1e-6


def test_p_gblk_backend_shape_validation():
    coords, labels, labeled, offsets, grid, grid_off, names = (
        _make_coupled_data(q=3)
    )
    with pytest.raises(ValueError, match="component_names length"):
        fit_gblk_joint(coords, labels, grid, component_names=("only_one",))


def test_p_gblk_backend_3d_prediction_responds_to_depth():
    rng = np.random.default_rng(8)
    coords = rng.uniform(0.0, 1.0, size=(60, 3))
    labels = (coords[:, 2] > 0.5).astype(float)[:, np.newaxis]
    xy = np.array([[0.25, 0.25], [0.75, 0.75]])
    grid = np.vstack(
        [np.column_stack([xy, np.full(2, z)]) for z in (0.1, 0.9)]
    )

    res = fit_gblk_joint(
        coords,
        labels,
        grid,
        component_names=("heat",),
        offsets=np.zeros((len(coords), 1)),
        grid_offsets=np.zeros((len(grid), 1)),
        nc=4,
        a_wght=6.5,
        max_outer_iter=100,
        irls_max_iter=50,
    )

    shallow = res.p_q_grid[:2, 0]
    deep = res.p_q_grid[2:, 0]
    assert np.all(deep > shallow + 0.5)
    assert res.diagnostics["spatial_dimension"] == 3
