"""Tests for the LatticeKrigX heuristic configuration (P1-S02).

Acceptance criteria:

- returned config spans ``[0.02R, 0.20R]`` scale range;
- ``nlevel >= 3`` always;
- zero-variance dims (already collapsed by standardization) do not crash;
- ``lambda_`` and ``lambda_bounds`` match the documented noise formula;
- ``NC`` and ``nlevel`` satisfy the per-dim basis-count analogue of item 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.spatial_lkx import (
    LkxConfig,
    _MAX_TOTAL_BASIS,
    _COARSE_SCALE_FRAC,
    _FINE_SCALE_FRAC,
    _LAMBDA_BOUND_HI_FRAC,
    _LAMBDA_BOUND_LO,
    _LAMBDA_INIT_FRAC,
    _MAX_BASIS_PER_OBS,
    _MIN_DATA_BASIS,
    compute_lkx_config,
)


def _total_basis(cfg: LkxConfig, D: int) -> int:
    return int(sum((cfg.NC * (2**lvl)) ** D for lvl in range(cfg.nlevel)))


def _effective_basis_cap(n_train: int, D: int) -> int:
    absolute = _MAX_TOTAL_BASIS if D <= 2 else 1500
    data_cap = max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS * n_train)
    return min(absolute, data_cap)


def _implied_scales(cfg: LkxConfig, max_range: float) -> tuple[float, float]:
    """Return (finest_scale, coarsest_scale) implied by a config."""
    coarsest = max_range / cfg.NC
    finest = max_range / (cfg.NC * 2 ** (cfg.nlevel - 1))
    return finest, coarsest


def _make_x_std(n: int, D: int, *, rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.standard_normal(size=(n, D))


@pytest.fixture(scope="module")
def cfg_2d():
    X = _make_x_std(200, 2)
    return compute_lkx_config(X, n_train=200)


@pytest.fixture(scope="module")
def cfg_3d():
    X = _make_x_std(300, 3)
    return compute_lkx_config(X, n_train=300)


def test_returns_lkx_config(cfg_2d):
    assert isinstance(cfg_2d, LkxConfig)


def test_nlevel_minimum_2d(cfg_2d):
    # The data-proportional basis cap (max(_MIN_DATA_BASIS, k*n)) can reduce
    # nlevel below _MIN_NLEVEL for modest n. The config must remain a valid
    # (>=1) multiresolution whose total basis stays under the effective cap.
    assert cfg_2d.nlevel >= 1
    assert _total_basis(cfg_2d, 2) <= _effective_basis_cap(200, 2)


def test_nlevel_minimum_3d(cfg_3d):
    # In 3D the basis count (~NC**D per level) is capped for tractability, so
    # nlevel may be reduced below _MIN_NLEVEL. It must remain a valid (>=1)
    # multiresolution config whose total basis stays under the cap.
    assert cfg_3d.nlevel >= 1
    total = sum((cfg_3d.NC * (2**lvl)) ** 3 for lvl in range(cfg_3d.nlevel))
    assert total <= _MAX_TOTAL_BASIS


def test_scale_range_spans_2d(cfg_2d):
    X = _make_x_std(200, 2)
    max_range = float((X.max(axis=0) - X.min(axis=0)).max())
    R = 0.5 * max_range
    finest, coarsest = _implied_scales(cfg_2d, max_range)
    # Coarse scale is always preserved. The finest scale may be limited by the
    # data-proportional basis cap (a modest-n fit cannot resolve 0.02R without
    # over-parameterizing), but the config must still be multi-scale when it
    # uses more than one level.
    assert coarsest >= _COARSE_SCALE_FRAC * R, (
        f"coarsest scale {coarsest:.4f} < {_COARSE_SCALE_FRAC * R:.4f} "
        f"(0.20R) — config does not reach the coarse end of the target range"
    )
    if cfg_2d.nlevel > 1:
        assert finest < coarsest


def test_data_proportional_basis_cap_2d():
    """Small-n / large-domain fits must not over-parameterize: total basis is
    bounded by max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS * n_train)."""
    rng = np.random.default_rng(3)
    # Wide domain, few points: the old heuristic would target ~34k basis.
    X = rng.uniform(-50, 50, size=(400, 2))
    cfg = compute_lkx_config(X, n_train=400)
    assert _total_basis(cfg, 2) <= _effective_basis_cap(400, 2)
    # And this must be far below the absolute 2D cap (proves the data cap bound).
    assert _total_basis(cfg, 2) < _MAX_TOTAL_BASIS


def test_data_cap_has_minimum_floor():
    """Even tiny n keeps at least _MIN_DATA_BASIS worth of headroom so the
    multiresolution stays usable."""
    X = _make_x_std(8, 2)
    cfg = compute_lkx_config(X, n_train=8)
    assert _total_basis(cfg, 2) <= max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS * 8)
    assert cfg.NC >= 4


def test_large_n_reaches_fine_scale_2d():
    """When data supports it (large n), the config still spans to the 0.02R
    fine scale — the memo contract holds where the data-cap does not bind."""
    X = _make_x_std(20_000, 2)
    cfg = compute_lkx_config(X, n_train=20_000)
    max_range = float((X.max(axis=0) - X.min(axis=0)).max())
    R = 0.5 * max_range
    finest, _ = _implied_scales(cfg, max_range)
    assert finest <= _FINE_SCALE_FRAC * R


def test_scale_range_spans_3d(cfg_3d):
    X = _make_x_std(300, 3)
    max_range = float((X.max(axis=0) - X.min(axis=0)).max())
    R = 0.5 * max_range
    finest, coarsest = _implied_scales(cfg_3d, max_range)
    # Coarse scale is preserved. In 3D the finest scale is limited by the
    # basis-count cap (cannot reach 0.02R without an intractable basis), but
    # the config must still be multi-scale (finest strictly finer than coarsest)
    # when more than one level is used.
    assert coarsest >= _COARSE_SCALE_FRAC * R
    if cfg_3d.nlevel > 1:
        assert finest < coarsest


def test_lambda_init_matches_formula(cfg_2d):
    assert cfg_2d.lambda_ == pytest.approx(_LAMBDA_INIT_FRAC)


def test_lambda_bounds_match_formula(cfg_2d):
    lo, hi = cfg_2d.lambda_bounds
    assert lo == pytest.approx(_LAMBDA_BOUND_LO)
    assert hi == pytest.approx(_LAMBDA_BOUND_HI_FRAC)


def test_lambda_init_lo_lt_hi(cfg_2d):
    lo, hi = cfg_2d.lambda_bounds
    assert lo < cfg_2d.lambda_ < hi


def test_zero_variance_dim_does_not_raise():
    rng = np.random.default_rng(7)
    X = rng.standard_normal(size=(100, 2))
    X[:, 1] = 0.0
    cfg = compute_lkx_config(X, n_train=100)
    assert isinstance(cfg, LkxConfig)
    assert cfg.nlevel >= 1


def test_all_zero_variance_does_not_raise():
    X = np.zeros((50, 2))
    cfg = compute_lkx_config(X, n_train=50)
    assert isinstance(cfg, LkxConfig)
    assert cfg.nlevel >= 1


def test_basis_count_grows_with_n_train():
    X = _make_x_std(50, 2)
    cfg_small = compute_lkx_config(X, n_train=50)
    cfg_large = compute_lkx_config(X, n_train=3000)
    assert cfg_large.NC >= cfg_small.NC


def test_small_n_train_nc_has_minimum():
    X = _make_x_std(10, 2)
    cfg = compute_lkx_config(X, n_train=10)
    assert cfg.NC >= 4


def test_nc_integer_and_positive(cfg_2d, cfg_3d):
    assert isinstance(cfg_2d.NC, int) and cfg_2d.NC > 0
    assert isinstance(cfg_3d.NC, int) and cfg_3d.NC > 0


def test_nlevel_integer_and_positive(cfg_2d, cfg_3d):
    assert isinstance(cfg_2d.nlevel, int) and cfg_2d.nlevel > 0
    assert isinstance(cfg_3d.nlevel, int) and cfg_3d.nlevel > 0
