"""Scale benchmark: latticekrigx streaming extrapolation + GBLK fit.

Demonstrates that peak memory during streaming prediction and GBLK fitting
is flat in ``n`` (the number of prediction/grid points) by comparing a
baseline grid against a 10x larger grid.

The key latticekrigx invariant:
- LKX basis size M is determined by ``(NC, nlevel)`` config, NOT by N.
- Prediction streams in fixed-size chunks (default 10 000 rows), so peak
  memory during extrapolation is O(chunk_size * M), not O(N_pred * M).
- GBLK fitting: the precision matrix is M x M (fixed); only the cross-product
  Phi^T y (M-vector) grows with N_obs at O(N * M) time but O(M) memory.

Usage::

    pixi run -e dev-gblk python scripts/benchmarks/scale_benchmark.py

Exit codes:
    0 — all assertions pass (memory flat within tolerance)
    1 — at least one memory assertion failed (regression)
"""

from __future__ import annotations

import tracemalloc
from typing import NamedTuple

import numpy as np

from geopfa.prob.gblk_backend import fit_gblk_joint
from geopfa.spatial_lkx import LkxConfig, fit_lkx_field, lkx_predict

_RNG = np.random.default_rng(42)
_N_TRAIN_SMALL = 200
_N_TRAIN_LARGE = 2_000
_CHUNK = 5_000
# Allow 4x headroom above baseline to call "flat" (streaming is not perfectly
# constant due to numpy temporaries and lkx_predict return arrays).
_MEMORY_RATIO_THRESHOLD = 4.0

_SMALL_GRID = 5_000
_LARGE_GRID = 50_000
_GBLK_PRED_GRID = 2_000


class _BenchResult(NamedTuple):
    label: str
    n_pred: int
    peak_mb: float


def _synthetic_2d(
    n: int,
    *,
    rng: np.random.Generator = _RNG,
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-1.0, 1.0, (n, 2))
    Y = np.sin(np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + rng.normal(0, 0.1, n)
    return X, Y


def _grid_2d(n_side: int) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n_side)
    gx, gy = np.meshgrid(t, t)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _bench_lkx_predict(
    n_grid_pts: int,
    model: object,
    label: str,
) -> _BenchResult:
    n_side = int(np.ceil(np.sqrt(n_grid_pts)))
    X_pred = _grid_2d(n_side)
    tracemalloc.start()
    tracemalloc.clear_traces()
    lkx_predict(model, X_pred, chunk_size=_CHUNK)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _BenchResult(label=label, n_pred=X_pred.shape[0], peak_mb=peak / 1e6)


def _bench_gblk_fit(
    n_obs: int,
    n_grid_pts: int,
    label: str,
) -> _BenchResult:
    """Benchmark GBLK fit peak memory for varying N_obs with a fixed grid.

    The fit precision matrix is M x M (fixed by LK config); the training-data
    cross-product accumulation is O(N_obs x M) time but O(M) memory. This
    function demonstrates that peak fit memory is flat in N_obs (10x increase).
    """
    X_obs, _ = _synthetic_2d(n_obs)
    labels = np.column_stack([
        _RNG.integers(0, 2, n_obs).astype(float),
        _RNG.integers(0, 2, n_obs).astype(float),
    ])
    n_side = int(np.ceil(np.sqrt(n_grid_pts)))
    X_pred = _grid_2d(n_side)

    tracemalloc.start()
    tracemalloc.clear_traces()
    fit_gblk_joint(
        coords_xy=X_obs,
        labels=labels,
        grid_xy=X_pred,
        component_names=("heat", "reservoir"),
        nc=6,
        a_wght=4.5,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return _BenchResult(label=label, n_pred=n_obs, peak_mb=peak / 1e6)


def run() -> bool:
    """Run the scale benchmark and return True when all assertions pass."""
    print("=" * 60)
    print("geoPFA latticekrigx scale benchmark")
    print("=" * 60)

    X_train, Y_train = _synthetic_2d(_N_TRAIN_SMALL)
    cfg = LkxConfig(nlevel=3, NC=8, lambda_=0.01)
    model = fit_lkx_field(X_train, Y_train, config=cfg)
    print(f"LKX model fitted: ndim={model.ndim}, geometry={model.geometry}")

    print("\n--- LKX streaming extrapolation ---")
    small = _bench_lkx_predict(_SMALL_GRID, model, "lkx_small")
    large = _bench_lkx_predict(_LARGE_GRID, model, "lkx_large")
    lkx_ratio = large.peak_mb / max(small.peak_mb, 1e-6)
    print(
        f"  small grid  n={small.n_pred:>6}  peak={small.peak_mb:.2f} MB"
    )
    print(
        f"  large grid  n={large.n_pred:>6}  peak={large.peak_mb:.2f} MB  "
        f"ratio={lkx_ratio:.2f}x (threshold <{_MEMORY_RATIO_THRESHOLD}x)"
    )
    lkx_ok = lkx_ratio < _MEMORY_RATIO_THRESHOLD
    print(f"  LKX memory-flat: {'PASS' if lkx_ok else 'FAIL'}")

    print("\n--- GBLK joint fit (memory flat in N_obs) ---")
    gblk_small = _bench_gblk_fit(_N_TRAIN_SMALL, _GBLK_PRED_GRID, "gblk_small")
    gblk_large = _bench_gblk_fit(_N_TRAIN_LARGE, _GBLK_PRED_GRID, "gblk_large")
    gblk_ratio = gblk_large.peak_mb / max(gblk_small.peak_mb, 1e-6)
    print(
        f"  N_obs={_N_TRAIN_SMALL:>5}  n_pred={gblk_small.n_pred}  peak={gblk_small.peak_mb:.2f} MB"
    )
    print(
        f"  N_obs={_N_TRAIN_LARGE:>5}  n_pred={gblk_large.n_pred}  peak={gblk_large.peak_mb:.2f} MB  "
        f"ratio={gblk_ratio:.2f}x (threshold <{_MEMORY_RATIO_THRESHOLD}x)"
    )
    gblk_ok = gblk_ratio < _MEMORY_RATIO_THRESHOLD
    print(f"  GBLK memory-flat: {'PASS' if gblk_ok else 'FAIL'}")

    all_ok = lkx_ok and gblk_ok
    print("\n" + ("=" * 60))
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
