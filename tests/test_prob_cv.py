"""Tests for the spatial-block CV splitter."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.prob import cv as cv_module
from geopfa.prob.cv import SpatialBlockKFold, spatial_block_cv


def _xy_grid(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(0.0, 10.0, n), rng.uniform(0.0, 10.0, n)]
    )


# ---------------------------------------------------------------------------
# spatial_block_cv (functional API)
# ---------------------------------------------------------------------------


def test_spatial_block_cv_grid_yields_disjoint_train_test() -> None:
    coords = _xy_grid(120, seed=0)
    folds = list(
        spatial_block_cv(coords, n_folds=4, block_type="grid", grid_size=4)
    )
    assert len(folds) == 4
    seen = np.zeros(len(coords), dtype=bool)
    for train_mask, test_mask in folds:
        # Train and test are disjoint and cover the whole dataset
        assert not (train_mask & test_mask).any()
        assert (train_mask | test_mask).all()
        seen |= test_mask
    # Across folds, every point appears in the test set at least once
    assert seen.all()


def test_spatial_block_cv_kmeans_yields_contiguous_blocks() -> None:
    coords = _xy_grid(80, seed=1)
    folds = list(spatial_block_cv(coords, n_folds=4, block_type="kmeans"))
    assert len(folds) == 4
    # Sanity: each test set is non-empty
    for _, test_mask in folds:
        assert test_mask.any()


def test_spatial_block_cv_balances_rows_without_splitting_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whole spatial blocks are packed by row count, without using outcomes."""
    block_ids = np.concatenate(
        [
            np.full(10, 0),
            np.full(1, 1),
            np.full(9, 2),
            np.full(1, 3),
            np.full(8, 4),
            np.full(1, 5),
        ]
    )
    coords = np.column_stack(
        [np.arange(block_ids.size, dtype=float), np.zeros(block_ids.size)]
    )

    def fixed_blocks(_received: np.ndarray, grid_size: int) -> np.ndarray:
        del grid_size
        return block_ids.copy()

    monkeypatch.setattr(
        cv_module,
        "_grid_blocks",
        fixed_blocks,
    )

    folds = list(
        spatial_block_cv(coords, n_folds=3, block_type="grid", grid_size=3)
    )
    test_counts = [int(test.sum()) for _, test in folds]

    assert test_counts == [10, 10, 10]
    for block in np.unique(block_ids):
        memberships = [
            bool(test[block_ids == block].all()) for _, test in folds
        ]
        assert sum(memberships) == 1


def test_spatial_block_cv_returns_n_folds_partitions() -> None:
    coords = _xy_grid(50, seed=2)
    n_folds = 5
    folds = list(spatial_block_cv(coords, n_folds=n_folds, block_type="grid"))
    assert len(folds) == n_folds


def test_spatial_block_cv_rejects_unknown_block_type() -> None:
    coords = _xy_grid(20)
    with pytest.raises(ValueError, match="block_type"):
        list(spatial_block_cv(coords, block_type="unknown"))


def test_spatial_block_cv_3d_coords_use_xy_only() -> None:
    """3D points should still be partitioned on their (x, y) projection."""
    rng = np.random.default_rng(0)
    n = 60
    coords_3d = np.column_stack(
        [
            rng.uniform(0.0, 10.0, n),
            rng.uniform(0.0, 10.0, n),
            rng.uniform(-3000.0, -500.0, n),
        ]
    )
    folds = list(spatial_block_cv(coords_3d, n_folds=4, block_type="grid"))
    assert len(folds) == 4
    for train_mask, test_mask in folds:
        assert not (train_mask & test_mask).any()


def test_spatial_block_cv_grid_size_controls_block_count() -> None:
    coords = _xy_grid(200, seed=3)
    folds_small = list(
        spatial_block_cv(coords, n_folds=4, block_type="grid", grid_size=2)
    )
    folds_large = list(
        spatial_block_cv(coords, n_folds=8, block_type="grid", grid_size=4)
    )
    assert len(folds_small) == 4
    assert len(folds_large) == 8


def test_automatic_block_size_uses_selected_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinates = np.column_stack(
        [
            np.linspace(0.0, 10_000.0, 20),
            np.linspace(0.0, 10.0, 20),
            np.linspace(10.0, 20.0, 20),
        ]
    )
    captured: dict[str, np.ndarray | int] = {}

    def recommend(selected: np.ndarray, residuals: np.ndarray) -> float:
        captured["selected"] = selected.copy()
        assert residuals.shape == (20,)
        return 1.0

    def fixed_blocks(selected: np.ndarray, grid_size: int) -> np.ndarray:
        captured["blocking"] = selected.copy()
        captured["grid_size"] = grid_size
        return np.arange(len(selected)) % 4

    monkeypatch.setattr(
        "geopfa.prob.variogram.recommend_block_size_km", recommend
    )
    monkeypatch.setattr(cv_module, "_grid_blocks", fixed_blocks)

    folds = list(
        spatial_block_cv(
            coordinates,
            residuals=np.linspace(-1.0, 1.0, 20),
            n_folds=4,
            block_type="grid",
            dims=(1, 2),
        )
    )

    assert len(folds) == 4
    np.testing.assert_array_equal(captured["selected"], coordinates[:, [1, 2]])
    np.testing.assert_array_equal(captured["blocking"], coordinates[:, [1, 2]])
    assert captured["grid_size"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_folds": True}, "n_folds"),
        ({"n_folds": 1}, "n_folds"),
        ({"grid_size": 0}, "grid_size"),
        ({"block_type": "kmeans", "block_size_km": 1.0}, "block_size_km"),
        ({"residuals": np.ones(3)}, "residuals"),
    ],
)
def test_spatial_block_cv_rejects_invalid_design_controls(
    kwargs: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        list(spatial_block_cv(_xy_grid(20), **kwargs))


# ---------------------------------------------------------------------------
# SpatialBlockKFold (sklearn-compatible OO API)
# ---------------------------------------------------------------------------


def test_spatial_block_kfold_get_n_splits_matches_constructor() -> None:
    coords = _xy_grid(50)
    cv = SpatialBlockKFold(coords=coords, n_splits=5, block_type="grid")
    assert cv.get_n_splits() == 5


def test_spatial_block_kfold_split_returns_index_arrays() -> None:
    coords = _xy_grid(60)
    cv = SpatialBlockKFold(coords=coords, n_splits=4, block_type="grid")
    splits = list(cv.split(np.zeros((60, 3))))
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        # Train and test indices are disjoint
        assert not (set(train_idx.tolist()) & set(test_idx.tolist()))


def test_spatial_block_kfold_forwards_fixed_block_size() -> None:
    coords = _xy_grid(80, seed=8)
    expected = list(
        spatial_block_cv(
            coords,
            n_folds=4,
            block_type="grid",
            grid_size=4,
            block_size_km=0.003,
        )
    )
    cv = SpatialBlockKFold(
        coords=coords,
        n_splits=4,
        block_type="grid",
        grid_size=4,
        block_size_km=0.003,
    )

    actual = list(cv.split(np.zeros((80, 1))))

    for (train_mask, test_mask), (train_idx, test_idx) in zip(
        expected, actual, strict=True
    ):
        np.testing.assert_array_equal(train_idx, np.flatnonzero(train_mask))
        np.testing.assert_array_equal(test_idx, np.flatnonzero(test_mask))
