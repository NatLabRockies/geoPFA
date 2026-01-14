import numpy as np
import pandas as pd

from geopfa.extrapolation import grid_from_tidy


def test_grid_from_tidy_shapes_and_order():
    df = pd.DataFrame(
        {
            "x": [2, 1, 1, 2],
            "y": [1, 1, 2, 2],
            "value": [10, 20, 30, 40],
        }
    )

    grid, xs, ys, x2d, y2d = grid_from_tidy(df)

    assert list(xs) == [1, 2]
    assert list(ys) == [1, 2]

    expected = np.array(
        [
            [20, 10],
            [30, 40],
        ]
    )
    assert np.allclose(grid.values, expected)

    assert x2d.shape == y2d.shape == expected.shape
    assert np.allclose(x2d, np.array([[1, 2], [1, 2]]))
    assert np.allclose(y2d, np.array([[1, 1], [2, 2]]))


def test_grid_from_tidy_handles_missing():
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [1, 2],
            "value": [10, np.nan],
        }
    )

    grid, xs, ys, *_ = grid_from_tidy(df)

    assert np.isnan(grid.values[1, 1])
