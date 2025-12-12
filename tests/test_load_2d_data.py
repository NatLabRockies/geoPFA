import numpy as np

from geopfa.extrapolation import load_2d_data
from tests.fixtures.data_generators import generate_campbell2d_grid


def test_load_2d_data_shapes():
    gdf, X, Y, Z_true, Z_obs, mask = generate_campbell2d_grid(
        nx=20, ny=20, missing_pattern="center_block"
    )

    out = load_2d_data(
        gdf,
        value_col="value",
        z_value=None,
        test_size=0.2,
        seed=123,
    )

    # Full grid shape
    assert out["x_2d"].shape == X.shape
    assert out["y_2d"].shape == Y.shape

    # Standardized arrays
    assert out["X_train_stdized"].ndim == 2
    assert out["Y_train_stdized"].ndim == 2

    # nan mask preserved
    assert np.array_equal(out["nan_mask"], np.isnan(Z_obs))


def test_load_2d_data_train_val_split():
    gdf, *_ = generate_campbell2d_grid(nx=10, ny=10)

    out = load_2d_data(
        gdf,
        value_col="value",
        z_value=None,
        test_size=0.25,
        seed=42,
    )

    n_total = (~np.isnan(out["Y_2d"])).sum()
    n_train = out["X_train"].shape[0]
    n_val   = out["X_val"].shape[0]

    assert n_train + n_val == n_total
    assert abs(n_val / n_total - 0.25) < 0.1  # allow sampling variance

