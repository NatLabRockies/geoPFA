#!/usr/bin/env python3
"""
exptrapolation.py - Gaussian Process Regression (GPR) for extrapolating values on a 2D grid over multiple Z slices.

This module provides functions for:
- **I/O and Data Handling**: reading data into GeoDataFrames and saving/loading GPy models.
- **Pre-processing**: preparing data for modeling (extracting slices, standardizing).
- **Modeling**: building Gaussian Process models (with composite kernels) and making predictions.
- **Validation/Evaluation**: assessing model performance and diagnostic tests on residuals.
- **Visualization**: plotting residuals, uncertainty, and comparison of predictions vs true values.

"""

import os
import sys
import logging
from pathlib import Path
from functools import reduce
import operator
import warnings

import geopandas as gpd
import GPy
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.stats import shapiro, normaltest, jarque_bera, levene
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from tqdm import trange
from pprint import pprint

# Suppress specific warnings or verbose logs (e.g., from paramz transformation in GPy)
logging.getLogger("paramz.transformations").setLevel(logging.ERROR)

# === I/O and Data Handling Functions ===


def save_gpy_model(model: "GPy.core.GP", filepath: str) -> None:
    """
    Save a GPy model to disk using joblib.

    Parameters
    ----------
    model : GPy.core.GP
        Trained GPy model object to save.
    filepath : str
        Destination file path (e.g., ``'my_model.joblib'``). Parent
        directories are created automatically.

    Returns
    -------
    None
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model with joblib
    joblib.dump(model, filepath)

    # Preserve original behavior: print confirmation
    print(f"GPy model saved to: {filepath}")


def load_gpy_model(filepath: str) -> "GPy.core.GP":
    """
    Load a serialized GPy model from disk using joblib.

    Parameters
    ----------
    filepath : str
        File path to the saved ``.joblib`` GPy model.

    Returns
    -------
    GPy.core.GP
        Loaded GPy model instance.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    file_path = Path(filepath)

    if not file_path.is_file():
        raise FileNotFoundError(f"No file found at {filepath}")

    # Load model
    model = joblib.load(filepath)

    # Preserve original behavior: print confirmation
    print(f"GPy model loaded from: {filepath}")

    return model


def prepare_slice(
    gdf: gpd.GeoDataFrame,
    *,
    value_col: str,
    z_value: float | None = None,
    z_tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Prepare a tidy slice of (x, y, value) for an optional z-level filter.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing either explicit ``x``/``y`` columns
        or a geometry column with Point coordinates. May also contain a
        ``z`` column if filtering by height is required.
    value_col : str
        Name of the column whose values will be extracted and renamed
        to ``value``.
    z_value : float or None, optional
        If provided, only rows satisfying ``|z - z_value| <= z_tol`` are kept.
    z_tol : float, optional
        Absolute tolerance for matching z-values.

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame with columns ``['x', 'y', 'value']``.

    Raises
    ------
    ValueError
        If z filtering is requested but the GeoDataFrame lacks a ``z`` column.
        If neither explicit x/y nor geometry is present.
    """
    # ------------------------------------------------------------------
    # Filter by z-value if provided
    # ------------------------------------------------------------------
    if z_value is not None:
        if "z" not in gdf.columns:
            raise ValueError(
                "`z_value` was provided but no 'z' column exists in gdf."
            )
        mask = np.abs(gdf["z"] - z_value) <= z_tol
        df_slice = gdf.loc[mask].copy()
    else:
        df_slice = gdf.copy()

    # ------------------------------------------------------------------
    # Ensure x and y exist
    # ------------------------------------------------------------------
    if "x" not in df_slice.columns or "y" not in df_slice.columns:
        if "geometry" not in df_slice.columns:
            raise ValueError(
                "GeoDataFrame must contain x/y or geometry for coordinate extraction."
            )
        df_slice["x"] = df_slice.geometry.x
        df_slice["y"] = df_slice.geometry.y

    # ------------------------------------------------------------------
    # Final tidy result
    # ------------------------------------------------------------------
    return df_slice[["x", "y", value_col]].rename(columns={value_col: "value"})


def grid_from_tidy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert tidy (x, y, value) data into a rectangular grid and aligned meshgrids.

    Parameters
    ----------
    df : pandas.DataFrame
        A tidy DataFrame with columns ``'x'``, ``'y'``, and ``'value'``.

    Returns
    -------
    grid : pandas.DataFrame
        Pivoted grid of shape ``(ny, nx)``, with sorted ``y`` as rows and
        sorted ``x`` as columns.
    xs : numpy.ndarray of shape (nx,)
        Sorted unique ``x`` coordinates.
    ys : numpy.ndarray of shape (ny,)
        Sorted unique ``y`` coordinates.
    x_2d : numpy.ndarray of shape (ny, nx)
        2D meshgrid of ``x`` coordinates.
    y_2d : numpy.ndarray of shape (ny, nx)
        2D meshgrid of ``y`` coordinates.

    Notes
    -----
    - This function assumes a rectangular lattice is desired, and therefore
      sorts x and y independently.
    - Unobserved (x, y) pairs appear as NaN in the returned grid.
    """
    # Sorted coordinate axes
    xs = np.sort(df["x"].unique())
    ys = np.sort(df["y"].unique())

    # Pivot into a grid and ensure sorted indexing
    grid = df.pivot_table(index="y", columns="x", values="value")
    grid = grid.reindex(index=ys, columns=xs)

    # Construct 2D meshgrid arrays explicitly
    x_2d = np.tile(xs, (len(ys), 1))
    y_2d = np.tile(ys.reshape(-1, 1), (1, len(xs)))

    return grid, xs, ys, x_2d, y_2d


def standardize_xy(
    X_train: np.ndarray,
    X_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize coordinate arrays using the mean and standard deviation computed
    from the training coordinates.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, 2)
        Training coordinate array.
    X_full : ndarray of shape (n_full, 2)
        Full coordinate array to be standardized using training statistics.

    Returns
    -------
    X_train_std : ndarray of shape (n_train, 2)
        Standardized training coordinates.
    X_full_std : ndarray of shape (n_full, 2)
        Standardized full-grid coordinates.
    mean : ndarray of shape (2,)
        Per-dimension mean of the training data.
    std : ndarray of shape (2,)
        Per-dimension standard deviation of the training data.

    Notes
    -----
    - This function computes statistics (mean, std) **only** from the training
      array `X_train`, which ensures proper scaling when evaluating generalization.
    """
    # Compute scaling statistics from training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    # Standardize training and full-grid coordinates
    X_train_std = (X_train - mean) / std
    X_full_std = (X_full - mean) / std

    return X_train_std, X_full_std, mean, std


def load_2d_data(  # noqa PLR0913, PLR0914
    gdf,
    *,
    value_col,
    z_value=None,
    z_tol=1e-6,
    test_size=0.2,
    seed=42,
):
    """
    Prepare all arrays needed for 2D Gaussian process interpolation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing coordinates and values.
    value_col : str
        Column name containing the target variable.
    z_value : float, optional
        Z slice to filter. If ``None``, no Z filtering is applied.
    z_tol : float, optional
        Tolerance for selecting points around ``z_value``.
    test_size : float, optional
        Fraction of non-missing data allocated to validation.
    seed : int, optional
        Random seed for the train/validation split.

    Returns
    -------
    dict
        Dictionary with the same keys as the original implementation:
        includes standardized XY, grids, masks, pivoted arrays, and stats.
    """
    # ---------------------------------------------------------
    # Slice preparation
    # ---------------------------------------------------------
    df_slice = prepare_slice(
        gdf,
        value_col=value_col,
        z_value=z_value,
        z_tol=z_tol,
    )

    # ---------------------------------------------------------
    # Pivot tidy data into rectangular grid
    # ---------------------------------------------------------
    df_pivot, _xs, _ys, x_2d, y_2d = grid_from_tidy(df_slice)
    Y_2d = df_pivot.to_numpy()
    nan_mask = np.isnan(Y_2d)

    # ---------------------------------------------------------
    # Extract non-missing training samples
    # ---------------------------------------------------------
    df_no_nan = df_slice.dropna(subset=["value"])
    sample_x = df_no_nan["x"].to_numpy()
    sample_y = df_no_nan["y"].to_numpy()
    sample_Y = df_no_nan["value"].to_numpy()

    X = np.column_stack([sample_x, sample_y])
    Y = sample_Y.reshape(-1, 1)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    # ---------------------------------------------------------
    # Standardize coordinates
    # ---------------------------------------------------------
    X_full = np.column_stack([x_2d.ravel(), y_2d.ravel()])
    X_train_stdized, X_full_stdized, X_mean, X_std = standardize_xy(
        X_train,
        X_full,
    )
    X_val_stdized = (X_val - X_mean) / X_std

    # Full grid standardized (reshaped)
    X_full_stdized_2d = X_full_stdized.reshape(*x_2d.shape, 2)

    # Missing positions / standardized
    X_missing = np.column_stack([x_2d[nan_mask], y_2d[nan_mask]])
    X_missing_stdized = X_full_stdized[nan_mask.ravel()]

    # ---------------------------------------------------------
    # Standardize Y
    # ---------------------------------------------------------
    Y_train_mean = Y_train.mean(axis=0)
    Y_train_std = Y_train.std(axis=0)

    Y_train_stdized = (Y_train - Y_train_mean) / Y_train_std
    Y_val_stdized = (Y_val - Y_train_mean) / Y_train_std

    # Standardized grid values
    Y_2d_stdized = (Y_2d - Y_train_mean) / Y_train_std

    # Also standardize original 1D x-y lists
    x = df_slice["x"].to_numpy()
    y = df_slice["y"].to_numpy()
    x_stdized = (x - X_mean[0]) / X_std[0]
    y_stdized = (y - X_mean[1]) / X_std[1]

    # ---------------------------------------------------------
    # Return dictionary in original structure
    # ---------------------------------------------------------
    return {
        "x_2d": x_2d,
        "y_2d": y_2d,
        "nan_mask": nan_mask,
        "X_missing_stdized": X_missing_stdized,
        "x": x,
        "y": y,
        "x_stdized": x_stdized,
        "y_stdized": y_stdized,
        "Y_2d": Y_2d,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_train_stdized": X_train_stdized,
        "Y_train_stdized": Y_train_stdized,
        "X_val_stdized": X_val_stdized,
        "Y_val_stdized": Y_val_stdized,
        "X_train_mean": X_mean,
        "X_train_std": X_std,
        "Y_train_mean": Y_train_mean,
        "Y_train_std": Y_train_std,
        "df_pivot": df_pivot,
        "df_slice": df_slice,
        "X_missing": X_missing,
    }


def drop_z_from_geometry(
    gdf: gpd.GeoDataFrame,
    geom_col: str = "geometry",
) -> gpd.GeoDataFrame:
    """
    Remove Z coordinates from 3D Point geometries in a GeoDataFrame,
    converting them into standard 2D Points.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame whose geometry column may contain 3D Points.
    geom_col : str, optional
        Name of the geometry column containing shapely Point geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        A new GeoDataFrame where all Point geometries are guaranteed to be 2D.

    Notes
    -----
    - Rows with geometries that are already 2D are left unchanged.
    - Only Point geometries are supported; behavior for LineString/Polygon
      geometries is not modified from the original implementation.
    """

    def _drop_z(geom: BaseGeometry) -> BaseGeometry:
        if hasattr(geom, "has_z") and geom.has_z:
            return Point(geom.x, geom.y)
        return geom

    gdf_out = gdf.copy()
    gdf_out[geom_col] = gdf_out[geom_col].apply(_drop_z)
    return gdf_out


def estimate_variance(Y: ArrayLike) -> float:
    """
    Estimate the unbiased sample variance of an array.

    Parameters
    ----------
    Y : array_like
        Observed values (1D or 2D). If 2D, the array is flattened before
        variance computation.

    Returns
    -------
    float
        Unbiased sample variance (using ``ddof=1``).
    """
    Y_arr = np.asarray(Y).flatten()
    return float(np.var(Y_arr, ddof=1))


def recommend_likelihood_params(
    Y: np.ndarray,
    Y_var: float | None = None,
) -> dict:
    """
    Recommend initialization parameters and a prior for the Gaussian likelihood
    (observation-noise variance) used in GPy models.

    Parameters
    ----------
    Y : numpy.ndarray
        Target values of shape (N, 1) or (N,). Flattening is handled internally.
    Y_var : float or None, optional
        Variance of the target values. If None, it is estimated using
        :func:`estimate_variance`.

    Returns
    -------
    dict
        A dictionary containing:
            - ``'init'`` : float
                Initial noise variance estimate.
            - ``'prior'`` : GPy.priors.Gamma
                Gamma prior placed on the noise variance.
            - ``'bounds'`` : tuple(float, float)
                Lower and upper bounds for the noise variance parameter.

        The dictionary is packaged under the key ``'variance'`` so that callers
        can directly configure ``model.Gaussian_noise.variance``.

    Notes
    -----
    - The initial noise variance is set to approximately 1% of the variance of Y.
    - A Gamma prior is used to bias the optimization toward small but positive
      values, reflecting typical assumptions about measurement noise.
    """
    # Estimate variance if not provided
    if Y_var is None:
        Y_var = estimate_variance(Y)

    # 1% of data variance as initial noise level
    noise_level = Y_var * 0.01

    # Gamma prior for noise variance
    variance_prior = GPy.priors.Gamma(2.0, noise_level / 2.0)

    return {
        "variance": {
            "init": noise_level,
            "prior": variance_prior,
            "bounds": (1e-5, Y_var * 0.1),
        }
    }


def compute_global_radius(X_std: np.ndarray) -> float:
    """
    Compute a global spatial radius for standardized 2D coordinates.

    This is used to construct stable, dataset-independent kernel
    lengthscale bounds by estimating the largest half-extent of the
    standardized domain.

    Parameters
    ----------
    X_std : numpy.ndarray of shape (N, 2)
        Standardized training coordinates. Each row is [x, y].

    Returns
    -------
    float
        The global radius of the dataset in standardized space, defined as
        half of the larger of the x-range or y-range.

    Notes
    -----
    - The input must already be standardized (e.g., via ``standardize_xy``).
    - This radius is used to derive lower and upper bounds for ARD lengthscales.
    """
    x_min, x_max = X_std[:, 0].min(), X_std[:, 0].max()
    y_min, y_max = X_std[:, 1].min(), X_std[:, 1].max()

    R = 0.5 * max(x_max - x_min, y_max - y_min)
    return float(R)


def compute_lengthscale_bounds_from_global_radius(
    R: float,
    lower_frac: float = 0.02,
    upper_frac: float = 0.20,
) -> tuple[float, float]:
    """
    Compute dataset-agnostic kernel lengthscale bounds based on the global
    spatial radius of the standardized coordinate domain.

    Parameters
    ----------
    R : float
        Global radius computed from :func:`compute_global_radius`.
    lower_frac : float, optional
        Fraction of ``R`` used to set the minimum allowed lengthscale.
    upper_frac : float, optional
        Fraction of ``R`` used to set the maximum allowed lengthscale.

    Returns
    -------
    (float, float)
        A tuple ``(lower_bound, upper_bound)`` specifying the recommended
        bounds for ARD lengthscales in the GP kernel.

    Notes
    -----
    - These scaled bounds help stabilize GP optimization across different
      datasets by relating lengthscale limits to domain size.
    """
    lower = lower_frac * R
    upper = upper_frac * R
    return lower, upper


def build_rbf_kernel_global(
    X_stdized: np.ndarray,
    Y_stdized: np.ndarray,
    input_dim: int = 2,
    lower_frac: float = 0.02,
    upper_frac: float = 0.20,
) -> tuple[GPy.kern.RBF, dict]:
    """
    Construct an RBF kernel whose ARD lengthscale bounds are derived from the
    global radius of the standardized training coordinates.

    Parameters
    ----------
    X_stdized : numpy.ndarray of shape (N, 2)
        Standardized training coordinates.
    Y_stdized : numpy.ndarray of shape (N, 1)
        Standardized training targets.
    input_dim : int, optional
        Dimensionality of the input space. Default is 2.
    lower_frac : float, optional
        Fraction of the global radius used as the lower lengthscale bound.
    upper_frac : float, optional
        Fraction of the global radius used as the upper lengthscale bound.

    Returns
    -------
    (GPy.kern.RBF, dict)
        The constructed RBF kernel and a dictionary describing the applied
        constraints and initialization values.

    Notes
    -----
    - The global radius stabilizes ARD lengthscale bounds across datasets
      with different spatial extents.
    - Variance is initialized based on the variance of `Y_stdized` and
      constrained with a Gamma prior.
    """
    # ------------------------------------------------------------------
    # Compute global-radius-based lengthscale bounds
    # ------------------------------------------------------------------
    R = compute_global_radius(X_stdized)
    lower, upper = compute_lengthscale_bounds_from_global_radius(
        R, lower_frac, upper_frac
    )

    # ------------------------------------------------------------------
    # Create RBF kernel with ARD
    # ------------------------------------------------------------------
    kern = GPy.kern.RBF(input_dim=input_dim, ARD=True)

    # ------------------------------------------------------------------
    # Kernel variance setup
    # ------------------------------------------------------------------
    Y_var = float(np.var(Y_stdized))
    kern.variance[:] = max(Y_var, 1e-6)
    kern.variance.set_prior(GPy.priors.Gamma(2.0, Y_var / 2.0))
    kern.variance.constrain_bounded(Y_var * 0.01, Y_var * 10.0)

    # ------------------------------------------------------------------
    # Lengthscale initialization
    # ------------------------------------------------------------------
    init_ls = 0.5 * (lower + upper)
    kern.lengthscale[:] = np.full(input_dim, init_ls)
    kern.lengthscale.constrain_bounded(lower, upper)

    # ------------------------------------------------------------------
    # Record applied constraints
    # ------------------------------------------------------------------
    constraints = {
        "R": R,
        "ls_lower": lower,
        "ls_upper": upper,
        "ls_init": init_ls,
        "var": kern.variance.values.copy(),  # noqa: PD011
    }

    return kern, constraints


def build_matern32_kernel_global(
    X_stdized: np.ndarray,
    Y_stdized: np.ndarray,
    input_dim: int = 2,
    lower_frac: float = 0.02,
    upper_frac: float = 0.20,
) -> tuple[GPy.kern.Matern32, dict]:
    """
    Construct a Matern 3/2 kernel with ARD lengthscales constrained by the
    global radius of the dataset in standardized space.

    Parameters
    ----------
    X_stdized : numpy.ndarray of shape (N, 2)
        Standardized training coordinates.
    Y_stdized : numpy.ndarray of shape (N, 1)
        Standardized training targets.
    input_dim : int, optional
        Dimensionality of the input space. Default is 2.
    lower_frac : float, optional
        Fraction of the global radius used for the lower lengthscale bound.
    upper_frac : float, optional
        Fraction of the global radius used for the upper lengthscale bound.

    Returns
    -------
    (GPy.kern.Matern32, dict)
        The Matern 3/2 kernel and a dictionary of initialization
        values and bound settings.

    Notes
    -----
    - Variance is initialized to half the variance of Y and given
      a Gamma prior, matching the original implementation.
    - Lengthscales are constrained to dataset-scaled bounds.
    """
    # ------------------------------------------------------------------
    # Global-radius lengthscale bounds
    # ------------------------------------------------------------------
    R = compute_global_radius(X_stdized)
    lower, upper = compute_lengthscale_bounds_from_global_radius(
        R, lower_frac, upper_frac
    )

    # ------------------------------------------------------------------
    # Base Matern32 kernel
    # ------------------------------------------------------------------
    kern = GPy.kern.Matern32(input_dim=input_dim, ARD=True)

    # ------------------------------------------------------------------
    # Variance initialization and prior
    # ------------------------------------------------------------------
    Y_var = float(np.var(Y_stdized))
    kern.variance[:] = max(0.5 * Y_var, 1e-6)
    kern.variance.set_prior(GPy.priors.Gamma(2.0, (0.5 * Y_var) / 2.0))
    kern.variance.constrain_bounded(Y_var * 0.01, Y_var * 10.0)

    # ------------------------------------------------------------------
    # Lengthscale initialization from midpoint of bounds
    # ------------------------------------------------------------------
    init_ls = 0.5 * (lower + upper)
    kern.lengthscale[:] = np.full(input_dim, init_ls)
    kern.lengthscale.constrain_bounded(lower, upper)

    # ------------------------------------------------------------------
    # Export constraint info
    # ------------------------------------------------------------------
    constraints = {
        "R": R,
        "ls_lower": lower,
        "ls_upper": upper,
        "ls_init": init_ls,
        "var": kern.variance.values.copy(),  # noqa: PD011
    }

    return kern, constraints


def build_combined_kernel(  # noqa: PLR0913, PLR0917
    X_stdized: np.ndarray,
    Y_stdized: np.ndarray,
    use_matern: bool = True,
    use_rbf: bool = True,
    bias: bool = True,
    white: bool = True,
    longscale: bool = True,
    lower_frac: float = 0.02,
    upper_frac: float = 0.20,
) -> tuple[GPy.kern.Kern, dict]:
    """
    Construct a composite kernel consisting of optional components:
    RBF, Matern 3/2, long-scale RBF, Bias, and White noise kernels.

    Parameters
    ----------
    X_stdized : numpy.ndarray
        Standardized training coordinates.
    Y_stdized : numpy.ndarray
        Standardized training targets.
    use_matern : bool, optional
        Whether to include a Matern 3/2 component.
    use_rbf : bool, optional
        Whether to include a short/medium-scale RBF component.
    bias : bool, optional
        Whether to include a constant Bias kernel.
    white : bool, optional
        Whether to include a White noise kernel.
    longscale : bool, optional
        Whether to include a very smooth long-scale RBF background kernel.
    lower_frac : float, optional
        Lower bound fraction (of global radius) for ARD lengthscales.
    upper_frac : float, optional
        Upper bound fraction (of global radius) for ARD lengthscales.

    Returns
    -------
    kernel : GPy.kern.Kern
        Combined kernel created by summing enabled components.
    info : dict
        Diagnostic information for each component, including learned bounds
        and initialization values.

    Notes
    -----
    - Behavior is identical to the original implementation.
    - The long-scale RBF kernel is fixed to very smooth lengthscales and
      tiny variance, acting as a gentle background component.
    """
    parts: list[GPy.kern.Kern] = []
    info: dict = {}

    # ------------------------------------------------------------------
    # Global radius (useful for long-scale kernel and subkernels)
    # ------------------------------------------------------------------
    R = compute_global_radius(X_stdized)
    info["global_radius"] = R

    # ------------------------------------------------------------------
    # RBF kernel
    # ------------------------------------------------------------------
    if use_rbf:
        rbf, rbf_info = build_rbf_kernel_global(
            X_stdized,
            Y_stdized,
            lower_frac=lower_frac,
            upper_frac=upper_frac,
        )
        parts.append(rbf)
        info["rbf"] = rbf_info

    # ------------------------------------------------------------------
    # Matern 3/2 kernel
    # ------------------------------------------------------------------
    if use_matern:
        m32, m32_info = build_matern32_kernel_global(
            X_stdized,
            Y_stdized,
            lower_frac=lower_frac,
            upper_frac=upper_frac,
        )
        parts.append(m32)
        info["matern32"] = m32_info

    # ------------------------------------------------------------------
    # Long-scale RBF (fixed ultra-smooth component)
    # ------------------------------------------------------------------
    if longscale:
        K_long = GPy.kern.RBF(input_dim=2, ARD=True)

        # Very small fixed variance
        K_long.variance[:] = 0.05
        K_long.variance.fix()

        # Very large fixed lengthscale
        long_ls = 2.0 * R
        K_long.lengthscale[:] = np.full(2, long_ls)
        K_long.lengthscale.fix()

        parts.append(K_long)
        info["longscale"] = {"variance": 0.05, "lengthscale": long_ls}

    # ------------------------------------------------------------------
    # Bias kernel: constant offset
    # ------------------------------------------------------------------
    if bias:
        b = GPy.kern.Bias(input_dim=2)
        b.variance.fix(1.0)
        parts.append(b)
        info["bias"] = {"variance": 1.0}

    # ------------------------------------------------------------------
    # White noise kernel
    # ------------------------------------------------------------------
    if white:
        w = GPy.kern.White(input_dim=2, variance=1e-5)
        w.variance.fix()
        parts.append(w)
        info["white"] = {"variance": 1e-5}

    # ------------------------------------------------------------------
    # Combine components
    # ------------------------------------------------------------------
    if not parts:
        raise ValueError("No kernel components were enabled.")

    if len(parts) == 1:
        return parts[0], info

    kernel = reduce(operator.add, parts)
    return kernel, info


def get_gaussian_noise_bounds(
    lik_params: dict,
) -> dict:
    """
    Extract Gaussian noise variance bounds from a likelihood-parameter dictionary.

    Parameters
    ----------
    lik_params : dict
        Dictionary returned by :func:`recommend_likelihood_params`, containing
        keys such as ``'variance' : {'init', 'prior', 'bounds'}``.

    Returns
    -------
    dict
        Dictionary containing:
            - ``'value'`` : float
                Initial noise variance.
            - ``'lower'`` : float
                Lower bound on the noise variance.
            - ``'upper'`` : float
                Upper bound on the noise variance.

        The returned structure mirrors the shape used for kernel constraint
        introspection throughout the codebase.
    """
    return {
        "variance": {
            "value": lik_params["variance"]["init"],
            "lower": lik_params["variance"]["bounds"][0],
            "upper": lik_params["variance"]["bounds"][1],
        }
    }


def build_and_fit_gp(  # noqa: PLR0913, PLR0917
    X_train_stdized: np.ndarray,
    Y_train_stdized: np.ndarray,
    optimize_restarts: int = 0,
    verbose: bool = False,
    save_path: str | None = None,
    n_inducing: int = 300,
    lower_frac: float = 0.02,
    upper_frac: float = 0.20,
) -> tuple[GPy.models.SparseGPRegression, dict]:
    """
    Build and train a Sparse Gaussian Process regression model using a globally
    stabilized multi-kernel combination (RBF + Matern32 + optional Bias/White).

    This method improves extrapolation stability by:
        - Constraining ARD lengthscales based on global radius,
        - Distributing inducing points across the domain via KMeans,
        - Anchoring the prior with a constant-mean mapping,
        - Adding bias and white kernels to reduce drift,
        - Using a Gamma prior on the Gaussian likelihood noise variance.

    Parameters
    ----------
    X_train_stdized : numpy.ndarray of shape (N, D)
        Standardized training coordinates.
    Y_train_stdized : numpy.ndarray of shape (N, 1)
        Standardized training targets.
    optimize_restarts : int, optional
        Number of multi-start optimization attempts. Default is 0.
    verbose : bool, optional
        Whether to print optimization diagnostics.
    save_path : str or None, optional
        If provided, the model is saved to this path via joblib.
    n_inducing : int, optional
        Target number of inducing points for SparseGPRegression.
    lower_frac : float, optional
        Fraction of global radius used for lower lengthscale bounds.
    upper_frac : float, optional
        Fraction of global radius used for upper lengthscale bounds.

    Returns
    -------
    model : GPy.models.SparseGPRegression
        Trained sparse Gaussian Process model.
    constraint_info : dict
        Dictionary containing kernel and likelihood constraint diagnostics.
    """
    # ----------------------------------------------------------------------
    # 1. Build combined kernel using global-radius constraints
    # ----------------------------------------------------------------------
    kernel, kernel_info = build_combined_kernel(
        X_train_stdized,
        Y_train_stdized,
        use_matern=True,
        use_rbf=True,
        bias=True,
        white=True,
        lower_frac=lower_frac,
        upper_frac=upper_frac,
    )

    # ----------------------------------------------------------------------
    # 2. Constant mean function (frozen)
    # ----------------------------------------------------------------------
    mean_func = GPy.mappings.Constant(
        input_dim=X_train_stdized.shape[1],
        output_dim=1,
    )
    mean_func.C[:] = float(Y_train_stdized.mean())
    mean_func.C.fix()

    # ----------------------------------------------------------------------
    # 3. Inducing point initialization via KMeans
    # ----------------------------------------------------------------------
    N = X_train_stdized.shape[0]
    M = min(n_inducing, max(20, N // 10))

    km = KMeans(n_clusters=M, n_init="auto")
    Z = km.fit(X_train_stdized).cluster_centers_

    # ----------------------------------------------------------------------
    # 4. Construct sparse GP model
    # ----------------------------------------------------------------------
    model = GPy.models.SparseGPRegression(
        X=X_train_stdized,
        Y=Y_train_stdized,
        kernel=kernel,
        mean_function=mean_func,
        Z=Z,
        normalizer=False,
    )

    # ----------------------------------------------------------------------
    # 5. Gaussian likelihood setup
    # ----------------------------------------------------------------------
    Y_var = float(np.var(Y_train_stdized))
    lik_params = recommend_likelihood_params(Y_train_stdized, Y_var=Y_var)

    model.Gaussian_noise.variance = lik_params["variance"]["init"]
    model.Gaussian_noise.variance.constrain_bounded(
        *lik_params["variance"]["bounds"]
    )
    model.Gaussian_noise.variance.set_prior(lik_params["variance"]["prior"])

    constraint_info = {
        "kernel": kernel_info,
        "Gaussian_noise": get_gaussian_noise_bounds(lik_params),
    }

    # ----------------------------------------------------------------------
    # 6. Optimize
    # ----------------------------------------------------------------------
    model.optimize(messages=verbose, max_iters=2000)

    if optimize_restarts > 0:
        model.optimize_restarts(
            num_restarts=optimize_restarts,
            verbose=verbose,
            robust=True,
            parallel=False,
        )

    # ----------------------------------------------------------------------
    # 7. Optional save to disk
    # ----------------------------------------------------------------------
    if save_path:
        save_gpy_model(model, save_path)

    # ----------------------------------------------------------------------
    # 8. Diagnostics
    # ----------------------------------------------------------------------
    if verbose:
        print("\n=== Kernel Lengthscale Diagnostics ===")
        for part in model.kern.parts:
            if hasattr(part, "lengthscale"):
                print(
                    f"{part.name}: lengthscales = {np.array(part.lengthscale.values)}"
                )

        print("\n=== Inducing Point Ranges ===")
        print(
            "X range:",
            model.Z[:, 0].min(),
            "→",
            model.Z[:, 0].max(),
            "| Y range:",
            model.Z[:, 1].min(),
            "→",
            model.Z[:, 1].max(),
        )

    return model, constraint_info


def get_predictions(
    model: GPy.models.GPRegression,
    X: np.ndarray,
    kvals_df: pd.DataFrame | dict | None = None,
    Y_mean: float | None = None,
    Y_std: float | None = None,
):
    """
    Generate GP predictions, optionally converting back to original Y-units and
    optionally reshaping predictions into a 2D grid defined by x/y coordinates.

    Parameters
    ----------
    model : GPy.models.GPRegression
        Trained GP regression model.
    X : numpy.ndarray of shape (N, D)
        Input features in the same standardized space used during training.
    kvals_df : pandas.DataFrame or dict, optional
        Must contain 'x' and 'y'. If provided, predictions are reshaped into a
        pivoted 2D grid of shape (len(y_unique), len(x_unique)).
    Y_mean : float or None, optional
        Mean of Y from the training dataset (for de-standardizing predictions).
    Y_std : float or None, optional
        Standard deviation of Y from the training dataset.

    Returns
    -------
    numpy.ndarray or tuple
        If ``kvals_df`` is None:
            (Y_pred_flat, Y_std_flat)
        If ``kvals_df`` is provided:
            (Y_pred_grid, Y_std_grid, x_unique, y_unique)

        Where:
            - ``Y_pred_flat`` : shape (N,)
            - ``Y_std_flat``  : shape (N,)
            - ``Y_pred_grid`` : shape (ny, nx)
            - ``Y_std_grid``  : shape (ny, nx)
            - ``x_unique``    : sorted unique x coordinates (from pivot index)
            - ``y_unique``    : sorted unique y coordinates (from grid index)

    Notes
    -----
    - This function reproduces the original behavior precisely:
        * No additional sorting is applied beyond pandas pivot ordering.
        * De-standardization (mean/std) is optional and only applied if both
          parameters are provided.
    """
    # ------------------------------------------------------------------
    # 1. Predict in standardized space
    # ------------------------------------------------------------------
    Y_pred, Y_var = model.predict(X)  # shapes: (N, 1)

    # ------------------------------------------------------------------
    # 2. Optional de-standardization
    # ------------------------------------------------------------------
    if Y_mean is not None and Y_std is not None:
        Y_pred = Y_pred * Y_std + Y_mean
        Y_var = (Y_std**2) * Y_var

    # Flatten outputs
    Y_pred_flat = Y_pred.ravel()
    Y_std_flat = np.sqrt(Y_var).ravel()

    # ------------------------------------------------------------------
    # 3. If no coordinate grid provided, return flat arrays
    # ------------------------------------------------------------------
    if kvals_df is None:
        return Y_pred_flat, Y_std_flat

    # ------------------------------------------------------------------
    # 4. Grid reshape using x/y coordinates supplied by user
    # ------------------------------------------------------------------
    coords = pd.DataFrame({"x": kvals_df["x"], "y": kvals_df["y"]})
    coords["pred"] = Y_pred_flat
    coords["std"] = Y_std_flat

    grid_pred = coords.pivot_table(index="y", columns="x", values="pred")
    grid_std = coords.pivot_table(index="y", columns="x", values="std")

    Y_pred_grid = grid_pred.to_numpy()
    Y_std_grid = grid_std.to_numpy()
    x_unique = grid_pred.columns.to_numpy()
    y_unique = grid_pred.index.to_numpy()

    return Y_pred_grid, Y_std_grid, x_unique, y_unique


def check_param_limits_hit_from_constraints(
    model: GPy.core.GP,
    constraints: dict,
) -> list[tuple]:
    """
    Identify kernel parameters whose optimized values lie at or extremely
    near their constrained lower or upper bounds.

    Parameters
    ----------
    model : GPy.core.GP
        Trained GPy model. Must expose ``parameter_names()`` and allow
        indexing parameters via ``model[name]``.
    constraints : dict
        Dictionary describing parameter bounds (e.g., from ``build_and_fit_gp`` or
        ``build_combined_kernel``). Expected format:
            {
                "rbf": {
                    "variance": {"value": ..., "lower": ..., "upper": ...},
                    "lengthscale": {...},
                },
                "matern32": { ... },
                "Gaussian_noise": {
                    "variance": {"value": ..., "lower": ..., "upper": ...}
                },
                ...
            }

    Returns
    -------
    list of tuple
        A list of entries:
            (parameter_name_in_model, current_value_array, (lower, upper))
        Each item corresponds to a parameter whose value is at/near its
        bounds (|value - bound| <= 1e-6). Returns an empty list if none.

    Notes
    -----
    - Uses substring-prefix mapping (e.g., "rbf" → "sum.rbf") to resolve to
      GPy's internal parameter names.
    - Bounds must be present in ``constraints`` for comparison.
    """
    param_limits_hit: list[tuple] = []

    # ------------------------------------------------------------------
    # Map short-names to the kernel prefixes used in GPy parameter names
    # ------------------------------------------------------------------
    kernel_prefix_map = {
        "rbf": "sum.rbf",
        "Mat32": "sum.Mat32",
        "white": "sum.white",
        "Gaussian_noise": "Gaussian_noise",
    }

    # ------------------------------------------------------------------
    # Make lookup table of model parameter names (case-insensitive)
    # ------------------------------------------------------------------
    model_params = {p.lower(): p for p in model.parameter_names()}

    # ------------------------------------------------------------------
    # Iterate over constraint groups
    # ------------------------------------------------------------------
    for short_name, params in constraints.items():
        prefix = kernel_prefix_map.get(short_name)
        if prefix is None:
            continue

        # Each param inside the constraint entry (e.g., variance, lengthscale)
        for param_name, info in params.items():
            full_name_key = f"{prefix}.{param_name}".lower()
            if full_name_key not in model_params:
                continue

            # Extract parameter from model
            param_value = model[model_params[full_name_key]].values  # noqa: PD011

            lower = info.get("lower")
            upper = info.get("upper")
            if lower is None or upper is None:
                continue

            # ------------------------------------------------------------------
            # Check if any element lies extremely close to a bound
            # ------------------------------------------------------------------
            at_lower = np.any(np.isclose(param_value, lower, atol=1e-6))
            at_upper = np.any(np.isclose(param_value, upper, atol=1e-6))

            if at_lower or at_upper:
                param_limits_hit.append(
                    (
                        model_params[full_name_key],
                        param_value.copy(),
                        (lower, upper),
                    )
                )

    return param_limits_hit


def assess_gp_model_fit(
    model: GPy.models.GPRegression,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    Y_pred_std: np.ndarray,
    constraints: dict,
) -> dict:
    """
    Compute regression performance metrics and model diagnostics for a trained
    Gaussian Process model.

    Parameters
    ----------
    model : GPy.models.GPRegression
        Trained GP regression model.
    Y_true : numpy.ndarray
        True observed target values. Flattening is handled internally.
    Y_pred : numpy.ndarray
        Predicted mean values from the GP model (same shape as Y_true).
    Y_pred_std : numpy.ndarray
        Predicted standard deviations for the GP predictions.
    constraints : dict
        Constraint information returned during GP construction, used to
        check whether any hyperparameters lie on their bounds.

    Returns
    -------
    dict
        A dictionary with the following entries:

        - ``RMSE`` : float
            Root Mean Square Error.

        - ``R2`` : float
            Coefficient of determination.

        - ``MAE`` : float
            Mean Absolute Error.

        - ``Coverage_95`` : float
            Fraction of observations lying inside ±2 sigma predictive intervals.

        - ``LogLikelihood`` : float
            GP model log marginal likelihood.

        - ``AIC`` : float
            Akaike Information Criterion ``2*k - 2*logL``.

        - ``BIC`` : float
            Bayesian Information Criterion ``k*ln(n) - 2*logL``.

        - ``Params_at_bounds`` : list of tuple
            Parameters effectively pinned at their allowable bounds.

    Notes
    -----
    - No transformations or standardization adjustments are performed here.
      Inputs should already be in the desired scale.
    - The ±2 sigma coverage statistic provides a simple approximate 95% interval
      assessment for GP predictive uncertainty.
    """
    # Flatten inputs (behavior preserved)
    Y_true = np.asarray(Y_true).ravel()
    Y_pred = np.asarray(Y_pred).ravel()
    Y_pred_std = np.asarray(Y_pred_std).ravel()

    # ------------------------------------------------------------------
    # Basic regression metrics
    # ------------------------------------------------------------------
    rmse = float(np.sqrt(mean_squared_error(Y_true, Y_pred)))
    r2 = float(r2_score(Y_true, Y_pred))
    mae = float(mean_absolute_error(Y_true, Y_pred))

    # ------------------------------------------------------------------
    # Likelihood and information criteria
    # ------------------------------------------------------------------
    logL = float(model.log_likelihood())
    k = model.num_params
    n = len(Y_true)

    aic = float(2 * k - 2 * logL)
    bic = float(k * np.log(n) - 2 * logL)

    # ------------------------------------------------------------------
    # Predictive interval coverage
    # ------------------------------------------------------------------
    lower = Y_pred - 2 * Y_pred_std
    upper = Y_pred + 2 * Y_pred_std
    coverage = float(np.mean((Y_true >= lower) & (Y_true <= upper)))

    # ------------------------------------------------------------------
    # Bound-check diagnostics
    # ------------------------------------------------------------------
    param_limits = check_param_limits_hit_from_constraints(model, constraints)

    return {
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae,
        "Coverage_95": coverage,
        "LogLikelihood": logL,
        "AIC": aic,
        "BIC": bic,
        "Params_at_bounds": param_limits,
    }


def bootstrap_assess_residuals_stats(  # noqa: PLR0913, PLR0914
    Y_true: ArrayLike,
    Y_pred: ArrayLike,
    *,
    alpha: float = 0.05,
    n_boot: int = 100,
    sample_size: int = 500,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Perform bootstrap-based residual diagnostic tests for Gaussian Process residuals.

    This function repeatedly draws bootstrap samples of residuals and computes
    standard diagnostic tests:
        - **Shapiro-Wilk** (normality, small/medium N)
        - **D'Agostino K²** (normality via skewness & kurtosis)
        - **Jarque-Bera** (normality via joint skew/kurtosis)
        - **Levene** (homoscedasticity / equal variance)
        - **Ljung-Box** (no autocorrelation at lag 10)

    Per-test p-values are collected across bootstrap samples, and summary
    statistics (mean/median p-value, rejection rate) are returned.

    Parameters
    ----------
    Y_true : array_like
        True target observations.
    Y_pred : array_like
        Predicted values from the GP.
    alpha : float, optional
        Significance threshold used for computing rejection rates.
    n_boot : int, optional
        Number of bootstrap iterations.
    sample_size : int, optional
        Size of each bootstrap sample. If larger than the dataset, the full
        dataset is used.
    random_state : int, numpy.random.Generator, or None, optional
        Seed or initialized generator for reproducibility.

    Returns
    -------
    pandas.DataFrame
        A dataframe with rows for each test:
            ['Shapiro-Wilk', "D'Agostino", 'Jarque-Bera', 'Levene', 'Ljung-Box']
        and columns:
            - ``Test``
            - ``Purpose``
            - ``Mean p-value``
            - ``Median p-value``
            - ``Rejection Rate`` (fraction of p < alpha)
            - ``Warn`` (True if rejection rate > 0.05)

    Notes
    -----
    - All exception handling is preserved exactly as in the original version.
    - Tests that fail or produce NaN p-values are handled identically.
    - The logic for the “Warn” column is unchanged.
    """
    # ------------------------------------------------------------------
    # Prepare residuals
    # ------------------------------------------------------------------
    Y_true = np.ravel(Y_true)
    Y_pred = np.ravel(Y_pred)
    residuals = Y_true - Y_pred

    n = len(residuals)
    sample_size = min(sample_size, n)

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Test definitions
    # ------------------------------------------------------------------
    tests = [
        "Shapiro-Wilk",
        "D'Agostino",
        "Jarque-Bera",
        "Levene",
        "Ljung-Box",
    ]

    purposes = {
        "Shapiro-Wilk": "Normality (N≤5000)",
        "D'Agostino": "Normality (skew & kurtosis)",
        "Jarque-Bera": "Normality (joint skew/kurt)",
        "Levene": "Equal Variance (homoscedasticity)",
        "Ljung-Box": "No autocorrelation (lag 10)",
    }

    # p-value storage
    results = {t: [] for t in tests}

    # ------------------------------------------------------------------
    # Constants preserved from original code
    # ------------------------------------------------------------------
    VARIANCE_TOL = 1e-12
    REJECTION_RATE = 0.05

    # ------------------------------------------------------------------
    # Bootstrap loop
    # ------------------------------------------------------------------
    for _ in trange(
        n_boot, desc="Bootstrapping", disable=not sys.stdout.isatty()
    ):
        idx = rng.choice(n, size=sample_size, replace=False)
        sample_resid = residuals[idx]
        sample_y = Y_true[idx]  # for Levene's homoscedasticity check

        # Shapiro-Wilk
        try:
            _, p_sw = shapiro(sample_resid)
        except Exception:
            p_sw = np.nan
        results["Shapiro-Wilk"].append(p_sw)

        # D'Agostino K²
        try:
            _, p_dag = normaltest(sample_resid)
        except Exception:
            p_dag = np.nan
        results["D'Agostino"].append(p_dag)

        # Jarque-Bera
        try:
            _, p_jb = jarque_bera(sample_resid)
        except Exception:
            p_jb = np.nan
        results["Jarque-Bera"].append(p_jb)

        # Levene (homoscedasticity: sample_y vs residuals)
        try:
            _, p_levene = levene(sample_y, sample_resid)
        except Exception:
            p_levene = np.nan
        results["Levene"].append(p_levene)

        # Ljung-Box (autocorrelation)
        if np.var(sample_resid) < VARIANCE_TOL:
            p_lb = np.nan
        else:
            try:
                lb_result = acorr_ljungbox(
                    sample_resid, lags=[10], return_df=True
                )
                p_lb = lb_result["lb_pvalue"].iloc[0]
            except Exception:
                p_lb = np.nan
        results["Ljung-Box"].append(p_lb)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    summary = []
    for test in tests:
        pvals = np.array(results[test], dtype=float)
        pvals = pvals[~np.isnan(pvals)]

        rejection_rate = np.mean(pvals < alpha) if pvals.size else np.nan

        summary.append(
            {
                "Test": test,
                "Purpose": purposes[test],
                "Mean p-value": np.nanmean(pvals) if pvals.size else np.nan,
                "Median p-value": np.nanmedian(pvals)
                if pvals.size
                else np.nan,
                "Rejection Rate": rejection_rate,
                "Warn": bool(rejection_rate > REJECTION_RATE)
                if np.isfinite(rejection_rate)
                else False,
            }
        )

    return pd.DataFrame(summary)


def report_model_fit(
    model_assessment: dict,
    report_metrics: bool = True,
) -> None:
    """
    Report (or warn about) model performance diagnostics.

    This function interprets the dictionary produced by
    :func:`assess_gp_model_fit` and prints metrics that fall within
    acceptable ranges. If metrics exceed pre-defined thresholds for
    poor fit, warnings are raised instead.

    Parameters
    ----------
    model_assessment : dict
        Dictionary of GP regression metrics as returned by
        :func:`assess_gp_model_fit`. Expected keys include:
        ``'RMSE'``, ``'R2'``, ``'Coverage_95'``, and
        ``'Params_at_bounds'``.
    report_metrics : bool, optional
        If True (default), metrics that do *not* trigger warnings
        are printed. If False, only warnings are emitted.

    Notes
    -----
    Thresholds used:

    - ``HIGH_RMSE = 0.1``
    - ``LOW_R2 = 0.8``
    - ``LOW_COVERAGE = 0.90``
    - ``HIGH_COVERAGE = 0.98``

    If any parameter lies at (or extremely near) its constraint bounds,
    a warning is raised regardless of ``report_metrics``.
    """
    # Thresholds
    HIGH_RMSE = 0.1
    LOW_R2 = 0.8
    LOW_COVERAGE = 0.90
    HIGH_COVERAGE = 0.98

    rmse = model_assessment.get("RMSE")
    r2 = model_assessment.get("R2")
    coverage = model_assessment.get("Coverage_95")
    param_limits_hit = model_assessment.get("Params_at_bounds", [])

    # ----------------------------
    # RMSE
    # ----------------------------
    if rmse is not None:
        if rmse > HIGH_RMSE:
            warnings.warn(f"High RMSE: {rmse:.4f}", stacklevel=2)
        elif report_metrics:
            print(f"RMSE: {rmse:.4f}")

    # ----------------------------
    # R²
    # ----------------------------
    if r2 is not None:
        if r2 < LOW_R2:
            warnings.warn(f"Low R²: {r2:.4f}", stacklevel=2)
        elif report_metrics:
            print(f"R²: {r2:.4f}")

    # ----------------------------
    # Predictive interval coverage
    # ----------------------------
    if coverage is not None:
        if coverage < LOW_COVERAGE or coverage > HIGH_COVERAGE:
            warnings.warn(
                f"Unusual 95% coverage: {coverage:.3f}", stacklevel=2
            )
        elif report_metrics:
            print(f"95% Coverage: {coverage:.3f}")

    # ----------------------------
    # Parameter bound warnings
    # ----------------------------
    if param_limits_hit:
        warnings.warn(
            f"Some parameters at constraint bounds: {param_limits_hit}",
            stacklevel=2,
        )


def plot_residuals(
    Y_true: ArrayLike,
    Y_pred: ArrayLike,
) -> None:
    """
    Plot residual diagnostics: a histogram of residuals and a scatter plot of
    residuals versus predicted values.

    Parameters
    ----------
    Y_true : array_like
        True observed target values. Will be reshaped to 1D.
    Y_pred : array_like
        Predicted target values from the model (same shape as ``Y_true``).

    Notes
    -----
    - This visualization helps identify skew, outliers, and heteroscedasticity.
    - Behavior, figure layout, and plotting style are identical to the original
      version of this function.
    """
    # Flatten for consistency
    Y_true = np.ravel(Y_true)
    Y_pred = np.ravel(Y_pred)

    residuals = Y_true - Y_pred

    # --------------------------------------------------------------
    # Create figure with two subplots: histogram + residual scatter
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 4))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=30, color="gray", edgecolor="black")
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    # Residuals vs predictions
    plt.subplot(1, 2, 2)
    plt.scatter(Y_pred, residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Residuals vs Predictions")
    plt.xlabel("Predicted Value")
    plt.ylabel("Residual")

    plt.tight_layout()
    plt.show()


def make_prediction_comparison_plot_2d(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_true: np.ndarray,
    Z_pred: np.ndarray,
    title: str | None = None,
) -> None:
    """
    Plot side-by-side heatmaps comparing ground truth and predicted values
    on a shared color scale.

    Parameters
    ----------
    X_grid : numpy.ndarray (2D)
        Meshgrid array of X coordinates, same shape as ``Z_true``.
    Y_grid : numpy.ndarray (2D)
        Meshgrid array of Y coordinates, same shape as ``Z_true``.
    Z_true : numpy.ndarray (2D)
        Ground truth values on the grid.
    Z_pred : numpy.ndarray (2D)
        Predicted values on the same grid.
    title : str or None, optional
        Optional title prefix applied above both subplots.

    Notes
    -----
    - Global ``vmin`` and ``vmax`` are computed jointly to ensure the two
      heatmaps use an identical color scale.
    - Plotting behavior matches the original function exactly.
    """
    # ------------------------------------------------------------------
    # Shared color scale for comparability
    # ------------------------------------------------------------------
    vmin = float(np.nanmin([np.nanmin(Z_true), np.nanmin(Z_pred)]))
    vmax = float(np.nanmax([np.nanmax(Z_true), np.nanmax(Z_pred)]))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Ground truth ------------------------------------------------------
    pcm_true = axs[0].pcolormesh(
        X_grid,
        Y_grid,
        Z_true,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_title(f"{title}\nGround Truth" if title else "Ground Truth")

    # Predictions -------------------------------------------------------
    pcm_pred = axs[1].pcolormesh(
        X_grid,
        Y_grid,
        Z_pred,
        cmap="viridis",
        norm=norm,
        shading="auto",
    )
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].set_title(f"{title}\nPredictions" if title else "Predictions")

    # Shared colorbar ---------------------------------------------------
    cbar = fig.colorbar(
        pcm_pred, ax=axs, location="right", fraction=0.046, pad=0.04
    )
    cbar.set_label("Value")

    plt.show()


def plot_array_with_coords(  # noqa: PLR0913, PLR0917
    x,
    y,
    Z,
    title=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    aspect="equal",
    cbar_size="4%",
    cbar_pad=0.04,
    cbar_label="Value",
    figsize=(7, 6),
):
    """
    Plot a 2D array `Z` using provided X-Y coordinates and a colorbar.

    Parameters
    ----------
    x : array_like
        X coordinates, either 1D (length ``Z.shape[1]``) or 2D (same shape as ``Z``).
    y : array_like
        Y coordinates, same rules as ``x``.
    Z : array_like
        2D array of values to plot.
    title : str, optional
        Plot title.
    cmap : str, optional
        Colormap name.
    vmin : float, optional
        Lower bound of colormap (default: min of Z).
    vmax : float, optional
        Upper bound of colormap (default: max of Z).
    aspect : str, optional
        Axis aspect ratio, default ``"equal"``.
    cbar_size : str, optional
        Width of colorbar relative to plot.
    cbar_pad : float, optional
        Padding between plot and colorbar.
    cbar_label : str, optional
        Colorbar label text.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    (Figure, Axes, Colorbar)
        The created matplotlib figure, axes, and colorbar.
    """
    Z = np.asarray(Z)
    maskZ = np.ma.masked_invalid(Z)

    x = np.asarray(x)
    y = np.asarray(y)

    # ---------------------------------------------------------
    # Coordinate grids
    # ---------------------------------------------------------
    if x.ndim == 1 and y.ndim == 1:
        if maskZ.shape != (y.size, x.size):
            raise ValueError(
                f"Z shape {maskZ.shape} does not match len(y) x len(x)."
            )
        X_grid, Y_grid = np.meshgrid(x, y)

    elif x.ndim == 2 and y.ndim == 2:  # noqa: PLR2004
        if x.shape != y.shape or x.shape != maskZ.shape:
            raise ValueError("If x and y are 2D, both must match shape of Z.")
        X_grid, Y_grid = x, y

    else:
        raise ValueError("x and y must be both 1D or both 2D arrays.")

    # ---------------------------------------------------------
    # Color normalization
    # ---------------------------------------------------------
    if vmin is None:
        vmin = float(np.nanmin(Z))
    if vmax is None:
        vmax = float(np.nanmax(Z))

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    pc = ax.pcolormesh(
        X_grid,
        Y_grid,
        maskZ,
        cmap=cmap,
        norm=norm,
        shading="auto",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(aspect)

    if title:
        ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)

    cbar = fig.colorbar(pc, cax=cax)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    return fig, ax, cbar


def update_gdf_with_predictions(
    gdf: gpd.GeoDataFrame,
    Y_grid: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> gpd.GeoDataFrame:
    """
    Update a GeoDataFrame with prediction values taken from a 2D grid.

    The function extracts each point's ``(x, y)`` coordinates, rounds them
    slightly to reduce floating-point mismatch, and looks up the corresponding
    predicted value from a flattened 2D prediction grid. Points whose rounded
    coordinates do not appear in the grid receive ``NaN``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing point geometries.
    Y_grid : numpy.ndarray (2D)
        Predicted values arranged on a grid of shape ``(n_y, n_x)``.
    x_grid : numpy.ndarray (2D)
        X-coordinates for each grid cell; must match ``Y_grid.shape``.
    y_grid : numpy.ndarray (2D)
        Y-coordinates for each grid cell; must match ``Y_grid.shape``.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of the input GeoDataFrame with an added column
        ``'value_extrapolated'`` containing predictions aligned by coordinate.

    Notes
    -----
    - Coordinates are matched *exactly* after rounding to 6 decimal places.
      Consider adjusting this value if coordinates are stored at a different
      precision.
    - Uses ``DataFrame.apply`` for readability; preserves your original behavior.
    - Temporary ``x`` and ``y`` columns are removed before return.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])
    >>> x, y = np.meshgrid([0, 1], [0, 1])
    >>> Y = np.array([[10, 20], [30, 40]])
    >>> update_gdf_with_predictions(gdf, Y, x, y)
       geometry  value_extrapolated
    0   POINT (0 0)               10.0
    1   POINT (1 1)               40.0
    """
    # ------------------------------------------------------------------
    # Extract coordinate columns
    # ------------------------------------------------------------------
    gdf = gdf.copy()
    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y

    # Flatten the input grids
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    Y_flat = Y_grid.ravel()

    coords = np.column_stack((x_flat, y_flat))

    # ------------------------------------------------------------------
    # Rounding to mitigate floating-point mismatches
    # ------------------------------------------------------------------
    def round_coord(x: float, y: float, nd: int = 6) -> tuple[float, float]:
        return (round(float(x), nd), round(float(y), nd))

    pred_dict = {
        round_coord(cx, cy): val for (cx, cy), val in zip(coords, Y_flat)
    }

    # ------------------------------------------------------------------
    # Map predictions row-by-row
    # ------------------------------------------------------------------
    gdf["value_extrapolated"] = gdf.apply(
        lambda row: pred_dict.get(
            round_coord(row.x, row.y),
            np.nan,
        ),
        axis=1,
    )

    # Remove temporary coordinate columns
    gdf = gdf.drop(columns=["x", "y"])

    return gdf


def backfill_gdf_at_height(  # noqa: PLR0913
    gdf: gpd.GeoDataFrame,
    *,
    value_col: str,
    z_value: float,
    z_tol: float,
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    backfilled_array: np.ndarray,
) -> gpd.GeoDataFrame:
    """
    Backfill missing values in a GeoDataFrame at a specific Z slice using
    a precomputed 2D grid of fill-in predictions.

    Only rows whose ``z`` coordinate is within ``z_value ± z_tol`` and whose
    ``value_col`` is NaN will be filled. The fill values are obtained via
    nearest-neighbor lookup in the X-Y plane from the provided grid.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing numeric columns ``'x'``, ``'y'``, ``'z'``,
        and the target ``value_col``.
    value_col : str
        Column name of the variable to fill.
    z_value : float
        The Z slice to target.
    z_tol : float
        Allowed absolute difference between a point's ``z`` and ``z_value``.
    X_grid : numpy.ndarray (2D)
        2D array of X coordinates defining the grid.
    Y_grid : numpy.ndarray (2D)
        2D array of Y coordinates defining the grid.
    backfilled_array : numpy.ndarray (2D)
        2D array of predicted values corresponding to ``X_grid`` / ``Y_grid``.

    Returns
    -------
    geopandas.GeoDataFrame
        A *copy* of the input GeoDataFrame where NaNs in ``value_col`` at the
        specified Z slice have been replaced with nearest-neighbor grid values.

    Notes
    -----
    - Points outside the Z tolerance range are not modified.
    - Only NaN entries are filled; existing values are left unchanged.
    - Nearest-neighbor search is performed using ``scipy.spatial.cKDTree``.
    - Grid shapes must match: ``X_grid.shape == Y_grid.shape == backfilled_array.shape``.
    """
    # ------------------------------------------------------------------
    # Validate grid shapes
    # ------------------------------------------------------------------
    if not (X_grid.shape == Y_grid.shape == backfilled_array.shape):
        raise ValueError(
            "X_grid, Y_grid, and backfilled_array must have identical shapes."
        )

    # Work on a copy
    gdf_out = gdf.copy()

    # ------------------------------------------------------------------
    # Identify slice by z-value within tolerance
    # ------------------------------------------------------------------
    if z_value is not None:
        z_mask = (gdf_out["z"] >= z_value - z_tol) & (
            gdf_out["z"] <= z_value + z_tol
        )
        slice_df = gdf_out.loc[z_mask].copy()
        null_mask = slice_df[value_col].isna()

        # Nothing to fill ─ return early
        if not null_mask.any():
            return gdf_out
    else:
        slice_df = gdf.copy()
        null_mask = slice_df[value_col].isna()

    # ------------------------------------------------------------------
    # Build KD-tree for nearest-neighbor mapping
    # ------------------------------------------------------------------
    x_flat = X_grid.ravel()
    y_flat = Y_grid.ravel()
    vals_flat = backfilled_array.ravel()

    # KD-tree over grid coordinates
    grid_tree = cKDTree(np.column_stack((x_flat, y_flat)))

    # Coordinates requiring fill
    missing_coords = slice_df.loc[null_mask, ["x", "y"]].to_numpy()

    # Query nearest grid point indices
    _, nn_idx = grid_tree.query(missing_coords)
    fill_values = vals_flat[nn_idx]

    # ------------------------------------------------------------------
    # Assign values back into the original gdf
    # ------------------------------------------------------------------
    gdf_out.loc[slice_df.index[null_mask], value_col] = fill_values

    return gdf_out


def backfill_gdf(  # noqa: PLR0913, PLR0914
    gdf: gpd.GeoDataFrame,
    value_col: str,
    *,
    z_value: float | None = None,
    z_tol: float = 1e-6,
    test_size: float = 0.2,
    seed: int = 42,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """
    Perform Gaussian Process-based extrapolation (or interpolation) to fill
    missing values in a GeoDataFrame at a specified Z slice.

    This high-level routine orchestrates a complete 2D GP modeling pipeline:

    1. **Slice preparation** via :func:`load_2d_data`
    2. **Sparse GP training** via :func:`build_and_fit_gp`
    3. **Model evaluation** (regression metrics + optional bootstrap diagnostics)
    4. **Prediction** on missing grid cells
    5. **Reconstruction** of a full 2D backfilled prediction array
    6. **GeoDataFrame merge** using either:
       - :func:`backfill_gdf_at_height` (if ``z_value`` provided), or
       - :func:`update_gdf_with_predictions` (if 2D / no Z slice)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing 3D point geometries and the target column.
        Must include columns ``'x'``, ``'y'``, ``'z'`` if Z slicing is used.
    value_col : str
        Name of the value column to fill.
    z_value : float or None, optional
        Z slice value to target. If ``None``, the method treats the data as 2D.
    z_tol : float, optional
        Allowed tolerance for selecting points with z ≈ ``z_value``.
    test_size : float, optional
        Fraction of known points reserved for model validation.
    seed : int, optional
        Random seed for train/validation splitting.
    verbose : bool, optional
        Whether to print progress, diagnostics, and plots.

    Returns
    -------
    geopandas.GeoDataFrame
        A copy of the input GeoDataFrame with missing values at the selected
        slice filled using GP predictions.

    Notes
    -----
    - All behavior from the original implementation is preserved.
    - No modifications are made to the original ``gdf``; a filled copy is returned.
    - Plotting and diagnostics occur only if ``verbose=True``.
    """
    # ------------------------------------------------------------------
    # 1. Slice + data preparation
    # ------------------------------------------------------------------
    data = load_2d_data(
        gdf,
        value_col=value_col,
        z_value=z_value,
        z_tol=z_tol,
        test_size=test_size,
        seed=seed,
    )

    nan_mask = data["nan_mask"]
    X_missing_stdized = data["X_missing_stdized"]

    X_train = data["X_train_stdized"]
    Y_train = data["Y_train_stdized"]
    X_val = data["X_val_stdized"]
    Y_val = data["Y_val"].ravel()

    Y_train_mean = data["Y_train_mean"]
    Y_train_std = data["Y_train_std"]

    # Standardized full grid (column-stacked version)
    X_full_stdized = np.column_stack([data["x_stdized"], data["y_stdized"]])

    X_grid = data["x_2d"]
    Y_grid_coords = data["y_2d"]
    Y_full = data["Y_2d"]

    # ------------------------------------------------------------------
    # 2. Train GP model
    # ------------------------------------------------------------------
    model, constraints = build_and_fit_gp(
        X_train,
        Y_train,
        optimize_restarts=0,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # 3. Predict on validation set
    # ------------------------------------------------------------------
    Y_pred_val, Y_pred_val_std = get_predictions(
        model,
        X_val,
        Y_mean=Y_train_mean,
        Y_std=Y_train_std,
    )

    # ------------------------------------------------------------------
    # 4. Fit assessment
    # ------------------------------------------------------------------
    if verbose:
        assessment = assess_gp_model_fit(
            model,
            Y_true=Y_val,
            Y_pred=Y_pred_val,
            Y_pred_std=Y_pred_val_std,
            constraints=constraints,
        )
        pprint(assessment)

    # ------------------------------------------------------------------
    # 5. Residual diagnostics (optional)
    # ------------------------------------------------------------------
    if verbose:
        residual_stats = bootstrap_assess_residuals_stats(
            Y_true=Y_val,
            Y_pred=Y_pred_val,
            alpha=0.05,
            n_boot=100,
            sample_size=500,
        )
        pprint(residual_stats)
        plot_residuals(Y_true=Y_val, Y_pred=Y_pred_val)

    # ------------------------------------------------------------------
    # 6. Predict at missing grid locations
    # ------------------------------------------------------------------
    Y_missing_pred, _Y_missing_pred_stddev = get_predictions(
        model,
        X_missing_stdized,
        Y_mean=Y_train_mean,
        Y_std=Y_train_std,
    )

    # ------------------------------------------------------------------
    # 7. Reconstruct full 2D backfilled grid
    # ------------------------------------------------------------------
    Y_full_pred = np.array(Y_full, copy=True)
    Y_full_pred[nan_mask] = Y_missing_pred

    if verbose:
        make_prediction_comparison_plot_2d(
            X_grid,
            Y_grid_coords,
            Z_true=Y_full,
            Z_pred=Y_full_pred,
            title=f"Z={z_value}",
        )

    # ------------------------------------------------------------------
    # 8. Merge back into GeoDataFrame
    # ------------------------------------------------------------------
    if z_value:
        gdf_filled = backfill_gdf_at_height(
            gdf,
            value_col=value_col,
            z_value=z_value,
            z_tol=z_tol,
            X_grid=X_grid,
            Y_grid=Y_grid_coords,
            backfilled_array=Y_full_pred,
        )
        gdf_filled = gdf_filled.rename(
            columns={value_col: "value_extrapolated"}
        )
    else:
        gdf_filled = update_gdf_with_predictions(
            gdf=gdf.copy(),
            y_grid=Y_grid_coords,
            x_grid=X_grid,
            Y_grid=Y_full_pred,
        )

    return gdf_filled
