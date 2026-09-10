"""LatticeKrigX field adapter for geoPFA spatial regression.

Reproduces the :func:`geopfa.extrapolation.build_and_fit_gp` /
:func:`geopfa.extrapolation.get_predictions` contract on **standardized**
inputs using the ``latticekrigx`` engine (`LKInfo` + `compute_basis` +
Gaussian coefficient solve + `predict_se`).

Notes
-----
- ``D in {2, 3}`` are supported here via ``LKRectangle`` / ``LKBox``
  geometries; ``D == 1`` maps to ``LKInterval``.
- Constant drift (build_and_fit_gp item 6) is represented as the LK
  fixed-effect intercept (``m = 1``); nugget (item 7) is represented as
  ``lambda_``.
- Heuristic auto-configuration (radius -> levels / range / nugget) is
  implemented in :func:`compute_lkx_config` (P1-S02).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from geopfa.exceptions import GEOPFAValueError
from latticekrigx.model.config import lk_setup
from latticekrigx.model.lattice_krig import lattice_krig
from latticekrigx.model.mle import joint_mle_optim
from latticekrigx.model.predict import predict_lkrig
from latticekrigx.model.predict_se import predict_se

__all__ = [
    "LkxConfig",
    "LkxModel",
    "compute_lkx_config",
    "fit_lkx_field",
    "lkx_predict",
]

_FINE_SCALE_FRAC = 0.02
_COARSE_SCALE_FRAC = 0.20
_MIN_NC = 4
_MIN_NLEVEL = 3
# Cap total multiresolution basis count to avoid CHOLMOD 32-bit integer
# overflow and intractable factorizations. Basis scales as NC**D per level,
# so 3D configs must be capped (2D configs stay well under this bound).
_MAX_TOTAL_BASIS = 40000
# 3D (and higher) LatticeKrig fits are far costlier per basis (dense-ish SAR
# assembly + 3D sparse Cholesky), so cap the 3D basis much lower to keep fits
# tractable. This limits how fine the 3D multiresolution can go.
_MAX_TOTAL_BASIS_3D = 1500
# Data-proportional basis cap. The absolute caps above bound the *domain*
# resolution, but a fit should not use vastly more basis functions than it has
# observations: for small n over a large domain the fine-scale multiresolution
# is unidentifiable (regularized away by lambda) and merely wastes compute.
# Effective cap = min(absolute_cap, max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS*n)).
_MAX_BASIS_PER_OBS = 10
_MIN_DATA_BASIS = 256
_LAMBDA_INIT_FRAC = 0.01
_LAMBDA_BOUND_HI_FRAC = 0.10
_LAMBDA_BOUND_LO = 1e-5


_GEOMETRY_BY_NDIM = {1: "LKInterval", 2: "LKRectangle", 3: "LKBox"}
_X_EXPECTED_NDIM = 2
_Y_COLUMN_NDIM = 2
_MIN_TRAINING_POINTS = 4
_CONSTANT_Y_STD_ATOL = 1e-10
_DEFAULT_PREDICT_CHUNK = 10_000
# Minimum a.wght per geometry (SAR stability lower bound).
_A_WGHT_LOWER_BOUND: dict[str, float] = {
    "LKInterval": 2.01,
    "LKRectangle": 4.01,
    "LKBox": 6.01,
}


@dataclass
class LkxConfig:
    """Explicit LatticeKrig configuration for :func:`fit_lkx_field`.

    Parameters
    ----------
    nlevel : int, default=3
        Number of multi-resolution levels.
    NC : int, default=8
        Number of lattice centers per dimension at the coarsest level.
    lambda_ : float, default=0.01
        Noise-to-signal ratio (nugget analog of build_and_fit_gp item 7).
    a_wght : float or None, optional
        SAR center weight per level. ``None`` uses the LK geometry
        default.
    nu : float or None, optional
        Wendland smoothness parameter passed through to lk_setup.
    m : int, default=1
        Fixed-effects polynomial degree: ``1`` = intercept-only (the
        constant-drift analog of build_and_fit_gp item 6).
    overlap : float or None, optional
        Basis overlap factor. ``None`` defers to lk_setup defaults.
    normalize : bool, default=True
        Whether to normalize the basis functions.
    extra : dict, optional
        Additional keyword arguments forwarded verbatim to
        :func:`latticekrigx.model.config.lk_setup`.
    lambda_bounds : tuple of float, optional
        Lower and upper bounds for MLE optimisation of ``lambda_``.
        Populated by :func:`compute_lkx_config`; ``None`` if not set.
    find_lambda : bool, default=False
        If ``True``, optimise ``lambda_`` by MLE (item 9 of the
        faithfulness contract). ``find_a_wght`` takes precedence when
        both are ``True``.
    find_a_wght : bool, default=False
        If ``True``, jointly optimise ``a_wght`` (and profile
        ``lambda_``) by MLE. Corresponds to R's ``LKrigFindLambdaAwght``.
    a_wght_per_level : bool, default=True
        When ``find_a_wght`` is ``True``, controls whether ``a_wght`` is
        optimised per resolution level (``True``) or as a shared scalar
        (``False``).
    """

    nlevel: int = 3
    NC: int = 8
    lambda_: float = 0.01
    a_wght: float | None = None
    nu: float | None = None
    m: int = 1
    overlap: float | None = None
    normalize: bool = True
    extra: dict[str, Any] = field(default_factory=dict)
    lambda_bounds: tuple[float, float] | None = None
    find_lambda: bool = False
    find_a_wght: bool = False
    a_wght_per_level: bool = True


@dataclass
class LkxModel:
    """Fitted LatticeKrigX field model returned by :func:`fit_lkx_field`.

    Parameters
    ----------
    fit : latticekrigx.model.fit.LKrigFit
        The underlying LatticeKrig fit.
    Y_train_mean : float
        Mean of the (already-standardized) training targets. Cached for
        parity with the GPy reference contract.
    ndim : int
        Number of input dimensions used to fit the model.
    geometry : str
        LatticeKrig geometry name used.
    constraint_info : dict
        Diagnostic dictionary echoing config, geometry, and MLE-adjacent
        fit stats. Mirrors the ``constraint_info`` returned by
        :func:`geopfa.extrapolation.build_and_fit_gp`.
    """

    fit: Any
    Y_train_mean: float
    ndim: int
    geometry: str
    constraint_info: dict[str, Any]


def _as_2d_x(X: NDArray[np.float64]) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != _X_EXPECTED_NDIM:
        raise GEOPFAValueError(
            f"X must be 2-D (N, D); got shape {X.shape!r}",
        )
    return X


def _as_1d_y(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == _Y_COLUMN_NDIM and Y.shape[1] == 1:
        Y = Y.ravel()
    if Y.ndim != 1:
        raise GEOPFAValueError(
            f"Y must be (N,) or (N, 1); got shape {Y.shape!r}",
        )
    return Y


def compute_lkx_config(
    X_std: NDArray[np.float64],
    n_train: int,
) -> LkxConfig:
    """Compute a heuristic LatticeKrig configuration from standardized coords.

    Parameters
    ----------
    X_std : numpy.ndarray
        Standardized coordinates (train or combined), shape ``(N, D)``.
        Per-dim range drives the radius and scale heuristics.
    n_train : int
        Number of training points; controls basis-count sizing (item 8).

    Returns
    -------
    LkxConfig
        Configuration with ``nlevel``, ``NC``, ``lambda_``, and
        ``lambda_bounds`` populated from the heuristics below.

    Notes
    -----
    Implements faithfulness-contract items 3-4, 7-8:

    - Item 3: global radius ``R = 0.5 * max(per-dim range of X_std)``.
      Zero-variance dims are already collapsed by standardization (std=1
      fallback); if *all* dims are zero-range, ``R`` defaults to 1.0.
    - Item 4: multi-resolution levels spanning ``[0.02R, 0.20R]``.
      ``NC`` is chosen so the coarsest lattice spacing is
      ``≥ _COARSE_SCALE_FRAC * R``; ``nlevel`` is chosen so the finest
      spacing is ``≤ _FINE_SCALE_FRAC * R``, with ``nlevel >= 3``.
    - Item 7: noise heuristic assuming unit-variance standardized target.
      ``lambda_ = _LAMBDA_INIT_FRAC * Var(Y_std) = 0.01``; bounds
      ``(_LAMBDA_BOUND_LO, _LAMBDA_BOUND_HI_FRAC * Var(Y_std)) =
      (1e-5, 0.10)``.
    - Item 8: GPy-analogue basis count ``M = min(300, max(20, n_train//10))``.
      Per-dim coarsest-level count from item 8: ``NC_m = M^(1/D)``.
      Final ``NC = max(NC_scale, NC_m, _MIN_NC)``, ensuring the heuristic
      spans the scale range while meeting the basis-richness criterion.
    - Data-proportional cap: total basis is additionally bounded by
      ``max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS * n_train)`` (as well as the
      absolute / 3D caps). For small ``n`` over a large domain the finest
      levels are dropped first, then ``NC`` is reduced, so the fit never uses
      far more basis functions than it has observations. The ``[0.02R, 0.20R]``
      fine-scale span is therefore only fully reached when ``n_train`` is large
      enough to support it.
    """
    X = _as_2d_x(X_std)
    D = X.shape[1]

    ranges = X.max(axis=0) - X.min(axis=0)
    max_range = float(ranges.max())
    if max_range == 0.0:
        max_range = 1.0
    R = 0.5 * max_range

    NC_scale = max(_MIN_NC, int(np.round(max_range / (_COARSE_SCALE_FRAC * R))))

    M = min(300, max(20, n_train // 10))
    NC_m = max(_MIN_NC, int(np.round(M ** (1.0 / D))))

    NC = max(NC_scale, NC_m)

    nlevel = max(
        _MIN_NLEVEL,
        int(np.ceil(np.log2(max_range / (_FINE_SCALE_FRAC * R * NC)))) + 1,
    )

    # Cap the total basis count so the SAR precision stays factorizable.
    # Per-level per-dim centers roughly double (NC * 2**level); total basis is
    # the sum of (per-dim centers)**D. In 3D this explodes, so reduce nlevel
    # first (drop the finest levels), then NC, until under _MAX_TOTAL_BASIS.
    def _total_basis(nc: int, nlev: int, d: int) -> int:
        return int(sum((nc * (2**lvl)) ** d for lvl in range(nlev)))

    basis_cap = _MAX_TOTAL_BASIS if D <= 2 else _MAX_TOTAL_BASIS_3D  # noqa: PLR2004
    # Also bound the basis relative to the data size so small-n / large-domain
    # fits do not over-parameterize (and stay fast). The floor keeps the
    # multiresolution usable for tiny n.
    data_cap = max(_MIN_DATA_BASIS, _MAX_BASIS_PER_OBS * n_train)
    basis_cap = min(basis_cap, data_cap)
    while nlevel > 1 and _total_basis(NC, nlevel, D) > basis_cap:
        nlevel -= 1
    while NC > _MIN_NC and _total_basis(NC, nlevel, D) > basis_cap:
        NC -= 1

    lambda_init = _LAMBDA_INIT_FRAC
    lambda_bounds = (_LAMBDA_BOUND_LO, _LAMBDA_BOUND_HI_FRAC)

    return LkxConfig(
        nlevel=nlevel,
        NC=NC,
        lambda_=lambda_init,
        lambda_bounds=lambda_bounds,
    )


def _flat_fallback_model(
    ndim: int,
    geometry: str,
    n_train: int,
    reason: str,
) -> LkxModel:
    return LkxModel(
        fit=None,
        Y_train_mean=0.0,
        ndim=ndim,
        geometry=geometry,
        constraint_info={
            "geometry": geometry,
            "ndim": ndim,
            "n_train": n_train,
            "fallback": reason,
        },
    )


def _fit_backend(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    geometry: str,
    cfg: LkxConfig,
    kwargs: dict[str, Any],
) -> tuple[Any, str]:
    """Dispatch to fixed / lambda-MLE / joint MLE fit and return (fit, kind)."""
    if cfg.find_a_wght:
        lkinfo = lk_setup(
            X,
            nlevel=cfg.nlevel,
            nc=cfg.NC,
            geometry=geometry,
            lambda_=None,
            nu=cfg.nu,
            a_wght=cfg.a_wght,
            m=cfg.m,
            **kwargs,
        )
        result = joint_mle_optim(
            X, Y, lkinfo, per_level=cfg.a_wght_per_level,
        )
        return result["optimal_fit"], "lambda+a_wght"

    find_lambda = bool(cfg.find_lambda)
    fit = lattice_krig(
        X,
        Y,
        nlevel=cfg.nlevel,
        NC=cfg.NC,
        geometry=geometry,
        lambda_=cfg.lambda_,
        a_wght=cfg.a_wght,
        m=cfg.m,
        find_lambda=find_lambda,
        **kwargs,
    )
    return fit, ("lambda" if find_lambda else "fixed")


def fit_lkx_field(
    X_train_std: NDArray[np.float64],
    Y_train_std: NDArray[np.float64],
    *,
    config: LkxConfig | dict[str, Any] | None = None,
) -> LkxModel:
    """Fit a LatticeKrigX field on standardized inputs.

    Parameters
    ----------
    X_train_std : numpy.ndarray
        Standardized training coordinates, shape ``(N, D)`` with
        ``D in {1, 2, 3}``.
    Y_train_std : numpy.ndarray
        Standardized training targets, shape ``(N,)`` or ``(N, 1)``.
    config : LkxConfig or dict, optional
        Explicit LatticeKrig configuration. Defaults to
        :class:`LkxConfig()` when ``None``.

    Returns
    -------
    LkxModel
        Fitted model bundling the underlying ``LKrigFit`` and the
        diagnostic ``constraint_info`` dictionary. When training data
        cannot support a fit (``N < 4`` or near-constant ``Y``), a
        degenerate model with ``fit=None`` is returned; downstream
        :func:`lkx_predict` calls on it emit a flat zero field.

    Notes
    -----
    Reproduces the build_and_fit_gp contract on standardized inputs:
    constant drift via ``m = 1`` (LK intercept fixed effect); nugget via
    ``lambda_``. When ``config.find_lambda`` (or ``config.find_a_wght``)
    is ``True``, ``lambda_`` (and optionally ``a_wght``) are chosen by
    MLE (item 9). Fallbacks (item 12) return a degenerate flat-zero
    model for pathological inputs.
    """
    if config is None:
        cfg = LkxConfig()
    elif isinstance(config, dict):
        cfg = LkxConfig(**config)
    else:
        cfg = config

    X = _as_2d_x(X_train_std)
    Y = _as_1d_y(Y_train_std)

    if X.shape[0] != Y.shape[0]:
        raise GEOPFAValueError(
            "X_train_std and Y_train_std must share N; "
            f"got {X.shape[0]} vs {Y.shape[0]}",
        )

    ndim = X.shape[1]
    if ndim not in _GEOMETRY_BY_NDIM:
        raise GEOPFAValueError(
            f"Unsupported ndim={ndim}; expected D in {{1, 2, 3}}",
        )
    geometry = _GEOMETRY_BY_NDIM[ndim]

    if X.shape[0] < _MIN_TRAINING_POINTS:
        return _flat_fallback_model(
            ndim=ndim,
            geometry=geometry,
            n_train=int(X.shape[0]),
            reason="too_few_training_points",
        )
    if float(Y.std()) < _CONSTANT_Y_STD_ATOL:
        return _flat_fallback_model(
            ndim=ndim,
            geometry=geometry,
            n_train=int(X.shape[0]),
            reason="constant_Y",
        )

    kwargs: dict[str, Any] = dict(cfg.extra)
    if cfg.overlap is not None:
        kwargs.setdefault("overlap", cfg.overlap)
    if cfg.normalize is not None:
        kwargs.setdefault("normalize", cfg.normalize)
    if cfg.nu is not None:
        kwargs.setdefault("nu", cfg.nu)

    fit, mle_kind = _fit_backend(X, Y, geometry, cfg, kwargs)

    lambda_bounds = (
        cfg.lambda_bounds
        if cfg.lambda_bounds is not None
        else (_LAMBDA_BOUND_LO, _LAMBDA_BOUND_HI_FRAC)
    )
    a_wght_fit: list[float] | None = None
    if cfg.find_a_wght:
        a_wght_fit = [float(v) for v in np.asarray(fit.lkinfo.a_wght).ravel()]

    constraint_info: dict[str, Any] = {
        "geometry": geometry,
        "ndim": ndim,
        "n_train": int(X.shape[0]),
        "config": {
            "nlevel": cfg.nlevel,
            "NC": cfg.NC,
            "lambda_": cfg.lambda_,
            "a_wght": cfg.a_wght,
            "nu": cfg.nu,
            "m": cfg.m,
            "overlap": cfg.overlap,
            "normalize": cfg.normalize,
            "find_lambda": cfg.find_lambda,
            "find_a_wght": cfg.find_a_wght,
        },
        "mle": mle_kind,
        "lambda_fit": float(np.asarray(fit.lkinfo.lambda_).ravel()[0]),
        "lambda_bounds": (float(lambda_bounds[0]), float(lambda_bounds[1])),
        "a_wght_fit": a_wght_fit,
        "a_wght_lower_bound": _A_WGHT_LOWER_BOUND.get(geometry),
        "sigma2_MLE": float(np.asarray(fit.sigma2_MLE).ravel()[0]),
        "lnProfileLike": float(fit.lnProfileLike),
    }

    return LkxModel(
        fit=fit,
        Y_train_mean=float(Y.mean()),
        ndim=ndim,
        geometry=geometry,
        constraint_info=constraint_info,
    )


def lkx_predict(
    model: LkxModel,
    X_std: NDArray[np.float64],
    *,
    chunk_size: int | None = _DEFAULT_PREDICT_CHUNK,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Predict mean and standard deviation on standardized inputs.

    Parameters
    ----------
    model : LkxModel
        Fitted model returned by :func:`fit_lkx_field`.
    X_std : numpy.ndarray
        Standardized prediction locations, shape ``(M, D)`` with
        ``D`` matching the training ndim.
    chunk_size : int or None, default=10000
        If not ``None``, predictions are computed in row-chunks of this
        size and concatenated (item 11 streaming prediction). Set to
        ``None`` to force a single unbatched call.

    Returns
    -------
    mean : numpy.ndarray
        Predictive mean on the standardized target scale, shape
        ``(M,)``.
    std : numpy.ndarray
        Predictive standard deviation for a new noisy observation on the
        standardized target scale, shape ``(M,)``. This includes both latent
        field uncertainty and the fitted observation nugget, matching the
        historical GPy adapter contract.

    Notes
    -----
    When ``model.fit is None`` (fallback fit), returns a flat zero
    ``(mean, std)`` pair of length ``M``, matching the ``build_and_fit_gp``
    fallback contract (item 12).
    """
    X = _as_2d_x(X_std)
    if X.shape[1] != model.ndim:
        raise GEOPFAValueError(
            "X_std ndim mismatch: model expects "
            f"D={model.ndim}, got D={X.shape[1]}",
        )

    m_pred = X.shape[0]
    if model.fit is None:
        return (
            np.zeros(m_pred, dtype=np.float64),
            np.zeros(m_pred, dtype=np.float64),
        )

    if chunk_size is None or chunk_size <= 0 or m_pred <= chunk_size:
        mean = np.asarray(
            predict_lkrig(model.fit, X), dtype=np.float64,
        ).ravel()
        latent_std = np.asarray(
            predict_se(model.fit, X), dtype=np.float64,
        ).ravel()
        observation_variance = model.fit.lambda_ * float(
            np.asarray(model.fit.sigma2_MLE).ravel()[0]
        )
        return mean, np.sqrt(latent_std**2 + observation_variance)

    mean = np.empty(m_pred, dtype=np.float64)
    std = np.empty(m_pred, dtype=np.float64)
    for start in range(0, m_pred, chunk_size):
        stop = min(start + chunk_size, m_pred)
        block = X[start:stop]
        mean[start:stop] = np.asarray(
            predict_lkrig(model.fit, block), dtype=np.float64,
        ).ravel()
        latent_std = np.asarray(
            predict_se(model.fit, block), dtype=np.float64,
        ).ravel()
        observation_variance = model.fit.lambda_ * float(
            np.asarray(model.fit.sigma2_MLE).ravel()[0]
        )
        std[start:stop] = np.sqrt(latent_std**2 + observation_variance)
    return mean, std
