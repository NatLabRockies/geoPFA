"""Reduced-form coordinate-trend sensitivity diagnostics.

This module does not fit a spatial random field.  Its optional coordinate terms
are ordinary linear predictors and must not be described as a spatial model.
Production probabilistic scenarios belong in the config-driven GBLK runner.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .decision_metrics import auc_tie_safe


@dataclass(frozen=True)
class CoordinateTrendScenarioSpec:
    """Configuration for a reduced-form coordinate-trend diagnostic."""

    name: str
    drop_features: tuple[str, ...] = ()
    include_priors: bool = True
    include_coordinate_trend: bool = False
    missing_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Reject ambiguous or non-identifiable diagnostic specifications."""
        if not self.name.strip():
            raise ValueError("scenario name must not be empty")
        if not np.isfinite(self.missing_fraction) or not (
            0.0 <= self.missing_fraction < 1.0
        ):
            raise ValueError("missing_fraction must be finite and in [0, 1)")


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return (X - mean) / scale, mean, scale


def _fit_logit(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    regularization: float = 1e-2,
) -> np.ndarray:
    def objective(beta: np.ndarray) -> float:
        eta = offset + X @ beta
        p = np.clip(expit(eta), 1e-9, 1.0 - 1e-9)
        nll = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        return float(nll + 0.5 * regularization * np.dot(beta, beta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = offset + X @ beta
        p = expit(eta)
        return X.T @ (p - y) + regularization * beta

    x0 = np.zeros(X.shape[1])
    for method in ("BFGS", "L-BFGS-B"):
        result = minimize(objective, x0=x0, jac=gradient, method=method)
        if result.success and np.isfinite(result.x).all():
            return result.x
        x0 = result.x if np.isfinite(result.x).all() else x0
    raise RuntimeError(
        "scenario logistic fit failed for both BFGS and L-BFGS-B: "
        f"{result.message}"
    )


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def _finite_mean(values: list[float | str]) -> float:
    numeric = np.asarray(values, dtype=float)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if finite.size else float("nan")


_DEFAULT_THRESHOLD = 0.5
_MIN_CLASSES = 2
_MIN_PLATT_SAMPLES = 4
_MIN_DIMS = 2


def _accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p >= _DEFAULT_THRESHOLD) == (y == 1)))


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    return auc_tie_safe(y, p)


def _fit_platt_scaler(
    scores: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Fit Platt scaling parameters a,b for p = sigmoid(a*s + b)."""
    if len(scores) < _MIN_PLATT_SAMPLES:
        raise ValueError(
            f"Platt scaling requires at least {_MIN_PLATT_SAMPLES} samples"
        )
    if np.unique(y).size < _MIN_CLASSES:
        raise ValueError("Platt scaling requires two outcome classes")

    def objective(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        p = np.clip(expit(a * scores + b), 1e-9, 1.0 - 1e-9)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    result = minimize(
        objective,
        x0=np.array([1.0, 0.0], dtype=float),
        method="BFGS",
    )
    if result.success and np.isfinite(result.x).all():
        return float(result.x[0]), float(result.x[1])
    raise RuntimeError(f"scenario Platt scaling failed: {result.message}")


def _block_ids(coords: np.ndarray, grid_size: int = 4) -> np.ndarray:
    if coords.ndim != _MIN_DIMS or coords.shape[1] != _MIN_DIMS:
        raise ValueError("coords must have shape (n, 2)")
    x = coords[:, 0]
    y = coords[:, 1]
    x_edges = np.linspace(np.nanmin(x), np.nanmax(x), grid_size + 1)
    y_edges = np.linspace(np.nanmin(y), np.nanmax(y), grid_size + 1)
    x_bin = np.clip(
        np.digitize(x, x_edges[1:-1], right=False),
        0,
        grid_size - 1,
    )
    y_bin = np.clip(
        np.digitize(y, y_edges[1:-1], right=False),
        0,
        grid_size - 1,
    )
    return x_bin + grid_size * y_bin


def _balanced_block_fold_ids(blocks: np.ndarray, n_folds: int) -> np.ndarray:
    """Pack whole blocks into folds while balancing observation counts.

    Blocks are considered from largest to smallest and assigned to the
    currently smallest fold. Ties are resolved by block ID and then fold ID,
    so the assignment is deterministic and does not inspect outcomes.
    """
    block_array = np.asarray(blocks)
    if block_array.ndim != 1 or block_array.size == 0:
        raise ValueError("blocks must be a nonempty one-dimensional array")
    if (
        isinstance(n_folds, bool | np.bool_)
        or not isinstance(n_folds, int | np.integer)
        or n_folds < _MIN_CLASSES
    ):
        raise ValueError("n_folds must be an integer >= 2")
    unique, inverse, counts = np.unique(
        block_array, return_inverse=True, return_counts=True
    )
    if n_folds > unique.size:
        raise ValueError("n_folds cannot exceed the number of occupied blocks")

    order = sorted(
        range(unique.size),
        key=lambda index: (-int(counts[index]), int(unique[index])),
    )
    fold_loads = np.zeros(int(n_folds), dtype=np.int64)
    block_folds = np.empty(unique.size, dtype=np.intp)
    for block_index in order:
        fold = int(np.argmin(fold_loads))
        block_folds[block_index] = fold
        fold_loads[fold] += int(counts[block_index])
    return block_folds[inverse]


def spatial_block_holdout_mask(
    coords: np.ndarray,
    holdout_fraction: float = 0.2,
    grid_size: int = 4,
) -> np.ndarray:
    """Return boolean mask where True marks holdout samples by spatial block."""

    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    blocks = _block_ids(coords, grid_size=grid_size)
    unique = np.unique(blocks)
    n_holdout = max(1, int(np.ceil(len(unique) * holdout_fraction)))
    holdout_blocks = set(unique[:n_holdout].tolist())
    return np.array([block in holdout_blocks for block in blocks], dtype=bool)


def _fit_and_eval(  # noqa: PLR0913, PLR0917, PLR0914
    X_train: np.ndarray,
    y_train: np.ndarray,
    offset_train: np.ndarray,
    coords_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    offset_test: np.ndarray,
) -> dict[str, float | str]:
    calibration_mask = spatial_block_holdout_mask(
        coords_train,
        holdout_fraction=0.2,
        grid_size=3,
    )
    model_mask = ~calibration_mask
    if np.unique(y_train).size < _MIN_CLASSES:
        raise ValueError("scenario outcome fit requires two training classes")
    if (
        model_mask.sum() < _MIN_CLASSES
        or calibration_mask.sum() < _MIN_CLASSES
    ):
        calibration_status = (
            "not_estimable_insufficient_spatial_calibration_rows"
        )
    elif np.unique(y_train[model_mask]).size < _MIN_CLASSES:
        calibration_status = "not_estimable_model_subset_single_class"
    elif np.unique(y_train[calibration_mask]).size < _MIN_CLASSES:
        calibration_status = "not_estimable_calibration_subset_single_class"
    else:
        calibration_status = "estimated_platt_on_spatial_holdout"
    if calibration_status != "estimated_platt_on_spatial_holdout":
        model_mask = np.ones(len(y_train), dtype=bool)
        calibration_mask = np.zeros(len(y_train), dtype=bool)

    X_model = X_train[model_mask]
    y_model = y_train[model_mask]
    offset_model = offset_train[model_mask]
    X_model_std, mean, scale = _standardize(X_model)
    X_test_std = (X_test - mean) / scale
    beta = _fit_logit(X_model_std, y_model, offset_model)
    test_scores = offset_test + X_test_std @ beta
    p_test = expit(test_scores)

    calibrated_metrics = {
        "log_loss_calibrated": float("nan"),
        "brier_calibrated": float("nan"),
        "accuracy_calibrated": float("nan"),
        "auc_calibrated": float("nan"),
    }
    if calibration_mask.any():
        X_cal = X_train[calibration_mask]
        y_cal = y_train[calibration_mask]
        offset_cal = offset_train[calibration_mask]
        X_cal_std = (X_cal - mean) / scale
        cal_scores = offset_cal + X_cal_std @ beta
        a, b = _fit_platt_scaler(cal_scores, y_cal)
        p_test_calibrated = expit(a * test_scores + b)
        calibrated_metrics = {
            "log_loss_calibrated": _log_loss(y_test, p_test_calibrated),
            "brier_calibrated": _brier(y_test, p_test_calibrated),
            "accuracy_calibrated": _accuracy(y_test, p_test_calibrated),
            "auc_calibrated": _roc_auc(y_test, p_test_calibrated),
        }

    return {
        "log_loss": _log_loss(y_test, p_test),
        "brier": _brier(y_test, p_test),
        "accuracy": _accuracy(y_test, p_test),
        "auc": _roc_auc(y_test, p_test),
        "calibration_status": calibration_status,
        **calibrated_metrics,
    }


def _scenario_arrays(
    X_df: pd.DataFrame,
    y: np.ndarray,
    alpha: np.ndarray,
    coords: np.ndarray,
    spec: CoordinateTrendScenarioSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    missing_features = [
        name for name in spec.drop_features if name not in X_df.columns
    ]
    if missing_features:
        raise ValueError(
            f"scenario {spec.name} references unknown features: {missing_features}"
        )

    selected = X_df.drop(columns=list(spec.drop_features)).copy()
    if selected.shape[1] == 0:
        raise ValueError(f"scenario {spec.name} dropped all features")

    work = selected.to_numpy(dtype=float)
    if spec.missing_fraction > 0:
        n_missing = int(np.floor(len(work) * spec.missing_fraction))
        if n_missing > 0:
            work = work.copy()
            work[:n_missing, 0] = np.nan

    if spec.include_coordinate_trend:
        work = np.column_stack([work, coords])

    offset = alpha.copy() if spec.include_priors else np.zeros_like(alpha)
    mask = np.isfinite(work).all(axis=1) & np.isfinite(offset) & np.isfinite(y)
    return (
        work[mask],
        y[mask],
        offset[mask],
        coords[mask],
        tuple(selected.columns.tolist()),
    )


def _json_numeric_distribution(values: list[float | str]) -> str:
    """Serialize fold values without emitting non-standard JSON NaN tokens."""
    serializable = [
        float(value) if np.isfinite(float(value)) else None for value in values
    ]
    return json.dumps(serializable, separators=(",", ":"))


def run_coordinate_trend_sensitivity(  # noqa: PLR0913, PLR0917, PLR0914
    X_df: pd.DataFrame,
    y: np.ndarray,
    alpha: np.ndarray,
    coords: np.ndarray,
    scenarios: Iterable[CoordinateTrendScenarioSpec],
    n_splits: int = 4,
) -> pd.DataFrame:
    """Run blocked diagnostics for evidence, priors, and linear coordinates."""
    if isinstance(n_splits, bool | np.bool_) or not isinstance(
        n_splits, int | np.integer
    ):
        raise TypeError("n_splits must be an integer")
    if n_splits < 2:  # noqa: PLR2004
        raise ValueError("n_splits must be at least 2 for cross-validation")
    y = np.asarray(y, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    coords = np.asarray(coords, dtype=float)
    n_rows = len(X_df)
    if y.shape != (n_rows,) or alpha.shape != (n_rows,):
        raise ValueError("X_df, y, and alpha must have aligned rows")
    if coords.shape != (n_rows, _MIN_DIMS):
        raise ValueError("coords must have shape (len(X_df), 2)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coords must contain only finite values")
    if np.any(np.isinf(y)) or np.any(np.isfinite(y) & ~np.isin(y, (0.0, 1.0))):
        raise ValueError("y must contain binary 0/1 labels or missing values")
    rows: list[dict[str, float | int | str]] = []

    for spec in scenarios:
        X_s, y_s, alpha_s, coords_s, feature_names = _scenario_arrays(
            X_df,
            y,
            alpha,
            coords,
            spec,
        )
        holdout_mask = spatial_block_holdout_mask(
            coords_s,
            holdout_fraction=0.2,
            grid_size=4,
        )
        train_mask = ~holdout_mask
        if train_mask.sum() < _MIN_CLASSES or holdout_mask.sum() < 1:
            raise ValueError(
                f"scenario {spec.name} has insufficient train/test samples",
            )

        holdout = _fit_and_eval(
            X_s[train_mask],
            y_s[train_mask],
            alpha_s[train_mask],
            coords_s[train_mask],
            X_s[holdout_mask],
            y_s[holdout_mask],
            alpha_s[holdout_mask],
        )

        blocks = _block_ids(coords_s, grid_size=4)
        unique_blocks = np.unique(blocks)
        n_cv_folds = min(int(n_splits), len(unique_blocks))
        if n_cv_folds < 2:  # noqa: PLR2004
            raise ValueError(
                f"scenario {spec.name} has fewer than two occupied spatial blocks"
            )
        cv_metrics: dict[str, list[float | str]] = {
            "log_loss": [],
            "brier": [],
            "accuracy": [],
            "auc": [],
            "log_loss_calibrated": [],
            "brier_calibrated": [],
            "accuracy_calibrated": [],
            "auc_calibrated": [],
            "calibration_status": [],
        }
        fold_ids = _balanced_block_fold_ids(blocks, n_cv_folds)
        for fold in range(n_cv_folds):
            test = fold_ids == fold
            train = ~test
            if train.sum() < _MIN_CLASSES or test.sum() < 1:
                raise ValueError(
                    f"scenario {spec.name} fold {fold} has insufficient "
                    "train/test rows"
                )
            fold_out = _fit_and_eval(
                X_s[train],
                y_s[train],
                alpha_s[train],
                coords_s[train],
                X_s[test],
                y_s[test],
                alpha_s[test],
            )
            for key, value in fold_out.items():
                cv_metrics[key].append(value)

        row: dict[str, float | int | str] = {
            "scenario": spec.name,
            "n_samples": len(y_s),
            "n_features": int(
                len(feature_names)
                + (2 if spec.include_coordinate_trend else 0)
            ),
            "priors_enabled": spec.include_priors,
            "coordinate_trend_enabled": spec.include_coordinate_trend,
            "dropped_features": ",".join(spec.drop_features),
            "holdout_log_loss": holdout["log_loss"],
            "holdout_brier": holdout["brier"],
            "holdout_accuracy": holdout["accuracy"],
            "holdout_auc": holdout["auc"],
            "holdout_log_loss_calibrated": holdout["log_loss_calibrated"],
            "holdout_brier_calibrated": holdout["brier_calibrated"],
            "holdout_accuracy_calibrated": holdout["accuracy_calibrated"],
            "holdout_auc_calibrated": holdout["auc_calibrated"],
            "holdout_calibration_status": holdout["calibration_status"],
            "cv_log_loss_mean": _finite_mean(cv_metrics["log_loss"]),
            "cv_brier_mean": _finite_mean(cv_metrics["brier"]),
            "cv_accuracy_mean": _finite_mean(cv_metrics["accuracy"]),
            "cv_auc_mean": _finite_mean(cv_metrics["auc"]),
            "cv_log_loss_by_fold": _json_numeric_distribution(
                cv_metrics["log_loss"]
            ),
            "cv_brier_by_fold": _json_numeric_distribution(
                cv_metrics["brier"]
            ),
            "cv_log_loss_calibrated_mean": _finite_mean(
                cv_metrics["log_loss_calibrated"]
            ),
            "cv_brier_calibrated_mean": _finite_mean(
                cv_metrics["brier_calibrated"]
            ),
            "cv_accuracy_calibrated_mean": _finite_mean(
                cv_metrics["accuracy_calibrated"]
            ),
            "cv_auc_calibrated_mean": _finite_mean(
                cv_metrics["auc_calibrated"]
            ),
            "cv_log_loss_calibrated_by_fold": _json_numeric_distribution(
                cv_metrics["log_loss_calibrated"]
            ),
            "cv_brier_calibrated_by_fold": _json_numeric_distribution(
                cv_metrics["brier_calibrated"]
            ),
            "cv_calibration_estimable_fraction": float(
                np.mean(
                    np.asarray(cv_metrics["calibration_status"], dtype=object)
                    == "estimated_platt_on_spatial_holdout"
                )
            ),
            "cv_calibration_statuses": ";".join(
                str(value) for value in cv_metrics["calibration_status"]
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)
