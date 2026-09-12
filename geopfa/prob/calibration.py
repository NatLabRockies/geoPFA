"""Probability calibration diagnostics for the Stage-1 probabilistic demo.

These helpers are framework-agnostic - they operate on `(y, p)` arrays of held-out
labels and predicted probabilities. They are used by the Stage-1 scenario runner
to produce reliability diagrams, ECE / MCE / Brier / log-loss metrics, and a
diagnostic temperature-scaling estimate.

Temperature scaling is reported as a *diagnostic*; it is not applied to map
outputs. A scalar temperature can sharpen or soften predictions around 0.5 but
cannot translate a systematic location bias. For datasets whose reliability
diagram is uniformly above or below the diagonal, treat T as a summary number
and use Platt or isotonic recalibration if a corrective transform is needed.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import norm as _scipy_norm

_CALIBRATION_SCORE_TOLERANCE = 1e-7


def _as_probability_vector(
    values: Sequence[float] | np.ndarray, *, name: str = "p"
) -> np.ndarray:
    """Return a real, finite one-dimensional probability vector."""

    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real probabilities")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric probabilities") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite probabilities")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    return array


def _validate_binary_probability_arrays(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary labels and probability forecasts."""

    y_raw = np.asarray(y)
    if np.iscomplexobj(y_raw):
        raise ValueError("y must contain real binary 0/1 labels")
    try:
        y_array = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y must contain numeric binary 0/1 labels") from exc
    p_array = _as_probability_vector(p)
    if y_array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if y_array.shape != p_array.shape:
        raise ValueError("y and p must have the same shape")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("y must contain only finite binary 0/1 labels")
    if not np.all(np.isin(y_array, (0.0, 1.0))):
        raise ValueError("y must contain binary 0/1 labels")
    return y_array.astype(int), p_array


@dataclass(frozen=True)
class CalibrationBin:
    """Per-bin reliability summary."""

    bin_index: int
    n: int
    bin_p_lo: float
    bin_p_hi: float
    mean_predicted_p: float
    observed_fraction: float
    wilson_ci_lo: float
    wilson_ci_hi: float

    def as_dict(self) -> dict:
        """Return a plain-dict representation of the bin row."""
        return {
            "bin": self.bin_index,
            "n": self.n,
            "bin_p_lo": self.bin_p_lo,
            "bin_p_hi": self.bin_p_hi,
            "mean_predicted_p": self.mean_predicted_p,
            "observed_fraction": self.observed_fraction,
            "wilson_ci_lo": self.wilson_ci_lo,
            "wilson_ci_hi": self.wilson_ci_hi,
        }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% (default) Wilson score interval for a binomial proportion ``k / n``.

    Wilson is preferred over the normal-approximation interval at small ``n``
    and when ``p_hat`` is near 0 or 1.
    """
    if isinstance(k, bool | np.bool_) or not isinstance(k, int | np.integer):
        raise TypeError("k and n must be integers satisfying 0 <= k <= n")
    if isinstance(n, bool | np.bool_) or not isinstance(n, int | np.integer):
        raise TypeError("k and n must be integers satisfying 0 <= k <= n")
    if n < 0 or k < 0 or k > n:
        raise ValueError("k and n must be integers satisfying 0 <= k <= n")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if n == 0:
        return (0.0, 1.0)
    z = float(_scipy_norm.ppf(1.0 - alpha / 2.0))
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = (
        z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    ) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def equal_frequency_reliability_table(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 5,
    alpha: float = 0.05,
) -> list[CalibrationBin]:
    """Equal-frequency (quantile) reliability table.

    Quantile cut points target ``n_bins`` groups of nearly equal size. Equal
    predictions are never split across bins, so permuting rows within a tied
    score group cannot change the reliability table or ECE. Ties can therefore
    produce fewer than ``n_bins`` nonempty groups. Each row carries the bin
    sample size, predicted-probability range, mean predicted probability,
    observed positive fraction, and the (1 - alpha) Wilson CI on the observed
    fraction.
    """
    y_arr, p_arr = _validate_binary_probability_arrays(y, p)
    if y_arr.size == 0:
        return []
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    effective_bins = min(n_bins, y_arr.size)
    quantile_edges = np.quantile(
        p_arr,
        np.linspace(0.0, 1.0, effective_bins + 1),
    )
    bin_ids = np.searchsorted(quantile_edges[1:-1], p_arr, side="left")
    bin_groups = [
        np.flatnonzero(bin_ids == bin_index)
        for bin_index in range(effective_bins)
        if np.any(bin_ids == bin_index)
    ]

    # Warn when any bin will have fewer than 5 samples — the resulting
    # observed fraction (0.0 or 1.0 for single-sample bins) and Wilson CIs
    # are statistically meaningless at such sizes.
    min_bin_size = min(len(g) for g in bin_groups if len(g) > 0)
    if min_bin_size < 5:  # noqa: PLR2004
        warnings.warn(
            f"equal_frequency_reliability_table: smallest bin has {min_bin_size} "
            f"sample(s) (n={y_arr.size}, n_bins={n_bins}). Reliability diagram "
            "bins with < 5 samples produce statistically unstable ECE/MCE estimates. "
            "Consider reducing n_bins or gathering more labeled data.",
            UserWarning,
            stacklevel=2,
        )

    rows: list[CalibrationBin] = []
    for b_idx, bin_idx in enumerate(bin_groups):
        if len(bin_idx) == 0:
            continue
        bin_p = p_arr[bin_idx]
        bin_y = y_arr[bin_idx]
        n_bin = len(bin_p)
        k_pos = int(bin_y.sum())
        ci_lo, ci_hi = wilson_ci(k_pos, n_bin, alpha=alpha)
        rows.append(
            CalibrationBin(
                bin_index=b_idx,
                n=n_bin,
                bin_p_lo=float(bin_p.min()),
                bin_p_hi=float(bin_p.max()),
                mean_predicted_p=float(bin_p.mean()),
                observed_fraction=k_pos / n_bin,
                wilson_ci_lo=ci_lo,
                wilson_ci_hi=ci_hi,
            )
        )
    return rows


def expected_calibration_error(rows: Iterable[CalibrationBin]) -> float:
    """Sample-weighted mean gap between predicted probability and observed fraction.

    Returns NaN when no rows are provided.
    """
    rows_list = list(rows)
    if not rows_list:
        return float("nan")
    n_total = sum(r.n for r in rows_list)
    if n_total <= 0:
        return float("nan")
    return float(
        sum(
            (r.n / n_total) * abs(r.mean_predicted_p - r.observed_fraction)
            for r in rows_list
        )
    )


def maximum_calibration_error(rows: Iterable[CalibrationBin]) -> float:
    """Worst-case absolute calibration gap across the supplied bins."""
    rows_list = list(rows)
    if not rows_list:
        return float("nan")
    return float(
        max(abs(r.mean_predicted_p - r.observed_fraction) for r in rows_list)
    )


def brier_score(
    y: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray
) -> float:
    """Brier score: ``mean((y - p) ** 2)``."""
    y_arr, p_arr = _validate_binary_probability_arrays(y, p)
    if y_arr.size == 0:
        return float("nan")
    return float(np.mean((y_arr - p_arr) ** 2))


def log_loss(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """Binary cross-entropy / log-loss with predictions clipped to ``[eps, 1-eps]``."""
    y_arr, p_raw = _validate_binary_probability_arrays(y, p)
    if y_arr.size == 0:
        return float("nan")
    p_arr = np.clip(p_raw, eps, 1.0 - eps)
    return float(
        -np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr))
    )


def fit_temperature(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 50.0),
    eps: float = 1e-9,
) -> tuple[float, np.ndarray]:
    """Fit the scalar temperature ``T`` that minimises held-out log-loss.

    The transform is ``p_T = sigmoid(logit(p) / T)``. ``T < 1`` sharpens
    predictions away from 0.5; ``T > 1`` softens them toward 0.5. Returns
    ``(T, p_T)``. A single-class sample is returned unchanged with ``T = 1``
    because temperature is not identifiable. Numerical optimizer failure is
    an error and is never converted into an identity calibration.
    """
    y_arr, p_raw = _validate_binary_probability_arrays(y, p)
    p_arr = np.clip(p_raw, eps, 1.0 - eps)
    n_classes_required = 2
    if y_arr.size == 0 or np.unique(y_arr).size < n_classes_required:
        return 1.0, p_arr.copy()

    logits = np.log(p_arr / (1.0 - p_arr))

    def _loss(temperature: float) -> float:
        if not np.isfinite(temperature) or temperature <= 0.0:
            return 1e9
        p_t = expit(logits / float(temperature))
        p_t = np.clip(p_t, eps, 1.0 - eps)
        return float(
            -np.mean(y_arr * np.log(p_t) + (1.0 - y_arr) * np.log(1.0 - p_t))
        )

    try:
        result = minimize_scalar(_loss, bounds=bounds, method="bounded")
    except Exception as exc:
        raise RuntimeError(
            "temperature calibration optimizer raised an error"
        ) from exc
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError(
            "temperature calibration optimizer failed: "
            f"{getattr(result, 'message', 'no diagnostic message')}"
        )
    T = float(result.x)
    p_T = expit(logits / T)
    return T, p_T


def calibration_intercept_slope(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Van Calster calibration intercept and slope.

    Fits a logistic regression ``y ~ Bernoulli(sigmoid(a + b * logit(p)))``
    by maximum likelihood. On well-calibrated data the intercept ``a`` is
    near ``0`` and the slope ``b`` is near ``1``. A slope below ``1``
    indicates over-fitting / over-confident predictions; a slope above
    ``1`` indicates under-confident predictions. A negative intercept
    indicates systematic over-prediction; a positive intercept indicates
    under-prediction.

    Parameters
    ----------
    y
        Held-out 0/1 labels.
    p
        Held-out predicted probabilities matching ``y``.
    eps : float, default=1e-9
        Clip used to keep ``logit(p)`` finite.

    Returns
    -------
    dict of str to float
        Dict with ``intercept`` and ``slope`` entries. Both are ``nan``
        when the labels contain fewer than two classes. Numerical optimizer
        failure raises an explicit exception rather than silently returning
        missing estimates.
    """
    from scipy.optimize import minimize  # noqa: PLC0415

    y_arr, p_raw = _validate_binary_probability_arrays(y, p)
    p_arr = np.clip(p_raw, eps, 1.0 - eps)
    n_classes_required = 2
    if y_arr.size == 0 or np.unique(y_arr).size < n_classes_required:
        return {"intercept": float("nan"), "slope": float("nan")}
    logits = np.log(p_arr / (1.0 - p_arr))
    if np.ptp(logits) <= np.sqrt(np.finfo(float).eps):
        return {"intercept": float("nan"), "slope": float("nan")}
    positive_logits = logits[y_arr == 1]
    negative_logits = logits[y_arr == 0]
    if np.max(negative_logits) <= np.min(positive_logits) or np.max(
        positive_logits
    ) <= np.min(negative_logits):
        return {"intercept": float("nan"), "slope": float("nan")}
    design = np.column_stack([np.ones(y_arr.size), logits])

    def _loss(theta: np.ndarray) -> float:
        z = design @ theta
        return float(np.mean(np.logaddexp(0.0, z) - y_arr * z))

    def _score(theta: np.ndarray) -> np.ndarray:
        z = design @ theta
        fitted = expit(z)
        return np.asarray(design.T @ (fitted - y_arr) / y_arr.size)

    try:
        res = minimize(
            _loss,
            x0=np.array([0.0, 1.0]),
            method="BFGS",
            jac=_score,
            options={"gtol": 1e-8, "maxiter": 1_000},
        )
    except Exception as exc:
        raise RuntimeError(
            "calibration intercept/slope optimizer raised an error"
        ) from exc
    theta = np.asarray(res.x, dtype=float)
    score_norm = (
        float(np.linalg.norm(_score(theta), ord=np.inf))
        if np.all(np.isfinite(theta))
        else float("inf")
    )
    if (
        not np.all(np.isfinite(theta))
        or score_norm > _CALIBRATION_SCORE_TOLERANCE
    ):
        raise RuntimeError(
            "calibration intercept/slope optimizer failed: "
            f"{getattr(res, 'message', 'no diagnostic message')}; "
            f"maximum absolute score={score_norm:.6g}"
        )
    return {
        "intercept": float(theta[0]),
        "slope": float(theta[1]),
    }


def calibration_summary(
    y: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 5,
    alpha: float = 0.05,
) -> dict:
    """One-call calibration summary used by the Stage-1 scenario runner.

    Returns a dict with raw and post-temperature ECE / MCE / Brier / log-loss
    plus the fitted temperature, the raw bin table, and the post-temperature
    bin table.
    """
    rows = equal_frequency_reliability_table(y, p, n_bins=n_bins, alpha=alpha)
    n = int(np.asarray(y).size)
    intercept_slope = calibration_intercept_slope(y, p)
    out: dict = {
        "n": n,
        "n_bins": n_bins,
        "ECE": expected_calibration_error(rows),
        "MCE": maximum_calibration_error(rows),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "calibration_intercept": intercept_slope["intercept"],
        "calibration_slope": intercept_slope["slope"],
        "bins_raw": [r.as_dict() for r in rows],
    }
    T, p_T = fit_temperature(y, p)
    rows_T = equal_frequency_reliability_table(
        y, p_T, n_bins=n_bins, alpha=alpha
    )
    out.update(
        {
            "temperature_T": T,
            "ECE_post_temperature": expected_calibration_error(rows_T),
            "MCE_post_temperature": maximum_calibration_error(rows_T),
            "brier_post_temperature": brier_score(y, p_T),
            "log_loss_post_temperature": log_loss(y, p_T),
            "bins_post_temperature": [r.as_dict() for r in rows_T],
        }
    )
    return out


# ---------------------------------------------------------------------------
# Post-hoc calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationMap:
    """Serialisable per-method calibration map fitted on held-out predictions.

    Attributes
    ----------
    method
        One of ``"platt"``, ``"isotonic"``, ``"temperature"``, ``"none"``.
    params
        Method-specific fitted parameters (see :func:`fit_posthoc_calibration`).
    """

    method: str
    params: dict[str, Any]

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Apply the calibration to a new probability vector."""
        p_arr = _as_probability_vector(p)
        if self.method == "none":
            return p_arr
        if self.method == "platt":
            a = float(self.params["a"])
            b = float(self.params["b"])
            p_clip = np.clip(p_arr, 1e-9, 1.0 - 1e-9)
            logits = np.log(p_clip / (1.0 - p_clip))
            return expit(a * logits + b)
        if self.method == "temperature":
            T = float(self.params["T"])
            p_clip = np.clip(p_arr, 1e-9, 1.0 - 1e-9)
            logits = np.log(p_clip / (1.0 - p_clip))
            return expit(logits / T)
        if self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=float)
            ys = np.asarray(self.params["y"], dtype=float)
            return np.interp(p_arr, xs, ys, left=ys[0], right=ys[-1])
        msg = f"unknown calibration method {self.method!r}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Round-trip-friendly dict representation."""
        return {"method": self.method, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationMap:
        """Reconstruct from a :meth:`to_dict` payload."""
        return cls(
            method=str(payload["method"]), params=dict(payload["params"])
        )


def _fit_platt(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p_clip = np.clip(p, 1e-9, 1.0 - 1e-9)
    logits = np.log(p_clip / (1.0 - p_clip))
    from scipy.optimize import minimize  # noqa: PLC0415

    def _loss(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        p_cal = expit(a * logits + b)
        p_cal = np.clip(p_cal, 1e-9, 1.0 - 1e-9)
        return float(
            -np.mean(y * np.log(p_cal) + (1.0 - y) * np.log(1.0 - p_cal))
        )

    try:
        res = minimize(_loss, x0=np.array([1.0, 0.0]), method="BFGS")
    except Exception as exc:
        raise RuntimeError(
            "Platt calibration optimizer raised an error"
        ) from exc
    if not res.success or not np.all(np.isfinite(res.x)):
        raise RuntimeError(
            "Platt calibration optimizer failed: "
            f"{getattr(res, 'message', 'no diagnostic message')}"
        )
    a, b = float(res.x[0]), float(res.x[1])
    return {"a": a, "b": b}


def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict[str, list[float]]:
    from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    # Sample the fitted monotone function on a fine grid for serialisation
    xs = np.linspace(0.0, 1.0, 201)
    ys = iso.predict(xs)
    return {"x": xs.tolist(), "y": ys.tolist()}


def fit_posthoc_calibration(
    p: Sequence[float] | np.ndarray,
    y: Sequence[int] | np.ndarray,
    *,
    method: str = "platt",
) -> CalibrationMap:
    """Fit a post-hoc calibration map on held-out predictions.

    Parameters
    ----------
    p
        Held-out predicted probabilities (typically out-of-fold from a
        spatial-block CV pass).
    y
        Held-out 0/1 labels matching ``p``.
    method
        ``"platt"``, ``"isotonic"``, ``"temperature"``, or ``"none"``.

    Returns
    -------
    CalibrationMap
        Apply with ``CalibrationMap.predict`` to new probability vectors.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    y_arr, p_arr = _validate_binary_probability_arrays(y, p)
    if method == "none":
        return CalibrationMap(method="none", params={})
    if np.unique(y_arr).size < 2:  # noqa: PLR2004
        raise ValueError("post-hoc calibration requires two outcome classes")
    if method == "platt":
        return CalibrationMap(method="platt", params=_fit_platt(p_arr, y_arr))
    if method == "isotonic":
        return CalibrationMap(
            method="isotonic", params=_fit_isotonic(p_arr, y_arr)
        )
    if method == "temperature":
        T, _ = fit_temperature(y_arr, p_arr)
        return CalibrationMap(method="temperature", params={"T": float(T)})
    msg = f"unknown calibration method {method!r}"
    raise ValueError(msg)


__all__ = [
    "CalibrationBin",
    "CalibrationMap",
    "brier_score",
    "calibration_intercept_slope",
    "calibration_summary",
    "equal_frequency_reliability_table",
    "expected_calibration_error",
    "fit_posthoc_calibration",
    "fit_temperature",
    "log_loss",
    "maximum_calibration_error",
    "wilson_ci",
]
