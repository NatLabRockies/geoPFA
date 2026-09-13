"""Componentwise predictive stacking against a fixed prior surface."""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter

import numpy as np
from scipy.optimize import minimize_scalar


_PROBABILITY_EPSILON = 1e-12
_DRAW_ARRAY_DIMENSIONS = 2


@dataclass(frozen=True)
class PredictiveStackingResult:
    """Weight and proper-score diagnostics for one component."""

    weight: float
    prior_log_score: float | None
    full_log_score: float | None
    selected_log_score: float | None
    n_observations: int
    n_wells: int | None
    status: str
    validation_depth_m: float | None = None


def _validated_binary_inputs(
    outcomes: np.ndarray,
    prior_probability: np.ndarray,
    full_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(outcomes, dtype=float)
    prior = np.asarray(prior_probability, dtype=float)
    full = np.asarray(full_probability, dtype=float)
    if y.ndim != 1 or prior.shape != y.shape or full.shape != y.shape:
        raise ValueError(
            "outcomes, prior_probability, and full_probability must be "
            "one-dimensional arrays with identical shapes"
        )
    if y.size == 0:
        raise ValueError(
            "predictive stacking requires at least one observation"
        )
    if not np.all(np.isfinite(y)) or not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("outcomes must contain finite binary values")
    for name, values in (
        ("prior_probability", prior),
        ("full_probability", full),
    ):
        if not np.all(np.isfinite(values)) or np.any(
            (values < 0.0) | (values > 1.0)
        ):
            raise ValueError(f"{name} must contain finite values in [0, 1]")
    return y, prior, full


def _binary_log_score(outcomes: np.ndarray, probability: np.ndarray) -> float:
    probability = np.clip(
        probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
    )
    losses = -(
        outcomes * np.log(probability)
        + (1.0 - outcomes) * np.log1p(-probability)
    )
    return float(np.mean(losses))


def select_predictive_stacking_weight(
    outcomes: np.ndarray,
    prior_probability: np.ndarray,
    full_probability: np.ndarray,
) -> PredictiveStackingResult:
    """Minimize held-out binary log score over a prior/update mixture.

    The returned weight is the contribution from the fitted spatial update.
    Zero retains the configured prior and one retains the full update.
    """
    y, prior, full = _validated_binary_inputs(
        outcomes, prior_probability, full_probability
    )

    def objective(weight: float) -> float:
        probability = (1.0 - weight) * prior + weight * full
        return _binary_log_score(y, probability)

    optimum = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not optimum.success or not np.isfinite(optimum.fun):
        raise RuntimeError("predictive-stacking optimization failed")

    candidates = (
        (0.0, objective(0.0)),
        (float(optimum.x), float(optimum.fun)),
        (1.0, objective(1.0)),
    )
    weight, selected_score = min(candidates, key=itemgetter(1))
    if np.isclose(weight, 0.0, atol=1e-8):
        weight = 0.0
    elif np.isclose(weight, 1.0, atol=1e-8):
        weight = 1.0

    return PredictiveStackingResult(
        weight=weight,
        prior_log_score=objective(0.0),
        full_log_score=objective(1.0),
        selected_log_score=selected_score,
        n_observations=int(y.size),
        n_wells=None,
        status="estimated",
    )


def apply_predictive_stacking(
    prior_probability: np.ndarray,
    full_probability_draws: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Mix every full-model event-probability draw with the prior surface."""
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be a finite value in [0, 1]")
    prior = np.asarray(prior_probability, dtype=float)
    draws = np.asarray(full_probability_draws, dtype=float)
    if (
        prior.ndim != 1
        or draws.ndim != _DRAW_ARRAY_DIMENSIONS
        or draws.shape[1] != prior.size
    ):
        raise ValueError(
            "prior_probability must be one-dimensional and match the second "
            "dimension of full_probability_draws"
        )
    if not np.all(np.isfinite(prior)) or np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError(
            "prior_probability must contain finite values in [0, 1]"
        )
    if not np.all(np.isfinite(draws)) or np.any((draws < 0.0) | (draws > 1.0)):
        raise ValueError(
            "full_probability_draws must contain finite values in [0, 1]"
        )
    return (1.0 - weight) * prior[np.newaxis, :] + weight * draws


__all__ = [
    "PredictiveStackingResult",
    "apply_predictive_stacking",
    "select_predictive_stacking_weight",
]
