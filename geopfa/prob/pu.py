"""Non-negative positive--unlabeled (nnPU) logistic risk estimation.

The estimator in this module targets an outcome model under the
selected-completely-at-random positive-label mechanism. It does not model
where observations were collected; preferential site selection is a separate
problem implemented in :mod:`geopfa.prob.site_selection`.

References
----------
Kiryo, R. et al. (2017). Positive-Unlabeled Learning with Non-Negative Risk
Estimator. NeurIPS 30.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class NNPULogitResult:
    """Fitted nnPU logistic model and its empirical-risk diagnostics."""

    coef: np.ndarray
    class_prior: float
    objective: float
    positive_risk: float
    negative_risk: float
    raw_negative_risk: float
    n_positive: int
    n_unlabeled: int
    success: bool
    optimizer_message: str
    n_iter: int

    @property
    def x(self) -> np.ndarray:
        """Coefficient vector, matching SciPy optimizer terminology."""
        return self.coef

    def predict_proba(
        self,
        features: np.ndarray,
        *,
        offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict latent-positive probabilities for a compatible design."""
        design = _finite_design(features)
        if design.shape[1] != self.coef.size:
            raise ValueError(
                f"features has {design.shape[1]} columns; expected {self.coef.size}"
            )
        offset = _finite_offsets(offsets, design.shape[0])
        return expit(offset + design @ self.coef)


def _finite_design(features: np.ndarray) -> np.ndarray:
    raw = np.asarray(features)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError("features must contain only finite real values")
    design = raw.astype(np.float64, copy=False)
    if design.ndim != 2 or design.shape[0] < 1 or design.shape[1] < 1:  # noqa: PLR2004
        raise ValueError(
            "features must have shape (n, p) with n and p positive"
        )
    if not np.all(np.isfinite(design)):
        raise ValueError("features must contain only finite real values")
    return design


def _finite_offsets(offsets: np.ndarray | None, n: int) -> np.ndarray:
    if offsets is None:
        return np.zeros(n, dtype=np.float64)
    raw = np.asarray(offsets)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError("offsets must contain only finite real values")
    result = raw.astype(np.float64, copy=False)
    if result.shape != (n,) or not np.all(np.isfinite(result)):
        raise ValueError(f"offsets must be a finite vector with shape ({n},)")
    return result


def _class_prior(value: float | None) -> float:
    if value is None:
        raise ValueError(
            "class_prior is required for nnPU; it must come from external "
            "knowledge or an explicitly identified estimator"
        )
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise TypeError(
            "class_prior must be a finite probability strictly between 0 and 1"
        )
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(
            "class_prior must be a finite probability strictly between 0 and 1"
        )
    return result


def fit_nnpu_logistic(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    features: np.ndarray,
    observed_positive: np.ndarray,
    *,
    class_prior: float | None,
    offsets: np.ndarray | None = None,
    penalty_weights: float | np.ndarray = 1e-3,
    prior_means: np.ndarray | None = None,
    max_iter: int = 1_000,
    tolerance: float = 1e-9,
) -> NNPULogitResult:
    """Minimize the non-negative PU logistic empirical risk.

    ``observed_positive`` is one for a labelled positive and zero for an
    unlabelled observation. The supplied ``class_prior`` is the population
    prevalence ``P(Y=1)``; estimating it from these same labels without an
    additional identification argument is deliberately unsupported.

    The objective is ``pi E_P[softplus(-eta)] + max(0,
    E_X[softplus(eta)] - pi E_P[softplus(eta)]) + ridge``, where ``E_X``
    averages over the complete marginal candidate covariate sample. It is not
    the conditional distribution of rows whose label indicator is zero.

    Callers control intercept handling explicitly by including an all-ones
    design column and assigning that column zero penalty when desired.
    """
    design = _finite_design(features)
    n, p = design.shape
    prior = _class_prior(class_prior)
    offset = _finite_offsets(offsets, n)

    labels = np.asarray(observed_positive)
    if labels.shape != (n,):
        raise ValueError(f"observed_positive must have shape ({n},)")
    if (
        np.iscomplexobj(labels)
        or labels.dtype.kind not in "biuf"
        or not np.all((labels == 0) | (labels == 1))
    ):
        raise ValueError(
            "observed_positive must contain only binary 0/1 values"
        )
    positive = labels.astype(bool)
    unlabeled = ~positive
    if not positive.any() or not unlabeled.any():
        raise ValueError(
            "nnPU requires at least one labelled positive and one unlabelled row"
        )

    if isinstance(penalty_weights, Real) and not isinstance(
        penalty_weights, bool | np.bool_
    ):
        penalties = np.full(p, float(penalty_weights), dtype=np.float64)
    else:
        raw_penalties = np.asarray(penalty_weights)
        if (
            np.iscomplexobj(raw_penalties)
            or raw_penalties.dtype.kind not in "fiu"
        ):
            raise ValueError(
                f"penalty_weights must be non-negative and have shape ({p},)"
            )
        penalties = raw_penalties.astype(np.float64, copy=False)
    if (
        penalties.shape != (p,)
        or not np.all(np.isfinite(penalties))
        or np.any(penalties < 0.0)
    ):
        raise ValueError(
            f"penalty_weights must be non-negative and have shape ({p},)"
        )
    if prior_means is None:
        means = np.zeros(p, dtype=np.float64)
    else:
        raw_means = np.asarray(prior_means)
        if np.iscomplexobj(raw_means) or raw_means.dtype.kind not in "fiu":
            raise ValueError(
                f"prior_means must be finite and have shape ({p},)"
            )
        means = raw_means.astype(np.float64, copy=False)
    if means.shape != (p,) or not np.all(np.isfinite(means)):
        raise ValueError(f"prior_means must be finite and have shape ({p},)")
    if (
        isinstance(max_iter, bool | np.bool_)
        or not isinstance(max_iter, Integral)
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(tolerance, bool | np.bool_)
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
    ):
        raise ValueError("tolerance must be positive and finite")

    x_pos = design[positive]
    # The marginal-risk term is evaluated on the full candidate covariate
    # sample.  Using only S=0 rows conditions on not being labelled and is not
    # the P(X) expectation in the unbiased PU-risk identity.
    x_marginal = design
    o_pos = offset[positive]
    o_marginal = offset

    def terms(beta: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
        eta_pos = o_pos + x_pos @ beta
        eta_marginal = o_marginal + x_marginal @ beta
        prob_pos = expit(eta_pos)
        prob_marginal = expit(eta_marginal)
        positive_risk = prior * float(np.mean(np.logaddexp(0.0, -eta_pos)))
        raw_negative = float(
            np.mean(np.logaddexp(0.0, eta_marginal))
            - prior * np.mean(np.logaddexp(0.0, eta_pos))
        )
        positive_gradient = prior * np.mean(
            (prob_pos - 1.0)[:, None] * x_pos,
            axis=0,
        )
        negative_gradient = np.mean(
            prob_marginal[:, None] * x_marginal, axis=0
        ) - prior * np.mean(prob_pos[:, None] * x_pos, axis=0)
        return (
            positive_risk,
            raw_negative,
            positive_gradient,
            negative_gradient,
        )

    def objective(beta: np.ndarray) -> float:
        positive_risk, raw_negative, _positive_gradient, _negative_gradient = (
            terms(beta)
        )
        diff = beta - means
        return float(
            positive_risk
            + max(0.0, raw_negative)
            + 0.5 * np.dot(penalties * diff, diff)
        )

    def gradient(beta: np.ndarray) -> np.ndarray:
        _positive_risk, raw_negative, positive_gradient, negative_gradient = (
            terms(beta)
        )
        result = positive_gradient.copy()
        if raw_negative > 0.0:
            result += negative_gradient
        result += penalties * (beta - means)
        return result

    optimized = minimize(
        objective,
        x0=means.copy(),
        jac=gradient,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iter),
            "ftol": float(tolerance),
            "gtol": float(tolerance),
        },
    )
    if not bool(optimized.success) or not np.all(np.isfinite(optimized.x)):
        raise RuntimeError(
            f"nnPU optimization did not converge: {optimized.message}"
        )

    positive_risk, raw_negative, _positive_gradient, _negative_gradient = (
        terms(optimized.x)
    )
    return NNPULogitResult(
        coef=np.asarray(optimized.x, dtype=np.float64),
        class_prior=prior,
        objective=float(optimized.fun),
        positive_risk=float(positive_risk),
        negative_risk=float(max(0.0, raw_negative)),
        raw_negative_risk=float(raw_negative),
        n_positive=int(positive.sum()),
        n_unlabeled=int(unlabeled.sum()),
        success=True,
        optimizer_message=str(optimized.message),
        n_iter=int(getattr(optimized, "nit", 0)),
    )


__all__ = ["NNPULogitResult", "fit_nnpu_logistic"]
