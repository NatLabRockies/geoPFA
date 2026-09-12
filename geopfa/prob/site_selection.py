"""Finite-candidate preferential site-selection likelihood.

This optional model jointly represents a binary outcome and whether each
member of a declared candidate-site frame was sampled. The outcome effect in
the selection equation is not identified from the observed data alone, so its
log-odds coefficient is supplied by the caller and evaluated as a sensitivity
parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp

from geopfa.prob.config import SiteSelectionConfig


def _design(
    values: np.ndarray | None, name: str, n: int | None = None
) -> np.ndarray:
    if values is None:
        raise ValueError(
            f"{name} is required: site selection is identifiable only on a "
            "complete candidate frame"
        )
    raw = np.asarray(values)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain only finite real values")
    result = raw.astype(np.float64, copy=False)
    if result.ndim != 2 or result.shape[0] < 1:  # noqa: PLR2004
        raise ValueError(f"{name} must have shape (n, p)")
    if n is not None and result.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows; got {result.shape[0]}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite real values")
    return result


def _with_intercept(values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(values.shape[0]), values])


@dataclass(frozen=True)
class SiteSelectionFit:
    """Joint outcome/selection fit at one fixed sensitivity parameter."""

    outcome_coef: np.ndarray
    selection_coef: np.ndarray
    outcome_selection_log_odds: float
    success: bool
    log_likelihood: float
    n_candidates: int
    n_selected: int
    n_unselected: int
    optimizer_message: str
    n_iter: int
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    def predict_outcome(self, outcome_features: np.ndarray) -> np.ndarray:
        """Predict outcome probability for candidate-compatible covariates."""
        raw = _design(outcome_features, "outcome_features")
        design = _with_intercept(raw)
        if design.shape[1] != self.outcome_coef.size:
            raise ValueError(
                f"outcome_features has {raw.shape[1]} columns; "
                f"expected {self.outcome_coef.size - 1}"
            )
        return expit(design @ self.outcome_coef)

    def predict_selection(
        self,
        selection_features: np.ndarray,
        *,
        outcome: int,
    ) -> np.ndarray:
        """Predict sampling probability conditional on outcome 0 or 1."""
        if outcome not in {0, 1}:
            raise ValueError("outcome must equal 0 or 1")
        raw = _design(selection_features, "selection_features")
        design = _with_intercept(raw)
        if design.shape[1] != self.selection_coef.size:
            raise ValueError(
                f"selection_features has {raw.shape[1]} columns; "
                f"expected {self.selection_coef.size - 1}"
            )
        return expit(
            design @ self.selection_coef
            + self.outcome_selection_log_odds * outcome
        )


def fit_site_selection_model(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    *,
    outcome_features: np.ndarray,
    selection_features: np.ndarray | None,
    selected: np.ndarray,
    observed_outcome: np.ndarray,
    outcome_selection_log_odds: float,
    outcome_penalty: float = 1e-3,
    selection_penalty: float = 1e-3,
    max_iter: int = 2_000,
) -> SiteSelectionFit:
    """Fit a joint Bernoulli outcome and preferential-selection likelihood.

    For selected candidates the likelihood contains both the observed outcome
    and its outcome-conditional sampling probability. For unselected
    candidates the unobserved outcome is marginalized exactly. The fixed
    ``outcome_selection_log_odds`` parameter is the log odds ratio relating a
    positive outcome to selection after conditioning on selection covariates.
    """
    x_raw = _design(outcome_features, "outcome_features")
    n = x_raw.shape[0]
    q_raw = _design(selection_features, "selection_features", n=n)
    x = _with_intercept(x_raw)
    q = _with_intercept(q_raw)

    selection = np.asarray(selected)
    if (
        selection.shape != (n,)
        or np.iscomplexobj(selection)
        or selection.dtype.kind not in "biuf"
        or not np.all((selection == 0) | (selection == 1))
    ):
        raise ValueError(f"selected must be a binary vector with shape ({n},)")
    selection = selection.astype(bool)
    if not selection.any() or selection.all():
        raise ValueError(
            "candidate frame must contain both selected and unselected sites"
        )

    raw_outcome = np.asarray(observed_outcome)
    if np.iscomplexobj(raw_outcome) or raw_outcome.dtype.kind not in "fiu":
        raise ValueError(
            "observed_outcome must contain binary values at selected sites"
        )
    outcome = raw_outcome.astype(np.float64, copy=False)
    if outcome.shape != (n,):
        raise ValueError(f"observed_outcome must have shape ({n},)")
    if not np.all(np.isfinite(outcome[selection])) or not np.all(
        (outcome[selection] == 0.0) | (outcome[selection] == 1.0)
    ):
        raise ValueError(
            "observed_outcome must be finite binary at every selected site"
        )
    if not np.all(np.isnan(outcome[~selection])):
        raise ValueError(
            "observed_outcome at every unselected candidate must be missing"
        )
    if np.unique(outcome[selection]).size < 2:  # noqa: PLR2004
        raise ValueError("selected sites must contain both outcome classes")

    if isinstance(
        outcome_selection_log_odds, bool | np.bool_
    ) or not isinstance(outcome_selection_log_odds, Real):
        raise TypeError("outcome_selection_log_odds must be finite")
    delta = float(outcome_selection_log_odds)
    if not np.isfinite(delta):
        raise ValueError("outcome_selection_log_odds must be finite")
    if (
        isinstance(outcome_penalty, bool | np.bool_)
        or not isinstance(outcome_penalty, Real)
        or not np.isfinite(outcome_penalty)
        or outcome_penalty < 0.0
    ):
        raise ValueError("outcome_penalty must be non-negative and finite")
    if (
        isinstance(selection_penalty, bool | np.bool_)
        or not isinstance(selection_penalty, Real)
        or not np.isfinite(selection_penalty)
        or selection_penalty < 0.0
    ):
        raise ValueError("selection_penalty must be non-negative and finite")
    if (
        isinstance(max_iter, bool | np.bool_)
        or not isinstance(max_iter, Integral)
        or max_iter < 1
    ):
        raise ValueError("max_iter must be a positive integer")

    y_selected = outcome[selection]
    x_selected = x[selection]
    q_selected = q[selection]
    x_unselected = x[~selection]
    q_unselected = q[~selection]
    p = x.shape[1]

    def likelihood_parts(  # noqa: PLR0914
        theta: np.ndarray,
    ) -> tuple[float, np.ndarray, float]:
        beta = theta[:p]
        gamma = theta[p:]

        eta_y_selected = x_selected @ beta
        prob_y_selected = expit(eta_y_selected)
        eta_s_selected = q_selected @ gamma + delta * y_selected
        prob_s_selected = expit(eta_s_selected)

        eta_y_unselected = x_unselected @ beta
        prob_y_unselected = expit(eta_y_unselected)
        eta_s0_unselected = q_unselected @ gamma
        eta_s1_unselected = eta_s0_unselected + delta
        prob_s0 = expit(eta_s0_unselected)
        prob_s1 = expit(eta_s1_unselected)

        selected_log_likelihood = float(
            np.sum(
                y_selected * log_expit(eta_y_selected)
                + (1.0 - y_selected) * log_expit(-eta_y_selected)
                + log_expit(eta_s_selected)
            )
        )
        unselected_log_terms = np.vstack(
            [
                log_expit(eta_y_unselected) + log_expit(-eta_s1_unselected),
                log_expit(-eta_y_unselected) + log_expit(-eta_s0_unselected),
            ]
        )
        unselected_log_marginal = logsumexp(unselected_log_terms, axis=0)
        log_likelihood = selected_log_likelihood + float(
            np.sum(unselected_log_marginal)
        )

        grad_beta = x_selected.T @ (y_selected - prob_y_selected)
        responsibilities = np.exp(
            unselected_log_terms - unselected_log_marginal[np.newaxis, :]
        )
        prob_y1_given_unselected = responsibilities[0]
        grad_beta += x_unselected.T @ (
            prob_y1_given_unselected - prob_y_unselected
        )

        grad_gamma = q_selected.T @ (1.0 - prob_s_selected)
        grad_gamma -= q_unselected.T @ (
            responsibilities[0] * prob_s1 + responsibilities[1] * prob_s0
        )

        penalized = log_likelihood - 0.5 * outcome_penalty * float(
            np.dot(beta[1:], beta[1:])
        )
        penalized -= (
            0.5 * selection_penalty * float(np.dot(gamma[1:], gamma[1:]))
        )
        grad_beta[1:] -= outcome_penalty * beta[1:]
        grad_gamma[1:] -= selection_penalty * gamma[1:]
        return (
            -penalized,
            -np.concatenate([grad_beta, grad_gamma]),
            log_likelihood,
        )

    optimized = minimize(
        lambda theta: likelihood_parts(theta)[0],
        x0=np.zeros(x.shape[1] + q.shape[1], dtype=np.float64),
        jac=lambda theta: likelihood_parts(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-8},
    )
    if (
        not bool(optimized.success)
        or not np.all(np.isfinite(optimized.x))
        or not np.isfinite(optimized.fun)
    ):
        raise RuntimeError(
            f"site-selection optimization did not converge: {optimized.message}"
        )

    beta = np.asarray(optimized.x[:p], dtype=np.float64)
    gamma = np.asarray(optimized.x[p:], dtype=np.float64)
    unpenalized_log_likelihood = likelihood_parts(optimized.x)[2]
    return SiteSelectionFit(
        outcome_coef=beta,
        selection_coef=gamma,
        outcome_selection_log_odds=delta,
        success=True,
        log_likelihood=float(unpenalized_log_likelihood),
        n_candidates=n,
        n_selected=int(selection.sum()),
        n_unselected=int((~selection).sum()),
        optimizer_message=str(optimized.message),
        n_iter=int(getattr(optimized, "nit", 0)),
        diagnostics={
            "estimand": "candidate_frame_outcome_probability",
            "selection_sensitivity_status": "fixed_by_caller",
            "penalized_log_objective": float(-optimized.fun),
            "outcome_penalty": float(outcome_penalty),
            "selection_penalty": float(selection_penalty),
        },
    )


@dataclass(frozen=True)
class ConfiguredSiteSelectionResult:
    """Sensitivity fits and candidate-frame outcome probabilities."""

    candidate_ids: np.ndarray
    sensitivity_values: np.ndarray
    outcome_probability: np.ndarray
    fits: tuple[SiteSelectionFit, ...]
    n_candidates: int
    n_selected: int
    outcome_center: np.ndarray
    outcome_scale: np.ndarray
    selection_center: np.ndarray
    selection_scale: np.ndarray


def _read_candidate_frame(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"candidate frame does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix in {".gpkg", ".geojson", ".json", ".shp"}:
        import geopandas as gpd  # noqa: PLC0415

        return gpd.read_file(source)
    raise ValueError(
        "candidate_source must be CSV, Parquet, GeoPackage, GeoJSON, or Shapefile"
    )


def _standardize_frame(
    frame: pd.DataFrame, columns: tuple[str, ...], *, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing column(s): {', '.join(missing)}")
    try:
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{name} must be complete and finite for every candidate"
        )
    center = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (values - center) / scale, center, scale


def _reject_reserved_features(
    config: SiteSelectionConfig, *, outcome_column: str
) -> None:
    """Keep identifiers and likelihood targets out of both design matrices."""
    role_columns = (config.id_col, config.selected_col, outcome_column)
    if len(set(role_columns)) != len(role_columns):
        raise ValueError(
            "site-selection identifier, selection, and outcome columns must be distinct"
        )
    reserved = set(role_columns)
    for name, columns in (
        ("outcome_feature_columns", config.outcome_feature_columns),
        ("selection_feature_columns", config.selection_feature_columns),
    ):
        leaked = sorted(reserved.intersection(columns))
        if leaked:
            raise ValueError(
                f"site_selection.{name} must not include reserved "
                "candidate-frame columns: " + ", ".join(leaked)
            )


def run_site_selection_analysis(
    config: SiteSelectionConfig,
    *,
    outcome_column: str,
) -> ConfiguredSiteSelectionResult:
    """Run the configured finite-frame sensitivity analysis.

    This optional analysis does not alter the LatticeKrigX likelihood. It
    estimates candidate-frame outcome probabilities at each externally fixed
    outcome-dependent selection odds ratio.
    """
    if config.mode != "joint_binary":
        raise ValueError(
            "run_site_selection_analysis requires mode='joint_binary'"
        )
    if (
        not isinstance(config.candidate_source, str)
        or not config.candidate_source.strip()
    ):
        raise ValueError(
            "site_selection.candidate_source must be a non-empty string"
        )
    if not isinstance(config.id_col, str) or not config.id_col.strip():
        raise ValueError("site_selection.id_col must be a non-empty string")
    _reject_reserved_features(config, outcome_column=outcome_column)
    frame = _read_candidate_frame(config.candidate_source)
    required = [config.id_col, config.selected_col, outcome_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "candidate frame missing required column(s): " + ", ".join(missing)
        )
    ids = frame[config.id_col]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError(
            "candidate-frame identifiers must be complete and unique"
        )
    selection = frame[config.selected_col].to_numpy()
    if not np.all((selection == 0) | (selection == 1)):
        raise ValueError("selected_col must contain only 0 and 1")
    selected = selection.astype(bool)
    try:
        outcome = pd.to_numeric(
            frame[outcome_column], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{outcome_column} must contain numeric binary outcomes at selected "
            "sites and missing values at unselected candidates"
        ) from exc
    if np.isfinite(outcome[~selected]).any():
        raise ValueError(
            "outcomes at unselected candidates must be missing; mark observed sites selected"
        )
    outcome_x, outcome_center, outcome_scale = _standardize_frame(
        frame,
        config.outcome_feature_columns,
        name="outcome features",
    )
    selection_x, selection_center, selection_scale = _standardize_frame(
        frame,
        config.selection_feature_columns,
        name="selection features",
    )
    fits = tuple(
        site_selection_sensitivity(
            outcome_selection_log_odds=list(config.outcome_selection_log_odds),
            outcome_features=outcome_x,
            selection_features=selection_x,
            selected=selected,
            observed_outcome=outcome,
            outcome_penalty=config.outcome_penalty,
            selection_penalty=config.selection_penalty,
        )
    )
    probabilities = np.column_stack(
        [fit.predict_outcome(outcome_x) for fit in fits]
    )
    return ConfiguredSiteSelectionResult(
        candidate_ids=ids.to_numpy(copy=True),
        sensitivity_values=np.asarray(
            config.outcome_selection_log_odds, dtype=np.float64
        ),
        outcome_probability=probabilities,
        fits=fits,
        n_candidates=len(frame),
        n_selected=int(selected.sum()),
        outcome_center=outcome_center,
        outcome_scale=outcome_scale,
        selection_center=selection_center,
        selection_scale=selection_scale,
    )


def site_selection_sensitivity(
    *,
    outcome_selection_log_odds: tuple[float, ...] | list[float],
    **fit_kwargs,
) -> list[SiteSelectionFit]:
    """Fit the likelihood across a declared selection-sensitivity grid."""
    if not outcome_selection_log_odds:
        raise ValueError(
            "outcome_selection_log_odds sensitivity grid must not be empty"
        )
    return [
        fit_site_selection_model(
            outcome_selection_log_odds=delta, **fit_kwargs
        )
        for delta in outcome_selection_log_odds
    ]


__all__ = [
    "ConfiguredSiteSelectionResult",
    "SiteSelectionFit",
    "fit_site_selection_model",
    "run_site_selection_analysis",
    "site_selection_sensitivity",
]
