"""Block-CV runner for the probabilistic method.

Produces per-fold metrics and out-of-fold (OOF) predictions per component
by re-running the same sequential fit on each fold's training mask and
evaluating on the held-out wells.

The OOF predictions are the right input to post-hoc calibration (Phase F.3)
because they were not seen during training — no optimistic bias from fitting
the calibration map on in-sample probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np

from .alpha import build_alpha_c
from .calibration import calibration_summary
from .config import ProbabilisticConfig
from .cv import spatial_block_cv
from .decision_metrics import auc_tie_safe
from .fitting import fit_component_probability
from .fit_dispatch import build_fit_kwargs
from .labels import load_labels
from .pfa_grid import PFAGridAdapter, validate_declared_components
from .spatial_alignment import (
    align_to_grid_crs,
    extract_coordinates,
    snap_to_grid_indices,
)


@dataclass(frozen=True)
class CVResult:
    """Output of :func:`run_block_cv`.

    Attributes
    ----------
    oof
        ``{component_name: {"p": np.ndarray, "y": np.ndarray, "fold": np.ndarray}}`` —
        out-of-fold probabilities, labels, and fold-index array, concatenated
        across folds.  The ``"fold"`` key maps each prediction to its source
        CV fold (integer 0..n_folds-1) to support per-fold slicing.
    fold_metrics
        ``{component_name: list[dict]}`` — per-fold {auc, brier, log_loss, n}.
    aggregate_metrics
        ``{component_name: dict}`` — pooled metrics over all OOF preds.
    """

    oof: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    fold_metrics: dict[str, list[dict[str, float]]] = field(
        default_factory=dict
    )
    aggregate_metrics: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


def _fit_kwargs_for(
    cfg: ProbabilisticConfig, component: str
) -> dict[str, Any]:
    return build_fit_kwargs(
        alpha_config=cfg.alpha[component],
        evidence_config=cfg.evidence,
        spatial_field_config=cfg.spatial_field,
        pu_mode=cfg.labels.pu_mode,
        pu_class_prior=cfg.labels.class_prior_for(component),
        min_wells=cfg.labels.min_wells_for_fit,
    )


def _component_well_overlay(
    component_data: dict,
    wells: gpd.GeoDataFrame,
    label_column: str,
) -> gpd.GeoDataFrame:
    """Sample wells at their nearest grid cell; return ordered alignment."""
    grid_gdf = component_data["pr_norm"]
    base = wells.to_crs(grid_gdf.crs) if grid_gdf.crs is not None else wells
    indices = snap_to_grid_indices(base, grid_gdf)
    sampled = base[[label_column, "geometry"]].copy()
    sampled["grid_index"] = indices
    return sampled.reset_index(drop=True)


def _evaluate_at_wells(
    surface: gpd.GeoDataFrame,
    wells: gpd.GeoDataFrame,
    label_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the fitted probability surface at well locations.

    Nearest-neighbour matching uses every declared geometry dimension.
    """
    if surface.crs is not None:
        wells = wells.to_crs(surface.crs)
    indices = snap_to_grid_indices(wells, surface)
    y = wells[label_column].astype(float).to_numpy()
    p = surface.iloc[indices]["probability"].astype(float).to_numpy()
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError(
            "held-out labels and probabilities must be finite; refusing to "
            "silently drop scoring rows"
        )
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("held-out labels must contain binary 0/1 values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("held-out probabilities must lie in [0, 1]")
    return y.astype(int), p


def component_oof_predictions(  # noqa: PLR0914
    pfa: dict,
    config: ProbabilisticConfig,
    *,
    component: str,
    label_column: str,
    criteria: str = "geologic",
) -> dict[str, np.ndarray]:
    """Run block-CV for one component and return concatenated OOF (p, y)."""
    config.validate_raise()
    adapter = PFAGridAdapter(
        pfa, criteria=criteria, dimensions=config.dimensions
    )
    validate_declared_components(
        adapter,
        set(config.labels.label_columns) | set(config.alpha),
    )
    configured_label = config.labels.label_columns.get(component)
    if configured_label is None:
        raise ValueError(
            f"component {component!r} has no configured label column"
        )
    if label_column != configured_label:
        raise ValueError(
            f"component {component!r} configured label column is "
            f"{configured_label!r}, not {label_column!r}"
        )
    labels = load_labels(config.labels)
    comp_data = adapter.component_data(component)
    grid_gdf = comp_data["pr_norm"]
    grid_crs = grid_gdf.crs
    if (
        config.cross_validation.buffer_km > 0.0
        or config.cross_validation.block_size_km is not None
    ) and (
        grid_crs is None
        or not grid_crs.is_projected
        or any(
            axis.unit_name.lower() not in {"metre", "meter"}
            for axis in grid_crs.axis_info[:2]
        )
    ):
        raise ValueError(
            "cross_validation block_size_km/buffer_km controls require a "
            "projected metre-based CRS"
        )

    wells = align_to_grid_crs(labels.gdf, grid_gdf)
    coords = extract_coordinates(wells)
    kwargs_base = _fit_kwargs_for(config, component)
    alpha_result = build_alpha_c(
        comp_data, config.alpha[component], grid_gdf=comp_data["pr_norm"]
    )
    extra_excluded = set(alpha_result.excluded_layer_names) | set(
        kwargs_base.get("excluded_layer_names", ())
    )
    kwargs_base["excluded_layer_names"] = tuple(sorted(extra_excluded))
    kwargs_base["alpha_offset"] = alpha_result.grid_offset
    kwargs_base["prior_layer_name"] = None

    all_p: list[float] = []
    all_y: list[int] = []
    all_fold: list[int] = []
    all_reference_prevalence: list[float] = []
    for fold_i, (train_mask, test_mask) in enumerate(
        spatial_block_cv(
            coords,
            n_folds=config.cross_validation.n_folds,
            block_type=config.cross_validation.block_type,
            grid_size=config.cross_validation.grid_size,
            seed=0,
            block_size_km=config.cross_validation.block_size_km,
            buffer_distance=config.cross_validation.buffer_km * 1000.0,
            dims=(0, 1),
        )
    ):
        train_wells = wells.iloc[np.flatnonzero(train_mask)].copy()
        test_wells = wells.iloc[np.flatnonzero(test_mask)].copy()
        fitted = fit_component_probability(
            comp_data,
            labeled_wells=train_wells,
            label_column=label_column,
            **kwargs_base,
        )
        y, p = _evaluate_at_wells(fitted.probability, test_wells, label_column)
        all_p.extend(p.tolist())
        all_y.extend(y.tolist())
        all_fold.extend([fold_i] * len(y))
        train_y = train_wells[label_column].to_numpy(dtype=float)
        train_y = train_y[np.isfinite(train_y)]
        if train_y.size == 0:
            raise ValueError(f"fold {fold_i} has no finite training labels")
        all_reference_prevalence.extend([float(np.mean(train_y))] * len(y))
    return {
        "p": np.asarray(all_p, dtype=float),
        "y": np.asarray(all_y, dtype=int),
        "fold": np.asarray(all_fold, dtype=int),
        "reference_prevalence": np.asarray(
            all_reference_prevalence, dtype=float
        ),
    }


def run_block_cv(
    pfa: dict,
    config: ProbabilisticConfig,
    *,
    criteria: str = "geologic",
) -> CVResult:
    """Run block CV for every component with a label mapping; return per-fold metrics + OOF."""
    config.validate_raise()
    oof_all: dict[str, dict[str, np.ndarray]] = {}
    fold_metrics: dict[str, list[dict[str, float]]] = {}
    aggregate_metrics: dict[str, dict[str, float]] = {}

    adapter = PFAGridAdapter(
        pfa, criteria=criteria, dimensions=config.dimensions
    )
    validate_declared_components(
        adapter,
        set(config.labels.label_columns) | set(config.alpha),
    )

    for name in adapter.components():
        if name not in config.labels.label_columns:
            continue
        if name not in config.alpha:
            continue
        label_column = config.labels.label_columns[name]
        oof = component_oof_predictions(
            pfa,
            config,
            component=name,
            label_column=label_column,
            criteria=criteria,
        )
        oof_all[name] = oof

        # Compute per-fold metrics by slicing the OOF arrays by fold index.
        per_fold: list[dict[str, float]] = []
        fold_idx = oof.get("fold", np.array([], dtype=int))
        for fi in range(config.cross_validation.n_folds):
            mask = fold_idx == fi
            if not mask.any():
                continue
            fold_p = oof["p"][mask]
            fold_y = oof["y"][mask]
            per_fold.append(
                _basic_metrics(
                    fold_y,
                    fold_p,
                    reference_prevalence=oof["reference_prevalence"][mask],
                )
            )
        fold_metrics[name] = per_fold

        if len(oof["p"]) > 0:
            aggregate_metrics[name] = _basic_metrics(
                oof["y"],
                oof["p"],
                reference_prevalence=oof["reference_prevalence"],
            )
        else:
            aggregate_metrics[name] = {
                "auc": float("nan"),
                "brier": float("nan"),
                "log_loss": float("nan"),
                "brier_skill": float("nan"),
                "ece": float("nan"),
                "calibration_intercept": float("nan"),
                "calibration_slope": float("nan"),
                "n": 0,
            }

    return CVResult(
        oof=oof_all,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
    )


def _basic_metrics(
    y: np.ndarray,
    p: np.ndarray,
    *,
    reference_prevalence: np.ndarray | float | None = None,
) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.ndim != 1 or p_arr.ndim != 1 or y_arr.shape != p_arr.shape:
        raise ValueError("y and p must be aligned one-dimensional arrays")
    if len(y_arr) == 0:
        return {
            "auc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "brier_skill": float("nan"),
            "ece": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
            "n": len(y_arr),
        }
    if not np.all(np.isfinite(y_arr)) or not np.all(np.isfinite(p_arr)):
        raise ValueError("y and p must contain only finite values")
    if not np.all(np.isin(y_arr, (0.0, 1.0))):
        raise ValueError("y must contain binary 0/1 labels")
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("predicted probabilities must lie in [0, 1]")
    y_arr = y_arr.astype(int)
    brier = float(np.mean((y_arr - p_arr) ** 2))
    if reference_prevalence is None:
        reference = np.full(len(y_arr), float(np.mean(y_arr)))
    else:
        reference = np.broadcast_to(
            np.asarray(reference_prevalence, dtype=float), y_arr.shape
        )
        if not np.all(np.isfinite(reference)) or np.any(
            (reference < 0.0) | (reference > 1.0)
        ):
            raise ValueError("reference prevalence must lie in [0, 1]")
    reference_brier = float(np.mean((y_arr - reference) ** 2))
    summary = calibration_summary(y_arr, p_arr, n_bins=min(5, len(y_arr)))
    p_log = np.clip(p_arr, 1e-9, 1.0 - 1e-9)
    return {
        "auc": (
            float(auc_tie_safe(y_arr, p_arr))
            if len(np.unique(y_arr)) == 2  # noqa: PLR2004
            else float("nan")
        ),
        "brier": brier,
        "log_loss": float(
            -np.mean(
                y_arr * np.log(p_log) + (1.0 - y_arr) * np.log(1.0 - p_log)
            )
        ),
        "brier_skill": (
            1.0 - brier / reference_brier
            if reference_brier > 0.0
            else float("nan")
        ),
        "ece": float(summary["ECE"]),
        "calibration_intercept": float(summary["calibration_intercept"]),
        "calibration_slope": float(summary["calibration_slope"]),
        "n": len(y_arr),
    }


__all__ = ["CVResult", "component_oof_predictions", "run_block_cv"]
