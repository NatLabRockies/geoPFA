"""Probabilistic component fits per geoPFA_prob.memo.20260612.pdf (near-term stage).

Implements:
  logit p_c(s,z) = alpha_c(s,z) + Sigma_k beta_ck x_k(s,z) + u_c(s)
                    physics-prior   evidence-layers          spatial-field

Near-term stage:
  - alpha_c: fixed offset derived from thermal model (pr0 in config)
  - beta_ck: learned from labeled wells via logistic regression
  - u_c: sequential approximation (GP-like smooth residual spatial field)

Evidence layers (beta_ck predictors) exclude spatial coordinates and sparse binary layers;
spatial structure is modeled separately via u_c, not mixed into the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import minimize
from scipy.special import expit

from .labels import _coerce_binary_labels
from .pfa_grid import component_grid
from .pu import fit_nnpu_logistic
from .spatial import fit_spatial_field_gp
from .spatial_alignment import (
    extract_coordinates,
    grid_values_on_reference,
    sample_layer_at_points,
    snap_to_grid_indices,
)


@dataclass(frozen=True)
class ComponentProbability:
    """Probability surface and fit metadata for one component."""

    probability: Any
    model: Any
    feature_names: tuple[str, ...]
    spatial_field: np.ndarray | None = None
    diagnostics: dict | None = None  # fit-time metadata, e.g. PU c_estimate


def _is_sparse_binary(values: np.ndarray, threshold: float = 0.90) -> bool:
    """Check if a column is sparse and binary-like (dominates spatial structure).

    Parameters
    ----------
    values
        Raw layer values.
    threshold
        Dominant-value threshold; defaults to 0.90, so a binary-like layer is
        rejected when at least 90% of finite cells share one value.
    """
    # Remove NaNs
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return False

    # Check if mostly binary (only two distinct values)
    unique_vals = np.unique(valid)
    max_unique_for_binary = 3
    if len(unique_vals) > max_unique_for_binary:
        return False

    # Check if sparse (large regions with one value, small regions with another)
    counts = np.array([np.sum(valid == v) for v in unique_vals])
    max_frac = counts.max() / len(valid)
    return bool(max_frac >= threshold)


_DEFAULT_COORD_BLACKLIST = frozenset(
    {
        "inverted_y",
        "x",
        "y",
        "X",
        "Y",
        "Z",
        "z",
        "longitude",
        "latitude",
        "easting",
        "northing",
        "row",
        "col",
    }
)


def _flatten_component_features(
    component_data: dict[str, Any],
    *,
    included_layer_names: set[str] | None = None,
    excluded_layer_names: set[str] | None = None,
    sparse_binary_threshold: float = 0.90,
    coordinate_blacklist: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from component's evidence layers (not including spatial).

    Per the near-term design, evidence layers are predictors beta_ck;
    spatial structure is handled separately via u_c (spatial random effect).
    This excludes:

    - Sparse binary layers (e.g., quaternary_faults) which should be
      spatial, not predictors. Controlled by ``sparse_binary_threshold``
      (reject when the dominant-value fraction reaches the threshold).
    - Spatial coordinates (which are handled via sequential GP fit).
    - Any non-evidence numeric columns (e.g., ``inverted_y`` coordinate
      columns). Controlled by ``coordinate_blacklist``.
    """
    excluded_layer_names = excluded_layer_names or set()
    reference_grid = component_grid(component_data)
    if reference_grid is None:
        raise ValueError(
            "component has no usable grid: pr_norm is absent and no layer "
            "model GeoDataFrame is available"
        )
    non_evidence_columns: set[str] = set(
        coordinate_blacklist
        if coordinate_blacklist is not None
        else _DEFAULT_COORD_BLACKLIST
    )

    features: list[np.ndarray] = []
    names: list[str] = []

    for layer_name, layer_data in component_data["layers"].items():
        if (
            included_layer_names is not None
            and layer_name not in included_layer_names
        ):
            continue
        if layer_name in excluded_layer_names:
            continue
        model = layer_data["model"]
        # Use only the layer's declared evidence column, or the canonical
        # processed column when no declaration exists. An arbitrary numeric
        # fallback can silently substitute coordinates or diagnostics for a
        # misspelled scientific variable.
        evidence_col = layer_data.get("model_data_col") or "value_interpolated"
        if evidence_col not in model.columns:
            raise ValueError(
                f"evidence layer {layer_name!r} declares column "
                f"{evidence_col!r}, but that column is absent from its model"
            )

        if evidence_col in non_evidence_columns:
            continue

        values = grid_values_on_reference(
            reference_grid,
            model,
            evidence_col,
            context=f"evidence layer {layer_name!r}",
        )

        # Skip sparse binary layers; they should be handled via spatial term, not predictors
        if _is_sparse_binary(values, threshold=sparse_binary_threshold):
            continue

        features.append(values)
        names.append(layer_name)

    if not features:
        raise ValueError(
            "No numeric layer features found — all layers were excluded "
            "(sparse binary, coordinate column, or explicitly excluded). "
            "Check excluded_layer_names, coordinate_blacklist, and "
            "sparse_binary_threshold in the config."
        )
    return np.column_stack(features), names


def _standardize_from_training(
    training: np.ndarray,
    prediction: np.ndarray,
    *,
    min_finite: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature scaling on training rows and apply it to both matrices."""

    train = np.asarray(training, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    matrix_ndim = 2
    if (
        train.ndim != matrix_ndim
        or pred.ndim != matrix_ndim
        or train.shape[1] != pred.shape[1]
    ):
        raise ValueError(
            "training and prediction evidence must be aligned 2-D matrices"
        )
    if np.any(np.isinf(train)) or np.any(np.isinf(pred)):
        raise ValueError("evidence features must not contain infinite values")
    finite_counts = np.sum(np.isfinite(train), axis=0)
    if np.any(finite_counts < min_finite):
        bad = np.flatnonzero(finite_counts < min_finite).tolist()
        raise ValueError(
            "evidence features have fewer than "
            f"{min_finite} finite training wells in column(s) {bad}"
        )
    center = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    train_imputed = np.where(np.isfinite(train), train, center)
    pred_imputed = np.where(np.isfinite(pred), pred, center)
    return (
        (train_imputed - center) / scale,
        (pred_imputed - center) / scale,
        center,
        scale,
        finite_counts,
    )


def _fit_offset_logit(  # noqa: PLR0913
    features: np.ndarray,
    labels: np.ndarray,
    offset: np.ndarray,
    regularization: float = 1e-2,
    *,
    per_feature_weights: np.ndarray | None = None,
    prior_means: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> Any:
    """Fit offset-logit model: logit p = alpha (offset) + X @ beta.

    Parameters
    ----------
    features
        Design matrix ``(n, k)``.
    labels
        Binary 0/1 labels ``(n,)``.
    offset
        Per-sample alpha offset ``(n,)`` (added to the linear predictor).
    regularization
        Scalar L2 penalty applied uniformly to every coefficient (legacy
        behaviour). Disabled by setting to 0; combine with per-feature
        weights below for fine-grained control.
    per_feature_weights
        Optional ``(k,)`` array of per-feature L2 penalty weights. When
        supplied, the penalty becomes
        ``0.5 * sum_k w_k * (beta_k - mu_k)^2``.
    prior_means
        Optional ``(k,)`` array of MAP prior means used together with
        ``per_feature_weights``. Defaults to all-zero (ridge regression).
    sample_weights
        Optional ``(n,)`` array of predeclared sampling weights. This is not
        a PU correction; PU estimation uses its own empirical-risk objective.
    """
    k = features.shape[1]
    w = (
        np.asarray(per_feature_weights, dtype=float)
        if per_feature_weights is not None
        else np.full(k, float(regularization))
    )
    mu = (
        np.asarray(prior_means, dtype=float)
        if prior_means is not None
        else np.zeros(k)
    )
    sw = (
        np.asarray(sample_weights, dtype=float)
        if sample_weights is not None
        else np.ones(len(labels), dtype=float)
    )

    def objective(beta: np.ndarray) -> float:
        eta = offset + features @ beta
        prob = expit(eta)
        prob = np.clip(prob, 1e-9, 1.0 - 1e-9)
        nll = -np.sum(
            sw * (labels * np.log(prob) + (1.0 - labels) * np.log(1.0 - prob))
        )
        diff = beta - mu
        penalty = 0.5 * float(np.dot(w * diff, diff))
        return float(nll + penalty)

    def gradient(beta: np.ndarray) -> np.ndarray:
        eta = offset + features @ beta
        prob = expit(eta)
        grad = features.T @ (sw * (prob - labels))
        grad += w * (beta - mu)
        return grad

    result = minimize(
        objective,
        x0=np.zeros(features.shape[1], dtype=float),
        jac=gradient,
        method="BFGS",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(
            f"offset-logistic optimization did not converge: {result.message}"
        )
    return result


def _fit_spatial_field_to_targets(
    source_coords: np.ndarray,
    residual: np.ndarray,
    target_coords: np.ndarray,
) -> np.ndarray:
    """Fit a D-dimensional RBF field and evaluate it at target points."""

    source = np.asarray(source_coords, dtype=float)
    target = np.asarray(target_coords, dtype=float)
    residual = np.asarray(residual, dtype=float)
    if (
        source.ndim != 2  # noqa: PLR2004
        or target.ndim != 2  # noqa: PLR2004
        or source.shape[1] != target.shape[1]
    ):
        raise ValueError(
            "source_coords and target_coords must have matching (n, D) shapes"
        )
    valid = np.all(np.isfinite(source), axis=1) & np.isfinite(residual)
    min_anchor_points = 4
    if valid.sum() < min_anchor_points:
        raise ValueError(
            f"at least {min_anchor_points} finite anchors are required for an RBF field"
        )

    anchors = source[valid]
    rs = residual[valid]
    smooth = max(float(np.nanvar(rs) * 0.25), 1e-3)
    try:
        rbf = Rbf(
            *[anchors[:, axis] for axis in range(anchors.shape[1])],
            rs,
            function="multiquadric",
            smooth=smooth,
        )
        u = rbf(*[target[:, axis] for axis in range(target.shape[1])])
        return np.clip(np.asarray(u, dtype=float), -3.0, 3.0)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise RuntimeError(f"RBF spatial field fit failed: {exc}") from exc


def fit_component_probability(  # noqa: PLR0912, PLR0913, PLR0914, PLR0915
    component_data: dict[str, Any],
    *,
    prior_probability: float,
    include_spatial: bool = False,
    included_layer_names: tuple[str, ...] | None = None,
    excluded_layer_names: tuple[str, ...] = (),
    labeled_wells: gpd.GeoDataFrame | None = None,
    label_column: str | None = None,
    prior_layer_name: str | None = None,
    prior_p_min: float = 0.2,
    prior_p_max: float = 0.8,
    alpha_offset: np.ndarray | None = None,
    spatial_backend: str = "rbf",
    sparse_binary_threshold: float = 0.90,
    coordinate_blacklist: tuple[str, ...] | None = None,
    force_prior_predictive: bool = False,
    per_feature_weights: dict[str, float] | None = None,
    prior_means: dict[str, float] | None = None,
    pu_mode: str = "off",
    pu_class_prior: float | None = None,
    min_wells: int = 4,
) -> ComponentProbability:
    """Fit near-term probabilistic component model.

    Structure: ``logit p_c = alpha_c + Sigma beta_ck x_k + u_c``

    The prior offset ``alpha_c`` is determined in this order of precedence:

    1. ``alpha_offset`` array (when supplied) — used verbatim. This is the
       hook the config-driven runner uses to inject a per-cell offset built
       by :func:`geopfa.prob.alpha.build_alpha_c` (any mode).
    2. ``prior_layer_name`` — min-max rescale that layer's
       ``value_interpolated`` to ``[prior_p_min, prior_p_max]`` and apply
       ``logit``. The layer is automatically added to ``excluded_layer_names``.
    3. ``prior_probability`` scalar — ``logit(prior_probability)`` applied
       uniformly.

    ``Sigma beta_ck x_k`` is the evidence-layer logistic regression.
    ``u_c`` is an optional spatial residual field fit by an RBF smoother.

    Training requires ``labeled_wells`` and ``label_column``. Prior-predictive
    inference is available only through the explicit
    ``force_prior_predictive=True`` configuration.

    Parameters
    ----------
    component_data : dict
        Component config with layers and a component or layer-model grid.
    prior_probability : float
        Scalar ``P(component | no wells)``, used if no prior layer.
    include_spatial : bool, optional
        If True, fit spatial residual field ``u_c(s)``.
    included_layer_names : tuple of str, optional
        Explicit allowlist of layers eligible for the evidence design.
    excluded_layer_names : tuple of str, optional
        Layers to exclude from fit.
    labeled_wells : geopandas.GeoDataFrame, optional
        GeoDataFrame of labelled wells (preferred training data).
    label_column : str, optional
        Column on ``labeled_wells`` with 0/1 labels for this component.
    prior_layer_name : str, optional
        Optional layer name to use as a spatial prior offset. If provided,
        the layer must exist; it overrides the scalar ``prior_probability``
        and is excluded from regression predictors.
    prior_p_min : float, optional
        Minimum probability bound for the spatial-prior rescale.
    prior_p_max : float, optional
        Maximum probability bound for the spatial-prior rescale.
    alpha_offset : numpy.ndarray, optional
        Pre-built per-cell prior offset array (length ``len(component_grid)``).
        When supplied, takes precedence over ``prior_layer_name`` and
        ``prior_probability``. The runner uses this hook to plumb the
        output of :func:`geopfa.prob.alpha.build_alpha_c` through any
        alpha mode (scalar / layer_logit / thermal_exceedance /
        multi_layer).
    pu_mode : str, optional
        Positive-Unlabeled correction mode.  One of:

        * ``"off"`` — no correction (default).
        * ``"naive_pseudo_absence"`` — treat unlabelled wells as true
          negatives; equivalent to ``"off"`` at the weight level.
        * ``"nnpu"`` — non-negative PU empirical-risk minimization under
          SCAR, requiring an externally identified ``pu_class_prior``.
    min_wells : int, optional
        Minimum number of overlapping labelled wells required before fitting
        the regression. The fit fails closed if fewer wells overlap the grid.
        Default is 4; use
        ``LabelsConfig.min_wells_for_fit`` from the config.
    Returns
    -------
    ComponentProbability
        Probability surface and model metadata.
    """
    # Determine prior offset (precedence: alpha_offset > prior_layer_name > scalar).
    layers = component_data.get("layers", {})
    scalar_offset_value = float(
        np.log(prior_probability / (1.0 - prior_probability))
    )
    use_alpha_offset = alpha_offset is not None
    if (
        not use_alpha_offset
        and prior_layer_name is not None
        and prior_layer_name not in layers
    ):
        raise ValueError(
            f"prior layer {prior_layer_name!r} is not present on component; "
            f"available layers: {sorted(layers)}"
        )
    use_spatial_prior = (
        not use_alpha_offset
        and prior_layer_name is not None
        and prior_layer_name in layers
    )

    excluded_set = set(excluded_layer_names)
    if use_spatial_prior:
        excluded_set.add(prior_layer_name)

    resolved_grid = component_grid(component_data)
    if resolved_grid is None:
        raise ValueError(
            "component has no usable grid: pr_norm is absent and no layer "
            "model GeoDataFrame is available"
        )
    grid_gdf = resolved_grid.copy()

    # Compute grid-level prior offset.
    if use_alpha_offset:
        if len(alpha_offset) != len(grid_gdf):
            msg = (
                f"alpha_offset length {len(alpha_offset)} does not match "
                f"component grid length {len(grid_gdf)}"
            )
            raise ValueError(msg)
        offset_grid = np.asarray(alpha_offset, dtype=float)
    elif use_spatial_prior:
        prior_layer = layers[prior_layer_name]["model"]
        prior_vals = sample_layer_at_points(
            grid_gdf,
            prior_layer,
            "value_interpolated",
        )
        if np.all(~np.isfinite(prior_vals)):
            raise ValueError(
                f"prior layer '{prior_layer_name}' has no finite values; cannot build offset"
            )
        finite = np.isfinite(prior_vals)
        prior_vals[~finite] = np.nanmean(prior_vals[finite])
        v_min = float(np.nanmin(prior_vals))
        v_max = float(np.nanmax(prior_vals))
        if v_max <= v_min:
            prior_p_grid = np.full_like(
                prior_vals, (prior_p_min + prior_p_max) / 2.0
            )
        else:
            prior_p_grid = prior_p_min + (prior_vals - v_min) / (
                v_max - v_min
            ) * (prior_p_max - prior_p_min)
        prior_p_grid = np.clip(prior_p_grid, 1e-4, 1.0 - 1e-4)
        offset_grid = np.log(prior_p_grid / (1.0 - prior_p_grid))
        # Store the per-cell prior probability for downstream plotting.
        grid_gdf["prior_probability_spatial"] = prior_p_grid
    else:
        offset_grid = np.full(len(grid_gdf), scalar_offset_value)

    # Prior-predictive opt-in: short-circuit before any well overlay.
    if force_prior_predictive:
        probabilities = expit(offset_grid)
        grid_gdf["probability"] = probabilities
        return ComponentProbability(
            probability=grid_gdf,
            model=None,
            feature_names=(),
            spatial_field=None,
            diagnostics={"inference_role": "prior_predictive"},
        )

    # Build raw grid features only for data-informed fits. Scaling is learned
    # later from training wells; prediction-grid statistics must not influence
    # a fitted model or CV fold.
    X_grid_raw, feature_names = _flatten_component_features(
        component_data,
        included_layer_names=(
            set(included_layer_names)
            if included_layer_names is not None
            else None
        ),
        excluded_layer_names=excluded_set,
        sparse_binary_threshold=sparse_binary_threshold,
        coordinate_blacklist=coordinate_blacklist,
    )
    feature_names_out = list(feature_names)
    spatial_u: np.ndarray | None = None

    if labeled_wells is None or label_column is None:
        raise ValueError(
            "labeled_wells and label_column are required unless "
            "force_prior_predictive=True"
        )
    if label_column not in labeled_wells.columns:
        raise ValueError(
            f"label column {label_column!r} is absent from labeled_wells"
        )

    if labeled_wells is not None:
        # Sample raw grid features (and prior offset) at well points.
        grid_feat = grid_gdf[["geometry"]].copy()
        for i, name in enumerate(feature_names):
            grid_feat[name] = X_grid_raw[:, i]
        grid_feat["__prior_offset__"] = offset_grid

        validated_labels = _coerce_binary_labels(
            labeled_wells[label_column], label_column=label_column
        )
        labeled_wells = labeled_wells.copy()
        labeled_wells[label_column] = validated_labels
        wells_proj = labeled_wells.to_crs(grid_gdf.crs)
        wells_proj = wells_proj[wells_proj[label_column].notna()].copy()

        min_overlapping_wells = min_wells  # from config.labels.min_wells_for_fit (validate ensures >= 2)
        if len(wells_proj) < min_overlapping_wells:
            raise ValueError(
                f"component has fewer than {min_overlapping_wells} labelled wells"
            )

        grid_indices = snap_to_grid_indices(wells_proj, grid_feat)
        sampled = wells_proj[[label_column, "geometry"]].copy()
        for name in [*feature_names, "__prior_offset__"]:
            sampled[name] = grid_feat.iloc[grid_indices][name].to_numpy()

        if len(sampled) < min_overlapping_wells:
            raise ValueError(
                f"fewer than {min_overlapping_wells} labelled wells overlap component grid"
            )

        y_wells = sampled[label_column].to_numpy(dtype=int)
        X_wells_raw = sampled[list(feature_names)].to_numpy(dtype=float)
        offset_wells = sampled["__prior_offset__"].to_numpy(dtype=float)

        (
            X_wells,
            X_grid_scaled,
            evidence_center,
            evidence_scale,
            evidence_finite_counts,
        ) = _standardize_from_training(
            X_wells_raw,
            X_grid_raw,
            min_finite=min_overlapping_wells,
        )

        if len(np.unique(y_wells)) < 2:  # noqa: PLR2004
            raise ValueError(
                "labels at overlapping wells are single-class; a data-informed "
                "probability model is not identified"
            )

        # Translate per-feature weight + prior-mean dicts (keyed by the
        # feature names produced by _flatten_component_features) into the
        # arrays the optimiser expects.
        w_arr = (
            np.array(
                [
                    float(per_feature_weights.get(name, 1e-2))
                    for name in feature_names
                ],
                dtype=float,
            )
            if per_feature_weights is not None
            else None
        )
        mu_arr = (
            np.array(
                [float(prior_means.get(name, 0.0)) for name in feature_names],
                dtype=float,
            )
            if prior_means is not None
            else None
        )
        if pu_mode in {"off", "naive_pseudo_absence"}:
            result = _fit_offset_logit(
                X_wells,
                y_wells,
                offset_wells,
                per_feature_weights=w_arr,
                prior_means=mu_arr,
            )
        elif pu_mode == "nnpu":
            result = fit_nnpu_logistic(
                X_wells,
                y_wells,
                class_prior=pu_class_prior,
                offsets=offset_wells,
                penalty_weights=(
                    w_arr
                    if w_arr is not None
                    else np.full(X_wells.shape[1], 1e-2)
                ),
                prior_means=mu_arr,
            )
        else:
            raise ValueError(f"unsupported pu_mode {pu_mode!r}")

        # Evaluate the fitted prior + evidence contribution before adding the
        # spatial field. This is a decomposition of the fitted model, not a
        # separately refitted no-prior or no-spatial ablation.
        eta_grid = offset_grid + X_grid_scaled @ result.x
        grid_gdf["nonspatial_probability"] = expit(eta_grid)

        if include_spatial:
            eta_wells = offset_wells + X_wells @ result.x
            p_wells = expit(eta_wells)
            # This is the Bernoulli IRLS working residual on the logit scale,
            # not a Pearson residual. The fitted correction is added to eta.
            variance = np.clip(p_wells * (1.0 - p_wells), 1e-3, None)
            residuals = np.clip((y_wells - p_wells) / variance, -3.0, 3.0)

            well_geom = sampled.geometry
            grid_geom = grid_gdf.geometry
            train_coords = extract_coordinates(
                gpd.GeoDataFrame(geometry=well_geom, crs=sampled.crs)
            )
            pred_coords = extract_coordinates(
                gpd.GeoDataFrame(geometry=grid_geom, crs=grid_gdf.crs)
            )
            if spatial_backend == "latticekrigx":
                gp_result = fit_spatial_field_gp(
                    train_coords,
                    residuals,
                    pred_coords,
                )
                spatial_u = gp_result.u_mean
                grid_gdf["spatial_u_std"] = gp_result.u_std
                feature_names_out.append("spatial:gp_residual_at_wells")
            elif spatial_backend == "rbf":
                spatial_u = _fit_spatial_field_to_targets(
                    train_coords,
                    residuals,
                    pred_coords,
                )
                feature_names_out.append("spatial:rbf_residual_at_wells")
            else:
                raise ValueError(
                    "include_spatial=True requires spatial_backend 'latticekrigx' "
                    f"or 'rbf'; got {spatial_backend!r}"
                )
            eta_grid += spatial_u

        probabilities = expit(eta_grid)
        grid_gdf["probability"] = probabilities
        if spatial_u is not None:
            grid_gdf["spatial_u"] = spatial_u
        diagnostics = {
            "pu_mode": pu_mode,
            "n_train": len(y_wells),
            "evidence_center": evidence_center.tolist(),
            "evidence_scale": evidence_scale.tolist(),
            "evidence_finite_per_feature": evidence_finite_counts.tolist(),
            "evidence_missing_value_policy": (
                "training_feature_mean_zero_standardized_contribution"
            ),
        }
        if pu_mode == "nnpu":
            diagnostics.update(
                {
                    "pu_class_prior": float(result.class_prior),
                    "pu_positive_risk": float(result.positive_risk),
                    "pu_negative_risk": float(result.negative_risk),
                    "pu_raw_negative_risk": float(result.raw_negative_risk),
                }
            )
        return ComponentProbability(
            probability=grid_gdf,
            model=result,
            feature_names=tuple(feature_names_out),
            spatial_field=spatial_u,
            diagnostics=diagnostics,
        )

    raise AssertionError("unreachable")


def combine_probability_surfaces(
    component_probabilities: list[np.ndarray],
) -> np.ndarray:
    """Combine component probabilities with the product rule."""

    if not component_probabilities:
        raise ValueError(
            "At least one component probability surface is required."
        )
    return np.prod(np.stack(component_probabilities, axis=0), axis=0)
