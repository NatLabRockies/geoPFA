"""Builder for the per-cell prior offset alpha_c(s, z).

Implements the four alpha modes described in the engineering manifest §5.B:

* ``scalar`` — uniform ``logit(scalar_fallback_pr0)`` across the grid.
* ``layer_logit`` — min-max rescale a named layer's value to
  ``[p_min, p_max]`` and apply ``logit``. The layer is auto-added to the
  ``excluded_layer_names`` set so it doesn't double-count as evidence.
* ``thermal_exceedance`` — load a thermal raster (and optionally an
  uncertainty raster), compute ``P(T > T*)`` per cell (step function or
  integration over ``Normal(T, sigma_T)``), clip, and apply ``logit``.
* ``thermal_layer_exceedance`` — compute the same exceedance directly from a
  processed point/voxel layer, preserving true 3-D depth variation.
* ``multi_layer`` — sum ``logit``-rescaled values from N named layers in
  logit space; all named layers are auto-excluded.

The return type carries the per-cell offset, the configured scalar reference,
the set of layer names that must be excluded from the regression, and a
provenance dict suitable for serialising to JSON. Declared non-scalar modes
fail closed when their required inputs are unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from scipy.stats import norm

from .config import AlphaModeConfig
from .spatial_alignment import grid_values_on_reference


@dataclass(frozen=True)
class AlphaCResult:
    """Per-component prior-offset build result.

    Attributes
    ----------
    grid_offset
        Per-cell ``alpha_c(s, z)`` array of length ``len(grid_gdf)`` (in the
        same row order as the grid GeoDataFrame).
    scalar_fallback
        The configured scalar reference value
        (``logit(scalar_fallback_pr0)``). Always populated for provenance;
        it is not substituted for a failed non-scalar mode.
    excluded_layer_names
        Layers that must be excluded from the evidence regression for this
        component (to avoid double-counting layers used to build alpha).
    provenance
        JSON-serialisable dict describing how this alpha was built.
    """

    grid_offset: np.ndarray
    scalar_fallback: float
    excluded_layer_names: set[str] = field(default_factory=set)
    provenance: dict[str, Any] = field(default_factory=dict)


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    return np.log(p / (1.0 - p))


def _scalar_offset(pr0: float) -> np.ndarray:
    return np.asarray(_logit(pr0))


def _rescale_to_logit(
    values: np.ndarray, *, p_min: float, p_max: float
) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError(
            "alpha layer must provide a finite value at every grid cell; "
            f"found {int(np.size(values) - np.count_nonzero(finite))} missing "
            "or non-finite values"
        )
    vals = np.asarray(values, dtype=float)
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    if v_max <= v_min:
        raise ValueError(
            "alpha layer is constant; min-max probability scaling is not "
            "identified"
        )
    p_grid = p_min + (vals - v_min) / (v_max - v_min) * (p_max - p_min)
    p_grid = np.clip(p_grid, p_min, p_max)
    return np.log(p_grid / (1.0 - p_grid))


def _layer_logit_offset(
    component_data: dict,
    grid_gdf: gpd.GeoDataFrame,
    *,
    layer_name: str,
    p_min: float,
    p_max: float,
) -> np.ndarray:
    layer = component_data["layers"][layer_name]
    layer_gdf = layer.get("model")
    if layer_gdf is None:
        raise ValueError(
            f"Layer {layer_name!r} has no 'model' key — the layer has not been "
            "pre-processed (interpolated/extrapolated) yet. Run the processing "
            "pipeline to create the 'model' GeoDataFrame before calling build_alpha_c."
        )
    evidence_col = layer.get("model_data_col", "value_interpolated")
    values = grid_values_on_reference(
        grid_gdf,
        layer_gdf,
        evidence_col,
        context=f"alpha layer {layer_name!r}",
    )
    return _rescale_to_logit(values, p_min=p_min, p_max=p_max)


def _thermal_exceedance(  # noqa: PLR0913
    raster_path: Path,
    uncertainty_path: Path | None,
    grid_gdf: gpd.GeoDataFrame,
    *,
    threshold: float,
    p_min: float,
    p_max: float,
) -> np.ndarray:
    if not raster_path.exists():
        raise FileNotFoundError(f"thermal raster not found: {raster_path}")

    target_crs = grid_gdf.crs
    coords_gdf = grid_gdf[["geometry"]].copy()
    if target_crs is None:
        raise ValueError(
            "grid_gdf must have a CRS for thermal raster sampling"
        )

    def sample_complete_raster(path: Path, *, context: str) -> np.ndarray:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise ValueError(f"{context} must declare a CRS")
            coords = coords_gdf.to_crs(src.crs)
            sample_xy = list(
                zip(
                    coords.geometry.x.to_numpy(),
                    coords.geometry.y.to_numpy(),
                    strict=False,
                )
            )
            sampled = np.ma.asarray(
                list(src.sample(sample_xy, indexes=1, masked=True))
            ).reshape(-1)
            values = np.asarray(sampled.filled(np.nan), dtype=float)
        finite = np.isfinite(values)
        if not np.all(finite):
            raise ValueError(
                f"{context} must provide a finite value at every grid cell; "
                f"found {int(values.size - np.count_nonzero(finite))} cells "
                "outside coverage or marked as nodata"
            )
        return values

    values = sample_complete_raster(raster_path, context="thermal raster")

    if uncertainty_path is not None:
        if not uncertainty_path.exists():
            raise FileNotFoundError(
                f"uncertainty raster not found: {uncertainty_path}",
            )
        sigma = sample_complete_raster(
            uncertainty_path, context="thermal uncertainty raster"
        )
        if np.any(sigma <= 0.0):
            raise ValueError(
                "thermal uncertainty raster must be strictly positive at "
                "every grid cell; omit it to request deterministic thresholding"
            )
        # P(T > T*) = sf((T* - T) / sigma).  Using the survival
        # function directly preserves precision in the rare-event tail.
        z = (threshold - values) / sigma
        probability = norm.sf(z)
    else:
        probability = (values > threshold).astype(float)
    p_grid = np.clip(probability, p_min, p_max)
    return np.log(p_grid / (1.0 - p_grid))


def _thermal_layer_exceedance(  # noqa: PLR0913
    component_data: dict,
    grid_gdf: gpd.GeoDataFrame,
    *,
    layer_name: str,
    uncertainty_column: str | None,
    threshold: float,
    p_min: float,
    p_max: float,
) -> np.ndarray:
    """Build a thermal exceedance offset from an aligned 2-D/3-D layer."""
    layer = component_data["layers"][layer_name]
    model = layer.get("model")
    if model is None:
        raise ValueError(
            f"Layer {layer_name!r} has no 'model' key — process it before "
            "building a thermal layer exceedance"
        )
    mean_column = layer.get("model_data_col", "value_interpolated")
    mean = grid_values_on_reference(
        grid_gdf,
        model,
        mean_column,
        context=f"thermal mean layer {layer_name!r}",
    )
    if not np.all(np.isfinite(mean)):
        raise ValueError(
            "thermal mean layer must provide a finite value at every grid cell"
        )
    if uncertainty_column is None:
        probability = (mean > threshold).astype(float)
    else:
        if uncertainty_column not in model.columns:
            raise ValueError(
                f"thermal layer {layer_name!r} is missing uncertainty column "
                f"{uncertainty_column!r}"
            )
        sigma = grid_values_on_reference(
            grid_gdf,
            model,
            uncertainty_column,
            context=f"thermal uncertainty layer {layer_name!r}",
        )
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError(
                "thermal layer uncertainty must be finite and strictly "
                "positive at every grid cell"
            )
        probability = norm.sf((threshold - mean) / sigma)
    clipped = np.clip(probability, p_min, p_max)
    return np.log(clipped / (1.0 - clipped))


def build_alpha_c(
    component_data: dict,
    cfg: AlphaModeConfig,
    *,
    grid_gdf: gpd.GeoDataFrame,
) -> AlphaCResult:
    """Build the per-cell prior offset for one component.

    Parameters
    ----------
    component_data
        The per-component dict from the PFA tree (``pr_norm``, ``layers``,
        ``pr0``).
    cfg
        The component's
        :class:`~geopfa.prob.config.AlphaModeConfig`.
    grid_gdf
        The prediction grid (typically ``component_data['pr_norm']``).

    Returns
    -------
    AlphaCResult
    """
    scalar_fallback = float(_logit(cfg.scalar_fallback_pr0))

    if cfg.mode == "scalar":
        offset = np.full(len(grid_gdf), scalar_fallback)
        provenance: dict[str, Any] = {
            "mode": "scalar",
            "scalar_fallback_pr0": cfg.scalar_fallback_pr0,
        }
        return AlphaCResult(
            grid_offset=offset,
            scalar_fallback=scalar_fallback,
            provenance=provenance,
        )

    if cfg.mode == "layer_logit":
        if cfg.layer not in component_data.get("layers", {}):
            available = sorted(component_data.get("layers", {}))
            raise ValueError(
                f"alpha layer {cfg.layer!r} is not present on component; "
                f"available layers: {available}"
            )
        offset = _layer_logit_offset(
            component_data,
            grid_gdf,
            layer_name=cfg.layer,
            p_min=cfg.p_min,
            p_max=cfg.p_max,
        )
        provenance = {
            "mode": "layer_logit",
            "layer": cfg.layer,
            "p_min": cfg.p_min,
            "p_max": cfg.p_max,
        }
        return AlphaCResult(
            grid_offset=offset,
            scalar_fallback=scalar_fallback,
            excluded_layer_names={cfg.layer},
            provenance=provenance,
        )

    if cfg.mode == "multi_layer":
        available_layers = set(component_data.get("layers", {}).keys())
        missing = [n for n in cfg.layers if n not in available_layers]
        if missing:
            raise ValueError(
                "alpha multi_layer is missing declared layer(s) "
                f"{missing}; available layers: {sorted(available_layers)}"
            )
        active_layers = list(cfg.layers)
        offsets = [
            _layer_logit_offset(
                component_data,
                grid_gdf,
                layer_name=name,
                p_min=cfg.p_min,
                p_max=cfg.p_max,
            )
            for name in active_layers
        ]
        offset = np.sum(offsets, axis=0)
        provenance = {
            "mode": "multi_layer",
            "layers": active_layers,
            "p_min": cfg.p_min,
            "p_max": cfg.p_max,
        }
        return AlphaCResult(
            grid_offset=offset,
            scalar_fallback=scalar_fallback,
            excluded_layer_names=set(cfg.layers),
            provenance=provenance,
        )

    if cfg.mode == "thermal_exceedance":
        offset = _thermal_exceedance(
            Path(cfg.thermal_raster),
            Path(cfg.uncertainty_raster) if cfg.uncertainty_raster else None,
            grid_gdf,
            threshold=cfg.threshold,
            p_min=cfg.p_min,
            p_max=cfg.p_max,
        )
        provenance = {
            "mode": "thermal_exceedance",
            "thermal_raster": str(cfg.thermal_raster),
            "uncertainty_raster": (
                str(cfg.uncertainty_raster) if cfg.uncertainty_raster else None
            ),
            "threshold": cfg.threshold,
            "p_min": cfg.p_min,
            "p_max": cfg.p_max,
        }
        return AlphaCResult(
            grid_offset=offset,
            scalar_fallback=scalar_fallback,
            provenance=provenance,
        )

    if cfg.mode == "thermal_layer_exceedance":
        if cfg.layer not in component_data.get("layers", {}):
            available = sorted(component_data.get("layers", {}))
            raise ValueError(
                f"thermal alpha layer {cfg.layer!r} is not present on "
                f"component; available layers: {available}"
            )
        offset = _thermal_layer_exceedance(
            component_data,
            grid_gdf,
            layer_name=cfg.layer,
            uncertainty_column=cfg.uncertainty_column,
            threshold=cfg.threshold,
            p_min=cfg.p_min,
            p_max=cfg.p_max,
        )
        provenance = {
            "mode": "thermal_layer_exceedance",
            "layer": cfg.layer,
            "uncertainty_column": cfg.uncertainty_column,
            "threshold": cfg.threshold,
            "p_min": cfg.p_min,
            "p_max": cfg.p_max,
        }
        return AlphaCResult(
            grid_offset=offset,
            scalar_fallback=scalar_fallback,
            excluded_layer_names={cfg.layer},
            provenance=provenance,
        )

    msg = f"unknown alpha mode {cfg.mode!r}"
    raise ValueError(msg)


def write_alpha_provenance(
    results: dict[str, AlphaCResult], path: Path
) -> None:
    """Serialise per-component alpha provenance to JSON."""
    payload = {
        name: {
            **result.provenance,
            "excluded_layer_names": sorted(result.excluded_layer_names),
            "scalar_fallback_logit": result.scalar_fallback,
        }
        for name, result in results.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


__all__ = ["AlphaCResult", "build_alpha_c", "write_alpha_provenance"]
