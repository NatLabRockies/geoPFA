"""ProbabilisticConfig dataclass and JSON loaders.

Mirrors the ``probabilistic`` block of the PFA config JSON as a strongly-typed
dataclass tree with validation. The dataclass is the single source of truth
that the runner, fitters, alpha builder, calibration helpers, and output writers
all consume; the user-facing schema is documented in
``docs/probabilistic_method.md``.

Design choices
--------------

* Plain ``dataclasses`` with ``frozen=True`` everywhere: configs flow into long
  pipelines and accidental mutation has been a recurring source of bugs.
* Defaults are baked into the dataclasses so a minimal user JSON works
  end-to-end.
* Cross-block validation lives in ``ProbabilisticConfig.__post_init__`` so any
  consumer that builds a ``ProbabilisticConfig`` (tests, programmatic
  callers, JSON loader) gets identical validation.
* Unknown top-level keys are rejected; this catches typos in user-facing
  config files early.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from geopfa.exceptions import GEOPFAValueError

# ---------------------------------------------------------------------------
# Allowed value sets
# ---------------------------------------------------------------------------

ALLOWED_DIMENSIONS: tuple[str, ...] = ("2d", "3d")
ALLOWED_ALPHA_MODES: tuple[str, ...] = (
    "scalar",
    "layer_logit",
    "thermal_exceedance",
    "thermal_layer_exceedance",
    "multi_layer",
)
ALLOWED_PU_MODES: tuple[str, ...] = (
    "off",
    "naive_pseudo_absence",
    "nnpu",
)
ALLOWED_SITE_SELECTION_MODES: tuple[str, ...] = ("off", "joint_binary")
ALLOWED_SPATIAL_BACKENDS: tuple[str, ...] = ("latticekrigx", "rbf", "none")
ALLOWED_COORDINATE_SCALINGS: tuple[str, ...] = (
    "axis_range",
    "physical_isotropic",
)
ALLOWED_KERNELS: tuple[str, ...] = ("rbf", "matern32", "rbf_matern32")
ALLOWED_INFERENCE_BACKENDS: tuple[str, ...] = ("sequential", "gblk")
ALLOWED_EVIDENCE_STANDARDIZATIONS: tuple[str, ...] = (
    "observed_labels",
    "prediction_support",
)
ALLOWED_CALIBRATION_METHODS: tuple[str, ...] = (
    "platt",
    "isotonic",
    "temperature",
    "none",
)
ALLOWED_CALIBRATION_FITS: tuple[str, ...] = (
    "block_cv",
    "holdout",
    "in_sample",
)
ALLOWED_BLOCK_TYPES: tuple[str, ...] = ("grid", "kmeans")
ALLOWED_COMBINATION_RULES: tuple[str, ...] = (
    "product",
    "geometric_mean",
)
ALLOWED_OUTPUT_FORMATS: tuple[str, ...] = ("geotiff", "csv", "parquet", "vtk")

DEFAULT_COORD_BLACKLIST: tuple[str, ...] = (
    "X",
    "Y",
    "Z",
    "x",
    "y",
    "z",
    "inverted_y",
    "longitude",
    "latitude",
)


def _reject_unknown_keys(
    raw: Mapping[str, Any],
    allowed: set[str] | frozenset[str],
    *,
    context: str,
    hint: str | None = None,
) -> None:
    """Fail closed when a declared configuration key is not consumed."""

    unknown = set(raw) - set(allowed)
    if not unknown:
        return
    message = f"unknown {context} config key(s): {', '.join(sorted(unknown))}"
    if hint is not None:
        message += f"; {hint}"
    raise ValueError(message)


def _require_json_bool(
    raw: Mapping[str, Any],
    key: str,
    default: bool,
    *,
    context: str,
) -> bool:
    """Return a literal JSON boolean without Python truth-value coercion."""

    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise GEOPFAValueError(f"{context}.{key} must be a JSON boolean")
    return value


def _require_integer_value(value: Any, *, context: str) -> int:
    """Return an integer without truncating floats or accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise GEOPFAValueError(f"{context} must be an integer")
    return int(value)


def _require_json_integer(
    raw: Mapping[str, Any],
    key: str,
    default: int,
    *,
    context: str,
) -> int:
    return _require_integer_value(
        raw.get(key, default), context=f"{context}.{key}"
    )


def _require_finite_real_value(value: Any, *, context: str) -> float:
    """Return a finite real number without accepting numeric strings/bools."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise GEOPFAValueError(f"{context} must be a finite real number")
    return result


def _require_json_real(
    raw: Mapping[str, Any],
    key: str,
    default: float,
    *,
    context: str,
) -> float:
    return _require_finite_real_value(
        raw.get(key, default), context=f"{context}.{key}"
    )


def _require_optional_json_real(
    raw: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    return _require_finite_real_value(value, context=f"{context}.{key}")


def _require_json_string_array(
    raw: Mapping[str, Any],
    key: str,
    default: list[str] | tuple[str, ...] | None,
    *,
    context: str,
    allow_none: bool = False,
) -> tuple[str, ...] | None:
    """Return a JSON array of non-empty strings without scalar coercion."""
    value = raw.get(key, default)
    if value is None and allow_none:
        return None
    if not isinstance(value, list | tuple):
        raise GEOPFAValueError(f"{context}.{key} must be an array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise GEOPFAValueError(
            f"{context}.{key} must contain only non-empty strings"
        )
    if len(set(value)) != len(value):
        raise GEOPFAValueError(
            f"{context}.{key} must not contain duplicate entries"
        )
    return tuple(value)


def _require_json_real_array(
    raw: Mapping[str, Any],
    key: str,
    default: tuple[float, ...],
    *,
    context: str,
) -> tuple[float, ...]:
    """Return a JSON array of finite reals without scalar coercion."""
    value = raw.get(key, default)
    if not isinstance(value, list | tuple):
        raise GEOPFAValueError(f"{context}.{key} must be an array of numbers")
    return tuple(
        _require_finite_real_value(
            item,
            context=f"{context}.{key}[{index}]",
        )
        for index, item in enumerate(value)
    )


def _require_json_string(
    raw: Mapping[str, Any],
    key: str,
    default: str | None,
    *,
    context: str,
    allow_none: bool = False,
) -> str | None:
    """Return a literal JSON string without scalar coercion."""
    value = raw.get(key, default)
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        suffix = " or null" if allow_none else ""
        raise GEOPFAValueError(
            f"{context}.{key} must be a JSON string{suffix}"
        )
    return value


# ---------------------------------------------------------------------------
# Sub-dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridConfig:
    """Optional override for the PFA grid; ``None`` means inherit from PFA dict."""

    nx: int | None = None
    ny: int | None = None
    nz: int | None = None
    extent: tuple[float, ...] | None = None
    crs: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> GridConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw, {"nx", "ny", "nz", "extent", "crs"}, context="grid"
        )
        extent = raw.get("extent")
        return cls(
            nx=(
                _require_integer_value(raw["nx"], context="grid.nx")
                if raw.get("nx") is not None
                else None
            ),
            ny=(
                _require_integer_value(raw["ny"], context="grid.ny")
                if raw.get("ny") is not None
                else None
            ),
            nz=(
                _require_integer_value(raw["nz"], context="grid.nz")
                if raw.get("nz") is not None
                else None
            ),
            extent=(
                tuple(
                    _require_finite_real_value(
                        value, context=f"grid.extent[{index}]"
                    )
                    for index, value in enumerate(extent)
                )
                if extent is not None
                else None
            ),
            crs=raw.get("crs"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "extent": list(self.extent) if self.extent is not None else None,
            "crs": self.crs,
        }


@dataclass(frozen=True)
class LabelsConfig:
    """Where the labelled wells live and how their labels are stored."""

    source: str
    id_col: str
    label_columns: Mapping[str, str]
    layer: str | None = None
    label_quality_col: str | None = None
    label_source_col: str | None = None
    min_wells_for_fit: int = 4
    pu_mode: str = "off"
    pu_class_prior: float | Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Reject unidentified PU configurations for every construction path."""
        if self.pu_mode not in ALLOWED_PU_MODES:
            allowed = ", ".join(ALLOWED_PU_MODES)
            raise ValueError(
                f"labels.pu_mode must be one of: {allowed} "
                f"(got {self.pu_mode!r})"
            )
        prior = self.pu_class_prior
        if self.pu_mode == "nnpu" and prior is None:
            raise ValueError(
                "labels.pu_mode='nnpu' requires an externally identified "
                "labels.pu_class_prior"
            )
        if prior is not None:
            values = prior.values() if isinstance(prior, Mapping) else (prior,)
            numeric_values = [
                _require_finite_real_value(
                    value,
                    context="labels.pu_class_prior",
                )
                for value in values
            ]
            if any(not 0.0 < value < 1.0 for value in numeric_values):
                raise ValueError(
                    "labels.pu_class_prior values must be strictly between 0 and 1"
                )
            if isinstance(prior, Mapping):
                missing = set(self.label_columns) - set(prior)
                if missing:
                    raise ValueError(
                        "labels.pu_class_prior is missing component(s): "
                        + ", ".join(sorted(missing))
                    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LabelsConfig:
        """Build from a parsed-JSON mapping."""
        _reject_unknown_keys(
            raw,
            {
                "source",
                "id_col",
                "label_columns",
                "layer",
                "label_quality_col",
                "label_source_col",
                "min_wells_for_fit",
                "pu_mode",
                "pu_class_prior",
            },
            context="labels",
        )
        if "source" not in raw:
            raise ValueError("labels.source is required")
        if "id_col" not in raw:
            raise ValueError("labels.id_col is required")
        if "label_columns" not in raw:
            raise ValueError("labels.label_columns is required")
        pu_mode = raw.get("pu_mode", "off")
        if pu_mode not in ALLOWED_PU_MODES:
            allowed = ", ".join(ALLOWED_PU_MODES)
            raise ValueError(
                f"labels.pu_mode must be one of: {allowed} (got {pu_mode!r})",
            )
        return cls(
            source=str(raw["source"]),
            id_col=str(raw["id_col"]),
            label_columns=dict(raw["label_columns"]),
            layer=raw.get("layer"),
            label_quality_col=raw.get("label_quality_col"),
            label_source_col=raw.get("label_source_col"),
            min_wells_for_fit=_require_json_integer(
                raw, "min_wells_for_fit", 4, context="labels"
            ),
            pu_mode=pu_mode,
            pu_class_prior=(
                dict(raw["pu_class_prior"])
                if isinstance(raw.get("pu_class_prior"), Mapping)
                else raw.get("pu_class_prior")
            ),
        )

    def class_prior_for(self, component: str) -> float | None:
        """Return the externally supplied class prior for one component."""
        prior = self.pu_class_prior
        if prior is None:
            return None
        if isinstance(prior, Mapping):
            return float(prior[component])
        return float(prior)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "source": self.source,
            "id_col": self.id_col,
            "label_columns": dict(self.label_columns),
            "layer": self.layer,
            "label_quality_col": self.label_quality_col,
            "label_source_col": self.label_source_col,
            "min_wells_for_fit": self.min_wells_for_fit,
            "pu_mode": self.pu_mode,
            "pu_class_prior": (
                dict(self.pu_class_prior)
                if isinstance(self.pu_class_prior, Mapping)
                else self.pu_class_prior
            ),
        }


@dataclass(frozen=True)
class AlphaModeConfig:
    """Per-component prior configuration for alpha_c(s, z)."""

    mode: str = "scalar"
    scalar_fallback_pr0: float = 0.5
    layer: str | None = None  # for mode="layer_logit" or "multi_layer"
    layers: tuple[str, ...] = ()  # for mode="multi_layer"
    thermal_raster: str | None = None  # for mode="thermal_exceedance"
    threshold: float | None = None  # for mode="thermal_exceedance"
    uncertainty_raster: str | None = None  # for mode="thermal_exceedance"
    uncertainty_column: str | None = None  # thermal_layer_exceedance
    p_min: float = 0.2
    p_max: float = 0.8
    force_prior_predictive: bool = False
    use_evidence_prior: bool = False

    def __post_init__(self) -> None:  # noqa: PLR0912
        """Reject structurally invalid modes for programmatic callers."""
        if self.mode not in ALLOWED_ALPHA_MODES:
            allowed = ", ".join(ALLOWED_ALPHA_MODES)
            raise ValueError(
                f"alpha.mode must be one of: {allowed} (got {self.mode!r})"
            )
        if self.mode == "layer_logit" and not self.layer:
            raise ValueError("alpha.mode=layer_logit requires layer")
        if self.mode == "multi_layer" and not self.layers:
            raise ValueError(
                "alpha.mode=multi_layer requires non-empty layers"
            )
        if not isinstance(self.layers, tuple) or any(
            not isinstance(layer, str) or not layer.strip()
            for layer in self.layers
        ):
            raise ValueError(
                "alpha.layers must be a tuple of non-empty strings"
            )
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("alpha.layers must not contain duplicates")
        if self.mode == "thermal_exceedance":
            if not self.thermal_raster:
                raise ValueError(
                    "alpha.mode=thermal_exceedance requires thermal_raster"
                )
            if self.threshold is None:
                raise ValueError(
                    "alpha.mode=thermal_exceedance requires threshold"
                )
        if self.mode == "thermal_layer_exceedance":
            if not self.layer:
                raise ValueError(
                    "alpha.mode=thermal_layer_exceedance requires layer"
                )
            if self.threshold is None:
                raise ValueError(
                    "alpha.mode=thermal_layer_exceedance requires threshold"
                )
        if self.use_evidence_prior and not self.force_prior_predictive:
            raise ValueError(
                "alpha.use_evidence_prior requires force_prior_predictive=True"
            )
        for name, value in (
            ("scalar_fallback_pr0", self.scalar_fallback_pr0),
            ("p_min", self.p_min),
            ("p_max", self.p_max),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"alpha.{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"alpha.{name} must be a finite real number")
        if not 0.0 < float(self.scalar_fallback_pr0) < 1.0:
            raise ValueError(
                "alpha.scalar_fallback_pr0 must be strictly in (0, 1)"
            )
        if not (0.0 < float(self.p_min) < float(self.p_max) < 1.0):
            raise ValueError(
                "alpha.p_min and alpha.p_max must be strictly in (0, 1) "
                "with p_min < p_max"
            )

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], component_name: str
    ) -> AlphaModeConfig:
        """Build from a parsed-JSON mapping for the named component."""
        _reject_unknown_keys(
            raw,
            {
                "mode",
                "scalar_fallback_pr0",
                "layer",
                "layers",
                "thermal_raster",
                "threshold",
                "uncertainty_raster",
                "uncertainty_column",
                "p_min",
                "p_max",
                "force_prior_predictive",
                "use_evidence_prior",
            },
            context=f"alpha.{component_name}",
        )
        mode = raw.get("mode", "scalar")
        if mode not in ALLOWED_ALPHA_MODES:
            allowed = ", ".join(ALLOWED_ALPHA_MODES)
            raise ValueError(
                f"alpha.{component_name}.mode must be one of: {allowed} "
                f"(got {mode!r})",
            )
        if mode == "layer_logit" and not raw.get("layer"):
            raise ValueError(
                f"alpha.{component_name}.mode=layer_logit requires a "
                "'layer' key naming the prior layer",
            )
        if mode == "thermal_exceedance" and not raw.get("thermal_raster"):
            raise ValueError(
                f"alpha.{component_name}.mode=thermal_exceedance requires "
                "a 'thermal_raster' path",
            )
        if mode == "thermal_layer_exceedance":
            if not raw.get("layer"):
                raise ValueError(
                    f"alpha.{component_name}.mode=thermal_layer_exceedance "
                    "requires a 'layer' key"
                )
            if "threshold" not in raw:
                raise ValueError(
                    f"alpha.{component_name}.mode=thermal_layer_exceedance "
                    "requires a 'threshold' value"
                )
        layers = _require_json_string_array(
            raw,
            "layers",
            (),
            context=f"alpha.{component_name}",
        )
        if layers is None:  # pragma: no cover - helper contract
            raise RuntimeError("alpha.layers parser returned an invalid value")
        if mode == "multi_layer" and not layers:
            raise ValueError(
                f"alpha.{component_name}.mode=multi_layer requires a "
                "non-empty 'layers' list",
            )
        p_min = _require_json_real(
            raw, "p_min", 0.2, context=f"alpha.{component_name}"
        )
        p_max = _require_json_real(
            raw, "p_max", 0.8, context=f"alpha.{component_name}"
        )
        if not p_min < p_max:
            raise ValueError(
                f"alpha.{component_name}: p_min ({p_min}) must be less than "
                f"p_max ({p_max})",
            )
        if p_min <= 0.0 or p_max >= 1.0:
            raise ValueError(
                f"alpha.{component_name}: p_min and p_max must be strictly in "
                f"(0, 1) to avoid infinite logit values "
                f"(got p_min={p_min}, p_max={p_max}). "
                "Use values like 0.05 and 0.95 rather than 0.0 or 1.0."
            )
        scalar_pr0 = _require_json_real(
            raw,
            "scalar_fallback_pr0",
            0.5,
            context=f"alpha.{component_name}",
        )
        if not (0.0 < scalar_pr0 < 1.0):
            raise ValueError(
                f"alpha.{component_name}: scalar_fallback_pr0 must be strictly "
                f"in (0, 1) to avoid infinite logit values (got {scalar_pr0}). "
                "Use values like 0.1-0.9."
            )
        return cls(
            mode=mode,
            scalar_fallback_pr0=scalar_pr0,
            layer=raw.get("layer"),
            layers=layers,
            thermal_raster=raw.get("thermal_raster"),
            threshold=_require_optional_json_real(
                raw, "threshold", context=f"alpha.{component_name}"
            ),
            uncertainty_raster=raw.get("uncertainty_raster"),
            uncertainty_column=raw.get("uncertainty_column"),
            p_min=p_min,
            p_max=p_max,
            force_prior_predictive=_require_json_bool(
                raw,
                "force_prior_predictive",
                False,
                context=f"alpha.{component_name}",
            ),
            use_evidence_prior=_require_json_bool(
                raw,
                "use_evidence_prior",
                False,
                context=f"alpha.{component_name}",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "mode": self.mode,
            "scalar_fallback_pr0": self.scalar_fallback_pr0,
            "layer": self.layer,
            "layers": list(self.layers),
            "thermal_raster": self.thermal_raster,
            "threshold": self.threshold,
            "uncertainty_raster": self.uncertainty_raster,
            "uncertainty_column": self.uncertainty_column,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "force_prior_predictive": self.force_prior_predictive,
            "use_evidence_prior": self.use_evidence_prior,
        }


@dataclass(frozen=True)
class RegularizationConfig:
    """L2 regularization config for beta_ck fits."""

    C: float = 1.0
    per_feature_weights: Mapping[str, float] = field(default_factory=dict)
    prior_means: Mapping[str, float] = field(default_factory=dict)
    prior_precisions: Mapping[str, float] = field(default_factory=dict)
    play_type: str | None = None

    def __post_init__(self) -> None:
        """Validate explicit Gaussian coefficient-prior parameters."""
        for name, value in self.prior_means.items():
            _require_finite_real_value(
                value,
                context=f"evidence.regularization.prior_means.{name}",
            )
        for name, value in self.prior_precisions.items():
            numeric = _require_finite_real_value(
                value,
                context=f"evidence.regularization.prior_precisions.{name}",
            )
            if numeric <= 0.0:
                raise ValueError(
                    "evidence.regularization.prior_precisions values must be "
                    f"strictly positive (got {name}={numeric})"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> RegularizationConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "C",
                "per_feature_weights",
                "prior_means",
                "prior_precisions",
                "play_type",
            },
            context="evidence.regularization",
        )
        return cls(
            C=_require_json_real(
                raw, "C", 1.0, context="evidence.regularization"
            ),
            per_feature_weights={
                str(name): _require_finite_real_value(
                    value,
                    context=f"evidence.regularization.per_feature_weights.{name}",
                )
                for name, value in dict(
                    raw.get("per_feature_weights", {})
                ).items()
            },
            prior_means={
                str(name): _require_finite_real_value(
                    value,
                    context=f"evidence.regularization.prior_means.{name}",
                )
                for name, value in dict(raw.get("prior_means", {})).items()
            },
            prior_precisions={
                str(name): _require_finite_real_value(
                    value,
                    context=(
                        f"evidence.regularization.prior_precisions.{name}"
                    ),
                )
                for name, value in dict(
                    raw.get("prior_precisions", {})
                ).items()
            },
            play_type=raw.get("play_type"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "C": self.C,
            "per_feature_weights": dict(self.per_feature_weights),
            "prior_means": dict(self.prior_means),
            "prior_precisions": dict(self.prior_precisions),
            "play_type": self.play_type,
        }


@dataclass(frozen=True)
class EvidenceConfig:
    """beta_ck evidence-regression configuration."""

    regularization: RegularizationConfig = field(
        default_factory=RegularizationConfig
    )
    include_layers: tuple[str, ...] | None = None
    exclude_layers: tuple[str, ...] = ()
    sparse_binary_threshold: float = 0.90
    coordinate_blacklist: tuple[str, ...] = DEFAULT_COORD_BLACKLIST
    standardization: str = "observed_labels"

    def __post_init__(self) -> None:
        """Reject ambiguous layer selectors for programmatic callers."""
        if self.standardization not in ALLOWED_EVIDENCE_STANDARDIZATIONS:
            allowed = ", ".join(ALLOWED_EVIDENCE_STANDARDIZATIONS)
            raise ValueError(
                "evidence.standardization must be one of: "
                f"{allowed} (got {self.standardization!r})"
            )
        for name, values in (
            ("include_layers", self.include_layers),
            ("exclude_layers", self.exclude_layers),
            ("coordinate_blacklist", self.coordinate_blacklist),
        ):
            if values is None and name == "include_layers":
                continue
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError(
                    f"evidence.{name} must be a tuple of non-empty strings"
                )
            if len(set(values)) != len(values):
                raise ValueError(
                    f"evidence.{name} must not contain duplicates"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EvidenceConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "regularization",
                "include_layers",
                "exclude_layers",
                "sparse_binary_threshold",
                "coordinate_blacklist",
                "standardization",
            },
            context="evidence",
        )
        include = _require_json_string_array(
            raw,
            "include_layers",
            None,
            context="evidence",
            allow_none=True,
        )
        exclude = _require_json_string_array(
            raw,
            "exclude_layers",
            (),
            context="evidence",
        )
        coordinate_blacklist = _require_json_string_array(
            raw,
            "coordinate_blacklist",
            DEFAULT_COORD_BLACKLIST,
            context="evidence",
        )
        if exclude is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "evidence.exclude_layers parser returned an invalid value"
            )
        if coordinate_blacklist is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "evidence.coordinate_blacklist parser returned an invalid value"
            )
        return cls(
            regularization=RegularizationConfig.from_dict(
                raw.get("regularization")
            ),
            include_layers=include,
            exclude_layers=exclude,
            sparse_binary_threshold=_require_json_real(
                raw,
                "sparse_binary_threshold",
                0.90,
                context="evidence",
            ),
            coordinate_blacklist=coordinate_blacklist,
            standardization=_require_json_string(
                raw,
                "standardization",
                "observed_labels",
                context="evidence",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "regularization": self.regularization.to_dict(),
            "include_layers": (
                list(self.include_layers)
                if self.include_layers is not None
                else None
            ),
            "exclude_layers": list(self.exclude_layers),
            "sparse_binary_threshold": self.sparse_binary_threshold,
            "coordinate_blacklist": list(self.coordinate_blacklist),
            "standardization": self.standardization,
        }


@dataclass(frozen=True)
class SpatialFieldConfig:
    """u_c(s, z) spatial-field configuration."""

    enabled: bool = True
    backend: str = "latticekrigx"
    kernel: str = "rbf_matern32"
    n_inducing: int = 300
    lengthscale_lower_frac: float = 0.02
    lengthscale_upper_frac: float = 0.20
    optimize_restarts: int = 0
    n_levels: int = 2
    lattice_centers_per_dimension: int = 6
    coordinate_scaling: str = "axis_range"

    def __post_init__(self) -> None:
        """Reject unknown spatial model choices for every construction path."""
        if self.backend not in ALLOWED_SPATIAL_BACKENDS:
            allowed = ", ".join(ALLOWED_SPATIAL_BACKENDS)
            raise ValueError(
                "spatial_field.backend must be one of: "
                f"{allowed} (got {self.backend!r})"
            )
        if self.kernel not in ALLOWED_KERNELS:
            allowed = ", ".join(ALLOWED_KERNELS)
            raise ValueError(
                f"spatial_field.kernel must be one of: {allowed} "
                f"(got {self.kernel!r})"
            )
        if self.coordinate_scaling not in ALLOWED_COORDINATE_SCALINGS:
            allowed = ", ".join(ALLOWED_COORDINATE_SCALINGS)
            raise ValueError(
                "spatial_field.coordinate_scaling must be one of: "
                f"{allowed} (got {self.coordinate_scaling!r})"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SpatialFieldConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "enabled",
                "backend",
                "kernel",
                "n_inducing",
                "lengthscale_lower_frac",
                "lengthscale_upper_frac",
                "optimize_restarts",
                "n_levels",
                "lattice_centers_per_dimension",
                "coordinate_scaling",
            },
            context="spatial_field",
        )
        backend = raw.get("backend", "latticekrigx")
        if backend not in ALLOWED_SPATIAL_BACKENDS:
            allowed = ", ".join(ALLOWED_SPATIAL_BACKENDS)
            raise ValueError(
                f"spatial_field.backend must be one of: {allowed} (got {backend!r})",
            )
        kernel = raw.get("kernel", "rbf_matern32")
        if kernel not in ALLOWED_KERNELS:
            allowed = ", ".join(ALLOWED_KERNELS)
            raise ValueError(
                f"spatial_field.kernel must be one of: {allowed} (got {kernel!r})",
            )
        coordinate_scaling = raw.get("coordinate_scaling", "axis_range")
        if coordinate_scaling not in ALLOWED_COORDINATE_SCALINGS:
            allowed = ", ".join(ALLOWED_COORDINATE_SCALINGS)
            raise ValueError(
                "spatial_field.coordinate_scaling must be one of: "
                f"{allowed} (got {coordinate_scaling!r})"
            )
        return cls(
            enabled=_require_json_bool(
                raw, "enabled", True, context="spatial_field"
            ),
            backend=backend,
            kernel=kernel,
            n_inducing=_require_json_integer(
                raw, "n_inducing", 300, context="spatial_field"
            ),
            lengthscale_lower_frac=_require_json_real(
                raw,
                "lengthscale_lower_frac",
                0.02,
                context="spatial_field",
            ),
            lengthscale_upper_frac=_require_json_real(
                raw,
                "lengthscale_upper_frac",
                0.20,
                context="spatial_field",
            ),
            optimize_restarts=_require_json_integer(
                raw, "optimize_restarts", 0, context="spatial_field"
            ),
            n_levels=_require_json_integer(
                raw, "n_levels", 2, context="spatial_field"
            ),
            lattice_centers_per_dimension=_require_json_integer(
                raw,
                "lattice_centers_per_dimension",
                6,
                context="spatial_field",
            ),
            coordinate_scaling=coordinate_scaling,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "kernel": self.kernel,
            "n_inducing": self.n_inducing,
            "lengthscale_lower_frac": self.lengthscale_lower_frac,
            "lengthscale_upper_frac": self.lengthscale_upper_frac,
            "optimize_restarts": self.optimize_restarts,
            "n_levels": self.n_levels,
            "lattice_centers_per_dimension": (
                self.lattice_centers_per_dimension
            ),
            "coordinate_scaling": self.coordinate_scaling,
        }


@dataclass(frozen=True)
class SiteSelectionConfig:
    """Optional joint outcome/selection model over a finite candidate frame.

    The outcome-dependent selection log-odds are a sensitivity grid, not a
    fitted parameter: preferential selection is not identified from selected
    outcomes alone without external information.
    """

    mode: str = "off"
    candidate_source: str | None = None
    id_col: str | None = None
    selected_col: str = "selected"
    outcome_feature_columns: tuple[str, ...] = ()
    selection_feature_columns: tuple[str, ...] = ()
    outcome_selection_log_odds: tuple[float, ...] = (0.0,)
    outcome_penalty: float = 1e-4
    selection_penalty: float = 1e-4

    def __post_init__(self) -> None:
        """Validate the fail-closed candidate-frame likelihood contract."""
        if self.mode not in ALLOWED_SITE_SELECTION_MODES:
            allowed = ", ".join(ALLOWED_SITE_SELECTION_MODES)
            raise ValueError(
                f"site_selection.mode must be one of: {allowed} "
                f"(got {self.mode!r})"
            )
        if self.mode == "joint_binary":
            for name, value in (
                ("candidate_source", self.candidate_source),
                ("id_col", self.id_col),
                ("selected_col", self.selected_col),
            ):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"site_selection.{name} must be a non-empty string"
                    )
            missing = [
                name
                for name, value in (
                    ("candidate_source", self.candidate_source),
                    ("id_col", self.id_col),
                    ("outcome_feature_columns", self.outcome_feature_columns),
                    (
                        "selection_feature_columns",
                        self.selection_feature_columns,
                    ),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "site_selection.mode='joint_binary' requires: "
                    + ", ".join(missing)
                )
        if (
            not isinstance(self.outcome_selection_log_odds, tuple)
            or not self.outcome_selection_log_odds
        ):
            raise ValueError(
                "site_selection.outcome_selection_log_odds must be a non-empty tuple"
            )
        for index, value in enumerate(self.outcome_selection_log_odds):
            _require_finite_real_value(
                value,
                context=f"site_selection.outcome_selection_log_odds[{index}]",
            )
        for name, value in (
            ("outcome_penalty", self.outcome_penalty),
            ("selection_penalty", self.selection_penalty),
        ):
            numeric = _require_finite_real_value(
                value,
                context=f"site_selection.{name}",
            )
            if numeric < 0.0:
                raise ValueError(f"site_selection.{name} must be non-negative")
        for name, values in (
            ("outcome_feature_columns", self.outcome_feature_columns),
            ("selection_feature_columns", self.selection_feature_columns),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError(
                    f"site_selection.{name} must be a tuple of non-empty strings"
                )
            if len(set(values)) != len(values):
                raise ValueError(
                    f"site_selection.{name} must not contain duplicates"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SiteSelectionConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "mode",
                "candidate_source",
                "id_col",
                "selected_col",
                "outcome_feature_columns",
                "selection_feature_columns",
                "outcome_selection_log_odds",
                "outcome_penalty",
                "selection_penalty",
            },
            context="site_selection",
        )
        outcome_features = _require_json_string_array(
            raw,
            "outcome_feature_columns",
            (),
            context="site_selection",
        )
        selection_features = _require_json_string_array(
            raw,
            "selection_feature_columns",
            (),
            context="site_selection",
        )
        if outcome_features is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "site_selection outcome-feature parser returned an invalid value"
            )
        if selection_features is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "site_selection selection-feature parser returned an invalid value"
            )
        return cls(
            mode=_require_json_string(
                raw, "mode", "off", context="site_selection"
            ),
            candidate_source=_require_json_string(
                raw,
                "candidate_source",
                None,
                context="site_selection",
                allow_none=True,
            ),
            id_col=_require_json_string(
                raw,
                "id_col",
                None,
                context="site_selection",
                allow_none=True,
            ),
            selected_col=_require_json_string(
                raw, "selected_col", "selected", context="site_selection"
            ),
            outcome_feature_columns=outcome_features,
            selection_feature_columns=selection_features,
            outcome_selection_log_odds=_require_json_real_array(
                raw,
                "outcome_selection_log_odds",
                (0.0,),
                context="site_selection",
            ),
            outcome_penalty=_require_json_real(
                raw,
                "outcome_penalty",
                1e-4,
                context="site_selection",
            ),
            selection_penalty=_require_json_real(
                raw,
                "selection_penalty",
                1e-4,
                context="site_selection",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "mode": self.mode,
            "candidate_source": self.candidate_source,
            "id_col": self.id_col,
            "selected_col": self.selected_col,
            "outcome_feature_columns": list(self.outcome_feature_columns),
            "selection_feature_columns": list(self.selection_feature_columns),
            "outcome_selection_log_odds": list(
                self.outcome_selection_log_odds
            ),
            "outcome_penalty": self.outcome_penalty,
            "selection_penalty": self.selection_penalty,
        }


@dataclass(frozen=True)
class GBLKBayesianConfig:
    """Configuration for geoPFA's sole Bayesian inference path.

    When ``enabled=True`` and ``inference.backend="gblk"``, the runner uses
    LatticeKrigX's public Paige/INLA joint fitter and propagates its paired
    posterior draws to component and joint-event probability summaries.

    Training alpha offsets enter the likelihood and prediction alpha offsets
    enter every posterior draw. The Paige prior parameters and, for a
    bivariate fit, the published Kleiber profile parameters must be frozen as
    part of a science-run specification.
    """

    enabled: bool = False
    n_draws: int = 200
    seed: int = 0
    ci_level: float = 0.9
    cor_scale_median: float = 0.10
    spatial_sd_u: float = 1.0
    spatial_sd_tail_probability: float = 0.05
    dirichlet_concentration: float = 1.5
    separate_ranges: bool = False
    kleiber_r0: float | None = None
    kleiber_r1: float | None = None
    cluster_effect: bool = True
    validate_inla: bool = True

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> GBLKBayesianConfig:
        """Build from a parsed-JSON mapping (None → defaults)."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "enabled",
                "n_draws",
                "seed",
                "ci_level",
                "cor_scale_median",
                "spatial_sd_u",
                "spatial_sd_tail_probability",
                "dirichlet_concentration",
                "separate_ranges",
                "kleiber_r0",
                "kleiber_r1",
                "cluster_effect",
                "validate_inla",
            },
            context="inference.gblk_bayesian",
        )
        return cls(
            enabled=_require_json_bool(
                raw, "enabled", False, context="inference.gblk_bayesian"
            ),
            n_draws=_require_json_integer(
                raw, "n_draws", 200, context="inference.gblk_bayesian"
            ),
            seed=_require_json_integer(
                raw, "seed", 0, context="inference.gblk_bayesian"
            ),
            ci_level=_require_json_real(
                raw, "ci_level", 0.9, context="inference.gblk_bayesian"
            ),
            cor_scale_median=_require_json_real(
                raw,
                "cor_scale_median",
                0.10,
                context="inference.gblk_bayesian",
            ),
            spatial_sd_u=_require_json_real(
                raw,
                "spatial_sd_u",
                1.0,
                context="inference.gblk_bayesian",
            ),
            spatial_sd_tail_probability=_require_json_real(
                raw,
                "spatial_sd_tail_probability",
                0.05,
                context="inference.gblk_bayesian",
            ),
            dirichlet_concentration=_require_json_real(
                raw,
                "dirichlet_concentration",
                1.5,
                context="inference.gblk_bayesian",
            ),
            separate_ranges=_require_json_bool(
                raw,
                "separate_ranges",
                False,
                context="inference.gblk_bayesian",
            ),
            kleiber_r0=_require_optional_json_real(
                raw,
                "kleiber_r0",
                context="inference.gblk_bayesian",
            ),
            kleiber_r1=_require_optional_json_real(
                raw,
                "kleiber_r1",
                context="inference.gblk_bayesian",
            ),
            cluster_effect=_require_json_bool(
                raw,
                "cluster_effect",
                True,
                context="inference.gblk_bayesian",
            ),
            validate_inla=_require_json_bool(
                raw,
                "validate_inla",
                True,
                context="inference.gblk_bayesian",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "enabled": self.enabled,
            "n_draws": self.n_draws,
            "seed": self.seed,
            "ci_level": self.ci_level,
            "cor_scale_median": self.cor_scale_median,
            "spatial_sd_u": self.spatial_sd_u,
            "spatial_sd_tail_probability": self.spatial_sd_tail_probability,
            "dirichlet_concentration": self.dirichlet_concentration,
            "separate_ranges": self.separate_ranges,
            "kleiber_r0": self.kleiber_r0,
            "kleiber_r1": self.kleiber_r1,
            "cluster_effect": self.cluster_effect,
            "validate_inla": self.validate_inla,
        }


@dataclass(frozen=True)
class InferenceConfig:
    """Inference backend selection + nested backend-specific configs."""

    backend: str = "gblk"
    gblk_bayesian: GBLKBayesianConfig = field(
        default_factory=GBLKBayesianConfig
    )

    def __post_init__(self) -> None:
        """Reject unsupported inference backends for direct construction."""
        if self.backend not in ALLOWED_INFERENCE_BACKENDS:
            allowed = ", ".join(ALLOWED_INFERENCE_BACKENDS)
            raise ValueError(
                f"inference.backend must be one of: {allowed} "
                f"(got {self.backend!r})"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> InferenceConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw, {"backend", "gblk_bayesian"}, context="inference"
        )
        backend = raw.get("backend", "gblk")
        if backend not in ALLOWED_INFERENCE_BACKENDS:
            allowed = ", ".join(ALLOWED_INFERENCE_BACKENDS)
            raise ValueError(
                f"inference.backend must be one of: {allowed} (got {backend!r})",
            )
        return cls(
            backend=backend,
            gblk_bayesian=GBLKBayesianConfig.from_dict(
                raw.get("gblk_bayesian")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "backend": self.backend,
            "gblk_bayesian": self.gblk_bayesian.to_dict(),
        }


@dataclass(frozen=True)
class CalibrationConfig:
    """Post-hoc calibration configuration."""

    method: str = "none"
    fit_on: str = "block_cv"
    report_temperature: bool = True
    n_bins: int = 5

    def __post_init__(self) -> None:
        """Reject unknown calibration choices for direct construction."""
        if self.method not in ALLOWED_CALIBRATION_METHODS:
            allowed = ", ".join(ALLOWED_CALIBRATION_METHODS)
            raise ValueError(
                f"calibration.method must be one of: {allowed} "
                f"(got {self.method!r})"
            )
        if self.fit_on not in ALLOWED_CALIBRATION_FITS:
            allowed = ", ".join(ALLOWED_CALIBRATION_FITS)
            raise ValueError(
                f"calibration.fit_on must be one of: {allowed} "
                f"(got {self.fit_on!r})"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CalibrationConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        known_keys = {"method", "fit_on", "report_temperature", "n_bins"}
        _reject_unknown_keys(
            raw,
            known_keys,
            context="calibration",
            hint="use 'fit_on', not 'fit'",
        )
        method = raw.get("method", "none")
        if method not in ALLOWED_CALIBRATION_METHODS:
            allowed = ", ".join(ALLOWED_CALIBRATION_METHODS)
            raise ValueError(
                f"calibration.method must be one of: {allowed} (got {method!r})",
            )
        fit_on = raw.get("fit_on", "block_cv")
        if fit_on not in ALLOWED_CALIBRATION_FITS:
            allowed = ", ".join(ALLOWED_CALIBRATION_FITS)
            raise ValueError(
                f"calibration.fit_on must be one of: {allowed} (got {fit_on!r})",
            )
        return cls(
            method=method,
            fit_on=fit_on,
            report_temperature=_require_json_bool(
                raw, "report_temperature", True, context="calibration"
            ),
            n_bins=_require_json_integer(
                raw, "n_bins", 5, context="calibration"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "method": self.method,
            "fit_on": self.fit_on,
            "report_temperature": self.report_temperature,
            "n_bins": self.n_bins,
        }


@dataclass(frozen=True)
class CrossValidationConfig:
    """Spatial block CV configuration."""

    n_folds: int = 5
    block_type: str = "grid"
    block_size_km: float | None = None
    grid_size: int = 4
    buffer_km: float = 0.0

    def __post_init__(self) -> None:
        """Reject unknown block strategies for direct construction."""
        if self.block_type not in ALLOWED_BLOCK_TYPES:
            allowed = ", ".join(ALLOWED_BLOCK_TYPES)
            raise ValueError(
                f"cross_validation.block_type must be one of: {allowed} "
                f"(got {self.block_type!r})"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CrossValidationConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        known_keys = {
            "n_folds",
            "block_type",
            "block_size_km",
            "grid_size",
            "buffer_km",
        }
        _reject_unknown_keys(
            raw,
            known_keys,
            context="cross_validation",
            hint="use 'n_folds', not 'n_splits'",
        )
        block_type = raw.get("block_type", "grid")
        if block_type not in ALLOWED_BLOCK_TYPES:
            allowed = ", ".join(ALLOWED_BLOCK_TYPES)
            raise ValueError(
                f"cross_validation.block_type must be one of: {allowed} "
                f"(got {block_type!r})",
            )
        return cls(
            n_folds=_require_json_integer(
                raw, "n_folds", 5, context="cross_validation"
            ),
            block_type=block_type,
            block_size_km=_require_optional_json_real(
                raw, "block_size_km", context="cross_validation"
            ),
            grid_size=_require_json_integer(
                raw, "grid_size", 4, context="cross_validation"
            ),
            buffer_km=_require_json_real(
                raw, "buffer_km", 0.0, context="cross_validation"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "n_folds": self.n_folds,
            "block_type": self.block_type,
            "block_size_km": self.block_size_km,
            "grid_size": self.grid_size,
            "buffer_km": self.buffer_km,
        }


@dataclass(frozen=True)
class CombinationConfig:
    """Component-combination rule."""

    rule: str = "product"
    barrier_inverse: bool = True

    def __post_init__(self) -> None:
        """Reject retired or unknown component-combination rules."""
        if self.rule not in ALLOWED_COMBINATION_RULES:
            allowed = ", ".join(ALLOWED_COMBINATION_RULES)
            raise ValueError(
                f"combination.rule must be one of: {allowed} "
                f"(got {self.rule!r})"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CombinationConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw, {"rule", "barrier_inverse"}, context="combination"
        )
        rule = raw.get("rule", "product")
        if rule not in ALLOWED_COMBINATION_RULES:
            allowed = ", ".join(ALLOWED_COMBINATION_RULES)
            raise ValueError(
                f"combination.rule must be one of: {allowed} (got {rule!r})",
            )
        return cls(
            rule=rule,
            barrier_inverse=_require_json_bool(
                raw, "barrier_inverse", True, context="combination"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {"rule": self.rule, "barrier_inverse": self.barrier_inverse}


@dataclass(frozen=True)
class ScenarioConfig:
    """One row of the ablation / sensitivity scenario grid."""

    name: str
    include_priors: bool = True
    include_spatial: bool = True
    drop_layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous identifiers and malformed overrides."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario.name must be a non-empty string")
        if (
            Path(self.name).name != self.name
            or self.name in {".", ".."}
            or "\\" in self.name
        ):
            raise ValueError("scenario.name must be one safe path component")
        if not isinstance(self.include_priors, bool) or not isinstance(
            self.include_spatial, bool
        ):
            raise TypeError("scenario include flags must be booleans")
        if not isinstance(self.drop_layers, tuple) or any(
            not isinstance(layer, str) or not layer.strip()
            for layer in self.drop_layers
        ):
            raise ValueError(
                "scenario.drop_layers must be a tuple of non-empty strings"
            )
        if len(set(self.drop_layers)) != len(self.drop_layers):
            raise ValueError(
                "scenario.drop_layers must not contain duplicates"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ScenarioConfig:
        """Build from a parsed-JSON mapping."""
        _reject_unknown_keys(
            raw,
            {"name", "include_priors", "include_spatial", "drop_layers"},
            context="scenario",
        )
        if "name" not in raw:
            raise ValueError("scenarios[*].name is required")
        name = raw["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("scenario.name must be a non-empty string")
        drop_layers = _require_json_string_array(
            raw, "drop_layers", (), context="scenario"
        )
        if drop_layers is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "scenario.drop_layers parser returned an invalid value"
            )
        return cls(
            name=name,
            include_priors=_require_json_bool(
                raw, "include_priors", True, context="scenario"
            ),
            include_spatial=_require_json_bool(
                raw, "include_spatial", True, context="scenario"
            ),
            drop_layers=drop_layers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "name": self.name,
            "include_priors": self.include_priors,
            "include_spatial": self.include_spatial,
            "drop_layers": list(self.drop_layers),
        }


@dataclass(frozen=True)
class OutputsConfig:
    """Output writer toggles."""

    probability_rasters: bool = True
    uncertainty_rasters: bool = True
    calibration_artifacts: bool = True
    decision_artifacts: bool = True
    scenarios: bool = True
    posterior_draw_blocks: bool = False
    posterior_draw_block_size: int = 20
    format: tuple[str, ...] = ("geotiff", "csv")

    def __post_init__(self) -> None:
        """Reject unknown artifact formats for direct construction."""
        toggles = (
            self.probability_rasters,
            self.uncertainty_rasters,
            self.calibration_artifacts,
            self.decision_artifacts,
            self.scenarios,
            self.posterior_draw_blocks,
        )
        if any(not isinstance(value, bool) for value in toggles):
            raise ValueError("outputs output toggles must be booleans")
        block_size = _require_integer_value(
            self.posterior_draw_block_size,
            context="outputs.posterior_draw_block_size",
        )
        if block_size < 1:
            raise ValueError("outputs.posterior_draw_block_size must be >= 1")
        if not isinstance(self.format, tuple) or any(
            not isinstance(fmt, str) or not fmt.strip() for fmt in self.format
        ):
            raise ValueError(
                "outputs.format must be a tuple of non-empty strings"
            )
        if len(set(self.format)) != len(self.format):
            raise ValueError("outputs.format must not contain duplicates")
        allowed = ", ".join(ALLOWED_OUTPUT_FORMATS)
        for fmt in self.format:
            if fmt not in ALLOWED_OUTPUT_FORMATS:
                raise ValueError(
                    f"outputs.format entry {fmt!r} must be one of: {allowed}"
                )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> OutputsConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "probability_rasters",
                "uncertainty_rasters",
                "calibration_artifacts",
                "decision_artifacts",
                "scenarios",
                "posterior_draw_blocks",
                "posterior_draw_block_size",
                "format",
            },
            context="outputs",
        )
        formats = _require_json_string_array(
            raw,
            "format",
            ("geotiff", "csv"),
            context="outputs",
        )
        if formats is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "outputs.format parser returned an invalid value"
            )
        for fmt in formats:
            if fmt not in ALLOWED_OUTPUT_FORMATS:
                allowed = ", ".join(ALLOWED_OUTPUT_FORMATS)
                raise ValueError(
                    f"outputs.format entry {fmt!r} must be one of: {allowed}",
                )
        return cls(
            probability_rasters=_require_json_bool(
                raw, "probability_rasters", True, context="outputs"
            ),
            uncertainty_rasters=_require_json_bool(
                raw, "uncertainty_rasters", True, context="outputs"
            ),
            calibration_artifacts=_require_json_bool(
                raw, "calibration_artifacts", True, context="outputs"
            ),
            decision_artifacts=_require_json_bool(
                raw, "decision_artifacts", True, context="outputs"
            ),
            scenarios=_require_json_bool(
                raw, "scenarios", True, context="outputs"
            ),
            posterior_draw_blocks=_require_json_bool(
                raw, "posterior_draw_blocks", False, context="outputs"
            ),
            posterior_draw_block_size=_require_json_integer(
                raw,
                "posterior_draw_block_size",
                20,
                context="outputs",
            ),
            format=formats,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "probability_rasters": self.probability_rasters,
            "uncertainty_rasters": self.uncertainty_rasters,
            "calibration_artifacts": self.calibration_artifacts,
            "decision_artifacts": self.decision_artifacts,
            "scenarios": self.scenarios,
            "posterior_draw_blocks": self.posterior_draw_blocks,
            "posterior_draw_block_size": self.posterior_draw_block_size,
            "format": list(self.format),
        }


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "enabled",
        "output_dir",
        "dimensions",
        "grid",
        "labels",
        "alpha",
        "evidence",
        "spatial_field",
        "site_selection",
        "inference",
        "calibration",
        "cross_validation",
        "combination",
        "scenarios",
        "outputs",
    },
)


@dataclass(frozen=True)
class ProbabilisticConfig:
    """Full configuration for the probabilistic method.

    Mirrors the ``probabilistic`` block of the PFA config JSON. See
    ``docs/probabilistic_method.md`` for the user-facing schema.

    Parameters
    ----------
    enabled
        Whether the probabilistic method should run.
    output_dir
        Directory where outputs are written.
    dimensions
        ``"2d"`` or ``"3d"``.
    grid
        Optional grid override; ``None`` fields are inherited from the PFA dict.
    labels
        Labelled-well source configuration.
    alpha
        Per-component ``alpha_c`` prior configuration.
    evidence
        ``beta_ck`` evidence-regression configuration.
    spatial_field
        ``u_c(s)`` spatial-field configuration.
    inference
        Inference backend selection.
    calibration
        Post-hoc calibration.
    cross_validation
        Spatial block CV.
    combination
        Component-combination rule.
    scenarios
        Optional ablation grid; empty by default.
    outputs
        Output writer toggles.
    """

    enabled: bool
    output_dir: Path
    dimensions: str
    grid: GridConfig
    labels: LabelsConfig
    alpha: Mapping[str, AlphaModeConfig]
    evidence: EvidenceConfig
    spatial_field: SpatialFieldConfig
    inference: InferenceConfig
    calibration: CalibrationConfig
    cross_validation: CrossValidationConfig
    combination: CombinationConfig
    scenarios: tuple[ScenarioConfig, ...]
    outputs: OutputsConfig
    site_selection: SiteSelectionConfig = field(
        default_factory=SiteSelectionConfig
    )

    def __post_init__(self) -> None:
        """Cross-block validation that depends on multiple sub-configs."""
        if self.dimensions not in ALLOWED_DIMENSIONS:
            allowed = ", ".join(ALLOWED_DIMENSIONS)
            raise ValueError(
                f"dimensions must be one of: {allowed} (got {self.dimensions!r})",
            )
        if (
            self.labels.pu_mode == "nnpu"
            and self.inference.backend != "sequential"
        ):
            raise ValueError(
                "labels.pu_mode='nnpu' currently requires "
                "inference.backend='sequential'"
            )
        if (
            self.inference.gblk_bayesian.enabled
            and self.inference.backend != "gblk"
        ):
            raise ValueError(
                "inference.gblk_bayesian.enabled=True requires "
                "inference.backend='gblk'"
            )
        if self.spatial_field.enabled and self.spatial_field.backend == "none":
            raise ValueError(
                "spatial_field.enabled=True requires a spatial backend"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ProbabilisticConfig:
        """Build a ``ProbabilisticConfig`` from the parsed JSON dict."""
        unknown = set(raw.keys()) - _TOP_LEVEL_KEYS
        if unknown:
            extras = ", ".join(sorted(unknown))
            raise ValueError(
                f"unknown top-level keys in probabilistic config: {extras}",
            )
        if "labels" not in raw:
            raise ValueError(
                "probabilistic config is missing required 'labels' block"
            )
        if "alpha" not in raw:
            raise ValueError(
                "probabilistic config is missing required 'alpha' block"
            )
        scenarios = raw.get("scenarios", ())
        if not isinstance(scenarios, list | tuple):
            raise GEOPFAValueError("probabilistic.scenarios must be an array")
        if any(not isinstance(scenario, Mapping) for scenario in scenarios):
            raise GEOPFAValueError(
                "probabilistic.scenarios must contain only objects"
            )
        cfg = cls(
            enabled=_require_json_bool(
                raw, "enabled", True, context="probabilistic"
            ),
            output_dir=Path(raw.get("output_dir", "outputs/probabilistic/")),
            dimensions=str(raw.get("dimensions", "2d")),
            grid=GridConfig.from_dict(raw.get("grid")),
            labels=LabelsConfig.from_dict(raw["labels"]),
            alpha={
                name: AlphaModeConfig.from_dict(sub, name)
                for name, sub in raw["alpha"].items()
            },
            evidence=EvidenceConfig.from_dict(raw.get("evidence")),
            spatial_field=SpatialFieldConfig.from_dict(
                raw.get("spatial_field")
            ),
            inference=InferenceConfig.from_dict(raw.get("inference")),
            calibration=CalibrationConfig.from_dict(raw.get("calibration")),
            cross_validation=CrossValidationConfig.from_dict(
                raw.get("cross_validation"),
            ),
            combination=CombinationConfig.from_dict(raw.get("combination")),
            scenarios=tuple(ScenarioConfig.from_dict(s) for s in scenarios),
            outputs=OutputsConfig.from_dict(raw.get("outputs")),
            site_selection=SiteSelectionConfig.from_dict(
                raw.get("site_selection")
            ),
        )
        cfg.validate_raise()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Round-trip serialization back to a plain dict."""
        return {
            "enabled": self.enabled,
            "output_dir": str(self.output_dir),
            "dimensions": self.dimensions,
            "grid": self.grid.to_dict(),
            "labels": self.labels.to_dict(),
            "alpha": {name: sub.to_dict() for name, sub in self.alpha.items()},
            "evidence": self.evidence.to_dict(),
            "spatial_field": self.spatial_field.to_dict(),
            "inference": self.inference.to_dict(),
            "calibration": self.calibration.to_dict(),
            "cross_validation": self.cross_validation.to_dict(),
            "combination": self.combination.to_dict(),
            "scenarios": [s.to_dict() for s in self.scenarios],
            "outputs": self.outputs.to_dict(),
            "site_selection": self.site_selection.to_dict(),
        }

    @classmethod
    def from_pfa(cls, pfa: Mapping[str, Any]) -> ProbabilisticConfig:
        """Build from an existing pfa dict that contains a ``"probabilistic"`` key.

        This is the primary integration point with the geoPFA config system.
        The probabilistic settings live under ``pfa["probabilistic"]`` alongside
        the existing ``pfa["criteria"]`` / ``pfa["components"]`` structure.

        Example pfa config JSON::

            {
              "criteria": { ... },
              "probabilistic": {
                "enabled": true,
                "labels": { ... },
                "alpha": { ... },
                ...
              }
            }
        """
        if "probabilistic" not in pfa:
            raise KeyError(
                "pfa dict has no 'probabilistic' key. "
                "Add a 'probabilistic' block to the pfa config JSON."
            )
        return cls.from_dict(pfa["probabilistic"])

    def validate(self) -> list[str]:  # noqa: PLR0912, PLR0915
        """Validate cross-field constraints and numeric hyperparameter ranges.

        Returns a list of error strings; empty list means the config is valid.
        Call :meth:`validate_raise` to raise :class:`ValueError` on first error.
        """
        errors: list[str] = []

        # output_dir must be non-empty (Path("") resolves to cwd)
        if str(self.output_dir).strip() in {"", "."}:
            errors.append(
                "output_dir is empty or '.'; provide an explicit output path to avoid "
                "writing probabilistic outputs into the current working directory"
            )
        if (
            self.outputs.posterior_draw_blocks
            and not self.inference.gblk_bayesian.enabled
        ):
            errors.append(
                "outputs.posterior_draw_blocks requires Bayesian GBLK inference "
                "(inference.backend='gblk' and "
                "inference.gblk_bayesian.enabled=true)"
            )
        if (
            self.outputs.posterior_draw_blocks
            and self.inference.gblk_bayesian.cluster_effect
        ):
            errors.append(
                "outputs.posterior_draw_blocks requires "
                "inference.gblk_bayesian.cluster_effect=false so grid prediction "
                "is projected in bounded draw blocks rather than materialized by INLA"
            )
        if (
            self.inference.gblk_bayesian.enabled
            and self.alpha
            and all(
                alpha.force_prior_predictive and not alpha.use_evidence_prior
                for alpha in self.alpha.values()
            )
        ):
            errors.append(
                "inference.gblk_bayesian.enabled=True has no stochastic "
                "components: every component is a fixed prior prediction"
            )

        for axis_name, size in (
            ("nx", self.grid.nx),
            ("ny", self.grid.ny),
            ("nz", self.grid.nz),
        ):
            if size is not None and size < 1:
                errors.append(f"grid.{axis_name} must be >= 1 (got {size})")
        if self.grid.extent is not None:
            expected_extent_length = 4 if self.dimensions == "2d" else 6
            if len(self.grid.extent) != expected_extent_length:
                errors.append(
                    "grid.extent must contain "
                    f"{expected_extent_length} values for {self.dimensions} "
                    f"(got {len(self.grid.extent)})"
                )
            else:
                half = expected_extent_length // 2
                if any(
                    lower >= upper
                    for lower, upper in zip(
                        self.grid.extent[:half],
                        self.grid.extent[half:],
                        strict=True,
                    )
                ):
                    errors.append(
                        "grid.extent lower bounds must be strictly less than "
                        "the corresponding upper bounds"
                    )

        # labels must have at least one component configured
        if not self.labels.label_columns:
            errors.append(
                "labels.label_columns is empty; no components are configured"
            )
        data_informed_components = {
            name
            for name, alpha in self.alpha.items()
            if not alpha.force_prior_predictive
        }
        minimum_allowed = 1 if self.inference.gblk_bayesian.enabled else 2
        if (
            data_informed_components
            and self.labels.min_wells_for_fit < minimum_allowed
        ):
            errors.append(
                "labels.min_wells_for_fit must be >= "
                f"{minimum_allowed} for the selected inference mode "
                f"(got {self.labels.min_wells_for_fit})"
            )

        # alpha dict must be non-empty
        if not self.alpha:
            errors.append(
                "alpha is empty; configure at least one component's prior offset"
            )

        label_components = set(self.labels.label_columns)
        alpha_components = set(self.alpha)
        missing_alpha = label_components - alpha_components
        if missing_alpha:
            errors.append(
                "every labels.label_columns component requires an alpha "
                "configuration; missing: " + ", ".join(sorted(missing_alpha))
            )
        unlabeled_alpha = alpha_components - label_components
        implicit_prior_only = {
            name
            for name in unlabeled_alpha
            if not self.alpha[name].force_prior_predictive
        }
        if implicit_prior_only:
            errors.append(
                "alpha components without label mappings must set "
                "force_prior_predictive=True; offending components: "
                + ", ".join(sorted(implicit_prior_only))
            )
        if self.inference.backend == "sequential" and unlabeled_alpha:
            errors.append(
                "inference.backend='sequential' does not support alpha-only "
                "components; add label mappings or use the GBLK prior-predictive "
                "path: " + ", ".join(sorted(unlabeled_alpha))
            )

        # Per-component alpha validation
        for comp_name, alpha_cfg in self.alpha.items():
            pr0 = alpha_cfg.scalar_fallback_pr0
            if not (0.0 < pr0 < 1.0):
                errors.append(
                    f"alpha.{comp_name}.scalar_fallback_pr0={pr0} must be strictly in (0, 1)"
                )
            if not (0.0 < alpha_cfg.p_min < 0.5):  # noqa: PLR2004
                errors.append(
                    f"alpha.{comp_name}.p_min must be in (0, 0.5) "
                    f"(got {alpha_cfg.p_min})"
                )
            if not (0.5 < alpha_cfg.p_max < 1.0):  # noqa: PLR2004
                errors.append(
                    f"alpha.{comp_name}.p_max must be in (0.5, 1) "
                    f"(got {alpha_cfg.p_max})"
                )
            if not (alpha_cfg.p_min < alpha_cfg.p_max):
                errors.append(
                    f"alpha.{comp_name}.p_min ({alpha_cfg.p_min}) must be "
                    f"< p_max ({alpha_cfg.p_max})"
                )

        # Numeric range guards
        if self.evidence.regularization.C <= 0:
            errors.append(
                f"evidence.regularization.C must be > 0 "
                f"(got {self.evidence.regularization.C})"
            )
        for (
            name,
            weight,
        ) in self.evidence.regularization.per_feature_weights.items():
            if weight < 0.0:
                errors.append(
                    "evidence.regularization.per_feature_weights values must "
                    f"be non-negative (got {name}={weight})"
                )
        for (
            name,
            precision,
        ) in self.evidence.regularization.prior_precisions.items():
            if precision <= 0.0:
                errors.append(
                    "evidence.regularization.prior_precisions values must be "
                    f"strictly positive (got {name}={precision})"
                )
        evidence_prior_components = {
            name
            for name, alpha_cfg in self.alpha.items()
            if alpha_cfg.use_evidence_prior
        }
        if (
            evidence_prior_components
            and not self.inference.gblk_bayesian.enabled
        ):
            errors.append(
                "alpha.use_evidence_prior requires "
                "inference.gblk_bayesian.enabled=True; offending components: "
                + ", ".join(sorted(evidence_prior_components))
            )
        if not 0.0 < self.evidence.sparse_binary_threshold <= 1.0:
            errors.append(
                "evidence.sparse_binary_threshold must be in (0, 1] "
                f"(got {self.evidence.sparse_binary_threshold})"
            )
        if self.cross_validation.n_folds < 2:  # noqa: PLR2004
            errors.append(
                f"cross_validation.n_folds must be >= 2 (got {self.cross_validation.n_folds})"
            )
        if self.cross_validation.grid_size < 1:
            errors.append(
                f"cross_validation.grid_size must be >= 1 (got {self.cross_validation.grid_size})"
            )
        if self.cross_validation.buffer_km < 0.0:
            errors.append(
                "cross_validation.buffer_km must be non-negative "
                f"(got {self.cross_validation.buffer_km})"
            )
        if (
            self.cross_validation.block_size_km is not None
            and self.cross_validation.block_size_km <= 0.0
        ):
            errors.append(
                "cross_validation.block_size_km must be > 0 when provided "
                f"(got {self.cross_validation.block_size_km})"
            )
        if (
            self.cross_validation.block_type != "grid"
            and self.cross_validation.block_size_km is not None
        ):
            errors.append(
                "cross_validation.block_size_km is only defined for "
                "block_type='grid'"
            )
        if self.calibration.n_bins < 1:
            errors.append(
                f"calibration.n_bins must be >= 1 (got {self.calibration.n_bins})"
            )
        if self.inference.gblk_bayesian.n_draws < 1:
            errors.append(
                "inference.gblk_bayesian.n_draws must be >= 1 "
                f"(got {self.inference.gblk_bayesian.n_draws})"
            )
        if not 0.0 < self.inference.gblk_bayesian.ci_level < 1.0:
            errors.append(
                "inference.gblk_bayesian.ci_level must be in (0, 1) "
                f"(got {self.inference.gblk_bayesian.ci_level})"
            )
        bayes = self.inference.gblk_bayesian
        for name, value in (
            ("cor_scale_median", bayes.cor_scale_median),
            ("spatial_sd_u", bayes.spatial_sd_u),
            ("dirichlet_concentration", bayes.dirichlet_concentration),
        ):
            if not math.isfinite(value) or value <= 0.0:
                errors.append(
                    f"inference.gblk_bayesian.{name} must be finite and > 0"
                )
        if not 0.0 < bayes.spatial_sd_tail_probability < 1.0:
            errors.append(
                "inference.gblk_bayesian.spatial_sd_tail_probability "
                "must be in (0, 1)"
            )
        if (bayes.kleiber_r0 is None) != (bayes.kleiber_r1 is None):
            errors.append(
                "inference.gblk_bayesian.kleiber_r0 and kleiber_r1 must be "
                "supplied together"
            )
        if bayes.kleiber_r0 is not None and bayes.kleiber_r1 is not None:
            if not math.isfinite(bayes.kleiber_r0) or not (
                -1.0 < bayes.kleiber_r0 < 1.0
            ):
                errors.append(
                    "inference.gblk_bayesian.kleiber_r0 must be finite and in "
                    "(-1, 1)"
                )
            if not math.isfinite(bayes.kleiber_r1) or bayes.kleiber_r1 < 0.0:
                errors.append(
                    "inference.gblk_bayesian.kleiber_r1 must be finite and >= 0"
                )
        if self.spatial_field.n_inducing < 1:
            errors.append(
                f"spatial_field.n_inducing must be >= 1 "
                f"(got {self.spatial_field.n_inducing})"
            )
        if self.spatial_field.optimize_restarts < 0:
            errors.append(
                "spatial_field.optimize_restarts must be >= 0 "
                f"(got {self.spatial_field.optimize_restarts})"
            )
        if self.spatial_field.n_levels < 1:
            errors.append(
                f"spatial_field.n_levels must be >= 1 "
                f"(got {self.spatial_field.n_levels})"
            )
        if self.spatial_field.lattice_centers_per_dimension < 1:
            errors.append(
                "spatial_field.lattice_centers_per_dimension must be >= 1 "
                f"(got {self.spatial_field.lattice_centers_per_dimension})"
            )
        if self.spatial_field.lengthscale_lower_frac <= 0:
            errors.append(
                f"spatial_field.lengthscale_lower_frac must be > 0 "
                f"(got {self.spatial_field.lengthscale_lower_frac})"
            )
        if (
            self.spatial_field.lengthscale_upper_frac
            <= self.spatial_field.lengthscale_lower_frac
        ):
            errors.append(
                f"spatial_field.lengthscale_upper_frac "
                f"({self.spatial_field.lengthscale_upper_frac}) must be > "
                f"lengthscale_lower_frac "
                f"({self.spatial_field.lengthscale_lower_frac})"
            )

        # Cross-field constraints
        if (
            self.labels.pu_mode == "nnpu"
            and self.inference.backend != "sequential"
        ):
            errors.append(
                "labels.pu_mode='nnpu' currently requires "
                "inference.backend='sequential'"
            )
        if self.spatial_field.enabled and self.spatial_field.backend == "none":
            errors.append(
                "spatial_field.enabled=True requires a spatial backend"
            )
        scenario_names = [scenario.name for scenario in self.scenarios]
        if len(set(scenario_names)) != len(scenario_names):
            errors.append("scenarios[*].name values must be unique")
        if self.inference.gblk_bayesian.enabled and any(
            not scenario.include_spatial for scenario in self.scenarios
        ):
            errors.append(
                "Bayesian GBLK scenarios must set include_spatial=True; "
                "the sole Bayesian fitter has no nonspatial latent-field mode"
            )
        return errors

    def validate_raise(self) -> None:
        """Raise :class:`~geopfa.exceptions.GEOPFAValueError` if :meth:`validate` returns any errors."""
        errors = self.validate()
        if errors:
            raise GEOPFAValueError(
                "ProbabilisticConfig validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )


def load_probabilistic_config(path: str | Path) -> ProbabilisticConfig:
    """Load a ``ProbabilisticConfig`` from a PFA config JSON file.

    The file is expected to contain a top-level ``probabilistic`` block; the
    rest of the PFA config (criteria, components, etc.) is ignored by this
    function — it is consumed by the runner separately.

    Calls :meth:`~ProbabilisticConfig.validate_raise` after parsing so
    range errors and cross-field inconsistencies surface immediately.
    """

    def _reject_nonfinite_json(token: str) -> None:
        raise ValueError(
            f"non-finite JSON constant {token!r} is not permitted"
        )

    config_path = Path(path).resolve()
    raw = json.loads(
        config_path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )
    if "probabilistic" not in raw:
        raise ValueError(
            f"PFA config at {path} has no 'probabilistic' block; nothing to load",
        )
    cfg = ProbabilisticConfig.from_dict(raw["probabilistic"])
    base_dir = config_path.parent

    def _resolve(value: str | Path | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return str(candidate.resolve())

    cfg = replace(
        cfg,
        output_dir=Path(_resolve(cfg.output_dir)),
        labels=replace(cfg.labels, source=_resolve(cfg.labels.source)),
        alpha={
            name: replace(
                alpha,
                thermal_raster=_resolve(alpha.thermal_raster),
                uncertainty_raster=_resolve(alpha.uncertainty_raster),
            )
            for name, alpha in cfg.alpha.items()
        },
        site_selection=replace(
            cfg.site_selection,
            candidate_source=_resolve(cfg.site_selection.candidate_source),
        ),
    )
    cfg.validate_raise()
    return cfg


# Public surface re-exported through geopfa.prob.__init__
__all__ = [
    "ALLOWED_ALPHA_MODES",
    "ALLOWED_BLOCK_TYPES",
    "ALLOWED_CALIBRATION_FITS",
    "ALLOWED_CALIBRATION_METHODS",
    "ALLOWED_COMBINATION_RULES",
    "ALLOWED_DIMENSIONS",
    "ALLOWED_EVIDENCE_STANDARDIZATIONS",
    "ALLOWED_INFERENCE_BACKENDS",
    "ALLOWED_KERNELS",
    "ALLOWED_OUTPUT_FORMATS",
    "ALLOWED_PU_MODES",
    "ALLOWED_SITE_SELECTION_MODES",
    "ALLOWED_SPATIAL_BACKENDS",
    "DEFAULT_COORD_BLACKLIST",
    "AlphaModeConfig",
    "CalibrationConfig",
    "CombinationConfig",
    "CrossValidationConfig",
    "EvidenceConfig",
    "GBLKBayesianConfig",
    "GridConfig",
    "InferenceConfig",
    "LabelsConfig",
    "OutputsConfig",
    "ProbabilisticConfig",
    "RegularizationConfig",
    "ScenarioConfig",
    "SiteSelectionConfig",
    "SpatialFieldConfig",
    "load_probabilistic_config",
]
