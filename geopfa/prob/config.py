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
from collections.abc import Iterable, Mapping
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
ALLOWED_OBSERVATION_FAMILIES: tuple[str, ...] = ("bernoulli", "gaussian")
ALLOWED_SITE_SELECTION_MODES: tuple[str, ...] = ("off", "joint_binary")
ALLOWED_SPATIAL_BACKENDS: tuple[str, ...] = ("latticekrigx", "rbf", "none")
ALLOWED_COORDINATE_SCALINGS: tuple[str, ...] = (
    "axis_range",
    "physical_isotropic",
)
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
ALLOWED_CALIBRATION_FITS: tuple[str, ...] = ("block_cv",)
ALLOWED_BLOCK_TYPES: tuple[str, ...] = ("grid", "kmeans")
ALLOWED_COMBINATION_RULES: tuple[str, ...] = ("product",)
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

_LAYER_ALPHA_PROBABILITY_BOUNDS = (0.2, 0.8)
_WINDOWS_RESERVED_FILE_STEMS = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def validate_surface_names(
    names: Iterable[object],
    *,
    context: str = "surface",
    reserved: Iterable[str] = (),
) -> None:
    """Require portable, collision-free identifiers used in output filenames."""
    reserved_names = {name.casefold() for name in reserved}
    normalized: dict[str, str] = {}
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or not name.isascii()
            or any(
                not (character.isalnum() or character in {"_", "-"})
                for character in name
            )
        ):
            raise GEOPFAValueError(
                f"{context} {name!r} must be a portable surface name using "
                "only ASCII letters, digits, underscores, and hyphens"
            )
        normalized_name = name.casefold()
        if normalized_name in _WINDOWS_RESERVED_FILE_STEMS:
            raise GEOPFAValueError(
                f"{context} {name!r} is not a portable surface name because "
                "it is reserved on Windows"
            )
        if normalized_name in reserved_names:
            raise GEOPFAValueError(
                f"{context} {name!r} uses reserved output name 'combined'"
            )
        previous = normalized.get(normalized_name)
        if previous is not None and previous != name:
            raise GEOPFAValueError(
                f"{context} names {previous!r} and {name!r} have a "
                "case-insensitive collision"
            )
        normalized[normalized_name] = name


_THERMAL_ALPHA_PROBABILITY_BOUNDS = (1e-12, 1.0 - 1e-12)
_ALPHA_COMMON_CONFIG_KEYS = frozenset(
    {
        "mode",
        "scalar_fallback_pr0",
        "force_prior_predictive",
        "use_evidence_prior",
    }
)
_ALPHA_MODE_CONFIG_KEYS: Mapping[str, frozenset[str]] = {
    "scalar": frozenset(),
    "layer_logit": frozenset({"layer", "p_min", "p_max"}),
    "multi_layer": frozenset({"layers", "p_min", "p_max"}),
    "thermal_exceedance": frozenset(
        {
            "thermal_raster",
            "threshold",
            "uncertainty_raster",
            "p_min",
            "p_max",
        }
    ),
    "thermal_layer_exceedance": frozenset(
        {"layer", "threshold", "uncertainty_column", "p_min", "p_max"}
    ),
}


def _default_alpha_probability_bounds(mode: str) -> tuple[float, float]:
    """Return mode-specific finite bounds for probability-to-logit conversion."""
    if mode in {"thermal_exceedance", "thermal_layer_exceedance"}:
        return _THERMAL_ALPHA_PROBABILITY_BOUNDS
    return _LAYER_ALPHA_PROBABILITY_BOUNDS


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
    """Prediction-grid contract.

    Current probabilistic runners require every field to be ``None`` and use
    the first configured PFA component grid as the canonical support. Explicit
    overrides fail validation so a requested grid is never silently ignored.
    """

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
class ObservationModelConfig:
    """Likelihood contract for one observed PFA component.

    Gaussian responses are divided by ``response_scale`` before fitting. This
    keeps the latent-field prior on a meaningful, dimensionless scale while
    preserving predictions in the response's original units.
    """

    family: str = "bernoulli"
    response_scale: float | None = None

    def __post_init__(self) -> None:
        """Reject unsupported families and invalid response scales."""
        if self.family not in ALLOWED_OBSERVATION_FAMILIES:
            allowed = ", ".join(ALLOWED_OBSERVATION_FAMILIES)
            raise ValueError(
                f"labels.observation_models family must be one of: {allowed}"
            )
        if self.family == "gaussian" and self.response_scale is not None:
            scale = _require_finite_real_value(
                self.response_scale,
                context="labels.observation_models.response_scale",
            )
            if scale <= 0.0:
                raise ValueError(
                    "labels.observation_models.response_scale must be positive"
                )
        elif self.response_scale is not None:
            raise ValueError(
                "response_scale is defined only for Gaussian observation models"
            )

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], component_name: str
    ) -> ObservationModelConfig:
        """Build a component likelihood from a parsed-JSON mapping."""
        context = f"labels.observation_models.{component_name}"
        _reject_unknown_keys(
            raw, {"family", "response_scale"}, context=context
        )
        return cls(
            family=str(raw.get("family", "bernoulli")),
            response_scale=_require_optional_json_real(
                raw, "response_scale", context=context
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        payload: dict[str, Any] = {"family": self.family}
        if self.response_scale is not None:
            payload["response_scale"] = self.response_scale
        return payload


@dataclass(frozen=True)
class LabelsConfig:
    """Where the labelled wells live and how their labels are stored."""

    source: str | None = None
    id_col: str | None = None
    label_columns: Mapping[str, str] = field(default_factory=dict)
    source_crs: str | None = None
    x_col: str | None = None
    y_col: str | None = None
    z_col: str | None = None
    depth_col: str | None = None
    observation_models: Mapping[str, ObservationModelConfig] = field(
        default_factory=dict
    )
    layer: str | None = None
    min_wells_for_fit: int = 4
    pu_mode: str = "off"
    pu_class_prior: float | Mapping[str, float] | None = None

    def __post_init__(self) -> None:  # noqa: PLR0912
        """Reject invalid coordinate, response, and PU declarations."""
        for name, value in (
            ("source", self.source),
            ("id_col", self.id_col),
            ("source_crs", self.source_crs),
            ("x_col", self.x_col),
            ("y_col", self.y_col),
            ("z_col", self.z_col),
            ("depth_col", self.depth_col),
        ):
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(f"labels.{name} must be a non-empty string")
        if not isinstance(self.label_columns, Mapping) or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(column, str)
            or not column.strip()
            for name, column in self.label_columns.items()
        ):
            raise ValueError(
                "labels.label_columns must map non-empty component names to "
                "non-empty column names"
            )
        if (self.x_col is None) != (self.y_col is None):
            raise ValueError(
                "labels.x_col and labels.y_col must be declared together"
            )
        if self.z_col is not None and self.x_col is None:
            raise ValueError(
                "labels.z_col requires labels.x_col and labels.y_col"
            )
        if (
            self.source is not None
            and Path(self.source).suffix.lower() in {".csv", ".txt"}
            and self.layer is not None
        ):
            raise ValueError(
                "labels.layer is not used for CSV label sources; omit it"
            )
        invalid_models = {
            name
            for name, model in self.observation_models.items()
            if not isinstance(model, ObservationModelConfig)
        }
        if invalid_models:
            raise TypeError(
                "labels.observation_models values must be "
                "ObservationModelConfig instances"
            )
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
    def from_dict(cls, raw: Mapping[str, Any] | None) -> LabelsConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {
                "source",
                "id_col",
                "label_columns",
                "source_crs",
                "x_col",
                "y_col",
                "z_col",
                "depth_col",
                "observation_models",
                "layer",
                "min_wells_for_fit",
                "pu_mode",
                "pu_class_prior",
            },
            context="labels",
        )
        label_columns = raw.get("label_columns", {})
        if not isinstance(label_columns, Mapping):
            raise TypeError("labels.label_columns must be a JSON object")
        pu_mode = raw.get("pu_mode", "off")
        if pu_mode not in ALLOWED_PU_MODES:
            allowed = ", ".join(ALLOWED_PU_MODES)
            raise ValueError(
                f"labels.pu_mode must be one of: {allowed} (got {pu_mode!r})",
            )
        return cls(
            source=_require_json_string(
                raw, "source", None, context="labels", allow_none=True
            ),
            id_col=_require_json_string(
                raw, "id_col", None, context="labels", allow_none=True
            ),
            label_columns=dict(label_columns),
            source_crs=_require_json_string(
                raw,
                "source_crs",
                None,
                context="labels",
                allow_none=True,
            ),
            x_col=_require_json_string(
                raw, "x_col", None, context="labels", allow_none=True
            ),
            y_col=_require_json_string(
                raw, "y_col", None, context="labels", allow_none=True
            ),
            z_col=_require_json_string(
                raw, "z_col", None, context="labels", allow_none=True
            ),
            depth_col=_require_json_string(
                raw, "depth_col", None, context="labels", allow_none=True
            ),
            observation_models={
                str(name): ObservationModelConfig.from_dict(model, str(name))
                for name, model in dict(
                    raw.get("observation_models", {})
                ).items()
            },
            layer=raw.get("layer"),
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

    def observation_model_for(self, component: str) -> ObservationModelConfig:
        """Return the declared model, defaulting to a Bernoulli response."""
        if component in self.observation_models:
            return self.observation_models[component]
        if component not in self.label_columns:
            raise KeyError(
                f"component {component!r} has neither a label column nor an "
                "explicit observation model"
            )
        return ObservationModelConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "source": self.source,
            "id_col": self.id_col,
            "label_columns": dict(self.label_columns),
            "source_crs": self.source_crs,
            "x_col": self.x_col,
            "y_col": self.y_col,
            "z_col": self.z_col,
            "depth_col": self.depth_col,
            "observation_models": {
                name: model.to_dict()
                for name, model in self.observation_models.items()
            },
            "layer": self.layer,
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
    p_min: float | None = None
    p_max: float | None = None
    force_prior_predictive: bool = False
    use_evidence_prior: bool = False

    def __post_init__(self) -> None:  # noqa: PLR0912
        """Reject structurally invalid modes for programmatic callers."""
        if self.mode not in ALLOWED_ALPHA_MODES:
            allowed = ", ".join(ALLOWED_ALPHA_MODES)
            raise ValueError(
                f"alpha.mode must be one of: {allowed} (got {self.mode!r})"
            )
        supplied_mode_fields = {
            name
            for name, supplied in (
                ("layer", self.layer is not None),
                ("layers", bool(self.layers)),
                ("thermal_raster", self.thermal_raster is not None),
                ("threshold", self.threshold is not None),
                ("uncertainty_raster", self.uncertainty_raster is not None),
                ("uncertainty_column", self.uncertainty_column is not None),
                ("p_min", self.p_min is not None),
                ("p_max", self.p_max is not None),
            )
            if supplied
        }
        unused = supplied_mode_fields - _ALPHA_MODE_CONFIG_KEYS[self.mode]
        if unused:
            raise ValueError(
                f"alpha.mode={self.mode} does not use field(s): "
                + ", ".join(sorted(unused))
            )
        if self.mode != "scalar":
            default_p_min, default_p_max = _default_alpha_probability_bounds(
                self.mode
            )
            if self.p_min is None:
                object.__setattr__(self, "p_min", default_p_min)
            if self.p_max is None:
                object.__setattr__(self, "p_max", default_p_max)
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
        numeric_fields = [("scalar_fallback_pr0", self.scalar_fallback_pr0)]
        if self.mode != "scalar":
            numeric_fields.extend(
                (("p_min", self.p_min), ("p_max", self.p_max))
            )
        for name, value in numeric_fields:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"alpha.{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"alpha.{name} must be a finite real number")
        if not 0.0 < float(self.scalar_fallback_pr0) < 1.0:
            raise ValueError(
                "alpha.scalar_fallback_pr0 must be strictly in (0, 1)"
            )
        if self.mode != "scalar" and not (
            0.0 < float(self.p_min) < float(self.p_max) < 1.0
        ):
            raise ValueError(
                "alpha.p_min and alpha.p_max must be strictly in (0, 1) "
                "with p_min < p_max"
            )

    @classmethod
    def from_dict(  # noqa: PLR0912
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
        unused = (
            set(raw)
            - _ALPHA_COMMON_CONFIG_KEYS
            - _ALPHA_MODE_CONFIG_KEYS[mode]
        )
        if unused:
            raise ValueError(
                f"alpha.{component_name}.mode={mode} does not use field(s): "
                + ", ".join(sorted(unused))
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
        default_p_min, default_p_max = _default_alpha_probability_bounds(mode)
        p_min = (
            None
            if mode == "scalar"
            else _require_json_real(
                raw,
                "p_min",
                default_p_min,
                context=f"alpha.{component_name}",
            )
        )
        p_max = (
            None
            if mode == "scalar"
            else _require_json_real(
                raw,
                "p_max",
                default_p_max,
                context=f"alpha.{component_name}",
            )
        )
        if mode != "scalar":
            if not p_min < p_max:
                raise ValueError(
                    f"alpha.{component_name}: p_min ({p_min}) must be less "
                    f"than p_max ({p_max})",
                )
            if p_min <= 0.0 or p_max >= 1.0:
                raise ValueError(
                    f"alpha.{component_name}: p_min and p_max must be "
                    "strictly in (0, 1) to avoid infinite logit values "
                    f"(got p_min={p_min}, p_max={p_max}). Use values like "
                    "0.05 and 0.95 rather than 0.0 or 1.0."
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
        payload: dict[str, Any] = {
            "mode": self.mode,
            "scalar_fallback_pr0": self.scalar_fallback_pr0,
            "force_prior_predictive": self.force_prior_predictive,
            "use_evidence_prior": self.use_evidence_prior,
        }
        if self.mode in {"layer_logit", "thermal_layer_exceedance"}:
            payload["layer"] = self.layer
        if self.mode == "multi_layer":
            payload["layers"] = list(self.layers)
        if self.mode == "thermal_exceedance":
            payload["thermal_raster"] = self.thermal_raster
            payload["uncertainty_raster"] = self.uncertainty_raster
        if self.mode in {"thermal_exceedance", "thermal_layer_exceedance"}:
            payload["threshold"] = self.threshold
        if self.mode == "thermal_layer_exceedance":
            payload["uncertainty_column"] = self.uncertainty_column
        if self.mode != "scalar":
            payload["p_min"] = self.p_min
            payload["p_max"] = self.p_max
        return payload


@dataclass(frozen=True)
class RegularizationConfig:
    """Explicit L2 and Gaussian-prior parameters for beta_ck fits."""

    C: float = 1.0
    per_feature_weights: Mapping[str, float] = field(default_factory=dict)
    prior_means: Mapping[str, float] = field(default_factory=dict)
    prior_precisions: Mapping[str, float] = field(default_factory=dict)

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
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "C": self.C,
            "per_feature_weights": dict(self.per_feature_weights),
            "prior_means": dict(self.prior_means),
            "prior_precisions": dict(self.prior_precisions),
        }


@dataclass(frozen=True)
class EvidenceFeatureExpansionConfig:
    """Optional fixed-effect expansion for one observed component.

    The expansion is applied to raw evidence and model coordinates before the
    existing fold-local standardization. Degree two adds squared terms.
    Pairwise evidence and evidence-coordinate interactions are separate,
    explicit choices so sparse studies do not silently acquire a large design.
    """

    degree: int = 1
    coordinate_degree: int = 1
    include_pairwise_interactions: bool = False
    coordinate_axes: tuple[str, ...] = ()
    include_evidence_coordinate_interactions: bool = False

    def __post_init__(self) -> None:
        """Reject unsupported degrees, axes, and inactive interactions."""
        if isinstance(self.degree, bool) or self.degree not in {1, 2}:
            raise ValueError(
                "evidence.feature_expansions degree must be 1 or 2"
            )
        if isinstance(
            self.coordinate_degree, bool
        ) or self.coordinate_degree not in {1, 2}:
            raise ValueError(
                "evidence.feature_expansions coordinate_degree must be 1 or 2"
            )
        allowed_axes = {"x", "y", "z"}
        if not isinstance(self.coordinate_axes, tuple) or any(
            not isinstance(axis, str) or axis not in allowed_axes
            for axis in self.coordinate_axes
        ):
            raise ValueError(
                "evidence.feature_expansions coordinate_axes must be a tuple "
                "containing only 'x', 'y', or 'z'"
            )
        if len(set(self.coordinate_axes)) != len(self.coordinate_axes):
            raise ValueError(
                "evidence.feature_expansions coordinate_axes must not contain "
                "duplicates"
            )
        if (
            self.include_evidence_coordinate_interactions
            and not self.coordinate_axes
        ):
            raise ValueError(
                "evidence.feature_expansions coordinate_axes are required "
                "when include_evidence_coordinate_interactions is true"
            )

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        component_name: str,
    ) -> EvidenceFeatureExpansionConfig:
        """Build one component expansion from a parsed JSON mapping."""
        context = f"evidence.feature_expansions.{component_name}"
        _reject_unknown_keys(
            raw,
            {
                "degree",
                "coordinate_degree",
                "include_pairwise_interactions",
                "coordinate_axes",
                "include_evidence_coordinate_interactions",
            },
            context=context,
        )
        axes = _require_json_string_array(
            raw,
            "coordinate_axes",
            (),
            context=context,
        )
        if axes is None:  # pragma: no cover - helper contract
            raise RuntimeError(
                "coordinate_axes parser returned an invalid value"
            )
        return cls(
            degree=_require_json_integer(
                raw,
                "degree",
                1,
                context=context,
            ),
            coordinate_degree=_require_json_integer(
                raw,
                "coordinate_degree",
                1,
                context=context,
            ),
            include_pairwise_interactions=_require_json_bool(
                raw,
                "include_pairwise_interactions",
                False,
                context=context,
            ),
            coordinate_axes=axes,
            include_evidence_coordinate_interactions=_require_json_bool(
                raw,
                "include_evidence_coordinate_interactions",
                False,
                context=context,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the compact user-visible representation."""
        payload: dict[str, Any] = {}
        if self.degree != 1:
            payload["degree"] = self.degree
        if self.coordinate_degree != 1:
            payload["coordinate_degree"] = self.coordinate_degree
        if self.include_pairwise_interactions:
            payload["include_pairwise_interactions"] = True
        if self.coordinate_axes:
            payload["coordinate_axes"] = list(self.coordinate_axes)
        if self.include_evidence_coordinate_interactions:
            payload["include_evidence_coordinate_interactions"] = True
        return payload


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
    feature_expansions: Mapping[str, EvidenceFeatureExpansionConfig] = field(
        default_factory=dict
    )

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
        if not isinstance(self.feature_expansions, Mapping) or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(expansion, EvidenceFeatureExpansionConfig)
            for name, expansion in self.feature_expansions.items()
        ):
            raise TypeError(
                "evidence.feature_expansions must map component names to "
                "EvidenceFeatureExpansionConfig instances"
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
                "feature_expansions",
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
        feature_expansions = raw.get("feature_expansions", {})
        if not isinstance(feature_expansions, Mapping):
            raise TypeError(
                "evidence.feature_expansions must be a JSON object"
            )
        if any(
            not isinstance(expansion, Mapping)
            for expansion in feature_expansions.values()
        ):
            raise TypeError(
                "evidence.feature_expansions values must be JSON objects"
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
            feature_expansions={
                str(name): EvidenceFeatureExpansionConfig.from_dict(
                    expansion,
                    str(name),
                )
                for name, expansion in feature_expansions.items()
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        payload: dict[str, Any] = {
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
        if self.feature_expansions:
            payload["feature_expansions"] = {
                name: expansion.to_dict()
                for name, expansion in self.feature_expansions.items()
            }
        return payload


@dataclass(frozen=True)
class SpatialFieldConfig:
    """u_c(s, z) spatial-field configuration."""

    enabled: bool = True
    backend: str = "latticekrigx"
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
class KleiberProfileConfig:
    """Frozen bivariate correlation profile for one likelihood family."""

    r0: float
    r1: float

    def __post_init__(self) -> None:
        """Validate direct programmatic construction."""
        r0 = _require_finite_real_value(
            self.r0, context="KleiberProfileConfig.r0"
        )
        r1 = _require_finite_real_value(
            self.r1, context="KleiberProfileConfig.r1"
        )
        if not -1.0 < r0 < 1.0:
            raise GEOPFAValueError(
                "KleiberProfileConfig.r0 must be in (-1, 1)"
            )
        if r1 < 0.0:
            raise GEOPFAValueError("KleiberProfileConfig.r1 must be >= 0")
        object.__setattr__(self, "r0", r0)
        object.__setattr__(self, "r1", r1)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, family: str
    ) -> KleiberProfileConfig:
        """Build one required family profile from parsed JSON."""
        context = f"inference.gblk_bayesian.kleiber_profiles.{family}"
        _reject_unknown_keys(raw, {"r0", "r1"}, context=context)
        missing = {"r0", "r1"} - set(raw)
        if missing:
            raise GEOPFAValueError(
                f"{context} requires " + " and ".join(sorted(missing))
            )
        return cls(
            r0=_require_finite_real_value(raw["r0"], context=f"{context}.r0"),
            r1=_require_finite_real_value(raw["r1"], context=f"{context}.r1"),
        )

    def to_dict(self) -> dict[str, float]:
        """Return a plain-dict representation."""
        return {"r0": self.r0, "r1": self.r1}


@dataclass(frozen=True)
class GBLKBayesianConfig:
    """Configuration for geoPFA's sole Bayesian inference path.

    When ``enabled=True`` and ``inference.backend="gblk"``, the runner uses
    LatticeKrigX's public Paige/INLA joint fitter and propagates its paired
    posterior draws to component and joint-event probability summaries.

    Training alpha offsets enter the likelihood and prediction alpha offsets
    enter every posterior draw. The Paige prior parameters and, for a
    bivariate likelihood family, a separate published Kleiber profile must be
    frozen as part of a science-run specification.
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
    kleiber_profiles: Mapping[str, KleiberProfileConfig] = field(
        default_factory=dict
    )
    cluster_effect: bool = True
    validate_inla: bool = True

    def __post_init__(self) -> None:
        """Require typed profiles keyed by a supported likelihood family."""
        for family, profile in self.kleiber_profiles.items():
            if family not in ALLOWED_OBSERVATION_FAMILIES:
                allowed = ", ".join(ALLOWED_OBSERVATION_FAMILIES)
                raise GEOPFAValueError(
                    "inference.gblk_bayesian.kleiber_profiles keys must be "
                    f"one of: {allowed}"
                )
            if not isinstance(profile, KleiberProfileConfig):
                raise GEOPFAValueError(
                    "inference.gblk_bayesian.kleiber_profiles values must be "
                    "KleiberProfileConfig instances"
                )

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
                "kleiber_profiles",
                "cluster_effect",
                "validate_inla",
            },
            context="inference.gblk_bayesian",
        )
        profiles_raw = raw.get("kleiber_profiles", {})
        if not isinstance(profiles_raw, Mapping):
            raise GEOPFAValueError(
                "inference.gblk_bayesian.kleiber_profiles must be a mapping"
            )
        profiles: dict[str, KleiberProfileConfig] = {}
        for family, profile_raw in profiles_raw.items():
            if not isinstance(family, str):
                raise GEOPFAValueError(
                    "inference.gblk_bayesian.kleiber_profiles keys must be strings"
                )
            if not isinstance(profile_raw, Mapping):
                raise GEOPFAValueError(
                    "inference.gblk_bayesian.kleiber_profiles."
                    f"{family} must be a mapping"
                )
            profiles[family] = KleiberProfileConfig.from_dict(
                profile_raw, family=family
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
            kleiber_profiles=profiles,
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
            "kleiber_profiles": {
                family: profile.to_dict()
                for family, profile in self.kleiber_profiles.items()
            },
            "cluster_effect": self.cluster_effect,
            "validate_inla": self.validate_inla,
        }


@dataclass(frozen=True)
class PredictiveStackingConfig:
    """Componentwise shrinkage selected from blocked predictive risk."""

    enabled: bool = False
    validation_depths_m: Mapping[str, float] = field(default_factory=dict)
    minimum_training_wells: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid component-specific target depths."""
        if self.minimum_training_wells is not None:
            minimum = _require_integer_value(
                self.minimum_training_wells,
                context=(
                    "inference.predictive_stacking.minimum_training_wells"
                ),
            )
            if minimum < 1:
                raise GEOPFAValueError(
                    "inference.predictive_stacking.minimum_training_wells "
                    "must be a positive integer"
                )
        if not isinstance(self.validation_depths_m, Mapping):
            raise TypeError(
                "inference.predictive_stacking.validation_depths_m must be "
                "an object"
            )
        for component, depth in self.validation_depths_m.items():
            if not isinstance(component, str) or not component:
                raise TypeError(
                    "inference.predictive_stacking.validation_depths_m keys "
                    "must be non-empty component names"
                )
            numeric_depth = _require_finite_real_value(
                depth,
                context=(
                    "inference.predictive_stacking.validation_depths_m."
                    f"{component}"
                ),
            )
            if numeric_depth < 0.0:
                raise GEOPFAValueError(
                    "inference.predictive_stacking.validation_depths_m values "
                    "must be non-negative"
                )

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any] | None
    ) -> PredictiveStackingConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        _reject_unknown_keys(
            raw,
            {"enabled", "validation_depths_m", "minimum_training_wells"},
            context="inference.predictive_stacking",
        )
        raw_depths = raw.get("validation_depths_m", {})
        if not isinstance(raw_depths, Mapping):
            raise TypeError(
                "inference.predictive_stacking.validation_depths_m must be "
                "an object"
            )
        return cls(
            enabled=_require_json_bool(
                raw,
                "enabled",
                False,
                context="inference.predictive_stacking",
            ),
            validation_depths_m={
                component: _require_finite_real_value(
                    depth,
                    context=(
                        "inference.predictive_stacking.validation_depths_m."
                        f"{component}"
                    ),
                )
                for component, depth in raw_depths.items()
            },
            minimum_training_wells=(
                None
                if raw.get("minimum_training_wells") is None
                else _require_integer_value(
                    raw["minimum_training_wells"],
                    context=(
                        "inference.predictive_stacking.minimum_training_wells"
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        output: dict[str, Any] = {"enabled": self.enabled}
        if self.validation_depths_m:
            output["validation_depths_m"] = dict(self.validation_depths_m)
        if self.minimum_training_wells is not None:
            output["minimum_training_wells"] = self.minimum_training_wells
        return output


@dataclass(frozen=True)
class InferenceConfig:
    """Inference backend selection + nested backend-specific configs."""

    backend: str = "gblk"
    gblk_bayesian: GBLKBayesianConfig = field(
        default_factory=GBLKBayesianConfig
    )
    predictive_stacking: PredictiveStackingConfig = field(
        default_factory=PredictiveStackingConfig
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
            raw,
            {"backend", "gblk_bayesian", "predictive_stacking"},
            context="inference",
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
            predictive_stacking=PredictiveStackingConfig.from_dict(
                raw.get("predictive_stacking")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "backend": self.backend,
            "gblk_bayesian": self.gblk_bayesian.to_dict(),
            "predictive_stacking": self.predictive_stacking.to_dict(),
        }


@dataclass(frozen=True)
class CalibrationConfig:
    """Post-hoc calibration configuration."""

    method: str = "none"
    fit_on: str = "block_cv"
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
        known_keys = {"method", "fit_on", "n_bins"}
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
            n_bins=_require_json_integer(
                raw, "n_bins", 5, context="calibration"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {
            "method": self.method,
            "fit_on": self.fit_on,
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
    """Product combination for conditional component probabilities."""

    rule: str = "product"

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
        _reject_unknown_keys(raw, {"rule"}, context="combination")
        rule = raw.get("rule", "product")
        if rule not in ALLOWED_COMBINATION_RULES:
            allowed = ", ".join(ALLOWED_COMBINATION_RULES)
            raise ValueError(
                f"combination.rule must be one of: {allowed} (got {rule!r})",
            )
        return cls(rule=rule)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return {"rule": self.rule}


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
        Reserved prediction-grid contract. All fields must currently be
        ``None``; predictions use the first configured PFA component grid.
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
        if self.inference.predictive_stacking.enabled and not (
            self.inference.backend == "gblk"
            and self.inference.gblk_bayesian.enabled
        ):
            raise ValueError(
                "inference.predictive_stacking.enabled=True requires "
                "Bayesian GBLK inference"
            )
        stacking_depths = (
            self.inference.predictive_stacking.validation_depths_m
        )
        if stacking_depths and self.dimensions != "3d":
            raise ValueError(
                "inference.predictive_stacking.validation_depths_m requires "
                "dimensions='3d'"
            )
        if stacking_depths and self.labels.depth_col is None:
            raise ValueError(
                "inference.predictive_stacking.validation_depths_m requires "
                "labels.depth_col"
            )
        unknown_stacking_depths = set(stacking_depths) - set(
            self.labels.label_columns
        )
        if unknown_stacking_depths:
            raise ValueError(
                "inference.predictive_stacking.validation_depths_m contains "
                "unknown component(s): "
                + ", ".join(sorted(unknown_stacking_depths))
            )
        unknown_expansions = set(self.evidence.feature_expansions) - set(
            self.labels.label_columns
        )
        if unknown_expansions:
            raise ValueError(
                "evidence.feature_expansions contains unknown component(s): "
                + ", ".join(sorted(unknown_expansions))
            )
        if self.dimensions != "3d":
            invalid_z = [
                name
                for name, expansion in self.evidence.feature_expansions.items()
                if "z" in expansion.coordinate_axes
            ]
            if invalid_z:
                raise ValueError(
                    "evidence.feature_expansions coordinate axis 'z' requires "
                    "dimensions='3d' for component(s): "
                    + ", ".join(sorted(invalid_z))
                )
        prior_only_expansions = [
            name
            for name in self.evidence.feature_expansions
            if name in self.alpha and self.alpha[name].force_prior_predictive
        ]
        if prior_only_expansions:
            raise ValueError(
                "evidence.feature_expansions require an outcome-updated "
                "component; force_prior_predictive is set for: "
                + ", ".join(sorted(prior_only_expansions))
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
        dimensions = str(raw.get("dimensions", "2d"))
        outputs_raw = raw.get("outputs")
        if dimensions == "3d" and (
            outputs_raw is None
            or (
                isinstance(outputs_raw, Mapping)
                and "format" not in outputs_raw
            )
        ):
            outputs_raw = {
                **({} if outputs_raw is None else outputs_raw),
                "format": ["vtk", "csv"],
            }
        cfg = cls(
            enabled=_require_json_bool(
                raw, "enabled", True, context="probabilistic"
            ),
            output_dir=Path(raw.get("output_dir", "outputs/probabilistic/")),
            dimensions=dimensions,
            grid=GridConfig.from_dict(raw.get("grid")),
            labels=LabelsConfig.from_dict(raw.get("labels")),
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
            outputs=OutputsConfig.from_dict(outputs_raw),
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

    def validate(self) -> list[str]:  # noqa: PLR0912, PLR0914, PLR0915
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
            self.outputs.posterior_draw_blocks
            and self.inference.predictive_stacking.enabled
        ):
            errors.append(
                "inference.predictive_stacking currently requires "
                "outputs.posterior_draw_blocks=false"
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
        if any(
            value is not None
            for value in (
                self.grid.nx,
                self.grid.ny,
                self.grid.nz,
                self.grid.extent,
                self.grid.crs,
            )
        ):
            errors.append(
                "grid overrides are not implemented by the probabilistic "
                "workflow; leave grid fields null to use the canonical PFA "
                "component grid"
            )
        if (
            self.inference.backend == "gblk"
            and self.spatial_field.enabled
            and self.spatial_field.backend != "latticekrigx"
        ):
            errors.append(
                "inference.backend='gblk' with spatial_field.enabled=true "
                "requires spatial_field.backend='latticekrigx'"
            )
        if (
            self.inference.backend == "gblk"
            and self.calibration.method != "none"
        ):
            errors.append(
                "GBLK requires calibration.method='none'; use the explicit "
                "raw cross-validation workflow for diagnostics"
            )
        if self.dimensions == "3d" and "geotiff" in self.outputs.format:
            errors.append(
                "GeoTIFF output is not defined for 3-D probability volumes; "
                "request VTK, CSV, or Parquet"
            )
        if self.dimensions == "2d" and "vtk" in self.outputs.format:
            errors.append(
                "VTK output is reserved for 3-D probability volumes; request "
                "GeoTIFF, CSV, or Parquet for 2-D analyses"
            )

        raster_thermal_components = {
            name
            for name, alpha in self.alpha.items()
            if alpha.mode == "thermal_exceedance"
        }
        if self.dimensions == "3d" and raster_thermal_components:
            errors.append(
                "alpha.mode='thermal_exceedance' samples only x/y from a "
                "2-D raster and cannot define a 3-D thermal prior; use "
                "alpha.mode='thermal_layer_exceedance' with a 3-D PFA layer "
                "for component(s): "
                + ", ".join(sorted(raster_thermal_components))
            )

        label_components = set(self.labels.label_columns)
        alpha_components = set(self.alpha)
        observation_model_components = set(self.labels.observation_models)
        try:
            validate_surface_names(
                (*self.labels.label_columns, *self.alpha),
                context="probabilistic component",
                reserved=("combined",),
            )
        except GEOPFAValueError as exc:
            errors.append(str(exc))
        data_informed_components = {
            name
            for name, alpha in self.alpha.items()
            if not alpha.force_prior_predictive
        }
        labels_contract_complete = (
            self.labels.source is not None
            and self.labels.id_col is not None
            and bool(label_components)
        )
        if data_informed_components and not labels_contract_complete:
            errors.append(
                "data-informed components require labels.source, "
                "labels.id_col, and non-empty labels.label_columns"
            )
        if self.site_selection.mode != "off" and not labels_contract_complete:
            errors.append(
                "site_selection requires labels.source, labels.id_col, and "
                "non-empty labels.label_columns"
            )
        bernoulli_label_components = {
            name
            for name in label_components
            if self.labels.observation_model_for(name).family == "bernoulli"
        }
        if (
            self.site_selection.mode != "off"
            and labels_contract_complete
            and not bernoulli_label_components
        ):
            errors.append(
                "site_selection requires at least one Bernoulli component; "
                "continuous Gaussian outcomes are not binary selection events"
            )
        unknown_observation_models = observation_model_components - (
            label_components | alpha_components
        )
        if unknown_observation_models:
            errors.append(
                "labels.observation_models contains unknown component(s): "
                + ", ".join(sorted(unknown_observation_models))
            )
        implicit_prior_observation_models = {
            name
            for name in observation_model_components - label_components
            if name not in self.alpha
            or not self.alpha[name].force_prior_predictive
        }
        if implicit_prior_observation_models:
            errors.append(
                "labels.observation_models components without label mappings "
                "must have an alpha configuration with "
                "force_prior_predictive=True; offending components: "
                + ", ".join(sorted(implicit_prior_observation_models))
            )
        gaussian_components = {
            name
            for name, model in self.labels.observation_models.items()
            if model.family == "gaussian"
        }
        fitted_gaussian_components = {
            name
            for name in gaussian_components
            & label_components
            & alpha_components
            if not self.alpha[name].force_prior_predictive
        }
        fitted_bernoulli_components = (
            data_informed_components
            & label_components
            & alpha_components - gaussian_components
        )
        prior_only_gaussian_components = {
            name
            for name in gaussian_components & alpha_components
            if self.alpha[name].force_prior_predictive
        }
        gaussian_prior_response_scales = {
            name
            for name in prior_only_gaussian_components
            if self.labels.observation_models[name].response_scale is not None
        }
        if gaussian_prior_response_scales:
            errors.append(
                "Gaussian prior-only components must omit response_scale "
                "because their configured thermal moments are already in "
                "physical units; offending components: "
                + ", ".join(sorted(gaussian_prior_response_scales))
            )
        gaussian_prior_evidence = {
            name
            for name in prior_only_gaussian_components
            if self.alpha[name].use_evidence_prior
        }
        if gaussian_prior_evidence:
            errors.append(
                "Gaussian prior-only components cannot use "
                "alpha.use_evidence_prior because that update is defined on "
                "the Bernoulli logit scale; offending components: "
                + ", ".join(sorted(gaussian_prior_evidence))
            )
        gaussian_prior_missing_raster_uncertainty = {
            name
            for name in prior_only_gaussian_components
            if self.alpha[name].mode == "thermal_exceedance"
            and self.alpha[name].uncertainty_raster is None
        }
        if gaussian_prior_missing_raster_uncertainty:
            errors.append(
                "Gaussian prior-only thermal_exceedance components require "
                "uncertainty_raster; offending components: "
                + ", ".join(sorted(gaussian_prior_missing_raster_uncertainty))
            )
        gaussian_prior_missing_layer_uncertainty = {
            name
            for name in prior_only_gaussian_components
            if self.alpha[name].mode == "thermal_layer_exceedance"
            and self.alpha[name].uncertainty_column is None
        }
        if gaussian_prior_missing_layer_uncertainty:
            errors.append(
                "Gaussian prior-only thermal_layer_exceedance components "
                "require uncertainty_column; offending components: "
                + ", ".join(sorted(gaussian_prior_missing_layer_uncertainty))
            )
        gaussian_fit_missing_response_scale = {
            name
            for name in fitted_gaussian_components
            if self.labels.observation_models[name].response_scale is None
        }
        if gaussian_fit_missing_response_scale:
            errors.append(
                "Fitted Gaussian observation models require response_scale; "
                "offending components: "
                + ", ".join(sorted(gaussian_fit_missing_response_scale))
            )
        if fitted_gaussian_components and not (
            self.inference.backend == "gblk"
            and self.inference.gblk_bayesian.enabled
        ):
            errors.append(
                "Gaussian component observations require Bayesian GBLK inference"
            )
        if fitted_gaussian_components and self.labels.pu_mode != "off":
            errors.append(
                "Gaussian component observations require labels.pu_mode='off'"
            )
        if fitted_gaussian_components and self.outputs.posterior_draw_blocks:
            errors.append(
                "Gaussian component observations currently require "
                "outputs.posterior_draw_blocks=false"
            )
        invalid_gaussian_alpha = {
            name
            for name in gaussian_components
            if self.alpha.get(name) is None
            or self.alpha[name].mode
            not in {"thermal_exceedance", "thermal_layer_exceedance"}
        }
        if invalid_gaussian_alpha:
            errors.append(
                "Gaussian heat components require a thermal exceedance alpha "
                "with a continuous prior mean; offending components: "
                + ", ".join(sorted(invalid_gaussian_alpha))
            )
        gaussian_stacking_without_uncertainty = {
            name
            for name in fitted_gaussian_components - invalid_gaussian_alpha
            if self.inference.predictive_stacking.enabled
            and not self.alpha[name].force_prior_predictive
            and (
                (
                    self.alpha[name].mode == "thermal_exceedance"
                    and self.alpha[name].uncertainty_raster is None
                )
                or (
                    self.alpha[name].mode == "thermal_layer_exceedance"
                    and self.alpha[name].uncertainty_column is None
                )
            )
        }
        if gaussian_stacking_without_uncertainty:
            errors.append(
                "Gaussian predictive stacking requires prior predictive "
                "uncertainty; offending components: "
                + ", ".join(sorted(gaussian_stacking_without_uncertainty))
            )
        if (
            self.inference.gblk_bayesian.enabled
            and data_informed_components
            and not self.spatial_field.enabled
        ):
            errors.append(
                "Bayesian GBLK with data-informed components requires "
                "spatial_field.enabled=True"
            )
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
            if alpha_cfg.mode == "scalar":
                continue
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
        if bayes.enabled:
            fitted_by_family = {
                "bernoulli": fitted_bernoulli_components,
                "gaussian": fitted_gaussian_components,
            }
            for family, component_names in fitted_by_family.items():
                count = len(component_names)
                profile_configured = family in bayes.kleiber_profiles
                if count > 2:  # noqa: PLR2004
                    errors.append(
                        "Bayesian GBLK supports at most two fitted "
                        f"{family} components; got {count}: "
                        + ", ".join(sorted(component_names))
                    )
                if count == 2 and not profile_configured:  # noqa: PLR2004
                    errors.append(
                        f"bivariate {family} Bayesian GBLK requires "
                        "inference.gblk_bayesian.kleiber_profiles."
                        f"{family}"
                    )
                if count != 2 and profile_configured:  # noqa: PLR2004
                    errors.append(
                        "inference.gblk_bayesian.kleiber_profiles."
                        f"{family} is valid only for exactly two fitted "
                        f"{family} components; got {count}"
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

    Calls ``ProbabilisticConfig.validate_raise`` after parsing so
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
    "ALLOWED_OBSERVATION_FAMILIES",
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
    "EvidenceFeatureExpansionConfig",
    "GBLKBayesianConfig",
    "GridConfig",
    "InferenceConfig",
    "KleiberProfileConfig",
    "LabelsConfig",
    "ObservationModelConfig",
    "OutputsConfig",
    "PredictiveStackingConfig",
    "ProbabilisticConfig",
    "RegularizationConfig",
    "ScenarioConfig",
    "SiteSelectionConfig",
    "SpatialFieldConfig",
    "load_probabilistic_config",
]
