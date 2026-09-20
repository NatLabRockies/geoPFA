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
* Unknown keys are rejected at every parsed config level; this catches typos
  in user-facing config files early.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
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
ORDINARY_LIKELIHOOD_SEMANTICS = "ordinary_bayesian_likelihood"
POWER_LIKELIHOOD_SEMANTICS = "generalized_bayesian_power_likelihood"
ALLOWED_OBSERVATION_WEIGHT_SEMANTICS: tuple[str, ...] = (
    ORDINARY_LIKELIHOOD_SEMANTICS,
    POWER_LIKELIHOOD_SEMANTICS,
)
_SHA256_HEX_LENGTH = 64
_DOMAIN_BOUND_COUNT = 2

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


def _require_optional_integer_value(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    return _require_integer_value(value, context=context)


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


def _require_spatial_domain(
    value: Any,
) -> tuple[tuple[float, float], ...] | None:
    """Return finite, ordered 2-D or 3-D physical-domain bounds."""
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) not in {2, 3}:
        raise GEOPFAValueError(
            "spatial_field.spatial_domain must contain two or three "
            "[lower, upper] axis bounds"
        )
    domain: list[tuple[float, float]] = []
    for axis, bounds in enumerate(value):
        if (
            not isinstance(bounds, list | tuple)
            or len(bounds) != _DOMAIN_BOUND_COUNT
        ):
            raise GEOPFAValueError(
                "spatial_field.spatial_domain axis "
                f"{axis} must contain [lower, upper]"
            )
        lower = _require_finite_real_value(
            bounds[0],
            context=f"spatial_field.spatial_domain[{axis}][0]",
        )
        upper = _require_finite_real_value(
            bounds[1],
            context=f"spatial_field.spatial_domain[{axis}][1]",
        )
        if lower >= upper:
            raise GEOPFAValueError(
                "spatial_field.spatial_domain upper bounds must be strictly "
                f"greater than lower bounds on axis {axis}"
            )
        domain.append((lower, upper))
    return tuple(domain)


def _validate_spatial_domain_dimension(
    spatial_domain: tuple[tuple[float, float], ...] | None,
    *,
    dimensions: str,
    enabled: bool,
) -> None:
    """Validate cross-block dimension and activation constraints."""
    if spatial_domain is None:
        return
    expected_dimension = 3 if dimensions == "3d" else 2
    if len(spatial_domain) != expected_dimension:
        raise ValueError(
            "spatial_field.spatial_domain must contain "
            f"{expected_dimension} axis bounds for dimensions={dimensions!r}"
        )
    if not enabled:
        raise ValueError(
            "spatial_field.spatial_domain requires spatial_field.enabled=True"
        )


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


def _parse_config_fields(
    raw: Mapping[str, Any],
    *,
    defaults: Any,
    schema: Mapping[str, str],
    context: str,
    hint: str | None = None,
) -> dict[str, Any]:
    """Parse common scalar config fields with strict JSON types."""
    _reject_unknown_keys(
        raw,
        set(schema),
        context=context,
        hint=hint,
    )
    parsers = {
        "raw": raw.get,
        "bool": lambda name, default: _require_json_bool(
            raw, name, default, context=context
        ),
        "int": lambda name, default: _require_json_integer(
            raw, name, default, context=context
        ),
        "optional_int": lambda name, default: (
            _require_optional_integer_value(
                raw.get(name, default), context=f"{context}.{name}"
            )
        ),
        "real": lambda name, default: _require_json_real(
            raw, name, default, context=context
        ),
        "optional_real": lambda name, _default: _require_optional_json_real(
            raw, name, context=context
        ),
        "optional_string": lambda name, default: _require_json_string(
            raw, name, default, context=context, allow_none=True
        ),
        "string": lambda name, default: _require_json_string(
            raw, name, default, context=context
        ),
        "string_array": lambda name, default: _require_json_string_array(
            raw, name, default, context=context
        ),
        "optional_string_array": lambda name, default: (
            _require_json_string_array(
                raw, name, default, context=context, allow_none=True
            )
        ),
        "real_array": lambda name, default: _require_json_real_array(
            raw, name, default, context=context
        ),
        "real_mapping": lambda name, default: {
            str(key): _require_finite_real_value(
                value, context=f"{context}.{name}.{key}"
            )
            for key, value in dict(raw.get(name, default)).items()
        },
        "spatial_domain": lambda name, default: _require_spatial_domain(
            raw.get(name, default)
        ),
    }
    parsed: dict[str, Any] = {}
    for name, kind in schema.items():
        if kind == "skip":
            continue
        parser = parsers.get(kind)
        if parser is None:  # pragma: no cover - internal schema definition
            raise RuntimeError(f"unknown config parser kind {kind!r}")
        parsed[name] = parser(name, getattr(defaults, name))
    return parsed


def _json_config_value(value: Any) -> Any:
    """Convert nested immutable config values to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Mapping):
        return {
            str(key): _json_config_value(item) for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_json_config_value(item) for item in value]
    return value


def _config_to_dict(
    config: Any,
    *,
    omit_none: frozenset[str] = frozenset(),
    omit_empty: frozenset[str] = frozenset(),
    omit_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one config dataclass without repeating field plumbing."""
    defaults = {} if omit_defaults is None else omit_defaults
    payload: dict[str, Any] = {}
    for config_field in fields(config):
        name = config_field.name
        value = getattr(config, name)
        if name in omit_none and value is None:
            continue
        if name in omit_empty and not value:
            continue
        if name in defaults and value == defaults[name]:
            continue
        payload[name] = _json_config_value(value)
    return payload


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
        extent = raw.get("extent")
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "nx": "optional_int",
                    "ny": "optional_int",
                    "nz": "optional_int",
                    "extent": "skip",
                    "crs": "raw",
                },
                context="grid",
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
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return _config_to_dict(self, omit_none=frozenset({"response_scale"}))


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
    prior_response_mean_columns: Mapping[str, str] = field(
        default_factory=dict
    )
    prior_response_sd_columns: Mapping[str, str] = field(default_factory=dict)
    observation_weight_columns: Mapping[str, str] = field(default_factory=dict)
    observation_weight_semantics: str | None = None
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
        for field_name, columns in (
            ("prior_response_mean_columns", self.prior_response_mean_columns),
            ("prior_response_sd_columns", self.prior_response_sd_columns),
        ):
            if not isinstance(columns, Mapping) or any(
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(column, str)
                or not column.strip()
                for name, column in columns.items()
            ):
                raise ValueError(
                    f"labels.{field_name} must map non-empty component names "
                    "to non-empty column names"
                )
            unknown_components = set(columns) - set(self.label_columns)
            if unknown_components:
                raise ValueError(
                    f"labels.{field_name} references component(s) without "
                    "label columns: " + ", ".join(sorted(unknown_components))
                )
            non_gaussian = {
                name
                for name in columns
                if self.observation_model_for(name).family != "gaussian"
            }
            if non_gaussian:
                raise ValueError(
                    f"labels.{field_name} is defined only for Gaussian "
                    "components; got " + ", ".join(sorted(non_gaussian))
                )
        sd_without_mean = set(self.prior_response_sd_columns) - set(
            self.prior_response_mean_columns
        )
        if sd_without_mean:
            raise ValueError(
                "labels.prior_response_sd_columns requires a matching "
                "labels.prior_response_mean_columns entry for: "
                + ", ".join(sorted(sd_without_mean))
            )
        if not isinstance(self.observation_weight_columns, Mapping) or any(
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(column, str)
            or not column.strip()
            for name, column in self.observation_weight_columns.items()
        ):
            raise ValueError(
                "labels.observation_weight_columns must map non-empty "
                "component names to non-empty column names"
            )
        unknown_weight_components = set(self.observation_weight_columns) - set(
            self.label_columns
        )
        if unknown_weight_components:
            raise ValueError(
                "labels.observation_weight_columns references component(s) "
                "without label columns: "
                + ", ".join(sorted(unknown_weight_components))
            )
        if (
            self.observation_weight_semantics is not None
            and self.observation_weight_semantics
            not in ALLOWED_OBSERVATION_WEIGHT_SEMANTICS
        ):
            allowed = ", ".join(ALLOWED_OBSERVATION_WEIGHT_SEMANTICS)
            raise ValueError(
                "labels.observation_weight_semantics must be one of: "
                f"{allowed} (got {self.observation_weight_semantics!r})"
            )
        if (
            self.observation_weight_columns
            and self.observation_weight_semantics is None
        ):
            raise ValueError(
                "labels.observation_weight_columns requires explicit "
                "labels.observation_weight_semantics"
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
        label_columns = raw.get("label_columns", {})
        if not isinstance(label_columns, Mapping):
            raise TypeError("labels.label_columns must be a JSON object")
        prior_response_mean_columns = raw.get(
            "prior_response_mean_columns", {}
        )
        prior_response_sd_columns = raw.get("prior_response_sd_columns", {})
        observation_weight_columns = raw.get("observation_weight_columns", {})
        if not isinstance(prior_response_mean_columns, Mapping):
            raise TypeError(
                "labels.prior_response_mean_columns must be a JSON object"
            )
        if not isinstance(prior_response_sd_columns, Mapping):
            raise TypeError(
                "labels.prior_response_sd_columns must be a JSON object"
            )
        if not isinstance(observation_weight_columns, Mapping):
            raise TypeError(
                "labels.observation_weight_columns must be a JSON object"
            )
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "source": "optional_string",
                    "id_col": "optional_string",
                    "label_columns": "skip",
                    "source_crs": "optional_string",
                    "x_col": "optional_string",
                    "y_col": "optional_string",
                    "z_col": "optional_string",
                    "depth_col": "optional_string",
                    "observation_models": "skip",
                    "prior_response_mean_columns": "skip",
                    "prior_response_sd_columns": "skip",
                    "observation_weight_columns": "skip",
                    "observation_weight_semantics": "optional_string",
                    "layer": "raw",
                    "min_wells_for_fit": "int",
                    "pu_mode": "raw",
                    "pu_class_prior": "skip",
                },
                context="labels",
            ),
            label_columns=dict(label_columns),
            observation_models={
                str(name): ObservationModelConfig.from_dict(model, str(name))
                for name, model in dict(
                    raw.get("observation_models", {})
                ).items()
            },
            prior_response_mean_columns=dict(prior_response_mean_columns),
            prior_response_sd_columns=dict(prior_response_sd_columns),
            observation_weight_columns=dict(observation_weight_columns),
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
        return _config_to_dict(
            self,
            omit_none=frozenset({"observation_weight_semantics"}),
        )


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "C": "real",
                    "per_feature_weights": "real_mapping",
                    "prior_means": "real_mapping",
                    "prior_precisions": "real_mapping",
                },
                context="evidence.regularization",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "degree": "int",
                    "coordinate_degree": "int",
                    "include_pairwise_interactions": "bool",
                    "coordinate_axes": "string_array",
                    "include_evidence_coordinate_interactions": "bool",
                },
                context=context,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the compact user-visible representation."""
        return _config_to_dict(
            self,
            omit_defaults={
                "degree": 1,
                "coordinate_degree": 1,
                "include_pairwise_interactions": False,
                "coordinate_axes": (),
                "include_evidence_coordinate_interactions": False,
            },
        )


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
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "regularization": "skip",
                    "include_layers": "optional_string_array",
                    "exclude_layers": "string_array",
                    "sparse_binary_threshold": "real",
                    "coordinate_blacklist": "string_array",
                    "standardization": "string",
                    "feature_expansions": "skip",
                },
                context="evidence",
            ),
            regularization=RegularizationConfig.from_dict(
                raw.get("regularization")
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
        return _config_to_dict(
            self,
            omit_empty=frozenset({"feature_expansions"}),
        )


@dataclass(frozen=True)
class SpatialFieldConfig:
    """u_c(s, z) spatial-field configuration."""

    enabled: bool = True
    backend: str = "latticekrigx"
    n_levels: int = 2
    lattice_centers_per_dimension: int = 6
    coordinate_scaling: str = "axis_range"
    spatial_domain: tuple[tuple[float, float], ...] | None = None

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
        object.__setattr__(
            self,
            "spatial_domain",
            _require_spatial_domain(self.spatial_domain),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> SpatialFieldConfig:
        """Build from a parsed-JSON mapping."""
        if raw is None:
            return cls()
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "enabled": "bool",
                    "backend": "raw",
                    "n_levels": "int",
                    "lattice_centers_per_dimension": "int",
                    "coordinate_scaling": "raw",
                    "spatial_domain": "spatial_domain",
                },
                context="spatial_field",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(
            self,
            omit_none=frozenset({"spatial_domain"}),
        )


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "mode": "string",
                    "candidate_source": "optional_string",
                    "id_col": "optional_string",
                    "selected_col": "string",
                    "outcome_feature_columns": "string_array",
                    "selection_feature_columns": "string_array",
                    "outcome_selection_log_odds": "real_array",
                    "outcome_penalty": "real",
                    "selection_penalty": "real",
                },
                context="site_selection",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return _config_to_dict(self)


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
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "enabled": "bool",
                    "n_draws": "int",
                    "seed": "int",
                    "ci_level": "real",
                    "cor_scale_median": "real",
                    "spatial_sd_u": "real",
                    "spatial_sd_tail_probability": "real",
                    "dirichlet_concentration": "real",
                    "separate_ranges": "bool",
                    "cluster_effect": "bool",
                    "validate_inla": "bool",
                    "kleiber_profiles": "skip",
                },
                context="inference.gblk_bayesian",
            ),
            kleiber_profiles=profiles,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        raw_depths = raw.get("validation_depths_m", {})
        if not isinstance(raw_depths, Mapping):
            raise TypeError(
                "inference.predictive_stacking.validation_depths_m must be "
                "an object"
            )
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "enabled": "bool",
                    "minimum_training_wells": "optional_int",
                    "validation_depths_m": "skip",
                },
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
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(
            self,
            omit_none=frozenset({"minimum_training_wells"}),
            omit_empty=frozenset({"validation_depths_m"}),
        )


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "backend": "raw",
                    "gblk_bayesian": "skip",
                    "predictive_stacking": "skip",
                },
                context="inference",
            ),
            gblk_bayesian=GBLKBayesianConfig.from_dict(
                raw.get("gblk_bayesian")
            ),
            predictive_stacking=PredictiveStackingConfig.from_dict(
                raw.get("predictive_stacking")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "method": "raw",
                    "fit_on": "raw",
                    "n_bins": "int",
                },
                context="calibration",
                hint="use 'fit_on', not 'fit'",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "n_folds": "int",
                    "block_type": "raw",
                    "block_size_km": "optional_real",
                    "grid_size": "int",
                    "buffer_km": "real",
                },
                context="cross_validation",
                hint="use 'n_folds', not 'n_splits'",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={"rule": "raw"},
                context="combination",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(self)


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
        return _config_to_dict(self)


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
    posterior_draw_cell_indices_source: str | None = None
    posterior_draw_cell_indices_sha256: str | None = None
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
        source = self.posterior_draw_cell_indices_source
        checksum = self.posterior_draw_cell_indices_sha256
        if (source is None) != (checksum is None):
            raise ValueError(
                "outputs.posterior_draw_cell_indices_source and "
                "outputs.posterior_draw_cell_indices_sha256 must be declared "
                "together"
            )
        if source is not None and (
            not isinstance(source, str) or not source.strip()
        ):
            raise ValueError(
                "outputs.posterior_draw_cell_indices_source must be a "
                "non-empty string"
            )
        if checksum is not None and (
            not isinstance(checksum, str)
            or len(checksum) != _SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in checksum
            )
        ):
            raise ValueError(
                "outputs.posterior_draw_cell_indices_sha256 must be a "
                "lowercase SHA-256 digest"
            )
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
        return cls(
            **_parse_config_fields(
                raw,
                defaults=cls(),
                schema={
                    "probability_rasters": "bool",
                    "uncertainty_rasters": "bool",
                    "calibration_artifacts": "bool",
                    "decision_artifacts": "bool",
                    "scenarios": "bool",
                    "posterior_draw_blocks": "bool",
                    "posterior_draw_block_size": "int",
                    "posterior_draw_cell_indices_source": "optional_string",
                    "posterior_draw_cell_indices_sha256": "optional_string",
                    "format": "string_array",
                },
                context="outputs",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation."""
        return _config_to_dict(
            self,
            omit_none=frozenset(
                {
                    "posterior_draw_cell_indices_source",
                    "posterior_draw_cell_indices_sha256",
                }
            ),
        )


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


def _validate_output_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate output and posterior-state controls."""
    if str(config.output_dir).strip() in {"", "."}:
        errors.append(
            "output_dir is empty or '.'; provide an explicit output path to avoid "
            "writing probabilistic outputs into the current working directory"
        )
    if (
        config.outputs.posterior_draw_blocks
        and not config.inference.gblk_bayesian.enabled
    ):
        errors.append(
            "outputs.posterior_draw_blocks requires Bayesian GBLK inference "
            "(inference.backend='gblk' and "
            "inference.gblk_bayesian.enabled=true)"
        )
    if (
        config.outputs.posterior_draw_blocks
        and config.inference.gblk_bayesian.cluster_effect
    ):
        errors.append(
            "outputs.posterior_draw_blocks requires "
            "inference.gblk_bayesian.cluster_effect=false so grid prediction "
            "is projected in bounded draw blocks rather than materialized by INLA"
        )
    if (
        config.outputs.posterior_draw_blocks
        and config.inference.predictive_stacking.enabled
    ):
        errors.append(
            "inference.predictive_stacking currently requires "
            "outputs.posterior_draw_blocks=false"
        )
    if (
        config.outputs.posterior_draw_cell_indices_source is not None
        and not config.outputs.posterior_draw_blocks
    ):
        errors.append(
            "outputs.posterior_draw_cell_indices_source requires "
            "outputs.posterior_draw_blocks=true"
        )
    if (
        config.inference.gblk_bayesian.enabled
        and config.alpha
        and all(
            alpha.force_prior_predictive and not alpha.use_evidence_prior
            for alpha in config.alpha.values()
        )
    ):
        errors.append(
            "inference.gblk_bayesian.enabled=True has no stochastic "
            "components: every component is a fixed prior prediction"
        )


def _validate_grid_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate the reserved prediction-grid contract."""
    for axis_name, size in (
        ("nx", config.grid.nx),
        ("ny", config.grid.ny),
        ("nz", config.grid.nz),
    ):
        if size is not None and size < 1:
            errors.append(f"grid.{axis_name} must be >= 1 (got {size})")
    if config.grid.extent is not None:
        expected_extent_length = 4 if config.dimensions == "2d" else 6
        if len(config.grid.extent) != expected_extent_length:
            errors.append(
                "grid.extent must contain "
                f"{expected_extent_length} values for {config.dimensions} "
                f"(got {len(config.grid.extent)})"
            )
        else:
            half = expected_extent_length // 2
            if any(
                lower >= upper
                for lower, upper in zip(
                    config.grid.extent[:half],
                    config.grid.extent[half:],
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
            config.grid.nx,
            config.grid.ny,
            config.grid.nz,
            config.grid.extent,
            config.grid.crs,
        )
    ):
        errors.append(
            "grid overrides are not implemented by the probabilistic "
            "workflow; leave grid fields null to use the canonical PFA "
            "component grid"
        )


def _validate_backend_dimension_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate backend and dimensional compatibility."""
    if (
        config.inference.backend == "gblk"
        and config.spatial_field.enabled
        and config.spatial_field.backend != "latticekrigx"
    ):
        errors.append(
            "inference.backend='gblk' with spatial_field.enabled=true "
            "requires spatial_field.backend='latticekrigx'"
        )
    if (
        config.inference.backend == "gblk"
        and config.calibration.method != "none"
    ):
        errors.append(
            "GBLK requires calibration.method='none'; use the explicit "
            "raw cross-validation workflow for diagnostics"
        )
    if config.dimensions == "3d" and "geotiff" in config.outputs.format:
        errors.append(
            "GeoTIFF output is not defined for 3-D probability volumes; "
            "request VTK, CSV, or Parquet"
        )
    if config.dimensions == "2d" and "vtk" in config.outputs.format:
        errors.append(
            "VTK output is reserved for 3-D probability volumes; request "
            "GeoTIFF, CSV, or Parquet for 2-D analyses"
        )

    raster_thermal_components = {
        name
        for name, alpha in config.alpha.items()
        if alpha.mode == "thermal_exceedance"
    }
    if config.dimensions == "3d" and raster_thermal_components:
        errors.append(
            "alpha.mode='thermal_exceedance' samples only x/y from a "
            "2-D raster and cannot define a 3-D thermal prior; use "
            "alpha.mode='thermal_layer_exceedance' with a 3-D PFA layer "
            "for component(s): " + ", ".join(sorted(raster_thermal_components))
        )


@dataclass(frozen=True)
class _ValidationComponents:
    """Component-family sets shared by cross-field validators."""

    label: frozenset[str]
    alpha: frozenset[str]
    observation_model: frozenset[str]
    data_informed: frozenset[str]
    bernoulli_label: frozenset[str]
    gaussian: frozenset[str]
    fitted_gaussian: frozenset[str]
    fitted_bernoulli: frozenset[str]
    prior_only_gaussian: frozenset[str]


def _validation_components(
    config: ProbabilisticConfig,
) -> _ValidationComponents:
    """Derive component families once for all cross-field checks."""
    label = frozenset(config.labels.label_columns)
    alpha = frozenset(config.alpha)
    observation_model = frozenset(config.labels.observation_models)
    data_informed = frozenset(
        name
        for name, alpha_config in config.alpha.items()
        if not alpha_config.force_prior_predictive
    )
    bernoulli_label = frozenset(
        name
        for name in label
        if config.labels.observation_model_for(name).family == "bernoulli"
    )
    gaussian = frozenset(
        name
        for name, model in config.labels.observation_models.items()
        if model.family == "gaussian"
    )
    fitted_gaussian = frozenset(
        name
        for name in gaussian & label & alpha
        if not config.alpha[name].force_prior_predictive
    )
    fitted_bernoulli = (data_informed & label & alpha) - gaussian
    prior_only_gaussian = frozenset(
        name
        for name in gaussian & alpha
        if config.alpha[name].force_prior_predictive
    )
    return _ValidationComponents(
        label=label,
        alpha=alpha,
        observation_model=observation_model,
        data_informed=data_informed,
        bernoulli_label=bernoulli_label,
        gaussian=gaussian,
        fitted_gaussian=fitted_gaussian,
        fitted_bernoulli=fitted_bernoulli,
        prior_only_gaussian=prior_only_gaussian,
    )


def _validate_label_contract(
    config: ProbabilisticConfig,
    components: _ValidationComponents,
    errors: list[str],
) -> None:
    """Validate label and observation-family membership contracts."""
    try:
        validate_surface_names(
            (*config.labels.label_columns, *config.alpha),
            context="probabilistic component",
            reserved=("combined",),
        )
    except GEOPFAValueError as exc:
        errors.append(str(exc))

    labels_contract_complete = (
        config.labels.source is not None
        and config.labels.id_col is not None
        and bool(components.label)
    )
    if components.data_informed and not labels_contract_complete:
        errors.append(
            "data-informed components require labels.source, "
            "labels.id_col, and non-empty labels.label_columns"
        )
    if config.site_selection.mode != "off" and not labels_contract_complete:
        errors.append(
            "site_selection requires labels.source, labels.id_col, and "
            "non-empty labels.label_columns"
        )
    if (
        config.site_selection.mode != "off"
        and labels_contract_complete
        and not components.bernoulli_label
    ):
        errors.append(
            "site_selection requires at least one Bernoulli component; "
            "continuous Gaussian outcomes are not binary selection events"
        )

    unknown_observation_models = components.observation_model - (
        components.label | components.alpha
    )
    if unknown_observation_models:
        errors.append(
            "labels.observation_models contains unknown component(s): "
            + ", ".join(sorted(unknown_observation_models))
        )
    implicit_prior_observation_models = {
        name
        for name in components.observation_model - components.label
        if name not in config.alpha
        or not config.alpha[name].force_prior_predictive
    }
    if implicit_prior_observation_models:
        errors.append(
            "labels.observation_models components without label mappings "
            "must have an alpha configuration with "
            "force_prior_predictive=True; offending components: "
            + ", ".join(sorted(implicit_prior_observation_models))
        )


def _validate_gaussian_component_contract(
    config: ProbabilisticConfig,
    components: _ValidationComponents,
    errors: list[str],
) -> None:
    """Validate continuous-observation and thermal-prior contracts."""
    gaussian_prior_response_scales = {
        name
        for name in components.prior_only_gaussian
        if config.labels.observation_models[name].response_scale is not None
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
        for name in components.prior_only_gaussian
        if config.alpha[name].use_evidence_prior
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
        for name in components.prior_only_gaussian
        if config.alpha[name].mode == "thermal_exceedance"
        and config.alpha[name].uncertainty_raster is None
    }
    if gaussian_prior_missing_raster_uncertainty:
        errors.append(
            "Gaussian prior-only thermal_exceedance components require "
            "uncertainty_raster; offending components: "
            + ", ".join(sorted(gaussian_prior_missing_raster_uncertainty))
        )
    gaussian_prior_missing_layer_uncertainty = {
        name
        for name in components.prior_only_gaussian
        if config.alpha[name].mode == "thermal_layer_exceedance"
        and config.alpha[name].uncertainty_column is None
    }
    if gaussian_prior_missing_layer_uncertainty:
        errors.append(
            "Gaussian prior-only thermal_layer_exceedance components "
            "require uncertainty_column; offending components: "
            + ", ".join(sorted(gaussian_prior_missing_layer_uncertainty))
        )
    gaussian_fit_missing_response_scale = {
        name
        for name in components.fitted_gaussian
        if config.labels.observation_models[name].response_scale is None
    }
    if gaussian_fit_missing_response_scale:
        errors.append(
            "Fitted Gaussian observation models require response_scale; "
            "offending components: "
            + ", ".join(sorted(gaussian_fit_missing_response_scale))
        )
    if components.fitted_gaussian and not (
        config.inference.backend == "gblk"
        and config.inference.gblk_bayesian.enabled
    ):
        errors.append(
            "Gaussian component observations require Bayesian GBLK inference"
        )
    if components.fitted_gaussian and config.labels.pu_mode != "off":
        errors.append(
            "Gaussian component observations require labels.pu_mode='off'"
        )
    if config.labels.observation_weight_columns and not (
        config.inference.backend == "gblk"
        and config.inference.gblk_bayesian.enabled
    ):
        errors.append(
            "configured observation weights require Bayesian GBLK inference"
        )

    invalid_gaussian_alpha = {
        name
        for name in components.gaussian
        if config.alpha.get(name) is None
        or config.alpha[name].mode
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
        for name in components.fitted_gaussian - invalid_gaussian_alpha
        if config.inference.predictive_stacking.enabled
        and not config.alpha[name].force_prior_predictive
        and (
            (
                config.alpha[name].mode == "thermal_exceedance"
                and config.alpha[name].uncertainty_raster is None
            )
            or (
                config.alpha[name].mode == "thermal_layer_exceedance"
                and config.alpha[name].uncertainty_column is None
            )
        )
    }
    if gaussian_stacking_without_uncertainty:
        errors.append(
            "Gaussian predictive stacking requires prior predictive "
            "uncertainty; offending components: "
            + ", ".join(sorted(gaussian_stacking_without_uncertainty))
        )


def _validate_data_fit_contract(
    config: ProbabilisticConfig,
    components: _ValidationComponents,
    errors: list[str],
) -> None:
    """Validate requirements shared by all data-informed components."""
    if (
        config.inference.gblk_bayesian.enabled
        and components.data_informed
        and not config.spatial_field.enabled
    ):
        errors.append(
            "Bayesian GBLK with data-informed components requires "
            "spatial_field.enabled=True"
        )
    minimum_allowed = 1 if config.inference.gblk_bayesian.enabled else 2
    if (
        components.data_informed
        and config.labels.min_wells_for_fit < minimum_allowed
    ):
        errors.append(
            "labels.min_wells_for_fit must be >= "
            f"{minimum_allowed} for the selected inference mode "
            f"(got {config.labels.min_wells_for_fit})"
        )


def _validate_alpha_contract(
    config: ProbabilisticConfig,
    components: _ValidationComponents,
    errors: list[str],
) -> None:
    """Validate component-prior coverage and probability bounds."""
    if not config.alpha:
        errors.append(
            "alpha is empty; configure at least one component's prior offset"
        )

    missing_alpha = components.label - components.alpha
    if missing_alpha:
        errors.append(
            "every labels.label_columns component requires an alpha "
            "configuration; missing: " + ", ".join(sorted(missing_alpha))
        )
    unlabeled_alpha = components.alpha - components.label
    implicit_prior_only = {
        name
        for name in unlabeled_alpha
        if not config.alpha[name].force_prior_predictive
    }
    if implicit_prior_only:
        errors.append(
            "alpha components without label mappings must set "
            "force_prior_predictive=True; offending components: "
            + ", ".join(sorted(implicit_prior_only))
        )
    if config.inference.backend == "sequential" and unlabeled_alpha:
        errors.append(
            "inference.backend='sequential' does not support alpha-only "
            "components; add label mappings or use the GBLK prior-predictive "
            "path: " + ", ".join(sorted(unlabeled_alpha))
        )

    for component_name, alpha_config in config.alpha.items():
        prior_probability = alpha_config.scalar_fallback_pr0
        if not 0.0 < prior_probability < 1.0:
            errors.append(
                f"alpha.{component_name}.scalar_fallback_pr0="
                f"{prior_probability} must be strictly in (0, 1)"
            )
        if alpha_config.mode == "scalar":
            continue
        if not 0.0 < alpha_config.p_min < 0.5:  # noqa: PLR2004
            errors.append(
                f"alpha.{component_name}.p_min must be in (0, 0.5) "
                f"(got {alpha_config.p_min})"
            )
        if not 0.5 < alpha_config.p_max < 1.0:  # noqa: PLR2004
            errors.append(
                f"alpha.{component_name}.p_max must be in (0.5, 1) "
                f"(got {alpha_config.p_max})"
            )
        if alpha_config.p_min >= alpha_config.p_max:
            errors.append(
                f"alpha.{component_name}.p_min ({alpha_config.p_min}) must be "
                f"< p_max ({alpha_config.p_max})"
            )


def _validate_evidence_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate evidence regularization and prior-update controls."""
    regularization = config.evidence.regularization
    if regularization.C <= 0:
        errors.append(
            f"evidence.regularization.C must be > 0 (got {regularization.C})"
        )
    for name, weight in regularization.per_feature_weights.items():
        if weight < 0.0:
            errors.append(
                "evidence.regularization.per_feature_weights values must "
                f"be non-negative (got {name}={weight})"
            )
    for name, precision in regularization.prior_precisions.items():
        if precision <= 0.0:
            errors.append(
                "evidence.regularization.prior_precisions values must be "
                f"strictly positive (got {name}={precision})"
            )
    evidence_prior_components = {
        name
        for name, alpha_config in config.alpha.items()
        if alpha_config.use_evidence_prior
    }
    if (
        evidence_prior_components
        and not config.inference.gblk_bayesian.enabled
    ):
        errors.append(
            "alpha.use_evidence_prior requires "
            "inference.gblk_bayesian.enabled=True; offending components: "
            + ", ".join(sorted(evidence_prior_components))
        )
    if not 0.0 < config.evidence.sparse_binary_threshold <= 1.0:
        errors.append(
            "evidence.sparse_binary_threshold must be in (0, 1] "
            f"(got {config.evidence.sparse_binary_threshold})"
        )


def _validate_diagnostics_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate cross-validation and calibration controls."""
    cross_validation = config.cross_validation
    if cross_validation.n_folds < 2:  # noqa: PLR2004
        errors.append(
            "cross_validation.n_folds must be >= 2 "
            f"(got {cross_validation.n_folds})"
        )
    if cross_validation.grid_size < 1:
        errors.append(
            "cross_validation.grid_size must be >= 1 "
            f"(got {cross_validation.grid_size})"
        )
    if cross_validation.buffer_km < 0.0:
        errors.append(
            "cross_validation.buffer_km must be non-negative "
            f"(got {cross_validation.buffer_km})"
        )
    if (
        cross_validation.block_size_km is not None
        and cross_validation.block_size_km <= 0.0
    ):
        errors.append(
            "cross_validation.block_size_km must be > 0 when provided "
            f"(got {cross_validation.block_size_km})"
        )
    if (
        cross_validation.block_type != "grid"
        and cross_validation.block_size_km is not None
    ):
        errors.append(
            "cross_validation.block_size_km is only defined for "
            "block_type='grid'"
        )
    if config.calibration.n_bins < 1:
        errors.append(
            f"calibration.n_bins must be >= 1 (got {config.calibration.n_bins})"
        )


def _validate_bayesian_contract(
    config: ProbabilisticConfig,
    components: _ValidationComponents,
    errors: list[str],
) -> None:
    """Validate Bayesian controls and fitted-family dimensions."""
    bayesian = config.inference.gblk_bayesian
    if bayesian.n_draws < 1:
        errors.append(
            "inference.gblk_bayesian.n_draws must be >= 1 "
            f"(got {bayesian.n_draws})"
        )
    if not 0.0 < bayesian.ci_level < 1.0:
        errors.append(
            "inference.gblk_bayesian.ci_level must be in (0, 1) "
            f"(got {bayesian.ci_level})"
        )
    for name, value in (
        ("cor_scale_median", bayesian.cor_scale_median),
        ("spatial_sd_u", bayesian.spatial_sd_u),
        ("dirichlet_concentration", bayesian.dirichlet_concentration),
    ):
        if not math.isfinite(value) or value <= 0.0:
            errors.append(
                f"inference.gblk_bayesian.{name} must be finite and > 0"
            )
    if not 0.0 < bayesian.spatial_sd_tail_probability < 1.0:
        errors.append(
            "inference.gblk_bayesian.spatial_sd_tail_probability "
            "must be in (0, 1)"
        )
    if not bayesian.enabled:
        return
    for family, component_names in (
        ("bernoulli", components.fitted_bernoulli),
        ("gaussian", components.fitted_gaussian),
    ):
        count = len(component_names)
        profile_configured = family in bayesian.kleiber_profiles
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


def _validate_spatial_scenario_contract(
    config: ProbabilisticConfig,
    errors: list[str],
) -> None:
    """Validate spatial controls and scenario compatibility."""
    spatial = config.spatial_field
    if spatial.n_levels < 1:
        errors.append(
            f"spatial_field.n_levels must be >= 1 (got {spatial.n_levels})"
        )
    if spatial.lattice_centers_per_dimension < 1:
        errors.append(
            "spatial_field.lattice_centers_per_dimension must be >= 1 "
            f"(got {spatial.lattice_centers_per_dimension})"
        )
    if config.labels.pu_mode == "nnpu" and config.inference.backend != (
        "sequential"
    ):
        errors.append(
            "labels.pu_mode='nnpu' currently requires "
            "inference.backend='sequential'"
        )
    if spatial.enabled and spatial.backend == "none":
        errors.append("spatial_field.enabled=True requires a spatial backend")
    scenario_names = [scenario.name for scenario in config.scenarios]
    if len(set(scenario_names)) != len(scenario_names):
        errors.append("scenarios[*].name values must be unique")
    if config.inference.gblk_bayesian.enabled and any(
        not scenario.include_spatial for scenario in config.scenarios
    ):
        errors.append(
            "Bayesian GBLK scenarios must set include_spatial=True; "
            "the sole Bayesian fitter has no nonspatial latent-field mode"
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
        _validate_spatial_domain_dimension(
            self.spatial_field.spatial_domain,
            dimensions=self.dimensions,
            enabled=self.spatial_field.enabled,
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
        return _config_to_dict(self)

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

    def validate(self) -> list[str]:
        """Validate cross-field constraints and numeric hyperparameter ranges.

        Returns a list of error strings; empty list means the config is valid.
        Call :meth:`validate_raise` to raise :class:`ValueError` on first error.
        """
        errors: list[str] = []
        _validate_output_contract(self, errors)
        _validate_grid_contract(self, errors)
        _validate_backend_dimension_contract(self, errors)

        components = _validation_components(self)
        _validate_label_contract(self, components, errors)
        _validate_gaussian_component_contract(self, components, errors)
        _validate_data_fit_contract(self, components, errors)
        _validate_alpha_contract(self, components, errors)
        _validate_evidence_contract(self, errors)
        _validate_diagnostics_contract(self, errors)
        _validate_bayesian_contract(self, components, errors)
        _validate_spatial_scenario_contract(self, errors)
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
        outputs=replace(
            cfg.outputs,
            posterior_draw_cell_indices_source=_resolve(
                cfg.outputs.posterior_draw_cell_indices_source
            ),
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
