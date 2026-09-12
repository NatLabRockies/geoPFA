"""Frozen deterministic forward models for downstream sensitivity analysis.

The archive contract in this module preserves a fitted GBLK linear predictor
as named logit-scale pieces.  It deliberately does not assign probability laws
to those pieces: downstream analyses own their factor definitions and may
evaluate them in bounded batches without refitting the Paper 3 model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

_SCHEMA_VERSION = 1
_MATRIX_NDIM = 2
_CUBE_NDIM = 3


def _array(value: NDArray, *, dtype: np.dtype) -> NDArray:
    """Return an immutable, contiguous NumPy array."""
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrozenGBLKForwardState:
    """Sufficient deterministic state for a fitted GBLK forward map.

    ``prior_logit + evidence_logit_contribution + spatial_logit`` reconstructs
    the fitted component linear predictors.  Each evidence term is assigned to
    exactly one component by ``evidence_term_component``.  Components declared
    prior-only must have neither evidence nor spatial contributions.
    """

    coordinates: NDArray[np.float64]
    coordinate_names: tuple[str, ...]
    coordinate_units: str
    component_names: tuple[str, ...]
    prior_only_components: tuple[str, ...]
    prior_logit: NDArray[np.float64]
    evidence_logit_contribution: NDArray[np.float64]
    evidence_term_names: tuple[str, ...]
    evidence_term_component: NDArray[np.int64]
    spatial_logit: NDArray[np.float64]
    baseline_component_probability: NDArray[np.float64]
    structural_scenario: str

    def __post_init__(self) -> None:  # noqa: PLR0912, PLR0914, PLR0915
        """Normalize arrays and fail closed on an incoherent decomposition."""
        coordinates = _array(self.coordinates, dtype=np.dtype(np.float64))
        prior = _array(self.prior_logit, dtype=np.dtype(np.float64))
        evidence = _array(
            self.evidence_logit_contribution, dtype=np.dtype(np.float64)
        )
        term_component = _array(
            self.evidence_term_component, dtype=np.dtype(np.int64)
        )
        spatial = _array(self.spatial_logit, dtype=np.dtype(np.float64))
        baseline = _array(
            self.baseline_component_probability, dtype=np.dtype(np.float64)
        )
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "prior_logit", prior)
        object.__setattr__(self, "evidence_logit_contribution", evidence)
        object.__setattr__(self, "evidence_term_component", term_component)
        object.__setattr__(self, "spatial_logit", spatial)
        object.__setattr__(self, "baseline_component_probability", baseline)

        if coordinates.ndim != _MATRIX_NDIM or coordinates.shape[0] == 0:
            raise ValueError("coordinates must have shape (cell, dimension)")
        n_cells, n_dimensions = coordinates.shape
        n_components = len(self.component_names)
        n_terms = len(self.evidence_term_names)
        if n_components == 0 or len(set(self.component_names)) != n_components:
            raise ValueError("component_names must be nonempty and unique")
        if len(self.coordinate_names) != n_dimensions:
            raise ValueError(
                "coordinate_names must match coordinate dimension"
            )
        if not self.coordinate_units or not self.structural_scenario:
            raise ValueError(
                "coordinate_units and structural_scenario must be nonempty"
            )
        if prior.shape != (n_cells, n_components):
            raise ValueError("prior_logit has an invalid shape")
        if spatial.shape != prior.shape or baseline.shape != prior.shape:
            raise ValueError(
                "spatial and baseline arrays must match prior_logit"
            )
        if evidence.shape != (n_cells, n_terms):
            raise ValueError("evidence contributions have an invalid shape")
        if term_component.shape != (n_terms,):
            raise ValueError("evidence_term_component has an invalid shape")
        if n_terms and (
            np.min(term_component) < 0
            or np.max(term_component) >= n_components
        ):
            raise ValueError(
                "evidence term component indices are out of range"
            )
        arrays = (coordinates, prior, evidence, spatial, baseline)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError(
                "frozen forward arrays must contain finite values"
            )
        if np.any((baseline <= 0.0) | (baseline >= 1.0)):
            raise ValueError(
                "baseline component probabilities must lie in (0, 1)"
            )

        prior_only = set(self.prior_only_components)
        if not prior_only <= set(self.component_names):
            raise ValueError(
                "prior-only components must be declared components"
            )
        for name in prior_only:
            index = self.component_names.index(name)
            if np.any(term_component == index):
                raise ValueError(
                    "prior-only components cannot have evidence terms"
                )
            if not np.allclose(spatial[:, index], 0.0, rtol=0.0, atol=1e-14):
                raise ValueError(
                    "prior-only components cannot have spatial terms"
                )

        eta = prior + spatial
        for term_index, component_index in enumerate(term_component):
            eta[:, component_index] += evidence[:, term_index]
        reconstructed = expit(eta)
        if not np.allclose(reconstructed, baseline, rtol=0.0, atol=1e-12):
            maximum_error = float(np.max(np.abs(reconstructed - baseline)))
            raise ValueError(
                "frozen forward decomposition does not reconstruct baseline "
                f"probabilities (maximum error {maximum_error:.3g})"
            )


@dataclass(frozen=True)
class FrozenGBLKForwardResult:
    """Vectorized component and plug-in co-occurrence probabilities."""

    component_probability: NDArray[np.float64]
    combined_probability: NDArray[np.float64]


def _factor_array(
    value: NDArray[np.float64] | None,
    *,
    width: int,
    fill: float,
    name: str,
) -> NDArray[np.float64]:
    """Normalize one factor block to ``(evaluation, width)``."""
    if value is None:
        return np.full((1, width), fill, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result[np.newaxis, :]
    if result.ndim != _MATRIX_NDIM or result.shape[1] != width:
        raise ValueError(f"{name} must have shape (evaluation, {width})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def evaluate_frozen_gblk(
    state: FrozenGBLKForwardState,
    *,
    evidence_multipliers: NDArray[np.float64] | None = None,
    spatial_multipliers: NDArray[np.float64] | None = None,
    component_logit_shift: NDArray[np.float64] | None = None,
    spatial_logit_delta: NDArray[np.float64] | None = None,
) -> FrozenGBLKForwardResult:
    """Evaluate independent factor blocks against a frozen fitted model.

    All two-dimensional inputs use ``(evaluation, factor)`` ordering.  The
    optional spatial innovation uses ``(evaluation, cell, component)``.  Inputs
    with one evaluation broadcast against the largest supplied evaluation
    count; other incompatible counts fail closed.  Callers should stream large
    Monte Carlo designs in batches.
    """
    n_components = len(state.component_names)
    n_terms = len(state.evidence_term_names)
    evidence = _factor_array(
        evidence_multipliers,
        width=n_terms,
        fill=1.0,
        name="evidence_multipliers",
    )
    spatial = _factor_array(
        spatial_multipliers,
        width=n_components,
        fill=1.0,
        name="spatial_multipliers",
    )
    shift = _factor_array(
        component_logit_shift,
        width=n_components,
        fill=0.0,
        name="component_logit_shift",
    )

    innovation: NDArray[np.float64]
    if spatial_logit_delta is None:
        innovation = np.zeros(
            (1, state.coordinates.shape[0], n_components), dtype=np.float64
        )
    else:
        innovation = np.asarray(spatial_logit_delta, dtype=np.float64)
        expected_tail = (state.coordinates.shape[0], n_components)
        if (
            innovation.ndim != _CUBE_NDIM
            or innovation.shape[1:] != expected_tail
        ):
            raise ValueError(
                "spatial_logit_delta must have shape "
                f"(evaluation, {expected_tail[0]}, {expected_tail[1]})"
            )
        if not np.all(np.isfinite(innovation)):
            raise ValueError("spatial_logit_delta must contain finite values")

    counts = (
        evidence.shape[0],
        spatial.shape[0],
        shift.shape[0],
        innovation.shape[0],
    )
    n_evaluations = max(counts)
    if any(count not in {1, n_evaluations} for count in counts):
        raise ValueError("factor blocks have incompatible evaluation counts")
    evidence = np.broadcast_to(evidence, (n_evaluations, n_terms))
    spatial = np.broadcast_to(spatial, (n_evaluations, n_components))
    shift = np.broadcast_to(shift, (n_evaluations, n_components))
    innovation = np.broadcast_to(
        innovation,
        (n_evaluations, state.coordinates.shape[0], n_components),
    )

    eta = np.broadcast_to(
        state.prior_logit,
        (n_evaluations, *state.prior_logit.shape),
    ).copy()
    eta += state.spatial_logit[np.newaxis, :, :] * spatial[:, np.newaxis, :]
    eta += shift[:, np.newaxis, :]
    eta += innovation
    for term_index, component_index in enumerate(
        state.evidence_term_component
    ):
        eta[:, :, component_index] += (
            evidence[:, term_index, np.newaxis]
            * state.evidence_logit_contribution[np.newaxis, :, term_index]
        )
    component_probability = expit(eta)
    return FrozenGBLKForwardResult(
        component_probability=component_probability,
        combined_probability=np.prod(component_probability, axis=2),
    )


def save_frozen_gblk_forward_state(
    path: str | Path, state: FrozenGBLKForwardState
) -> None:
    """Write a compressed, pickle-free forward-state archive."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray(_SCHEMA_VERSION, dtype=np.int64),
        coordinates=state.coordinates,
        coordinate_names=np.asarray(state.coordinate_names, dtype=np.str_),
        coordinate_units=np.asarray(state.coordinate_units, dtype=np.str_),
        component_names=np.asarray(state.component_names, dtype=np.str_),
        prior_only_components=np.asarray(
            state.prior_only_components, dtype=np.str_
        ),
        prior_logit=state.prior_logit,
        evidence_logit_contribution=state.evidence_logit_contribution,
        evidence_term_names=np.asarray(
            state.evidence_term_names, dtype=np.str_
        ),
        evidence_term_component=state.evidence_term_component,
        spatial_logit=state.spatial_logit,
        baseline_component_probability=state.baseline_component_probability,
        structural_scenario=np.asarray(
            state.structural_scenario, dtype=np.str_
        ),
    )


def load_frozen_gblk_forward_state(path: str | Path) -> FrozenGBLKForwardState:
    """Load and validate a compressed forward-state archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        schema_version = int(np.asarray(archive["schema_version"]).item())
        if schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported frozen forward schema version {schema_version}"
            )
        return FrozenGBLKForwardState(
            coordinates=archive["coordinates"],
            coordinate_names=tuple(archive["coordinate_names"].tolist()),
            coordinate_units=str(
                np.asarray(archive["coordinate_units"]).item()
            ),
            component_names=tuple(archive["component_names"].tolist()),
            prior_only_components=tuple(
                archive["prior_only_components"].tolist()
            ),
            prior_logit=archive["prior_logit"],
            evidence_logit_contribution=archive["evidence_logit_contribution"],
            evidence_term_names=tuple(archive["evidence_term_names"].tolist()),
            evidence_term_component=archive["evidence_term_component"],
            spatial_logit=archive["spatial_logit"],
            baseline_component_probability=archive[
                "baseline_component_probability"
            ],
            structural_scenario=str(
                np.asarray(archive["structural_scenario"]).item()
            ),
        )


__all__ = [
    "FrozenGBLKForwardResult",
    "FrozenGBLKForwardState",
    "evaluate_frozen_gblk",
    "load_frozen_gblk_forward_state",
    "save_frozen_gblk_forward_state",
]
