"""Play-type defaults registry for per-feature L2 regularization weights.

Maps play-type names (e.g. ``"extensional"``, ``"magmatic"``) to
per-feature regularization weights and MAP prior means via substring
matching on layer names. This lets users declare
``regularization.play_type = "extensional"`` in the probabilistic config
instead of having to specify per-feature weights by hand.

Design principle
----------------
The registry is keyed by **layer-name substrings** (lower-cased). A layer
named ``"quaternary_fault_density"`` matches the pattern ``"fault"`` in the
extensional play's definition. This makes the registry dataset-agnostic:
users can bring any evidence layers with any names and the registry will
match what it can, silently ignoring layers it doesn't recognize.

Extension
---------
Add a new play type by inserting an entry into ``PLAY_TYPE_REGISTRY``::

    PLAY_TYPE_REGISTRY["my_play"] = {
        "pattern_weights": {"keyword": weight_value, ...},
        "pattern_means": {"keyword": prior_mean, ...},
    }

where ``weight_value`` is the per-feature L2 penalty (higher = more
regularization toward ``prior_mean``).
"""

from __future__ import annotations

import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLAY_TYPE_REGISTRY: dict[str, dict[str, Any]] = {
    "extensional": {
        # Structural layers: strong positive prior (faults favor reservoir).
        "pattern_weights": {
            "fault": 1.0,
            "slip": 1.0,
            "dilation": 1.0,
            "seism": 0.5,
            "strain": 0.5,
            "gravity": 0.5,
            "heat": 2.0,
            "thermal": 2.0,
            "temperature": 2.0,
        },
        "pattern_means": {
            "fault": 0.5,
            "slip": 0.5,
            "dilation": 0.5,
            "seism": 0.3,
            "strain": 0.3,
            "gravity": 0.2,
            "heat": 1.0,
            "thermal": 1.0,
            "temperature": 1.0,
        },
    },
    "magmatic": {
        "pattern_weights": {
            "heat": 2.0,
            "thermal": 2.0,
            "temperature": 2.0,
            "volcanic": 2.0,
            "fault": 0.5,
            "gravity": 1.0,
        },
        "pattern_means": {
            "heat": 1.5,
            "thermal": 1.5,
            "temperature": 1.5,
            "volcanic": 1.5,
            "fault": 0.2,
            "gravity": 0.5,
        },
    },
    "convective": {
        "pattern_weights": {
            "permeability": 2.0,
            "porosity": 2.0,
            "fault": 1.0,
            "fracture": 1.0,
            "heat": 1.0,
            "thermal": 1.0,
        },
        "pattern_means": {
            "permeability": 1.0,
            "porosity": 1.0,
            "fault": 0.5,
            "fracture": 0.5,
            "heat": 0.8,
            "thermal": 0.8,
        },
    },
}


def available_play_types() -> list[str]:
    """Return the list of registered play-type names."""
    return sorted(PLAY_TYPE_REGISTRY.keys())


def play_type_defaults(
    play_type: str, *, layer_names: tuple[str, ...] | list[str]
) -> dict[str, dict[str, float]]:
    """Look up per-feature weights and prior means for a given play type.

    Parameters
    ----------
    play_type
        One of the keys in ``PLAY_TYPE_REGISTRY``.
    layer_names
        The layer names produced by ``_flatten_component_features``
        (e.g. ``"fault_slip"``). Matching is done case-insensitively against
        each pattern substring.

    Returns
    -------
    dict with two keys:

    * ``"per_feature_weights"`` — ``{layer_name: weight}`` for matched layers.
    * ``"prior_means"`` — ``{layer_name: mean}`` for matched layers.

    Layers that do not match any pattern in the registry are silently omitted;
    the caller (``build_fit_kwargs``) applies a scalar default regularization
    to all unmatched features.
    """
    if play_type not in PLAY_TYPE_REGISTRY:
        warnings.warn(
            f"unknown play type {play_type!r}; ignoring regularization hints. "
            f"Available types: {available_play_types()}",
            UserWarning,
            stacklevel=2,
        )
        return {"per_feature_weights": {}, "prior_means": {}}

    entry = PLAY_TYPE_REGISTRY[play_type]
    pattern_weights: dict[str, float] = entry["pattern_weights"]
    pattern_means: dict[str, float] = entry["pattern_means"]

    per_feature_weights: dict[str, float] = {}
    prior_means: dict[str, float] = {}
    for name in layer_names:
        name_lower = name.lower()
        for pattern, weight in pattern_weights.items():
            if pattern in name_lower:
                per_feature_weights[name] = weight
                prior_means[name] = float(pattern_means.get(pattern, 0.0))
                break  # first match wins; patterns are checked in insertion order

    return {
        "per_feature_weights": per_feature_weights,
        "prior_means": prior_means,
    }


__all__ = ["PLAY_TYPE_REGISTRY", "available_play_types", "play_type_defaults"]
