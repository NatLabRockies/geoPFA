"""Tests for the post-hoc probability-calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from geopfa.prob.calibration import (
    CalibrationMap,
    expected_calibration_error,
    fit_posthoc_calibration,
)
from geopfa.prob.calibration import (
    equal_frequency_reliability_table as _bins,
)


def _ece(y: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    return expected_calibration_error(_bins(y, p, n_bins=n_bins))


def test_fit_posthoc_platt_reduces_ece_on_underconfident_predictions() -> None:
    rng = np.random.default_rng(0)
    n = 800
    y = rng.binomial(1, 0.6, size=n).astype(int)
    # Under-confident: shrink p toward 0.5
    p_true = np.where(y == 1, 0.85, 0.15)
    p_shrunken = 0.4 + 0.2 * (p_true - 0.5)  # roughly [0.31, 0.49]
    raw_ece = _ece(y, p_shrunken)
    cm = fit_posthoc_calibration(p_shrunken, y, method="platt")
    p_cal = cm.predict(p_shrunken)
    cal_ece = _ece(y, p_cal)
    assert isinstance(cm, CalibrationMap)
    assert cm.method == "platt"
    assert cal_ece < raw_ece * 0.8  # at least a 20% reduction


def test_fit_posthoc_isotonic_reduces_ece_on_overconfident_predictions() -> (
    None
):
    rng = np.random.default_rng(1)
    n = 1000
    y = rng.binomial(1, 0.5, size=n).astype(int)
    # Over-confident: push p away from 0.5
    p_true = np.where(y == 1, 0.7, 0.3)
    p_sharpened = np.clip((p_true - 0.5) * 1.8 + 0.5, 0.01, 0.99)
    raw_ece = _ece(y, p_sharpened)
    cm = fit_posthoc_calibration(p_sharpened, y, method="isotonic")
    p_cal = cm.predict(p_sharpened)
    cal_ece = _ece(y, p_cal)
    assert cm.method == "isotonic"
    assert cal_ece < raw_ece


def test_fit_posthoc_temperature_method() -> None:
    rng = np.random.default_rng(2)
    n = 400
    y = rng.binomial(1, 0.5, size=n).astype(int)
    # Over-confident: sharpen
    p = np.where(y == 1, 0.95, 0.05)
    cm = fit_posthoc_calibration(p, y, method="temperature")
    assert cm.method == "temperature"
    p_cal = cm.predict(p)
    # Temperature scaling preserves rank ordering; just check it's defined
    assert p_cal.shape == p.shape
    assert np.all(np.isfinite(p_cal))


def test_fit_posthoc_none_method_is_identity() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, size=50)
    y = rng.binomial(1, p)
    cm = fit_posthoc_calibration(p, y, method="none")
    out = cm.predict(p)
    np.testing.assert_allclose(out, p)


def test_fit_posthoc_unknown_method_raises() -> None:
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="method"):
        fit_posthoc_calibration(p, y, method="vibes")


def test_calibration_map_serialise_roundtrip() -> None:
    rng = np.random.default_rng(4)
    p = rng.uniform(0.05, 0.95, size=120)
    y = rng.binomial(1, p)
    cm = fit_posthoc_calibration(p, y, method="platt")
    payload = cm.to_dict()
    assert payload["method"] == "platt"
    rebuilt = CalibrationMap.from_dict(payload)
    np.testing.assert_allclose(cm.predict(p), rebuilt.predict(p))
