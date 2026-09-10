"""Unit tests for :mod:`geopfa.prob.calibration`."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from geopfa.prob import calibration


def test_wilson_ci_n_zero_returns_full_range() -> None:
    lo, hi = calibration.wilson_ci(k=0, n=0)
    assert lo == 0.0 and hi == 1.0


def test_wilson_ci_extreme_observations_remain_in_unit_interval() -> None:
    lo_zero, hi_zero = calibration.wilson_ci(k=0, n=10)
    lo_one, hi_one = calibration.wilson_ci(k=10, n=10)
    assert 0.0 <= lo_zero <= hi_zero <= 1.0
    assert 0.0 <= lo_one <= hi_one <= 1.0
    assert lo_zero == 0.0
    assert hi_one == pytest.approx(1.0)


def test_wilson_ci_centred_observation_brackets_p_hat() -> None:
    lo, hi = calibration.wilson_ci(k=15, n=30)
    assert lo < 0.5 < hi


def test_wilson_ci_narrows_with_sample_size() -> None:
    lo_small, hi_small = calibration.wilson_ci(k=2, n=10)
    lo_large, hi_large = calibration.wilson_ci(k=20, n=100)
    assert (hi_small - lo_small) > (hi_large - lo_large)


@pytest.mark.parametrize(
    ("k", "n", "alpha"),
    [(-1, 10, 0.05), (11, 10, 0.05), (1, -1, 0.05), (1, 10, 1.0)],
)
def test_wilson_ci_rejects_invalid_binomial_inputs(
    k: int, n: int, alpha: float
) -> None:
    with pytest.raises(ValueError):
        calibration.wilson_ci(k=k, n=n, alpha=alpha)


def test_equal_frequency_table_handles_empty_input() -> None:
    rows = calibration.equal_frequency_reliability_table([], [], n_bins=5)
    assert rows == []


def test_equal_frequency_table_n_bins_must_be_positive() -> None:
    with pytest.raises(ValueError):
        calibration.equal_frequency_reliability_table(
            [0, 1], [0.1, 0.9], n_bins=0
        )


def test_equal_frequency_table_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        calibration.equal_frequency_reliability_table(
            [0, 1, 0], [0.1, 0.9], n_bins=2
        )


def test_equal_frequency_table_partitions_full_sample() -> None:
    rng = np.random.default_rng(0)
    n = 25
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)
    rows = calibration.equal_frequency_reliability_table(y, p, n_bins=5)
    assert len(rows) == 5
    assert sum(r.n for r in rows) == n
    # Equal-frequency: 25 / 5 = 5 per bin exactly
    assert {r.n for r in rows} == {5}


def test_equal_frequency_table_uneven_split() -> None:
    p = np.linspace(0.0, 1.0, 7)
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    rows = calibration.equal_frequency_reliability_table(y, p, n_bins=3)
    # 7 split into 3: numpy.array_split → [3, 2, 2]
    assert [r.n for r in rows] == [3, 2, 2]
    assert sum(r.n for r in rows) == 7


def test_equal_frequency_table_is_invariant_to_tied_score_order() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.5, 0.5, 0.5, 0.5, 0.9])
    permutation = np.array([0, 4, 3, 2, 1, 5])

    def signature(labels: np.ndarray, probability: np.ndarray) -> list[tuple]:
        rows = calibration.equal_frequency_reliability_table(
            labels, probability, n_bins=3
        )
        return [
            (
                row.n,
                row.bin_p_lo,
                row.bin_p_hi,
                row.mean_predicted_p,
                row.observed_fraction,
            )
            for row in rows
        ]

    assert signature(y, p) == signature(y[permutation], p[permutation])


def test_perfectly_calibrated_predictions_have_zero_ece() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)
    rows = calibration.equal_frequency_reliability_table(y, p, n_bins=10)
    ece = calibration.expected_calibration_error(rows)
    mce = calibration.maximum_calibration_error(rows)
    # Empirical ECE on calibrated data should be small (sampling noise only).
    assert ece < 0.05
    assert mce < 0.10


def test_systematically_underconfident_predictions_have_positive_ece() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    true_p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(int)
    # Halve the predictions: model is under-confident.
    p = true_p * 0.5
    rows = calibration.equal_frequency_reliability_table(y, p, n_bins=5)
    ece = calibration.expected_calibration_error(rows)
    assert ece > 0.1
    # Every bin's observed fraction should sit above its mean prediction.
    for row in rows:
        assert row.observed_fraction > row.mean_predicted_p


def test_brier_score_zero_for_perfect_predictions() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert calibration.brier_score(y, p) == 0.0


def test_brier_score_known_value() -> None:
    y = np.array([1, 0])
    p = np.array([0.5, 0.5])
    assert calibration.brier_score(y, p) == pytest.approx(0.25)


def test_log_loss_clips_extreme_predictions() -> None:
    y = np.array([1])
    p = np.array([0.0])
    # Without clipping this would diverge; with eps it must be finite.
    assert math.isfinite(calibration.log_loss(y, p))


def test_log_loss_known_value_for_uniform_predictions() -> None:
    y = np.array([0, 1])
    p = np.array([0.5, 0.5])
    assert calibration.log_loss(y, p) == pytest.approx(-math.log(0.5))


def test_log_loss_handles_empty_arrays() -> None:
    assert math.isnan(calibration.log_loss([], []))


@pytest.mark.parametrize(
    "function",
    [
        calibration.equal_frequency_reliability_table,
        calibration.brier_score,
        calibration.log_loss,
        calibration.fit_temperature,
        calibration.calibration_intercept_slope,
        calibration.calibration_summary,
    ],
)
def test_calibration_functions_reject_impossible_probabilities(
    function,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        function(np.array([0, 1]), np.array([0.2, 1.2]))


@pytest.mark.parametrize(
    "function",
    [
        calibration.equal_frequency_reliability_table,
        calibration.brier_score,
        calibration.log_loss,
        calibration.fit_temperature,
        calibration.calibration_intercept_slope,
        calibration.calibration_summary,
    ],
)
def test_calibration_functions_reject_fractional_labels(function) -> None:
    with pytest.raises(ValueError, match="binary 0/1"):
        function(np.array([0.5, 1.0]), np.array([0.2, 0.8]))


def test_calibration_map_rejects_invalid_prediction_probability() -> None:
    mapping = calibration.CalibrationMap(method="none", params={})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mapping.predict(np.array([-0.1, 0.8]))


def test_posthoc_calibration_rejects_single_class_fit() -> None:
    with pytest.raises(ValueError, match="two outcome classes"):
        calibration.fit_posthoc_calibration(
            np.array([0.1, 0.2, 0.3]),
            np.array([0, 0, 0]),
            method="platt",
        )


def test_fit_temperature_returns_one_for_single_class() -> None:
    y = np.array([0, 0, 0, 0])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    T, p_T = calibration.fit_temperature(y, p)
    assert T == 1.0
    np.testing.assert_allclose(p_T, p, atol=1e-9)


def test_fit_temperature_softens_overconfident_predictions() -> None:
    rng = np.random.default_rng(3)
    n = 2000
    true_p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(int)
    # Sharpen via T_true=0.5 to simulate over-confidence.
    logits = np.log(
        np.clip(true_p, 1e-9, 1 - 1e-9)
        / (1.0 - np.clip(true_p, 1e-9, 1 - 1e-9))
    )
    p_overconfident = 1.0 / (1.0 + np.exp(-logits / 0.5))
    T, _ = calibration.fit_temperature(y, p_overconfident)
    assert T > 1.2  # temperature should soften


def test_fit_temperature_sharpens_underconfident_predictions() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    true_p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < true_p).astype(int)
    logits = np.log(
        np.clip(true_p, 1e-9, 1 - 1e-9)
        / (1.0 - np.clip(true_p, 1e-9, 1 - 1e-9))
    )
    # Under-confident: divide logits by 2.0 (i.e., apply T=2.0 to true logits).
    p_underconfident = 1.0 / (1.0 + np.exp(-logits / 2.0))
    T, _ = calibration.fit_temperature(y, p_underconfident)
    assert T < 0.85


def test_fit_temperature_optimizer_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        calibration,
        "minimize_scalar",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False, x=1.0, message="sentinel failure"
        ),
    )

    with pytest.raises(RuntimeError, match="sentinel failure"):
        calibration.fit_temperature(
            np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6])
        )


def test_calibration_intercept_optimizer_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize  # noqa: PLC0415

    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            x=np.array([0.0, 1.0]),
            message="sentinel failure",
        ),
    )

    with pytest.raises(RuntimeError, match="sentinel failure"):
        calibration.calibration_intercept_slope(
            np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6])
        )


def test_platt_optimizer_failure_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize  # noqa: PLC0415

    monkeypatch.setattr(
        scipy.optimize,
        "minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            x=np.array([1.0, 0.0]),
            message="sentinel failure",
        ),
    )

    with pytest.raises(RuntimeError, match="sentinel failure"):
        calibration.fit_posthoc_calibration(
            np.array([0.2, 0.8, 0.4, 0.6]),
            np.array([0, 1, 0, 1]),
            method="platt",
        )


def test_calibration_summary_reports_all_expected_keys() -> None:
    rng = np.random.default_rng(5)
    n = 200
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(0.0, 1.0, size=n) < p).astype(int)
    summary = calibration.calibration_summary(y, p, n_bins=4)
    expected_keys = {
        "n",
        "n_bins",
        "ECE",
        "MCE",
        "brier",
        "log_loss",
        "bins_raw",
        "temperature_T",
        "ECE_post_temperature",
        "MCE_post_temperature",
        "brier_post_temperature",
        "log_loss_post_temperature",
        "bins_post_temperature",
    }
    assert expected_keys.issubset(summary.keys())
    assert summary["n"] == n
    assert summary["n_bins"] == 4
    assert len(summary["bins_raw"]) == 4
    assert len(summary["bins_post_temperature"]) == 4
