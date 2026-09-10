"""Unit tests for :mod:`geopfa.prob.decision_metrics`."""

from __future__ import annotations

import math

import numpy as np
import pytest

from geopfa.prob import decision_metrics as dm


def test_confusion_at_threshold_basic_counts() -> None:
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.6, 0.9, 0.4, 0.8, 0.2])
    cm = dm.confusion_at_threshold(y, p, threshold=0.5)
    assert cm.tp == 2  # 0.9 and 0.8 from positives
    assert cm.fp == 1  # 0.6 from negative
    assert cm.tn == 2  # 0.1 and 0.2 from negatives
    assert cm.fn == 1  # 0.4 from positive
    assert cm.correct == 4
    assert cm.incorrect == 2
    assert cm.accuracy == pytest.approx(4 / 6)
    assert cm.precision == pytest.approx(2 / 3)
    assert cm.recall == pytest.approx(2 / 3)
    assert cm.specificity == pytest.approx(2 / 3)


def test_confusion_perfect_classifier() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    cm = dm.confusion_at_threshold(y, p, threshold=0.5)
    assert cm.fp == 0 and cm.fn == 0
    assert cm.precision == 1.0
    assert cm.recall == 1.0
    assert cm.specificity == 1.0
    assert cm.accuracy == 1.0


def test_confusion_threshold_zero_predicts_all_positive() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    cm = dm.confusion_at_threshold(y, p, threshold=0.0)
    assert cm.tp == 2 and cm.fp == 2
    assert cm.tn == 0 and cm.fn == 0


def test_confusion_threshold_one_predicts_all_negative() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])
    cm = dm.confusion_at_threshold(y, p, threshold=1.0)
    assert cm.tp == 0 and cm.fp == 0
    assert cm.tn == 2 and cm.fn == 2


def test_confusion_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        dm.confusion_at_threshold([0, 1], [0.5, 0.5, 0.5])


def test_confusion_to_dict_round_trip() -> None:
    cm = dm.confusion_at_threshold([0, 1], [0.1, 0.9])
    payload = cm.as_dict()
    assert payload["tp"] == 1 and payload["tn"] == 1
    assert payload["correct"] == 2
    assert payload["accuracy"] == 1.0


def test_auc_perfect_separation() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert dm.auc_tie_safe(y, p) == 1.0


def test_auc_inverted_predictions() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert dm.auc_tie_safe(y, p) == 0.0


def test_auc_returns_half_on_constant_scores() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert dm.auc_tie_safe(y, p) == 0.5


def test_auc_returns_nan_when_only_one_class_present() -> None:
    y = np.array([1, 1, 1])
    p = np.array([0.1, 0.2, 0.3])
    assert math.isnan(dm.auc_tie_safe(y, p))


def test_auc_handles_ties_with_average_ranking() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.4, 0.4, 0.6, 0.6])  # half informative, half tied
    auc = dm.auc_tie_safe(y, p)
    assert 0.4 <= auc <= 0.6


def test_decision_class_table_uses_surface_quantiles_not_well_quantiles() -> (
    None
):
    surface_values = np.linspace(0.0, 1.0, 1001)  # uniform 0..1 cells
    rng = np.random.default_rng(0)
    well_scores = rng.uniform(0.0, 1.0, size=80)
    well_labels = (well_scores >= 0.5).astype(int)  # perfectly separable

    rows = dm.decision_class_table(
        surface_name="probability",
        surface_values=surface_values,
        well_scores=well_scores,
        well_labels=well_labels,
    )
    assert len(rows) == len(dm.DEFAULT_DECISION_LABELS)
    assert sum(r.n_wells for r in rows) == 80
    # Top 5% bin should contain only positives (intra-bin accuracy 100%)
    top_bin = rows[-1]
    assert top_bin.intuitive_pct_correct == pytest.approx(100.0)
    # Bottom 5% bin should contain only negatives
    bottom_bin = rows[0]
    assert bottom_bin.intuitive_pct_correct == pytest.approx(100.0)


def test_decision_class_table_validates_edge_label_count() -> None:
    with pytest.raises(ValueError):
        dm.decision_class_table(
            surface_name="probability",
            surface_values=[0.1, 0.5, 0.9],
            well_scores=[0.5],
            well_labels=[1],
            edges=[0.0, 0.5, 1.0],  # 3 edges
            labels=("low", "mid", "high"),  # 3 labels — should be 2
        )


def test_decision_class_table_uncertain_bin_uses_max_correct() -> None:
    # 4 wells in the uncertain (34–66 percentile) bin: 3 positive, 1 negative.
    # "Correct" should be 3 (max of pos vs neg) -> 75% intuitive correct.
    rng = np.random.default_rng(0)
    surface = np.linspace(0.0, 1.0, 1001)
    # Place 4 wells precisely in the 34–66 range.
    well_scores = np.array([0.40, 0.50, 0.55, 0.60])
    well_labels = np.array([1, 1, 1, 0])
    rows = dm.decision_class_table(
        surface_name="probability",
        surface_values=surface,
        well_scores=well_scores,
        well_labels=well_labels,
    )
    uncertain = next(r for r in rows if "Uncertain" in r.decision_class)
    assert uncertain.n_wells == 4
    assert uncertain.intuitive_pct_correct == pytest.approx(75.0)


def test_top_n_targeting_full_curve_endpoints() -> None:
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3])
    rows = dm.top_n_targeting(y, p)
    assert [r.n_picked for r in rows] == list(range(1, 7))
    # Top 3 by score: 0.9 (pos), 0.8 (pos), 0.7 (pos) → 3 hits
    assert rows[2].hits == 3
    # All 6 picked → all positives recovered
    assert rows[-1].hits == int(y.sum())
    # Oracle row at n=3: min(3, n_pos=3) = 3
    assert rows[2].max_possible_hits == 3
    # Random expectation at n=3: 3 * 3 / 6 = 1.5
    assert rows[2].random_expected_hits == pytest.approx(1.5)


def test_top_n_targeting_explicit_ns_clamps_to_total() -> None:
    y = np.array([1, 0, 1])
    p = np.array([0.5, 0.4, 0.6])
    rows = dm.top_n_targeting(y, p, ns=[1, 3, 99])
    assert rows[-1].n_picked == 3  # clamped to len(y)


def test_top_n_targeting_averages_boundary_ties_without_row_order_bias() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    permutation = np.array([1, 0, 3, 2])

    original = dm.top_n_targeting(y, scores, ns=[1, 3])
    reordered = dm.top_n_targeting(y[permutation], scores[permutation], ns=[1, 3])

    assert [row.hits for row in original] == pytest.approx([0.5, 1.5])
    assert [row.hits for row in reordered] == pytest.approx([0.5, 1.5])


def test_top_n_targeting_invalid_n_raises() -> None:
    with pytest.raises(ValueError):
        dm.top_n_targeting([1, 0], [0.5, 0.5], ns=[0])


@pytest.mark.parametrize("invalid_n", [1.5, True, "1"])
def test_top_n_targeting_rejects_noninteger_budgets(invalid_n: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        dm.top_n_targeting(
            [1, 0],
            [0.5, 0.5],
            ns=[invalid_n],  # type: ignore[list-item]
        )


def test_top_n_targeting_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        dm.top_n_targeting([1, 0], [0.5])


@pytest.mark.parametrize(
    "function,kwargs",
    [
        (dm.confusion_at_threshold, {}),
        (dm.auc_tie_safe, {}),
        (dm.top_n_targeting, {}),
    ],
)
def test_decision_metrics_reject_nonbinary_labels(function, kwargs) -> None:
    with pytest.raises(ValueError, match="binary 0/1"):
        function(
            np.array([0.0, 0.5, 1.0]), np.array([0.1, 0.5, 0.9]), **kwargs
        )


@pytest.mark.parametrize(
    "function",
    [dm.confusion_at_threshold, dm.auc_tie_safe, dm.top_n_targeting],
)
def test_decision_metrics_reject_nonfinite_scores(function) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(np.array([0, 1]), np.array([0.1, np.nan]))


def test_decision_class_table_rejects_invalid_quantile_edges() -> None:
    with pytest.raises(ValueError, match="edges"):
        dm.decision_class_table(
            surface_name="probability",
            surface_values=np.linspace(0.0, 1.0, 10),
            well_scores=np.array([0.2, 0.8]),
            well_labels=np.array([0, 1]),
            edges=(0.0, 0.8, 0.4, 1.0),
            labels=("low", "middle", "high"),
        )
