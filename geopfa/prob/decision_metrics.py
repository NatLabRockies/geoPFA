"""Decision-relevant metrics for the Stage-1 probabilistic demo.

These helpers translate predicted-probability surfaces into decision summaries
that non-experts can interpret directly:

* threshold-based confusion matrix (TP / FP / TN / FN) and the standard rates
* AUC (tie-safe Mann-Whitney rank statistic)
* decision-class binning by surface-percentile, with intra-bin accuracy
* top-N targeting curve

All functions operate on plain ``numpy`` arrays so they are framework-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy.stats import rankdata


def _validated_binary_scores(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite vectors without coercing invalid labels."""
    if np.iscomplexobj(y) or np.iscomplexobj(scores):
        raise ValueError("y and scores must contain finite real values")
    try:
        y_values = np.asarray(y, dtype=float)
        score_values = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y and scores must contain numeric values") from exc
    if y_values.ndim != 1 or score_values.ndim != 1:
        raise ValueError("y and scores must be one-dimensional")
    if y_values.shape != score_values.shape:
        raise ValueError("y and scores must have the same shape")
    if not np.all(np.isfinite(y_values)) or not np.all(
        np.isfinite(score_values)
    ):
        raise ValueError("y and scores must contain only finite values")
    if not np.all(np.isin(y_values, (0.0, 1.0))):
        raise ValueError("y must contain binary 0/1 labels")
    return y_values.astype(int), score_values


@dataclass(frozen=True)
class ConfusionAtThreshold:
    """Confusion matrix and derived rates at a fixed probability threshold."""

    threshold: float
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float

    @property
    def correct(self) -> int:
        """Number of correctly classified observations (TP + TN)."""
        return self.tp + self.tn

    @property
    def incorrect(self) -> int:
        """Number of misclassified observations (FP + FN)."""
        return self.fp + self.fn

    def as_dict(self) -> dict:
        """Return a plain-dict representation of the confusion matrix."""
        return {
            "threshold": self.threshold,
            "n": self.n,
            "correct": self.correct,
            "incorrect": self.incorrect,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
        }


def confusion_at_threshold(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> ConfusionAtThreshold:
    """Confusion matrix at a fixed threshold, plus accuracy / precision / recall / specificity."""
    y_arr, s_arr = _validated_binary_scores(y, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    pred = (s_arr >= threshold).astype(int)
    tp = int(((pred == 1) & (y_arr == 1)).sum())
    fp = int(((pred == 1) & (y_arr == 0)).sum())
    tn = int(((pred == 0) & (y_arr == 0)).sum())
    fn = int(((pred == 0) & (y_arr == 1)).sum())
    n = int(y_arr.size)
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return ConfusionAtThreshold(
        threshold=float(threshold),
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
    )


def auc_tie_safe(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float:
    """Mann-Whitney AUC implemented from ``scipy.stats.rankdata``.

    Uses the average-rank tie-handling rule, which is correct when many
    predictions take identical values. Returns NaN when only one class is
    present in ``y``; returns 0.5 when all scores are identical.
    """
    y_arr, s_arr = _validated_binary_scores(y, scores)
    n_classes_required = 2
    if y_arr.size == 0 or np.unique(y_arr).size < n_classes_required:
        return float("nan")
    if np.nanstd(s_arr) == 0:
        return 0.5
    ranks = rankdata(s_arr, method="average")
    pos = y_arr == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg),
    )


# ---------------------------------------------------------------------------
# Decision-class binning by surface percentile
# ---------------------------------------------------------------------------

DEFAULT_DECISION_EDGES: tuple[float, ...] = (0.0, 0.05, 0.34, 0.66, 0.95, 1.0)
DEFAULT_DECISION_LABELS: tuple[str, ...] = (
    "Confident negative (\u22645%)",
    "Leaning negative (5\u201334%)",
    "Uncertain (34\u201366%)",
    "Leaning positive (66\u201395%)",
    "Confident positive (\u226595%)",
)


@dataclass(frozen=True)
class DecisionClassRow:
    """One row of the decision-class table for one surface."""

    surface: str
    decision_class: str
    n_wells: int
    actual_positives: int
    actual_positive_rate: float
    intuitive_pct_correct: float
    bin_value_lo: float
    bin_value_hi: float

    def as_dict(self) -> dict:
        """Return a plain-dict representation of the decision-class row."""
        return {
            "surface": self.surface,
            "decision_class": self.decision_class,
            "n_wells": self.n_wells,
            "actual_positives": self.actual_positives,
            "actual_positive_rate": self.actual_positive_rate,
            "intuitive_pct_correct": self.intuitive_pct_correct,
            "bin_value_lo": self.bin_value_lo,
            "bin_value_hi": self.bin_value_hi,
        }


def _surface_quantile_thresholds(
    surface_values: Sequence[float] | np.ndarray,
    edges: Sequence[float],
) -> list[float]:
    """Quantile cut points of a surface distribution at the requested edges."""
    v = np.asarray(surface_values, dtype=float)
    if v.ndim != 1:
        raise ValueError("surface_values must be one-dimensional")
    if not np.all(np.isfinite(v)):
        raise ValueError("surface_values must contain only finite values")
    if v.size == 0:
        return [float("nan")] * len(edges)
    return [float(np.quantile(v, q)) for q in edges]


def decision_class_table(  # noqa: PLR0913
    *,
    surface_name: str,
    surface_values: Sequence[float] | np.ndarray,
    well_scores: Sequence[float] | np.ndarray,
    well_labels: Sequence[int] | np.ndarray,
    edges: Sequence[float] = DEFAULT_DECISION_EDGES,
    labels: Sequence[str] = DEFAULT_DECISION_LABELS,
) -> list[DecisionClassRow]:
    """Bin labelled wells by where their score falls in the *full surface* percentile.

    The cut points come from quantiles of ``surface_values`` (the entire grid),
    not the labelled-well sample, so a well in the "Top 5%" bin really is in
    the top 5% of the surface globally. Within each bin we report the well
    count, the number of actual positives, and an "intuitive % correct"
    interpretation:

    * upper bins (positive-leaning) → correct = positive count
    * lower bins (negative-leaning) → correct = negative count
    * the middle uncertain bin → correct = max(positives, negatives) so it
      does not unfairly penalise an honest "uncertain" prediction

    ``edges`` must have one more element than ``labels``.
    """
    if len(edges) != len(labels) + 1:
        raise ValueError("len(edges) must equal len(labels) + 1")
    edge_values = np.asarray(edges, dtype=float)
    if (
        not np.all(np.isfinite(edge_values))
        or edge_values[0] != 0.0
        or edge_values[-1] != 1.0
        or np.any(np.diff(edge_values) <= 0.0)
    ):
        raise ValueError(
            "edges must be finite, strictly increasing quantiles from 0 to 1"
        )
    y_arr, s_arr = _validated_binary_scores(well_labels, well_scores)

    thresholds = _surface_quantile_thresholds(surface_values, edges)
    rows: list[DecisionClassRow] = []
    last_idx = len(labels) - 1
    for i, label in enumerate(labels):
        lo, hi = thresholds[i], thresholds[i + 1]
        if i == last_idx:
            in_bin = (s_arr >= lo) & (s_arr <= hi)
        else:
            in_bin = (s_arr >= lo) & (s_arr < hi)
        n = int(in_bin.sum())
        pos = int(y_arr[in_bin].sum()) if n else 0
        if i > last_idx // 2:  # upper half: positive-leaning
            correct = pos
        elif i < last_idx // 2:  # lower half: negative-leaning
            correct = n - pos
        else:  # uncertain middle
            correct = max(pos, n - pos)
        rows.append(
            DecisionClassRow(
                surface=surface_name,
                decision_class=label,
                n_wells=n,
                actual_positives=pos,
                actual_positive_rate=(pos / n) if n else float("nan"),
                intuitive_pct_correct=(correct / n * 100.0)
                if n
                else float("nan"),
                bin_value_lo=float(lo),
                bin_value_hi=float(hi),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Top-N targeting curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopNRow:
    """One row of the top-N targeting curve."""

    n_picked: int
    hits: float
    random_expected_hits: float
    max_possible_hits: int

    def as_dict(self) -> dict:
        """Return a plain-dict representation of the top-N row."""
        return {
            "n_picked": self.n_picked,
            "hits": self.hits,
            "random_expected_hits": self.random_expected_hits,
            "max_possible_hits": self.max_possible_hits,
        }


def _expected_cumulative_tie_hits(
    labels: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    """Return order-invariant expected cumulative hits across score ties."""
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    cumulative = np.empty(labels.size, dtype=float)
    group_start = 0
    hits_above = 0.0
    while group_start < labels.size:
        group_end = group_start + 1
        while (
            group_end < labels.size
            and scores_sorted[group_end] == scores_sorted[group_start]
        ):
            group_end += 1
        group_size = group_end - group_start
        group_hits = float(labels_sorted[group_start:group_end].sum())
        within_group = np.arange(1, group_size + 1, dtype=float)
        cumulative[group_start:group_end] = (
            hits_above + within_group * group_hits / group_size
        )
        hits_above += group_hits
        group_start = group_end
    return cumulative


def top_n_targeting(
    y: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    ns: Sequence[int] | None = None,
) -> list[TopNRow]:
    """Return top-N hits, averaging fairly over a tie at the budget boundary.

    If ``ns`` is None, returns a row for every ``n in range(1, len(y)+1)``.
    Also reports the perfect-ranking oracle (``min(n, n_pos)``) and the
    random-baseline expectation (``n * n_pos / n_total``) for comparison.
    ``hits`` is the expected number of positives when a score tie crosses the
    selection boundary and tied candidates are sampled uniformly. It remains an
    integer-valued float when the boundary is untied.
    """
    y_arr, s_arr = _validated_binary_scores(y, scores)
    n_total = int(y_arr.size)
    n_pos = int(y_arr.sum())
    ns_iter = list(ns) if ns is not None else list(range(1, n_total + 1))

    expected_cumulative_hits = _expected_cumulative_tie_hits(y_arr, s_arr)

    rows: list[TopNRow] = []
    for n_pick in ns_iter:
        if (
            isinstance(n_pick, bool | np.bool_)
            or not isinstance(n_pick, Integral)
            or n_pick < 1
        ):
            raise ValueError("ns entries must be positive integers")
        n_pick_clipped = min(int(n_pick), n_total)
        rows.append(
            TopNRow(
                n_picked=n_pick_clipped,
                hits=(
                    float(expected_cumulative_hits[n_pick_clipped - 1])
                    if n_total
                    else 0.0
                ),
                random_expected_hits=(n_pick_clipped * n_pos / n_total)
                if n_total
                else 0.0,
                max_possible_hits=min(n_pick_clipped, n_pos),
            ),
        )
    return rows


__all__ = [
    "DEFAULT_DECISION_EDGES",
    "DEFAULT_DECISION_LABELS",
    "ConfusionAtThreshold",
    "DecisionClassRow",
    "TopNRow",
    "auc_tie_safe",
    "confusion_at_threshold",
    "decision_class_table",
    "top_n_targeting",
]
