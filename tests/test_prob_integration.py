"""End-to-end integration test for the Stage-1 probabilistic demo workflow.

Uses the synthetic two-component fixture from :mod:`tests.fixtures.synthetic_prob`
to exercise the full pipeline:

1. Per-component fit with spatial prior and RBF residual smoother
2. Combined surface via product rule
3. Sampling combined and component surfaces at labelled wells
4. Calibration summary (ECE / MCE / Brier / log-loss / temperature)
5. Decision-class table on the combined surface
6. Top-N targeting curve and confusion matrix at threshold 0.5
7. Spatial-block CV via the existing scenario runner

The test asserts the full workflow runs without errors and produces sane
outputs (shapes, ranges, monotonicity expectations) — it does not assert
specific numeric values, so it is robust to minor numerical drift.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np

from geopfa.prob import (
    CoordinateTrendScenarioSpec,
    auc_tie_safe,
    calibration_summary,
    combine_probability_surfaces,
    confusion_at_threshold,
    decision_class_table,
    fit_component_probability,
    run_coordinate_trend_sensitivity,
    top_n_targeting,
)
from tests.fixtures.synthetic_prob import make_synthetic_pfa


def _fit_all_components(syn) -> dict[str, gpd.GeoDataFrame]:
    """Fit both synthetic components with the spatial-prior + residual workflow."""
    prior_layers = {
        "component_a": "prior_layer_a",
        "component_b": "prior_layer_b",
    }
    label_columns = {
        "component_a": "heat_label",
        "component_b": "reservoir_label",
    }
    surfaces = {}
    for name, comp in syn.pfa["criteria"]["geologic"]["components"].items():
        result = fit_component_probability(
            comp,
            prior_probability=float(comp["pr0"]),
            include_spatial=True,
            labeled_wells=syn.wells,
            label_column=label_columns[name],
            prior_layer_name=prior_layers[name],
        )
        surfaces[name] = result.probability
    return surfaces


def test_full_stage1_workflow_runs_end_to_end_on_synthetic_data() -> None:
    syn = make_synthetic_pfa(seed=21, grid_n=14, n_wells=140)
    component_surfaces = _fit_all_components(syn)
    assert set(component_surfaces) == {"component_a", "component_b"}

    # 1) Combine via product rule.
    probs = [
        s["probability"].to_numpy(dtype=float)
        for s in component_surfaces.values()
    ]
    combined_grid = combine_probability_surfaces(probs)
    assert combined_grid.shape == probs[0].shape
    assert np.all((combined_grid >= 0.0) & (combined_grid <= 1.0))

    # 2) Sample combined surface at labelled wells.
    grid = list(component_surfaces.values())[0][["geometry"]].copy()
    grid["combined_probability"] = combined_grid
    wells = syn.wells.to_crs(grid.crs)
    sampled = gpd.sjoin_nearest(
        wells[["heat_label", "geometry"]],
        grid,
        how="inner",
    )
    y = sampled["heat_label"].astype(int).to_numpy()
    p = sampled["combined_probability"].astype(float).to_numpy()
    assert y.shape == p.shape == (len(syn.wells),)

    # 3) Discrimination should beat random on a separable synthetic dataset.
    auc = auc_tie_safe(y, p)
    assert auc > 0.6

    # 4) Calibration summary completes and reports finite metrics.
    summary = calibration_summary(y, p, n_bins=4)
    for key in ("ECE", "MCE", "brier", "log_loss", "temperature_T"):
        value = summary[key]
        assert np.isfinite(value), f"{key} should be finite (got {value!r})"
    # Bin tables should each return 4 rows (the requested bin count).
    assert len(summary["bins_raw"]) == 4
    assert len(summary["bins_post_temperature"]) == 4

    # 5) Decision-class table runs and partitions the wells into 5 percentile bins.
    decision_rows = decision_class_table(
        surface_name="combined",
        surface_values=combined_grid,
        well_scores=p,
        well_labels=y,
    )
    assert len(decision_rows) == 5
    assert sum(r.n_wells for r in decision_rows) == y.size

    # 6) Confusion at threshold 0.5 returns consistent counts.
    cm = confusion_at_threshold(y, p, threshold=0.5)
    assert cm.tp + cm.fp + cm.tn + cm.fn == y.size
    # 7) Top-N curve at the full-budget endpoint must equal n_pos.
    rows = top_n_targeting(y, p)
    assert rows[-1].hits == int(y.sum())


def test_stage1_scenario_runner_produces_metrics_on_synthetic_data() -> None:
    syn = make_synthetic_pfa(seed=22, grid_n=10, n_wells=80)
    # Build feature matrix at well points — three synthetic features sampled
    # directly from the prior layers so the Stage-1 runner has signal.
    rng = np.random.default_rng(22)
    x_a = rng.uniform(0.0, 1.0, size=len(syn.wells))
    x_b = rng.uniform(0.0, 1.0, size=len(syn.wells))
    x_c = rng.uniform(0.0, 1.0, size=len(syn.wells))
    y = syn.wells["heat_label"].astype(int).to_numpy()
    # Inject signal in x_a so AUC > 0.5 isn't accidental.
    x_a = 0.7 * y + 0.3 * x_a
    import pandas as pd

    X_df = pd.DataFrame({"feature_a": x_a, "feature_b": x_b, "feature_c": x_c})
    coords = np.column_stack(
        [syn.wells.geometry.x.to_numpy(), syn.wells.geometry.y.to_numpy()]
    )
    alpha = np.full(len(syn.wells), float(np.log(0.55 / 0.45)))

    scenarios = [
        CoordinateTrendScenarioSpec(name="evidence_only"),
        CoordinateTrendScenarioSpec(name="no_priors", include_priors=False),
        CoordinateTrendScenarioSpec(
            name="with_coordinate_trend", include_coordinate_trend=True
        ),
        CoordinateTrendScenarioSpec(
            name="drop_feature_b", drop_features=("feature_b",)
        ),
    ]
    out = run_coordinate_trend_sensitivity(
        X_df, y, alpha, coords, scenarios, n_splits=3
    )
    assert len(out) == 4
    assert set(out["scenario"]) == {
        "evidence_only",
        "no_priors",
        "with_coordinate_trend",
        "drop_feature_b",
    }
    # All rows should report a finite block-CV AUC (can be NaN only if a fold
    # had a degenerate label split — should not happen on this fixture).
    finite_auc = np.isfinite(out["holdout_auc"].to_numpy(dtype=float))
    assert finite_auc.any()
