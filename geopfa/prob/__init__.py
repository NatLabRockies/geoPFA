"""Probabilistic workflow for geoPFA.

This package implements the hierarchical-component probabilistic model as a
standalone, config-driven workflow that users can choose instead of the
existing deterministic favorability calculation.

Target model (same structure at every level of fidelity)::

    logit p_c(s, z) = alpha_c(s, z) + Sigma_k beta_ck x_k(s, z) + u_c(s)
    p_resource(s, z) = product over c of p_c(s, z)

Modules
-------
* ``fitting`` — per-component logistic-regression fit with optional spatial
  prior, sparse-GP spatial residual field, and per-feature regularization.
* ``alpha`` — physics-informed prior offset builder (scalar / layer-logit /
  thermal-exceedance / multi-layer).
* ``data`` — generic raster sampling at labelled point locations.
* ``labels`` — config-driven labelled-well loading.
* ``play_types`` — play-type defaults registry for per-feature regularization.
* ``pu`` — non-negative Positive-Unlabeled logistic risk estimation.
* ``site_selection`` — optional finite-candidate preferential-sampling model.
* ``regions`` — multi-region label splitting for hierarchical pooling.
* ``scenario`` — reduced-form coordinate-trend sensitivity diagnostics.
* ``variogram`` — empirical variogram range estimation for auto block-size CV.
* ``calibration`` — ECE / MCE / Brier / log-loss, post-hoc calibration maps.
* ``decision_metrics`` — decision-class tables, top-N targeting, AUC,
  confusion matrices.
* ``config`` — strongly-typed ``ProbabilisticConfig`` dataclass tree.
* ``runner`` — top-level ``run_probabilistic`` entry point.
* ``io`` — output writers (GeoTIFF / Parquet / VTK / manifest).
* ``plotting`` — 2D + 3D plotter helpers (region-agnostic, style-configurable).
* ``spatial`` — spatial random-field GP wrapper over ``geopfa.extrapolation``.
* ``cv`` — spatial-block CV splitter (with variogram-based auto block-size).
* ``cv_runner`` — block-CV runner producing out-of-fold predictions.
* ``inference`` — backend-neutral fit result protocol; Bayesian spatial
  inference is provided only by the GBLK path.

Nothing in this package is region-specific. Study-specific configuration and
visualization live only in the output-free notebooks under ``examples/``.
"""

from .alpha import AlphaCResult, build_alpha_c, write_alpha_provenance
from .calibration import (
    CalibrationBin,
    CalibrationMap,
    brier_score,
    calibration_summary,
    equal_frequency_reliability_table,
    expected_calibration_error,
    fit_posthoc_calibration,
    fit_temperature,
    log_loss,
    maximum_calibration_error,
    wilson_ci,
)
from .inference import ComponentFitResult, Fitter
from .inference.sequential import SequentialFitter
from .pu import NNPULogitResult, fit_nnpu_logistic
from .site_selection import (
    ConfiguredSiteSelectionResult,
    SiteSelectionFit,
    fit_site_selection_model,
    run_site_selection_analysis,
    site_selection_sensitivity,
)
from .regions import (
    RegionLabels,
    check_region_label_coverage,
    combine_region_beta_summaries,
    split_by_region,
)
from .variogram import (
    empirical_variogram,
    estimate_variogram_range,
    recommend_block_size_km,
)
from .config import (
    AlphaModeConfig,
    CalibrationConfig,
    CombinationConfig,
    CrossValidationConfig,
    EvidenceConfig,
    GridConfig,
    InferenceConfig,
    LabelsConfig,
    OutputsConfig,
    ProbabilisticConfig,
    RegularizationConfig,
    ScenarioConfig,
    SiteSelectionConfig,
    SpatialFieldConfig,
    load_probabilistic_config,
)
from .cv import SpatialBlockKFold, spatial_block_cv
from .cv_runner import CVResult, component_oof_predictions, run_block_cv
from .data import sample_evidence_at_wells
from .decision_metrics import (
    ConfusionAtThreshold,
    DecisionClassRow,
    TopNRow,
    auc_tie_safe,
    confusion_at_threshold,
    decision_class_table,
    top_n_targeting,
)
from .fitting import (
    ComponentProbability,
    combine_probability_surfaces,
    fit_component_probability,
)
from .forward import (
    FrozenGBLKForwardResult,
    FrozenGBLKForwardState,
    evaluate_frozen_gblk,
    load_frozen_gblk_forward_state,
    save_frozen_gblk_forward_state,
)
from .fit_dispatch import build_fit_kwargs, fit_component_from_config
from .io import (
    PersistedPosteriorDrawState,
    PosteriorDrawBlockWriter,
    PosteriorDrawSummary,
    load_posterior_draw_state,
    verify_manifest,
    write_geotiff_outputs,
    write_manifest,
    write_parquet_outputs,
    write_probability_outputs,
    write_vtk_outputs,
    verify_posterior_draw_bundle,
)
from .labels import (
    LoadedLabels,
    available_components,
    component_labels,
    load_labels,
)
from .pfa_grid import (
    PFAGridAdapter,
    component_layers,
    component_names,
    extract_grid_extent,
    iter_components,
    layer_data_column,
    layer_model_gdf,
    validate_pfa_for_probabilistic,
)
from .play_types import (
    PLAY_TYPE_REGISTRY,
    available_play_types,
    play_type_defaults,
)
from .plotting import (
    PlotStyle,
    plot_component_panel,
    plot_confusion_matrix,
    plot_decision_class_bar,
    plot_depth_slices,
    plot_reliability_diagram,
    plot_top_n_curve,
    plot_vertical_cross_section,
    plot_well_overlay,
)
from .runner import (
    ProbabilisticResult,
    run_probabilistic,
    run_probabilistic_pfa,
)
from .scenario import (
    CoordinateTrendScenarioSpec,
    run_coordinate_trend_sensitivity,
    spatial_block_holdout_mask,
)
from .spatial import SpatialFieldResult, fit_spatial_field_gp

__all__ = [
    "PLAY_TYPE_REGISTRY",
    "AlphaCResult",
    "AlphaModeConfig",
    "CVResult",
    "CalibrationBin",
    "CalibrationConfig",
    "CalibrationMap",
    "CombinationConfig",
    "ComponentFitResult",
    "ComponentProbability",
    "ConfiguredSiteSelectionResult",
    "ConfusionAtThreshold",
    "CoordinateTrendScenarioSpec",
    "CrossValidationConfig",
    "DecisionClassRow",
    "EvidenceConfig",
    "Fitter",
    "FrozenGBLKForwardResult",
    "FrozenGBLKForwardState",
    "GridConfig",
    "InferenceConfig",
    "LabelsConfig",
    "LoadedLabels",
    "NNPULogitResult",
    "OutputsConfig",
    "PFAGridAdapter",
    "PersistedPosteriorDrawState",
    "PlotStyle",
    "PosteriorDrawBlockWriter",
    "PosteriorDrawSummary",
    "ProbabilisticConfig",
    "ProbabilisticResult",
    "RegionLabels",
    "RegularizationConfig",
    "ScenarioConfig",
    "SequentialFitter",
    "SiteSelectionConfig",
    "SiteSelectionFit",
    "SpatialBlockKFold",
    "SpatialFieldConfig",
    "SpatialFieldResult",
    "TopNRow",
    "auc_tie_safe",
    "available_components",
    "available_play_types",
    "brier_score",
    "build_alpha_c",
    "build_fit_kwargs",
    "calibration_summary",
    "check_region_label_coverage",
    "combine_probability_surfaces",
    "combine_region_beta_summaries",
    "component_labels",
    "component_layers",
    "component_names",
    "component_oof_predictions",
    "confusion_at_threshold",
    "decision_class_table",
    "empirical_variogram",
    "equal_frequency_reliability_table",
    "estimate_variogram_range",
    "evaluate_frozen_gblk",
    "expected_calibration_error",
    "extract_grid_extent",
    "fit_component_from_config",
    "fit_component_probability",
    "fit_nnpu_logistic",
    "fit_posthoc_calibration",
    "fit_site_selection_model",
    "fit_spatial_field_gp",
    "fit_temperature",
    "iter_components",
    "layer_data_column",
    "layer_model_gdf",
    "load_frozen_gblk_forward_state",
    "load_labels",
    "load_posterior_draw_state",
    "load_probabilistic_config",
    "log_loss",
    "maximum_calibration_error",
    "play_type_defaults",
    "plot_component_panel",
    "plot_confusion_matrix",
    "plot_decision_class_bar",
    "plot_depth_slices",
    "plot_reliability_diagram",
    "plot_top_n_curve",
    "plot_vertical_cross_section",
    "plot_well_overlay",
    "recommend_block_size_km",
    "run_block_cv",
    "run_coordinate_trend_sensitivity",
    "run_probabilistic",
    "run_probabilistic_pfa",
    "run_site_selection_analysis",
    "sample_evidence_at_wells",
    "save_frozen_gblk_forward_state",
    "site_selection_sensitivity",
    "spatial_block_cv",
    "spatial_block_holdout_mask",
    "split_by_region",
    "top_n_targeting",
    "validate_pfa_for_probabilistic",
    "verify_manifest",
    "verify_posterior_draw_bundle",
    "wilson_ci",
    "write_alpha_provenance",
    "write_geotiff_outputs",
    "write_manifest",
    "write_parquet_outputs",
    "write_probability_outputs",
    "write_vtk_outputs",
]
