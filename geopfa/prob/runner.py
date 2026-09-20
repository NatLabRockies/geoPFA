"""Top-level probabilistic-method entry point: ``run_probabilistic(pfa, config)``.

Translates a :class:`~geopfa.prob.config.ProbabilisticConfig` into
:func:`~geopfa.prob.fitting.fit_component_probability` calls, combines per-component surfaces via
the configured rule, runs scenario ablations when requested, and writes the
configured outputs.

The runner is dimension-agnostic at the API surface and executes both 2D and
3D PFA dictionaries. Three-dimensional coordinates are retained through grid
alignment, spatial fitting, blocked validation, and output generation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from geopfa.exceptions import GEOPFAValueError

from .alpha import AlphaCResult, build_alpha_c, write_alpha_provenance
from .calibration import (
    CalibrationMap,
    brier_score,
    equal_frequency_reliability_table,
    expected_calibration_error,
    fit_posthoc_calibration,
    log_loss,
    maximum_calibration_error,
)
from .config import ProbabilisticConfig, ScenarioConfig
from .cv_runner import CVResult, run_block_cv
from .decision_metrics import decision_class_table, top_n_targeting
from .fitting import (
    ComponentProbability,
    combine_probability_surfaces,
    fit_component_probability,
)
from .fit_dispatch import build_fit_kwargs
from .io import (
    _begin_streamed_run_resume,
    _probabilistic_implementation_hash,
    validate_output_namespace,
    write_manifest,
    write_probability_outputs,
)
from .labels import LoadedLabels, load_labels
from .pfa_grid import PFAGridAdapter, validate_declared_components
from .predictive_stacking import PredictiveStackingResult


@dataclass(frozen=True)
class ProbabilisticResult:
    """Output of a :func:`run_probabilistic` invocation.

    Attributes
    ----------
    components
        ``{component_name: ComponentProbability}`` for each fitted
        component (raw probabilities).
    combined
        Grid GeoDataFrame holding the combined surface with a
        ``probability`` column. Empty when no components were fitted.
    calibrated_components
        ``{component_name: GeoDataFrame}`` of calibrated probabilities,
        produced when ``config.calibration.method != "none"``. Empty
        otherwise.
    calibration_maps
        ``{component_name: CalibrationMap}`` for each component that
        received a calibration fit.
    cv
        :class:`~geopfa.prob.cv_runner.CVResult` from the block-CV run, when calibration is
        configured. Empty when calibration is off.
    scenarios
        ``{scenario_name: {component_name: ComponentProbability}}``
        when scenarios were configured; ``{}`` otherwise.
    config
        The :class:`~geopfa.prob.config.ProbabilisticConfig` that drove this run (carried for
        downstream consumers).
    skipped
        ``True`` if the run short-circuited because ``config.enabled`` was
        ``False``.
    """

    components: dict[str, ComponentProbability] = field(default_factory=dict)
    combined: gpd.GeoDataFrame = field(default_factory=gpd.GeoDataFrame)
    calibrated_components: dict[str, gpd.GeoDataFrame] = field(
        default_factory=dict
    )
    calibration_maps: dict[str, CalibrationMap] = field(default_factory=dict)
    cv: CVResult | None = None
    scenarios: dict[str, dict[str, ComponentProbability]] = field(
        default_factory=dict,
    )
    site_selection: dict[str, Any] = field(default_factory=dict)
    component_probability_draws: dict[str, np.ndarray] = field(
        default_factory=dict
    )
    combined_probability_draws: np.ndarray | None = None
    predictive_stacking: dict[str, PredictiveStackingResult] = field(
        default_factory=dict
    )
    posterior_draw_index: Path | None = None
    config: ProbabilisticConfig | None = None
    skipped: bool = False


def _run_site_selection(config: ProbabilisticConfig) -> dict[str, Any]:
    """Run the optional, separate candidate-frame sensitivity analysis."""
    if config.site_selection.mode == "off":
        return {}
    from .site_selection import run_site_selection_analysis  # noqa: PLC0415

    return {
        component: run_site_selection_analysis(
            config.site_selection,
            outcome_column=outcome_column,
        )
        for component, outcome_column in config.labels.label_columns.items()
        if config.labels.observation_model_for(component).family == "bernoulli"
    }


def _fit_component(
    adapter: PFAGridAdapter,
    component: str,
    cfg: ProbabilisticConfig,
    labels: LoadedLabels | None,
    alpha_result: AlphaCResult,
) -> ComponentProbability | None:
    """Fit one component, routing to the configured inference backend."""
    alpha_config = cfg.alpha.get(component)
    if alpha_config is None:
        warnings.warn(
            f"component {component!r} has no entry in alpha; "
            "skipping (configure an alpha mode to include it in the run)",
            UserWarning,
            stacklevel=3,
        )
        return None
    if (
        not alpha_config.force_prior_predictive
        and component not in cfg.labels.label_columns
    ):
        warnings.warn(
            f"component {component!r} has no entry in labels.label_columns; "
            "skipping (configure a label column to include it in the run)",
            UserWarning,
            stacklevel=3,
        )
        return None
    if not alpha_config.force_prior_predictive and labels is None:
        raise RuntimeError(
            "data-informed sequential fit reached dispatch without loaded labels"
        )

    component_data = adapter.component_data(component)

    # The componentwise path is the explicit sequential estimator. Bayesian
    # inference is available only through the joint GBLK runner.
    kwargs = build_fit_kwargs(
        alpha_config=alpha_config,
        evidence_config=cfg.evidence,
        spatial_field_config=cfg.spatial_field,
        pu_mode=cfg.labels.pu_mode,
        pu_class_prior=cfg.labels.class_prior_for(component),
        min_wells=cfg.labels.min_wells_for_fit,
    )
    extra_excluded = set(alpha_result.excluded_layer_names) | set(
        kwargs.get("excluded_layer_names", ())
    )
    kwargs["excluded_layer_names"] = tuple(sorted(extra_excluded))
    kwargs["alpha_offset"] = alpha_result.grid_offset
    kwargs["prior_layer_name"] = None

    return fit_component_probability(
        component_data,
        labeled_wells=labels.gdf if labels is not None else None,
        label_column=cfg.labels.label_columns.get(component),
        **kwargs,
    )


def _validate_probability_grid(
    reference: gpd.GeoDataFrame,
    candidate: gpd.GeoDataFrame,
    *,
    context: str,
) -> None:
    """Require exact CRS, geometry, and row-order agreement."""
    if "probability" not in candidate.columns:
        raise GEOPFAValueError(f"{context} lacks a probability column")
    for name, grid in (("canonical", reference), ("candidate", candidate)):
        geometry = grid.geometry
        if geometry.isna().any() or geometry.is_empty.any():
            raise GEOPFAValueError(
                f"{context} {name} grid contains missing or empty geometry"
            )
    if reference.crs != candidate.crs:
        raise GEOPFAValueError(
            f"{context} CRS differs from the canonical grid"
        )
    reference_wkb = reference.geometry.to_wkb().to_numpy(dtype=object)
    candidate_wkb = candidate.geometry.to_wkb().to_numpy(dtype=object)
    if not np.array_equal(reference_wkb, candidate_wkb):
        raise GEOPFAValueError(
            f"{context} differs from the canonical grid order"
        )


def _validated_probability_arrays(
    surfaces: Mapping[str, gpd.GeoDataFrame],
    *,
    context: str,
) -> tuple[gpd.GeoDataFrame, list[np.ndarray]]:
    """Return aligned arrays only after validating one exact common grid."""
    if not surfaces:
        return gpd.GeoDataFrame(), []
    reference = next(iter(surfaces.values()))
    arrays: list[np.ndarray] = []
    for name, surface in surfaces.items():
        _validate_probability_grid(
            reference,
            surface,
            context=f"{context} component {name!r}",
        )
        arrays.append(surface["probability"].astype(float).to_numpy())
    return reference[[reference.geometry.name]].copy(), arrays


def _validate_result_surface_grids(
    components: Mapping[str, ComponentProbability],
    combined: gpd.GeoDataFrame,
    *,
    context: str,
) -> None:
    """Validate component and paired-combination grids without recomputing draws."""
    surfaces = {
        name: component.probability for name, component in components.items()
    }
    reference, _ = _validated_probability_arrays(surfaces, context=context)
    if len(combined) > 0:
        if len(reference) == 0:
            raise GEOPFAValueError(
                f"{context} has a combined surface but no component surfaces"
            )
        _validate_probability_grid(
            reference,
            combined,
            context=f"{context} combined surface",
        )


def _combine_components(
    components: dict[str, ComponentProbability],
) -> gpd.GeoDataFrame:
    """Multiply component probabilities on their shared canonical grid."""
    if not components:
        return gpd.GeoDataFrame()
    base_gdf, component_arrays = _validated_probability_arrays(
        {name: surface.probability for name, surface in components.items()},
        context="component combination",
    )
    base_gdf["probability"] = combine_probability_surfaces(component_arrays)
    return base_gdf


def _validate_output_format_request(config: ProbabilisticConfig) -> None:
    """Reject contradictory output toggles before model fitting begins."""
    requested_spatial = tuple(
        fmt for fmt in config.outputs.format if fmt in {"geotiff", "vtk"}
    )
    if requested_spatial and not config.outputs.probability_rasters:
        requested = ", ".join(requested_spatial)
        raise GEOPFAValueError(
            "outputs.probability_rasters=false conflicts with requested "
            f"spatial format(s): {requested}"
        )


def _write_configured_probability_outputs(
    components: dict[str, ComponentProbability],
    combined: gpd.GeoDataFrame,
    config: ProbabilisticConfig,
    output_dir: Path,
) -> None:
    """Write tabular and spatial probability products under output toggles."""
    _validate_output_format_request(config)
    if not config.outputs.format:
        return
    all_surfaces: dict[str, gpd.GeoDataFrame] = {
        name: surface.probability for name, surface in components.items()
    }
    if len(combined) > 0:
        all_surfaces["combined"] = combined
    write_probability_outputs(
        all_surfaces,
        output_dir,
        formats=config.outputs.format,
        include_uncertainty=config.outputs.uncertainty_rasters,
    )


def _write_calibration_metrics(
    cv_result: CVResult, config: ProbabilisticConfig
) -> Path:
    """Write per-component calibration metrics and reliability rows."""
    import json  # noqa: PLC0415

    per_component: dict[str, dict[str, Any]] = {}
    for name, oof in cv_result.oof.items():
        if len(oof["p"]) == 0:
            continue
        rows = equal_frequency_reliability_table(
            oof["y"], oof["p"], n_bins=config.calibration.n_bins
        )
        per_component[name] = {
            "n": len(oof["p"]),
            "ECE": float(expected_calibration_error(rows)),
            "MCE": float(maximum_calibration_error(rows)),
            "brier": float(brier_score(oof["y"], oof["p"])),
            "log_loss": float(log_loss(oof["y"], oof["p"])),
            "reliability": [row.as_dict() for row in rows],
            "aggregate_metrics": cv_result.aggregate_metrics.get(name, {}),
        }
    payload = {
        "method": config.calibration.method,
        "n_bins": config.calibration.n_bins,
        "per_component": per_component,
    }
    out_path = config.output_dir / "calibration_metrics.json"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    # Also write a human-readable Markdown summary
    _write_calibration_report_md(per_component, config)
    return out_path


def _json_ready(value: Any) -> Any:
    """Convert nested diagnostics to strict JSON values without fake numbers."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer | np.floating):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_decision_metrics(
    cv_result: CVResult,
    components: dict[str, ComponentProbability],
    config: ProbabilisticConfig,
) -> Path:
    """Write decision summaries based only on raw spatial OOF predictions."""
    import json  # noqa: PLC0415

    per_component: dict[str, dict[str, Any]] = {}
    for name, oof in cv_result.oof.items():
        if name not in components or len(oof["p"]) == 0:
            continue
        surface_probability = (
            components[name].probability["probability"].to_numpy(dtype=float)
        )
        decision_rows = decision_class_table(
            surface_name=name,
            surface_values=surface_probability,
            well_scores=oof["p"],
            well_labels=oof["y"],
        )
        targeting_rows = top_n_targeting(oof["y"], oof["p"])
        per_component[name] = {
            "prediction_source": "raw_spatial_oof",
            "n": len(oof["p"]),
            "decision_classes": [row.as_dict() for row in decision_rows],
            "top_n_targeting": [row.as_dict() for row in targeting_rows],
        }
    payload = {
        "calibration_applied_to_decisions": False,
        "per_component": per_component,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.output_dir / "decision_metrics.json"
    out_path.write_text(
        json.dumps(
            _json_ready(payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return out_path


def _write_calibration_report_md(
    per_component: dict[str, dict[str, Any]], config: ProbabilisticConfig
) -> None:
    """Write a plain-text Markdown calibration summary alongside the JSON."""
    lines = [
        "# Calibration diagnostics",
        "",
        f"- Method: `{config.calibration.method}`",
        f"- Bins: {config.calibration.n_bins} equal-frequency bins",
        f"- Fit on: {config.calibration.fit_on}",
        "",
    ]
    for name, m in per_component.items():
        agg = m.get("aggregate_metrics", {})
        lines += [
            f"## {name}",
            "",
            f"- n: {m['n']}",
            f"- ECE: {m['ECE']:.4f}",
            f"- MCE: {m['MCE']:.4f}",
            f"- Brier score: {m['brier']:.4f}",
            f"- Log-loss: {m['log_loss']:.4f}",
        ]
        if agg:
            lines += [
                f"- Block-CV AUC: {agg.get('auc', float('nan')):.4f}",
                f"- Block-CV Brier: {agg.get('brier', float('nan')):.4f}",
            ]
        lines.append("")
    md_path = config.output_dir / "calibration_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _apply_scenario(
    cfg: ProbabilisticConfig, scenario: ScenarioConfig
) -> ProbabilisticConfig:
    """Return a new config with the given scenario's overrides applied."""
    if not scenario.include_priors:
        gaussian = sorted(
            name
            for name in cfg.alpha
            if (
                (observation := cfg.labels.observation_models.get(name))
                is not None
                and observation.family == "gaussian"
            )
        )
        if gaussian:
            raise GEOPFAValueError(
                "Gaussian scenario include_priors=false is undefined because "
                "the physical response mean and uncertainty are supplied by "
                "the prior; offending components: " + ", ".join(gaussian)
            )
    new_alpha = {}
    for name, alpha in cfg.alpha.items():
        if scenario.include_priors:
            new_alpha[name] = alpha
        else:
            # A neutral Bernoulli offset removes alpha without changing whether
            # the component is fitted or remains prior predictive.
            new_alpha[name] = type(alpha)(
                mode="scalar",
                scalar_fallback_pr0=0.5,
                force_prior_predictive=alpha.force_prior_predictive,
                use_evidence_prior=alpha.use_evidence_prior,
            )
    new_spatial = replace(cfg.spatial_field, enabled=scenario.include_spatial)
    new_exclude = tuple(
        sorted(
            set(cfg.evidence.exclude_layers)
            | set(scenario.drop_layers)
            | {
                layer_name
                for alpha in cfg.alpha.values()
                for layer_name in (
                    ((alpha.layer,) if alpha.layer is not None else ())
                    + alpha.layers
                )
                if not scenario.include_priors
            }
        )
    )
    new_evidence = replace(cfg.evidence, exclude_layers=new_exclude)
    new_outputs = (
        cfg.outputs
        if cfg.outputs.scenarios
        else replace(cfg.outputs, posterior_draw_blocks=False)
    )
    return type(cfg)(
        enabled=cfg.enabled,
        output_dir=cfg.output_dir,
        dimensions=cfg.dimensions,
        grid=cfg.grid,
        labels=cfg.labels,
        alpha=new_alpha,
        evidence=new_evidence,
        spatial_field=new_spatial,
        inference=cfg.inference,
        calibration=cfg.calibration,
        cross_validation=cfg.cross_validation,
        combination=cfg.combination,
        scenarios=(),
        outputs=new_outputs,
        site_selection=cfg.site_selection,
    )


def _validate_evidence_layer_references(
    adapter: PFAGridAdapter,
    cfg: ProbabilisticConfig,
) -> None:
    """Fail when an evidence or ablation layer name cannot affect the run."""
    declared_components = set(cfg.labels.label_columns) | set(cfg.alpha)
    available = {
        layer
        for component in adapter.components()
        if component in declared_components
        for layer in adapter.layers(component)
    }
    references: dict[str, set[str]] = {
        "evidence.include_layers": set(cfg.evidence.include_layers or ()),
        "evidence.exclude_layers": set(cfg.evidence.exclude_layers),
    }
    references.update(
        {
            f"scenario {scenario.name!r} drop_layers": set(
                scenario.drop_layers
            )
            for scenario in cfg.scenarios
        }
    )
    for context, names in references.items():
        unknown = names - available
        if unknown:
            raise GEOPFAValueError(
                f"{context} references layer(s) absent from the declared "
                f"PFA components: {', '.join(sorted(unknown))}"
            )


def _build_alphas(
    adapter: PFAGridAdapter,
    config: ProbabilisticConfig,
) -> dict[str, AlphaCResult]:
    """Build every configured component prior on its prediction grid."""
    return {
        name: build_alpha_c(
            adapter.component_data(name),
            config.alpha[name],
            grid_gdf=adapter.pr_norm(name),
        )
        for name in adapter.components()
        if name in config.alpha
    }


def run_probabilistic(  # noqa: PLR0912, PLR0914, PLR0915
    pfa: dict,
    config: ProbabilisticConfig,
    *,
    criteria: str = "geologic",
    input_artifacts: Mapping[str, str | Path] | None = None,
) -> ProbabilisticResult:
    """Execute the probabilistic method end-to-end.

    Parameters
    ----------
    pfa
        The geoPFA dict (output of the preprocessing pipeline).
    config
        Validated :class:`~geopfa.prob.config.ProbabilisticConfig` (e.g. from
        :func:`~geopfa.prob.config.load_probabilistic_config`).
    criteria
        Criteria key to operate on. Defaults to ``"geologic"``.
    input_artifacts
        Optional named source files to bind into ``manifest.json``. The CLI
        supplies the processed config and layer files automatically.
        Programmatic callers should provide every external source needed to
        reproduce the assembled PFA.

    Returns
    -------
    ProbabilisticResult
        Per-component fitted surfaces, the combined surface, scenario
        ablations (when configured), and the originating config.
    """
    config.validate_raise()
    _validate_output_format_request(config)
    if not config.enabled:
        return ProbabilisticResult(config=config, skipped=True)
    implementation_sha256 = _probabilistic_implementation_hash()
    adapter = PFAGridAdapter(
        pfa, criteria=criteria, dimensions=config.dimensions
    )
    validate_declared_components(
        adapter,
        set(config.labels.label_columns) | set(config.alpha),
    )
    _validate_evidence_layer_references(adapter, config)
    scenario_configs = []
    for scenario in config.scenarios:
        scenario_config = replace(
            _apply_scenario(config, scenario),
            output_dir=config.output_dir / "scenarios" / scenario.name,
        )
        scenario_config.validate_raise()
        scenario_configs.append((scenario, scenario_config))
    scope_configs = {
        "baseline": config,
        **{
            f"scenario:{scenario.name}": scenario_config
            for scenario, scenario_config in scenario_configs
        },
    }
    streamed_run = (
        config.inference.backend == "gblk"
        and config.inference.gblk_bayesian.enabled
        and config.outputs.posterior_draw_blocks
    )
    resume_guard = None
    if streamed_run:
        resume_guard = _begin_streamed_run_resume(
            config.output_dir,
            config,
            scope_configs=scope_configs,
            input_artifacts=input_artifacts,
            expected_implementation_sha256=implementation_sha256,
        )
    else:
        validate_output_namespace(config.output_dir, config)

    if config.inference.backend == "gblk":
        if config.calibration.method != "none":
            raise GEOPFAValueError(
                "the GBLK top-level runner does not apply post-hoc calibration; "
                "set calibration.method='none' and use "
                "run_gblk_calibration_cv for explicit raw out-of-fold diagnostics"
            )
        from .gblk_runner import run_gblk_probabilistic  # noqa: PLC0415

        alphas = _build_alphas(adapter, config)
        gblk_result = run_gblk_probabilistic(
            pfa,
            config,
            criteria=criteria,
            nc=config.spatial_field.lattice_centers_per_dimension,
        )
        _validate_result_surface_grids(
            gblk_result.components,
            gblk_result.combined,
            context="GBLK baseline",
        )
        gblk_scenario_results: dict[str, ProbabilisticResult] = {}
        for scenario, scenario_config in scenario_configs:
            scope = f"scenario:{scenario.name}"
            scenario_result = run_gblk_probabilistic(
                pfa,
                scenario_config,
                criteria=criteria,
                nc=scenario_config.spatial_field.lattice_centers_per_dimension,
                posterior_scope=scope,
            )
            _validate_result_surface_grids(
                scenario_result.components,
                scenario_result.combined,
                context=f"GBLK scenario {scenario.name!r}",
            )
            gblk_scenario_results[scenario.name] = scenario_result
        gblk_scenarios = {
            name: scenario_result.components
            for name, scenario_result in gblk_scenario_results.items()
        }
        gblk_result = replace(
            gblk_result,
            scenarios=gblk_scenarios,
            site_selection=_run_site_selection(config),
        )
        if not gblk_result.skipped:
            _write_configured_probability_outputs(
                gblk_result.components,
                gblk_result.combined,
                config,
                config.output_dir,
            )
            if config.outputs.scenarios:
                for scenario, scenario_config in scenario_configs:
                    scenario_result = gblk_scenario_results[scenario.name]
                    _write_configured_probability_outputs(
                        scenario_result.components,
                        scenario_result.combined,
                        scenario_config,
                        config.output_dir / "scenarios" / scenario.name,
                    )
            if alphas:
                config.output_dir.mkdir(parents=True, exist_ok=True)
                write_alpha_provenance(
                    alphas,
                    config.output_dir / "alpha_provenance.json",
                )
        manifest_path = write_manifest(
            config.output_dir,
            config=config,
            input_artifacts=input_artifacts,
            expected_implementation_sha256=implementation_sha256,
        )
        if resume_guard is not None:
            resume_guard.finish(manifest_path)
        return gblk_result

    if config.inference.backend == "sequential":
        warnings.warn(
            "inference.backend='sequential' is deprecated and will be removed "
            "in a future release; set inference.backend='gblk' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    sequential_configs = [config, *(cfg for _, cfg in scenario_configs)]
    requires_labels = any(
        not alpha.force_prior_predictive
        for sequential_config in sequential_configs
        for alpha in sequential_config.alpha.values()
    )
    labels = load_labels(config.labels) if requires_labels else None

    # Build alpha_c per component first so we can write provenance even if
    # the regression fit fails for some components.
    alphas = _build_alphas(adapter, config)

    components: dict[str, ComponentProbability] = {}
    for name in adapter.components():
        if name not in alphas:
            continue
        fitted = _fit_component(adapter, name, config, labels, alphas[name])
        if fitted is not None:
            components[name] = fitted

    combined = _combine_components(components)
    scenarios: dict[str, dict[str, ComponentProbability]] = {}
    for scenario, sub_cfg in scenario_configs:
        scenario_alphas = {
            name: build_alpha_c(
                adapter.component_data(name),
                sub_cfg.alpha[name],
                grid_gdf=adapter.pr_norm(name),
            )
            for name in adapter.components()
            if name in sub_cfg.alpha
        }
        scenario_components: dict[str, ComponentProbability] = {}
        for name in adapter.components():
            if name not in scenario_alphas:
                continue
            fitted = _fit_component(
                adapter, name, sub_cfg, labels, scenario_alphas[name]
            )
            if fitted is not None:
                scenario_components[name] = fitted
        scenarios[scenario.name] = scenario_components

    # ------------------------------------------------------------------
    # Calibration: run block-CV → fit calibration map → apply to surfaces
    # ------------------------------------------------------------------
    calibrated_components: dict[str, gpd.GeoDataFrame] = {}
    calibration_maps: dict[str, CalibrationMap] = {}
    cv_result: CVResult | None = None
    if config.calibration.method != "none" and components:
        fit_on = config.calibration.fit_on
        if fit_on != "block_cv":
            raise GEOPFAValueError(
                f"calibration.fit_on={fit_on!r} is not implemented; "
                "use 'block_cv' rather than silently changing the estimand"
            )
        cv_result = run_block_cv(pfa, config, criteria=criteria)
        for name, oof in cv_result.oof.items():
            if name not in components:
                continue
            if len(oof["p"]) < 2 or len(np.unique(oof["y"])) < 2:  # noqa: PLR2004
                raise GEOPFAValueError(
                    f"component {name!r} lacks two outcome classes in OOF predictions"
                )
            cm = fit_posthoc_calibration(
                oof["p"], oof["y"], method=config.calibration.method
            )
            calibration_maps[name] = cm
            surface = components[name].probability.copy()
            surface["probability_raw"] = surface["probability"].to_numpy()
            surface["probability"] = cm.predict(
                surface["probability"].to_numpy()
            )
            calibrated_components[name] = surface

        # Re-combine calibrated component probabilities under the same product.
        if calibrated_components:
            base, arrays = _validated_probability_arrays(
                calibrated_components,
                context="calibrated component combination",
            )
            base["probability"] = combine_probability_surfaces(arrays)
            calibrated_components["combined"] = base

    _write_configured_probability_outputs(
        components,
        combined,
        config,
        config.output_dir,
    )
    if config.outputs.scenarios:
        for scenario, scenario_config in scenario_configs:
            scenario_components = scenarios[scenario.name]
            _write_configured_probability_outputs(
                scenario_components,
                _combine_components(scenario_components),
                scenario_config,
                config.output_dir / "scenarios" / scenario.name,
            )
    if config.outputs.format and calibrated_components:
        write_probability_outputs(
            calibrated_components,
            config.output_dir / "calibrated",
            formats=config.outputs.format,
            include_uncertainty=False,
        )

    # Calibration metrics JSON
    if (
        config.calibration.method != "none"
        and cv_result is not None
        and config.outputs.calibration_artifacts
    ):
        _write_calibration_metrics(cv_result, config)
    if cv_result is not None and config.outputs.decision_artifacts:
        _write_decision_metrics(cv_result, components, config)

    # Always write alpha provenance when there's anything to record.
    if alphas:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        write_alpha_provenance(
            alphas, config.output_dir / "alpha_provenance.json"
        )

    # Manifest with hashes of every produced file
    manifest_path = write_manifest(
        config.output_dir,
        config=config,
        input_artifacts=input_artifacts,
        expected_implementation_sha256=implementation_sha256,
    )
    return ProbabilisticResult(
        components=components,
        combined=combined,
        calibrated_components=calibrated_components,
        calibration_maps=calibration_maps,
        cv=cv_result,
        scenarios=scenarios,
        site_selection=_run_site_selection(config),
        config=config,
        skipped=False,
    )


__all__ = ["ProbabilisticResult", "run_probabilistic"]
