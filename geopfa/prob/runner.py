"""Top-level probabilistic-method entry point: ``run_probabilistic(pfa, config)``.

Translates a :class:`ProbabilisticConfig` into the existing
:func:`fit_component_probability` calls, combines per-component surfaces via
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
    _probabilistic_implementation_hash,
    validate_output_namespace,
    write_manifest,
    write_probability_outputs,
)
from .labels import LoadedLabels, available_components, load_labels
from .pfa_grid import PFAGridAdapter, validate_declared_components


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
        :class:`CVResult` from the block-CV run, when calibration is
        configured. Empty when calibration is off.
    scenarios
        ``{scenario_name: {component_name: ComponentProbability}}``
        when scenarios were configured; ``{}`` otherwise.
    config
        The :class:`ProbabilisticConfig` that drove this run (carried for
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
    per_region_beta: dict[str, dict[str, Any]] = field(default_factory=dict)
    site_selection: dict[str, Any] = field(default_factory=dict)
    component_probability_draws: dict[str, np.ndarray] = field(
        default_factory=dict
    )
    combined_probability_draws: np.ndarray | None = None
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
    }


def _fit_component(
    adapter: PFAGridAdapter,
    component: str,
    cfg: ProbabilisticConfig,
    labels: LoadedLabels,
    alpha_result: AlphaCResult,
) -> ComponentProbability | None:
    """Fit one component, routing to the configured inference backend."""
    if component not in cfg.labels.label_columns:
        warnings.warn(
            f"component {component!r} has no entry in labels.label_columns; "
            "skipping (configure a label column to include it in the run)",
            UserWarning,
            stacklevel=3,
        )
        return None
    if component not in cfg.alpha:
        warnings.warn(
            f"component {component!r} has no entry in alpha; "
            "skipping (configure an alpha mode to include it in the run)",
            UserWarning,
            stacklevel=3,
        )
        return None

    component_data = adapter.component_data(component)

    # The componentwise path is the explicit sequential estimator. Bayesian
    # inference is available only through the joint GBLK runner.
    kwargs = build_fit_kwargs(
        alpha_config=cfg.alpha[component],
        evidence_config=cfg.evidence,
        spatial_field_config=cfg.spatial_field,
        pu_mode=cfg.labels.pu_mode,
        pu_class_prior=cfg.labels.class_prior_for(component),
        min_wells=cfg.labels.min_wells_for_fit,
    )
    # Resolve play-type defaults with actual layer names from this component.
    play_type = kwargs.pop("_play_type", None)
    if play_type:
        from .play_types import play_type_defaults  # noqa: PLC0415

        layer_names = list(component_data.get("layers", {}).keys())
        pt_defaults = play_type_defaults(play_type, layer_names=layer_names)
        merged_weights = {**pt_defaults["per_feature_weights"]}
        merged_weights.update(kwargs.get("per_feature_weights") or {})
        merged_means = {**pt_defaults["prior_means"]}
        kwargs["per_feature_weights"] = merged_weights or None
        kwargs["prior_means"] = merged_means or None

    extra_excluded = set(alpha_result.excluded_layer_names) | set(
        kwargs.get("excluded_layer_names", ())
    )
    kwargs["excluded_layer_names"] = tuple(sorted(extra_excluded))
    kwargs["alpha_offset"] = alpha_result.grid_offset
    kwargs["prior_layer_name"] = None

    return fit_component_probability(
        component_data,
        labeled_wells=labels.gdf,
        label_column=cfg.labels.label_columns[component],
        **kwargs,
    )


def _combine_components(
    components: dict[str, ComponentProbability],
    cfg: ProbabilisticConfig,
) -> gpd.GeoDataFrame:
    """Combine per-component surfaces into a single combined GeoDataFrame."""
    if not components:
        return gpd.GeoDataFrame()
    rule = cfg.combination.rule
    component_arrays: list[np.ndarray] = []
    base_gdf: gpd.GeoDataFrame | None = None
    for surface in components.values():
        gdf = surface.probability
        if base_gdf is None:
            base_gdf = gdf[["geometry"]].copy()
        component_arrays.append(gdf["probability"].astype(float).to_numpy())
    if rule == "product":
        combined_arr = combine_probability_surfaces(component_arrays)
    elif rule == "geometric_mean":
        stacked = np.stack(component_arrays, axis=0)
        combined_arr = np.exp(
            np.mean(np.log(np.clip(stacked, 1e-12, 1.0)), axis=0)
        )
    else:  # pragma: no cover - config validation prevents unknown rules
        raise AssertionError(f"unexpected combination rule {rule!r}")
    if base_gdf is None:  # pragma: no cover - guarded by nonempty components
        raise RuntimeError("component combination produced no base geometry")
    base_gdf["probability"] = combined_arr
    return base_gdf


def _write_one_csv(gdf: gpd.GeoDataFrame, path: Path) -> None:
    if len(gdf) == 0:
        return
    out = gdf.copy()
    out["x"] = out.geometry.x
    out["y"] = out.geometry.y
    if out.geometry.has_z.any():
        out["z"] = out.geometry.z
    out.drop(columns=["geometry"]).to_csv(path, index=False)


def _write_csv_outputs(
    components: dict[str, ComponentProbability],
    combined: gpd.GeoDataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, surface in components.items():
        _write_one_csv(
            surface.probability, output_dir / f"{name}_probability.csv"
        )
    if len(combined) > 0:
        _write_one_csv(combined, output_dir / "combined_probability.csv")


def _write_configured_probability_outputs(
    components: dict[str, ComponentProbability],
    combined: gpd.GeoDataFrame,
    config: ProbabilisticConfig,
    output_dir: Path,
) -> None:
    """Write tabular and spatial probability products under output toggles."""
    if "csv" in config.outputs.format:
        _write_csv_outputs(components, combined, output_dir)
    formats_for_writer = tuple(
        fmt
        for fmt in config.outputs.format
        if fmt == "parquet"
        or (config.outputs.probability_rasters and fmt in {"geotiff", "vtk"})
    )
    if not formats_for_writer:
        return
    all_surfaces: dict[str, gpd.GeoDataFrame] = {
        name: surface.probability for name, surface in components.items()
    }
    if len(combined) > 0:
        all_surfaces["combined"] = combined
    write_probability_outputs(
        all_surfaces,
        output_dir,
        formats=formats_for_writer,
        include_uncertainty=config.outputs.uncertainty_rasters,
    )


def _write_calibrated_csv_outputs(
    calibrated: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, gdf in calibrated.items():
        if len(gdf) == 0:
            continue
        _write_one_csv(gdf, output_dir / f"{name}_probability.csv")


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
    new_alpha = {}
    for name, alpha in cfg.alpha.items():
        if scenario.include_priors:
            new_alpha[name] = alpha
        else:
            # Disable spatial/layer priors → fall back to scalar mode
            new_alpha[name] = type(alpha)(
                mode="scalar", scalar_fallback_pr0=alpha.scalar_fallback_pr0
            )
    new_spatial = type(cfg.spatial_field)(
        enabled=scenario.include_spatial,
        backend=cfg.spatial_field.backend,
        kernel=cfg.spatial_field.kernel,
        n_inducing=cfg.spatial_field.n_inducing,
        lengthscale_lower_frac=cfg.spatial_field.lengthscale_lower_frac,
        lengthscale_upper_frac=cfg.spatial_field.lengthscale_upper_frac,
        optimize_restarts=cfg.spatial_field.optimize_restarts,
        n_levels=cfg.spatial_field.n_levels,
        lattice_centers_per_dimension=(
            cfg.spatial_field.lattice_centers_per_dimension
        ),
        coordinate_scaling=cfg.spatial_field.coordinate_scaling,
    )
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
    new_evidence = type(cfg.evidence)(
        regularization=cfg.evidence.regularization,
        include_layers=cfg.evidence.include_layers,
        exclude_layers=new_exclude,
        sparse_binary_threshold=cfg.evidence.sparse_binary_threshold,
        coordinate_blacklist=cfg.evidence.coordinate_blacklist,
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
        outputs=cfg.outputs,
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
        Validated :class:`ProbabilisticConfig` (e.g. from
        :func:`load_probabilistic_config`).
    criteria
        Criteria key to operate on. Defaults to ``"geologic"``.
    input_artifacts
        Optional named source files to bind into ``manifest.json``. The CLI
        supplies the config and PFA pickle automatically; programmatic callers
        should provide any serialized PFA source needed to reproduce the run.

    Returns
    -------
    ProbabilisticResult
        Per-component fitted surfaces, the combined surface, scenario
        ablations (when configured), and the originating config.
    """
    config.validate_raise()
    if not config.enabled:
        return ProbabilisticResult(config=config, skipped=True)
    implementation_sha256 = _probabilistic_implementation_hash()
    validate_output_namespace(
        config.output_dir, config, input_artifacts=input_artifacts
    )
    adapter = PFAGridAdapter(
        pfa, criteria=criteria, dimensions=config.dimensions
    )
    validate_declared_components(
        adapter,
        set(config.labels.label_columns) | set(config.alpha),
    )
    _validate_evidence_layer_references(adapter, config)
    scenario_configs = [
        (scenario, _apply_scenario(config, scenario))
        for scenario in config.scenarios
    ]
    for _, scenario_config in scenario_configs:
        scenario_config.validate_raise()

    if config.inference.backend == "gblk":
        if config.calibration.method != "none":
            raise GEOPFAValueError(
                "the GBLK top-level runner does not apply post-hoc calibration; "
                "set calibration.method='none' and use "
                "run_gblk_calibration_cv for explicit raw out-of-fold diagnostics"
            )
        from .gblk_runner import run_gblk_probabilistic  # noqa: PLC0415

        gblk_result = run_gblk_probabilistic(
            pfa,
            config,
            criteria=criteria,
            nc=config.spatial_field.lattice_centers_per_dimension,
        )
        gblk_scenarios: dict[str, dict[str, ComponentProbability]] = {}
        for scenario, scenario_config in scenario_configs:
            scenario_output_dir = (
                config.output_dir / "scenarios" / scenario.name
            )
            scenario_run_config = replace(
                scenario_config, output_dir=scenario_output_dir
            )
            scenario_result = run_gblk_probabilistic(
                pfa,
                scenario_run_config,
                criteria=criteria,
                nc=scenario_config.spatial_field.lattice_centers_per_dimension,
                posterior_scope=f"scenario:{scenario.name}",
            )
            gblk_scenarios[scenario.name] = scenario_result.components
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
                    scenario_components = gblk_scenarios[scenario.name]
                    _write_configured_probability_outputs(
                        scenario_components,
                        _combine_components(
                            scenario_components,
                            scenario_config,
                        ),
                        scenario_config,
                        config.output_dir / "scenarios" / scenario.name,
                    )
            if any(config.output_dir.glob("*")):
                write_manifest(
                    config.output_dir,
                    config=config,
                    input_artifacts=input_artifacts,
                    expected_implementation_sha256=implementation_sha256,
                )
        return gblk_result

    if config.inference.backend == "sequential":
        warnings.warn(
            "inference.backend='sequential' is deprecated and will be removed "
            "in a future release; set inference.backend='gblk' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    labels = load_labels(config.labels)

    # Build alpha_c per component first so we can write provenance even if
    # the regression fit fails for some components.
    alphas: dict[str, AlphaCResult] = {}
    for name in adapter.components():
        if name not in config.alpha:
            continue
        comp_data = adapter.component_data(name)
        alphas[name] = build_alpha_c(
            comp_data, config.alpha[name], grid_gdf=adapter.pr_norm(name)
        )

    components: dict[str, ComponentProbability] = {}
    per_region_beta: dict[str, dict[str, Any]] = {}
    for name in adapter.components():
        if name not in available_components(config.labels):
            warnings.warn(
                f"component {name!r} has no entry in labels.label_columns; "
                "skipping (configure a label column to include it in the run)",
                UserWarning,
                stacklevel=2,
            )
            continue
        if name not in alphas:
            continue
        fitted = _fit_component(adapter, name, config, labels, alphas[name])
        if fitted is not None:
            components[name] = fitted

    combined = _combine_components(components, config)

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
            if name not in available_components(sub_cfg.labels):
                continue
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

        # Re-combine using calibrated surfaces (product / geometric_mean)
        if calibrated_components:
            arrays = [
                gdf["probability"].astype(float).to_numpy()
                for gdf in calibrated_components.values()
            ]
            base = next(iter(calibrated_components.values()))[
                ["geometry"]
            ].copy()
            if config.combination.rule == "product":
                base["probability"] = combine_probability_surfaces(arrays)
            elif config.combination.rule == "geometric_mean":
                stacked = np.stack(arrays, axis=0)
                base["probability"] = np.exp(
                    np.mean(np.log(np.clip(stacked, 1e-12, 1.0)), axis=0)
                )
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
                _combine_components(scenario_components, scenario_config),
                scenario_config,
                config.output_dir / "scenarios" / scenario.name,
            )
    if "csv" in config.outputs.format and calibrated_components:
        _write_calibrated_csv_outputs(
            calibrated_components, config.output_dir / "calibrated"
        )

    # Write calibrated non-CSV products; raw products were written above.
    formats_for_writer = tuple(
        fmt
        for fmt in config.outputs.format
        if fmt == "parquet"
        or (config.outputs.probability_rasters and fmt in {"geotiff", "vtk"})
    )
    if formats_for_writer and calibrated_components:
        write_probability_outputs(
            calibrated_components,
            config.output_dir / "calibrated",
            formats=formats_for_writer,
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
    if any(config.output_dir.glob("*")):
        write_manifest(
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
        per_region_beta=per_region_beta,
        site_selection=_run_site_selection(config),
        config=config,
        skipped=False,
    )


def run_probabilistic_pfa(
    pfa: dict,
    *,
    criteria: str = "geologic",
) -> ProbabilisticResult:
    """Run the probabilistic workflow directly from a pfa dict.

    This is the primary integration point with the existing geoPFA pipeline.
    The pfa dict must contain a ``"probabilistic"`` key with all settings,
    alongside the usual ``"criteria"`` / components structure::

        pfa = {
            "criteria": { "geologic": { "components": { ... } } },
            "probabilistic": {
                "enabled": true,
                "labels": { "source": "wells.gpkg", ... },
                ...
            }
        }

    Parameters
    ----------
    pfa
        geoPFA dict (output of the preprocessing pipeline) that contains a
        ``"probabilistic"`` config block.
    criteria
        Criteria key to operate on.  Defaults to ``"geologic"``.

    Returns
    -------
    ProbabilisticResult
    """
    from .config import ProbabilisticConfig  # noqa: PLC0415

    config = ProbabilisticConfig.from_pfa(pfa)
    return run_probabilistic(pfa, config, criteria=criteria)


__all__ = ["ProbabilisticResult", "run_probabilistic", "run_probabilistic_pfa"]
