# The probabilistic method

This page is the user-facing reference for the `geopfa.prob` package. It describes
the **Generalized Bayesian LatticeKrig (GBLK)** probabilistic method, which is the
default and recommended inference backend as of the current release.

The GBLK method is described in full in *Geothermal Play Fairway Analysis: A
Generalized Bayesian LatticeKrig Framework for Supercritical Targets* (Hettinger,
2026). This page is the operational guide and configuration reference.

> **Deprecated:** The per-component sequential logistic plus standalone spatial
> smoothing method (`inference.backend="sequential"`) is deprecated and will
> be removed in a future release. A `DeprecationWarning` is emitted when it is used. Migrate
> to `inference.backend="gblk"` (the default) using the guide in
> `docs/migration_guide.md`.

There is exactly one Bayesian implementation. Set
`inference.backend="gblk"` and `inference.gblk_bayesian.enabled=true` to use the
public LatticeKrigX Paige/INLA model and paired posterior draws. There is no
independent `backend="bayesian"` model.

## The GBLK model

The GBLK method supports binary component outcomes and continuous Gaussian
heat observations. Components with the same response family are fit jointly.
For a binary component $q$ (e.g. reservoir or seal) at well location $i$ with
observation support $H_{qi}$:

$$
\text{logit}\, p_{qi} \;=\; o_q(s_i) \;+\; Z_{qi}\, d_{qr} \;+\; [H_{qi}\, g_q]
$$

where:

- **$o_q(s_i)$ — physics-informed prior offset.** Logit-scale prior constructed by
  `geopfa.prob.alpha.build_alpha_c` from domain constraints (thermal exceedance,
  layer logit, scalar, or multi-layer combination).
- **$Z_{qi}\, d_q$ — evidence regression.** Evidence-layer values at the well
  multiplied by component-specific coefficients. The empirical-Bayes GBLK path
  estimates these coefficients with the spatial field in one likelihood under
  the declared Gaussian coefficient prior. The separate hierarchical regional
  runner can pool regional coefficients by play type.
- **$[H_{qi}\, g_q]$ — latent spatial field.** Multiresolution LatticeKrig basis
  $H_{qi}$ (averaged over the observation support) times latent field coefficients
  $g_q$. All components share the same LatticeKrig basis; cross-component dependence
  is captured by the estimated $Q \times Q$ correlation matrix $\Omega$ within
  a same-family fit.

For a Gaussian heat component, the corresponding model is

$$
T_i / a_T = \mu_T(s_i) / a_T + Z_{Ti}d_T + [H_{Ti}g_T] + \epsilon_i,
$$

where $T_i$ is the continuous temperature observation, $\mu_T(s_i)$ is the
configured thermal-model mean, $a_T$ is the declared response scale, and
$\epsilon_i\mid\tau_T\sim N(0,\tau_T^{-1})$. The
scale controls numerical conditioning and the physical interpretation of the
spatial prior; predictions and coefficients are returned in the original
temperature units. Each posterior mean-field draw remains paired with its
Gaussian likelihood-precision draw. Their normal survival probability gives
$\Pr(T(s)>T^*)$ at the configured threshold. Those probability draws then pass
to the same component-combination step used for binary outcomes. Gaussian and
Bernoulli components are fit separately, so geoPFA does not claim estimated
cross-response-family dependence.

The default combined surface is the **conditional plug-in co-occurrence score**:

$$
p_{\text{plugin}}(s) \;=\; \prod_q \widehat{p}_q(s)
$$

Same-family component fields can be estimated jointly through $\Omega$. The
deterministic combined surface is the product of fitted marginal
probabilities. The Bayesian path combines paired component probabilities
within each draw and then summarizes those combined draws. Neither quantity is
an unconditional joint-event probability because the component product is a
declared conditional combination rule. This is the `combined` surface in the
`ProbabilisticResult`.

### Paper-to-geoPFA mapping

| Paper construct | geoPFA module | `latticekrigx.glk` engine |
|---|---|---|
| components $q$ | `heat`, `reservoir`, `seal`, … | `y` columns / `n_components` |
| prior offset $o_q$ | `alpha.py::build_alpha_c` | `offsets` |
| continuous heat response | `labels.observation_models` | Gaussian family + identity link |
| evidence $Z_q d_{qr}$ | processed evidence layers | design matrix + `glk.hierarchy` |
| latent field $g_q$, $\Omega$ | — | `glk.multivariate` + `fit_joint` |
| support operator $H_{qi}$ | well point/interval/trajectory/areal | `glk.support` averaged rows |
| PU / undrilled unlabeled | `pu.py` (non-negative PU risk; sequential outcome path only) | not part of the GBLK likelihood |
| preferential site collection | `site_selection.py` (finite-candidate sensitivity model) | deliberately separate from LatticeKrigX |
| plug-in co-occurrence | `result.combined` | product of joint-fit marginals |
| regional pooling by play-type | `regions.py` / `play_types.py` | `glk.hierarchy` |
| calibration (Brier/BSS, spatial-block CV) | `calibration.py`, `cv.py` | `glk.calibration` |
| sole Bayesian option | `gblk_bayesian.enabled=true` | public Paige/INLA joint draws for 2-D or 3-D geometry |

## Quick start

The whole pipeline is driven by a single JSON config. Drop a `probabilistic` block into your existing PFA config:

```javascript
{
  "criteria": { ... },             // your existing PFA config (unchanged)
  "pfa_pickle": "outputs/pfa.pkl", // (optional) pre-processed PFA dict from geoPFA
  "probabilistic": {
    "enabled": true,
    "output_dir": "outputs/probabilistic/",
    "dimensions": "2d",            // "2d" or "3d"

    "labels": {
      "source": "data/wells.gpkg",
      "id_col": "well_id",
      "layer": "wells",
      "label_columns": {
        "heat": "temperature_c",
        "reservoir": "reservoir_label"
      },
      "observation_models": {
        "heat": {"family": "gaussian", "response_scale": 50.0}
      }
    },

    "alpha": {
      "heat": {
        "mode": "thermal_exceedance",
        "thermal_raster": "data/thermal_at_3km.tif",
        "threshold": 200.0,
        "uncertainty_raster": "data/thermal_sd.tif"
      },
      "reservoir": {
        "mode": "layer_logit",
        "layer": "fault_slip_dilation"
      }
    },

    "inference": {
      "backend": "gblk",
      "gblk_bayesian": {"enabled": true},
      "predictive_stacking": {
        "enabled": true,
        "validation_depths_m": {"heat": 3000.0}
      }
    },

    "calibration": { "method": "none" },

    "outputs": {
      "format": ["geotiff", "csv", "parquet"]
    }
  }
}
```

Then run:

```bash
pixi run -e dev-gblk geopfa-prob run --config path/to/config.json
```

All relative paths in the JSON—including `pfa_pickle`, `output_dir`, label,
thermal, uncertainty, and candidate-frame paths—are resolved relative to the
config file, not the shell's working directory. The run manifest records the
resolved inputs with byte counts and SHA-256 digests, the full effective config,
the producing geoPFA and LatticeKrigX versions, and a SHA-256 fingerprint of
both runtime source trees. A non-empty output directory is accepted only for a
compatible completed run or resumable posterior workspace. Use a fresh
directory when the config or implementation changes so artifacts from
different runs cannot mix.
The runner also compares its start/end runtime fingerprints and rejects an
artifact if either source tree changes while the model is executing.

That call:

1. Loads the labelled wells from the configured source (GeoPackage / shapefile / CSV).
2. Builds the configured event prior per component and retains the physical
   thermal mean for Gaussian heat components.
3. Assembles component responses, missingness masks, evidence, and offsets via
   `gblk_assemble.assemble_gblk_inputs`.
4. Fits each same-family component group through `fit_joint`. Gaussian heat
   uses an identity link; binary components use a Bernoulli-logit likelihood.
   Components explicitly configured as prior-predictive bypass the outcome
   likelihood.
5. Optionally selects a separate prior/update stacking weight for each fitted
   component from blocked out-of-fold log score.
6. Produces per-component probability surfaces `p_q(s)` and the conditional
   plug-in co-occurrence surface `∏_q p_q(s)`.
7. Runs any configured ablation scenarios.
8. Writes outputs to `output_dir/`: per-component CSV + GeoTIFF + Parquet
   (and VTK for 3D), optional paired Bayesian posterior-probability blocks,
   `alpha_provenance.json`, and `manifest.json` (input/output SHA-256 hashes +
   config and implementation hashes + run id).

## Config reference

The complete schema is enforced by `ProbabilisticConfig`; the principal fields
are summarized below.

### `labels`

| Key | Default | Purpose |
| --- | --- | --- |
| `source` | required | Path to a GPKG / SHP / CSV holding labelled wells. |
| `id_col` | required | Unique-id column on the wells file. |
| `layer` | `null` | Sub-layer for GPKG sources. |
| `label_columns` | required | Per-component label-column mapping (e.g. `{"heat": "heat_label", "reservoir": "reservoir_label"}`). |
| `observation_models` | `{}` | Optional per-component likelihood. Omitted components are Bernoulli. A Gaussian component requires `family="gaussian"` and a positive `response_scale`. |
| `min_wells_for_fit` | `4` | Minimum observed support required by the selected fit; insufficient support raises rather than silently changing models. |
| `pu_mode` | `"off"` | `"off"`, `"naive_pseudo_absence"`, or modern non-negative PU risk estimation (`"nnpu"`; sequential outcome path only). |
| `pu_class_prior` | `null` | Externally identified population prevalence required by `pu_mode="nnpu"`; a scalar or per-component mapping with values strictly inside `(0, 1)`. |

Positive--unlabeled correction is opt-in. The nnPU estimator minimizes the
non-negative logistic PU risk of Kiryo et al. (2017) under the
selected-completely-at-random assumption. geoPFA deliberately does not estimate
the class prior from the same labeled positives: the caller must supply an
externally justified `pu_class_prior`. nnPU addresses missing negative labels;
it does not correct preferential site collection. Use the optional
finite-candidate site-selection analysis for that distinct problem.

### `site_selection`

`mode="joint_binary"` enables a separate outcome/selection sensitivity model on
a declared finite candidate frame. It jointly represents the binary outcome
and whether each candidate was sampled, marginalizing the unobserved outcomes
at unsampled candidates. Because the outcome effect on selection is not
identified from the observed sites alone, callers must freeze one or more
`outcome_selection_log_odds` values and report the resulting sensitivity
envelope. This analysis does not alter or wrap the LatticeKrigX likelihood.

### `alpha[component]`

| Mode | Required keys | What it does |
| --- | --- | --- |
| `scalar` | `scalar_fallback_pr0` | Uniform $\alpha = \text{logit}(p_0)$. |
| `layer_logit` | `layer`, `p_min`, `p_max` | Min-max rescale a named layer to $[p_{\min}, p_{\max}]$ and apply `logit`. Layer is auto-excluded from regression. |
| `thermal_exceedance` | `thermal_raster`, `threshold` (`uncertainty_raster` optional) | Step function or $\Phi$-integration of `P(T > T*)`. Clipped then `logit`. |
| `thermal_layer_exceedance` | `layer`, `threshold` (`uncertainty_column` optional) | Compute `P(T > T*)` from temperature mean/SD columns already carried by a PFA layer. |
| `multi_layer` | `layers` | Sum of `layer_logit` offsets in logit space; all named layers auto-excluded. |

`p_min` and `p_max` must be finite and strictly inside `(0, 1)` so every
logit offset is finite. `force_prior_predictive=true` explicitly prevents an
outcome update. With `use_evidence_prior=true`, named Gaussian coefficient
priors are sampled; otherwise the component is a fixed prior prediction and
does not produce replicated uncertainty draws.

### `evidence`

| Key | Default | Purpose |
| --- | --- | --- |
| `regularization.C` | `1.0` | Sklearn-style `1/lambda`. |
| `regularization.per_feature_weights` | `{}` | Per-layer L2 weight overrides. |
| `regularization.prior_means`, `prior_precisions` | `{}`, `{}` | Named Gaussian coefficient priors, required for prior-predictive evidence coefficients. |
| `regularization.play_type` | `null` | Optional key into the play-type defaults registry (Phase C.2, future). |
| `include_layers` | `null` | Whitelist (default: all non-excluded). |
| `exclude_layers` | `[]` | Blacklist; alpha layers are auto-added. |
| `sparse_binary_threshold` | `0.90` | Reject layers where ≥90% of cells share one value. |
| `coordinate_blacklist` | sensible default | Coordinate-like column names that must never enter the design matrix. |
| `standardization` | `"observed_labels"` | Fit transformations on observed training rows; `"prediction_support"` is an explicit prior-predictive option when no outcome-trained transformation exists. |

### `spatial_field`

The GBLK backend uses `enabled`, `n_levels`, and
`lattice_centers_per_dimension` to define its joint LatticeKrig field. The
remaining options configure the deprecated sequential spatial smoother.

| Key | Default | Purpose |
| --- | --- | --- |
| `enabled` | `true` | Toggle `u_c`. |
| `backend` | `"latticekrigx"` | `"latticekrigx"` for GBLK, `"rbf"` for the lightweight sequential smoother, or `"none"`. |
| `kernel` | `"rbf_matern32"` | `"rbf"` / `"matern32"` / `"rbf_matern32"`. |
| `n_inducing` | `300` | Inducing-point target for the sparse GP. |
| `lengthscale_lower_frac`, `lengthscale_upper_frac` | `0.02`, `0.20` | ARD lengthscale bound fractions of the coordinate radius. |
| `optimize_restarts` | `0` | Multi-start ML-II restarts. |
| `n_levels` | `2` | Number of multiresolution LatticeKrig levels. In 3D, the basis is constructed jointly over x, y, and z. |
| `lattice_centers_per_dimension` | `6` | Coarsest-level centers per spatial dimension for GBLK. Choose this before outcome evaluation and keep the resulting basis commensurate with the effective training sample. |
| `coordinate_scaling` | `"axis_range"` | `"axis_range"` gives domain-relative axes; `"physical_isotropic"` preserves metre-scale axis ratios and requires commensurate coordinate units. |

### `inference`

| Key | Default | Purpose |
| --- | --- | --- |
| `backend` | `"gblk"` | `"gblk"` (default, joint GBLK method) or `"sequential"` (**deprecated**). |
| `gblk_bayesian.enabled` | `false` | Select the sole Bayesian dispatch: public Paige/INLA. |
| `gblk_bayesian.n_draws` | `200` | Number of paired posterior draws. |
| `gblk_bayesian.seed` | `0` | RNG seed for reproducible draws. |
| `gblk_bayesian.ci_level` | `0.9` | Credible-interval level for `probability_lo` / `probability_hi` surfaces. |
| `gblk_bayesian.cor_scale_median` | `0.1` | Paige correlation-scale prior median in model coordinates. |
| `gblk_bayesian.spatial_sd_u` | `1.0` | Paige spatial standard-deviation threshold. |
| `gblk_bayesian.spatial_sd_tail_probability` | `0.05` | Prior probability above `spatial_sd_u`. |
| `gblk_bayesian.dirichlet_concentration` | `1.5` | Symmetric Paige level-weight concentration. |
| `gblk_bayesian.kleiber_r0`, `kleiber_r1` | `null` | Required frozen profile parameters for a bivariate fit. |
| `predictive_stacking.enabled` | `false` | Select a component-specific mixture of the configured event prior and full Bayesian update by buffered or blocked out-of-fold logarithmic score. Zero retains the prior and one retains the full update. |
| `predictive_stacking.validation_depths_m` | `{}` | In a 3-D analysis, optionally map component names to positive-down target depths. Each mapped component selects its stacking weight only from held-out observations at that depth. If fewer than `labels.min_wells_for_fit` distinct wells occur there, that component retains its prior. Unmapped components use all of their held-out observations. |

Predictive stacking uses the spatial split declared in `cross_validation` and
never scores in-sample predictions. Bernoulli components use held-out binary
log score. Gaussian components use the held-out continuous posterior predictive
log density, including the sampled residual precision, rather than discarding
information by thresholding the observations. The selected distribution
mixture induces the same mixture of event probabilities passed to component
combination. The selected weight and the prior, full, and selected held-out log
scores are recorded in component diagnostics. A component-specific validation
depth aligns this model-selection step with a target-depth map while the
Gaussian fit can still use complete temperature profiles and other components
retain their appropriate validation support. Gaussian predictive stacking
requires the thermal prior to supply an uncertainty raster or column so its
continuous predictive density is defined.
Incremental posterior-block storage is not currently available with Gaussian
components or predictive stacking; those combinations fail during config
validation instead of silently omitting either operation.

### `calibration`

| Key | Default | Purpose |
| --- | --- | --- |
| `method` | `"none"` | `"platt"`, `"isotonic"`, `"temperature"`, `"none"`. GBLK currently requires `"none"`. |
| `fit_on` | `"block_cv"` | Calibration maps are fit only from held-out block-CV predictions. |
| `report_temperature` | `true` | Include diagnostic temperature in reports. |
| `n_bins` | `5` | Equal-frequency reliability bin count. |

### `cross_validation`

| Key | Default | Purpose |
| --- | --- | --- |
| `n_folds` | `5` | Number of spatial-block folds. |
| `block_type` | `"grid"` | `"grid"` or `"kmeans"`. |
| `block_size_km` | `null` | Spatial block width in kilometres; when `null`, use the configured grid partition. |
| `grid_size` | `4` | Block-grid side length (for `block_type="grid"`). |
| `buffer_km` | `0` | Exclude training sites within this distance of each held-out block; never relabel buffered sites. |

### `scenarios`

A list of ablation scenarios; each is `{name, include_priors, include_spatial, drop_layers}`. The runner produces per-scenario per-component surfaces under `result.scenarios[name]`.

### `outputs`

| Key | Default | Purpose |
| --- | --- | --- |
| `probability_rasters` | `true` | Emit spatial GeoTIFF/VTK probability products. CSV and Parquet tables are controlled directly by `format`. |
| `uncertainty_rasters` | `true` | Also emit posterior-std rasters when available (GP backend). |
| `calibration_artifacts` | `true` | When top-level spatial OOF calibration is run, emit strict-JSON metrics and reliability rows. |
| `decision_artifacts` | `true` | When spatial OOF predictions are available, emit raw-OOF decision-class and top-N summaries. |
| `scenarios` | `true` | Emit each configured scenario beneath `output_dir/scenarios/<name>/`; scenarios remain available in memory when this is false. |
| `posterior_draw_blocks` | `false` | Persist paired, decomposed probability draws for Bayesian posterior, prior-predictive, or mixed runs beneath `output_dir/posterior_draws/`. All-fixed runs are rejected because replicated constants are not uncertainty draws. |
| `posterior_draw_block_size` | `20` | Number of posterior draws projected and persisted per compressed block; bounds draw-by-grid working memory without changing the estimand. |
| `format` | `["geotiff", "csv"]` | Any subset of `geotiff` / `csv` / `parquet` / `vtk`. |

Draw blocks are available only when `inference.backend="gblk"`,
`inference.gblk_bayesian.enabled=true`, and
`inference.gblk_bayesian.cluster_effect=false`. The last condition prevents
INLA from materializing the complete draw-by-grid predictor. geoPFA instead
persists the immutable field/fixed/prior coefficient draws and prediction
design, then projects one configured draw block at a time. A restart reopens
that exact hash-verified state before fitting; it never appends draws from a
second posterior fit.

The schema-version-2 `index.json` fixes the run/scenario scope, component order,
coordinate columns and CRS, posterior seed, draw IDs, combination rule, and
SHA-256 hash of every coordinate, state, and draw payload. Each `.npz` block
contains `prior_logit`, `evidence_logit`, `spatial_logit`,
`component_probability`, and `combined_probability`. The writer and
`verify_posterior_draw_bundle` independently verify
`component_probability = logit^{-1}(prior + evidence + spatial)` and the
configured within-draw product or geometric mean. Its
`uncertainty_semantics` is derived from the component roles and distinguishes
posterior, prior-predictive, and mixed draws. Exact draw means and
quantiles are reconstructed through disk-backed cell chunks, not a resident
full draw cube.

These posterior or prior-predictive draws are coupled model states, not
independent Sobol factors.
Sensitivity analyses must keep a whole draw fixed, use an identified independent
innovation representation, or otherwise document a valid dependent-input
estimand. Configured scenarios are fitted and persisted in separate namespaces
under `output_dir/scenarios/<name>/posterior_draws/`. Their indexes explicitly
declare cross-scenario pairing unidentified; equal integer draw IDs do not imply
a common-innovation coupling across separately fitted scenarios.

> Note: comparing the probabilistic surface against the deterministic
> `VoterVeto` favorability is a separate workflow. Run `VoterVeto.do_voter_veto`
> directly on the same `pfa` dict and use the existing `geopfa.geopfa2d.plotters`
> /`geopfa.geopfa3d.plotters` helpers (or your own) to build a side-by-side
> comparison. The probabilistic runner intentionally does not orchestrate the
> favorability path.

## Calibration interpretation

The runner exposes `geopfa.prob.calibration_summary(y, p, n_bins=5)` (already merged) returning a dict with both raw and post-temperature ECE / MCE / Brier / log-loss, plus the fitted scalar temperature `T`. **Temperature scaling is reported as a diagnostic only; the runner does not apply it to map outputs.** Use it as a summary number:

- `T < 1` → predictions were *under-confident* (true rate higher than predicted at the high bins).
- `T > 1` → predictions were *over-confident*.
- `T ≈ 1` → already well-calibrated.

For GBLK, keep `calibration.method = "none"` in the map-runner config and call
`geopfa.prob.gblk_runner.run_gblk_calibration_cv` for explicit raw spatial-block
CV diagnostics. The top-level GBLK runner fails closed if a post-hoc method is
configured because it does not yet apply that map to GBLK outputs. The generic
`fit_posthoc_calibration(p_oof, y_oof, method=...)` primitive remains available
for workflows that explicitly construct and preserve out-of-fold predictions.

## Decision diagnostics

`geopfa.prob.decision_class_table` returns the per-bin breakdown (decision class, well count, positive count, observed positive rate, intuitive % correct, bin value bounds) using **full-surface percentiles** for the cut points (not the well-sample percentiles). This is the diagnostic value: each bin is defined by where the surface sits, not where the wells sit, so the table tells you what the labelled wells say about each region of the surface.

`geopfa.prob.top_n_targeting` computes the cumulative top-N targeting curve
with random and oracle baselines. If the requested budget crosses a tied-score
group, reported hits are the expected hits under uniform random selection
within that boundary group; this makes the diagnostic invariant to row order.

`geopfa.prob.confusion_at_threshold` is the classical 0.5-threshold confusion matrix with accuracy, precision, recall, and specificity.

## Plotters

Region-agnostic 2D plotter helpers in `geopfa.prob.plotting`:

- `plot_component_panel(surface, name, path, style=None)` — 2×2 panel (prior, evidence, spatial residual, final).
- `plot_well_overlay(surface, wells, label_col, path, style=None)` — combined surface with labelled wells.
- `plot_reliability_diagram(y, p, path, n_bins=5, style=None)` — reliability diagram with Wilson CIs.
- `plot_decision_class_bar({method_name: probs}, labels, path, style=None)` — decision-class bar chart for arbitrary number of methods.
- `plot_top_n_curve(y, p, path, style=None)` — top-N targeting curve with random + oracle baselines.
- `plot_confusion_matrix(y, p, threshold=0.5, path, style=None)` — 2×2 heatmap.

Every helper accepts a `PlotStyle` dataclass for theme overrides (cmaps, point colours, CI color). Applications can subclass `PlotStyle` to inject NREL / corporate palettes without touching the plotter logic.

## 3D usage

Set `dimensions: "3d"` in the config. The runner expects the PFA dict to carry 3D `Point(x, y, z)` geometries on every `pr_norm` and layer `model`. The labelled wells file must include a depth column referenced by `labels.z_col` (or named `depth_m` by default). All other knobs work the same way.

The GBLK backend fits a genuine 3-D LKBox field using `(x, y, z)` point
geometries in both prediction and labelled-well inputs. For numerical
conditioning, the union of training and prediction coordinates is range-scaled
to a unit cube before basis construction; the transform is recorded in fit
diagnostics. Predictions can therefore vary with depth even when offsets are
constant.

Outputs in 3D mode include a `.vtp` file per component for direct loading into PyVista / ParaView.

## Public API

```python
from geopfa.prob import (
    # config + entry point
    ProbabilisticConfig,
    load_probabilistic_config,
    run_probabilistic,

    # priors
    AlphaCResult,
    build_alpha_c,
    write_alpha_provenance,

    # evidence + dispatch
    fit_component_probability,        # legacy direct API (deprecated, sequential path)
    fit_component_from_config,        # config-driven thin wrapper (sequential path)
    build_fit_kwargs,                 # translator (pure)
    combine_probability_surfaces,     # legacy utility for component diagnostics

    # spatial field (sequential backend only, deprecated)
    SpatialFieldResult,
    fit_spatial_field_gp,

    # validation
    SpatialBlockKFold,
    spatial_block_cv,
    CalibrationMap,
    fit_posthoc_calibration,
    calibration_summary,

    # decision metrics
    confusion_at_threshold,
    decision_class_table,
    top_n_targeting,
    auc_tie_safe,

    # outputs
    write_probability_outputs,
    write_geotiff_outputs,
    write_parquet_outputs,
    PosteriorDrawBlockWriter,
    load_posterior_draw_state,
    verify_manifest,
    verify_posterior_draw_bundle,
    write_vtk_outputs,
    write_manifest,

    # plotting
    PlotStyle,
    plot_component_panel,
    plot_well_overlay,
    plot_reliability_diagram,
    plot_decision_class_bar,
    plot_top_n_curve,
    plot_confusion_matrix,
)
```

## Migration from the Stage-1 demo and the sequential backend

See `docs/migration_guide.md` for full migration instructions. In brief:

- If you previously called `fit_component_probability` directly, it still works unchanged.
- If you used `inference.backend="sequential"` (or relied on the old default), set
  `inference.backend="gblk"` in your config. The `ProbabilisticResult` schema is identical.
- For GBLK, set `spatial_field.enabled`, `n_levels`, and
  `lattice_centers_per_dimension` explicitly when the default lattice is not
  appropriate for the available spatial-label support.

```python
from geopfa.prob import (
    ProbabilisticConfig,
    AlphaModeConfig,
    LabelsConfig,
    InferenceConfig,
    EvidenceConfig,
    OutputsConfig,
    run_probabilistic,
)

cfg = ProbabilisticConfig(
    enabled=True,
    output_dir=Path("outputs/probabilistic"),
    dimensions="2d",
    labels=LabelsConfig(
        source="data/wells.gpkg",
        id_col="well_id",
        label_columns={"heat": "heat_label"},
        layer="wells",
    ),
    alpha={
        "heat": AlphaModeConfig(
            mode="layer_logit", layer="heat_source_t3km", scalar_fallback_pr0=0.55
        ),
    },
    evidence=EvidenceConfig(),
    inference=InferenceConfig(backend="gblk"),
    outputs=OutputsConfig(format=("geotiff", "csv")),
)
result = run_probabilistic(pfa, cfg)
```

The result is a `ProbabilisticResult` with `result.combined` as the joint resource
co-occurrence surface; `result.components[name].probability` holds per-component
probability surfaces.

## Limitations and roadmap

The GBLK method is implemented and is the default backend. Current limitations:

- **3D plotter helpers** — depth-slice panel, isosurface via
  `ConceptualModeling.plot_isosurface`, vertical cross-section.
- **GBLK post-hoc calibration application** — raw block-CV diagnostics are
  available, but calibrated GBLK map export is not yet implemented.
- **Joint upstream uncertainty** — the sole Bayesian GBLK path jointly samples
  evidence coefficients and the spatial field. Physics-informed alpha offsets
  enter training and every prediction draw as fixed inputs, so posterior
  uncertainty remains conditional on their construction and on the supplied
  evidence covariates.
- **nnPU with GBLK** — nnPU is currently available only through the sequential
  outcome fitter. GBLK rejects that configuration rather than treating
  unlabeled candidates as negatives.

All provided Pixi environments include the exact pinned LatticeKrigX revision
and `scikit-sparse`. Authenticated developers can therefore run both the
default `inference.backend="gblk"` path and
`inference.gblk_bayesian.enabled=true` from a standalone checkout. Public
installation requires the pinned LatticeKrigX revision to be published first.

The legacy `inference.backend="sequential"` path is a per-component penalized
logistic model with a standalone RBF or LatticeKrig spatial smoother. It is
deprecated and retained only for explicitly supported diagnostics such as nnPU.

---

## Recent updates

### GBLK is now the default backend

`inference.backend` defaults to `"gblk"`. The GBLK method fits all components
jointly via `latticekrigx.glk.joint.fit_joint`, replacing the sequential
per-component logistic plus standalone spatial-smoother path. No config change
is needed if you did not previously set `inference.backend` explicitly.

The `sequential` backend is deprecated: a `DeprecationWarning` is emitted when
it is used. Remove `inference.backend="sequential"` from any existing configs.

### `run_probabilistic_pfa(pfa)` — single-file entry point

The recommended way to run the workflow when your geoPFA dict already
contains a ``"probabilistic"`` config block:

```python
from geopfa.prob import run_probabilistic_pfa

# pfa must contain pfa["probabilistic"] = { ... config ... }
result = run_probabilistic_pfa(pfa)
```

This is equivalent to `ProbabilisticConfig.from_pfa(pfa)` + `run_probabilistic(pfa, cfg)`.

### `ProbabilisticConfig.from_pfa(pfa)` — pfa-dict integration

```python
cfg = ProbabilisticConfig.from_pfa(pfa)  # reads pfa["probabilistic"]
```

### `ProbabilisticConfig.validate()` / `validate_raise()`

```python
errors = cfg.validate()       # returns list[str]; empty = valid
cfg.validate_raise()          # raises ValueError on first error
```

Checks performed by `validate()`:
- `output_dir` is non-empty
- `alpha` dict is non-empty; `scalar_fallback_pr0` in (0, 1)
- `labels.min_wells_for_fit >= 2`
- `cross_validation.n_folds >= 2`
- Bayesian GBLK prior scales, draw count, and interval level are in valid ranges
- Cross-field constraints between labels, priors, and resource-model settings

`validate_raise()` is called automatically at the start of
`run_probabilistic()` (after the `enabled=False` short-circuit check) and
inside `load_probabilistic_config()`.

### `ComponentProbability.diagnostics`

A ``dict | None`` field holding per-fit metadata. Currently populated fields:

| Key | When present | Description |
|---|---|---|
| `pu_mode` | sequential well path | the PU mode used |
| `n_train` | sequential well path | number of training sites |
| `pu_class_prior` | sequential nnPU fit | externally supplied population prevalence |
| `pu_positive_risk`, `pu_negative_risk`, `pu_raw_negative_risk` | sequential nnPU fit | empirical-risk decomposition |
| `backend` | Bayesian GBLK path | always `"gblk_bayesian"` |
| `estimator` | Bayesian GBLK path | `"paige_inla"` |
| `posterior_scope` | Bayesian GBLK path | states whether evidence coefficients and the spatial field are sampled jointly; alpha construction remains conditioned upon |
| `n_draws` | Bayesian GBLK path | number of paired posterior draws |

Access: `result.components[name].diagnostics`.
