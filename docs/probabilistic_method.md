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
The config-driven workflow treats each label-table row as a point observation
at location $s_i$. For a binary component $q$ (e.g. reservoir or seal):

$$
\text{logit}\, p_{qi} \;=\; o_q(s_i) \;+\; Z_{qi}\, d_q \;+\; B(s_i)^\mathsf{T}g_q
$$

where:

- **$o_q(s_i)$ — physics-informed prior offset.** Logit-scale prior constructed by
  `geopfa.prob.alpha.build_alpha_c` from domain constraints (thermal exceedance,
  layer logit, scalar, or multi-layer combination).
- **$Z_{qi}\, d_q$ — evidence regression.** Evidence-layer values at the well
  multiplied by component-specific coefficients. The empirical-Bayes GBLK path
  estimates these coefficients with the spatial field in one likelihood under
  the declared Gaussian coefficient prior. Regional hierarchical coefficients
  are not currently implemented in the joint likelihood.
- **$B(s_i)^\mathsf{T}g_q$ — latent spatial field.** The multiresolution
  LatticeKrig basis is evaluated at the row's point location and multiplied by
  latent field coefficients $g_q$. All components share the same LatticeKrig
  basis; cross-component dependence is captured by the estimated $Q \times Q$
  correlation matrix $\Omega$ within a same-family fit.

For a Gaussian heat component, the corresponding model is

$$
T_i / a_T = \mu_T(s_i) / a_T + Z_{Ti}d_T + B(s_i)^\mathsf{T}g_T + \epsilon_i,
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

By default, every observed row contributes one ordinary likelihood term. A
component may instead name a positive observation-weight column and explicitly
declare `generalized_bayesian_power_likelihood`. In that case geoPFA and
LatticeKrigX pass the row-aligned weights $w_{qi}$ to R-INLA, and the likelihood
part of the target is

$$
\sum_{q,i:\,y_{qi}\ \mathrm{observed}} w_{qi}\log p(y_{qi}\mid\theta).
$$

R-INLA does not recompute the likelihood normalizing constant for fractional
weights. Non-unit values therefore define a power likelihood and generalized
Bayesian posterior, not an ordinary heteroskedastic or within-well Gaussian
sampling model. Any fitted Gaussian precision under this contract is a working
likelihood precision under the declared generalized posterior. The config must
state the semantics explicitly; geoPFA records the observed weight count,
minimum, maximum, and sum and rejects a semantics/values mismatch. Balancing
rows can control repeated-profile influence, but it does not estimate
within-well covariance.

Gaussian component tables report `response_predictive_mean`,
`response_predictive_lo`, and `response_predictive_hi` in physical response
units. For a fitted component, the posterior-predictive interval includes
Gaussian likelihood variation and latent-field uncertainty. For a component
declared with `force_prior_predictive=true`, the interval is the analytic
Normal interval from the configured thermal mean and required thermal-model
standard deviation; it contains no outcome update or fitted-likelihood
variance. If predictive stacking is enabled for a fitted component, these
summaries describe the same selected mixture of the configured prior
predictive distribution and fitted posterior predictive distribution used for
the component event probability.

A temperature profile is represented by one point-observation row per sampled
depth. Rows from the same physical well repeat the configured well identifier,
which keeps the complete profile together in grouped spatial validation. The
canonical config workflow does not currently expose interval, trajectory, or
areal observation supports. Low-level support-quadrature utilities remain
available for direct development use, but they are not connected to
`run_probabilistic`.

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
| evidence $Z_q d_q$ | processed evidence layers | design matrix + declared Gaussian coefficient prior |
| latent field $g_q$, $\Omega$ | — | `glk.multivariate` + `fit_joint` |
| observation geometry | point rows; profiles are depth-specific rows grouped by well id | basis evaluated at each point |
| PU / undrilled unlabeled | `pu.py` (non-negative PU risk; sequential outcome path only) | not part of the GBLK likelihood |
| preferential site collection | `site_selection.py` (finite-candidate sensitivity model) | deliberately separate from LatticeKrigX |
| plug-in co-occurrence | `result.combined` | product of joint-fit marginals |
| calibration (Brier/BSS, spatial-block CV) | `calibration.py`, `cv.py` | `glk.calibration` |
| sole Bayesian option | `gblk_bayesian.enabled=true` | public Paige/INLA joint draws for 2-D or 3-D geometry |

## Quick start

The whole pipeline is driven by a single JSON config. Drop a `probabilistic` block into your existing PFA config:

```javascript
{
  "criteria": { ... },             // your existing PFA config (unchanged)
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
        "enabled": true
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
pixi run -e dev-gblk geopfa-prob run \
  --config path/to/config.json \
  --processed-data-dir path/to/processed/data \
  --crs EPSG:26911
```

The processed-data directory must contain the standard
`criteria/component/*_processed.csv` layer tree. The CRS is explicit because
CSV geometry does not carry that metadata. Relative `output_dir`, label,
thermal, uncertainty, and candidate-frame paths in the JSON are resolved
relative to the config file, not the shell's working directory. The manifest
binds the config and every consumed processed layer by byte count and SHA-256,
along with the effective config and the geoPFA/LatticeKrigX runtime identities.
Completed output namespaces are immutable and are never accepted for another
execution, even when their manifests still verify. Only a verified incomplete
posterior workspace with a valid progress record may resume its completed
block prefix. Use a fresh directory when the config or implementation changes,
or when abrupt termination leaves an incompletely published state or derived
product, so artifacts from different runs cannot mix.
The runner also compares its start/end runtime fingerprints and rejects an
artifact if either source tree changes while the model is executing.

That call:

1. When an outcome update or site-selection analysis is configured, loads
   point-observation rows from the configured source (GeoPackage / shapefile /
   CSV). A profile contains repeated depth-specific rows with a shared well id.
   An all-prior-predictive run does not require a label source; omit it so the
   manifest does not bind unused data.
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
8. Writes only the formats selected by `outputs.format` to `output_dir/`:
   GeoTIFF for 2-D grids, VTK for 3-D point volumes, and CSV or Parquet for
   either dimension. Optional products include paired Bayesian probability
   blocks. The run also writes `alpha_provenance.json` and `manifest.json`
   (input/output SHA-256 hashes + config and implementation hashes + run id).

## Study notebooks

The three probabilistic study notebooks start from the standard processed
config and `*_processed.csv` layer tree. They do not repeat the educational
cleaning and feature-processing walkthroughs in the voter-veto notebooks. Each
notebook runs `VoterVeto` on those processed inputs, runs the probabilistic
model, and compares the two final surfaces (`pfa_vv["pr_norm"]` and
`result.combined`). Generated outputs remain outside Git.

Local paths use documented defaults and can be overridden without editing the
notebooks:

| Input | Environment variable |
| --- | --- |
| output root shared by all demos | `GEOPFA_DEMO_OUTPUT_ROOT` |
| Newberry processed config and layer tree | `GEOPFA_NEWBERRY_PROCESSED_CONFIG`, `GEOPFA_NEWBERRY_PROCESSED_DATA` |
| Nevada processed config and layer tree | `GEOPFA_NEVADA_PROCESSED_CONFIG`, `GEOPFA_NEVADA_PROCESSED_DATA` |
| Nevada conventional labels | `GEOPFA_NEVADA_LABELS` |
| Nevada 3-km thermal mean and standard deviation | `GEOPFA_NEVADA_3KM_MEAN`, `GEOPFA_NEVADA_3KM_SD` |
| Nevada 7-km thermal mean and standard deviation | `GEOPFA_NEVADA_7KM_MEAN`, `GEOPFA_NEVADA_7KM_SD` |

The Newberry notebook is an illustrative run. It retains full-grid posterior
summaries but disables raw full-grid draw blocks. Large analyses should choose
a retained-draw cell subset or another storage contract before execution.

## Config reference

The complete schema is enforced by `ProbabilisticConfig`; the principal fields
are summarized below.

### `labels`

| Key | Default | Purpose |
| --- | --- | --- |
| `source` | required for fitted or site-selection paths | Path to a GPKG / SHP / CSV holding labelled wells. |
| `id_col` | required for fitted or site-selection paths | Unique-id column on the wells file. |
| `layer` | `null` | Sub-layer for GPKG sources. Rejected for CSV sources. |
| `source_crs` | `null` | CRS of CSV coordinates; required for CSV. For vector files, this may supply missing CRS metadata but must agree with an embedded CRS. |
| `x_col`, `y_col` | `null` | Cartesian coordinate columns; both are required for CSV and are not used for vector sources. |
| `z_col` | `null` | Optional Cartesian model-Z column for CSV. Values are used unchanged. |
| `depth_col` | `null` | Optional nonnegative, positive-down scientific depth. If CSV `z_col` is omitted, model geometry uses `z = -depth`; if both are present, `z_col` controls geometry and depth remains available for target-depth validation. |
| `label_columns` | required for fitted or site-selection paths | Per-component label-column mapping (e.g. `{"heat": "heat_label", "reservoir": "reservoir_label"}`). |
| `observation_models` | `{}` | Optional per-component response family. Omitted labelled components are Bernoulli. A fitted Gaussian component requires `family="gaussian"` and a positive `response_scale`. A Gaussian component with `force_prior_predictive=true` is already in physical units, must declare thermal response uncertainty, and omits `response_scale`. An alpha-only declaration is valid only when that component sets `force_prior_predictive=true`. |
| `observation_weight_columns` | `{}` | Optional per-component mapping to positive finite row weights. The weights are used only at observed responses and require Bayesian GBLK inference. Non-unit weights define a generalized Bayesian power likelihood. |
| `observation_weight_semantics` | `null` | Required when weight columns are configured. Use `generalized_bayesian_power_likelihood` for non-unit weights; unit observed weights require `ordinary_bayesian_likelihood`. The runner fails closed if the declared semantics and realized observed values differ. |
| `min_wells_for_fit` | `4` | Minimum observed support required by the selected fit; insufficient support raises rather than silently changing models. |
| `pu_mode` | `"off"` | `"off"`, `"naive_pseudo_absence"`, or modern non-negative PU risk estimation (`"nnpu"`; sequential outcome path only). |
| `pu_class_prior` | `null` | Externally identified population prevalence required by `pu_mode="nnpu"`; a scalar or per-component mapping with values strictly inside `(0, 1)`. |

The `labels` block may be omitted when every component sets
`force_prior_predictive=true` and `site_selection.mode="off"`. A prior-only
Gaussian component can provide only its `observation_models` entry; no label
source is read or recorded as an input in that case.

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
envelope. Only Bernoulli components enter this binary analysis; Gaussian heat
observations remain in the continuous outcome model. This analysis does not
alter or wrap the LatticeKrigX likelihood.

### `alpha[component]`

| Mode | Required keys | What it does |
| --- | --- | --- |
| `scalar` | `scalar_fallback_pr0` | Uniform $\alpha = \text{logit}(p_0)$. |
| `layer_logit` | `layer` | Min-max rescale a named layer to $[p_{\min}, p_{\max}]$ and apply `logit`. Layer is auto-excluded from regression. |
| `thermal_exceedance` | `thermal_raster`, `threshold` (`uncertainty_raster` required for a Gaussian prior-only component) | For 2-D runs, compute a step function or $\Phi$-integration of `P(T > T*)` from a raster. Clipped then `logit`. |
| `thermal_layer_exceedance` | `layer`, `threshold` (`uncertainty_column` required for a Gaussian prior-only component) | Compute `P(T > T*)` from temperature mean/SD columns already carried by a PFA layer. |
| `multi_layer` | `layers` | Sum of `layer_logit` offsets in logit space; all named layers auto-excluded. |

`p_min` and `p_max` must be finite and strictly inside `(0, 1)` so every
logit offset is finite. Layer and multi-layer rescaling defaults to
`[0.2, 0.8]`. Thermal exceedance defaults to the near-open finite interval
`[1e-12, 1 - 1e-12]` so a computed thermal probability is not materially
truncated. `force_prior_predictive=true` explicitly prevents an outcome
update. For a Bernoulli component, `use_evidence_prior=true` samples
named Gaussian coefficient priors on the event-logit scale; otherwise the
component is a fixed prior prediction and does not produce replicated
uncertainty draws. A Gaussian prior-only component rejects
`use_evidence_prior` because that logit update is not a continuous-response
model. Fields outside the selected mode's row in the table are rejected rather
than ignored.

### `evidence`

| Key | Default | Purpose |
| --- | --- | --- |
| `regularization.C` | `1.0` | Sklearn-style `1/lambda`. |
| `regularization.per_feature_weights` | `{}` | Per-layer L2 weight overrides. |
| `regularization.prior_means`, `prior_precisions` | `{}`, `{}` | Named Gaussian coefficient priors, required for prior-predictive evidence coefficients. |
| `include_layers` | `null` | Whitelist (default: all non-excluded). |
| `exclude_layers` | `[]` | Blacklist; alpha layers are auto-added. |
| `sparse_binary_threshold` | `0.90` | Reject layers where ≥90% of cells share one value. |
| `coordinate_blacklist` | sensible default | Coordinate-like column names that must never enter the design matrix. |
| `standardization` | `"observed_labels"` | Fit transformations on observed training rows; `"prediction_support"` is an explicit prior-predictive option when no outcome-trained transformation exists. |
| `feature_expansions` | `{}` | Optional component-keyed quadratic and interaction terms. Each component may set `degree` to 1 or 2, enable pairwise evidence interactions, add selected model-coordinate axes, and interact evidence with those axes. |

Feature expansions are explicit because a quadratic design can grow quickly.
For example, the following adds squared evidence terms and pairwise evidence
interactions to `hydraulic`, while `heat` receives a quadratic vertical trend
and evidence-by-Z interactions:

```json
{
  "feature_expansions": {
    "hydraulic": {
      "degree": 2,
      "include_pairwise_interactions": true
    },
    "heat": {
      "coordinate_degree": 2,
      "coordinate_axes": ["z"],
      "include_evidence_coordinate_interactions": true
    }
  }
}
```

`degree` controls evidence powers; `coordinate_degree` separately controls
coordinate powers. Generated terms enter the same Gaussian coefficient-prior contract as their
source layers. They are standardized inside each training fold, so held-out
outcomes and held-out feature distributions do not define the fitted scaling.
Coordinate terms use the Cartesian model coordinates (`x`, `y`, and, in 3-D,
`z`); `z` is not silently interpreted as positive-down scientific depth.
Generated feature names are recorded in diagnostics and can receive explicit
regularization overrides.

### `spatial_field`

The GBLK backend uses these controls to define its joint LatticeKrig field.
The deprecated sequential path accepts the backend toggle but does not expose
unsupported sparse-GP tuning vocabulary.

| Key | Default | Purpose |
| --- | --- | --- |
| `enabled` | `true` | Toggle `u_c`. |
| `backend` | `"latticekrigx"` | `"latticekrigx"` for GBLK, `"rbf"` for the lightweight sequential smoother, or `"none"`. |
| `n_levels` | `2` | Number of multiresolution LatticeKrig levels. In 3D, the basis is constructed jointly over x, y, and z. |
| `lattice_centers_per_dimension` | `6` | Coarsest-level centers per spatial dimension for GBLK. Choose this before outcome evaluation and keep the resulting basis commensurate with the effective training sample. |
| `coordinate_scaling` | `"axis_range"` | `"axis_range"` gives domain-relative axes; `"physical_isotropic"` preserves metre-scale axis ratios and requires commensurate coordinate units. |
| `spatial_domain` | omitted | Optional physical-coordinate bounds written as `[[x_min, x_max], [y_min, y_max]]` or the corresponding three-axis array. When supplied, every final, blocked, and buffered fit uses this exact transform and lattice domain. All training and prediction points must lie inside it. |

By default, geoPFA derives the domain from the assembled observations and
prediction grid. That is convenient for a single fit but is inappropriate when
omission scenarios must share one prior and basis. Such analyses should freeze
`spatial_domain` before fitting and reuse it unchanged in every scenario. With
`coordinate_scaling="physical_isotropic"`, geoPFA divides all axes by the
largest horizontal span. Length-scale priors are therefore specified in those
model coordinates. For example, a 5,000 m coarsest-level ELK-F range-proxy
median on a domain whose largest horizontal span is 30,000 m is configured as
`cor_scale_median=5000/30000`, not `5000`.

### `inference`

| Key | Default | Purpose |
| --- | --- | --- |
| `backend` | `"gblk"` | `"gblk"` (default, joint GBLK method) or `"sequential"` (**deprecated**). |
| `gblk_bayesian.enabled` | `false` | Select the sole Bayesian dispatch: public Paige/INLA. |
| `gblk_bayesian.n_draws` | `200` | Number of paired posterior draws. |
| `gblk_bayesian.seed` | `0` | RNG seed for reproducible draws. |
| `gblk_bayesian.ci_level` | `0.9` | Credible-interval level for Bayesian probability surfaces and the analytic response interval returned by a fixed Gaussian prior, including when Bayesian fitting is disabled. |
| `gblk_bayesian.cor_scale_median` | `0.1` | Paige coarsest-level ELK-F range-proxy prior median in model coordinates. Under physical-isotropic scaling, multiply by the recorded common coordinate scale to recover the physical value. Finer-level ELK-F ranges halve with lattice width. |
| `gblk_bayesian.spatial_sd_u` | `1.0` | Paige spatial standard-deviation threshold. |
| `gblk_bayesian.spatial_sd_tail_probability` | `0.05` | Prior probability above `spatial_sd_u`. |
| `gblk_bayesian.dirichlet_concentration` | `1.5` | Symmetric Paige level-weight concentration. |
| `gblk_bayesian.kleiber_profiles` | `{}` | Frozen `r0`/`r1` profile keyed by likelihood family (`bernoulli` or `gaussian`). Each bivariate family requires its own profile; a univariate family must omit one. Bayesian GBLK supports at most two fitted components per family. |
| `predictive_stacking.enabled` | `false` | Select a component-specific mixture of the configured event prior and full Bayesian update by buffered or blocked out-of-fold logarithmic score. Zero retains the prior and one retains the full update. |
| `predictive_stacking.validation_depths_m` | `{}` | In a 3-D analysis, optionally map component names to positive-down target depths. Each mapped component selects its stacking weight only from held-out observations at that depth. If fewer than `labels.min_wells_for_fit` distinct wells occur there, that component retains its prior. Unmapped components use all of their held-out observations. |
| `predictive_stacking.minimum_training_wells` | `null` | Optional positive minimum for each fold's training support. `null` uses `labels.min_wells_for_fit` unchanged. |

Predictive stacking uses the spatial split declared in `cross_validation` and
never scores in-sample predictions. Bernoulli components use held-out binary
log score. Gaussian components use the held-out continuous posterior predictive
log density, including the sampled residual precision, rather than discarding
information by thresholding the observations. The selected distribution
mixture induces the same mixture of event probabilities passed to component
combination. The selected weight and the prior, full, and selected held-out log
scores are recorded in component diagnostics. Stacked component tables also
retain `probability_prior` and `probability_full_update`, the two unstacked
mean event-probability surfaces, so users can audit the selected mixture. The
held-out evidence for a Gaussian component retains both continuous predictive
log densities and target-event probabilities. A component-specific validation
depth aligns this model-selection step with a target-depth map while the
Gaussian fit can still use complete temperature profiles and other components
retain their appropriate validation support. Gaussian predictive stacking
requires the thermal prior to supply an uncertainty raster or column so its
continuous predictive density is defined. If spatial blocking or buffering
leaves any fold below the declared training-well support, geoPFA does not score
an incomplete out-of-fold prediction set. It retains the affected family
prior, records `prior_retained_incomplete_spatial_cv`, and preserves the
missing predictions in the stacking evidence.
Incremental posterior-block storage supports a fitted Gaussian family with
paired latent-mean and likelihood-precision draws, including mixed products
with fixed prior-only Bernoulli components. Predictive stacking remains
incompatible with posterior blocks and fails during config validation rather
than silently omitting either operation. One posterior-block run may contain
only one fitted response family; separate fitted Gaussian and Bernoulli
families must be run separately because their posterior innovations are not
jointly identified.

### `calibration`

| Key | Default | Purpose |
| --- | --- | --- |
| `method` | `"none"` | `"platt"`, `"isotonic"`, `"temperature"`, `"none"`. GBLK currently requires `"none"`. |
| `fit_on` | `"block_cv"` | Calibration maps are fit only from held-out block-CV predictions. |
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
| `posterior_draw_cell_indices_source` | `null` | Optional `.npy` file containing a non-empty, strictly increasing subset of zero-based full-grid cell indices to materialize in every draw block. Exact full-grid posterior means and intervals are still accumulated and persisted. |
| `posterior_draw_cell_indices_sha256` | `null` | Required lowercase SHA-256 digest of `posterior_draw_cell_indices_source`. The run rejects a mismatch before fitting or resuming. |
| `format` | 2-D: `["geotiff", "csv"]`; 3-D: `["vtk", "csv"]` | GeoTIFF is 2-D only and VTK is 3-D only. CSV and Parquet support either dimension. |

Draw blocks are available only when `inference.backend="gblk"`,
`inference.gblk_bayesian.enabled=true`, and
`inference.gblk_bayesian.cluster_effect=false`. The last condition prevents
INLA from materializing the complete draw-by-grid predictor. geoPFA instead
persists the immutable field/fixed/prior coefficient draws and prediction
design, then projects one configured draw block at a time. After an ordinary
execution error, only a verified incomplete posterior workspace can resume; it
reopens the exact hash-verified state and completed block prefix before
projecting the missing draw blocks.
It never appends draws from a second posterior fit. Abrupt termination inside
state or final-output publication is rejected rather than repaired or silently
mixed.

The posterior `index.json` fixes the run/scenario scope, component order,
coordinate columns and CRS, posterior seed, draw IDs, product-combination
contract, and SHA-256 hash of every coordinate, state, draw, and summary
payload. Each `.npz` block contains the family-appropriate prior, evidence,
and spatial decomposition plus component and combined probabilities. Internal
validation checks the predictor decomposition, Gaussian exceedance calculation
where applicable, and the within-draw product. Its
`uncertainty_semantics` is derived from the component roles and distinguishes
posterior, prior-predictive, and mixed draws. Exact draw means and
quantiles are reconstructed through disk-backed cell chunks, not a resident
full draw cube.

These posterior or prior-predictive draws are coupled model states, not
independent Sobol factors.
Sensitivity analyses must keep a whole draw fixed, use an identified independent
innovation representation, or otherwise document a valid dependent-input
estimand. When `outputs.scenarios=true`, configured scenario products are
persisted in separate namespaces under `output_dir/scenarios/<name>/`. Paired
draw blocks appear beneath each namespace only when
`outputs.posterior_draw_blocks=true`. Their indexes explicitly declare
cross-scenario pairing unidentified; equal integer draw IDs do not imply a
common-innovation coupling across separately fitted scenarios. When scenario
output is disabled, the scenario surfaces remain in memory and no scenario
artifacts are written.

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

For deterministic Bernoulli GBLK, keep `calibration.method = "none"` in the
map-runner config and call
`geopfa.prob.run_gblk_calibration_cv` for explicit raw spatial-block
CV diagnostics. That function derives its lattice size and reliability-bin count
from the config and rejects conflicting call-time overrides. Bayesian and
Gaussian GBLK calibration is not implemented by this helper and fails closed.
The GBLK configuration also fails closed if a post-hoc method is configured
because the map runner does not apply one to GBLK outputs. The generic
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

Every helper accepts a `PlotStyle` dataclass for theme overrides (cmaps, point colours, CI color). Applications can subclass `PlotStyle` to inject project-specific palettes without touching the plotter logic.

## 3D usage

Set `dimensions: "3d"` in the config. The runner expects the PFA dict to carry 3D `Point(x, y, z)` geometries on every `pr_norm` and layer `model`. For CSV labels, declare either Cartesian `labels.z_col`, positive-down `labels.depth_col`, or both. With depth alone the loader constructs `z = -depth`; with both, Cartesian Z defines model geometry and the separate depth column supports scientific target-depth validation. A 3-D thermal prior must use `alpha.mode="thermal_layer_exceedance"` with a processed 3-D PFA layer. The raster-backed `thermal_exceedance` mode samples only x and y, so config validation rejects it for 3-D runs rather than repeating one 2-D prior through every depth.

The GBLK backend fits a genuine 3-D LKBox field using `(x, y, z)` point
geometries in both prediction and labelled-well inputs. For numerical
conditioning, the union of training and prediction coordinates is range-scaled
to a unit cube before basis construction; the transform is recorded in fit
diagnostics. Predictions can therefore vary with depth even when offsets are
constant.

When `vtk` is selected for a 3-D run, outputs include a `.vtp` point-cloud file
per component for direct loading into PyVista / ParaView.

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
    run_gblk_calibration_cv,
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
    write_vtk_outputs,

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

- **3D visualization scope** — the helper API provides static depth slices and
  vertical cross-sections. Interactive volume rendering remains outside the
  probabilistic plotting API.
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

`inference.backend` defaults to `"gblk"`. The GBLK method fits each
same-family component group jointly via
`latticekrigx.glk.joint.fit_joint`; Gaussian and Bernoulli components remain
separate likelihood groups. This replaces the sequential per-component
logistic plus standalone spatial-smoother path. No config change is needed if
you did not previously set `inference.backend` explicitly.

The `sequential` backend is deprecated: a `DeprecationWarning` is emitted when
it is used. Remove `inference.backend="sequential"` from any existing configs.

### Processed-input entry point

Load the standard processed config and layer tree, then call the canonical
runner:

```python
from geopfa.prob import load_processed_pfa, load_probabilistic_config
from geopfa.prob import run_probabilistic

config_path = "path/to/config.json"
pfa, artifacts = load_processed_pfa(
    config_path,
    "path/to/processed/data",
    crs="EPSG:26911",
)
config = load_probabilistic_config(config_path)
result = run_probabilistic(pfa, config, input_artifacts=artifacts)
```

`load_processed_pfa` validates the processed layers and returns the complete
input-artifact mapping used by the run manifest.

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
- the data-informed support meets the minimum required by the selected
  inference mode; prior-only components do not require wells
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
