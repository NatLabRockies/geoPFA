# Migration guide

## Migrating to the GBLK method

The **Generalized Bayesian LatticeKrig (GBLK)** method is now the default
probabilistic backend (`inference.backend="gblk"`). This guide covers:

1. Migrating from `inference.backend="sequential"` (deprecated) to `"gblk"`.
2. Migrating from the Stage-1 direct-call API to the config-driven runner.
3. Enabling posterior draws for the one Bayesian GBLK path.

---

## 1. Migrating from `sequential` to `gblk`

### What changed

The `sequential` backend fits one component at a time with an independent
penalized logistic regression and standalone spatial smoother. The `gblk` backend
fits **all components jointly** in a single multivariate Bernoulli-logit model via
`latticekrigx.glk.joint.fit_joint`, estimating the cross-component correlation
$\Omega$ among component-specific LatticeKrig latent fields.

Key differences:

| Aspect | `sequential` (deprecated) | `gblk` (default) |
|---|---|---|
| Fit | Per-component penalized logistic model + spatial smoother | Joint multivariate GBLK |
| Spatial field | Independent RBF or LatticeKrig field (`spatial_field.backend`) | Shared LatticeKrig multiresolution basis |
| Cross-component dependence | Assumed independent | Estimated $\Omega$ |
| combined surface | Product of independently fit marginals | Product of marginals estimated by the joint fit (conditional plug-in co-occurrence) |
| Environment | Any provided Pixi environment | Any provided Pixi environment |

A `DeprecationWarning` is emitted whenever `inference.backend="sequential"` is used.

### Config migration

Remove or replace any explicit `inference.backend` setting. Under GBLK,
`spatial_field.enabled`, `n_levels`, and `lattice_centers_per_dimension`
configure the joint LatticeKrig field; sequential-only kernel and inducing-point
settings can be removed:

```jsonc
// Before (sequential, deprecated)
{
  "probabilistic": {
    "inference": { "backend": "sequential" },
    "spatial_field": { "enabled": true, "backend": "gpy_sparse" }
  }
}

// After (gblk, default)
{
  "probabilistic": {
    "inference": { "backend": "gblk" },
    "spatial_field": {
      "enabled": true,
      "n_levels": 2,
      "lattice_centers_per_dimension": 4,
      "coordinate_scaling": "axis_range"
    }
  }
}
```

Since `"gblk"` is the default you can omit `inference` entirely if you have no
other inference settings to configure.

### Result schema (unchanged)

The `ProbabilisticResult` schema is identical between backends:

- `result.combined` — joint resource co-occurrence surface (GeoDataFrame with a
  `probability` column).
- `result.components[name].probability` — per-component probability surface.
- `result.components[name].diagnostics` — per-fit metadata dict.

The `gblk` backend additionally populates `result.diagnostics["omega"]` with the
estimated $Q \times Q$ cross-component correlation matrix.

### Running in the correct environment

The Pixi lock supplies `latticekrigx` and `scikit-sparse` in every environment
and pins the exact LatticeKrigX Git revision. The Git dependency currently
requires organization access; public installation requires LatticeKrigX to be
published first. Use `dev-gblk` for a complete authenticated developer
environment:

```bash
pixi run -e dev-gblk geopfa-prob run --config my_config.json
```

Or programmatically:

```bash
pixi run -e dev-gblk python my_script.py
```

---

## 2. Migrating from the Stage-1 direct-call API to the config-driven runner

### What changed (symbol renames from PR #73 Stage-1 demo)

| Old symbol | New symbol | Module |
|---|---|---|
| `DemoComponentProbability` | `ComponentProbability` | `geopfa.prob.fitting` |
| `Stage1ScenarioSpec` | retired | It mislabeled linear x/y covariates as a spatial model. Use config-driven `ScenarioConfig` for real GBLK ablations. |
| `geopfa.prob.demo.fit_component_probability` | `geopfa.prob.fitting.fit_component_probability` | (same function, new module) |
| `geopfa.prob.stage1_demo.run_region_state_matrix` | retired | Use config-driven scenarios; `run_coordinate_trend_sensitivity` is only an explicitly reduced-form diagnostic. |

The supported direct fit helper remains available as
`from geopfa.prob import fit_component_probability`. The former Stage-1
scenario API is intentionally not aliased: its `include_spatial` flag appended
linear x/y predictors and was not a spatial random-field model.

### Direct-call pattern (still supported, sequential path only)

```python
from geopfa.prob import fit_component_probability, combine_probability_surfaces

result = fit_component_probability(
    component_data,
    prior_probability=0.55,
    prior_layer_name="heat_source_t3km",
    include_spatial=True,
    labeled_wells=wells_gdf,
    label_column="heat_label",
)
combined = combine_probability_surfaces(
    [result_heat.probability["probability"].to_numpy(),
     result_reservoir.probability["probability"].to_numpy()]
)
```

Note: `fit_component_probability` uses the `sequential` backend internally and will
emit a `DeprecationWarning` in future releases. Migrate to the config-driven
`run_probabilistic` path with `inference.backend="gblk"` for new workflows.

### Config-driven pattern (recommended)

```python
from pathlib import Path
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
        label_columns={"heat": "heat_label", "reservoir": "reservoir_label"},
        layer="wells",
    ),
    alpha={
        "heat": AlphaModeConfig(mode="layer_logit", layer="heat_source_t3km"),
        "reservoir": AlphaModeConfig(mode="scalar", scalar_fallback_pr0=0.45),
    },
    evidence=EvidenceConfig(),
    inference=InferenceConfig(backend="gblk"),
    outputs=OutputsConfig(format=("geotiff", "csv")),
)

result = run_probabilistic(pfa, cfg)
for name, surface in result.components.items():
    print(f"{name}: mean p = {surface.probability['probability'].mean():.3f}")
print(f"combined surface shape: {result.combined.shape}")
```

### Config JSON equivalent

```jsonc
{
  "pfa_pickle": "outputs/pfa.pkl",
  "probabilistic": {
    "enabled": true,
    "output_dir": "outputs/probabilistic/",
    "dimensions": "2d",
    "labels": {
      "source": "data/wells.gpkg",
      "id_col": "well_id",
      "layer": "wells",
      "label_columns": {
        "heat": "heat_label",
        "reservoir": "reservoir_label"
      }
    },
    "alpha": {
      "heat": { "mode": "layer_logit", "layer": "heat_source_t3km" },
      "reservoir": { "mode": "scalar", "scalar_fallback_pr0": 0.45 }
    },
    "inference": { "backend": "gblk" },
    "outputs": { "format": ["geotiff", "csv"] }
  }
}
```

Then run:

```bash
pixi run -e dev-gblk geopfa-prob run --config my_config.json
```

### Key differences from the Stage-1 demo

1. **`pfa_pickle` is required** for the CLI — the CLI loads a serialised PFA dict.
2. **`alpha` is per-component** — each component has its own alpha mode.
3. **`inference.backend="gblk"` is the default** — components are fit jointly; no
   `spatial_field` config is needed or used.
4. **Calibration and CV** — configure `calibration.method` to `"platt"` or
   `"isotonic"` for block-CV calibration.

### Behaviour changes since the Stage-1 demo merge

| Setting | Old default | New default | Impact |
|---|---|---|---|
| `inference.backend` | `"sequential"` | `"gblk"` | **Breaking**: joint fit replaces per-component fit; probability maps will differ. Set `backend: "sequential"` to reproduce old behaviour (deprecated). |
| `labels.min_wells_for_fit` | Hardcoded `4` | Configurable, default `4` | No change to default. |

#### Backend default change (`inference.backend: "gblk"`)

The joint GBLK fit is now the default. If you need to reproduce outputs from an
older run that used the sequential path, set `inference.backend: "sequential"` in
your config. Note that this emits a `DeprecationWarning`.

#### Positive--unlabeled outcomes (`pu_mode: "nnpu"`)

PU correction is opt-in; `labels.pu_mode` remains `"off"` by default. The
supported modern correction is non-negative PU empirical-risk minimization
(nnPU) under a selected-completely-at-random labeling assumption. It requires a
population class prior justified independently of the same labeled-positive
sample:

```jsonc
{
  "labels": {
    "pu_mode": "nnpu",
    "pu_class_prior": 0.25
  },
  "inference": { "backend": "sequential" }
}
```

The direct notation is `pu_mode: "nnpu"`. A per-component mapping may replace
the scalar class prior. nnPU is currently implemented only by the sequential
outcome fitter; GBLK rejects it explicitly. It addresses unlabeled outcomes,
not preferential site collection. For the latter, use the separate
finite-candidate `site_selection.mode="joint_binary"` sensitivity analysis and
freeze the outcome-dependent selection log odds externally.

---

## 3. Enabling the sole Bayesian GBLK path

The `gblk_bayesian` config enables the sole Bayesian path. It uses the GBLK
likelihood and LatticeKrigX's public Paige/INLA inference. This produces
posterior-mean probability surfaces plus credible-interval bands
(`probability_lo` / `probability_hi` columns) from paired posterior draws.

```jsonc
{
  "inference": {
    "backend": "gblk",
    "gblk_bayesian": {
      "enabled": true,
      "n_draws": 200,
      "seed": 42,
      "ci_level": 0.9,
      "cor_scale_median": 0.1,
      "spatial_sd_u": 1.0,
      "spatial_sd_tail_probability": 0.05,
      "dirichlet_concentration": 1.5,
      "kleiber_r0": 0.25,
      "kleiber_r1": 0.10
    }
  }
}
```

Or programmatically:

```python
from geopfa.prob import InferenceConfig, GBLKBayesianConfig

cfg = ProbabilisticConfig(
    ...
    inference=InferenceConfig(
        backend="gblk",
        gblk_bayesian=GBLKBayesianConfig(
            enabled=True,
            n_draws=200,
            seed=42,
            ci_level=0.9,
            cor_scale_median=0.1,
            spatial_sd_u=1.0,
            spatial_sd_tail_probability=0.05,
            dirichlet_concentration=1.5,
            kleiber_r0=0.25,
            kleiber_r1=0.10,
        ),
    ),
    ...
)
```

`kleiber_r0` and `kleiber_r1` are required for two-component fits and must be
frozen profile estimates, not tuning values. They are omitted for a univariate
fit. The Paige model supports 2-D `LKRectangle` and 3-D `LKBox` geometry and
consumes geoPFA's component-specific Gaussian fixed-coefficient priors through
the public joint interface. The authenticated pyINLA 0.1.8/INLA runtime has
passed the end-to-end projector and paired-draw preflight. Runtime and output
validation are described in `docs/probabilistic_method.md`.

The former independent PyMC backend and its `shared_field` combination rule
have been removed. Configurations using `backend="bayesian"`, a nested
`inference.bayesian` block, or `combination.rule="shared_field"` now fail
validation. Use `backend="gblk"` and set `gblk_bayesian.enabled=true`; this is
the sole Bayesian model and prevents two implementations from assigning
different meanings to the same posterior outputs.
