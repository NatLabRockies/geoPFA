
[![PyPi](https://badge.fury.io/py/geoPFA.svg)](https://pypi.org/project/geoPFA/)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.17316283.svg)](https://doi.org/10.5281/zenodo.17316283)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-orange.svg)](https://opensource.org/licenses/BSD-3-Clause)
![SWR](https://img.shields.io/badge/SWR--25--73_-blue?label=NLR)

[![PythonV](https://img.shields.io/pypi/pyversions/geoPFA.svg)](https://pypi.org/project/geoPFA/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pixi](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![codecov](https://codecov.io/gh/NatLabRockies/geoPFA/graph/badge.svg?token=W2COSPBX4Z)](https://codecov.io/gh/NatLabRockies/geoPFA)

# Geothermal PFA

geoPFA is an open-source Python library for conducting Play Fairway Analysis
(PFA) in 2D and 3D, designed to reduce exploration risk by integrating surface
and subsurface considerations into a single, transparent workflow. Built around
NLR’s Geothermal PFA Best Practices and aligned with FAIR software principles,
geoPFA provides modular, extensible tools for cleaning, processing, weighting,
and combining diverse datasets into quantitative favorability maps. These
datasets can include geological, geophysical, geochemical, and
thermo-hydro-mechanical-chemical simulation results, as well as surface-level
factors such as energy demand, transmission access, and natural hazard
exposure.

The framework is fully customizable, enabling users to define criteria,
components, and indicators for any geothermal resource type—from
low-temperature and conventional hydrothermal to superhot systems—and to extend
the methodology to other subsurface applications if desired. geoPFA supports multiple data
processing approaches, including interpolation, density mapping, distance-based
scoring, extrapolation, and thermal modeling, while allowing integration of
expert-derived weightings or analytical hierarchy methods.

geoPFA has been successfully demonstrated in diverse contexts: a 3D PFA for
the Nesjavellir field in Iceland, where results aligned with known subsurface
conditions and guided scenario-based development strategies (Taverna et al.,
2025); and 2D PFAs of the Denver Basin and Alaska for lower-enthalpy geothermal
with greater emphasis on surface constraints (Davalos-Elizondo et al., 2024;
in work). By making advanced exploration workflows reproducible, transparent,
and openly accessible, geoPFA enables research teams, developers, and agencies
to make better-informed decisions through reducing time required for developing
workflows, allowing more time to be spent on feature engineering and interpretation
of results.

## Probabilistic method (`geopfa.prob`)

geoPFA includes an experimental probabilistic PFA workflow in `geopfa.prob`.
The default GBLK backend combines physics-informed priors, penalized evidence
effects, and LatticeKrig spatial fields. It supports Bernoulli component
outcomes and continuous Gaussian heat observations, which are converted to a
declared posterior predictive temperature-exceedance event using paired
mean-field and residual-precision draws before component combination. It emits
raw per-component probability surfaces and a conditional plug-in co-occurrence
surface. Calibration must be evaluated explicitly against held-out labels; the
software does not imply that a map is calibrated merely because it is a
probability map.

**Features:**

- Config-driven runner (`run_probabilistic`) — one JSON or Python config
  drives the full 2D/3D pipeline from labeled wells to GeoTIFF outputs. JSON
  paths are resolved relative to the config file and manifests bind the
  effective config plus input/output hashes.
- Multiple prior modes: scalar, layer-based logit, thermal exceedance,
  in-grid thermal exceedance, and multi-layer composite.
- Play-type regularization — data-informed L2 priors for extensional,
  magmatic, and convective geothermal settings.
- PU estimation — non-negative PU logistic risk with an externally supplied
  class prior; pseudo-absence fitting remains available only as an explicitly
  named naive comparator.
- Spatial LatticeKrig field — multiresolution LatticeKrig exact sparse basis
  (latticekrigx backend, default) or lightweight RBF smoother. Bayesian GBLK
  runs produce paired uncertainty draws; MAP fits produce point estimates.
  Sparse precision reduces coefficient-side
  cost, but current joint training materializes the assembled design;
  prediction streams only on code paths that explicitly expose batching.
- Block-CV diagnostics — spatially blocked Brier score, Brier skill score,
  and reliability summaries. Post-hoc calibration is currently available on
  the sequential backend; GBLK reports raw out-of-fold diagnostics explicitly.
- Adaptive component updates — optional predictive stacking uses buffered or
  blocked out-of-fold logarithmic score to mix each Bayesian update with its
  configured event prior before component combination.
- GBLK inference — same-family multivariate Bernoulli-logit or Gaussian-identity
  fits via `latticekrigx.glk` (default), with a conditional plug-in
  co-occurrence surface. The sole Bayesian
  dispatch selects LatticeKrigX's public Paige/INLA model, including real 2-D or
  3-D geometry and declared fixed-coefficient priors. The runtime contract and
  validation requirements are documented in `docs/probabilistic_method.md`.
- Outputs: GeoTIFF, CSV, Parquet, VTK (.vtp for 3D), JSON calibration
  metrics, Markdown diagnostics report, SHA-256 manifest binding the effective
  config, inputs, outputs, and the geoPFA/LatticeKrigX runtime implementations,
  and optional
  incremental, resumable, hash-verified posterior, prior-predictive, or mixed
  state/probability blocks for baseline and configured scenarios.
- Frozen forward-state export — deterministic GBLK MAP fits can be decomposed
  into prior, named evidence, and spatial logit contributions and evaluated in
  vectorized batches by downstream sensitivity studies without refitting.

**Quick start:**

The GBLK dependency is LatticeKrigX (Python 3.11–3.12). The Pixi lock pins an
exact LatticeKrigX Git revision so authenticated development and
continuous-integration environments install the same implementation. A public
geoPFA package release requires that LatticeKrigX revision to be published
first; geoPFA declares the compatible package version in its distribution
metadata and does not fall back to a different model.

Raw study downloads, evidence rasters, prepared grids, labels, serialized
`pfa.pkl` caches, and generated run directories are deliberately excluded from
Git. The three study notebooks contain their complete probabilistic
configurations and identify the required local inputs. Notebooks are committed
without cell outputs; no fitted models, result tables, summaries, or generated
figures are version-controlled. The repository gate also rejects tracked files
of 10 MiB or larger so generated artifacts cannot enter release history.

Each completed run owns its untracked output directory. A rerun may resume an incomplete
posterior or reuse a directory only when its manifest matches both the current
effective config and implementation and all bound files still verify; otherwise
choose a fresh output directory. Study notebooks display comparisons directly
and do not contain committed execution output.

```python
from geopfa.prob import ProbabilisticConfig, run_probabilistic

cfg = ProbabilisticConfig.from_dict({
    "enabled": True,
    "output_dir": "outputs/probabilistic",
    "dimensions": "2d",
    "labels": {
        "source": "data/wells.gpkg",
        "id_col": "well_id",
        "label_columns": {"heat": "heat_label", "reservoir": "reservoir_label"},
    },
    "alpha": {
        "heat": {"mode": "layer_logit", "layer": "heat_source_t3km"},
        "reservoir": {"mode": "scalar", "scalar_fallback_pr0": 0.45},
    },
    "spatial_field": {
        "enabled": True,
        "backend": "latticekrigx",
        "n_levels": 2,
        "lattice_centers_per_dimension": 4,
    },
    "calibration": {"method": "none"},
    "outputs": {"format": ["geotiff", "csv"]},
})
result = run_probabilistic(pfa, cfg)
# result.components["heat"].probability  — GeoDataFrame with 'probability' column
# result.combined                         — combined resource probability surface
# run_gblk_calibration_cv(...)             — explicit raw block-CV diagnostics
```

**Documentation:**

- User guide: `docs/probabilistic_method.md`
- Config reference: `docs/probabilistic_method.md#config-reference`
- Migration from Stage-1 demo: `docs/migration_guide.md`
- Bayesian GBLK path: `docs/migration_guide.md#enabling-the-sole-bayesian-gblk-path`

<!-- start-license -->
# NOTICE

Copyright © 2025 Alliance for Energy Innovation, LLC

This work was authored by the National Laboratory of the Rockies for the 
U.S. Department of Energy (DOE), operated under Contract No. DE-AC36-08GO28308. 
Funding provided by Department of Energy Hydrocarbons and Geothermal Energy Office, 
Office of Geothermal. The views expressed in the article do not necessarily represent 
the views of the DOE or the U.S. Government. The U.S. Government retains and the 
publisher, by accepting the article for publication, acknowledges that the U.S. 
Government retains a nonexclusive, paid-up, irrevocable, worldwide license to 
publish or reproduce the published form of this work, or allow others to do so, 
for U.S. Government purposes. 
