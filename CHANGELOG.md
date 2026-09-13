# Changelog

All notable changes to geoPFA are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- A config-driven probabilistic PFA workflow for 2-D and 3-D component grids.
- Bernoulli component models and a Gaussian heat-observation stage that maps
  posterior predictive temperature distributions to declared exceedance
  probabilities before component combination.
- One Bayesian spatial path through LatticeKrigX's Paige/INLA implementation,
  with multiresolution spatial fields, fixed-effect priors, and paired
  uncertainty draws.
- Physics-informed scalar, layer, composite, and thermal-exceedance priors.
- Buffered or blocked out-of-fold predictive stacking against each configured
  event prior, using Bernoulli log score or Gaussian predictive density as
  appropriate.
- Spatial cross-validation diagnostics, reliability summaries, proper scores,
  optional non-negative PU logistic estimation, and an independent
  candidate-site selection sensitivity model.
- GeoTIFF, CSV, Parquet, and VTK writers with dimension checks, plus
  checksum-bound input, output, configuration, and implementation provenance.
- Incremental posterior draw blocks with exact state validation and guarded
  baseline/scenario restart after recoverable execution errors.
- Output-free Newberry and Nevada notebooks comparing probabilistic results
  with the traditional VoterVeto workflow.

### Changed

- The probabilistic runner now uses one canonical component grid, explicit
  coordinate and depth columns, grouped well-profile folds, and metric spatial
  controls only with projected metre coordinate systems.
- Gaussian heat outputs now report physical-unit posterior predictive response
  summaries as well as threshold-exceedance probabilities.
- Configuration keys that had no implemented effect are rejected rather than
  accepted silently.

### Removed

- The unsupported regional coefficient-pooling path and its diagnostics.
- Stale references to a second PyMC Bayesian implementation.

## Prior history

See individual GitHub pull requests and commits on `main` for changes prior to the
`feature/probabilistic-method` branch. Notable merged work:

- **PR #73** (2026-06-16) — `geopfa.prob` baseline: calibration, decision_metrics,
  scenario, data, synthetic fixture, 62 tests.
- **PR #68** — pixi.lock v7 update.
- **PR #62** — tutorial notebook + functionality updates.
- **PR #57** — processing module refactor/reconcile.
- **PR #53** — VoterVeto rename/relocation.
- **PR #48** — plotters refactor/reconcile.
- **PR #35** — uncertainty module removal.
- **PR #31** — 2D/3D VoterVeto reconciliation + NaN handling + vectorized rasterization.
- **PR #30** — GPy-based extrapolation module.
