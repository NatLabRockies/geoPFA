# Changelog

All notable changes to geoPFA are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — `feature/probabilistic-method`

> Full probabilistic Play Fairway Analysis system. Branch pending PR against `main`.

### Added — Phase A: Config + Entry Point + 2D/3D Dispatch
- `geopfa/prob/config.py` — `ProbabilisticConfig` dataclass with full JSON-driven schema,
  per-region component/layer nesting, validation, `from_json()` / `to_json()` round-trip.
- `geopfa/prob/runner.py` — `run_probabilistic()` entry point; automatic 2D/3D dispatch
  from region geometry; per-component orchestration loop.
- `geopfa/prob/__init__.py` — public API exports.
- 21 tests covering config validation, schema round-trip, 2D/3D dispatch.

### Added — Phase B: α_c Alpha Prior Builder
- `geopfa/prob/alpha.py` — `build_alpha_prior()` for three modes:
  - `"data"` — Beta MLE fit to well success/failure counts.
  - `"literature"` — user-supplied Beta(a, b) parameters.
  - `"uniform"` — Beta(1, 1) uninformative fallback.
  - 2D (`thermal`, `depth`) and 3D (joint depth-slice) layer integration.
  - Provenance metadata attached to every `AlphaPrior` result object.
- 30 tests: all modes, edge cases, 2D/3D parity, provenance fields.

### Added — Phase C: β_ck Feature Coefficient Builder
- `geopfa/prob/beta.py` — `BetaEstimator` with:
  - L2-penalized logistic regression (MAP estimation via `scipy.optimize.minimize` BFGS).
  - Per-feature regularization weights from `play_types.py` registry.
  - Elkan–Noto Positive-Unlabeled (PU) correction for biased well labels.
  - MAP prior means from play-type registry (non-zero regularization targets).
- `geopfa/prob/play_types.py` — `PLAY_TYPE_REGISTRY` for extensional, magmatic,
  compressional, and transform play types with per-keyword weight and mean tables.
- 35 tests: PU modes, play-type matching, MAP convergence, coefficient shapes.

### Added — Phase D: Spatial Field u_c via GP Extrapolation
- `geopfa/prob/spatial.py` — `fit_spatial_field_gp()` wrapping
  `geopfa.extrapolation.build_and_fit_gp` with `SpatialFieldConfig` (n_inducing,
  kernel, noise_var, max_iters, standardize).
- `geopfa/prob/config.py` — `SpatialFieldConfig` sub-dataclass.
- Sparse GP with KMeans inducing points; RBF + Matérn-3/2 kernel choices.
- ARD lengthscale heuristic (`compute_global_radius`); standardized coordinates.
- 22 tests: kernel selection, inducing-point counts, output shapes.

### Added — Phase E: Inference Backends
- `geopfa/prob/fitting.py` — `SequentialFitter`:
  - Sequential MAP inference: α → β → u → posterior Bernoulli.
  - Prior-predictive fallback for zero/sparse data (Phase I).
  - `force_prior_predictive` opt-in mode for supercritical targets.
  - Posterior mean + std output per grid cell.
- `geopfa/prob/bayesian.py` — `BayesianFitter` (optional, PyMC env):
  - Joint GP + hierarchical β MCMC via PyMC `NUTS` sampler.
  - Non-centered parameterization for β to reduce posterior geometry issues.
  - `n_draws`, `n_tune`, `target_accept`, `random_seed` all configurable.
- 28 tests (sequential); Bayesian tested separately in `bayesian` Pixi env.

### Added — Phase F: Spatial Block CV + Calibration
- `geopfa/prob/calibration.py` — isotonic regression calibration
  (`sklearn.isotonic.IsotonicRegression`) on held-out OOF predictions.
- `geopfa/prob/cross_validation.py` — spatial block k-fold CV:
  - Grid-based spatial block assignment to prevent data leakage.
  - OOF probability collection across folds.
  - Brier score, log-loss, ECE, reliability diagram generation.
- `geopfa/prob/diagnostics.py` — Shapiro–Wilk residual normality test,
  residual vs predicted scatter, coverage check.
- 40 tests covering fold counts, calibration monotonicity, Brier score bounds.

### Added — Phase G: Output Writers + Plotters
- `geopfa/prob/io.py` — `write_probability_outputs()` for GeoPackage and GeoTIFF
  with coordinate-reference-system preservation; manifest hash injection.
- `geopfa/prob/data.py` — `load_probability_outputs()` for round-trip reload;
  validation of required columns.
- `geopfa/prob/plotting.py` — `plot_probability_map()`, `plot_calibration_curve()`,
  `plot_spatial_uncertainty()` using Matplotlib; 3D depth-slice support.
- SHA-256 manifest hash written to output metadata for reproducibility tracking.
- 35 tests: CRS round-trips, GeoTIFF pixel values, calibration plot axes.

### Added — Phase H: Hierarchical β Across Regions
- `geopfa/prob/hierarchical_diag.py` — diagonal hierarchical pooling:
  - Per-region β estimates pooled toward a shared global mean (MAP shrinkage).
  - Pooling strength `tau` controls region-level vs global-level weight.
  - Automatic fallback to local estimate when only one region is active.
- Config key `hierarchical_beta.enabled` + `tau` in `ProbabilisticConfig`.
- 18 tests: pooling direction, tau=0 (full pooling), tau=∞ (no pooling), single-region.

### Added — Phase I: Prior-Predictive / Zero-Data Mode
- `geopfa/prob/fitting.py` — automatic fallback in `SequentialFitter._fit_component()`
  when `n_wells < min_wells_for_fit`: returns α prior predictive as the component
  probability without attempting β or u fits.
- Config keys: `min_wells_for_fit` (default 5), `force_prior_predictive` (default False).
- 3 tests: threshold trigger, `force_prior_predictive` override, metadata flag.

### Added — Phase J: Test Coverage ≥85%
- `tests/test_prob_regions.py` — 9 tests targeting `regions.py` (100% coverage).
- `tests/test_prob_hierarchical_diag.py` — 12 tests targeting `hierarchical_diag.py` (100%).
- `tests/test_prob_variogram.py` — 9 tests targeting `variogram.py` (87%).
- `tests/test_prob_runner_internals.py` — 12 tests for private runner branches.
- `tests/test_prob_data_io_gaps.py` — 16 tests for `data.py` / `io.py` uncovered paths.
- `pyproject.toml` — added `bayesian.py` to `[tool.coverage.report] omit` (PyMC-only
  module; tested separately in `bayesian` Pixi env to avoid PyMC install overhead).
- Coverage: **87%** on `geopfa.prob` (419 total tests, up from 74%).

### Added — Phase K: Documentation
- `README.md` — `## Probabilistic method (geopfa.prob)` section: feature list,
  quick-start snippet, cross-references to full docs.
- `docs/probabilistic_method.md` — updated Limitations/roadmap section; all
  Phases E/F/C.2/C.3/H/I now marked complete.
- `docs/migration_guide.md` — `## Upgrading to the Bayesian backend` section
  covering install, config snippets, and hierarchical-β setup.
- `examples/probabilistic/2d/probabilistic_2d.ipynb` — end-to-end 2D example:
  synthetic data, `run_probabilistic`, calibration curve, VoterVeto comparison,
  output file listing. Validated error-free via `jupyter nbconvert --execute`.
- `examples/probabilistic/3d/probabilistic_3d.ipynb` — end-to-end 3D example:
  synthetic depth-sliced data, depth-profile visualization, VoterVeto comparison.
  Validated error-free via `jupyter nbconvert --execute`.
- `docs/scope_backlog.md` — deferred backlog items (K.2 notebooks, dead CSV branch).
- Sphinx build: 0 content warnings (`sphinx-build -W`). Network-only SSL warnings
  present in air-gapped environments (intersphinx inventory fetches); no content errors.

### Changed
- `geopfa/extrapolation.py` — fixed RST docstring formatting (block-quote continuation
  in module docstring caused `docutils` warning).
- `geopfa/prob/play_types.py` — fixed RST literal-block indicator (`: ` → `::`) in
  module docstring to eliminate `docutils` warning.
- `docs/source/conf.py` — added `suppress_warnings` comment explaining SSL/network
  limitation; updated `nitpick_ignore_regex` for new prob module types.
- `docs/ENGINEERING_MANIFEST.md` — all Phase A–K checkboxes marked `[x]`.
- `docs/AGENT_SYNC.md` — updated throughout implementation; final state: all phases done.

### Infrastructure
- Style: `ruff check` + `ruff format --check` on `./geopfa/` — fully clean.
- New Pixi environments: `bayesian` (PyMC + ArviZ) isolated from standard test env.
- `pyproject.toml` `[tool.coverage.report]` omit list updated.

---

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
