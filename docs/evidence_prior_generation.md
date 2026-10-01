# Evidence-coefficient prior generation for unlabeled components

`geopfa.prob.evidence_priors` makes the prior-only part of a probabilistic PFA
explicit and auditable.  It is intended for components such as reservoir or
insulation when a study has evidence rasters but no component outcome labels.

## Scientific boundary

An unlabeled raster matrix can reveal coverage, scale, missingness, and
redundancy.  It cannot establish whether a higher value raises or lowers an
unobserved component probability, nor can it estimate that effect size.  The
generator therefore never infers a directional relationship from a feature
name, map distribution, or raster correlation alone.

Its default output for an unmatched feature is:

\[
\beta_{c,j} \sim \operatorname{Normal}(0,\;0.5^2),
\]

where the coefficient applies to a one-standard-deviation contrast over the
declared prediction support.  This is a neutral *uncertainty scenario*, not a
calibrated component relationship.

Directional effects must come from a rule that records a rationale and source,
such as a peer-reviewed study, a government technical report, or an
independently labelled reference domain.  The rule is still a prior assumption
until it is updated and checked against target-component labels.

## Minimal programmatic use

```python
import math
import numpy as np

from geopfa.prob.evidence_priors import (
    EvidencePriorGenerationConfig,
    EvidencePriorRule,
    derive_evidence_coefficient_priors,
)

# rows are prediction-support cells; columns correspond to layer_names
evidence = np.asarray([[...], [...]])
layer_names = ("fault_distance_inverted", "resistivity_inverted")

reservoir_fault_rule = EvidencePriorRule(
    mean_log_odds=math.log(2.0),
    sd_log_odds=0.50,
    references=("https://doi.org/10.15121/1493758",),
    rationale=(
        "Mapped fault proximity is a reservoir-permeability proxy in the "
        "declared extensional geothermal setting."
    ),
)

recommendation = derive_evidence_coefficient_priors(
    evidence,
    layer_names,
    component="reservoir",
    config=EvidencePriorGenerationConfig(
        redundancy_correlation=0.70,
        rules={"reservoir:fault_distance_inverted": reservoir_fault_rule},
    ),
)

# Feed the output into the existing explicit geoPFA configuration contract.
config["evidence"]["regularization"].update(
    recommendation.regularization_values()
)
```

The returned recommendation includes support centre and scale, layer ranges,
correlation-derived redundancy groups, source references, rationale, and a
status.  `regularization_values()` produces the component-qualified
`prior_means` and `prior_precisions` accepted by the current probabilistic
configuration.

## Rule precedence and custom heuristics

A rule key of `"reservoir:fault_distance_inverted"` applies only to that
component and wins over a global `"fault_distance_inverted"` rule.  This is
important because the same raw layer may legitimately be transformed or
interpreted differently across heat, reservoir, and insulation components.

For simple project-specific automation, callers can supply a callable instead
of maintaining a large profile:

```python
def local_rule(component, summary):
    if component == "reservoir" and summary.feature_name.endswith("faults"):
        return EvidencePriorRule(
            mean_log_odds=math.log(1.5),
            sd_log_odds=0.60,
            references=("https://doi.org/10.15121/1493758",),
            rationale="Declared structural-permeability sensitivity rule.",
        )
    return None
```

Unmatched layers remain neutral.  This fail-safe behavior prevents a newly
encountered layer from silently acquiring a geological direction just because
its name resembles a known layer.

## Redundancy adjustment

When `redundancy_correlation` is set, the module finds connected groups whose
absolute pairwise raster correlation meets that threshold.  For a group of
size \(m\), it changes each selected prior from \(\mu, \sigma\) to
\((\mu/m, \sigma/\sqrt{m})\).  If duplicate layers have the same rule, their
combined expected effect and variance therefore stay near the single-layer
budget instead of multiplying merely because the same signal was supplied
twice.

This is a transparent anti-double-counting heuristic, not a learned
conditional-dependence model.  It must be reported and included in sensitivity
analysis.  A user may omit the threshold to retain independent priors.

## Recommended evidence hierarchy

Use the strongest available source in this order:

1. **Target-component labels:** fit and spatially validate the component model.
2. **External labelled reference domains:** estimate a hierarchical or
   meta-analytic prior, transport it with explicit geological-context limits,
   and validate on held-out domains.
3. **Quantitative literature rule:** record the layer transformation, target
   setting, effect-size distribution, citation, and applicability boundary.
4. **Neutral generic prior:** retain the layer only as direction-neutral
   uncertainty, or exclude it if it cannot influence the stated decision.

This design is compatible with the geothermal PFA practice of integrating
geological, geophysical, and geochemical evidence while keeping uncertainty
and assumptions explicit. It also follows Bayesian workflow guidance that
prior predictive behavior and sensitivity must be checked rather than inferred
from a probability-map label alone.

## Next implementation stage

The next scientifically meaningful automation is not a larger name-based rule
catalog. It is an optional **reference-domain prior trainer** that accepts
multiple independently labelled studies, fits a regularised hierarchical model
on common standardised layer semantics, and exports a versioned prior profile.
That profile should carry training-domain identifiers, preprocessing hashes,
coefficient covariance, spatial holdout results, and transferability limits.
The same public rule interface can then consume it without changing a study's
probabilistic runner.

## Sources for the methodology boundary

- The National Laboratory of the Rockies’ [Geothermal Play Fairway Analysis
  Best Practices](https://www.nrel.gov/docs/fy23osti/86139.pdf) identifies
  heat, permeability/reservoir quality, fluid, cap rock, and related evidence
  categories, but does not supply universal coefficient magnitudes for every
  setting.
- Piironen and Vehtari’s [regularised shrinkage-prior
  paper](https://doi.org/10.1214/17-EJS1337SI) motivates explicit control of
  sparsity and coefficient regularisation rather than unbounded generic
  effects.
- Gelman et al.’s [Bayesian workflow
  paper](https://arxiv.org/abs/2011.01808) motivates prior-predictive checks,
  model sensitivity, and posterior-predictive evaluation as separate steps.
