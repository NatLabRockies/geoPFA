# Learning evidence priors from labelled reference domains

`geopfa.prob.reference_evidence_priors` is the data-driven route for creating
component-evidence coefficient priors. It exists for the situation where the
target study has no reservoir or insulation labels, but comparable, separately
labelled geothermal studies do.

It does **not** look at names such as `faults`, `resistivity`, or `gravity` and
invent a geological direction. It learns an association only from a supplied
binary component target, then carries its uncertainty and provenance into a
portable `EvidencePriorRule`.

## What it does

For each independent reference domain, the learner:

1. requires finite binary component labels and both outcome classes;
2. standardises each nonconstant evidence layer inside that domain;
3. fits a multivariable, intercept-free-ridge logistic model;
4. records the effect and curvature-based uncertainty for every active layer;
5. pools each layer that occurs in at least two domains with a
   DerSimonian--Laird random-effects calculation; and
6. exports an explicit normal coefficient prior whose SD includes both
   within-domain estimation error and cross-domain heterogeneity.

The target-only or one-domain-only layers are not given a learned rule. When
the resulting configuration is applied to a target grid, those layers keep the
neutral default specified by the caller.

## Minimal use

```python
from geopfa.prob.evidence_priors import derive_evidence_coefficient_priors
from geopfa.prob.reference_evidence_priors import (
    ReferenceEvidenceDomain,
    learn_reference_domain_evidence_priors,
)

reference_domains = (
    ReferenceEvidenceDomain(
        domain_id="basin-a-2026",
        component="reservoir",
        evidence=basin_a_evidence,
        labels=basin_a_reservoir_labels,
        feature_names=("fault_proximity_inverted", "mt_resistivity"),
        source_id="doi:... or an immutable dataset identifier",
        target_definition="Observed productive-reservoir indicator.",
    ),
    ReferenceEvidenceDomain(
        domain_id="basin-b-2026",
        component="reservoir",
        evidence=basin_b_evidence,
        labels=basin_b_reservoir_labels,
        feature_names=("fault_proximity_inverted", "mt_resistivity"),
        source_id="doi:... or an immutable dataset identifier",
        target_definition="Observed productive-reservoir indicator.",
    ),
)

learned = learn_reference_domain_evidence_priors(reference_domains)

# Newberry is not one of reference_domains. Values here are only its rasters.
newberry_recommendation = derive_evidence_coefficient_priors(
    newberry_reservoir_evidence,
    newberry_layer_names,
    component="reservoir",
    config=learned.generation_config(
        default_mean_log_odds=0.0,
        default_sd_log_odds=0.5,
        redundancy_correlation=0.7,
    ),
)
config["evidence"]["regularization"].update(
    newberry_recommendation.regularization_values()
)
```

`learned.to_dict()` is an auditable record of the source datasets, target
definitions, individual-domain coefficients, standard errors, pooled mean,
heterogeneity SD, and exported rule. Save that alongside the generated profile
and the model run rather than copying only coefficient values into a config.

## Required scientific controls

This is automated estimation, not an automatic declaration that studies are
comparable. A valid use needs all of the following.

| Control | Why it matters |
| --- | --- |
| Canonical layer semantics | A matching feature name must represent the same measurement, transformation, depth/support, and favorable direction in every domain. For example, a raw distance and an inverted distance are different features. |
| Comparable target definition | Every `labels` array must represent the same component event. A productive-well indicator cannot be casually pooled with a cap-rock interpretation. |
| Target exclusion | Do not use Newberry cells to train the reservoir or insulation prior that will be reported for Newberry. |
| Spatial validation | Cell-level fits can be overly optimistic under spatial autocorrelation. Validate inside each reference domain with spatial blocks and evaluate transfer with leave-one-domain-out prediction. |
| Domain heterogeneity review | The random-effects SD signals disagreement; it is not permission to average conflicting geological regimes without a transferability argument. |
| Prior-predictive sensitivity | Run low/central/high learned-rule scenarios, plus a neutral rule, before using results in a decision. |

The current code cannot infer these controls from raster pixels alone. It
records enough information to audit them, fails for malformed labels, and
refuses to issue a learned rule for a layer with too little independent-domain
support.

## Scope for Newberry superhot

The available Newberry superhot run has no reservoir or insulation labels, so
this learner must not be trained on its current prior-predictive component
maps. Those maps are outputs of assumed coefficient distributions, not
observations. Until comparable labelled reference domains are assembled,
Newberry should use the neutral/unlabeled generator and explicitly report any
literature-derived directional rule as an assumption.

This separation is deliberate: it lets geoPFA add data-driven coefficient
generation without disguising an unlabeled spatial correlation as a calibrated
component probability.
