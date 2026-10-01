# Evidence-coefficient prior profiles

This guide separates three roles that are often conflated in a PFA map:

1. **Neutral coefficient priors** express no directional claim.  For a
   standardized evidence feature \(x_{c,j}\), the default is
   \(\beta_{c,j} \sim N(0, 0.5^2)\).  A one-standard-deviation increase has
   prior median odds ratio one; the 95% interval is intentionally broad
   (about 0.38 to 2.66).
2. **Reference-based coefficient priors** encode a specific, reviewable claim
   from a paper, data product, or documented expert judgement.  The profile
   stores the reference and rationale alongside the mean and scale.  A
   direction from a paper is not treated as a measured universal effect size;
   use a wide scale unless a transferable quantitative estimate exists.
3. **Learned coefficients** are estimated by the normal component fit when
   component outcomes exist.  The profile remains a proper regularizing prior,
   but the likelihood updates it.  A prior-predictive component has no such
   update and must retain its profile provenance in the result metadata.

For a binary prior-predictive component, geoPFA evaluates

\[
p_c(s) = \operatorname{logit}^{-1}\left\{a_c(s) +
  \sum_j \beta_{c,j}\tilde{x}_{c,j}(s)\right\},
\]

where \(\tilde{x}\) is standardized over the declared prediction support.
The coefficient prior is therefore an effect per support-standard-deviation,
not a raw-unit effect.  The component map is a probability conditional on the
declared prior model; without labels it is not outcome calibrated.

## User-editable profile

Profiles are JSON-like mappings.  Every feature not named in `components` uses
the explicit `default`.  Component-qualified rules prevent an evidence layer
from silently inheriting a rule intended for another component.

```json
{
  "profile_id": "study-prior-v1",
  "default": {
    "mean_log_odds": 0.0,
    "sd_log_odds": 0.5,
    "rationale": "Neutral proper prior where no directional evidence exists."
  },
  "components": {
    "reservoir": {
      "fault_proximity": {
        "mean_log_odds": 0.4054651081,
        "sd_log_odds": 0.6,
        "references": ["https://doi.org/example"],
        "rationale": "Weak positive directional rule; effect size remains uncertain."
      }
    },
    "insulation": {
      "shallow_temperature": {
        "fixed_value": 0.0,
        "rationale": "Exclude a duplicate heat proxy from this component."
      }
    }
  }
}
```

Load and resolve the profile only after the study configuration has determined
the active feature names:

```python
from dataclasses import replace
import json

from geopfa.prob.config import EvidenceConfig, RegularizationConfig
from geopfa.prob.evidence_prior_profiles import evidence_prior_profile_from_dict
from geopfa.prob.evidence_priors import derive_evidence_coefficient_priors

profile = evidence_prior_profile_from_dict(json.loads(profile_path.read_text()))
resolved = derive_evidence_coefficient_priors(
    profile,
    {"reservoir": ("fault_proximity", "density"), "insulation": ("mt",)},
)
regularization = RegularizationConfig(
    prior_means=resolved.prior_means,
    prior_precisions=resolved.prior_precisions,
    fixed_coefficients=resolved.fixed_coefficients,
)
config = replace(
    config,
    evidence=replace(config.evidence, regularization=regularization),
)
```

`derive_evidence_coefficient_priors` fails if a profile rule names an inactive
feature.  This is deliberate: configuration changes must not silently remove
a literature rule.  A `fixed_value` is applied to every coefficient draw;
`fixed_value: 0.0` is an auditable feature exclusion, rather than an extremely
tight pseudo-prior.

## Evidence review before assigning direction

An automated workflow can supply neutral priors for all valid numeric evidence
layers.  It may also flag constant fields, missingness, collinearity, duplicate
provenance, and overlap with a target-defining layer.  Those diagnostics make a
review more efficient, but they do not establish a geological sign or an effect
size.  Promote a layer to a directional prior only when the cited source and
its spatial support, depth, time window, transformation, and component meaning
match the current study.

When no target labels exist, avoid fitting signs or effect sizes from the same
evidence stack and presenting the result as data learned.  If labelled outcomes
become available, fit that component with geoPFA's ordinary outcome-informed
model and report held-out calibration separately.
