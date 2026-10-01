# Evidence-prior profile schema

An evidence-prior profile is a small JSON file that makes a literature-derived
or reference-domain-derived heuristic portable. Load it with
`load_evidence_prior_profile()` and pass the result to
`derive_evidence_coefficient_priors()`.

```json
{
  "default_mean_log_odds": 0.0,
  "default_sd_log_odds": 0.5,
  "redundancy_correlation": 0.7,
  "rules": {
    "reservoir:fault_distance_inverted": {
      "mean_log_odds": 0.6931471805599453,
      "sd_log_odds": 0.4,
      "references": ["https://doi.org/10.15121/1493758"],
      "rationale": "Declared structural-permeability rule for an extensional geothermal setting."
    }
  }
}
```

## Fields

| Field | Meaning |
| --- | --- |
| `default_mean_log_odds` | Mean used for a layer with no matching rule. Use zero for direction-neutral automation. |
| `default_sd_log_odds` | Positive coefficient SD for an unmatched layer, on the one-support-SD log-odds scale. |
| `redundancy_correlation` | Optional threshold in `(0, 1]` for the declared effect-budget-sharing heuristic. |
| `rules` | Map of a global layer name or `component:layer` name to a specific distribution. |
| `mean_log_odds` | Rule-specific expected log-odds change per support SD. |
| `sd_log_odds` | Rule-specific positive uncertainty SD. |
| `references` | Source identifiers or durable URLs supporting the declared rule. |
| `rationale` | Plain-language scientific mechanism and applicability boundary. |

Component-qualified rules take precedence over global rules. Unknown keys,
missing distribution parameters, non-finite numbers, and malformed references
fail closed rather than being ignored.

The profile itself is a declaration of assumptions. It is not training data and
does not turn unlabeled component maps into calibrated probabilities. A profile
derived from external labelled domains should additionally record the source
study IDs, preprocessing version, target definition, holdout performance, and
transferability limits in the profile's project-level provenance record.
