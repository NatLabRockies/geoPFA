"""Fail-closed contract for unavailable regional hierarchical GBLK."""

from __future__ import annotations

import importlib.util

import geopfa.prob as probabilistic
from geopfa.prob import gblk_assemble, gblk_runner, regions
from geopfa.prob.runner import ProbabilisticResult


def test_hierarchical_regional_runner_is_not_exposed() -> None:
    assert not hasattr(gblk_runner, "run_gblk_hierarchical_regional")
    assert "run_gblk_hierarchical_regional" not in gblk_runner.__all__


def test_unimplemented_regional_hierarchy_has_no_public_or_result_path() -> (
    None
):
    assert not hasattr(gblk_assemble, "pool_regional_coefficients")
    assert "pool_regional_coefficients" not in gblk_assemble.__all__
    assert not hasattr(regions, "combine_region_beta_summaries")
    assert not hasattr(probabilistic, "combine_region_beta_summaries")
    assert "per_region_beta" not in ProbabilisticResult.__dataclass_fields__
    assert importlib.util.find_spec("geopfa.prob.hierarchical_diag") is None
