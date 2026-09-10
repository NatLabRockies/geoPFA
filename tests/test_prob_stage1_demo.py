import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import geopfa.prob.scenario as scenario_module
from geopfa.prob.scenario import (
    CoordinateTrendScenarioSpec,
    run_coordinate_trend_sensitivity,
    spatial_block_holdout_mask,
)


class ProbStage1DemoTest(unittest.TestCase):
    def test_scenario_matrix_rejects_non_cv_split_count(self):
        with self.assertRaisesRegex(ValueError, "n_splits"):
            run_coordinate_trend_sensitivity(
                pd.DataFrame({"signal": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}),
                np.array([0, 0, 1, 0, 1, 1]),
                np.zeros(6),
                np.array(
                    [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0],
                    ]
                ),
                [CoordinateTrendScenarioSpec(name="invalid_cv")],
                n_splits=1,
            )

    def test_failed_scenario_fits_are_not_replaced_by_identity_models(self):
        failed = SimpleNamespace(
            success=False,
            x=np.zeros(2),
            message="forced optimizer failure",
        )
        with patch.object(scenario_module, "minimize", return_value=failed):
            with self.assertRaisesRegex(
                RuntimeError, "forced optimizer failure"
            ):
                scenario_module._fit_logit(  # noqa: SLF001
                    np.column_stack([np.ones(8), np.arange(8.0)]),
                    np.tile([0.0, 1.0], 4),
                    np.zeros(8),
                )
            with self.assertRaisesRegex(
                RuntimeError, "forced optimizer failure"
            ):
                scenario_module._fit_platt_scaler(  # noqa: SLF001
                    np.linspace(-2.0, 2.0, 8), np.tile([0.0, 1.0], 4)
                )

        with self.assertRaisesRegex(ValueError, "two outcome classes"):
            scenario_module._fit_platt_scaler(  # noqa: SLF001
                np.linspace(-2.0, 2.0, 8), np.zeros(8)
            )

    def test_spatial_block_holdout_mask_keeps_train_and_test(self):
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        holdout = spatial_block_holdout_mask(
            coords, holdout_fraction=0.33, grid_size=2
        )

        self.assertEqual(holdout.dtype, bool)
        self.assertGreater(holdout.sum(), 0)
        self.assertLess(holdout.sum(), len(holdout))

    def test_coordinate_trend_sensitivity_returns_expected_rows(self):
        X_df = pd.DataFrame(
            {
                "heat": [0.1, 0.2, 0.8, 0.7, 0.3, 0.9],
                "fault": [0.2, 0.1, 0.9, 0.8, 0.4, 0.95],
            }
        )
        y = np.array([0, 0, 1, 1, 0, 1], dtype=int)
        alpha = np.array([-1.0, -0.8, 0.8, 1.0, -0.5, 1.2], dtype=float)
        coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ]
        )
        scenarios = [
            CoordinateTrendScenarioSpec(name="all_data"),
            CoordinateTrendScenarioSpec(
                name="no_priors", include_priors=False
            ),
            CoordinateTrendScenarioSpec(
                name="drop_fault", drop_features=("fault",)
            ),
        ]

        out = run_coordinate_trend_sensitivity(
            X_df, y, alpha, coords, scenarios, n_splits=2
        )

        self.assertEqual(len(out), 3)
        self.assertEqual(
            set(out["scenario"]), {"all_data", "no_priors", "drop_fault"}
        )
        self.assertTrue((out["n_samples"] > 0).all())
        self.assertIn("holdout_log_loss_calibrated", out.columns)
        self.assertIn("holdout_brier_calibrated", out.columns)
        self.assertIn("cv_log_loss_calibrated_mean", out.columns)
        self.assertTrue(
            out["holdout_calibration_status"]
            .str.startswith("not_estimable_")
            .all()
        )
        self.assertTrue(out["holdout_brier_calibrated"].isna().all())
        self.assertIn("coordinate_trend_enabled", out.columns)
        self.assertIn("cv_brier_by_fold", out.columns)

    def test_coordinate_trend_auc_is_tie_safe(self):
        y = np.array([0, 1, 0, 1], dtype=int)
        p = np.full(4, 0.5)
        self.assertEqual(scenario_module._roc_auc(y, p), 0.5)  # noqa: SLF001

    def test_coordinate_trend_sensitivity_rejects_fractional_labels(self):
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            run_coordinate_trend_sensitivity(
                pd.DataFrame({"signal": [0.0, 0.2, 0.8, 1.0]}),
                np.array([0.0, 0.5, 1.0, 1.0]),
                np.zeros(4),
                np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                [CoordinateTrendScenarioSpec(name="invalid_labels")],
                n_splits=2,
            )
