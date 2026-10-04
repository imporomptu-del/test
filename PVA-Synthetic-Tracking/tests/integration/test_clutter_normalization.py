from __future__ import annotations

import unittest

from tiny_target.clutter_benchmark import run_benchmark


class ClutterNormalizationScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_benchmark(75)

    def test_local_cfar_recovers_target_hidden_by_global_raw_ranking(self) -> None:
        scenario = self.report["scenarios"]["heterogeneous_clutter"]
        self.assertFalse(scenario["raw_target_recovered"])
        self.assertTrue(scenario["cfar_target_recovered"])
        self.assertLess(
            scenario["target_cfar_surface_rank_lower_bound"],
            scenario["target_raw_surface_rank_lower_bound"],
        )

    def test_spatial_quota_prevents_one_region_from_owning_output(self) -> None:
        scenario = self.report["scenarios"]["spatial_monopoly"]
        self.assertEqual(scenario["unbalanced_occupied_output_cells"], 1)
        self.assertEqual(scenario["balanced_occupied_output_cells"], 4)

    def test_all_acceptance_checks_pass(self) -> None:
        self.assertTrue(all(self.report["acceptance"].values()))


if __name__ == "__main__":
    unittest.main()
