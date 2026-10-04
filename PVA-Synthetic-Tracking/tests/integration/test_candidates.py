from __future__ import annotations

import unittest

from tiny_target.candidate_benchmark import run_benchmark


class CandidateScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_benchmark(75)

    def test_one_physical_track_yields_one_candidate(self) -> None:
        scenario = self.report["scenarios"]["one_target"]
        self.assertTrue(scenario["all_targets_recovered"])
        self.assertEqual(scenario["candidate_count"], 1)

    def test_two_close_and_crossing_tracks_are_recovered(self) -> None:
        self.assertTrue(
            self.report["scenarios"]["two_close_targets"]["all_targets_recovered"]
        )
        self.assertTrue(
            self.report["scenarios"]["crossing_tracks"]["all_targets_recovered"]
        )

    def test_noise_only_sequence_has_no_candidates(self) -> None:
        self.assertEqual(self.report["scenarios"]["no_target"]["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
