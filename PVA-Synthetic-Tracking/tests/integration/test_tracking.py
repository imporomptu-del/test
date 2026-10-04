from __future__ import annotations

import unittest

from tiny_target.tracking_benchmark import run_benchmark


class TrackingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_benchmark()

    def test_persistent_target_confirms_without_overlap_double_counting(self) -> None:
        acceptance = self.report["acceptance"]
        self.assertTrue(acceptance["persistent_target_confirms"])
        self.assertTrue(acceptance["overlapping_windows_not_credited_independently"])

    def test_crossing_track_identities_are_preserved(self) -> None:
        self.assertTrue(self.report["acceptance"]["crossing_identities_preserved"])

    def test_isolated_noise_candidates_do_not_confirm(self) -> None:
        self.assertTrue(self.report["acceptance"]["isolated_noise_does_not_confirm"])


if __name__ == "__main__":
    unittest.main()
