from __future__ import annotations

import unittest

from tiny_target.evaluation_benchmark import EvaluationModeConfig, run_benchmark


class EvaluationModeTests(unittest.TestCase):
    def test_mode_contract_rejects_hidden_or_unbounded_queues(self) -> None:
        for mode in ("correctness", "throughput", "real_time", "soak"):
            configured = EvaluationModeConfig(
                mode=mode,
                soak_repetitions=2 if mode == "soak" else 1,
            )
            self.assertEqual(configured.queue_capacity_frames, 0)
        with self.assertRaisesRegex(ValueError, "queue capacity of zero"):
            EvaluationModeConfig(queue_capacity_frames=1)
        with self.assertRaisesRegex(ValueError, "only in soak"):
            EvaluationModeConfig(mode="throughput", soak_repetitions=2)


class EndToEndEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_benchmark(mode="correctness", seed=75)

    def test_correctness_replay_and_manifest_categories(self) -> None:
        self.assertTrue(self.report["correctness"]["deterministic_curve_replay_equal"])
        categories = {
            entry["category"] for entry in self.report["dataset_inventory"]["entries"]
        }
        self.assertEqual(
            categories,
            {
                "noise_only",
                "real_targets",
                "raw_injected",
                "stress_scene",
                "controlled_geometry",
            },
        )

    def test_threshold_and_integration_length_curves_are_complete(self) -> None:
        curves = self.report["accuracy_curve_points"]
        self.assertEqual(len(curves), 21)
        self.assertEqual(
            {row["integration_window_frames"] for row in curves}, {2, 4, 8}
        )
        for length in (2, 4, 8):
            rows = [row for row in curves if row["integration_window_frames"] == length]
            false_counts = [row["false_alarms"]["count"] for row in rows]
            self.assertEqual(false_counts, sorted(false_counts, reverse=True))

    def test_metrics_include_detection_false_alarm_tracking_and_errors(self) -> None:
        row = next(
            item
            for item in self.report["accuracy_curve_points"]
            if item["integration_window_frames"] == 4 and item["threshold_snr"] == 8
        )
        self.assertIsNotNone(row["probability_of_detection"])
        self.assertIsNotNone(row["precision"])
        self.assertIsNotNone(row["false_alarms"]["per_minute"])
        self.assertIsNotNone(row["localization_error_px"])
        self.assertIsNotNone(row["velocity_error_px_s"])
        self.assertEqual(row["false_confirmed_track_count"], 0)

    def test_performance_includes_tail_latency_queue_and_stage_accounting(self) -> None:
        performance = self.report["performance"]
        self.assertEqual(performance["dropped_frames"], 0)
        self.assertEqual(performance["maximum_application_queue_depth_frames"], 0)
        latency = performance["end_to_end_frame_latency"]
        for key in ("median_ms", "p90_ms", "p95_ms", "p99_ms", "maximum_ms"):
            self.assertIn(key, latency)
        self.assertIn("candidate_and_tracking", performance["stages"])


if __name__ == "__main__":
    unittest.main()
