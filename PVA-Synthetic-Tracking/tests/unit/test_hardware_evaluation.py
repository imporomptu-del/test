from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tiny_target.evaluation import EvaluationError
from tiny_target.hardware_evaluation import analyze


class HardwareEvaluationTests(unittest.TestCase):
    def test_rejects_wrong_report_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            motion = root / "motion.json"
            cuda = root / "cuda.json"
            motion.write_text(json.dumps({"schema_version": "wrong"}))
            cuda.write_text(
                json.dumps(
                    {"schema_version": "seaqr.tiny-target.cuda-tracking-benchmark.v1"}
                )
            )
            with self.assertRaisesRegex(EvaluationError, "motion.v9"):
                analyze(motion, cuda)

    def test_checked_reports_produce_honest_non_realtime_summary(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        report = analyze(
            repository
            / "results/tiny_target/phase10/raw16_cuda_tracking_15pairs_v2.json",
            repository / "results/tiny_target/phase8/cuda_tracking_benchmark.json",
        )
        self.assertFalse(report["claims"]["real_time"])
        self.assertFalse(report["detection_and_tracking"]["false_alarm_metrics_valid"])
        self.assertIsNone(report["throughput"]["dropped_frames"])
        self.assertIn("p99_ms", report["latency"]["per_stage"]["candidate_extraction"])


if __name__ == "__main__":
    unittest.main()
