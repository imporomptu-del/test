from __future__ import annotations

from pathlib import Path
import unittest

from tiny_target.cuda_build import SOURCE, build_cuda_library
from tiny_target.cuda_tracking_benchmark import _tegrastats_summary


class CudaBuildAndTelemetryTests(unittest.TestCase):
    def test_cuda_source_is_present_and_missing_compiler_fails_clearly(self) -> None:
        self.assertTrue(SOURCE.is_file())
        with self.assertRaisesRegex(FileNotFoundError, "nvcc is missing"):
            build_cuda_library(nvcc=Path("/definitely/missing/nvcc"))

    def test_tegrastats_summary_extracts_gpu_memory_power_and_temperature(self) -> None:
        summary = _tegrastats_summary(
            "RAM 5000/62841MB GR3D_FREQ 40% gpu@44.5C tj@48.0C "
            "VDD_GPU_SOC 6000mW/5000mW\n"
            "RAM 5300/62841MB GR3D_FREQ 90% gpu@46.0C tj@49.0C "
            "VDD_GPU_SOC 8000mW/6500mW\n"
        )
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["gpu_utilization_percent"]["median"], 65)
        self.assertEqual(summary["gpu_utilization_percent"]["maximum"], 90)
        self.assertEqual(summary["maximum_ram_used_mb"], 5300)
        self.assertEqual(summary["gpu_soc_power_mw"]["maximum"], 8000)
        self.assertEqual(summary["maximum_temperature_c"]["tj"], 49.0)


if __name__ == "__main__":
    unittest.main()
