from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from tiny_target.evaluation import SyntheticTarget, transformed_target_truth
from tiny_target.injected_raw_evaluation import (
    _match_serialized,
    _probe_summary,
    analyze,
)


class InjectedRawTruthTests(unittest.TestCase):
    def test_truth_is_transformed_and_fit_in_stabilized_coordinates(self) -> None:
        target = SyntheticTarget(
            target_id="t",
            flux_dn=100,
            reference_timestamp_ns=0,
            reference_position_xy_px=(10, 20),
            velocity_xy_px_s=(2, -1),
            first_frame_index=0,
        )
        translation = np.array([[1, 0, 5], [0, 1, 3], [0, 0, 1]], float)
        truth = transformed_target_truth(
            target,
            (0, 1, 2),
            1_000_000_000,
            {
                0: (0, translation),
                1: (1_000_000_000, translation),
                2: (2_000_000_000, translation),
            },
        )
        np.testing.assert_allclose(truth["position_xy_px"], [17, 22], atol=1e-10)
        np.testing.assert_allclose(truth["velocity_xy_px_s"], [2, -1], atol=1e-10)
        self.assertLess(truth["fit_residual_rms_px"], 1e-10)

    def test_serialized_candidate_matching_is_one_to_one(self) -> None:
        candidates = [
            {
                "candidate_index": 0,
                "discrete_position_xy_px": [10, 20],
                "discrete_velocity_xy_px_s": [1, 0],
                "normalized_score_snr": 9,
            },
            {
                "candidate_index": 1,
                "discrete_position_xy_px": [30, 40],
                "discrete_velocity_xy_px_s": [0, 0],
                "normalized_score_snr": 8,
            },
        ]
        truths = [
            {
                "target_id": "t",
                "flux_dn": 100,
                "position_xy_px": [10.2, 19.9],
                "velocity_xy_px_s": [1, 0],
            }
        ]
        result = _match_serialized(candidates, truths, 2, 1)
        self.assertEqual(len(result["matches"]), 1)
        self.assertEqual(result["unmatched_candidate_count"], 1)

    def test_probe_summary_keeps_raw_and_selection_ranks_distinct(self) -> None:
        summary = _probe_summary(
            [
                {
                    "targets": [
                        {
                            "flux_dn": 8000,
                            "valid_score_available": True,
                            "local_peak_score_snr": 190,
                            "surface_rank_lower_bound": 17000,
                            "selection_score_available": True,
                            "local_peak_selection_score": 9.5,
                            "selection_score_units": "local_robust_sigma",
                            "selection_surface_rank_lower_bound": 12,
                        }
                    ]
                }
            ]
        )
        self.assertEqual(summary["8000"]["surface_rank_lower_bound"]["best"], 17000)
        self.assertEqual(
            summary["8000"]["selection_surface_rank_lower_bound"]["best"],
            12,
        )
        self.assertEqual(
            summary["8000"]["selection_score_units"],
            "local_robust_sigma",
        )

    def test_analysis_separates_in_frame_from_valid_support_probability(self) -> None:
        target = {
            "target_id": "supported",
            "flux_dn": 1000,
            "reference_timestamp_ns": 0,
            "reference_position_xy_px": [10, 20],
            "velocity_xy_px_s": [0, 0],
            "acceleration_xy_px_s2": [0, 0],
            "first_frame_index": 0,
            "last_frame_index": 3,
        }
        frames = (0, 1, 2, 3)
        report = {
            "schema_version": "seaqr.tiny-target.motion.v10",
            "preprocessed_frames": [
                {"frame_index": index, "timestamp_ns": index * 100_000_000}
                for index in frames
            ],
            "stabilized_frames": [
                {
                    "frame_index": index,
                    "source_to_reference_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
                for index in frames
            ],
            "evaluation": {
                "injection": {
                    "identity": {},
                    "specification": {
                        "schema_version": 1,
                        "random_seed": 75,
                        "psf_sigma_px": 0.8,
                        "psf_radius_px": 3,
                        "targets": [target],
                    },
                    "frame_records": [],
                },
                "candidate_threshold_sweep_parameter": "cfar_threshold_sigma",
                "candidate_threshold_sweep": {
                    "6": [
                        {
                            "candidate_batch": {
                                "frame_indices": list(frames),
                                "reference_timestamp_ns": 150_000_000,
                                "candidates": [
                                    {
                                        "candidate_index": 0,
                                        "discrete_position_xy_px": [10, 20],
                                        "discrete_velocity_xy_px_s": [0, 0],
                                        "normalized_score_snr": 12,
                                        "selection": {
                                            "ranking_score": 9,
                                            "ranking_score_units": "local_robust_sigma",
                                        },
                                    }
                                ],
                                "metrics": {},
                            },
                            "confirmed_or_coasted_tracks": [
                                {
                                    "track_id": 7,
                                    "lifecycle_state": "confirmed",
                                    "state": {"mean": [10, 20, 0, 0]},
                                    "confirmation": {
                                        "timestamp_ns": 150_000_000,
                                        "latency_s": 0.4,
                                        "independent_hits": 2,
                                    },
                                }
                            ],
                        }
                    ]
                },
                "injected_truth_score_probes": [
                    {
                        "frame_indices": list(frames),
                        "reference_timestamp_ns": 150_000_000,
                        "targets": [
                            target
                            | {
                                "valid_score_available": True,
                                "local_peak_score_snr": 12,
                                "surface_rank_lower_bound": 100,
                                "selection_score_available": True,
                                "local_peak_selection_score": 9,
                                "selection_score_units": "local_robust_sigma",
                                "selection_surface_rank_lower_bound": 2,
                            }
                        ],
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.json"
            path.write_text(json.dumps(report))
            curve = analyze(path)["accuracy_curve_points"][0]
        self.assertEqual(curve["cfar_threshold_sigma"], 6)
        self.assertEqual(curve["probability_of_detection"], 1)
        self.assertTrue(curve["valid_support_evidence_complete"])
        self.assertEqual(curve["valid_support_truth_opportunity_count"], 1)
        self.assertEqual(curve["probability_of_detection_given_valid_support"], 1)
        self.assertEqual(curve["truth_matched_confirmed_track_count"], 1)
        self.assertEqual(curve["unmatched_confirmed_or_coasted_track_count"], 0)
        self.assertEqual(curve["confirmed_injected_target_ids"], ["supported"])
        self.assertEqual(
            curve["confirmed_injected_target_probability_given_any_valid_support"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
