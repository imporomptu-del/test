from __future__ import annotations

import unittest

from tiny_target.motion_cli import estimate_config


class MotionCliOverrideTests(unittest.TestCase):
    def test_input_and_timestamp_overrides_must_be_supplied_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            estimate_config("unused", input_path_override="held_out.mkv")
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            estimate_config("unused", timestamp_csv_override="held_out.csv")

    def test_velocity_grid_override_requires_five_finite_ordered_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain"):
            estimate_config("unused", velocity_grid_override=(-3, 3, -3, 3))
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_config(
                "unused",
                velocity_grid_override=(-3, 3, -3, 3, float("nan")),
            )
        with self.assertRaisesRegex(ValueError, "minimum"):
            estimate_config("unused", velocity_grid_override=(3, -3, -3, 3, 1))
        with self.assertRaisesRegex(ValueError, "step"):
            estimate_config("unused", velocity_grid_override=(-3, 3, -3, 3, 0))

    def test_max_frames_override_requires_positive_integer(self) -> None:
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                estimate_config("unused", max_frames_override=value)  # type: ignore[arg-type]

    def test_grid_coverage_override_requires_finite_fraction(self) -> None:
        for value in (-0.01, 1.01, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, r"finite and in \[0, 1\]"
            ):
                estimate_config(
                    "unused",
                    minimum_grid_coverage_override=value,
                )

    def test_per_cell_candidate_override_requires_positive_integer(self) -> None:
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "positive integer"
            ):
                estimate_config(
                    "unused",
                    max_candidates_per_cell_override=value,
                )

    def test_track_reservation_overrides_are_all_or_none_and_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "supplied together"):
            estimate_config(
                "unused", track_reservation_position_radius_px_override=8
            )
        for value in (0, -1, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                estimate_config(
                    "unused",
                    track_reservation_position_radius_px_override=value,
                    track_reservation_velocity_radius_px_s_override=3,
                    max_track_reservations_per_window_override=4,
                )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            estimate_config(
                "unused",
                track_reservation_position_radius_px_override=8,
                track_reservation_velocity_radius_px_s_override=3,
                max_track_reservations_per_window_override=1.5,
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            estimate_config(
                "unused",
                track_reservation_minimum_mean_speed_px_s_override=0.25,
            )
        for value in (-0.1, float("nan"), True):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                estimate_config(
                    "unused",
                    track_reservation_position_radius_px_override=3,
                    track_reservation_velocity_radius_px_s_override=1,
                    track_reservation_minimum_mean_speed_px_s_override=value,
                    max_track_reservations_per_window_override=8,
                )


if __name__ == "__main__":
    unittest.main()
