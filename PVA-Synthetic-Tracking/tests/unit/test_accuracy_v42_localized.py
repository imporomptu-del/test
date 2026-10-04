"""Synthetic numerical contracts, not validation of an airborne classifier."""

import importlib.util
import inspect
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "accuracy_v42_localized", Path(__file__).parents[2] / "scripts" / "accuracy_v42_localized.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)

Y, X = np.indices((129, 129))


def point(x, y, amplitude=30.0):
    return amplitude * np.exp(-((X-x)**2 + (Y-y)**2)/2.0)


def fixture(polarity="bright", centres=None, offset=(0.0, 0.0)):
    background = 50.0 + 20.0*(X >= 64) + 8.0*np.sin(Y/12.0)
    centres = [[29+4*i, 64] for i in range(8)] if centres is None else centres
    sign = 1 if polarity == "bright" else -1
    history = np.stack([background + sign*point(*centre) for centre in centres])
    current = background + sign*point(64+offset[0], 64+offset[1])
    return current, history, centres, np.asarray(offset), polarity


def test_moving_feature_over_fixed_edge_has_localized_evidence(polarity):
    result = core.evaluate_localized(*fixture(polarity))
    assert result["available"]
    assert result["mse_augmented"] < result["mse_stationary"] * 0.01
    assert result["model_complexity"]["extra_augmented_parameters"] == 1
    assert result["common_support_count"] == 625
    assert result["fold_support_counts"] == [313, 312]
    assert all(fold["moving_nonnegative_amplitude"] > 0 for fold in result["folds"])
    json.dumps(result, allow_nan=False)


def test_independently_flickering_fixed_lights_get_signed_amplitudes():
    current, history, centres, offset, polarity = fixture()
    for index in range(8):
        history[index] += point(59, 60, 20+2*index) + point(70, 67, 40-index)
    background = 50.0 + 20.0*(X >= 64) + 8.0*np.sin(Y/12.0)
    current = background + point(59, 60, 55) + point(70, 67, 10)
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"]
    assert result["components"]["fixed_anchor_count"] >= 2
    for fold in result["folds"]:
        assert min(fold["stationary_signed_amplitudes"]) < 0
        assert max(fold["stationary_signed_amplitudes"]) > 0
    assert abs(result["advantage_stationary_minus_augmented"]) < 0.01


def test_exact_redundancy_is_ambiguous_not_a_rejection():
    args = fixture()
    components = core.prepare_components(*args[1:])
    components["fixed_templates"] = components["moving_template"][None].copy()
    components["metadata"]["fixed_anchor_count"] = 1
    with mock.patch.object(core, "prepare_components", lambda *unused: components):
        result = core.evaluate_localized(*args)
    assert result["available"] and result["ambiguous"]
    assert "moving_template_redundant_with_fixed_dictionary" in result["ambiguity_reasons"]
    assert result["moving_relative_energy_outside_fixed_span"] < 1.0e-8
    assert np.isclose(result["mse_augmented"], result["mse_stationary"])
    assert all(not fold["moving_amplitude_identifiable"] for fold in result["folds"])
    assert "reject" not in result and "class" not in result


def test_hover_or_slow_overlapping_masks_remain_unknown(step):
    centres = [[64-step*(8-i), 64] for i in range(8)]
    result = core.evaluate_localized(*fixture(centres=centres))
    assert not result["available"]
    assert "insufficient_causal_foreground_template" in result["reasons"]
    assert result["mse_stationary"] is None
    assert result["mse_augmented"] is None
    assert "reject" not in result


def test_blinking_history_gaps_use_only_actual_visible_stamps():
    current, history, centres, offset, polarity = fixture()
    for frame in (1, 4, 6):
        history[frame] -= point(*centres[frame])
        centres[frame] = None
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"]
    assert result["components"]["usable_moving_stamp_indices"] == [0, 2, 3, 5, 7]


def test_too_few_prior_actuals_never_synthesizes_track_history():
    current, history, centres, offset, polarity = fixture()
    centres[:6] = [None]*6
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert not result["available"]
    assert "insufficient_causal_foreground_template" in result["reasons"]


def test_supplied_turning_subpixel_prediction_is_not_searched_or_recentred():
    centres = [[32, 42], [36, 44], [40, 47], [44, 51], [48, 56], [52, 62], [56, 63], [60, 64]]
    args = fixture(centres=centres, offset=(0.3, -0.2))
    components = core.prepare_components(*args[1:])
    expected = core._place(components["moving_stamp"], np.array([64.3, 63.8]))
    np.testing.assert_array_equal(expected, components["moving_template"])
    result = core.evaluate_localized(*args)
    assert result["available"]
    assert result["mse_augmented"] < result["mse_stationary"]


def test_broad_deformation_output_contains_no_physical_class_or_policy():
    current, history, centres, offset, polarity = fixture()
    current += 12*np.exp(-((X-65)**2/50 + (Y-62)**2/90))
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"]
    assert not ({"class", "airborne", "accept", "reject", "confidence", "probability"} & set(result))


def test_masked_saturation_and_missing_pixels_are_excluded_not_filled():
    current, history, centres, offset, polarity = fixture()
    current[64, 64] = np.nan
    history[:, 70, 70] = np.nan
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"]
    assert result["common_support_count"] < 625
    components = core.prepare_components(history, centres, offset, polarity)
    assert np.isnan(components["background"][70, 70])
    assert components["background_observation_counts"][70, 70] == 0
    json.dumps(result, allow_nan=False)


def test_missing_entire_annulus_is_unknown():
    current, history, centres, offset, polarity = fixture()
    distance = np.maximum(np.abs(X-64), np.abs(Y-64))
    current[(distance >= 16) & (distance <= 30)] = np.nan
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert not result["available"]
    assert result["reasons"] == ["insufficient_annulus_support"]


def test_flat_background_gain_is_unidentifiable_and_unknown():
    current, history, centres, offset, polarity = fixture()
    old_background = 50 + 20*(X >= 64) + 8*np.sin(Y/12)
    current = current-old_background+50
    history = history-old_background+50
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert not result["available"]
    assert "flat_or_rank_deficient_annulus_background" in result["reasons"]


def test_annulus_fit_cannot_see_current_core():
    current, history, centres, offset, polarity = fixture()
    first = core.evaluate_localized(current, history, centres, offset, polarity)
    current[52:77, 52:77] += 11.0
    second = core.evaluate_localized(current, history, centres, offset, polarity)
    assert first["background_fit"] == second["background_fit"]
    assert first["components"] == second["components"]


def test_learned_components_have_no_current_image_argument_or_dependency():
    assert list(inspect.signature(core.prepare_components).parameters) == [
        "history129", "prior_centers_xy", "predicted_offset_xy", "polarity"]
    current, history, centres, offset, polarity = fixture()
    first = core.prepare_components(history, centres, offset, polarity)
    current[:] = 133
    second = core.prepare_components(history, centres, offset, polarity)
    for name in ("background", "moving_template", "fixed_templates", "moving_stamp"):
        np.testing.assert_array_equal(first[name], second[name])
    assert first["metadata"] == second["metadata"]


def test_noise_reproducible_and_inputs_unmodified():
    current, history, centres, offset, polarity = fixture()
    rng = np.random.default_rng(42)
    current += rng.normal(0, 0.4, current.shape)
    history += rng.normal(0, 0.4, history.shape)
    original_current, original_history, original_offset = current.copy(), history.copy(), offset.copy()
    original_centres = json.dumps(centres)
    first = core.evaluate_localized(current, history, centres, offset, polarity)
    second = core.evaluate_localized(current, history, centres, offset, polarity)
    assert first == second
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(history, original_history)
    np.testing.assert_array_equal(offset, original_offset)
    assert json.dumps(centres) == original_centres


def test_fixed_dictionary_cap_is_flagged_ambiguous():
    current, history, centres, offset, polarity = fixture()
    for x, y in ((50, 50), (59, 50), (70, 50), (78, 59), (75, 75), (55, 76)):
        history += point(x, y, 20)
        current += point(x, y, 20)
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["components"]["persistent_anchor_count_before_cap"] > 4
    assert result["components"]["fixed_anchor_count"] <= 4
    assert result["ambiguous"]
    assert "persistent_anchor_dictionary_truncated" in result["ambiguity_reasons"]


def test_negative_moving_amplitude_is_exactly_constrained_to_zero():
    current, history, centres, offset, polarity = fixture()
    current -= 2*point(64, 64)
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"]
    assert all(fold["moving_nonnegative_constraint_active"] for fold in result["folds"])
    assert all(fold["moving_nonnegative_amplitude"] == 0.0 for fold in result["folds"])
    assert result["mse_augmented"] == result["mse_stationary"]


def test_malformed_input_rejected(field, value):
    args = list(fixture())
    args[field] = value
    with unittest.TestCase().assertRaises(ValueError):
        core.evaluate_localized(*args)


def test_bilinear_zero_weight_nan_does_not_destroy_integer_sample():
    image = np.arange(25, dtype=float).reshape(5, 5)
    image[2, 3] = np.nan
    assert core._sample(image, np.array([2.0]), np.array([2.0]))[0] == 12
    assert np.isnan(core._sample(image, np.array([2.5]), np.array([2.0]))[0])


def test_moving_mask_uses_prior_not_current_predicted_centre():
    current, history, centres, offset, polarity = fixture()
    components = core.prepare_components(history, centres, offset, polarity)
    # The earliest source at x29 is excluded from that frame, whereas x100 is
    # untouched. The predicted current centre never supplies a history mask.
    assert components["background_observation_counts"][64, 29] < 8
    assert components["background_observation_counts"][64, 100] == 8


def test_joint_linear_fit_recovers_known_signed_and_nonnegative_coefficients():
    current, history, centres, offset, polarity = fixture()
    components = core.prepare_components(history, centres, offset, polarity)
    moving = core._place(core._positive_unit_stamp(core._stamp(point(64, 64), (64, 64))), (64, 64))
    fixed = np.stack([core._place(core._positive_unit_stamp(core._stamp(point(64, 64), (64, 64))), centre)
                      for centre in ((58, 60), (70, 68))])
    components["moving_template"] = moving
    components["fixed_templates"] = fixed
    components["metadata"]["fixed_anchor_count"] = 2
    current = components["background"] + 11*fixed[0] - 7*fixed[1] + 19*moving
    with mock.patch.object(core, "prepare_components", lambda *unused: components):
        result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"] and not result["ambiguous"]
    assert result["mse_augmented"] < 1e-20
    for fold in result["folds"]:
        np.testing.assert_allclose(fold["augmented_signed_fixed_amplitudes"], [11, -7], atol=1e-10)
        assert abs(fold["moving_nonnegative_amplitude"] - 19) < 1e-10


def test_reported_mse_weights_all_heldout_pixels_exactly_once():
    current, history, centres, offset, polarity = fixture()
    rng = np.random.default_rng(814)
    current += rng.normal(0, 0.8, current.shape)
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    for name in ("stationary", "augmented"):
        expected = sum(fold[f"mse_{name}"] * fold["heldout_count"] for fold in result["folds"]) / result["common_support_count"]
        assert abs(result[f"mse_{name}"]-expected) < 1e-12
    assert sum(fold["heldout_count"] for fold in result["folds"]) == result["common_support_count"]


def test_one_checkerboard_without_moving_support_is_explicitly_ambiguous():
    args = fixture()
    components = core.prepare_components(*args[1:])
    moving = np.zeros((129, 129))
    moving[64, 64] = 1.0
    components["moving_template"] = moving
    components["fixed_templates"] = np.empty((0, 129, 129))
    components["metadata"]["fixed_anchor_count"] = 0
    with mock.patch.object(core, "prepare_components", lambda *unused: components):
        result = core.evaluate_localized(*args)
    assert result["available"] and result["ambiguous"]
    assert "moving_template_unidentifiable_on_training_fold:1" in result["ambiguity_reasons"]
    assert not result["folds"][1]["moving_amplitude_identifiable"]


def test_natural_fixed_light_at_causal_prediction_is_physical_ambiguity():
    current, history, centres, offset, polarity = fixture()
    for frame in range(8):
        history[frame] += point(64, 64, 25+frame)
    current += point(64, 64, 35)
    result = core.evaluate_localized(current, history, centres, offset, polarity)
    assert result["available"] and result["ambiguous"]
    assert result["components"]["persistent_anchor_overlaps_prediction"]
    assert result["components"]["persistent_anchor_centres_near_prediction"]
    assert "causal_prediction_overlaps_persistent_fixed_anchor" in result["ambiguity_reasons"]
    assert "reject" not in result


def load_tests(loader, tests, pattern):
    """Expose the small parameterized numerical cases to standard unittest."""
    suite = unittest.TestSuite()
    parameterized = {
        "test_moving_feature_over_fixed_edge_has_localized_evidence": [("bright",), ("dark",)],
        "test_hover_or_slow_overlapping_masks_remain_unknown": [(0.0,), (0.2,)],
        "test_malformed_input_rejected": [
            (0, np.zeros((25, 25))), (0, np.full((129, 129), np.inf)),
            (1, np.zeros((7, 129, 129))), (1, np.zeros((8, 129, 129), dtype=bool)),
            (2, [None]*7), (2, [[np.nan, 64]]*8),
            (3, [0.51, 0]), (3, [True, False]), (4, "unknown"),
        ],
    }
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            for index, arguments in enumerate(parameterized.get(name, [()])):
                def call(function=function, arguments=arguments):
                    function(*arguments)
                call.__name__ = f"{name}_{index}"
                suite.addTest(unittest.FunctionTestCase(call, description=call.__name__))
    return suite


if __name__ == "__main__":
    unittest.main()
