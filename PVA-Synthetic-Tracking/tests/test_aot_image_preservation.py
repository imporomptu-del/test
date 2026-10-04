"""Generated runner fixtures only; no approved source images or experiment fits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("generated_preservation_runner",
        ROOT / "scripts/run_aot_image_preservation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_row(previous=0):
    cells = []
    for i in range(48):
        fitted = dict(training_eligible=True, fit=dict(valid=True,
            native_matrix=[[1., 0., i*1.25], [0., 1., -i*.5], [0., 0., 1.]]))
        cells.append(dict(cell_id=i, arms={arm:copy.deepcopy(fitted) for arm in
            ("global_translation", "local_translation", "local_affine")}))
    return dict(previous_index=previous, current_index=previous+1, cells=cells)


class ImagePreservationInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_exact_full_probe_inventory_and_no_source_dependent_selection(self):
        totals, identifiers = dict(main=0, boundary=0, border=0), set()
        for previous in self.runner.PREVIOUS:
            row = saved_row(previous)
            before = copy.deepcopy(row)
            specs = self.runner.probe_specs(row)
            self.assertEqual(row, before)
            self.assertEqual(len(specs), 440)
            self.assertEqual([sum(s["group"] == name for s in specs) for name in totals], [384, 40, 16])
            for spec in specs:
                totals[spec["group"]] += 1
                self.assertNotIn(spec["id"], identifiers)
                identifiers.add(spec["id"])
                np.testing.assert_allclose(np.array(spec["current_center_xy"])-spec["previous_center_xy"],
                    np.array(spec["nominal_displacement_xy"])+spec["independent_offset_xy"], rtol=0, atol=1e-10)
        self.assertEqual(totals, dict(main=3072, boundary=320, border=128))
        self.assertEqual(len(identifiers), 3520)
        self.assertEqual(3*len(identifiers), 10560)
        self.assertEqual(self.runner.DESIGN["counts"], {**totals, "probes":3520, "arm_evaluations":10560})

    def test_boundary_sweep_uses_fixed_anchor_transform_not_each_step_cell(self):
        row = saved_row()
        specs = self.runner.probe_specs(row)
        for site in range(4):
            sweep = [s for s in specs if s["group"] == "boundary" and s["site_id"] == site and s["amplitude"] == 16]
            self.assertEqual([s["sweep_step"] for s in sweep], [-1., -.5, 0., .5, 1.])
            self.assertEqual(len({tuple(s["nominal_displacement_xy"]) for s in sweep}), 1)
            expected = self.runner.nominal_displacement(row, self.runner.DESIGN["boundary_anchors"][site])
            for s in sweep:
                np.testing.assert_array_equal(s["nominal_displacement_xy"], expected)
            self.assertEqual(len({self.runner.cell_id(s["previous_center_xy"]) for s in sweep}), 2)
            np.testing.assert_allclose(np.diff([s["current_center_xy"] for s in sweep], axis=0),
                np.tile([.5, 0] if site < 2 else [0, .5], (4, 1)), atol=1e-12)

    def test_main_border_parameters_and_fixed_visual_selection(self):
        chosen = []
        for previous in self.runner.PREVIOUS:
            specs = self.runner.probe_specs(saved_row(previous))
            for spec in specs:
                if spec["group"] == "main":
                    rr, cc = divmod(spec["site_id"], 8)
                    np.testing.assert_allclose(spec["previous_center_xy"], [(cc+.5)*306+.25, (rr+.5)*2048/6+.25])
                    self.assertIn(spec["sigma"], (.6, 1.2))
                    self.assertIn(spec["amplitude"], (-16., 16.))
                if self.runner.visual_selected(spec, previous):
                    chosen.append((previous, spec))
        self.assertEqual(len(chosen), 16)
        self.assertEqual(sum(s["group"] == "main" for _, s in chosen), 6)
        self.assertEqual(sum(s["group"] == "boundary" for _, s in chosen), 10)
        self.assertEqual(len(chosen)*3, 48)
        self.assertFalse(any(s["group"] == "border" for _, s in chosen))

    def test_unavailable_nominal_fit_fails_without_identity_replacement(self):
        row = saved_row()
        row["cells"][0]["arms"]["global_translation"]["training_eligible"] = False
        with self.assertRaises(ValueError):
            self.runner.probe_specs(row)

    def test_approved_inventory_is_exact_sixteen_metadata_rows_without_decoding(self):
        metadata = [dict(frame_index=i, img_name=f"generated_{i}.png") for i in range(300)]
        loader = SimpleNamespace(image_metadata=Mock(return_value=metadata), load_approved_image=Mock())
        with patch.object(self.runner, "read", return_value={}):
            rows = self.runner.approved_inventory(loader)
        self.assertEqual([r["frame_index"] for r in rows], [0,1,42,43,85,86,127,128,170,171,212,213,255,256,298,299])
        loader.load_approved_image.assert_not_called()
        metadata[42]["frame_index"] = 41
        with patch.object(self.runner, "read", return_value={}), self.assertRaises(ValueError):
            self.runner.approved_inventory(loader)

    def test_statistics_keep_empty_unavailable_and_signed_rms(self):
        empty = self.runner.statistics([])
        self.assertEqual(empty["count"], 0)
        self.assertIsNone(empty["median"])
        self.assertIsNone(empty["rms"])
        values = self.runner.statistics([-3, 4])
        self.assertEqual(values["median"], 3.5)
        self.assertAlmostEqual(values["rms"], np.sqrt(12.5))
        with self.assertRaises(ValueError):
            self.runner.statistics([np.nan])

    def test_dense_census_keeps_all_cells_masks_and_displacement_seams(self):
        yy, xx = np.indices((12, 16), dtype=np.float32)
        valid = np.ones(xx.shape, bool)
        valid[0, 0] = False
        field = {key:valid.copy() for key in ("numerical", "model_support", "kernel_valid", "valid_pre", "valid")}
        field.update(qx=xx+3*(xx >= 8), qy=yy)
        residual = np.full(xx.shape, -2., np.float32)
        residual[~valid] = np.nan
        with patch.object(self.runner, "WIDTH", 16), patch.object(self.runner, "HEIGHT", 12):
            result = self.runner.dense_summary(field, residual)
        self.assertEqual(len(result["cells"]), 48)
        self.assertEqual(sum(c["native_pixels"] for c in result["cells"]), 192)
        self.assertEqual(sum(c["valid"] for c in result["cells"]), 191)
        self.assertEqual((result["upper_half_valid"], result["lower_half_valid"]), (95, 96))
        vertical = result["cell_seams"][0]["boundaries"]
        self.assertEqual(vertical[3]["coordinate"], 8)
        self.assertEqual(vertical[3]["adjacent_displacement_difference_px"]["maximum"], 3.)
        self.assertEqual(vertical[2]["adjacent_displacement_difference_px"]["maximum"], 0.)


class ImagePreservationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_pinned_inputs_and_artifacts_are_hashed_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, artifact = root/"input.json", root/"code.py"
            source.write_bytes(b"generated")
            artifact.write_bytes(b"# generated")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.object(self.runner, "PINS", dict(source=(source,digest))), \
                    patch.object(self.runner, "ARTIFACTS", dict(code=artifact)):
                result = self.runner.hashes()
                self.assertEqual(result["source"], digest)
                self.assertEqual(result["code"], hashlib.sha256(artifact.read_bytes()).hexdigest())
                source.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    self.runner.hashes()
            linked = root/"linked.py"
            linked.symlink_to(artifact)
            with patch.object(self.runner, "PINS", {}), patch.object(self.runner, "ARTIFACTS", dict(code=linked)), self.assertRaises(ValueError):
                self.runner.hashes()

    def test_manifest_changes_or_current_hash_mutation_fail_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            current = dict(core_sha256="a"*64)
            manifest = dict(schema=self.runner.PLAN_SCHEMA, design=copy.deepcopy(self.runner.DESIGN), hashes=current)
            path = output/"manifest.json"
            path.write_text(json.dumps(manifest))
            with patch.object(self.runner, "OUTPUT", output), patch.object(self.runner, "hashes", return_value=current):
                self.assertIn("manifest_sha256", self.runner.bound_hashes(output))
                changed = copy.deepcopy(manifest)
                changed["design"]["support_erosion_radius"] = 1
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.runner.bound_hashes(output)
                path.write_text(json.dumps(manifest))
                with patch.object(self.runner, "hashes", return_value=dict(core_sha256="b"*64)), self.assertRaises(ValueError):
                    self.runner.bound_hashes(output)

    def test_freeze_checks_before_after_hashes_and_never_decodes_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()/"fresh"
            loader = SimpleNamespace(load_approved_image=Mock())
            before = dict(image_loader_sha256="a"*64)
            with patch.object(self.runner, "OUTPUT", output), \
                    patch.object(self.runner, "hashes", return_value=before), \
                    patch.object(self.runner, "load_module", return_value=loader), \
                    patch.object(self.runner, "approved_inventory", return_value=[dict(frame_index=i) for i in range(16)]), \
                    patch("builtins.print"):
                self.runner.freeze(output)
                with self.assertRaises(ValueError):
                    self.runner.freeze(output)
            loader.load_approved_image.assert_not_called()
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertFalse(manifest["source_images_opened"])
            self.assertEqual(manifest["hashes"], before)

    def test_changed_during_freeze_creates_no_accepted_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()/"fresh"
            with patch.object(self.runner, "OUTPUT", output), \
                    patch.object(self.runner, "hashes", side_effect=[dict(image_loader_sha256="a"*64), {}]), \
                    patch.object(self.runner, "load_module", return_value=object()), \
                    patch.object(self.runner, "approved_inventory", return_value=[]), self.assertRaises(ValueError):
                self.runner.freeze(output)
            self.assertFalse(output.exists())

    def test_existing_outputs_and_dangling_links_stop_before_hashing_or_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            with patch.object(self.runner, "OUTPUT", output), patch.object(self.runner, "bound_hashes") as hashes:
                for name in ("result.json", "failure.json", "review"):
                    path = output/name
                    for linked in (False, True):
                        if linked:
                            path.symlink_to(output/"missing")
                        else:
                            path.write_text("preserved")
                        with self.subTest(name=name, linked=linked), self.assertRaises(ValueError):
                            self.runner.run(output)
                        hashes.assert_not_called()
                        path.unlink()

    def test_outside_scope_and_exclusive_output_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()
            with patch.object(self.runner, "OUTPUT", output), self.assertRaises(ValueError):
                self.runner.check_scope(output/"other")
            path = output/"generated.json"
            self.runner.write_exclusive(path, dict(generated=True))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.runner.write_exclusive(path, dict(generated=False))
            self.assertEqual(path.read_bytes(), before)


def load_auxiliary(name):
    spec = importlib.util.spec_from_file_location("generated_"+name,ROOT/"scripts"/(name+".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ImagePreservationTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = load_auxiliary("aot_image_compensation")
        cls.controls = load_auxiliary("aot_image_temporal_controls")
        cls.result = cls.controls.run_generated(cls.core)

    def test_complete_fixed_temporal_inventory_and_declared_denominators(self):
        result = self.result
        self.assertTrue(result["passed"])
        self.assertFalse(result["source_images_used"])
        self.assertFalse(result["detector_run"])
        self.assertEqual((len(result["trajectories"]),len(result["support_dropouts"]),len(result["static_repeats"])),(72,3,8))
        all_cases = result["trajectories"]+result["support_dropouts"]+result["static_repeats"]
        self.assertEqual(sum(len(c["frames"]) for c in all_cases),747)
        self.assertEqual(sum(len(c["adjacent"]) for c in all_cases),664)
        self.assertEqual(sum(c["missing_frames"] for c in all_cases),6)
        self.assertEqual(sum(c["missing_adjacent_pairs"] for c in all_cases),9)
        inventory = {(c["camera"],c["sigma_px"],c["signed_peak_dn"],tuple(c["initial_phase_xy"])) for c in result["trajectories"]}
        expected = {(camera,sigma,amplitude,phase) for camera in ("identity","translation","affine")
            for sigma in (.6,1.2) for amplitude in (-8.,8.,-32.,32.) for phase in ((0.,0.),(.25,.25),(.5,.5))}
        self.assertEqual(inventory,expected)

    def test_dropout_remains_missing_and_never_bridges_frame_two_to_five(self):
        for case in self.result["support_dropouts"]:
            self.assertEqual([f["frame_index"] for f in case["frames"] if not f["available"]],[3,4])
            self.assertEqual([(p["previous_frame"],p["current_frame"]) for p in case["adjacent"]],[(i,i+1) for i in range(8)])
            self.assertEqual([(p["previous_frame"],p["current_frame"]) for p in case["adjacent"] if not p["available"]],[(2,3),(3,4),(4,5)])
            for p in case["adjacent"]:
                if not p["available"]:
                    self.assertEqual(p["common_pixel_count"],0)
                    self.assertIsNone(p["centroid_step_error_px"])
                    self.assertIsNone(p["isolated_target_residual"]["oracle_rmse_dn"])
                    self.assertIsNone(p["isolated_target_residual"]["observed"]["centroid_xy"])
            clean_match = self.controls.trajectory(self.core,case["camera"],.6,16.,(0.,0.))
            self.assertEqual(case["frames"][5],clean_match["frames"][5],"Recovery must sample the original current frame")

    def test_static_repeated_originals_have_zero_residual_and_null_locations(self):
        for case in self.result["static_repeats"]:
            identities = [f["original_source_sha256"] for f in case["frames"]]
            self.assertTrue(all(value==identities[0] for value in identities))
            for pair in case["adjacent"]:
                self.assertTrue(pair["available"])
                metric = pair["isolated_target_residual"]
                self.assertEqual(metric["observed"]["l1_dn"],0.)
                self.assertEqual(metric["observed"]["energy_dn_squared"],0.)
                self.assertIsNone(metric["observed"]["peak_xy"])
                self.assertIsNone(metric["observed"]["centroid_xy"])
                self.assertIsNone(metric["energy_ratio"])
                self.assertIsNone(metric["core_observed"]["template_response_dn"])
                self.assertEqual(pair["clean_adjacent_residual"]["l1_dn"],0.)

    def test_affine_reference_target_path_and_original_frame_generation_are_independent(self):
        case = next(c for c in self.result["trajectories"] if c["camera"]=="affine" and c["initial_phase_xy"]==[.25,.25])
        for frame in case["frames"]:
            t=frame["frame_index"]
            linear=np.array([[1+.001*t,.002*t],[-.001*t,1-.0005*t]])
            offset=np.array([1224.,1024.])-linear@np.array([1224.,1024.])+[.25*t,-.125*t]
            reference=np.array([1222+.5*t+.25,1024+.0625*t*t+.25])
            np.testing.assert_allclose(frame["target_reference_xy"],reference,atol=1e-8,rtol=0)
            np.testing.assert_allclose(frame["target_current_xy"],linear@reference+offset,atol=1e-8,rtol=0)
        first=self.controls.render_original("affine",7,.6,16.,(.25,.25))
        self.controls.render_original("translation",2,1.2,-32.,(.5,.5))
        second=self.controls.render_original("affine",7,.6,16.,(.25,.25))
        np.testing.assert_array_equal(first[3],second[3])
        np.testing.assert_array_equal(first[4],second[4])


def summary_record(full=True,roi_full=True,partial=False,unavailable=False,value=1.):
    return dict(full_support_eligible=full,full_roi_support=roi_full,partial=partial,unavailable=unavailable,
        source_psf_partial=False,support_reasons=[],valid_fraction=1. if roi_full else .5,
        oracle_absolute_mass_coverage=None if unavailable else 1.,metrics=dict(
            current_target_delta_comparison=dict(ratios=dict(peak_abs_dn=value,l1_dn=value,l2_dn=value),centroid_error_px=value),
            current_target_delta=dict(centroid_xy=[value,0.] if value is not None else None)))


class ImagePreservationSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary=load_auxiliary("summarize_aot_image_preservation")

    def test_summary_null_denominators_and_nonfinite_rejection(self):
        result=self.summary.stats([None,2.,None,4.])
        self.assertEqual((result["total"],result["count"],result["null_count"]),(4,2,2))
        self.assertEqual(result["median"],3.)
        self.assertIsNone(self.summary.stats([None,None])["maximum"])
        with self.assertRaises(ValueError):self.summary.stats([float("nan")])

    def test_target_full_support_is_distinct_from_partial_roi_and_unavailable(self):
        rows=[summary_record(),summary_record(roi_full=False,partial=True),
              summary_record(full=False,roi_full=False,partial=True,unavailable=True,value=None)]
        result=self.summary.arm_summary(rows)
        self.assertEqual((result["total"],result["full_support_eligible"],result["roi_fully_supported"],
                          result["partial"],result["unavailable"]),(3,2,1,2,1))
        self.assertEqual(result["metrics_on_all_available"]["current_peak_ratio"]["null_count"],1)
        self.assertEqual(result["metrics_on_full_support"]["current_peak_ratio"]["total"],2)

    def test_common_probe_metrics_compare_identical_ids_not_each_arm_survivors(self):
        flags=((True,True,False),(True,False,True),(True,True,True))
        probes=[dict(id=f"generated{i}",arms={arm:summary_record(full=flag,value=float(10*i+j))
            for j,(arm,flag) in enumerate(zip(self.summary.ARMS,row))}) for i,row in enumerate(flags)]
        result=self.summary.summarize_probes(probes)
        expected=(["generated0","generated2"],["generated1","generated2"],["generated2"])
        for pair,ids in zip(result["pairwise"].values(),expected):
            self.assertEqual(pair["query_ids"],ids)
            self.assertEqual(pair["common_full_support"],len(ids))
            self.assertEqual(pair["missing_count"],3-len(ids))
            for metrics in pair["arms"].values():self.assertEqual(metrics["current_peak_ratio"]["total"],len(ids))
        self.assertEqual(result["all_three_full_support_count"],1)

    def test_boundary_sequences_keep_missing_centroids_and_do_not_bridge_gaps(self):
        probes=[]
        for site in range(4):
            for sign in (-16.,16.):
                for i,step in enumerate((-1.,-.5,0.,.5,1.)):
                    probes.append(dict(group="boundary",site_id=site,amplitude=sign,sweep_step=step,
                        arms={arm:summary_record(full=i!=2,value=float(i)) for arm in self.summary.ARMS}))
        result=self.summary.boundary_sequences([dict(previous_index=0,probes=probes)])
        self.assertEqual(len(result),24)
        for row in result:
            self.assertEqual(row["current_centroids_xy"],[[0.,0.],[1.,0.],None,[3.,0.],[4.,0.]])
            self.assertEqual(row["adjacent_centroid_steps_xy"],[[1.,0.],None,None,[1.,0.]])
            self.assertEqual(row["adjacent_eligibility_changes"],2)


if __name__ == "__main__":
    unittest.main()
