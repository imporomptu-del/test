"""Generated arrays/metadata only. Never open experiment media or remote data."""
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/review_feature_residual_pixels.py"
SPEC = importlib.util.spec_from_file_location("feature_residual_pixel_tests", PATH)
pixels = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pixels)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gray(shape=(80, 80)):
    y, x = np.indices(shape)
    return np.rint(95 + 15 * np.sin(x * .21) + 20 * np.cos(y * .17)).astype(np.uint8)


def arrays():
    p = np.array([[30 + i, 33] for i in range(12)], np.float32)
    residuals = np.array([.1, .2, .3, .4, .8, .8, .9, 1.5, 2, 2, 3, 4], np.float64)
    q = p + np.column_stack((residuals, np.zeros(12))).astype(np.float32)
    residuals = np.linalg.norm(q.astype(np.float64) - p, axis=1)
    return p, q, np.arange(12, dtype=np.float32), np.zeros(12, np.float32), residuals <= 1, residuals


def descriptor(value):
    value = np.ascontiguousarray(value)
    raw = value.tobytes()
    return dict(dtype=value.dtype.str, shape=list(value.shape), data_base64=base64.b64encode(raw).decode(),
                sha256=hashlib.sha256(raw).hexdigest())


def trace_row(index):
    p, q, s, fb, mask, residuals = arrays()
    return dict(previous_frame_index=index-1, current_frame_index=index,
        previous_timestamp_ns=(index-1)*100_000_000, current_timestamp_ns=index*100_000_000,
        full_image_size=[4784,3190], motion_image_size=[2392,1595],
        native_gray_pixel_sha256=dict(previous="a"*64,current="a"*64),
        correspondence=dict(previous_points=descriptor(p),current_points=descriptor(q),harris_scores=descriptor(s),
            forward_backward_error_px=descriptor(fb),metrics=dict(accepted_count=12),backends={}),
        fit=dict(model="translation",quality_status="accepted",rejection_reasons=[],
            parameters=dict(translation_x_px=0,translation_y_px=0),metrics=dict(correspondence_count=12),
            previous_to_current_matrix=descriptor(np.eye(3)),inlier_mask=descriptor(mask),residuals_px=descriptor(residuals)))


class PhotometryTests(unittest.TestCase):
    def test_fixed_support_and_fractional_ramp(self):
        self.assertEqual(pixels.PATCH_SIZE, 23)
        y, x = np.indices((64,64))
        image = (2*x+y).astype(np.uint8)
        before = image.copy()
        result, status = pixels.bilinear_patch(image, (30.25,31.75))
        yy, xx = np.meshgrid(np.arange(-11,12),np.arange(-11,12),indexing="ij")
        np.testing.assert_allclose(result,2*(30.25+xx)+31.75+yy,rtol=0,atol=0)
        self.assertEqual(status,"available")
        np.testing.assert_array_equal(image,before)

    def test_edges_unavailable_not_padded_and_exact_boundary_allowed(self):
        image=gray((23,23))
        result,status=pixels.bilinear_patch(image,(11,11))
        self.assertEqual(status,"available")
        np.testing.assert_array_equal(result,image)
        for point in ((10.999,11),(11,11.001),(-1,12),(90,12)):
            result,status=pixels.bilinear_patch(image,point)
            self.assertIsNone(result)
            self.assertEqual(status,"edge_insufficient_native_support")

    def test_nonfinite_coordinates_flagged(self):
        for point in ((np.nan,20),(20,np.inf)):
            result=pixels.photometry(gray(),gray(),point,(30,30))
            self.assertEqual(result["previous_status"],"nonfinite_coordinate")
            self.assertIsNone(result["previous"])
            self.assertIsNone(result["aligned_zero_mean_rmse"])
        with self.assertRaisesRegex(ValueError,"uint8"):
            pixels.bilinear_patch(np.ones((50,50),np.float32),(25,25))

    def test_flat_structure_unavailable_anisotropy_not_one(self):
        result=pixels.patch_statistics(np.full((23,23),40))
        self.assertEqual(result["mean"],40)
        self.assertEqual(result["std"],0)
        self.assertEqual(result["gradient_structure_eigenvalues_ascending"],[0,0])
        self.assertIsNone(result["anisotropy"])
        self.assertEqual(result["gradient_support_pixels"],441)

    def test_ramp_structure_tensor_units(self):
        y,x=np.indices((23,23))
        result=pixels.patch_statistics(2*x+3*y)
        np.testing.assert_allclose(result["gradient_structure_tensor"],[[4,6],[6,9]])
        np.testing.assert_allclose(result["gradient_structure_eigenvalues_ascending"],[0,13],atol=1e-14)
        self.assertAlmostEqual(result["anisotropy"],1)

    def test_translated_patch_and_photometric_offset(self):
        before=gray()
        after=np.roll(before,(1,2),(0,1))+np.uint8(17)
        result=pixels.photometry(before,after,(30.25,32.5),(32.25,33.5))
        self.assertAlmostEqual(result["aligned_zero_mean_rmse"],0,places=12)
        self.assertAlmostEqual(result["aligned_mean_change"],17)
        self.assertAlmostEqual(result["aligned_raw_rmse"],17)
        wrong=pixels.photometry(before,after,(30.25,32.5),(37.25,33.5))
        self.assertGreater(wrong["aligned_zero_mean_rmse"],1)

    def test_selection_exact_tie_breaking_and_empty_populations(self):
        residuals=np.array([.8,.8,.9,.1,2,2,3,4],float)
        mask=residuals<=1
        self.assertEqual(pixels.selected_indices(residuals,mask),dict(inliers=[2,0,1],outliers=[7,6,4]))
        self.assertEqual(pixels.selected_indices(np.empty(0),np.empty(0,bool)),dict(inliers=[],outliers=[]))
        with self.assertRaises(ValueError):pixels.selected_indices([np.nan],np.array([False]))

    def test_every_point_retained_missing_display_patch_not_replaced(self):
        data=arrays()
        data[0][6]=[2,2]
        data[1][6]=[2.9,2]
        result=pixels.analyze_pixels(gray(),gray(),data)
        self.assertEqual(result["all_lk_accepted_points"],12)
        self.assertEqual(len(result["all_points"]),12)
        self.assertEqual(result["valid_aligned_patches"],11)
        self.assertEqual(result["displayed_point_indices"]["inliers"][0],6)
        self.assertIsNone(result["all_points"][6]["photometry"]["aligned_zero_mean_rmse"])
        self.assertEqual(result["original_ransac_inliers"]+result["original_ransac_outliers"],12)


class GrayIdentityTests(unittest.TestCase):
    def test_actual_frame_pixel_hash_includes_dtype_shape(self):
        image=np.arange(120,dtype=np.uint8).reshape(10,12)
        expected=hashlib.sha256(b"seaqr-frame-pixels-v1\0|u1\0"+b"10x12\0"+image.tobytes()).hexdigest()
        self.assertEqual(pixels.native_pixel_sha(image),expected)
        self.assertEqual(pixels.native_pixel_sha(image,40),expected)
        self.assertNotEqual(pixels.native_pixel_sha(image.reshape(12,10)),expected)

    def test_adjacent_pair_hash_agreement(self):
        rows={3:dict(previous_frame_index=2,current_frame_index=3,native_gray_pixel_sha256=dict(previous="a"*64,current="b"*64)),
              4:dict(previous_frame_index=3,current_frame_index=4,native_gray_pixel_sha256=dict(previous="b"*64,current="c"*64))}
        self.assertEqual(pixels.expected_gray_hashes(rows),{2:"a"*64,3:"b"*64,4:"c"*64})
        rows[4]["native_gray_pixel_sha256"]["previous"]="d"*64
        with self.assertRaisesRegex(ValueError,"disagree"):pixels.expected_gray_hashes(rows)

    def test_causal_reader_never_seeks_and_verifies_before_yield(self):
        image=gray()
        digest=pixels.native_pixel_sha(image)
        rows={i:dict(previous_frame_index=i-1,current_frame_index=i,
                      native_gray_pixel_sha256=dict(previous=digest,current=digest)) for i in (3,4)}
        instances=[]
        class Reader:
            def __init__(self,source,shape,execution,max_frames):
                self.expected=673;self.fps=10.;self.count=0;self.maximum=max_frames;self.closed=False
                self.execution=execution;instances.append(self)
            def __enter__(self):return self
            def __exit__(self,*args):self.closed=True
            def read(self):
                result=SimpleNamespace(index=self.count,gray=image)
                self.count+=1
                return result,0
        with patch("tiny_target.visible_decode.VisibleFrameReader",Reader),patch.object(pixels,"NATIVE_SHAPE",image.shape):
            result=list(pixels.decode_verified_pairs("generated-only",rows))
            self.assertEqual([r[0] for r in result],[3,4])
            self.assertEqual(instances[-1].count,5)
            self.assertEqual(instances[-1].maximum,5)
            self.assertEqual(instances[-1].execution,"sequential")
            self.assertTrue(instances[-1].closed)
            rows[3]["native_gray_pixel_sha256"]["previous"]="f"*64
            with self.assertRaisesRegex(ValueError,"Native gray differs"):
                next(pixels.decode_verified_pairs("generated-only",rows))
            self.assertEqual(instances[-1].count,3)
            self.assertTrue(instances[-1].closed)


class RenderingTests(unittest.TestCase):
    def test_generated_renders_keep_native_scale_and_bounded_examples(self):
        import cv2
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            image=gray((100,120));data=arrays()
            stats=pixels.analyze_pixels(image,image,data)
            overview=pixels.render_overview(directory/"overview.png",image,"0170",452,data,np.eye(3))
            contact=pixels.render_contact_sheet(directory/"patches.png",image,image,"0170",452,data,stats)
            png=cv2.imread(str(directory/"overview.png"))
            self.assertEqual(png.shape,(224,120,3))
            np.testing.assert_array_equal(png[124:134,:10,0],image[:10,:10])
            self.assertEqual(overview["vector_magnification"],32)
            self.assertFalse(contact["contrast_enhancement"])
            self.assertEqual(contact["display_scales"],[1,8])
            self.assertEqual(sum(map(len,contact["displayed_point_indices"].values())),6)
            self.assertEqual(pixels.sha(directory/"patches.png"),contact["sha256"])
            with self.assertRaisesRegex(ValueError,"replace"):
                pixels.render_contact_sheet(directory/"patches.png",image,image,"0170",452,data,stats)

    def test_inconsistent_saved_matrix_rejected_not_refitted(self):
        with tempfile.TemporaryDirectory() as directory:
            matrix=np.eye(3);matrix[0,2]=2
            path=Path(directory)/"bad.png"
            with self.assertRaisesRegex(ValueError,"transform/residual"):
                pixels.render_overview(path,gray(),"0170",452,arrays(),matrix)
            self.assertFalse(path.exists())


class BindingTests(unittest.TestCase):
    def test_strict_json_and_post_render_hash_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"metadata.json"
            for value in ('{"a":1,"a":2}','{"a":NaN}','{"a":1e999}'):
                path.write_text(value)
                with self.assertRaises(ValueError):pixels.read_json(path)
            path.write_text('{"a":1}')
            bound={str(path.resolve()):pixels.sha(path)}
            pixels.check_hashes(bound)
            path.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError,"changed"):pixels.check_hashes(bound)

    def fixture(self,root):
        repo=root/"skymove"
        (repo/"scripts").mkdir(parents=True)
        (repo/"tiny_target").mkdir()
        (repo/"tests/unit").mkdir(parents=True)
        own=repo/"scripts/review_feature_residual_pixels.py";own.write_text("# generated fixture\n")
        analyzer=repo/"scripts/analyze_feature_residual_trace.py"
        analyzer.write_text((ROOT/"scripts/analyze_feature_residual_trace.py").read_text())
        for name in ("tiny_target/types.py","tiny_target/visible_decode.py","tests/unit/test_feature_residual_pixels.py"):
            (repo/name).write_text("# generated fixture\n")
        bindings={str(analyzer):pixels.sha(analyzer)}
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            pixels.write_json(path,value)
            bindings[str(path)]=pixels.sha(path)
            return path
        plan=dict(local_visual_followup=dict(pair_ends=pixels.PAIR_ENDS),
                  sources={c:dict(sha256=h,frames=673) for c,h in pixels.SOURCE_HASHES.items()},
                  capture_pairs=pixels.PAIR_ENDS)
        plan_path=save(root/"bundle/feature_residual_trace_plan.json",plan)
        clips={}
        for clip,ends in pixels.PAIR_ENDS.items():
            trace=save(root/clip/"trace.json",dict(schema="seaqr.feature-residual-trace.v1.trace",clip=clip,
                source=plan["sources"][clip],capture_pairs=ends,rows=[trace_row(i) for i in ends]))
            parity=save(root/clip/"parity.json",dict(schema="seaqr.feature-residual-trace.v1.parity",passed=True,
                                                    rows_compared=673,mismatch_frames=[]))
            save(root/clip/"execution_receipt.json",dict(schema="seaqr.discovery-feature-residual-trace.v1",passed=True,
                processed_frames=673,clip=clip,trace_sha256=bindings[str(trace)],parity_sha256=bindings[str(parity)]))
            clips[clip]=dict(trace_path=str(trace),trace_sha256=bindings[str(trace)],full_causal_non_timing_parity_verified_frames=673,
                rows=[dict(current_frame_index=i,native_gray_pixel_sha256=dict(previous="a"*64,current="a"*64)) for i in ends])
        analysis=root/"analysis.json"
        pixels.write_json(analysis,dict(schema=pixels.ANALYSIS_SCHEMA,passed_integrity=True,input_sha256=bindings,
                          plan_path=str(plan_path),plan_sha256=bindings[str(plan_path)],clips=clips))
        return repo,own,analysis

    def test_whole_json_trace_and_full_receipt_bindings_before_any_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();repo,own,analysis=self.fixture(root)
            with patch.object(pixels,"REPOSITORY",repo),patch.object(pixels,"__file__",str(own)):
                result,_,rows,bindings=pixels.load_verified_analysis(analysis,pixels.sha(analysis))
                self.assertTrue(result["passed_integrity"])
                self.assertEqual({c:list(v) for c,v in rows.items()},pixels.PAIR_ENDS)
                self.assertIn(str(repo/"tests/unit/test_feature_residual_pixels.py"),bindings)
                self.assertIn(str(analysis),bindings)
                # Even with the analyzer hash updated, a failed trace receipt
                # must independently stop this tool before source decoding.
                receipt_path=root/"0170/execution_receipt.json"
                receipt=pixels.read_json(receipt_path);receipt["passed"]=False
                receipt_path.write_text(json.dumps(receipt))
                result["input_sha256"][str(receipt_path)]=pixels.sha(receipt_path)
                analysis.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,"execution has not passed"):
                    pixels.load_verified_analysis(analysis,pixels.sha(analysis))

    def test_failed_analysis_and_hash_tamper_stop_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"analysis.json"
            pixels.write_json(path,dict(schema=pixels.ANALYSIS_SCHEMA,passed_integrity=False))
            with self.assertRaisesRegex(ValueError,"has not passed"):
                pixels.load_verified_analysis(path,pixels.sha(path))
            with self.assertRaisesRegex(ValueError,"identity differs"):
                pixels.load_verified_analysis(path,"0"*64)

    def test_generated_end_to_end_receipt_and_failed_recheck(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            source_root=root/"approved_generated_stubs";source_root.mkdir()
            hashes={}
            for clip in pixels.PAIR_ENDS:
                path=source_root/("chunk_"+clip+".avi")
                path.write_bytes(b"generated stub, never passed to a video decoder: "+clip.encode())
                hashes[clip]=pixels.sha(path)
            metadata=root/"generated_metadata.json";metadata.write_text("{}")
            bindings={str(metadata):pixels.sha(metadata)}
            rows={clip:{i:trace_row(i) for i in ends} for clip,ends in pixels.PAIR_ENDS.items()}
            image=gray()
            fake_analyzer=SimpleNamespace(unpack_row=lambda row,clip:arrays(),
                decode_array=lambda record,dtype,shape:np.frombuffer(base64.b64decode(record["data_base64"]),dtype=dtype).reshape(shape))
            def generated_pairs(source,selected):
                for index in selected:yield index,image,image
            def loaded(*unused):return {},fake_analyzer,rows,dict(bindings)
            with patch.object(pixels,"SOURCE_DIRECTORY",source_root),patch.object(pixels,"SOURCE_HASHES",hashes),\
                    patch.object(pixels,"load_verified_analysis",side_effect=loaded),\
                    patch.object(pixels,"decode_verified_pairs",side_effect=generated_pairs):
                output=root/"passed"
                with patch("builtins.print"):
                    receipt=pixels.run(metadata,"a"*64,output)
                self.assertTrue(receipt["passed"])
                self.assertEqual(receipt["pair_count"],6)
                self.assertEqual(len(receipt["figures"]),12)
                self.assertEqual(receipt["model_fits"],0)
                stats=pixels.read_json(output/"pixel_statistics.json")
                self.assertEqual(sum(len(c["rows"]) for c in stats["clips"].values()),6)
                self.assertEqual(pixels.sha(output/"pixel_statistics.json"),receipt["statistics_sha256"])
                self.assertEqual(len(list(output.glob("*.png"))),12)
                with self.assertRaisesRegex(ValueError,"Fresh output"):
                    pixels.run(metadata,"a"*64,output)
                failed=root/"failed"
                with patch.object(pixels,"check_hashes",side_effect=ValueError("changed during rendering")),patch("builtins.print"):
                    with self.assertRaisesRegex(ValueError,"changed during rendering"):
                        pixels.run(metadata,"a"*64,failed)
                self.assertFalse((failed/"execution_receipt.json").exists())
                self.assertFalse(pixels.read_json(failed/"failed_receipt.json")["passed"])


if __name__ == "__main__":
    unittest.main()
