"""Generated arrays and journals only; no camera media or prior results."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
import accuracy_v56_capture as capture
import summarize_accuracy_v56_capture as summary


def fixture(frame=213, center=(16.,16.), matrix=None):
    matrix = np.eye(3) if matrix is None else np.asarray(matrix, dtype=float)
    ref = summary.mapped(matrix, list(center))
    rectangle = capture.tile_rectangle((64,64), 32, ref, 7)
    x0,y0,x1,y1 = rectangle['capture_bounds_exclusive_xyxy']
    h,w = y1-y0,x1-x0
    values = np.zeros((h,w,len(capture.FLOAT_FIELDS)), np.float32)
    values[...,summary.VALUE['noise']] = 1
    values[...,summary.VALUE['temporal_threshold_dn']] = 4
    values[...,summary.VALUE['spatial_threshold_dn']] = 3
    flags = np.zeros((h,w,len(capture.FLAG_FIELDS)),np.uint8)
    flags[...,[0,1,2,3,4,9,12]] = 1
    peaks = np.zeros((8,12),capture.PEAK_DTYPE)
    peaks['x']=-1;peaks['y']=-1
    snap = dict(values=values,flags=flags,precise_sigmas=np.ones((h,w),np.float64),
                native_counts=np.zeros(8,np.int32),native_peaks=peaks)
    probe = dict(frame_index=frame,source_xy=list(center),original_reference_match_gate_px=7.,
                 original_strict_assigned_identity=None)
    meta = dict(frame=frame,source_probe=probe,source_to_reference=matrix.tolist(),rectangle=rectangle,
        prelearning=True,full_exposed_state_unchanged=True,native_state_before={'B':'a'},native_state_after={'B':'a'},
        float_fields=list(capture.FLOAT_FIELDS),flag_fields=list(capture.FLAG_FIELDS),
        decoded_pre_frame_quota_cells=[],pre_shape_post_frame_quota=[],post_shape=[],tracks=[])
    row = dict(frame_index=frame,source_to_reference=matrix.tolist(),candidates=[],tracks=[])
    finish(snap,meta,row)
    return snap,meta,row


def set_pixel(snap,meta,x,y,*,temporal=6.,spatial=5.,peak=True,candidate=True,eligible=True):
    x0,y0,_,_=meta['rectangle']['capture_bounds_exclusive_xyxy']
    xx,yy=x-x0,y-y0
    for name,value in {'temporal':temporal,'positive_signed_temporal':temporal,
        'centered_temporal':temporal,'negative_signed_temporal':-temporal,
        'spatial':spatial,'positive_signed_spatial':spatial,'negative_signed_spatial':-spatial,
        'positive_score':temporal,'negative_score':-temporal,'background':20.,'variance':1.}.items():
        snap['values'][yy,xx,summary.VALUE[name]]=value
    for name,value in {'native_eligible':eligible,'eligible':eligible,
        'positive_temporal_pass':temporal>=4,'positive_spatial_pass':spatial>=3,
        'negative_temporal_pass':-temporal>=4,'negative_spatial_pass':-spatial>=3,
        'raw_absolute_peak':peak,'positive_candidate':candidate}.items():
        snap['flags'][yy,xx,summary.FLAG[name]]=value


def finish(snap,meta,row):
    ranks=capture.rank_candidates(dict(snap,metadata=meta),max_candidates_per_tile_polarity=12,verify_native=False)
    snap['native_counts'].fill(0)
    snap['native_peaks'].fill(0)
    snap['native_peaks']['x']=-1;snap['native_peaks']['y']=-1
    x0,y0,_,_=meta['rectangle']['capture_bounds_exclusive_xyxy']
    for item in ranks['tile_polarity_summary']:
        cell=2*item['tile_id']+item['polarity']
        snap['native_counts'][cell]=item['prequota_count']
    for polarity,label in enumerate(('positive','negative')):
        for yy,xx in np.argwhere((ranks[label+'_prequota_rank']>0)&(ranks[label+'_prequota_rank']<=12)):
            x,y=int(xx+x0),int(yy+y0)
            cell=2*((y//32)*2+x//32)+polarity
            rank=int(ranks[label+'_prequota_rank'][yy,xx])-1
            snap['native_peaks'][cell,rank]=(x,y,snap['values'][yy,xx,13+polarity],
                                            snap['values'][yy,xx,5],snap['values'][yy,xx,9])
    snap.update({key:value for key,value in ranks.items() if key.endswith('_rank')})
    cells=[]
    for index,cell in enumerate(snap['native_peaks']):
        present=cell[cell['x']>=0].tolist()
        if present:
            cells.append([dict(x=x,y=y,polarity='bright' if index%2==0 else 'dark',score=s,
                               response_dn=r,noise_sigma_dn=n) for x,y,s,r,n in present])
    meta['decoded_pre_frame_quota_cells']=cells
    pre=[cell[r] for r in range(12) for cell in cells if r<len(cell)][:512]
    meta['pre_shape_post_frame_quota']=deepcopy(pre)
    meta['post_shape']=deepcopy(pre)
    inverse=np.linalg.inv(meta['source_to_reference'])
    row['candidates']=[dict(p,source_xy=summary.mapped(inverse,[p['x'],p['y']])) for p in pre]


class GateTests(unittest.TestCase):
    def test_circular_not_square_gate(self):
        s,m,r=fixture()
        mask,_,_=summary.source_gate(m)
        self.assertEqual(int(mask.sum()),149)
        self.assertFalse(mask[23,23])
        self.assertTrue(mask[16,23])

    def test_source_gate_uses_transform(self):
        s,m,r=fixture(matrix=[[1,0,4],[0,1,3],[0,0,1]])
        mask,_,_=summary.source_gate(m)
        self.assertTrue(mask[19,20])
        self.assertFalse(mask[16,8])
        self.assertEqual(int(mask.sum()),149)

    def test_affine_scale_gate_in_source_coordinates(self):
        s,m,r=fixture(center=(8.,8.),matrix=[[2,0,0],[0,2,0],[0,0,1]])
        mask,_,_=summary.source_gate(m)
        self.assertTrue(mask[16,30])
        self.assertFalse(mask[16,31])

    def test_projective_transform_refused(self):
        s,m,r=fixture()
        m['source_to_reference'][2][0]=.001
        with self.assertRaises(ValueError):summary.source_gate(m)

    def test_gate_entering_unranked_halo_refused(self):
        s,m,r=fixture(center=(8.,8.),matrix=[[2.3,0,-2.4],[0,2.3,-2.4],[0,0,1]])
        with self.assertRaises(ValueError):summary.source_gate(m)

    def test_singular_transform_refused(self):
        s,m,r=fixture();m['source_to_reference'][1]=[0,0,0]
        with self.assertRaises(np.linalg.LinAlgError):summary.source_gate(m)

    def test_recenter_refused(self):
        s,m,r=fixture();m['source_probe']['source_xy']=[17.,16.]
        with self.assertRaises(ValueError):summary.source_gate(m)

    def test_gate_change_refused(self):
        s,m,r=fixture();m['source_probe']['original_reference_match_gate_px']=8.
        with self.assertRaises(ValueError):summary.source_gate(m)


class ProbeTests(unittest.TestCase):
    def test_empty_gate_and_unknown_assignment(self):
        result=summary.summarize_probe(*fixture())
        self.assertEqual(result['gate_counts']['integer_pixels_in_original_source_gate'],149)
        self.assertEqual(result['gate_counts']['post_shape_candidates_in_original_gate'],0)
        self.assertIsNone(result['nearest_actual_bright_candidate'])
        self.assertIsNone(result['original_assignment_preserved'])
        self.assertFalse(result['selected_diagnostic_pixel_is_object_truth'])

    def test_success_through_candidate_stage(self):
        s,m,r=fixture();set_pixel(s,m,16,16);finish(s,m,r)
        result=summary.summarize_probe(s,m,r)
        for key in ('eligible_both_thresholds_raw_peak_pass','tile_quota_surviving_seeds',
                    'frame_quota_surviving_seeds','post_shape_candidates_in_original_gate'):
            self.assertEqual(result['gate_counts'][key],1)
        self.assertEqual(result['nearest_actual_bright_candidate']['identity'],'candidate:0')
        self.assertEqual(result['strongest_bright_spatial_pixel']['values']['background'],20.)

    def test_stronger_opposite_polarity_neighbor_suppresses_raw_peak(self):
        s,m,r=fixture()
        set_pixel(s,m,16,16,peak=False,candidate=False)
        set_pixel(s,m,18,16,temporal=-9.,spatial=-5.,candidate=False)
        finish(s,m,r)
        result=summary.summarize_probe(s,m,r)
        p=result['strongest_bright_spatial_pixel']
        self.assertEqual(p['reference_xy'],[16,16])
        self.assertEqual(p['stronger_raw_absolute_neighbors'][0]['reference_xy'],[18,16])
        self.assertEqual(p['stronger_raw_absolute_neighbors'][0]['raw_temporal_dn'],-9.)
        self.assertEqual(result['gate_counts']['eligible_both_thresholds_pass'],1)
        self.assertEqual(result['gate_counts']['eligible_both_thresholds_raw_peak_pass'],0)

    def test_equal_absolute_neighbor_is_not_suppressor(self):
        s,m,r=fixture();set_pixel(s,m,16,16);set_pixel(s,m,18,16,temporal=-6.,spatial=-5.,candidate=False)
        finish(s,m,r)
        p=summary.summarize_probe(s,m,r)['strongest_bright_spatial_pixel']
        self.assertEqual(p['stronger_raw_absolute_neighbors'],[])

    def test_extrema_not_conditioned_on_success(self):
        s,m,r=fixture();set_pixel(s,m,16,16,temporal=1.,spatial=9.,candidate=False)
        set_pixel(s,m,17,16,temporal=10.,spatial=1.,candidate=False);finish(s,m,r)
        result=summary.summarize_probe(s,m,r)
        self.assertEqual(result['strongest_bright_spatial_pixel']['reference_xy'],[16,16])
        self.assertEqual(result['strongest_positive_temporal_pixel']['reference_xy'],[17,16])
        self.assertEqual(result['gate_counts']['eligible_both_thresholds_pass'],0)

    def test_extrema_ignore_outside_gate(self):
        s,m,r=fixture();set_pixel(s,m,16,16);set_pixel(s,m,23,23,temporal=100.,spatial=100.)
        finish(s,m,r)
        self.assertEqual(summary.summarize_probe(s,m,r)['strongest_bright_spatial_pixel']['reference_xy'],[16,16])

    def test_nonfinite_selection_unknown_not_zero(self):
        s,m,r=fixture();s['values'][...,summary.VALUE['positive_signed_spatial']]=np.nan
        result=summary.summarize_probe(s,m,r)
        self.assertIsNone(result['strongest_bright_spatial_pixel'])
        self.assertEqual(result['gate_counts']['nonfinite_value_pixels'],149)

    def test_stage_flags_tampering_refused(self):
        s,m,r=fixture();set_pixel(s,m,16,16,candidate=False);finish(s,m,r)
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)

    def test_saved_rank_tampering_refused(self):
        s,m,r=fixture();s['positive_prequota_rank'][16,16]=1
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)

    def test_decoded_native_cells_tampering_refused(self):
        s,m,r=fixture();set_pixel(s,m,16,16);finish(s,m,r)
        m['decoded_pre_frame_quota_cells'][0][0]['score']=999
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)

    def test_frame_quota_order_tampering_refused(self):
        s,m,r=fixture();set_pixel(s,m,16,16);finish(s,m,r);m['pre_shape_post_frame_quota']=[]
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)

    def test_shape_centroid_lineage_distinct_from_seed(self):
        s,m,r=fixture();set_pixel(s,m,16,16);finish(s,m,r)
        p=m['post_shape'][0];p['x']=17
        p['shape']={'member_peak_reference_xy':[[16,16]],'centroid_reference_xy':[16.7,16.]}
        r['candidates']=[dict(p,source_xy=[17.,16.])]
        result=summary.summarize_probe(s,m,r)
        nearest=result['nearest_actual_bright_candidate']
        self.assertEqual(nearest['shape_seed_lineage'][0]['diagnostic']['reference_xy'],[16,16])
        self.assertEqual(nearest['source_xy'],[17.,16.])

    def test_capture_state_mutation_refused(self):
        s,m,r=fixture();m['native_state_after']={'B':'b'}
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)

    def test_journal_tracks_disagreement_refused(self):
        s,m,r=fixture();r['tracks']=[{'different':True}]
        with self.assertRaises(ValueError):summary.summarize_probe(s,m,r)


def track(identifier='bright:1',*,measured=True,qualified=True,measurement=(5.,5.),source=(99.,99.)):
    return dict(segment=0,track_id=identifier,measured=measured,qualified_moving=qualified,
                measurement_source_xy=list(measurement) if measured else None,source_xy=list(source))


def generated_run(root):
    """Synthetic producer-independent summary fixture, not a real audit claim."""
    root=Path(root).resolve()
    (root/'captures').mkdir();(root/'probe').mkdir()
    selected={};specs=[];required=['freeze.json','probes.json','probe/frames.jsonl']
    for frame in summary.FRAMES:
        snap,meta,row=fixture(frame)
        set_pixel(snap,meta,16,16);finish(snap,meta,row)
        specs.append(meta['source_probe']);selected[frame]=row
        stem=f'captures/frame_{frame:06d}'
        np.savez_compressed(root/(stem+'.npz'),**snap)
        (root/(stem+'.json')).write_text(json.dumps(meta))
        required.extend([stem+'.npz',stem+'.json'])
    controls=[dict(frames_inclusive=[0,673],crop_xywh=[i*20,0,10,10],
                   baseline_measured=0,baseline_predicted=0) for i in range(7)]
    (root/'probes.json').write_text(json.dumps(dict(reference_probes=specs,provisional_controls=controls)))
    (root/'freeze.json').write_text(json.dumps(dict(pre_run=True,files_sha256={'probes.json':summary.sha(root/'probes.json')})))
    with (root/'probe/frames.jsonl').open('w') as stream:
        for index in range(674):
            stream.write(json.dumps(selected.get(index,dict(frame_index=index,tracks=[])))+'\n')
    audit=dict(schema='seaqr.accuracy-v56-independent-audit.v1',passed=True,frames=674,
        capture_frames=list(summary.FRAMES),exact_full_journal_semantics=True,
        exact_private_state_and_learning=True,files_sha256={name:summary.sha(root/name) for name in required})
    (root/'independent_audit.json').write_text(json.dumps(audit))
    return root,audit


class ControlTests(unittest.TestCase):
    def test_measurement_coordinate_and_predicted_state_are_distinct(self):
        controls=[dict(frames_inclusive=[1,2],crop_xywh=[0,0,10,10],baseline_measured=1,baseline_predicted=1)]
        rows=[dict(frame_index=1,tracks=[track(),track('bright:2',measured=False,source=(5.,5.)),
                                       track('bright:3',qualified=False)])]
        result=summary.control_counts(rows,controls)
        self.assertEqual(result['qualified_measured_states'],1)
        self.assertEqual(result['qualified_predicted_states'],1)
        self.assertTrue(result['all_archived_scope_counts_match'])

    def test_half_open_crop_inclusive_frame_boundaries(self):
        controls=[dict(frames_inclusive=[1,2],crop_xywh=[0,0,10,10],baseline_measured=2,baseline_predicted=0)]
        rows=[dict(frame_index=i,tracks=[track(measurement=(0.,0.)),track('bright:2',measurement=(10.,5.))])
              for i in range(4)]
        self.assertEqual(summary.control_counts(rows,controls)['qualified_measured_states'],2)

    def test_missing_actual_measurement_fails(self):
        t=track();t['measurement_source_xy']=None
        with self.assertRaises(ValueError):summary.control_counts([dict(frame_index=1,tracks=[t])],[])

    def test_duplicate_identity_fails(self):
        with self.assertRaises(ValueError):summary.control_counts([dict(frame_index=1,tracks=[track(),track()])],[])

    def test_count_mismatch_reported_not_tuned(self):
        c=dict(frames_inclusive=[1,2],crop_xywh=[0,0,10,10],baseline_measured=99,baseline_predicted=0)
        result=summary.control_counts([dict(frame_index=1,tracks=[])],[c])
        self.assertFalse(result['all_archived_scope_counts_match'])
        self.assertIsNone(result['false_positive_rate'])


class EvidenceTests(unittest.TestCase):
    def test_complete_generated_summary_serializes(self):
        with tempfile.TemporaryDirectory() as d:
            root,_=generated_run(d);result=summary.summarize_run(root)
            self.assertTrue(result['completed'])
            self.assertEqual(len(result['probes']),6)
            self.assertTrue(result['provisional_controls']['all_archived_scope_counts_match'])
            self.assertFalse(result['provisional_controls']['matches_archived_total_70_measured_64_predicted'])
            self.assertFalse(result['source_media_accessed'])
            json.dumps(result,allow_nan=False)

    def test_omitted_audit_binding_fails_before_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            root,audit=generated_run(d);del audit['files_sha256']['captures/frame_000213.npz']
            (root/'independent_audit.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):summary.summarize_run(root)

    def test_tampered_capture_fails_hash_check(self):
        with tempfile.TemporaryDirectory() as d:
            root,_=generated_run(d)
            with (root/'captures/frame_000213.npz').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):summary.summarize_run(root)

    def test_incomplete_journal_fails_even_if_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root,audit=generated_run(d);path=root/'probe/frames.jsonl'
            lines=path.read_text().splitlines();path.write_text('\n'.join(lines[:-1])+'\n')
            audit['files_sha256']['probe/frames.jsonl']=summary.sha(path)
            (root/'independent_audit.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):summary.summarize_run(root)

    def test_noncontiguous_journal_fails_even_if_hashed(self):
        with tempfile.TemporaryDirectory() as d:
            root,audit=generated_run(d);path=root/'probe/frames.jsonl'
            lines=path.read_text().splitlines();lines[10]=lines[11];path.write_text('\n'.join(lines)+'\n')
            audit['files_sha256']['probe/frames.jsonl']=summary.sha(path)
            (root/'independent_audit.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):summary.summarize_run(root)

    def test_manifest_not_bound_by_freeze_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root,audit=generated_run(d);path=root/'freeze.json'
            frozen=json.loads(path.read_text());frozen['files_sha256']['probes.json']='0'*64
            path.write_text(json.dumps(frozen));audit['files_sha256']['freeze.json']=summary.sha(path)
            (root/'independent_audit.json').write_text(json.dumps(audit))
            with self.assertRaises(ValueError):summary.summarize_run(root)

    def test_missing_audit_fails(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):summary.summarize_run(d)

    def test_failed_audit_fails(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'independent_audit.json').write_text('{"passed": false}')
            with self.assertRaises(ValueError):summary.summarize_run(d)

    def test_duplicate_and_nonfinite_json_refused(self):
        for text in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):summary.loads(text)

    def test_inputs_reject_path_traversal_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'actual').write_text('x');(root/'link').symlink_to(root/'actual')
            data=summary.Inputs(root)
            for name in ('../outside','link'):
                with self.assertRaises(ValueError):data.path(name)

    def test_bound_input_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'data').write_text('first');data=summary.Inputs(root)
            data.path('data');(root/'data').write_text('second')
            with self.assertRaises(ValueError):data.unchanged()


if __name__=='__main__':unittest.main()
