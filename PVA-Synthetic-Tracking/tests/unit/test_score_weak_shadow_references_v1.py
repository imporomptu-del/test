"""Generated metadata-only fixtures; never read new experiment outputs/media."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts' if (ROOT/'scripts/score_weak_shadow_references_v1.py').is_file() else HERE.parent))
import score_weak_shadow_references_v1 as m

OLD = m.historical()
A,B = 'a'*64,'b'*64


def sample(frame=0, panel='dense', x=10., assigned='0/bright:1', gated=None, clip='0029'):
    value = dict(hit=assigned is not None,assigned_id=assigned,
                 all_gated_ids=([assigned] if assigned is not None else []) if gated is None else gated)
    return dict(panel=panel,clip_id=clip,window_id='one',frame_index=frame,
        source_xy=[x,20.],position_uncertainty_px=1.,polarity='bright',
        saved=dict(actual_measurement=deepcopy(value),qualified_measurement=deepcopy(value)))


def track(identity='bright:1', x=10., measured=True, qualified=True, applied=False):
    return dict(segment=0,track_id=identity,measured=measured,qualified_moving=qualified,
        measurement_source_xy=[x,20.] if measured else None,source_xy=[1000.,1000.],
        weak_evidence=dict(identity='0/'+identity,applied=applied,is_ordinary_measurement=False,
            physical_identity_verified=False,status='weak_kinematic_correction' if applied else 'strong_measurement_priority',
            measurement_reference_xy=[x,20.]))


def baseline(tracks=None):
    return dict(frame_index=0,timestamp_ns=0,segment=0,tracks=[track()] if tracks is None else tracks,
        source_to_reference=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
        coverage=dict(full_shape_hw=[3190,4784],configured_crop=None,native_pixel_sampling=True,
                      detection_ready=True,warmup=False,searchable_pixels=123),motion={})


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return path


class HistoricalScorerTests(unittest.TestCase):
    def test_original_assign_semantics_max_cardinality_and_ambiguity(self):
        samples=[sample(x=10.),sample(x=13.)]
        observations=[dict(id='0/bright:1',xy=[11.,20.],polarity='bright'),
                      dict(id='0/bright:2',xy=[8.,20.],polarity='bright')]
        matched,neighbors=OLD.assign(samples,observations)
        self.assertEqual(matched,{0:1,1:0})
        values=m.assigned(OLD,samples,observations)
        self.assertEqual([v['assigned_id'] for v in values],['0/bright:2','0/bright:1'])
        self.assertEqual(values[0]['all_gated_ids'],['0/bright:1','0/bright:2'])
        self.assertTrue(values[0]['multiple_gated_alternatives'])
        self.assertTrue(all(v['shared_gated_observation'] for v in values))

    def test_polarity_gate_and_boundary_are_original(self):
        s=sample()
        items=[dict(id='0/dark:1',xy=[10.,20.],polarity='dark'),
               dict(id='0/bright:1',xy=[13.,20.],polarity='bright'),
               dict(id='0/bright:2',xy=[13.00000001,20.],polarity='bright')]
        value=m.assigned(OLD,[s],items)[0]
        self.assertEqual(value['all_gated_ids'],['0/bright:1'])

    def test_weak_transform_uses_actual_weak_point_not_posterior(self):
        b=baseline();b['source_to_reference']=[[2.,0.,4.],[0.,2.,6.],[0.,0.,1.]]
        t=track(measured=False,applied=True);t['weak_evidence']['measurement_reference_xy']=[24.,46.]
        self.assertEqual(m.weak_observations(OLD,dict(segment=0,records=[t]),b),
                         [dict(id='0/bright:1',xy=[10.,20.],polarity='bright')])
        self.assertEqual(OLD.observations(dict(segment=0,tracks=[t])),
                         dict(actual_measurement=[],qualified_measurement=[]))

    def test_projective_coordinate_lift(self):
        b=baseline();b['source_to_reference']=[[1.,0.,0.],[0.,1.,0.],[.01,0.,1.]]
        t=track(measured=False,applied=True);t['weak_evidence']['measurement_reference_xy']=[10./1.1,20./1.1]
        got=m.weak_observations(OLD,dict(segment=0,records=[t]),b)[0]['xy']
        self.assertAlmostEqual(got[0],10.);self.assertAlmostEqual(got[1],20.)

    def test_weak_cannot_be_measured_or_identity_truth(self):
        for edit in ('measured','truth','duplicate'):
            t=track(measured=False,applied=True)
            if edit=='measured':t['measured']=True
            if edit=='truth':t['weak_evidence']['physical_identity_verified']=True
            rows=[t,t] if edit=='duplicate' else [t]
            with self.assertRaises(ValueError):m.weak_observations(OLD,dict(segment=0,records=rows),baseline())

    def test_empty_weak_no_invalid_transform_needed(self):
        b=baseline();b['source_to_reference']=None
        self.assertEqual(m.weak_observations(OLD,dict(segment=0,records=[track()]),b),[])

    def test_same_frame_distinct_panel_cohorts_retain_original_overlap(self):
        samples=[sample(),sample(panel='pilot')]
        t=track(measured=False,applied=True)
        result=m.weak_score_frame(OLD,dict(segment=0,records=[t]),baseline(),samples)
        self.assertEqual(len(result),2)
        self.assertTrue(all(v['hit'] for v in result.values()))

    def test_same_cohort_one_to_one_and_original_shared_ambiguity(self):
        a=sample();b=sample(x=11.);b['window_id']='one';b['panel']='dense'
        # Distinct reference keys normally encode separate window samples; test
        # the original assignment primitive directly without key deduplication.
        values=m.assigned(OLD,[a,b],[dict(id='0/bright:1',xy=[10.,20.],polarity='bright')])
        self.assertEqual(sum(v['hit'] for v in values),1)
        self.assertTrue(all(v['shared_gated_observation'] for v in values))

    def test_unavailable_cannot_count_weak(self):
        b=baseline();b['coverage']['detection_ready']=False
        with self.assertRaises(ValueError):
            m.weak_score_frame(OLD,dict(segment=0,records=[track(measured=False,applied=True)]),b,[sample()])

    def run_clip(self, before, records, samples):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            paths={'clean/frames.jsonl':dump(root/'before.jsonl',before),
                   'shadow/shadow_trace.jsonl':dump(root/'trace.jsonl',dict(frame_index=0,timestamp_ns=0,segment=0,records=records))}
            with patch.dict(OLD.COUNTS,{'0029':1}):
                return m.score_clip(OLD,paths,'0029',samples)

    def test_ordinary_actual_vs_qualified_and_weak_separate(self):
        before,after,weak,coords=self.run_clip(baseline(),[track(qualified=False)],[sample()])
        comparison=OLD.compare(before,after)
        r=comparison['records'][0]
        self.assertFalse(r['changes']['actual_measurement']['newly_lost'])
        self.assertTrue(r['changes']['qualified_measurement']['newly_lost'])
        self.assertFalse(weak[OLD.key(sample())]['hit'])
        self.assertEqual(coords['differing_frame_count'],1)

    def test_weak_annotation_does_not_hide_strong_loss(self):
        before,after,weak,_=self.run_clip(baseline(),[track(measured=False,applied=True)],[sample()])
        comparison=OLD.compare(before,after)
        self.assertTrue(comparison['records'][0]['changes']['actual_measurement']['newly_lost'])
        self.assertTrue(comparison['records'][0]['changes']['qualified_measurement']['newly_lost'])
        self.assertTrue(weak[OLD.key(sample())]['hit'])

    def test_original_miss_preserved_and_id_change_not_false_loss(self):
        s=sample(assigned=None)
        before,after,weak,_=self.run_clip(baseline(tracks=[]),[track()],[s])
        self.assertTrue(OLD.compare(before,after)['records'][0]['changes']['qualified_measurement']['newly_recovered'])
        before,after,_,coords=self.run_clip(baseline(),[track(identity='bright:9')],[sample()])
        self.assertTrue(OLD.compare(before,after)['no_new_qualified_losses'])
        self.assertTrue(coords['all_frames_equal'])
        self.assertEqual(coords['differing_frame_count'],0)

    def test_saved_gated_alternatives_must_reproduce(self):
        with self.assertRaisesRegex(ValueError,'historical assignment'):
            self.run_clip(baseline(),[track()],[sample(gated=['0/bright:1','0/bright:7'])])

    def test_explicit_excluded_denominator(self):
        samples=[sample(),sample(clip='0082'),sample(clip='0126')]
        kept=[s for s in samples if s['clip_id'] in m.CLIPS]
        excluded=[s for s in samples if s['clip_id'] not in m.CLIPS]
        self.assertEqual(m.sample_counts(kept)['samples'],2)
        self.assertEqual(m.sample_counts(excluded)['by_clip'],{'0082':1})

    def test_portable_reference_export_load_does_not_reopen_history(self):
        with tempfile.TemporaryDirectory() as temp:
            samples=[sample()]
            excluded_panels=dict(OLD.PANELS);excluded_panels['dense']-=1
            doc=dict(schema=m.REFERENCE_SCHEMA,clips=list(m.CLIPS),baseline_saved_assignments_verified=True,
                references_supplied_to_tracker=False,new_weak_outcomes_read=False,
                frames={c:OLD.COUNTS[c] for c in m.CLIPS},sources={c:OLD.SOURCES[c] for c in m.CLIPS},
                provenance_sha256={'/nonexistent/previous-machine/score_feature_selection_references.py':m.HISTORICAL_SHA},
                samples=samples,counts=m.sample_counts(samples),
                original_all_four_clip_counts=dict(samples=356,by_clip={'0029':1,'0082':355},by_panel=OLD.PANELS),
                excluded_counts=dict(samples=355,by_clip={'0082':355},by_panel=excluded_panels))
            p=dump(Path(temp).resolve()/'references.json',doc)
            with patch.object(OLD,'references',side_effect=AssertionError('Must not reopen historical files')):
                self.assertEqual(m.load_references(OLD,p,OLD.sha(p),{}),doc)
            changed=deepcopy(doc);changed['counts']['samples']=0;dump(p,changed)
            with self.assertRaises(ValueError):m.load_references(OLD,p,OLD.sha(p),{})

    def test_pinned_source_refuses_change(self):
        with patch.object(m,'HISTORICAL_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'Historical scorer changed'):m.historical()

    def test_output_nonoverwrite_and_input_rehash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();p=root/'result.json'
            m.write_fresh(OLD,p,dict(test=True))
            with self.assertRaises(ValueError):m.write_fresh(OLD,p,dict(test=False))
            digest=OLD.sha(p);dump(p,dict(test=False))
            with self.assertRaises(ValueError):m.unchanged(OLD,{str(p):digest})

    def test_unsafe_metadata_path_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            for name in ('../escape.json','/tmp/escape.json','captures/example.npz'):
                with self.assertRaises(ValueError):m.relative_metadata(OLD,root,name,A,{})

    def test_audit_bindings_and_complete_pass_required(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();base=root/'0029'
            hashes={}
            names=('clean/frames.jsonl','shadow/frames.jsonl','shadow/shadow_trace.jsonl',
                   'clean.shadow.json','shadow.shadow.json','clean.v29.json','shadow.v29.json')
            for name in names:
                p=dump(base/name,{})
                hashes[name]=OLD.sha(p)
            for arm in ('clean','shadow'):
                receipt=dict(schema='seaqr.weak-continuation-shadow.run.v1',passed=True,error=None,clip='0029',arm=arm,
                    processed_frames=687,expected_frames=687,freeze_sha256=A,plan_sha256=B,
                    source_sha256=OLD.SOURCES['0029'],production_changed=False,weak_learning_enabled=False,
                    trace_sha256=hashes['shadow/shadow_trace.jsonl'] if arm=='shadow' else None)
                p=dump(base/(arm+'.shadow.json'),receipt);hashes[arm+'.shadow.json']=OLD.sha(p)
            doc=dict(schema='seaqr.weak-continuation-shadow.audit.v1',passed=True,clip='0029',frames=687,
                source_sha256=OLD.SOURCES['0029'],freeze_sha256=A,plan_sha256=B,
                baseline_journal_non_timing_exact=True,baseline_output_state_learning_digests_exact=True,
                native_state_guards_unchanged=True,production_changed=False,weak_learning_enabled=False,files_sha256=hashes)
            p=dump(base/'independent_audit.json',doc)
            self.assertEqual(set(m.audited_inputs(OLD,root,'0029',A,B,{})),set(names))
            for change in ({'passed':False},{'frames':686},{'freeze_sha256':B},{'production_changed':True}):
                dump(p,dict(doc,**change))
                with self.assertRaises(ValueError):m.audited_inputs(OLD,root,'0029',A,B,{})
            dump(p,doc);dump(base/'shadow/shadow_trace.jsonl',{'tampered':True})
            with self.assertRaises(ValueError):m.audited_inputs(OLD,root,'0029',A,B,{})


if __name__=='__main__':
    unittest.main()
