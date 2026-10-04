"""Generated full674frame metadata only; no real journals, pixels or media."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve()
SCRIPTS=HERE.parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS if (SCRIPTS/'extract_weak_shadow_divergence_v1.py').is_file() else HERE.parent))
import extract_weak_shadow_divergence_v1 as m

A='a'*64


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return path


def fixture(root):
    directory=root/'original';directory.mkdir()
    launch=dict(source='/home/serg/project/camera_reader_sky/srcsky/chunks/chunk_0126.avi',
        source_sha256=m.SOURCE_SHA,config_sha256=m.CONFIG_SHA,motion_config_sha256=m.MOTION_SHA,
        expected_frames=674,max_frames=None,fps=10,annotations_supplied_to_detector=False,
        configuration=dict(input_bit_depth=8,example='full frozen settings retained'),package_sha256={'one.py':A},code_sha256={'two.py':A})
    reference=dump(directory/'reference_0126.json',launch)
    plan=dump(directory/'plan.json',dict(schema='seaqr.weak-continuation-shadow.plan.v1',full_causal_replay=True,
        clips={'0126':dict(frames=674,source_sha256=m.SOURCE_SHA)}))
    freeze=dump(directory/'freeze.json',dict(schema='seaqr.weak-continuation-shadow.freeze.v1',pre_run=True,
        files_sha256={'reference_0126.json':m.sha(reference)},plan_sha256=m.sha(plan)))
    hashes={};raw={}
    for arm in ('clean','shadow'):
        path=dump(directory/'0126'/arm/'launch.json',launch);hashes[arm+'/launch.json']=m.sha(path)
        name='clean/frames.jsonl' if arm=='clean' else 'shadow/shadow_trace.jsonl'
        lines=[]
        for frame in range(674):
            row=dict(frame_index=frame,timestamp_ns=frame*100000000,segment=0,
                     opaque_unfiltered_payload={'frame':frame,'values':[1,2,3]})
            row.update(dict(tracks=[{'record':frame}],candidates=[{'proposal':frame}]) if arm=='clean'
                       else dict(records=[{'record':frame}],strong_proposals=[{'proposal':frame}],prior_forecasts=[{'forecast':frame}]))
            lines.append(('  '+json.dumps(row,separators=(', ',': '))+'\r\n').encode())
        path=directory/'0126'/name;path.write_bytes(b''.join(lines));hashes[name]=m.sha(path);raw[arm]=lines
    audit=dump(directory/'0126'/'independent_audit.json',dict(schema='seaqr.weak-continuation-shadow.audit.v1',
        passed=True,clip='0126',frames=674,freeze_sha256=m.sha(freeze),plan_sha256=m.sha(plan),source_sha256=m.SOURCE_SHA,
        baseline_journal_non_timing_exact=True,baseline_output_state_learning_digests_exact=True,
        native_state_guards_unchanged=True,production_changed=False,weak_learning_enabled=False,files_sha256=hashes))
    return directory,m.sha(audit),m.sha(freeze),raw


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.directory,self.audit_sha,self.freeze_sha,self.raw=fixture(self.root)
        self.fixed=patch.object(m,'FIXED_DIRECTORY',self.directory);self.fixed.start()

    def tearDown(self):
        self.fixed.stop();self.temp.cleanup()

    def run_extract(self,name='output'):
        return m.extract(self.directory,self.audit_sha,self.freeze_sha,self.root/name)

    def update_audit(self,changes):
        path=self.directory/'0126/independent_audit.json';doc=json.loads(path.read_text());doc.update(changes)
        dump(path,doc);self.audit_sha=m.sha(path)

    def test_exact41_raw_lines_and_full_parent_hashes(self):
        result=self.run_extract();out=self.root/'output'
        self.assertTrue(result['passed']);self.assertEqual(result['frames'],41)
        for arm,name in (('clean','clean/frames.jsonl'),('shadow','shadow/shadow_trace.jsonl')):
            gz=out/(arm+'_prefix.jsonl.gz')
            self.assertEqual(gzip.decompress(gz.read_bytes()),b''.join(self.raw[arm][:41]))
            entry=result['artifacts'][gz.name]
            self.assertEqual(entry['full_parent_sha256'],m.sha(self.directory/'0126'/name))
            self.assertEqual(entry['raw_prefix_sha256'],m.bytes_sha(b''.join(self.raw[arm][:41])))
        self.assertEqual(json.loads((out/'reference_0126.json').read_text())['configuration']['example'],'full frozen settings retained')
        self.assertFalse(result['source_media_accessed']);self.assertFalse(result['detector_or_tracker_rerun'])

    def test_gzip_deterministic(self):
        self.run_extract('first');self.run_extract('second')
        for arm in ('clean','shadow'):
            self.assertEqual((self.root/'first'/(arm+'_prefix.jsonl.gz')).read_bytes(),
                             (self.root/'second'/(arm+'_prefix.jsonl.gz')).read_bytes())

    def test_only41_line_contents_parsed(self):
        original=m.decode;frames=[]
        def observed(raw):
            doc=original(raw)
            if type(doc) is dict and 'frame_index' in doc:frames.append(doc['frame_index'])
            return doc
        with patch.object(m,'decode',side_effect=observed):self.run_extract()
        self.assertEqual(frames,list(range(41))*2)

    def test_fixed_scope_and_fresh_output(self):
        with self.assertRaises(ValueError):m.extract(self.root,self.audit_sha,self.freeze_sha,self.root/'output')
        with self.assertRaises(ValueError):m.extract(self.directory,self.audit_sha,self.freeze_sha,self.directory/'newoutput')
        self.run_extract()
        with self.assertRaises(ValueError):self.run_extract()

    def test_wrong_caller_hashes_fail_before_output(self):
        for audit,freeze in ((A,self.freeze_sha),(self.audit_sha,A)):
            with self.assertRaises(ValueError):m.extract(self.directory,audit,freeze,self.root/'output')
            self.assertFalse((self.root/'output').exists())

    def test_partial_or_failed_or_changed_guards_rejected(self):
        path=self.directory/'0126/independent_audit.json';original=path.read_bytes()
        for changes in ({'frames':41},{'passed':False},{'clip':'0029'},{'production_changed':True},
                        {'weak_learning_enabled':True},{'baseline_output_state_learning_digests_exact':False},
                        {'source_sha256':A},{'plan_sha256':A},{'freeze_sha256':A}):
            path.write_bytes(original);self.update_audit(changes)
            with self.assertRaises(ValueError):self.run_extract()

    def test_tail_change_outside_prefix_is_detected(self):
        path=self.directory/'0126/clean/frames.jsonl'
        raw=path.read_bytes();path.write_bytes(raw.replace(b'"frame_index": 673',b'"frame_index": 672'))
        with self.assertRaisesRegex(ValueError,'Changed pinned parent'):self.run_extract()

    def test_bound_config_change_rejected(self):
        path=self.directory/'0126/clean/launch.json';doc=json.loads(path.read_text());doc['config_sha256']=A;dump(path,doc)
        a=json.loads((self.directory/'0126/independent_audit.json').read_text());a['files_sha256']['clean/launch.json']=m.sha(path)
        self.update_audit(a)
        with self.assertRaisesRegex(ValueError,'frozen configuration'):self.run_extract()

    def test_reference_file_change_rejected(self):
        path=self.directory/'reference_0126.json';dump(path,{'changed':True})
        with self.assertRaises(ValueError):self.run_extract()

    def test_missing_prefix_frame_rejected_even_if_rebound(self):
        path=self.directory/'0126/clean/frames.jsonl';path.write_bytes(b''.join(self.raw['clean'][:20]+self.raw['clean'][21:]))
        a=json.loads((self.directory/'0126/independent_audit.json').read_text());a['files_sha256']['clean/frames.jsonl']=m.sha(path)
        self.update_audit(a)
        with self.assertRaisesRegex(ValueError,'causal prefix frame'):self.run_extract()

    def test_prefix_records_never_filtered(self):
        result=self.run_extract()
        row=json.loads(gzip.decompress((self.root/'output/shadow_prefix.jsonl.gz').read_bytes()).splitlines()[34])
        self.assertEqual(row['records'],[{'record':34}]);self.assertEqual(row['strong_proposals'],[{'proposal':34}])
        self.assertEqual(row['prior_forecasts'],[{'forecast':34}]);self.assertFalse(result['rows_filtered'])

    def test_post_read_rehash_detects_mutation(self):
        original=m.prefix
        def changed(path,kind):
            result=original(path,kind)
            if kind=='shadow':
                target=self.directory/'0126/clean/frames.jsonl';target.write_bytes(target.read_bytes()+b'\n')
            return result
        with patch.object(m,'prefix',side_effect=changed):
            with self.assertRaises(ValueError):self.run_extract()
        self.assertFalse((self.root/'output').exists())

    def test_after_write_change_retains_failed_receipt(self):
        original=m.BoundInputs.unchanged;count=0
        def changed(inputs):
            nonlocal count
            count+=1
            if count==2:
                target=self.directory/'0126/clean/frames.jsonl';target.write_bytes(target.read_bytes()+b'\n')
            return original(inputs)
        with patch.object(m.BoundInputs,'unchanged',changed):
            with self.assertRaises(ValueError):self.run_extract()
        receipt=json.loads((self.root/'output/receipt.json').read_text())
        self.assertFalse(receipt['passed']);self.assertIn('Changed pinned parent',receipt['error'])

    def test_symlink_parent_rejected(self):
        path=self.directory/'0126/clean/frames.jsonl';saved=self.root/'saved.jsonl';path.rename(saved);path.symlink_to(saved)
        with self.assertRaises(ValueError):self.run_extract()

    def test_strict_json(self):
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.assertRaises(ValueError):m.decode(raw)


if __name__=='__main__':unittest.main()
