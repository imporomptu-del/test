import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_phase20_kernel_evidence import verify_build,check_prefix
from tiny_target.visible_baseline import sha256


class KernelEvidenceTests(unittest.TestCase):
    def fixture(self,root):
        (root/'scripts').mkdir();(root/'test_sources').mkdir()
        (root/'scripts/source.cu').write_text('frozen source')
        (root/'test_sources/generated.cu').write_text('generated source')
        (root/'test.so').write_bytes(b'library fixture, never loaded')
        record=dict(library_sha256=sha256(root/'test.so'),
            sources_sha256={'source.cu':sha256(root/'scripts/source.cu')},
            generated_sha256={'generated.cu':sha256(root/'test_sources/generated.cu')},
            command=['/usr/local/cuda/bin/nvcc','-O3','--fmad=false','-arch=sm_87','-Xptxas=-v',
                '-Xcompiler','-fPIC','-shared','/test/generated.cu','-o','/test/test.so'])
        (root/'test.so.build.json').write_text(json.dumps(record))
        return record

    def test_build_verification_and_binary_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=self.fixture(root)
            self.assertEqual(record,verify_build(root,'test'))
            (root/'test.so').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed library'):verify_build(root,'test')

    def test_source_and_compiler_flags_fail_closed(self):
        for target in ('scripts/source.cu','test_sources/generated.cu','test.so.build.json'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);record=self.fixture(root)
                if target.endswith('json'):
                    record['command'][2]='--use_fast_math'
                    (root/target).write_text(json.dumps(record))
                else:(root/target).write_text('changed')
                with self.assertRaises(ValueError):verify_build(root,'test')

    def test_prefix_checks_geometry_scores_and_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a';b=root/'b';a.mkdir();b.mkdir()
            rows=[dict(frame_index=i,candidates=[dict(x=2.,score=1.)],timings_ms=dict(total=i)) for i in range(2)]
            def write(path,values):
                (path/'frames.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
            write(a,rows);write(b,rows);check_prefix(a,b,2)
            rows[1]['timings_ms']['total']=100;write(b,rows);check_prefix(a,b,2)
            rows[1]['candidates'][0]['score']=1.000001;write(b,rows)
            with self.assertRaises(AssertionError):check_prefix(a,b,2)
            write(b,rows[:1])
            with self.assertRaises(ValueError):check_prefix(a,b,2)


if __name__=='__main__':unittest.main()
