from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from visible_threads_v31 import schedule,traces,full,gate
from verify_visible_threads_v31 import specs,independent_gate
from combined_v29_protocol import samples
import test_visible_threads_v31 as fixtures
import verify_visible_interaction_v30 as relocated


def parsed(rows):
    lookup={s['name']:s for s in schedule()}
    return ({n:dict(r,count=r['processed_frames'],mode=lookup[n]['mode'],traced=False) for n,r in rows.items()},
            {n:samples(r) for n,r in rows.items()})


class VerifyThreadsTest(unittest.TestCase):
    def test_independent_schedule(self):
        self.assertEqual(specs(),(schedule()+traces(),full()))

    def test_independent_gate(self):
        for case in ('nominal','extra_control_faster','single_faster','slow_pair'):
            rows=fixtures.ThreadsTest().timing_receipts()
            if case=='extra_control_faster':
                for i in range(3):rows[f'0082_repeat{i}_v26_default'].update(fps=2.,wall_s=64.)
            elif case=='single_faster':
                for i in range(3):rows[f'0126_repeat{i}_v26'].update(fps=2.,wall_s=64.)
            elif case=='slow_pair':rows['0126_repeat0_combined'].update(fps=.8,wall_s=160.)
            self.assertEqual(independent_gate(*parsed(rows)),gate(rows))

    def test_no_partial_or_instrumented_controls(self):
        trials,values=parsed(fixtures.ThreadsTest().timing_receipts())
        trials.pop('0082_repeat0_v26_default')
        with self.assertRaises(AssertionError):independent_gate(trials,values)
        for key,value in [('traced',True),('wall_s',float('nan')),('fps',0.)]:
            trials,values=parsed(fixtures.ThreadsTest().timing_receipts())
            trials['0082_repeat0_v26_default'][key]=value
            with self.assertRaises(AssertionError):independent_gate(trials,values)

    def test_relocation_only_changes_dependency_path(self):
        root=Path('/new/evidence')
        def audit(*unused):
            return [relocated.previous_audit.sha(root/n) for n in ('freeze.json','frames.jsonl','trial.v29.json')]
        with patch.object(relocated.previous_audit,'sha',side_effect=lambda p:str(p)), patch.object(relocated,'audit_trial',side_effect=audit):
            result=relocated.audit_relocated_trial(root,None,None,None,None,None,None)
        self.assertEqual(result,[str(relocated.V29_LOCAL/'freeze.json'),str(root/'frames.jsonl'),str(root/'trial.v29.json')])


if __name__=='__main__':unittest.main()
