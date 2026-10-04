"""Independent convergence audit rejects inconsistent or incomplete evidence."""
import copy
import sys
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_visible_clocks_v33 as a
import visible_clocks_v33 as v
from export_visible_clocks_v33 import files_for


def fixture():
    saved={n:dict(minimum=p['expected_min'],maximum=p['expected_max'],governor=p['expected_governor'])
        for n,p in v.POLICIES.items()}
    target={n:dict(p,minimum=p['maximum']) for n,p in saved.items()}
    receipt=dict(writes=[dict(policy=n,path=p['path']+'/'+p['minimum'],value=target[n]['minimum'],written=True)
        for n,p in v.POLICIES.items()],verification=dict(verified=True,error=None,target=target,
        timeout_s=3,poll_interval_s=.05,required_stable_samples=3,elapsed_s=.11,
        observations=[dict(actual=copy.deepcopy(target),read_error=None,elapsed_s=t,consecutive_matches=i+1,
            safety=dict(temperatures={'cpu-thermal':50000,'tj-thermal':51000})) for i,t in enumerate((.01,.06,.11))]))
    return saved,receipt


class AuditTests(unittest.TestCase):
    def test_independent_schedule(self):
        self.assertEqual(a.specs(),v.schedule())

    def test_valid_stability(self):
        saved,r=fixture()
        self.assertEqual(a.verify_stability(r,saved,True,True)['observations'],3)

    def test_corruption_rejected(self):
        mutators=[
            lambda r:r['writes'].pop(),
            lambda r:r['writes'][0].update(path='/tmp/outside'),
            lambda r:r['verification'].update(verified=False),
            lambda r:r['verification'].update(elapsed_s=4),
            lambda r:r['verification']['observations'][1].update(elapsed_s=.01),
            lambda r:r['verification']['observations'][-1].update(consecutive_matches=2),
            lambda r:r['verification']['observations'][-1]['actual']['cpu0'].update(minimum=729600),
            lambda r:r['verification']['observations'][0]['actual']['cpu0'].update(maximum=999),
            lambda r:r['verification']['observations'][0]['safety']['temperatures'].update(**{'cpu-thermal':75000}),
        ]
        for fn in mutators:
            saved,r=fixture(); fn(r)
            with self.subTest(mutator=fn), self.assertRaises(AssertionError):
                a.verify_stability(r,saved,True,True)

    def test_export_has_no_media_and_rejects_traversal(self):
        names=files_for(a.specs())
        self.assertEqual(len(names),len(set(names)))
        self.assertTrue(all(Path(n).suffix in ('.py','.md','.sh','.json','.jsonl','.log') for n in names))
        for name in ('../x','/tmp/x','..'):
            with self.assertRaises(ValueError):files_for([dict(name=name)])

    def test_complete_pooled_comparison(self):
        trials={s['name']:dict(count=128,wall_s=64 if s['fixed'] else 128,fps=2 if s['fixed'] else 1)
            for s in a.specs() if not s['audit']}
        samples={n:dict(consumer_cadence=[1]*128) for n in trials}
        result=a.summarize(trials,samples)
        for clip in ('0126','0082'):
            self.assertEqual(result[clip]['matched_repeats'],[0,1,2])
            self.assertEqual(result[clip]['clock_speedups']['combined_default']['pooled'],2)


if __name__=='__main__':unittest.main()
