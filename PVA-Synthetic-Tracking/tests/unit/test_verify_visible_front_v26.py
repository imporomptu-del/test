from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from verify_visible_front_v26 import performance,schedules,safe_relative,distribution,WORKLOADS,MODES


def timings():
    trials,samples={},{}
    for c in WORKLOADS:
        for i in range(3):
            for m in MODES:
                name=f'{c}_repeat{i}_{m}';fps=1.3 if m=='candidate' else 1.
                samples[name]={k:[1.]*128 for k in ('queue_aware','consumer_cadence','request_to_complete')}
                trials[name]=dict(count=128,frames=128,clip=c,mode=m,fps=fps,wall_s=128/fps,
                    metrics={k:distribution(v) for k,v in samples[name].items()})
    return trials,samples


class FrontAuditTests(unittest.TestCase):
    def test_bounded_schedule_and_alternating_order(self):
        from batch_visible_front_v26 import initial_schedule
        prefix,full=schedules()
        self.assertEqual(prefix,initial_schedule())
        self.assertEqual((len(prefix),len(full)),(14,4))
        self.assertEqual([s['clip'] for s in full],['0029','0126','0055','0082'])
        self.assertTrue(all(s['frames'] is None and s['mode']=='candidate' for s in full))

    def test_missing_workload_never_passes(self):
        trials,samples=timings();self.assertTrue(performance(trials,samples)['passed'])
        del trials['0082_repeat2_reference']
        self.assertFalse(performance(trials,samples)['passed'])

    def test_threshold_and_each_pair_are_enforced(self):
        trials,samples=timings()
        trials['0126_repeat0_candidate'].update(fps=.99,wall_s=128/.99)
        self.assertFalse(performance(trials,samples)['passed'])
        trials,samples=timings()
        for name,t in trials.items():
            if name.endswith('candidate'):t.update(fps=1.199,wall_s=128/1.199)
        self.assertFalse(performance(trials,samples)['passed'])

    def test_request_and_cadence_regressions_are_not_hidden(self):
        for metric in ('queue_aware','request_to_complete','consumer_cadence'):
            trials,samples=timings()
            for i in (0,1):
                name=f'0126_repeat{i}_candidate';samples[name][metric]=[2.]*128
                trials[name]['metrics'][metric]=distribution(samples[name][metric])
            with self.subTest(metric=metric):self.assertFalse(performance(trials,samples)['passed'])

    def test_wrong_extent_mode_or_clip_rejected(self):
        for key,value in (('count',127),('frames',None),('mode','reference'),('clip','0029')):
            trials,samples=timings();trials['0126_repeat0_candidate'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'benchmark samples'):
                performance(trials,samples)

    def test_pooled_fps_uses_total_wall_not_mean_fps(self):
        trials,samples=timings()
        for i,fps in enumerate((1.1,1.3,1.7)):
            trials[f'0126_repeat{i}_candidate'].update(fps=fps,wall_s=128/fps)
        self.assertAlmostEqual(performance(trials,samples)['clips']['0126']['pooled_fps']['candidate'],
                               3/(1/1.1+1/1.3+1/1.7))

    def test_artifacts_must_be_bounded_relative_paths(self):
        self.assertEqual(safe_relative('build_03/source/kernel.cu'),Path('build_03/source/kernel.cu'))
        for value in ('/tmp/elsewhere','../outside','build/../../outside','.',''):
            with self.subTest(value=value),self.assertRaises(ValueError):safe_relative(value)


if __name__=='__main__':unittest.main()
