from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from attribute_visible_native_v25 import Attribution


class AttributionTests(unittest.TestCase):
    def test_thread_local_frame_identity_under_concurrency(self):
        binding=SimpleNamespace(active=threading.local(),current=SimpleNamespace(index=3))
        trace=Attribution(binding);barrier=threading.Barrier(2)
        measured=trace.wrap(lambda:barrier.wait(),'worker')
        def call(index):binding.active.frame=index;measured()
        threads=[threading.Thread(target=call,args=(i,)) for i in (10,20)]
        for t in threads:t.start()
        for t in threads:t.join(2);self.assertFalse(t.is_alive())
        self.assertEqual({r['frame'] for r in trace.events},{10,20})
        self.assertEqual(len({r['thread'] for r in trace.events}),2)
        self.assertEqual(trace.wrap(lambda:42,'main',False)(),42)
        self.assertEqual(trace.events[-1]['frame'],3)
        for r in trace.events:
            self.assertLess(r['start_ns'],r['end_ns']);self.assertGreaterEqual(r['thread_cpu_ns'],0)

    def test_initialization_and_exceptions(self):
        binding=SimpleNamespace(active=threading.local(),current=None);trace=Attribution(binding)
        self.assertEqual(trace.wrap(lambda:4,'init')(),4);self.assertEqual(trace.events,[])
        binding.active.frame=5
        def fail():raise RuntimeError('generated')
        with self.assertRaisesRegex(RuntimeError,'generated'):trace.wrap(fail,'failure')()
        self.assertIn('generated',trace.events[0]['error'])


if __name__=='__main__':unittest.main()
