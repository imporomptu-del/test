import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('host_trace_v21',
    Path(__file__).resolve().parents[2]/'scripts/profile_visible_architecture_v21.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HostTraceTest(unittest.TestCase):
    def test_nested_accounting_and_return(self):
        trace = module.HostTrace()
        child = trace.wrap(lambda: 4, 'child')
        parent = trace.wrap(lambda: child()+1, 'parent')
        self.assertEqual(parent(), 5)
        a, b = trace.rows
        self.assertEqual(b['parent'], a['id'])
        self.assertEqual(a['children_ns'], b['duration_ns'])
        self.assertEqual(a['exclusive_host_ns']+b['duration_ns'], a['duration_ns'])
        self.assertGreaterEqual(a['exclusive_host_ns'], 0)
        self.assertGreaterEqual(b['thread_cpu_ns'], 0)

    def test_exception_restores_stack(self):
        trace = module.HostTrace()
        def bad():
            raise ValueError('expected')
        with self.assertRaises(ValueError):
            trace.wrap(bad, 'bad')()
        trace.wrap(lambda: None, 'later')()
        self.assertIsNone(trace.rows[1]['parent'])
        self.assertIn('expected', trace.rows[0]['error'])

    def test_frame_propagates_to_nested_calls(self):
        trace = module.HostTrace()
        child = trace.wrap(lambda: None, 'child')
        entry = trace.wrap(lambda self, gray, frame_index, ts: child(), 'motion', True)
        entry(None, None, 12, 100)
        self.assertEqual([r['frame'] for r in trace.rows], [12, 12])

    def test_scope_rejected_before_runtime_imports(self):
        with self.assertRaises(ValueError):
            module.run('not-allowed', Path('/tmp/not-created-v21'))


if __name__ == '__main__':
    unittest.main()
