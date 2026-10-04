from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from profile_raw16_v6 import Spans, run


class Clock:
    def __init__(self): self.time = 0
    def __call__(self): return self.time
    def advance(self, ns): self.time += ns


class StageProfileTests(unittest.TestCase):
    def test_nested_spans_are_exclusive_and_additive(self):
        clock = Clock(); spans = Spans(clock)
        def outer():
            clock.advance(10)
            spans.call('child', 'compute', clock.advance, 30)
            clock.advance(20)
        spans.call('root', 'other', outer)
        result = spans.summary()
        self.assertEqual(result['accounting_error_ns'], 0)
        self.assertEqual(result['wall_s'], 60e-9)
        self.assertEqual(result['groups']['compute']['exclusive_s'], 30e-9)
        self.assertEqual(result['groups']['other']['exclusive_s'], 30e-9)

    def test_same_name_different_parents_and_inherited_group(self):
        clock = Clock(); spans = Spans(clock)
        def outer():
            for group in ('left', 'right'):
                spans.call(group, group, lambda: spans.call('inner', None, clock.advance, 5))
        spans.call('root', 'other', outer)
        result = spans.summary()
        self.assertEqual(len(result['spans']), 5)
        self.assertEqual(result['groups']['left']['exclusive_s'], 5e-9)
        self.assertEqual(result['groups']['right']['exclusive_s'], 5e-9)

    def test_exception_still_closes_spans(self):
        clock = Clock(); spans = Spans(clock)
        def failed():
            clock.advance(9)
            raise RuntimeError('test')
        with self.assertRaises(RuntimeError): spans.call('root', 'other', failed)
        self.assertFalse(spans.stack)
        self.assertEqual(spans.summary()['wall_s'], 9e-9)

    def test_generator_consumer_time_is_not_decode_time(self):
        clock = Clock(); spans = Spans(clock)
        def source():
            for i in range(2):
                clock.advance(3)
                yield i
        def consume():
            for _ in spans.iterator('decode', 'decode', source()): clock.advance(100)
        spans.call('root', 'other', consume)
        result = spans.summary()
        self.assertEqual(result['groups']['decode']['exclusive_s'], 6e-9)
        self.assertEqual(result['groups']['other']['exclusive_s'], 200e-9)

    def test_patch_returns_identical_value_and_restores_function(self):
        clock = Clock(); spans = Spans(clock)
        def f(value):
            clock.advance(2)
            return value
        owner = SimpleNamespace(f=f); value = object()
        with ExitStack() as context:
            spans.wrap(context, owner, 'f', 'compute')
            self.assertIs(spans.call('root', 'other', owner.f, value), value)
        self.assertIs(owner.f, f)
        self.assertEqual(spans.summary()['wall_s'], 2e-9)

    def test_unapproved_id_fails_before_other_arguments_or_media(self):
        with self.assertRaises(ValueError): run(SimpleNamespace(clip='not_authorized'))

    def test_unfinished_or_multiple_roots_rejected(self):
        clock = Clock(); spans = Spans(clock)
        with self.assertRaises(ValueError): spans.call('root', 'other', spans.summary)
        spans.call('second', 'other', clock.advance, 1)
        with self.assertRaises(ValueError): spans.summary()


if __name__ == '__main__':
    unittest.main()
