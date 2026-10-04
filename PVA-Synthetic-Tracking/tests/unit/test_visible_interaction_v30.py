import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from profile_visible_interaction_v30 import Trace, scope, parse_stat, MAJOR, read_pseudo
from batch_visible_interaction_v30 import schedule


class InteractionTest(unittest.TestCase):
    def test_scope(self):
        for c in ('0126', '0082'):
            for a in ('v26', 'combined'):
                scope(c, a)
        for c, a in (('0029', 'v26'), ('0126', 'v20'), ('0082', 'unknown'), ('../0082', 'v26')):
            with self.assertRaises(ValueError):
                scope(c, a)

    def test_schedule(self):
        rows = schedule()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r['name'] for r in rows}), 12)
        self.assertEqual([r['arm'] for r in rows[:4]], ['v26', 'combined', 'combined', 'v26'])
        self.assertEqual([r['mode'] for r in rows], ['trace']*6+['clean']*6)
        self.assertEqual([(r['clip'], r['arm']) for r in rows[:6]],
                         [(r['clip'], r['arm']) for r in rows[6:]])

    def test_nested_and_exception(self):
        t = Trace()
        with t.mark(8, 'parent'):
            self.assertEqual(t.wrap(lambda: 9, 'child')(), 9)
            with self.assertRaisesRegex(RuntimeError, 'test'):
                t.wrap(lambda: (_ for _ in ()).throw(RuntimeError('test')), 'error')()
        self.assertEqual(t.counts, [3, 3])
        self.assertIsNone(t.local.frame)
        self.assertTrue(all(r['frame'] == 8 and r['end_ns'] >= r['start_ns']
                            and r['thread_cpu_ns'] >= 0 for r in t.rows))
        with self.assertRaises(AssertionError):
            t.validate()

    def test_coverage(self):
        t = Trace()
        for i in range(128):
            for stage in MAJOR:
                with t.mark(i, stage):
                    pass
        t.validate()
        t.rows.append(dict(t.rows[0]))
        with self.assertRaises(AssertionError):
            t.validate()

    def test_proc_stat(self):
        fields = ['R']+['0']*37
        fields[11], fields[12], fields[36] = '123', '456', '7'
        self.assertEqual(parse_stat('123 (worker ( name)) '+' '.join(fields)),
            dict(state='R', user_ticks=123, system_ticks=456, processor=7))

    def test_sysfs_unavailable_closes_descriptor(self):
        with patch('os.open', return_value=37), patch('os.read', side_effect=BlockingIOError('EAGAIN')), patch('os.close') as close:
            with self.assertRaises(BlockingIOError):
                read_pseudo('/not-read')
            close.assert_called_once_with(37)
        with patch('os.open', return_value=38), patch('os.read', return_value=b'123\n'), patch('os.close') as close:
            self.assertEqual(read_pseudo('/not-read'), '123\n')
            close.assert_called_once_with(38)


if __name__ == '__main__':
    unittest.main()
