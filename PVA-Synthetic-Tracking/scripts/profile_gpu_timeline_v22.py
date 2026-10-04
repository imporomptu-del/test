"""Add NVTX labels around frozen v21 host tracing of the v20 candidate."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

V21 = Path('/tmp/seaqr_architecture_v21_XMExvL')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(clip, output, library, host_sha):
    if clip not in ('0126', '0082'):
        raise ValueError('Only two authorized development AVI prefixes')
    if sha(V21/'profile_visible_architecture_v21.py') != host_sha:
        raise ValueError('Frozen host tracing source changed')
    receipt = output.with_suffix('.nvtx.json')
    if output.exists() or receipt.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(V21))
    import profile_visible_architecture_v21 as host
    bridge = ctypes.CDLL(str(library.resolve(strict=True)))
    bridge.seaqr_trace_push.argtypes = [ctypes.c_char_p]
    bridge.seaqr_trace_push.restype = ctypes.c_int
    bridge.seaqr_trace_pop.argtypes = []
    bridge.seaqr_trace_pop.restype = ctypes.c_int
    original = host.HostTrace.wrap
    counter = [0, 0]
    def wrap(trace, function, name, frame_entry=False):
        timed = original(trace, function, name, frame_entry)
        def annotated(*args, **kwargs):
            frame = (args[2] if len(args) > 2 else kwargs['frame_index']) if frame_entry else trace.frame
            bridge.seaqr_trace_push(f'seaqr|{frame}|{name}'.encode())
            counter[0] += 1
            try:
                return timed(*args, **kwargs)
            finally:
                bridge.seaqr_trace_pop()
                counter[1] += 1
        return annotated
    result = dict(passed=False, error=None, clip=clip, frames=128,
                  script_sha256=sha(__file__), host_script_sha256=host_sha,
                  bridge_sha256=sha(library), raw16_accessed=False, defaults_changed=False)
    try:
        with patch.object(host.HostTrace, 'wrap', wrap):
            host.run(clip, output)
        if counter[0] != counter[1] or not counter[0]:
            raise AssertionError('Unbalanced annotation ranges')
        result.update(passed=True, host_receipt_sha256=sha(output.with_suffix('.host_trace.json')))
    except BaseException as exc:
        result['error'] = repr(exc)
        raise
    finally:
        result['push_pop_counts'] = counter
        with receipt.open('x') as handle:
            json.dump(result, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--host-sha', required=True)
    args = parser.parse_args()
    run(args.clip, args.output, args.library, args.host_sha)
