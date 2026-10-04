"""Opt-in exact sparse-mask execution; reference validation/dense path unchanged."""
import ctypes as C
import hashlib
import inspect
from pathlib import Path
import numpy as np
from tiny_target.visible_learning import shape_learning_mask as reference

REFERENCE_SHA = 'eb6fd224adf0a6e958d0a6b9a8f65586d48d6e452d6eca4e9917e4386283c868'
OLD = '''        learn = support & True
        if points:
            xy = np.concatenate(points)
            for dy, dx in offsets:
                x, y = xy[:, 0] + dx, xy[:, 1] + dy
                inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                learn[y[inside], x[inside]] = False
        return learn'''


class LearningMaskV17:
    def __init__(self, library):
        path = Path(inspect.getfile(reference))
        if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA:
            raise ValueError('Unknown reference learning-mask implementation')
        source = inspect.getsource(reference)
        if source.count(OLD) != 1:
            raise ValueError('Sparse reference branch changed')
        self.lib = C.CDLL(str(Path(library).resolve(strict=True)))
        self.fn = self.lib.seaqr_learning_mask_v17
        self.fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_void_p, C.c_int]
        self.fn.restype = C.c_int
        self.calls = 0
        namespace = dict(reference.__globals__, _sparse_v17=self.sparse)
        exec(compile(source.replace(OLD, '        return _sparse_v17(support, points, offsets)'),
                     '<learning_mask_v17:exact-sparse>', 'exec'), namespace)
        self.implementation = namespace['shape_learning_mask']

    def sparse(self, support, points, offsets):
        # Frozen pipeline uses bool masks. Preserve generic reference behavior
        # for other dtypes through an explicit unchanged Python sparse operation.
        if support.dtype != np.bool_ or not support.size or support.size > 32000000:
            learn = support & True
            if points:
                xy = np.concatenate(points)
                for dy, dx in offsets:
                    x, y = xy[:, 0]+dx, xy[:, 1]+dy
                    inside = (x>=0)&(x<support.shape[1])&(y>=0)&(y<support.shape[0])
                    learn[y[inside], x[inside]] = False
            return learn
        source = np.ascontiguousarray(support)
        xy = np.ascontiguousarray(np.concatenate(points) if points else np.empty((0, 2)), dtype=np.int64)
        offsets = np.ascontiguousarray(offsets, dtype=np.int64)
        output = np.empty(source.shape, np.bool_)
        status = self.fn(source.ctypes.data, output.ctypes.data, *source.shape,
                         xy.ctypes.data, len(xy), offsets.ctypes.data, len(offsets))
        if status:
            raise RuntimeError(f'Native sparse learning mask failed ({status}); no fallback')
        self.calls += 1
        return output

    def __call__(self, support, regions, margin_px):
        return self.implementation(support, regions, margin_px)
