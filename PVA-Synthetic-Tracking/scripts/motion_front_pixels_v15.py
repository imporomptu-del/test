"""Fused mapping and installed-VPI half-scale interpolation; motion images only."""
import copy
import ctypes as C
from pathlib import Path
import numpy as np
from tiny_target.motion import pva_pyrlk as pva


class FrontLibrary:
    def __init__(self, path):
        self.lib = C.CDLL(str(Path(path).resolve(strict=True)))
        self.fn = self.lib.seaqr_motion_front_v15
        self.fn.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_float, C.c_float]
        self.fn.restype = C.c_int

    def prepare(self, image, scale, offset):
        if (image.ndim != 2 or image.dtype not in (np.dtype('uint8'), np.dtype('uint16'))
                or not image.flags.c_contiguous or not image.size or image.size > 32000000
                or any(v % 2 for v in image.shape)):
            raise ValueError('Bounded contiguous even U8/U16 image required')
        output = np.empty((image.shape[0]//2, image.shape[1]//2), image.dtype)
        status = self.fn(image.ctypes.data, output.ctypes.data, *image.shape, image.dtype.itemsize, scale, offset)
        if status:
            raise RuntimeError(f'Native motion preparation failed ({status}); no fallback')
        return output


def proxy_pixels(frame, mapping, scale=.5, *, library):
    if scale != .5 or any(size % 2 for size in frame.shape):
        raise ValueError('v15 requires even dimensions and unchanged half scale')
    image = frame.image
    if frame.bit_depth == 8 and image.dtype == np.uint8 and mapping in ('bit_shift', 'raw_robust_u16_v1'):
        return library.prepare(image, 1., 0.)
    if (mapping != 'raw_robust_u16_v1' or image.dtype != np.uint16
            or type(frame.bit_depth) is not int or not 9 <= frame.bit_depth <= 16):
        raise ValueError('v15 supports only U8 or frozen robust U16 motion preparation')
    if frame.bit_depth < 16 and np.any(image > (1 << frame.bit_depth)-1):
        raise pva.PvaMotionError('RAW source samples exceed the declared bit depth')
    # Full-source sample distribution is identical to the baseline, not a new
    # estimate derived from the proxy or from the current set of feature points.
    scale_value, offset = pva._raw_affine_parameters(frame)
    return library.prepare(image, scale_value, offset)


def logical_identity(value):
    """Remove ONLY documented execution metadata, recursively across wrappers.

    Timing is removed by the existing compact() first. Keep all quality fields,
    points, coordinates, timestamps, hashes and acceptance/rejection decisions.
    """
    value = copy.deepcopy(value)
    def visit(node):
        if isinstance(node, dict):
            # Match the typed correspondence dictionary, not arbitrary 'backend'.
            if (('correspondences' in node or ('previous_points' in node and 'current_points' in node))
                    and 'motion_image_size' in node and 'full_image_size' in node):
                node.get('backends', {}).pop('motion_image_rescale', None)
                memory = node.get('metrics', {}).get('memory_bytes', {})
                memory.pop('motion_u8_frames_created', None)
                memory.pop('motion_u16_frames_created', None)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(value)
    return value
