"""Research adapter: prepare the same motion lattice without a full-size upload."""
from motion_reuse_v12 import ReuseMotionV12, Prepared
from raw16_speed_v8_common import FeaturePixelsCache
from pathlib import Path
from motion_front_pixels_v15 import proxy_pixels, FrontLibrary


class DirectMotionV15(ReuseMotionV12):
    def __init__(self, config):
        super().__init__(config)
        if self.config.feature_image_scale != .5:
            self.close()
            raise ValueError('v15 retains the frozen half-size motion geometry')
        self._front = FrontLibrary(Path(__file__).parent/'build/libmotion_front_v15.so')
        self._pixel_cache = FeaturePixelsCache(lambda frame, mapping: proxy_pixels(frame, mapping, library=self._front))

    def _prepare_images(self, previous, current, previous_pixels, current_pixels,
                        motion_size, full_size, uses_u16):
        if any(p.shape != (motion_size[1], motion_size[0]) for p in (previous_pixels, current_pixels)):
            raise ValueError('Prepared proxy geometry mismatch')
        self._active_hit = self._eligible(previous, current)
        vpi = self._vpi
        image_format = vpi.Format.U16 if uses_u16 else vpi.Format.U8
        if self._active_hit:
            previous_image = self._cached.image
            previous_motion = self._cached.motion
            self.hits += 1
        else:
            self._cached = None
            self.misses += 1
            previous_image = vpi.asimage(previous_pixels, image_format)
            previous_motion = previous_image
        current_image = vpi.asimage(current_pixels, image_format)
        current_motion = current_image
        self._pending = Prepared(current, current_pixels, current_image, current_motion,
                                 None, self.config, self._stream)
        return previous_image, current_image, previous_motion, current_motion, 'CPU_fused_half_linear_v15'
