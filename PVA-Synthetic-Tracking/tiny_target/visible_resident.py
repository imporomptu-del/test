"""Experimental GPU-resident detector core with exact CPU sampling/shape oracle."""
import ctypes as C
import math
from pathlib import Path
import time
import cv2
import numpy as np
from .visible_learning import shape_learning_mask
from .visible_noise import tile_noise_statistics
from .visible_shapes import consolidate_half_height
from .visible_warp_exact import CudaWarpFrame

PEAK_DTYPE=np.dtype([('x','<i4'),('y','<i4'),('score','<f4'),('response','<f4'),('noise','<f4')])


def decode_peak_cells(peaks):
    """Convert only valid records, preserving cell and within-cell order exactly."""
    cells = []
    for j, cell in enumerate(peaks):
        # NumPy's structured tolist converts scalar types once in C. Filtering
        # invalid slots before Python avoids thousands of scalar field reads.
        values = cell[cell['x'] >= 0].tolist()
        if values:
            polarity = 'bright' if j % 2 == 0 else 'dark'
            cells.append([dict(x=x, y=y, polarity=polarity, score=score,
                response_dn=response, noise_sigma_dn=noise)
                for x, y, score, response, noise in values])
    return cells


def sample_layout(shape,tile,stride):
    h,w=shape;layout=[];indices=[];offset=0
    for y in range(0,h,tile):
        for x in range(0,w,tile):
            ys=slice(y,min(y+tile,h));xs=slice(x,min(x+tile,w))
            yy=np.arange(ys.start,ys.stop,stride,dtype=np.int32)
            xx=np.arange(xs.start,xs.stop,stride,dtype=np.int32)
            ids=(yy[:,None]*w+xx).ravel();indices.append(ids)
            layout.append((ys,xs,offset,len(ids)));offset+=len(ids)
    return layout,np.concatenate(indices)


class SparseSpatial:
    """Only downloaded windows exist; accidental full-image reads fail explicitly."""
    ndim=2
    def __init__(self,shape,seeds,patches):
        self.shape=tuple(shape);self.windows={};centers=[];values=[];h,w=shape
        for (x,y),patch in zip(seeds,patches):
            if x<8 or y<8 or x+8>=w or y+8>=h:continue
            self.windows[y-8,x-8]=patch
            centers.append(int(y)*w+int(x));values.append(patch.ravel())
        if centers:
            delta=np.arange(-8,9,dtype=np.int64)
            offsets=(delta[:,None]*w+delta).ravel()
            keys=(np.asarray(centers,np.int64)[:,None]+offsets).ravel()
            self.keys,first=np.unique(keys,return_index=True)
            self.values=np.concatenate(values)[first]
        else:
            self.keys=np.empty(0,np.int64);self.values=np.empty(0,np.float32)

    def __getitem__(self,item):
        yy,xx=item
        if isinstance(yy,slice) and isinstance(xx,slice):
            if yy.step not in (None,1) or xx.step not in (None,1) or yy.stop-yy.start!=17 or xx.stop-xx.start!=17:
                raise ValueError('Only downloaded17x17 windows may be read')
            return self.windows[yy.start,xx.start]
        yy,xx=np.asarray(yy),np.asarray(xx)
        if yy.shape!=xx.shape or np.any(yy<0) or np.any(xx<0) or np.any(yy>=self.shape[0]) or np.any(xx>=self.shape[1]):
            raise ValueError('Invalid sparse coordinates')
        keys=yy*self.shape[1]+xx;positions=np.searchsorted(self.keys,keys)
        if np.any(positions>=len(self.keys)) or not np.array_equal(self.keys[positions],keys):
            raise ValueError('Requested pixels were not downloaded')
        return self.values[positions]


class VisibleCudaResident:
    def __init__(self,config):
        self.config=config;self.handle=None;self.shape=None;self.segment=None;self.count=0;self.previous_valid=None
        self.lib=C.CDLL(str(Path(config.cuda_median_library).resolve(strict=True)))
        ptr=C.c_void_p;i=C.c_int;f=C.c_float
        signatures={
            'create':([i,i,i,i,ptr],ptr),'destroy':([ptr],None),
            'prepare':([ptr,ptr,ptr,ptr,i,f,ptr],i),
            'select':([ptr,ptr,i,i,f,f,ptr,ptr],i),'patches':([ptr,ptr,i,ptr],i),
            'finish':([ptr,ptr,f,f,f,f,i],i),'debug':([ptr,ptr,ptr],i)}
        for name,(args,result) in signatures.items():
            fn=getattr(self.lib,'seaqr_resident_'+name);fn.argtypes=args;fn.restype=result
        self.native_shapes=None
        if config.native_shape_library is not None:
            from .visible_shapes_native import NativeShapes
            self.native_shapes=NativeShapes(config.native_shape_library,config.native_shape_library_sha256)

    def _check(self,code):
        if code:raise RuntimeError(f'CUDA resident operation failed ({code}); no CPU fallback')

    def close(self):
        if self.handle:self.lib.seaqr_resident_destroy(self.handle);self.handle=None
        self.shape=None;self.segment=None;self.previous_valid=None;self.count=0

    def debug_state(self):
        if not self.handle:raise ValueError('No initialized state')
        b=np.empty(self.shape,np.float32);v=np.empty_like(b)
        self._check(self.lib.seaqr_resident_debug(self.handle,b.ctypes.data,v.ctypes.data))
        return b,v

    def update(self,image,valid,segment,learning_centers=()):
        cfg=self.config
        device=isinstance(image,CudaWarpFrame)
        if device:image.validate(valid)
        if image.ndim!=2 or image.shape!=valid.shape or (not device and not np.isfinite(image).all()) or min(image.shape)<1:
            raise ValueError('Finite nonempty grayscale image and matching mask required')
        start=time.perf_counter()
        if not device:image=np.ascontiguousarray(image,dtype=np.float32)
        shape=image.shape;h,w=shape
        if self.shape!=shape:
            if self.shape is not None and segment==self.segment:raise ValueError('Shape change requires coordinate reset')
            self.close();self.layout,indices=sample_layout(shape,cfg.tile_size,cfg.noise_sample_stride)
            self.handle=self.lib.seaqr_resident_create(h,w,cfg.tile_size,len(indices),indices.ctypes.data)
            if not self.handle:raise RuntimeError('CUDA resident allocation failed; no CPU fallback')
            self.samples=np.empty(len(indices),np.float32)
            self.stats=np.empty((len(self.layout),2),np.float32)
            self.peaks=np.empty((2*len(self.layout),cfg.max_candidates_per_tile_polarity),PEAK_DTYPE)
            self.counts=np.empty(2*len(self.layout),np.int32)
            self.shape=shape;self.segment=None
        support=cv2.erode(valid.astype(np.uint8),np.ones((13,13),np.uint8),
            borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
        reset=self.segment is None or segment!=self.segment
        if reset:self.count=0;self.previous_valid=support.copy();self.segment=segment
        self.count+=1;ready=self.count>cfg.warmup_frames
        eligible=support & self.previous_valid if ready else np.zeros_like(support)
        if device:
            image.prepare(self.lib,self.handle,support,reset,cfg.noise_floor_dn**2,self.samples)
        else:
            blur=cv2.GaussianBlur(image,(5,5),0.8)
            self._check(self.lib.seaqr_resident_prepare(self.handle,image.ctypes.data,blur.ctypes.data,
                support.ctypes.data,int(reset),cfg.noise_floor_dn**2,self.samples.ctypes.data))
        stats,noise_sigma_median=tile_noise_statistics(self.samples,support,self.layout,
            cfg.noise_sample_stride,cfg.noise_floor_dn)
        self.stats[:]=stats
        self._check(self.lib.seaqr_resident_select(self.handle,self.stats.ctypes.data,
            cfg.max_candidates_per_tile_polarity,int(ready),cfg.temporal_threshold_sigma,
            cfg.spatial_threshold_sigma,self.peaks.ctypes.data,self.counts.ctypes.data))
        cells=decode_peak_cells(self.peaks)
        proposals=[cell[rank] for rank in range(cfg.max_candidates_per_tile_polarity) for cell in cells if rank<len(cell)]
        dropped=max(0,len(proposals)-cfg.max_candidates_per_frame);proposals=proposals[:cfg.max_candidates_per_frame]
        shape_metrics={}
        if cfg.shape_measurement_mode=='mutual_half_height_r8':
            seeds=np.asarray([[p['x'],p['y']] for p in proposals],dtype=np.int32).reshape(-1,2)
            patches=np.empty((len(seeds),17,17),np.float32)
            if len(seeds):self._check(self.lib.seaqr_resident_patches(self.handle,seeds.ctypes.data,len(seeds),patches.ctypes.data))
            if self.native_shapes is None:
                sparse=SparseSpatial(shape,seeds,patches)
                proposals,shape_metrics=consolidate_half_height(proposals,sparse,eligible,
                    include_support=cfg.learning_protection_geometry=='observed_shape')
            else:
                proposals,shape_metrics=self.native_shapes.consolidate(proposals,shape,seeds,patches,eligible,
                    include_support=cfg.learning_protection_geometry=='observed_shape')
        learn=support
        if learning_centers and len(learning_centers)>2*cfg.max_active_tracks_per_polarity:
            raise ValueError('Learning protection exceeds active-track bound')
        if learning_centers and cfg.learning_protection_geometry=='observed_shape':
            learn=shape_learning_mask(support,learning_centers,cfg.position_sigma_px)
        elif learning_centers and cfg.learning_exclusion_radius_px:
            learn=support.copy();radius=cfg.learning_exclusion_radius_px
            for cx,cy in learning_centers:
                if not math.isfinite(cx) or not math.isfinite(cy):raise ValueError('Finite learning center required')
                x,y=round(cx),round(cy);x0,x1=max(0,x-radius),min(w,x+radius+1);y0,y1=max(0,y-radius),min(h,y+radius+1)
                if x0>=x1 or y0>=y1:continue
                yy,xx=np.ogrid[y0:y1,x0:x1];learn[y0:y1,x0:x1]&=(xx-cx)**2+(yy-cy)**2>radius**2
        self._check(self.lib.seaqr_resident_finish(self.handle,learn.ctypes.data,cfg.background_alpha,
            cfg.pixel_noise_alpha,cfg.pixel_noise_clip_sigma**2,cfg.noise_floor_dn**2,
            int(cfg.learning_protection_mode=='variance_only')))
        self.previous_valid=support
        return proposals,dict(full_shape_hw=[h,w],configured_crop=None,native_pixel_sampling=True,
            searchable_pixels=int(eligible.sum()),total_pixels=int(image.size),warmup=not ready,
            filter_support_margin_px=6,above_threshold_count=int(self.counts.sum()),
            dropped_at_tile_cap=int(np.maximum(0,self.counts-cfg.max_candidates_per_tile_polarity).sum()),
            dropped_at_frame_cap=dropped,noise_sigma_median_dn=noise_sigma_median,
            detection_ms=1000*(time.perf_counter()-start),
            **({'shape_measurement':shape_metrics} if shape_metrics else {}))
