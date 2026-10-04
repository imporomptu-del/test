"""Explicit serial resident-front adapter; original shape/track arithmetic."""
import ctypes as C
import hashlib
import math
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from tiny_target.visible_resident import VisibleCudaResident,PEAK_DTYPE,decode_peak_cells
from tiny_target.visible_warp_exact import CudaWarpFrame

REFERENCE_LIBRARY_SHA='0877b81e2332c329c3bbae2d07a0db5b615c821aa11943c6a68c797f37bdc197'
CUDA_SOURCES={
 'phase20_cuda_resident.cu':'cf64594508ef47c599e30c6c41fa2c0ee2888c3afa2cdbf37df783113dc8261a',
 'phase20_cuda_median.cu':'452e1cfe96636cb8dad74dd74471507a1318992ec1c62fa43767e2876ebf05f7',
 'phase20_cuda_warp_exact.cu':'7eb54309d9d0e504ec8d0fd86532346d4549a0ce2d7d2b1d8e0943bda45c8374',
 'phase20_cuda_integrated.cu':'ea607056117051b9655c28ecd2ee7bf444fc53f344e629a732a1c51cd88a8366'}


def pack_learning(shape,regions,margin):
    """Original validation/rounding/disk, compact integer transfer only."""
    if len(shape)!=2 or min(shape)<1 or max(shape)>=32767 or not math.isfinite(margin) or not 0<margin<=16:
        raise ValueError('Bounded shape and positive uncertainty margin required')
    if len(regions)>512:raise ValueError('At most512 prior measured regions')
    h,w=shape;points=[]
    for region in regions:
        xy=np.asarray(region['support_reference_xy'],dtype=np.float64)
        if xy.ndim!=2 or xy.shape[1]!=2 or not 0<len(xy)<=1024 or not np.isfinite(xy).all():
            raise ValueError('Bounded finite prior observed footprint required')
        xy=np.rint(xy)
        inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        xx,yy=xy[inside].astype(np.int64).T
        if len(xx):points.append(np.column_stack((xx,yy)))
    radius=math.ceil(margin);yy,xx=np.mgrid[-radius:radius+1,-radius:radius+1]
    disk=(xx*xx+yy*yy<=margin*margin).astype(np.uint8)
    offsets=np.column_stack(np.nonzero(disk))-radius
    # ABI takes (dx,dy); reference sparse loops use (dy,dx).
    return (np.ascontiguousarray(np.concatenate(points) if points else np.empty((0,2)),dtype=np.int32),
            np.ascontiguousarray(offsets[:,::-1],dtype=np.int32))


def signatures(lib):
    p=C.c_void_p;i=C.c_int;f=C.c_float;d=C.c_double
    specs={'abi':([],i),'create':([i]*4,p),'destroy':([p],None),'core':([p],p),
        'prepare_warp':([p,p,i,i,i,f,d,i,f,f]+[p]*5,i),
        'prepare_host':([p]*4+[i,i,f,d,i,f,f]+[p]*5,i),
        'finish':([p,p,i,p,i,f,f,f,f,i],i),'debug':([p]*5,i),
        'noise_probe':([p,p,p,d,p,p],i)}
    for name,(args,result) in specs.items():
        fn=getattr(lib,'seaqr_front_v26_'+name);fn.argtypes=args;fn.restype=result
    if lib.seaqr_front_v26_abi()!=1:raise RuntimeError('Unsupported resident-front ABI')


def attach_warp(original):
    def call(self,image,mask,matrix,*,device=False,erosion_px=2):
        result=original(self,image,mask,matrix,device=device,erosion_px=erosion_px)
        if device:result[0].front_erosion_v26=erosion_px
        return result
    return call


class ResidentFrontV26(VisibleCudaResident):
    """Original API with a different, explicitly selected implementation."""
    def __init__(self,config):
        if (config.learning_protection_geometry!='observed_shape' or config.learning_exclusion_radius_px!=0
            or config.native_shape_library is None or config.shape_measurement_mode!='mutual_half_height_r8'
            or not 1<=config.tile_size<=256 or not 1<=config.noise_sample_stride<=256
            or math.ceil(config.tile_size/config.noise_sample_stride)**2>4096):
            raise ValueError('Unsupported bounded resident-front configuration')
        super().__init__(config);signatures(self.lib)
        self.front=None;self.owner_thread=threading.get_ident();self.busy=False;self.poisoned=False
        self.calls=0;self.device_calls=0;self.host_calls=0;self.finish_calls=0;self.learning_points=0

    def _owned(self):
        if threading.get_ident()!=self.owner_thread:raise RuntimeError('Resident front is single-owner')

    def close(self):
        self._owned()
        if self.busy:raise RuntimeError('Cannot close active resident front')
        if self.front:self.lib.seaqr_front_v26_destroy(self.front)
        self.front=None;self.handle=None;self.shape=None;self.segment=None;self.count=0
        self.previous_valid=None;self.poisoned=False

    def _allocate(self,shape,segment):
        if self.shape==shape:return
        if self.shape is not None and segment==self.segment:raise ValueError('Shape change requires coordinate reset')
        self.close();h,w=shape;cfg=self.config
        self.front=self.lib.seaqr_front_v26_create(h,w,cfg.tile_size,cfg.noise_sample_stride)
        if not self.front:raise RuntimeError('Resident front allocation failed')
        self.handle=self.lib.seaqr_front_v26_core(self.front)
        if not self.handle:self.close();raise RuntimeError('Missing resident core')
        try:
            tiles=math.ceil(h/cfg.tile_size)*math.ceil(w/cfg.tile_size)
            self.eligible=np.empty(shape,np.bool_);self.sigmas=np.empty(tiles,np.float64)
            self.peaks=np.empty((2*tiles,cfg.max_candidates_per_tile_polarity),PEAK_DTYPE)
            self.counts=np.empty(2*tiles,np.int32);self.searchable=np.empty(1,np.int32)
            self.shape=shape
        except BaseException:
            self.close();raise

    def debug_front(self):
        self._owned()
        if not self.front or self.busy:raise ValueError('No idle initialized front')
        support=np.empty(self.shape,np.bool_);learn=np.empty_like(support)
        stats=np.empty((len(self.sigmas),2),np.float32);sigmas=np.empty_like(self.sigmas)
        self._check(self.lib.seaqr_front_v26_debug(self.front,*(a.ctypes.data for a in (support,learn,stats,sigmas))))
        return support,learn,stats,sigmas

    def debug_state(self):
        self._owned()
        if self.busy:raise ValueError('Active resident front')
        return super().debug_state()

    def update(self,image,valid,segment,learning_centers=()):
        self._owned()
        if self.busy or self.poisoned:raise RuntimeError('Active or failed resident front; close before reuse')
        cfg=self.config;device=isinstance(image,CudaWarpFrame)
        if device:
            image.validate(valid)
            if image.owner.lib._name!=self.lib._name:raise ValueError('No cross-library device handles')
            erosion=getattr(image,'front_erosion_v26',None)
            if type(erosion) is not int or not 0<=erosion<=16:raise ValueError('Missing validated warp erosion')
        if (image.ndim!=2 or image.shape!=valid.shape or min(image.shape)<1 or max(image.shape)>=32767
            or image.size>32000000 or valid.dtype!=np.bool_
            or (not device and not np.isfinite(image).all())):
            raise ValueError('Finite image and matching bounded boolean validity required')
        if len(learning_centers)>2*cfg.max_active_tracks_per_polarity:
            raise ValueError('Learning protection exceeds active-track bound')
        start=time.perf_counter()
        points,offsets=pack_learning(image.shape,learning_centers,cfg.position_sigma_px)
        self._allocate(image.shape,segment)
        reset=self.segment is None or segment!=self.segment
        if reset:self.count=0;self.segment=segment
        self.count+=1;ready=self.count>cfg.warmup_frames;self.busy=True
        try:
            tail=(int(reset),int(ready),cfg.noise_floor_dn**2,cfg.noise_floor_dn,
                cfg.max_candidates_per_tile_polarity,cfg.temporal_threshold_sigma,cfg.spatial_threshold_sigma,
                *(a.ctypes.data for a in (self.eligible,self.sigmas,self.peaks,self.counts,self.searchable)))
            if device:
                # Consume even on error: native state may already have advanced.
                try:self._check(self.lib.seaqr_front_v26_prepare_warp(self.front,image.owner.handle,erosion,*tail))
                finally:image.consumed=True
                self.device_calls+=1
            else:
                image=np.ascontiguousarray(image,dtype=np.float32)
                blur=cv2.GaussianBlur(image,(5,5),.8);valid=np.ascontiguousarray(valid)
                self._check(self.lib.seaqr_front_v26_prepare_host(self.front,image.ctypes.data,blur.ctypes.data,valid.ctypes.data,*tail))
                self.host_calls+=1
            cells=decode_peak_cells(self.peaks)
            proposals=[cell[rank] for rank in range(cfg.max_candidates_per_tile_polarity) for cell in cells if rank<len(cell)]
            dropped=max(0,len(proposals)-cfg.max_candidates_per_frame);proposals=proposals[:cfg.max_candidates_per_frame]
            seeds=np.asarray([[p['x'],p['y']] for p in proposals],dtype=np.int32).reshape(-1,2)
            patches=np.empty((len(seeds),17,17),np.float32)
            if len(seeds):self._check(self.lib.seaqr_resident_patches(self.handle,seeds.ctypes.data,len(seeds),patches.ctypes.data))
            proposals,shape_metrics=self.native_shapes.consolidate(proposals,image.shape,seeds,patches,self.eligible,
                include_support=True)
            self._check(self.lib.seaqr_front_v26_finish(self.front,points.ctypes.data,len(points),offsets.ctypes.data,len(offsets),
                cfg.background_alpha,cfg.pixel_noise_alpha,cfg.pixel_noise_clip_sigma**2,cfg.noise_floor_dn**2,
                int(cfg.learning_protection_mode=='variance_only')))
            self.finish_calls+=1;self.learning_points+=len(points);self.calls+=1
            return proposals,dict(full_shape_hw=list(image.shape),configured_crop=None,native_pixel_sampling=True,
                searchable_pixels=int(self.searchable[0]),total_pixels=int(image.size),warmup=not ready,
                filter_support_margin_px=6,above_threshold_count=int(self.counts.sum()),
                dropped_at_tile_cap=int(np.maximum(0,self.counts-cfg.max_candidates_per_tile_polarity).sum()),
                dropped_at_frame_cap=dropped,noise_sigma_median_dn=float(np.median(self.sigmas)),
                detection_ms=1000*(time.perf_counter()-start),shape_measurement=shape_metrics)
        except BaseException:
            self.poisoned=True;raise
        finally:self.busy=False
