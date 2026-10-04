"""Explicit, translation-only cubic CUDA conformance adapter; no fallback."""
import ctypes as C
import hashlib
from pathlib import Path
import cv2
import numpy as np


def cubic_reference_table():
    """Sample the installed CPU interpolation operator with unit basis images."""
    yy,xx=np.mgrid[:32,:32].astype(np.float32)
    xx=1+xx/32;yy=1+yy/32
    table=np.empty((32,32,16),np.float32)
    for i in range(16):
        basis=np.zeros((4,4),np.float32);basis[i//4,i%4]=1
        table[:,:,i]=cv2.remap(basis,xx,yy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.ascontiguousarray(table)


def translation_maps(shape,matrix):
    h,w=shape;m=np.asarray(matrix,dtype=np.float64)
    if m.shape!=(3,3) or not np.isfinite(m).all():raise ValueError('Finite 3x3 transform required')
    base=m.copy();base[:2,2]=0
    if not np.array_equal(base,np.eye(3)):raise ValueError('Only exact camera translations supported')
    if not 0<h<32767 or not 0<w<32767 or np.max(np.abs(m[:2,2]))>32700:raise ValueError('Unsupported shape/translation bound')
    bw=min(1024//min(16,h),w)
    x=np.arange(w,dtype=np.float64);y=np.arange(h,dtype=np.float64)
    # Preserve the CPU warp's block-relative coordinate arithmetic before its
    # existing 1/32 interpolation-table quantization. No new motion quantization.
    u=(np.floor(x/bw)*bw-m[0,2])+np.remainder(x,bw)
    v=y-m[1,2]
    maps=[]
    for values in (u,v):
        fixed=np.rint(values*32).astype(np.int32)
        maps.append(np.ascontiguousarray(np.column_stack((fixed>>5,fixed&31,np.rint(values).astype(np.int32))),dtype=np.int32))
    return maps


class CudaWarpFrame:
    """Single-use, generation-checked reference to a private device workspace.

    This is intentionally not a Frame or an ndarray. No implicit download or
    arbitrary pointer conversion is allowed. The serial detector consumes it.
    """
    ndim=2

    def __init__(self,owner,valid):
        self.owner=owner;self.generation=owner.generation;self.valid=valid
        self.shape=valid.shape;self.size=valid.size;self.consumed=False

    def validate(self,valid):
        if self.consumed or not self.owner.handle or self.generation!=self.owner.generation:
            raise ValueError('Stale, closed or already consumed GPU frame')
        if valid is not self.valid:raise ValueError('GPU frame requires its original validity mask')

    def download_for_verification(self):
        self.validate(self.valid)
        image=np.empty(self.shape,np.float32);blur=np.empty_like(image)
        self.owner._check(self.owner.lib.seaqr_warp_download(self.owner.handle,image.ctypes.data,blur.ctypes.data))
        return image,blur

    def prepare(self,library,handle,support,reset,floor2,samples):
        self.validate(self.valid)
        if library._name!=self.owner.lib._name:
            raise ValueError('Warp and resident detector must use the same compiled library')
        fn=library.seaqr_resident_prepare_warp
        fn.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_float,C.c_void_p];fn.restype=C.c_int
        self.owner._check(fn(handle,self.owner.handle,support.ctypes.data,int(reset),floor2,samples.ctypes.data))
        self.consumed=True


class CudaCubicTranslation:
    def __init__(self,library,mode=3):
        if mode not in (0,1,2,3):raise ValueError('Unknown arithmetic mode')
        self.mode=mode;self.handle=None;self.shape=None;self.table=cubic_reference_table();self.generation=0
        self.lib=C.CDLL(str(Path(library).resolve(strict=True)));ptr=C.c_void_p;i=C.c_int
        self.lib.seaqr_warp_create.argtypes=[i,i,ptr];self.lib.seaqr_warp_create.restype=ptr
        self.lib.seaqr_warp_destroy.argtypes=[ptr];self.lib.seaqr_warp_destroy.restype=None
        self.lib.seaqr_warp_run.argtypes=[ptr,ptr,ptr,ptr,ptr,i,i,ptr,ptr];self.lib.seaqr_warp_run.restype=i

    @staticmethod
    def _check(code):
        if code:raise RuntimeError(f'CUDA cubic operation failed ({code}); no fallback')

    def close(self):
        if self.handle:self.lib.seaqr_warp_destroy(self.handle);self.handle=None
        self.shape=None;self.generation+=1

    def __call__(self,image,mask,matrix,*,device=False,erosion_px=2):
        if image.ndim!=2 or image.dtype not in (np.uint8,np.float32) or image.shape!=mask.shape or mask.dtype not in (np.bool_,np.uint8) or (image.dtype==np.float32 and not np.isfinite(image).all()):
            raise ValueError('Finite uint8/float32 image and matching mask required')
        if not isinstance(erosion_px,int) or isinstance(erosion_px,bool) or not 0<=erosion_px<=16:
            raise ValueError('Bounded integer erosion radius required')
        xm,ym=translation_maps(image.shape,matrix)
        if self.shape!=image.shape:
            self.close();self.handle=self.lib.seaqr_warp_create(*image.shape,self.table.ctypes.data)
            if not self.handle:raise RuntimeError('CUDA cubic allocation failed')
            self.shape=image.shape
        image=np.ascontiguousarray(image);mask=np.ascontiguousarray(mask,dtype=np.uint8)
        self.generation+=1
        output=None if device else np.empty(image.shape,np.float32);valid=np.empty(image.shape,np.uint8)
        code=self.lib.seaqr_warp_run(self.handle,image.ctypes.data,mask.ctypes.data,xm.ctypes.data,ym.ctypes.data,
                                   int(image.dtype==np.uint8),self.mode,output.ctypes.data if output is not None else None,valid.ctypes.data)
        self._check(code)
        if device:
            if self.mode!=3:raise ValueError('Only conformant cubic mode is allowed for device integration')
            self.gaussian()
            if erosion_px:valid=cv2.erode(valid,np.ones((2*erosion_px+1,2*erosion_px+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            valid=valid.astype(bool);valid.setflags(write=False)
            return CudaWarpFrame(self,valid),valid
        return output,valid

    def gaussian(self,download=False):
        if not self.handle:raise ValueError('Warp must precede Gaussian filtering')
        ptr=C.c_void_p;f=C.c_float
        fn=self.lib.seaqr_warp_gaussian;fn.argtypes=[ptr,f,f,f,C.c_int,ptr];fn.restype=C.c_int
        self.lib.seaqr_warp_download.argtypes=[ptr,ptr,ptr];self.lib.seaqr_warp_download.restype=C.c_int
        k=cv2.getGaussianKernel(5,.8,cv2.CV_32F).ravel()
        out=np.empty(self.shape,np.float32) if download else None
        self._check(fn(self.handle,k[2],k[1],k[0],6,out.ctypes.data if out is not None else None))
        return out

    def verify_reference(self,include_gaussian=False):
        """Fail closed on incompatible OpenCV arithmetic; no label-dependent tuning."""
        rng=np.random.default_rng(247);warp_cases=0;gaussian_cases=0
        try:
            for n in range(32):
                shape=(17,31);im=rng.normal(40,30,shape).astype(np.float32)
                mask=(rng.random(shape)>.05).astype(np.uint8);m=np.eye(3)
                m[:2,2]=(n+.5)/32,-((n*11)%32)/32
                actual,valid=self(im,mask,m)
                expected=cv2.warpPerspective(im,m,(31,17),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                expected_valid=cv2.warpPerspective(mask,m,(31,17),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                if not np.array_equal(actual,expected) or not np.array_equal(valid,expected_valid):
                    raise RuntimeError('Installed OpenCV cubic arithmetic is incompatible; GPU execution refused')
                warp_cases+=1
            if include_gaussian:
                for width in range(1,34):
                    im=rng.normal(40,30,(17,width)).astype(np.float32)
                    self(im,np.ones(im.shape,np.uint8),np.eye(3))
                    if not np.array_equal(self.gaussian(download=True),cv2.GaussianBlur(im,(5,5),.8)):
                        raise RuntimeError('Installed OpenCV Gaussian arithmetic is incompatible; GPU execution refused')
                    gaussian_cases+=1
            return dict(exact=True,warp_cases=warp_cases,gaussian_cases=gaussian_cases,opencv_version=cv2.__version__,
                        opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
                        cubic_table_sha256=hashlib.sha256(self.table.tobytes()).hexdigest())
        finally:self.close()
