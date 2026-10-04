"""Explicit binary64 subnormal inputs, independent of arithmetic fixture creation."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from profile_visible_v17 import sha,write
from tracking_batch_v27 import BatchGeometryV27
from tracking_geometry_v20 import GeometryV20,reference
from check_tracking_geometry_v20 import same,primitive_cases

PATTERNS=(1,2,0x000fffffffffffff,0x8000000000000001)


def cases():
    for d in (2,4):
        for bits in PATTERNS:
            for sign in (0,0x8000000000000000):
                words=np.zeros((17,4),np.uint64);words[:,0]=bits
                means=np.full(4,sign,np.uint64)
                yield f'd{d}_bits{bits:016x}_mean{sign:016x}',words.view(np.float64),means.view(np.float64),d,bits,sign


def check(library,geometry):
    helper=BatchGeometryV27(library);helper.fallback=GeometryV20(geometry);rows=[]
    for name,values,mean,d,bits,sign in cases():
        before=(values.tobytes(),mean.tobytes());native=helper.calls
        if int(values.view(np.uint64)[0,0])!=bits or int(mean.view(np.uint64)[0])!=sign:
            raise AssertionError('Explicit binary64 fixture lost its bits')
        result=helper(values,{0:SimpleNamespace(mean=mean)},d,5.,5.)
        with np.errstate(all='ignore'):
            expected=reference(values,mean,d,5.,5.)
            same(expected,helper.fallback(values,mean,d,5.,5.))
            same(expected,result.get(0,values,mean,d,5.,5.))
        if before!=(values.tobytes(),mean.tobytes()):raise AssertionError('Input bits changed')
        rows.append(dict(name=name,exact=True,input_bits_hex=f'{bits:016x}',mean_bits_hex=f'{sign:016x}',
            native=helper.calls>native,residual_first_bits_hex=f'{int(expected[0].view(np.uint64)[0,0]):016x}'))
    smallest=np.array([np.nextafter(0.,1.)],dtype=np.float64).view(np.uint64)[0]
    return dict(passed=True,cases=rows,arithmetic_fixture_nextafter_bits_hex=f'{int(smallest):016x}',
        arithmetic_fixture_display=format(np.nextafter(0.,1.)),
        original_subnormal_fixtures=[dict(index=i,name=row[0],input_bits_hex=f'{int(row[1].view(np.uint64)[1,0]):016x}')
            for i,row in enumerate(primitive_cases()) if i in (64,132)],
        original_library_sha256=sha(geometry),library_sha256=sha(library),media_read=False,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in (
            'check_tracking_subnormal_v27.py','tracking_batch_v27.py','tracking_geometry_v20.py','check_tracking_geometry_v20.py')},
        caveat='Exact comparison under the current process numerical environment; explicit input bits survive fixture creation. '
               'This does not assert an unmeasured global floating-point control setting.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    # Match the original checker import order/process environment.
    import replay_tracking_v27
    write(a.output,check(a.library,a.geometry))
