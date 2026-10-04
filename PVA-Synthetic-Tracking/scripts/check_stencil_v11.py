"""Additional generated interpolation cutoff, tie and support equality checks."""
import argparse
from collections import deque
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from check_resident_v10 import reference_objects,arrays_equal,digest_arrays
from resident_tracking_v10 import ResidentTracker
from raw16_speed_v8_common import dense
from profile_raw16_efficiency import sha,write_json


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    screen,reference,_=reference_objects();normal=reference.velocity_grid.copy()
    tiny=np.array([0,1e-8,1e-7,2e-7,.99999994,1.0000001,-.99999994,-1.0000001],np.float32)
    edge=np.array([(tiny[i%8],tiny[(i//8)%8]) for i in range(48)],np.float32)
    record=dict(real_media_read=False,library_sha256=sha(args.library),checker_sha256=sha(__file__),rows=[],passed=False)
    try:
        for name,grid in (('normal',normal),('cutoffs',edge),('velocity_ties',np.zeros((48,2),np.float32))):
            reference.velocity_grid=grid
            for shape in ((7,13),(65,97)):
                p=ResidentTracker(shape,grid,args.library)
                try:
                    for polarity in ('bright','dark'):
                        for scene in ('full','empty','sparse','border','flat_ties'):
                            p.reset(polarity=polarity);frames=deque(maxlen=16);rng=np.random.default_rng(11693)
                            for i in range(24):
                                a=rng.normal(0,3,shape).astype(np.float32);mask=np.ones(shape,bool)
                                if scene=='empty':mask[:]=False
                                if scene=='sparse':mask=rng.random(shape)>.75
                                if scene=='border':mask[1:-1,1:-1]=False
                                if scene=='flat_ties':a[:]=1
                                t=i*100000000+(31234567 if i>=9 else 0)
                                frames.append(dense._DenseMatchedFrame(a,mask,t,i,0,True,polarity))
                                p.push(a,mask,i,t,polarity=polarity)
                            old=reference.integrate(list(frames));p.run();out=p.download()
                            exact=arrays_equal(old,out)
                            record['rows'].append(dict(grid=name,shape=shape,polarity=polarity,scene=scene,
                                exact=exact,output_sha256=digest_arrays(out)))
                            if not exact:raise AssertionError('Stencil arithmetic/support/ties changed')
                finally:p.close()
        record['passed']=len(record['rows'])==60
    finally:screen.close();write_json(args.output,record)
    print(json.dumps({'passed':record['passed'],'cases':len(record['rows'])}));return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
