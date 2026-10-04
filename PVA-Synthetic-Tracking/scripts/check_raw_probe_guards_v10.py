"""Generated-only reset and source-mask persistence checks on the actual GPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from resident_frontend_v10 import ResidentRawProbe
from check_raw_probe_v10 import generated
from check_resident_v10 import reference_objects,digest_arrays
from profile_raw16_efficiency import sha,write_json


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    screen,tracker,_=reference_objects();shape=(192,256)
    p=ResidentRawProbe(screen.config,shape,tracker.velocity_grid,args.library)
    q=ResidentRawProbe(screen.config,shape,tracker.velocity_grid,args.library)
    record=dict(real_media_read=False,generated_data_only=True,production_approved=False,
        library_sha256=sha(args.library),script_sha256=sha(__file__),checks={},passed=False)
    try:
        # Dirty temporal state, ring slots and source mask before resetting.
        for i in range(20):
            raw,mask,matrix=generated(shape,i,128.,'holes');p.push(raw,i,i*100000000,matrix,mask)
        p.ring.run();p.reset()
        try:p.debug()
        except ValueError:record['checks']['reset_debug_rejected']=True
        try:p.ring.download()
        except RuntimeError:record['checks']['reset_stale_output_rejected']=True
        for i in range(20):
            raw,mask,matrix=generated(shape,i,64.,'dense')
            # None must mean all-valid after reset, not the previous hole mask.
            p.push(raw,i,i*100000000,matrix,None);q.push(raw,i,i*100000000,matrix,mask)
        p.ring.run();q.ring.run()
        record['checks']['reset_matches_fresh']=digest_arrays(p.ring.download())==digest_arrays(q.ring.download())
        record['checks']['reset_debug_matches_fresh']=digest_arrays(p.debug())==digest_arrays(q.debug())
        p.reset();q.reset()
        mask=np.ones(shape,bool);mask[65:93,90:130]=False
        for i in range(20):
            raw,_,matrix=generated(shape,i,128.,'dense')
            p.push(raw,i,i*100000000,matrix,mask if i==0 else None)
            q.push(raw,i,i*100000000,matrix,mask)
        p.ring.run();q.ring.run()
        record['checks']['cached_mask_matches_uploaded_mask']=digest_arrays(p.ring.download())==digest_arrays(q.ring.download())
        record['checks']['cached_mask_debug_matches']=digest_arrays(p.debug())==digest_arrays(q.debug())
        record['passed']=len(record['checks'])==6 and all(record['checks'].values())
    finally:p.close();q.close();screen.close();write_json(args.output,record)
    print(json.dumps(record));return 0 if record['passed'] else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','output'):p.add_argument('--'+name,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
