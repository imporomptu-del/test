"""Read-only installed-source provenance for rejected stabilization alternatives."""
import argparse
import hashlib
import json
from pathlib import Path


def run(output):
    records=[]
    for path,start,end in (
        (Path('/home/serg/src/opencv/modules/imgproc/src/imgwarp.cpp'),152,175),
        (Path('/home/serg/src/opencv/modules/core/include/opencv2/core/cuda/filters.hpp'),126,180),
    ):
        content=path.read_bytes();lines=content.decode().splitlines()
        records.append(dict(path=str(path),sha256=hashlib.sha256(content).hexdigest(),
                            first_line=start,last_line=end,excerpt='\n'.join(lines[start-1:end])))
    with output.open('x') as handle:
        json.dump(dict(records=records,real_media_read=False,
            interpretation='Installed CPU cubic uses A=-0.75; CUDA CubicFilter uses coefficients '
                'corresponding to A=-0.5 and continuous coordinates with weight normalization. '
                'These are different resampling definitions, not merely a float32 last-bit discrepancy. '
                'Neither replacement was enabled on RAW media.'),handle,indent=2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
