"""Record installed source/assembly evidence; no camera files are read."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2

def run(output):
    root=Path('/home/serg/src/opencv');records=[]
    for relative,start,end in (
        ('modules/imgproc/src/filter.dispatch.cpp',1275,1342),
        ('modules/imgproc/src/templmatch.cpp',566,752),
        ('modules/imgproc/src/imgwarp.cpp',908,1007),
        ('modules/imgproc/src/imgwarp.cpp',3190,3290),
        ('build/modules/imgproc/CMakeFiles/opencv_imgproc.dir/flags.make',1,10),
    ):
        path=root/relative;data=path.read_bytes()
        records.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),
            first_line=start,last_line=end,excerpt='\n'.join(data.decode().splitlines()[start-1:end])))
    obj=root/'build/modules/imgproc/CMakeFiles/opencv_imgproc.dir/src/imgwarp.cpp.o'
    assembly=subprocess.run(['objdump','-d','-C',str(obj)],check=True,capture_output=True,text=True).stdout
    symbol='<void cv::remapBicubic<cv::Cast<float, float>, float, 1, false>'
    start=assembly.index('0000000000000000 '+symbol)
    end=assembly.find('\nDisassembly of section',start)
    fragment=assembly[start:end if end>=0 else None]
    mapped=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        if 'libopencv_imgproc.so' in line:mapped.add(line.split()[-1])
    binaries={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sorted(mapped)}
    with output.open('x') as handle:
        json.dump(dict(real_media_read=False,opencv_version=cv2.__version__,
            opencv_build_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            sources=records,object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
            remap_float_cubic_assembly=fragment,loaded_imgproc_binaries=binaries),handle,indent=2,sort_keys=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
