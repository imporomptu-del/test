"""Audit source-only annotation centroids and export a nearest-pixel review sheet."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/tiny_target/accuracy_v38_20260925"


def main():
    ref=json.loads((BASE/"compact_light_reference_v1.json").read_text())
    for path,digest in ref["inputs_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
            raise ValueError("Changed reference source input")
    with np.load(ref["source_binding"]["native_archive_path"],allow_pickle=False) as archive:
        frames=archive["00_0029_0012_0024"]
    assert frames.shape==(13,192,256,3) and frames.dtype==np.uint8
    for r in ref["frames"]:
        if r["visibility"]!="visible":
            assert r["source_xy"] is None
            continue
        x,y,w,h=r["manual_measurement_roi_crop_xywh"]
        data=frames[r["frame_index"]-12,y:y+h,x:x+w].mean(axis=2)
        yy,xx=np.indices(data.shape)
        excess=np.maximum(data-np.median(data),0)
        point=[round(float(np.sum(excess*(xx+x))/excess.sum()),1),round(float(np.sum(excess*(yy+y))/excess.sum()),1)]
        assert point==r["crop_local_xy"]
        np.testing.assert_allclose(np.asarray(point)+(2619,2998),r["source_xy"],atol=1e-9,rtol=0)
    # Fixed lower part of the already-reviewed crop, not a new media window.
    # Every source pixel is repeated twice, without contrast enhancement.
    canvas=np.full((4*164,4*408,3),24,np.uint8)
    for i,frame in enumerate(frames):
        row,col=divmod(i,4)
        x,y=col*408,row*164
        repeated=np.repeat(np.repeat(frame[125:185,:200],2,axis=0),2,axis=1)
        canvas[y:y+120,x:x+400]=repeated
        cv2.putText(canvas,f'frame {i+12} | local x0..199 y125..184 | 2x nearest',
                    (x+4,y+139),cv2.FONT_HERSHEY_SIMPLEX,.40,(240,240,240),1,cv2.LINE_AA)
    out=BASE/"reference_unmarked_lower_crop_2x.png"
    if out.exists():
        raise FileExistsError(out)
    if not cv2.imwrite(str(out),canvas):
        raise RuntimeError("Review image write failed")
    print("Eight independent source-intensity centroids verified; unmarked source-only 2x sheet:",out)


if __name__=="__main__":
    main()
