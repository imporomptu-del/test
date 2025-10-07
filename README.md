# Noosphera Whale & Dolphin Detection — Training & Eval Harness

Lightweight repo for **training** and **notebook-free evaluation** of Ultralytics YOLO models (v8/v11 family) to detect whales/dolphins and related marine objects.
Heavy data, `runs/`, videos, and model blobs are **kept out of Git** (see `.gitignore`). This repo holds scripts, tiny samples, configs, quick docs, and small reviewable artifacts.

---

## Repo layout

```
artifacts/
  pr2/
    from_eval29_*/           # shareable overlays (20 each) + manifest & q75 JSON (used in PR #2)
configs/
  data.local.example.yaml    # template for absolute paths on your machine (ignored by Git once copied)
  data.smoke.yaml            # tiny smoke config for CI / local checks
data/
  eval/                      # 29-image eval set (built by scripts)
  in/val_smoke*/             # tiny smoke images
docs/
  EVAL_QUICKSTART.md         # one-page eval instructions
scripts/
  val_test.sh                # val/eval → metrics.json + plots (no notebooks needed)
  publish.sh                 # rsync an outputs/ run to a GCS path
  filter_by_class_thresholds.py  # post-filter YOLO txt by per-class conf
scripts_from_VM/
  kolomverse_dataset_pipeline.py # VM-side helpers
  relabel_whale_dolphin.py.py    # (utility from VM)
  train_yolo.sh
setup_eval.sh                # one-shot: deps + 29-image sweep + optional upload
requirements.txt             # pinned, CPU/GPU-friendly
*.ipynb                      # ad-hoc exploration; NOT required for CI/eval
```

---

## Quickstart (CPU-friendly)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.txt
```

> On a GPU VM, install CUDA wheels matching your driver/CUDA (see `setup_eval.sh`).

---

## Local dataset config

Create a **local** YAML (ignored by Git) from the template:

```bash
cp configs/data.local.example.yaml configs/data.local.yaml
# edit absolute paths under 'path'
```

Example (edit to your dataset & class list; **order matters**):

```yaml
path: /abs/path/to/dataset       # contains images/ and labels/
train: images/train
val:   images/val
test:  images/test
names:                            # example 7-class order
  0: whale
  1: dolphin
  2: boat
  3: buoy
  4: fishnet_buoy
  5: lighthouse
  6: ship
```

**Important:** The `names` index **must match the order used when the model was trained**. A mismatch causes nonsense like “dolphin → bird” or “whale → boat”.

---

## Training (examples)

Minimal:

```bash
# From this repo folder (with your configs/data.local.yaml set):
yolo detect train \
  model=/abs/path/to/base_or_pretrained.pt \
  data=configs/data.local.yaml \
  imgsz=1280 epochs=80 batch=8 \
  project=runs/train name=whales_yolo
```

There’s also an older example at `train_from_kolomverse.sh` (adjust paths before use).

---

## Notebook-free evaluation

### Option A — One-shot 29-image sweep

`setup_eval.sh` will:

1. install minimal deps,
2. assemble `data/eval` from `configs/suspect_images.txt` (or the first 29 images under `data/in`),
3. run a small grid over `conf ∈ {0.25,0.35,0.50}` × `IoU ∈ {0.45,0.50,0.60}` with/without TTA,
4. save labels **with confidences** (`save_conf=True`),
5. optionally tar + upload to GCS.

Run it (or open it and copy the bits you want):

```bash
bash setup_eval.sh
```

Artifacts used in PR #2 live under `artifacts/pr2/from_eval29_*`.

### Option B — Full val on your dataset (plots + metrics.json)

```bash
# REQUIRED:
export RUN_ID="test_eval_$(date +%F)"
export WEIGHTS="/abs/path/to/best.pt"        # or gs://.../best.pt
export DATASET_YAML="$PWD/configs/data.local.yaml"

bash scripts/val_test.sh
# → outputs/${RUN_ID}/metrics.json and outputs/${RUN_ID}/val_test/plots/
```

---

## Ad-hoc predict (debug)

```bash
MODEL=$(ls -dt runs/train/*/weights/best.pt 2>/dev/null | head -n1)
[ -z "$MODEL" ] && echo "Set MODEL=/abs/path/to/your_best.pt" && MODEL=/abs/path/to/your_best.pt

SRC="data/eval"
yolo predict model="$MODEL" source="$SRC" imgsz=1280 conf=0.05 iou=0.5 \
     save_txt=True save_conf=True \
     project=runs/predict_whales name=eval_debug exist_ok=True
```

Sanity checks:

```bash
L=runs/predict_whales/eval_debug/labels

# Each detection line should be 6 fields: class x y w h conf
awk '{print NF}' "$L"/*.txt | sort -u    # expect: 6

# Which images had zero detections?
find "$L" -type f -size 0 -printf '%f\n' | sed -n '1,200p'
```

Inspect class names baked into your weights:

```python
from ultralytics import YOLO
print("names:", YOLO("/abs/path/to/best.pt").names)
# If you see COCO names (person,bicycle,car,...) you loaded the wrong weights.
```

---

## Pulling weights from GCS

```bash
gsutil ls -r gs://seaqr-ml-seaqr-detection-123-us-central1/**/weights/best.pt | sed -n '1,50p'
URI="gs://seaqr-ml-seaqr-detection-123-us-central1/runs/2025-09-20/yolo_run_20250919_212059/weights/best.pt"
gsutil cp "$URI" ./weights_best.pt
export MODEL="$PWD/weights_best.pt"
```

---

## Publishing results to GCS

```bash
export RUN_ID="test_eval_$(date +%F)"
export PUBLISH="gs://<your-bucket>/<path>/${RUN_ID}/"
bash scripts/publish.sh
```

---

## Why you might see **no boxes** or **wrong classes**

* **Wrong weights / COCO names:** You ran a generic COCO model. Use your trained `best.pt` and verify `YOLO(...).names`.
* **Class order mismatch:** Ensure the YAML `names` order used **during training** matches your current interpretation.
* **Conf threshold too high:** Re-run with lower `conf` (e.g., `0.05`) and always keep `save_conf=True` so the 6th field is present.

---

## Troubleshooting

* **“No space left on device”**
  Use CPU wheels (see Quickstart), clean caches:
  `rm -rf ~/.cache/pip ~/.cache/torch` and avoid committing `runs/`.
* **`yolo: command not found`**
  Use `python -m ultralytics ...` instead of `yolo`.
* **GPU capacity errors (L4 busy)**
  Try another zone/region or do CPU-only eval; resume GPU training when capacity frees up.

---

## Roadmap / TODO

* ✅ PR #2: notebook-free eval wiring + small reviewable artifacts.
* ⬜ Finalize **GStreamer chunk writer** with `splitmuxsink` (webcam path first; emulator later).
* ⬜ Re-run the model on the shared Drive videos to verify whale/dolphin detections.
* ⬜ Validate class mapping end-to-end (no “dolphin→bird”, “whale→boat” mislabels).
* ⬜ Continue training with proper **background images** included.
* ⬜ Optional: per-class Q75 confidence filter (`scripts/filter_by_class_thresholds.py`).

---

## Contributing

Branch from `master`, keep heavy artifacts out (see `.gitignore`), and open a PR.
Small, visual artifacts for review can go under `artifacts/pr2/...`.
