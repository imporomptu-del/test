#!/usr/bin/env bash
set -euo pipefail

# ---- config you can tweak ----
MODEL="/data/dataset/yolo11n.pt"                 # try yolo11m.pt if VRAM allows
DATA="/data/dataset/data.yaml"     # your dataset yaml
IMGSZ=640                          # 1280 for higher accuracy if VRAM fits
EPOCHS=80
BATCH=16                           # lower if CUDA OOM (e.g., 8 or 4)
WORKERS=8
DEVICE=0                           # 0 = first GPU, or "cpu"
PROJECT="/data/runs"
NAME="yolo_run_$(date +%Y%m%d_%H%M%S)"  # unique run name
LOG="/data/runs/${NAME}.log"
# --------------------------------

echo "[INFO] Starting YOLO training → $PROJECT/detect/$NAME"
echo "[INFO] Logging to $LOG"
echo "[INFO] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'nvidia-smi not available')"

# (optional) activate your venv if you use one
source ~/yolo-env/bin/activate

# run training and tee logs
yolo task=detect mode=train \
     model="$MODEL" \
     data="$DATA" \
     imgsz="$IMGSZ" epochs="$EPOCHS" batch="$BATCH" workers="$WORKERS" \
     device="$DEVICE" \
     project="$PROJECT" name="$NAME" 2>&1 | tee "$LOG"

echo "[INFO] Finished. Best weights (if any): $PROJECT/detect/$NAME/weights/best.pt"
