#!/usr/bin/env bash
set -euo pipefail

# Packages
sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip ffmpeg \
  gstreamer1.0-tools gstreamer1.0-plugins-{base,good,bad,ugly} gstreamer1.0-libav

# Python env
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install -U pip wheel
if command -v nvidia-smi >/dev/null; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
pip install ultralytics opencv-python-headless pandas gdown

# Code + data
[[ -d ~/seaqr ]] || git clone https://github.com/imporomptu-del/test.git ~/seaqr
cd ~/seaqr
mkdir -p data/in
gdown --folder https://drive.google.com/drive/folders/1anbWJ_m-hPW4R2nZxuzxcyf2aDty5eUx -O data/in || true

# Build 29-image eval set
mkdir -p data/eval
if [[ -f configs/suspect_images.txt ]]; then
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    hit=$(find data/in -type f -name "$(basename "$f")" -print -quit)
    [[ -n "$hit" ]] && cp -n "$hit" data/eval/
  done < configs/suspect_images.txt
else
  find data/in -type f \( -iname '*.jpg' -o -iname '*.png' \) | head -n 29 | xargs -I{} cp -n "{}" data/eval/
fi
echo "Eval count: $(ls -1 data/eval | wc -l)"

# Pick model (prefer trained best.pt)
MODEL_WEIGHTS=$(ls -1dt runs/*/weights/best.pt 2>/dev/null | head -n1 || true)
[[ -z "$MODEL_WEIGHTS" ]] && MODEL_WEIGHTS=yolo11m.pt
IMGSZ=1280
OUT="runs/eval_$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"

for conf in 0.25 0.35 0.50; do
  for iou in 0.45 0.50 0.60; do
    for tta in off on; do
      name="conf${conf}_iou${iou}_tta${tta}"
      aug=""; [[ "$tta" == "on" ]] && aug="--augment"
      yolo predict model="$MODEL_WEIGHTS" source="data/eval" imgsz="$IMGSZ" \
        conf="$conf" iou="$iou" $aug name="$name" project="$OUT" save_txt save_conf
    done
  done
done

# Package + upload
TS=$(date +%Y%m%d-%H%M%S); TAR="$OUT-$TS.tgz"
tar czf "$TAR" "$OUT"
BUCKET=gs://seaqr-ml-seaqr-detection-123-us-central1
(gcloud storage cp "$TAR" "$BUCKET/runs/eval/" || gsutil cp "$TAR" "$BUCKET/runs/eval/") \
  && echo "Uploaded -> $BUCKET/runs/eval/$(basename "$TAR")"
