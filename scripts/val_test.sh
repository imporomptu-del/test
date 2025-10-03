#!/usr/bin/env bash
set -euo pipefail
# REQUIRED env:
#   RUN_ID         e.g., yolo_run_20250919_212059 or test_eval_$(date +%F)
#   WEIGHTS        local path or gs://.../best.pt
#   DATASET_YAML   e.g., /data/dataset/data.yaml (on the VM) or a local path
# OPTIONAL:
#   IMG_SIZE (1280)  CONF (0.001)  IOU (0.6)  DEVICE (0|cpu)

RUN_ID="${RUN_ID:?set RUN_ID}"; WEIGHTS="${WEIGHTS:?set WEIGHTS}"; DATASET_YAML="${DATASET_YAML:?set DATASET_YAML}"
IMG_SIZE="${IMG_SIZE:-1280}"; CONF="${CONF:-0.001}"; IOU="${IOU:-0.6}"; DEVICE="${DEVICE:-0}"
OUTDIR="outputs/${RUN_ID}"; mkdir -p "${OUTDIR}"

# If WEIGHTS is a GCS path, copy local for stability
if [[ "${WEIGHTS}" == gs://* ]]; then
  gsutil cp "${WEIGHTS}" "${OUTDIR}/best.pt"
  WEIGHTS="${OUTDIR}/best.pt"
fi

# Run Ultralytics (>=8.3). Use python -m to avoid PATH issues with 'yolo'
python -m ultralytics val \
  model="${WEIGHTS}" data="${DATASET_YAML}" split=test \
  imgsz="${IMG_SIZE}" conf="${CONF}" iou="${IOU}" device="${DEVICE}" \
  project="${OUTDIR}" name="val_test" plots=True save_json=True

# Summarize to metrics.json
python - <<'PY'
import csv, json, pathlib, sys, datetime
out = pathlib.Path(sys.argv[1])/'val_test'
row = list(csv.DictReader((out/'results.csv').open()))[-1]
def f(x): 
  try: return float(x)
  except: return None
metrics = {"timestamp": datetime.datetime.utcnow().isoformat()+"Z",
           "mAP50": f(row.get("metrics/mAP50")),
           "mAP50-95": f(row.get("metrics/mAP50-95")),
           "precision": f(row.get("metrics/precision")),
           "recall": f(row.get("metrics/recall")),
           "raw": row}
(out.parent/'metrics.json').write_text(json.dumps(metrics, indent=2))
print("Wrote", out.parent/'metrics.json')
PY
echo "Done → ${OUTDIR}/metrics.json and ${OUTDIR}/val_test/plots/"
