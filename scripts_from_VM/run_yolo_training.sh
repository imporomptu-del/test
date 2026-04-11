#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_yolo_training.sh --data /mnt/yolo/data/prepared/<dataset>/data.yaml [options]

Required:
  --data PATH                 Path to prepared dataset data.yaml

Optional:
  --model PATH                Model checkpoint to fine-tune
                              Default: /mnt/yolo/data/yolo11m_11_28_2025.pt
  --project DIR               YOLO project output directory
                              Default: /mnt/yolo/data/runs
  --venv PATH                 Virtualenv activate script
                              Default: /mnt/yolo/home/romanmaksymiuk/yolo-new-env/bin/activate
  --imgsz INT                 Default: 960
  --epochs INT                Default: 15
  --batch INT                 Default: 16
  --workers INT               Default: 8
  --device VALUE              Default: 0
  --freeze INT                Default: 0
  --lr0 FLOAT                 Default: 0.001
  --close-mosaic INT          Default: 5
  --session-name NAME         Optional tmux session name override
  --name NAME                 Optional YOLO run name override
  --resume                    Pass resume=True to YOLO
  --dry-run                   Print resolved configuration and exit
  -h, --help                  Show this help

Example:
  bash gcp_scripts/run_yolo_training.sh \
    --data /mnt/yolo/data/prepared/my_dataset/data.yaml \
    --epochs 20 \
    --batch 16
EOF
}

MODEL="/mnt/yolo/data/yolo11m_11_28_2025.pt"
PROJECT="/mnt/yolo/data/runs"
VENV_ACTIVATE="/mnt/yolo/home/romanmaksymiuk/yolo-new-env/bin/activate"
IMGSZ=960
EPOCHS=15
BATCH=16
WORKERS=8
DEVICE=0
FREEZE=0
LR0=0.001
CLOSE_MOSAIC=5
DATA=""
SESSION_NAME=""
RUN_NAME=""
RESUME=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA="${2:?missing value for --data}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --project)
      PROJECT="${2:?missing value for --project}"
      shift 2
      ;;
    --venv)
      VENV_ACTIVATE="${2:?missing value for --venv}"
      shift 2
      ;;
    --imgsz)
      IMGSZ="${2:?missing value for --imgsz}"
      shift 2
      ;;
    --epochs)
      EPOCHS="${2:?missing value for --epochs}"
      shift 2
      ;;
    --batch)
      BATCH="${2:?missing value for --batch}"
      shift 2
      ;;
    --workers)
      WORKERS="${2:?missing value for --workers}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?missing value for --device}"
      shift 2
      ;;
    --freeze)
      FREEZE="${2:?missing value for --freeze}"
      shift 2
      ;;
    --lr0)
      LR0="${2:?missing value for --lr0}"
      shift 2
      ;;
    --close-mosaic)
      CLOSE_MOSAIC="${2:?missing value for --close-mosaic}"
      shift 2
      ;;
    --session-name)
      SESSION_NAME="${2:?missing value for --session-name}"
      shift 2
      ;;
    --name)
      RUN_NAME="${2:?missing value for --name}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DATA" ]]; then
  echo "--data is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$DATA" ]]; then
  echo "data.yaml not found: $DATA" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model checkpoint not found: $MODEL" >&2
  exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtualenv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi not found; GPU visibility will not be checked." >&2
fi

DATA_DIR="$(cd "$(dirname "$DATA")" && pwd)"
DATASET_DIR="$(cd "$DATA_DIR/.." && pwd)"
DATASET_NAME="$(basename "$DATASET_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="${DATASET_NAME}_train_${TIMESTAMP}"
fi

if [[ -z "$SESSION_NAME" ]]; then
  SESSION_NAME="yolo-${RUN_NAME}"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$PROJECT"
RUN_DIR="$PROJECT/$RUN_NAME"
LOG="$RUN_DIR/train.log"
LAUNCH_CMD_FILE="$RUN_DIR/launch_command.sh"
INNER_SCRIPT="$RUN_DIR/run_inside_tmux.sh"

YOLO_CMD=(
  yolo task=detect mode=train
  "model=$MODEL"
  "data=$DATA"
  "imgsz=$IMGSZ"
  "epochs=$EPOCHS"
  "batch=$BATCH"
  "workers=$WORKERS"
  "device=$DEVICE"
  "freeze=$FREEZE"
  "lr0=$LR0"
  "close_mosaic=$CLOSE_MOSAIC"
  "project=$PROJECT"
  "name=$RUN_NAME"
)

if [[ "$RESUME" -eq 1 ]]; then
  YOLO_CMD+=("resume=True")
fi

echo "Resolved training configuration:"
echo "  data:          $DATA"
echo "  dataset dir:   $DATASET_DIR"
echo "  model:         $MODEL"
echo "  project:       $PROJECT"
echo "  run name:      $RUN_NAME"
echo "  run dir:       $RUN_DIR"
echo "  tmux session:  $SESSION_NAME"
echo "  venv:          $VENV_ACTIVATE"
echo "  imgsz:         $IMGSZ"
echo "  epochs:        $EPOCHS"
echo "  batch:         $BATCH"
echo "  workers:       $WORKERS"
echo "  device:        $DEVICE"
echo "  freeze:        $FREEZE"
echo "  lr0:           $LR0"
echo "  close_mosaic:  $CLOSE_MOSAIC"
echo "  resume:        $RESUME"

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '  command:       '
  printf '%q ' "${YOLO_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$RUN_DIR"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo
  echo "source $(printf '%q' "$VENV_ACTIVATE")"
  echo "mkdir -p $(printf '%q' "$RUN_DIR")"
  echo "echo \"[INFO] Starting YOLO training -> $RUN_DIR\""
  echo "echo \"[INFO] Logging to $LOG\""
  echo "echo \"[INFO] Dataset: $DATA\""
  echo "echo \"[INFO] GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)\""
  printf '%q ' "${YOLO_CMD[@]}"
  echo " 2>&1 | tee $(printf '%q' "$LOG")"
  echo "echo \"[INFO] Finished. Best weights: $RUN_DIR/weights/best.pt\""
} > "$INNER_SCRIPT"

chmod +x "$INNER_SCRIPT"

{
  echo "#!/usr/bin/env bash"
  printf '%q ' "$0"
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
} > "$LAUNCH_CMD_FILE"
chmod +x "$LAUNCH_CMD_FILE"

tmux new-session -d -s "$SESSION_NAME" "bash $(printf '%q' "$INNER_SCRIPT")"

echo
echo "Training started."
echo "  tmux session:  $SESSION_NAME"
echo "  attach:        tmux attach -t $SESSION_NAME"
echo "  log:           $LOG"
echo "  run dir:       $RUN_DIR"
echo "  launch cmd:    $LAUNCH_CMD_FILE"
