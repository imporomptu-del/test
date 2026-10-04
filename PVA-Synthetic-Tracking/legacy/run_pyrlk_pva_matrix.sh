#!/usr/bin/env bash
# SCRUM-75: shared-frame OFA vs VPI PVA PyrLK matrix at both resolutions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=20
WARMUP=5
SEED=75
SOURCE_MODE=""
OUTPUT_DIR=""
CAMERA_ID=""
CAMERA_NAME="SkyEye"
RESOLUTION_INDEX=""
HARRIS_STRENGTH="20.0"
CAMERA_LOCK_HELD=0

usage() {
  echo "Usage: $0 (--camera|--synthetic) [options]"
  echo ""
  echo "Options:"
  echo "  --runs N                 Timed pairs per resolution (default: 20)"
  echo "  --warmup N               Warmup pairs per resolution (default: 5)"
  echo "  --seed N                 Synthetic seed (default: 75)"
  echo "  --output-dir PATH        New, non-existing batch directory"
  echo "  --camera-id N            Explicit camera index"
  echo "  --camera-name TEXT       Camera name match (default: SkyEye)"
  echo "  --resolution-index N     Camera SDK resolution index"
  echo "  --harris-strength VALUE  Harris threshold (default: 20.0)"
  echo "  -h, --help               Show this help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --camera)
      [[ -z "$SOURCE_MODE" ]] || { echo "Choose only one source mode" >&2; exit 2; }
      SOURCE_MODE="camera"
      shift
      ;;
    --synthetic)
      [[ -z "$SOURCE_MODE" ]] || { echo "Choose only one source mode" >&2; exit 2; }
      SOURCE_MODE="synthetic"
      shift
      ;;
    --runs)
      [[ $# -ge 2 ]] || { echo "--runs requires a value" >&2; exit 2; }
      RUNS="$2"
      shift 2
      ;;
    --warmup)
      [[ $# -ge 2 ]] || { echo "--warmup requires a value" >&2; exit 2; }
      WARMUP="$2"
      shift 2
      ;;
    --seed)
      [[ $# -ge 2 ]] || { echo "--seed requires a value" >&2; exit 2; }
      SEED="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { echo "--output-dir requires a value" >&2; exit 2; }
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --camera-id)
      [[ $# -ge 2 ]] || { echo "--camera-id requires a value" >&2; exit 2; }
      CAMERA_ID="$2"
      shift 2
      ;;
    --camera-name)
      [[ $# -ge 2 ]] || { echo "--camera-name requires a value" >&2; exit 2; }
      CAMERA_NAME="$2"
      shift 2
      ;;
    --resolution-index)
      [[ $# -ge 2 ]] || { echo "--resolution-index requires a value" >&2; exit 2; }
      RESOLUTION_INDEX="$2"
      shift 2
      ;;
    --harris-strength)
      [[ $# -ge 2 ]] || { echo "--harris-strength requires a value" >&2; exit 2; }
      HARRIS_STRENGTH="$2"
      shift 2
      ;;
    --camera-lock-held)
      CAMERA_LOCK_HELD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$SOURCE_MODE" ]] || { echo "Select exactly one of --camera or --synthetic" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be a positive integer" >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "--warmup must be a non-negative integer" >&2; exit 2; }
[[ "$SEED" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
if [[ "$CAMERA_LOCK_HELD" -eq 1 && "$SOURCE_MODE" != "camera" ]]; then
  echo "--camera-lock-held is valid only with --camera" >&2
  exit 2
fi

VENV_ACTIVATE="${SKYMOVE_VENV_ACTIVATE:-/home/serg/optical-flow/bin/activate}"
if [[ ! -r "$VENV_ACTIVATE" ]]; then
  echo "Required existing environment is unavailable: $VENV_ACTIVATE" >&2
  echo "No dependency installation or environment change was attempted." >&2
  exit 3
fi
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

if [[ "$SOURCE_MODE" == "camera" ]]; then
  command -v flock >/dev/null 2>&1 || {
    echo "flock is required for exclusive camera benchmarking" >&2
    exit 3
  }
  if [[ "$CAMERA_LOCK_HELD" -eq 1 ]]; then
    if ! flock -n 9; then
      echo "Inherited camera benchmark lock is unavailable." >&2
      exit 2
    fi
  else
    CAMERA_LOCK="/tmp/skymove_camera_benchmark.lock"
    exec 9>>"$CAMERA_LOCK"
    if ! flock -n 9; then
      echo "Camera benchmark lock is held; refusing to continue." >&2
      python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" --check-camera-ownership || true
      exit 2
    fi
  fi
  python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" --check-camera-ownership
fi

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="SCRUM75_${TS}_UTC"
if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ "$SOURCE_MODE" == "camera" ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/results/ofa/pyrlk_cam_run_${TS}"
  else
    OUTPUT_DIR="$SCRIPT_DIR/results/ofa/pyrlk_synthetic_run_${TS}"
  fi
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite or reuse existing output: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"

COMMON_ARGS=(
  --runs "$RUNS"
  --warmup "$WARMUP"
  --seed "$SEED"
  --run-id "$RUN_ID"
  --harris-strength "$HARRIS_STRENGTH"
  --pyramid-levels 4
  --ofa-gridsize 4
)
if [[ "$SOURCE_MODE" == "camera" ]]; then
  SOURCE_ARGS=(--camera --camera-ownership-confirmed --camera-name "$CAMERA_NAME")
  if [[ -n "$CAMERA_ID" ]]; then
    SOURCE_ARGS+=(--camera-id "$CAMERA_ID")
  fi
  if [[ -n "$RESOLUTION_INDEX" ]]; then
    SOURCE_ARGS+=(--resolution-index "$RESOLUTION_INDEX")
  fi
else
  SOURCE_ARGS=(--synthetic)
fi

SAMPLES_1080="$OUTPUT_DIR/samples_1920x1080.json"
SAMPLES_FULL="$OUTPUT_DIR/samples_3184x2124.json"

echo "SCRUM-75 PVA PyrLK matrix"
echo "  run_id      : $RUN_ID"
echo "  source      : $SOURCE_MODE"
echo "  runs/warmup : $RUNS/$WARMUP"
echo "  output      : $OUTPUT_DIR"

python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" \
  "${SOURCE_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --width 1920 \
  --height 1080 \
  --method-order pva-first \
  --output-json "$SAMPLES_1080"

python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" \
  "${SOURCE_ARGS[@]}" \
  "${COMMON_ARGS[@]}" \
  --width 3184 \
  --height 2124 \
  --method-order ofa-first \
  --output-json "$SAMPLES_FULL"

python3 "$SCRIPT_DIR/run_pyrlk_pva_summary.py" \
  --input-1080 "$SAMPLES_1080" \
  --input-full "$SAMPLES_FULL" \
  --output-dir "$OUTPUT_DIR"

echo "SCRUM-75 matrix complete"
echo "  JSON: $OUTPUT_DIR/flow_report.json"
echo "  Markdown: $OUTPUT_DIR/flow_report.md"
