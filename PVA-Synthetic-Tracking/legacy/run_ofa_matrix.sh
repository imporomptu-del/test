#!/usr/bin/env bash
# Run the Study 08 vs docs-style OFA matrix, then the SCRUM-75 shared-frame
# OFA vs PVA PyrLK stage, in one shot.
#
#   ./run_ofa_matrix.sh
#   ./run_ofa_matrix.sh --runs 20 --warmup 5
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=20
WARMUP=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--runs N] [--warmup N]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# shellcheck source=/dev/null
source ~/optical-flow/bin/activate

command -v flock >/dev/null 2>&1 || {
  echo "flock is required for exclusive camera benchmarking" >&2
  exit 3
}
CAMERA_LOCK="/tmp/skymove_camera_benchmark.lock"
exec 9>>"$CAMERA_LOCK"
if ! flock -n 9; then
  echo "Camera benchmark lock is held; refusing to continue." >&2
  python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" --check-camera-ownership || true
  exit 2
fi
python3 "$SCRIPT_DIR/bench_pyrlk_pva.py" --check-camera-ownership

RESULTS="$SCRIPT_DIR/results/ofa"
mkdir -p "$RESULTS"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS/run_log_${TS}.txt"
DIRS_FILE="$RESULTS/run_dirs_${TS}.txt"
for candidate in \
  "$LOG" \
  "$DIRS_FILE" \
  "$RESULTS/ofa_cam_run_1920x1080_g4_${TS}" \
  "$RESULTS/ofa_cam_run_1920x1080_g2_${TS}" \
  "$RESULTS/ofa_cam_run_1920x1080_g1_${TS}" \
  "$RESULTS/ofa_cam_run_3184x2124_g4_${TS}" \
  "$RESULTS/ofa_cam_run_3184x2124_g1_${TS}" \
  "$RESULTS/ofa_cam_run_3184x2124_g2_${TS}"
do
  if [[ -e "$candidate" ]]; then
    echo "Refusing to overwrite existing matrix artifact: $candidate" >&2
    exit 2
  fi
done
: >"$DIRS_FILE"

run_one() {
  local w=$1 h=$2 g=$3 tag=$4
  local out="$RESULTS/ofa_cam_run_${tag}_${TS}"
  echo "=== RUN ${tag} ===" | tee -a "$LOG"
  python3 bench_ofa_pair.py \
    --width "$w" --height "$h" --gridsize "$g" \
    --runs "$RUNS" --warmup "$WARMUP" \
    -o "$out" 2>&1 | tee -a "$LOG"
  echo "$out" >>"$DIRS_FILE"
}

echo "OFA matrix batch: $TS  runs=$RUNS warmup=$WARMUP" | tee "$LOG"

run_one 1920 1080 4 1920x1080_g4
run_one 1920 1080 2 1920x1080_g2
run_one 1920 1080 1 1920x1080_g1
run_one 3184 2124 4 3184x2124_g4
run_one 3184 2124 1 3184x2124_g1
run_one 3184 2124 2 3184x2124_g2

LEGACY_SUMMARY="$RESULTS/combined_ofa_summary.md"
if [[ -e "$LEGACY_SUMMARY" ]]; then
  echo "Preserving existing legacy summary: $LEGACY_SUMMARY" | tee -a "$LOG"
else
  python3 "$SCRIPT_DIR/run_ofa_matrix_summary.py" \
    --batch "$TS" \
    --dirs-file "$DIRS_FILE" \
    --runs "$RUNS" \
    --warmup "$WARMUP"
fi

# SCRUM-75: run the shared-frame OFA vs PVA PyrLK stage automatically.
"$SCRIPT_DIR/run_pyrlk_pva_matrix.sh" \
  --camera \
  --camera-lock-held \
  --runs "$RUNS" \
  --warmup "$WARMUP"

echo "ALL_DONE batch=$TS" | tee -a "$LOG"
echo "Legacy OFA summary: $LEGACY_SUMMARY"
