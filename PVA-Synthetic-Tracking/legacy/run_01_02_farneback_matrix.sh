#!/usr/bin/env bash
# Run Farneback CPU + nvof + Farneback GPU matrix (2 resolutions) and build combined table.
#
#   ./run_01_02_farneback_matrix.sh
#   ./run_01_02_farneback_matrix.sh --runs 20 --warmup 3
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=20
WARMUP=3
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

RESULTS="$SCRIPT_DIR/results/ofa"
mkdir -p "$RESULTS"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$RESULTS/run_01_02_log_${TS}.txt"
MANIFEST="$RESULTS/run_01_02_dirs_${TS}.json"
: >"$LOG"

run_01_02() {
  local w=$1 h=$2 tag=$3
  local out="$RESULTS/flow_01_02_run_${tag}_${TS}"
  echo "=== bench_01_02_flow ${tag} ===" | tee -a "$LOG" >&2
  python3 bench_01_02_flow.py \
    --width "$w" --height "$h" \
    --flow-downsample 1.0 \
    --runs "$RUNS" --warmup "$WARMUP" \
    -o "$out" --verbose 2>&1 | tee -a "$LOG" >&2
  echo "$out"
}

run_gpu() {
  local w=$1 h=$2 tag=$3
  local out="$RESULTS/farneback_gpu_run_${tag}_${TS}"
  echo "=== bench_01_farneback_gpu ${tag} ===" | tee -a "$LOG" >&2
  python3 bench_01_farneback_gpu.py \
    --width "$w" --height "$h" \
    --flow-downsample 1.0 \
    --runs "$RUNS" --warmup "$WARMUP" \
    -o "$out" --verbose 2>&1 | tee -a "$LOG" >&2
  echo "$out"
}

echo "01/02 + Farneback GPU batch: $TS  runs=$RUNS warmup=$WARMUP" | tee "$LOG"

OUT_1080_01=$(run_01_02 1920 1080 1920x1080)
OUT_FULL_01=$(run_01_02 3184 2124 3184x2124)
OUT_1080_GPU=$(run_gpu 1920 1080 1920x1080)
OUT_FULL_GPU=$(run_gpu 3184 2124 3184x2124)

python3 "$SCRIPT_DIR/run_01_02_farneback_summary.py" \
  --batch "$TS" \
  --runs "$RUNS" \
  --warmup "$WARMUP" \
  --flow-01-02-1080 "$OUT_1080_01" \
  --flow-01-02-full "$OUT_FULL_01" \
  --farneback-gpu-1080 "$OUT_1080_GPU" \
  --farneback-gpu-full "$OUT_FULL_GPU" \
  --manifest "$MANIFEST"

echo "ALL_DONE batch=$TS" | tee -a "$LOG"
echo "Combined summary: $RESULTS/combined_01_02_report/combined_report.md"
