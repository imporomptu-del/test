#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${RUN_ID:?set RUN_ID}"
OUTDIR="outputs/${RUN_ID}"
PUBLISH="${PUBLISH:?set PUBLISH}"  # e.g., gs://seaqr-detection-data-roman/test_results/${RUN_ID}/
gsutil -m rsync -r "${OUTDIR}" "${PUBLISH}"
echo "Published to ${PUBLISH}"
