#!/bin/bash
# One approved external pilot. No sudo, clock changes, installs, or source changes.
set -eu
if [[ -z "${TMUX:-}" ]]; then
    printf '%s\n' 'Start inside the dedicated AOT tmux session.' >&2
    exit 1
fi
aot_workspace=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! "$aot_workspace" =~ ^/tmp/seaqr_aot_pilot_20260927_[A-Za-z0-9]{6}$ ]]; then
    printf '%s\n' 'Unexpected workspace.' >&2
    exit 1
fi
if [[ -e "$aot_workspace/run" || -e "$aot_workspace/execution_receipt.json" || -e "$aot_workspace/baseline.log" ]]; then
    printf '%s\n' 'Existing run evidence; refusing to overwrite or restart.' >&2
    exit 1
fi
if [[ ! -f "$aot_workspace/preflight.json" || ! -f "$aot_workspace/scoring_freeze.json" ]]; then
    printf '%s\n' 'Preflight and frozen scoring policy required.' >&2
    exit 1
fi
set -o noclobber
exec /usr/bin/python3 -I -u "$aot_workspace/run_aot_frozen_baseline.py" \
    --workspace "$aot_workspace" --run > "$aot_workspace/baseline.log" 2>&1
