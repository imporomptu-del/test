#!/bin/bash
# Isolated fixed diagnostic only; no production, clock or previous-evidence writes.
set -eu
if [[ -z "${TMUX:-}" ]]; then
    printf '%s\n' 'Use the dedicated larger-shifts tmux session.' >&2
    exit 1
fi
aot_shifts_workspace=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! "$aot_shifts_workspace" =~ ^/tmp/seaqr_aot_large_shifts_20260928_[A-Za-z0-9]{6}$ ]]; then
    printf '%s\n' 'Unexpected larger-shifts workspace.' >&2
    exit 1
fi
for name in result.json result.json.gz compression_receipt.json failure.json diagnostic.log; do
    if [[ -e "$aot_shifts_workspace/$name" || -L "$aot_shifts_workspace/$name" ]]; then
        printf '%s\n' 'Existing diagnostic evidence; refusing to overwrite.' >&2
        exit 1
    fi
done
for name in manifest.json plan.md diagnose_aot_large_shifts.py test_aot_large_shifts.py diagnose_aot_features.py diagnose_aot_feature_factors.py; do
    if [[ ! -f "$aot_shifts_workspace/$name" || -L "$aot_shifts_workspace/$name" ]]; then
        printf '%s\n' 'Required diagnostic inputs are missing or redirected.' >&2
        exit 1
    fi
done
set -o noclobber
exec /usr/bin/python3 -I -u "$aot_shifts_workspace/diagnose_aot_large_shifts.py" \
    --workspace "$aot_shifts_workspace" > "$aot_shifts_workspace/diagnostic.log" 2>&1
