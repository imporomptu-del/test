#!/bin/bash
# Approved isolated feature experiment only; no installs, clocks or baseline writes.
set -eu
if [[ -z "${TMUX:-}" ]]; then
    printf '%s\n' 'Use the dedicated feature-factors tmux session.' >&2
    exit 1
fi
aot_factors_workspace=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! "$aot_factors_workspace" =~ ^/tmp/seaqr_aot_factors_20260927_[A-Za-z0-9]{6}$ ]]; then
    printf '%s\n' 'Unexpected factor workspace.' >&2
    exit 1
fi
for name in result.json failure.json diagnostic.log; do
    if [[ -e "$aot_factors_workspace/$name" || -L "$aot_factors_workspace/$name" ]]; then
        printf '%s\n' 'Existing factor evidence; refusing to overwrite.' >&2
        exit 1
    fi
done
for name in manifest.json plan.md diagnose_aot_feature_factors.py test_aot_feature_factors.py diagnose_aot_features.py reference_diagnostic_result.json; do
    if [[ ! -f "$aot_factors_workspace/$name" || -L "$aot_factors_workspace/$name" ]]; then
        printf '%s\n' 'Required factor inputs are missing or redirected.' >&2
        exit 1
    fi
done
set -o noclobber
exec /usr/bin/python3 -I -u "$aot_factors_workspace/diagnose_aot_feature_factors.py" \
    --workspace "$aot_factors_workspace" > "$aot_factors_workspace/diagnostic.log" 2>&1
