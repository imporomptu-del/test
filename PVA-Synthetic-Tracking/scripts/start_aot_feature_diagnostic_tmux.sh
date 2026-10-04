#!/bin/bash
# Isolated diagnostic only: no installation, sudo, clock or baseline writes.
set -eu
if [[ -z "${TMUX:-}" ]]; then
    printf '%s\n' 'Use the dedicated feature-diagnostic tmux session.' >&2
    exit 1
fi
aot_feature_workspace=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! "$aot_feature_workspace" =~ ^/tmp/seaqr_aot_features_20260927_[A-Za-z0-9]{6}$ ]]; then
    printf '%s\n' 'Unexpected diagnostic workspace.' >&2
    exit 1
fi
for name in result.json failure.json diagnostic.log; do
    if [[ -e "$aot_feature_workspace/$name" ]]; then
        printf '%s\n' 'Existing diagnostic evidence; refusing to overwrite.' >&2
        exit 1
    fi
done
for name in manifest.json plan.md diagnose_aot_features.py test_aot_feature_diagnostic.py; do
    if [[ ! -f "$aot_feature_workspace/$name" || -L "$aot_feature_workspace/$name" ]]; then
        printf '%s\n' 'Required diagnostic inputs are missing or redirected.' >&2
        exit 1
    fi
done
set -o noclobber
exec /usr/bin/python3 -I -u "$aot_feature_workspace/diagnose_aot_features.py" \
    --workspace "$aot_feature_workspace" > "$aot_feature_workspace/diagnostic.log" 2>&1
