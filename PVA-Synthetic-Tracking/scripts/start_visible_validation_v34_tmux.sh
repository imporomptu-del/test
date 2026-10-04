#!/bin/bash
# Disconnect-safe full validation; authentication remains in the terminal.
set -eu

if [[ -z "${TMUX:-}" ]]; then
    printf '%s\n' 'Refusing to start outside tmux. Attach to the prepared seaqr-v34 session.' >&2
    exit 1
fi

launch_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ! -f "$launch_dir/freeze.json" || ! -f "$launch_dir/visible_validation_v34.py" ]]; then
    printf '%s\n' 'The prepared benchmark files are missing.' >&2
    exit 1
fi
if [[ -e "$launch_dir/run" ]]; then
    printf '%s\n' 'This workspace already has run evidence; refusing to overwrite or restart it.' >&2
    exit 1
fi

printf '%s\n' \
    'SEAQR v34: disconnect-safe benchmark session' \
    "Workspace: $launch_dir" \
    'The benchmark is NOT running yet.' \
    'Press Enter below, then enter your sudo password if prompted.' \
    'First: six guarded clock transitions without video; then 2 correctness smokes, 12 full-video runs, then 2 separate profiles.' \
    'After Starting smoke_0126 appears, detach with Ctrl+B, release, then D.' \
    'Detaching leaves the benchmark and its restoration safeguards running on Jetson.' \
    'Do not use Ctrl+C unless you intend to stop the benchmark.' \
    'The finished pane will remain available for review.'
read -r -p 'Press Enter to start: ' acknowledgment

# Keep the authentication prompt on the tmux pseudo-terminal. Never capture
# the password or pass it in argv, environment, logs, or files.
exec /usr/bin/sudo /usr/bin/python3 -I "$launch_dir/visible_validation_v34.py" run


