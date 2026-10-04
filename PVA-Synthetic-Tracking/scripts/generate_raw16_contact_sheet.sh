#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_DIRECTORY OUTPUT_DIRECTORY CLIP_INDEX" >&2
  exit 2
fi

source_directory=$1
output_directory=$2
clip_index=$3

if [[ ! $clip_index =~ ^[0-9]{4}$ ]]; then
  echo "clip index must contain exactly four digits: $clip_index" >&2
  exit 2
fi

input_path="${source_directory}/chunk_${clip_index}.mkv"
final_directory="${output_directory}/chunk_${clip_index}"
work_directory="${output_directory}/.chunk_${clip_index}.work"

if [[ -f "${final_directory}/.complete" ]]; then
  exit 0
fi
if [[ ! -f $input_path ]]; then
  echo "missing input: $input_path" >&2
  exit 1
fi
if [[ -e $final_directory || -e $work_directory ]]; then
  echo "refusing to overwrite partial preview for chunk $clip_index" >&2
  exit 1
fi

mkdir -p "$output_directory" "$work_directory"

ffmpeg \
  -nostdin \
  -v error \
  -i "$input_path" \
  -vf "select=not(mod(n\\,15)),scale=600:-2:flags=fast_bilinear,tile=4x4:nb_frames=16:padding=2:margin=2" \
  -vsync vfr \
  -q:v 2 \
  "${work_directory}/sheet_%02d.jpg"

mv "$work_directory" "$final_directory"
touch "${final_directory}/.complete"
