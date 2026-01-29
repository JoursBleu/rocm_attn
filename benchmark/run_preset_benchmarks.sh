#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHAPES_LIST=(
  "1,32,1024,128"
  "1,32,2048,128"
  "1,32,4096,128"
  "1,64,2048,128"
  "1,64,4096,128"
  "1,32,2048,256"
  "1,32,128,2048,128"
  "1,32,256,2048,128"
  "1,32,1024,2048,128"
  "1,64,1024,2048,128"
)
SHAPES="$(IFS=';'; echo "${SHAPES_LIST[*]}")"

cd "${REPO_ROOT}"

PYTHONPATH=./python python benchmark/bench_attention.py \
  --shapes "${SHAPES}" \
  --dtype fp16 \
  --check-backends \
  "$@"
