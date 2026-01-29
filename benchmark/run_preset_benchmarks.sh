#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SHAPES_FILE="${SCRIPT_DIR}/attn_shapes_unifolm_infer.txt"

if [[ ! -f "${SHAPES_FILE}" ]]; then
  echo "Shapes file not found: ${SHAPES_FILE}" >&2
  exit 1
fi

SHAPES="$(grep -v '^[[:space:]]*$' "${SHAPES_FILE}" | paste -sd ';' -)"

cd "${REPO_ROOT}"

PYTHONPATH=./python python benchmark/bench_attention.py \
  --shapes "${SHAPES}" \
  --dtype fp16 \
  --check-backends \
  "$@"
