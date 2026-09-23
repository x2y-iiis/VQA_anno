#!/usr/bin/env bash
set -euo pipefail
set +x

if [ "$#" -lt 1 ]; then
  echo "usage: $0 OUTPUT_DIR [ANNOTATOR_ARGS...]" >&2
  exit 64
fi

OUTPUT_DIR=$1
shift
API_PROVIDER="${API_PROVIDER:-dashscope}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-/root/miniconda3/envs/sam3/bin/python}"
REQUEST_WORKERS="${REQUEST_WORKERS:-64}"

[ -x "$ANNOTATION_PYTHON" ] || {
  echo "python_not_found:$ANNOTATION_PYTHON" >&2
  exit 67
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ARGS=(
  --api "$API_PROVIDER"
  --api-key-env "$API_KEY_ENV"
  --output "$OUTPUT_DIR"
  --workers "$REQUEST_WORKERS"
)
if [ -n "${API_ENDPOINT:-}" ]; then
  ARGS+=(--endpoint "$API_ENDPOINT")
fi

exec "$ANNOTATION_PYTHON" scripts/annotate_videos.py "${ARGS[@]}" "$@"
