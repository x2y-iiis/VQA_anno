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
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-python3}"
REQUEST_WORKERS="${REQUEST_WORKERS:-64}"

case "$API_PROVIDER" in
  las)
    API_KEY_ENV="${API_KEY_ENV:-LAS_API_KEY}"
    API_ENDPOINT="${API_ENDPOINT:-https://operator.las.cn-beijing.volces.com/api/v1/submit}"
    ;;
  ark)
    API_KEY_ENV="${API_KEY_ENV:-ARK_API_KEY}"
    API_ENDPOINT="${API_ENDPOINT:-https://ark.cn-beijing.volces.com/api/v3/chat/completions}"
    ;;
  dashscope)
    API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
    API_ENDPOINT="${API_ENDPOINT:-https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions}"
    ;;
  *)
    API_KEY_ENV="${API_KEY_ENV:-API_KEY}"
    ;;
esac

if [[ "$ANNOTATION_PYTHON" == */* ]]; then
  [ -x "$ANNOTATION_PYTHON" ] || {
    echo "python_not_found:$ANNOTATION_PYTHON" >&2
    exit 67
  }
elif ! command -v "$ANNOTATION_PYTHON" >/dev/null 2>&1; then
  echo "python_not_found:$ANNOTATION_PYTHON" >&2
  exit 67
fi

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
