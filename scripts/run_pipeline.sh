#!/usr/bin/env bash
set -euo pipefail
set +x

if [ "$#" -lt 2 ]; then
  echo "usage: $0 INPUT OUTPUT [ANNOTATOR_ARGS...]" >&2
  exit 64
fi

INPUT=$1
OUTPUT=$2
shift 2
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${ANNOTATION_PYTHON:-python3}

exec "$PYTHON_BIN" -u "$ROOT/scripts/annotate_videos.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  "$@"
