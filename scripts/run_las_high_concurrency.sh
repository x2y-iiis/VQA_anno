#!/usr/bin/env bash
set -euo pipefail
set +x

if [ "$#" -lt 3 ]; then
  echo "usage: $0 INPUT OUTPUT TASKS [ANNOTATOR_ARGS...]" >&2
  exit 64
fi

INPUT=$1
OUTPUT=$2
TASKS=$3
shift 3

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${ANNOTATION_PYTHON:-python3}

HTTP_CONCURRENCY=${HTTP_CONCURRENCY:-2048}
FRAME_WORKERS=${FRAME_WORKERS:-512}
VIDEO_WORKERS=${VIDEO_WORKERS:-64}
LAS_REQUEST_WORKERS=${LAS_REQUEST_WORKERS:-2048}
LAS_OPERATOR_CONCURRENCY=${LAS_OPERATOR_CONCURRENCY:-2048}
MEDIA_WORKERS=${MEDIA_WORKERS:-32}
CLIP_WORKERS=${CLIP_WORKERS:-64}
BATCH_WORKERS=${BATCH_WORKERS:-16}
MAX_PENDING=${MAX_PENDING:-4096}
RESUME_VALIDATION_WORKERS=${RESUME_VALIDATION_WORKERS:-16}
DURABLE_WRITE_WORKERS=${DURABLE_WRITE_WORKERS:-128}
REQUEST_MEMORY_MIB=${REQUEST_MEMORY_MIB:-49152}
THREAD_STACK_KIB=${THREAD_STACK_KIB:-256}
REQUEST_START_INTERVAL=${REQUEST_START_INTERVAL:-0.005}
FIXED_HTTP_CONCURRENCY=${FIXED_HTTP_CONCURRENCY:-1}

for value in \
  "$HTTP_CONCURRENCY" "$FRAME_WORKERS" "$VIDEO_WORKERS" \
  "$LAS_REQUEST_WORKERS" "$LAS_OPERATOR_CONCURRENCY" "$MEDIA_WORKERS" \
  "$CLIP_WORKERS" "$BATCH_WORKERS" "$MAX_PENDING" \
  "$RESUME_VALIDATION_WORKERS" "$DURABLE_WRITE_WORKERS" \
  "$REQUEST_MEMORY_MIB" "$THREAD_STACK_KIB"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "invalid_nonnegative_integer:$value" >&2
    exit 65
  }
done

case "$FIXED_HTTP_CONCURRENCY" in
  0) fixed_args=() ;;
  1) fixed_args=(--fixed-http-concurrency) ;;
  *) echo "FIXED_HTTP_CONCURRENCY must be 0 or 1" >&2; exit 65 ;;
esac

exec "$PYTHON_BIN" -u "$ROOT/scripts/annotate_videos.py" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --tasks "$TASKS" \
  --api las \
  --api-key-env LAS_API_KEY \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit \
  --request-parallel \
  --model-specific-admission \
  --workers "$VIDEO_WORKERS" \
  --max-record-active "$VIDEO_WORKERS" \
  --shared-frame-workers "$FRAME_WORKERS" \
  --thread-stack-kib "$THREAD_STACK_KIB" \
  --max-pending "$MAX_PENDING" \
  --las-request-workers "$LAS_REQUEST_WORKERS" \
  --max-las-operators "$LAS_OPERATOR_CONCURRENCY" \
  --max-total-http-active "$HTTP_CONCURRENCY" \
  --max-http-active "$HTTP_CONCURRENCY" \
  --request-working-memory-mib "$REQUEST_MEMORY_MIB" \
  --las-media-workers "$MEDIA_WORKERS" \
  --video-prepare-workers "$MEDIA_WORKERS" \
  --video-clip-workers "$CLIP_WORKERS" \
  --batch-workers "$BATCH_WORKERS" \
  --resume-validation-workers "$RESUME_VALIDATION_WORKERS" \
  --durable-write-workers "$DURABLE_WRITE_WORKERS" \
  --ecot-workers "$FRAME_WORKERS" \
  --grd-workers "$FRAME_WORKERS" \
  --sta-bbox-workers "$FRAME_WORKERS" \
  --error-burst-threshold 0 \
  --request-start-interval "$REQUEST_START_INTERVAL" \
  --immutable-jsonl \
  "${fixed_args[@]}" \
  "$@"
