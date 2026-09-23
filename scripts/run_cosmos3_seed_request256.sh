#!/usr/bin/env bash
# Full-scope resume with shared request admission and independent LAS scheduling.
set -euo pipefail
set +x
PROJECT_ROOT=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-las-seed-full-20260906}
export OUTPUT_ROOT
REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-256}
LAS_REQUEST_WORKERS=${LAS_REQUEST_WORKERS:-256}
MAX_LAS_OPERATORS=${MAX_LAS_OPERATORS:-64}
VIDEO_WORKERS=${VIDEO_WORKERS:-16}
FRAME_WORKERS=${FRAME_WORKERS:-16}
SHARED_FRAME_WORKERS=${SHARED_FRAME_WORKERS:-0}
STA_EVENT_WORKERS=${STA_EVENT_WORKERS:-1}
DOWNSTREAM_STAGE_WORKERS=${DOWNSTREAM_STAGE_WORKERS:-1}
MEDIA_WORKERS=${MEDIA_WORKERS:-4}
CLIP_WORKERS=${CLIP_WORKERS:-8}
CLIP_PROCESS_POOL=${CLIP_PROCESS_POOL:-0}
BATCH_WORKERS=${BATCH_WORKERS:-2}
MAX_PENDING=${MAX_PENDING:-1024}
RESUME_VALIDATION_WORKERS=${RESUME_VALIDATION_WORKERS:-0}
THREAD_STACK_KIB=${THREAD_STACK_KIB:-0}
REQUEST_START_INTERVAL=${REQUEST_START_INTERVAL:-0.02}
FIXED_HTTP_CONCURRENCY=${FIXED_HTTP_CONCURRENCY:-0}
INDEPENDENT_STAGE_PIPELINE=${INDEPENDENT_STAGE_PIPELINE:-0}
REQUEST_MEMORY_MIB=${REQUEST_MEMORY_MIB:-12288}
IMMUTABLE_JSONL=${IMMUTABLE_JSONL:-0}
DURABLE_WRITE_WORKERS=${DURABLE_WRITE_WORKERS:-8}
clip_execution_args=()
case "$CLIP_PROCESS_POOL" in
  0) ;;
  1) clip_execution_args+=(--video-clip-process-pool) ;;
  *) echo 'CLIP_PROCESS_POOL must be 0 or 1' >&2; exit 2 ;;
esac
case "$IMMUTABLE_JSONL" in
  0) ;;
  1) clip_execution_args+=(--immutable-jsonl) ;;
  *) echo 'IMMUTABLE_JSONL must be 0 or 1' >&2; exit 2 ;;
esac
if [[ "$RESUME_VALIDATION_WORKERS" -lt 0 || "$RESUME_VALIDATION_WORKERS" -gt 32 ]]; then
  echo 'RESUME_VALIDATION_WORKERS must be between 0 and 32' >&2
  exit 2
fi
if [[ "$RESUME_VALIDATION_WORKERS" -gt 0 ]]; then
  clip_execution_args+=(--resume-validation-workers "$RESUME_VALIDATION_WORKERS")
fi
if [[ "$FIXED_HTTP_CONCURRENCY" == 1 ]]; then
  clip_execution_args+=(--fixed-http-concurrency)
elif [[ "$FIXED_HTTP_CONCURRENCY" != 0 ]]; then
  echo 'FIXED_HTTP_CONCURRENCY must be 0 or 1' >&2
  exit 2
fi
if [[ "$INDEPENDENT_STAGE_PIPELINE" == 1 ]]; then
  clip_execution_args+=(--independent-stage-pipeline --follow-subtask-path)
elif [[ "$INDEPENDENT_STAGE_PIPELINE" != 0 ]]; then
  echo 'INDEPENDENT_STAGE_PIPELINE must be 0 or 1' >&2
  exit 2
fi

# Preserve all previous checkpoints and fatal markers. This profile uses fresh
# provider rate-state files and does not inherit the old sticky 2/4 admission cap.
# Provider 429 cooldown/adaptive admission and durable fatal stops remain active.
CONTROL_ROOT=${VQA_SUPERVISOR_TMP_ROOT:-$OUTPUT_ROOT/_state}
exec bash "$PROJECT_ROOT/scripts/run_cosmos3_full_seed_1024.sh" \
  --request-parallel --model-specific-admission \
  --workers "$VIDEO_WORKERS" --max-record-active "$VIDEO_WORKERS" \
  --thread-stack-kib "$THREAD_STACK_KIB" \
  --shared-frame-workers "$SHARED_FRAME_WORKERS" \
  --max-pending "$MAX_PENDING" --las-request-workers "$LAS_REQUEST_WORKERS" \
  --max-las-operators "$MAX_LAS_OPERATORS" \
  --max-total-http-active "$REQUEST_CONCURRENCY" --max-http-active "$REQUEST_CONCURRENCY" \
  --request-working-memory-mib "$REQUEST_MEMORY_MIB" \
  --las-media-workers "$MEDIA_WORKERS" --video-prepare-workers "$MEDIA_WORKERS" \
  --video-clip-workers "$CLIP_WORKERS" --batch-workers "$BATCH_WORKERS" \
  "${clip_execution_args[@]}" \
  --durable-write-workers "$DURABLE_WRITE_WORKERS" \
  --ecot-workers "$FRAME_WORKERS" --grd-workers "$FRAME_WORKERS" --sta-bbox-workers "$FRAME_WORKERS" \
  --sta-event-workers "$STA_EVENT_WORKERS" \
  --downstream-stage-workers "$DOWNSTREAM_STAGE_WORKERS" \
  --error-burst-threshold 0 --request-start-interval "$REQUEST_START_INTERVAL" \
  --rate-state-file "$CONTROL_ROOT/request256-rate-state.json" "$@"
