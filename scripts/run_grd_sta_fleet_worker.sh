#!/usr/bin/env bash
set -euo pipefail
set +x

usage() {
  echo 'Usage: run_grd_sta_fleet_worker.sh TASK WORKER_ID UID_PATH OUTPUT_ROOT [RESUME_OUTPUT] [STA_MODEL] [ARK_UPLOAD_WORKERS]' >&2
  exit 2
}

[[ $# -ge 4 && $# -le 7 ]] || usage
TASK=$1
WORKER_ID=$2
UID_PATH=$3
OUTPUT_ROOT=$4
RESUME_OUTPUT=${5:-}
STA_MODEL=${6:-doubao-seed-2-0-lite-260215}
ARK_UPLOAD_WORKERS=${7:-}
[[ "$TASK" == grd || "$TASK" == sta ]] || usage
if [[ -z "$ARK_UPLOAD_WORKERS" ]]; then
  # STA emits several independently reusable media objects per episode.  The
  # COS-to-TOS Fetch runs outside the model request slots.  A 256-wide window
  # leaves GRD model admission starved on the large production fleet, so both
  # task families use the verified 1,024-wide transfer window.
  ARK_UPLOAD_WORKERS=1024
fi
[[ "$ARK_UPLOAD_WORKERS" =~ ^[0-9]+$ && "$ARK_UPLOAD_WORKERS" -ge 1 && "$ARK_UPLOAD_WORKERS" -le 4096 ]] || {
  echo "ark_upload_workers_out_of_range:$ARK_UPLOAD_WORKERS" >&2
  exit 2
}
[[ -s "$UID_PATH" ]] || { echo "record_uid_path_missing:$UID_PATH" >&2; exit 2; }
[[ "$OUTPUT_ROOT" == /mnt/human_data/video_cleaning/* || "$OUTPUT_ROOT" == /mnt/human_data/video-cleaning/* ]] || {
  echo "output_root_outside_production_namespace:$OUTPUT_ROOT" >&2
  exit 2
}
if [[ -n "$RESUME_OUTPUT" ]]; then
  [[ -d "$RESUME_OUTPUT" && "$RESUME_OUTPUT" != "$OUTPUT_ROOT" ]] || {
    echo 'resume_output_must_exist_and_differ_from_output' >&2
    exit 2
  }
fi

PROJECT=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
SUBTASK_PATH=${SUBTASK_PATH:-/run/ti/vqa-subtask-index-20260916}
# TOS credentials must survive a supervised tmux/process restart.  Some early
# fleet sessions inherited them only from the interactive shell, which made a
# safe code rollout enter a restart loop after that shell environment was gone.
# Load the existing protected fleet env file only when the three variables are
# not already present; never place credential values in argv or logs.
TOS_ENV_FILE=${TOS_ENV_FILE:-/mnt/doubao_las_annotation/.runtime/tos.env}
tos_fetch_override_is_set=${TOS_UPLOAD_VIA_COS_FETCH+x}
tos_fetch_override=${TOS_UPLOAD_VIA_COS_FETCH:-}
if [[ -z "${TOS_ACCESS_KEY:-}" || -z "${TOS_SECRET_KEY:-}" || -z "${TOS_BUCKET:-}" ]]; then
  if [[ -s "$TOS_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$TOS_ENV_FILE"
    set +a
  fi
fi
# Sourcing the credential bundle must not silently undo an explicit transport
# canary supplied by the production launcher.  In particular, "0" is a valid
# override and cannot be tested with a simple non-empty/default expression.
if [[ -n "$tos_fetch_override_is_set" ]]; then
  export TOS_UPLOAD_VIA_COS_FETCH="$tos_fetch_override"
fi
[[ -n "${TOS_ACCESS_KEY:-}" && -n "${TOS_SECRET_KEY:-}" && -n "${TOS_BUCKET:-}" ]] || {
  echo "tos_credentials_missing:file=$TOS_ENV_FILE" >&2
  exit 2
}
if [[ -z "${LAS_KEY_FILE:-}" ]]; then
  if [[ ( "$TASK" == sta && "$STA_MODEL" == doubao-seed-2-1-turbo-260628 ) \
      || ( "$TASK" == grd && "${GRD_MODEL:-}" == doubao-seed-2-1-turbo-260628 ) ]] \
      && [[ -s /mnt/doubao_las_annotation/.runtime/las_api_key_turbo ]]; then
    LAS_KEY_FILE=/mnt/doubao_las_annotation/.runtime/las_api_key_turbo
  elif [[ -s /mnt/doubao_las_annotation/.runtime/las_api_key_2_1_pro ]]; then
    LAS_KEY_FILE=/mnt/doubao_las_annotation/.runtime/las_api_key_2_1_pro
  else
    LAS_KEY_FILE=/mnt/doubao_las_annotation/.runtime/las_api_key
  fi
fi
ARK_KEY_FILE=${ARK_KEY_FILE:-/mnt/doubao_las_annotation/.runtime/ark_api_key}
SCRATCH=${VQA_WORKER_SCRATCH:-/run/ti/vqa-grd-sta-fleet/$TASK-$WORKER_ID}
[[ "$SCRATCH" == /run/ti/* ]] || {
  echo "worker_scratch_must_be_local_run_ti:$SCRATCH" >&2
  exit 2
}
# OUTPUT_ROOT is normally a COS FUSE path. Re-running `mkdir -p` against that
# mount on every supervisor restart can block in the kernel for minutes even
# when the directory already exists. Only local scratch is a launch prerequisite;
# the durable writers create their exact remote parents when they publish.
mkdir -p "$SCRATCH/checkpoints" "$SCRATCH/tmp"
[[ -x "$PYTHON_BIN" ]] || { echo "python_missing:$PYTHON_BIN" >&2; exit 2; }
[[ -d "$SUBTASK_PATH" ]] || { echo "subtask_cache_missing:$SUBTASK_PATH" >&2; exit 2; }
SUBTASK_EXPECTED_RECORDS=${SUBTASK_EXPECTED_RECORDS:-319355}
SUBTASK_VALIDATED_MARKER="$SUBTASK_PATH/.complete-$SUBTASK_EXPECTED_RECORDS-records"
[[ "$SUBTASK_EXPECTED_RECORDS" =~ ^[0-9]+$ ]] || {
  echo "subtask_expected_records_invalid:$SUBTASK_EXPECTED_RECORDS" >&2
  exit 2
}
if [[ ! -f "$SUBTASK_VALIDATED_MARKER" ]] \
    || [[ "$(<"$SUBTASK_VALIDATED_MARKER")" != "$SUBTASK_EXPECTED_RECORDS" ]]; then
  subtask_audit_dir=$(mktemp -d "/run/ti/subtask-worker-audit-${WORKER_ID}.XXXXXX")
  /usr/bin/python3 "$PROJECT/scripts/write_ready_subtask_uids.py" \
    "$SUBTASK_PATH" "$subtask_audit_dir/uids.txt" \
    --summary "$subtask_audit_dir/summary.json" >/dev/null
  read -r subtask_valid subtask_records subtask_uids < <(
    /usr/bin/python3 - "$subtask_audit_dir/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    str(bool(summary.get("valid"))).lower(),
    summary.get("records", -1),
    summary.get("selected_uids", -1),
)
PY
  )
  if [[ "$subtask_valid" != true \
      || "$subtask_records" -ne "$SUBTASK_EXPECTED_RECORDS" \
      || "$subtask_uids" -ne "$SUBTASK_EXPECTED_RECORDS" ]]; then
    echo "subtask_cache_incomplete:path=$SUBTASK_PATH:valid=$subtask_valid:records=$subtask_records:selected_uids=$subtask_uids:required=$SUBTASK_EXPECTED_RECORDS" >&2
    rm -rf -- "$subtask_audit_dir"
    exit 2
  fi
  printf '%s\n' "$subtask_uids" > "$SUBTASK_VALIDATED_MARKER.tmp.$$"
  mv "$SUBTASK_VALIDATED_MARKER.tmp.$$" "$SUBTASK_VALIDATED_MARKER"
  rm -rf -- "$subtask_audit_dir"
fi
[[ -s "$LAS_KEY_FILE" ]] || { echo "las_key_missing:$LAS_KEY_FILE" >&2; exit 2; }
[[ -s "$ARK_KEY_FILE" ]] || { echo "ark_key_missing:$ARK_KEY_FILE" >&2; exit 2; }

export LAS_API_KEY="$(tr -d '\r\n' < "$LAS_KEY_FILE")"
export ARK_API_KEY="$(tr -d '\r\n' < "$ARK_KEY_FILE")"
# Some notebook images export OMP_NUM_THREADS=1 globally.  That also makes
# `nproc` report one even when the worker has a large CPU quota.  The pipeline
# explicitly bounds codec and BLAS thread pools, so remove the misleading
# inherited value before launching the scheduler.
unset OMP_NUM_THREADS
export PYTHON_BIN OUTPUT_ROOT
export INPUT_ROOT=${VQA_PHYSICAL_INPUT_ROOT:-/run/ti/cosmos3-physical-catalog}
export ADDRESS_SPACE_GIB=${ADDRESS_SPACE_GIB:-512}
export REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-8192}
export LAS_REQUEST_WORKERS=${LAS_REQUEST_WORKERS:-8192}
export MAX_LAS_OPERATORS=${MAX_LAS_OPERATORS:-8192}
export VIDEO_WORKERS=${VIDEO_WORKERS:-2048}
export FRAME_WORKERS=${FRAME_WORKERS:-4096}
if [[ "$TASK" == sta ]]; then
  # STA used to create one bbox ThreadPoolExecutor per resident episode.  The
  # shared executor now owns this work, so bound it below the 2,048 episode
  # lane to avoid scheduler collapse while retaining high request overlap.
  export SHARED_FRAME_WORKERS=${SHARED_FRAME_WORKERS:-1024}
else
  export SHARED_FRAME_WORKERS=${SHARED_FRAME_WORKERS:-4096}
fi

# More resident GRD records do not increase provider throughput once every
# process has a paced request backlog.  In production, 4,096 record workers
# created 8,000-10,000 native threads per process and Linux load averages above
# 800.  The single pacing dispatcher then went unscheduled for tens of seconds,
# even though permits were already due.  Bound the three thread-producing GRD
# pools at the largest profile that still sustained a full 25 requests/second
# pacing lane.  The request/operator limits remain high, so long-running LAS
# tasks can stay in flight without turning the host scheduler into the limiter.
if [[ "$TASK" == grd ]]; then
  GRD_VIDEO_WORKERS_CAP=${GRD_VIDEO_WORKERS_CAP:-2048}
  GRD_FRAME_WORKERS_CAP=${GRD_FRAME_WORKERS_CAP:-4096}
  GRD_SHARED_FRAME_WORKERS_CAP=${GRD_SHARED_FRAME_WORKERS_CAP:-4096}
  # `free` reports host RAM inside the notebook container.  Use the memory
  # cgroup limit instead: the local production pod has a 120 GiB limit while
  # sharing the pod with four GPU CPA workers.  Its old 2,048-record GRD
  # profile repeatedly reached the cgroup wall despite more than 1 TiB being
  # free on the host.
  memory_cgroup_limit=0
  if [[ -r /sys/fs/cgroup/memory.max ]]; then
    read -r memory_cgroup_value </sys/fs/cgroup/memory.max
    [[ "$memory_cgroup_value" == max ]] || memory_cgroup_limit=$memory_cgroup_value
  elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    read -r memory_cgroup_limit </sys/fs/cgroup/memory/memory.limit_in_bytes
  fi
  if [[ "$memory_cgroup_limit" =~ ^[0-9]+$ ]] \
      && (( memory_cgroup_limit > 0 && memory_cgroup_limit < 171798691840 )); then
    # The 1,024-record profile still reached 98-100 GiB anonymous RSS when it
    # overlapped four resident SAM3 workers and was OOM-killed.  Keep enough
    # records resident to cover the 2.1-pro wait, but bound the thread/future
    # population below the pod's non-reclaimable-memory ceiling.
    (( GRD_VIDEO_WORKERS_CAP > 512 )) && GRD_VIDEO_WORKERS_CAP=512
    (( GRD_FRAME_WORKERS_CAP > 1024 )) && GRD_FRAME_WORKERS_CAP=1024
    (( GRD_SHARED_FRAME_WORKERS_CAP > 1024 )) && GRD_SHARED_FRAME_WORKERS_CAP=1024
    GRD_LOW_MEMORY_CGROUP=1
  fi
  (( VIDEO_WORKERS > GRD_VIDEO_WORKERS_CAP )) && VIDEO_WORKERS=$GRD_VIDEO_WORKERS_CAP
  (( FRAME_WORKERS > GRD_FRAME_WORKERS_CAP )) && FRAME_WORKERS=$GRD_FRAME_WORKERS_CAP
  (( SHARED_FRAME_WORKERS > GRD_SHARED_FRAME_WORKERS_CAP )) && SHARED_FRAME_WORKERS=$GRD_SHARED_FRAME_WORKERS_CAP
  export VIDEO_WORKERS FRAME_WORKERS SHARED_FRAME_WORKERS
else
  # Episode-level concurrency already supplies thousands of independent LAS
  # calls.  Inner contact and bbox work runs inline per episode; keeping a
  # second 1,024-thread frame pool created a process-global submit-lock convoy
  # while HTTP occupancy stayed below two percent.
  if [[ -e "$SCRATCH/enable-sta-x4" ]]; then
    # The x4 profile is enabled only on hosts that have demonstrated ample
    # memory and scheduler headroom at x3.  It covers the several-minute LAS
    # queue latency with more independent episode waiters; inner pools remain
    # disabled so this does not recreate the submit-lock convoy.
    STA_VIDEO_WORKERS_CAP=${STA_VIDEO_WORKERS_CAP:-8192}
  elif [[ -e "$SCRATCH/enable-sta-x3" ]]; then
    # Canary/rollout profile for hosts with ample CPU and memory.  Keeping the
    # inner frame pool at zero avoids the submit-lock convoy seen in the old
    # x3 implementation; only independent episode waiters are increased.
    STA_VIDEO_WORKERS_CAP=${STA_VIDEO_WORKERS_CAP:-6144}
  else
    STA_VIDEO_WORKERS_CAP=${STA_VIDEO_WORKERS_CAP:-2048}
  fi
  STA_FRAME_WORKERS_CAP=${STA_FRAME_WORKERS_CAP:-1}
  STA_SHARED_FRAME_WORKERS_CAP=${STA_SHARED_FRAME_WORKERS_CAP:-0}
  (( VIDEO_WORKERS > STA_VIDEO_WORKERS_CAP )) && VIDEO_WORKERS=$STA_VIDEO_WORKERS_CAP
  (( FRAME_WORKERS > STA_FRAME_WORKERS_CAP )) && FRAME_WORKERS=$STA_FRAME_WORKERS_CAP
  (( SHARED_FRAME_WORKERS > STA_SHARED_FRAME_WORKERS_CAP )) && SHARED_FRAME_WORKERS=$STA_SHARED_FRAME_WORKERS_CAP
  export VIDEO_WORKERS FRAME_WORKERS SHARED_FRAME_WORKERS
fi
export BATCH_WORKERS=${BATCH_WORKERS:-32}
export MAX_PENDING=${MAX_PENDING:-8192}
if [[ "$TASK" == grd && "${GRD_LOW_MEMORY_CGROUP:-0}" == 1 ]]; then
  (( MAX_PENDING > 4096 )) && MAX_PENDING=4096
  (( ARK_UPLOAD_WORKERS > 512 )) && ARK_UPLOAD_WORKERS=512
  export MAX_PENDING ARK_UPLOAD_WORKERS
fi
if [[ "$TASK" == sta ]]; then
  (( MAX_PENDING > 16384 )) && MAX_PENDING=16384
  export MAX_PENDING
fi
export INDEPENDENT_STAGE_PIPELINE=1
# Production shards contain tens of thousands of records and keep their resume
# source on local XFS.  Sixteen validators made a safe worker restart spend
# tens of minutes replaying checkpoints before it could refill the model queue.
export RESUME_VALIDATION_WORKERS=${RESUME_VALIDATION_WORKERS:-32}
export THREAD_STACK_KIB=${THREAD_STACK_KIB:-256}
# GRD currently has eleven fleet processes sharing one 40,000 RPM Pro quota.
# Keep a small positive per-process floor to prevent synchronized submit/poll
# bursts, but allow the two independently paced GRD lanes to supply roughly
# 48,000 starts/minute before provider feedback.  This leaves the provider's
# 40,000 RPM limit, rather than an obsolete local 20,000-RPM profile, as the
# controlling ceiling.  Turbo STA has a separate quota and keeps its existing
# adaptive floor.
if [[ "$TASK" == grd ]]; then
  export REQUEST_START_INTERVAL=${REQUEST_START_INTERVAL:-0.025}
else
  export REQUEST_START_INTERVAL=${REQUEST_START_INTERVAL:-0.001}
fi
# Fleet quota and scheduling profiles are changed deliberately by this
# launcher. Do not resurrect an interval learned under an obsolete quota after
# a supervised restart; fresh 429 feedback can immediately adapt the new run.
export VQA_IGNORE_PERSISTED_RATE_STATE=1
# Optional per-worker quota canary.  The marker is deliberately local to the
# worker scratch tree; deploying code cannot activate aggressive pacing on the
# rest of the fleet.
if [[ "$TASK" == grd && -e "$SCRATCH/enable-grd-fast-quota-recovery" ]]; then
  export VQA_REQUEST_PACER_MAX_INTERVAL_SECONDS=0.05
fi
export FIXED_HTTP_CONCURRENCY=1
# Keep thousands of asynchronous LAS tasks live while bounding only the
# short-lived Submit/Poll HTTPS transactions.  This avoids TLS connection
# storms without reducing model-side task concurrency.
export VQA_LAS_CONTROL_HTTP_CONCURRENCY=${VQA_LAS_CONTROL_HTTP_CONCURRENCY:-256}
export VQA_LAS_POLL_INTERVAL_SECONDS=${VQA_LAS_POLL_INTERVAL_SECONDS:-60}
# Directly upload generated request clips.  A production A/B on 2026-09-22
# found that the server-side COS-to-TOS Fetch control path had become the
# dominant bottleneck (hundreds of seconds per object), whereas direct PUTs
# completed in seconds without permission errors.
export TOS_UPLOAD_VIA_COS_FETCH=${TOS_UPLOAD_VIA_COS_FETCH:-0}
export MEDIA_WORKERS=${MEDIA_WORKERS:-128}
export CLIP_WORKERS=${CLIP_WORKERS:-256}
export GRD_MEDIA_PREFETCH=${GRD_MEDIA_PREFETCH:-2048}
export DURABLE_WRITE_WORKERS=${DURABLE_WRITE_WORKERS:-128}
export REQUEST_MEMORY_MIB=${REQUEST_MEMORY_MIB:-81920}
if [[ "$TASK" == grd && "${GRD_LOW_MEMORY_CGROUP:-0}" == 1 ]]; then
  (( REQUEST_MEMORY_MIB > 16384 )) && REQUEST_MEMORY_MIB=16384
  (( GRD_MEDIA_PREFETCH > 512 )) && GRD_MEDIA_PREFETCH=512
  (( MEDIA_WORKERS > 64 )) && MEDIA_WORKERS=64
  (( CLIP_WORKERS > 128 )) && CLIP_WORKERS=128
  (( DURABLE_WRITE_WORKERS > 64 )) && DURABLE_WRITE_WORKERS=64
  (( RESUME_VALIDATION_WORKERS > 16 )) && RESUME_VALIDATION_WORKERS=16
  export REQUEST_MEMORY_MIB GRD_MEDIA_PREFETCH MEDIA_WORKERS CLIP_WORKERS
  export DURABLE_WRITE_WORKERS RESUME_VALIDATION_WORKERS
fi
# The process already keeps up to 2,048 independent episodes resident.  Giving
# every episode another 16-way submitter caused ~1,900 producer threads to
# queue on ThreadPoolExecutor's process-global submit lock while only tens of
# model calls were active.  Run subtasks sequentially inside one episode and
# use episode-level parallelism to fill the provider; STA bbox calls still use
# the shared bounded frame pool below.
export STA_EVENT_WORKERS=${STA_EVENT_WORKERS:-1}
if [[ "$TASK" == sta ]]; then
  # Keep enough episodes resident to prevent the LAS request fleet from
  # draining at every 1,024-record physical-batch boundary.  Request-memory
  # admission remains the hard safety limit.
  if [[ -e "$SCRATCH/enable-sta-x4" ]]; then
    # Leave one full worker lane available for refill while the current lane
    # waits on LAS submit/poll completion.
    export PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-16384}
  elif [[ -e "$SCRATCH/enable-sta-x3" ]]; then
    # The physical batch must be larger than the 6,144-record worker lane;
    # otherwise a nominal x3 profile silently plateaus at 4,096 episodes.
    export PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-8192}
  else
    export PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-4096}
  fi
else
  export PHYSICAL_BATCH_SIZE=${PHYSICAL_BATCH_SIZE:-1024}
fi
export VQA_SUPERVISOR_TMP_ROOT="$SCRATCH/tmp"
export TMPDIR="$SCRATCH/tmp"
# LAS task locks and atomic state updates create several files per request.
# Keep that write-heavy state on local XFS and consult the existing COS state
# as a bounded read-only fallback so already-submitted tasks remain reusable.
export VQA_LAS_TASK_STATE_ROOT=${VQA_LAS_TASK_STATE_ROOT:-$SCRATCH/las-operator-tasks}
export VQA_LAS_TASK_STATE_FALLBACK_CONCURRENCY=${VQA_LAS_TASK_STATE_FALLBACK_CONCURRENCY:-64}
if [[ -z "${VQA_LAS_TASK_STATE_FALLBACK_ROOT:-}" \
    && -n "$RESUME_OUTPUT" \
    && -d "$RESUME_OUTPUT/_state/las-operator-tasks" ]]; then
  export VQA_LAS_TASK_STATE_FALLBACK_ROOT="$RESUME_OUTPUT/_state/las-operator-tasks"
fi
# Fresh fleet nodes keep only the compact Cosmos3 catalog locally.  FileSlice
# uses these mappings to range-read unfinished media directly from COS instead
# of requiring every multi-GB physical WebDataset shard to be pre-copied.
# Catalog-only expansion nodes intentionally do not materialize the multi-GB
# WebDataset shards.  Their FileSlice paths still use the canonical physical
# dataset root, so use that root for COS key translation even when INPUT_ROOT
# is the compact catalog alias (which may not exist as a local directory).
canonical_physical_root=/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset
if [[ -f "$INPUT_ROOT/RELEASE.json" && -d "$INPUT_ROOT/catalog" && -d "$INPUT_ROOT/shards" ]]; then
  # FileSlice paths are rooted at the directory passed to the annotator.
  # Catalog-only nodes must translate that exact path to COS object keys.
  physical_cos_local_root=$(readlink -f -- "$INPUT_ROOT")
elif [[ -d "$canonical_physical_root" ]]; then
  physical_cos_local_root=$canonical_physical_root
else
  physical_cos_local_root=$(readlink -f -- "$INPUT_ROOT" 2>/dev/null || printf '%s' "$INPUT_ROOT")
fi
export VQA_COS_PHYSICAL_LOCAL_ROOT=${VQA_COS_PHYSICAL_LOCAL_ROOT:-$physical_cos_local_root}
export VQA_COS_PHYSICAL_PREFIX=${VQA_COS_PHYSICAL_PREFIX:-video-cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset}
export VQA_COS_PHYSICAL_FALLBACK_LOCAL_ROOTS=${VQA_COS_PHYSICAL_FALLBACK_LOCAL_ROOTS:-$canonical_physical_root:/run/ti/cosmos3-physical-catalog}
# Model concurrency and physical-shard download concurrency are independent.
# Thousands of episode waiters may remain live, but bounding range GETs avoids
# exhausting one node's COS connection path. FileSlice retries transient reads
# before the record-level retry budget is consumed.
export VQA_COS_INPUT_CONCURRENCY=${VQA_COS_INPUT_CONCURRENCY:-256}
export VQA_COS_INPUT_MAX_ATTEMPTS=${VQA_COS_INPUT_MAX_ATTEMPTS:-5}
# LASVideoPublisher uses the same mapping to server-copy a FileSlice from COS
# into TOS without opening or statting the intentionally absent local shard.
export TOS_COS_SOURCE_MOUNT_ROOTS=${TOS_COS_SOURCE_MOUNT_ROOTS:-$VQA_COS_PHYSICAL_LOCAL_ROOT:$VQA_COS_PHYSICAL_FALLBACK_LOCAL_ROOTS}
export TOS_COS_SOURCE_KEY_PREFIX=${TOS_COS_SOURCE_KEY_PREFIX:-$VQA_COS_PHYSICAL_PREFIX}

extra=(
  --tasks "$TASK"
  --api las
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
  --api-key-env LAS_API_KEY
  --immutable-jsonl
  --durable-cloud-transport cos-direct
  --runtime-state-dir "$SCRATCH/runtime-state"
  --subtask-path "$SUBTASK_PATH"
  --record-uid-path "$UID_PATH"
  --final-publication-spool-root "$SCRATCH/final-records"
  --final-publication-local-batch-size 1
  --checkpoint-sync-workers 256
  --physical-batch-size "$PHYSICAL_BATCH_SIZE"
  --grd-media-prefetch "$GRD_MEDIA_PREFETCH"
  --ark-upload-workers "$ARK_UPLOAD_WORKERS"
)
# GRD inventory review/correction must use the same LAS submit/poll transport
# as inventory and first-object inference.  Direct Ark review calls previously
# created thousands of socket writers per process, produced sustained write
# timeouts, and bypassed the fleet's LAS task-state reuse path.
if [[ "$TASK" == grd ]]; then
  extra+=(
    --grd-review-api las
    --grd-review-endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
    --grd-review-api-key-env LAS_API_KEY
  )
  if [[ -n "${GRD_MODEL:-}" ]]; then
    extra+=(--grd-model "${GRD_MODEL}")
    # Expansion and handoff workers that opt into Turbo must use Turbo for
    # the complete GRD chain.  Leaving inventory review/name correction on
    # the parser's Pro default made nominal Turbo workers wait tens of
    # minutes in the Pro queue and prevented any parent record from closing.
    if [[ "${GRD_MODEL}" == doubao-seed-2-1-turbo-260628 ]]; then
      extra+=(--grd-review-model "${GRD_MODEL}")
    fi
  fi
fi
if [[ -n "$RESUME_OUTPUT" ]]; then
  extra+=(--resume-output "$RESUME_OUTPUT" --skip-destination-resume-scan)
fi
if [[ -n "${ADDITIONAL_RESUME_OUTPUTS:-}" ]]; then
  IFS=':' read -r -a additional_resume_outputs <<< "$ADDITIONAL_RESUME_OUTPUTS"
  for additional_resume_output in "${additional_resume_outputs[@]}"; do
    [[ -n "$additional_resume_output" ]] || continue
    extra+=(--resume-output "$additional_resume_output")
  done
fi
if [[ "$TASK" == sta ]]; then
  extra+=(
    --sta-model "$STA_MODEL"
    --sta-event-workers "$STA_EVENT_WORKERS"
    --final-publication-workers 128
  )
  if [[ "$STA_MODEL" == doubao-seed-2-1-turbo-260628 ]]; then
    extra+=(
      --contact-frame-model doubao-seed-2-1-turbo-260628
      --cpa-model doubao-seed-2-1-turbo-260628
    )
  fi
fi

cd "$PROJECT"
echo "fleet_worker_start task=$TASK worker=$WORKER_ID uids=$(wc -l < "$UID_PATH") output=$OUTPUT_ROOT resume=${RESUME_OUTPUT:-none} sta_model=$STA_MODEL ark_upload_workers=$ARK_UPLOAD_WORKERS"

# Run each annotation attempt in its own process group.  tmux pane replacement
# otherwise kills only the outer shell and may leave Python/ffmpeg descendants
# holding the output and checkpoint locks.
child_pid=
record_error_restarts=0
max_record_error_restarts=${VQA_MAX_RECORD_ERROR_RESTARTS:-3}
[[ "$max_record_error_restarts" =~ ^[0-9]+$ ]] || {
  echo "invalid_record_error_restart_limit:$max_record_error_restarts" >&2
  exit 2
}
cleanup_child_group() {
  local rc=$?
  trap - EXIT TERM INT HUP
  if [[ -n "${child_pid:-}" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$child_pid" 2>/dev/null; then
      kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup_child_group EXIT TERM INT HUP

while true; do
  setsid bash scripts/run_cosmos3_seed_request256.sh "${extra[@]}" &
  child_pid=$!
  set +e
  wait "$child_pid"
  rc=$?
  set -e
  child_pid=
  if [[ "$rc" -eq 0 ]]; then
    echo "fleet_worker_complete task=$TASK worker=$WORKER_ID"
    exit 0
  elif [[ "$rc" -eq 10 ]]; then
    record_error_restarts=$((record_error_restarts + 1))
    if (( record_error_restarts <= max_record_error_restarts )); then
      echo "fleet_worker_retry_record_errors task=$TASK worker=$WORKER_ID pass=$record_error_restarts max_passes=$max_record_error_restarts restart_seconds=10"
      sleep 10
    else
      echo "fleet_worker_complete_with_record_errors task=$TASK worker=$WORKER_ID error_records_preserved=true retry_passes=$record_error_restarts"
      exit 10
    fi
  else
    echo "fleet_worker_recoverable_exit task=$TASK worker=$WORKER_ID exit_code=$rc restart_seconds=30"
    sleep 30
  fi
done
