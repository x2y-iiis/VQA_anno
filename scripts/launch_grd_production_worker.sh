#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 WORKER_ID UID_PATH OUTPUT_ROOT RESUME_ROOT PROFILE SESSION" >&2
  exit 2
fi

worker_id=$1
uid_path=$2
output_root=$3
resume_root=$4
profile=$5
session=$6
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
scratch="/run/ti/vqa-grd-sta-fleet/production-${worker_id}"
log_root=/run/ti/vqa-grd-sta-fleet/production-logs
log_path="${log_root}/${worker_id}.log"
export GRD_LOG_PATH="${log_path}"
default_python=/usr/bin/python3
if [[ -x /mnt/venvs/cpa-sam3/bin/python ]]; then
  default_python=/mnt/venvs/cpa-sam3/bin/python
fi
export PYTHON_BIN=${PYTHON_BIN:-${default_python}}

[[ -s ${uid_path} ]] || { echo "missing_uid_path:${uid_path}" >&2; exit 1; }
[[ -d ${resume_root} ]] || { echo "missing_resume_root:${resume_root}" >&2; exit 1; }
mkdir -p "${scratch}" "${log_root}"

stop_tmux_process_tree() {
  local target=$1
  local root child
  local -a pending=() processes=()
  while read -r root; do
    [[ -n ${root} ]] && pending+=("${root}")
  # Prefix matching is unsafe here.  During fanout the new session name does
  # not exist yet and often extends an existing sibling name; without the '='
  # exact-target marker tmux may select and terminate that sibling.
  done < <(tmux list-panes -t "=${target}" -F '#{pane_pid}' 2>/dev/null || true)
  while ((${#pending[@]})); do
    root=${pending[0]}
    pending=("${pending[@]:1}")
    processes+=("${root}")
    while read -r child; do
      [[ -n ${child} ]] && pending+=("${child}")
    done < <(pgrep -P "${root}" 2>/dev/null || true)
  done
  ((${#processes[@]})) || return 0
  kill -TERM "${processes[@]}" 2>/dev/null || true
  for _ in {1..15}; do
    local alive=0
    for root in "${processes[@]}"; do
      kill -0 "${root}" 2>/dev/null && alive=1
    done
    ((alive)) || return 0
    sleep 1
  done
  kill -KILL "${processes[@]}" 2>/dev/null || true
}

case "${profile}" in
  extreme)
    export VIDEO_WORKERS=1536 FRAME_WORKERS=12288 SHARED_FRAME_WORKERS=12288
    export REQUEST_MEMORY_MIB=196608 REQUEST_CONCURRENCY=12288
    ;;
  large)
    # Keep record-level scheduling at 4,096.  An 8,192-record canary retained
    # ample memory but reduced completed RPM because thread and dispatcher
    # contention outweighed the additional waiters.
    export VIDEO_WORKERS=4096 FRAME_WORKERS=8192 SHARED_FRAME_WORKERS=8192
    export REQUEST_MEMORY_MIB=163840 REQUEST_CONCURRENCY=8192
    ;;
  medium)
    export VIDEO_WORKERS=3072 FRAME_WORKERS=6144 SHARED_FRAME_WORKERS=6144
    export REQUEST_MEMORY_MIB=98304 REQUEST_CONCURRENCY=8192
    ;;
  shared)
    export VIDEO_WORKERS=2048 FRAME_WORKERS=4096 SHARED_FRAME_WORKERS=4096
    export REQUEST_MEMORY_MIB=81920 REQUEST_CONCURRENCY=6144
    ;;
  *)
    echo "invalid_profile:${profile}" >&2
    exit 2
    ;;
esac
export VQA_WORKER_SCRATCH="${scratch}"
export LAS_REQUEST_WORKERS="${REQUEST_CONCURRENCY}"
export MAX_LAS_OPERATORS="${REQUEST_CONCURRENCY}"
export MAX_PENDING=$((REQUEST_CONCURRENCY * 2))
export VQA_LAS_CONTROL_HTTP_CONCURRENCY=${VQA_LAS_CONTROL_HTTP_CONCURRENCY:-512}
# GRD shares the same LAS Submit/Poll control plane as STA.  Long-lived model
# tasks do not benefit from 15-second polling, which only creates poll storms.
if [[ ${GRD_MODEL:-} == doubao-seed-2-1-turbo-260628 ]]; then
  # Turbo GRD calls commonly finish within one minute.  A five-minute poll
  # interval leaves completed LAS tasks occupying request slots and cuts the
  # observed throughput several-fold.  Keep Pro's conservative interval, but
  # collect Turbo completions promptly.
  export VQA_LAS_FIRST_POLL_TARGET_SECONDS=${VQA_LAS_FIRST_POLL_TARGET_SECONDS:-30}
  export VQA_LAS_POLL_INTERVAL_SECONDS=${VQA_LAS_POLL_INTERVAL_SECONDS:-60}
else
  # The established Pro fleet runs stably at first=60s / poll=120s.  A newer
  # 300-second default delayed the discovery of completed tasks and regressed
  # donor throughput after a tail split.
  export VQA_LAS_FIRST_POLL_TARGET_SECONDS=${VQA_LAS_FIRST_POLL_TARGET_SECONDS:-60}
  export VQA_LAS_POLL_INTERVAL_SECONDS=${VQA_LAS_POLL_INTERVAL_SECONDS:-120}
fi
# Small generated GRD request clips are faster and more reliable through
# direct TOS PUT than through the saturated COS-to-TOS Fetch control path.
tos_fetch_default=0
[[ -e "${scratch}/use-cos-fetch" ]] && tos_fetch_default=1
export TOS_UPLOAD_VIA_COS_FETCH=${TOS_UPLOAD_VIA_COS_FETCH:-${tos_fetch_default}}
export VQA_LAS_TASK_STATE_FALLBACK_ROOT=${VQA_LAS_TASK_STATE_FALLBACK_ROOT:-}
export GRD_UPLOAD_WORKERS=${GRD_UPLOAD_WORKERS:-1024}
export MEDIA_WORKERS=${MEDIA_WORKERS:-128}
export CLIP_WORKERS=${CLIP_WORKERS:-256}
export GRD_MEDIA_PREFETCH=${GRD_MEDIA_PREFETCH:-2048}
# Leave existing workers on the pipeline default (Seed 2.1 Pro), while
# allowing expansion workers to use Seed 2.1 Turbo without changing their
# output or resume contract.
export GRD_MODEL=${GRD_MODEL:-}
if [[ ${GRD_MODEL} == doubao-seed-2-1-turbo-260628 && -z ${LAS_KEY_FILE:-} ]]; then
  export LAS_KEY_FILE=/mnt/doubao_las_annotation/.runtime/las_api_key_turbo
else
  export LAS_KEY_FILE=${LAS_KEY_FILE:-}
fi

stop_tmux_process_tree "${session}"
tmux kill-session -t "=${session}" 2>/dev/null || true
tmux new-session -d -s "${session}" \
  env \
    GRD_LOG_PATH="${GRD_LOG_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    VQA_WORKER_SCRATCH="${VQA_WORKER_SCRATCH}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" FRAME_WORKERS="${FRAME_WORKERS}" \
    SHARED_FRAME_WORKERS="${SHARED_FRAME_WORKERS}" \
    REQUEST_MEMORY_MIB="${REQUEST_MEMORY_MIB}" \
    REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY}" \
    LAS_REQUEST_WORKERS="${LAS_REQUEST_WORKERS}" \
    MAX_LAS_OPERATORS="${MAX_LAS_OPERATORS}" MAX_PENDING="${MAX_PENDING}" \
    VQA_LAS_CONTROL_HTTP_CONCURRENCY="${VQA_LAS_CONTROL_HTTP_CONCURRENCY}" \
    VQA_LAS_POLL_INTERVAL_SECONDS="${VQA_LAS_POLL_INTERVAL_SECONDS}" \
    VQA_LAS_FIRST_POLL_TARGET_SECONDS="${VQA_LAS_FIRST_POLL_TARGET_SECONDS:-}" \
    MEDIA_WORKERS="${MEDIA_WORKERS}" CLIP_WORKERS="${CLIP_WORKERS}" \
    GRD_MEDIA_PREFETCH="${GRD_MEDIA_PREFETCH}" \
    GRD_VIDEO_WORKERS_CAP="${GRD_VIDEO_WORKERS_CAP:-${VIDEO_WORKERS}}" \
    GRD_FRAME_WORKERS_CAP="${GRD_FRAME_WORKERS_CAP:-${REQUEST_CONCURRENCY}}" \
    GRD_SHARED_FRAME_WORKERS_CAP="${GRD_SHARED_FRAME_WORKERS_CAP:-${REQUEST_CONCURRENCY}}" \
    TOS_UPLOAD_VIA_COS_FETCH="${TOS_UPLOAD_VIA_COS_FETCH}" \
    VQA_LAS_TASK_STATE_FALLBACK_ROOT="${VQA_LAS_TASK_STATE_FALLBACK_ROOT}" \
    GRD_MODEL="${GRD_MODEL}" \
    LAS_KEY_FILE="${LAS_KEY_FILE}" \
  bash -c 'exec "$@" >>"$GRD_LOG_PATH" 2>&1' _ \
    bash "${project}/scripts/run_grd_sta_fleet_worker.sh" \
      grd "${worker_id}" "${uid_path}" "${output_root}" "${resume_root}" \
      doubao-seed-2-0-lite-260215 "${GRD_UPLOAD_WORKERS}"
echo "started_grd_worker session=${session} worker=${worker_id} profile=${profile} log=${log_path} records=$(wc -l < "${uid_path}")"
