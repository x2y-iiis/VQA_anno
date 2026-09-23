#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 WORKER_ID UID_PATH OUTPUT_ROOT RESUME_ROOT SESSION" >&2
  exit 2
fi

worker_id=$1
uid_path=$2
output_root=$3
resume_root=$4
session=$5
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
scratch="/run/ti/vqa-grd-sta-fleet/production-${worker_id}"
log_root=/run/ti/sta-dynamic-v2/production-logs
log_path="${log_root}/${worker_id}.log"
export STA_LOG_PATH="${log_path}"
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
  # Require an exact target.  A non-existent child session commonly has an
  # existing sibling as its name prefix during fanout, and tmux otherwise
  # resolves that sibling and kills valid work.
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

export VQA_WORKER_SCRATCH="${scratch}"
# Under fleet load, hundreds of concurrent public TOS PUTs spent 60--200
# seconds per clip and frequently timed out.  A same-node production A/B on
# 2026-09-22 sustained about 130 successful media publications/minute/process
# through COS-to-TOS Fetch versus 1--2/minute/process through direct PUT.
export TOS_UPLOAD_VIA_COS_FETCH=${TOS_UPLOAD_VIA_COS_FETCH:-1}
# LAS tasks usually take several minutes.  Polling every 15 seconds across
# thousands of in-flight tasks overwhelms the LAS control plane and produces
# large bursts of control-plane 429s without increasing model throughput.
# A 60-second interval reduced those 429s by more than an order of magnitude
# in the production A/B test while preserving submit throughput.
export VQA_LAS_POLL_INTERVAL_SECONDS=${VQA_LAS_POLL_INTERVAL_SECONDS:-60}
export REQUEST_CONCURRENCY=${REQUEST_CONCURRENCY:-8192}
export LAS_REQUEST_WORKERS=${LAS_REQUEST_WORKERS:-12288}
export MAX_LAS_OPERATORS=${MAX_LAS_OPERATORS:-12288}
export MAX_PENDING=${MAX_PENDING:-16384}
export VQA_LAS_CONTROL_HTTP_CONCURRENCY=${VQA_LAS_CONTROL_HTTP_CONCURRENCY:-1536}
export VIDEO_WORKERS=${VIDEO_WORKERS:-8192}
export FRAME_WORKERS=${FRAME_WORKERS:-1}
export SHARED_FRAME_WORKERS=${SHARED_FRAME_WORKERS:-0}
export STA_EVENT_WORKERS=${STA_EVENT_WORKERS:-4}
export VQA_COS_INPUT_CONCURRENCY=${VQA_COS_INPUT_CONCURRENCY:-256}
export VQA_COS_INPUT_MAX_ATTEMPTS=${VQA_COS_INPUT_MAX_ATTEMPTS:-5}
# Direct TOS PUT is bandwidth-bound, not CPU-bound.  Thousands of simultaneous
# PUTs overload the node's egress path and turn otherwise sub-second uploads
# into write timeouts, which makes whole STA records retry.  Keep model/LAS
# concurrency high while bounding only the short-lived upload lane.
export ARK_UPLOAD_WORKERS=${ARK_UPLOAD_WORKERS:-256}
export VQA_LAS_TASK_STATE_FALLBACK_ROOT=${VQA_LAS_TASK_STATE_FALLBACK_ROOT:-}

stop_tmux_process_tree "${session}"
tmux kill-session -t "=${session}" 2>/dev/null || true
tmux new-session -d -s "${session}" \
  env \
    STA_LOG_PATH="${STA_LOG_PATH}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    VQA_WORKER_SCRATCH="${VQA_WORKER_SCRATCH}" \
    TOS_UPLOAD_VIA_COS_FETCH="${TOS_UPLOAD_VIA_COS_FETCH}" \
    VQA_LAS_POLL_INTERVAL_SECONDS="${VQA_LAS_POLL_INTERVAL_SECONDS}" \
    REQUEST_CONCURRENCY="${REQUEST_CONCURRENCY}" \
    LAS_REQUEST_WORKERS="${LAS_REQUEST_WORKERS}" \
    MAX_LAS_OPERATORS="${MAX_LAS_OPERATORS}" MAX_PENDING="${MAX_PENDING}" \
    VQA_LAS_CONTROL_HTTP_CONCURRENCY="${VQA_LAS_CONTROL_HTTP_CONCURRENCY}" \
    VIDEO_WORKERS="${VIDEO_WORKERS}" FRAME_WORKERS="${FRAME_WORKERS}" \
    SHARED_FRAME_WORKERS="${SHARED_FRAME_WORKERS}" \
    STA_EVENT_WORKERS="${STA_EVENT_WORKERS}" \
    VQA_COS_INPUT_CONCURRENCY="${VQA_COS_INPUT_CONCURRENCY}" \
    VQA_COS_INPUT_MAX_ATTEMPTS="${VQA_COS_INPUT_MAX_ATTEMPTS}" \
    VQA_LAS_TASK_STATE_FALLBACK_ROOT="${VQA_LAS_TASK_STATE_FALLBACK_ROOT}" \
  bash -c 'exec "$@" >>"$STA_LOG_PATH" 2>&1' _ \
    bash "${project}/scripts/run_grd_sta_fleet_worker.sh" \
      sta "${worker_id}" "${uid_path}" "${output_root}" "${resume_root}" \
      doubao-seed-2-1-turbo-260628 "${ARK_UPLOAD_WORKERS}"
echo "started_sta_worker session=${session} worker=${worker_id} log=${log_path} records=$(wc -l < "${uid_path}")"
