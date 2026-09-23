#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 NODE_LABEL GLOBAL_SHARD GPU_INDEX FLEET_ROOT" >&2
  exit 2
fi

node_label=$1
global_shard=$2
gpu_index=$3
fleet_root=$4
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
input=${CPA_INPUT_ROOT:-/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset}
output=/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-cpa-object2dp-full-20260913
subtask=${CPA_SUBTASK_ROOT:-/mnt/cpa-input-cache/subtask}
python_bin=${CPA_PYTHON:-/usr/bin/python3}
runtime="${fleet_root}/${node_label}/shard-${global_shard}"
scratch="/run/ti/cpa-fleet/${node_label}/shard-${global_shard}"
uid_path=${CPA_UID_PATH:-${fleet_root}/partitions/uids-shard-${global_shard}.txt}
completed="${fleet_root}/completed-cpa-uids.txt"
sta_reuse_outputs=${CPA_STA_REUSE_OUTPUTS:-/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-seed-sta-full-20260909:/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-turbo-default-20260912/sta-output:/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-grd-sta-fleet-20260918/sta-x2y-6_5}
http_limit=${CPA_HTTP_LIMIT:-1024}
resident_episodes=${CPA_RESIDENT_EPISODES:-96}
worker_threads=${CPA_WORKERS:-192}
pending_records=${CPA_MAX_PENDING:-288}
event_workers=${CPA_EVENT_WORKERS:-24}
prepare_workers=${CPA_PREPARE_WORKERS:-12}
clip_workers=${CPA_CLIP_WORKERS:-24}
request_memory_mib=${CPA_REQUEST_MEMORY_MIB:-49152}
tracker_replicas=${CPA_TRACKER_REPLICAS:-8}
logical_shard_index=${CPA_SHARD_INDEX:-${global_shard}}
logical_shard_count=${CPA_SHARD_COUNT:-11}

# The local notebook exposes host memory through `free`, but its annotation
# pod is limited to 120 GiB and shares that cgroup with four CPA replicas and
# GRD.  Bound per-replica resident work on low-memory cgroups; SAM3 stays loaded
# on every GPU slot, while transient frame/request memory no longer drives the
# entire pod into OOM.
memory_cgroup_limit=0
if [[ -r /sys/fs/cgroup/memory.max ]]; then
  read -r memory_cgroup_value </sys/fs/cgroup/memory.max
  [[ "$memory_cgroup_value" == max ]] || memory_cgroup_limit=$memory_cgroup_value
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
  read -r memory_cgroup_limit </sys/fs/cgroup/memory/memory.limit_in_bytes
fi
if [[ "$memory_cgroup_limit" =~ ^[0-9]+$ ]] \
    && (( memory_cgroup_limit > 0 && memory_cgroup_limit < 171798691840 )); then
  (( http_limit > 512 )) && http_limit=512
  (( resident_episodes > 48 )) && resident_episodes=48
  (( worker_threads > 96 )) && worker_threads=96
  (( pending_records > 144 )) && pending_records=144
  (( event_workers > 12 )) && event_workers=12
  (( prepare_workers > 8 )) && prepare_workers=8
  (( clip_workers > 12 )) && clip_workers=12
  (( request_memory_mib > 24576 )) && request_memory_mib=24576
  (( tracker_replicas > 4 )) && tracker_replicas=4
fi

[[ ${tracker_replicas} =~ ^([1-9]|1[0-6])$ ]] || {
  echo "invalid_cpa_tracker_replicas:${tracker_replicas}" >&2
  exit 2
}

for required in \
  "${uid_path}" "${completed}" "${python_bin}" \
  /mnt/cpa-ark-new/.env /root/.cos.yaml /mnt/SAM3/checkpoints/sam3.pt \
  /mnt/robot_vqa_sta_cpa/src/generate_robot.py \
  /mnt/co-tracker/cotracker/__init__.py \
  /mnt/co-tracker/checkpoints/scaled_offline.pth; do
  [[ -e "${required}" ]] || { echo "missing_required_path:${required}" >&2; exit 1; }
done
IFS=: read -r -a sta_reuse_roots <<< "${sta_reuse_outputs}"
sta_reuse_args=()
sta_cos_prefixes=()
for root in "${sta_reuse_roots[@]}"; do
  [[ -d "${root}/sta" || "$(basename "${root}")" == sta && -d "${root}" ]] || {
    echo "missing_sta_reuse_output:${root}" >&2
    exit 1
  }
  sta_reuse_args+=(--reuse-sta-output "${root}")
  task_root="${root}"
  [[ "$(basename "${root}")" == sta ]] || task_root="${root}/sta"
  case "${task_root}" in
    /mnt/human_data/video_cleaning/*)
      sta_cos_prefixes+=("video-cleaning/${task_root#/mnt/human_data/video_cleaning/}")
      ;;
    /mnt/human_data/video-cleaning/*)
      sta_cos_prefixes+=("video-cleaning/${task_root#/mnt/human_data/video-cleaning/}")
      ;;
    *)
      echo "unsupported_sta_reuse_cos_mapping:${task_root}" >&2
      exit 1
      ;;
  esac
done
[[ ${#sta_reuse_args[@]} -gt 0 ]] || { echo 'empty_sta_reuse_outputs' >&2; exit 1; }
mkdir -p "${input}/shards" "${subtask}/shards/cosmos3_v1_5" "${runtime}" "${scratch}"
if [[ ! -e "${input}/catalog" && -d /run/ti/cosmos3-catalog ]]; then
  ln -s /run/ti/cosmos3-catalog "${input}/catalog"
fi
[[ -f "${input}/RELEASE.json" ]] || { echo "missing_physical_release_metadata:${input}/RELEASE.json" >&2; exit 1; }

export PYTHONPATH="${project}/scripts:${PYTHONPATH-}"
export CUDA_VISIBLE_DEVICES="${gpu_index}"
export VQA_COS_PHYSICAL_LOCAL_ROOT="${input}"
export VQA_COS_PHYSICAL_PREFIX='video-cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset'
export VQA_COS_SUBTASK_LOCAL_ROOT="${subtask}"
export VQA_COS_SUBTASK_PREFIX='video-cleaning/cosmos3-video-generation-general-v1.5-vqa-las-seed-full-20260906/subtask'
export VQA_COS_STA_PREFIXES="$(IFS=:; echo "${sta_cos_prefixes[*]}")"
export VQA_STA_PACK_CACHE="${VQA_STA_PACK_CACHE:-/run/ti/sta-pack-cache}"
export PYTHONUNBUFFERED=1
export CPA_TRACKER_REPLICAS="${tracker_replicas}"

attempt=0
child_pid=

terminate_child_group() {
  local pid=${1:-}
  [[ -n "${pid}" ]] || return 0
  if kill -0 -- "-${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 -- "-${pid}" 2>/dev/null || return 0
      sleep 0.5
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
}

cleanup_child_group() {
  local code=$?
  trap - EXIT TERM INT HUP
  terminate_child_group "${child_pid:-}"
  [[ -z "${child_pid:-}" ]] || wait "${child_pid}" 2>/dev/null || true
  exit "${code}"
}
trap cleanup_child_group EXIT TERM INT HUP

while true; do
  attempt=$((attempt + 1))
  attempt_log="${runtime}/attempt-${attempt}.log"
  echo "cpa_fleet_start node=${node_label} shard=${global_shard} gpu=${gpu_index} attempt=${attempt} tracker_replicas=${tracker_replicas} started_at=$(date --iso-8601=seconds)"
  set +e
  # Keep the CPA main process and all multiprocessing tracker children in one
  # process group.  A pipeline through `tee` previously left `tee` and orphaned
  # tracker children alive after cgroup OOM killed the main process, so tmux
  # looked healthy forever while the GPU slot produced no records.
  setsid "${python_bin}" -u \
    "${project}/_runtime/v3-endpoint-switch-20260916/x2y3/run_cosmos3_cpa.py" \
    --input "${input}" --output "${output}" --runtime "${runtime}" --scratch-root "${scratch}" \
    --reuse-subtask-root "${subtask}" --record-uid-path "${uid_path}" \
    "${sta_reuse_args[@]}" \
    --completed-uid-path "${completed}" \
    --shard-index "${logical_shard_index}" --shard-count "${logical_shard_count}" \
    --writer-lock-id "${node_label}-${global_shard}" \
    --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit --api-key-env LAS_API_KEY \
    --sam3-device cuda --http-limit "${http_limit}" --episodes "${resident_episodes}" \
    --workers "${worker_threads}" --max-pending "${pending_records}" \
    --event-workers "${event_workers}" --prepare-workers "${prepare_workers}" \
    --clip-workers "${clip_workers}" --request-memory-mib "${request_memory_mib}" \
    --cpa-model doubao-seed-2-1-turbo-260628 \
    --contact-frame-model doubao-seed-2-1-turbo-260628 \
    --semantic-review-model doubao-seed-2-0-lite-260215 --no-material-point-review \
    >>"${attempt_log}" 2>&1 &
  child_pid=$!
  wait "${child_pid}"
  code=$?
  terminate_child_group "${child_pid}"
  wait "${child_pid}" 2>/dev/null || true
  child_pid=
  set -e
  echo "cpa_fleet_exit node=${node_label} shard=${global_shard} attempt=${attempt} code=${code} stopped_at=$(date --iso-8601=seconds)" | tee -a "${runtime}/run.log"
  # annotate_videos returns 10 when a finite scope completed with durable
  # per-record failures.  Older CPA launchers can also surface that terminal
  # condition as 1 after printing run_complete.  Re-running the whole static
  # UID scope in either case only burns GPU/API capacity on already committed
  # records; the dynamic feeder owns retrying the small failed remainder.
  if [[ ${code} -eq 0 || ${code} -eq 10 ]] || grep -q '^run_complete ' "${attempt_log}"; then
    # The controller polls every 30 seconds, while a tiny tail lease can start
    # and finish between two polls.  Publish terminal, durable completions to a
    # node-local append-only ledger so the incremental collector cannot miss
    # those short-lived workers.  annotate_videos drains the final outbox
    # before returning; crashed/non-terminal attempts never reach this block.
    {
      flock -x 9
      grep -haoE 'stage_written uid=[^ ]+ task=cpa' "${attempt_log}" 2>/dev/null \
        | sed -E 's/^stage_written uid=([^ ]+) task=cpa$/\1/' \
        >> /run/ti/cpa-completed-uids.log
    } 9>/run/ti/cpa-completed-uids.lock
    echo "cpa_fleet_scope_complete node=${node_label} shard=${global_shard} attempt=${attempt} code=${code}" | tee -a "${runtime}/run.log"
    exit 0
  fi
  sleep 30
done
