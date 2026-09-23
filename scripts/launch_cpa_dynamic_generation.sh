#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 NODE_LABEL FIRST_SLOT SLOT_COUNT FLEET_ROOT [GPU_OFFSET]" >&2
  exit 2
fi

node_label=$1
first_slot=$2
slot_count=$3
fleet_root=$4
gpu_offset=${5:-0}
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
env_file=${CPA_ENV_FILE:-/mnt/cpa-ark-new/.env}

[[ ${first_slot} =~ ^[0-9]+$ && ${slot_count} =~ ^[0-9]+$ && ${slot_count} -gt 0 ]] || {
  echo "invalid_slot_range:first=${first_slot}:count=${slot_count}" >&2
  exit 2
}
[[ ${gpu_offset} =~ ^[0-9]+$ ]] || { echo "invalid_gpu_offset:${gpu_offset}" >&2; exit 2; }
[[ -f ${env_file} ]] || { echo "missing_env_file:${env_file}" >&2; exit 1; }
[[ -n ${CPA_STA_REUSE_OUTPUTS:-} ]] || { echo "missing_environment:CPA_STA_REUSE_OUTPUTS" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a

# Older CPA GPU images keep the production dependencies in this venv, while
# newer images install them into the system interpreter.  Select the prepared
# venv automatically instead of requiring a controller-side per-node override.
default_cpa_python=/usr/bin/python3
if [[ -x /mnt/venvs/cpa-sam3/bin/python ]]; then
  default_cpa_python=/mnt/venvs/cpa-sam3/bin/python
elif [[ -x /root/miniconda3/envs/sam3/bin/python ]]; then
  default_cpa_python=/root/miniconda3/envs/sam3/bin/python
fi
default_cpa_input=/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset
if [[ -s /run/ti/cosmos3-physical-catalog/RELEASE.json \
      && -s /run/ti/cosmos3-physical-catalog/catalog/samples.parquet ]]; then
  default_cpa_input=/run/ti/cosmos3-physical-catalog
fi

for ((offset=0; offset<slot_count; offset++)); do
  slot=$((first_slot + offset))
  gpu_index=$((gpu_offset + offset))
  slot_name=$(printf '%02d' "${slot}")
  uid_path="${fleet_root}/partitions/slot-${slot_name}.txt"
  [[ -s ${uid_path} ]] || { echo "missing_or_empty_slot:${uid_path}" >&2; exit 1; }
  session="vqa-cpa-dynamic-v3-slot-${slot_name}"
  # Use an exact session target so launching a newly numbered slot cannot
  # terminate another slot whose name merely shares the same prefix.
  tmux kill-session -t "=${session}" 2>/dev/null || true
  tmux new-session -d -s "${session}" \
    env \
      CPA_PYTHON="${CPA_PYTHON:-${default_cpa_python}}" \
      CPA_INPUT_ROOT="${CPA_INPUT_ROOT:-${default_cpa_input}}" \
      CPA_SUBTASK_ROOT="${CPA_SUBTASK_ROOT:-/mnt/cpa-input-cache/subtask}" \
      CPA_UID_PATH="${uid_path}" \
      CPA_STA_REUSE_OUTPUTS="${CPA_STA_REUSE_OUTPUTS}" \
      VQA_STA_LOCATION_INDEX="${fleet_root}/sta-location-index.json" \
      CPA_SHARD_INDEX=0 CPA_SHARD_COUNT=1 \
      bash "${project}/scripts/run_cpa_fleet_shard.sh" \
        "${node_label}" "${slot_name}" "${gpu_index}" "${fleet_root}"
  echo "started_cpa_slot session=${session} node=${node_label} slot=${slot_name} gpu=${gpu_index} records=$(wc -l < "${uid_path}")"
done
