#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 HOST ROLE" >&2
  echo "ROLE must be one of: grd sta cpa" >&2
  exit 2
fi

host=$1
role=$2
case "${role}" in grd|sta|cpa) ;; *) echo "invalid_role:${role}" >&2; exit 2;; esac

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cos_root=cos://datasets-1409717487/video-cleaning/_runtime/vqa-fleet-bootstrap/v1
remote_stage=/run/ti/vqa-fleet-bootstrap-v1

ssh_options=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
ssh "${ssh_options[@]}" "root@${host}" \
  "mkdir -p '${project}' '${remote_stage}' /mnt/doubao_las_annotation/.runtime /mnt/cpa-ark-new /root"

scp -q "${ssh_options[@]}" /root/coscli "root@${host}:/root/coscli"
scp -q "${ssh_options[@]}" /root/.cos.yaml "root@${host}:/root/.cos.yaml"
scp -q "${ssh_options[@]}" /mnt/cpa-ark-new/.env "root@${host}:/mnt/cpa-ark-new/.env"
scp -q "${ssh_options[@]}" \
  /mnt/doubao_las_annotation/.runtime/las_api_key \
  /mnt/doubao_las_annotation/.runtime/las_api_key_2_1_pro \
  /mnt/doubao_las_annotation/.runtime/las_api_key_turbo \
  /mnt/doubao_las_annotation/.runtime/ark_api_key \
  /mnt/doubao_las_annotation/.runtime/tos.env \
  "root@${host}:/mnt/doubao_las_annotation/.runtime/"

ssh "${ssh_options[@]}" "root@${host}" bash -s -- \
  "${role}" "${cos_root}" "${remote_stage}" "${project}" <<'REMOTE'
set -euo pipefail
role=$1
cos_root=$2
stage=$3
project=$4

chmod 700 /root/coscli
  chmod 600 /root/.cos.yaml /mnt/cpa-ark-new/.env \
    /mnt/doubao_las_annotation/.runtime/*key* \
    /mnt/doubao_las_annotation/.runtime/tos.env

fetch() {
  local name=$1
  if [[ ! -s "${stage}/${name}" ]]; then
    /root/coscli cp "${cos_root}/${name}" "${stage}/${name}.partial" \
      --thread-num 256 --disable-log
    mv "${stage}/${name}.partial" "${stage}/${name}"
  fi
}

fetch project-production.tar
fetch physical-catalog.tar
fetch subtask-cache.tar
tar -C "${project}" -xf "${stage}/project-production.tar"
tar -C /run/ti -xf "${stage}/physical-catalog.tar"
tar -C /run/ti -xf "${stage}/subtask-cache.tar"
mkdir -p /mnt/cpa-input-cache
if [[ ! -e /mnt/cpa-input-cache/subtask ]]; then
  ln -s /run/ti/vqa-subtask-index-20260916 /mnt/cpa-input-cache/subtask
fi

# CPA is the only role that requires local GPUs, SAM3, and the CPA Python
# environment.  GRD and STA are remote-model workloads and must remain
# deployable on CPU-only nodes; forcing them through the CPA asset gate kept
# freshly restarted CPU nodes out of production.
if [[ ${role} == cpa ]]; then
  command -v nvidia-smi >/dev/null || { echo nvidia_smi_missing >&2; exit 3; }
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -eq 4 ]] || {
    echo cpa_requires_four_visible_gpus >&2
    exit 3
  }
  fetch cpa-assets.tar
  tar -C /mnt -xf "${stage}/cpa-assets.tar"
  mkdir -p /mnt/SAM3/checkpoints
  if [[ ! -s /mnt/SAM3/checkpoints/sam3.pt ]]; then
    /root/coscli cp "${cos_root}/sam3.pt" /mnt/SAM3/checkpoints/sam3.pt.partial \
      --thread-num 256 --disable-log
    mv /mnt/SAM3/checkpoints/sam3.pt.partial /mnt/SAM3/checkpoints/sam3.pt
  fi
  if [[ -x /mnt/venvs/cpa-sam3/bin/python ]]; then
    cpa_python=/mnt/venvs/cpa-sam3/bin/python
  elif [[ -x /root/miniconda3/envs/sam3/bin/python ]]; then
    cpa_python=/root/miniconda3/envs/sam3/bin/python
  else
    cpa_python=/usr/bin/python3
  fi
  "${cpa_python}" -c 'import cv2, numpy, torch, tos; assert torch.cuda.is_available()'
  # The exact-location index dynamically adds every live STA publisher.  Keep
  # one canonical logical root so the first worker can start even on a clean
  # node with no COS/FUSE mount.
  mkdir -p \
    /mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-seed-sta-full-20260909/sta/shards
fi

/usr/bin/python3 -c 'import cv2, pyarrow, requests, tos'
/usr/bin/python3 -m py_compile \
  "${project}/scripts/annotate_videos.py" \
  "${project}/scripts/cpa_fleet_controller.py"
test -s /run/ti/cosmos3-physical-catalog/RELEASE.json
test -s /run/ti/cosmos3-physical-catalog/catalog/samples.parquet
test -s /run/ti/vqa-subtask-index-20260916/.complete-319355-records
echo "new_fleet_node_ready:role=${role}:host=$(hostname):subtask_records=$(cat /run/ti/vqa-subtask-index-20260916/.complete-319355-records)"
REMOTE
