#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 HOST" >&2; exit 2; }
host=$1
project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
physical_local=/run/ti/cosmos3-physical-catalog
physical_cos=cos://datasets-1409717487/video-cleaning/cosmos3-video-generation-general-v1.5-physical-webdataset

ssh "root@${host}" "mkdir -p '${project}' '${physical_local}/catalog' '${physical_local}/shards' /mnt/doubao_las_annotation/.runtime /mnt/cpa-ark-new"
tar -C "${project}" -cf - scripts configs requirements.txt README.md README_zh.md .gitignore \
  third_party/doubao_las_annotation/las_annotation \
  | ssh "root@${host}" "tar -C '${project}' -xf -"
scp -q /root/coscli "root@${host}:/root/coscli"
scp -q /root/.cos.yaml "root@${host}:/root/.cos.yaml"
scp -q /mnt/cpa-ark-new/.env "root@${host}:/mnt/cpa-ark-new/.env"
scp -q /mnt/doubao_las_annotation/.runtime/las_api_key \
  /mnt/doubao_las_annotation/.runtime/las_api_key_2_1_pro \
  /mnt/doubao_las_annotation/.runtime/las_api_key_turbo \
  /mnt/doubao_las_annotation/.runtime/ark_api_key \
  "root@${host}:/mnt/doubao_las_annotation/.runtime/"
ssh "root@${host}" "
  chmod 700 /root/coscli
  chmod 600 /root/.cos.yaml /mnt/cpa-ark-new/.env /mnt/doubao_las_annotation/.runtime/*key*
  if [[ ! -s '${physical_local}/RELEASE.json' ]]; then
    /root/coscli cp \
      '${physical_cos}/RELEASE.json' \
      '${physical_local}/RELEASE.json' --thread-num 32 --disable-log
  fi
  for file in samples.parquet members.parquet shards.parquet; do
    if [[ ! -s '${physical_local}/catalog/'\"\${file}\" ]]; then
      /root/coscli cp \
        '${physical_cos}/catalog/'\"\${file}\" \
        '${physical_local}/catalog/'\"\${file}\" --thread-num 256 --disable-log
    fi
  done
  python3 -m py_compile '${project}/scripts/annotate_videos.py'
  grep -q VQA_COS_PHYSICAL_LOCAL_ROOT '${project}/scripts/run_grd_sta_fleet_worker.sh'
  echo STA_PROVISION_COMPLETE
"
