#!/usr/bin/env bash
set -euo pipefail

project=${VQA_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
stage=/run/ti/vqa-fleet-bootstrap-v1
cos_root=cos://datasets-1409717487/video-cleaning/_runtime/vqa-fleet-bootstrap/v1

required=(
  /root/coscli
  /root/.cos.yaml
  /mnt/SAM3/checkpoints/sam3.pt
  /mnt/SAM3/repo
  /mnt/co-tracker
  /mnt/robot_vqa_sta_cpa/src/generate_robot.py
  /run/ti/cosmos3-physical-catalog/RELEASE.json
  /run/ti/cosmos3-physical-catalog/catalog/samples.parquet
  /run/ti/vqa-subtask-index-20260916/.complete-319355-records
)
for path in "${required[@]}"; do
  [[ -e ${path} ]] || { echo "bootstrap_source_missing:${path}" >&2; exit 1; }
done

mkdir -p "${stage}"

tar -C "${project}" -cf "${stage}/project-production.tar" \
  scripts configs requirements.txt README.md README_zh.md .gitignore \
  third_party/doubao_las_annotation/las_annotation \
  _runtime/v3-endpoint-switch-20260916/x2y3
tar -C /run/ti -cf "${stage}/physical-catalog.tar" cosmos3-physical-catalog
tar -C /run/ti -cf "${stage}/subtask-cache.tar" vqa-subtask-index-20260916
tar -C /mnt -cf "${stage}/cpa-assets.tar" \
  SAM3/repo co-tracker robot_vqa_sta_cpa/src/generate_robot.py

for file in project-production.tar physical-catalog.tar subtask-cache.tar cpa-assets.tar; do
  /root/coscli cp "${stage}/${file}" "${cos_root}/${file}" \
    --thread-num 256 --disable-log
  echo "bootstrap_asset_staged:file=${file}:bytes=$(stat -c %s "${stage}/${file}")"
done
/root/coscli cp /mnt/SAM3/checkpoints/sam3.pt "${cos_root}/sam3.pt" \
  --thread-num 256 --disable-log
echo "bootstrap_asset_staged:file=sam3.pt:bytes=$(stat -c %s /mnt/SAM3/checkpoints/sam3.pt)"

/root/coscli ls "${cos_root}/" --disable-log
echo "bootstrap_stage_complete:cos_root=${cos_root}"
