# VQA Annotation Pipeline

[中文文档](README_zh.md)

A resumable production pipeline for Subtask, ECoT, GRD, STA, and CPA video
annotation. This clean tree contains only production code, runtime dependencies,
and the existing-GRD audit entry point. Datasets, checkpoints, credentials,
runtime state, probes, benchmarks, dashboards, and generated outputs are excluded.

Entry point: `scripts/annotate_videos.py`  
Generic wrapper: `scripts/run_pipeline.sh`  
LAS high-concurrency template: `scripts/run_las_high_concurrency.sh`

The Chinese [architecture](docs/ARCHITECTURE_zh.md) and
[operations runbook](docs/OPERATIONS_zh.md) document the current production
data flow, provider routing, recovery boundaries, and concurrency controls.

## Current contracts

| Stage | Behavior | Default Seed model |
|---|---|---|
| Subtask | Whole-video segmentation through vendored `doubao_las_annotation`; merge adjacent, contiguous segments only when normalized skill and description both match | `doubao-seed-2-1-pro-260628`; English postprocess `doubao-seed-2-0-lite-260428` |
| ECoT | Targets live on a 2 FPS grid and default to every fourth frame; the complete teacher video is 0.5 FPS | `doubao-seed-2-0-lite-260215` |
| GRD | Inventory every 2 FPS frame with at most four objects; only first-operated-object selection sees the 3 FPS × 4 s future clip | `doubao-seed-2-0-pro-260215` |
| STA | Use CPA-final contacts; sample eligible preceding frames without an intervening final contact; ground the named event and compute TTC | `doubao-seed-2-0-lite-260215` |
| CPA | Review events, select exact contact frames, choose points in a 15%-expanded crop, then snap with SAM3 in the crop and full frame | `doubao-seed-2-0-pro-260215` |

GRD distinguishability review/repair/re-review defaults to
`doubao-seed-2-1-pro-260628`. Frames that still contain ambiguous object names
after one constrained repair are rejected from task registration.

## Provider paths

`--api las` is the current high-volume production path. It calls
`las_long_video_understand/v1` through:

```text
https://operator.las.cn-beijing.volces.com/api/v1/submit
https://operator.las.cn-beijing.volces.com/api/v1/poll
```

The Submit payload uses `data.video_url`, `query`, `fps`, `model_name`, and
`ark_api_key`. `LAS_API_KEY` authenticates LAS, while `ARK_API_KEY` is supplied
to the operator for the customer's Ark inference. Media is stored in private COS
objects and passed as short-lived signed HTTPS URLs; secrets and signed URLs are
not persisted in public records or logs.

For LAS ECoT, one complete 0.5 FPS episode video is uploaded and reused. Each
target request identifies the exact timestamp and teacher-frame index in that
same URL; no target JPEG or per-target recomposed video is uploaded. GRD/STA
single images and clips are encoded as short MP4 inputs because the operator
accepts `video_url`. Operator sampling is 0.5 FPS for ECoT, 2 FPS for GRD/STA,
and 5 FPS for CPA-family requests. Sub-second composed clips are padded to 1.2 s.

Other supported downstream paths are Ark
`https://ark.cn-beijing.volces.com/api/v3/chat/completions`, DashScope's
OpenAI-compatible endpoint, and a configurable OpenAI-compatible endpoint.
Subtask generation always uses the vendored LAS implementation independently of
the downstream `--api` selection.

## Inputs

`--input` accepts:

- one `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, or `.m4v` video;
- a recursively scanned video directory; unrelated files are ignored;
- a JSONL manifest with unique `uid`/`record_uid` and `video_path` fields;
- the supported unified VQA WebDataset layout;
- a Cosmos3 physical WebDataset with `RELEASE.json` and Parquet catalogs.

Example manifest row:

```json
{"uid":"episode-001","video_path":"videos/001.mp4","task_name":"put the cup in the sink","source_key":"demo"}
```

## Installation and configuration

Python 3.10+, FFmpeg, and a C compiler are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp configs/api.env.example configs/api.env
```

For high-concurrency Linux runs, prebuild the optional native helpers:

```bash
PYTHONPATH=scripts python scripts/native_http_metrics.py --build
PYTHONPATH=scripts python scripts/native_availability_latch.py --build
PYTHONPATH=scripts python scripts/native_ecot_gate.py --build
```

Load a populated local configuration with
`set -a; source configs/api.env; set +a`. LAS customer-Ark mode requires
`LAS_API_KEY`, `ARK_API_KEY`, a configured
`coscli`, and the `VQA_COS_*` settings shown in the example file. CPA additionally
requires external SAM3 and CoTracker repositories/checkpoints. Large model assets
are intentionally not vendored.

Run `python scripts/check_install.py` for a local dependency/configuration
check that does not send network requests.

## Examples

Run all stages:

```bash
bash scripts/run_pipeline.sh /path/to/video.mp4 ./outputs/demo \
  --tasks subtask,ecot,grd,sta,cpa \
  --task-instruction "put the cup into the sink" \
  --api las \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
```

Run LAS ECoT with one reusable episode URL:

```bash
bash scripts/run_pipeline.sh /path/to/videos ./outputs/ecot \
  --tasks ecot --api las \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit \
  --ecot-video-transport cos-presigned \
  --ecot-image-transport cos-presigned \
  --immutable-jsonl
```

Reuse existing Subtask records for GRD:

```bash
bash scripts/run_pipeline.sh /path/to/videos ./outputs/grd \
  --tasks grd --subtask-path ./outputs/subtasks/subtask \
  --api las --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit \
  --grd-review-api las \
  --grd-review-endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
```

`--tasks` accepts any comma-separated subset of
`subtask,ecot,grd,sta,cpa`; `grounding` aliases `grd`. GRD, STA, or CPA without
Subtask requires `--subtask-path`. See `python scripts/annotate_videos.py --help`
for all concurrency, checkpoint, provider, and transport options.

## Output and recovery

```text
OUTPUT/<task>/shards/<source>/<batch>.jsonl
OUTPUT/errors/pipeline/<source>/<batch>.jsonl
OUTPUT/_state/annotation-unit-checkpoints/<task>/...
OUTPUT/_state/grd-window-checkpoints/...
OUTPUT/_state/las-pipeline/...
```

All public records use `unified-vqa-record/v2`.

| Task | Contract ID |
|---|---|
| Subtask | `vqa-anno-raw-subtask/v2` |
| ECoT | `vqa-anno-raw-ecot-privileged-teacher-0.5fps-atomic/v3` |
| GRD | `vqa-anno-raw-grd-2fps-inventory-future-first-object/v3` |
| STA | `vqa-anno-raw-sta-cpa-final-random-event-bbox/v5` |
| CPA | `vqa-anno-cpa-object-only-two-decimal/v13` |

The production path provides per-unit checkpoints, immutable JSONL, local
SQLite outboxes, asynchronous cloud publication, read-only parallel resume
validation, process-wide bounded request/event pools, and bounded media caches.
A configured HTTP ceiling is not a throughput claim; media preparation, resident
records, batch workers, provider latency, and machine resources determine useful
concurrency. Existing GRD records can be processed with
`scripts/audit_grounding_records.py`.

High-concurrency runs must use `--shared-frame-workers`; allocating a large
frame executor per episode causes thread and memory explosion. The production
template also exposes bounded media/clip pools, resume validators, durable
writers, thread-stack sizing, and a fixed HTTP ceiling. Start with its 2048
default and raise `HTTP_CONCURRENCY` only after checking durable throughput,
memory, transcoding capacity, and structured provider rate-limit counters.

Third-party provenance is documented in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
