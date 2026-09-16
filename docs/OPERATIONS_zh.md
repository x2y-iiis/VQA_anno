# 生产运行手册

## 运行前检查

1. 使用 Python 3.10+，确认 `ffmpeg`、`ffprobe` 和 `coscli` 可执行。
2. 加载 `LAS_API_KEY`、`ARK_API_KEY` 及 `configs/api.env.example` 中的 COS 配置。
3. 确认输出、checkpoint spool、最终发布 spool 和媒体临时目录有足够空间。
4. 先用 `--max-records 1 --dry-run` 验证输入与配置，再做一个真实 episode canary。
5. 长任务放在具名 `tmux` session 中；不要同时对同一 task/output 启动两个 writer。

本地自检不会发送任何请求：

```bash
python scripts/check_install.py
```

## 普通运行

```bash
bash scripts/run_pipeline.sh INPUT OUTPUT \
  --tasks subtask,ecot,grd,sta,cpa \
  --api las \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
```

如果 `--tasks` 不包含 `subtask` 但包含 GRD、STA 或 CPA，必须增加：

```text
--subtask-path /path/to/subtask/output
```

## 高并发模板

`scripts/run_las_high_concurrency.sh` 复现当前生产链路的关键约束：LAS Submit/Poll、共享
帧池、独立媒体池、immutable JSONL 和有界写入。默认值面向大内存机器，但仍采用 2048 的
保守 HTTP 上限：

```bash
tmux new-session -d -s vqa-ecot \
  "cd /path/to/VQA-anno-clean && \
   bash scripts/run_las_high_concurrency.sh INPUT OUTPUT ecot \
   --ecot-video-transport cos-presigned \
   --ecot-image-transport cos-presigned 2>&1 | tee -a run-ecot.log"
```

在已经验证媒体准备、内存和供应商额度均有余量的 100+ GiB 容器中，可显式扩大：

```bash
HTTP_CONCURRENCY=8192 \
FRAME_WORKERS=2048 \
VIDEO_WORKERS=128 \
LAS_REQUEST_WORKERS=8192 \
LAS_OPERATOR_CONCURRENCY=8192 \
REQUEST_MEMORY_MIB=81920 \
bash scripts/run_las_high_concurrency.sh INPUT OUTPUT grd \
  --subtask-path /path/to/subtask/output
```

STA 通常比 GRD 更容易受完整 Subtask clip 转码限制。先观察 CPU 和 FFmpeg 数量，再增加
`MEDIA_WORKERS`/`CLIP_WORKERS`；盲目增加 HTTP 上限不会绕过转码瓶颈。

## 恢复与重启

正常重启保持相同的 INPUT、OUTPUT、任务集合和 Subtask 来源即可。不要删除：

- `OUTPUT/_state/annotation-unit-checkpoints/`；
- `OUTPUT/_state/grd-window-checkpoints/`；
- `OUTPUT/_state/las-pipeline/`；
- 指定的 checkpoint/final-publication spool。

可以清理已经确认没有进程打开的临时转码目录，但它只应包含可再生媒体。不得把输出根、
checkpoint 根或 spool 根当作临时目录清理。

内存墙后的 supervisor 应重启同一命令并复用以上状态，而不是换一个输出目录。若存在
`provider-fatal-stop.json`、`grd-review-fatal-stop.json` 等 fatal marker，先检查原因；不要
循环重启掩盖鉴权、欠费或 contract 错误。

## 监控信号

至少同时观察：

- durable units 的新增速率，而不只是 request started；
- HTTP active/peak/queued、响应中位延迟和 retry 类型；
- LAS Submit/Poll 的失败码，特别是 429、鉴权和余额错误；
- RSS、cgroup memory、page cache、临时盘占用；
- FFmpeg 子进程数量与 CPU；
- checkpoint outbox pending 和 sync error。

`rate_limit_events` 需来自结构化计数，不能通过包含 `rate_limit` 字样的 heartbeat 配置项
判断。高 active 但 durable 不增长通常是供应商/响应解析问题；低 active 且 CPU/FFmpeg
繁忙通常是媒体准备问题；低 active 且 CPU 空闲通常是 episode/batch/request 供给不足。

## GRD 存量审核

```bash
python scripts/audit_grounding_records.py \
  --input /path/to/output/grd \
  --in-place \
  --checkpoint-root /path/to/output/_state/grd-inventory-reviews \
  --workers 128 \
  --episode-workers 8 \
  --max-http-active 128
```

该命令执行语言审核、必要时仅修改歧义物体的 `name`/`alternate_name`、再次审核，并拒绝
仍不可区分的帧。它不会修改 bbox、物体数量或物体顺序。
