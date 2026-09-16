# VQA 视频标注流水线

[English](README.md)

本仓库是可恢复、可并行的五阶段生产标注代码，整理自当前实际运行的
`VQA_subtask-ecot-grd-sta-cpa`。仓库只保留生产入口、运行依赖和必要的存量审核工具，
不包含数据集、模型权重、密钥、运行日志、网页预览、探测脚本或压测脚本。

主入口：`scripts/annotate_videos.py`  
通用包装：`scripts/run_pipeline.sh`  
按环境变量运行：`scripts/run_api.sh`  
LAS 高并发模板：`scripts/run_las_high_concurrency.sh`

当前实现说明见 [架构与数据流](docs/ARCHITECTURE_zh.md)，部署、恢复和监控见
[生产运行手册](docs/OPERATIONS_zh.md)，ECoT 的正式四字段提示词见
[ECoT prompt contract](docs/ECOT_PROMPT_CONTRACT.md)。整理日期：2026-09-16。

## 当前五阶段

| 阶段 | 当前逻辑 | 默认 Seed 模型 |
|---|---|---|
| Subtask | vendored `doubao_las_annotation` 对完整视频切分；相邻且时间连续、规范化 skill 与 description 均相同的段会合并 | Step 1/3: `doubao-seed-2-1-pro-260628`；英文后处理: `doubao-seed-2-0-lite-260428` |
| ECoT | 在 2 FPS 目标网格上默认每 4 帧选一个目标，即每 2 秒一条；teacher 是完整 0.5 FPS episode 视频 | `doubao-seed-2-0-lite-260215` |
| GRD | 标注 2 FPS 网格的每一帧；当前帧 inventory 最多 4 个物体；只有首个操作物体选择会看 3 FPS × 4 秒未来视频 | `doubao-seed-2-0-pro-260215` |
| STA | 使用 CPA 审核并精确选定的最终接触帧；从此前且无其他最终接触的窗口随机取帧，给出事件名后框目标并计算 TTC | `doubao-seed-2-0-lite-260215` |
| CPA | 审核事件、选择精确接触帧；在扩大 15% 的 bbox crop 中选点，先吸附到 crop SAM3 mask，再映射回原图并二次吸附 | `doubao-seed-2-0-pro-260215` |

GRD inventory 的“审核 → 必要时改名 → 再审核”默认使用
`doubao-seed-2-1-pro-260628`。修正只能调整歧义物体的 `name` 和
`alternate_name`，不能改 bbox、数量或顺序；再审仍不通过的帧不会注册为训练任务。

## 供应商链路

`--api` 支持 `las`、`ark`、`dashscope` 和通用 OpenAI-compatible 服务。当前大批量
生产链路使用 `las`；Subtask 也始终使用 vendored LAS 实现。

### `--api las`：当前批量生产链路

下游请求通过 LAS 的 `las_long_video_understand/v1` Submit/Poll 接口：

```text
POST https://operator.las.cn-beijing.volces.com/api/v1/submit
POST https://operator.las.cn-beijing.volces.com/api/v1/poll
```

Submit 的核心字段为：

```json
{
  "operator_id": "las_long_video_understand",
  "operator_version": "v1",
  "data": {
    "video_url": "https://<private-cos-object>?<temporary-signature>",
    "query": "<prompt>",
    "fps": 0.5,
    "model_name": "doubao-seed-2-0-lite-260215",
    "ark_api_key": "<ARK_API_KEY>"
  }
}
```

LAS 鉴权使用 `LAS_API_KEY`，算子内部调用客户方舟服务使用 `ARK_API_KEY`。
媒体先上传到私有 COS，再生成短期签名 HTTPS URL；签名 URL 和密钥不会写入最终记录或日志。

- ECoT：上传一份完整 0.5 FPS teacher 视频；每个目标请求只复用同一视频 URL，并用精确时间戳和 teacher 帧序号定位目标，不再逐目标上传 JPEG，也不再为每一帧重新拼接视频。
- GRD/STA：单图、未来片段或接触片段会按算子要求编码成 MP4，再通过 `video_url` 提交。
- 算子采样 FPS：ECoT 为 0.5，GRD/STA 为 2，CPA 类请求为 5。
- 小于算子最小时长的视频会补最后一帧到 1.2 秒，避免 `DurationTooShort`。

### `--api ark` 与 `--api dashscope`

- Ark 默认端点：`https://ark.cn-beijing.volces.com/api/v3/chat/completions`
- DashScope 默认端点：`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`

Subtask 始终使用 vendored LAS 流程，不随下游 `--api` 切换。

## 阶段细节

### ECoT

目标时间轴仍以 2 FPS 定义，默认 `--ecot-interval 4`。LAS 模式直接把目标映射到
0.5 FPS teacher 中的对应时间；非 LAS 模式仍可发送 teacher 视频和独立目标图。
输出字段为 `scene_description`、`task_progress`、`current_subtask` 和
`atomic_action`。`atomic_action` 只要求非空、小写、无控制字符且不超过 240 字符；
不再强制“一句话”、至少两个英文词或禁止 `then`。

### GRD

每个 2 FPS 当前帧依次执行：

1. 只输入当前帧与当前 Subtask，产生不超过 4 个可区分任务相关物体的 inventory。
2. 输入 inventory、Subtask 与未来 3 FPS × 4 秒视频，选择首先操作的物体。
3. 对 inventory 做语言可分辨性审核；初审失败才输入当前图片和固定 bbox 做一次最小改名。
4. 再审仍失败则 reject 该帧；成功帧逐帧持久化，重启时直接复用。

### STA 与 CPA 接触帧

STA proposal 不是最终接触帧。CPA 先审核接触事件，再在候选窗口内选择精确帧；最终
STA 和 CPA 都使用这份 CPA-final contact。STA 在前置窗口中随机取帧，并确保观察帧到
本次接触之间不存在其他最终接触；标注模型看到单图和显式事件名，输出 noun、verb、
TTC 与目标 bbox。

### CPA 接触点

VLM 在扩大 15% 的交互 bbox crop 中选择 `H_i/O_i`。随后 SAM3 在 crop mask 上吸附，
映射到原图后再在完整帧 mask 上吸附。可选 CoTracker 反向追踪使用有界、进程级 tracker
池，默认最多两个副本，可用 `CPA_TRACKER_REPLICAS=1..4` 调整。

## 输入格式

`--input` 支持：

- 单个视频：`.mp4`、`.mov`、`.mkv`、`.webm`、`.avi`、`.m4v`；
- 递归视频目录：会进入内层目录，忽略其他后缀文件；
- JSONL manifest：每行至少包含唯一 `uid`/`record_uid` 和 `video_path`；相对路径以 manifest 所在目录解析；
- 支持的 unified VQA WebDataset；
- 带 `RELEASE.json` 与 Parquet catalog 的 Cosmos3 physical WebDataset。

最小 manifest：

```json
{"uid":"episode-001","video_path":"videos/001.mp4","task_name":"put the cup in the sink","source_key":"demo"}
```

Physical WebDataset 每个 episode 选择一个主视角，优先级为 `ego`、`head_left`、
`head_right`、`left_wrist`、`right_wrist`，然后才是其他视角。

## 安装与配置

建议 Python 3.10+，系统需提供 FFmpeg；CPA 本地模型需要可用 NVIDIA GPU。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
cp configs/api.env.example configs/api.env
```

编辑 `configs/api.env` 后加载：

```bash
set -a
source configs/api.env
set +a
```

不发送网络请求的本地自检：

```bash
python scripts/check_install.py
```

Linux 高并发运行建议预编译三个可选 native helper：

```bash
PYTHONPATH=scripts python scripts/native_http_metrics.py --build
PYTHONPATH=scripts python scripts/native_availability_latch.py --build
PYTHONPATH=scripts python scripts/native_ecot_gate.py --build
```

LAS customer-Ark 模式至少需要：

- `LAS_API_KEY` 与 `ARK_API_KEY`；
- 已登录且可执行的 `coscli`；
- `VQA_COS_BUCKET`、`VQA_COS_ENDPOINT`、`VQA_COS_REGION`；
- `VQA_LAS_OPERATOR_VIDEO_PREFIX`，以及 ECoT 的 `VQA_COS_VIDEO_PREFIX`；
- 如使用 LAS CPA 选点，还需 `CPA_LAS_COS_PREFIX`。

SAM3/CoTracker 不随仓库分发，通过 `SAM3_REPO`、`SAM3_CHECKPOINT`、
`COTRACKER_REPO`、`COTRACKER_CHECKPOINT` 配置。

## 运行示例

完整五阶段：

```bash
bash scripts/run_pipeline.sh /path/to/video.mp4 ./outputs/demo \
  --tasks subtask,ecot,grd,sta,cpa \
  --task-instruction "put the cup into the sink" \
  --api las \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
```

只生成 Subtask：

```bash
bash scripts/run_pipeline.sh /path/to/videos ./outputs/subtasks \
  --tasks subtask --workers 64
```

复用已有 Subtask，单独运行 GRD：

```bash
bash scripts/run_pipeline.sh /path/to/videos ./outputs/grd \
  --tasks grd --subtask-path ./outputs/subtasks/subtask \
  --api las --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit \
  --grd-review-api las \
  --grd-review-endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit
```

LAS ECoT 的单 URL 模式：

```bash
bash scripts/run_pipeline.sh /path/to/videos ./outputs/ecot \
  --tasks ecot --api las \
  --endpoint https://operator.las.cn-beijing.volces.com/api/v1/submit \
  --ecot-video-transport cos-presigned \
  --ecot-image-transport cos-presigned \
  --immutable-jsonl
```

`--tasks` 接受 `subtask,ecot,grd,sta,cpa` 的逗号分隔子集，`grounding` 是 `grd`
别名。选择 GRD、STA 或 CPA 但不同时执行 Subtask 时，必须提供 `--subtask-path`。
完整参数见：

```bash
python scripts/annotate_videos.py --help
```

## 输出、恢复与高并发

```text
OUTPUT/<task>/shards/<source>/<batch>.jsonl
OUTPUT/errors/pipeline/<source>/<batch>.jsonl
OUTPUT/_state/annotation-unit-checkpoints/<task>/...
OUTPUT/_state/grd-window-checkpoints/...
OUTPUT/_state/las-pipeline/...
```

公开结果均为 `unified-vqa-record/v2`。当前 contract：

| Task | Contract ID |
|---|---|
| Subtask | `vqa-anno-raw-subtask/v2` |
| ECoT | `vqa-anno-raw-ecot-privileged-teacher-0.5fps-atomic/v3` |
| GRD | `vqa-anno-raw-grd-2fps-inventory-future-first-object/v3` |
| STA | `vqa-anno-raw-sta-cpa-final-random-event-bbox/v5` |
| CPA | `vqa-anno-cpa-object-only-two-decimal/v13` |

生产实现包含：逐帧/逐事件 checkpoint、immutable JSONL、本地 SQLite outbox、异步云端
同步、只读并行 resume 校验、进程级共享请求/事件池、有界媒体准备和有界内存缓存。
`--max-http-active 8192` 只是上限；有效并发还受媒体准备、`--max-record-active`、
`--batch-workers`、供应商延迟和机器资源约束。record executor 会把全局 record 上限按
batch worker 分摊，避免每个 batch 重复创建完整线程池。

高并发运行必须设置进程级 `--shared-frame-workers`，避免每个 episode 创建独立的大型
线程池。当前生产模板还显式设置 `--thread-stack-kib`、媒体/clip worker、resume validator
和 durable writer。可从保守配置开始：

```bash
bash scripts/run_las_high_concurrency.sh INPUT OUTPUT ecot \
  --ecot-video-transport cos-presigned \
  --ecot-image-transport cos-presigned
```

扩到 8192 前请阅读 [生产运行手册](docs/OPERATIONS_zh.md)；HTTP 上限不是实际并发或吞吐量
承诺。

已有 GRD 可以通过 `scripts/audit_grounding_records.py` 单独执行审核/修复/再审核。
第三方来源见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。
