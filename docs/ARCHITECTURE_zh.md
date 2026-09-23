# 流水线架构与数据流

本文描述当前生产实现，不包含历史探测链路。主入口是
`scripts/annotate_videos.py`，五个阶段共享同一套输入解析、请求准入、媒体缓存、恢复记录和
最终发布机制。

## 数据流

```text
video / directory / JSONL / supported WebDataset
                         |
                         v
              episode + main camera
                         |
            +------------+-------------+
            |                          |
            v                          v
   LAS Subtask segmentation     ECoT teacher at 0.5 FPS
            |                          |
            +------------+-------------+
                         v
             GRD on every 2 FPS frame
                         |
                         v
              STA contact proposals
                         |
                         v
        CPA review + exact contact frame
                         |
              +----------+----------+
              |                     |
              v                     v
  STA pre-contact bbox/TTC   CPA crop points + SAM3 snap
```

Subtask、GRD、STA 和 CPA 之间按 `record_uid` 与 source/batch 身份关联。ECoT 使用全局任务
指令，不依赖 Subtask 内容。只运行下游阶段时，必须用 `--subtask-path` 提供同一输出格式的
Subtask 记录。

## 供应商路由

当前大批量路径使用 `--api las`。客户端只调用 LAS 的 Submit/Poll：

```text
POST https://operator.las.cn-beijing.volces.com/api/v1/submit
POST https://operator.las.cn-beijing.volces.com/api/v1/poll
```

Submit 的 `data` 使用 `video_url`、`query`、`fps`、`model_name` 和 `ark_api_key`。
`LAS_API_KEY` 鉴权 LAS；`ARK_API_KEY` 由 LAS 算子转交客户自己的方舟推理服务。这里不调用
`/api/v3/responses` 或 `/api/v3/chat/completions`。

模型默认值：

| 阶段 | 模型 |
|---|---|
| Subtask 主分析 | `doubao-seed-2-1-pro-260628` |
| Subtask 英文后处理 | `doubao-seed-2-0-lite-260428` |
| ECoT | `doubao-seed-2-0-lite-260215` |
| GRD | `doubao-seed-2-1-pro-260628` |
| GRD 名称审核/修复 | `doubao-seed-2-1-pro-260628` |
| STA | `doubao-seed-2-0-lite-260215` |
| CPA | `doubao-seed-2-1-pro-260628` |

Ark、DashScope 和通用 OpenAI-compatible 路径仍被保留，用于小规模运行或回放已有结果。

## 并发模型

并发不是一个单独的 worker 数：

- `--max-http-active`：请求层上限；
- `--shared-frame-workers`：进程级共享帧任务池，避免每个 episode 创建一套线程；
- `--workers` / `--max-record-active`：同时驻留的 episode 数；
- `--video-prepare-workers`、`--video-clip-workers`：转码与媒体准备；
- `--batch-workers`：输入 catalog 扫描与供给；
- `--durable-write-workers`：checkpoint/最终结果发布；
- `--max-pending`：已提交但尚未完成的本地工作上限。

设置 `--max-http-active 8192` 不等于会产生 8192 个在途请求。请求供给还受共享帧池、驻留
episode、视频转码、供应商延迟和内存预算约束。高并发路径必须使用进程级共享帧池；不要把
每个 episode 的局部帧池都设为 8192。

## 持久化边界

Subtask 按 episode，ECoT 按目标帧，GRD 按 2 FPS 帧，STA/CPA 按候选事件和最终观察帧
持久化。恢复时优先读取 unit checkpoint，再读取最终 immutable JSONL；已成功单位不会重新
请求。最终记录采用 `unified-vqa-record/v2`，请求日志、签名 URL、API key 和本地临时媒体
不进入公开结果。
