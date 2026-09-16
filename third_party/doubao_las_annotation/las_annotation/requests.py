"""LAS request-body construction for annotation stages."""

from __future__ import annotations

from typing import Any

from .config import OBJECT_TEMPLATE, PipelineConfig
from .models import AnnotationTask


def build_object_request(
    task: AnnotationTask,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    config = config or PipelineConfig()
    if not task.has_wrist_view:
        raise ValueError("Object detection requires a wrist video URL.")
    task_name = task.instruction
    actor_context = ""
    if task.embodiment == "human":
        actor_context = (
            "【动作主体】\n"
            "这是人类佩戴相机完成操作的视频。请将画面中的操作主体描述为"
            "左手、右手或双手，不要描述成机器人、机械臂、机械手、夹爪或"
            "末端执行器。\n\n"
        )
    prompt_context = f"""【任务名称】
{task_name}

{actor_context}任务名称中的目标物体类别可视为可靠的命名。对于与任务目标对应的被操作物体，应优先使用任务名称中的具体类别名称，不得退化为更宽泛的类别。若某物体在画面中明显属于其他类别，则以画面为准。

"""
    return {
        "operator_id": config.operator_id,
        "operator_version": config.operator_version,
        "data": {
            "video_url": task.wrist_video_url,
            "task_template": OBJECT_TEMPLATE,
            "model_name": config.step1_model or config.model,
            "task_context": {"prompt_context": prompt_context},
        },
    }


def build_action_request(
    task: AnnotationTask,
    prompt_context: str,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    config = config or PipelineConfig()
    return {
        "operator_id": config.operator_id,
        "operator_version": config.operator_version,
        "data": {
            "video_url": task.main_video_url,
            "task_template": config.action_template,
            "task_context": {"prompt_context": prompt_context},
            "model_name": config.model,
        },
    }
