"""Data contracts for pipeline inputs and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
import re


class PipelineDataError(ValueError):
    """Raised when an input or LAS result violates the pipeline contract."""


@dataclass(frozen=True)
class AnnotationTask:
    task: str
    episode_id: str
    episode_index: int
    wrist_video_url: str | None
    main_video_url: str
    embodiment: str = "robot"
    task_instruction: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.task):
            raise PipelineDataError(f"Invalid task name: {self.task!r}.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.episode_id):
            raise PipelineDataError(f"Invalid episode id: {self.episode_id!r}.")
        if self.episode_index < 0:
            raise PipelineDataError("episode_index must be non-negative.")
        if self.embodiment not in {"robot", "human"}:
            raise PipelineDataError(
                "embodiment must be either 'robot' or 'human'."
            )
        if not isinstance(self.main_video_url, str) or not self.main_video_url.strip():
            raise PipelineDataError("main_video_url must be a non-empty string.")
        if self.wrist_video_url is not None and (
            not isinstance(self.wrist_video_url, str)
            or not self.wrist_video_url.strip()
        ):
            raise PipelineDataError(
                "wrist_video_url must be a non-empty string or None."
            )
        if self.task_instruction is not None and not self.task_instruction.strip():
            raise PipelineDataError(
                "task_instruction must be a non-empty string or None."
            )

    @property
    def identifier(self) -> str:
        return f"{self.task}/{self.episode_id}"

    @property
    def instruction(self) -> str:
        return self.task_instruction or self.task

    @property
    def has_wrist_view(self) -> bool:
        return self.wrist_video_url is not None


@dataclass(frozen=True)
class SampleResult:
    task: str
    episode_id: str
    status: str
    error: str | None = None

    @property
    def identifier(self) -> str:
        return f"{self.task}/{self.episode_id}"
