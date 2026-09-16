"""Configuration shared by the annotation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os

from .postprocessing import DEFAULT_POSTPROCESS_MODEL
from .review_video import (
    DEFAULT_REVIEW_FONT,
    DEFAULT_REVIEW_TARGET_FPS,
    DEFAULT_REVIEW_VIDEO_ROOT,
    DEFAULT_REVIEW_VIDEO_STREAM,
)


DEFAULT_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_OPERATOR_ID = "las_long_video_understand"
DEFAULT_OPERATOR_VERSION = "v1"
DEFAULT_ACTION_TEMPLATE = "embodied_action_captioning_v2"
OBJECT_TEMPLATE = "embodied_active_object_detection"
DEFAULT_MAPPING_PATH = (
    "/mnt/project/subtask_annotation_abc130k/"
    "abc130k-2000h_episode_index_mapping.json"
)
DEFAULT_VIDEO_BASE_URL = "tos://abc-130k/abc-130k-doubao-no-anno-new/"
DEFAULT_MAIN_STREAM = "observation.images.cam_high"
DEFAULT_WRIST_STREAM = "observation.images.cam_wrist"


@dataclass(frozen=True)
class PipelineConfig:
    """Business-level LAS request configuration."""

    model: str = DEFAULT_MODEL # pyright: ignore[reportUndefinedVariable]
    step1_model: str | None = None
    action_template: str = DEFAULT_ACTION_TEMPLATE
    operator_id: str = DEFAULT_OPERATOR_ID
    operator_version: str = DEFAULT_OPERATOR_VERSION
    skill_profile: str = "66"
    skills_file: str | None = None
    postprocess: bool = False
    rewrite_descriptions: bool = True
    merge_none_segments: bool = True
    postprocess_model: str = DEFAULT_POSTPROCESS_MODEL
    render_review_videos: bool = False # type: ignore
    review_source_kind: str = "mcap"
    review_video_root: str = DEFAULT_REVIEW_VIDEO_ROOT
    review_video_stream: str = DEFAULT_REVIEW_VIDEO_STREAM
    review_fps: float = DEFAULT_REVIEW_TARGET_FPS
    review_font: str = DEFAULT_REVIEW_FONT
    review_font_size: int = 22
    review_crf: int = 23
    review_preset: str = "veryfast" # pyright: ignore[reportUndefinedVariable]
    stop_after: str | None = None


@dataclass(frozen=True)
class VideoAssetConfig:
    """ABC130K TOS video layout shared by all input episodes."""

    base_url: str = os.getenv("LAS_VIDEO_BASE_URL", DEFAULT_VIDEO_BASE_URL)
    main_stream: str = DEFAULT_MAIN_STREAM
    wrist_stream: str = DEFAULT_WRIST_STREAM
    episodes_per_chunk: int = 1000

    def video_url(self, task: str, episode_index: int, stream: str) -> str:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative.") # pyright: ignore[reportUndefinedVariable]
        if self.episodes_per_chunk <= 0:
            raise ValueError("episodes_per_chunk must be greater than 0.")
        chunk = episode_index // self.episodes_per_chunk
        return (
            f"{self.base_url.rstrip('/')}/{task}/videos/chunk-{chunk:03d}/"
            f"{stream}/episode_{episode_index:06d}.mp4"
        )
