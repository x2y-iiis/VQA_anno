"""Render hard-subtitled review videos from MCAP or indexed local MP4 data."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .mcap_video import (
    DEFAULT_MCAP_CAMERA_TOPIC,
    DEFAULT_MCAP_ROOT,
    DEFAULT_MCAP_TARGET_FPS,
    mcap_source_signature,
    normalize_camera_topic,
    normalize_target_fps,
    render_mcap_video,
    resolve_episode_mcap,
    verify_mcap_video,
)
from .models import AnnotationTask, PipelineDataError
from .postprocessing import annotation_digest


# Retain the historical names as API/CLI aliases. They now point to raw MCAP,
# never to the fixed-rate MP4 export.
DEFAULT_REVIEW_VIDEO_ROOT = DEFAULT_MCAP_ROOT
DEFAULT_REVIEW_VIDEO_STREAM = DEFAULT_MCAP_CAMERA_TOPIC
DEFAULT_REVIEW_TARGET_FPS = DEFAULT_MCAP_TARGET_FPS
DEFAULT_REVIEW_FONT = "Noto Sans CJK SC"
# Keep the historical generation number while a long-running scheduler from
# the previous deployment is still active. The MCAP renderer/policy fields
# below are the authoritative cache invalidators.
REVIEW_RENDER_VERSION = 2
MP4_REVIEW_RENDER_VERSION = 1
REVIEW_SOURCE_KINDS = ("mcap", "mp4")


@dataclass(frozen=True)
class ReviewVideoConfig:
    font: str = DEFAULT_REVIEW_FONT
    font_size: int = 22
    crf: int = 23
    preset: str = "veryfast"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    target_fps: float = DEFAULT_REVIEW_TARGET_FPS
    camera_topic: str = DEFAULT_REVIEW_VIDEO_STREAM
    source_kind: str = "mcap"

    def __post_init__(self) -> None:
        if not self.font.strip():
            raise ValueError("review font must be a non-empty string.")
        if self.font_size <= 0:
            raise ValueError("review font size must be greater than 0.")
        if not 0 <= self.crf <= 51:
            raise ValueError("review CRF must be between 0 and 51.")
        if self.source_kind not in REVIEW_SOURCE_KINDS:
            raise ValueError(
                f"review source kind must be one of {REVIEW_SOURCE_KINDS}."
            )
        if self.source_kind == "mcap":
            normalize_target_fps(self.target_fps)
            normalize_camera_topic(self.camera_topic)
        elif not self.camera_topic.strip():
            raise ValueError("review MP4 stream must be a non-empty string.")


def review_source_path(
    task: AnnotationTask,
    root: str | Path = DEFAULT_REVIEW_VIDEO_ROOT,
    stream: str = DEFAULT_REVIEW_VIDEO_STREAM,
    source_kind: str = "mcap",
) -> Path:
    if source_kind == "mcap":
        # Camera selection is performed by ReviewVideoConfig.
        return resolve_episode_mcap(task.task, task.episode_id, root)
    if source_kind == "mp4":
        if not stream.strip():
            raise ValueError("review MP4 stream must be a non-empty string.")
        return (
            Path(root)
            / task.task
            / "videos"
            / f"chunk-{task.episode_index // 1000:03d}"
            / stream
            / f"episode_{task.episode_index:06d}.mp4"
        )
    raise ValueError(f"Unsupported review source kind: {source_kind!r}.")


def render_review_video(
    source: str | Path,
    segments: Sequence[Mapping[str, Any]],
    destination: str | Path,
    config: ReviewVideoConfig | None = None,
) -> None:
    config = config or ReviewVideoConfig()
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Review source {config.source_kind.upper()} does not exist: "
            f"{source_path}"
        )
    _validate_segments(segments)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    subtitle_path = destination_path.with_name(f".{destination_path.stem}.{token}.ass")
    try:
        subtitle_path.write_text(
            build_ass_subtitles(segments, config.font, config.font_size),
            encoding="utf-8-sig",
        )
        if config.source_kind == "mp4":
            _render_subtitled_mp4(
                source_path,
                destination_path,
                subtitle_path,
                config,
            )
        else:
            render_mcap_video(
                source_path,
                destination_path,
                target_fps=config.target_fps,
                preferred_topic=config.camera_topic,
                ass_path=subtitle_path,
                crf=config.crf,
                preset=config.preset,
                ffmpeg=config.ffmpeg,
                ffprobe=config.ffprobe,
            )
    finally:
        subtitle_path.unlink(missing_ok=True)


def build_ass_subtitles(
    segments: Sequence[Mapping[str, Any]],
    font: str = DEFAULT_REVIEW_FONT,
    font_size: int = 22,
) -> str:
    _validate_segments(segments)
    safe_font = font.replace(",", " ").strip()
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 640",
        "PlayResY: 480",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{safe_font},{font_size},&H00FFFFFF,&H000000FF,"
        "&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,2,1,2,28,28,24,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for segment in segments:
        start = float(segment["start"])
        end = max(float(segment["end"]), start + 0.01)
        skill = segment.get("skill")
        label = "None" if skill is None else str(skill).strip()
        description = _escape_ass_text(str(segment["description"]).strip())
        text = f"[{_escape_ass_text(label)}] {description}"
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},"
            f"Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def review_signature(
    annotation: Mapping[str, Any],
    source: str | Path,
    config: ReviewVideoConfig,
) -> dict[str, Any]:
    source_path = Path(source)
    if config.source_kind == "mp4":
        stat = source_path.stat()
        return {
            "annotation_digest": annotation_digest(annotation),
            "source_kind": "local_mp4",
            "source": str(source_path.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "font": config.font,
            "font_size": config.font_size,
            "crf": config.crf,
            "preset": config.preset,
            "render_version": MP4_REVIEW_RENDER_VERSION,
        }
    source_signature = mcap_source_signature(
        source_path,
        target_fps=config.target_fps,
        preferred_topic=config.camera_topic,
        crf=config.crf,
        preset=config.preset,
    )
    return {
        "annotation_digest": annotation_digest(annotation),
        **source_signature,
        "font": config.font,
        "font_size": config.font_size,
        "render_version": REVIEW_RENDER_VERSION,
    }


def review_artifact_matches(
    manifest: object,
    key: str,
    annotation: object,
    source: str | Path,
    destination: str | Path,
    config: ReviewVideoConfig,
) -> bool:
    if not isinstance(manifest, Mapping) or not isinstance(annotation, Mapping):
        return False
    record = manifest.get(key)
    if not isinstance(record, Mapping):
        return False
    signature = record.get("signature")
    if not isinstance(signature, Mapping):
        return False
    output = Path(destination)
    try:
        expected = review_signature(annotation, source, config)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        dict(signature) == expected
        and record.get("output") == output.name
        and output.is_file()
        and output.stat().st_size > 0
    )


def _render_subtitled_mp4(
    source: Path,
    destination: Path,
    subtitle: Path,
    config: ReviewVideoConfig,
) -> None:
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary = destination.with_name(
        f".{destination.stem}.{token}.tmp.mp4"
    )
    try:
        command = [
            config.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            f"ass=filename={_escape_filter_path(subtitle)}",
            "-c:v",
            "libx264",
            "-preset",
            config.preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        _run_command(command, "ffmpeg review render")
        _verify_playable_mp4(temporary, config.ffprobe)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _run_command(command: Sequence[str], label: str) -> None:
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:] or "unknown error"
        raise RuntimeError(
            f"{label} failed with code {result.returncode}: {detail}"
        )


def _verify_playable_mp4(path: Path, ffprobe: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Rendered review video is empty: {path}")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:] or "unknown error"
        raise RuntimeError(
            f"ffprobe failed with code {result.returncode}: {detail}"
        )
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"ffprobe found no playable video stream in {path}"
        ) from exc
    if stream.get("codec_name") != "h264" or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Rendered review video is not a playable H.264 MP4: {path}"
        )


def _escape_filter_path(path: Path) -> str:
    value = str(path.resolve())
    return (
        value.replace("\\", r"\\")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(",", r"\,")
    )


def _validate_segments(segments: Sequence[Mapping[str, Any]]) -> None:
    if not segments:
        raise PipelineDataError("Review video requires at least one segment.")
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise PipelineDataError(f"segments[{index}] must be an object.")
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineDataError(
                f"segments[{index}] requires numeric start/end values."
            ) from exc
        if start < 0 or end < start:
            raise PipelineDataError(f"segments[{index}] has an invalid time range.")
        description = segment.get("description")
        if not isinstance(description, str) or not description.strip():
            raise PipelineDataError(f"segments[{index}].description is required.")


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"


def _escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\n", r"\N")
        .replace("\r", r"\N")
    )


def verify_review_video(
    source: str | Path,
    output: str | Path,
    ffprobe: str = "ffprobe",
    *,
    target_fps: float = DEFAULT_REVIEW_TARGET_FPS,
    camera_topic: str = DEFAULT_REVIEW_VIDEO_STREAM,
) -> None:
    """Require exact timestamp-derived target FPS and a non-shortened tail."""

    verify_mcap_video(
        source,
        output,
        target_fps=target_fps,
        preferred_topic=camera_topic,
        ffprobe=ffprobe,
    )


__all__ = [
    "DEFAULT_REVIEW_FONT",
    "DEFAULT_REVIEW_TARGET_FPS",
    "DEFAULT_REVIEW_VIDEO_ROOT",
    "DEFAULT_REVIEW_VIDEO_STREAM",
    "MP4_REVIEW_RENDER_VERSION",
    "REVIEW_RENDER_VERSION",
    "REVIEW_SOURCE_KINDS",
    "ReviewVideoConfig",
    "build_ass_subtitles",
    "render_review_video",
    "review_artifact_matches",
    "review_signature",
    "review_source_path",
    "verify_review_video",
]
