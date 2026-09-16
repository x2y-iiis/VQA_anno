"""Render playable CFR video directly from timestamped MCAP camera messages."""

from __future__ import annotations

import bisect
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


DEFAULT_MCAP_ROOT = "/mnt/abc130k/data"
DEFAULT_MCAP_CAMERA_TOPIC = "/top-camera"
DEFAULT_MCAP_TARGET_FPS = 30.0
TOP_CAMERA_TOPICS = (
    "/top-camera",
    "/top-left-camera",
    "/top-right-camera",
)
MCAP_VIDEO_RENDER_VERSION = 1
MCAP_TIMESTAMP_POLICY = "protobuf_timestamp_then_log_time_v1"
MCAP_SAMPLING_VERSION = "nearest_target_grid_include_last_v1"

_LEGACY_VIDEO_ROOT = Path("/mnt/abc130k-2000h")
_LEGACY_VIDEO_STREAM = "observation.images.cam_high"
_SAFE_CACHE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class McapVideoInfo:
    """The source-camera timeline used for one render."""

    source: Path
    topic: str
    codec: str
    timestamp_source: str
    source_frame_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    width: int = 0
    height: int = 0

    @property
    def timestamp_span_ns(self) -> int:
        return self.last_timestamp_ns - self.first_timestamp_ns

    @property
    def timestamp_span_seconds(self) -> float:
        return self.timestamp_span_ns / 1_000_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source.resolve()),
            "topic": self.topic,
            "codec": self.codec,
            "timestamp_source": self.timestamp_source,
            "source_frame_count": self.source_frame_count,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "timestamp_span_ns": self.timestamp_span_ns,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McapVideoInfo":
        return cls(
            source=Path(str(value["source"])),
            topic=str(value["topic"]),
            codec=str(value["codec"]),
            timestamp_source=str(value["timestamp_source"]),
            source_frame_count=int(value["source_frame_count"]),
            first_timestamp_ns=int(value["first_timestamp_ns"]),
            last_timestamp_ns=int(value["last_timestamp_ns"]),
            width=int(value.get("width", 0)),
            height=int(value.get("height", 0)),
        )


@dataclass(frozen=True)
class McapRenderResult:
    """Metadata for a verified MCAP-derived MP4."""

    path: Path
    info: McapVideoInfo
    fps_numerator: int
    fps_denominator: int
    output_frame_count: int
    output_duration_seconds: float

    @property
    def target_fps(self) -> float:
        return self.fps_numerator / self.fps_denominator

    @property
    def target_fps_text(self) -> str:
        return _fraction_text(
            Fraction(self.fps_numerator, self.fps_denominator)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path.resolve()),
            "source_video": self.info.as_dict(),
            "target_fps": self.target_fps_text,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
            "output_frame_count": self.output_frame_count,
            "output_duration_seconds": self.output_duration_seconds,
            "render_version": MCAP_VIDEO_RENDER_VERSION,
            "timestamp_policy": MCAP_TIMESTAMP_POLICY,
            "sampling_version": MCAP_SAMPLING_VERSION,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, path: Path | None = None
    ) -> "McapRenderResult":
        source_video = value.get("source_video")
        if not isinstance(source_video, Mapping):
            raise ValueError("Cached MCAP video metadata has no source_video.")
        return cls(
            path=path or Path(str(value["path"])),
            info=McapVideoInfo.from_dict(source_video),
            fps_numerator=int(value["fps_numerator"]),
            fps_denominator=int(value["fps_denominator"]),
            output_frame_count=int(value["output_frame_count"]),
            output_duration_seconds=float(value["output_duration_seconds"]),
        )


@dataclass(frozen=True)
class _OutputProbe:
    width: int
    height: int
    codec: str
    pixel_format: str
    frame_rate: Fraction
    packet_count: int
    stream_duration: float
    container_duration: float


def normalize_mcap_root(root: str | Path) -> Path:
    """Translate the retired MP4 root and accept either abc130k or its data dir."""

    path = Path(root).expanduser()
    if path == _LEGACY_VIDEO_ROOT:
        return Path(DEFAULT_MCAP_ROOT)
    if path.name == "data":
        return path
    data_path = path / "data"
    if data_path.is_dir() or path.name == "abc130k":
        return data_path
    return path


def normalize_camera_topic(topic: str) -> str:
    """Translate the retired LeRobot stream name to the raw MCAP topic."""

    value = topic.strip()
    if value == _LEGACY_VIDEO_STREAM:
        return DEFAULT_MCAP_CAMERA_TOPIC
    if value not in TOP_CAMERA_TOPICS:
        raise ValueError(
            f"MCAP main camera topic must be one of {TOP_CAMERA_TOPICS}: "
            f"{topic!r}"
        )
    return value


def resolve_episode_mcap(
    task: str,
    episode_id: str,
    root: str | Path = DEFAULT_MCAP_ROOT,
) -> Path:
    """Resolve one episode without recursively scanning the object-store tree."""

    data_root = normalize_mcap_root(root)
    episode_names = [episode_id]
    if not episode_id.startswith("episode_"):
        episode_names.append(f"episode_{episode_id}")
    roots = (
        (data_root,)
        if data_root.name in {"train", "val"}
        else (data_root / "train", data_root / "val")
    )
    matches: list[Path] = []
    for split_root in roots:
        for episode_name in episode_names:
            candidate = split_root / task / episode_name / "episode.mcap"
            try:
                if candidate.is_file() and candidate.stat().st_size > 0:
                    matches.append(candidate)
            except OSError:
                continue
    unique = list(dict.fromkeys(matches))
    if not unique:
        tried = ", ".join(
            str(split_root / task / episode_name / "episode.mcap")
            for split_root in roots
            for episode_name in episode_names
        )
        raise FileNotFoundError(
            f"Raw episode MCAP does not exist for {task}/{episode_id}; "
            f"tried: {tried}"
        )
    if len(unique) != 1:
        raise RuntimeError(
            f"Ambiguous raw episode MCAP for {task}/{episode_id}: "
            + ", ".join(str(path) for path in unique)
        )
    return unique[0]


def list_camera_topics(source: str | Path) -> tuple[str, ...]:
    """List image-like camera topics from the MCAP summary."""

    source_path = Path(source)
    stat = source_path.stat()
    return _list_camera_topics_cached(
        str(source_path.resolve()), stat.st_size, stat.st_mtime_ns
    )


@lru_cache(maxsize=2048)
def _list_camera_topics_cached(
    source: str, _source_size: int, _source_mtime_ns: int
) -> tuple[str, ...]:
    make_reader, _decoder_factory = _mcap_imports()
    source_path = Path(source)
    with source_path.open("rb") as file:
        summary = make_reader(file).get_summary()
    if summary is None:
        raise RuntimeError(f"MCAP has no readable summary: {source_path}")
    topics = {
        channel.topic
        for channel in summary.channels.values()
        if "camera" in channel.topic and "info" not in channel.topic
    }
    return tuple(sorted(topics))


def select_camera_topic(
    source: str | Path,
    preferred: str = DEFAULT_MCAP_CAMERA_TOPIC,
) -> str:
    """Select only a top-mounted camera, never a wrist camera."""

    normalized = normalize_camera_topic(preferred)
    available = set(list_camera_topics(source))
    order = [normalized]
    order.extend(topic for topic in TOP_CAMERA_TOPICS if topic not in order)
    for topic in order:
        if topic in available:
            return topic
    raise RuntimeError(
        f"No usable top camera topic in {source}; tried {order}, "
        f"available cameras={sorted(available)}"
    )


def mcap_source_signature(
    source: str | Path,
    *,
    target_fps: float | int | str | Fraction = DEFAULT_MCAP_TARGET_FPS,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
    crf: int = 23,
    preset: str = "veryfast",
) -> dict[str, Any]:
    """Build a cheap cache/checkpoint signature without decoding the episode."""

    source_path = Path(source)
    stat = source_path.stat()
    fps = normalize_target_fps(target_fps)
    topic = select_camera_topic(source_path, preferred_topic)
    return {
        "source": str(source_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "topic": topic,
        "target_fps": _fraction_text(fps),
        "crf": crf,
        "preset": preset,
        "timestamp_policy": MCAP_TIMESTAMP_POLICY,
        "sampling_version": MCAP_SAMPLING_VERSION,
        "mcap_video_render_version": MCAP_VIDEO_RENDER_VERSION,
    }


def normalize_target_fps(
    value: float | int | str | Fraction,
) -> Fraction:
    try:
        fps = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid target FPS: {value!r}") from exc
    fps = fps.limit_denominator(1_000_000)
    if fps <= 0 or fps > 120:
        raise ValueError("Target FPS must be greater than 0 and no more than 120.")
    return fps


def resample_indices_by_timestamp(
    timestamps_ns: Sequence[int],
    target_fps: float | int | str | Fraction,
) -> list[int]:
    """Map a target CFR grid to nearest source timestamps.

    The target grid starts at the first camera message. Its frame count is
    ``floor((last-first) * fps) + 1`` so the encoded MP4 duration never ends
    before the final source timestamp. The final source frame is explicitly
    retained to avoid losing a tail frame at an FPS quantization boundary.
    """

    if not timestamps_ns:
        raise ValueError("At least one source timestamp is required.")
    normalized = [int(value) for value in timestamps_ns]
    if any(
        current < previous
        for previous, current in zip(normalized, normalized[1:])
    ):
        raise ValueError("Source camera timestamps must be monotonic.")
    fps = normalize_target_fps(target_fps)
    offsets = [value - normalized[0] for value in normalized]
    span = offsets[-1]
    output_count = (
        span * fps.numerator
    ) // (1_000_000_000 * fps.denominator) + 1
    picks: list[int] = []
    for index in range(output_count):
        target_offset = (
            index * 1_000_000_000 * fps.denominator
        ) // fps.numerator
        source_index = bisect.bisect_left(offsets, target_offset)
        if source_index >= len(offsets):
            source_index = len(offsets) - 1
        elif source_index > 0 and (
            target_offset - offsets[source_index - 1]
            <= offsets[source_index] - target_offset
        ):
            source_index -= 1
        picks.append(source_index)
    picks[-1] = len(offsets) - 1
    return picks


def inspect_mcap_video(
    source: str | Path,
    *,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
) -> McapVideoInfo:
    """Read the selected camera timestamps without creating an MP4."""

    source_path = _validate_source(source)
    topic = select_camera_topic(source_path, preferred_topic)
    info, _timestamps = _read_camera_messages(source_path, topic, None)
    return info


def render_mcap_video(
    source: str | Path,
    destination: str | Path,
    *,
    target_fps: float | int | str | Fraction = DEFAULT_MCAP_TARGET_FPS,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
    ass_path: str | Path | None = None,
    crf: int = 23,
    preset: str = "veryfast",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> McapRenderResult:
    """Decode the complete camera stream, timestamp-resample it, and encode MP4."""

    if not 0 <= crf <= 51:
        raise ValueError("CRF must be between 0 and 51.")
    if not preset.strip():
        raise ValueError("Encoder preset must be non-empty.")
    fps = normalize_target_fps(target_fps)
    source_path = _validate_source(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = Path(ass_path) if ass_path is not None else None
    if subtitle_path is not None and not subtitle_path.is_file():
        raise FileNotFoundError(f"ASS subtitle file does not exist: {subtitle_path}")
    topic = select_camera_topic(source_path, preferred_topic)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    temporary = destination_path.with_name(
        f".{destination_path.stem}.{token}.tmp.mp4"
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix=".mcap-video-", dir=destination_path.parent
        ) as temp_dir:
            elementary_path = Path(temp_dir) / "camera.es"
            info, timestamps = _read_camera_messages(
                source_path, topic, elementary_path
            )
            width, height = _probe_elementary_video(
                elementary_path, info.codec, ffprobe
            )
            info = McapVideoInfo(
                source=info.source,
                topic=info.topic,
                codec=info.codec,
                timestamp_source=info.timestamp_source,
                source_frame_count=info.source_frame_count,
                first_timestamp_ns=info.first_timestamp_ns,
                last_timestamp_ns=info.last_timestamp_ns,
                width=width,
                height=height,
            )
            picks = resample_indices_by_timestamp(timestamps, fps)
            decoded_count = _encode_resampled_video(
                elementary_path,
                info.codec,
                temporary,
                picks,
                fps,
                width,
                height,
                subtitle_path,
                crf,
                preset,
                ffmpeg,
                Path(temp_dir),
            )
            if decoded_count != info.source_frame_count:
                raise RuntimeError(
                    "Decoded MCAP camera frame count changed from "
                    f"{info.source_frame_count} messages to "
                    f"{decoded_count} decoded frames."
                )
            result = McapRenderResult(
                path=temporary,
                info=info,
                fps_numerator=fps.numerator,
                fps_denominator=fps.denominator,
                output_frame_count=len(picks),
                output_duration_seconds=len(picks) / float(fps),
            )
            verify_render_result(result, ffprobe=ffprobe)
        os.replace(temporary, destination_path)
        return McapRenderResult(
            path=destination_path,
            info=result.info,
            fps_numerator=result.fps_numerator,
            fps_denominator=result.fps_denominator,
            output_frame_count=result.output_frame_count,
            output_duration_seconds=result.output_duration_seconds,
        )
    finally:
        temporary.unlink(missing_ok=True)


def verify_mcap_video(
    source: str | Path,
    output: str | Path,
    *,
    target_fps: float | int | str | Fraction = DEFAULT_MCAP_TARGET_FPS,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
    ffprobe: str = "ffprobe",
) -> McapRenderResult:
    """Strictly verify an existing MP4 against its MCAP camera timeline."""

    fps = normalize_target_fps(target_fps)
    source_path = _validate_source(source)
    stat = source_path.stat()
    info = _inspect_mcap_video_with_dimensions_cached(
        str(source_path.resolve()),
        stat.st_size,
        stat.st_mtime_ns,
        normalize_camera_topic(preferred_topic),
        ffprobe,
    )
    output_count = len(
        resample_indices_by_timestamp(
            [info.first_timestamp_ns, info.last_timestamp_ns],
            fps,
        )
    )
    result = McapRenderResult(
        path=Path(output),
        info=info,
        fps_numerator=fps.numerator,
        fps_denominator=fps.denominator,
        output_frame_count=output_count,
        output_duration_seconds=output_count / float(fps),
    )
    verify_render_result(result, ffprobe=ffprobe)
    return result


def verify_render_result(
    result: McapRenderResult,
    *,
    ffprobe: str = "ffprobe",
) -> None:
    """Require exact target FPS/frame count and non-shortened MCAP duration."""

    probe = _probe_output(result.path, ffprobe)
    if result.info.width > 0 and result.info.height > 0 and (
        probe.width,
        probe.height,
    ) != (result.info.width, result.info.height):
        raise RuntimeError(
            "MCAP-derived video dimensions changed from "
            f"{result.info.width}x{result.info.height} to "
            f"{probe.width}x{probe.height}."
        )
    if probe.codec != "h264":
        raise RuntimeError(
            f"MCAP-derived video codec is {probe.codec!r}, expected 'h264'."
        )
    if probe.pixel_format != "yuv420p":
        raise RuntimeError(
            "MCAP-derived video pixel format is "
            f"{probe.pixel_format!r}, expected 'yuv420p'."
        )
    expected_rate = Fraction(
        result.fps_numerator, result.fps_denominator
    )
    if probe.frame_rate != expected_rate:
        raise RuntimeError(
            "MCAP-derived video FPS changed from target "
            f"{_fraction_text(expected_rate)} to "
            f"{_fraction_text(probe.frame_rate)}."
        )
    if probe.packet_count != result.output_frame_count:
        raise RuntimeError(
            "MCAP-derived video frame count changed from expected "
            f"{result.output_frame_count} to {probe.packet_count}."
        )
    tolerance = 0.005
    for label, duration in (
        ("video stream", probe.stream_duration),
        ("container", probe.container_duration),
    ):
        if abs(duration - result.output_duration_seconds) > tolerance:
            raise RuntimeError(
                f"MCAP-derived video {label} duration is {duration:.6f}s; "
                f"expected {result.output_duration_seconds:.6f}s."
            )
        if duration + tolerance < result.info.timestamp_span_seconds:
            raise RuntimeError(
                f"MCAP-derived video {label} ends at {duration:.6f}s, "
                "before the final source camera timestamp at "
                f"{result.info.timestamp_span_seconds:.6f}s."
            )


def cached_mcap_video(
    source: str | Path,
    cache_dir: str | Path,
    *,
    target_fps: float | int | str | Fraction = DEFAULT_MCAP_TARGET_FPS,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
    crf: int = 23,
    preset: str = "veryfast",
    cache_label: str = "episode",
) -> McapRenderResult | None:
    """Return a matching verified-on-write cache record, without rendering."""

    signature = mcap_source_signature(
        source,
        target_fps=target_fps,
        preferred_topic=preferred_topic,
        crf=crf,
        preset=preset,
    )
    output, metadata, _lock = _cache_paths(cache_dir, cache_label, signature)
    return _read_cached_result(output, metadata, signature)


def ensure_mcap_video(
    source: str | Path,
    cache_dir: str | Path,
    *,
    target_fps: float | int | str | Fraction = DEFAULT_MCAP_TARGET_FPS,
    preferred_topic: str = DEFAULT_MCAP_CAMERA_TOPIC,
    crf: int = 23,
    preset: str = "veryfast",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    cache_label: str = "episode",
) -> McapRenderResult:
    """Render a clean MCAP-derived MP4 once, with a cross-process cache lock."""

    signature = mcap_source_signature(
        source,
        target_fps=target_fps,
        preferred_topic=preferred_topic,
        crf=crf,
        preset=preset,
    )
    output, metadata, lock_path = _cache_paths(
        cache_dir, cache_label, signature
    )
    cached = _read_cached_result(output, metadata, signature)
    if cached is not None:
        return cached
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = _read_cached_result(output, metadata, signature)
        if cached is not None:
            return cached
        result = render_mcap_video(
            source,
            output,
            target_fps=target_fps,
            preferred_topic=preferred_topic,
            crf=crf,
            preset=preset,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        payload = {
            "signature": signature,
            "output_size": output.stat().st_size,
            "result": result.as_dict(),
        }
        _atomic_write_json(metadata, payload)
        return result


def _mcap_imports():
    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as exc:
        raise RuntimeError(
            "MCAP video rendering requires the 'mcap' and "
            "'mcap-protobuf-support' packages."
        ) from exc
    return make_reader, DecoderFactory


def _validate_source(source: str | Path) -> Path:
    source_path = Path(source)
    try:
        usable = source_path.is_file() and source_path.stat().st_size > 0
    except OSError:
        usable = False
    if not usable:
        raise FileNotFoundError(f"Raw episode MCAP does not exist: {source_path}")
    return source_path


def _read_camera_messages(
    source: Path,
    topic: str,
    elementary_path: Path | None,
    elementary_message_limit: int | None = None,
) -> tuple[McapVideoInfo, list[int]]:
    make_reader, decoder_factory = _mcap_imports()
    timestamps: list[int] = []
    detected_codec: str | None = None
    timestamp_kinds: set[str] = set()
    output: BinaryIO | None = None
    try:
        if elementary_path is not None:
            output = elementary_path.open("wb")
        with source.open("rb") as file:
            reader = make_reader(
                file, decoder_factories=[decoder_factory()]
            )
            for _schema, _channel, message, decoded in (
                reader.iter_decoded_messages(topics=[topic])
            ):
                if decoded is None:
                    continue
                payload = getattr(decoded, "data", None)
                if payload is None:
                    continue
                if isinstance(payload, memoryview):
                    payload = payload.tobytes()
                if not isinstance(payload, (bytes, bytearray)):
                    raise RuntimeError(
                        f"Camera message on {topic} has non-bytes data."
                    )
                timestamp_ns, timestamp_kind = _message_timestamp_ns(
                    decoded, int(message.log_time)
                )
                if timestamps and timestamp_ns < timestamps[-1]:
                    raise RuntimeError(
                        f"Camera timestamps go backwards on {topic}: "
                        f"{timestamp_ns} < {timestamps[-1]}."
                    )
                timestamps.append(timestamp_ns)
                timestamp_kinds.add(timestamp_kind)
                raw_format = getattr(decoded, "format", None)
                if raw_format:
                    codec = _normalize_codec(raw_format)
                    if detected_codec is None:
                        detected_codec = codec
                    elif codec != detected_codec:
                        raise RuntimeError(
                            f"Camera codec changes from {detected_codec} to "
                            f"{codec} on {topic}."
                        )
                if output is not None and (
                    elementary_message_limit is None
                    or len(timestamps) <= elementary_message_limit
                ):
                    output.write(payload)
    finally:
        if output is not None:
            output.close()
    if not timestamps:
        raise RuntimeError(f"No decodable camera frames found on {topic}: {source}")
    timestamp_source = (
        next(iter(timestamp_kinds))
        if len(timestamp_kinds) == 1
        else "protobuf_timestamp_and_log_time"
    )
    info = McapVideoInfo(
        source=source,
        topic=topic,
        codec=detected_codec or "h264",
        timestamp_source=timestamp_source,
        source_frame_count=len(timestamps),
        first_timestamp_ns=timestamps[0],
        last_timestamp_ns=timestamps[-1],
    )
    return info, timestamps


@lru_cache(maxsize=256)
def _inspect_mcap_video_with_dimensions_cached(
    source: str,
    _source_size: int,
    _source_mtime_ns: int,
    preferred_topic: str,
    ffprobe: str,
) -> McapVideoInfo:
    source_path = Path(source)
    topic = select_camera_topic(source_path, preferred_topic)
    with tempfile.TemporaryDirectory(prefix="mcap-video-probe-") as temp_dir:
        elementary_path = Path(temp_dir) / "camera-head.es"
        info, _timestamps = _read_camera_messages(
            source_path,
            topic,
            elementary_path,
            # SPS/PPS/VPS and an independently decodable frame are normally in
            # the first access unit. Keep a bounded prefix for unusual streams
            # while still scanning every message timestamp.
            elementary_message_limit=120,
        )
        width, height = _probe_elementary_video(
            elementary_path, info.codec, ffprobe
        )
    return McapVideoInfo(
        source=info.source,
        topic=info.topic,
        codec=info.codec,
        timestamp_source=info.timestamp_source,
        source_frame_count=info.source_frame_count,
        first_timestamp_ns=info.first_timestamp_ns,
        last_timestamp_ns=info.last_timestamp_ns,
        width=width,
        height=height,
    )


def _message_timestamp_ns(decoded: object, log_time_ns: int) -> tuple[int, str]:
    timestamp = getattr(decoded, "timestamp", None)
    try:
        seconds = int(getattr(timestamp, "seconds"))
        nanos = int(getattr(timestamp, "nanos"))
        value = seconds * 1_000_000_000 + nanos
    except (AttributeError, TypeError, ValueError):
        value = 0
    if value > 0:
        return value, "protobuf_timestamp"
    return log_time_ns, "message_log_time"


def _normalize_codec(value: object) -> str:
    text = str(value or "h264").strip().lower()
    if text in {"h265", "hevc", "h.265"}:
        return "hevc"
    if text in {"h264", "avc", "h.264"}:
        return "h264"
    raise RuntimeError(f"Unsupported MCAP camera codec: {value!r}")


def _probe_elementary_video(
    path: Path, codec: str, ffprobe: str
) -> tuple[int, int]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-f",
            codec,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ffprobe cannot decode MCAP camera stream: "
            f"{_last_error(completed.stderr)}"
        )
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot parse MCAP camera dimensions from ffprobe: {path}"
        ) from exc
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise RuntimeError(
            f"MCAP camera has invalid yuv420p dimensions: {width}x{height}."
        )
    return width, height


def _encode_resampled_video(
    elementary_path: Path,
    codec: str,
    destination: Path,
    picks: Sequence[int],
    fps: Fraction,
    width: int,
    height: int,
    subtitle_path: Path | None,
    crf: int,
    preset: str,
    ffmpeg: str,
    temp_dir: Path,
) -> int:
    decoder_error_path = temp_dir / "decoder.stderr"
    encoder_error_path = temp_dir / "encoder.stderr"
    decoder_command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        codec,
        "-i",
        str(elementary_path),
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "pipe:1",
    ]
    encoder_command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        _fraction_text(fps),
        "-i",
        "pipe:0",
    ]
    if subtitle_path is not None:
        encoder_command.extend(
            [
                "-vf",
                f"ass=filename={_escape_filter_path(subtitle_path)}",
            ]
        )
    encoder_command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    frame_size = width * height * 3 // 2
    decoded_count = 0
    written_count = 0
    pick_position = 0
    decoder: subprocess.Popen[bytes] | None = None
    encoder: subprocess.Popen[bytes] | None = None
    with decoder_error_path.open("wb") as decoder_error, (
        encoder_error_path.open("wb")
    ) as encoder_error:
        try:
            decoder = subprocess.Popen(
                decoder_command,
                stdout=subprocess.PIPE,
                stderr=decoder_error,
            )
            encoder = subprocess.Popen(
                encoder_command,
                stdin=subprocess.PIPE,
                stderr=encoder_error,
            )
            assert decoder.stdout is not None
            assert encoder.stdin is not None
            while True:
                frame = _read_exact(decoder.stdout, frame_size)
                if not frame:
                    break
                while (
                    pick_position < len(picks)
                    and picks[pick_position] == decoded_count
                ):
                    encoder.stdin.write(frame)
                    written_count += 1
                    pick_position += 1
                decoded_count += 1
            encoder.stdin.close()
            encoder.wait()
            decoder.stdout.close()
            decoder.wait()
        except BrokenPipeError:
            if encoder is not None and encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
            if encoder is not None:
                encoder.wait()
            if decoder is not None:
                decoder.terminate()
                decoder.wait()
        finally:
            for process in (decoder, encoder):
                if process is not None and process.poll() is None:
                    process.kill()
                    process.wait()
    decoder_error = decoder_error_path.read_text(
        encoding="utf-8", errors="replace"
    )
    encoder_error = encoder_error_path.read_text(
        encoding="utf-8", errors="replace"
    )
    if decoder is None or decoder.returncode:
        raise RuntimeError(
            "ffmpeg failed while decoding MCAP camera stream: "
            f"{_last_error(decoder_error)}"
        )
    if encoder is None or encoder.returncode:
        raise RuntimeError(
            "ffmpeg failed while encoding MCAP-derived video: "
            f"{_last_error(encoder_error)}"
        )
    if pick_position != len(picks) or written_count != len(picks):
        raise RuntimeError(
            "MCAP decoder ended before all timestamp-selected frames were "
            f"written ({written_count}/{len(picks)})."
        )
    return decoded_count


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if data and len(data) != size:
        raise RuntimeError(
            f"MCAP decoder emitted a truncated raw frame ({len(data)}/{size})."
        )
    return data


def _probe_output(path: Path, ffprobe: str) -> _OutputProbe:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            (
                "stream=width,height,codec_name,pix_fmt,avg_frame_rate,"
                "duration,nb_read_packets:format=duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise RuntimeError(
            f"ffprobe failed for {path}: {_last_error(completed.stderr)}"
        )
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
        container_duration = float(payload["format"]["duration"])
        raw_stream_duration = stream.get("duration")
        stream_duration = (
            float(raw_stream_duration)
            if raw_stream_duration not in (None, "N/A")
            else container_duration
        )
        probe = _OutputProbe(
            width=int(stream["width"]),
            height=int(stream["height"]),
            codec=str(stream["codec_name"]),
            pixel_format=str(stream["pix_fmt"]),
            frame_rate=Fraction(str(stream["avg_frame_rate"])),
            packet_count=int(stream["nb_read_packets"]),
            stream_duration=stream_duration,
            container_duration=container_duration,
        )
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse ffprobe result for {path}.") from exc
    if (
        probe.width <= 0
        or probe.height <= 0
        or probe.packet_count <= 0
        or not math.isfinite(probe.stream_duration)
        or not math.isfinite(probe.container_duration)
    ):
        raise RuntimeError(f"Invalid video metadata reported for {path}.")
    return probe


def _cache_paths(
    cache_dir: str | Path,
    cache_label: str,
    signature: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    directory = Path(cache_dir) / f"mcap-video-v{MCAP_VIDEO_RENDER_VERSION}"
    digest = hashlib.sha256(
        json.dumps(
            signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:20]
    safe_label = _SAFE_CACHE_NAME.sub("_", cache_label).strip("._") or "episode"
    fps_slug = str(signature["target_fps"]).replace("/", "-")
    stem = f"{safe_label}__{fps_slug}fps__{digest}"
    output = directory / f"{stem}.mp4"
    return output, directory / f"{stem}.json", directory / f"{stem}.lock"


def _read_cached_result(
    output: Path,
    metadata: Path,
    signature: Mapping[str, Any],
) -> McapRenderResult | None:
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, Mapping)
            or payload.get("signature") != signature
            or not output.is_file()
            or output.stat().st_size <= 0
            or output.stat().st_size != int(payload["output_size"])
            or not isinstance(payload.get("result"), Mapping)
        ):
            return None
        return McapRenderResult.from_dict(payload["result"], path=output)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _fraction_text(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _last_error(value: str) -> str:
    lines = value.strip().splitlines()
    return lines[-1] if lines else "unknown error"


__all__ = [
    "DEFAULT_MCAP_CAMERA_TOPIC",
    "DEFAULT_MCAP_ROOT",
    "DEFAULT_MCAP_TARGET_FPS",
    "MCAP_SAMPLING_VERSION",
    "MCAP_TIMESTAMP_POLICY",
    "MCAP_VIDEO_RENDER_VERSION",
    "McapRenderResult",
    "McapVideoInfo",
    "TOP_CAMERA_TOPICS",
    "cached_mcap_video",
    "ensure_mcap_video",
    "inspect_mcap_video",
    "list_camera_topics",
    "mcap_source_signature",
    "normalize_camera_topic",
    "normalize_mcap_root",
    "normalize_target_fps",
    "render_mcap_video",
    "resample_indices_by_timestamp",
    "resolve_episode_mcap",
    "select_camera_topic",
    "verify_mcap_video",
    "verify_render_result",
]
