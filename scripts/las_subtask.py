"""Run the vendored LAS pipeline and adapt it to the unified subtask contract."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDORED_ROOT = PROJECT_ROOT / "third_party" / "doubao_las_annotation"
if str(VENDORED_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_ROOT))

from las_annotation.client import LASClient  # noqa: E402
from las_annotation.config import PipelineConfig  # noqa: E402
from las_annotation.models import AnnotationTask  # noqa: E402
from las_annotation.pipeline import process_sample  # noqa: E402


ADAPTER_VERSION = "unified-las-subtask-adapter/v2"
DEFAULT_LAS_MODEL = "doubao-seed-2-1-pro-260628"
DEFAULT_LAS_STEP1_MODEL = DEFAULT_LAS_MODEL
DEFAULT_LAS_POSTPROCESS_MODEL = "doubao-seed-2-0-lite-260428"
DEFAULT_LAS_COS_URI_PREFIX = os.environ.get("LAS_COS_URI_PREFIX")
DEFAULT_COSCLI = os.environ.get("LAS_COSCLI", "coscli")
DEFAULT_SIGNED_URL_SECONDS = 7 * 24 * 3600


@dataclass(frozen=True)
class LASSubtaskConfig:
    """Configuration for the vendored LAS backend and media publication."""

    output_root: Path
    cos_uri_prefix: str | None = DEFAULT_LAS_COS_URI_PREFIX
    video_url_template: str | None = None
    coscli: str = DEFAULT_COSCLI
    signed_url_seconds: int = DEFAULT_SIGNED_URL_SECONDS
    model: str = DEFAULT_LAS_MODEL
    step1_model: str = DEFAULT_LAS_STEP1_MODEL
    postprocess_model: str = DEFAULT_LAS_POSTPROCESS_MODEL
    use_main_as_wrist_for_robot: bool = False

    def __post_init__(self) -> None:
        if bool(self.cos_uri_prefix) == bool(self.video_url_template):
            raise ValueError(
                "exactly one of cos_uri_prefix or video_url_template is required"
            )
        if self.cos_uri_prefix and not self.cos_uri_prefix.startswith("cos://"):
            raise ValueError("las_cos_uri_prefix_must_start_with_cos")
        if self.video_url_template and "{" not in self.video_url_template:
            raise ValueError("las_video_url_template_requires_a_format_field")
        if self.signed_url_seconds < 600:
            raise ValueError("las_signed_url_seconds_must_be_at_least_600")

    @property
    def pipeline_root(self) -> Path:
        return self.output_root / "_state" / "las-pipeline"

    @property
    def publication_cache_root(self) -> Path:
        return self.output_root / "_state" / "las-video-urls"


class LASVideoPublisher:
    """Publish a local video and return a URL accepted by LAS."""

    def __init__(self, config: LASSubtaskConfig):
        self.config = config

    def resolve(self, video: Path, source_uid: str) -> tuple[str, dict[str, Any]]:
        digest = file_sha256(video)
        if self.config.video_url_template:
            url = self.config.video_url_template.format(
                filename=video.name,
                stem=video.stem,
                sha256=digest,
                source_uid=source_uid,
            )
            if not re.match(r"^(?:https?|tos)://", url):
                raise RuntimeError("las_video_url_template_rendered_invalid_url")
            return url, {
                "method": "url_template",
                "source_sha256": digest,
                "remote_object": None,
            }

        assert self.config.cos_uri_prefix is not None
        remote = f"{self.config.cos_uri_prefix.rstrip('/')}/{digest[:2]}/{digest}.mp4"
        cache = self.config.publication_cache_root / f"{digest}.json"
        cached = read_json(cache)
        now = time.time()
        if (
            isinstance(cached, Mapping)
            and cached.get("source_sha256") == digest
            and cached.get("remote_object") == remote
            and isinstance(cached.get("video_url"), str)
            and float(cached.get("expires_at", 0)) > now + 600
        ):
            return str(cached["video_url"]), {
                "method": "cos_upload_signed_url",
                "source_sha256": digest,
                "remote_object": remote,
            }

        upload = subprocess.run(
            [
                self.config.coscli, "cp", str(video), remote, "--disable-log",
                "--routines", "1", "--thread-num", "4",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if upload.returncode:
            raise RuntimeError(
                f"las_video_upload_failed:{upload.returncode}:"
                f"{upload.stderr.strip()[-1000:]}"
            )
        signed = subprocess.run(
            [
                self.config.coscli, "signurl", remote, "--simple-output",
                "--time", str(self.config.signed_url_seconds),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if signed.returncode:
            raise RuntimeError(
                f"las_video_sign_failed:{signed.returncode}:"
                f"{signed.stderr.strip()[-1000:]}"
            )
        url = signed.stdout.strip()
        if not url.startswith("https://"):
            raise RuntimeError("las_video_sign_returned_invalid_url")
        atomic_json(
            cache,
            {
                "source_sha256": digest,
                "remote_object": remote,
                "video_url": url,
                "created_at": now,
                "expires_at": now + self.config.signed_url_seconds,
            },
            mode=0o600,
        )
        return url, {
            "method": "cos_upload_signed_url",
            "source_sha256": digest,
            "remote_object": remote,
        }


class LASSubtaskAnnotator:
    """Invoke vendored LAS and return the existing unified subtask result."""

    def __init__(
        self,
        config: LASSubtaskConfig,
        publisher: LASVideoPublisher | None = None,
    ) -> None:
        self.config = config
        self.publisher = publisher or LASVideoPublisher(config)

    def annotate(
        self,
        video: Path,
        *,
        source_uid: str,
        task_instruction: str,
        embodiment: str,
        source_has_video: bool,
        source_media_count: int,
        prepared: tuple | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        if prepared is None:
            video_url, publication = self.publisher.resolve(video, source_uid)
            timing = video_timing(video)
        else:
            video_url, publication, timing = prepared
        digest = publication["source_sha256"]
        episode_id = f"sample_{hashlib.sha256(source_uid.encode()).hexdigest()[:24]}"
        wrist_url = (
            video_url
            if embodiment == "robot" and self.config.use_main_as_wrist_for_robot
            else None
        )
        task = AnnotationTask(
            task="subtask",
            episode_id=episode_id,
            episode_index=int(digest[:12], 16),
            wrist_video_url=wrist_url,
            main_video_url=video_url,
            embodiment=embodiment,
            task_instruction=task_instruction,
        )
        pipeline_config = PipelineConfig(
            model=self.config.model,
            step1_model=self.config.step1_model,
            postprocess=True,
            rewrite_descriptions=True,
            merge_none_segments=True,
            postprocess_model=self.config.postprocess_model,
            render_review_videos=False,
        )
        client_factory = getattr(self, 'client_factory', LASClient)
        with ExitStack() as resources, client_factory(api_key=os.environ.get("LAS_API_KEY")) as client:
            options = {}
            postprocess_factory = getattr(self, 'postprocess_client_factory', None)
            if postprocess_factory is not None:
                def managed_postprocess_client():
                    value = postprocess_factory()
                    resources.callback(value.close)
                    return value
                options['postprocess_client_factory'] = managed_postprocess_client
            outcome = process_sample(
                task, client, self.config.pipeline_root, config=pipeline_config, **options
            )
        item_dir = self.config.pipeline_root / task.task / task.episode_id
        if outcome.status == "failed":
            error = read_json(item_dir / "error.json") or {}
            on_failed = getattr(self, 'on_failed_sample', None)
            if on_failed is not None:
                on_failed(item_dir, error)
            raise RuntimeError(
                f"las_subtask_failed:{error.get('stage', 'unknown')}:"
                f"{error.get('error_message') or outcome.error or 'unknown error'}"
            )
        final = read_json(item_dir / "postprocessed_annotation.json")
        if not isinstance(final, Mapping):
            raise RuntimeError(f"las_subtask_final_missing:{item_dir}")
        video_fps, video_frames, video_duration = timing
        subtasks = normalize_segments(
            final.get("segments"),
            source_has_video=source_has_video,
            source_media_count=source_media_count,
            video_duration=video_duration,
        )
        task_summary = re.sub(r"\s+", " ", task_instruction).strip()[:80]
        result = {
            "task_summary": task_summary,
            "subtasks": subtasks,
            "provider_output": {
                "schema_version": "vendored-las-subtasks/v1",
                "adapter_version": ADAPTER_VERSION,
                "source_duration_seconds": (
                    video_duration if source_has_video else None
                ),
                "source_media_count": (
                    None if source_has_video else source_media_count
                ),
                "las_model": self.config.model,
                "las_step1_model": self.config.step1_model,
                "las_postprocess_model": self.config.postprocess_model,
                "embodiment": embodiment,
                "video_sha256": digest,
                "publication_method": publication["method"],
                "remote_object": publication["remote_object"],
                "las_checkpoint_dir": str(item_dir),
                "las_video_fps": video_fps,
                "las_video_frame_count": video_frames,
                "raw_segments": final.get("segments"),
            },
        }
        prompt = (
            "Vendored LAS embodied_action_captioning_v2 pipeline; "
            f"task instruction: {task_instruction}"
        )
        return result, json.dumps(final, ensure_ascii=False), prompt


def normalize_segments(
    value: Any,
    *,
    source_has_video: bool,
    source_media_count: int,
    video_duration: float,
) -> list[dict[str, Any]]:
    """Map LAS segments to ordered, contiguous unified subtask intervals."""
    if not isinstance(value, list) or not value:
        raise ValueError("las_subtask_result_requires_nonempty_segments")
    segments = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"las_subtask_segment_not_object:{index}")
        description = re.sub(r"\s+", " ", str(item.get("description") or "")).strip()
        if not description:
            raise ValueError(f"las_subtask_segment_description_empty:{index}")
        try:
            start = float(item["start"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"las_subtask_segment_start_invalid:{index}") from error
        if not math.isfinite(start):
            raise ValueError(f"las_subtask_segment_start_invalid:{index}")
        segments.append({
            "start": min(video_duration, max(0.0, start)),
            "description": description,
            "skill": str(item.get("skill") or "None").strip() or "None",
        })
    segments.sort(key=lambda item: item["start"])
    collapsed = []
    for item in segments:
        if collapsed and abs(item["start"] - collapsed[-1]["start"]) <= 1e-6:
            collapsed[-1] = item
        else:
            collapsed.append(item)

    steps = []
    previous: float | int = 0.0 if source_has_video else 0
    maximum: float | int = (
        video_duration if source_has_video else max(0, source_media_count - 1)
    )
    for index, item in enumerate(collapsed):
        last = index == len(collapsed) - 1
        if source_has_video:
            end: float | int = (
                maximum if last else max(float(previous), collapsed[index + 1]["start"])
            )
            if float(end) <= float(previous):
                continue
        else:
            ratio = 1.0 if last else collapsed[index + 1]["start"] / video_duration
            end = int(maximum) if last else max(int(previous), round(float(maximum) * ratio))
            if int(end) < int(previous):
                continue
        skill = item["skill"]
        step = {
            "id": len(steps) + 1,
            "subtask": item["description"].lower(),
            "action": skill if skill == "None" else skill.lower(),
            "object": "None",
            "source": "None",
            "target": "None",
            "media_start_index": None if source_has_video else int(previous),
            "media_end_index": None if source_has_video else int(end),
            "start_time_seconds": float(previous) if source_has_video else None,
            "end_time_seconds": float(end) if source_has_video else None,
        }
        steps.append(step)
        previous = end
    if not steps:
        raise ValueError("las_subtask_result_has_no_positive_intervals")
    if source_has_video:
        steps[0]["start_time_seconds"] = 0.0
        steps[-1]["end_time_seconds"] = video_duration
    else:
        steps[0]["media_start_index"] = 0
        steps[-1]["media_end_index"] = max(0, source_media_count - 1)
    return steps


def video_timing(path: Path) -> tuple[float, int, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"las_video_timing_unavailable:{path}")
    return fps, frames, frames / fps


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
