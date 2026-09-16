"""LAS long-video Submit/Poll transport backed by a customer Ark key."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Callable
import urllib.error
import urllib.request

import cv2
import numpy as np

from request_parallel import atomic_json


OPERATOR_ID = "las_long_video_understand"
OPERATOR_VERSION = "v1"


def _is_url_reference(value) -> bool:
    return callable(getattr(value, "resolve", None)) and hasattr(value, "sha256")


class _UploadedVideoReference:
    def __init__(self, publisher, key: str, digest: str, size: int):
        self.publisher = publisher
        self.key = key
        self.sha256 = digest
        self.size_bytes = size
        self.closed = False

    def resolve(self) -> str:
        if self.closed:
            raise RuntimeError("las_operator_video_reference_closed")
        return self.publisher.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.publisher.bucket, "Key": self.key},
            ExpiresIn=7200,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.publisher.client.delete_object(Bucket=self.publisher.bucket, Key=self.key)
        except Exception as error:
            print(f"las_operator_media_cleanup_failed error_type={type(error).__name__}", flush=True)


class LASVideoPublisher:
    """Upload operator input MP4s to private COS objects and issue short URLs."""

    def __init__(self, root: Path, workers: int = 64, check: Callable | None = None):
        del root
        import boto3
        from botocore.config import Config
        coscli = os.environ.get("LAS_COSCLI", "coscli")
        bucket = os.environ.get("VQA_COS_BUCKET", "").strip()
        if not bucket:
            raise RuntimeError("missing_required_configuration:VQA_COS_BUCKET")
        endpoint = os.environ.get(
            "VQA_COS_ENDPOINT", "https://cos.ap-shanghai.myqcloud.com"
        ).strip()
        region = os.environ.get("VQA_COS_REGION", "ap-shanghai").strip()
        prefix = os.environ.get(
            "VQA_LAS_OPERATOR_VIDEO_PREFIX", "vqa-annotation/las-operator-videos"
        ).strip("/")
        result = subprocess.run(
            [coscli, "config", "show", "--disable-log"],
            capture_output=True, text=True, timeout=30,
        )
        fields = dict(re.findall(
            r"^[ \t]*(Secret ID|Secret Key|Session Token):[ \t]*([^\r\n]*)",
            result.stdout, flags=re.MULTILINE,
        ))
        if result.returncode or not fields.get("Secret ID", "").startswith("AKID") or not fields.get("Secret Key"):
            raise RuntimeError("cos_existing_credentials_unavailable")
        self.bucket = bucket
        self.prefix = prefix + "/"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=fields["Secret ID"].strip(),
            aws_secret_access_key=fields["Secret Key"].strip(),
            aws_session_token=fields.get("Session Token", "").strip() or None,
            config=Config(
                signature_version="s3",
                s3={"addressing_style": "virtual"},
                max_pool_connections=max(1, workers),
                connect_timeout=20,
                read_timeout=120,
                retries={"max_attempts": 2, "mode": "standard"},
                proxies={},
            ),
        )
        self.slots = threading.BoundedSemaphore(max(1, workers))
        self.check = check or (lambda: None)

    def start_cleanup_reaper(self) -> None:
        return

    def publish(self, path: Path, model: str, mime_type: str = "video/mp4"):
        del model
        if mime_type != "video/mp4":
            raise ValueError("las_operator_media_requires_video_mp4")
        path = Path(path)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        key = f"{self.prefix}{uuid.uuid4().hex}/{digest}.mp4"
        self.check()
        with self.slots, path.open("rb") as stream:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=stream, ContentType="video/mp4",
            )
        print(f"las_operator_video_uploaded bytes={path.stat().st_size}", flush=True)
        return _UploadedVideoReference(self, key, digest, path.stat().st_size)


class SignedMediaPathPublisher:
    """Adapt an existing byte-oriented signed-media publisher to MP4 paths."""

    def __init__(self, publisher):
        self.publisher = publisher

    def start_cleanup_reaper(self) -> None:
        return

    def publish(self, path: Path, model: str, mime_type: str = "video/mp4"):
        return self.publisher.publish(Path(path).read_bytes(), model, mime_type)


class LASOperatorError(RuntimeError):
    """A structured LAS operator failure."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _media_digest(media) -> str:
    digest = hashlib.sha256()
    for mime_type, value in media:
        digest.update(mime_type.encode())
        if _is_url_reference(value):
            digest.update(str(value.sha256).encode())
        else:
            digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as stream:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _compose_video(media, destination: Path, work: Path, output_fps: float = 5.0) -> None:
    """Encode ordered image/video inputs as one operator-compatible MP4."""
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("las_operator_invalid_composition_fps")
    local = []
    for index, (mime_type, value) in enumerate(media):
        suffix = ".mp4" if mime_type.startswith("video/") else ".jpg"
        path = work / f"media-{index:03d}{suffix}"
        if _is_url_reference(value):
            _download(value.resolve(), path)
        else:
            path.write_bytes(value)
        local.append((mime_type, path))

    first = None
    for mime_type, path in local:
        if mime_type.startswith("image/"):
            first = cv2.imread(str(path), cv2.IMREAD_COLOR)
        else:
            capture = cv2.VideoCapture(str(path))
            ok, first = capture.read()
            capture.release()
            if not ok:
                first = None
        if first is not None:
            break
    if first is None:
        raise LASOperatorError("las_operator_media_decode_failed")
    height, width = first.shape[:2]
    width -= width % 2
    height -= height % 2
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height),
    )
    if not writer.isOpened():
        raise LASOperatorError("las_operator_video_writer_open_failed")
    frames_written = 0
    try:
        for mime_type, path in local:
            if mime_type.startswith("image/"):
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise LASOperatorError("las_operator_image_decode_failed")
                frame = _fit_frame(frame, width, height)
                # Preserve the existing one-second image segment without
                # upsampling the following low-FPS teacher video to 5 FPS.
                for _ in range(max(1, round(output_fps))):
                    writer.write(frame)
                    frames_written += 1
                continue
            capture = cv2.VideoCapture(str(path))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or output_fps)
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = output_fps
            source_index = 0
            next_output_time = 0.0
            wrote_video = False
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                source_time = source_index / source_fps
                source_index += 1
                while next_output_time <= source_time + 1e-9:
                    writer.write(_fit_frame(frame, width, height))
                    frames_written += 1
                    wrote_video = True
                    next_output_time += 1.0 / output_fps
            capture.release()
            if not wrote_video:
                raise LASOperatorError("las_operator_video_decode_failed")
    finally:
        writer.release()
    if frames_written == 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise LASOperatorError("las_operator_composed_video_empty")


def _ensure_minimum_video_duration(path: Path, work: Path, minimum_seconds: float = 1.2) -> Path:
    """Pad a locally materialized clip so LAS never receives a sub-second video."""
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if not np.isfinite(fps) or fps <= 0 or frames <= 0:
        raise LASOperatorError("las_operator_video_decode_failed")
    duration = frames / fps
    if duration >= minimum_seconds:
        return path
    padded = work / "input-padded.mp4"
    completed = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
        "-vf", f"tpad=stop_mode=clone:stop_duration={minimum_seconds - duration:.6f}",
        "-t", f"{minimum_seconds:.6f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(padded),
    ], capture_output=True, text=True, timeout=600)
    if completed.returncode or not padded.is_file() or padded.stat().st_size == 0:
        raise LASOperatorError(
            "las_operator_short_video_padding_failed:" + completed.stderr[-1000:]
        )
    return padded


class LASCustomerArkOperator:
    """Durable adapter for LAS asynchronous video understanding."""

    def __init__(
        self,
        submit_endpoint: str,
        task_root: Path,
        publisher,
        *,
        las_key_env: str = "LAS_API_KEY",
        ark_key_env: str = "ARK_API_KEY",
        poll_interval: float = 30.0,
        poll_timeout: float = 7200.0,
    ) -> None:
        if not submit_endpoint.rstrip("/").endswith("/api/v1/submit"):
            raise ValueError("las_operator_endpoint_must_end_with_api_v1_submit")
        self.submit_endpoint = submit_endpoint.rstrip("/")
        self.poll_endpoint = self.submit_endpoint.removesuffix("/submit") + "/poll"
        self.task_root = Path(task_root)
        self.publisher = publisher
        self.las_key_env = las_key_env
        self.ark_key_env = ark_key_env
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self._thread_state = threading.local()

    def invalidate_last_completed(self) -> None:
        path = getattr(self._thread_state, "path", None)
        if path is not None and path.is_file():
            os.replace(path, path.with_suffix(f".invalid.{time.time_ns()}.json"))

    def ensure_available(self) -> None:
        for name in (self.las_key_env, self.ark_key_env):
            if not os.environ.get(name, "").strip():
                raise LASOperatorError(f"missing_api_key_environment_variable:{name}")

    def _post(self, endpoint: str, payload: dict) -> dict:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {os.environ[self.las_key_env]}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            retry_after = error.headers.get("Retry-After")
            try:
                retry_after_value = float(retry_after) if retry_after is not None else None
            except ValueError:
                retry_after_value = None
            raise LASOperatorError(
                f"las_operator_http_error:{error.code}:{body[-2000:]}",
                status=error.code,
                retry_after=retry_after_value,
            ) from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LASOperatorError(f"las_operator_transport_error:{type(error).__name__}:{error}") from error
        if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
            raise LASOperatorError("las_operator_response_metadata_missing")
        return result

    @contextlib.contextmanager
    def _video_url(self, media, model: str, composition_fps: float = 5.0):
        if len(media) == 1 and media[0][0].startswith("video/") and _is_url_reference(media[0][1]):
            yield media[0][1].resolve()
            return
        with tempfile.TemporaryDirectory(prefix="vqa-las-operator-") as directory:
            root = Path(directory)
            video = root / "input.mp4"
            if len(media) == 1 and media[0][0].startswith("video/") and not _is_url_reference(media[0][1]):
                video.write_bytes(media[0][1])
            else:
                _compose_video(media, video, root, composition_fps)
            video = _ensure_minimum_video_duration(video, root)
            reference = self.publisher.publish(video, model)
            try:
                yield reference.resolve()
            finally:
                close = getattr(reference, "close", None)
                if close is not None:
                    close()

    def _poll(self, state: dict, state_path: Path) -> tuple[str, dict]:
        deadline = time.monotonic() + self.poll_timeout
        poll_payload = {
            "operator_id": OPERATOR_ID,
            "operator_version": OPERATOR_VERSION,
            "task_id": state["task_id"],
        }
        while True:
            response = self._post(self.poll_endpoint, poll_payload)
            metadata = response["metadata"]
            status = str(metadata.get("task_status") or "").upper()
            code = str(metadata.get("business_code") or "")
            if status == "COMPLETED" and code in {"", "0"}:
                data = response.get("data")
                if not isinstance(data, dict) or not isinstance(data.get("final_summary"), str):
                    raise LASOperatorError("las_operator_final_summary_missing")
                state.update(status=status, response=response, updated_at_unix=time.time())
                atomic_json(state_path, state)
                return data["final_summary"], response
            if status in {"FAILED", "TIMEOUT", "CANCELLED"} or (status == "COMPLETED" and code not in {"", "0"}):
                state.update(status="FAILED", response=response, updated_at_unix=time.time())
                atomic_json(state_path, state)
                raise LASOperatorError(
                    f"las_operator_task_failed:{status}:{code}:{metadata.get('error_msg', '')}"
                )
            if time.monotonic() >= deadline:
                raise LASOperatorError("las_operator_poll_timeout")
            time.sleep(self.poll_interval)

    def call(self, task: str, model: str, system: str, prompt: str, media, json_schema=None) -> tuple[str, dict]:
        self.ensure_available()
        identity = {
            "version": "las-customer-ark-operator/v1",
            "task": task,
            "model": model,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(json.dumps(json_schema, sort_keys=True).encode()).hexdigest(),
            "media_sha256": _media_digest(media),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        directory = self.task_root / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / f"{digest}.json"
        self._thread_state.path = state_path
        lock_path = directory / f"{digest}.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(state_path.read_text()) if state_path.is_file() else None
            if isinstance(state, dict) and state.get("status") == "COMPLETED":
                response = state.get("response")
                return response["data"]["final_summary"], response
            if isinstance(state, dict) and state.get("status") in {"FAILED", "TIMEOUT", "CANCELLED"}:
                os.replace(state_path, state_path.with_suffix(f".failed.{time.time_ns()}.json"))
                state = None
            query = system + "\n\n" + prompt
            if len(media) > 1:
                query += (
                    "\n\nThe LAS input video concatenates the supplied visual media items in their "
                    "original order. Treat each consecutive segment as the corresponding media item."
                )
            if not isinstance(state, dict) or not state.get("task_id"):
                sampling_fps = 0.5 if task == "ecot" else 5.0 if task.startswith("cpa") else 2.0
                # LAS will sample the uploaded video at ``sampling_fps``. Do
                # not manufacture and upload duplicate frames above that rate;
                # retain at least 1 FPS so a target image still occupies its
                # original one-second segment.
                composition_fps = max(1.0, sampling_fps)
                with self._video_url(media, model, composition_fps) as video_url:
                    payload = {
                        "operator_id": OPERATOR_ID,
                        "operator_version": OPERATOR_VERSION,
                        "data": {
                            "video_url": video_url,
                            "query": query,
                            "fps": sampling_fps,
                            "model_name": model,
                            "ark_api_key": os.environ[self.ark_key_env],
                        },
                    }
                    submitted = self._post(self.submit_endpoint, payload)
                    metadata = submitted["metadata"]
                    task_id = metadata.get("task_id")
                    if not task_id:
                        raise LASOperatorError("las_operator_submit_task_id_missing")
                    state = {
                        "identity": identity,
                        "task_id": task_id,
                        "status": str(metadata.get("task_status") or "PENDING").upper(),
                        "updated_at_unix": time.time(),
                    }
                    atomic_json(state_path, state)
                    print(f"las_operator_submitted task={task} model={model} request_hash={digest}", flush=True)
                    # Keep uploaded media alive until the asynchronous task is terminal.
                    return self._poll(state, state_path)
            return self._poll(state, state_path)


def token_usage(response: dict, model: str) -> dict:
    usages = response.get("data", {}).get("token_usages") or []
    for item in usages:
        if item.get("model_name") == model and isinstance(item.get("token_usage"), dict):
            return item["token_usage"]
    return {}
