"""LAS long-video Submit/Poll transport backed by a customer Ark key."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

from ark_file_transport import ArkFileReference
from request_parallel import atomic_json


OPERATOR_ID = "las_long_video_understand"
OPERATOR_VERSION = "v1"


def _tos_setting(name: str, default: str = "") -> str:
    """Read a TOS setting from the environment or its explicitly named file."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    default_files = {
        "TOS_ACCESS_KEY": "/mnt/doubao_las_annotation/.runtime/tos_access_key",
        "TOS_SECRET_KEY": "/mnt/doubao_las_annotation/.runtime/tos_secret_key",
        "TOS_SECURITY_TOKEN": "/mnt/doubao_las_annotation/.runtime/tos_security_token",
        "TOS_BUCKET": "/mnt/doubao_las_annotation/.runtime/tos_bucket",
        "TOS_SIGNING_ENDPOINT": "/mnt/doubao_las_annotation/.runtime/tos_signing_endpoint",
    }
    path = os.environ.get(f"{name}_FILE", default_files.get(name, "")).strip()
    if path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return default


def _tos_int_setting(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read and validate an integer TOS transport tuning setting."""
    raw = _tos_setting(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"invalid_{name.lower()}") from error
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name.lower()}_must_be_between_{minimum}_and_{maximum}")
    return value


def _tos_bool_setting(name: str, default: bool = False) -> bool:
    raw = _tos_setting(name, "1" if default else "0").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"invalid_{name.lower()}")


def _is_url_reference(value) -> bool:
    return callable(getattr(value, "resolve", None)) and hasattr(value, "sha256")


class _UploadedVideoReference(ArkFileReference):
    # Keep the established URL-media marker so the ECoT request validation
    # accepts this reference.  The actual transport is TOS Policy Presigned
    # HTTPS, as reported by the publisher's own upload log.
    transport_method = "cos-presigned"

    def __init__(self, publisher, key: str, digest: str, size: int, model: str):
        super().__init__(publisher, None, model, "video/mp4", digest, size)
        self.key = key

    def resolve(self) -> str:
        if self.closed:
            raise RuntimeError("las_operator_video_reference_closed")
        return self.publisher.resolve_url(self.key)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.publisher.delete_owned(self.key)


class LASVideoPublisher:
    """Upload operator input MP4s to private TOS objects and return ``tos://`` URIs."""

    def __init__(self, root: Path, workers: int = 64, check: Callable | None = None):
        del root
        try:
            import tos
        except ImportError as error:
            raise RuntimeError("tos_sdk_unavailable:install_python_package_tos") from error

        required = {
            "TOS_ACCESS_KEY": _tos_setting("TOS_ACCESS_KEY"),
            "TOS_SECRET_KEY": _tos_setting("TOS_SECRET_KEY"),
            "TOS_BUCKET": _tos_setting("TOS_BUCKET"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError("missing_tos_environment_variable:" + ",".join(missing))
        self.bucket = required["TOS_BUCKET"]
        self.prefix = _tos_setting(
            "TOS_PREFIX", "video-cleaning/vqa-las-customer-ark-inputs",
        ).strip("/")
        self.region = _tos_setting("TOS_REGION", "cn-beijing")
        self.endpoint = _tos_setting(
            "TOS_ENDPOINT", f"tos-{self.region}.volces.com",
        ).removeprefix("https://").removeprefix("http://").rstrip("/")
        # Upload and cleanup run from the annotation hosts, which may not have
        # a route to TOS's private endpoint.  URL signing is local, so use a
        # separate client to issue LAS-facing private-network URLs while the
        # data plane continues to use the reachable upload endpoint.
        self.signing_endpoint = _tos_setting(
            "TOS_SIGNING_ENDPOINT", self.endpoint,
        ).removeprefix("https://").removeprefix("http://").rstrip("/")
        self.url_mode = _tos_setting("TOS_LAS_VIDEO_URL_MODE", "policy-presigned")
        if self.url_mode not in {"policy-presigned", "native-uri"}:
            raise RuntimeError("invalid_tos_las_video_url_mode")
        try:
            self.url_expires_seconds = int(_tos_setting(
                "TOS_POLICY_URL_EXPIRES_SECONDS", str(7 * 24 * 3600),
            ))
        except ValueError as error:
            raise RuntimeError("invalid_tos_policy_url_expiry") from error
        if not 3600 <= self.url_expires_seconds <= 7 * 24 * 3600:
            raise RuntimeError("tos_policy_url_expiry_must_be_between_3600_and_604800")
        # A stalled public PUT used to occupy one scarce upload slot for up to
        # four 120-second SDK attempts.  Keep conservative defaults, while
        # allowing production canaries to fail a bad connection quickly and
        # retry the record through a fresh request path.
        self.max_retry_count = _tos_int_setting("TOS_MAX_RETRY_COUNT", 3, 0, 10)
        self.connection_timeout_seconds = _tos_int_setting(
            "TOS_CONNECTION_TIMEOUT_SECONDS", 20, 1, 300,
        )
        self.socket_timeout_seconds = _tos_int_setting(
            "TOS_SOCKET_TIMEOUT_SECONDS", 120, 1, 600,
        )
        self.upload_relay_url = _tos_setting("TOS_UPLOAD_RELAY_URL").rstrip("/")
        self.upload_relay_token = _tos_setting("TOS_UPLOAD_RELAY_TOKEN")
        relay_token_file = _tos_setting("TOS_UPLOAD_RELAY_TOKEN_FILE")
        if relay_token_file and not self.upload_relay_token:
            token_path = Path(relay_token_file)
            if not token_path.is_file():
                raise RuntimeError("tos_upload_relay_token_file_missing")
            self.upload_relay_token = token_path.read_text(encoding="utf-8").strip()
        if self.upload_relay_url and not self.upload_relay_token:
            raise RuntimeError("tos_upload_relay_token_missing")
        if self.upload_relay_url:
            parsed_relay = urllib.parse.urlsplit(self.upload_relay_url)
            if parsed_relay.scheme != "http" or parsed_relay.hostname not in {"127.0.0.1", "localhost"}:
                raise RuntimeError("tos_upload_relay_must_use_loopback_http")
        # The annotation hosts can upload to COS at line rate, while large
        # public TOS PUTs stall or time out.  TOS Fetch pulls a short-lived,
        # private COS URL inside the storage services, so only the small Fetch
        # control request traverses the annotation host's TOS connection.
        self.upload_via_cos_fetch = _tos_bool_setting("TOS_UPLOAD_VIA_COS_FETCH")
        self.cos_fetch_client = None
        self.cos_fetch_bucket = ""
        self.cos_fetch_prefix = ""
        if self.upload_via_cos_fetch:
            from cos_ecot_images import BUCKET, make_client
            self.cos_fetch_client = make_client(max(1, workers))
            self.cos_fetch_bucket = BUCKET
            self.cos_fetch_prefix = _tos_setting(
                "TOS_COS_FETCH_PREFIX",
                "video-cleaning/vqa-subtask-grd-sta-cpa-las-inputs/tos-fetch-staging",
            ).strip("/")
            self.cos_fetch_url_expires_seconds = _tos_int_setting(
                "TOS_COS_FETCH_URL_EXPIRES_SECONDS", 3600, 300, 86400,
            )
            source_roots = _tos_setting(
                "TOS_COS_SOURCE_MOUNT_ROOTS",
                "/mnt/human_data/video_cleaning:/mnt/human_data/video-cleaning",
            )
            self.cos_source_mount_roots = tuple(
                Path(value).absolute() for value in source_roots.split(":") if value
            )
            self.cos_source_key_prefix = _tos_setting(
                "TOS_COS_SOURCE_KEY_PREFIX", "video-cleaning",
            ).strip("/")
        self.client = tos.TosClientV2(
            required["TOS_ACCESS_KEY"],
            required["TOS_SECRET_KEY"],
            self.endpoint,
            self.region,
            security_token=_tos_setting("TOS_SECURITY_TOKEN") or None,
            max_retry_count=self.max_retry_count,
            max_connections=max(1, workers),
            connection_time=self.connection_timeout_seconds,
            socket_timeout=self.socket_timeout_seconds,
        )
        self.signing_client = (
            self.client
            if self.signing_endpoint == self.endpoint
            else tos.TosClientV2(
                required["TOS_ACCESS_KEY"],
                required["TOS_SECRET_KEY"],
                self.signing_endpoint,
                self.region,
                security_token=_tos_setting("TOS_SECURITY_TOKEN") or None,
                max_retry_count=self.max_retry_count,
                max_connections=max(1, workers),
                connection_time=self.connection_timeout_seconds,
                socket_timeout=self.socket_timeout_seconds,
            )
        )
        self.slots = threading.BoundedSemaphore(max(1, workers))
        self.check = check or (lambda: None)
        print(
            f"las_operator_tos_transport upload_workers={max(1, workers)} "
            f"connection_timeout_seconds={self.connection_timeout_seconds} "
            f"socket_timeout_seconds={self.socket_timeout_seconds} "
            f"sdk_retries={self.max_retry_count} "
            "upload_route=" + (
                "cos-to-tos-fetch" if self.upload_via_cos_fetch
                else "ssh-relay" if self.upload_relay_url else "direct"
            ),
            flush=True,
        )

    def _relay_request(self, method: str, route: str, *, path: Path | None = None,
                       digest: str | None = None, key: str | None = None) -> dict:
        parsed = urllib.parse.urlsplit(self.upload_relay_url)
        connection = http.client.HTTPConnection(
            parsed.hostname, parsed.port or 80, timeout=max(60, self.socket_timeout_seconds),
        )
        target = (parsed.path.rstrip("/") + route) or route
        headers = {"Authorization": f"Bearer {self.upload_relay_token}"}
        body = None
        if path is not None:
            headers.update({
                "Content-Type": "video/mp4",
                "Content-Length": str(path.stat().st_size),
                "X-Content-SHA256": digest or "",
            })
            body = path.open("rb")
        elif key is not None:
            encoded = json.dumps({"key": key}).encode()
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(encoded))})
            body = encoded
        try:
            try:
                connection.request(method, target, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
            except (OSError, http.client.HTTPException) as error:
                raise RuntimeError(
                    f"tos_upload_relay_transport_error:{type(error).__name__}:{error}"
                ) from error
        finally:
            if hasattr(body, "close"):
                body.close()
            connection.close()
        if response.status != 200:
            raise RuntimeError(f"tos_upload_relay_http_error:{response.status}:{payload[-500:].decode(errors='replace')}")
        try:
            result = json.loads(payload)
        except (ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("tos_upload_relay_invalid_json") from error
        if not isinstance(result, dict):
            raise RuntimeError("tos_upload_relay_invalid_response")
        return result

    def _relay_upload(self, path: Path, digest: str) -> tuple[str, str]:
        result = self._relay_request("POST", "/v1/upload", path=path, digest=digest)
        key = result.get("key")
        request_id = result.get("request_id", "")
        if not isinstance(key, str) or not key.startswith(self.prefix.rstrip("/") + "/"):
            raise RuntimeError("tos_upload_relay_key_mismatch")
        return key, str(request_id)

    def _relay_delete(self, key: str) -> None:
        self._relay_request("POST", "/v1/delete", key=key)

    def _existing_cos_source_key(self, path: Path) -> str | None:
        if not self.upload_via_cos_fetch:
            return None
        absolute = path.absolute()
        for root in self.cos_source_mount_roots:
            try:
                relative = absolute.relative_to(root)
            except ValueError:
                continue
            if relative.parts and ".." not in relative.parts:
                return f"{self.cos_source_key_prefix}/{relative.as_posix()}"
        return None

    def _cos_to_tos_fetch(self, path: Path, digest: str,
                          source_cos_key: str | None = None,
                          source_range: tuple[int, int] | None = None,
                          expected_size: int | None = None) -> tuple[str, str, dict]:
        identity = uuid.uuid4().hex
        direct_existing = source_cos_key is not None and source_range is None
        cos_key = (
            source_cos_key if direct_existing
            else f"{self.cos_fetch_prefix}/{identity}/{digest}.mp4"
        )
        key = f"{self.prefix}/{identity}/{digest}.mp4"
        stage_started = time.monotonic()
        tos_created = False
        staged = not direct_existing
        multipart_upload_id = None
        range_copy_seconds = 0.0
        try:
            if source_range is not None:
                if source_cos_key is None:
                    raise RuntimeError("cos_range_copy_requires_source_object")
                offset, length = source_range
                created = self.cos_fetch_client.create_multipart_upload(
                    Bucket=self.cos_fetch_bucket, Key=cos_key,
                    ContentType="video/mp4",
                )
                multipart_upload_id = created["UploadId"]
                copied = self.cos_fetch_client.upload_part_copy(
                    Bucket=self.cos_fetch_bucket,
                    Key=cos_key,
                    PartNumber=1,
                    UploadId=multipart_upload_id,
                    CopySource={"Bucket": self.cos_fetch_bucket, "Key": source_cos_key},
                    CopySourceRange=f"bytes={offset}-{offset + length - 1}",
                )
                etag = copied["CopyPartResult"]["ETag"]
                self.cos_fetch_client.complete_multipart_upload(
                    Bucket=self.cos_fetch_bucket,
                    Key=cos_key,
                    UploadId=multipart_upload_id,
                    MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": etag}]},
                )
                multipart_upload_id = None
                range_copy_seconds = time.monotonic() - stage_started
            elif staged:
                with path.open("rb") as stream:
                    self.cos_fetch_client.put_object(
                        Bucket=self.cos_fetch_bucket,
                        Key=cos_key,
                        Body=stream,
                        ContentType="video/mp4",
                    )
            stage_seconds = time.monotonic() - stage_started
            source_url = self.cos_fetch_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.cos_fetch_bucket, "Key": cos_key},
                ExpiresIn=self.cos_fetch_url_expires_seconds,
            )
            fetch_started = time.monotonic()
            expected_size = path.stat().st_size if expected_size is None else expected_size
            fetch_error = None
            try:
                result = self.client.fetch_object(self.bucket, key, source_url)
                tos_created = True
            except Exception as error:
                # TOS Fetch is server-side and can finish successfully even
                # when its small JSON acknowledgement is truncated.  At high
                # concurrency the SDK then raises TosClientError("unable to do
                # serialization").  Reissuing the whole annotation record is
                # wasteful and, when a record uploads many clips, makes
                # eventual completion very unlikely.  Verify the destination
                # object below before treating this acknowledgement failure as
                # a failed upload.
                fetch_error = error
                result = None
            fetch_seconds = time.monotonic() - fetch_started
            verify_started = time.monotonic()
            head = None
            head_error = None
            for verify_attempt in range(1, 9):
                try:
                    head = self.client.head_object(self.bucket, key)
                    head_error = None
                    break
                except Exception as error:
                    head_error = error
                    if verify_attempt < 8:
                        time.sleep(min(2.0, 0.1 * (2 ** (verify_attempt - 1))))
            if head is None:
                if fetch_error is not None:
                    raise fetch_error
                raise head_error
            # A successful HEAD proves that Fetch created the object even if
            # parsing its acknowledgement failed.  Mark it as owned so normal
            # cleanup still applies.
            tos_created = True
            verify_seconds = time.monotonic() - verify_started
            if int(getattr(head, "content_length", -1)) != expected_size:
                raise RuntimeError("tos_fetch_content_length_mismatch")
            if fetch_error is not None:
                print(
                    "las_operator_fetch_ack_recovered_by_head "
                    f"error_type={type(fetch_error).__name__}",
                    flush=True,
                )
            return key, str(getattr(result, "request_id", "")) if result is not None else "", {
                "cos_stage_seconds": stage_seconds,
                "tos_fetch_seconds": fetch_seconds,
                "verify_seconds": verify_seconds,
                "cos_source_reused": float(source_cos_key is not None),
                "cos_range_copy_seconds": range_copy_seconds,
            }
        except BaseException:
            if multipart_upload_id is not None:
                try:
                    self.cos_fetch_client.abort_multipart_upload(
                        Bucket=self.cos_fetch_bucket,
                        Key=cos_key,
                        UploadId=multipart_upload_id,
                    )
                except Exception:
                    pass
            if tos_created:
                try:
                    self.client.delete_object(self.bucket, key)
                except Exception:
                    pass
            raise
        finally:
            if staged:
                try:
                    self.cos_fetch_client.delete_object(
                        Bucket=self.cos_fetch_bucket, Key=cos_key,
                    )
                except Exception as error:
                    print(
                        "las_operator_cos_fetch_staging_cleanup_failed "
                        f"error_type={type(error).__name__}",
                        flush=True,
                    )

    def start_cleanup_reaper(self) -> None:
        return

    def resolve_url(self, key: str) -> str:
        if self.url_mode == "native-uri":
            return f"tos://{self.bucket}/{key}"
        from tos.models2 import PolicySignatureCondition
        policy = self.signing_client.pre_signed_policy_url(
            bucket=self.bucket,
            conditions=[PolicySignatureCondition(key="key", value=key)],
            expires=self.url_expires_seconds,
        )
        return policy.get_signed_url_for_get_or_head(key)

    def delete_owned(self, key: str) -> None:
        prefix = self.prefix.rstrip("/") + "/"
        if not key.startswith(prefix) or ".." in key.split("/"):
            raise RuntimeError("las_operator_tos_cleanup_ownership_mismatch")
        try:
            if self.upload_relay_url:
                self._relay_delete(key)
            else:
                self.client.delete_object(self.bucket, key)
        except Exception as error:
            print(f"las_operator_media_cleanup_failed error_type={type(error).__name__}", flush=True)

    def publish(self, path: Path, model: str, mime_type: str = "video/mp4"):
        if mime_type != "video/mp4":
            raise ValueError("las_operator_media_requires_video_mp4")
        source_range = None
        if all(hasattr(path, name) for name in ("path", "offset", "length")):
            source_path = Path(path.path)
            source_range = (int(path.offset), int(path.length))
            if source_range[0] < 0 or source_range[1] < 1:
                raise ValueError("invalid_cos_source_file_slice")
            source_size = source_range[1]
        else:
            source_path = Path(path)
            source_size = source_path.stat().st_size
        source_cos_key = self._existing_cos_source_key(source_path)
        if source_range is not None and source_cos_key is None:
            raise RuntimeError("cos_source_file_slice_outside_owned_mount")
        if source_range is not None:
            # A compact local catalog deliberately contains no multi-GB shard
            # files.  FileSlice identifies immutable bytes already resident in
            # COS, so neither stat nor open the synthetic local shard path.
            # The object key and byte range form a stable media identity.
            digest = hashlib.sha256(
                f"cos-range-v1:{self.cos_fetch_bucket}:{source_cos_key}:"
                f"{source_range[0]}:{source_range[1]}".encode()
            ).hexdigest()
        elif source_cos_key is not None:
            # Dataset objects are immutable.  Avoid reading the whole source
            # through COSFS merely to hash bytes that already live in COS.
            source_stat = source_path.stat()
            digest = hashlib.sha256(
                f"cos-object-v1:{self.cos_fetch_bucket}:{source_cos_key}:"
                f"{source_range}:{source_size}:{source_stat.st_mtime_ns}".encode()
            ).hexdigest()
        else:
            hasher = hashlib.sha256()
            with source_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        self.check()
        wait_started = time.monotonic()
        with self.slots:
            slot_wait_seconds = time.monotonic() - wait_started
            upload_started = time.monotonic()
            stage_metrics = {}
            if self.upload_via_cos_fetch:
                key, request_id, stage_metrics = self._cos_to_tos_fetch(
                    source_path, digest, source_cos_key, source_range, source_size,
                )
            elif self.upload_relay_url:
                key, request_id = self._relay_upload(source_path, digest)
            else:
                key = f"{self.prefix}/{uuid.uuid4().hex}/{digest}.mp4"
                result = self.client.put_object_from_file(
                    self.bucket, key, str(path), content_type="video/mp4",
                )
                request_id = getattr(result, "request_id", "")
            upload_seconds = time.monotonic() - upload_started
        upload_route = (
            "cos-to-tos-fetch" if self.upload_via_cos_fetch
            else "ssh-relay" if self.upload_relay_url else "direct"
        )
        stage_fields = "".join(
            f" {name}={value:.3f}" for name, value in stage_metrics.items()
        )
        print(
            f"las_operator_video_uploaded transport=tos-{self.url_mode} bytes={source_size} "
            f"slot_wait_seconds={slot_wait_seconds:.3f} upload_seconds={upload_seconds:.3f} "
            f"upload_endpoint={self.endpoint} signing_endpoint={self.signing_endpoint} "
            f"upload_route={upload_route}{stage_fields} request_id={request_id}",
            flush=True,
        )
        return _UploadedVideoReference(self, key, digest, source_size, model)


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


class _LASOperatorStalePending(LASOperatorError):
    """A remotely confirmed pending task exceeded its one-time reuse age."""


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
    duration = frames / fps if np.isfinite(fps) and fps > 0 and frames > 0 else 0.0
    if duration <= 0:
        # OpenCV occasionally fails to reopen a freshly finalized MP4 while
        # hundreds of workers are creating clips concurrently. ffprobe uses the
        # same decoder family as LAS and is a more authoritative fallback than
        # treating that transient OpenCV metadata failure as a corrupt input.
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ], capture_output=True, text=True, timeout=60)
        try:
            duration = float(probe.stdout.strip()) if probe.returncode == 0 else 0.0
        except ValueError:
            duration = 0.0
        if not np.isfinite(duration) or duration <= 0:
            size = path.stat().st_size if path.is_file() else -1
            detail = (probe.stderr or "").strip().replace("\n", " ")[-500:]
            raise LASOperatorError(
                f"las_operator_video_decode_failed:size={size}:ffprobe={detail or 'no_duration'}"
            )
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


def _encode_single_image_video(
    image_path: Path,
    destination: Path,
    fps: float,
    duration_seconds: float = 1.2,
) -> None:
    """Use ffmpeg as a narrow fallback when OpenCV emits a header-only MP4."""
    destination.unlink(missing_ok=True)
    completed = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-loop", "1", "-i", str(image_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-r", f"{fps:g}", "-t", f"{duration_seconds:g}", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination),
    ], capture_output=True, text=True, timeout=600)
    if completed.returncode or not destination.is_file() or destination.stat().st_size <= 262:
        detail = (completed.stderr or "").strip().replace("\n", " ")[-1000:]
        raise LASOperatorError(
            "las_operator_single_image_fallback_failed:" + (detail or "empty_output")
        )


class LASCustomerArkOperator:
    """Durable adapter for LAS asynchronous video understanding."""

    def __init__(
        self,
        submit_endpoint: str,
        task_root: Path,
        publisher,
        *,
        fallback_task_root: Path | None = None,
        fallback_read_concurrency: int = 64,
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
        self.fallback_task_root = (
            Path(fallback_task_root)
            if fallback_task_root is not None and Path(fallback_task_root) != self.task_root
            else None
        )
        if not 1 <= fallback_read_concurrency <= 1024:
            raise ValueError("las_operator_fallback_read_concurrency_out_of_range")
        self._fallback_read_slots = (
            threading.BoundedSemaphore(fallback_read_concurrency)
            if self.fallback_task_root is not None else None
        )
        self.publisher = publisher
        self.las_key_env = las_key_env
        self.ark_key_env = ark_key_env
        configured_poll_interval = os.environ.get("VQA_LAS_POLL_INTERVAL_SECONDS")
        self.poll_interval = (
            float(configured_poll_interval)
            if configured_poll_interval is not None
            else poll_interval
        )
        if not 0 <= self.poll_interval <= 3600:
            raise ValueError("las_operator_poll_interval_out_of_range")
        self.poll_initial_stagger = os.environ.get(
            "VQA_LAS_POLL_INITIAL_STAGGER", "1"
        ) == "1"
        first_poll_target = os.environ.get("VQA_LAS_FIRST_POLL_TARGET_SECONDS", "")
        self.first_poll_target = (
            float(first_poll_target) if first_poll_target else None
        )
        if self.first_poll_target is not None and not 0 <= self.first_poll_target <= 3600:
            raise ValueError("las_operator_first_poll_target_out_of_range")
        try:
            self.control_http_concurrency = int(os.environ.get(
                "VQA_LAS_CONTROL_HTTP_CONCURRENCY", "0",
            ))
        except ValueError as error:
            raise ValueError("las_operator_control_http_concurrency_invalid") from error
        if not 0 <= self.control_http_concurrency <= 8192:
            raise ValueError("las_operator_control_http_concurrency_out_of_range")
        try:
            self.pending_resubmit_seconds = float(os.environ.get(
                "VQA_LAS_PENDING_RESUBMIT_SECONDS", str(poll_timeout),
            ))
        except ValueError as error:
            raise ValueError("las_operator_pending_resubmit_seconds_invalid") from error
        if not 0 <= self.pending_resubmit_seconds <= 86400:
            raise ValueError("las_operator_pending_resubmit_seconds_out_of_range")
        # A LAS task occupies an annotation request slot for its whole
        # Submit/Poll lifetime, but only needs a control-plane connection for
        # the short Submit or Poll transaction.  Keeping these two limits
        # separate prevents thousands of live tasks from opening thousands of
        # simultaneous TLS connections to the LAS control plane.
        self._control_http_slots = (
            threading.BoundedSemaphore(self.control_http_concurrency)
            if self.control_http_concurrency else None
        )
        self.poll_timeout = poll_timeout
        self._thread_state = threading.local()
        print(
            "las_operator_poll_transport "
            f"interval_seconds={self.poll_interval:g} "
            f"initial_stagger={str(self.poll_initial_stagger).lower()} "
            f"first_poll_target_seconds={self.first_poll_target} "
            f"control_http_concurrency={self.control_http_concurrency or 'unbounded'}",
            flush=True,
        )
        print(
            "las_operator_pending_recovery "
            f"resubmit_seconds={self.pending_resubmit_seconds:g} "
            "maximum_resubmissions=1 remote_status_check=true",
            flush=True,
        )
        print(
            "las_operator_task_state "
            f"primary={self.task_root} "
            f"fallback={self.fallback_task_root or 'none'} "
            f"fallback_read_concurrency={fallback_read_concurrency}",
            flush=True,
        )

    def _load_state(self, state_path: Path, digest: str) -> dict | None:
        if state_path.is_file():
            return json.loads(state_path.read_text())
        # A completed LAS task may contain malformed model JSON.  Retrying the
        # HTTP call used to promote the same completed fallback state forever.
        # A local tombstone makes the next attempt submit a genuinely new task
        # while retaining the bad state for audit.
        if state_path.with_suffix('.invalid-marker').is_file():
            return None
        if self.fallback_task_root is None:
            return None
        fallback = self.fallback_task_root / digest[:2] / f"{digest}.json"
        with self._fallback_read_slots:
            if not fallback.is_file():
                return None
            state = json.loads(fallback.read_text())
        if isinstance(state, dict):
            # Promote once into the local state tier.  All subsequent polling,
            # terminal updates, locks and atomic renames stay off COS FUSE.
            atomic_json(state_path, state)
            print(f"las_operator_task_state_promoted request_hash={digest}", flush=True)
        return state

    def invalidate_last_completed(self) -> None:
        path = getattr(self._thread_state, "path", None)
        if path is not None and path.is_file():
            os.replace(path, path.with_suffix(f".invalid.{time.time_ns()}.json"))
        if path is not None:
            marker = path.with_suffix('.invalid-marker')
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f'{time.time_ns()}\n', encoding='utf-8')

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
        slots = self._control_http_slots
        if slots is not None:
            slots.acquire()
        try:
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
        finally:
            if slots is not None:
                slots.release()
        if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
            raise LASOperatorError("las_operator_response_metadata_missing")
        return result

    @contextlib.contextmanager
    def _video_url(self, media, model: str, composition_fps: float = 5.0):
        if len(media) == 1 and media[0][0].startswith("video/") and _is_url_reference(media[0][1]):
            yield media[0][1].resolve(), None, lambda: None
            return
        with tempfile.TemporaryDirectory(prefix="vqa-las-operator-") as directory:
            root = Path(directory)
            video = root / "input.mp4"
            if len(media) == 1 and media[0][0].startswith("video/") and not _is_url_reference(media[0][1]):
                video.write_bytes(media[0][1])
            elif len(media) == 1 and media[0][0].startswith("image/"):
                # OpenCV's mp4v writer intermittently produces a 262-byte
                # header-only file when thousands of single-frame STA/GRD
                # requests encode concurrently.  The ffmpeg path is bounded
                # to one codec thread and has been stable under that load.
                image = root / "media-000.jpg"
                image.write_bytes(media[0][1])
                _encode_single_image_video(
                    image, video, composition_fps,
                )
            else:
                _compose_video(media, video, root, composition_fps)
            try:
                video = _ensure_minimum_video_duration(video, root)
            except LASOperatorError as error:
                if not (
                    len(media) == 1
                    and media[0][0].startswith("image/")
                    and str(error).startswith("las_operator_video_decode_failed")
                ):
                    raise
                # Some unusual source dimensions make OpenCV's mp4v writer
                # report success while producing only a 262-byte MP4 header.
                # Repair that one image locally instead of retrying an API call
                # that was never submitted.
                _encode_single_image_video(
                    root / "media-000.jpg", video, composition_fps,
                )
                video = _ensure_minimum_video_duration(video, root)
                print(
                    "las_operator_single_image_video_fallback "
                    f"fps={composition_fps:g} bytes={video.stat().st_size}",
                    flush=True,
                )
            reference = self.publisher.publish(video, model)
            yield reference.resolve(), {
                "transport": "tos",
                "bucket": self.publisher.bucket,
                "key": reference.key,
            }, reference.close

    def _cleanup_state_media(self, state: dict) -> None:
        media = state.get("media")
        if not isinstance(media, dict) or media.get("transport") != "tos":
            return
        if media.get("bucket") != self.publisher.bucket or not isinstance(media.get("key"), str):
            raise LASOperatorError("las_operator_tos_cleanup_state_mismatch")
        self.publisher.delete_owned(media["key"])
        state.pop("media", None)

    @staticmethod
    def _identity(
        task: str,
        model: str,
        system: str,
        prompt: str,
        media,
        json_schema,
    ) -> tuple[dict, str]:
        identity = {
            "version": "las-customer-ark-operator/v1",
            "task": task,
            "model": model,
            "system_sha256": hashlib.sha256(system.encode()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(
                json.dumps(json_schema, sort_keys=True).encode()
            ).hexdigest(),
            "media_sha256": _media_digest(media),
        }
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return identity, digest

    def cached_call(
        self,
        task: str,
        model: str,
        system: str,
        prompt: str,
        media,
        json_schema=None,
    ) -> tuple[str, dict] | None:
        """Return a completed durable result without consuming API admission."""
        identity, digest = self._identity(
            task, model, system, prompt, media, json_schema,
        )
        directory = self.task_root / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / f"{digest}.json"
        self._thread_state.path = state_path
        lock_path = directory / f"{digest}.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load_state(state_path, digest)
            if not isinstance(state, dict) or state.get("status") != "COMPLETED":
                return None
            response = state.get("response")
            try:
                return response["data"]["final_summary"], response
            except (KeyError, TypeError) as error:
                raise LASOperatorError("las_operator_completed_state_invalid") from error

    def _poll(
        self,
        state: dict,
        state_path: Path,
        *,
        abandon_if_stale: bool = False,
    ) -> tuple[str, dict]:
        deadline = time.monotonic() + self.poll_timeout
        poll_payload = {
            "operator_id": OPERATOR_ID,
            "operator_version": OPERATOR_VERSION,
            "task_id": state["task_id"],
        }
        # Large fleets often submit thousands of tasks in a short burst.  If
        # every waiter polls immediately and then sleeps for the same fixed
        # period, the LAS control plane receives a synchronized poll storm.
        # Spread the first poll deterministically while preserving task state
        # and the configured long-lived poll deadline.
        if self.poll_initial_stagger and self.poll_interval > 0:
            digest = hashlib.sha256(str(state["task_id"]).encode()).digest()
            fraction = int.from_bytes(digest[:8], "big") / 2**64
            if self.first_poll_target is None:
                first_delay = fraction * self.poll_interval
            else:
                # Most GRD tasks are already terminal at the first 60-second
                # check.  A random 0..600-second first poll causes completed
                # remote tasks to occupy local request slots for minutes.
                # Keep a small deterministic spread to avoid a restart storm.
                age = max(0.0, time.time() - float(state.get("updated_at_unix") or 0))
                if age >= self.first_poll_target:
                    # On a restart, thousands of saved PENDING tasks may
                    # already be terminal. Drain them over two minutes rather
                    # than hammering the LAS account-level Poll RPM limit.
                    first_delay = 120.0 * fraction
                else:
                    first_delay = self.first_poll_target - age + 30.0 * fraction
            if first_delay > 0:
                time.sleep(first_delay)
        transient_errors = 0
        while True:
            try:
                response = self._post(self.poll_endpoint, poll_payload)
                transient_errors = 0
            except LASOperatorError as error:
                status = error.status or 0
                recoverable = status == 0 or status == 429 or status >= 500
                if not recoverable:
                    raise
                transient_errors += 1
                if time.monotonic() >= deadline:
                    raise LASOperatorError("las_operator_poll_timeout") from error
                # LAS account-level RPM limits also apply to Poll.  A fixed
                # retry interval keeps a large fleet above that limit forever.
                # Back off only the affected task; other requests keep running.
                delay = max(self.poll_interval, error.retry_after or 0)
                if status == 429:
                    delay = max(
                        delay,
                        min(600.0, self.poll_interval * (2 ** min(transient_errors, 3))),
                    )
                print(
                    "las_operator_poll_transient_retry "
                    f"status={status} consecutive_errors={transient_errors} "
                    f"delay_seconds={delay:g} error_type={type(error).__name__}",
                    flush=True,
                )
                if delay > 0:
                    time.sleep(delay)
                continue
            metadata = response["metadata"]
            status = str(metadata.get("task_status") or "").upper()
            code = str(metadata.get("business_code") or "")
            if status == "COMPLETED" and code in {"", "0"}:
                data = response.get("data")
                if not isinstance(data, dict) or not isinstance(data.get("final_summary"), str):
                    raise LASOperatorError("las_operator_final_summary_missing")
                state.update(status=status, response=response, updated_at_unix=time.time())
                self._cleanup_state_media(state)
                atomic_json(state_path, state)
                return data["final_summary"], response
            if status in {"FAILED", "TIMEOUT", "CANCELLED"} or (status == "COMPLETED" and code not in {"", "0"}):
                state.update(status="FAILED", response=response, updated_at_unix=time.time())
                self._cleanup_state_media(state)
                atomic_json(state_path, state)
                raise LASOperatorError(
                    f"las_operator_task_failed:{status}:{code}:{metadata.get('error_msg', '')}"
                )
            # A call-level timeout used to restart the same saved PENDING task
            # forever.  Before the one permitted resubmission, verify its
            # remote status once; only a still-nonterminal task is abandoned.
            # Keep the old journal and TOS object as audit evidence.  The new
            # deterministic publication reuses the same media object.
            if abandon_if_stale:
                raise _LASOperatorStalePending("las_operator_stale_pending_task")
            if time.monotonic() >= deadline:
                raise LASOperatorError("las_operator_poll_timeout")
            time.sleep(self.poll_interval)

    def call(self, task: str, model: str, system: str, prompt: str, media, json_schema=None) -> tuple[str, dict]:
        self.ensure_available()
        identity, digest = self._identity(
            task, model, system, prompt, media, json_schema,
        )
        directory = self.task_root / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        state_path = directory / f"{digest}.json"
        self._thread_state.path = state_path
        lock_path = directory / f"{digest}.lock"
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self._load_state(state_path, digest)
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
            # LAS task IDs are scoped to the credential that submitted them.
            # Fleet migrations used to promote a legacy task state and poll it
            # with a different valid LAS key; LAS correctly returned 401 and a
            # global fatal marker then made every supervisor restart fail.  A
            # credential mismatch on an existing task is recoverable once: keep
            # the old state as evidence and submit a new task with the active
            # credentials.  A 401 from that fresh Submit still propagates as a
            # genuine fatal credential error.
            if isinstance(state, dict) and state.get("task_id"):
                stale_age = time.time() - float(state.get("updated_at_unix") or 0)
                stale_archives = list(state_path.parent.glob(
                    f"{state_path.stem}.stale-pending.*.json"
                ))
                abandon_if_stale = bool(
                    self.pending_resubmit_seconds
                    and stale_age >= self.pending_resubmit_seconds
                    and not stale_archives
                )
                try:
                    return self._poll(
                        state, state_path, abandon_if_stale=abandon_if_stale,
                    )
                except _LASOperatorStalePending:
                    archived = state_path.with_suffix(
                        f".stale-pending.{time.time_ns()}.json"
                    )
                    os.replace(state_path, archived)
                    state = None
                    print(
                        "las_operator_stale_pending_resubmit "
                        f"task={task} model={model} request_hash={digest} "
                        f"pending_age_seconds={stale_age:.0f}",
                        flush=True,
                    )
                except LASOperatorError as error:
                    if error.status != 401 or "ApiKey.Invalid" not in str(error):
                        raise
                    if state_path.is_file():
                        os.replace(
                            state_path,
                            state_path.with_suffix(
                                f".credential-mismatch.{time.time_ns()}.json"
                            ),
                        )
                    state = None
                    print(
                        "las_operator_task_credential_mismatch_resubmit "
                        f"task={task} model={model} request_hash={digest}",
                        flush=True,
                    )
            if not isinstance(state, dict) or not state.get("task_id"):
                sampling_fps = 0.5 if task == "ecot" else 5.0 if task.startswith("cpa") else 2.0
                # LAS will sample the uploaded video at ``sampling_fps``. Do
                # not manufacture and upload duplicate frames above that rate;
                # retain at least 1 FPS so a target image still occupies its
                # original one-second segment.
                composition_fps = max(1.0, sampling_fps)
                with self._video_url(media, model, composition_fps) as media_lease:
                    video_url, publication, cleanup_unsubmitted = media_lease
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
                    try:
                        submitted = self._post(self.submit_endpoint, payload)
                    except BaseException:
                        cleanup_unsubmitted()
                        raise
                    metadata = submitted["metadata"]
                    task_id = metadata.get("task_id")
                    if not task_id:
                        cleanup_unsubmitted()
                        raise LASOperatorError("las_operator_submit_task_id_missing")
                    state = {
                        "identity": identity,
                        "task_id": task_id,
                        "status": str(metadata.get("task_status") or "PENDING").upper(),
                        "updated_at_unix": time.time(),
                    }
                    if publication is not None:
                        state["media"] = publication
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
