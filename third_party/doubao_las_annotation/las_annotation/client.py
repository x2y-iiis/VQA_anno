"""Reusable client for invoking LAS online operators."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import requests


DEFAULT_BASE_URL = "https://operator.las.cn-beijing.volces.com"
LONG_VIDEO_OPERATOR_ID = "las_long_video_understand"


class LASClientError(RuntimeError):
    """Raised when an LAS request cannot return a valid response."""


@dataclass(frozen=True)
class LASConfig:
    """Connection settings for LAS without credential data."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = 300.0
    max_retries: int = 2
    retry_backoff: float = 1.0
    poll_interval: float = 5.0
    poll_timeout: float = 7200.0

    @property
    def submit_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/submit"

    @property
    def poll_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/poll"


class LASClient:
    """A connection-reusing client for LAS online operator calls."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: LASConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("LAS_API_KEY")
        if not self._api_key:
            raise ValueError("LAS API Key is required; set LAS_API_KEY or pass api_key.")

        if config is None:
            config = LASConfig(base_url=os.getenv("LAS_BASE_URL", DEFAULT_BASE_URL))
        if config.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0.")
        if config.timeout <= 0:
            raise ValueError("timeout must be greater than 0.")
        if config.poll_interval < 0:
            raise ValueError("poll_interval must be greater than or equal to 0.")
        if config.poll_timeout <= 0:
            raise ValueError("poll_timeout must be greater than 0.")

        self.config = config
        self._session = session or requests.Session()

    def call_operator(
        self,
        request_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Submit an operator request and poll until it reaches a terminal state."""
        payload = dict(request_body)
        submit_result = self.submit_operator(payload)
        metadata = _metadata(submit_result, "submit")
        task_id = metadata.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise LASClientError("LAS submit response does not contain metadata.task_id.")

        poll_body = {
            "operator_id": payload.get("operator_id"),
            "operator_version": payload.get("operator_version"),
            "task_id": task_id,
        }
        deadline = time.monotonic() + self.config.poll_timeout
        while True:
            poll_result = self.poll_operator(poll_body)
            poll_metadata = _metadata(poll_result, "poll")
            status = str(poll_metadata.get("task_status", "")).upper()

            if status == "COMPLETED":
                business_code = str(poll_metadata.get("business_code", ""))
                if business_code not in {"", "0"}:
                    raise _task_error(poll_metadata)
                return poll_result
            if status in {"FAILED", "TIMEOUT"}:
                raise _task_error(poll_metadata)
            if time.monotonic() >= deadline:
                raise LASClientError(
                    f"LAS task {task_id} did not complete within "
                    f"{self.config.poll_timeout:g} seconds."
                )
            time.sleep(self.config.poll_interval)

    def submit_operator(
        self,
        request_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Submit an LAS operator request, injecting Ark credentials at send time."""
        payload = dict(request_body)
        if payload.get("operator_id") == LONG_VIDEO_OPERATOR_ID:
            data = payload.get("data")
            if not isinstance(data, Mapping):
                raise ValueError(
                    f"{LONG_VIDEO_OPERATOR_ID} requires data to be a JSON object."
                )
            ark_api_key = os.getenv("ARK_API_KEY")
            if not ark_api_key:
                raise ValueError(
                    f"{LONG_VIDEO_OPERATOR_ID} requires ARK_API_KEY in the environment."
                )
            wire_data = dict(data)
            wire_data["ark_api_key"] = ark_api_key
            payload["data"] = wire_data
        return self._post_json(self.config.submit_url, payload)

    def poll_operator(
        self,
        request_body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Poll an LAS task using a complete poll request body."""
        return self._post_json(self.config.poll_url, dict(request_body))

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        response: requests.Response | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= self.config.max_retries:
                    raise LASClientError(
                        f"LAS request failed after {attempt + 1} attempts: {exc}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue

            if response.status_code not in {429, 500, 502, 503, 504}:
                break
            if attempt >= self.config.max_retries:
                break
            self._sleep_before_retry(attempt)

        assert response is not None
        if not response.ok:
            body = response.text[:2000]
            raise LASClientError(f"LAS returned HTTP {response.status_code}: {body}")

        try:
            result = response.json()
        except ValueError as exc:
            raise LASClientError("LAS returned a non-JSON response.") from exc
        if not isinstance(result, dict):
            raise LASClientError(
                f"LAS returned JSON type {type(result).__name__}; expected object."
            )
        return result

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "LASClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.config.retry_backoff * (2**attempt))


def call_las_operator(
    request_body: Mapping[str, Any],
    *,
    api_key: str | None = None,
    config: LASConfig | None = None,
) -> dict[str, Any]:
    """One-shot convenience wrapper; prefer LASClient for repeated calls."""
    with LASClient(api_key=api_key, config=config) as client:
        return client.call_operator(request_body)


def _metadata(result: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    metadata = result.get("metadata")
    if not isinstance(metadata, Mapping):
        raise LASClientError(
            f"LAS {operation} response does not contain a metadata object."
        )
    return metadata


def _task_error(metadata: Mapping[str, Any]) -> LASClientError:
    task_id = metadata.get("task_id", "unknown")
    status = metadata.get("task_status", "unknown")
    code = metadata.get("business_code", "unknown")
    message = metadata.get("error_msg", "")
    return LASClientError(
        f"LAS task {task_id} ended with status {status}, "
        f"business_code {code}: {message}"
    )
