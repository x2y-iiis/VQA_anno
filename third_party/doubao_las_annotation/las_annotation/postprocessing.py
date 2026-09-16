"""Step 4 text rewriting and deterministic segment cleanup."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from typing import Any, Mapping, Sequence

from .models import PipelineDataError


DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_POSTPROCESS_MODEL = "doubao-seed-2-0-lite-260428"
NONE_SEGMENT_MERGE_VERSION = 1
ADJACENT_IDENTICAL_MERGE_VERSION = 1
REWRITE_PROMPT_RESOURCE = "postprocessing_prompt.txt"
HUMAN_REWRITE_PROMPT_RESOURCE = "postprocessing_prompt_human.txt"
_RESOURCE_PACKAGE = "las_annotation.resources"


@lru_cache(maxsize=2)
def load_rewrite_system_prompt(embodiment: str = "robot") -> str:
    """Load the Step 4 rewrite prompt from the package resources."""
    if embodiment not in {"robot", "human"}:
        raise ValueError("embodiment must be either 'robot' or 'human'.")
    resource_name = (
        HUMAN_REWRITE_PROMPT_RESOURCE
        if embodiment == "human"
        else REWRITE_PROMPT_RESOURCE
    )
    try:
        text = (
            files(_RESOURCE_PACKAGE)
            .joinpath(resource_name)
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError as exc:
        raise PipelineDataError(
            f"Cannot read resource {resource_name}: {exc}"
        ) from exc
    if not text:
        raise PipelineDataError(
            f"Resource {resource_name} cannot be empty."
        )
    return text


def rewrite_prompt_digest(embodiment: str = "robot") -> str:
    """Return a checkpoint fingerprint for the active Step 4 prompt."""
    return sha256(
        load_rewrite_system_prompt(embodiment).encode("utf-8")
    ).hexdigest()


class DoubaoTextClient:
    """Small synchronous wrapper around Ark's OpenAI-compatible Responses API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_ARK_BASE_URL,
        client: Any | None = None,
    ) -> None:
        key = api_key or os.getenv("ARK_API_KEY")
        if not key:
            raise ValueError(
                "Ark API Key is required for Step 4; set ARK_API_KEY or "
                "disable postprocessing with --no-postprocess."
            )
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base_url)
        self._client = client

    def call(self, request: Mapping[str, Any]) -> dict[str, Any]:
        response = self._client.responses.create(**dict(request))
        if hasattr(response, "model_dump"):
            value = response.model_dump(mode="json")
        elif isinstance(response, Mapping):
            value = dict(response)
        else:
            raise PipelineDataError(
                "Ark Responses API returned an unsupported response object."
            )
        if not isinstance(value, dict):
            raise PipelineDataError("Ark Responses API response must be an object.")
        return value

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def build_rewrite_request(
    descriptions: Sequence[str],
    model: str = DEFAULT_POSTPROCESS_MODEL,
    *,
    skills: Sequence[str | None] | None = None,
    embodiment: str = "robot",
) -> dict[str, Any]:
    if not model.strip():
        raise ValueError("postprocess model must be a non-empty string.")
    if not descriptions:
        raise PipelineDataError("Step 4 requires at least one segment description.")
    if skills is not None and len(skills) != len(descriptions):
        raise PipelineDataError(
            "Step 4 skills and descriptions must contain the same number of items."
        )

    numbered = []
    for index, description in enumerate(descriptions, start=1):
        if not isinstance(description, str) or not description.strip():
            raise PipelineDataError(
                f"Step 4 description {index - 1} must be a non-empty string."
            )
        normalized_description = " ".join(description.split())
        if skills is None:
            numbered.append(f"{index}. {normalized_description}")
            continue
        skill = skills[index - 1]
        if skill is None:
            normalized_skill = "None"
        elif isinstance(skill, str) and skill.strip():
            normalized_skill = skill.strip()
        else:
            raise PipelineDataError(
                f"Step 4 skill {index - 1} must be a non-empty string or None."
            )
        numbered.append(
            f"{index}. [Skill: {normalized_skill}] "
            f"[Description: {normalized_description}]"
        )
    return {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": load_rewrite_system_prompt(embodiment),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "\n".join(numbered)}
                ],
            },
        ],
        "temperature": 0.0,
    }


def parse_rewrite_response(
    response: Mapping[str, Any], expected_count: int
) -> list[str]:
    if expected_count <= 0:
        raise ValueError("expected_count must be greater than 0.")
    text = _extract_output_text(response)
    if not text:
        raise PipelineDataError("Ark response does not contain output_text.")
    stripped = text.strip()
    fence = re.fullmatch(r"```(?:text)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    lines = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^(?:[-*]\s+|\d+\s*[.)、:：-]\s*)", "", line).strip()
        if line:
            lines.append(line)
    if len(lines) != expected_count:
        raise PipelineDataError(
            "Ark rewrite returned "
            f"{len(lines)} non-empty lines; expected {expected_count}."
        )
    return lines


def merge_none_segments(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge every skill=None segment into an adjacent real action.

    Leading None segments have no previous action, so they are merged forward
    into the first real action. Every later None segment is merged backward into
    the most recent real action. If the annotation contains no real action, it
    is preserved unchanged.
    """
    copied = [deepcopy(dict(segment)) for segment in segments]
    if not copied:
        return copied
    first_action = 0
    while first_action < len(copied) and _is_none_skill(
        copied[first_action].get("skill")
    ):
        first_action += 1
    if first_action == len(copied):
        return copied

    action = copied[first_action]
    if first_action:
        action["start"] = copied[0]["start"]

    merged = [action]
    for segment in copied[first_action + 1 :]:
        if _is_none_skill(segment.get("skill")):
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(segment)
    return merged


def merge_adjacent_identical_segments(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Merge temporally contiguous segments with the same skill and description.

    LAS may split a sustained action into multiple segments. Step 4 can also
    rewrite slightly different source descriptions into the same concise English
    sentence. In both cases, retaining the artificial boundary creates duplicate
    adjacent subtasks. Only exact normalized matches are merged so distinct
    repeated actions and descriptions with meaningful differences stay separate.
    """
    copied = [deepcopy(dict(segment)) for segment in segments]
    merged: list[dict[str, Any]] = []
    for segment in copied:
        if merged and _segments_are_adjacent_and_identical(merged[-1], segment):
            merged[-1]["end"] = max(merged[-1]["end"], segment["end"])
        else:
            merged.append(segment)
    return merged


def annotation_digest(annotation: Mapping[str, Any]) -> str:
    payload = json.dumps(
        annotation,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _extract_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    output = response.get("output")
    if not isinstance(output, list):
        return ""
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, Mapping)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    return "\n".join(texts)


def _is_none_skill(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and value.strip().casefold() == "none"
    )


def _normalized_segment_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _segments_are_adjacent_and_identical(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    try:
        boundary_matches = abs(float(left["end"]) - float(right["start"])) <= 1e-6
    except (KeyError, TypeError, ValueError):
        return False
    return (
        boundary_matches
        and _normalized_segment_text(left.get("skill"))
        == _normalized_segment_text(right.get("skill"))
        and _normalized_segment_text(left.get("description"))
        == _normalized_segment_text(right.get("description"))
    )


__all__ = [
    "ADJACENT_IDENTICAL_MERGE_VERSION",
    "DEFAULT_ARK_BASE_URL",
    "DEFAULT_POSTPROCESS_MODEL",
    "HUMAN_REWRITE_PROMPT_RESOURCE",
    "NONE_SEGMENT_MERGE_VERSION",
    "DoubaoTextClient",
    "REWRITE_PROMPT_RESOURCE",
    "annotation_digest",
    "build_rewrite_request",
    "load_rewrite_system_prompt",
    "merge_none_segments",
    "merge_adjacent_identical_segments",
    "parse_rewrite_response",
    "rewrite_prompt_digest",
]
