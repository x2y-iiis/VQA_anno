"""Normalization and validation of LAS template responses."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping

from .models import PipelineDataError


def parse_objects_response(response: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = _find_result_payload(response, required_key="objects")
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise PipelineDataError("Step 1 result field 'objects' must be a list.")

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise PipelineDataError(f"objects[{index}] must be an object.")
        category = _required_alias(
            item, ("category", "object_category", "type"), f"objects[{index}].category"
        )
        if "instances" in item:
            instances = item["instances"]
            if not isinstance(instances, list):
                raise PipelineDataError(f"objects[{index}].instances must be a list.")
            for instance_index, instance in enumerate(instances):
                if not isinstance(instance, Mapping):
                    raise PipelineDataError(
                        f"objects[{index}].instances[{instance_index}] must be an object."
                    )
                normalized.append(
                    _normalize_object(
                        instance,
                        category,
                        f"objects[{index}].instances[{instance_index}]",
                    )
                )
        else:
            normalized.append(_normalize_object(item, category, f"objects[{index}]"))
    return normalized


def _normalize_object(
    item: Mapping[str, Any], category: str, label: str
) -> dict[str, str]:
    name = _required_alias(item, ("name", "object_name"), f"{label}.name")
    description = _required_alias(
        item,
        ("description", "appearance_description", "appearance"),
        f"{label}.description",
    )
    return {"category": category, "name": name, "description": description}


def parse_actions_response(response: Mapping[str, Any]) -> dict[str, Any]:
    payload = _find_result_payload(
        response, required_key="segments", additional_key="task_description"
    )
    task_description = _required_text(
        payload.get("task_description"), "task_description"
    )
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise PipelineDataError("Step 3 result field 'segments' must be a list.")

    normalized: list[dict[str, Any]] = []
    previous_start = -1.0
    for index, item in enumerate(segments):
        if not isinstance(item, Mapping):
            raise PipelineDataError(f"segments[{index}] must be an object.")
        start = _timestamp(item.get("start"), f"segments[{index}].start")
        end = _timestamp(item.get("end"), f"segments[{index}].end")
        if end < start:
            raise PipelineDataError(
                f"segments[{index}].end must be greater than or equal to start."
            )
        if start < previous_start:
            raise PipelineDataError("segments must be ordered by start time.")
        previous_start = start
        description = _required_text(
            item.get("description"), f"segments[{index}].description"
        )
        skill_value = item.get("skill")
        skill = (
            "None"
            if skill_value is None
            else _required_text(skill_value, f"segments[{index}].skill")
        )
        normalized.append(
            {
                "start": start,
                "end": end,
                "description": description,
                "skill": skill,
            }
        )
    return {"task_description": task_description, "segments": normalized}


def _find_result_payload(
    response: Mapping[str, Any],
    *,
    required_key: str,
    additional_key: str | None = None,
) -> Mapping[str, Any]:
    data = response.get("data")
    candidates: list[Any] = [data]
    if isinstance(data, Mapping):
        candidates.append(data.get("final_summary"))
    for candidate in candidates:
        parsed = _parse_jsonish(candidate)
        if isinstance(parsed, Mapping) and required_key in parsed:
            if additional_key is None or additional_key in parsed:
                return parsed
    keys = required_key if additional_key is None else f"{additional_key}/{required_key}"
    raise PipelineDataError(f"LAS response does not contain parseable {keys} fields.")


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, (Mapping, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    fence = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[index:])
                return parsed
            except json.JSONDecodeError:
                continue
    return None


def _required_alias(
    item: Mapping[str, Any], aliases: Iterable[str], label: str
) -> str:
    for key in aliases:
        if key in item:
            return _required_text(item[key], label)
    raise PipelineDataError(f"{label} is required.")


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineDataError(f"{label} must be a non-empty string.")
    return value.strip()


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PipelineDataError(f"{label} must be a non-negative number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineDataError(f"{label} must be a non-negative number.") from exc
    if result < 0:
        raise PipelineDataError(f"{label} must be a non-negative number.")
    if not math.isfinite(result):
        raise PipelineDataError(f"{label} must be a finite number.")
    return result
