"""Atomic stage storage and checkpoint helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .models import PipelineDataError
from .parsing import parse_objects_response


STAGE_FILES = (
    "step1_raw.json",
    "objects.json",
    "context.json",
    "step3_raw.json",
    "annotation.json",
    "step4_raw.json",
    "postprocessed_annotation.json",
    "step3_review.mp4",
    "step4_review.mp4",
    "review_videos.json",
    "error.json",
)


def stage_paths(item_dir: Path) -> dict[str, Path]:
    return {name: item_dir / name for name in STAGE_FILES}


def clear_stage_files(paths: Mapping[str, Path]) -> None:
    for path in paths.values():
        path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_name = file.name
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def read_json_or_none(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def is_valid_json(path: Path) -> bool:
    return read_json_or_none(path) is not None


def load_objects_checkpoint(path: Path) -> list[dict[str, str]] | None:
    value = read_json_or_none(path)
    if not isinstance(value, Mapping) or not isinstance(value.get("objects"), list):
        return None
    try:
        return parse_objects_response({"data": value})
    except PipelineDataError:
        return None


def task_id(response: Mapping[str, Any]) -> str | None:
    metadata = response.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("task_id"), str):
        return metadata["task_id"]
    return None


def checkpoint_task_id(record: Any, path: Path) -> str | None:
    if not isinstance(record, Mapping):
        record = read_json_or_none(path)
    if isinstance(record, Mapping) and isinstance(record.get("response"), Mapping):
        return task_id(record["response"])
    return None
