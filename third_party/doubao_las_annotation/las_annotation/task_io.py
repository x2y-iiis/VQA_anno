"""ABC130K TXT/TSV ingestion and automatic TOS video URL resolution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .config import DEFAULT_MAPPING_PATH, VideoAssetConfig
from .models import AnnotationTask, PipelineDataError


def load_tasks(
    input_path: str | Path,
    mapping_path: str | Path = DEFAULT_MAPPING_PATH,
    video_config: VideoAssetConfig | None = None,
    *,
    offset: int = 0,
    limit: int | None = None,
    episodes_per_task: int | None = None,
    embodiment: str = "robot",
    include_wrist: bool = True,
) -> list[AnnotationTask]:
    """Load a task-list TXT or episode TSV and resolve all video URLs."""
    if offset < 0:
        raise PipelineDataError("offset must be non-negative.")
    if limit is not None and limit < 0:
        raise PipelineDataError("limit must be non-negative.")
    if episodes_per_task is not None and episodes_per_task < 0:
        raise PipelineDataError("episodes_per_task must be non-negative.")
    source = Path(input_path)
    mapping = load_episode_mapping(mapping_path)
    video_config = video_config or VideoAssetConfig()

    suffix = source.suffix.lower()
    if suffix == ".txt":
        identities = _load_task_list(source, mapping)
    elif suffix == ".tsv":
        identities = _load_episode_tsv(source)
    else:
        raise PipelineDataError(
            f"Unsupported input format {source.suffix!r}; expected .txt or .tsv."
        )

    if episodes_per_task is not None:
        task_counts: dict[str, int] = {}
        filtered_identities: list[tuple[str, str]] = []
        for identity in identities:
            task_name, _ = identity
            count = task_counts.get(task_name, 0)
            if count >= episodes_per_task:
                continue
            task_counts[task_name] = count + 1
            filtered_identities.append(identity)
        identities = filtered_identities

    reverse = _reverse_mapping(mapping)
    identities = identities[offset:]
    if limit is not None:
        identities = identities[:limit]

    tasks: list[AnnotationTask] = []
    seen: set[tuple[str, str]] = set()
    for task_name, episode_id in identities:
        identity = (task_name, episode_id)
        if identity in seen:
            raise PipelineDataError(
                f"Duplicate task/episode in {source}: {task_name}/{episode_id}."
            )
        seen.add(identity)
        try:
            episode_index = reverse[task_name][episode_id]
        except KeyError as exc:
            raise PipelineDataError(
                f"No UUID-index mapping for {task_name}/{episode_id}."
            ) from exc
        tasks.append(
            AnnotationTask(
                task=task_name,
                episode_id=episode_id,
                episode_index=episode_index,
                wrist_video_url=video_config.video_url(
                    task_name, episode_index, video_config.wrist_stream
                ) if include_wrist else None,
                main_video_url=video_config.video_url(
                    task_name, episode_index, video_config.main_stream
                ),
                embodiment=embodiment,
            )
        )
    if not tasks:
        raise PipelineDataError(f"{source}: no episodes found.")
    return tasks


def load_indexed_tos_tasks(
    input_path: str | Path,
    video_config: VideoAssetConfig | None = None,
    *,
    episodes_per_task: int,
    episode_start_index: int = 0,
    offset: int = 0,
    limit: int | None = None,
    embodiment: str = "robot",
    include_wrist: bool = True,
) -> list[AnnotationTask]:
    """Expand a task split directly into deterministic, index-based TOS URLs.

    This is intentionally separate from :func:`load_tasks`: the established
    loader keeps using the UUID-index mapping, while this loader is for TOS
    datasets whose episode objects are numbered consecutively from zero.
    """
    if offset < 0:
        raise PipelineDataError("offset must be non-negative.")
    if limit is not None and limit < 0:
        raise PipelineDataError("limit must be non-negative.")
    if episodes_per_task <= 0:
        raise PipelineDataError(
            "episodes_per_task must be greater than 0 for indexed TOS input."
        )
    if episode_start_index < 0:
        raise PipelineDataError("episode_start_index must be non-negative.")

    source = Path(input_path)
    task_names = _load_indexed_tos_task_names(source)
    indexed_episodes = [
        (task_name, episode_index)
        for task_name in task_names
        for episode_index in range(
            episode_start_index,
            episode_start_index + episodes_per_task,
        )
    ]
    indexed_episodes = indexed_episodes[offset:]
    if limit is not None:
        indexed_episodes = indexed_episodes[:limit]

    return _build_indexed_tos_tasks(
        source,
        indexed_episodes,
        video_config,
        embodiment=embodiment,
        include_wrist=include_wrist,
    )


def load_indexed_tos_manifest(
    input_path: str | Path,
    video_config: VideoAssetConfig | None = None,
    *,
    offset: int = 0,
    limit: int | None = None,
    embodiment: str = "robot",
    include_wrist: bool = True,
) -> list[AnnotationTask]:
    """Load exact task/index pairs without requiring a UUID mapping.

    The TSV must contain ``task`` and ``episode_index`` columns. This mode is
    useful for datasets whose tasks have different episode counts or gaps.
    """
    if offset < 0:
        raise PipelineDataError("offset must be non-negative.")
    if limit is not None and limit < 0:
        raise PipelineDataError("limit must be non-negative.")

    source = Path(input_path)
    if source.suffix.lower() != ".tsv":
        raise PipelineDataError(
            f"Indexed TOS manifest requires a .tsv file, got {source.suffix!r}."
        )

    indexed_episodes: list[tuple[str, int]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required_columns = {"task", "episode_index"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise PipelineDataError(
                f"{source}: indexed TOS manifest must contain 'task' and "
                "'episode_index' columns."
            )
        for line_number, row in enumerate(reader, start=2):
            task_name = (row.get("task") or "").strip()
            raw_index = (row.get("episode_index") or "").strip()
            if not task_name:
                raise PipelineDataError(
                    f"{source}:{line_number}: missing task name."
                )
            try:
                episode_index = int(raw_index)
            except ValueError as exc:
                raise PipelineDataError(
                    f"{source}:{line_number}: invalid episode_index {raw_index!r}."
                ) from exc
            if episode_index < 0:
                raise PipelineDataError(
                    f"{source}:{line_number}: episode_index must be non-negative."
                )
            indexed_episodes.append((task_name, episode_index))

    indexed_episodes = indexed_episodes[offset:]
    if limit is not None:
        indexed_episodes = indexed_episodes[:limit]
    return _build_indexed_tos_tasks(
        source,
        indexed_episodes,
        video_config,
        embodiment=embodiment,
        include_wrist=include_wrist,
    )


def load_url_manifest(
    input_path: str | Path,
    *,
    offset: int = 0,
    limit: int | None = None,
    default_embodiment: str = "robot",
    include_wrist: bool = True,
) -> list[AnnotationTask]:
    """Load exact main/wrist video URLs from a TSV manifest.

    Required columns are ``task``, ``episode_id``, and ``main_video_url``.
    ``episode_index``, ``wrist_video_url``, ``embodiment``, and
    ``task_instruction`` are optional. An empty wrist URL is the explicit
    single-head-view signal and disables Step 1 object detection.
    """
    if offset < 0:
        raise PipelineDataError("offset must be non-negative.")
    if limit is not None and limit < 0:
        raise PipelineDataError("limit must be non-negative.")
    if default_embodiment not in {"robot", "human"}:
        raise PipelineDataError(
            "default_embodiment must be either 'robot' or 'human'."
        )
    source = Path(input_path)
    if source.suffix.lower() != ".tsv":
        raise PipelineDataError(
            f"URL manifest requires a .tsv file, got {source.suffix!r}."
        )

    tasks: list[AnnotationTask] = []
    seen: set[tuple[str, str]] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"task", "episode_id", "main_video_url"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise PipelineDataError(
                f"{source}: URL manifest must contain task, episode_id, and "
                "main_video_url columns."
            )
        for line_number, row in enumerate(reader, start=2):
            task_name = (row.get("task") or "").strip()
            episode_id = (row.get("episode_id") or "").strip()
            main_url = (row.get("main_video_url") or "").strip()
            wrist_url = (
                (row.get("wrist_video_url") or "").strip() or None
            ) if include_wrist else None
            task_instruction = (
                (row.get("task_instruction") or "").strip() or None
            )
            embodiment = (
                (row.get("embodiment") or "").strip().lower()
                or default_embodiment
            )
            raw_index = (row.get("episode_index") or "").strip()
            try:
                episode_index = int(raw_index) if raw_index else len(tasks)
            except ValueError as exc:
                raise PipelineDataError(
                    f"{source}:{line_number}: invalid episode_index {raw_index!r}."
                ) from exc
            identity = (task_name, episode_id)
            if identity in seen:
                raise PipelineDataError(
                    f"Duplicate task/episode in {source}: "
                    f"{task_name}/{episode_id}."
                )
            seen.add(identity)
            try:
                tasks.append(AnnotationTask(
                    task=task_name,
                    episode_id=episode_id,
                    episode_index=episode_index,
                    wrist_video_url=wrist_url,
                    main_video_url=main_url,
                    embodiment=embodiment,
                    task_instruction=task_instruction,
                ))
            except PipelineDataError as exc:
                raise PipelineDataError(
                    f"{source}:{line_number}: {exc}"
                ) from exc

    tasks = tasks[offset:]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise PipelineDataError(f"{source}: no episodes found.")
    return tasks


def _build_indexed_tos_tasks(
    source: Path,
    indexed_episodes: list[tuple[str, int]],
    video_config: VideoAssetConfig | None,
    *,
    embodiment: str = "robot",
    include_wrist: bool = True,
) -> list[AnnotationTask]:
    video_config = video_config or VideoAssetConfig()
    tasks: list[AnnotationTask] = []
    seen: set[tuple[str, int]] = set()
    for task_name, episode_index in indexed_episodes:
        identity = (task_name, episode_index)
        if identity in seen:
            raise PipelineDataError(
                f"Duplicate task/episode index in {source}: "
                f"{task_name}/{episode_index}."
            )
        seen.add(identity)
        episode_id = f"episode_{episode_index:06d}"
        tasks.append(
            AnnotationTask(
                task=task_name,
                episode_id=episode_id,
                episode_index=episode_index,
                wrist_video_url=video_config.video_url(
                    task_name, episode_index, video_config.wrist_stream
                ) if include_wrist else None,
                main_video_url=video_config.video_url(
                    task_name, episode_index, video_config.main_stream
                ),
                embodiment=embodiment,
            )
        )
    if not tasks:
        raise PipelineDataError(f"{source}: no episodes found.")
    return tasks


def load_episode_mapping(path: str | Path) -> dict[str, dict[int, str]]:
    mapping_path = Path(path)
    try:
        value = json.loads(mapping_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PipelineDataError(f"Cannot read mapping {mapping_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineDataError(
            f"Invalid mapping JSON {mapping_path}: {exc.msg}"
        ) from exc
    if not isinstance(value, Mapping):
        raise PipelineDataError("Episode mapping root must be a JSON object.")

    normalized: dict[str, dict[int, str]] = {}
    for task_name, entries in value.items():
        if not isinstance(task_name, str) or not isinstance(entries, Mapping):
            raise PipelineDataError("Invalid task entry in episode mapping.")
        task_entries: dict[int, str] = {}
        for raw_index, episode_id in entries.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise PipelineDataError(
                    f"Invalid episode index {raw_index!r} for task {task_name}."
                ) from exc
            if index < 0 or not isinstance(episode_id, str) or not episode_id:
                raise PipelineDataError(
                    f"Invalid mapping value for {task_name}[{raw_index!r}]."
                )
            task_entries[index] = episode_id
        normalized[task_name] = task_entries
    return normalized


def _load_task_list(
    path: Path, mapping: Mapping[str, Mapping[int, str]]
) -> list[tuple[str, str]]:
    task_names = _load_task_names(path)
    identities: list[tuple[str, str]] = []
    for task_name in task_names:
        if task_name not in mapping:
            raise PipelineDataError(f"Task {task_name!r} is missing from episode mapping.")
        identities.extend(
            (task_name, episode_id)
            for _, episode_id in sorted(mapping[task_name].items())
        )
    return identities


def _load_task_names(path: Path) -> list[str]:
    task_names: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if value and not value.startswith("#"):
                task_names.append(value)
    return task_names


def _load_indexed_tos_task_names(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _load_task_names(path)
    if suffix == ".tsv":
        task_names: list[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            if not reader.fieldnames or "task" not in reader.fieldnames:
                raise PipelineDataError(
                    f"{path}: indexed TOS TSV must contain a 'task' column."
                )
            for line_number, row in enumerate(reader, start=2):
                task_name = (row.get("task") or "").strip()
                if not task_name:
                    raise PipelineDataError(
                        f"{path}:{line_number}: missing task name."
                    )
                task_names.append(task_name)
        return task_names
    raise PipelineDataError(
        f"Indexed TOS input requires a task-list .txt or .tsv file, got "
        f"{path.suffix!r}."
    )


def _load_episode_tsv(path: Path) -> list[tuple[str, str]]:
    identities: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if not reader.fieldnames or "task" not in reader.fieldnames:
            raise PipelineDataError(f"{path}: TSV must contain a 'task' column.")
        for line_number, row in enumerate(reader, start=2):
            task_name = (row.get("task") or "").strip()
            episode_id = _episode_id_from_row(row)
            if not task_name or not episode_id:
                raise PipelineDataError(
                    f"{path}:{line_number}: missing task or episode path/id."
                )
            identities.append((task_name, episode_id))
    return identities


def _episode_id_from_row(row: Mapping[str, Any]) -> str:
    direct = row.get("episode") or row.get("episode_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    episode_dir = row.get("episode_dir")
    if isinstance(episode_dir, str) and episode_dir.strip():
        return Path(episode_dir.strip()).name
    episode_mcap = row.get("episode_mcap") or row.get("mcap")
    if isinstance(episode_mcap, str) and episode_mcap.strip():
        return Path(episode_mcap.strip()).parent.name
    return ""


def _reverse_mapping(
    mapping: Mapping[str, Mapping[int, str]],
) -> dict[str, dict[str, int]]:
    reverse: dict[str, dict[str, int]] = {}
    for task_name, entries in mapping.items():
        task_reverse: dict[str, int] = {}
        for index, episode_id in entries.items():
            if episode_id in task_reverse:
                raise PipelineDataError(
                    f"Duplicate UUID {episode_id!r} in mapping task {task_name}."
                )
            task_reverse[episode_id] = index
        reverse[task_name] = task_reverse
    return reverse
