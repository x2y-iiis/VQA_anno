"""Stage orchestration, checkpoint recovery, and batch concurrency."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .client import LASClient
from .config import PipelineConfig
from .models import AnnotationTask, PipelineDataError, SampleResult
from .parsing import parse_actions_response, parse_objects_response
from .postprocessing import (
    ADJACENT_IDENTICAL_MERGE_VERSION,
    DoubaoTextClient,
    NONE_SEGMENT_MERGE_VERSION,
    annotation_digest,
    build_rewrite_request,
    merge_adjacent_identical_segments,
    merge_none_segments,
    parse_rewrite_response,
    rewrite_prompt_digest,
)
from .prompts import build_prompt_context, load_vocabulary, prompt_policy_digest
from .requests import build_action_request, build_object_request
from .review_video import (
    ReviewVideoConfig,
    render_review_video,
    review_artifact_matches,
    review_signature,
    review_source_path,
)
from .storage import (
    atomic_write_json,
    checkpoint_task_id,
    clear_stage_files,
    load_objects_checkpoint,
    read_json_or_none,
    stage_paths,
    task_id,
)
from .task_io import load_tasks


def process_sample(
    task: AnnotationTask,
    client: LASClient,
    output_dir: str | Path,
    *,
    config: PipelineConfig | None = None,
    force: bool = False,
    postprocess_client_factory: Callable[[], DoubaoTextClient] = DoubaoTextClient,
    review_renderer: Callable[..., None] = render_review_video,
) -> SampleResult:
    config = config or PipelineConfig()
    item_dir = Path(output_dir) / task.task / task.episode_id
    item_dir.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(item_dir)
    vocabulary = load_vocabulary(config.skill_profile, config.skills_file)

    annotation_checkpoint = read_json_or_none(paths["annotation.json"])
    if force:
        clear_stage_files(paths)
    elif config.stop_after == "step1" and _step1_outputs_present(
        paths, task, config
    ):
        return SampleResult(task.task, task.episode_id, "skipped")
    elif config.stop_after == "context" and _context_outputs_present(
        paths, vocabulary.digest, task, config
    ):
        return SampleResult(task.task, task.episode_id, "skipped")
    elif (
        config.stop_after == "step3"
        and _annotation_matches(
            annotation_checkpoint, task, config, vocabulary.digest
        )
        and _step3_outputs_present(paths, vocabulary.digest, task, config)
    ):
        return SampleResult(task.task, task.episode_id, "skipped")
    elif _annotation_matches(
        annotation_checkpoint, task, config, vocabulary.digest
    ) and (
        not config.postprocess
        or _postprocessed_matches(
            read_json_or_none(paths["postprocessed_annotation.json"]),
            annotation_checkpoint,
            config,
        )
    ) and (
        not config.render_review_videos
        or _review_outputs_match(
            task,
            annotation_checkpoint,
            read_json_or_none(paths["postprocessed_annotation.json"]),
            paths,
            config,
        )
    ) and _required_stage_outputs_present(paths, config, task):
        return SampleResult(task.task, task.episode_id, "skipped")

    stage = "step1"
    try:
        objects = None if force else load_objects_checkpoint(paths["objects.json"])
        step1_record = None if force else read_json_or_none(paths["step1_raw.json"])
        if not task.has_wrist_view:
            expected_objects = _no_wrist_objects_checkpoint(task)
            if read_json_or_none(paths["objects.json"]) != expected_objects:
                _clear_downstream_after_step1(paths)
                atomic_write_json(paths["objects.json"], expected_objects)
            paths["step1_raw.json"].unlink(missing_ok=True)
            objects = []
            step1_record = None
        else:
            step1_request = build_object_request(task, config)
            if isinstance(step1_record, Mapping) and isinstance(
                step1_record.get("response"), Mapping
            ) and _step1_record_matches(step1_record, step1_request, config):
                if objects is None:
                    objects = parse_objects_response(step1_record["response"])
                    atomic_write_json(paths["objects.json"], {"objects": objects})
            else:
                preserve_downstream = step1_record is None and _annotation_matches(
                    annotation_checkpoint,
                    task,
                    config,
                    vocabulary.digest,
                )
                step1_response = client.call_operator(step1_request)
                paths["objects.json"].unlink(missing_ok=True)
                if not preserve_downstream:
                    _clear_downstream_after_step1(paths)
                step1_record = {
                    "request": step1_request,
                    "response": step1_response,
                }
                atomic_write_json(
                    paths["step1_raw.json"],
                    step1_record,
                )
                objects = parse_objects_response(step1_response)
                atomic_write_json(paths["objects.json"], {"objects": objects})

        if config.stop_after == "step1":
            _clear_error_for_stage(paths["error.json"], "step1")
            return SampleResult(task.task, task.episode_id, "success")

        stage = "context"
        context_record = None if force else read_json_or_none(paths["context.json"])
        if isinstance(context_record, Mapping) and isinstance(
            context_record.get("prompt_context"), str
        ) and _context_matches(context_record, vocabulary.digest, task):
            prompt_context = context_record["prompt_context"]
        else:
            prompt_context = build_prompt_context(
                objects,
                vocabulary,
                embodiment=task.embodiment,
                has_wrist_view=task.has_wrist_view,
                task_instruction=task.instruction,
            )
            atomic_write_json(
                paths["context.json"],
                {
                    "embodiment": task.embodiment,
                    "has_wrist_view": task.has_wrist_view,
                    "task_instruction": task.instruction,
                    "prompt_policy_digest": prompt_policy_digest(
                        task.embodiment, task.has_wrist_view
                    ),
                    "skill_vocabulary": {
                        "name": vocabulary.name,
                        "source": vocabulary.source,
                        "digest": vocabulary.digest,
                        "count": len(vocabulary.items),
                    },
                    "prompt_context": prompt_context,
                },
            )

        if config.stop_after == "context":
            _clear_error_for_stages(paths["error.json"], {"step1", "context"})
            return SampleResult(task.task, task.episode_id, "success")

        stage = "step3"
        step3_request = build_action_request(task, prompt_context, config)
        step3_record = None if force else read_json_or_none(paths["step3_raw.json"])
        if isinstance(step3_record, Mapping) and isinstance(
            step3_record.get("response"), Mapping
        ) and step3_record.get("request") == step3_request:
            step3_response = step3_record["response"]
        else:
            step3_response = client.call_operator(step3_request)
            atomic_write_json(
                paths["step3_raw.json"],
                {"request": step3_request, "response": step3_response},
            )
        actions = parse_actions_response(step3_response)

        stage = "annotation"
        annotation = {
            "task": task.task,
            "task_instruction": task.instruction,
            "episode_id": task.episode_id,
            "episode_index": task.episode_index,
            "embodiment": task.embodiment,
            "inputs": {
                "wrist_video_url": task.wrist_video_url,
                "main_video_url": task.main_video_url,
            },
            "object_detection": {
                "performed": task.has_wrist_view,
                "source_view": "wrist" if task.has_wrist_view else None,
                "skip_reason": (
                    None if task.has_wrist_view else "no_wrist_view"
                ),
            },
            "model": config.model,
            "step1_model": config.step1_model or config.model,
            "action_template": config.action_template,
            "skill_vocabulary": {
                "name": vocabulary.name,
                "source": vocabulary.source,
                "digest": vocabulary.digest,
                "count": len(vocabulary.items),
            },
            "objects": objects,
            "task_description": actions["task_description"],
            "segments": actions["segments"],
            "las_task_ids": {
                "step1": (
                    checkpoint_task_id(step1_record, paths["step1_raw.json"])
                    if task.has_wrist_view
                    else None
                ),
                "step3": task_id(step3_response),
            },
        }
        atomic_write_json(paths["annotation.json"], annotation)

        if config.stop_after == "step3":
            _clear_error_for_stages(
                paths["error.json"],
                {"step1", "context", "step3", "annotation"},
            )
            return SampleResult(task.task, task.episode_id, "success")

        if config.render_review_videos:
            stage = "step3_review"
            _run_review_render(
                "step3",
                task,
                annotation,
                paths,
                config,
                review_renderer,
            )

        if config.postprocess:
            stage = "step4"
            _run_postprocessing(
                annotation,
                paths,
                config,
                postprocess_client_factory,
            )
            if config.render_review_videos:
                stage = "step4_review"
                postprocessed = read_json_or_none(
                    paths["postprocessed_annotation.json"]
                )
                if not isinstance(postprocessed, Mapping):
                    raise PipelineDataError(
                        "postprocessed_annotation.json is invalid for review rendering."
                    )
                _run_review_render(
                    "step4",
                    task,
                    postprocessed,
                    paths,
                    config,
                    review_renderer,
                )
        paths["error.json"].unlink(missing_ok=True)
        return SampleResult(task.task, task.episode_id, "success")
    except Exception as exc:
        atomic_write_json(
            paths["error.json"],
            {
                "task": task.task,
                "episode_id": task.episode_id,
                "episode_index": task.episode_index,
                "stage": stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
        return SampleResult(task.task, task.episode_id, "failed", str(exc))


def run_batch(
    tasks: Sequence[AnnotationTask],
    output_dir: str | Path,
    *,
    workers: int = 2,
    config: PipelineConfig | None = None,
    force: bool = False,
    client_factory: Callable[[], LASClient] = LASClient,
    postprocess_client_factory: Callable[[], DoubaoTextClient] = DoubaoTextClient,
    review_renderer: Callable[..., None] = render_review_video,
) -> list[SampleResult]:
    if workers <= 0:
        raise ValueError("workers must be greater than 0.")
    config = config or PipelineConfig()
    load_vocabulary(config.skill_profile, config.skills_file)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    local = threading.local()
    clients: list[LASClient] = []
    postprocess_clients: list[DoubaoTextClient] = []
    clients_lock = threading.Lock()

    def initialize_worker() -> None:
        client = client_factory()
        local.client = client
        with clients_lock:
            clients.append(client)

    def run_one(task: AnnotationTask) -> SampleResult:
        def get_postprocess_client() -> DoubaoTextClient:
            client = getattr(local, "postprocess_client", None)
            if client is None:
                client = postprocess_client_factory()
                local.postprocess_client = client
                with clients_lock:
                    postprocess_clients.append(client)
            return client

        return process_sample(
            task,
            local.client,
            output_dir,
            config=config,
            force=force,
            postprocess_client_factory=get_postprocess_client,
            review_renderer=review_renderer,
        )

    results: list[SampleResult] = []
    try:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="las-annotation",
            initializer=initialize_worker,
        ) as executor:
            future_to_task = {executor.submit(run_one, task): task for task in tasks}
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = SampleResult(
                        task.task, task.episode_id, "failed", str(exc)
                    )
                results.append(result)
                message = f"[{result.status.upper()}] {result.identifier}"
                if result.error:
                    message += f": {result.error}"
                print(message, flush=True)
    finally:
        for client in clients:
            client.close()
        for client in postprocess_clients:
            client.close()
    return results


def _run_postprocessing(
    annotation: Mapping[str, object],
    paths: Mapping[str, Path],
    config: PipelineConfig,
    client_factory: Callable[[], DoubaoTextClient],
) -> None:
    raw_segments = annotation.get("segments")
    if not isinstance(raw_segments, list):
        raise PipelineDataError("annotation segments must be a list for Step 4.")
    segments = (
        merge_none_segments(raw_segments)
        if config.merge_none_segments
        else [dict(segment) for segment in raw_segments]
    )

    if config.rewrite_descriptions:
        descriptions = []
        skills = []
        for index, segment in enumerate(segments):
            description = segment.get("description")
            if not isinstance(description, str) or not description.strip():
                raise PipelineDataError(
                    f"segments[{index}].description is required for Step 4."
                )
            descriptions.append(description)
            skills.append(segment.get("skill"))
        request = build_rewrite_request(
            descriptions,
            config.postprocess_model,
            skills=skills,
            embodiment=str(annotation.get("embodiment") or "robot"),
        )
        rewritten: list[str] | None = None
        record = read_json_or_none(paths["step4_raw.json"])
        if (
            isinstance(record, Mapping)
            and record.get("request") == request
            and isinstance(record.get("response"), Mapping)
        ):
            response = record["response"]
            try:
                rewritten = parse_rewrite_response(response, len(segments))
            except PipelineDataError:
                rewritten = None
        if rewritten is None:
            response = client_factory().call(request)
            atomic_write_json(
                paths["step4_raw.json"],
                {"request": request, "response": response},
            )
            rewritten = parse_rewrite_response(response, len(segments))
        for segment, description in zip(segments, rewritten):
            segment["description"] = description
    else:
        paths["step4_raw.json"].unlink(missing_ok=True)

    segment_count_before_identical_merge = len(segments)
    segments = merge_adjacent_identical_segments(segments)

    result = dict(annotation)
    result["segments"] = segments
    result["postprocessing"] = {
        "source": "annotation.json",
        "source_annotation_digest": annotation_digest(annotation),
        "english_rewrite": config.rewrite_descriptions,
        "model": config.postprocess_model if config.rewrite_descriptions else None,
        "prompt_digest": (
            rewrite_prompt_digest(str(annotation.get("embodiment") or "robot"))
            if config.rewrite_descriptions
            else None
        ),
        "embodiment": str(annotation.get("embodiment") or "robot"),
        "merge_none_segments": config.merge_none_segments,
        "none_segment_merge_version": (
            NONE_SEGMENT_MERGE_VERSION if config.merge_none_segments else None
        ),
        "merge_adjacent_identical_segments": True,
        "adjacent_identical_merge_version": ADJACENT_IDENTICAL_MERGE_VERSION,
        "adjacent_identical_segments_removed": (
            segment_count_before_identical_merge - len(segments)
        ),
    }
    atomic_write_json(paths["postprocessed_annotation.json"], result)


def _run_review_render(
    key: str,
    task: AnnotationTask,
    annotation: Mapping[str, Any],
    paths: Mapping[str, Path],
    config: PipelineConfig,
    renderer: Callable[..., None],
) -> None:
    review_config = _review_config(config)
    source = review_source_path(
        task,
        config.review_video_root,
        config.review_video_stream,
        config.review_source_kind,
    )
    destination = paths[f"{key}_review.mp4"]
    manifest = read_json_or_none(paths["review_videos.json"])
    if review_artifact_matches(
        manifest, key, annotation, source, destination, review_config
    ):
        return
    segments = annotation.get("segments")
    if not isinstance(segments, list):
        raise PipelineDataError(f"{key} annotation segments must be a list.")
    renderer(source, segments, destination, review_config)
    manifest_value = dict(manifest) if isinstance(manifest, Mapping) else {}
    manifest_value[key] = {
        "signature": review_signature(annotation, source, review_config),
        "output": destination.name,
    }
    atomic_write_json(paths["review_videos.json"], manifest_value)


def _review_outputs_match(
    task: AnnotationTask,
    annotation: object,
    postprocessed: object,
    paths: Mapping[str, Path],
    config: PipelineConfig,
) -> bool:
    if not isinstance(annotation, Mapping):
        return False
    review_config = _review_config(config)
    source = review_source_path(
        task,
        config.review_video_root,
        config.review_video_stream,
        config.review_source_kind,
    )
    manifest = read_json_or_none(paths["review_videos.json"])
    if not review_artifact_matches(
        manifest,
        "step3",
        annotation,
        source,
        paths["step3_review.mp4"],
        review_config,
    ):
        return False
    return not config.postprocess or review_artifact_matches(
        manifest,
        "step4",
        postprocessed,
        source,
        paths["step4_review.mp4"],
        review_config,
    )


def _review_config(config: PipelineConfig) -> ReviewVideoConfig:
    return ReviewVideoConfig(
        font=config.review_font,
        font_size=config.review_font_size,
        crf=config.review_crf,
        preset=config.review_preset,
        target_fps=config.review_fps,
        camera_topic=config.review_video_stream,
        source_kind=config.review_source_kind,
    )


def _context_matches(
    record: Mapping[str, object],
    digest: str,
    task: AnnotationTask,
) -> bool:
    vocabulary = record.get("skill_vocabulary")
    return (
        isinstance(vocabulary, Mapping)
        and vocabulary.get("digest") == digest
        and record.get("embodiment", "robot") == task.embodiment
        and record.get("has_wrist_view", True) is task.has_wrist_view
        and (
            task.has_wrist_view
            or record.get("task_instruction") == task.instruction
        )
        and record.get("prompt_policy_digest")
        == prompt_policy_digest(task.embodiment, task.has_wrist_view)
    )


def _required_stage_outputs_present(
    paths: Mapping[str, Path],
    config: PipelineConfig,
    task: AnnotationTask,
) -> bool:
    context = read_json_or_none(paths["context.json"])
    step3 = read_json_or_none(paths["step3_raw.json"])
    if not (
        _step1_outputs_present(paths, task, config)
        and isinstance(context, Mapping)
        and isinstance(context.get("prompt_context"), str)
        and isinstance(step3, Mapping)
        and isinstance(step3.get("request"), Mapping)
        and isinstance(step3.get("response"), Mapping)
    ):
        return False

    if config.postprocess and config.rewrite_descriptions:
        step4 = read_json_or_none(paths["step4_raw.json"])
        if not (
            isinstance(step4, Mapping)
            and isinstance(step4.get("request"), Mapping)
            and isinstance(step4.get("response"), Mapping)
        ):
            return False
    return not paths["error.json"].exists()


def _no_wrist_objects_checkpoint(task: AnnotationTask) -> dict[str, object]:
    return {
        "objects": [],
        "source": "skipped_no_wrist_view",
        "embodiment": task.embodiment,
    }


def _step1_outputs_present(
    paths: Mapping[str, Path],
    task: AnnotationTask,
    config: PipelineConfig,
) -> bool:
    if not task.has_wrist_view:
        return read_json_or_none(paths["objects.json"]) == (
            _no_wrist_objects_checkpoint(task)
        )
    step1 = read_json_or_none(paths["step1_raw.json"])
    objects = read_json_or_none(paths["objects.json"])
    return (
        isinstance(step1, Mapping)
        and isinstance(step1.get("response"), Mapping)
        and _step1_record_matches(
            step1,
            build_object_request(task, config),
            config,
        )
        and isinstance(objects, Mapping)
        and isinstance(objects.get("objects"), list)
    )


def _step1_record_matches(
    record: Mapping[str, object],
    expected_request: Mapping[str, object],
    config: PipelineConfig,
) -> bool:
    request = record.get("request")
    if request == expected_request:
        return True
    # Historical checkpoints sometimes stored an empty request. Preserve their
    # old resume behavior only when Step 1 has not been explicitly separated
    # from the Step 3 model; an explicit --step1-model always requires proof.
    return config.step1_model is None and request == {}


def _step3_outputs_present(
    paths: Mapping[str, Path],
    vocabulary_digest: str,
    task: AnnotationTask,
    config: PipelineConfig,
) -> bool:
    context = read_json_or_none(paths["context.json"])
    step3 = read_json_or_none(paths["step3_raw.json"])
    annotation = read_json_or_none(paths["annotation.json"])
    vocabulary = (
        annotation.get("skill_vocabulary")
        if isinstance(annotation, Mapping)
        else None
    )
    return (
        _step1_outputs_present(paths, task, config)
        and isinstance(context, Mapping)
        and isinstance(context.get("prompt_context"), str)
        and _context_matches(context, vocabulary_digest, task)
        and isinstance(step3, Mapping)
        and isinstance(step3.get("request"), Mapping)
        and isinstance(step3.get("response"), Mapping)
        and isinstance(annotation, Mapping)
        and isinstance(annotation.get("segments"), list)
        and isinstance(vocabulary, Mapping)
        and vocabulary.get("digest") == vocabulary_digest
    )


def _context_outputs_present(
    paths: Mapping[str, Path],
    vocabulary_digest: str,
    task: AnnotationTask,
    config: PipelineConfig,
) -> bool:
    context = read_json_or_none(paths["context.json"])
    return (
        _step1_outputs_present(paths, task, config)
        and isinstance(context, Mapping)
        and isinstance(context.get("prompt_context"), str)
        and _context_matches(context, vocabulary_digest, task)
    )


def _clear_error_for_stage(path: Path, stage: str) -> None:
    error = read_json_or_none(path)
    if isinstance(error, Mapping) and error.get("stage") == stage:
        path.unlink(missing_ok=True)


def _clear_downstream_after_step1(paths: Mapping[str, Path]) -> None:
    for name in (
        "context.json",
        "step3_raw.json",
        "annotation.json",
        "step4_raw.json",
        "postprocessed_annotation.json",
        "step3_review.mp4",
        "step4_review.mp4",
        "review_videos.json",
    ):
        paths[name].unlink(missing_ok=True)


def _clear_error_for_stages(path: Path, stages: set[str]) -> None:
    error = read_json_or_none(path)
    if isinstance(error, Mapping) and error.get("stage") in stages:
        path.unlink(missing_ok=True)


def _annotation_matches(
    value: object,
    task: AnnotationTask,
    config: PipelineConfig,
    vocabulary_digest: str,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    inputs = value.get("inputs")
    vocabulary = value.get("skill_vocabulary")
    recorded_step1_model = value.get("step1_model", value.get("model"))
    return (
        value.get("task") == task.task
        and value.get("task_instruction", value.get("task")) == task.instruction
        and value.get("episode_id") == task.episode_id
        and value.get("episode_index") == task.episode_index
        and value.get("embodiment", "robot") == task.embodiment
        and value.get("model") == config.model
        and recorded_step1_model == (config.step1_model or config.model)
        and value.get("action_template") == config.action_template
        and isinstance(inputs, Mapping)
        and inputs.get("wrist_video_url") == task.wrist_video_url
        and inputs.get("main_video_url") == task.main_video_url
        and isinstance(vocabulary, Mapping)
        and vocabulary.get("digest") == vocabulary_digest
    )


def _postprocessed_matches(
    value: object,
    annotation: object,
    config: PipelineConfig,
) -> bool:
    if not isinstance(value, Mapping) or not isinstance(annotation, Mapping):
        return False
    metadata = value.get("postprocessing")
    embodiment = str(annotation.get("embodiment") or "robot")
    return (
        isinstance(metadata, Mapping)
        and metadata.get("source_annotation_digest") == annotation_digest(annotation)
        and metadata.get("english_rewrite") is config.rewrite_descriptions
        and metadata.get("model")
        == (config.postprocess_model if config.rewrite_descriptions else None)
        and metadata.get("prompt_digest")
        == (
            rewrite_prompt_digest(embodiment)
            if config.rewrite_descriptions
            else None
        )
        and metadata.get("embodiment", "robot") == embodiment
        and metadata.get("merge_none_segments") is config.merge_none_segments
        and metadata.get("none_segment_merge_version")
        == (NONE_SEGMENT_MERGE_VERSION if config.merge_none_segments else None)
        and metadata.get("merge_adjacent_identical_segments") is True
        and metadata.get("adjacent_identical_merge_version")
        == ADJACENT_IDENTICAL_MERGE_VERSION
        and isinstance(value.get("segments"), list)
    )


__all__ = ["load_tasks", "process_sample", "run_batch"]
