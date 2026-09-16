#!/usr/bin/env python3
"""Validate persisted GRD/STA result invariants without decoding source media."""

from __future__ import annotations

from collections import Counter
import json
import math
import re


GRD_FRAME_FPS = 2.0
GRD_FUTURE_FPS = 3.0
GRD_FUTURE_SECONDS = 4.0
GRD_MAX_OBJECTS = 4
ECOT_FPS = 2.0
ECOT_TEACHER_FPS = 0.5
ECOT_SAMPLE_TYPE = 'uniform_2fps_interval'
ECOT_SUBTASK_FIELDS = {'subtask', 'action', 'object', 'source', 'target'}
STA_POLICY_VERSION = 'cpa_final_no_intervening_contact_random_event_bbox_v1'
STA_SAMPLING_VERSION = 'cpa_final_no_intervening_contact_random_v1'
SHA256_PATTERN = re.compile(r'^[0-9a-f]{64}$')


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _bbox(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_number(item) for item in value)
        and 0 <= value[0] < value[2] <= 1000
        and 0 <= value[1] < value[3] <= 1000
    )


def _sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _same_identifier(left: object, right: object) -> bool:
    return str(left) == str(right)


def _event_name(event: dict) -> str:
    return ' '.join(
        str(event.get(field) or '').strip()
        for field in ('contact_verb', 'object_name')
        if str(event.get(field) or '').strip()
    )


def _parent_steps(parent_subtask: dict | None) -> list[dict]:
    if not isinstance(parent_subtask, dict):
        return []
    steps = parent_subtask.get('subtasks')
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, dict)]


def _parent_duration(parent_subtask: dict | None) -> float | None:
    if not isinstance(parent_subtask, dict):
        return None
    direct = parent_subtask.get('_source_duration_seconds')
    if _number(direct) and direct > 0:
        return float(direct)
    provider = parent_subtask.get('provider_output')
    if not isinstance(provider, dict):
        return None
    boundary = provider.get('boundary_mapping')
    candidates = [provider.get('source_duration_seconds')]
    if isinstance(boundary, dict):
        candidates.append(boundary.get('source_duration_seconds'))
    return next(
        (float(value) for value in candidates if _number(value) and value > 0),
        None,
    )


def _segment_common_errors(segment: object) -> tuple[Counter[str], dict, dict]:
    errors: Counter[str] = Counter()
    if not isinstance(segment, dict):
        errors['invalid_subtask_result_entry'] += 1
        return errors, {}, {}
    step = segment.get('subtask')
    if not isinstance(step, dict) or step.get('id') is None:
        errors['invalid_segment_subtask'] += 1
        step = {}
    instruction = step.get('subtask')
    if not isinstance(instruction, str) or not instruction.strip():
        errors['empty_segment_subtask_instruction'] += 1
    if segment.get('task_instruction_source') != 'subtask':
        errors['wrong_segment_instruction_source'] += 1
    if segment.get('task_instruction') != instruction:
        errors['wrong_segment_task_instruction'] += 1
    value = segment.get('result')
    if not isinstance(value, dict):
        errors['invalid_segment_result'] += 1
        value = {}
    return errors, step, value


def _validate_subtask(result: dict) -> Counter[str]:
    errors: Counter[str] = Counter()
    steps = result.get('subtasks')
    if not isinstance(steps, list) or not steps:
        errors['invalid_subtasks'] += 1
        return errors
    signature_fields = ('subtask', 'action', 'object', 'source', 'target')
    previous: dict | None = None
    for step in steps:
        if not isinstance(step, dict):
            errors['invalid_subtask'] += 1
            previous = None
            continue
        if previous is not None and all(
            ' '.join(str(previous.get(field) or '').split()).casefold()
            == ' '.join(str(step.get(field) or '').split()).casefold()
            for field in signature_fields
        ):
            errors['adjacent_identical_subtask'] += 1
        previous = step
    return errors


def _validate_ecot(result: dict, requests: object) -> Counter[str]:
    errors: Counter[str] = Counter()
    if result.get('task_instruction_source') != 'global_task':
        errors['wrong_ecot_instruction_source'] += 1
    interval = result.get('ecot_interval')
    sampled_count = result.get('sampled_frame_count')
    if result.get('ecot_sample_type') != ECOT_SAMPLE_TYPE:
        errors['wrong_ecot_sample_type'] += 1
    if result.get('sampling_fps') != ECOT_FPS:
        errors['wrong_ecot_sampling_fps'] += 1
    if not _integer(interval) or interval < 1:
        errors['invalid_ecot_interval'] += 1
        interval = 1
    if not _integer(sampled_count) or sampled_count < 1:
        errors['invalid_ecot_sampled_frame_count'] += 1
        sampled_count = 0
    contract = result.get('annotation_contract')
    legacy_teacher = isinstance(contract, dict) and (
        contract.get('schema') == 'privileged_video_ecot_atomic_v2'
        and contract.get('one_teacher_request_per_selected_frame') is True
        and contract.get('teacher_receives_complete_2fps_video') is True
        and contract.get('training_excludes_privileged_video') is True
    )
    current_teacher = isinstance(contract, dict) and (
        contract.get('schema') == 'privileged_video_ecot_atomic_v2'
        and contract.get('one_teacher_request_per_selected_frame') is True
        and contract.get('teacher_receives_complete_0_5fps_video') is True
        and contract.get('teacher_sampling_fps') == ECOT_TEACHER_FPS
        and contract.get('target_sampling_fps') == ECOT_FPS
        and contract.get('training_excludes_privileged_video') is True
    )
    if not (legacy_teacher or current_teacher):
        errors['invalid_ecot_annotation_contract'] += 1
    frames = result.get('frames')
    if not isinstance(frames, list):
        errors['invalid_ecot_frames'] += 1
        return errors
    expected = list(range(0, sampled_count, interval))
    actual = []
    for frame in frames:
        if not isinstance(frame, dict):
            errors['invalid_ecot_frame'] += 1
            continue
        index = frame.get('sampled_frame_index')
        timestamp = frame.get('sampled_time_seconds')
        if not _integer(index) or index < 0:
            errors['invalid_ecot_frame_index'] += 1
        else:
            actual.append(index)
            if not _number(timestamp) or not math.isclose(
                float(timestamp), index / ECOT_FPS, abs_tol=1e-6,
            ):
                errors['ecot_frame_time_index_mismatch'] += 1
        structured = frame.get('structured_ecot')
        if not isinstance(structured, dict) or set(structured) != {
            'scene_description', 'task_progress', 'current_subtask', 'atomic_action',
        }:
            errors['invalid_structured_ecot'] += 1
            continue
        if frame.get('scene_description') != structured.get('scene_description'):
            errors['ecot_scene_description_mismatch'] += 1
        from ecot_contract import validate_atomic_action
        try:
            validate_atomic_action(structured['atomic_action'])
        except ValueError:
            errors['invalid_ecot_atomic_action'] += 1
        for field in ('task_progress', 'current_subtask', 'atomic_action'):
            if frame.get(field) != structured.get(field):
                errors[f'ecot_{field}_mismatch'] += 1
        messages = frame.get('messages')
        if not isinstance(messages, list) or len(messages) != 2:
            errors['invalid_ecot_training_messages'] += 1
        else:
            try:
                if messages[1]['role'] != 'assistant' or json.loads(messages[1]['content']) != structured:
                    errors['ecot_training_message_target_mismatch'] += 1
                user_content = messages[0]['content']
                if (messages[0]['role'] != 'user' or not isinstance(user_content, list)
                        or len(user_content) != 2
                        or [item.get('type') for item in user_content] != ['image', 'text']
                        or user_content[0].get('image') != frame.get('training_input', {}).get('image')
                        or 'atomic_action' not in user_content[1].get('text', '')):
                    errors['ecot_training_message_input_mismatch'] += 1
            except (ValueError, TypeError, KeyError, AttributeError):
                errors['invalid_ecot_training_message_target'] += 1
        progress = structured.get('task_progress')
        if not isinstance(progress, str) or not progress.endswith((
            'Task complete.', 'Task not yet complete.',
        )):
            errors['invalid_ecot_task_progress'] += 1
        subtask = structured.get('current_subtask')
        if not isinstance(subtask, dict) or set(subtask) != ECOT_SUBTASK_FIELDS:
            errors['invalid_ecot_current_subtask'] += 1
        training_input = frame.get('training_input')
        if not isinstance(training_input, dict) or set(training_input) != {
            'task_instruction', 'image',
        }:
            errors['ecot_training_input_leaks_or_is_invalid'] += 1
        if frame.get('training_target') != structured:
            errors['ecot_training_target_mismatch'] += 1
    if actual != expected:
        errors['ecot_frame_selection_coverage_mismatch'] += 1
    if not isinstance(requests, list):
        errors['invalid_request_audit_list'] += 1
    else:
        teacher_requests = [
            item for item in requests
            if isinstance(item, dict)
            and item.get('stage') == 'privileged_complete_video_ecot'
        ]
        if len(teacher_requests) != len(frames):
            errors['ecot_request_count_mismatch'] += 1
        expected_media_kind = (
            'complete_2fps_episode_video_plus_target_image'
            if legacy_teacher else
            'complete_0.5fps_episode_video_plus_target_image'
        )
        if any(item.get('media_kind') != expected_media_kind
               for item in teacher_requests):
            errors['wrong_ecot_request_media_kind'] += 1
    return errors


def _validate_grd(
    result: dict,
    requests: object,
    parent_subtask: dict | None,
) -> Counter[str]:
    errors: Counter[str] = Counter()
    if result.get('task_instruction_source') != 'subtask':
        errors['wrong_result_instruction_source'] += 1
    if not _sha256(result.get('subtask_result_sha256')):
        errors['invalid_subtask_result_hash'] += 1
    segments = result.get('subtask_results')
    if not isinstance(segments, list):
        errors['invalid_subtask_results'] += 1
        return errors
    if 'inventory_review_summary' in result:
        from grd_inventory_review import POLICY, REVIEW_PROVIDERS, review_is_current
        summary = result['inventory_review_summary']
        review_model = summary.get('model') if isinstance(summary, dict) else None
        if review_model not in REVIEW_PROVIDERS:
            errors['invalid_grd_review_model'] += 1
        rejected = result.get('rejected_subtask_results')
        if not isinstance(rejected, list):
            errors['invalid_grd_rejected_segments'] += 1
            rejected = []
        for accepted, group in ((True, segments), (False, rejected)):
            for item in group:
                frames = item.get('result', {}).get('frames', []) if isinstance(item, dict) else []
                if len(frames) != 1 or not review_is_current(frames[0], review_model) or frames[0].get('registered') is not accepted:
                    errors['invalid_grd_inventory_review'] += 1
        expected = {'policy': POLICY, 'model': review_model, 'total_frames': len(segments) + len(rejected),
                    'accepted_frames': len(segments), 'rejected_frames': len(rejected)}
        if summary != expected:
            errors['invalid_grd_review_summary'] += 1
        # Coverage accounts for every original anchor, including audit-only rejections.
        segments = segments + rejected
    if not segments:
        errors['empty_subtask_results'] += 1

    actual_anchors: list[tuple[str, int]] = []
    observed_steps: dict[str, dict] = {}
    for segment in segments:
        common, step, value = _segment_common_errors(segment)
        errors.update(common)
        if step.get('id') is not None:
            observed_steps.setdefault(str(step['id']), step)
        scope = segment.get('media_scope') if isinstance(segment, dict) else None
        if not isinstance(scope, dict) or scope.get('kind') != 'grounding_frame':
            errors['invalid_grd_media_scope'] += 1
            scope = {}
        grounding_media = scope.get('grounding_media')
        if not isinstance(grounding_media, dict) or (
            grounding_media.get('kind') != 'single_current_image'
            or grounding_media.get('sampling_fps') != GRD_FRAME_FPS
        ):
            errors['invalid_grd_current_image_contract'] += 1
        future = scope.get('first_object_media')
        if not isinstance(future, dict) or (
            future.get('fps') != GRD_FUTURE_FPS
            or future.get('encoded_duration_seconds') != GRD_FUTURE_SECONDS
            or future.get('maximum_future_seconds') != GRD_FUTURE_SECONDS
            or future.get('anchor_fps') != GRD_FRAME_FPS
        ):
            errors['invalid_grd_future_video_contract'] += 1
        anchor_index = scope.get('anchor_frame_index')
        anchor_time = scope.get('anchor_time_seconds')
        if not _integer(anchor_index) or anchor_index < 0:
            errors['invalid_grd_anchor_index'] += 1
        if not _number(anchor_time) or anchor_time < 0:
            errors['invalid_grd_anchor_time'] += 1
        if _integer(anchor_index) and _number(anchor_time) and not math.isclose(
            float(anchor_time), anchor_index / GRD_FRAME_FPS, abs_tol=1e-6,
        ):
            errors['grd_anchor_time_index_mismatch'] += 1
        if step.get('id') is not None and _integer(anchor_index):
            actual_anchors.append((str(step['id']), anchor_index))

        frames = value.get('frames')
        if not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], dict):
            errors['grd_requires_exactly_one_output_frame'] += 1
            continue
        frame = frames[0]
        if frame.get('media_index') != anchor_index:
            errors['grd_frame_anchor_index_mismatch'] += 1
        if not (
            _number(frame.get('timestamp_seconds'))
            and _number(anchor_time)
            and math.isclose(
                float(frame['timestamp_seconds']), float(anchor_time), abs_tol=1e-6,
            )
        ):
            errors['grd_frame_anchor_time_mismatch'] += 1
        objects = frame.get('objects')
        if not isinstance(objects, list):
            errors['invalid_grd_inventory'] += 1
            objects = []
        elif len(objects) > GRD_MAX_OBJECTS:
            errors['grd_inventory_over_four_objects'] += 1
        names = []
        for item in objects:
            if not isinstance(item, dict) or not str(item.get('name') or '').strip():
                errors['invalid_grd_inventory_object'] += 1
                continue
            names.append(str(item['name']).strip())
            bbox = item.get('bbox_xyxy_1000')
            if not _bbox(bbox):
                errors['invalid_grd_inventory_bbox'] += 1
            elif item.get('center_xy_1000') != [
                int(round((bbox[0] + bbox[2]) / 2)),
                int(round((bbox[1] + bbox[3]) / 2)),
            ]:
                errors['invalid_grd_inventory_center'] += 1
            if item.get('clearly_visible') is not True:
                errors['grd_inventory_contains_nonvisible_object'] += 1
        selected = frame.get('task_first_object')
        selection = frame.get('task_first_object_selection')
        if selected is not None and selected not in names:
            errors['grd_first_object_not_in_inventory'] += 1
        if not isinstance(selection, dict) or selection.get('task_first_object') != selected:
            errors['invalid_grd_first_object_selection'] += 1

    if len(actual_anchors) != len(set(actual_anchors)):
        errors['duplicate_grd_anchor'] += 1
    parent_steps = _parent_steps(parent_subtask)
    coverage_steps = parent_steps or list(observed_steps.values())
    if coverage_steps and all(
        _number(step.get('start_time_seconds'))
        and _number(step.get('end_time_seconds'))
        for step in coverage_steps
    ):
        expected_anchors = set()
        parent_duration = _parent_duration(parent_subtask)
        for step in coverage_steps:
            start = float(step['start_time_seconds'])
            end = float(step['end_time_seconds'])
            if parent_duration is not None:
                end = min(end, parent_duration)
            first = max(0, int(math.ceil(start * GRD_FRAME_FPS - 1e-9)))
            final = int(math.ceil(end * GRD_FRAME_FPS - 1e-9))
            expected_anchors.update(
                (str(step.get('id')), index)
                for index in range(first, final)
                if index / GRD_FRAME_FPS < end
            )
        if set(actual_anchors) != expected_anchors:
            errors['grd_anchor_coverage_mismatch'] += 1

    if not isinstance(requests, list):
        errors['invalid_request_audit_list'] += 1
    else:
        current = [item for item in requests if isinstance(item, dict) and item.get('stage') == 'current_frame_inventory']
        future = [item for item in requests if isinstance(item, dict) and item.get('stage') == 'future_first_object_selection']
        if len(current) != len(segments):
            errors['grd_current_request_count_mismatch'] += 1
        if len(future) != len(segments):
            errors['grd_future_request_count_mismatch'] += 1
        if any(item.get('media_kind') != 'single_current_image' for item in current):
            errors['wrong_grd_current_request_media_kind'] += 1
        if any(item.get('media_kind') != 'future_video_3fps_4s' for item in future):
            errors['wrong_grd_future_request_media_kind'] += 1
    return errors


def _validate_sta(
    result: dict,
    requests: object,
    parent_subtask: dict | None,
) -> Counter[str]:
    errors: Counter[str] = Counter()
    if result.get('task_instruction_source') != 'subtask':
        errors['wrong_result_instruction_source'] += 1
    if not _sha256(result.get('subtask_result_sha256')):
        errors['invalid_subtask_result_hash'] += 1
    if result.get('contact_frame_authority') != 'cpa_final':
        errors['sta_contact_frame_not_cpa_final'] += 1
    if not _sha256(result.get('parent_cpa_result_sha256')):
        errors['invalid_sta_parent_cpa_hash'] += 1
    policy = result.get('sta_observation_policy')
    if not isinstance(policy, dict) or (
        policy.get('version') != STA_POLICY_VERSION
        or policy.get('contact_frame_authority') != 'cpa_final'
        or policy.get('bbox_input') != 'single_observation_image_plus_contact_event_name'
        or not _integer(policy.get('observations_per_contact'))
        or policy.get('observations_per_contact', 0) < 1
    ):
        errors['invalid_sta_observation_policy'] += 1
    segments = result.get('subtask_results')
    if not isinstance(segments, list):
        errors['invalid_subtask_results'] += 1
        return errors
    if not segments:
        errors['empty_subtask_results'] += 1

    segment_ids = []
    all_events: list[tuple[str, dict]] = []
    rejected_total = 0
    for segment in segments:
        common, step, value = _segment_common_errors(segment)
        errors.update(common)
        step_id = str(step.get('id'))
        segment_ids.append(step_id)
        if value.get('contact_frame_authority') != 'cpa_final':
            errors['sta_segment_contact_frame_not_cpa_final'] += 1
        if not _sha256(value.get('parent_cpa_segment_sha256')):
            errors['invalid_sta_parent_cpa_segment_hash'] += 1
        events = value.get('contact_events')
        if not isinstance(events, list):
            errors['invalid_sta_contact_events'] += 1
            events = []
        rejected = value.get('rejected_contact_event_proposals')
        if not isinstance(rejected, list):
            errors['invalid_sta_rejected_contact_events'] += 1
        else:
            rejected_total += len(rejected)
        seen_event_ids = set()
        for event in events:
            if not isinstance(event, dict):
                errors['invalid_sta_contact_event'] += 1
                continue
            event_id = str(event.get('event_id') or '')
            if not event_id or event_id in seen_event_ids:
                errors['invalid_or_duplicate_sta_event_id'] += 1
            seen_event_ids.add(event_id)
            all_events.append((step_id, event))
            if event.get('accepted') is not True:
                errors['sta_contains_nonaccepted_contact_event'] += 1
            if event.get('contact_frame_source') != 'cpa_final':
                errors['sta_event_contact_frame_not_cpa_final'] += 1
            selection = event.get('contact_frame_selection')
            if not isinstance(selection, dict) or selection.get('valid_contact') is not True:
                errors['invalid_sta_contact_frame_selection'] += 1
            else:
                if not _same_identifier(selection.get('requested_subtask_id'), step.get('id')):
                    errors['sta_requested_subtask_mismatch'] += 1
                candidate = selection.get('selected_candidate')
                if not isinstance(candidate, dict) or not _same_identifier(candidate.get('subtask_id'), step.get('id')):
                    errors['sta_selected_subtask_mismatch'] += 1
            if not _same_identifier(event.get('contact_subtask_id'), step.get('id')):
                errors['sta_contact_subtask_mismatch'] += 1
            if not _number(event.get('contact_time_seconds')):
                errors['invalid_sta_contact_time'] += 1
            if event.get('contact_event_name') != _event_name(event):
                errors['wrong_sta_contact_event_name'] += 1
            sampling = event.get('observation_sampling')
            if not isinstance(sampling, dict) or (
                sampling.get('version') != STA_SAMPLING_VERSION
                or sampling.get('no_intervening_cpa_final_contact') is not True
            ):
                errors['invalid_sta_observation_sampling'] += 1
                sampling = {}
            observations = event.get('observations')
            if not isinstance(observations, list):
                errors['invalid_sta_observations'] += 1
                observations = []
            selected_count = sampling.get('selected_frame_count')
            grounded_count = sampling.get('grounded_observation_count')
            invisible_count = sampling.get('invisible_observation_count')
            if not all(_integer(item) and item >= 0 for item in (
                selected_count, grounded_count, invisible_count,
            )) or (
                grounded_count != len(observations)
                or grounded_count + invisible_count != selected_count
            ):
                errors['sta_observation_count_mismatch'] += 1
            previous_time = sampling.get('previous_contact_time_seconds')
            contact_time = event.get('contact_time_seconds')
            for observation in observations:
                if not isinstance(observation, dict):
                    errors['invalid_sta_observation'] += 1
                    continue
                if not _bbox(observation.get('bbox_xyxy_1000')):
                    errors['invalid_sta_observation_bbox'] += 1
                if observation.get('bbox_source') != 'single_image_vlm_event_name_grounding':
                    errors['wrong_sta_bbox_source'] += 1
                if observation.get('contact_frame_source') != 'cpa_final':
                    errors['sta_observation_contact_frame_not_cpa_final'] += 1
                if observation.get('no_intervening_cpa_final_contact') is not True:
                    errors['sta_observation_allows_intervening_contact'] += 1
                if observation.get('contact_event_name') != event.get('contact_event_name'):
                    errors['sta_observation_event_name_mismatch'] += 1
                if observation.get('next_contact_noun') != event.get('object_name'):
                    errors['sta_observation_noun_mismatch'] += 1
                if observation.get('next_contact_verb') != event.get('contact_verb'):
                    errors['sta_observation_verb_mismatch'] += 1
                when = observation.get('time_seconds')
                if _number(when) and _number(contact_time):
                    if not float(when) < float(contact_time):
                        errors['sta_observation_not_before_contact'] += 1
                    if _number(previous_time) and not float(when) > float(previous_time):
                        errors['sta_observation_not_after_previous_contact'] += 1
                    ttc = observation.get('time_to_contact_seconds')
                    if not _number(ttc) or not math.isclose(
                        float(ttc), float(contact_time) - float(when), abs_tol=2e-6,
                    ):
                        errors['sta_observation_ttc_mismatch'] += 1

    parent_steps = _parent_steps(parent_subtask)
    if parent_steps:
        expected_ids = [str(step.get('id')) for step in parent_steps]
        if segment_ids != expected_ids:
            errors['sta_subtask_segment_coverage_mismatch'] += 1
    summary = result.get('contact_frame_summary')
    if not isinstance(summary, dict) or (
        summary.get('accepted') != len(all_events)
        or summary.get('rejected') != rejected_total
    ):
        errors['sta_contact_frame_summary_mismatch'] += 1
    timed_events = [
        event for _, event in all_events
        if _number(event.get('contact_time_seconds'))
    ]
    for event in timed_events:
        contact_time = float(event['contact_time_seconds'])
        expected_previous = max((
            float(other['contact_time_seconds'])
            for other in timed_events
            if other is not event
            and float(other['contact_time_seconds']) < contact_time - 1e-9
        ), default=None)
        reported_previous = (
            event.get('observation_sampling') or {}
        ).get('previous_contact_time_seconds')
        if expected_previous is None:
            if reported_previous is not None:
                errors['sta_previous_contact_mismatch'] += 1
        elif not (
            _number(reported_previous)
            and math.isclose(
                float(reported_previous), expected_previous, abs_tol=2e-6,
            )
        ):
            errors['sta_previous_contact_mismatch'] += 1
    if isinstance(requests, list):
        sync = [item for item in requests if isinstance(item, dict) and item.get('stage') == 'cpa_final_random_sta_observation_sync']
        if len(sync) != 1 or sync[0].get('parent_cpa_result_sha256') != result.get('parent_cpa_result_sha256'):
            errors['invalid_sta_cpa_sync_request'] += 1
        selected_total = sum(
            int(event.get('observation_sampling', {}).get('selected_frame_count', 0))
            for _, event in all_events
            if _integer(event.get('observation_sampling', {}).get('selected_frame_count'))
        )
        bbox_requests = [item for item in requests if isinstance(item, dict) and item.get('stage') == 'cpa_final_random_observation_bbox']
        if len(bbox_requests) != selected_total:
            errors['sta_bbox_request_count_mismatch'] += 1
        if any(item.get('media_kind') != 'single_random_pre_contact_image' for item in bbox_requests):
            errors['wrong_sta_bbox_request_media_kind'] += 1
    else:
        errors['invalid_request_audit_list'] += 1
    return errors


def validate_annotation_result_contract(
    task: str,
    result: object,
    requests: object,
    parent_subtask: dict | None = None,
) -> Counter[str]:
    """Return invariant failure counts for one persisted annotation result."""
    if not isinstance(result, dict):
        return Counter({'missing_result': 1})
    if task == 'subtask':
        return _validate_subtask(result)
    if task == 'ecot':
        return _validate_ecot(result, requests)
    if task == 'grd':
        return _validate_grd(result, requests, parent_subtask)
    if task == 'sta':
        return _validate_sta(result, requests, parent_subtask)
    return Counter()
