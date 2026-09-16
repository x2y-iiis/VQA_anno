"""Represent a learner's 5 FPS history while preserving its final query frame."""
from __future__ import annotations

import copy
import math


STUDENT_FPS = 5.0


def student_frame_indices(start_frame: int, query_frame: int, source_fps: float) -> list[int]:
    """Sample backward from the existing query frame, never revealing future frames."""
    if (isinstance(start_frame, bool) or isinstance(query_frame, bool)
            or not isinstance(start_frame, int) or not isinstance(query_frame, int)
            or start_frame < 0 or query_frame < start_frame
            or not math.isfinite(source_fps) or source_fps < STUDENT_FPS):
        raise ValueError('cpa_student_video_invalid_interval_or_fps')
    # The unchanged observation/query frame anchors the 5 Hz lattice. Rounding
    # is required when source FPS is not an integer multiple of five.
    count = int(math.floor((query_frame - start_frame) * STUDENT_FPS / source_fps))
    return [query_frame - round(k * source_fps / STUDENT_FPS)
            for k in range(count, -1, -1)]


def with_student_video(question: dict, *, start_frame: int, source_fps: float) -> dict:
    """Replace image-only known conditions; keep the question target and answer."""
    result = copy.deepcopy(question)
    query = int(result['query_frame_index'])
    contact = int(result['contact_frame_index'])
    if query >= contact:
        raise ValueError('cpa_student_query_must_precede_contact')
    indices = student_frame_indices(start_frame, query, source_fps)
    result['known_conditions'] = {
        'type': 'video', 'sampling_fps': STUDENT_FPS,
        'source_frame_indices': indices,
        'frame_index_convention': 'zero_based_original_video',
    }
    result['question_frame_index'] = query
    result['tracking_target_frame_indices'] = [query]
    if 'learner_inputs' in result:
        result['learner_inputs'] = [
            'task_instruction', 'precontact_video_5fps_with_final_frame_choice_markers', 'question',
        ]
    if 'visibility_policy' in result:
        result['visibility_policy'] = (
            'learner_sees_only_5fps_precontact_video_ending_at_the_query_frame_'
            'with_neutral_choice_markers_on_that_final_frame_plus_task_and_question'
        )
    result['question'] = (
        'Watch the provided 5 FPS video. Answer the following question only for '
        'its final frame. ' + str(question['question'])
    )
    return result


def build_student_questions(audit: dict, observation: dict, segmentation: dict,
                            *, start_frame: int, source_fps: float,
                            legacy_root=None) -> list[dict]:
    """Reuse the existing participant/choice policy, changing only learner context."""
    import hashlib
    import importlib.util
    from pathlib import Path

    project = Path(__file__).resolve().parents[1]
    root = Path(legacy_root or project/'third_party/robot_vqa_sta_cpa')
    source = root / 'src/generate_robot.py'
    spec = importlib.util.spec_from_file_location('cpa_legacy_question_builder', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.build_cpa_multiple_choice_vqa_records(audit, observation, segmentation)
    result = [with_student_video(question, start_frame=start_frame, source_fps=source_fps)
              for question in original]
    for question in result:
        question['question_builder_provenance'] = {
            'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'function': 'build_cpa_multiple_choice_vqa_records',
            'change': 'learner_context_only',
        }
    return result
