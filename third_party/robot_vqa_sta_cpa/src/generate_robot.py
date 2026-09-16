#!/usr/bin/env python3
"""Build Robot grounding, ECoT, and short-term interaction anticipation labels."""
from __future__ import annotations

import argparse
import base64
import copy
import concurrent.futures
import hashlib
import html
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
import urllib.parse

import cv2
import pyarrow.parquet as pq


DEFAULT_DATASET_PATH = Path(
    '/mnt/poke_real_dataset/robot_ur7e_lerobotv21_0808_1280'
)
API_PROVIDERS = {
    'ark': {
        'endpoint': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
        'model': 'doubao-seed-2-0-lite-260215',
        'requires_api_key': True,
        'response_format': 'json_schema',
        'backend_name': 'volcengine_ark',
    },
    'dashscope': {
        'endpoint': (
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
        ),
        'model': 'qwen3-vl-plus',
        'requires_api_key': True,
        'response_format': 'json_object',
        'backend_name': 'dashscope',
    },
    'openai-compatible': {
        'endpoint': 'http://127.0.0.1:8000/v1/chat/completions',
        'model': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'requires_api_key': False,
        'response_format': 'json_schema',
        'backend_name': 'openai_compatible',
    },
}
API_PROVIDER_ALIASES = {
    'volcengine_ark': 'ark',
    'openai_compatible': 'openai-compatible',
}
DASHSCOPE_LITE_MODEL = 'qwen3-vl-plus'
DASHSCOPE_PRO_MODEL = 'qwen3.8-max'


def normalize_api_provider(value: str) -> str:
    normalized = value.strip().lower()
    return API_PROVIDER_ALIASES.get(normalized, normalized)


API_PROVIDER = normalize_api_provider(
    os.environ.get('API_PROVIDER') or os.environ.get('ROBOT_BACKEND', 'ark')
)
if API_PROVIDER not in API_PROVIDERS:
    API_PROVIDER = 'ark'
_PROVIDER_DEFAULTS = API_PROVIDERS[API_PROVIDER]
IMAGE_W, IMAGE_H = 1280, 720
MODEL = (
    os.environ.get('API_MODEL') or os.environ.get('ROBOT_SERVED_MODEL') or
    _PROVIDER_DEFAULTS['model']
)
REVIEW_MODEL = os.environ.get('ROBOT_REVIEW_MODEL', MODEL)
CPA_POINT_MODEL = os.environ.get('ROBOT_CPA_POINT_MODEL', MODEL)
REVIEW_REASONING_EFFORT = os.environ.get('ROBOT_REVIEW_REASONING_EFFORT', 'medium')
BACKEND = _PROVIDER_DEFAULTS['backend_name']
API_KEY_ENV = (
    os.environ.get('API_KEY_ENV') or os.environ.get('ROBOT_API_KEY_ENV') or
    ('ARK_API_KEY' if API_PROVIDER == 'ark' else 'API_KEY')
).strip()
GROUNDING_TYPES = ('object_location', 'color_object_location', 'neighbor_name', 'direction_relation', 'presence')
GRIPPER_CLOSED_MAX = 0.2
GRIPPER_OPEN_MIN = 0.8
STA_CONTEXT_SECONDS = 2.0
STA_ANTICIPATION_SECONDS = 1.0
STA_ANTICIPATION_WINDOW_FRACTIONS = (0.25, 0.5, 0.75)
STA_VIDEO_FPS = 4.0
STA_VIDEO_WIDTH = 640
VLM_COORDINATE_MAX = 1000
ECOT_SUBTASK_FIELDS = ('subtask', 'action', 'object', 'source', 'target')
ANNOTATION_MODES = ('sta', 'cpa')
ECOT_PRIMARY_SAMPLE_TYPE = 'transition_start'
ECOT_MIDPOINT_SAMPLE_TYPE = 'between_transitions_midpoint'
CPA_REGION_SYSTEM_PROMPT = '''You localize the interaction region at an intentional contact frame. The image contains either a robot gripper or a human/dexterous hand contacting a manipulable object. Return one tight box that includes the participating end-effector fingers or jaws and the contacted object surface around their physical interaction. Exclude unrelated object extent, support surfaces, and background. Coordinates use an integer 0-to-1000 grid with upper-left origin, x right, and y down. Return only the requested JSON.'''
CPA_POINTS_SYSTEM_PROMPT = '''You annotate paired contact-agent/object target points inside a cropped interaction region at the exact contact frame. H_i is the contact-agent-side point: depending on the supplied Pro-reviewed event role, the contact agent is specifically the bare human/dexterous hand, a held tool, or the robot end effector. O_i is the contacted-object-side point. For a held-tool event, H_i must lie on the held tool's contacting material and never on the holder's hand or body. For every distinct visible physical contact interface i, place H_i and O_i immediately across that same interface on opposite participants; never place either point in the gap, at a generic object center, or twice on the same participant. Use one pair for a single interface and multiple pairs only for genuinely distinct visible interfaces. Coordinates use an integer 0-to-1000 grid relative to this crop, with upper-left origin, x right, and y down. Return only the requested JSON.'''
CPA_CONTACT_AGENT_ROLE_VERSION = 'pro_contact_event_agent_role_v1'
CPA_POINT_PAIR_VERSION = 'contact_agent_object_pairs_v2'
CPA_SAM2_POINT_PAIR_VERSION = 'sam2_contact_agent_object_pair_v2'
CPA_POINT_METHODS = ('seed_vlm', 'sam2')
CPA_POINT_METHOD = os.environ.get('ROBOT_CPA_POINT_METHOD', 'seed_vlm').strip().lower()
CPA_SAM2_PROMPT_SYSTEM_PROMPT = '''You prepare geometric SAM2 prompts for one cropped contact image. Identify the explicitly supplied Pro-reviewed contact agent as HAND for schema compatibility: this is specifically either the bare human/dexterous hand, a held tool, or the robot end effector. Identify the discrete contacted object as OBJECT. For a held-tool event, HAND means only the held tool and must exclude the holder's hand and body. Describe each participant concisely. Return one tight integer 0-to-1000 bbox and one positive seed point clearly inside visible material for each participant. The HAND seed must not lie on OBJECT, the OBJECT seed must not lie on HAND, and neither seed may lie in background or the contact gap. Boxes may touch but must not be identical. Do not produce final contact points; SAM2 masks and a deterministic closest-mask calculation will do that. Return only the requested JSON.'''
CPA_SAM2_BACKEND = os.environ.get('ROBOT_SAM2_BACKEND', 'transformers').strip().lower()
CPA_SAM2_MODEL_PATH = Path(os.environ.get(
    'ROBOT_SAM2_MODEL_PATH', '/mnt/SAM2/sam2-hiera-large',
))
CPA_SAM2_ROOT = Path(os.environ.get('ROBOT_SAM2_ROOT', '/mnt/SAM2'))
CPA_SAM2_CHECKPOINT = Path(os.environ.get(
    'ROBOT_SAM2_CHECKPOINT',
    '/mnt/SAM2/sam2-hiera-large/sam2_hiera_large.pt',
))
CPA_SAM2_MODEL_CONFIG = os.environ.get(
    'ROBOT_SAM2_MODEL_CONFIG', '/mnt/SAM2/sam2-hiera-large/sam2_hiera_l.yaml',
)
CPA_SAM2_PYTHON = os.environ.get(
    'ROBOT_SAM2_PYTHON', '/mnt/outputs/envs/cpa_sam2/bin/python',
)
CPA_SAM2_DEVICE = os.environ.get('ROBOT_SAM2_DEVICE', 'auto')
CPA_OBSERVATION_SAM2_BATCH_SIZE = int(os.environ.get(
    'ROBOT_CPA_SAM2_BATCH_SIZE', '8',
))
CPA_TRACKING_REVIEW_VERSION = 'lite_contact_agent_pair_semantic_review_v2'
CPA_PAIR_SEPARATION_REVIEW_VERSION = 'contact_pair_distance_growth_gt_100pct_v2'
CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO = 1.0
CPA_MULTIPLE_CHOICE_VQA_VERSION = (
    'observation_local_sam2_robot_object_only_cardinality_hint_multiselect_v7'
)
CPA_MULTIPLE_CHOICE_ENABLED = os.environ.get(
    'ROBOT_CPA_MULTIPLE_CHOICE_ENABLED', '1',
).strip().lower() not in {'0', 'false', 'no', 'off'}
CPA_NEGATIVE_A_AT_720P = float(os.environ.get('ROBOT_CPA_NEGATIVE_A', '24'))
CPA_NEGATIVE_DROP_RATE = float(os.environ.get('ROBOT_CPA_NEGATIVE_DROP_RATE', '0.2'))
CPA_AGENT_PARTICIPANT_QUESTION_TYPES = {
    'human_hand': ('hand', 'next_contact_hand_point'),
    'robot_gripper': ('gripper', 'next_contact_gripper_point'),
    'held_tool': ('held tool', 'next_contact_tool_point'),
}
CPA_TRACKING_REVIEW_SYSTEM_PROMPT = '''You audit tracked paired contact points in one current observation. MEDIA A is the raw observation. MEDIA B is the same image with labels H_i and O_i. The request explicitly states the Pro-reviewed contact-agent role. For every supplied pair index, independently decide whether H_i lies on that exact visible contact agent (bare hand, held tool, or robot end effector as specified) and whether O_i lies on the specified contacted object's visible pixels. For a held-tool event, an H_i point on the holder's hand or body is false. Judge the labeled pixel, not the intended nearby region. A point on background, another object, empty space, or the wrong participant is false. Occluded or visually indefensible points are false. Do not move or add points. Return one check for every supplied pair index and only the requested JSON.'''
CPA_TRACKER_CHECKPOINT = Path(os.environ.get(
    'ROBOT_COTRACKER_CHECKPOINT', '/mnt/co-tracker/checkpoints/scaled_offline.pth',
))
CPA_TRACKER_PYTHON = os.environ.get('ROBOT_COTRACKER_PYTHON', '/usr/bin/python3')
ECOT_MISSING_SEMANTIC = 'None'
ECOT_PRIVILEGED_LEAK_PATTERN = re.compile(
    r'\b(?:video|future frames?|past frames?|timestamps?|progress percentage|'
    r'episode progress|later in the episode|earlier in the episode)\b',
    flags=re.I,
)
ECOT_SYSTEM_PROMPT = '''You generate concise embodied chain-of-thought (ECoT) annotations for robot manipulation.
Use exactly three parts: Scene Description, Task Progress Assessment, and Subtask. Each prose part must be exactly one short sentence: use at most 35 words for Scene Description, at most 30 words for Task Progress Assessment, and at most 24 words for Subtask. Scene Description must summarize only currently observable task-relevant objects, spatial relations, arm position, and gripper state. Task Progress Assessment must concisely evaluate completed and remaining subgoals and end verbatim with either "Task complete." or "Task not yet complete."; join that judgment with a semicolon, never make it a separate sentence.
Avoid decimal numbers and abbreviations containing periods so each field has only its final sentence-ending period.

Subtask is a goal-level task phase, not the next low-level robot action or next contact event. It may span several arm motions or contacts, but must describe one coherent task goal and must not predict a contact timestamp, bounding box, or standalone next-contact verb/noun label. Return Subtask as a lowercase concise sentence plus lowercase semantic fields: action, object, source, and target. Action is the main commanded lemma; object is the directly acted-on entity; source is an explicitly stated origin; target is an explicitly stated destination or final relation. Use the exact string "None" for a missing semantic field and never infer an unstated source or target. If the overall task is complete, use "no further subtask remains." and set all four semantic fields to "None".

Privileged trajectory context may be supplied only to improve annotation accuracy. Never mention that context, future frames, timestamps, progress percentages, or unseen outcomes in the ECoT text. Express the final three parts using only evidence supportable by the current observation and task instruction. Return only the requested JSON.'''
ECOT_PROGRESS_SYSTEM_PROMPT_LEGACY = '''You summarize where one target frame lies in a complete robot-manipulation episode for annotation-time context. Watch the complete episode video, match the supplied target image to the trajectory, and return one concise sentence describing the target frame's stage, what has happened, and what remains. Avoid decimal numbers and abbreviations containing periods so the field has only its final sentence-ending period. This summary is privileged annotation context, not a training target. Return only the requested JSON.'''
ECOT_PROGRESS_SYSTEM_PROMPT = '''You summarize where one target frame lies in a complete robot-manipulation episode for annotation-time context. Watch the complete episode video, match the supplied target image to the trajectory, and return exactly one concise sentence of at most 48 words describing the target frame's stage, what has happened, and what remains. Avoid decimal numbers and abbreviations containing periods so the field has only its final sentence-ending period. This summary is privileged annotation context, not a training target. Return only the requested JSON.'''
ECOT_PROGRESS_SYSTEM_PROMPT_VERSIONS = {
    ECOT_PROGRESS_SYSTEM_PROMPT_LEGACY,
    ECOT_PROGRESS_SYSTEM_PROMPT,
}
ECOT_SUBTASK_REWRITE_SYSTEM_PROMPT = '''You rewrite only the Subtask part of an existing embodied chain-of-thought annotation. Preserve the supplied Scene Description and Task Progress Assessment conceptually, but return only a task-level Subtask JSON object. The Subtask is one coherent goal-level phase, not the next low-level robot action or next contact event. It may span multiple arm motions or contacts and must not predict a contact timestamp, bounding box, or standalone next-contact verb/noun label. Return a lowercase concise subtask sentence plus lowercase action, object, source, and target fields. Use the exact string "None" for an unstated semantic role. If the supplied progress says Task complete, return "no further subtask remains." and set action, object, source, and target to "None". Return only the requested JSON.'''
FORBIDDEN_EXACT_NAMES = {
    'table', 'desk', 'floor', 'wall', 'ceiling', 'background', 'workspace',
    'counter', 'countertop', 'shelf', 'cabinet', 'robot', 'gripper', 'claw',
    'robot hand', 'robot arm', 'robotic gripper', 'person', 'human', 'human hand',
}
FORBIDDEN_NAME_PARTS = (
    'table surface', 'counter surface', 'work surface', 'robot gripper',
    'robotic hand', 'robotic arm', 'human hand', 'background area',
    'tabletop', 'desktop', 'floor area',
)
SAFE_SYNONYM_GROUPS = (
    {'cup', 'mug'}, {'plate', 'dish'}, {'stuffed animal', 'plush toy'},
    {'shoe', 'slipper'}, {'spoon', 'ladle'}, {'cabbage', 'lettuce'},
    {'tape', 'adhesive tape'}, {'box', 'carton'}, {'bag', 'tote bag'},
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def nonempty_path(value: str) -> Path:
    """Parse a CLI path without silently treating an empty value as cwd."""
    if not value.strip():
        raise argparse.ArgumentTypeError('path must not be empty')
    return Path(value)


def validate_cpa_negative_a(value: float) -> float:
    """Validate the 720p-reference exclusion radius used by f(r)=1/(r-a)."""
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError('cpa_negative_a_must_be_finite_and_nonnegative')
    return normalized


def validate_cpa_negative_drop_rate(value: float) -> float:
    """Validate the probability of sampling a negative outside the mask."""
    normalized = float(value)
    if not math.isfinite(normalized) or not 0 <= normalized <= 1:
        raise ValueError('cpa_negative_drop_rate_must_be_between_zero_and_one')
    return normalized


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def archive_legacy_artifact(path: Path, legacy_name: str) -> Path | None:
    """Move an obsolete combined artifact aside without overwriting a backup."""
    if not path.is_file():
        return None
    destination = path.with_name(legacy_name)
    index = 2
    while destination.exists():
        destination = path.with_name(
            f'{Path(legacy_name).stem}_{index}{Path(legacy_name).suffix}',
        )
        index += 1
    os.replace(path, destination)
    return destination


def read_annotation_records(root: Path) -> list[dict]:
    """Read separated final records, with legacy combined output as fallback."""
    separated = (root/'grounding_records.jsonl', root/'ecot_records.jsonl')
    paths = [path for path in separated if path.is_file()]
    if not paths:
        legacy = root/'records.jsonl'
        paths = [legacy] if legacy.is_file() else []
    records = []
    for path in paths:
        records.extend(
            json.loads(line)
            for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        )
    return records


def write_annotation_records(
    root: Path,
    records: list[dict],
    kinds: set[str] | None = None,
) -> None:
    """Write selected grounding/ECoT artifacts without clobbering other modes."""
    unknown = sorted({record.get('kind') for record in records} - {'grounding', 'ecot'})
    if unknown:
        raise ValueError(f'unknown_annotation_record_kinds:{unknown}')
    selected_kinds = {'grounding', 'ecot'} if kinds is None else set(kinds)
    invalid_kinds = sorted(selected_kinds - {'grounding', 'ecot'})
    if invalid_kinds:
        raise ValueError(f'unknown_annotation_output_kinds:{invalid_kinds}')
    for kind in sorted(selected_kinds):
        atomic_jsonl(
            root/f'{kind}_records.jsonl',
            sorted((record for record in records if record['kind'] == kind), key=lambda x: x['id']),
        )
    archive_legacy_artifact(root/'records.jsonl', 'records_combined_legacy.jsonl')


def normalize_ecot_vocabulary(record: dict) -> dict:
    """Upgrade legacy CoT metadata names without changing ECoT content."""
    if record.get('kind') == 'cot':
        record['kind'] = 'ecot'
    if 'cot_methods' in record:
        if 'ecot_methods' in record:
            raise ValueError(f'duplicate_ecot_method_containers:{record.get("id")}')
        record['ecot_methods'] = record.pop('cot_methods')
    return record


def video_codec(path: Path) -> str:
    """Return the primary video codec name reported by ffprobe."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name', '-of', 'default=nw=1:nk=1',
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f'video_codec_probe_failed:{path}:{result.stderr[-2000:]}')
    codec = result.stdout.strip()
    if not codec:
        raise RuntimeError(f'video_codec_missing:{path}')
    return codec


def complete_jpeg(path: Path) -> bool:
    """Return whether an existing JPEG has intact start/end markers."""
    try:
        if not path.is_file() or path.stat().st_size < 4:
            return False
        with path.open('rb') as handle:
            if handle.read(2) != b'\xff\xd8':
                return False
            handle.seek(-2, os.SEEK_END)
            return handle.read(2) == b'\xff\xd9'
    except OSError:
        return False


def browser_h264_ready(path: Path) -> bool:
    """Cheaply validate fast-start H.264 MP4 files produced by this pipeline."""
    try:
        if not path.is_file() or path.stat().st_size < 16:
            return False
        # ffmpeg's fast-start outputs declare ``avc1``/``avc3`` in the ftyp
        # compatibility list (offset 24 for pipeline-produced files), so a
        # fixed tiny read avoids spawning ffprobe or scanning video payloads.
        with path.open('rb') as handle:
            header = handle.read(64)
        return b'ftyp' in header and (b'avc1' in header or b'avc3' in header)
    except OSError:
        return False


def encode_browser_h264(source: Path, destination: Path) -> None:
    """Encode a video as browser-compatible H.264 and replace atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f'.{destination.stem}.{os.getpid()}.h264.mp4')
    tmp.unlink(missing_ok=True)
    result = subprocess.run(
        [
            'ffmpeg', '-y', '-loglevel', 'error', '-i', str(source), '-an',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f'browser_h264_encode_failed:{source}:{result.stderr[-2000:]}')
    os.replace(tmp, destination)


def prepare_episode_preview(source: Path, destination: Path) -> None:
    """Create one browser-ready full-episode preview without duplicating it per candidate."""
    if browser_h264_ready(destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f'.{destination.stem}.{os.getpid()}.remux.mp4')
    tmp.unlink(missing_ok=True)
    result = subprocess.run(
        [
            'ffmpeg', '-y', '-loglevel', 'error', '-i', str(source),
            '-map', '0:v:0', '-an', '-c:v', 'copy', '-movflags', '+faststart', str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
        os.replace(tmp, destination)
        if video_codec(destination) == 'h264':
            return
    tmp.unlink(missing_ok=True)
    encode_browser_h264(source, destination)


def task_text(task_dir: Path) -> str:
    return json.loads((task_dir / 'meta' / 'info.json').read_text())['tasks'][0]


def parse_episode_count(value: str) -> int | str:
    """Parse a positive episode count or the exact sentinel ``all``."""
    normalized = value.strip().lower()
    if normalized == 'all':
        return 'all'
    try:
        count = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            'episode count must be a positive integer or all'
        ) from error
    if count < 1:
        raise argparse.ArgumentTypeError(
            'episode count must be a positive integer or all'
        )
    return count


def episode_identity(task_dir: Path | str, episode_index: int) -> tuple[str, int]:
    """Return the canonical identity used to exclude already annotated episodes."""
    return str(Path(task_dir).expanduser().resolve()), int(episode_index)


def discover_eligible_episodes(
    dataset_path: Path,
    seed: int,
) -> list[tuple[Path, int, int]]:
    """Discover every annotatable episode in deterministic task-diverse order."""
    task_dirs = sorted(
        Path(entry.path) for entry in os.scandir(dataset_path) if entry.is_dir()
    )
    rng = random.Random(seed)
    rng.shuffle(task_dirs)
    episodes_by_task: list[list[tuple[Path, int, int]]] = []
    for task_dir in task_dirs:
        try:
            lines = (task_dir/'meta'/'episodes.jsonl').read_text(
                encoding='utf-8',
            ).splitlines()
        except (FileNotFoundError, OSError):
            continue
        task_episodes = []
        seen_indices = set()
        for line in lines:
            if not line.strip():
                continue
            episode = json.loads(line)
            episode_index = int(episode['episode_index'])
            episode_length = int(episode.get('length', 0))
            if episode_length < 20 or episode_index in seen_indices:
                continue
            seen_indices.add(episode_index)
            task_episodes.append((task_dir, episode_index, episode_length))
        if task_episodes:
            episodes_by_task.append(task_episodes)

    # Round-robin across shuffled tasks: numeric selections retain broad task
    # diversity, while ``all`` eventually covers every eligible manifest row.
    candidates = []
    maximum_task_episodes = max((len(items) for items in episodes_by_task), default=0)
    for episode_offset in range(maximum_task_episodes):
        for task_episodes in episodes_by_task:
            if episode_offset < len(task_episodes):
                candidates.append(task_episodes[episode_offset])
    return candidates


def build_sample_rows(
    candidates: list[tuple[Path, int, int]],
    seed: int,
    start_index: int = 0,
) -> list[dict]:
    """Build stable selection rows for discovered episode candidates."""
    rows = []
    task_instructions = {}
    for offset, (task_dir, episode_index, length) in enumerate(candidates):
        task_key = str(task_dir.resolve())
        if task_key not in task_instructions:
            task_instructions[task_key] = task_text(task_dir)
        frame_seed = hashlib.sha256(
            f'{seed}:{task_key}:{episode_index}'.encode('utf-8')
        ).digest()
        frame_rng = random.Random(int.from_bytes(frame_seed[:8], 'big'))
        frame_index = frame_rng.randrange(
            max(1, length // 8), max(2, length * 7 // 8),
        )
        row_index = start_index + offset
        rows.append({
            'id': f'robot_demo_{row_index:03d}',
            'kind': 'episode',
            'task_dir': task_key,
            'task_instruction': task_instructions[task_key],
            'episode_index': episode_index,
            'episode_length': length,
            'requested_frame_index': frame_index,
        })
    return rows


def select_samples(
    seed: int,
    episodes: int | str = 50,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    excluded: set[tuple[str, int]] | None = None,
    start_index: int = 0,
) -> list[dict]:
    """Select new unique episodes, or every eligible episode with ``all``."""
    candidates = [
        candidate for candidate in discover_eligible_episodes(dataset_path, seed)
        if episode_identity(candidate[0], candidate[1]) not in (excluded or set())
    ]
    if episodes != 'all':
        requested = int(episodes)
        if len(candidates) < requested:
            raise RuntimeError(
                f'insufficient_unannotated_episode_selection:'
                f'{len(candidates)}:{requested}'
            )
        candidates = candidates[:requested]
    return build_sample_rows(candidates, seed, start_index)


def load_samples(
    seed: int,
    selection: Path | None,
    episodes: int | None = 50,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[dict]:
    if selection is None:
        if episodes is None:
            raise RuntimeError('episode_count_required_without_selection')
        return select_samples(seed, episodes, dataset_path)
    rows = json.loads(selection.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError('frozen_selection_must_be_nonempty_array')
    if episodes is not None and len(rows) != episodes:
        raise RuntimeError(
            f'frozen_selection_episode_count_mismatch:{len(rows)}:{episodes}'
        )
    if len({
        episode_identity(r['task_dir'], int(r['episode_index'])) for r in rows
    }) != len(rows):
        raise RuntimeError('frozen_selection_has_duplicate_episodes')
    rng = random.Random(seed)
    task_instructions = {}
    for i, row in enumerate(rows):
        row['id'] = f'robot_demo_{i:03d}'
        row['kind'] = 'episode'
        task_dir = Path(row['task_dir'])
        task_key = str(task_dir.expanduser().resolve())
        if task_key not in task_instructions:
            task_instructions[task_key] = task_text(task_dir)
        row['task_dir'] = task_key
        row['task_instruction'] = task_instructions[task_key]
        row['episode_index'] = int(row['episode_index'])
        row['episode_length'] = int(row['episode_length'])
        row['requested_frame_index'] = int(row.get(
            'requested_frame_index',
            rng.randrange(
                max(1, row['episode_length'] // 8),
                max(2, row['episode_length'] * 7 // 8),
            ),
        ))
        row.pop('grounding_question_type', None)
        for key in ('image_path', 'source_video', 'selected_frame_index', 'gripper_segmentation', 'sta_candidates'):
            row.pop(key, None)
    return rows


def build_grounding_rows(rows: list[dict]) -> list[dict]:
    """Create one independent grounding sample for every source episode."""
    grounding_rows = []
    for index, source in enumerate(rows):
        row = copy.deepcopy(source)
        row.update({
            'id': f'{source["id"]}_grounding',
            'kind': 'grounding',
            'parent_episode_id': source['id'],
            'grounding_question_type': GROUNDING_TYPES[index % len(GROUNDING_TYPES)],
        })
        grounding_rows.append(row)
    return grounding_rows


def video_path(row: dict) -> Path:
    return Path(row['task_dir']) / 'videos' / 'chunk-000' / 'observation.images.cam_left_wrist' / f"episode_{int(row['episode_index']):06d}.mp4"


def action_path(row: dict) -> Path:
    return Path(row['task_dir']) / 'data' / 'chunk-000' / f"episode_{int(row['episode_index']):06d}.parquet"


def gripper_segments(row: dict, seed: int) -> dict:
    """Split an episode by hysteretic action.left_gripper open/closed changes."""
    table = pq.read_table(action_path(row), columns=['action.left_gripper'])
    values = table.column(0).to_pylist()
    flat = [float(x[0] if isinstance(x, (list, tuple)) else x) for x in values]
    if len(flat) < 2:
        raise RuntimeError(f'gripper_series_too_short:{row["id"]}')
    state = 'open' if flat[0] >= 0.5 else 'closed'
    initial_state = state
    changes = []
    for i, value in enumerate(flat[1:], 1):
        next_state = state
        if state == 'open' and value <= GRIPPER_CLOSED_MAX:
            next_state = 'closed'
        elif state == 'closed' and value >= GRIPPER_OPEN_MIN:
            next_state = 'open'
        if next_state != state:
            changes.append({'frame': i, 'from': state, 'to': next_state, 'value': round(value, 6)})
            state = next_state
    if not changes:
        raise RuntimeError(f'no_gripper_state_change:{row["id"]}')
    boundaries = [0] + [x['frame'] for x in changes] + [len(flat)]
    segments = []
    state = initial_state
    for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segments.append({
            'segment_index': i, 'start_frame': start, 'end_frame_exclusive': end,
            'gripper_state': state, 'ends_with_change': i < len(changes),
            'next_gripper_change': changes[i] if i < len(changes) else None,
        })
        if i < len(changes):
            state = changes[i]['to']
    candidates = [s for s in segments if s['ends_with_change'] and s['end_frame_exclusive'] - s['start_frame'] >= 15]
    if not candidates:
        raise RuntimeError(f'no_valid_gripper_segment:{row["id"]}')
    rng = random.Random(seed * 1009 + int(row['id'].rsplit('_', 1)[1]))
    chosen = dict(rng.choice(candidates))
    return {
        'source': 'action.left_gripper',
        'closed_threshold_max': GRIPPER_CLOSED_MAX,
        'open_threshold_min': GRIPPER_OPEN_MIN,
        'initial_state': initial_state,
        'transitions': changes,
        'segments': segments,
        'selected_segment': chosen,
    }


def ecot_midpoint_row(row: dict, seed: int) -> dict:
    """Clone one ECoT row at a frame strictly between two gripper transitions."""
    segmentation = row.get('gripper_segmentation')
    if not segmentation:
        raise RuntimeError(f'gripper_segmentation_required:{row["id"]}')
    candidates = [
        segment for segment in segmentation['segments']
        if (
            int(segment['segment_index']) > 0 and
            segment['ends_with_change'] is True and
            int(segment['end_frame_exclusive']) - int(segment['start_frame']) >= 3
        )
    ]
    if not candidates:
        raise RuntimeError(f'no_frame_strictly_between_transitions:{row["id"]}')
    parent_episode_id = row.get('parent_episode_id', row['id'])
    episode_number = int(parent_episode_id.rsplit('_', 1)[1])
    rng = random.Random(seed * 2027 + episode_number)
    selected_segment = copy.deepcopy(rng.choice(candidates))
    start = int(selected_segment['start_frame'])
    end = int(selected_segment['end_frame_exclusive'])
    midpoint = start + (end - start) // 2
    if not start < midpoint < end:
        raise RuntimeError(f'invalid_transition_midpoint:{row["id"]}:{start}:{midpoint}:{end}')
    extra = copy.deepcopy(row)
    extra.update({
        'id': f'{parent_episode_id}_ecot_mid',
        'kind': 'ecot',
        'ecot_sample_type': ECOT_MIDPOINT_SAMPLE_TYPE,
        'parent_episode_id': parent_episode_id,
        'requested_frame_index': midpoint,
        'episode_preview': f'{parent_episode_id}.mp4',
    })
    extra['gripper_segmentation']['selected_segment'] = selected_segment
    for key in ('image_path', 'selected_frame_index', 'sta_candidates'):
        extra.pop(key, None)
    return extra


def build_ecot_rows(rows: list[dict], seed: int) -> list[dict]:
    """Create transition-start and midpoint ECoT samples for every episode."""
    primary = []
    for source in rows:
        row = copy.deepcopy(source)
        row.update({
            'id': f'{source["id"]}_ecot',
            'kind': 'ecot',
            'parent_episode_id': source['id'],
            'ecot_sample_type': ECOT_PRIMARY_SAMPLE_TYPE,
            'episode_preview': f'{source["id"]}.mp4',
        })
        primary.append(row)
    midpoint = [ecot_midpoint_row(row, seed) for row in primary]
    return primary + midpoint


def extract_media(
    row: dict,
    image_dir: Path,
    seed: int,
    resume_existing: bool = False,
) -> Path:
    if row['kind'] == 'ecot' and 'gripper_segmentation' not in row:
        row['gripper_segmentation'] = gripper_segments(row, seed)
    if row['kind'] == 'ecot':
        if row.get('ecot_sample_type') == ECOT_MIDPOINT_SAMPLE_TYPE:
            index = int(row['requested_frame_index'])
        else:
            row.setdefault('ecot_sample_type', ECOT_PRIMARY_SAMPLE_TYPE)
            index = int(row['gripper_segmentation']['selected_segment']['start_frame'])
    else:
        index = int(row['requested_frame_index'])
    out = image_dir / f'{row["id"]}.jpg'
    row['selected_frame_index'] = index
    row['image_path'] = str(out)
    row['source_video'] = str(video_path(row))
    if resume_existing and complete_jpeg(out):
        return out
    cap = cv2.VideoCapture(str(video_path(row)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'frame_decode_failed:{row["id"]}:{index}')
    if not cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f'frame_write_failed:{out}')
    return out


def sta_observation_frames(
    candidate_frame: int,
    previous_transition_frame: int,
    primary_query_frame: int,
) -> list[int]:
    """Sample observations no earlier than the bounded primary STA query."""
    segment_start = previous_transition_frame + 1
    window_start = max(segment_start, primary_query_frame)
    window_last = candidate_frame - 1
    if window_start > window_last:
        raise RuntimeError(f'empty_precontact_window:{window_start}:{candidate_frame}')
    if not segment_start <= primary_query_frame <= window_last:
        raise RuntimeError(
            f'primary_query_outside_segment:{primary_query_frame}:{segment_start}:{candidate_frame}'
        )
    span = window_last - window_start
    frames = {primary_query_frame}
    frames.update(
        window_start + round(span * fraction)
        for fraction in STA_ANTICIPATION_WINDOW_FRACTIONS
    )
    return sorted(frames)


def extract_sta_candidate_media(
    row: dict,
    transition_index: int,
    sta_image_dir: Path,
    contact_clip_dir: Path,
    context_seconds: float,
    anticipation_seconds: float,
    require_object_bbox: bool = True,
    resume_existing: bool = False,
) -> dict:
    """Create a teacher clip around one gripper transition and a learner query image."""
    transition = row['gripper_segmentation']['transitions'][transition_index]
    source = video_path(row)
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or IMAGE_W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or IMAGE_H)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f'video_probe_failed:{row["id"]}:{fps}:{frame_count}')
    candidate_frame = int(transition['frame'])
    if not 0 <= candidate_frame < frame_count:
        cap.release()
        raise RuntimeError(f'transition_outside_video:{row["id"]}:{candidate_frame}:{frame_count}')

    half_context = max(1, round(fps * context_seconds / 2.0))
    context_start = max(0, candidate_frame - half_context)
    context_end = min(frame_count - 1, candidate_frame + half_context)
    previous_transition = (
        int(row['gripper_segmentation']['transitions'][transition_index - 1]['frame'])
        if transition_index else -1
    )
    # Starting after the previous gripper-transition candidate makes this event
    # the next candidate relative to the learner observation.
    query_frame = max(
        previous_transition + 1,
        candidate_frame - max(1, round(fps * anticipation_seconds)),
    )
    if query_frame >= candidate_frame:
        cap.release()
        raise RuntimeError(f'no_precontact_query_frame:{row["id"]}:{candidate_frame}')

    candidate_id = f'{row["id"]}_contact_{transition_index:02d}'
    query_path = sta_image_dir / f'{candidate_id}_query.jpg'
    if not (resume_existing and complete_jpeg(query_path)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, query_frame)
        ok, query_image = cap.read()
        if not ok or query_image is None or not cv2.imwrite(
            str(query_path), query_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        ):
            cap.release()
            raise RuntimeError(f'sta_query_frame_failed:{candidate_id}:{query_frame}')

    observation_frames = sta_observation_frames(
        candidate_frame, previous_transition, query_frame,
    )
    observations = []
    segment_start = previous_transition + 1
    observation_window_start = query_frame
    for observation_index, observation_frame in enumerate(observation_frames):
        is_primary = observation_frame == query_frame
        observation_path = (
            query_path if is_primary else
            sta_image_dir/f'{candidate_id}_obs_{observation_index:02d}.jpg'
        )
        if not is_primary and not (
            resume_existing and complete_jpeg(observation_path)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, observation_frame)
            ok, observation_image = cap.read()
            if not ok or observation_image is None or not cv2.imwrite(
                str(observation_path), observation_image,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            ):
                cap.release()
                raise RuntimeError(
                    f'sta_observation_frame_failed:{candidate_id}:{observation_frame}'
                )
        observations.append({
            'observation_id': f'{candidate_id}_obs_{observation_index:02d}',
            'query_frame_index': observation_frame,
            'query_image_path': str(observation_path),
            'query_image': observation_path.name,
            'time_to_contact_seconds': round(
                (candidate_frame - observation_frame) / fps, 6,
            ),
            'segment_progress': round(
                (observation_frame - segment_start) /
                max(1, candidate_frame - segment_start),
                6,
            ),
            'anticipation_window_progress': round(
                (observation_frame - observation_window_start) /
                max(1, candidate_frame - observation_window_start),
                6,
            ),
            'is_primary_teacher_observation': is_primary,
        })

    target_samples = max(3, round((context_end - context_start) / fps * STA_VIDEO_FPS) + 1)
    sampled = sorted({
        round(context_start + i * (context_end - context_start) / max(1, target_samples - 1))
        for i in range(target_samples)
    } | {candidate_frame})
    output_width = min(STA_VIDEO_WIDTH, width)
    output_height = max(2, round(height * output_width / width))
    output_height -= output_height % 2
    clip_path = contact_clip_dir / f'{candidate_id}_context.mp4'
    raw_clip_path = contact_clip_dir / f'.{candidate_id}.{os.getpid()}.mpeg4.mp4'
    if resume_existing and browser_h264_ready(clip_path):
        cap.release()
        written = sampled
    else:
        raw_clip_path.unlink(missing_ok=True)
        writer = cv2.VideoWriter(
            str(raw_clip_path), cv2.VideoWriter_fourcc(*'mp4v'), STA_VIDEO_FPS,
            (output_width, output_height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f'contact_clip_writer_failed:{raw_clip_path}')
        sample_set = set(sampled)
        written = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, context_start)
        for frame_index in range(context_start, context_end + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_index not in sample_set:
                continue
            frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
            is_candidate = frame_index == candidate_frame
            color = (20, 220, 20) if is_candidate else (230, 230, 230)
            label = f'frame {frame_index}' + ('  CANDIDATE' if is_candidate else '')
            cv2.rectangle(frame, (0, 0), (min(output_width - 1, 310), 34), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            if is_candidate:
                cv2.rectangle(frame, (2, 2), (output_width - 3, output_height - 3), color, 4)
            writer.write(frame)
            written.append(frame_index)
        cap.release()
        writer.release()
        if written != sampled:
            raw_clip_path.unlink(missing_ok=True)
            raise RuntimeError(f'contact_clip_incomplete:{candidate_id}:{written}:{sampled}')
        try:
            encode_browser_h264(raw_clip_path, clip_path)
        finally:
            raw_clip_path.unlink(missing_ok=True)
        probe = cv2.VideoCapture(str(clip_path))
        decoded = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        probe.release()
        if decoded != len(written):
            raise RuntimeError(f'contact_clip_probe_failed:{candidate_id}:{decoded}:{len(written)}')
    return {
        'candidate_id': candidate_id,
        'row_id': row['id'],
        'transition_index': transition_index,
        'candidate_frame': candidate_frame,
        'transition': dict(transition),
        'query_frame_index': query_frame,
        'time_to_contact_seconds': round((candidate_frame - query_frame) / fps, 6),
        'source_video_fps': fps,
        'source_video_frame_count': frame_count,
        'context_seconds_requested': context_seconds,
        'context_start_frame': context_start,
        'context_end_frame': context_end,
        'sampled_frame_indices': written,
        'query_image_path': str(query_path),
        'query_image': query_path.name,
        'segment_start_frame': segment_start,
        'segment_end_frame_exclusive': candidate_frame,
        'observation_window_start_frame': observation_window_start,
        'observation_window_end_frame_exclusive': candidate_frame,
        'anticipation_seconds_requested': anticipation_seconds,
        'observations': observations,
        'teacher_clip_path': str(clip_path),
        'teacher_clip': clip_path.name,
        'teacher_clip_fps': STA_VIDEO_FPS,
        'episode_video': f'{row["id"]}.mp4',
        # This legacy field means that the contact teacher must localize the
        # primary-observation object for an STA bbox row. Complete and STA-only
        # modes require it; CPA-only mode deliberately does not.
        'grounding_episode': require_object_bbox,
    }


def expand_sta_audit_observations(row: dict, audit: dict, sta_image_dir: Path) -> None:
    """Attach bounded anticipation-window observations to a contact audit."""
    transition_index = int(audit['transition_index'])
    transitions = row['gripper_segmentation']['transitions']
    previous_transition = (
        int(transitions[transition_index - 1]['frame'])
        if transition_index else -1
    )
    candidate_frame = int(audit['candidate_frame'])
    primary_query_frame = int(audit['query_frame_index'])
    frames = sta_observation_frames(
        candidate_frame, previous_transition, primary_query_frame,
    )
    source = Path(row.get('source_video') or video_path(row))
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'sta_observation_video_open_failed:{source}')
    fps = float(audit['source_video_fps'])
    segment_start = previous_transition + 1
    observation_window_start = max(segment_start, primary_query_frame)
    observations = []
    for observation_index, frame_index in enumerate(frames):
        is_primary = frame_index == primary_query_frame
        image_path = (
            Path(audit['query_image_path']) if is_primary else
            sta_image_dir/f'{audit["candidate_id"]}_obs_{observation_index:02d}.jpg'
        )
        if not is_primary:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok or image is None or not cv2.imwrite(
                str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92],
            ):
                cap.release()
                raise RuntimeError(
                    f'sta_observation_frame_failed:{audit["candidate_id"]}:{frame_index}'
                )
        observations.append({
            'observation_id': f'{audit["candidate_id"]}_obs_{observation_index:02d}',
            'query_frame_index': frame_index,
            'query_image_path': str(image_path),
            'query_image': image_path.name,
            'time_to_contact_seconds': round(
                (candidate_frame - frame_index) / fps, 6,
            ),
            'segment_progress': round(
                (frame_index - segment_start) /
                max(1, candidate_frame - segment_start),
                6,
            ),
            'anticipation_window_progress': round(
                (frame_index - observation_window_start) /
                max(1, candidate_frame - observation_window_start),
                6,
            ),
            'is_primary_teacher_observation': is_primary,
        })
    cap.release()
    audit['segment_start_frame'] = segment_start
    audit['segment_end_frame_exclusive'] = candidate_frame
    audit['observation_window_start_frame'] = observation_window_start
    audit['observation_window_end_frame_exclusive'] = candidate_frame
    audit.setdefault(
        'anticipation_seconds_requested',
        round((candidate_frame - primary_query_frame) / fps, 6),
    )
    audit['observations'] = observations


def configure_api(
    provider: str,
    endpoint: str | None,
    model: str | None,
    api_key_env: str,
) -> str:
    """Resolve provider defaults and configure shared request globals."""
    global API_PROVIDER, BACKEND, MODEL, REVIEW_MODEL, CPA_POINT_MODEL, API_KEY_ENV
    resolved_provider = normalize_api_provider(provider)
    if resolved_provider not in API_PROVIDERS:
        raise ValueError(f'unsupported_api_provider:{resolved_provider}')
    defaults = API_PROVIDERS[resolved_provider]
    resolved_endpoint = str(endpoint or defaults['endpoint']).strip()
    resolved_model = str(model or defaults['model']).strip()
    resolved_key_env = str(api_key_env or 'API_KEY').strip()
    if not resolved_endpoint:
        raise ValueError('api_endpoint_must_not_be_empty')
    if not resolved_model:
        raise ValueError('api_model_must_not_be_empty')
    if not resolved_key_env:
        raise ValueError('api_key_environment_name_must_not_be_empty')
    previous_model = MODEL
    API_PROVIDER = resolved_provider
    BACKEND = defaults['backend_name']
    MODEL = resolved_model
    default_review_model = (
        DASHSCOPE_PRO_MODEL
        if resolved_provider == 'dashscope' and resolved_model == DASHSCOPE_LITE_MODEL
        else resolved_model
    )
    if REVIEW_MODEL == previous_model and 'ROBOT_REVIEW_MODEL' not in os.environ:
        REVIEW_MODEL = default_review_model
    if CPA_POINT_MODEL == previous_model and 'ROBOT_CPA_POINT_MODEL' not in os.environ:
        CPA_POINT_MODEL = default_review_model
    API_KEY_ENV = resolved_key_env
    return resolved_endpoint


def append_json_schema_instruction(payload: dict, schema: dict) -> None:
    instruction = (
        'Return only one JSON object that validates against this JSON Schema: '
        + json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
    )
    for message in reversed(payload.get('messages') or []):
        if message.get('role') != 'user':
            continue
        content = message.get('content')
        if isinstance(content, list):
            content.append({'type': 'text', 'text': instruction})
        elif isinstance(content, str):
            message['content'] = content + '\n\n' + instruction
        else:
            message['content'] = instruction
        return
    payload.setdefault('messages', []).append({
        'role': 'user', 'content': instruction,
    })


def prepare_request(endpoint: str, payload: dict) -> tuple[dict, dict[str, str]]:
    """Apply provider authentication and response-format compatibility."""
    provider_name = normalize_api_provider(BACKEND)
    if provider_name not in API_PROVIDERS:
        raise RuntimeError(f'unsupported_vlm_backend:{BACKEND}')
    provider = API_PROVIDERS[provider_name]
    adapted = copy.deepcopy(payload)
    headers = {'Content-Type': 'application/json'}
    api_key = os.environ.get(API_KEY_ENV, '').strip()
    if provider['requires_api_key']:
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname:
            raise RuntimeError(f'{provider_name}_endpoint_must_be_https')
        if not api_key:
            raise RuntimeError(f'missing_api_key_environment_variable:{API_KEY_ENV}')
        headers['Authorization'] = f'Bearer {api_key}'
    elif api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    response_format = adapted.get('response_format')
    if (
        provider['response_format'] == 'json_object' and
        isinstance(response_format, dict) and
        response_format.get('type') == 'json_schema'
    ):
        schema = response_format.get('json_schema', {}).get('schema')
        if not isinstance(schema, dict):
            raise RuntimeError('json_schema_payload_missing_schema')
        append_json_schema_instruction(adapted, schema)
        adapted['response_format'] = {'type': 'json_object'}
        adapted.setdefault('enable_thinking', False)
    return adapted, headers


def post(endpoint: str, payload: dict, timeout: int = 900) -> dict:
    adapted, headers = prepare_request(endpoint, payload)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(adapted, ensure_ascii=False).encode('utf-8'),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        body = error.read().decode('utf-8', errors='replace')[-4000:]
        raise RuntimeError(f'vlm_http_error:{error.code}:{body}') from error


def image_url(path: Path) -> str:
    return 'data:image/jpeg;base64,' + base64.b64encode(path.read_bytes()).decode()


def video_url(path: Path) -> str:
    return 'data:video/mp4;base64,' + base64.b64encode(path.read_bytes()).decode()


def get_content(response: dict) -> str:
    msg = response['choices'][0]['message']
    return str(msg.get('content') or msg.get('reasoning_content') or '')


def grounding_schema() -> dict:
    return {'name': 'robot_grounding_inventory_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['suitable', 'reason', 'objects'],
        'properties': {
            'suitable': {'type': 'boolean'}, 'reason': {'type': 'string', 'maxLength': 240},
            'objects': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': {'type': 'object', 'additionalProperties': False,
                'required': ['name', 'alternate_name', 'color', 'bbox_xyxy', 'center_xy', 'clearly_visible'],
                'properties': {'name': {'type': 'string', 'maxLength': 80}, 'alternate_name': {'type': 'string', 'maxLength': 80},
                    'color': {'type': 'string', 'maxLength': 50}, 'bbox_xyxy': {'type': 'array', 'minItems': 4, 'maxItems': 4, 'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX}},
                    'center_xy': {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX}}, 'clearly_visible': {'type': 'boolean'}}}}
        }}}


def ecot_schema() -> dict:
    return {'name': 'robot_structured_ecot_subtask_v3', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['scene_description', 'task_progress_assessment', 'subtask'],
        'properties': {
            'scene_description': {'type': 'string', 'minLength': 1, 'maxLength': 320},
            'task_progress_assessment': {'type': 'string', 'minLength': 1, 'maxLength': 260},
            'subtask': {
                'type': 'object', 'additionalProperties': False,
                'required': list(ECOT_SUBTASK_FIELDS),
                'properties': {
                    'subtask': {'type': 'string', 'minLength': 1, 'maxLength': 240},
                    'action': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                    'object': {'type': 'string', 'minLength': 1, 'maxLength': 120},
                    'source': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                    'target': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                },
            },
        },
    }}


def ecot_subtask_schema() -> dict:
    return {'name': 'robot_ecot_subtask_semantics_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': list(ECOT_SUBTASK_FIELDS),
        'properties': {
            'subtask': {'type': 'string', 'minLength': 1, 'maxLength': 240},
            'action': {'type': 'string', 'minLength': 1, 'maxLength': 80},
            'object': {'type': 'string', 'minLength': 1, 'maxLength': 120},
            'source': {'type': 'string', 'minLength': 1, 'maxLength': 160},
            'target': {'type': 'string', 'minLength': 1, 'maxLength': 160},
        },
    }}


def ecot_progress_schema() -> dict:
    return {'name': 'robot_ecot_episode_progress_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['target_frame_progress_summary'],
        'properties': {
            'target_frame_progress_summary': {
                'type': 'string', 'minLength': 1, 'maxLength': 320,
            },
        },
    }}


def contact_schema() -> dict:
    return {'name': 'robot_sta_contact_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['valid_contact', 'reason', 'verb', 'noun', 'object_bbox_xyxy'],
        'properties': {
            'valid_contact': {'type': 'boolean'},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 400},
            'verb': {'type': ['string', 'null'], 'maxLength': 80},
            'noun': {'type': ['string', 'null'], 'maxLength': 80},
            'object_bbox_xyxy': {
                'type': ['array', 'null'], 'minItems': 4, 'maxItems': 4,
                'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
            },
        },
    }}


def cpa_region_schema() -> dict:
    return {'name': 'contact_point_anticipation_region_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['reason', 'interaction_bbox_xyxy'],
        'properties': {
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 300},
            'interaction_bbox_xyxy': {
                'type': 'array', 'minItems': 4, 'maxItems': 4,
                'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
            },
        },
    }}


def cpa_points_schema(actor: str = 'robot') -> dict:
    return {'name': 'contact_point_anticipation_pairs_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['reason', 'contact_pairs'],
        'properties': {
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 300},
            'contact_pairs': {
                'type': 'array', 'minItems': 1, 'maxItems': 5,
                'items': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['pair_index', 'hand_point_xy', 'object_point_xy'],
                    'properties': {
                        'pair_index': {'type': 'integer', 'minimum': 0, 'maximum': 4},
                        'hand_point_xy': {
                            'type': 'array', 'minItems': 2, 'maxItems': 2,
                            'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
                        },
                        'object_point_xy': {
                            'type': 'array', 'minItems': 2, 'maxItems': 2,
                            'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
                        },
                    },
                },
            },
        },
    }}


def cpa_sam2_prompt_schema() -> dict:
    coordinate = {
        'type': 'array', 'minItems': 2, 'maxItems': 2,
        'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
    }
    box = {
        'type': 'array', 'minItems': 4, 'maxItems': 4,
        'items': {'type': 'integer', 'minimum': 0, 'maximum': VLM_COORDINATE_MAX},
    }
    return {'name': 'cpa_sam2_hand_object_prompts_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': [
            'reason', 'hand_description', 'object_description',
            'hand_bbox_xyxy', 'object_bbox_xyxy',
            'hand_seed_point_xy', 'object_seed_point_xy',
        ],
        'properties': {
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 300},
            'hand_description': {'type': 'string', 'minLength': 1, 'maxLength': 120},
            'object_description': {'type': 'string', 'minLength': 1, 'maxLength': 120},
            'hand_bbox_xyxy': box,
            'object_bbox_xyxy': box,
            'hand_seed_point_xy': coordinate,
            'object_seed_point_xy': coordinate,
        },
    }}


def cpa_tracking_review_schema(pair_indices: list[int]) -> dict:
    return {'name': 'cpa_tracked_pair_semantic_review_v1', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['reason', 'checks'],
        'properties': {
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 400},
            'checks': {
                'type': 'array', 'minItems': len(pair_indices), 'maxItems': len(pair_indices),
                'items': {
                    'type': 'object', 'additionalProperties': False,
                    'required': ['pair_index', 'hand_on_end_effector', 'object_on_contacted_object'],
                    'properties': {
                        'pair_index': {'type': 'integer', 'enum': pair_indices},
                        'hand_on_end_effector': {'type': 'boolean'},
                        'object_on_contacted_object': {'type': 'boolean'},
                    },
                },
            },
        },
    }}


def grounding_prompt(task: str) -> str:
    return f'''You create conservative robot grounding pseudo-labels for one 1280x720 wrist-camera image. Global task: {task}
First decide whether the image is suitable: objects must be visible enough to identify and localize. If unsuitable, set suitable=false but still return any clearly visible objects.
Return AT MOST FOUR clearly visible, discrete manipulable objects, prioritizing objects needed for the global task and spatial relations. NEVER list support surfaces, scene regions, robot parts, or people: tables/desks/counters/floors/walls/background, gripper/robot hand/robot arm, and human/person are forbidden objects. For each valid object, return two natural English referring names. The alternate name must preserve object identity: never substitute a different species/category (for example pear is not melon); if there is no true synonym, use a descriptive phrase retaining the original head noun (for example green pear). Return dominant color and a [x1,y1,x2,y2] bounding box plus [cx,cy] center on an INTEGER 0-to-1000 coordinate grid. The coordinate convention is mandatory: the upper-left image corner is (0,0); the lower-right is (1000,1000); x increases right and y increases down. Use meaningful integer coordinates such as [120,250,430,690], never normalized fractions and never a degenerate box. The generator deterministically normalizes accepted coordinates to [0,1]. Do not invent objects. Return only the requested JSON.'''


def ecot_user_prompt(task: str, context: dict) -> str:
    return f'''Global task: {task}
The CURRENT observation is source frame {context["target_frame_index"]} of {context["source_video_frame_count"]} at {context["target_progress_ratio"]:.6f} episode progress.
Privileged whole-episode progress summary: {context["target_frame_progress_summary"]}
The labeled PAST and FUTURE images sample the episode at one-second intervals around CURRENT. Use them only as auxiliary annotation-time context to resolve task stage; do not mention them or any information unsupported by CURRENT in the final ECoT.
Generate the strict three-part ECoT now. Keep Scene Description, Task Progress Assessment, and Subtask to one concise sentence each. Subtask must describe the current goal-level task phase, not the next contact or one atomic motor action, and must include action/object/source/target semantics.'''


def contact_prompt(row: dict, candidate: dict) -> str:
    bbox_rule = (
        'Because this is a grounding episode, object_bbox_xyxy MUST localize the primary next-active '
        'object in MEDIA A on an integer 0-to-1000 [x1,y1,x2,y2] grid. The upper-left is (0,0), '
        'the lower-right is (1000,1000), and the box must be non-degenerate. Do not output normalized fractions.'
        if candidate['grounding_episode'] else
        'Because this is not a grounding episode, set object_bbox_xyxy to null.'
    )
    transition = candidate['transition']
    return f'''You annotate robot short-term object-interaction anticipation (STA). Global task: {row["task_instruction"]}
MEDIA A is the learner-facing observation before the candidate event. MEDIA B is teacher-only evidence covering {candidate["context_seconds_requested"]:.3f} seconds around a gripper transition. The frame with the green border and CANDIDATE label is source frame {candidate["candidate_frame"]}; the commanded gripper changes from {transition["from"]} to {transition["to"]} there.

Use the STA contact convention: a valid contact is the beginning of an intentional physical interaction with a primary manipulable object, when that object becomes active through direct gripper contact or tool-mediated contact. The transition is only a noisy candidate. Reject empty gripper open/close events, motion near an object without contact, accidental contact, and a state change that merely continues an already-active hold. Accept a tightly aligned transition when the short clip clearly establishes that intentional contact begins at the candidate neighborhood; keep the candidate source frame as the contact timestamp.

If valid_contact is true, return a concise lowercase action-category verb and the lowercase singular category noun of the primary contacted object. Use snake_case for a multiword category, no articles, colors, explanations, or destination objects. Robot parts, the gripper, people, support surfaces, and scene regions are never valid primary contact objects; if no eligible manipulable object begins contact, set valid_contact=false. If false, set verb, noun, and object_bbox_xyxy to null. {bbox_rule} Return only the requested JSON.'''


def cpa_contact_agent_semantics(audit: dict) -> dict:
    """Resolve the H-side participant from the final Pro contact-event review."""
    review = audit.get('contact_event_review') or {}
    event_type = str(
        audit.get('contact_event_type') or review.get('event_type') or ''
    ).strip() or None
    raw_pair = audit.get('pro_contact_pair') or review.get('contact_pair') or []
    contact_pair = [
        str(value).strip() if value is not None else None
        for value in raw_pair[:2]
    ] if isinstance(raw_pair, list) else []
    while len(contact_pair) < 2:
        contact_pair.append(None)
    explicit_role = audit.get('contact_agent_role')
    if explicit_role in {'held_tool', 'human_hand', 'robot_gripper'}:
        role = explicit_role
    elif audit.get('actor') == 'human' and event_type == 'tool_object_contact_onset':
        role = 'held_tool'
    elif audit.get('actor') == 'human':
        role = 'human_hand'
    else:
        role = 'robot_gripper'
    default_agent = {
        'held_tool': 'held tool',
        'human_hand': 'human or dexterous hand',
        'robot_gripper': 'parallel-jaw robot gripper',
    }[role]
    return {
        'role_version': CPA_CONTACT_AGENT_ROLE_VERSION,
        'role': role,
        'agent_name': str(audit.get('contact_agent_name') or contact_pair[0] or default_agent),
        'object_name': str(
            audit.get('contact_object_participant') or contact_pair[1] or
            (audit.get('contact') or {}).get('noun') or 'contacted object'
        ),
        'event_type': event_type,
        'contact_pair': contact_pair,
    }


def cpa_actor_description(audit: dict) -> str:
    agent = cpa_contact_agent_semantics(audit)
    if agent['role'] == 'held_tool':
        return f'HELD TOOL CONTACT AGENT ({agent["agent_name"]})'
    if agent['role'] == 'human_hand':
        return f'HUMAN OR DEXTEROUS HAND ({agent["agent_name"]})'
    return 'PARALLEL-JAW ROBOT GRIPPER'


def cpa_contact_agent_instruction(audit: dict) -> str:
    agent = cpa_contact_agent_semantics(audit)
    pair = json.dumps(agent['contact_pair'], ensure_ascii=False)
    if agent['role'] == 'held_tool':
        rule = (
            f'Pro classified this as tool_object_contact_onset with contact pair {pair}. '
            f'The H-side contact agent is the held tool "{agent["agent_name"]}". '
            'Every H_i must lie on that tool at its contacting tip or surface; never place H_i '
            'on the hand or body holding the tool. '
        )
    elif agent['role'] == 'human_hand':
        rule = (
            f'Pro classified this as direct hand-object contact with contact pair {pair}. '
            f'The H-side contact agent is the bare hand "{agent["agent_name"]}". '
            'Every H_i must lie on the visibly contacting hand or finger material. '
        )
    else:
        rule = (
            'The H-side contact agent is the parallel-jaw robot gripper. '
            'Every H_i must lie on the visibly contacting gripper jaw or end-effector material. '
        )
    return (
        rule +
        f'The O-side participant is "{agent["object_name"]}"; every O_i must lie on that contacted participant.'
    )


def cpa_region_prompt(row: dict, audit: dict) -> str:
    contact = audit['contact']
    return f'''Global task: {row["task_instruction"]}
End-effector type in this image: {cpa_actor_description(audit)}.
{cpa_contact_agent_instruction(audit)}
This is source contact frame {int(audit["candidate_frame"])} for the accepted intentional interaction {contact["verb"]} {contact["noun"]}. Localize the tight physical interaction region containing the contact agent's touching material and contacted object surface. For a held-tool event, prioritize the tool tip/object interface rather than the holder's hand.'''


def cpa_points_prompt(row: dict, audit: dict) -> str:
    actor = cpa_actor_description(audit)
    count_rule = (
        'First count distinct visibly touching fingers, hand surfaces, or held-tool interfaces. A single-finger touch is exactly one H_i/O_i pair; use multiple pairs only for separate visible interfaces.'
        if audit.get('actor') == 'human' else
        'A parallel-jaw grasp normally has one H_i/O_i pair per visibly contacting jaw, but use the actual visible interface count.'
    )
    return f'''Global task: {row["task_instruction"]}
End-effector type in this crop: {actor}.
{cpa_contact_agent_instruction(audit)}
The crop comes from accepted contact frame {int(audit["candidate_frame"])} for {audit["contact"]["verb"]} {audit["contact"]["noun"]}. {count_rule} For each interface i, H_i must lie on the explicitly specified contact-agent-side material and O_i on the contacted-object-side material immediately across that same interface. Return paired points on the crop's integer 0-to-1000 coordinate grid.'''


def cpa_sam2_prompt(row: dict, audit: dict) -> str:
    return f'''Global task: {row["task_instruction"]}
End-effector type in this crop: {cpa_actor_description(audit)}.
{cpa_contact_agent_instruction(audit)}
The contacted object category is {audit["contact"]["noun"]}. The crop is from accepted contact frame {int(audit["candidate_frame"])} for {audit["contact"]["verb"]} {audit["contact"]["noun"]}. Identify only the explicitly specified contact agent and contacted object, then provide their separate SAM2 boxes and interior seed points. For a held-tool event, the HAND-schema mask must segment the tool and exclude the holder's hand. Do not output H_i/O_i contact targets.'''


def forbidden_object_name(name: object) -> bool:
    normalized = re.sub(r'\s+', ' ', str(name or '').strip().lower())
    return normalized in FORBIDDEN_EXACT_NAMES or any(part in normalized for part in FORBIDDEN_NAME_PARTS)


def normalize_vlm_coordinates(values: object, expected_length: int) -> list[float]:
    """Normalize either native 0..1000 VLM coordinates or legacy 0..1 values."""
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError('coordinates_not_numeric') from None
    if len(coordinates) != expected_length:
        raise ValueError('coordinates_wrong_length')
    if any(value < 0 or value > VLM_COORDINATE_MAX for value in coordinates):
        raise ValueError('coordinates_out_of_range')
    scale = VLM_COORDINATE_MAX if any(value > 1 for value in coordinates) else 1
    return [round(value / scale, 6) for value in coordinates]


def safe_alternate_name(name: object, alternate: object, color: object) -> tuple[str, bool]:
    primary = re.sub(r'\s+', ' ', str(name or '').strip())
    candidate = re.sub(r'\s+', ' ', str(alternate or '').strip())
    p, a = primary.lower(), candidate.lower()
    color_tokens = set(re.findall(r'[a-z]+', str(color or '').lower()))
    p_tokens = set(re.findall(r'[a-z]+', p)) - color_tokens
    a_tokens = set(re.findall(r'[a-z]+', a)) - color_tokens
    shared_head = bool(p_tokens & a_tokens)
    approved = any(any(term in p for term in group) and any(term in a for term in group) for group in SAFE_SYNONYM_GROUPS)
    if candidate and (shared_head or approved):
        return candidate, True
    descriptive = f'{str(color).strip()} {primary}'.strip()
    if not str(color).strip() or str(color).strip().lower() in p:
        descriptive = primary
    return descriptive, False


def validate_inventory(inv: dict) -> tuple[list[dict], list[dict]]:
    out, filtered = [], []
    for obj in inv.get('objects') or []:
        try:
            box = normalize_vlm_coordinates(obj['bbox_xyxy'], 4)
            valid = 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
            if not valid or not obj.get('clearly_visible') or not obj.get('name'):
                continue
            if forbidden_object_name(obj['name']) or forbidden_object_name(obj.get('alternate_name')):
                filtered.append({'name': obj.get('name'), 'alternate_name': obj.get('alternate_name'), 'reason': 'non_manipulable_or_scene_entity'})
                continue
            reported_center = normalize_vlm_coordinates(obj.get('center_xy', []), 2)
            center = [round((box[0] + box[2]) / 2, 6), round((box[1] + box[3]) / 2, 6)]
            safe_alt, alt_accepted = safe_alternate_name(obj['name'], obj.get('alternate_name'), obj.get('color'))
            out.append({**obj, 'alternate_name': safe_alt, 'bbox_xyxy': box, 'center_xy': center, 'model_reported_center_xy': reported_center, 'model_reported_alternate_name': obj.get('alternate_name'), 'alternate_name_accepted': alt_accepted})
        except (KeyError, TypeError, ValueError):
            continue
    return out, filtered


def interval_overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def bbox_edge_distance(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a['bbox_xyxy']; bx1, by1, bx2, by2 = b['bbox_xyxy']
    dx = max(ax1 - bx2, bx1 - ax2, 0.0)
    dy = max(ay1 - by2, by1 - ay2, 0.0)
    return math.hypot(dx, dy)


def strict_bbox_relation(anchor: dict, candidate: dict) -> str | None:
    """Return candidate's direction from anchor only under full-box separation."""
    ax1, ay1, ax2, ay2 = anchor['bbox_xyxy']; bx1, by1, bx2, by2 = candidate['bbox_xyxy']
    x_align = interval_overlap(ax1, ax2, bx1, bx2) / max(1e-9, min(ax2 - ax1, bx2 - bx1)) >= 0.2
    y_align = interval_overlap(ay1, ay2, by1, by2) / max(1e-9, min(ay2 - ay1, by2 - by1)) >= 0.2
    if x_align and by2 <= ay1:
        return 'above'
    if x_align and by1 >= ay2:
        return 'below'
    if y_align and bx2 <= ax1:
        return 'left'
    if y_align and bx1 >= ax2:
        return 'right'
    return None


def answer_for(obj: dict, include_color: bool = False) -> dict:
    answer = {
        'object': obj['name'], 'alternate_name': obj['alternate_name'],
        'bbox_xyxy': obj['bbox_xyxy'], 'center_xy': obj['center_xy'],
    }
    if include_color:
        answer['color'] = obj['color']
    return answer


def nearest_unique(anchor: dict, objects: list[dict]) -> tuple[dict | None, dict]:
    ranked = sorted(((bbox_edge_distance(anchor, other), other) for other in objects if other is not anchor), key=lambda x: (x[0], x[1]['name']))
    audit = {'method': 'minimum_bbox_edge_distance', 'distances': [{'object': o['name'], 'distance': round(d, 6)} for d, o in ranked]}
    if not ranked:
        audit['accepted'] = False; audit['reason'] = 'no_other_eligible_object'
        return None, audit
    if len(ranked) == 1:
        audit['accepted'] = True; audit['uniqueness_rule'] = 'only_candidate'
        return ranked[0][1], audit
    first, second = ranked[0][0], ranked[1][0]
    unique = (second - first >= 0.03) or (first <= 0.6 * second)
    audit.update({'accepted': unique, 'nearest_margin': round(second - first, 6), 'nearest_ratio': round(first / max(second, 1e-9), 6), 'uniqueness_rule': 'margin>=0.03_or_ratio<=0.6'})
    return (ranked[0][1] if unique else None), audit


def strict_direction_pair(primary: dict, objects: list[dict]) -> tuple[dict, dict, str, dict] | None:
    candidates = []
    anchors = [primary] + [x for x in objects if x is not primary]
    for anchor_rank, anchor in enumerate(anchors):
        for other in objects:
            if other is anchor:
                continue
            direction = strict_bbox_relation(anchor, other)
            if direction:
                candidates.append((anchor_rank, bbox_edge_distance(anchor, other), anchor, other, direction))
    if not candidates:
        return None
    anchor_rank, distance, anchor, other, direction = min(candidates, key=lambda x: (x[0], x[1], x[2]['name'], x[3]['name']))
    return anchor, other, direction, {
        'method': 'strict_full_bbox_separation_with_orthogonal_overlap',
        'full_box_relation': True, 'orthogonal_overlap_min_ratio': 0.2,
        'bbox_edge_distance': round(distance, 6), 'primary_anchor_used': anchor_rank == 0,
    }


def grounding_qa(row: dict, inv: dict) -> tuple[str, str, str, dict]:
    objects, filtered = validate_inventory(inv)
    if not objects:
        raise ValueError('grounding_no_valid_visible_objects')
    obj = objects[0]
    typ = row['grounding_question_type']; name = obj['name']; alt = obj['alternate_name']; color = obj['color']; box = obj['bbox_xyxy']; center = obj['center_xy']
    answer = answer_for(obj, include_color=True)
    relation_audit = None
    actual_type = typ
    if typ == 'object_location': question = f'Where is the {alt}?'
    elif typ == 'color_object_location': question = f'Where is the {color} {name}?'
    elif typ == 'neighbor_name':
        other, relation_audit = nearest_unique(obj, objects)
        if other is None:
            actual_type = 'object_location'; question = f'Where is the {alt}?'
        else:
            question = f'Which eligible object is closest to the {name} by bounding-box edge distance?'; answer = answer_for(other)
    elif typ == 'direction_relation':
        pair = strict_direction_pair(obj, objects)
        if pair is None:
            actual_type = 'object_location'; question = f'Where is the {alt}?'
            relation_audit = {'method': 'strict_full_bbox_separation_with_orthogonal_overlap', 'accepted': False, 'reason': 'no_unambiguous_direction_pair'}
        else:
            anchor, other, direction, relation_audit = pair
            question = f'What eligible object is strictly {direction} the {anchor["name"]}?'; answer = answer_for(other)
    elif typ == 'presence': question = f'Is there a {name} in the image?'; answer = {'present': True, **answer}
    else: actual_type = 'object_location'; question = f'Where is the {name}?'
    audit = {
        'objects': objects, 'filtered_objects': filtered, 'inventory': inv,
        'inventory_suitable_for_global_task': bool(inv.get('suitable')),
        'inventory_suitability_reason': str(inv.get('reason') or ''),
        'relation_audit': relation_audit,
        'requested_question_type': typ, 'actual_question_type': actual_type,
    }
    return question, json.dumps(answer, ensure_ascii=False), actual_type, audit


def concise_sentence(value: object, field: str, max_words: int) -> str:
    """Validate the deliberately strict one-sentence ECoT style contract."""
    text = re.sub(r'\s+', ' ', str(value or '').strip())
    if field == 'task_progress_assessment':
        # Some OpenAI-compatible VLMs preserve the required judgment words but
        # lowercase the leading ``Task``.  Canonicalize only that fixed suffix;
        # the semantic complete/incomplete decision remains unchanged.
        text = re.sub(
            r'task not yet complete\.$', 'Task not yet complete.', text,
            flags=re.I,
        )
        text = re.sub(
            r'task complete\.$', 'Task complete.', text,
            flags=re.I,
        )
        text = re.sub(
            r'\.\s+(Task (?:not yet )?complete\.)$', r'; \1', text,
        )
    if not text or not text.endswith('.') or re.findall(r'[.!?]', text) != ['.']:
        raise ValueError(f'ecot_one_sentence_contract_failed:{field}:{text}')
    if len(text.split()) > max_words:
        raise ValueError(f'ecot_conciseness_contract_failed:{field}:{len(text.split())}')
    return text


def is_concise_sentence(value: object, max_words: int) -> bool:
    text = re.sub(r'\s+', ' ', str(value or '').strip())
    return (
        bool(text) and text.endswith('.') and
        re.findall(r'[.!?]', text) == ['.'] and
        len(text.split()) <= max_words
    )


def is_valid_ecot_subtask(value: object, progress: object) -> bool:
    if not isinstance(value, dict) or set(value) != set(ECOT_SUBTASK_FIELDS):
        return False
    text = str(value.get('subtask') or '')
    if not is_concise_sentence(text, 24) or text != text.lower():
        return False
    if re.search(r'\bthen\b|\bafter that\b|;', text, flags=re.I):
        return False
    if any(
        not str(value.get(field) or '').strip() or (
            str(value[field]).strip() != ECOT_MISSING_SEMANTIC and
            str(value[field]).strip() != str(value[field]).strip().lower()
        )
        for field in ECOT_SUBTASK_FIELDS[1:]
    ):
        return False
    if value['action'] != ECOT_MISSING_SEMANTIC and ' and ' in value['action']:
        return False
    complete = str(progress).endswith('Task complete.')
    empty = (
        text == 'no further subtask remains.' and
        all(value[field] == ECOT_MISSING_SEMANTIC for field in ECOT_SUBTASK_FIELDS[1:])
    )
    return empty if complete else not empty


def parse_ecot_subtask(value: object, progress: str) -> dict:
    raw_subtask = value if isinstance(value, dict) else {}
    if set(raw_subtask) != set(ECOT_SUBTASK_FIELDS):
        raise ValueError(f'ecot_subtask_fields_invalid:{sorted(raw_subtask)}')
    raw_text = re.sub(
        r'\s+', ' ', str(raw_subtask.get('subtask') or '').strip(),
    ).lower()
    if raw_text and not re.search(r'[.!?]$', raw_text):
        raw_text += '.'
    subtask_text = concise_sentence(raw_text, 'subtask', 24)
    subtask = {'subtask': subtask_text}
    for field in ECOT_SUBTASK_FIELDS[1:]:
        value = re.sub(r'\s+', ' ', str(raw_subtask.get(field) or '').strip())
        if not value:
            raise ValueError(f'ecot_subtask_semantic_invalid:{field}:{value}')
        value = (
            ECOT_MISSING_SEMANTIC
            if value.casefold() == ECOT_MISSING_SEMANTIC.casefold()
            else value.lower()
        )
        if field == 'action' and value != ECOT_MISSING_SEMANTIC:
            value = value.rsplit(' and ', 1)[-1]
        subtask[field] = value
    complete = progress.endswith('Task complete.')
    if complete and (
        subtask_text != 'no further subtask remains.' or
        any(
            subtask[field] != ECOT_MISSING_SEMANTIC
            for field in ECOT_SUBTASK_FIELDS[1:]
        )
    ):
        raise ValueError('ecot_completed_task_must_have_empty_subtask')
    if not complete and subtask_text == 'no further subtask remains.':
        raise ValueError('ecot_incomplete_task_missing_subtask')
    if re.search(r'\bthen\b|\bafter that\b|;', subtask_text, flags=re.I):
        raise ValueError(f'ecot_subtask_not_single_goal:{subtask_text}')
    return subtask


def render_ecot(scene: str, progress: str, subtask: dict) -> str:
    return (
        f'Scene Description: {scene}\n'
        f'Task Progress Assessment: {progress}\n'
        f'Subtask [action={subtask["action"]}; object={subtask["object"]}; '
        f'source={subtask["source"]}; target={subtask["target"]}]: '
        f'{subtask["subtask"]}'
    )


def parse_ecot_response(response: dict) -> tuple[dict, str]:
    data = json.loads(get_content(response))
    scene = concise_sentence(data.get('scene_description'), 'scene_description', 35)
    progress = concise_sentence(
        data.get('task_progress_assessment'), 'task_progress_assessment', 30,
    )
    if not progress.endswith(('Task complete.', 'Task not yet complete.')):
        raise ValueError(f'ecot_completion_judgment_missing:{progress}')
    subtask = parse_ecot_subtask(data.get('subtask'), progress)
    if ECOT_PRIVILEGED_LEAK_PATTERN.search(
        f'{scene} {progress} {subtask["subtask"]}',
    ):
        raise ValueError('ecot_privileged_context_leak')
    structured = {
        'scene_description': scene,
        'task_progress_assessment': progress,
        'subtask': subtask,
    }
    rendered = render_ecot(scene, progress, subtask)
    return structured, rendered


def parse_ecot_progress_response(response: dict) -> str:
    data = json.loads(get_content(response))
    return concise_sentence(
        data.get('target_frame_progress_summary'),
        'target_frame_progress_summary', 48,
    )


def extract_ecot_context_frames(row: dict, context_root: Path) -> dict:
    """Extract one auxiliary frame per second before and after the target frame."""
    source = Path(row['source_video'])
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise RuntimeError(f'ecot_video_probe_failed:{row["id"]}:{fps}:{frame_count}')
    target = int(row['selected_frame_index'])
    step = max(1, round(fps))
    past = list(range(target - step, -1, -step))[::-1]
    future = list(range(target + step, frame_count, step))
    row_dir = context_root/row['id']
    row_dir.mkdir(parents=True, exist_ok=True)
    sampled_frames = []
    for relation, indices in (('past', past), ('future', future)):
        for frame_index in indices:
            offset_seconds = round((frame_index - target) / fps, 6)
            image_path = row_dir/f'{relation}_{abs(round(offset_seconds)):03d}s_frame_{frame_index:06d}.jpg'
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = cap.read()
            if not ok or image is None or not cv2.imwrite(
                str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90],
            ):
                cap.release()
                raise RuntimeError(f'ecot_context_frame_failed:{row["id"]}:{frame_index}')
            sampled_frames.append({
                'temporal_relation': relation,
                'offset_seconds': offset_seconds,
                'frame_index': frame_index,
                'image': f'{row["id"]}/{image_path.name}',
                'image_path': str(image_path),
                'image_sha256': sha256(image_path),
            })
    cap.release()
    return {
        'source_video': str(source),
        'source_video_fps': fps,
        'source_video_frame_count': frame_count,
        'target_frame_index': target,
        'target_time_seconds': round(target / fps, 6),
        'target_progress_ratio': round(target / max(1, frame_count - 1), 6),
        'sample_interval_seconds': 1.0,
        'sampled_frames': sampled_frames,
    }


def ecot_progress_request(endpoint: str, row: dict, context: dict) -> tuple[str, str, str]:
    user_text = f'''Global task: {row["task_instruction"]}
The target is source frame {context["target_frame_index"]} of {context["source_video_frame_count"]}, timestamp {context["target_time_seconds"]:.6f} seconds, or {context["target_progress_ratio"]:.6f} through the episode. Watch the full episode first, match the target image to it, and summarize the target frame's task stage in exactly one concise sentence.'''
    content = [
        {'type': 'text', 'text': 'FULL EPISODE VIDEO:'},
        {'type': 'video_url', 'video_url': {'url': video_url(Path(context['source_video']))}},
        {'type': 'text', 'text': 'TARGET FRAME IMAGE:'},
        {'type': 'image_url', 'image_url': {'url': image_url(Path(row['image_path']))}},
        {'type': 'text', 'text': user_text},
    ]
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': ECOT_PROGRESS_SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
        'temperature': 0, 'max_tokens': 300,
        'response_format': {'type': 'json_schema', 'json_schema': ecot_progress_schema()},
    }
    response = post(endpoint, payload)
    raw = get_content(response)
    return parse_ecot_progress_response(response), user_text, raw


def ecot_request(endpoint: str, row: dict, context: dict) -> tuple[dict, str, str, str]:
    user_text = ecot_user_prompt(row['task_instruction'], context)
    content = [
        {'type': 'text', 'text': 'CURRENT TARGET OBSERVATION (training-visible image):'},
        {'type': 'image_url', 'image_url': {'url': image_url(Path(row['image_path']))}},
    ]
    for sampled in context['sampled_frames']:
        label = (
            f'AUXILIARY {sampled["temporal_relation"].upper()} '
            f'{abs(sampled["offset_seconds"]):.3f} seconds '
            f'(source frame {sampled["frame_index"]}):'
        )
        content.extend([
            {'type': 'text', 'text': label},
            {'type': 'image_url', 'image_url': {'url': image_url(Path(sampled['image_path']))}},
        ])
    content.append({'type': 'text', 'text': user_text})
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': ECOT_SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
        'temperature': 0, 'max_tokens': 700,
        'response_format': {'type': 'json_schema', 'json_schema': ecot_schema()},
    }
    response = post(endpoint, payload)
    structured, rendered = parse_ecot_response(response)
    return structured, rendered, user_text, get_content(response)


def annotation_call_with_retries(label: str, function, max_attempts: int = 3):
    attempts = []
    for attempt in range(1, max_attempts + 1):
        try:
            result = function()
            attempts.append({'attempt': attempt, 'passed': True, 'error': None})
            return result, attempts
        except Exception as error:
            attempts.append({
                'attempt': attempt, 'passed': False,
                'error': f'{type(error).__name__}:{error}',
            })
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f'{label}_failed:{attempts}')


def cpa_payload(
    system_prompt: str,
    image: Path,
    user_prompt: str,
    schema: dict,
    model: str | None = None,
) -> dict:
    return {
        'model': model or MODEL,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': image_url(image)}},
                {'type': 'text', 'text': user_prompt},
            ]},
        ],
        'temperature': 0,
        'max_tokens': 600,
        'response_format': {'type': 'json_schema', 'json_schema': schema},
    }


def cpa_tracking_review_payload(
    raw_image: Path,
    tracked_overlay: Path,
    user_prompt: str,
    pair_indices: list[int],
) -> dict:
    return {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': CPA_TRACKING_REVIEW_SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'MEDIA A — raw tracked-frame observation:'},
                {'type': 'image_url', 'image_url': {'url': image_url(raw_image)}},
                {'type': 'text', 'text': 'MEDIA B — the same observation with tracked H_i/O_i labels:'},
                {'type': 'image_url', 'image_url': {'url': image_url(tracked_overlay)}},
                {'type': 'text', 'text': user_prompt},
            ]},
        ],
        'temperature': 0,
        'max_tokens': 700,
        'response_format': {
            'type': 'json_schema',
            'json_schema': cpa_tracking_review_schema(pair_indices),
        },
    }


def parse_cpa_region(response: dict) -> tuple[list[float], str, str]:
    raw = get_content(response)
    data = json.loads(raw)
    reason = re.sub(r'\s+', ' ', str(data.get('reason') or '').strip())
    if not reason:
        raise ValueError('cpa_region_reason_missing')
    bbox = normalize_vlm_coordinates(data.get('interaction_bbox_xyxy'), 4)
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise ValueError(f'cpa_region_bbox_degenerate:{bbox}')
    return bbox, reason, raw


def parse_cpa_points(response: dict, actor: str) -> tuple[list[dict], str, str]:
    raw = get_content(response)
    data = json.loads(raw)
    reason = re.sub(r'\s+', ' ', str(data.get('reason') or '').strip())
    if not reason:
        raise ValueError('cpa_points_reason_missing')
    raw_pairs = data.get('contact_pairs')
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError('cpa_contact_pairs_missing')
    pairs = []
    seen_indices = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, dict):
            raise ValueError('cpa_contact_pair_not_object')
        pair_index = raw_pair.get('pair_index')
        if not isinstance(pair_index, int) or pair_index in seen_indices:
            raise ValueError(f'cpa_contact_pair_index_invalid:{pair_index}')
        hand_point = normalize_vlm_coordinates(raw_pair.get('hand_point_xy'), 2)
        object_point = normalize_vlm_coordinates(raw_pair.get('object_point_xy'), 2)
        if hand_point == object_point:
            raise ValueError(f'cpa_contact_pair_points_identical:{pair_index}')
        seen_indices.add(pair_index)
        pairs.append({
            'pair_index': pair_index,
            'hand_point_xy': hand_point,
            'object_point_xy': object_point,
        })
    pairs.sort(key=lambda item: item['pair_index'])
    if [item['pair_index'] for item in pairs] != list(range(len(pairs))):
        raise ValueError('cpa_contact_pair_indices_not_contiguous')
    if not 1 <= len(pairs) <= 5:
        raise ValueError(f'cpa_contact_pair_count_invalid:{actor}:{len(pairs)}')
    return pairs, reason, raw


def parse_cpa_sam2_prompts(response: dict) -> tuple[dict, str]:
    raw = get_content(response)
    data = json.loads(raw)
    parsed = {}
    for key in ('reason', 'hand_description', 'object_description'):
        value = re.sub(r'\s+', ' ', str(data.get(key) or '').strip())
        if not value:
            raise ValueError(f'cpa_sam2_{key}_missing')
        parsed[key] = value
    for role in ('hand', 'object'):
        bbox = normalize_vlm_coordinates(data.get(f'{role}_bbox_xyxy'), 4)
        point = normalize_vlm_coordinates(data.get(f'{role}_seed_point_xy'), 2)
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError(f'cpa_sam2_{role}_bbox_degenerate:{bbox}')
        if not (bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]):
            raise ValueError(f'cpa_sam2_{role}_seed_outside_bbox:{point}:{bbox}')
        parsed[f'{role}_bbox_xyxy'] = bbox
        parsed[f'{role}_seed_point_xy'] = point
    if parsed['hand_seed_point_xy'] == parsed['object_seed_point_xy']:
        raise ValueError('cpa_sam2_seed_points_identical')
    if parsed['hand_bbox_xyxy'] == parsed['object_bbox_xyxy']:
        raise ValueError('cpa_sam2_bboxes_identical')
    return parsed, raw


def parse_cpa_tracking_review(response: dict, pair_indices: list[int]) -> dict:
    raw = get_content(response)
    data = json.loads(raw)
    reason = re.sub(r'\s+', ' ', str(data.get('reason') or '').strip())
    if not reason:
        raise ValueError('cpa_tracking_review_reason_missing')
    raw_checks = data.get('checks')
    if not isinstance(raw_checks, list):
        raise ValueError('cpa_tracking_review_checks_missing')
    checks = {}
    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            raise ValueError('cpa_tracking_review_check_not_object')
        pair_index = raw_check.get('pair_index')
        if pair_index in checks:
            raise ValueError(f'cpa_tracking_review_duplicate_pair:{pair_index}')
        hand_valid = raw_check.get('hand_on_end_effector')
        object_valid = raw_check.get('object_on_contacted_object')
        if pair_index not in pair_indices or not isinstance(hand_valid, bool) or not isinstance(object_valid, bool):
            raise ValueError(f'cpa_tracking_review_check_invalid:{pair_index}')
        checks[pair_index] = {
            'pair_index': pair_index,
            'hand_on_end_effector': hand_valid,
            'object_on_contacted_object': object_valid,
        }
    if sorted(checks) != sorted(pair_indices):
        raise ValueError(f'cpa_tracking_review_pair_coverage:{sorted(checks)}:{sorted(pair_indices)}')
    return {
        'reason': reason,
        'checks': [checks[index] for index in sorted(checks)],
        'raw_model_output': raw,
    }


def write_cpa_pair_overlay(
    crop_path: Path,
    pairs: list[dict],
    output_path: Path,
) -> None:
    image = cv2.imread(str(crop_path))
    if image is None:
        raise RuntimeError(f'cpa_pair_overlay_decode_failed:{crop_path}')
    for pair in pairs:
        index = int(pair['pair_index'])
        color = CPA_POINT_COLORS[index % len(CPA_POINT_COLORS)]
        draw_cpa_point(image, pair['hand_point_xy'], f'H{index}', color, 'hand')
        draw_cpa_point(image, pair['object_point_xy'], f'O{index}', color, 'object')
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
        raise RuntimeError(f'cpa_pair_overlay_write_failed:{output_path}')


def cpa_crop_from_contact_frame(
    row: dict,
    audit: dict,
    bbox: list[float],
    contact_frame_dir: Path,
    crop_dir: Path,
) -> tuple[Path, Path, list[int], int, int]:
    source = Path(row.get('source_video') or video_path(row))
    capture = cv2.VideoCapture(str(source))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(audit['candidate_frame']))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f'cpa_contact_frame_decode_failed:{audit["candidate_id"]}')
    height, width = frame.shape[:2]
    contact_path = contact_frame_dir/f'{audit["candidate_id"]}.jpg'
    if not cv2.imwrite(str(contact_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f'cpa_contact_frame_write_failed:{contact_path}')
    x1 = max(0, min(width - 2, math.floor(bbox[0] * width)))
    y1 = max(0, min(height - 2, math.floor(bbox[1] * height)))
    x2 = max(x1 + 2, min(width, math.ceil(bbox[2] * width)))
    y2 = max(y1 + 2, min(height, math.ceil(bbox[3] * height)))
    crop = frame[y1:y2, x1:x2]
    crop_path = crop_dir/f'{audit["candidate_id"]}.jpg'
    if crop.size == 0 or not cv2.imwrite(
        str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 96],
    ):
        raise RuntimeError(f'cpa_crop_write_failed:{crop_path}')
    return contact_path, crop_path, [x1, y1, x2, y2], width, height


def annotate_cpa_contact(
    endpoint: str,
    row: dict,
    audit: dict,
    contact_frame_dir: Path,
    crop_dir: Path,
    max_attempts: int,
) -> dict:
    started = time.time()
    source = Path(row.get('source_video') or video_path(row))
    audit_for_request = {**audit, 'source_video': str(source)}
    region_reference = audit_for_request.get('cpa_region_reference')
    if isinstance(region_reference, dict):
        bbox = normalize_vlm_coordinates(
            region_reference.get('interaction_bbox_xyxy'), 4,
        )
        if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
            raise ValueError(f'cpa_reused_region_bbox_degenerate:{bbox}')
        region_reason = str(region_reference.get('region_reason') or '').strip()
        region_raw = str(region_reference.get('region_raw_model_output') or '')
        if not region_reason or not region_raw:
            raise ValueError('cpa_reused_region_provenance_missing')
        region_prompt = str(
            region_reference.get('region_user_prompt') or
            cpa_region_prompt(row, audit_for_request)
        )
        region_attempts = [{
            'attempt': 0, 'passed': True, 'error': None,
            'source': 'reused_seed_cpa_interaction_region',
        }]
        region_reused_from = region_reference.get('candidate_id')
    else:
        temporary_frame, _, _, _, _ = cpa_crop_from_contact_frame(
            row, audit_for_request, [0.0, 0.0, 1.0, 1.0],
            contact_frame_dir, crop_dir,
        )
        region_prompt = cpa_region_prompt(row, audit_for_request)
        region_result, region_attempts = annotation_call_with_retries(
            'cpa_interaction_region',
            lambda: parse_cpa_region(post(
                endpoint,
                cpa_payload(
                    CPA_REGION_SYSTEM_PROMPT, temporary_frame, region_prompt,
                    cpa_region_schema(),
                ),
            )),
            max_attempts,
        )
        bbox, region_reason, region_raw = region_result
        region_reused_from = None
    contact_path, crop_path, crop_pixels, width, height = cpa_crop_from_contact_frame(
        row, audit_for_request, bbox, contact_frame_dir, crop_dir,
    )
    sam2_prompts = None
    if CPA_POINT_METHOD == 'seed_vlm':
        points_system_prompt = CPA_POINTS_SYSTEM_PROMPT
        points_prompt = cpa_points_prompt(row, audit_for_request)
        points_result, points_attempts = annotation_call_with_retries(
            'cpa_contact_points',
            lambda: parse_cpa_points(post(
                endpoint,
                cpa_payload(
                    points_system_prompt, crop_path, points_prompt,
                    cpa_points_schema(audit_for_request.get('actor', 'robot')),
                    model=CPA_POINT_MODEL,
                ),
            ), audit_for_request.get('actor', 'robot')),
            max_attempts,
        )
        crop_pairs, points_reason, points_raw = points_result
        pair_overlay_path = crop_dir/f'{audit["candidate_id"]}_pair_candidates.jpg'
        write_cpa_pair_overlay(crop_path, crop_pairs, pair_overlay_path)
        point_pair_version = CPA_POINT_PAIR_VERSION
    elif CPA_POINT_METHOD == 'sam2':
        points_system_prompt = CPA_SAM2_PROMPT_SYSTEM_PROMPT
        points_prompt = cpa_sam2_prompt(row, audit_for_request)
        prompt_result, points_attempts = annotation_call_with_retries(
            'cpa_sam2_prompts',
            lambda: parse_cpa_sam2_prompts(post(
                endpoint,
                cpa_payload(
                    points_system_prompt, crop_path, points_prompt,
                    cpa_sam2_prompt_schema(),
                    model=CPA_POINT_MODEL,
                ),
            )),
            max_attempts,
        )
        sam2_prompts, points_raw = prompt_result
        points_reason = sam2_prompts['reason']
        crop_pairs = []
        pair_overlay_path = None
        point_pair_version = CPA_SAM2_POINT_PAIR_VERSION
    else:
        raise RuntimeError(f'unsupported_cpa_point_method:{CPA_POINT_METHOD}')
    x1, y1, x2, y2 = bbox
    full_pairs = []
    for pair in crop_pairs:
        full_pair = {'pair_index': pair['pair_index']}
        for role in ('hand', 'object'):
            point = pair[f'{role}_point_xy']
            full_pair[f'{role}_point_xy'] = [
                round(x1 + point[0] * (x2 - x1), 6),
                round(y1 + point[1] * (y2 - y1), 6),
            ]
        full_pairs.append(full_pair)
    contact_agent = cpa_contact_agent_semantics(audit_for_request)
    return {
        'candidate_id': audit['candidate_id'],
        'row_id': audit['row_id'],
        'dataset': audit.get('dataset', 'Robot'),
        'dataset_name': audit.get('dataset_name', audit.get('dataset', 'Robot')),
        'dataset_family': audit.get('dataset_family', audit.get('actor', 'robot')),
        'actor': audit.get('actor', 'robot'),
        'task_instruction': row['task_instruction'],
        'source_video': str(source),
        'source_video_fps': float(audit['source_video_fps']),
        'episode_video': audit.get('episode_video', f'{audit["row_id"]}.mp4'),
        'candidate_frame': int(audit['candidate_frame']),
        'contact': audit['contact'],
        'observations': audit['observations'],
        'end_effector_type': cpa_actor_description(audit_for_request),
        'contact_agent_role_version': contact_agent['role_version'],
        'contact_agent_role': contact_agent['role'],
        'contact_agent_name': contact_agent['agent_name'],
        'contact_object_participant': contact_agent['object_name'],
        'contact_event_type': contact_agent['event_type'],
        'pro_contact_pair': contact_agent['contact_pair'],
        'interaction_bbox_xyxy': bbox,
        'interaction_bbox_pixel_xyxy': crop_pixels,
        'contact_point_pairs_crop_xy': crop_pairs,
        'contact_point_pairs_full_xy': full_pairs,
        'contact_point_pair_count': len(full_pairs),
        'contact_frame_image': contact_path.name,
        'contact_frame_image_path': str(contact_path),
        'interaction_crop_image': crop_path.name,
        'interaction_crop_image_path': str(crop_path),
        'source_resolution': [width, height],
        'region_reason': region_reason,
        'region_reused_from': region_reused_from,
        'points_reason': points_reason,
        'point_pair_version': point_pair_version,
        'point_model': CPA_POINT_MODEL,
        'point_method': CPA_POINT_METHOD,
        'pair_candidate_overlay': pair_overlay_path.name if pair_overlay_path else None,
        'sam2_prompts': sam2_prompts,
        'region_system_prompt': CPA_REGION_SYSTEM_PROMPT,
        'region_user_prompt': region_prompt,
        'points_system_prompt': points_system_prompt,
        'points_user_prompt': points_prompt,
        'region_raw_model_output': region_raw,
        'points_raw_model_output': points_raw,
        'region_raw_model_output_sha256': hashlib.sha256(region_raw.encode()).hexdigest(),
        'points_raw_model_output_sha256': hashlib.sha256(points_raw.encode()).hexdigest(),
        'region_attempts': region_attempts,
        'points_attempts': points_attempts,
        'elapsed_seconds': round(time.time() - started, 3),
    }


def normalize_sta_category(value: object) -> str:
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower())).strip('_')


def parse_contact_response(response: dict, candidate: dict) -> dict:
    data = json.loads(get_content(response))
    valid = bool(data.get('valid_contact'))
    reason = re.sub(r'\s+', ' ', str(data.get('reason') or '').strip())
    if not reason:
        raise ValueError('contact_reason_missing')
    verb = normalize_sta_category(data.get('verb'))
    noun = normalize_sta_category(data.get('noun'))
    bbox = data.get('object_bbox_xyxy')
    normalized_bbox = None
    bbox_error = None
    policy_overrides = []
    if bbox is not None:
        try:
            normalized_bbox = normalize_vlm_coordinates(bbox, 4)
        except ValueError as error:
            bbox_error = f'contact_bbox_invalid:{error}'
        if normalized_bbox is not None:
            x1, y1, x2, y2 = normalized_bbox
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                bbox_error = 'contact_bbox_invalid_geometry'
    if valid:
        if not verb or not noun:
            raise ValueError('valid_contact_missing_verb_or_noun')
        if forbidden_object_name(noun.replace('_', ' ')):
            policy_overrides.append(f'rejected_forbidden_primary_object:{noun}')
            valid, verb, noun, normalized_bbox = False, '', '', None
        elif candidate['grounding_episode'] and (normalized_bbox is None or bbox_error):
            policy_overrides.append(f'rejected_unreliable_grounding_bbox:{bbox_error or "missing"}')
            valid, verb, noun, normalized_bbox = False, '', '', None
        elif not candidate['grounding_episode']:
            if bbox is not None:
                policy_overrides.append('discarded_unrequested_nongrounding_bbox')
            normalized_bbox = None
    else:
        if verb or noun or normalized_bbox is not None:
            policy_overrides.append('discarded_labels_for_invalid_contact')
        verb, noun, normalized_bbox = '', '', None
    return {
        'valid_contact': valid,
        'reason': reason,
        'verb': verb if valid else None,
        'noun': noun if valid else None,
        'object_bbox_xyxy': normalized_bbox if valid else None,
        'policy_overrides': policy_overrides,
    }


def contact_request(endpoint: str, row: dict, candidate: dict, max_attempts: int) -> dict:
    attempts = []
    for attempt in range(1, max_attempts + 1):
        content = [
            {'type': 'text', 'text': 'MEDIA A — learner-facing pre-contact query image:'},
            {'type': 'image_url', 'image_url': {'url': image_url(Path(candidate['query_image_path']))}},
            {'type': 'text', 'text': 'MEDIA B — teacher-only two-second contact-validation clip:'},
            {'type': 'video_url', 'video_url': {'url': video_url(Path(candidate['teacher_clip_path']))}},
            {'type': 'text', 'text': contact_prompt(row, candidate)},
        ]
        payload = {
            'model': MODEL,
            'messages': [{'role': 'user', 'content': content}],
            'temperature': 0,
            'max_tokens': 700,
            'response_format': {'type': 'json_schema', 'json_schema': contact_schema()},
        }
        try:
            response = post(endpoint, payload)
            parsed = parse_contact_response(response, candidate)
            attempts.append({'attempt': attempt, 'passed': True, 'error': None})
            contact = None
            if parsed['valid_contact']:
                contact = {
                    'time': int(candidate['candidate_frame']),
                    'verb': parsed['verb'],
                    'noun': parsed['noun'],
                }
            return {
                **candidate,
                'inference_backend': BACKEND,
                'model': MODEL,
                'valid_contact': parsed['valid_contact'],
                'contact': contact,
                'object_bbox_xyxy': parsed['object_bbox_xyxy'],
                'object_bbox_provenance': (
                    'contact_teacher_localization_on_primary_sta_query_image'
                    if parsed['object_bbox_xyxy'] is not None else None
                ),
                'static_grounding_record_used': False,
                'model_reason': parsed['reason'],
                'policy_overrides': parsed['policy_overrides'],
                'attempts': attempts,
                'raw_model_output': get_content(response),
                'teacher_visibility_policy': 'query_image_plus_centered_contact_clip_for_annotation_only',
            }
        except Exception as error:
            attempts.append({'attempt': attempt, 'passed': False, 'error': f'{type(error).__name__}:{error}'})
    raise RuntimeError(f'contact_annotation_failed:{candidate["candidate_id"]}:{attempts}')


def build_sta_records(row: dict, contact_audits: list[dict]) -> list[dict]:
    records = []
    episode = {
        'task_dir': row['task_dir'], 'episode_index': row['episode_index'],
        'length': row['episode_length'],
    }
    specs = (
        ('next_contact_noun', 'What is the noun category of the next contact object?'),
        ('next_contact_verb', 'What verb describes the next contact action?'),
        ('time_to_next_contact', 'How many seconds until the next contact?'),
    )
    for audit in contact_audits:
        if not audit['valid_contact']:
            continue
        contact = audit['contact']
        observations = audit.get('observations') or [{
            'observation_id': f'{audit["candidate_id"]}_obs_00',
            'query_frame_index': audit['query_frame_index'],
            'query_image_path': audit['query_image_path'],
            'query_image': audit['query_image'],
            'time_to_contact_seconds': audit['time_to_contact_seconds'],
            'segment_progress': None,
            'is_primary_teacher_observation': True,
        }]
        if any(
            observation['query_frame_index'] < audit['query_frame_index'] or
            observation['query_frame_index'] >= contact['time'] or
            observation['time_to_contact_seconds'] >
            audit['time_to_contact_seconds'] + 1e-6
            for observation in observations
        ):
            raise RuntimeError(
                f'sta_observation_outside_anticipation_window:{audit["candidate_id"]}'
            )
        dataset_name = str(
            row.get('dataset_name') or row.get('dataset') or
            audit.get('dataset_name') or audit.get('dataset') or 'Robot'
        )
        dataset_family = str(
            row.get('dataset_family') or audit.get('dataset_family') or
            row.get('actor') or audit.get('actor') or 'robot'
        )
        actor = str(row.get('actor') or audit.get('actor') or dataset_family)
        for observation in observations:
            observation_id = observation['observation_id']
            base = {
                'kind': 'sta', 'contact_id': audit['candidate_id'],
                'dataset': dataset_name, 'dataset_name': dataset_name,
                'dataset_family': dataset_family, 'actor': actor,
                'observation_id': observation_id,
                'episode': episode, 'task_instruction': row['task_instruction'],
                'query_frame_index': observation['query_frame_index'],
                'segment_start_frame': audit.get('segment_start_frame'),
                'segment_end_frame_exclusive': audit.get(
                    'segment_end_frame_exclusive', audit['candidate_frame'],
                ),
                'observation_window_start_frame': audit.get(
                    'observation_window_start_frame', audit['query_frame_index'],
                ),
                'observation_window_end_frame_exclusive': audit.get(
                    'observation_window_end_frame_exclusive', audit['candidate_frame'],
                ),
                'segment_progress': observation.get('segment_progress'),
                'anticipation_window_progress': observation.get(
                    'anticipation_window_progress',
                ),
                'is_primary_teacher_observation': observation[
                    'is_primary_teacher_observation'
                ],
                'image': observation['query_image'],
                'image_sha256': sha256(Path(observation['query_image_path'])),
                'next_contact': contact,
                'teacher_evidence': {
                    'candidate_transition_frame': audit['candidate_frame'],
                    'clip': audit['teacher_clip'],
                    'context_start_frame': audit['context_start_frame'],
                    'context_end_frame': audit['context_end_frame'],
                    'anticipation_seconds_requested': audit.get(
                        'anticipation_seconds_requested',
                    ),
                    'future_media_passed_to_learner': False,
                },
                'visibility_policy': 'learner_input_is_one_bounded_anticipation_window_precontact_image_only; centered_contact_clip_is_teacher_audit_only',
            }
            values = {
                'next_contact_noun': contact['noun'],
                'next_contact_verb': contact['verb'],
                'time_to_next_contact': observation['time_to_contact_seconds'],
            }
            for question_type, question in specs:
                value = values[question_type]
                answer = (
                    f'{value:.6f} seconds'
                    if question_type == 'time_to_next_contact' else str(value)
                )
                records.append({
                    **base,
                    'id': f'{observation_id}_{question_type}',
                    'question_type': question_type,
                    'question': question,
                    'answer': answer,
                    'answer_value': value,
                })
            # The teacher localized the object only in the original query image.
            # Reusing that box on other wrist-camera frames would be incorrect.
            if (
                audit['grounding_episode'] and
                observation['is_primary_teacher_observation']
            ):
                bbox = audit['object_bbox_xyxy']
                records.append({
                    **base,
                    'id': f'{observation_id}_next_contact_bbox',
                    'question_type': 'next_contact_bbox',
                    'question': 'What is the normalized bounding box of the next contact object?',
                    'answer': json.dumps(bbox, ensure_ascii=False),
                    'answer_value': bbox,
                    'bbox_label_source': 'teacher_localization_on_this_observation',
                })
    return records


def annotate(
    row: dict,
    endpoint: str,
    ecot_context_dir: Path | None = None,
    max_attempts: int = 3,
) -> dict:
    image = Path(row['image_path']); started = time.time()
    base = {
        'id': row['id'], 'kind': row['kind'],
        'parent_episode_id': row.get('parent_episode_id', row['id']),
        'dataset': row['dataset'], 'dataset_name': row['dataset_name'],
        'dataset_family': row['dataset_family'], 'actor': row['actor'],
        'inference_backend': BACKEND, 'model': MODEL,
        'episode': {
            'task_dir': row['task_dir'], 'episode_index': row['episode_index'],
            'length': row['episode_length'],
        },
        'task_instruction': row['task_instruction'],
        'selected_frame_index': row['selected_frame_index'],
        'source_video': row['source_video'], 'image': image.name,
        'image_sha256': sha256(image),
    }
    if row['kind'] == 'ecot':
        base.update({
            'ecot_sample_type': row.get('ecot_sample_type', ECOT_PRIMARY_SAMPLE_TYPE),
        })
    if row['kind'] == 'grounding':
        content = [{'type': 'image_url', 'image_url': {'url': image_url(image)}}, {'type': 'text', 'text': grounding_prompt(row['task_instruction'])}]
        payload = {
            'model': MODEL, 'messages': [{'role': 'user', 'content': content}], 'temperature': 0, 'max_tokens': 1024,
            'response_format': {'type': 'json_schema', 'json_schema': grounding_schema()},
        }
        def grounding_once():
            response = post(endpoint, payload)
            raw = get_content(response)
            inv = json.loads(raw)
            question, answer, actual_type, audit = grounding_qa(row, inv)
            return raw, question, answer, actual_type, audit

        grounding_result, attempts = annotation_call_with_retries(
            'grounding_annotation', grounding_once, max_attempts,
        )
        raw, question, answer, actual_type, audit = grounding_result
        return {**base, 'question_type': actual_type, 'question': question, 'answer': answer, 'model_inventory': audit, 'raw_model_output': raw, 'attempts': attempts, 'visibility_policy': 'current_image_only', 'elapsed_s': round(time.time()-started, 3)}

    if ecot_context_dir is None:
        raise RuntimeError('ecot_context_dir_required')
    annotation_context = extract_ecot_context_frames(row, ecot_context_dir)
    progress_result, progress_attempts = annotation_call_with_retries(
        'ecot_progress_annotation',
        lambda: ecot_progress_request(endpoint, row, annotation_context),
        max_attempts,
    )
    progress_summary, progress_user_prompt, progress_raw = progress_result
    annotation_context['target_frame_progress_summary'] = progress_summary
    ecot_result, ecot_attempts = annotation_call_with_retries(
        'structured_ecot_annotation',
        lambda: ecot_request(endpoint, row, annotation_context),
        max_attempts,
    )
    structured_ecot, rendered_ecot, ecot_user_text, ecot_raw = ecot_result
    episode_preview = row.get('episode_preview', f'{row["id"]}.mp4')
    annotation_context.update({
        'available_views': ['observation.images.cam_left_wrist'],
        'whole_episode_video': f'episode_videos/{episode_preview}',
        'progress_system_prompt': ECOT_PROGRESS_SYSTEM_PROMPT,
        'progress_user_prompt': progress_user_prompt,
        'progress_raw_model_output': progress_raw,
        'progress_attempts': progress_attempts,
        'ecot_system_prompt': ECOT_SYSTEM_PROMPT,
        'ecot_user_prompt': ecot_user_text,
        'ecot_attempts': ecot_attempts,
        'privileged_context_excluded_from_training_inputs': True,
    })
    current = {
        'method': 'structured_three_part_ecot',
        'question': 'Generate a concise three-part ECoT with task-level subtask semantics for the current observation.',
        'answer': rendered_ecot,
        'structured_ecot': structured_ecot,
        'raw_model_output': ecot_raw,
        'input_media': {'type': 'image', 'file': image.name, 'frame_index': row['selected_frame_index']},
        'annotation_time_context_passed_to_teacher': True,
        'future_media_passed_to_training_model': False,
    }
    return {
        **base, 'question_type': 'structured_ecot', 'question': current['question'],
        'answer': rendered_ecot,
        'ecot_methods': {'structured_three_part_ecot': current},
        'annotation_context': annotation_context,
        'gripper_segmentation': row['gripper_segmentation'],
        'visibility_policy': 'training_input_is_task_plus_current_image_only; past_and_future_one_hz_frames_and_whole_episode_progress_summary_are_annotation_teacher_only',
        'elapsed_s': round(time.time()-started, 3),
    }


def draw_grounding_annotations(record: dict, image_dir: Path, annotated_dir: Path) -> None:
    image = cv2.imread(str(image_dir / record['image']))
    if image is None:
        raise RuntimeError(f'annotation_image_decode_failed:{record["id"]}')
    height, width = image.shape[:2]
    answer = json.loads(record['answer'])
    answer_name = str(answer.get('object') or '').lower()
    answer_bbox = answer.get('bbox_xyxy')
    palette = [(255, 170, 0), (255, 0, 255), (0, 200, 255), (180, 255, 0)]
    for i, obj in enumerate(record['model_inventory']['objects']):
        x1, y1, x2, y2 = obj['bbox_xyxy']; cx, cy = obj['center_xy']
        p1, p2 = (round(x1 * width), round(y1 * height)), (round(x2 * width), round(y2 * height))
        center = (round(cx * width), round(cy * height))
        target = (
            isinstance(answer_bbox, list) and len(answer_bbox) == 4 and
            all(abs(float(left) - float(right)) <= 1e-6 for left, right in zip(obj['bbox_xyxy'], answer_bbox))
        ) or (
            answer_bbox is None and str(obj['name']).lower() == answer_name
        )
        color = (20, 220, 20) if target else palette[i % len(palette)]
        thickness = 5 if target else 3
        cv2.rectangle(image, p1, p2, color, thickness)
        cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 22, thickness)
        label = f"{obj['name']} c=({cx:.3f},{cy:.3f})"
        font_scale = 0.7
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        tx, ty = max(0, min(p1[0], width - tw - 4)), max(th + 4, p1[1] - 5)
        cv2.rectangle(image, (tx, ty - th - 4), (tx + tw + 4, ty + baseline), (0, 0, 0), -1)
        cv2.putText(image, label, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    out = annotated_dir / record['image']
    if not cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f'annotation_image_write_failed:{out}')
    record['annotated_image'] = out.name
    record['annotation_legend'] = 'All eligible object boxes and recomputed centers; green marks the answer object. Coordinates use normalized upper-left origin, x-right, y-down.'


def draw_sta_bbox_annotation(audit: dict, annotated_dir: Path) -> None:
    """Draw the STA next-active-object box on the learner query image."""
    bbox = audit.get('object_bbox_xyxy')
    if not audit.get('valid_contact') or not audit.get('grounding_episode') or bbox is None:
        return
    image = cv2.imread(audit['query_image_path'])
    if image is None:
        raise RuntimeError(f'sta_bbox_image_decode_failed:{audit["candidate_id"]}')
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    p1 = (round(x1 * width), round(y1 * height))
    p2 = (round(x2 * width), round(y2 * height))
    color = (20, 220, 20)
    cv2.rectangle(image, p1, p2, color, 5)
    noun = str(audit['contact']['noun'])
    label = f'next contact: {noun}'
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    tx, ty = max(0, min(p1[0], width - tw - 6)), max(th + 6, p1[1] - 6)
    cv2.rectangle(image, (tx, ty - th - 5), (tx + tw + 6, ty + baseline), (0, 0, 0), -1)
    cv2.putText(image, label, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    out = annotated_dir / f'{audit["candidate_id"]}.jpg'
    if not cv2.imwrite(str(out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f'sta_bbox_image_write_failed:{out}')
    audit['annotated_query_image'] = out.name


CPA_POINT_COLORS = (
    (20, 20, 240), (20, 180, 255), (220, 40, 180), (40, 210, 40), (240, 160, 20),
)


def draw_cpa_point(
    image,
    point: list[float],
    label: str,
    color: tuple[int, int, int],
    role: str | None = None,
) -> None:
    height, width = image.shape[:2]
    center = (
        round(float(point[0]) * max(1, width - 1)),
        round(float(point[1]) * max(1, height - 1)),
    )
    if role == 'hand':
        cv2.rectangle(
            image, (center[0] - 11, center[1] - 11),
            (center[0] + 11, center[1] + 11), (255, 255, 255), 5, cv2.LINE_AA,
        )
        cv2.rectangle(
            image, (center[0] - 8, center[1] - 8),
            (center[0] + 8, center[1] + 8), color, -1, cv2.LINE_AA,
        )
        label_y = max(20, center[1] - 14)
    else:
        cv2.circle(image, center, 10, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.circle(image, center, 8, color, -1, cv2.LINE_AA)
        label_y = min(height - 5, center[1] + 28) if role == 'object' else max(20, center[1] - 10)
    cv2.putText(
        image, label, (min(width - 45, center[0] + 11), label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
    )


def draw_cpa_intermediates(audit: dict, intermediate_dir: Path) -> None:
    contact = cv2.imread(audit['contact_frame_image_path'])
    crop = cv2.imread(audit['interaction_crop_image_path'])
    if contact is None or crop is None:
        raise RuntimeError(f'cpa_intermediate_image_decode_failed:{audit["candidate_id"]}')
    height, width = contact.shape[:2]
    x1, y1, x2, y2 = audit['interaction_bbox_xyxy']
    cv2.rectangle(
        contact,
        (round(x1 * width), round(y1 * height)),
        (round(x2 * width), round(y2 * height)),
        (40, 220, 40), 5,
    )
    for pair in audit['contact_point_pairs_full_xy']:
        index = int(pair['pair_index'])
        color = CPA_POINT_COLORS[index % len(CPA_POINT_COLORS)]
        draw_cpa_point(contact, pair['hand_point_xy'], f'H{index}', color, 'hand')
        draw_cpa_point(contact, pair['object_point_xy'], f'O{index}', color, 'object')
    for pair in audit['contact_point_pairs_crop_xy']:
        index = int(pair['pair_index'])
        color = CPA_POINT_COLORS[index % len(CPA_POINT_COLORS)]
        draw_cpa_point(crop, pair['hand_point_xy'], f'H{index}', color, 'hand')
        draw_cpa_point(crop, pair['object_point_xy'], f'O{index}', color, 'object')
    contact_out = intermediate_dir/f'{audit["candidate_id"]}_contact.jpg'
    crop_out = intermediate_dir/f'{audit["candidate_id"]}_crop.jpg'
    if not cv2.imwrite(str(contact_out), contact) or not cv2.imwrite(str(crop_out), crop):
        raise RuntimeError(f'cpa_intermediate_image_write_failed:{audit["candidate_id"]}')
    audit['annotated_contact_frame'] = contact_out.name
    audit['annotated_interaction_crop'] = crop_out.name


def write_cpa_tracked_overlay(
    image_path: Path,
    contact_pairs: list[dict],
    output_path: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f'cpa_tracking_review_image_decode_failed:{image_path}')
    for pair in contact_pairs:
        index = int(pair['pair_index'])
        color = CPA_POINT_COLORS[index % len(CPA_POINT_COLORS)]
        draw_cpa_point(image, pair['hand_point_xy'], f'H{index}', color, 'hand')
        draw_cpa_point(image, pair['object_point_xy'], f'O{index}', color, 'object')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 96]):
        raise RuntimeError(f'cpa_tracking_review_overlay_write_failed:{output_path}')


def cpa_tracking_review_prompt(audit: dict, tracked: dict) -> str:
    compact_pairs = [
        {
            'pair_index': pair['pair_index'],
            'H_i': pair['hand_point_xy'],
            'O_i': pair['object_point_xy'],
        }
        for pair in tracked['contact_pairs']
    ]
    return f'''Global task: {audit["task_instruction"]}
Tracked observation: source frame {int(tracked["query_frame_index"])} before accepted contact frame {int(audit["candidate_frame"])}.
End-effector type: {audit["end_effector_type"]}.
{cpa_contact_agent_instruction(audit)}
Contacted object category: {audit["contact"]["noun"]}.
Tracked normalized points: {json.dumps(compact_pairs, ensure_ascii=False)}.
For every pair_index, independently check the labeled H_i pixel against the explicitly specified contact agent and the labeled O_i pixel against the contacted object. Do not repair coordinates. A false judgment removes only that labeled point from this observation.'''


def cpa_pair_distance_pixels(pair: dict, source_resolution: list[int]) -> float:
    if (
        not isinstance(source_resolution, list) or len(source_resolution) != 2 or
        not all(isinstance(value, (int, float)) and value > 0 for value in source_resolution)
    ):
        raise ValueError(f'cpa_source_resolution_invalid:{source_resolution}')
    width, height = float(source_resolution[0]), float(source_resolution[1])
    hand = pair.get('hand_point_xy')
    obj = pair.get('object_point_xy')
    if (
        not isinstance(hand, list) or not isinstance(obj, list) or
        len(hand) != 2 or len(obj) != 2
    ):
        raise ValueError('cpa_pair_coordinates_invalid')
    dx = (float(hand[0]) - float(obj[0])) * max(1.0, width - 1.0)
    dy = (float(hand[1]) - float(obj[1])) * max(1.0, height - 1.0)
    return math.hypot(dx, dy)


def cpa_pair_separation_review(audit: dict, tracked_pair: dict) -> dict:
    pair_index = int(tracked_pair['pair_index'])
    contact_pairs = {
        int(pair['pair_index']): pair
        for pair in audit.get('contact_point_pairs_full_xy', [])
    }
    if pair_index not in contact_pairs:
        raise ValueError(f'cpa_contact_pair_missing:{pair_index}')
    source_resolution = audit.get('source_resolution')
    contact_distance = cpa_pair_distance_pixels(
        contact_pairs[pair_index], source_resolution,
    )
    tracked_distance = cpa_pair_distance_pixels(tracked_pair, source_resolution)
    if contact_distance <= 1e-9:
        growth_ratio = None
        sufficient = False
        reason = 'contact_pair_distance_nonpositive'
    else:
        growth_ratio = tracked_distance / contact_distance - 1.0
        sufficient = (
            growth_ratio > CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO and
            not math.isclose(
                growth_ratio,
                CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        reason = (
            'distance_growth_above_100pct'
            if sufficient else 'distance_growth_not_above_100pct'
        )
    return {
        'pair_index': pair_index,
        'contact_distance_pixels': round(contact_distance, 6),
        'tracked_distance_pixels': round(tracked_distance, 6),
        'distance_growth_ratio': (
            round(growth_ratio, 6) if growth_ratio is not None else None
        ),
        'minimum_distance_growth_ratio': CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO,
        'distance_growth_sufficient': sufficient,
        'reason': reason,
    }


def cpa_tracking_review_compatible(
    review: dict,
    audit: dict,
    tracked: dict,
) -> bool:
    return bool(
        review.get('tracking_review_version') == CPA_TRACKING_REVIEW_VERSION and
        review.get('tracking_review_model') == MODEL and
        review.get('candidate_id') == audit.get('candidate_id') and
        review.get('query_frame_index') == tracked.get('query_frame_index') and
        review.get('tracked_contact_pairs') == tracked.get('contact_pairs') and
        review.get('contact_frame_contact_pairs') == audit.get('contact_point_pairs_full_xy') and
        review.get('source_resolution') == audit.get('source_resolution') and
        review.get('pair_separation_review_version') == CPA_PAIR_SEPARATION_REVIEW_VERSION and
        review.get('pair_minimum_distance_growth_ratio') == CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO and
        isinstance(review.get('pair_distance_checks'), list) and
        len(review.get('pair_distance_checks', [])) == len(tracked.get('contact_pairs', [])) and
        isinstance(review.get('validated_points'), list) and
        review.get('raw_model_output')
    )


def review_cpa_tracking_observation(
    endpoint: str,
    audit: dict,
    source_observation: dict,
    tracked: dict,
    overlay_dir: Path,
    max_attempts: int,
) -> dict:
    observation_id = tracked['observation_id']
    overlay_path = overlay_dir/f'{observation_id}_before_review.jpg'
    write_cpa_tracked_overlay(
        Path(source_observation['query_image_path']), tracked['contact_pairs'], overlay_path,
    )
    pair_indices = [int(pair['pair_index']) for pair in tracked['contact_pairs']]
    prompt = cpa_tracking_review_prompt(audit, tracked)
    review_result, attempts = annotation_call_with_retries(
        'cpa_tracking_semantic_review',
        lambda: parse_cpa_tracking_review(post(
            endpoint,
            cpa_tracking_review_payload(
                Path(source_observation['query_image_path']), overlay_path,
                prompt, pair_indices,
            ),
        ), pair_indices),
        max_attempts,
    )
    checks = {item['pair_index']: item for item in review_result['checks']}
    validated_points = []
    validated_pairs = []
    removed_point_ids = []
    pair_distance_checks = []
    for pair in tracked['contact_pairs']:
        index = int(pair['pair_index'])
        check = checks[index]
        distance_check = cpa_pair_separation_review(audit, pair)
        pair_distance_checks.append(distance_check)
        distance_valid = bool(distance_check['distance_growth_sufficient'])
        kept_pair = {
            'pair_index': index,
            'distance_growth_valid': distance_valid,
            'contact_distance_pixels': distance_check['contact_distance_pixels'],
            'tracked_distance_pixels': distance_check['tracked_distance_pixels'],
            'distance_growth_ratio': distance_check['distance_growth_ratio'],
        }
        for role, valid_key, prefix in (
            ('hand', 'hand_on_end_effector', 'H'),
            ('object', 'object_on_contacted_object', 'O'),
        ):
            point_id = f'{prefix}{index}'
            semantic_valid = bool(check[valid_key])
            valid = semantic_valid and distance_valid
            point = pair[f'{role}_point_xy'] if valid else None
            kept_pair[f'{role}_point_xy'] = point
            kept_pair[f'{role}_valid'] = valid
            kept_pair[f'{role}_semantic_valid'] = semantic_valid
            kept_pair[f'{role}_tracker_visible'] = bool(pair[f'{role}_visible'])
            if valid:
                validated_points.append({
                    'point_id': point_id,
                    'pair_index': index,
                    'role': role,
                    'xy': point,
                    'tracker_visible': bool(pair[f'{role}_visible']),
                })
            else:
                removed_point_ids.append(point_id)
        validated_pairs.append(kept_pair)
    raw = review_result['raw_model_output']
    return {
        'candidate_id': audit['candidate_id'],
        'observation_id': observation_id,
        'query_frame_index': int(tracked['query_frame_index']),
        'tracked_contact_pairs': tracked['contact_pairs'],
        'contact_frame_contact_pairs': audit['contact_point_pairs_full_xy'],
        'source_resolution': audit['source_resolution'],
        'validated_pairs': validated_pairs,
        'validated_points': validated_points,
        'removed_point_ids': removed_point_ids,
        'pair_distance_checks': pair_distance_checks,
        'pair_separation_review_version': CPA_PAIR_SEPARATION_REVIEW_VERSION,
        'pair_minimum_distance_growth_ratio': CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO,
        'review_reason': review_result['reason'],
        'tracking_review_version': CPA_TRACKING_REVIEW_VERSION,
        'tracking_review_model': MODEL,
        'tracking_review_overlay': overlay_path.name,
        'tracking_review_overlay_path': str(overlay_path),
        'tracking_review_prompt': prompt,
        'raw_model_output': raw,
        'raw_model_output_sha256': hashlib.sha256(raw.encode()).hexdigest(),
        'attempts': attempts,
    }


def review_cpa_tracking_results(
    endpoint: str,
    audits: list[dict],
    tracking_by_candidate: dict[str, dict],
    overlay_dir: Path,
    cache_dir: Path,
    request_workers: int,
    max_attempts: int,
    resume_existing: bool,
) -> tuple[list[dict], list[dict]]:
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs_spec = []
    cached = {}
    for audit in audits:
        tracking = tracking_by_candidate.get(audit['candidate_id'])
        if not tracking:
            continue
        source_by_id = {item['observation_id']: item for item in audit['observations']}
        for tracked in tracking['tracked_observations']:
            observation_id = tracked['observation_id']
            cache_path = cache_dir/f'{observation_id}.json'
            if resume_existing and cache_path.is_file():
                item = json.loads(cache_path.read_text(encoding='utf-8'))
                if cpa_tracking_review_compatible(item, audit, tracked):
                    cached[observation_id] = item
                    continue
            jobs_spec.append((audit, source_by_id[observation_id], tracked, cache_path))
    reviews = list(cached.values())
    errors = []
    total = len(reviews) + len(jobs_spec)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=request_workers,
    ) as executor:
        jobs = {
            executor.submit(
                review_cpa_tracking_observation,
                endpoint, audit, source_observation, tracked,
                overlay_dir, max_attempts,
            ): (tracked['observation_id'], cache_path)
            for audit, source_observation, tracked, cache_path in jobs_spec
        }
        for future in concurrent.futures.as_completed(jobs):
            observation_id, cache_path = jobs[future]
            try:
                review = future.result()
                reviews.append(review)
                atomic_json(cache_path, review)
            except Exception as error:
                errors.append({
                    'id': observation_id,
                    'kind': 'cpa_tracking_semantic_review',
                    'error': f'{type(error).__name__}:{error}',
                })
            print(
                f'cpa_tracking_review_progress completed={len(reviews) + len(errors)}/{total} '
                f'accepted={len(reviews)} errors={len(errors)}',
                flush=True,
            )
    return sorted(reviews, key=lambda item: item['observation_id']), errors


def build_cpa_records(
    audits: list[dict],
    tracking_by_candidate: dict[str, dict],
    reviews_by_observation: dict[str, dict],
    annotated_dir: Path,
    multiple_choice_by_observation: dict[str, dict] | None = None,
) -> list[dict]:
    multiple_choice_by_observation = multiple_choice_by_observation or {}
    records = []
    for audit in sorted(audits, key=lambda item: item['candidate_id']):
        tracking = tracking_by_candidate.get(audit['candidate_id'])
        if tracking is None:
            continue
        audit['tracking'] = tracking
        observations_by_id = {
            item['observation_id']: item for item in tracking['tracked_observations']
        }
        source_observations = {item['observation_id']: item for item in audit['observations']}
        for observation_id, tracked in observations_by_id.items():
            review = reviews_by_observation.get(observation_id)
            if review is None:
                continue
            source_observation = source_observations[observation_id]
            image = cv2.imread(source_observation['query_image_path'])
            if image is None:
                raise RuntimeError(f'cpa_observation_image_decode_failed:{observation_id}')
            for point in review['validated_points']:
                index = int(point['pair_index'])
                draw_cpa_point(
                    image, point['xy'], point['point_id'],
                    CPA_POINT_COLORS[index % len(CPA_POINT_COLORS)], point['role'],
                )
            annotated_path = annotated_dir/f'{observation_id}.jpg'
            if not cv2.imwrite(str(annotated_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f'cpa_observation_image_write_failed:{annotated_path}')
            question = (
                'Where are the reviewed tracked H_i hand/end-effector points and '
                'O_i contacted-object points for the next intentional contact?'
            )
            points = review['validated_points']
            multiple_choice_questions = build_cpa_multiple_choice_vqa_records(
                audit,
                source_observation,
                multiple_choice_by_observation.get(observation_id),
            )
            records.append({
                'id': f'{observation_id}_contact_points',
                'kind': 'cpa',
                'dataset': audit.get('dataset', 'Robot'),
                'dataset_name': audit.get('dataset_name', audit.get('dataset', 'Robot')),
                'dataset_family': audit.get('dataset_family', audit.get('actor', 'robot')),
                'actor': audit.get('actor', 'robot'),
                'candidate_id': audit['candidate_id'],
                'observation_id': observation_id,
                'task_instruction': audit['task_instruction'],
                'query_frame_index': tracked['query_frame_index'],
                'contact_frame_index': audit['candidate_frame'],
                'time_to_contact_seconds': source_observation['time_to_contact_seconds'],
                'question_type': 'contact_point_anticipation',
                'question': question,
                'answer': json.dumps(points, ensure_ascii=False),
                'answer_value': points,
                'multiple_choice_questions': multiple_choice_questions,
                'multiple_choice_question_count': len(multiple_choice_questions),
                'multiple_choice_question_version': CPA_MULTIPLE_CHOICE_VQA_VERSION,
                'answer_pairs': review['validated_pairs'],
                'removed_point_ids': review['removed_point_ids'],
                'point_visibility': {
                    point['point_id']: point['tracker_visible'] for point in points
                },
                'point_count': len(points),
                'image': source_observation['query_image'],
                'image_sha256': sha256(Path(source_observation['query_image_path'])),
                'annotated_image': annotated_path.name,
                'contact_points_source': (
                    'contact_frame_sam2_masks_radius_r_disk_prototypes_then_cotracker_then_distance_and_lite_frame_review'
                    if audit.get('point_method') == 'sam2_disk_prototype' else
                    'contact_frame_lite_prompts_sam2_masks_closest_pair_then_cotracker_then_distance_and_lite_frame_review'
                    if audit.get('point_method') == 'sam2' else
                    'contact_frame_paired_vlm_then_cotracker_then_distance_and_lite_frame_review'
                ),
                'tracking_review': review,
                'teacher_evidence': {
                    'contact_frame': audit['candidate_frame'],
                    'point_method': audit.get('point_method', 'seed_vlm'),
                    'interaction_bbox_xyxy': audit['interaction_bbox_xyxy'],
                    'contact_point_pairs_full_xy': audit['contact_point_pairs_full_xy'],
                    'tracker': tracking['tracker'],
                    'future_media_passed_to_learner': False,
                },
            })
    return sorted(records, key=lambda item: item['id'])


def resolved_cpa_contact_agent_role(audit: dict) -> str:
    """Resolve the Pro role, with a compatibility fallback for legacy robot runs."""
    contact_agent_role = audit.get('contact_agent_role')
    if contact_agent_role not in CPA_AGENT_PARTICIPANT_QUESTION_TYPES:
        end_effector = str(audit.get('end_effector_type', '')).lower()
        if 'tool' in end_effector:
            contact_agent_role = 'held_tool'
        elif audit.get('actor') == 'human':
            contact_agent_role = 'human_hand'
        else:
            contact_agent_role = 'robot_gripper'
    return contact_agent_role


def cpa_is_robot_data(record: dict) -> bool:
    """Return whether dataset metadata identifies a record as robot data."""
    dataset_family = str(record.get('dataset_family') or '').strip().lower()
    if dataset_family in {'robot', 'human'}:
        return dataset_family == 'robot'
    return str(record.get('actor') or '').strip().lower() == 'robot'


def cpa_participant_vqa_role(
    audit: dict,
    tracked_role: str,
) -> tuple[str, str, str, str]:
    """Return entity label, question type, semantic role, and contact-agent role."""
    contact_agent_role = resolved_cpa_contact_agent_role(audit)
    if tracked_role == 'object':
        return (
            'next contact object', 'next_contact_object_point',
            'contacted_object', contact_agent_role,
        )
    if tracked_role != 'hand':
        raise ValueError(f'unsupported_cpa_point_role:{tracked_role}')
    entity_label, question_type = CPA_AGENT_PARTICIPANT_QUESTION_TYPES[
        contact_agent_role
    ]
    return entity_label, question_type, 'contact_agent', contact_agent_role


def build_cpa_multiple_choice_vqa_records(
    audit: dict,
    source_observation: dict,
    segmentation: dict | None,
) -> list[dict]:
    """Build one observation-local SAM2 multiselect row per available participant."""
    if not segmentation:
        return []
    query_frame = int(source_observation['query_frame_index'])
    contact_frame = int(audit['candidate_frame'])
    if query_frame >= contact_frame:
        raise ValueError(
            f'cpa_choice_observation_not_precontact:{audit["candidate_id"]}:'
            f'{query_frame}:{contact_frame}'
        )
    image_path = Path(source_observation['query_image_path'])
    source_episode_identity = str(
        audit.get('source_video') or
        f'{audit.get("dataset", "dataset")}:{audit.get("row_id", audit["candidate_id"])}'
    )
    source_episode_uid = hashlib.sha256(
        source_episode_identity.encode('utf-8')
    ).hexdigest()[:16]
    records = []
    for participant in segmentation.get('participants', []):
        tracked_role = str(participant['tracked_role'])
        # Robot end-effector points remain in tracking and review artifacts, but
        # are intentionally excluded from learner-facing CPA supervision.
        if cpa_is_robot_data(audit) and tracked_role == 'hand':
            continue
        entity_label, base_question_type, semantic_role, contact_agent_role = (
            cpa_participant_vqa_role(audit, tracked_role)
        )
        question_type = f'{base_question_type}_multiple_choice'
        choices = participant.get('choices') or []
        answer_labels = participant.get('answer_labels') or []
        choice_labels = [str(choice['label']) for choice in choices]
        if (
            not choices or not answer_labels or
            not set(answer_labels).issubset(choice_labels)
        ):
            continue
        options = ', '.join(choice_labels)
        required_selection_count = len(answer_labels)
        choice_word = 'choice' if required_selection_count == 1 else 'choices'
        question = (
            f'Which marked locations on the {entity_label} correspond to the verified '
            'tracked contact targets for the upcoming intentional contact? '
            f'Select exactly {required_selection_count} supported {choice_word} '
            f'from {options}.'
        )
        learner_image = str(participant['learner_choice_image'])
        records.append({
            'id': (
                f'{source_episode_uid}_{source_observation["observation_id"]}_'
                f'{question_type}'
            ),
            'kind': 'cpa',
            'dataset': audit.get('dataset', 'Robot'),
            'dataset_name': audit.get('dataset_name', audit.get('dataset', 'Robot')),
            'dataset_family': audit.get('dataset_family', audit.get('actor', 'robot')),
            'actor': audit.get('actor', 'robot'),
            'candidate_id': audit['candidate_id'],
            'observation_id': source_observation['observation_id'],
            'source_episode_uid': source_episode_uid,
            'task_instruction': audit['task_instruction'],
            'query_frame_index': query_frame,
            'contact_frame_index': contact_frame,
            'time_to_contact_seconds': source_observation['time_to_contact_seconds'],
            'question_type': question_type,
            'question': question,
            'choices': {
                str(choice['label']): choice['xy'] for choice in choices
            },
            'answer': json.dumps(answer_labels, ensure_ascii=False),
            'answer_value': answer_labels,
            'answer_mode': 'multi_select',
            'required_selection_count': required_selection_count,
            'tracked_point_role': tracked_role,
            'semantic_participant_role': semantic_role,
            'target_entity': entity_label,
            'contact_agent_role': contact_agent_role,
            'contact_agent_name': audit.get('contact_agent_name'),
            'contact_object_participant': audit.get(
                'contact_object_participant', audit.get('contact', {}).get('noun'),
            ),
            'source_image': source_observation['query_image'],
            'source_image_path': str(image_path),
            'image': learner_image,
            'image_path': str(Path(segmentation['artifact_directory'])/learner_image),
            'multiple_choice_question_version': CPA_MULTIPLE_CHOICE_VQA_VERSION,
            'learner_inputs': [
                'task_instruction', 'marked_precontact_observation_image', 'question',
            ],
            'visibility_policy': (
                'learner_sees_only_the_precontact_observation_with_neutral_choice_'
                'markers_plus_task_and_question'
            ),
            'teacher_evidence': {
                'contact_frame': contact_frame,
                'future_media_passed_to_learner': False,
                'observation_local_sam2_mask_image': participant['mask_image'],
                'observation_local_sam2_mask_and_candidates_image': participant[
                    'mask_and_candidates_image'
                ],
                'sam_prompt_consistent': participant['sam_prompt_consistent'],
                'choice_metadata': choices,
                'positive_point_ids': participant['positive_point_ids'],
                'negative_sampling': participant['negative_sampling'],
            },
        })
    return sorted(records, key=lambda item: item['id'])


def flatten_cpa_multiple_choice_vqa_records(records: list[dict]) -> list[dict]:
    """Flatten observation-level CPA rows into participant multiselect rows."""
    flattened = [
        question
        for record in records
        for question in record.get('multiple_choice_questions', [])
    ]
    return sorted(flattened, key=lambda item: item['id'])


def active_cpa_point_pair_version() -> str:
    if CPA_POINT_METHOD == 'seed_vlm':
        return CPA_POINT_PAIR_VERSION
    if CPA_POINT_METHOD == 'sam2':
        return CPA_SAM2_POINT_PAIR_VERSION
    raise RuntimeError(f'unsupported_cpa_point_method:{CPA_POINT_METHOD}')


def merge_cpa_sam2_result(annotation: dict, segmentation: dict) -> dict:
    if annotation.get('candidate_id') != segmentation.get('candidate_id'):
        raise ValueError('cpa_sam2_candidate_mismatch')
    crop_pair = segmentation.get('closest_pair_crop_xy')
    if not isinstance(crop_pair, dict):
        raise ValueError('cpa_sam2_closest_pair_missing')
    normalized_pair = {
        'pair_index': 0,
        'hand_point_xy': normalize_vlm_coordinates(crop_pair.get('hand_point_xy'), 2),
        'object_point_xy': normalize_vlm_coordinates(crop_pair.get('object_point_xy'), 2),
    }
    if normalized_pair['hand_point_xy'] == normalized_pair['object_point_xy']:
        raise ValueError('cpa_sam2_closest_pair_identical')
    x1, y1, x2, y2 = annotation['interaction_bbox_xyxy']
    full_pair = {'pair_index': 0}
    for role in ('hand', 'object'):
        point = normalized_pair[f'{role}_point_xy']
        full_pair[f'{role}_point_xy'] = [
            round(x1 + point[0] * (x2 - x1), 6),
            round(y1 + point[1] * (y2 - y1), 6),
        ]
    return {
        **annotation,
        'contact_point_pairs_crop_xy': [normalized_pair],
        'contact_point_pairs_full_xy': [full_pair],
        'contact_point_pair_count': 1,
        'point_pair_version': CPA_SAM2_POINT_PAIR_VERSION,
        'point_method': 'sam2',
        'sam2_segmentation': segmentation,
        'pair_candidate_overlay': None,
    }


def run_cpa_sam2(
    annotations: list[dict],
    out: Path,
    resume_existing: bool,
) -> tuple[list[dict], list[dict]]:
    prompt_path = out/'cpa_sam2_prompts.jsonl'
    output_path = out/'cpa_sam2_results.jsonl'
    errors_path = out/'cpa_sam2_errors.jsonl'
    artifacts = out/'cpa_sam2'
    log_path = out/'cpa_sam2.log'
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(prompt_path, annotations)
    missing = []
    if not Path(CPA_SAM2_PYTHON).is_file():
        missing.append(f'python:{CPA_SAM2_PYTHON}')
    if CPA_SAM2_BACKEND == 'transformers':
        for name in ('config.json', 'model.safetensors', 'preprocessor_config.json'):
            if not (CPA_SAM2_MODEL_PATH/name).is_file():
                missing.append(f'model_file:{CPA_SAM2_MODEL_PATH/name}')
    elif CPA_SAM2_BACKEND == 'official':
        if not (CPA_SAM2_ROOT/'sam2'/'build_sam.py').is_file():
            missing.append(f'official_code:{CPA_SAM2_ROOT}')
        if not CPA_SAM2_CHECKPOINT.is_file():
            missing.append(f'checkpoint:{CPA_SAM2_CHECKPOINT}')
    else:
        missing.append(f'backend:{CPA_SAM2_BACKEND}')
    if missing:
        message = 'sam2_preflight_missing:' + ','.join(missing)
        log_path.write_text(message + '\n', encoding='utf-8')
        errors = [{'id': 'cpa_sam2_preflight', 'kind': 'cpa_sam2', 'error': message}]
        atomic_jsonl(errors_path, errors)
        return [], errors
    command = [
        CPA_SAM2_PYTHON,
        str(Path(__file__).resolve().parents[1]/'scripts'/'segment_cpa_sam2.py'),
        '--input', str(prompt_path), '--output', str(output_path),
        '--errors', str(errors_path), '--artifacts', str(artifacts),
        '--backend', CPA_SAM2_BACKEND,
        '--model-path', str(CPA_SAM2_MODEL_PATH),
        '--sam2-root', str(CPA_SAM2_ROOT),
        '--checkpoint', str(CPA_SAM2_CHECKPOINT),
        '--model-config', CPA_SAM2_MODEL_CONFIG,
        '--device', CPA_SAM2_DEVICE,
    ]
    if resume_existing:
        command.append('--resume-existing')
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        result.stdout + ('\nSTDERR:\n' + result.stderr if result.stderr else ''),
        encoding='utf-8',
    )
    segmentations = [
        json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if output_path.is_file() else []
    errors = [
        json.loads(line) for line in errors_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if errors_path.is_file() else []
    if result.returncode != 0 and not errors:
        errors.append({
            'id': 'cpa_sam2_process', 'kind': 'cpa_sam2',
            'error': f'sam2_exit_{result.returncode}:{result.stderr[-2000:]}',
        })
    by_candidate = {item['candidate_id']: item for item in segmentations}
    merged = []
    for annotation in annotations:
        segmentation = by_candidate.get(annotation['candidate_id'])
        if segmentation is None:
            continue
        try:
            merged.append(merge_cpa_sam2_result(annotation, segmentation))
        except Exception as error:
            errors.append({
                'id': annotation['candidate_id'], 'kind': 'cpa_sam2_merge',
                'error': f'{type(error).__name__}:{error}',
            })
    return sorted(merged, key=lambda item: item['candidate_id']), errors


def build_cpa_observation_choice_prompts(
    audits: list[dict],
    tracking_by_candidate: dict[str, dict],
    reviews_by_observation: dict[str, dict],
) -> list[dict]:
    """Build SAM2 jobs directly on existing learner observation images."""
    prompts = []
    for audit in sorted(audits, key=lambda item: item['candidate_id']):
        tracking = tracking_by_candidate.get(audit['candidate_id'])
        if tracking is None:
            continue
        source_observations = {
            item['observation_id']: item for item in audit['observations']
        }
        for tracked in tracking['tracked_observations']:
            observation_id = tracked['observation_id']
            review = reviews_by_observation.get(observation_id)
            source = source_observations.get(observation_id)
            if review is None or source is None:
                continue
            points_by_role = {'hand': [], 'object': []}
            for point in review.get('validated_points', []):
                role = str(point.get('role'))
                if role not in points_by_role:
                    continue
                points_by_role[role].append({
                    'point_id': str(point['point_id']),
                    'pair_index': int(point['pair_index']),
                    'xy': [round(float(value), 6) for value in point['xy']],
                })
            participants = []
            participant_roles = (
                ('object',) if cpa_is_robot_data(audit) else ('hand', 'object')
            )
            for role in participant_roles:
                positive_points = sorted(
                    points_by_role[role], key=lambda item: item['pair_index'],
                )
                if not positive_points:
                    continue
                opposite = 'object' if role == 'hand' else 'hand'
                participants.append({
                    'tracked_role': role,
                    'positive_points': positive_points,
                    'opposite_points': sorted(
                        points_by_role[opposite],
                        key=lambda item: item['pair_index'],
                    ),
                })
            if not participants:
                continue
            prompts.append({
                'candidate_id': audit['candidate_id'],
                'observation_id': observation_id,
                'query_frame_index': int(tracked['query_frame_index']),
                'contact_frame_index': int(audit['candidate_frame']),
                'image_path': source['query_image_path'],
                'image_sha256': sha256(Path(source['query_image_path'])),
                'participants': participants,
            })
    return sorted(prompts, key=lambda item: item['observation_id'])


def run_cpa_observation_choice_sam2(
    prompts: list[dict],
    out: Path,
    resume_existing: bool,
    negative_a_at_720p: float | None = None,
    negative_drop_rate: float | None = None,
    batch_size: int = CPA_OBSERVATION_SAM2_BATCH_SIZE,
) -> tuple[list[dict], list[dict]]:
    """Segment each learner frame locally and construct weighted choices."""
    negative_a_at_720p = validate_cpa_negative_a(
        CPA_NEGATIVE_A_AT_720P
        if negative_a_at_720p is None else negative_a_at_720p
    )
    negative_drop_rate = validate_cpa_negative_drop_rate(
        CPA_NEGATIVE_DROP_RATE
        if negative_drop_rate is None else negative_drop_rate
    )
    prompt_path = out/'cpa_multiple_choice_sam2_prompts.jsonl'
    output_path = out/'cpa_multiple_choice_sam2_results.jsonl'
    errors_path = out/'cpa_multiple_choice_sam2_errors.jsonl'
    artifacts = out/'cpa_multiple_choice_sam2'
    log_path = out/'cpa_multiple_choice_sam2.log'
    artifacts.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(prompt_path, prompts)
    if not prompts:
        atomic_jsonl(output_path, [])
        atomic_jsonl(errors_path, [])
        log_path.write_text('No reviewed CPA participant points to segment.\n', encoding='utf-8')
        return [], []
    missing = []
    if not Path(CPA_SAM2_PYTHON).is_file():
        missing.append(f'python:{CPA_SAM2_PYTHON}')
    for name in ('config.json', 'model.safetensors', 'preprocessor_config.json'):
        if not (CPA_SAM2_MODEL_PATH/name).is_file():
            missing.append(f'model_file:{CPA_SAM2_MODEL_PATH/name}')
    if missing:
        message = 'choice_sam2_preflight_missing:' + ','.join(missing)
        log_path.write_text(message + '\n', encoding='utf-8')
        errors = [{
            'id': 'cpa_multiple_choice_sam2_preflight',
            'kind': 'cpa_multiple_choice_sam2',
            'error': message,
        }]
        atomic_jsonl(errors_path, errors)
        return [], errors
    command = [
        CPA_SAM2_PYTHON,
        str(Path(__file__).resolve().parents[1]/'scripts'/'segment_cpa_observation_choices.py'),
        '--input', str(prompt_path), '--output', str(output_path),
        '--errors', str(errors_path), '--artifacts', str(artifacts),
        '--backend', 'transformers', '--model-path', str(CPA_SAM2_MODEL_PATH),
        '--device', CPA_SAM2_DEVICE,
        '--negative-a-at-720p', str(negative_a_at_720p),
        '--drop-rate', str(negative_drop_rate),
        '--batch-size', str(batch_size),
    ]
    if resume_existing:
        command.append('--resume-existing')
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        result.stdout + ('\nSTDERR:\n' + result.stderr if result.stderr else ''),
        encoding='utf-8',
    )
    rows = [
        json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if output_path.is_file() else []
    errors = [
        json.loads(line) for line in errors_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if errors_path.is_file() else []
    if result.returncode != 0 and not errors:
        errors.append({
            'id': 'cpa_multiple_choice_sam2_process',
            'kind': 'cpa_multiple_choice_sam2',
            'error': f'choice_sam2_exit_{result.returncode}:{result.stderr[-2000:]}',
        })
    return rows, errors


def run_cpa_tracker(
    audits_path: Path,
    output_path: Path,
    errors_path: Path,
    log_path: Path,
    resume_existing: bool,
    batch_size: int = 1,
) -> tuple[list[dict], list[dict]]:
    command = [
        CPA_TRACKER_PYTHON,
        str(Path(__file__).resolve().parents[1]/'scripts'/'track_cpa_points.py'),
        '--input', str(audits_path), '--output', str(output_path),
        '--errors', str(errors_path), '--checkpoint', str(CPA_TRACKER_CHECKPOINT),
        '--batch-size', str(batch_size),
    ]
    if resume_existing:
        command.append('--resume-existing')
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        result.stdout + ('\nSTDERR:\n' + result.stderr if result.stderr else ''),
        encoding='utf-8',
    )
    tracking = [
        json.loads(line) for line in output_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if output_path.is_file() else []
    errors = [
        json.loads(line) for line in errors_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ] if errors_path.is_file() else []
    if result.returncode != 0 and not errors:
        errors.append({
            'id': 'cpa_tracker_process', 'kind': 'cpa_tracking',
            'error': f'tracker_exit_{result.returncode}:{result.stderr[-2000:]}',
        })
    return tracking, errors


REPORT_CSS = '''body{font-family:system-ui;margin:20px;background:#f4f5f7;color:#18202a}.summary{background:#e8f4ff;padding:12px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:18px}.card{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #bbb}.card img,.card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}.ecot-media h4{margin:.3rem 0}.show-context{margin:9px 0;padding:9px;border:1px solid #829ab1;border-radius:7px;background:#f7fafc}.show-context summary{cursor:pointer;font-weight:750;color:#174ea6}.context-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.context-grid figure{margin:0;border:1px solid #ccd6e0}.context-grid figcaption{padding:4px;font-size:11px}.legend,.policy{font-size:13px;color:#334e68}.timeline{font-family:ui-monospace,monospace;font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f2f2;padding:8px}.meta{color:#555;font-size:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}.context-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}'''


def report_grounding(records: list[dict], out: Path) -> None:
    grounding = [record for record in records if record['kind'] == 'grounding']
    dataset_name = str(grounding[0].get('dataset_name') or grounding[0].get('dataset')) if grounding else 'Robot'
    cards = []
    for record in grounding:
        audit = record['model_inventory'].get('relation_audit')
        relation_html = f"<details><summary>Geometry audit</summary><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></details>" if audit else ''
        body = f'''<img src="annotated/{html.escape(record['annotated_image'])}"><p class="legend">{html.escape(record['annotation_legend'])}</p><p><b>Question:</b> {html.escape(record['question'])}</p><pre>{html.escape(record['answer'])}</pre>{relation_html}<details><summary>Eligible / filtered objects</summary><pre>{html.escape(json.dumps({'eligible': record['model_inventory']['objects'], 'filtered': record['model_inventory']['filtered_objects']}, ensure_ascii=False, indent=2))}</pre></details>'''
        cards.append(f'''<article class="card"><h3>{html.escape(record['id'])} · grounding</h3><p><b>Task:</b> {html.escape(record['task_instruction'])}</p>{body}<p class="meta">episode={record['episode']['episode_index']}, frame={record['selected_frame_index']}, type={html.escape(record['question_type'])}, policy={html.escape(record['visibility_policy'])}</p></article>''')
    header = f'''<h1>{html.escape(dataset_name)} Grounding review</h1><p class="summary">Grounding records: {len(grounding)}</p><p>Eligible objects exclude support surfaces, scene regions, robot parts, and people. All boxes and answers on this page belong only to the grounding artifact.</p>'''
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(dataset_name)} Grounding report</title><style>{REPORT_CSS}</style>{header}<section class="grid">' + ''.join(cards) + '</section>', encoding='utf-8')
    archive_legacy_artifact(out.parent/'report.html', 'report_combined_legacy.html')


def report_ecot(records: list[dict], out: Path) -> None:
    ecot = [record for record in records if record['kind'] == 'ecot']
    dataset_name = str(ecot[0].get('dataset_name') or ecot[0].get('dataset')) if ecot else 'Robot'
    cards = []
    for record in ecot:
        current = record['ecot_methods'].get('structured_three_part_ecot')
        if current is None:
            body = f'''<div class="ecot-media"><div><h4>Current target frame</h4><img src="images/{html.escape(record['image'])}"><p><b>{html.escape(record['question'])}</b></p><pre>{html.escape(record['answer'])}</pre><p class="policy">This legacy ECoT artifact uses only its stored current-image prediction fields.</p></div></div>'''
            cards.append(f'''<article class="card"><h3>{html.escape(record['id'])} · ECoT</h3><p><b>Task:</b> {html.escape(record['task_instruction'])}</p>{body}<p class="meta">episode={record['episode']['episode_index']}, frame={record['selected_frame_index']}, type={html.escape(record['question_type'])}, policy={html.escape(record['visibility_policy'])}</p></article>''')
            continue
        seg = record['gripper_segmentation']['selected_segment']
        context = record['annotation_context']
        timeline = ' | '.join(f"S{s['segment_index']} [{s['start_frame']},{s['end_frame_exclusive']}) {s['gripper_state']}{' ← selected' if s['segment_index']==seg['segment_index'] else ''}" for s in record['gripper_segmentation']['segments'])
        context_frames = ''.join(
            f'''<figure><img src="ecot_context/{html.escape(frame['image'])}"><figcaption>{html.escape(frame['temporal_relation'])} {float(frame['offset_seconds']):+.3f}s · frame {int(frame['frame_index'])}</figcaption></figure>'''
            for frame in context['sampled_frames']
        )
        sample_type = record.get('ecot_sample_type', ECOT_PRIMARY_SAMPLE_TYPE)
        body = f'''<div class="ecot-media"><div><h4>Current target frame</h4><img src="images/{html.escape(record['image'])}">
<details class="show-context"><summary>Show ECoT teacher context</summary><p><b>Target position:</b> frame {int(context['target_frame_index'])}/{int(context['source_video_frame_count'])}, {float(context['target_progress_ratio'])*100:.2f}% through episode, {float(context['target_time_seconds']):.3f}s.</p><p><b>Whole-episode VLM progress summary:</b> {html.escape(context['target_frame_progress_summary'])}</p><h4>Full episode video</h4><video controls preload="metadata" src="{html.escape(context['whole_episode_video'])}"></video><h4>Past/future one-frame-per-second context</h4><div class="context-grid">{context_frames}</div></details>
<p><b>{html.escape(current['question'])}</b></p><pre>{html.escape(current['answer'])}</pre><p class="policy">Training-visible input: global task plus this current image only. The full video, one-Hz past/future frames, and progress summary are privileged ECoT annotation context and are excluded from training inputs.</p></div></div><details><summary>Offline ECoT frame-selection audit</summary><p class="timeline">{html.escape(timeline)}</p><pre>{html.escape(json.dumps(record['gripper_segmentation'], ensure_ascii=False, indent=2))}</pre></details>'''
        cards.append(f'''<article class="card"><h3>{html.escape(record['id'])} · ECoT</h3><p><b>Task:</b> {html.escape(record['task_instruction'])}</p>{body}<p class="meta">episode={record['episode']['episode_index']}, frame={record['selected_frame_index']}, sample={html.escape(sample_type)}, type={html.escape(record['question_type'])}, policy={html.escape(record['visibility_policy'])}</p></article>''')
    midpoint_count = sum(
        record.get('ecot_sample_type') == ECOT_MIDPOINT_SAMPLE_TYPE for record in ecot
    )
    header = f'''<h1>{html.escape(dataset_name)} ECoT review</h1><p class="summary">ECoT records: {len(ecot)} · transition starts: {len(ecot)-midpoint_count} · between-transition midpoints: {midpoint_count}</p><p>Each record contains a concise Scene Description, Task Progress Assessment, and goal-level Subtask with action/object/source/target semantics.</p>'''
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(dataset_name)} ECoT report</title><style>{REPORT_CSS}</style>{header}<section class="grid">' + ''.join(cards) + '</section>', encoding='utf-8')
    archive_legacy_artifact(out.parent/'report.html', 'report_combined_legacy.html')


def report_cpa(audits: list[dict], records: list[dict], out: Path) -> None:
    dataset_source = audits[0] if audits else (records[0] if records else {})
    dataset_name = str(
        dataset_source.get('dataset_name') or dataset_source.get('dataset') or 'Dataset'
    )
    def record_point_count(record: dict) -> int:
        return int(record.get('point_count', len(record.get('answer_value') or [])))

    def render_observation(record: dict) -> str:
        review = record.get('tracking_review') or {}
        review_overlay = review.get('tracking_review_overlay')
        review_details = ''
        if review_overlay:
            review_details = f'''<details><summary>Show tracked points before review</summary><img src="cpa_tracking_review/{html.escape(review_overlay)}"><pre>{html.escape(json.dumps({key: review.get(key) for key in ('validated_pairs', 'pair_distance_checks', 'removed_point_ids', 'review_reason', 'tracking_review_model', 'tracking_review_version', 'pair_separation_review_version')}, ensure_ascii=False, indent=2))}</pre></details>'''
        multiple_choice_questions = record.get('multiple_choice_questions') or []
        multiple_choice_html = ''.join(
            f'''<section class="cpa-multiple-choice-question" data-question-type="{html.escape(question['question_type'])}"><h5>{html.escape(question['target_entity'])} · <code>{html.escape(question['question_type'])}</code></h5><img class="cpa-choice-image" loading="lazy" src="cpa_multiple_choice_sam2/{html.escape(question['image'])}"><p>{html.escape(question['question'])}</p><p><b>Choices:</b> <code>{html.escape(json.dumps(question['choices'], ensure_ascii=False))}</code></p><p><b>Answer:</b> <code>{html.escape(question['answer'])}</code></p><details><summary>Show observation-local SAM2 sampling evidence</summary><img loading="lazy" src="cpa_multiple_choice_sam2/{html.escape(question['teacher_evidence']['observation_local_sam2_mask_and_candidates_image'])}"><pre>{html.escape(json.dumps({key: question['teacher_evidence'].get(key) for key in ('positive_point_ids', 'sam_prompt_consistent', 'negative_sampling', 'choice_metadata')}, ensure_ascii=False, indent=2))}</pre></details></section>'''
            for question in multiple_choice_questions
        )
        multiple_choice_block = (
            f'''<section class="cpa-multiple-choice-questions"><h4>Additional CPA multiple-choice questions</h4>{multiple_choice_html}</section>'''
            if multiple_choice_html else
            '<section class="cpa-multiple-choice-questions cpa-no-questions"><h4>Additional CPA multiple-choice questions</h4><p class="policy">No multiselect question was generated for this observation.</p></section>'
        )
        dataset_policy = (
            'Robot data emits only the contacted-object question; gripper points '
            'remain teacher-only.'
            if cpa_is_robot_data(record) else
            'Human data may emit one hand/held-tool question and one contacted-object question.'
        )
        learner_questions = f'''<div class="cpa-learner-questions">{multiple_choice_block}<p class="policy">Training-visible input: global task, one pre-contact observation with neutral A/B/C markers, and one participant-specific multiselect question. {html.escape(dataset_policy)} SAM2 masks, positive/negative identities, contact frames, tracking, and reviews are teacher-only.</p></div>'''
        return f'''<div class="observation-review" data-has-contact-points="{'true' if record_point_count(record) > 0 else 'false'}"><div class="observation-student-row"><figure class="observation-visual"><img src="cpa_annotated/{html.escape(record['annotated_image'])}"><figcaption><code>{html.escape(record['observation_id'])}</code><br>frame {int(record['query_frame_index'])} · TTC {float(record['time_to_contact_seconds']):.3f}s<br>kept={html.escape(record['answer'])}<br>removed={html.escape(json.dumps(record.get('removed_point_ids', [])))}</figcaption></figure>{learner_questions}</div>{review_details}</div>'''

    records_by_candidate: dict[str, list[dict]] = {}
    for record in records:
        records_by_candidate.setdefault(record['candidate_id'], []).append(record)
    episode_groups: dict[str, dict] = {}
    for audit in sorted(audits, key=lambda item: item['candidate_id']):
        candidate_records = sorted(
            records_by_candidate.get(audit['candidate_id'], []),
            key=lambda item: item['query_frame_index'],
        )
        if not candidate_records:
            continue
        nonempty_observation_count = sum(
            record_point_count(record) > 0
            for record in candidate_records
        )
        observations = ''.join(map(render_observation, candidate_records))
        intermediate = {
            'point_method': audit.get('point_method', 'seed_vlm'),
            'contact_agent_role_version': audit.get('contact_agent_role_version'),
            'contact_agent_role': audit.get('contact_agent_role'),
            'contact_agent_name': audit.get('contact_agent_name'),
            'contact_object_participant': audit.get('contact_object_participant'),
            'contact_event_type': audit.get('contact_event_type'),
            'pro_contact_pair': audit.get('pro_contact_pair'),
            'interaction_bbox_xyxy': audit['interaction_bbox_xyxy'],
            'interaction_bbox_pixel_xyxy': audit['interaction_bbox_pixel_xyxy'],
            'contact_point_pairs_crop_xy': audit.get(
                'contact_point_pairs_crop_xy', audit.get('contact_points_crop_xy'),
            ),
            'contact_point_pairs_full_xy': audit.get(
                'contact_point_pairs_full_xy', audit.get('contact_points_full_xy'),
            ),
            'region_reason': audit['region_reason'],
            'points_reason': audit['points_reason'],
            'point_pair_version': audit.get('point_pair_version'),
            'point_model': audit.get('point_model'),
            'region_prompt': audit['region_user_prompt'],
            'points_prompt': audit['points_user_prompt'],
            'sam2_prompts': audit.get('sam2_prompts'),
            'sam2_segmentation': audit.get('sam2_segmentation'),
            'tracking': audit.get('tracking'),
        }
        pair_figure = ''
        if audit.get('pair_candidate_overlay'):
            pair_figure = f'''<figure><img src="cpa_crops/{html.escape(audit['pair_candidate_overlay'])}"><figcaption>Lite-generated paired H_i/O_i source targets</figcaption></figure>'''
        point_method = audit.get('point_method', 'seed_vlm')
        if point_method in {'sam2', 'sam2_disk_prototype'}:
            sam2 = audit.get('sam2_segmentation') or {}
            if point_method == 'sam2_disk_prototype':
                radius = int(sam2.get('prototype_radius_pixels', 0))
                method_description = f'Lite identifies the participating hand/end effector and contacted object and emits geometric prompts. SAM2 segments Bh and Bo; H0/O0 are the centers of equal-radius {radius}px disk prototypes fully contained in their masks with minimum center distance.'
            else:
                method_description = 'Lite identifies the participating hand/end effector and contacted object and emits geometric prompts. SAM2 segments Bh and Bo; the source H0/O0 pair is the deterministic minimum-distance pair across their mask boundaries.'
            sam2_figures = ''.join(
                f'''<figure><img src="cpa_sam2/{html.escape(sam2[name])}"><figcaption>{caption}</figcaption></figure>'''
                for name, caption in (
                    ('hand_mask_image', 'SAM2 hand/end-effector mask Bh'),
                    ('object_mask_image', 'SAM2 contacted-object mask Bo'),
                    ('mask_pair_overlay', 'Bh/Bo overlay and deterministic closest H0/O0 pair'),
                ) if sam2.get(name)
            )
        else:
            method_description = 'Lite directly labels paired H_i end-effector and O_i object targets near each visible contact interface.'
            sam2_figures = ''
        episode_video = audit.get('episode_video') or f'{audit["row_id"]}.mp4'
        episode_id = str(audit.get('row_id') or audit['candidate_id'])
        group = episode_groups.setdefault(episode_id, {
            'task_instruction': audit['task_instruction'],
            'nonempty_observation_count': 0,
            'contacts': [],
        })
        group['nonempty_observation_count'] += nonempty_observation_count
        group['contacts'].append(f'''<section class="cpa-contact-section" data-candidate-id="{html.escape(audit['candidate_id'])}"><h4>{html.escape(audit['candidate_id'])} · contact</h4><p><b>Contact:</b> frame {int(audit['candidate_frame'])} · {html.escape(audit['contact']['verb'])} {html.escape(audit['contact']['noun'])} · {html.escape(audit['end_effector_type'])}</p><div class="cpa-observation-list">{observations}</div>
<details class="show-intermediates"><summary>Show intermediates</summary><p>{html.escape(method_description)} CoTracker projects every resulting point backward. For each observation, the geometric review removes both H_i and O_i unless their separation has grown by more than 100% from the contact frame; Lite then independently removes any remaining semantically invalid point.</p><h4>Full episode video</h4><video class="cpa-episode-video" controls preload="metadata" src="episode_videos/{html.escape(episode_video)}"><source src="episode_videos/{html.escape(episode_video)}" type="video/mp4"></video><p class="meta">Accepted contact frame: {int(audit['candidate_frame'])}.</p><div class="context-grid"><figure><img src="cpa_intermediates/{html.escape(audit['annotated_contact_frame'])}"><figcaption>Contact frame: interaction box and paired H_i/O_i targets</figcaption></figure><figure><img src="cpa_intermediates/{html.escape(audit['annotated_interaction_crop'])}"><figcaption>Interaction crop with paired targets</figcaption></figure>{pair_figure}{sam2_figures}</div><pre>{html.escape(json.dumps(intermediate, ensure_ascii=False, indent=2))}</pre></details></section>''')
    cards = [
        f'''<article class="card cpa-episode-row" data-cpa-episode="{html.escape(episode_id)}" data-nonempty-observation-count="{int(group['nonempty_observation_count'])}"><h3>{html.escape(episode_id)} · CPA episode</h3><p><b>Task:</b> {html.escape(group['task_instruction'])}</p>{''.join(group['contacts'])}</article>'''
        for episode_id, group in episode_groups.items()
    ]
    css = REPORT_CSS + '''.cpa-report-grid{display:grid;grid-template-columns:minmax(0,1fr);gap:20px}.cpa-episode-row{min-width:0;width:auto}.cpa-contact-section{min-width:0;margin-top:16px;padding-top:14px;border-top:2px solid #d9e2ec}.cpa-contact-section:first-of-type{margin-top:8px}.cpa-contact-section>h4{margin:0 0 6px;color:#243b53}.cpa-observation-list{display:flex;flex-direction:column;gap:14px;margin-top:12px}.observation-review{min-width:0;padding:12px;border:1px solid #ccd6e0;border-radius:8px;background:#fbfcfe}.observation-student-row{display:grid;grid-template-columns:minmax(320px,0.9fr) minmax(480px,1.1fr);gap:16px;align-items:start}.observation-visual{min-width:0;margin:0;border:1px solid #ccd6e0;background:#fff}.observation-visual figcaption{padding:8px;overflow-wrap:anywhere}.cpa-learner-questions{min-width:0;padding:10px;border:1px solid #b8c5d6;border-radius:7px;background:#fff}.cpa-learner-questions h4{margin:0 0 8px}.cpa-learner-questions h5{margin:0 0 7px}.cpa-multiple-choice-question{margin-top:10px;padding:9px;border:1px solid #9fb3c8;border-radius:7px;background:#f7fafc}.cpa-multiple-choice-question .cpa-choice-image{aspect-ratio:16/9;object-fit:contain;background:#111}.cpa-multiple-choice-question details{margin-top:7px}.cpa-learner-questions code{white-space:normal;overflow-wrap:anywhere}.cpa-no-questions{padding:8px;background:#f7fafc}.show-intermediates{margin-top:12px;padding:10px;border:1px solid #829ab1;border-radius:8px;background:#f7fafc}.show-intermediates summary,.observation-review summary{cursor:pointer;font-weight:750;color:#174ea6}.show-intermediates .cpa-episode-video{display:block;width:min(100%,960px);max-height:70vh;background:#111}.context-grid{margin-top:10px}.context-grid figure{background:#fff}.context-grid img{aspect-ratio:16/9}.context-grid figcaption{padding:7px;overflow-wrap:anywhere}.observation-review>details{margin-top:10px;padding:6px;border:1px solid #ccd6e0;background:#f7fafc}.observation-review>details img{margin-top:7px}.cpa-hide-empty .observation-review[data-has-contact-points="false"],.cpa-hide-empty .card[data-nonempty-observation-count="0"]{display:none}.cpa-filter-control{position:fixed;right:18px;bottom:18px;z-index:1000;width:min(340px,calc(100vw - 36px));padding:11px 13px;border:1px solid #829ab1;border-radius:10px;background:rgba(255,255,255,.96);box-shadow:0 3px 14px #0003;font-size:13px}.cpa-filter-control label{display:flex;align-items:center;gap:8px;font-weight:750;cursor:pointer}.cpa-filter-control input{width:18px;height:18px}.cpa-filter-status{display:block;margin-top:5px;color:#52606d;font-size:12px}@media(max-width:1050px){.observation-student-row{grid-template-columns:minmax(0,1fr)}}'''
    actor_counts = {
        actor: sum(audit.get('actor', 'robot') == actor for audit in audits)
        for actor in ('robot', 'human')
    }
    removed = sum(len(record.get('removed_point_ids', [])) for record in records)
    kept = sum(record_point_count(record) for record in records)
    multiple_choice_question_count = sum(
        len(record.get('multiple_choice_questions') or []) for record in records
    )
    header = f'''<h1>{html.escape(dataset_name)} Contact Point Anticipation review</h1><p class="summary">Accepted contacts with CPA: {len(audits)} · learner observations: {len(records)} · multiselect questions: {multiple_choice_question_count} · reviewed points kept: {kept} · removed: {removed} · robot: {actor_counts['robot']} · human: {actor_counts['human']}</p><p>Tracked contact annotations remain as audit and as positive supervision for participant multiselect questions. Robot data emits contacted-object learner questions only; human data may also emit a hand/held-tool question. SAM2 re-segments the exact pre-contact observation. Each negative chooses the mask-exterior domain with probability {CPA_NEGATIVE_DROP_RATE:g}; both mask-interior and mask-exterior candidates use f(r)=1/(r-a), with exclusion radius {CPA_NEGATIVE_A_AT_720P:g} pixels at 720p scaled with observation height. No negative candidate is passed through CoTracker. Future media, masks, sampling identities, and reviews are teacher-only.</p>'''
    filter_control = '''<aside class="cpa-filter-control"><label><input id="cpa-hide-empty-toggle" type="checkbox" checked>Hide frames without tracked contact points</label><span id="cpa-filter-status" class="cpa-filter-status" aria-live="polite"></span></aside>'''
    filter_script = '''<script>(()=>{const toggle=document.getElementById('cpa-hide-empty-toggle');const status=document.getElementById('cpa-filter-status');const frames=[...document.querySelectorAll('.observation-review')];const cards=[...document.querySelectorAll('.card[data-nonempty-observation-count]')];function apply(){const hide=toggle.checked;document.body.classList.toggle('cpa-hide-empty',hide);const visibleFrames=hide?frames.filter(frame=>frame.dataset.hasContactPoints==='true').length:frames.length;const visibleCards=hide?cards.filter(card=>Number(card.dataset.nonemptyObservationCount)>0).length:cards.length;status.textContent=`Showing ${visibleFrames}/${frames.length} frames in ${visibleCards}/${cards.length} contact episodes`;toggle.setAttribute('aria-checked',String(hide));}toggle.addEventListener('change',apply);apply();})();</script>'''
    out.write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(dataset_name)} CPA report</title><style>{css}</style></head><body class="cpa-hide-empty">{header}<section class="cpa-report-grid">' + ''.join(cards) + f'</section>{filter_control}{filter_script}</body></html>',
        encoding='utf-8',
    )


def _report_asset_prefix(root: Path, report_parent: Path) -> str:
    try:
        relative = root.resolve().relative_to(Path('/mnt/outputs').resolve())
        return f'/files/{urllib.parse.quote(relative.as_posix(), safe="/")}'
    except ValueError:
        relative = os.path.relpath(root.resolve(), report_parent.resolve())
        return '' if relative == '.' else Path(relative).as_posix().rstrip('/')


def _report_asset(prefix: str, relative: str) -> str:
    path = relative.lstrip('/')
    return f'{prefix}/{path}' if prefix else path


def report_cpa_point_compare(
    seed_root: Path,
    sam2_root: Path,
    out: Path,
) -> dict:
    """Compare direct Seed-VLM CPA points with SAM2-derived points and tracks."""
    def read_jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ] if path.is_file() else []

    def method_labels(audit: dict) -> tuple[str, str]:
        method = audit.get('point_method', 'seed_vlm')
        if method == 'sam2_disk_prototype':
            segmentation = audit.get('sam2_segmentation') or {}
            radius = int(segmentation.get('prototype_radius_pixels', 0))
            label = f'SAM2 disk prototype r={radius}px'
            return label, label
        if method == 'sam2':
            return 'SAM2', 'SAM2 closest-mask'
        if audit.get('contact_agent_role_version') == CPA_CONTACT_AGENT_ROLE_VERSION:
            return 'Seed VLM Pro-role-aware', 'Seed VLM Pro-role-aware'
        return 'Seed VLM', 'Seed VLM'

    seed_audits = {
        item['candidate_id']: item
        for item in read_jsonl(seed_root/'cpa_contact_audit.jsonl')
    }
    sam2_audits = {
        item['candidate_id']: item
        for item in read_jsonl(sam2_root/'cpa_contact_audit.jsonl')
    }
    seed_records = {
        item['observation_id']: item
        for item in read_jsonl(seed_root/'cpa_records.jsonl')
    }
    sam2_records = {
        item['observation_id']: item
        for item in read_jsonl(sam2_root/'cpa_records.jsonl')
    }
    prototype_tuning = {
        item['candidate_id']: item
        for item in read_jsonl(sam2_root/'cpa_sam2_prototype_tuning.jsonl')
    }
    seed_prefix = _report_asset_prefix(seed_root, out.parent)
    sam2_prefix = _report_asset_prefix(sam2_root, out.parent)
    matched_ids = sorted(set(seed_audits) & set(sam2_audits))
    cards = []
    matched_observations = 0
    for candidate_id in matched_ids:
        seed = seed_audits[candidate_id]
        sam2 = sam2_audits[candidate_id]
        seed_short_label, seed_label = method_labels(seed)
        sam2_short_label, sam2_label = method_labels(sam2)
        observation_ids = sorted(
            {
                item['observation_id'] for item in seed_records.values()
                if item['candidate_id'] == candidate_id
            } & {
                item['observation_id'] for item in sam2_records.values()
                if item['candidate_id'] == candidate_id
            },
            key=lambda observation_id: seed_records[observation_id]['query_frame_index'],
        )
        matched_observations += len(observation_ids)
        observation_rows = ''.join(
            f'''<div class="point-observation"><h4><code>{html.escape(observation_id)}</code> · frame {int(seed_records[observation_id]['query_frame_index'])}</h4><div class="method-grid"><figure><img src="{html.escape(_report_asset(seed_prefix, 'cpa_annotated/' + seed_records[observation_id]['annotated_image']))}"><figcaption>{html.escape(seed_label)} after distance + Lite review · kept={int(seed_records[observation_id].get('point_count', 0))} · removed={html.escape(json.dumps(seed_records[observation_id].get('removed_point_ids', [])))}</figcaption></figure><figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_annotated/' + sam2_records[observation_id]['annotated_image']))}"><figcaption>{html.escape(sam2_label)} after distance + Lite review · kept={int(sam2_records[observation_id].get('point_count', 0))} · removed={html.escape(json.dumps(sam2_records[observation_id].get('removed_point_ids', [])))}</figcaption></figure></div></div>'''
            for observation_id in observation_ids
        )
        sam2_segmentation = sam2.get('sam2_segmentation') or {}
        seed_pair_figure = (
            f'''<figure><img src="{html.escape(_report_asset(seed_prefix, 'cpa_crops/' + seed['pair_candidate_overlay']))}"><figcaption>Seed VLM direct H_i/O_i source pairs</figcaption></figure>'''
            if seed.get('pair_candidate_overlay') else ''
        )
        seed_segmentation = seed.get('sam2_segmentation') or {}
        seed_pair_figure += ''.join(
            f'''<figure><img src="{html.escape(_report_asset(seed_prefix, 'cpa_sam2/' + seed_segmentation[name]))}"><figcaption>{html.escape(seed_label)} · {caption}</figcaption></figure>'''
            for name, caption in (
                ('hand_mask_image', 'hand/end-effector mask Bh'),
                ('object_mask_image', 'contacted-object mask Bo'),
                ('mask_pair_overlay', 'source target overlay'),
            ) if seed_segmentation.get(name)
        )
        sam2_figures = ''.join(
            f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_sam2/' + sam2_segmentation[name]))}"><figcaption>{caption}</figcaption></figure>'''
            for name, caption in (
                ('hand_mask_image', 'SAM2 hand/end-effector mask Bh'),
                ('object_mask_image', 'SAM2 contacted-object mask Bo'),
                (
                    'mask_pair_overlay',
                    'Bh/Bo overlay and source H0/O0 targets',
                ),
            ) if sam2_segmentation.get(name)
        )
        if sam2.get('pair_candidate_overlay'):
            sam2_figures += f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_crops/' + sam2['pair_candidate_overlay']))}"><figcaption>{html.escape(sam2_label)} candidate points and selected H0/O0 pair</figcaption></figure>'''
        tuning = prototype_tuning.get(candidate_id)
        prototype_summary = ''
        prototype_figures = ''
        if tuning:
            artifacts = tuning.get('artifacts') or {}
            best = tuning.get('best') or {}
            best_radius = int(tuning.get('best_radius_pixels', 0))
            best_overlay = artifacts.get('best_overlay')
            contact_sheet = artifacts.get('contact_sheet')
            best_figure = (
                f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_sam2_prototypes/' + best_overlay))}"><figcaption>Selected disk prototypes: r={best_radius}px · center distance={float(best.get('center_distance_pixels', 0.0)):.2f}px · edge gap={float(best.get('prototype_edge_gap_pixels', 0.0)):.2f}px</figcaption></figure>'''
                if best_overlay else ''
            )
            sheet_figure = (
                f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_sam2_prototypes/' + contact_sheet))}"><figcaption>Radius sweep and contact-branch stability</figcaption></figure>'''
                if contact_sheet else ''
            )
            prototype_summary = f'''<section class="prototype-tuning"><h4>Disk-prototype radius tuning</h4><p><code>ph</code> and <code>po</code> are filled radius-r disks fully contained in <code>Bh</code> and <code>Bo</code>. The selected radius is <b>{best_radius}px</b>: it is the largest tested radius that stays on the small-radius contact branch and keeps the two disk boundaries within {float(tuning.get('maximum_prototype_edge_gap_pixels', 0.0)):.2f}px. This visualization is a source-point experiment; the observation tracks below still use the original SAM2 point pair.</p><div class="method-grid">{best_figure}{sheet_figure}</div></section>'''
            prototype_figures = ''.join(
                f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_sam2_prototypes/' + str(candidate['overlay_image'])))}"><figcaption>r={int(candidate['radius_pixels'])}px · drift={float(candidate.get('max_center_drift_from_anchor_pixels', 0.0)):.2f}px · stable={str(bool(candidate.get('stable_contact_branch'))).lower()}</figcaption></figure>'''
                for candidate in tuning.get('candidates', [])
                if candidate.get('overlay_image')
            )
            prototype_figures += ''.join(
                f'''<figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_sam2_prototypes/' + artifacts[name]))}"><figcaption>{caption}</figcaption></figure>'''
                for name, caption in (
                    ('best_hand_feasible_centers', f'Feasible ph centers after Bh erosion at r={best_radius}px'),
                    ('best_object_feasible_centers', f'Feasible po centers after Bo erosion at r={best_radius}px'),
                ) if artifacts.get(name)
            )
        provenance = {
            'contact_frame_identical': seed.get('candidate_frame') == sam2.get('candidate_frame'),
            'contact_identical': seed.get('contact') == sam2.get('contact'),
            'interaction_bbox_identical': seed.get('interaction_bbox_xyxy') == sam2.get('interaction_bbox_xyxy'),
            'seed': {
                'point_method': seed.get('point_method', 'seed_vlm'),
                'point_pair_version': seed.get('point_pair_version'),
                'source_pairs': seed.get('contact_point_pairs_full_xy'),
            },
            'sam2': {
                'point_method': sam2.get('point_method'),
                'point_pair_version': sam2.get('point_pair_version'),
                'region_reused_from': sam2.get('region_reused_from'),
                'prompts': sam2.get('sam2_prompts'),
                'segmentation': sam2_segmentation,
                'source_pairs': sam2.get('contact_point_pairs_full_xy'),
            },
            'disk_prototype_tuning': tuning,
        }
        cards.append(f'''<article class="card"><h3>{html.escape(candidate_id)} · Point Compare</h3><p><b>Contact:</b> frame {int(sam2['candidate_frame'])} · {html.escape(str(sam2['contact']['verb']))} {html.escape(str(sam2['contact']['noun']))}</p><div class="method-grid"><figure><img src="{html.escape(_report_asset(seed_prefix, 'cpa_intermediates/' + seed['annotated_interaction_crop']))}"><figcaption>{html.escape(seed_label)} source targets · {int(seed.get('contact_point_pair_count', 0))} pair(s)</figcaption></figure><figure><img src="{html.escape(_report_asset(sam2_prefix, 'cpa_intermediates/' + sam2['annotated_interaction_crop']))}"><figcaption>{html.escape(sam2_label)} source targets · {int(sam2.get('contact_point_pair_count', 0))} pair(s)</figcaption></figure></div>{prototype_summary}<div class="observation-list">{observation_rows}</div><details class="show-intermediates"><summary>Show intermediates</summary><p>The interaction bbox, crop, contact frame, observations, CoTracker, and Lite tracking-review policy are held constant. Only source-point selection changes.</p><div class="intermediate-grid"><figure><img src="{html.escape(_report_asset(seed_prefix, 'cpa_crops/' + seed['interaction_crop_image']))}"><figcaption>Shared interaction crop</figcaption></figure>{seed_pair_figure}{sam2_figures}{prototype_figures}</div><pre>{html.escape(json.dumps(provenance, ensure_ascii=False, indent=2))}</pre></details></article>''')

    seed_kept = sum(int(item.get('point_count', 0)) for item in seed_records.values())
    sam2_kept = sum(int(item.get('point_count', 0)) for item in sam2_records.values())
    seed_removed = sum(len(item.get('removed_point_ids', [])) for item in seed_records.values())
    sam2_removed = sum(len(item.get('removed_point_ids', [])) for item in sam2_records.values())
    metrics = {
        'matched_contacts': len(matched_ids),
        'matched_observations': matched_observations,
        'seed_contacts': len(seed_audits),
        'sam2_contacts': len(sam2_audits),
        'seed_points_kept': seed_kept,
        'sam2_points_kept': sam2_kept,
        'seed_points_removed': seed_removed,
        'sam2_points_removed': sam2_removed,
        'missing_in_sam2': sorted(set(seed_audits) - set(sam2_audits)),
        'missing_in_seed': sorted(set(sam2_audits) - set(seed_audits)),
        'prototype_tuned_contacts': len(set(matched_ids) & set(prototype_tuning)),
    }
    css = REPORT_CSS + '''.method-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.method-grid figure,.intermediate-grid figure{margin:0;border:1px solid #ccd6e0;background:#fff}.method-grid figcaption,.intermediate-grid figcaption{padding:7px;overflow-wrap:anywhere}.prototype-tuning{margin:14px 0;padding:10px;border:2px solid #2f855a;border-radius:9px;background:#f0fff4}.prototype-tuning h4{margin:.1rem 0 .5rem;color:#276749}.prototype-tuning figure{margin:0}.prototype-tuning img{object-fit:contain;background:#111}.point-observation{margin-top:12px;padding-top:9px;border-top:1px solid #d9dfeb}.point-observation h4{margin:.2rem 0 .6rem}.show-intermediates{margin-top:14px;padding:10px;border:1px solid #829ab1;border-radius:8px;background:#f7fafc}.show-intermediates summary{cursor:pointer;font-weight:750;color:#174ea6}.intermediate-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin-top:10px}.intermediate-grid img{aspect-ratio:16/9;object-fit:contain;background:#111}@media(max-width:900px){.method-grid,.intermediate-grid{grid-template-columns:1fr}}'''
    if matched_ids:
        left_short, _ = method_labels(seed_audits[matched_ids[0]])
        right_short, _ = method_labels(sam2_audits[matched_ids[0]])
    else:
        left_short, right_short = 'baseline', 'candidate'
    header = f'''<h1>CPA Point Compare — {html.escape(left_short)} vs {html.escape(right_short)}</h1><p class="summary">Matched contacts: {metrics['matched_contacts']} · matched observations: {matched_observations} · baseline kept/removed: {seed_kept}/{seed_removed} · candidate kept/removed: {sam2_kept}/{sam2_removed} · disk-prototype tuning: {metrics['prototype_tuned_contacts']}</p><p>Both branches use the same accepted contact frame, interaction crop, learner observations, CoTracker, pair-separation review, and Lite per-frame semantic review. The pair-separation review removes both H_i and O_i unless tracked separation has grown by more than 100% from the contact-frame distance. Only the contact-frame source-target construction changes. Optional disk-prototype panels are shown when radius-tuning metadata exists.</p>'''
    out.write_text(
        f'<!doctype html><meta charset="utf-8"><title>CPA Point Compare</title><style>{css}</style>{header}<section class="grid">' + ''.join(cards) + '</section>',
        encoding='utf-8',
    )
    return metrics


def cpa_annotation_compatible(annotation: dict, expected_audit: dict | None = None) -> bool:
    pairs = annotation.get('contact_point_pairs_full_xy')
    expected_agent = cpa_contact_agent_semantics(expected_audit or annotation)
    return (
        (
            expected_audit is None or
            (
                annotation.get('candidate_frame') == expected_audit.get('candidate_frame') and
                annotation.get('contact') == expected_audit.get('contact')
            )
        ) and
        annotation.get('point_pair_version') == active_cpa_point_pair_version() and
        annotation.get('contact_agent_role_version') == CPA_CONTACT_AGENT_ROLE_VERSION and
        annotation.get('contact_agent_role') == expected_agent['role'] and
        annotation.get('contact_agent_name') == expected_agent['agent_name'] and
        annotation.get('contact_object_participant') == expected_agent['object_name'] and
        annotation.get('contact_event_type') == expected_agent['event_type'] and
        annotation.get('pro_contact_pair') == expected_agent['contact_pair'] and
        annotation.get('point_method', 'seed_vlm') == CPA_POINT_METHOD and
        annotation.get('point_model') == CPA_POINT_MODEL and
        (
            CPA_POINT_METHOD != 'sam2' or
            isinstance(annotation.get('sam2_segmentation'), dict)
        ) and
        bool(annotation.get('points_raw_model_output')) and
        isinstance(pairs, list) and 1 <= len(pairs) <= 5 and
        [pair.get('pair_index') for pair in pairs] == list(range(len(pairs))) and
        all(
            isinstance(pair, dict) and
            all(
                isinstance(pair.get(f'{role}_point_xy'), list) and
                len(pair[f'{role}_point_xy']) == 2 and
                all(
                    isinstance(coordinate, (int, float)) and 0 <= coordinate <= 1
                    for coordinate in pair[f'{role}_point_xy']
                )
                for role in ('hand', 'object')
            )
            for pair in pairs
        )
    )


def generate_cpa_dataset(
    endpoint: str,
    rows_by_id: dict[str, dict],
    contact_audits: list[dict],
    out: Path,
    request_workers: int,
    track_workers: int,
    max_attempts: int,
    resume_existing: bool,
    cpa_sam2_batch_size: int = CPA_OBSERVATION_SAM2_BATCH_SIZE,
    on_local_sam2_start=None,
) -> tuple[list[dict], list[dict], list[dict]]:
    contact_frame_dir = out/'cpa_contact_frames'
    crop_dir = out/'cpa_crops'
    intermediate_dir = out/'cpa_intermediates'
    annotated_dir = out/'cpa_annotated'
    cache_dir = out/'cpa_cache'
    tracking_review_dir = out/'cpa_tracking_review'
    tracking_review_cache_dir = out/'cpa_tracking_review_cache'
    for directory in (
        contact_frame_dir, crop_dir, intermediate_dir, annotated_dir, cache_dir,
        tracking_review_dir, tracking_review_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    accepted = [audit for audit in contact_audits if audit.get('valid_contact')]
    accepted_by_id = {audit['candidate_id']: audit for audit in accepted}
    expected_ids = {audit['candidate_id'] for audit in accepted}
    cached = {}
    if resume_existing:
        for path in sorted(cache_dir.glob('*.json')):
            item = json.loads(path.read_text(encoding='utf-8'))
            if (
                item.get('candidate_id') in expected_ids and
                cpa_annotation_compatible(
                    item, accepted_by_id.get(item.get('candidate_id')),
                )
            ):
                cached[item['candidate_id']] = item
    annotations = list(cached.values())
    errors = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=request_workers,
    ) as executor:
        jobs = {}
        for audit in accepted:
            if audit['candidate_id'] in cached:
                continue
            row = rows_by_id[audit['row_id']]
            normalized_audit = {
                **audit,
                'actor': audit.get('actor', row.get('actor', 'robot')),
                'dataset': audit.get('dataset', row.get('dataset', 'Robot')),
                'dataset_name': audit.get(
                    'dataset_name', row.get('dataset_name', row.get('dataset', 'Robot')),
                ),
                'dataset_family': audit.get(
                    'dataset_family', row.get('dataset_family', row.get('actor', 'robot')),
                ),
            }
            jobs[executor.submit(
                annotate_cpa_contact, endpoint, row, normalized_audit,
                contact_frame_dir, crop_dir, max_attempts,
            )] = normalized_audit
        for future in concurrent.futures.as_completed(jobs):
            audit = jobs[future]
            try:
                annotation = future.result()
                annotations.append(annotation)
                atomic_json(cache_dir/f'{annotation["candidate_id"]}.json', annotation)
            except Exception as error:
                errors.append({
                    'id': audit['candidate_id'], 'kind': 'cpa_vlm',
                    'error': f'{type(error).__name__}:{error}',
                })
            print(
                f'cpa_vlm_progress completed={len(annotations) + len(errors)}/{len(accepted)} '
                f'accepted={len(annotations)} errors={len(errors)}',
                flush=True,
            )
    annotations.sort(key=lambda item: item['candidate_id'])
    # Dataset identity belongs to the current run, not to a potentially reused
    # annotation cache.  Always refresh it from the selected source row.
    for annotation in annotations:
        row = rows_by_id[annotation['row_id']]
        annotation.update({
            'dataset': row['dataset'],
            'dataset_name': row['dataset_name'],
            'dataset_family': row['dataset_family'],
            'actor': row['actor'],
        })
        atomic_json(cache_dir/f'{annotation["candidate_id"]}.json', annotation)
    if CPA_POINT_METHOD == 'sam2' and annotations:
        annotations, sam2_errors = run_cpa_sam2(
            annotations, out, resume_existing,
        )
        errors.extend(sam2_errors)
        for annotation in annotations:
            atomic_json(cache_dir/f'{annotation["candidate_id"]}.json', annotation)
    for annotation in annotations:
        draw_cpa_intermediates(annotation, intermediate_dir)
    pretrack_path = out/'cpa_contact_audit_pretracking.jsonl'
    atomic_jsonl(pretrack_path, annotations)
    tracking_path = out/'cpa_tracking_results.jsonl'
    tracking_errors_path = out/'cpa_tracking_errors.jsonl'
    tracking = []
    tracking_errors = []
    if annotations:
        tracking, tracking_errors = run_cpa_tracker(
            pretrack_path, tracking_path, tracking_errors_path,
            out/'cpa_tracker.log', resume_existing, track_workers,
        )
    else:
        atomic_jsonl(tracking_path, [])
        atomic_jsonl(tracking_errors_path, [])
        (out/'cpa_tracker.log').write_text('No accepted CPA contacts to track.\n', encoding='utf-8')
    errors.extend(tracking_errors)
    tracking_by_candidate = {item['candidate_id']: item for item in tracking}
    tracking_reviews, tracking_review_errors = review_cpa_tracking_results(
        endpoint, annotations, tracking_by_candidate,
        tracking_review_dir, tracking_review_cache_dir,
        request_workers, max_attempts, resume_existing,
    )
    errors.extend(tracking_review_errors)
    atomic_jsonl(out/'cpa_tracking_review_results.jsonl', tracking_reviews)
    atomic_json(out/'cpa_tracking_review_errors.json', tracking_review_errors)
    reviews_by_observation = {
        item['observation_id']: item for item in tracking_reviews
    }
    multiple_choice_results = []
    multiple_choice_errors = []
    if CPA_MULTIPLE_CHOICE_ENABLED:
        multiple_choice_prompts = build_cpa_observation_choice_prompts(
            annotations, tracking_by_candidate, reviews_by_observation,
        )
        if on_local_sam2_start is not None:
            on_local_sam2_start()
        multiple_choice_results, multiple_choice_errors = (
            run_cpa_observation_choice_sam2(
                multiple_choice_prompts, out, resume_existing,
                batch_size=cpa_sam2_batch_size,
            )
        )
        errors.extend(multiple_choice_errors)
    else:
        atomic_jsonl(out/'cpa_multiple_choice_sam2_prompts.jsonl', [])
        atomic_jsonl(out/'cpa_multiple_choice_sam2_results.jsonl', [])
        atomic_jsonl(out/'cpa_multiple_choice_sam2_errors.jsonl', [])
        (out/'cpa_multiple_choice_sam2.log').write_text(
            'CPA multiple-choice generation disabled.\n', encoding='utf-8',
        )
    multiple_choice_by_observation = {
        item['observation_id']: item for item in multiple_choice_results
    }
    records = build_cpa_records(
        annotations, tracking_by_candidate, reviews_by_observation, annotated_dir,
        multiple_choice_by_observation,
    )
    multiple_choice_vqa_records = flatten_cpa_multiple_choice_vqa_records(records)
    completed_annotations = [
        annotation for annotation in annotations
        if annotation['candidate_id'] in tracking_by_candidate
    ]
    atomic_jsonl(out/'cpa_records.jsonl', records)
    (out/'cpa_vqa_records.jsonl').unlink(missing_ok=True)
    atomic_jsonl(
        out/'cpa_multiple_choice_vqa_records.jsonl', multiple_choice_vqa_records,
    )
    atomic_jsonl(out/'cpa_contact_audit.jsonl', completed_annotations)
    atomic_json(out/'cpa_errors.json', errors)
    report_cpa(completed_annotations, records, out/'report_cpa.html')
    observations_expected = sum(len(audit['observations']) for audit in accepted)
    checks = {
        'all_accepted_contacts_have_vlm_cpa': len(annotations) == len(accepted),
        'all_vlm_cpa_contacts_tracked': len(completed_annotations) == len(accepted),
        'one_cpa_record_per_contact_observation': len(records) == observations_expected,
        'cpa_coordinate_question_artifact_removed': (
            not (out/'cpa_vqa_records.jsonl').exists() and
            all('participant_questions' not in record for record in records)
        ),
        'cpa_multiple_choice_questions_are_observation_local_sam2_multiselect': (
            not CPA_MULTIPLE_CHOICE_ENABLED or (
                len(multiple_choice_results) == len(
                    build_cpa_observation_choice_prompts(
                        annotations, tracking_by_candidate, reviews_by_observation,
                    )
                ) and
                len({record['id'] for record in multiple_choice_vqa_records}) ==
                len(multiple_choice_vqa_records) and
                len(multiple_choice_vqa_records) == sum(
                    len(item.get('participants') or [])
                    for item in multiple_choice_results
                ) and
                all(
                    record['answer_mode'] == 'multi_select' and
                    record['answer_value'] and
                    record.get('required_selection_count') ==
                    len(record['answer_value']) and
                    set(record['answer_value']).issubset(record['choices']) and
                    record['learner_inputs'] == [
                        'task_instruction', 'marked_precontact_observation_image',
                        'question',
                    ] and
                    record['teacher_evidence']['future_media_passed_to_learner'] is False and
                    record['multiple_choice_question_version'] ==
                    CPA_MULTIPLE_CHOICE_VQA_VERSION and
                    Path(record['image_path']).is_file()
                    for record in multiple_choice_vqa_records
                )
            )
        ),
        'robot_cpa_emits_only_object_multiselect_questions': all(
            not cpa_is_robot_data(record) or (
                len(record.get('multiple_choice_questions') or []) <= 1 and
                all(
                    question.get('tracked_point_role') == 'object' and
                    question.get('semantic_participant_role') == 'contacted_object' and
                    question.get('question_type') ==
                    'next_contact_object_point_multiple_choice'
                    for question in record.get('multiple_choice_questions') or []
                )
            )
            for record in records
        ),
        'zero_drop_rate_cpa_negatives_are_strictly_inside_participant_mask': (
            CPA_NEGATIVE_DROP_RATE != 0 or all(
                question['teacher_evidence']['negative_sampling'].get(
                    'outside_mask_count'
                ) == 0 and
                question['teacher_evidence']['negative_sampling'].get(
                    'inside_mask_count'
                ) == 3 and
                all(
                    choice.get('outside_participant_mask') is False
                    for choice in question['teacher_evidence'].get(
                        'choice_metadata', []
                    )
                    if not choice.get('is_positive')
                )
                for question in multiple_choice_vqa_records
            )
        ),
        'normalized_interaction_boxes': all(
            0 <= item['interaction_bbox_xyxy'][0] < item['interaction_bbox_xyxy'][2] <= 1 and
            0 <= item['interaction_bbox_xyxy'][1] < item['interaction_bbox_xyxy'][3] <= 1
            for item in completed_annotations
        ),
        'normalized_contact_point_pairs': all(
            1 <= len(item['contact_point_pairs_full_xy']) <= 5 and
            all(
                0 <= coordinate <= 1
                for pair in item['contact_point_pairs_full_xy']
                for role in ('hand', 'object')
                for coordinate in pair[f'{role}_point_xy']
            )
            for item in completed_annotations
        ),
        'all_contact_pairs_use_selected_method': all(
            item.get('point_pair_version') == active_cpa_point_pair_version() and
            item.get('contact_agent_role_version') == CPA_CONTACT_AGENT_ROLE_VERSION and
            item.get('contact_agent_role') in {
                'held_tool', 'human_hand', 'robot_gripper',
            } and
            item.get('point_method', 'seed_vlm') == CPA_POINT_METHOD and
            item.get('point_model') == CPA_POINT_MODEL and
            bool(item.get('points_raw_model_output')) and
            (
                CPA_POINT_METHOD != 'sam2' or
                isinstance(item.get('sam2_segmentation'), dict)
            )
            for item in completed_annotations
        ),
        'all_tracking_frames_lite_reviewed': (
            len(tracking_reviews) == observations_expected and
            all(
                item.get('tracking_review_version') == CPA_TRACKING_REVIEW_VERSION and
                item.get('tracking_review_model') == MODEL and
                bool(item.get('raw_model_output'))
                for item in tracking_reviews
            )
        ),
        'all_tracking_pairs_distance_reviewed': (
            len(tracking_reviews) == observations_expected and
            all(
                review.get('pair_separation_review_version') ==
                CPA_PAIR_SEPARATION_REVIEW_VERSION and
                review.get('pair_minimum_distance_growth_ratio') ==
                CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO and
                len(review.get('pair_distance_checks', [])) ==
                len(review.get('tracked_contact_pairs', []))
                for review in tracking_reviews
            )
        ),
        'not_above_100pct_distance_growth_pairs_removed': all(
            all(
                check['distance_growth_sufficient'] or
                (
                    pair['hand_point_xy'] is None and
                    pair['object_point_xy'] is None
                )
                for check, pair in zip(
                    review.get('pair_distance_checks', []),
                    review.get('validated_pairs', []),
                )
            )
            for review in tracking_reviews
        ),
        'invalid_tracked_points_removed_per_frame': all(
            all(
                (pair['hand_point_xy'] is not None) == pair['hand_valid'] and
                (pair['object_point_xy'] is not None) == pair['object_valid']
                for pair in review['validated_pairs']
            )
            for review in tracking_reviews
        ),
        'normalized_tracked_points': all(
            all(
                0 <= coordinate <= 1
                for point in record['answer_value']
                for coordinate in point['xy']
            )
            for record in records
        ),
        'teacher_media_excluded_from_learner': all(
            record['teacher_evidence']['future_media_passed_to_learner'] is False
            for record in records
        ),
        'all_visualizations_exist': all(
            (annotated_dir/record['annotated_image']).is_file() for record in records
        ),
        'all_selected_point_method_artifacts_exist': all(
            (
                item.get('point_method', 'seed_vlm') != 'sam2' or
                all(
                    (out/'cpa_sam2'/item['sam2_segmentation'].get(name, '')).is_file()
                    for name in (
                        'hand_mask_image', 'object_mask_image', 'mask_pair_overlay',
                    )
                )
            )
            for item in completed_annotations
        ),
        'report_has_show_intermediates_per_contact': (
            (out/'report_cpa.html').read_text(encoding='utf-8').count(
                '<details class="show-intermediates">'
            ) == len(completed_annotations)
        ),
        'report_has_no_cpa_coordinate_questions': (
            'coordinate question' not in
            (out/'report_cpa.html').read_text(encoding='utf-8').lower() and
            '<tr class="cpa-participant-question"' not in
            (out/'report_cpa.html').read_text(encoding='utf-8')
        ),
        'report_has_every_cpa_multiple_choice_question': (
            (out/'report_cpa.html').read_text(encoding='utf-8').count(
                '<section class="cpa-multiple-choice-question"'
            ) == len(multiple_choice_vqa_records)
        ),
        'report_has_one_full_width_row_per_episode': (
            (out/'report_cpa.html').read_text(encoding='utf-8').count(
                '<article class="card cpa-episode-row"'
            ) == len({
                item.get('row_id') or item['candidate_id']
                for item in completed_annotations
            })
        ),
        'report_pairs_each_visual_with_its_student_question_table': (
            (out/'report_cpa.html').read_text(encoding='utf-8').count(
                'class="observation-student-row"'
            ) == len(records) and
            'grid-template-columns:minmax(320px,0.9fr) minmax(480px,1.1fr)' in
            (out/'report_cpa.html').read_text(encoding='utf-8')
        ),
        'report_has_full_episode_video_per_contact': (
            (out/'report_cpa.html').read_text(encoding='utf-8').count(
                '<video class="cpa-episode-video"'
            ) == len(completed_annotations) and
            all(
                (out/'episode_videos'/(
                    item.get('episode_video') or f'{item["row_id"]}.mp4'
                )).is_file()
                for item in completed_annotations
            )
        ),
        'no_cpa_errors': not errors,
    }
    dataset_source = (
        completed_annotations[0] if completed_annotations else
        accepted[0] if accepted else records[0] if records else {}
    )
    dataset_name = dataset_source.get('dataset_name') or dataset_source.get('dataset')
    completion = {
        'passed': all(checks.values()), 'checks': checks,
        'mode': 'cpa', 'dataset': dataset_name, 'dataset_name': dataset_name,
        'dataset_family': dataset_source.get('dataset_family'),
        'actor': dataset_source.get('actor'),
        'model': MODEL, 'review_model': REVIEW_MODEL, 'inference_backend': BACKEND,
        'accepted_contacts': len(accepted),
        'cpa_contacts': len(completed_annotations),
        'cpa_records': len(records),
        'cpa_multiple_choice_vqa_records': len(multiple_choice_vqa_records),
        'cpa_multiple_choice_question_version': CPA_MULTIPLE_CHOICE_VQA_VERSION,
        'cpa_multiple_choice_enabled': CPA_MULTIPLE_CHOICE_ENABLED,
        'cpa_negative_a_at_720p': CPA_NEGATIVE_A_AT_720P,
        'cpa_negative_drop_rate': CPA_NEGATIVE_DROP_RATE,
        'cpa_observation_sam2_batch_size': cpa_sam2_batch_size,
        'cpa_multiple_choice_sam2_observations': len(multiple_choice_results),
        'cpa_multiple_choice_sam2_errors': multiple_choice_errors,
        'errors': errors,
        'tracker_checkpoint': str(CPA_TRACKER_CHECKPOINT),
        'point_method': CPA_POINT_METHOD,
        'point_model': CPA_POINT_MODEL,
        'point_pair_version': active_cpa_point_pair_version(),
        'tracking_review_version': CPA_TRACKING_REVIEW_VERSION,
        'tracking_review_model': MODEL,
        'pair_separation_review_version': CPA_PAIR_SEPARATION_REVIEW_VERSION,
        'pair_minimum_distance_growth_ratio': CPA_PAIR_MINIMUM_DISTANCE_GROWTH_RATIO,
        'tracking_review_frames': len(tracking_reviews),
        'tracking_review_frames_with_removal': sum(
            bool(review['removed_point_ids']) for review in tracking_reviews
        ),
        'tracking_review_empty_frames': sum(
            not review['validated_points'] for review in tracking_reviews
        ),
        'tracked_points_before_review': sum(
            len(review['tracked_contact_pairs']) * 2 for review in tracking_reviews
        ),
        'tracked_points_after_review': sum(
            len(review['validated_points']) for review in tracking_reviews
        ),
        'tracked_points_removed': sum(
            len(review['removed_point_ids']) for review in tracking_reviews
        ),
        'tracked_pairs_distance_reviewed': sum(
            len(review.get('pair_distance_checks', [])) for review in tracking_reviews
        ),
        'tracked_pairs_removed_by_distance_review': sum(
            not check['distance_growth_sufficient']
            for review in tracking_reviews
            for check in review.get('pair_distance_checks', [])
        ),
        'created_at': time.time(),
    }
    atomic_json(out/'complete_cpa.json', completion)
    return completed_annotations, records, errors


def report_sta(contact_audits: list[dict], sta_records: list[dict], out: Path) -> None:
    dataset_source = contact_audits[0] if contact_audits else (sta_records[0] if sta_records else {})
    dataset_name = str(
        dataset_source.get('dataset_name') or dataset_source.get('dataset') or 'Dataset'
    )
    human_dataset = bool(contact_audits) and all(
        audit.get('actor') == 'human' for audit in contact_audits
    )
    questions_by_contact: dict[str, list[dict]] = {}
    for record in sta_records:
        questions_by_contact.setdefault(record['contact_id'], []).append(record)
    question_order = {
        'next_contact_noun': 0,
        'next_contact_verb': 1,
        'next_contact_bbox': 2,
        'time_to_next_contact': 3,
    }
    cards = []
    for audit in sorted(contact_audits, key=lambda x: x['candidate_id']):
        status = 'accepted contact' if audit['valid_contact'] else 'rejected transition'
        compact = audit['contact'] if audit['valid_contact'] else None
        bbox = audit['object_bbox_xyxy'] if audit['valid_contact'] else None
        questions = sorted(
            questions_by_contact.get(audit['candidate_id'], []),
            key=lambda x: (
                int(x['query_frame_index']), question_order[x['question_type']],
            ),
        )
        question_rows = ''.join(
            f'''<tr class="sta-question"><td><code>{html.escape(q.get('observation_id', audit['candidate_id']))}</code><br>frame {int(q['query_frame_index'])}</td><td data-question-type="{html.escape(q['question_type'])}"><code>{html.escape(q['question_type'])}</code></td><td>{html.escape(q['question'])}</td><td><code>{html.escape(q['answer'])}</code></td></tr>'''
            for q in questions
        )
        if question_rows:
            question_html = f'''<h4>STA learner questions</h4><table><thead><tr><th>Observation</th><th>Type</th><th>Question</th><th>Answer</th></tr></thead><tbody>{question_rows}</tbody></table>'''
        else:
            question_html = '<p class="rejected">No learner question is emitted for a rejected transition.</p>'
        observations = audit.get('observations') or [{
            'observation_id': f'{audit["candidate_id"]}_obs_00',
            'query_frame_index': audit['query_frame_index'],
            'query_image': audit['query_image'],
            'time_to_contact_seconds': audit['time_to_contact_seconds'],
            'segment_progress': None,
            'is_primary_teacher_observation': True,
        }]
        observation_cards = []
        for observation in observations:
            if (
                observation['is_primary_teacher_observation'] and
                audit.get('annotated_query_image')
            ):
                observation_src = 'sta_annotated/' + audit['annotated_query_image']
                annotation_note = ' · teacher bbox'
            else:
                observation_src = 'sta_images/' + observation['query_image']
                annotation_note = (
                    ' · bbox not propagated to this moving frame'
                    if audit.get('object_bbox_xyxy') is not None else ''
                )
            ttc = observation.get('time_to_contact_seconds')
            ttc_text = f'{float(ttc):.3f}s' if isinstance(ttc, (int, float)) else 'n/a'
            observation_cards.append(f'''<figure><img src="{html.escape(observation_src)}"><figcaption><code>{html.escape(observation['observation_id'])}</code><br>frame {int(observation['query_frame_index'])} · TTC {ttc_text}{annotation_note}</figcaption></figure>''')
        episode_video = audit.get('episode_video') or f'{audit["row_id"]}.mp4'
        fps = float(audit['source_video_fps'])
        query_frames = [int(observation['query_frame_index']) for observation in observations]
        query_frames_attr = ','.join(str(frame) for frame in query_frames)
        contact_frame = int(audit['candidate_frame'])
        observation_buttons = ''.join(
            f'''<button type="button" data-seek-frame="{int(observation['query_frame_index'])}">Jump to {html.escape(observation['observation_id'].rsplit('_', 2)[-2] + '_' + observation['observation_id'].rsplit('_', 1)[-1])} · frame {int(observation['query_frame_index'])}</button>'''
            for observation in observations
        )
        cards.append(f'''<article class="card"><h3>{html.escape(audit['candidate_id'])} · {status}</h3>
<div class="media"><div><h4>Bounded anticipation-window learner observations</h4><div class="observation-grid">{''.join(observation_cards)}</div></div>
<div><h4>Teacher-only centered {audit['context_seconds_requested']:.3f} s context</h4><video controls preload="metadata"><source src="contact_clips/{html.escape(audit['teacher_clip'])}" type="video/mp4">Your browser cannot play this H.264 MP4 video.</video></div></div>
<details class="episode-preview"><summary>Full episode video with current/contact markers</summary>
<div class="episode-buttons">{observation_buttons}<button type="button" data-seek-frame="{contact_frame}">Jump to contact event {contact_frame}</button></div>
<div class="video-stage"><video controls preload="none" data-src="episode_videos/{html.escape(episode_video)}" data-fps="{fps:.9g}" data-query-frames="{query_frames_attr}" data-contact-frame="{contact_frame}"></video>
<div class="event-marker current-marker" aria-hidden="true"><span class="event-dot"></span><span>CURRENT</span></div>
<div class="event-marker contact-marker" aria-hidden="true"><span class="event-dot"></span><span>CONTACT</span></div></div>
<p class="marker-legend"><span class="legend-dot green"></span> green = learner current/query frame &nbsp; <span class="legend-dot red"></span> red = contact event frame</p></details>
<p><b>Transition:</b> frame {audit['candidate_frame']}, {html.escape(audit['transition']['from'])} → {html.escape(audit['transition']['to'])}</p>
<p><b>Model decision:</b> {html.escape(audit['model_reason'])}</p>
<pre>{html.escape(json.dumps({'contact': compact, 'object_bbox_xyxy': bbox, 'object_bbox_provenance': audit.get('object_bbox_provenance', 'contact_teacher_localization_on_primary_sta_query_image' if bbox is not None else None), 'static_grounding_record_used': audit.get('static_grounding_record_used', False), 'query_frame': audit['query_frame_index'], 'time_to_contact_seconds': audit['time_to_contact_seconds'], 'policy_overrides': audit.get('policy_overrides', [])}, ensure_ascii=False, indent=2))}</pre>{question_html}</article>''')
    accepted = sum(audit['valid_contact'] for audit in contact_audits)
    css = '''body{font-family:system-ui;margin:20px;background:#f4f5f7;color:#18202a}.summary{background:#e8f4ff;padding:12px;border-radius:8px}.question-filters{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:14px 0;padding:11px 13px;border:1px solid #b8c5d6;border-radius:9px;background:rgba(255,255,255,.96);box-shadow:0 2px 8px rgba(31,45,61,.12);backdrop-filter:blur(5px)}.question-filters strong{margin-right:3px}.question-filter{cursor:pointer;padding:7px 11px;border:1px solid #829ab1;border-radius:999px;background:white;color:#334e68;font-weight:700}.question-filter.active{border-color:#2457d6;background:#2457d6;color:white}.question-filter-count{margin-left:auto;color:#52637a;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(700px,1fr));gap:18px}.card{background:white;padding:14px;border-radius:10px;box-shadow:0 1px 5px #bbb}.media{display:grid;grid-template-columns:1.3fr 1fr;gap:10px}.media img,.media video,.video-stage video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111}.observation-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.observation-grid figure{margin:0;border:1px solid #ccd6e0;background:#f7fafc}.observation-grid figcaption{padding:5px;font-size:12px;color:#334e68}.episode-preview{margin:12px 0;padding:10px;background:#f7fafc;border:1px solid #ccd6e0;border-radius:7px}.episode-preview summary{cursor:pointer;font-weight:700}.episode-buttons{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.episode-buttons button{cursor:pointer;padding:7px 11px;border:1px solid #829ab1;border-radius:5px;background:white}.video-stage{position:relative}.event-marker{display:none;position:absolute;top:18px;right:18px;align-items:center;gap:9px;padding:7px 11px;border-radius:8px;background:rgba(0,0,0,.72);color:white;font-weight:900;font-size:18px;letter-spacing:.04em;pointer-events:none}.event-marker.visible{display:flex}.event-dot{width:38px;height:38px;border-radius:50%;box-shadow:0 0 0 4px rgba(255,255,255,.92),0 2px 9px #000}.current-marker .event-dot{background:#00d84f}.contact-marker .event-dot{background:#f40000}.marker-legend{font-size:13px;color:#334e68}.legend-dot{display:inline-block;width:14px;height:14px;border-radius:50%;vertical-align:-2px}.legend-dot.green{background:#00d84f}.legend-dot.red{background:#f40000}pre{white-space:pre-wrap;background:#f2f2f2;padding:8px}table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;border:1px solid #ccd6e0;padding:7px}th{background:#edf2f7}.rejected{color:#8a2d2d}@media(max-width:800px){.grid,.media{grid-template-columns:1fr}.question-filter-count{width:100%;margin-left:0}.event-marker{top:10px;right:10px;font-size:14px}.event-dot{width:30px;height:30px}}'''
    script = '''<script>
const questionRows = Array.from(document.querySelectorAll('.sta-question'));
const questionCount = document.querySelector('.question-filter-count');
document.querySelectorAll('.question-filter').forEach((button) => {
  button.addEventListener('click', () => {
    const enabled = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', String(enabled));
    button.classList.toggle('active', enabled);
    questionRows.forEach((row) => {
      const typeCell = row.querySelector('[data-question-type]');
      if (typeCell && typeCell.dataset.questionType === button.dataset.questionType) {
        row.hidden = !enabled;
      }
    });
    const visible = questionRows.filter((row) => !row.hidden).length;
    questionCount.textContent = `Showing ${visible} / ${questionRows.length} questions`;
  });
});
document.querySelectorAll('.episode-preview').forEach((preview) => {
  const video = preview.querySelector('video');
  const currentMarker = preview.querySelector('.current-marker');
  const contactMarker = preview.querySelector('.contact-marker');
  const fps = Number(video.dataset.fps);
  const currentFrames = video.dataset.queryFrames.split(',').map(Number);
  const contactFrame = Number(video.dataset.contactFrame);
  let animationFrame = null;

  function loadVideo() {
    if (!video.getAttribute('src')) {
      video.src = video.dataset.src;
      video.load();
    }
  }
  function updateMarkers() {
    const frame = Math.round(video.currentTime * fps);
    const atCurrent = currentFrames.some((currentFrame) => Math.abs(frame - currentFrame) <= 2);
    const atContact = Math.abs(frame - contactFrame) <= 2;
    currentMarker.classList.toggle('visible', atCurrent && !atContact);
    contactMarker.classList.toggle('visible', atContact);
    if (!video.paused && !video.ended) {
      animationFrame = requestAnimationFrame(updateMarkers);
    }
  }
  function seekToFrame(frame) {
    loadVideo();
    const seek = () => {
      video.pause();
      video.currentTime = frame / fps;
      updateMarkers();
    };
    if (video.readyState >= 1) seek();
    else video.addEventListener('loadedmetadata', seek, {once: true});
  }

  preview.addEventListener('toggle', () => { if (preview.open) loadVideo(); });
  preview.querySelectorAll('[data-seek-frame]').forEach((button) => {
    button.addEventListener('click', () => seekToFrame(Number(button.dataset.seekFrame)));
  });
  video.addEventListener('play', () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    updateMarkers();
  });
  ['pause', 'seeking', 'seeked', 'timeupdate', 'loadedmetadata'].forEach((eventName) => {
    video.addEventListener(eventName, updateMarkers);
  });
});
</script>'''
    observation_count = sum(len(audit.get('observations') or [None]) for audit in contact_audits)
    if human_dataset:
        description = '''Each intentional human hand/object contact is first localized in a downsampled whole-episode teacher survey and then refined in a short labeled clip. Observations are sampled at 0%, 25%, 50%, and 75% only between the bounded pre-contact query and the frame immediately before contact. Expand the full-episode preview to see green learner-observation markers and the red contact marker. Noun, verb, and TTC are shared or computed per observation. The primary-query bbox is reused directly from static grounding and is never copied across moving first-person frames.'''
        title = f'{dataset_name} STA question and contact report'
    else:
        description = '''Every hysteretic gripper transition is reviewed. Observations are sampled at 0%, 25%, 50%, and 75% only inside the bounded interval from the original approximately one-second query through the frame immediately before contact; no expanded observation may precede that primary query or exceed its TTC. Browser-compatible H.264 clips centered on the candidate are privileged teacher evidence. Expand the full-episode preview to see a large green marker at every sampled learner observation and a large red marker at the contact event. Noun, verb, and TTC labels are shared or computed per observation. A bbox is emitted only for the original observation actually localized by the teacher, never copied across wrist-camera frames.'''
        title = f'{dataset_name} STA question and contact report'
    question_labels = {
        'next_contact_noun': 'Contact noun',
        'next_contact_verb': 'Contact verb',
        'next_contact_bbox': 'Contact bbox',
        'time_to_next_contact': 'Time to contact',
    }
    filter_buttons = ''.join(
        f'''<button type="button" class="question-filter active" data-question-type="{question_type}" aria-pressed="true">{label} ({sum(record['question_type'] == question_type for record in sta_records)})</button>'''
        for question_type, label in question_labels.items()
    )
    filters = f'''<nav class="question-filters" aria-label="STA question filters"><strong>Question filters</strong>{filter_buttons}<span class="question-filter-count">Showing {len(sta_records)} / {len(sta_records)} questions</span></nav>'''
    header = f'''<h1>{html.escape(title)}</h1><p class="summary">Candidates: {len(contact_audits)} · accepted contacts: {accepted} · bounded-window observations: {observation_count} · learner questions: {len(sta_records)}</p><p>{description}</p>{filters}'''
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(dataset_name)} STA report</title><style>{css}</style>{header}<section class="grid">' + ''.join(cards) + f'</section>{script}', encoding='utf-8')


def generate_base_annotations(
    base_rows: list[dict],
    cached_records: dict[str, dict],
    row_by_id: dict[str, dict],
    endpoint: str,
    ecot_context: Path,
    max_attempts: int,
    request_workers: int,
    record_cache_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Generate ECoT/Grounding records within one bounded API request pool."""
    base_row_ids = {row['id'] for row in base_rows}
    records = [
        record for record_id, record in cached_records.items()
        if record_id in base_row_ids
    ]
    errors = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=request_workers,
    ) as executor:
        jobs = {
            executor.submit(
                annotate, row, endpoint, ecot_context, max_attempts,
            ): row
            for row in base_rows if row['id'] not in cached_records
        }
        for future in concurrent.futures.as_completed(jobs):
            row = jobs[future]
            try:
                record = future.result()
                records.append(record)
                atomic_json(record_cache_dir/f'{record["id"]}.json', record)
            except Exception as error:
                errors.append({
                    'id': row['id'], 'kind': row['kind'],
                    'error': f'{type(error).__name__}:{error}',
                })
            completed = len(records) + len(errors)
            print(
                f'base_progress completed={completed}/{len(base_rows)} '
                f'ecot={sum(record["kind"] == "ecot" for record in records)} '
                f'errors={len(errors)}',
                flush=True,
            )
    records.sort(key=lambda item: item['id'])
    # Cached records can predate explicit dataset display names. Refresh their
    # run identity without repeating a model request.
    for record in records:
        source_id = record.get('parent_episode_id') or record['id']
        source_row = row_by_id[source_id]
        record.update({
            'dataset': source_row['dataset'],
            'dataset_name': source_row['dataset_name'],
            'dataset_family': source_row['dataset_family'],
            'actor': source_row['actor'],
        })
        atomic_json(record_cache_dir/f'{record["id"]}.json', record)
    return records, errors


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Generate complete Robot STA and CPA annotations.',
    )
    p.add_argument(
        '--api', type=normalize_api_provider, choices=tuple(API_PROVIDERS),
        default=API_PROVIDER,
        help='API provider; selects provider-specific endpoint/model defaults.',
    )
    p.add_argument('--endpoint', help='Override the selected provider endpoint.')
    p.add_argument('--model', help='Override the selected provider model.')
    p.add_argument(
        '--api-key-env', default=API_KEY_ENV,
        help='Environment variable containing the provider API key.',
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument(
        '--dataset-name',
        help=(
            'Explicit dataset display name stored in records and shown by the '
            'report hub; inferred from run_config.json for append/resume runs.'
        ),
    )
    p.add_argument(
        '--dataset-path', type=nonempty_path,
        help=(
            'Robot dataset root containing task directories '
            f'(default: {DEFAULT_DATASET_PATH}).'
        ),
    )
    p.add_argument('--selection', type=Path)
    p.add_argument(
        '--episodes', type=parse_episode_count,
        help='Number of distinct source episodes, or all (default: 50).',
    )
    p.add_argument(
        '--append-episodes', type=parse_episode_count,
        help=(
            'Append this many previously unselected episodes to an existing '
            'run, or append all remaining eligible episodes.'
        ),
    )
    p.add_argument(
        '--seed', type=int,
        help='Selection seed (default: 260820; inherited when appending).',
    )
    p.add_argument(
        '--request-workers', '--workers', dest='request_workers', type=int,
        default=8,
        help='Concurrent VLM/API requests, including Lite tracking review.',
    )
    p.add_argument(
        '--track-workers', type=int,
        help=(
            'CoTracker inference batch size (default: request worker count for '
            'backward compatibility).'
        ),
    )
    p.add_argument(
        '--cpa-sam2-batch-size', type=int,
        default=CPA_OBSERVATION_SAM2_BATCH_SIZE,
        help=(
            'Observation images per local SAM2 image-encoder batch '
            f'(default: {CPA_OBSERVATION_SAM2_BATCH_SIZE}).'
        ),
    )
    p.add_argument('--contact-max-attempts', type=int, default=2)
    p.add_argument('--base-max-attempts', type=int, default=3)
    p.add_argument('--sta-context-seconds', type=float, default=STA_CONTEXT_SECONDS)
    p.add_argument('--sta-anticipation-seconds', type=float, default=STA_ANTICIPATION_SECONDS)
    p.add_argument(
        '--cpa-point-method', choices=CPA_POINT_METHODS,
        help=(
            'CPA source-point method: direct Seed VLM pairs or Lite-guided '
            'SAM2 masks; inherited when appending.'
        ),
    )
    p.add_argument(
        '--cpa-negative-a', type=float,
        help=(
            'CPA negative-sampling exclusion radius a in pixels at 720p; '
            'scaled linearly with observation height '
            f'(default: {CPA_NEGATIVE_A_AT_720P:g}).'
        ),
    )
    p.add_argument(
        '--drop-rate', type=float,
        help=(
            'Probability that each CPA negative is sampled outside the '
            'hand/tool/gripper or object SAM2 mask; both domains use f(r) '
            f'(default: {CPA_NEGATIVE_DROP_RATE:g}).'
        ),
    )
    p.add_argument('--resume-existing', action='store_true')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--sta', action='store_true', help='Generate only STA annotations.')
    modes.add_argument('--cpa', action='store_true', help='Generate only contact point anticipation annotations.')
    return p


def requested_annotation_mode(args: argparse.Namespace) -> str:
    selected = [mode for mode in ANNOTATION_MODES if getattr(args, mode, False)]
    return selected[0] if selected else 'all'


def enabled_annotation_tasks(mode: str) -> dict[str, bool]:
    """Return exact task switches; a single-task mode enables only itself."""
    if mode not in ('all', *ANNOTATION_MODES):
        raise ValueError(f'unknown_annotation_mode:{mode}')
    return {
        'grounding': False,
        'ecot': False,
        'sta': mode in ('all', 'sta'),
        'cpa': mode in ('all', 'cpa'),
    }


def main() -> None:
    global CPA_POINT_METHOD, CPA_NEGATIVE_A_AT_720P, CPA_NEGATIVE_DROP_RATE
    p = build_argument_parser()
    args = p.parse_args()
    try:
        args.endpoint = configure_api(
            args.api, args.endpoint, args.model, args.api_key_env,
        )
    except ValueError as error:
        p.error(str(error))
    out = args.out
    run_config_path = out/'run_config.json'
    previous_config = {}
    if run_config_path.is_file() and (args.resume_existing or args.append_episodes):
        previous_config = json.loads(run_config_path.read_text(encoding='utf-8'))
    if args.append_episodes:
        args.resume_existing = True
        if not previous_config:
            p.error('--append-episodes requires an existing run_config.json')
        if args.selection:
            p.error('--append-episodes cannot be combined with --selection')
        if args.episodes is not None:
            p.error('--append-episodes cannot be combined with --episodes')

    supplied_dataset_name = str(args.dataset_name or '').strip()
    previous_name = str(
        previous_config.get('dataset_name') or previous_config.get('dataset') or ''
    ).strip()
    args.dataset_name = supplied_dataset_name or previous_name
    if not args.dataset_name:
        p.error('--dataset-name must not be empty for a new run')
    if supplied_dataset_name and previous_name and supplied_dataset_name != previous_name:
        p.error(
            '--dataset-name does not match the existing run: '
            f'{supplied_dataset_name!r} != {previous_name!r}'
        )

    previous_dataset_path = previous_config.get('dataset_root')
    requested_dataset_path = args.dataset_path
    args.dataset_path = Path(
        requested_dataset_path or previous_dataset_path or DEFAULT_DATASET_PATH
    ).expanduser().resolve()
    if not args.dataset_path.is_dir():
        p.error(f'--dataset-path is not a directory: {args.dataset_path}')
    if (
        requested_dataset_path is not None and previous_dataset_path and
        Path(previous_dataset_path).expanduser().resolve() != args.dataset_path
    ):
        p.error(
            '--dataset-path does not match the existing run: '
            f'{args.dataset_path} != '
            f'{Path(previous_dataset_path).expanduser().resolve()}'
        )

    previous_seed = previous_config.get('seed')
    args.seed = int(
        args.seed if args.seed is not None else
        previous_seed if previous_seed is not None else 260820
    )
    previous_model = str(previous_config.get('model') or '').strip()
    if args.append_episodes and previous_model and previous_model != MODEL:
        p.error(
            'ROBOT_SERVED_MODEL does not match the existing run: '
            f'{MODEL} != {previous_model}'
        )
    explicit_mode = any(getattr(args, name, False) for name in ANNOTATION_MODES)
    mode = requested_annotation_mode(args)
    previous_mode = str(previous_config.get('mode') or '').strip()
    if args.append_episodes and previous_mode in ('all', *ANNOTATION_MODES):
        if explicit_mode and mode != previous_mode:
            p.error(
                '--append-episodes annotation mode does not match the existing run: '
                f'{mode} != {previous_mode}'
            )
        if not explicit_mode:
            mode = previous_mode

    previous_point_method = previous_config.get('cpa_point_method')
    if (
        args.append_episodes and args.cpa_point_method and
        previous_point_method and args.cpa_point_method != previous_point_method
    ):
        p.error(
            '--cpa-point-method does not match the existing run: '
            f'{args.cpa_point_method} != {previous_point_method}'
        )
    CPA_POINT_METHOD = (
        args.cpa_point_method or previous_point_method or CPA_POINT_METHOD
    )
    previous_negative_a = previous_config.get('cpa_negative_a_at_720p')
    previous_drop_rate = previous_config.get('cpa_negative_drop_rate')
    if (
        args.append_episodes and args.cpa_negative_a is not None and
        previous_negative_a is not None and
        float(args.cpa_negative_a) != float(previous_negative_a)
    ):
        p.error(
            '--cpa-negative-a does not match the existing run: '
            f'{args.cpa_negative_a} != {previous_negative_a}'
        )
    if (
        args.append_episodes and args.drop_rate is not None and
        previous_drop_rate is not None and
        float(args.drop_rate) != float(previous_drop_rate)
    ):
        p.error(
            '--drop-rate does not match the existing run: '
            f'{args.drop_rate} != {previous_drop_rate}'
        )
    try:
        CPA_NEGATIVE_A_AT_720P = validate_cpa_negative_a(
            args.cpa_negative_a
            if args.cpa_negative_a is not None else
            previous_negative_a
            if previous_negative_a is not None else CPA_NEGATIVE_A_AT_720P
        )
        CPA_NEGATIVE_DROP_RATE = validate_cpa_negative_drop_rate(
            args.drop_rate
            if args.drop_rate is not None else
            previous_drop_rate
            if previous_drop_rate is not None else CPA_NEGATIVE_DROP_RATE
        )
    except ValueError as error:
        p.error(str(error))
    prepare_request(args.endpoint, {'messages': []})
    if args.track_workers is None:
        args.track_workers = args.request_workers
    if (
        args.request_workers < 1 or args.track_workers < 1 or
        args.cpa_sam2_batch_size < 1 or
        args.contact_max_attempts < 1 or args.base_max_attempts < 1
    ):
        p.error(
            'request-workers, track-workers, cpa-sam2-batch-size, '
            'contact-max-attempts, and base-max-attempts must be positive'
        )
    if args.sta_context_seconds <= 0 or args.sta_anticipation_seconds <= 0:
        p.error('STA durations must be positive')

    enabled_tasks = enabled_annotation_tasks(mode)
    run_grounding = enabled_tasks['grounding']
    run_ecot = enabled_tasks['ecot']
    run_sta = enabled_tasks['sta']
    run_cpa = enabled_tasks['cpa']
    run_contact_pipeline = run_sta or run_cpa
    images, annotated = out/'images', out/'annotated'
    sta_images, sta_annotated = out/'sta_images', out/'sta_annotated'
    contact_clips, contacts_dir = out/'contact_clips', out/'contacts'
    episode_videos = out/'episode_videos'
    ecot_context = out/'ecot_context'
    record_cache_dir = out/'record_cache'
    for directory in (
        images, annotated, sta_images, sta_annotated, contact_clips, contacts_dir,
        episode_videos, ecot_context, record_cache_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    selection_path = out/'selection.json'
    append_request = args.append_episodes
    episodes_before_append = 0
    episodes_appended = 0
    eligible_episode_count = None
    if append_request:
        if not selection_path.is_file():
            p.error('--append-episodes requires an existing selection.json')
        existing_rows = load_samples(
            args.seed, selection_path, None, args.dataset_path,
        )
        episodes_before_append = len(existing_rows)
        previous_episodes = previous_config.get('episodes_requested')
        if (
            previous_episodes is not None and
            int(previous_episodes) != episodes_before_append
        ):
            p.error(
                'existing selection count does not match run_config.json: '
                f'{episodes_before_append} != {previous_episodes}'
            )
        excluded = {
            episode_identity(row['task_dir'], row['episode_index'])
            for row in existing_rows
        }
        eligible_candidates = discover_eligible_episodes(args.dataset_path, args.seed)
        eligible_episode_count = len(eligible_candidates)
        remaining_candidates = [
            candidate for candidate in eligible_candidates
            if episode_identity(candidate[0], candidate[1]) not in excluded
        ]
        if append_request == 'all':
            selected_candidates = remaining_candidates
        else:
            requested_append_count = int(append_request)
            if len(remaining_candidates) < requested_append_count:
                p.error(
                    'not enough unannotated episodes to append: '
                    f'{len(remaining_candidates)} < {requested_append_count}'
                )
            selected_candidates = remaining_candidates[:requested_append_count]
        new_rows = build_sample_rows(
            selected_candidates, args.seed, start_index=episodes_before_append,
        )
        rows = existing_rows + new_rows
        episodes_appended = len(new_rows)
        args.episodes = len(rows)
    else:
        previous_episodes = previous_config.get('episodes_requested')
        if args.episodes is None:
            args.episodes = (
                int(previous_episodes)
                if args.resume_existing and previous_episodes is not None else 50
            )
        selection_input = args.selection
        if selection_input is None and args.resume_existing and selection_path.is_file():
            selection_input = selection_path
        if selection_input is not None:
            expected_count = None if args.episodes == 'all' else int(args.episodes)
            rows = load_samples(
                args.seed, selection_input, expected_count, args.dataset_path,
            )
            if args.episodes == 'all':
                eligible_candidates = discover_eligible_episodes(
                    args.dataset_path, args.seed,
                )
                eligible_episode_count = len(eligible_candidates)
                selected_identities = {
                    episode_identity(row['task_dir'], row['episode_index'])
                    for row in rows
                }
                eligible_identities = {
                    episode_identity(candidate[0], candidate[1])
                    for candidate in eligible_candidates
                }
                if selected_identities != eligible_identities:
                    p.error(
                        '--episodes all with --selection requires exactly every '
                        'eligible dataset episode'
                    )
        else:
            rows = select_samples(
                args.seed, args.episodes, args.dataset_path,
            )
            if args.episodes == 'all':
                eligible_episode_count = len(rows)
        args.episodes = len(rows)
        if (
            args.resume_existing and previous_episodes is not None and
            int(previous_episodes) != args.episodes
        ):
            p.error(
                '--episodes does not match the existing run: '
                f'{args.episodes} != {previous_episodes}'
            )

    append_history = list(previous_config.get('append_history') or [])
    now = time.time()
    if append_request:
        append_history.append({
            'requested': append_request,
            'episodes_before': episodes_before_append,
            'episodes_appended': episodes_appended,
            'episodes_after': args.episodes,
            'timestamp': now,
        })
    for row in rows:
        row['dataset'] = args.dataset_name
        row['dataset_name'] = args.dataset_name
        row['dataset_family'] = 'robot'
        row['actor'] = 'robot'
    # Freeze the expanded ordered selection before any lengthy media or VLM
    # work so an interrupted append can resume the exact same new episode IDs.
    atomic_json(selection_path, rows)
    atomic_json(run_config_path, {
        'dataset': args.dataset_name,
        'dataset_name': args.dataset_name,
        'dataset_family': 'robot',
        'dataset_root': str(args.dataset_path),
        'actor': 'robot',
        'mode': mode,
        'episodes_requested': args.episodes,
        'episode_annotation_policy': 'append_safe_all_selected_episodes_v3',
        'eligible_dataset_episodes': eligible_episode_count,
        'last_append_request': append_request,
        'last_append_episode_count': episodes_appended,
        'append_history': append_history,
        'seed': args.seed,
        'model': MODEL,
        'api_provider': API_PROVIDER,
        'api_endpoint': args.endpoint,
        'api_key_env': API_KEY_ENV,
        'cpa_point_method': CPA_POINT_METHOD,
        'cpa_negative_a_at_720p': CPA_NEGATIVE_A_AT_720P,
        'cpa_negative_drop_rate': CPA_NEGATIVE_DROP_RATE,
        'request_workers': args.request_workers,
        'track_workers': args.track_workers,
        'cpa_sam2_batch_size': args.cpa_sam2_batch_size,
        'created_at': previous_config.get('created_at', now),
        'updated_at': now,
    })
    archive_legacy_artifact(out/'records.jsonl', 'records_combined_legacy.jsonl')
    cached_contact_audits: dict[str, dict] = {}
    cached_records: dict[str, dict] = {}
    if args.resume_existing:
        contact_cache = out/'contact_audit.jsonl'
        if run_contact_pipeline and contact_cache.is_file():
            cached_contact_audits = {
                record['candidate_id']: record
                for line in contact_cache.read_text(encoding='utf-8').splitlines()
                if line.strip() for record in [json.loads(line)]
            }
        if run_grounding or run_ecot:
            cached_records = {
                record['id']: normalize_ecot_vocabulary(record)
                for record in read_annotation_records(out)
                if (
                    record.get('kind') != 'cot' or
                    record.get('question_type') == 'structured_ecot'
                )
            }
            for cache_path in sorted(record_cache_dir.glob('*.json')):
                record = json.loads(cache_path.read_text(encoding='utf-8'))
                if (
                    record.get('kind') != 'cot' or
                    record.get('question_type') == 'structured_ecot'
                ):
                    normalize_ecot_vocabulary(record)
                    cached_records[record['id']] = record
    row_by_id = {row['id']: row for row in rows}
    for row in rows:
        if run_contact_pipeline or run_ecot:
            row['gripper_segmentation'] = gripper_segments(row, args.seed)
        row['source_video'] = str(video_path(row))
        if run_contact_pipeline or run_ecot:
            prepare_episode_preview(
                Path(row['source_video']), episode_videos/f'{row["id"]}.mp4',
            )

    base_rows = []
    if run_grounding:
        grounding_rows = build_grounding_rows(rows)
        for row in grounding_rows:
            extract_media(row, images, args.seed, args.resume_existing)
        base_rows.extend(grounding_rows)
    if run_ecot:
        ecot_rows = build_ecot_rows(rows, args.seed)
        for row in ecot_rows:
            extract_media(row, images, args.seed, args.resume_existing)
        base_rows.extend(ecot_rows)
        atomic_json(out/'ecot_selection.json', ecot_rows)

    all_candidates = []
    if run_contact_pipeline:
        for row in rows:
            row['sta_candidates'] = []
            for transition_index, _ in enumerate(row['gripper_segmentation']['transitions']):
                candidate = extract_sta_candidate_media(
                    row, transition_index, sta_images, contact_clips,
                    args.sta_context_seconds, args.sta_anticipation_seconds,
                    require_object_bbox=run_sta,
                    resume_existing=args.resume_existing,
                )
                row['sta_candidates'].append(candidate)
                all_candidates.append(candidate)
    atomic_json(out/'selection.json', rows)
    print(
        f'media_ready mode={mode} episodes={len(rows)} '
        f'base_samples={len(base_rows)} contact_candidates={len(all_candidates)}',
        flush=True,
    )

    candidate_ids = {candidate['candidate_id'] for candidate in all_candidates}
    candidates_by_id = {
        candidate['candidate_id']: candidate for candidate in all_candidates
    }
    contact_audits = [
        audit for candidate_id, audit in cached_contact_audits.items()
        if candidate_id in candidate_ids
    ]
    contact_errors = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.request_workers,
    ) as ex:
        jobs = {
            ex.submit(
                contact_request, args.endpoint, row_by_id[candidate['row_id']],
                candidate, args.contact_max_attempts,
            ): candidate
            for candidate in all_candidates
            if candidate['candidate_id'] not in cached_contact_audits
        }
        for future in concurrent.futures.as_completed(jobs):
            candidate = jobs[future]
            try:
                contact_audits.append(future.result())
            except Exception as error:
                contact_errors.append({
                    'id': candidate['candidate_id'], 'kind': 'sta_contact',
                    'error': f'{type(error).__name__}:{error}',
                })
            completed = len(contact_audits) + len(contact_errors)
            print(
                f'contact_progress completed={completed}/{len(all_candidates)} '
                f'errors={len(contact_errors)}',
                flush=True,
            )
    contact_audits.sort(key=lambda x: x['candidate_id'])
    for audit in contact_audits:
        candidate = candidates_by_id[audit['candidate_id']]
        source_row = row_by_id[audit['row_id']]
        audit.update({
            'dataset': source_row['dataset'],
            'dataset_name': source_row['dataset_name'],
            'dataset_family': source_row['dataset_family'],
            'actor': source_row['actor'],
        })
        for key in (
            'segment_start_frame', 'segment_end_frame_exclusive',
            'observation_window_start_frame',
            'observation_window_end_frame_exclusive',
            'anticipation_seconds_requested', 'observations',
        ):
            audit[key] = candidate[key]
        audit.setdefault('episode_video', f'{audit["row_id"]}.mp4')
    contact_errors.sort(key=lambda x: x['id'])
    audits_by_row = {row['id']: [] for row in rows}
    for audit in contact_audits:
        audits_by_row[audit['row_id']].append(audit)
        if run_sta:
            draw_sta_bbox_annotation(audit, sta_annotated)

    contact_summaries = []
    if run_contact_pipeline:
        for row in rows:
            accepted = [a['contact'] for a in audits_by_row[row['id']] if a['valid_contact']]
            accepted.sort(key=lambda x: x['time'])
            atomic_json(contacts_dir/f'{row["id"]}.json', accepted)
            contact_summaries.append({
                'id': row['id'], 'source_video': row['source_video'],
                'episode_index': row['episode_index'], 'contacts': accepted,
            })
        atomic_jsonl(out/'contacts.jsonl', contact_summaries)
        atomic_jsonl(out/'contact_audit.jsonl', contact_audits)

    cpa_audits = []
    cpa_records = []
    cpa_errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as stage_executor:
        base_future = None

        def start_base_stage() -> None:
            nonlocal base_future
            if base_future is None:
                print(
                    'base_stage_start request_pool=base_only '
                    f'request_workers={args.request_workers}',
                    flush=True,
                )
                base_future = stage_executor.submit(
                    generate_base_annotations,
                    base_rows, cached_records, row_by_id, args.endpoint,
                    ecot_context, args.base_max_attempts, args.request_workers,
                    record_cache_dir,
                )

        if run_cpa:
            cpa_audits, cpa_records, cpa_errors = generate_cpa_dataset(
                args.endpoint, row_by_id, contact_audits, out,
                args.request_workers, args.track_workers,
                args.base_max_attempts, args.resume_existing,
                args.cpa_sam2_batch_size, start_base_stage,
            )
        start_base_stage()
        records, base_errors = base_future.result()
    errors = sorted(contact_errors + base_errors + cpa_errors, key=lambda x: x['id'])
    atomic_json(out/f'errors_{mode}.json', errors)
    if mode == 'all' or not (out/'errors.json').exists():
        atomic_json(out/'errors.json', errors)
    for record in records:
        if record['kind'] == 'grounding':
            draw_grounding_annotations(record, images, annotated)
    counts = {k: sum(r['kind'] == k for r in records) for k in ('grounding', 'ecot')}
    types = sorted({r['question_type'] for r in records if r['kind'] == 'grounding'})
    requested_types = sorted({
        r['model_inventory']['requested_question_type']
        for r in records if r['kind'] == 'grounding'
    })
    requested_base_kinds = {
        kind for kind, enabled in (('grounding', run_grounding), ('ecot', run_ecot))
        if enabled
    }
    if requested_base_kinds:
        write_annotation_records(out, records, requested_base_kinds)
    sta_records = []
    if run_sta:
        for row in rows:
            sta_records.extend(build_sta_records(row, audits_by_row[row['id']]))
    sta_records.sort(key=lambda x: x['id'])
    if run_sta:
        atomic_jsonl(out/'sta_records.jsonl', sta_records)
    if run_grounding:
        report_grounding(records, out/'report_grounding.html')
    if run_ecot:
        report_ecot(records, out/'report_ecot.html')
    if run_sta:
        report_sta(contact_audits, sta_records, out/'report_sta.html')
        # Keep the previous filename as a compatibility alias containing the same report.
        report_sta(contact_audits, sta_records, out/'sta_report.html')
    grounding = [r for r in records if r['kind'] == 'grounding']
    ecot = [r for r in records if r['kind'] == 'ecot']
    normalized = all(
        0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and
        obj['center_xy'] == [round((x1+x2)/2, 6), round((y1+y2)/2, 6)]
        for r in grounding for obj in r['model_inventory']['objects']
        for x1,y1,x2,y2 in [obj['bbox_xyxy']]
    )
    relation_geometry = all(
        (r['question_type'] != 'neighbor_name' or (r['model_inventory']['relation_audit'] or {}).get('accepted') is True) and
        (r['question_type'] != 'direction_relation' or (r['model_inventory']['relation_audit'] or {}).get('full_box_relation') is True)
        for r in grounding
    )
    ecot_contract_valid = all(
        set(r['ecot_methods']) == {'structured_three_part_ecot'} and
        r['question_type'] == 'structured_ecot' and
        is_concise_sentence(
            r['ecot_methods']['structured_three_part_ecot']['structured_ecot']['scene_description'], 35,
        ) and
        is_concise_sentence(
            r['ecot_methods']['structured_three_part_ecot']['structured_ecot']['task_progress_assessment'], 30,
        ) and
        r['ecot_methods']['structured_three_part_ecot']['structured_ecot']['task_progress_assessment'].endswith(
            ('Task complete.', 'Task not yet complete.'),
        ) and
        is_valid_ecot_subtask(
            r['ecot_methods']['structured_three_part_ecot']['structured_ecot']['subtask'],
            r['ecot_methods']['structured_three_part_ecot']['structured_ecot']['task_progress_assessment'],
        ) and
        r['ecot_methods']['structured_three_part_ecot']['future_media_passed_to_training_model'] is False and
        r['annotation_context']['privileged_context_excluded_from_training_inputs'] is True and
        r['annotation_context']['ecot_system_prompt'] == ECOT_SYSTEM_PROMPT and
        r['annotation_context']['progress_system_prompt'] in ECOT_PROGRESS_SYSTEM_PROMPT_VERSIONS and
        bool(r['annotation_context']['target_frame_progress_summary']) and
        all(Path(frame['image_path']).is_file() for frame in r['annotation_context']['sampled_frames']) and
        r['gripper_segmentation']['source'] == 'action.left_gripper' and
        (
            (
                r.get('ecot_sample_type', ECOT_PRIMARY_SAMPLE_TYPE) == ECOT_PRIMARY_SAMPLE_TYPE and
                r['selected_frame_index'] == r['gripper_segmentation']['selected_segment']['start_frame']
            ) or (
                r.get('ecot_sample_type') == ECOT_MIDPOINT_SAMPLE_TYPE and
                r['gripper_segmentation']['selected_segment']['start_frame'] <
                r['selected_frame_index'] <
                r['gripper_segmentation']['selected_segment']['end_frame_exclusive'] and
                r['gripper_segmentation']['selected_segment']['segment_index'] > 0
            )
        ) and
        r['gripper_segmentation']['selected_segment']['ends_with_change'] is True and
        r['gripper_segmentation']['selected_segment']['end_frame_exclusive'] == r['gripper_segmentation']['selected_segment']['next_gripper_change']['frame']
        for r in ecot
    )
    accepted_audits = [a for a in contact_audits if a['valid_contact']]
    accepted_grounding = [a for a in accepted_audits if a['grounding_episode']]
    accepted_observations = sum(
        len(audit['observations']) for audit in accepted_audits
    )
    sta_type_counts = {
        key: sum(r['question_type'] == key for r in sta_records)
        for key in ('next_contact_noun', 'next_contact_verb', 'next_contact_bbox', 'time_to_next_contact')
    }
    compact_contacts_valid = all(
        set(contact) == {'time', 'verb', 'noun'} and
        isinstance(contact['time'], int) and bool(contact['verb']) and bool(contact['noun'])
        for summary in contact_summaries for contact in summary['contacts']
    )
    per_episode_contacts_match_audits = all(
        summary['contacts'] == sorted(
            [audit['contact'] for audit in audits_by_row[summary['id']] if audit['valid_contact']],
            key=lambda contact: contact['time'],
        )
        for summary in contact_summaries
    )
    episodes_without_valid_contact = [
        summary['id'] for summary in contact_summaries if not summary['contacts']
    ]
    sta_bbox_valid = all(
        len(a['object_bbox_xyxy']) == 4 and
        0 <= a['object_bbox_xyxy'][0] < a['object_bbox_xyxy'][2] <= 1 and
        0 <= a['object_bbox_xyxy'][1] < a['object_bbox_xyxy'][3] <= 1
        for a in accepted_grounding
    )
    sta_temporal_valid = all(
        r['query_frame_index'] < r['next_contact']['time'] and
        r['teacher_evidence']['future_media_passed_to_learner'] is False
        for r in sta_records
    )
    sta_records_by_contact: dict[str, list[dict]] = {}
    sta_records_by_observation: dict[str, list[dict]] = {}
    for sta_record in sta_records:
        sta_records_by_contact.setdefault(sta_record['contact_id'], []).append(sta_record)
        sta_records_by_observation.setdefault(
            sta_record['observation_id'], [],
        ).append(sta_record)
    sta_question_groups_valid = all(
        {r['question_type'] for r in sta_records_by_observation.get(
            observation['observation_id'], []
        )} == (
            {'next_contact_noun', 'next_contact_verb', 'time_to_next_contact'} |
            ({'next_contact_bbox'} if (
                audit['grounding_episode'] and
                observation['is_primary_teacher_observation']
            ) else set())
        )
        for audit in accepted_audits
        for observation in audit['observations']
    ) and all(
        not sta_records_by_contact.get(a['candidate_id'])
        for a in contact_audits if not a['valid_contact']
    )
    report_sta_text = (
        (out/'report_sta.html').read_text(encoding='utf-8') if run_sta else ''
    )
    sta_report_alias_text = (
        (out/'sta_report.html').read_text(encoding='utf-8') if run_sta else ''
    )
    grounding_report_text = (
        (out/'report_grounding.html').read_text(encoding='utf-8') if run_grounding else ''
    )
    ecot_report_text = (
        (out/'report_ecot.html').read_text(encoding='utf-8') if run_ecot else ''
    )
    expected_counts = {
        'grounding': args.episodes if run_grounding else 0,
        'ecot': 2 * args.episodes if run_ecot else 0,
    }
    primary_ecot = [
        record for record in ecot
        if record.get('ecot_sample_type', ECOT_PRIMARY_SAMPLE_TYPE) == ECOT_PRIMARY_SAMPLE_TYPE
    ]
    midpoint_ecot = [
        record for record in ecot
        if record.get('ecot_sample_type') == ECOT_MIDPOINT_SAMPLE_TYPE
    ]
    checks = {
        'requested_mode': mode in ('all', *ANNOTATION_MODES),
        'exact_base_records': len(records) == sum(expected_counts.values()),
        'no_errors': not errors,
        'split': counts == expected_counts,
        'final_record_files_are_physically_separated': (
            (not run_grounding or (out/'grounding_records.jsonl').is_file()) and
            (not run_ecot or (out/'ecot_records.jsonl').is_file()) and
            not (out/'records.jsonl').exists()
        ),
        'record_ids_unique': len({r['id'] for r in records}) == len(records),
        'source_episode_selection_unique': (
            len(rows) == args.episodes and
            len({(r['task_dir'], r['episode_index']) for r in rows}) == args.episodes
        ),
        'first_manipulated_object_grounding_removed': (
            'task_first_object_location' not in GROUNDING_TYPES and
            all(r['model_inventory']['requested_question_type'] != 'task_first_object_location' for r in grounding)
        ),
        'all_grounding_types_requested': (
            not run_grounding or set(requested_types) == {
                GROUNDING_TYPES[index % len(GROUNDING_TYPES)]
                for index in range(args.episodes)
            }
        ),
        'no_forbidden_scene_or_robot_objects': all(not forbidden_object_name(o['name']) and not forbidden_object_name(o.get('alternate_name')) for r in grounding for o in r['model_inventory']['objects']),
        'normalized_coordinates_and_recomputed_centers': normalized,
        'relation_geometry_contract': relation_geometry,
        'annotated_grounding_images': (
            not run_grounding or all((annotated/r['annotated_image']).is_file() for r in grounding)
        ),
        'structured_concise_ecot_with_privileged_context_excluded_from_training': (
            not run_ecot or ecot_contract_valid
        ),
        'ecot_has_one_transition_start_and_one_midpoint_per_episode': (
            not run_ecot or (
                len(primary_ecot) == args.episodes and
                len(midpoint_ecot) == args.episodes and
                len({r['parent_episode_id'] for r in primary_ecot}) == args.episodes and
                len({r['parent_episode_id'] for r in midpoint_ecot}) == args.episodes and
                all(
                    r['gripper_segmentation']['selected_segment']['start_frame'] <
                    r['selected_frame_index'] <
                    r['gripper_segmentation']['selected_segment']['end_frame_exclusive'] and
                    r['gripper_segmentation']['selected_segment']['segment_index'] > 0
                    for r in midpoint_ecot
                )
            )
        ),
        'all_selected_episodes_scanned_for_gripper_transitions': (
            not run_contact_pipeline or (
                len(audits_by_row) == args.episodes and
                all(row['sta_candidates'] for row in rows)
            )
        ),
        'every_transition_candidate_reviewed': (
            not run_contact_pipeline or len(contact_audits) == len(all_candidates)
        ),
        'every_episode_has_contact_json': (
            not run_contact_pipeline or all((contacts_dir/f'{row["id"]}.json').is_file() for row in rows)
        ),
        'per_episode_contact_lists_match_audited_outcomes': (
            not run_contact_pipeline or per_episode_contacts_match_audits
        ),
        'compact_contact_schema': not run_contact_pipeline or compact_contacts_valid,
        'sta_question_counts_match_contacts': not run_sta or sta_type_counts == {
            'next_contact_noun': accepted_observations,
            'next_contact_verb': accepted_observations,
            'next_contact_bbox': len(accepted_grounding),
            'time_to_next_contact': accepted_observations,
        },
        'multiple_observations_bounded_by_primary_query_and_contact': not run_contact_pipeline or all(
            len(audit['observations']) >= 3 and
            all(
                audit['segment_start_frame'] <=
                audit['observation_window_start_frame'] <=
                observation['query_frame_index'] < audit['candidate_frame'] and
                observation['query_frame_index'] >= audit['query_frame_index'] and
                observation['time_to_contact_seconds'] <=
                audit['time_to_contact_seconds'] + 1e-6
                for observation in audit['observations']
            ) and
            len({
                observation['query_frame_index']
                for observation in audit['observations']
            }) == len(audit['observations'])
            for audit in contact_audits
        ),
        'observations_after_previous_accepted_contact': not run_contact_pipeline or all(
            all(
                observation['query_frame_index'] > previous_contact
                for observation in audit['observations']
            )
            for row_audits in audits_by_row.values()
            for audit in row_audits
            for previous_contact in [max(
                [
                    other['contact']['time'] for other in row_audits
                    if other['valid_contact'] and
                    other['contact']['time'] < audit['candidate_frame']
                ],
                default=-1,
            )]
        ),
        'grounding_contact_bboxes_normalized': not run_sta or sta_bbox_valid,
        'sta_queries_are_precontact_and_teacher_video_is_not_learner_input': (
            not run_sta or sta_temporal_valid
        ),
        'sta_record_ids_unique': (
            not run_sta or len({r['id'] for r in sta_records}) == len(sta_records)
        ),
        'sta_questions_grouped_by_exact_contact': not run_sta or sta_question_groups_valid,
        'sta_bbox_audit_images': (
            not run_sta or
            all((sta_annotated/a['annotated_query_image']).is_file() for a in accepted_grounding)
        ),
        'report_grounding_has_only_grounding_cards': (
            not run_grounding or (
            grounding_report_text.count('<article class="card">') == len(grounding) and
            '· ECoT</h3>' not in grounding_report_text and
            'STA' not in grounding_report_text)
        ),
        'report_ecot_has_only_ecot_cards': (
            not run_ecot or (
            ecot_report_text.count('<article class="card">') == len(ecot) and
            '· grounding</h3>' not in ecot_report_text and
            'STA' not in ecot_report_text)
        ),
        'report_show_context_for_every_ecot_frame': (
            not run_ecot or ecot_report_text.count(
                '<details class="show-context">'
            ) == len(ecot)
        ),
        'report_sta_all_candidates': (
            not run_sta or report_sta_text.count('<article class="card">') == len(contact_audits)
        ),
        'report_sta_all_questions': (
            not run_sta or report_sta_text.count('<tr class="sta-question">') == len(sta_records)
        ),
        'teacher_clips_are_browser_h264': not run_contact_pipeline or all(
            browser_h264_ready(contact_clips/audit['teacher_clip'])
            for audit in contact_audits
        ),
        'full_episode_previews_are_browser_h264': (
            (not run_contact_pipeline and not run_ecot) or
            all(
                browser_h264_ready(episode_videos/f'{row["id"]}.mp4')
                for row in rows
            )
        ),
        'report_sta_episode_previews_and_markers': (
            not run_sta or (
            report_sta_text.count('<details class="episode-preview">') == len(contact_audits) and
            report_sta_text.count('class="event-marker current-marker"') == len(contact_audits) and
            report_sta_text.count('class="event-marker contact-marker"') == len(contact_audits))
        ),
        'sta_report_compatibility_alias': not run_sta or sta_report_alias_text == report_sta_text,
        'cpa_pipeline_complete': (
            not run_cpa or (
                (out/'complete_cpa.json').is_file() and
                json.loads((out/'complete_cpa.json').read_text(encoding='utf-8'))['passed'] is True
            )
        ),
        'cpa_report_is_separate': (
            not run_cpa or (
                (out/'report_cpa.html').is_file() and
                (out/'report_cpa.html').read_text(encoding='utf-8').count(
                    '<details class="show-intermediates">'
                ) == len(cpa_audits)
            )
        ),
    }
    completion = {
        'passed': all(checks.values()), 'checks': checks,
        'mode': mode,
        'episodes_requested': args.episodes,
        'episodes_completed': len(rows),
        'eligible_dataset_episodes': eligible_episode_count,
        'last_append_request': append_request,
        'episodes_before_append': episodes_before_append,
        'episodes_appended': episodes_appended,
        'append_history': append_history,
        'counts': counts, 'grounding_types': types, 'grounding_types_requested': requested_types,
        'records': len(records), 'sta_records': len(sta_records),
        'cpa_contacts': len(cpa_audits), 'cpa_records': len(cpa_records),
        'cpa_multiple_choice_vqa_records': sum(
            len(record.get('multiple_choice_questions') or [])
            for record in cpa_records
        ),
        'sta_observations': accepted_observations,
        'contact_candidates': len(all_candidates), 'valid_contacts': len(accepted_audits),
        'episodes_without_valid_contact': episodes_without_valid_contact,
        'sta_question_counts': sta_type_counts, 'errors': errors,
        'dataset': args.dataset_name, 'dataset_name': args.dataset_name,
        'dataset_family': 'robot', 'dataset_root': str(args.dataset_path),
        'actor': 'robot', 'model': MODEL, 'inference_backend': BACKEND,
        'cpa_negative_a_at_720p': CPA_NEGATIVE_A_AT_720P,
        'cpa_negative_drop_rate': CPA_NEGATIVE_DROP_RATE,
        'spatial_rules': {
            'origin': 'upper_left', 'x_axis': 'right', 'y_axis': 'down',
            'coordinates': 'normalized_[0,1]', 'next_to': 'unique_minimum_bbox_edge_distance',
            'direction': 'strict_full_bbox_separation_plus_20_percent_orthogonal_overlap',
            'excluded_entities': sorted(FORBIDDEN_EXACT_NAMES),
        },
        'ecot_rules': {
            'segmentation_source': 'action.left_gripper_offline_transition_segmentation',
            'closed_threshold_max': GRIPPER_CLOSED_MAX, 'open_threshold_min': GRIPPER_OPEN_MIN,
            'method': 'structured_three_part_ecot',
            'samples': {
                ECOT_PRIMARY_SAMPLE_TYPE: args.episodes if run_ecot else 0,
                ECOT_MIDPOINT_SAMPLE_TYPE: args.episodes if run_ecot else 0,
            },
            'midpoint_definition': 'strictly_inside_a_segment_bounded_by_two_gripper_transitions',
            'output_parts': ['scene_description', 'task_progress_assessment', 'subtask'],
            'subtask_semantics': list(ECOT_SUBTASK_FIELDS),
            'subtask_granularity': 'one_goal_level_task_phase_not_next_contact_or_atomic_motor_action',
            'annotation_teacher_inputs': [
                'global_task_prompt', 'current_sampled_image',
                'full_episode_video_for_target_progress_summary',
                'past_and_future_frames_sampled_once_per_second',
                'target_frame_position_and_progress_text',
            ],
            'training_inputs': ['global_task_prompt', 'current_sampled_image'],
            'annotation_context_excluded_from_training': True,
        },
        'sta_rules': {
            'contact_definition': 'start_of_intentional_physical_interaction_with_primary_object',
            'candidate_source': 'all_hysteretic_action.left_gripper_open_close_transitions',
            'teacher_inputs': ['precontact_query_image', 'centered_contact_context_video', 'global_task_prompt', 'transition_direction'],
            'learner_inputs': ['precontact_query_image', 'sta_question'],
            'context_seconds': args.sta_context_seconds,
            'anticipation_seconds_max': args.sta_anticipation_seconds,
            'compact_contact_schema': {'time': 'frame_id', 'verb': 'lowercase_category', 'noun': 'lowercase_category'},
        },
        'cpa_rules': {
            'multiple_choice_question_version': CPA_MULTIPLE_CHOICE_VQA_VERSION,
            'coordinate_questions_emitted': False,
            'one_multiselect_question_per_eligible_participant': True,
            'robot_contact_agent_questions_emitted': False,
            'answer_cardinality_revealed_in_question': True,
            'answer_unit': 'all_verified_choice_labels_for_that_participant',
            'question_types': [
                'next_contact_hand_point_multiple_choice',
                'next_contact_tool_point_multiple_choice',
                'next_contact_object_point_multiple_choice',
            ],
            'robot_question_types': [
                'next_contact_object_point_multiple_choice',
            ],
            'human_question_types': [
                'next_contact_hand_point_multiple_choice',
                'next_contact_tool_point_multiple_choice',
                'next_contact_object_point_multiple_choice',
            ],
            'learner_inputs': [
                'global_task_prompt', 'marked_precontact_observation_image',
                'english_participant_multiselect_question',
            ],
            'annotation_context_excluded_from_training': True,
        },
        'created_at': time.time(),
    }
    atomic_json(out/'complete.json', completion)
    atomic_json(out/f'complete_{mode}.json', completion)
    if not all(checks.values()):
        raise SystemExit(2)

if __name__ == '__main__': main()
