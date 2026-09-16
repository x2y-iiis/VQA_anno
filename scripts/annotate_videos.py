#!/usr/bin/env python3
"""Run Subtask, ECoT, Grounding, STA, and CPA annotation on videos or WDS."""

from __future__ import annotations

import argparse
import base64
from collections import deque
import concurrent.futures
import contextlib
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import json
import math
import mimetypes
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Iterable
import urllib.error
import urllib.request

import cv2
import numpy as np

VIDEO_CLIP_LIMITER = threading.BoundedSemaphore(4)
VIDEO_CLIP_POOL = None


def open_video_capture(path):
    """Bound FFmpeg decoder threads explicitly; capture-options is insufficient."""
    threads = int(os.environ.get('VQA_OPENCV_DECODER_THREADS', '0'))
    if threads == 0:
        return cv2.VideoCapture(str(path))
    if not 1 <= threads <= 8 or not hasattr(cv2, 'CAP_PROP_N_THREADS'):
        raise ValueError('explicit_opencv_decoder_threads_require_supported_opencv_and_1_to_8')
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, threads])

from bounded_media import FrameCache, FrameSelection, REQUEST_BUDGET, fit_video_file, prepare_inline_media
from dashscope_temporary_video import (
    DashScopeTemporaryPublisher, TemporaryVideo, RESOLVE_HEADER, UPLOAD_TIMEOUT,
    prepare_request_media, refresh_request_urls, redact_media_urls,
)
from request_parallel import (
    RequestAdmission, PartitionedRequestAdmission, atomic_json,
    run_stage_jobs, run_bounded_batches,
)
from provider_workers import frame_executor, FollowingSubtaskIndex
from provider_content_rejection import (
    ProviderContentRejected, EpisodeContentScope, check_content_scope,
    drain_content_jobs, is_input_media_rejection,
    is_input_video_rejection, reject_current_video, rejection_identity,
    check_saved_rejection, save_rejection,
)
from stage_pipeline import StagePipeline, MediaPrefetchPool, DeferredMedia, run_prepared_jobs
from durable_jsonl import (atomic_jsonl, clear_checkpoint_files, load_checkpoint_files,
                           final_record_spool_enabled, pending_final_uids,
                           put_checkpoint_unit, put_checkpoint_units, put_final_record)
from cosmos3_task_name import VLA_TASK_NAME_MAX_LENGTH, select_vla_task_name
from las_subtask import (
    ADAPTER_VERSION as LAS_ADAPTER_VERSION,
    DEFAULT_COSCLI as DEFAULT_LAS_COSCLI,
    DEFAULT_LAS_COS_URI_PREFIX,
    DEFAULT_LAS_MODEL,
    DEFAULT_LAS_POSTPROCESS_MODEL,
    DEFAULT_LAS_STEP1_MODEL,
    DEFAULT_SIGNED_URL_SECONDS as DEFAULT_LAS_SIGNED_URL_SECONDS,
    LASSubtaskAnnotator,
    LASSubtaskConfig,
)
from sam3_snap import Sam3Snapper, snap_pair_twice
from validate_annotation_result_contract import validate_annotation_result_contract
from grd_inventory_review import (
    InventoryReviewer, MODEL as GRD_REVIEW_MODEL, learner_result,
    partition_result, review_item, review_is_current,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(os.environ.get('VQA_DEFAULT_INPUT', 'data/input'))
DEFAULT_OUTPUT = Path(os.environ.get('VQA_DEFAULT_OUTPUT', 'outputs'))
DEFAULT_SAM3_REPO = Path(os.environ.get('SAM3_REPO', PROJECT_ROOT/'third_party/sam3'))
DEFAULT_SAM3_CHECKPOINT = Path(os.environ.get(
    'SAM3_CHECKPOINT', PROJECT_ROOT/'models/sam3/sam3.pt',
))
DEFAULT_ENDPOINTS = {
    'dashscope': 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    'ark': 'https://ark.cn-beijing.volces.com/api/v3/chat/completions',
    'las': 'https://operator.las.cn-beijing.volces.com/api/v1/submit',
    'vla-anno': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
}
DEFAULT_MODELS = {
    'dashscope': {
        'subtask': DEFAULT_LAS_MODEL,
        'ecot': 'qwen3-vl-plus',
        'grd': 'qwen3-vl-plus',
        'sta': 'qwen3-vl-plus',
        'cpa': 'qwen3.8-max',
        'contact_frame': 'qwen3.8-max',
        'cpa_point': 'qwen3.8-max',
    },
    'ark': {
        'subtask': DEFAULT_LAS_MODEL,
        'ecot': 'doubao-seed-2-0-lite-260215',
        'grd': 'doubao-seed-2-0-pro-260215',
        'sta': 'doubao-seed-2-0-lite-260215',
        'cpa': 'doubao-seed-2-0-pro-260215',
        'contact_frame': 'doubao-seed-2-0-pro-260215',
        'cpa_point': 'doubao-seed-2-0-pro-260215',
    },
    'las': {
        'subtask': DEFAULT_LAS_MODEL,
        'ecot': 'doubao-seed-2-0-lite-260215',
        'grd': 'doubao-seed-2-0-pro-260215',
        'sta': 'doubao-seed-2-0-lite-260215',
        'cpa': 'doubao-seed-2-0-pro-260215',
        'contact_frame': 'doubao-seed-2-0-pro-260215',
        'cpa_point': 'doubao-seed-2-0-pro-260215',
    },
    'openai-compatible': {
        'subtask': DEFAULT_LAS_MODEL, 'ecot': '', 'grd': '', 'sta': '', 'cpa': '',
        'contact_frame': '', 'cpa_point': '',
    },
    'vla-anno': {
        'subtask': DEFAULT_LAS_MODEL,
        'ecot': '', 'grd': '', 'sta': '', 'cpa': '',
        'contact_frame': '', 'cpa_point': '',
    },
}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
TASK_ORDER = ('subtask', 'ecot', 'grd', 'sta', 'cpa')
TASK_ALIASES = {'grounding': 'grd'}
TASK_CONTRACT_IDS = {
    'subtask': 'vqa-anno-raw-subtask/v2',
    'ecot': 'vqa-anno-raw-ecot-privileged-teacher-0.5fps-atomic/v3',
    'grd': 'vqa-anno-raw-grd-2fps-inventory-future-first-object/v3',
    'sta': 'vqa-anno-raw-sta-cpa-final-random-event-bbox/v5',
    'cpa': 'vqa-anno-cpa-object-only-two-decimal/v13',
}
LEGACY_ECOT_CONTRACT_IDS = {
    'vqa-anno-raw-ecot-privileged-2fps-atomic/v2',
}
MAX_DATA_URI_ITEM_BYTES = 19 * 1024 * 1024
ECOT_FPS = 2.0
ECOT_TEACHER_FPS = 0.5
ECOT_INTERVAL = 4
ECOT_SAMPLE_TYPE = 'uniform_2fps_interval'
ECOT_MEDIA_KIND = 'complete_0.5fps_episode_video_plus_target_image'
ECOT_SUBTASK_FIELDS = ('subtask', 'action', 'object', 'source', 'target')
ECOT_MISSING_SEMANTIC = 'None'
ECOT_PRIVILEGED_LEAK_PATTERN = re.compile(
    r'\b(?:video|future frames?|past frames?|timestamps?|progress percentage|'
    r'episode progress|later in the episode|earlier in the episode)\b',
    flags=re.I,
)
GRD_FRAME_FPS = 2.0
GRD_FUTURE_FPS = 3.0
GRD_FUTURE_SECONDS = 4.0
GRD_MAX_OBJECTS = 4
GRD_FORBIDDEN_OBJECT_TOKENS = {
    'arm', 'arms', 'background', 'counter', 'counters', 'desk',
    'desks', 'floor', 'floors', 'gripper', 'grippers', 'hand', 'hands',
    'human', 'humans', 'people', 'person', 'table', 'tables', 'wall', 'walls',
}
CONTACT_FRAME_WINDOW_SECONDS = 0.75
CONTACT_FRAME_MAX_IMAGES = 24
STA_LOOKBACK_SECONDS = 1.0
STA_OBSERVATIONS_PER_CONTACT = 3
STA_MIN_CONTACT_GAP_SECONDS = 0.1
VLA_CHUNK_SECONDS = 295.0
VLA_DIRECT_UPLOAD_BYTES = 95 * 1000 * 1000
VLA_LOCAL_VIDEO_BITRATE = 1_600_000
COSMOS3_PHYSICAL_SCHEMA = 'cosmos3_video_physical_webdataset_v1'

ECOT_SYSTEM_PROMPT = '''You create structured embodied action labels for one target observation in a manipulation episode.

You receive the global task, the target image, and the complete 0.5 FPS episode video. The video is privileged annotation-only evidence. Match the target image to the episode and use the video to identify task progress, the active goal-level subtask, and the immediate atomic action. Never mention privileged context, timestamps, frame indices, or progress percentages in the output. Do not describe later actions or outcomes as if they were already visible at the target observation.

Return exactly four top-level JSON fields:

1. scene_description: Concisely describe task-relevant objects, spatial relations, and the visible hands, grippers, or held tools. Every statement must be supported by the target image alone. There is no hard word limit.

2. task_progress: Concisely describe completed and remaining task goals, with no hard word limit. End with either "; Task complete." or "; Task not yet complete." Do not credit actions completed only after the target observation.

3. current_subtask: An object containing exactly subtask, action, object, source, and target. The subtask is concise lowercase text describing the active goal-level phase, not a sequence of motor actions; there is no hard word limit. The other four fields are lowercase action semantics; use the exact string "None" for an unstated semantic field. Do not invent a source or destination. If the overall task is complete, set subtask to "no further subtask remains." and all four semantic fields to "None".

4. atomic_action: One lowercase English action phrase describing the single, immediate observable action after the target observation. If the action is already underway at the target frame, describe its continuation. Use a base-form verb plus the directly manipulated object; include a destination or necessary spatial relation when that is part of this one action. Its granularity must be smaller than the goal-level subtask. Examples: "grasp the green block", "reach to the lemon", "move the lemon to the plate".

For atomic_action, use the video only to resolve what immediately happens next. Never skip an intervening reach or move to label a later grasp or final goal. Do not require physical contact: reach, move, place, release, rotate, press, and grasp are all valid when supported by the episode. Use only visually supported object descriptions, attributes, and destinations. Return one action, not a list or an "and then" sequence. Do not copy a coarse subtask such as "prepare the ingredients" as an atomic action. Use the exact string "None" if no further task-related action occurs or the immediate action cannot be determined. Do not force every subtask into a fixed action sequence.

Return only the JSON object with exactly these fields; no commentary or extra keys.'''

# The approved prompt supports both human and robot embodiments without guessing.
HUMAN_ECOT_SYSTEM_PROMPT = ECOT_SYSTEM_PROMPT

GRD_FIRST_OBJECT_SYSTEM = '''You identify only the first discrete object that will be intentionally manipulated for the supplied current subtask. Use the future video as teacher-only temporal evidence. Ignore actions belonging to later subtasks. Never select people, hands, robot arms, grippers, held tools, destinations, support surfaces, or scene regions. Return only JSON and do not return any bounding box.'''

GRD_SYSTEM = '''You create conservative task-relevant object grounding annotations for exactly one current image. Return only JSON. List at most four clearly visible, discrete manipulable objects or task-relevant receptacles, prioritizing objects needed for the supplied current subtask. Identified objects should be distinguishable by their name and alternate_name. Use tight integer [x1,y1,x2,y2] boxes on a 0-to-1000 coordinate grid. Never include people, robot arms, hands, grippers, background, floors, walls, generic work surfaces, or occluded objects.'''

STA_SYSTEM = '''You propose intentional contact onsets for later CPA review. Return only JSON. Find every intentional hand, held-tool, or robot-end-effector contact onset with a manipulable object. For each event return its approximate contact time, contact agent role, contacted object name, and contact verb. Do not create learner observation frames or bounding boxes at this stage. Do not label sustained contact as a new onset. Exclude accidental contact, support surfaces, self-contact, and contacts not justified by the supplied media.'''

STA_BBOX_SYSTEM = '''You ground the target object of one known future contact event in exactly one pre-contact observation image. Use the explicit contact event name to identify the object that will be contacted. Return one tight integer [x1,y1,x2,y2] bbox on a 0-to-1000 grid. Exclude hands, robot arms, grippers, held tools, support surfaces, and unrelated objects. If the target object is not clearly visible, report target_visible=false and bbox_xyxy_1000=null. Return only JSON.'''

CPA_SYSTEM = '''You independently review proposed intentional contact events against the complete supplied subtask clip. Confirm or reject every proposal, correct the contact time and semantic participants when needed, and return a tight interaction bbox containing the contact agent, contacted object, and visible interface. Reject unsupported events. Do not return pre-contact learner observations or contact points in this review stage. Return only JSON.'''

CONTACT_FRAME_SYSTEM = '''You select the exact first frame of an intentional contact onset from an ordered candidate-frame sequence. Use the supplied subtask timeline as semantic evidence. A video may transition between subtasks inside the candidate window, so never assume that every image belongs to the same subtask. Select an onset only when the candidate frame belongs to the requested event subtask and the immediately preceding visible candidate shows separation. Reject continued holding, release, accidental contact, support-surface contact, and onsets belonging to an adjacent subtask. Return only JSON.'''


class FatalProviderError(RuntimeError):
    """Stop a run after a credential, billing, or quota failure."""


@dataclass(frozen=True)
class FileSlice:
    """Lazy byte range backed by one immutable physical WebDataset shard."""

    path: Path
    offset: int
    length: int

    def read(self) -> bytes:
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            payload = stream.read(self.length)
        if len(payload) != self.length:
            raise RuntimeError(
                f'file_slice_short_read:{self.path}:{self.offset}:'
                f'{len(payload)}:{self.length}'
            )
        return payload

    def write_to(self, destination, chunk_size: int = 8 * 1024 * 1024) -> None:
        """Copy this slice without retaining the complete payload in memory."""
        remaining = self.length
        with self.path.open('rb') as source:
            source.seek(self.offset)
            while remaining:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    raise RuntimeError(
                        f'file_slice_short_read:{self.path}:{self.offset}:'
                        f'{self.length - remaining}:{self.length}'
                    )
                destination.write(chunk)
                remaining -= len(chunk)


class AdaptiveConcurrencyLimiter:
    """Keep a large work queue while adapting HTTP concurrency to provider 429s."""

    mode = 'condition-single-wake/v2-fixed-feedback-bypass'

    def __init__(self, maximum: int, initial_limit: int | None = None, fixed: bool = False):
        self.maximum = maximum
        self.fixed = fixed
        self.limit = maximum if fixed else min(maximum, max(1, initial_limit or maximum))
        self.active = 0
        self.success_streak = 0
        self.last_reduction = 0.0
        # No limiter method acquires this mutex recursively. A plain lock also
        # avoids the recursive-lock release/restore path in Condition.wait.
        self.condition = threading.Condition(threading.Lock())

    def __enter__(self) -> AdaptiveConcurrencyLimiter:
        with self.condition:
            while self.active >= self.limit:
                self.condition.wait()
            self.active += 1
        return self

    def __exit__(self, *_: object) -> None:
        with self.condition:
            self.active -= 1
            # One completed request returns one slot. Waking every waiter
            # makes them contend with both request starts and completions.
            if self.active < self.limit:
                self.condition.notify(1)

    def record_success(self) -> None:
        # Fixed mode is selected at construction and has no success feedback.
        if self.fixed:
            return
        with self.condition:
            self.success_streak += 1
            threshold = max(256, self.limit * 4)
            if self.limit < self.maximum and self.success_streak >= threshold:
                self.limit += 1
                self.success_streak = 0
                self.condition.notify(1)

    def reduce_after_throttle(self) -> tuple[int, int, bool]:
        with self.condition:
            if self.fixed:
                return self.limit, self.active, False
            now = time.monotonic()
            changed = False
            if now - self.last_reduction >= 5.0:
                new_limit = 64 if self.limit > 64 else max(1, int(self.limit * 0.8))
                if new_limit < self.limit:
                    self.limit = new_limit
                    changed = True
                self.last_reduction = now
                self.success_streak = 0
                self.condition.notify_all()
            return self.limit, self.active, changed

    def snapshot(self) -> tuple[int, int]:
        with self.condition:
            return self.limit, self.active


class BurstFailureBackoff:
    """Permanently cap this run after a configurable burst of failures.

    Admission limits change in place: existing requests finish and checkpoints
    remain intact. The same controller covers downstream/review HTTP failures
    and record failures, including LAS exceptions surfaced by its adapter.
    """

    def __init__(self, threshold, window_seconds, state_path, http_limit=64, record_limit=32):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.state_path = Path(state_path)
        self.http_limit = http_limit
        self.record_limit = record_limit
        self.events = deque()
        self.targets = []
        self.lock = threading.Lock()
        self.conservative = False
        if self.state_path.is_file():
            self.conservative = json.loads(self.state_path.read_text()).get('mode') == 'conservative'

    @staticmethod
    def cap(limiter, ceiling):
        with limiter.condition:
            limiter.maximum = min(limiter.maximum, ceiling)
            limiter.limit = min(limiter.limit, limiter.maximum)
            limiter.condition.notify_all()

    def register(self, limiter, kind):
        with self.lock:
            ceiling = self.http_limit if kind == 'http' else self.record_limit
            self.targets.append((limiter, ceiling))
            if self.conservative:
                self.cap(limiter, ceiling)
                print(f'failure_burst_restored kind={kind} effective_limit={limiter.limit}', flush=True)

    def record_failure(self):
        if not self.threshold:
            return
        with self.lock:
            if self.conservative:
                return
            now = time.monotonic()
            self.events.append(now)
            while self.events and self.events[0] < now - self.window_seconds:
                self.events.popleft()
            if len(self.events) < self.threshold:
                return
            self.conservative = True
            for limiter, ceiling in self.targets:
                self.cap(limiter, ceiling)
            state = {'mode': 'conservative', 'failure_events': len(self.events),
                     'window_seconds': self.window_seconds, 'http_limit': self.http_limit,
                     'record_limit': self.record_limit, 'updated_at_unix': time.time()}
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                json.dump(state, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
            print('failure_burst_backoff ' + json_dumps(state), flush=True)


BURST_GROWTH_THROTTLE_MARKERS = (
    'limit_burst_rate',
    'throttling.burstrate',
    'requestbursttoofast',
)

TRANSIENT_SERVICE_OVERLOAD_MARKERS = (
    'serveroverloaded',
    'serviceoverloaded',
    'server overloaded',
    'service overloaded',
)


def is_burst_growth_throttle(message: str) -> bool:
    """Return whether a 429 reports request-rate growth rather than quota use."""
    lowered = message.lower()
    return any(marker in lowered for marker in BURST_GROWTH_THROTTLE_MARKERS)


def is_transient_service_overload(message: str) -> bool:
    """Identify provider load shedding that is not an account quota signal."""
    lowered = message.lower()
    quota_markers = (
        'tpm', 'rpm', 'tokens per minute', 'requests per minute',
        'limit_requests', 'limit_burst_rate', 'requestbursttoofast',
        'rate limit exceeded', 'ratelimitexceeded',
    )
    return (any(marker in lowered for marker in TRANSIENT_SERVICE_OVERLOAD_MARKERS)
            and not any(marker in lowered for marker in quota_markers))


class RequestStartPacer:
    """Space request starts so a cooldown does not end in another burst."""

    mode = 'fifo-directed/v2-zero-interval-bypass'

    def __init__(self, interval_seconds: float, timed_recovery=False):
        self.interval_seconds = interval_seconds
        self.timed_recovery = timed_recovery
        self.base_interval = interval_seconds
        self.max_interval = max(2.0, interval_seconds)
        self.last_increase = float('-inf')
        self.last_recovery = time.monotonic()
        self.next_start = 0.0
        self.success_streak = 0
        self.last_throttle = 0.0
        self.lock = threading.Lock()
        self.waiters = deque()
        self.cohort_feedback = None
        self.zero_interval_bypasses = 0
        self.dispatcher = None

    def enable_independent_dispatch(self):
        from paced_request_dispatcher import PacedRequestDispatcher
        with self.lock:
            if self.waiters or self.dispatcher is not None:
                raise RuntimeError('pacing_dispatch_must_be_enabled_before_waiting')
            self.dispatcher = PacedRequestDispatcher(self)
            self.mode = self.dispatcher.mode

    def enable_cohort_feedback(self, mature=False, initial_interval=None, smooth=False):
        from cohort_burst_feedback import CohortBurstFeedback
        from mature_burst_feedback import MatureBurstFeedback
        from smooth_burst_feedback import SmoothBurstFeedback
        if mature and smooth:
            raise ValueError('incompatible_cohort_feedback_modes')
        with self.lock:
            if not self.timed_recovery or not getattr(self, 'server_queue_mode', False):
                raise ValueError('cohort_feedback_requires_timed_server_queue')
            feedback_type = (MatureBurstFeedback if mature else
                             SmoothBurstFeedback if smooth else CohortBurstFeedback)
            self.cohort_feedback = feedback_type(time.monotonic())
            # A smooth 20-RPS cold start replaces an unbounded initial burst.
            self.interval_seconds = max(self.base_interval, self.interval_seconds, .05)
            if initial_interval is not None:
                if not math.isfinite(initial_interval) or not 0 <= initial_interval <= self.max_interval:
                    raise ValueError('invalid_cohort_initial_interval')
                # A one-time, explicitly chosen warm start is not a permanent
                # floor. Fresh successful feedback can still accelerate.
                self.interval_seconds = max(self.base_interval, initial_interval)

    def record_start(self, started_at):
        # Feedback mode is configured before workers start. Arrival-time mode
        # has no start bookkeeping; do not contend with permit dispatch.
        if self.cohort_feedback is None:
            return None
        with self.lock:
            if hasattr(self.cohort_feedback, 'start'):
                return self.cohort_feedback.start(started_at)
        return None

    def warm_start(self, interval):
        """Override learned pacing once without changing the recovery floor."""
        if not self.timed_recovery or not math.isfinite(interval) or not 0 <= interval <= self.max_interval:
            raise ValueError('invalid_request_pacing_warm_start')
        with self.lock:
            self.interval_seconds = max(self.base_interval, interval)

    def record_unclassified(self, feedback_ticket):
        if feedback_ticket is not None:
            with self.lock:
                self.interval_seconds = self.cohort_feedback.abandon(
                    time.monotonic(), feedback_ticket, self.interval_seconds,
                    self.base_interval, self.max_interval)

    def feedback_snapshot(self):
        with self.lock:
            result = (self.cohort_feedback.snapshot() if self.cohort_feedback is not None
                      else {'mode': 'arrival-time/v1'})
            state = dict(result, queued_tickets=len(self.waiters),
                         next_start_delay_seconds=max(0, self.next_start - time.monotonic()),
                         zero_interval_bypasses=self.zero_interval_bypasses)
            if self.dispatcher is not None:
                state.update(self.dispatcher.snapshot_unlocked())
            return state

    def _recover(self, now):
        if not self.timed_recovery:
            return
        if self.cohort_feedback is not None:
            if hasattr(self.cohort_feedback, 'advance'):
                self.interval_seconds = self.cohort_feedback.advance(now, self.interval_seconds)
            self.next_start = min(self.next_start, now + self.interval_seconds)
            return
        # Recovery must also run in wait(): success-dependent recovery can
        # deadlock a model whose next request is scheduled hours in the future.
        self.interval_seconds = min(self.max_interval, self.interval_seconds)
        fast_recovery = getattr(self, 'queue_recovery', False)
        smooth_growth = getattr(self, 'smooth_growth_recovery', False)
        recovery_period = 10 if fast_recovery else 60
        if now - max(self.last_throttle, self.last_recovery) >= recovery_period:
            # Ark's RequestBurstTooFast is specifically about traffic slope. A
            # 2x RPS jump every ten seconds simply recreates the same burst, so
            # recover by 11% instead. DashScope's established queue recovery is
            # intentionally unchanged.
            recovery_factor = .9 if smooth_growth else (.5 if fast_recovery else .75)
            self.interval_seconds = max(self.base_interval,
                                        self.interval_seconds * recovery_factor)
            if self.interval_seconds < .01 and self.base_interval == 0 and self.dispatcher is None:
                self.interval_seconds = 0
            self.last_recovery = now
        # Never retain a deadline created by an obsolete, larger interval.
        self.next_start = min(self.next_start, now + self.interval_seconds)

    def wait(self, check=None) -> None:
        if self.dispatcher is not None:
            return self.dispatcher.wait(check)
        if check is not None:
            check()
        with self.lock:
            now = time.monotonic()
            self._recover(now)
            # With no rate spacing, the network admission gate is the only
            # concurrency limit. An old FIFO backlog must not serialize starts
            # behind a slow queue head after throttle recovery.
            if self.interval_seconds == 0 or (not self.waiters and self.next_start <= now):
                if self.interval_seconds == 0 and self.waiters:
                    self.zero_interval_bypasses += 1
                self.next_start = now + self.interval_seconds
                return
            ticket = threading.Event()
            first = not self.waiters
            self.waiters.append(ticket)
        if first:
            ticket.set()
        try:
            # Only the queue head sleeps against the start deadline. Followers
            # wait for a directed handoff, not the same deadline and mutex race.
            while not ticket.wait(1.0):
                if check is not None:
                    check()
                with self.lock:
                    self._recover(time.monotonic())
                    if self.interval_seconds == 0:
                        self.zero_interval_bypasses += 1
                        return
            while True:
                if check is not None:
                    check()
                with self.lock:
                    now = time.monotonic()
                    self._recover(now)
                    delay = self.next_start - now
                    if delay <= 0:
                        self.next_start = now + self.interval_seconds
                        return
                time.sleep(min(delay, 1.0))
        finally:
            with self.lock:
                was_head = self.waiters[0] is ticket
                self.waiters.remove(ticket)
                successor = self.waiters[0] if was_head and self.waiters else None
            if successor is not None:
                successor.set()

    def record_throttle(self, message: str, started_at=None, feedback_ticket=None) -> float:
        lowered = message.lower()
        burst_growth = is_burst_growth_throttle(message)
        ark_growth_burst = 'requestbursttoofast' in lowered
        queue_burst = (self.timed_recovery and burst_growth
                       and (getattr(self, 'server_queue_mode', False) or ark_growth_burst))
        if 'limit_requests' in lowered:
            minimum = 0.2
        elif 'limit_burst_rate' in lowered:
            minimum = 0.1
        else:
            minimum = 0.05
        with self.lock:
            now = time.monotonic()
            if self.cohort_feedback is not None:
                if queue_burst:
                    self.interval_seconds = self.cohort_feedback.observe(
                        now, started_at, True, self.interval_seconds,
                        self.base_interval, self.max_interval, ticket=feedback_ticket)
                    return self.interval_seconds
                # Explicit RPM/TPM responses retain the existing conservative
                # path and cooldown; invalidate feedback from the previous rate.
                self.cohort_feedback.reset_epoch(now)
            self.queue_recovery = queue_burst
            self.smooth_growth_recovery = ark_growth_burst
            if queue_burst and self.timed_recovery:
                candidate = max(.005, self.interval_seconds)
                if now - self.last_increase >= 5:
                    candidate *= 1.25
                    self.last_increase = now
                self.interval_seconds = min(self.max_interval, candidate)
                self.last_throttle = now
                self.success_streak = 0
                return self.interval_seconds
            if self.timed_recovery:
                # Coalesce a wave of in-flight 429 responses; Retry-After and
                # provider cooldown remain independently enforced by ApiClient.
                candidate = max(minimum, self.interval_seconds)
                if now - self.last_increase >= 30:
                    candidate = max(candidate, self.interval_seconds * 1.5)
                    self.last_increase = now
                self.interval_seconds = min(self.max_interval, candidate)
                self.last_throttle = now
                self.success_streak = 0
                return self.interval_seconds
            candidate = max(minimum, self.interval_seconds)
            if now - self.last_throttle >= 5.0:
                candidate = max(
                    candidate,
                    self.interval_seconds * 1.5
                    if self.interval_seconds else minimum,
                )
            if candidate > self.interval_seconds:
                self.interval_seconds = candidate
                self.success_streak = 0
                self.last_throttle = now
            return self.interval_seconds

    def record_systemic_service_overload(self) -> float:
        """Apply gradual feedback after repeated transient service load shedding.

        A single ``ServerOverloaded`` response is not evidence of an RPM/TPM
        ceiling. Repeated responses in a short window are useful capacity
        feedback, but should not impose the legacy 20-RPS (.05 second) cliff.
        """
        with self.lock:
            now = time.monotonic()
            self.queue_recovery = True
            self.smooth_growth_recovery = True
            candidate = max(.001, self.interval_seconds)
            if now - self.last_increase >= 5:
                candidate *= 1.25
                self.last_increase = now
            self.interval_seconds = min(self.max_interval, candidate)
            self.last_throttle = now
            self.success_streak = 0
            return self.interval_seconds

    def record_success(self, started_at=None, feedback_ticket=None) -> None:
        if self.timed_recovery and self.cohort_feedback is None:
            now = time.monotonic()
            recovery_period = 10 if getattr(self, 'queue_recovery', False) else 60
            # Most successes have no rate update to apply. The dispatcher also
            # runs recovery before every permit. Keep the locked path for due
            # recovery and obsolete deadline/interval repair. A concurrent
            # throttle can only defer recovery, never bypass its cooldown.
            if (now - max(self.last_throttle, self.last_recovery) < recovery_period
                    and self.interval_seconds <= self.max_interval
                    and self.next_start <= now + self.interval_seconds):
                return
        with self.lock:
            if self.cohort_feedback is not None:
                self.interval_seconds = self.cohort_feedback.observe(
                    time.monotonic(), started_at, False, self.interval_seconds,
                    self.base_interval, self.max_interval, ticket=feedback_ticket)
                return
            if self.timed_recovery:
                self._recover(time.monotonic())
                return
            self.success_streak += 1
            if self.success_streak >= 2000 and self.interval_seconds > 0.01:
                self.interval_seconds = max(0.01, self.interval_seconds * 0.95)
                self.success_streak = 0

    def pause_growth(self, started_at=None):
        with self.lock:
            if hasattr(self.cohort_feedback, 'pause_growth'):
                self.interval_seconds = self.cohort_feedback.pause_growth(
                    time.monotonic(), self.interval_seconds, started_at)
            return self.interval_seconds

    def snapshot(self) -> float:
        with self.lock:
            return self.interval_seconds


def is_fatal_provider_response(status: int, response_body: str) -> bool:
    lowered = response_body.lower()
    fatal_tokens = (
        'insufficient_quota',
        'account balance',
        'billing',
        'overdue-payment',
        'overdue payment',
        'arrearage',
        'account is in good standing',
    )
    return status in {401, 403} or any(token in lowered for token in fatal_tokens)


def is_transient_presigned_media_response(status: int, response_body: str) -> bool:
    """Identify Ark failures while fetching an otherwise valid signed media URL.

    Ark reports failures made by its media fetcher as an outer HTTP 400.  They
    are not bad annotation requests and must not contribute to the permanent
    provider-failure streak.  A retry receives a newly signed URL.
    """
    if status != 400:
        return False
    lowered = response_body.lower()
    media_parameter = 'video_url' in lowered or 'image_url' in lowered
    fetch_failure = any(marker in lowered for marker in (
        'error while connecting',
        'timeout while connecting',
        'timeout while downloading url',
        'timeout while processing video_url',
        'timeout while processing image_url',
    ))
    return media_parameter and fetch_failure


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8'),
    ).hexdigest()


def annotation_checkpoint_path(root: Path, task: str, uid: str) -> Path:
    """Map one source record and output task to a stable checkpoint path."""
    digest = hashlib.sha256(uid.encode('utf-8')).hexdigest()
    return root/task/digest[:2]/f'{digest}.jsonl'


class AnnotationUnitCheckpoint:
    """Append and fsync independently recoverable annotation units.

    The identity hash prevents results produced by a different model, prompt,
    upstream annotation, or sampling policy from being reused accidentally.
    """

    SCHEMA_VERSION = 'annotation-unit-checkpoint/v1'

    def __init__(
        self,
        root: Path,
        uid: str,
        task: str,
        identity: dict,
        immutable_files: bool = False,
    ):
        self.uid = uid
        self.task = task
        self.identity = copy.deepcopy(identity)
        self.identity_sha256 = sha256_json(self.identity)
        self.path = annotation_checkpoint_path(root, task, uid)
        self.lock = threading.Lock()
        self.immutable_files = immutable_files
        self.key_locks = {}
        self.entries: dict[str, object] = {}
        load_checkpoint_files(self)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        repair_partial_jsonl_tail(self.path)
        incompatible = False
        with self.path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f'invalid_annotation_checkpoint_json:'
                        f'{self.path}:{line_number}:{error}'
                    ) from error
                if (
                    record.get('schema_version') != self.SCHEMA_VERSION
                    or record.get('input_record_uid') != self.uid
                    or record.get('task') != self.task
                    or record.get('identity_sha256') != self.identity_sha256
                    or record.get('identity') != self.identity
                ):
                    incompatible = True
                    break
                key = record.get('unit_key')
                if not isinstance(key, str) or 'value' not in record:
                    raise ValueError(
                        f'invalid_annotation_checkpoint_record:'
                        f'{self.path}:{line_number}'
                    )
                value = record['value']
                previous = self.entries.get(key)
                if previous is not None and sha256_json(previous) != sha256_json(value):
                    raise ValueError(
                        f'conflicting_annotation_checkpoint_unit:{self.path}:{key}'
                    )
                self.entries[key] = value
        if incompatible:
            preserved = self.path.with_name(
                f'{self.path.name}.incompatible.{time.time_ns()}'
            )
            os.replace(self.path, preserved)
            self.entries.clear()
            print(
                f'annotation_checkpoint_incompatible_preserved task={self.task} '
                f'path={self.path} evidence={preserved}', flush=True,
            )
        elif self.entries and not getattr(self, 'quiet_unit_load', False):
            print(
                f'annotation_checkpoint_loaded task={self.task} uid={self.uid} '
                f'units={len(self.entries)} path={self.path}', flush=True,
            )

    def get(self, key: str) -> object | None:
        with self.lock:
            value = self.entries.get(key)
            return copy.deepcopy(value) if value is not None else None

    def put(self, key: str, value: object) -> None:
        if self.immutable_files:
            put_checkpoint_unit(self, key, value, {
                'schema_version': self.SCHEMA_VERSION,
                'input_record_uid': self.uid, 'task': self.task,
                'identity_sha256': self.identity_sha256, 'identity': self.identity,
                'unit_key': key, 'value': value,
            })
            return
        with self.lock:
            previous = self.entries.get(key)
            if previous is not None:
                if sha256_json(previous) != sha256_json(value):
                    raise ValueError(f'conflicting_annotation_checkpoint_unit:{key}')
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                'schema_version': self.SCHEMA_VERSION,
                'input_record_uid': self.uid,
                'task': self.task,
                'identity_sha256': self.identity_sha256,
                'identity': self.identity,
                'unit_key': key,
                'value': value,
            }
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.entries[key] = copy.deepcopy(value)

    def put_many(self, items: list[tuple[str, object]]) -> None:
        if self.immutable_files:
            put_checkpoint_units(self, [
                (key, value, {
                    'schema_version': self.SCHEMA_VERSION,
                    'input_record_uid': self.uid, 'task': self.task,
                    'identity_sha256': self.identity_sha256, 'identity': self.identity,
                    'unit_key': key, 'value': value,
                })
                for key, value in items
            ])
            return
        for key, value in items:
            self.put(key, value)

    def clear(self) -> None:
        with self.lock:
            clear_checkpoint_files(self.path)
            self.entries.clear()


def parse_json_object(value: str) -> dict:
    text = value.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find('{')
    if start < 0:
        raise ValueError('response_does_not_contain_json_object')
    # LAS models occasionally append an explanation or repeat a corrected JSON
    # object after the first complete answer. Decode exactly the first object
    # instead of expanding to the final closing brace and raising Extra data.
    result, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(result, dict):
        raise ValueError('response_json_is_not_object')
    return result


def response_content(response: dict) -> str:
    content = response['choices'][0]['message']['content']
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(
            str(item.get('text', '')) for item in content if isinstance(item, dict)
        )
    raise ValueError('response_content_is_not_text')


def data_uri_size(mime_type: str, payload: bytes) -> int:
    return len(f'data:{mime_type};base64,') + 4 * ((len(payload) + 2) // 3)


def normalize_image(mime_type: str, payload: bytes) -> tuple[str, bytes]:
    if data_uri_size(mime_type, payload) <= MAX_DATA_URI_ITEM_BYTES:
        return mime_type, payload
    if not mime_type.startswith('image/'):
        return mime_type, payload
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('oversized_request_image_decode_failed')
    quality = 90
    for _ in range(12):
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('oversized_request_image_encode_failed')
        candidate = encoded.tobytes()
        if data_uri_size('image/jpeg', candidate) <= MAX_DATA_URI_ITEM_BYTES:
            return 'image/jpeg', candidate
        ratio = min(0.9, (MAX_DATA_URI_ITEM_BYTES / data_uri_size('image/jpeg', candidate)) ** 0.5 * 0.95)
        image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio))),
            interpolation=cv2.INTER_AREA,
        )
        quality = max(65, quality - 5)
    raise RuntimeError('oversized_request_image_cannot_fit_data_uri_limit')


from ark_file_transport import ArkFilePublisher, ArkFileReference, responses_payload, responses_as_chat


class ApiClient:
    def __init__(self, args: argparse.Namespace):
        self.ecot_image_transport = getattr(args, 'ecot_image_transport', 'inline')
        self.cos_image_publisher = getattr(args, '_cos_image_publisher', None)
        if self.ecot_image_transport == 'cos-presigned' and self.cos_image_publisher is None:
            from cos_ecot_images import CosImagePublisher
            self.cos_image_publisher = CosImagePublisher(
                Path(__file__).resolve().parents[1] / '_runtime/cos-ecot-images/pending',
                workers=args.ark_upload_workers, check=self.ensure_available)
            self.cos_image_publisher.start_cleanup_reaper()
            args._cos_image_publisher = self.cos_image_publisher
        self.ecot_video_publisher = getattr(args, '_ecot_video_publisher', None)
        if (getattr(args, 'ecot_video_transport', 'inline') == 'ark-files'
                and self.ecot_video_publisher is None):
            if args.api != 'ark':
                raise ValueError('ark_files_transport_requires_ark')
            self.ecot_video_publisher = ArkFilePublisher(
                args.ark_file_cache, args.api_key_env, args.endpoint,
                args.ark_upload_workers, check=self.ensure_available,
                on_fatal=lambda status, message: self.record_fatal(status, message, 'ecot', args.models['ecot']),
            )
            args._ecot_video_publisher = self.ecot_video_publisher
            self.ecot_video_publisher.start_cleanup_reaper()
        if (getattr(args, 'ecot_video_transport', 'inline') == 'cos-presigned'
                and self.ecot_video_publisher is None):
            if args.api not in {'ark', 'las'}:
                raise ValueError('cos_presigned_video_transport_requires_ark_or_las')
            from cos_ecot_images import CosVideoPublisher
            self.ecot_video_publisher = CosVideoPublisher(
                Path(__file__).resolve().parents[1] / '_runtime/cos-ecot-videos/pending',
                workers=args.ark_upload_workers, check=self.ensure_available,
            )
            args._ecot_video_publisher = self.ecot_video_publisher
            self.ecot_video_publisher.start_cleanup_reaper()
        if (getattr(args, 'ecot_video_transport', 'inline') == 'dashscope-temporary'
                and self.ecot_video_publisher is None):
            if args.api != 'dashscope':
                raise ValueError('dashscope_temporary_transport_requires_dashscope')
            self.ecot_video_publisher = DashScopeTemporaryPublisher(
                args.dashscope_media_cache, args.api_key_env, args.dashscope_upload_workers,
                policy_qps=getattr(args, 'dashscope_upload_policy_qps', 5),
                pooled_uploads=getattr(args, 'dashscope_pooled_uploads', False),
                isolated_policy=getattr(args, 'dashscope_isolated_policy', False),
            )
            args._ecot_video_publisher = self.ecot_video_publisher
        self.grd_media_transport = getattr(args, 'grd_media_transport', 'inline')
        self.grd_media_publisher = getattr(args, '_grd_media_publisher', None)
        if self.grd_media_transport == 'dashscope-temporary':
            if args.api != 'dashscope':
                raise ValueError('grd_temporary_transport_requires_dashscope')
            if (self.grd_media_publisher is None
                    or self.grd_media_publisher.api_key_env != args.api_key_env):
                self.grd_media_publisher = DashScopeTemporaryPublisher(
                    args.dashscope_media_cache, args.api_key_env, args.dashscope_upload_workers,
                    policy_qps=getattr(args, 'dashscope_upload_policy_qps', 5),
                    compact_video_cache=True,
                    pooled_uploads=getattr(args, 'dashscope_pooled_uploads', False),
                    isolated_policy=getattr(args, 'dashscope_isolated_policy', False),
                )
                if self.ecot_video_publisher is not None:
                    self.grd_media_publisher.slots = self.ecot_video_publisher.slots
                args._grd_media_publisher = self.grd_media_publisher
        self.model_specific_admission = getattr(args, 'model_specific_admission', False)
        self.model_args = copy.copy(args)
        self.model_clients = {}
        self.model_clients_lock = threading.Lock()
        self.api = args.api
        self.endpoint = args.endpoint
        self.las_gateway = self.endpoint.rstrip('/') == DEFAULT_ENDPOINTS['las'].rstrip('/')
        self.api_key_env = args.api_key_env
        self.max_attempts = args.max_attempts
        self.request_timeout_seconds = getattr(args, 'request_timeout_seconds', 900)
        self.server_wait_seconds = (getattr(args, 'dashscope_server_wait_seconds', 0)
                                    if self.api == 'dashscope' else 0)
        self.fatal_stop_file = args.fatal_stop_file
        self.rate_state_file = args.rate_state_file
        from coalesced_rate_state import CoalescedRateState
        fair_network = getattr(getattr(args, 'request_admission', None), 'fair_network', False)
        self.rate_state_writer = (CoalescedRateState(self.rate_state_file)
                                  if self.server_wait_seconds or fair_network else None)
        self.rate_utilization = args.rate_utilization
        self.request_admission = getattr(args, 'request_admission', None)
        self.fixed_http_concurrency = getattr(args, 'fixed_http_concurrency', False)
        initial_http_limit = args.max_http_active
        initial_request_interval = args.request_start_interval
        restored = False
        restore_path = self.rate_state_file
        if not restore_path.is_file():
            restore_path = getattr(args, 'legacy_rate_state_file', restore_path)
        if not self.fixed_http_concurrency and not self.model_specific_admission and restore_path.is_file():
            try:
                state = json.loads(restore_path.read_text(encoding='utf-8'))
                expected_model = getattr(args, 'admission_model', None)
                if state.get('provider') == self.api and (
                    expected_model is None or state.get('model') == expected_model
                ):
                    initial_http_limit = min(
                        args.max_http_active,
                        max(1, int(state.get('effective_http_limit') or args.max_http_active)),
                    )
                    initial_request_interval = max(
                        args.request_start_interval,
                        float(state.get('request_start_interval_seconds') or 0.0),
                    )
                    restored = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        self.http_limiter = AdaptiveConcurrencyLimiter(
            args.max_http_active, initial_http_limit, fixed=self.fixed_http_concurrency,
        )
        # Ark's fixed per-model limiter is redundant when the fair global gate
        # already covers the same (or a smaller hot-reloadable) capacity.  At
        # thousands of waiter threads, entering and leaving both gates makes
        # the unused Python Condition a severe GIL convoy.  Keep the limiter
        # for every adaptive/multi-provider configuration and bypass it only
        # when the global gate proves it owns the complete capacity envelope.
        self.bypass_fixed_http_limiter = bool(
            self.api in {'ark', 'las'}
            and fair_network
            and self.fixed_http_concurrency
            and getattr(self.request_admission, 'control_maximum', None) is not None
            and self.http_limiter.maximum >= self.request_admission.control_maximum
        )
        self.burst_backoff = getattr(args, 'burst_backoff', None)
        if self.burst_backoff is not None:
            self.burst_backoff.register(self.http_limiter, 'http')
        self.request_pacer = RequestStartPacer(initial_request_interval,
                                             timed_recovery=getattr(self.request_admission, 'fair_network', False) is True)
        self.request_pacer.server_queue_mode = self.server_wait_seconds > 0
        if self.api in {'ark', 'las'} and self.request_pacer.timed_recovery:
            self.request_pacer.enable_independent_dispatch()
        # Fixed concurrency must not restore an old reduced HTTP cap. It can
        # still warm-start recent model-specific pacing without turning that
        # learned interval into the user's permanent recovery floor.
        expected_model = getattr(args, 'admission_model', None)
        if (self.fixed_http_concurrency and self.request_pacer.timed_recovery
                and not self.model_specific_admission and expected_model):
            try:
                state = json.loads(restore_path.read_text(encoding='utf-8'))
                if not isinstance(state, dict):
                    raise ValueError('invalid_rate_state_object')
                stamp = float(state.get('updated_at_unix', float('nan')))
                interval = float(state.get('request_start_interval_seconds', float('nan')))
                if (state.get('schema_version') == 'vqa-provider-rate-state/v1'
                        and state.get('provider') == self.api and state.get('model') == expected_model
                        and math.isfinite(stamp) and 0 <= time.time()-stamp <= 3600
                        and math.isfinite(interval) and interval >= 0):
                    self.request_pacer.interval_seconds = max(
                        self.request_pacer.base_interval, min(self.request_pacer.max_interval, interval))
                    cohort_warm_start = (os.environ.get('VQA_DASHSCOPE_COHORT_PACING') in {'1', 'mature'}
                                         and str(expected_model).startswith('qwen3.8-flash'))
                    restored_error = str(state.get('last_error', ''))
                    if ((self.server_wait_seconds or self.api in {'ark', 'las'}) and not cohort_warm_start
                            and (state.get('concurrency_error')
                                 or is_burst_growth_throttle(restored_error))):
                        self.request_pacer.interval_seconds = max(
                            self.request_pacer.base_interval,
                            min(.05, self.request_pacer.interval_seconds))
                        self.request_pacer.queue_recovery = True
                        self.request_pacer.smooth_growth_recovery = (
                            self.api in {'ark', 'las'} and 'requestbursttoofast' in restored_error.lower())
                    print(f'api_fixed_pacing_restored model={expected_model} '
                          f'request_start_interval_seconds={self.request_pacer.interval_seconds} '
                          f'http_cap_unchanged={args.max_http_active}', flush=True)
            except (OSError, ValueError, TypeError, OverflowError):
                pass
        warm_start = getattr(args, 'request_warm_start_interval', None)
        if warm_start is not None:
            self.request_pacer.warm_start(warm_start)
            print(f'api_pacing_warm_start model={expected_model} '
                  f'interval_seconds={self.request_pacer.interval_seconds} '
                  f'recovery_floor_seconds={self.request_pacer.base_interval}', flush=True)
        self.rate_lock = threading.Lock()
        self.transient_overload_times = deque()
        self.transient_overload_window_seconds = 10.0
        self.transient_overload_threshold = 8
        self.transient_overload_events = 0
        self.systemic_overload_events = 0
        from http_transport_metrics import HTTPTransportMetrics
        # Both Ark Files and COS-presigned ECoT use thousands of URL-only
        # request workers.  Keeping transport counters behind a Python mutex
        # makes those workers convoy after they have already acquired a scarce
        # global HTTP slot.  COS was added after the Ark Files optimization and
        # must use the same native counters.
        native_ecot_transport = bool(
            self.api in {'ark', 'las'} and self.ecot_video_publisher is not None
        )
        self.http_transport_metrics = HTTPTransportMetrics(
            backend='auto' if native_ecot_transport else 'python')
        self.service_circuit = None
        if isinstance(self.ecot_video_publisher, ArkFilePublisher):
            from recoverable_service_circuit import RecoverableServiceCircuit
            self.service_circuit = RecoverableServiceCircuit()
        if (os.environ.get('VQA_DASHSCOPE_RECOVERABLE_5XX') == '1' and self.api == 'dashscope'
                and self.server_wait_seconds > 0 and self.request_pacer.timed_recovery
                and str(getattr(args, 'admission_model', '')).startswith('qwen3.8-flash')):
            from recoverable_service_circuit import RecoverableServiceCircuit
            self.service_circuit = RecoverableServiceCircuit()
        if (os.environ.get('VQA_DASHSCOPE_COHORT_PACING') in {'1', 'mature'} and self.api == 'dashscope'
                and self.server_wait_seconds > 0 and self.request_pacer.timed_recovery
                and str(getattr(args, 'admission_model', '')).startswith('qwen3.8-flash')):
            self.request_pacer.enable_cohort_feedback(
                mature=os.environ.get('VQA_DASHSCOPE_COHORT_PACING') == 'mature',
                smooth=os.environ.get('VQA_DASHSCOPE_SMOOTH_GROWTH') == '1',
                initial_interval=(float(os.environ['VQA_DASHSCOPE_COHORT_INITIAL_INTERVAL'])
                                  if 'VQA_DASHSCOPE_COHORT_INITIAL_INTERVAL' in os.environ else None))
        self.cooldown_until = 0.0
        self.rate_limit_events = 0
        self.presigned_media_refreshes = 0
        self.presigned_media_fetch_errors = 0
        self.request_failures = 0
        self.max_consecutive_request_failures = getattr(args, 'max_consecutive_request_failures', 20)
        self.las_operator = None
        if self.api == 'las' and self.las_gateway:
            from las_customer_ark_operator import LASCustomerArkOperator, LASVideoPublisher
            publisher = getattr(args, '_las_operator_media_publisher', None)
            if publisher is None:
                media_root = Path(getattr(args, 'runtime_state_dir', args.output / '_state')) / 'las-operator-media'
                publisher = LASVideoPublisher(
                    media_root, workers=args.ark_upload_workers, check=self.ensure_available,
                )
                publisher.start_cleanup_reaper()
                args._las_operator_media_publisher = publisher
            self.model_args._las_operator_media_publisher = publisher
            self.las_operator = LASCustomerArkOperator(
                self.endpoint,
                Path(getattr(args, 'runtime_state_dir', args.output / '_state')) / 'las-operator-tasks',
                publisher,
            )
        if restored:
            print(
                f'api_rate_state_restored effective_http_limit={initial_http_limit} '
                f'request_start_interval_seconds={initial_request_interval}',
                flush=True,
            )

    def client_for_model(self, model: str) -> ApiClient:
        if not self.model_specific_admission:
            return self
        with self.model_clients_lock:
            if model not in self.model_clients:
                args = copy.copy(self.model_args)
                args.model_specific_admission = False
                args.admission_model = model
                args.legacy_rate_state_file = self.rate_state_file
                digest = hashlib.sha256(model.encode()).hexdigest()[:16]
                args.rate_state_file = self.rate_state_file.with_name(
                    f'{self.rate_state_file.stem}-{digest}.json'
                )
                self.model_clients[model] = ApiClient(args)
            return self.model_clients[model]

    def admission_snapshot(self) -> dict:
        if self.model_specific_admission:
            with self.model_clients_lock:
                clients = list(self.model_clients.items())
            return {model: client.admission_snapshot() for model, client in clients}
        limit, active = self.http_limiter.snapshot()
        with self.rate_lock:
            cooldown = max(0.0, self.cooldown_until - time.monotonic())
            events = self.rate_limit_events
            transient_overload_events = self.transient_overload_events
            systemic_overload_events = self.systemic_overload_events
            presigned_media_refreshes = self.presigned_media_refreshes
            presigned_media_fetch_errors = self.presigned_media_fetch_errors
        result = {'effective_http_limit': limit, 'active_or_waiting_global': active,
                'model_limiter_mode': ('bypassed-redundant-fixed/v1'
                                       if self.bypass_fixed_http_limiter
                                       else getattr(self.http_limiter, 'mode', 'legacy')),
                'concurrency_mode': 'fixed' if self.fixed_http_concurrency else 'adaptive',
                'request_start_interval_seconds': self.request_pacer.snapshot(),
                'request_pacing_mode': self.request_pacer.mode,
                'request_pacing_feedback': self.request_pacer.feedback_snapshot(),
                'cooldown_seconds': cooldown, 'rate_limit_events': events,
                'transient_overload_events': transient_overload_events,
                'systemic_overload_events': systemic_overload_events,
                'presigned_media_refreshes': presigned_media_refreshes,
                'presigned_media_fetch_errors': presigned_media_fetch_errors,
                'http_transport': self.http_transport_metrics.snapshot(),
                'rate_state_file': str(self.rate_state_file)}
        if self.service_circuit is not None:
            result['service_circuit'] = self.service_circuit.snapshot()
        if self.server_wait_seconds:
            result['server_wait_seconds'] = self.server_wait_seconds
        if self.rate_state_writer is not None:
            result['rate_state_persistence'] = self.rate_state_writer.snapshot()
        publishers = {name: publisher for name, publisher in (
            ('ecot', self.ecot_video_publisher), ('grd', self.grd_media_publisher),
        ) if isinstance(publisher, DashScopeTemporaryPublisher) and publisher.pooled_uploads}
        if isinstance(self.ecot_video_publisher, ArkFilePublisher):
            result['ark_files'] = self.ecot_video_publisher.snapshot()
        if getattr(self.ecot_video_publisher, 'cleanup_log_name', None) == 'cos_video':
            result['cos_videos'] = self.ecot_video_publisher.snapshot()
        if self.cos_image_publisher is not None:
            result['cos_images'] = self.cos_image_publisher.snapshot()
        if publishers:
            result['temporary_upload_phases'] = {
                name: publisher.metrics.snapshot() for name, publisher in publishers.items()}
            result['temporary_upload_metrics_mode'] = {
                name: publisher.metrics.mode for name, publisher in publishers.items()}
            result['temporary_policy_workers'] = {
                name: publisher.policy_process.snapshot() for name, publisher in publishers.items()
                if publisher.policy_process is not None}
        return result

    def ensure_available(self) -> None:
        check_content_scope()
        if getattr(self.request_admission, 'fair_network', False):
            if self.request_admission.check_available is not None:
                self.request_admission.check_available()
        elif self.fatal_stop_file.is_file():
            raise FatalProviderError(f'fatal_stop_file_exists:{self.fatal_stop_file}')
        if not os.environ.get(self.api_key_env, '').strip():
            raise RuntimeError(f'missing_api_key_environment_variable:{self.api_key_env}')
        if self.las_operator is not None:
            self.las_operator.ensure_available()

    def record_fatal(self, status: int, message: str, task: str, model: str) -> None:
        # Multiple request threads may observe the same fatal provider state.
        # The shared atomic writer uses a per-thread temporary path, avoiding
        # races where one thread replaces another thread's fixed ``.tmp`` file.
        atomic_json(self.fatal_stop_file, {
            'http_status': status, 'error': message[-2000:], 'task': task,
            'model': model, 'created_at_unix': time.time(),
        })

    @staticmethod
    def retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
        value = error.headers.get('Retry-After') if error.headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                when = parsedate_to_datetime(value)
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def record_request_success(self):
        # A no-op success linearizes at this read; do not reset a later failure.
        if self.request_failures:
            with self.rate_lock:
                self.request_failures = 0

    def wait_for_provider(self) -> None:
        while True:
            if getattr(self.request_admission, 'fair_network', False):
                self.ensure_available()
            # CPython publishes this float reference atomically. Read-only
            # checks must not convoy requests on the rate-state update lock.
            delay = self.cooldown_until - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 5.0))

    def record_rate_limit(
        self,
        status: int,
        task: str,
        model: str,
        attempt: int,
        retry_after: float | None,
        message: str,
        started_at: float | None = None,
        feedback_ticket=None,
    ) -> float:
        concurrency_error = 'throttling.concurrency' in message.lower()
        burst_growth = is_burst_growth_throttle(message)
        transient_service_overload = (
            self.api in {'ark', 'las'} and is_transient_service_overload(message)
        )
        overload_window_events = 0
        systemic_service_overload = False
        if transient_service_overload:
            now = time.monotonic()
            with self.rate_lock:
                self.transient_overload_events += 1
                self.transient_overload_times.append(now)
                cutoff = now - self.transient_overload_window_seconds
                while (self.transient_overload_times
                       and self.transient_overload_times[0] < cutoff):
                    self.transient_overload_times.popleft()
                overload_window_events = len(self.transient_overload_times)
                systemic_service_overload = (
                    overload_window_events >= self.transient_overload_threshold
                )
                if systemic_service_overload:
                    self.systemic_overload_events += 1
        effective_limit, active_requests, limit_changed = (
            self.http_limiter.reduce_after_throttle()
        )
        queue_mode = getattr(self, 'server_wait_seconds', 0) > 0
        # A concurrency rejection is an in-flight capacity signal, not a
        # reason to exponentially reduce every future request's start rate.
        if transient_service_overload:
            request_start_interval = (
                self.request_pacer.record_systemic_service_overload()
                if systemic_service_overload
                else self.request_pacer.pause_growth(started_at)
            )
        else:
            request_start_interval = (
                self.request_pacer.pause_growth(started_at) if queue_mode and concurrency_error
                else self.request_pacer.record_throttle(message, started_at, feedback_ticket)
            )
        default_delay = (
            min(120.0, 2 ** attempt)
            if concurrency_error else min(120.0, 10 * (2 ** (attempt - 1)))
        )
        if any(token in message.lower() for token in ('tpm', 'rpm', 'tokens per minute', 'requests per minute')):
            default_delay = max(default_delay, 60.0)
        base_delay = retry_after if retry_after is not None else default_delay
        cooldown = max(1.0, base_delay / self.rate_utilization)
        fair_network = getattr(self.request_admission, 'fair_network', False)
        if ((queue_mode or fair_network)
                and retry_after is None
                and (concurrency_error or burst_growth or transient_service_overload)):
            # Bounded per-request backoff below remains active. Other requests
            # need not all stop for an unrequested model-wide cooldown.
            cooldown = 0.0
        with self.rate_lock:
            now = time.monotonic()
            self.cooldown_until = max(self.cooldown_until, now + cooldown)
            self.rate_limit_events += 1
            state = {
                'schema_version': 'vqa-provider-rate-state/v1',
                'provider': self.api,
                'http_status': status,
                'task': task,
                'model': model,
                'attempt': attempt,
                'retry_after_seconds': retry_after,
                'global_cooldown_seconds': cooldown,
                'rate_limit_events': self.rate_limit_events,
                'rate_limit_scope': (
                    'systemic-service-overload' if systemic_service_overload
                    else 'request-local-service-overload' if transient_service_overload
                    else 'concurrency' if concurrency_error
                    else 'burst-growth' if burst_growth
                    else 'quota-or-unclassified'
                ),
                'transient_overload_events': self.transient_overload_events,
                'systemic_overload_events': self.systemic_overload_events,
                'overload_window_events': overload_window_events,
                'concurrency_error': concurrency_error,
                'effective_http_limit': effective_limit,
                'active_http_requests': active_requests,
                'effective_limit_changed': limit_changed,
                'request_start_interval_seconds': request_start_interval,
                'last_error': message[-2000:],
                'updated_at_unix': time.time(),
            }
            if getattr(self, 'rate_state_writer', None) is None:
                self._persist_rate_state(state)
        if getattr(self, 'rate_state_writer', None) is not None:
            self.rate_state_writer.submit(state)
        return cooldown

    def _persist_rate_state(self, state):
        """Legacy path; queue-aware clients use the independent bounded writer."""
        self.rate_state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.rate_state_file.with_name(
            f'.{self.rate_state_file.name}.tmp.{os.getpid()}.{threading.get_ident()}'
        )
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(temporary, self.rate_state_file)

    def request_json(
        self,
        task: str,
        model: str,
        system: str,
        prompt: str,
        media: list[tuple[str, bytes | TemporaryVideo]],
        json_schema: dict | None = None,
    ) -> tuple[dict, str]:
        if self.model_specific_admission:
            return self.client_for_model(model).request_json(
                task, model, system, prompt, media, json_schema,
            )
        remote_reference_only = bool(media) and all(
            isinstance(payload, (ArkFileReference, TemporaryVideo))
            for _, payload in media
        )
        # Memory is reserved before serialization, without holding HTTP slots
        # across upload resolution, pacing, cooldowns, retries, or parsing.
        if getattr(self.request_admission, 'fair_network', False):
            # Native Ark Files/COS references serialize to a few KiB and never
            # copy the referenced video/image bytes into this process.  The
            # frame executor and HTTP gate already bound these requests.  Do
            # not convoy thousands of URL-only requests through the weighted
            # Python memory queue; the queue remains mandatory for every
            # inline or mixed-media request where serialization can expand.
            if self.api in {'ark', 'las'} and task == 'ecot' and remote_reference_only:
                return self._request_json_admitted(
                    task, model, system, prompt, media, json_schema,
                )
            reservation = (6 * sum(len(payload) for _, payload in media)
                           + 6 * (len(system.encode()) + len(prompt.encode())) + 1024*1024)
            with self.request_admission.reserve(task, reservation):
                return self._request_json_admitted(task, model, system, prompt, media, json_schema)
        # Preserve legacy admission for callers that have not opted in.
        with self.http_limiter:
            reservation = (6 * sum(len(payload) for _, payload in media)
                           + 6 * (len(system.encode()) + len(prompt.encode())) + 1024*1024)
            context = (self.request_admission.admit(task, reservation)
                       if self.request_admission is not None else contextlib.nullcontext())
            with context:
                return self._request_json_admitted(task, model, system, prompt, media, json_schema)

    @contextlib.contextmanager
    def attempt_admission(self, task):
        from recoverable_service_circuit import ServiceAdmissionExpired
        fair = getattr(self.request_admission, 'fair_network', False)
        while True:
            circuit = (self.service_circuit.admit(self.ensure_available)
                       if self.service_circuit is not None else contextlib.nullcontext())
            try:
                with circuit as ticket:
                    def check_ticket():
                        self.ensure_available()
                        if self.service_circuit is not None and not self.service_circuit.allows(ticket):
                            raise ServiceAdmissionExpired()
                        if self.api in {'ark', 'las'} and self.request_pacer.dispatcher is not None:
                            if self.cooldown_until > time.monotonic():
                                raise ServiceAdmissionExpired()

                    pacing = self.request_admission.pacing(task) if fair else contextlib.nullcontext()
                    with pacing:
                        self.wait_for_provider()
                        if fair or self.service_circuit is not None:
                            # Cancel obsolete queued permits without consuming
                            # one pacing interval each ahead of the recovery probe.
                            self.request_pacer.wait(check=check_ticket)
                        else:
                            self.request_pacer.wait()
                    # Throttling can begin while a previously admitted request
                    # is queued at the pacer. Recheck before consuming HTTP slots.
                    check_ticket()
                    with contextlib.ExitStack() as admission:
                        if fair:
                            if not self.bypass_fixed_http_limiter:
                                admission.enter_context(self.http_limiter)
                            admission.enter_context(self.request_admission.network(task))
                        # Cover invalidation after pacing but before HTTP admission.
                        check_ticket()
                        yield
                        return
            except ServiceAdmissionExpired:
                # No HTTP attempt or retry was spent; wait at the service gate.
                continue

    def _request_json_admitted(
        self, task, model, system, prompt, media, json_schema=None,
    ) -> tuple[dict, str]:
        self.ensure_available()
        if task == 'cpa_student_coordinate':
            if self.api not in {'ark', 'las'} or len(media) != 1 or not media[0][0].startswith('video/'):
                raise ValueError('cpa_coordinate_student_requires_one_volc_video')
            # Preserve the exact validated 5 FPS student clip; generic media
            # fitting must not silently change its frames or final query.
            return self._request_json_serialized(task, model, system, prompt, media, json_schema)
        if any(isinstance(value, ArkFileReference) for _, value in media):
            if (self.api not in {'ark', 'las'} or task != 'ecot' or not all(
                    isinstance(value, ArkFileReference) for _, value in media)
                    or (self.api == 'las' and not all(
                        getattr(value, 'transport_method', None) == 'cos-presigned'
                        for _, value in media))):
                raise ValueError('volc_url_media_requires_ecot_and_supported_references')
        else:
            media = prepare_request_media(media, prepare_inline_media)
        if self.grd_media_transport == 'dashscope-temporary' and (task == 'grd' or task.startswith('grd_')):
            # Publish exactly the normalized bytes the inline path would send.
            # Keep files alive through all HTTP retries and URL-expiry refreshes.
            with contextlib.ExitStack() as resources:
                references = []
                for mime, value in media:
                    if not isinstance(value, TemporaryVideo):
                        self.ensure_available()
                        value = resources.enter_context(
                            self.grd_media_publisher.published_bytes(value, model, mime))
                    references.append((mime, value))
                return self._request_json_serialized(task, model, system, prompt, references, json_schema)
        return self._request_json_serialized(task, model, system, prompt, media, json_schema)

    def _request_json_las_operator(self, task, model, system, prompt, media, json_schema=None):
        from las_customer_ark_operator import LASOperatorError, token_usage
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            request_started = None
            feedback_ticket = None
            try:
                with self.attempt_admission(task):
                    request_started = time.monotonic()
                    feedback_ticket = self.request_pacer.record_start(request_started)
                    print(
                        f'api_request_start task={task} model={model} attempt={attempt} '
                        f'media_items={len(media)} transport=las_submit_poll '
                        f'started_at_unix={time.time():.6f}',
                        flush=True,
                    )
                    with self.http_transport_metrics.track():
                        raw, response = self.las_operator.call(
                            task, model, system, prompt, media, json_schema,
                        )
                self.http_limiter.record_success()
                self.request_pacer.record_success(request_started, feedback_ticket)
                usage = token_usage(response, model)
                print(
                    f'api_request_complete task={task} model={model} attempt={attempt} '
                    f'transport=las_submit_poll completed_at_unix={time.time():.6f} '
                    f'elapsed_seconds={time.monotonic() - request_started:.3f} '
                    f'completion_tokens={usage.get("completion_tokens")} '
                    f'prompt_tokens={usage.get("prompt_tokens")} '
                    f'cached_tokens={usage.get("cached_tokens")} reasoning_tokens=None '
                    f'finish_reason=completed',
                    flush=True,
                )
                try:
                    parsed = parse_json_object(raw)
                except (ValueError, KeyError, json.JSONDecodeError) as error:
                    self.las_operator.invalidate_last_completed()
                    raise LASOperatorError(f'las_operator_invalid_json:{error}') from error
                self.record_request_success()
                if task in {'cpa_semantic_review', 'cpa_student_coordinate', 'cpa_hand_side'}:
                    raw = json_dumps(response)
                return parsed, raw
            except LASOperatorError as error:
                last_error = error
                status = error.status or 0
                message = str(error)
                if status and is_fatal_provider_response(status, message):
                    self.record_fatal(status, message, task, model)
                    raise FatalProviderError(message) from error
                if status == 429:
                    self.record_rate_limit(
                        status, task, model, attempt, error.retry_after, message,
                        started_at=request_started, feedback_ticket=feedback_ticket,
                    )
                if attempt < self.max_attempts:
                    delay = max(error.retry_after or 0, min(60.0, 2 ** attempt))
                    print(
                        f'api_request_retry task={task} model={model} attempt={attempt} '
                        f'delay_seconds={delay:.3f} error={message[-500:]}', flush=True,
                    )
                    time.sleep(delay)
        raise RuntimeError(f'las_operator_request_failed:{last_error}')

    def _request_json_serialized(self, task, model, system, prompt, media, json_schema=None):
        from recoverable_service_circuit import transient_transport_timeout
        if self.las_operator is not None:
            return self._request_json_las_operator(
                task, model, system, prompt, media, json_schema,
            )
        content = []
        url_references = []
        has_ark_references = any(isinstance(value, ArkFileReference) for _, value in media)
        ark_files = self.api == 'ark' and not self.las_gateway and has_ark_references
        presigned_media = has_ark_references and any(
            getattr(value, 'transport_method', None) == 'cos-presigned'
            for _, value in media
        )
        for mime_type, payload in media:
            if isinstance(payload, ArkFileReference):
                if self.api == 'las' or self.las_gateway:
                    if (payload.model != model or payload.mime_type != mime_type
                            or getattr(payload, 'transport_method', None) != 'cos-presigned'
                            or not mime_type.startswith(('video/', 'image/'))):
                        raise ValueError('las_presigned_media_model_or_type_mismatch')
                    # Keep signed URLs out of queued request bodies.  The real
                    # URL is generated only after final HTTP admission below.
                    url = 'https://pending.invalid/media'
                    url_references.append((len(content), payload, url))
                    kind = 'video_url' if mime_type.startswith('video/') else 'image_url'
                    content.append({'type': kind, kind: {'url': url}})
                continue
            if isinstance(payload, TemporaryVideo):
                if (self.api != 'dashscope' or payload.model != model
                        or payload.mime_type != mime_type
                        or not mime_type.startswith(('video/', 'image/'))):
                    raise ValueError('temporary_video_provider_model_or_media_mismatch')
                url = payload.resolve()
                url_references.append((len(content), payload, url))
                kind = 'video_url' if mime_type.startswith('video/') else 'image_url'
                content.append({'type': kind, kind: {'url': url}})
                continue
            encoded = base64.b64encode(payload).decode('ascii')
            if mime_type.startswith('video/'):
                content.append({
                    'type': 'video_url',
                    'video_url': {'url': f'data:{mime_type};base64,{encoded}'},
                })
                if task == 'cpa_student_coordinate':
                    content[-1]['video_url']['fps'] = 5
            else:
                content.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{mime_type};base64,{encoded}'},
                })
            del encoded
        redact_external_media = bool(url_references) or presigned_media
        content.append({'type': 'text', 'text': prompt})
        response_format = {'type': 'json_object'}
        schema_supported = self.api in {'ark', 'las', 'openai-compatible'} or (
            self.api == 'dashscope' and model.startswith(('qwen3.8-flash', 'qwen3.8-max'))
        )
        if json_schema is not None and schema_supported:
            response_format = {'type': 'json_schema', 'json_schema': json_schema}
        request_endpoint = self.endpoint
        if ark_files:
            # Native Ark file IDs may require an upload/status poll. Keep that
            # network work outside inference admission; the cheap cached lookup
            # is repeated when the final payload is built. COS references are
            # deliberately excluded because their resolve operation signs the
            # expiring URL that must be created at dispatch time.
            for _, value in media:
                if getattr(value, 'transport_method', None) != 'cos-presigned':
                    value.resolve()
            # Do not resolve expiring media here. Thousands of workers can wait
            # in pacing/admission for longer than the URL lifetime.
            request_bytes = None
            request_endpoint = self.endpoint.removesuffix('/chat/completions') + '/responses'
        else:
            payload = {
                'model': model,
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': content},
                ],
                'temperature': 0,
                'response_format': response_format,
            }
            if self.api in {'ark', 'las'}:
                payload['thinking'] = {
                    'type': 'enabled' if task == 'cpa_semantic_review' else 'disabled',
                }
            else:
                payload['enable_thinking'] = False
            # Keep only the serialized body while waiting on HTTP. Previously
            # the Base64 content tree and its JSON bytes coexisted on retries.
            request_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            del payload
            if len(request_bytes) > REQUEST_BUDGET:
                raise ValueError(
                    f'request_body_exceeds_budget:{len(request_bytes)}:{REQUEST_BUDGET}'
                )
        del content
        last_error: BaseException | None = None
        json_mode = True
        throttled_attempts = 0
        transient_service_attempts = 0
        presigned_media_attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            self.ensure_available()
            request_started = None
            feedback_ticket = None
            feedback_classified = False
            attempt_status = None
            attempt_outcome = 'unclassified'
            try:
                with self.attempt_admission(task):
                    # Refresh URLs only after obtaining the final send permit.
                    # Signing is local and fast, and the resulting one-hour URL
                    # therefore cannot expire while this request waits in the
                    # high-concurrency pacing/admission queues.
                    if url_references:
                        request_bytes = refresh_request_urls(request_bytes, url_references)
                    if ark_files:
                        refreshed = responses_payload(
                            model, system, prompt, media, response_format,
                        )
                        if not json_mode:
                            refreshed.pop('text', None)
                        request_bytes = json.dumps(
                            refreshed, ensure_ascii=False,
                        ).encode('utf-8')
                        del refreshed
                        if presigned_media:
                            with self.rate_lock:
                                self.presigned_media_refreshes += sum(
                                    getattr(value, 'transport_method', None) == 'cos-presigned'
                                    for _, value in media
                                )
                    if request_bytes is None or len(request_bytes) > REQUEST_BUDGET:
                        size = 0 if request_bytes is None else len(request_bytes)
                        raise ValueError(
                            f'request_body_exceeds_budget:{size}:{REQUEST_BUDGET}'
                        )
                    request = urllib.request.Request(
                        request_endpoint, data=request_bytes, headers={
                            'Authorization': f'Bearer {os.environ[self.api_key_env]}',
                            'Content-Type': 'application/json',
                            **({'X-DashScope-Wait-Timeout': str(self.server_wait_seconds)}
                               if self.server_wait_seconds else {}),
                            **(RESOLVE_HEADER if url_references and self.api == 'dashscope' else {}),
                        },
                    )
                    request_started = time.monotonic()
                    feedback_ticket = self.request_pacer.record_start(request_started)
                    print(
                        f'api_request_start task={task} model={model} attempt={attempt} '
                        f'media_items={len(media)} request_bytes={len(request_bytes)} '
                        f'started_at_unix={time.time():.6f}',
                        flush=True,
                    )
                    with self.http_transport_metrics.track():
                        with urllib.request.urlopen(request, timeout=self.request_timeout_seconds + self.server_wait_seconds) as response:
                            attempt_status = getattr(response, 'status', 200)
                            response_bytes = response.read()
                    body = json.loads(response_bytes.decode('utf-8'))
                    del response_bytes
                    if ark_files:
                        body = responses_as_chat(body)
                attempt_outcome = 'http_success'
                self.http_limiter.record_success()
                self.request_pacer.record_success(request_started, feedback_ticket)
                feedback_classified = True
                raw = response_content(body)
                usage = body.get('usage') or {}
                reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
                print(
                    f'api_request_complete task={task} model={model} attempt={attempt} '
                    f'completed_at_unix={time.time():.6f} '
                    f'elapsed_seconds={time.monotonic() - request_started:.3f} '
                    f'completion_tokens={usage.get("completion_tokens")} '
                    f'prompt_tokens={usage.get("prompt_tokens")} '
                    f'cached_tokens={(usage.get("prompt_tokens_details") or {}).get("cached_tokens")} '
                    f'reasoning_tokens={reasoning_tokens} '
                    f'finish_reason={(body.get("choices") or [{}])[0].get("finish_reason")}',
                    flush=True,
                )
                try:
                    parsed = parse_json_object(raw)
                except (ValueError, KeyError) as error:
                    # Log response-only evidence, never request bodies or headers.
                    print('api_response_invalid ' + json_dumps({
                        'task': task, 'model': model, 'attempt': attempt,
                        'finish_reason': (body.get('choices') or [{}])[0].get('finish_reason'),
                        'completion_tokens': usage.get('completion_tokens'),
                        'error': str(error), 'response_chars': len(raw),
                        'response_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                        'response_head': redact_media_urls(raw[:512]) if redact_external_media else raw[:512],
                        'response_tail': redact_media_urls(raw[-512:]) if redact_external_media else raw[-512:],
                    }), flush=True)
                    raise
                self.record_request_success()
                if task in {'cpa_semantic_review', 'cpa_student_coordinate', 'cpa_hand_side'}:
                    # Review provenance includes model, token usage and reasoning output.
                    raw = json_dumps(body)
                return parsed, raw
            except urllib.error.HTTPError as error:
                attempt_status = error.code
                attempt_outcome = 'http_error'
                if error.code in {500, 502, 503, 504}:
                    transient_service_attempts += 1
                if self.burst_backoff is not None:
                    self.burst_backoff.record_failure()
                response_body = error.read().decode('utf-8', errors='replace')
                transient_presigned_media = (
                    presigned_media
                    and is_transient_presigned_media_response(error.code, response_body)
                )
                if redact_external_media:
                    response_body = redact_media_urls(response_body)
                last_error = RuntimeError(
                    f'{self.api}_http_error:{error.code}:{response_body[-2000:]}'
                )
                lowered = response_body.lower()
                if is_fatal_provider_response(error.code, response_body):
                    self.record_fatal(error.code, response_body, task, model)
                    raise FatalProviderError(str(last_error)) from error
                if (self.api == 'dashscope' and (task == 'ecot' or task.startswith('grd'))
                        and is_input_media_rejection(error.code, response_body)):
                    reject_current_video(task, model, response_body)
                if error.code == 429:
                    throttled_attempts += 1
                    cooldown = self.record_rate_limit(
                        error.code, task, model, attempt,
                        self.retry_after_seconds(error), response_body,
                        started_at=request_started,
                        feedback_ticket=feedback_ticket,
                    )
                    # A concurrency rejection bypasses rate feedback and must
                    # settle as unknown instead of leaking a sampled ticket.
                    feedback_classified = 'throttling.concurrency' not in lowered
                    attempt_outcome = ('burst_429' if is_burst_growth_throttle(lowered)
                                       else 'other_429')
                    print(
                        f'api_rate_limited task={task} model={model} attempt={attempt} '
                        f'global_cooldown_seconds={cooldown:.3f}',
                        flush=True,
                    )
                if transient_presigned_media:
                    presigned_media_attempts += 1
                    attempt_outcome = 'presigned_media_fetch_error'
                    with self.rate_lock:
                        self.presigned_media_fetch_errors += 1
                    print(
                        f'api_presigned_media_retry task={task} model={model} '
                        f'attempt={attempt} http_status={error.code}',
                        flush=True,
                    )
                elif error.code == 400 and any(
                    token in lowered for token in ('response_format', 'json_object', 'generation')
                ):
                    if json_mode:
                        fallback = json.loads(request_bytes)
                        fallback.pop('response_format', None)
                        if ark_files:
                            fallback.pop('text', None)
                        request_bytes = json.dumps(fallback, ensure_ascii=False).encode('utf-8')
                        del fallback
                        json_mode = False
                elif error.code == 400:
                    break
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
                if request_started is not None and transient_transport_timeout(error):
                    transient_service_attempts += 1
                attempt_outcome = ('timeout' if isinstance(error, TimeoutError) or
                                   isinstance(getattr(error, 'reason', None), TimeoutError)
                                   else type(error).__name__)
                if self.burst_backoff is not None:
                    self.burst_backoff.record_failure()
                last_error = RuntimeError(redact_media_urls(str(error))) if redact_external_media else error
            finally:
                if not feedback_classified:
                    self.request_pacer.record_unclassified(feedback_ticket)
                if request_started is not None and self.request_pacer.cohort_feedback is not None:
                    print('api_attempt_outcome '+json_dumps({
                        'task': task, 'model': model, 'attempt': attempt,
                        'started_monotonic': request_started, 'completed_at_unix': time.time(),
                        'elapsed_seconds': round(time.monotonic()-request_started, 3),
                        'http_status': attempt_status, 'outcome': attempt_outcome,
                        'feedback_sampled': feedback_ticket is not None,
                    }), flush=True)
            if attempt < self.max_attempts:
                print(
                    f'api_request_retry task={task} model={model} attempt={attempt} '
                    f'error={type(last_error).__name__}:{str(last_error)[-500:]}',
                    flush=True,
                )
                self.wait_for_provider()
                time.sleep(min(60, 2 ** attempt) + random.uniform(0, min(10, 2 ** attempt)))
        if ((self.service_circuit is not None or (self.api == 'dashscope' and self.server_wait_seconds > 0))
                and throttled_attempts == self.max_attempts):
            # Transient throttling must not permanently stop independent models.
            # Keep the HTTP and record retry budgets; partial frame checkpoints
            # survive exhaustion and remain available to a later resume.
            print(f'api_throttle_retries_exhausted task={task} model={model} '
                  f'attempts={throttled_attempts} provider_circuit_unchanged=true', flush=True)
            raise RuntimeError(f'{self.api}_request_throttled_retries_exhausted:{last_error}')
        if (self.service_circuit is not None and transient_service_attempts > 0
                and transient_service_attempts + throttled_attempts == self.max_attempts):
            # Retries remain bounded. The model circuit controls service recovery
            # outside HTTP slots; this episode's durable checkpoints remain valid.
            raise RuntimeError(f'{self.api}_transient_service_retries_exhausted:{last_error}')
        if (presigned_media_attempts > 0
                and presigned_media_attempts + transient_service_attempts
                + throttled_attempts == self.max_attempts):
            # A provider-side media fetch failure is scoped to this temporary
            # object/request. Preserve successful frame checkpoints and let the
            # record retry later; never stop every unrelated episode.
            print(f'api_presigned_media_retries_exhausted task={task} model={model} '
                  f'attempts={presigned_media_attempts} provider_circuit_unchanged=true',
                  flush=True)
            raise RuntimeError(f'{self.api}_presigned_media_retries_exhausted:{last_error}')
        with self.rate_lock:
            self.request_failures += 1
            failed = self.request_failures
        if failed >= self.max_consecutive_request_failures:
            message = f'provider_circuit_open:consecutive_failed_requests={failed}:{last_error}'
            self.record_fatal(0, message, task, model)
            raise FatalProviderError(message)
        raise RuntimeError(f'{self.api}_request_failed:{last_error}')


def vla_task_name(prompt: str) -> str | None:
    """Extract a compact global task name from the normal subtask prompt."""
    context = prompt.rsplit('Global task context:\n', 1)[-1]
    context = re.sub(r'\s+', ' ', context).strip()
    patterns = (
        r'(?i)(?:the\s+)?(?:agent|robot)(?:\s+in\s+the\s+video)?\s+was\s+given\s+'
        r'the\s+instruction\s*[-:]\s*(.+?)(?:[?.]|$)',
        r'(?i)current\s+goal\s+is\s*:\s*(.+?)(?:[?.]|\blast\s+\d+\s+steps\s*:|$)',
        r'(?i)^(?:task\s+instruction|task|instruction)\s*:\s*(.+)$',
    )
    selected = None
    for pattern in patterns:
        match = re.search(pattern, context)
        if match:
            selected = match.group(1).strip()
            break
    if selected is None:
        assistant_parts = re.split(r'\bassistant\s*:\s*', context, flags=re.I)
        selected = assistant_parts[-1].strip() if len(assistant_parts) > 1 else context
        answer = re.search(r'<answer>\s*(.*?)\s*</answer>', selected, flags=re.I | re.S)
        if answer:
            selected = answer.group(1).strip()
        else:
            selected = re.sub(r'<think>.*?</think>', '', selected, flags=re.I | re.S).strip()
    selected = re.sub(r'^(?:user|assistant)\s*:\s*', '', selected, flags=re.I)
    return selected[:VLA_TASK_NAME_MAX_LENGTH].strip() or None


def parse_vla_anno_content(raw: str) -> list[dict]:
    """Parse the fenced JSON array emitted by the vla-anno operator."""
    value = raw.strip()
    fenced = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', value, flags=re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    parsed = json.loads(value)
    if isinstance(parsed, dict):
        parsed = parsed.get('subtasks') or parsed.get('steps')
    if not isinstance(parsed, list) or not parsed:
        raise ValueError('vla_anno_result_requires_nonempty_array')
    if not all(isinstance(item, dict) for item in parsed):
        raise ValueError('vla_anno_result_items_must_be_objects')
    return parsed


def vla_skill_action(skill: object) -> str:
    value = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', str(skill or '')).strip()
    value = re.sub(r'[_-]+', ' ', value)
    return re.sub(r'\s+', ' ', value).lower() or 'None'


def video_path_timing(path: str | Path) -> tuple[float, int, float]:
    capture = open_video_capture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if fps <= 0 or count <= 0:
        raise ValueError('vla_anno_video_timing_unavailable')
    return fps, count, count / fps


def write_video_payload(destination, payload: bytes | FileSlice) -> None:
    if isinstance(payload, FileSlice):
        payload.write_to(destination)
    else:
        destination.write(payload)


def decode_video_path(payload: bytes | FileSlice) -> tuple[str, str]:
    """Borrow a complete immutable video file or return an owned temporary copy."""
    if isinstance(payload, FileSlice) and payload.offset == 0:
        if payload.path.stat().st_size == payload.length:
            return str(payload.path), ''
    name = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            name = stream.name
            write_video_payload(stream, payload)
        return name, name
    except BaseException:
        if name:
            Path(name).unlink(missing_ok=True)
        raise


def contact_source_media(media, owner, media_factory=None):
    """Reuse the full-resolution source, never the teacher's 2 FPS encoding."""
    result = list(media)
    for index, (mime, payload) in enumerate(media):
        if not mime.startswith('video/'):
            continue
        source = getattr(media_factory, 'source_path', None)
        if source is None:
            source, temporary = decode_video_path(payload)
            source = Path(source)
            if temporary:
                owner.callback(Path(temporary).unlink, missing_ok=True)
        result[index] = (mime, FileSlice(source, 0, source.stat().st_size))
        break
    return result


def video_frame_timing(payload: bytes | FileSlice) -> tuple[float, int, float]:
    name = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            write_video_payload(stream, payload)
            name = stream.name
        return video_path_timing(name)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def vla_anno_to_subtasks(
    raw: str,
    media: list[tuple[str, bytes | FileSlice]],
    task_name: str | None,
    video_timing: tuple[float, int, float] | None = None,
) -> dict:
    """Convert operator frame boundaries to the integrated subtask contract."""
    items = parse_vla_anno_content(raw)
    has_video = any(mime.startswith('video/') for mime, _ in media)
    if has_video:
        if len(media) != 1 or not media[0][0].startswith('video/'):
            raise ValueError('vla_anno_requires_one_video_or_an_ordered_frame_sequence')
        fps, frame_count, duration = video_timing or video_frame_timing(media[0][1])
        maximum = duration
    else:
        fps, frame_count, maximum = 1.0, len(media), float(max(0, len(media) - 1))

    try:
        provider_start = float(items[0].get('start_frame'))
        provider_end = float(items[-1].get('end_frame'))
    except (TypeError, ValueError):
        raise ValueError('vla_anno_invalid_outer_frame_range') from None
    if provider_end <= provider_start:
        raise ValueError(
            f'vla_anno_invalid_outer_frame_range:{provider_start}:{provider_end}'
        )

    steps = []
    previous: float | int = 0.0 if has_video else 0
    for source_index, item in enumerate(items):
        last = source_index == len(items) - 1
        try:
            raw_end = float(item.get('end_frame'))
        except (TypeError, ValueError):
            raise ValueError(f'vla_anno_invalid_end_frame:{source_index + 1}') from None
        relative_end = min(1.0, max(0.0, (raw_end - provider_start) / (provider_end - provider_start)))
        if has_video:
            end: float | int = maximum if last else maximum * relative_end
            if float(end) <= float(previous) + 1e-6:
                continue
        else:
            end = int(maximum) if last else int(round(maximum * relative_end))
            if int(end) < int(previous):
                continue

        description = str(item.get('description_en') or item.get('description') or '').strip()
        if not description:
            raise ValueError(f'vla_anno_missing_description:{source_index + 1}')
        step = {
            'id': len(steps) + 1,
            'subtask': description.lower(),
            'action': vla_skill_action(item.get('skill')),
            'object': 'None',
            'source': 'None',
            'target': 'None',
            'provider_frame_range': {
                'start_frame': item.get('start_frame'),
                'end_frame': item.get('end_frame'),
            },
        }
        if has_video:
            step.update({
                'media_start_index': None,
                'media_end_index': None,
                'start_time_seconds': float(previous),
                'end_time_seconds': float(end),
            })
        else:
            step.update({
                'media_start_index': int(previous),
                'media_end_index': int(end),
                'start_time_seconds': None,
                'end_time_seconds': None,
            })
        steps.append(step)
        previous = end

    if not steps:
        raise ValueError('vla_anno_result_has_no_valid_ranges')
    if has_video:
        steps[-1]['end_time_seconds'] = duration
    else:
        steps[-1]['media_end_index'] = frame_count - 1
    return {
        'task_summary': task_name or '',
        'subtasks': steps,
        'provider_output': {
            'schema_version': 'vla-anno-frame-subtasks/v1',
            'boundary_mapping': {
                'method': 'normalize_provider_outer_frame_range',
                'provider_start_frame': provider_start,
                'provider_end_frame': provider_end,
                'source_frame_count': frame_count,
                'source_fps': fps if has_video else None,
                'source_duration_seconds': maximum if has_video else None,
                'source_media_count': None if has_video else frame_count,
            },
            'raw_subtasks': items,
        },
    }


def vla_chunk_ranges(
    duration: float,
    chunk_seconds: float = VLA_CHUNK_SECONDS,
) -> list[tuple[float, float]]:
    """Return balanced source offsets/spans within the operator duration limit."""
    if duration <= 300.0:
        return [(0.0, duration)]
    chunk_count = max(2, math.ceil(duration / chunk_seconds))
    nominal_span = duration / chunk_count
    ranges = []
    start = 0.0
    for chunk_index in range(chunk_count):
        span = duration - start if chunk_index == chunk_count - 1 else nominal_span
        ranges.append((start, span))
        start += span
    return ranges


def extract_vla_video_chunk(media_path: str, start: float, span: float) -> str:
    """Materialize one stream-copy chunk and return its temporary path."""
    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    chunk_path = temporary.name
    temporary.close()
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-ss', f'{start:.6f}', '-i', media_path, '-t', f'{span:.6f}',
        '-map', '0:v:0', '-an', '-c:v', 'copy',
        '-avoid_negative_ts', 'make_zero', chunk_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if (
        completed.returncode or not Path(chunk_path).is_file()
        or not Path(chunk_path).stat().st_size
    ):
        Path(chunk_path).unlink(missing_ok=True)
        raise RuntimeError(
            f'ffmpeg_vla_chunk_failed:start={start}:span={span}:'
            f'{completed.stderr[-1000:]}'
        )
    return chunk_path


def split_video_for_vla(
    media_path: str,
    duration: float,
    chunk_seconds: float = VLA_CHUNK_SECONDS,
) -> list[tuple[str, float, float, bool]]:
    """Compatibility helper that materializes all balanced chunks."""
    ranges = vla_chunk_ranges(duration, chunk_seconds)
    if len(ranges) == 1:
        return [(media_path, ranges[0][0], ranges[0][1], False)]
    chunks = []
    try:
        for start, span in ranges:
            chunk_path = extract_vla_video_chunk(media_path, start, span)
            chunks.append((chunk_path, start, span, True))
    except Exception:
        for chunk_path, _, _, remove in chunks:
            if remove:
                Path(chunk_path).unlink(missing_ok=True)
        raise
    return chunks


def merge_vla_chunk_results(
    chunks: list[tuple[dict, str, float, float]],
    task_name: str | None,
    source_timing: tuple[float, int, float],
) -> dict:
    """Map independent chunk timelines onto one contiguous source timeline."""
    source_fps, source_frames, source_duration = source_timing
    merged = []
    chunk_audit = []
    for chunk_index, (result, raw, start, span) in enumerate(chunks):
        provider_output = result.get('provider_output') or {}
        local_duration = float(
            (provider_output.get('boundary_mapping') or {}).get(
                'source_duration_seconds', span,
            ) or span
        )
        scale = span / local_duration if local_duration > 0 else 1.0
        mapped_steps = []
        for step in result['subtasks']:
            mapped = copy.deepcopy(step)
            mapped['start_time_seconds'] = start + float(step['start_time_seconds']) * scale
            mapped['end_time_seconds'] = start + float(step['end_time_seconds']) * scale
            mapped_steps.append(mapped)
        if mapped_steps:
            mapped_steps[0]['start_time_seconds'] = start
            mapped_steps[-1]['end_time_seconds'] = start + span
        for mapped in mapped_steps:
            if merged and all(
                merged[-1].get(field) == mapped.get(field)
                for field in ('subtask', 'action', 'object', 'source', 'target')
            ):
                merged[-1]['end_time_seconds'] = mapped['end_time_seconds']
                merged[-1].setdefault('provider_frame_ranges', []).append(
                    mapped.get('provider_frame_range')
                )
            else:
                mapped['provider_chunk_index'] = chunk_index
                merged.append(mapped)
        chunk_audit.append({
            'chunk_index': chunk_index,
            'source_start_seconds': start,
            'source_duration_seconds': span,
            'provider_output': provider_output,
            'raw_response': raw,
        })
    if not merged:
        raise ValueError('vla_anno_chunk_merge_has_no_subtasks')
    for index, step in enumerate(merged, 1):
        step['id'] = index
        if index > 1:
            step['start_time_seconds'] = merged[index - 2]['end_time_seconds']
    merged[0]['start_time_seconds'] = 0.0
    merged[-1]['end_time_seconds'] = source_duration
    return {
        'task_summary': task_name or '',
        'subtasks': merged,
        'provider_output': {
            'schema_version': 'vla-anno-chunked-subtasks/v1',
            'chunk_seconds': VLA_CHUNK_SECONDS,
            'source_frame_count': source_frames,
            'source_fps': source_fps,
            'source_duration_seconds': source_duration,
            'chunks': chunk_audit,
        },
    }


def write_vla_media_file(media: list[tuple[str, bytes | FileSlice]]) -> str:
    """Write one video or encode an ordered image sequence for the SDK."""
    if len(media) == 1 and media[0][0].startswith('video/'):
        suffix = mimetypes.guess_extension(media[0][0]) or '.mp4'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
            write_video_payload(stream, media[0][1])
            stream.flush()
            # Disposable media only needs to be visible to the uploader/decoder.
            # Durable annotation records and recovery checkpoints still fsync.
            return stream.name
    if not media or any(not mime.startswith('image/') for mime, _ in media):
        raise ValueError('vla_anno_media_must_be_one_video_or_images')
    frames = []
    for index, (_, payload) in enumerate(media):
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f'vla_anno_frame_decode_failed:{index}')
        frames.append(frame)
    height, width = frames[0].shape[:2]
    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    name = temporary.name
    temporary.close()
    writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 1.0, (width, height))
    if not writer.isOpened():
        Path(name).unlink(missing_ok=True)
        raise RuntimeError('vla_anno_frame_video_writer_open_failed')
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return name


def transcode_vla_upload_if_needed(
    media_path: str, duration: float,
) -> tuple[str, bool]:
    """Avoid the SDK's highly threaded >100 MB transcoder under tight cgroups."""
    source_size = Path(media_path).stat().st_size
    if source_size <= VLA_DIRECT_UPLOAD_BYTES:
        return media_path, False
    temporary = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    destination = temporary.name
    temporary.close()
    started = time.monotonic()
    print(
        f'vla_local_transcode_start source_bytes={source_size} '
        f'duration_seconds={duration:.3f}', flush=True,
    )
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', media_path,
        '-vf', 'scale=640:480', '-filter_threads', '2',
        '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2',
        '-b:v', str(VLA_LOCAL_VIDEO_BITRATE),
        '-maxrate', str(VLA_LOCAL_VIDEO_BITRATE),
        '-bufsize', str(VLA_LOCAL_VIDEO_BITRATE * 2),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', destination,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    output_size = Path(destination).stat().st_size if Path(destination).is_file() else 0
    if completed.returncode or not output_size or output_size > VLA_DIRECT_UPLOAD_BYTES:
        Path(destination).unlink(missing_ok=True)
        raise RuntimeError(
            f'ffmpeg_vla_local_transcode_failed:returncode={completed.returncode}:'
            f'source_bytes={source_size}:output_bytes={output_size}:'
            f'{completed.stderr[-1000:]}'
        )
    print(
        f'vla_local_transcode_complete source_bytes={source_size} '
        f'output_bytes={output_size} elapsed_seconds={time.monotonic() - started:.3f}',
        flush=True,
    )
    return destination, True


class VlaAnnoClient:
    """Adapter for the fixed vla-anno subtask operator SDK."""

    TERMINAL_ERROR_TOKENS = (
        'billingautherror', 'in arrears', 'arrearage', 'invalid api key',
        'unauthorized', 'authentication failed', 'permission denied',
    )
    THROTTLE_ERROR_TOKENS = (
        'throttling.concurrency', 'limit_requests', 'limit_burst_rate',
        'too many requests', 'http 429', 'status code 429',
    )

    def __init__(self, args: argparse.Namespace):
        self.api = args.api
        self.api_key_env = args.api_key_env
        self.max_attempts = args.max_attempts
        self.fatal_stop_file = args.fatal_stop_file
        self.rate_state_file = args.rate_state_file
        initial_http_limit = args.max_http_active
        initial_request_interval = args.request_start_interval
        restored = False
        if self.rate_state_file.is_file():
            try:
                state = json.loads(self.rate_state_file.read_text(encoding='utf-8'))
                last_error = str(state.get('last_error') or '').lower()
                if (
                    state.get('provider') == self.api
                    and not last_error.startswith('vla_anno_finish_reason:')
                ):
                    initial_http_limit = min(
                        args.max_http_active,
                        max(1, int(state.get('effective_http_limit') or args.max_http_active)),
                    )
                    initial_request_interval = max(
                        args.request_start_interval,
                        float(state.get('request_start_interval_seconds') or 0.0),
                    )
                    restored = True
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        self.request_pacer = RequestStartPacer(initial_request_interval)
        self.http_limiter = AdaptiveConcurrencyLimiter(
            args.max_http_active, initial_http_limit,
        )
        self.media_limiter = threading.BoundedSemaphore(
            max(1, int(getattr(args, 'vla_media_workers', 8))),
        )
        self.rate_lock = threading.Lock()
        self.rate_limit_events = 0
        if restored:
            print(
                f'api_rate_state_restored effective_http_limit={initial_http_limit} '
                f'request_start_interval_seconds={initial_request_interval}',
                flush=True,
            )
        sdk_dir = args.vla_sdk_dir.resolve()
        if str(sdk_dir) not in sys.path:
            sys.path.insert(0, str(sdk_dir))
        try:
            from embodied_vl_client import EmbodiedVLClient
        except ImportError as error:
            raise RuntimeError(f'vla_anno_sdk_import_failed:{sdk_dir}:{error}') from error
        self.client = EmbodiedVLClient(
            api_key=os.environ.get(self.api_key_env),
            app_id=os.environ.get('DASHSCOPE_APP_ID'),
            user_id=os.environ.get('DASHSCOPE_USER_ID'),
            device_uuid=os.environ.get('DASHSCOPE_DEVICE_UUID'),
            version=args.models['subtask'],
        )

    def ensure_available(self) -> None:
        if self.fatal_stop_file.is_file():
            raise FatalProviderError(f'fatal_stop_file_exists:{self.fatal_stop_file}')
        required = (
            self.api_key_env, 'DASHSCOPE_APP_ID',
            'DASHSCOPE_USER_ID', 'DASHSCOPE_DEVICE_UUID',
        )
        missing = [name for name in required if not os.environ.get(name, '').strip()]
        if missing:
            raise RuntimeError(f'missing_vla_anno_environment_variables:{missing}')

    def record_fatal(self, message: str, task: str, model: str) -> None:
        self.fatal_stop_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.fatal_stop_file.with_suffix('.tmp')
        temporary.write_text(json.dumps({
            'error': message[-2000:], 'task': task, 'model': model,
            'provider': self.api, 'created_at_unix': time.time(),
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, self.fatal_stop_file)

    def record_rate_state(
        self,
        task: str,
        model: str,
        attempt: int,
        message: str,
        effective_limit: int,
        active_requests: int,
        limit_changed: bool,
        request_interval: float,
    ) -> None:
        with self.rate_lock:
            self.rate_limit_events += 1
            state = {
                'schema_version': 'vqa-provider-rate-state/v1',
                'provider': self.api,
                'task': task,
                'model': model,
                'attempt': attempt,
                'rate_limit_events': self.rate_limit_events,
                'effective_http_limit': effective_limit,
                'active_http_requests': active_requests,
                'effective_limit_changed': limit_changed,
                'request_start_interval_seconds': request_interval,
                'last_error': message[-2000:],
                'updated_at_unix': time.time(),
            }
            self.rate_state_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.rate_state_file.with_name(
                f'.{self.rate_state_file.name}.tmp.{os.getpid()}.'
                f'{threading.get_ident()}'
            )
            temporary.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + '\n',
                encoding='utf-8',
            )
            os.replace(temporary, self.rate_state_file)

    def _request_raw(
        self,
        task: str,
        model: str,
        media_path: str,
        task_name: str | None,
        chunk_index: int,
        chunk_count: int,
    ) -> str:
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.ensure_available()
            try:
                with self.http_limiter:
                    self.request_pacer.wait()
                    started = time.monotonic()
                    print(
                        f'api_request_start task={task} model={model} attempt={attempt} '
                        f'chunk={chunk_index + 1}/{chunk_count}', flush=True,
                    )
                    response = self.client.chat.completions.create(
                        messages=[{
                            'role': 'user',
                            'content': [{
                                'type': 'video_url',
                                'video_url': {'url': media_path},
                            }],
                        }],
                        task_name=task_name,
                    )
                choice = response.choices[0]
                raw = choice.message.content or ''
                if choice.finish_reason != 'stop' or not raw:
                    error_message = f'vla_anno_finish_reason:{choice.finish_reason}'
                    print(
                        f'api_request_failed_response task={task} model={model} '
                        f'attempt={attempt} chunk={chunk_index + 1}/{chunk_count} '
                        f'finish_reason={choice.finish_reason}',
                        flush=True,
                    )
                    raise RuntimeError(error_message)
                self.http_limiter.record_success()
                self.request_pacer.record_success()
                print(
                    f'api_request_complete task={task} model={model} attempt={attempt} '
                    f'chunk={chunk_index + 1}/{chunk_count} '
                    f'elapsed_seconds={time.monotonic() - started:.3f}', flush=True,
                )
                return raw
            except FatalProviderError:
                raise
            except Exception as error:
                last_error = error
                message = f'{type(error).__name__}:{error}'
                if any(token in message.lower() for token in self.TERMINAL_ERROR_TOKENS):
                    self.record_fatal(message, task, model)
                    raise FatalProviderError(message) from error
                if any(token in message.lower() for token in self.THROTTLE_ERROR_TOKENS):
                    effective_limit, active_requests, changed = (
                        self.http_limiter.reduce_after_throttle()
                    )
                    request_interval = self.request_pacer.record_throttle(message)
                    self.record_rate_state(
                        task, model, attempt, message, effective_limit,
                        active_requests, changed, request_interval,
                    )
                    print(
                        f'api_rate_limited task={task} model={model} attempt={attempt} '
                        f'effective_http_limit={effective_limit} '
                        f'active_http_requests={active_requests} '
                        f'effective_limit_changed={changed} '
                        f'request_start_interval_seconds={request_interval}',
                        flush=True,
                    )
                if attempt < self.max_attempts:
                    delay = min(60.0, 2 ** attempt) + random.uniform(0, min(5, 2 ** attempt))
                    print(
                        f'api_request_retry task={task} model={model} attempt={attempt} '
                        f'delay_seconds={delay:.3f} error={message[-500:]}', flush=True,
                    )
                    time.sleep(delay)
        raise RuntimeError(f'vla_anno_request_failed:{last_error}')

    def request_json(
        self,
        task: str,
        model: str,
        system: str,
        prompt: str,
        media: list[tuple[str, bytes | FileSlice]],
        checkpoint: AnnotationUnitCheckpoint | None = None,
    ) -> tuple[dict, str]:
        del system
        if task != 'subtask':
            raise RuntimeError(f'vla_anno_unsupported_task:{task}')
        self.ensure_available()
        task_name = vla_task_name(prompt)
        media_context = getattr(self, 'media_limiter', contextlib.nullcontext())
        with media_context:
            media_path = write_vla_media_file(media)
            has_video = len(media) == 1 and media[0][0].startswith('video/')
            source_timing = video_path_timing(media_path) if has_video else None
            if has_video and source_timing[2] <= 300.0:
                upload_path, replace_source = transcode_vla_upload_if_needed(
                    media_path, source_timing[2],
                )
                if replace_source:
                    Path(media_path).unlink(missing_ok=True)
                    media_path = upload_path
        try:
            if not has_video:
                key = 'vla_chunk:0/1:ordered_frames'
                cached = checkpoint.get(key) if checkpoint is not None else None
                if isinstance(cached, dict):
                    result = copy.deepcopy(cached['result'])
                    raw = str(cached['raw'])
                else:
                    raw = self._request_raw(task, model, media_path, task_name, 0, 1)
                    result = vla_anno_to_subtasks(raw, media, task_name)
                    if checkpoint is not None:
                        checkpoint.put(key, {'result': result, 'raw': raw})
                return result, json_dumps(result)
            ranges = vla_chunk_ranges(source_timing[2])
            if len(ranges) == 1:
                key = f'vla_chunk:0/1:0.000000:{source_timing[2]:.6f}'
                cached = checkpoint.get(key) if checkpoint is not None else None
                if isinstance(cached, dict):
                    result = copy.deepcopy(cached['result'])
                    raw = str(cached['raw'])
                else:
                    raw = self._request_raw(task, model, media_path, task_name, 0, 1)
                    result = vla_anno_to_subtasks(
                        raw, media, task_name, video_timing=source_timing,
                    )
                    if checkpoint is not None:
                        checkpoint.put(key, {'result': result, 'raw': raw})
                return result, json_dumps(result)
            results = []
            for index, (start, span) in enumerate(ranges):
                key = (
                    f'vla_chunk:{index}/{len(ranges)}:'
                    f'{start:.6f}:{span:.6f}'
                )
                cached = checkpoint.get(key) if checkpoint is not None else None
                if isinstance(cached, dict):
                    results.append((
                        copy.deepcopy(cached['result']), str(cached['raw']),
                        start, span,
                    ))
                    continue
                with media_context:
                    chunk_path = extract_vla_video_chunk(media_path, start, span)
                    chunk_timing = video_path_timing(chunk_path)
                    upload_path, replace_chunk = transcode_vla_upload_if_needed(
                        chunk_path, chunk_timing[2],
                    )
                    if replace_chunk:
                        Path(chunk_path).unlink(missing_ok=True)
                        chunk_path = upload_path
                try:
                    raw = self._request_raw(
                        task, model, chunk_path, task_name, index, len(ranges),
                    )
                    result = vla_anno_to_subtasks(
                        raw, [('video/mp4', FileSlice(
                            Path(chunk_path), 0, Path(chunk_path).stat().st_size,
                        ))], task_name,
                        video_timing=chunk_timing,
                    )
                    if checkpoint is not None:
                        checkpoint.put(key, {'result': result, 'raw': raw})
                finally:
                    Path(chunk_path).unlink(missing_ok=True)
                results.append((result, raw, start, span))
            result = merge_vla_chunk_results(
                results, task_name, source_timing,
            )
            return result, json_dumps(result)
        finally:
            Path(media_path).unlink(missing_ok=True)


def task_context(record: dict) -> str:
    task_name = str(record.get('task_name') or '').strip()
    if task_name:
        return f'Task instruction: {task_name}'
    task_text_ref = record.get('_task_text_ref')
    if isinstance(task_text_ref, FileSlice):
        text = task_text_ref.read().decode('utf-8', 'replace').strip()
        if text:
            return f'Task instruction: {text}'[-12000:]
    turns = (record.get('dialogue') or {}).get('turns') or []
    return '\n'.join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns
    )[-12000:]


def is_temporal(
    record: dict,
    temporal_media_types: set[str] | None = None,
) -> bool:
    items = (record.get('media') or {}).get('items') or []
    accepted = temporal_media_types or {'video', 'frame'}
    return any(str(item.get('type') or '').lower() in accepted for item in items)


def media_item_mime_type(item: dict) -> str:
    explicit = str(item.get('mime_type') or '').strip().lower()
    if explicit:
        return explicit
    guessed, _ = mimetypes.guess_type(str(item.get('member') or ''))
    if guessed:
        return guessed
    media_type = str(item.get('type') or '').lower()
    if media_type == 'video':
        return 'video/mp4'
    if media_type in {'image', 'frame'}:
        return 'image/jpeg'
    return 'application/octet-stream'


def iter_wds_samples(tar_path: Path, temporal_media_types: set[str]):
    with tarfile.open(tar_path, 'r:') as archive:
        record = None
        include_record = False
        blobs: dict[str, tuple[str, bytes]] = {}
        expected: dict[str, str] = {}
        for member in archive:
            if member.name.endswith('.json'):
                if record is not None and include_record:
                    yield record, blobs
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f'tar_member_unreadable:{tar_path}:{member.name}')
                record = json.load(stream)
                include_record = is_temporal(record, temporal_media_types)
                blobs = {}
                expected = {
                    str(item['member']): media_item_mime_type(item)
                    for item in (record.get('media') or {}).get('items') or []
                    if include_record and item.get('member')
                }
            elif record is not None and include_record and member.name in expected:
                stream = archive.extractfile(member)
                if stream is None:
                    raise RuntimeError(f'tar_member_unreadable:{tar_path}:{member.name}')
                blobs[member.name] = (expected[member.name], stream.read())
        if record is not None and include_record:
            yield record, blobs


def cosmos3_physical_root(input_path: Path) -> Path | None:
    """Resolve a logical v1.5 release to its sibling physical WebDataset."""
    candidates = [input_path]
    candidates.append(input_path.with_name(f'{input_path.name}-physical-webdataset'))
    for candidate in candidates:
        release_path = candidate/'RELEASE.json'
        if not release_path.is_file():
            continue
        try:
            release = json.loads(release_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            release.get('schema') == COSMOS3_PHYSICAL_SCHEMA
            and (candidate/'catalog'/'samples.parquet').is_file()
            and (candidate/'catalog'/'members.parquet').is_file()
            and (candidate/'shards').is_dir()
        ):
            return candidate.resolve()
    return None


def cosmos3_shard_location(root: Path, shard_name: str) -> Path:
    """Return the canonical shard path without touching remote storage."""
    match = re.fullmatch(r'train-(\d{6})\.tar', shard_name)
    if not match:
        raise ValueError(f'unsupported_cosmos3_shard_name:{shard_name}')
    return root/'shards'/f'{int(match.group(1)) % 64:02x}'/shard_name


def cosmos3_shard_path(root: Path, shard_name: str) -> Path:
    path = cosmos3_shard_location(root, shard_name)
    if not path.is_file():
        raise FileNotFoundError(f'cosmos3_shard_missing:{path}')
    return path


def cosmos3_primary_view(video_members: dict[str, str]) -> str:
    for view in ('ego', 'head_left', 'head_right', 'left_wrist', 'right_wrist'):
        if view in video_members:
            return view
    if not video_members:
        raise ValueError('cosmos3_sample_has_no_video_members')
    return sorted(video_members)[0]


def cosmos3_physical_sample(
    root: Path,
    row: dict,
    members: dict[str, dict],
    shard_paths: dict[str, Path] | None = None,
    verify_shard: bool = True,
) -> tuple[dict, dict[str, tuple[str, FileSlice]]]:
    video_members = json.loads(row['video_members_json'])
    view = cosmos3_primary_view(video_members)
    video_member = str(video_members[view])
    member = members.get(video_member)
    if member is None:
        raise RuntimeError(f'cosmos3_video_member_missing:{row["key"]}:{video_member}')
    shard_name = str(row['shard'])
    tar_path = shard_paths.get(shard_name) if shard_paths is not None else None
    if tar_path is None:
        # A full resume scans hundreds of thousands of already-completed rows.
        # Avoid a serial COSFS stat for media that will never be opened. The
        # FileSlice read remains authoritative for every unfinished episode.
        tar_path = (cosmos3_shard_path(root, shard_name) if verify_shard
                    else cosmos3_shard_location(root, shard_name))
        if shard_paths is not None:
            shard_paths[shard_name] = tar_path
    video_ref = FileSlice(
        tar_path, int(member['data_offset']), int(member['length']),
    )
    text_member = f'{row["key"]}.txt'
    text = members.get(text_member)
    text_ref = (
        FileSlice(tar_path, int(text['data_offset']), int(text['length']))
        if text is not None else None
    )
    task_name = select_vla_task_name(
        row.get('task_description'), row.get('task_name'),
    ) or ''
    source = {
        'schema_version': 'unified-vqa-record/v2',
        'uid': str(row['record_uid']),
        'source_key': str(row['source_key']),
        'split': 'train',
        'task_name': task_name,
        'dialogue': {'turns': []},
        'media': {
            'layout': 'sequence',
            'items': [{
                'media_id': 'media-0',
                'type': 'video',
                'mime_type': 'video/mp4',
                'member': video_member,
                'relative_path': str(tar_path),
                'container_format': 'tar',
                'view': view,
            }],
        },
        'provenance': {
            'source_record_uid': str(row['record_uid']),
            'source_locator': {
                'storage_kind': 'tar_member',
                'source_file': str(tar_path),
                'member': video_member,
                'physical_dataset_root': str(root),
                'catalog_key': str(row['key']),
                'view': view,
            },
        },
    }
    target_duration_ns = row.get('target_duration_ns')
    if isinstance(target_duration_ns, (int, float)) and target_duration_ns > 0:
        source['_duration_seconds'] = float(target_duration_ns) / 1_000_000_000.0
    if text_ref is not None:
        source['_task_text_ref'] = text_ref
    return source, {video_member: ('video/mp4', video_ref)}


def iter_cosmos3_physical_batches(
    root: Path,
    source_filter: set[str],
    batch_size: int,
    max_batches: int | None,
    record_uids: set[str] | None = None,
):
    """Yield lightweight catalog batches; media bytes stay lazy until a worker runs."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError('pyarrow_required_for_cosmos3_physical_webdataset') from error
    samples_file = pq.ParquetFile(root/'catalog'/'samples.parquet')
    members_file = pq.ParquetFile(root/'catalog'/'members.parquet')
    if samples_file.num_row_groups != members_file.num_row_groups:
        raise RuntimeError(
            'cosmos3_catalog_row_group_mismatch:'
            f'{samples_file.num_row_groups}:{members_file.num_row_groups}'
        )
    sample_columns = [
        'key', 'record_uid', 'source_key', 'task_name', 'task_description',
        'video_members_json', 'shard', 'target_duration_ns',
    ]
    member_columns = ['shard', 'member', 'data_offset', 'length']
    batch = []
    batch_index = 0
    total_selected = 0
    for group_index in range(samples_file.num_row_groups):
        sample_rows = samples_file.read_row_group(
            group_index, columns=sample_columns,
        ).to_pylist()
        if source_filter and not any(
            str(row['source_key']) in source_filter for row in sample_rows
        ):
            continue
        needs_members = record_uids is None or any(
            str(row['record_uid']) in record_uids
            and (not source_filter or str(row['source_key']) in source_filter)
            for row in sample_rows
        )
        member_rows = (members_file.read_row_group(
            group_index, columns=member_columns,
        ).to_pylist() if needs_members else [])
        members = {str(row['member']): row for row in member_rows}
        # Bound positive stat reuse to one catalog row group. Source opening
        # remains authoritative; missing paths are never cached or suppressed.
        shard_paths: dict[str, Path] = {}
        member_shards = {str(row['shard']) for row in member_rows}
        sample_shards = {str(row['shard']) for row in sample_rows}
        if needs_members and member_shards != sample_shards:
            raise RuntimeError(
                f'cosmos3_catalog_shard_mismatch:{group_index}:'
                f'{sample_shards}:{member_shards}'
            )
        for row in sample_rows:
            if source_filter and str(row['source_key']) not in source_filter:
                continue
            if record_uids is None or str(row['record_uid']) in record_uids:
                batch.append(cosmos3_physical_sample(
                    root, row, members, shard_paths, verify_shard=False,
                ))
            total_selected += 1
            # Keep the original catalog batch label even when most rows are
            # filtered, but do not stat/decode their media just to discard them.
            if total_selected % batch_size == 0:
                label = f'catalog-{batch_index:06d}'
                if batch:
                    yield label, str(root), 'cosmos3_v1_5', batch, len(batch)
                batch = []
                batch_index += 1
                if max_batches is not None and batch_index >= max_batches:
                    return
    if batch and (max_batches is None or batch_index < max_batches):
        label = f'catalog-{batch_index:06d}'
        yield label, str(root), 'cosmos3_v1_5', batch, len(batch)
    print(
        f'cosmos3_catalog_scan_complete selected_records={total_selected} '
        f'batches={batch_index + bool(batch)}', flush=True,
    )


def direct_video_sample(path: Path) -> tuple[dict, dict[str, tuple[str, bytes]]]:
    payload = path.read_bytes()
    digest = sha256_bytes(payload)
    uid = f'direct-video:{digest}'
    record = {
        'schema_version': 'unified-vqa-record/v2',
        'uid': uid,
        'source_key': 'direct_video',
        'split': 'unspecified',
        'dialogue': {'turns': []},
        'media': {
            'layout': 'sequence',
            'items': [{
                'media_id': 'media-0', 'type': 'video', 'mime_type': 'video/mp4',
                'member': path.name, 'relative_path': str(path.resolve()), 'sha256': digest,
            }],
        },
        'provenance': {
            'source_record_uid': uid,
            'source_locator': {'source_file': str(path.resolve()), 'storage_kind': 'video'},
        },
    }
    return record, {path.name: ('video/mp4', payload)}


def video_manifest_sample(row: dict, manifest_path: Path):
    raw_path = Path(str(row.get('video_path') or ''))
    path = raw_path if raw_path.is_absolute() else manifest_path.parent/raw_path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'video_manifest_media_missing:{path}')
    uid = str(row.get('uid') or row.get('record_uid') or '').strip()
    if not uid:
        raise ValueError('video_manifest_uid_missing')
    task_name = select_vla_task_name(
        row.get('task_description'), row.get('task_name'),
    ) or ''
    source_key = str(row.get('source_key') or 'video_manifest')
    member = path.name
    source = {
        'schema_version': 'unified-vqa-record/v2',
        'uid': uid,
        'source_key': source_key,
        'actor': str(row.get('actor') or '').strip() or None,
        'dataset_family': str(row.get('dataset_family') or '').strip() or None,
        'original_source_key': str(row.get('original_source_key') or '').strip() or None,
        'split': str(row.get('split') or 'train'),
        'task_name': task_name,
        'dialogue': {'turns': []},
        'media': {
            'layout': 'sequence',
            'items': [{
                'media_id': 'media-0', 'type': 'video', 'mime_type': 'video/mp4',
                'member': member, 'relative_path': str(path),
            }],
        },
        'provenance': {
            'source_record_uid': uid,
            'source_locator': {
                'storage_kind': 'video_file', 'source_file': str(path),
                'manifest': str(manifest_path.resolve()),
                **(
                    copy.deepcopy(row['physical_source_locator'])
                    if isinstance(row.get('physical_source_locator'), dict)
                    else {}
                ),
            },
        },
    }
    clip = row.get('clip') if isinstance(row.get('clip'), dict) else {}
    duration_hint = row.get('duration_seconds') or clip.get('duration_sec')
    if isinstance(duration_hint, (int, float)) and duration_hint > 0:
        source['_duration_seconds'] = float(duration_hint)
    return source, {member: ('video/mp4', FileSlice(path, 0, path.stat().st_size))}


def iter_video_manifest_batches(
    manifest_path: Path,
    source_filter: set[str],
    batch_size: int,
    max_batches: int | None,
):
    batch = []
    batch_index = 0
    seen_uids = set()
    with manifest_path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f'invalid_video_manifest_json:{manifest_path}:{line_number}:{error}'
                ) from error
            if source_filter and str(row.get('source_key')) not in source_filter:
                continue
            uid = str(row.get('uid') or row.get('record_uid') or '').strip()
            if uid in seen_uids:
                raise ValueError(
                    f'duplicate_video_manifest_uid:{manifest_path}:{line_number}:{uid}'
                )
            seen_uids.add(uid)
            batch.append(video_manifest_sample(row, manifest_path))
            if len(batch) >= batch_size:
                label = f'manifest-{batch_index:06d}'
                source_key = str(batch[0][0]['source_key'])
                yield label, str(manifest_path.resolve()), source_key, batch, len(batch)
                batch = []
                batch_index += 1
                if max_batches is not None and batch_index >= max_batches:
                    return
    if batch and (max_batches is None or batch_index < max_batches):
        label = f'manifest-{batch_index:06d}'
        source_key = str(batch[0][0]['source_key'])
        yield label, str(manifest_path.resolve()), source_key, batch, len(batch)


def media_payload(
    record: dict,
    blobs: dict[str, tuple[str, bytes | FileSlice]],
    preserve_video_file_slices: bool = False,
) -> list[tuple[str, bytes | FileSlice]]:
    media = []
    for item in (record.get('media') or {}).get('items') or []:
        member = item.get('member')
        if member in blobs:
            mime_type, payload = blobs[member]
            if isinstance(payload, FileSlice):
                if preserve_video_file_slices and mime_type.startswith('video/'):
                    media.append((mime_type, payload))
                    continue
                payload = payload.read()
            media.append(normalize_image(mime_type, payload))
    if not media:
        raise RuntimeError('record_has_no_readable_media')
    return media


def video_duration(payload: bytes | FileSlice) -> float | None:
    name = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            write_video_payload(stream, payload)
            name = stream.name
        return video_path_duration(name)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def video_path_duration(path: str | Path) -> float | None:
    """Read video duration without copying an already materialized source."""
    capture = open_video_capture(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    return frames / fps if fps > 0 and frames > 0 else None


def media_duration(media: list[tuple[str, bytes | FileSlice]]) -> float | None:
    for mime_type, payload in media:
        if mime_type.startswith('video/'):
            return video_duration(payload)
    return None


def fallback_frames(
    media: list[tuple[str, bytes | FileSlice]], max_frames: int = 12,
) -> list[tuple[str, bytes]]:
    result = []
    for mime_type, payload in media:
        if not mime_type.startswith('video/'):
            result.append((mime_type, payload))
            continue
        name = ''
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
                write_video_payload(stream, payload)
                name = stream.name
            capture = open_video_capture(name)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            indices = sorted(set(int(value) for value in np.linspace(0, count - 1, min(max_frames, count)))) if count else []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if encoded:
                        result.append(('image/jpeg', buffer.tobytes()))
            capture.release()
        finally:
            if name:
                Path(name).unlink(missing_ok=True)
    if not result:
        raise RuntimeError('video_fallback_produced_no_frames')
    return result


def request_with_video_fallback(
    client: ApiClient | VlaAnnoClient,
    task: str,
    model: str,
    system: str,
    prompt: str,
    media: list[tuple[str, bytes | FileSlice]],
) -> tuple[dict, str]:
    if isinstance(client, VlaAnnoClient):
        # The fixed operator accepts one video and maps its own frame ranges.
        # Converting to still images would change the output contract from
        # seconds to media indices and cannot validate against the source video.
        return client.request_json(task, model, system, prompt, media)
    try:
        return client.request_json(task, model, system, prompt, media)
    except RuntimeError as error:
        message = str(error).lower()
        if not any(mime.startswith('video/') for mime, _ in media) or not any(
            token in message for token in (
                'video file is too short', 'provided url does not appear to be valid',
                'max bytes per data-uri item', 'data_inspection_failed',
                'invalid video file',
            )
        ):
            raise
        return client.request_json(
            task, model, system,
            prompt + '\nThe clip was decoded into uniformly ordered frames; media order is chronological.',
            fallback_frames(media),
        )


def validate_subtasks(
    result: dict, media: list[tuple[str, bytes | FileSlice]],
) -> dict:
    value = copy.deepcopy(result)
    if 'subtasks' not in value and isinstance(value.get('steps'), list):
        value['subtasks'] = value.pop('steps')
    steps = value.get('subtasks')
    if not isinstance(steps, list) or not steps:
        raise ValueError('subtask_result_requires_nonempty_subtasks')
    has_video = any(mime.startswith('video/') for mime, _ in media)
    previous_end: float | int | None = None
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or not str(step.get('subtask', '')).strip():
            raise ValueError(f'subtask_invalid:{index}')
        step['id'] = step.get('id', index)
        step['subtask'] = str(step['subtask']).strip().lower()
        for field in ('action', 'object', 'source', 'target'):
            step[field] = str(step.get(field, 'None')).strip() or 'None'
            if step[field] != 'None':
                step[field] = step[field].lower()
        if 'start' in step and 'start_time_seconds' not in step:
            step['start_time_seconds'] = step.pop('start')
        if 'end' in step and 'end_time_seconds' not in step:
            step['end_time_seconds'] = step.pop('end')
        start_field, end_field = (
            ('start_time_seconds', 'end_time_seconds') if has_video
            else ('media_start_index', 'media_end_index')
        )
        start, end = step.get(start_field), step.get(end_field)
        bounds_invalid = (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or start < 0
            or (end <= start if has_video else end < start)
        )
        if bounds_invalid:
            raise ValueError(f'subtask_invalid_bounds:{index}:{start_field}={start}:{end_field}={end}')
        if previous_end is not None and abs(float(start) - float(previous_end)) > 1e-4:
            raise ValueError(f'subtask_noncontiguous:{index}:{start}:{previous_end}')
        previous_end = end
    merged_steps = []
    signature_fields = ('subtask', 'action', 'object', 'source', 'target')
    for step in steps:
        same_as_previous = bool(merged_steps) and all(
            ' '.join(str(merged_steps[-1].get(field) or '').split()).casefold()
            == ' '.join(str(step.get(field) or '').split()).casefold()
            for field in signature_fields
        )
        if same_as_previous:
            merged_steps[-1][end_field] = step[end_field]
        else:
            merged_steps.append(step)
    if len(merged_steps) != len(steps):
        for index, step in enumerate(merged_steps, 1):
            step['id'] = index
        value['subtasks'] = steps = merged_steps
    first = steps[0]
    first_start = first['start_time_seconds' if has_video else 'media_start_index']
    if abs(float(first_start)) > 1e-4:
        raise ValueError(f'subtask_first_start_not_zero:{first_start}')
    if has_video:
        provider_output = value.get('provider_output') or {}
        boundary_mapping = provider_output.get('boundary_mapping') or {}
        duration = (
            provider_output.get('source_duration_seconds')
            or boundary_mapping.get('source_duration_seconds')
        )
        if not isinstance(duration, (int, float)) or duration <= 0:
            duration = media_duration(media)
        final_end = float(steps[-1]['end_time_seconds'])
        if duration is not None and abs(final_end - duration) > max(
            0.05, 1.0 / GRD_FRAME_FPS,
        ):
            raise ValueError(f'subtask_final_end_mismatch:{final_end}:{duration}')
    else:
        final_end = int(steps[-1]['media_end_index'])
        if final_end != len(media) - 1:
            raise ValueError(f'subtask_final_end_mismatch:{final_end}:{len(media) - 1}')
    return value


def las_task_instruction(record: dict, override: str | None) -> str:
    """Return the task instruction passed unchanged into the LAS pipeline."""
    value = str(override or task_context(record) or '').strip()
    if value.lower().startswith('task instruction:'):
        value = value.split(':', 1)[1].strip()
    return value or 'annotate the visible manipulation task'


def las_embodiment(record: dict, configured: str) -> str:
    """Resolve LAS human/robot prompt semantics without changing output fields."""
    if configured in {'human', 'robot'}:
        return configured
    candidates = ' '.join(
        str(record.get(key) or '')
        for key in ('actor', 'embodiment', 'dataset_family', 'source_key')
    ).lower()
    if any(token in candidates for token in ('human', 'egodex', 'ego4d')):
        return 'human'
    return 'robot'


def ecot_system_prompt(source: dict, configured_embodiment: str) -> str:
    """Use the source ECoT prompt with the matching human/robot embodiment."""
    return (
        HUMAN_ECOT_SYSTEM_PROMPT
        if las_embodiment(source, configured_embodiment) == 'human'
        else ECOT_SYSTEM_PROMPT
    )


def ecot_user_prompt(task_instruction: str) -> str:
    return f'''Global task: {task_instruction}

The first media item is the COMPLETE 0.5 FPS EPISODE VIDEO. The second is the TARGET FRAME IMAGE. Label that observation using the four-field JSON contract: scene_description, task_progress, current_subtask, and atomic_action. Keep current_subtask goal-level and atomic_action immediate and fine-grained. The complete video is teacher-only context and is not part of the learner's prediction input.'''


def ecot_training_question(task_instruction: str) -> str:
    return f'''Global task: {task_instruction}

For this current observation, predict a concise scene description, task progress, current goal-level subtask, and the immediate atomic action. Return JSON with scene_description, task_progress, current_subtask, and atomic_action; current_subtask must contain subtask, action, object, source, and target. atomic_action is one lowercase English base-form verb plus object phrase, finer-grained than the subtask, describing the immediate next action or continuation of the action underway. Include a destination only when part of that one action. Use "None" when no further task-related action occurs or the immediate action cannot be determined.'''


def ecot_schema() -> dict:
    return {'name': 'privileged_video_ecot_atomic_v2', 'strict': True, 'schema': {
        'type': 'object', 'additionalProperties': False,
        'required': ['scene_description', 'task_progress', 'current_subtask', 'atomic_action'],
        'properties': {
            'scene_description': {'type': 'string', 'minLength': 1},
            'task_progress': {'type': 'string', 'minLength': 1},
            'atomic_action': {'type': 'string', 'minLength': 1, 'maxLength': 240},
            'current_subtask': {
                'type': 'object', 'additionalProperties': False,
                'required': list(ECOT_SUBTASK_FIELDS),
                'properties': {
                    'subtask': {'type': 'string', 'minLength': 1},
                    'action': {'type': 'string', 'minLength': 1, 'maxLength': 80},
                    'object': {'type': 'string', 'minLength': 1, 'maxLength': 120},
                    'source': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                    'target': {'type': 'string', 'minLength': 1, 'maxLength': 160},
                },
            },
        },
    }}


def concise_ecot_sentence(value: object, field: str, max_words: int | None = None) -> str:
    """Normalize text; legacy name and max_words argument impose no length limit."""
    text = re.sub(r'\s+', ' ', str(value or '').strip())
    if text and not re.search(r'[.!?]', text):
        text += '.'
    if field == 'task_progress':
        text = re.sub(r'\.\s+(Task (?:not yet )?complete\.)$', r'; \1', text)
        # Remove duplicated punctuation before the required completion suffix.
        # This does not merge or discard any substantive sentences.
        text = re.sub(r'\.\s*;\s*(Task (?:not yet )?complete\.)$', r'; \1', text)
    if not text:
        raise ValueError(f'ecot_empty_text:{field}')
    return text


def parse_ecot_subtask(value: object, progress: str) -> dict:
    raw = value if isinstance(value, dict) else {}
    if set(raw) != set(ECOT_SUBTASK_FIELDS):
        raise ValueError(f'ecot_subtask_fields_invalid:{sorted(raw)}')
    text = re.sub(r'\s+', ' ', str(raw.get('subtask') or '').strip())
    if text and not re.search(r'[.!?]$', text):
        text += '.'
    text = concise_ecot_sentence(text, 'current_subtask')
    # Case is representational, not a visual judgment. Canonicalize acronyms,
    # brands and labels (for example USB/Vans/B-labeled) locally instead of
    # spending another teacher request on the same image and video.
    text = text.lower()
    parsed = {'subtask': text}
    for field in ECOT_SUBTASK_FIELDS[1:]:
        item = re.sub(r'\s+', ' ', str(raw.get(field) or '').strip())
        if not item:
            raise ValueError(f'ecot_subtask_semantic_invalid:{field}:{item}')
        if item != ECOT_MISSING_SEMANTIC:
            item = item.lower()
        parsed[field] = item
    complete = progress.endswith('Task complete.')
    empty = (
        text == 'no further subtask remains.'
        and all(parsed[field] == ECOT_MISSING_SEMANTIC for field in ECOT_SUBTASK_FIELDS[1:])
    )
    if complete != empty:
        raise ValueError('ecot_subtask_completion_mismatch')
    if re.search(r'\bthen\b|\bafter that\b|;', text, flags=re.I):
        raise ValueError(f'ecot_subtask_not_single_goal:{text}')
    return parsed


def parse_ecot_result(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        'scene_description', 'task_progress', 'current_subtask', 'atomic_action',
    }:
        raise ValueError('ecot_top_level_fields_invalid')
    scene = concise_ecot_sentence(value['scene_description'], 'scene_description')
    from ecot_completion_normalization import normalize_completion_suffix
    progress_text = normalize_completion_suffix(str(value['task_progress'] or ''))
    progress = concise_ecot_sentence(progress_text, 'task_progress')
    if not progress.endswith(('Task complete.', 'Task not yet complete.')):
        raise ValueError(f'ecot_completion_judgment_missing:{progress}')
    current_subtask = parse_ecot_subtask(value['current_subtask'], progress)
    from ecot_contract import validate_atomic_action
    atomic_value = value['atomic_action']
    if isinstance(atomic_value, str) and atomic_value != ECOT_MISSING_SEMANTIC:
        atomic_value = atomic_value.lower()
    atomic_action = validate_atomic_action(atomic_value)
    if ECOT_PRIVILEGED_LEAK_PATTERN.search(
        f'{scene} {progress} {current_subtask["subtask"]} {atomic_action}'
    ):
        raise ValueError('ecot_privileged_context_leak')
    return {
        'scene_description': scene,
        'task_progress': progress,
        'current_subtask': current_subtask,
        'atomic_action': atomic_action,
    }


def request_validated_subtasks(
    annotator: LASSubtaskAnnotator,
    source: dict,
    media: list[tuple[str, bytes | FileSlice]],
    task_instruction: str | None,
    embodiment: str = 'auto',
    checkpoint: AnnotationUnitCheckpoint | None = None,
    media_limiter: threading.BoundedSemaphore | None = None,
) -> tuple[dict, str, str]:
    """Run only the vendored LAS implementation, then validate the old interface."""
    cached = checkpoint.get('validated_result') if checkpoint is not None else None
    if isinstance(cached, dict):
        return (
            validate_subtasks(copy.deepcopy(cached['result']), media),
            str(cached['raw']), str(cached['prompt']),
        )
    videos = [item for item in media if item[0].startswith('video/')]
    source_has_video = bool(videos)
    las_media = [videos[0]] if videos else media
    media_context = media_limiter or contextlib.nullcontext()
    prepared = None
    media_path = None
    try:
        with media_context:
            # A queued LAS-only job may acquire this gate after a shared stop.
            # Check before copying/uploading media, not only at the API call.
            check_available = getattr(annotator, 'check_available', None)
            if check_available is not None:
                check_available()
            media_path = write_vla_media_file(las_media)
            if getattr(annotator, 'release_published_media', False) is True:
                from las_subtask import video_timing
                video_url, publication = annotator.publisher.resolve(
                    Path(media_path), str(source.get('uid') or sha256_json(source)),
                )
                prepared = (video_url, publication, video_timing(Path(media_path)))
                # Only this function's generated copy is removed, never source media.
                Path(media_path).unlink()
        try:
            result, raw, prompt = annotator.annotate(
                Path(media_path),
                source_uid=str(source.get('uid') or sha256_json(source)),
                task_instruction=las_task_instruction(source, task_instruction),
                embodiment=las_embodiment(source, embodiment),
                source_has_video=source_has_video,
                source_media_count=len(las_media),
                **({'prepared': prepared} if prepared is not None else {}),
            )
        except RuntimeError as error:
            message = str(error).lower()
            if any(token in message for token in (
                'billingautherror', 'in arrears', 'arrearage',
                'invalid api key', 'unauthorized', 'authentication failed',
                'permission denied', 'insufficient_quota',
                'http 401', 'http 403',
            )):
                raise FatalProviderError(str(error)) from error
            raise
        validated = validate_subtasks(result, media)
        if checkpoint is not None:
            checkpoint.put(
                'validated_result', {
                    'result': validated, 'raw': raw, 'prompt': prompt,
                    'backend': 'vendored_las',
                    'adapter_version': LAS_ADAPTER_VERSION,
                },
            )
        return validated, raw, prompt
    finally:
        if media_path is not None:
            Path(media_path).unlink(missing_ok=True)


def clip_video(
    payload: bytes | FileSlice,
    start: float,
    end: float,
    fps: float | None = None,
    minimum_duration: float | None = None,
) -> bytes:
    if start < 0 or end <= start:
        raise ValueError(f'invalid_clip_bounds:{start}:{end}')
    temporary = ''
    try:
        source, temporary = decode_video_path(payload)
        return clip_video_path(source, start, end, fps, minimum_duration)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)


def clip_video_path(
    source: str | Path,
    start: float,
    end: float,
    fps: float | None = None,
    minimum_duration: float | None = None,
) -> bytes:
    """Clip an existing video path without rematerializing its full payload."""
    # HTTP admission does not cover video encoding that precedes a request.
    with VIDEO_CLIP_LIMITER:
        if VIDEO_CLIP_POOL is not None:
            return VIDEO_CLIP_POOL.clip(source, start, end, fps, minimum_duration)
        return _clip_video_path_admitted(source, start, end, fps, minimum_duration)


@contextlib.contextmanager
def video_clip_execution(workers, process_pool=False):
    """Isolate repeated ffmpeg launches without changing media or admission."""
    global VIDEO_CLIP_POOL
    if not process_pool:
        yield
        return
    from media_clip_pool import MediaClipPool
    with MediaClipPool(workers) as pool:
        VIDEO_CLIP_POOL = pool
        try:
            yield
        finally:
            VIDEO_CLIP_POOL = None


def prewarm_thread_pool(executor, workers, label='shared-frame'):
    """Create request workers before episode submitters contend on submit().

    ``ThreadPoolExecutor`` starts one worker while holding both its shutdown
    lock and the interpreter-wide executor shutdown lock. Thousands of episode
    threads submitting their first frame concurrently otherwise form a lock
    convoy while the pool is still cold. Blocking warm-up jobs force worker
    creation from one caller before paid work is admitted.
    """
    if executor is None or workers <= 0:
        return
    maximum = int(getattr(executor, '_max_workers', 0))
    if workers > maximum:
        raise ValueError(f'prewarm_workers_exceed_executor:{workers}:{maximum}')
    release = threading.Event()
    ready = threading.Semaphore(0)

    def hold():
        ready.release()
        release.wait()

    started = time.monotonic()
    futures = []
    try:
        for _ in range(workers):
            futures.append(executor.submit(hold))
        for _ in range(workers):
            if not ready.acquire(timeout=300):
                raise TimeoutError(f'{label}_prewarm_timeout:{workers}')
    finally:
        release.set()
        if futures:
            concurrent.futures.wait(futures)
    print(json_dumps({
        'event': 'thread_pool_prewarmed', 'pool': label,
        'workers': workers,
        'elapsed_seconds': round(time.monotonic() - started, 3),
    }), flush=True)


def _clip_video_path_admitted(source, start, end, fps=None, minimum_duration=None, codec_threads=None):
    if start < 0 or end <= start:
        raise ValueError(f'invalid_clip_bounds:{start}:{end}')
    codec_threads = (int(os.environ.get('VQA_VIDEO_CLIP_CODEC_THREADS', '1'))
                     if codec_threads is None else codec_threads)
    if not 1 <= codec_threads <= 8:
        raise ValueError('video_clip_codec_threads_must_be_between_1_and_8')
    destination = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            destination = stream.name
        source_duration = end - start
        output_duration = max(source_duration, float(minimum_duration or 0))
        if fps is not None and output_duration > source_duration:
            source_fps, source_frame_count, _ = video_path_timing(source)
            last_frame_time = max(0.0, (source_frame_count - 1) / source_fps)
            if start > last_frame_time + 1e-6:
                # Some VFR containers report a duration slightly beyond the
                # final decoded frame.  FFmpeg's trim+tpad path can segfault
                # when seeking into that frame-less tail.  The future view at
                # this point is semantically the final frame held for the
                # requested encoded duration.
                return repeated_video_frame_at_path(
                    source, start, fps, output_duration,
                )
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-threads', str(codec_threads), '-filter_threads', '1',
            '-ss', f'{start:.6f}', '-i', str(source),
        ]
        if fps is not None:
            filters = [
                f'trim=duration={source_duration:.6f}', 'setpts=PTS-STARTPTS',
            ]
            if output_duration > source_duration:
                filters.append(
                    f'tpad=stop_mode=clone:stop_duration={output_duration - source_duration:.6f}'
                )
            filters.append(f'fps={fps:g}')
            command.extend(('-vf', ','.join(filters)))
        command.extend((
            '-t', f'{output_duration:.6f}', '-an',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-threads', str(codec_threads),
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', destination,
        ))
        completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if completed.returncode:
            raise RuntimeError(
                f'ffmpeg_subtask_clip_failed:returncode={completed.returncode}:'
                f'{completed.stderr[-1000:]}'
            )
        value = Path(destination).read_bytes()
        if not value:
            raise RuntimeError('ffmpeg_subtask_clip_empty')
        return value
    finally:
        if destination:
            Path(destination).unlink(missing_ok=True)


def repeated_video_frame_at_path(
    source: str | Path,
    timestamp_seconds: float,
    fps: float,
    duration_seconds: float,
) -> bytes:
    """Encode one decoded source frame as a fixed-duration video."""
    frame_count = max(1, int(round(fps * duration_seconds)))
    mime_type, payload = video_frame_at_path(source, timestamp_seconds)
    if mime_type != 'image/jpeg' or not payload:
        raise RuntimeError('grounding_tail_frame_decode_failed')
    with tempfile.TemporaryDirectory(prefix='vqa-tail-frame-') as directory:
        root = Path(directory)
        frame_path = root/'frame.jpg'
        frame_path.write_bytes(payload)
        destination = root/'future.mp4'
        completed = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-loop', '1', '-framerate', f'{fps:g}', '-i', str(frame_path),
            '-frames:v', str(frame_count), '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', '-threads', '1',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(destination),
        ], capture_output=True, text=True, timeout=600)
        if completed.returncode:
            raise RuntimeError(
                'ffmpeg_grounding_tail_video_failed:'
                f'returncode={completed.returncode}:{completed.stderr[-1000:]}'
            )
        value = destination.read_bytes()
    if not value:
        raise RuntimeError('grounding_tail_video_empty')
    return value


def video_frame_at(
    payload: bytes | FileSlice,
    timestamp_seconds: float,
) -> tuple[str, bytes]:
    """Decode one source-video frame at the requested timestamp as JPEG."""
    name = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            write_video_payload(stream, payload)
            name = stream.name
        return video_frame_at_path(name, timestamp_seconds)
    finally:
        if name:
            Path(name).unlink(missing_ok=True)


def video_frame_at_path(
    path: str | Path,
    timestamp_seconds: float,
) -> tuple[str, bytes]:
    """Decode one frame from an existing source path."""
    capture = open_video_capture(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise RuntimeError('grounding_current_frame_video_metadata_invalid')
        frame_index = min(
            count - 1,
            max(0, int(round(float(timestamp_seconds) * fps))),
        )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f'grounding_current_frame_decode_failed:{timestamp_seconds}'
            )
        encoded_ok, encoded = cv2.imencode(
            '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not encoded_ok:
            raise RuntimeError('grounding_current_frame_encode_failed')
        return 'image/jpeg', encoded.tobytes()
    finally:
        capture.release()


def ordered_future_video(
    media: list[tuple[str, bytes]],
    anchor_media_index: int,
) -> bytes:
    """Encode exactly four seconds of ordered images as a 3 FPS MP4."""
    frame_count = int(round(GRD_FUTURE_FPS * GRD_FUTURE_SECONDS))
    selected = media[anchor_media_index:anchor_media_index + frame_count]
    frames = []
    for mime_type, payload in selected:
        if not mime_type.startswith('image/'):
            continue
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f'grounding_future_video_has_no_images:{anchor_media_index}')
    height, width = frames[0].shape[:2]
    frames = [
        frame if frame.shape[:2] == (height, width) else cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_AREA,
        )
        for frame in frames
    ]
    frames.extend([frames[-1]] * (frame_count - len(frames)))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, frame in enumerate(frames):
            if not cv2.imwrite(str(root/f'{index:06d}.jpg'), frame):
                raise RuntimeError('grounding_future_frame_encode_failed')
        destination = root/'future.mp4'
        completed = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-framerate', f'{GRD_FUTURE_FPS:g}', '-i', str(root/'%06d.jpg'),
            '-frames:v', str(frame_count), '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', '-threads', '1',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(destination),
        ], capture_output=True, text=True, timeout=600)
        if completed.returncode:
            raise RuntimeError(
                f'ffmpeg_grounding_future_video_failed:{completed.stderr[-1000:]}'
            )
        value = destination.read_bytes()
    if not value:
        raise RuntimeError('grounding_future_video_empty')
    return value


class EcotMediaFactory:
    """Cache a 2 FPS target grid and a separate 0.5 FPS teacher video."""

    def __init__(self, media: list[tuple[str, bytes | FileSlice]], codec_threads: int = 1,
                 frame_cache_write_through: bool = True, frame_cache_memory_mib: int = 64,
                 teacher_only: bool = False):
        if not 1 <= codec_threads <= 8:
            raise ValueError('video_prepare_codec_threads_must_be_between_1_and_8')
        self.codec_threads = codec_threads
        self.frame_cache_write_through = frame_cache_write_through
        if not 1 <= frame_cache_memory_mib <= 512:
            raise ValueError('frame_cache_memory_mib_must_be_between_1_and_512')
        self.frame_cache_memory_mib = frame_cache_memory_mib
        self.teacher_only = teacher_only
        self.media = media
        self.temporary: tempfile.TemporaryDirectory | None = None
        self.root: Path | None = None
        self.source_path: Path | None = None
        self.video_path: Path | None = None
        self.teacher_video_path: Path | None = None
        self._video_payload = None
        self._video_transport_path = None
        self.sampled_frame_count = 0
        self.source_fps: float | None = None
        self.source_frame_count: int | None = None
        self.target_images: dict[int, tuple[str, bytes]] = {}
        self.target_image_decode_passes = 0

    def __enter__(self):
        try:
            return self._enter_media()
        except BaseException:
            self.__exit__()
            raise

    def _enter_media(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='vqa-ecot-')
        self.root = Path(self.temporary.name)
        self.target_images = FrameCache(self.root/'frame_cache',
                                       budget=self.frame_cache_memory_mib * 1024**2,
                                       write_through=self.frame_cache_write_through)
        video_item = next(
            ((mime, payload) for mime, payload in self.media if mime.startswith('video/')),
            None,
        )
        self.video_path = self.root/'episode_2fps.mp4'
        self.teacher_video_path = self.root/'episode_teacher_0.5fps.mp4'
        if video_item is not None:
            self.source_path = self.root/'source.mp4'
            with self.source_path.open('wb') as stream:
                write_video_payload(stream, video_item[1])
            self.source_fps, self.source_frame_count, _ = video_path_timing(
                self.source_path,
            )
            if self.teacher_only:
                # LAS targets frames by timestamp in the 0.5 FPS teacher. A
                # separate 2 FPS MP4 is never decoded or sent, so generating it
                # and then transcoding it a second time only burns CPU and I/O.
                teacher = subprocess.run([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-threads', str(self.codec_threads), '-i', str(self.source_path),
                    '-an', '-vf', f'fps={ECOT_TEACHER_FPS:g}',
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-threads', str(self.codec_threads), '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(self.teacher_video_path),
                ], capture_output=True, text=True, timeout=1800)
                if teacher.returncode or not self.teacher_video_path.is_file():
                    raise RuntimeError(
                        f'ffmpeg_ecot_teacher_video_failed:{teacher.stderr[-2000:]}'
                    )
                teacher_fps, teacher_frames, _ = video_path_timing(self.teacher_video_path)
                if abs(teacher_fps - ECOT_TEACHER_FPS) > 0.01 or teacher_frames < 1:
                    raise RuntimeError(
                        f'ecot_teacher_fps_contract_failed:{teacher_fps}:{teacher_frames}'
                    )
                ratio = int(round(ECOT_FPS / ECOT_TEACHER_FPS))
                self.sampled_frame_count = teacher_frames * ratio
                self.video_path = None
                return self
            command = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-threads', str(self.codec_threads), '-i', str(self.source_path), '-an', '-vf', f'fps={ECOT_FPS:g}',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                '-threads', str(self.codec_threads),
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                str(self.video_path),
            ]
        else:
            frame_dir = self.root/'ordered_frames'
            frame_dir.mkdir()
            frame_count = 0
            for mime_type, payload in self.media:
                if not mime_type.startswith('image/'):
                    continue
                if isinstance(payload, FileSlice):
                    payload = payload.read()
                image = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR,
                )
                if image is None or not cv2.imwrite(
                    str(frame_dir/f'{frame_count:06d}.jpg'), image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                ):
                    raise RuntimeError(f'ecot_ordered_frame_decode_failed:{frame_count}')
                frame_count += 1
            if frame_count < 1:
                raise RuntimeError('ecot_requires_video_or_ordered_images')
            command = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-framerate', f'{ECOT_FPS:g}', '-i', str(frame_dir/'%06d.jpg'),
                '-frames:v', str(frame_count), '-an', '-c:v', 'libx264',
                '-preset', 'veryfast', '-crf', '20', '-threads', '1',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(self.video_path),
            ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=1800,
        )
        if completed.returncode or not self.video_path.is_file():
            raise RuntimeError(
                f'ffmpeg_ecot_2fps_video_failed:{completed.stderr[-2000:]}'
            )
        sampled_fps, self.sampled_frame_count, _ = video_path_timing(self.video_path)
        if abs(sampled_fps - ECOT_FPS) > 0.01 or self.sampled_frame_count < 1:
            raise RuntimeError(
                f'ecot_2fps_contract_failed:{sampled_fps}:{self.sampled_frame_count}'
            )
        teacher = subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-threads', str(self.codec_threads), '-i', str(self.video_path),
            '-an', '-vf', f'fps={ECOT_TEACHER_FPS:g}',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-threads', str(self.codec_threads), '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.teacher_video_path),
        ], capture_output=True, text=True, timeout=1800)
        if teacher.returncode or not self.teacher_video_path.is_file():
            raise RuntimeError(
                f'ffmpeg_ecot_teacher_video_failed:{teacher.stderr[-2000:]}'
            )
        teacher_fps, teacher_frames, _ = video_path_timing(self.teacher_video_path)
        if abs(teacher_fps - ECOT_TEACHER_FPS) > 0.01 or teacher_frames < 1:
            raise RuntimeError(
                f'ecot_teacher_fps_contract_failed:{teacher_fps}:{teacher_frames}'
            )
        return self

    @property
    def video_transport_path(self):
        if self._video_transport_path is None:
            if self.teacher_video_path is None or self.root is None:
                raise RuntimeError('ecot_media_factory_not_entered')
            self._video_transport_path = fit_video_file(
                self.teacher_video_path, self.root/'teacher_transport.mp4',
            )
        return self._video_transport_path

    @property
    def video_payload(self):
        if self._video_payload is None:
            self._video_payload = self.video_transport_path.read_bytes()
        return self._video_payload

    def __exit__(self, *_exc) -> None:
        self._video_payload = None
        self._video_transport_path = None
        self.target_images.clear()
        self.source_path = None
        self.video_path = None
        self.teacher_video_path = None
        if self.temporary is not None:
            self.temporary.cleanup()
            self.temporary = None

    def selected_target_images(
        self, sampled_frame_indices: list[int],
    ) -> dict[int, tuple[str, bytes]]:
        """Return selected 2 FPS frames, decoding every requested frame at most once."""
        if self.video_path is None:
            raise RuntimeError('ecot_media_factory_not_entered')
        wanted = set(sampled_frame_indices)
        if not wanted:
            return {}
        invalid = {
            index for index in wanted
            if index < 0 or index >= self.sampled_frame_count
        }
        if invalid:
            raise ValueError(f'ecot_target_frame_indices_invalid:{sorted(invalid)}')
        missing = wanted - set(self.target_images)
        if missing:
            self.target_image_decode_passes += 1
            capture = open_video_capture(self.video_path)
            try:
                for index in range(max(missing) + 1):
                    if not self.frame_cache_write_through and index not in missing:
                        # Advance the same decoder without materializing unused
                        # BGR frames; selected frames still use the exact read path.
                        if not capture.grab():
                            raise RuntimeError(f'ecot_target_frame_decode_failed:{index}')
                        continue
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise RuntimeError(f'ecot_target_frame_decode_failed:{index}')
                    if index not in missing:
                        continue
                    encoded_ok, encoded = cv2.imencode(
                        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92],
                    )
                    if not encoded_ok:
                        raise RuntimeError(f'ecot_target_frame_encode_failed:{index}')
                    self.target_images[index] = (
                        'image/jpeg', encoded.tobytes(),
                    )
            finally:
                capture.release()
        if not wanted <= set(self.target_images):
            raise RuntimeError(
                'ecot_target_frame_coverage_failed:'
                f'{sorted(wanted - set(self.target_images))}'
            )
        return FrameSelection(self.target_images, wanted)

    def source_frame_index(self, sampled_frame_index: int) -> int | None:
        if self.source_fps is None or self.source_frame_count is None:
            return None
        return min(
            self.source_frame_count - 1,
            int(round(sampled_frame_index / ECOT_FPS * self.source_fps)),
        )


ECOT_VALIDATION_MAX_RETRIES = 1


class EcotValidationRetriesExhausted(ValueError):
    """Do not multiply a spent frame validation budget through record retries."""


def ecot_frame_record(structured, task_instruction, training_question,
                      sampled_frame_index, source_frame_index):
    """Build the unchanged learner record independently of teacher transport."""
    image_binding = {
        'kind': 'source_video_frame',
        'sampled_frame_index': sampled_frame_index,
        'sampled_time_seconds': round(sampled_frame_index / ECOT_FPS, 6),
        'sampling_fps': ECOT_FPS,
    }
    if source_frame_index is not None:
        image_binding['source_frame_index'] = source_frame_index
    return {
        **image_binding,
        'structured_ecot': structured,
        'scene_description': structured['scene_description'],
        'task_progress': structured['task_progress'],
        'current_subtask': structured['current_subtask'],
        'atomic_action': structured['atomic_action'],
        'training_input': {
            'task_instruction': task_instruction,
            'image': copy.deepcopy(image_binding),
        },
        'training_target': structured,
        'messages': [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': copy.deepcopy(image_binding)},
                {'type': 'text', 'text': training_question},
            ]},
            {'role': 'assistant', 'content': json_dumps(structured)},
        ],
    }


def annotate_ecot_frame(
    client: ApiClient,
    model: str,
    system_prompt: str,
    user_prompt: str,
    training_question: str,
    task_instruction: str,
    teacher_video: bytes | TemporaryVideo,
    target_image: tuple[str, bytes] | None,
    sampled_frame_index: int,
    source_frame_index: int | None,
) -> tuple[dict, dict]:
    sampled_time = round(sampled_frame_index / ECOT_FPS, 6)
    teacher_frame_index = int(round(sampled_time * ECOT_TEACHER_FPS))
    timestamp_target = (
        getattr(client, 'api', None) == 'las'
        and callable(getattr(teacher_video, 'resolve', None))
    )
    if timestamp_target:
        user_prompt = (
            user_prompt
            + '\n\nFor this request, the TARGET OBSERVATION is the frame at '
            + f'{sampled_time:g} seconds (0-based frame {teacher_frame_index} on the '
              f'{ECOT_TEACHER_FPS:g} FPS teacher grid) in the complete episode video. '
              'Judge and label that exact frame. Do not substitute the final frame or '
              'summarize the whole video.'
        )
        request_media = [('video/mp4', teacher_video)]
    else:
        if target_image is None:
            raise RuntimeError('ecot_target_image_missing_for_non_las_transport')
        request_media = [('video/mp4', teacher_video), target_image]
    base_prompt = user_prompt
    validation_attempts = []
    max_attempts = 1 + ECOT_VALIDATION_MAX_RETRIES
    for validation_attempt in range(1, max_attempts + 1):
        raw_result, raw_response = client.request_json(
            'ecot', model, system_prompt, user_prompt,
            request_media, ecot_schema(),
        )
        try:
            structured = parse_ecot_result(raw_result)
            break
        except ValueError as error:
            validation_attempts.append({
                'attempt': validation_attempt, 'error': str(error),
                'prompt': user_prompt, 'raw_response': raw_response,
            })
            if validation_attempt == max_attempts:
                raise EcotValidationRetriesExhausted(
                    f'ecot_validation_retries_exhausted:frame={sampled_frame_index}:'
                    f'attempts={max_attempts}:{error}'
                ) from error
            print(f'ecot_frame_validation_retry frame={sampled_frame_index} '
                  f'attempt={validation_attempt} error={error}', flush=True)
            user_prompt = (
                base_prompt + '\n\nThe previous response failed validation: ' + str(error)
                + '\nPrevious response: ' + json_dumps(raw_result)
                + '\nRe-examine the same target image and video and return a corrected '
                'concise response following the original schema and semantic requirements. '
                'There is no hard word limit. '
                'Task completion and current_subtask must agree: only when the task '
                'is complete use "no further subtask remains." and "None" for all '
                'four semantic fields. Otherwise give the actual remaining subtask '
                'and end task_progress with "Task not yet complete." '
                'Do not change your visual judgment merely to pass validation.'
            )
    frame = ecot_frame_record(structured, task_instruction, training_question,
                              sampled_frame_index, source_frame_index)
    request = {
        'stage': 'privileged_complete_video_ecot',
        'sampled_frame_index': sampled_frame_index,
        'sampled_time_seconds': sampled_time,
        'model': model,
        'task_instruction': task_instruction,
        'task_instruction_source': 'global_task',
        'media_kind': ECOT_MEDIA_KIND,
        'system_prompt': system_prompt,
        'prompt': user_prompt,
        'raw_response': raw_response,
        'target_locator': {
            'method': ('teacher_video_timestamp' if timestamp_target else 'separate_target_image'),
            'sampled_time_seconds': sampled_time,
            'teacher_frame_index': teacher_frame_index,
            'teacher_sampling_fps': ECOT_TEACHER_FPS,
        },
    }
    provider = getattr(client, 'api', None)
    if isinstance(provider, str):
        request['provider'] = provider
    if isinstance(teacher_video, (TemporaryVideo, ArkFileReference)):
        request['video_transport'] = {
            'method': getattr(
                teacher_video, 'transport_method',
                'ark-files' if isinstance(teacher_video, ArkFileReference) else 'dashscope-temporary',
            ),
            'sha256': teacher_video.sha256,
            'size_bytes': teacher_video.size_bytes, 'model': teacher_video.model,
        }
    if target_image is not None and isinstance(target_image[1], (TemporaryVideo, ArkFileReference)):
        reference = target_image[1]
        request['image_transport'] = {
            'method': getattr(reference, 'transport_method',
                'ark-files' if isinstance(reference, ArkFileReference) else 'dashscope-temporary'),
            'sha256': reference.sha256,
            'size_bytes': reference.size_bytes, 'model': reference.model,
        }
    if validation_attempts:
        request['base_prompt'] = base_prompt
        request['validation_attempts'] = validation_attempts
    return frame, request


def annotate_ecot_source(
    source: dict,
    media: list[tuple[str, bytes | FileSlice]],
    client: ApiClient,
    model: str,
    task_instruction_override: str | None,
    configured_embodiment: str,
    interval: int,
    workers: int,
    checkpoint: AnnotationUnitCheckpoint | None,
    media_factory: EcotMediaFactory | None = None,
) -> tuple[dict, list[dict]]:
    task_instruction = las_task_instruction(source, task_instruction_override)
    system_prompt = ecot_system_prompt(source, configured_embodiment)
    user_prompt = ecot_user_prompt(task_instruction)
    training_question = ecot_training_question(task_instruction)
    factory_context = (
        contextlib.nullcontext(media_factory)
        if media_factory is not None else EcotMediaFactory(media)
    )
    with factory_context as factory, frame_executor(client, workers, 'ecot') as executor:
        selected_indices = list(range(0, factory.sampled_frame_count, interval))
        prior_units = {}
        prior_sources = (getattr(checkpoint, 'ecot_prior_sources', None)
                         if checkpoint is not None else None)
        if isinstance(prior_sources, (list, tuple)) and prior_sources:
            from ecot_checkpoint_reuse import read_prior_units
            prior_units = read_prior_units(
                checkpoint, prior_sources, selected_indices,
                system_prompt, user_prompt, training_question, factory.source_frame_index,
                sys.modules[__name__],
            )
        completed: dict[int, tuple[dict, dict]] = {}
        missing_indices = []
        for sampled_index in selected_indices:
            key = f'ecot_frame:{sampled_index}'
            cached = checkpoint.get(key) if checkpoint is not None else None
            if cached is None:
                cached = prior_units.get(key)
            if isinstance(cached, dict):
                completed[sampled_index] = (cached['frame'], cached['request'])
            else:
                missing_indices.append(sampled_index)
        # Initialize the lazy teacher payload once before concurrent access.
        publisher = getattr(client, 'ecot_video_publisher', None)
        timestamp_target = getattr(client, 'api', None) == 'las' and publisher is not None
        # LAS can identify every selected target directly on the complete 0.5
        # FPS teacher grid. Avoid decoding and uploading one redundant JPEG per
        # request, and avoid the operator composing a new full video per frame.
        target_images = (
            {} if timestamp_target else factory.selected_target_images(missing_indices)
        )
        teacher_payload = None
        if missing_indices:
            teacher_payload = (
                publisher.publish(factory.video_transport_path, model)
                if publisher is not None else factory.video_payload
            )
        futures = {}
        pending_checkpoint = {}
        content_scope = EpisodeContentScope()
        def annotate_one(sampled_index):
            if timestamp_target:
                return annotate_ecot_frame(
                    client, model, system_prompt, user_prompt,
                    training_question, task_instruction, teacher_payload,
                    None, sampled_index, factory.source_frame_index(sampled_index),
                )
            target_image = target_images[sampled_index]
            image_url = getattr(client, 'ecot_image_transport', 'inline') in ('dashscope-temporary', 'ark-files', 'cos-presigned')
            if image_url:
                target_image = normalize_image(*target_image)
            image_context = (
                (getattr(client, 'cos_image_publisher', None) or publisher).published_bytes(target_image[1], model, target_image[0])
                if image_url else contextlib.nullcontext(target_image[1])
            )
            with image_context as image_payload:
                frame, request = annotate_ecot_frame(
                    client, model, system_prompt, user_prompt,
                    training_question, task_instruction, teacher_payload,
                    (target_image[0], image_payload), sampled_index,
                    factory.source_frame_index(sampled_index),
                )
            return frame, request
        def annotate_frame(sampled_index):
            with content_scope.activate():
                return annotate_one(sampled_index)
        def collect(done, propagate=True):
            first_error = None
            for future in done:
                sampled_index, key = futures.pop(future)
                try:
                    frame, request = future.result()
                except BaseException as error:
                    first_error = first_error or error
                    continue
                if checkpoint is not None:
                    pending_checkpoint[key] = {'frame': frame, 'request': request}
                completed[sampled_index] = (frame, request)
            if first_error is not None and propagate:
                raise first_error
            return first_error
        try:
            for sampled_index in missing_indices:
                content_scope.check()
                key = f'ecot_frame:{sampled_index}'
                client.ensure_available()
                future = executor.submit(annotate_frame, sampled_index)
                futures[future] = (sampled_index, key)
                if len(futures) >= max(1, workers*2):
                    done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    collect(done)
            while futures:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                collect(done)
            if checkpoint is not None and pending_checkpoint:
                checkpoint.put_many(list(pending_checkpoint.items()))
                pending_checkpoint.clear()
        finally:
            # Shared executors outlive this episode. Keep its media alive until
            # submitted work finishes. Preserve independently successful frame
            # checkpoints after a peer validation failure so resume never pays
            # for them again. Explicit provider content rejection is the one
            # case where starting more requests for this episode is pointless.
            if content_scope.error_args is not None:
                for future in futures:
                    future.cancel()
            if futures:
                done, _ = concurrent.futures.wait(futures)
                # Do not mask the exception already propagating out of this
                # scope.  Still make every independently successful response
                # durable so a restart never pays for it again.
                collect(done, propagate=False)
            if checkpoint is not None and pending_checkpoint:
                # Preserve the primary provider/validation exception, matching
                # the old peer-drain behavior. On the normal path, local
                # durability failures still propagate to the episode caller.
                primary_error_active = sys.exc_info()[0] is not None
                try:
                    checkpoint.put_many(list(pending_checkpoint.items()))
                    pending_checkpoint.clear()
                except BaseException:
                    if not primary_error_active:
                        raise
            if isinstance(teacher_payload, ArkFileReference):
                teacher_payload.close()
        frames = [completed[index][0] for index in selected_indices]
        requests = [completed[index][1] for index in selected_indices]
        sampled_frame_count = factory.sampled_frame_count
    result = {
        'status': 'accepted',
        'task_instruction': task_instruction,
        'task_instruction_source': 'global_task',
        'ecot_sample_type': ECOT_SAMPLE_TYPE,
        'sampling_fps': ECOT_FPS,
        'sampled_frame_count': sampled_frame_count,
        'ecot_interval': interval,
        'frame_selection': {
            'method': ECOT_SAMPLE_TYPE,
            'selected_when_sampled_index_mod_interval_equals_zero': True,
        },
        'annotation_input': {
            'task_instruction': task_instruction,
            'target': (
                'target_frame_timestamp_in_complete_0.5fps_episode_video'
                if timestamp_target else 'one_selected_target_image_per_request'
            ),
            'privileged_context': 'complete_0.5fps_episode_video',
        },
        'visibility_policy': (
            'training_input_is_global_task_plus_current_image_only; '
            'complete_0.5fps_episode_video_is_annotation_teacher_only'
        ),
        'annotation_contract': {
            'schema': 'privileged_video_ecot_atomic_v2',
            'system_prompt_sha256': sha256_bytes(system_prompt.encode('utf-8')),
            'user_prompt_sha256': sha256_bytes(user_prompt.encode('utf-8')),
            'one_teacher_request_per_selected_frame': True,
            'teacher_receives_complete_0_5fps_video': True,
            'teacher_sampling_fps': ECOT_TEACHER_FPS,
            'target_sampling_fps': ECOT_FPS,
            'training_excludes_privileged_video': True,
        },
        'frames': frames,
    }
    from ecot_checkpoint_reuse import provenance_summary
    provenance = provenance_summary(requests, getattr(client, 'api', None), model)
    if provenance is not None:
        result['annotation_provenance'] = provenance
    return result, requests


def subtask_media(media: list[tuple[str, bytes]], step: dict) -> tuple[list[tuple[str, bytes]], dict]:
    for mime_type, payload in media:
        if mime_type.startswith('video/'):
            start = float(step['start_time_seconds'])
            end = float(step['end_time_seconds'])
            return [('video/mp4', clip_video(payload, start, end))], {
                'kind': 'video_clip', 'start_time_seconds': start,
                'end_time_seconds': end, 'duration_seconds': end - start,
            }
    start = int(step['media_start_index'])
    end = int(step['media_end_index'])
    selected = media[start:end + 1]
    if not selected:
        raise ValueError(f'subtask_media_range_empty:{start}:{end}:{len(media)}')
    return selected, {
        'kind': 'ordered_media_slice', 'media_start_index': start,
        'media_end_index': end,
    }


def grd_video_windows(
    payload: bytes | FileSlice,
    step: dict,
    duration: float,
) -> Iterable[tuple[list[tuple[str, bytes]], dict]]:
    """Compatibility wrapper that materializes a source only once."""
    source = ''
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
            write_video_payload(stream, payload)
            source = stream.name
        yield from grd_video_path_windows(source, step, duration)
    finally:
        if source:
            Path(source).unlink(missing_ok=True)


def grd_video_path_windows(
    source: str | Path,
    step: dict,
    duration: float,
    skip_keys: set[str] | None = None,
    prefetch_pool=None,
) -> Iterable[tuple[list[tuple[str, bytes]], dict]]:
    """Yield 2 FPS anchors while reusing one materialized source video."""
    for anchor in grd_anchor_times(step, duration):
        future_end = min(duration, anchor + GRD_FUTURE_SECONDS)
        source_future_duration = future_end - anchor
        scope = {
            '_source_video_path': str(source),
            'kind': 'future_video',
            'anchor_frame_index': int(round(anchor * GRD_FRAME_FPS)),
            'anchor_time_seconds': round(anchor, 6),
            'future_start_time_seconds': round(anchor, 6),
            'future_end_time_seconds': round(future_end, 6),
            'future_duration_seconds': round(source_future_duration, 6),
            'encoded_duration_seconds': GRD_FUTURE_SECONDS,
            'end_padding': (
                'repeat_last_source_frame'
                if source_future_duration < GRD_FUTURE_SECONDS else None
            ),
            'anchor_fps': GRD_FRAME_FPS,
            'fps': GRD_FUTURE_FPS,
            'maximum_future_seconds': GRD_FUTURE_SECONDS,
            'subtask_start_time_seconds': float(step['start_time_seconds']),
            'subtask_end_time_seconds': min(float(step['end_time_seconds']), duration),
        }
        if skip_keys is not None and grd_window_key(step, scope) in skip_keys:
            yield [], scope
            continue
        def prepare(anchor=anchor, future_end=future_end):
            return [('video/mp4', clip_video_path(
                source, anchor, future_end, fps=GRD_FUTURE_FPS,
                minimum_duration=GRD_FUTURE_SECONDS,
            ))]
        yield (DeferredMedia(prefetch_pool, prepare) if prefetch_pool is not None
               else prepare()), scope


def grd_anchor_times(step: dict, duration: float) -> list[float]:
    """Return every 2 FPS grounding-frame anchor belonging to one subtask."""
    step_start = float(step['start_time_seconds'])
    step_end = min(float(step['end_time_seconds']), duration)
    first_index = max(0, int(np.ceil(step_start * GRD_FRAME_FPS - 1e-9)))
    final_index = int(np.ceil(min(step_end, duration) * GRD_FRAME_FPS - 1e-9))
    return [
        round(anchor_index / GRD_FRAME_FPS, 9)
        for anchor_index in range(first_index, final_index)
        if anchor_index / GRD_FRAME_FPS < step_end
        and anchor_index / GRD_FRAME_FPS < duration
    ]


def grd_media_windows(
    media: list[tuple[str, bytes]],
    step: dict,
    skip_keys: set[str] | None = None,
) -> Iterable[tuple[list[tuple[str, bytes]], dict]]:
    for mime_type, payload in media:
        if mime_type.startswith('video/'):
            source = ''
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
                    write_video_payload(stream, payload)
                    source = stream.name
                duration = video_path_duration(source)
                if duration is None:
                    raise RuntimeError('grounding_video_duration_unavailable')
                yield from grd_video_path_windows(
                    source, step, duration, skip_keys,
                )
            finally:
                if source:
                    Path(source).unlink(missing_ok=True)
            return
    start = int(step['media_start_index'])
    end = int(step['media_end_index'])
    for anchor in range(start, end + 1):
        selected_count = len(
            media[anchor:anchor + int(round(GRD_FUTURE_FPS * GRD_FUTURE_SECONDS))]
        )
        if selected_count:
            scope = {
                'kind': 'future_ordered_media',
                'anchor_media_index': anchor,
                'future_media_start_index': anchor,
                'future_media_end_index': anchor + selected_count - 1,
                'source_future_frame_count': selected_count,
                'encoded_frame_count': int(round(
                    GRD_FUTURE_FPS * GRD_FUTURE_SECONDS
                )),
                'end_padding': (
                    'repeat_last_source_frame'
                    if selected_count < int(round(
                        GRD_FUTURE_FPS * GRD_FUTURE_SECONDS
                    ))
                    else None
                ),
                'anchor_fps': GRD_FRAME_FPS,
                'fps': GRD_FUTURE_FPS,
                'encoded_duration_seconds': GRD_FUTURE_SECONDS,
                'maximum_future_seconds': GRD_FUTURE_SECONDS,
            }
            if skip_keys is not None and grd_window_key(step, scope) in skip_keys:
                yield [], scope
                continue
            yield [('video/mp4', ordered_future_video(media, anchor))], scope


class GrdWindowFactory:
    """Reuse one source video and one episode-level 2 FPS frame cache."""

    def __init__(
        self,
        media: list[tuple[str, bytes | FileSlice]],
        media_factory: EcotMediaFactory | None = None,
        prefetch_pool=None,
    ):
        self.media = media
        self.media_factory = media_factory
        self.prefetch_pool = prefetch_pool if isinstance(prefetch_pool, MediaPrefetchPool) else None
        self.source = ''
        self.owns_source = False
        self.duration: float | None = None
        self.current_frames: dict[int, tuple[str, bytes]] = {}

    def __enter__(self):
        if (
            self.media_factory is not None
            and self.media_factory.source_path is not None
        ):
            self.source = str(self.media_factory.source_path)
            self.duration = video_path_duration(self.source)
            if self.duration is None:
                raise RuntimeError('grounding_video_duration_unavailable')
            return self
        for mime_type, payload in self.media:
            if not mime_type.startswith('video/'):
                continue
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as stream:
                write_video_payload(stream, payload)
                self.source = stream.name
                self.owns_source = True
            self.duration = video_path_duration(self.source)
            if self.duration is None:
                Path(self.source).unlink(missing_ok=True)
                self.source = ''
                raise RuntimeError('grounding_video_duration_unavailable')
            break
        return self

    def __exit__(self, *_exc) -> None:
        if self.source and self.owns_source:
            Path(self.source).unlink(missing_ok=True)
        self.source = ''
        self.current_frames.clear()

    def prepare_current_frames(self, subtasks: dict) -> None:
        """Decode all required 2 FPS GRD anchors in one batch."""
        if self.media_factory is None or not self.source:
            return
        required_indices = {
            int(round(anchor * GRD_FRAME_FPS))
            for step in subtasks['subtasks']
            for anchor in grd_anchor_times(step, float(self.duration))
        }
        # Source-container duration can extend a fraction beyond the final
        # frame emitted by ffmpeg's 2 FPS filter.  Such a tail anchor has no
        # corresponding cached frame; leave it uncached so grd_current_frame
        # decodes the exact source timestamp instead of failing or clamping to
        # the preceding 2 FPS frame.
        indices = sorted(
            index for index in required_indices
            if 0 <= index < self.media_factory.sampled_frame_count
        )
        self.current_frames = self.media_factory.selected_target_images(indices)

    def windows(self, step: dict, skip_keys: set[str] | None = None):
        if self.source:
            options = {'prefetch_pool': self.prefetch_pool} if self.prefetch_pool is not None else {}
            windows = grd_video_path_windows(
                self.source, step, float(self.duration), skip_keys,
                **options,
            )
        elif skip_keys is None:
            windows = grd_media_windows(self.media, step)
        else:
            windows = grd_media_windows(self.media, step, skip_keys)
        return self._attach_current_frames(windows)

    def _attach_current_frames(self, windows):
        for future_media, scope in windows:
            index = scope.get('anchor_frame_index')
            if index in self.current_frames:
                scope['_current_frame_media'] = self.current_frames[index]
            yield future_media, scope


def grd_current_frame(
    media: list[tuple[str, bytes]],
    scope: dict,
) -> list[tuple[str, bytes]]:
    """Return exactly one current image corresponding to a GRD anchor."""
    cached = scope.get('_current_frame_media')
    if cached is not None:
        return [cached]
    source_path = scope.get('_source_video_path')
    if source_path:
        return [video_frame_at_path(
            source_path, float(scope['anchor_time_seconds']),
        )]
    for mime_type, payload in media:
        if mime_type.startswith('video/'):
            return [video_frame_at(payload, float(scope['anchor_time_seconds']))]
    anchor = int(scope['anchor_media_index'])
    if anchor < 0 or anchor >= len(media):
        raise ValueError(f'grounding_anchor_media_index_invalid:{anchor}:{len(media)}')
    mime_type, payload = normalize_image(*media[anchor])
    if not mime_type.startswith('image/'):
        raise ValueError(f'grounding_current_media_is_not_image:{mime_type}')
    return [(mime_type, payload)]


def current_instruction(step: dict, scope: dict) -> str:
    public_scope = {
        key: value for key, value in scope.items()
        if not str(key).startswith('_')
    }
    return (
        f'Current task instruction: {step["subtask"]}\n'
        'This subtask replaces the global task instruction for the complete supplied segment. '
        f'Segment scope: {json_dumps(public_scope)}. '
    )


def grd_prompt(step: dict, scope: dict) -> str:
    current_scope = {
        'kind': 'current_image',
        'anchor_time_seconds': scope.get('anchor_time_seconds'),
        'anchor_media_index': scope.get('anchor_media_index'),
        'sampling_fps': GRD_FRAME_FPS,
    }
    return current_instruction(step, current_scope) + (
        'The supplied media contains exactly one current-frame image. Build the '
        'complete inventory of clearly visible, discrete objects relevant to this '
        f'current subtask, with no more than {GRD_MAX_OBJECTS} objects. Do not reason '
        'from or describe future media in this request. '
        'Identified objects should be distinguishable by their name and alternate_name. Return '
        '{"suitable":true,"reason":"...","objects":[{'
        '"name":"...","alternate_name":"...","color":"...",'
        '"bbox_xyxy_1000":[0,0,1000,1000],'
        '"center_xy_1000":[500,500],"clearly_visible":true}]}. '
        'Every box and center uses the integer 0-to-1000 coordinate grid.'
    )


def grd_first_object_prompt(step: dict, scope: dict, inventory: dict) -> str:
    padding_note = (
        'Near the source-video end, the last real frame is repeated only to make a '
        'four-second video; repeated frames are not new future evidence. '
        if scope.get('end_padding') else ''
    )
    selectable_inventory = {
        'suitable': inventory['suitable'],
        'reason': inventory['reason'],
        'objects': inventory['objects'],
    }
    return current_instruction(step, scope) + (
        'The supplied teacher-only future observation is a video downsampled to exactly '
        '3 FPS, beginning at the current image and encoded to exactly four seconds. '
        + padding_note
        + 'Select the object that is intentionally manipulated first for the current '
        'subtask. The selection must exactly match name or alternate_name of one object '
        'in the current-frame inventory below. Select null if none of those inventoried '
        'objects can be identified as the first manipulated object. Do not return boxes '
        'or add objects. Return {"task_first_object":"...","reason":"..."}. '
        f'Current-frame inventory:\n{json_dumps(selectable_inventory)}'
    )


def validate_grd_inventory(result: dict) -> dict:
    """Validate and normalize one current-image inventory."""
    objects = result.get('objects')
    if not isinstance(objects, list):
        raise ValueError('grounding_inventory_requires_objects')
    if len(objects) > GRD_MAX_OBJECTS:
        raise ValueError(
            f'grounding_inventory_too_many_objects:{len(objects)}:'
            f'maximum={GRD_MAX_OBJECTS}'
        )
    normalized = []
    filtered = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict) or not str(item.get('name', '')).strip():
            raise ValueError(f'grounding_inventory_object_invalid:{index}')
        name = str(item['name']).strip()
        name_tokens = set(re.findall(r"[a-z0-9]+", name.casefold()))
        if item.get('clearly_visible') is not True:
            filtered.append({'name': name, 'reason': 'not_clearly_visible'})
            continue
        if name_tokens & GRD_FORBIDDEN_OBJECT_TOKENS:
            filtered.append({
                'name': name, 'reason': 'forbidden_non_object_or_scene_entity',
            })
            continue
        bbox = item.get('bbox_xyxy_1000', item.get('bbox_xyxy'))
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f'grounding_inventory_bbox_invalid:{index}:{bbox}')
        try:
            bbox = [int(round(float(value))) for value in bbox]
        except (TypeError, ValueError):
            raise ValueError(
                f'grounding_inventory_bbox_not_numeric:{index}:{bbox}'
            ) from None
        if not (
            0 <= bbox[0] < bbox[2] <= 1000
            and 0 <= bbox[1] < bbox[3] <= 1000
        ):
            raise ValueError(f'grounding_inventory_bbox_invalid:{index}:{bbox}')
        alternate_name = str(item.get('alternate_name') or name).strip()
        center = [
            int(round((bbox[0] + bbox[2]) / 2)),
            int(round((bbox[1] + bbox[3]) / 2)),
        ]
        normalized.append({
            'name': name,
            'alternate_name': alternate_name,
            'color': str(item.get('color') or 'unknown').strip(),
            'bbox_xyxy_1000': bbox,
            'center_xy_1000': center,
            'clearly_visible': item.get('clearly_visible') is not False,
        })
    return {
        'suitable': result.get('suitable') is not False,
        'reason': str(result.get('reason') or ''),
        'objects': normalized,
        'filtered_objects': filtered,
    }


def validate_grd_first_object(result: dict, inventory: dict) -> dict:
    """Require the future-video choice to refer to the current-frame inventory."""
    reported = result.get('task_first_object')
    normalized_reported = re.sub(r'\s+', ' ', str(reported or '').strip()).casefold()
    if reported is None or normalized_reported in {'', 'null', 'none', 'n/a'}:
        return {
            'task_first_object': None,
            'reported_task_first_object': reported,
            'reason': str(result.get('reason') or ''),
        }
    matched = None
    exact_matches = []
    for item in inventory['objects']:
        names = (item['name'], item['alternate_name'])
        if normalized_reported in {
            re.sub(r'\s+', ' ', name.strip()).casefold() for name in names
        }:
            exact_matches.append(item['name'])
    if len(set(exact_matches)) == 1:
        matched = exact_matches[0]
    if matched is None and not exact_matches:
        # Providers occasionally copy an inventory label and append a color or
        # parenthetical visual descriptor despite the exact-copy instruction.
        # Canonicalize only when the expanded wording identifies one inventory
        # object unambiguously; never accept a new object based on loose overlap.
        reported_tokens = re.findall(r'[a-z0-9]+', normalized_reported)
        reported_phrase = ' '.join(reported_tokens)
        candidates = []
        for item in inventory['objects']:
            labels = (
                item['name'], item['alternate_name'],
                f'{item.get("color", "")} {item["name"]}',
                f'{item.get("color", "")} {item["alternate_name"]}',
            )
            scores = []
            for label in labels:
                tokens = re.findall(r'[a-z0-9]+', str(label).casefold())
                phrase = ' '.join(tokens)
                if not phrase:
                    continue
                if phrase in reported_phrase or reported_phrase in phrase:
                    common = len(set(tokens) & set(reported_tokens))
                    scores.append((common, len(tokens)))
            if scores:
                candidates.append((max(scores), item['name']))
        if candidates:
            best_score = max(score for score, _ in candidates)
            best_names = {
                name for score, name in candidates if score == best_score
            }
            if len(best_names) == 1:
                matched = best_names.pop()
    if matched is None:
        raise ValueError(
            f'grounding_first_object_not_in_inventory:{reported}'
        )
    return {
        'task_first_object': matched,
        'reported_task_first_object': str(reported).strip(),
        'reason': str(result.get('reason') or ''),
    }


def merge_grd_frame(
    inventory: dict,
    selection: dict,
    scope: dict,
) -> dict:
    """Combine both teacher stages into one current-frame GRD result."""
    return {
        'frames': [{
            'media_index': int(scope.get(
                'anchor_frame_index', scope.get('anchor_media_index', 0),
            )),
            'timestamp_seconds': scope.get('anchor_time_seconds'),
            'suitable': inventory['suitable'],
            'reason': inventory['reason'],
            'objects': copy.deepcopy(inventory['objects']),
            'filtered_objects': copy.deepcopy(inventory['filtered_objects']),
            'task_first_object': selection['task_first_object'],
            'task_first_object_selection': copy.deepcopy(selection),
        }],
    }


def sta_prompt(step: dict, scope: dict) -> str:
    return current_instruction(step, scope) + (
        'Find intentional contact onsets within this segment. All returned times are relative '
        'to the start of this supplied segment. Return {"contact_events":[{'
        '"event_id":"contact-1","contact_time_seconds":0.5,'
        '"agent_role":"human_hand|held_tool|robot_gripper","object_name":"...",'
        '"contact_verb":"grasp|touch|push|place|release|other"}]}. '
        'These are contact proposals only. Do not return observations or bboxes.'
    )


def cpa_prompt(step: dict, scope: dict, sta_result: dict) -> str:
    return current_instruction(step, scope) + (
        'Review the STA proposal below. All returned times are relative to this segment. Return '
        '{"reviewed_contact_events":[{"event_id":"contact-1","accepted":true,'
        '"review_reason":"...","contact_time_seconds":0.5,'
        '"agent_role":"human_hand|held_tool|robot_gripper","object_name":"...",'
        '"contact_verb":"...","interaction_bbox_xyxy_1000":'
        '[0,0,1000,1000]}]}. The interaction bbox must tightly '
        'contain the contact agent, contacted object, and visible interface at the exact '
        'contact frame. Do not return contact points in this review stage. '
        f'STA proposal:\n{json_dumps(sta_result)}'
    )


def inherit_missing_cpa_fields(cpa_result: dict, sta_result: dict) -> dict:
    """Fill omitted CPA review fields from the reviewed STA proposal."""
    from contact_validation import inherit_cpa_event_ids
    value = inherit_cpa_event_ids(cpa_result, sta_result)
    proposals = [
        item for item in sta_result.get('contact_events') or []
        if isinstance(item, dict)
    ]
    by_id = {
        str(item.get('event_id')): item for item in proposals
        if item.get('event_id') is not None
    }
    for index, event in enumerate(value.get('reviewed_contact_events') or []):
        if not isinstance(event, dict):
            continue
        proposal = by_id.get(str(event.get('event_id')))
        if proposal is None and index < len(proposals):
            proposal = proposals[index]
        if proposal is None:
            continue
        inherited = []
        for field in (
            'contact_time_seconds', 'contact_media_index', 'agent_role',
            'object_name', 'contact_verb', 'observations',
        ):
            missing = field not in event or event.get(field) is None
            if field == 'observations':
                missing = not isinstance(event.get(field), list)
            if missing and field in proposal:
                event[field] = copy.deepcopy(proposal[field])
                inherited.append(field)
        if inherited:
            event['inherited_from_sta_fields'] = list(dict.fromkeys(
                list(event.get('inherited_from_sta_fields') or []) + inherited))
    return value


def subtask_at_time(subtasks: dict, timestamp_seconds: float) -> dict | None:
    """Return the half-open subtask interval active at a source-video time."""
    steps = subtasks.get('subtasks') or []
    timestamp = float(timestamp_seconds)
    for index, step in enumerate(steps):
        start = float(step.get('start_time_seconds', 0))
        end = float(step.get('end_time_seconds', start))
        if start <= timestamp < end or (index == len(steps) - 1 and timestamp == end):
            return step
    return None


def subtask_at_media_index(subtasks: dict, media_index: int) -> dict | None:
    """Return the inclusive ordered-media subtask active at one image index."""
    for step in subtasks.get('subtasks') or []:
        start = int(step.get('media_start_index', 0))
        end = int(step.get('media_end_index', start))
        if start <= media_index <= end:
            return step
    return None


def subtask_timeline(subtasks: dict) -> list[dict]:
    result = []
    for step in subtasks.get('subtasks') or []:
        result.append({
            'id': step.get('id'),
            'subtask': step.get('subtask'),
            'start_time_seconds': step.get('start_time_seconds'),
            'end_time_seconds': step.get('end_time_seconds'),
            'media_start_index': step.get('media_start_index'),
            'media_end_index': step.get('media_end_index'),
        })
    return result


def bounded_frame_indices(
    first: int,
    last: int,
    important: Iterable[int],
    maximum: int,
) -> list[int]:
    if last < first:
        return []
    priority = sorted({value for value in important if first <= value <= last})
    if last - first + 1 <= maximum:
        return list(range(first, last + 1))
    if len(priority) >= maximum:
        return priority[:maximum]
    uniform = [
        int(round(value))
        for value in np.linspace(first, last, maximum - len(priority))
    ]
    selected = sorted(set(priority + uniform))
    if len(selected) < maximum:
        for value in range(first, last + 1):
            if value not in selected:
                selected.append(value)
                if len(selected) == maximum:
                    break
        selected.sort()
    return selected[:maximum]


def label_contact_candidate(
    frame: np.ndarray,
    media_index: int,
    source_frame_index: int,
    timestamp_seconds: float | None,
    step: dict | None,
) -> bytes:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / max(width, height))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(2, round(width * scale)), max(2, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    label_time = 'n/a' if timestamp_seconds is None else f'{timestamp_seconds:.6f}s'
    step_id = 'none' if step is None else str(step.get('id'))
    step_text = 'outside subtask timeline' if step is None else str(step.get('subtask'))
    lines = [
        f'candidate={media_index} source_frame={source_frame_index} time={label_time}',
        f'active_subtask={step_id}: {step_text}',
    ]
    overlay_height = 58
    cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
    for line_index, text in enumerate(lines):
        cv2.putText(
            frame, text[:150], (8, 22 + 27 * line_index), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError('contact_frame_candidate_encode_failed')
    return encoded.tobytes()


def contact_frame_candidates(
    media: list[tuple[str, bytes]],
    subtasks: dict,
    proposed_time_seconds: float | None,
    proposed_media_index: int | None = None,
    window_seconds: float = CONTACT_FRAME_WINDOW_SECONDS,
    maximum_images: int = CONTACT_FRAME_MAX_IMAGES,
) -> tuple[list[tuple[str, bytes]], list[dict]]:
    """Build labeled candidates that retain subtask changes around a contact."""
    for mime_type, payload in media:
        if not mime_type.startswith('video/'):
            continue
        if proposed_time_seconds is None:
            raise ValueError('contact_frame_video_requires_proposed_time')
        name = ''
        try:
            source, name = decode_video_path(payload)
            capture = open_video_capture(source)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or count <= 0:
                capture.release()
                raise RuntimeError('contact_frame_video_metadata_invalid')
            proposed = min(max(0.0, float(proposed_time_seconds)), (count - 1) / fps)
            first = max(0, int(np.floor((proposed - window_seconds) * fps)))
            last = min(count - 1, int(np.ceil((proposed + window_seconds) * fps)))
            important = [int(round(proposed * fps))]
            for step in subtasks.get('subtasks') or []:
                boundary = step.get('start_time_seconds')
                if isinstance(boundary, (int, float)) and first / fps <= boundary <= last / fps:
                    boundary_frame = int(round(float(boundary) * fps))
                    important.extend((boundary_frame - 1, boundary_frame))
            indices = bounded_frame_indices(first, last, important, maximum_images)
            request_media = []
            candidates = []
            for media_index, frame_index in enumerate(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                timestamp = frame_index / fps
                active = subtask_at_time(subtasks, timestamp)
                request_media.append((
                    'image/jpeg',
                    label_contact_candidate(
                        frame, media_index, frame_index, timestamp, active,
                    ),
                ))
                candidates.append({
                    'media_index': media_index,
                    'source_frame_index': frame_index,
                    'timestamp_seconds': round(timestamp, 6),
                    'subtask_id': None if active is None else active.get('id'),
                    'subtask': None if active is None else active.get('subtask'),
                })
            capture.release()
            if not request_media:
                raise RuntimeError('contact_frame_candidates_empty')
            return request_media, candidates
        finally:
            if name:
                Path(name).unlink(missing_ok=True)

    images = []
    for mime_type, payload in media:
        if mime_type.startswith('image/'):
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
    if not images:
        raise RuntimeError('contact_frame_candidates_unavailable')
    center = min(max(0, int(proposed_media_index or 0)), len(images) - 1)
    first, last = max(0, center - maximum_images // 2), min(len(images) - 1, center + maximum_images // 2)
    indices = bounded_frame_indices(first, last, [center], maximum_images)
    request_media, candidates = [], []
    for media_index, source_index in enumerate(indices):
        active = subtask_at_media_index(subtasks, source_index)
        request_media.append((
            'image/jpeg',
            label_contact_candidate(images[source_index], media_index, source_index, None, active),
        ))
        candidates.append({
            'media_index': media_index,
            'source_media_index': source_index,
            'subtask_id': None if active is None else active.get('id'),
            'subtask': None if active is None else active.get('subtask'),
        })
    return request_media, candidates


def contact_frame_prompt(
    step: dict,
    subtasks: dict,
    event: dict,
    candidates: list[dict],
) -> str:
    return (
        f'Requested event subtask: {json_dumps(subtask_timeline({"subtasks": [step]})[0])}\n'
        f'Complete video subtask timeline: {json_dumps(subtask_timeline(subtasks))}\n'
        f'Proposed event: {json_dumps(event)}\n'
        f'Candidate mapping in chronological media order: {json_dumps(candidates)}\n'
        'The candidate window may cross one or more subtask boundaries. Use the active '
        'subtask printed on every image and the complete timeline to distinguish adjacent '
        'activities. Choose the first visible separation-to-contact onset only if its '
        'candidate subtask_id equals the requested event subtask id. Otherwise reject it. '
        'Return {"valid_contact":true,"selected_media_index":0,'
        '"selected_subtask_id":1,"reason":"..."}. For rejection, return '
        '{"valid_contact":false,"selected_media_index":null,'
        '"selected_subtask_id":null,"reason":"..."}.'
    )


def refine_contact_frames(
    client: ApiClient,
    model: str,
    step: dict,
    subtasks: dict,
    source_media: list[tuple[str, bytes]],
    source_time_offset: float,
    contact_result: dict,
    checkpoint: AnnotationUnitCheckpoint | None = None,
    checkpoint_prefix: str = '',
) -> tuple[dict, list[dict]]:
    """Use subtask-aware evidence to refine CPA reviews or raw STA proposals."""
    value = copy.deepcopy(contact_result)
    audits = []
    events = value.get('reviewed_contact_events')
    if not isinstance(events, list):
        events = value.get('contact_events') or []
    for event_index, event in enumerate(events):
        if event.get('accepted') is False:
            continue
        checkpoint_key = (
            f'{checkpoint_prefix}contact_frame:{event_index}:'
            f'{event.get("event_id")}'
        )
        cached = checkpoint.get(checkpoint_key) if checkpoint is not None else None
        if isinstance(cached, dict):
            events[event_index] = copy.deepcopy(cached['event'])
            audits.append(copy.deepcopy(cached['audit']))
            continue
        local_time = event.get('contact_time_seconds')
        proposed_global = (
            source_time_offset + float(local_time)
            if isinstance(local_time, (int, float)) else None
        )
        request_media, candidates = contact_frame_candidates(
            source_media, subtasks, proposed_global,
            event.get('contact_media_index'),
        )
        prompt = contact_frame_prompt(step, subtasks, event, candidates)
        selection, raw = client.request_json(
            'contact_frame', model, CONTACT_FRAME_SYSTEM, prompt, request_media,
        )
        audit = {
            'event_id': event.get('event_id'), 'model': model,
            'prompt': prompt, 'raw_response': raw, 'candidates': candidates,
        }
        audits.append(audit)
        selected_index = selection.get('selected_media_index')
        valid = selection.get('valid_contact') is True
        if valid and isinstance(selected_index, (int, float)):
            selected_index = int(selected_index)
            valid = 0 <= selected_index < len(candidates)
        else:
            valid = False
        candidate = candidates[selected_index] if valid else None
        if valid:
            selected_step_id = str(candidate.get('subtask_id'))
            requested_step_id = str(step.get('id'))
            response_step_id = str(selection.get('selected_subtask_id'))
            valid = selected_step_id == requested_step_id == response_step_id
        selection_record = {
            'version': 'subtask_timeline_candidate_frames_v1',
            'valid_contact': valid,
            'requested_subtask_id': step.get('id'),
            'selected_media_index': selected_index if valid else None,
            'selected_candidate': candidate if valid else None,
            'reason': str(selection.get('reason') or ''),
            'candidate_count': len(candidates),
            'crosses_subtask_boundary': len({
                str(item.get('subtask_id')) for item in candidates
            }) > 1,
        }
        event['contact_frame_selection'] = selection_record
        if not valid:
            event['accepted'] = False
            event['review_reason'] = (
                f"{event.get('review_reason', '')}; contact frame rejected: "
                f"{selection_record['reason']}"
            ).strip('; ')
        else:
            event['accepted'] = True
            event['proposed_contact_time_seconds'] = event.get('contact_time_seconds')
            event['contact_subtask_id'] = candidate.get('subtask_id')
            event['contact_subtask'] = candidate.get('subtask')
            if 'timestamp_seconds' in candidate:
                selected_global = float(candidate['timestamp_seconds'])
                event['contact_time_seconds'] = round(
                    selected_global - source_time_offset, 6,
                )
                event['contact_source_frame_index'] = candidate.get(
                    'source_frame_index'
                )
            else:
                event['contact_media_index'] = candidate.get('source_media_index')
        if checkpoint is not None:
            checkpoint.put(checkpoint_key, {'event': event, 'audit': audit})
    return value, audits


def extract_media_frame(media: list[tuple[str, bytes]], event: dict) -> np.ndarray:
    for mime_type, payload in media:
        if not mime_type.startswith('video/'):
            continue
        name = ''
        try:
            source, name = decode_video_path(payload)
            capture = open_video_capture(source)
            when = max(0.0, float(event.get('contact_time_seconds', 0)))
            capture.set(cv2.CAP_PROP_POS_MSEC, when * 1000.0)
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                raise RuntimeError(f'cpa_contact_frame_decode_failed:{when}')
            return frame
        finally:
            if name:
                Path(name).unlink(missing_ok=True)
    images = []
    for mime_type, payload in media:
        if mime_type.startswith('image/'):
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                images.append(image)
    if not images:
        raise RuntimeError('cpa_contact_frame_unavailable')
    index = event.get('contact_media_index', 0)
    index = int(index) if isinstance(index, (int, float)) else 0
    return images[min(max(0, index), len(images) - 1)]


def expanded_interaction_crop(
    frame: np.ndarray,
    bbox_xyxy_1000: list,
    padding: float,
    maximum_size: int,
) -> tuple[np.ndarray, list[int]]:
    if not isinstance(bbox_xyxy_1000, list) or len(bbox_xyxy_1000) != 4:
        raise ValueError(f'cpa_interaction_bbox_invalid:{bbox_xyxy_1000}')
    values = [float(value) for value in bbox_xyxy_1000]
    if not (0 <= values[0] < values[2] <= 1000 and 0 <= values[1] < values[3] <= 1000):
        raise ValueError(f'cpa_interaction_bbox_invalid:{bbox_xyxy_1000}')
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (
        values[0] / 1000 * width, values[1] / 1000 * height,
        values[2] / 1000 * width, values[3] / 1000 * height,
    )
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    box = [
        max(0, int(np.floor(x1 - pad_x))),
        max(0, int(np.floor(y1 - pad_y))),
        min(width, int(np.ceil(x2 + pad_x))),
        min(height, int(np.ceil(y2 + pad_y))),
    ]
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError(f'cpa_interaction_crop_degenerate:{box}')
    crop = frame[box[1]:box[3], box[0]:box[2]]
    scale = min(1.0, maximum_size / max(crop.shape[:2]))
    if scale != 1.0:
        crop = cv2.resize(
            crop,
            (max(2, round(crop.shape[1] * scale)), max(2, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    elif max(crop.shape[:2]) < maximum_size:
        scale = maximum_size / max(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (max(2, round(crop.shape[1] * scale)), max(2, round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop, box


def cpa_contact_event_name(event: dict) -> str:
    """Return the verb-object label supplied to contact-point grounding."""
    return ' '.join(
        str(event.get(field) or '').strip()
        for field in ('contact_verb', 'object_name')
        if str(event.get(field) or '').strip()
    )


def cpa_points_prompt(step: dict, event: dict) -> str:
    event_name = cpa_contact_event_name(event)
    return current_instruction(step, {'kind': 'enlarged_interaction_bbox_at_contact'}) + (
        f'Contact event name (verb + contacted object): {json_dumps(event_name)}. '
        f'This enlarged crop is from the exact contact frame for a {event.get("agent_role")} '
        f'{event.get("contact_verb")} with {event.get("object_name")}. Identify the exact '
        'contact-agent material and contacted object. For every distinct visible interface, '
        'select paired points immediately across that interface. H_i must be on the contact '
        'agent and O_i on the contacted object; neither may be in the gap or at a generic '
        'center. Return {"hand_description":"...","object_description":"...",'
        '"contact_pairs":[{"pair_index":0,"h_xy_1000":[0,0],'
        '"o_xy_1000":[0,0]}]}. Coordinates are on this crop.'
    )


def validate_vlm_point_result(
    value: dict,
    default_hand: str = '',
    default_object: str = '',
) -> tuple[str, str, list[dict]]:
    hand = str(
        value.get('hand_description')
        or value.get('contact_agent_description')
        or default_hand
    ).strip()
    obj = str(
        value.get('object_description')
        or value.get('contact_object_description')
        or default_object
    ).strip()
    pairs = value.get('contact_pairs') or value.get('pairs')
    if not pairs and any(key in value for key in ('h_xy_1000', 'H_i', 'hand_point')):
        pairs = [value]
    if not hand or not obj or not isinstance(pairs, list) or not pairs:
        raise ValueError('cpa_vlm_point_result_invalid')
    normalized = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(f'cpa_vlm_pair_invalid:{index}')
        item = {'pair_index': int(pair.get('pair_index', index))}
        for sources, target in (
            (('h_xy_1000', 'H_i', 'hand_point', 'agent_point'), 'h_xy_1000'),
            (('o_xy_1000', 'O_i', 'object_point'), 'o_xy_1000'),
        ):
            point = next((pair.get(source) for source in sources if pair.get(source) is not None), None)
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f'cpa_vlm_point_invalid:{index}:{target}')
            item[target] = [min(1000, max(0, int(round(float(value))))) for value in point]
        normalized.append(item)
    return hand, obj, normalized


def add_sam3_points(
    client: ApiClient,
    snapper: Sam3Snapper,
    point_model: str,
    step: dict,
    segment: list[tuple[str, bytes]],
    source_media: list[tuple[str, bytes]],
    source_time_offset: float,
    cpa_result: dict,
    crop_padding: float,
    crop_size: int,
    checkpoint: AnnotationUnitCheckpoint | None = None,
    checkpoint_prefix: str = '',
) -> tuple[dict, list[dict]]:
    value = copy.deepcopy(cpa_result)
    audits = []
    events = value.get('reviewed_contact_events') or []
    semantic_reviewer = getattr(client, 'cpa_semantic_reviewer', None)
    las_selector = getattr(client, 'cpa_las_selector', None)
    for event_index, event in enumerate(events):
        if not event.get('accepted'):
            continue
        checkpoint_key = (
            f'{checkpoint_prefix}cpa_points:{event_index}:'
            f'{event.get("event_id")}'
        )
        if las_selector is not None:
            checkpoint_key += ':las-default-contact-points-v1'
        if semantic_reviewer is not None:
            from cpa_semantic_review import REVIEW_VERSION, REVIEW_MODEL
            checkpoint_key += f':{REVIEW_VERSION}:{REVIEW_MODEL}'
        cached = checkpoint.get(checkpoint_key) if checkpoint is not None else None
        if isinstance(cached, dict):
            events[event_index] = copy.deepcopy(cached['event'])
            audits.append(copy.deepcopy(cached['audit']))
            continue
        source_event = copy.deepcopy(event)
        if isinstance(source_event.get('contact_time_seconds'), (int, float)):
            source_event['contact_time_seconds'] = (
                float(source_event['contact_time_seconds']) + source_time_offset
            )
        frame_bgr = extract_media_frame(source_media, source_event)
        crop_bgr, crop_box = expanded_interaction_crop(
            frame_bgr, event.get('interaction_bbox_xyxy_1000'), crop_padding, crop_size,
        )
        def review_snapped(candidate):
            from cpa_semantic_review import review_contact_points
            previous_attempts = (checkpoint.get(checkpoint_key + ':review_attempts') or {}).get('attempts', []) if checkpoint is not None else []
            def save_attempts(attempts):
                if checkpoint is not None:
                    checkpoint.put(checkpoint_key + ':review_attempts', {'attempts': previous_attempts + attempts})
            reviewed = review_contact_points(semantic_reviewer, candidate, frame_bgr, crop_bgr,
                                              attempt_callback=save_attempts)
            reviewed['contact_semantic_review']['attempts'] = previous_attempts + reviewed['contact_semantic_review']['attempts']
            return reviewed
        initial = checkpoint.get(checkpoint_key + ':initial') if checkpoint is not None else None
        if semantic_reviewer is not None and isinstance(initial, dict):
            event = review_snapped(initial['event'])
            events[event_index] = event
            event['point_pipeline']['semantic_review'] = event['contact_semantic_review']['version']
            audit = copy.deepcopy(initial['audit'])
            audit['semantic_review'] = copy.deepcopy(event['contact_semantic_review'])
            audits.append(audit)
            checkpoint.put(checkpoint_key, {'event': event, 'audit': audit})
            continue
        event_name = cpa_contact_event_name(event)
        prompt = cpa_points_prompt(step, event)
        ok, encoded = cv2.imencode('.jpg', crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            raise RuntimeError('cpa_enlarged_crop_encode_failed')
        point_attempts = []
        las_audit = None
        if las_selector is not None:
            point_value, las_audit = las_selector.select(encoded.tobytes(), event)
            hand_prompt, object_prompt = point_value['hand_description'], point_value['object_description']
            vlm_pairs = point_value['contact_pairs']
            point_raw = json_dumps(las_audit['response'])
            prompt = 'LAS embodied_interaction_grounding image detect; operator default models.'
            point_attempts.append({'status': 'completed', 'operator_task_id': las_audit['response']['metadata']['task_id']})
        else:
            request_prompt = prompt
            for point_attempt in range(1, 4):
                point_value, point_raw = client.request_json(
                    'cpa_point', point_model,
                    'You select precise paired contact points on an enlarged exact-contact-frame crop. Return only JSON.',
                    request_prompt, [('image/jpeg', encoded.tobytes())],
                )
                try:
                    hand_prompt, object_prompt, vlm_pairs = validate_vlm_point_result(
                        point_value,
                        str(event.get('agent_role') or '').replace('_', ' '),
                        str(event.get('object_name') or ''),
                    )
                    point_attempts.append({'attempt': point_attempt, 'status': 'accepted', 'raw_response': point_raw})
                    break
                except ValueError as error:
                    point_attempts.append({'attempt': point_attempt, 'status': 'invalid', 'error': str(error), 'raw_response': point_raw})
                    if point_attempt == 3:
                        raise
                    request_prompt = (
                        prompt
                        + '\nYour previous response was invalid because it did not contain at least '
                        'one complete paired H_i/O_i coordinate. Correct it. Both coordinates must '
                        'be two-number arrays on the crop 0-to-1000 grid. Previous invalid JSON:\n'
                        + json_dumps(point_value)
                    )
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        snapped_pairs = [
            snap_pair_twice(
                snapper, crop_rgb, frame_rgb, crop_box, pair,
                pair.get('hand_description', hand_prompt), pair.get('object_description', object_prompt),
                hand_fallback_prompts=[
                    str(event.get('agent_role') or '').replace('_', ' '),
                    'robot gripper' if event.get('agent_role') == 'robot_gripper' else 'hand',
                    'held tool' if event.get('agent_role') == 'held_tool' else '',
                ],
                object_fallback_prompts=[
                    str(event.get('object_name') or ''), 'manipulated object', 'object',
                ],
            )
            for pair in vlm_pairs
        ]
        event['interaction_bbox_pixel_xyxy_expanded'] = crop_box
        event['contact_event_name'] = event_name
        event['enlarged_crop_resolution'] = [crop_bgr.shape[1], crop_bgr.shape[0]]
        event['vlm_hand_description'] = hand_prompt
        event['vlm_object_description'] = object_prompt
        event['vlm_contact_point_pairs_crop_xy_1000'] = vlm_pairs
        event['contact_point_pairs'] = snapped_pairs
        event['point_pipeline'] = {
            'version': 'las_crop_sam3_crop_map_sam3_full_v1' if las_selector is not None else 'vlm_crop_sam3_crop_map_sam3_full_v1',
            'sam_model': 'sam3',
            'crop_snap': True,
            'full_image_resnap': True,
            'final_coordinates': 'sam3_full',
        }
        audit = {
            'event_id': event.get('event_id'), 'model': 'las_operator_default' if las_selector is not None else point_model,
            'contact_event_name': event_name,
            'prompt': prompt, 'raw_response': point_raw,
            'point_attempts': point_attempts,
            'crop_box_pixel_xyxy': crop_box,
        }
        if las_audit is not None:
            event['las_contact_grounding'] = las_audit
            event['initial_contact_point_pairs_crop_xy_1000'] = copy.deepcopy(vlm_pairs)
            audit['las_operator'] = las_audit
        if semantic_reviewer is not None:
            # Keep the pre-review points durable even if review fails or is interrupted.
            if checkpoint is not None:
                checkpoint.put(checkpoint_key + ':initial', {'event': event, 'audit': audit})
            event = review_snapped(event)
            events[event_index] = event
            event['point_pipeline']['semantic_review'] = event['contact_semantic_review']['version']
            audit['semantic_review'] = copy.deepcopy(event['contact_semantic_review'])
        audits.append(audit)
        if checkpoint is not None:
            checkpoint.put(checkpoint_key, {'event': event, 'audit': audit})
    return value, audits


def globalize_times(result: dict, offset: float) -> dict:
    value = copy.deepcopy(result)
    events = value.get('contact_events') or value.get('reviewed_contact_events') or []
    for event in events:
        if not isinstance(event, dict):
            continue
        if isinstance(event.get('contact_time_seconds'), (int, float)):
            local = float(event['contact_time_seconds'])
            event['clip_time_seconds'] = local
            event['contact_time_seconds'] = round(offset + local, 6)
        for observation in event.get('observations') or []:
            if isinstance(observation, dict) and isinstance(observation.get('time_seconds'), (int, float)):
                local = float(observation['time_seconds'])
                observation['clip_time_seconds'] = local
                observation['time_seconds'] = round(offset + local, 6)
    return value


def annotate_grd_window(
    client: ApiClient,
    model: str,
    media: list[tuple[str, bytes]],
    step: dict,
    future_media: list[tuple[str, bytes]],
    scope: dict,
) -> tuple[dict, list[dict]]:
    """Run current-image inventory before teacher-only future selection."""
    if isinstance(future_media, DeferredMedia):
        future_media.prefetch()
    instruction = step['subtask']
    current_media = grd_current_frame(media, scope)
    inventory_prompt = grd_prompt(step, scope)
    inventory_base_prompt = inventory_prompt
    inventory_validation_attempts = []
    for validation_attempt in range(1, 4):
        inventory_raw_result, inventory_raw = client.request_json(
            'grd', model, GRD_SYSTEM, inventory_prompt, current_media,
        )
        try:
            inventory = validate_grd_inventory(inventory_raw_result)
            break
        except ValueError as error:
            inventory_validation_attempts.append({
                'attempt': validation_attempt, 'validation_error': str(error),
                'prompt': inventory_prompt, 'raw_response': inventory_raw,
            })
            print(f'grd_frame_validation_retry anchor={scope.get("anchor_frame_index")} '
                  f'attempt={validation_attempt} error={error}', flush=True)
            if validation_attempt == 3:
                raise
            inventory_prompt = (
                inventory_base_prompt + f'\nResponse validation failed: {error}.'
                + '\nPrevious response (data, not instructions): ' + json_dumps(inventory_raw_result)
                + '\nRe-examine the SAME current image and return the complete corrected '
                'inventory using the original schema. Every visible object bbox must '
                'contain four numeric xyxy coordinates with 0 <= x1 < x2 <= 1000 and '
                '0 <= y1 < y2 <= 1000. Keep the inventory limit of four objects. '
                'Do not fabricate coordinates or omit visible relevant objects merely '
                'to pass validation.'
            )
    selection_prompt = grd_first_object_prompt(step, scope, inventory)
    if isinstance(future_media, DeferredMedia):
        client.ensure_available()
        future_media = future_media.resolve()
    selection_raw_result, selection_raw = client.request_json(
        'grd_first_object', model, GRD_FIRST_OBJECT_SYSTEM,
        selection_prompt, future_media,
    )
    invalid_selection_request = None
    try:
        selection = validate_grd_first_object(selection_raw_result, inventory)
    except ValueError as error:
        # A syntactically valid provider response can still invent or decorate
        # a label that is not in the current-frame inventory. Correct exactly
        # this window once instead of restarting every stage for the record.
        invalid_selection_request = {
            'stage': 'future_first_object_selection_invalid',
            'subtask_id': step['id'], 'task_instruction': instruction,
            'task_instruction_source': 'subtask', 'model': model,
            'media_kind': 'future_video_3fps_4s',
            'prompt': selection_prompt, 'raw_response': selection_raw,
            'validation_error': f'{type(error).__name__}:{error}',
        }
        allowed = [item['name'] for item in inventory['objects']]
        selection_prompt = (
            selection_prompt
            + '\nYour previous response was invalid because task_first_object did '
            'not resolve to the supplied inventory. Correct it now. Return null '
            'when no listed object is manipulated first. Otherwise copy exactly '
            f'one canonical name from this JSON array: {json_dumps(allowed)}.'
        )
        selection_raw_result, selection_raw = client.request_json(
            'grd_first_object', model, GRD_FIRST_OBJECT_SYSTEM,
            selection_prompt, future_media,
        )
        selection = validate_grd_first_object(selection_raw_result, inventory)
    result = merge_grd_frame(inventory, selection, scope)
    recorded_scope = {
        'kind': 'grounding_frame',
        'anchor_frame_index': scope.get('anchor_frame_index'),
        'anchor_media_index': scope.get('anchor_media_index'),
        'anchor_time_seconds': scope.get('anchor_time_seconds'),
        'subtask_start_time_seconds': scope.get('subtask_start_time_seconds'),
        'subtask_end_time_seconds': scope.get('subtask_end_time_seconds'),
        'grounding_media': {
            'kind': 'single_current_image',
            'sampling_fps': GRD_FRAME_FPS,
        },
        'first_object_media': {
            key: copy.deepcopy(value)
            for key, value in scope.items()
            if not key.startswith('_')
        },
    }
    output = {
        'subtask': copy.deepcopy(step),
        'task_instruction': instruction,
        'task_instruction_source': 'subtask',
        'media_scope': recorded_scope,
        'result': result,
    }
    requests = [{
        'stage': 'current_frame_inventory',
        'subtask_id': step['id'], 'task_instruction': instruction,
        'task_instruction_source': 'subtask', 'model': model,
        'media_kind': 'single_current_image',
        'prompt': inventory_prompt, 'raw_response': inventory_raw,
    }, {
        'stage': 'future_first_object_selection',
        'subtask_id': step['id'], 'task_instruction': instruction,
        'task_instruction_source': 'subtask', 'model': model,
        'media_kind': 'future_video_3fps_4s',
        'prompt': selection_prompt, 'raw_response': selection_raw,
    }]
    if invalid_selection_request is not None:
        requests.insert(1, invalid_selection_request)
    if inventory_validation_attempts:
        requests[0]['base_prompt'] = inventory_base_prompt
        requests[0]['validation_attempts'] = inventory_validation_attempts
    return output, requests


def grd_window_key(step: dict, scope: dict) -> str:
    """Return a stable key for one subtask-conditioned GRD anchor."""
    anchor = scope.get('anchor_frame_index')
    kind = 'frame'
    if anchor is None:
        anchor = scope.get('anchor_media_index')
        kind = 'media'
    if not isinstance(anchor, int) or isinstance(anchor, bool) or anchor < 0:
        raise ValueError(f'grounding_checkpoint_anchor_invalid:{anchor}')
    return f'{step.get("id")}:{kind}:{anchor}'


class GrdWindowCheckpoint:
    """Durably retain each completed GRD frame until its parent record is written.

    One checkpoint line is committed only after both the current-frame inventory
    and future-video first-object selection have completed.  ``flush`` plus
    ``fsync`` makes every completed frame independently recoverable after a
    provider failure, process termination, or host interruption.
    """

    SCHEMA_VERSION = 'grd-window-checkpoint/v1'

    def __init__(
        self,
        root: Path,
        uid: str,
        subtask_sha256: str,
        model: str,
        provider: str,
        immutable_files: bool = False,
    ):
        self.identity = {
            'schema_version': self.SCHEMA_VERSION,
            'input_record_uid': uid,
            'subtask_result_sha256': subtask_sha256,
            'model': model,
            'provider': provider,
            'contract_id': TASK_CONTRACT_IDS['grd'],
        }
        self.path = grd_checkpoint_path(root, uid)
        self.lock = threading.Lock()
        self.immutable_files = immutable_files
        self.key_locks = {}
        self.entries: dict[str, tuple[dict, list[dict]]] = {}
        load_checkpoint_files(self)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        repair_partial_jsonl_tail(self.path)
        incompatible = False
        with self.path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f'invalid_grd_checkpoint_json:{self.path}:{line_number}:{error}'
                    ) from error
                if any(record.get(key) != value for key, value in self.identity.items()):
                    incompatible = True
                    break
                key = record.get('window_key')
                item = record.get('item')
                requests = record.get('requests')
                if (
                    not isinstance(key, str)
                    or not isinstance(item, dict)
                    or not isinstance(requests, list)
                ):
                    raise ValueError(
                        f'invalid_grd_checkpoint_record:{self.path}:{line_number}'
                    )
                previous = self.entries.get(key)
                value = (item, requests)
                if previous is not None and sha256_json(previous) != sha256_json(value):
                    raise ValueError(
                        f'conflicting_grd_checkpoint_window:{self.path}:{key}'
                    )
                self.entries[key] = value
        if incompatible:
            preserved = self.path.with_name(
                f'{self.path.name}.incompatible.{time.time_ns()}'
            )
            os.replace(self.path, preserved)
            self.entries.clear()
            print(
                f'grd_checkpoint_incompatible_preserved path={self.path} '
                f'evidence={preserved}', flush=True,
            )
        elif self.entries and not getattr(self, 'quiet_unit_load', False):
            print(
                f'grd_checkpoint_loaded uid={self.identity["input_record_uid"]} '
                f'windows={len(self.entries)} path={self.path}', flush=True,
            )

    def get(self, key: str) -> tuple[dict, list[dict]] | None:
        with self.lock:
            value = self.entries.get(key)
            return copy.deepcopy(value) if value is not None else None

    def keys(self) -> set[str]:
        with self.lock:
            return set(self.entries)

    def put(self, key: str, item: dict, requests: list[dict]) -> None:
        value = (item, requests)
        if self.immutable_files:
            put_checkpoint_unit(self, key, value, {
                **self.identity, 'window_key': key, 'item': item, 'requests': requests,
            })
            return
        with self.lock:
            previous = self.entries.get(key)
            if previous is not None:
                if sha256_json(previous) != sha256_json(value):
                    raise ValueError(f'conflicting_grd_checkpoint_window:{key}')
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                **self.identity,
                'window_key': key,
                'item': item,
                'requests': requests,
            }
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            self.entries[key] = copy.deepcopy(value)

    def clear(self) -> None:
        with self.lock:
            clear_checkpoint_files(self.path)
            self.entries.clear()


def grd_checkpoint_for(
    args: argparse.Namespace,
    uid: str,
    subtasks: dict,
) -> GrdWindowCheckpoint:
    return GrdWindowCheckpoint(
        args.grd_checkpoint_root,
        uid,
        sha256_json(subtasks),
        args.models['grd'],
        args.api,
        immutable_files=getattr(args, 'immutable_jsonl', False),
    )


def grd_checkpoint_path(root: Path, uid: str) -> Path:
    """Map an arbitrary record UID to a filesystem-safe checkpoint path."""
    digest = hashlib.sha256(uid.encode('utf-8')).hexdigest()
    return root/digest[:2]/f'{digest}.jsonl'


def clear_grd_checkpoint(args: argparse.Namespace, uid: str) -> None:
    """Remove a cache only after the corresponding GRD record is durable."""
    path = grd_checkpoint_path(args.grd_checkpoint_root, uid)
    clear_checkpoint_files(path)


def annotation_checkpoint_for(
    args: argparse.Namespace,
    uid: str,
    task: str,
    dependency: dict,
) -> AnnotationUnitCheckpoint | None:
    """Create a task checkpoint whose identity covers every semantic input."""
    root = getattr(args, 'annotation_checkpoint_root', None)
    if root is None:
        return None
    model_names = {
        'subtask': ('subtask',),
        'ecot': ('ecot',),
        'sta': ('sta', 'cpa', 'contact_frame'),
        'cpa': ('sta', 'cpa', 'contact_frame', 'cpa_point'),
    }[task]
    policy = {}
    if task == 'subtask':
        policy = {
            'backend': 'vendored_las',
            'adapter_version': LAS_ADAPTER_VERSION,
            'step1_model': args.las_step1_model,
            'postprocess_model': args.las_postprocess_model,
            'embodiment': args.las_embodiment,
            'use_main_as_wrist_for_robot': args.las_use_main_as_wrist_for_robot,
        }
    elif task == 'ecot':
        policy = {
            'sampling_fps': ECOT_FPS,
            'teacher_sampling_fps': ECOT_TEACHER_FPS,
            'ecot_interval': args.ecot_interval,
            'sample_type': ECOT_SAMPLE_TYPE,
            'privileged_context': 'complete_0.5fps_episode_video',
            'training_context': 'global_task_plus_current_image_only',
            'embodiment': args.las_embodiment,
        }
    elif task == 'sta':
        policy = {
            'sta_lookback_seconds': args.sta_lookback_seconds,
            'sta_observations_per_contact': args.sta_observations_per_contact,
            'sta_min_contact_gap_seconds': args.sta_min_contact_gap_seconds,
            'sta_random_seed': args.sta_random_seed,
        }
    elif task == 'cpa':
        policy = {
            'cpa_crop_padding': args.cpa_crop_padding,
            'cpa_crop_size': args.cpa_crop_size,
            'sam3_confidence_threshold': args.sam3_confidence_threshold,
            'point_backend': getattr(args, 'cpa_point_backend', 'las'),
            'las_point_model_policy': 'operator_default',
            'semantic_review': getattr(args, 'cpa_semantic_review', True),
            'semantic_review_version': 'sam3-contact-semantic-review-paired/v2',
            'semantic_review_model': 'doubao-seed-2-1-turbo-260628',
            'student_questions': getattr(args, 'cpa_student_questions', True),
            'student_history_seconds': getattr(args, 'cpa_student_history_seconds', 2.0),
            'student_fps': 5,
            'student_question_mode': getattr(args, 'cpa_question_mode', 'coordinates'),
            'student_instruction_source': 'subtask',
            'contact_handedness_version': 'contact-point-handedness-egocentric-context/v2',
            'contact_handedness_model': 'doubao-seed-2-0-lite-260215',
            'pair_policy': 'cpa-strict-ho-pairs/v1',
            'cpa_min_contact_gap_seconds': getattr(args,'cpa_min_contact_gap_seconds',0.2),
            'cpa_max_contact_gap_seconds': getattr(args,'cpa_max_contact_gap_seconds',0.4),
            'robot_agent_points': getattr(args, 'cpa_robot_agent_points', False),
        }
    checkpoint = AnnotationUnitCheckpoint(
        root,
        uid,
        task,
        {
            'provider': annotation_provider(args, task),
            'contract_id': TASK_CONTRACT_IDS[task],
            'models': {
                name: args.models.get(name)
                for name in model_names
                if args.models.get(name)
            },
            'dependency': dependency,
            'policy': policy,
        },
        immutable_files=getattr(args, 'immutable_jsonl', False),
    )
    if task == 'ecot' and getattr(args, 'ecot_reuse_partial_checkpoints', False):
        checkpoint.ecot_prior_sources = args.ecot_reuse_sources
    return checkpoint


def clear_annotation_checkpoint(
    args: argparse.Namespace,
    uid: str,
    task: str,
) -> None:
    """Remove task recovery state only after its final record is durable."""
    root = getattr(args, 'annotation_checkpoint_root', None)
    if root is None:
        return
    clear_checkpoint_files(annotation_checkpoint_path(root, task, uid))


def review_grd_inventory_item(item, reviewer, current_media, future_media=None, source_path=None, media=None):
    """Review names and retain the physical identity of the future-selected object."""
    def resolve_first_object(corrected):
        scope = item['media_scope']['first_object_media']
        supplied = future_media
        if not supplied:
            if source_path is not None:
                supplied = [('video/mp4', clip_video_path(
                    source_path, scope['future_start_time_seconds'], scope['future_end_time_seconds'],
                    fps=GRD_FUTURE_FPS, minimum_duration=GRD_FUTURE_SECONDS,
                ))]
            elif media and all(mime.startswith('image/') for mime, _ in media):
                supplied = [('video/mp4', ordered_future_video(media, scope['anchor_media_index']))]
            else:
                raise ValueError('corrected_first_object_reselection_requires_future_media')
        prompt = grd_first_object_prompt(item['subtask'], scope, corrected)
        answer = reviewer.request_stage(
            'grd_corrected_first_object', GRD_FIRST_OBJECT_SYSTEM, prompt, supplied,
            lambda value: validate_grd_first_object(value, corrected),
        )
        return answer['value'], {'media_kind': 'future_video_3fps_4s', 'model': getattr(reviewer, 'model', GRD_REVIEW_MODEL),
                                 'reason': 'Original selected name matched multiple boxes; reselect instead of guessing.',
                                 'selection': answer['value'], 'attempts': answer['attempts']}
    return review_item(item, reviewer, current_media, resolve_first_object)


def annotate_grd_window_checkpointed(
    checkpoint: GrdWindowCheckpoint | None,
    key: str,
    client: ApiClient,
    model: str,
    media: list[tuple[str, bytes]],
    step: dict,
    future_media: list[tuple[str, bytes]],
    scope: dict,
) -> tuple[dict, list[dict]]:
    deferred = future_media if isinstance(future_media, DeferredMedia) else None
    try:
        item, requests = annotate_grd_window(
            client, model, media, step, future_media, scope,
        )
        if deferred is not None:
            future_media = deferred.resolve()
        if checkpoint is not None:
            checkpoint.put(key, item, requests)
        result = review_cached_grd_window(client, media, item, requests, future_media, scope)
        admission = getattr(client, 'request_admission', None)
        if getattr(admission, 'fair_network', False) is True:
            admission.record_grd_completion(bool(result[0]['result']['frames'][0]['registered']))
        return result
    finally:
        if deferred is not None:
            deferred.close()


def review_cached_grd_window(client, media, item, requests, future_media, scope):
    """Review either a fresh or checkpointed inventory inside the frame pool."""
    reviewer = getattr(client, 'inventory_reviewer', None)
    if reviewer is None:
        reviewer = InventoryReviewer(client)
    review_media = grd_current_frame(media, scope) if len(item['result']['frames'][0]['objects']) >= 2 else []
    item = review_grd_inventory_item(item, reviewer, review_media, future_media,
                                     scope.get('_source_video_path'), media)
    return item, requests


def downstream_result(
    task: str,
    source: dict,
    media: list[tuple[str, bytes]],
    subtasks: dict,
    client: ApiClient,
    models: dict[str, str],
    snapper: Sam3Snapper | None,
    cpa_crop_padding: float,
    cpa_crop_size: int,
    sta_cache: dict[str, tuple[dict, str]],
    grd_workers: int = 1,
    include_cpa_points: bool = True,
    grd_checkpoint: GrdWindowCheckpoint | None = None,
    unit_checkpoint: AnnotationUnitCheckpoint | None = None,
    media_factory: EcotMediaFactory | None = None,
    sta_event_workers: int = 1,
) -> tuple[dict, list[dict]]:
    output = []
    requests = []
    if task == 'grd':
        completed = []
        cached_keys = grd_checkpoint.keys() if grd_checkpoint is not None else None
        content_scope = EpisodeContentScope()
        jobs = {}

        def scoped_grd_call(function, *arguments):
            with content_scope.activate():
                return function(*arguments)

        with GrdWindowFactory(
            media, media_factory, getattr(client, 'media_prefetch_pool', None),
        ) as window_factory, frame_executor(client, grd_workers, 'grd') as executor, \
                content_scope.activate(), drain_content_jobs(jobs):
            window_factory.prepare_current_frames(subtasks)
            order = 0

            def finish_grounding(done) -> None:
                for future in done:
                    item, item_requests = future.result()
                    completed.append((jobs.pop(future), item, item_requests))

            for step in subtasks['subtasks']:
                windows = iter(window_factory.windows(step, cached_keys))
                while True:
                    # Deferred windows contain descriptors only. Clip preparation
                    # overlaps inventory HTTP instead of serializing submission.
                    client.ensure_available()
                    try:
                        future_media, scope = next(windows)
                    except StopIteration:
                        break
                    key = (
                        grd_window_key(step, scope)
                        if grd_checkpoint is not None else ''
                    )
                    cached = grd_checkpoint.get(key) if grd_checkpoint is not None else None
                    if cached is not None:
                        item, item_requests = cached
                        if review_is_current(item['result']['frames'][0]):
                            completed.append((order, item, item_requests))
                            order += 1
                            continue
                        future = executor.submit(
                            scoped_grd_call, review_cached_grd_window, client, media, item,
                            item_requests, future_media, scope,
                        )
                    else:
                        future = executor.submit(
                            scoped_grd_call, annotate_grd_window_checkpointed,
                            grd_checkpoint, key, client, models['grd'], media,
                            step, future_media, scope,
                        )
                    jobs[future] = order
                    order += 1
                    if len(jobs) >= max(1, grd_workers * 2):
                        done, _ = concurrent.futures.wait(
                            jobs, return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        finish_grounding(done)
            while jobs:
                done, _ = concurrent.futures.wait(
                    jobs, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                finish_grounding(done)
        for _, item, item_requests in sorted(completed):
            output.append(item)
            requests.extend(item_requests)
        return partition_result({
            'task_instruction_source': 'subtask',
            'subtask_result_sha256': sha256_json(subtasks),
        }, output), requests
    def annotate_contact_segment(step):
        from contact_validation import request_contact_unit, validate_contact_events, inherit_cpa_event_ids
        client.ensure_available()
        instruction = step['subtask']
        cache_key = str(step['id'])
        proposal_key = f'sta_proposal:{cache_key}'
        video = any(mime.startswith('video/') for mime, _ in media) or not media
        validate_proposal = lambda value: validate_contact_events(value, 'sta', video)
        segment_key = (
            f'segment:{task}:{cache_key}:points={int(task == "cpa" and include_cpa_points)}'
        )
        if task == 'cpa' and include_cpa_points and getattr(client, 'cpa_semantic_reviewer', None) is not None:
            segment_key += ':sam3-contact-semantic-review-v1'
        if task == 'cpa' and include_cpa_points and getattr(client, 'cpa_las_selector', None) is not None:
            segment_key += ':las-default-contact-points-v1'
        cached_segment = (
            unit_checkpoint.get(segment_key)
            if unit_checkpoint is not None else None
        )
        invalid_segment = None
        if isinstance(cached_segment, dict):
            try:
                cached_sta = cached_segment['sta_cache']
                validate_proposal(cached_sta['result'])
                if task == 'cpa':
                    cached_segment['item']['result'] = inherit_cpa_event_ids(
                        cached_segment['item']['result'], cached_sta['result'])
                validate_contact_events(cached_segment['item']['result'], task, video)
                replacement = ((unit_checkpoint.get(proposal_key+':recovered-v2')
                                or unit_checkpoint.get(proposal_key+':recovered-v1'))
                               if unit_checkpoint is not None else None)
                if replacement is not None and sha256_json(replacement['result']) != sha256_json(cached_sta['result']):
                    raise ValueError('contact_segment_stale_proposal_dependency')
            except (ValueError, TypeError, KeyError) as error:
                invalid_segment = {'checkpoint_key': segment_key, 'error': str(error),
                                   'cached_value': copy.deepcopy(cached_segment)}
                print(f'contact_segment_invalid_preserved key={segment_key} error={error}', flush=True)
            else:
                sta_cache[cache_key] = (
                    copy.deepcopy(cached_sta['result']), str(cached_sta['raw']),
                )
                return copy.deepcopy(cached_segment['item']), copy.deepcopy(cached_segment['requests'])
        if any(mime.startswith('video/') for mime, _ in media):
            start, end = float(step['start_time_seconds']), float(step['end_time_seconds'])
            scope = {'kind': 'video_clip', 'start_time_seconds': start,
                     'end_time_seconds': end, 'duration_seconds': end-start}
            segment = None
        else:
            segment, scope = subtask_media(media, step)
        def request_segment():
            # Valid proposal/review checkpoints do not need another video encode.
            nonlocal segment
            if segment is None:
                segment, _ = subtask_media(media, step)
            return segment
        sta_request = sta_prompt(step, scope)
        cached_memory = sta_cache.get(cache_key)
        proposal_unit, used_proposal_key = request_contact_unit(
            unit_checkpoint, proposal_key, 'sta', sta_request,
            lambda prompt: request_with_video_fallback(client, 'sta', models['sta'], STA_SYSTEM, prompt, request_segment()),
            validate_proposal,
            fallback=({'result': cached_memory[0], 'raw': cached_memory[1]} if cached_memory is not None else None),
        )
        sta_local, sta_raw = copy.deepcopy(proposal_unit['result']), str(proposal_unit['raw'])
        sta_cache[cache_key] = copy.deepcopy((sta_local, sta_raw))
        proposal_recovered = used_proposal_key != proposal_key
        extra_segment_evidence = []
        if proposal_recovered or invalid_segment is not None:
            legacy_segment_key = segment_key + ':sta=' + sha256_json(sta_local)
            segment_key += ':contact-v2:sta=' + sha256_json(sta_local)
            for candidate_key in (segment_key, legacy_segment_key):
                recovered_segment = unit_checkpoint.get(candidate_key) if unit_checkpoint is not None else None
                if not isinstance(recovered_segment, dict):
                    continue
                try:
                    validate_proposal(recovered_segment['sta_cache']['result'])
                    if task == 'cpa':
                        recovered_segment['item']['result'] = inherit_cpa_event_ids(
                            recovered_segment['item']['result'], sta_local)
                    validate_contact_events(recovered_segment['item']['result'], task, video)
                    if sha256_json(recovered_segment['sta_cache']['result']) != sha256_json(sta_local):
                        raise ValueError('recovered_contact_segment_dependency_mismatch')
                except (ValueError, TypeError, KeyError) as error:
                    extra_segment_evidence.append({'checkpoint_key': candidate_key, 'error': str(error),
                                                   'cached_value': copy.deepcopy(recovered_segment)})
                    continue
                return copy.deepcopy(recovered_segment['item']), copy.deepcopy(recovered_segment['requests'])
        validation_history = list(proposal_unit.get('validation_attempts', []))
        validation_history.extend(extra_segment_evidence)
        if invalid_segment is not None:
            validation_history.append(invalid_segment)
        if task == 'sta':
            prompt, raw, model = proposal_unit.get('prompt', sta_request), sta_raw, models['sta']
            result = globalize_times(sta_local, float(scope.get('start_time_seconds', 0)))
            extra_requests = []
        else:
            prompt = cpa_prompt(step, scope, sta_local)
            review_key = f'cpa_review:{cache_key}'
            if proposal_recovered:
                review_key += ':sta=' + sha256_json(sta_local)
            def validate_review(value):
                validate_contact_events(value, 'cpa', video, require_time=False)
                value = inherit_missing_cpa_fields(value, sta_local)
                return validate_contact_events(value, 'cpa', video)
            review_unit, used_review_key = request_contact_unit(
                unit_checkpoint, review_key, 'cpa', prompt,
                lambda query: request_with_video_fallback(client, 'cpa', models['cpa'], CPA_SYSTEM, query, request_segment()),
                validate_review,
            )
            cpa_local, raw = copy.deepcopy(review_unit['result']), str(review_unit['raw'])
            prompt = review_unit.get('prompt', prompt)
            validation_history.extend(review_unit.get('validation_attempts', []))
            contact_prefix = f'subtask:{cache_key}:'
            if proposal_recovered or used_review_key != review_key:
                contact_prefix += 'review=' + sha256_json(cpa_local) + ':'
            cpa_local, contact_frame_requests = refine_contact_frames(
                client, models['contact_frame'], step, subtasks, media,
                float(scope.get('start_time_seconds', 0)), cpa_local,
                checkpoint=unit_checkpoint,
                checkpoint_prefix=contact_prefix,
            )
            if include_cpa_points:
                if snapper is None:
                    raise RuntimeError('sam3_snapper_required_for_cpa')
                cpa_local, point_requests = add_sam3_points(
                    client, snapper, models['cpa_point'], step, segment or [], media,
                    float(scope.get('start_time_seconds', 0)), cpa_local,
                    cpa_crop_padding, cpa_crop_size,
                    checkpoint=unit_checkpoint,
                    checkpoint_prefix=contact_prefix,
                )
            else:
                point_requests = []
            cpa_local['parent_sta_result_sha256'] = sha256_json(sta_local)
            result = globalize_times(cpa_local, float(scope.get('start_time_seconds', 0)))
            model = models['cpa']
            extra_requests = contact_frame_requests + point_requests
        item = {
            'subtask': copy.deepcopy(step),
            'task_instruction': instruction,
            'task_instruction_source': 'subtask',
            'media_scope': scope,
            'result': result,
        }
        item_requests = [{
            'subtask_id': step['id'], 'task_instruction': instruction,
            'task_instruction_source': 'subtask', 'model': model,
            'prompt': prompt, 'raw_response': raw,
            **({'contact_validation_attempts': validation_history} if validation_history else {}),
            **({'sta_proposal': sta_local} if task == 'cpa' else {}),
        }] + extra_requests
        if unit_checkpoint is not None:
            unit_checkpoint.put(segment_key, {
                'item': item,
                'requests': item_requests,
                'sta_cache': {'result': sta_local, 'raw': sta_raw},
            })
        return item, item_requests

    # Subtask proposals/reviews are independent; preserve chronological output
    # and complete-timeline context while bounding pending work and HTTP/memory.
    # Sam3Snapper serializes its mutable predictor internally. Independent CPA
    # segments can overlap remote requests without sharing predictor state.
    event_workers = max(1, sta_event_workers)
    if event_workers == 1:
        for step in subtasks['subtasks']:
            item, item_requests = annotate_contact_segment(step)
            output.append(item)
            requests.extend(item_requests)
    else:
        steps = iter(subtasks['subtasks'])
        pending = deque()
        # Reuse the process-wide bounded frame executor. Creating one pool per
        # resident episode multiplied 512 records by event_workers and reached
        # tens of thousands of threads before useful LAS concurrency was full.
        with frame_executor(client, event_workers, 'sta') as executor:
            try:
                while True:
                    while len(pending) < event_workers * 2:
                        step = next(steps, None)
                        if step is None:
                            break
                        pending.append(executor.submit(annotate_contact_segment, step))
                    if not pending:
                        break
                    item, item_requests = pending.popleft().result()
                    output.append(item)
                    requests.extend(item_requests)
            finally:
                for future in pending:
                    future.cancel()
    return {
        'task_instruction_source': 'subtask',
        'subtask_result_sha256': sha256_json(subtasks),
        'subtask_results': output,
    }, requests


def align_sta_to_cpa_final(sta_result: dict, cpa_result: dict) -> dict:
    """Replace STA proposals with the exact contact frames accepted by CPA."""
    value = copy.deepcopy(sta_result)
    cpa_segments = {
        str(segment.get('subtask', {}).get('id')): segment
        for segment in cpa_result.get('subtask_results') or []
    }
    accepted_total = 0
    rejected_total = 0
    for sta_segment in value.get('subtask_results') or []:
        step_id = str(sta_segment.get('subtask', {}).get('id'))
        cpa_segment = cpa_segments.get(step_id, {})
        cpa_events = (
            cpa_segment.get('result', {}).get('reviewed_contact_events') or []
        )
        proposals = sta_segment.get('result', {}).get('contact_events') or []
        proposals_by_id = {
            str(event.get('event_id')): event for event in proposals
            if event.get('event_id') is not None
        }
        accepted_events = []
        rejected_events = []
        reviewed_ids = set()
        for index, review in enumerate(cpa_events):
            event_id = str(review.get('event_id'))
            reviewed_ids.add(event_id)
            proposal = proposals_by_id.get(event_id)
            if proposal is None and index < len(proposals):
                proposal = proposals[index]
            if proposal is not None and proposal.get('event_id') is not None:
                reviewed_ids.add(str(proposal.get('event_id')))
            selection = review.get('contact_frame_selection') or {}
            accepted = (
                review.get('accepted') is True
                and selection.get('valid_contact') is True
                and isinstance(review.get('contact_time_seconds'), (int, float))
            )
            if not accepted:
                rejected = copy.deepcopy(proposal or review)
                rejected['accepted'] = False
                rejected['cpa_review_reason'] = str(
                    review.get('review_reason') or selection.get('reason') or ''
                )
                rejected['contact_frame_selection'] = copy.deepcopy(selection)
                rejected_events.append(rejected)
                rejected_total += 1
                continue
            final_event = copy.deepcopy(proposal or review)
            final_time = float(review['contact_time_seconds'])
            proposed_time = final_event.get('contact_time_seconds')
            if final_event.get('event_id') != review.get('event_id'):
                final_event['sta_proposed_event_id'] = final_event.get('event_id')
            final_event.update({
                'event_id': review.get('event_id'),
                'accepted': True,
                'contact_time_seconds': final_time,
                'sta_proposed_contact_time_seconds': proposed_time,
                'agent_role': review.get('agent_role', final_event.get('agent_role')),
                'object_name': review.get('object_name', final_event.get('object_name')),
                'contact_verb': review.get('contact_verb', final_event.get('contact_verb')),
                'contact_source_frame_index': review.get('contact_source_frame_index'),
                'contact_media_index': review.get(
                    'contact_media_index', final_event.get('contact_media_index'),
                ),
                'contact_subtask_id': review.get('contact_subtask_id'),
                'contact_subtask': review.get('contact_subtask'),
                'contact_frame_selection': copy.deepcopy(selection),
                'contact_frame_source': 'cpa_final',
                'cpa_review_reason': str(review.get('review_reason') or ''),
            })
            proposed_observations = copy.deepcopy(
                final_event.get('observations') or []
            )
            if proposed_observations:
                final_event['sta_proposed_observations'] = proposed_observations
            final_event['observations'] = []
            accepted_events.append(final_event)
            accepted_total += 1
        for proposal in proposals:
            if str(proposal.get('event_id')) in reviewed_ids:
                continue
            rejected = copy.deepcopy(proposal)
            rejected['accepted'] = False
            rejected['cpa_review_reason'] = 'CPA did not return this STA proposal.'
            rejected_events.append(rejected)
            rejected_total += 1
        segment_result = sta_segment.setdefault('result', {})
        segment_result['contact_events'] = accepted_events
        segment_result['rejected_contact_event_proposals'] = rejected_events
        segment_result['contact_frame_authority'] = 'cpa_final'
        segment_result['parent_cpa_segment_sha256'] = sha256_json(cpa_segment)
    value['contact_frame_authority'] = 'cpa_final'
    value['parent_cpa_result_sha256'] = sha256_json(cpa_result)
    value['contact_frame_summary'] = {
        'accepted': accepted_total, 'rejected': rejected_total,
    }
    return value


def sta_bbox_prompt(step: dict, event: dict, observation: dict) -> str:
    """Build the single-image grounding prompt for one CPA-final event."""
    event_name = cpa_contact_event_name(event)
    scope = {
        'kind': 'random_pre_contact_observation_image',
        'time_seconds': observation.get('time_seconds'),
        'source_frame_index': observation.get('source_frame_index'),
        'source_media_index': observation.get('source_media_index'),
        'time_to_contact_seconds': observation.get('time_to_contact_seconds'),
        'time_to_contact_frames': observation.get('time_to_contact_frames'),
    }
    return current_instruction(step, scope) + (
        f'CPA-final contact event name (verb + contacted object): '
        f'{json_dumps(event_name)}. '
        'This is exactly one randomly sampled pre-contact observation image. '
        'Ground only the object named by the contact event that will receive the '
        'future contact. Return {"contact_event_name":"...",'
        '"target_visible":true,"bbox_xyxy_1000":[0,0,1000,1000],'
        '"reason":"..."}. If that target is not clearly visible, return '
        '{"contact_event_name":"...","target_visible":false,'
        '"bbox_xyxy_1000":null,"reason":"..."}.'
    )


def validate_sta_bbox_result(result: dict) -> list[int] | None:
    """Validate one target-object bbox, allowing an explicit invisible result."""
    if not isinstance(result, dict):
        raise ValueError('sta_bbox_result_must_be_object')
    if result.get('target_visible') is False:
        return None
    bbox = result.get('bbox_xyxy_1000')
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f'sta_bbox_invalid:{bbox}')
    try:
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in bbox):
            raise ValueError('coordinates must be finite numbers, not booleans')
        values = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'sta_bbox_not_numeric:{bbox}') from error
    if not (
        0 <= values[0] < values[2] <= 1000
        and 0 <= values[1] < values[3] <= 1000
    ):
        raise ValueError(f'sta_bbox_invalid:{values}')
    return values


def deterministic_sample(
    population: list[int],
    count: int,
    seed_parts: list[object],
) -> list[int]:
    """Sample without replacement reproducibly across processes and resumes."""
    if count <= 0 or not population:
        return []
    digest = hashlib.sha256(
        json_dumps(seed_parts).encode('utf-8')
    ).hexdigest()
    generator = random.Random(int(digest[:16], 16))
    return sorted(generator.sample(population, min(count, len(population))))


def sta_random_observation_frames(
    media: list[tuple[str, bytes]],
    step: dict,
    event: dict,
    all_events: list[dict],
    lookback_seconds: float,
    observations_per_contact: int,
    minimum_contact_gap_seconds: float,
    random_seed: int,
    source_uid: str,
) -> tuple[list[tuple[dict, tuple[str, bytes]]], dict]:
    """Sample frames after the prior CPA contact and before the target contact."""
    event_time = event.get('contact_time_seconds')
    if not isinstance(event_time, (int, float)):
        raise ValueError('sta_random_observation_requires_contact_time')
    contact_time = float(event_time)
    earlier_events = [
        other
        for other in all_events
        if isinstance(other.get('contact_time_seconds'), (int, float))
        and float(other['contact_time_seconds']) < contact_time - 1e-6
    ]
    previous_event = max(
        earlier_events,
        key=lambda other: float(other['contact_time_seconds']),
        default=None,
    )
    earlier_times = [
        float(other['contact_time_seconds']) for other in earlier_events
    ]
    previous_time = (
        float(previous_event['contact_time_seconds'])
        if previous_event is not None else None
    )
    seed_parts = [
        random_seed, source_uid, step.get('id'), event.get('event_id'),
        event.get('contact_source_frame_index'), contact_time,
    ]

    for mime_type, payload in media:
        if not mime_type.startswith('video/'):
            continue
        name = ''
        try:
            source, name = decode_video_path(payload)
            capture = open_video_capture(source)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if not capture.isOpened() or fps <= 0 or frame_count <= 0:
                capture.release()
                raise RuntimeError('sta_observation_video_metadata_invalid')
            contact_frame = event.get('contact_source_frame_index')
            if not isinstance(contact_frame, (int, float)):
                contact_frame = round(contact_time * fps)
            contact_frame = min(frame_count - 1, max(0, int(contact_frame)))
            step_start = float(step.get('start_time_seconds', 0))
            step_end = float(step.get('end_time_seconds', contact_time))
            lower_time = max(step_start, contact_time - lookback_seconds)
            first_frame = max(0, int(np.ceil(lower_time * fps - 1e-9)))
            if previous_time is not None:
                first_frame = max(
                    first_frame,
                    int(np.floor(previous_time * fps + 1e-9)) + 1,
                )
            if previous_event is not None and isinstance(
                previous_event.get('contact_source_frame_index'), (int, float),
            ):
                first_frame = max(
                    first_frame,
                    int(previous_event['contact_source_frame_index']) + 1,
                )
            upper_time = contact_time - minimum_contact_gap_seconds
            last_frame = min(
                frame_count - 1,
                contact_frame - 1,
                int(np.floor(upper_time * fps + 1e-9)),
                int(np.ceil(step_end * fps - 1e-9)) - 1,
            )
            contact_frames = {
                int(other.get('contact_source_frame_index'))
                if isinstance(other.get('contact_source_frame_index'), (int, float))
                else int(round(float(other['contact_time_seconds']) * fps))
                for other in all_events
                if isinstance(other.get('contact_time_seconds'), (int, float))
            }
            eligible = [
                index for index in range(first_frame, last_frame + 1)
                if index not in contact_frames
            ] if last_frame >= first_frame else []
            selected = deterministic_sample(
                eligible, observations_per_contact, seed_parts,
            )
            samples = []
            for frame_index in selected:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or frame is None:
                    capture.release()
                    raise RuntimeError(
                        f'sta_observation_frame_decode_failed:{frame_index}'
                    )
                encoded_ok, encoded = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                if not encoded_ok:
                    capture.release()
                    raise RuntimeError(
                        f'sta_observation_frame_encode_failed:{frame_index}'
                    )
                timestamp = frame_index / fps
                intervening = [
                    value for value in earlier_times
                    if timestamp < value < contact_time - 1e-6
                ]
                if intervening:
                    capture.release()
                    raise ValueError(
                        f'sta_observation_contains_intervening_contact:{intervening}'
                    )
                metadata = {
                    'time_seconds': round(timestamp, 6),
                    'clip_time_seconds': round(timestamp - step_start, 6),
                    'source_frame_index': frame_index,
                    'time_to_contact_seconds': round(contact_time - timestamp, 6),
                    'horizon_seconds': round(contact_time - timestamp, 6),
                }
                samples.append((metadata, ('image/jpeg', encoded.tobytes())))
            capture.release()
            return samples, {
                'version': 'cpa_final_no_intervening_contact_random_v1',
                'random_seed': random_seed,
                'lookback_seconds': lookback_seconds,
                'minimum_contact_gap_seconds': minimum_contact_gap_seconds,
                'requested_observations': observations_per_contact,
                'eligible_frame_count': len(eligible),
                'selected_frame_count': len(samples),
                'interval_start_time_seconds': round(lower_time, 6),
                'interval_end_time_seconds': round(upper_time, 6),
                'previous_contact_time_seconds': previous_time,
                'previous_contact_source_frame_index': (
                    previous_event.get('contact_source_frame_index')
                    if previous_event is not None else None
                ),
                'target_contact_time_seconds': contact_time,
                'no_intervening_cpa_final_contact': True,
            }
        finally:
            if name:
                Path(name).unlink(missing_ok=True)

    contact_index = event.get('contact_media_index')
    if not isinstance(contact_index, (int, float)):
        raise ValueError('sta_random_observation_requires_contact_media_index')
    contact_index = int(contact_index)
    earlier_indices = [
        int(other['contact_media_index'])
        for other in all_events
        if isinstance(other.get('contact_media_index'), (int, float))
        and int(other['contact_media_index']) < contact_index
    ]
    previous_index = max(earlier_indices) if earlier_indices else None
    step_start_index = int(step.get('media_start_index', 0))
    step_end_index = int(step.get('media_end_index', len(media) - 1))
    lookback_frames = max(1, int(round(
        lookback_seconds * GRD_FUTURE_FPS
    )))
    gap_frames = max(1, int(np.ceil(
        minimum_contact_gap_seconds * GRD_FUTURE_FPS
    )))
    first_index = max(step_start_index, contact_index - lookback_frames)
    if previous_index is not None:
        first_index = max(first_index, previous_index + 1)
    last_index = min(step_end_index, contact_index - gap_frames)
    contact_indices = {
        int(other['contact_media_index']) for other in all_events
        if isinstance(other.get('contact_media_index'), (int, float))
    }
    eligible = [
        index for index in range(first_index, last_index + 1)
        if index not in contact_indices and 0 <= index < len(media)
    ] if last_index >= first_index else []
    selected = deterministic_sample(
        eligible, observations_per_contact, seed_parts,
    )
    samples = []
    for media_index in selected:
        selected_mime, selected_payload = normalize_image(*media[media_index])
        if not selected_mime.startswith('image/'):
            raise ValueError(f'sta_observation_media_is_not_image:{selected_mime}')
        distance = contact_index - media_index
        samples.append(({
            'source_media_index': media_index,
            'time_to_contact_frames': distance,
            'horizon_frames': distance,
        }, (selected_mime, selected_payload)))
    return samples, {
        'version': 'cpa_final_no_intervening_contact_random_v1',
        'random_seed': random_seed,
        'lookback_ordered_media_frames': lookback_frames,
        'minimum_contact_gap_frames': gap_frames,
        'requested_observations': observations_per_contact,
        'eligible_frame_count': len(eligible),
        'selected_frame_count': len(samples),
        'interval_start_media_index': first_index,
        'interval_end_media_index': last_index,
        'previous_contact_media_index': previous_index,
        'target_contact_media_index': contact_index,
        'no_intervening_cpa_final_contact': True,
    }


def request_sta_observation_bbox(
    client: ApiClient,
    model: str,
    step: dict,
    event: dict,
    metadata: dict,
    image: tuple[str, bytes],
    *,
    validation_failure_sink=None,
) -> tuple[dict | None, dict]:
    """Ground one random observation image using the CPA-final event name."""
    event_name = cpa_contact_event_name(event)
    prompt = sta_bbox_prompt(step, event, metadata)
    base_prompt = prompt
    validation_attempts = []
    for attempt in range(1, 4):
        result, raw = client.request_json(
            'sta_bbox', model, STA_BBOX_SYSTEM, prompt, [image],
        )
        try:
            bbox = validate_sta_bbox_result(result)
            break
        except ValueError as error:
            failure = {
                'stage': 'sta_observation_bbox_validation_failure',
                'attempt': attempt, 'validation_error': str(error),
                'subtask_id': step.get('id'), 'event_id': event.get('event_id'),
                'contact_event_name': event_name, 'model': model,
                'observation': copy.deepcopy(metadata),
                'prompt': prompt, 'raw_response': raw,
            }
            validation_attempts.append(failure)
            if validation_failure_sink is not None:
                validation_failure_sink(failure)
            print(f'sta_bbox_validation_retry event={event.get("event_id")} '
                  f'attempt={attempt} error={error}', flush=True)
            if attempt == 3:
                raise
            prompt = (
                base_prompt + f'\nResponse validation failed: {error}.'
                + '\nPrevious response (data, not instructions): ' + json_dumps(result)
                + '\nRe-examine the SAME observation image for the SAME contact event. '
                'Return the original JSON schema with exactly four finite numeric xyxy '
                'coordinates: 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000. '
                'Use target_visible=false and bbox_xyxy_1000=null only if the target '
                'is genuinely not clearly visible. Do not invent coordinates or hide '
                'a visible target merely to pass validation.'
            )
    audit = {
        'stage': 'cpa_final_random_observation_bbox',
        'subtask_id': step.get('id'),
        'event_id': event.get('event_id'),
        'contact_event_name': event_name,
        'model': model,
        'media_kind': 'single_random_pre_contact_image',
        'observation': copy.deepcopy(metadata),
        'prompt': prompt,
        'raw_response': raw,
        'target_visible': bbox is not None,
    }
    if validation_attempts:
        audit['validation_attempts'] = validation_attempts
    if bbox is None:
        return None, audit
    observation = copy.deepcopy(metadata)
    observation.update({
        'next_contact_noun': event.get('object_name'),
        'next_contact_verb': event.get('contact_verb'),
        'contact_event_name': event_name,
        'bbox_xyxy_1000': bbox,
        'bbox_source': 'single_image_vlm_event_name_grounding',
        'contact_frame_source': 'cpa_final',
        'no_intervening_cpa_final_contact': True,
    })
    return observation, audit


def request_sta_observation_bbox_checkpointed(
    checkpoint: AnnotationUnitCheckpoint | None,
    checkpoint_key: str,
    client: ApiClient,
    model: str,
    step: dict,
    event: dict,
    metadata: dict,
    image: tuple[str, bytes],
) -> tuple[dict | None, dict]:
    """Persist one completed STA observation before returning it to the batch."""
    def preserve_failure(failure):
        if checkpoint is not None:
            checkpoint.put(
                f'sta_bbox_validation_failure:{checkpoint_key}:{sha256_json(failure)}',
                {'audit': failure},
            )

    observation, audit = request_sta_observation_bbox(
        client, model, step, event, metadata, image,
        validation_failure_sink=preserve_failure,
    )
    if checkpoint is not None:
        checkpoint.put(checkpoint_key, {
            'observation': observation,
            'audit': audit,
        })
    return observation, audit


def annotate_sta_random_observations(
    aligned_sta: dict,
    media: list[tuple[str, bytes]],
    client: ApiClient,
    model: str,
    source_uid: str,
    lookback_seconds: float,
    observations_per_contact: int,
    minimum_contact_gap_seconds: float,
    random_seed: int,
    workers: int,
    checkpoint: AnnotationUnitCheckpoint | None = None,
) -> tuple[dict, list[dict]]:
    """Create STA bboxes only after CPA has fixed every accepted contact frame."""
    value = copy.deepcopy(aligned_sta)
    dependency_sha256 = sha256_json(value)
    entries = []
    for segment in value.get('subtask_results') or []:
        for event in segment.get('result', {}).get('contact_events') or []:
            entries.append((segment, event))
    all_events = [event for _, event in entries]
    jobs = []
    completed = []
    for entry_index, (segment, event) in enumerate(entries):
        event['contact_event_name'] = cpa_contact_event_name(event)
        event['observations'] = []
        samples, policy = sta_random_observation_frames(
            media, segment.get('subtask') or {}, event, all_events,
            lookback_seconds, observations_per_contact,
            minimum_contact_gap_seconds, random_seed, source_uid,
        )
        event['observation_sampling'] = policy
        for sample_index, (metadata, image) in enumerate(samples):
            frame_identity = metadata.get(
                'source_frame_index', metadata.get('source_media_index')
            )
            checkpoint_key = (
                f'sta_bbox:{dependency_sha256}:{entry_index}:'
                f'{event.get("event_id")}:{sample_index}:{frame_identity}'
            )
            cached = (
                checkpoint.get(checkpoint_key)
                if checkpoint is not None else None
            )
            if isinstance(cached, dict):
                completed.append((
                    entry_index, sample_index,
                    copy.deepcopy(cached.get('observation')),
                    copy.deepcopy(cached['audit']),
                ))
                continue
            jobs.append((
                entry_index, sample_index, segment.get('subtask') or {},
                event, metadata, image, checkpoint_key,
            ))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                request_sta_observation_bbox_checkpointed,
                checkpoint, checkpoint_key, client, model, step, event,
                metadata, image,
            ): (entry_index, sample_index)
            for (
                entry_index, sample_index, step, event, metadata, image,
                checkpoint_key,
            ) in jobs
        }
        for future in concurrent.futures.as_completed(pending):
            entry_index, sample_index = pending[future]
            observation, audit = future.result()
            completed.append((entry_index, sample_index, observation, audit))
    audits = []
    for entry_index, _, observation, audit in sorted(completed):
        if observation is not None:
            entries[entry_index][1]['observations'].append(observation)
        audits.append(audit)
    for _, event in entries:
        sampling = event.get('observation_sampling') or {}
        grounded_count = len(event.get('observations') or [])
        sampling['grounded_observation_count'] = grounded_count
        sampling['invisible_observation_count'] = max(
            0, int(sampling.get('selected_frame_count', 0)) - grounded_count,
        )
    value['sta_observation_policy'] = {
        'version': 'cpa_final_no_intervening_contact_random_event_bbox_v1',
        'contact_frame_authority': 'cpa_final',
        'lookback_seconds': lookback_seconds,
        'observations_per_contact': observations_per_contact,
        'minimum_contact_gap_seconds': minimum_contact_gap_seconds,
        'random_seed': random_seed,
        'bbox_input': 'single_observation_image_plus_contact_event_name',
    }
    return value, audits


def validate_sta_cpa_contact_frame_identity(
    sta_result: dict,
    cpa_result: dict,
) -> None:
    """Fail before writing if STA and CPA disagree on any final contact frame."""
    def keyed_events(result: dict, cpa: bool) -> dict[tuple[str, str], dict]:
        keyed = {}
        for segment in result.get('subtask_results') or []:
            step_id = str(segment.get('subtask', {}).get('id'))
            field = 'reviewed_contact_events' if cpa else 'contact_events'
            for event in segment.get('result', {}).get(field) or []:
                if cpa and not (
                    event.get('accepted') is True
                    and event.get('contact_frame_selection', {}).get(
                        'valid_contact'
                    ) is True
                ):
                    continue
                keyed[(step_id, str(event.get('event_id')))] = event
        return keyed

    sta_events = keyed_events(sta_result, False)
    cpa_events = keyed_events(cpa_result, True)
    if set(sta_events) != set(cpa_events):
        raise ValueError(
            'sta_cpa_final_contact_event_set_mismatch:'
            f'{sorted(sta_events)}:{sorted(cpa_events)}'
        )
    for key, sta_event in sta_events.items():
        cpa_event = cpa_events[key]
        if sta_event.get('contact_frame_selection') != cpa_event.get(
            'contact_frame_selection'
        ):
            raise ValueError(f'sta_cpa_contact_selection_mismatch:{key}')
        sta_time = sta_event.get('contact_time_seconds')
        cpa_time = cpa_event.get('contact_time_seconds')
        if isinstance(sta_time, (int, float)) or isinstance(cpa_time, (int, float)):
            if not (
                isinstance(sta_time, (int, float))
                and isinstance(cpa_time, (int, float))
                and abs(float(sta_time) - float(cpa_time)) <= 1e-6
            ):
                raise ValueError(f'sta_cpa_contact_time_mismatch:{key}')
        for field in ('contact_source_frame_index', 'contact_media_index'):
            if sta_event.get(field) != cpa_event.get(field):
                raise ValueError(
                    f'sta_cpa_contact_frame_index_mismatch:{key}:{field}'
                )


def geometry(task: str, result: dict) -> list[dict]:
    values = []
    for segment_index, segment in enumerate(result.get('subtask_results') or []):
        value = segment.get('result') or {}
        if task == 'grd':
            for frame_index, frame in enumerate(value.get('frames') or []):
                for object_index, item in enumerate(frame.get('objects') or []):
                    bbox = item.get('bbox_xyxy_1000')
                    if isinstance(bbox, list) and len(bbox) == 4:
                        values.append({
                            'geometry_id': f'grd:segment-{segment_index}:frame-{frame_index}:object-{object_index}',
                            'type': 'bbox', 'binding': {'subtask_id': segment['subtask']['id'], 'media_index': frame.get('media_index')},
                            'canonical': {'order': 'xyxy', 'unit': 'norm1000', 'value': bbox},
                            'label': item.get('name'),
                        })
            continue
        events = value.get('contact_events') or value.get('reviewed_contact_events') or []
        for event_index, event in enumerate(events):
            event_binding = {
                'subtask_id': segment['subtask']['id'],
                'event_index': event_index,
                'contact_time_seconds': event.get('contact_time_seconds'),
            }
            for observation_index, observation in enumerate(event.get('observations') or []):
                binding = {
                    'subtask_id': segment['subtask']['id'], 'event_index': event_index,
                    'observation_index': observation_index,
                    'time_seconds': observation.get('time_seconds'),
                }
                bbox = observation.get('bbox_xyxy_1000')
                if isinstance(bbox, list) and len(bbox) == 4:
                    values.append({
                        'geometry_id': f'{task}:segment-{segment_index}:event-{event_index}:observation-{observation_index}:bbox',
                        'type': 'bbox', 'binding': binding,
                        'canonical': {'order': 'xyxy', 'unit': 'norm1000', 'value': bbox},
                    })
                for pair_index, pair in enumerate(observation.get('contact_point_pairs') or []):
                    pair_index = int(pair.get('pair_index', pair_index))
                    for role, field in (('contact_agent', 'h_xy_1000'), ('contact_object', 'o_xy_1000')):
                        point = pair.get(field)
                        if isinstance(point, list) and len(point) == 2:
                            values.append({
                                'geometry_id': f'{task}:segment-{segment_index}:event-{event_index}:observation-{observation_index}:pair-{pair_index}:{role}',
                                'type': 'point', 'binding': binding,
                                'canonical': {'order': 'xy', 'unit': 'norm1000', 'value': point},
                                'participant_role': role, 'pair_index': pair_index,
                            })
            if task == 'cpa':
                for pair_index, pair in enumerate(event.get('contact_point_pairs') or []):
                    pair_index = int(pair.get('pair_index', pair_index))
                    for role, field in (('contact_agent', 'h_xy_1000'), ('contact_object', 'o_xy_1000')):
                        point = pair.get(field)
                        if isinstance(point, list) and len(point) == 2:
                            values.append({
                                'geometry_id': f'cpa:segment-{segment_index}:event-{event_index}:pair-{pair_index}:{role}',
                                'type': 'point', 'binding': event_binding,
                                'canonical': {'order': 'xy', 'unit': 'norm1000', 'value': point},
                                'participant_role': role, 'pair_index': pair_index,
                            })
    return values


def media_references(source: dict, input_locator: str) -> dict:
    media = copy.deepcopy(source.get('media') or {'layout': 'unknown', 'items': []})
    for item in media.get('items') or []:
        if item.get('container_format') == 'tar' and item.get('member'):
            continue
        member = item.pop('member', None)
        if input_locator.endswith('.tar'):
            item['relative_path'] = input_locator
            item['member'] = member
            item['container_format'] = 'tar'
        elif not item.get('relative_path'):
            item['relative_path'] = input_locator
    return media


def make_record(
    source: dict,
    task: str,
    model: str,
    result: dict,
    raw_response: str,
    prompt: str,
    input_locator: str,
    requests: list[dict] | None = None,
    provider: str = 'openai-compatible',
) -> dict:
    if task == 'grd':
        if 'inventory_review_summary' not in result:
            raise ValueError('grounding_registration_requires_inventory_review')
        reviewed = partition_result(result, result.get('subtask_results', [])
                                    + result.get('rejected_subtask_results', []))
        if reviewed != result:
            raise ValueError('grounding_registration_review_partition_invalid')
    answer = json_dumps(learner_result(result) if task == 'grd' else result)
    provenance = copy.deepcopy(source.get('provenance') or {})
    record = {
        'schema_version': 'unified-vqa-record/v2',
        'uid': f"{source['uid']}:annotation:{task}:{model}",
        'source_key': source.get('source_key'), 'split': source.get('split'),
        'dialogue': {
            'thinking_enabled': False,
            'turns': [
                {'role': 'user', 'content': prompt, 'thinking': '', 'loss': False, 'source_turn_index': 0},
                {'role': 'assistant', 'content': answer, 'thinking': '',
                 'loss': task != 'grd' or bool(result.get('subtask_results')), 'source_turn_index': 1},
            ],
        },
        'media': media_references(source, input_locator),
        'annotations': {
            'geometry': geometry(task, result) if task in {'grd', 'sta', 'cpa'} else [],
            'robot_extension': {
                'annotation_task': task, 'provider': provider,
                'model': model, 'result': result, 'requests': requests or [],
            },
        },
        'cleaning': {
            'grounding_decision': (
                'drop' if task == 'grd' and not result.get('subtask_results')
                else 'defer' if result.get('status') == 'deferred' else 'keep'
            ),
            'vlm': {'status': result.get('status', 'accepted')},
            'defer': ({'reason': result.get('reason')} if result.get('status') == 'deferred' else None),
        },
        'provenance': {
            'input_record_uid': source.get('uid'),
            'source_record_uid': provenance.get('source_record_uid') or source.get('uid'),
            'source_locator': provenance.get('source_locator') or {},
            'source_text_sha256': provenance.get('source_text_sha256'),
            'input_record_sha256': provenance.get('input_record_sha256'),
            'qa_index_original': provenance.get('qa_index_original'),
            'annotation': {
                'contract_id': TASK_CONTRACT_IDS[task],
                'input_locator': input_locator, 'model': model, 'provider': provider,
                'subtask_conditioned': task in {'grd', 'sta', 'cpa'},
                'task_instruction_source': (
                    'global_task' if task == 'ecot'
                    else None if task == 'subtask' else 'subtask'
                ),
            },
        },
    }
    if task == 'cpa' and 'student_examples' in result:
        # Each query owns its own media window. The episode container is teacher
        # evidence and must not train a dialogue against future contact footage.
        record['annotations']['robot_extension']['teacher_media'] = record['media']
        record['media'] = []
        record['dialogue']['turns'] = []
        record['student_records'] = []
        for example in result['student_examples']:
            record['student_records'].append({
                'uid': f"{record['uid']}:{example['id']}",
                'known_conditions': copy.deepcopy(example['known_conditions']),
                'media': [{'source_video': example['source_video'],
                           **copy.deepcopy(example.get('source_media',{})),
                           **copy.deepcopy(example['known_conditions']),
                           **({'choice_markers': example['choices'],
                               'choice_marker_source_frame_index': example['question_frame_index']}
                              if 'choices' in example else {})}],
                'question_frame_index': example['question_frame_index'],
                'question_mode': example.get('question_mode', 'multiple_choice'),
                'instruction_source': example.get('instruction_source', 'global_task'),
                'target_point_counts': copy.deepcopy(example.get('target_point_counts')),
                **{key:copy.deepcopy(example[key]) for key in ('hand_point_counts','response_point_counts') if key in example},
                'dialogue': {'thinking_enabled': False, 'turns': [
                    {'role': 'user', 'content': example['task_instruction'] + '\n' + example['question'], 'loss': False},
                    {'role': 'assistant', 'content': example.get('answer_text',json_dumps(example['answer'])), 'loss': True},
                ]},
            })
    return record


def subtask_lookup_keys(record: dict) -> set[str]:
    keys = {str(record.get('uid', ''))}
    provenance = record.get('provenance') or {}
    keys.add(str(provenance.get('source_record_uid', '')))
    locator = provenance.get('source_locator') or {}
    source_file = locator.get('source_file')
    storage_kind = str(locator.get('storage_kind') or '').lower()
    if source_file and (
        storage_kind in {'video', 'video_file', 'file_video'}
        or Path(str(source_file)).suffix.lower() in VIDEO_SUFFIXES
    ):
        path = Path(str(source_file))
        keys.update((str(path), str(path.resolve()), path.name, path.stem))
    return {key for key in keys if key}


class SubtaskIndex:
    def __init__(self):
        self.values: dict[str, dict] = {}
        self.lock = threading.Lock()

    def add(self, keys: Iterable[str], result: dict, source: str) -> None:
        with self.lock:
            for key in keys:
                existing = self.values.get(key)
                if existing is not None and sha256_json(existing) != sha256_json(result):
                    raise ValueError(f'conflicting_subtask_records:{key}:{source}')
                self.values[key] = result

    def find(self, source: dict) -> dict | None:
        with self.lock:
            for key in subtask_lookup_keys(source):
                if key in self.values:
                    return copy.deepcopy(self.values[key])
        return None


def subtask_from_record(record: dict) -> tuple[set[str], dict] | None:
    extension = ((record.get('annotations') or {}).get('robot_extension') or {})
    if extension.get('annotation_task') == 'subtask' and isinstance(extension.get('result'), dict):
        keys = subtask_lookup_keys(record)
        keys.add(str((record.get('provenance') or {}).get('input_record_uid', '')))
        return {key for key in keys if key}, extension['result']
    if record.get('kind') == 'subtask' and isinstance(record.get('steps'), list):
        keys = {
            str(record.get('parent_episode_id', '')), str(record.get('video_id', '')),
            str(record.get('source_video', '')),
        }
        if record.get('source_video'):
            path = Path(str(record['source_video']))
            keys.update((str(path.resolve()), path.name, path.stem))
        return {key for key in keys if key}, {
            'task_summary': record.get('task_instruction', ''), 'subtasks': record['steps'],
        }
    if isinstance(record.get('subtasks'), list) and record.get('input_record_uid'):
        return {str(record['input_record_uid'])}, record
    return None


def load_subtask_index(paths: Iterable[Path]) -> SubtaskIndex:
    index = SubtaskIndex()
    seen_files = set()
    for path in paths:
        if not path.exists():
            continue
        files = sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]
        for file in files:
            resolved = file.resolve()
            if resolved in seen_files:
                continue
            seen_files.add(resolved)
            with file.open('rb') as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    # A producer may be appending this file concurrently. Only a
                    # final non-newline-terminated fragment is ignored; malformed
                    # complete records remain fatal.
                    if not raw_line.endswith(b'\n'):
                        print(
                            f'subtask_index_partial_tail_ignored file={file} '
                            f'line={line_number} bytes={len(raw_line)}',
                            flush=True,
                        )
                        break
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f'invalid_subtask_jsonl:{file}:{line_number}:{error}'
                        ) from error
                    extracted = subtask_from_record(record)
                    if extracted is not None:
                        keys, result = extracted
                        index.add(keys, result, f'{file}:{line_number}')
    return index


def _annotate_source_from_media(
    source: dict,
    media: list[tuple[str, bytes | FileSlice]],
    input_locator: str,
    needed: set[str],
    args: argparse.Namespace,
    media_factory: EcotMediaFactory | None,
) -> dict[str, dict]:
    source_uid = str(source.get('uid'))
    records = {}
    def commit(task):
        callback = getattr(args, 'stage_commit', None)
        if callback is not None:
            callback(source_uid, task, records[task])
    subtasks: dict | None = None
    needs_subtasks = bool(needed & {'subtask', 'grd', 'sta', 'cpa'})
    if needs_subtasks:
        parent_subtasks = getattr(args, '_validated_stage_subtasks', None)
        shared_parent = parent_subtasks is not None and parent_subtasks[0] == source_uid
        subtasks = copy.deepcopy(parent_subtasks[1]) if shared_parent else args.subtask_index.find(source)
        if subtasks is not None:
            if not shared_parent:
                subtasks = validate_subtasks(subtasks, media)
            subtask_raw = json_dumps(subtasks)
            subtask_user = 'Loaded from --subtask-path or an existing output record.'
        else:
            if 'subtask' not in args.tasks:
                raise RuntimeError(f'subtask_record_not_found:{source.get("uid")}')
            subtask_checkpoint = annotation_checkpoint_for(
                args, source_uid, 'subtask', {
                    'backend': 'vendored_las',
                    'adapter_version': LAS_ADAPTER_VERSION,
                    'task_instruction': las_task_instruction(
                        source, args.task_instruction,
                    ),
                    'embodiment': las_embodiment(source, args.las_embodiment),
                },
            )
            subtasks, subtask_raw, subtask_user = request_validated_subtasks(
                args.las_subtask_annotator, source, media, args.task_instruction,
                args.las_embodiment,
                checkpoint=subtask_checkpoint,
                media_limiter=getattr(args, 'las_media_limiter', None),
            )
            if args.tasks & {'grd', 'sta', 'cpa'}:
                args.subtask_index.add(
                    subtask_lookup_keys(source), subtasks, 'current_run',
                )
    if 'subtask' in needed and subtasks is not None:
        records['subtask'] = make_record(
            source, 'subtask', args.models['subtask'], subtasks,
            subtask_raw, subtask_user, input_locator,
            requests=[{
                'backend': 'vendored_las',
                'adapter_version': LAS_ADAPTER_VERSION,
                'model': args.models['subtask'],
                'step1_model': args.las_step1_model,
                'postprocess_model': args.las_postprocess_model,
                'prompt': subtask_user,
                'raw_response': subtask_raw,
            }],
            provider='las',
        )
        commit('subtask')
    if 'grd' in needed and media_factory is not None:
        # Populate the disk-backed grid once so ECoT and GRD do not require
        # separate sequential decode passes over the same episode.
        media_factory.selected_target_images(list(range(media_factory.sampled_frame_count)))
    stage_groups = [needed & {'ecot'}, needed & {'grd'}, needed & {'sta', 'cpa'}]
    stage_groups = [group for group in stage_groups if group]
    stage_workers = getattr(args, 'downstream_stage_workers', 1)
    if (stage_workers > 1 and len(stage_groups) > 1
            and not getattr(args, '_parallel_stage_child', False)):
        # Keep STA and CPA in one dependency group. All groups borrow the same
        # original source and prepopulated 2 FPS grid until every worker exits.
        child_args = copy.copy(args)
        child_args._parallel_stage_child = True
        child_args._validated_stage_subtasks = (source_uid, subtasks)
        failures = []
        print(f'annotation_stage_parallel_start uid={source_uid} '
              f'groups={json_dumps([sorted(group) for group in stage_groups])} '
              f'workers={min(stage_workers, len(stage_groups))}', flush=True)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(stage_workers, len(stage_groups)), thread_name_prefix='annotation-stage',
        ) as executor:
            futures = {executor.submit(_annotate_source_from_media, source, media,
                                       input_locator, group, child_args, media_factory): group
                       for group in stage_groups}
            for future in concurrent.futures.as_completed(futures):
                try:
                    records.update(future.result())
                except Exception as error:
                    # Independent stages still persist their successful results.
                    # Do not release shared media while any stage is using it.
                    failures.append(error)
                    print(f'annotation_stage_failed uid={source_uid} tasks={sorted(futures[future])} '
                          f'error={type(error).__name__}:{error}', flush=True)
        if failures:
            fatal = next((error for error in failures if isinstance(error, FatalProviderError)), None)
            raise fatal if fatal is not None else failures[0]
        return records
    if 'ecot' in needed:
        ecot_checkpoint = annotation_checkpoint_for(
            args, source_uid, 'ecot', {
                'task_instruction': las_task_instruction(
                    source, args.task_instruction,
                ),
                'source_media_sha256': sha256_json(
                    source.get('media') or {},
                ),
            },
        )
        ecot_result, ecot_requests = annotate_ecot_source(
            source, media, args.client, args.models['ecot'],
            args.task_instruction, args.las_embodiment,
            args.ecot_interval, args.ecot_workers, ecot_checkpoint,
            media_factory,
        )
        records['ecot'] = make_record(
            source, 'ecot', args.models['ecot'], ecot_result,
            json_dumps(ecot_result),
            ecot_training_question(ecot_result['task_instruction']),
            input_locator, ecot_requests, provider=args.api,
        )
        commit('ecot')
    sta_cache: dict[str, tuple[dict, str]] = {}
    grd_checkpoint = (
        grd_checkpoint_for(args, source_uid, subtasks)
        if 'grd' in needed and subtasks is not None
        and getattr(args, 'grd_checkpoint_root', None) is not None
        else None
    )
    unit_checkpoints = {
        task: annotation_checkpoint_for(
            args, source_uid, task, {
                'subtask_result_sha256': sha256_json(subtasks),
                'include_cpa_points': task == 'cpa',
            },
        )
        for task in needed
        if task in {'sta', 'cpa'}
    }

    def sync_sta_record(cpa_result: dict, cpa_requests: list[dict]) -> None:
        sta_extension = records['sta']['annotations']['robot_extension']
        sta_checkpoint = unit_checkpoints.get('sta')
        legacy_final_key = f'sta_final:{sha256_json(cpa_result)}'
        final_key = f'sta_final:v2:{sha256_json(cpa_result)}'
        cached_final = None
        if sta_checkpoint is not None:
            for candidate_key in (final_key, legacy_final_key):
                candidate = sta_checkpoint.get(candidate_key)
                if not isinstance(candidate, dict):
                    continue
                try:
                    validate_sta_cpa_contact_frame_identity(candidate['result'], cpa_result)
                except (ValueError, TypeError, KeyError) as error:
                    print(f'sta_final_checkpoint_invalid_preserved key={candidate_key} '
                          f'error={str(error)[:180]}', flush=True)
                    continue
                cached_final = candidate
                break
        if isinstance(cached_final, dict):
            aligned_sta = copy.deepcopy(cached_final['result'])
            bbox_requests = copy.deepcopy(cached_final['requests'])
        else:
            aligned_sta = align_sta_to_cpa_final(
                sta_extension['result'], cpa_result,
            )
            aligned_sta, bbox_requests = annotate_sta_random_observations(
                aligned_sta, media, args.client, args.models['sta'],
                source_uid or input_locator,
                args.sta_lookback_seconds, args.sta_observations_per_contact,
                args.sta_min_contact_gap_seconds, args.sta_random_seed,
                args.sta_bbox_workers, checkpoint=sta_checkpoint,
            )
            validate_sta_cpa_contact_frame_identity(aligned_sta, cpa_result)
            if sta_checkpoint is not None:
                sta_checkpoint.put(final_key, {
                    'result': aligned_sta, 'requests': bbox_requests,
                })
        validate_sta_cpa_contact_frame_identity(aligned_sta, cpa_result)
        sync_request = {
            'stage': 'cpa_final_random_sta_observation_sync',
            'model': args.models['cpa'],
            'contact_frame_model': args.models['contact_frame'],
            'parent_cpa_result_sha256': sha256_json(cpa_result),
            'policy': (
                'Only CPA-accepted exact contact frames are retained as STA '
                'contact events. Random pre-contact frames are sampled after '
                'the prior final contact, and each bbox is grounded from one '
                'image plus the CPA-final contact event name.'
            ),
        }
        records['sta'] = make_record(
            source, 'sta', args.models['sta'], aligned_sta,
            json_dumps(aligned_sta),
            'Ground random pre-contact STA observations using CPA-final contact events.',
            input_locator,
            list(sta_extension.get('requests') or []) + list(cpa_requests) + bbox_requests + [sync_request],
            provider=args.api,
        )

    if needed & {'grd', 'sta', 'cpa'} and subtasks is None:
        raise RuntimeError(f'subtask_record_not_found:{source.get("uid")}')
    for task in ('grd', 'sta', 'cpa'):
        if task not in needed:
            continue
        result, requests = downstream_result(
            task, source, media, subtasks, args.client, args.models,
            args.sam3_snapper, args.cpa_crop_padding, args.cpa_crop_size,
            sta_cache, args.grd_workers,
            grd_checkpoint=grd_checkpoint,
            unit_checkpoint=unit_checkpoints.get(task),
            media_factory=media_factory,
            sta_event_workers=getattr(args, 'sta_event_workers', 1),
        )
        if task == 'cpa' and getattr(args, 'cpa_student_questions', False):
            from cpa_downstream import complete_source_cpa
            result = complete_source_cpa(source, media, result, args)
        records[task] = make_record(
            source, task, args.models[task], result, json_dumps(result),
            f'Run {task} independently for each subtask; each request uses the subtask as the task instruction.',
            input_locator, requests, provider=args.api,
        )
        if task != 'sta':
            commit(task)
        if task == 'cpa' and 'sta' in records:
            sync_sta_record(result, requests)
    if 'sta' in records and records['sta']['annotations']['robot_extension'][
        'result'
    ].get('contact_frame_authority') != 'cpa_final':
        cpa_review, cpa_review_requests = downstream_result(
            'cpa', source, media, subtasks, args.client, args.models,
            None, args.cpa_crop_padding, args.cpa_crop_size,
            sta_cache, args.grd_workers, include_cpa_points=False,
            unit_checkpoint=unit_checkpoints.get('sta'),
            media_factory=media_factory,
            sta_event_workers=getattr(args, 'sta_event_workers', 1),
        )
        sync_sta_record(cpa_review, cpa_review_requests)
    if 'sta' in records:
        commit('sta')
    return records


def annotate_source(
    source: dict,
    blobs: dict[str, tuple[str, bytes | FileSlice]],
    input_locator: str,
    needed: set[str],
    args: argparse.Namespace,
) -> dict[str, dict]:
    if args.client is not None:
        args.client.ensure_available()
    # Keep physical WebDataset video slices lazy, then materialize/downsample
    # exactly once per episode when ECoT or GRD needs the shared 2 FPS grid.
    media = media_payload(
        source, blobs, preserve_video_file_slices=True,
    )
    with contextlib.ExitStack() as resources:
        media_factory = getattr(args, '_record_media_factory', None)
        if needed & {'ecot', 'grd'}:
            if media_factory is None:
                prepare_limit = getattr(args, 'video_prepare_limiter', None)
                owner = getattr(args, '_record_media_resources', resources)
                with prepare_limit if prepare_limit is not None else contextlib.nullcontext():
                    media_factory = owner.enter_context(EcotMediaFactory(media))
                if owner is not resources:
                    args._record_media_factory = media_factory
            else:
                print(f'record_media_reused uid={source.get("uid")} '
                      f'sampled_frames={media_factory.sampled_frame_count}', flush=True)
        if needed & {'sta', 'cpa'}:
            cached_media = getattr(args, '_record_contact_media', None)
            if cached_media is None:
                owner = getattr(args, '_record_media_resources', resources)
                cached_media = contact_source_media(media, owner, media_factory)
                if owner is not resources:
                    args._record_contact_media = cached_media
            media = cached_media
        return _annotate_source_from_media(
            source, media, input_locator, needed, args, media_factory,
        )


def annotate_source_with_retries(
    source: dict,
    blobs: dict[str, tuple[str, bytes | FileSlice]],
    input_locator: str,
    needed: set[str],
    args: argparse.Namespace,
) -> dict[str, dict]:
    limiter = getattr(args, 'record_limiter', None)
    context = limiter if limiter is not None else contextlib.nullcontext()
    with context, contextlib.ExitStack() as media_resources:
        # Keep media alive across retries of this record only. The resident
        # gate still covers its entire lifetime, including final cleanup.
        args = copy.copy(args)
        args._record_media_resources = media_resources
        args._record_media_factory = getattr(args, '_shared_prepared_factory', None)
        content_identities = {}
        if (needed & {'ecot', 'grd'} and getattr(args, 'api', None) == 'dashscope'
                and getattr(args, 'output', None) is not None
                and isinstance(getattr(args, 'models', None), dict)):
            for content_task in sorted(needed & {'ecot', 'grd'}):
                content_identities[content_task] = rejection_identity(
                    source.get('uid'), args.models[content_task],
                    TASK_CONTRACT_IDS[content_task], content_task)
                check_saved_rejection(args.output, content_identities[content_task])
        last_error: BaseException | None = None
        for attempt in range(1, args.record_attempts + 1):
            try:
                is_complete = getattr(args, 'stage_is_complete', lambda uid, task: False)
                remaining = {task for task in needed if not is_complete(str(source.get('uid')), task)}
                if not remaining:
                    return {}
                check_available = getattr(getattr(args, 'request_admission', None), 'check_available', None)
                if check_available is not None:
                    check_available()
                return annotate_source(source, blobs, input_locator, remaining, args)
            except FatalProviderError as error:
                if args.client is not None and not args.client.fatal_stop_file.exists():
                    args.client.record_fatal(0, str(error), 'record', 'pipeline')
                elif (args.client is None and getattr(args, 'fatal_stop_file', None) is not None
                      and not args.fatal_stop_file.exists()):
                    # Subtask-only runs have no downstream ApiClient. Publish a
                    # shared stop signal so queued LAS jobs cannot keep spending.
                    from request_parallel import atomic_json
                    atomic_json(args.fatal_stop_file, {
                        'provider': 'las', 'task': 'subtask',
                        'reason': 'fatal_provider_error_in_subtask_only_run',
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                    })
                raise
            except ProviderContentRejected as error:
                content_task = 'grd' if error.task.startswith('grd') else error.task
                content_identity = content_identities.get(content_task)
                if content_identity is not None:
                    try:
                        path = save_rejection(args.output, content_identity, error.reason)
                    except Exception as storage_error:
                        raise FatalProviderError('provider_content_rejection_persistence_failed') from storage_error
                    print(f'provider_content_rejection_saved uid={source.get("uid")} '
                          f'task={content_task} path={path} registered_as_success=false', flush=True)
                raise
            except EcotValidationRetriesExhausted:
                # Successful frame checkpoints remain reusable; a later explicit
                # resume is distinct from retrying this exhausted annotation call.
                raise
            except Exception as error:
                last_error = error
                backoff = getattr(args, 'burst_backoff', None)
                if backoff is not None:
                    backoff.record_failure()
                if attempt >= args.record_attempts:
                    break
                delay = min(30.0, 2 ** attempt) + random.uniform(0, 2)
                message = str(error)
                if len(message) > 500:
                    message = message[:300] + '...[truncated]...' + message[-180:]
                print(
                    f'record_retry uid={source.get("uid")} attempt={attempt} '
                    f'delay_seconds={delay:.3f} '
                    f'error={type(error).__name__}:{message}',
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(
            f'record_failed_after_{args.record_attempts}_attempts:'
            f'{type(last_error).__name__}:{last_error}'
        ) from last_error


def completed_uids(
    path: Path,
    task: str,
    model: str,
    provider: str,
) -> set[str]:
    """Return only records compatible with the active task contract."""
    result = set()
    if not path.is_file():
        return result
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                record = json.loads(line)
                uid = compatible_annotation_uid(record, task, model, provider)
                if uid is None:
                    continue
                result.add(uid)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return result


def prior_completed_uids(roots, task, source_key, label, model, provider):
    """Read compatible prior records without repairing or modifying the producer."""
    result = set()
    for root in roots:
        path = root/task/'shards'/source_key/f'{label}.jsonl'
        for candidate in [path, *sorted(path.with_suffix('.records').glob('*.jsonl'))]:
            result.update(completed_uids(candidate, task, model, provider))
    return result


def compatible_annotation_uid(
    record: object,
    task: str,
    model: str,
    provider: str,
) -> str | None:
    """Return a UID only when identity and persisted result semantics match."""
    if not isinstance(record, dict):
        return None
    try:
        extension = record['annotations']['robot_extension']
        annotation = record['provenance']['annotation']
        uid = str(record['provenance']['input_record_uid'])
    except (KeyError, TypeError):
        return None
    if not uid or not (
        extension.get('annotation_task') == task
        and extension.get('model') == model
        and extension.get('provider') == provider
        and annotation.get('contract_id') in (
            ({TASK_CONTRACT_IDS['ecot']} | LEGACY_ECOT_CONTRACT_IDS)
            if task == 'ecot' else {TASK_CONTRACT_IDS[task]}
        )
    ):
        return None
    if validate_annotation_result_contract(
        task, extension.get('result'), extension.get('requests'),
    ):
        return None
    return uid


def sanitize_annotation_jsonl(
    path: Path,
    task: str,
    model: str,
    provider: str,
) -> set[str]:
    """Atomically quarantine invalid/duplicate rows before resuming append."""
    if not path.is_file():
        return set()
    stamp = time.time_ns()
    temporary = path.with_name(f'.{path.name}.sanitize.{os.getpid()}.{stamp}')
    evidence = path.with_name(f'{path.name}.rejected.{stamp}.jsonl.evidence')
    evidence_temporary = path.with_name(
        f'.{path.name}.rejected.{os.getpid()}.{stamp}.tmp'
    )
    seen: set[str] = set()
    rejected_count = 0
    # Large frame-rich JSONL rows otherwise require thousands of tiny reads.
    # Each read releases/reacquires the GIL amid HTTP and validation threads.
    with path.open('rb', buffering=4 * 1024 * 1024) as source:
        for raw in source:
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                rejected_count += 1
                continue
            uid = compatible_annotation_uid(record, task, model, provider)
            if uid is None or uid in seen:
                rejected_count += 1
                continue
            seen.add(uid)
    if not rejected_count:
        return seen

    accepted_count = 0
    rejected_count = 0
    seen.clear()
    try:
        with (
            path.open('rb', buffering=4 * 1024 * 1024) as source,
            temporary.open('xb') as accepted_stream,
            evidence_temporary.open('xb') as rejected_stream,
        ):
            for raw in source:
                normalized = raw if raw.endswith(b'\n') else raw + b'\n'
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    rejected_stream.write(normalized)
                    rejected_count += 1
                    continue
                uid = compatible_annotation_uid(record, task, model, provider)
                if uid is None or uid in seen:
                    rejected_stream.write(normalized)
                    rejected_count += 1
                    continue
                seen.add(uid)
                accepted_stream.write(normalized)
                accepted_count += 1
            accepted_stream.flush()
            os.fsync(accepted_stream.fileno())
            rejected_stream.flush()
            os.fsync(rejected_stream.fileno())
        os.replace(evidence_temporary, evidence)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        evidence_temporary.unlink(missing_ok=True)
    print(
        f'annotation_shard_sanitized path={path} task={task} '
        f'accepted={accepted_count} rejected={rejected_count} evidence={evidence}',
        flush=True,
    )
    return seen


def repair_partial_jsonl_tail(path: Path) -> Path | None:
    """Preserve and remove an incomplete final JSONL line before resuming append."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open('rb+') as stream:
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b'\n':
            return None
        end = stream.tell()
        position = end
        cutoff = 0
        block_size = 1024 * 1024
        while position > 0:
            start = max(0, position - block_size)
            stream.seek(start)
            block = stream.read(position - start)
            newline = block.rfind(b'\n')
            if newline >= 0:
                cutoff = start + newline + 1
                break
            position = start
        stream.seek(cutoff)
        tail = stream.read()
        evidence = path.with_name(
            f'{path.name}.partial-tail.{time.time_ns()}.bin'
        )
        with evidence.open('xb') as output:
            output.write(tail)
            output.flush()
            os.fsync(output.fileno())
        stream.truncate(cutoff)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        f'jsonl_partial_tail_repaired path={path} cutoff={cutoff} '
        f'preserved_bytes={len(tail)} evidence={evidence}', flush=True,
    )
    return evidence


def record_executor_workers(args: argparse.Namespace) -> int:
    """Return native worker threads without exceeding useful record concurrency."""
    # Every concurrently processed physical batch owns one record executor.
    # Divide the global record ceiling across those executors; otherwise a
    # 64-batch run with max_record_active=512 creates 32,768 mostly blocked
    # threads and starves heartbeat/request scheduling.
    per_batch = math.ceil(args.max_record_active / max(1, args.batch_workers))
    return max(1, min(args.workers, per_batch))


def prioritize_failed_samples(samples, error_path):
    """Reorder a bounded batch for recovery; never filter or relabel records."""
    if not isinstance(samples, list) or not error_path.is_file():
        return samples
    failed = set()
    with error_path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict) and value.get('source_record_uid'):
                    failed.add(str(value['source_record_uid']))
            except json.JSONDecodeError:
                continue
    if not failed:
        return samples
    ordered = sorted(samples, key=lambda item: str(item[0].get('uid')) not in failed)
    count = sum(str(source.get('uid')) in failed for source, _ in ordered)
    print(f'failed_record_priority count={count} batch_records={len(samples)} scope_unchanged=true', flush=True)
    return ordered


def process_batch(
    args: argparse.Namespace,
    label: str,
    input_locator: str,
    source_key: str,
    samples: Iterable[tuple[dict, dict[str, tuple[str, bytes]]]],
    expected_rows: int | None,
) -> dict:
    check_pipeline = getattr(args, 'pipeline_check', None)
    if check_pipeline is not None:
        check_pipeline()
    following = getattr(args, 'follow_subtask_path', False)
    if following:
        args.subtask_index = FollowingSubtaskIndex(
            args.subtask_path, source_key, label, sys.modules[__name__],
        )
    paths = {task: args.output/task/'shards'/source_key/f'{label}.jsonl' for task in args.tasks}
    immutable = getattr(args, 'immutable_jsonl', False)
    completed = {}
    for task, path in paths.items():
        if check_pipeline is not None:
            check_pipeline()
        path.parent.mkdir(parents=True, exist_ok=True)
        completed[task] = prior_completed_uids(
            getattr(args, 'resume_output', []), task, source_key, label,
            args.models[task], annotation_provider(args, task),
        )
        locally_durable = pending_final_uids(path)
        completed[task].update(locally_durable)
        if locally_durable:
            print(f'final_outbox_resume task={task} batch={label} '
                  f'compatible_episodes={len(locally_durable)}', flush=True)
        if task == 'ecot' and getattr(args, 'ecot_reuse_sources', None):
            from ecot_prior_results import completed_uids as prior_ecot_uids
            reused = prior_ecot_uids(
                args.ecot_reuse_sources, source_key, label, args.ecot_interval,
                compatible_annotation_uid,
            )
            completed[task].update(reused)
            print(f'ecot_cross_provider_final_resume batch={label} '
                  f'episodes={len(reused)} source_unchanged=true '
                  'partial_checkpoints_imported=false', flush=True)
        if completed[task]:
            print(f'prior_output_resume task={task} batch={label} '
                  f'compatible_episodes={len(completed[task])}', flush=True)
        # Read both layouts regardless of the chosen write mode, including
        # restart/fallback runs that switch back to the legacy writer.
        candidates = [path] + sorted(path.with_suffix('.records').glob('*.jsonl'))
        isolated_validated = 0
        for candidate in candidates:
            if check_pipeline is not None:
                check_pipeline()
            repair_partial_jsonl_tail(candidate)
            validator = getattr(args, '_resume_validation_pool', None)
            if validator is not None and candidate.is_file():
                validated = validator.validate(candidate, task, args.models[task], annotation_provider(args, task),
                                               task == 'grd' and getattr(args.client, 'inventory_reviewer', None) is not None)
                if check_pipeline is not None:
                    check_pipeline()
                if validated is not None:
                    completed[task].update(validated)
                    isolated_validated += 1
                    continue
            if task == 'grd' and candidate.is_file() and getattr(args.client, 'inventory_reviewer', None) is not None:
                from audit_grounding_records import audit_shard
                audit_shard(candidate, candidate, args.client.inventory_reviewer,
                            workers=min(args.max_http_active, args.workers),
                            episode_workers=min(8, args.max_record_active))
            completed[task].update(sanitize_annotation_jsonl(
                candidate, task, args.models[task], annotation_provider(args, task),
            ))
        if isolated_validated:
            print(f'resume_validated_offprocess task={task} batch={label} files={isolated_validated} '
                  f'compatible_episodes={len(completed[task])} source_unchanged=true', flush=True)
    streams = ({task: path.open('a', encoding='utf-8') for task, path in paths.items()}
               if not immutable else {})
    error_label = (f'{label}.{next(iter(args.tasks))}'
                   if getattr(args, 'independent_stage_pipeline', False) else label)
    error_path = args.output/'errors'/'pipeline'/source_key/f'{error_label}.jsonl'
    error_path.parent.mkdir(parents=True, exist_ok=True)
    repair_partial_jsonl_tail(error_path)
    if getattr(args, 'prioritize_failed_records', False):
        samples = prioritize_failed_samples(samples, error_path)
    error_stream = error_path.open('a', encoding='utf-8')
    counts = {task: 0 for task in args.tasks}
    errors = 0
    pending = {}

    # ``--workers`` is also the logical in-batch queue width.  Creating that
    # many native Python threads is wasteful when the provider limiter permits
    # far fewer records to run: every extra thread blocks on record_limiter and
    # reserves a stack without increasing request throughput.  This matters for
    # production runs that intentionally use WORKERS=4096 with a much smaller
    # provider in-flight limit.
    executor_workers = record_executor_workers(args)
    write_lock = threading.Lock()
    record_locks = {}

    def commit_stage(uid, task, record):
        if immutable:
            with write_lock:
                record_lock = record_locks.setdefault((task, uid), threading.Lock())
            with record_lock:
                with write_lock:
                    if uid in completed[task]:
                        return
                destination = paths[task].with_suffix('.records')/(sha256_bytes(uid.encode())+'.jsonl')
                started = time.monotonic()
                if final_record_spool_enabled():
                    put_final_record(destination, record)
                else:
                    atomic_jsonl(destination, record)
                with write_lock:
                    completed[task].add(uid)
                    counts[task] += 1
                    print(f'stage_written uid={uid} task={task} batch={label} '
                          f'written={json_dumps(counts)} persist_seconds={time.monotonic()-started:.3f}', flush=True)
            return
        with write_lock:
            if uid in completed[task]:
                return
            streams[task].write(json.dumps(record, ensure_ascii=False) + '\n')
            streams[task].flush()
            os.fsync(streams[task].fileno())
            completed[task].add(uid)
            counts[task] += 1
            print(f'stage_written uid={uid} task={task} batch={label} '
                  f'written={json_dumps(counts)}', flush=True)

    args.stage_commit = commit_stage
    args.stage_is_complete = lambda uid, task: uid in completed[task]

    def finish(future, uid: str, needed: set[str]) -> bool:
        nonlocal errors
        try:
            produced = future.result()
            for task, record in produced.items():
                commit_stage(uid, task, record)
            # Delete recovery state only after every produced task record has
            # reached durable storage. If the process dies earlier, completed
            # chunks, subtasks, events, frames, and observations remain reusable.
            for task in produced:
                if not getattr(args, 'retain_unit_checkpoints_after_final', False):
                    if task == 'grd':
                        clear_grd_checkpoint(args, uid)
                    else:
                        clear_annotation_checkpoint(args, uid, task)
            return True
        except FatalProviderError:
            raise
        except Exception as error:
            errors += 1
            error_stream.write(json.dumps({
                'source_record_uid': uid, 'input_locator': label,
                'tasks': sorted(needed), 'error': f'{type(error).__name__}:{error}',
            }, ensure_ascii=False) + '\n')
            error_stream.flush()
            print(f'stage_failed uid={uid} tasks={",".join(sorted(needed))} '
                  f'error_type={type(error).__name__}', flush=True)
            return False

    try:
        if getattr(args, 'request_parallel', False):
            las_args = copy.copy(args)
            las_args.record_limiter = None
            def jobs():
                submitted_uids = set()
                for index, (source, blobs) in enumerate(samples, 1):
                    if args.max_records is not None and index > args.max_records:
                        break
                    if not is_temporal(source, args.temporal_media_types):
                        continue
                    uid = str(source.get('uid'))
                    if uid in submitted_uids:
                        continue
                    submitted_uids.add(uid)
                    needed = {task for task in args.tasks if uid not in completed[task]}
                    if needed:
                        yield uid, needed, source, blobs
            def upstream(job):
                uid, needed, source, blobs = job
                if following:
                    args.subtask_index.wait_for(source, args.client.ensure_available)
                    return {}
                return annotate_source_with_retries(source, blobs, input_locator, {'subtask'}, las_args)
            def downstream(job):
                uid, needed, source, blobs = job
                return annotate_source_with_retries(source, blobs, input_locator, needed, args)
            if getattr(args, 'independent_stage_pipeline', False):
                lane = next(iter(args.tasks))
                @contextlib.contextmanager
                def prepare(job):
                    getattr(args, 'pipeline_check', args.client.ensure_available)()
                    media = media_payload(job[2], job[3], preserve_video_file_slices=True)
                    # STA/CPA consume the original subtask clip and never use
                    # the shared 2 FPS ECoT/GRD grid. Avoid transcoding every
                    # episode before contact requests can start.
                    if lane not in {'ecot', 'grd'}:
                        yield None
                        return
                    options = ({'codec_threads': args.video_prepare_codec_threads}
                               if args.video_prepare_codec_threads != 1 else {})
                    sparse_ecot_cache = (args.tasks == {'ecot'} and args.api in {'ark', 'las'}
                                         and args.ecot_video_transport in ('ark-files', 'cos-presigned'))
                    timestamp_las_target = (
                        sparse_ecot_cache and args.api == 'las'
                        and args.ecot_video_transport == 'cos-presigned'
                    )
                    if sparse_ecot_cache:
                        options['frame_cache_write_through'] = False
                        options['frame_cache_memory_mib'] = 256
                        options['teacher_only'] = timestamp_las_target
                    elif lane == 'grd':
                        # Hundreds of resident GRD episodes otherwise reserve
                        # up to 64 MiB each and hit the 48 GiB process address
                        # ceiling while frames are decoded. Spill older JPEGs
                        # to the local cache after a small hot working set.
                        options['frame_cache_memory_mib'] = 16
                    prepare_started = time.monotonic()
                    with EcotMediaFactory(media, **options) as factory:
                        grid_ready = time.monotonic()
                        # Keep the full teacher video, but only materialize the
                        # ECoT targets when no GRD consumer shares this runtime.
                        step = args.ecot_interval if sparse_ecot_cache else 1
                        if not timestamp_las_target:
                            factory.selected_target_images(
                                list(range(0, factory.sampled_frame_count, step))
                            )
                        if sparse_ecot_cache:
                            print('ecot_media_prepared ' + json_dumps({
                                'uid': job[0], 'sampled_frames': factory.sampled_frame_count,
                                'cached_target_frames': len(factory.target_images),
                                'frame_cache_mode': (
                                    'teacher-timestamp-no-target-decode'
                                    if timestamp_las_target else 'selected-writeback'
                                ),
                                'frame_cache_budget_mib': factory.frame_cache_memory_mib,
                                'video_prepare_seconds': round(grid_ready - prepare_started, 3),
                                'frame_cache_seconds': round(time.monotonic() - grid_ready, 3),
                                'completed_at_unix': time.time(),
                            }), flush=True)
                        yield factory
                def prepared_downstream(job, factory):
                    getattr(args, 'pipeline_check', args.client.ensure_available)()
                    config = copy.copy(args)
                    config.record_limiter = None
                    config._shared_prepared_factory = factory
                    return annotate_source_with_retries(
                        job[2], job[3], input_locator, job[1], config,
                    )
                def ready(job):
                    return lane == 'ecot' or args.subtask_index.find(job[2]) is not None
                deferred = run_prepared_jobs(
                    jobs(), args.stage_pipeline, lane, prepare, prepared_downstream,
                    finish, ready, getattr(args, 'pipeline_check', args.client.ensure_available), args.max_pending,
                )
                return {'written': counts, 'errors': errors, 'deferred_upstream': deferred}
            run_stage_jobs(jobs(), upstream, downstream, finish, args.las_request_workers,
                           executor_workers, args.max_pending,
                           needs_upstream=(lambda job: bool(job[1] & {'grd', 'sta', 'cpa'})) if following else None,
                           las_executor=getattr(args, 'las_executor', None))
            return {'written': counts, 'errors': errors}
        with concurrent.futures.ThreadPoolExecutor(max_workers=executor_workers) as executor:
            for index, (source, blobs) in enumerate(samples, 1):
                if args.max_records is not None and index > args.max_records:
                    break
                uid = str(source.get('uid'))
                if not is_temporal(source, args.temporal_media_types):
                    continue
                needed = {task for task in args.tasks if uid not in completed[task]}
                if not needed:
                    continue
                future = executor.submit(
                    annotate_source_with_retries,
                    source, blobs, input_locator, needed, args,
                )
                pending[future] = (uid, needed)
                if len(pending) >= args.max_pending:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for item in done:
                        finish(item, *pending.pop(item))
                if index % 50 == 0:
                    print(
                        f'batch_progress input={label} seen={index}/{expected_rows or "?"} '
                        f'written={json_dumps(counts)} errors={errors} pending={len(pending)}',
                        flush=True,
                    )
            for future in concurrent.futures.as_completed(pending):
                finish(future, *pending[future])
    finally:
        for stream in streams.values():
            stream.close()
        error_stream.close()
    return {'written': counts, 'errors': errors}


def parse_tasks(parser: argparse.ArgumentParser, value: str) -> set[str]:
    tasks = {
        TASK_ALIASES.get(item.strip().lower(), item.strip().lower())
        for item in value.split(',') if item.strip()
    }
    invalid = tasks - set(TASK_ORDER)
    if invalid:
        parser.error(f'unsupported tasks: {sorted(invalid)}')
    if not tasks:
        parser.error('tasks must not be empty')
    return tasks


def annotation_provider(args: argparse.Namespace, task: str) -> str:
    """Subtask always uses LAS; other tasks keep the selected VLM provider."""
    return 'las' if task == 'subtask' else args.api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--runtime-state-dir', type=Path,
                        help='Task-scoped control/status directory; annotation paths remain unchanged.')
    parser.add_argument('--resume-output', type=Path, action='append', default=[],
                        help='Read-only prior native output root; skip compatible completed episodes')
    parser.add_argument('--ecot-reuse-output', nargs=3, action='append', default=[],
                        metavar=('ROOT', 'PROVIDER', 'MODEL'),
                        help='ECoT-only: reuse validated final episodes under their original provider/model; '
                             'source records stay unchanged; does not import partial checkpoints')
    parser.add_argument('--ecot-reuse-partial-checkpoints', action='store_true',
                        help='Also read matching prior ECoT frame units; retain per-frame provider/model provenance')
    parser.add_argument('--prioritize-failed-records', action='store_true',
                        help='Retry prior failures first within each bounded batch without filtering the full scope')
    parser.add_argument('--checkpoint-spool-root', type=Path,
                        help='Opt-in local fsynced checkpoint outbox with background cloud replication')
    parser.add_argument('--durable-cloud-transport', choices=('filesystem', 'cos-direct'),
                        default='filesystem',
                        help='Publish durable JSONL through the mounted filesystem or direct COS PUT')
    parser.add_argument('--checkpoint-sync-workers', type=int, default=32)
    parser.add_argument('--final-publication-workers', type=int, default=8)
    parser.add_argument('--final-publication-spool-root', type=Path,
                        help='Opt-in local fsynced final-record outbox with asynchronous publication')
    parser.add_argument('--final-publication-spool-max-mib', type=int, default=8192)
    parser.add_argument('--final-publication-spool-max-files', type=int, default=50000)
    parser.add_argument('--final-publication-local-batch-size', type=int, default=64)
    parser.add_argument('--checkpoint-local-batch-size', type=int, default=1,
                        help='Local FULL-synchronous group commit size; acknowledgement waits for COMMIT')
    parser.add_argument('--retain-unit-checkpoints-after-final', action='store_true',
                        help='Keep durable per-unit recovery records after the parent final record is durable')
    parser.add_argument('--checkpoint-spool-max-mib', type=int, default=2048)
    parser.add_argument('--checkpoint-spool-max-files', type=int, default=20000)
    parser.add_argument('--tasks', default=','.join(TASK_ORDER))
    parser.add_argument('--ecot-video-transport', choices=('inline', 'dashscope-temporary', 'ark-files', 'cos-presigned'),
                        default='inline', help='Opt-in model-bound, 48-hour DashScope temporary video URLs')
    parser.add_argument('--ecot-image-transport', choices=('inline', 'dashscope-temporary', 'ark-files', 'cos-presigned'),
                        default='inline', help='Send ECoT target images through the same temporary storage method')
    parser.add_argument('--ark-file-cache', type=Path,
                        default=Path(__file__).resolve().parents[1]/'_runtime'/'ark-files')
    parser.add_argument('--ark-upload-workers', type=int, default=16)
    parser.add_argument('--dashscope-upload-policy-qps', type=float, default=5,
                        help='Temporary upload-policy acquisitions per second, maximum 90; does not cap inference')
    parser.add_argument('--dashscope-pooled-uploads', action='store_true',
                        help='Separate policy and upload queues and reuse exclusive HTTP sessions')
    parser.add_argument('--dashscope-isolated-policy', action='store_true',
                        help='Run paced getPolicy HTTP in private child processes, preserving upload bytes and retry limits')
    parser.add_argument('--resume-validation-workers', type=int, default=0,
                        help='Opt-in read-only child workers for fully validated existing output files (0..32)')
    parser.add_argument('--dashscope-server-wait-seconds', type=int, default=0,
                        help='Opt-in server burst queue (0..120 seconds) and request-local transient backoff')
    parser.add_argument('--video-prepare-codec-threads', type=int, default=1,
                        help='Codec threads per shared episode-grid preparation (1..8)')
    parser.add_argument('--dashscope-media-cache', type=Path,
                        default=Path(__file__).resolve().parents[1]/'_runtime'/'dashscope-temporary-media',
                        help='Private persistent local cache; do not use a public directory or object-store mount')
    parser.add_argument('--grd-media-transport', choices=['inline', 'dashscope-temporary'], default='inline',
                        help='Opt-in exact-byte temporary URLs for GRD images and future video clips')
    parser.add_argument('--dashscope-upload-workers', type=int, default=2,
                        help='Bound simultaneous temporary uploads separately from inference requests')
    parser.add_argument('--subtask-path', type=Path)
    parser.add_argument('--follow-subtask-path', action='store_true',
                        help='Wait read-only for native upstream subtask shards, independently of their producer')
    parser.add_argument('--independent-stage-pipeline', action='store_true',
                        help='Separate external-subtask ECoT/GRD batch, preparation and episode queues')
    parser.add_argument('--stage-prefetch', type=int, default=4,
                        help='Additional prepared episodes per independent stage beyond active workers')
    parser.add_argument('--grd-media-prefetch', type=int, default=64,
                        help='Maximum speculative or queued future-video preparations')
    parser.add_argument('--shared-frame-workers', type=int, default=0,
                        help='Process-wide shared ECoT/GRD worker pool; zero retains per-video pools')
    parser.add_argument('--prewarm-shared-frame-workers', type=int, default=0,
                        help='Eagerly start this many shared frame workers before episode scheduling')
    parser.add_argument('--thread-stack-kib', type=int, default=0,
                        help='Native thread stack size; zero uses the platform default')
    parser.add_argument(
        '--temporal-manifest', type=Path,
        help='Exact JSONL shard manifest produced by scan_wds_temporal.py',
    )
    parser.add_argument('--task-instruction')
    parser.add_argument('--source', action='append', default=[])
    parser.add_argument(
        '--record-uid-path', type=Path,
        help=(
            'Optional UTF-8 file containing one input record UID per line. '
            'Filtering happens inside each original input batch so shard labels '
            'remain stable across incremental runs.'
        ),
    )
    parser.add_argument(
        '--temporal-media-types', default='video,frame',
        help='Comma-separated WDS media item types eligible for temporal annotation',
    )
    parser.add_argument('--workers', type=int, default=64)
    parser.add_argument(
        '--grd-workers', type=int, default=1,
        help='Concurrent per-frame GRD requests within each source video',
    )
    parser.add_argument(
        '--ecot-workers', type=int, default=1,
        help='Concurrent selected-frame ECoT requests within each source video',
    )
    parser.add_argument(
        '--ecot-interval', type=int, default=ECOT_INTERVAL,
        help='Select every Nth target from the 2 FPS grid; teacher video is 0.5 FPS',
    )
    parser.add_argument(
        '--grd-checkpoint-root', type=Path,
        help=(
            'Durable per-frame GRD JSONL checkpoints. Each 2 FPS anchor is '
            'fsynced after inventory and first-object selection, loaded on '
            'resume, and removed only after the complete parent GRD record is '
            'fsynced. Defaults to OUTPUT/_state/grd-window-checkpoints'
        ),
    )
    parser.add_argument(
        '--annotation-checkpoint-root', type=Path,
        help=(
            'Durable JSONL checkpoints for subtask chunks, ECoT target frames, '
            'STA subtasks/observation frames, and CPA review/contact units. Defaults to '
            'OUTPUT/_state/annotation-unit-checkpoints'
        ),
    )
    parser.add_argument('--max-pending', type=int)
    parser.add_argument('--request-parallel', action='store_true',
                        help='Schedule LAS separately from bounded resident-video workers')
    parser.add_argument('--model-specific-admission', action='store_true',
                        help='Isolate adaptive HTTP limits and cooldowns by model')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print resolved configuration and exit without model requests')
    parser.add_argument('--las-request-workers', type=int, default=256)
    parser.add_argument('--shared-las-workers', type=int, default=0,
                        help='One LAS episode executor shared across batches; zero retains legacy pools')
    parser.add_argument('--las-transport-control', type=Path,
                        help='Hot-reloaded LAS operator limit and endpoint RPM pacing JSON')
    parser.add_argument('--max-las-operators', type=int, default=64,
                        help='Maximum LAS remote operators sharing the global request slots')
    parser.add_argument('--max-total-http-active', type=int, default=256,
                        help='Shared LAS/downstream/reviewer request admission ceiling')
    parser.add_argument('--request-working-memory-mib', type=int, default=12288)
    parser.add_argument('--request-timeout-seconds', type=float, default=900,
                        help='Per HTTP attempt socket timeout; successful checkpoints remain reusable')
    parser.add_argument('--grd-shared-frame-workers', type=int, default=0,
                        help='Dedicated GRD pool, isolated from the shared ECoT pool; zero shares the pool')
    parser.add_argument('--ecot-request-memory-mib', type=int, default=0,
                        help='Reserve this portion for ECoT and the remainder for GRD; zero shares FIFO')
    parser.add_argument('--video-prepare-workers', type=int, default=4)
    parser.add_argument('--video-clip-workers', type=int, default=4,
                        help='Global pre-request future-video clipping concurrency')
    parser.add_argument('--video-clip-process-pool', action='store_true',
                        help='Use bounded persistent clip workers instead of spawning from the main process')
    parser.add_argument('--immutable-jsonl', action='store_true',
                        help='Atomically save independent JSONL files for result records and checkpoint units')
    parser.add_argument('--durable-write-workers', type=int, default=8,
                        help='Shared immutable JSONL write concurrency')
    parser.add_argument('--batch-workers', type=int, default=1,
                        help='Bounded simultaneous batch contexts sharing all resource gates')
    parser.add_argument(
        '--max-record-active', type=int,
        help=(
            'Maximum complete source videos resident in workers at once; '
            'defaults to --max-http-active'
        ),
    )
    parser.add_argument('--max-attempts', type=int, default=6)
    parser.add_argument('--max-consecutive-request-failures', type=int, default=20)
    parser.add_argument(
        '--record-attempts', type=int, default=3,
        help='End-to-end attempts for a record after request and validation retries',
    )
    parser.add_argument('--max-shards', type=int)
    parser.add_argument('--max-records', type=int)
    parser.add_argument(
        '--physical-batch-size', type=int, default=4096,
        help='Catalog records per resumable batch for a Cosmos3 physical WebDataset',
    )
    parser.add_argument('--api', choices=tuple(DEFAULT_MODELS), default='dashscope')
    parser.add_argument('--endpoint')
    parser.add_argument('--api-key-env')
    parser.add_argument(
        '--las-media-workers', '--vla-media-workers',
        dest='las_media_workers', type=int, default=8,
        help=(
            'Concurrent local video materialization operations for LAS; '
            '--vla-media-workers is a deprecated alias'
        ),
    )
    parser.add_argument(
        '--subtask-model', '--las-model', dest='subtask_model',
        default=os.environ.get('LAS_MODEL', DEFAULT_LAS_MODEL),
        help='LAS Step 3 model; --subtask-model is retained as a CLI alias',
    )
    parser.add_argument(
        '--las-step1-model',
        default=os.environ.get('LAS_STEP1_MODEL', DEFAULT_LAS_STEP1_MODEL),
    )
    parser.add_argument(
        '--las-postprocess-model',
        default=os.environ.get(
            'LAS_POSTPROCESS_MODEL', DEFAULT_LAS_POSTPROCESS_MODEL,
        ),
    )
    las_media = parser.add_mutually_exclusive_group()
    las_media.add_argument(
        '--las-cos-uri-prefix', default=os.environ.get('LAS_COS_URI_PREFIX'),
    )
    las_media.add_argument(
        '--las-video-url-template',
        default=os.environ.get('LAS_VIDEO_URL_TEMPLATE'),
    )
    parser.add_argument(
        '--las-coscli',
        default=os.environ.get('LAS_COSCLI', DEFAULT_LAS_COSCLI),
    )
    parser.add_argument(
        '--las-signed-url-seconds', type=int,
        default=int(os.environ.get(
            'LAS_SIGNED_URL_SECONDS', str(DEFAULT_LAS_SIGNED_URL_SECONDS),
        )),
    )
    parser.add_argument(
        '--las-embodiment', choices=('auto', 'robot', 'human'), default='auto',
    )
    parser.add_argument(
        '--las-use-main-as-wrist-for-robot',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Run LAS Step 1 on the main view for robot videos (default: disabled)',
    )
    parser.add_argument('--grd-model')
    parser.add_argument('--grounding-model', dest='grd_model')
    parser.add_argument('--grd-review-endpoint', default=DEFAULT_ENDPOINTS['ark'])
    parser.add_argument('--grd-review-api-key-env', default='ARK_API_KEY')
    parser.add_argument('--grd-review-model', choices=(GRD_REVIEW_MODEL, 'qwen3.8-max'), default=GRD_REVIEW_MODEL)
    parser.add_argument('--grd-review-api', choices=('ark', 'las', 'dashscope'), default='ark')
    parser.add_argument('--ecot-model')
    parser.add_argument('--sta-model')
    parser.add_argument(
        '--sta-lookback-seconds', type=float, default=STA_LOOKBACK_SECONDS,
        help='Maximum random-observation lookback before each CPA-final contact',
    )
    parser.add_argument(
        '--sta-observations-per-contact', type=int,
        default=STA_OBSERVATIONS_PER_CONTACT,
        help='Random pre-contact images grounded for each CPA-final event',
    )
    parser.add_argument(
        '--sta-min-contact-gap-seconds', type=float,
        default=STA_MIN_CONTACT_GAP_SECONDS,
        help='Minimum time between a random STA observation and its contact',
    )
    parser.add_argument('--sta-random-seed', type=int, default=0)
    parser.add_argument(
        '--sta-bbox-workers', type=int, default=8,
        help='Concurrent single-image STA bbox requests within one source',
    )
    parser.add_argument('--sta-event-workers', type=int, default=1,
                        help='Concurrent subtask contact proposal/review segments per video')
    parser.add_argument('--downstream-stage-workers', type=int, default=1,
                        help='Concurrent independent ECoT, GRD, and STA/CPA stage groups per video')
    parser.add_argument('--cpa-model')
    parser.add_argument('--cpa-min-contact-gap-seconds',type=float,default=0.2,
                        help='Minimum CPA query-to-contact gap in seconds')
    parser.add_argument('--cpa-max-contact-gap-seconds',type=float,default=0.4,
                        help='Maximum CPA query-to-contact gap in seconds')
    parser.add_argument('--contact-frame-model')
    parser.add_argument('--cpa-point-model')
    parser.add_argument('--cpa-point-backend', choices=('las', 'vlm'), default='las',
                        help='Initial contact points: LAS operator defaults, or legacy VLM for comparison')
    parser.add_argument('--cpa-las-cos-prefix', default=os.environ.get('CPA_LAS_COS_PREFIX'),
                        help='COS prefix used to publish CPA contact images for LAS')
    parser.add_argument('--cpa-semantic-review', action=argparse.BooleanOptionalAction,
                        default=True, help='Audit final SAM points with thinking-enabled Ark Seed 2.1 Turbo')
    parser.add_argument('--cpa-student-questions', action=argparse.BooleanOptionalAction,
                        default=True, help='Run native CoTracker, original choice policy and 5 FPS learner records')
    parser.add_argument('--cpa-student-history-seconds', type=float, default=2.0,
                        help='History duration before the unchanged final query frame')
    parser.add_argument('--cpa-question-mode', choices=('coordinates','multiple_choice'), default='coordinates',
                        help='Predict normalized final-frame coordinates with three significant digits')
    parser.add_argument('--cpa-robot-agent-points', action=argparse.BooleanOptionalAction, default=False,
                        help='Include robot gripper contact points along with object points')
    parser.add_argument('--cpa-crop-padding', type=float, default=0.15)
    parser.add_argument('--cpa-crop-size', type=int, default=1024)
    parser.add_argument('--sam3-repo', type=Path, default=DEFAULT_SAM3_REPO)
    parser.add_argument(
        '--sam3-checkpoint', type=Path,
        default=DEFAULT_SAM3_CHECKPOINT,
    )
    parser.add_argument('--sam3-device', default='cuda')
    parser.add_argument('--sam3-confidence-threshold', type=float, default=0.25)
    parser.add_argument('--fatal-stop-file', type=Path)
    parser.add_argument('--rate-state-file', type=Path)
    parser.add_argument('--request-warm-start-interval', type=float,
                        help='One-time Ark ECoT pacing override after a confirmed quota change')
    parser.add_argument('--rate-utilization', type=float, default=0.8)
    parser.add_argument(
        '--request-start-interval', type=float, default=0.0,
        help='Minimum seconds between shared HTTP request starts',
    )
    parser.add_argument('--max-http-active', type=int, default=64)
    parser.add_argument('--fixed-http-concurrency', action='store_true',
                        help='Use the configured HTTP cap without restoring/reducing adaptive caps; retain 429 cooldowns and retries')
    parser.add_argument('--fair-http-control', type=Path,
                        help='Local JSON control file enabling fair, hot-adjustable per-attempt HTTP admission')
    parser.add_argument('--fair-http-capacity', type=int, choices=(2048, 4096, 8192), default=2048,
                        help='Opt-in maximum hot-control capacity for native Ark Files ECoT')
    parser.add_argument('--ecot-memory-request-slots', type=int, default=2048,
                        help='Request-count ceiling independent of weighted memory and active HTTP')
    parser.add_argument('--error-burst-threshold', type=int, default=0,
                        help='Failure events within the window before sticky conservative admission; 0 disables')
    parser.add_argument('--error-burst-window-seconds', type=float, default=60)
    parser.add_argument('--conservative-http-limit', type=int, default=64)
    parser.add_argument('--conservative-record-limit', type=int, default=32)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.fair_http_capacity <= args.ecot_memory_request_slots <= 16384:
        parser.error('ECoT memory request slots must cover fair HTTP capacity and not exceed 16384')
    if (args.fair_http_capacity != 2048 or args.ecot_memory_request_slots != 2048) and not (
            args.api in {'ark', 'las'} and args.tasks == 'ecot' and args.fair_http_control
            and args.ecot_video_transport in ('ark-files', 'cos-presigned')
            and args.ecot_image_transport in ('ark-files', 'cos-presigned')):
        parser.error('Expanded request slots require Volc URL-media ECoT and fair HTTP control')
    if args.grd_media_transport == 'dashscope-temporary' and (
            args.api != 'dashscope' or args.grd_review_api != 'dashscope'):
        parser.error('temporary GRD media requires DashScope for both grounding and inventory review')
    if args.fair_http_control is not None and (
            (args.api != 'dashscope' and not (args.api in {'ark', 'las'} and args.tasks in {'sta', 'ecot'}))
            or not args.fixed_http_concurrency or not args.request_parallel
            or not args.ecot_request_memory_mib or not args.fair_http_control.is_absolute()):
        parser.error('fair HTTP control requires DashScope or Volc STA/ECoT, fixed concurrency, partitioned request memory and an absolute local control path')
    if not 1 <= args.final_publication_workers <= 256:
        parser.error('--final-publication-workers must be between 1 and 256')
    if min(args.final_publication_spool_max_mib, args.final_publication_spool_max_files,
           args.final_publication_local_batch_size) < 1:
        parser.error('final publication spool limits must be positive')
    if min(args.checkpoint_sync_workers, args.checkpoint_spool_max_mib,
           args.checkpoint_spool_max_files, args.checkpoint_local_batch_size) < 1:
        parser.error('checkpoint spool limits must be positive')
    if args.checkpoint_spool_root is not None and not args.immutable_jsonl:
        parser.error('--checkpoint-spool-root requires --immutable-jsonl')
    if args.final_publication_spool_root is not None and not args.immutable_jsonl:
        parser.error('--final-publication-spool-root requires --immutable-jsonl')
    if args.checkpoint_local_batch_size != 1 and args.checkpoint_spool_root is None:
        parser.error('--checkpoint-local-batch-size requires --checkpoint-spool-root')
    if args.durable_cloud_transport == 'cos-direct' and not (
            args.api in {'ark', 'las'} and args.tasks == 'ecot'
            and args.checkpoint_spool_root is not None
            and args.final_publication_spool_root is not None
            and (args.dry_run or (
                os.environ.get('VQA_COS_MOUNT_ROOT')
                and args.output.absolute().is_relative_to(
                    Path(os.environ['VQA_COS_MOUNT_ROOT']).absolute()
                )
            ))):
        parser.error('direct COS durable output requires Volc ECoT, both local outboxes, and the writable COS mount')
    if ('ark-files' in (args.ecot_video_transport, args.ecot_image_transport)
            or 'cos-presigned' in (args.ecot_video_transport, args.ecot_image_transport)) and not (
            args.api in {'ark', 'las'} and args.tasks == 'ecot'
            and args.ecot_video_transport in ('ark-files', 'cos-presigned')
            and args.ecot_image_transport in ('ark-files', 'cos-presigned')):
        parser.error('Volc ECoT URL media requires Files or COS-presigned video and image inputs')
    if not 1 <= args.ark_upload_workers <= 512:
        parser.error('--ark-upload-workers must be between 1 and 512')
    if args.ecot_video_transport == 'dashscope-temporary' and args.api != 'dashscope':
        parser.error('--ecot-video-transport dashscope-temporary requires --api dashscope')
    if args.ecot_image_transport == 'dashscope-temporary' and args.ecot_video_transport != 'dashscope-temporary':
        parser.error('--ecot-image-transport dashscope-temporary requires --ecot-video-transport dashscope-temporary')
    if not 0 < args.dashscope_upload_policy_qps <= 90:
        parser.error('--dashscope-upload-policy-qps must be greater than 0 and at most 90')
    if args.dashscope_isolated_policy and (args.api != 'dashscope' or not args.dashscope_pooled_uploads):
        parser.error('--dashscope-isolated-policy requires DashScope and --dashscope-pooled-uploads')
    if not 0 <= args.resume_validation_workers <= 32:
        parser.error('--resume-validation-workers must be between 0 and 32')
    if args.resume_validation_workers and (
            (args.api != 'dashscope' and not (
                args.api in {'ark', 'las'} and args.tasks in {'ecot', 'grd', 'sta'}
            ))
            or not args.immutable_jsonl):
        parser.error('--resume-validation-workers requires DashScope or single-task Volc ECoT/GRD/STA and --immutable-jsonl')
    if not 0 <= args.dashscope_server_wait_seconds <= 120:
        parser.error('--dashscope-server-wait-seconds must be between 0 and 120')
    if not 1 <= args.video_prepare_codec_threads <= 8:
        parser.error('--video-prepare-codec-threads must be between 1 and 8')
    if args.dashscope_server_wait_seconds and args.api != 'dashscope':
        parser.error('--dashscope-server-wait-seconds requires DashScope')
    if args.dashscope_upload_workers < 1:
        parser.error('--dashscope-upload-workers must be positive')
    if args.ecot_request_memory_mib < 0 or args.ecot_request_memory_mib >= args.request_working_memory_mib:
        parser.error('ecot-request-memory-mib must be nonnegative and below request-working-memory-mib')
    if args.ecot_request_memory_mib and (
        not args.request_parallel or (set(args.tasks.split(',')) - {'ecot', 'grd', 'grounding'}
                                     and not (args.api in {'ark', 'las'} and args.tasks == 'sta' and args.fair_http_control))
    ):
        parser.error('ECoT/GRD memory partitions require request-parallel and only ecot,grd tasks')
    if (args.shared_frame_workers < 0 or args.prewarm_shared_frame_workers < 0
            or args.prewarm_shared_frame_workers > args.shared_frame_workers
            or (args.thread_stack_kib and args.thread_stack_kib < 256)):
        parser.error('shared/prewarm frame workers must be valid; explicit thread-stack-kib must be >=256')
    if args.follow_subtask_path and (args.subtask_path is None or not args.request_parallel):
        parser.error('--follow-subtask-path requires --subtask-path and --request-parallel')
    from grd_inventory_review import REVIEW_PROVIDERS
    expected_review_api = REVIEW_PROVIDERS[args.grd_review_model]
    if expected_review_api != args.grd_review_api and not (
            expected_review_api == 'ark' and args.grd_review_api == 'las'):
        parser.error('grd-review-model and grd-review-api must match')
    args.tasks = parse_tasks(parser, args.tasks)
    # Existing saved fair-mode launch commands opt in without needing a new flag.
    if (args.fair_http_control is not None and args.follow_subtask_path
            and args.request_parallel and args.immutable_jsonl
            and args.tasks <= {'ecot', 'grd'}):
        args.independent_stage_pipeline = True
    if args.stage_prefetch < 1 or args.grd_media_prefetch < 1 or (
        args.independent_stage_pipeline and args.grd_media_prefetch < args.video_clip_workers
    ):
        parser.error('invalid_stage_or_media_prefetch_limit')
    if args.independent_stage_pipeline and (
        not args.request_parallel or not args.follow_subtask_path
        or not args.immutable_jsonl or not args.tasks <= {'ecot', 'grd'}
    ):
        parser.error('independent_stage_pipeline_requires_external_subtasks_immutable_ecot_grd')
    args.temporal_media_types = {
        item.strip().lower()
        for item in args.temporal_media_types.split(',') if item.strip()
    }
    if not args.temporal_media_types:
        parser.error('--temporal-media-types must not be empty')
    subtask_conditioned = args.tasks & {'grd', 'sta', 'cpa'}
    if not math.isfinite(args.cpa_student_history_seconds) or args.cpa_student_history_seconds <= 0:
        parser.error('--cpa-student-history-seconds must be finite and positive')
    if subtask_conditioned and 'subtask' not in args.tasks and args.subtask_path is None:
        parser.error('--subtask-path is required when --tasks omits subtask')
    if args.subtask_path is not None and not args.subtask_path.exists():
        parser.error(f'--subtask-path does not exist: {args.subtask_path}')
    if args.temporal_manifest is not None and not args.temporal_manifest.is_file():
        parser.error(f'--temporal-manifest does not exist: {args.temporal_manifest}')
    if args.record_uid_path is not None and not args.record_uid_path.is_file():
        parser.error(f'--record-uid-path does not exist: {args.record_uid_path}')
    if (
        args.workers < 1 or args.grd_workers < 1 or args.ecot_workers < 1
        or args.ecot_interval < 1 or args.sta_bbox_workers < 1 or args.sta_event_workers < 1
        or args.downstream_stage_workers < 1
        or args.request_timeout_seconds <= 0 or args.grd_shared_frame_workers < 0
        or args.max_attempts < 1 or args.record_attempts < 1
        or args.max_consecutive_request_failures < 1
        or args.max_http_active < 1 or args.physical_batch_size < 1
        or args.las_media_workers < 1
    ):
        parser.error(
            'workers, grd-workers, ecot-workers, ecot-interval, '
            'sta-bbox-workers, sta-event-workers, downstream-stage-workers, max-attempts, '
            'record-attempts, max-http-active, physical-batch-size, and '
            'las-media-workers must be positive'
        )
    if args.sta_lookback_seconds <= 0:
        parser.error('--sta-lookback-seconds must be positive')
    if not 0 < args.cpa_min_contact_gap_seconds < args.cpa_max_contact_gap_seconds:
        parser.error('CPA contact gaps must satisfy 0 < minimum < maximum')
    if not 0 <= args.shared_las_workers <= 4096:
        parser.error('shared_las_workers_out_of_range')
    if args.las_transport_control and (not args.request_parallel or args.tasks != {'subtask'}):
        parser.error('las_transport_control_requires_subtask_only_request_parallel')
    if min(args.las_request_workers, args.max_las_operators, args.max_total_http_active,
           args.request_working_memory_mib, args.video_prepare_workers,
           args.video_clip_workers, args.batch_workers, args.durable_write_workers) < 1:
        parser.error('request parallelism and memory limits must be positive')
    if args.sta_observations_per_contact < 1:
        parser.error('--sta-observations-per-contact must be positive')
    if not 0 < args.sta_min_contact_gap_seconds < args.sta_lookback_seconds:
        parser.error(
            '--sta-min-contact-gap-seconds must be positive and smaller than '
            '--sta-lookback-seconds'
        )
    args.max_pending = args.max_pending or max(args.workers * 2, 8)
    if args.max_record_active is None:
        args.max_record_active = args.max_http_active
    if args.max_record_active < 1:
        parser.error('max-record-active must be positive')
    if args.max_pending < args.workers:
        parser.error('max-pending must be at least workers')
    if not 0 < args.rate_utilization <= 1:
        parser.error('rate-utilization must be in (0, 1]')
    if args.request_start_interval < 0:
        parser.error('--request-start-interval must be non-negative')
    if args.request_warm_start_interval is not None and (
            not math.isfinite(args.request_warm_start_interval)
            or not 0 <= args.request_warm_start_interval <= 2
            or args.api not in {'ark', 'las'} or args.tasks != {'ecot'}
            or not args.fixed_http_concurrency or not args.fair_http_control):
        parser.error('Pacing warm start requires fixed-concurrency fair Volc ECoT and an interval in [0, 2]')
    if args.las_signed_url_seconds < 600:
        parser.error('--las-signed-url-seconds must be at least 600')
    args.endpoint = args.endpoint or DEFAULT_ENDPOINTS.get(args.api)
    if not args.endpoint:
        parser.error('--endpoint is required for openai-compatible')
    args.api_key_env = args.api_key_env or (
        'DASHSCOPE_API_KEY' if args.api in {'dashscope', 'vla-anno'} else
        'ARK_API_KEY' if args.api == 'ark' else
        'LAS_API_KEY' if args.api == 'las' else 'API_KEY'
    )
    defaults = DEFAULT_MODELS[args.api]
    args.models = {
        task: getattr(args, f'{task}_model') or defaults[task] for task in TASK_ORDER
    }
    args.models['subtask'] = args.subtask_model
    args.models['contact_frame'] = args.contact_frame_model or defaults['contact_frame']
    args.models['cpa_point'] = args.cpa_point_model or defaults['cpa_point']
    if args.cpa_point_backend == 'las':
        args.models['cpa_point'] = 'las_operator_default'
    for task in args.tasks:
        if not args.models[task]:
            parser.error(f'--{task}-model is required for {args.api}')
    if args.api == 'vla-anno' and args.tasks - {'subtask'}:
        parser.error(
            '--api vla-anno cannot run ECoT/GRD/STA/CPA; subtask now always uses LAS'
        )
    if 'cpa' in args.tasks and not args.models['sta']:
        parser.error('--sta-model is required because CPA uses STA as its proposal stage')
    if 'sta' in args.tasks and not args.models['cpa']:
        parser.error('--cpa-model is required because STA uses CPA-reviewed contact frames')
    if {'sta', 'cpa'} & args.tasks and not args.models['contact_frame']:
        parser.error(
            '--contact-frame-model is required because STA and CPA use '
            'CPA-reviewed exact contact frames'
        )
    if 'cpa' in args.tasks and not args.models['cpa_point']:
        parser.error('--cpa-point-model is required for CPA crop point selection')
    if not 0 <= args.cpa_crop_padding <= 1:
        parser.error('--cpa-crop-padding must be in [0, 1]')
    if args.cpa_crop_size < 64:
        parser.error('--cpa-crop-size must be at least 64')
    if not 0 < args.sam3_confidence_threshold <= 1:
        parser.error('--sam3-confidence-threshold must be in (0, 1]')
    args.output = args.output.resolve()
    from task_process_state import state_directory
    if args.runtime_state_dir is not None:
        if len(args.tasks) != 1 or next(iter(args.tasks)) not in {'ecot', 'grd', 'sta'}:
            parser.error('runtime-state-dir requires exactly one downstream task')
        expected_state = state_directory(args.output, next(iter(args.tasks)))
        if args.runtime_state_dir.resolve() != expected_state:
            parser.error('runtime-state-dir must be OUTPUT/_state/processes/TASK')
        args.runtime_state_dir = expected_state
    else:
        args.runtime_state_dir = args.output/'_state'
    args.resume_output = [root.resolve() for root in args.resume_output]
    from ecot_prior_results import parse_sources as parse_ecot_reuse_sources
    try:
        args.ecot_reuse_sources = parse_ecot_reuse_sources(
            args.ecot_reuse_output, args.output, args.tasks,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.ecot_reuse_partial_checkpoints and not args.ecot_reuse_sources:
        parser.error('--ecot-reuse-partial-checkpoints requires --ecot-reuse-output')
    for root in args.resume_output:
        if not root.is_dir() or root == args.output:
            parser.error('--resume-output must be an existing directory distinct from --output')
    args.grd_checkpoint_root = (
        args.grd_checkpoint_root.resolve()
        if args.grd_checkpoint_root is not None
        else args.output/'_state'/'grd-window-checkpoints'
    )
    args.annotation_checkpoint_root = (
        args.annotation_checkpoint_root.resolve()
        if args.annotation_checkpoint_root is not None
        else args.output/'_state'/'annotation-unit-checkpoints'
    )
    for source in args.ecot_reuse_sources:
        if (args.annotation_checkpoint_root.is_relative_to(source.root)
                or source.root.is_relative_to(args.annotation_checkpoint_root)):
            parser.error('ECoT destination checkpoints must not overlap a read-only source output')
    args.fatal_stop_file = args.fatal_stop_file or args.runtime_state_dir/'provider-fatal-stop.json'
    args.rate_state_file = args.rate_state_file or args.runtime_state_dir/'rate-state.json'
    if args.record_uid_path is None:
        args.record_uid_filter = None
    else:
        args.record_uid_filter = {
            line.strip()
            for line in args.record_uid_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        }
        if not args.record_uid_filter:
            parser.error('--record-uid-path contains no record UIDs')
    return args


def filter_batch_samples(
    samples: Iterable[tuple[dict, dict[str, tuple[str, bytes]]]],
    record_uids: set[str] | None,
) -> tuple[Iterable[tuple[dict, dict[str, tuple[str, bytes]]]], int | None]:
    """Filter within an existing batch without changing its stable label."""
    if record_uids is None:
        return samples, len(samples) if isinstance(samples, list) else None
    if isinstance(samples, list):
        selected = [
            item for item in samples if str(item[0].get('uid')) in record_uids
        ]
        return selected, len(selected)
    return (
        (item for item in samples if str(item[0].get('uid')) in record_uids),
        None,
    )


def input_batches(args: argparse.Namespace):
    source_filter = set(args.source)
    if args.input.is_file() and args.input.suffix.lower() in VIDEO_SUFFIXES:
        yield args.input.stem, str(args.input.resolve()), 'direct_video', [direct_video_sample(args.input)], 1
        return
    if args.input.is_file() and args.input.suffix.lower() == '.jsonl':
        yield from iter_video_manifest_batches(
            args.input, source_filter, args.physical_batch_size, args.max_shards,
        )
        return
    physical_root = cosmos3_physical_root(args.input) if args.input.is_dir() else None
    if physical_root is not None:
        print(
            f'cosmos3_physical_input logical_or_physical={args.input.resolve()} '
            f'physical_root={physical_root} batch_size={args.physical_batch_size}',
            flush=True,
        )
        yield from iter_cosmos3_physical_batches(
            physical_root, source_filter, args.physical_batch_size, args.max_shards,
            record_uids=getattr(args, 'record_uid_filter', None),
        )
        return
    if args.input.is_dir() and (args.input/'TAR_INDEX.jsonl').is_file():
        rows = [json.loads(line) for line in (args.input/'TAR_INDEX.jsonl').open()]
        temporal_manifest = None
        if args.temporal_manifest is not None:
            manifest_rows = [
                json.loads(line)
                for line in args.temporal_manifest.open()
                if line.strip()
            ]
            temporal_manifest = {str(row['tar']): row for row in manifest_rows}
            rows = [
                row for row in rows
                if str(row['tar']) in temporal_manifest
                and int(temporal_manifest[str(row['tar'])]['target_rows']) > 0
            ]
        if source_filter:
            rows = [row for row in rows if row['source_key'] in source_filter]
        if args.max_shards is not None:
            rows = rows[:args.max_shards]
        for row in rows:
            tar_path = args.input/row['tar']
            expected_rows = (
                int(temporal_manifest[str(row['tar'])]['target_rows'])
                if temporal_manifest is not None else row.get('rows')
            )
            yield (
                tar_path.stem, str(tar_path.resolve()), str(row['source_key']),
                iter_wds_samples(tar_path, args.temporal_media_types), expected_rows,
            )
        return
    if args.input.is_dir():
        videos = sorted(path for path in args.input.rglob('*') if path.suffix.lower() in VIDEO_SUFFIXES)
        if args.max_shards is not None:
            videos = videos[:args.max_shards]
        for path in videos:
            yield path.stem, str(path.resolve()), 'direct_video', [direct_video_sample(path)], 1
        return
    raise ValueError(f'unsupported_input:{args.input}')


def main(argv: list[str] | None = None) -> None:
    global VIDEO_CLIP_LIMITER
    args = parse_args(argv)
    if not args.dry_run:
        from task_process_state import check_handoff_ownership
        check_handoff_ownership(args.tasks)
    if args.thread_stack_kib:
        threading.stack_size(args.thread_stack_kib * 1024)
    if args.dry_run:
        print(json_dumps({
            'input': str(args.input), 'output': str(args.output), 'models': args.models,
            'tasks': sorted(args.tasks), 'request_parallel': args.request_parallel,
            'model_specific_admission': args.model_specific_admission,
            'fixed_http_concurrency': args.fixed_http_concurrency,
            'fair_http_control': str(args.fair_http_control) if args.fair_http_control else None,
            'shared_frame_workers': args.shared_frame_workers,
            'prewarm_shared_frame_workers': args.prewarm_shared_frame_workers,
            'grd_shared_frame_workers': args.grd_shared_frame_workers,
            'request_timeout_seconds': args.request_timeout_seconds,
            'ecot_video_transport': args.ecot_video_transport,
            'ecot_image_transport': args.ecot_image_transport,
            'grd_media_transport': args.grd_media_transport,
            'dashscope_upload_policy_qps': args.dashscope_upload_policy_qps,
            'dashscope_media_post_timeout_seconds': list(UPLOAD_TIMEOUT),
            'ecot_completion_normalization': 'explicit-final-negative/v1',
            'dashscope_pooled_uploads': args.dashscope_pooled_uploads,
            'dashscope_isolated_policy': args.dashscope_isolated_policy,
            'resume_validation_workers': args.resume_validation_workers,
            'video_prepare_codec_threads': args.video_prepare_codec_threads,
            'dashscope_server_wait_seconds': args.dashscope_server_wait_seconds,
            'checkpoint_spool_root': str(args.checkpoint_spool_root) if args.checkpoint_spool_root else None,
            'durable_cloud_transport': args.durable_cloud_transport,
            'ecot_reuse_sources': [source.as_dict() for source in args.ecot_reuse_sources],
            'ecot_reuse_partial_checkpoints': args.ecot_reuse_partial_checkpoints,
            'checkpoint_sync_workers': args.checkpoint_sync_workers,
            'final_publication_workers': args.final_publication_workers,
            'final_publication_spool_root': (str(args.final_publication_spool_root)
                                             if args.final_publication_spool_root else None),
            'final_publication_spool_max_mib': args.final_publication_spool_max_mib,
            'final_publication_spool_max_files': args.final_publication_spool_max_files,
            'final_publication_local_batch_size': args.final_publication_local_batch_size,
            'checkpoint_local_batch_size': args.checkpoint_local_batch_size,
            'dashscope_upload_workers': args.dashscope_upload_workers,
            'follow_subtask_path': args.follow_subtask_path,
            'independent_stage_pipeline': args.independent_stage_pipeline,
            'stage_prefetch': args.stage_prefetch,
            'grd_media_prefetch': args.grd_media_prefetch,
            'grd_review_model': args.grd_review_model, 'grd_review_api': args.grd_review_api,
            'total_request_limit': args.max_total_http_active,
            'las_request_workers': args.las_request_workers,
            'max_las_operators': args.max_las_operators,
            'resident_video_limit': args.max_record_active,
            'frame_workers': {'ecot': args.ecot_workers, 'grd': args.grd_workers, 'sta': args.sta_bbox_workers},
            'sta_event_workers': args.sta_event_workers,
            'downstream_stage_workers': args.downstream_stage_workers,
            'las_media_workers': args.las_media_workers, 'video_prepare_workers': args.video_prepare_workers,
            'video_clip_workers': args.video_clip_workers, 'batch_workers': args.batch_workers,
            'video_clip_process_pool': args.video_clip_process_pool,
            'video_clip_codec_threads': int(os.environ.get('VQA_VIDEO_CLIP_CODEC_THREADS', '1')),
            'prioritize_failed_records': args.prioritize_failed_records,
            'immutable_jsonl': args.immutable_jsonl, 'durable_write_workers': args.durable_write_workers,
            'request_working_memory_mib': args.request_working_memory_mib,
            'ecot_request_memory_mib': args.ecot_request_memory_mib,
            'sticky_failure_backoff_enabled': bool(args.error_burst_threshold),
            'physical_batch_size': args.physical_batch_size,
            'max_records': args.max_records, 'max_shards': args.max_shards,
            'uid_filter_count': len(args.record_uid_filter) if args.record_uid_filter is not None else None,
            'rate_state_file': str(args.rate_state_file),
        }))
        return
    cv2.setNumThreads(1)
    VIDEO_CLIP_LIMITER = threading.BoundedSemaphore(args.video_clip_workers)
    import durable_jsonl
    durable_jsonl.IO_LIMITER = threading.BoundedSemaphore(args.durable_write_workers)
    if args.error_burst_threshold < 0 or args.error_burst_window_seconds <= 0 or min(args.conservative_http_limit, args.conservative_record_limit) < 1:
        raise SystemExit('invalid_failure_burst_configuration')
    args.burst_backoff = (BurstFailureBackoff(
        args.error_burst_threshold, args.error_burst_window_seconds,
        args.runtime_state_dir/'failure-burst-state.json',
        args.conservative_http_limit, args.conservative_record_limit,
    ) if args.error_burst_threshold else None)
    if 'cpa' in args.tasks:
        if not args.sam3_repo.is_dir():
            raise SystemExit(f'sam3_repo_missing:{args.sam3_repo}')
        if not args.sam3_checkpoint.is_file():
            raise SystemExit(
                f'sam3_checkpoint_missing:{args.sam3_checkpoint}:request access to '
                'https://huggingface.co/facebook/sam3 and pass --sam3-checkpoint'
            )
    api_tasks = args.tasks - {'subtask'}
    args.las_transport = None
    args.request_admission = (RequestAdmission(args.max_total_http_active,
                             args.request_working_memory_mib * 1024**2)
                              if args.request_parallel else None)
    if args.ecot_request_memory_mib:
        args.request_admission = PartitionedRequestAdmission(
            args.max_total_http_active, args.request_working_memory_mib * 1024**2,
            args.ecot_request_memory_mib * 1024**2,
        )
    if args.request_admission is not None:
        if args.fair_http_control is not None:
            from fair_request_admission import FairRequestAdmission, CachedAvailabilityCheck
            args.request_admission = FairRequestAdmission(
                args.max_total_http_active, args.request_working_memory_mib * 1024**2,
                args.ecot_request_memory_mib * 1024**2, args.fair_http_control, args.output,
                control_maximum=args.fair_http_capacity,
                memory_request_slots=args.ecot_memory_request_slots,
                network_backend=('auto' if args.api in {'ark', 'las'} and args.tasks == {'ecot'}
                                 and args.ecot_video_transport in ('ark-files', 'cos-presigned')
                                 and args.independent_stage_pipeline else 'python'))
        def check_request_provider():
            if (args.fatal_stop_file.exists()
                    or (args.runtime_state_dir/'grd-review-fatal-stop.json').exists()
                    or ('cpa' in args.tasks and (args.runtime_state_dir/'cpa-review-fatal-stop.json').exists())):
                raise FatalProviderError('request_parallel_provider_fatal_marker_present')
            try:
                durable_jsonl.check_checkpoint_storage_health()
            except RuntimeError as error:
                raise FatalProviderError(str(error)) from error
        if args.fair_http_control or args.las_transport_control or 'cpa' in args.tasks:
            from fair_request_admission import CachedAvailabilityCheck
            native_ecot_checks = (args.api in {'ark', 'las'} and args.tasks == {'ecot'}
                                  and args.ecot_video_transport in ('ark-files', 'cos-presigned')
                                  and args.independent_stage_pipeline)
            args.request_admission.check_available = CachedAvailabilityCheck(
                check_request_provider, notification_backend='auto' if native_ecot_checks else 'private-locks')
            args.request_admission.availability_checker = args.request_admission.check_available
        else:
            args.request_admission.check_available = check_request_provider
    args.video_prepare_limiter = (threading.BoundedSemaphore(args.video_prepare_workers)
                                  if args.request_parallel else None)
    args.client = ApiClient(args) if api_tasks else None
    if 'cpa' in args.tasks and args.cpa_point_backend == 'las':
        from cpa_las_points import LasContactPointSelector
        if not args.cpa_las_cos_prefix:
            raise SystemExit('missing_required_configuration:CPA_LAS_COS_PREFIX')
        args.client.cpa_las_selector = LasContactPointSelector(
            args.output/'_state'/'las-contact-grounding',
            cos_prefix=args.cpa_las_cos_prefix, coscli=args.las_coscli,
            check_available=args.client.ensure_available,
            request_admission=args.request_admission,
        )
    if 'cpa' in args.tasks and (args.cpa_semantic_review or args.cpa_student_questions):
        cpa_review_args = copy.copy(args)
        cpa_review_args.api = 'las' if args.api == 'las' else 'ark'
        cpa_review_args.endpoint = DEFAULT_ENDPOINTS[cpa_review_args.api]
        cpa_review_args.api_key_env = 'LAS_API_KEY' if cpa_review_args.api == 'las' else 'ARK_API_KEY'
        cpa_review_args.ecot_video_transport = 'inline'
        cpa_review_args.grd_media_transport = 'inline'
        cpa_review_args.fatal_stop_file = args.runtime_state_dir/'cpa-review-fatal-stop.json'
        cpa_review_args.rate_state_file = args.runtime_state_dir/'cpa-review-rate-state.json'
        args.cpa_downstream_client = ApiClient(cpa_review_args)
        args.cpa_downstream_client.ensure_available()
        if args.cpa_semantic_review:
            args.client.cpa_semantic_reviewer = args.cpa_downstream_client
    if 'grd' in args.tasks:
        review_args = copy.copy(args)
        review_args.api = args.grd_review_api
        review_args.endpoint = args.grd_review_endpoint
        review_args.api_key_env = args.grd_review_api_key_env
        review_args.fatal_stop_file = args.runtime_state_dir/'grd-review-fatal-stop.json'
        review_args.rate_state_file = args.runtime_state_dir/'grd-review-rate-state.json'
        if args.request_parallel:
            review_args.rate_state_file = args.runtime_state_dir/'request256-grd-review-rate-state.json'
        review_client = ApiClient(review_args)
        review_client.ensure_available()
        args.client.inventory_reviewer = InventoryReviewer(
            review_client, args.output/'_state'/'grd-inventory-reviews', model=args.grd_review_model,
        )
    if 'subtask' in args.tasks:
        if not os.environ.get('LAS_API_KEY', '').strip():
            raise SystemExit('missing_api_key_environment_variable:LAS_API_KEY')
        if not os.environ.get('ARK_API_KEY', '').strip():
            raise SystemExit('missing_api_key_environment_variable:ARK_API_KEY')
        cos_uri_prefix = (
            None if args.las_video_url_template
            else args.las_cos_uri_prefix or DEFAULT_LAS_COS_URI_PREFIX
        )
        args.las_subtask_annotator = LASSubtaskAnnotator(LASSubtaskConfig(
            output_root=args.output,
            cos_uri_prefix=cos_uri_prefix,
            video_url_template=args.las_video_url_template,
            coscli=args.las_coscli,
            signed_url_seconds=args.las_signed_url_seconds,
            model=args.models['subtask'],
            step1_model=args.las_step1_model,
            postprocess_model=args.las_postprocess_model,
            use_main_as_wrist_for_robot=args.las_use_main_as_wrist_for_robot,
        ))
        args.las_media_limiter = threading.BoundedSemaphore(
            args.las_media_workers,
        )
        if args.request_parallel:
            from request_parallel import configure_las_admission
            if args.las_transport_control:
                from las_transport import LasTransport
                args.las_transport = LasTransport(args.request_admission, args.las_transport_control)
            configure_las_admission(args.las_subtask_annotator, args.request_admission,
                                    args.output/'_state'/'las-request-tasks', args.max_las_operators,
                                    transport=args.las_transport)
    else:
        args.las_subtask_annotator = None
        args.las_media_limiter = None
    # WORKERS may intentionally be much larger than the provider concurrency.
    # Bound full-record media materialization so queued physical videos stay lazy.
    args.record_limiter = AdaptiveConcurrencyLimiter(args.max_record_active)
    if args.burst_backoff is not None:
        args.burst_backoff.register(args.record_limiter, 'record')
    args.sam3_snapper = (
        Sam3Snapper(
            args.sam3_repo, args.sam3_checkpoint, args.sam3_device,
            args.sam3_confidence_threshold,
        )
        if 'cpa' in args.tasks else None
    )
    if args.follow_subtask_path or (not (args.tasks & {'grd', 'sta', 'cpa'}) and args.subtask_path is None):
        args.subtask_index = SubtaskIndex()
    else:
        load_paths = (
            [args.subtask_path]
            if args.subtask_path is not None else [args.output/'subtask']
        )
        args.subtask_index = load_subtask_index(load_paths)
    if args.client is not None:
        args.client.ensure_available()
    totals = {'batches': 0, 'errors': 0, 'written': {task: 0 for task in args.tasks}}
    runtime_config = {
        'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
        'input': str(args.input), 'output': str(args.output),
        'tasks': [task for task in TASK_ORDER if task in args.tasks],
        'models': args.models, 'subtask_backend': 'las',
        'http_limit': (args.client.http_limiter.limit if args.client is not None
                       and not args.model_specific_admission else None),
        'model_specific_admission': args.model_specific_admission,
        'fixed_http_concurrency': args.fixed_http_concurrency,
        'fair_http_control': str(args.fair_http_control) if args.fair_http_control else None,
        'shared_frame_workers': args.shared_frame_workers,
        'prewarm_shared_frame_workers': args.prewarm_shared_frame_workers,
        'fair_http_capacity': args.fair_http_capacity,
        'ecot_memory_request_slots': args.ecot_memory_request_slots,
        'grd_shared_frame_workers': args.grd_shared_frame_workers,
        'request_timeout_seconds': args.request_timeout_seconds,
        'request_working_memory_mib': args.request_working_memory_mib,
        'ecot_request_memory_mib': args.ecot_request_memory_mib,
        'ecot_interval': args.ecot_interval,
        'ecot_contract_id': TASK_CONTRACT_IDS['ecot'],
        'checkpoint_spool_root': str(args.checkpoint_spool_root) if args.checkpoint_spool_root else None,
        'durable_cloud_transport': args.durable_cloud_transport,
        'ecot_reuse_sources': [source.as_dict() for source in args.ecot_reuse_sources],
        'ecot_reuse_partial_checkpoints': args.ecot_reuse_partial_checkpoints,
        'checkpoint_sync_workers': args.checkpoint_sync_workers,
        'final_publication_workers': args.final_publication_workers,
        'final_publication_spool_root': (str(args.final_publication_spool_root)
                                         if args.final_publication_spool_root else None),
        'final_publication_spool_max_mib': args.final_publication_spool_max_mib,
        'final_publication_spool_max_files': args.final_publication_spool_max_files,
        'final_publication_local_batch_size': args.final_publication_local_batch_size,
        'checkpoint_local_batch_size': args.checkpoint_local_batch_size,
        'retain_unit_checkpoints_after_final': args.retain_unit_checkpoints_after_final,
        'ecot_video_transport': args.ecot_video_transport,
        'ecot_image_transport': args.ecot_image_transport,
        'grd_media_transport': args.grd_media_transport,
        'dashscope_upload_policy_qps': args.dashscope_upload_policy_qps,
        'dashscope_media_post_timeout_seconds': list(UPLOAD_TIMEOUT),
        'ecot_completion_normalization': 'explicit-final-negative/v1',
        'dashscope_pooled_uploads': args.dashscope_pooled_uploads,
        'dashscope_isolated_policy': args.dashscope_isolated_policy,
        'resume_validation_workers': args.resume_validation_workers,
        'video_prepare_codec_threads': args.video_prepare_codec_threads,
        'dashscope_server_wait_seconds': args.dashscope_server_wait_seconds,
        'dashscope_cohort_pacing_enabled': os.environ.get('VQA_DASHSCOPE_COHORT_PACING') in {'1', 'mature'},
        'dashscope_cohort_pacing_mode': os.environ.get('VQA_DASHSCOPE_COHORT_PACING', '0'),
        'dashscope_cohort_initial_interval': os.environ.get('VQA_DASHSCOPE_COHORT_INITIAL_INTERVAL'),
        'dashscope_recoverable_5xx_enabled': os.environ.get('VQA_DASHSCOPE_RECOVERABLE_5XX') == '1',
        'dashscope_upload_workers': args.dashscope_upload_workers,
        'follow_subtask_path': args.follow_subtask_path,
        'independent_stage_pipeline': args.independent_stage_pipeline,
        'stage_prefetch': args.stage_prefetch,
        'grd_media_prefetch': args.grd_media_prefetch,
        'grd_review_model': args.grd_review_model, 'grd_review_api': args.grd_review_api,
        'record_limit': args.record_limiter.limit,
        'request_parallel': args.request_parallel,
        'total_http_limit': args.max_total_http_active if args.request_parallel else None,
        'las_request_workers': args.las_request_workers if args.request_parallel else None,
        'shared_las_workers': args.shared_las_workers,
        'las_transport_control': str(args.las_transport_control) if args.las_transport_control else None,
        'max_las_operators': args.max_las_operators if args.request_parallel else None,
        'video_prepare_workers': args.video_prepare_workers if args.request_parallel else None,
        'video_clip_workers': args.video_clip_workers,
        'video_clip_process_pool': args.video_clip_process_pool,
        'video_clip_codec_threads': int(os.environ.get('VQA_VIDEO_CLIP_CODEC_THREADS', '1')),
        'prioritize_failed_records': args.prioritize_failed_records,
        'immutable_jsonl': args.immutable_jsonl,
        'durable_write_workers': args.durable_write_workers,
        'sta_event_workers': args.sta_event_workers,
        'downstream_stage_workers': args.downstream_stage_workers,
        'batch_workers': args.batch_workers,
        'request_budget_bytes': REQUEST_BUDGET,
        'max_consecutive_request_failures': args.max_consecutive_request_failures,
        'ecot_word_limits_enforced': False,
        'ecot_validation_max_retries': ECOT_VALIDATION_MAX_RETRIES,
        'opencv_decoder_threads': int(os.environ.get('VQA_OPENCV_DECODER_THREADS', '0')),
        'video_clip_cpu_budget': int(os.environ.get('VQA_VIDEO_CLIP_CPU_BUDGET', '0')),
        'physical_batch_size': args.physical_batch_size,
        'record_uid_filter_count': len(args.record_uid_filter) if args.record_uid_filter is not None else None,
        'max_records': args.max_records, 'max_shards': args.max_shards,
    }
    runtime_config['runtime_state_dir'] = str(args.runtime_state_dir)
    runtime_path = args.runtime_state_dir/'runtime-config.json'
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_tmp_path = runtime_path.with_name(f'{runtime_path.name}.{os.getpid()}.tmp')
    with runtime_tmp_path.open('w') as stream:
        json.dump(runtime_config, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(runtime_tmp_path, runtime_path)
    print(
        f'run_start tasks={[task for task in TASK_ORDER if task in args.tasks]} '
        f'input={args.input} output={args.output} workers={args.workers} '
        f'subtask_backend={"las" if "subtask" in args.tasks else "external"} '
        f'executor_workers={record_executor_workers(args)} '
        f'max_record_active={args.max_record_active} '
        f'las_media_workers={args.las_media_workers} '
        f'max_http_active={args.max_http_active} max_pending={args.max_pending} '
        f'request_start_interval={args.request_start_interval} '
        f'temporal_media_types={sorted(args.temporal_media_types)} '
        f'temporal_manifest={args.temporal_manifest} '
        f'grd_checkpoint_root={args.grd_checkpoint_root if "grd" in args.tasks else "disabled"} '
        f'annotation_checkpoint_root={args.annotation_checkpoint_root}',
        flush=True,
    )
    from request_parallel import report_admission
    def model_admission_metrics():
        from durable_jsonl import commit_metrics
        result = {'downstream': args.client.admission_snapshot()} if args.client is not None else {}
        if 'grd' in args.tasks:
            result['inventory_review'] = review_client.admission_snapshot()
        metrics = {'model_admission': result, **commit_metrics()}
        durable_publisher = getattr(args, '_durable_cloud_publisher', None)
        if durable_publisher is not None:
            metrics['durable_cloud_output'] = durable_publisher.snapshot()
        resume_validator = getattr(args, '_resume_validation_pool', None)
        if resume_validator is not None:
            metrics['resume_validation'] = resume_validator.snapshot()
        if args.las_transport is not None:
            metrics['las_transport'] = args.las_transport.snapshot()
        runtime = getattr(args, 'stage_pipeline', None)
        if runtime is not None:
            metrics['stage_pipeline'] = runtime.snapshot()
        prefetch = getattr(args.client, 'media_prefetch_pool', None) if args.client else None
        if prefetch is not None:
            metrics['media_prefetch'] = prefetch.snapshot()
        if VIDEO_CLIP_POOL is not None:
            metrics['video_clip_pool'] = VIDEO_CLIP_POOL.snapshot()
        return metrics
    metrics_context = (report_admission(args.request_admission, args.runtime_state_dir/'request-parallel-status.json',
                                        extra_metrics=model_admission_metrics)
                       if args.request_admission is not None else contextlib.nullcontext())
    def selected_batches():
        seen_labels = set()
        for label, input_locator, source_key, samples, expected in input_batches(args):
            if args.record_uid_filter is not None:
                samples, filtered_expected = filter_batch_samples(
                    samples, args.record_uid_filter,
                )
                if filtered_expected == 0:
                    continue
                expected = filtered_expected
            if args.batch_workers > 1:
                identity = (source_key, label)
                if identity in seen_labels:
                    raise ValueError(f'parallel_batch_output_collision:{identity}')
                seen_labels.add(identity)
            yield label, input_locator, source_key, samples, expected

    def process_one_batch(batch, lane=None):
        # Stage commit/check callbacks belong to one output shard. Shared clients,
        # record admission, media gates and task caches remain the same objects.
        batch_args = copy.copy(args)
        if lane is not None:
            batch_args.tasks = {lane}
        return process_batch(batch_args, *batch)

    shared_frame_context = (concurrent.futures.ThreadPoolExecutor(
        args.shared_frame_workers, thread_name_prefix=f'{args.api}-frame',
    ) if args.shared_frame_workers else contextlib.nullcontext(None))
    grd_frame_context = (concurrent.futures.ThreadPoolExecutor(
        args.grd_shared_frame_workers, thread_name_prefix=f'{args.api}-grd-frame',
    ) if args.grd_shared_frame_workers else contextlib.nullcontext(None))
    durable_cloud_publisher = None
    if args.durable_cloud_transport == 'cos-direct':
        from cos_output_transport import CosOutputPublisher
        durable_cloud_publisher = CosOutputPublisher(
            max(args.checkpoint_sync_workers, args.final_publication_workers)
        )
    args._durable_cloud_publisher = durable_cloud_publisher
    storage_context = durable_jsonl.checkpoint_storage(
        args.checkpoint_spool_root, args.output, args.checkpoint_sync_workers,
        args.checkpoint_spool_max_mib, args.checkpoint_spool_max_files,
        local_batch_size=args.checkpoint_local_batch_size,
        writer=(durable_cloud_publisher.write_one if durable_cloud_publisher else None),
        batch_writer=(durable_cloud_publisher.write_checkpoint_batch
                      if durable_cloud_publisher else None),
    )
    publication_context = (durable_jsonl.publication_capacity(args.final_publication_workers)
                           if args.final_publication_spool_root is None
                           else contextlib.nullcontext())
    final_storage_context = durable_jsonl.final_record_storage(
        args.final_publication_spool_root, args.output,
        workers=args.final_publication_workers,
        max_mib=args.final_publication_spool_max_mib,
        max_files=args.final_publication_spool_max_files,
        local_batch_size=args.final_publication_local_batch_size,
        writer=(durable_cloud_publisher.write_one if durable_cloud_publisher else None),
        batch_writer=(durable_cloud_publisher.write_final_record_batch
                      if durable_cloud_publisher else None),
    )
    stage_context = (StagePipeline(
        args.tasks, args.max_record_active, args.video_prepare_workers, args.stage_prefetch,
    ) if args.independent_stage_pipeline else contextlib.nullcontext(None))
    prefetch_context = (MediaPrefetchPool(args.video_clip_workers, args.grd_media_prefetch)
                        if args.independent_stage_pipeline else contextlib.nullcontext(None))
    las_context = (concurrent.futures.ThreadPoolExecutor(args.shared_las_workers, thread_name_prefix='shared-las')
                   if args.shared_las_workers else contextlib.nullcontext(None))
    from resume_validation_pool import ResumeValidationPool
    resume_context = (ResumeValidationPool(args.resume_validation_workers)
                      if args.resume_validation_workers else contextlib.nullcontext(None))
    availability_checker = getattr(args.request_admission, 'availability_checker', None)
    availability_context = (availability_checker.refreshing()
        if availability_checker is not None and args.api in {'ark', 'las'} and args.tasks == {'ecot'}
        and args.ecot_video_transport in ('ark-files', 'cos-presigned') and args.independent_stage_pipeline
        else contextlib.nullcontext(None))
    with publication_context, storage_context, final_storage_context, availability_context, video_clip_execution(args.video_clip_workers, args.video_clip_process_pool), prefetch_context as media_prefetch, metrics_context, shared_frame_context as shared_frames, grd_frame_context as grd_frames, stage_context as stage_runtime, las_context as las_executor, resume_context as resume_validator:
        if resume_validator is not None:
            resume_validator.prewarm()
        prewarm_thread_pool(
            shared_frames, args.prewarm_shared_frame_workers,
            f'{args.api}-frame',
        )
        args._resume_validation_pool = resume_validator
        args.las_executor = las_executor
        args.stage_pipeline = stage_runtime
        if args.client is not None:
            args.client.frame_executor = shared_frames
            args.client.frame_executors = {'ecot': shared_frames, 'grd': grd_frames or shared_frames}
            args.client.media_prefetch_pool = media_prefetch
        totals_lock = threading.Lock()
        def record_status(batch, status):
            label = batch[0]
            with totals_lock:
                totals['batches'] += 1
                totals['errors'] += status['errors']
                for task, count in status['written'].items():
                    totals['written'][task] += count
                print(f'batch_complete input={label} totals={json_dumps(totals)}', flush=True)
        if args.independent_stage_pipeline:
            stopped = threading.Event()
            previous_check = args.request_admission.check_available
            def check_cancelled_or_provider():
                if stopped.is_set():
                    raise RuntimeError('independent_stage_pipeline_cancelled')
                if previous_check is not None:
                    previous_check()
            args.request_admission.check_available = check_cancelled_or_provider
            def check_pipeline():
                if stopped.is_set():
                    raise RuntimeError('independent_stage_pipeline_cancelled')
                args.client.ensure_available()
            args.pipeline_check = check_pipeline
            def run_lane(lane):
                deferred_labels = None
                try:
                    while True:
                        pending_labels = set()
                        def batches():
                            for batch in selected_batches():
                                check_pipeline()
                                if deferred_labels is None or (batch[2], batch[0]) in deferred_labels:
                                    yield batch
                        for batch, status in run_bounded_batches(
                            batches(), lambda batch: process_one_batch(batch, lane), args.batch_workers,
                        ):
                            record_status(batch, status)
                            if status.get('deferred_upstream'):
                                pending_labels.add((batch[2], batch[0]))
                        if not pending_labels:
                            return
                        deferred_labels = pending_labels
                        print(f'stage_upstream_deferred lane={lane} batches={len(pending_labels)}', flush=True)
                        for _ in range(30):
                            check_pipeline()
                            stopped.wait(1)
                except BaseException:
                    stopped.set()
                    raise
            with concurrent.futures.ThreadPoolExecutor(len(args.tasks), thread_name_prefix='stage-catalog') as lanes:
                futures = [lanes.submit(run_lane, lane) for lane in sorted(args.tasks)]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for batch, status in run_bounded_batches(selected_batches(), process_one_batch, args.batch_workers):
                record_status(batch, status)
    print(f'run_complete totals={json_dumps(totals)}', flush=True)
    if totals['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
