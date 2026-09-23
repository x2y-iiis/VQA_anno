"""Durable native-rate backtracking and original CPA participant question policy."""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
import threading
from pathlib import Path

import cv2

import annotate_videos as core
from cpa_backtracking import load_tracker, track_contact_points
from cpa_semantic_review import point_rows
from cpa_student_video import with_student_video

VERSION = 'cpa-native-tracking-object-two-decimal/v5'
CPA_MIN_CONTACT_GAP_SECONDS = 0.2
CPA_MAX_CONTACT_GAP_SECONDS = 0.4
LEGACY_SOURCE = Path('/mnt/robot_vqa_sta_cpa/src/generate_robot.py')
PIPELINE_LOCK = threading.Lock()
PIPELINES = {}


def complete_source_cpa(source, media, value, args):
    """Connect the production annotator to durable CPA learner records."""
    # Student tracking state is process recovery data, not a published
    # annotation artifact.  Keeping it below the shared output made newer CPA
    # identities collide with stale checkpoints from older fleet versions.
    root = Path(args.runtime_state_dir) / 'cpa-student-coordinates'
    root.mkdir(parents=True, exist_ok=True)
    video_item = next(((mime,payload) for mime,payload in media if mime.startswith('video/')),None)
    if video_item is None:
        raise ValueError('cpa_student_pipeline_requires_original_video')
    video_path, temporary = core.decode_video_path(video_item[1])
    try:
        source_hash = hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
        capture = cv2.VideoCapture(str(video_path))
        source_items=source.get('media',{}).get('items',[])
        source_media=next((copy.deepcopy(item) for item in source_items if item.get('type')=='video'),{})
        durable_source=source_media.get('relative_path') or str(video_path)
        if temporary and not source_media:
            raise ValueError('CPA requires a durable source media reference for temporary input')
        sample = {'id':'source-'+digest(str(source['uid']))[:16],
                  'selection_source_uid':str(source['uid']), 'sha256':source_hash,'video':str(video_path),
                  'source_media':source_media,'source_locator_video':durable_source,
                  'family':core.las_embodiment(source,args.las_embodiment),
                  'task_instruction':core.las_task_instruction(source,args.task_instruction),
                  'width':int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  'height':int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  'fps':float(capture.get(cv2.CAP_PROP_FPS))}
        capture.release()
        directory=root/digest({'uid':source['uid'],'source':source_hash})
        # Share the tracker model; only GPU inference holds its lock. Remote
        # review requests and independent episodes remain concurrent.
        with PIPELINE_LOCK:
            pipeline_key=(id(args.cpa_downstream_client),args.sam3_device,args.cpa_student_history_seconds,args.cpa_question_mode)
            pipeline=PIPELINES.get(pipeline_key)
            if pipeline is None:
                pipeline=CpaDownstream(args.cpa_downstream_client,device=args.sam3_device,
                    history_seconds=args.cpa_student_history_seconds,question_mode=args.cpa_question_mode,
                    robot_agent=False)
                PIPELINES[pipeline_key]=pipeline
        return pipeline.annotate(sample,value,directory,new_pipeline=True,require_review=args.cpa_semantic_review,
            lookback_seconds=getattr(args,'cpa_max_contact_gap_seconds',CPA_MAX_CONTACT_GAP_SECONDS),
            observations_per_contact=args.sta_observations_per_contact,
            minimum_contact_gap_seconds=getattr(args,'cpa_min_contact_gap_seconds',CPA_MIN_CONTACT_GAP_SECONDS),
            random_seed=args.sta_random_seed)
    finally:
        if temporary:Path(video_path).unlink(missing_ok=True)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def legacy_module():
    spec = importlib.util.spec_from_file_location('cpa_original_policy', LEGACY_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairs_for_legacy(points):
    pairs = {}
    for point in points:
        index = point['pair_index']
        pair = pairs.setdefault(index, {'pair_index': index, 'hand_point_xy': None,
                                       'object_point_xy': None, 'hand_visible': False,
                                       'object_visible': False})
        role = 'hand' if point['role'] == 'contact_agent' else 'object'
        pair[f'{role}_point_xy'] = [v / 1000 for v in point['xy_1000']]
        pair[f'{role}_visible'] = point.get('visible') is True
    return list(pairs.values())


def tracking_review(client, legacy, audit, source, tracked, directory):
    from cpa_visibility_review import review_tracks
    return review_tracks(client, legacy, audit, source, tracked, directory,
                         digest=digest, atomic_json=atomic_json, pipeline_version=VERSION)


class CpaDownstream:
    def __init__(self, client, *, device='cuda', history_seconds=2.0,
                 question_mode='coordinates', robot_agent=False):
        self.client, self.device = client, device
        self.history_seconds = history_seconds
        self.question_mode, self.robot_agent = question_mode, robot_agent
        self.legacy = legacy_module()
        # ECoT keeps bounded process-wide pools instead of creating a model
        # (and a serial lock) for every resident episode.  CPA uses two
        # tracker replicas by default so independent contact events can make
        # progress concurrently on a 72 GiB GPU.  The environment override
        # lets a smaller GPU fall back to one replica without changing the
        # annotation contract or checkpoint identity.
        try:
            replicas = int(os.environ.get('CPA_TRACKER_REPLICAS', '2'))
        except (TypeError, ValueError):
            replicas = 2
        self.tracker_replicas = max(1, min(16, replicas))
        self.trackers = [None] * self.tracker_replicas
        self.tracker_locks = [threading.Lock() for _ in range(self.tracker_replicas)]
        self.tracker_init_lock = threading.Lock()
        self.tracker_next = 0

    def _next_tracker(self):
        """Lazily initialize and reserve a bounded tracker slot."""
        with self.tracker_init_lock:
            index = self.tracker_next % self.tracker_replicas
            self.tracker_next += 1
            if self.trackers[index] is None:
                self.trackers[index] = load_tracker(device=self.device)
            return index, self.trackers[index], self.tracker_locks[index]

    def annotate(self, sample, value, directory, *, new_pipeline=True,
                 require_review=None,
                 lookback_seconds=CPA_MAX_CONTACT_GAP_SECONDS,
                 observations_per_contact=core.STA_OBSERVATIONS_PER_CONTACT,
                 minimum_contact_gap_seconds=CPA_MIN_CONTACT_GAP_SECONDS, random_seed=0):
        directory = Path(directory)
        if require_review is None:
            require_review = new_pipeline
        directory.mkdir(parents=True, exist_ok=True)
        identity = {'version': VERSION, 'source_sha256': sample['sha256'],
                    'point_result_sha256': digest(value), 'new_pipeline': new_pipeline,
                    'require_contact_review': require_review,
                    'legacy_sha256': hashlib.sha256(LEGACY_SOURCE.read_bytes()).hexdigest(),
                    'history_seconds': self.history_seconds, 'lookback_seconds': lookback_seconds,
                    'observations_per_contact': observations_per_contact,
                    'minimum_contact_gap_seconds': minimum_contact_gap_seconds, 'random_seed': random_seed}
        if self.question_mode == 'coordinates':
            identity.update(question_mode='object_coordinates_two_decimal_places',robot_agent=False,
                            instruction_source='subtask',contact_handedness='not_required_for_object_only')
        final = directory / 'complete.json'
        if final.exists():
            cached = json.loads(final.read_text())
            if cached['identity'] != identity:
                raise ValueError('cpa_downstream_checkpoint_mismatch')
            return cached['result']
        result = copy.deepcopy(value)
        media = [('video/mp4', Path(sample['video']).read_bytes())]
        all_events = [e for segment in result['subtask_results']
                      for e in segment['result'].get('reviewed_contact_events', []) if e.get('accepted')]
        audits, tracking_map, reviews_map, event_map = [], {}, {}, {}
        review_jobs = []
        for segment in result['subtask_results']:
            step = segment['subtask']
            for event in segment['result'].get('reviewed_contact_events', []):
                if not event.get('accepted'):
                    continue
                cid = f'{sample["id"]}-s{step["id"]}-{event["event_id"]}'
                event['student_questions'] = []
                event['tracking_reviews'] = []
                event.pop('backtracking',None)
                event_dir = directory / cid
                event_dir.mkdir(exist_ok=True)
                # Use exact native-frame time so rounded API timestamps cannot shift interval endpoints.
                sampling_event={**event,'contact_time_seconds':event['contact_source_frame_index']/sample['fps']}
                sampling_events=[{**e,'contact_time_seconds':e['contact_source_frame_index']/sample['fps']}
                    for e in all_events if e.get('contact_source_frame_index') is not None]
                observations, selection = core.sta_random_observation_frames(media, step, sampling_event, sampling_events,
                    lookback_seconds, observations_per_contact, minimum_contact_gap_seconds, random_seed,
                    sample.get('selection_source_uid', sample['id']))
                sources = []
                for metadata, (_, payload) in observations:
                    query = metadata['source_frame_index']
                    gap=(event['contact_source_frame_index']-query)/sample['fps']
                    if not minimum_contact_gap_seconds-1e-8 <= gap <= lookback_seconds+1e-8:
                        raise ValueError('CPA query outside configured contact-gap interval')
                    oid = f'{cid}-q{query}'
                    image_path = event_dir / f'{oid}.jpg'
                    image_path.write_bytes(payload)
                    sources.append({'observation_id': oid, 'query_frame_index': query,
                                    'query_image': image_path.name, 'query_image_path': str(image_path),
                                    'time_to_contact_seconds': (event['contact_source_frame_index'] - query) / sample['fps']})
                selection.update(minimum_contact_gap_seconds=minimum_contact_gap_seconds,lookback_seconds=lookback_seconds,
                    actual_contact_gaps_seconds=[s['time_to_contact_seconds'] for s in sources])
                event['student_observation_selection'] = {'policy': 'uniform_random_native_frames_within_contact_gap_interval',
                    'metadata': selection, 'query_frame_indices': [s['query_frame_index'] for s in sources]}
                points = point_rows(event.get('contact_point_pairs', []))
                if not points or not sources:
                    event['student_pipeline_status'] = 'no_contact_points' if not points else 'no_eligible_query_frames'
                    continue
                track_path = event_dir / 'tracking.json'
                track_identity = digest({'points': points, 'queries': sources, 'identity': identity, 'cid': cid})
                if track_path.exists():
                    cached = json.loads(track_path.read_text())
                    if cached['identity'] != track_identity:
                        raise ValueError('cpa_tracking_checkpoint_mismatch')
                    tracking = cached['tracking']
                else:
                    slot, tracker, tracker_lock = self._next_tracker()
                    with tracker_lock:
                        print(f'cpa_tracking {cid} slot={slot} targets={[s["query_frame_index"] for s in sources]}', flush=True)
                        tracking = track_contact_points(tracker, Path(sample['video']), event,
                            [s['query_frame_index'] for s in sources], device=self.device, require_review=require_review)
                    atomic_json(track_path, {'identity': track_identity, 'tracking': tracking})
                event['backtracking'] = tracking
                role = event.get('agent_role') or ('human_hand' if sample['family'] == 'human' else 'robot_gripper')
                audit = {'candidate_id': cid, 'candidate_frame': event['contact_source_frame_index'],
                         'task_instruction': sample['task_instruction'], 'subtask_instruction': step['subtask'], 'actor': sample['family'],
                         'dataset_family': sample['family'], 'source_video': sample['video'], 'source_sha256':sample['sha256'],
                         'end_effector_type': role, 'contact_agent_role': role,
                         'contact': {'noun': event['object_name']},
                         'source_resolution': [sample['width'], sample['height']], 'observations': sources,
                         'contact_point_pairs_full_xy': pairs_for_legacy(points)}
                converted = {'tracked_observations': []}
                for observation in tracking['tracked_observations']:
                    source = next(s for s in sources if s['query_frame_index'] == observation['query_frame_index'])
                    tracked = {'observation_id': source['observation_id'], 'query_frame_index': source['query_frame_index'],
                               'contact_pairs': pairs_for_legacy(observation['points'])}
                    converted['tracked_observations'].append(tracked)
                    review_jobs.append((audit, source, tracked, event_dir))
                audits.append(audit)
                tracking_map[cid], event_map[cid] = converted, event
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(tracking_review, self.client, self.legacy, *job): job[1]['observation_id']
                    for job in review_jobs}
            for future in concurrent.futures.as_completed(jobs):
                reviews_map[jobs[future]] = future.result()
        if self.question_mode == 'coordinates':
            from cpa_coordinate_questions import questions_from_result, learner_example
            from cpa_handedness import label_event
            for audit in audits:
                event_map[audit['candidate_id']]['tracking_reviews'] = [reviews_map[s['observation_id']] for s in audit['observations']]
            from cpa_pair_policy import enforce_event_pairs
            for event in all_events:
                enforce_event_pairs(event,require_hand_side=False)
            questions = questions_from_result(sample, result, history_seconds=self.history_seconds,
                                               robot_agent=self.robot_agent)
            for segment in result['subtask_results']:
                for event in segment['result'].get('reviewed_contact_events', []):
                    if not event.get('accepted'):
                        continue
                    event['student_questions'] = [q for q in questions if q['subtask_id'] == segment['subtask']['id'] and q['event_id'] == event['event_id']]
                    event['student_pipeline_status'] = 'completed' if event['student_questions'] else event.get('student_pipeline_status', 'no_valid_observation_points')
            result['student_pipeline'] = {**identity, 'status':'completed', 'question_count':len(questions),
                'student_context':'unmarked_5fps_history_ending_at_query', 'answer_precision':'two_decimal_places'}
            result['student_examples'] = [learner_example(q, sample.get('source_locator_video',sample['video'])) for q in questions]
            if sample.get('source_media'):
                for example in result['student_examples']:example['source_media']=copy.deepcopy(sample['source_media'])
            (directory / 'student_examples.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in result['student_examples']))
            atomic_json(final, {'identity':identity, 'result':result})
            return result
        prompts = self.legacy.build_cpa_observation_choice_prompts(audits, tracking_map, reviews_map)
        segmentations, errors = self.legacy.run_cpa_observation_choice_sam2(prompts, directory, True)
        if errors:
            raise RuntimeError('cpa_choice_sam2_failed:' + json.dumps(errors))
        segmented = {s['observation_id']: s for s in segmentations}
        questions = []
        for audit in audits:
            event = event_map[audit['candidate_id']]
            event['tracking_reviews'] = []
            event['observation_choice_segmentation'] = []
            for source in audit['observations']:
                oid = source['observation_id']
                event['tracking_reviews'].append(reviews_map[oid])
                segmentation = segmented.get(oid)
                if segmentation is None:
                    continue
                event['observation_choice_segmentation'].append(segmentation)
                rows = self.legacy.build_cpa_multiple_choice_vqa_records(audit, source, segmentation)
                for row in rows:
                    if new_pipeline:
                        row = with_student_video(row, start_frame=max(0, row['query_frame_index'] - round(self.history_seconds * sample['fps'])), source_fps=sample['fps'])
                    row['question_builder_provenance'] = {'source': str(LEGACY_SOURCE), 'sha256': identity['legacy_sha256'],
                                                         'function': 'build_cpa_multiple_choice_vqa_records'}
                    event['student_questions'].append(row)
                    questions.append(row)
            event['student_pipeline_status'] = 'completed' if event['student_questions'] else 'no_valid_observation_choices'
        result['student_pipeline'] = {**identity, 'status': 'completed', 'question_count': len(questions),
                                      'tracking_target_policy': 'existing_query_frames_only',
                                      'student_context': '5fps_history_ending_at_query' if new_pipeline else 'original_query_image'}
        # Separate learner examples exclude all future images and teacher review records.
        examples = []
        for q in questions:
            known = q.get('known_conditions') or {'type': 'image', 'source_frame_indices': [q['query_frame_index']], 'image_path': q['image_path']}
            examples.append({'id': q['id'], 'source_video': sample['video'],
                             'task_instruction': q['task_instruction'], 'known_conditions': known,
                             'question_frame_index': q['query_frame_index'], 'question': q['question'],
                             'choices': q['choices'], 'answer': q['answer_value'],
                             'choice_marker_source_frame_index': q['query_frame_index']})
        result['student_examples'] = examples
        (directory / 'student_examples.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in examples))
        atomic_json(final, {'identity': identity, 'result': result})
        return result
