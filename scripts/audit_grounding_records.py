#!/usr/bin/env python3
"""Review, repair failed names, and re-review grounding with Seed 2.1 Pro."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from grd_inventory_review import InventoryReviewer, learner_result, partition_result, review_is_current, review_item


def large_text_lines(stream, chunk_size=4 * 1024 * 1024):
    """Preserve text-mode newline/encoding semantics with fewer raw reads.

    Very large JSONL rows otherwise reacquire the GIL after every small read,
    competing with hundreds of request callbacks and other resume validators.
    """
    pending = []
    while chunk := stream.read(chunk_size):
        pieces = chunk.split('\n')
        pending.append(pieces[0])
        if len(pieces) > 1:
            yield ''.join(pending) + '\n'
            for line in pieces[1:-1]:
                yield line + '\n'
            pending = [pieces[-1]]
    if pending and any(pending):
        yield ''.join(pending)


def record_media(record, video_map=None):
    from annotate_videos import FileSlice
    uid = str(record['provenance']['input_record_uid'])
    override = (video_map or {}).get(uid)
    if override:
        path = Path(override)
        return [('video/mp4', FileSlice(path, 0, path.stat().st_size))]
    media = []
    for item in record.get('media', {}).get('items', []):
        path = Path(str(item.get('relative_path') or ''))
        if not path.is_file():
            continue
        mime = item.get('mime_type') or ('video/mp4' if item.get('type') == 'video' else 'image/jpeg')
        if item.get('member'):
            with tarfile.open(path, 'r:') as tar:
                member = tar.getmember(item['member'])
                payload = FileSlice(path, member.offset_data, member.size)
        else:
            payload = FileSlice(path, 0, path.stat().st_size)
        media.append((mime, payload if mime.startswith('video/') else payload.read()))
    if media:
        return media
    locator = record.get('provenance', {}).get('source_locator', {})
    path = Path(str(locator.get('source_file') or ''))
    if locator.get('storage_kind') == 'tar_member' and path.is_file():
        return [('video/mp4', FileSlice(path, int(locator['data_offset']), int(locator['length'])))]
    raise ValueError(f'grounding_review_media_not_found:{uid}:use_--video-manifest')


def record_is_reviewed(record):
    from annotate_videos import geometry
    result = record.get('annotations', {}).get('robot_extension', {}).get('result', {})
    items = result.get('subtask_results', []) + result.get('rejected_subtask_results', [])
    valid = bool(items) and 'inventory_review_summary' in result and all(
        len(item.get('result', {}).get('frames', [])) == 1
        and review_is_current(item['result']['frames'][0]) for item in items
    )
    if not valid:
        return False
    expected = partition_result(result, items)
    assistant_turns = [turn for turn in record.get('dialogue', {}).get('turns', [])
                       if turn.get('role') == 'assistant']
    expected_answer = learner_result(expected)
    try:
        training_is_current = bool(assistant_turns) and all(
            json.loads(turn['content']) == expected_answer
            and turn.get('loss') is bool(expected['subtask_results'])
            for turn in assistant_turns
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (result['inventory_review_summary'] == expected['inventory_review_summary']
            and result['subtask_results'] == expected['subtask_results']
            and result.get('rejected_subtask_results') == expected['rejected_subtask_results']
            and training_is_current
            and record['annotations'].get('geometry') == geometry('grd', expected)
            and record.get('cleaning', {}).get('grounding_decision') == (
                'keep' if expected['subtask_results'] else 'drop'))


def audit_record(record, reviewer, executor, video_map=None):
    from annotate_videos import EcotMediaFactory, geometry, video_frame_at_path, review_grd_inventory_item
    if record_is_reviewed(record):
        return record
    record = copy.deepcopy(record)
    extension = record['annotations']['robot_extension']
    result = extension['result']
    items = result['subtask_results'] + result.get('rejected_subtask_results', [])
    jobs = {}
    reviewed = [None] * len(items)
    needs_images = []
    for index, item in enumerate(items):
        frame = item['result']['frames'][0]
        if review_is_current(frame):
            reviewed[index] = item
        elif len(frame['objects']) < 2:
            reviewed[index] = review_item(item, reviewer, [])
        else:
            needs_images.append((index, item))
    if needs_images:
        media = record_media(record, video_map)
        with EcotMediaFactory(media) as factory:
            indices = sorted({int(item['result']['frames'][0]['media_index']) for _, item in needs_images})
            images = factory.selected_target_images([i for i in indices if 0 <= i < factory.sampled_frame_count])
            for index, item in needs_images:
                frame = item['result']['frames'][0]
                image = images.get(frame['media_index'])
                if image is None:
                    if factory.source_path is None:
                        raise ValueError('grounding_review_image_index_out_of_bounds')
                    image = video_frame_at_path(factory.source_path, frame['timestamp_seconds'])
                jobs[executor.submit(review_grd_inventory_item, item, reviewer, [image],
                                     source_path=factory.source_path, media=media)] = index
            for future in concurrent.futures.as_completed(jobs):
                reviewed[jobs[future]] = future.result()
    reviewed.sort(key=lambda item: (item['media_scope'].get('anchor_time_seconds') or 0,
                                    str(item['subtask']['id'])))
    result = partition_result(result, reviewed)
    extension['result'] = result
    record['annotations']['geometry'] = geometry('grd', result)
    for turn in record.get('dialogue', {}).get('turns', []):
        if turn.get('role') == 'assistant':
            turn['content'] = json.dumps(learner_result(result), ensure_ascii=False, separators=(',', ':'))
            turn['loss'] = bool(result['subtask_results'])
    record.setdefault('cleaning', {})['grounding_decision'] = 'keep' if result['subtask_results'] else 'drop'
    record['provenance']['annotation']['inventory_review_policy'] = result['inventory_review_summary']['policy']
    from validate_annotation_result_contract import validate_annotation_result_contract
    failures = validate_annotation_result_contract('grd', result, extension.get('requests'))
    if failures:
        raise ValueError(f'reviewed_grounding_contract_invalid:{dict(failures)}')
    return record


def audit_shard(source, destination, reviewer, workers=128, episode_workers=8, video_map=None):
    """Write only after every frame is reviewed; durable frame caches survive failures."""
    source, destination = Path(source), Path(destination)
    before = source.stat()
    with source.open() as stream:
        records = [json.loads(line) for line in large_text_lines(stream) if line.strip()]
    indices = [i for i, record in enumerate(records)
               if record.get('annotations', {}).get('robot_extension', {}).get('annotation_task') == 'grd']
    changed = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as frames, concurrent.futures.ThreadPoolExecutor(max_workers=episode_workers) as episodes:
        # Bound queued episode payloads; each active episode shares one frame cache.
        pending = {}
        # Record validation reconstructs training payloads and geometry. Reuse
        # this per-invocation decision for scheduling and progress accounting.
        reviewed_flags = {i: record_is_reviewed(records[i]) for i in indices}
        todo = iter(i for i in indices if not reviewed_flags[i])
        def submit_one():
            i = next(todo, None)
            if i is not None:
                pending[episodes.submit(audit_record, records[i], reviewer, frames, video_map)] = i
                return True
            return False
        for _ in range(episode_workers):
            submit_one()
        completed = sum(reviewed_flags.values())
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                i = pending.pop(future)
                records[i] = future.result()
                changed = True
                completed += 1
                summary = records[i]['annotations']['robot_extension']['result']['inventory_review_summary']
                print('inventory_review_episode_complete ' + json.dumps({
                    'completed': completed, 'total': len(indices),
                    'uid': records[i]['provenance']['input_record_uid'], **summary,
                }), flush=True)
                submit_one()
    if changed or source.resolve() != destination.resolve():
        after = source.stat()
        if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
            raise RuntimeError(f'grounding_input_changed_during_review:{source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if destination.exists():
            backup = destination.with_name(f'{destination.name}.before-inventory-review.{time.time_ns()}.bak')
            shutil.copy2(destination, backup)
        with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'inventory_review_shard_written path={destination} backup={backup}', flush=True)
    summaries = [records[i]['annotations']['robot_extension']['result']['inventory_review_summary'] for i in indices]
    return {'episodes': len(indices), **{key: sum(s[key] for s in summaries)
                                      for key in ('total_frames', 'accepted_frames', 'rejected_frames')}}


def main():
    from annotate_videos import ApiClient, DEFAULT_ENDPOINTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Grounding JSONL, grd directory, or pipeline output root')
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--output', type=Path, help='Destination directory; preserves relative shard paths')
    target.add_argument('--in-place', action='store_true', help='Back up and atomically replace reviewed input shards')
    parser.add_argument('--video-manifest', type=Path, help='Optional UID-to-video JSONL or showcase JSON')
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--episode-workers', type=int, default=8)
    parser.add_argument('--max-http-active', type=int, default=128)
    parser.add_argument('--max-attempts', type=int, default=6)
    parser.add_argument('--endpoint', default=DEFAULT_ENDPOINTS['ark'])
    parser.add_argument('--api-key-env', default='ARK_API_KEY')
    parser.add_argument('--rate-utilization', type=float, default=0.8)
    parser.add_argument('--request-start-interval', type=float, default=0)
    args = parser.parse_args()
    if min(args.workers, args.episode_workers, args.max_http_active, args.max_attempts) < 1:
        parser.error('Concurrency and attempt counts must be positive')
    if not 0 < args.rate_utilization <= 1 or args.request_start_interval < 0:
        parser.error('Invalid rate limit settings')
    root = args.input/'grd' if (args.input/'grd').is_dir() else args.input
    files = [root] if root.is_file() else sorted(root.rglob('*.jsonl'))
    if not files:
        parser.error('No grounding JSONL files found')
    state = args.checkpoint_root or (args.output or (root.parent if root.is_file() else root))/'_state'/'grd-inventory-reviews'
    args.api = 'ark'
    args.fatal_stop_file = state/'fatal-stop.json'
    args.rate_state_file = state/'rate-state.json'
    client = ApiClient(args)
    client.ensure_available()
    reviewer = InventoryReviewer(client, state)
    video_map = {}
    if args.video_manifest:
        if args.video_manifest.suffix == '.json':
            raw = json.loads(args.video_manifest.read_text())
            rows = raw.get('samples', []) if isinstance(raw, dict) else raw
        else:
            rows = [json.loads(line) for line in args.video_manifest.open() if line.strip()]
        video_map = {str(r.get('record_uid') or r.get('uid')): r.get('video_path') or r.get('video') for r in rows}
    totals = {'episodes': 0, 'total_frames': 0, 'accepted_frames': 0, 'rejected_frames': 0}
    for source in files:
        destination = source if args.in_place else args.output/(source.name if root.is_file() else source.relative_to(root))
        counts = audit_shard(source, destination, reviewer, args.workers, args.episode_workers, video_map)
        for key in totals:
            totals[key] += counts[key]
    state.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'complete', 'input': str(args.input.resolve()), **totals}
    (state/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('inventory_review_complete ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
