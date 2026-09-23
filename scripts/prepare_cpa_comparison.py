"""Prepare ten paired episodes and their unchanged CPA contact-frame baseline."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import cv2

import annotate_videos as core
from las_subtask import LASSubtaskAnnotator, LASSubtaskConfig


ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/mnt/cpa-ark-new')
BUNDLE = Path('/mnt/outputs/showcase_roboapi10_egodex_cpa10_20260901_transfer')
CONTACT_FUNCTIONS = ('sta_prompt', 'cpa_prompt', 'contact_frame_candidates',
                     'contact_frame_prompt', 'refine_contact_frames')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_env(path):
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


def function_hashes(path):
    return {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in CONTACT_FUNCTIONS}


def prepare_manifest(output):
    samples = json.loads((OLD / 'configs/samples.json').read_text())['samples']
    for family, directory, ids in (('human', 'human_video', ('003', '004')),
                                    ('robot', 'robot_video', ('000', '002'))):
        rows = {r['id']: r for r in (json.loads(line) for line in
                (BUNDLE / directory / 'meta.jsonl').read_text().splitlines())}
        for sample_id in ids:
            row = rows[sample_id]
            samples.append({'id': f'{family}-showcase-{sample_id}', 'family': family,
                            'video': str(BUNDLE / directory / row['video_path']),
                            'task_instruction': row['task_instruction']})
    for sample in samples:
        path = Path(sample['video'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if sample.get('sha256', digest) != digest:
            raise ValueError(f'Source changed: {sample["id"]}')
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        sample.update(sha256=digest, fps=fps, frame_count=count,
                      width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      duration_seconds=count / fps)
        capture.release()
    manifest = {'schema_version': 'cpa-paired10/v1',
                'selection_policy': 'The existing six demo episodes plus two human and two robot '
                                    'episodes from the transferred showcase; five per family. '
                                    'Both point models use the same ten episodes and contact frames.',
                'samples': samples}
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Comparison manifest changed; use a separate output directory')
    atomic_json(path, manifest)
    return samples


def baseline(sample, output):
    out = output / 'baseline' / sample['id']
    final = out / 'result.json'
    if final.exists():
        saved = json.loads(final.read_text())
        if saved['source_sha256'] != sample['sha256']:
            raise ValueError('Baseline source changed')
        print(f'baseline {sample["id"]} resumed', flush=True)
        return
    reference = OLD / 'outputs/ark' / sample['id'] / 'result.json'
    hashes = function_hashes(ROOT / 'scripts/annotate_videos.py')
    if reference.exists():
        legacy_hashes = function_hashes(OLD / 'ark_cpa/legacy/scripts/annotate_videos.py')
        if hashes != legacy_hashes or set(hashes) != set(CONTACT_FUNCTIONS):
            raise ValueError('Cannot reuse baseline: contact-frame functions differ')
        old = json.loads(reference.read_text())
        if old['source_sha256'] != sample['sha256']:
            raise ValueError('Legacy baseline source mismatch')
        value = copy.deepcopy(old['result'])
        for segment in value['subtask_results']:
            for event in segment['result'].get('reviewed_contact_events', []):
                for key in list(event):
                    if key in {'contact_point_pairs', 'point_pipeline', 'enlarged_crop_resolution',
                               'interaction_bbox_pixel_xyxy_expanded', 'contact_event_name'} or key.startswith('vlm_'):
                        event.pop(key)
        provenance = {'reference': str(reference), 'reference_sha256': hashlib.sha256(reference.read_bytes()).hexdigest(),
                      'contact_function_sha256': hashes, 'contact_function_ast_equal': True}
        atomic_json(out / 'subtasks.json', json.loads((reference.parent / 'subtasks.json').read_text()))
        requests = old['requests']
    else:
        args = core.parse_args(['--input', sample['video'], '--output', str(out),
                                '--tasks', 'subtask,cpa', '--api', 'ark', '--workers', '1',
                                '--max-http-active', '2', '--max-attempts', '3',
                                '--las-embodiment', sample['family']])
        client = core.ApiClient(args)
        models = core.DEFAULT_MODELS['ark'].copy()
        media = [('video/mp4', Path(sample['video']).read_bytes())]
        subpath = out / 'subtasks.json'
        if subpath.exists():
            subtasks = core.validate_subtasks(json.loads(subpath.read_text())['result'], media)
        else:
            print(f'baseline {sample["id"]} subtask', flush=True)
            config = LASSubtaskConfig(output_root=out,
                cos_uri_prefix='cos://datasets-1409717487/video-cleaning/cpa-paired10')
            result, raw, prompt = LASSubtaskAnnotator(config).annotate(
                Path(sample['video']), source_uid=sample['id'], task_instruction=sample['task_instruction'],
                embodiment=sample['family'], source_has_video=True, source_media_count=1)
            subtasks = core.validate_subtasks(result, media)
            atomic_json(subpath, {'result': subtasks, 'raw': raw, 'prompt': prompt})
        checkpoint = core.AnnotationUnitCheckpoint(out, sample['id'], 'cpa', {
            'source_sha256': sample['sha256'], 'subtasks': subtasks,
            'models': models, 'contact_function_sha256': hashes, 'include_cpa_points': False,
        })
        print(f'baseline {sample["id"]} contact_frame_selection', flush=True)
        value, requests = core.downstream_result(
            'cpa', {'uid': sample['id']}, media, subtasks, client, models, None, .15, 1024, {},
            unit_checkpoint=checkpoint, include_cpa_points=False)
        provenance = {'contact_function_sha256': hashes, 'contact_frame_selection': 'unchanged_main_pipeline'}
    atomic_json(final, {'sample_id': sample['id'], 'source_sha256': sample['sha256'],
                        'result': value, 'requests': requests, 'provenance': provenance})
    count = sum(bool(e.get('accepted')) for s in value['subtask_results']
                for e in s['result'].get('reviewed_contact_events', []))
    print(f'baseline {sample["id"]} completed contacts={count}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '_runtime/cpa-paired10-20260912')
    parser.add_argument('--env-file', type=Path, default=OLD / '.env')
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    load_env(args.env_file)
    cv2.setNumThreads(1)
    samples = prepare_manifest(args.output)
    if args.manifest_only:
        return
    failures = []
    for sample in samples:
        try:
            baseline(sample, args.output)
        except Exception as error:
            traceback.print_exc()
            failures.append({'sample_id': sample['id'], 'type': type(error).__name__, 'error': str(error)})
        atomic_json(args.output / 'baseline-progress.json', {
            'completed': sum((args.output / 'baseline' / s['id'] / 'result.json').exists() for s in samples),
            'total': len(samples), 'failures': failures, 'updated_at': time.time(),
        })
    if failures:
        raise SystemExit('Some baseline episodes failed; inspect baseline-progress.json')


if __name__ == '__main__':
    main()
