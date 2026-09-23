"""Launch an isolated Files canary or verified stopped-worker recovery."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from request_parallel import atomic_json
from reload_independent_workers import verify_empty_outbox
from reload_ark_ecot_capacity import archive_markers
from run_cosmos3_ark_ecot import MODEL, PROJECT
from switch_ecot_to_ark import canary_evidence
from task_process_state import acquire_writer_locks, state_directory


ORIGINAL = PROJECT / '_runtime/ark-ecot-seedlite-20260911-r1'
CANARY = PROJECT / '_runtime/ark-files-canary-20260912'
FULL = PROJECT / '_runtime/ark-files-70g-20260912-r3'
OUTPUT = Path('/mnt/human_data/video_cleaning/cosmos3-video-generation-general-v1.5-vqa-ark-seedlite-ecot-full-20260911')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('canary', 'full'), required=True)
    parser.add_argument('--monitor-only', action='store_true')
    args = parser.parse_args()
    full = args.phase == 'full'
    archive = FULL if full else CANARY
    output = OUTPUT if full else archive / 'output'
    cap = 2048 if full else 256
    def monitor_manifest(started):
        atomic_json(archive / 'launch.json', {
            'phase': 'ark-files', 'session': 'vqa-ark-ecot2048',
            'source_drain_certificate': str(ORIGINAL / 'source-drain/certificate.json'),
            'las_completion_audit': str(PROJECT / '_runtime/las-tail-coverage-20260911-0745.json'),
            'tasks': {'ecot': {'output': str(output), 'state': str(state_directory(output, 'ecot')),
                      'log': str(archive / 'supervisor.log'), 'configured_http_cap': cap,
                      'started_at_unix': started}},
            'handed_off_tasks': {'grd': 'x2y-5, unchanged', 'sta': 'x2y-5, unchanged'}})
    if args.monitor_only:
        start = json.loads((archive / 'start-command.json').read_text())
        monitor_manifest(start['created_at_unix'])
        return
    uid_path = ORIGINAL / 'canary-uids.txt'
    uid = uid_path.read_text().strip()
    if archive.exists():
        raise SystemExit('Archive exists; inspect before another launch.')
    if full:
        evidence = canary_evidence(CANARY / 'output', uid, 'catalog-000020')
        record = json.loads(Path(evidence['file']).read_text())
        requests = record['annotations']['robot_extension']['requests']
        if any(r.get(k, {}).get('method') != 'ark-files'
               for r in requests for k in ('video_transport', 'image_transport')):
            raise RuntimeError('Canary did not exercise native Files for every media input')
        state = state_directory(output, 'ecot')
        status = json.loads((state / 'supervisor-status.json').read_text())
        runtime = json.loads((state / 'runtime-config.json').read_text())
        if (status.get('status') != 'stopped' or status.get('pid') != 159587
                or Path('/proc/159587').exists() or status.get('model') != MODEL):
            raise RuntimeError('Stopped source identity changed; preserve markers')
        markers = [state / name for name in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json')]
        markers = [p for p in markers if p.exists()]
        if not markers:
            raise RuntimeError('Expected diagnosed source marker is missing')
        originals = [p.read_bytes() for p in markers]
        for raw in originals:
            marker = json.loads(raw)
            if (marker.get('http_status') != 0 or marker.get('model') != MODEL
                    or not marker.get('error', '').startswith('provider_circuit_open:')
                    or 'The write operation timed out' not in marker['error']):
                raise RuntimeError('Unexpected provider marker; do not clear')
        with ExitStack() as held:
            for name in ('supervisor.lock', 'full-run.lock'):
                for fd in acquire_writer_locks(output, name, 'ecot'):
                    held.callback(os.close, fd)
            outbox = verify_empty_outbox(runtime, output)
            archive.mkdir(mode=0o700)
            for name in ('runtime-config.json', 'supervisor-status.json', 'request-parallel-status.json'):
                shutil.copy2(state / name, archive / ('before-' + name))
            atomic_json(archive / 'recovery-evidence.json', {
                'canary': evidence, 'source_pid': 159587, 'source_workers_exited': True,
                'writer_locks_verified': True, 'outbox': outbox,
                'diagnosis': 'Repeated inline video write timeouts; native Files transport verified',
                'user_reported_rpm': 30000, 'user_reported_tpm': 300000000,
                'created_at_unix': time.time()})
            if [p.read_bytes() for p in markers] != originals:
                raise RuntimeError('Provider marker changed during recovery')
            archive_markers(markers, archive)
    else:
        archive.mkdir(mode=0o700)
    monitor_manifest(time.time())
    atomic_json(archive / 'control.json', {
        'schema_version': 'qwen-http-control/v1', 'output': str(output),
        'max_http_active': cap, 'revision': 'ark-files-70g' if full else 'ark-files-canary',
        'python_switch_interval_ms': 1.0})
    command = [sys.executable, '-u', str(PROJECT / 'scripts/run_cosmos3_ark_ecot.py'),
               '--output', str(output), '--archive', str(archive),
               '--control', str(archive / 'control.json'), '--http-limit', str(cap),
               '--request-memory-mib', '71680' if full else '4096',
               '--guard-gib', '92' if full else '16', '--media-transport', 'ark-files',
               '--upload-workers', '32' if full else '4', '--initial-pacing-seconds', '.02']
    if full:
        command += ['--source-drain-certificate', str(ORIGINAL / 'source-drain/certificate.json')]
    else:
        command += ['--record-uid-path', str(uid_path), '--canary-no-reuse']
    shell = shlex.join(command) + ' > ' + shlex.quote(str(archive / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-window', '-d', '-t', 'vqa-ark-ecot2048',
                    '-n', 'files-full' if full else 'files-canary', '-c', str(PROJECT), shell], check=True)
    atomic_json(archive / 'start-command.json', {'command': command, 'created_at_unix': time.time()})
    print('ark_files_launched phase=' + args.phase + ' archive=' + str(archive), flush=True)


if __name__ == '__main__':
    main()
