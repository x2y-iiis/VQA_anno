"""Drain and resume only Ark ECoT after an explicitly authorized quota upgrade."""
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
from types import SimpleNamespace

import annotate_videos as pipeline
from annotation_memory_profile import effective_memory_bytes
from reload_independent_workers import verify_empty_outbox
from request_parallel import atomic_json
from run_cosmos3_ark_ecot import MODEL, PROJECT, build_command
from switch_independent_task_processes import inspect_worker, same_live_process
from task_process_state import acquire_writer_locks


def capacity_settings(args):
    settings = {key: getattr(args, key, default) for key, default in (
        ('media_transport', 'inline'), ('upload_workers', 16),
        ('http_capacity', 2048), ('frame_workers', 2048), ('resident_episodes', 32),
        ('prepare_workers', 8), ('prepare_codec_threads', 4),
        ('batch_workers', 8), ('resume_validation_workers', 4))}
    # Keep old drain/resume plans byte-for-byte compatible while allowing a
    # deliberate zero-value override for the proven lazy frame-pool profile.
    if getattr(args, 'prewarm_frame_workers', None) is not None:
        settings['prewarm_frame_workers'] = args.prewarm_frame_workers
    if getattr(args, 'media_temp_root', None) is not None:
        settings['media_temp_root'] = str(args.media_temp_root)
    if getattr(args, 'local_spool_root', None) is not None:
        settings['local_spool_root'] = str(args.local_spool_root)
    return settings


def verify_cos_video_canary(archive):
    """Verify an end-to-end Ark result before changing full-run transport."""
    if archive is None:
        raise RuntimeError('cos_video_transport_requires_verified_canary')
    archive = Path(archive).resolve()
    runtime_root = (PROJECT / '_runtime').resolve()
    if not archive.is_dir() or not archive.is_relative_to(runtime_root):
        raise RuntimeError('cos_video_canary_must_be_under_project_runtime')
    control = json.loads((archive / 'control.json').read_text())
    output = Path(control['output']).resolve()
    supervisor = json.loads(
        (output / '_state/processes/ecot/supervisor-status.json').read_text()
    )
    if (supervisor.get('status') != 'complete' or supervisor.get('exit_code') != 0
            or supervisor.get('provider') != 'ark' or supervisor.get('model') != MODEL):
        raise RuntimeError('cos_video_canary_supervisor_not_successful')
    records = sorted((output / 'ecot/shards').glob('**/*.jsonl'))
    if len(records) != 1:
        raise RuntimeError('cos_video_canary_requires_one_final_record')
    rows = [json.loads(line) for line in records[0].read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('cos_video_canary_final_record_invalid')
    extension = rows[0].get('annotations', {}).get('robot_extension', {})
    requests = extension.get('requests') or []
    if (extension.get('annotation_task') != 'ecot' or extension.get('provider') != 'ark'
            or extension.get('model') != MODEL or not requests
            or any(request.get('video_transport', {}).get('method') != 'cos-presigned'
                   for request in requests)):
        raise RuntimeError('cos_video_canary_transport_evidence_invalid')
    log = (archive / 'supervisor.log').read_text(errors='replace')
    if ('canary_exit=0' not in log or 'cos_video_upload ' not in log
            or 'cos_video_cleanup_failed' in log):
        raise RuntimeError('cos_video_canary_log_evidence_invalid')
    pending = PROJECT / '_runtime/cos-ecot-videos/pending'
    if pending.is_dir() and any(pending.iterdir()):
        raise RuntimeError('cos_video_canary_cleanup_incomplete')
    return {'archive': str(archive), 'output': str(output),
            'worker_pid': supervisor['pid'], 'final_records': 1,
            'requests': len(requests), 'transport': 'cos-presigned'}


def archive_markers(markers, archive):
    """Preserve all markers before removing any, including across filesystems."""
    originals = {p: p.read_bytes() for p in markers}
    for p, content in originals.items():
        destination = archive/p.name
        with destination.open('wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.read_bytes() != content:
            raise RuntimeError('Marker archive verification failed')
    if any(p.read_bytes() != content for p, content in originals.items()):
        raise RuntimeError('Provider marker changed during archive')
    for p in markers:
        p.unlink()


def launch_worker(previous, archive, output, args, pid):
    certificate = Path(previous['source_drain_certificate'])
    spec = previous['tasks']['ecot']
    launch = dict(previous)
    launch.update(phase='ark_capacity_memory70g', session='vqa-ark-ecot2048',
                  capacity_settings=capacity_settings(args))
    launch['tasks'] = {'ecot': dict(spec, log=str(archive/'supervisor.log'),
                       started_at_unix=time.time(), previous_worker=pid)}
    atomic_json(archive/'launch.json', launch)
    argv = [sys.executable, '-u', str(PROJECT/'scripts/run_cosmos3_ark_ecot.py'),
            '--output', str(output), '--archive', str(archive), '--control', str(archive/'control.json'),
            '--source-drain-certificate', str(certificate), '--request-memory-mib', str(args.request_memory_mib),
            '--guard-gib', str(args.guard_gib), '--initial-pacing-seconds', str(args.initial_pacing_seconds)]
    for key, value in capacity_settings(args).items():
        if isinstance(value, bool):
            if value:
                argv += ['--' + key.replace('_', '-')]
        else:
            argv += ['--' + key.replace('_', '-'), str(value)]
    shell = shlex.join(argv)+' >> '+shlex.quote(str(archive/'supervisor.log'))+' 2>&1'
    subprocess.run(['tmux', 'new-window', '-d', '-t', 'vqa-ark-ecot2048', '-n', 'capacity-70g',
                    '-c', str(PROJECT), shell], check=True)
    monitor = [sys.executable, '-u', str(PROJECT/'scripts/monitor_independent_tasks.py'),
               '--archive', str(archive), '--interval', '60']
    monitor_target = ('vqa-ark-ecot2048:files-monitor.0' if args.media_transport != 'inline'
                      else 'vqa-ark-ecot2048:monitor.0')
    pane_pid = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t',
                      monitor_target, '#{pane_pid}'], text=True).strip())
    monitor_command = Path(f'/proc/{pane_pid}/cmdline').read_bytes().split(b'\0')
    if not any(p.endswith(b'/monitor_independent_tasks.py') for p in monitor_command):
        raise RuntimeError('Worker launched; monitor identity changed, update monitor manually')
    subprocess.run(['tmux', 'respawn-pane', '-k', '-t', monitor_target,
                    shlex.join(monitor)], check=True)
    print('ark_capacity_launched actual_concurrency_not_yet_verified=true', flush=True)


def resume_drained(args, archive):
    plan = json.loads((archive/'plan.json').read_text())
    previous = json.loads((args.previous_archive/'launch.json').read_text())
    output = Path(plan['output'])
    state = Path(previous['tasks']['ecot']['state'])
    pid = plan['source_pid']
    certificate_path = archive/'drain-verification.json'
    if not certificate_path.is_file():
        # A final-record outbox may outlive its old process: unlike checkpoint
        # rows it has a stable root and is explicitly replayed by the successor.
        # Reconstruct the handoff only after the old process and both writers
        # have exited and the original planned markers still match exactly.
        if same_live_process(pid, plan['original_start']):
            raise RuntimeError('Source worker is still draining')
        with ExitStack() as held:
            for name in ('supervisor.lock', 'full-run.lock'):
                for fd in acquire_writer_locks(output, name, 'ecot'):
                    held.callback(os.close, fd)
            runtime = json.loads((state/'runtime-config.json').read_text())
            if runtime['pid'] != pid or runtime['models']['ecot'] != MODEL:
                raise RuntimeError('Output writer changed before retained-outbox handoff')
            markers = [state/name for name in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json')]
            values = [json.loads(path.read_text()) for path in markers]
            if (values[0] != values[1] or values[0].get('worker_pid') != pid
                    or values[0].get('archive') != str(archive)
                    or values[0].get('kind') != 'planned_ark_ecot_capacity_upgrade'):
                raise RuntimeError('Retained-outbox marker identity mismatch')
            outbox = verify_empty_outbox(runtime, output, allow_pending_final=True)
            atomic_json(certificate_path, dict(
                source_pid=pid, source_workers_exited=True,
                writer_locks_verified=True, outbox=outbox,
                final_outbox_replay_required=bool(outbox.get('pending')),
                completed_at_unix=time.time(),
            ))
    certificate = json.loads(certificate_path.read_text())
    canary = (verify_cos_video_canary(args.cos_video_canary_archive)
              if plan.get('transport_canary') is not None else None)
    if (plan['ecot_memory_mib'] != args.request_memory_mib or plan['guard_gib'] != args.guard_gib
            or plan.get('capacity_settings', capacity_settings(args)) != capacity_settings(args)
            or plan.get('transport_canary') != canary
            or plan['initial_pacing_seconds'] != args.initial_pacing_seconds
            or plan.get('python_switch_interval_ms', 5.0) != args.python_switch_interval_ms
            or previous['tasks']['ecot']['output'] != str(output)
            or certificate.get('source_pid') != pid or not certificate.get('source_workers_exited')
            or not certificate.get('writer_locks_verified')
            or plan.get('final_publication_spool_root') != str(
                (args.local_spool_root / 'final-publication') if args.local_spool_root
                else (PROJECT / '_runtime' / 'ark-final-publication-outbox'))
            or same_live_process(pid, plan['original_start'])):
        raise RuntimeError('Drained handoff identity mismatch')
    with ExitStack() as held:
        for name in ('supervisor.lock', 'full-run.lock'):
            for fd in acquire_writer_locks(output, name, 'ecot'):
                held.callback(os.close, fd)
        runtime = json.loads((state/'runtime-config.json').read_text())
        if runtime['pid'] != pid or runtime['models']['ecot'] != MODEL:
            raise RuntimeError('Output writer has changed since drain')
        verify_empty_outbox(runtime, output, allow_pending_final=True)
        markers = [state/name for name in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json')]
        values = [json.loads(p.read_text()) for p in markers]
        if (values[0] != values[1] or values[0].get('worker_pid') != pid
                or values[0].get('archive') != str(archive)
                or values[0].get('kind') != 'planned_ark_ecot_capacity_upgrade'):
            raise RuntimeError('Only this reload operation may remove its markers')
        if not args.execute:
            print('ark_capacity_drained_resume_preflight_passed', flush=True)
            return
        archive_markers(markers, archive)
    launch_worker(previous, archive, output, args, pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-archive', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--request-memory-mib', type=int, default=71680)
    parser.add_argument('--guard-gib', type=int, default=92)
    parser.add_argument('--initial-pacing-seconds', type=float, default=.2)
    parser.add_argument('--python-switch-interval-ms', type=float, default=5.0)
    parser.add_argument('--media-transport', choices=('inline', 'ark-files', 'ark-files-cos-images', 'cos-presigned'), default='inline')
    parser.add_argument('--upload-workers', type=int, default=16)
    parser.add_argument('--media-temp-root', type=Path)
    parser.add_argument('--local-spool-root', type=Path)
    parser.add_argument('--cos-video-canary-archive', type=Path,
                        help='Successful isolated Ark/COS ECoT run required for Ark Files migration')
    parser.add_argument('--http-capacity', type=int, choices=(2048, 4096, 8192), default=2048)
    parser.add_argument('--frame-workers', type=int, default=2048)
    parser.add_argument('--prewarm-frame-workers', type=int)
    parser.add_argument('--resident-episodes', type=int, default=32)
    parser.add_argument('--prepare-workers', type=int, default=8)
    parser.add_argument('--prepare-codec-threads', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--batch-workers', type=int, choices=range(1, 65), default=8)
    parser.add_argument('--resume-validation-workers', type=int, choices=range(1, 33), default=4)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume-drained', action='store_true')
    parser.add_argument('--drain-only', action='store_true',
                        help='Keep own stop markers after verifying the empty outbox; do not launch')
    args = parser.parse_args()
    from run_cosmos3_ark_ecot import validate_local_spool_root, validate_media_temp_root
    args.media_temp_root = validate_media_temp_root(args.media_temp_root)
    args.local_spool_root = validate_local_spool_root(args.local_spool_root)
    archive = args.archive.resolve()
    if (archive.exists() and not args.resume_drained) or not archive.is_relative_to((PROJECT / '_runtime').resolve()):
        parser.error('Archive must be a new directory under project _runtime')
    if not 8 <= args.guard_gib <= effective_memory_bytes() / 1024**3 * .8:
        parser.error('Guard must retain at least 20 percent effective memory headroom')
    if not 0 < args.request_memory_mib < args.guard_gib * 1024:
        parser.error('Request memory must be positive and below the process guard')
    if (args.prewarm_frame_workers is not None
            and not 0 <= args.prewarm_frame_workers <= args.frame_workers):
        parser.error('Prewarmed frame workers must be between zero and frame workers')
    if not .1 <= args.python_switch_interval_ms <= 10:
        parser.error('Python switch interval must be between 0.1 and 10 milliseconds')
    if args.resume_drained:
        return resume_drained(args, archive)
    previous = json.loads((args.previous_archive / 'launch.json').read_text())
    if set(previous['tasks']) != {'ecot'}:
        parser.error('Only a single Ark ECoT task may be reloaded')
    spec = previous['tasks']['ecot']
    output, state = Path(spec['output']), Path(spec['state'])
    runtime = json.loads((state / 'runtime-config.json').read_text())
    supervisor = json.loads((state / 'supervisor-status.json').read_text())
    pid = supervisor['pid']
    start = inspect_worker(pid, output, 'ecot')
    process_args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    api_options = [i for i, value in enumerate(process_args) if value == b'--api']
    if (runtime.get('pid') != pid or supervisor.get('provider') != 'ark'
            or runtime['models']['ecot'] != MODEL or not api_options
            or process_args[api_options[-1]+1] != b'ark'):
        raise RuntimeError('Ark worker identity mismatch')
    certificate = Path(previous['source_drain_certificate'])
    if not certificate.is_file():
        raise RuntimeError('Original Qwen drain certificate is missing')
    markers = [state / name for name in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json')]
    if any(p.exists() for p in markers):
        raise RuntimeError('Existing provider marker requires inspection')
    command_args = SimpleNamespace(output=output, archive=archive, control=archive/'control.json',
                                  http_limit=2048, request_memory_mib=args.request_memory_mib,
                                  initial_pacing_seconds=args.initial_pacing_seconds,
                                  record_uid_path=None, dry_run=False, **capacity_settings(args))
    parsed = pipeline.parse_args(build_command(command_args)[3:])
    assert parsed.api == 'ark' and parsed.models['ecot'] == MODEL and parsed.tasks == {'ecot'}
    assert parsed.ecot_reuse_partial_checkpoints and parsed.ecot_interval == runtime['ecot_interval']
    old_transport = runtime.get('ecot_video_transport', 'inline')
    transport_canary = None
    if parsed.ecot_video_transport != old_transport:
        if old_transport == 'ark-files' and parsed.ecot_video_transport == 'cos-presigned':
            transport_canary = verify_cos_video_canary(args.cos_video_canary_archive)
        else:
            raise RuntimeError('Capacity reload must preserve the verified media transport')
    plan = dict(source_pid=pid, output=str(output), model=MODEL, http_limit=2048,
                final_publication_workers=parsed.final_publication_workers,
                final_publication_spool_root=str(parsed.final_publication_spool_root),
                final_publication_local_batch_size=parsed.final_publication_local_batch_size,
                final_publication_spool_max_mib=parsed.final_publication_spool_max_mib,
                ecot_memory_mib=args.request_memory_mib, guard_gib=args.guard_gib,
                initial_pacing_seconds=args.initial_pacing_seconds,
                python_switch_interval_ms=args.python_switch_interval_ms,
                capacity_settings=capacity_settings(args),
                transport_canary=transport_canary,
                source_drain_certificate=str(certificate), original_start=start)
    print('ark_capacity_preflight ' + json.dumps(plan), flush=True)
    if not args.execute:
        return
    archive.mkdir(mode=0o700)
    atomic_json(archive/'plan.json', plan)
    for name in ('runtime-config.json', 'supervisor-status.json', 'request-parallel-status.json'):
        shutil.copy2(state/name, archive/('before-'+name))
    control = dict(schema_version='qwen-http-control/v1', output=str(output), max_http_active=2048,
                   revision='ark-70g-capacity',
                   python_switch_interval_ms=args.python_switch_interval_ms)
    atomic_json(archive/'control.json', control)
    marker = dict(kind='planned_ark_ecot_capacity_upgrade', worker_pid=pid,
                  created_at_unix=time.time(), resume_authorized=True, archive=str(archive))
    for p in markers:
        atomic_json(p, marker)
    print('ark_capacity_drain_started force_kill=false source_results_preserved=true', flush=True)
    deadline = time.monotonic()+1800
    while same_live_process(pid, start):
        if time.monotonic() > deadline:
            raise RuntimeError('Drain timeout; no force kill or second writer')
        print('ark_capacity_drain_waiting', flush=True)
        time.sleep(30)
    with ExitStack() as held:
        while True:
            descriptors = []
            try:
                for name in ('supervisor.lock', 'full-run.lock'):
                    descriptors += acquire_writer_locks(output, name, 'ecot')
                for fd in descriptors:
                    held.callback(os.close, fd)
                break
            except BlockingIOError:
                for fd in descriptors:
                    os.close(fd)
                if time.monotonic() > deadline:
                    raise RuntimeError('Original writer locks remain held')
                time.sleep(5)
        outbox = verify_empty_outbox(runtime, output)
        if any(json.loads(p.read_text()) != marker for p in markers):
            raise RuntimeError('Provider marker changed during drain')
        atomic_json(archive/'drain-verification.json', dict(source_pid=pid, source_workers_exited=True,
                    writer_locks_verified=True, outbox=outbox, completed_at_unix=time.time()))
        for p in state.glob('rate-state-*.json'):
            shutil.copy2(p, archive/('before-'+p.name))
        if args.drain_only:
            print('ark_capacity_drained markers_preserved=true new_worker_started=false', flush=True)
            return
        archive_markers(markers, archive)
    launch_worker(previous, archive, output, args, pid)


if __name__ == '__main__':
    main()
