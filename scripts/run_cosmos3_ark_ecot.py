"""Supervise an isolated Ark ECoT consumer with the existing request scheduler."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
import time

from annotation_memory_profile import effective_memory_bytes
from run_cosmos3_qwen2048 import command as downstream_command, ARK_OUTPUT, QWEN_OUTPUT, PROJECT
from supervise_cosmos3_seed_stable import (
    process_tree_rss, guard_memory_snapshot, terminate_process_group, write_status,
)
from task_process_state import acquire_writer_locks, state_directory

DEFAULT_MODEL = 'doubao-seed-2-0-lite-260215'
# Backward-compatible export for launch tests and external diagnostics.
MODEL = DEFAULT_MODEL


def validate_media_temp_root(path):
    if path is None:
        return None
    root = Path(path).resolve()
    if not root.is_dir() or root in {Path('/'), Path('/mnt'), Path('/run/ti'), PROJECT}:
        raise ValueError('media_temp_root_requires_dedicated_existing_directory')
    if shutil.disk_usage(root).free < 128 * 1024**3:
        raise ValueError('media_temp_root_requires_128GiB_free')
    with tempfile.TemporaryFile(dir=root) as stream:
        stream.write(b'vqa-media-scratch-probe\n')
        stream.flush()
        os.fsync(stream.fileno())
    return root


def validate_local_spool_root(path):
    """Require durable local outboxes to use a spacious dedicated data disk."""
    if path is None:
        return None
    root = Path(path).resolve()
    if not root.is_dir() or root in {Path('/'), Path('/mnt'), Path('/run/ti'), PROJECT}:
        raise ValueError('local_spool_root_requires_dedicated_existing_directory')
    if shutil.disk_usage(root).free < 128 * 1024**3:
        raise ValueError('local_spool_root_requires_128GiB_free')
    with tempfile.TemporaryFile(dir=root) as stream:
        stream.write(b'vqa-local-spool-probe\n')
        stream.flush()
        os.fsync(stream.fileno())
    return root


def build_command(args):
    provider = getattr(args, 'provider', 'ark')
    model = getattr(args, 'model', DEFAULT_MODEL)
    capacity = getattr(args, 'http_capacity', None) or args.http_limit
    frame_workers = getattr(args, 'frame_workers', None) or capacity
    prewarm_frame_workers = getattr(args, 'prewarm_frame_workers', None)
    if prewarm_frame_workers is None:
        prewarm_frame_workers = min(frame_workers, capacity)
    resident_episodes = getattr(args, 'resident_episodes', 32)
    native_media = getattr(args, 'media_transport', 'inline') in ('ark-files', 'ark-files-cos-images', 'cos-presigned')
    if provider == 'las' and getattr(args, 'media_transport', 'inline') != 'cos-presigned':
        raise ValueError('las_ecot_requires_cos_presigned_media')
    # Expanded episode supply relies on sparse target-frame caching and compact
    # media references; keep the legacy bound for inline/small-memory profiles.
    # Short episodes can exhaust the resident-record supply before the HTTP
    # ceiling. This expansion retains the existing 70 GiB request budget and
    # process guard; it does not increase per-request media or sampling density.
    resident_limit = 2048 if native_media and args.request_memory_mib >= 71680 else 128
    # HTTP admission and local media preparation are deliberately independent:
    # thousands of HTTP slots do not require thousands of Python decode threads.
    # Keeping the frame pool bounded avoids GIL/scheduler collapse while queued
    # requests wait for the provider.
    if (capacity < args.http_limit or frame_workers < 1
            or not 0 <= prewarm_frame_workers <= frame_workers
            or not 1 <= resident_episodes <= resident_limit):
        raise ValueError('invalid_ark_ecot_worker_capacity')
    if (capacity > 2048 or frame_workers > 2048) and getattr(args, 'media_transport', 'inline') not in ('ark-files', 'ark-files-cos-images', 'cos-presigned'):
        raise ValueError('expanded_ark_ecot_capacity_requires_native_files')
    command = downstream_command(args.output, ARK_OUTPUT, True, 'ecot')
    local_spool_root = getattr(args, 'local_spool_root', None)
    checkpoint_spool_root = (
        Path(local_spool_root) / 'checkpoints'
        if local_spool_root is not None else args.archive / 'checkpoint-spool'
    )
    final_spool_root = (
        Path(local_spool_root) / 'final-publication'
        if local_spool_root is not None else PROJECT / '_runtime' / 'ark-final-publication-outbox'
    )
    # The ECoT-only path transcodes its episode representation in
    # EcotMediaFactory under video_prepare_workers.  The generic clip process
    # pool serves GRD/STA clip_video_path calls and would only pre-spawn idle
    # Python processes here.
    while '--video-clip-process-pool' in command:
        command.remove('--video-clip-process-pool')
    command += [
        # Keep the existing model and annotation identity while routing the
        # request through the documented LAS Submit/Poll operator.
        '--api', 'las' if provider == 'las' else 'ark',
        '--api-key-env', 'LAS_API_KEY' if provider == 'las' else 'ARK_API_KEY',
        '--endpoint', ('https://operator.las.cn-beijing.volces.com/api/v1/submit'
                       if provider == 'las'
                       else 'https://ark.cn-beijing.volces.com/api/v3/chat/completions'),
        '--ecot-model', model, '--request-timeout-seconds', '300',
        '--max-http-active', str(capacity), '--max-total-http-active', str(args.http_limit),
        '--shared-frame-workers', str(frame_workers),
        '--prewarm-shared-frame-workers', str(prewarm_frame_workers),
        '--thread-stack-kib', str(getattr(args, 'thread_stack_kib', 0)),
        # The inherited base command carries a 2048 pending-job ceiling.  It
        # must scale with native-URL HTTP capacity or expanded profiles never
        # have enough request producers to fill their admission window.
        '--max-pending', str(capacity),
        '--max-record-active', str(resident_episodes), '--stage-prefetch', '2',
        '--batch-workers', str(getattr(args, 'batch_workers', 8)),
        '--video-prepare-workers', str(getattr(args, 'prepare_workers', 8)),
        '--video-prepare-codec-threads', str(getattr(args, 'prepare_codec_threads', 4)),
        '--request-working-memory-mib', str(args.request_memory_mib + 4096),
        '--ecot-request-memory-mib', str(args.request_memory_mib),
        '--fair-http-control', str(args.control),
        '--resume-validation-workers', str(getattr(args, 'resume_validation_workers', 4)),
        '--prioritize-failed-records',
        '--checkpoint-spool-root', str(checkpoint_spool_root),
        # Smaller commit cohorts avoid waking hundreds of blocked Python
        # callers at once.  The production contention benchmark is faster at
        # 64 than 512 while retaining FULL-synchronous local durability.
        '--checkpoint-sync-workers', '256', '--checkpoint-local-batch-size', '64',
        '--final-publication-workers', str(128 if native_media and args.request_memory_mib >= 71680 else 8),
        '--resume-output', str(ARK_OUTPUT),
        '--ecot-reuse-output', str(QWEN_OUTPUT), 'dashscope', 'qwen3.8-flash',
    ]
    if native_media and args.request_memory_mib >= 71680:
        command += [
            '--final-publication-spool-root',
            str(final_spool_root),
            '--final-publication-spool-max-mib', '8192',
            '--final-publication-spool-max-files', '50000',
            '--final-publication-local-batch-size', '64',
            '--retain-unit-checkpoints-after-final',
        ]
        if getattr(args, 'media_transport', 'inline') == 'cos-presigned':
            command += ['--durable-cloud-transport', 'cos-direct']
    if args.record_uid_path:
        command += ['--record-uid-path', str(args.record_uid_path)]
    else:
        command += ['--ecot-reuse-partial-checkpoints']
    if getattr(args, 'initial_pacing_seconds', None) is not None:
        command += ['--request-warm-start-interval', str(args.initial_pacing_seconds)]
    if getattr(args, 'media_transport', 'inline') in ('ark-files', 'ark-files-cos-images', 'cos-presigned'):
        images = 'ark-files' if args.media_transport == 'ark-files' else 'cos-presigned'
        videos = 'cos-presigned' if args.media_transport == 'cos-presigned' else 'ark-files'
        command += ['--ecot-video-transport', videos, '--ecot-image-transport', images,
                    '--ark-upload-workers', str(getattr(args, 'upload_workers', 16))]
        if provider == 'las' and getattr(args, 'las_source_video_direct', False):
            command += ['--ecot-las-source-video-direct']
        command += ['--fair-http-capacity', str(max(2048, capacity)),
                    '--ecot-memory-request-slots', str(max(2048, capacity, frame_workers))]
    if getattr(args, 'canary_no_reuse', False):
        if not args.record_uid_path:
            raise ValueError('canary_no_reuse_requires_explicit_uid_filter')
        for flag, count in (('--resume-output', 1), ('--ecot-reuse-output', 3)):
            while flag in command:
                index = command.index(flag)
                del command[index:index + count + 1]
    if args.dry_run:
        command += ['--dry-run']
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider', choices=('ark', 'las'), default='ark')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help='ECoT model passed to the Ark-backed LAS operator')
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--record-uid-path', type=Path)
    parser.add_argument('--source-drain-certificate', type=Path)
    parser.add_argument('--http-limit', type=int, default=2048)
    parser.add_argument('--http-capacity', type=int, choices=(2048, 4096, 8192),
                        help='Model ceiling; fair control retains the lower initial HTTP limit')
    parser.add_argument('--frame-workers', type=int,
                        help='Allow upload/preparation/retry workers beyond active HTTP slots')
    parser.add_argument('--prewarm-frame-workers', type=int,
                        help='Eagerly start this many shared frame workers; zero keeps startup lazy')
    parser.add_argument('--thread-stack-kib', type=int, default=0,
                        help='Native stack size for high-concurrency Python workers')
    parser.add_argument('--resident-episodes', type=int, default=32)
    parser.add_argument('--prepare-workers', type=int, default=8)
    parser.add_argument('--prepare-codec-threads', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--batch-workers', type=int, choices=range(1, 65), default=8,
                        help='Parallel catalog batches used to discover sparse unfinished episodes')
    parser.add_argument('--resume-validation-workers', type=int, choices=range(1, 33), default=4,
                        help='Read-only workers validating existing immutable result records')
    parser.add_argument('--request-memory-mib', type=int, default=20480)
    parser.add_argument('--guard-gib', type=int, default=36)
    parser.add_argument('--memory-guard-restart-delay-seconds', type=float, default=30,
                        help='Delay before restarting an unchanged worker after a memory guard stop')
    parser.add_argument('--max-recoverable-restarts', type=int, default=3,
                        help='Bound non-memory worker restarts; prevents permanent tail failures from looping')
    parser.add_argument('--initial-pacing-seconds', type=float)
    parser.add_argument('--media-transport', choices=('inline', 'ark-files', 'ark-files-cos-images', 'cos-presigned'), default='inline')
    parser.add_argument('--las-source-video-direct', action='store_true',
                        help='Upload the source MP4 and let LAS perform 0.5 FPS sampling')
    parser.add_argument('--upload-workers', type=int, default=16)
    parser.add_argument('--media-temp-root', type=Path,
                        help='Dedicated large local scratch directory for decoded media')
    parser.add_argument('--local-spool-root', type=Path,
                        help='Dedicated large local data directory for durable SQLite outboxes')
    parser.add_argument('--canary-no-reuse', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.media_temp_root = validate_media_temp_root(args.media_temp_root)
    args.local_spool_root = validate_local_spool_root(args.local_spool_root)
    args.output, args.archive, args.control = (
        value.resolve() for value in (args.output, args.archive, args.control))
    if (args.output in {ARK_OUTPUT.resolve(), QWEN_OUTPUT.resolve()}
            or not args.archive.is_relative_to((PROJECT / '_runtime').resolve())):
        parser.error('Ark output must be isolated and archive must be under project _runtime')
    if not 1 <= args.http_limit <= 8192 or args.request_memory_mib <= 0:
        parser.error('Invalid request limits')
    if not 8 <= args.guard_gib <= effective_memory_bytes() / 1024**3 * .8:
        parser.error('Guard must retain at least 20 percent container memory headroom')
    if not 0 <= args.memory_guard_restart_delay_seconds <= 300:
        parser.error('Memory guard restart delay must be between 0 and 300 seconds')
    if not 0 <= args.max_recoverable_restarts <= 100:
        parser.error('Max recoverable restarts must be between 0 and 100')
    command = build_command(args)
    if args.dry_run:
        return subprocess.run(command, check=True).returncode
    if args.record_uid_path is None:
        if args.source_drain_certificate is None:
            parser.error('Full scope requires a verified source drain certificate')
        certificate = json.loads(args.source_drain_certificate.read_text())
        if not (certificate.get('source_output') == str(QWEN_OUTPUT)
                and certificate.get('source_workers_exited') is True
                and certificate.get('writer_locks_verified') is True
                and certificate.get('outbox', {}).get('pending') == 0):
            raise SystemExit('Invalid source drain certificate.')
    args.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = state_directory(args.output, 'ecot')
    state.mkdir(parents=True, exist_ok=True)
    for marker in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json'):
        if (state / marker).exists() or (args.output / '_state' / marker).exists():
            raise SystemExit('Provider marker requires inspection before restart.')
    control = json.loads(args.control.read_text())
    if (control.get('output') != str(args.output)
            or control.get('max_http_active') != args.http_limit):
        raise SystemExit('Control identity or initial HTTP cap mismatch.')
    if args.worker:
        key_env = 'LAS_API_KEY' if args.provider == 'las' else 'ARK_API_KEY'
        key_file_env = 'LAS_KEY_FILE' if args.provider == 'las' else 'ARK_KEY_FILE'
        default_key_file = ('/mnt/doubao_las_annotation/.runtime/las_api_key'
                            if args.provider == 'las'
                            else '/mnt/doubao_las_annotation/.runtime/ark_api_key')
        key = os.environ.get(key_env) or Path(os.environ.get(
            key_file_env, default_key_file)).read_text().strip()
        if not key:
            raise SystemExit(f'Missing {args.provider.upper()} credential.')
        os.environ[key_env] = key
        if args.provider == 'las':
            ark_key = os.environ.get('ARK_API_KEY') or Path(os.environ.get(
                'ARK_KEY_FILE', '/mnt/doubao_las_annotation/.runtime/ark_api_key')).read_text().strip()
            if not ark_key:
                raise SystemExit('Missing ARK credential for LAS customer-Ark mode.')
            os.environ['ARK_API_KEY'] = ark_key
        descriptors = acquire_writer_locks(args.output, 'full-run.lock', 'ecot')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = '1'
        os.environ.update(MALLOC_ARENA_MAX='2', VQA_OPENCV_DECODER_THREADS='1',
                          OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;1')
        if args.media_temp_root is not None:
            os.environ['TMPDIR'] = str(args.media_temp_root)
            print('ark_media_temp_root=' + str(args.media_temp_root), flush=True)
        bypass = list(dict.fromkeys(part.strip() for name in ('no_proxy', 'NO_PROXY')
                                    for part in os.environ.get(name, '').split(',') if part.strip()))
        bypass.extend(domain for domain in ('.myqcloud.com', '.volces.com') if domain not in bypass)
        os.environ.update(no_proxy=','.join(bypass), NO_PROXY=','.join(bypass))
        for name in list(os.environ):
            if name.startswith('VQA_DASHSCOPE_'):
                os.environ.pop(name)
        resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**3, 384 * 1024**3))
        os.execv(sys.executable, command)
    with ExitStack() as held:
        if args.record_uid_path is None:
            # Keep the old ECoT namespace fenced for the entire Ark run.
            for name in ('supervisor.lock', 'full-run.lock'):
                for descriptor in acquire_writer_locks(QWEN_OUTPUT, name, 'ecot'):
                    held.callback(os.close, descriptor)
            from reload_independent_workers import verify_empty_outbox
            previous_runtime = json.loads((state_directory(QWEN_OUTPUT, 'ecot') / 'runtime-config.json').read_text())
            verify_empty_outbox(previous_runtime, QWEN_OUTPUT)
        for descriptor in acquire_writer_locks(args.output, 'supervisor.lock', 'ecot'):
            held.callback(os.close, descriptor)
        def interrupted(signum, frame):
            raise SystemExit(f'Ark ECoT supervisor interrupted by signal {signum}.')
        # tmux pane replacement closes the controlling terminal with SIGHUP.
        # Treat it like an explicit stop so ExitStack terminates the detached
        # worker process group instead of leaving an orphan writer behind.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            prior = signal.signal(sig, interrupted)
            held.callback(signal.signal, sig, prior)
        supervisor_started = time.time()
        memory_guard_restarts = 0
        recoverable_restarts = 0
        while True:
            for marker in ('provider-fatal-stop.json', 'grd-review-fatal-stop.json'):
                if (state / marker).exists() or (args.output / '_state' / marker).exists():
                    raise SystemExit('Provider marker requires inspection before restart.')
            worker = subprocess.Popen(
                [sys.executable, '-u', __file__, *sys.argv[1:], '--worker'],
                start_new_session=True,
            )
            held.callback(terminate_process_group, worker)
            started = time.time()
            peak = 0
            status = {}
            memory_guard_hit = False
            while worker.poll() is None:
                rss = process_tree_rss(worker.pid)
                # A busy spawn/fork wave can make per-process smaps_rollup PSS
                # snapshots overlap in time and grossly over-count the live tree.
                # Confirm the guard value against this worker's memory cgroup so
                # shared/transient address spaces cannot trigger a false stop.
                memory = guard_memory_snapshot(
                    worker.pid, rss, confirm_cgroup=True,
                )
                peak = max(peak, memory['guard_memory_bytes'])
                status = dict(status='running', pid=worker.pid, tasks=['ecot'], provider='ark',
                              request_gateway=args.provider,
                              model=args.model, started_at_unix=started,
                              supervisor_started_at_unix=supervisor_started,
                              memory_guard_restarts=memory_guard_restarts,
                              recoverable_restarts=recoverable_restarts,
                              updated_at=datetime.now(timezone.utc).isoformat(),
                              rss_bytes=rss, peak_guard_memory_bytes=peak,
                              memory_guard_gib=args.guard_gib, **memory)
                write_status(state / 'supervisor-status.json', status)
                print('supervisor_heartbeat ' + json.dumps(status), flush=True)
                if memory['guard_memory_bytes'] > args.guard_gib * 1024**3:
                    status['reason'] = f'{args.provider}_ecot_memory_guard'
                    memory_guard_hit = True
                    terminate_process_group(worker)
                    break
                try:
                    worker.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    pass
            status.update(status='complete' if worker.returncode == 0 else 'stopped',
                          exit_code=worker.returncode, updated_at=datetime.now(timezone.utc).isoformat())
            if memory_guard_hit:
                memory_guard_restarts += 1
                status.update(
                    status='restarting',
                    memory_guard_restarts=memory_guard_restarts,
                    recoverable_restarts=recoverable_restarts,
                    restart_delay_seconds=args.memory_guard_restart_delay_seconds,
                )
                write_status(state / 'supervisor-status.json', status)
                print('supervisor_memory_guard_restart ' + json.dumps(status), flush=True)
                time.sleep(args.memory_guard_restart_delay_seconds)
                continue
            write_status(state / 'supervisor-status.json', status)
            print('supervisor_exit ' + json.dumps(status), flush=True)
            if worker.returncode:
                # Fatal provider states are persisted as markers and are
                # rejected at the top of the next iteration.  Other non-zero
                # exits commonly mean that a first pass durably completed most
                # records but retained recoverable per-record failures.  Resume
                # the same immutable configuration so completed UIDs are
                # skipped and failed UIDs are retried first.
                recoverable_restarts += 1
                if recoverable_restarts > args.max_recoverable_restarts:
                    status.update(
                        status='terminal_failure', reason='recoverable_restart_budget_exhausted',
                        memory_guard_restarts=memory_guard_restarts,
                        recoverable_restarts=recoverable_restarts,
                        max_recoverable_restarts=args.max_recoverable_restarts,
                        record_uid_path=(str(args.record_uid_path)
                                         if args.record_uid_path is not None else None),
                    )
                    write_status(state / 'supervisor-status.json', status)
                    (state / 'terminal-failure.json').write_text(
                        json.dumps(status, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8',
                    )
                    print('supervisor_terminal_failure ' + json.dumps(status), flush=True)
                    raise SystemExit(worker.returncode)
                status.update(
                    status='restarting', reason='recoverable_worker_exit',
                    memory_guard_restarts=memory_guard_restarts,
                    recoverable_restarts=recoverable_restarts,
                    restart_delay_seconds=args.memory_guard_restart_delay_seconds,
                )
                write_status(state / 'supervisor-status.json', status)
                print('supervisor_recoverable_restart ' + json.dumps(status), flush=True)
                time.sleep(args.memory_guard_restart_delay_seconds)
                continue
            return


if __name__ == '__main__':
    main()
