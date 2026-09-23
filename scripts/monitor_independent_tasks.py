"""Native, per-process progress for independent ECoT, GRD, and STA workers."""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import time

from request_parallel import atomic_json
from run_cosmos3_qwen2048 import ARK_OUTPUT, PROJECT


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def process_alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in {'Z','X'}
    except OSError:
        return False


def observe(task, spec, history):
    root = Path(spec['state'])
    supervisor = read(root/'supervisor-status.json')
    metrics = read(root/'request-parallel-status.json')
    runtime = read(root/'runtime-config.json')
    pid = supervisor.get('pid')
    matched = (pid is not None and metrics.get('pid') == pid
               and metrics.get('updated_at_unix',0) >= spec.get('started_at_unix',0))
    fresh = matched and time.time()-metrics.get('updated_at_unix',0) < 90
    current = {'time':metrics.get('updated_at_unix',0), 'pid':pid,
               'units':metrics.get('durable_units_committed',{}).get(task,0) if matched else 0}
    if fresh and (not history or history[-1]['time'] != current['time']):
        if history and (history[-1]['pid'] != pid or history[-1]['units'] > current['units']):
            history.clear()
        history.append(current)
    samples = [x for x in history if x['time'] >= current['time']-900]
    seconds = current['time']-samples[0]['time'] if samples else 0
    rate = (current['units']-samples[0]['units'])*60/seconds if fresh and seconds >= 60 else None
    live = process_alive(pid)
    identity_ok = runtime.get('pid') == pid and runtime.get('tasks') == [task]
    return {'pid':pid, 'alive':live, 'status':supervisor.get('status','starting'),
            'runtime_identity_valid':identity_ok, 'fresh_heartbeat':fresh,
            'heartbeat_age_seconds':time.time()-metrics.get('updated_at_unix',time.time()),
            'active_http':metrics.get('active_requests') if fresh else None,
            'peak_http':metrics.get('peak_active_requests') if matched else None,
            'http_cap':metrics.get('http_cap') if matched else spec['configured_http_cap'],
            'durable_units':current['units'], 'durable_units_per_minute':rate,
            'window_seconds':seconds, 'guard_memory_gib':supervisor.get('guard_memory_bytes',0)/1024**3,
            'metrics':metrics}


def checkpoint_status_line(metrics, fresh):
    spool = metrics.get('checkpoint_sync')
    if not spool:
        return None
    if not fresh:
        return '  CHECKPOINT: stale snapshot; current queues unavailable'
    return (f'  CHECKPOINT: local_waiting={spool.get("local_queued_files", 0):,} '
            f'cloud_pending={spool.get("pending_files", 0):,} '
            f'cloud_acks_waiting={spool.get("queued_cloud_acknowledgements", 0)} '
            f'sync_errors={spool.get("sync_errors", 0)} '
            f'db={spool.get("database_writer_mode", "legacy")}')


def availability_status_line(metrics, fresh):
    check = metrics.get('availability_check')
    if not check:
        return None
    if not fresh:
        return '  AVAILABILITY: stale snapshot; current waits unavailable'
    return (f'  AVAILABILITY: refreshes={check["refreshes"]} '
            f'waiters={check["pending_waiters"]} '
            f'check_last/max={check["last_check_seconds"]:.3f}/{check["max_check_seconds"]:.3f}s '
            f'notify_last/max={check["last_notify_seconds"]:.3f}/{check["max_notify_seconds"]:.3f}s '
            f'backend={check.get("mode", "unknown")}')


def publication_status_line(metrics, fresh):
    value = metrics.get('final_publication')
    if not value:
        return None
    if not fresh:
        return '  PUBLICATION: stale snapshot; current waits unavailable'
    def seconds(key):
        number = value.get(key)
        return '--' if number is None else f'{number:.3f}s'
    return (f'  PUBLICATION: active={value["active"]}/{value["workers"]} '
            f'waiting={value["waiting"]} completed={value["completed"]} failed={value["failed"]} '
            f'wait_median={seconds("wait_median_seconds")} write_median={seconds("write_median_seconds")}')


def final_outbox_status_line(metrics, fresh):
    value = metrics.get('final_registration_outbox')
    if not value:
        return None
    if not fresh:
        return '  FINAL OUTBOX: stale snapshot; current queues unavailable'
    return (f'  FINAL OUTBOX: local_waiting={value.get("local_queued_files", 0):,} '
            f'cloud_pending={value.get("pending_files", 0):,} '
            f'uploading={value.get("active_cloud_writers", 0)}/'
            f'{value.get("cloud_writer_limit", 0)} '
            f'sync_errors={value.get("sync_errors", 0)} '
            f'oldest={value.get("oldest_pending_seconds", 0):.1f}s')


def presigned_media_status_line(client, fresh):
    """Expose URL signing and provider-side media fetch failures separately."""
    refreshes = client.get('presigned_media_refreshes')
    errors = client.get('presigned_media_fetch_errors')
    if refreshes is None and errors is None:
        return None
    if not fresh:
        return '  PRESIGNED MEDIA: stale snapshot; current counters unavailable'
    return (f'  PRESIGNED MEDIA: url_signatures={refreshes or 0:,} '
            f'fetch_errors={errors or 0:,}')


def stage_status_line(metrics, fresh):
    value = metrics.get('stage_pipeline')
    if not value:
        return None
    if not fresh:
        return '  STAGE: stale snapshot; current pipeline state unavailable'
    lane_total = lambda key: sum(value.get(key, {}).values())
    return (f'  STAGE: submitted={value.get("submitted", 0):,} '
            f'running={lane_total("running_by_lane"):,} '
            f'prepared={lane_total("prepared_by_lane"):,} '
            f'pending={lane_total("pending_by_lane"):,} '
            f'preparing={value.get("preparing_active", 0):,} '
            f'completed={value.get("completed", 0):,} '
            f'failed={value.get("failed", 0):,}')


def resume_validation_status_line(metrics, fresh):
    value = metrics.get('resume_validation')
    if not value:
        return None
    if not fresh:
        return '  RESUME VALIDATION: stale snapshot; current workers unavailable'
    return (f'  RESUME VALIDATION: reported_workers={len(value.get("worker_pids", []))}/'
            f'{value.get("configured_workers", 0)}')


def stability_status_lines(archive):
    report = read(archive / 'stability-report.json')
    if report:
        state = 'PASSED' if report.get('passed') else 'FAILED'
        rate = report.get('durable_units_per_minute')
        rate_text = '--' if rate is None else f'{rate:,.1f}/min'
        return [f'  STABILITY GATE: {state} rate={rate_text} '
                f'window={report.get("observation_seconds", 0):.0f}s '
                f'reasons={",".join(report.get("reasons") or []) or "none"}']
    progress = read(archive / 'stability-progress.json')
    if not progress:
        return []
    phase = progress.get('phase', 'unknown')
    last = progress.get('last', {})
    if phase == 'waiting_for_first_durable_result':
        return [f'  STABILITY GATE: waiting_for_production_ramp '
                f'durable={last.get("durable_units", 0):,}/'
                f'{progress.get("minimum_start_durable_units", "--")} '
                f'http_peak={last.get("http_peak", 0)}/'
                f'{progress.get("minimum_start_peak_http", "--")}']
    baseline = progress.get('baseline', {})
    elapsed = progress.get('elapsed_seconds', 0)
    units = last.get('durable_units', 0) - baseline.get('durable_units', 0)
    rate = units * 60 / elapsed if elapsed > 0 else 0
    return [f'  STABILITY GATE: observing elapsed={elapsed:.0f}/'
            f'{progress.get("target_seconds", 0):.0f}s rate={rate:,.1f}/min '
            f'url_signatures={last.get("url_signatures", 0) - baseline.get("url_signatures", 0):,} '
            f'media_fetch_errors={last.get("media_fetch_errors", 0) - baseline.get("media_fetch_errors", 0):,}']


def promotion_status_lines(archive):
    observation = read(archive/'loaded-window-report.json')
    first = read(archive/'online-promotion-3072/decision.json')
    chain = read(archive/'online-promotion-4096/chain-status.json')
    rows = []
    if observation:
        rows.append(f'  CAPACITY GATE 2048->3072: phase={observation.get("phase", "unknown")} '
                    f'samples={observation.get("sample_count", 0)}')
    if first:
        state = ('applied' if first.get('executed') else
                 'passed-not-applied' if first.get('allowed') else 'held')
        rows.append(f'  CAPACITY DECISION 3072: {state} '
                    f'reasons={",".join(first.get("reasons") or []) or "none"}')
    if chain:
        rows.append(f'  CAPACITY GATE 3072->4096: status={chain.get("status", "unknown")}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.interval < 30:
        parser.error('monitor_interval_must_be_at_least_30_seconds')
    launch = read(args.archive/'launch.json')
    las_audit = read(Path(launch['las_completion_audit'])) if launch.get('las_completion_audit') else {}
    histories = {task:deque(maxlen=64) for task in launch['tasks']}
    while True:
        snapshot = {'updated_at_unix':time.time(), 'tasks':{}}
        rows = ['INDEPENDENT TASKS | '+datetime.now().astimezone().isoformat(timespec='seconds'),
                'TASK    PID       STATE       HTTP/CAP     NEW CHECKPOINTS     RECENT / MIN   MEMORY GiB']
        for task, spec in launch['tasks'].items():
            value = observe(task, spec, histories[task])
            snapshot['tasks'][task] = value
            state = ('RUNNING' if value['fresh_heartbeat'] and value['runtime_identity_valid'] else 'STARTING') if value['alive'] else 'STOPPED'
            rate = '--' if value['durable_units_per_minute'] is None else f'{value["durable_units_per_minute"]:.1f}'
            http = f'{value["active_http"] if value["active_http"] is not None else "--"}/{value["http_cap"] or 2048}'
            rows.append(f'{task.upper():<7} {str(value["pid"] or "--"):<9} {state:<11} {http:<12} '
                        f'{value["durable_units"]:>15,} {rate:>16} {value["guard_memory_gib"]:>12.1f}')
            checkpoint_line = checkpoint_status_line(value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if checkpoint_line:
                rows.append(checkpoint_line)
            publication_line = publication_status_line(value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if publication_line:
                rows.append(publication_line)
            final_outbox_line = final_outbox_status_line(
                value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if final_outbox_line:
                rows.append(final_outbox_line)
            availability_line = availability_status_line(value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if availability_line:
                rows.append(availability_line)
            stage_line = stage_status_line(
                value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if stage_line:
                rows.append(stage_line)
            validation_line = resume_validation_status_line(
                value['metrics'], value['fresh_heartbeat'] and value['alive'])
            if validation_line:
                rows.append(validation_line)
            models = value['metrics'].get('model_admission',{})
            for group, clients in models.items():
                for model, client in clients.items():
                    rows.append(f'  {group}/{model}: throttles={client.get("rate_limit_events",0)} '
                                f'pacing={client.get("request_start_interval_seconds",0):.3f}s')
                    media_line = presigned_media_status_line(
                        client, value['fresh_heartbeat'] and value['alive'])
                    if media_line:
                        rows.append(media_line)
                    transport = client.get('http_transport')
                    if transport:
                        active = transport['active'] if value['fresh_heartbeat'] and value['alive'] else '--'
                        rows.append(f'  HTTP CALLS: active={active} peak={transport["peak"]} '
                                    f'started={transport["started"]:,} finished={transport["finished"]:,} '
                                    f'failed={transport["failed"]:,} (excludes admission waits)')
                        duration = transport.get('recent_call_median_seconds')
                        if duration is not None:
                            rows.append(f'  HTTP CALL DURATION: recent median={duration:.3f}s '
                                        '(connect/upload/body-read only; not provider execution time)')
                    for label, key in (('ARK FILES', 'ark_files'), ('COS VIDEOS', 'cos_videos'),
                                       ('COS IMAGES', 'cos_images')):
                        media = client.get(key)
                        if media:
                            rows.append(f'  {label}: uploads={media.get("uploads",0):,} '
                                f'active={media.get("active_uploads",0)}/{media.get("workers",media.get("upload_workers",0))} '
                                f'queued={media.get("waiting_uploads",0)} '
                                f'deleted={media.get("deleted",media.get("deleted_files",0)):,} '
                                f'cleanup_errors={media.get("cleanup_errors",0)}')
        las = read(ARK_OUTPUT/'_state/supervisor-status.json')
        remote_las = launch.get('las_remote', False)
        snapshot['las'] = {'pid':las.get('pid'), 'alive':None if remote_las else process_alive(las.get('pid')),
                           'status':'remote_not_locally_verified' if remote_las else las.get('status')}
        las_display = 'REMOTE (not verified from local PID)' if remote_las else ('RUNNING' if snapshot['las']['alive'] else 'STOPPED')
        audited_complete = (not remote_las and not snapshot['las']['alive']
                            and las.get('status') == 'complete'
                            and las_audit.get('saved_episodes_union') == 319355
                            and las_audit.get('remaining_missing_uids') == [])
        if audited_complete:
            las_display = 'COMPLETE'
            snapshot['las']['verified_episodes'] = las_audit['saved_episodes_union']
            snapshot['las']['verified_video_hours'] = las_audit['catalog_video_hours_union']
        rows += [f'LAS     {las.get("pid")} | {las_display} | configuration unchanged',
                 'CPA     DISABLED (STA still uses final contact auditing)',
                 'Counters are per-worker checkpoints, not complete episodes; STA includes intermediate contact/audit units.',
                 'HTTP occupancy is admission accounting, not proof of provider-side inference concurrency.']
        if audited_complete:
            rows.append(f'LAS verified saved episodes: {las_audit["saved_episodes_union"]:,} / 319,355 | '
                        f'video hours: {las_audit["catalog_video_hours_union"]:,.2f} | all reused on final resume')
        elif not remote_las:
            progress=read(PROJECT/'_runtime/las2048-paced-20260909-r3/live-progress.json')
            if progress.get('pid')==las.get('pid'):
                snapshot['las_progress']=progress
                rows.append(f'LAS saved episodes: {progress.get("completed_episodes", "--")} / '
                            f'{progress.get("scope_episodes", "--")} | recent episodes/min: '
                            f'{progress.get("recent_episodes_per_minute", "--")} | snapshot: {progress.get("time")}')
        for task,status in launch.get('handed_off_tasks',{}).items():
            rows.append(f'{task.upper()}     HANDED OFF | {status} | not running locally')
        rows += stability_status_lines(args.archive)
        rows += promotion_status_lines(args.archive)
        atomic_json(args.archive/'latest.json', snapshot)
        with (args.archive/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(snapshot)+'\n')
        print('\033[2J\033[H'+'\n'.join(rows), flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
