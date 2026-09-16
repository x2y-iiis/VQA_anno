"""Task-scoped runtime state with exclusion against legacy combined writers."""
import fcntl
import os
from pathlib import Path


def check_handoff_ownership(tasks, project=None):
    root = Path(project) if project is not None else Path(__file__).resolve().parents[1]
    if set(tasks) & {'grd', 'sta'} and (root/'_runtime/grd-sta-handoff-fence.json').exists():
        raise RuntimeError('local_grd_sta_ownership_handed_off_inspect_source_fence')


def state_directory(output, task=None):
    root = Path(output).resolve() / '_state'
    if task is None:
        return root
    if task not in {'ecot', 'grd', 'sta'}:
        raise ValueError('unsupported_independent_task')
    return root / 'processes' / task


def acquire_writer_locks(output, name, task=None):
    """Hold a shared legacy barrier and an exclusive task lock until exec/exit.

    A legacy combined writer holds the barrier exclusively. Two independent
    tasks may coexist, but a second writer for either task cannot start.
    """
    if name not in {'full-run.lock', 'supervisor.lock'} and not name.startswith('full-run-shard-') and not name.endswith('.lock'):
        raise ValueError('unsupported_writer_lock')
    root, state = state_directory(output), state_directory(output, task)
    state.mkdir(parents=True, exist_ok=True)
    descriptors = []
    try:
        targets = [(root / name, fcntl.LOCK_SH if task else fcntl.LOCK_EX)]
        if task:
            targets.append((state / name, fcntl.LOCK_EX))
        for path, mode in targets:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
            os.set_inheritable(descriptor, True)
        return descriptors
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise
