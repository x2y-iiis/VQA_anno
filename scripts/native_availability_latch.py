"""Optional Linux generation latch; one kernel broadcast releases all followers."""
import argparse
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def library_path():
    source = Path(__file__).with_suffix('.c')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return source.parent.parent/'_runtime/native-availability'/f'latch-{digest}.so'


def build():
    if sys.platform != 'linux':
        raise RuntimeError('native_availability_latch_requires_linux')
    target = library_path()
    if target.is_file():
        return target
    compiler = shutil.which('cc')
    if compiler is None:
        raise RuntimeError('native_availability_latch_requires_c_compiler')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.build-', suffix='.so', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        subprocess.run([compiler, '-O2', '-std=c11', '-D_GNU_SOURCE', '-fPIC', '-shared',
                        '-Wall', '-Wextra', '-Werror', str(Path(__file__).with_suffix('.c')),
                        '-o', str(temporary)], check=True, timeout=60)
        # Content-addressed destination and atomic publication avoid changing
        # an already-mapped shared library underneath a running worker.
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


@lru_cache(maxsize=1)
def load_library():
    if sys.platform != 'linux' or not library_path().is_file():
        return None
    library = ctypes.CDLL(str(library_path()))
    library.vqa_latch_abi.argtypes = []
    library.vqa_latch_abi.restype = ctypes.c_int
    if library.vqa_latch_abi() != 1 or ctypes.sizeof(ctypes.c_uint32) != 4:
        raise RuntimeError('native_availability_latch_abi_mismatch')
    pointer = ctypes.POINTER(ctypes.c_uint32)
    library.vqa_latch_wait.argtypes = [pointer, ctypes.c_int]
    library.vqa_latch_wait.restype = ctypes.c_int
    library.vqa_latch_notify.argtypes = [pointer]
    library.vqa_latch_notify.restype = ctypes.c_int
    return library


class NativeRefreshLatch:
    def __init__(self):
        self.library = load_library()
        if self.library is None:
            raise RuntimeError('native_availability_latch_not_built')
        self.word = ctypes.c_uint32(0)
        self.waiters = 0

    def __len__(self):
        return self.waiters

    def notify(self):
        result = self.library.vqa_latch_notify(ctypes.byref(self.word))
        if result < 0:
            raise OSError(-result, 'native_availability_notify_failed')

    def wait(self):
        # CDLL releases the GIL while the kernel waits. Periodic return keeps
        # Python signal handling responsive; a timeout never means success.
        while True:
            result = self.library.vqa_latch_wait(ctypes.byref(self.word), 1000)
            if result < 0:
                raise OSError(-result, 'native_availability_wait_failed')
            if result:
                return


def benchmark(waiters):
    from concurrent.futures import ThreadPoolExecutor
    import statistics
    import time
    if not 1 <= waiters <= 4096:
        raise ValueError('native_latch_benchmark_waiters_must_be_1_to_4096')
    latch = NativeRefreshLatch()
    ready = [False]*waiters
    def wait(index):
        ready[index] = True
        latch.wait()
        return time.monotonic()
    with ThreadPoolExecutor(waiters, thread_name_prefix='latch-benchmark') as pool:
        try:
            futures = [pool.submit(wait, index) for index in range(waiters)]
            deadline = time.monotonic()+30
            while not all(ready):
                if time.monotonic() > deadline:
                    raise TimeoutError('native_latch_benchmark_startup_timeout')
                time.sleep(.005)
            started = time.monotonic()
            latch.notify()
            notify_seconds = time.monotonic()-started
            latencies = sorted(future.result(timeout=30)-started for future in futures)
        finally:
            latch.notify()
    return {'schema': 'native-availability-broadcast-benchmark/v1', 'waiters': waiters,
            'notify_seconds': notify_seconds, 'first_waiter_seconds': latencies[0],
            'median_waiter_seconds': statistics.median(latencies),
            'p95_waiter_seconds': latencies[min(len(latencies)-1, int(len(latencies)*.95))],
            'last_waiter_seconds': latencies[-1], 'model_requests': 0,
            'caveat': 'Local notification test, not production HTTP concurrency evidence'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--build', action='store_true')
    action.add_argument('--benchmark', action='store_true')
    parser.add_argument('--waiters', type=int, default=2048)
    args = parser.parse_args()
    if args.build:
        print('native_availability_latch_built path=' + str(build()))
    else:
        import json
        print(json.dumps(benchmark(args.waiters), indent=2))
