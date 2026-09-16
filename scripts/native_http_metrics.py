"""Short GIL-held native counter updates; never use CDLL for these functions."""
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


class CounterState(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ('active', 'peak', 'started', 'finished', 'failed')]+[
                    ('completed_seconds', ctypes.c_double),
                    ('recent_count', ctypes.c_uint64),
                    ('recent', ctypes.c_double * 2048)]


def supported():
    return (sys.platform == 'linux' and sys.implementation.name == 'cpython'
            and getattr(sys, '_is_gil_enabled', lambda: True)())


def library_path():
    source = Path(__file__).with_suffix('.c')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return source.parent.parent/'_runtime/native-http-metrics'/f'counters-{digest}.so'


def build():
    if not supported():
        raise RuntimeError('native_http_metrics_requires_gil_enabled_linux_cpython')
    target = library_path()
    if target.is_file():
        return target
    compiler = shutil.which('cc')
    if compiler is None:
        raise RuntimeError('native_http_metrics_requires_c_compiler')
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix='.build-', suffix='.so', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        subprocess.run([compiler, '-O2', '-std=c11', '-fPIC', '-shared', '-Wall', '-Wextra',
                        '-Werror', str(Path(__file__).with_suffix('.c')), '-o', str(temporary)],
                       check=True, timeout=60)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


@lru_cache(maxsize=1)
def load_library():
    if not supported() or not library_path().is_file():
        return None
    # PyDLL retains the GIL throughout each bounded C update and struct copy.
    library = ctypes.PyDLL(str(library_path()))
    library.vqa_http_metrics_abi.argtypes = []
    library.vqa_http_metrics_abi.restype = ctypes.c_int
    library.vqa_http_metrics_size.argtypes = []
    library.vqa_http_metrics_size.restype = ctypes.c_size_t
    if library.vqa_http_metrics_abi() != 1 or library.vqa_http_metrics_size() != ctypes.sizeof(CounterState):
        raise RuntimeError('native_http_metrics_abi_mismatch')
    pointer = ctypes.POINTER(CounterState)
    for name, args in [('start', [pointer]), ('finish', [pointer, ctypes.c_int, ctypes.c_double]),
                       ('copy', [pointer, pointer])]:
        function = getattr(library, 'vqa_http_metrics_'+name)
        function.argtypes, function.restype = args, None
    return library


class NativeHTTPCounters:
    def __init__(self):
        self.library = load_library()
        if self.library is None:
            raise RuntimeError('native_http_metrics_not_built')
        self.state = CounterState()

    def start(self):
        self.library.vqa_http_metrics_start(ctypes.byref(self.state))

    def finish(self, failed, elapsed):
        self.library.vqa_http_metrics_finish(ctypes.byref(self.state), int(failed), elapsed)

    def snapshot(self):
        copied = CounterState()
        self.library.vqa_http_metrics_copy(ctypes.byref(self.state), ctypes.byref(copied))
        return ({key: getattr(copied, key) for key in ('active', 'peak', 'started', 'finished', 'failed')}
                | {'completed_call_seconds': copied.completed_seconds},
                list(copied.recent[:copied.recent_count]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    print('native_http_metrics_built path=' + str(build()))
