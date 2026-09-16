"""FIFO single-task HTTP admission with bounded, GIL-held native mutations."""
import argparse
from contextlib import contextmanager
import ctypes
from functools import lru_cache
import hashlib
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

from native_http_metrics import supported


class Node(ctypes.Structure):
    _fields_ = [('previous', ctypes.c_void_p), ('next', ctypes.c_void_p),
                ('status', ctypes.c_uint64), ('identifier', ctypes.c_uint64)]


class GateState(ctypes.Structure):
    _fields_ = [('head', ctypes.c_void_p), ('tail', ctypes.c_void_p)]+[
        (name, ctypes.c_uint64) for name in ('maximum', 'ecot_limit', 'active', 'peak', 'grants', 'queued')]


def library_path():
    source = Path(__file__).with_suffix('.c')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return source.parent.parent/'_runtime/native-ecot-gate'/f'gate-{digest}.so'


def build():
    if not supported():
        raise RuntimeError('native_ecot_gate_requires_gil_enabled_linux_cpython')
    target = library_path()
    if target.is_file():
        return target
    compiler = shutil.which('cc')
    if compiler is None:
        raise RuntimeError('native_ecot_gate_requires_c_compiler')
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
    library = ctypes.PyDLL(str(library_path()))
    for name in ('abi', 'size'):
        function = getattr(library, 'vqa_gate_'+name)
        function.argtypes, function.restype = [], ctypes.c_size_t
    library.vqa_gate_abi.restype = ctypes.c_int
    library.vqa_node_size.argtypes, library.vqa_node_size.restype = [], ctypes.c_size_t
    if (library.vqa_gate_abi() != 1 or library.vqa_gate_size() != ctypes.sizeof(GateState)
            or library.vqa_node_size() != ctypes.sizeof(Node)):
        raise RuntimeError('native_ecot_gate_abi_mismatch')
    gate, node = ctypes.POINTER(GateState), ctypes.POINTER(Node)
    for name in ('enter', 'leave'):
        function = getattr(library, 'vqa_gate_'+name)
        function.argtypes, function.restype = [gate, node], ctypes.c_uint64
    library.vqa_gate_limit.argtypes = [gate, ctypes.c_uint64, ctypes.c_uint64,
                                      ctypes.POINTER(ctypes.c_uint64)]
    library.vqa_gate_limit.restype = ctypes.c_size_t
    library.vqa_gate_copy.argtypes, library.vqa_gate_copy.restype = [gate, gate], None
    return library


class _Ticket:
    def __init__(self, identifier):
        self.node = Node(identifier=identifier)
        self.ready = threading.Lock()
        self.ready.acquire()


class NativeEcotNetworkSlots:
    """Only for independent ECoT; other tasks must use the mixed-task gate."""
    def __init__(self, maximum):
        self.library = load_library()
        if self.library is None:
            raise RuntimeError('native_ecot_gate_not_built')
        self.state = GateState()
        self.tickets = {}
        self.identifiers = itertools.count(1)
        self.set_limit(maximum)

    @property
    def maximum(self):
        return self.state.maximum

    def _notify(self, identifier):
        # Unique IDs prevent a delayed notification from targeting a new node
        # that reused a cancelled node's address. No raw pointer is dereferenced.
        ticket = self.tickets.get(identifier)
        if ticket is not None:
            ticket.ready.release()

    def set_limit(self, maximum, ecot_limit=None):
        if type(maximum) is not int or not 1 <= maximum <= 8192:
            raise ValueError('http_limit_must_be_1_to_8192')
        if ecot_limit is not None and (type(ecot_limit) is not int or not 1 <= ecot_limit <= maximum):
            raise ValueError('ecot_limit_must_fit_global_limit')
        # At most maximum slots can be newly granted. This buffer is used
        # only on explicit control changes, not on the request hot path.
        wakeups = (ctypes.c_uint64 * maximum)()
        count = self.library.vqa_gate_limit(ctypes.byref(self.state), maximum,
                                            ecot_limit or 0, wakeups)
        for identifier in wakeups[:count]:
            self._notify(identifier)

    @contextmanager
    def admit(self, task, check=None):
        if task != 'ecot':
            raise ValueError('native_ecot_gate_rejects_other_tasks')
        if check is not None:
            check()
        identifier = next(self.identifiers)
        if identifier >= 2**64:
            raise RuntimeError('native_ecot_gate_ticket_ids_exhausted')
        ticket = _Ticket(identifier)
        # Keep the node alive while C's intrusive queue can reference it.
        self.tickets[identifier] = ticket
        try:
            wakeup = self.library.vqa_gate_enter(ctypes.byref(self.state), ctypes.byref(ticket.node))
            if wakeup and wakeup != identifier:
                self._notify(wakeup)
            while ticket.node.status != 2:
                ticket.ready.acquire(timeout=.5)
                if check is not None:
                    check()
            if check is not None:
                check()
            yield
        finally:
            wakeup = self.library.vqa_gate_leave(ctypes.byref(self.state), ctypes.byref(ticket.node))
            self.tickets.pop(identifier, None)
            if wakeup:
                self._notify(wakeup)

    def snapshot(self):
        state = GateState()
        self.library.vqa_gate_copy(ctypes.byref(self.state), ctypes.byref(state))
        return {'http_cap': state.maximum, 'active_requests': state.active,
                'network_grant_mode': 'gil-held-native-ecot-fifo/v1',
                'peak_active_requests': state.peak, 'active_by_task': {'ecot': state.active},
                'active_by_lane': {'ecot': state.active, 'grd': 0},
                'queued_requests': state.queued, 'queued_by_lane': {'ecot': state.queued, 'grd': 0},
                'ecot_http_ceiling': state.ecot_limit or state.maximum,
                'ecot_contention_share': max(1, state.maximum//4),
                'work_conserving': not state.ecot_limit or state.ecot_limit >= state.maximum,
                'http_attempts_admitted': {'ecot': state.grants}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', required=True)
    parser.parse_args()
    print('native_ecot_gate_built path=' + str(build()))
