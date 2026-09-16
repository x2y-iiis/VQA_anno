"""Isolate temporary-upload credential I/O from annotation interpreter contention.

The private UNIX socket never receives an API key. The child inherits the same
environment and calls the fixed DashScope endpoint, using the existing on-disk
namespace pacing protocol. No upload or model-inference retries are added.
"""
import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

MAX_MESSAGE_BYTES = 256 * 1024


def send_message(connection, value):
    payload = json.dumps(value, separators=(',', ':')).encode() + b'\n'
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError('policy_ipc_message_too_large')
    connection.sendall(payload)


def receive_message(connection):
    payload = bytearray()
    while len(payload) <= MAX_MESSAGE_BYTES:
        chunk = connection.recv(min(16384, MAX_MESSAGE_BYTES + 1 - len(payload)))
        if not chunk:
            raise OSError('policy_ipc_connection_closed')
        payload.extend(chunk)
        if b'\n' in chunk:
            line, remainder = payload.split(b'\n', 1)
            if remainder or len(line) >= MAX_MESSAGE_BYTES:
                raise ValueError('policy_ipc_invalid_message')
            return json.loads(line)
    raise ValueError('policy_ipc_message_too_large')


class PolicyProcess:
    """One parent-owned, bounded credential worker; no automatic crash restart."""

    def __init__(self, root, api_key_env, policy_qps, capacity=64, test_endpoint=None):
        self.root = Path(root).resolve()
        self.api_key_env = api_key_env
        self.policy_qps = policy_qps
        self.capacity = min(64, capacity)
        if self.capacity < 1 or not 0 < policy_qps <= 90:
            raise ValueError('invalid_policy_process_capacity_or_rate')
        if test_endpoint and not test_endpoint.startswith('http://127.0.0.1:'):
            raise ValueError('test_policy_endpoint_must_be_loopback')
        self.test_endpoint = test_endpoint
        self.start_lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(self.capacity)
        self.process = None
        self.directory = None
        self.closed = False
        self.ready = False
        self._last_snapshot = {}
        self._snapshot_at = 0
        atexit.register(self.close)

    def start(self):
        # Publish readiness only after initialization. Steady-state callers must
        # not serialize behind the startup mutex for every credential request.
        if self.closed:
            raise OSError('policy_worker_closed')
        if self.ready:
            if self.process.poll() is not None:
                raise OSError('policy_worker_exited')
            return
        with self.start_lock:
            if self.closed:
                raise OSError('policy_worker_closed')
            if self.process is not None:
                if self.process.poll() is not None:
                    raise OSError('policy_worker_exited')
                if not self.ready:
                    raise OSError('policy_worker_start_incomplete')
                return
            self.directory = tempfile.TemporaryDirectory(prefix='vqa-policy-')
            self.socket_path = Path(self.directory.name)/'worker.sock'
            command = [sys.executable, str(Path(__file__).resolve()),
                       '--socket', str(self.socket_path), '--root', str(self.root),
                       '--api-key-env', self.api_key_env, '--qps', str(self.policy_qps),
                       '--capacity', str(self.capacity), '--parent-pid', str(os.getpid())]
            if self.test_endpoint:
                command += ['--test-endpoint', self.test_endpoint]
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                            close_fds=True)
            deadline = time.monotonic() + 15
            while not self.socket_path.exists():
                if self.process.poll() is not None:
                    raise OSError('policy_worker_start_failed')
                if time.monotonic() >= deadline:
                    raise OSError('policy_worker_start_timeout')
                time.sleep(.02)
            self.ready = True

    def _request(self, value, timeout=90):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.socket_path))
            send_message(connection, value)
            return receive_message(connection)

    def get_policy(self, model, namespace, metrics):
        self.start()
        with metrics.phase('policy_queue'):
            self.slots.acquire()
        try:
            with metrics.phase('policy_rpc'):
                reply = self._request({'op': 'get_policy', 'model': model, 'namespace': namespace})
            if 'error' in reply:
                if reply['error'] == 'invalid_policy':
                    raise ValueError('policy_worker_invalid_policy')
                raise OSError('policy_worker_' + reply['error'])
            return reply['status'], reply.get('policy')
        finally:
            self.slots.release()

    def snapshot(self, force=False):
        process = self.process
        if process is None:
            return {'started': False}
        if process.poll() is not None:
            return {'started': True, 'pid': process.pid, 'alive': False}
        if force or time.monotonic() - self._snapshot_at >= 5:
            try:
                self._last_snapshot = self._request({'op': 'metrics'}, timeout=2)
                self._snapshot_at = time.monotonic()
            except (OSError, ValueError):
                return {'started': True, 'pid': process.pid, 'alive': True,
                        'metrics_unavailable': True}
        return {'started': True, 'pid': process.pid, 'alive': True, **self._last_snapshot}

    def close(self):
        with self.start_lock:
            if self.closed:
                return
            self.closed = True
            self.ready = False
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.directory is not None:
                self.directory.cleanup()


def serve(args):
    # A lost annotation parent must not leave a credential-issuing orphan.
    # PDEATHSIG is tied to the creating *thread* on Linux, which can retire
    # while its process is still live. Observe actual process parentage instead.
    if os.getppid() != args.parent_pid:
        raise RuntimeError('policy_worker_parent_changed')
    def watch_parent():
        while True:
            time.sleep(1)
            if os.getppid() != args.parent_pid:
                os._exit(0)
    threading.Thread(target=watch_parent, daemon=True, name='policy-parent-watch').start()
    from dashscope_temporary_video import DashScopeTemporaryPublisher, UPLOAD_ENDPOINT
    from temporary_upload_pool import SessionPool, UploadMetrics
    publisher = DashScopeTemporaryPublisher(args.root, args.api_key_env, policy_qps=args.qps)
    sessions = SessionPool(args.capacity)
    metrics = UploadMetrics()
    endpoint = args.test_endpoint or UPLOAD_ENDPOINT
    if args.test_endpoint and not args.test_endpoint.startswith('http://127.0.0.1:'):
        raise ValueError('test_policy_endpoint_must_be_loopback')
    pending_slots = threading.BoundedSemaphore(args.capacity)

    def acquire(connection, message):
        reply = {'error': 'acquisition_failed'}
        try:
            root, key = publisher._namespace(message['model'])
            if root.name != message['namespace']:
                raise ValueError('namespace_mismatch')
            with metrics.phase('policy_pacing'):
                publisher._pace(root)
            with sessions.lease() as session, metrics.phase('policy_http'):
                if args.test_endpoint:
                    session.trust_env = False
                with session.get(endpoint, headers={'Authorization': f'Bearer {key}',
                                 'Content-Type': 'application/json'},
                                 params={'action': 'getPolicy', 'model': message['model']},
                                 timeout=(15, 60), allow_redirects=False) as response:
                    reply = {'status': response.status_code}
                    if response.status_code == 200:
                        reply['policy'] = response.json()['data']
        except (ValueError, KeyError, TypeError):
            reply = {'error': 'invalid_policy'}
        except Exception:
            # Never return response bodies, keys, exception text, or policy signatures.
            reply = {'error': 'acquisition_failed'}
        finally:
            # Release admission before the parent can receive the reply and
            # submit a replacement; otherwise a valid caller can see queue_full.
            pending_slots.release()
            try:
                send_message(connection, reply)
            except (OSError, ValueError):
                pass
            connection.close()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener, ThreadPoolExecutor(args.capacity) as pool:
        listener.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        listener.listen(128)
        while True:
            connection, _ = listener.accept()
            connection.settimeout(90)
            try:
                message = receive_message(connection)
                if message.get('op') == 'metrics':
                    send_message(connection, {'phases': metrics.snapshot(), 'capacity': args.capacity,
                                              'policy_qps': args.qps})
                    connection.close()
                elif (message.get('op') == 'get_policy' and isinstance(message.get('model'), str)
                      and isinstance(message.get('namespace'), str)):
                    if pending_slots.acquire(blocking=False):
                        pool.submit(acquire, connection, message)
                    else:
                        send_message(connection, {'error': 'queue_full'})
                        connection.close()
                else:
                    send_message(connection, {'error': 'invalid_request'})
                    connection.close()
            except (OSError, ValueError, TypeError):
                connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--api-key-env', required=True)
    parser.add_argument('--qps', type=float, required=True)
    parser.add_argument('--capacity', type=int, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--test-endpoint')
    args = parser.parse_args()
    if not 1 <= args.capacity <= 64 or not 0 < args.qps <= 90:
        parser.error('invalid_capacity_or_rate')
    serve(args)


if __name__ == '__main__':
    main()
