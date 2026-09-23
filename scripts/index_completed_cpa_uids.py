#!/usr/bin/env python3
"""Build a durable UID index from immutable CPA final-record objects."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uid-path', type=Path, required=True)
    parser.add_argument('--cpa-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cos-prefix',
                        help='Use one authoritative recursive COS listing instead of a FUSE walk')
    args = parser.parse_args()

    digest_to_uid = {}
    with args.uid_path.open() as stream:
        for line in stream:
            uid = line.strip()
            if uid:
                digest_to_uid[hashlib.sha256(uid.encode()).hexdigest()] = uid
    completed = set()
    files = packs = 0
    started = time.monotonic()
    if args.cos_prefix:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from cos_ecot_images import BUCKET, make_client
        match = re.fullmatch(r'cos://([^/]+)/(.+)', args.cos_prefix.rstrip('/'))
        if match is None or match.group(1) != BUCKET:
            parser.error('--cos-prefix must use the configured production bucket')
        client = make_client(32)
        prefix = match.group(2).rstrip('/') + '/'
        def listed_names():
            token = None
            while True:
                request = {'Bucket': BUCKET, 'Prefix': prefix, 'MaxKeys': 1000}
                if token:
                    request['ContinuationToken'] = token
                response = client.list_objects_v2(**request)
                for item in response.get('Contents', []):
                    name = Path(item['Key']).name
                    if re.fullmatch(r'[0-9a-f]{64}\.jsonl', name):
                        yield None, name
                if not response.get('IsTruncated'):
                    return
                token = response['NextContinuationToken']
        names_and_directories = listed_names()
        process = None
    else:
        names_and_directories = (
            (directory, name)
            for directory, _, names in os.walk(args.cpa_root)
            for name in names if name.endswith('.jsonl')
        )
        process = None
    for directory, name in names_and_directories:
        if not name.endswith('.jsonl'):
            continue
        files += 1
        stem = name[:-6]
        uid = digest_to_uid.get(stem)
        if uid is not None:
            completed.add(uid)
        elif stem.startswith('pack-') and directory is not None:
            packs += 1
            with (Path(directory) / name).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_uid = str(record.get('uid') or record.get('source', {}).get('uid') or '')
                    if record_uid:
                        completed.add(record_uid)
        if files % 1000 == 0:
            print(f'completed_uid_scan files={files} uids={len(completed)} elapsed_seconds={time.monotonic()-started:.1f}', flush=True)
    if process is not None and process.wait() != 0:
        raise RuntimeError(f'coscli_listing_failed:{process.returncode}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=args.output.name + '.', dir=args.output.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            for uid in sorted(completed):
                stream.write(uid + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, args.output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(f'completed_uid_scan_complete files={files} packs={packs} uids={len(completed)} elapsed_seconds={time.monotonic()-started:.1f}', flush=True)


if __name__ == '__main__':
    main()
