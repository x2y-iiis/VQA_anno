#!/usr/bin/env python3
"""Audit task-stage record identity and contract coverage without decoding media."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from validate_annotation_result_contract import validate_annotation_result_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--task', required=True,
        choices=('subtask', 'ecot', 'grd', 'sta', 'cpa'),
    )
    parser.add_argument('--provider', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--contract-id', required=True)
    parser.add_argument('--expected-records', required=True, type=int)
    parser.add_argument('--source-key')
    parser.add_argument(
        '--expected-source-counts',
        help='JSON object mapping source_key to its expected record count',
    )
    parser.add_argument('--max-error-examples', type=int, default=20)
    args = parser.parse_args()
    expected_source_counts = None
    if args.expected_source_counts:
        try:
            expected_source_counts = {
                str(key): int(value)
                for key, value in json.loads(args.expected_source_counts).items()
            }
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f'invalid --expected-source-counts JSON object: {error}')

    root = args.output/args.task/'shards'
    if args.source_key:
        root = root/args.source_key
    files = sorted(root.rglob('*.jsonl')) if root.is_dir() else []
    seen: set[str] = set()
    errors: Counter[str] = Counter()
    examples = []
    by_source: Counter[str] = Counter()
    records = 0
    for path in files:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                records += 1
                try:
                    record = json.loads(line)
                    extension = record['annotations']['robot_extension']
                    annotation = record['provenance']['annotation']
                    uid = str(record['provenance']['input_record_uid'])
                    failures = []
                    if extension.get('annotation_task') != args.task:
                        failures.append('wrong_annotation_task')
                    if extension.get('provider') != args.provider:
                        failures.append('wrong_provider')
                    if extension.get('model') != args.model:
                        failures.append('wrong_model')
                    if annotation.get('contract_id') != args.contract_id:
                        failures.append('wrong_contract')
                    result = extension.get('result')
                    failures.extend(validate_annotation_result_contract(
                        args.task, result, extension.get('requests'),
                    ).elements())
                    if not uid:
                        failures.append('empty_input_uid')
                    elif uid in seen:
                        failures.append('duplicate_input_uid')
                    else:
                        seen.add(uid)
                    by_source[str(record.get('source_key') or 'unknown')] += 1
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    failures = [f'invalid_record:{type(error).__name__}']
                for failure in failures:
                    errors[failure] += 1
                if failures and len(examples) < args.max_error_examples:
                    examples.append({
                        'file': str(path), 'line': line_number,
                        'errors': failures,
                    })
    if records != args.expected_records:
        errors['unexpected_record_count'] += 1
    if len(seen) != args.expected_records:
        errors['unexpected_unique_uid_count'] += 1
    actual_source_counts = dict(sorted(by_source.items()))
    if expected_source_counts is not None and actual_source_counts != dict(
        sorted(expected_source_counts.items())
    ):
        errors['unexpected_source_counts'] += 1
    summary = {
        'schema_version': 'vqa-annotation-stage-audit/v2',
        'task': args.task,
        'provider': args.provider,
        'model': args.model,
        'contract_id': args.contract_id,
        'path': str(root.resolve()),
        'files': len(files),
        'records': records,
        'unique_input_uids': len(seen),
        'expected_records': args.expected_records,
        'valid': not errors,
        'by_source': actual_source_counts,
        'expected_source_counts': expected_source_counts,
        'error_counts': dict(sorted(errors.items())),
        'error_examples': examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['valid'] else 1


if __name__ == '__main__':
    sys.exit(main())
