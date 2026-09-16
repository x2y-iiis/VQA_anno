"""Explicit, read-only reuse of final ECoT records across provider switches.

This module never rewrites records or imports partial checkpoints. The original
provider/model remains authoritative in its original output namespace.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat


@dataclass(frozen=True)
class EcotPriorOutput:
    root: Path
    provider: str
    model: str

    def as_dict(self):
        return dict(root=str(self.root), provider=self.provider, model=self.model)


def parse_sources(values, output: Path, tasks) -> list[EcotPriorOutput]:
    if values and set(tasks) != {'ecot'}:
        raise ValueError('ecot_reuse_output_requires_ecot_only')
    sources = []
    output = output.resolve()
    for raw_root, provider, model in values:
        root = Path(raw_root).resolve()
        if provider not in {'ark', 'dashscope'} or not model.strip():
            raise ValueError('ecot_reuse_output_requires_explicit_provider_and_model')
        if (not root.is_dir() or root == output or root.is_relative_to(output)
                or output.is_relative_to(root)):
            raise ValueError('ecot_reuse_output_requires_separate_existing_source_root')
        if not (root / 'ecot/shards').is_dir():
            raise ValueError('ecot_reuse_output_requires_native_ecot_shards')
        source = EcotPriorOutput(root, provider, model)
        if source not in sources:
            sources.append(source)
    return sources


def completed_uids(sources, source_key, label, interval, compatible_uid):
    """Validate both final layouts against their declared original identity.

    I/O failures propagate: an unreadable source must not silently trigger
    duplicate inference. Malformed/incompatible records are not successes.
    """
    result = set()
    for source in sources:
        if not stat.S_ISDIR((source.root / 'ecot/shards').stat().st_mode):
            raise ValueError('ecot_reuse_source_shards_no_longer_a_directory')
        legacy = source.root / 'ecot/shards' / source_key / f'{label}.jsonl'
        try:
            candidates = [legacy] if stat.S_ISREG(legacy.stat().st_mode) else []
        except FileNotFoundError:
            candidates = []
        # Path.glob may suppress directory access failures. Only a genuinely
        # absent batch is empty; denied/failed reads must stop the transition.
        try:
            with os.scandir(legacy.with_suffix('.records')) as entries:
                candidates += sorted(Path(entry.path) for entry in entries
                                     if entry.name.endswith('.jsonl') and not entry.name.startswith('.'))
        except FileNotFoundError:
            pass
        for path in candidates:
            with path.open(encoding='utf-8', buffering=4 * 1024 * 1024) as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        uid = compatible_uid(record, 'ecot', source.model, source.provider)
                        if uid is None:
                            continue
                        annotation = record['annotations']['robot_extension']
                        if annotation['result'].get('ecot_interval') != interval:
                            continue
                    except (ValueError, KeyError, TypeError):
                        continue
                    result.add(uid)
    return result
