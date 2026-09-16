#!/usr/bin/env python3
"""Select a VLA-compatible global task name from Cosmos3 catalog fields."""

from __future__ import annotations

import re


VLA_TASK_NAME_MAX_LENGTH = 80


def select_vla_task_name(
    task_description: object,
    task_name: object,
) -> str | None:
    """Prefer a complete description, then a valid short catalog task name."""
    normalized = [
        re.sub(r'\s+', ' ', str(value or '')).strip()
        for value in (task_description, task_name)
    ]
    for value in normalized:
        if value and len(value) <= VLA_TASK_NAME_MAX_LENGTH:
            return value
    fallback = next((value for value in normalized if value), '')
    return fallback[:VLA_TASK_NAME_MAX_LENGTH].strip() or None
