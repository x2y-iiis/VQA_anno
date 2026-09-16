#!/usr/bin/env python3
"""Validate local runtime dependencies without sending network requests."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures: list[str] = []
    if sys.version_info < (3, 10):
        failures.append(f"python>=3.10 required; found {sys.version.split()[0]}")

    for module in (
        "boto3", "cv2", "httpx", "numpy", "openai", "PIL", "pyarrow", "requests",
    ):
        try:
            importlib.import_module(module)
        except Exception as error:  # pragma: no cover - depends on host setup
            failures.append(f"python module unavailable: {module}: {error}")

    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            failures.append(f"executable unavailable: {executable}")

    vendored = PROJECT_ROOT / "third_party" / "doubao_las_annotation" / "las_annotation"
    if not (vendored / "pipeline.py").is_file():
        failures.append(f"vendored LAS source unavailable: {vendored}")

    configured = {
        name: bool(os.environ.get(name))
        for name in ("LAS_API_KEY", "ARK_API_KEY", "LAS_COS_URI_PREFIX")
    }
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1

    print(f"OK python={sys.version.split()[0]}")
    print("OK ffmpeg, ffprobe, and core Python modules")
    print("OK vendored doubao_las_annotation")
    print("CONFIG " + " ".join(
        f"{name}={'set' if present else 'unset'}"
        for name, present in configured.items()
    ))
    print("No network request was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
