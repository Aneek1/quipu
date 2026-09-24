"""Filesystem helpers shared by every module that swaps a temp file into place.

Windows (via antivirus or the search indexer) can hold a brief, transient lock on
a file right after it's written, which turns an otherwise-safe `os.replace` into a
`PermissionError` ([WinError 5]) even though nothing is actually wrong. Measured on
this machine: ~0.75% of back-to-back replaces failed this way, which over a
2,861-step, ~23-hour run is near-certain to hit at least once. A short retry with
backoff absorbs it without masking a real, persistent permission problem.
"""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(
    src: str | Path, dst: str | Path, attempts: int = 6, base_delay: float = 0.05
) -> None:
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * 2**i)
