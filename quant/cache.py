"""Filesystem cache location resolution."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """Returns the data cache directory, creating it if missing.

    Defaults to <repo>/data/cache/. Override with QUANT_CACHE_DIR.
    """
    override = os.environ.get("QUANT_CACHE_DIR")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parent.parent / "data" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
