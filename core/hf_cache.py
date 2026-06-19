from __future__ import annotations

import os
from pathlib import Path


def configure_huggingface_cache() -> None:
    """Use a workspace-local cache when the caller did not choose one."""

    cache_root = Path.cwd() / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hub"))
