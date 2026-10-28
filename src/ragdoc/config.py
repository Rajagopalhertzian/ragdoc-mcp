"""Runtime configuration. Everything is read lazily from the environment so tests can override it."""
from __future__ import annotations

import os
import re
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
STATE_DIRNAME = ".ragdoc"


class Limits:
    """Size, rate and time limits applied to every call."""

    @staticmethod
    def max_file_bytes() -> int:
        return int(os.getenv("RAGDOC_MAX_FILE_MB", "200")) * 1024 * 1024

    @staticmethod
    def max_files() -> int:
        return int(os.getenv("RAGDOC_MAX_FILES", "20000"))

    @staticmethod
    def timeout_s() -> float:
        return float(os.getenv("RAGDOC_TIMEOUT_S", "1500"))

    max_pdf_pages = 5000
    max_query_chars = 2000
    max_k = 100
    max_tokens_cap = 8000
    max_eval_n = 200
    max_cases = 200
    rate_calls = 120
    rate_window_s = 10.0


def base_dir() -> Path:
    """Only folders under this directory are readable. Defaults to the server's working directory."""
    return Path(os.getenv("RAGDOC_BASE_DIR") or Path.cwd()).expanduser().resolve()


def state_dir() -> Path:
    return base_dir() / STATE_DIRNAME / "state"


def collections_dir() -> Path:
    return base_dir() / STATE_DIRNAME / "collections"


def shadow_dir() -> Path:
    return base_dir() / STATE_DIRNAME / "shadow"
