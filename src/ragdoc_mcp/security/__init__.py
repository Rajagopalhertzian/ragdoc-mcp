"""Security utilities for RAG Doctor MCP - path allowlisting, untrusted text handling."""

import os
import re
from pathlib import Path
from typing import Optional


class PathSecurityError(Exception):
    """Raised when a path violates security policies."""
    pass


class PathAllowlist:
    """Manages allowed paths for file operations."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()
        self._allowed_dirs: list[Path] = [self.base_dir]

    def add_allowed_dir(self, path: Path) -> None:
        """Add an additional allowed directory (must be under base_dir)."""
        resolved = path.resolve()
        if not self._is_under_base(resolved):
            raise PathSecurityError(f"Directory {path} is not under base directory {self.base_dir}")
        self._allowed_dirs.append(resolved)

    def validate_read(self, path: Path) -> Path:
        """Validate and resolve a path for reading."""
        return self._validate_path(path, "read")

    def validate_write(self, path: Path) -> Path:
        """Validate and resolve a path for writing (must be under collection dir)."""
        return self._validate_path(path, "write")

    def _validate_path(self, path: Path, operation: str) -> Path:
        """Validate a path against the allowlist."""
        if not path.is_absolute():
            path = Path.cwd() / path

        # Resolve symlinks and normalize
        try:
            resolved = path.resolve(strict=False)
        except (OSError, RuntimeError) as e:
            raise PathSecurityError(f"Cannot resolve path {path}: {e}")

        # Check for path traversal attempts
        if ".." in path.parts:
            raise PathSecurityError(f"Path traversal detected in {path}")

        # Check if under base directory
        if not self._is_under_base(resolved):
            raise PathSecurityError(
                f"Path {resolved} is not under allowed base directory {self.base_dir}"
            )

        return resolved

    def _is_under_base(self, path: Path) -> bool:
        """Check if path is under base directory."""
        try:
            path.relative_to(self.base_dir)
            return True
        except ValueError:
            return False

    def get_collection_dir(self, collection_name: str) -> Path:
        """Get the collection directory path, creating if needed."""
        collection_dir = self.base_dir / "collections" / collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        return collection_dir


# Instruction-like text patterns to flag
INSTRUCTION_PATTERNS = [
    r"ignore\s+(?:previous|all|above)\s+instructions?",
    r"disregard\s+(?:previous|all|above)\s+instructions?",
    r"forget\s+(?:previous|all|above)\s+instructions?",
    r"system\s*:\s*you\s+are",
    r"assistant\s*:\s*you\s+are",
    r"<\|system\|>",
    r"<\|user\|>",
    r"<\|assistant\|>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
    r"###\s*Instruction",
    r"###\s*Response",
]


def flag_instruction_like_text(text: str) -> list[dict]:
    """
    Scan text for instruction-like patterns that might indicate prompt injection.
    Returns list of findings with pattern, position, and context.
    """
    findings = []
    for pattern in INSTRUCTION_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            findings.append({
                "pattern": pattern,
                "match": match.group(),
                "position": match.start(),
                "context": text[start:end],
            })
    return findings


def wrap_untrusted_text(text: str, source: str = "document") -> dict:
    """
    Wrap document text with metadata marking it as untrusted.
    This prevents agents from following instructions found in documents.
    """
    flags = flag_instruction_like_text(text)
    return {
        "text": text,
        "source": source,
        "trusted": False,
        "instruction_flags": flags,
        "char_count": len(text),
    }


def sanitize_for_logging(text: str, max_len: int = 200) -> str:
    """Sanitize text for safe logging (truncate, remove potential secrets)."""
    if len(text) > max_len:
        text = text[:max_len] + "... [truncated]"

    # Redact potential API keys, tokens, passwords
    patterns = [
        (r"(api[_-]?key|token|password|secret)['\"]?\s*[:=]\s*['\"]?[a-zA-Z0-9_-]{20,}", r"\1=***REDACTED***"),
        (r"sk-[a-zA-Z0-9]{32,}", "sk-***REDACTED***"),
        (r"Bearer\s+[a-zA-Z0-9._-]{20,}", "Bearer ***REDACTED***"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


# Global path allowlist instance (initialized by server)
_path_allowlist: PathAllowlist | None = None


def get_path_allowlist() -> PathAllowlist:
    """Get the global path allowlist instance."""
    global _path_allowlist
    if _path_allowlist is None:
        from ragdoc_mcp.config import settings
        _path_allowlist = PathAllowlist(settings.base_dir)
    return _path_allowlist


def init_path_allowlist(base_dir: Path) -> PathAllowlist:
    """Initialize the global path allowlist."""
    global _path_allowlist
    _path_allowlist = PathAllowlist(base_dir)
    return _path_allowlist
