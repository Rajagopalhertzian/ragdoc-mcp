from __future__ import annotations


class RagDocError(Exception):
    """A user-actionable error: what failed and what to try."""

    def __init__(self, message: str, hint: str | None = None, code: str = "ERROR"):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.code = code
