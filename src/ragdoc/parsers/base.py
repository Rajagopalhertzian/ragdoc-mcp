from __future__ import annotations

from dataclasses import asdict, dataclass, field

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}
PDF_EXTS = {".pdf"}
CODE_EXTS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".java": "java", ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".sh": "shell", ".kt": "kotlin", ".swift": "swift",
}
SUPPORTED_EXTS = TEXT_EXTS | PDF_EXTS | set(CODE_EXTS)


class ParserUnavailable(RuntimeError):
    """The requested backend is not installed or not allowed."""


@dataclass
class ParsedDoc:
    path: str  # relative to the corpus root
    file_hash: str
    parser: str
    kind: str  # text | pdf | code
    pages: list[dict] = field(default_factory=list)  # [{"num": int, "text": str}]
    meta: dict = field(default_factory=dict)  # scanned_pages, n_pages, language, notes

    def full_text(self) -> str:
        return "\n".join(p["text"] for p in self.pages)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> ParsedDoc:
        return ParsedDoc(**d)
