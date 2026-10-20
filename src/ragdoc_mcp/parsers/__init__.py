"""Base parser interface and implementations for RAG Doctor MCP."""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragdoc_mcp.schemas import ContentType


@dataclass
class ParsedDocument:
    """Result of parsing a document."""
    file_path: str
    file_hash: str
    content_type: ContentType
    text: str
    pages: list[dict[str, Any]]  # page_num, text, tables, equations, metadata
    metadata: dict[str, Any]
    parser_used: str
    parse_time_ms: float


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Parser identifier."""
        pass

    @property
    @abstractmethod
    def supported_types(self) -> list[ContentType]:
        """Content types this parser can handle."""
        pass

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a document file."""
        pass

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _detect_content_type(self, file_path: Path, text: str) -> ContentType:
        """Detect content type from file extension and content."""
        suffix = file_path.suffix.lower()

        if suffix in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go"}:
            return ContentType.CODE
        if suffix in {".md", ".markdown"}:
            return ContentType.MARKDOWN
        if suffix in {".txt", ".rst"}:
            return ContentType.TEXT
        if suffix == ".pdf":
            # Check for equations, tables, code in text
            if re.search(r'\$.*?\$|\\begin\{.*?\}|\\\[.*?\\\]', text):
                return ContentType.EQUATION
            if re.search(r'\|.*\|.*\|', text) or 'table' in text.lower():
                return ContentType.TABLE
            return ContentType.PDF
        return ContentType.UNKNOWN


class FastParser(BaseParser):
    """Fast parser using pypdf for PDFs, native for text/markdown/code."""

    name = "fast"
    supported_types = [ContentType.TEXT, ContentType.MARKDOWN, ContentType.CODE, ContentType.PDF]

    def __init__(self):
        self._pypdf = None

    def _get_pypdf(self):
        """Lazy import pypdf."""
        if self._pypdf is None:
            try:
                import pypdf
                self._pypdf = pypdf
            except ImportError:
                raise RuntimeError("pypdf not installed. Install with: pip install pypdf")
        return self._pypdf

    def parse(self, file_path: Path) -> ParsedDocument:
        import time
        start = time.perf_counter()

        file_hash = self._compute_hash(file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text, pages = self._parse_pdf(file_path)
        elif suffix in {".md", ".markdown"}:
            text, pages = self._parse_markdown(file_path)
        elif suffix in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go"}:
            text, pages = self._parse_code(file_path)
        else:
            text, pages = self._parse_text(file_path)

        content_type = self._detect_content_type(file_path, text)
        parse_time_ms = (time.perf_counter() - start) * 1000

        return ParsedDocument(
            file_path=str(file_path),
            file_hash=file_hash,
            content_type=content_type,
            text=text,
            pages=pages,
            metadata={"parser": self.name, "file_size": file_path.stat().st_size},
            parser_used=self.name,
            parse_time_ms=parse_time_ms,
        )

    def _parse_pdf(self, file_path: Path) -> tuple[str, list[dict]]:
        pypdf = self._get_pypdf()
        pages = []
        full_text = []

        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                pages.append({
                    "page_num": i + 1,
                    "text": page_text,
                    "tables": [],
                    "equations": [],
                    "metadata": {"page_width": page.mediabox.width if page.mediabox else 0,
                                "page_height": page.mediabox.height if page.mediabox else 0}
                })
                full_text.append(page_text)

        return "\n\n".join(full_text), pages

    def _parse_markdown(self, file_path: Path) -> tuple[str, list[dict]]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        # Split by headings for pseudo-pages
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        pages = []
        for i, section in enumerate(sections):
            pages.append({
                "page_num": i + 1,
                "text": section,
                "tables": [],
                "equations": [],
                "metadata": {}
            })
        return text, pages

    def _parse_code(self, file_path: Path) -> tuple[str, list[dict]]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        # Treat whole file as one "page" with line numbers
        lines = text.split('\n')
        pages = [{
            "page_num": 1,
            "text": text,
            "tables": [],
            "equations": [],
            "metadata": {"lines": len(lines), "language": file_path.suffix[1:]}
        }]
        return text, pages

    def _parse_text(self, file_path: Path) -> tuple[str, list[dict]]:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        pages = [{
            "page_num": 1,
            "text": text,
            "tables": [],
            "equations": [],
            "metadata": {}
        }]
        return text, pages


class DoclingParser(BaseParser):
    """Docling parser for advanced PDF parsing (layout, tables, formulas, OCR)."""

    name = "docling"
    supported_types = [ContentType.PDF, ContentType.EQUATION, ContentType.TABLE]

    def __init__(self, enable_ocr: bool = False):
        self._docling = None
        self._pipeline_options = None
        self.enable_ocr = enable_ocr

    def _get_docling(self):
        """Lazy import docling."""
        if self._docling is None:
            try:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter
                self._docling = {
                    "DocumentConverter": DocumentConverter,
                    "PdfPipelineOptions": PdfPipelineOptions,
                    "InputFormat": InputFormat,
                }
            except ImportError:
                raise RuntimeError("docling not installed. Install with: pip install docling")
        return self._docling

    def parse(self, file_path: Path) -> ParsedDocument:
        import time
        start = time.perf_counter()

        file_hash = self._compute_hash(file_path)
        dl = self._get_docling()

        # Configure pipeline options
        pipeline_options = dl["PdfPipelineOptions"]()
        pipeline_options.do_ocr = self.enable_ocr
        pipeline_options.do_table_structure = True
        pipeline_options.do_formula_enrichment = True

        converter = dl["DocumentConverter"](pipeline_options=pipeline_options)
        result = converter.convert(str(file_path))

        # Extract structured content
        text = result.document.export_to_markdown()
        pages = self._extract_pages(result.document)

        content_type = self._detect_content_type(file_path, text)
        parse_time_ms = (time.perf_counter() - start) * 1000

        return ParsedDocument(
            file_path=str(file_path),
            file_hash=file_hash,
            content_type=content_type,
            text=text,
            pages=pages,
            metadata={
                "parser": self.name,
                "file_size": file_path.stat().st_size,
                "ocr_enabled": self.enable_ocr,
                "num_tables": len(result.document.tables) if hasattr(result.document, 'tables') else 0,
                "num_formulas": len(result.document.formulas) if hasattr(result.document, 'formulas') else 0,
            },
            parser_used=self.name,
            parse_time_ms=parse_time_ms,
        )

    def _extract_pages(self, document) -> list[dict]:
        """Extract page-level content from Docling document."""
        pages = []
        # Docling provides page-level access
        if hasattr(document, 'pages'):
            for page_num, page in document.pages.items():
                page_text = page.text if hasattr(page, 'text') else ""
                tables = []
                if hasattr(page, 'tables'):
                    for table in page.tables:
                        tables.append({
                            "bbox": table.bbox.as_tuple() if hasattr(table, 'bbox') else None,
                            "markdown": table.export_to_markdown() if hasattr(table, 'export_to_markdown') else "",
                            "data": table.data if hasattr(table, 'data') else [],
                        })
                equations = []
                if hasattr(page, 'formulas'):
                    for formula in page.formulas:
                        equations.append({
                            "bbox": formula.bbox.as_tuple() if hasattr(formula, 'bbox') else None,
                            "latex": formula.latex if hasattr(formula, 'latex') else "",
                        })
                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                    "tables": tables,
                    "equations": equations,
                    "metadata": {}
                })
        else:
            # Fallback: treat whole document as one page
            pages.append({
                "page_num": 1,
                "text": document.export_to_markdown(),
                "tables": [],
                "equations": [],
                "metadata": {}
            })
        return pages


class ParserRegistry:
    """Registry for managing parsers and routing by content type."""

    def __init__(self):
        self._parsers: dict[str, BaseParser] = {}
        self._type_routes: dict[ContentType, str] = {}

    def register(self, parser: BaseParser) -> None:
        """Register a parser."""
        self._parsers[parser.name] = parser
        for ctype in parser.supported_types:
            if ctype not in self._type_routes:
                self._type_routes[ctype] = parser.name

    def get_parser(self, name: str) -> BaseParser:
        """Get parser by name."""
        if name not in self._parsers:
            raise ValueError(f"Parser '{name}' not registered. Available: {list(self._parsers.keys())}")
        return self._parsers[name]

    def get_parser_for_type(self, content_type: ContentType) -> BaseParser:
        """Get the default parser for a content type."""
        parser_name = self._type_routes.get(content_type, "fast")
        return self.get_parser(parser_name)

    def parse(self, file_path: Path, parser_name: str | None = None) -> ParsedDocument:
        """Parse a file, optionally specifying parser."""
        if parser_name:
            parser = self.get_parser(parser_name)
        else:
            # Auto-detect content type using fast parser first
            fast_parser = self.get_parser("fast")
            # Quick read to detect type
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                # For PDFs, use Docling by default for better extraction
                parser = self.get_parser("docling")
            else:
                parser = fast_parser
        return parser.parse(file_path)


# Global parser registry
_parser_registry: ParserRegistry | None = None


def get_parser_registry() -> ParserRegistry:
    """Get the global parser registry, initializing if needed."""
    global _parser_registry
    if _parser_registry is None:
        _parser_registry = ParserRegistry()
        _parser_registry.register(FastParser())
        _parser_registry.register(DoclingParser())
    return _parser_registry
