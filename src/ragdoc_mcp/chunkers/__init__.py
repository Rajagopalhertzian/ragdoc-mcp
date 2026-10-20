"""Chunking strategies for RAG Doctor MCP."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ragdoc_mcp.parsers import ParsedDocument
from ragdoc_mcp.schemas import ContentType


@dataclass
class Chunk:
    """A document chunk with metadata."""
    chunk_id: str
    text: str
    file_path: str
    page: int | None
    line_start: int | None
    line_end: int | None
    heading_path: str | None
    content_type: ContentType
    token_count: int
    metadata: dict[str, Any]


class BaseChunker(ABC):
    """Abstract base class for chunkers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Chunker identifier."""
        pass

    @property
    @abstractmethod
    def supported_types(self) -> list[ContentType]:
        """Content types this chunker handles well."""
        pass

    @abstractmethod
    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        """Chunk a parsed document."""
        pass

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(text) // 4

    def _make_chunk_id(self, file_path: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        import hashlib
        h = hashlib.md5(f"{file_path}:{index}".encode()).hexdigest()[:12]
        return f"chunk_{h}"


class HeadingAwareChunker(BaseChunker):
    """Heading-aware chunking for markdown and structured documents."""

    name = "heading_aware"
    supported_types = [ContentType.MARKDOWN, ContentType.TEXT, ContentType.PDF, ContentType.PROSE]

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        chunks = []
        overlap_tokens = int(chunk_size * overlap)

        for page in doc.pages:
            page_num = page.get("page_num", 1)
            text = page.get("text", "")
            if not text.strip():
                continue

            # Split by headings
            sections = self._split_by_headings(text)
            current_chunk = ""
            current_heading = None
            current_lines = []
            line_offset = 0

            for section in sections:
                heading = section.get("heading")
                content = section.get("content", "")

                if heading:
                    current_heading = heading

                # Add content to current chunk
                lines = content.split('\n')
                for line in lines:
                    current_lines.append(line)
                    current_chunk = '\n'.join(current_lines)

                    if self._estimate_tokens(current_chunk) >= chunk_size:
                        # Create chunk
                        chunk = Chunk(
                            chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                            text=current_chunk,
                            file_path=doc.file_path,
                            page=page_num,
                            line_start=line_offset - len(current_lines) + 1,
                            line_end=line_offset,
                            heading_path=current_heading,
                            content_type=doc.content_type,
                            token_count=self._estimate_tokens(current_chunk),
                            metadata={"parser": doc.parser_used}
                        )
                        chunks.append(chunk)

                        # Handle overlap
                        if overlap_tokens > 0:
                            overlap_text = self._get_overlap_text(current_chunk, overlap_tokens)
                            current_lines = [overlap_text] if overlap_text else []
                            current_chunk = overlap_text
                        else:
                            current_lines = []
                            current_chunk = ""
                    line_offset += 1

            # Flush remaining
            if current_chunk.strip():
                chunk = Chunk(
                    chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                    text=current_chunk,
                    file_path=doc.file_path,
                    page=page_num,
                    line_start=line_offset - len(current_lines) + 1,
                    line_end=line_offset,
                    heading_path=current_heading,
                    content_type=doc.content_type,
                    token_count=self._estimate_tokens(current_chunk),
                    metadata={"parser": doc.parser_used}
                )
                chunks.append(chunk)

        return chunks

    def _split_by_headings(self, text: str) -> list[dict]:
        """Split text by markdown headings."""
        # Pattern matches # ## ### etc headings
        pattern = r'^(#{1,6})\s+(.+)$'
        sections = []
        current_heading = None
        current_content = []

        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                # Save previous section
                if current_content:
                    sections.append({
                        "heading": current_heading,
                        "content": '\n'.join(current_content)
                    })
                current_heading = match.group(2)
                current_content = [line]
            else:
                current_content.append(line)

        # Last section
        if current_content:
            sections.append({
                "heading": current_heading,
                "content": '\n'.join(current_content)
            })

        return sections

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Get last N tokens worth of text for overlap."""
        words = text.split()
        overlap_words = overlap_tokens * 4 // 5  # rough conversion
        if len(words) <= overlap_words:
            return text
        return ' '.join(words[-overlap_words:])


class TokenWindowChunker(BaseChunker):
    """Fixed-size token window with overlap."""

    name = "token_window"
    supported_types = [ContentType.CODE, ContentType.TEXT, ContentType.PROSE, ContentType.EQUATION]

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        chunks = []
        overlap_tokens = int(chunk_size * overlap)

        for page in doc.pages:
            page_num = page.get("page_num", 1)
            text = page.get("text", "")
            if not text.strip():
                continue

            words = text.split()
            if not words:
                continue

            step = max(1, chunk_size - overlap_tokens)
            token_estimate = lambda w: len(' '.join(w)) // 4

            i = 0
            while i < len(words):
                # Take chunk_size tokens worth of words
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)

                # Estimate line range (rough)
                line_start = text[:text.find(chunk_text[:50])].count('\n') + 1 if chunk_text[:50] in text else 1
                line_end = line_start + chunk_text.count('\n') + 1

                chunk = Chunk(
                    chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                    text=chunk_text,
                    file_path=doc.file_path,
                    page=page_num,
                    line_start=line_start,
                    line_end=line_end,
                    heading_path=None,
                    content_type=doc.content_type,
                    token_count=token_estimate(chunk_words),
                    metadata={"parser": doc.parser_used}
                )
                chunks.append(chunk)

                i += step

        return chunks


class TablePerChunkChunker(BaseChunker):
    """Keep tables intact as single chunks with summaries."""

    name = "table_per_chunk"
    supported_types = [ContentType.TABLE, ContentType.PDF]

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        chunks = []

        for page in doc.pages:
            page_num = page.get("page_num", 1)
            tables = page.get("tables", [])
            text = page.get("text", "")

            # Process each table as its own chunk
            for i, table in enumerate(tables):
                table_md = table.get("markdown", "")
                if not table_md.strip():
                    continue

                # Create summary
                summary = self._summarize_table(table)

                chunk_text = f"TABLE SUMMARY:\n{summary}\n\nTABLE DATA:\n{table_md}"

                chunk = Chunk(
                    chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                    text=chunk_text,
                    file_path=doc.file_path,
                    page=page_num,
                    line_start=None,
                    line_end=None,
                    heading_path=f"Table {i+1}",
                    content_type=ContentType.TABLE,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={
                        "parser": doc.parser_used,
                        "table_index": i,
                        "bbox": table.get("bbox"),
                        "is_table": True
                    }
                )
                chunks.append(chunk)

            # Also chunk remaining text with heading-aware chunker
            if text.strip() and not tables:
                heading_chunker = HeadingAwareChunker()
                text_chunks = heading_chunker.chunk(doc, chunk_size, overlap)
                chunks.extend(text_chunks)

        return chunks

    def _summarize_table(self, table: dict) -> str:
        """Generate a brief summary of the table."""
        data = table.get("data", [])
        if not data:
            return "Empty table"

        rows = len(data)
        cols = len(data[0]) if data else 0
        headers = data[0] if data else []

        summary = f"Table with {rows} rows and {cols} columns."
        if headers:
            summary += f" Headers: {', '.join(str(h) for h in headers[:5])}"
            if len(headers) > 5:
                summary += f" and {len(headers) - 5} more"
        return summary


class FormulaPreservingChunker(BaseChunker):
    """Never split inside formulas/equations."""

    name = "formula_preserving"
    supported_types = [ContentType.EQUATION, ContentType.PDF]

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        chunks = []
        overlap_tokens = int(chunk_size * overlap)

        for page in doc.pages:
            page_num = page.get("page_num", 1)
            text = page.get("text", "")
            equations = page.get("equations", [])

            if not text.strip() and not equations:
                continue

            # Split text but preserve equation boundaries
            # Find equation positions in text
            eq_positions = []
            for eq in equations:
                latex = eq.get("latex", "")
                if latex and latex in text:
                    pos = text.find(latex)
                    eq_positions.append((pos, pos + len(latex), latex))

            # Sort by position
            eq_positions.sort(key=lambda x: x[0])

            # Chunk with equation awareness
            chunks.extend(self._chunk_with_equations(
                text, eq_positions, doc, page_num, chunk_size, overlap_tokens
            ))

        return chunks

    def _chunk_with_equations(
        self, text: str, eq_positions: list, doc: ParsedDocument,
        page_num: int, chunk_size: int, overlap_tokens: int
    ) -> list[Chunk]:
        chunks = []
        words = text.split()
        if not words:
            return chunks

        step = max(1, chunk_size - overlap_tokens)
        i = 0
        char_pos = 0

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            # Check if this chunk splits an equation
            chunk_end_pos = char_pos + len(chunk_text)

            for eq_start, eq_end, eq_latex in eq_positions:
                if eq_start < chunk_end_pos < eq_end:
                    # Extend chunk to include full equation
                    remaining = text[chunk_end_pos:eq_end]
                    chunk_text += remaining
                    chunk_words = chunk_text.split()
                    break

            line_start = text[:char_pos].count('\n') + 1
            line_end = line_start + chunk_text.count('\n') + 1

            chunk = Chunk(
                chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                text=chunk_text,
                file_path=doc.file_path,
                page=page_num,
                line_start=line_start,
                line_end=line_end,
                heading_path=None,
                content_type=doc.content_type,
                token_count=self._estimate_tokens(chunk_text),
                metadata={"parser": doc.parser_used, "has_equations": len(eq_positions) > 0}
            )
            chunks.append(chunk)

            char_pos += len(' '.join(words[i:i + step]))
            i += step

        return chunks


class ASTCodeChunker(BaseChunker):
    """AST-based chunking for source code by function/class."""

    name = "ast_code"
    supported_types = [ContentType.CODE]

    def __init__(self):
        self._parsers = {}

    def _get_parser(self, language: str):
        """Get tree-sitter parser for language."""
        if language in self._parsers:
            return self._parsers[language]

        try:
            import tree_sitter
            lang_map = {
                "python": "tree_sitter_python",
                "javascript": "tree_sitter_javascript",
                "typescript": "tree_sitter_typescript",
            }
            if language not in lang_map:
                return None

            lang_module = __import__(lang_map[language], fromlist=['language'])
            lang = lang_module.language()
            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            return parser
        except ImportError:
            return None

    def chunk(self, doc: ParsedDocument, chunk_size: int, overlap: float) -> list[Chunk]:
        chunks = []

        for page in doc.pages:
            page_num = page.get("page_num", 1)
            text = page.get("text", "")
            language = page.get("metadata", {}).get("language", "python")

            if not text.strip():
                continue

            parser = self._get_parser(language)
            if parser is None:
                # Fallback to token window
                return TokenWindowChunker().chunk(doc, chunk_size, overlap)

            tree = parser.parse(text.encode('utf-8'))
            root = tree.root_node

            # Extract functions and classes
            nodes = self._extract_code_nodes(root, text)

            for node in nodes:
                node_text = text[node['start_byte']:node['end_byte']]
                token_count = self._estimate_tokens(node_text)

                if token_count > chunk_size * 2:
                    # Too large, split further
                    sub_chunks = self._split_large_node(node_text, doc, page_num, chunk_size, overlap)
                    chunks.extend(sub_chunks)
                else:
                    chunk = Chunk(
                        chunk_id=self._make_chunk_id(doc.file_path, len(chunks)),
                        text=node_text,
                        file_path=doc.file_path,
                        page=page_num,
                        line_start=node['start_line'],
                        line_end=node['end_line'],
                        heading_path=node['name'],
                        content_type=ContentType.CODE,
                        token_count=token_count,
                        metadata={
                            "parser": doc.parser_used,
                            "node_type": node['type'],
                            "language": language,
                            "imports": node.get('imports', [])
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _extract_code_nodes(self, root, text: str) -> list[dict]:
        """Extract function/class definitions from AST."""
        nodes = []

        def traverse(node):
            if node.type in ('function_definition', 'class_definition',
                            'function_declaration', 'class_declaration',
                            'method_definition', 'arrow_function'):
                # Get name
                name = None
                for child in node.children:
                    if child.type == 'identifier':
                        name = text[child.start_byte:child.end_byte]
                        break

                # Get imports (simplified)
                imports = []
                # Could traverse siblings for import statements

                nodes.append({
                    'type': node.type,
                    'name': name or 'anonymous',
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'start_line': node.start_point[0] + 1,
                    'end_line': node.end_point[0] + 1,
                    'imports': imports
                })

            for child in node.children:
                traverse(child)

        traverse(root)
        return nodes

    def _split_large_node(self, node_text: str, doc: ParsedDocument,
                          page_num: int, chunk_size: int, overlap: float) -> list[Chunk]:
        """Split a large code node using token window."""
        return TokenWindowChunker().chunk(
            ParsedDocument(
                file_path=doc.file_path,
                file_hash=doc.file_hash,
                content_type=ContentType.CODE,
                text=node_text,
                pages=[{"page_num": page_num, "text": node_text, "tables": [], "equations": [], "metadata": {}}],
                metadata=doc.metadata,
                parser_used=doc.parser_used,
                parse_time_ms=0
            ),
            chunk_size, overlap
        )


class ChunkerRegistry:
    """Registry for managing chunkers."""

    def __init__(self):
        self._chunkers: dict[str, BaseChunker] = {}
        self._type_routes: dict[ContentType, str] = {}

    def register(self, chunker: BaseChunker) -> None:
        self._chunkers[chunker.name] = chunker
        for ctype in chunker.supported_types:
            if ctype not in self._type_routes:
                self._type_routes[ctype] = chunker.name

    def get_chunker(self, name: str) -> BaseChunker:
        if name not in self._chunkers:
            raise ValueError(f"Chunker '{name}' not registered. Available: {list(self._chunkers.keys())}")
        return self._chunkers[name]

    def get_chunker_for_type(self, content_type: ContentType) -> BaseChunker:
        chunker_name = self._type_routes.get(content_type, "heading_aware")
        return self.get_chunker(chunker_name)

    def chunk(self, doc: ParsedDocument, chunker_name: str | None,
              chunk_size: int, overlap: float) -> list[Chunk]:
        if chunker_name:
            chunker = self.get_chunker(chunker_name)
        else:
            chunker = self.get_chunker_for_type(doc.content_type)
        return chunker.chunk(doc, chunk_size, overlap)


# Global chunker registry
_chunker_registry: ChunkerRegistry | None = None


def get_chunker_registry() -> ChunkerRegistry:
    global _chunker_registry
    if _chunker_registry is None:
        _chunker_registry = ChunkerRegistry()
        _chunker_registry.register(HeadingAwareChunker())
        _chunker_registry.register(TokenWindowChunker())
        _chunker_registry.register(TablePerChunkChunker())
        _chunker_registry.register(FormulaPreservingChunker())
        _chunker_registry.register(ASTCodeChunker())
    return _chunker_registry
