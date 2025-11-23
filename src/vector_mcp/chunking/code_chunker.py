"""AST-based code chunker for Python files."""

import ast
import logging
from typing import Optional

from vector_mcp.chunking.base import AbstractChunker, Chunk

logger = logging.getLogger(__name__)


class CodeChunker(AbstractChunker):
    """AST-based chunker that preserves function and class boundaries.

    Uses Python's ast module to parse code structure and chunk by semantic
    units (functions, classes, methods). Falls back to line-based chunking
    for unparseable code.
    """

    def __init__(self, chunk_size: int = 100):
        """Initialize code chunker.

        Args:
            chunk_size: Maximum number of lines per chunk for fallback chunking
        """
        self.chunk_size = chunk_size

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split Python code into semantic chunks.

        Args:
            content: Python source code to chunk
            source_file: Path to source file for provenance tracking

        Returns:
            List of Chunk objects with sequential indexing

        Raises:
            ValueError: If content is empty or invalid
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Try to parse with AST
        ast_module = self._parse_ast(content)

        if ast_module is not None:
            # Successfully parsed, use AST-based chunking
            return self._chunk_by_definitions(content, source_file, ast_module)
        else:
            # Parsing failed, fall back to line-based chunking
            logger.warning(
                f"Failed to parse {source_file} as Python, using line-based chunking"
            )
            return self._fallback_line_chunking(content, source_file)

    def _parse_ast(self, content: str) -> Optional[ast.Module]:
        """Parse Python code into AST.

        Args:
            content: Python source code

        Returns:
            AST Module if parsing succeeds, None otherwise
        """
        try:
            return ast.parse(content)
        except SyntaxError as e:
            logger.debug(f"Syntax error during AST parsing: {e}")
            return None
        except Exception as e:
            logger.debug(f"Unexpected error during AST parsing: {e}")
            return None

    def _extract_definitions(
        self, ast_module: ast.Module
    ) -> list[tuple[str, int, int, Optional[str]]]:
        """Extract top-level function and class definitions.

        Args:
            ast_module: Parsed AST module

        Returns:
            List of tuples (definition_type, start_line, end_line, class_name)
            where definition_type is 'function', 'class', or 'async_function'
            and class_name is set for class definitions
        """
        definitions = []

        for node in ast_module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line, end_line = self._get_node_lines(node)
                def_type = (
                    "async_function"
                    if isinstance(node, ast.AsyncFunctionDef)
                    else "function"
                )
                definitions.append((def_type, start_line, end_line, None))

            elif isinstance(node, ast.ClassDef):
                start_line, end_line = self._get_node_lines(node)
                definitions.append(("class", start_line, end_line, node.name))

        return definitions

    def _get_node_lines(self, node: ast.AST) -> tuple[int, int]:
        """Get start and end line numbers for an AST node.

        For function and class definitions, includes decorators if present.

        Args:
            node: AST node

        Returns:
            Tuple of (start_line, end_line) (1-indexed)
        """
        start_line = node.lineno if hasattr(node, "lineno") else 1
        end_line = node.end_lineno if hasattr(node, "end_lineno") else start_line

        # Include decorators for functions and classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if hasattr(node, "decorator_list") and node.decorator_list:
                # Get the first decorator's line number
                first_decorator = node.decorator_list[0]
                if hasattr(first_decorator, "lineno"):
                    start_line = first_decorator.lineno

        return start_line, end_line

    def _chunk_by_definitions(
        self, content: str, source_file: str, ast_module: ast.Module
    ) -> list[Chunk]:
        """Chunk code by AST definitions.

        Args:
            content: Python source code
            source_file: Path to source file
            ast_module: Parsed AST module

        Returns:
            List of chunks based on code structure
        """
        lines = content.splitlines(keepends=False)
        definitions = self._extract_definitions(ast_module)
        chunks = []
        chunk_index = 0

        # Extract module-level imports and docstring
        imports = []
        module_docstring = None

        for node in ast_module.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start, end = self._get_node_lines(node)
                import_lines = lines[start - 1 : end]
                imports.extend(import_lines)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                # Module docstring
                if node.lineno == 1 or (
                    node.lineno == 2 and lines[0].startswith("#")
                ):
                    module_docstring = node.value.value

        # Create first chunk with imports and module docstring if present
        if imports or module_docstring:
            import_content_parts = []
            if module_docstring:
                import_content_parts.append(f'"""{module_docstring}"""')
            if imports:
                import_content_parts.extend(imports)

            import_content = "\n".join(import_content_parts)
            if import_content.strip():
                chunks.append(
                    Chunk(
                        content=import_content,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata={
                            "type": "imports",
                            "has_docstring": module_docstring is not None,
                        },
                    )
                )
                chunk_index += 1

        # Chunk each definition
        for def_type, start_line, end_line, class_name in definitions:
            # Extract definition lines (AST uses 1-based indexing)
            definition_lines = lines[start_line - 1 : end_line]
            definition_content = "\n".join(definition_lines)

            if definition_content.strip():
                metadata = {"type": def_type, "start_line": start_line}
                if class_name:
                    metadata["class_name"] = class_name

                chunks.append(
                    Chunk(
                        content=definition_content,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        # If no chunks were created (empty file or only comments), create one chunk
        if not chunks:
            chunks.append(
                Chunk(
                    content=content,
                    source_file=source_file,
                    chunk_index=0,
                    metadata={"type": "other"},
                )
            )

        return chunks

    def _fallback_line_chunking(self, content: str, source_file: str) -> list[Chunk]:
        """Fall back to simple line-based chunking.

        Args:
            content: Source code content
            source_file: Path to source file

        Returns:
            List of chunks split by line count
        """
        lines = content.splitlines(keepends=False)
        chunks = []
        chunk_index = 0

        for i in range(0, len(lines), self.chunk_size):
            chunk_lines = lines[i : i + self.chunk_size]
            chunk_content = "\n".join(chunk_lines)

            if chunk_content.strip():
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        source_file=source_file,
                        chunk_index=chunk_index,
                        metadata={
                            "type": "fallback",
                            "start_line": i + 1,
                            "end_line": min(i + self.chunk_size, len(lines)),
                        },
                    )
                )
                chunk_index += 1

        # If no chunks were created, create one with all content
        if not chunks:
            chunks.append(
                Chunk(
                    content=content,
                    source_file=source_file,
                    chunk_index=0,
                    metadata={"type": "fallback"},
                )
            )

        return chunks
