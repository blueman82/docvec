"""MCP tools for document indexing operations.

This module provides MCP tool handlers for indexing operations:
- index_file: Index a single document file
- index_directory: Index all supported files in a directory

Handlers validate inputs, leverage batch processing with deduplication,
and return structured results with counts and error details.
"""

import logging
from pathlib import Path
from typing import Any

from docvec.indexing.batch_processor import BatchProcessor
from docvec.indexing.indexer import Indexer, IndexingError

logger = logging.getLogger(__name__)


# Tool schema definitions
INDEX_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Absolute or relative path to the file to index",
        }
    },
    "required": ["file_path"],
}

INDEX_DIRECTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "dir_path": {
            "type": "string",
            "description": "Absolute or relative path to the directory to index",
        },
        "recursive": {
            "type": "boolean",
            "description": "Whether to recursively index subdirectories (default: true)",
            "default": True,
        },
    },
    "required": ["dir_path"],
}


class IndexingTools:
    """MCP tool handlers for document indexing operations.

    Provides async handlers for indexing files and directories with proper
    validation, error handling, and structured result formatting.

    Args:
        batch_processor: BatchProcessor for directory indexing with deduplication
        indexer: Indexer for single file indexing

    Attributes:
        batch_processor: Batch processing instance
        indexer: Document indexing instance

    Example:
        >>> tools = IndexingTools(batch_processor, indexer)
        >>> result = await tools.index_file(file_path="/path/to/doc.md")
        >>> print(result)
        {'success': True, 'data': {'file': '/path/to/doc.md', 'chunks': 5}}
    """

    def __init__(self, batch_processor: BatchProcessor, indexer: Indexer):
        """Initialize indexing tools with dependencies.

        Args:
            batch_processor: BatchProcessor instance for batch operations
            indexer: Indexer instance for single file operations
        """
        self.batch_processor = batch_processor
        self.indexer = indexer
        logger.info("Initialized IndexingTools")

    async def index_file(self, file_path: str) -> dict[str, Any]:
        """Index a single document file.

        Validates file path, indexes the document using the Indexer,
        and returns structured result with chunk count.

        Args:
            file_path: Path to file to index (absolute or relative)

        Returns:
            Structured result dictionary:
            {
                'success': bool,
                'data': {
                    'file': str,
                    'chunks': int,
                    'chunk_ids': list[str]
                },
                'error': str (optional, only if success=False)
            }

        Example:
            >>> result = await tools.index_file(file_path="docs/readme.md")
            >>> result
            {'success': True, 'data': {'file': 'docs/readme.md', 'chunks': 3, ...}}
        """
        logger.info(f"MCP tool invoked: index_file({file_path})")

        try:
            # Validate and resolve path
            path = self._validate_path(file_path)

            # Verify it's a file
            if not path.is_file():
                return self._format_error(
                    operation="index_file", error=f"Path is not a file: {file_path}"
                )

            # Index the file
            chunk_ids = self.indexer.index_document(path)

            # Format success response
            return self._format_result(
                operation="index_file",
                data={
                    "file": str(path),
                    "chunks": len(chunk_ids),
                    "chunk_ids": chunk_ids,
                },
            )

        except FileNotFoundError:
            return self._format_error(
                operation="index_file", error=f"File not found: {file_path}"
            )
        except IndexingError as e:
            return self._format_error(
                operation="index_file", error=f"Indexing failed: {str(e)}"
            )
        except Exception as e:
            logger.exception(f"Unexpected error in index_file: {e}")
            return self._format_error(
                operation="index_file", error=f"Unexpected error: {str(e)}"
            )

    async def index_directory(
        self, dir_path: str, recursive: bool = True
    ) -> dict[str, Any]:
        """Index all supported files in a directory.

        Validates directory path, uses BatchProcessor for efficient
        indexing with deduplication, and returns detailed statistics.

        Args:
            dir_path: Path to directory to index (absolute or relative)
            recursive: Whether to scan subdirectories (default: True)

        Returns:
            Structured result dictionary:
            {
                'success': bool,
                'data': {
                    'directory': str,
                    'recursive': bool,
                    'new_documents': int,
                    'duplicates_skipped': int,
                    'total_chunks': int,
                    'errors': list[dict],
                    'indexed_files': list[str]
                },
                'error': str (optional, only if success=False)
            }

        Example:
            >>> result = await tools.index_directory(
            ...     dir_path="docs",
            ...     recursive=True
            ... )
            >>> result['data']['new_documents']
            15
        """
        logger.info(
            f"MCP tool invoked: index_directory({dir_path}, recursive={recursive})"
        )

        try:
            # Validate and resolve path
            path = self._validate_path(dir_path)

            # Verify it's a directory
            if not path.is_dir():
                return self._format_error(
                    operation="index_directory",
                    error=f"Path is not a directory: {dir_path}",
                )

            # Process directory using batch processor
            result = self.batch_processor.process_directory(path, recursive=recursive)

            # Calculate total chunks
            total_chunks = sum(len(ids) for ids in result.chunk_ids.values())

            # Format errors for response
            formatted_errors = [
                {"file": file, "error": error} for file, error in result.errors
            ]

            # Format success response
            return self._format_result(
                operation="index_directory",
                data={
                    "directory": str(path),
                    "recursive": recursive,
                    "new_documents": result.new_documents,
                    "duplicates_skipped": result.duplicates_skipped,
                    "total_chunks": total_chunks,
                    "errors": formatted_errors,
                    "indexed_files": list(result.chunk_ids.keys()),
                },
            )

        except FileNotFoundError:
            return self._format_error(
                operation="index_directory", error=f"Directory not found: {dir_path}"
            )
        except NotADirectoryError:
            return self._format_error(
                operation="index_directory",
                error=f"Path is not a directory: {dir_path}",
            )
        except Exception as e:
            logger.exception(f"Unexpected error in index_directory: {e}")
            return self._format_error(
                operation="index_directory", error=f"Unexpected error: {str(e)}"
            )

    def _validate_path(self, path_str: str) -> Path:
        """Validate and resolve a path string.

        Converts string to Path object, resolves to absolute path,
        and validates that the path exists.

        Args:
            path_str: Path string to validate

        Returns:
            Resolved Path object

        Raises:
            ValueError: If path string is empty
            FileNotFoundError: If path does not exist
        """
        if not path_str or not path_str.strip():
            raise ValueError("Path cannot be empty")

        # Convert to Path and resolve to absolute path
        path = Path(path_str).expanduser().resolve()

        # Check existence
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path_str}")

        return path

    def _format_result(self, operation: str, data: dict[str, Any]) -> dict[str, Any]:
        """Format successful operation result.

        Args:
            operation: Name of operation (e.g., 'index_file')
            data: Operation result data

        Returns:
            Structured success response
        """
        return {"success": True, "data": data}

    def _format_error(self, operation: str, error: str) -> dict[str, Any]:
        """Format error response.

        Args:
            operation: Name of operation that failed
            error: Error message

        Returns:
            Structured error response
        """
        return {"success": False, "error": error}
