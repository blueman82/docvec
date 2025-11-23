"""Tests for MCP indexing tools.

This module tests the IndexingTools class including:
- Tool initialization
- index_file handler with various scenarios
- index_directory handler with recursive/non-recursive modes
- Path validation and error handling
- Result formatting
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from docvec.indexing.batch_processor import BatchProcessor, BatchResult
from docvec.indexing.indexer import Indexer, IndexingError
from docvec.mcp_tools.indexing_tools import (
    IndexingTools,
    INDEX_FILE_SCHEMA,
    INDEX_DIRECTORY_SCHEMA,
)


@pytest.fixture
def mock_indexer():
    """Provide mock Indexer."""
    indexer = Mock(spec=Indexer)
    indexer.index_document.return_value = ["chunk_1", "chunk_2", "chunk_3"]
    return indexer


@pytest.fixture
def mock_batch_processor():
    """Provide mock BatchProcessor."""
    processor = Mock(spec=BatchProcessor)

    # Default successful result
    result = BatchResult(
        new_documents=5,
        duplicates_skipped=2,
        errors=[],
        chunk_ids={
            "/path/file1.txt": ["c1", "c2"],
            "/path/file2.md": ["c3", "c4", "c5"],
        }
    )
    processor.process_directory.return_value = result

    return processor


@pytest.fixture
def indexing_tools(mock_batch_processor, mock_indexer):
    """Provide IndexingTools instance with mocks."""
    return IndexingTools(
        batch_processor=mock_batch_processor,
        indexer=mock_indexer
    )


@pytest.fixture
def temp_test_file(tmp_path):
    """Create temporary test file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content for indexing")
    return file_path


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directory with test files."""
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    (test_dir / "file1.txt").write_text("Content 1")
    (test_dir / "file2.md").write_text("# Markdown")
    (test_dir / "file3.py").write_text("def hello(): pass")

    return test_dir


class TestToolSchemas:
    """Test tool schema definitions."""

    def test_index_file_schema_structure(self):
        """Test index_file schema has correct structure."""
        assert INDEX_FILE_SCHEMA["type"] == "object"
        assert "file_path" in INDEX_FILE_SCHEMA["properties"]
        assert "file_path" in INDEX_FILE_SCHEMA["required"]
        assert INDEX_FILE_SCHEMA["properties"]["file_path"]["type"] == "string"

    def test_index_directory_schema_structure(self):
        """Test index_directory schema has correct structure."""
        assert INDEX_DIRECTORY_SCHEMA["type"] == "object"
        assert "dir_path" in INDEX_DIRECTORY_SCHEMA["properties"]
        assert "recursive" in INDEX_DIRECTORY_SCHEMA["properties"]
        assert "dir_path" in INDEX_DIRECTORY_SCHEMA["required"]
        assert INDEX_DIRECTORY_SCHEMA["properties"]["dir_path"]["type"] == "string"
        assert INDEX_DIRECTORY_SCHEMA["properties"]["recursive"]["type"] == "boolean"
        assert INDEX_DIRECTORY_SCHEMA["properties"]["recursive"]["default"] is True


class TestIndexingToolsInitialization:
    """Test IndexingTools initialization."""

    def test_init_with_dependencies(self, mock_batch_processor, mock_indexer):
        """Test initialization with batch processor and indexer."""
        tools = IndexingTools(
            batch_processor=mock_batch_processor,
            indexer=mock_indexer
        )

        assert tools.batch_processor == mock_batch_processor
        assert tools.indexer == mock_indexer

    def test_init_stores_dependencies(self, indexing_tools):
        """Test that dependencies are properly stored."""
        assert indexing_tools.batch_processor is not None
        assert indexing_tools.indexer is not None


class TestPathValidation:
    """Test path validation functionality."""

    def test_validate_path_with_valid_file(self, indexing_tools, temp_test_file):
        """Test validating existing file path."""
        path = indexing_tools._validate_path(str(temp_test_file))

        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_absolute()

    def test_validate_path_with_relative_path(self, indexing_tools, temp_test_file, tmp_path):
        """Test validating relative path."""
        # Change to temp directory and use relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            relative_path = temp_test_file.relative_to(tmp_path)

            path = indexing_tools._validate_path(str(relative_path))

            assert path.exists()
            assert path.is_absolute()
        finally:
            os.chdir(original_cwd)

    def test_validate_path_with_absolute_path(self, indexing_tools, tmp_path):
        """Test path with absolute path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        # Use absolute path
        path = indexing_tools._validate_path(str(test_file))

        assert path.exists()
        assert path.is_absolute()

    def test_validate_path_nonexistent_raises_error(self, indexing_tools):
        """Test that nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            indexing_tools._validate_path("/nonexistent/path/file.txt")

    def test_validate_path_empty_string_raises_error(self, indexing_tools):
        """Test that empty path string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            indexing_tools._validate_path("")

    def test_validate_path_whitespace_only_raises_error(self, indexing_tools):
        """Test that whitespace-only path raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            indexing_tools._validate_path("   ")


class TestIndexFileTool:
    """Test index_file tool handler."""

    @pytest.mark.asyncio
    async def test_index_file_success(self, indexing_tools, temp_test_file, mock_indexer):
        """Test successful file indexing."""
        result = await indexing_tools.index_file(str(temp_test_file))

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["file"] == str(temp_test_file)
        assert result["data"]["chunks"] == 3
        assert result["data"]["chunk_ids"] == ["chunk_1", "chunk_2", "chunk_3"]

        mock_indexer.index_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_file_relative_path(self, indexing_tools, temp_test_file, tmp_path, mock_indexer):
        """Test indexing file with relative path."""
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            relative_path = temp_test_file.relative_to(tmp_path)

            result = await indexing_tools.index_file(str(relative_path))

            assert result["success"] is True
            assert result["data"]["chunks"] == 3
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_index_file_nonexistent_file(self, indexing_tools):
        """Test indexing nonexistent file returns error."""
        result = await indexing_tools.index_file("/nonexistent/file.txt")

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_file_directory_instead_of_file(self, indexing_tools, tmp_path):
        """Test indexing directory path returns error."""
        result = await indexing_tools.index_file(str(tmp_path))

        assert result["success"] is False
        assert "error" in result
        assert "not a file" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_file_indexing_error(self, indexing_tools, temp_test_file, mock_indexer):
        """Test handling IndexingError from indexer."""
        mock_indexer.index_document.side_effect = IndexingError("Failed to index")

        result = await indexing_tools.index_file(str(temp_test_file))

        assert result["success"] is False
        assert "error" in result
        assert "Indexing failed" in result["error"]

    @pytest.mark.asyncio
    async def test_index_file_empty_path(self, indexing_tools):
        """Test indexing with empty path string."""
        result = await indexing_tools.index_file("")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_index_file_no_chunks_generated(self, indexing_tools, temp_test_file, mock_indexer):
        """Test indexing file that generates no chunks."""
        mock_indexer.index_document.return_value = []

        result = await indexing_tools.index_file(str(temp_test_file))

        assert result["success"] is True
        assert result["data"]["chunks"] == 0
        assert result["data"]["chunk_ids"] == []

    @pytest.mark.asyncio
    async def test_index_file_unexpected_error(self, indexing_tools, temp_test_file, mock_indexer):
        """Test handling unexpected exception."""
        mock_indexer.index_document.side_effect = RuntimeError("Unexpected error")

        result = await indexing_tools.index_file(str(temp_test_file))

        assert result["success"] is False
        assert "error" in result
        assert "Unexpected error" in result["error"]


class TestIndexDirectoryTool:
    """Test index_directory tool handler."""

    @pytest.mark.asyncio
    async def test_index_directory_success(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test successful directory indexing."""
        result = await indexing_tools.index_directory(str(temp_test_dir), recursive=True)

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["directory"] == str(temp_test_dir)
        assert result["data"]["recursive"] is True
        assert result["data"]["new_documents"] == 5
        assert result["data"]["duplicates_skipped"] == 2
        assert result["data"]["total_chunks"] == 5  # 2 + 3 from chunk_ids

        mock_batch_processor.process_directory.assert_called_once_with(
            temp_test_dir, recursive=True
        )

    @pytest.mark.asyncio
    async def test_index_directory_non_recursive(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test non-recursive directory indexing."""
        result = await indexing_tools.index_directory(str(temp_test_dir), recursive=False)

        assert result["success"] is True
        assert result["data"]["recursive"] is False

        mock_batch_processor.process_directory.assert_called_once_with(
            temp_test_dir, recursive=False
        )

    @pytest.mark.asyncio
    async def test_index_directory_default_recursive(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test that recursive defaults to True."""
        result = await indexing_tools.index_directory(str(temp_test_dir))

        assert result["success"] is True
        assert result["data"]["recursive"] is True

        mock_batch_processor.process_directory.assert_called_once_with(
            temp_test_dir, recursive=True
        )

    @pytest.mark.asyncio
    async def test_index_directory_with_errors(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test directory indexing with some file errors."""
        result_with_errors = BatchResult(
            new_documents=3,
            duplicates_skipped=1,
            errors=[
                ("/path/bad1.txt", "Failed to read"),
                ("/path/bad2.md", "Invalid format")
            ],
            chunk_ids={
                "/path/good1.txt": ["c1", "c2"],
                "/path/good2.py": ["c3"],
            }
        )
        mock_batch_processor.process_directory.return_value = result_with_errors

        result = await indexing_tools.index_directory(str(temp_test_dir))

        assert result["success"] is True
        assert result["data"]["new_documents"] == 3
        assert len(result["data"]["errors"]) == 2
        assert result["data"]["errors"][0]["file"] == "/path/bad1.txt"
        assert result["data"]["errors"][0]["error"] == "Failed to read"

    @pytest.mark.asyncio
    async def test_index_directory_nonexistent(self, indexing_tools):
        """Test indexing nonexistent directory returns error."""
        result = await indexing_tools.index_directory("/nonexistent/directory")

        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_directory_file_instead_of_directory(self, indexing_tools, temp_test_file):
        """Test indexing file path instead of directory."""
        result = await indexing_tools.index_directory(str(temp_test_file))

        assert result["success"] is False
        assert "error" in result
        assert "not a directory" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_directory_empty_directory(self, indexing_tools, tmp_path, mock_batch_processor):
        """Test indexing empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        empty_result = BatchResult(
            new_documents=0,
            duplicates_skipped=0,
            errors=[],
            chunk_ids={}
        )
        mock_batch_processor.process_directory.return_value = empty_result

        result = await indexing_tools.index_directory(str(empty_dir))

        assert result["success"] is True
        assert result["data"]["new_documents"] == 0
        assert result["data"]["total_chunks"] == 0
        assert len(result["data"]["indexed_files"]) == 0

    @pytest.mark.asyncio
    async def test_index_directory_all_duplicates(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test indexing directory where all files are duplicates."""
        all_duplicates = BatchResult(
            new_documents=0,
            duplicates_skipped=10,
            errors=[],
            chunk_ids={}
        )
        mock_batch_processor.process_directory.return_value = all_duplicates

        result = await indexing_tools.index_directory(str(temp_test_dir))

        assert result["success"] is True
        assert result["data"]["new_documents"] == 0
        assert result["data"]["duplicates_skipped"] == 10

    @pytest.mark.asyncio
    async def test_index_directory_unexpected_error(self, indexing_tools, temp_test_dir, mock_batch_processor):
        """Test handling unexpected exception."""
        mock_batch_processor.process_directory.side_effect = RuntimeError("Unexpected")

        result = await indexing_tools.index_directory(str(temp_test_dir))

        assert result["success"] is False
        assert "error" in result
        assert "Unexpected error" in result["error"]


class TestResultFormatting:
    """Test result formatting helpers."""

    def test_format_result_structure(self, indexing_tools):
        """Test format_result creates correct structure."""
        result = indexing_tools._format_result(
            operation="test_op",
            data={"key": "value", "count": 42}
        )

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["key"] == "value"
        assert result["data"]["count"] == 42

    def test_format_error_structure(self, indexing_tools):
        """Test format_error creates correct structure."""
        result = indexing_tools._format_error(
            operation="test_op",
            error="Something went wrong"
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Something went wrong"

    def test_format_result_with_empty_data(self, indexing_tools):
        """Test formatting result with empty data."""
        result = indexing_tools._format_result(
            operation="test_op",
            data={}
        )

        assert result["success"] is True
        assert result["data"] == {}

    def test_format_result_with_nested_data(self, indexing_tools):
        """Test formatting result with nested data structures."""
        result = indexing_tools._format_result(
            operation="test_op",
            data={
                "summary": {"total": 10, "success": 8},
                "details": [{"id": 1}, {"id": 2}]
            }
        )

        assert result["success"] is True
        assert result["data"]["summary"]["total"] == 10
        assert len(result["data"]["details"]) == 2


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_index_file_and_directory_sequence(self, indexing_tools, tmp_path, mock_indexer, mock_batch_processor):
        """Test indexing file followed by directory."""
        # Create test files
        file1 = tmp_path / "single.txt"
        file1.write_text("Single file")

        dir1 = tmp_path / "docs"
        dir1.mkdir()
        (dir1 / "doc1.txt").write_text("Doc 1")

        # Index single file
        result1 = await indexing_tools.index_file(str(file1))
        assert result1["success"] is True

        # Index directory
        result2 = await indexing_tools.index_directory(str(dir1))
        assert result2["success"] is True

        # Verify both operations called their respective handlers
        assert mock_indexer.index_document.called
        assert mock_batch_processor.process_directory.called

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, indexing_tools, tmp_path, mock_indexer):
        """Test handling mixed success and failure operations."""
        good_file = tmp_path / "good.txt"
        good_file.write_text("Content")

        # Successful index
        result1 = await indexing_tools.index_file(str(good_file))
        assert result1["success"] is True

        # Failed index (nonexistent)
        result2 = await indexing_tools.index_file("/nonexistent.txt")
        assert result2["success"] is False

        # Verify state remains consistent
        result3 = await indexing_tools.index_file(str(good_file))
        assert result3["success"] is True
