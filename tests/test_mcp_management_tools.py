"""Tests for MCP management tools for delete operations.

This module tests ManagementTools functionality including:
- Deleting chunks by IDs
- Deleting chunks by source file
- Clearing entire collection
- Getting collection statistics
- Error handling
"""

from unittest.mock import Mock

import pytest

from docvec.mcp_tools.management_tools import ManagementTools
from docvec.storage.chroma_store import ChromaStore, StorageError


@pytest.fixture
def mock_storage():
    """Provide mock ChromaStore for testing."""
    storage = Mock(spec=ChromaStore)
    return storage


@pytest.fixture
def management_tools(mock_storage):
    """Provide ManagementTools instance with mocked dependencies."""
    return ManagementTools(storage=mock_storage)


class TestManagementToolsInitialization:
    """Test ManagementTools initialization."""

    def test_init_with_storage(self, mock_storage):
        """Test initialization with storage dependency."""
        tools = ManagementTools(storage=mock_storage)

        assert tools._storage is mock_storage


class TestDeleteByIds:
    """Test delete_by_ids functionality."""

    @pytest.mark.asyncio
    async def test_delete_by_ids_success(self, management_tools, mock_storage):
        """Test successful deletion by IDs."""
        ids_to_delete = ["1", "2", "3"]

        result = await management_tools.delete_by_ids(ids_to_delete)

        mock_storage.delete.assert_called_once_with(ids_to_delete)
        assert result["success"] is True
        assert result["data"]["deleted_count"] == 3
        assert result["data"]["deleted_ids"] == ids_to_delete
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_delete_by_ids_empty_list(self, management_tools, mock_storage):
        """Test deletion with empty ID list."""
        result = await management_tools.delete_by_ids([])

        mock_storage.delete.assert_not_called()
        assert result["success"] is True
        assert result["data"]["deleted_count"] == 0
        assert result["data"]["deleted_ids"] == []
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_delete_by_ids_storage_error(self, management_tools, mock_storage):
        """Test that storage errors are handled properly."""
        mock_storage.delete.side_effect = StorageError("Delete failed")

        result = await management_tools.delete_by_ids(["1"])

        assert result["success"] is False
        assert result["data"] is None
        assert "Delete failed" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_by_ids_invalid_input(self, management_tools):
        """Test that invalid input raises appropriate error."""
        result = await management_tools.delete_by_ids(None)

        assert result["success"] is False
        assert result["data"] is None
        assert "ids must be a list" in result["error"]


class TestDeleteByFile:
    """Test delete_by_file functionality."""

    @pytest.mark.asyncio
    async def test_delete_by_file_success(self, management_tools, mock_storage):
        """Test successful deletion by source file."""
        mock_storage.delete_by_source_file.return_value = 5

        result = await management_tools.delete_by_file("test.md")

        mock_storage.delete_by_source_file.assert_called_once_with("test.md")
        assert result["success"] is True
        assert result["data"]["deleted_count"] == 5
        assert result["data"]["source_file"] == "test.md"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_delete_by_file_not_found(self, management_tools, mock_storage):
        """Test deletion when file not found."""
        mock_storage.delete_by_source_file.return_value = 0

        result = await management_tools.delete_by_file("nonexistent.md")

        assert result["success"] is True
        assert result["data"]["deleted_count"] == 0
        assert result["data"]["source_file"] == "nonexistent.md"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_delete_by_file_empty_path(self, management_tools):
        """Test deletion with empty source file path."""
        result = await management_tools.delete_by_file("")

        assert result["success"] is False
        assert result["data"] is None
        assert "source_file cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_by_file_whitespace_only(self, management_tools):
        """Test deletion with whitespace-only path."""
        result = await management_tools.delete_by_file("   ")

        assert result["success"] is False
        assert result["data"] is None
        assert "source_file cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_by_file_storage_error(self, management_tools, mock_storage):
        """Test that storage errors are handled properly."""
        mock_storage.delete_by_source_file.side_effect = StorageError("Delete failed")

        result = await management_tools.delete_by_file("test.md")

        assert result["success"] is False
        assert result["data"] is None
        assert "Delete failed" in result["error"]


class TestDeleteAll:
    """Test delete_all functionality."""

    @pytest.mark.asyncio
    async def test_delete_all_success_with_confirm(
        self, management_tools, mock_storage
    ):
        """Test successful deletion of all documents with confirmation."""
        mock_storage.clear_collection.return_value = 100

        result = await management_tools.delete_all(confirm=True)

        mock_storage.clear_collection.assert_called_once()
        assert result["success"] is True
        assert result["data"]["deleted_count"] == 100
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_delete_all_without_confirm(self, management_tools, mock_storage):
        """Test deletion fails without confirmation."""
        result = await management_tools.delete_all(confirm=False)

        mock_storage.clear_collection.assert_not_called()
        assert result["success"] is False
        assert result["data"] is None
        assert "confirm=True" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_all_default_no_confirm(self, management_tools, mock_storage):
        """Test deletion fails with default (no) confirmation."""
        result = await management_tools.delete_all()

        mock_storage.clear_collection.assert_not_called()
        assert result["success"] is False
        assert result["data"] is None
        assert "confirm=True" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_all_storage_error(self, management_tools, mock_storage):
        """Test that storage errors are handled properly."""
        mock_storage.clear_collection.side_effect = StorageError("Clear failed")

        result = await management_tools.delete_all(confirm=True)

        assert result["success"] is False
        assert result["data"] is None
        assert "Clear failed" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_all_empty_collection(self, management_tools, mock_storage):
        """Test deletion of empty collection."""
        mock_storage.clear_collection.return_value = 0

        result = await management_tools.delete_all(confirm=True)

        assert result["success"] is True
        assert result["data"]["deleted_count"] == 0
        assert result["error"] is None


class TestGetStats:
    """Test get_stats functionality."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, management_tools, mock_storage):
        """Test successful stats retrieval."""
        mock_storage.get_stats.return_value = {
            "total_chunks": 100,
            "unique_files": 10,
            "source_files": ["file1.md", "file2.md"],
        }

        result = await management_tools.get_stats()

        mock_storage.get_stats.assert_called_once()
        assert result["success"] is True
        assert result["data"]["total_chunks"] == 100
        assert result["data"]["unique_files"] == 10
        assert result["data"]["source_files"] == ["file1.md", "file2.md"]
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_stats_empty_collection(self, management_tools, mock_storage):
        """Test stats for empty collection."""
        mock_storage.get_stats.return_value = {
            "total_chunks": 0,
            "unique_files": 0,
            "source_files": [],
        }

        result = await management_tools.get_stats()

        assert result["success"] is True
        assert result["data"]["total_chunks"] == 0
        assert result["data"]["unique_files"] == 0
        assert result["data"]["source_files"] == []
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_get_stats_storage_error(self, management_tools, mock_storage):
        """Test that storage errors are handled properly."""
        mock_storage.get_stats.side_effect = StorageError("Stats failed")

        result = await management_tools.get_stats()

        assert result["success"] is False
        assert result["data"] is None
        assert "Stats failed" in result["error"]


class TestIntegration:
    """Integration tests with real components."""

    @pytest.mark.asyncio
    async def test_full_delete_workflow(self, tmp_path):
        """Test complete delete workflow with real ChromaStore."""
        storage = ChromaStore(db_path=tmp_path / "test_db")

        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        documents = [
            "Document one content",
            "Document two content",
            "Document three content",
        ]
        metadatas = [
            {"doc_hash": "hash1", "source_file": "file1.md"},
            {"doc_hash": "hash2", "source_file": "file1.md"},
            {"doc_hash": "hash3", "source_file": "file2.md"},
        ]

        ids = storage.add(embeddings, documents, metadatas)

        tools = ManagementTools(storage=storage)

        # Test get_stats
        stats_result = await tools.get_stats()
        assert stats_result["success"] is True
        assert stats_result["data"]["total_chunks"] == 3
        assert stats_result["data"]["unique_files"] == 2

        # Test delete_by_ids
        delete_result = await tools.delete_by_ids([ids[0]])
        assert delete_result["success"] is True
        assert delete_result["data"]["deleted_count"] == 1

        # Verify count reduced
        stats_result = await tools.get_stats()
        assert stats_result["data"]["total_chunks"] == 2

        # Test delete_by_file
        delete_file_result = await tools.delete_by_file("file1.md")
        assert delete_file_result["success"] is True
        assert (
            delete_file_result["data"]["deleted_count"] == 1
        )  # Only one left from file1.md

        # Test delete_all
        delete_all_result = await tools.delete_all(confirm=True)
        assert delete_all_result["success"] is True
        assert (
            delete_all_result["data"]["deleted_count"] == 1
        )  # Only file2.md chunk left

        # Verify empty
        stats_result = await tools.get_stats()
        assert stats_result["data"]["total_chunks"] == 0
