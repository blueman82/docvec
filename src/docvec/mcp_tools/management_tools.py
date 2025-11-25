"""MCP management tools for delete operations and collection management.

This module provides management tools for ChromaDB operations including:
- Delete chunks by IDs
- Delete chunks by source file
- Clear entire collection with safety gate
- Get collection statistics
"""

from typing import Any

from docvec.storage.chroma_store import ChromaStore, StorageError


class ManagementError(Exception):
    """Custom exception for management-related errors."""

    pass


class ManagementTools:
    """MCP tools for collection management operations.

    Provides tools for managing documents in ChromaDB including deletion
    operations and statistics retrieval.

    Args:
        storage: ChromaStore instance for document management

    Attributes:
        _storage: ChromaDB storage layer

    Example:
        >>> storage = ChromaStore(Path("./db"))
        >>> tools = ManagementTools(storage)
        >>> result = await tools.delete_by_file("old_document.md")
    """

    def __init__(self, storage: ChromaStore):
        """Initialize management tools with dependencies.

        Args:
            storage: ChromaStore for document management
        """
        self._storage = storage

    async def delete_by_ids(self, ids: list[str]) -> dict[str, Any]:
        """Delete specific chunks by their IDs.

        Args:
            ids: List of chunk IDs to delete

        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - data: Deletion details (deleted_count, deleted_ids) if successful
                - error: Error message if failed

        Example:
            >>> result = await tools.delete_by_ids(["1732360800_0", "1732360800_1"])
            >>> print(f"Deleted {result['data']['deleted_count']} chunks")
        """
        if ids is None:
            return {
                "success": False,
                "data": None,
                "error": "ids must be a list",
            }

        if not ids:
            return {
                "success": True,
                "data": {"deleted_count": 0, "deleted_ids": []},
                "error": None,
            }

        try:
            self._storage.delete(ids)
            return {
                "success": True,
                "data": {"deleted_count": len(ids), "deleted_ids": ids},
                "error": None,
            }
        except StorageError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to delete by IDs: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Unexpected error: {e}",
            }

    async def delete_by_file(self, source_file: str) -> dict[str, Any]:
        """Delete all chunks from a specific source file.

        Args:
            source_file: Source file path to delete chunks for

        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - data: Deletion details (deleted_count, source_file) if successful
                - error: Error message if failed

        Example:
            >>> result = await tools.delete_by_file("outdated_docs.md")
            >>> print(f"Deleted {result['data']['deleted_count']} chunks from file")
        """
        if not source_file or not source_file.strip():
            return {
                "success": False,
                "data": None,
                "error": "source_file cannot be empty",
            }

        try:
            deleted_count = self._storage.delete_by_source_file(source_file)
            return {
                "success": True,
                "data": {"deleted_count": deleted_count, "source_file": source_file},
                "error": None,
            }
        except StorageError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to delete by file: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Unexpected error: {e}",
            }

    async def delete_all(self, confirm: bool = False) -> dict[str, Any]:
        """Delete all documents from the collection.

        Requires explicit confirmation to prevent accidental data loss.

        Args:
            confirm: Must be True to proceed with deletion

        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - data: Deletion details (deleted_count) if successful
                - error: Error message if failed or not confirmed

        Example:
            >>> result = await tools.delete_all(confirm=True)
            >>> print(f"Cleared {result['data']['deleted_count']} chunks")
        """
        if not confirm:
            return {
                "success": False,
                "data": None,
                "error": "Safety check: Set confirm=True to delete all documents",
            }

        try:
            deleted_count = self._storage.clear_collection()
            return {
                "success": True,
                "data": {"deleted_count": deleted_count},
                "error": None,
            }
        except StorageError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to clear collection: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Unexpected error: {e}",
            }

    async def get_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dictionary containing:
                - success: Whether the operation succeeded
                - data: Stats (total_chunks, unique_files, source_files) if successful
                - error: Error message if failed

        Example:
            >>> result = await tools.get_stats()
            >>> print(f"Collection has {result['data']['total_chunks']} chunks")
        """
        try:
            stats = self._storage.get_stats()
            return {
                "success": True,
                "data": stats,
                "error": None,
            }
        except StorageError as e:
            return {
                "success": False,
                "data": None,
                "error": f"Failed to get stats: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": f"Unexpected error: {e}",
            }
