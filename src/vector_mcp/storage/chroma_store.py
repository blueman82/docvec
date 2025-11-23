"""ChromaDB storage layer with search, metadata filtering, and document management.

This module provides persistent vector storage using ChromaDB with support for:
- Semantic search with metadata filtering
- Hash-based deduplication tracking
- Auto-increment ID generation
- Persistent local storage
"""

import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection


class StorageError(Exception):
    """Custom exception for storage-related errors."""

    pass


class ChromaStore:
    """Vector storage layer using ChromaDB.

    Provides persistent storage for embeddings with metadata, semantic search,
    and deduplication tracking via document hashes.

    Args:
        db_path: Path to ChromaDB persistent storage directory
        collection_name: Name of the collection (default: "documents")

    Attributes:
        db_path: Path to database storage
        collection_name: Name of the active collection
        _client: ChromaDB persistent client instance
        _collection: ChromaDB collection instance
        _id_counter: Counter for generating unique IDs
    """

    def __init__(self, db_path: Path, collection_name: str = "documents"):
        """Initialize ChromaStore with persistent storage.

        Args:
            db_path: Path to ChromaDB persistent storage directory
            collection_name: Name of the collection to use

        Raises:
            StorageError: If database initialization fails
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self._id_counter = 0

        try:
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)

            # Initialize persistent ChromaDB client
            self._client = chromadb.PersistentClient(path=str(self.db_path))

            # Get or create collection
            self._collection = self._get_or_create_collection()

        except Exception as e:
            raise StorageError(f"Failed to initialize ChromaDB storage: {e}") from e

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one.

        Returns:
            ChromaDB Collection instance

        Raises:
            StorageError: If collection operations fail
        """
        try:
            # Try to get existing collection
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity for search
            )
            return collection

        except Exception as e:
            raise StorageError(f"Failed to get or create collection: {e}") from e

    def _generate_id(self) -> str:
        """Generate unique, sortable ID using timestamp and sequence.

        Returns:
            Unique ID string in format: timestamp_sequence

        Example:
            "1732360800_0", "1732360800_1", etc.
        """
        timestamp = int(time.time())
        unique_id = f"{timestamp}_{self._id_counter}"
        self._id_counter += 1
        return unique_id

    def add(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> list[str]:
        """Add embeddings with documents and metadata to storage.

        Args:
            embeddings: List of embedding vectors
            documents: List of document content strings
            metadatas: List of metadata dictionaries (must include doc_hash)

        Returns:
            List of generated IDs for the added documents

        Raises:
            StorageError: If add operation fails
            ValueError: If input lists have different lengths

        Example:
            >>> embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            >>> documents = ["chunk 1", "chunk 2"]
            >>> metadatas = [{"doc_hash": "abc123"}, {"doc_hash": "def456"}]
            >>> ids = store.add(embeddings, documents, metadatas)
        """
        if not (len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(
                f"Length mismatch: embeddings={len(embeddings)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}"
            )

        if not embeddings:
            return []

        try:
            # Generate unique IDs for each document
            ids = [self._generate_id() for _ in range(len(embeddings))]

            # Add to ChromaDB collection
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            return ids

        except Exception as e:
            raise StorageError(f"Failed to add documents to storage: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """Perform semantic search with optional metadata filtering.

        Args:
            query_embedding: Query vector to search for
            n_results: Number of results to return (default: 5)
            where: Optional metadata filter (e.g., {"source_file": "readme.md"})

        Returns:
            Dictionary with search results containing:
                - ids: List of document IDs
                - documents: List of document content
                - metadatas: List of metadata dictionaries
                - distances: List of distance scores (lower is better)

        Raises:
            StorageError: If search operation fails

        Example:
            >>> results = store.search([0.1, 0.2, 0.3], n_results=5)
            >>> filtered = store.search(
            ...     [0.1, 0.2, 0.3],
            ...     where={"source_file": "readme.md"}
            ... )
        """
        try:
            # Perform semantic search
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # ChromaDB returns results wrapped in lists (for batch queries)
            # Extract the first (and only) result set
            return {
                "ids": results["ids"][0] if results["ids"] else [],
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }

        except Exception as e:
            raise StorageError(f"Failed to search documents: {e}") from e

    def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Raises:
            StorageError: If delete operation fails

        Example:
            >>> store.delete(["1732360800_0", "1732360800_1"])
        """
        if not ids:
            return

        try:
            self._collection.delete(ids=ids)

        except Exception as e:
            raise StorageError(f"Failed to delete documents: {e}") from e

    def get_by_hash(self, doc_hash: str) -> Optional[dict]:
        """Get document by hash for deduplication checking.

        Args:
            doc_hash: Document hash to search for

        Returns:
            Dictionary with document data if found, None otherwise.
            Contains: ids, documents, metadatas

        Raises:
            StorageError: If query operation fails

        Example:
            >>> result = store.get_by_hash("abc123def456")
            >>> if result:
            ...     print(f"Found {len(result['ids'])} documents")
        """
        try:
            # Query for documents with matching hash
            results = self._collection.get(
                where={"doc_hash": doc_hash},
                include=["documents", "metadatas"],
            )

            # Return None if no results found
            if not results["ids"]:
                return None

            return {
                "ids": results["ids"],
                "documents": results["documents"],
                "metadatas": results["metadatas"],
            }

        except Exception as e:
            raise StorageError(f"Failed to get document by hash: {e}") from e

    def count(self) -> int:
        """Get total number of documents in collection.

        Returns:
            Number of documents stored

        Raises:
            StorageError: If count operation fails
        """
        try:
            return self._collection.count()

        except Exception as e:
            raise StorageError(f"Failed to count documents: {e}") from e
