"""Tests for ChromaDB storage layer.

This module tests ChromaStore functionality including:
- Initialization and collection management
- Adding embeddings with metadata
- Semantic search with and without filters
- Document deletion
- Hash-based deduplication
"""

import tempfile
from pathlib import Path

import pytest

from docvec.storage.chroma_store import ChromaStore, StorageError


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary database path for testing."""
    return tmp_path / "test_chroma_db"


@pytest.fixture
def chroma_store(temp_db_path):
    """Provide ChromaStore instance for testing."""
    return ChromaStore(db_path=temp_db_path, collection_name="test_collection")


@pytest.fixture
def sample_embeddings():
    """Provide sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 0.8, 0.7, 0.6],
    ]


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "This is the first document about Python programming.",
        "This is the second document about data science.",
        "This is the third document about machine learning.",
    ]


@pytest.fixture
def sample_metadatas():
    """Provide sample metadata for testing."""
    return [
        {"doc_hash": "hash123", "source_file": "file1.py", "chunk_index": 0},
        {"doc_hash": "hash456", "source_file": "file2.md", "chunk_index": 0},
        {"doc_hash": "hash789", "source_file": "file1.py", "chunk_index": 1},
    ]


class TestChromaStoreInitialization:
    """Test ChromaStore initialization and setup."""

    def test_init_creates_db_directory(self, temp_db_path):
        """Test that initialization creates database directory."""
        assert not temp_db_path.exists()
        store = ChromaStore(db_path=temp_db_path)
        assert temp_db_path.exists()
        assert temp_db_path.is_dir()

    def test_init_with_existing_directory(self, temp_db_path):
        """Test initialization with existing directory."""
        temp_db_path.mkdir(parents=True, exist_ok=True)
        store = ChromaStore(db_path=temp_db_path)
        assert store.db_path == temp_db_path

    def test_init_with_custom_collection_name(self, temp_db_path):
        """Test initialization with custom collection name."""
        store = ChromaStore(db_path=temp_db_path, collection_name="custom_name")
        assert store.collection_name == "custom_name"

    def test_init_creates_collection(self, chroma_store):
        """Test that initialization creates collection."""
        assert chroma_store._collection is not None
        assert chroma_store._collection.name == "test_collection"

    def test_persistent_storage(self, temp_db_path):
        """Test that data persists across ChromaStore instances."""
        # Create first instance and add data
        store1 = ChromaStore(db_path=temp_db_path)
        ids1 = store1.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"doc_hash": "test_hash"}],
        )

        # Create second instance and verify data exists
        store2 = ChromaStore(db_path=temp_db_path)
        assert store2.count() == 1


class TestChromaStoreAdd:
    """Test adding documents to ChromaStore."""

    def test_add_single_document(self, chroma_store):
        """Test adding a single document."""
        ids = chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"doc_hash": "hash1"}],
        )

        assert len(ids) == 1
        assert isinstance(ids[0], str)
        assert chroma_store.count() == 1

    def test_add_multiple_documents(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test adding multiple documents."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert len(ids) == 3
        assert chroma_store.count() == 3

    def test_add_generates_unique_ids(self, chroma_store):
        """Test that add generates unique IDs."""
        ids1 = chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["doc1"],
            metadatas=[{"doc_hash": "hash1"}],
        )
        ids2 = chroma_store.add(
            embeddings=[[0.4, 0.5, 0.6]],
            documents=["doc2"],
            metadatas=[{"doc_hash": "hash2"}],
        )

        assert ids1[0] != ids2[0]

    def test_add_empty_list(self, chroma_store):
        """Test adding empty list returns empty list."""
        ids = chroma_store.add(embeddings=[], documents=[], metadatas=[])
        assert ids == []
        assert chroma_store.count() == 0

    def test_add_length_mismatch_raises_error(self, chroma_store):
        """Test that length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            chroma_store.add(
                embeddings=[[0.1, 0.2, 0.3]],
                documents=["doc1", "doc2"],  # Different length
                metadatas=[{"doc_hash": "hash1"}],
            )

    def test_add_preserves_metadata(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test that metadata is preserved during add."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        # Retrieve and verify metadata
        result = chroma_store.get_by_hash("hash123")
        assert result is not None
        assert result["metadatas"][0]["source_file"] == "file1.py"
        assert result["metadatas"][0]["chunk_index"] == 0


class TestChromaStoreSearch:
    """Test semantic search functionality."""

    def test_search_returns_results(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test basic search returns results."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4], n_results=2)

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert len(results["ids"]) <= 2

    def test_search_with_n_results(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test search respects n_results parameter."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(query_embedding=[0.5, 0.6, 0.7, 0.8], n_results=1)
        assert len(results["ids"]) == 1

        results = chroma_store.search(query_embedding=[0.5, 0.6, 0.7, 0.8], n_results=3)
        assert len(results["ids"]) == 3

    def test_search_with_metadata_filter(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test search with metadata filtering."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        # Search with source_file filter
        results = chroma_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            n_results=5,
            where={"source_file": "file1.py"},
        )

        # Should only return documents from file1.py
        assert len(results["ids"]) == 2  # Two chunks from file1.py
        for metadata in results["metadatas"]:
            assert metadata["source_file"] == "file1.py"

    def test_search_empty_collection(self, chroma_store):
        """Test search on empty collection returns empty results."""
        results = chroma_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert results["ids"] == []
        assert results["documents"] == []
        assert results["metadatas"] == []
        assert results["distances"] == []

    def test_search_returns_distances(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test that search returns distance scores."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        results = chroma_store.search(query_embedding=[0.1, 0.2, 0.3, 0.4])

        assert len(results["distances"]) > 0
        for distance in results["distances"]:
            assert isinstance(distance, (int, float))
            assert distance >= 0  # Distances should be non-negative


class TestChromaStoreDelete:
    """Test document deletion functionality."""

    def test_delete_single_document(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test deleting a single document."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        assert chroma_store.count() == 3
        chroma_store.delete([ids[0]])
        assert chroma_store.count() == 2

    def test_delete_multiple_documents(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test deleting multiple documents."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        chroma_store.delete([ids[0], ids[1]])
        assert chroma_store.count() == 1

    def test_delete_empty_list(self, chroma_store):
        """Test deleting empty list does nothing."""
        chroma_store.delete([])
        assert chroma_store.count() == 0

    def test_delete_nonexistent_id(self, chroma_store):
        """Test deleting nonexistent ID doesn't raise error."""
        # ChromaDB silently ignores nonexistent IDs
        chroma_store.delete(["nonexistent_id"])
        assert chroma_store.count() == 0


class TestChromaStoreGetByHash:
    """Test hash-based deduplication queries."""

    def test_get_by_hash_existing(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test getting document by existing hash."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )

        result = chroma_store.get_by_hash("hash123")

        assert result is not None
        assert len(result["ids"]) == 1
        assert result["metadatas"][0]["doc_hash"] == "hash123"
        assert "Python programming" in result["documents"][0]

    def test_get_by_hash_nonexistent(self, chroma_store):
        """Test getting document by nonexistent hash returns None."""
        result = chroma_store.get_by_hash("nonexistent_hash")
        assert result is None

    def test_get_by_hash_multiple_chunks_same_hash(self, chroma_store):
        """Test getting multiple chunks with same hash."""
        # Add multiple chunks with same hash (different chunk_index)
        chroma_store.add(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            documents=["chunk 1", "chunk 2"],
            metadatas=[
                {"doc_hash": "same_hash", "chunk_index": 0},
                {"doc_hash": "same_hash", "chunk_index": 1},
            ],
        )

        result = chroma_store.get_by_hash("same_hash")

        assert result is not None
        assert len(result["ids"]) == 2
        assert all(m["doc_hash"] == "same_hash" for m in result["metadatas"])


class TestChromaStoreCount:
    """Test document counting."""

    def test_count_empty_collection(self, chroma_store):
        """Test count on empty collection."""
        assert chroma_store.count() == 0

    def test_count_after_add(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test count after adding documents."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )
        assert chroma_store.count() == 3

    def test_count_after_delete(
        self, chroma_store, sample_embeddings, sample_documents, sample_metadatas
    ):
        """Test count after deleting documents."""
        ids = chroma_store.add(
            embeddings=sample_embeddings,
            documents=sample_documents,
            metadatas=sample_metadatas,
        )
        chroma_store.delete([ids[0]])
        assert chroma_store.count() == 2


class TestChromaStoreEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_embedding_dimension(self, chroma_store):
        """Test that inconsistent embedding dimensions are handled."""
        # First add with dimension 3
        chroma_store.add(
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["doc1"],
            metadatas=[{"doc_hash": "hash1"}],
        )

        # Try to add with different dimension
        with pytest.raises(StorageError):
            chroma_store.add(
                embeddings=[[0.1, 0.2]],  # Different dimension
                documents=["doc2"],
                metadatas=[{"doc_hash": "hash2"}],
            )

    def test_large_batch_add(self, chroma_store):
        """Test adding a large batch of documents."""
        batch_size = 100
        embeddings = [[0.1, 0.2, 0.3] for _ in range(batch_size)]
        documents = [f"document {i}" for i in range(batch_size)]
        metadatas = [{"doc_hash": f"hash{i}"} for i in range(batch_size)]

        ids = chroma_store.add(
            embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        assert len(ids) == batch_size
        assert chroma_store.count() == batch_size

    def test_search_with_invalid_filter(self, chroma_store, sample_embeddings):
        """Test search with nonexistent filter key."""
        chroma_store.add(
            embeddings=sample_embeddings,
            documents=["doc1", "doc2", "doc3"],
            metadatas=[{"key": "val1"}, {"key": "val2"}, {"key": "val3"}],
        )

        # Search with filter that doesn't match any documents
        results = chroma_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            where={"nonexistent_key": "value"},
        )

        assert results["ids"] == []
