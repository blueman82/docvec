"""Tests for document indexer."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from vector_mcp.chunking.base import Chunk
from vector_mcp.embedding.ollama_client import EmbeddingError
from vector_mcp.indexing.indexer import Indexer, IndexingError
from vector_mcp.storage.chroma_store import StorageError


@pytest.fixture
def mock_embedder():
    """Provide mock OllamaClient."""
    embedder = Mock()
    embedder.embed_batch.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    return embedder


@pytest.fixture
def mock_storage():
    """Provide mock ChromaStore."""
    storage = Mock()
    storage.add.return_value = ["id1", "id2", "id3"]
    return storage


@pytest.fixture
def indexer(mock_embedder, mock_storage):
    """Provide Indexer instance with mocks."""
    return Indexer(
        embedder=mock_embedder,
        storage=mock_storage,
        chunk_size=512,
        batch_size=32,
    )


@pytest.fixture
def temp_txt_file(tmp_path):
    """Create temporary text file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("This is a test document.\n\nSecond paragraph here.")
    return file_path


@pytest.fixture
def temp_py_file(tmp_path):
    """Create temporary Python file."""
    file_path = tmp_path / "test.py"
    content = '''def hello():
    """Say hello."""
    print("Hello, world!")
'''
    file_path.write_text(content)
    return file_path


class TestIndexerInitialization:
    """Test Indexer initialization."""

    def test_init_with_valid_params(self, mock_embedder, mock_storage):
        """Test initialization with valid parameters."""
        indexer = Indexer(
            embedder=mock_embedder,
            storage=mock_storage,
            chunk_size=512,
            batch_size=32,
        )

        assert indexer.embedder == mock_embedder
        assert indexer.storage == mock_storage
        assert indexer.chunk_size == 512
        assert indexer.batch_size == 32

    def test_init_with_default_params(self, mock_embedder, mock_storage):
        """Test initialization uses default parameters."""
        indexer = Indexer(embedder=mock_embedder, storage=mock_storage)

        assert indexer.chunk_size == 512
        assert indexer.batch_size == 32

    def test_init_invalid_chunk_size(self, mock_embedder, mock_storage):
        """Test that invalid chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Indexer(
                embedder=mock_embedder, storage=mock_storage, chunk_size=0
            )

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Indexer(
                embedder=mock_embedder, storage=mock_storage, chunk_size=-10
            )

    def test_init_invalid_batch_size(self, mock_embedder, mock_storage):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            Indexer(
                embedder=mock_embedder, storage=mock_storage, batch_size=0
            )

    def test_chunker_map_initialized(self, indexer):
        """Test that chunker map is properly initialized."""
        assert ".py" in indexer._chunker_map
        assert ".txt" in indexer._chunker_map


class TestIndexerFileLoading:
    """Test file loading functionality."""

    def test_load_file_utf8(self, indexer, temp_txt_file):
        """Test loading UTF-8 encoded file."""
        content = indexer._load_file(temp_txt_file)

        assert "This is a test document" in content
        assert "Second paragraph" in content

    def test_load_file_nonexistent(self, indexer):
        """Test loading nonexistent file raises error."""
        with pytest.raises(IndexingError, match="Failed to read"):
            indexer._load_file(Path("/nonexistent/file.txt"))

    def test_load_file_fallback_encoding(self, indexer, tmp_path):
        """Test fallback to latin-1 encoding."""
        # Create file with latin-1 encoding
        file_path = tmp_path / "latin1.txt"
        content = "Café résumé"
        file_path.write_bytes(content.encode("latin-1"))

        loaded = indexer._load_file(file_path)
        assert loaded is not None


class TestIndexerChunkerSelection:
    """Test chunker selection logic."""

    def test_select_chunker_python_file(self, indexer, tmp_path):
        """Test selecting CodeChunker for .py files."""
        from vector_mcp.chunking.code_chunker import CodeChunker

        file_path = tmp_path / "test.py"
        file_path.touch()

        chunker = indexer._select_chunker(file_path)
        assert isinstance(chunker, CodeChunker)

    def test_select_chunker_text_file(self, indexer, tmp_path):
        """Test selecting TextChunker for .txt files."""
        from vector_mcp.chunking.text_chunker import TextChunker

        file_path = tmp_path / "test.txt"
        file_path.touch()

        chunker = indexer._select_chunker(file_path)
        assert isinstance(chunker, TextChunker)

    def test_select_chunker_default_fallback(self, indexer, tmp_path):
        """Test default fallback to TextChunker for unknown extensions."""
        from vector_mcp.chunking.text_chunker import TextChunker

        file_path = tmp_path / "test.unknown"
        file_path.touch()

        chunker = indexer._select_chunker(file_path)
        assert isinstance(chunker, TextChunker)

    def test_select_chunker_case_insensitive(self, indexer, tmp_path):
        """Test that extension matching is case-insensitive."""
        from vector_mcp.chunking.code_chunker import CodeChunker

        file_path = tmp_path / "test.PY"
        file_path.touch()

        chunker = indexer._select_chunker(file_path)
        assert isinstance(chunker, CodeChunker)


class TestIndexerChunkValidation:
    """Test chunk validation logic."""

    def test_validate_chunks_all_valid(self, indexer):
        """Test validating chunks when all are valid."""
        chunks = [
            Chunk(
                content="Short content",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                content="Another short chunk",
                source_file="test.txt",
                chunk_index=1,
                metadata={},
            ),
        ]

        valid = indexer._validate_chunks(chunks)
        assert len(valid) == 2

    def test_validate_chunks_filters_empty(self, indexer):
        """Test that empty chunks are filtered out."""
        chunks = [
            Chunk(
                content="Valid content",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                content="   ",
                source_file="test.txt",
                chunk_index=1,
                metadata={},
            ),
        ]

        valid = indexer._validate_chunks(chunks)
        assert len(valid) == 1
        assert valid[0].content == "Valid content"

    def test_validate_chunks_filters_oversized(self, indexer):
        """Test that oversized chunks are filtered out."""
        # Create chunk that exceeds token limit (512 tokens ~ 2048 chars)
        large_content = "x" * 3000

        chunks = [
            Chunk(
                content="Valid small chunk",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                content=large_content,
                source_file="test.txt",
                chunk_index=1,
                metadata={},
            ),
        ]

        valid = indexer._validate_chunks(chunks)
        assert len(valid) == 1
        assert valid[0].chunk_index == 0

    def test_validate_chunks_empty_list(self, indexer):
        """Test validating empty chunk list."""
        valid = indexer._validate_chunks([])
        assert valid == []


class TestIndexerEmbedding:
    """Test embedding generation."""

    def test_embed_chunks_success(self, indexer, mock_embedder):
        """Test successful embedding generation."""
        chunks = [
            Chunk(
                content="Chunk 1",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                content="Chunk 2",
                source_file="test.txt",
                chunk_index=1,
                metadata={},
            ),
        ]

        mock_embedder.embed_batch.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        embeddings = indexer._embed_chunks(chunks)

        assert len(embeddings) == 2
        mock_embedder.embed_batch.assert_called_once()

        # Verify correct texts were passed
        call_args = mock_embedder.embed_batch.call_args
        texts = call_args[0][0]
        assert texts == ["Chunk 1", "Chunk 2"]

    def test_embed_chunks_uses_batch_size(self, indexer, mock_embedder):
        """Test that batch_size is passed to embedder."""
        chunks = [
            Chunk(
                content=f"Chunk {i}",
                source_file="test.txt",
                chunk_index=i,
                metadata={},
            )
            for i in range(5)
        ]

        mock_embedder.embed_batch.return_value = [[0.1, 0.2]] * 5

        indexer._embed_chunks(chunks)

        call_args = mock_embedder.embed_batch.call_args
        assert call_args[1]["batch_size"] == 32

    def test_embed_chunks_empty_list(self, indexer):
        """Test embedding empty chunk list."""
        embeddings = indexer._embed_chunks([])
        assert embeddings == []

    def test_embed_chunks_embedding_error(self, indexer, mock_embedder):
        """Test that EmbeddingError is propagated."""
        chunks = [
            Chunk(
                content="Test",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            )
        ]

        mock_embedder.embed_batch.side_effect = EmbeddingError("API failed")

        with pytest.raises(EmbeddingError, match="API failed"):
            indexer._embed_chunks(chunks)


class TestIndexerStorage:
    """Test storage operations."""

    def test_store_chunks_success(self, indexer, mock_storage):
        """Test successful chunk storage."""
        chunks = [
            Chunk(
                content="Content 1",
                source_file="/path/file.txt",
                chunk_index=0,
                metadata={"extra": "data"},
            ),
            Chunk(
                content="Content 2",
                source_file="/path/file.txt",
                chunk_index=1,
                metadata={},
            ),
        ]

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        mock_storage.add.return_value = ["id1", "id2"]

        chunk_ids = indexer._store_chunks(chunks, embeddings)

        assert chunk_ids == ["id1", "id2"]
        mock_storage.add.assert_called_once()

        # Verify metadata includes doc_hash and source_file
        call_args = mock_storage.add.call_args
        metadatas = call_args[1]["metadatas"]
        assert len(metadatas) == 2
        assert "doc_hash" in metadatas[0]
        assert metadatas[0]["source_file"] == "/path/file.txt"
        assert metadatas[0]["chunk_index"] == 0
        assert metadatas[0]["extra"] == "data"

    def test_store_chunks_length_mismatch(self, indexer):
        """Test that length mismatch raises error."""
        chunks = [
            Chunk(
                content="Test",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            )
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # Different length

        with pytest.raises(IndexingError, match="length mismatch"):
            indexer._store_chunks(chunks, embeddings)

    def test_store_chunks_empty_list(self, indexer):
        """Test storing empty list."""
        chunk_ids = indexer._store_chunks([], [])
        assert chunk_ids == []

    def test_store_chunks_storage_error(self, indexer, mock_storage):
        """Test that StorageError is propagated."""
        chunks = [
            Chunk(
                content="Test",
                source_file="test.txt",
                chunk_index=0,
                metadata={},
            )
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        mock_storage.add.side_effect = StorageError("DB failed")

        with pytest.raises(StorageError, match="DB failed"):
            indexer._store_chunks(chunks, embeddings)

    def test_compute_hash_consistent(self, indexer):
        """Test that hash computation is consistent."""
        content = "Test content"

        hash1 = indexer._compute_hash(content)
        hash2 = indexer._compute_hash(content)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

    def test_compute_hash_different_content(self, indexer):
        """Test that different content produces different hashes."""
        hash1 = indexer._compute_hash("Content 1")
        hash2 = indexer._compute_hash("Content 2")

        assert hash1 != hash2


class TestIndexerDocumentIndexing:
    """Test end-to-end document indexing."""

    def test_index_document_text_file(self, indexer, temp_txt_file):
        """Test indexing a text file."""
        chunk_ids = indexer.index_document(temp_txt_file)

        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) > 0
        indexer.embedder.embed_batch.assert_called_once()
        indexer.storage.add.assert_called_once()

    def test_index_document_python_file(self, indexer, temp_py_file):
        """Test indexing a Python file."""
        chunk_ids = indexer.index_document(temp_py_file)

        assert isinstance(chunk_ids, list)
        assert len(chunk_ids) > 0

    def test_index_document_nonexistent_file(self, indexer):
        """Test indexing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            indexer.index_document(Path("/nonexistent/file.txt"))

    def test_index_document_not_a_file(self, indexer, tmp_path):
        """Test indexing directory raises error."""
        with pytest.raises(IndexingError, match="not a file"):
            indexer.index_document(tmp_path)

    def test_index_document_embedding_error(self, indexer, temp_txt_file, mock_embedder):
        """Test that embedding errors are wrapped in IndexingError."""
        mock_embedder.embed_batch.side_effect = EmbeddingError("Failed")

        with pytest.raises(IndexingError, match="Failed to index"):
            indexer.index_document(temp_txt_file)

    def test_index_document_storage_error(self, indexer, temp_txt_file, mock_storage):
        """Test that storage errors are wrapped in IndexingError."""
        mock_storage.add.side_effect = StorageError("DB error")

        with pytest.raises(IndexingError, match="Failed to index"):
            indexer.index_document(temp_txt_file)

    def test_index_document_no_valid_chunks(self, indexer, tmp_path):
        """Test indexing file that produces no valid chunks."""
        # Create file with only whitespace
        file_path = tmp_path / "empty.txt"
        file_path.write_text("   \n\n\t  ")

        # This will raise ValueError from chunker
        with pytest.raises(IndexingError):
            indexer.index_document(file_path)


class TestIndexerBatchIndexing:
    """Test batch indexing functionality."""

    def test_index_batch_multiple_files(self, indexer, tmp_path):
        """Test indexing multiple files."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        results = indexer.index_batch([file1, file2])

        assert str(file1) in results
        assert str(file2) in results
        assert len(results[str(file1)]) > 0
        assert len(results[str(file2)]) > 0

    def test_index_batch_handles_failures(self, indexer, tmp_path):
        """Test that batch indexing handles individual failures."""
        good_file = tmp_path / "good.txt"
        good_file.write_text("Valid content")

        bad_file = Path("/nonexistent/bad.txt")

        results = indexer.index_batch([good_file, bad_file])

        # Good file should succeed
        assert len(results[str(good_file)]) > 0

        # Bad file should have empty list
        assert results[str(bad_file)] == []

    def test_index_batch_empty_list(self, indexer):
        """Test indexing empty file list."""
        results = indexer.index_batch([])
        assert results == {}

    def test_index_batch_all_fail(self, indexer):
        """Test indexing when all files fail."""
        bad_files = [
            Path("/nonexistent/file1.txt"),
            Path("/nonexistent/file2.txt"),
        ]

        results = indexer.index_batch(bad_files)

        assert all(ids == [] for ids in results.values())


class TestIndexerIntegration:
    """Integration tests with real chunkers."""

    def test_integration_text_file_full_pipeline(self, mock_embedder, mock_storage, tmp_path):
        """Test full pipeline with real TextChunker."""
        indexer = Indexer(
            embedder=mock_embedder,
            storage=mock_storage,
            chunk_size=512,
            batch_size=32,
        )

        # Create realistic text file
        file_path = tmp_path / "article.txt"
        content = """Introduction to the topic.

First main paragraph with details. This provides context and examples.

Second main paragraph. This builds on the previous section.

Conclusion paragraph wrapping up."""
        file_path.write_text(content)

        mock_embedder.embed_batch.return_value = [
            [0.1, 0.2] * 100 for _ in range(4)
        ]
        mock_storage.add.return_value = ["id1", "id2", "id3", "id4"]

        chunk_ids = indexer.index_document(file_path)

        # Verify full pipeline executed
        assert len(chunk_ids) > 0
        mock_embedder.embed_batch.assert_called_once()
        mock_storage.add.assert_called_once()

    def test_integration_python_file_full_pipeline(self, mock_embedder, mock_storage, tmp_path):
        """Test full pipeline with real CodeChunker."""
        indexer = Indexer(
            embedder=mock_embedder,
            storage=mock_storage,
        )

        # Create realistic Python file
        file_path = tmp_path / "module.py"
        content = '''"""Module docstring."""
import os

def function1():
    """First function."""
    return 1

def function2():
    """Second function."""
    return 2

class MyClass:
    """A class."""
    def method(self):
        return 42
'''
        file_path.write_text(content)

        mock_embedder.embed_batch.return_value = [
            [0.1, 0.2] * 100 for _ in range(4)
        ]
        mock_storage.add.return_value = ["id1", "id2", "id3", "id4"]

        chunk_ids = indexer.index_document(file_path)

        assert len(chunk_ids) > 0
        mock_embedder.embed_batch.assert_called_once()

        # Verify chunks have correct metadata
        call_args = mock_storage.add.call_args
        metadatas = call_args[1]["metadatas"]

        # Should have imports chunk and function/class chunks
        assert any(m.get("type") == "imports" for m in metadatas)
