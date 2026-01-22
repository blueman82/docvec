"""Tests for batch processor with deduplication."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from docvec.deduplication.hasher import DocumentHasher
from docvec.indexing.batch_processor import BatchProcessor, BatchResult
from docvec.indexing.indexer import IndexingError


@pytest.fixture
def mock_indexer():
    """Provide mock Indexer."""
    indexer = Mock()
    # Mock batch_size for accumulator threshold
    indexer.batch_size = 32

    # Mock chunk_file to return fake Chunk objects
    def chunk_file_side_effect(file_path):
        # Return 2 mock chunks per file
        chunk1 = Mock()
        chunk1.content = f"Content from {file_path.name} chunk 1"
        chunk1.source_file = str(file_path)
        chunk1.chunk_index = 0
        chunk1.metadata = {}

        chunk2 = Mock()
        chunk2.content = f"Content from {file_path.name} chunk 2"
        chunk2.source_file = str(file_path)
        chunk2.chunk_index = 1
        chunk2.metadata = {}

        return [chunk1, chunk2]

    indexer.chunk_file.side_effect = chunk_file_side_effect

    # Mock embed_and_store_chunks to return chunk IDs
    def embed_store_side_effect(chunks):
        return [f"chunk_id_{i}" for i in range(len(chunks))]

    indexer.embed_and_store_chunks.side_effect = embed_store_side_effect

    # Keep index_document for backwards compatibility tests
    indexer.index_document.return_value = ["chunk_id_1", "chunk_id_2"]
    return indexer


@pytest.fixture
def mock_hasher():
    """Provide mock DocumentHasher."""
    hasher = Mock()

    # Return different hashes for different files
    def hash_side_effect(file_path):
        return f"hash_{file_path.name}"

    hasher.hash_document.side_effect = hash_side_effect
    return hasher


@pytest.fixture
def mock_storage():
    """Provide mock ChromaStore."""
    storage = Mock()
    # Default: no duplicates (file not indexed, no content match)
    storage.get_by_source_file.return_value = None
    storage.get_by_hash.return_value = None
    return storage


@pytest.fixture
def processor(mock_indexer, mock_hasher, mock_storage):
    """Provide BatchProcessor instance with mocks."""
    return BatchProcessor(
        indexer=mock_indexer,
        hasher=mock_hasher,
        storage=mock_storage,
    )


@pytest.fixture
def temp_docs_dir(tmp_path):
    """Create temporary directory with test files."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create various file types
    (docs_dir / "file1.txt").write_text("Content 1")
    (docs_dir / "file2.md").write_text("# Markdown content")
    (docs_dir / "file3.py").write_text("def hello(): pass")
    (docs_dir / "README.md").write_text("# README")

    # Create subdirectory
    subdir = docs_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("Nested content")

    # Create unsupported file
    (docs_dir / "image.png").write_bytes(b"\x89PNG")

    return docs_dir


class TestBatchProcessorInitialization:
    """Test BatchProcessor initialization."""

    def test_init_with_dependencies(self, mock_indexer, mock_hasher, mock_storage):
        """Test initialization with all dependencies."""
        processor = BatchProcessor(
            indexer=mock_indexer,
            hasher=mock_hasher,
            storage=mock_storage,
        )

        assert processor.indexer == mock_indexer
        assert processor.hasher == mock_hasher
        assert processor.storage == mock_storage


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_batch_result_default_values(self):
        """Test BatchResult default initialization."""
        result = BatchResult()

        assert result.new_documents == 0
        assert result.duplicates_skipped == 0
        assert result.errors == []
        assert result.chunk_ids == {}

    def test_batch_result_with_values(self):
        """Test BatchResult with custom values."""
        result = BatchResult(
            new_documents=5,
            duplicates_skipped=2,
            errors=[("file1.txt", "error1")],
            chunk_ids={"file1.txt": ["id1", "id2"]},
        )

        assert result.new_documents == 5
        assert result.duplicates_skipped == 2
        assert len(result.errors) == 1
        assert len(result.chunk_ids) == 1


class TestBatchProcessorFileDiscovery:
    """Test file discovery functionality."""

    def test_find_files_recursive(self, processor, temp_docs_dir):
        """Test finding files recursively."""
        files = processor._find_files(temp_docs_dir, recursive=True)

        # Should find all files including nested (extension-agnostic)
        file_names = {Path(f).name for f in files}
        assert "file1.txt" in file_names
        assert "file2.md" in file_names
        assert "file3.py" in file_names
        assert "README.md" in file_names
        assert "nested.txt" in file_names
        # Now includes all file types (extension-agnostic)
        assert "image.png" in file_names

    def test_find_files_non_recursive(self, processor, temp_docs_dir):
        """Test finding files in single directory only."""
        files = processor._find_files(temp_docs_dir, recursive=False)

        file_names = {Path(f).name for f in files}
        assert "file1.txt" in file_names
        assert "file2.md" in file_names
        assert "README.md" in file_names
        # Includes all extensions now
        assert "image.png" in file_names

        # Should not include nested files
        assert "nested.txt" not in file_names

    def test_find_files_sorted_order(self, processor, temp_docs_dir):
        """Test that files are returned in sorted order."""
        files = processor._find_files(temp_docs_dir, recursive=False)

        # Verify sorted order
        file_names = [Path(f).name for f in files]
        assert file_names == sorted(file_names)

    def test_find_files_empty_directory(self, processor, tmp_path):
        """Test finding files in empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        files = processor._find_files(empty_dir, recursive=True)
        assert files == []

    def test_find_files_excludes_hidden(self, processor, tmp_path):
        """Test that hidden files (starting with .) are excluded."""
        dir_path = tmp_path / "with_hidden"
        dir_path.mkdir()

        (dir_path / "visible.txt").write_text("visible")
        (dir_path / ".hidden").write_text("hidden")
        (dir_path / ".gitignore").write_text("ignored")

        files = processor._find_files(dir_path, recursive=True)
        file_names = {Path(f).name for f in files}

        assert "visible.txt" in file_names
        assert ".hidden" not in file_names
        assert ".gitignore" not in file_names

    def test_find_files_any_extension(self, processor, tmp_path):
        """Test that any file extension is discovered (extension-agnostic)."""
        dir_path = tmp_path / "various"
        dir_path.mkdir()

        # Create files with various extensions
        (dir_path / "code.ts").write_text("typescript")
        (dir_path / "code.js").write_text("javascript")
        (dir_path / "data.json").write_text("{}")
        (dir_path / "config.yaml").write_text("key: value")
        (dir_path / "readme.rst").write_text("rst doc")

        files = processor._find_files(dir_path, recursive=True)
        file_names = {Path(f).name for f in files}

        assert "code.ts" in file_names
        assert "code.js" in file_names
        assert "data.json" in file_names
        assert "config.yaml" in file_names
        assert "readme.rst" in file_names


class TestBatchProcessorDeduplication:
    """Test deduplication functionality."""

    def test_is_duplicate_true_same_file_same_hash(
        self, processor, mock_storage, tmp_path
    ):
        """Test detecting duplicate when file path exists with same hash."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage: file is already indexed with same hash
        mock_storage.get_by_source_file.return_value = {
            "ids": ["existing_id"],
            "documents": ["Content"],
            "metadatas": [{"doc_hash": "abc123"}],
        }

        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is True
        mock_storage.get_by_source_file.assert_called_once_with(str(file_path))

    def test_is_duplicate_false_same_file_different_hash(
        self, processor, mock_storage, tmp_path
    ):
        """Test re-indexing when file content changed (different hash)."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("New content")

        # Mock storage: file exists with different hash (content changed)
        mock_storage.get_by_source_file.return_value = {
            "ids": ["existing_id"],
            "documents": ["Old content"],
            "metadatas": [{"doc_hash": "old_hash"}],
        }
        mock_storage.get_by_hash.return_value = None

        is_dup = processor._is_duplicate(file_path, "new_hash")

        # Should NOT be duplicate - will re-index
        assert is_dup is False
        # Should have deleted old chunks
        mock_storage.delete_by_source_file.assert_called_once_with(str(file_path))

    def test_is_duplicate_true_content_duplicate(
        self, processor, mock_storage, tmp_path
    ):
        """Test detecting content duplicate under different path."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage: file path not indexed, but same content exists
        mock_storage.get_by_source_file.return_value = None
        mock_storage.get_by_hash.return_value = {
            "ids": ["existing_id"],
            "documents": ["Content"],
            "metadatas": [{"doc_hash": "abc123"}],
        }

        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is True
        mock_storage.get_by_hash.assert_called_once_with("abc123")

    def test_is_duplicate_false_new_file(self, processor, mock_storage, tmp_path):
        """Test detecting new document (not indexed, no content match)."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage: neither file nor content exists
        mock_storage.get_by_source_file.return_value = None
        mock_storage.get_by_hash.return_value = None

        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is False

    def test_is_duplicate_storage_error(self, processor, mock_storage, tmp_path):
        """Test handling storage error during duplicate check."""
        from docvec.storage.chroma_store import StorageError

        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage to raise error
        mock_storage.get_by_source_file.side_effect = StorageError("DB error")

        # Should return False on error to avoid skipping files
        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is False


class TestBatchProcessorProcessFiles:
    """Test batch file processing."""

    def test_process_files_all_new(self, processor, tmp_path):
        """Test processing all new files."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        result = processor.process_files([file1, file2])

        assert result.new_documents == 2
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 0
        assert len(result.chunk_ids) == 2
        assert str(file1) in result.chunk_ids
        assert str(file2) in result.chunk_ids

    def test_process_files_with_duplicates(self, processor, mock_storage, tmp_path):
        """Test processing with some duplicates."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        # Mock storage to mark file1 as duplicate
        def get_by_hash_side_effect(file_hash):
            if file_hash == "hash_file1.txt":
                return {"ids": ["existing"], "documents": ["Content 1"]}
            return None

        mock_storage.get_by_hash.side_effect = get_by_hash_side_effect

        result = processor.process_files([file1, file2])

        assert result.new_documents == 1  # Only file2 indexed
        assert result.duplicates_skipped == 1
        assert len(result.errors) == 0
        assert str(file1) not in result.chunk_ids
        assert str(file2) in result.chunk_ids

    def test_process_files_with_errors(self, processor, mock_indexer, tmp_path):
        """Test processing with some files failing."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        # Mock chunk_file to fail on file1
        original_side_effect = mock_indexer.chunk_file.side_effect

        def chunk_side_effect(file_path):
            if file_path.name == "file1.txt":
                raise IndexingError("Failed to chunk")
            return original_side_effect(file_path)

        mock_indexer.chunk_file.side_effect = chunk_side_effect

        result = processor.process_files([file1, file2])

        assert result.new_documents == 1  # Only file2 succeeded
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 1
        assert result.errors[0][0] == str(file1)
        assert "Failed to chunk" in result.errors[0][1]

    def test_process_files_empty_list(self, processor):
        """Test processing empty file list."""
        result = processor.process_files([])

        assert result.new_documents == 0
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 0
        assert len(result.chunk_ids) == 0

    def test_process_files_all_fail(self, processor, mock_indexer, tmp_path):
        """Test processing when all files fail."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        # Mock chunk_file to always fail
        mock_indexer.chunk_file.side_effect = IndexingError("Failed to chunk")

        result = processor.process_files([file1, file2])

        assert result.new_documents == 0
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 2


class TestBatchProcessorProcessDirectory:
    """Test directory processing."""

    def test_process_directory_recursive(self, processor, temp_docs_dir):
        """Test processing directory recursively."""
        result = processor.process_directory(temp_docs_dir, recursive=True)

        # Should process all 6 files (5 in root + 1 nested) - extension-agnostic
        assert result.new_documents == 6
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 0

    def test_process_directory_non_recursive(self, processor, temp_docs_dir):
        """Test processing directory non-recursively."""
        result = processor.process_directory(temp_docs_dir, recursive=False)

        # Should process only 5 files in root (not nested) - extension-agnostic
        assert result.new_documents == 5
        assert result.duplicates_skipped == 0

    def test_process_directory_nonexistent(self, processor):
        """Test processing nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            processor.process_directory(Path("/nonexistent/dir"))

    def test_process_directory_not_a_directory(self, processor, tmp_path):
        """Test processing file as directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("Content")

        with pytest.raises(NotADirectoryError):
            processor.process_directory(file_path)

    def test_process_directory_empty(self, processor, tmp_path):
        """Test processing empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = processor.process_directory(empty_dir)

        assert result.new_documents == 0
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 0

    def test_process_directory_with_duplicates(
        self, processor, mock_storage, temp_docs_dir
    ):
        """Test directory processing with duplicates."""

        # Mock storage to mark some files as duplicates
        def get_by_hash_side_effect(file_hash):
            if file_hash in ["hash_file1.txt", "hash_README.md"]:
                return {"ids": ["existing"], "documents": ["content"]}
            return None

        mock_storage.get_by_hash.side_effect = get_by_hash_side_effect

        result = processor.process_directory(temp_docs_dir, recursive=False)

        # 5 files total (extension-agnostic), 2 duplicates, 3 new
        assert result.new_documents == 3
        assert result.duplicates_skipped == 2

    def test_process_directory_with_errors(
        self, processor, mock_indexer, temp_docs_dir
    ):
        """Test directory processing with some errors."""
        # Get the original chunk_file side effect
        original_side_effect = mock_indexer.chunk_file.side_effect

        # Mock chunk_file to fail on specific files
        def chunk_side_effect(file_path):
            if file_path.name == "file1.txt":
                raise IndexingError("Failed to chunk")
            return original_side_effect(file_path)

        mock_indexer.chunk_file.side_effect = chunk_side_effect

        result = processor.process_directory(temp_docs_dir, recursive=False)

        # 5 files (extension-agnostic), 1 error, 4 success
        assert result.new_documents == 4
        assert len(result.errors) == 1


class TestBatchProcessorIntegration:
    """Integration tests with real hasher."""

    def test_integration_with_real_hasher(self, mock_indexer, mock_storage, tmp_path):
        """Test batch processor with real DocumentHasher."""
        # Use real hasher
        hasher = DocumentHasher()
        processor = BatchProcessor(
            indexer=mock_indexer,
            hasher=hasher,
            storage=mock_storage,
        )

        # Create test files
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        # Create duplicate of file1
        file3 = tmp_path / "file3.txt"
        file3.write_text("Content 1")  # Same content as file1

        # Mock storage to detect duplicate based on actual hash
        stored_hashes = set()

        def get_by_hash_side_effect(file_hash):
            if file_hash in stored_hashes:
                return {"ids": ["existing"], "documents": ["content"]}
            stored_hashes.add(file_hash)
            return None

        mock_storage.get_by_hash.side_effect = get_by_hash_side_effect

        result = processor.process_files([file1, file2, file3])

        # file1 and file2 are new, file3 is duplicate of file1
        assert result.new_documents == 2
        assert result.duplicates_skipped == 1
        assert len(result.errors) == 0
