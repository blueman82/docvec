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
    # Default: no duplicates
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

    def test_supported_extensions(self, processor):
        """Test that supported extensions are correctly set."""
        expected = {".md", ".pdf", ".txt", ".py"}
        assert processor.supported_extensions == expected


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

        # Should find all supported files including nested
        file_names = {Path(f).name for f in files}
        assert "file1.txt" in file_names
        assert "file2.md" in file_names
        assert "file3.py" in file_names
        assert "README.md" in file_names
        assert "nested.txt" in file_names

        # Should not include unsupported files
        assert "image.png" not in file_names

    def test_find_files_non_recursive(self, processor, temp_docs_dir):
        """Test finding files in single directory only."""
        files = processor._find_files(temp_docs_dir, recursive=False)

        file_names = {Path(f).name for f in files}
        assert "file1.txt" in file_names
        assert "file2.md" in file_names
        assert "README.md" in file_names

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

    def test_find_files_only_unsupported(self, processor, tmp_path):
        """Test directory with only unsupported files."""
        dir_path = tmp_path / "unsupported"
        dir_path.mkdir()

        (dir_path / "image1.png").write_bytes(b"data")
        (dir_path / "image2.jpg").write_bytes(b"data")

        files = processor._find_files(dir_path, recursive=True)
        assert files == []


class TestBatchProcessorDeduplication:
    """Test deduplication functionality."""

    def test_is_duplicate_true(self, processor, mock_storage, tmp_path):
        """Test detecting duplicate document."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage to return existing document
        mock_storage.get_by_hash.return_value = {
            "ids": ["existing_id"],
            "documents": ["Content"],
            "metadatas": [{"doc_hash": "abc123"}],
        }

        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is True
        mock_storage.get_by_hash.assert_called_once_with("abc123")

    def test_is_duplicate_false(self, processor, mock_storage, tmp_path):
        """Test detecting non-duplicate document."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage to return None (not found)
        mock_storage.get_by_hash.return_value = None

        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is False

    def test_is_duplicate_storage_error(self, processor, mock_storage, tmp_path):
        """Test handling storage error during duplicate check."""
        from docvec.storage.chroma_store import StorageError

        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock storage to raise error
        mock_storage.get_by_hash.side_effect = StorageError("DB error")

        # Should return False on error to avoid skipping files
        is_dup = processor._is_duplicate(file_path, "abc123")

        assert is_dup is False


class TestBatchProcessorSingleFile:
    """Test single file processing."""

    def test_process_single_file_success(self, processor, mock_indexer, tmp_path):
        """Test successful single file processing."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content")

        chunk_ids = processor._process_single_file(file_path, "hash123")

        assert chunk_ids == ["chunk_id_1", "chunk_id_2"]
        mock_indexer.index_document.assert_called_once_with(file_path)

    def test_process_single_file_no_chunks(self, processor, mock_indexer, tmp_path):
        """Test processing file that generates no chunks."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock indexer to return empty list
        mock_indexer.index_document.return_value = []

        chunk_ids = processor._process_single_file(file_path, "hash123")

        assert chunk_ids is None

    def test_process_single_file_indexing_error(
        self, processor, mock_indexer, tmp_path
    ):
        """Test handling indexing error."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Content")

        # Mock indexer to raise error
        mock_indexer.index_document.side_effect = IndexingError("Failed")

        with pytest.raises(IndexingError, match="Failed to index"):
            processor._process_single_file(file_path, "hash123")

    def test_process_single_file_not_found(self, processor, mock_indexer):
        """Test processing nonexistent file."""
        file_path = Path("/nonexistent/file.txt")

        mock_indexer.index_document.side_effect = FileNotFoundError("Not found")

        with pytest.raises(IndexingError, match="Failed to index"):
            processor._process_single_file(file_path, "hash123")


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

        # Mock indexer to fail on file1
        def index_side_effect(file_path):
            if file_path.name == "file1.txt":
                raise IndexingError("Failed to index")
            return ["chunk_id_1", "chunk_id_2"]

        mock_indexer.index_document.side_effect = index_side_effect

        result = processor.process_files([file1, file2])

        assert result.new_documents == 1  # Only file2 succeeded
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 1
        assert result.errors[0][0] == str(file1)
        assert "Failed to index" in result.errors[0][1]

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

        # Mock indexer to always fail
        mock_indexer.index_document.side_effect = IndexingError("Failed")

        result = processor.process_files([file1, file2])

        assert result.new_documents == 0
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 2


class TestBatchProcessorProcessDirectory:
    """Test directory processing."""

    def test_process_directory_recursive(self, processor, temp_docs_dir):
        """Test processing directory recursively."""
        result = processor.process_directory(temp_docs_dir, recursive=True)

        # Should process all 5 supported files (4 in root + 1 nested)
        assert result.new_documents == 5
        assert result.duplicates_skipped == 0
        assert len(result.errors) == 0

    def test_process_directory_non_recursive(self, processor, temp_docs_dir):
        """Test processing directory non-recursively."""
        result = processor.process_directory(temp_docs_dir, recursive=False)

        # Should process only 4 files in root (not nested)
        assert result.new_documents == 4
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

        # 4 files total, 2 duplicates, 2 new
        assert result.new_documents == 2
        assert result.duplicates_skipped == 2

    def test_process_directory_with_errors(
        self, processor, mock_indexer, temp_docs_dir
    ):
        """Test directory processing with some errors."""

        # Mock indexer to fail on specific files
        def index_side_effect(file_path):
            if file_path.name == "file1.txt":
                raise IndexingError("Failed")
            return ["chunk_id_1", "chunk_id_2"]

        mock_indexer.index_document.side_effect = index_side_effect

        result = processor.process_directory(temp_docs_dir, recursive=False)

        # 4 files, 1 error, 3 success
        assert result.new_documents == 3
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
