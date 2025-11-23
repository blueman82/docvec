"""Tests for document hash calculator."""

from pathlib import Path

import pytest

from vector_mcp.deduplication.hasher import DocumentHasher


@pytest.fixture
def hasher():
    """Provide DocumentHasher instance."""
    return DocumentHasher()


@pytest.fixture
def temp_file(tmp_path):
    """Create temporary test file."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test content for hashing")
    return file_path


class TestDocumentHasher:
    """Test DocumentHasher functionality."""

    def test_hash_document_success(self, hasher, temp_file):
        """Test successful document hashing."""
        file_hash = hasher.hash_document(temp_file)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64  # SHA-256 produces 64 hex chars

    def test_hash_document_consistent(self, hasher, temp_file):
        """Test that hashing same file produces same hash."""
        hash1 = hasher.hash_document(temp_file)
        hash2 = hasher.hash_document(temp_file)

        assert hash1 == hash2

    def test_hash_document_different_content(self, hasher, tmp_path):
        """Test that different content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file1.write_text("Content 1")

        file2 = tmp_path / "file2.txt"
        file2.write_text("Content 2")

        hash1 = hasher.hash_document(file1)
        hash2 = hasher.hash_document(file2)

        assert hash1 != hash2

    def test_hash_document_nonexistent_file(self, hasher):
        """Test hashing nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            hasher.hash_document(Path("/nonexistent/file.txt"))

    def test_hash_document_directory(self, hasher, tmp_path):
        """Test hashing directory raises error."""
        with pytest.raises(IOError, match="not a file"):
            hasher.hash_document(tmp_path)

    def test_hash_document_binary_file(self, hasher, tmp_path):
        """Test hashing binary file."""
        file_path = tmp_path / "binary.bin"
        file_path.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        file_hash = hasher.hash_document(file_path)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64

    def test_hash_document_empty_file(self, hasher, tmp_path):
        """Test hashing empty file."""
        file_path = tmp_path / "empty.txt"
        file_path.write_text("")

        file_hash = hasher.hash_document(file_path)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64

    def test_hash_document_large_file(self, hasher, tmp_path):
        """Test hashing large file."""
        file_path = tmp_path / "large.txt"
        # Create 1MB file
        content = "x" * (1024 * 1024)
        file_path.write_text(content)

        file_hash = hasher.hash_document(file_path)

        assert isinstance(file_hash, str)
        assert len(file_hash) == 64

    def test_hash_content_success(self, hasher):
        """Test successful content hashing."""
        content = "Test content"
        content_hash = hasher.hash_content(content)

        assert isinstance(content_hash, str)
        assert len(content_hash) == 64

    def test_hash_content_consistent(self, hasher):
        """Test that hashing same content produces same hash."""
        content = "Test content"
        hash1 = hasher.hash_content(content)
        hash2 = hasher.hash_content(content)

        assert hash1 == hash2

    def test_hash_content_different(self, hasher):
        """Test that different content produces different hashes."""
        hash1 = hasher.hash_content("Content 1")
        hash2 = hasher.hash_content("Content 2")

        assert hash1 != hash2

    def test_hash_content_empty_string(self, hasher):
        """Test hashing empty string."""
        content_hash = hasher.hash_content("")

        assert isinstance(content_hash, str)
        assert len(content_hash) == 64

    def test_hash_content_unicode(self, hasher):
        """Test hashing unicode content."""
        content = "Hello 世界 🌍"
        content_hash = hasher.hash_content(content)

        assert isinstance(content_hash, str)
        assert len(content_hash) == 64

    def test_hash_matches_file_and_content(self, hasher, tmp_path):
        """Test that file hash matches content hash."""
        content = "Test content"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        # Note: file hash uses binary mode, content hash uses UTF-8
        # They should match for text files
        file_hash = hasher.hash_document(file_path)
        content_hash = hasher.hash_content(content)

        assert file_hash == content_hash
