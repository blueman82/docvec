"""Tests for base chunking interface and Chunk dataclass."""

import pytest

from docvec.chunking.base import AbstractChunker, Chunk


class TestChunk:
    """Test suite for Chunk dataclass."""

    def test_chunk_creation_valid(self):
        """Test creating a valid chunk."""
        chunk = Chunk(
            content="This is test content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunk.content == "This is test content"
        assert chunk.source_file == "/path/to/file.txt"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {}
        assert chunk.token_count is None

    def test_chunk_creation_with_metadata(self):
        """Test creating chunk with metadata."""
        metadata = {"page": 1, "section": "introduction"}
        chunk = Chunk(
            content="Content with metadata",
            source_file="/path/to/doc.pdf",
            chunk_index=5,
            metadata=metadata,
        )
        assert chunk.metadata == metadata
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["section"] == "introduction"

    def test_chunk_creation_with_token_count(self):
        """Test creating chunk with token count."""
        chunk = Chunk(
            content="Content with tokens",
            source_file="/path/to/file.txt",
            chunk_index=0,
            token_count=42,
        )
        assert chunk.token_count == 42

    def test_chunk_immutability(self):
        """Test that chunks are immutable (frozen)."""
        chunk = Chunk(
            content="Immutable content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        with pytest.raises(AttributeError):
            chunk.content = "Modified content"

    def test_chunk_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        with pytest.raises(ValueError, match="Chunk content cannot be empty"):
            Chunk(
                content="",
                source_file="/path/to/file.txt",
                chunk_index=0,
            )

    def test_chunk_whitespace_only_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        with pytest.raises(ValueError, match="Chunk content cannot be empty"):
            Chunk(
                content="   \n\t  ",
                source_file="/path/to/file.txt",
                chunk_index=0,
            )

    def test_chunk_negative_index_raises_error(self):
        """Test that negative chunk index raises ValueError."""
        with pytest.raises(ValueError, match="Chunk index must be non-negative"):
            Chunk(
                content="Valid content",
                source_file="/path/to/file.txt",
                chunk_index=-1,
            )

    def test_chunk_validate_content_method(self):
        """Test validate_content method directly."""
        chunk = Chunk(
            content="Valid content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        # Should not raise
        chunk.validate_content()

    def test_chunk_equality(self):
        """Test chunk equality comparison."""
        chunk1 = Chunk(
            content="Same content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        chunk2 = Chunk(
            content="Same content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunk1 == chunk2

    def test_chunk_inequality_different_content(self):
        """Test chunk inequality with different content."""
        chunk1 = Chunk(
            content="Content A",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        chunk2 = Chunk(
            content="Content B",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunk1 != chunk2

    def test_chunk_inequality_different_index(self):
        """Test chunk inequality with different index."""
        chunk1 = Chunk(
            content="Same content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        chunk2 = Chunk(
            content="Same content",
            source_file="/path/to/file.txt",
            chunk_index=1,
        )
        assert chunk1 != chunk2


class ConcreteChunker(AbstractChunker):
    """Concrete implementation of AbstractChunker for testing."""

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Simple implementation that splits on newlines."""
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        lines = [line.strip() for line in content.split("\n") if line.strip()]
        return [
            Chunk(
                content=line,
                source_file=source_file,
                chunk_index=idx,
            )
            for idx, line in enumerate(lines)
        ]


class TestAbstractChunker:
    """Test suite for AbstractChunker interface."""

    def test_concrete_chunker_implementation(self):
        """Test that concrete implementation works."""
        chunker = ConcreteChunker()
        content = "Line 1\nLine 2\nLine 3"
        chunks = chunker.chunk(content, "/path/to/file.txt")

        assert len(chunks) == 3
        assert chunks[0].content == "Line 1"
        assert chunks[0].chunk_index == 0
        assert chunks[1].content == "Line 2"
        assert chunks[1].chunk_index == 1
        assert chunks[2].content == "Line 3"
        assert chunks[2].chunk_index == 2

    def test_chunker_empty_content_raises_error(self):
        """Test that chunker raises error for empty content."""
        chunker = ConcreteChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/path/to/file.txt")

    def test_validate_chunk_quality_valid(self):
        """Test validate_chunk_quality with valid chunk."""
        chunker = ConcreteChunker()
        chunk = Chunk(
            content="Valid content",
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        assert chunker.validate_chunk_quality(chunk) is True

    def test_validate_chunk_quality_invalid(self):
        """Test validate_chunk_quality with invalid chunk."""
        chunker = ConcreteChunker()

        # Create an invalid chunk by bypassing validation
        # We'll test with a chunk that would fail validation
        class InvalidChunk:
            def __init__(self):
                self.content = ""
                self.chunk_index = -1

            def validate_content(self):
                raise ValueError("Invalid chunk")

        invalid = InvalidChunk()
        assert chunker.validate_chunk_quality(invalid) is False

    def test_abstract_chunker_cannot_instantiate(self):
        """Test that AbstractChunker cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AbstractChunker()

    def test_chunker_preserves_source_file(self):
        """Test that source file is preserved in all chunks."""
        chunker = ConcreteChunker()
        source_file = "/documents/important.txt"
        content = "Line 1\nLine 2"
        chunks = chunker.chunk(content, source_file)

        for chunk in chunks:
            assert chunk.source_file == source_file

    def test_chunker_sequential_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = ConcreteChunker()
        content = "A\nB\nC\nD\nE"
        chunks = chunker.chunk(content, "/path/to/file.txt")

        assert len(chunks) == 5
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx


class TestSplitOversizedChunk:
    """Test suite for split_oversized_chunk method."""

    def test_chunk_under_limit_returns_unchanged(self):
        """Test that a chunk under the token limit is returned unchanged."""
        chunker = ConcreteChunker()
        chunk = Chunk(
            content="Short content",
            source_file="/path/to/file.txt",
            chunk_index=0,
            metadata={"existing": "data"},
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=100)

        assert len(result) == 1
        assert result[0].content == "Short content"
        assert result[0].chunk_index == 0
        assert result[0].metadata["existing"] == "data"
        assert "split_part" not in result[0].metadata

    def test_chunk_at_exact_limit_returns_unchanged(self):
        """Test that a chunk at exactly the token limit is returned unchanged."""
        chunker = ConcreteChunker()
        # 40 chars = 10 tokens (using chars/4 approximation)
        content = "a" * 40
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=10)

        assert len(result) == 1
        assert result[0].content == content

    def test_split_at_paragraph_boundaries(self):
        """Test that oversized chunks are split at paragraph boundaries."""
        chunker = ConcreteChunker()
        # Create content with clear paragraph boundaries
        # Each paragraph ~10 tokens, total ~30 tokens
        content = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        # Set limit to ~15 tokens (60 chars) to force splits
        result = chunker.split_oversized_chunk(chunk, max_tokens=15)

        assert len(result) > 1
        # All parts should have split_part metadata
        for i, part in enumerate(result):
            assert part.metadata["split_part"] == i
            assert part.source_file == "/path/to/file.txt"

    def test_split_preserves_source_file(self):
        """Test that split chunks preserve the source file."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two.\n\nPara three."
        chunk = Chunk(
            content=content,
            source_file="/important/document.txt",
            chunk_index=5,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        for part in result:
            assert part.source_file == "/important/document.txt"

    def test_split_adds_split_part_metadata(self):
        """Test that split chunks have split_part metadata."""
        chunker = ConcreteChunker()
        content = "First part.\n\nSecond part.\n\nThird part."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        assert len(result) > 1
        for i, part in enumerate(result):
            assert part.metadata["split_part"] == i

    def test_split_preserves_existing_metadata(self):
        """Test that split preserves existing metadata from original chunk."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two.\n\nPara three."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
            metadata={"page": 5, "section": "intro"},
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        for part in result:
            assert part.metadata["page"] == 5
            assert part.metadata["section"] == "intro"

    def test_split_uses_base_index_for_chunk_indices(self):
        """Test that split uses base_index parameter for chunk indices."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two.\n\nPara three."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5, base_index=10)

        assert result[0].chunk_index == 10
        if len(result) > 1:
            assert result[1].chunk_index == 11

    def test_split_falls_back_to_line_boundaries(self):
        """Test fallback to line boundaries when paragraphs are too large."""
        chunker = ConcreteChunker()
        # Single paragraph with multiple lines
        content = "Line one.\nLine two.\nLine three.\nLine four."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        assert len(result) > 1
        # Should split at line boundaries
        for part in result:
            assert part.content.strip()

    def test_split_handles_very_long_single_line(self):
        """Test handling of a very long single line that exceeds token limit."""
        chunker = ConcreteChunker()
        # Very long line with no natural break points
        content = "word " * 100  # ~500 chars = ~125 tokens
        chunk = Chunk(
            content=content.strip(),
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=20)

        assert len(result) > 1
        # All parts should be within limits (approximately)
        for part in result:
            # Each part should be roughly under the limit
            assert len(part.content) < 100  # ~25 tokens max with some buffer

    def test_split_empty_result_for_whitespace_paragraphs(self):
        """Test that whitespace-only paragraphs are filtered out."""
        chunker = ConcreteChunker()
        content = "Content one.\n\n   \n\nContent two."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=0,
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        # Should not include empty chunks
        for part in result:
            assert part.content.strip()

    def test_split_with_default_base_index(self):
        """Test that default base_index is 0."""
        chunker = ConcreteChunker()
        content = "Para one.\n\nPara two."
        chunk = Chunk(
            content=content,
            source_file="/path/to/file.txt",
            chunk_index=5,  # Original index doesn't matter
        )
        result = chunker.split_oversized_chunk(chunk, max_tokens=5)

        assert result[0].chunk_index == 0
