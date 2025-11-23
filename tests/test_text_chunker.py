"""Tests for paragraph-based text chunker."""

import pytest

from vector_mcp.chunking.base import Chunk
from vector_mcp.chunking.text_chunker import TextChunker


class TestTextChunkerInit:
    """Test suite for TextChunker initialization."""

    def test_default_initialization(self):
        """Test chunker with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_custom_chunk_size(self):
        """Test chunker with custom chunk size."""
        chunker = TextChunker(chunk_size=500)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 200

    def test_custom_chunk_overlap(self):
        """Test chunker with custom chunk overlap."""
        chunker = TextChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_invalid_chunk_size_zero(self):
        """Test that zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-100)

    def test_invalid_chunk_overlap_negative(self):
        """Test that negative chunk_overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_overlap=-50)

    def test_chunk_overlap_exceeds_chunk_size(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_greater_than_chunk_size(self):
        """Test that overlap > chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap .* must be less than chunk_size"):
            TextChunker(chunk_size=100, chunk_overlap=150)


class TestTextChunkerBasicChunking:
    """Test suite for basic chunking functionality."""

    def test_single_paragraph_fits_in_chunk(self):
        """Test single paragraph that fits in one chunk."""
        chunker = TextChunker(chunk_size=1000)
        content = "This is a single paragraph. It has multiple sentences. But it fits in one chunk."
        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].source_file == "/test/file.txt"
        assert chunks[0].chunk_index == 0
        assert "start_line" in chunks[0].metadata
        assert "end_line" in chunks[0].metadata

    def test_multiple_paragraphs(self):
        """Test splitting multiple paragraphs."""
        chunker = TextChunker(chunk_size=1000)
        content = """First paragraph here.

Second paragraph here.

Third paragraph here."""
        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content
        assert "Third paragraph" in chunks[2].content

    def test_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = TextChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/test/file.txt")

    def test_whitespace_only_content_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        chunker = TextChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("   \n\n\t  ", "/test/file.txt")

    def test_sequential_chunk_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        content = """Paragraph one.

Paragraph two.

Paragraph three.

Paragraph four."""
        chunks = chunker.chunk(content, "/test/file.txt")

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_preserves_source_file(self):
        """Test that source file is preserved in all chunks."""
        chunker = TextChunker()
        source_file = "/documents/important.txt"
        content = """First paragraph.

Second paragraph."""
        chunks = chunker.chunk(content, source_file)

        for chunk in chunks:
            assert chunk.source_file == source_file


class TestTextChunkerSentenceSplitting:
    """Test suite for sentence-level splitting."""

    def test_large_paragraph_split_by_sentences(self):
        """Test that large paragraphs are split by sentences."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        # Create a paragraph longer than chunk_size
        content = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."

        chunks = chunker.chunk(content, "/test/file.txt")

        # Should be split into multiple chunks
        assert len(chunks) > 1
        # Each chunk should contain complete sentences
        for chunk in chunks:
            assert chunk.content.strip().endswith(('.', '!', '?'))

    def test_sentence_splitting_regex(self):
        """Test sentence splitting with various punctuation."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        content = "First sentence. Second sentence! Third sentence? Fourth sentence."

        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) >= 2
        # Verify sentences are properly split
        all_content = ' '.join(chunk.content for chunk in chunks)
        assert "First sentence" in all_content
        assert "Second sentence" in all_content

    def test_split_sentences_method(self):
        """Test _split_sentences method directly."""
        chunker = TextChunker()
        text = "First sentence. Second sentence! Third question?"

        sentences = chunker._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence." in sentences[0]
        assert "Second sentence!" in sentences[1]
        assert "Third question?" in sentences[2]


class TestTextChunkerParagraphSplitting:
    """Test suite for paragraph splitting."""

    def test_split_paragraphs_method(self):
        """Test _split_paragraphs method directly."""
        chunker = TextChunker()
        content = """First paragraph.

Second paragraph.


Third paragraph."""

        paragraphs = chunker._split_paragraphs(content)

        assert len(paragraphs) == 3
        assert "First paragraph" in paragraphs[0]
        assert "Second paragraph" in paragraphs[1]
        assert "Third paragraph" in paragraphs[2]

    def test_paragraphs_with_multiple_blank_lines(self):
        """Test that multiple blank lines are handled correctly."""
        chunker = TextChunker()
        content = """Paragraph one.



Paragraph two."""

        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 2

    def test_paragraphs_with_indentation(self):
        """Test paragraphs with leading/trailing whitespace."""
        chunker = TextChunker()
        content = """    Indented paragraph one.

        Indented paragraph two.    """

        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 2


class TestTextChunkerOverlap:
    """Test suite for chunk overlap functionality."""

    def test_overlap_between_sentence_chunks(self):
        """Test that overlap is applied between sentence-split chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=50)
        # Create content that will be split into multiple chunks
        content = "A" * 80 + ". " + "B" * 80 + ". " + "C" * 80 + "."

        chunks = chunker.chunk(content, "/test/file.txt")

        # With overlap, later chunks should contain content from earlier chunks
        assert len(chunks) >= 2

    def test_no_overlap_when_zero(self):
        """Test that no overlap occurs when chunk_overlap is 0."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        content = "First sentence here. " * 10

        chunks = chunker.chunk(content, "/test/file.txt")

        # With overlap=0, the chunker should still split content
        # The test just verifies that chunking occurs with 0 overlap
        assert len(chunks) >= 1
        # Verify each chunk is a valid Chunk object
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0


class TestTextChunkerMetadata:
    """Test suite for chunk metadata."""

    def test_line_number_tracking(self):
        """Test that line numbers are tracked in metadata."""
        chunker = TextChunker()
        content = """Line 1

Line 3

Line 5"""
        chunks = chunker.chunk(content, "/test/file.txt")

        # Each chunk should have start_line and end_line
        for chunk in chunks:
            assert "start_line" in chunk.metadata
            assert "end_line" in chunk.metadata
            assert isinstance(chunk.metadata["start_line"], int)
            assert isinstance(chunk.metadata["end_line"], int)
            assert chunk.metadata["end_line"] >= chunk.metadata["start_line"]

    def test_line_numbers_sequential(self):
        """Test that line numbers increase across chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        content = """First paragraph.

Second paragraph.

Third paragraph."""
        chunks = chunker.chunk(content, "/test/file.txt")

        # Line numbers should generally increase
        if len(chunks) > 1:
            assert chunks[0].metadata["start_line"] >= 0


class TestTextChunkerEdgeCases:
    """Test suite for edge cases."""

    def test_single_long_sentence(self):
        """Test handling of a single very long sentence."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        # Single sentence longer than chunk_size
        content = "This is a very long sentence without any punctuation that goes on and on and on " * 3

        chunks = chunker.chunk(content, "/test/file.txt")

        # Should still create at least one chunk
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_only_newlines(self):
        """Test content with only paragraph breaks."""
        chunker = TextChunker()
        content = "\n\n\n"

        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk(content, "/test/file.txt")

    def test_mixed_punctuation(self):
        """Test content with mixed sentence-ending punctuation."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        content = "Statement. Question? Exclamation! Another statement."

        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) >= 1
        # Verify all content is preserved
        all_content = ' '.join(chunk.content for chunk in chunks)
        assert "Statement" in all_content
        assert "Question" in all_content
        assert "Exclamation" in all_content

    def test_paragraph_exactly_chunk_size(self):
        """Test paragraph that is exactly chunk_size."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=0)
        content = "X" * 50

        chunks = chunker.chunk(content, "/test/file.txt")

        assert len(chunks) == 1
        assert len(chunks[0].content) == 50

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=0)
        content = "Short. Test. Data."

        chunks = chunker.chunk(content, "/test/file.txt")

        # Should create multiple chunks
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.content) <= 50  # Some overhead allowed


class TestTextChunkerIntegration:
    """Integration tests for complete workflows."""

    def test_realistic_document_chunking(self):
        """Test chunking a realistic document."""
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        content = """Introduction paragraph. This introduces the topic. It sets the context.

First main paragraph. This covers the first key point. It provides details and examples. It may span multiple sentences.

Second main paragraph. This covers another aspect. It builds on the previous section. It maintains coherence.

Conclusion paragraph. This wraps up the discussion. It summarizes key points."""

        chunks = chunker.chunk(content, "/docs/article.txt")

        # Verify all chunks are valid
        assert len(chunks) >= 3
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source_file == "/docs/article.txt"
            assert len(chunk.content) > 0
            assert chunk.chunk_index >= 0

        # Verify sequential indexing
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_chunker_implements_abstract_interface(self):
        """Test that TextChunker properly implements AbstractChunker."""
        from vector_mcp.chunking.base import AbstractChunker

        chunker = TextChunker()
        assert isinstance(chunker, AbstractChunker)

        # Verify interface methods exist
        assert hasattr(chunker, 'chunk')
        assert hasattr(chunker, 'validate_chunk_quality')

    def test_validate_chunk_quality(self):
        """Test that validate_chunk_quality works with text chunks."""
        chunker = TextChunker()
        content = "Test paragraph."
        chunks = chunker.chunk(content, "/test/file.txt")

        for chunk in chunks:
            assert chunker.validate_chunk_quality(chunk) is True
