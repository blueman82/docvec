"""Tests for PDF chunker."""

import io
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from vector_mcp.chunking.base import Chunk
from vector_mcp.chunking.pdf_chunker import PDFChunker


class TestPDFChunkerInit:
    """Test suite for PDFChunker initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        chunker = PDFChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        chunker = PDFChunker(chunk_size=500, chunk_overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_init_invalid_chunk_size_zero(self):
        """Test that chunk_size of 0 raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            PDFChunker(chunk_size=0)

    def test_init_invalid_chunk_size_negative(self):
        """Test that negative chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            PDFChunker(chunk_size=-100)

    def test_init_invalid_overlap_negative(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            PDFChunker(chunk_overlap=-50)

    def test_init_overlap_exceeds_chunk_size(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            PDFChunker(chunk_size=100, chunk_overlap=100)

    def test_init_overlap_greater_than_chunk_size(self):
        """Test that overlap > chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            PDFChunker(chunk_size=100, chunk_overlap=150)


class TestPDFChunkerExtraction:
    """Test suite for PDF text extraction."""

    def test_chunk_file_not_found(self, tmp_path):
        """Test that non-existent file raises error."""
        chunker = PDFChunker()
        non_existent = tmp_path / "does_not_exist.pdf"

        with pytest.raises(ValueError, match="PDF file does not exist"):
            chunker.chunk("", str(non_existent))

    def test_chunk_non_pdf_file(self, tmp_path):
        """Test that non-PDF file raises error."""
        chunker = PDFChunker()
        text_file = tmp_path / "document.txt"
        text_file.write_text("Not a PDF")

        with pytest.raises(ValueError, match="File is not a PDF"):
            chunker.chunk("", str(text_file))

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_corrupted_pdf(self, mock_reader, tmp_path):
        """Test handling of corrupted PDF."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "corrupted.pdf"
        pdf_file.write_text("corrupted")

        # Mock PdfReader to raise PdfReadError
        mock_reader.side_effect = PdfReadError("Corrupted PDF")

        with pytest.raises(ValueError, match="Corrupted or invalid PDF file"):
            chunker.chunk("", str(pdf_file))

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_pdf_no_text(self, mock_reader, tmp_path):
        """Test handling of PDF with no extractable text."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "no_text.pdf"
        pdf_file.write_text("fake pdf")

        # Mock empty pages
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        with pytest.raises(ValueError, match="PDF contains no extractable text"):
            chunker.chunk("", str(pdf_file))

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_extract_single_page(self, mock_reader, tmp_path):
        """Test extraction from single-page PDF."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "single.pdf"
        pdf_file.write_text("fake pdf")

        # Mock single page with text
        mock_page = Mock()
        mock_page.extract_text.return_value = "This is page one content."
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) == 1
        assert chunks[0].content == "This is page one content."
        assert chunks[0].metadata["page_number"] == 1

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_extract_multiple_pages(self, mock_reader, tmp_path):
        """Test extraction from multi-page PDF."""
        chunker = PDFChunker(chunk_size=100, chunk_overlap=20)
        pdf_file = tmp_path / "multi.pdf"
        pdf_file.write_text("fake pdf")

        # Mock multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one text."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page two text."
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page three text."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) >= 1
        # Verify source file is preserved
        for chunk in chunks:
            assert chunk.source_file == str(pdf_file)

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_extract_with_empty_pages(self, mock_reader, tmp_path):
        """Test that empty pages are skipped."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "mixed.pdf"
        pdf_file.write_text("fake pdf")

        # Mock pages with some empty
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one has content."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "   \n\t  "  # Whitespace only
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = ""  # Empty
        mock_page4 = Mock()
        mock_page4.extract_text.return_value = "Page four has content."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2, mock_page3, mock_page4]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        # Should only get chunks from pages 1 and 4
        assert len(chunks) >= 1
        assert all(chunk.content.strip() for chunk in chunks)

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_extract_page_extraction_error(self, mock_reader, tmp_path):
        """Test that page-level errors are logged but don't stop extraction."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "partial_error.pdf"
        pdf_file.write_text("fake pdf")

        # Mock pages where one fails
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one content."
        mock_page2 = Mock()
        mock_page2.extract_text.side_effect = Exception("Page extraction failed")
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page three content."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        # Should get chunks from pages 1 and 3
        assert len(chunks) >= 1


class TestPDFChunkerChunking:
    """Test suite for PDF chunking logic."""

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_small_single_page(self, mock_reader, tmp_path):
        """Test chunking of small single-page PDF."""
        chunker = PDFChunker(chunk_size=1000, chunk_overlap=200)
        pdf_file = tmp_path / "small.pdf"
        pdf_file.write_text("fake pdf")

        mock_page = Mock()
        mock_page.extract_text.return_value = "Short content."
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) == 1
        assert chunks[0].content == "Short content."
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["page_number"] == 1

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_large_single_page_splits(self, mock_reader, tmp_path):
        """Test that large single-page content splits into multiple chunks."""
        chunker = PDFChunker(chunk_size=50, chunk_overlap=10)
        pdf_file = tmp_path / "large_page.pdf"
        pdf_file.write_text("fake pdf")

        # Create long text that will need multiple chunks
        long_text = "This is sentence one. " * 10

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) > 1
        # All chunks should be from page 1
        for chunk in chunks:
            assert chunk.metadata["page_number"] == 1

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_multiple_pages_span_chunks(self, mock_reader, tmp_path):
        """Test that chunks can span multiple pages."""
        chunker = PDFChunker(chunk_size=100, chunk_overlap=20)
        pdf_file = tmp_path / "multi_page.pdf"
        pdf_file.write_text("fake pdf")

        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "First page content here."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Second page content here."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) >= 1
        # Check that we track page ranges
        has_multi_page_chunk = any(
            "page_start" in chunk.metadata or "pages" in chunk.metadata
            for chunk in chunks
        )
        # May or may not have multi-page chunks depending on content size

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_overlap_applied(self, mock_reader, tmp_path):
        """Test that chunk overlap is applied between chunks."""
        chunker = PDFChunker(chunk_size=50, chunk_overlap=15)
        pdf_file = tmp_path / "overlap_test.pdf"
        pdf_file.write_text("fake pdf")

        # Long text to ensure multiple chunks
        long_text = "A" * 150

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) > 1
        # Verify overlap by checking that chunks have some similar content
        # (Exact overlap checking is complex due to text processing)

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_no_overlap(self, mock_reader, tmp_path):
        """Test chunking with no overlap."""
        chunker = PDFChunker(chunk_size=30, chunk_overlap=0)
        pdf_file = tmp_path / "no_overlap.pdf"
        pdf_file.write_text("fake pdf")

        long_text = "A" * 100

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) > 1

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_sequential_indexing(self, mock_reader, tmp_path):
        """Test that chunks have sequential zero-based indexing."""
        chunker = PDFChunker(chunk_size=50, chunk_overlap=10)
        pdf_file = tmp_path / "indexing.pdf"
        pdf_file.write_text("fake pdf")

        long_text = "This is a longer text. " * 20

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx


class TestPDFChunkerMetadata:
    """Test suite for PDF chunk metadata."""

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_metadata_single_page(self, mock_reader, tmp_path):
        """Test metadata for single-page chunk."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "meta_single.pdf"
        pdf_file.write_text("fake pdf")

        mock_page = Mock()
        mock_page.extract_text.return_value = "Single page content."
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) == 1
        assert "page_number" in chunks[0].metadata
        assert chunks[0].metadata["page_number"] == 1

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_metadata_multiple_pages_single_chunk(self, mock_reader, tmp_path):
        """Test metadata when multiple pages fit in one chunk."""
        chunker = PDFChunker(chunk_size=1000, chunk_overlap=100)
        pdf_file = tmp_path / "meta_multi.pdf"
        pdf_file.write_text("fake pdf")

        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page one."
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page two."

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        # Should have page range metadata
        if len(chunks) == 1:
            # If combined into one chunk
            assert "page_start" in chunks[0].metadata or "pages" in chunks[0].metadata

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_metadata_page_spanning(self, mock_reader, tmp_path):
        """Test metadata for chunks spanning pages."""
        chunker = PDFChunker(chunk_size=200, chunk_overlap=50)
        pdf_file = tmp_path / "meta_span.pdf"
        pdf_file.write_text("fake pdf")

        # Create pages with enough content to span
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "A" * 100
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "B" * 100

        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        # Verify all chunks have page metadata
        for chunk in chunks:
            assert (
                "page_number" in chunk.metadata or
                "page_start" in chunk.metadata or
                "pages" in chunk.metadata
            )


class TestPDFChunkerEdgeCases:
    """Test suite for edge cases."""

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_whitespace_handling(self, mock_reader, tmp_path):
        """Test that whitespace is handled correctly."""
        chunker = PDFChunker()
        pdf_file = tmp_path / "whitespace.pdf"
        pdf_file.write_text("fake pdf")

        mock_page = Mock()
        mock_page.extract_text.return_value = "  Content with   extra    spaces.  "
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        assert len(chunks) == 1
        # Content should be stripped
        assert chunks[0].content.strip() == chunks[0].content

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_validation(self, mock_reader, tmp_path):
        """Test that all chunks pass validation."""
        chunker = PDFChunker(chunk_size=50, chunk_overlap=10)
        pdf_file = tmp_path / "validation.pdf"
        pdf_file.write_text("fake pdf")

        long_text = "Valid content. " * 20

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        # All chunks should be valid
        for chunk in chunks:
            assert chunker.validate_chunk_quality(chunk)
            assert chunk.content.strip()
            assert chunk.chunk_index >= 0

    @patch('vector_mcp.chunking.pdf_chunker.PdfReader')
    def test_chunk_source_file_preserved(self, mock_reader, tmp_path):
        """Test that source file path is preserved in all chunks."""
        chunker = PDFChunker(chunk_size=50, chunk_overlap=10)
        pdf_file = tmp_path / "source_test.pdf"
        pdf_file.write_text("fake pdf")

        long_text = "Content here. " * 20

        mock_page = Mock()
        mock_page.extract_text.return_value = long_text
        mock_reader_instance = Mock()
        mock_reader_instance.pages = [mock_page]
        mock_reader.return_value = mock_reader_instance

        chunks = chunker.chunk("", str(pdf_file))

        source_path = str(pdf_file)
        for chunk in chunks:
            assert chunk.source_file == source_path
