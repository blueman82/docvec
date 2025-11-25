"""Tests for markdown chunker."""

import pytest

from docvec.chunking.markdown_chunker import MarkdownChunker


class TestMarkdownChunker:
    """Test suite for MarkdownChunker."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=200)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        chunker = MarkdownChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_init_invalid_chunk_size(self):
        """Test that negative or zero chunk_size raises error."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MarkdownChunker(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            MarkdownChunker(chunk_size=-100)

    def test_init_invalid_overlap(self):
        """Test that negative overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            MarkdownChunker(chunk_overlap=-50)

    def test_init_overlap_exceeds_chunk_size(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(
            ValueError, match="chunk_overlap.*must be less than chunk_size"
        ):
            MarkdownChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(
            ValueError, match="chunk_overlap.*must be less than chunk_size"
        ):
            MarkdownChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_empty_content_raises_error(self):
        """Test that empty content raises ValueError."""
        chunker = MarkdownChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("", "/path/to/file.md")

    def test_chunk_whitespace_only_raises_error(self):
        """Test that whitespace-only content raises ValueError."""
        chunker = MarkdownChunker()
        with pytest.raises(ValueError, match="Content cannot be empty"):
            chunker.chunk("   \n\t  ", "/path/to/file.md")

    def test_chunk_simple_markdown_single_header(self):
        """Test chunking markdown with single header."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Introduction

This is the introduction section.
It has multiple lines of content."""

        chunks = chunker.chunk(content, "/docs/readme.md")

        assert len(chunks) == 1
        assert (
            chunks[0].content
            == "This is the introduction section.\nIt has multiple lines of content."
        )
        assert chunks[0].source_file == "/docs/readme.md"
        assert chunks[0].chunk_index == 0
        assert chunks[0].metadata["header_path"] == "Introduction"
        assert chunks[0].metadata["header_level"] == 1
        assert chunks[0].metadata["header_title"] == "Introduction"

    def test_chunk_multiple_h1_headers(self):
        """Test chunking markdown with multiple top-level headers."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Chapter 1

Content of chapter 1.

# Chapter 2

Content of chapter 2."""

        chunks = chunker.chunk(content, "/docs/book.md")

        assert len(chunks) == 2
        assert chunks[0].metadata["header_path"] == "Chapter 1"
        assert chunks[0].content == "Content of chapter 1."
        assert chunks[1].metadata["header_path"] == "Chapter 2"
        assert chunks[1].content == "Content of chapter 2."

    def test_chunk_nested_headers(self):
        """Test chunking markdown with nested header hierarchy."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Chapter 1

Introduction to chapter 1.

## Section 1.1

Content of section 1.1.

## Section 1.2

Content of section 1.2.

### Subsection 1.2.1

Content of subsection 1.2.1."""

        chunks = chunker.chunk(content, "/docs/manual.md")

        assert len(chunks) == 4
        assert chunks[0].metadata["header_path"] == "Chapter 1"
        assert chunks[0].metadata["header_level"] == 1

        assert chunks[1].metadata["header_path"] == "Chapter 1 > Section 1.1"
        assert chunks[1].metadata["header_level"] == 2

        assert chunks[2].metadata["header_path"] == "Chapter 1 > Section 1.2"
        assert chunks[2].metadata["header_level"] == 2

        assert (
            chunks[3].metadata["header_path"]
            == "Chapter 1 > Section 1.2 > Subsection 1.2.1"
        )
        assert chunks[3].metadata["header_level"] == 3

    def test_chunk_header_hierarchy_reset(self):
        """Test that header hierarchy resets properly when moving up levels."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Chapter 1

## Section 1.1

### Subsection 1.1.1

Content here.

## Section 1.2

Back to level 2."""

        chunks = chunker.chunk(content, "/docs/test.md")

        # Only sections with content are included
        assert len(chunks) == 2
        assert (
            chunks[0].metadata["header_path"]
            == "Chapter 1 > Section 1.1 > Subsection 1.1.1"
        )
        assert chunks[1].metadata["header_path"] == "Chapter 1 > Section 1.2"

    def test_chunk_no_headers(self):
        """Test chunking plain text without headers."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """This is plain text without any headers.

It should still be chunked properly."""

        chunks = chunker.chunk(content, "/docs/plain.md")

        assert len(chunks) == 1
        assert chunks[0].metadata["header_path"] == "Document"
        assert chunks[0].metadata["header_level"] == 0
        assert "This is plain text" in chunks[0].content

    def test_chunk_long_section_splits_by_paragraphs(self):
        """Test that long sections are split by paragraphs."""
        chunker = MarkdownChunker(chunk_size=100, chunk_overlap=20)
        content = """# Long Section

This is the first paragraph with some content that will make it reasonably long.

This is the second paragraph with more content to ensure we exceed the chunk size.

This is the third paragraph with even more content."""

        chunks = chunker.chunk(content, "/docs/long.md")

        # Should create multiple chunks due to length
        assert len(chunks) > 1

        # All chunks should have same header metadata
        for chunk in chunks:
            assert chunk.metadata["header_path"] == "Long Section"
            assert chunk.metadata["header_level"] == 1

    def test_chunk_overlap_between_chunks(self):
        """Test that overlap is applied between chunks."""
        chunker = MarkdownChunker(chunk_size=150, chunk_overlap=50)
        content = """# Section

First paragraph with enough content to trigger chunking behavior properly.

Second paragraph with enough content to ensure we have multiple chunks created.

Third paragraph to make sure we test overlap."""

        chunks = chunker.chunk(content, "/docs/overlap.md")

        # Should have multiple chunks
        assert len(chunks) > 1

        # Check that chunks have sequential indices
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_chunk_sequential_indexing(self):
        """Test that chunks have sequential zero-based indexing."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Header 1

Content 1.

# Header 2

Content 2.

# Header 3

Content 3."""

        chunks = chunker.chunk(content, "/docs/multi.md")

        assert len(chunks) == 3
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx

    def test_chunk_preserves_source_file(self):
        """Test that source file is preserved in all chunks."""
        chunker = MarkdownChunker(chunk_size=1000)
        source_file = "/documents/important.md"
        content = """# Header 1

Content 1.

# Header 2

Content 2."""

        chunks = chunker.chunk(content, source_file)

        for chunk in chunks:
            assert chunk.source_file == source_file

    def test_chunk_all_header_levels(self):
        """Test chunking with all six header levels."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# H1

## H2

### H3

#### H4

##### H5

###### H6

Content."""

        chunks = chunker.chunk(content, "/docs/headers.md")

        # Find the deepest chunk
        deepest_chunk = chunks[-1]
        assert deepest_chunk.metadata["header_path"] == "H1 > H2 > H3 > H4 > H5 > H6"
        assert deepest_chunk.metadata["header_level"] == 6

    def test_chunk_empty_sections_skipped(self):
        """Test that sections with no content are skipped."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Header 1

# Header 2

Content here.

# Header 3"""

        chunks = chunker.chunk(content, "/docs/empty.md")

        # Should only have chunk for Header 2 which has content
        assert len(chunks) == 1
        assert chunks[0].metadata["header_title"] == "Header 2"

    def test_chunk_whitespace_handling(self):
        """Test proper handling of whitespace in content."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Header

Content with    multiple    spaces.

And multiple


blank lines."""

        chunks = chunker.chunk(content, "/docs/whitespace.md")

        assert len(chunks) == 1
        # Content should be stripped but internal whitespace preserved
        assert "multiple    spaces" in chunks[0].content

    def test_chunk_special_characters_in_headers(self):
        """Test headers with special characters."""
        chunker = MarkdownChunker(chunk_size=1000)
        content = """# Introduction: Getting Started

Content here.

## Step 1: Install Dependencies

More content.

### Note: Important!

Final content."""

        chunks = chunker.chunk(content, "/docs/special.md")

        assert len(chunks) == 3
        assert chunks[0].metadata["header_title"] == "Introduction: Getting Started"
        assert chunks[1].metadata["header_title"] == "Step 1: Install Dependencies"
        assert chunks[2].metadata["header_title"] == "Note: Important!"

    def test_parse_sections_basic(self):
        """Test _parse_sections method directly."""
        chunker = MarkdownChunker()
        content = """# Header

Content."""

        sections = chunker._parse_sections(content)

        assert len(sections) == 1
        assert sections[0]["level"] == 1
        assert sections[0]["title"] == "Header"
        assert sections[0]["content"] == "Content."

    def test_get_overlap_short_text(self):
        """Test _get_overlap with text shorter than overlap size."""
        chunker = MarkdownChunker(chunk_overlap=100)
        text = "Short text."

        overlap = chunker._get_overlap(text)
        assert overlap is None

    def test_get_overlap_long_text(self):
        """Test _get_overlap with long text."""
        chunker = MarkdownChunker(chunk_overlap=50)
        text = "This is a long sentence. And this is another sentence that should be extracted."

        overlap = chunker._get_overlap(text)
        # Should get last sentence or portion of text
        assert overlap is not None
        assert len(overlap) <= 50

    def test_chunk_validates_content(self):
        """Test that created chunks pass validation."""
        chunker = MarkdownChunker()
        content = """# Header

Valid content."""

        chunks = chunker.chunk(content, "/docs/valid.md")

        for chunk in chunks:
            # Should not raise
            chunk.validate_content()
            assert chunker.validate_chunk_quality(chunk) is True

    def test_chunk_real_world_example(self):
        """Test with a realistic markdown document."""
        chunker = MarkdownChunker(chunk_size=500, chunk_overlap=100)
        content = """# User Guide

Welcome to our application user guide.

## Installation

### Prerequisites

Before installing, ensure you have:
- Python 3.8 or higher
- pip package manager

### Installing the Package

Run the following command:

```bash
pip install our-package
```

## Getting Started

### Basic Usage

Here's a simple example:

```python
from our_package import main
main()
```

### Advanced Features

Our package supports many advanced features for power users.

## Troubleshooting

If you encounter issues, check our FAQ."""

        chunks = chunker.chunk(content, "/docs/guide.md")

        # Should create multiple chunks
        assert len(chunks) >= 3

        # Verify hierarchy is preserved
        header_paths = [c.metadata["header_path"] for c in chunks]
        assert "User Guide" in header_paths[0]
        assert "Installation" in str(header_paths)
        assert "Getting Started" in str(header_paths)

        # All chunks should be valid
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.source_file == "/docs/guide.md"
