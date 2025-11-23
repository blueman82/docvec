"""Header-aware markdown chunker respecting document structure."""

import re
from typing import Optional

from vector_mcp.chunking.base import AbstractChunker, Chunk


class MarkdownChunker(AbstractChunker):
    """Chunker that respects markdown header hierarchy.

    Parses markdown by headers (H1-H6), builds a hierarchy tree to track context,
    and groups content under each header into chunks. Includes parent headers in
    metadata for context preservation. Falls back to paragraph splitting if content
    is too long.

    Attributes:
        chunk_size: Maximum characters per chunk (soft limit)
        chunk_overlap: Number of characters to overlap between chunks
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize markdown chunker.

        Args:
            chunk_size: Maximum characters per chunk (soft limit)
            chunk_overlap: Number of characters to overlap between chunks

        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split markdown content into chunks respecting header hierarchy.

        Args:
            content: The markdown document content to chunk
            source_file: Path to the source file for provenance tracking

        Returns:
            List of Chunk objects with sequential indexing and header context

        Raises:
            ValueError: If content is empty or invalid
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        # Parse headers and build sections
        sections = self._parse_sections(content)

        # Build chunks from sections
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section, source_file, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks

    def _parse_sections(self, content: str) -> list[dict]:
        """Parse markdown into sections based on headers.

        Args:
            content: Markdown content to parse

        Returns:
            List of section dictionaries with 'level', 'title', 'content', and 'header_path'
        """
        # Pattern to match markdown headers (# Header)
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        sections = []
        header_stack = []  # Stack to track header hierarchy

        lines = content.split('\n')
        current_section = None
        current_content_lines = []

        for line in lines:
            header_match = header_pattern.match(line)

            if header_match:
                # Save previous section if exists
                if current_section is not None:
                    current_section['content'] = '\n'.join(current_content_lines).strip()
                    if current_section['content']:  # Only add non-empty sections
                        sections.append(current_section)

                # Parse new header
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Update header stack to maintain hierarchy
                # Remove headers at same or deeper level
                while header_stack and header_stack[-1]['level'] >= level:
                    header_stack.pop()

                # Add current header to stack
                header_stack.append({'level': level, 'title': title})

                # Build header path from stack
                header_path = ' > '.join(h['title'] for h in header_stack)

                # Create new section
                current_section = {
                    'level': level,
                    'title': title,
                    'header_path': header_path,
                    'content': ''
                }
                current_content_lines = []
            else:
                # Accumulate content for current section
                current_content_lines.append(line)

        # Don't forget the last section
        if current_section is not None:
            current_section['content'] = '\n'.join(current_content_lines).strip()
            if current_section['content']:
                sections.append(current_section)

        # If no headers found, treat entire content as one section
        if not sections:
            full_content = content.strip()
            if full_content:
                sections.append({
                    'level': 0,
                    'title': 'Document',
                    'header_path': 'Document',
                    'content': full_content
                })

        return sections

    def _chunk_section(
        self, section: dict, source_file: str, start_index: int
    ) -> list[Chunk]:
        """Create chunks from a single section.

        If section content fits in chunk_size, creates single chunk.
        Otherwise, splits by paragraphs with overlap.

        Args:
            section: Section dictionary with content and metadata
            source_file: Source file path
            start_index: Starting chunk index

        Returns:
            List of chunks for this section
        """
        content = section['content']

        # If content fits in one chunk, return it
        if len(content) <= self.chunk_size:
            return [
                Chunk(
                    content=content,
                    source_file=source_file,
                    chunk_index=start_index,
                    metadata={
                        'header_path': section['header_path'],
                        'header_level': section['level'],
                        'header_title': section['title']
                    }
                )
            ]

        # Content is too long, split by paragraphs
        return self._chunk_long_section(section, source_file, start_index)

    def _chunk_long_section(
        self, section: dict, source_file: str, start_index: int
    ) -> list[Chunk]:
        """Split long section content into multiple chunks with overlap.

        Splits by paragraphs (blank lines) and ensures overlap between chunks
        for coherence.

        Args:
            section: Section dictionary with content and metadata
            source_file: Source file path
            start_index: Starting chunk index

        Returns:
            List of chunks for this section
        """
        content = section['content']

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = start_index

        for para_idx, para in enumerate(paragraphs):
            para_length = len(para)

            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if current_chunk and current_length + para_length > self.chunk_size:
                # Create chunk from accumulated paragraphs
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(
                    Chunk(
                        content=chunk_content,
                        source_file=source_file,
                        chunk_index=chunk_idx,
                        metadata={
                            'header_path': section['header_path'],
                            'header_level': section['level'],
                            'header_title': section['title']
                        }
                    )
                )
                chunk_idx += 1

                # Start new chunk with overlap
                # Include last sentence of previous chunk for context
                overlap_content = self._get_overlap(chunk_content)
                if overlap_content:
                    current_chunk = [overlap_content, para]
                    current_length = len(overlap_content) + len(para) + 2  # +2 for \n\n
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                # Add paragraph to current chunk
                current_chunk.append(para)
                current_length += para_length + (2 if current_chunk else 0)  # +2 for \n\n separator

        # Add final chunk if any content remains
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(
                Chunk(
                    content=chunk_content,
                    source_file=source_file,
                    chunk_index=chunk_idx,
                    metadata={
                        'header_path': section['header_path'],
                        'header_level': section['level'],
                        'header_title': section['title']
                    }
                )
            )

        return chunks

    def _get_overlap(self, text: str) -> Optional[str]:
        """Extract overlap content from end of text for next chunk.

        Gets the last sentence (or up to chunk_overlap characters) to maintain
        context between chunks.

        Args:
            text: Text to extract overlap from

        Returns:
            Overlap text, or None if text is too short
        """
        if len(text) <= self.chunk_overlap:
            return None

        # Try to get last sentence
        sentences = re.split(r'[.!?]\s+', text)
        if len(sentences) > 1:
            last_sentence = sentences[-1]
            # Make sure it's not too long
            if len(last_sentence) <= self.chunk_overlap:
                return last_sentence

        # Fall back to last chunk_overlap characters
        overlap = text[-self.chunk_overlap:]
        # Try to start at a word boundary
        space_idx = overlap.find(' ')
        if space_idx > 0:
            return overlap[space_idx + 1:]

        return overlap
