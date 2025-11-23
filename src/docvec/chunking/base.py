"""Base chunking interface and data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Chunk:
    """Immutable chunk of document content with metadata.

    Attributes:
        content: The actual text content of the chunk
        source_file: Path to the original source file
        chunk_index: Zero-based index of this chunk in the document
        metadata: Additional metadata about the chunk
        token_count: Optional token count (computed during embedding prep)
    """

    content: str
    source_file: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate chunk content after initialization."""
        self.validate_content()

    def validate_content(self) -> None:
        """Validate that chunk content meets quality constraints.

        Raises:
            ValueError: If content is empty or invalid
        """
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty or whitespace-only")

        if self.chunk_index < 0:
            raise ValueError(f"Chunk index must be non-negative, got {self.chunk_index}")


class AbstractChunker(ABC):
    """Abstract base class for document chunking strategies.

    Concrete implementations must provide a chunking strategy that splits
    documents into semantically meaningful pieces suitable for embedding.
    """

    @abstractmethod
    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split content into chunks.

        Args:
            content: The document content to chunk
            source_file: Path to the source file for provenance tracking

        Returns:
            List of Chunk objects with sequential indexing

        Raises:
            ValueError: If content is empty or invalid
        """
        pass

    def validate_chunk_quality(self, chunk: Chunk) -> bool:
        """Validate that a chunk meets quality standards.

        Args:
            chunk: The chunk to validate

        Returns:
            True if chunk passes quality checks, False otherwise
        """
        try:
            chunk.validate_content()
            return True
        except ValueError:
            return False
