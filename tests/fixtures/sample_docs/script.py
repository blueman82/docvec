"""Sample Python script for testing code chunking and indexing.

This module demonstrates various Python constructs including classes,
functions, and documentation that should be properly chunked.
"""

import logging
from typing import List, Optional


logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and transform data using various algorithms.

    This class demonstrates object-oriented programming patterns
    and serves as a test case for code chunking functionality.

    Attributes:
        name: Identifier for this processor instance
        buffer_size: Maximum number of items to buffer
    """

    def __init__(self, name: str, buffer_size: int = 100):
        """Initialize the data processor.

        Args:
            name: Name for this processor
            buffer_size: Size of internal buffer
        """
        self.name = name
        self.buffer_size = buffer_size
        self._buffer: List[str] = []
        logger.info(f"Initialized DataProcessor: {name}")

    def process_item(self, item: str) -> str:
        """Process a single data item.

        Applies transformation logic to the input item and
        adds it to the internal buffer if space is available.

        Args:
            item: Data item to process

        Returns:
            Processed item string
        """
        processed = item.strip().upper()

        if len(self._buffer) < self.buffer_size:
            self._buffer.append(processed)

        return processed

    def get_buffer_contents(self) -> List[str]:
        """Retrieve current buffer contents.

        Returns:
            List of buffered items
        """
        return self._buffer.copy()

    def clear_buffer(self) -> None:
        """Clear all items from the buffer."""
        self._buffer.clear()
        logger.debug(f"Buffer cleared for {self.name}")


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity score between two text strings.

    Uses a simple character-based similarity metric for demonstration.
    In production, you would use more sophisticated algorithms.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0

    # Simple character overlap similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def batch_process(items: List[str], processor: Optional[DataProcessor] = None) -> List[str]:
    """Process a batch of items.

    Args:
        items: List of items to process
        processor: Optional processor instance to use

    Returns:
        List of processed items
    """
    if processor is None:
        processor = DataProcessor("default")

    results = []
    for item in items:
        result = processor.process_item(item)
        results.append(result)

    return results
