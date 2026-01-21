"""Embedding provider protocol for pluggable backends.

This module defines the EmbeddingProvider protocol that allows different
embedding backends (Ollama, MLX) to be used interchangeably.
"""

from typing import List, Protocol, runtime_checkable

__all__ = ["EmbeddingProvider"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers.

    This protocol enables structural typing for embedding backends,
    allowing any class implementing these methods to be used as a provider
    without explicit inheritance.

    Required Methods:
        embed: Generate embedding for a single text
        embed_batch: Generate embeddings for multiple texts
        health_check: Check if the provider is ready
        close: Release resources

    Example:
        >>> class CustomProvider:
        ...     def embed(self, text: str, is_query: bool = False) -> List[float]:
        ...         ...
        ...     def embed_batch(
        ...         self, texts: List[str], batch_size: int = 32, is_query: bool = False
        ...     ) -> List[List[float]]:
        ...         ...
        ...     def health_check(self) -> bool:
        ...         ...
        ...     def close(self) -> None:
        ...         ...
        >>>
        >>> # Runtime check works due to @runtime_checkable
        >>> isinstance(CustomProvider(), EmbeddingProvider)
        True
    """

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed. Must not be empty.
            is_query: If True, apply query-specific preprocessing.
                     For mxbai-embed-large, this adds the query prefix.
                     Providers that don't support query prefixes may ignore this.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.
        """
        ...

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed. Must not be empty.
            batch_size: Number of texts per batch for processing.
            is_query: If True, apply query-specific preprocessing.

        Returns:
            List of embedding vectors, one per input text.
            Order is preserved: result[i] corresponds to texts[i].

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.
        """
        ...

    def health_check(self) -> bool:
        """Check if the provider is available and ready.

        Returns:
            True if provider is ready, False otherwise.
        """
        ...

    def close(self) -> None:
        """Release resources held by the provider.

        Should be called when the provider is no longer needed.
        Implementations should be idempotent (safe to call multiple times).
        """
        ...
