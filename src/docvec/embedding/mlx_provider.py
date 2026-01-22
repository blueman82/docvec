"""MLX-based embedding provider for Apple Silicon.

This module provides embedding generation using the mlx-embeddings library,
optimized for Apple Silicon Macs.
"""

import logging
from typing import Any, List

logger = logging.getLogger(__name__)

# Query prefix for mxbai-embed models (asymmetric retrieval)
EMBED_PREFIX = "Represent this sentence for searching relevant passages: "

__all__ = ["MLXProvider", "MLXNotAvailableError", "EmbeddingError"]


class MLXNotAvailableError(Exception):
    """Raised when mlx-embeddings is not installed or unavailable."""

    pass


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


def _check_mlx_available() -> None:
    """Check if mlx-embeddings is available.

    Raises:
        MLXNotAvailableError: If mlx-embeddings is not installed.
    """
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError as e:
        raise MLXNotAvailableError(
            "mlx-embeddings is not installed. " "Install with: uv add mlx-embeddings"
        ) from e


class MLXProvider:
    """MLX-based embedding provider for Apple Silicon.

    Implements the EmbeddingProvider protocol using mlx-embeddings for
    local embedding generation. Optimized for Apple Silicon Macs.

    The provider lazily loads the model on first use to avoid startup
    overhead when the provider is instantiated but not used.

    Args:
        model: HuggingFace model path (default: "mlx-community/mxbai-embed-large-v1")

    Example:
        >>> provider = MLXProvider()
        >>> # Query embedding (with prefix)
        >>> query_emb = provider.embed("What is Python?", is_query=True)
        >>> # Document embedding (no prefix)
        >>> doc_emb = provider.embed("Python is a programming language.")

    Raises:
        MLXNotAvailableError: If mlx-embeddings is not installed.
    """

    def __init__(
        self,
        model: str = "mlx-community/mxbai-embed-large-v1",
    ):
        """Initialize MLX provider.

        Args:
            model: HuggingFace model path for mlx-embeddings.
        """
        self.model = model
        self._model_instance: Any = None
        self._tokenizer: Any = None
        self._is_mxbai = "mxbai" in model.lower()

    def _ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded.

        Lazily loads the model on first use.

        Raises:
            MLXNotAvailableError: If mlx-embeddings is not installed.
            EmbeddingError: If model loading fails.
        """
        if self._model_instance is not None:
            return

        _check_mlx_available()

        try:
            from mlx_embeddings import load

            logger.info(f"Loading MLX embedding model: {self.model}")
            self._model_instance, self._tokenizer = load(self.model)
            logger.info(f"Successfully loaded MLX model: {self.model}")
        except ImportError:
            raise MLXNotAvailableError(
                "mlx-embeddings is not installed. "
                "Install with: uv add mlx-embeddings"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load MLX model {self.model}: {e}") from e

    def _generate_embedding_sync(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings synchronously using MLX.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        self._ensure_loaded()

        try:
            from mlx_embeddings import generate

            # mlx_embeddings.generate returns an output object with text_embeds attribute
            # Note: texts must be passed as keyword argument
            output = generate(self._model_instance, self._tokenizer, texts=texts)

            # Extract embeddings from the output object
            # output.text_embeds contains the normalized embeddings as MLX array
            if hasattr(output, "text_embeds"):
                embeddings = output.text_embeds
            elif hasattr(output, "embeddings"):
                embeddings = output.embeddings
            else:
                raise EmbeddingError(
                    f"Unexpected MLX output format: {type(output)}. "
                    f"Available attributes: {[a for a in dir(output) if not a.startswith('_')]}"
                )

            # Convert MLX array to Python list of floats
            if hasattr(embeddings, "tolist"):
                result = embeddings.tolist()
            else:
                result = [
                    emb.tolist() if hasattr(emb, "tolist") else list(emb)
                    for emb in embeddings
                ]

            # Clear MLX cache to prevent GPU memory accumulation across batches
            # Without this, intermediate tensors accumulate and can consume 18+ GB
            try:
                import mlx.core as mx

                mx.clear_cache()
            except Exception:
                pass  # Non-critical - just memory optimization

            return result
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"MLX embedding generation failed: {e}") from e

    def embed(self, text: str, is_query: bool = False) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed.
            is_query: If True, apply mxbai query prefix for search queries.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If text is empty.

        Example:
            >>> provider = MLXProvider()
            >>> # For search queries - adds prefix
            >>> query_emb = provider.embed("What is Python?", is_query=True)
            >>> # For documents - no prefix
            >>> doc_emb = provider.embed("Python is a language.")
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Apply query prefix for mxbai models when is_query=True
        if is_query and self._is_mxbai:
            prefixed_text = f"{EMBED_PREFIX}{text}"
        else:
            prefixed_text = text

        embeddings = self._generate_embedding_sync([prefixed_text])

        if not embeddings or len(embeddings) == 0:
            raise EmbeddingError("No embedding returned from MLX")

        return embeddings[0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        is_query: bool = False,
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed.
            batch_size: Number of texts per batch (default: 32).
            is_query: If True, apply mxbai query prefix for search queries.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
            ValueError: If texts list is empty.

        Example:
            >>> provider = MLXProvider()
            >>> texts = ["doc1", "doc2", "doc3"]
            >>> embeddings = provider.embed_batch(texts)
            >>> len(embeddings)
            3
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Apply query prefix if needed
            if is_query and self._is_mxbai:
                processed_batch = [f"{EMBED_PREFIX}{text}" for text in batch]
            else:
                processed_batch = batch

            batch_embeddings = self._generate_embedding_sync(processed_batch)

            if not batch_embeddings or len(batch_embeddings) != len(batch):
                raise EmbeddingError(
                    f"Expected {len(batch)} embeddings, "
                    f"got {len(batch_embeddings) if batch_embeddings else 0}"
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def health_check(self) -> bool:
        """Check if MLX provider is available and ready.

        Returns:
            True if mlx-embeddings is available, False otherwise.
        """
        try:
            _check_mlx_available()
            return True
        except MLXNotAvailableError:
            return False

    def close(self) -> None:
        """Release resources held by the provider.

        Clears the model from memory.
        """
        if self._model_instance is not None:
            # Clear references to allow garbage collection
            self._model_instance = None
            self._tokenizer = None
            logger.debug("MLX provider resources released")
