"""Embedding generation module for vector MCP.

This module provides pluggable embedding backends for document indexing
and semantic search. The default backend is MLX (optimized for Apple Silicon).

Available backends:
    - mlx: Local embedding using mlx-embeddings (Apple Silicon)
    - ollama: Remote embedding using Ollama server

Usage:
    >>> from docvec.embedding import create_embedding_provider
    >>> provider = create_embedding_provider("mlx")  # or "ollama"
    >>> embedding = provider.embed("Hello world")
"""

from .factory import EmbeddingBackend, create_embedding_provider
from .ollama_client import EmbeddingError, OllamaClient
from .provider import EmbeddingProvider

# MLXProvider is imported lazily in factory to handle missing mlx-embeddings

__all__ = [
    "EmbeddingProvider",
    "EmbeddingBackend",
    "create_embedding_provider",
    "OllamaClient",
    "EmbeddingError",
]
