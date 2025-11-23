"""Embedding generation module for vector MCP."""

from .ollama_client import OllamaClient, EmbeddingError

__all__ = ["OllamaClient", "EmbeddingError"]
