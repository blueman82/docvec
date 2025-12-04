"""Ollama embedding client with retry logic and health checks.

This module provides an HTTP client for the Ollama embeddings API with:
- Exponential backoff retry logic for network resilience
- Health checks to validate model availability
- Batch embedding support to reduce API calls
- Configurable timeout handling
"""

import logging
import time
from functools import wraps
from typing import Callable

import requests

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""

    pass


def retry_with_backoff(
    max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 10.0
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 10.0)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry_with_backoff(max_retries=3)
        >>> def fetch_data():
        ...     return requests.get("http://example.com")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, EmbeddingError):
                    if attempt == max_retries - 1:
                        raise

                    delay = min(base_delay * (2**attempt), max_delay)
                    time.sleep(delay)

            return func(*args, **kwargs)

        return wrapper

    return decorator


class OllamaClient:
    """HTTP client for Ollama embeddings API.

    Provides embedding generation with retry logic, health checks,
    and batch processing capabilities.

    Args:
        host: Ollama server host URL (default: "http://localhost:11434")
        model: Embedding model name (default: "nomic-embed-text")
        timeout: Request timeout in seconds (default: 30)

    Attributes:
        host: Ollama server URL
        model: Name of the embedding model
        timeout: Request timeout value
        _session: Persistent requests session

    Example:
        >>> client = OllamaClient(host="http://localhost:11434")
        >>> embedding = client.embed("Hello world")
        >>> embeddings = client.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        timeout: int = 30,
    ):
        """Initialize Ollama client.

        Args:
            host: Ollama server host URL
            model: Embedding model name
            timeout: Request timeout in seconds

        Raises:
            EmbeddingError: If initialization fails
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Check if Ollama server is available and model is loaded.

        Validates that:
        1. Ollama server is responding
        2. The specified model exists and is available

        Returns:
            True if health check passes, False otherwise

        Example:
            >>> client = OllamaClient()
            >>> if client.health_check():
            ...     print("Ollama is ready")
        """
        try:
            # Check if server is responding
            response = self._session.get(f"{self.host}/api/tags", timeout=self.timeout)
            response.raise_for_status()

            # Verify model exists in available models
            models_data = response.json()
            available_models = [
                m.get("name", "") for m in models_data.get("models", [])
            ]

            # Check if our model is in the list
            model_available = any(
                self.model in model_name for model_name in available_models
            )

            if not model_available:
                # Try to pull/load the model by making a test embedding request
                try:
                    test_response = self._request_with_retry(
                        {"model": self.model, "prompt": "test"}
                    )
                    return bool(test_response.status_code == 200)
                except Exception:
                    return False

            return True

        except Exception:
            return False

    def is_model_available(self) -> bool:
        """Check if the configured model is available locally.

        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models_data = response.json()
            available_models = [
                m.get("name", "") for m in models_data.get("models", [])
            ]
            return any(self.model in model_name for model_name in available_models)
        except Exception:
            return False

    def pull_model(self, stream: bool = True) -> bool:
        """Pull/download the configured model from Ollama registry.

        This is a blocking operation that downloads the model if not present.
        For large models (like mxbai-embed-large ~670MB), this may take
        several minutes depending on network speed.

        Args:
            stream: If True, streams progress to logger (default: True)

        Returns:
            True if model was pulled successfully, False otherwise

        Example:
            >>> client = OllamaClient(model="mxbai-embed-large")
            >>> if not client.is_model_available():
            ...     client.pull_model()
        """
        logger.info(f"Pulling model '{self.model}' from Ollama registry...")

        try:
            response = self._session.post(
                f"{self.host}/api/pull",
                json={"name": self.model, "stream": stream},
                timeout=None,  # No timeout for model pulls (can take minutes)
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                # Process streaming response to show progress
                last_status = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            import json

                            data = json.loads(line)
                            status = data.get("status", "")

                            # Log progress updates (deduplicated)
                            if status != last_status:
                                if "pulling" in status:
                                    total = data.get("total", 0)
                                    completed = data.get("completed", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        logger.info(
                                            f"Pulling {self.model}: {pct:.1f}% "
                                            f"({completed}/{total} bytes)"
                                        )
                                    else:
                                        logger.info(f"Pulling {self.model}: {status}")
                                elif status == "success":
                                    logger.info(
                                        f"Successfully pulled model '{self.model}'"
                                    )
                                else:
                                    logger.info(f"Pull status: {status}")
                                last_status = status
                        except (ValueError, KeyError):
                            continue
            else:
                logger.info(f"Successfully pulled model '{self.model}'")

            return True

        except requests.RequestException as e:
            logger.error(f"Failed to pull model '{self.model}': {e}")
            return False

    def ensure_model(self) -> bool:
        """Ensure the model is available, pulling it if necessary.

        This is the recommended method to call during initialization to
        guarantee the model is ready for use.

        Returns:
            True if model is available (was already present or pulled successfully),
            False if model could not be made available

        Example:
            >>> client = OllamaClient(model="mxbai-embed-large")
            >>> if client.ensure_model():
            ...     # Safe to use embeddings
            ...     embedding = client.embed("Hello world")
        """
        if self.is_model_available():
            logger.info(f"Model '{self.model}' is already available")
            return True

        logger.info(f"Model '{self.model}' not found locally, attempting to pull...")
        return self.pull_model()

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0)
    def _request_with_retry(self, payload: dict) -> requests.Response:
        """Make HTTP request to Ollama API with retry logic.

        Args:
            payload: Request payload dictionary

        Returns:
            HTTP response object

        Raises:
            EmbeddingError: If request fails after all retries
        """
        try:
            response = self._session.post(
                f"{self.host}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response

        except requests.Timeout as e:
            raise EmbeddingError(
                f"Request timeout after {self.timeout}s. "
                f"Consider increasing timeout for model {self.model}"
            ) from e

        except requests.RequestException as e:
            raise EmbeddingError(f"Ollama API request failed: {e}") from e

    def _get_query_prefix(self) -> str:
        """Get the query prefix for the current model.

        Different embedding models require different prefixes for queries
        to achieve optimal retrieval performance.

        Returns:
            Query prefix string, or empty string if no prefix needed
        """
        model_lower = self.model.lower()

        # mxbai-embed models require this specific prefix for queries
        if "mxbai" in model_lower:
            return "Represent this sentence for searching relevant passages: "

        # nomic-embed-text uses search_query prefix
        if "nomic" in model_lower:
            return "search_query: "

        # Default: no prefix
        return ""

    def _get_document_prefix(self) -> str:
        """Get the document prefix for the current model.

        Some embedding models require prefixes for documents during indexing.

        Returns:
            Document prefix string, or empty string if no prefix needed
        """
        model_lower = self.model.lower()

        # nomic-embed-text uses search_document prefix
        if "nomic" in model_lower:
            return "search_document: "

        # mxbai and most others: no prefix for documents
        return ""

    def embed(self, text: str, is_query: bool = False) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text to embed
            is_query: If True, apply query-specific prefix for the model

        Returns:
            Embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Example:
            >>> client = OllamaClient()
            >>> embedding = client.embed("Hello world")
            >>> len(embedding)
            768
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Apply model-specific prefix
            if is_query:
                prefix = self._get_query_prefix()
            else:
                prefix = self._get_document_prefix()

            prefixed_text = f"{prefix}{text}" if prefix else text

            payload = {"model": self.model, "prompt": prefixed_text}
            response = self._request_with_retry(payload)

            data = response.json()
            embedding = data.get("embedding")

            if not embedding:
                raise EmbeddingError("No embedding returned from Ollama API")

            return list(embedding) if embedding else []

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query (with model-specific prefix).

        This is a convenience method that calls embed() with is_query=True.
        Use this for search queries to get optimal retrieval performance.

        Args:
            text: Query text to embed

        Returns:
            Query embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Example:
            >>> client = OllamaClient(model="mxbai-embed-large")
            >>> # Automatically prefixes with "Represent this sentence..."
            >>> embedding = client.embed_query("What is machine learning?")
        """
        return self.embed(text, is_query=True)

    def embed_document(self, text: str) -> list[float]:
        """Generate embedding for a document (with model-specific prefix).

        This is a convenience method that calls embed() with is_query=False.
        Use this for indexing documents.

        Args:
            text: Document text to embed

        Returns:
            Document embedding vector as list of floats

        Raises:
            EmbeddingError: If embedding generation fails
            ValueError: If text is empty

        Example:
            >>> client = OllamaClient(model="nomic-embed-text")
            >>> # Automatically prefixes with "search_document: "
            >>> embedding = client.embed_document("Python is a programming language.")
        """
        return self.embed(text, is_query=False)

    def embed_batch(
        self, texts: list[str], batch_size: int = 32, is_query: bool = False
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches.

        Processes texts in batches to reduce API calls and improve performance.
        For example, 100 texts with batch_size=32 results in 4 API calls
        instead of 100.

        By default, applies document prefixes (for indexing). Use is_query=True
        for batch query embedding (less common).

        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process per batch (default: 32)
            is_query: If True, apply query prefix; else apply document prefix

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If any embedding generation fails
            ValueError: If texts list is empty or batch_size < 1

        Example:
            >>> client = OllamaClient()
            >>> texts = ["text1", "text2", "text3"]
            >>> embeddings = client.embed_batch(texts, batch_size=2)
            >>> len(embeddings)
            3
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

        embeddings = []

        try:
            # Process texts in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Generate embeddings for each text in the batch
                for text in batch:
                    embedding = self.embed(text, is_query=is_query)
                    embeddings.append(embedding)

            return embeddings

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close session."""
        self._session.close()
