"""Tests for Ollama embedding client."""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from docvec.embedding.ollama_client import (
    OllamaClient,
    EmbeddingError,
    retry_with_backoff,
)


class TestRetryDecorator:
    """Test retry_with_backoff decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_backoff(max_retries=3)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_success_after_failures(self):
        """Test function succeeds after initial failures."""
        mock_func = Mock(
            side_effect=[
                requests.RequestException("error1"),
                requests.RequestException("error2"),
                "success",
            ]
        )
        decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_retry_exhausted(self):
        """Test all retries exhausted."""
        mock_func = Mock(side_effect=requests.RequestException("persistent error"))
        decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

        with pytest.raises(requests.RequestException, match="persistent error"):
            decorated()

        assert mock_func.call_count == 3

    def test_retry_exponential_backoff(self):
        """Test exponential backoff delays."""
        mock_func = Mock(
            side_effect=[
                requests.RequestException("error1"),
                requests.RequestException("error2"),
                "success",
            ]
        )
        decorated = retry_with_backoff(max_retries=3, base_delay=0.1, max_delay=1.0)(
            mock_func
        )

        start_time = time.time()
        result = decorated()
        elapsed = time.time() - start_time

        assert result == "success"
        # Should take at least base_delay * (2^0 + 2^1) = 0.1 + 0.2 = 0.3s
        assert elapsed >= 0.2


class TestOllamaClient:
    """Test OllamaClient class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        client = OllamaClient()

        assert client.host == "http://localhost:11434"
        assert client.model == "nomic-embed-text"
        assert client.timeout == 30
        assert client._session is not None

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        client = OllamaClient(
            host="http://custom-host:8080", model="custom-model", timeout=60
        )

        assert client.host == "http://custom-host:8080"
        assert client.model == "custom-model"
        assert client.timeout == 60

    def test_init_strips_trailing_slash(self):
        """Test host URL trailing slash is removed."""
        client = OllamaClient(host="http://localhost:11434/")

        assert client.host == "http://localhost:11434"

    @patch("requests.Session.get")
    def test_health_check_success_model_available(self, mock_get):
        """Test health check when model is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "nomic-embed-text"}, {"name": "llama2"}]
        }
        mock_get.return_value = mock_response

        client = OllamaClient()
        result = client.health_check()

        assert result is True
        mock_get.assert_called_once()

    @patch("requests.Session.get")
    @patch("requests.Session.post")
    def test_health_check_model_not_listed_but_loadable(self, mock_post, mock_get):
        """Test health check when model not listed but can be loaded."""
        # GET /api/tags returns empty models list
        mock_get_response = Mock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # POST /api/embeddings succeeds (model loads)
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_post.return_value = mock_post_response

        client = OllamaClient()
        result = client.health_check()

        assert result is True

    @patch("requests.Session.get")
    def test_health_check_server_unreachable(self, mock_get):
        """Test health check when server is unreachable."""
        mock_get.side_effect = requests.RequestException("Connection refused")

        client = OllamaClient()
        result = client.health_check()

        assert result is False

    @patch("requests.Session.post")
    def test_embed_success(self, mock_post):
        """Test successful single text embedding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}
        mock_post.return_value = mock_response

        client = OllamaClient()
        embedding = client.embed("Hello world")

        assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.assert_called_once()

        # Verify request payload
        # Note: nomic-embed-text adds "search_document: " prefix for documents
        call_args = mock_post.call_args
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["prompt"] == "search_document: Hello world"

    @patch("requests.Session.post")
    def test_embed_query_with_prefix(self, mock_post):
        """Test query embedding adds model-specific prefix."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        # Test nomic-embed-text query prefix
        client = OllamaClient(model="nomic-embed-text")
        client.embed_query("What is Python?")
        call_args = mock_post.call_args
        assert call_args[1]["json"]["prompt"] == "search_query: What is Python?"

    @patch("requests.Session.post")
    def test_embed_query_mxbai_prefix(self, mock_post):
        """Test mxbai-embed-large query prefix."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        client = OllamaClient(model="mxbai-embed-large")
        client.embed_query("What is Python?")
        call_args = mock_post.call_args
        expected_prefix = "Represent this sentence for searching relevant passages: "
        assert call_args[1]["json"]["prompt"] == f"{expected_prefix}What is Python?"

    @patch("requests.Session.post")
    def test_embed_document_mxbai_no_prefix(self, mock_post):
        """Test mxbai-embed-large document embedding has no prefix."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_post.return_value = mock_response

        client = OllamaClient(model="mxbai-embed-large")
        client.embed_document("Python is a programming language.")
        call_args = mock_post.call_args
        # mxbai has no document prefix
        assert call_args[1]["json"]["prompt"] == "Python is a programming language."

    def test_embed_empty_text(self):
        """Test embedding with empty text raises ValueError."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            client.embed("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            client.embed("   ")

    @patch("requests.Session.post")
    def test_embed_timeout(self, mock_post):
        """Test embedding request timeout."""
        mock_post.side_effect = requests.Timeout("Request timeout")

        client = OllamaClient(timeout=5)

        with pytest.raises(EmbeddingError, match="Request timeout after 5s"):
            client.embed("Hello")

    @patch("requests.Session.post")
    def test_embed_no_embedding_in_response(self, mock_post):
        """Test handling of response without embedding."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(EmbeddingError, match="No embedding returned"):
            client.embed("Hello")

    @patch("requests.Session.post")
    def test_embed_request_exception(self, mock_post):
        """Test handling of request exceptions."""
        mock_post.side_effect = requests.RequestException("Network error")

        client = OllamaClient()

        with pytest.raises(EmbeddingError, match="Ollama API request failed"):
            client.embed("Hello")

    @patch("requests.Session.post")
    def test_embed_with_retry(self, mock_post):
        """Test embedding succeeds after retry."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.RequestException("Temporary error"),
            Mock(status_code=200, json=lambda: {"embedding": [0.1, 0.2, 0.3]}),
        ]

        client = OllamaClient()
        embedding = client.embed("Hello")

        assert embedding == [0.1, 0.2, 0.3]
        assert mock_post.call_count == 2

    @patch.object(OllamaClient, "embed")
    def test_embed_batch_success(self, mock_embed):
        """Test successful batch embedding."""
        mock_embed.side_effect = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

        client = OllamaClient()
        texts = ["text1", "text2", "text3"]
        embeddings = client.embed_batch(texts, batch_size=2)

        assert len(embeddings) == 3
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
        assert embeddings[2] == [0.7, 0.8, 0.9]
        assert mock_embed.call_count == 3

    def test_embed_batch_empty_list(self):
        """Test batch embedding with empty list."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            client.embed_batch([])

    def test_embed_batch_invalid_batch_size(self):
        """Test batch embedding with invalid batch size."""
        client = OllamaClient()

        with pytest.raises(ValueError, match="Batch size must be at least 1"):
            client.embed_batch(["text1"], batch_size=0)

        with pytest.raises(ValueError, match="Batch size must be at least 1"):
            client.embed_batch(["text1"], batch_size=-1)

    @patch.object(OllamaClient, "embed")
    def test_embed_batch_large_dataset(self, mock_embed):
        """Test batch embedding with large dataset."""
        # Generate 100 mock embeddings
        mock_embed.side_effect = [[i * 0.1] * 3 for i in range(100)]

        client = OllamaClient()
        texts = [f"text{i}" for i in range(100)]
        embeddings = client.embed_batch(texts, batch_size=32)

        assert len(embeddings) == 100
        # With batch_size=32, should process in 4 batches (32+32+32+4)
        assert mock_embed.call_count == 100

    @patch.object(OllamaClient, "embed")
    def test_embed_batch_propagates_errors(self, mock_embed):
        """Test batch embedding propagates individual embedding errors."""
        mock_embed.side_effect = [
            [0.1, 0.2, 0.3],
            EmbeddingError("Failed to embed text2"),
        ]

        client = OllamaClient()
        texts = ["text1", "text2", "text3"]

        with pytest.raises(EmbeddingError, match="Failed to embed text2"):
            client.embed_batch(texts, batch_size=5)

    def test_context_manager(self):
        """Test OllamaClient as context manager."""
        with patch("requests.Session.close") as mock_close:
            with OllamaClient() as client:
                assert client is not None

            mock_close.assert_called_once()

    @patch("requests.Session.post")
    def test_custom_timeout_configuration(self, mock_post):
        """Test custom timeout is used in requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_post.return_value = mock_response

        client = OllamaClient(timeout=60)
        client.embed("Hello")

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 60

    @patch("requests.Session.post")
    def test_http_error_handling(self, mock_post):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.HTTPError("Server error")
        mock_post.return_value = mock_response

        client = OllamaClient()

        with pytest.raises(EmbeddingError):
            client.embed("Hello")
