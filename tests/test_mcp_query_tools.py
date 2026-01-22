"""Tests for MCP query tools with semantic search and filtering.

This module tests QueryTools functionality including:
- Semantic search with embeddings
- Metadata filtering
- Token budget management
- Result formatting
- Error handling
"""

from unittest.mock import Mock

import pytest

from docvec.embedding.ollama_client import EmbeddingError, OllamaClient
from docvec.mcp_tools.query_tools import (
    QueryError,
    QueryResult,
    QueryTools,
    TokenCounter,
)
from docvec.storage.chroma_store import ChromaStore, StorageError


@pytest.fixture
def mock_embedder():
    """Provide mock OllamaClient for testing."""
    embedder = Mock(spec=OllamaClient)
    # Return a consistent embedding vector for all embed methods
    embedder.embed.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    embedder.embed_document.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return embedder


@pytest.fixture
def mock_storage():
    """Provide mock ChromaStore for testing."""
    storage = Mock(spec=ChromaStore)
    return storage


@pytest.fixture
def query_tools(mock_embedder, mock_storage):
    """Provide QueryTools instance with mocked dependencies."""
    return QueryTools(embedder=mock_embedder, storage=mock_storage)


@pytest.fixture
def sample_raw_results():
    """Provide sample raw results from ChromaDB."""
    return {
        "ids": ["1", "2", "3"],
        "documents": [
            "Python is a high-level programming language.",
            "Machine learning uses statistical methods.",
            "Data science combines programming and statistics.",
        ],
        "metadatas": [
            {"source_file": "python.md", "chunk_index": 0},
            {"source_file": "ml.md", "chunk_index": 0},
            {"source_file": "ds.md", "chunk_index": 1},
        ],
        "distances": [0.1, 0.2, 0.3],
    }


class TestTokenCounter:
    """Test TokenCounter utility."""

    def test_count_simple_text(self):
        """Test counting tokens in simple text."""
        count = TokenCounter.count("Hello world")
        assert count == 2  # 2 words * 1.3 = 2.6, int() = 2

    def test_count_with_punctuation(self):
        """Test counting with punctuation."""
        count = TokenCounter.count("Hello, world! How are you?")
        # 5 words * 1.3 = 6.5, int() = 6
        assert count == 6

    def test_count_empty_string(self):
        """Test counting empty string."""
        assert TokenCounter.count("") == 0

    def test_count_whitespace_only(self):
        """Test counting whitespace-only string."""
        assert TokenCounter.count("   ") == 0

    def test_count_long_text(self):
        """Test counting longer text."""
        text = "The quick brown fox jumps over the lazy dog."
        # 9 words * 1.3 = 11.7, int() = 11
        count = TokenCounter.count(text)
        assert count == 11


class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test creating a QueryResult."""
        result = QueryResult(
            chunk_id="123",
            content="Test content",
            score=0.5,
            metadata={"source": "test.md"},
            token_count=10,
        )

        assert result.chunk_id == "123"
        assert result.content == "Test content"
        assert result.score == 0.5
        assert result.metadata == {"source": "test.md"}
        assert result.token_count == 10


class TestQueryToolsInitialization:
    """Test QueryTools initialization."""

    def test_init_with_all_dependencies(self, mock_embedder, mock_storage):
        """Test initialization with all dependencies."""
        counter = TokenCounter()
        tools = QueryTools(
            embedder=mock_embedder, storage=mock_storage, counter=counter
        )

        assert tools._embedder is mock_embedder
        assert tools._storage is mock_storage
        assert tools._counter is counter

    def test_init_with_default_counter(self, mock_embedder, mock_storage):
        """Test initialization with default counter."""
        tools = QueryTools(embedder=mock_embedder, storage=mock_storage)

        assert tools._embedder is mock_embedder
        assert tools._storage is mock_storage
        assert isinstance(tools._counter, TokenCounter)


class TestSearch:
    """Test basic semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_basic(self, query_tools, mock_embedder, mock_storage):
        """Test basic search operation."""
        mock_storage.search.return_value = {
            "ids": ["1"],
            "documents": ["Python programming"],
            "metadatas": [{"source_file": "test.md"}],
            "distances": [0.1],
        }

        result = await query_tools.search("python", n_results=5)

        # Verify embedder was called with embed(is_query=True) for model-specific prefixes
        mock_embedder.embed.assert_called_once_with("python", is_query=True)

        # Verify storage search was called with correct parameters
        mock_storage.search.assert_called_once()
        call_args = mock_storage.search.call_args
        assert call_args[1]["n_results"] == 5
        assert call_args[1]["where"] is None

        # Verify result structure
        assert "results" in result
        assert "total_results" in result
        assert "total_tokens" in result
        assert "query" in result
        assert result["query"] == "python"
        assert result["total_results"] == 1

    @pytest.mark.asyncio
    async def test_search_formats_results(
        self, query_tools, mock_embedder, mock_storage, sample_raw_results
    ):
        """Test that search formats results correctly."""
        mock_storage.search.return_value = sample_raw_results

        result = await query_tools.search("test query", n_results=3)

        # Check results are formatted as QueryResult objects
        assert len(result["results"]) == 3
        for query_result in result["results"]:
            assert isinstance(query_result, QueryResult)
            assert query_result.chunk_id in ["1", "2", "3"]
            assert query_result.token_count > 0

    @pytest.mark.asyncio
    async def test_search_calculates_total_tokens(
        self, query_tools, mock_storage, sample_raw_results
    ):
        """Test that search calculates total tokens."""
        mock_storage.search.return_value = sample_raw_results

        result = await query_tools.search("test", n_results=3)

        # Total should be sum of individual token counts
        expected_total = sum(r.token_count for r in result["results"])
        assert result["total_tokens"] == expected_total
        assert result["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_search_empty_query_raises_error(self, query_tools):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await query_tools.search("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            await query_tools.search("   ")

    @pytest.mark.asyncio
    async def test_search_invalid_n_results_raises_error(self, query_tools):
        """Test that invalid n_results raises ValueError."""
        with pytest.raises(ValueError, match="n_results must be at least 1"):
            await query_tools.search("test", n_results=0)

        with pytest.raises(ValueError, match="n_results must be at least 1"):
            await query_tools.search("test", n_results=-5)

    @pytest.mark.asyncio
    async def test_search_embedding_error_raises_query_error(
        self, query_tools, mock_embedder
    ):
        """Test that embedding error is wrapped in QueryError."""
        mock_embedder.embed.side_effect = EmbeddingError("Embedding failed")

        with pytest.raises(QueryError, match="Search failed"):
            await query_tools.search("test")

    @pytest.mark.asyncio
    async def test_search_storage_error_raises_query_error(
        self, query_tools, mock_storage
    ):
        """Test that storage error is wrapped in QueryError."""
        mock_storage.search.side_effect = StorageError("Storage failed")

        with pytest.raises(QueryError, match="Search failed"):
            await query_tools.search("test")

    @pytest.mark.asyncio
    async def test_search_empty_results(self, query_tools, mock_storage):
        """Test search with no results."""
        mock_storage.search.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        result = await query_tools.search("nonexistent")

        assert result["total_results"] == 0
        assert result["total_tokens"] == 0
        assert len(result["results"]) == 0


class TestSearchWithFilters:
    """Test filtered semantic search functionality."""

    @pytest.mark.asyncio
    async def test_search_with_filters_basic(
        self, query_tools, mock_embedder, mock_storage
    ):
        """Test basic filtered search."""
        filters = {"source_file": "test.md"}
        mock_storage.search.return_value = {
            "ids": ["1"],
            "documents": ["Filtered content"],
            "metadatas": [{"source_file": "test.md"}],
            "distances": [0.1],
        }

        result = await query_tools.search_with_filters(
            "test query", filters=filters, n_results=5
        )

        # Verify storage search was called with filters
        mock_storage.search.assert_called_once()
        call_args = mock_storage.search.call_args
        assert call_args[1]["where"] == filters

        # Verify result includes filters
        assert result["filters"] == filters
        assert result["total_results"] == 1

    @pytest.mark.asyncio
    async def test_search_with_complex_filters(self, query_tools, mock_storage):
        """Test search with complex nested filters."""
        filters = {
            "source_file": "api.md",
            "page_number": 1,
            "section": "authentication",
        }
        mock_storage.search.return_value = {
            "ids": ["1", "2"],
            "documents": ["Auth content 1", "Auth content 2"],
            "metadatas": [filters, filters],
            "distances": [0.1, 0.2],
        }

        result = await query_tools.search_with_filters(
            "authentication", filters=filters, n_results=10
        )

        assert result["total_results"] == 2
        assert result["filters"] == filters

    @pytest.mark.asyncio
    async def test_search_with_filters_empty_query_raises_error(self, query_tools):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await query_tools.search_with_filters("", filters={})

    @pytest.mark.asyncio
    async def test_search_with_filters_invalid_filters_raises_error(self, query_tools):
        """Test that invalid filters raise ValueError."""
        with pytest.raises(ValueError, match="Filters must be a dictionary"):
            await query_tools.search_with_filters("test", filters="not a dict")

        with pytest.raises(ValueError, match="Filters must be a dictionary"):
            await query_tools.search_with_filters("test", filters=["list"])

    @pytest.mark.asyncio
    async def test_search_with_filters_invalid_n_results_raises_error(
        self, query_tools
    ):
        """Test that invalid n_results raises ValueError."""
        with pytest.raises(ValueError, match="n_results must be at least 1"):
            await query_tools.search_with_filters("test", filters={}, n_results=0)

    @pytest.mark.asyncio
    async def test_search_with_filters_no_matches(self, query_tools, mock_storage):
        """Test filtered search with no matching results."""
        filters = {"source_file": "nonexistent.md"}
        mock_storage.search.return_value = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": [],
        }

        result = await query_tools.search_with_filters(
            "test", filters=filters, n_results=5
        )

        assert result["total_results"] == 0
        assert result["total_tokens"] == 0


class TestSearchWithBudget:
    """Test token budget-aware search functionality."""

    @pytest.mark.asyncio
    async def test_search_with_budget_within_limit(self, query_tools, mock_storage):
        """Test search that fits within budget."""
        # Create results with known token counts
        # "short text" = 2 words * 1.3 = 2 tokens each
        mock_storage.search.return_value = {
            "ids": ["1", "2", "3"],
            "documents": ["short text", "short text", "short text"],
            "metadatas": [{}, {}, {}],
            "distances": [0.1, 0.2, 0.3],
        }

        result = await query_tools.search_with_budget("test", max_tokens=10)

        # Should return all 3 results (6 tokens total < 10)
        assert result["total_results"] == 3
        assert result["total_tokens"] <= 10
        assert result["max_tokens"] == 10
        assert not result["budget_exceeded"]

    @pytest.mark.asyncio
    async def test_search_with_budget_exceeds_limit(self, query_tools, mock_storage):
        """Test search that exceeds budget."""
        # Each result: "this is a longer text" = 5 words * 1.3 = 6 tokens
        mock_storage.search.return_value = {
            "ids": ["1", "2", "3", "4"],
            "documents": [
                "this is a longer text",
                "this is a longer text",
                "this is a longer text",
                "this is a longer text",
            ],
            "metadatas": [{}, {}, {}, {}],
            "distances": [0.1, 0.2, 0.3, 0.4],
        }

        result = await query_tools.search_with_budget("test", max_tokens=15)

        # Should only return 2 results (12 tokens, next would exceed 15)
        assert result["total_results"] == 2
        assert result["total_tokens"] <= 15
        assert result["budget_exceeded"]

    @pytest.mark.asyncio
    async def test_search_with_budget_empty_query_raises_error(self, query_tools):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await query_tools.search_with_budget("", max_tokens=100)

    @pytest.mark.asyncio
    async def test_search_with_budget_invalid_max_tokens_raises_error(
        self, query_tools
    ):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            await query_tools.search_with_budget("test", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be at least 1"):
            await query_tools.search_with_budget("test", max_tokens=-10)

    @pytest.mark.asyncio
    async def test_search_with_budget_no_results_fit(self, query_tools, mock_storage):
        """Test when no results fit within budget."""
        # Create a result that exceeds budget
        mock_storage.search.return_value = {
            "ids": ["1"],
            "documents": ["this is a very long document that exceeds budget"],
            "metadatas": [{}],
            "distances": [0.1],
        }

        result = await query_tools.search_with_budget("test", max_tokens=5)

        # Should return 0 results
        assert result["total_results"] == 0
        assert result["total_tokens"] == 0
        assert result["budget_exceeded"]

    @pytest.mark.asyncio
    async def test_search_with_budget_respects_relevance_order(
        self, query_tools, mock_storage
    ):
        """Test that budget search respects relevance ordering."""
        # Results ordered by relevance (distance)
        mock_storage.search.return_value = {
            "ids": ["1", "2", "3"],
            "documents": ["first result", "second result", "third result"],
            "metadatas": [{}, {}, {}],
            "distances": [0.1, 0.2, 0.3],  # Ordered by relevance
        }

        result = await query_tools.search_with_budget("test", max_tokens=20)

        # Should return results in order until budget exceeded
        assert result["results"][0].chunk_id == "1"
        assert result["results"][0].score == 0.1


class TestFormatResults:
    """Test result formatting functionality."""

    def test_format_results_basic(self, query_tools, sample_raw_results):
        """Test formatting raw ChromaDB results."""
        results = query_tools._format_results(sample_raw_results)

        assert len(results) == 3
        assert all(isinstance(r, QueryResult) for r in results)

        # Verify first result
        assert results[0].chunk_id == "1"
        assert results[0].content == "Python is a high-level programming language."
        assert results[0].score == 0.1
        assert results[0].metadata == {"source_file": "python.md", "chunk_index": 0}
        assert results[0].token_count > 0

    def test_format_results_empty(self, query_tools):
        """Test formatting empty results."""
        raw_results = {"ids": [], "documents": [], "metadatas": [], "distances": []}

        results = query_tools._format_results(raw_results)

        assert results == []

    def test_format_results_calculates_tokens(self, query_tools):
        """Test that formatting calculates token counts."""
        raw_results = {
            "ids": ["1", "2"],
            "documents": ["short", "this is a longer document with more words"],
            "metadatas": [{}, {}],
            "distances": [0.1, 0.2],
        }

        results = query_tools._format_results(raw_results)

        # Longer document should have more tokens
        assert results[1].token_count > results[0].token_count


class TestCalculateTotalTokens:
    """Test token calculation functionality."""

    def test_calculate_total_tokens_basic(self, query_tools):
        """Test calculating total tokens."""
        results = [
            QueryResult("1", "text", 0.1, {}, 10),
            QueryResult("2", "text", 0.2, {}, 20),
            QueryResult("3", "text", 0.3, {}, 30),
        ]

        total = query_tools._calculate_total_tokens(results)

        assert total == 60

    def test_calculate_total_tokens_empty(self, query_tools):
        """Test calculating tokens for empty list."""
        total = query_tools._calculate_total_tokens([])

        assert total == 0

    def test_calculate_total_tokens_single(self, query_tools):
        """Test calculating tokens for single result."""
        results = [QueryResult("1", "text", 0.1, {}, 42)]

        total = query_tools._calculate_total_tokens(results)

        assert total == 42


class TestEmbedQuery:
    """Test query embedding functionality."""

    def test_embed_query_calls_embedder(self, query_tools, mock_embedder):
        """Test that _embed_query calls embedder.embed(is_query=True) correctly."""
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        result = query_tools._embed_query("test query")

        mock_embedder.embed.assert_called_once_with("test query", is_query=True)
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_propagates_error(self, query_tools, mock_embedder):
        """Test that embedding errors are propagated."""
        mock_embedder.embed.side_effect = EmbeddingError("Failed")

        with pytest.raises(EmbeddingError):
            query_tools._embed_query("test")


class TestIntegration:
    """Integration tests with real components."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self, tmp_path):
        """Test complete search workflow with real components."""
        # Create real ChromaStore
        storage = ChromaStore(db_path=tmp_path / "test_db")

        # Add some test data
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        documents = [
            "Python programming language",
            "Machine learning algorithms",
            "Data science tools",
        ]
        metadatas = [
            {"doc_hash": "hash1", "source_file": "python.md"},
            {"doc_hash": "hash2", "source_file": "ml.md"},
            {"doc_hash": "hash3", "source_file": "ds.md"},
        ]

        storage.add(embeddings, documents, metadatas)

        # Create mock embedder
        mock_embedder = Mock(spec=OllamaClient)
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        # Create QueryTools
        tools = QueryTools(embedder=mock_embedder, storage=storage)

        # Perform search
        result = await tools.search("python", n_results=2)

        # Verify results
        assert result["total_results"] > 0
        assert result["total_tokens"] > 0
        assert len(result["results"]) > 0
        assert all(isinstance(r, QueryResult) for r in result["results"])
