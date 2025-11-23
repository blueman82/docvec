"""MCP query tools for semantic search with ChromaDB and Ollama integration.

This module provides query tools that embed user queries and search ChromaDB with:
- Semantic search with relevance scoring
- Metadata filtering support
- Token budget awareness for prompt engineering
- Result formatting for easy inclusion in prompts
"""

from dataclasses import dataclass
from typing import Optional

from docvec.embedding.ollama_client import EmbeddingError, OllamaClient
from docvec.storage.chroma_store import ChromaStore, StorageError


class QueryError(Exception):
    """Custom exception for query-related errors."""

    pass


class TokenCounter:
    """Simple token counter using whitespace splitting approximation.

    Provides fast token counting for result filtering. Uses a simple
    whitespace-based approximation: tokens ~= words * 1.3 (accounting
    for punctuation and common patterns).

    For production use with specific models, consider using tiktoken
    or model-specific tokenizers.
    """

    @staticmethod
    def count(text: str) -> int:
        """Count approximate tokens in text.

        Args:
            text: Input text to count tokens

        Returns:
            Approximate token count

        Example:
            >>> TokenCounter.count("Hello world!")
            3
        """
        if not text:
            return 0

        # Simple approximation: words * 1.3 to account for punctuation
        # This is a conservative estimate for most English text
        words = len(text.split())
        return int(words * 1.3)


@dataclass
class QueryResult:
    """Result from a semantic search query.

    Attributes:
        chunk_id: Unique identifier for the chunk
        content: The actual text content of the chunk
        score: Relevance score (lower distance = higher relevance)
        metadata: Associated metadata (source_file, chunk_index, etc.)
        token_count: Number of tokens in the content
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict
    token_count: int


class QueryTools:
    """MCP tools for semantic search queries.

    Provides tools for embedding queries and searching ChromaDB with support
    for filtering, relevance scoring, and token budget management.

    Args:
        embedder: OllamaClient instance for query embedding
        storage: ChromaStore instance for vector search
        counter: Optional TokenCounter instance (defaults to TokenCounter)

    Attributes:
        _embedder: Ollama embedding client
        _storage: ChromaDB storage layer
        _counter: Token counter utility

    Example:
        >>> embedder = OllamaClient()
        >>> storage = ChromaStore(Path("./db"))
        >>> tools = QueryTools(embedder, storage)
        >>> results = await tools.search("machine learning", n_results=5)
    """

    def __init__(
        self,
        embedder: OllamaClient,
        storage: ChromaStore,
        counter: Optional[TokenCounter] = None,
    ):
        """Initialize query tools with dependencies.

        Args:
            embedder: OllamaClient for query embedding
            storage: ChromaStore for vector search
            counter: Optional TokenCounter (defaults to TokenCounter())
        """
        self._embedder = embedder
        self._storage = storage
        self._counter = counter or TokenCounter()

    async def search(self, query: str, n_results: int = 5) -> dict:
        """Perform semantic search for a query.

        Embeds the query using Ollama and searches ChromaDB for the most
        relevant chunks. Returns formatted results with relevance scores.

        Args:
            query: Search query string
            n_results: Maximum number of results to return (default: 5)

        Returns:
            Dictionary containing:
                - results: List of QueryResult objects
                - total_results: Total number of results returned
                - total_tokens: Sum of tokens across all results
                - query: Original query string

        Raises:
            QueryError: If search fails
            ValueError: If query is empty or n_results < 1

        Example:
            >>> results = await tools.search("Python programming", n_results=3)
            >>> for result in results["results"]:
            ...     print(f"{result.content[:50]}... (score: {result.score})")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        try:
            # Embed the query
            query_embedding = self._embed_query(query)

            # Search ChromaDB
            raw_results = self._storage.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where=None,
            )

            # Format results
            results = self._format_results(raw_results)
            total_tokens = self._calculate_total_tokens(results)

            return {
                "results": results,
                "total_results": len(results),
                "total_tokens": total_tokens,
                "query": query,
            }

        except (EmbeddingError, StorageError) as e:
            raise QueryError(f"Search failed: {e}") from e
        except Exception as e:
            raise QueryError(f"Unexpected error during search: {e}") from e

    async def search_with_filters(
        self, query: str, filters: dict, n_results: int = 5
    ) -> dict:
        """Perform semantic search with metadata filtering.

        Embeds the query and searches ChromaDB with metadata filters applied.
        Filters are passed directly to ChromaDB's WHERE clause.

        Args:
            query: Search query string
            filters: Metadata filters (e.g., {"source_file": "readme.md"})
            n_results: Maximum number of results to return (default: 5)

        Returns:
            Dictionary containing:
                - results: List of QueryResult objects matching filters
                - total_results: Total number of results returned
                - total_tokens: Sum of tokens across all results
                - query: Original query string
                - filters: Applied filters

        Raises:
            QueryError: If search fails
            ValueError: If query is empty, filters invalid, or n_results < 1

        Example:
            >>> filters = {"source_file": "docs/api.md", "page_number": 1}
            >>> results = await tools.search_with_filters(
            ...     "authentication",
            ...     filters=filters,
            ...     n_results=5
            ... )
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")

        if n_results < 1:
            raise ValueError("n_results must be at least 1")

        try:
            # Embed the query
            query_embedding = self._embed_query(query)

            # Search ChromaDB with filters
            raw_results = self._storage.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where=filters,
            )

            # Format results
            results = self._format_results(raw_results)
            total_tokens = self._calculate_total_tokens(results)

            return {
                "results": results,
                "total_results": len(results),
                "total_tokens": total_tokens,
                "query": query,
                "filters": filters,
            }

        except (EmbeddingError, StorageError) as e:
            raise QueryError(f"Filtered search failed: {e}") from e
        except Exception as e:
            raise QueryError(f"Unexpected error during filtered search: {e}") from e

    async def search_with_budget(self, query: str, max_tokens: int) -> dict:
        """Search and return results within a token budget.

        Performs semantic search and returns the top results that fit within
        the specified token budget. Critical for prompt engineering where
        context window size is limited.

        Args:
            query: Search query string
            max_tokens: Maximum total tokens allowed in results

        Returns:
            Dictionary containing:
                - results: List of QueryResult objects within budget
                - total_results: Number of results returned
                - total_tokens: Actual tokens used (≤ max_tokens)
                - query: Original query string
                - max_tokens: Requested token budget
                - budget_exceeded: Whether more results were available

        Raises:
            QueryError: If search fails
            ValueError: If query is empty or max_tokens < 1

        Example:
            >>> # Get results that fit in 1000 tokens
            >>> results = await tools.search_with_budget(
            ...     "API documentation",
            ...     max_tokens=1000
            ... )
            >>> print(f"Used {results['total_tokens']} of {results['max_tokens']}")
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        try:
            # Embed the query
            query_embedding = self._embed_query(query)

            # Search with a large initial batch to have candidates
            # Use a reasonable upper limit (e.g., 50 results)
            raw_results = self._storage.search(
                query_embedding=query_embedding,
                n_results=50,
                where=None,
            )

            # Format all results
            all_results = self._format_results(raw_results)

            # Filter results to fit within budget
            filtered_results = []
            current_tokens = 0

            for result in all_results:
                if current_tokens + result.token_count <= max_tokens:
                    filtered_results.append(result)
                    current_tokens += result.token_count
                else:
                    # Budget exceeded, stop adding results
                    break

            return {
                "results": filtered_results,
                "total_results": len(filtered_results),
                "total_tokens": current_tokens,
                "query": query,
                "max_tokens": max_tokens,
                "budget_exceeded": len(all_results) > len(filtered_results),
            }

        except (EmbeddingError, StorageError) as e:
            raise QueryError(f"Budget search failed: {e}") from e
        except Exception as e:
            raise QueryError(f"Unexpected error during budget search: {e}") from e

    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string using Ollama.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector

        Raises:
            EmbeddingError: If embedding fails
        """
        return self._embedder.embed(query)

    def _format_results(self, raw_results: dict) -> list[QueryResult]:
        """Format raw ChromaDB results into QueryResult objects.

        Args:
            raw_results: Raw results from ChromaDB search containing
                        ids, documents, metadatas, and distances

        Returns:
            List of QueryResult objects with token counts

        Example:
            >>> raw = {
            ...     "ids": ["1", "2"],
            ...     "documents": ["text1", "text2"],
            ...     "metadatas": [{"source": "f1"}, {"source": "f2"}],
            ...     "distances": [0.1, 0.2]
            ... }
            >>> results = tools._format_results(raw)
        """
        results = []

        # ChromaDB returns parallel lists
        ids = raw_results.get("ids", [])
        documents = raw_results.get("documents", [])
        metadatas = raw_results.get("metadatas", [])
        distances = raw_results.get("distances", [])

        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            token_count = self._counter.count(content)

            result = QueryResult(
                chunk_id=chunk_id,
                content=content,
                score=distance,
                metadata=metadata,
                token_count=token_count,
            )
            results.append(result)

        return results

    def _calculate_total_tokens(self, results: list[QueryResult]) -> int:
        """Calculate total tokens across all results.

        Args:
            results: List of QueryResult objects

        Returns:
            Sum of token counts across all results

        Example:
            >>> results = [
            ...     QueryResult("1", "text", 0.1, {}, 10),
            ...     QueryResult("2", "text", 0.2, {}, 20)
            ... ]
            >>> total = tools._calculate_total_tokens(results)
            >>> print(total)
            30
        """
        return sum(result.token_count for result in results)
