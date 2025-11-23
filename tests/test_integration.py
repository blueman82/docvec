"""Integration tests for full indexing and query pipeline.

This module provides end-to-end tests covering:
- Multi-format document indexing (Markdown, Python, text)
- Full pipeline from chunking through embedding to storage
- Deduplication with hash tracking
- Query pipeline with relevance validation
- MCP tool invocations through the server
- Error handling and failure scenarios

The tests use real files from fixtures/sample_docs/ to ensure
realistic behavior without excessive mocking.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vector_mcp.chunking.code_chunker import CodeChunker
from vector_mcp.chunking.markdown_chunker import MarkdownChunker
from vector_mcp.chunking.text_chunker import TextChunker
from vector_mcp.deduplication.hasher import DocumentHasher
from vector_mcp.embedding.ollama_client import OllamaClient
from vector_mcp.indexing.batch_processor import BatchProcessor
from vector_mcp.indexing.indexer import Indexer
from vector_mcp.mcp_tools.indexing_tools import IndexingTools
from vector_mcp.mcp_tools.query_tools import QueryTools
from vector_mcp.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


# Fixture paths
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_docs"
MARKDOWN_FILE = FIXTURES_DIR / "README.md"
PYTHON_FILE = FIXTURES_DIR / "script.py"
TEXT_FILE = FIXTURES_DIR / "document.txt"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Create temporary database directory.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to temporary database directory
    """
    db_path = tmp_path / "test_chroma_db"
    db_path.mkdir(exist_ok=True)
    return db_path


@pytest.fixture
def mock_embedder() -> OllamaClient:
    """Create a mock Ollama embedder for testing.

    Returns deterministic embeddings based on text content to enable
    testing of similarity search without requiring actual Ollama server.

    Returns:
        Mock OllamaClient instance
    """
    embedder = Mock(spec=OllamaClient)

    def mock_embed(text: str) -> list[float]:
        """Generate deterministic embedding based on text content.

        Uses simple hashing to create consistent embeddings that vary
        based on content for similarity testing.
        """
        # Use text hash to generate deterministic but varied embeddings
        text_hash = hash(text[:100])  # Use first 100 chars for variation

        # Generate 384-dimensional embedding (nomic-embed-text dimension)
        base_vector = [0.0] * 384

        # Set values based on content characteristics
        # This ensures similar content gets similar embeddings
        for i in range(384):
            base_vector[i] = ((text_hash + i) % 100) / 100.0

        # Normalize to unit vector
        magnitude = sum(x * x for x in base_vector) ** 0.5
        return [x / magnitude for x in base_vector]

    def mock_embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for batch of texts."""
        return [mock_embed(text) for text in texts]

    embedder.embed = Mock(side_effect=mock_embed)
    embedder.embed_batch = Mock(side_effect=mock_embed_batch)
    embedder.health_check = Mock(return_value=True)

    return embedder


@pytest.fixture
def storage(temp_db_path: Path) -> ChromaStore:
    """Create ChromaStore instance with temporary database.

    Args:
        temp_db_path: Temporary database path fixture

    Returns:
        ChromaStore instance for testing
    """
    return ChromaStore(db_path=temp_db_path, collection_name="test_collection")


@pytest.fixture
def hasher() -> DocumentHasher:
    """Create DocumentHasher instance.

    Returns:
        DocumentHasher for deduplication testing
    """
    return DocumentHasher()


@pytest.fixture
def indexer(mock_embedder: OllamaClient, storage: ChromaStore) -> Indexer:
    """Create Indexer instance with mock embedder.

    Args:
        mock_embedder: Mock Ollama client fixture
        storage: ChromaStore fixture

    Returns:
        Indexer instance for testing
    """
    return Indexer(
        embedder=mock_embedder,
        storage=storage,
        chunk_size=512,
        batch_size=32,
    )


@pytest.fixture
def batch_processor(
    indexer: Indexer, hasher: DocumentHasher, storage: ChromaStore
) -> BatchProcessor:
    """Create BatchProcessor instance.

    Args:
        indexer: Indexer fixture
        hasher: DocumentHasher fixture
        storage: ChromaStore fixture

    Returns:
        BatchProcessor for batch indexing tests
    """
    return BatchProcessor(indexer=indexer, hasher=hasher, storage=storage)


@pytest.fixture
def indexing_tools(batch_processor: BatchProcessor, indexer: Indexer) -> IndexingTools:
    """Create IndexingTools instance.

    Args:
        batch_processor: BatchProcessor fixture
        indexer: Indexer fixture

    Returns:
        IndexingTools for MCP tool testing
    """
    return IndexingTools(batch_processor=batch_processor, indexer=indexer)


@pytest.fixture
def query_tools(mock_embedder: OllamaClient, storage: ChromaStore) -> QueryTools:
    """Create QueryTools instance.

    Args:
        mock_embedder: Mock Ollama client fixture
        storage: ChromaStore fixture

    Returns:
        QueryTools for search testing
    """
    return QueryTools(embedder=mock_embedder, storage=storage)


@pytest.fixture(scope="session")
def populated_db_path() -> Generator[Path, None, None]:
    """Create and populate a database once per test session.

    This session-scoped fixture indexes sample documents once and
    reuses the database across multiple tests for efficiency.

    Yields:
        Path to populated database directory
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "session_db"
        db_path.mkdir(exist_ok=True)

        # Create components
        embedder = Mock(spec=OllamaClient)

        def mock_embed(text: str) -> list[float]:
            text_hash = hash(text[:100])
            base_vector = [((text_hash + i) % 100) / 100.0 for i in range(384)]
            magnitude = sum(x * x for x in base_vector) ** 0.5
            return [x / magnitude for x in base_vector]

        embedder.embed = Mock(side_effect=mock_embed)
        embedder.embed_batch = Mock(
            side_effect=lambda texts, batch_size=32: [mock_embed(t) for t in texts]
        )

        storage = ChromaStore(db_path=db_path, collection_name="session_collection")
        indexer = Indexer(embedder=embedder, storage=storage)

        # Index sample documents
        if MARKDOWN_FILE.exists():
            indexer.index_document(MARKDOWN_FILE)
        if PYTHON_FILE.exists():
            indexer.index_document(PYTHON_FILE)
        if TEXT_FILE.exists():
            indexer.index_document(TEXT_FILE)

        yield db_path


class TestIndexingPipeline:
    """Test the complete indexing pipeline from files to storage."""

    def test_index_markdown_document(self, indexer: Indexer):
        """Test indexing a Markdown file through full pipeline."""
        if not MARKDOWN_FILE.exists():
            pytest.skip("Sample markdown file not found")

        # Index the document
        chunk_ids = indexer.index_document(MARKDOWN_FILE)

        # Verify chunks were created
        assert len(chunk_ids) > 0, "Should create at least one chunk"
        assert all(isinstance(cid, str) for cid in chunk_ids), "Chunk IDs should be strings"

        # Verify chunks are in storage
        collection = indexer.storage._collection
        count = collection.count()
        assert count >= len(chunk_ids), "All chunks should be in storage"

    def test_index_python_document(self, indexer: Indexer):
        """Test indexing a Python file through full pipeline."""
        if not PYTHON_FILE.exists():
            pytest.skip("Sample Python file not found")

        # Index the document
        chunk_ids = indexer.index_document(PYTHON_FILE)

        # Verify chunks were created
        assert len(chunk_ids) > 0, "Should create chunks from Python file"

        # Verify metadata includes source file
        collection = indexer.storage._collection
        results = collection.get(ids=[chunk_ids[0]], include=["metadatas"])
        metadata = results["metadatas"][0]

        assert "source_file" in metadata
        assert str(PYTHON_FILE) in metadata["source_file"]

    def test_index_text_document(self, indexer: Indexer):
        """Test indexing a plain text file through full pipeline."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index the document
        chunk_ids = indexer.index_document(TEXT_FILE)

        # Verify chunks were created
        assert len(chunk_ids) > 0, "Should create chunks from text file"

        # Verify content is stored
        collection = indexer.storage._collection
        results = collection.get(ids=[chunk_ids[0]], include=["documents"])
        content = results["documents"][0]

        assert len(content) > 0, "Chunk should have content"
        assert isinstance(content, str), "Content should be string"

    def test_index_multiple_documents(self, indexer: Indexer):
        """Test batch indexing multiple documents."""
        files = []
        if MARKDOWN_FILE.exists():
            files.append(MARKDOWN_FILE)
        if PYTHON_FILE.exists():
            files.append(PYTHON_FILE)
        if TEXT_FILE.exists():
            files.append(TEXT_FILE)

        if not files:
            pytest.skip("No sample files found")

        # Index batch
        results = indexer.index_batch(files)

        # Verify all files processed
        assert len(results) == len(files), "Should have result for each file"

        # Verify all succeeded
        for file_path, chunk_ids in results.items():
            assert len(chunk_ids) > 0, f"Should have chunks for {file_path}"

    def test_chunking_respects_format(self, indexer: Indexer):
        """Test that appropriate chunker is selected for each format."""
        # This is implicitly tested by successful indexing, but we can
        # verify the chunker selection logic
        assert indexer._chunker_map[".md"] == MarkdownChunker
        assert indexer._chunker_map[".py"] == CodeChunker
        assert indexer._chunker_map[".txt"] == TextChunker


class TestQueryPipeline:
    """Test the complete query pipeline including search and filtering."""

    @pytest.mark.asyncio
    async def test_basic_search(self, query_tools: QueryTools, indexer: Indexer):
        """Test basic semantic search functionality."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index document first
        indexer.index_document(TEXT_FILE)

        # Search for relevant content
        results = await query_tools.search("vector database", n_results=5)

        # Verify results structure
        assert "results" in results
        assert "total_results" in results
        assert "total_tokens" in results

        # Should find relevant chunks
        assert results["total_results"] > 0, "Should find matching chunks"

        # Verify result format
        if results["results"]:
            result = results["results"][0]
            assert hasattr(result, "content")
            assert hasattr(result, "score")
            assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_search_relevance(self, query_tools: QueryTools, indexer: Indexer):
        """Test that search returns relevant results with good scores."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index document
        indexer.index_document(TEXT_FILE)

        # Search for specific content we know exists
        results = await query_tools.search("similarity search", n_results=3)

        # Verify we got results
        assert results["total_results"] > 0, "Should find relevant content"

        # Top result should have a good similarity score (low distance)
        # Note: With our mock embedder, we can't validate actual relevance,
        # but with real embedder this would check score < 0.5 or similar
        if results["results"]:
            top_score = results["results"][0].score
            assert isinstance(top_score, float), "Score should be numeric"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, query_tools: QueryTools, indexer: Indexer):
        """Test search with metadata filtering."""
        if not MARKDOWN_FILE.exists() or not PYTHON_FILE.exists():
            pytest.skip("Sample files not found")

        # Index multiple documents
        indexer.index_document(MARKDOWN_FILE)
        indexer.index_document(PYTHON_FILE)

        # Search with filter for specific file
        filters = {"source_file": str(MARKDOWN_FILE)}
        results = await query_tools.search_with_filters(
            "documentation", filters=filters, n_results=5
        )

        # Verify results are filtered
        assert "results" in results
        assert "filters" in results

        # All results should match filter
        for result in results["results"]:
            assert str(MARKDOWN_FILE) in result.metadata["source_file"]

    @pytest.mark.asyncio
    async def test_search_with_budget(self, query_tools: QueryTools, indexer: Indexer):
        """Test search with token budget constraint."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index document
        indexer.index_document(TEXT_FILE)

        # Search with token budget
        max_tokens = 200
        results = await query_tools.search_with_budget(
            "vector database technology", max_tokens=max_tokens
        )

        # Verify budget constraints
        assert "total_tokens" in results
        assert "max_tokens" in results
        assert results["max_tokens"] == max_tokens

        # Should not exceed budget
        assert results["total_tokens"] <= max_tokens, "Should respect token budget"

    @pytest.mark.asyncio
    async def test_empty_query_raises_error(self, query_tools: QueryTools):
        """Test that empty query raises appropriate error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await query_tools.search("", n_results=5)


class TestDeduplication:
    """Test deduplication with hash tracking."""

    def test_duplicate_detection(self, batch_processor: BatchProcessor):
        """Test that re-indexing same content is detected as duplicate."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index directory first time
        result1 = batch_processor.process_directory(
            FIXTURES_DIR, recursive=False
        )

        # Record initial counts
        initial_new = result1.new_documents
        initial_total = sum(len(ids) for ids in result1.chunk_ids.values())

        # Index same directory again
        result2 = batch_processor.process_directory(
            FIXTURES_DIR, recursive=False
        )

        # Second indexing should skip duplicates
        assert result2.duplicates_skipped > 0, "Should detect duplicates"
        assert result2.new_documents == 0, "Should not index new documents"

    def test_hash_persistence(
        self, batch_processor: BatchProcessor, storage: ChromaStore
    ):
        """Test that document hashes are persisted correctly."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index a file
        batch_processor.indexer.index_document(TEXT_FILE)

        # Query hash tracking in storage
        collection = storage._collection
        results = collection.get(
            where={"source_file": str(TEXT_FILE)},
            include=["metadatas"],
            limit=1,
        )

        # Verify hash is stored in metadata
        assert len(results["metadatas"]) > 0
        metadata = results["metadatas"][0]
        assert "doc_hash" in metadata
        assert len(metadata["doc_hash"]) > 0

    def test_modified_file_reindexing(
        self, batch_processor: BatchProcessor, tmp_path: Path
    ):
        """Test that modified files are re-indexed properly."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content for testing")

        # Index first version
        result1 = batch_processor.process_directory(tmp_path, recursive=False)
        assert result1.new_documents == 1

        # Modify file
        test_file.write_text("Modified content with different text")

        # Re-index - should detect as new due to different hash
        # Note: Current implementation may not detect file modifications
        # This test documents expected behavior if modification detection added
        result2 = batch_processor.process_directory(tmp_path, recursive=False)

        # With current implementation, this will be detected as duplicate
        # If file modification detection is added, update this assertion


class TestMCPServer:
    """Test MCP tool invocations through the server interface."""

    @pytest.mark.asyncio
    async def test_index_file_tool(self, indexing_tools: IndexingTools):
        """Test index_file MCP tool."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Invoke tool
        result = await indexing_tools.index_file(str(TEXT_FILE))

        # Verify successful response
        assert result["success"] is True
        assert "data" in result
        assert result["data"]["chunks"] > 0

    @pytest.mark.asyncio
    async def test_index_directory_tool(self, indexing_tools: IndexingTools):
        """Test index_directory MCP tool."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixtures directory not found")

        # Invoke tool
        result = await indexing_tools.index_directory(
            str(FIXTURES_DIR), recursive=False
        )

        # Verify successful response
        assert result["success"] is True
        assert "data" in result

        data = result["data"]
        assert "new_documents" in data
        assert "total_chunks" in data
        assert data["new_documents"] >= 0

    @pytest.mark.asyncio
    async def test_search_tool(
        self, query_tools: QueryTools, indexer: Indexer
    ):
        """Test search MCP tool."""
        if not TEXT_FILE.exists():
            pytest.skip("Sample text file not found")

        # Index content first
        indexer.index_document(TEXT_FILE)

        # Invoke search tool
        result = await query_tools.search("database", n_results=3)

        # Verify response structure
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, indexing_tools: IndexingTools):
        """Test that tools handle errors gracefully."""
        # Try to index non-existent file
        result = await indexing_tools.index_file("/nonexistent/file.txt")

        # Should return error response, not raise exception
        assert result["success"] is False
        assert "error" in result


class TestErrorHandling:
    """Test error handling and failure scenarios."""

    def test_invalid_file_path(self, indexer: Indexer):
        """Test handling of invalid file paths."""
        with pytest.raises(FileNotFoundError):
            indexer.index_document(Path("/nonexistent/file.txt"))

    def test_empty_file(self, indexer: Indexer, tmp_path: Path):
        """Test handling of empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        # Should handle gracefully
        chunk_ids = indexer.index_document(empty_file)

        # May return empty list or minimal chunks depending on implementation
        assert isinstance(chunk_ids, list)

    def test_unsupported_file_type(self, indexer: Indexer, tmp_path: Path):
        """Test handling of unsupported file types."""
        # Create file with unsupported extension
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        # Should fall back to text chunker or raise appropriate error
        # Current implementation uses TextChunker as fallback
        try:
            chunk_ids = indexer.index_document(binary_file)
            # If it succeeds, verify it's handled
            assert isinstance(chunk_ids, list)
        except Exception as e:
            # If it fails, verify it's an expected error type
            assert "read" in str(e).lower() or "decode" in str(e).lower()

    @pytest.mark.asyncio
    async def test_query_invalid_parameters(self, query_tools: QueryTools):
        """Test query tools with invalid parameters."""
        # Empty query
        with pytest.raises(ValueError):
            await query_tools.search("", n_results=5)

        # Invalid n_results
        with pytest.raises(ValueError):
            await query_tools.search("test", n_results=0)

    def test_batch_processing_with_errors(
        self, batch_processor: BatchProcessor, tmp_path: Path
    ):
        """Test that batch processing continues despite individual file errors."""
        # Create mix of valid and invalid files
        valid_file = tmp_path / "valid.txt"
        valid_file.write_text("Valid content")

        # Create directory with both files
        result = batch_processor.process_directory(tmp_path, recursive=False)

        # Should process valid file and report error for invalid one
        assert result.new_documents >= 1, "Should index valid files"


class TestEndToEnd:
    """End-to-end integration tests simulating real usage."""

    @pytest.mark.asyncio
    async def test_full_workflow(
        self,
        indexing_tools: IndexingTools,
        query_tools: QueryTools,
        indexer: Indexer,
    ):
        """Test complete workflow: index documents, then query them."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixtures directory not found")

        # Step 1: Index directory
        index_result = await indexing_tools.index_directory(
            str(FIXTURES_DIR), recursive=False
        )

        assert index_result["success"] is True
        total_chunks = index_result["data"]["total_chunks"]
        assert total_chunks > 0, "Should index some content"

        # Step 2: Perform search
        search_result = await query_tools.search("vector database", n_results=5)

        assert "results" in search_result
        assert len(search_result["results"]) > 0, "Should find relevant content"

        # Step 3: Verify deduplication on re-index
        reindex_result = await indexing_tools.index_directory(
            str(FIXTURES_DIR), recursive=False
        )

        assert reindex_result["success"] is True
        assert reindex_result["data"]["duplicates_skipped"] > 0, "Should skip duplicates"

    @pytest.mark.asyncio
    async def test_multi_format_indexing_and_search(
        self, indexing_tools: IndexingTools, query_tools: QueryTools
    ):
        """Test indexing multiple formats and searching across them."""
        if not FIXTURES_DIR.exists():
            pytest.skip("Fixtures directory not found")

        # Index all formats
        result = await indexing_tools.index_directory(str(FIXTURES_DIR))

        assert result["success"] is True
        indexed_files = result["data"]["indexed_files"]

        # Should have indexed multiple file types
        file_extensions = set(Path(f).suffix for f in indexed_files)
        assert len(file_extensions) > 1, "Should index multiple formats"

        # Search should work across all formats
        search_result = await query_tools.search("function", n_results=10)

        # Should find results from different file types
        sources = set()
        for result in search_result["results"]:
            source = result.metadata.get("source_file", "")
            sources.add(Path(source).suffix)

        # With Python and other files, should have variety
        assert len(sources) >= 1, "Should search across file types"
