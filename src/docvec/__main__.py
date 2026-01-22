"""MCP server entry point for vector database document indexing.

This module provides the main entry point for the MCP server with:
- CLI argument parsing for flexible configuration
- Component initialization in dependency order
- Tool registration for indexing and query operations
- Signal handling for graceful shutdown
- Comprehensive logging throughout

Usage:
    python -m docvec [options]

    Options:
        --host HOST         Ollama server host (default: http://localhost:11434)
        --model MODEL       Embedding model name (default: nomic-embed-text)
        --db-path PATH      ChromaDB storage path (default: ./chroma_db)
        --collection NAME   Collection name (default: documents)
        --log-level LEVEL   Logging level (default: INFO)
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from docvec.deduplication.hasher import DocumentHasher
from docvec.embedding import create_embedding_provider
from docvec.indexing.batch_processor import BatchProcessor
from docvec.indexing.indexer import Indexer
from docvec.mcp_tools.indexing_tools import IndexingTools
from docvec.mcp_tools.management_tools import ManagementTools
from docvec.mcp_tools.query_tools import QueryTools
from docvec.storage.chroma_store import ChromaStore

# Initialize FastMCP server
mcp = FastMCP("vector-mcp")

# Global components (initialized in main)
indexing_tools: Optional[IndexingTools] = None
management_tools: Optional[ManagementTools] = None
query_tools: Optional[QueryTools] = None

logger = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure logging to stderr (never stdout for MCP servers).

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Note:
        STDIO-based MCP servers must never write to stdout as it corrupts
        JSON-RPC messages. All logging goes to stderr.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger to write to stderr
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,  # Critical: never use stdout in MCP servers
    )

    logger.info(f"Logging initialized at {log_level.upper()} level")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments with configuration defaults.

    Returns:
        Parsed arguments namespace

    Configuration precedence:
        1. CLI arguments (highest priority)
        2. Environment variables (if implemented)
        3. Config file (if implemented)
        4. Defaults (lowest priority)
    """
    parser = argparse.ArgumentParser(
        description="Vector MCP server for document indexing and semantic search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Embedding backend configuration
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default=os.environ.get("DOCVEC_EMBEDDING_BACKEND", "mlx"),
        choices=["mlx", "ollama"],
        help="Embedding backend to use ('mlx' for Apple Silicon, 'ollama' for server)",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=os.environ.get(
            "DOCVEC_MLX_MODEL", "mlx-community/mxbai-embed-large-v1"
        ),
        help="MLX embedding model (HuggingFace path)",
    )

    # Ollama configuration (used when --embedding-backend=ollama)
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("DOCVEC_HOST", "http://localhost:11434"),
        help="Ollama server host URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("DOCVEC_MODEL", "nomic-embed-text"),
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("DOCVEC_TIMEOUT", "30")),
        help="Ollama request timeout in seconds",
    )

    # ChromaDB configuration
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.environ.get("DOCVEC_DB_PATH", "./chroma_db"),
        help="ChromaDB persistent storage directory path",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=os.environ.get("DOCVEC_COLLECTION", "documents"),
        help="ChromaDB collection name",
    )

    # Indexing configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("DOCVEC_CHUNK_SIZE", "512")),
        help="Maximum tokens per chunk",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("DOCVEC_BATCH_SIZE", "128")),
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("DOCVEC_MAX_TOKENS", "512")),
        help="Maximum tokens per chunk for embedding model limits",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.environ.get("DOCVEC_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def initialize_components(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize all components in dependency order.

    Initialization order follows dependency graph:
    1. Configuration (parsed arguments)
    2. Services: EmbeddingProvider (MLX or Ollama), ChromaStore, DocumentHasher
    3. Processors: Indexer, BatchProcessor
    4. Tools: IndexingTools, QueryTools

    Args:
        args: Parsed CLI arguments

    Returns:
        Dictionary containing all initialized components

    Raises:
        Exception: If any component initialization fails
    """
    logger.info("Initializing components...")
    components: dict[str, Any] = {}

    try:
        # 1. Initialize embedding provider (MLX or Ollama based on config)
        logger.info(
            f"Initializing embedding provider (backend={args.embedding_backend})"
        )
        embedder = create_embedding_provider(
            backend=args.embedding_backend,
            host=args.host,
            model=args.model,
            timeout=args.timeout,
            mlx_model=args.mlx_model,
        )

        # For Ollama backend, ensure model is available (auto-pull if needed)
        if args.embedding_backend == "ollama":
            if hasattr(embedder, "ensure_model") and not embedder.ensure_model():
                logger.error(
                    f"Failed to ensure model '{args.model}' is available. "
                    "Please check that Ollama is running and the model name is correct."
                )
                raise RuntimeError(
                    f"Model '{args.model}' could not be loaded or pulled"
                )

        components["embedder"] = embedder

        # 2. Initialize ChromaDB storage
        db_path = Path(args.db_path).resolve()
        logger.info(
            f"Initializing ChromaStore (path={db_path}, collection={args.collection})"
        )
        storage = ChromaStore(
            db_path=db_path,
            collection_name=args.collection,
        )
        components["storage"] = storage

        # 3. Initialize ManagementTools (MCP tools for collection management)
        logger.info("Initializing ManagementTools")
        management = ManagementTools(storage=storage)
        components["management_tools"] = management

        # 4. Initialize DocumentHasher for deduplication
        logger.info("Initializing DocumentHasher")
        hasher = DocumentHasher()
        components["hasher"] = hasher

        # 5. Initialize Indexer (orchestrates chunking, embedding, storage)
        logger.info(
            f"Initializing Indexer (chunk_size={args.chunk_size}, batch_size={args.batch_size}, max_tokens={args.max_tokens})"
        )
        indexer = Indexer(
            embedder=embedder,
            storage=storage,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
        components["indexer"] = indexer

        # 6. Initialize BatchProcessor (directory indexing with deduplication)
        logger.info("Initializing BatchProcessor")
        batch_processor = BatchProcessor(
            indexer=indexer,
            hasher=hasher,
            storage=storage,
        )
        components["batch_processor"] = batch_processor

        # 7. Initialize IndexingTools (MCP tools for indexing)
        logger.info("Initializing IndexingTools")
        indexing = IndexingTools(
            batch_processor=batch_processor,
            indexer=indexer,
        )
        components["indexing_tools"] = indexing

        # 8. Initialize QueryTools (MCP tools for search)
        logger.info("Initializing QueryTools")
        query = QueryTools(
            embedder=embedder,
            storage=storage,
        )
        components["query_tools"] = query

        logger.info("All components initialized successfully")
        return components

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise


# MCP Tool Handlers - Index File
@mcp.tool()
async def index_file(file_path: str) -> dict[str, Any]:
    """Index a single document file.

    Args:
        file_path: Absolute or relative path to the file to index

    Returns:
        Result dictionary with success status, data, and optional error message
    """
    if indexing_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await indexing_tools.index_file(file_path)


# MCP Tool Handlers - Index Directory
@mcp.tool()
async def index_directory(dir_path: str, recursive: bool = True) -> dict[str, Any]:
    """Index all supported files in a directory.

    Args:
        dir_path: Absolute or relative path to the directory to index
        recursive: Whether to recursively index subdirectories (default: True)

    Returns:
        Result dictionary with statistics and indexed file list
    """
    if indexing_tools is None:
        return {"success": False, "error": "Server not initialized"}

    return await indexing_tools.index_directory(dir_path, recursive)


# MCP Tool Handlers - Search
@mcp.tool()
async def search(query: str, n_results: int = 5) -> dict[str, Any]:
    """Perform semantic search for a query.

    Args:
        query: Search query string
        n_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with search results, total count, and token usage
    """
    if query_tools is None:
        return {"error": "Server not initialized"}

    try:
        return await query_tools.search(query, n_results)
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return {"error": str(e)}


# MCP Tool Handlers - Search with Filters
@mcp.tool()
async def search_with_filters(
    query: str, filters: dict[str, Any], n_results: int = 5
) -> dict[str, Any]:
    """Perform semantic search with metadata filtering.

    Args:
        query: Search query string
        filters: Metadata filters (e.g., {"source_file": "readme.md"})
        n_results: Maximum number of results to return (default: 5)

    Returns:
        Dictionary with filtered search results
    """
    if query_tools is None:
        return {"error": "Server not initialized"}

    try:
        return await query_tools.search_with_filters(query, filters, n_results)
    except Exception as e:
        logger.error(f"Filtered search failed: {e}", exc_info=True)
        return {"error": str(e)}


# MCP Tool Handlers - Search with Budget
@mcp.tool()
async def search_with_budget(query: str, max_tokens: int) -> dict[str, Any]:
    """Search and return results within a token budget.

    Args:
        query: Search query string
        max_tokens: Maximum total tokens allowed in results

    Returns:
        Dictionary with results that fit within the token budget
    """
    if query_tools is None:
        return {"error": "Server not initialized"}

    try:
        return await query_tools.search_with_budget(query, max_tokens)
    except Exception as e:
        logger.error(f"Budget search failed: {e}", exc_info=True)
        return {"error": str(e)}


# MCP Tool Handlers - Delete by IDs
@mcp.tool()
async def delete_chunks(ids: list[str]) -> dict[str, Any]:
    """Delete specific chunks by their IDs.

    Args:
        ids: List of chunk IDs to delete

    Returns:
        Result dictionary with deleted count and IDs
    """
    if management_tools is None:
        return {"error": "Server not initialized"}

    return await management_tools.delete_by_ids(ids)


# MCP Tool Handlers - Delete by File
@mcp.tool()
async def delete_file(source_file: str) -> dict[str, Any]:
    """Delete all chunks from a specific source file.

    Args:
        source_file: Source file path to delete chunks for

    Returns:
        Result dictionary with deleted count and source file
    """
    if management_tools is None:
        return {"error": "Server not initialized"}

    return await management_tools.delete_by_file(source_file)


# MCP Tool Handlers - Clear Index
@mcp.tool()
async def clear_index(confirm: bool) -> dict[str, Any]:
    """Delete all documents from the collection.

    Requires explicit confirmation to prevent accidental data loss.

    Args:
        confirm: Must be True to proceed with deletion

    Returns:
        Result dictionary with deleted count or error if not confirmed
    """
    if management_tools is None:
        return {"error": "Server not initialized"}

    return await management_tools.delete_all(confirm)


# MCP Tool Handlers - Get Index Stats
@mcp.tool()
async def get_index_stats() -> dict[str, Any]:
    """Get collection statistics.

    Returns:
        Result dictionary with total chunks, unique files, and source file list
    """
    if management_tools is None:
        return {"error": "Server not initialized"}

    return await management_tools.get_stats()


def handle_shutdown(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown.

    Logs shutdown signal and performs cleanup before exit.

    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    # Close resources if needed
    # Note: FastMCP handles cleanup automatically
    sys.exit(0)


def main() -> None:
    """Main entry point for MCP server.

    Workflow:
    1. Parse CLI arguments
    2. Setup logging
    3. Initialize components
    4. Register signal handlers
    5. Run MCP server with stdio transport

    Note:
        Uses stdio transport for MCP communication. All logging
        goes to stderr to avoid corrupting JSON-RPC messages on stdout.
    """
    global indexing_tools, management_tools, query_tools

    # Parse arguments
    args = parse_arguments()

    # Setup logging to stderr
    setup_logging(args.log_level)

    logger.info("Starting Vector MCP Server...")
    logger.info(
        f"Configuration: host={args.host}, model={args.model}, db_path={args.db_path}, max_tokens={args.max_tokens}"
    )

    try:
        # Initialize all components
        components = initialize_components(args)

        # Set global tool instances for handlers
        indexing_tools = components["indexing_tools"]
        management_tools = components["management_tools"]
        query_tools = components["query_tools"]

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        logger.info("MCP server ready, starting stdio transport...")

        # Run MCP server with stdio transport
        # This blocks until server shuts down
        # Note: mcp.run() is synchronous and manages its own event loop
        mcp.run(transport="stdio")

    except Exception as e:
        logger.error(f"Server failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
