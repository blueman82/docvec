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
import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from docvec.deduplication.hasher import DocumentHasher
from docvec.embedding.ollama_client import OllamaClient
from docvec.indexing.batch_processor import BatchProcessor
from docvec.indexing.indexer import Indexer
from docvec.mcp_tools.indexing_tools import IndexingTools
from docvec.mcp_tools.query_tools import QueryTools
from docvec.storage.chroma_store import ChromaStore

# Initialize FastMCP server
mcp = FastMCP("vector-mcp")

# Global components (initialized in main)
indexing_tools: Optional[IndexingTools] = None
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

    # Ollama configuration
    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:11434",
        help="Ollama server host URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model name",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Ollama request timeout in seconds",
    )

    # ChromaDB configuration
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="ChromaDB persistent storage directory path",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="ChromaDB collection name",
    )

    # Indexing configuration
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Maximum tokens per chunk",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def initialize_components(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize all components in dependency order.

    Initialization order follows dependency graph:
    1. Configuration (parsed arguments)
    2. Services: OllamaClient, ChromaStore, DocumentHasher
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
    components = {}

    try:
        # 1. Initialize Ollama client (embedding service)
        logger.info(f"Initializing OllamaClient (host={args.host}, model={args.model})")
        embedder = OllamaClient(
            host=args.host,
            model=args.model,
            timeout=args.timeout,
        )

        # Health check
        if not embedder.health_check():
            logger.warning(
                f"Ollama health check failed. Model {args.model} may not be available."
            )

        components["embedder"] = embedder

        # 2. Initialize ChromaDB storage
        db_path = Path(args.db_path).resolve()
        logger.info(f"Initializing ChromaStore (path={db_path}, collection={args.collection})")
        storage = ChromaStore(
            db_path=db_path,
            collection_name=args.collection,
        )
        components["storage"] = storage

        # 3. Initialize DocumentHasher for deduplication
        logger.info("Initializing DocumentHasher")
        hasher = DocumentHasher()
        components["hasher"] = hasher

        # 4. Initialize Indexer (orchestrates chunking, embedding, storage)
        logger.info(f"Initializing Indexer (chunk_size={args.chunk_size}, batch_size={args.batch_size})")
        indexer = Indexer(
            embedder=embedder,
            storage=storage,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        components["indexer"] = indexer

        # 5. Initialize BatchProcessor (directory indexing with deduplication)
        logger.info("Initializing BatchProcessor")
        batch_processor = BatchProcessor(
            indexer=indexer,
            hasher=hasher,
            storage=storage,
        )
        components["batch_processor"] = batch_processor

        # 6. Initialize IndexingTools (MCP tools for indexing)
        logger.info("Initializing IndexingTools")
        indexing = IndexingTools(
            batch_processor=batch_processor,
            indexer=indexer,
        )
        components["indexing_tools"] = indexing

        # 7. Initialize QueryTools (MCP tools for search)
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
    global indexing_tools, query_tools

    # Parse arguments
    args = parse_arguments()

    # Setup logging to stderr
    setup_logging(args.log_level)

    logger.info("Starting Vector MCP Server...")
    logger.info(f"Configuration: host={args.host}, model={args.model}, db_path={args.db_path}")

    try:
        # Initialize all components
        components = initialize_components(args)

        # Set global tool instances for handlers
        indexing_tools = components["indexing_tools"]
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
