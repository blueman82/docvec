#!/usr/bin/env python3
"""Script to re-index the documentation directory using docvec MCP server.

This script initializes the docvec components and re-indexes the docs directory
to update all file paths from the old repository path to the new one.

Usage:
    python reindex_docs.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docvec.deduplication.hasher import DocumentHasher
from docvec.embedding.ollama_client import OllamaClient
from docvec.indexing.batch_processor import BatchProcessor
from docvec.indexing.indexer import Indexer
from docvec.mcp_tools.indexing_tools import IndexingTools
from docvec.storage.chroma_store import ChromaStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def reindex_docs():
    """Re-index the documentation directory."""
    logger.info("Starting documentation re-indexing process...")

    try:
        # Initialize components
        logger.info("Initializing components...")

        # OllamaClient for embeddings
        embedder = OllamaClient(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
        )

        # Check Ollama availability
        if not embedder.health_check():
            logger.warning("Ollama health check failed. Ensure Ollama is running.")
            return {
                "success": False,
                "error": "Ollama server not available"
            }

        logger.info("Ollama connection successful")

        # ChromaDB storage
        db_path = Path(__file__).parent / "chroma_db"
        storage = ChromaStore(
            db_path=db_path,
            collection_name="documents",
        )
        logger.info(f"ChromaDB initialized at {db_path}")

        # DocumentHasher for deduplication
        hasher = DocumentHasher()

        # Indexer for document processing
        indexer = Indexer(
            embedder=embedder,
            storage=storage,
            chunk_size=256,
            batch_size=16,
        )

        # BatchProcessor for directory indexing
        batch_processor = BatchProcessor(
            indexer=indexer,
            hasher=hasher,
            storage=storage,
        )

        # IndexingTools for MCP interface
        indexing_tools = IndexingTools(
            batch_processor=batch_processor,
            indexer=indexer,
        )

        logger.info("All components initialized successfully")

        # Index the docs directory
        docs_path = Path(__file__).parent / "docs"
        logger.info(f"Starting re-indexing of {docs_path}...")

        result = await indexing_tools.index_directory(
            dir_path=str(docs_path),
            recursive=True
        )

        if result.get("success"):
            data = result.get("data", {})
            logger.info("Re-indexing completed successfully!")
            logger.info(f"  - New documents indexed: {data.get('new_documents', 0)}")
            logger.info(f"  - Duplicates skipped: {data.get('duplicates_skipped', 0)}")
            logger.info(f"  - Total chunks created: {data.get('total_chunks', 0)}")
            logger.info(f"  - Indexed files: {len(data.get('indexed_files', []))}")

            if data.get('errors'):
                logger.warning(f"  - Errors encountered: {len(data['errors'])}")
                for error in data['errors']:
                    logger.warning(f"    - {error['file']}: {error['error']}")

            indexed_files = data.get('indexed_files', [])
            if indexed_files:
                logger.info("\nIndexed files:")
                for file in sorted(indexed_files):
                    logger.info(f"  - {file}")

            return result
        else:
            logger.error(f"Re-indexing failed: {result.get('error', 'Unknown error')}")
            return result

    except Exception as e:
        logger.error(f"Error during re-indexing: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


async def main():
    """Main entry point."""
    result = await reindex_docs()

    # Print summary
    print("\n" + "="*70)
    print("RE-INDEXING SUMMARY")
    print("="*70)

    if result.get("success"):
        data = result.get("data", {})
        print(f"Status: SUCCESS")
        print(f"New documents indexed: {data.get('new_documents', 0)}")
        print(f"Duplicates skipped: {data.get('duplicates_skipped', 0)}")
        print(f"Total chunks created: {data.get('total_chunks', 0)}")
        print(f"Files processed: {len(data.get('indexed_files', []))}")

        if data.get('errors'):
            print(f"Errors: {len(data['errors'])}")
    else:
        print(f"Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("="*70)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    asyncio.run(main())
