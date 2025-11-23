#!/usr/bin/env python3
"""Script to update old paths and re-index documentation.

This script:
1. Updates all existing paths in ChromaDB from /Users/harrison/Github/claude_graph to /Users/harrison/Github/docvec
2. Re-indexes the docs directory to ensure all files are indexed with correct paths

Usage:
    python update_paths_and_reindex.py
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docvec.storage.chroma_store import ChromaStore
from docvec.deduplication.hasher import DocumentHasher
from docvec.embedding.ollama_client import OllamaClient
from docvec.indexing.indexer import Indexer
from docvec.indexing.batch_processor import BatchProcessor


def update_paths_in_db() -> dict[str, Any]:
    """Update old repository paths to new paths in ChromaDB.

    Returns:
        Dictionary with update statistics
    """
    print("\n" + "="*70)
    print("PHASE 1: UPDATE OLD PATHS IN DATABASE")
    print("="*70)

    old_path = "/Users/harrison/Github/claude_graph"
    new_path = "/Users/harrison/Github/docvec"

    db_path = Path(__file__).parent / "chroma_db"
    storage = ChromaStore(db_path=db_path, collection_name="documents")

    try:
        # Get count before update
        count_before = storage.count()
        print(f"Total documents in database: {count_before}")

        if count_before == 0:
            print("Database is empty. No paths to update.")
            return {
                "success": True,
                "documents_updated": 0,
                "paths_changed": 0,
                "message": "Database was empty"
            }

        # Get all documents from collection
        all_results = storage._collection.get(include=["metadatas", "documents"])

        documents_updated = 0
        paths_changed = 0

        # Process all documents
        if all_results["ids"]:
            print(f"Scanning {len(all_results['ids'])} documents for old paths...")

            for i, metadata in enumerate(all_results["metadatas"]):
                old_source = metadata.get("source_file", "")

                # Check if this document has an old path
                if old_source and old_path in old_source:
                    new_source = old_source.replace(old_path, new_path)
                    metadata["source_file"] = new_source

                    # Update the metadata in the collection
                    storage._collection.update(
                        ids=[all_results["ids"][i]],
                        metadatas=[metadata]
                    )

                    paths_changed += 1
                    print(f"  Updated: {old_source}")
                    print(f"       to: {new_source}")

                documents_updated += 1
                if (documents_updated % 10) == 0:
                    print(f"  ... processed {documents_updated} documents")

        print(f"\nPath Update Summary:")
        print(f"  Total documents scanned: {documents_updated}")
        print(f"  Documents with old paths: {paths_changed}")
        print(f"  Total documents in DB: {storage.count()}")

        return {
            "success": True,
            "documents_updated": documents_updated,
            "paths_changed": paths_changed
        }

    except Exception as e:
        print(f"ERROR updating paths: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def reindex_docs() -> dict[str, Any]:
    """Re-index the documentation directory.

    Returns:
        Dictionary with indexing statistics
    """
    print("\n" + "="*70)
    print("PHASE 2: RE-INDEX DOCUMENTATION DIRECTORY")
    print("="*70)

    try:
        # Initialize components
        print("Initializing components...")

        # OllamaClient for embeddings
        embedder = OllamaClient(
            host="http://localhost:11434",
            model="nomic-embed-text",
            timeout=30,
        )

        # Check Ollama availability
        if not embedder.health_check():
            print("WARNING: Ollama health check failed.")
            print("Make sure Ollama is running: ollama serve")
            return {
                "success": False,
                "error": "Ollama server not available",
                "note": "Skipping re-indexing since embeddings cannot be generated"
            }

        print("Ollama connection successful")

        # ChromaDB storage
        db_path = Path(__file__).parent / "chroma_db"
        storage = ChromaStore(db_path=db_path, collection_name="documents")

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

        # Index the docs directory
        docs_path = Path(__file__).parent / "docs"
        print(f"\nIndexing directory: {docs_path}")

        # Use batch processor to index
        result = batch_processor.process_directory(docs_path, recursive=True)

        # Prepare result summary
        total_chunks = sum(len(ids) for ids in result.chunk_ids.values())

        print(f"\nIndexing Summary:")
        print(f"  New documents indexed: {result.new_documents}")
        print(f"  Duplicates skipped: {result.duplicates_skipped}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Files indexed: {len(result.chunk_ids)}")

        if result.indexed_files:
            print(f"\n  Indexed files:")
            for file in sorted(result.indexed_files):
                print(f"    - {file}")

        if result.errors:
            print(f"\n  Errors encountered: {len(result.errors)}")
            for file, error in result.errors:
                print(f"    - {file}: {error}")

        return {
            "success": True,
            "new_documents": result.new_documents,
            "duplicates_skipped": result.duplicates_skipped,
            "total_chunks": total_chunks,
            "files_indexed": len(result.chunk_ids),
            "indexed_files": list(result.chunk_ids.keys()),
            "errors": result.errors
        }

    except Exception as e:
        print(f"ERROR during re-indexing: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }


def print_final_summary(path_update_result: dict, reindex_result: dict) -> None:
    """Print final summary of operations."""
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print("\nPhase 1 - Path Updates:")
    if path_update_result.get("success"):
        print(f"  Status: SUCCESS")
        print(f"  Documents scanned: {path_update_result.get('documents_updated', 0)}")
        print(f"  Paths updated: {path_update_result.get('paths_changed', 0)}")
    else:
        print(f"  Status: FAILED")
        print(f"  Error: {path_update_result.get('error', 'Unknown')}")

    print("\nPhase 2 - Re-Indexing:")
    if reindex_result.get("success"):
        print(f"  Status: SUCCESS")
        print(f"  New documents: {reindex_result.get('new_documents', 0)}")
        print(f"  Duplicates skipped: {reindex_result.get('duplicates_skipped', 0)}")
        print(f"  Total chunks: {reindex_result.get('total_chunks', 0)}")
        print(f"  Files processed: {reindex_result.get('files_indexed', 0)}")

        if reindex_result.get('errors'):
            print(f"  Errors: {len(reindex_result['errors'])}")
    else:
        error_msg = reindex_result.get('error', 'Unknown error')
        print(f"  Status: FAILED")
        print(f"  Error: {error_msg}")

        if reindex_result.get('note'):
            print(f"  Note: {reindex_result['note']}")

    print("\n" + "="*70)
    print("Operation completed!")
    print("="*70)


def main() -> None:
    """Main entry point."""
    print("Documentation Path Update and Re-Indexing Tool")
    print("Repository: /Users/harrison/Github/docvec")

    # Phase 1: Update paths
    path_update_result = update_paths_in_db()

    # Phase 2: Re-index docs
    reindex_result = reindex_docs()

    # Print final summary
    print_final_summary(path_update_result, reindex_result)

    # Exit with appropriate code
    success = path_update_result.get("success", False) and reindex_result.get("success", False)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
