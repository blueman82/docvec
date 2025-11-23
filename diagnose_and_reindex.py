#!/usr/bin/env python3
"""Diagnostic tool to inspect database and perform path updates and re-indexing.

This script:
1. Diagnoses the current state of the ChromaDB database
2. Shows which paths are currently indexed
3. Updates old paths to new paths
4. Re-indexes documentation to ensure all files are current

Usage:
    python diagnose_and_reindex.py [--skip-reindex]

    Options:
        --skip-reindex   Skip the re-indexing phase (diagnose only)
"""

import sys
import argparse
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docvec.storage.chroma_store import ChromaStore
from docvec.deduplication.hasher import DocumentHasher
from docvec.embedding.ollama_client import OllamaClient
from docvec.indexing.indexer import Indexer
from docvec.indexing.batch_processor import BatchProcessor


class Diagnostician:
    """Diagnostic tool for ChromaDB inspection and updates."""

    OLD_REPO_PATH = "/Users/harrison/Github/claude_graph"
    NEW_REPO_PATH = "/Users/harrison/Github/docvec"

    def __init__(self):
        """Initialize diagnostician."""
        self.db_path = Path(__file__).parent / "chroma_db"
        self.storage = ChromaStore(db_path=self.db_path, collection_name="documents")

    def diagnose(self) -> dict[str, Any]:
        """Run diagnostic checks on the database."""
        print("\n" + "="*70)
        print("PHASE 1: DATABASE DIAGNOSTICS")
        print("="*70)

        diagnosis = {
            "total_documents": 0,
            "documents_with_old_paths": 0,
            "documents_with_new_paths": 0,
            "documents_with_unknown_paths": 0,
            "old_path_examples": [],
            "new_path_examples": [],
            "all_source_files": []
        }

        try:
            # Get all documents
            count = self.storage.count()
            print(f"\nTotal chunks in database: {count}")
            diagnosis["total_documents"] = count

            if count == 0:
                print("Database is empty - no documents to diagnose.")
                return diagnosis

            # Get all documents with metadata
            all_results = self.storage._collection.get(
                include=["metadatas"]
            )

            print(f"\nScanning {len(all_results['ids'])} chunks for source file paths...")

            source_files = set()
            old_path_count = 0
            new_path_count = 0
            unknown_count = 0

            for metadata in all_results["metadatas"]:
                source_file = metadata.get("source_file", "")

                if source_file:
                    source_files.add(source_file)

                    if self.OLD_REPO_PATH in source_file:
                        old_path_count += 1
                        if len(diagnosis["old_path_examples"]) < 3:
                            diagnosis["old_path_examples"].append(source_file)
                    elif self.NEW_REPO_PATH in source_file:
                        new_path_count += 1
                        if len(diagnosis["new_path_examples"]) < 3:
                            diagnosis["new_path_examples"].append(source_file)
                    else:
                        unknown_count += 1
                else:
                    unknown_count += 1

            diagnosis["documents_with_old_paths"] = old_path_count
            diagnosis["documents_with_new_paths"] = new_path_count
            diagnosis["documents_with_unknown_paths"] = unknown_count
            diagnosis["all_source_files"] = sorted(list(source_files))

            # Print diagnostics
            print(f"\nPath Analysis:")
            print(f"  Documents with old paths: {old_path_count}")
            print(f"  Documents with new paths: {new_path_count}")
            print(f"  Documents with unknown paths: {unknown_count}")

            if old_path_count > 0:
                print(f"\n  Sample old paths (first 3):")
                for path in diagnosis["old_path_examples"]:
                    print(f"    - {path}")

            if new_path_count > 0:
                print(f"\n  Sample new paths (first 3):")
                for path in diagnosis["new_path_examples"]:
                    print(f"    - {path}")

            # List all unique source files
            print(f"\n  Unique source files ({len(source_files)} total):")
            for file in sorted(source_files)[:20]:  # Show first 20
                marker = " [OLD]" if self.OLD_REPO_PATH in file else ""
                marker = marker or (" [NEW]" if self.NEW_REPO_PATH in file else "")
                print(f"    - {file}{marker}")

            if len(source_files) > 20:
                print(f"    ... and {len(source_files) - 20} more")

            return diagnosis

        except Exception as e:
            print(f"ERROR during diagnostics: {e}")
            import traceback
            traceback.print_exc()
            return diagnosis

    def update_paths(self) -> dict[str, Any]:
        """Update old paths to new paths in the database."""
        print("\n" + "="*70)
        print("PHASE 2: UPDATE OLD PATHS TO NEW PATHS")
        print("="*70)

        result = {
            "success": False,
            "documents_updated": 0,
            "paths_changed": 0,
            "updated_files": []
        }

        try:
            # Get all documents
            all_results = self.storage._collection.get(
                include=["metadatas"]
            )

            if not all_results["ids"]:
                print("\nNo documents to update.")
                result["success"] = True
                return result

            print(f"\nUpdating paths in {len(all_results['ids'])} chunks...")

            paths_changed = 0

            for i, metadata in enumerate(all_results["metadatas"]):
                old_source = metadata.get("source_file", "")

                if old_source and self.OLD_REPO_PATH in old_source:
                    new_source = old_source.replace(self.OLD_REPO_PATH, self.NEW_REPO_PATH)
                    metadata["source_file"] = new_source

                    # Update the metadata in the collection
                    self.storage._collection.update(
                        ids=[all_results["ids"][i]],
                        metadatas=[metadata]
                    )

                    paths_changed += 1

                    if len(result["updated_files"]) < 20:
                        result["updated_files"].append({
                            "old": old_source,
                            "new": new_source
                        })

                    if paths_changed % 10 == 0:
                        print(f"  ... updated {paths_changed} paths so far")

            result["success"] = True
            result["documents_updated"] = len(all_results["ids"])
            result["paths_changed"] = paths_changed

            print(f"\nPath Update Complete:")
            print(f"  Total documents scanned: {result['documents_updated']}")
            print(f"  Paths updated: {paths_changed}")

            if result["updated_files"]:
                print(f"\n  Sample updated paths (first {len(result['updated_files'])}):")
                for item in result["updated_files"]:
                    print(f"    FROM: {item['old']}")
                    print(f"    TO:   {item['new']}")

            return result

        except Exception as e:
            print(f"ERROR updating paths: {e}")
            import traceback
            traceback.print_exc()
            result["success"] = False
            result["error"] = str(e)
            return result

    def reindex_docs(self) -> dict[str, Any]:
        """Re-index the documentation directory."""
        print("\n" + "="*70)
        print("PHASE 3: RE-INDEX DOCUMENTATION DIRECTORY")
        print("="*70)

        result = {
            "success": False,
            "new_documents": 0,
            "duplicates_skipped": 0,
            "total_chunks": 0,
            "files_indexed": 0,
            "indexed_files": [],
            "errors": [],
            "note": None
        }

        try:
            # Initialize embedder
            print("\nInitializing OllamaClient...")
            embedder = OllamaClient(
                host="http://localhost:11434",
                model="nomic-embed-text",
                timeout=30,
            )

            if not embedder.health_check():
                print("WARNING: Ollama health check failed.")
                print("Ensure Ollama is running: ollama serve")
                result["note"] = "Ollama not available - re-indexing skipped"
                return result

            print("Ollama connection successful")

            # Initialize other components
            hasher = DocumentHasher()

            indexer = Indexer(
                embedder=embedder,
                storage=self.storage,
                chunk_size=256,
                batch_size=16,
            )

            batch_processor = BatchProcessor(
                indexer=indexer,
                hasher=hasher,
                storage=self.storage,
            )

            # Index the docs directory
            docs_path = Path(__file__).parent / "docs"
            print(f"\nIndexing directory: {docs_path}")

            bp_result = batch_processor.process_directory(docs_path, recursive=True)

            # Calculate total chunks
            total_chunks = sum(len(ids) for ids in bp_result.chunk_ids.values())

            result["success"] = True
            result["new_documents"] = bp_result.new_documents
            result["duplicates_skipped"] = bp_result.duplicates_skipped
            result["total_chunks"] = total_chunks
            result["files_indexed"] = len(bp_result.chunk_ids)
            result["indexed_files"] = list(bp_result.chunk_ids.keys())
            result["errors"] = bp_result.errors

            print(f"\nRe-Indexing Complete:")
            print(f"  New documents indexed: {bp_result.new_documents}")
            print(f"  Duplicates skipped: {bp_result.duplicates_skipped}")
            print(f"  Total chunks created: {total_chunks}")
            print(f"  Files indexed: {len(bp_result.chunk_ids)}")

            if bp_result.indexed_files:
                print(f"\n  Indexed files:")
                for file in sorted(bp_result.indexed_files):
                    print(f"    - {file}")

            if bp_result.errors:
                print(f"\n  Errors ({len(bp_result.errors)}):")
                for file, error in bp_result.errors:
                    print(f"    - {file}: {error}")

            return result

        except Exception as e:
            print(f"ERROR during re-indexing: {e}")
            import traceback
            traceback.print_exc()
            result["success"] = False
            result["error"] = str(e)
            return result

    def print_summary(self, diagnosis: dict, path_update: dict, reindex: dict) -> None:
        """Print comprehensive summary."""
        print("\n" + "="*70)
        print("COMPREHENSIVE SUMMARY")
        print("="*70)

        print("\n1. CURRENT DATABASE STATE:")
        print(f"   Total chunks: {diagnosis['total_documents']}")
        print(f"   With old paths: {diagnosis['documents_with_old_paths']}")
        print(f"   With new paths: {diagnosis['documents_with_new_paths']}")
        print(f"   With unknown paths: {diagnosis['documents_with_unknown_paths']}")

        print("\n2. PATH UPDATE OPERATION:")
        if path_update.get("success"):
            print(f"   Status: SUCCESS")
            print(f"   Documents scanned: {path_update['documents_updated']}")
            print(f"   Paths changed: {path_update['paths_changed']}")
        else:
            print(f"   Status: FAILED")
            print(f"   Error: {path_update.get('error', 'Unknown')}")

        print("\n3. RE-INDEXING OPERATION:")
        if reindex.get("success"):
            print(f"   Status: SUCCESS")
            print(f"   New documents: {reindex['new_documents']}")
            print(f"   Duplicates skipped: {reindex['duplicates_skipped']}")
            print(f"   Total chunks: {reindex['total_chunks']}")
            print(f"   Files indexed: {reindex['files_indexed']}")

            if reindex.get("errors"):
                print(f"   Errors: {len(reindex['errors'])}")
        else:
            error_msg = reindex.get('error', 'Unknown error')
            print(f"   Status: {'SKIPPED' if reindex.get('note') else 'FAILED'}")
            if reindex.get("note"):
                print(f"   Note: {reindex['note']}")
            else:
                print(f"   Error: {error_msg}")

        print("\n" + "="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose and update documentation database"
    )
    parser.add_argument(
        "--skip-reindex",
        action="store_true",
        help="Skip re-indexing phase (diagnostics and path updates only)"
    )
    args = parser.parse_args()

    print("Documentation Diagnostics and Update Tool")
    print("Repository: /Users/harrison/Github/docvec")

    diagnostician = Diagnostician()

    # Phase 1: Diagnose
    diagnosis = diagnostician.diagnose()

    # Phase 2: Update paths
    path_update = diagnostician.update_paths()

    # Phase 3: Re-index (optional)
    if args.skip_reindex:
        print("\n" + "="*70)
        print("PHASE 3: RE-INDEX - SKIPPED")
        print("="*70)
        print("(use without --skip-reindex flag to re-index)")
        reindex = {"success": True, "note": "Skipped by user"}
    else:
        reindex = diagnostician.reindex_docs()

    # Print summary
    diagnostician.print_summary(diagnosis, path_update, reindex)

    # Exit with appropriate code
    success = path_update.get("success", False)
    if not args.skip_reindex:
        success = success and reindex.get("success", False)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
