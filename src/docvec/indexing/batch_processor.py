"""Batch processing with hash-based deduplication.

This module provides batch indexing capabilities with automatic deduplication
to prevent re-indexing identical documents. Supports directory scanning and
detailed progress reporting.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docvec.chunking.base import Chunk
from docvec.deduplication.hasher import DocumentHasher
from docvec.indexing.indexer import Indexer, IndexingError
from docvec.storage.chroma_store import ChromaStore, StorageError

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Results from batch processing operation.

    Attributes:
        new_documents: Number of newly indexed documents
        duplicates_skipped: Number of duplicate documents skipped
        errors: List of (file_path, error_message) tuples for failed files
        chunk_ids: Dictionary mapping file path to list of chunk IDs
    """

    new_documents: int = 0
    duplicates_skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    chunk_ids: dict[str, list[str]] = field(default_factory=dict)


class BatchProcessor:
    """Batch document processor with hash-based deduplication.

    Scans directories for files, checks for duplicates using
    document hashes, and indexes only new files. Provides detailed
    statistics and error reporting.

    The processor is extension-agnostic: any text-based file can be indexed.
    Specialized chunkers are used for known formats (.md, .pdf, .py),
    while unknown extensions fall back to a generic text chunker.

    Args:
        indexer: Indexer instance for document processing
        hasher: DocumentHasher for computing file hashes
        storage: ChromaStore for deduplication checking

    Attributes:
        indexer: Document indexer
        hasher: Document hasher
        storage: Vector storage

    Example:
        >>> processor = BatchProcessor(indexer, hasher, storage)
        >>> result = processor.process_directory(Path("docs"))
        >>> print(f"Indexed {result.new_documents} new documents")
        >>> print(f"Skipped {result.duplicates_skipped} duplicates")
    """

    def __init__(
        self,
        indexer: Indexer,
        hasher: DocumentHasher,
        storage: ChromaStore,
    ):
        """Initialize batch processor with dependencies.

        Args:
            indexer: Indexer for document processing
            hasher: DocumentHasher for computing hashes
            storage: ChromaStore for deduplication checking
        """
        self.indexer = indexer
        self.hasher = hasher
        self.storage = storage

    def process_directory(self, dir_path: Path, recursive: bool = True) -> BatchResult:
        """Process all supported files in a directory.

        Scans directory for supported file types, checks for duplicates,
        and indexes only new files. Supports both recursive and
        single-level scanning.

        Args:
            dir_path: Directory path to process
            recursive: If True, scan subdirectories recursively (default: True)

        Returns:
            BatchResult with processing statistics

        Raises:
            FileNotFoundError: If directory does not exist
            NotADirectoryError: If path is not a directory

        Example:
            >>> result = processor.process_directory(Path("docs"), recursive=True)
            >>> if result.errors:
            ...     print(f"Errors: {len(result.errors)}")
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        logger.info(f"Processing directory: {dir_path} (recursive={recursive})")

        # Find all supported files
        file_paths = self._find_files(dir_path, recursive)

        logger.info(f"Found {len(file_paths)} supported files")

        # Process files
        result = self.process_files([Path(f) for f in file_paths])

        logger.info(
            f"Batch processing complete: {result.new_documents} new, "
            f"{result.duplicates_skipped} duplicates, {len(result.errors)} errors"
        )

        return result

    def process_files(self, file_paths: list[Path]) -> BatchResult:
        """Process a list of files with deduplication and cross-file batching.

        Uses an accumulator pattern to batch chunks from multiple files
        before embedding, reducing MLX/GPU round-trips significantly.

        Args:
            file_paths: List of file paths to process

        Returns:
            BatchResult with detailed statistics

        Example:
            >>> files = [Path("a.txt"), Path("b.md"), Path("c.py")]
            >>> result = processor.process_files(files)
            >>> result.new_documents
            3
        """
        result = BatchResult()

        # Accumulator for cross-file batching
        chunk_accumulator: list[Chunk] = []
        # Track which file each chunk came from (for result reporting)
        chunk_file_map: list[str] = []
        # Track files that contributed chunks (for new_documents count)
        files_with_chunks: set[str] = set()

        # Use indexer's batch_size for accumulator threshold
        batch_threshold = self.indexer.batch_size

        def flush_accumulator() -> None:
            """Embed and store accumulated chunks."""
            nonlocal chunk_accumulator, chunk_file_map

            if not chunk_accumulator:
                return

            logger.info(f"Flushing batch of {len(chunk_accumulator)} chunks")

            try:
                chunk_ids = self.indexer.embed_and_store_chunks(chunk_accumulator)

                # Map chunk IDs back to source files
                for chunk_id, source_file in zip(chunk_ids, chunk_file_map):
                    if source_file not in result.chunk_ids:
                        result.chunk_ids[source_file] = []
                    result.chunk_ids[source_file].append(chunk_id)

            except IndexingError as e:
                logger.error(f"Failed to embed/store batch: {e}")
                # Record error for all files in this batch
                for source_file in set(chunk_file_map):
                    if source_file not in [err[0] for err in result.errors]:
                        result.errors.append((source_file, str(e)))

            # Clear accumulator
            chunk_accumulator = []
            chunk_file_map = []

        for file_path in file_paths:
            try:
                # Hash the file
                file_hash = self.hasher.hash_document(file_path)

                # Check if duplicate
                if self._is_duplicate(file_path, file_hash):
                    logger.info(f"Skipping duplicate: {file_path}")
                    result.duplicates_skipped += 1
                    continue

                # Chunk the file (without embedding)
                chunks = self.indexer.chunk_file(file_path)

                if not chunks:
                    logger.warning(f"No chunks generated from {file_path}")
                    continue

                # Add chunks to accumulator
                for chunk in chunks:
                    chunk_accumulator.append(chunk)
                    chunk_file_map.append(str(file_path))

                files_with_chunks.add(str(file_path))
                logger.info(f"Chunked {file_path.name}: {len(chunks)} chunks (accumulator: {len(chunk_accumulator)})")

                # Flush if accumulator is full
                if len(chunk_accumulator) >= batch_threshold:
                    flush_accumulator()

            except Exception as e:
                error_msg = str(e)
                result.errors.append((str(file_path), error_msg))
                logger.error(f"Failed to process {file_path}: {error_msg}")

        # Flush remaining chunks
        flush_accumulator()

        # Count new documents (files that had chunks successfully stored)
        for file_path_str in files_with_chunks:
            if file_path_str in result.chunk_ids and result.chunk_ids[file_path_str]:
                result.new_documents += 1
                logger.info(
                    f"Indexed {Path(file_path_str).name} with "
                    f"{len(result.chunk_ids[file_path_str])} chunks"
                )

        return result

    def _find_files(self, dir_path: Path, recursive: bool) -> list[str]:
        """Find all files in directory.

        Discovers all files for indexing. The system is extension-agnostic:
        specialized chunkers handle known formats (.md, .pdf, .py), while
        unknown extensions use a generic text chunker. Binary files that
        cannot be read as text will fail gracefully during indexing.

        Args:
            dir_path: Directory to scan
            recursive: If True, scan recursively

        Returns:
            List of file paths (sorted for deterministic processing)
        """
        if recursive:
            # Recursive search: all files in all subdirectories
            files = [p for p in dir_path.rglob("*") if p.is_file()]
        else:
            # Single-level search: files in this directory only
            files = [p for p in dir_path.iterdir() if p.is_file()]

        # Filter out hidden files (starting with .)
        files = [f for f in files if not f.name.startswith(".")]

        # Sort for deterministic processing order
        return sorted(str(p) for p in files)

    def _is_duplicate(self, file_path: Path, file_hash: str) -> bool:
        """Check if document is already indexed with same content.

        If file is indexed but content changed (different hash), deletes old
        chunks and returns False to trigger re-indexing.

        Args:
            file_path: Path to file to check
            file_hash: SHA-256 hash of file content

        Returns:
            True if already indexed with same content, False otherwise
        """
        try:
            # Check if this file path is already indexed
            existing = self.storage.get_by_source_file(str(file_path))
            if existing is not None:
                # File is indexed - check if content changed
                existing_hash = existing["metadatas"][0].get("doc_hash") if existing["metadatas"] else None
                if existing_hash == file_hash:
                    # Same content, skip
                    return True
                else:
                    # Content changed - delete old chunks, re-index
                    logger.info(f"Content changed, re-indexing: {file_path}")
                    self.storage.delete_by_source_file(str(file_path))
                    return False

            # Check if same content exists under different path (content duplicate)
            result = self.storage.get_by_hash(file_hash)
            return result is not None

        except StorageError as e:
            logger.warning(f"Failed to check duplicate for {file_path}: {e}")
            # On error, assume not duplicate to avoid skipping files
            return False

    def _process_single_file(
        self, file_path: Path, file_hash: str
    ) -> Optional[list[str]]:
        """Process a single file through the indexing pipeline.

        Indexes the file and returns chunk IDs. The file hash is
        passed through for metadata tracking.

        Args:
            file_path: Path to file to process
            file_hash: Pre-computed hash of file content

        Returns:
            List of chunk IDs if successful, None if no chunks generated

        Raises:
            IndexingError: If indexing fails
        """
        try:
            # Index the document
            chunk_ids = self.indexer.index_document(file_path)

            return chunk_ids if chunk_ids else None

        except (IndexingError, FileNotFoundError) as e:
            raise IndexingError(f"Failed to index {file_path}: {e}") from e
