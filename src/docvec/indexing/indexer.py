"""Document indexer orchestrating chunking, embedding, and storage.

This module coordinates the entire document indexing pipeline:
1. Auto-detect file format by extension
2. Load and chunk document using appropriate chunker
3. Validate chunks meet token constraints
4. Generate embeddings in batches
5. Store in ChromaDB with full metadata
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from vector_mcp.chunking.base import AbstractChunker, Chunk
from vector_mcp.chunking.code_chunker import CodeChunker
from vector_mcp.chunking.markdown_chunker import MarkdownChunker
from vector_mcp.chunking.pdf_chunker import PDFChunker
from vector_mcp.chunking.text_chunker import TextChunker
from vector_mcp.embedding.ollama_client import OllamaClient, EmbeddingError
from vector_mcp.storage.chroma_store import ChromaStore, StorageError

logger = logging.getLogger(__name__)


class IndexingError(Exception):
    """Custom exception for indexing-related errors."""

    pass


class Indexer:
    """Orchestrates document chunking, embedding, and storage.

    Auto-detects file types and coordinates the indexing pipeline from
    raw documents to embedded chunks in ChromaDB.

    Args:
        embedder: OllamaClient for generating embeddings
        storage: ChromaStore for persisting embeddings
        chunk_size: Maximum token count per chunk (default: 512)
        batch_size: Number of chunks to embed per batch (default: 32)

    Attributes:
        embedder: The embedding client
        storage: The vector storage
        chunk_size: Maximum tokens per chunk
        batch_size: Batch size for embedding
        _chunker_map: Mapping of file extensions to chunker classes
    """

    def __init__(
        self,
        embedder: OllamaClient,
        storage: ChromaStore,
        chunk_size: int = 256,
        batch_size: int = 16,
    ):
        """Initialize indexer with dependencies.

        Args:
            embedder: Ollama embedding client
            storage: ChromaDB storage instance
            chunk_size: Maximum tokens per chunk
            batch_size: Batch size for embedding calls

        Raises:
            ValueError: If chunk_size or batch_size are invalid
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.embedder = embedder
        self.storage = storage
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        # Map file extensions to chunker classes
        self._chunker_map: dict[str, type[AbstractChunker]] = {
            ".md": MarkdownChunker,
            ".pdf": PDFChunker,
            ".txt": TextChunker,
            ".py": CodeChunker,
        }

    def index_document(self, file_path: Path) -> list[str]:
        """Index a single document through the full pipeline.

        Workflow:
        1. Load file content
        2. Select appropriate chunker by extension
        3. Chunk document
        4. Validate chunks
        5. Generate embeddings in batches
        6. Store in ChromaDB with metadata

        Args:
            file_path: Path to document to index

        Returns:
            List of chunk IDs generated during storage

        Raises:
            IndexingError: If indexing fails at any stage
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise IndexingError(f"Path is not a file: {file_path}")

        logger.info(f"Indexing document: {file_path}")

        try:
            # Load file content
            content = self._load_file(file_path)

            # Select chunker and chunk document
            chunker = self._select_chunker(file_path)
            chunks = chunker.chunk(content, str(file_path))

            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")

            # Validate chunks
            valid_chunks = self._validate_chunks(chunks)

            if not valid_chunks:
                logger.warning(f"No valid chunks generated from {file_path}")
                return []

            logger.info(f"{len(valid_chunks)} chunks passed validation")

            # Generate embeddings
            embeddings = self._embed_chunks(valid_chunks)

            # Store chunks with embeddings
            chunk_ids = self._store_chunks(valid_chunks, embeddings)

            logger.info(f"Successfully indexed {file_path.name} with {len(chunk_ids)} chunks")
            return chunk_ids

        except (EmbeddingError, StorageError) as e:
            raise IndexingError(f"Failed to index {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error indexing {file_path}: {e}")
            raise IndexingError(f"Unexpected error indexing {file_path}: {e}") from e

    def index_batch(self, file_paths: list[Path]) -> dict[str, list[str]]:
        """Index multiple documents.

        Args:
            file_paths: List of paths to documents

        Returns:
            Dictionary mapping file path to list of chunk IDs.
            Failed files will have empty lists.

        Example:
            >>> indexer = Indexer(embedder, storage)
            >>> results = indexer.index_batch([Path("a.txt"), Path("b.py")])
            >>> results[str(Path("a.txt"))]
            ['1732360800_0', '1732360800_1']
        """
        results = {}

        for file_path in file_paths:
            try:
                chunk_ids = self.index_document(file_path)
                results[str(file_path)] = chunk_ids
            except (IndexingError, FileNotFoundError) as e:
                logger.error(f"Failed to index {file_path}: {e}")
                results[str(file_path)] = []

        return results

    def _load_file(self, file_path: Path) -> str:
        """Load file content.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            IndexingError: If file cannot be read
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                return file_path.read_text(encoding="latin-1")
            except Exception as e:
                raise IndexingError(f"Failed to read {file_path}: {e}") from e
        except Exception as e:
            raise IndexingError(f"Failed to read {file_path}: {e}") from e

    def _select_chunker(self, file_path: Path) -> AbstractChunker:
        """Select appropriate chunker based on file extension.

        Strategy:
        - .md -> MarkdownChunker
        - .pdf -> PDFChunker
        - .py -> CodeChunker
        - .txt -> TextChunker
        - default -> TextChunker

        Args:
            file_path: Path to file

        Returns:
            Instantiated chunker for the file type
        """
        suffix = file_path.suffix.lower()

        chunker_class = self._chunker_map.get(suffix, TextChunker)

        # Instantiate chunker with appropriate parameters
        # Use character-based chunk_size for most chunkers (4 chars per token approximation)
        char_chunk_size = self.chunk_size * 4

        if chunker_class == CodeChunker:
            # CodeChunker uses line-based chunk_size for fallback
            return CodeChunker(chunk_size=100)
        elif chunker_class == MarkdownChunker:
            # MarkdownChunker uses character-based chunk_size
            return MarkdownChunker(chunk_size=char_chunk_size, chunk_overlap=200)
        elif chunker_class == PDFChunker:
            # PDFChunker uses character-based chunk_size
            return PDFChunker(chunk_size=char_chunk_size, chunk_overlap=200)
        else:
            # TextChunker uses character-based chunk_size
            return TextChunker(chunk_size=char_chunk_size, chunk_overlap=200)

    def _validate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Validate chunks meet token constraints.

        Filters out chunks that exceed token limits or are invalid.

        Args:
            chunks: List of chunks to validate

        Returns:
            List of valid chunks
        """
        valid_chunks = []

        for chunk in chunks:
            # Check chunk has content
            if not chunk.content or not chunk.content.strip():
                logger.warning(f"Skipping empty chunk {chunk.chunk_index}")
                continue

            # Estimate token count (rough approximation: 4 chars per token)
            estimated_tokens = len(chunk.content) // 4

            if estimated_tokens > self.chunk_size:
                logger.warning(
                    f"Skipping oversized chunk {chunk.chunk_index} "
                    f"({estimated_tokens} tokens > {self.chunk_size} limit)"
                )
                continue

            valid_chunks.append(chunk)

        return valid_chunks

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Generate embeddings for chunks in batches.

        Batches chunks for efficient embedding generation.
        For example, 100 chunks with batch_size=32 results in 4 API calls.

        Args:
            chunks: List of chunks to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not chunks:
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks in batches of {self.batch_size}")

        try:
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]

            # Generate embeddings using batch processing
            embeddings = self.embedder.embed_batch(texts, batch_size=self.batch_size)

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings

        except EmbeddingError as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _store_chunks(
        self, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> list[str]:
        """Store chunks with embeddings in ChromaDB.

        Args:
            chunks: List of chunks
            embeddings: List of embedding vectors

        Returns:
            List of generated chunk IDs

        Raises:
            StorageError: If storage operation fails
            IndexingError: If chunks and embeddings length mismatch
        """
        if len(chunks) != len(embeddings):
            raise IndexingError(
                f"Chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings"
            )

        if not chunks:
            return []

        logger.info(f"Storing {len(chunks)} chunks in ChromaDB")

        try:
            # Prepare documents and metadata
            documents = [chunk.content for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                # Compute document hash for deduplication
                doc_hash = self._compute_hash(chunk.content)

                # Build metadata
                metadata = {
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "doc_hash": doc_hash,
                }

                # Merge with chunk-specific metadata
                metadata.update(chunk.metadata)

                metadatas.append(metadata)

            # Store in ChromaDB
            chunk_ids = self.storage.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(f"Successfully stored {len(chunk_ids)} chunks")
            return chunk_ids

        except StorageError as e:
            logger.error(f"Failed to store chunks: {e}")
            raise

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content for deduplication.

        Args:
            content: Text content to hash

        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
