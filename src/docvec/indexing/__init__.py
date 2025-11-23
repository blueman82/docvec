"""Document indexing module.

Orchestrates document chunking, embedding, and storage.
"""

from vector_mcp.indexing.indexer import Indexer, IndexingError

__all__ = ["Indexer", "IndexingError"]
