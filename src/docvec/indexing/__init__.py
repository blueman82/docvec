"""Document indexing module.

Orchestrates document chunking, embedding, and storage.
"""

from docvec.indexing.indexer import Indexer, IndexingError

__all__ = ["Indexer", "IndexingError"]
