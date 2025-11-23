#!/usr/bin/env python3
"""Index all documentation files in docs/ and docs/examples"""
import asyncio
from pathlib import Path
from vector_mcp.embedding.ollama_client import OllamaClient
from vector_mcp.storage.chroma_store import ChromaStore
from vector_mcp.indexing.batch_processor import BatchProcessor
from vector_mcp.indexing.indexer import Indexer
from vector_mcp.deduplication.hasher import DocumentHasher

async def main():
    # Initialize components
    embedder = OllamaClient(host="http://localhost:11434", model="nomic-embed-text")
    storage = ChromaStore(db_path=Path("./chroma_db"), collection_name="documentation")
    hasher = DocumentHasher()
    indexer = Indexer(embedder=embedder, storage=storage, chunk_size=512, batch_size=32)
    batch_processor = BatchProcessor(indexer=indexer, hasher=hasher)
    
    # Index docs directory
    docs_path = Path("docs")
    print(f"Indexing {docs_path}...")
    result = batch_processor.process_directory(docs_path, recursive=True)
    
    print(f"\n✅ Indexing complete!")
    print(f"   New documents: {result.new_documents}")
    print(f"   Duplicates skipped: {result.duplicates_skipped}")
    print(f"   Total chunks: {sum(len(ids) for ids in result.chunk_ids.values())}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.errors:
        print("\n⚠️  Errors encountered:")
        for file, error in result.errors:
            print(f"   - {file}: {error}")

if __name__ == "__main__":
    asyncio.run(main())
