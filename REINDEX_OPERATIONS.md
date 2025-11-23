# Re-Indexing Operations: Complete Technical Guide

## Repository Migration Overview

```
BEFORE: /Users/harrison/Github/claude_graph/
AFTER:  /Users/harrison/Github/docvec/

ChromaDB Metadata:
  source_file: "/Users/harrison/Github/claude_graph/docs/API.md"
                           ↓ (after update)
  source_file: "/Users/harrison/Github/docvec/docs/API.md"
```

## Tool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  diagnose_and_reindex.py                    │
│              (Primary Recommended Tool)                     │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  Diagnostician Class                        │
├─────────────────────────────────────────────────────────────┤
│  Methods:                                                   │
│  • diagnose()      → Inspect database state                │
│  • update_paths()  → Update old paths to new paths         │
│  • reindex_docs()  → Re-index documentation                │
│  • print_summary() → Print comprehensive report            │
└─────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────┬──────────────────────┬───────────────┐
│  ChromaStore         │  OllamaClient        │  Indexer      │
│  (Database Layer)    │  (Embeddings)        │  (Processing) │
└──────────────────────┴──────────────────────┴───────────────┘
         ↓
┌──────────────────────────────────────────────────────────────┐
│                     ChromaDB SQLite                          │
│              (Persistent Vector Database)                   │
└──────────────────────────────────────────────────────────────┘
```

## Phase 1: Diagnostics Workflow

```
START: diagnose()
  │
  ├─→ Connect to ChromaDB
  │     └─→ ChromaStore.__init__() → chroma_db/chroma.sqlite3
  │
  ├─→ Count total documents
  │     └─→ storage.count() → 45 chunks
  │
  ├─→ Retrieve all metadata
  │     └─→ collection.get(include=['metadatas'])
  │         Result: 45 metadata dictionaries
  │
  ├─→ Categorize by path type
  │     ├─→ Check: "/Users/harrison/Github/claude_graph" in source_file
  │     │   └─→ Old paths: 15 documents
  │     │
  │     ├─→ Check: "/Users/harrison/Github/docvec" in source_file
  │     │   └─→ New paths: 30 documents
  │     │
  │     └─→ Other/Unknown: 0 documents
  │
  ├─→ Collect examples and unique files
  │     ├─→ Sample old paths: [3 examples]
  │     ├─→ Sample new paths: [3 examples]
  │     └─→ All unique files: [sorted list]
  │
  └─→ RETURN diagnosis dict
      ├─ total_documents: 45
      ├─ documents_with_old_paths: 15
      ├─ documents_with_new_paths: 30
      ├─ documents_with_unknown_paths: 0
      ├─ old_path_examples: [...]
      ├─ new_path_examples: [...]
      └─ all_source_files: [...]
```

## Phase 2: Path Update Workflow

```
START: update_paths()
  │
  ├─→ Retrieve all documents from ChromaDB
  │     └─→ collection.get(include=['metadatas'])
  │         Result: {ids: [...], metadatas: [...]}
  │
  ├─→ Iterate through each metadata dictionary
  │     │
  │     For each metadata:
  │     │
  │     ├─→ Get source_file from metadata
  │     │     └─→ Example: "/Users/harrison/Github/claude_graph/docs/API.md"
  │     │
  │     ├─→ Check if old path exists in source_file
  │     │     └─→ if "/Users/harrison/Github/claude_graph" in source_file:
  │     │
  │     ├─→ Replace old with new
  │     │     └─→ new_source = source_file.replace(
  │     │           "/Users/harrison/Github/claude_graph",
  │     │           "/Users/harrison/Github/docvec"
  │     │         )
  │     │         Result: "/Users/harrison/Github/docvec/docs/API.md"
  │     │
  │     ├─→ Update metadata in ChromaDB
  │     │     └─→ collection.update(
  │     │           ids=[document_id],
  │     │           metadatas=[updated_metadata]
  │     │         )
  │     │
  │     └─→ Increment paths_changed counter
  │
  ├─→ Report progress (every 10 documents)
  │
  └─→ RETURN update result dict
      ├─ success: True
      ├─ documents_updated: 45
      ├─ paths_changed: 15
      └─ updated_files: [
         {old: "...", new: "..."},
         ...
      ]
```

## Phase 3: Re-Indexing Workflow

```
START: reindex_docs()
  │
  ├─→ Initialize OllamaClient
  │     ├─→ host: "http://localhost:11434"
  │     ├─→ model: "nomic-embed-text"
  │     └─→ timeout: 30 seconds
  │
  ├─→ Health check Ollama
  │     └─→ embedder.health_check()
  │         └─→ Returns True if responsive, False otherwise
  │
  ├─→ Initialize ChromaStore
  │     └─→ Connect to chroma_db/ with 'documents' collection
  │
  ├─→ Initialize supporting components
  │     ├─→ DocumentHasher (for deduplication)
  │     ├─→ Indexer (document processing)
  │     └─→ BatchProcessor (directory scanning)
  │
  ├─→ Scan documentation directory
  │     ├─→ Path: /Users/harrison/Github/docvec/docs
  │     ├─→ Recursive: True (includes subdirectories)
  │     │
  │     ├─→ Find files with supported extensions:
  │     │     ├─ .md  → MarkdownChunker
  │     │     ├─ .pdf → PDFChunker
  │     │     ├─ .py  → CodeChunker
  │     │     └─ .txt → TextChunker
  │     │
  │     └─→ Files found:
  │           ├─ API.md
  │           ├─ ARCHITECTURE.md
  │           ├─ DEPLOYMENT.md
  │           ├─ plans/IMPLEMENTATION_SUMMARY.md
  │           └─ plans/vector-db-mcp-server.yaml
  │
  ├─→ Process each file
  │     │
  │     For each file:
  │     │
  │     ├─→ Compute SHA-256 hash
  │     │     └─→ hash_document(file_path) → "abc123..."
  │     │
  │     ├─→ Check for duplicates
  │     │     └─→ storage.get_by_hash(hash)
  │     │         ├─→ If found: Skip (already indexed)
  │     │         └─→ If not found: Continue indexing
  │     │
  │     ├─→ Select appropriate chunker
  │     │     ├─ API.md → MarkdownChunker
  │     │     ├─ ARCHITECTURE.md → MarkdownChunker
  │     │     ├─ etc.
  │     │
  │     ├─→ Chunk the document
  │     │     └─→ chunker.chunk(content, source_file)
  │     │         Result: [Chunk, Chunk, ...]
  │     │
  │     ├─→ Generate embeddings (batched)
  │     │     └─→ embedder.embed_batch(chunk_texts)
  │     │         Result: [[0.1, 0.2, ...], [...], ...]
  │     │
  │     ├─→ Store in ChromaDB
  │     │     └─→ storage.add(embeddings, texts, metadatas)
  │     │         ├─ embeddings: list of vectors
  │     │         ├─ documents: list of chunk texts
  │     │         └─ metadatas: [{source_file, chunk_index, doc_hash}]
  │     │
  │     ├─→ Record chunk IDs
  │     │     └─→ result.chunk_ids[file_path] = [id1, id2, ...]
  │     │
  │     └─→ Update counters
  │           ├─ new_documents += 1
  │           └─ total_chunks += len(chunk_ids)
  │
  ├─→ Compile results
  │     └─→ BatchResult object with all statistics
  │
  └─→ RETURN indexing result dict
      ├─ success: True
      ├─ new_documents: 4
      ├─ duplicates_skipped: 5
      ├─ total_chunks: 45
      ├─ files_indexed: 5
      ├─ indexed_files: [list of file paths]
      └─ errors: []
```

## Data Flow: Path Update Example

```
Document in ChromaDB BEFORE:
  ID: "1732360800_0"
  Text: "API documentation..."
  Metadata: {
    "source_file": "/Users/harrison/Github/claude_graph/docs/API.md",
    "chunk_index": 0,
    "doc_hash": "abc123def456...",
    "header_path": "## Introduction"
  }

                        ↓ [STRING REPLACE]
        source_file.replace(
            "/Users/harrison/Github/claude_graph",
            "/Users/harrison/Github/docvec"
        )
                        ↓

Document in ChromaDB AFTER:
  ID: "1732360800_0"
  Text: "API documentation..."
  Metadata: {
    "source_file": "/Users/harrison/Github/docvec/docs/API.md",  ← UPDATED
    "chunk_index": 0,
    "doc_hash": "abc123def456...",
    "header_path": "## Introduction"
  }
```

## Data Flow: Re-Indexing Example

```
File: API.md
  │
  ├─→ Read Content (5000 characters)
  │
  ├─→ Compute Hash: "hash1234..."
  │
  ├─→ Check Duplicate: Not found in DB → Continue
  │
  ├─→ Chunk Content:
  │     ├─ Chunk 0: "# API\n\nDocumentation..." (256 tokens ≈ 1024 chars)
  │     ├─ Chunk 1: "## Endpoints\n\nGET /..." (256 tokens)
  │     └─ Chunk 2: "Response format..." (256 tokens)
  │
  ├─→ Generate Embeddings (batched):
  │     ├─ Input: ["# API\n...", "## Endpoints\n...", "Response..."]
  │     │
  │     └─→ Ollama Server
  │         └─→ Returns: [[0.1, 0.2, ...], [...], [...]]
  │
  ├─→ Store in ChromaDB:
  │     ID: "1732360801_0" (timestamp-based)
  │     Embedding: [0.1, 0.2, 0.3, ...]
  │     Document: "# API\n..."
  │     Metadata: {
  │       source_file: "/Users/harrison/Github/docvec/docs/API.md",
  │       chunk_index: 0,
  │       doc_hash: "hash1234...",
  │       header_path: "# API"
  │     }
  │
  └─→ Result:
      ├─ File: API.md
      ├─ Status: Indexed
      ├─ Chunks: 3
      └─ ChunkIDs: ["1732360801_0", "1732360801_1", "1732360801_2"]
```

## Query Process (After Re-Indexing)

```
User Query: "What is the API structure?"
  │
  ├─→ Embed query with OllamaClient
  │     └─→ query_embedding = [0.05, 0.15, 0.25, ...]
  │
  ├─→ Search in ChromaDB
  │     ├─→ Input: query_embedding, n_results=5
  │     ├─→ Algorithm: Cosine similarity search
  │     │
  │     └─→ Results (sorted by similarity):
  │         ├─ (Distance: 0.15) "# API\n..."
  │         ├─ (Distance: 0.22) "## Endpoints\n..."
  │         ├─ (Distance: 0.35) "Response format..."
  │         └─ ...
  │
  ├─→ Retrieve metadata for top results
  │     └─→ source_file: "/Users/harrison/Github/docvec/docs/API.md"
  │
  └─→ Return to user with path and content
```

## Error Handling

```
During Diagnostics:
  Error → Caught → Logged → Diagnosis continues with partial data

During Path Update:
  For each document:
    Try: update metadata
    Catch: Log error, continue to next document
  Final: Report success with error count (if any)

During Re-Indexing:
  For each file:
    Try: index file
    Catch:
      └─→ Add to errors list
      └─→ Continue with next file
  Final: Report successes and failures separately
```

## State Transitions

```
Database State Progression:

INITIAL STATE:
├─ Total chunks: 45
├─ Old paths: 15
├─ New paths: 30
└─ Unknown: 0

                    ↓ [after update_paths()]

AFTER PATH UPDATE:
├─ Total chunks: 45 (unchanged)
├─ Old paths: 0 (all updated)
├─ New paths: 45 (now includes updated ones)
└─ Unknown: 0

                    ↓ [after reindex_docs()]

AFTER RE-INDEXING:
├─ Total chunks: 50 (45 old + 5 new from re-index)
├─ Old paths: 0 (none)
├─ New paths: 50 (all with new path)
└─ Unknown: 0

NOTE: Exact counts may vary based on:
- File content changes
- Deduplication (skipped duplicates)
- Chunking strategy
```

## Performance Characteristics

```
OPERATION COMPLEXITY:

Diagnostics:
  └─→ O(n) where n = number of chunks
  └─→ 45 chunks ≈ < 1 second

Path Update:
  └─→ O(n) where n = number of chunks
  └─→ 45 chunks ≈ 1-5 seconds
  └─→ Database write operations are fast

Re-Indexing:
  └─→ O(f * c * e) where:
      f = number of files (5)
      c = average chunks per file (9)
      e = embedding generation (1-20s per batch)
  └─→ 5 files ≈ 30-120 seconds
  └─→ Bottleneck: Ollama embedding latency

TOTAL OPERATION:
  └─→ 30-125 seconds (1-2 minutes)
```

## Database Schema

```
ChromaDB Collection: "documents"

Metadata per Chunk:
{
  "id": "1732360800_0",           ← Auto-generated timestamp ID
  "source_file": "/Users/.../docs/API.md",  ← Path to source file (UPDATED)
  "chunk_index": 0,               ← Position in document
  "doc_hash": "abc123...",        ← SHA-256 hash for deduplication
  "header_path": "# API",         ← For markdown (optional)
  "page": 1                       ← For PDF (optional)
}

Vector Storage:
{
  "embedding": [0.1, 0.2, 0.3, ...],  ← 384-dim vector from nomic-embed-text
  "document": "# API\n...",            ← Chunk text content
  "metadata": {...}                    ← Metadata dict
}

Search Index:
  HNSW (Hierarchical Navigable Small World)
  Space: Cosine similarity
```

## Integration Points

```
After Re-Indexing, tools can use updated database:

MCP Server (docvec):
  ├─→ search(query) → Uses updated embeddings
  ├─→ index_file(path) → Adds new documents with new paths
  ├─→ index_directory(path) → Batch indexes with new paths
  └─→ search_with_filters(query, filters) → Filters by source_file (new path)

Python Applications:
  └─→ from docvec.storage import ChromaStore
      └─→ store = ChromaStore(Path("chroma_db"))
      └─→ results = store.search(embedding) → All paths are updated

Claude Claude via MCP:
  └─→ index_directory("/Users/harrison/Github/docvec/docs")
  └─→ search("API documentation")
  └─→ All results will use new paths
```

---

This technical guide provides a complete understanding of the re-indexing operation workflow, data transformations, and architecture.
