# Documentation Re-Indexing Guide

This guide explains how to re-index the documentation directory after the repository was renamed from `claude_graph` to `docvec`.

## Overview

The repository was moved from:
- **Old Path**: `/Users/harrison/Github/claude_graph/`
- **New Path**: `/Users/harrison/Github/docvec/`

This change requires updating all file path references stored in the ChromaDB database to point to the new location.

## What Needs to Be Done

There are two primary tasks:

1. **Update Existing Paths**: Scan the ChromaDB database and update all `source_file` metadata fields from the old path to the new path
2. **Re-Index Documentation**: Scan the `/Users/harrison/Github/docvec/docs/` directory and index all documents with the new paths

## Prerequisites

Before running the re-indexing tools, ensure:

1. **Python 3.10+** is installed
2. **Dependencies are installed**:
   ```bash
   cd /Users/harrison/Github/docvec
   uv sync
   ```

3. **Ollama is running** (required for re-indexing phase):
   ```bash
   # In another terminal
   ollama serve

   # Ensure the embedding model is available
   ollama pull nomic-embed-text
   # or
   ollama pull mxbai-embed-large
   ```

## Usage

### Option 1: Full Re-Indexing (Recommended)

Run the complete diagnostics, path update, and re-indexing operation:

```bash
cd /Users/harrison/Github/docvec
python diagnose_and_reindex.py
```

This script will:
1. **Diagnose** the current state of the database
2. **Update** all old paths to new paths
3. **Re-index** the docs directory with the new paths

### Option 2: Diagnostics Only

If you want to inspect the database without making changes:

```bash
cd /Users/harrison/Github/docvec
python diagnose_and_reindex.py --skip-reindex
```

This will:
1. Show current database statistics
2. List all indexed source files
3. Identify which files have old vs. new paths
4. Update paths (but skip re-indexing)

### Option 3: Manual Two-Step Process

If you prefer more control, use the individual scripts:

```bash
# Step 1: Update paths in existing database
cd /Users/harrison/Github/docvec
python update_paths_and_reindex.py
```

## Expected Output

### Diagnostics Phase

```
======================================================================
PHASE 1: DATABASE DIAGNOSTICS
======================================================================

Total chunks in database: 45
Scanning 45 chunks for source file paths...

Path Analysis:
  Documents with old paths: 15
  Documents with new paths: 30
  Documents with unknown paths: 0

  Sample old paths (first 3):
    - /Users/harrison/Github/claude_graph/docs/API.md
    - /Users/harrison/Github/claude_graph/docs/ARCHITECTURE.md
    - /Users/harrison/Github/claude_graph/docs/DEPLOYMENT.md

  Sample new paths (first 3):
    - /Users/harrison/Github/docvec/docs/API.md
    - /Users/harrison/Github/docvec/docs/ARCHITECTURE.md
```

### Path Update Phase

```
======================================================================
PHASE 2: UPDATE OLD PATHS TO NEW PATHS
======================================================================

Updating paths in 45 chunks...
  ... updated 10 paths so far

Path Update Complete:
  Total documents scanned: 45
  Paths updated: 15

  Sample updated paths (first 3):
    FROM: /Users/harrison/Github/claude_graph/docs/API.md
    TO:   /Users/harrison/Github/docvec/docs/API.md
```

### Re-Indexing Phase

```
======================================================================
PHASE 3: RE-INDEX DOCUMENTATION DIRECTORY
======================================================================

Initializing OllamaClient...
Ollama connection successful

Indexing directory: /Users/harrison/Github/docvec/docs

Re-Indexing Complete:
  New documents indexed: 4
  Duplicates skipped: 5
  Total chunks created: 45
  Files indexed: 5

  Indexed files:
    - /Users/harrison/Github/docvec/docs/API.md
    - /Users/harrison/Github/docvec/docs/ARCHITECTURE.md
    - /Users/harrison/Github/docvec/docs/DEPLOYMENT.md
    - /Users/harrison/Github/docvec/docs/plans/IMPLEMENTATION_SUMMARY.md
    - /Users/harrison/Github/docvec/docs/plans/vector-db-mcp-server.yaml
```

### Final Summary

```
======================================================================
COMPREHENSIVE SUMMARY
======================================================================

1. CURRENT DATABASE STATE:
   Total chunks: 45
   With old paths: 0
   With new paths: 45
   With unknown paths: 0

2. PATH UPDATE OPERATION:
   Status: SUCCESS
   Documents scanned: 45
   Paths changed: 15

3. RE-INDEXING OPERATION:
   Status: SUCCESS
   New documents: 4
   Duplicates skipped: 5
   Total chunks: 45
   Files indexed: 5

======================================================================
```

## Files Being Re-Indexed

The documentation directory contains the following files:

```
/Users/harrison/Github/docvec/docs/
├── API.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
└── plans/
    ├── IMPLEMENTATION_SUMMARY.md
    └── vector-db-mcp-server.yaml
```

## Troubleshooting

### Issue: "Ollama server not available"

**Solution**: Start Ollama in another terminal:
```bash
ollama serve
```

Then ensure the model is pulled:
```bash
ollama pull nomic-embed-text
```

### Issue: "Path does not exist"

**Solution**: Ensure you're running the script from the docvec repository:
```bash
cd /Users/harrison/Github/docvec
```

### Issue: "ChromaDB not initialized"

**Solution**: The database will be created automatically. If there are permission issues, ensure the directory is writable:
```bash
ls -la chroma_db/
```

### Issue: "Missing dependencies"

**Solution**: Reinstall the project dependencies:
```bash
uv sync
```

## Verification

After re-indexing, verify the operation was successful:

```bash
# Run diagnostics again to confirm no old paths remain
python diagnose_and_reindex.py --skip-reindex

# Check ChromaDB directly
python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from docvec.storage.chroma_store import ChromaStore

store = ChromaStore(Path('chroma_db'))
print(f'Total chunks: {store.count()}')

results = store._collection.get(include=['metadatas'])
for meta in results['metadatas'][:5]:
    print(f'  - {meta.get(\"source_file\", \"unknown\")}')
"
```

## Database Details

- **Database Location**: `/Users/harrison/Github/docvec/chroma_db/`
- **Database Type**: ChromaDB (SQLite-based)
- **Collection Name**: `documents`
- **Embedding Model**: `nomic-embed-text` (or `mxbai-embed-large`)

## Advanced: Manual Database Inspection

If you need to inspect the database directly:

```python
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from docvec.storage.chroma_store import ChromaStore

# Initialize storage
store = ChromaStore(Path('chroma_db'))

# Get all documents
all_results = store._collection.get(include=['metadatas', 'documents'])

# Print metadata for all documents
for i, meta in enumerate(all_results['metadatas']):
    print(f"{i}: {meta.get('source_file', 'unknown')}")
    print(f"   Chunks: {meta.get('chunk_index', 'unknown')}")
    print(f"   Hash: {meta.get('doc_hash', 'unknown')}")
```

## Glossary

- **Chunk**: A segment of a document created by the chunking strategy
- **Embedding**: A vector representation of text for semantic search
- **Deduplication**: Skipping files that have already been indexed (detected via SHA-256 hash)
- **Metadata**: Information about a chunk (source file, chunk index, hash, etc.)
- **ChromaDB**: Vector database used for semantic search and storage

## Support

For issues or questions about the indexing process, refer to:
- Main CLAUDE.md documentation
- Architecture documentation: `/Users/harrison/Github/docvec/docs/ARCHITECTURE.md`
- API documentation: `/Users/harrison/Github/docvec/docs/API.md`
