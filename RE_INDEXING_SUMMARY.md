# Documentation Re-Indexing Operation Summary

## Overview

This document summarizes the re-indexing operation created to update all file paths in the ChromaDB database after the repository was renamed from `claude_graph` to `docvec`.

**Repository Migration:**
- Old path: `/Users/harrison/Github/claude_graph/`
- New path: `/Users/harrison/Github/docvec/`

## Problem Statement

When the repository was renamed, the `source_file` metadata stored in ChromaDB still contained references to the old path (`/Users/harrison/Github/claude_graph/docs/...`). This needed to be updated to the new path (`/Users/harrison/Github/docvec/docs/...`).

## Solution

Three tools have been created to handle this operation:

### 1. **diagnose_and_reindex.py** (Primary Tool - Recommended)

**Purpose**: Complete diagnostic and re-indexing solution

**Location**: `/Users/harrison/Github/docvec/diagnose_and_reindex.py`

**Features**:
- Diagnoses current database state
- Shows which documents have old vs. new paths
- Lists all indexed source files
- Updates all old paths to new paths
- Re-indexes the documentation directory
- Provides comprehensive summary report

**Usage**:
```bash
cd /Users/harrison/Github/docvec

# Full operation (diagnose + update paths + re-index)
python diagnose_and_reindex.py

# Diagnose and update paths only (skip re-indexing)
python diagnose_and_reindex.py --skip-reindex
```

**Output Includes**:
- Current database statistics
- Breakdown of documents by path type
- Sample old and new paths
- Complete list of unique indexed files
- Operation results with counts
- Comprehensive final summary

### 2. **update_paths_and_reindex.py** (Alternative Tool)

**Purpose**: Direct path update and re-indexing script

**Location**: `/Users/harrison/Github/docvec/update_paths_and_reindex.py`

**Features**:
- Simpler two-phase approach
- Updates paths directly in ChromaDB
- Re-indexes docs directory with Ollama
- Detailed progress reporting

**Usage**:
```bash
cd /Users/harrison/Github/docvec
python update_paths_and_reindex.py
```

### 3. **reindex_docs.py** (Supplementary Tool)

**Purpose**: Re-indexing only (no path updates)

**Location**: `/Users/harrison/Github/docvec/reindex_docs.py`

**Features**:
- Focuses on re-indexing operation
- Useful if path updates are already done
- Shows indexed files and chunk counts

**Usage**:
```bash
cd /Users/harrison/Github/docvec
python reindex_docs.py
```

## Implementation Details

### Technology Stack

- **Database**: ChromaDB with SQLite backend (`/Users/harrison/Github/docvec/chroma_db/`)
- **Embeddings**: Ollama with `nomic-embed-text` model
- **Language**: Python 3.10+
- **Project**: docvec (MCP server for document indexing)

### Operation Flow

```
PHASE 1: DIAGNOSTICS
├─ Connect to ChromaDB
├─ Count total documents (chunks)
├─ Scan metadata for old/new paths
├─ Identify documents needing updates
└─ Report current state

PHASE 2: PATH UPDATES
├─ Retrieve all document metadata
├─ Find entries with old paths
├─ Replace old paths with new paths
├─ Update metadata in ChromaDB
└─ Report changes made

PHASE 3: RE-INDEXING
├─ Initialize OllamaClient
├─ Check Ollama health
├─ Initialize Indexer and BatchProcessor
├─ Scan /docs directory
├─ Index all supported file types (.md, .pdf, .txt, .py)
├─ Skip duplicates (hash-based deduplication)
└─ Report indexing results
```

### Key Components Used

1. **ChromaStore**: Vector database wrapper for persistent storage
2. **DocumentHasher**: SHA-256 hashing for deduplication
3. **OllamaClient**: Embedding generation client
4. **Indexer**: Document processing and chunking orchestrator
5. **BatchProcessor**: Directory scanning and batch indexing

### Data Being Updated

**Documents in Index**:
- `/Users/harrison/Github/docvec/docs/API.md`
- `/Users/harrison/Github/docvec/docs/ARCHITECTURE.md`
- `/Users/harrison/Github/docvec/docs/DEPLOYMENT.md`
- `/Users/harrison/Github/docvec/docs/plans/IMPLEMENTATION_SUMMARY.md`
- `/Users/harrison/Github/docvec/docs/plans/vector-db-mcp-server.yaml`

## Prerequisites

### Required Software
1. Python 3.10 or higher
2. Dependencies installed via `uv sync`
3. Ollama running locally (for re-indexing only)

### Ollama Setup

```bash
# Start Ollama server (in another terminal)
ollama serve

# Ensure embedding model is available
ollama pull nomic-embed-text
# Alternative models:
# ollama pull mxbai-embed-large
```

## Running the Re-Indexing

### Step 1: Ensure Environment is Ready

```bash
cd /Users/harrison/Github/docvec

# Verify Python
python3 --version  # Should be 3.10+

# Verify dependencies
python3 -c "import chromadb; import requests; print('Dependencies OK')"

# Start Ollama (in another terminal)
ollama serve
```

### Step 2: Run the Re-Indexing Tool

```bash
# Option A: Full operation (recommended)
python3 diagnose_and_reindex.py

# Option B: Diagnose only
python3 diagnose_and_reindex.py --skip-reindex
```

### Step 3: Verify Results

The tool will output:
- Database diagnostics
- Path update counts
- Indexed files list
- Total chunks created
- Any errors encountered

## Expected Results

### Database After Operation

```
Total chunks: ~45-50 (varies based on file content)
Documents with old paths: 0 (all updated)
Documents with new paths: 100%
Path example: /Users/harrison/Github/docvec/docs/API.md
```

### Index Coverage

All documentation files should be indexed:
- ✓ .md files (Markdown chunking with header awareness)
- ✓ .py files (Python AST-based chunking)
- ✓ .txt files (Paragraph-based chunking)
- ✓ .yaml files (Text-based chunking)

## Troubleshooting

### Common Issues

**Issue**: "Ollama server not available"
```
Solution: Ensure Ollama is running: ollama serve
```

**Issue**: "Module not found errors"
```
Solution: Run uv sync to install dependencies
```

**Issue**: "ChromaDB permission denied"
```
Solution: Check directory permissions: ls -la chroma_db/
```

**Issue**: "Timeout waiting for embeddings"
```
Solution: Check Ollama is responsive: curl http://localhost:11434/api/tags
```

## Verification

### Verify Paths Were Updated

```bash
python3 diagnose_and_reindex.py --skip-reindex

# Should show:
# - 0 documents with old paths
# - All documents with new paths
```

### Check Database Directly

```python
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from docvec.storage.chroma_store import ChromaStore

store = ChromaStore(Path('chroma_db'))
results = store._collection.get(include=['metadatas'])

for meta in results['metadatas']:
    print(meta.get('source_file'))
```

## Performance Notes

### Operation Timing

- **Diagnostics**: < 1 second
- **Path Updates**: 1-5 seconds (depends on document count)
- **Re-Indexing**: 30-120 seconds (depends on:
  - File sizes
  - Ollama embedding speed
  - Network latency

### Database Size

- **ChromaDB Location**: `./chroma_db/`
- **Typical Size**: 1-10 MB (includes all embeddings)
- **Growth**: Increases with more indexed documents

## Files Created

### Executable Scripts

1. **`diagnose_and_reindex.py`** (400+ lines)
   - Comprehensive diagnostic and update tool
   - Recommended for most users
   - Supports `--skip-reindex` flag

2. **`update_paths_and_reindex.py`** (250+ lines)
   - Simpler two-phase approach
   - Alternative implementation

3. **`reindex_docs.py`** (200+ lines)
   - Re-indexing focused
   - Can be used standalone

### Documentation

1. **`REINDEX_README.md`**
   - User-friendly guide
   - Troubleshooting section
   - Verification instructions

2. **`RE_INDEXING_SUMMARY.md`** (This file)
   - Technical overview
   - Implementation details
   - Operation workflow

## Design Decisions

### Why Three Tools?

1. **`diagnose_and_reindex.py`**: Complete solution for most users
2. **`update_paths_and_reindex.py`**: Simpler alternative
3. **`reindex_docs.py`**: Focused on re-indexing only

Users can choose the tool that best fits their needs.

### Database Update Strategy

The tools update paths by:
1. Reading all metadata from ChromaDB
2. Scanning for old path strings
3. Replacing with new path strings
4. Writing updated metadata back to storage

This approach:
- Preserves all other metadata
- Uses ChromaDB's update operation
- Maintains document relationships

### Deduplication Approach

The re-indexing uses hash-based deduplication:
- Files are hashed (SHA-256) before indexing
- Hashes are compared against existing documents
- Duplicate files are skipped
- Ensures only new/updated files are indexed

## Next Steps

### Recommended Workflow

1. **Review** this summary document
2. **Read** `/Users/harrison/Github/docvec/REINDEX_README.md`
3. **Ensure** Ollama is running
4. **Run** `python3 diagnose_and_reindex.py`
5. **Verify** results in console output
6. **Test** search queries against updated index

### Integration with MCP Server

After re-indexing, the docvec MCP server can use the updated index:

```bash
# Start the MCP server
python3 -m docvec

# The server will use the re-indexed documents with updated paths
```

## Support Resources

### Documentation Files

- **Architecture**: `/Users/harrison/Github/docvec/docs/ARCHITECTURE.md`
- **API Reference**: `/Users/harrison/Github/docvec/docs/API.md`
- **Deployment**: `/Users/harrison/Github/docvec/docs/DEPLOYMENT.md`
- **Project Guide**: `/Users/harrison/Github/docvec/CLAUDE.md`

### Code References

- **ChromaStore**: `src/docvec/storage/chroma_store.py`
- **Indexer**: `src/docvec/indexing/indexer.py`
- **BatchProcessor**: `src/docvec/indexing/batch_processor.py`
- **OllamaClient**: `src/docvec/embedding/ollama_client.py`

## Conclusion

The re-indexing operation provides a complete solution to update all file paths in the docvec documentation index from the old repository location to the new one. The `diagnose_and_reindex.py` tool is recommended for most users as it provides comprehensive diagnostics and clear reporting of all operations performed.

**Recommended Command**:
```bash
cd /Users/harrison/Github/docvec && python3 diagnose_and_reindex.py
```

This will ensure all documentation is properly indexed with the new repository paths.
