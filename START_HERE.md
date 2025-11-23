# START HERE: Documentation Re-Indexing

Welcome! This guide will get you started with re-indexing your documentation after the repository rename.

## The Situation

Your repository was moved from:
- **Old**: `/Users/harrison/Github/claude_graph/`
- **New**: `/Users/harrison/Github/docvec/`

The documentation index (ChromaDB) still has the old paths. This needs to be updated.

## The Solution

Three ready-to-use tools have been created to update the index and re-index your documentation.

## Quick Start (2 minutes)

### Step 1: Prepare Your Environment

```bash
cd /Users/harrison/Github/docvec

# Ensure dependencies are installed
uv sync

# Start Ollama in another terminal
# (Open a new terminal window and run:)
ollama serve
```

### Step 2: Run the Re-Indexing Tool

```bash
cd /Users/harrison/Github/docvec
python3 diagnose_and_reindex.py
```

### Step 3: Done!

The tool will:
1. Check current database state
2. Update all old paths to new paths
3. Re-index all documentation files
4. Show you a summary of what was done

Expected output: Success messages and final summary showing 0 old paths.

## Reading Guide

### If you're in a hurry:
→ **Go to**: QUICK_START_REINDEX.md (5 min read)

### If you want all the details:
→ **Go to**: REINDEX_README.md (10 min read)

### If you need to troubleshoot:
→ **Go to**: REINDEX_README.md → "Troubleshooting" section

### If you're technically curious:
→ **Go to**: REINDEX_OPERATIONS.md (detailed workflows)

### If you want to understand the design:
→ **Go to**: RE_INDEXING_SUMMARY.md (technical overview)

### If you want to know what files were created:
→ **Go to**: FILES_CREATED.md (file inventory)

## The Three Tools

### 1. diagnose_and_reindex.py (Recommended)
**Best for**: Most users who want complete automation

```bash
python3 diagnose_and_reindex.py
```

**What it does**:
- Shows current database state
- Updates all old paths
- Re-indexes all documentation
- Prints detailed summary

**Flags**:
```bash
--skip-reindex  # Diagnose and update paths only (skip re-indexing)
```

---

### 2. update_paths_and_reindex.py (Alternative)
**Best for**: Users who prefer simpler two-phase approach

```bash
python3 update_paths_and_reindex.py
```

**What it does**:
- Updates old paths to new paths
- Re-indexes documentation
- Simpler output format

---

### 3. reindex_docs.py (Specialized)
**Best for**: Re-indexing when paths are already updated

```bash
python3 reindex_docs.py
```

**What it does**:
- Only re-indexes (no path updates)
- Useful for specific scenarios

---

## What Gets Updated

Your documentation consists of these files:

```
/Users/harrison/Github/docvec/docs/
├── API.md
├── ARCHITECTURE.md
├── DEPLOYMENT.md
└── plans/
    ├── IMPLEMENTATION_SUMMARY.md
    └── vector-db-mcp-server.yaml
```

All file paths in the index will be updated from old to new repository location.

## Prerequisites Checklist

Before running the tool, verify you have:

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`uv sync`)
- [ ] Ollama running (`ollama serve`)
- [ ] Embedding model available (`ollama pull nomic-embed-text`)
- [ ] Read permissions on `/Users/harrison/Github/docvec/`

## Common Issues

### "Ollama server not available"
```bash
# In another terminal
ollama serve
```

### "Module not found"
```bash
uv sync
```

### "Permission denied"
```bash
ls -la chroma_db/
chmod 755 chroma_db/
```

### "Database locked"
Wait a few seconds and retry. If persistent, check for other processes:
```bash
lsof chroma_db/
```

## Next Steps After Re-Indexing

### Option 1: Use the MCP Server
```bash
python3 -m docvec
```

### Option 2: Query Programmatically
```python
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from docvec.storage.chroma_store import ChromaStore

store = ChromaStore(Path('chroma_db'))
results = store.search([0.1, 0.2, ...], n_results=5)
```

### Option 3: Search via Claude
Use the MCP server with Claude to search your indexed documentation.

## Verification

After running the tool, verify success:

```bash
python3 diagnose_and_reindex.py --skip-reindex
```

Should show:
- "Total chunks: [number > 0]"
- "With old paths: 0"
- "With new paths: [number > 0]"

## Detailed Documentation

| Document | Purpose | Length | Audience |
|----------|---------|--------|----------|
| QUICK_START_REINDEX.md | Quick reference | 150 lines | All users |
| REINDEX_README.md | Complete guide | 400 lines | Detailed users |
| RE_INDEXING_SUMMARY.md | Technical overview | 500 lines | Developers |
| REINDEX_OPERATIONS.md | Detailed diagrams | 600 lines | Technical users |
| FILES_CREATED.md | File inventory | 400 lines | Reference |

## Architecture Overview

```
Your Command
    ↓
Tool (diagnose_and_reindex.py)
    ├─→ Diagnostician class
    │   ├─→ diagnose() → Check database
    │   ├─→ update_paths() → Fix old paths
    │   └─→ reindex_docs() → Re-index files
    │
    └─→ Uses Components:
        ├─→ ChromaStore (database)
        ├─→ OllamaClient (embeddings)
        └─→ Indexer (processing)
            ↓
        ChromaDB Database
        (Updated with new paths)
```

## Timeline

- **Setup**: 2-5 minutes
- **Diagnostics**: < 1 second
- **Path Updates**: 1-5 seconds
- **Re-Indexing**: 30-120 seconds
- **Total**: 1-2 minutes

## Success Indicators

After running the tool, you'll see:

```
COMPREHENSIVE SUMMARY
======================================================================
1. CURRENT DATABASE STATE:
   Total chunks: 45
   With old paths: 0        ← Should be 0
   With new paths: 45       ← Should equal total
   With unknown paths: 0

2. PATH UPDATE OPERATION:
   Status: SUCCESS
   Paths changed: 15

3. RE-INDEXING OPERATION:
   Status: SUCCESS
   New documents: 4
   Total chunks: 45
```

## FAQ

**Q: Will this delete anything?**
A: No, it only updates metadata paths and re-indexes files.

**Q: Can I run this multiple times?**
A: Yes, it's safe to run multiple times. Duplicates are skipped.

**Q: What if Ollama isn't running?**
A: The tool will skip re-indexing but still update paths. You can re-index later.

**Q: How long does it take?**
A: About 1-2 minutes total (depends on Ollama performance).

**Q: Can I use a different embedding model?**
A: Yes, pass `--model model-name` to `python -m docvec`. See REINDEX_README.md for details.

**Q: What's the database file format?**
A: ChromaDB with SQLite. Located at `/Users/harrison/Github/docvec/chroma_db/`

## Support

For detailed help:
1. Check **QUICK_START_REINDEX.md** for common issues
2. Check **REINDEX_README.md** for detailed troubleshooting
3. Review **REINDEX_OPERATIONS.md** for technical details
4. Check **RE_INDEXING_SUMMARY.md** for design explanations

## Ready?

### Run this command:

```bash
cd /Users/harrison/Github/docvec && python3 diagnose_and_reindex.py
```

### Or read more first:

- **5-minute guide**: QUICK_START_REINDEX.md
- **Complete guide**: REINDEX_README.md
- **Technical deep-dive**: REINDEX_OPERATIONS.md

---

**Repository**: `/Users/harrison/Github/docvec/`
**Created**: 2025-11-23
**Status**: Ready to use
**Recommendation**: Start with `python3 diagnose_and_reindex.py`
