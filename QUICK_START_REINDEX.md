# Quick Start: Re-Index Documentation

## TL;DR

```bash
# 1. Navigate to project
cd /Users/harrison/Github/docvec

# 2. Make sure dependencies are installed
uv sync

# 3. Start Ollama (in another terminal)
ollama serve

# 4. Run re-indexing
python3 diagnose_and_reindex.py

# Done! All paths updated and docs re-indexed.
```

## What Happens

1. **Diagnostics**: Shows current state of database
2. **Path Updates**: Changes `/Users/harrison/Github/claude_graph` → `/Users/harrison/Github/docvec`
3. **Re-Indexing**: Indexes all documentation files with new paths

## Typical Output

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

...

======================================================================
PHASE 2: UPDATE OLD PATHS TO NEW PATHS
======================================================================
Updating paths in 45 chunks...

Path Update Complete:
  Total documents scanned: 45
  Paths updated: 15

...

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

...

======================================================================
COMPREHENSIVE SUMMARY
======================================================================
1. CURRENT DATABASE STATE:
   Total chunks: 45
   With old paths: 0
   With new paths: 45

2. PATH UPDATE OPERATION:
   Status: SUCCESS
   Documents scanned: 45
   Paths changed: 15

3. RE-INDEXING OPERATION:
   Status: SUCCESS
   New documents: 4
   Total chunks: 45
   Files indexed: 5
```

## Options

### Full Operation (Recommended)
```bash
python3 diagnose_and_reindex.py
```

### Diagnose Only (No Changes)
```bash
python3 diagnose_and_reindex.py --skip-reindex
```

### Alternative: Two-Step Process
```bash
python3 update_paths_and_reindex.py
```

## Troubleshooting

### Ollama Not Running?
```bash
# In another terminal
ollama serve
```

### Dependencies Missing?
```bash
uv sync
```

### Want to See What's in Database?
```bash
python3 diagnose_and_reindex.py --skip-reindex
```

## Files Indexed

```
/Users/harrison/Github/docvec/docs/
├── API.md                              ✓
├── ARCHITECTURE.md                     ✓
├── DEPLOYMENT.md                       ✓
└── plans/
    ├── IMPLEMENTATION_SUMMARY.md       ✓
    └── vector-db-mcp-server.yaml       ✓
```

## Success Criteria

After running, check:
1. ✓ "Total chunks:" > 0
2. ✓ "With old paths: 0" (in final summary)
3. ✓ "With new paths:" = total chunks
4. ✓ All 5 docs indexed

## Next: Use the Index

```bash
# Start the MCP server
python3 -m docvec

# Or query directly
python3 -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from docvec.storage.chroma_store import ChromaStore

store = ChromaStore(Path('chroma_db'))
print(f'Total chunks indexed: {store.count()}')
"
```

## More Info

- Detailed guide: `REINDEX_README.md`
- Technical details: `RE_INDEXING_SUMMARY.md`
- Project guide: `CLAUDE.md`
- Architecture: `docs/ARCHITECTURE.md`

## Timeline

- **Diagnostics**: < 1 second
- **Path Updates**: 1-5 seconds
- **Re-Indexing**: 30-120 seconds
- **Total**: 1-2 minutes

---

That's it! Run `python3 diagnose_and_reindex.py` and you're done.
