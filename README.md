# DocVec - Document Vector Database

A Model Context Protocol (MCP) server that provides semantic document indexing and retrieval using ChromaDB and local Ollama embeddings. Reduce token usage in Claude conversations by efficiently retrieving only relevant document chunks.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [Indexing Documents](#indexing-documents)
  - [Searching Documents](#searching-documents)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
  - [Ollama Connection Failed](#ollama-connection-failed)
  - [ChromaDB Permission Error](#chromadb-permission-error)
  - [PDF Extraction Failed](#pdf-extraction-failed)
  - [Memory Issues with Large Documents](#memory-issues-with-large-documents)
  - [Search Returns Irrelevant Results](#search-returns-irrelevant-results)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Code Quality](#code-quality)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Performance](#performance)
- [License](#license)
- [Contributing](#contributing)
- [Support](#support)

## Features

- **Multi-format Support**: Index markdown, PDF, text, and Python code files
- **Smart Chunking**:
  - Header-aware chunking for markdown documents
  - Page-aware chunking for PDFs
  - AST-based chunking for Python code
  - Paragraph-based chunking for plain text
- **Local Embeddings**: Privacy-first approach using Ollama (mxbai-embed-large)
- **Persistent Storage**: ChromaDB vector database with metadata filtering
- **Hash-based Deduplication**: Automatic detection and skipping of duplicate documents
- **Token-aware Retrieval**: Control result size to fit within token budgets
- **MCP Integration**: Seamless integration with Claude Code and other MCP clients

[↑ Back to top](#table-of-contents)

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Ollama installed and running
- 2GB+ free disk space for ChromaDB

### Installation

1. Install Ollama and pull the embedding model:
```bash
# Install Ollama (https://ollama.ai)
# macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the embedding model
ollama pull mxbai-embed-large
```

2. Install the MCP server:
```bash
# Clone the repository
git clone https://github.com/yourusername/docvec.git
cd docvec

# Install dependencies using uv
uv sync
```

3. Configure Claude Code to use the MCP server:

Add to `~/.claude/config/mcp.json`:
```json
{
  "mcpServers": {
    "docvec": {
      "command": "uv",
      "args": ["run", "python", "-m", "docvec"],
      "env": {
        "DOCVEC_DB_PATH": "/Users/yourusername/.docvec/chroma_db"
      }
    }
  }
}
```

4. Start using the server from Claude Code:
```
Ask Claude to index your documents:
"Index all markdown files in /path/to/docs"

Search your indexed documents:
"Search for information about authentication"
```

[↑ Back to top](#table-of-contents)

## Basic Usage

### Indexing Documents

Index a single file:
```python
# Via MCP tool from Claude
index_file(file_path="/path/to/document.md")
```

Index a directory:
```python
# Via MCP tool from Claude
index_directory(dir_path="/path/to/docs", recursive=true)
```

### Searching Documents

Basic semantic search:
```python
# Via MCP tool from Claude
search(query="How do I configure authentication?", n_results=5)
```

Search with metadata filters:
```python
# Via MCP tool from Claude
search_with_filters(
    query="API endpoints",
    filters={"source_file": "api.md"},
    n_results=10
)
```

Token-budget aware search:
```python
# Via MCP tool from Claude
search_with_budget(query="deployment guide", max_tokens=2000)
```

[↑ Back to top](#table-of-contents)

## Configuration

Configuration is done via CLI arguments (takes precedence) or environment variables with the `DOCVEC_` prefix:

| Environment Variable | CLI Argument | Default | Description |
|----------|----------|---------|-------------|
| `DOCVEC_DB_PATH` | `--db-path` | `./chroma_db` | ChromaDB storage location |
| `DOCVEC_HOST` | `--host` | `http://localhost:11434` | Ollama API endpoint |
| `DOCVEC_MODEL` | `--model` | `nomic-embed-text` | Embedding model name |
| `DOCVEC_TIMEOUT` | `--timeout` | `30` | Ollama request timeout in seconds |
| `DOCVEC_CHUNK_SIZE` | `--chunk-size` | `256` | Maximum tokens per chunk |
| `DOCVEC_BATCH_SIZE` | `--batch-size` | `16` | Batch size for embedding generation |
| `DOCVEC_COLLECTION` | `--collection` | `documents` | ChromaDB collection name |
| `DOCVEC_LOG_LEVEL` | `--log-level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

[↑ Back to top](#table-of-contents)

## Project Structure

```
docvec/
├── src/
│   └── docvec/
│       ├── config.py              # Configuration management
│       ├── token_counter.py       # Token counting with tiktoken
│       ├── chunking/
│       │   ├── base.py           # Abstract chunker interface
│       │   ├── markdown_chunker.py
│       │   ├── pdf_chunker.py
│       │   ├── text_chunker.py
│       │   └── code_chunker.py
│       ├── embedding/
│       │   └── ollama_client.py  # Ollama API client
│       ├── storage/
│       │   └── chroma_store.py   # ChromaDB wrapper
│       ├── deduplication/
│       │   └── hasher.py         # SHA-256 document hashing
│       ├── indexing/
│       │   ├── indexer.py        # Document indexing orchestrator
│       │   └── batch_processor.py
│       ├── mcp_tools/
│       │   ├── indexing_tools.py # MCP indexing tools
│       │   └── query_tools.py    # MCP query tools
│       ├── mcp_server.py         # MCP server implementation
│       └── __main__.py           # Entry point
├── tests/                        # Test suite
├── docs/                         # Documentation
├── pyproject.toml               # Project configuration
└── README.md                    # This file
```

[↑ Back to top](#table-of-contents)

## Troubleshooting

### Ollama Connection Failed

**Error**: `Failed to connect to Ollama at http://localhost:11434`

**Solution**:
1. Check if Ollama is running: `ollama list`
2. Start Ollama if needed: `ollama serve`
3. Verify the embedding model is available: `ollama pull mxbai-embed-large`

[↑ Back to top](#table-of-contents)

### ChromaDB Permission Error

**Error**: `Permission denied: ~/.docvec/chroma_db`

**Solution**:
1. Create the directory manually: `mkdir -p ~/.docvec/chroma_db`
2. Set correct permissions: `chmod 755 ~/.docvec`
3. Or specify a different path via `DOCVEC_DB_PATH`

[↑ Back to top](#table-of-contents)

### PDF Extraction Failed

**Error**: `Failed to extract text from PDF`

**Solution**:
1. Ensure the PDF contains selectable text (not just scanned images)
2. For scanned PDFs, use OCR preprocessing
3. Check PDF file integrity with `file document.pdf`

[↑ Back to top](#table-of-contents)

### Memory Issues with Large Documents

**Error**: `MemoryError during indexing`

**Solution**:
1. Reduce `--chunk-size` (e.g., from 256 to 128) or set `DOCVEC_CHUNK_SIZE` environment variable
2. Reduce `--batch-size` to process fewer embeddings at once
3. Process files individually instead of batch indexing
4. Increase system swap space

[↑ Back to top](#table-of-contents)

### Search Returns Irrelevant Results

**Issue**: Search results don't match expectations

**Solution**:
1. Verify documents were indexed: check ChromaDB directory size
2. Try more specific queries with domain terminology
3. Use metadata filters to narrow search scope
4. Increase `n_results` to see more candidates

[↑ Back to top](#table-of-contents)

## Development

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src/docvec --cov-report=html

# Run specific test file
python -m pytest tests/test_indexer.py -v

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

### Code Quality

```bash
# Format code
python -m black src/ tests/

# Lint code
python -m ruff check src/ tests/

# Type checking
python -m mypy src/
```

[↑ Back to top](#table-of-contents)

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

[↑ Back to top](#table-of-contents)

## API Reference

See [docs/API.md](docs/API.md) for complete MCP tools specification.

[↑ Back to top](#table-of-contents)

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment guides.

[↑ Back to top](#table-of-contents)

## Performance

Typical performance characteristics:

- **Indexing**: ~0.5-1.0 seconds per page of content
- **Query**: <2 seconds for semantic search with 5 results
- **Token Reduction**: 80-95% for large documents (only relevant chunks retrieved)
- **Storage**: ~10MB per 100 pages of indexed content

[↑ Back to top](#table-of-contents)

## License

MIT License - see LICENSE file for details

[↑ Back to top](#table-of-contents)

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

[↑ Back to top](#table-of-contents)

## Support

- Issues: https://github.com/yourusername/docvec/issues
- Discussions: https://github.com/yourusername/docvec/discussions
- Documentation: https://github.com/yourusername/docvec/wiki

[↑ Back to top](#table-of-contents)
