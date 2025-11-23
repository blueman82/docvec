# Vector Database MCP Server

A Model Context Protocol (MCP) server that provides semantic document indexing and retrieval using ChromaDB and local Ollama embeddings. Reduce token usage in Claude conversations by efficiently retrieving only relevant document chunks.

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
git clone https://github.com/yourusername/vector-mcp-server.git
cd vector-mcp-server

# Install dependencies using uv
uv sync
```

3. Configure Claude Code to use the MCP server:

Add to `~/.claude/config/mcp.json`:
```json
{
  "mcpServers": {
    "vector-db": {
      "command": "uv",
      "args": ["run", "python", "-m", "vector_mcp"],
      "env": {
        "VECTOR_MCP_DB_PATH": "/Users/yourusername/.vector_mcp/chroma_db"
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

## Configuration

All configuration is done via environment variables with the `VECTOR_MCP_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_MCP_DB_PATH` | `~/.vector_mcp/chroma_db` | ChromaDB storage location |
| `VECTOR_MCP_OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `VECTOR_MCP_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model name |
| `VECTOR_MCP_CHUNK_SIZE` | `512` | Maximum tokens per chunk |
| `VECTOR_MCP_CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `VECTOR_MCP_MAX_RESULTS` | `5` | Default search result limit |
| `VECTOR_MCP_MAX_TOKENS` | `3000` | Default token budget for results |

## Project Structure

```
vector-mcp-server/
├── src/
│   └── vector_mcp/
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

## Troubleshooting

### Ollama Connection Failed

**Error**: `Failed to connect to Ollama at http://localhost:11434`

**Solution**:
1. Check if Ollama is running: `ollama list`
2. Start Ollama if needed: `ollama serve`
3. Verify the embedding model is available: `ollama pull mxbai-embed-large`

### ChromaDB Permission Error

**Error**: `Permission denied: ~/.vector_mcp/chroma_db`

**Solution**:
1. Create the directory manually: `mkdir -p ~/.vector_mcp/chroma_db`
2. Set correct permissions: `chmod 755 ~/.vector_mcp`
3. Or specify a different path via `VECTOR_MCP_DB_PATH`

### PDF Extraction Failed

**Error**: `Failed to extract text from PDF`

**Solution**:
1. Ensure the PDF contains selectable text (not just scanned images)
2. For scanned PDFs, use OCR preprocessing
3. Check PDF file integrity with `file document.pdf`

### Memory Issues with Large Documents

**Error**: `MemoryError during indexing`

**Solution**:
1. Reduce `VECTOR_MCP_CHUNK_SIZE` (e.g., from 512 to 256)
2. Process files individually instead of batch indexing
3. Increase system swap space

### Search Returns Irrelevant Results

**Issue**: Search results don't match expectations

**Solution**:
1. Verify documents were indexed: check ChromaDB directory size
2. Try more specific queries with domain terminology
3. Use metadata filters to narrow search scope
4. Increase `n_results` to see more candidates

## Development

### Running Tests

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src/vector_mcp --cov-report=html

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

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## API Reference

See [docs/API.md](docs/API.md) for complete MCP tools specification.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production deployment guides.

## Performance

Typical performance characteristics:

- **Indexing**: ~0.5-1.0 seconds per page of content
- **Query**: <2 seconds for semantic search with 5 results
- **Token Reduction**: 80-95% for large documents (only relevant chunks retrieved)
- **Storage**: ~10MB per 100 pages of indexed content

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Support

- Issues: https://github.com/yourusername/vector-mcp-server/issues
- Discussions: https://github.com/yourusername/vector-mcp-server/discussions
- Documentation: https://github.com/yourusername/vector-mcp-server/wiki
