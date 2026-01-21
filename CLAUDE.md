# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocVec - A Model Context Protocol server that provides semantic document indexing and retrieval using ChromaDB and local embeddings. The system is designed to reduce token usage in Claude conversations by efficiently retrieving only relevant document chunks.

**Core Principles**:
- Local-first processing using MLX embeddings (default) or Ollama
- Privacy-preserving (no external API calls)
- Token-efficient retrieval
- Apple Silicon optimized (MLX backend)

## Development Commands

### Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (includes mlx-embeddings for Apple Silicon)
uv sync

# MLX backend (default) - no additional setup needed on Apple Silicon
# The model is auto-downloaded on first use

# Ollama backend (alternative) - requires Ollama server
ollama serve &  # if not already running
ollama pull nomic-embed-text  # or mxbai-embed-large
```

### Testing
```bash
# Run all tests with coverage
uv run pytest tests/ -v --cov=src/docvec --cov-report=html

# Run specific test file
uv run pytest tests/test_indexer.py -v

# Run single test function
uv run pytest tests/test_indexer.py::test_index_document -v

# Run integration tests only
uv run pytest tests/test_integration.py -v
```

### Running the MCP Server
```bash
# Run with default configuration (MLX backend)
uv run python -m docvec

# Run with Ollama backend instead
uv run python -m docvec --embedding-backend ollama

# Run with custom MLX model
uv run python -m docvec --mlx-model mlx-community/mxbai-embed-large-v1

# Run with custom Ollama configuration
uv run python -m docvec --embedding-backend ollama --host http://localhost:11434 --model nomic-embed-text

# View available CLI options
uv run python -m docvec --help
```

### Code Quality
```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type checking
uv run mypy src/
```

## Architecture

### Component Hierarchy

The system follows a layered architecture with clear separation of concerns:

**Layer 1: MCP Server** (`__main__.py`, `mcp_server.py`)
- Entry point and FastMCP server setup
- Tool registration and JSON-RPC protocol handling
- CLI argument parsing and component initialization
- CRITICAL: Uses stdio transport - all logging MUST go to stderr, never stdout

**Layer 2: Tool Interfaces** (`mcp_tools/`)
- `indexing_tools.py`: Orchestrates file/directory indexing workflows
- `query_tools.py`: Manages search and retrieval operations
- `management_tools.py`: Handles delete operations and collection management
- Validates inputs, formats responses, handles errors

**Layer 3: Business Logic**
- `indexing/indexer.py`: Core document indexing orchestrator
  - Auto-detects file format by extension
  - Selects appropriate chunker
  - Batches embeddings for efficiency
  - Coordinates storage operations
- `indexing/batch_processor.py`: Multi-file indexing with deduplication
  - Directory scanning and filtering
  - Hash-based duplicate detection
  - Progress tracking

**Layer 4: Chunking Strategies** (`chunking/`)
- `base.py`: Abstract chunker interface and Chunk dataclass
- `markdown_chunker.py`: Header-aware chunking for markdown
- `pdf_chunker.py`: Page-aware chunking for PDFs
- `code_chunker.py`: AST-based chunking for Python code
- `text_chunker.py`: Paragraph-based chunking for plain text

**Automatic Chunk Splitting**: Chunks that exceed `max_tokens` are automatically split via `split_oversized_chunk()` in `base.py`. The splitting strategy prioritizes semantic boundaries:
1. Paragraph boundaries (double newline) first
2. Falls back to line boundaries if needed
3. Word boundaries as final fallback
Split chunks include `split_part` metadata for tracking.

**Layer 5: Services**
- `embedding/provider.py`: EmbeddingProvider protocol for pluggable backends
- `embedding/mlx_provider.py`: MLX embedding provider (Apple Silicon, default)
- `embedding/ollama_client.py`: Ollama API client with retry logic
- `embedding/factory.py`: Factory function for creating embedding providers
- `storage/chroma_store.py`: ChromaDB wrapper for vector storage
- `deduplication/hasher.py`: SHA-256 document hashing

### Key Data Flows

**Indexing Pipeline**:
1. File path → Indexer validates and reads content
2. File extension → Appropriate chunker selected
3. Content → Chunker splits into Chunk objects
4. Chunks → Validated against token limits
5. Chunk texts → Batched and sent to EmbeddingProvider for embeddings
6. Chunks + Embeddings → Stored in ChromaStore with metadata

**Query Pipeline**:
1. Query text → Embedded via EmbeddingProvider (with query prefix)
2. Query embedding → ChromaStore performs cosine similarity search
3. Results → Filtered by metadata if specified
4. Results → Token budget enforcement (if budget specified)
5. Results → Formatted with similarity scores and returned

## MCP Tools Reference

The server exposes nine tools organized into three categories:

### Indexing Tools
- `index_file(file_path)` - Index a single document file
- `index_directory(dir_path, recursive=True)` - Batch index multiple documents with deduplication

### Search Tools
- `search(query, n_results=5)` - Basic semantic search
- `search_with_filters(query, filters, n_results=5)` - Search with metadata filtering
- `search_with_budget(query, max_tokens)` - Token-budget aware search

### Management Tools
- `delete_chunks(ids)` - Delete specific chunks by their IDs
- `delete_file(source_file)` - Delete all chunks from a source file
- `clear_index(confirm)` - Clear entire collection (**requires `confirm=True`** to prevent accidental data loss)
- `get_index_stats()` - Get collection statistics (total chunks, unique files, file list)

For full API specifications including input/output schemas, examples, and error codes, see `docs/API.md`.

### Important Design Patterns

**Dependency Injection**: All components receive dependencies via constructor
- Makes testing easier (mock dependencies)
- Makes initialization order explicit
- Example: `Indexer(embedder, storage, chunk_size, batch_size, max_tokens)`

**Format-Specific Chunking**: File extension determines chunker strategy
- `.md` → MarkdownChunker (preserves header hierarchy)
- `.pdf` → PDFChunker (preserves page boundaries)
- `.py` → CodeChunker (uses AST for function/class boundaries)
- `.txt` or unknown → TextChunker (paragraph-based)

**Batch Processing**: Embeddings generated in configurable batches
- Reduces API calls to Ollama
- Default batch_size=32 in Indexer
- Trade-off: memory usage vs API efficiency

**Hash-Based Deduplication**: SHA-256 hashes stored in metadata
- Prevents re-indexing identical documents
- Hash stored in `doc_hash` metadata field
- BatchProcessor checks against existing hashes before indexing

**Metadata Strategy**: Rich metadata attached to each chunk
- Required: `source_file`, `chunk_index`, `doc_hash`
- Chunk-specific: varies by chunker (e.g., `header_path` for markdown, `page` for PDF)
- Enables powerful filtering in queries

## Critical Implementation Details

### MCP Server stdio Protocol
- **NEVER write to stdout** - it corrupts JSON-RPC messages
- All logging configured to stderr in `setup_logging()`
- This is a hard requirement for stdio-based MCP servers

### Component Initialization Order
The initialization sequence in `initialize_components()` follows the dependency graph:
1. EmbeddingProvider via factory (MLX or Ollama based on config)
2. ChromaStore (no dependencies)
3. ManagementTools (depends on storage)
4. DocumentHasher (no dependencies)
5. Indexer (depends on embedder, storage)
6. BatchProcessor (depends on indexer, hasher, storage)
7. IndexingTools (depends on batch_processor, indexer)
8. QueryTools (depends on embedder, storage)

### Token Approximation
- Chunkers use character-based chunk sizes (chunk_size * 4 chars)
- Validation uses rough approximation: 4 characters ≈ 1 token
- Actual token counting happens during query results (if implemented)

### Error Handling
- Custom exceptions: `IndexingError`, `EmbeddingError`, `StorageError`
- Tool functions return structured dicts with `success`, `data`, `error` fields
- Errors logged with full stack traces but sanitized in responses

### Async Patterns
- MCP tool handlers are async functions (`async def`)
- Business logic may be sync or async
- FastMCP server manages event loop automatically

## Common Workflows

### Adding Support for a New File Format

1. Create new chunker in `src/docvec/chunking/`:
   ```python
   class MyFormatChunker(AbstractChunker):
       def chunk(self, content: str, source_file: str) -> list[Chunk]:
           # Implement format-specific chunking logic
           pass
   ```

2. Add to Indexer's `_chunker_map`:
   ```python
   self._chunker_map = {
       ".md": MarkdownChunker,
       ".myext": MyFormatChunker,  # Add here
       # ...
   }
   ```

3. Update `_select_chunker()` if special initialization needed

4. Write tests in `tests/test_myformat_chunker.py`

### Adding a New MCP Tool

1. Implement logic in appropriate tool class (`indexing_tools.py`, `query_tools.py`, or `management_tools.py`)

2. Register tool handler in `__main__.py`:
   ```python
   @mcp.tool()
   async def my_new_tool(param: str) -> dict[str, Any]:
       """Tool description for Claude."""
       if indexing_tools is None:
           return {"success": False, "error": "Server not initialized"}
       return await indexing_tools.my_method(param)
   ```

3. Add tests in `tests/test_mcp_indexing_tools.py`, `tests/test_mcp_query_tools.py`, or `tests/test_mcp_management_tools.py`

### Testing with Mock Services

Tests extensively use fixtures for isolated testing:
- `mock_embedder`: Returns fake embeddings without Ollama
- `mock_storage`: In-memory ChromaDB for fast tests
- `temp_dir`: Temporary directories for file operations

Example pattern:
```python
def test_something(mock_embedder, mock_storage):
    indexer = Indexer(mock_embedder, mock_storage)
    # Test without external dependencies
```

## Configuration

Configuration is done via CLI arguments (takes precedence) or environment variables with the `DOCVEC_` prefix:

| Environment Variable | CLI Argument | Default | Description |
|----------|----------|---------|-------------|
| `DOCVEC_EMBEDDING_BACKEND` | `--embedding-backend` | `mlx` | Embedding backend (`mlx` or `ollama`) |
| `DOCVEC_MLX_MODEL` | `--mlx-model` | `mlx-community/mxbai-embed-large-v1` | MLX embedding model (HuggingFace path) |
| `DOCVEC_HOST` | `--host` | `http://localhost:11434` | Ollama API endpoint |
| `DOCVEC_MODEL` | `--model` | `nomic-embed-text` | Ollama embedding model name |
| `DOCVEC_TIMEOUT` | `--timeout` | `30` | Ollama request timeout in seconds |
| `DOCVEC_DB_PATH` | `--db-path` | `./chroma_db` | ChromaDB storage location |
| `DOCVEC_COLLECTION` | `--collection` | `documents` | ChromaDB collection name |
| `DOCVEC_CHUNK_SIZE` | `--chunk-size` | `512` | Maximum tokens per chunk |
| `DOCVEC_BATCH_SIZE` | `--batch-size` | `32` | Batch size for embedding generation |
| `DOCVEC_MAX_TOKENS` | `--max-tokens` | `512` | Maximum tokens per chunk for embedding model limits |
| `DOCVEC_LOG_LEVEL` | `--log-level` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

CLI arguments take precedence over environment variables. All configuration is parsed in `parse_arguments()` in `__main__.py`.

## File Paths and Structure

When working with paths:
- Always use `pathlib.Path` objects, not strings
- Resolve paths with `.resolve()` for absolute paths
- Use `Path.exists()`, `Path.is_file()`, `Path.is_dir()` for validation
- ChromaDB path stored as `Path` object in ChromaStore
- Source files tracked as strings in Chunk metadata

## Testing Philosophy

- **Unit tests**: Test each component in isolation with mocks
- **Integration tests**: Test full pipelines (indexing + query)
- **Fixtures**: Use pytest fixtures for shared test setup
- **Coverage**: Aim for comprehensive coverage of error paths
- Tests live in `tests/` with naming convention `test_<module>.py`
