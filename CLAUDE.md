# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocVec - A Model Context Protocol server that provides semantic document indexing and retrieval using ChromaDB and local Ollama embeddings. The system is designed to reduce token usage in Claude conversations by efficiently retrieving only relevant document chunks.

**Core Principles**:
- Local-first processing using Ollama embeddings
- Privacy-preserving (no external API calls)
- Token-efficient retrieval

## Development Commands

### Environment Setup
```bash
# Install dependencies
uv sync

# Ensure Ollama is running with the embedding model
ollama pull mxbai-embed-large
# or
ollama pull nomic-embed-text
```

### Testing
```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src/docvec --cov-report=html

# Run specific test file
python -m pytest tests/test_indexer.py -v

# Run single test function
python -m pytest tests/test_indexer.py::test_index_document -v

# Run integration tests only
python -m pytest tests/test_integration.py -v
```

### Running the MCP Server
```bash
# Run with default configuration
python -m vector_mcp

# Run with custom configuration
python -m vector_mcp --host http://localhost:11434 --model nomic-embed-text --db-path ./my_chroma_db --log-level DEBUG

# View available CLI options
python -m vector_mcp --help
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

**Layer 5: Services**
- `embedding/ollama_client.py`: Ollama API client with retry logic
- `storage/chroma_store.py`: ChromaDB wrapper for vector storage
- `deduplication/hasher.py`: SHA-256 document hashing

### Key Data Flows

**Indexing Pipeline**:
1. File path → Indexer validates and reads content
2. File extension → Appropriate chunker selected
3. Content → Chunker splits into Chunk objects
4. Chunks → Validated against token limits
5. Chunk texts → Batched and sent to OllamaClient for embeddings
6. Chunks + Embeddings → Stored in ChromaStore with metadata

**Query Pipeline**:
1. Query text → Embedded via OllamaClient
2. Query embedding → ChromaStore performs cosine similarity search
3. Results → Filtered by metadata if specified
4. Results → Token budget enforcement (if budget specified)
5. Results → Formatted with similarity scores and returned

### Important Design Patterns

**Dependency Injection**: All components receive dependencies via constructor
- Makes testing easier (mock dependencies)
- Makes initialization order explicit
- Example: `Indexer(embedder, storage, chunk_size, batch_size)`

**Format-Specific Chunking**: File extension determines chunker strategy
- `.md` → MarkdownChunker (preserves header hierarchy)
- `.pdf` → PDFChunker (preserves page boundaries)
- `.py` → CodeChunker (uses AST for function/class boundaries)
- `.txt` or unknown → TextChunker (paragraph-based)

**Batch Processing**: Embeddings generated in configurable batches
- Reduces API calls to Ollama
- Default batch_size=16 in Indexer
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
1. OllamaClient (no dependencies)
2. ChromaStore (no dependencies)
3. DocumentHasher (no dependencies)
4. Indexer (depends on embedder, storage)
5. BatchProcessor (depends on indexer, hasher, storage)
6. IndexingTools (depends on batch_processor, indexer)
7. QueryTools (depends on embedder, storage)

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

1. Create new chunker in `src/vector_mcp/chunking/`:
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

1. Implement logic in appropriate tool class (`indexing_tools.py` or `query_tools.py`)

2. Register tool handler in `__main__.py`:
   ```python
   @mcp.tool()
   async def my_new_tool(param: str) -> dict[str, Any]:
       """Tool description for Claude."""
       if indexing_tools is None:
           return {"success": False, "error": "Server not initialized"}
       return await indexing_tools.my_method(param)
   ```

3. Add tests in `tests/test_mcp_indexing_tools.py` or `tests/test_mcp_query_tools.py`

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

All configuration via CLI arguments (parsed in `parse_arguments()`):
- `--host`: Ollama server URL
- `--model`: Embedding model name
- `--db-path`: ChromaDB storage directory
- `--collection`: Collection name
- `--chunk-size`: Max tokens per chunk
- `--batch-size`: Embedding batch size
- `--log-level`: Logging verbosity

Default values in argument parser provide sensible defaults for development.

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
