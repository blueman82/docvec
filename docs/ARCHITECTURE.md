# Architecture Documentation

This document explains the design and architecture of the Vector Database MCP Server.

## System Overview

The Vector Database MCP Server is a document indexing and retrieval system built on three core principles:

1. **Local-first**: All processing happens locally using Ollama embeddings
2. **Privacy-preserving**: No data sent to external APIs
3. **Token-efficient**: Retrieve only relevant chunks to minimize Claude token usage

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude Code)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ JSON-RPC over stdio
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      MCP Server Layer                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Tool Registration & Dispatch                         │  │
│  │  - index_file, index_directory                        │  │
│  │  - search, search_with_filters, search_with_budget    │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             │ Function Calls
                             │
┌────────────────────────────▼────────────────────────────────┐
│                    Business Logic Layer                      │
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │ Indexing Tools  │              │   Query Tools       │   │
│  │ - Validation    │              │ - Query embedding   │   │
│  │ - Orchestration │              │ - Result filtering  │   │
│  │ - Error handling│              │ - Token budgeting   │   │
│  └────────┬────────┘              └──────────┬──────────┘   │
│           │                                   │              │
│           │                                   │              │
│  ┌────────▼──────────────────────────────────▼──────────┐   │
│  │           Batch Processor                            │   │
│  │  - Deduplication (hash checking)                     │   │
│  │  - Directory scanning                                │   │
│  │  - Progress tracking                                 │   │
│  └────────┬─────────────────────────────────────────────┘   │
│           │                                                  │
│  ┌────────▼──────────────────────────────────────────────┐  │
│  │           Document Indexer                            │  │
│  │  - Format detection                                   │  │
│  │  - Chunker selection and execution                   │  │
│  │  - Batch embedding orchestration                     │  │
│  │  - Storage coordination                              │  │
│  └─────┬────────────────────────────┬────────────────────┘  │
└────────┼────────────────────────────┼───────────────────────┘
         │                            │
         │                            │
┌────────▼────────────┐      ┌────────▼───────────┐
│  Document Chunkers  │      │  Service Layer     │
│                     │      │                    │
│  ┌──────────────┐  │      │  ┌──────────────┐  │
│  │   Markdown   │  │      │  │ Ollama Client│  │
│  │   Chunker    │  │      │  │ - Embeddings │  │
│  └──────────────┘  │      │  │ - Retry logic│  │
│  ┌──────────────┐  │      │  │ - Health chk │  │
│  │     PDF      │  │      │  └──────────────┘  │
│  │   Chunker    │  │      │                    │
│  └──────────────┘  │      │  ┌──────────────┐  │
│  ┌──────────────┐  │      │  │ ChromaDB     │  │
│  │     Text     │  │      │  │  Storage     │  │
│  │   Chunker    │  │      │  │ - Add chunks │  │
│  └──────────────┘  │      │  │ - Search     │  │
│  ┌──────────────┐  │      │  │ - Filtering  │  │
│  │     Code     │  │      │  └──────────────┘  │
│  │   Chunker    │  │      │                    │
│  └──────────────┘  │      │  ┌──────────────┐  │
└─────────────────────┘      │  │ Token Counter│  │
                             │  │ - tiktoken   │  │
                             │  └──────────────┘  │
                             │                    │
                             │  ┌──────────────┐  │
                             │  │ Hash Calc    │  │
                             │  │ - SHA-256    │  │
                             │  └──────────────┘  │
                             └────────────────────┘
```

## Component Descriptions

### MCP Server Layer

**Responsibility**: Protocol handling and tool registration

**Implementation**: `src/vector_mcp/mcp_server.py`

The MCP Server implements the Model Context Protocol specification, exposing tools to Claude Code via JSON-RPC over stdio. It handles:

- Tool schema registration
- Input validation and deserialization
- Response serialization
- Error propagation
- Lifecycle management (startup/shutdown)

**Key Design Decisions**:
- Async-first: All handlers use async/await for non-blocking I/O
- Thin layer: Business logic delegated to tool classes
- Schema-driven: Input/output schemas define contracts

### Business Logic Layer

#### Indexing Tools

**Responsibility**: Document indexing workflow orchestration

**Implementation**: `src/vector_mcp/mcp_tools/indexing_tools.py`

Coordinates the indexing pipeline:
1. Path validation
2. File/directory detection
3. Delegation to BatchProcessor or Indexer
4. Result formatting for MCP response

**Key Design Decisions**:
- Validation-first: Fail fast on invalid inputs
- Structured errors: Return actionable error messages
- Progress transparency: Return detailed statistics

#### Query Tools

**Responsibility**: Search and retrieval workflow

**Implementation**: `src/vector_mcp/mcp_tools/query_tools.py`

Manages semantic search pipeline:
1. Query embedding via Ollama
2. Vector search in ChromaDB
3. Metadata filtering
4. Token budget enforcement
5. Result ranking and formatting

**Key Design Decisions**:
- Token awareness: Always calculate and return token counts
- Flexible filtering: Support complex metadata queries
- Score transparency: Include similarity scores in results

#### Batch Processor

**Responsibility**: Multi-document indexing with deduplication

**Implementation**: `src/vector_mcp/indexing/batch_processor.py`

Handles directory-level operations:
- File discovery (glob patterns)
- SHA-256 hash calculation
- Duplicate detection
- Parallel processing coordination
- Error collection and reporting

**Key Design Decisions**:
- Hash-based deduplication: O(1) duplicate detection
- Fail-soft: Continue on individual file errors
- Detailed reporting: Track new/duplicate/error counts

#### Document Indexer

**Responsibility**: Single-document indexing orchestration

**Implementation**: `src/vector_mcp/indexing/indexer.py`

**INTEGRATION POINT**: Coordinates multiple subsystems

Pipeline:
1. Format detection (by extension)
2. Chunker selection and execution
3. Token validation
4. Batch embedding (via Ollama)
5. Storage (via ChromaDB)

**Key Design Decisions**:
- Strategy pattern: Chunkers selected dynamically
- Batch embedding: Minimize API calls to Ollama
- Transactional storage: Rollback on partial failure

**Integration Points**:
- Chunkers: MarkdownChunker, PDFChunker, TextChunker, CodeChunker
- OllamaClient: Batch embedding API
- ChromaStore: Vector storage
- TokenCounter: Chunk validation

### Document Chunkers

**Responsibility**: Format-specific content segmentation

**Base Interface**: `src/vector_mcp/chunking/base.py`

All chunkers implement `AbstractChunker` interface:

```python
class AbstractChunker(ABC):
    @abstractmethod
    def chunk(self, content: str, source_file: str) -> list[Chunk]:
        """Split content into semantically meaningful chunks."""
        pass
```

#### Markdown Chunker

**Implementation**: `src/vector_mcp/chunking/markdown_chunker.py`

**Strategy**: Header-aware hierarchical chunking

Algorithm:
1. Parse markdown to identify headers (H1-H6)
2. Build hierarchy tree (track parent headers)
3. Group content under each header
4. Split large sections at paragraph boundaries
5. Include header path in metadata

**Metadata**:
- `header_path`: "Chapter 1 > Section 1.2 > Subsection"
- `header_level`: 1-6
- `chunk_index`: Position in document

**Design Rationale**: Preserves document structure for context-aware retrieval

#### PDF Chunker

**Implementation**: `src/vector_mcp/chunking/pdf_chunker.py`

**Strategy**: Page-aware with content continuity

Algorithm:
1. Extract text per page (pypdf)
2. Concatenate pages
3. Chunk with page boundary awareness
4. Track page numbers in metadata
5. Handle cross-page chunks with overlap

**Metadata**:
- `page_number`: Source page
- `page_position`: start/middle/end
- `chunk_index`: Position in document

**Design Rationale**: Page numbers essential for reference verification

#### Text Chunker

**Implementation**: `src/vector_mcp/chunking/text_chunker.py`

**Strategy**: Paragraph-first with sentence fallback

Algorithm:
1. Split on blank lines (paragraphs)
2. For oversized paragraphs, split on sentences
3. Apply overlap between chunks
4. Track line numbers

**Metadata**:
- `start_line`: First line number
- `end_line`: Last line number
- `chunk_index`: Position in document

**Design Rationale**: Simplicity and broad applicability

#### Code Chunker

**Implementation**: `src/vector_mcp/chunking/code_chunker.py`

**Strategy**: AST-based semantic chunking

Algorithm:
1. Parse Python AST
2. Extract top-level definitions (functions, classes)
3. Group class methods with class definition
4. Preserve imports in metadata
5. Fall back to line-based for syntax errors

**Metadata**:
- `function_name`: Function/method name
- `class_name`: Containing class (if applicable)
- `start_line`: Starting line
- `end_line`: Ending line
- `node_type`: FunctionDef, ClassDef, AsyncFunctionDef

**Design Rationale**: Semantic units improve code retrieval relevance

### Service Layer

#### Ollama Client

**Responsibility**: Embedding generation

**Implementation**: `src/vector_mcp/embedding/ollama_client.py`

Features:
- HTTP client for Ollama API
- Batch embedding support (reduces API calls)
- Exponential backoff retry (3 attempts)
- Health check (validates model availability)

**Key Design Decisions**:
- Batch size configurable (default: 32 chunks/batch)
- Retry on transient failures (network, temp unavailability)
- Timeout configurable (default: 30s)

**API**:
```python
class OllamaClient:
    def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

    def health_check(self) -> bool:
        """Verify Ollama service and model availability."""
```

#### ChromaDB Storage

**Responsibility**: Vector database operations

**Implementation**: `src/vector_mcp/storage/chroma_store.py`

Features:
- Persistent local storage
- Vector similarity search
- Metadata filtering (WHERE clauses)
- Deduplication tracking (by hash)

**Key Design Decisions**:
- Persistent client (data survives restarts)
- Collection auto-creation (initialize on first use)
- Metadata-first design (rich filtering capabilities)

**API**:
```python
class ChromaStore:
    def add(self, embeddings, documents, metadatas) -> list[str]:
        """Store embeddings with metadata."""

    def search(self, query_embedding, n_results=5, where=None):
        """Search by similarity with optional filtering."""

    def delete(self, ids: list[str]) -> None:
        """Remove chunks by ID."""

    def get_by_hash(self, doc_hash: str) -> Optional[dict]:
        """Check if document hash exists (deduplication)."""
```

#### Token Counter

**Responsibility**: Accurate token counting

**Implementation**: `src/vector_mcp/token_counter.py`

Uses `tiktoken` with cl100k_base encoding (Claude/GPT-4 tokenizer)

**Key Design Decisions**:
- Cache encoding instance (expensive to initialize)
- cl100k_base encoding (matches Claude tokenization)
- Batch counting support

#### Hash Calculator

**Responsibility**: Document deduplication

**Implementation**: `src/vector_mcp/deduplication/hasher.py`

Uses SHA-256 for content hashing

**Key Design Decisions**:
- Streaming hash (memory-efficient for large files)
- Consistent UTF-8 encoding
- Cryptographic hash (negligible collision probability)

## Data Flow

### Indexing Flow

```
┌──────────┐
│ User     │
│ Command  │
└────┬─────┘
     │
     │ index_directory("/docs")
     ▼
┌────────────────┐
│ IndexingTools  │ ─── Validate path
└────┬───────────┘
     │
     │ Call batch_processor
     ▼
┌───────────────────┐
│ BatchProcessor    │ ─── Scan directory
└────┬──────────────┘   ─── Calculate hashes
     │                   ─── Check for duplicates
     │
     │ For each new file
     ▼
┌──────────────┐
│ Indexer      │ ─── Detect format (.md, .pdf, etc.)
└────┬─────────┘   ─── Select chunker
     │
     │ Read file content
     ▼
┌──────────────┐
│ Chunker      │ ─── Parse structure
│ (Markdown)   │ ─── Split into chunks
└────┬─────────┘   ─── Add metadata
     │
     │ chunks: [Chunk, Chunk, ...]
     ▼
┌──────────────┐
│ TokenCounter │ ─── Validate chunk sizes
└────┬─────────┘
     │
     │ valid_chunks
     ▼
┌──────────────┐
│ OllamaClient │ ─── Batch chunks (32 at a time)
└────┬─────────┘   ─── Call Ollama API
     │              ─── Retry on failure
     │
     │ embeddings: [[0.1, 0.2, ...], ...]
     ▼
┌──────────────┐
│ ChromaStore  │ ─── Store embeddings
└────┬─────────┘   ─── Store metadata
     │              ─── Store doc_hash
     │
     │ chunk_ids: ["chunk_001", ...]
     ▼
┌──────────────┐
│ Response     │ ─── Format results
└──────────────┘   ─── Return to user
```

### Query Flow

```
┌──────────┐
│ User     │
│ Query    │
└────┬─────┘
     │
     │ search("authentication config")
     ▼
┌─────────────┐
│ QueryTools  │ ─── Validate query
└────┬────────┘
     │
     │ query text
     ▼
┌──────────────┐
│ OllamaClient │ ─── Generate query embedding
└────┬─────────┘
     │
     │ query_embedding: [0.3, 0.1, ...]
     ▼
┌──────────────┐
│ ChromaStore  │ ─── Similarity search
└────┬─────────┘   ─── Apply filters
     │              ─── Rank by score
     │
     │ raw_results: [{chunk, score, metadata}, ...]
     ▼
┌──────────────┐
│ TokenCounter │ ─── Count tokens per chunk
└────┬─────────┘   ─── Enforce budget
     │
     │ filtered_results
     ▼
┌──────────────┐
│ Response     │ ─── Format results
└──────────────┘   ─── Include metadata
                    ─── Return to user
```

## Design Decisions and Rationale

### Why Local Embeddings (Ollama)?

**Decision**: Use Ollama for local embedding generation

**Rationale**:
- **Privacy**: Documents never leave local machine
- **Cost**: No API fees for embedding generation
- **Latency**: Faster for small batches (no network round trip)
- **Control**: Model selection and version control

**Trade-offs**:
- Requires local compute resources
- Slower than cloud APIs for very large batches
- Limited to models supported by Ollama

### Why ChromaDB?

**Decision**: Use ChromaDB as vector database

**Rationale**:
- **Simplicity**: Embedded database, no server setup
- **Python-native**: Easy integration
- **Metadata filtering**: Rich query capabilities
- **Persistence**: File-based storage

**Trade-offs**:
- Single-machine only (no distributed setup)
- Performance degrades with very large datasets (>1M chunks)

### Why Format-Specific Chunkers?

**Decision**: Implement separate chunkers per format

**Rationale**:
- **Semantic integrity**: Preserve document structure (headers, functions, pages)
- **Context preservation**: Include format-specific metadata
- **Retrieval quality**: Better search results with context

**Alternative considered**: Generic text splitter
- Simpler implementation
- Lost structure and context
- Poorer retrieval relevance

### Why SHA-256 Hashing?

**Decision**: Use SHA-256 for deduplication

**Rationale**:
- **Collision resistance**: Cryptographically secure
- **Fast**: Efficient for file sizes we handle
- **Standard**: Widely supported

**Alternative considered**: File modification timestamps
- Unreliable (timestamps can be manipulated)
- Misses content changes with same mtime

### Why Token Counting?

**Decision**: Use tiktoken for accurate token counts

**Rationale**:
- **Accuracy**: Matches Claude tokenization
- **Budget control**: Precise token budget enforcement
- **Cost visibility**: Users know exact token usage

### Why Batch Embedding?

**Decision**: Embed chunks in batches (default: 32)

**Rationale**:
- **Performance**: Reduces API calls by 32x
- **Latency**: Ollama handles batches efficiently
- **Resource usage**: Better GPU utilization

**Trade-off**: Higher memory usage during embedding

## Scalability Considerations

### Current Limitations

1. **Single-machine**: ChromaDB is not distributed
2. **In-process**: All processing in single Python process
3. **Memory-bound**: Large batches limited by RAM

### Scaling Strategies

**For larger datasets (>10,000 documents)**:

1. **Index incrementally**: Process files in batches
2. **Increase chunking**: Larger chunks reduce total count
3. **Distributed storage**: Replace ChromaDB with Qdrant/Milvus
4. **Background processing**: Async indexing with job queue

**For high query volume**:

1. **Cache embeddings**: Store query embeddings
2. **Read replicas**: Multiple ChromaDB instances
3. **Connection pooling**: Reuse Ollama connections

### Performance Benchmarks

Typical performance on M1 MacBook Pro:

| Operation | Performance |
|-----------|-------------|
| Markdown chunking | 10,000 lines/sec |
| PDF extraction | 2-5 pages/sec |
| Embedding (CPU) | 50 chunks/sec |
| Embedding (GPU) | 200 chunks/sec |
| ChromaDB add | 1000 chunks/sec |
| ChromaDB search | 50ms (10k chunks), 200ms (100k chunks) |

## Error Handling Philosophy

### Fail Fast for User Errors

Invalid inputs (bad paths, malformed queries) return immediately with clear error messages.

### Retry for Transient Failures

Network errors, temporary Ollama unavailability use exponential backoff.

### Fail Soft for Batch Operations

Individual file failures don't stop batch processing. Errors collected and reported.

### Propagate Critical Errors

Storage failures, corruption issues propagate to user with full context.

## Testing Strategy

### Unit Tests

Each component tested in isolation with mocked dependencies.

**Coverage target**: 80%+

### Integration Tests

Full pipeline tests with real Ollama and ChromaDB.

**Key scenarios**:
- Multi-format indexing
- Deduplication verification
- Query relevance validation
- Error recovery

### Performance Tests

Benchmark indexing and query performance with realistic datasets.

## Future Enhancements

1. **Incremental updates**: Re-index only changed chunks
2. **Multi-language support**: Extend code chunker beyond Python
3. **OCR integration**: Handle scanned PDFs
4. **Hybrid search**: Combine semantic and keyword search
5. **Query expansion**: Automatic query enhancement
6. **Relevance feedback**: Learn from user interactions
