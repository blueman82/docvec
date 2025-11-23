# MCP Tools API Reference

This document provides detailed specifications for all MCP tools exposed by the Vector Database MCP Server.

## Overview

The server exposes five primary tools for document indexing and semantic search:

1. `index_file` - Index a single document
2. `index_directory` - Batch index multiple documents
3. `search` - Basic semantic search
4. `search_with_filters` - Search with metadata filtering
5. `search_with_budget` - Token-budget aware search

## Tool: index_file

Index a single document into the vector database.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Absolute or relative path to the document"
    }
  },
  "required": ["file_path"]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file_path` | string | Yes | Path to document file. Supports: .md, .pdf, .txt, .py |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether indexing succeeded"
    },
    "data": {
      "type": "object",
      "properties": {
        "doc_hash": {
          "type": "string",
          "description": "SHA-256 hash of document content"
        },
        "chunks_created": {
          "type": "integer",
          "description": "Number of chunks indexed"
        },
        "chunk_ids": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of chunk IDs stored in database"
        },
        "total_tokens": {
          "type": "integer",
          "description": "Total tokens across all chunks"
        },
        "file_format": {
          "type": "string",
          "description": "Detected file format (markdown, pdf, text, code)"
        }
      }
    },
    "error": {
      "type": "string",
      "description": "Error message if success is false"
    }
  }
}
```

### Example Usage

Request:
```json
{
  "file_path": "/Users/harrison/docs/README.md"
}
```

Response (success):
```json
{
  "success": true,
  "data": {
    "doc_hash": "a7f3d2e1b4c5...",
    "chunks_created": 12,
    "chunk_ids": ["chunk_001", "chunk_002", ...],
    "total_tokens": 4523,
    "file_format": "markdown"
  }
}
```

Response (error):
```json
{
  "success": false,
  "error": "File not found: /Users/harrison/docs/README.md"
}
```

### Error Codes

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `File not found: {path}` | File doesn't exist | Check file path |
| `Permission denied: {path}` | No read access | Fix file permissions |
| `Unsupported file format: {ext}` | Unknown extension | Use supported format |
| `Failed to extract content` | Corrupted file | Verify file integrity |
| `Embedding service unavailable` | Ollama not running | Start Ollama |

---

## Tool: index_directory

Batch index multiple documents from a directory with deduplication.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "dir_path": {
      "type": "string",
      "description": "Directory path to scan for documents"
    },
    "recursive": {
      "type": "boolean",
      "description": "Scan subdirectories recursively",
      "default": true
    }
  },
  "required": ["dir_path"]
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `dir_path` | string | Yes | - | Directory path to scan |
| `recursive` | boolean | No | true | Include subdirectories |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Overall operation success"
    },
    "data": {
      "type": "object",
      "properties": {
        "new_documents": {
          "type": "integer",
          "description": "Number of new documents indexed"
        },
        "duplicates_skipped": {
          "type": "integer",
          "description": "Number of duplicate documents skipped"
        },
        "errors": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "file_path": {"type": "string"},
              "error": {"type": "string"}
            }
          },
          "description": "List of files that failed to index"
        },
        "total_chunks": {
          "type": "integer",
          "description": "Total chunks created across all documents"
        },
        "chunk_ids": {
          "type": "object",
          "description": "Map of file paths to their chunk IDs"
        }
      }
    }
  }
}
```

### Example Usage

Request:
```json
{
  "dir_path": "/Users/harrison/docs",
  "recursive": true
}
```

Response:
```json
{
  "success": true,
  "data": {
    "new_documents": 15,
    "duplicates_skipped": 3,
    "errors": [
      {
        "file_path": "/Users/harrison/docs/corrupted.pdf",
        "error": "Failed to extract content from PDF"
      }
    ],
    "total_chunks": 187,
    "chunk_ids": {
      "/Users/harrison/docs/README.md": ["chunk_001", "chunk_002"],
      "/Users/harrison/docs/API.md": ["chunk_003", "chunk_004"]
    }
  }
}
```

### Behavior

- Automatically detects supported file formats (.md, .pdf, .txt, .py)
- Skips files that have already been indexed (based on SHA-256 hash)
- Continues processing on individual file failures
- Returns comprehensive statistics and error details

---

## Tool: search

Perform semantic search across indexed documents.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "n_results": {
      "type": "integer",
      "description": "Maximum number of results to return",
      "default": 5,
      "minimum": 1,
      "maximum": 50
    }
  },
  "required": ["query"]
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text |
| `n_results` | integer | No | 5 | Maximum results (1-50) |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean"
    },
    "data": {
      "type": "object",
      "properties": {
        "results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "chunk_id": {"type": "string"},
              "content": {"type": "string"},
              "score": {
                "type": "number",
                "description": "Similarity score (0.0-1.0, higher is better)"
              },
              "metadata": {
                "type": "object",
                "properties": {
                  "source_file": {"type": "string"},
                  "chunk_index": {"type": "integer"},
                  "file_format": {"type": "string"}
                }
              },
              "token_count": {"type": "integer"}
            }
          }
        },
        "total_tokens": {
          "type": "integer",
          "description": "Total tokens in all results"
        },
        "query_embedding_time": {
          "type": "number",
          "description": "Time to generate query embedding (seconds)"
        },
        "search_time": {
          "type": "number",
          "description": "ChromaDB search time (seconds)"
        }
      }
    }
  }
}
```

### Metadata Fields

Different document formats include format-specific metadata:

**Markdown**:
- `header_path`: Full header hierarchy (e.g., "Chapter 1 > Section 1.2")
- `header_level`: Header depth (1-6)

**PDF**:
- `page_number`: Source page number
- `page_position`: Position within page (start, middle, end)

**Code**:
- `class_name`: Containing class (if applicable)
- `function_name`: Function/method name
- `start_line`: Starting line number
- `end_line`: Ending line number

**Text**:
- `start_line`: Starting line number
- `end_line`: Ending line number

### Example Usage

Request:
```json
{
  "query": "How do I configure authentication?",
  "n_results": 5
}
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "chunk_id": "chunk_042",
        "content": "## Authentication Configuration\n\nTo configure authentication, set the following environment variables:\n- AUTH_PROVIDER: oauth2 or jwt\n- AUTH_SECRET: Your secret key",
        "score": 0.89,
        "metadata": {
          "source_file": "/Users/harrison/docs/config.md",
          "chunk_index": 3,
          "file_format": "markdown",
          "header_path": "Configuration > Authentication Configuration",
          "header_level": 2
        },
        "token_count": 47
      }
    ],
    "total_tokens": 235,
    "query_embedding_time": 0.23,
    "search_time": 0.15
  }
}
```

---

## Tool: search_with_filters

Search with metadata filtering for scoped retrieval.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language search query"
    },
    "filters": {
      "type": "object",
      "description": "Metadata filters to apply",
      "properties": {
        "source_file": {"type": "string"},
        "file_format": {"type": "string"},
        "page_number": {"type": "integer"},
        "header_level": {"type": "integer"}
      }
    },
    "n_results": {
      "type": "integer",
      "default": 5
    }
  },
  "required": ["query"]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `filters` | object | No | Metadata filters (see below) |
| `n_results` | integer | No | Maximum results |

### Filter Operators

Filters use ChromaDB WHERE clause syntax:

**Exact match**:
```json
{"source_file": "/path/to/doc.md"}
```

**Multiple conditions (AND)**:
```json
{
  "source_file": "/path/to/doc.md",
  "header_level": 2
}
```

**Range conditions**:
```json
{
  "page_number": {"$gte": 10, "$lte": 20}
}
```

**Pattern matching**:
```json
{
  "source_file": {"$contains": "config"}
}
```

### Example Usage

Request:
```json
{
  "query": "API endpoints",
  "filters": {
    "source_file": "/Users/harrison/docs/api.md",
    "header_level": 2
  },
  "n_results": 10
}
```

Response (same schema as `search` tool)

---

## Tool: search_with_budget

Search with token budget constraints.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Maximum tokens in results",
      "default": 3000
    }
  },
  "required": ["query"]
}
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query |
| `max_tokens` | integer | No | 3000 | Token budget limit |

### Behavior

1. Retrieves candidate results (up to 50)
2. Ranks by similarity score
3. Returns top results fitting within token budget
4. Prioritizes highest-scoring chunks

### Example Usage

Request:
```json
{
  "query": "deployment instructions",
  "max_tokens": 2000
}
```

Response:
```json
{
  "success": true,
  "data": {
    "results": [...],
    "total_tokens": 1987,
    "budget_utilized": 0.99,
    "truncated": true,
    "truncated_count": 3
  }
}
```

The `truncated` field indicates if some results were excluded to stay within budget.

---

## Error Handling

All tools return a consistent error response format:

```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "error_code": "ERROR_CODE",
  "details": {
    "additional": "context-specific information"
  }
}
```

### Common Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `FILE_NOT_FOUND` | File path doesn't exist | Verify path is correct |
| `PERMISSION_DENIED` | Cannot read file | Fix permissions |
| `UNSUPPORTED_FORMAT` | File format not supported | Use .md, .pdf, .txt, or .py |
| `EMBEDDING_FAILED` | Ollama embedding error | Check Ollama is running |
| `STORAGE_ERROR` | ChromaDB operation failed | Check disk space and permissions |
| `INVALID_INPUT` | Invalid parameter values | Check parameter types and ranges |
| `QUERY_FAILED` | Search operation failed | Check database integrity |
| `BUDGET_EXCEEDED` | Token budget too small | Increase max_tokens |

---

## Rate Limits and Performance

### Embedding Rate Limits

Ollama local embeddings have no rate limits, but performance depends on hardware:

- CPU: ~50-100 chunks/second
- GPU: ~200-500 chunks/second

### Recommended Batch Sizes

- Small files (<10 pages): Index individually
- Large directories: Use `index_directory` with recursive=true
- Very large files (>100 pages): Indexing may take 1-2 minutes

### Query Performance

Typical query latency:
- Query embedding: 200-500ms
- ChromaDB search: 50-200ms (depends on database size)
- Total: <1 second for most queries

### Database Size Considerations

| Indexed Pages | ChromaDB Size | Query Latency |
|--------------|---------------|---------------|
| 100 | ~10 MB | <100ms |
| 1,000 | ~100 MB | <200ms |
| 10,000 | ~1 GB | <500ms |
| 100,000 | ~10 GB | <2s |

---

## Best Practices

### Indexing

1. **Batch index directories** rather than individual files
2. **Use deduplication** - re-indexing is automatically skipped
3. **Monitor storage** - ChromaDB grows with content
4. **Index incrementally** - add new files as needed

### Searching

1. **Be specific** - detailed queries return better results
2. **Use filters** - narrow scope for faster, more relevant results
3. **Set appropriate budgets** - balance context vs. token usage
4. **Review metadata** - understand result provenance

### Maintenance

1. **Periodically clean database** - remove outdated documents
2. **Monitor query performance** - large databases may need optimization
3. **Update embeddings** - if embedding model changes, re-index
4. **Backup ChromaDB** - copy database directory regularly
