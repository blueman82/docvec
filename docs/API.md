# MCP Tools API Reference

This document provides detailed specifications for all MCP tools exposed by the DocVec server.

## Overview

The server exposes nine tools for document indexing, semantic search, and collection management:

**Indexing Tools**
1. `index_file` - Index a single document
2. `index_directory` - Batch index multiple documents

**Search Tools**
3. `search` - Basic semantic search
4. `search_with_filters` - Search with metadata filtering
5. `search_with_budget` - Token-budget aware search

**Management Tools**
6. `delete_chunks` - Delete specific chunks by ID
7. `delete_file` - Delete all chunks from a source file
8. `clear_index` - Clear entire collection (requires confirmation)
9. `get_index_stats` - Get collection statistics

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
| `Embedding service unavailable` | MLX/Ollama error | Check embedding backend configuration |

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

## Tool: delete_chunks

Delete specific chunks by their IDs. Useful for removing individual chunks that are outdated or incorrect.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of chunk IDs to delete"
    }
  },
  "required": ["ids"]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ids` | array[string] | Yes | List of chunk IDs to delete |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether deletion succeeded"
    },
    "data": {
      "type": "object",
      "properties": {
        "deleted_count": {
          "type": "integer",
          "description": "Number of chunks deleted"
        },
        "deleted_ids": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of deleted chunk IDs"
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
  "ids": ["1732360800_0", "1732360800_1", "1732360800_2"]
}
```

Response (success):
```json
{
  "success": true,
  "data": {
    "deleted_count": 3,
    "deleted_ids": ["1732360800_0", "1732360800_1", "1732360800_2"]
  },
  "error": null
}
```

Response (empty list):
```json
{
  "success": true,
  "data": {
    "deleted_count": 0,
    "deleted_ids": []
  },
  "error": null
}
```

### Error Cases

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `ids must be a list` | ids parameter is null | Provide a valid list |
| `Failed to delete by IDs: {error}` | Storage operation failed | Check database state |

---

## Tool: delete_file

Delete all chunks associated with a specific source file. Useful when removing an entire document from the index.

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "source_file": {
      "type": "string",
      "description": "Source file path to delete chunks for"
    }
  },
  "required": ["source_file"]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `source_file` | string | Yes | Path of source file to remove from index |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether deletion succeeded"
    },
    "data": {
      "type": "object",
      "properties": {
        "deleted_count": {
          "type": "integer",
          "description": "Number of chunks deleted"
        },
        "source_file": {
          "type": "string",
          "description": "Source file that was deleted"
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
  "source_file": "/Users/harrison/docs/outdated.md"
}
```

Response (success):
```json
{
  "success": true,
  "data": {
    "deleted_count": 15,
    "source_file": "/Users/harrison/docs/outdated.md"
  },
  "error": null
}
```

Response (file not indexed):
```json
{
  "success": true,
  "data": {
    "deleted_count": 0,
    "source_file": "/Users/harrison/docs/not_indexed.md"
  },
  "error": null
}
```

### Error Cases

| Error Message | Cause | Solution |
|--------------|-------|----------|
| `source_file cannot be empty` | Empty or whitespace-only path | Provide valid file path |
| `Failed to delete by file: {error}` | Storage operation failed | Check database state |

---

## Tool: clear_index

Delete all documents from the collection. **Requires explicit confirmation to prevent accidental data loss.**

### Input Schema

```json
{
  "type": "object",
  "properties": {
    "confirm": {
      "type": "boolean",
      "description": "Safety gate - must be true to proceed"
    }
  },
  "required": ["confirm"]
}
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | boolean | Yes | Must be `true` to proceed with deletion |

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether clearing succeeded"
    },
    "data": {
      "type": "object",
      "properties": {
        "deleted_count": {
          "type": "integer",
          "description": "Total number of chunks deleted"
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

Request (confirmed):
```json
{
  "confirm": true
}
```

Response (success):
```json
{
  "success": true,
  "data": {
    "deleted_count": 1523
  },
  "error": null
}
```

Request (not confirmed):
```json
{
  "confirm": false
}
```

Response (safety gate):
```json
{
  "success": false,
  "data": null,
  "error": "Safety check: Set confirm=True to delete all documents"
}
```

### Safety Considerations

- **Always returns error if `confirm` is not explicitly `true`**
- This is an irreversible operation - all indexed documents will be deleted
- Consider using `get_index_stats` first to review what will be deleted
- Back up the ChromaDB directory before clearing if data recovery may be needed

---

## Tool: get_index_stats

Retrieve statistics about the current collection, including total chunks and indexed files.

### Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

### Parameters

This tool takes no parameters.

### Response Schema

```json
{
  "type": "object",
  "properties": {
    "success": {
      "type": "boolean",
      "description": "Whether stats retrieval succeeded"
    },
    "data": {
      "type": "object",
      "properties": {
        "total_chunks": {
          "type": "integer",
          "description": "Total number of chunks in collection"
        },
        "unique_files": {
          "type": "integer",
          "description": "Number of unique source files"
        },
        "source_files": {
          "type": "array",
          "items": {"type": "string"},
          "description": "List of indexed source file paths"
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
{}
```

Response (success):
```json
{
  "success": true,
  "data": {
    "total_chunks": 487,
    "unique_files": 23,
    "source_files": [
      "/Users/harrison/docs/README.md",
      "/Users/harrison/docs/API.md",
      "/Users/harrison/docs/ARCHITECTURE.md"
    ]
  },
  "error": null
}
```

Response (empty collection):
```json
{
  "success": true,
  "data": {
    "total_chunks": 0,
    "unique_files": 0,
    "source_files": []
  },
  "error": null
}
```

### Use Cases

- **Before clearing index**: Review what documents are indexed
- **Monitoring**: Track collection growth over time
- **Debugging**: Verify files were indexed correctly
- **Maintenance**: Identify which files need re-indexing

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
| `EMBEDDING_FAILED` | Embedding generation error | Check MLX (Apple Silicon) or Ollama is configured correctly |
| `STORAGE_ERROR` | ChromaDB operation failed | Check disk space and permissions |
| `INVALID_INPUT` | Invalid parameter values | Check parameter types and ranges |
| `QUERY_FAILED` | Search operation failed | Check database integrity |
| `BUDGET_EXCEEDED` | Token budget too small | Increase max_tokens |
| `DELETE_FAILED` | Delete operation failed | Check database state |
| `CONFIRMATION_REQUIRED` | clear_index called without confirm=True | Set confirm=True to proceed |

---

## Rate Limits and Performance

### Embedding Rate Limits

Local embeddings (MLX or Ollama) have no rate limits, but performance depends on hardware:

**MLX (Apple Silicon)**:
- M1/M2/M3: ~100-300 chunks/second (GPU accelerated via Metal)

**Ollama**:
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

1. **Use `get_index_stats`** - monitor collection size and indexed files
2. **Use `delete_file`** - remove outdated or updated documents before re-indexing
3. **Use `delete_chunks`** - surgically remove specific chunks when needed
4. **Use `clear_index`** - reset collection when starting fresh (requires `confirm=true`)
5. **Monitor query performance** - large databases may need optimization
6. **Update embeddings** - if embedding model changes, re-index
7. **Backup ChromaDB** - copy database directory before using `clear_index`
